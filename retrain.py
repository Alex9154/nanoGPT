"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 以下为默认配置值，用于在OpenWebText数据集上训练GPT-2（124M参数）模型
# 模型输出文件夹
out_dir = 'out'

# 评估时间间隔
eval_interval = 2000
# 日志间隔
log_interval = 1
# 设置评估迭代次数为200次
eval_iters = 200
# 这个变量用于控制脚本的行为：如果 eval_only 为 True，则脚本在第一次评估后立即退出；如果为 False，则脚本将继续执行后续的操作。
eval_only = False
# 如果这个变量为True，则在每次评估后都会保存一个检查点。
always_save_checkpoint = True
# 训练的形式：从头训练、恢复之前训练、还是使用预训练模型进行训练，'scratch' or 'resume' or 'gpt2*'
init_from = 'scratch'

# Weights & Biases
wandb_log = False
# wandb项目名称设置为owt(OpenWebText)
wandb_project = 'owt'
# 为当前运行设置一个名称，建议为'run' + str(time.time())这种形式
wandb_run_name = 'gpt2'

# 数据相关配置
dataset = 'openwebtext'
# 梯度累积步数：用于计算更大的批次大小
gradient_accumulation_steps = 5 * 8
# 批次大小，如果梯度累积步数 > 1，那么batch_size变量为最小批次大小
batch_size = 12
# 输入限制长度
block_size = 1024

# 模型配置参数
# 模型层数
n_layer = 12
# 多头自注意力机制的头数
n_head = 12
# 嵌入维数
n_embd = 768
# dropout的值，预训练最好设置为0，微调的话建议设置为0.1以上
dropout = 0.0
# 设置是否在归一化层和线性层设置偏执bias
bias = False

# adamw优化器配置参数
# 设置最大学习率
learning_rate = 6e-4
# 设置最大的训练循环次数
max_iters = 600000
# 设置权重衰减参数值
weight_decay = 1e-1
# 设置adamw的一阶矩估计和二阶矩估计的指数衰减率
beta1 = 0.9
beta2 = 0.95
# grad_clip参数用于梯度裁剪，这是一种防止梯度爆炸的技术。
# grad_clip = 1.0表示梯度的范数会被限制在 1.0 以内。如果某个梯度的范数大于 1.0，它会被按比例缩放，使得其范数等于 1.0。
grad_clip = 1.0

# 学习率衰减设置
# 判断是否进行学习率衰减
decay_lr = True
# 学习率预热，在这训练前2000步内，学习率会从一个较小的值逐渐增加到设定的初始学习率，防止梯度爆炸
warmup_iters = 2000
# lr_decay_iters 设置为与 max_iters 近似相等，意味着学习率的衰减过程会贯穿整个训练过程
# 确保学习率在整个训练过程中逐渐减小，而不是在某个特定的时间点突然下降，从而避免训练过程中的不稳定性和梯度消失问题
lr_decay_iters = 600000
# 建议设置为最大学习率的10分之一
min_lr = 6e-5

# 分布式数据并行计算参数
# 指定分布式训练的通信后端为nccl，可选范围'nccl', 'gloo', etc.
backend = 'nccl'
# 设置设备为cpu或者gpu，可选范围'cpu', 'cuda', 'cuda:0', 'cuda:1' etc.
device = 'cpu'
# 设置变量类型，可选范围'float32', 'bfloat16', or 'float16'
# 如果选择了float16，PyTorch会自动启用GradScaler
# 其是动态地放大损失函数的梯度，以防止梯度下溢(梯度为0，模型将停止训练)。在前向传播时，损失值会被放大；在反向传播时，梯度也会相应地放大。
# 这样可以确保梯度值足够大，不会被低精度计算所截断
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# compile 变量设置为 True，表示使用 PyTorch 2.0 编译模型以提高性能
compile = True

# -----------------------------------------------------------------------------
# 读取配置参数的键
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# 执行configurator.py文件：从配置文件或者命令行里面覆盖变量的值
exec(open('configurator.py').read())
# 根据 config_keys 列表，从全局变量中提取对应的值，生成配置字典 config，方便后续日志记录等操作
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# 各种初始化、派生属性、I/O设置
# 判断是否为分布式数据并行运算
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    # 初始化进程组
    init_process_group(backend=backend)
    # 获取（所有设备所有卡上的）全局排名
    ddp_rank = int(os.environ['RANK'])
    # 获取本地（当前设备显卡上的）排名
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    # 获取世界大小（进程总数）
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    # 设置设备为本地排名对应的CUDA设备
    device = f'cuda:{ddp_local_rank}'
    # 设置当前默认CUDA设备
    torch.cuda.set_device(device)
    # 排名为0的进程作为主进程，负责记录日志、检查点等任务
    master_process = ddp_rank == 0
    # 每个进程获得不同的随机种子，保证结果可复现，且避免所有进程生成重复随机数序列
    seed_offset = ddp_rank
    # 由于将有世界大小数量的进程同时进行训练，我们可以按比例减少每个进程所需的梯度累积迭代次数
    # 断言梯度累积步数可以被世界大小整除
    assert gradient_accumulation_steps % ddp_world_size == 0
    # 按比例减少梯度累积步数
    gradient_accumulation_steps //= ddp_world_size
else:
    # 如果不是数据分布式训练，训练将在一个GPU和进程中进行
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
# 打印每次iter训练中token的总数
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")


# 如果是主进程，创建输出目录（如果目录已存在则不报错）
if master_process:
    os.makedirs(out_dir, exist_ok=True)
# 设置PyTorch的随机种子，每个进程的种子不同
torch.manual_seed(1337 + seed_offset)
# 允许在矩阵乘法中使用TensorFloat-32 (TF32)
torch.backends.cuda.matmul.allow_tf32 = True
# 允许在cuDNN中使用TensorFloat-32 (TF32)
torch.backends.cudnn.allow_tf32 = True
# 检测设备类型，用于后续的torch.autocast
# torch.autocast 是 PyTorch 中的一个上下文管理器，用于在训练过程中自动启用混合精度训练
device_type = 'cuda' if 'cuda' in device else 'cpu'
# 注意：float16数据类型将自动使用GradScaler
# GradScaler一般与torch.autocast一起使用，以解决低精度计算中常见的梯度下溢问题
# 使用 torch.autocast 上下文管理器，前向传播中的计算会自动选择合适的精度。损失值会被 GradScaler 放大。
# 反向传播时，GradScaler会在调用loss.backward()之前，将损失值乘以一个缩放因子放大，然后在更新权重时再将梯度除以该缩放因子。
# 根据配置选择数据类型
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# 设置自动混合精度训练的上下文
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 简易的数据加载器
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # 每个批次重新创建 np.memmap 以避免内存泄漏，根据
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        # 加载训练数据
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        # 加载验证数据
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # 生成随机起始索引
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 从数据中提取输入序列
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # 从数据中提取目标序列（输入序列向后移动一个位置）
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # 如果使用GPU，固定数组x和y，以便异步传输到GPU
        # 在手动管理数据传输时，可以使用 pin_memory 方法将数据固定在页锁定内存中
        # 然后使用 to(device, non_blocking=True) 将数据异步传输到 GPU
        # pin_memory意味着数据传输可以在后台进行，而不会阻塞主线程
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        # 如果使用 CPU，直接将数据移动到设备
        x, y = x.to(device), y.to(device)
    return x, y

# 初始化这些变量，如果从检查点恢复（init_from='resume'），可以覆盖这些值
iter_num = 0  # 当前的迭代次数
best_val_loss = 1e9  # 最佳验证损失，初始值设为一个很大的数

# 尝试从数据集中推导词汇表大小
meta_path = os.path.join(data_dir, 'meta.pkl')  # 元数据文件的路径
meta_vocab_size = None  # 初始化词汇表大小为 None
# 检查元数据文件是否存在
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)  # 从元数据文件中加载元数据
    meta_vocab_size = meta['vocab_size']  # 从元数据中获取词汇表大小
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")  # 打印找到的词汇表大小

# 模型初始化
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)  # 从命令行参数开始初始化模型参数
if init_from == 'scratch':
    # 从头开始初始化一个新模型
    print("Initializing a new model from scratch")
    # 确定从头开始训练时使用的词汇表大小
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # 从检查点恢复训练
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # 强制这些配置属性必须与检查点中的值相同，否则无法恢复训练
    # 其他属性（如 dropout）可以保留命令行中的值
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # 创建模型
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 修复状态字典的键名
    # 有时检查点会带有前缀，具体原因需要进一步调试
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
# 将模型移动到指定设备
model.to(device)

# 初始化一个 GradScaler。如果 enabled=False，scaler 将是一个无操作对象（no-op）
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 实例化optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
# 释放检查点占用的内存
checkpoint = None

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
