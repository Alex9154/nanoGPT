import inspect
import math
import torch
from torch.nn import functional as F
import torch.nn as nn

class LayerNorm(nn.Module):
    """LayerNorm but with a optional bias. Pytorch doesn't support simply bias=False"""
    def __init__(self, ndim, bias):
        super().__init__()
        # 对归一化后的张量进行缩放
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forword(self, input):
        return F.layer_norm(input,self.weight.shape,self.weight,self.bias,eps=1e-5)
    
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here.
    """
    def __init__(self,config):
        super().__init__()
        # 断言输入嵌入维度是否能被head数整除
        assert config.n_embd % config.n_head == 0
        # 一次性计算QKV矩阵
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
        # 输出映射
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # 将config的属性值保存到类中
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.dropout = config.dropout
        # 判断pytorch版本是否支持flashattention
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        # 如果不支持flashattention则需要将输入进行掩码
        if not self.flash:
            print("Warning: using slow attention. Flash Attention requires Pytorch >= 2.0")
            # 生成掩码下三角矩阵bias
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size,config.block_size))


    def forward(self,x):
        # 获取输入张量维度，B -> batch size, T -> Text length, C -> embedding size
        # 这里C其实和self.n_embd变量是一样的
        B, T, C = x.size()
        # 获取q, k, v矩阵
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # (B, T, C)
        # 分头
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        # 计算注意力得分，(B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # 这里的attn_mask为None表示不自定义一个掩码张量，is_causal设置为True函数自动生成掩码张量
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # 手动计算注意力
            att = (q @ k.transpose(-2,-1)) * (1.0 / torch.sqrt(k.size(-1))) # (B, nh, T, T)
            # 对得分进行掩码，这里的bias大小为(1,1,block_size,block_size)，block_size（输入最大长度）
            att = att.masked_fill(self.bias[:,:,:T,:T]==0, float("-inf")) # (B, nh, T, T)
            # 使用softmax进行(按行)缩放，计算注意力得分
            att = F.softmax(att, dim=-1)
            # 对注意力得分进行一次dropout
            att = self.attn_dropout(att)
            # 计算最终输出
            y = att @ v # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        # 将最终结果合并，contiguous()的目：确保张量在内存连续，transpose可能导致不连续，方便后续view函数执行
        y = y.transpose(1,2).contiguous().view(B, T, C)
        # 输出需要经过一次映射和dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd*4)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias = config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTConfig:
    block_size: int = 1024 # 最大支持1024长度的句子输入
    vocab_size: int = 50304 # 词表大小
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # 当bias为True时，和GPT-2一样；当bias为False时，运算速度会更快一些

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias)
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        # 输入嵌入层和输出投影层权重共享
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化所有参数
        self.apply(self._init_weight)
        # 根据GPT-2论文中的建议，将投影层的权重，使用不同初始化方法
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # 打印参数个数
        # print("number of parameters: %2fM" % (self.get_num_))

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self,idx,targets=None):
        # 获取idx所在的位置，确保后面的pos和其在同一个GPU或CPU中
        device = idx.device
        # 获取idx的形状
        b, t = idx.size()
        # 断言t的长度小于block_size
        assert t < self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # 生成位置索引
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 进行前向推导
        tok_embd = self.transformer.wte(idx) # token embedding of shape (b, t, n_embd)
        pos_embd = self.transformer.wpe(pos) # position embedding of shape (b, t, n_embd)
        x = self.transformer.drop(tok_embd + pos_embd) # dropout一些信息，防止过拟合
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # 计算loss和最终输出的logits
        if targets is not None:
            logits = self.lm_head(x)
            # logits.size(-1)获取vocab size，ignore_index是忽略掉标签中的无意义值，比如填充标记等；
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 计算最后一个词的vocab logits
            logits = self.lm_head(x[:,[-1],:])
            loss = None
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 获取所有参数，并存放为字典
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉不需要学习的参数
        param_dict = {pn: p for pn,p in param_dict.items() if p.requires_grad}
        # 创建优化器组，大于二维的参数使用正则化权重衰减，小于二维的参数不使用正则化方法
        # 即所有矩阵乘法有关的参数将使用正则化权重衰减，所有偏置项将不使用
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params':decay_params, 'weight_decay':weight_decay},
            {'params':nodecay_params, 'weight_decay':0.0}
        ]
        # 打印权重衰减的参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} paramters.")
        print(f"num decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} paramters.")
        # 创建AdamW优化器，如果可以使用fused版本则使用
        # 判断是否支持fused融合操作模式，融合操作即将多个操作合并成一个操作
        fused_aviliable = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        used_fused = fused_aviliable and device_type == 'cuda'
        extra_args = dict(fused=True) if used_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, learning_rate, betas, **extra_args)
        print(f"Using fused AdamW: {used_fused}")

        return optimizer
    
    @torch.no_grad
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # 首先判断idx有没有超过最大长度限制
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 模型前推
            logits, _ = self(idx_cond)
            # 取模型最后一个词的logits并且进行temperature缩放
            logits = logits[:, -1, :] / temperature
            # 根据top_k参数切除logits
            if top_k is not None:
                # torch.topk函数返回的是每个vocab size中最大k个logits的值v，和最大logits对应的index
                v, _ = torch.topk(logits, min(top_k, idx.size(-1)))
                # v[:, [-1]]代表的是k个logits中最小的那个logits的值
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 将logits通过softmax函数转换为逻辑值
            probs = F.softmax(logits, dim=-1)
            # 从以上的概率值中进行采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将预测的词idx_next拼接到idx
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
