import math
import torch
from torch import nn
from config import ModelArgs
from DeepSeekMoE import Linear, ColumnParallelLinear, RowParallelLinear
from RMSNorm import RMSNorm
from typing import Tuple, Optional
from kernel import act_quant, weight_dequant, fp8_gemm


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """ 预计算旋转位置嵌入的基于频率的复指数值 """
    dim = args.qk_rope_head_dim
    seq_len = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        计算旋转位置嵌入中给定旋转数的校正维度
        Args:
            num_rotations (float): 要计算校正的旋转数
            dim (int): 嵌入空间的维度
            base (float): 指数计算的基值
            max_seq_len (int): 最大序列长度
        Returns:
            float:基于输入参数的校正维度
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
       计算旋转位置嵌入的校正维度范围
        Args:
            low_rot (float): 旋转次数的下限
            high_rot (float): 旋转次数的上限
            dim (int): 嵌入空间的维度
            base (float): 指数计算的基值
            max_seq_len (int): 最大序列长度
        Returns:
            Tuple[int, int]: 校正维度的范围(低、高)，钳制到有效索引
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        计算用于平滑最小和最大范围之间的值的线性斜坡函数。
        Args:
            min (float): 斜坡函数的最小值
            max (float): 斜坡函数的最大值
            dim (int): 斜坡张量的维度
        Returns:
            torch.Tensor: 张量:形状的张量(dim，)其值在0和1之间线性插值，钳制在范围[0，1]之内。
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func
    # 计算初始频率张量freqs，torch.arange()生成从0到dim的偶数索引，以便分为偶数和奇数维度
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # 如果当前序列长度超过原始序列长度，则进行频率校正
    if seq_len > args.original_seq_len:
        # 计算校正范围
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        # 生成平滑因子
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        # 调整频率
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    # 生成一个从0到seq_len-1的时间轴张量t
    t = torch.arange(seq_len)
    # 使用torch.outer计算时间轴t和频率freqs的外积，生成一个二维张量，形状为 (seq_len, dim)。
    freqs = torch.outer(t, freqs)
    # 使用torch.polar将频率张量转换为复数形式
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """ x:输入张量，freqs_cis:包含旋转频率信息的张量 """
    dtype = x.dtype  # 保存输入的数据类型
    # torch.view_as_complex()：用于将一个实数张量转换为复数张量，其输入张量的最后一个维度大小必须为2，表示复数的实部和虚部
    # x.float().view(*x.shape[:-1], -1, 2)：将x转换为浮点类型，并重塑形状，最后一个维度的大小为2，中间维度自动调整
    # *x.shape[:-1]表示将x除最后一个维度的前序维度解包，即展开为多个参数
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    # 调整freqs_cis的维度，以便与x的形状对齐
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    # torch.view_as_real()：将复数张量转换为实数
    # x * freqs_cis：复数乘法，实现旋转位置编码的核心操作
    # flatten(3)：从张量的第3个维度全部展开为一个维度
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)  # 转换回原始数据类型并返回


class MLA(nn.Module):
    """
    Multi-Head Latent Attention
    Attributes:
        dim (int): 输入特征的维度
        n_heads (int): 多头自注意力的总头数
        n_local_heads (int): 分布式系统中每个进程上的头数
        q_lora_rank (int): 查询的低秩投影秩
        kv_lora_rank (int): 键和值的低秩投影秩
        qk_nope_head_dim (int): 无位置编码的q/k头维度
        qk_rope_head_dim (int): 旋转位置编码的q/k头维度
        qk_head_dim (int): q/k的头维度
        v_head_dim (int): v的头维度
        softmax_scale (float): 注意力计算中softmax函数的缩放因子
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // args.world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.attn_impl = args.attn_impl
        self.block_size = args.block_size

        """ 低秩适应（LoRA）的实现 """
        # 如果q的低秩投影秩为0，则应用
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        # 如果q的低秩投影秩大于0，则使用低秩分解
        else:
            # 将输入投影到低秩空间
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            # 对投影到低秩空间的结果进行归一化
            self.q_norm = RMSNorm(self.q_lora_rank)
            # 将低秩空间归一化后的结果投影到最终的查询/键维度，总头数（n_heads）* 每个头上的键值维度（qk_head_dim）=总维度
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        # 将键值投影到低秩空间，并结合旋转位置嵌入
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        # 对旋转位置编码的键值投影结果进行归一化
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        # 将归一化后的结果进一步投影到键值的总维度
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))

        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)

        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        """ 缓存机制：用于存储键和值的中间结果，以便在后续计算中复用 """
        # 如果attn_impl="naive"，则分别缓存键和值
        if self.attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads,
                                                        self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads,
                                                        self.v_head_dim), persistent=False)
        # 否则，缓存低秩键值和键旋转位置嵌入的结果
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank),
                                 persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim),
                                 persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        # 输入x：[batch_size、seq_len、dim]，start_pos：序列的起始位置，用于缓存机制
        # freqs_cis：预计算的旋转位置嵌入的复数指数值，mask：可选的掩码张量，用于排除某些位置的注意力计算。
        batch_size, seq_len, _ = x.size()
        end_pos = start_pos + seq_len

        """ 查询q的计算 """
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        # 将查询q的计算结果重塑为[batch_size, seq_len, n_local_heads, qk_head_dim]
        q = q.view(batch_size, seq_len, self.n_local_heads, self.qk_head_dim)

        """ 查询q的拆分与旋转位置嵌入 """
        # 将q按照指定的大小，在最后一个维度上分割为无旋转位置编码和有旋转位置编码两部分
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_nope_head_dim], dim=-1)
        # 对旋转位置编码部分应用旋转位置嵌入
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        """ 键值的计算与键的旋转位置嵌入 """
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        """ 注意力计算 """
        if self.attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(batch_size, seq_len, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:batch_size, start_pos:end_pos] = k
            self.v_cache[:batch_size, start_pos:end_pos] = v
            # torch.einsum: 根据给定的字符串对输入张量执行指定的操作
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:batch_size, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight,
                                                                                      self.wkv_b.scale, self.block_size)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:batch_size, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:batch_size, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:batch_size, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:batch_size, :end_pos])) * self.softmax_scale

        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if self.attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:batch_size, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:batch_size, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x
