# 导入math模块，用于数学运算
import math

# 导入dataclass装饰器，用于创建数据类
from dataclasses import dataclass

# 导入Any类型，用于类型注解
from typing import Any

# 导入torch，用于PyTorch操作
import torch

# 导入torch.nn，用于神经网络模块
import torch.nn as nn

# 导入torch.nn.functional，用于函数式神经网络操作
import torch.nn.functional as F

# 从torch.nn导入CrossEntropyLoss，用于交叉熵损失计算
from torch.nn import CrossEntropyLoss

# 从transformers导入AutoConfig，用于自动配置加载
from transformers import AutoConfig

# 从transformers.activations导入ACT2FN，用于激活函数映射
from transformers.activations import ACT2FN

# 从transformers.cache_utils导入缓存类
from transformers.cache_utils import (
    Cache,  # 缓存基类
    DynamicCache,  # 动态缓存
    StaticCache,  # 静态缓存
)

# 从transformers.generation导入GenerationMixin，用于生成混合
from transformers.generation import GenerationMixin

# 从transformers.modeling_attn_mask_utils导入AttentionMaskConverter，用于注意力掩码转换
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

# 从transformers.modeling_outputs导入模型输出类
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput

# 从transformers.modeling_rope_utils导入ROPE_INIT_FUNCTIONS，用于旋转位置编码初始化
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# 从transformers.modeling_utils导入PreTrainedModel，用于预训练模型基类
from transformers.modeling_utils import PreTrainedModel

# 从transformers.utils导入各种实用函数
from transformers.utils import (
    add_start_docstrings,  # 添加开始文档字符串
    add_start_docstrings_to_model_forward,  # 添加模型前向传播的文档字符串
    is_flash_attn_2_available,  # 检查Flash Attention 2是否可用
    # is_flash_attn_greater_or_equal_2_10,  # 检查Flash Attention版本
    is_flash_attn_greater_or_equal,  # 修改这一行
    is_torchdynamo_compiling,  # 检查TorchDynamo是否正在编译
    logging,  # 日志记录
    replace_return_docstrings,  # 替换返回文档字符串
)

# 从当前包的configuration_qwen2_5_vl模块导入配置类
from .configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig


# TODO(Steven): SlidingWindowCache was removed in transformers v5. Define a placeholder so isinstance checks
# always return False (which is the correct behavior when no sliding window cache is in use).
# 滑动窗口缓存占位符类，用于兼容性。在transformers v5中SlidingWindowCache被移除了，所以定义一个占位符，使得isinstance检查始终返回False。
class _SlidingWindowCachePlaceholder:
    pass  # 占位符，不包含任何功能


# 将SlidingWindowCache定义为占位符类，以便代码中已有的isinstance检查不会因类不存在而报错。
SlidingWindowCache = _SlidingWindowCachePlaceholder

# 检查Flash Attention 2是否可用
if is_flash_attn_2_available():
    # 如果可用，从flash_attn导入Flash Attention相关的函数和层
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb
else:
    # 如果不可用，将相关的函数设置为None，这样当代码尝试调用它们时会抛出AttributeError，从而回退到其他实现。
    flash_attn_varlen_func = None
    apply_rotary_emb = None
    flash_attn_func = None


# 这里有一个重复的if-else块，似乎是冗余的。上面的代码已经处理了导入，这里再次将flash_attn_varlen_func设置为None是多余的。
if is_flash_attn_2_available():
    pass  # 如果可用，什么也不做
else:
    flash_attn_varlen_func = None  # 这行代码是多余的，因为上面已经设置了


# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 用于文档字符串中引用配置类的常量
_CONFIG_FOR_DOC = "Qwen2_5_VLConfig"


class Qwen2_5_VLMLP(nn.Module):
    """Qwen2.5-VL的多层感知机模块，采用了SwiGLU结构。"""
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size  # 隐藏层维度
        self.intermediate_size = config.intermediate_size  # 中间层维度
        # 门控投影层，将隐藏状态映射到中间维度
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        # 上投影层，同样将隐藏状态映射到中间维度
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        # 下投影层，将中间维度映射回隐藏维度
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        # 激活函数，从配置中获取，如'silu', 'gelu'等
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        # SwiGLU: down_proj( act_fn(gate_proj(x)) * up_proj(x) )
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5_VisionPatchEmbed(nn.Module):
    """将图像/视频分割成patch并进行嵌入的模块。使用3D卷积来处理时间维度和空间维度。"""
    def __init__(
        self,
        patch_size: int = 14,  # 空间patch大小（高和宽）
        temporal_patch_size: int = 2,  # 时间patch大小，即每次卷积处理多少帧
        in_channels: int = 3,  # 输入通道数，例如RGB图像的3通道
        embed_dim: int = 1152,  # 输出嵌入向量的维度
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # 卷积核大小：[时间维度，高度，宽度]
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        # 使用Conv3d一次性将时空patch映射到嵌入向量
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,  # 步长等于卷积核大小，实现无重叠的patch划分
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            hidden_states: 输入张量，形状为 (batch_size * temporal, channels, height, width)，
                           即原始图像或视频帧的堆叠。

        Returns:
            输出张量，形状为 (num_patches, embed_dim)，其中num_patches是所有样本的patch总数。
        """
        target_dtype = self.proj.weight.dtype  # 获取投影层权重的数据类型
        # 将输入重新reshape为 (batch, channels, temporal, height, width) 的形状，以便Conv3d处理
        # 输入hidden_states的形状是 (total_frames, C, H, W)
        # 这里将其view为 (total_frames // temporal_patch_size, C, temporal_patch_size, H, W)？不，是下面的形式：
        # 假设输入形状为 (N, C, H, W)，其中N = batch_size * temporal_patch_size * ?
        # 更准确地说，在vision encoder的forward中，hidden_states是原始像素值，形状为 (total_patches, C, H, W)?
        # 查看调用处，此处的hidden_states是原始像素值，形状为 (total_pixels, C, H, W)，其中total_pixels是所有图像/视频的像素点总数。
        # 这里将其reshape为 (num_samples, C, temporal_patch_size, patch_size, patch_size) 形式。
        hidden_states = hidden_states.view(
            -1,  # 自动推断batch维度，这里的batch实际上是时空patch的总数
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        # 进行3D卷积，输出形状为 (num_patches, embed_dim, 1, 1, 1) 或类似，然后view成 (num_patches, embed_dim)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    """为视觉特征计算RoPE (Rotary Position Embedding) 的基础频率。"""
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        # 计算逆频率，用于后续生成位置编码
        # inv_freq = 1 / (theta^(2i/d)) for i in [0, d/2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        # 注册为缓冲区，不参与训练，且不会在保存模型状态字典时持久化（persistent=False）
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        """生成给定长度的位置编码。

        Args:
            seqlen: 序列长度（例如，最大网格尺寸）。

        Returns:
            一个张量，形状为 (seqlen, dim//2)，包含每个位置的角度值。
        """
        # 生成位置索引序列 [0, 1, 2, ..., seqlen-1]
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # 计算外积，得到形状为 (seqlen, dim//2) 的矩阵，其中每行是位置的角度值
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2RMSNorm(nn.Module):
    """Qwen2使用的RMSNorm (Root Mean Square Layer Normalization)。"""
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        # 可学习的缩放因子
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps  # 防止除零的微小常数

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype  # 保存输入的数据类型
        # 转换为float32进行计算，以提高数值稳定性
        hidden_states = hidden_states.to(torch.float32)
        # 计算每个样本在最后一个维度上的方差
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # RMSNorm: x / sqrt(mean(x^2) + eps)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 缩放并转换回原始数据类型
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        """返回模块的额外信息，用于打印。"""
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2_5_VLPatchMerger(nn.Module):
    """将视觉patch的特征合并，使其维度与语言模型的隐藏维度对齐。"""
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        # 合并后的特征维度。spatial_merge_size用于在空间上合并相邻的patch。
        # 例如，如果spatial_merge_size=2，则将2x2的patch合并为一个。
        # 所以合并后的特征维度是 context_dim * (spatial_merge_size^2)
        self.hidden_size = context_dim * (spatial_merge_size**2)
        # 对输入特征进行RMSNorm
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        # 一个简单的两层MLP，将合并后的特征映射到目标维度 dim
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先进行归一化，然后reshape以便将空间相邻的patch特征拼接起来
        # 输入x的形状是 (num_patches, context_dim)
        # 在调用此函数前，patch的顺序已经被重新排列，使得需要合并的patch在序列中是连续的。
        # 这里通过view将其形状变为 (num_merged_patches, spatial_merge_size^2, context_dim)
        # 然后通过mlp处理。mlp的第一层输入是合并后的特征。
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """为Flash Attention应用RoPE。由于flash_attn的apply_rotary_emb函数对输入形状有特定要求，此函数进行适配。"""
    # 输入cos, sin的形状为 (seq_len, head_dim//2)，而flash_attn的apply_rotary_emb期望cos, sin的形状为 (seq_len, head_dim)
    # 因此这里将cos, sin沿最后一个维度重复一次，使其变为 (seq_len, head_dim)
    # 但更常见的做法是，flash_attn的apply_rotary_emb内部会处理，这里只取一半？注释可能有误。
    # 实际代码中，cos = cos.chunk(2, dim=-1)[0] 意味着它只取cos的前半部分，然后contiguous()。
    # 这通常是因为在flash_attn的实现中，cos和sin需要是head_dim一半的长度。
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    # 将q和k转换为float32，应用旋转，然后转换回原始类型
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed


class Qwen2_5_VLVisionFlashAttention2(nn.Module):
    """视觉编码器中，使用Flash Attention 2的注意力层。"""
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads  # 注意力头数
        # QKV投影层，输出维度为 dim * 3
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        # 输出投影层
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            hidden_states: 输入张量，形状为 (total_seq_len, dim)
            cu_seqlens: 累积序列长度，用于标记不同样本的边界，形状为 (batch_size + 1)
            max_seqlen: 批次中最大序列长度
            rotary_pos_emb: 旧版API，旋转位置编码的张量
            position_embeddings: 新版API，包含cos和sin的元组

        Returns:
            输出张量，形状为 (total_seq_len, dim)
        """
        seq_length = hidden_states.shape[0]
        # 计算Q, K, V。qkv投影后reshape为 (seq_len, 3, num_heads, head_dim)，然后permute和unbind得到三个独立的张量。
        # 最终q, k, v的形状都是 (num_heads, seq_len, head_dim)
        q, k, v = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        # 处理位置编码（兼容新旧API）
        if position_embeddings is None:
            # 使用旧的rotary_pos_emb
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
        # 应用RoPE
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)  # 去掉batch维度
        k = k.squeeze(0)

        # 如果没有提供max_seqlen，则从cu_seqlens计算
        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        # 调用flash_attn的变长序列函数
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        # 输出投影
        attn_output = self.proj(attn_output)
        return attn_output


def rotate_half(x):
    """将输入张量的一半维度进行旋转，用于RoPE计算。"""
    # x1取前一半，x2取后一半
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # 返回 [-x2, x1]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """为视觉注意力应用RoPE的标准实现。"""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()  # 转换为float32提高精度
    # 扩展cos和sin的维度，使其能与q, k进行广播
    # cos, sin的形状为 (seq_len, head_dim//2)，unsqueeze(-2)后变为 (seq_len, 1, head_dim//2)
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    # 应用旋转公式： q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen2_5_VLVisionAttention(nn.Module):
    """视觉编码器的标准（eager模式）注意力层。使用手动实现的注意力机制。"""
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个头的维度
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        # 计算Q, K, V
        q, k, v = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        # 处理位置编码
        if position_embeddings is None:
            logger.warning_once(...)  # 与FlashAttention2中相同的警告
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
        # 应用RoPE
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # 创建注意力掩码，只允许同一图像/视频内的token相互注意
        attention_mask = torch.full(
            [1, seq_length, seq_length],
            torch.finfo(q.dtype).min,  # 使用很小的值填充，表示被遮蔽
            device=q.device,
            dtype=q.dtype,
        )
        # 根据cu_seqlens将同一序列内的注意力掩码设为0（允许注意）
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 0

        # 转换维度以进行矩阵乘法: (seq_len, num_heads, head_dim) -> (num_heads, seq_len, head_dim)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        # 计算注意力分数: Q @ K^T / sqrt(d_k)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        # 应用掩码
        attn_weights = attn_weights + attention_mask
        # Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        # 加权求和: attn @ V
        attn_output = torch.matmul(attn_weights, v)
        # 恢复维度: (num_heads, seq_len, head_dim) -> (seq_len, num_heads*head_dim)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        # 输出投影
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionSdpaAttention(nn.Module):
    """视觉编码器中使用PyTorch的scaled_dot_product_attention的注意力层。"""
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        # 计算Q, K, V
        q, k, v = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        # 处理位置编码
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
        # 应用RoPE
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # 创建注意力掩码，但这里是布尔类型，True表示需要被注意
        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = True

        # 转换维度
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        # 调用PyTorch的scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        # 恢复维度
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


# 一个字典，用于根据配置选择视觉注意力层的实现
QWEN2_5_VL_VISION_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLVisionAttention,
    "flash_attention_2": Qwen2_5_VLVisionFlashAttention2,
    "sdpa": Qwen2_5_VLVisionSdpaAttention,
}


class Qwen2_5_VLVisionBlock(nn.Module):
    """视觉编码器的一个标准块，包含一个注意力层和一个MLP层，遵循Pre-LN结构。"""
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        # 第一个归一化层（用于注意力）
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        # 第二个归一化层（用于MLP）
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        # 根据配置选择注意力层
        self.attn = QWEN2_5_VL_VISION_ATTENTION_CLASSES[attn_implementation](
            config.hidden_size, num_heads=config.num_heads
        )
        # MLP层
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Pre-LN: 输入先通过norm1，然后经过注意力，最后残差连接
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        # Pre-LN: 输入先通过norm2，然后经过MLP，最后残差连接
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# 模型文档字符串
Qwen2_5_VL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2_5_VLConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2_5_VL Model outputting raw hidden-states without any specific head on top.",
    Qwen2_5_VL_START_DOCSTRING,
)
class Qwen2_5_VLPreTrainedModel(PreTrainedModel):
    """Qwen2.5-VL模型的基类，继承自PreTrainedModel，处理权重的初始化和加载。"""
    config_class = Qwen2_5_VLConfig  # 指定配置类
    base_model_prefix = "model"  # 模型基础前缀
    supports_gradient_checkpointing = True  # 支持梯度检查点
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]  # 不应被分割的模块名列表
    _skip_keys_device_placement = "past_key_values"  # 跳过设备放置的键
    _supports_flash_attn_2 = True  # 支持Flash Attention 2
    _supports_sdpa = True  # 支持SDPA
    _supports_cache_class = True  # 支持缓存类
    _supports_static_cache = (
        False  # TODO (joao): fix. torch.compile failing probably due to `cache_positions`
    )  # 暂不支持静态缓存

    def _init_weights(self, module):
        """初始化模型权重。"""
        std = self.config.initializer_range  # 标准差
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            # 线性层和Conv3d层使用正态分布初始化
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()  # 偏置初始化为0
        elif isinstance(module, nn.Embedding):
            # Embedding层使用正态分布初始化
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()  # padding_idx对应的权重设为0


class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPreTrainedModel):
    """视觉Transformer模型，继承自预训练基类。"""
    config_class = Qwen2_5_VLVisionConfig  # 使用视觉配置类
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size  # 空间合并大小，用于将相邻patch合并
        self.patch_size = config.patch_size  # patch大小
        self.fullatt_block_indexes = config.fullatt_block_indexes  # 使用全局注意力的块索引
        self.window_size = config.window_size  # 窗口大小，用于窗口注意力
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size  # 空间合并单元大小

        # Patch嵌入层
        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        # 计算每个头的维度，用于RoPE
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        # 构建所有视觉块
        self.blocks = nn.ModuleList(
            [Qwen2_5_VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        # Patch合并器，用于将视觉特征映射到语言模型的维度
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False  # 是否使用梯度检查点

    def rot_pos_emb(self, grid_thw):
        """根据图像/视频的网格尺寸（时间、高、宽）生成旋转位置编码。"""
        pos_ids = []
        # 遍历每个图像/视频的网格尺寸 (t, h, w)
        for t, h, w in grid_thw:
            # 生成高度方向的位置ID，并考虑空间合并
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)  # 重新排列，使得相邻的patch被合并
            hpos_ids = hpos_ids.flatten()

            # 生成宽度方向的位置ID
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            # 将高度和宽度位置ID堆叠，并在时间维度上重复t次
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        # 将所有图像/视频的位置ID拼接起来
        pos_ids = torch.cat(pos_ids, dim=0)
        # 获取最大网格尺寸，用于生成完整的RoPE编码
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        # 根据位置ID索引出对应的RoPE编码
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        """根据窗口大小，重新排列patch索引，使得同一窗口内的patch在序列中是连续的。"""
        window_index: list = []
        cu_window_seqlens: list = [0]  # 累积窗口序列长度
        window_index_id = 0
        # 计算在合并后的网格上，窗口的大小
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            # 合并后的网格尺寸
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            # 生成原始索引，形状为 (t, llm_grid_h, llm_grid_w)
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            # 计算需要填充的大小，以使网格能被窗口大小整除
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size  # 高度方向上的窗口数
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size  # 宽度方向上的窗口数
            # 填充索引，用-100填充
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            # 将填充后的索引重塑为窗口结构
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            # 重新排列维度并重塑，得到形状为 (grid_t, num_windows_h*num_windows_w, window_size, window_size)
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            # 计算每个窗口的有效token数（排除填充的-100）
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            # 展平窗口内的索引，并移除填充的-100
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            # 累加索引ID，并添加到列表中
            window_index.append(index_new + window_index_id)
            # 计算累积的窗口序列长度
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        # 1. Patch Embedding: 将原始像素转换为patch嵌入
        hidden_states = self.patch_embed(hidden_states)

        # 2. 生成旋转位置编码
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # 3. 生成窗口索引，用于后续的窗口注意力重排
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        window_index = window_index.to(hidden_states.device)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)  # 确保连续且唯一

        # 4. 根据窗口索引重新排列patch序列
        seq_len, _ = hidden_states.size()
        # 将hidden_states重塑为 (num_spatial_units, spatial_merge_unit, hidden_dim)
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        # 根据窗口索引重新排列spatial_units的顺序
        hidden_states = hidden_states[window_index, :, :]
        # 恢复形状为 (seq_len, hidden_dim)
        hidden_states = hidden_states.reshape(seq_len, -1)

        # 5. 同样对旋转位置编码进行重排
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        # 生成完整的cos和sin位置编码
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # 6. 计算累积序列长度，用于区分不同的图像/视频
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)  # 在开头补0
        max_seqlen_full = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        max_seqlen_window = (cu_window_seqlens[1:] - cu_window_seqlens[:-1]).max().item()

        # 7. 遍历所有视觉块
        for layer_num, blk in enumerate(self.blocks):
            # 根据当前层是否在fullatt_block_indexes中，决定使用全局注意力还是窗口注意力
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen_full
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_seqlen_window
            # 如果启用梯度检查点且在训练模式，使用梯度检查点函数
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__,
                    hidden_states,
                    cu_seqlens_now,
                    None,
                    position_embeddings,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    max_seqlen=max_seqlen_now,
                    position_embeddings=position_embeddings,
                )

        # 8. 使用PatchMerger将特征维度映射到语言模型的维度
        hidden_states = self.merger(hidden_states)

        # 9. 恢复原始的patch顺序（因为之前根据窗口重排了）
        reverse_indices = torch.argsort(window_index)  # 计算逆排序索引
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


def _compute_default_rope_parameters_qwen2_5_vl(config, device=None):
    """
    compute default rope parameters for Qwen2_5_VL
    """
    base = config.text_config.rope_parameters["rope_theta"]
    dim = config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


class Qwen2_5_VLRotaryEmbedding(nn.Module):
    """语言模型的RoPE模块，支持动态更新。"""
    def __init__(self, config: Qwen2_5_VLConfig, device=None):
        super().__init__()
        # 确定RoPE类型
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        elif hasattr(config, "rope_parameters") and config.rope_parameters is not None:
            self.rope_type = config.rope_parameters.get("rope_type", "default")
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings  # 缓存的最大序列长度
        self.original_max_seq_len = config.max_position_embeddings  # 原始最大序列长度

        self.config = config

        # 根据rope_type选择初始化函数
        if self.rope_type == "default":
            self.rope_init_fn = _compute_default_rope_parameters_qwen2_5_vl
            self.rope_kwargs = {}
        else:
            rope_type_key = "linear" if self.rope_type == "linear" else self.rope_type
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type_key]
            self.rope_kwargs = {}

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """动态更新频率，当序列长度超过缓存长度或从长序列回到短序列时。"""
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # 如果序列长度增长，重新计算频率
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len
        ):  # 如果序列长度缩小，重置为原始频率
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        """前向传播，生成cos和sin位置编码。

        Args:
            x: 输入张量，用于确定数据类型和设备
            position_ids: 位置ID，形状为 (3, batch_size, seq_len) 或类似，用于多模态RoPE
        """
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_5_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        # 将逆频率扩展到 (3, 1, head_dim//2, 1) 的形状，以匹配position_ids的3个维度
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        # 将position_ids转换为float并扩展维度以进行矩阵乘法
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # 计算频率: inv_freq @ position_ids
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # 应用注意力缩放因子
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2MLP(nn.Module):
    """语言模型的MLP层，结构与Qwen2_5_VLMLP相同，但无偏置。"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """为多模态输入（文本、图像、视频）应用RoPE。
    对于文本，三个维度（时间、高、宽）的位置ID相同，因此等效于1D RoPE。
    对于视觉，三个维度分别应用不同的位置ID，因此需要将通道维度分割成三部分，分别应用旋转。
    """
    mrope_section = mrope_section * 2  # 因为cos和sin已经拼接了两次
    # 将cos和sin沿着最后一个维度（通道维）分割成三个部分，然后按循环顺序（i%3）重新拼接起来
    # 这样做的目的是为了使得对于视觉特征，不同的通道部分对应不同的位置编码维度（t, h, w）
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    # 应用旋转公式
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复key/value头，以匹配query的头数。用于Grouped-Query Attention (GQA)。"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 扩展并重塑
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2_5_VLAttention(nn.Module):
    """语言模型的标准注意力层。"""
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # Q, K, V投影层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # RoPE模块
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # necessary, but kept here for BC
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()

        # 计算Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 重塑并转置为 (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # 应用多模态RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # 更新缓存
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # 重复K/V头以匹配Q的头数（GQA）
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 计算注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 应用注意力掩码
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # 修复float16推理中的精度问题
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(
                torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights
            )

        # Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # 重塑并输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2_5_VLFlashAttention2(Qwen2_5_VLAttention):
    """语言模型的Flash Attention 2实现。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # 用于处理不同版本Flash Attention的掩码对齐问题
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # necessary, but kept here for BC
    ):
        bsz, q_len, _ = hidden_states.size()
        # 计算Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # 应用RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # 注意：这里没有重复K/V头，因为Flash Attention的API期望的是未重复的K/V，其内部会处理GQA
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # 处理类型转换
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(...)  # 警告信息
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # 重塑为Flash Attention期望的形状: (batch, seq_len, num_heads, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # 调用flash_attn_func
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_rate,
            softmax_scale=None,
            causal=self.is_causal,
        )

        # 重塑并输出
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2_5_VLSdpaAttention(Qwen2_5_VLAttention):
    """语言模型的SDPA (Scaled Dot Product Attention) 实现。"""
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # necessary, but kept here for BC
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        # 如果需要输出注意力权重，则回退到标准实现
        if output_attentions:
            logger.warning_once(...)
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()
        # 计算Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # 应用RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # 重复K/V头以匹配Q的头数
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 处理注意力掩码
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # 对于CUDA设备，确保张量是连续的
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # 判断是否使用因果掩码
        is_causal = True if causal_mask is None and q_len > 1 else False

        # 调用SDPA
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # 重塑并输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# 语言模型注意力实现的字典
QWEN2_5_VL_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLAttention,
    "flash_attention_2": Qwen2_5_VLFlashAttention2,
    "sdpa": Qwen2_5_VLSdpaAttention,
}


class Qwen2_5_VLDecoderLayer(nn.Module):
    """语言模型的一个解码器层，包含自注意力和MLP。"""
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        # 自注意力层
        self.self_attn = QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        # MLP层
        self.mlp = Qwen2MLP(config)
        # 层归一化
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        # 预层归一化 (Pre-LN) 应用于自注意力
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # 预层归一化 (Pre-LN) 应用于MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@add_start_docstrings(
    "The bare Qwen2_5_VL Model outputting raw hidden-states without any specific head on top.",
    Qwen2_5_VL_START_DOCSTRING,
)
class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    """Qwen2.5-VL的核心语言模型，包含嵌入层、解码器层和归一化层。"""
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 解码器层列表
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        # 最终层归一化
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # RoPE模块
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # 梯度检查点与缓存不兼容
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 初始化动态缓存
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        # 获取输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 设置缓存位置
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # 设置位置ID（硬编码3个维度）
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # 创建因果注意力掩码
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # 为所有解码器层共享的位置嵌入
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 遍历所有解码器层
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 最终层归一化
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        """更新因果注意力掩码，兼容不同的注意力实现。"""
        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2 内部处理掩码
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2_5_VL. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # 对于SDPA，尽量使用is_causal参数
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # 确定目标长度
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # 生成4D因果注意力掩码
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        # 对于SDPA，处理特殊掩码情况
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2_5_VLConfig,
        past_key_values: Cache,
    ):
        """从2D掩码生成4D因果掩码。"""
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            # 创建因果掩码：位置i只能看到位置j <= i
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                # 结合填充掩码
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
    因果语言模型输出类，包含rope_deltas。

    Args:
        loss: Language modeling loss.
        logits: Prediction scores.
        past_key_values: Cached key-value states.
        hidden_states: Hidden states.
        attentions: Attention weights.
        rope_deltas: Rope index difference.
    """
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None


QWEN2_5_VL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        pixel_values (`torch.FloatTensor` of shape `(seq_length, num_channels * image_size * image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2_5_VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2_5_VLImageProcessor`] for processing images.
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2_5_VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2_5_VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
"""


class Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    """用于条件生成的Qwen2.5-VL模型，包含视觉编码器和语言模型，支持多模态输入。"""
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # 权重绑定
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        # 视觉编码器
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        # 语言模型
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        # 语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # 缓存rope_deltas

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算多模态RoPE所需的3D位置索引。
        对于纯文本，三个维度（时间、高、宽）使用相同的位置ID。
        对于视觉特征，三个维度分别根据视频/图像的时空结构计算位置ID。
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            # 初始化位置ID张量，形状为 (3, batch, seq_len)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                # 只考虑非padding的token
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                # 找到所有 <|vision_start|> 的位置，其后的token即为视觉token
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                # 遍历所有图像和视频
                for _ in range(image_nums + video_nums):
                    # 找到下一个图像或视频的开始位置
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    # 处理先出现的那个
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    # 合并后的网格尺寸
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    # 当前视觉token之前的文本token数量
                    text_len = ed - st

                    # 添加文本token的位置ID
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # 生成视觉token的位置ID (t, h, w)
                    # 时间维度ID
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    time_tensor = (
                        expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                    )
                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()
                    # 高度维度ID
                    h_index = (
                        torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    )
                    # 宽度维度ID
                    w_index = (
                        torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    )
                    # 添加到列表，注意偏移量是之前的token总数 (text_len + st_idx)
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    # 更新st指针到视觉token之后
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                # 处理剩余的文本
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # 拼接所有位置ID
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            # 纯文本情况
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    @add_start_docstrings_to_model_forward(QWEN2_5_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2_5_VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
    ) -> tuple | Qwen2_5_VLCausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.)

        Returns:
            Qwen2_5_VLCausalLMOutputWithPast

        Example:
            ... (示例代码)
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 获取文本嵌入
            inputs_embeds = self.model.embed_tokens(input_ids)
            # 处理图像
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                # 将图像嵌入替换到输入嵌入中对应的位置
                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # 处理视频
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # 计算位置ID和rope_deltas（如果需要）
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # 在pre-fill阶段或第一次生成时计算
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                # 后续生成步骤，使用缓存的rope_deltas来计算位置ID
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # 前向传播语言模型
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 计算语言模型损失
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        """准备生成所需的输入，处理缓存和视觉输入。"""
        # 根据缓存位置切片输入
        if past_key_values is not None:
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None or (  # Exception 1
                is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1]
            ):  # Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # 在第一个生成步骤之后，不再需要视觉输入
        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # 决定使用inputs_embeds还是input_ids
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        # 处理静态缓存的注意力掩码
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        # 更新模型输入字典
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
            }
        )
        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: torch.LongTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """获取每个样本中图像和视频的数量，用于扩展输入。"""
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        """扩展输入以支持beam search等生成方法。"""
        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    samples = torch.split(image_grid_thw, list(image_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    if not isinstance(dict_to_expand[key], list):
                        raise TypeError(
                            f"Expected value for key '{key}' to be a list, but got {type(dict_to_expand[key])} instead."
                        )
                    tensor = torch.tensor(dict_to_expand[key])
                    lengths = list(video_nums)
                    tensor = _repeat_interleave_samples(tensor, lengths=lengths, repeat_times=expand_size)
                    dict_to_expand[key] = tensor.tolist()
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # 扩展视觉输入（如果需要）
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        # 扩展input_ids
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        # 扩展其他输入
        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


@dataclass
class Qwen2_5_VLACausalLMOutputWithPast(ModelOutput):
    """

    Base class for Qwen2_5_VL causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.

    
    
    一个扩展的输出类，用于支持额外的损失类型（如flow loss）。
    
    """

    loss: torch.FloatTensor | None = None
    flow_loss: torch.FloatTensor | None = None
    cross_entropy_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None

    channel_loss_dict: dict[torch.FloatTensor] | None = None
    channel_loss_count_dict: dict[torch.FloatTensor] | None = None


class BlockSparseMLP(nn.Module):
    """用于MoE (Mixture of Experts) 的稀疏MLP块。"""
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.hidden_act = config["hidden_act"]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[self.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class SparseMoeBlock(nn.Module):
    """稀疏MoE块，包含多个专家，并根据输入选择专家。"""
    def __init__(self, config, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        # 创建多个专家MLP
        self.experts = nn.ModuleList([BlockSparseMLP(config.experts[i]) for i in range(num_experts)])

        if not hasattr(config, "dim_inputs") or not config.dim_inputs:
            raise ValueError("Config must contain valid dim_inputs")

        self.dim_inputs = config.dim_inputs

    def forward(self, hidden_states: torch.Tensor, experts_indices: torch.Tensor) -> torch.Tensor:
        """
        将不同的hidden_states路由到对应的专家进行处理。

        Args:
            hidden_states (torch.Tensor): 形状为 (batch_size, seq_length, hidden_dim) 的张量。
            experts_indices (torch.Tensor): 形状为 (batch_size, seq_length) 的张量，
                指示每个token分配给哪个专家。

        Returns:
            output (torch.Tensor): 形状为 (batch_size, seq_length, hidden_dim) 的输出张量。
        """
        batch_size, seq_length, hidden_dim = hidden_states.size()
        output = torch.zeros_like(hidden_states)

        for expert_idx, expert in enumerate(self.experts):
            mask = experts_indices == expert_idx  # 找到分配给当前专家的token
            if mask.sum() == 0:
                continue
            dim_input = self.dim_inputs[expert_idx]

            # 选择对应的hidden_states
            selected_hidden = hidden_states[mask]
            # 通过专家处理，只取前dim_input维
            processed_hidden = expert(selected_hidden[:, :dim_input])

            # 将处理后的结果放回输出张量
            batch_indices, seq_indices = torch.where(mask)
            output[batch_indices, seq_indices, :dim_input] = processed_hidden

        return output


# 再次定义注意力类字典（上面已经定义过，这里是重复的，可能是为了保持代码完整性或后续修改）
QWEN2_5_VL_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLAttention,
    "flash_attention_2": Qwen2_5_VLFlashAttention2,
    "sdpa": Qwen2_5_VLSdpaAttention,
}


class Qwen2_5_VLDecoderLayer_with_MoE(nn.Module):
    """支持MoE的解码器层。"""
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int, num_experts: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

        # 自注意力层
        self.self_attn = QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        # 层归一化
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 根据配置选择MoE或标准MLP
        if config.mlp_moe:
            self.moe = SparseMoeBlock(config, num_experts=num_experts)
            self.mlp = None
        else:
            self.mlp = Qwen2_5_VLMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        token_types=None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        # 确保数据类型匹配
        hidden_states = hidden_states.to(self.input_layernorm.weight.dtype)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states.to(self.self_attn.q_proj.weight.dtype)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = hidden_states.to(self.post_attention_layernorm.weight.dtype)
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.mlp is None:  # 使用MoE
            hidden_states = hidden_states.to(self.moe.experts[0].down_proj.weight.dtype)
            hidden_states = self.moe(hidden_states, token_types)
        else:  # 使用标准MLP
            hidden_states = hidden_states.to(self.mlp.down_proj.weight.dtype)
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class Qwen2_5_VLMoEModel(Qwen2_5_VLPreTrainedModel):
    """Qwen2.5-VL模型，使用MoE (Mixture of Experts) 架构。"""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_experts: int | None = None,
        *args,
        **kwargs,
    ):
        """从预训练模型加载，并可选地配置MoE专家数量。"""
        config = kwargs.get("config")
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        if num_experts is not None:
            config.num_experts = num_experts

        kwargs["config"] = config
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # 创建带有MoE的解码器层
        self.layers = nn.ModuleList(
            [
                Qwen2_5_VLDecoderLayer_with_MoE(config, layer_idx, config.num_experts)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        moe_token_types: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        # 设置默认值
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 输入验证
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if moe_token_types is None:
            raise ValueError("moe_token_types must be provided for MoE routing")

        # 梯度检查点与缓存不兼容
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 初始化动态缓存
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        # 获取输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 设置缓存位置
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # 设置位置ID
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # 创建因果注意力掩码（传入moe_token_types以支持双向注意力）
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
            moe_token_types,
        )

        hidden_states = inputs_embeds

        # 创建位置嵌入
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 输出收集器
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # 遍历解码器层
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    moe_token_types,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    token_types=moe_token_types,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 最终层归一化
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        moe_token_types: torch.LongTensor | None = None,
    ):
        """更新因果注意力掩码，对于特定token类型（如MoE路由token）支持双向注意力。"""
        # Flash Attention 2 内部处理掩码
        if self.config._attn_implementation == "flash_attention_2":
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # 对于SDPA，尽量使用is_causal参数
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # 确定目标长度
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # 生成4D因果注意力掩码
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        # 根据token类型修改掩码，支持双向注意力
        if moe_token_types is not None:
            # 找到类型为1的token（MoE路由token）
            type1_tokens = (moe_token_types == 1).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]

            # 创建双向注意力区域
            type1_mask = torch.zeros_like(causal_mask)
            type1_region = type1_tokens & type1_tokens.transpose(-1, -2)  # [B, 1, S, S]
            type1_mask = type1_mask.masked_fill(type1_region, 1.0).to(torch.bool)

            # 在类型1token的区域，将掩码设为0（允许双向注意力）
            causal_mask = torch.where(
                type1_mask,
                torch.zeros_like(causal_mask),
                causal_mask,
            )

        # 对于SDPA，处理特殊掩码情况
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2_5_VLConfig,
        past_key_values: Cache,
    ):
        """生成4D因果注意力掩码（与Qwen2_5_VLModel中的方法相同）。"""
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


# 导出公共接口
__all__ = [
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLModel",
    "Qwen2_5_VLPreTrainedModel",
    "Qwen2_5_VLDecoderLayer_with_MoE",
    "Qwen2_5_VLMoEModel",
]
