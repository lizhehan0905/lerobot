# 从transformers.configuration_utils导入PretrainedConfig基类
from transformers.configuration_utils import PretrainedConfig

# 从transformers.modeling_rope_utils导入rope_config_validation函数
from transformers.modeling_rope_utils import rope_config_validation


# Qwen2.5-VL视觉配置类，继承自PretrainedConfig
class Qwen2_5_VLVisionConfig(PretrainedConfig):
    # 模型类型标识符
    model_type = "qwen2_5_vl"
    # 基础配置键名
    base_config_key = "vision_config"

    # 初始化方法，设置视觉编码器的参数
    def __init__(
        self,
        depth=32,  # 视觉编码器的深度（层数）
        hidden_size=3584,  # 隐藏层大小
        hidden_act="silu",  # 隐藏层激活函数
        intermediate_size=3420,  # 中间层大小
        num_heads=16,  # 注意力头数
        in_channels=3,  # 输入通道数（RGB图像）
        patch_size=14,  # 图像块大小
        spatial_merge_size=2,  # 空间合并大小
        temporal_patch_size=2,  # 时间块大小
        tokens_per_second=4,  # 每秒令牌数
        window_size=112,  # 窗口大小
        out_hidden_size=3584,  # 输出隐藏层大小
        fullatt_block_indexes=[7, 15, 23, 31],  # 全注意力块索引
        initializer_range=0.02,  # 初始化范围
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置实例属性
        self.depth = depth  # 视觉编码器深度
        self.hidden_size = hidden_size  # 隐藏层大小
        self.hidden_act = hidden_act  # 激活函数
        self.intermediate_size = intermediate_size  # 中间层大小
        self.num_heads = num_heads  # 注意力头数
        self.in_channels = in_channels  # 输入通道数
        self.patch_size = patch_size  # 图像块大小
        self.spatial_merge_size = spatial_merge_size  # 空间合并大小
        self.temporal_patch_size = temporal_patch_size  # 时间块大小
        self.tokens_per_second = tokens_per_second  # 每秒令牌数
        self.window_size = window_size  # 窗口大小
        self.fullatt_block_indexes = fullatt_block_indexes  # 全注意力块索引
        self.out_hidden_size = out_hidden_size  # 输出隐藏层大小
        self.initializer_range = initializer_range  # 初始化范围

# 定义一个配置类，继承自PretrainedConfig，用于存储Qwen2_5_VL模型的配置
class Qwen2_5_VLConfig(PretrainedConfig):
    r"""
    这是用于存储[`Qwen2_5_VLModel`]配置的类。它用于根据指定的参数实例化一个Qwen2-VL模型，
    定义模型架构。使用默认值实例化配置将产生与Qwen2-VL-7B-Instruct相似的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。详细信息请阅读[`PretrainedConfig`]的文档。

    参数说明：
        vocab_size (int, 可选, 默认152064):
            Qwen2_5_VL模型的词汇表大小。定义了调用[`Qwen2_5_VLModel`]时输入的`inputs_ids`可以表示的不同token数量。
        hidden_size (int, 可选, 默认8192):
            隐藏表示的维度。
        intermediate_size (int, 可选, 默认29568):
            MLP表示的维度。
        num_hidden_layers (int, 可选, 默认80):
            Transformer编码器中隐藏层的数量。
        num_attention_heads (int, 可选, 默认64):
            Transformer编码器中每个注意力层的注意力头数量。
        num_key_value_heads (int, 可选, 默认8):
            用于实现分组查询注意力的键值头数量。如果`num_key_value_heads=num_attention_heads`，
            模型使用多头注意力(MHA)；如果`num_key_value_heads=1`，使用多查询注意力(MQA)；
            否则使用分组查询注意力(GQA)。更多细节请参考论文。
        hidden_act (str或function, 可选, 默认"silu"):
            解码器中使用的非线性激活函数。
        max_position_embeddings (int, 可选, 默认32768):
            模型可能处理的最大序列长度。
        initializer_range (float, 可选, 默认0.02):
            初始化所有权重矩阵的截断正态分布的标准差。
        rms_norm_eps (float, 可选, 默认1e-05):
            RMS归一化层使用的epsilon值。
        use_cache (bool, 可选, 默认True):
            模型是否应该返回最后的键/值注意力（并非所有模型都使用）。
        tie_word_embeddings (bool, 可选, 默认False):
            模型的输入和输出词嵌入是否应该绑定。
        rope_theta (float, 可选, 默认1000000.0):
            RoPE嵌入的基频。
        use_sliding_window (bool, 可选, 默认False):
            是否使用滑动窗口注意力。
        sliding_window (int, 可选, 默认4096):
            滑动窗口注意力(SWA)的窗口大小。
        max_window_layers (int, 可选, 默认80):
            使用SWA的层数。底层使用SWA，顶层使用全注意力。
        attention_dropout (float, 可选, 默认0.0):
            注意力概率的dropout比率。
        vision_config (Dict, 可选):
            视觉编码器初始化的配置。
        rope_scaling (Dict, 可选):
            包含RoPE嵌入缩放配置的字典。
        ...（其他参数说明省略）
    """

    # 定义模型类型，用于标识这是qwen2_5_vl模型
    model_type = "qwen2_5_vl"
    # 定义子配置，将vision_config映射到Qwen2_5_VLVisionConfig类
    sub_configs = {"vision_config": Qwen2_5_VLVisionConfig}
    # 在推理时需要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 默认的张量并行计划，定义各层权重如何切分
    # colwise表示按列切分，rowwise表示按行切分
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",   # 查询投影按列切分
        "layers.*.self_attn.k_proj": "colwise",   # 键投影按列切分
        "layers.*.self_attn.v_proj": "colwise",   # 值投影按列切分
        "layers.*.self_attn.o_proj": "rowwise",   # 输出投影按行切分
        "layers.*.mlp.gate_proj": "colwise",      # MLP门控投影按列切分
        "layers.*.mlp.up_proj": "colwise",        # MLP上投影按列切分
        "layers.*.mlp.down_proj": "rowwise",      # MLP下投影按行切分
    }
    # 默认的流水线并行计划，定义各阶段的输入输出
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),  # 嵌入层：输入token ids，输出嵌入
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),  # 层：输入隐藏状态和注意力掩码，输出隐藏状态
        "norm": (["hidden_states"], ["hidden_states"]),  # 归一化层：输入隐藏状态，输出隐藏状态
    }

    # 构造函数，初始化配置实例
    def __init__(
        self,
        vocab_size=152064,           # 词汇表大小
        hidden_size=8192,            # 隐藏层维度
        intermediate_size=29568,     # 中间层维度
        num_hidden_layers=80,        # 隐藏层数量
        num_attention_heads=64,      # 注意力头数量
        num_key_value_heads=8,       # 键值头数量（用于GQA）
        hidden_act="silu",           # 激活函数，使用SiLU
        max_position_embeddings=32768,  # 最大位置编码长度
        initializer_range=0.02,      # 初始化范围
        rms_norm_eps=1e-05,          # RMS归一化的epsilon
        use_cache=True,              # 是否使用KV缓存
        tie_word_embeddings=False,   # 是否绑定词嵌入
        rope_theta=1000000.0,        # RoPE的theta参数
        use_sliding_window=False,    # 是否使用滑动窗口
        sliding_window=4096,         # 滑动窗口大小
        max_window_layers=80,        # 滑动窗口最大层数
        attention_dropout=0.0,       # 注意力dropout
        vision_config=None,          # 视觉配置
        rope_scaling=None,           # RoPE缩放配置
        num_experts=4,               # 专家数量（用于MoE）
        experts=None,                # 专家配置
        dof_config=None,             # 自由度配置
        noise_scheduler=None,        # 噪声调度器
        dim_inputs=(1536, 1536),     # 输入维度
        attention_moe=False,         # 注意力是否使用MoE
        mlp_moe=False,               # MLP是否使用MoE
        **kwargs,                    # 其他参数
    ):
        # 处理视觉配置：如果传入字典，则实例化视觉配置对象
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        # 如果没有视觉配置，使用默认配置
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        # 设置基本配置参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        # 初始化所有层类型为密集层（dense）
        self.layer_types = ["dense"] * num_hidden_layers

        # 向后兼容：如果没有指定num_key_value_heads，则使用num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        # MoE（混合专家）相关配置
        self.num_experts = num_experts
        self.experts = experts
        self.dof_config = dof_config
        self.noise_scheduler = noise_scheduler
        # 将dim_inputs转换为元组
        self.dim_inputs = tuple(dim_inputs)
        self.attention_moe = attention_moe
        self.mlp_moe = mlp_moe

        # 处理RoPE缩放配置：如果存在且类型为"mrope"，转换为"default"
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            # 添加rope_type字段以保持兼容
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        # 验证rope配置，忽略mrope_section键
        rope_config_validation(self, ignore_keys={"mrope_section"})

        # 调用父类初始化，传递词嵌入绑定和其他参数
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    # 属性方法，返回当前配置作为文本配置
    @property
    def text_config(self):
        return self


# 模块导出列表，定义哪些符号可以被外部导入
__all__ = ["Qwen2_5_VLConfig"]
