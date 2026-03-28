# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 从dataclasses模块导入dataclass和field装饰器，用于创建配置类
from dataclasses import dataclass, field

# 从lerobot.configs.policies导入PreTrainedConfig基类
from lerobot.configs.policies import PreTrainedConfig

# 从lerobot.configs.types导入特征类型、归一化模式和策略特征类
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

# 从lerobot.optim.optimizers导入AdamW优化器配置类
from lerobot.optim.optimizers import AdamWConfig

# 从lerobot.optim.schedulers导入余弦退火学习率调度器配置类
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

# 从lerobot.utils.constants导入动作和观测状态常量
from lerobot.utils.constants import ACTION, OBS_STATE


# 使用装饰器注册WallXConfig子类，名称为"wall_x"
@PreTrainedConfig.register_subclass("wall_x")
# 使用dataclass装饰器创建数据类
@dataclass
class WallXConfig(PreTrainedConfig):
    """
    Configuration class for Wall-X policy.

    Wall-X is based on Qwen2.5-VL with action prediction capabilities using flow matching.
    It supports cross-embodiment robotic control through unified action representations.

    This config supports multi-modal learning with vision, language, and action data.
    """

    # ==================== Input / Output Structure ====================
    # 观测步数，表示每次策略调用时处理的观测帧数
    n_obs_steps: int = 1
    # 动作块大小，wall-x中的action_horizon，表示每次生成的动作序列长度
    chunk_size: int = 32  # action_horizon in wall-x
    # 执行的动作步数，表示每次环境执行的动作步数
    n_action_steps: int = 32

    # 动作维度 - wall-x使用20维动作空间
    max_action_dim: int = 20
    # 状态维度 - 用于本体感觉，最大为20维
    max_state_dim: int = 20  # For proprioception

    # 归一化映射字典，指定不同特征类型的归一化模式
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,  # 视觉特征使用恒等归一化（不归一化）
            "STATE": NormalizationMode.MEAN_STD,  # 状态特征使用均值标准差归一化
            "ACTION": NormalizationMode.MEAN_STD,  # 动作特征使用均值标准差归一化
        }
    )

    # ==================== Action Prediction ====================
    # 预训练模型路径，指定从HuggingFace Hub加载的预训练模型
    pretrained_name_or_path: str = "x-square-robot/wall-oss-flow"

    # 分词器设置，指定动作分词器的路径
    action_tokenizer_path: str | None = "lerobot/fast-action-tokenizer"

    # 动作预测模式，可以是"diffusion"（扩散模型）或"fast"（快速离散动作预测）
    prediction_mode: str = "diffusion"

    # 注意力实现方式，选项："eager"（默认）、"flash_attention_2"、"sdpa"
    # 注意：flash_attention_2需要flash-attn==2.7.4.post1
    attn_implementation: str = "eager"

    # ==================== Optimizer Presets ====================
    # 优化器学习率
    optimizer_lr: float = 2e-5
    # 优化器beta参数，用于AdamW优化器
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    # 优化器epsilon参数，用于数值稳定性
    optimizer_eps: float = 1e-8
    # 优化器权重衰减系数
    optimizer_weight_decay: float = 0.01
    # 梯度裁剪范数，防止梯度爆炸
    optimizer_grad_clip_norm: float = 1.0

    # 学习率调度器预热步数
    scheduler_warmup_steps: int = 1000
    # 学习率调度器衰减步数
    scheduler_decay_steps: int = 100000
    # 学习率调度器最终学习率
    scheduler_decay_lr: float = 1e-6

    # 初始化后处理方法，用于验证和设置配置
    def __post_init__(self):
        # 调用父类的初始化后处理方法
        super().__post_init__()

        # 输入验证：确保动作步数不超过块大小
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        # 验证预测模式是否有效
        if self.prediction_mode not in ["diffusion", "fast"]:
            raise ValueError(f"prediction_mode must be 'diffusion' or 'fast', got {self.prediction_mode}")

        # 根据预测模式分配use_fast_tokenizer标志
        if self.prediction_mode == "fast":
            # 快速模式使用快速分词器
            self.use_fast_tokenizer = True
        elif self.prediction_mode == "diffusion":
            # 扩散模式不使用快速分词器，并禁用动作分词器
            self.use_fast_tokenizer = False
            self.action_tokenizer_path = None  # disable action tokenizer for diffusion mode
        else:
            # 无效的预测模式
            raise ValueError(f"prediction_mode must be 'diffusion' or 'fast', got {self.prediction_mode}")

    # 验证特征方法，确保输入输出特征符合Wall-X策略的要求
    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        # 从输入特征中筛选出视觉特征
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        # Wall-X策略必须至少有一个视觉输入特征
        if not image_features:
            raise ValueError(
                "Wall-X policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        # 检查观测状态特征是否存在
        if OBS_STATE not in self.input_features:
            # 如果不存在，创建状态特征并添加到输入特征中
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature
        else:
            # 如果存在，验证状态维度不超过最大值
            state_shape = self.input_features[OBS_STATE].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    f"Either reduce state dimension or increase max_state_dim in config."
                )

        # 检查动作特征是否存在
        if ACTION not in self.output_features:
            # 如果不存在，创建动作特征并添加到输出特征中
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature
        else:
            # 如果存在，验证动作维度不超过最大值
            action_shape = self.output_features[ACTION].shape
            action_dim = action_shape[0] if action_shape else 0
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}. "
                    f"Either reduce action dimension or increase max_action_dim in config."
                )

    # 获取优化器预设方法，返回AdamW优化器配置
    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,  # 学习率
            betas=self.optimizer_betas,  # beta参数
            eps=self.optimizer_eps,  # epsilon参数
            weight_decay=self.optimizer_weight_decay,  # 权重衰减
            grad_clip_norm=self.optimizer_grad_clip_norm,  # 梯度裁剪范数
        )

    # 获取调度器预设方法，返回余弦退火学习率调度器配置
    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,  # 峰值学习率
            decay_lr=self.scheduler_decay_lr,  # 衰减后学习率
            num_warmup_steps=self.scheduler_warmup_steps,  # 预热步数
            num_decay_steps=self.scheduler_decay_steps,  # 衰减步数
        )

    # 观测delta索引属性，返回None（表示没有时间偏移）
    @property
    def observation_delta_indices(self) -> list:
        return None  # 没有观测时间偏移

    # 动作delta索引属性，返回从0到chunk_size的列表
    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))  # 动作时间偏移索引

    # 奖励delta索引属性，返回None（表示没有奖励时间偏移）
    @property
    def reward_delta_indices(self) -> None:
        return None  # 没有奖励时间偏移
