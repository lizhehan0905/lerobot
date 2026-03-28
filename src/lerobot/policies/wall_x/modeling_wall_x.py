#!/usr/bin/env python  # Python解释器路径声明，指定使用Python解释器执行脚本

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

"""
Wall-X: Cross-embodiment robotic control using Qwen2.5-VL with flow matching.

[Paper](https://github.com/x2-robot/wall-x)

Install wall-x extra dependencies:
```bash
pip install -e ".[wall_x]"
```

Example of finetuning a wall-x model:
```bash
lerobot-train \
--policy.type=wall_x \
--dataset.repo_id=your/dataset \
--batch_size=32 \
--steps=100000
```
"""



# 导入标准库
import math
from collections import deque  # 双端队列，用于管理动作序列缓存
from os import PathLike  # 路径类型提示
from typing import Any  # 任意类型注解

# 导入第三方科学计算和深度学习库
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入参数高效微调（LoRA）相关模块
from peft import LoraConfig, get_peft_model

# 导入图像处理库
from PIL import Image

# 导入 Qwen-VL 工具函数
from qwen_vl_utils.vision_process import smart_resize

# 导入 PyTorch 张量类型和分布
from torch import Tensor
from torch.distributions import Beta
from torch.nn import CrossEntropyLoss

# 导入常微分方程求解器，用于扩散模型的ODE采样
from torchdiffeq import odeint

# 导入 HuggingFace Transformers 相关模块
from transformers import AutoProcessor, BatchFeature
from transformers.cache_utils import StaticCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.utils import is_torchdynamo_compiling, logging

# 导入 LeRobot 框架相关模块
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.policies.wall_x.configuration_wall_x import WallXConfig
from lerobot.policies.wall_x.constant import (
    GENERATE_SUBTASK_RATIO,  # 生成子任务的比例
    IMAGE_FACTOR,            # 图像缩放因子
    MAX_PIXELS,              # 最大像素数
    MIN_PIXELS,              # 最小像素数
    MODEL_TYPE,              # 模型类型
    PRIORITY_ORDER,          # 优先级顺序
    RESOLUTION,              # 分辨率
    TOKENIZER_MAX_LENGTH,    # 分词器最大长度
)
# 导入自定义的 Qwen 模型配置和组件
from lerobot.policies.wall_x.qwen_model.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from lerobot.policies.wall_x.qwen_model.qwen2_5_vl_moe import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLACausalLMOutputWithPast,
    Qwen2_5_VLMoEModel,
)
# 导入 Wall-X 工具函数
from lerobot.policies.wall_x.utils import (
    get_wallx_normal_text,          # 生成标准文本指令
    preprocesser_call,               # 调用处理器
    process_grounding_points,        # 处理接地点
    replace_action_token,            # 替换动作令牌
)
# 导入 LeRobot 常量
from lerobot.utils.constants import ACTION, OBS_STATE

# 获取日志记录器
logger = logging.get_logger(__name__)


class SinusoidalPosEmb(nn.Module):
    """
    正弦位置嵌入模块。
    用于扩散模型中的时间步长编码，生成周期性的位置编码。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 嵌入维度

    def forward(self, x):
        """
        前向传播：根据输入时间步 x 生成位置嵌入。

        参数：
            x: 形状为 (batch_size,) 的时间步张量

        返回：
            形状为 (batch_size, dim) 的正弦位置嵌入
        """
        device = x.device
        half_dim = self.dim // 2
        # 计算对数空间的指数底数
        emb = math.log(10000) / (half_dim - 1)
        # 生成频率向量
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 将时间步 x 与频率向量相乘
        emb = x[:, None] * emb[None, :]
        # 拼接正弦和余弦分量，形成最终嵌入
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionHead(nn.Module):
    """
    动作预测头，基于流匹配（Flow Matching）。
    使用 Beta 分布进行噪声调度，并为动作序列生成时间嵌入。
    该模块在训练时对动作进行加噪，并预测从噪声到真实动作的“流”（flow）。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # 计算动作维度（所有自由度之和）
        self.action_dim = sum(config.dof_config.values())
        # 计算本体感知维度（所有智能体位置参数之和）
        self.propri_dim = sum(config.agent_pos_config.values())
        self.hidden_size = config.hidden_size  # 隐藏层维度

        # Beta 分布参数，用于噪声调度
        self.beta_alpha = 1.5
        self.beta_beta = 1.0
        self.s = 0.999  # 时间缩放因子

        # 时间步嵌入模块
        self.time_embed = SinusoidalPosEmb(config.hidden_size)

        # 动作嵌入网络
        # w1: 输入为 [动作 + DOF掩码] -> 隐藏层
        self.w1 = nn.Linear(self.action_dim * 2, self.hidden_size, bias=False)
        # w2: 输入为 [动作嵌入 + 时间嵌入] -> 隐藏层
        self.w2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        # w3: 从隐藏层到隐藏层的变换
        self.w3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()  # Sigmoid Linear Unit 激活函数

        # 将隐藏状态投影回动作空间
        self.action_proj_back = nn.Linear(self.hidden_size, self.action_dim, bias=False)

        # 本体感知投影网络（输入为本体感知 + DOF掩码）
        self.propri_proj = nn.Linear(self.propri_dim * 2, self.hidden_size, bias=False)

    def sample_time(self, batch_size, device):
        """
        使用 Beta 分布采样时间步 t（始终使用 float32 以保证数值稳定性）。

        参数：
            batch_size: 批量大小
            device: 设备

        返回：
            形状为 (batch_size,) 的采样时间步张量
        """
        beta_dist = Beta(
            torch.tensor(self.beta_alpha, dtype=torch.float32, device=device),
            torch.tensor(self.beta_beta, dtype=torch.float32, device=device),
        )
        sample = beta_dist.sample([batch_size])
        time = (1 - sample) * self.s
        return time

    def forward(self, action_chunk, dof_mask=None):
        """
        训练阶段的前向传播：对动作序列注入噪声，并计算对应的流（flow）。

        参数：
            action_chunk: 动作序列 [batch, seq_len, action_dim]
            dof_mask: 自由度掩码 [batch, seq_len, action_dim]，标记有效自由度

        返回：
            tuple: (动作嵌入, 流目标)
        """
        batch_size = action_chunk.shape[0]
        device = action_chunk.device
        weight_dtype = self.w1.weight.dtype  # 获取网络权重的数据类型

        # 采样时间步（在 autocast 之外进行，Beta分布需要 float32）
        time = self.sample_time(batch_size, device)
        t = time.unsqueeze(-1).unsqueeze(-1)  # 扩展维度以便广播

        # 在 float32 下计算噪声和流，以保证数值稳定性
        noise = torch.randn_like(action_chunk, dtype=torch.float32)
        action_chunk_f32 = action_chunk.to(torch.float32)
        # 线性插值：带噪动作 = (1-t) * 噪声 + t * 真实动作
        noisy_action = (1 - t) * noise + t * action_chunk_f32
        # 流目标 = 真实动作 - 噪声
        flow = action_chunk_f32 - noise

        # 如果提供了自由度掩码，则将其与带噪动作拼接
        if dof_mask is not None:
            noisy_action = torch.cat([noisy_action, dof_mask.to(torch.float32)], dim=-1)

        # 将带噪动作转换为网络权重的数据类型，输入线性层
        noisy_action = noisy_action.to(dtype=weight_dtype)
        action_embed = self.w1(noisy_action)

        # 生成时间嵌入
        time_embed = self.time_embed(time)
        time_embed = time_embed.unsqueeze(1).repeat(1, action_embed.shape[1], 1)
        time_embed = time_embed.to(dtype=weight_dtype)

        # 拼接动作嵌入和时间嵌入
        concat_embed = torch.cat([action_embed, time_embed], dim=-1)
        concat_embed = self.w2(concat_embed)
        embed = self.w3(self.act_fn(concat_embed))

        return embed, flow

    def step(self, timestep, noisy_action, dof_mask=None):
        """
        推理阶段的一个去噪步骤。

        参数：
            timestep: 当前时间步
            noisy_action: 带噪动作
            dof_mask: 自由度掩码

        返回：
            当前时间步的动作嵌入
        """
        weight_dtype = self.w1.weight.dtype

        if dof_mask is not None:
            noisy_action = torch.cat([noisy_action, dof_mask], dim=-1)
        noisy_action = noisy_action.to(dtype=weight_dtype)

        time_embed = self.time_embed(timestep)
        action_embed = self.w1(noisy_action)

        time_embed = time_embed.unsqueeze(1).repeat(1, action_embed.shape[1], 1)
        time_embed = time_embed.to(device=noisy_action.device, dtype=weight_dtype)

        concat_embed = torch.cat([action_embed, time_embed], dim=-1)
        concat_embed = self.w2(concat_embed)
        embed = self.w3(self.act_fn(concat_embed))

        return embed

    def flow_loss(self, action_hidden_states, flow, dof_mask=None):
        """
        计算流匹配损失（MSE）。

        参数：
            action_hidden_states: 预测的隐藏状态
            flow: 真实的流目标
            dof_mask: 自由度掩码

        返回：
            计算得到的损失张量
        """
        # 将所有输入转为 float32 以保证稳定性
        action_hidden_states = action_hidden_states.to(torch.float32)
        flow = flow.to(torch.float32)

        action_pred = self.action_proj_back(action_hidden_states)
        loss = F.mse_loss(action_pred, flow, reduction="none")

        if dof_mask is not None:
            dof_mask = dof_mask.reshape(-1, dof_mask.shape[-1]).to(torch.float32)
            loss = loss * dof_mask  # 仅对有效自由度计算损失

        return loss

    def proprioception_proj(self, proprioception, dof_mask=None, use_history=False):
        """
        将本体感知数据投影到隐藏空间。

        参数：
            proprioception: 本体感知数据 [batch, seq_len, prop_dim]
            dof_mask: 自由度掩码
            use_history: 是否使用历史信息（此处未使用，但为接口保留）

        返回：
            投影后的隐藏状态
        """
        # 确保数据类型和设备与权重对齐
        proprioception = proprioception.to(device=self.propri_proj.weight.device).to(
            dtype=self.propri_proj.weight.dtype
        )

        if dof_mask is not None:
            # 将本体感知数据与自由度掩码拼接
            proprioception = torch.cat([proprioception, dof_mask], dim=-1)

        proprioception = proprioception.to(device=self.propri_proj.weight.device).to(
            dtype=self.propri_proj.weight.dtype
        )
        return self.propri_proj(proprioception)


class Qwen2_5_VLMoEForAction(Qwen2_5_VLForConditionalGeneration):
    """
    用于动作处理的 Qwen2.5 视觉-语言混合专家（MoE）模型。
    该类扩展了基础的 Qwen2.5 VL 模型，添加了动作令牌处理能力，
    并可选地支持 LoRA 微调。
    """

    # 定义权重绑定的键（语言模型头与嵌入层共享权重）
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen2_5_VLConfig  # 关联的配置类
    # 指定不应拆分的模块（用于模型并行）
    _no_split_modules = ["Qwen2_5_VLDecoderLayer_with_MoE", "Qwen2_5_VLVisionBlock"]

    def init_weights(self):
        """初始化权重。如果语言模型组件已存在，则跳过基类的初始化。"""
        if getattr(self.model, "language_model", None) is not None:
            return
        super().init_weights()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path,
        config=None,
        action_tokenizer_path=None,
        attn_implementation: str = "eager",
        cache_dir: str | PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        strict: bool = False,
        **kwargs: Any,
    ):
        """
        从预训练路径加载模型。

        参数：
            pretrained_name_or_path: 预训练模型路径
            config: 模型配置，若为 None 则自动加载
            action_tokenizer_path: 动作分词器路径
            attn_implementation: 注意力实现方式（如 "eager", "flash_attention_2"）
            cache_dir: 缓存目录
            force_download: 是否强制下载
            local_files_only: 是否仅使用本地文件
            token: 认证令牌
            revision: 模型版本
            strict: 是否严格加载状态字典
            **kwargs: 其他参数

        返回：
            加载好的 Qwen2_5_VLMoEForAction 模型实例
        """
        # 如果未提供配置，则从预训练路径加载
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                strict=strict,
                **kwargs,
            )
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation

        # 加载处理器
        processor = AutoProcessor.from_pretrained(pretrained_name_or_path, use_fast=True)
        if action_tokenizer_path is not None:
            action_tokenizer = AutoProcessor.from_pretrained(action_tokenizer_path, trust_remote_code=True)
            processor.action_processor = action_tokenizer
        else:
            action_tokenizer = None

        # 将 pad_token_id 添加到配置中
        config.pad_token_id = processor.tokenizer.pad_token_id
        config.text_config.pad_token_id = processor.tokenizer.pad_token_id

        # 初始化模型
        model = cls(config, processor=processor, action_tokenizer=action_tokenizer, **kwargs)

        # 调整词嵌入大小以匹配分词器词汇表大小
        model.resize_token_embeddings(len(processor.tokenizer))

        # 尝试加载模型权重文件（model.safetensors）
        print(f"Loading model from: {pretrained_name_or_path}")
        try:
            from transformers.utils import cached_file
            from safetensors.torch import load_file

            # 先尝试加载 safetensors 格式
            resolved_file = cached_file(
                pretrained_name_or_path,
                "model.safetensors",
                cache_dir=kwargs.get("cache_dir"),
                force_download=kwargs.get("force_download", False),
                resume_download=kwargs.get("resume_download"),
                proxies=kwargs.get("proxies"),
                token=kwargs.get("token"),
                revision=kwargs.get("revision"),
                local_files_only=kwargs.get("local_files_only", False),
            )
            sd = load_file(resolved_file)
            print("✓ Loaded state dict from model.safetensors")
        except Exception as e:
            print(f"Could not load state dict from remote files: {e}")
            print("Returning model without loading pretrained weights")
            return model

        # 过滤掉归一化器统计参数（这些参数由外部管理）
        state_dict = {}
        del_keys = []
        for key in sd.keys():
            if "action_preprocessor.normalizer" in key:
                del_keys.append(key)
        for key in del_keys:
            del sd[key]
        state_dict.update(sd)

        # 非严格加载状态字典（允许缺少某些键）
        model.load_state_dict(state_dict, strict=False)

        return model

    def __init__(
        self,
        config,
        use_fast_tokenizer=False,
        processor=None,
        action_tokenizer=None,
        action_mapper=None,
        flow_loss_weight=1.0,
    ):
        """
        初始化模型。

        参数：
            config: 模型配置
            use_fast_tokenizer: 是否使用快速分词器
            processor: 文本和图像处理器
            action_tokenizer: 动作专用分词器
            action_mapper: 动作映射工具（未使用）
            flow_loss_weight: 流匹配损失的权重
        """
        super().__init__(config)

        # 初始化视觉变换器和语言模型组件
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLMoEModel(config)  # 混合专家（MoE）模型
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化损失函数（不进行归约，以便逐通道计算损失）
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.flow_loss_weight = flow_loss_weight
        self.use_fast_tokenizer = use_fast_tokenizer
        self.processor = processor
        self.action_tokenizer = action_tokenizer

        # 定义动作令牌 ID 映射
        self.define_action_token_id()

        # 用于缓存 rope 位置增量
        self.rope_deltas = None

        # 初始化动作预处理器（即 ActionHead）
        self.action_preprocessor = ActionHead(config)

        # 如果配置中指定了 LoRA，则添加 LoRA 适配器
        if hasattr(config, "use_lora") and config.use_lora:
            self.add_lora(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
            )

        # 初始化权重并应用最终处理
        self.post_init()

    def to_bfloat16_for_selected_params(self):
        """
        将模型转换为 bfloat16，但将特定参数（层归一化、动作预处理器等）保留为 float32
        以提高数值稳定性。
        """
        self.to(dtype=torch.bfloat16)

        params_to_keep_float32 = []

        # 确定需要保持为 float32 的参数名称
        for name, param in self.named_parameters():
            if "input_layernorm" in name or "post_attention_layernorm" in name or "model.norm" in name:
                params_to_keep_float32.append(name)
            if "action_preprocessor" in name:
                params_to_keep_float32.append(name)

        # 将这些参数的数据类型转回 float32
        for name, param in self.named_parameters():
            if name in params_to_keep_float32:
                param.data = param.data.to(torch.float32)

    def define_action_token_id(self):
        """
        根据分词器配置定义动作令牌 ID。
        创建快速动作令牌列表、本体感知令牌 ID 和动作令牌 ID 的映射。
        """
        fast_action_token_list = []
        if self.use_fast_tokenizer:
            # 生成所有快速动作令牌的 ID
            for i in range(self.processor.tokenizer.init_kwargs["action_token_vocab_size"]):
                action_token_id = self.processor.tokenizer.convert_tokens_to_ids(f"<|action_token_{i}|>")
                fast_action_token_list.append(action_token_id)

        # 获取特殊动作令牌的 ID
        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")
        propri_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|propri|>")

        # 存储到字典中
        self.action_token_id_set = {
            "fast_action_token_list": fast_action_token_list,
            "propri_token_id": propri_token_id,
            "action_token_id": action_token_id,
        }

    def add_lora(self, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1):
        """
        向模型添加 LoRA（低秩适应）适配器。

        参数：
            r: 秩
            lora_alpha: 缩放参数
            target_modules: 要应用 LoRA 的模块名称列表
            lora_dropout: Dropout 概率
        """
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)

        # 打印可训练参数信息
        self.model.print_trainable_parameters()
        print("==============use lora===================")

    def get_input_embeddings(self):
        """返回输入嵌入层。"""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """设置输入嵌入层。"""
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """返回输出嵌入层（语言模型头）。"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """设置输出嵌入层。"""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """设置解码器模型。"""
        self.model = decoder

    def get_decoder(self):
        """返回解码器模型。"""
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
        为视觉和文本令牌计算 3D RoPE（旋转位置嵌入）索引。

        该方法处理视觉令牌（图像/视频）的 3D 位置嵌入（时间、高度、宽度），
        并为文本令牌使用标准的 1D 位置嵌入。

        参数：
            input_ids: 输入令牌 ID
            image_grid_thw: 图像网格尺寸（每个图像的 [时间, 高度, 宽度]）
            video_grid_thw: 视频网格尺寸
            second_per_grid_ts: 每个网格的时间间隔（用于视频）
            attention_mask: 注意力掩码

        返回：
            tuple: (position_ids, mrope_position_deltas)
                position_ids: 形状为 (3, batch_size, sequence_length) 的位置 ID
                mrope_position_deltas: 形状为 (batch_size, 1) 的位置增量
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

            # 初始化 3D 位置 ID 张量
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)

            # 逐样本处理
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0

                # 找到视觉令牌并统计图像/视频数量
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()

                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums

                # 处理每个视觉令牌（图像或视频）
                for _ in range(image_nums + video_nums):
                    # 查找下一个图像或视频令牌的位置
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1

                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1

                    # 根据位置决定处理图像还是视频
                    if ed_image < ed_video:
                        # 处理图像
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
                        # 处理视频
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

                    # 空间合并后的网格尺寸
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    # 为视觉令牌前的文本令牌添加位置 ID
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # 计算视觉令牌的 3D 位置嵌入
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    # 计算时间位置 ID（带时间缩放）
                    time_tensor = (
                        expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                    )
                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    # 计算空间位置 ID（高度和宽度）
                    h_index = (
                        torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    )

                    # 将视觉令牌的 3D 位置 ID 添加到列表中
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                # 添加剩余文本令牌的位置 ID
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # 拼接该样本的所有位置 ID
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            # 没有视觉令牌的情况：使用标准 1D 位置嵌入
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

    def train_step_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        moe_token_types: torch.LongTensor | None = None,  # MoE 令牌类型分配
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        action_chunk: torch.FloatTensor | None = None,  # 动作轨迹块
        proprioception: torch.FloatTensor | None = None,  # 关节位置/方向数据
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        dof_mask: torch.FloatTensor | None = None,
        agent_pos_mask: torch.FloatTensor | None = None,
        **kwargs,
    ) -> tuple | Qwen2_5_VLACausalLMOutputWithPast:
        """
        训练阶段的前向传播，处理多模态输入（视觉、文本、动作）。

        该方法处理图像、视频、文本、本体感知数据和动作序列，计算语言建模损失
        和流匹配损失。

        参数：
            input_ids: 输入令牌 ID
            attention_mask: 注意力掩码
            position_ids: 位置 ID
            past_key_values: 缓存的键值对（用于生成）
            inputs_embeds: 预计算的输入嵌入
            moe_token_types: MoE 路由的令牌类型分配
            labels: 损失计算的目标标签
            use_cache: 是否使用键值缓存
            output_attentions: 是否返回注意力权重
            output_hidden_states: 是否返回隐藏状态
            return_dict: 是否返回结构化输出
            pixel_values: 图像像素值
            pixel_values_videos: 视频像素值
            image_grid_thw: 图像网格尺寸
            video_grid_thw: 视频网格尺寸
            action_chunk: 动作轨迹数据块
            proprioception: 本体感知传感器数据
            rope_deltas: RoPE 位置增量
            cache_position: 缓存位置索引
            second_per_grid_ts: 每个网格的时间间隔
            dof_mask: 动作令牌的自由度掩码
            agent_pos_mask: 本体感知数据的智能体位置掩码
            **kwargs: 其他参数

        返回：
            模型输出，包含损失、logits 和辅助信息
        """
        batch_size, seq_length = input_ids.shape

        # 从模型配置中设置输出配置（如果未指定）
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供位置 ID，则计算 RoPE 位置 ID
        # 注意：无法使用 4D 注意力掩码计算 rope 增量。TODO: 修复此限制
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # 仅在预填充阶段计算一次 RoPE 索引
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
            # 使用先前计算的 rope 增量来获取正确的位置 ID
            else:
                delta = (
                    (cache_position[0] + self.rope_deltas).to(self.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=self.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # 使用多模态数据处理输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            # 处理图像嵌入
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # 处理视频嵌入
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]

                # 验证视频令牌与特征数量匹配
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

            # 处理本体感知数据
            if proprioception is not None:
                proprioception = proprioception.to(inputs_embeds.device).to(inputs_embeds.dtype)
                agent_pos_mask = agent_pos_mask.to(inputs_embeds.device).to(inputs_embeds.dtype)
                proprioception = self.action_preprocessor.proprioception_proj(
                    proprioception,
                    agent_pos_mask,
                    use_history=proprioception.shape[1] > 1,
                )
                mask = input_ids == self.action_token_id_set["propri_token_id"]
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                proprioception_mask = mask_expanded.to(inputs_embeds.device)

                proprioception = proprioception.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(proprioception_mask, proprioception)
            elif self.training:
                # DDP 中的虚拟前向传播，以确保梯度注册
                # 处理某些进程有本体感知数据而其他进程没有的情况
                dummy_input = torch.randn(
                    2,
                    self.action_preprocessor.propri_dim * 2,
                    device=inputs_embeds.device,
                )
                dummy_forward = self.action_preprocessor.proprioception_proj(dummy_input)
                dummy_loss = sum(p.sum() for p in dummy_forward)
                inputs_embeds = inputs_embeds + 0 * dummy_loss

            # 处理动作块数据
            if action_chunk is not None:
                action_chunk = action_chunk.to(inputs_embeds.device).to(inputs_embeds.dtype)
                dof_mask = dof_mask.to(inputs_embeds.device).to(inputs_embeds.dtype)
                noisy_action_emb, flow = self.action_preprocessor(action_chunk, dof_mask)
                mask = input_ids == self.action_token_id_set["action_token_id"]
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                action_mask = mask_expanded.to(inputs_embeds.device)

                noisy_action_emb = noisy_action_emb.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(action_mask, noisy_action_emb)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # 通过主模型进行前向传播
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            moe_token_types=moe_token_types,  # 传递令牌类型用于 MoE 路由
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        logits = self.lm_head(hidden_states)

        # 初始化损失计算变量
        loss = None
        cross_entropy_loss, flow_loss = None, None
        channel_loss_dict = None
        channel_loss_count_dict = None

        # 如果提供了标签，则计算损失,VQA的语音损失
        if labels is not None:
            loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)

            # 计算标准交叉熵损失（语言建模）
            shift_logits = logits[..., :-1, :].contiguous().to(torch.float32)
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            non_ignored_mask = shift_labels != -100
            _cross_entropy_loss = self.loss_fct(shift_logits, shift_labels)
            cross_entropy_loss = (
                _cross_entropy_loss[non_ignored_mask].mean()
                if non_ignored_mask.any()
                else torch.tensor(0.0, device=shift_logits.device, dtype=torch.float32)
            )

            if not torch.isnan(cross_entropy_loss):
                loss = loss + cross_entropy_loss.to(torch.float32)
            else:
                with torch.no_grad():
                    cross_entropy_loss.detach()

        # 计算流匹配损失
        if action_chunk is not None:
            action_mask = input_ids == self.action_token_id_set["action_token_id"]
            if action_mask.any():
                action_hidden_states = hidden_states[action_mask].to(torch.float32)
                flow = flow.reshape(-1, flow.shape[-1]).to(torch.float32)
                _flow_loss = self.action_preprocessor.flow_loss(action_hidden_states, flow, dof_mask)
                if isinstance(_flow_loss, torch.Tensor):
                    flow_loss = _flow_loss.mean()
                if loss is not None:
                    loss = loss + self.flow_loss_weight * flow_loss.to(torch.float32)
                else:
                    loss = self.flow_loss_weight * flow_loss.to(torch.float32)
                _flow_loss = _flow_loss.view(dof_mask.shape[0], dof_mask.shape[1], dof_mask.shape[2])

        # 根据 return_dict 设置返回输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLACausalLMOutputWithPast(
            loss=loss,
            cross_entropy_loss=(cross_entropy_loss.clone() if cross_entropy_loss is not None else None),
            flow_loss=flow_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            channel_loss_dict=channel_loss_dict,
            channel_loss_count_dict=channel_loss_count_dict,
        )

    def predict_action(self, predict_mode: str, **kwargs):
        """
        使用指定的预测模式预测动作。

        参数：
            predict_mode: 预测模式，可选 "fast" 或 "diffusion"
            **kwargs: 传递给 predict 方法的额外参数

        返回：
            tuple: (预测动作, 真实动作) 真实动作可能为 None
        """
        assert predict_mode in ["fast", "diffusion"]

        output = self.predict(predict_mode=predict_mode, **kwargs)

        return output["predict_action"], output.get("gt_action", None)

    @torch.no_grad()
    def predict(
        self,
        predict_mode: str,
        pred_horizon: int | None = None,
        action_dim: int | None = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        moe_token_types: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        action_chunk: torch.FloatTensor | None = None,
        proprioception: torch.FloatTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        num_inference_timesteps: int | None = 10,
        dof_mask: torch.FloatTensor | None = None,
        agent_pos_mask: torch.FloatTensor | None = None,
        re_generate: bool = False,
        **kwargs,
    ):
        """
        多模态预测方法，支持文本生成、快速动作预测和基于扩散的动作预测。

        该方法支持三种预测模式：
        1. "text": 使用自回归解码进行纯文本生成
        2. "fast": 使用离散动作令牌进行快速动作预测
        3. "diffusion": 使用扩散/流匹配进行连续动作预测

        参数：
            predict_mode: 预测模式 ("text", "fast", 或 "diffusion")
            pred_horizon: 动作序列的预测范围
            action_dim: 动作空间的维度
            input_ids: 输入令牌 ID
            attention_mask: 注意力掩码
            position_ids: 位置 ID
            past_key_values: 缓存的键值对
            inputs_embeds: 预计算的输入嵌入
            moe_token_types: MoE 令牌类型分配
            labels: 评估的目标标签
            use_cache: 是否使用键值缓存
            output_attentions: 是否返回注意力权重
            output_hidden_states: 是否返回隐藏状态
            return_dict: 是否返回结构化输出
            pixel_values: 图像像素值
            pixel_values_videos: 视频像素值
            image_grid_thw: 图像网格尺寸
            video_grid_thw: 视频网格尺寸
            action_chunk: 真实动作序列
            proprioception: 本体感知传感器数据
            rope_deltas: RoPE 位置增量
            cache_position: 缓存位置索引
            second_per_grid_ts: 每个网格的时间间隔
            num_inference_timesteps: 扩散推理步数
            dof_mask: 自由度掩码
            agent_pos_mask: 智能体位置掩码
            re_generate: 是否使用采样进行重新生成
            **kwargs: 其他参数

        返回：
            dict: 包含预测结果的字典，键可能包括：
                - 'predict_action': 预测的动作序列
                - 'gt_action': 真实动作（如果可用）
                - 'input_text': 输入文本（文本/快速模式）
                - 'predict_output_text': 生成的文本（文本/快速模式）
                - 'gt_output_text': 真实文本（文本/快速模式）
        """
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        # 文本和快速模式要求批量大小为 1，用于自回归生成
        if predict_mode in ["text", "fast"]:
            assert batch_size == 1, "predict only support batch size 1 for ar generation"

        # 从模型配置中设置输出配置（如果未指定）
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用多模态数据处理输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            # 处理图像嵌入
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]

                # 验证图像令牌与特征数量匹配
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # 处理视频嵌入
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]

                # 验证视频令牌与特征数量匹配
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

            # 处理本体感知数据
            if proprioception is not None:
                proprioception = proprioception.to(inputs_embeds.device).to(inputs_embeds.dtype)
                agent_pos_mask = agent_pos_mask.to(inputs_embeds.device).to(inputs_embeds.dtype)
                proprio_embed = self.action_preprocessor.proprioception_proj(
                    proprioception,
                    agent_pos_mask,
                    use_history=proprioception.shape[1] > 1,
                )
                proprioception_mask = input_ids == self.action_token_id_set["propri_token_id"]
                proprio_embed = proprio_embed.to(torch.bfloat16)
                inputs_embeds[proprioception_mask] = proprio_embed.reshape(-1, inputs_embeds.shape[-1])

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # 如果未提供位置 ID，则计算 RoPE 位置 ID
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # 仅在预填充阶段计算一次 RoPE 索引
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
            # 使用先前计算的 rope 增量来获取正确的位置 ID
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # 如果提供了动作块，则准备动作块数据
        if action_chunk is not None:
            action_chunk = action_chunk.to(inputs_embeds.device).to(torch.float32)

        output = {}

        # 为文本和快速模式分割输入序列（扩散模式不需要）
        if predict_mode == "text" or predict_mode == "fast":
            # 查找生成提示令牌：<|im_start|>assistant
            generation_prompt_ids = torch.tensor(
                [151644, 77091], device=input_ids.device, dtype=input_ids.dtype
            )
            matches = (input_ids[0, :-1] == generation_prompt_ids[0]) & (
                input_ids[0, 1:] == generation_prompt_ids[1]
            )

            if matches.any():
                split_pos = torch.nonzero(matches, as_tuple=True)[0][0].item()
                # 提取真实输出令牌（包括换行符）
                gt_output_ids = input_ids[:, split_pos + 3 :]
                # 从输入中移除输出部分，只保留提示
                input_ids = input_ids[:, : split_pos + 3]
                inputs_embeds = inputs_embeds[:, : split_pos + 3, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, : split_pos + 3]
                if labels is not None:
                    labels = labels[:, split_pos + 3 :]
            else:
                raise ValueError(
                    "input_ids does not contain the generation prompt tokens <|im_start|>assistant"
                )

            # 解码输入文本用于输出
            input_text = self.processor.batch_decode(
                input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
            )
            output["input_text"] = input_text

        # 处理文本和快速预测模式：使用自回归生成
        if predict_mode == "text" or predict_mode == "fast":
            # 初始化 MoE 令牌类型用于生成
            moe_token_types = torch.zeros_like(input_ids)
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "moe_token_types": moe_token_types,
                "image_grid_thw": image_grid_thw,
                "dof_mask": dof_mask,
                "agent_pos_mask": agent_pos_mask,
                "proprioception": proprioception,
            }

            predict_output_ids = self.generate(
                **batch,
                max_new_tokens=100,
                eos_token_id=[self.processor.tokenizer.eos_token_id],
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                temperature=(1.0 if not re_generate else 0.7),  # 重新生成时使用更高温度
                do_sample=(False if not re_generate else True),  # 重新生成时启用采样
            )

            # 解码生成的文本和真实文本
            gt_output_text = self.processor.batch_decode(
                gt_output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            predict_output_text = self.processor.batch_decode(
                predict_output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            output["gt_output_text"] = gt_output_text
            output["predict_output_text"] = predict_output_text
            # print("output:", output["predict_output_text"])
            # print("gt_output_text:", output["gt_output_text"])
            # print("++++++++++++++++")

        # 将令牌转换为动作（快速预测模式）
        if predict_mode == "fast":
            action_id = []
            # 从生成的序列中提取动作令牌
            for token_id_i in predict_output_ids[0]:
                if token_id_i.item() >= self.processor.tokenizer.init_kwargs["action_token_start_index"]:
                    action_id.append(
                        token_id_i.item() - self.processor.tokenizer.init_kwargs["action_token_start_index"]
                    )

            predict_action = self.processor.action_processor.decode(
                [action_id], time_horizon=pred_horizon, action_dim=action_dim
            )
            # 处理动作解码错误
            if np.sum(predict_action) == 0:
                print("Error in decoding action, predict_action is None")
                output["predict_action"] = None
            else:
                # 将离散令牌转换为连续动作
                predict_action = torch.tensor(predict_action, device=self.device)
                dof_mask = dof_mask.to(self.device).to(pixel_values.dtype)
                # 暂时移除了反归一化步骤
                predict_action = predict_action[:, :, dof_mask[0, 0, :].bool()]
                output["predict_action"] = predict_action

            # 处理真实动作（如果可用）
            if action_chunk is not None:
                # 应用 DOF 掩码获取真实动作
                # 暂时移除了反归一化步骤
                action_chunk = action_chunk[:, :, dof_mask[0, 0, :].bool()]
                output["gt_action"] = action_chunk
            else:
                output["gt_action"] = None

        # 处理基于扩散的动作预测
        if predict_mode == "diffusion":
            # 从随机噪声开始
            noisy_action = torch.randn(
                size=(batch_size, pred_horizon, action_dim),
                dtype=torch.float32,
                device=inputs_embeds.device,
            )
            dof_mask = dof_mask.to(inputs_embeds.device).to(torch.float32)

            def step(timestep, noisy_action):
                """
                扩散过程的单步去噪函数。

                参数：
                    timestep: 当前扩散时间步
                    noisy_action: 当前带噪动作估计

                返回：
                    torch.Tensor: 预测的干净动作
                """
                action_mask = input_ids == self.action_token_id_set["action_token_id"]
                assert action_mask.any(), "No action token found in input_ids"

                # 为批量处理准备时间步
                timestep = timestep.unsqueeze(0).repeat(noisy_action.shape[0])
                action_embed = self.action_preprocessor.step(
                    timestep=timestep, noisy_action=noisy_action, dof_mask=dof_mask
                )
                action_embed = action_embed.reshape(-1, inputs_embeds.shape[-1])

                # 在赋值前确保 action_embed 具有正确的 dtype 和设备
                action_embed = action_embed.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

                # 创建输入嵌入的临时副本（clone 保留 dtype）
                temp_inputs_embeds = inputs_embeds.clone()
                temp_inputs_embeds[action_mask] = action_embed

                # 通过 Transformer 进行前向传播
                transformer_outputs = self.model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=temp_inputs_embeds,
                    moe_token_types=moe_token_types,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

                # 从隐藏状态中提取动作预测
                hidden_states = transformer_outputs.last_hidden_state
                action_mask = input_ids == self.action_token_id_set["action_token_id"]
                action_hidden_states = hidden_states[action_mask].to(torch.float32)
                pred = self.action_preprocessor.action_proj_back(action_hidden_states)
                return pred.reshape(batch_size, pred_horizon, action_dim)

            # 执行 ODE 积分进行扩散采样
            times = torch.linspace(
                0,
                1,
                num_inference_timesteps + 1,
                device=inputs_embeds.device,
                dtype=torch.float32,
            )
            action_trajectory = odeint(step, noisy_action, times, method="euler")

            # 提取最终预测的动作
            # 暂时移除了反归一化步骤
            predict_action = action_trajectory[-1]
            output["predict_action"] = predict_action

            # 处理真实动作（如果可用）
            # 暂时移除了反归一化步骤
            if action_chunk is not None:
                output["gt_action"] = action_chunk[:, :, dof_mask[0, 0, :].bool()]

        return output

    def forward(self, mode: str | None = None, predict_mode: str | None = "text", **kwargs):
        """
        主前向传播调度器，根据指定模式路由到不同的前向函数。

        模式：
        - 无模式（None）：禁用梯度的训练步骤
        - 'predict'：预测/推理模式
        - 'train'：启用梯度的训练模式
        - 'validate'：禁用梯度的验证模式

        参数：
            mode: 执行模式
            predict_mode: 'predict' 模式的预测模式（"text", "fast", 或 "diffusion"）
            **kwargs: 传递给选定前向函数的额外参数

        返回：
            选定模式对应的模型输出
        """

        if not mode:
            with torch.no_grad():
                return self.train_step_forward(**kwargs)
        elif mode == "predict":
            return self.predict(predict_mode=predict_mode, **kwargs)
        elif mode == "train":
            return self.train_step_forward(use_cache=False, **kwargs)
        elif mode == "validate":
            with torch.no_grad():
                return self.train_step_forward(use_cache=False, **kwargs)
        else:
            raise NotImplementedError("invalid key")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        moe_token_types=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        proprioception=None,
        dof_mask=None,
        agent_pos_mask=None,
        **kwargs,
    ):
        """
        为自回归生成准备输入，支持多模态数据。

        该方法处理输入准备，包括根据缓存位置对输入进行切片、MoE 令牌类型管理
        以及多模态数据处理。视觉输入仅在生成期间需要时选择性传递。

        参数：
            input_ids: 输入令牌 ID
            past_key_values: 先前生成步骤的缓存键值对
            attention_mask: 注意力掩码
            inputs_embeds: 预计算的输入嵌入
            moe_token_types: MoE 路由的令牌类型分配
            cache_position: 当前生成缓存位置
            position_ids: 位置 ID
            use_cache: 是否使用键值缓存
            pixel_values: 图像像素值
            pixel_values_videos: 视频像素值
            image_grid_thw: 图像网格尺寸
            video_grid_thw: 视频网格尺寸
            second_per_grid_ts: 每个网格的时间间隔
            proprioception: 本体感知传感器数据
            dof_mask: 自由度掩码
            agent_pos_mask: 智能体位置掩码
            **kwargs: 其他参数

        返回：
            dict: 为生成步骤准备好的模型输入

        注意：
            这是一个重写方法，处理多模态生成的具体情况：
            - 通过 cache_position 对 input_ids 进行切片，只保留未处理的令牌
            - 处理 input_embeds、生成方法和 GPU 同步的特殊情况
            - 管理视觉输入，避免不必要的前向传播
        """
        # 如果未提供 MoE 令牌类型，则初始化
        if moe_token_types is None:
            moe_token_types = torch.zeros_like(
                input_ids
            )  # FIXME: 处理使用 input_embeds 的情况
        else:
            # 确保 moe_token_types 长度与 input_ids 匹配
            if moe_token_types.shape[1] < input_ids.shape[1]:
                # 计算需要的填充长度
                pad_length = input_ids.shape[1] - moe_token_types.shape[1]
                # 创建默认令牌类型（0）的填充张量
                pad_tensor = torch.zeros(
                    (moe_token_types.shape[0], pad_length),
                    dtype=moe_token_types.dtype,
                    device=moe_token_types.device,
                )
                # 将填充拼接到现有的 moe_token_types
                moe_token_types = torch.cat([moe_token_types, pad_tensor], dim=1)

        # 根据缓存状态和特殊情况处理输入切片
        if past_key_values is not None:
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # 例外4：input_embeds 情况
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
                moe_token_types = moe_token_types[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None or (  # 例外1：提供了 input_embeds
                is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1]
            ):  # 例外3：GPU 同步边缘情况
                input_ids = input_ids[:, -cache_position.shape[0] :]
                moe_token_types = moe_token_types[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # 默认情况（例外2 是无操作）
                cache_pos = cache_position.clone()
                input_ids = input_ids[:, cache_pos]
                moe_token_types = moe_token_types[:, cache_pos]

        # 对于连续步骤（非初始生成），跳过视觉输入
        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # 确定此生成步骤是使用 inputs_embeds 还是 input_ids
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        # 为静态缓存准备 4D 因果注意力掩码
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

        # 组装所有生成所需的模型输入
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "moe_token_types": moe_token_types,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
                "proprioception": proprioception,
                "dof_mask": dof_mask,
                "agent_pos_mask": agent_pos_mask,
            }
        )
        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: torch.LongTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取每个样本的图像和视频数量，用于计算张量分离长度。

        这些参数直接从 input_ids 计算，而不是通过处理器传递，
        以避免接口修改带来的不可预测影响。

        参数：
            input_ids: 形状为 (batch_size, sequence_length) 的输入令牌 ID

        返回：
            tuple:
                - image_nums: 每个样本的图像数量
                - video_nums: 每个样本的视频数量
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        # 查找视觉开始令牌及其后的令牌
        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id

        # 统计视觉开始令牌后的图像和视频数量
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
        """
        为生成过程扩展输入，支持多模态张量。

        这是一个重写方法，支持扩展没有标准批量维度的张量，
        特别是与视觉相关的张量：
        - pixel_values.shape[0] = sum(所有图像样本的序列长度)
        - image_grid_thw.shape[0] = sum(所有样本的图像数量)
        - 视频张量的类似模式

        参数：
            expand_size: 扩展因子（用于束搜索等）
            is_encoder_decoder: 是否使用编码器-解码器架构
            input_ids: 输入令牌 ID
            **model_kwargs: 要扩展的其他模型参数

        返回：
            tuple: (扩展后的 input_ids, 扩展后的 model_kwargs)
        """
        if expand_size == 1:
            return input_ids, model_kwargs

        # 定义需要特殊处理的视觉相关张量键
        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ]

        def _expand_dict_for_generation_visual(dict_to_expand):
            """基于每个样本的图像/视频计数扩展视觉相关张量。"""
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                """按长度分割张量并重复每个样本。"""
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # 将图像分割成样本并计算序列长度
                    samples = torch.split(image_grid_thw, list(image_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # 基于每个样本的图像数量进行扩展
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    # 将视频分割成样本并计算序列长度
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    # 基于每个样本的视频数量进行扩展
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    # 处理列表类型的时间网格数据
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
            """使用 repeat_interleave 扩展标准张量。"""
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # 只有在 input_ids 可用且非空时才扩展视觉输入
        # 如果 input_ids 不可用，则不会使用视觉输入，因此无需扩展
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        # 使用标准 repeat_interleave 扩展 input_ids
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        # 扩展所有其他模型参数
        model_kwargs = _expand_dict_for_generation(model_kwargs)

        # 处理编码器-解码器特定的扩展
        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


class WallXPolicy(PreTrainedPolicy):
    """
    Wall-X 策略，用于跨具身机器人控制。

    集成 Qwen2.5-VL 视觉语言模型与流匹配（Flow Matching）动作预测，
    适用于连续动作空间。
    """

    config_class = WallXConfig  # 关联的配置类
    name = "wall_x"  # 策略名称

    def __init__(self, config: WallXConfig, **kwargs):
        super().__init__(config)
        config.validate_features()  # 验证配置特性
        self.config = config

        # 初始化 wall-x 模型
        self.model = Qwen2_5_VLMoEForAction.from_pretrained(
            pretrained_name_or_path=config.pretrained_name_or_path,
            action_tokenizer_path=config.action_tokenizer_path,
            attn_implementation=config.attn_implementation,
        )
        self.model.to(config.device)  # 移动到指定设备
        self.model.to_bfloat16_for_selected_params()  # 将部分参数转为 bfloat16

        self.reset()  # 重置动作队列

    def reset(self):
        """重置动作队列。"""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),  # 动作队列，最大长度为动作步数
        }

    def get_optim_params(self):
        """获取优化参数。"""
        return self.parameters()

    def preprocess_inputs(
        self,
        batch: dict[str, Any],
    ) -> BatchFeature:
        """
        将 LeRobot 数据集的一个批次转换为 Wall-X 模型输入格式。

        该方法处理一个批次的字典，其中张量的第一维是批次维度。

        参数：
            batch: 包含批次张量的字典：
                - "observation.state": (batch_size, state_dim) 或 (batch_size, n_obs_steps, state_dim)
                - "action": (batch_size, chunk_size, action_dim)
                - "observation.images.<key>": (batch_size, C, H, W)
                - "task": 长度为 batch_size 的字符串列表

        返回：
            包含批次模型输入的 BatchFeature 对象
        """
        use_fast_tokenizer = self.config.use_fast_tokenizer

        # 从状态张量获取批次大小
        batch_size = batch[OBS_STATE].shape[0]

        # ==================== 处理所有样本 ====================
        all_image_inputs = []
        all_texts = []

        # 在批次中查找图像键
        img_keys = [key for key in self.config.image_features if key in batch]

        for i in range(batch_size):
            # 每个样本的视觉预处理
            processed_frames = []
            orig_height, orig_width = None, None
            resized_height, resized_width = None, None

            for key in img_keys:
                current_obs = batch[key][i].clone()  # (C, H, W)
                if current_obs.dim() == 3:
                    current_obs = current_obs.permute(1, 2, 0)  # 转换为 (H, W, C)

                # 将张量转换为 PIL 图像
                img_pil = Image.fromarray((current_obs * 255).to(torch.uint8).cpu().numpy())
                orig_width, orig_height = img_pil.size

                target_size = RESOLUTION
                if target_size != -1:
                    # 根据目标大小调整图像尺寸
                    if orig_width > orig_height:
                        new_width = target_size
                        new_height = int(target_size * orig_height / orig_width)
                    else:
                        new_height = target_size
                        new_width = int(target_size * orig_width / orig_height)
                    img_pil = img_pil.resize((new_width, new_height))

                current_width, current_height = img_pil.size
                # 使用 smart_resize 进行智能调整，确保尺寸是 factor 的倍数
                resized_height, resized_width = smart_resize(
                    current_height,
                    current_width,
                    factor=IMAGE_FACTOR,
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS,
                )
                resized_img = img_pil.resize((resized_width, resized_height))
                processed_frames.append(resized_img)

            all_image_inputs.append(processed_frames)

            # 文本预处理
            task_text = batch["task"][i] if isinstance(batch["task"], list) else batch["task"]
            instruction_info = {"instruction": task_text}

            frame_index = batch["frame_index"][i] if "frame_index" in batch else 0
            # 生成标准文本
            complete_text, _ = get_wallx_normal_text(
                instruction_info,
                self.config.chunk_size,
                frame_index,
                PRIORITY_ORDER,
                img_keys,
                generate_subtask_ratio=GENERATE_SUBTASK_RATIO,
            )

            # 处理接地点
            text = process_grounding_points(
                complete_text, orig_height, orig_width, resized_height, resized_width, MODEL_TYPE
            )
            all_texts.append(text)

        # ==================== 处理智能体位置 ====================
        agent_pos = batch[OBS_STATE]  # (batch_size, state_dim)
        if agent_pos.dim() == 2:
            agent_pos = agent_pos.unsqueeze(1)  # 添加时间维度 (batch_size, 1, state_dim)
        agent_pos_mask = (~torch.isnan(agent_pos)).float()  # 有效位置掩码
        agent_pos = agent_pos.nan_to_num(nan=0.0)  # 将 NaN 替换为 0

        # 将智能体位置填充到固定维度 20
        if agent_pos.shape[-1] != 20:
            pad_size = 20 - agent_pos.shape[-1]
            agent_pos = torch.cat(
                [
                    agent_pos,
                    torch.zeros(agent_pos.shape[0], agent_pos.shape[1], pad_size, device=agent_pos.device),
                ],
                dim=-1,
            )
            agent_pos_mask = torch.cat(
                [
                    agent_pos_mask,
                    torch.zeros(
                        agent_pos_mask.shape[0],
                        agent_pos_mask.shape[1],
                        pad_size,
                        device=agent_pos_mask.device,
                    ),
                ],
                dim=-1,
            )

        # ==================== 处理动作 ====================
        action = batch.get(ACTION)  # (batch_size, chunk_size, action_dim)
        if action is not None:
            if action.dim() == 2:
                action = action.unsqueeze(1)  # 添加时间维度
            dof_mask = (~torch.isnan(action)).float()  # 有效自由度掩码
            action = action.nan_to_num(nan=0.0)  # 将 NaN 替换为 0

            # 将动作填充到固定维度 20
            if action.shape[-1] != 20:
                pad_size = 20 - action.shape[-1]
                action = torch.cat(
                    [action, torch.zeros(action.shape[0], action.shape[1], pad_size, device=action.device)],
                    dim=-1,
                )
                dof_mask = torch.cat(
                    [
                        dof_mask,
                        torch.zeros(dof_mask.shape[0], dof_mask.shape[1], pad_size, device=dof_mask.device),
                    ],
                    dim=-1,
                )
        else:
            # 如果没有动作数据，则创建全有效掩码和零动作
            action_dim = self.config.output_features[ACTION].shape[0]
            dof_mask = torch.cat(
                [
                    torch.ones(
                        batch_size, self.config.chunk_size, action_dim, device=batch[OBS_STATE].device
                    ),
                    torch.zeros(
                        batch_size, self.config.chunk_size, 20 - action_dim, device=batch[OBS_STATE].device
                    ),
                ],
                dim=-1,
            )

        # ==================== 动作令牌替换 ====================
        all_texts = replace_action_token(
            all_texts,
            action,
            self.model.action_tokenizer if use_fast_tokenizer else None,
            dof_mask,
        )

        # ==================== 分词 ====================
        inputs = preprocesser_call(
            processor=self.model.processor,
            text=all_texts,
            images=all_image_inputs,
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=TOKENIZER_MAX_LENGTH,
        )

        # ==================== 添加额外输入 ====================
        action_token_id = self.model.processor.tokenizer.convert_tokens_to_ids("<|action|>")
        moe_token_types = inputs.input_ids == action_token_id  # 标记哪些位置是动作令牌

        inputs["proprioception"] = agent_pos
        inputs["agent_pos_mask"] = agent_pos_mask
        inputs["action_chunk"] = action
        inputs["dof_mask"] = dof_mask
        inputs["moe_token_types"] = moe_token_types
        inputs["frame_index"] = (
            batch["frame_index"]
            if "frame_index" in batch
            else torch.zeros(batch_size, device=batch[OBS_STATE].device)
        )

        # 将所有张量移动到正确的设备
        device = self.config.device
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)

        return inputs

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        使用 Qwen2_5_VLMoEForAction 进行训练前向传播。

        参数：
            batch: 包含 preprocess_inputs() 预处理输入的字典
                   预期键：input_ids, attention_mask, pixel_values, image_grid_thw,
                   proprioception, agent_pos_mask, action_chunk, dof_mask, moe_token_types 等。

        返回：
            tuple: (loss, loss_dict)
        """
        batch = self.preprocess_inputs(
            batch,
        )

               
        # 调用底层模型的前向传播，模式为 "train"
        outputs = self.model(**batch, mode="train")
        # outputs = self.model(**batch, mode="predict", predict_mode="text")

        # 从输出中提取损失
        loss = outputs.loss
        loss_dict = {
            "loss": loss.item() if loss is not None else 0.0,
        }

        if outputs.flow_loss is not None:
            loss_dict["flow_loss"] = outputs.flow_loss.item()
        if outputs.cross_entropy_loss is not None:
            loss_dict["cross_entropy_loss"] = outputs.cross_entropy_loss.item()

        # 如果存在通道损失，则添加到字典
        if outputs.channel_loss_dict is not None:
            for key, value in outputs.channel_loss_dict.items():
                if isinstance(value, torch.Tensor):
                    loss_dict[f"channel_{key}"] = value.item()

        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """评估阶段预测动作块。"""
        self.eval()
        # 填充动作队列（排除动作本身）
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        batch = self.preprocess_inputs(
            batch,
        )

        # 根据配置选择预测模式
        if self.config.prediction_mode == "diffusion":
            output = self.model(
                **batch,
                action_dim=self.config.max_action_dim,
                pred_horizon=self.config.chunk_size,
                mode="predict",
                predict_mode="diffusion",
            )
        elif self.config.prediction_mode == "fast":
            output = self.model(
                **batch,
                action_dim=self.config.output_features[ACTION].shape[0],
                pred_horizon=self.config.chunk_size,
                mode="predict",
                predict_mode="fast",
            )
        else:
            raise NotImplementedError(f"Prediction mode {self.config.prediction_mode} not implemented")

        # 从输出字典中提取动作张量
        actions = output["predict_action"]

        # 去除填充，恢复实际动作维度
        action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """为环境执行选择单个动作。"""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # 使用动作队列
        if len(self._queues[ACTION]) == 0:
            # 预测一个动作块
            actions = self.predict_action_chunk(batch)
            # 将动作块按时间步拆分并扩展到队列中
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        # 从队列中弹出一个动作
        return self._queues[ACTION].popleft()