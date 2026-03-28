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
Wall-X Utility Functions.

Contains data processing utilities, text formatting functions, and helper classes
for the Wall-X cross-embodiment robotic control model.
"""

# ============================================================================
# 标准库导入
# ============================================================================
import random  # 导入random模块，用于随机数生成和概率采样
import re  # 导入re模块，用于正则表达式操作，处理文本模式匹配
from collections import OrderedDict  # 导入OrderedDict，用于保持字典键值对的插入顺序
from dataclasses import dataclass, field  # 导入dataclass装饰器和field，用于创建数据类
from typing import Any  # 导入Any类型，用于灵活的类型注解

# ============================================================================
# 第三方库导入
# ============================================================================
import torch  # 导入PyTorch，用于张量操作和深度学习计算
from transformers import BatchFeature  # 导入BatchFeature，用于批量特征的管理

# ============================================================================
# 本地模块导入
# ============================================================================
from lerobot.policies.wall_x.constant import (
    CAMERA_NAME_MAPPING,  # 相机名称映射字典，将内部相机键名映射为可读名称
)
from lerobot.utils.constants import OBS_IMAGES  # 观测图像键名常量，用于标识图像观测


# ============================================================================
# X2R数据预处理配置类
# ============================================================================
@dataclass  # 使用dataclass装饰器自动生成__init__等方法
class X2RDataProcessingConfig:
    """X2R数据预处理流程的配置类。
    
    该类包含了处理机器人数据所需的所有参数，包括：
    - 相机映射配置
    - 触觉传感器配置
    - 动作预测配置
    - 各种处理选项
    
    Attributes:
        predict_action_keys: 需要预测的动作键名列表
        obs_action_keys: 观测到的动作键名列表
        resolution: 不同视图的图像分辨率设置
        train_test_split: 训练集/测试集划分比例
        split_seed: 数据集划分的随机种子
        priority_order: 指令优先级顺序字典
        model_type: 模型类型标识
        max_pixels: 最大像素数限制
        min_pixels: 最小像素数限制
        image_factor: 图像处理因子
        generate_subtask_ratio: 生成子任务而非动作的概率
    """

    # 动作预测配置：需要预测的动作键名列表，默认为空列表
    predict_action_keys: list[str] = field(default_factory=list)
    # 动作观测配置：观测到的动作键名列表，默认为空列表
    obs_action_keys: list[str] = field(default_factory=list)

    # 不同视图的图像分辨率设置
    # 使用lambda工厂函数创建默认字典，避免可变默认参数问题
    resolution: dict[str, int] = field(
        default_factory=lambda: {
            "face_view": -1,          # 面部视图：-1表示使用原始分辨率，不进行缩放
            "left_wrist_view": 128,   # 左手腕视图：缩放至128x128像素
            "right_wrist_view": 128,  # 右手腕视图：缩放至128x128像素
        }
    )

    # 数据集分割参数：训练集比例，默认为90%
    train_test_split: float = 0.9
    # 数据集分割的随机种子，用于保证可复现性
    split_seed: int = 42

    # 指令处理：优先级顺序字典，用于采样不同的指令字段
    # 可选类型，默认为None表示使用默认优先级
    priority_order: dict[str, float] | None = None

    # 视觉模型相关参数
    model_type: str = "qwen2_5"                    # 模型类型，默认为qwen2_5
    max_pixels: int = 16384 * 28 * 28              # 最大像素数：16384 * 784 = 约12.8M像素
    min_pixels: int = 4 * 28 * 28                  # 最小像素数：4 * 784 = 3136像素
    image_factor: int = 28                         # 图像因子，用于图像分块处理

    # 生成子任务的概率比率，默认为0.0（不生成子任务）
    generate_subtask_ratio: float = 0.0

    def __post_init__(self):
        """初始化后处理方法，用于验证配置参数的有效性。
        
        Raises:
            ValueError: 当train_test_split不在(0,1)范围内时抛出异常
        """
        # 验证训练/测试分割比例是否在有效范围内（0到1之间，不包含边界）
        if not 0 < self.train_test_split < 1:
            raise ValueError(f"train_test_split must be between 0 and 1, got {self.train_test_split}")

    def as_dict(self) -> dict:
        """将配置对象转换为字典格式。
        
        Returns:
            dict: 配置对象的字典表示，包含所有实例属性
        """
        return self.__dict__  # 返回实例的__dict__字典，包含所有属性

    def update(self, **kwargs) -> "X2RDataProcessingConfig":
        """动态更新配置参数。
        
        Args:
            **kwargs: 需要更新的键值对参数
            
        Returns:
            X2RDataProcessingConfig: 更新后的配置实例，支持链式调用
            
        Raises:
            ValueError: 当尝试更新不存在的配置参数时抛出异常
        """
        # 遍历所有传入的关键字参数
        for key, value in kwargs.items():
            # 检查配置对象是否包含该属性
            if hasattr(self, key):
                setattr(self, key, value)  # 更新属性值
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return self  # 返回自身以支持链式调用


# ============================================================================
# 统一预处理函数
# ============================================================================
def preprocesser_call(
    processor,  # 多模态处理器，包含tokenizer和image_processor
    images: list | Any | None = None,      # 输入图像（PIL Image、numpy数组或torch张量）
    text: str | list[str] | None = None,   # 需要分词化的文本或文本列表
    videos: list | Any | None = None,      # 输入视频（numpy数组或torch张量，形状：[T, C, H, W]）
    padding: bool | str = False,           # 是否填充序列到相同长度
    truncation: bool | None = None,        # 是否截断超过max_length的序列
    max_length: int | None = None,         # 截断/填充的最大长度
    return_tensors: str = "pt",            # 返回张量的格式：'pt'表示PyTorch张量，'np'表示NumPy数组
) -> BatchFeature:
    """Wall-X模型的统一预处理函数，处理文本、图像和视频输入。
    
    该函数将多种模态的输入处理为适合多模态Transformer模型的格式，包括：
    - 文本分词化和特殊令牌处理
    - 图像/视频通过图像处理器进行处理
    - 注意力掩码和标签生成
    - 填充和截断处理
    
    Args:
        processor: 多模态处理器，包含tokenizer和image_processor组件
        images: 输入图像，支持PIL Image、numpy数组或torch张量
        text: 需要分词化的文本或文本列表
        videos: 输入视频，支持numpy数组或torch张量
        padding: 是否填充序列到相同长度
        truncation: 是否截断超过max_length的序列
        max_length: 序列的最大长度
        return_tensors: 返回张量的格式（'pt'、'np'等）
        
    Returns:
        BatchFeature: 包含处理后的输入数据的批量特征对象，包括：
            - input_ids: 分词化后的token IDs
            - attention_mask: 文本的注意力掩码
            - pixel_values: 处理后的图像像素值
            - pixel_values_videos: 处理后的视频帧像素值
            - image_grid_thw: 图像的网格维度（T, H, W），用于LLM
            - video_grid_thw: 视频的网格维度（T, H, W），用于LLM
            - labels: 训练标签，包含掩码处理
    """
    # ========================================================================
    # 处理图像输入
    # ========================================================================
    if images is not None and len(images) > 0:
        # 使用处理器的图像处理器处理图像
        # 返回包含pixel_values、image_grid_thw等字段的字典
        image_inputs = processor.image_processor(images=images, return_tensors=return_tensors)
        # 获取图像网格尺寸 [num_images, T, H, W]，其中T通常为1
        image_grid_thw = image_inputs["image_grid_thw"]
    else:
        # 没有图像输入时初始化为空字典和None
        image_inputs = {}
        image_grid_thw = None

    # ========================================================================
    # 处理视频输入
    # ========================================================================
    if videos is not None:
        # 使用图像处理器处理视频（视频被视为多帧图像的序列）
        videos_inputs = processor.image_processor(videos=videos, return_tensors=return_tensors)
        # 获取视频网格尺寸 [num_videos, T, H, W]
        video_grid_thw = videos_inputs["video_grid_thw"]
    else:
        # 没有视频输入时初始化为空字典和None
        videos_inputs = {}
        video_grid_thw = None

    # ========================================================================
    # 处理文本输入
    # ========================================================================
    # 确保文本输入是列表格式，方便统一处理
    if not isinstance(text, list):
        text = [text]  # 将单个字符串转换为包含一个元素的列表

    # 处理文本中的图像占位符令牌 <|image_pad|>
    # 根据实际的图像网格尺寸替换为正确数量的令牌
    if image_grid_thw is not None:
        # 计算合并大小：merge_size是图像分块合并的尺寸，其平方表示每个图像块的令牌数
        merge_length = processor.image_processor.merge_size ** 2
        index = 0  # 当前处理的图像索引
        # 遍历每个文本样本
        for i in range(len(text)):
            # 循环替换文本中的所有图像占位符
            while "<|image_pad|>" in text[i]:
                # 添加边界检查，避免索引溢出
                if index >= len(image_grid_thw):
                    # 打印警告信息，提示占位符数量超过实际图像数量
                    print(
                        f"Warning: Number of image placeholders ({index + 1}) "
                        f"exceeds actual images ({len(image_grid_thw)}), "
                        f"skipping remaining placeholder processing"
                    )
                    break
                # 计算当前图像需要的实际令牌数量
                # image_grid_thw[index].prod() = T * H * W
                # token_count = (T * H * W) / merge_size^2
                token_count = image_grid_thw[index].prod() // merge_length
                # 将第一个图像占位符替换为对应数量的临时占位符
                text[i] = text[i].replace("<|image_pad|>", "<|placeholder|>" * token_count, 1)
                index += 1
            # 将所有临时占位符替换回图像占位符
            text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

    # 处理文本中的视频占位符令牌 <|video_pad|>
    if video_grid_thw is not None:
        # 计算合并大小
        merge_length = processor.image_processor.merge_size ** 2
        index = 0  # 当前处理的视频索引
        for i in range(len(text)):
            # 循环替换文本中的所有视频占位符
            while "<|video_pad|>" in text[i]:
                # 计算当前视频需要的实际令牌数量
                token_count = video_grid_thw[index].prod() // merge_length
                # 将第一个视频占位符替换为对应数量的临时占位符
                text[i] = text[i].replace("<|video_pad|>", "<|placeholder|>" * token_count, 1)
                index += 1
            # 将所有临时占位符替换回视频占位符
            text[i] = text[i].replace("<|placeholder|>", "<|video_pad|>")

    # ========================================================================
    # 文本分词化
    # ========================================================================
    # 使用处理器的tokenizer对文本进行分词化
    text_inputs = processor.tokenizer(
        text,
        return_tensors=return_tensors,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
    )

    # ========================================================================
    # 生成训练标签（支持多轮对话，仅保留助手的响应部分的损失）
    # ========================================================================
    # 获取填充令牌ID，如果没有则使用结束令牌ID
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    # 创建标签张量，初始化为-100（-100在CrossEntropyLoss中会被忽略）
    labels = torch.full_like(text_inputs.input_ids, -100)
    
    # 定义助手的起始标记和结束标记
    assistant_marker = "<|im_start|>assistant\n"  # 助手响应的起始标记
    im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")  # 对话结束标记的ID
    # 获取助手起始标记的token IDs（不添加特殊标记）
    assistant_tokens = processor.tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids

    # 为每个样本生成标签区域
    for i in range(len(text)):
        assistant_regions = []  # 存储助手响应区域的起始和结束位置
        # 按助手起始标记分割文本
        parts = text[i].split(assistant_marker)

        # 计算左侧填充令牌的数量（用于处理填充后的序列）
        num_left_pads = 0
        for token_id in text_inputs.input_ids[i]:
            if token_id == pad_token_id:
                num_left_pads += 1
            else:
                break
        current_pos = num_left_pads  # 当前位置从填充结束开始

        # 遍历每个部分
        for j, part in enumerate(parts):
            # 对当前部分进行分词
            part_tokens = processor.tokenizer(part, add_special_tokens=False).input_ids
            
            if j == 0:
                # 第一部分是系统提示或用户问题，所有标签都设为-100（不计算损失）
                current_pos += len(part_tokens)
                continue

            # 从第二部分开始，每个部分以助手响应开头
            # 查找当前部分的结束标记位置
            for k in range(current_pos + 1, len(text_inputs.input_ids[i])):
                if text_inputs.input_ids[i][k] == im_end_token_id:
                    # 记录助手响应区域：[起始位置, 结束位置+2)
                    # 起始位置 = 当前位置 + 助手起始标记长度
                    # 结束位置 = 找到的结束标记位置 + 2（包含结束标记）
                    assistant_regions.append((current_pos + len(assistant_tokens), k + 2))
                    break
            # 更新当前位置，加上当前部分的令牌数和分隔符
            current_pos += len(part_tokens) + 3

        # 为助手响应区域设置标签（保留这些位置的原始token ID）
        for start, end in assistant_regions:
            labels[i][start:end] = text_inputs.input_ids[i][start:end]

    # ========================================================================
    # 掩码特殊动作令牌
    # ========================================================================
    # 获取特殊令牌的ID
    action_token_id = processor.tokenizer.encode("<|action|>")[0]      # 动作令牌
    propri_token_id = processor.tokenizer.encode("<|propri|>")[0]     # 本体感知令牌
    
    # 将这些特殊令牌的标签设为-100，使其不参与损失计算
    labels[labels == action_token_id] = -100
    labels[labels == propri_token_id] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # 如果存在有效的标签（不是全为-100），则添加到text_inputs中
    if (labels != -100).any().item():
        text_inputs["labels"] = labels
    else:
        # 否则设为None，跳过交叉熵损失计算
        text_inputs["labels"] = None

    # 合并所有输入并返回BatchFeature对象
    return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})


# ============================================================================
# 处理grounding点坐标
# ============================================================================
def process_grounding_points(
    text: str,
    orig_height: int,
    orig_width: int,
    resized_height: int,
    resized_width: int,
    model_type: str,
) -> str:
    """根据图像缩放调整文本中的grounding点坐标。
    
    该函数处理<point>标签中的坐标值，根据图像尺寸的缩放比例调整坐标，
    支持两种模型类型的坐标表示方式：
    - qwen2_5: 使用像素坐标（0到resized_width-1）
    - qwen2: 使用归一化坐标（0到1000，表示为整数）
    
    Args:
        text: 包含<point>标签的输入文本
        orig_height: 原始图像高度
        orig_width: 原始图像宽度
        resized_height: 缩放后的图像高度
        resized_width: 缩放后的图像宽度
        model_type: 模型类型，用于确定坐标处理方式（'qwen2'或'qwen2_5'）
        
    Returns:
        str: 坐标值调整后的文本
        
    Raises:
        ValueError: 当model_type不支持时抛出异常
    """
    # 正则表达式模式，匹配<point>标签及其内容
    # 使用非贪婪匹配(.*?)来捕获标签内的所有内容
    point_pattern = re.compile(r"<point>(.*?)</point>")

    def process_match(match):
        """处理单个point匹配，调整坐标值。
        
        Args:
            match: 正则表达式匹配对象，包含完整的point标签
            
        Returns:
            str: 处理后的point标签字符串
        """
        # 提取标签内的坐标字符串
        coords_str = match.group(1)
        try:
            # 从字符串中提取所有数字，转换为整数列表
            # 支持2个坐标（点）或4个坐标（矩形框）
            coords = list(map(int, re.findall(r"\d+", coords_str)))

            # 计算缩放因子
            scale_w = resized_width / orig_width   # 宽度缩放比例
            scale_h = resized_height / orig_height # 高度缩放比例

            # 处理2个坐标的情况（点坐标）
            if len(coords) == 2:
                x, y = coords
                if model_type == "qwen2_5":
                    # Qwen2.5使用像素坐标，需要四舍五入到整数
                    new_x = max(0, min(round(x * scale_w), resized_width - 1))
                    new_y = max(0, min(round(y * scale_h), resized_height - 1))
                elif model_type == "qwen2":
                    # Qwen2归一化到[0, 1000)范围
                    new_x = max(0, min(999.999, (x / orig_width) * 1000))
                    new_y = max(0, min(999.999, (y / orig_height) * 1000))
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                coords = [new_x, new_y]

            # 处理4个坐标的情况（矩形框坐标）
            elif len(coords) == 4:
                x1, y1, x2, y2 = coords
                if model_type == "qwen2_5":
                    new_x1 = max(0, min(round(x1 * scale_w), resized_width - 1))
                    new_y1 = max(0, min(round(y1 * scale_h), resized_height - 1))
                    new_x2 = max(0, min(round(x2 * scale_w), resized_width - 1))
                    new_y2 = max(0, min(round(y2 * scale_h), resized_height - 1))
                elif model_type == "qwen2":
                    new_x1 = max(0, min(999.999, (x1 / orig_width) * 1000))
                    new_y1 = max(0, min(999.999, (y1 / orig_height) * 1000))
                    new_x2 = max(0, min(999.999, (x2 / orig_width) * 1000))
                    new_y2 = max(0, min(999.999, (y2 / orig_height) * 1000))
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                coords = [new_x1, new_y1, new_x2, new_y2]

            # 返回处理后的point标签，坐标使用逗号分隔
            return f"<point>[{', '.join(map(str, coords))}]</point>"

        except (ValueError, TypeError):
            # 如果处理失败（如坐标格式错误），返回原始内容
            return match.group(0)

    # 使用sub方法替换所有匹配的point标签
    processed_text = point_pattern.sub(process_match, text)
    return processed_text


# ============================================================================
# 提取帧级指令
# ============================================================================
def get_frame_instruction(
    instruction_info: dict[str, Any],
    frame_idx: int | None = None,
    truncate_keys: list[str] | None = None,
) -> tuple[dict[str, Any], int | None]:
    """从指令字典中提取当前帧特定的指令。
    
    指令信息可能包含帧范围特定的指令，格式为：
        {
            "subtask_generation": {
                "0 100": "do task A",
                "100 200": "do task B"
            },
            "instruction": "base instruction"
        }
    
    Args:
        instruction_info: 包含指令组件的字典
        frame_idx: 当前帧索引
        truncate_keys: 触发截断的键列表，当找到这些键时会返回分割结束帧
        
    Returns:
        Tuple[dict, int|None]: 
            - frame_instruction_dict: 当前帧的指令字典
            - split_end_frame: 截断结束帧索引（如果适用），否则为None
    """
    # 默认的截断键列表
    if truncate_keys is None:
        truncate_keys = [
            "subtask_generation",      # 子任务生成
            "distribute",              # 分发指令
            "subtask_generation_zh",   # 中文子任务生成
            "distribute_zh",           # 中文分发指令
        ]

    instruction_for_frame = {}  # 存储当前帧的指令
    split_end = None            # 截断结束帧

    # 遍历所有指令字段
    for key, value in instruction_info.items():
        if isinstance(value, dict):
            # 处理帧范围特定的指令
            # 格式：{"start_frame end_frame": instruction_content}
            for frame_range, frame_instruction in value.items():
                # 解析帧范围字符串，获取起始和结束帧
                start_frame, end_frame = map(int, frame_range.split(" "))
                # 检查当前帧是否在范围内
                if start_frame <= frame_idx < end_frame or (start_frame == frame_idx):
                    instruction_for_frame[key] = frame_instruction
                    # 如果当前键在截断键列表中且尚未设置截断帧，记录截断结束帧
                    if truncate_keys is not None and split_end is None and key in truncate_keys:
                        split_end = end_frame + 1
                    break
        else:
            # 非字典类型的指令直接复制
            instruction_for_frame[key] = value

    return instruction_for_frame, split_end


# ============================================================================
# 获取任务指令
# ============================================================================
def get_task_instruction(
    frame_instruction_info: dict[str, Any], 
    priority_order: OrderedDict | None = None
) -> str:
    """从可用的指令字段中使用优先级采样构建任务指令。
    
    根据priority_order中定义的优先级顺序和概率，从多个指令字段中选择一个作为任务指令。
    这允许在不同类型的指令之间进行采样，增加训练数据的多样性。
    
    Args:
        frame_instruction_info: 包含指令字段的字典
        priority_order: OrderedDict，指定每个字段的采样概率，按优先级排序
        
    Returns:
        str: 组合后的指令字符串
    """
    # 默认的优先级设置
    default_priority_order = OrderedDict(
        {
            "subtask_generation": 0.25,      # 子任务生成指令
            "subtask_generation_zh": 0.25,   # 中文子任务生成指令
            "distribute": 0.25,              # 分发指令
            "distribute_zh": 0.25,           # 中文分发指令
        }
    )

    # 使用传入的优先级顺序或默认值
    if priority_order is not None:
        priority_order = OrderedDict(priority_order)
    else:
        priority_order = default_priority_order

    got_instruction = False      # 是否已获取到指令
    task_instruction = ""        # 最终的任务指令

    # 按优先级顺序采样指令组件
    for key, prob in priority_order.items():
        # 检查当前键是否存在且不为空
        if key in frame_instruction_info and frame_instruction_info[key] != "":
            # 如果已经获取到指令，根据概率决定是否替换
            if got_instruction:
                # 以概率prob跳过当前指令，保留之前的指令
                if random.random() >= prob:
                    continue

            # 添加当前指令（前面加换行符）
            task_instruction += f"\n{frame_instruction_info[key]}"
            got_instruction = True
            break  # 找到指令后立即停止

    # 回退：如果没有找到优先级指令，使用基础指令
    if not got_instruction:
        task_instruction = frame_instruction_info.get("instruction", "")

    return task_instruction


# ============================================================================
# 构建Wall-X模型的完整提示文本
# ============================================================================
def get_wallx_normal_text(
    instruction_info: dict[str, Any],
    action_chunk_size: int,
    frame_idx: int,
    priority_order: OrderedDict | None = None,
    img_keys: list[str] | None = None,
    generate_subtask_ratio: float = 0.0,
) -> tuple[str, bool]:
    """为Wall-X模型构建完整的多模态提示文本。
    
    该函数使用特殊令牌格式化输入，包括：
    - 系统消息
    - 用户观测（带图像占位符）
    - 任务指令
    - 本体感知提示
    - 助手响应（带动作令牌）
    
    支持两种输出模式：
    1. 动作预测模式：输出动作令牌序列
    2. 子任务生成模式：输出自然语言子任务描述
    
    Args:
        instruction_info: 包含指令组件的字典
        action_chunk_size: 需要生成的动作令牌数量
        frame_idx: 当前帧索引
        priority_order: 指令采样的优先级顺序
        img_keys: 图像键名列表
        generate_subtask_ratio: 生成子任务而非动作的概率（0.0-1.0）
        
    Returns:
        Tuple[str, bool]: 
            - formatted_prompt_text: 格式化后的提示文本
            - is_subtask_generation: 是否为子任务生成模式
    """
    # ========================================================================
    # 定义特殊令牌
    # ========================================================================
    role_start_symbol = "<|im_start|>"          # 角色起始标记
    role_end_symbol = "<|im_end|>"              # 角色结束标记
    vision_start_symbol = "<|vision_start|>"    # 视觉内容起始标记
    vision_end_symbol = "<|vision_end|>"        # 视觉内容结束标记
    image_pad_symbol = "<|image_pad|>"          # 图像占位符
    propri_symbol = "<|propri|>"                # 本体感知占位符
    action_symbol = "<|action|>"                # 动作占位符
    action_fast_symbol = "<|action_fast|>"      # 快速动作占位符

    # ========================================================================
    # 构建系统提示
    # ========================================================================
    prologue = f"{role_start_symbol}system\nYou are a helpful assistant.{role_end_symbol}\n"

    # ========================================================================
    # 构建用户请求（包含观测信息）
    # ========================================================================
    user_request = f"{role_start_symbol}user\nObservation:"
    
    # 处理图像键名，添加视觉占位符
    if img_keys:
        # 将内部图像键名映射为可读的相机名称
        img_keys = img_key_mapping(img_keys)
        for key in img_keys:
            # 每个图像视图添加对应的视觉占位符
            user_request += f" {key}: {vision_start_symbol}{image_pad_symbol}{vision_end_symbol}"
    user_request += "\nInstruction:"

    # ========================================================================
    # 获取当前帧的指令
    # ========================================================================
    frame_instruction_info, _ = get_frame_instruction(instruction_info, frame_idx=frame_idx)

    # ========================================================================
    # 决定生成模式：子任务生成 或 动作预测
    # ========================================================================
    generate_subtask = False
    priority_keys = ["subtask_generation", "distribute"]  # 优先级键名

    # 检查是否应该生成子任务
    if (
        bool(set(frame_instruction_info.keys()) & set(priority_keys))  # 存在优先级指令
        and random.random() < generate_subtask_ratio                  # 随机采样
    ):
    # 随机采样 
        # ================================================================
        # 子任务生成模式（类似VQA任务）
        # ================================================================
        instruction = frame_instruction_info.get("instruction", "")
        text_prompt = "\nPredict the next action in language.\n"
        user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"

        # 从优先级键中查找输出指令
        for key in priority_keys:
            if key in frame_instruction_info:
                output_instruction = frame_instruction_info[key]
                break

        # 构建助手输出
        assistant_output = f"{role_start_symbol}assistant\n{output_instruction}\n{role_end_symbol}"
        
        # 调试输出
        print("assistant_output:", assistant_output)
        print("generate_subtask_ratio:", generate_subtask_ratio)
        generate_subtask = True
    else:
        # ================================================================
        # 动作预测模式
        # ================================================================
        # 获取任务指令（带优先级采样）
        instruction = get_task_instruction(frame_instruction_info, priority_order=priority_order)
        text_prompt = f"\nPredict the next action in robot action.\nProprioception: {propri_symbol}\n"
        user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
        # 构建助手输出，包含动作占位符
        assistant_output = f"{role_start_symbol}assistant\n{action_fast_symbol}{role_end_symbol}\n{action_symbol * action_chunk_size}"

    # 组合完整文本
    complete_text = prologue + user_message + assistant_output
    return complete_text, generate_subtask


# ============================================================================
# 图像键名映射
# ============================================================================
def img_key_mapping(img_keys: list[str]) -> list[str]:
    """将内部图像键名映射为可读的相机名称。
    
    该函数处理图像键名，使其在提示文本中更具可读性：
    - 移除观测前缀
    - 使用预定义的相机名称映射
    - 格式化视图名称
    
    Args:
        img_keys: 图像键名列表
        
    Returns:
        list[str]: 处理后的相机名称列表
    """
    processed_img_keys = []
    for key in img_keys:
        # 移除观测图像的前缀（如"observation.images."）
        key = key.replace(OBS_IMAGES + ".", "")
        
        # 使用预定义的映射表
        if key in CAMERA_NAME_MAPPING:
            key = CAMERA_NAME_MAPPING[key]
        else:
            # 处理未映射的键名
            if "view" in key:
                # 将下划线替换为空格，如"left_wrist_view" -> "left wrist view"
                key = key.replace("_", " ")
            else:
                # 添加" view"后缀
                key = key + " view"
        processed_img_keys.append(key)
    return processed_img_keys


# ============================================================================
# 动作转令牌
# ============================================================================
def get_action_tokens(normalized_actions: torch.Tensor | list, action_tokenizer) -> list[list[str]]:
    """将归一化的动作转换为动作令牌字符串。
    
    Args:
        normalized_actions: 归一化的动作数组/张量，形状为[batch_size, action_dim]
        action_tokenizer: 用于将动作转换为令牌的tokenizer
        
    Returns:
        list[list[str]]: 每个样本的动作令牌字符串列表
    """
    # 如果是PyTorch张量，转换为NumPy数组
    if isinstance(normalized_actions, torch.Tensor):
        normalized_actions = normalized_actions.cpu().numpy()

    all_action_tokens = []
    for i in range(len(normalized_actions)):
        # 确保每个样本是NumPy数组
        if isinstance(normalized_actions[i], torch.Tensor):
            normalized_actions[i] = normalized_actions[i].cpu().numpy()

        # 使用action_tokenizer将动作转换为令牌ID
        token_id = action_tokenizer(normalized_actions[i])
        # 生成令牌字符串，格式为"<|action_token_{j}|>"
        action_tokens = [f"<|action_token_{j}|>" for j in token_id[0]]
        all_action_tokens.append(action_tokens)

    return all_action_tokens


# ============================================================================
# 填充动作令牌字符串
# ============================================================================
def pad_action_token_strs(
    actions_token_lists: list[list[str]],
    pad_token: str = "<|endoftext|>",  # nosec B107
) -> list[str]:
    """将动作令牌列表填充到相同长度并连接为字符串。
    
    Args:
        actions_token_lists: 每个样本的动作令牌列表
        pad_token: 用于填充的令牌
        
    Returns:
        list[str]: 填充后的动作令牌字符串列表
    """
    # 找出最大长度
    max_len = max(len(tokens) for tokens in actions_token_lists)
    padded_action_strs = []

    for tokens in actions_token_lists:
        # 添加结束标记和填充令牌
        padded_tokens = tokens + ["<|im_end|>\n"] + [pad_token] * (max_len - len(tokens))
        padded_action_strs.append("".join(padded_tokens))

    return padded_action_strs


# ============================================================================
# 替换文本中的动作占位符
# ============================================================================
def replace_action_token(
    text: list[str],
    norm_action: torch.Tensor | None,
    action_tokenizer,
    dof_masks: torch.Tensor | None = None,
) -> list[str]:
    """替换文本中的动作占位符为实际的动作令牌。
    
    该函数将文本中的<|action_fast|>占位符替换为具体的动作令牌序列，
    并移除<|action|>占位符。
    
    Args:
        text: 包含动作占位符的文本列表
        norm_action: 归一化的动作张量，形状为[batch_size, chunk_size, action_dim]
        action_tokenizer: 动作tokenizer，用于将数值动作转换为令牌
        dof_masks: 自由度掩码，用于选择哪些动作维度有效
        
    Returns:
        list[str]: 替换后的文本列表
    """
    if action_tokenizer is not None and norm_action is not None:
        # ================================================================
        # 根据自由度掩码提取有效动作维度
        # ================================================================
        # norm_action形状: [batch_size, chunk_size, action_dim]
        # dof_masks形状: [batch_size, action_dim]，布尔类型
        # 提取前32个时间步的动作，并根据dof_mask选择有效维度
        norm_action = [action[:32, dof_masks[i, 0].bool()] for i, action in enumerate(norm_action)]

        # ================================================================
        # 转换为动作令牌并填充
        # ================================================================
        # 获取每个样本的动作令牌列表
        actions_fast_tokens = get_action_tokens(norm_action, action_tokenizer)
        # 填充到相同长度并连接为字符串
        actions_fast_token_strs = pad_action_token_strs(actions_fast_tokens)

        # ================================================================
        # 替换文本中的动作占位符
        # ================================================================
        actions_fast_token_idx = 0
        for i in range(len(text)):
            if "<|action_fast|>" in text[i]:
                # 将<|action_fast|><|im_end|>\n替换为实际的动作令牌字符串
                text[i] = text[i].replace(
                    "<|action_fast|><|im_end|>\n",
                    actions_fast_token_strs[actions_fast_token_idx],
                )
                actions_fast_token_idx += 1

        # 移除剩余的<|action|>占位符
        text = [t.replace("<|action|>", "") for t in text]
    else:
        # 没有action_tokenizer时，直接移除动作占位符
        text = [t.replace("<|action_fast|><|im_end|>\n", "") for t in text]

    return text