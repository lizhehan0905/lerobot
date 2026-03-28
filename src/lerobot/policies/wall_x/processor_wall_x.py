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

# 从typing模块导入Any类型，用于类型注解
from typing import Any

# 导入torch，用于PyTorch操作
import torch

# 从lerobot.configs.types导入管道特征类型和策略特征类
from lerobot.configs.types import PipelineFeatureType, PolicyFeature

# 从当前包导入WallXConfig配置类
from lerobot.policies.wall_x.configuration_wall_x import WallXConfig

# 从lerobot.processor导入各种处理器步骤类
from lerobot.processor import (
    AddBatchDimensionProcessorStep,  # 添加批次维度的处理器步骤
    ComplementaryDataProcessorStep,  # 补充数据处理器步骤
    DeviceProcessorStep,  # 设备处理器步骤
    NormalizerProcessorStep,  # 归一化处理器步骤
    PolicyAction,  # 策略动作类
    PolicyProcessorPipeline,  # 策略处理器管道类
    ProcessorStepRegistry,  # 处理器步骤注册表
    RenameObservationsProcessorStep,  # 重命名观测处理器步骤
    UnnormalizerProcessorStep,  # 反归一化处理器步骤
)

# 从lerobot.processor.converters导入策略动作和状态转换函数
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

# 从lerobot.utils.constants导入策略处理器默认名称常量
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


# 创建Wall-X策略的预处理和后处理处理器的工厂函数
def make_wall_x_pre_post_processors(
    config: WallXConfig,  # Wall-X策略的配置对象
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,  # 用于归一化的数据集统计信息
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],  # 预处理管道类型
    PolicyProcessorPipeline[PolicyAction, PolicyAction],  # 后处理管道类型
]:
    """
    Constructs pre-processor and post-processor pipelines for the Wall-X policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations
    2. Adding a batch dimension
    4. Normalizing input and output features based on dataset statistics
    5. Moving all data to the specified device

    The post-processing pipeline handles the model's output by:
    1. Unnormalizing the output actions to their original scale
    2. Moving data to the CPU

    Args:
        config: The configuration object for the Wall-X policy
        dataset_stats: A dictionary of statistics for normalization

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines
    """

    # 预处理步骤列表
    input_steps = [
        # 重命名观测步骤（当前为空映射）
        RenameObservationsProcessorStep(rename_map={}),
        # 添加批次维度步骤
        AddBatchDimensionProcessorStep(),
        # Wall-X任务处理器步骤，用于处理任务描述
        WallXTaskProcessor(),  # Process task description
        # 归一化处理器步骤，根据数据集统计信息归一化输入和输出特征
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},  # 输入和输出特征
            norm_map=config.normalization_mapping,  # 归一化映射
            stats=dataset_stats,  # 数据集统计信息
        ),
        # 设备处理器步骤，将数据移动到指定设备
        DeviceProcessorStep(device=config.device),
    ]

    # 后处理步骤列表
    output_steps = [
        # 反归一化处理器步骤，将输出动作恢复到原始尺度
        UnnormalizerProcessorStep(
            features=config.output_features,  # 输出特征
            norm_map=config.normalization_mapping,  # 归一化映射
            stats=dataset_stats,  # 数据集统计信息
        ),
        # 设备处理器步骤，将数据移动到CPU
        DeviceProcessorStep(device="cpu"),
    ]

    # 返回预处理和后处理管道
    return (
        # 预处理管道：输入字典到输出字典
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,  # 预处理步骤
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,  # 预处理管道名称
        ),
        # 后处理管道：策略动作到策略动作
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,  # 后处理步骤
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,  # 后处理管道名称
            to_transition=policy_action_to_transition,  # 转换为状态转换函数
            to_output=transition_to_policy_action,  # 转换为策略动作函数
        ),
    )


# 使用处理器步骤注册表注册WallXTaskProcessor类
@ProcessorStepRegistry.register(name="wall_x_task_processor")
# WallXTaskProcessor类，继承自ComplementaryDataProcessorStep
class WallXTaskProcessor(ComplementaryDataProcessorStep):
    """
    A processor step that ensures the task description is properly formatted for Wall-X.

    This step handles task preprocessing similar to Qwen-VL requirements.
    """

    # 处理补充数据的方法
    def complementary_data(self, complementary_data):
        # 检查补充数据中是否有"task"键
        if "task" not in complementary_data:
            return complementary_data  # 如果没有，直接返回原数据

        # 获取任务描述
        task = complementary_data["task"]
        # 检查任务是否为None
        if task is None:
            # 如果没有指定任务，提供默认任务
            complementary_data["task"] = "Execute the robot action."
            return complementary_data  # 返回更新后的数据

        # 创建补充数据的副本
        new_complementary_data = dict(complementary_data)

        # 处理字符串和字符串列表两种格式
        if isinstance(task, str):
            # 单个字符串：确保以句点结尾
            if not task.endswith("."):
                new_complementary_data["task"] = f"{task}."
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # 字符串列表：格式化每个字符串
            new_complementary_data["task"] = [t if t.endswith(".") else f"{t}." for t in task]

        # 返回新的补充数据
        return new_complementary_data

    # 转换特征的方法，保持特征不变
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features  # 直接返回特征，不做任何转换
