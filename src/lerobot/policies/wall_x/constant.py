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
Wall-X Constants and Configuration Data.
"""

# 相机名称映射字典，将内部相机名称映射到人类可读的名称
CAMERA_NAME_MAPPING = {
    "face_view": "front view",  # 面部视图映射到前视图
    "left_wrist_view": "left wrist view",  # 左手腕视图映射到左手腕视图
    "right_wrist_view": "right wrist view",  # 右手腕视图映射到右手腕视图
    "move1_view": "move view",  # 移动视图1映射到移动视图
    "move2_view": "move view",  # 移动视图2映射到移动视图
    "wall_view": "wall view",  # 墙壁视图映射到墙壁视图
    "top_view": "top view",  # 顶部视图映射到顶部视图
}

# 图像分辨率常量，所有图像调整到此分辨率
RESOLUTION = 256

# 预处理参数常量
# 最大像素数限制，用于图像预处理
MAX_PIXELS = 16384 * 28 * 28  # 计算最大像素数：16384 * 784 = 12,845,056
# 最小像素数限制，用于图像预处理
MIN_PIXELS = 4 * 28 * 28  # 计算最小像素数：4 * 784 = 3,136
# 图像因子，用于智能调整图像大小
IMAGE_FACTOR = 28  # 图像尺寸调整的因子，确保尺寸能被28整除
# 优先级顺序，用于指令采样
PRIORITY_ORDER = None  # 默认为None，使用函数内部的默认优先级
# 生成子任务的概率比率，设置为1.0表示总是生成子任务
GENERATE_SUBTASK_RATIO = 1.0  # 控制子任务生成的概率
# 模型类型常量
MODEL_TYPE = "qwen2_5"  # 指定使用的Qwen模型版本

# 分词器最大序列长度
TOKENIZER_MAX_LENGTH = 768  # 分词器处理的最大token序列长度
