# WALL-OSS

<!-- 这是Wall-OSS的README文件，提供了模型的概述和相关信息 -->

This repository contains the Hugging Face port of [**WALL-OSS**](https://x2robot.com/en/research/68bc2cde8497d7f238dde690), a Vision-Language-Action model for cross-embodiment robotic control based on Qwen2.5-VL with flow matching/FAST action prediction.
<!-- 本仓库包含WALL-OSS的Hugging Face移植版本，这是一个基于Qwen2.5-VL的视觉-语言-动作模型，用于跨具身机器人控制 -->

---

## Model Overview

<!-- 模型概览表格，描述模型的主要特性 -->
| Feature            | Description                                           |
| ------------------ | ----------------------------------------------------- |
| Base Model         | Qwen2.5-VL (Vision-Language Model)                    |
| <!-- 基础模型 --> | <!-- Qwen2.5-VL视觉语言模型 --> |
| Action Prediction  | Flow Matching (diffusion) or FAST (discrete tokens)   |
| <!-- 动作预测方式 --> | <!-- 流匹配（扩散）或FAST离散令牌 --> |
| Architecture       | Mixture of Experts (MoE) with action-specific routing |
| <!-- 架构 --> | <!-- 混合专家模型，带有动作特定路由 --> |
| Multi-Modal Inputs | Vision (images/videos), Language, Proprioception      |
| <!-- 多模态输入 --> | <!-- 视觉（图像/视频）、语言、本体感觉 --> |

---

## Additional Resources

<!-- 附加资源链接 -->
Paper: https://arxiv.org/pdf/2509.11766
<!-- 论文链接 -->

Official Repository: https://github.com/X-Square-Robot/wall-x
<!-- 官方仓库链接 -->

---

## 中文翻译

本仓库包含 [**WALL-OSS**](https://x2robot.com/en/research/68bc2cde8497d7f238dde690) 的 Hugging Face 移植版本，这是一个基于 Qwen2.5-VL 的视觉-语言-动作模型，用于跨具身机器人控制，支持流匹配/FAST 动作预测。

### 模型概览

| 特性 | 描述 |
| --- | --- |
| 基础模型 | Qwen2.5-VL（视觉语言模型） |
| 动作预测 | 流匹配（扩散）或 FAST（离散令牌） |
| 架构 | 带动作特定路由的混合专家模型（MoE） |
| 多模态输入 | 视觉（图像/视频）、语言、本体感觉 |

### 附加资源

论文：https://arxiv.org/pdf/2509.11766

官方仓库：https://github.com/X-Square-Robot/wall-x

Hugging Face：https://huggingface.co/x-square-robot

### 引用

如果您使用此工作，请引用以下论文：

```bibtex
@article{zhai2025igniting,
    title   = {Igniting VLMs Toward the Embodied Space},
    author  = {Zhai, Andy and Liu, Brae and Fang, Bruno and Cai, Chalse and Ma, Ellie and Yin, Ethan and Wang, Hao and Zhou, Hugo and Wang, James and Shi, Lights and Liang, Lucy and Wang, Make and Wang, Qian and Gan, Roy and Yu, Ryan and Li, Shalfun and Liu, Starrick and Chen, Sylas and Chen, Vincent and Xu, Zach},
    journal = {arXiv preprint arXiv:2509.11766},
    year    = {2025}
}
```

### 许可证

本模型遵循 **Apache 2.0 许可证**，与原始 [WallX 仓库](https://github.com/X-Square-Robot/wall-x) 保持一致。
