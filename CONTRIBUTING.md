# How to contribute to 🤗 LeRobot

Everyone is welcome to contribute, and we value everybody's contribution. Code is not the only way to help the community. Answering questions, helping others, reaching out, and improving the documentation are immensely valuable.

Whichever way you choose to contribute, please be mindful to respect our [code of conduct](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md) and our [AI policy](https://github.com/huggingface/lerobot/blob/main/AI_POLICY.md).

## Ways to Contribute

You can contribute in many ways:

- **Fixing issues:** Resolve bugs or improve existing code.
- **New features:** Develop new features.
- **Extend:** Implement new models/policies, robots, or simulation environments and upload datasets to the Hugging Face Hub.
- **Documentation:** Improve examples, guides, and docstrings.
- **Feedback:** Submit tickets related to bugs or desired new features.

If you are unsure where to start, join our [Discord Channel](https://discord.gg/q8Dzzpym3f).

## Development Setup

To contribute code, you need to set up a development environment.

### 1. Fork and Clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-handle>/lerobot.git
cd lerobot
git remote add upstream https://github.com/huggingface/lerobot.git
```

### 2. Environment Installation

Please follow our [Installation Guide](https://huggingface.co/docs/lerobot/installation) for the environment setup & installation from source.

## Running Tests & Quality Checks

### Code Style (Pre-commit)

Install `pre-commit` hooks to run checks automatically before you commit:

```bash
pre-commit install
```

To run checks manually on all files:

```bash
pre-commit run --all-files
```

### Running Tests

We use `pytest`. First, ensure you have test artifacts by installing **git-lfs**:

```bash
git lfs install
git lfs pull
```

Run the full suite (this may require extras installed):

```bash
pytest -sv ./tests
```

Or run a specific test file during development:

```bash
pytest -sv tests/test_specific_feature.py
```

## Submitting Issues & Pull Requests

Use the templates for required fields and examples.

- **Issues:** Follow the [ticket template](https://github.com/huggingface/lerobot/blob/main/.github/ISSUE_TEMPLATE/bug-report.yml).
- **Pull requests:** Rebase on `upstream/main`, use a descriptive branch (don't work on `main`), run `pre-commit` and tests locally, and follow the [PR template](https://github.com/huggingface/lerobot/blob/main/.github/PULL_REQUEST_TEMPLATE.md).

One member of the LeRobot team will then review your contribution.

Thank you for contributing to LeRobot!

---

# 中文翻译

# 如何为 🤗 LeRobot 做出贡献

欢迎每个人做出贡献，我们重视每个人的贡献。代码不是帮助社区的唯一方式。回答问题、帮助他人、联系和改进文档都非常有价值。

无论您选择以何种方式做出贡献，请注意尊重我们的[行为准则](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)和[AI 政策](https://github.com/huggingface/lerobot/blob/main/AI_POLICY.md)。

## 贡献方式

您可以通过多种方式做出贡献：

- **修复问题：** 解决错误或改进现有代码。
- **新功能：** 开发新功能。
- **扩展：** 实现新的模型/策略、机器人或模拟环境，并将数据集上传到 Hugging Face Hub。
- **文档：** 改进示例、指南和文档字符串。
- **反馈：** 提交与错误或所需新功能相关的工单。

如果您不确定从哪里开始，请加入我们的 [Discord 频道](https://discord.gg/q8Dzzpym3f)。

## 开发设置

要贡献代码，您需要设置开发环境。

### 1. Fork 和 Clone

在 GitHub 上 Fork 仓库，然后克隆您的 fork：

```bash
git clone https://github.com/<your-handle>/lerobot.git
cd lerobot
git remote add upstream https://github.com/huggingface/lerobot.git
```

### 2. 环境安装

请按照我们的[安装指南](https://huggingface.co/docs/lerobot/installation)进行环境设置和从源码安装。

## 运行测试和质量检查

### 代码风格（Pre-commit）

安装 `pre-commit` 钩子以在提交前自动运行检查：

```bash
pre-commit install
```

手动对所有文件运行检查：

```bash
pre-commit run --all-files
```

### 运行测试

我们使用 `pytest`。首先，确保您通过安装 **git-lfs** 拥有测试工件：

```bash
git lfs install
git lfs pull
```

运行完整套件（这可能需要安装额外的依赖）：

```bash
pytest -sv ./tests
```

或在开发期间运行特定的测试文件：

```bash
pytest -sv tests/test_specific_feature.py
```

## 提交问题和拉取请求

使用模板获取必填字段和示例。

- **问题：** 遵循[工单模板](https://github.com/huggingface/lerobot/blob/main/.github/ISSUE_TEMPLATE/bug-report.yml)。
- **拉取请求：** 在 `upstream/main` 上变基，使用描述性分支（不要在 `main` 上工作），在本地运行 `pre-commit` 和测试，并遵循[PR 模板](https://github.com/huggingface/lerobot/blob/main/.github/PULL_REQUEST_TEMPLATE.md)。

LeRobot 团队的一名成员将审查您的贡献。

感谢您为 LeRobot 做出贡献！
