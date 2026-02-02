# ModelScope 集成与部署指南

本文档基于你提供的 ModelScope 安装与下载指引，给出面向本项目的两种集成路径：

- **Docker 镜像集成（推荐）**：环境最稳定，适合生产或快速验证。
- **Python 依赖安装（灵活）**：适合已有开发环境或需要精细控制依赖的场景。

---

## 方案一：基于 Docker 镜像集成（推荐）

如果本项目以容器方式运行，建议直接使用 ModelScope 官方镜像作为基础镜像。

### GPU 环境（推荐用于生产/推理）

```dockerfile
FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py311-torch2.3.1-tf2.16.1-1.34.0
# 之后添加您的 Agent 代码依赖
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
```

### CPU 环境（仅用于开发/测试）

```dockerfile
FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-py311-torch2.3.1-1.34.0
```

### 大模型（LLM）专用环境

```dockerfile
FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.9.1-1.34.0-LLM
```

---

## 方案二：手动安装依赖（基于 requirements.txt）

### 1. 基础环境准备

确保 Python 版本为 3.8+。如果是 Linux 环境且涉及语音模型，推荐 Python 3.8。

```bash
conda create -n modelscope python=3.11
conda activate modelscope
```

### 2. 核心库安装

在 requirements.txt 中添加 `modelscope`，或直接安装：

- 轻量版（仅下载模型）

```text
modelscope
```

- 完整版（推荐）

```text
modelscope[framework]
```

### 3. 分领域依赖（按需添加）

- NLP（自然语言处理）

```bash
pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

- CV（计算机视觉）

```bash
pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

> 注意：CV 领域少数模型需要 mmcv-full 1.x 版本，请勿安装 2.x。

- Multi-modal（多模态）

```bash
pip install "modelscope[multi-modal]"
```

- Audio（语音）

```bash
pip install "modelscope[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

> 注意：大部分语音模型推荐 Linux + Python 3.8；Linux 用户需安装 libsndfile1：

```bash
sudo apt-get install libsndfile1
```

### 4. 大模型训练/推理依赖（SWIFT）

```bash
pip install ms-swift -U
```

---

## 部署后验证

在 Agent 初始化或本地验证阶段，建议先执行 NLP 任务测试：

```python
from modelscope.pipelines import pipeline


def test_modelscope_env() -> None:
    """验证 ModelScope 环境是否可用。"""
    try:
        word_segmentation = pipeline("word-segmentation")
        result = word_segmentation("今天天气不错，适合出去游玩")
        print("ModelScope 环境验证成功，分词结果：", result)
    except Exception as exc:
        print("ModelScope 环境配置有误：", exc)


if __name__ == "__main__":
    test_modelscope_env()
```
