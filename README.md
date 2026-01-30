# 🚄 Project Rail-CRAG: Railway Standard QA Agent

> 基于 [Corrective Retrieval Augmented Generation (CRAG)](https://arxiv.org/abs/2401.15884) 论文实现的铁路标准智能问答系统。

![CRAG Architecture](https://img.shields.io/badge/Architecture-CRAG-blue) ![Python](https://img.shields.io/badge/Python-3.10-green) ![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 📖 项目简介 (Introduction)

本项目旨在解决传统 RAG 在处理专业铁路标准（如 TB/T 系列）时的检索不准和幻觉问题。我们利用 **MinerU 2.5** 进行高保真文档解析，并实现了 CRAG 论文中的核心机制：

1.  [cite_start]**检索评估器 (Retrieval Evaluator)**: 对检索文档进行可信度打分 (Correct/Incorrect/Ambiguous) [cite: 9]。
2.  [cite_start]**知识提炼 (Knowledge Refinement)**: 对 "Correct" 文档进行 "Decompose-then-Recompose" 粒度清洗 [cite: 11]。
3.  [cite_start]**网络搜索扩展 (Web Search)**: 当内部知识不足 ("Incorrect") 时，自动重写查询并联网搜索 [cite: 10]。

## 🏗️ 系统架构 (Architecture)

```mermaid
graph TD
	User[用户提问] --> R[检索器]
	R --> E{评估器 (Evaluator)}
	E -- Correct --> K[知识提炼]
	E -- Incorrect --> W[Query Rewrite + Web Search]
	E -- Ambiguous --> H[混合模式]
	K --> G[生成器]
	W --> G
	H --> G
	G --> Final[最终回复]

```

## 🚀 快速启动 (Quick Start)

### 方法 A: Docker 一键启动 (推荐)

确保本地已安装 Docker 和 Docker Compose。

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY 和 TAVILY_API_KEY

# 2. 启动服务
docker-compose up --build

```

* **Web UI**: http://localhost:8501
* **API Doc**: http://localhost:8000/docs

### 方法 B: 本地开发运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 API
python -m src.server

# 启动 UI
streamlit run ui.py

```

## 📂 数据处理 (Data Ingestion)

本项目依赖 **MinerU (Magic-PDF)** 进行高精度解析。

1. 准备 PDF 文件放入 data/ 目录。
2. 运行解析脚本（需 GPU 环境）：
```bash
magic-pdf -p data/standard.pdf -o data/output -m auto

```


3. 将生成的 Markdown 导入向量库：
```bash
# 使用 CLI 或 API
python -m src.main ingest --file data/output/standard.md

```

## 🧩 示例数据 (Sample Data)

无需 GPU 也可体验 Ingest：

```bash
python -m src.main ingest --file data/sample_standard.md

```



## 🧪 基准测试 (Benchmark)

运行对比脚本，查看 CRAG 与 Standard RAG 的效果差异：

```bash
python -m src.evaluation.benchmark_comparison

```

## 📝 引用 (Reference)

* **CRAG Paper**: [arXiv:2401.15884 [cs.CL]](https://arxiv.org/abs/2401.15884) 


* **MinerU**: [OpenDataLab/MinerU](https://github.com/opendatalab/MinerU)

---

*Built with LangGraph, FastAPI, and Streamlit.*

## 检索与搜索参数
- RETRIEVER_K：向量检索 Top-K
- SEARCH_K：Web 搜索 Top-K
- CRAG_UPPER_THRESHOLD / CRAG_LOWER_THRESHOLD：Correct/Incorrect 阈值
