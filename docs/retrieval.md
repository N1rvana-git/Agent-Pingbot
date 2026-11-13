# 检索链路说明

本文说明 Rail-CRAG 的检索层为什么这样设计、参数如何影响结果，以及出问题时从哪里排查。实现位于 `src/rag/`，评测位于 `scripts/eval_retrieval.py`。

## 1. 链路总览

```text
Markdown/PDF
   │  清洗：去图床 base64、HTML、转义、表格分隔线
   ▼
层级切分：heading path + 段落装箱（≤800 字，超长按句切）
   ▼
ChromaDB：documents + metadata(source/path/part)，确定性 chunk id
   │
   ├─ BM25 腿：CJK 单字 + bigram + 英文/数字 token
   └─ 向量腿：embedding 后端（hashing / OpenAI 兼容）
   ▼
RRF 融合：score = Σ 1/(k + rank)，k=60
   ▼
重排候选池：查询覆盖率 + 章节命中 + 标准号命中 + 长度惩罚
   ▼
CRAG 评估器：Correct / Incorrect / Ambiguous
```

## 2. 解析与切分

MinerU 产出的 Markdown 有三类噪声会直接毁掉检索质量：

1. **图床 base64**。一个附录里的 base64 图片定义可以让单个片段膨胀到数万字符，向量被图片字节主导，任何查询都召回同一个片段。解析器先删除 `[imageN]: <data:...>` 和行内图片。
2. **装饰性标题**。`## ---`、`**2\.1**` 这类标题会把无意义字符串写进章节路径，先清洗再入栈。
3. **超长小节**。按空行分段后装箱到 `max_chars=800`，单个超长段落按句号切分；`part/parts` 写入 metadata，便于回溯源文档位置。

`min_chars=12` 以下的片段直接丢弃：它们通常是表头或图注，除了稀释 BM25 的文档长度没有作用。

## 3. 为什么是 BM25 + 向量 + RRF

- **BM25**：铁路标准里大量查询以标准号、限值、指标名为主（`GB 146.2`、`SIL4`、`20Pa`），词法匹配稳定且可解释。分词不依赖第三方分词器：中文单字 + 相邻 bigram 能覆盖绝大多数双字词，英文/数字按 NFKC 归一后切分，`TB 10621` 与 `TB10621` 会得到同一组 token。
- **向量**：处理同义改写、语序变化和长句提问。默认的哈希向量是离线基线（`zlib.crc32` + 亚线性 TF + L2 归一化），它保证 CI 和本地演示可复现；配置 `EMBEDDING_BASE_URL` 后切换到真实 embedding，接口不变。
- **RRF**：BM25 分数与余弦相似度量纲不同，加权平均需要对每个语料重新调参。RRF 只读名次：`Σ 1/(k + rank)`，`k=60` 是信息检索里的常用默认值，无需针对本语料重新标定。

融合后再取候选池（默认 20 条）做重排，而不是直接取 top-k：重排的输入越多，排序信息越充分，但延迟也越高。`RETRIEVAL_POOL` 是精度与延迟的旋钮。

## 4. 重排打分

`src/rag/reranker.py` 的启发式打分：

```text
rerank_score =
    1.00 × 融合分数
  + 0.80 × 查询词在正文中的覆盖率
  + 0.25 × 查询词命中章节路径
  + 0.45 × 查询中的标准号出现在片段/路径
  + 0.15 × 查询意图是"限值/安全/要求"且片段含同类词
  - 0.05 × 片段过短或过长
```

标准号加分是专门为这个领域加的：查询 `TB 10621 最小曲线半径` 时，正文相似的通用章节会输给标题里明确写着 `TB 10621` 的那一节。权重是显式常量，评审可以直接质疑和修改，不需要训练数据。

## 5. 失败与降级

| 场景 | 行为 | 排查点 |
| --- | --- | --- |
| embedding 服务不可用 | 向量腿抛错 → 只用 BM25 返回，`retrieval_trace.vector_error` 记录原因 | `/healthz`、API 日志 |
| 索引为空 | 返回空结果，生成器回答"未检索到相关内容" | 先执行 `ingest` |
| 索引维度与当前后端不一致 | `IndexMismatchError`，提示用 `--reset` 重建 | `python -m src.main ingest ... --reset` |
| 集合被切换过 embedding 后端 | 集合 metadata 记录后端名，打开时校验并拒绝混用 | `stats` 输出 |
| 同一来源重复入库 | 先按 `source` 删除旧片段再写入，chunk id 由内容哈希生成 | `store.add_chunks` |

向量腿失败时仍然给出答案，但降级是**可见的**：trace 里带 `vector_error`，UI 显示警告，日志记 warning。这与"静默回退"是两种工程质量。

## 6. 评测口径

`tests/data/retrieval_eval.json` 每条样本包含 `id`、`query`、`expected_keywords`。命中定义：结果正文或章节路径包含任一关键词。

- `HitRate@k`：前 k 条至少命中一次的查询占比（召回可用性）
- `MRR`：首个命中位次倒数的均值（找得靠不靠前）
- `nDCG@k`：二值相关性下的排序质量，`DCG = 1/log2(rank+1)`
- `latency_ms`：单查询平均耗时（含重排）

```bash
python scripts/eval_retrieval.py --top-k 1
python scripts/eval_retrieval.py --top-k 1 --json
python scripts/eval_retrieval.py --top-k 1 --min-hit-rate 0.95   # CI 门槛
```

结果与解读见 README 第 4 节。结论是：k=5 时召回饱和，k=1 才能区分排序策略；哈希向量与 BM25 高度相关，语义召回的收益要换成真实 embedding 才能观察到。

## 7. 扩展点

1. **真实语义向量**：设置 `EMBEDDING_BASE_URL` + `EMBEDDING_MODEL`（兼容 `/embeddings` 的 bge、TEI、vLLM 均可），重建索引后重跑消融。评测口径不变。
2. **Cross-encoder 重排**：`Reranker` 是 Protocol，实现 `rerank(query, candidates, top_k)` 即可替换启发式实现；建议保留启发式作为降级路径。
3. **条款级切分**：当前按标题层级切分；若要精确到"第 5.3.2 条"，可在 `mineru_parser.py` 增加条款编号识别，把条款号写入 metadata 并参与重排。
4. **多语料**：`ChunkRecord.metadata` 已保留 `source`，可按来源过滤或做来源级加权。
