# PDF 接入（可选）

Markdown 语料不需要任何额外依赖。只有需要直接解析 PDF 时才安装 MinerU 工具链：

```bash
pip install -r requirements-pdf.txt
```

`requirements-pdf.txt` 会拉取 PyTorch、transformers 和 ModelScope，体积较大，因此不放进主依赖，也不进 Docker 默认镜像。

## 1. 首次初始化

MinerU 首次运行需要下载布局/OCR 模型。按官方文档准备模型目录后执行一次：

```bash
magic-pdf --help
```

国内网络可先配置 ModelScope 模型源（环境变量 `MODELSCOPE_CACHE` 指定缓存目录），或在能联网的机器上准备好模型后拷贝整个缓存目录。

## 2. 解析并入库

```bash
python -m src.main ingest --file data/standard.pdf --reset
```

流程：

1. `MinerULoader` 调用 `magic-pdf -p <pdf> -o data/mineru_output -m auto`；
2. 读取生成目录中的 Markdown；
3. 走与 Markdown 完全相同的清洗、切分、索引流水线。

`data/mineru_output/` 已在 `.gitignore` 中，解析中间产物不会进入仓库。

## 3. 注意事项

- **扫描件**依赖 OCR，MinerU 的 `-m auto` 会自动判断；纯文字 PDF 建议显式使用文本模式以获得更快的解析。
- **表格**会被转成 Markdown 表格，解析器保留表格行、去掉分隔线；跨行跨列表头会丢失部分语义，重要表格建议人工校对后以 Markdown 入库。
- **失败处理**：`magic-pdf` 不存在时命令会给出明确提示；解析失败按文件记录 `SKIP`，不会中断其余文件。
- **CPU 也能跑**，但大文档解析时间会明显变长；生产环境建议在独立任务里做解析，只把 Markdown 交给服务。
