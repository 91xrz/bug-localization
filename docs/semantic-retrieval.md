# 语义检索与分块

流程：修复前快照 → 四种代码分块 → 同一个本地 Qwen 模型 → 四套 ES 向量索引 → 文件排名 → @10 指标。

## 固定设置

| 设置 | 当前值 |
| --- | --- |
| 模型 | `models/Qwen3-Embedding-0.6B`，不微调 |
| 编码 | 1024 维，最后一个有效 token 池化，L2 归一化 |
| 推理 | CUDA / BF16 / batch size 16；可显式 `--device cpu` |
| 总输入 | 最多 512 tokens，包含前缀、分隔符和特殊 token |
| 结构前缀 | 最多 64 tokens，由同一个 Qwen tokenizer 计数 |
| 固定块正文 | 最多 400 tokens，无重叠，保留末块 |
| 文件得分 | 默认 Max；同时评测 Top-3 Mean、Normalized LogSumExp（τ=0.1） |
| 评测 | MAP@10、MRR@10、Top-1/5/10、MeanRecall@10 |

安装依赖统一使用根目录 `requirements.txt`。编码器直接使用 Transformers，`local_files_only=True`、`trust_remote_code=False`；不需要 Sentence Transformers、llama-index 或 Flash Attention。

## 四种分块

**动态块**沿用 [原始 BLAZE 参考代码](../references/blaze/processor.py) 的 DP 递推、行号约定和相同代价下的断点选择：普通位置代价 50、类代价 3、函数代价 5、窗口 30 行。先保存原始边界，超出模型预算后再沿语句边界细分。它是原动态算法加长度和前缀适配，不等于完整 BLAZE 复现。

**方法块**以顶层函数或类方法为原始块，装饰器与函数一起保留。嵌套函数保留在外层函数块内；方法外的导入、模块代码和类属性生成补充块。纯空白间隔与相邻块合并，长块沿语句边界细分。

**固定块**按完整文件的 token 顺序每 400 tokens 建立窗口，不沿语句或函数调整，无重叠，末块不足 400 也保留。400 指代码正文，结构前缀另计。若窗口切断 Unicode 字符的字节 token，或子串独立分词后超过预算，向前回退并记录 `boundary_adjusted`。因此尾块和安全回退块可能更短。

动态和方法路线仅在单条语句仍超长时才使用 token 边界兜底。三路均逐字节保存并核对源码完整覆盖，不调用原脚本的全局字符串替换去注释逻辑。原始参考文件保留在 `references/blaze/`，不作为运行入口。

**BLagent 风格 AST 块**递归拆分超长语法节点、合并相邻节点，显式使用 1500 字符预算，再适配共同的 512-token 输入限制。支持九种语言、十种语法；旧三路在非 Python 文件上也使用对应语言的 AST 提取结构。原三路 Python 实现保持原样，详见 [多语言 AST 切块](blagent-ast.md)。

## 真实结构前缀与报告编码

前缀从修复前源码的语法树提取，例如：

```text
Path: requests/models.py
Class: Response
Signature: def json(self, **kwargs):
```

没有所属类或函数时使用 `<module>` / `<none>`。跨方法的块记录实际相交的类与签名；前缀超长时保留三个字段标签、缩短内容，完整信息另存元数据。块保存起止行/字节、原始块编号、token 数、前缀截断及语法恢复状态。

旧版 Python 语法可能让解析器标记错误。此时保留完整源码及可恢复结构，在结果的 `parse_error_files` 中列出相关文件，不能把该标记当作源码被删除。

SWE-bench 仅从 `body` 取报告；固定任务指令也计入 512-token 上限。超长报告保留开头，并记录原始长度、实际输入及截断状态。四路共用同一个报告向量，不使用修复后的内容或真实标签构造向量。

## ES 检索与缓存

每块一个 ES 文档，向量字段为 `dense_vector`、`index=false`。`script_score` 精确计算当前快照所有块的 `1 + cosineSimilarity`，通过 scroll 取回全部块分数，再减去偏移 1，按文件计算三种聚合。不会先折叠文件而丢失其他块，也不会只获取全局前 K 个块。这是精确向量检索，当前没有使用 ANN。

先完成文件排名，再取前 10 计算指标。每次独立用 NumPy 核对每个块的余弦相似度及全部三种文件聚合分数，容差 `2e-5`；验证不参与排名。同分按相对路径排序，近似浮点同分的顺序另有核对字段。聚合公式见 [文件聚合说明](file-aggregation.md)，新增路线对比见 [多语言 AST 切块](blagent-ast.md)。

- `cache/embeddings/`：按模型文件 SHA-256、编码实现、库版本、精度和设备生成指纹。SQLite 保存文本向量及校验和，读取时验证维度、有限值及归一化。
- `cache/chunks/`：按相对路径、源码指纹、模型、分块参数及实现缓存完整分块，核对校验和、源码覆盖、字节边界和预算。不同快照中未变化的文件可以复用。
- ES 索引：绑定仓库、提交、路线、块清单及向量内容指纹，检查项目归属与文档数量。

## 运行与输出

ES 启动后，在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -B scripts\retrieval\run_vector_batch.py
```

默认使用前 5 条 `psf/requests` 报告，输出动态、方法、固定和 BLagent AST 四条独立路线。配置在 `configs/retrieval/vector.json`，可用 `--limit 1` 做快速检查，或用 `--routes blagent_ast` 只运行新增路线。需要和已有实验逐条对齐时使用 `--selection <selection.json路径>`，不能同时指定数量或仓库过滤。

每次创建独立的 `results/vector/batches/<run_id>/`：

| 路径 | 内容 |
| --- | --- |
| `summary.json/csv` | 各路线主指标、成功数、失败数及缓存统计 |
| `aggregation_summary.json/csv` | 四种分块 × 三种聚合的十二组指标 |
| `aggregation_samples.json/csv` | 十二组逐报告指标及共同 Chunk 分数指纹 |
| `selection.json`、`effective_config.json` | 样本、标签与查询指纹，模型文件及实现版本 |
| `dynamic/`、`method/`、`fixed/`、`blagent_ast/` | 各路线逐报告结果和汇总 |
| `cases/*/candidates.json`、`query.json` | 候选清单、排除原因和实际模型查询 |
| `cases/*/*_chunks.json` | 分块、真实前缀、原始边界及 token 数 |
| `cases/*/*_chunk_scores.json` | 所有 Chunk 的原始余弦分数及校验值，供三种聚合共同使用 |
| `cases/*/*_result.json` | 完整文件排名、最佳命中块、指标及 ES 核对 |
| `cases/*/*error.json` | 运行失败时的诊断 |

默认 `enable_fusion=false`。显式 `--fusion` 才额外运行动态＋方法的 `rrf` 和动态＋方法＋固定的 `rrf_three`：等权、常数 60、使用完整文件排名，按相对路径处理同分。BM25 与语义对比入口始终关闭融合。

与 BM25 一起运行及当前结果见 [评测说明](evaluation.md)。
