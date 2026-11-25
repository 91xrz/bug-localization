# BLAZE 发布编码器基线

更新：2026-11-16。BLAZE 发布编码器适配版已完成固定 500 条：三条路线均 500/500 成功；与 RRL 和本项目方法的结果见 [500 条外部基线对比](external-comparison-500.md)。选型范围见 [外部论文对比基线](external-baselines.md)。

## 实验身份与公开材料

当前实验名称为 **BLAZE released encoder / author-demo adaptation**。论文为 Partha Chakraborty、Mahmoud Alfadel、Meiyappan Nagappan 的 *BLAZE: Cross-Language and Cross-Project Bug Localization via Dynamic Chunking and Hard Example Learning*，TSE 2025，[DOI](https://doi.org/10.1109/TSE.2025.3579574)。

- 作者源码、训练程序、示例和数据：[Zenodo v16](https://zenodo.org/records/15122980)，发布材料标 CC BY 4.0。动态分块、清理及融合规则据此适配；原始参考文件仍保存在 `references/blaze/`。
- 发布权重：[bug-localization/BLAZE](https://huggingface.co/bug-localization/BLAZE/tree/1b14d31e5b3094e357ab00e971689f402ed62ca1)，模型卡标 MIT；固定 revision 为 `1b14d31e5b3094e357ab00e971689f402ed62ca1`。
- CodeSage 模型定义：[codesage/codesage-small](https://huggingface.co/codesage/codesage-small/tree/4d2a4f3fa73a3c8a4dd59dba2712202f1f54e109)，模型卡标 Apache-2.0；固定 revision 为 `4d2a4f3fa73a3c8a4dd59dba2712202f1f54e109`。

**发布权重与训练结构存在差异。** 作者训练代码的 `UniEmbeddingPL` 包含报告/源码两套残差适配器；本地及 HF 发布的 safetensors 只有 76 个 CodeSage 编码器张量，没有适配器参数。作者最小演示直接使用该编码器与 masked mean pooling，当前沿用此公开推理路径，没有随机补齐适配器，也没有换用 Qwen 编码器。其结果不能称为重新训练或原论文全部设置的完整复现。

权重 SHA-256 为 `15363f2a8398bceeecb427041e20560abfae5dda0ebef83a52162a417575b920`。准备脚本逐项核验固定版本的小文件及模型代码；本机已有 `BlazeModel` 权重通过校验后仅复用文件，新配置和 tokenizer 放在独立 `models/BLAZE-release/` 中，原目录保持原样。每次启动重新校验资产，再以严格模式加载全部张量。

## 已接入的流程

| 环节 | 当前行为 |
| --- | --- |
| 报告、源码、候选 | 复用现有数据选择和源码校验；默认 SWE 格式化版、body-only、修复前 SHA、`.py` 候选及相同排除规则 |
| 词法路 | 每个原始完整文件为一个文档；Elasticsearch standard analyzer、默认 BM25 k1=1.2/b=0.75、`match(code, query)`；前 1,000 文件 |
| 动态分块 | 作者 CommentCleaner、空行合并、30 行 DP、类/函数/一般边界成本 3/5/50；无结构前缀，无额外 token 切块 |
| 语义编码 | 发布的 CodeSage 编码器，attention-mask mean pooling、L2 归一化；源码 512、报告 1,024 token 上限，逐项记录截断 |
| 语义检索 | 本地 float32 精确 cosine；前 1,000 块；并列按固定源码/块顺序 |
| BLAZE 融合 | 按作者顺序累加每个命中块的 `1/(60+rank)`，再加词法文件贡献；同文件多个块分别贡献，融合前不去重 |
| 对照输出 | `blaze_bm25`、`blaze_dense`、`blaze_rrf`；dense 文件对照仅在取前 1,000 块后按首次出现去重 |
| 统一评价 | MAP/MRR/Recall@10、Top-1/5/10；AP 分母保留全部 gold，候选缺失标签不会被剔除 |
| 留痕 | 样本清单、查询、模型和代码校验值、候选指纹、块排名、三路排名、截断/空块/失败记录、分阶段耗时 |

源码位置为 `src/bug_localization/blaze.py`、`blaze_model.py`、`blaze_experiment.py`；入口为 `scripts/retrieval/run_blaze.py`，默认配置为 `configs/retrieval/blaze.json`。

## 明确记录的适配差异

1. 作者使用 Milvus COSINE / IVF_FLAT；这里使用精确 cosine。没有近似索引误差，但检索后端、建索引和检索耗时不同。
2. 使用本项目统一候选与 body-only 报告；词法路保留完整查询，不使用现有 BM25 的 identifier analyzer 或 512 词查询截断。`base.bm25`、`base.max_query_terms` 不控制 BLAZE 词法路。Elasticsearch 为本机 9.x，并列词法分数按文件路径稳定排序。
3. 分块使用当前 Tree-sitter grammar。保留作者原节点类型、lambda 子串判断、首个子节点为 ERROR 时的 2,000 字符回退。作者 CommentCleaner 的全局字符串替换及 Unicode 字节/字符偏移问题也保留；这是一项已知限制，不宣称为修正后的新清理算法。当前入口只允许作者转换脚本的 Python/Java/C++/JavaScript 扩展映射；Kotlin 尚未接入。
4. 空文件、清理后为空且被 tokenizer 编成零 token 的块仅从语义路排除；文件仍在共同候选和 BM25 中，gold 分母不变。排除项逐个写入 `embedding_cache.excluded_empty_chunks`。全库没有有效块会明确记录失败，不生成伪向量。
5. token 上限采用作者训练配置的源码 512、报告 1,024；作者最小演示本身没有显式截断。当前按最长样本动态 padding，全部使用 float32。公开权重缺失适配器的限制见上文。
6. 为兼容当前 Transformers 5，只调整 CodeSage 的 Conv1D 导入位置、模型初始化 API、无 head pruning 时的空 mask。用原版源代码与 Transformers 4.49.0 做对照：4 段输入 × 2 个 token 上限，共 8 个向量，最大绝对差为 **0.0**；含中文、不同长度 padding 和截断输入。记录在 `results/blaze/encoder-compatibility/comparison.json`。

## 运行方式

在项目根目录执行，使用现有 Python 环境和本地 Elasticsearch 服务：

```bash
.venv/bin/python scripts/datasets/prepare_blaze_model.py
.venv/bin/python scripts/retrieval/run_blaze.py
```

首次准备会下载固定版本的小文件及模型定义；没有可复用权重时会下载约 512 MB 权重。之后模型推理只读本地文件；缺失的源码快照仍由现有源码模块准备。默认是 Requests 按原始数据顺序前 3 条、CPU 4 线程，不占用 GPU。指定单报告功能检查：

```bash
.venv/bin/python scripts/retrieval/run_blaze.py --limit 1 --repository psf/requests
```

与已有 500 条开发实验对齐的入口：

```bash
.venv/bin/python scripts/retrieval/run_blaze.py \
  --selection configs/experiments/selections/crossproject_500_seed20261111.json \
  --device cuda:0
```

须先确保该 GPU 可用；500 条已用于开发分析，不能称独立测试集。固定清单不能与 `--limit` / `--repository` 混用。其他数据集通过配置中的 `base_config` 指向对应数据及源码提供者，仍要先核查语言、SHA 和作者数据的对应关系。

输出在 `results/blaze/<时间>/`。`selection.json` 和 `effective_config.json` 固定输入，`provenance.json` 固定实现与模型，`cases/*/result.json` 保存完整排名；三个路线子目录各有样本和汇总 JSON/CSV。缓存位于 `cache/blaze/`，身份包括源码、分块、模型、实现、依赖、token 上限和空块规则，读取时校验向量文件。重跑生成新结果目录并复用有效缓存；不会覆盖历史结果。

## 最小实测结果

### 2026-11-16：500 条失败重跑

原 500 条运行中 219 条成功、281 条失败，其中 271 条因共享 Elasticsearch 分片上限、10 条因查询子句上限。原始结果保留在 `results/blaze/crossproject500-gpu-20261116/`。

重跑使用同版本 Elasticsearch 9.5.5 的隔离节点 `blaze-repair-local`（本机 19200，4 GB heap、search 线程 1），数据和索引位于 `.runtime/blaze-repair-es/`，不修改共享节点或其索引。该节点使用相同索引定义、standard analyzer 和完整原始查询；没有分段、去重或截断词法查询。第一条此前超出查询限制的报告已完整通过三条路线；另将一条原成功报告在新节点重新运行词法检索，883 个文件的分数和排名逐项完全一致，记录在 `results/blaze-repair-20261116/service-parity.json`。这两条验证不能代替全部报告的逐条验证。

仅重跑原来至少一条路线失败的 281 条，选择清单及有效配置在 `results/blaze-repair-20261116/`；编码器、模型文件、查询字段、候选过滤、排名与指标未改变。GPU 0 执行，tmux 会话为 `irbl-blaze-repair-500`，隔离检索节点在 `irbl-blaze-repair-es`。输出为 `results/blaze-repair-20261116/retry/`，日志为同目录下 `retry.log`。重跑完成后自动执行 `scripts/experiments/merge_blaze_repair.py`，逐路线复用原成功结果并替换原失败结果，验证样本身份、查询和标签哈希、模型、ES 版本及参数，并依据保存排名重新计算成功记录的指标。合并目录为 `results/blaze-repair-20261116/merged500/`；所有样本尝试完毕之前不输出完整 500 条对比成绩，若仍有失败须如实报告。

分片压力通过隔离服务解决，因此无需删除原有 BLAZE 或其他实验的共享索引。隔离节点单次最多需容纳本次 281 条的索引；实验结束后这些本地数据可继续用于审计。

重跑及合并已完成：原 219 条成功结果与重跑 281 条均通过核验，三条路线均 500/500 成功。完整 500 条中词法／dense／融合 MAP@10 分别为 0.391012／0.371827／0.302903；逐项目、失败口径及与其他方法的比较见 [500 条外部基线对比](external-comparison-500.md)。

2026-11-14，`psf/requests` 的 #1142、#1327、#1339；CPU 4 线程、batch=4，3/3 成功，全部 gold 都在共同候选中。产物：`results/blaze/requests3-release-smoke-v2-20261114/`。

| 路线 | MAP@10 | MRR@10 | Top-1 | MeanRecall@10 |
| --- | ---: | ---: | ---: | ---: |
| BLAZE 的 standard BM25 | 0.880952 | 1.000000 | 3/3 | 1.000000 |
| 发布编码器 dense | 0.583333 | 0.666667 | 1/3 | 1.000000 |
| 发布编码器 + 作者 chunk-RRF | 0.555556 | 0.611111 | 1/3 | 0.833333 |

三个源码版本分别有 70/74/74 个候选文件、456/490/483 个有效语义块，排除 3/2/2 个零 token 块。首次分块与源码编码分别约 97/104/102 秒，报告编码约 0.04/0.27/0.69 秒。词法索引在排查过程中已建立，这些时间不是全流程冷启动性能测量。

这 3 条仅证明流程可执行，不用于判断 BLAZE 优劣。最初试跑发现零 token 块并失败，排查产物保留在 `requests3-release-smoke-20261114/`；上表只来自修正后独立运行。后续正式比较应使用共同冻结清单并报告全部失败，不把初次调试失败混入方法精度。

回归测试覆盖原始预处理/DP 一致性、重复块融合、排名深度与去重、缓存重用/失效/损坏、空块策略、查询和索引边界以及失败分母。运行：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_blaze.py' -v
```
