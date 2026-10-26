# BLagent 风格的多语言 AST 切块

新增路线名为 `blagent_ast`，与 `dynamic`、`method`、`fixed` 一起接入现有语义入口。四路共用本地 Qwen3-Embedding-0.6B、ES 精确余弦检索、64-token 结构前缀、512-token 总输入和 @10 指标。每路仍在同一份完整 Chunk 分数上比较 Max、Top-3 Mean、Normalized LogSumExp（τ=0.1），共 12 组。

## 参考实现与适配范围

用户提供的两个文件已原样归档到 [references/blagent](../references/blagent/README.md)。它们使用 LlamaIndex `CodeSplitter`，并包含 Chroma 的建库和检索流程。当前运行代码不导入这些文件，不安装或调用 Chroma、LangChain、LlamaIndex，也不采用原文件中的另一套嵌入模型或测试文件过滤规则。

AST 路线参考 [CodeSplitter 上游实现](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/code.py) 的策略：依次合并相邻语法节点，遇到超长子节点继续递归。字符预算显式设为 `ast_max_chars=1500`。提供的包装文件没有锁定 LlamaIndex 版本，且行数参数分别为 10 和 35；当前路线是明确记录参数的适配实现，不是 BLagent 完整论文系统的严格复现。

适配保留全部源码字节及空白，不剥离每块首尾；不丢弃超长字符串、注释或解析错误文件。超长叶节点按 Unicode 字符安全拆分；字符分块之后，超出模型输入预算的部分优先沿语句/声明边界拆分，仍超长时才使用 token 兜底。每块保存起止行/字节、原始父块、语言、解析恢复和拆分原因。当前无重叠。

## 语言适配

| 语言/语法 | 扩展名 | 语法包 |
| --- | --- | --- |
| Python | `.py`、`.pyi` | tree-sitter-python |
| Java | `.java` | tree-sitter-java |
| JavaScript/JSX | `.js`、`.mjs`、`.cjs`、`.jsx` | tree-sitter-javascript |
| TypeScript | `.ts`、`.mts`、`.cts` | tree-sitter-typescript |
| TSX | `.tsx` | tree-sitter-typescript |
| C | `.c`、`.h` | tree-sitter-c |
| C++ | `.cpp`、`.cc`、`.cxx`、`.hpp`、`.hh`、`.hxx` | tree-sitter-cpp |
| C# | `.cs` | tree-sitter-c-sharp |
| Go | `.go` | tree-sitter-go |
| Rust | `.rs` | tree-sitter-rust |

这些语法均使用独立的本地 wheel，不在解析时下载语法库。`languages.py` 将各语言真实的类、函数、方法、构造函数等节点映射为统一结构前缀；支持 Go 接收者与 Rust impl 的所属类型。没有类或方法时使用 `<module>` / `<none>`，完整签名另存元数据。未注册的扩展名会明确报错。

旧三路在 Python 文件上继续使用原实现；在其他受支持语言上使用相同参数和各自语言的语法结构。动态块仍按类/函数节点代价及行窗口生成边界，方法块按函数/方法及剩余代码切分，固定块仍使用 400-token 正文窗口。

候选扩展名由基础配置的 `extensions` 决定，默认仍为 `.py`，保证当前 44 条报告与历史候选一致。多语言实验可复制基础配置、修改扩展名，再让语义配置的 `base_config` 指向它。例如：

```json
{
  "extensions": [".py", ".java", ".js", ".ts", ".tsx", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs"]
}
```

这是基础配置的字段片段，需保留其余字段。若 `.h` 是 C++，在语义配置中设置 `"language_overrides": {".h": "cpp"}`；默认 `.h` 按 C 解析。多语言能力不补全数据集缺失的修复前提交，Ye 数据仍需先解决该问题。

## 运行与输出

安装更新后的 `requirements.txt`，启动 ES 后执行：

```powershell
.\.venv\Scripts\python.exe -B scripts\retrieval\run_vector_batch.py --repository psf/requests --limit 44 --no-fusion
```

可用 `--selection <selection.json>` 固定已有 44 条清单；不能与数量/仓库过滤混用。只运行新增路线可使用 `--routes blagent_ast`，字符预算可用 `--ast-max-chars 1500` 明确指定。

- `results/vector/batches/<run_id>/summary.csv`：四种分块的主聚合指标，默认 Max。
- `aggregation_summary.csv/json`：全部 12 组指标。
- `aggregation_samples.csv/json`：44 条完整运行时有 528 行。
- `cases/*/blagent_ast_chunks.json`：AST 分块、真实前缀与原始边界。
- `cases/*/blagent_ast_chunk_scores.json`：完整 ES Chunk 分数。
- `cases/*/blagent_ast_result.json`：三种聚合排名、指标及 NumPy 独立核对。

分块缓存和 ES 身份包含 AST 实现、语法包版本、语言映射和参数指纹；原三路 Python 缓存及索引继续复用。RRF 默认关闭，原有双路/三路融合控制仍只使用原三路。

## 多语言验证

`tests/fixtures/multilingual/` 包含十种语法的小型合法源码。单元测试对每种语法运行四条路线，检查结构前缀、无源码丢失、无重叠、token 预算和真实语法解析；另外覆盖错误语法、超长 Unicode 叶节点及缓存篡改。

以下入口进一步使用实际 Qwen 模型和 ES 对这些文件执行四路向量检索，并独立核对分数和排名：

```powershell
.\.venv\Scripts\python.exe -B scripts\retrieval\verify_multilingual_ast.py
```

输出在 `results/vector/multilingual/<run_id>/`。这是多语言功能验证，不是多语言缺陷定位性能评测；当前 44 条报告的指标仍来自 Python 项目 `psf/requests`。

## 本次 44 条报告的结果

语义运行：`results/vector/batches/swe_bench_formatted_test_20261016_112428_262955/`。四路均成功 44 条，失败为 0；相同报告、候选文件、修复前提交、标签、Qwen 模型和输入预算，未启用 RRF。

| 分块 | 聚合 | MAP@10 | MRR@10 | Top-1 | Top-5 | Top-10 | MeanRecall@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 动态 | Max | 0.5650 | 0.6082 | 0.4091 | 0.8636 | 0.9545 | 0.9276 |
| 方法 | Max | 0.5239 | 0.5684 | 0.4091 | 0.7727 | 0.9091 | 0.8949 |
| 固定 400 tokens | Max | 0.5466 | 0.5606 | 0.3409 | 0.8864 | 0.9318 | 0.9176 |
| BLagent AST | Max | **0.6297** | **0.6526** | 0.4773 | 0.8636 | 0.9545 | **0.9418** |
| BLagent AST | Top-3 Mean | 0.6105 | 0.6485 | **0.5000** | 0.8636 | 0.9545 | 0.9290 |
| BLagent AST | Normalized LogSumExp，τ=0.1 | 0.3531 | 0.4084 | 0.2500 | 0.6364 | 0.7955 | 0.7457 |

完整 12 组见运行目录中的 `aggregation_summary.csv`。相比动态块 Max，AST Max 的 MAP@10 提高 0.06467，MRR@10 提高 0.04438；Top-1 从 18/44 提高到 21/44。按逐报告 AP@10，13 条改善、21 条不变、10 条变差。此结果仅描述当前单项目范围，尚不代表多项目或非 Python 数据集的表现。

`verification.json` 记录独立复算：原三路的 132 个样本路线的 Chunk、原始 ES 分数和三种聚合排名均与前次一致；全部 528 份逐报告聚合指标独立复算通过。ES 与 NumPy 最大 Chunk 分数误差约 `3.21e-7`，528 份前十排名核对全部一致。新增 AST 路线共 61,902 个报告—Chunk 分数，最长输入 512 tokens，最长前缀 64 tokens，所有文件源码均完整重建。

同时复用已验证的 BM25 原始结果，生成五路汇总 `results/comparison/20261016_112957_714562/`，其候选、查询和清单逐条核对通过。多语言端到端验证记录位于 `results/vector/multilingual/20261016_112835_750237/verification.json`，四路均完成十个语法样本的模型编码及 ES 检索。

## 轻量参数对比（2026-10-25）

这一轮不修改 AST 算法、模型或评分公式，只比较字符预算与结构前缀预算。总输入上限保持 512 tokens，主聚合 Max、无 RRF，固定同一份 44 条 requests 清单。1500/64 复用已核对的完整批次，其他四组真实重跑，每组均 44/44 成功。测试按顺序执行两个单因素小网格：先字符 1000/1500/2000，再在字符 1500 下比较前缀 32/64/96；没有搜索更多取值，也未搜索两参数组合。

| AST 字符预算 | 前缀 tokens 上限 | MAP@10 | MRR@10 | Top-1 | Top-5 | Top-10 | MeanRecall@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 64 | 0.5526 | 0.5853 | 0.4091 | 0.8864 | 0.9318 | 0.9190 |
| 1500 | 32 | 0.6121 | 0.6398 | 0.4545 | 0.8636 | 0.9318 | 0.9176 |
| **1500（原配置）** | **64** | **0.6297** | **0.6526** | **0.4773** | **0.8636** | **0.9545** | **0.9418** |
| 1500 | 96 | **0.6351** | **0.6636** | **0.5000** | 0.8636 | 0.9545 | 0.9418 |
| 2000 | 64 | 0.6129 | 0.6444 | 0.4773 | 0.8409 | 0.9545 | 0.9304 |

这批数据中，扩大前缀到 96 是唯一取得更高 MAP/MRR 的参数调整：MAP 增加 0.00537（0.54 个百分点），MRR 增加 0.01105（1.10 个百分点），Top-1 从 21/44 增至 22/44。逐报告 AP 为 **1 胜、42 平、1 负**；改善来自 #1944，AP 从 0.39286 提至 0.64286，退化为 #2317，AP 从 0.125 降至 0.11111。

配对 bootstrap 20,000 次、seed=20261025，MAP 差值的描述性 95% 区间为 `[-0.000947, 0.017045]`，MRR 为 `[-0.000947, 0.034091]`，均包含 0。44 条同项目数据此前已被反复观察，本轮又比较了多个参数，不能把其中最高值当作独立测试集上的可靠提升，也未作多重比较校正。因此 **96 只作为候选配置，默认仍保留 1500/64**。参数调整本身不构成新的切块算法或已证实的论文创新。

### 成本与机制检查

| 字符 / 前缀 | 报告—块实例数 | 输入 token 中位数 | 小于128 tokens 的比例 | 被截断的结构前缀数 |
| --- | ---: | ---: | ---: | ---: |
| 1000 / 64 | 79,503 | 223 | 22.72% | 1,461 |
| 1500 / 32 | 61,668 | 363 | 20.72% | 9,871 |
| 1500 / 64 | 61,902 | 374 | 15.15% | 2,031 |
| 1500 / 96 | 62,610 | 370 | 14.54% | 462 |
| 2000 / 64 | 55,606 | 437 | 13.90% | 2,333 |

前缀 96 使被截断的前缀从 2,031 降至 462，代价是块实例增加约 1.14%。字符 2000 则减少约 10.17% 的块实例，但本批 MAP/MRR 下降；字符 1000 产生更多短块，效果下降更明显。块实例重复计算多个报告快照中的相同源码，不能当成唯一向量数。原配置缓存已热、其他配置混合首次计算和复用，不作公平耗时倍数比较。

`prefix_max_tokens` 不只是文本截断参数：现有安全拆分还预留 `prefix_max_tokens + 8` 的空间，所以它也会改变超长正文的边界。#1944 中，真实文件 `requests/sessions.py` 的最佳块及余弦分数保持不变；原来排名第 1 的错误候选 `requests/packages/urllib3/response.py`，同一正文块的前缀从截断的 64 tokens 变为完整的 69 tokens，相似度由 0.675856 降至 0.660426，使真实文件升到第 1。不能简单概括为“所有块加长前缀都会更准确”，也不能把这两种机制混为一种已验证因果结论。

已核对各组模型身份、实现哈希、候选配置、报告/查询/标签清单相同，全部逐块/文件 ES 分数和三种聚合前十顺序核对通过。所有指标由逐报告记录重新汇总后与原始汇总一致。运行与审计记录保存于 `results/vector/ast_budget_sweeps/20261025_184818/`，包含实验前计划、各次日志、前缀配置，以及 `parameter_analysis.json/csv`；各批次完整排名与块数据仍在原结果目录。

### 复现候选配置

项目提供独立的 `configs/experiments/ast_prefix96_candidate.json`，仅运行原 AST，字符预算 1500、总输入 512、前缀上限 96、44 条 requests，不改变默认 vector 配置：

```bash
python -B -u scripts/retrieval/run_vector_batch.py --config configs/experiments/ast_prefix96_candidate.json
# 若已有清单，可在同一条命令追加 --selection <selection.json路径>
```

字符预算实验直接使用原 CLI，无需新增配置或依赖：

```bash
python -B -u scripts/retrieval/run_vector_batch.py --routes blagent_ast --no-fusion --limit 44 --ast-max-chars 1000
python -B -u scripts/retrieval/run_vector_batch.py --routes blagent_ast --no-fusion --limit 44 --ast-max-chars 2000
```

若需复现前缀 32，复制候选 JSON 并仅将 `prefix_max_tokens` 改为 32。下一步应在预先固定的其他项目验证候选 96，同时保留 64 基线；本轮不追加参数网格。

实际可影响 AST 的参数包括 `ast_max_chars`、`max_tokens`、`prefix_max_tokens`；总输入上限还会改变查询截断，不能把窗口变长后的效果全部归因于切块。`embedding_batch_size` 主要控制速度/显存，`dynamic_max_lines` 和 `fixed_chunk_tokens` 不控制 AST 切块，Max 聚合不使用 `aggregation_temperature`，这些都不宜当作本轮准确率调参项。
