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
