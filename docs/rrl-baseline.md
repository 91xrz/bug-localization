# RRL 最小对比实验

更新：2026-11-15。本轮接入 *Reformulate, Retrieve, Localize: Agents for Repository-Level Bug Localization*（BoatSE 2026，ICSE workshop）的作者代理流程。论文、其他数据集相关工作及两项实测范围见 [外部论文对比基线](external-baselines.md)。

## 当前实验身份

最小测试使用 **RRL author agent + Qwen2.5-Coder-7B-Instruct Q4_K_M（CPU）**，验证报告改写、BM25、代理定位和自检能连贯执行。这是模型与运行后端适配，不是论文 30B/32B 配置的精度复现。

- [作者源码](https://github.com/gencau/boatse-query-reformulation/tree/50903acf3d6124514ff4f4c179cfd67a3e8ea414)，Apache-2.0，固定 commit `50903acf3d6124514ff4f4c179cfd67a3e8ea414`。
- [论文](https://arxiv.org/abs/2512.07022)，作者实验为 SWE-bench Lite 与 LCA。当前最小测试使用本地 SWE 格式化版 Requests 按原始顺序第一条报告，与 BLAZE 的首条功能检查对应；不宣称是完整 Lite 300 的复现。
- [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/tree/c03e6d358207e414f1eca0bb1891e29f1db0e242)，Apache-2.0，复用本地固定版本权重并转换为 Q4_K_M；不调用闭源 API。
- [llama.cpp b10549](https://github.com/ggml-org/llama.cpp/tree/b2e5e9b28b2484fbf94b543432ece638996a8b97)，MIT，固定 commit `b2e5e9b28b2484fbf94b543432ece638996a8b97`；模型转换器与推理程序对应。

作者代理配置使用 Qwen3-Coder-30B 或 Qwen2.5-Coder-32B、Ollama、16K context；作者抽取信息的发布流程还会读取提前生成的 Qwen3-Coder 改写 CSV。当前运行使用同一个本地 7B 模型现场完成抽取和后续代理步骤，保留作者提示词，不把已发布 CSV 的行号当成本地实例 ID。

## 接入内容

`src/bug_localization/rrl.py` 直接调用固定版本作者代码中的 `_run_turn`、`_turn3_loop` 和 `_self_evaluate_and_adopt`。运行顺序为：

1. 读取共同报告清单、修复前源码与候选过滤规则；SWE 仅使用 `body`。
2. 用作者 `AgentContextPrompt` 从当前报告生成结构化改写；写出原始请求、响应和查询校验值。
3. 执行作者第一轮与第二轮提示，调用 `extract_relevant`、BM25 工具。
4. 按作者循环处理文件查看、无效路径重试及最终排名；文件查看由模型选择，实际次数单独记录，不把允许调用工具等同于已查看文件。
5. 执行作者自检流程，导出合法、去重的前 10 个候选文件；最后才读取 gold 计算统一指标。

为每个报告创建私有源码副本和 Lucene 索引。模型只能读取共同源码候选，以及原源码中确实存在的 `README.md`；后者的校验值单独记录。不能访问补丁、gold、运行结果目录或修复后的仓库内容。原作者 `safe_repo_path` 未完整检查越界，本接入在调用边界补充路径和候选约束，源码 checkout 保持原样。

Lucene 使用作者的完整文件 JSON 表示：`id`、原始 `contents`、相对 `path`，默认 English analyzer，BM25 `k1=0.9, b=0.4`。保留作者 top-10/20/30 工具选择；由于作者 Pyserini 导入位置与当前版本不兼容，通过同一 Anserini Java 搜索类提供兼容接口。没有用本项目 Elasticsearch 的代码词法分析器替换它。

## 已声明的协议差异

| 项目 | 作者材料 | 当前最小测试 |
| --- | --- | --- |
| 模型 | 30B/32B，抽取 CSV 由指定作者模型预生成 | 本地 7B Q4_K_M，逐报告现场抽取；明确是小模型验证 |
| 推理后端 | Ollama | llama.cpp，本机 HTTP，CPU |
| 输出预算 | 代理程序设置 4,096；抽取设置 2,048 | 每次最多 1,024，最多 24 次调用，单次超时 600 秒 |
| context | 16,384 | 16,384；实际 tokenizer 计数，超限明确失败，不静默截断报告 |
| 输入 | 作者 LCA/SWE CSV 与数据准备 | 当前冻结报告、修复前 SHA、候选清单，SWE body-only |
| 候选 | 作者 helper 默认可排除测试文件 | 保持本项目共同候选规则，不按 gold 筛选 |
| 排名/评价 | 作者程序默认 top-5，支持修改 | 输出 top-10；使用项目 MAP/MRR/Recall@10、Top-1/5/10 |
| 依赖 | requirements 多项没有固定版本 | 隔离环境固定版本；完整环境和索引记录保存于结果 |

失败、预算耗尽、输出被 token 上限截断、未执行关键改写/检索阶段都不报为“跑通”；失败样本在主指标中计 0。原始调用与错误保留。即使单条命中，也只能证明该条流程和输出有效，不能用于论文优劣结论。

## 运行入口与产物

运行入口为 `scripts/retrieval/run_rrl.py`，配置为 `configs/retrieval/rrl.json`。准备脚本使用 `.runtime/rrl-deps` 与 `.runtime/rrl-model-deps`，不替换项目主 Python 环境；模型位于 `models/RRL-Qwen2.5-Coder-7B-GGUF/`，作者源码位于 `.runtime/external-baselines/rrl/`。

```bash
.venv/bin/python scripts/datasets/prepare_rrl_environment.py
.venv/bin/python scripts/datasets/prepare_rrl_model.py
```

模型准备完成后，在一个终端启动 CPU 服务（本机已有 b10549 程序）：

```bash
CUDA_VISIBLE_DEVICES='' .runtime/llama/b10549/llama-b10549/llama-server \
  -m models/RRL-Qwen2.5-Coder-7B-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8096 --alias rrl-local-7b \
  -c 16384 -np 1 -t 12 -ngl 0 --no-op-offload
```

服务加载完成后，在另一个终端执行：

```bash
.venv/bin/python scripts/retrieval/run_rrl.py --limit 1 --repository psf/requests
```

默认只处理一条。支持 `--selection` 绑定已有共同清单，不能与 `--limit` / `--repository` 混用。正式大批量前应先改为目标模型并冻结相应预算；当前 CPU 小模型配置不作为默认的论文正式对比设置。

结果位于 `results/rrl/<时间>/`：

- `selection.json`、`effective_config.json`、`provenance.json`：报告清单、配置、作者源码、实际服务模型与 tokenizer 身份。
- `cases/*/inputs.json`：允许进入方法的查询、候选与 README 校验值。
- `call_*.json`、`reformulation.json`、`author_logs/`：逐次模型请求/响应、现场改写和作者工具日志。
- `cases/*/result.json`：初始及自检后排名、实际查看文件、调用次数、索引、统一指标；失败则保存 `error.json`。
- `samples.json/csv`、`summary.json/csv`：完整分母、失败情况和汇总。

## 实际验证状态

2026-11-16：已在固定 500 条报告上完成 7B 适配版，前 3 条为 CPU、后 497 条为 GPU，合计 455 成功、45 失败。与 BLAZE 及项目方法的同清单指标、失败计分、成功子集敏感性及分项目比较见 [500 条外部基线对比](external-comparison-500.md)；GPU 服务在实验结束后已关闭。

### 2026-11-16：500 条任务切换到 GPU 1

同一 b10549 版本的 CUDA 服务已编译到 `.runtime/llama/rrl-cuda-build/`，固定使用原有 Qwen2.5-Coder-7B Q4_K_M 权重。服务通过 `CUDA_VISIBLE_DEVICES=1` 选择物理 GPU 1；在局部设备视图中它显示为 CUDA0。服务监听 `127.0.0.1:8097`，一次实际推理成功，显存占用约 6.5 GB。编译选项为 CUDA 架构 89、`GGML_CUDA_FA=OFF`，编译工具位于独立目录 `.runtime/rrl-build-deps/`。

500 条固定清单原始 SHA-256 为 `0eb6d67e8c56c1207ad7419e9b5b9e1b9f53f1a839a2df5c0fc7d285e903d2e8`。CPU 批次完成前 3 条（3 成功、0 失败）后停止，正在处理的第 4 条在 GPU 批次重新运行。剩余 497 条的清单及设备配置保存在 `results/external-comparison-500-gpu-20261116/rrl-remaining-selection.json`、`rrl-gpu-config.json`，分别记录原清单位置和校验值。

- CPU 前 3 条：`results/rrl/crossproject500-cpu-retry-20261116/`。
- GPU 剩余 497 条：`results/rrl/crossproject500-gpu1-remaining-20261116/`；其位置 1 对应原始清单位置 4。
- 运行窗口：`tmux attach -t irbl-external-500-gpu`，其中 `rrl-gpu-server` 为服务、`rrl-gpu` 为批次。

最终合并时必须按原清单位置配对，保留两个批次的失败和不同运行设备；不能将 497 条的单独汇总当作完整 500 条的成绩。GPU 不改变论文与本地 7B 模型、量化和后端之间的差异；跨设备数值差异仍需在结果中注明。

2026-11-15 已跑通 Requests #1142，**1/1 成功、0 失败**。产物在 `results/rrl/requests1-local7b-smoke-v2-20261115/`。

| 项目 | 实测结果 |
| --- | --- |
| 共同候选 | 70 个文件；Lucene 实际索引 69 个 |
| 空文件 | Lucene 自行跳过 1 个空 `__init__.py`；仍保留共同候选和 gold 分母，逐项写入索引审计 |
| 完整流程 | 现场抽取 → 代理选择抽取工具 → BM25 top-10 → 文件查看 → 排名 → 自检 |
| 模型调用 | 12 次，全部正常结束，无输出 token 截断 |
| 实际查看 | 6 个不同文件；包括 `requests/models.py`、`requests/sessions.py` 等 |
| 格式纠正 | 1 次；模型一度返回非 JSON 内容，作者重试逻辑成功纠正 |
| 自检后正确文件 | `requests/models.py`，排名第 2 |
| AP@10 / MRR@10 | 0.5 / 0.5 |
| Top-1 / Top-5 / Top-10 | 0 / 1 / 1 |
| Recall@10 | 1.0 |
| 单报告总耗时 | 约 128 秒，包含索引和代理流程，不含一次性模型转换、服务加载及预检 |

自检前正确文件排第 1，自检后排第 2；按完整方法的最终结果评价，没有挑选较好阶段作为成绩。首次尝试因索引空文件计数检查过严而失败，调试记录保留在 `requests1-local7b-smoke-20261115/`；修正后在新目录完整重跑。

5 项 RRL 回归测试通过，包括真实 Lucene 空文件审计与缓存重用、模型调用预算、上下文边界和源码读取范围。本轮启动的 CPU 服务已停止，模型、索引、原始响应和日志保留；再次运行需按上方命令启动服务。这一条只证明小模型适配流程可执行，不足以评价方法优劣。

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_rrl.py' -v
```
