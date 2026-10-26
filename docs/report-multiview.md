# 模块2：面向检索收益的多视角报告查询

后续固定参数的新44条跨项目测试见 [Seaborn / Flask / pytest复核](report-multiview-crossproject.md)。

第一版无需训练，独立于现有默认流程。生成模型只读取报告字段，不能接触修复补丁、真实缺陷文件、候选源码、检索结果或指标。本地 SWE-bench 格式化数据继续只使用 `body`，其中已有原始问题标题；独立 `title` 可能是修复 PR 标题，不启用。

## 固定实验协议

| 通道 | 查询 | 检索 |
| --- | --- | --- |
| q0 | 原始报告，保留原查询通道 | AST96，Max |
| q1 | 原文中的标识符、异常、路径、错误文本 | BM25 |
| q2 | 触发条件、实际行为、明确写出的期望行为 | AST96，Max |

每份报告一次模型调用，抽取四组结构化条目：`identifiers / trigger / actual / expected`。每个条目包含 `source` 和逐字 `text`。程序在原文中校验，并记录字符起止位置（Python Unicode 字符偏移，非字节偏移；同一引文重复出现时记录首次位置）。第一版行为描述也采用摘录，只有 Trigger/Actual/Expected 分组标签由程序补充。

没有逐字出处的条目被剔除，原始响应、剔除条目和原因都保留；不做模糊匹配，不自动修复标识符，不推断根因。结构错误、输出截断及调用失败明确报错。原文过长拒绝生成，不静默截断。摘录存在不代表其语义分类一定正确，仍需人工检查。

融合为 `sum(w_j / (60 + rank_j))`，初始权重均为1；原 q0 权重必须大于0。某视角为空时，其排名为空、贡献为0，不复制 q0、不重新归一化权重。零权重通道不能引入新候选。BM25 使用全部匹配文件，语义使用完整文件排名；缺席通道贡献0。同分以文件路径排序。

同时输出原文 BM25（`q0_bm25`）、原文 AST96（`q0`）以及两者等权 RRF（`original_rrf`），用于区分查询增强和混合检索的影响。主要比较最终 `multiview_rrf` 与这些原文对照，均使用现有 @10 指标函数。

非默认权重必须在配置填写 `weight_development_reference`，记录开发集及调节依据。程序只能要求记录，不能自动证明数据集互不重叠；不能在当前44条测试报告上搜索权重再宣称独立测试收益。

## 文件

- `src/bug_localization/report_views.py`：提示词、结构约束、原文校验、生成缓存与加权 RRF。
- `src/bug_localization/multiview.py`：准备查询和检索两个阶段的调度，复用已有 BM25、AST、编码、ES 检索与评价函数。
- `configs/experiments/report_multiview.json`：固定实验参数，引用已有 AST96 配置。
- `scripts/retrieval/run_multiview.py`：两阶段命令行入口。
- `scripts/llm/serve_report_llm.py`：启动用户已安装的 llama.cpp，不下载软件或模型。
- `scripts/analysis/analyze_multiview.py`：重算融合和评价，核对证据，并与历史 AST96 完整排名比较。
- `tests/test_report_views.py`、`tests/test_multiview.py`：证据、缓存、融合及调度测试。

没有新增 Python 第三方依赖，沿用当前 requirements。GGUF 服务是外部运行时，不写入 pip 依赖。

## 运行：先生成，释放显存，再检索

当前模型为用户提供的 `Qwen3.5-9B-Q8_0.gguf`，SHA256：

```text
809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4
```

本机使用用户提供的 llama.cpp build10549、commit b2e5e9b28，服务程序和 DLL 存在 `.runtime/llama/`。模型文件继续保留在用户原目录，不复制进 Git。Windows 二进制不能复制到 Ubuntu 使用；Ubuntu 请自行准备对应平台的 llama.cpp。服务接口使用 [llama.cpp 官方文档](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)中的 chat completions 和结构化 JSON 输出。

在项目根目录、激活虚拟环境后，Windows PowerShell 配置：

```powershell
$env:IRBL_LLM_FILE = '你的GGUF模型完整路径'
$env:IRBL_LLAMA_SERVER = (Resolve-Path '.runtime/llama/llama-server.exe').Path
python scripts/llm/serve_report_llm.py
```

本机已记录的 `.runtime/llama/local-runtime.json` 可作为模型位置的本地默认值；它不上传 Git，也不能作为服务器的路径配置。启动器会检查模型 SHA256 与实验配置是否一致。

Ubuntu Bash：

```bash
export IRBL_LLM_FILE=/your/models/Qwen3.5-9B-Q8_0.gguf
export IRBL_LLAMA_SERVER=/your/llama.cpp/build/bin/llama-server
python scripts/llm/serve_report_llm.py
```

默认只监听 `127.0.0.1:8080`，8192上下文、单请求槽位、关闭 thinking。需要后台服务时追加 `--background`：日志写入 `results/llm/`，PID、版本、命令和模型哈希写入本机 `.runtime/llama/local-runtime.json`。服务是否就绪以日志为准；本模块不管理已有的其他推理服务。

在另一个终端生成查询（开始时可加 `--limit 1`）：

```bash
python -B -u scripts/retrieval/run_multiview.py --prepare
```

已有基准选择清单时，两个阶段使用同一个 `--selection`，例如本机当前44条：

```bash
python -B -u scripts/retrieval/run_multiview.py --prepare --selection results/comparison/20261016_112957_714562/selection.json
```

确认全部生成成功后，在 LLM 服务终端按 Ctrl+C 释放显存。后台服务则按保存的 PID 核对进程后停止；不要按进程名批量结束其他人的服务。启动既有 ES，再执行：

```bash
python -B -u scripts/retrieval/run_multiview.py --run --selection results/comparison/20261016_112957_714562/selection.json
```

`--run` 只使用本地生成缓存，不会调用 LLM；缓存缺失会在加载 Embedding 模型前报错。Embedding 模型每次批次只创建一个实例，q0/q2 共用代码向量和索引，查询编码分别缓存。源码索引、切块和指标算法未改动。

接口地址可通过 `IRBL_LLM_URL`（包含 `/v1`）覆盖，模型别名通过 `IRBL_LLM_MODEL` 覆盖，可选密钥只从 `IRBL_LLM_API_KEY` 读取，不写入配置或结果。更换模型文件或推理版本时，更新实验配置的 `model_revision`，不要沿用旧模型的缓存身份。远程服务器可用 SSH 端口转发；不需要公开服务端口。

## 产物与复核

每次运行独立保存到 `results/multiview/<UTC时间>_prepare/` 或 `_retrieval/`，不覆盖旧批次：

- `selection.json`、`effective_config.json`：报告选择、模型与检索配置、实现哈希。
- 准备阶段：每条 `*_queries.json`，原文、响应、证据、三个查询、模型身份、token用量及缓存状态。
- 检索阶段：`cases/` 内的查询、候选、源码块、ES Chunk 分数、各通道完整文件排名和融合贡献。
- 每个通道的 `samples.json/csv`、`summary.json/csv`：逐报告和总体指标。
- `status.json`：运行状态与错误；失败样本不从评价分母中删除。

生成缓存位于 `cache/report_queries/`，按报告内容、提示词、结构约束、模型修订与生成参数区分。证据校验失败的首次响应可离线重新校验；保留历史拒绝记录，不能从多次生成里挑选效果最好的一份。不要并发执行同一缓存配置的准备任务。

```bash
python -B scripts/analysis/analyze_multiview.py results/multiview/实际检索批次 --reference results/vector/batches/原AST96批次
python -B -m unittest discover -s tests -q
```

## 2026-10-26：44条真实实验

准备结果：`results/multiview/20261026_055159_781931_prepare/`。
检索结果：`results/multiview/20261026_055415_531357_retrieval/`。
选择清单 SHA256：`b14ba40016f54793c6fdada6ef996a7cd3cb343f80c19f627e39ee0a804d24fc`。

| 路线 | MAP@10 | MRR@10 | Top-1 | Top-5 | Top-10 | MeanRecall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原文 AST96（q0） | **0.6351** | **0.6636** | **50.00%** | 86.36% | 95.45% | 0.9418 |
| 标识符 BM25（q1） | 0.3773 | 0.4234 | 31.82% | 52.27% | 72.73% | 0.6866 |
| 行为 AST96（q2） | 0.5201 | 0.5650 | 38.64% | 79.55% | 86.36% | 0.8153 |
| 原文 BM25 | 0.4596 | 0.5080 | 27.27% | 81.82% | 93.18% | 0.8632 |
| 原文 BM25 + AST96 RRF | 0.5921 | 0.6327 | 43.18% | 86.36% | **100.00%** | **0.9669** |
| 三视角等权 RRF | 0.5223 | 0.5468 | 31.82% | 81.82% | 93.18% | 0.9190 |

六个输出通道均44/44成功，指标分母不因空查询而改变。相比原 AST96，三视角融合 MAP下降0.1127，逐报告AP为7条提升、20条相同、17条下降；相比原文混合RRF，MAP下降0.0698，8条提升、16条相同、20条下降。

这版模块完成了功能接入，但没有得到检索收益，不能作为已经验证有效的改进方法。当前44条属于此前已经观察过的单项目探索集，不是留出测试集。

### 实际观察

- 保留641条有逐字出处的证据，剔除81条无效摘录；6条 q1 为空，q2均非空。缺失来源不能通过模型改写或模糊匹配补齐。
- #1657 的 q1 为空，q2只剩“one simply overwrites the other”，脱离了关键上下文，融合后AP从1降到0.2。
- #1766 的 q2抽取了 curl 的行为描述，未能保留足够的缺陷定位信息，AP从1降到0.125。这说明逐字存在不等价于检索有用。
- #1888 抽取了大量第三方包、环境与调用栈信息，融合后正确文件跌出Top-10。#3718则AP提升0.5，说明不同报告的收益并不一致。
- 此轮没有训练选择器、没有生成猜测文件名、没有根据指标调整提示词或权重。查询生成前的接口冒烟修复了JSON结构约束；检索前将非逐字条目改为离线剔除，复用了首次响应，没有为失败摘录重新向LLM请求答案。

### 验证与成本

- 96项单元测试通过；其中调度测试的虚构分数仅用于临时测试目录，不计入上述真实结果。
- 44条原 q0 的查询、候选与完整文件排名和历史 AST96 对照一致；完整排名和指标均44/44匹配。
- 审计重新计算六路指标及两种RRF，校验每条接受证据的原文偏移，并确认 q0/q2 使用同一ES索引。
- ES 的 Chunk/文件分数由现有本地计算独立校验。
- 已接受的44份模型响应共32,194输入 tokens、18,899输出 tokens（不含接口冒烟失败请求）。
- 检索阶段约40.08秒，利用原代码向量和索引缓存，仅为新的行为查询编码；这是热缓存耗时，不能当作冷启动速度。
- 生成模型在检索前已停止，Embedding模型随后单次加载；未同时占用两份模型的显存。

复核本次结果：

```bash
python -B scripts/analysis/analyze_multiview.py results/multiview/20261026_055415_531357_retrieval --reference results/vector/batches/swe_bench_formatted_test_20261025_190040_327947
```

完整校验写入检索批次的 `verification.json`。生成缓存和实验产物由 `.gitignore` 排除，复核时需单独备份；运行中的新实验不覆盖本次结果。

## 查询互补性复核

复用上述44条完整排名，无新增LLM或检索调用。命中定义是Top-K内至少有一个真实缺陷文件。“补救”表示原查询未命中而增强查询命中；“损失”表示原查询命中而增强查询未命中。

| 相对原文 AST96 | Top-1 补救 / 损失 | Top-5 补救 / 损失 | Top-10 补救 / 损失 |
| --- | ---: | ---: | ---: |
| q1 标识符 BM25 | 6 / 14 | 3 / 18 | 1 / 11 |
| q2 行为 AST96 | 4 / 9 | 1 / 4 | 1 / 5 |
| q1、q2任一路命中 | 10 / 6 | 4 / 2 | 2 / 3 |

最后一行是两个查询的命中事件取“或”，不代表一个实际融合排名，也不表示其总候选数量仍限于K。以逐报告AP比较，q1有9条改善、7条不变、28条下降；q2有11条改善、16条不变、17条下降。

两条Top-10补救案例的首个正确文件排名：

| 报告 | 原文 AST96 | q1 | q2 | 原文 BM25 | 三视角融合 |
| --- | ---: | ---: | ---: | ---: | ---: |
| #2153 | 15 | 16 | 9 | 4 | 14 |
| #2873 | 15 | 5 | 20 | 4 | 12 |

增强查询确实补救了AST96的两个漏报，但原文BM25已经覆盖它们；当前融合还丢掉了增强通道的补救效果。相对原文AST96与原文BM25两路的命中并集，增强查询新增Top-10报告数为0，新增Top-5报告数为1（#2317）。这是报告级命中结论，不代表所有真实文件都已找齐：增强查询仍增加了2个“报告—真实文件”Top-10命中对。

头部排序有更明确的互补：#1376、#2413、#2466、#2617、#2674、#3718 共6条报告，在两种原文检索中都不是首位命中，而q1或q2能够首位命中。例如#2674，原AST96第7、原BM25第2、q1第1；#2466，原AST96第2、原BM25第24、q2第1。

### 事后理想选择的诊断上限

假设已知真实缺陷文件，为每条报告选择AP@10最高的整条排名（相同AP优先原文），得到：

| 可选择的通道 | 理想选择 MAP@10 |
| --- | ---: |
| 只用原文 AST96 | 0.6351 |
| 原文 AST96 / q2 | 0.6887 |
| 原文 AST96 / 原文 BM25 | 0.7036 |
| q0 / q1 / q2 | 0.7563 |
| 两种原文通道 / q1 / q2 | 0.7735 |

这些是使用真实标签计算的事后上限，**不是实际可实现的选择器成绩**，也不是所有可能融合算法的上限。程序另存各指标独立取最大值的上限；不同指标可能选择不同通道，不能混称同一个系统的实测结果。

新增查询相对两种原文通道的理想选择，MAP上限提高约0.0700。它说明存在额外的头部排序信号，但没有证明能从报告或检索分数中提前识别这些案例。q1与q0还改变了检索器，不能把两者的全部差异归因于报告增强；q0与q2则使用相同的AST96检索器。

结论：不能根据平均指标下降就判定模块2完全不可行；它在当前探索集上有互补信号，主要是重排而不是新增报告覆盖。当前无条件等权融合尚不能利用这些信号。若后续研究选择器，应在独立开发集上验证选择规则，并同时比较“不使用LLM、只在两种原文通道间选择”的对照；本轮不继续拟合当前44条报告。

复核命令：

```bash
python -B scripts/analysis/analyze_query_complementarity.py results/multiview/20261026_055415_531357_retrieval
```

输出同一结果目录下的 `complementarity.json`（每种补救/损失的报告ID、额外真实文件、指标上限）和 `complementarity_samples.csv`（逐报告六路指标及真实文件排名）。
