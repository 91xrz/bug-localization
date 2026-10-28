# Bug Localization

基于 Elasticsearch 的文件级缺陷定位研究项目：**BM25 + 动态块 / 方法块 / 固定块 / BLagent 风格 AST 块**，使用同一个 Qwen Embedding 模型，无需微调。当前通过 Python CLI 运行，不包含 Web UI、交互启动器或 HTML 报告工具。

本文复现目标：**SWE-bench 格式化版中 `psf/requests` 的全部 44 条报告，五条检索路线，@10 指标，关闭 RRF**。语义检索还会输出四种分块 × 三种文件聚合的 12 组指标。这不是完整 SWE-bench 或 BLAZE 论文成绩。

目录导航：常规配置在 `configs/retrieval/`，研究实验配置在 `configs/experiments/`；运行入口在 `scripts/retrieval/`，已有结果的分析脚本在 `scripts/analysis/`。各文件用途见 [项目文件说明](docs/project-layout.md)。近期实验配置和分析脚本已迁移至上述目录，旧命令请使用本文或实验文档中的新路径。

**Bench4BL 已接入 BM25、Java AST96 和模块2。** 本机已转换46个项目、9,459份报告，运行前启动原有ES，然后执行：

```bash
python scripts/environment/verify_environment.py --check-bench4bl
python scripts/retrieval/run_retrieval_comparison.py --config configs/retrieval/bench4bl/comparison.json
```

默认验证CSV项目5条报告。Bench4BL使用作者发布版本分组，区别于下文SWE-bench的逐缺陷修复前提交；范围、缺失标签与另一台机器的准备步骤见 [Bench4BL使用说明](docs/bench4bl.md)。

当前实验服务器为 Ubuntu x86_64，项目目录是 `/home/xrz/IRBL/bug-localization`。Python 3.12 虚拟环境、数据、Qwen 模型和 Elasticsearch 9.5.5 已准备；2026-10-28 已完成 requests 44 条对比。默认使用两块可见 GPU 和 8 个分块进程，具体资源应遵守服务器分配。所有命令从项目根目录执行。

## 1. 服务器上的文件

| 路径 | 用途 |
| --- | --- |
| `.venv/` | 本服务器 Python 环境 |
| `datasets/` | SWE-bench、Bench4BL 等数据，不随 Git 上传 |
| `models/Qwen3-Embedding-0.6B/` | 语义检索模型，本地加载 |
| `repos/` | 按固定提交保存的源码快照 |
| `.runtime/elasticsearch/` | Elasticsearch 程序、配置和索引数据 |
| `cache/` | 分块与向量缓存，删除后需要重新计算 |
| `results/` | 结果、日志与实验配置记录 |

2026-10-28 的 44 条实验结果位于 `results/comparison/20261028_182534_372429/`；结果、数据、模型、快照和缓存均被 Git 忽略，需单独备份。

## 2. 进入环境

```bash
cd /home/xrz/IRBL/bug-localization
source .venv/bin/activate
python -m pip check
nvidia-smi
```

如需重建环境，使用 Python 3.12 创建 `.venv` 并安装 `requirements.txt`；不要覆盖仍需保留的现有环境。固定直接依赖包括 PyTorch `2.11.0+cu128`。`configs/environment/requirements.lock.txt` 是原 Windows 环境记录，不是服务器当前锁文件。服务器实测依赖记录见 `results/reproduction/requests44-20261028/requirements.txt`。

## 3. 确认数据与模型就绪

当前服务器中的数据和模型目录应包含：

```text
datasets/swe_bench_formatted/dataset_dict.json
datasets/swe_bench_formatted/test/data-*.arrow
datasets/swe_bench_formatted/test/dataset_info.json
datasets/swe_bench_formatted/test/state.json
models/Qwen3-Embedding-0.6B/model.safetensors
models/Qwen3-Embedding-0.6B/config.json
models/Qwen3-Embedding-0.6B/tokenizer.json
models/Qwen3-Embedding-0.6B/tokenizer_config.json
```

运行以下验证，检查本轮实际需要的数据和基准模型文件：

```bash
python - <<'PY'
import ast
import hashlib
from pathlib import Path
from datasets import load_from_disk

data = load_from_disk("datasets/swe_bench_formatted")["test"]
assert len(data) == 2294, f"test 行数不匹配：{len(data)}"
rows = [row for row in data if row["repo_name"] == "psf/requests"]
assert len(rows) == 44, f"requests 报告数不匹配：{len(rows)}"
assert all(len(row["before_fix_sha"] or "") == 40 for row in rows)
assert all(isinstance(ast.literal_eval(row["updated_files"]), list) for row in rows)
expected = {
    "model.safetensors": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    "config.json": "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
    "tokenizer.json": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    "tokenizer_config.json": "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
}
for name, checksum in expected.items():
    with (Path("models/Qwen3-Embedding-0.6B") / name).open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == checksum, name + " 与本机基准不同"
print("通过：2,294 条数据，44 条 requests 报告，四个模型文件与基准一致")
PY
```

模型哈希取自原始实验的 `effective_config.json`，用于识别是否使用了同一版本。模型运行时仅从本地读取，不会自动下载。

<details>
<summary>没有本机资产时：从公开来源重新准备</summary>

数据来源为 [Partha117/swe_bench_formatted](https://huggingface.co/datasets/Partha117/swe_bench_formatted)，不是字段不同的 SWE-bench 官方原始版。固定修订后转换为项目要求的本地格式；目标目录应尚不存在：

```bash
python - <<'PY'
from pathlib import Path
from datasets import load_dataset
target = Path("datasets/swe_bench_formatted")
assert not target.exists(), "目标已存在，请先确认已有数据，避免覆盖"
data = load_dataset("Partha117/swe_bench_formatted",
                    revision="dffcc648c8b2fda4b0056fa67f841e8aaabb0a92")
assert set(data) == {"test"} and len(data["test"]) == 2294
data.save_to_disk(str(target))
PY
```

新序列化的数据元数据文件可能与原始拷贝不同；使用同一服务器重新生成的样本清单进行配对，不要混用 Windows 的旧清单。字段、样本顺序和标签仍需核对。[Datasets 加载说明](https://huggingface.co/docs/datasets/loading)。

模型来源为 [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)。可以在可联网机器完整下载，再传到指定模型目录。若用 SDK 下载，先将远端修订解析为具体 commit 并记录：

```bash
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo = "Qwen/Qwen3-Embedding-0.6B"
target = Path("models/Qwen3-Embedding-0.6B")
assert not target.exists(), "目标已存在，请避免覆盖"
revision = HfApi().model_info(repo).sha
snapshot_download(repo_id=repo, revision=revision, local_dir=str(target))
out = Path("results/reproduction")
out.mkdir(parents=True, exist_ok=True)
(out / "model_source.json").write_text(
    json.dumps({"repo": repo, "revision": revision}, indent=2), encoding="utf-8")
PY
```

重新下载后仍须执行上面的四文件哈希检查；上游最新版本不保证等于本机基准。哈希不同应先取回基准文件或核实正确历史修订，不能悄悄替换。[Hub 下载说明](https://huggingface.co/docs/huggingface_hub/guides/download)。

</details>

## 4. 启动本服务器的 Elasticsearch

当前项目专用 Elasticsearch 9.5.5 已安装在 `.runtime/elasticsearch/`，配置只监听 `127.0.0.1`，不启用认证。启动前先检查是否已有服务：

```bash
curl --fail --max-time 3 http://127.0.0.1:9200
```

如果已响应，确认 `cluster_name` 为 `bug-localization-local` 且版本为 9.5.5；不要启动第二个实例，也不要停止其他人的服务。没有服务时，在项目根目录以普通用户启动：

```bash
ss -ltn | grep -E ':(9200|9300)\b' || true
ES_PATH_CONF="$PWD/.runtime/elasticsearch/config" \
  "$PWD/.runtime/elasticsearch/elasticsearch-9.5.5/bin/elasticsearch" \
  -d -p "$PWD/.runtime/elasticsearch/elasticsearch.pid" \
  -Epath.data="$PWD/.runtime/elasticsearch/data" \
  -Epath.logs="$PWD/results/elasticsearch/server"
curl --fail --retry 30 --retry-connrefused --retry-delay 2 \
  'http://127.0.0.1:9200/_cluster/health?wait_for_status=green&timeout=60s'
```

等待响应中的 `status` 为 `green` 且 `timed_out` 为 `false`，再运行实验；仅 HTTP 端口就绪时索引仍可能正在恢复。日志位于 `results/elasticsearch/server/`。当前节点采用本地无认证配置，不要暴露到公网或覆盖共享服务。

安装包和 SHA512 文件保存在 `.runtime/downloads/`；如果必须重建，先确认现有数据已备份，然后参考 [Elasticsearch 说明](docs/elasticsearch.md) 和本服务器的 Linux 压缩包。不要重复解压覆盖正在使用的安装或数据。多次实验串行运行：当前 CLI 没有并发写入锁。

## 5. 环境与单元测试

默认向量实验配置使用两块可见 GPU，编码总批次为 64（通常每卡 32）。选择实验室分配的两块 GPU；有调度系统时遵守其设备分配，不要覆盖调度器已设置的 `CUDA_VISIBLE_DEVICES`：

```bash
export CUDA_VISIBLE_DEVICES=0,1
python scripts/environment/verify_environment.py --semantic
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA 不可用"
assert torch.cuda.is_bf16_supported(), "当前 GPU 不支持本项目要求的 BF16"
print("torch:", torch.__version__, "CUDA:", torch.version.cuda,
      "GPU:", torch.cuda.get_device_name(0))
PY
python -B -m unittest discover -s tests -v
python -m pip check
```

当前测试共 **103 项**；完整依赖安装后应全部通过，注意不要忽略因缺少语法包而出现的 skip。测试不依赖运行中的 ES。环境检查记录写入 `results/environment/verification.json`。

这里只准备 SWE-bench 时，**不要加 `--check-data`**：该开关要求 BeetleBox、SWE-bench、Ye 三份数据同时存在。只复现 44 条结果不需要另外两份数据。

## 6. 先跑一条样本，确认整条流程

```bash
python -u scripts/retrieval/run_retrieval_comparison.py --limit 1 --repository psf/requests
```

此入口依次运行 BM25 和四条语义路线，固定相同样本、修复前提交及候选文件，并生成三种文件聚合结果。默认不训练模型、不运行 RRF。第一次缺少缓存时会下载源码快照、编码代码和建立 ES 索引；正常情况下不下载模型。

预期选到 issue `1142`，五路均成功，控制台打印 `Comparison results: ...`。先核对结果成功，再扩大到 44 条；不要将这一条的高分当作正式实验结论。

## 7. 正式运行 44 条，避免 SSH 断线中断

建议在实验室允许的计算节点上使用 tmux；若必须通过 Slurm/PBS 分配资源，请在作业分配内执行本节命令，不能在登录节点直接占用 GPU。

```bash
tmux new -s irbl-44
```

进入会话后，切换到实际项目根目录并激活环境，再执行：

```bash
source .venv/bin/activate
# 仅在没有调度器分配设备时设置；数字应替换为自己获准使用的 GPU。
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p results/reproduction
TASK_RUN=$(date +%Y%m%d_%H%M%S)
git rev-parse HEAD > "results/reproduction/$TASK_RUN.commit.txt"
git status --short > "results/reproduction/$TASK_RUN.git-status.txt"
git diff > "results/reproduction/$TASK_RUN.git-diff.patch"
python -m pip freeze > "results/reproduction/$TASK_RUN.requirements.txt"
nvidia-smi > "results/reproduction/$TASK_RUN.gpu.txt"
set -o pipefail
python -u scripts/retrieval/run_retrieval_comparison.py --limit 44 --repository psf/requests \
  2>&1 | tee "results/reproduction/$TASK_RUN.log"
TASK_EXIT=$?
printf '%s\n' "$TASK_EXIT" > "results/reproduction/$TASK_RUN.exit-code.txt"
```

按 **Ctrl+B，然后 D** 分离会话，之后可断开 SSH；回来用 `tmux attach -t irbl-44`。查看日志可用 `tail -f results/reproduction/<实际时间戳>.log`。退出码应为 0，且结果无失败。tmux 不提供崩溃恢复；强制停止后不能续写原批次，但重新运行会复用有效缓存并创建新的结果目录。

当前批量入口限制 **1–50 条**；这一轮是 44 条复现，不是全数据集长跑。扩大到更多样本需另行调整并验证，不要误把 `--limit 50` 当成全量实验。

## 8. 查看结果与验收

综合结果位于控制台打印的 `results/comparison/<run_id>/`：

| 文件 | 用途 |
| --- | --- |
| `summary.json/csv` | 五路主指标、成功/失败数、原始批次路径 |
| `samples.json/csv` | 44 × 5 = 220 条样本路线指标 |
| `selection.json` | 本次样本顺序、提交及查询/标签指纹 |
| `verification.json` | 配对、查询与候选一致性验证 |
| `effective_config.json`、`status.json` | 配置、阶段和状态 |

`summary.json` 的 `runs.semantic` 指向 `results/vector/batches/<run_id>/`，其中：

- `aggregation_summary.csv`：四种切块 × 三种聚合，共 12 组；
- `aggregation_samples.csv`：44 × 4 × 3 = 528 条逐报告记录；
- `cases/`：真实查询、候选、分块、所有 Chunk 分数及文件排名；
- `effective_config.json`：模型文件哈希、设备、依赖版本及实现指纹。

可执行以下验收脚本（选择最新综合批次；有多个实验时建议把 `folder` 改成目标路径）：

```bash
python - <<'PY'
import json
from pathlib import Path
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
folder = max(Path("results/comparison").glob("*/summary.json"),
             key=lambda p: p.stat().st_mtime).parent
report = read(folder / "summary.json")
assert report["status"] == "completed"
assert report["selected_samples"] == 44 and report["evaluation_k"] == 10
assert set(report["routes"]) == {"bm25", "dynamic", "method", "fixed", "blagent_ast"}
assert all(v["successful_samples"] == 44 and v["failed_samples"] == 0
           for v in report["routes"].values())
assert len(read(folder / "samples.json")) == 220
verification = read(folder / "verification.json")
assert verification["candidate_and_query_verified_samples"] == 44
assert verification["same_sample_order_and_snapshots"] and verification["metric_cutoffs_match"]
assert verification["no_fusion"] and not verification["not_fully_verified_samples"]
assert report["selection_sha256"] == "b14ba40016f54793c6fdada6ef996a7cd3cb343f80c19f627e39ee0a804d24fc"
semantic = Path(report["runs"]["semantic"].replace("\\", "/"))
assert len(read(semantic / "aggregation_summary.json")["routes"]) == 12
assert len(read(semantic / "aggregation_samples.json")) == 528
print("验收通过：", folder)
for route, values in report["routes"].items():
    print(route, values["metrics"])
PY
```

本服务器 2026-10-28 运行的 **Max 主聚合**结果（双卡、总批次 64；此次运行前已存在部分缓存）：

| 路线 | MAP@10 | MRR@10 | Top-1 | Top-5 | Top-10 | MeanRecall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.4596 | 0.5080 | 0.2727 | 0.8182 | 0.9318 | 0.8632 |
| 动态块 | 0.5711 | 0.6103 | 0.4091 | 0.8636 | 0.9545 | 0.9276 |
| 方法块 | 0.5349 | 0.5797 | 0.4318 | 0.7727 | 0.9091 | 0.8949 |
| 固定块 400 tokens | 0.5463 | 0.5609 | 0.3409 | 0.8864 | 0.9318 | 0.9176 |
| BLagent AST | 0.6289 | 0.6521 | 0.4773 | 0.8636 | 0.9545 | 0.9403 |

服务器 GPU、内核和数值运算不同，不能承诺浮点分数逐位相同；先核对样本、模型哈希、参数和成功数，再比较排名与指标。ES/NumPy 分数核对沿用代码中的 `2e-5` 容差，这不等于允许任意指标差异。

基准保留了缺失候选标签：72 个标签中 71 个存在于候选，issue 2678 有一个标签不在修复前快照中；没有为了提高成绩删掉该报告。17 条语义查询被截断，细节见[评测说明](docs/evaluation.md)。主指标以全部选定报告为分母，失败计 0；不可只比较成功样本子集。

## 9. 单独运行或更改实验参数

```bash
# 仅 BM25
python scripts/retrieval/run_bm25_batch.py --limit 44 --repository psf/requests
# 四路语义，不融合
python scripts/retrieval/run_vector_batch.py --limit 44 --repository psf/requests --no-fusion
# 仅 AST；只验证流程时可将 limit 改为 1
python scripts/retrieval/run_vector_batch.py --routes blagent_ast --limit 44 --no-fusion
# 查看全部有效 CLI 参数
python scripts/retrieval/run_vector_batch.py --help
```

| 配置文件 | 主要内容 |
| --- | --- |
| `configs/retrieval/bm25.json` | 数据集、split、ES 地址、候选扩展名、BM25 参数及 top_k |
| `configs/retrieval/bm25_batch.json` | BM25 批次范围，默认 5 条 |
| `configs/retrieval/vector.json` | 模型、device、batch size、512/64 token 预算、固定块 400、AST 1500 字符、聚合及路线 |
| `configs/experiments/ast_prefix96_candidate.json` | 原 AST 的轻量调参候选：1500 字符、512/96 tokens、仅 AST、44 条；[对比与限制](docs/blagent-ast.md#轻量参数对比2026-10-25) |
| `configs/experiments/ast_dependency_experiment.json` | AST96 与调用依赖增强对照：每块最多2个函数、总输入仍为512 tokens；[方案、复现与44条结果](docs/dependency-ast.md) |
| `configs/experiments/report_multiview.json` | 模块2：LLM 一次抽取、q0/q1/q2 三视角查询及加权 RRF；[运行流程与限制](docs/report-multiview.md) |
| `configs/experiments/report_multiview_crossproject.json` | 模块2跨项目复核：Seaborn 22条、Flask 11条、pytest 11条；[固定协议与结果](docs/report-multiview-crossproject.md) |
| `configs/retrieval/comparison.json` | 基础/语义配置路径、仓库过滤、44 条、@10 |

`gpu_count` 指定使用前几块可见 CUDA 设备；`embedding_batch_size` 是所有设备合计的批次。只用一块卡时将 `gpu_count` 改为 1；显存不足时降低批次。批次和设备数量纳入向量缓存身份，变化后会重新编码，数值可能存在细微差异。ES 索引和报告处理仍可能让 GPU 等待，利用率不一定始终满载。不要随意修改分块长度、模型精度或评测 K。综合入口没有 `--device` 或 `--routes` 参数，需修改它引用的 vector JSON。使用已有 `selection.json` 时，语义入口支持 `--selection`，不能与 `--limit/--repository` 混用。

`chunk_workers` 设置按文件并行分块的进程数，默认 8；设为 1 使用串行分块。进程数应遵守实验室资源分配，增加进程不一定更快。依赖图路线 `ast_dependency` 仍按文件串行处理，ES 索引和报告顺序不受此设置影响。

新增的 token 预算 AST / 动态规划路线是独立实验配置，不改变上述四路默认基线。方案、消融对比及复现命令见 [AST token 实验](docs/ast-token-experiment.md)。

已有 BM25 和语义批次后，可用 `scripts/retrieval/fuse_rankings.py --bm25-run BM25_BATCH_DIRECTORY --semantic-run ast96=SEMANTIC_BATCH_DIRECTORY` 做文件级 RRF（运行时在前面加 `python`，并替换实际目录）。它复用已保存的语义排名，从已有 ES 索引补齐 BM25 匹配文件，不重新编码模型。[融合说明与排名结果](docs/hybrid-retrieval.md)记录了本次 @10 结果：Top-10 覆盖提高，但等权融合的 MAP/MRR 低于单独 AST。

若比较“每路最多一票、不按输入名次加权”，可用 `python scripts/retrieval/vote_rankings.py --input-run HYBRID_BATCH_DIRECTORY --depths all 10` 离线读取混合结果。本次纯投票同分很多，指标与同票处理有关；[投票说明](docs/hybrid-retrieval.md#不使用名次的路线投票2026-10-25)同时报告固定展示顺序与随机同票排序期望。

## 10. 常见问题

| 现象 | 处理 |
| --- | --- |
| CUDA 不可用 / BF16 不支持 | 检查分配的 GPU、驱动和 CUDA wheel；当前语义 CUDA 路线要求 BF16 |
| 显存不足 | 先确认其他进程占用，再降低 vector.json 的 embedding_batch_size，记录新配置 |
| ES 连接失败 / 集群名不符 | 核对服务、端口、日志和 bm25.json；不要连接不属于自己的共享节点 |
| 首次运行很慢 | 无缓存时需下载固定提交快照、编码、建索引；后续复用 |
| 服务器无法访问 GitHub | 从本机传完整 repos/psf/requests，包括 source、snapshot.json 和 snapshot.zip |
| 快照校验失败 | 检查传输是否改了源码换行或缺文件，不要删除校验逻辑 |
| 提示三份数据缺失 | 只准备 SWE-bench 时不要使用环境检查的 --check-data |
| pip 找不到固定版本 | 检查 Python 3.12、架构、索引和网络；先解决依赖来源，不任意换版本 |
| 结果含失败或低于基准 | 先查 cases/*error.json、运行日志、样本指纹和模型哈希，不能直接当算法退化 |
| 只看到旧结果 | 每次批次目录唯一；以本次日志的输出路径为准 |

服务器使用项目 `.venv/bin/python`；在 IDE 中将 Working directory 设置为项目根目录。

更多细节：[数据来源](docs/datasets.md) · [语义检索](docs/semantic-retrieval.md) · [文件聚合](docs/file-aggregation.md) · [多语言 AST](docs/blagent-ast.md) · [文件结构](docs/project-layout.md)。数据、模型、缓存、快照、环境和结果继续由 `.gitignore` 排除，实验结束后单独备份所需结果。
