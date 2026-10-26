# Bug Localization

基于 Elasticsearch 的文件级缺陷定位研究项目：**BM25 + 动态块 / 方法块 / 固定块 / BLagent 风格 AST 块**，使用同一个 Qwen Embedding 模型，无需微调。当前通过 Python CLI 运行，不包含 Web UI、交互启动器或 HTML 报告工具。

本文复现目标：**SWE-bench 格式化版中 `psf/requests` 的全部 44 条报告，五条检索路线，@10 指标，关闭 RRF**。语义检索还会输出四种分块 × 三种文件聚合的 12 组指标。这不是完整 SWE-bench 或 BLAZE 论文成绩。

目录导航：常规配置在 `configs/retrieval/`，研究实验配置在 `configs/experiments/`；运行入口在 `scripts/retrieval/`，已有结果的分析脚本在 `scripts/analysis/`。各文件用途见 [项目文件说明](docs/project-layout.md)。近期实验配置和分析脚本已迁移至上述目录，旧命令请使用本文或实验文档中的新路径。

代码基线：`5760e1cf78d13782dfe4105dc126c7e40d09086e`（2026-10-16 核对 GitHub main）。本机已验证 Windows / Python 3.12.14 / RTX 5080；下文提供 Ubuntu GPU 服务器部署步骤，**尚未在实验室服务器实测**。全部命令默认在项目根目录执行，Ubuntu 命令使用 Bash。

## 1. 去实验室前：准备代码、数据与模型

Git 仓库不包含数据、模型、环境、ES 软件及实验结果。建议直接携带本机已验证的文件：

| 文件或目录 | 是否需要 | 说明 |
| --- | --- | --- |
| Git 仓库 | 必须 | 到服务器 clone；也可携带当前代码及本 README |
| `datasets/swe_bench_formatted/` | 必须 | 整个目录原样复制，包括 Arrow、JSON 元数据；test 为 2,294 条 |
| `models/Qwen3-Embedding-0.6B/` | 必须 | 整个模型目录原样复制，约 1.2 GB 权重 |
| `repos/psf/requests/` | 建议 | 已缓存的修复前源码快照；服务器无 GitHub 网络时必须准备齐全 |
| `datasets/BeetleBox/`、`datasets/ye/` | 本轮不需要 | BeetleBox 可供后续测试；Ye 缺修复前提交映射，目前不能直接运行 |
| `results/` 中原始基准结果 | 可选 | 用于逐样本核对，不能替代服务器重新运行 |
| Windows `.venv/`、`.runtime/` | 不复制 | Python 环境与 ES Linux 发行包须在服务器重新准备 |

推荐用文件传输工具原样上传；不要经编辑器重存 Arrow、源码或 tokenizer 文件。可在 Windows PowerShell 打包必要资产（运行前确认输出位置）：

```powershell
tar.exe -czf ..\bug-localization-assets.tar.gz datasets/swe_bench_formatted models/Qwen3-Embedding-0.6B repos/psf/requests
Get-FileHash ..\bug-localization-assets.tar.gz -Algorithm SHA256
```

若没有 `repos/psf/requests`，去掉命令最后一项，后续检索会按固定提交下载缺失快照。到服务器后将压缩包放入项目根目录，核对传输前后的 SHA256，再解压：

```bash
sha256sum bug-localization-assets.tar.gz
tar -xzf bug-localization-assets.tar.gz
```

不要复制 Windows 的 ES 索引数据到新的 Linux ES 实例。向量/分块缓存也非必需；GPU、设备或库版本变化可能产生新缓存指纹，重新编码属于正常现象。

## 2. 服务器：获取代码并创建 Python 环境

需要普通用户账户、Git、Python 3.12（含 venv）、NVIDIA 驱动及可用 GPU。长实验推荐已安装 tmux；这些系统软件由用户或实验室管理员准备。以下主流程面向 **Ubuntu x86_64 + 支持 BF16 的 NVIDIA GPU**。

```bash
git clone https://github.com/91xrz/bug-localization.git
cd bug-localization
git rev-parse HEAD
nvidia-smi

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

需要严格固定算法版本时，在创建环境前执行 `git checkout 5760e1cf78d13782dfe4105dc126c7e40d09086e`；复现过程按本文操作。若服务器只有已安装的 Conda，可先执行 `conda create -n irbl-python python=3.12 -y`、`conda activate irbl-python`，再使用 `python -m venv .venv` 继续以上步骤。无需同时重新安装 Conda 和 Python。

`requirements.txt` 固定直接依赖，包括 PyTorch `2.11.0+cu128`、Transformers `5.19.0` 及多语言 AST 语法包，已配置 CUDA wheel 索引。服务器驱动必须能运行该 wheel；不依靠单独安装 CUDA Toolkit 修复驱动不兼容问题。当前 CUDA 编码器要求 BF16；不支持时只能显式切换 CPU 或另行设计实验，不能称为相同 GPU 配置。

`configs/environment/requirements.lock.txt` 是 Windows 已安装环境的记录；Ubuntu 先安装根目录依赖并验证，再保存服务器自己的完整依赖清单。不要直接复制 Windows 虚拟环境，也不要遇到找不到版本就任意升级后继续声称相同配置。只测试 BM25 时可改装 `configs/environment/requirements-base.txt`。

## 3. 确认数据与模型就绪

完成第 1 节传输后，目录应为：

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

模型哈希取自本机原始实验的 `effective_config.json`，用于识别是否携带了同一版本。模型运行时仅从本地读取，不会自动下载。

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

## 4. Ubuntu：手动准备并启动 Elasticsearch 9.5.5

**ES 是独立服务，BM25 已内置，不需要另外安装 BM25、Kibana 或 Chroma。** 请用户自行下载官方 [Linux x86_64 压缩包](https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-9.5.5-linux-x86_64.tar.gz) 和对应 [SHA512 文件](https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-9.5.5-linux-x86_64.tar.gz.sha512)，例如放到服务器 `~/Downloads/`。仓库的 `setup_elasticsearch.py` / `.ps1` 仅支持 Windows，Ubuntu 不运行它们。

以下针对新建的个人实验实例；不要覆盖实验室已有 ES 配置。先校验下载包：

```bash
(cd ~/Downloads && sha512sum -c elasticsearch-9.5.5-linux-x86_64.tar.gz.sha512)
mkdir -p .runtime/elasticsearch results/elasticsearch/server
tar -xzf ~/Downloads/elasticsearch-9.5.5-linux-x86_64.tar.gz -C .runtime/elasticsearch
mkdir -p .runtime/elasticsearch/config
cp -a .runtime/elasticsearch/elasticsearch-9.5.5/config/. .runtime/elasticsearch/config/
cp configs/elasticsearch/elasticsearch.yml .runtime/elasticsearch/config/elasticsearch.yml
mkdir -p .runtime/elasticsearch/config/jvm.options.d
printf '%s\n' '-Xms1g' '-Xmx1g' > .runtime/elasticsearch/config/jvm.options.d/heap.options
```

配置沿用项目的 `bug-localization-local` 集群名、单节点、`127.0.0.1:9200` 和无认证模式。只用于隔离的个人实验，勿将该配置暴露到公网或覆盖共享生产服务。运行前检查：

```bash
ss -ltn | grep -E ':(9200|9300)\b' || true
sysctl vm.max_map_count
```

若端口被占用，先确认归属，不能直接结束别人的服务。若 `vm.max_map_count` 小于 `1048576`，请管理员依照 [Elastic 系统配置说明](https://www.elastic.co/docs/deploy-manage/deploy/self-managed/vm-max-map-count) 调整；不要自行修改共享服务器全局配置。

使用普通用户启动独立后台 ES，并记录 PID：

```bash
ES_PATH_CONF="$PWD/.runtime/elasticsearch/config" \
  "$PWD/.runtime/elasticsearch/elasticsearch-9.5.5/bin/elasticsearch" \
  -d -p "$PWD/.runtime/elasticsearch/elasticsearch.pid" \
  -Epath.data="$PWD/.runtime/elasticsearch/data" \
  -Epath.logs="$PWD/results/elasticsearch/server"
curl --retry 20 --retry-connrefused --retry-delay 2 --fail http://127.0.0.1:9200
```

预期响应：`cluster_name = bug-localization-local`、`version.number = 9.5.5`。如果失败，查看 `results/elasticsearch/server/` 中的服务日志。后台启动及 PID 用法来自 [Elastic 官方安装说明](https://www.elastic.co/docs/deploy-manage/deploy/self-managed/install-elasticsearch-from-archive-on-linux-macos)。

若必须使用其他空闲端口，修改本地生成的 `elasticsearch.yml` 的 HTTP/transport 端口，同时修改 `configs/retrieval/bm25.json` 的 `endpoint`。语义和综合对比共用该基础配置。当前客户端没有额外的用户名/密码/TLS 参数入口，不应直接连接需要这些设置的共享实例。同一项目目录中串行运行实验；现有 CLI 没有资源调度或并发写入锁。

## 5. 环境与单元测试

选择实验室分配的 GPU，例如 GPU 0；有调度系统时遵守其设备分配，不要覆盖调度器已设置的 `CUDA_VISIBLE_DEVICES`：

```bash
export CUDA_VISIBLE_DEVICES=0
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

基线当前有 **52 项单元测试**；完整依赖安装后应全部通过，注意不要忽略因缺少语法包而出现的 skip。测试不依赖运行中的 ES。环境检查记录写入 `results/environment/verification.json`。

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
export CUDA_VISIBLE_DEVICES=0
mkdir -p results/reproduction
TASK_RUN=$(date +%Y%m%d_%H%M%S)
git rev-parse HEAD > "results/reproduction/$TASK_RUN.commit.txt"
git status --short > "results/reproduction/$TASK_RUN.git-status.txt"
git diff -- configs/retrieval > "results/reproduction/$TASK_RUN.config-diff.txt"
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

本机 **Max 主聚合**参考值：

| 路线 | MAP@10 | MRR@10 | Top-1 | Top-5 | Top-10 | MeanRecall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.4596 | 0.5080 | 0.2727 | 0.8182 | 0.9318 | 0.8632 |
| 动态块 | 0.5650 | 0.6082 | 0.4091 | 0.8636 | 0.9545 | 0.9276 |
| 方法块 | 0.5239 | 0.5684 | 0.4091 | 0.7727 | 0.9091 | 0.8949 |
| 固定块 400 tokens | 0.5466 | 0.5606 | 0.3409 | 0.8864 | 0.9318 | 0.9176 |
| BLagent AST | 0.6297 | 0.6526 | 0.4773 | 0.8636 | 0.9545 | 0.9418 |

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
| `configs/retrieval/comparison.json` | 基础/语义配置路径、仓库过滤、44 条、@10 |

修改编码 batch size 可缓解显存压力，但应记录配置变化；不要为跑通而随意修改分块长度、模型精度或评测 K。综合入口没有 `--device` 或 `--routes` 参数，需修改它引用的 vector JSON。使用已有 `selection.json` 时，语义入口支持 `--selection`，不能与 `--limit/--repository` 混用。

新增的 token 预算 AST / 动态规划路线是独立实验配置，不改变上述四路默认基线。方案、消融对比及复现命令见 [AST token 实验](docs/ast-token-experiment.md)。

已有 BM25 和语义批次后，可用 `scripts/retrieval/fuse_rankings.py --bm25-run BM25_BATCH_DIRECTORY --semantic-run ast96=SEMANTIC_BATCH_DIRECTORY` 做文件级 RRF（运行时在前面加 `python`，并替换实际目录）。它复用已保存的语义排名，从已有 ES 索引补齐 BM25 匹配文件，不重新编码模型。[融合说明与排名结果](docs/hybrid-retrieval.md)记录了本次 @10 结果：Top-10 覆盖提高，但等权融合的 MAP/MRR 低于单独 AST。

若比较“每路最多一票、不按输入名次加权”，可用 `python scripts/retrieval/vote_rankings.py --input-run HYBRID_BATCH_DIRECTORY --depths all 10` 离线读取混合结果。本次纯投票同分很多，指标与同票处理有关；[投票说明](docs/hybrid-retrieval.md#不使用名次的路线投票2026-10-25)同时报告固定展示顺序与随机同票排序期望。

## 10. 常见问题与 Windows 运行

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

Windows 已有环境与 ES 时，仍使用原入口：

```powershell
.\scripts\elasticsearch\start_elasticsearch.ps1
# 另一个终端，在项目根目录运行：
.\.venv\Scripts\python.exe -B scripts\retrieval\run_retrieval_comparison.py --limit 44
```

Windows 新建环境、手动下载 ES ZIP 与本地配置参见 [ES 说明](docs/elasticsearch.md)。PyCharm 选择项目 `.venv/Scripts/python.exe`（Ubuntu 为 `.venv/bin/python`），运行 `scripts/retrieval/run_retrieval_comparison.py`；Working directory 设为项目根目录。

更多细节：[数据来源](docs/datasets.md) · [语义检索](docs/semantic-retrieval.md) · [文件聚合](docs/file-aggregation.md) · [多语言 AST](docs/blagent-ast.md) · [文件结构](docs/project-layout.md)。数据、模型、缓存、快照、环境和结果继续由 `.gitignore` 排除，实验结束后单独备份所需结果。
