# 使用 Bench4BL

检索配置现集中于 `configs/retrieval/bench4bl/`，原来的 `bench4bl_*.json` 文件已迁移。请使用本文的新命令；旧实验结果中保存的原配置路径作为历史记录保留。

已接入当前 BM25、Java AST96、统一指标与模块2入口。沿用现有 Python 环境、Qwen Embedding 模型和 Elasticsearch，不新增第三方 Python 依赖。

## 当前适配范围和口径

- 原始下载：`datasets/Bench4BL/`，51个项目包保留原样。
- 可运行数据：`datasets/bench4bl_formatted/`，46个现代项目、9,459份报告、690个发布版本。
- `Previous` 的5个旧项目（共558份报告）尚未自动转换：缺少相同格式的 `bugs.txt`，部分版本和已有源码目录不一致。例如PDE元数据写4.5，但源码目录是PDE_4_4；不能自行猜测映射。它们继续保存在原始目录中。
- `test` 表示评价语料，没有人为制造训练/测试划分。

**此处使用作者发布版本分组，不是逐缺陷修复前提交。** 从 `versions.txt` 解析精确tag，从 `bugs.txt` 读取作者报告分组，再在本地Git对象库中解析提交。为兼容现有入口，`before_fix_sha` 暂存该发布提交；真正含义同时写入 `source_revision`、`source_revision_role`、`snapshot_protocol` 及结果 `source_provenance`，不得把它解释成SWE-bench式修复前SHA。

标题与正文来自作者 `bugrepo/repository.xml`，按XML语义保留，不使用补丁、fixdate或标签作为查询。默认使用标题和正文；未合并重复报告的描述。只使用作者 `bugs.txt` 的 `all` 清单，不额外纳入XML中未被作者选用的报告，也不重新筛选“容易”样本。

## 已在本机准备，可直接运行

在项目根目录激活虚拟环境，并按原说明启动ES，然后执行：

```bash
# 环境和数据注册检查
python scripts/environment/verify_environment.py --check-bench4bl

# CSV项目5条报告：相同查询/源码/标签下比较BM25和AST96
python scripts/retrieval/run_retrieval_comparison.py --config configs/retrieval/bench4bl/comparison.json
```

也可以单独运行：

```bash
python scripts/retrieval/run_bm25.py --config configs/retrieval/bench4bl/bm25.json
python scripts/retrieval/run_bm25_batch.py --config configs/retrieval/bench4bl/bm25_batch.json
python scripts/retrieval/run_vector_batch.py --config configs/retrieval/bench4bl/vector.json
```

配置默认是Commons CSV，5条、Java文件、AST96、Max、@10。这是一组功能验证入口，不是完整Bench4BL论文成绩。原SWE-bench配置和默认切块不改动。

换项目时，批量CLI可指定 `--repository Bench4BL/Commons_CODEC --limit 10`（仅BM25批次与语义批次支持；综合入口修改其JSON里的 `repositories` 和 `limit`）。项目标识格式为 `Bench4BL/<Group>_<Subject>`，完整列表在转换报告里。当前批次仍沿用1–50条的原有范围限制；未实现一次性运行9,459条的大规模调度。

## 另一台机器首次准备

把原始 `datasets/Bench4BL/data/` 连同各项目的 `gitrepo/.git` 复制到服务器，或用下载脚本准备原始包，再运行一次转换：

```bash
python scripts/datasets/prepare_bench4bl.py
```

转换只读取本地Git对象，不下载GitHub仓库，不checkout原始仓库，不生成全部690份源码副本。运行每个报告时才将对应版本的普通Java文件物化到 `repos/bench4bl/<Group>/<Subject>/<SHA>/source/`，多个报告共享同一版本缓存。读取时核对tag、树指纹和Git blob校验值；已缓存源码若被修改会拒绝复用。

已有输出时转换脚本会拒绝覆盖。需要另一种转换范围时，可另设 `--output datasets/新的目录名`，并同步修改BM25配置的 `dataset`。`--subjects Commons/CSV Commons/CODEC` 可限制转换项目，不改变这些项目内部的作者选择顺序。Windows与Ubuntu使用同一套Python代码；服务器须安装Git。

默认从转换注册文件读取原始数据位置。原始数据迁到其他目录后，可通过 `BENCH4BL_ROOT` 环境变量指定原始 `Bench4BL` 根目录。只携带HF文件而不携带本地Git对象库，不能创建新的版本源码缓存。

## 缺陷文件映射与限制

将作者的 `org.example.Type.java` 映射到对应版本中唯一的路径后缀，如 `src/main/java/org/example/Type.java`；已是相对路径时先做精确匹配。保留测试文件，候选不按标签筛选。不存在或有多个候选时，不猜测、更不删除报告；用独立缺失标记保留该gold，并在 `label_mapping_json` 保存原名、原因与所有歧义候选。它不会参与检索，只会按现有缺失gold规则计入指标分母。

全量转换的38,217个标签中，31,196个唯一映射成功，6,969个在指定发布版本中未找到，52个有歧义；1,794份报告至少有一个未解决标签，其中621份全部标签未解决。结果必须同时查看覆盖率，不能忽略这些记录，也不能直接与采用其他过滤/重命名规则的论文数字比较。原始标签始终另存，后续若研究重命名映射，应产生新数据版本。

## 模块2和融合

```bash
# 启动既有本地LLM服务后，先生成；可加 --limit 1 冒烟
python scripts/retrieval/run_multiview.py --config configs/experiments/bench4bl_multiview.json --prepare
# 停止生成服务释放显存；ES保持运行，再检索
python scripts/retrieval/run_multiview.py --config configs/experiments/bench4bl_multiview.json --run
```

已有BM25、AST批次也可继续用 `fuse_rankings.py` 或 `vote_rankings.py`，保持两端选择清单一致；核心融合与评价无需另写。模块2说明见 [多视角报告查询](report-multiview.md)。

## 验证记录（2026-10-26）

- CSV前5条：BM25和AST96均5/5成功，查询、候选、标签、提交和@10口径核对通过，10个gold全部在候选中。
- BM25 MAP@10=0.5583、MRR@10=0.5333；AST96 MAP@10=0.7400、MRR@10=0.8000。仅用于确认接入流程。
- 综合结果：`results/comparison/20261026_184114_808935/`。
- 模块2：CSV #96的真实LLM生成及六路检索均成功，`results/multiview/20261026_104511_192937_retrieval/`，用于功能验证，不据此声称查询增强有效。
- 原Requests #1142 AST96的完整排名、分数和指标与历史批次一致。
- 回归测试涵盖安全元数据解析、缺失/歧义gold保留、精确tag解析、使用发布版本而非HEAD、离线缓存篡改检测与原仓库HEAD不变。
- 当前102项单元测试全部通过。

转换审计在 `datasets/bench4bl_formatted/conversion_report.json`，逐行保留原标签和映射审计，版本注册表在 `bench4bl_sources.json`。原始数据、转换数据、源码缓存和实验结果均由Git忽略。
