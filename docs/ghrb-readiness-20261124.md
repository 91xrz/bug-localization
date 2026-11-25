# GHRB 下载与当前流程兼容性

2026-11-24。**数据已下载到数据盘，已完成原始报告补全与格式转换；当前完整AB/C流程不能直接切换数据集运行。** 已通过数据读取、Java切块及C输入构造检查，未执行模型推理或系数实验，未下载全部项目源码快照。

## 下载结果

来源：[GHRB官方仓库](https://github.com/coinse/GHRB)，固定提交 `49729ca84ef96a5921eab9f189afc3cad7c82128`。官方仓库采用浅克隆，保留此版本的全部文件；不是完整Git历史。

| 内容 | 数据盘路径 | 状态 |
|---|---|---|
| 官方元数据、生产/测试补丁、工具 | `/data/xrz/bug-localization/datasets/GHRB/` | 已下载，约30MiB；Git对象检查通过，347个普通跟踪文件生成SHA256清单 |
| 原始issue页面、提取正文、下载与核验清单 | `/data/xrz/bug-localization/datasets/GHRB_acquisition/` | 133份不同issue全部获取，约36MiB |
| 当前项目可读取的HF格式数据 | `/data/xrz/bug-localization/datasets/ghrb_formatted/` | 134条缺陷，16个项目，约956KiB |
| Gson修复前源码样例 | `/data/xrz/bug-localization/repos/google/gson/710a76c8b8835e36935c62024001ba6b91ceee01/` | 1份完整快照验证通过，约3.1MiB |

工作区原有的`datasets`、`repos`符号链接可访问上述数据，不在系统盘重复保存。转换脚本和配置仍在代码工作区。目录中的`test`仅是评价语料存储名，没有据此伪造训练/测试划分。

按`verified_bug/*.json`当前有效清单统计134条；废弃的`checkstyle_checkstyle-10857`不在有效清单中。某些官方派生表仍写135条或checkstyle 24条，未使用这些陈旧数量覆盖有效记录。

原仓库不含其元数据引用的`bug_report_all/`正文，也不含16个项目的完整源码；因此单纯clone不能视为实验数据准备齐全。本轮另外获取了全部原始issue正文。抓取只提取GitHub页面中原issue的标题和原始Markdown正文，保存页面校验和；未将评论、PR描述、提交消息、测试失败信息或补丁作为查询。

## 实际兼容性检查

检查入口：`scripts/datasets/check_ghrb_compatibility.py`。结果：`datasets/GHRB_acquisition/compatibility_check.json`。

- 134条记录全部能由现有数据加载和按项目选择逻辑读取；检索配置校验通过。
- 使用现有快照接口取得Gson原始`buggy_commit`，不施加测试或生产补丁。
- 使用本地Qwen3-Embedding-0.6B的真实分词器及现有Java AST切块器，处理样例快照全部217个Java文件，得到1,655块，解析错误0。
- 核对每个文件的代码块可重建完整源码，块长度不超过512 tokens，前缀预算64；每文件前3块可由现有C联合输入构造器生成文档。
- 该样例的3个生产修复文件均存在于候选文件中。
- 没有运行向量编码、功能推测生成、重排或ES检索；本检查不证明模型效果，也不包含完整线上耗时。

## 为什么完整AB/C不能直接使用

| 部分 | 可复用程度 | 尚需处理 |
|---|---|---|
| 数据读取、指标 | 已实际验证可复用 | 系数实验须单独固定开发/验证划分 |
| Java AST切块 | 已实际验证可复用 | 全量源码尚需下载、校验与覆盖审计 |
| 原报告语义检索A、功能推测B | 核心模型/文本处理可复用，未执行推理验证 | 原实验入口绑定SWE选择清单、缓存与模型身份记录，需要独立GHRB入口 |
| AB结构分数E | **不能直接复用Python声明提取规则** | 增加Java类、接口、枚举、方法、构造器及包名/类名匹配，并建立独立版本 |
| C逐块与联合文档构造 | 输入构造已验证，评分器核心可复用 | 当前500条实验入口强制SWE候选及旧评分来源；GHRB必须新生成A/B/C评分 |
| 快照获取 | Gson通过；改名项目已复现失败 | 核验仓库更名/迁移映射并适配快照身份检查，不直接放宽校验 |

结构提取的具体证据：样例217个Java文件中，Java语法树在211个文件发现声明；现有Python正则仅6个文件有匹配，不能视为可靠Java提取。真实修复文件`Gson.java`的Java语法树可提取`Gson`、`fromJson`、`getAdapter`等声明，当前结构扫描器却返回空集合。仅把扩展名从`.py`改成`.java`会使结构分数含义改变，无法公平比较AB系数。

仓库更名的具体证据：`Hakky54/sslcontext-kickstart`的issue现重定向至`Hakky54/ayza`。报告下载已根据最终URL核验同一编号及页面身份并成功获取。源码下载得到根目录`ayza-9775b98981527834d912e2174bbd3589a9c3362f`，而当前快照读取器要求旧仓库名，因此抛出`Archive root does not match the requested repository revision`。原压缩包保留用于诊断，没有修改核心接口绕过检查。详见`renamed_repository_smoke.json`。

另外观察到报告URL迁移：`seata/seata → apache/incubator-seata`、`square/retrofit → lysine-dev/retrofit`。不能仅根据issue所属仓库推导所有源码仓库，因为GHRB也存在真实的跨仓库报告关联；需要为源码单独核验映射。

## 标签与查询口径

1. 文件标签来自作者生产补丁的修改前路径，保留全部622个文件实例，其中619个Java、3个XML。不将测试补丁新增的测试文件混入生产标签。
2. 其中70个是新增文件，涉及`OpenAPITools_openapi-generator-20887`与`alibaba_nacos-10755`，这些文件不在修改前源码中。转换保留标签并显式记录`added_file`，没有静默删除或重新分配路径。仅Java候选也无法覆盖3个XML标签。正式比较必须先固定“全部补丁文件”与“修复前可定位生产文件”的评价口径，分别报告覆盖，不能按模型结果过滤。
3. 一条OpenAPI缺陷的生产补丁包含331个文件，可能显著影响多文件AP/召回分母。应审计其标签范围；本轮没有按补丁大小删样本。
4. 134条缺陷对应133份不同issue；`sslcontext-kickstart-197`与`-217`共同引用issue 196。两条缺陷保持独立记录，但实验分组须将相同issue、相同/近重复正文放在同一侧。当前通用选择器按`(repo_name, issue_id)`去重，因此适配字段`issue_id`使用明确的GHRB缺陷键；原始编号另存`original_issue_number`，不冒充GitHub issue编号。
5. 两条记录的issue属于另一仓库：`jackson-core-923`、`jackson-databind-3418`。查询从元数据指定的issue URL读取，源码从PR所属仓库取得，二者不能强行设为一致。
6. GHRB官方CLI的buggy检出默认还会施加新增测试补丁。本项目报告检索应使用**未施加任何补丁的buggy_commit**，否则可能引入缺陷修复过程中才产生的测试信息。若后续需复现实验测试，另建隔离工作目录。
7. 报告正文是当前下载时的版本，不是自动还原到修复前的历史编辑版本。元数据中11条issue创建于2021年9月之前；“近期收集/近期合并”不能等同于“所有报告均新出现”，也不能保证当前模型未见过。

## 可运行入口与边界

下载补全文本和格式转换：

```bash
.venv/bin/python scripts/datasets/prepare_ghrb.py
```

当前输出已存在，脚本拒绝覆盖；复跑可指定新的`--output`目录。已有正文缓存会核验后复用，失败不导出悄悄缩小的评价集合。官方仓库保持原样。

CPU数据/Java输入检查：

```bash
.venv/bin/python scripts/datasets/check_ghrb_compatibility.py
```

基础检索配置位于`configs/retrieval/ghrb/bm25.json`及`vector.json`，默认Gson一条，用于后续功能验证。服务和GPU资源可用时可以使用既有BM25/向量入口；本轮没有启动这些推理。这两份配置不是完整AB/C实验配置。

后续GHRB系数实验应先完成全量源码与标签覆盖检查、Java结构模块和独立数据入口，再按项目及重复报告分组固定划分，补齐A50∪B50所有必要C分数后进行AB/C联合系数搜索。保留当前SWE已冻结基线，不把新语言适配继续称为相同的`ab_structure_v1`实现。

准备脚本的4项测试覆盖修改前标签/新增标签保留、补丁路径拒绝、报告与评论隔离、仓库更名身份核验，全部通过。来源记录、下载页面SHA256、仓库文件SHA256、转换审计与兼容检查均保存在数据盘。
