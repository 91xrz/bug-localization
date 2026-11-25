# 多语言结构匹配 structure_ast_v2

新增入口复用现有Tree-sitter语法注册表，支持Python、Java、Kotlin、JavaScript、TypeScript/TSX、C、C++、C#、Go、Rust。结构特征仅来自原报告和修复前源码，不读取修复标签、补丁或生成推测。

`src/bug_localization/structural_matching.py`提供：

- `extract_symbols(path, content, language_overrides=None)`：提取声明名称、限定名称与文件定位别名，返回解析错误标记、语言和源码SHA256。
- `anchors(report, symbols)`、`score_sources(report, sources)`：计算IDF加权匹配分数及可复核的名称/路径匹配记录。
- `rank_candidates(report, a, b, sources, alpha=.5, anchor_weight=.5)`：直接为A50∪B50计算新结构分数，融合归一化A/B分数并保留Top50。A/B须覆盖同一文件全集，源码输入必须恰好覆盖候选并集。
- `feature_identity()`：返回算法版本、语法包版本和实现校验和，供实验缓存与来源记录使用。

同原公式，名称权重为`log((候选数+1)/(包含名称的候选数+1))`，文件分数为已匹配名称权重之和除以报告在整个候选池内可匹配名称的总权重。名称在所有候选文件都出现时IDF为0；分母为0则名称分数为0。显式文件路径、可定位到文件的模块或限定类型名称命中时为1。仅提及共享包名/命名空间不会让全部文件自动得1分。

声明使用语言对应的AST节点，不把普通调用、import、注释、字符串当作定义。支持Java/Kotlin包名与类/方法、C#/C++命名空间、Rust模块及impl、Go接收者方法、JS/TS命名函数与赋值的箭头函数、C/C++函数原型。限定名称有边界检查，例如`org.first.Cache.readValue`不会同时命中`org.second.Cache.readValue`。还支持Java的`Type#method`形式及文件路径`:行号`后缀。短名称和限定名称作为不同可解释特征，重复出现同一名称不会累积次数。

不再对其他语言套用Python内置名称过滤。匹配区分大小写，支持Unicode标识符及JS的`$`。保留名称长度至少3、过滤`self/this/cls/args/kwargs`和Python双下划线特殊名称的规则。声明的返回类型、参数变量和普通局部变量不单独充当结构锚点。未支持扩展名或缺少语法包明确报错；语法恢复结果带`parse_has_error`，不静默声称解析完整。`.h`默认C，可用`language_overrides={".h":"cpp"}`指定C++。

## 与已有基线的关系

`lightweight_hypothesis.py`及其Python正则实现保留原样，历史实验继续使用已冻结的`ab_structure_v1`。新语言和新实验使用`structure_ast_v2`入口。没有修改旧指纹或将新特征写入旧缓存。

Python也可以走新AST路线，但不声称新旧Python特征逐项相同：新版本能排除字符串中的伪声明，识别限定名称，采用更严格的路径边界，因此需要单独评测。保留的是历史结果的可复现性，不是通过保留错误强制新旧分数一致。

## 命令行

```bash
.venv/bin/python scripts/retrieval/run_structural_matching.py \
  --input /absolute/path/inputs.json --output /absolute/path/new_scores.json
```

输入示例：

```json
{
  "report": "org.example.Cache.readValue fails",
  "sources": {
    "src/Cache.java": "package org.example; class Cache { void readValue() {} }",
    "src/Other.java": "package org.other; class Cache { void readValue() {} }"
  }
}
```

可再传`a`和`b`完整排名列表（每项`rank/file_path/score`），以及`alpha`、`anchor_weight`，直接生成新AB排名。输出必须是新文件。输入接口拒绝额外评价字段。

单元测试覆盖现有十份多语言切块样例及Kotlin、限定名称区分、注释/调用排除、函数原型、Unicode/大小写、路径边界、IDF与零分母、解析错误记录、完整AB融合。真实GHRB样例的结构覆盖及之后的系数实验须另存，不能用单元测试推断定位性能。
