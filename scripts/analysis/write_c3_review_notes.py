"""Persist assistant qualitative triage, explicitly separate from human adjudication."""
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src'))
from bug_localization.reranking import read_json
from bug_localization.batch import write_json

# Case-specific reading notes. Labels describe hypotheses, not validated prevalence.
NOTES = {
'development500_264': ('selection_score_gain', 'mostly_restates_with_speculation', ['b10077cff75a','fe9d7596de33'], '原报告已给出 rotation/linespacing 和 setter 要求。C3 保留 setter 块，新增 Text.__init__ 参数块，最高分升至4.81、文件5→1。推测没有补出关键缺失功能，更多是重述加字典/集合猜测；不能只凭提升认定根因推断正确。'),
'development500_464': ('selection_gain', 'mixed_supported_and_wrong_context', ['3d9467740593'], 'C3 新增 ColorizeFormatter.format，直接涉及日志颜色，文件4→1；A 的块主要为过滤器和setup。推测把构建控制台警告说成HTML渲染，场景不准确，却仍选到有用的日志着色块。'),
'development500_047': ('selection_score_gain', 'unsupported_timer_hypothesis', ['1ed7e64bbb72','4f9ba938a74e'], 'A 已选 MultiCursor.onmove/_update，C3 新增构造器块获7.38分，文件3→1。事件/绘制描述相关，但timer及idle原因未经补丁验证，不能把分数提升等同于补中根因。'),
'development500_222': ('selection_gain', 'summary_with_contradicted_implementation_detail', ['99e4536ce35c','104f5a0ca83b'], '报告明确点名_check_list_display_item；A只选到外围_check_list_display与editable块，C3选到目标函数，文件3→1。推测称hasattr检查被移除，但保存代码仍含hasattr(obj.model,item)，不能当成可靠的实现事实。'),
'development500_033': ('selection_gain', 'mostly_restates', ['79344a9be7bf','2af15cefd786'], '报告已包含ModelChoiceField错误模板；A已有抛ValidationError的to_python块，C3再选到缺少value占位符的模板，最高分6.75，文件2→1。属于证据补充，不是发现报告中没有的功能。'),
'development500_234': ('relevant_block_undervalued_and_score_bearing_block_removed', 'plausible_unverified', ['e7cfa99c6c0b','913a084d6818','6099a4beead3'], '推测指向GROUP BY别名；C3选到set_group_by，语义上与报告直接相关，但分数为-2.75/-2.00；A的get_aggregation说明块7.56被移除，文件1→4。更相关不一定更高分；缺少补丁核对，不能判定推测错误。'),
'development500_295': ('misdirected_hypothesis_and_changed_evidence', 'contradicts_report', ['6fcfe989f68f','b0a32d4f6ba6','a3e83ffa7566'], '报告抱怨输出过长、希望约40行截断；推测反说threshold太低、表示不完整，方向相反。C3保留short_numpy_repr，却把高分的截断工具块换成按字符宽度平铺的format_array_flat，文件1→4。错误推测与选块变化同时出现，单例不证明因果。'),
'development500_455': ('selection_loss_candidate', 'partly_contradicts_report', ['ce5e84d043b0','fc70dea0ec79','f404261318fa'], '报告明确相对路径按rst文件解析；推测却说固定基目录而非源文件相对，部分不符。C3用低分Figure块替换A最高分模块块，文件1→4；两组CSVTable块主要是标题/源码位置处理，尚未证实所选块含路径解析根因，需要扩展上下文人工核查。'),
'development500_068': ('selection_loss', 'plausible_functional_expansion', ['d0fcee828de6','99a76cc55eeb'], '报告只有groupby应支持name=None。A选中明确的if getattr(group,"name",None) is None: raise ValueError，得5.12；C3丢掉此块，换成无组/bins错误与状态赋值，文件1→2。可直接看到有效判定分支丢失；推测本身并不明显错误。'),
'development500_133': ('relevant_block_undervalued', 'supported_mechanistic_expansion', ['e4beade71c7f','75bf2f359268'], 'C3选到_parallel_pairwise中的np.empty(...dtype=X.dtype)与线程写回，恰与整数输入导致小数被截断的现象一致；该块仅3.94分。A泛化的pairwise_distances参数说明块7.88被替换，文件1→2。这支持核查重排器对执行逻辑的低估，不支持把退化简单归因于错误推测。'),
'development500_374': ('unchanged_evidence_and_rank', 'restates_feature_request', ['47d6815224'], '普通并列样本：两组deprecated.py块相同，文件均第3。推测重述文档迁移需求，没有产生额外区分力；仍需看警告定义与文档标签口径。'),
'development500_356': ('changed_blocks_no_metric_change', 'unsupported_existing_implementation_guess', ['2c8759759f'], '普通并列样本：报告给出目标文件并要求新增glob支持。推测猜测已有split/glob逻辑无法递归，未经证实。C3换块但最高分仍来自共有块，正确文件保持第2。'),
'development500_267': ('source_inventory_and_partial_recall_gap', 'supported_restatement', [], '普通并列样本包含大量新增下划线模块标签，修复前候选清单中不存在。两组均有8个此类路径缺口；不能把这些算作语义召回或选块失败。C3召回了更多旧模块，但Top10 AP不变。'),
'development500_236': ('relevant_check_blocks_ranked_behind', 'mostly_restates', ['d855459599'], '两组均选择_check_indexes，最高分共有块4.44且文件均第3；C3新增_check_field_name_clashes没有改善。适合作为普通低排名对照，不能仅凭文件标签宣布选块已充分。'),
'development500_240': ('already_solved', 'supported_restatement', ['c6cd6eb605'], '两组均保留删除方法块，均排第1；原报告已明确with块外访问rowcount。推测正确地复述这一点，但此样本没有额外准确率收益。'),
'development500_022': ('already_solved', 'supported_restatement', ['b13af4c169'], '报告已有确切文件、save调用和数据库说明。两组均选_rename、均第1；新增推测没有带来AP改善。'),
'development500_271': ('already_solved', 'plausible_version_change_summary', ['ef95bfd1e3'], '两组均以共有style.use/fix_style块获得最高分，均第1；版本变化的推测未进一步提高已满分结果。'),
'development500_207': ('source_inventory_gap_and_unchanged_evidence', 'invented_existing_failure', ['12878fb69e'], '报告请求新增常量版本context manager；推测把它描述成现有context manager恢复状态失败，属于无依据的现状假设。常量utils.py标签在修复前不存在，两组gold代码块实际相同，微小收益来自其他文件相对顺序，不能归因于修复了该假设。'),
'development500_152': ('competition_or_scoring_effect', 'plausible_restatement', ['2f392e0505'], '两组均保留_resolve_ref_xref且得分0.75不变；C3替换的另一个块得分更低，gold文件5→6。应核查竞争文件而非直接指责gold块不足。'),
'development500_094': ('relevant_preprocessing_block_undervalued_candidate', 'plausible_restatement', ['346e566289','df72e9ac6a'], 'C3新增_preprocess_options，符合CLI选项问题，但得-4.56；最高分仍为共有扩展加载块-1.00。文件8→10；需要完整函数上下文确认相关分支，暂不下选块失败结论。'),
'external47_001': ('recall_rescue', 'adds_missing_functional_description', ['b0496755bfb7'], '报告只给数值、链接，未明说3/5倍数求和；推测补出算术级数和整数除法方向，与保存代码及补丁将/改为//一致。A未召回目标，C3召回并排1。这是功能补充有帮助的代表，但不能由单例区分推理与已有知识。'),
'external47_015': ('candidate_competition_or_batch_score_effect', 'mostly_restates', ['35f11a4063e0'], '报告已给完整traceback、函数与缺失header。两组gold选块完全相同，包含补丁所修改的header.index表达式，但模型块分数9.56→9.75，文件2→1；不能把增益记作gold选块改善。不同候选/输入批次可能改变BF16分数及竞争顺序。'),
'external47_022': ('relevant_block_undervalued', 'mostly_supported_type_mismatch_summary', ['85d64f1dc398','e974a204a44b'], '两组都选到补丁增加str转换的位置：deserialize_properties直接构造FileAsset等；该块两组均为-3.00。C3用serialize_property块替换另一较高分块，文件2→5。有效故障代码已在输入中仍被低估，不能认定主要是漏选它。'),
'external47_027': ('relevant_block_present_competition_and_scoring', 'misleading_runtime_value_explanation', ['792645cc728f'], '实现文件两组3块完全相同，含补丁修复的_inner_schema分支，文件3→4且该块分数5.62→5.75。推测偏向validator改变值造成Literal不匹配，补丁实际处理validator包装schema。实现文件退化不能归为漏选块；测试文件选块则有变化。'),
'external47_019': ('selection_gap_candidate_with_recall_complete', 'correct_core_but_wrong_usage_detail', ['0bc4710493aa'], '普通并列失败：全部3个gold文件都进候选；exceptions.py仍在15/16。补丁修改首行from sanic.http import STATUS_CODES，但其已选块为异常类、abort和孤立装饰器，不含该导入，支持选块不足待人工确认。推测正确指出模块遮蔽，但把response/exceptions的http用途说成cookies，与补丁的状态码用途不符。'),
'external47_016': ('partial_recall_gap', 'shallow_event_loop_speculation', [], '普通并列失败：A漏4/6个gold，C3漏3/6并把新增tests/test_widget排46；await_remove/message_pump/_context仍未召回。推测停留在示例的populate/click和事件循环，没有具体补出这些内部同步组件。'),
'external47_003': ('already_solved', 'supported_compatibility_explanation', ['edaa66d935df'], '两组均第1，选块相同，包含subprocess.run；补丁改用check_call兼容Python2。推测的版本兼容说明正确，但原traceback与Python2.7信息已足够，准确率无增益。'),
'external47_040': ('partial_recall_and_causal_gap', 'symptom_restatement_missing_underlying_cause', ['57d84d391248'], '普通微小改善：推测复述安装目录/状态文件缺失，未识别补丁新增的shims/executable判定。C3丢失一个测试标签候选，helpers仍第18，ruby由4→3带来小幅AP提升。不能据此认为故障机制解释改善。'),
}


def main():
    out=PROJECT/'results/c3_failure_attribution/20261112';notes=[]
    templates=read_json(out/'human_review_template.json')
    for item in templates:
        key=item['case_id'];mechanism,hyp,anchors,text=NOTES[key]
        packet=read_json(out/'review_packets'/(key+'.json'))
        chunks={c['chunk_id'] for e in packet['selected_evidence'].values() for c in e['gold_chunks']}
        resolved=[]
        for prefix in anchors:
            match=[c for c in chunks if c.startswith(prefix)]
            if len(match)!=1:raise ValueError('Unverifiable evidence anchor: '+key+' '+prefix)
            resolved.extend(match)
        notes.append({'case_id':key,'review_type':'assistant_qualitative_triage','human_confirmed':False,
            'sampling_stratum':item['sampling_stratum'],'mechanism_hypothesis':mechanism,'hypothesis_assessment':hyp,
            'evidence_chunk_ids':resolved,'notes_zh':text,'next_action':'Human reviewer to inspect full packet, surrounding code and patch where available; adjudicate relevance and causality separately.'})
    write_json(out/'assistant_review_notes.json',{'human_adjudicated_cases':0,'assistant_triaged_cases':len(notes),
        'warning':'Diagnostic stratified sample, not an unbiased estimate of mechanism prevalence. No causal proof or human review is claimed. Patch is retrospective evidence only, never a retrieval input.','cases':notes})
    print('Written',len(notes),'evidence-anchored preliminary notes')

if __name__=='__main__':main()
