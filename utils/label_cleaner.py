# -*- coding: utf-8 -*-
"""分诊数据清洗规则：纠正高置信度标签误标 + 过滤模板题。

纯函数、无 IO，供 jsonl2chroma.py 的 parse_lite 实时调用，也供 run_label_audit.py 审核复用。
只针对「明确、高置信度」的系统性误标，**宁可漏、不可错**（误纠正会引入新噪声）。

不在范围：更深的「疾病所属科室 ≠ 分诊科室」语义重标（如胸痛→肿瘤科，需 LLM 全面重标），留企业级。
"""

# 模板题关键词：出现在 question 里即为「疾病知识点」提问（医考/科普题），非「症状→分诊」主诉。
# 证据：急诊科 21 条里绝大多数是「X 的辅助治疗/判断依据」这类，不是真急诊主诉。
TEMPLATE_HINTS = (
    "辅助治疗", "判断依据", "诊断标准", "鉴别诊断",
    "治疗原则", "护理措施", "发病机制", "临床表现",
)

# 精神心理词：question 含这些词说明「头疼/失眠」更可能伴随精神心理问题（抑郁/焦虑等），
# 此时「头疼+失眠」归心理科学是合理的，不纠正。
MENTAL_HINTS = ("抑郁", "焦虑", "情绪", "心情", "压力", "烦躁", "郁闷", "紧张")

# 非具体科室标签：兜底/缺失类 label（无真实分诊指向，如「其他」「未知科室」）。
# 检索层（tools_config._search_and_rank）与灌库层（jsonl2chroma.parse_lite）统一过滤，
# 避免「其他」兜底条目入库后再被检索层丢弃（全量约 1133 条，占 2.8%）。
INVALID_LABELS = ("其他", "未知科室")


def is_template_question(question: str) -> bool:
    """识别模板题（非分诊主诉），返回 True 表示应过滤。

    匹配「辅助治疗 / 判断依据 / 诊断标准 / 鉴别诊断 / 治疗原则 / 护理措施 / 发病机制 / 临床表现」
    等医考/知识题关键词。这类是「问疾病知识点」，不是「自述症状该挂哪个科」。
    """
    if not question:
        return False
    return any(h in question for h in TEMPLATE_HINTS)


def correct_label(question: str, label: str) -> str:
    """纠正高置信度系统性误标，返回新 label（可能不变）。

    规则：question 同时含「头疼/头痛」+「失眠」，且不含精神心理词，label==「心理科学」
          → 纠正为「神经科学」。

    依据（已量化）：头疼+失眠叠加样本里 47% 被标「心理科学」、仅 27% 标「神经科学」；
    纯头疼则基本正确（心理科学仅 2.7%）。故只纠正「躯体性头痛+失眠」的叠加误标，
    保留「抑郁/焦虑/失眠为主」的正确心理科样本。
    """
    if label != "心理科学":
        return label
    if not question:
        return label
    has_headache = ("头疼" in question) or ("头痛" in question)
    has_insomnia = "失眠" in question
    has_mental = any(m in question for m in MENTAL_HINTS)
    if has_headache and has_insomnia and not has_mental:
        return "神经科学"
    return label
