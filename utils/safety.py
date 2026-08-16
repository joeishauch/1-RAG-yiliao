# -*- coding: utf-8 -*-
"""生成前/生成后规则校验：纯规则引擎（无 LLM），对危险信号与诊断性表述做硬兜底。

与 prompt 软约束互补：prompt 是「尽量不做」，这里规则是「命中就拦/追加」。
- check_input_danger：入口拦截危险输入（自杀/自残/伤人等），不进图；
- check_output_diagnostic：出口检测肯定式诊断表述，返回追加的安全提示；
- safety_flags：返回命中的规则名列表，供人工审核中断 payload 展示。
"""
import logging

logger = logging.getLogger(__name__)

# 危险信号词表（入口拦截）。命中即拦截，宁可误拦也不放过。
DANGER_KEYWORDS = [
    "自杀", "轻生", "自残", "割腕", "跳楼", "上吊", "服毒", "想死",
    "不想活", "活不下去", "杀人", "伤人", "报复社会", "制造爆炸",
    "买凶", "投毒", "自尽",
]

# 肯定式诊断表述（出口检测）。LLM 若输出这类「下结论」的话，追加免责提示。
# 注意：「确诊」拆成精确短语（确诊为/确诊了等），避免子串误伤「不能确诊」「不代表确诊」
# 这类否定式免责声明（免责声明本身不是诊断性表述，不应触发追加提示）。
DIAGNOSTIC_KEYWORDS = [
    "确诊为", "确诊了", "可以确诊", "能够确诊", "基本确诊", "已确诊",
    "你得了", "你患了", "你一定是", "你肯定是", "你绝对是",
    "百分百是", "肯定是得了", "你一定是得了",
]

# 命中危险信号时的固定拦截话术（不进入图，直接返回）
_DANGER_BLOCK_MESSAGE = (
    "很抱歉，我无法就此提供帮助。如果你正处于痛苦中或有伤害自己/他人的念头，"
    "请立即联系身边可信任的人，或拨打心理援助热线（如全国 24 小时心理援助热线 12356）寻求专业帮助。"
)

# 命中诊断性表述时追加的安全提示
_DIAGNOSTIC_HINT = (
    "（以上内容仅为可能性参考，不能替代医生的诊断结论，请及时就医确诊。）"
)


def check_input_danger(text):
    """入口危险信号拦截。

    Args:
        text: 用户输入文本。

    Returns:
        Optional[str]: 命中返回固定拦截话术，否则 None。
    """
    if not text:
        return None
    for kw in DANGER_KEYWORDS:
        if kw in text:
            logger.warning(f"入口拦截：输入命中危险信号「{kw}」")
            return _DANGER_BLOCK_MESSAGE
    return None


def check_output_diagnostic(text):
    """出口诊断性表述检测。

    Args:
        text: AI 输出文本。

    Returns:
        Optional[str]: 命中返回追加的安全提示，否则 None。
    """
    if not text:
        return None
    for kw in DIAGNOSTIC_KEYWORDS:
        if kw in text:
            logger.warning(f"出口检测：输出命中诊断性表述「{kw}」")
            return _DIAGNOSTIC_HINT
    return None


def safety_flags(text):
    """返回命中的规则名列表（供人工审核中断 payload 展示）。

    Args:
        text: 待检测文本（草稿或最终回答）。

    Returns:
        list[str]: 命中的规则名（如 ["诊断性表述"]），无命中返回空列表。
    """
    if not text:
        return []
    flags = []
    if any(kw in text for kw in DANGER_KEYWORDS):
        flags.append("危险信号")
    if any(kw in text for kw in DIAGNOSTIC_KEYWORDS):
        flags.append("诊断性表述")
    return flags
