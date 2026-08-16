# -*- coding: utf-8 -*-
"""分诊标签清洗的可插拔审核层。

审核对象：清洗候选（question / original_label / corrected_label / reason）。
审核者可以换，接口（audit）稳定不变：
    - LLMAuditor    ：独立模型裁判（默认 qwen-max），避免用 deepseek 自查（自我评估偏差）
    - HumanAuditor  ：医生人工审核（HITL），企业落地时把 LLMAuditor 换成它即可，管线其余不动

审核结论字段对齐 eval_triage_judge.py 的 JudgeResult 范式（结构化输出 + method="function_calling"）。
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from utils.llms import get_llm


class AuditVerdict(BaseModel):
    """审核结论（结构化输出）。"""
    approved: bool = Field(description="清洗是否正确：corrected_label 是否比 original_label 更符合该主诉应挂的科室")
    authoritative_label: str = Field(description="审核认为的正确科室名")
    reason: str = Field(description="一句话理由")


AUDIT_TEMPLATE = """你是医疗分诊数据质量审核专家。请审核下面这条「科室标签清洗」是否合理。

=== 患者主诉 ===
{question}

=== 原标签 ===
{original_label}

=== 清洗后标签 ===
{corrected_label}

审核要求：
1. 判断「清洗后标签」是否比「原标签」更符合该主诉「首发症状→应挂科室」的分诊语义。
2. approved=true 表示清洗合理（采纳清洗后标签）；false 表示清洗错误（应保留原标签或另有正确科室）。
3. authoritative_label 填你认为正确的科室名（16 个科室之一：妇产科/内科/皮肤性病科/儿科/眼耳鼻喉科/肿瘤科/神经科学/外科/男性健康科/感染与免疫科/口腔科/心理科学/中医科/生殖健康科/急诊科/其他）。
4. 只输出结构化结果，不要输出其他文字。"""


class LabelAuditor:
    """审核接口（稳定不变）：清洗候选列表 → 审核结论列表。"""

    def audit(self, samples: list[dict]) -> list[AuditVerdict]:
        raise NotImplementedError


class LLMAuditor(LabelAuditor):
    """独立模型裁判（默认 qwen-max，与清洗/生成所用 deepseek 隔离，避免自我评估偏差）。"""

    def __init__(self, llm_type: str = "qwen"):
        self.llm_chat, _ = get_llm(llm_type)
        prompt = ChatPromptTemplate.from_messages([("human", AUDIT_TEMPLATE)])
        self.chain = prompt | self.llm_chat.with_structured_output(
            AuditVerdict, method="function_calling"
        )

    def audit(self, samples):
        results = []
        for s in samples:
            try:
                verdict = self.chain.invoke({
                    "question": s["question"],
                    "original_label": s["original_label"],
                    "corrected_label": s["corrected_label"],
                })
                results.append(verdict)
            except Exception as e:
                # 审核异常不阻塞整体，标记为驳回 + 记录原因，交由人工核对
                results.append(AuditVerdict(
                    approved=False,
                    authoritative_label=s["original_label"],
                    reason=f"审核异常: {e}",
                ))
        return results


class HumanAuditor(LabelAuditor):
    """医生人工审核（HITL）：企业落地时接入医生审核队列/界面。

    当前不可用（无外部医生资源），仅占位接口。落地时把 LLMAuditor 换成 HumanAuditor 即可，
    管线其余部分不动：把 samples 推入医生审核队列，医生标注 approved/authoritative_label 后回填。
    """

    def audit(self, samples):
        raise NotImplementedError(
            "医生人工审核尚未接入：企业落地时将清洗候选推入医生审核队列，标注后回填"
        )
