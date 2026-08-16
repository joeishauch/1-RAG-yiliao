# -*- coding: utf-8 -*-
"""LLM-as-judge 分诊质量评估：跑完整分诊链路，用 LLM 裁判打分。

链路：retrieve（第一层科室分布）→ consult（各科室会诊）→ summarize（最终四段式建议）
裁判：LLM 对最终建议输出结构化评分（科室命中 + 综合质量 1-5 + 简评）

与 eval_triage_retrieval.py 的区别：
    检索评估只度量「检索返回的病例 label 是否命中期望科室」（Hit@K/MRR）；
    本脚本度量「最终分诊建议」的整体质量（科室准不准 + 依据/建议/免责好不好）。

用法：
    python eval_triage_judge.py       # 跑全部用例
    python eval_triage_judge.py 5     # 只跑前 5 条（快速验证）

依赖：.env 里的 DEEPSEEK_API_KEY（chat/judge）+ DASHSCOPE_API_KEY（embedding）
"""
import json
import logging
import os
import sys
import time

# 保证脚本在任意 cwd 下都能 import 项目内模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)  # ragAgent 模块级日志 handler 需要

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_triage_judge")

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from utils.config import Config
from utils.llms import get_llm
from utils.tools_config import get_tools
import ragAgent

# 复用检索评估的同一批测试用例（症状 → 期望科室），保证两类评估可比
from eval_triage_retrieval import TEST_CASES


class JudgeResult(BaseModel):
    """裁判结构化输出。"""
    department_hit: bool = Field(description="期望科室是否出现在推荐科室中")
    department_rank: int = Field(description="期望科室在推荐科室中的排名，1=第一推荐，0=未命中", ge=0)
    quality_score: int = Field(description="综合质量分 1-5", ge=1, le=5)
    reasoning: str = Field(description="一句话简评")


JUDGE_TEMPLATE = """你是一名医疗分诊质量评估专家。请客观评估下面这条智能分诊建议的质量。

=== 患者主诉 ===
{question}

=== 期望科室（金标准）===
{expected}

=== 系统分诊建议 ===
{final}

评估维度：
1. 科室准确度：期望科室是否出现在「推荐科室」中？若出现，排第几位（1=第一推荐）？若未出现，department_hit=false、department_rank=0。
2. 综合质量（1-5 分）：分诊依据是否合理有支撑、就医建议是否具体可操作、是否含免责声明、是否避免误导。

只输出结构化 JSON，不要输出其他文字。"""


def run_triage(question, llm_chat, llm_embedding, retrieve_tool):
    """跑分诊核心链路（retrieve → consult → summarize），返回最终建议文本。

    手动串起三个节点，绕过图/PG，便于独立评估（分诊场景 agent 节点必调 retrieve）。
    """
    # 1. 第一层：retrieve 工具输出候选科室分布（字符串）
    first_pass = retrieve_tool.invoke({"query": question})
    # 2. 会诊：各科室专科意见
    consult_state = {"messages": [HumanMessage(content=question)], "consultations": []}
    consultations = ragAgent.consult(consult_state, llm_chat, llm_embedding).get("consultations", [])
    # 3. 汇总：最终四段式建议（summarize 读 messages[-1] 作 first_pass）
    sum_state = {
        "messages": [HumanMessage(content=question), AIMessage(content=first_pass)],
        "consultations": consultations,
    }
    final_msgs = ragAgent.summarize(sum_state, llm_chat).get("messages", [])
    return final_msgs[0].content if final_msgs else ""


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(TEST_CASES)
    cases = TEST_CASES[:limit]
    logger.info(f"评估用例数：{len(cases)}（judge 模型：{Config.LLM_TYPE}）")

    llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
    retrieve_tool = get_tools(llm_embedding)[0]

    # judge 链：结构化输出（deepseek 不支持 response_format，需用 function_calling）
    judge_prompt = ChatPromptTemplate.from_messages([("human", JUDGE_TEMPLATE)])
    judge_chain = judge_prompt | llm_chat.with_structured_output(JudgeResult, method="function_calling")

    details = []
    for i, (question, expected) in enumerate(cases, 1):
        t0 = time.time()
        logger.info(f"[{i}/{len(cases)}] 主诉：{question} → 期望 {expected}")
        try:
            final = run_triage(question, llm_chat, llm_embedding, retrieve_tool)
            verdict = judge_chain.invoke({
                "question": question,
                "expected": "、".join(expected),
                "final": final,
            })
            details.append({
                "question": question,
                "expected": expected,
                "department_hit": verdict.department_hit,
                "department_rank": verdict.department_rank,
                "quality_score": verdict.quality_score,
                "reasoning": verdict.reasoning,
                "final": final,
                "elapsed_s": round(time.time() - t0, 1),
            })
            logger.info(f"    hit={verdict.department_hit} rank={verdict.department_rank} "
                        f"score={verdict.quality_score} | {verdict.reasoning}")
        except Exception as e:
            logger.error(f"    [失败] {question}: {e}")
            details.append({
                "question": question, "expected": expected,
                "error": str(e), "elapsed_s": round(time.time() - t0, 1),
            })

    # 汇总指标
    ok = [d for d in details if "error" not in d]
    n = len(ok)
    if n == 0:
        logger.error("无有效评估结果")
        return

    hit_rate = sum(1 for d in ok if d["department_hit"]) / n
    top1_rate = sum(1 for d in ok if d["department_rank"] == 1) / n
    avg_quality = sum(d["quality_score"] for d in ok) / n
    hit_ranks = [d["department_rank"] for d in ok if d["department_hit"]]
    avg_rank = sum(hit_ranks) / len(hit_ranks) if hit_ranks else 0.0

    summary = {
        "n_cases": n,
        "department_hit_rate": round(hit_rate, 4),
        "top1_hit_rate": round(top1_rate, 4),
        "avg_quality_score": round(avg_quality, 3),
        "avg_rank_when_hit": round(avg_rank, 3),
    }

    print("\n=== LLM-as-judge 分诊质量评估 ===")
    print(f"用例数:            {n}")
    print(f"科室命中率:        {hit_rate:.1%}")
    print(f"Top-1 命中率:      {top1_rate:.1%}")
    print(f"平均质量分(1-5):   {avg_quality:.2f}")
    print(f"命中时平均排名:    {avg_rank:.2f}")

    # JSON 报告
    os.makedirs("eval_reports", exist_ok=True)
    report = {
        "config": {"limit": limit, "judge_model": Config.LLM_TYPE, "retrieval_mode": Config.TRIAGE_RETRIEVAL_MODE},
        "summary": summary,
        "details": details,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"eval_reports/triage_judge_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"报告已写入 {path}")


if __name__ == "__main__":
    main()
