# -*- coding: utf-8 -*-
"""医疗分诊 MCP Server：把 4 个领域工具暴露为 MCP 工具（方向1：对外提供）。

复用 utils/tools_config.py 的模块级检索/图谱/药物上下文，零重构现有文件。
运行：python mcp_server.py          （stdio 传输，供 MCP client 连接）
"""
import os
import sys
from pathlib import Path

# 对齐 cli.py：统一 CWD + sys.path + UTF-8（保证相对路径/.env 正确 + 中文不乱码）
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from fastmcp import FastMCP

from utils.config import Config
from utils.llms import get_llm
from utils.tools_config import (
    get_triage_context, get_qa_context, get_drug_context,
    _search_and_rank, search_drug_taboo, QA_TOP_K,
)
from utils.kg_query import get_kg, query_by_symptom

mcp = FastMCP("medical-triage")

# ---- 模块级上下文（进程启动时初始化一次，与 tools_config 的懒加载单例等价）----
print("初始化 LLM 与检索上下文 ...", file=sys.stderr)
_llm_chat, _llm_embedding = get_llm(Config.LLM_TYPE)
_vectorstore, _prior_counter, _prior_total, _hybrid = get_triage_context(_llm_embedding)
_qa_vectorstore = get_qa_context(_llm_embedding)


@mcp.tool()
def retrieve(query: str) -> str:
    """分诊查询工具：根据症状描述检索相似病例，返回候选科室及校准后的置信度分布。"""
    scored, examples = _search_and_rank(_hybrid, _prior_counter, _prior_total, query)
    if not scored:
        return "未检索到相关分诊信息。"
    total_score = sum(s for _, _, s in scored) or 1.0
    dist_lines = []
    for label, cnt, score in scored:
        conf = score / total_score
        dist_lines.append(f"{label}：{cnt}次（{conf:.0%}）")
    example_lines = [f"- {label}：{ans}" for label, ans in examples.items() if ans]
    parts = ["候选科室分布（已按全库先验分布校准）：", "；".join(dist_lines)]
    if example_lines:
        parts.append("代表病例摘要：")
        parts.extend(example_lines)
    return "\n".join(parts)


@mcp.tool()
def medical_qa(query: str) -> str:
    """医学知识问答工具：检索医学知识库，回答疾病/症状/用药/保健等科普类问题。"""
    docs = _qa_vectorstore.similarity_search(query, k=QA_TOP_K)
    if not docs:
        return "未检索到相关医学知识。"
    lines = []
    for i, d in enumerate(docs, 1):
        ans = d.metadata.get("answer", "").strip()
        if ans:
            lines.append(f"{i}. {ans}")
    return "\n".join(lines) if lines else "未检索到相关医学知识。"


@mcp.tool()
def kg_query(symptom: str) -> str:
    """症状疾病推理工具：根据症状查询医学知识图谱，返回可能的疾病及治疗/药物/科室。"""
    G = get_kg(Config.KG_GRAPH_PATH)
    res = query_by_symptom(G, symptom, top_k=5)
    if res is None or not res["diseases"]:
        return "知识图谱中未检索到该症状对应的疾病信息。"
    lines = [f"症状「{res['symptom']}」可能关联的疾病（按相关性排序）："]
    for i, d in enumerate(res["diseases"], 1):
        seg = [f"{i}. {d['name']}"]
        if d["department"]:
            seg.append(f"科室：{'、'.join(x[0] for x in d['department'][:3])}")
        if d["drugs"]:
            seg.append(f"药物：{'、'.join(x[0] for x in d['drugs'][:5])}")
        if d["treatments"]:
            seg.append(f"治疗：{'、'.join(x[0] for x in d['treatments'][:5])}")
        if d["symptoms"]:
            seg.append(f"伴随症状：{'、'.join(x[0] for x in d['symptoms'][:5])}")
        lines.append(" | ".join(seg))
    return "\n".join(lines)


@mcp.tool()
def drug_taboo(drug: str) -> str:
    """药物禁忌查询工具：根据药物名查询禁忌症/禁忌人群。"""
    records = get_drug_context()
    hits = search_drug_taboo(drug, records)
    if not hits:
        return f"未在知识库中找到药物「{drug}」的禁忌信息。"
    lines = [f"药物「{drug}」的禁忌信息（共 {len(hits)} 条匹配）："]
    for r in hits[:5]:
        name = r.get("drug_name", "")
        items = "；".join(r.get("contraindications") or []) or "（该药暂无明确禁忌记录）"
        lines.append(f"- 【{name}】{items}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(show_banner=False)  # 默认 stdio 传输；关闭 fastmcp 3.x 启动 banner