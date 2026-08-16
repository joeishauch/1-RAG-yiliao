# -*- coding: utf-8 -*-
"""药物禁忌接入 LangGraph 的分项验证（不依赖 PG）：
drug_taboo 工具 + generate_drug 节点 + route_after_tools 路由 + agent 工具选择。

与 e2e_regression.py 的「完整图流」不同，本脚本照搬 verify_kg_integration.py 的
「分项 mock」模式，逐项验证四个环节，不真正读写 PG，也不走完整 create_graph。

用法（需先有 drug_contraindications.json）：
    python verify_drug_taboo.py
"""
import os
import sys

# 切换到脚本所在目录，保证 drug_contraindications.json / .env / prompts 等相对路径正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

# Windows 下 stdout 可能是 GBK，统一转 UTF-8 避免中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from utils.config import Config
from utils.llms import get_llm
from utils.tools_config import get_tools, get_drug_context
import ragAgent
from ragAgent import ToolConfig


class MockStore:
    """mock PostgresStore：不真正读写记忆，仅满足 agent 节点调用。"""
    def search(self, namespace, query=None, **kwargs):
        return []
    def put(self, namespace, key, value, **kwargs):
        pass


def main():
    llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
    tools = get_tools(llm_embedding)
    print(f"工具列表: {[t.name for t in tools]}")
    retrieve_tool, medical_qa_tool, kg_query_tool, drug_taboo_tool = tools[0], tools[1], tools[2], tools[3]

    # 取知识库第一条药名做「命中」测试，保证有真实结果；「不存在的药xyz」测未命中分支
    records = get_drug_context()
    sample_drug = records[0]["drug_name"] if records else "阿司匹林"
    print(f"知识库条数: {len(records)}，命中测试用药物: {sample_drug}")

    # 1. drug_taboo 工具（命中 + 未命中两条）
    print("\n=== 1. drug_taboo 工具 ===")
    for q in [sample_drug, "不存在的药xyz"]:
        r = drug_taboo_tool.invoke({"drug": q})
        print(f"--- 药物「{q}」---")
        print(r[:600])
        print()

    # 2. generate_drug 节点
    print("=== 2. generate_drug 节点 ===")
    drug_result = drug_taboo_tool.invoke({"drug": sample_drug})
    drug_state = {
        "messages": [
            HumanMessage(content=f"{sample_drug}有什么禁忌？"),
            AIMessage(content="", tool_calls=[{"name": "drug_taboo", "args": {"drug": sample_drug}, "id": "d1"}]),
            ToolMessage(content=drug_result, name="drug_taboo", tool_call_id="d1"),
        ],
        "consultations": [],
    }
    final = ragAgent.generate_drug(drug_state, llm_chat)
    print("generate_drug 输出:", final["messages"][0].content)

    # 3. route_after_tools 路由
    print("\n=== 3. route_after_tools 路由 ===")
    print("drug_taboo →", ragAgent.route_after_tools(drug_state))
    triage_state = {
        "messages": [
            HumanMessage(content="鼻塞流鼻涕"),
            AIMessage(content="", tool_calls=[{"name": "retrieve", "args": {"query": "鼻塞流鼻涕"}, "id": "t1"}]),
            ToolMessage(content="候选科室分布...", name="retrieve", tool_call_id="t1"),
        ],
        "consultations": [],
    }
    print("retrieve →", ragAgent.route_after_tools(triage_state))

    # 4. agent 工具选择（mock store）：验证意图路由，不依赖知识库是否真的含该药
    print("\n=== 4. agent 工具选择 ===")
    tool_config = ToolConfig(tools)
    config = {"configurable": {"user_id": "test", "thread_id": "1"}}
    for q in [f"孕妇能吃{sample_drug}吗", "阿司匹林有什么禁忌", "我鼻塞流鼻涕该挂哪个科"]:
        state = {"messages": [HumanMessage(content=q)], "consultations": []}
        resp = ragAgent.agent(state, config, store=MockStore(), llm_chat=llm_chat, tool_config=tool_config)
        last = resp["messages"][0]
        calls = [tc["name"] for tc in last.tool_calls] if getattr(last, "tool_calls", None) else []
        print(f"问题「{q}」→ 选择工具: {calls}")


if __name__ == "__main__":
    main()
