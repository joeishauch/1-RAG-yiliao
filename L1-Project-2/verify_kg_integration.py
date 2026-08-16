# -*- coding: utf-8 -*-
"""知识图谱接入 LangGraph 的分项验证（不依赖 PG）：
kg_query 工具 + generate_kg 节点 + route_after_tools 路由 + agent 工具选择。

用法：
    python verify_kg_integration.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from utils.config import Config
from utils.llms import get_llm
from utils.tools_config import get_tools
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
    retrieve_tool, medical_qa_tool, kg_query_tool = tools[0], tools[1], tools[2]

    # 1. kg_query 工具（规范症状 + 口语症状，验证模糊匹配在工具链路里也生效）
    print("\n=== 1. kg_query 工具 ===")
    for q in ["胸痛", "胸口闷"]:
        r = kg_query_tool.invoke({"symptom": q})
        print(f"--- 症状「{q}」---")
        print(r[:600])
        print()

    # 2. generate_kg 节点
    print("=== 2. generate_kg 节点 ===")
    kg_result = kg_query_tool.invoke({"symptom": "胸痛"})
    kg_state = {
        "messages": [
            HumanMessage(content="胸痛可能是哪些病？"),
            AIMessage(content="", tool_calls=[{"name": "kg_query", "args": {"symptom": "胸痛"}, "id": "kg1"}]),
            ToolMessage(content=kg_result, name="kg_query", tool_call_id="kg1"),
        ],
        "consultations": [],
    }
    final = ragAgent.generate_kg(kg_state, llm_chat)
    print("generate_kg 输出:", final["messages"][0].content)

    # 3. route_after_tools 路由
    print("\n=== 3. route_after_tools 路由 ===")
    print("kg_query →", ragAgent.route_after_tools(kg_state))
    triage_state = {
        "messages": [
            HumanMessage(content="鼻塞流鼻涕"),
            AIMessage(content="", tool_calls=[{"name": "retrieve", "args": {"query": "鼻塞流鼻涕"}, "id": "t1"}]),
            ToolMessage(content="候选科室分布...", name="retrieve", tool_call_id="t1"),
        ],
        "consultations": [],
    }
    print("retrieve →", ragAgent.route_after_tools(triage_state))
    qa_state = {
        "messages": [
            HumanMessage(content="什么是高血压"),
            AIMessage(content="", tool_calls=[{"name": "medical_qa", "args": {"query": "什么是高血压"}, "id": "q1"}]),
            ToolMessage(content="高血压是...", name="medical_qa", tool_call_id="q1"),
        ],
        "consultations": [],
    }
    print("medical_qa →", ragAgent.route_after_tools(qa_state))

    # 4. agent 工具选择（mock store）
    print("\n=== 4. agent 工具选择 ===")
    tool_config = ToolConfig(tools)
    config = {"configurable": {"user_id": "test", "thread_id": "1"}}
    for q in ["胸痛可能是哪些病", "我鼻塞流鼻涕该挂哪个科", "什么是高血压平时要注意什么"]:
        state = {"messages": [HumanMessage(content=q)], "consultations": []}
        resp = ragAgent.agent(state, config, store=MockStore(), llm_chat=llm_chat, tool_config=tool_config)
        last = resp["messages"][0]
        calls = [tc["name"] for tc in last.tool_calls] if getattr(last, "tool_calls", None) else []
        print(f"问题「{q}」→ 选择工具: {calls}")


if __name__ == "__main__":
    main()
