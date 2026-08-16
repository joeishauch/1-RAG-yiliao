# -*- coding: utf-8 -*-
"""科普兜底功能验证（不依赖 PG）：medical_qa 工具 + generate_qa 节点 + 路由 + agent 工具选择。"""
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
    retrieve_tool, medical_qa_tool = tools[0], tools[1]
    print(f"工具列表: {[t.name for t in tools]}")

    # 1. medical_qa 工具检索
    print("\n=== 1. medical_qa 工具检索「什么是糖尿病」 ===")
    qa_result = medical_qa_tool.invoke({"query": "什么是糖尿病"})
    print(qa_result[:400])

    # 2. generate_qa 节点
    print("\n=== 2. generate_qa 节点 ===")
    qa_state = {
        "messages": [
            HumanMessage(content="什么是糖尿病"),
            AIMessage(content="", tool_calls=[{"name": "medical_qa", "args": {"query": "什么是糖尿病"}, "id": "qa1"}]),
            ToolMessage(content=qa_result, name="medical_qa", tool_call_id="qa1"),
        ],
        "consultations": [],
    }
    final = ragAgent.generate_qa(qa_state, llm_chat)
    print("generate_qa 输出:", final["messages"][0].content)

    # 3. route_after_tools 路由
    print("\n=== 3. route_after_tools 路由 ===")
    print("科普 →", ragAgent.route_after_tools(qa_state))
    triage_state = {
        "messages": [
            HumanMessage(content="鼻塞流鼻涕"),
            AIMessage(content="", tool_calls=[{"name": "retrieve", "args": {"query": "鼻塞流鼻涕"}, "id": "t1"}]),
            ToolMessage(content="候选科室分布...", name="retrieve", tool_call_id="t1"),
        ],
        "consultations": [],
    }
    print("分诊 →", ragAgent.route_after_tools(triage_state))

    # 4. 真实 agent 节点（mock store）测工具选择
    print("\n=== 4. agent 工具选择 ===")
    tool_config = ToolConfig(tools)
    config = {"configurable": {"user_id": "test", "thread_id": "1"}}
    for q in ["我最近鼻塞流鼻涕打喷嚏，该挂哪个科", "什么是高血压，平时饮食要注意什么"]:
        state = {"messages": [HumanMessage(content=q)], "consultations": []}
        resp = ragAgent.agent(state, config, store=MockStore(), llm_chat=llm_chat, tool_config=tool_config)
        last = resp["messages"][0]
        calls = [tc["name"] for tc in last.tool_calls] if getattr(last, "tool_calls", None) else []
        print(f"问题「{q}」→ 选择工具: {calls}")


if __name__ == "__main__":
    main()
