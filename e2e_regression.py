# -*- coding: utf-8 -*-
"""端到端回归：真实 PG + create_graph 编译 + 四链路真实图流验证。

与 verify_kg_integration.py 的「分项 mock」不同，本脚本走完整 create_graph：
真实 PostgresSaver（checkpoint）+ PostgresStore（记忆）+ ChromaDB 四工具，
验证 agent→call_tools→(consult→summarize / generate_kg / generate_qa / generate_drug) 四条链路。

用法（需 langgraph_postgres 容器已起，xiangmu-2 环境）：
    python e2e_regression.py
"""
import os
import sys

# 切换到脚本所在目录，保证 .env / chromaDB / prompts / kg_graph.pkl 等相对路径正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

from utils.config import Config
from utils.llms import get_llm
from utils.tools_config import get_tools
import ragAgent
from ragAgent import ToolConfig, create_graph
from psycopg_pool import ConnectionPool


def run_case(graph, question: str, thread_id: str) -> tuple:
    """跑单条 query 的完整图流，打印节点流转 + 工具调用 + 最终回答。

    Returns:
        (ok, nodes): ok 是否成功，nodes 节点流转列表。
    """
    config = {"configurable": {"thread_id": thread_id, "user_id": "e2e"}}
    nodes = []
    tool_names = []
    final_content = None
    print(f"\n{'=' * 70}\n问题：{question}\n{'=' * 70}")
    try:
        events = graph.stream(
            {"messages": [{"role": "user", "content": question}]},
            config,
            stream_mode="updates",
        )
        for event in events:
            for node, value in event.items():
                nodes.append(node)
                print(f"  → 节点 [{node}]")
                if isinstance(value, dict) and "messages" in value:
                    for m in value["messages"]:
                        # 工具调用（AI 消息上的 tool_calls）
                        if getattr(m, "tool_calls", None):
                            for tc in m.tool_calls:
                                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                                if name:
                                    tool_names.append(name)
                                    print(f"      工具调用: {name}")
                        # 工具结果消息
                        if getattr(m, "type", "") == "tool":
                            tool_names.append(getattr(m, "name", ""))
                        # 最终 AI 回答（有 content 的 AI 消息）
                        content = getattr(m, "content", "")
                        if content and getattr(m, "type", "") == "ai":
                            final_content = content
        ok = True
    except Exception as e:
        print(f"  ❌ 失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        ok = False

    print(f"\n  节点流转: {' -> '.join(nodes)}")
    print(f"  使用工具: {list(dict.fromkeys(tool_names))}")
    if final_content:
        print(f"  最终回答:\n{final_content[:1500]}")
    return ok, nodes


def main():
    print("初始化 LLM 与工具 ...")
    llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
    tools = get_tools(llm_embedding)
    print(f"工具列表: {[t.name for t in tools]}")
    tool_config = ToolConfig(tools)

    print("\n建立 PostgreSQL 连接池 ...")
    connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 5}
    db_connection_pool = ConnectionPool(
        conninfo=Config.DB_URI, max_size=20, min_size=2,
        kwargs=connection_kwargs, timeout=10,
    )
    db_connection_pool.open()
    print("连接池打开 OK")

    try:
        print("\n编译状态图（PostgresSaver + PostgresStore）...")
        graph = create_graph(db_connection_pool, llm_chat, llm_embedding, tool_config)
        print("✅ create_graph 编译成功")

        cases = [
            ("分诊链路 retrieve→consult→summarize", "我鼻塞流鼻涕该挂哪个科", "t1"),
            ("知识图谱链路 kg_query→generate_kg", "胸痛可能是哪些病", "t2"),
            ("科普链路 medical_qa→generate_qa", "什么是高血压平时要注意什么", "t3"),
            ("药物禁忌链路 drug_taboo→generate_drug", "孕妇能吃阿司匹林吗", "t4"),
        ]
        results = {}
        for label, q, tid in cases:
            results[label] = run_case(graph, q, tid)

        print(f"\n{'=' * 70}\n端到端回归结果汇总\n{'=' * 70}")
        all_ok = True
        for label, (ok, nodes) in results.items():
            mark = "✅" if ok else "❌"
            print(f"  {mark} {label}: {' -> '.join(nodes)}")
            all_ok = all_ok and ok
        print(f"\n总体：{'✅ 全部链路跑通' if all_ok else '❌ 存在失败链路'}")
    finally:
        db_connection_pool.close()
        print("连接池已关闭")


if __name__ == "__main__":
    main()
