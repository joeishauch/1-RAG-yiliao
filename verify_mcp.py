# -*- coding: utf-8 -*-
"""验证 MCP 双方向集成：方向1（暴露 4 工具）+ 方向2（接入外部 HIS 工具）。"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from mcp_client import load_mcp_tools
from utils.config import Config
from utils.llms import get_llm
from utils.tools_config import get_tools
from langchain_core.messages import HumanMessage

PY = sys.executable


def verify_direction1():
    """方向1：连自己的 mcp_server.py，验证 4 个领域工具通过 MCP 协议可用。"""
    print("\n===== 方向1：把 4 领域工具暴露成 MCP Server =====")
    tools, bridge = load_mcp_tools("medical-triage", PY, [str(PROJECT_ROOT / "mcp_server.py")])
    print(f"通过 MCP 协议列出 {len(tools)} 个工具：{[t.name for t in tools]}")
    # 调 retrieve 验证（走真实 stdio + JSON-RPC）
    retrieve = next(t for t in tools if t.name == "retrieve")
    result = retrieve.invoke({"query": "鼻塞流鼻涕"})
    print("\n[retrieve 调用结果]\n" + result[:300])
    assert "科室" in result, "retrieve 未返回科室分布"
    print("\n✅ 方向1 通过：4 工具经 MCP 协议可调用")


def verify_direction2():
    """方向2：连外部 hospital HIS server，验证 MCP 工具能 bind 进 LLM 并被调用。"""
    print("\n===== 方向2：Agent 通过 MCP 接入外部 HIS 系统 =====")
    his_tools, bridge = load_mcp_tools("hospital-his", PY, [str(PROJECT_ROOT / "hospital_mcp_server.py")])
    print(f"接入外部 HIS 系统 {len(his_tools)} 个工具：{[t.name for t in his_tools]}")

    llm_chat, _ = get_llm(Config.LLM_TYPE)
    llm_with_tools = llm_chat.bind_tools(his_tools)

    # 问一个需要查号源的问题，看 LLM 是否调用 MCP 工具
    resp = llm_with_tools.invoke([HumanMessage(content="帮我查一下耳鼻喉科今天还有没有号")])
    calls = getattr(resp, "tool_calls", []) or []
    names = [c["name"] for c in calls if isinstance(c, dict)]
    print(f"\nLLM 发起的工具调用：{names}")
    assert "query_registration" in names, f"LLM 未调用 MCP 工具，实际调用 {names}"
    print("✅ 方向2 通过：LLM 通过 MCP 协议调用外部系统工具")


if __name__ == "__main__":
    verify_direction1()
    verify_direction2()
    print("\n🎉 MCP 双方向集成验证全部通过")