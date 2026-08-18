# -*- coding: utf-8 -*-
"""轻量 MCP → LangChain 工具适配器（方向2 核心，不依赖 langchain-mcp-adapters）。

用官方 mcp SDK 的 stdio_client 连接一个 MCP Server，把它的工具动态转成
LangChain StructuredTool。后台线程常驻 asyncio event loop，实现 async→sync 桥接。
"""
import asyncio
import threading
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import StructuredTool
from pydantic import create_model

# JSON Schema type → Python type（演示场景够用；复杂 schema 兜底为 str）
_TYPE_MAP = {"string": str, "integer": int, "number": float, "boolean": bool}


class _MCPBridge:
    """后台 event loop + 常驻 ClientSession，供同步工具跨线程调用。"""

    def __init__(self, command, args):
        self._params = StdioServerParameters(command=command, args=args)
        self._loop = asyncio.new_event_loop()
        self._session = None
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)

    def start(self):
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        fut.result(timeout=120)  # 阻塞直到连接建立（server 首次加载 Chroma/BM25 较慢）

    async def _connect(self):
        self._stdio_cm = stdio_client(self._params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

    def list_tools(self):
        fut = asyncio.run_coroutine_threadsafe(self._list(), self._loop)
        return fut.result(timeout=60)

    async def _list(self):
        return (await self._session.list_tools()).tools

    def call(self, tool_name, arguments):
        fut = asyncio.run_coroutine_threadsafe(self._call(tool_name, arguments), self._loop)
        return fut.result(timeout=180)

    async def _call(self, tool_name, arguments):
        result = await self._session.call_tool(tool_name, arguments=arguments)
        # 取首个 text 内容；无内容返回空串
        if not result.content:
            return ""
        first = result.content[0]
        return first.text if hasattr(first, "text") else str(first)


def load_mcp_tools(server_name, command, args):
    """连接一个 MCP Server，返回 (LangChain 工具列表, bridge)。

    调用方需持有返回的 bridge（保持后台线程/连接存活）。
    """
    bridge = _MCPBridge(command, args)
    bridge.start()
    tools = bridge.list_tools()

    langchain_tools = []
    for t in tools:
        # 依据 MCP 工具的 inputSchema 动态生成 pydantic args 模型
        schema = t.inputSchema or {"type": "object", "properties": {}}
        props = schema.get("properties", {})
        required = schema.get("required", []) or []
        fields = {}
        for name, spec in props.items():
            py_type = _TYPE_MAP.get(spec.get("type", "string"), str)
            fields[name] = (py_type, ... if name in required else None)
        args_model = create_model(f"{t.name}_args", **fields)

        def _make_func(name=t.name, bridge=bridge):
            def _func(**kwargs):
                return bridge.call(name, kwargs)
            return _func

        langchain_tools.append(
            StructuredTool.from_function(
                func=_make_func(),
                name=t.name,
                description=t.description or "",
                args_schema=args_model,
            )
        )
    return langchain_tools, bridge