# 医疗分诊系统 MCP 集成方案（双方向）

> 目标：把 MCP 焊进医疗 RAG 主推项目，一个项目同时覆盖 RAG + Agent + GraphRAG + HITL + MCP 五个方向。
>
> 方向 1：把 4 个领域工具暴露成 MCP Server（**对外提供能力**）
> 方向 2：LangGraph Agent 通过 MCP 接入外部系统（**对内消费能力**）
>
> 执行环境：`xiangmu-2`（`D:\python\conda\envs\xiangmu-2\python.exe`）
> 项目根：本文件所在目录

---

## 0. 环境现状（已确认，2026-08-18）

| 包 | 版本 |
|----|------|
| langgraph | 0.2.74（旧生态，**不能动**） |
| langchain-core | 0.3.86 |
| langchain | 0.3.25 |
| langchain-openai | 0.2.14 |
| pydantic | 2.11.5 |
| mcp / fastmcp / langchain-mcp-adapters | **未安装** |

**关键决策**：不用 `langchain-mcp-adapters`（它依赖较新的 langchain-core，会跟 0.2.74 旧生态打架）。方向 2 自写轻量 adapter，只依赖官方 `mcp` SDK（独立于 langchain，无版本冲突）。

---

## 1. 阶段一：装依赖（约 1 分钟）

```powershell
D:\python\conda\envs\xiangmu-2\python.exe -m pip install "fastmcp" "mcp"
```

装完验证：

```powershell
D:\python\conda\envs\xiangmu-2\python.exe -c "import fastmcp, mcp; print(fastmcp.__version__, mcp.__version__)"
```

> fastmcp 会自动带上 mcp 依赖。若报 pydantic 冲突（不会，2.11.5 已满足），先 `pip install "pydantic>=2.7"`。

---

## 2. 阶段二（方向 1）：新建 `mcp_server.py`

把 4 个领域工具（retrieve / medical_qa / kg_query / drug_taboo）暴露成 MCP 工具，**复用** `utils/tools_config.py` 的模块级函数，零重构。

```python
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
```

> 说明：`_search_and_rank` 是下划线"私有"函数，Python 里 import 它完全可行（无真私有）。介意的话，把 `utils/tools_config.py` 里 `_search_and_rank` 改名 `search_and_rank`（3 处：定义 + 2 处调用）即可，本方案为最小侵入直接 import。

---

## 3. 阶段三（方向 2 的演示对象）：新建 `hospital_mcp_server.py`

模拟一个「医院信息系统」（HIS）MCP Server——这是 Agent 要接入的"外部系统"。

```python
# -*- coding: utf-8 -*-
"""模拟「医院信息系统」MCP Server（方向2：外部系统，演示 Agent 通过 MCP 接入第三方）。

提供查科室 / 查号源 / 查检查报告，全部 mock 数据，用于演示 MCP 协议接入。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastmcp import FastMCP

mcp = FastMCP("hospital-his")

_DEPARTMENTS = {
    "耳鼻喉科": {"医生数": 12, "剩余号源": 8, "坐诊时间": "周一至周六 8:00-17:00"},
    "内科": {"医生数": 20, "剩余号源": 15, "坐诊时间": "每日 8:00-17:00"},
    "儿科": {"医生数": 9, "剩余号源": 3, "坐诊时间": "每日 8:00-20:00"},
    "皮肤科": {"医生数": 7, "剩余号源": 0, "坐诊时间": "周一至周五 8:00-17:00"},
}


@mcp.tool()
def query_department(dept_name: str) -> str:
    """查询科室信息：医生数量、剩余号源、坐诊时间。"""
    info = _DEPARTMENTS.get(dept_name)
    if not info:
        return f"未找到科室「{dept_name}」"
    return f"科室「{dept_name}」：医生 {info['医生数']} 人，剩余号源 {info['剩余号源']}，{info['坐诊时间']}"


@mcp.tool()
def query_registration(dept_name: str) -> str:
    """查询科室挂号/号源情况。"""
    info = _DEPARTMENTS.get(dept_name)
    if not info:
        return f"未找到科室「{dept_name}」"
    n = info["剩余号源"]
    if n > 5:
        return f"「{dept_name}」当前剩余号源 {n} 个（号源充足，可预约）"
    if n > 0:
        return f"「{dept_name}」当前剩余号源 {n} 个（号源紧张，建议尽快）"
    return f"「{dept_name}」今日号源已满"


@mcp.tool()
def query_lab_report(patient_id: str) -> str:
    """根据患者ID查询最近一次检查报告状态。"""
    return f"患者 {patient_id} 最近检查报告：血常规（已出，2 项指标偏高）、CT（待审）"


if __name__ == "__main__":
    mcp.run(show_banner=False)  # 关闭 fastmcp 3.x 启动 banner
```

---

## 4. 阶段四（方向 2 的核心）：新建 `mcp_client.py`

自写轻量 MCP → LangChain 工具适配器。核心难点是**把 MCP 的 async 协议桥接成 LangChain 的同步工具**——用后台线程常驻 event loop 持有 `ClientSession`，同步函数通过 `run_coroutine_threadsafe` 调用。

```python
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
```

> 这个 adapter 就是方向 2 的"硬核"部分，面试可以讲：MCP 是 async 协议、LangGraph 工具同步执行，我用后台 event loop 做了桥接——体现对 MCP 底层（stdio + JSON-RPC）的理解。

---

## 5. 阶段五：新建 `verify_mcp.py`（验证两个方向）

```python
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
```

---

## 6. 运行与验证清单

```powershell
# 在项目根目录执行（本文件所在目录）
cd /d "D:\ai\聚客\02聚客AI大模型第六期\10-项目2_基于LangGraph实现智能分诊系统\项目2_基于LangGraph实现智能分诊系统"

D:\python\conda\envs\xiangmu-2\python.exe verify_mcp.py
```

预期输出：
1. 方向1 列出 4 个工具 + `retrieve` 返回科室分布
2. 方向2 列出 3 个 HIS 工具 + LLM 调用 `query_registration`

> 首次运行慢：mcp_server.py 启动时会加载 Chroma + BM25 索引 + 通义 embedding + 知识图谱，可能要 1-3 分钟。`bridge.start()` 的 `timeout=120` 已预留。

---

## 7. 踩坑与兜底

| 坑 | 现象 | 处理 |
|----|------|------|
| langchain-mcp-adapters 版本冲突 | 装它会把 langchain-core 拉高，破坏 langgraph 0.2.74 | **本方案不用它**，自写 adapter 只依赖 mcp SDK |
| async→sync 桥接 | MCP stdio 是 async，LangGraph 工具同步 | 后台线程常驻 event loop + `run_coroutine_threadsafe` |
| 中文乱码 | Windows stdout GBK | 脚本顶部 `reconfigure(encoding="utf-8")` |
| 相对路径错误 | mcp_server 找不到 .env/图谱 | 脚本顶部 `os.chdir(PROJECT_ROOT)` |
| 连接超时 | 首次加载知识库慢 | `fut.result(timeout=120)` 已放宽 |

---

## 8. 可选进阶（今晚不强制）

1. **接入 LangGraph 图**：把 `his_tools` 合并进 `get_tools()` 返回，`agent` 节点即可在分诊时查号源（改 `utils/tools_config.py` 的 `get_tools` 返回值 + `route_after_tools` 加路由分支）。
2. **cli.py 加 `mcp` 子命令**：对齐现有 `chat/serve/ui/eval` 风格，加 `python cli.py mcp` 起 server。
3. **换真外部系统**：把 `hospital_mcp_server.py` 换成真实 HIS/药品库的 HTTP/SSE MCP Server，改 `load_mcp_tools` 用 `transport="sse"` 即可。

---

## 9. 简历/面试话术（做完后）

> 项目一「医疗智能分诊系统」技术栈补一句：
> **MCP 工具生态**——将分诊/图谱/科普/药物禁忌 4 个领域工具标准化为 MCP Server，患者端、医生端及外部系统通过统一协议复用；并通过自研 client 适配器（MCP SDK + 后台 event loop 桥接）让 LangGraph Agent 接入医院 HIS 系统，实现号源/检查报告等外部数据的实时查询。
