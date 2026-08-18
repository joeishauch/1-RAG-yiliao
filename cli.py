# -*- coding: utf-8 -*-
"""智能分诊系统 — 统一命令行入口（对齐 MinerU cli.py 风格）。

把散落的运行/评估入口收进一个 argparse 子命令入口，各命令内部「延迟导入」，
避免 `python cli.py`（无命令）时加载 chromadb/langchain/torch 等重依赖。

用法:
    python cli.py chat                 交互式问答（命令行）
    python cli.py serve                启动 FastAPI API 服务
    python cli.py ui                   启动 Gradio Web 界面（用户端）
    python doctor_ui.py                启动 Gradio Web 界面（医生端）
    python cli.py eval retrieval       检索质量评估（Hit@K/MRR）
    python cli.py eval judge            LLM-as-judge 分诊质量评估
    python cli.py eval e2e             端到端四链路回归
    python cli.py mcp                  启动 MCP Server（方向1：对外提供 4 领域工具）
    python cli.py mcp --his            启动医院 HIS MCP Server（方向2 外部系统 mock）

低频数据脚本（build_kg / jsonl2chroma / extract_drug_contra / run_label_audit）
保持独立脚本，不收入此入口。
"""
import argparse
import os
import sys
import uuid
from pathlib import Path

# 项目根目录 = 本脚本所在目录（L1-Project-2），统一 CWD 保证延迟导入脚本的相对路径正确
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
os.makedirs("output", exist_ok=True)

# Windows 控制台中文编码：stdout/stderr/stdin 走 UTF-8，避免中文乱码（等价 python -X utf8）
for _stream in (sys.stdout, sys.stderr, sys.stdin):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


# ---------------- chat：交互式问答 ----------------

def _extract_last_ai_content(state):
    """逆序遍历 state 的消息，取最后一条有 content 的 AI 消息。"""
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
            return m.content
    return None


def _terminal_review():
    """终端人工审核交互：输入 approve/revise/reject；revise 时继续读改写文本。"""
    while True:
        a = input("审核 [approve/revise/reject]：").strip().lower()
        if a in ("approve", "a"):
            return {"action": "approve", "reviewer": "cli"}
        if a in ("reject", "r"):
            return {"action": "reject", "reviewer": "cli"}
        if a in ("revise", "v"):
            text = input("改写后的回答：").strip()
            return {"action": "revise", "revised_answer": text, "reviewer": "cli"}
        print("无效输入，请输入 approve / revise / reject")


def stream_answer(graph, question, thread_id, user_id="cli", verbose=False):
    """跑一次图流，返回最终 AI 回答文本；verbose 时额外打印节点流转 + 工具调用。

    与 e2e_regression.run_case 同款提取逻辑：遍历 updates 流，取最后一个
    有 content 的 AI 消息作为最终回答（分诊=summarize、其余=generate_* 节点）。
    """
    # 延迟导入：保持 cli.py 顶部轻量（无命令时避免加载 langgraph 等重依赖）
    from langgraph.types import Command
    from utils.privacy import desensitize
    from utils.safety import check_input_danger, check_output_diagnostic
    from utils.audit import write_audit, build_record

    # 入口脱敏（先脱敏，保证后续所有审计事件——含 block——的 user_input 都不含 PII）
    question, redaction_count = desensitize(question)
    # 入口危险信号拦截（脱敏后文本，危险词不受脱敏影响；不进图）
    blocked = check_input_danger(question)
    if blocked:
        write_audit(build_record(event="block", thread_id=thread_id, user_id=user_id,
                                 user_input=question,
                                 redacted=redaction_count > 0,
                                 redaction_count=redaction_count))
        return blocked

    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    nodes, tool_names, final_content = [], [], None
    try:
        events = graph.stream(
            {"messages": [{"role": "user", "content": question}]},
            config,
            stream_mode="updates",
        )
        for event in events:
            # 高风险链路人工审核中断：打印草稿，读终端审核结果，续跑
            if "__interrupt__" in event:
                payload = event["__interrupt__"][0].value
                draft = payload.get("draft", "")
                risk = payload.get("risk_level", "")
                write_audit(build_record(event="draft", thread_id=thread_id, user_id=user_id,
                                         risk_level=risk, draft=draft,
                                         redacted=redaction_count > 0,
                                         redaction_count=redaction_count))
                print(f"\n[待人工审核] 风险等级：{risk}")
                print(f"[草稿]\n{draft}\n")
                decision = _terminal_review()
                final_state = graph.invoke(Command(resume=decision), config)
                final_content = _extract_last_ai_content(final_state)
                if final_content:
                    hint = check_output_diagnostic(final_content)
                    if hint:
                        final_content = final_content + "\n\n" + hint
                write_audit(build_record(event="review_decision", thread_id=thread_id, user_id=user_id,
                                         action=decision.get("action"),
                                         revised_answer=decision.get("revised_answer"),
                                         reviewer=decision.get("reviewer"),
                                         redacted=redaction_count > 0,
                                         redaction_count=redaction_count))
                write_audit(build_record(event="final", thread_id=thread_id, user_id=user_id,
                                         final_answer=final_content,
                                         redacted=redaction_count > 0,
                                         redaction_count=redaction_count))
                if verbose:
                    print(f"\n[节点流转] {' -> '.join(nodes)} -> review")
                return final_content
            for node, value in event.items():
                nodes.append(node)
                if isinstance(value, dict) and "messages" in value:
                    for m in value["messages"]:
                        if getattr(m, "tool_calls", None):
                            for tc in m.tool_calls:
                                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                                if name:
                                    tool_names.append(name)
                        if getattr(m, "type", "") == "tool":
                            tool_names.append(getattr(m, "name", ""))
                        content = getattr(m, "content", "")
                        if content and getattr(m, "type", "") == "ai":
                            final_content = content  # 覆盖式：取最后一个 AI 回答
    except Exception as e:
        print(f"❌ 出错: {type(e).__name__}: {e}")
        return None

    if verbose:
        print(f"\n[节点流转] {' -> '.join(nodes)}")
        if tool_names:
            print(f"[使用工具] {list(dict.fromkeys(tool_names))}")
    # 出口诊断性表述检测 + 审计（低/中风险直出，或高风险未开启审核时）
    if final_content:
        hint = check_output_diagnostic(final_content)
        if hint:
            final_content = final_content + "\n\n" + hint
    write_audit(build_record(event="final", thread_id=thread_id, user_id=user_id,
                             final_answer=final_content,
                             redacted=redaction_count > 0,
                             redaction_count=redaction_count))
    return final_content


def _build_graph():
    """初始化 LLM + 工具 + PG 连接池 + 编译状态图，返回 (graph, pool)。"""
    from utils.config import Config
    from utils.llms import get_llm
    from utils.tools_config import get_tools
    from ragAgent import ToolConfig, create_graph
    from psycopg_pool import ConnectionPool

    print("初始化 LLM 与工具 ...")
    llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
    tools = get_tools(llm_embedding)
    tool_config = ToolConfig(tools)
    print(f"工具: {[t.name for t in tools]}")

    print("建立 PostgreSQL 连接池 ...")
    connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 5}
    pool = ConnectionPool(
        conninfo=Config.DB_URI, max_size=20, min_size=2,
        kwargs=connection_kwargs, timeout=10,
    )
    pool.open()

    print("编译状态图（PostgresSaver + PostgresStore）...")
    graph = create_graph(pool, llm_chat, llm_embedding, tool_config)
    print("✅ 编译成功")
    return graph, pool


def cmd_chat(args):
    """交互式命令行问答"""
    print("=" * 60)
    print("智能分诊系统 - 命令行交互（chat）")
    print("=" * 60)

    graph, pool = _build_graph()
    try:
        thread_id = f"cli_{uuid.uuid4().hex[:8]}"
        print("\n提示：输入症状或问题；exit / quit / 退出 = 结束；/new = 开新会话\n")
        while True:
            try:
                question = input("你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit") or question in ("退出", "结束"):
                print("再见！")
                break
            if question in ("/new", "/新会话"):
                thread_id = f"cli_{uuid.uuid4().hex[:8]}"
                print("✅ 已开启新会话\n")
                continue
            answer = stream_answer(graph, question, thread_id, verbose=args.verbose)
            if answer:
                print(f"\n助手：\n{answer}\n")
            else:
                print("\n（本次未生成回答）\n")
    finally:
        pool.close()
        print("连接池已关闭")


def cmd_serve(args):
    """启动 FastAPI API 服务"""
    from main import app
    import uvicorn
    print(f"FastAPI 服务启动: http://{args.host}:{args.port}")
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_ui(args):
    """启动 Gradio Web 界面"""
    from webUI import demo
    print("Gradio 界面启动: http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)


def cmd_eval_retrieval(args):
    """检索质量评估"""
    from eval_triage_retrieval import main
    main()


def cmd_eval_judge(args):
    """LLM-as-judge 分诊质量评估（可选 --limit 限制用例数）"""
    from eval_triage_judge import main
    sys.argv = ["eval_triage_judge.py"]
    if args.limit:
        sys.argv.append(str(args.limit))
    main()


def cmd_eval_e2e(args):
    """端到端四链路回归"""
    from e2e_regression import main
    main()


def cmd_mcp(args):
    """启动 MCP Server（默认方向1：对外提供 4 领域工具；--his 起外部系统 mock）"""
    if args.his:
        from hospital_mcp_server import mcp
        print("医院 HIS MCP Server 启动（方向2：外部系统 mock，stdio 传输）...", flush=True)
    else:
        from mcp_server import mcp
        print("医疗分诊 MCP Server 启动（方向1：对外提供 4 领域工具，stdio 传输）...", flush=True)
    mcp.run(show_banner=False)


def main():
    parser = argparse.ArgumentParser(
        description="智能分诊系统 — 统一命令行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python cli.py chat                 交互式问答
  python cli.py chat -v              交互式问答（打印节点流转+工具）
  python cli.py serve --port 8012    启动 API 服务（自定义端口）
  python cli.py ui                   启动 Web 界面
  python cli.py eval retrieval       检索质量评估
  python cli.py eval judge --limit 5 LLM-as-judge 评估（5 条）
  python cli.py eval e2e             端到端四链路回归
  python cli.py mcp                  启动 MCP Server（方向1：对外提供 4 工具）
  python cli.py mcp --his            启动医院 HIS MCP Server（方向2 外部系统 mock）
        """,
    )

    sub = parser.add_subparsers(dest="command", help="可用命令")

    # chat
    p_chat = sub.add_parser("chat", help="交互式命令行问答")
    p_chat.add_argument("-v", "--verbose", action="store_true", help="打印节点流转 + 工具调用")

    # serve
    p_serve = sub.add_parser("serve", help="启动 FastAPI 服务")
    p_serve.add_argument("--host", default="0.0.0.0", help="监听地址")
    p_serve.add_argument("--port", type=int, default=8012, help="监听端口")

    # ui
    sub.add_parser("ui", help="启动 Gradio Web 界面")

    # mcp
    p_mcp = sub.add_parser("mcp", help="启动 MCP Server")
    p_mcp.add_argument("--his", action="store_true", help="启动医院 HIS MCP Server（方向2 外部系统 mock）")

    # eval（二级子命令）
    p_eval = sub.add_parser("eval", help="质量评估")
    p_eval_sub = p_eval.add_subparsers(dest="eval_type", help="评估类型")
    p_eval_sub.add_parser("retrieval", help="检索质量评估（Hit@K/MRR）")
    p_eval_judge = p_eval_sub.add_parser("judge", help="LLM-as-judge 分诊质量评估")
    p_eval_judge.add_argument("--limit", type=int, default=None, help="评估用例数（默认全量）")
    p_eval_sub.add_parser("e2e", help="端到端四链路回归")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "eval":
        eval_routes = {
            "retrieval": cmd_eval_retrieval,
            "judge": cmd_eval_judge,
            "e2e": cmd_eval_e2e,
        }
        if args.eval_type is None:
            p_eval.print_help()
            return
        eval_routes[args.eval_type](args)
    else:
        routes = {
            "chat": cmd_chat,
            "serve": cmd_serve,
            "ui": cmd_ui,
            "mcp": cmd_mcp,
        }
        routes[args.command](args)


if __name__ == "__main__":
    main()
