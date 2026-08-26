# -*- coding: utf-8 -*-
"""智能分诊系统 — 统一命令行入口。

把散落的运行/评估入口收进一个 argparse 子命令入口，各命令内部「延迟导入」，
避免 `python cli.py`（无命令）时加载 chromadb/langchain/torch 等重依赖。

用法:
    python cli.py chat                 交互式问答（命令行）
    python cli.py serve                启动 FastAPI API 服务
    python cli.py ui                   启动 Gradio Web 界面（用户端）
    python doctor_ui.py                启动 Gradio Web 界面（医生端）
    python cli.py sync [--source ...]   文档同步（变更感知 → 增量/全量 → 缓存失效）
    python cli.py sync --migrate        一次性回填 doc_id
    python cli.py sync --migrate-embedding  B.4：补 embedding_model 标签
    python cli.py eval retrieval       检索质量评估（Hit@K/MRR）
    python cli.py eval judge            LLM-as-judge 分诊质量评估
    python cli.py eval e2e             端到端四链路回归
    python cli.py mcp                  启动 MCP Server（方向1：对外提供 4 领域工具）
    python cli.py mcp --his            启动医院 HIS MCP Server（方向2 外部系统 mock）
    python cli.py backup               B.6：ChromaDB 灾备快照
    python cli.py cleanup              清理旧测试 collection
    python cli.py verify b3-kg-sync    B.3 KG↔向量联动 7 项 mock 测试
    python cli.py verify b3-version    B.3 KG 图缓存版本校验
    python cli.py verify drug           药物禁忌查询验证
    python cli.py verify kg             KG 多跳推理验证
    python cli.py verify kg-integration KG 接入 LangGraph 验证
    python cli.py verify label          科室标签均衡性
    python cli.py verify mcp            MCP Server 验证
    python cli.py verify qa             医疗问答质量
    python cli.py verify qa-retrieval   医疗问答检索质量
    python cli.py verify symptom-match  症状模糊匹配验证
    python cli.py metrics              B.7：读取 metrics.jsonl 汇总或最近记录
    python cli.py metrics --last 20     显示最近 20 条原始指标
    python cli.py dedup --sources a,b --dry-run  B.9：跨源精确去重预览

低频数据脚本（build_kg / jsonl2chroma / extract_drug_contra / run_label_audit）
保持独立脚本，不收入此入口。
"""
import argparse
import json
import os
import sys
import time
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


def cmd_eval_qa_retrieval(args):
    """医疗问答检索质量评估（B.2 新建，32 用例）"""
    from eval_qa_retrieval import main
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


# ---------------- backup：灾备快照 ----------------

def cmd_backup(args):
    """ChromaDB 灾备快照（打包/恢复/列表三合一）。

    转发参数给 backup.py 的 main()。"""
    import backup as _backup
    argv = ["backup.py"]
    if args.list:
        argv.append("--list")
    if args.no_cleanup:
        argv.append("--no-cleanup")
    if args.dry_run:
        argv.append("--dry-run")
    if args.out:
        argv += ["--out", args.out]
    if args.restore:
        argv += ["--restore", args.restore]
    # backup.py 的 main() 用 sys.argv
    sys.argv = argv
    _backup.main()


# ---------------- cleanup：清理旧测试 collection ----------------

def cmd_cleanup(args):
    """清理旧测试 collection（保留 medical_triage / medical_qa）。"""
    import cleanup_collections as _cleanup
    argv = ["cleanup_collections.py"]
    if args.confirm:
        argv.append("--confirm")
    sys.argv = argv
    _cleanup.main()


def cmd_quality(args):
    """B.8：扫描指定 source 的质量门禁并写报告。

    默认不触 Chroma、不调 embedding；仅按 parser 解析 + 确定性规则。
    """
    from quality_gate import QualityGate
    from jsonl2chroma import SOURCES, _parse_valid, _balanced_parsed

    source = next((s for s in SOURCES if s["name"] == args.source), None)
    if not source:
        print(f"未找到 source: {args.source}，可选: {[s['name'] for s in SOURCES]}")
        return
    gate = QualityGate(source["name"])
    if args.dry_run:
        gate.rejected_path = gate.rejected_path.with_suffix(
            gate.rejected_path.suffix + ".dryrun"
        )
    if source.get("per_label_limit") and args.limit is None:
        records = list(_balanced_parsed(source, source["per_label_limit"]))
    else:
        from jsonl2chroma import _read_records
        records = []
        for idx, raw in enumerate(_read_records(source)):
            if args.limit and idx >= args.limit:
                break
            parsed = _parse_valid(source, raw)
            records.append(parsed)
    accepted = gate.check_batch(records)
    report_path = gate.write_report(extra={
        "source": source["name"],
        "dry_run": args.dry_run,
        "limit": args.limit,
        "accepted_count": len(accepted),
    })
    print(f"[quality] source={source['name']} 总={gate.stats.total} "
          f"accepted={gate.stats.accepted} rejected={gate.stats.rejected} "
          f"rate={gate.stats.rejection_rate:.2%}")
    print(f"[quality] 拒绝原因分布：{dict(gate.stats.rejected_by_reason)}")
    print(f"[quality] 报告：{report_path}")


# ---------------- dedup：跨源精确去重预览 ----------------

def cmd_dedup(args):
    """扫描 source 并预览跨源去重；此命令永不触碰 ChromaDB。"""
    from collections import Counter
    from dedup import dedup_and_write
    from jsonl2chroma import SOURCES, _collect_source_candidates
    from utils.config import Config

    requested = []
    for value in args.sources:
        requested.extend(part.strip() for part in value.split(",") if part.strip())
    known = {source["name"]: source for source in SOURCES}
    unknown = [name for name in requested if name not in known]
    if unknown:
        print(f"未找到 source: {unknown}，可选: {list(known)}")
        return
    selected = [source for source in SOURCES if source["name"] in set(requested)]
    if len(selected) < 2:
        print("至少需要两个不同 source 才能执行跨源去重预览")
        return

    candidates = []
    for source in selected:
        candidates.extend(_collect_source_candidates(source, args.limit))

    result = dedup_and_write(
        candidates,
        report_path=args.report or Config.DEDUP_REPORT_PATH,
        dropped_path=args.dropped or Config.DEDUP_DROPPED_PATH,
        context={
            "path": "cli dedup",
            "dry_run": True,
            "sources": [source["name"] for source in selected],
            "scope": "collection",
        },
    )
    try:
        from utils.audit import build_record, write_audit
        write_audit(build_record(
            event="dedup",
            thread_id="cli::dedup",
            user_id="cli",
            comment=json.dumps({
                "sources": [source["name"] for source in selected],
                "input": result.input_count,
                "kept": result.kept_count,
                "dropped": result.dropped_count,
                "report_path": str(args.report or Config.DEDUP_REPORT_PATH),
            }, ensure_ascii=False),
        ))
    except Exception as e:
        print(f"[dedup] 审计写入失败（不影响预览）：{e}")
    drop_counts = Counter(item["source"] for item in result.dropped)
    print(f"[dedup] sources={','.join(s['name'] for s in selected)}")
    print(f"[dedup] input={result.input_count} kept={result.kept_count} dropped={result.dropped_count}")
    print(f"[dedup] dropped_by_source={dict(drop_counts)}")
    print(f"[dedup] report={args.report or Config.DEDUP_REPORT_PATH}")
    print(f"[dedup] dropped_log={args.dropped or Config.DEDUP_DROPPED_PATH}")
    print("[dedup] 仅扫描与写审计报告，未初始化 embedding，未修改 ChromaDB 或 sync manifest")



def cmd_metrics(args):
    """B.7：读取 output/metrics.jsonl 并汇总。

    --last N    显示最近 N 条原始记录
    --window M  仅汇总最近 M 分钟（默认全量）
    --reset     清空 metrics.jsonl（仅测试用）
    """
    import metrics as _metrics

    if args.reset:
        _metrics.reset()
        print("[metrics] 已清空 output/metrics.jsonl")
        return

    if args.diagnose:
        findings = _metrics.diagnose()
        print(f"[metrics] 诊断 {len(findings)} 条：")
        for finding in findings:
            value = finding.get("value", "")
            metric = finding.get("metric", finding.get("reason", ""))
            print(f"  [{finding['severity']}] {metric} value={value} "
                  f"{finding.get('message', '')}")
        return

    if args.last:
        records = _metrics.read_metrics(limit=args.last)
        print(f"[metrics] 最近 {len(records)} 条原始记录：")
        for r in records:
            print(f"  {r['ts']}  {r['metric']:<60}  {r['value']}")
        return

    window = args.window if args.window > 0 else None
    summary = _metrics.summarize(window_minutes=window)
    if not summary:
        print("[metrics] 暂无数据，先运行一次 sync / backup / eval 等")
        return

    label = f"最近 {window} 分钟" if window else "全部"
    print(f"[metrics] 汇总（{label}）：")
    print(f"  {'指标':<60} {'次数':>6} {'累计':>14} {'平均':>12} {'最大':>12}")
    print("  " + "-" * 108)
    for name in sorted(summary.keys()):
        s = summary[name]
        kind = s.get("kind", "counter")
        if kind == "gauge":
            print(f"  {name:<60} {'gauge':>6} {'latest':>14} {s.get('latest', s['avg']):>12.4f} {s['max']:>12.4f}")
        else:
            print(f"  {name:<60} {s['count']:>6} {s['sum']:>14.4f} {s['avg']:>12.4f} {s['max']:>12.4f}")


# ---------------- verify：分项验证 ----------------

def _run_verify_script(module_name, argv_extra=None):
    """通过 subprocess 调用 verify_* 脚本，避免污染主进程 sys.argv。"""
    import subprocess
    cmd = [sys.executable, f"{module_name}.py"]
    if argv_extra:
        cmd += argv_extra
    print(f"[verify] 运行：{' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
    if rc != 0:
        sys.exit(rc)


def cmd_verify_b3_kg_sync(args):
    """B.3：KG↔向量联动 7 项 mock 集成测试。"""
    _run_verify_script("verify_b3_kg_sync")


def cmd_verify_b3_version(args):
    """B.3：KG 图缓存版本校验单测。"""
    _run_verify_script("verify_b3_version")


def cmd_verify_drug_taboo(args):
    """药物禁忌查询分项验证（drug_taboo 工具 + 生成节点 + 路由 + agent 工具选择）。"""
    _run_verify_script("verify_drug_taboo")


def cmd_verify_kg(args):
    """KG 多跳推理验证：抽取质量统计 + 多跳合理性 + trace。"""
    argv = []
    if args.sample:
        argv += ["--sample", str(args.sample)]
    _run_verify_script("verify_kg", argv or None)


def cmd_verify_kg_integration(args):
    """KG 接入 LangGraph 的分项验证。"""
    _run_verify_script("verify_kg_integration")


def cmd_verify_label(args):
    """科室标签均衡性评估。"""
    argv = []
    if args.limit:
        argv += ["--limit", str(args.limit)]
    _run_verify_script("verify_label_balance", argv or None)


def cmd_verify_mcp(args):
    """MCP Server 端到端验证。"""
    _run_verify_script("verify_mcp")


def cmd_verify_qa(args):
    """医疗问答质量验证。"""
    _run_verify_script("verify_qa")


def cmd_verify_qa_retrieval(args):
    """医疗问答检索质量评估（medical_qa，32 用例）。"""
    _run_verify_script("verify_qa_retrieval")


def cmd_verify_symptom_match(args):
    """症状模糊匹配验证（口语症状 → 规范症状节点）。"""
    _run_verify_script("verify_symptom_match")


def cmd_sync(args):
    """文档同步：变更感知 → doc 级全删全重建 → 缓存失效 → 审计留痕。

    支持 --once（默认）和 --watch（守护模式）；--migrate 一次性回填 doc_id。
    进程级文件锁 acquire_sync_lock 包裹，并发 --watch 安全。
    """
    from doc_sync import acquire_sync_lock, run_migration, sync_all
    from jsonl2chroma import SOURCES
    from utils.llms import get_embedding

    sources = [s for s in SOURCES if args.source is None or s["name"] in args.source]
    if not sources:
        print(f"未找到数据源。可选: {[s['name'] for s in SOURCES]}")
        return

    # dry-run 不需要 embedding；其它模式按 --embedding-type 选源（deepseek/zhipu/local_bge）
    llm_embedding = None if args.dry_run else get_embedding(args.embedding_type)

    # --migrate：一次性回填 doc_id 给存量无标识 chunk（不重 embed）
    if args.migrate:
        with acquire_sync_lock():
            run_migration(sources, llm_embedding, args.manifest)
        print(f"[migrate] 完成：{len(sources)} 个 source 已尝试回填 doc_id")
        return

    # B.4 --migrate-embedding：给存量 chunks 补 embedding_model 字段（不重 embed）
    if args.migrate_embedding:
        from doc_sync import run_embedding_migration
        with acquire_sync_lock():
            run_embedding_migration()
        print("[migrate-embedding] 完成：存量 chunks 已补 embedding_model 字段")
        return

    # --once / --watch 循环
    while True:
        with acquire_sync_lock():
            results = sync_all(
                sources,
                llm_embedding,
                dry_run=args.dry_run,
                override_collection=args.collection,
                rebuild=args.rebuild,
                manifest_path=args.manifest,
                include_demo=args.include_demo,
            )
        for r in results:
            err = f" err={r.error}" if r.error else ""
            print(
                f"[{r.source}] status={r.status} inserted={r.inserted} "
                f"deleted={r.deleted} duration={r.duration_s:.1f}s{err}"
            )
        if not args.watch:
            break
        time.sleep(args.interval)


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
  python cli.py backup               灾备快照
  python cli.py backup --list        列出所有备份
  python cli.py cleanup --confirm    清理旧测试 collection
  python cli.py verify b3-kg-sync    B.3 联动验证
  python cli.py verify drug           药物禁忌验证
  python cli.py sync --migrate-embedding  B.4：补 embedding 标签
  python cli.py metrics              B.7：查看指标汇总
  python cli.py metrics --last 20     B.7：查看最近 20 条原始记录
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

    # sync（同步方案 A.6：变更感知+全删全重建+缓存失效）
    p_sync = sub.add_parser("sync", help="文档同步（变更感知 → doc 级全删全重建 → 缓存失效）")
    p_sync.add_argument("--source", action="append", default=None,
                        help="指定 source 名（可重复；不传则全部）。可选: huatuo_lite / huatuo_encyclopedia / huatuo_knowledge_graph / chinese_medical_dialogue")
    p_sync.add_argument("--collection", type=str, default=None, help="覆盖目标 collection（小样本验证灌到临时库）")
    p_sync.add_argument("--dry-run", action="store_true", help="只统计变更不写库")
    p_sync.add_argument("--rebuild", action="store_true", help="强制清空目标 collection 后重建（review 补强 #10：默认排除 demo001）")
    p_sync.add_argument("--include-demo", action="store_true", help="--rebuild 时包含 demo001 子集（默认排除）")
    p_sync.add_argument("--migrate", action="store_true", help="一次性回填 doc_id 给存量无标识 chunk（不重 embed）")
    p_sync.add_argument("--migrate-embedding", action="store_true", help="B.4：给存量 chunks 补 embedding_model 字段（不重 embed）")
    p_sync.add_argument("--manifest", type=str, default=None, help="自定义 manifest 路径")
    p_sync.add_argument("--watch", action="store_true", help="守护模式：循环执行")
    p_sync.add_argument("--interval", type=int, default=60, help="--watch 间隔秒数（默认 60）")
    p_sync.add_argument("--embedding-type", type=str, default=None,
                        help="embedding 来源：None=走 .env LLM_EMBEDDING_TYPE；deepseek/zhipu/local_bge")

    # eval（二级子命令）
    p_eval = sub.add_parser("eval", help="质量评估")
    p_eval_sub = p_eval.add_subparsers(dest="eval_type", help="评估类型")
    p_eval_sub.add_parser("retrieval", help="分诊检索质量评估（Hit@K/MRR，50 用例）")
    p_eval_sub.add_parser("qa", help="医疗问答检索质量评估（medical_qa，32 用例）")
    p_eval_judge = p_eval_sub.add_parser("judge", help="LLM-as-judge 分诊质量评估")
    p_eval_judge.add_argument("--limit", type=int, default=None, help="评估用例数（默认全量）")
    p_eval_sub.add_parser("e2e", help="端到端四链路回归")

    # backup（B.6 灾备快照）
    p_backup = sub.add_parser("backup", help="B.6 灾备快照（打包/恢复/列表）")
    p_backup.add_argument("--out", type=str, default=None, help="备份目录（默认 backups/）")
    p_backup.add_argument("--list", action="store_true", help="列出所有备份")
    p_backup.add_argument("--restore", type=str, default=None, help="从指定备份文件恢复")
    p_backup.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p_backup.add_argument("--no-cleanup", action="store_true", help="跳过过期备份清理")

    # cleanup（清理旧测试 collection）
    p_cleanup = sub.add_parser("cleanup", help="清理旧测试 collection（默认预览）")
    p_cleanup.add_argument("--confirm", action="store_true", help="确认执行删除")

    # dedup（B.9 只读预览）
    p_dedup = sub.add_parser("dedup", help="B.9：跨源精确去重预览（永不修改 ChromaDB）")
    p_dedup.add_argument("--sources", nargs="+", required=True,
                         help="source 名，可空格或逗号分隔")
    p_dedup.add_argument("--limit", type=int, default=None,
                         help="每个 source 扫描的原始记录数")
    p_dedup.add_argument("--dry-run", action="store_true", default=True,
                         help="显式只读模式（该命令始终只读）")
    p_dedup.add_argument("--report", type=str, default=None, help="覆盖报告路径")
    p_dedup.add_argument("--dropped", type=str, default=None, help="覆盖 dropped JSONL 路径")

    p_metrics = sub.add_parser("metrics", help="B.7：读取 metrics.jsonl 汇总或最近记录")
    p_metrics.add_argument("--last", type=int, default=0, help="显示最近 N 条原始记录")
    p_metrics.add_argument("--window", type=int, default=0, help="仅汇总最近 N 分钟（默认全量）")
    p_metrics.add_argument("--reset", action="store_true", help="清空 metrics.jsonl（仅测试用）")
    p_metrics.add_argument("--diagnose", action="store_true", help="诊断历史指标异常（不修改原始文件）")

    # quality（B.8 数据质量门禁）
    p_quality = sub.add_parser("quality", help="B.8：扫描指定 source 的质量门禁并写报告")
    p_quality.add_argument("--source", type=str, default="huatuo_lite",
                           help="可选: huatuo_lite / huatuo_encyclopedia / huatuo_knowledge_graph / chinese_medical_dialogue")
    p_quality.add_argument("--limit", type=int, default=None, help="只扫描前 N 条原始记录（不触库）")
    p_quality.add_argument("--dry-run", action="store_true", help="只扫描，不写入 rejected_chunks.jsonl")

    # verify（分项验证）
    p_verify = sub.add_parser("verify", help="分项验证脚本")
    p_verify_sub = p_verify.add_subparsers(dest="verify_type", help="验证类型")
    p_verify_sub.add_parser("b3-kg-sync", help="B.3 KG↔向量联动 mock 测试（7 项）")
    p_verify_sub.add_parser("b3-version", help="B.3 KG 图缓存版本校验")
    p_verify_sub.add_parser("drug", help="药物禁忌查询分项验证")
    p_v_kg = p_verify_sub.add_parser("kg", help="KG 多跳推理验证")
    p_v_kg.add_argument("--sample", type=int, default=None, help="抽样条数")
    p_verify_sub.add_parser("kg-integration", help="KG 接入 LangGraph 分项验证")
    p_v_label = p_verify_sub.add_parser("label", help="科室标签均衡性评估")
    p_v_label.add_argument("--limit", type=int, default=None, help="评估用例数")
    p_verify_sub.add_parser("mcp", help="MCP Server 端到端验证")
    p_verify_sub.add_parser("qa", help="医疗问答质量验证")
    p_verify_sub.add_parser("qa-retrieval", help="医疗问答检索质量评估")
    p_verify_sub.add_parser("symptom-match", help="症状模糊匹配验证")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "eval":
        eval_routes = {
            "retrieval": cmd_eval_retrieval,
            "qa": cmd_eval_qa_retrieval,
            "judge": cmd_eval_judge,
            "e2e": cmd_eval_e2e,
        }
        if args.eval_type is None:
            p_eval.print_help()
            return
        eval_routes[args.eval_type](args)
    elif args.command == "verify":
        verify_routes = {
            "b3-kg-sync": cmd_verify_b3_kg_sync,
            "b3-version": cmd_verify_b3_version,
            "drug": cmd_verify_drug_taboo,
            "kg": cmd_verify_kg,
            "kg-integration": cmd_verify_kg_integration,
            "label": cmd_verify_label,
            "mcp": cmd_verify_mcp,
            "qa": cmd_verify_qa,
            "qa-retrieval": cmd_verify_qa_retrieval,
            "symptom-match": cmd_verify_symptom_match,
        }
        if args.verify_type is None:
            p_verify.print_help()
            return
        verify_routes[args.verify_type](args)
    else:
        routes = {
            "chat": cmd_chat,
            "serve": cmd_serve,
            "ui": cmd_ui,
            "mcp": cmd_mcp,
            "sync": cmd_sync,
            "backup": cmd_backup,
            "cleanup": cmd_cleanup,
            "dedup": cmd_dedup,
            "metrics": cmd_metrics,
            "quality": cmd_quality,
        }
        routes[args.command](args)


if __name__ == "__main__":
    main()
