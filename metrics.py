# -*- coding: utf-8 -*-
"""B.7 可观测性：轻量指标采集。

不引入 Prometheus / OpenTelemetry 等重型依赖，采用：
- 进程内线程安全计数器（Counter / Gauge）
- 定时或显式 flush 到 output/metrics.jsonl（追加写，不阻塞主流程）
- CLI 子命令 cli.py metrics [--last N] [--summary] 读取汇总

主要指标：
- embedding_request_count / embedding_latency_ms / embedding_cost_usd
- sync_duration_seconds{source} / sync_inserted_total{source} / sync_failed_total{source}
- collection_size_rows{collection}
- 审计事件计数（按事件类型统计）
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 输出目录：项目根/output/metrics.jsonl
_METRICS_FILE = Path("output") / "metrics.jsonl"
_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
# 进程内计数器（重启清零，仅作为本次运行的实时指标）
_counters: dict[str, float] = defaultdict(float)
# 时间序列观测点（每次调用 record() 追加一条记录）
# 在锁内追加，写盘由后台线程或显式 flush

# Embedding 成本（每千 token 美元，简化近似）
# dashscope text-embedding-v1: 0.0007 / 1k tokens
# local_bge: 0
_EMBEDDING_COST_PER_1K_TOKENS = {
    "dashscope-text-embedding-v1": 0.0007,
    "zhipu-embedding-3": 0.001,
    "bge-m3-local": 0.0,
    "bge-m3": 0.0,
    "unknown": 0.0,
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数：英文 ~4 字符/token，中文 ~1.5-2 字符/token。

    此处用 2 字符/token 的保守估计，避免低估成本。
    """
    return max(1, len(text) // 2)


def estimate_cost(token_count: int, model_id: str) -> float:
    """估算 embedding 成本（美元）。"""
    rate = _EMBEDDING_COST_PER_1K_TOKENS.get(model_id, 0.0)
    return round(token_count / 1000 * rate, 6)


# ---------------- 计数器 ----------------

def incr(name: str, value: float = 1.0) -> None:
    """递增一个计数器（例如 embedding_request_count、sync_failed_total）。"""
    with _lock:
        _counters[name] += value


def gauge(name: str, value: float, labels: Optional[dict] = None) -> None:
    """设置一个瞬时值（例如 collection_size_rows）。"""
    key = _format_label(name, labels)
    with _lock:
        _counters[key] = value


def _format_label(name: str, labels: Optional[dict]) -> str:
    """拼接 metric_name + label 维度，例如 sync_duration_seconds{source=huatuo_lite}"""
    if not labels:
        return name
    parts = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{parts}}}"


def parse_label(name: str) -> tuple[str, dict]:
    """_format_label 的逆操作，返回 (metric_name, labels_dict)。"""
    if "{" not in name:
        return name, {}
    base, _, label_str = name.partition("{")
    label_str = label_str.rstrip("}")
    labels = {}
    for kv in label_str.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            labels[k.strip()] = v.strip()
    return base, labels


# ---------------- 高层封装：embedding ----------------

def time_embedding(
    model_id: str,
    texts: list[str],
    context: Optional[dict] = None,
):
    """记录一次 ``embed_documents`` 包装器调用。

    ``embedding_request_count`` 的口径是 *wrapper batch 次数*，不保证等于
    provider 底层 HTTP 请求数（provider 可能再次拆批）。``context`` 记录
    source、collection、run_id 等关联信息，但不进入指标名，避免 label 爆炸。
    """
    return _EmbeddingTimer(model_id, texts, context=context)


class _EmbeddingTimer:
    """embedding 调用的耗时/成本计时器。"""

    def __init__(self, model_id: str, texts: list[str], context: Optional[dict] = None):
        self.model_id = model_id
        self.texts = texts
        self.tokens = sum(estimate_tokens(t) for t in texts)
        self.items = len(texts)
        self.cost = estimate_cost(self.tokens, model_id)
        self.context = dict(context or {})
        self.latency_ms = 0.0
        self.success = False

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = (time.perf_counter() - self._t0) * 1000
        self.latency_ms = elapsed
        self.success = exc_type is None
        # 记录到 jsonl（失败也记录）
        metric_context = {**self.context, "items": self.items}
        record_metric(
            "embedding_request_count",
            1,
            labels={"model": self.model_id, "status": "ok" if self.success else "fail"},
            kind="counter",
            context=metric_context,
        )
        if self.success:
            record_metric(
                "embedding_latency_ms",
                elapsed,
                labels={"model": self.model_id},
                kind="histogram",
                context={**self.context, "items": self.items},
            )
            record_metric(
                "embedding_items_total",
                self.items,
                labels={"model": self.model_id},
                kind="counter",
                context=self.context,
            )
            record_metric(
                "embedding_tokens_total",
                self.tokens,
                labels={"model": self.model_id},
                kind="counter",
                context=self.context,
            )
            if self.cost > 0:
                record_metric(
                    "embedding_cost_usd",
                    self.cost,
                    labels={"model": self.model_id},
                    kind="counter",
                    context=self.context,
                )
        return False


# ---------------- 写入 ----------------

def _infer_kind(name: str) -> str:
    """为旧版 JSONL 记录推断指标类型，保持历史文件兼容。"""
    if name.startswith("collection_size_rows"):
        return "gauge"
    if name.startswith("embedding_latency_ms") or name.startswith("sync_duration_seconds"):
        return "histogram"
    return "counter"


def record_metric(
    name: str,
    value: float,
    labels: Optional[dict] = None,
    *,
    kind: Optional[str] = None,
    context: Optional[dict] = None,
) -> None:
    """记录一条指标。

    ``kind`` 只影响汇总语义：gauge 取最新值，其它类型按事件样本累计。
    ``context`` 保存 source/collection/run_id 等关联信息，不参与 metric key，
    避免把每次运行都变成新的 label 时间序列。
    """
    full_name = _format_label(name, labels)
    kind = kind or _infer_kind(full_name)
    record = {
        "ts": _now_iso(),
        "metric": full_name,
        "value": value,
        "kind": kind,
    }
    if context:
        record["context"] = dict(context)
    with _lock:
        if kind == "gauge":
            _counters[full_name] = value
        else:
            _counters[full_name] += value
        try:
            with _METRICS_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning(f"写入 metrics.jsonl 失败: {e}")


def record_sync_done(
    source: str,
    status: str,
    duration_s: float,
    inserted: int = 0,
    deleted: int = 0,
    *,
    collection: Optional[str] = None,
    context: Optional[dict] = None,
    deleted_scope: Optional[str] = None,
    deleted_estimated: bool = False,
    deleted_known: bool = True,
) -> None:
    """记录一次 source 同步事件；duration 单位固定为秒。"""
    event_context = dict(context or {})
    event_context.update({"source": source, "status": status})
    if collection:
        event_context["collection"] = collection
    if deleted_scope:
        event_context["deleted_scope"] = deleted_scope
    if deleted_estimated:
        event_context["deleted_estimated"] = True
    event_context["deleted_known"] = deleted_known
    record_metric(
        "sync_duration_seconds",
        duration_s,
        labels={"source": source, "status": status},
        kind="histogram",
        context=event_context,
    )
    if status in {"ok", "rebuilt"}:
        record_metric(
            "sync_inserted_total", inserted, labels={"source": source},
            kind="counter", context=event_context,
        )
        record_metric(
            "sync_deleted_total", deleted, labels={"source": source},
            kind="counter", context=event_context,
        )
    elif status == "error":
        record_metric(
            "sync_failed_total", 1, labels={"source": source},
            kind="counter", context={**event_context, "inserted": inserted, "deleted": deleted},
        )
        # 失败部分也保留实际已写入/删除量，避免失败事件的增量丢失。
        if inserted:
            record_metric(
                "sync_inserted_total", inserted, labels={"source": source},
                kind="counter", context=event_context,
            )
        if deleted:
            record_metric(
                "sync_deleted_total", deleted, labels={"source": source},
                kind="counter", context=event_context,
            )


def record_collection_size(
    collection: str,
    row_count: int,
    *,
    context: Optional[dict] = None,
) -> None:
    """记录 collection 的当前 chunk/vector 数快照，不是累计计数。"""
    labels = {"collection": collection}
    # 内存 gauge 只更新一次；record_metric(kind=gauge) 负责持久化快照。
    record_metric(
        "collection_size_rows", row_count, labels=labels,
        kind="gauge", context={**(context or {}), "collection": collection},
    )


def record_audit_event(event: str) -> None:
    """记录审计事件计数（无需访问 audit.jsonl，直接 incr）。"""
    incr(f"audit_event_total{{event={event}}}")


def record_dedup_done(
    *,
    kept_by_source: Optional[dict[str, int]] = None,
    dropped_by_source: Optional[dict[str, int]] = None,
    context: Optional[dict] = None,
) -> None:
    """记录一次跨源去重结果；source 仅作为 label，运行信息放 context。"""
    event_context = dict(context or {})
    kept_total = sum(int(value) for value in (kept_by_source or {}).values())
    if kept_total:
        record_metric(
            "dedup_kept_total", kept_total,
            kind="counter",
            context=event_context,
        )
    for source, value in (dropped_by_source or {}).items():
        record_metric(
            "dedup_dropped_total", value,
            labels={"source": source},
            kind="counter",
            context=event_context,
        )
    record_audit_event("dedup")


# ---------------- 读取/汇总 ----------------

def read_metrics(limit: Optional[int] = None) -> list[dict]:
    """读取最近 N 条指标记录。limit=None 读全部。"""
    if not _METRICS_FILE.exists():
        return []
    with _METRICS_FILE.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    records = []
    for line in lines[-limit:] if limit else lines:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize(window_minutes: Optional[int] = None) -> dict:
    """汇总指标；gauge 返回最新快照，其它类型按事件样本统计。"""
    records = read_metrics()
    if window_minutes:
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        filtered = []
        for r in records:
            try:
                ts = datetime.fromisoformat(r["ts"])
            except (KeyError, TypeError, ValueError):
                continue
            if ts >= cutoff:
                filtered.append(r)
        records = filtered

    summary = defaultdict(
        lambda: {"count": 0, "sum": 0.0, "max": 0.0, "min": float("inf"), "kind": None, "latest": None}
    )
    for r in records:
        try:
            name = r["metric"]
            value = float(r["value"])
        except (KeyError, TypeError, ValueError):
            continue
        s = summary[name]
        kind = r.get("kind") or _infer_kind(name)
        s["kind"] = kind
        s["count"] += 1
        s["sum"] += value
        s["max"] = max(s["max"], value)
        s["min"] = min(s["min"], value)
        s["latest"] = value

    out = {}
    for name, s in summary.items():
        is_gauge = s["kind"] == "gauge"
        out[name] = {
            "kind": s["kind"],
            "count": s["count"],
            # gauge 的 sum/avg 仅为兼容旧调用保留；展示应使用 latest。
            "sum": round(s["latest"] if is_gauge else s["sum"], 4),
            "avg": round(s["latest"] if is_gauge else s["sum"] / s["count"], 4),
            "min": round(s["min"], 4) if s["min"] != float("inf") else 0,
            "max": round(s["max"], 4),
            "latest": round(s["latest"], 4) if is_gauge and s["latest"] is not None else None,
        }
    return out


def diagnose(records: Optional[list[dict]] = None) -> list[dict]:
    """诊断历史指标中的明显异常，不修改原始 metrics.jsonl。"""
    records = records if records is not None else read_metrics()
    findings = []
    for r in records:
        metric = r.get("metric", "")
        value = r.get("value")
        try:
            value = float(value)
        except (TypeError, ValueError):
            findings.append({"severity": "error", "reason": "invalid_value", "record": r})
            continue
        if metric.startswith("sync_duration_seconds") and value >= 7200:
            findings.append({
                "severity": "warning",
                "reason": "duration_requires_log_correlation",
                "value": value,
                "metric": metric,
                "message": "耗时达到或超过 7200 秒；请以对应 sync 日志起止时间核对，不能视为默认值。",
            })
        if metric.startswith("collection_size_rows"):
            findings.append({
                "severity": "info",
                "reason": "collection_snapshot",
                "value": value,
                "metric": metric,
                "message": "这是目标 collection 的 chunk/vector 快照，不是本次 source 的 inserted 累计量。",
            })
        if metric.startswith("embedding_request_count"):
            findings.append({
                "severity": "info",
                "reason": "wrapper_batch_semantics",
                "value": value,
                "metric": metric,
                "message": "request 按 embed_documents 包装器 batch 计数，不等于底层 HTTP 请求数。",
            })
        if metric.startswith("dedup_"):
            findings.append({
                "severity": "info",
                "reason": "dedup_candidate_counts",
                "value": value,
                "metric": metric,
                "message": "跨源去重指标按实际 chunk 候选累计；未启用或单 source 不产生该指标。",
            })
    return findings


def reset() -> None:
    """清空指标文件 + 进程内计数器（仅用于测试）。"""
    global _counters
    with _lock:
        _counters = defaultdict(float)
        if _METRICS_FILE.exists():
            _METRICS_FILE.unlink()
