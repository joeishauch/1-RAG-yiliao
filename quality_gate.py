# -*- coding: utf-8 -*-
"""B.8 数据质量门禁：确定性规则、拒绝记录和扫描报告。

设计目标：
- 在解析/切片/灌库之前拦截明显脏数据：结构缺失、控制字符、超长、空/无效标签、模板题等。
- 复用 ``utils.label_cleaner`` 的纯函数规则（不依赖 LLM），保证同步热路径不阻塞、不调用外部模型。
- 仅对带分诊标签的 source 启用 label 规则，其它 source 只执行通用规则。
- 拒绝记录以 JSONL 写入 ``output/rejected_chunks.jsonl``，含 source/规则版本/原因/截断样本；
  摘要统计写入 ``output/quality_report.json``，便于离线审计与回归。
- 通过 ``run_label_audit.py``（LLM 抽样审核）作为离线人工/模型闭环，热路径不依赖它。

不修改任何历史库；同步/灌库失败时由调用方决定是否阻断或重试。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from utils.config import Config
from utils.label_cleaner import (
    INVALID_LABELS,
    correct_label,
    is_template_question,
)

logger = logging.getLogger(__name__)

RULE_VERSION = "b8-1.0"

# 仅对带分诊 label 的 source 启用 label/template 规则
TRIAGE_SOURCES = {"huatuo_lite"}

# 通用规则阈值
DEFAULT_MIN_DOC_LEN = 2           # 字符数；过滤纯空白/单字符
DEFAULT_MAX_DOC_LEN = 5000        # 同步灌库路径的最大接受长度；超过可由切片器处理
SAMPLE_PREVIEW_CHARS = 120        # rejected 记录里的样本截断长度

# 拒绝 reason 分类（用于聚合报告和 metrics）
REASON_INVALID_STRUCTURE = "invalid_structure"
REASON_EMPTY_DOCUMENT = "empty_document"
REASON_TOO_SHORT = "document_too_short"
REASON_TOO_LONG = "document_too_long"
REASON_BAD_METADATA = "bad_metadata"
REASON_INVALID_LABEL = "invalid_label"
REASON_EMPTY_LABEL = "empty_label"
REASON_TEMPLATE_QUESTION = "template_question"
REASON_PARSER_ERROR = "parser_error"

# ChromaDB metadata 允许的类型
_CHROMA_VALUE_TYPES = (str, int, float, bool)


@dataclass
class RejectionRecord:
    """单条拒绝记录。"""

    ts: str
    source: str
    index: int
    doc_id: Optional[str]
    reason: str
    rule_version: str = RULE_VERSION
    sample: Optional[str] = None
    details: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


@dataclass
class GateStats:
    """单次门禁的统计。"""

    accepted: int = 0
    rejected: int = 0
    rejected_by_reason: Counter = field(default_factory=Counter)
    parser_errors: int = 0

    @property
    def total(self) -> int:
        return self.accepted + self.rejected

    @property
    def rejection_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.rejected / self.total

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejected_by_reason": dict(self.rejected_by_reason),
            "parser_errors": self.parser_errors,
            "rejection_rate": round(self.rejection_rate, 4),
        }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _is_chroma_safe(metadata: dict) -> bool:
    """ChromaDB metadata 只接受 str/int/float/bool；其它值需被剔除。"""
    for value in metadata.values():
        if value is None:
            continue
        if isinstance(value, _CHROMA_VALUE_TYPES):
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                if not isinstance(item, _CHROMA_VALUE_TYPES):
                    return False
            continue
        return False
    return True


class QualityGate:
    """对单个 source 执行确定性质量门禁并写出拒绝记录/报告。"""

    def __init__(
        self,
        source_name: str,
        *,
        rejected_path: str | None = None,
        report_path: str | None = None,
        min_doc_len: int = DEFAULT_MIN_DOC_LEN,
        max_doc_len: int = DEFAULT_MAX_DOC_LEN,
        apply_label_rules: bool | None = None,
    ) -> None:
        self.source_name = source_name
        self.min_doc_len = min_doc_len
        self.max_doc_len = max_doc_len
        self.apply_label_rules = (
            apply_label_rules
            if apply_label_rules is not None
            else source_name in TRIAGE_SOURCES
        )
        default_rejected = Path(Config.QUALITY_REJECTED_PATH)
        default_report = Path(Config.QUALITY_REPORT_PATH)
        self.rejected_path = Path(rejected_path or default_rejected)
        self.report_path = Path(report_path or default_report)
        self.rejected_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self.stats = GateStats()
        self._rejections: list[RejectionRecord] = []
        self._start_ts = time.time()
        self._report_written = False

    # ---------------- 单条检查 ----------------

    def check_record(
        self,
        parsed,
        *,
        index: int,
        doc_id: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """对单条 parsed 进行检查；返回 (accepted, reason)。

        ``parsed`` 应为 ``{document, metadata}`` 或 ``None``。
        任何由 ``check_record`` 产生的拒绝都会在 ``record_rejection`` 里写文件。
        """
        if parsed is None:
            self.record_rejection(index, doc_id, REASON_PARSER_ERROR, sample="")
            return False, REASON_PARSER_ERROR
        if not isinstance(parsed, dict):
            self.record_rejection(
                index, doc_id, REASON_INVALID_STRUCTURE,
                sample=str(parsed)[:SAMPLE_PREVIEW_CHARS],
            )
            return False, REASON_INVALID_STRUCTURE
        document = parsed.get("document")
        metadata = parsed.get("metadata")
        sample = document if isinstance(document, str) else str(document)

        if not isinstance(document, str) or not document.strip():
            self.record_rejection(index, doc_id, REASON_EMPTY_DOCUMENT, sample="")
            return False, REASON_EMPTY_DOCUMENT
        if len(document.strip()) < self.min_doc_len:
            self.record_rejection(
                index, doc_id, REASON_TOO_SHORT,
                sample=document[:SAMPLE_PREVIEW_CHARS],
            )
            return False, REASON_TOO_SHORT
        if len(document) > self.max_doc_len:
            self.record_rejection(
                index, doc_id, REASON_TOO_LONG,
                sample=document[:SAMPLE_PREVIEW_CHARS],
                details={"length": len(document)},
            )
            return False, REASON_TOO_LONG

        if not isinstance(metadata, dict):
            self.record_rejection(
                index, doc_id, REASON_BAD_METADATA,
                sample=document[:SAMPLE_PREVIEW_CHARS],
            )
            return False, REASON_BAD_METADATA
        if not _is_chroma_safe(metadata):
            self.record_rejection(
                index, doc_id, REASON_BAD_METADATA,
                sample=document[:SAMPLE_PREVIEW_CHARS],
                details={"meta_keys": list(metadata.keys())[:10]},
            )
            return False, REASON_BAD_METADATA

        if self.apply_label_rules:
            label = metadata.get("label")
            if label is None or (isinstance(label, str) and not label.strip()):
                self.record_rejection(
                    index, doc_id, REASON_EMPTY_LABEL,
                    sample=document[:SAMPLE_PREVIEW_CHARS],
                )
                return False, REASON_EMPTY_LABEL
            if isinstance(label, str) and label in INVALID_LABELS:
                self.record_rejection(
                    index, doc_id, REASON_INVALID_LABEL,
                    sample=document[:SAMPLE_PREVIEW_CHARS],
                    details={"label": label},
                )
                return False, REASON_INVALID_LABEL
            question = document
            if is_template_question(question):
                self.record_rejection(
                    index, doc_id, REASON_TEMPLATE_QUESTION,
                    sample=question[:SAMPLE_PREVIEW_CHARS],
                )
                return False, REASON_TEMPLATE_QUESTION

        return True, None

    # ---------------- 拒绝记录 ----------------

    def record_rejection(
        self,
        index: int,
        doc_id: Optional[str],
        reason: str,
        sample: str = "",
        details: Optional[dict] = None,
    ) -> None:
        record = RejectionRecord(
            ts=_now_iso(),
            source=self.source_name,
            index=index,
            doc_id=doc_id,
            reason=reason,
            sample=sample[:SAMPLE_PREVIEW_CHARS],
            details=details or {},
        )
        self._rejections.append(record)
        self.stats.rejected_by_reason[reason] += 1
        if reason == REASON_PARSER_ERROR:
            self.stats.parser_errors += 1
        with self._write_lock:
            try:
                with self.rejected_path.open("a", encoding="utf-8") as f:
                    f.write(record.to_json() + "\n")
            except OSError as e:
                logger.warning(f"写入 rejected_chunks.jsonl 失败: {e}")

    # ---------------- 批量 ----------------

    def check_batch(
        self,
        records: Iterable,
        *,
        doc_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list:
        """对一批 parsed 记录执行门禁；返回 accepted 列表。"""
        accepted: list = []
        for index, parsed in enumerate(records):
            ok, _reason = self.check_record(parsed, index=index, doc_id=doc_id)
            if ok:
                accepted.append(parsed)
                self.stats.accepted += 1
            else:
                self.stats.rejected += 1
            if progress_callback and index % 1000 == 0:
                progress_callback(index, self.stats.total)
        return accepted

    # ---------------- 报告 ----------------

    def write_report(
        self,
        *,
        extra: Optional[dict] = None,
    ) -> str:
        """把本次门禁摘要写到 ``self.report_path``（原子写）。"""
        duration_s = round(time.time() - self._start_ts, 2)
        report = {
            "ts": _now_iso(),
            "source": self.source_name,
            "rule_version": RULE_VERSION,
            "duration_s": duration_s,
            "stats": self.stats.as_dict(),
            "rejected_total": len(self._rejections),
            "rejected_by_reason": dict(self.stats.rejected_by_reason),
            "rejected_path": str(self.rejected_path),
            "config": {
                "min_doc_len": self.min_doc_len,
                "max_doc_len": self.max_doc_len,
                "apply_label_rules": self.apply_label_rules,
                "gate_enabled": Config.QUALITY_GATE_ENABLED,
            },
        }
        if extra:
            report["extra"] = extra
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix=self.report_path.name + ".", suffix=".tmp",
                dir=str(self.report_path.parent),
            )
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.report_path)
            self._report_written = True
        except OSError as e:
            logger.warning(f"写入 quality_report.json 失败: {e}")
        return str(self.report_path)

    @property
    def report_written(self) -> bool:
        return self._report_written


def is_within_rate_limit(stats: GateStats, max_rate: float) -> bool:
    """判定拒绝率是否在阈值内；max_rate <= 0 视为无限大。"""
    if max_rate <= 0:
        return True
    return stats.rejection_rate <= max_rate