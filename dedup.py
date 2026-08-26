# -*- coding: utf-8 -*-
"""B.9 跨 source 的确定性文本去重。

本模块只负责可解释的、精确的文本去重；不依赖 Chroma、embedding 或 LLM。
调用方在 parser、质量门禁和切片完成后，把实际准备写入向量库的 chunk 作为
records 传入。相同规范化正文只保留 score 最高的一条，分数相同时保留先入项。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
import unicodedata


RULE_VERSION = "b9-exact-1.0"
REASON_LOWER_SCORE = "lower_score"
REASON_TIE_FIRST_SEEN = "tie_first_seen"
REASON_EXACT_DUPLICATE = "normalized_document_exact_duplicate"
REASON_SIMILAR_DUPLICATE = "similarity_threshold_duplicate"

# NFKC already handles full-width ASCII punctuation. These mappings additionally
# make common Chinese sentence punctuation equivalent without deleting punctuation.
_PUNCTUATION_MAP = str.maketrans({
    "，": ",", "。": ".", "！": "!", "？": "?", "：": ":", "；": ";",
    "（": "(", "）": ")", "【": "[", "】": "]", "“": '"', "”": '"',
    "‘": "'", "’": "'", "、": ",", "－": "-", "—": "-", "–": "-",
    "…": "...", "．": ".",
})
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_document(text: Any) -> str:
    """Return the stable comparison form of a document.

    The operation is deliberately conservative: it does not remove punctuation or
    change word order. It applies Unicode compatibility normalization, case folding,
    common Chinese/ASCII punctuation equivalence, whitespace folding and trimming.
    Non-string values normalize to the empty string and are never used as a key.
    """
    if not isinstance(text, str):
        return ""
    value = unicodedata.normalize("NFKC", text).casefold()
    value = value.translate(_PUNCTUATION_MAP)
    return _WHITESPACE_RE.sub(" ", value).strip()


def normalized_hash(text: Any) -> str:
    """Return a content fingerprint without exposing medical text in reports."""
    normalized = normalize_document(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _score(record: dict) -> float:
    value = record.get("score")
    if value is None:
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get("score", 0)
    try:
        value = float(value if value is not None else 0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _source(record: dict) -> str:
    value = record.get("source")
    if value is None and isinstance(record.get("metadata"), dict):
        value = record["metadata"].get("source")
    return str(value or "")


def _identity(record: dict, index: int) -> dict:
    value = record.get("id")
    if value is None:
        value = record.get("chunk_id")
    if value is None and isinstance(record.get("metadata"), dict):
        value = record["metadata"].get("id")
    return {"index": index, "id": value, "source": _source(record)}


def _record_with_meta(record: dict, index: int) -> dict:
    """Make a shallow copy without changing the caller's record shape."""
    # Provenance and normalized score live in the internal entry; adding them to
    # the returned record would unexpectedly alter Chroma-ready metadata.
    return dict(record)


@dataclass
class DedupResult:
    """Result of one deterministic deduplication pass."""

    kept: list[dict]
    dropped: list[dict]
    stats: dict
    groups: list[dict]
    collection: Optional[str] = None

    def __getitem__(self, key: str):
        return getattr(self, key)

    def as_dict(self) -> dict:
        return {
            "kept": self.kept,
            "dropped": self.dropped,
            "stats": self.stats,
            "groups": self.groups,
            "collection": self.collection,
        }

    @property
    def input_count(self) -> int:
        return int(self.stats.get("input_total", 0))

    @property
    def kept_count(self) -> int:
        return len(self.kept)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)


def _similarity_match(
    normalized: str,
    groups: list[dict],
    similarity_fn: Optional[Callable[[str, str], float]],
    threshold: Optional[float],
    collection: Optional[str],
) -> Optional[dict]:
    if similarity_fn is None or threshold is None:
        return None
    for group in groups:
        if group.get("collection") != collection:
            continue
        try:
            similarity = float(similarity_fn(group["normalized"], normalized))
        except (TypeError, ValueError, ArithmeticError):
            continue
        if math.isfinite(similarity) and similarity >= threshold:
            return group
    return None


def dedup_records(
    records: Iterable[dict],
    *,
    collection: Optional[str] = None,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    similarity_threshold: Optional[float] = None,
) -> DedupResult:
    """Deduplicate ordered chunk records without mutating the input.

    ``records`` may contain ``document``, ``metadata``, ``source``, ``score`` and
    ``id``. Score is read from the top level first and then metadata, with malformed
    and missing values treated as zero. Empty documents remain in ``kept`` but are
    never compared with one another. Winner selection is independent of input order;
    the returned kept list is restored to input order for stable source batching.

    ``similarity_fn`` is an explicit second-phase extension hook. It is never called
    unless ``similarity_threshold`` is supplied, so the default is exact-only.
    """
    original = list(records)
    normalized_records: list[dict] = []
    groups_by_key: dict[tuple[Optional[str], str], dict] = {}
    groups: list[dict] = []
    dropped: list[dict] = []
    winners: dict[int, dict] = {}
    empty_count = 0

    for index, raw in enumerate(original):
        record = _record_with_meta(raw if isinstance(raw, dict) else {"document": raw}, index)
        document = record.get("document")
        normalized = normalize_document(document)
        entry_collection = record.get("collection", collection)
        entry = {
            "record": record,
            "index": index,
            "normalized": normalized,
            "collection": entry_collection,
            "score": _score(record),
            "source": _source(record),
            "id": _identity(record, index).get("id"),
            "hash": normalized_hash(document),
        }
        normalized_records.append(entry)
        if not normalized:
            empty_count += 1
            winners[index] = record
            continue

        group = groups_by_key.get((entry_collection, normalized))
        reason = REASON_EXACT_DUPLICATE
        if group is None:
            group = _similarity_match(
                normalized, groups, similarity_fn, similarity_threshold, entry_collection
            )
            if group is not None:
                reason = REASON_SIMILAR_DUPLICATE
        if group is None:
            group = {
                "collection": entry_collection,
                "normalized": normalized,
                "hash": entry["hash"],
                "winner": entry,
                "members": [entry],
            }
            groups.append(group)
            groups_by_key[(entry_collection, normalized)] = group
            winners[index] = record
            continue

        group["members"].append(entry)
        winner = group["winner"]
        if entry["score"] > winner["score"]:
            # The old winner becomes dropped; the new winner may appear later in
            # input order, so final output is assembled only after this pass.
            if winner["index"] in winners:
                del winners[winner["index"]]
            group["winner"] = entry
            winners[index] = record
            drop_reason = REASON_LOWER_SCORE
        else:
            drop_reason = REASON_TIE_FIRST_SEEN if entry["score"] == winner["score"] else REASON_LOWER_SCORE

        dropped.append({
            "source": entry["source"],
            "score": entry["score"],
            "winner_score": group["winner"]["score"],
            "record_index": entry["index"],
            "id": entry["id"],
            "normalized_hash": entry["hash"],
            "duplicate_of": _identity(group["winner"]["record"], group["winner"]["index"]),
            "reason": reason if drop_reason == REASON_TIE_FIRST_SEEN else drop_reason,
        })

        # If the new entry superseded the old winner, the old entry must be
        # represented as dropped and point to the new winner. Rewriting its row
        # after the winner is known keeps duplicate_of accurate.
        if drop_reason == REASON_LOWER_SCORE and entry["score"] > winner["score"]:
            dropped.append({
                "source": winner["source"],
                "score": winner["score"],
                "winner_score": entry["score"],
                "record_index": winner["index"],
                "id": winner["id"],
                "normalized_hash": winner["hash"],
                "duplicate_of": _identity(entry["record"], entry["index"]),
                "reason": REASON_LOWER_SCORE,
            })

    # A winner can be replaced more than once. Recompute drops from each group
    # so every dropped record points to the final winner exactly once.
    dropped = []
    for group in groups:
        winner = group["winner"]
        for entry in group["members"]:
            if entry["index"] == winner["index"]:
                continue
            dropped.append({
                "source": entry["source"],
                "score": entry["score"],
                "winner_score": winner["score"],
                "record_index": entry["index"],
                "id": entry["id"],
                "normalized_hash": entry["hash"],
                "duplicate_of": _identity(winner["record"], winner["index"]),
                "reason": (
                    REASON_TIE_FIRST_SEEN
                    if entry["score"] == winner["score"]
                    else REASON_LOWER_SCORE
                ),
            })

    kept = [winners[index] for index in sorted(winners)]
    dropped.sort(key=lambda item: item["record_index"])
    by_source = Counter(_source(record) for record in original)
    kept_by_source = Counter(_source(record) for record in kept)
    dropped_by_source = Counter(item["source"] for item in dropped)
    reason_counts = Counter(item["reason"] for item in dropped)
    group_rows = []
    for group in groups:
        winner = group["winner"]
        group_rows.append({
            "collection": group.get("collection"),
            "normalized_hash": group["hash"],
            "member_count": len(group["members"]),
            "winner": _identity(winner["record"], winner["index"]),
            "winner_score": winner["score"],
        })

    stats = {
        "input_total": len(original),
        "kept_total": len(kept),
        "dropped_total": len(dropped),
        "empty_text_kept": empty_count,
        "duplicate_groups": sum(1 for g in groups if len(g["members"]) > 1),
        "by_source": dict(by_source),
        "kept_by_source": dict(kept_by_source),
        "dropped_by_source": dict(dropped_by_source),
        "dropped_by_reason": dict(reason_counts),
        "similarity_enabled": similarity_fn is not None and similarity_threshold is not None,
        "similarity_threshold": similarity_threshold,
        "rule_version": RULE_VERSION,
    }
    return DedupResult(kept, dropped, stats, group_rows, collection)


# Friendly aliases for callers/tests using alternate naming.
deduplicate_records = dedup_records
normalize_text = normalize_document


def build_report(result: DedupResult, *, report_path: Optional[str] = None, dropped_path: Optional[str] = None, context: Optional[dict] = None) -> dict:
    """Build a JSON-serializable report from a result."""
    report = {
        "rule_version": RULE_VERSION,
        "collection": result.collection,
        "stats": result.stats,
        "groups": result.groups,
        "report_path": str(report_path) if report_path is not None else None,
        "dropped_path": str(dropped_path) if dropped_path is not None else None,
    }
    if context:
        report["context"] = dict(context)
    return report


def write_report_atomic(report: dict, path: str | os.PathLike[str]) -> str:
    """Atomically replace a JSON report and remove stale temporary files."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    for stale in target.parent.glob(target.name + ".tmp.*"):
        try:
            stale.unlink()
        except OSError:
            pass
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".tmp.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return str(target)


def append_dropped_jsonl(dropped: Iterable[dict], path: str | os.PathLike[str]) -> int:
    """Append dropped records as one UTF-8 JSON object per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(dropped)
    if not rows:
        return 0
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def dedup_and_write(
    records: Iterable[dict],
    *,
    report_path: Optional[str] = None,
    dropped_path: Optional[str] = None,
    collection: Optional[str] = None,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    similarity_threshold: Optional[float] = None,
    context: Optional[dict] = None,
) -> DedupResult:
    """Run dedup and optionally persist its report/audit detail."""
    result = dedup_records(
        records,
        collection=collection,
        similarity_fn=similarity_fn,
        similarity_threshold=similarity_threshold,
    )
    report = build_report(
        result,
        report_path=report_path,
        dropped_path=dropped_path,
        context=context,
    )
    if report_path:
        write_report_atomic(report, report_path)
    if dropped_path:
        append_dropped_jsonl(result.dropped, dropped_path)
    return result


def is_dedup_active(enabled: bool, sources: Iterable[str]) -> bool:
    """Return whether cross-source dedup should run for the selected sources."""
    return bool(enabled) and len(set(str(source) for source in sources)) >= 2
