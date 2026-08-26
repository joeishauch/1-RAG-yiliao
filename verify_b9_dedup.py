# -*- coding: utf-8 -*-
"""B.9 deterministic cross-source dedup verification (no Chroma/LLM)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from dedup import (
    REASON_LOWER_SCORE,
    REASON_TIE_FIRST_SEEN,
    append_dropped_jsonl,
    dedup_and_write,
    dedup_records,
    normalize_document,
    write_report_atomic,
)


def main() -> None:
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    # Normalization is compatibility-aware, case-insensitive and whitespace-stable.
    check(normalize_document("  ＨＥＬＬＯ，\tWorld！  ") == "hello, world!",
          "normalization did not fold compatibility/case/whitespace")
    check(normalize_document("a,b") != normalize_document("ab"),
          "punctuation must remain significant")

    rows = [
        {"document": "症状 甲", "metadata": {"source": "a", "score": 1, "doc_id": "da"},
         "source": "a", "score": 1, "collection": "medical_qa", "id": "a1"},
        {"document": "症状\t甲", "metadata": {"source": "b", "score": 3, "doc_id": "db"},
         "source": "b", "score": 3, "collection": "medical_qa", "id": "b1"},
        {"document": "症状 甲", "metadata": {"source": "c", "score": 9, "doc_id": "dc"},
         "source": "c", "score": 9, "collection": "medical_triage", "id": "c1"},
        {"document": "", "source": "a", "id": "empty1"},
        {"document": "   ", "source": "b", "id": "empty2"},
    ]
    result = dedup_records(rows)
    check([row["id"] for row in result.kept] == ["b1", "c1", "empty1", "empty2"],
          "winner/order/collection scope incorrect")
    check(result.dropped_count == 1, "expected one duplicate drop")
    check(result.dropped[0]["duplicate_of"]["id"] == "b1", "duplicate winner not traceable")
    check(result.dropped[0]["reason"] == REASON_LOWER_SCORE, "score reason incorrect")

    ties = dedup_records([
        {"document": "tie", "source": "first", "score": 2, "id": "first"},
        {"document": "tie", "source": "second", "score": 2, "id": "second"},
    ])
    check(ties.kept[0]["id"] == "first", "equal scores must keep first")
    check(ties.dropped[0]["reason"] == REASON_TIE_FIRST_SEEN, "tie reason incorrect")
    invalid = dedup_records([
        {"document": "invalid", "source": "x", "score": "not-a-number", "id": "x"},
        {"document": "invalid", "source": "y", "score": 0, "id": "y"},
    ])
    check(invalid.kept[0]["id"] == "x", "invalid score must behave as zero")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report_path = tmp_path / "nested" / "dedup_report.json"
        dropped_path = tmp_path / "nested" / "dedup_dropped.jsonl"
        write_report_atomic({"old": False}, report_path)
        atomic_result = dedup_and_write(rows[:3], report_path=str(report_path),
                                        dropped_path=str(dropped_path))
        check(json.loads(report_path.read_text(encoding="utf-8"))["stats"]["dropped_total"] == 1,
              "atomic report contents incorrect")
        append_dropped_jsonl([{"extra": True}], dropped_path)
        lines = [json.loads(line) for line in dropped_path.read_text(encoding="utf-8").splitlines()]
        check(len(lines) == atomic_result.dropped_count + 1, "JSONL append overwrote existing rows")
        check(not list(report_path.parent.glob(report_path.name + ".tmp.*")),
              "atomic temporary file remains")

    # Similarity hook is opt-in; exact-only path must never call it.
    calls = []
    def similarity(left, right):
        calls.append((left, right))
        return 1.0
    exact = dedup_records([
        {"document": "one", "source": "a"}, {"document": "two", "source": "b"}
    ])
    check(len(calls) == 0 and exact.kept_count == 2, "similarity hook ran by default")
    fuzzy = dedup_records([
        {"document": "one", "source": "a"}, {"document": "two", "source": "b"}
    ], similarity_fn=similarity, similarity_threshold=0.9)
    check(len(calls) > 0 and fuzzy.dropped_count == 1, "enabled similarity hook did not run")

    if failures:
        print("[FAIL] B.9 dedup verification:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("B9_DEDUP_TEST_OK")


if __name__ == "__main__":
    main()
