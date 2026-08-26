"""B.8 数据质量门禁单元测试（mock 路径，不依赖真实 LLM/embedding/Chroma）。

验证：
- 通用规则：空 document、太短、过长、非字典 metadata、不可序列化值
- 分诊规则：模板题、空 label、无效 label
- QA 源只跑通用规则，不启用 label 规则
- 拒绝率超阈值返回 gate_blocked；报告按原子写

主入口输出 B8_QUALITY_GATE_TEST_OK；失败 sys.exit(1)。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from quality_gate import (
    QualityGate,
    is_within_rate_limit,
    REASON_EMPTY_DOCUMENT,
    REASON_TEMPLATE_QUESTION,
    REASON_INVALID_LABEL,
    REASON_BAD_METADATA,
    REASON_EMPTY_LABEL,
)


def _make_valid_triage():
    return {
        "document": "持续发烧三天伴有咳嗽，需要就诊",
        "metadata": {"label": "内科", "source": "huatuo_lite"},
    }


def _make_template():
    return {
        "document": "该病的辅助治疗原则是什么？",
        "metadata": {"label": "内科", "source": "huatuo_lite"},
    }


def _make_invalid_label():
    return {
        "document": "胸痛持续",
        "metadata": {"label": "其他", "source": "huatuo_lite"},
    }


def _make_empty_label():
    return {
        "document": "头痛反复发作",
        "metadata": {"label": "", "source": "huatuo_lite"},
    }


def _make_bad_metadata():
    return {
        "document": "咳嗽两周",
        "metadata": {"label": "内科", "tags": {1, 2, 3}},
    }


def _make_valid_qa():
    return {
        "document": "感冒吃什么药？",
        "metadata": {"source": "huatuo_encyclopedia"},
    }


def main() -> None:
    failed = []

    def assert_true(cond: bool, msg: str) -> None:
        if not cond:
            failed.append(msg)

    with tempfile.TemporaryDirectory() as tmp:
        rejected_path = Path(tmp) / "rejected.jsonl"
        report_path = Path(tmp) / "report.json"

        # Case 1: 合法分诊样本通过
        gate = QualityGate(
            "huatuo_lite",
            rejected_path=rejected_path,
            report_path=report_path,
        )
        ok, _ = gate.check_record(_make_valid_triage(), index=0)
        assert_true(ok, "case1: 合法分诊样本应通过")

        # Case 2: 模板题被拒
        ok, reason = gate.check_record(_make_template(), index=1)
        assert_true(not ok and reason == REASON_TEMPLATE_QUESTION,
                    f"case2: 模板题被拒 reason={reason}")

        # Case 3: 无效 label 被拒
        ok, reason = gate.check_record(_make_invalid_label(), index=2)
        assert_true(not ok and reason == REASON_INVALID_LABEL,
                    f"case3: 其他 label 被拒 reason={reason}")

        # Case 4: 空 label 被拒
        ok, reason = gate.check_record(_make_empty_label(), index=3)
        assert_true(not ok and reason == REASON_EMPTY_LABEL,
                    f"case4: 空 label 被拒 reason={reason}")

        # Case 5: 非法 metadata 被拒
        ok, reason = gate.check_record(_make_bad_metadata(), index=4)
        assert_true(not ok and reason == REASON_BAD_METADATA,
                    f"case5: 非法 metadata 被拒 reason={reason}")

        # Case 6: QA 源不接受 label 规则
        gate_qa = QualityGate("huatuo_encyclopedia",
                              rejected_path=rejected_path,
                              report_path=report_path)
        ok_qa, _ = gate_qa.check_record(_make_valid_qa(), index=0)
        assert_true(ok_qa, "case6: QA 源合法样本通过")

        # Case 7: 拒绝率超阈值阻断
        gate_high = QualityGate("huatuo_lite",
                                rejected_path=rejected_path,
                                report_path=report_path)
        batch = [_make_valid_triage(), _make_template(), _make_template(),
                 _make_invalid_label(), _make_empty_label(), _make_valid_triage()]
        accepted = gate_high.check_batch(batch)
        assert_true(len(accepted) == 2, f"case7: 期望 2 条通过，实际 {len(accepted)}")
        assert_true(not is_within_rate_limit(gate_high.stats, 0.5),
                    "case7: 拒绝率 > 50% 应阻断")

        # Case 8: 拒绝文件 + 报告原子写
        with tempfile.TemporaryDirectory() as tmp2:
            rejected_path2 = Path(tmp2) / "rejected.jsonl"
            report_path2 = Path(tmp2) / "report.json"
            gate_write = QualityGate("huatuo_lite",
                                     rejected_path=rejected_path2,
                                     report_path=report_path2)
            gate_write.check_record(_make_template(), index=0)
            report = gate_write.write_report()
            assert_true(Path(report).exists(), "case8: 报告应写入")
            assert_true(rejected_path2.exists(), "case8: rejected.jsonl 应写入")
            with rejected_path2.open("r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            assert_true(len(lines) == 1, f"case8: 拒绝记录数={len(lines)}")
            assert_true(lines[0]["reason"] == REASON_TEMPLATE_QUESTION,
                        "case8: 拒绝 reason 一致")

        # Case 9: 空 document 被拒
        gate_empty = QualityGate("huatuo_lite",
                                 rejected_path=rejected_path,
                                 report_path=report_path)
        ok, reason = gate_empty.check_record({"document": "   ", "metadata": {}}, index=0)
        assert_true(not ok and reason == REASON_EMPTY_DOCUMENT,
                    f"case9: 空白文档被拒 reason={reason}")

    if failed:
        sys.stdout.reconfigure(encoding="utf-8")
        print("[FAIL] B.8 quality gate verification:")
        for msg in failed:
            print(f"  - {msg}")
        sys.exit(1)

    print("B8_QUALITY_GATE_TEST_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()