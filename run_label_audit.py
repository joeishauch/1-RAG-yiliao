# -*- coding: utf-8 -*-
"""标签清洗审核 CLI：验证 label_cleaner 规则质量，输出审核报告。

流程：读全量 format_data.jsonl → 应用清洗规则（is_template_question / correct_label）
      → 收集两类候选 → 用 qwen-max 独立裁判审核 → 输出报告到 eval_reports/。

两类候选：
  1. 纠正候选：correct_label 导致 label 变化（「头疼+失眠」心理科学→神经科学），全量审；
  2. 过滤候选：is_template_question 判为模板题（应过滤），抽样审「是否误伤真主诉」。

审核不阻塞重灌：先跑本脚本看通过率，通过率低就调规则再重灌。

用法：
    python run_label_audit.py                    # 全量扫描 + 审核（纠正全量、过滤抽样）
    python run_label_audit.py --dry              # 只统计候选分布，不调 LLM（快速、零 token）
    python run_label_audit.py --scan-limit 20000 # 只扫前 N 条（调试）
    python run_label_audit.py --filter-n 150     # 过滤候选抽样审核条数
"""
import os
import sys
import json
import random
import argparse
from collections import Counter

# 处理 Windows 中文 stdout 乱码
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# 切换到脚本目录，保证相对路径正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("eval_reports", exist_ok=True)

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from utils.config import Config
from utils.llms import get_llm
from utils.label_cleaner import is_template_question, correct_label
from utils.label_audit import LLMAuditor

DATA_PATH = "input/Huatuo26M-Lite/format_data.jsonl"
REPORT_PATH = "eval_reports/label_audit_report.json"


class FilterVerdict(BaseModel):
    """过滤候选审核结论。"""
    keep: bool = Field(description="是否应保留为分诊主诉（true=规则误伤真主诉，false=过滤正确）")
    reason: str = Field(description="一句话理由")


FILTER_AUDIT_TEMPLATE = """你是医疗分诊数据质量审核专家。下面这条患者提问被清洗规则判为「模板题（非分诊主诉）」，将被从分诊库过滤掉。

=== 患者提问 ===
{question}

请判断这条提问是否确实是在「问疾病知识点（模板题）」、而非「自述症状该挂哪个科」的分诊主诉：
- keep=false：确实是模板题/知识题，过滤正确（不该进分诊库）
- keep=true：这其实是患者症状主诉，过滤属于误伤（应保留）
只输出结构化结果。"""


def _read_all():
    """逐行读 jsonl，yield (question, label) 元组"""
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield rec.get("question", ""), rec.get("label", "")


def _build_filter_chain(llm_type):
    llm_chat, _ = get_llm(llm_type)
    prompt = ChatPromptTemplate.from_messages([("human", FILTER_AUDIT_TEMPLATE)])
    return prompt | llm_chat.with_structured_output(FilterVerdict, method="function_calling")


def main():
    parser = argparse.ArgumentParser(description="审核 label_cleaner 清洗规则质量")
    parser.add_argument("--dry", action="store_true", help="只统计候选分布，不调用 LLM")
    parser.add_argument("--scan-limit", type=int, default=None, help="只扫前 N 条（调试）")
    parser.add_argument("--filter-n", type=int, default=150, help="过滤候选抽样审核条数")
    parser.add_argument("--correct-max", type=int, default=500, help="纠正候选审核上限（超过则抽样）")
    args = parser.parse_args()

    # ---------- 1. 扫描全量，应用清洗规则 ----------
    scanned = 0
    template_filtered = 0
    template_labels = Counter()
    corrected = []           # 纠正候选：{question, original_label, corrected_label}
    filter_samples = []      # 过滤候选：question 列表

    for question, label in _read_all():
        scanned += 1
        if args.scan_limit and scanned > args.scan_limit:
            break
        if is_template_question(question):
            template_filtered += 1
            template_labels[label] += 1
            filter_samples.append(question)
            continue  # 模板题优先过滤，不再参与纠正（避免重复计）
        new_label = correct_label(question, label)
        if new_label != label:
            corrected.append({
                "question": question,
                "original_label": label,
                "corrected_label": new_label,
            })

    print(f"扫描 {scanned} 条")
    print(f"  模板题过滤：{template_filtered} 条（原科室分布：{dict(template_labels)}）")
    print(f"  纠正候选（label 变化）：{len(corrected)} 条")

    report = {
        "scanned": scanned,
        "template_filtered": template_filtered,
        "template_by_label": dict(template_labels),
        "corrected_candidates": len(corrected),
        "correction_audit": None,
        "filter_audit": None,
    }

    if args.dry:
        print("（--dry，跳过 LLM 审核）")
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"候选统计已写入 {REPORT_PATH}")
        return

    # ---------- 2. 审核纠正候选（qwen-max 独立裁判）----------
    if corrected:
        to_audit = corrected[:args.correct_max]
        if len(corrected) > args.correct_max:
            to_audit = random.sample(corrected, args.correct_max)
        auditor = LLMAuditor("qwen")
        verdicts = auditor.audit(to_audit)

        approved = rejected = err = 0
        rejected_samples = []
        for s, v in zip(to_audit, verdicts):
            if "审核异常" in v.reason:
                err += 1
            elif v.approved:
                approved += 1
            else:
                rejected += 1
                rejected_samples.append({
                    "question": s["question"],
                    "original_label": s["original_label"],
                    "corrected_label": s["corrected_label"],
                    "audit_label": v.authoritative_label,
                    "reason": v.reason,
                })
        report["correction_audit"] = {
            "audited": len(to_audit),
            "approved": approved,
            "rejected": rejected,
            "error": err,
            "approve_rate": round(approved / len(to_audit), 3) if to_audit else None,
            "rejected_samples": rejected_samples,
        }
        print(f"  纠正候选审核：审 {len(to_audit)} 条，通过 {approved}、驳回 {rejected}、异常 {err}")
        for s in rejected_samples[:10]:
            print(f"    [驳回] {s['original_label']}→{s['corrected_label']}(审{s['audit_label']}): {s['question'][:40]}")

    # ---------- 3. 抽样审核过滤候选（是否误伤真主诉）----------
    if filter_samples:
        n = min(args.filter_n, len(filter_samples))
        sampled = random.sample(filter_samples, n)
        chain = _build_filter_chain("qwen")
        keep_cnt = 0
        misclassified = []
        for q in sampled:
            try:
                v = chain.invoke({"question": q})
            except Exception as e:
                v = FilterVerdict(keep=False, reason=f"审核异常: {e}")
            if v.keep:
                keep_cnt += 1
                misclassified.append({"question": q, "reason": v.reason})
        report["filter_audit"] = {
            "sampled": n,
            "misclassified": keep_cnt,
            "misclassified_samples": misclassified,
        }
        print(f"  过滤候选审核：抽 {n} 条，误伤 {keep_cnt} 条")
        for s in misclassified[:10]:
            print(f"    [误伤] {s['question'][:60]}: {s['reason'][:40]}")

    # ---------- 4. 落盘报告 ----------
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n审核报告已写入 {REPORT_PATH}")


if __name__ == "__main__":
    main()
