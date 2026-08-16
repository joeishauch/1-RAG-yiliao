# -*- coding: utf-8 -*-
"""验证均衡抽样重灌后的小科室样本量是否回升。

连接 ChromaDB 的 medical_triage collection，按 metadata.label 统计各科室条数，
对比「旧 head 抽样基线（前 2 万条）」与「全量分布」，判断小科室是否被补足。

用法：
    python verify_label_balance.py                          # 查默认 medical_triage
    python verify_label_balance.py --collection medical_triage

基线数据来自数据探索（2026-08-14）：
- 旧 head 抽样（limit=20000 只灌前 2 万条）：急诊科 3、心理科学 345、中医科 281、生殖健康科 154
- 全量 17.7 万条分布见 FULL_DIST

注意：急诊科全量仅 21 条且几乎全是模板题（「X 的辅助治疗/判断依据」），
      清洗过滤后可能接近 0，这是预期行为，不是 bug——真实急症数据需外部补强。
"""
import os
import sys
import argparse
from collections import Counter

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb

CHROMADB_DIRECTORY = "chromaDB"
DEFAULT_COLLECTION = "medical_triage"

# 全量 17.7 万条分布（数据探索测得，仅参考）
FULL_DIST = {
    "妇产科": 34313, "内科": 29677, "皮肤性病科": 24668, "儿科": 21202,
    "眼耳鼻喉科": 13791, "肿瘤科": 10107, "神经科学": 10009, "外科": 9577,
    "男性健康科": 8109, "感染与免疫科": 5028, "口腔科": 3223, "心理科学": 2989,
    "中医科": 2493, "生殖健康科": 1363, "其他": 1133, "急诊科": 21,
}

# 旧 head 抽样（前 2 万条）里被淹没的重点小科室基线
BASELINE_HEAD = {
    "急诊科": 3, "心理科学": 345, "中医科": 281, "生殖健康科": 154,
}

# 均衡抽样每科上限（与 config.PER_LABEL_LIMIT 保持一致）
PER_LABEL_LIMIT = 3000


def main():
    parser = argparse.ArgumentParser(description="验证重灌后小科室样本量回升")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="目标 collection")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=CHROMADB_DIRECTORY)
    try:
        collection = client.get_collection(args.collection)
    except Exception as e:
        print(f"❌ 未找到 collection「{args.collection}」: {e}")
        return

    total = collection.count()
    print(f"collection「{args.collection}」总量：{total} 条")
    if total == 0:
        print("⚠️ 库为空，请先重灌（python jsonl2chroma.py --clear --source huatuo_lite）")
        return

    # 拉全部 metadata 统计 label（label 缺失/空记为「(空标签)」，与真实「其他」科室区分）
    metadatas = collection.get(limit=total, include=["metadatas"])["metadatas"]
    counter = Counter(m.get("label") or "(空标签)" for m in metadatas)

    print(f"\n各科室样本量（共 {len(counter)} 个科室，上限 {PER_LABEL_LIMIT}）：")
    for label, cnt in counter.most_common():
        full = FULL_DIST.get(label, "-")
        print(f"  {label:<8} {cnt:>6} 条    全量 {full}")

    print(f"\n重点小科室回升对比：")
    print(f"  {'科室':<8} {'旧 head':>8} {'当前':>6} {'全量':>8}  判定")
    all_up = True
    for label, baseline in BASELINE_HEAD.items():
        cur = counter.get(label, 0)
        full = FULL_DIST.get(label, "-")
        if cur > baseline:
            mark = "✅ 回升"
        elif cur == baseline:
            mark = "⚠️ 未回升"
            all_up = False
        else:
            mark = "⚠️ 反而下降"
            all_up = False
        print(f"  {label:<8} {baseline:>8} {cur:>6} {full:>8}  {mark}")

    print("\n结论：")
    if all_up:
        print("  ✅ 重点小科室样本量均已回升")
    else:
        print("  ⚠️ 存在小科室未回升，请确认是否执行了 --clear 重灌、以及均衡抽样是否生效")
    print("  注：急诊科全量仅 21 条且几乎全是模板题，清洗后可能接近 0，属预期（真实急症需外部补强）")


if __name__ == "__main__":
    main()
