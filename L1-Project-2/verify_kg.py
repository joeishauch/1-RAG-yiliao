# -*- coding: utf-8 -*-
"""知识图谱多跳推理验证：抽取质量统计 + 多跳合理性断言 + 人工可读 trace。

用法：
    python verify_kg.py --sample 200000   # 与 build_kg.py 同规模
    python verify_kg.py --sample 50000    # 快速验证

验证目标（MVP）：
    1. 规则切分能产出高质量三元组（各关系分布合理、疾病词典去噪有效）
    2. 多跳查询「症状 → 疾病 → 治疗/药物」能走通（不 crash、返回合理结果）
    3. 暴露「精确匹配」的边界（用户口语症状名未必与图节点名一致）
"""
import argparse
import random
import logging
from collections import defaultdict

from utils.kg_builder import build_from_jsonl, normalize_entity
from utils.kg_aliases import canonicalize
from utils import kg_schema as S
from utils.kg_query import query_by_symptom, resolve_entity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

DEFAULT_INPUT = "input/huatuo_knowledge_graph_qa/train_datasets.jsonl"

# 覆盖率断言用常见疾病（精确匹配，图里可能用「2型糖尿病」等变体，缺失只提示不 fail）
COMMON_DISEASES = [
    "糖尿病", "高血压", "冠心病", "肺炎", "胃癌", "哮喘", "甲亢",
    "乙肝", "脑梗死", "肺癌", "肾病综合征", "贫血", "癫痫",
    "类风湿关节炎", "白血病", "青光眼", "胆囊炎", "前列腺炎", "湿疹", "肝硬化",
]

# 多跳查询的症状（选常见症状，观察第一跳能否命中 + 第二跳能否展开）
SYMPTOM_QUERIES = [
    "发热", "咳嗽", "头痛", "胸痛", "腹痛", "呕吐", "腹泻",
    "皮疹", "水肿", "心绞痛", "晨僵", "呕血",
]

# 硬断言对（症状 -> 预期 top-1 疾病）；第一轮先打印实际结果，不因 miss 退出
# expected 允许包含匹配（如「急性上消化道出血」⊃「上消化道出血」）
HARD_ASSERTS = [
    ("心绞痛", "冠心病"),
    ("晨僵", "类风湿关节炎"),
    ("多尿", "糖尿病"),
    ("呕血", "上消化道出血"),
]


def _iter_out_relations(G, node):
    for _, nb, key in G.out_edges(node, keys=True):
        yield (nb, G.edges[node, nb, key]["relation"])


def _node_has_relation(G, node, rel):
    return any(r == rel for _, r in _iter_out_relations(G, node))


def part_a(G, triples, disease_dict, sample):
    print("\n" + "=" * 60)
    print("Part A — 抽取质量统计")
    print("=" * 60)

    pred_count = defaultdict(int)
    pred_heads = defaultdict(set)
    pred_tails = defaultdict(set)
    for (h, p, t), cnt in triples.items():
        pred_count[p] += 1
        pred_heads[p].add(h)
        pred_tails[p].add(t)

    print("\n[A1] 各关系分布（过滤后，去重三元组数）:")
    for p, cnt in sorted(pred_count.items(), key=lambda x: -x[1]):
        print(f"  {p:22s} 去重 {cnt:7d} 条 | 唯一 head {len(pred_heads[p]):6d} | 唯一 tail {len(pred_tails[p]):6d}")

    total_instances = sum(triples.values())
    print(f"\n[A2] 疾病词典规模: {len(disease_dict)} 个疾病")
    print(f"     过滤后三元组实例总数: {total_instances}")
    print(f"     图规模: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    if sample:
        print(f"     平均每条记录保留三元组: {total_instances / sample:.2f}（越接近 0 过滤越狠）")

    print("\n[A3] 常见疾病覆盖率（缺失仅提示，不 fail）:")
    hit = 0
    for d in COMMON_DISEASES:
        node = resolve_entity(G, d)
        if node is None:
            print(f"  ✗ {d}: 图中不存在（可能写法不同，如「2型糖尿病」）")
            continue
        flags = []
        for rel, label in [(S.REL_HAS_SYMPTOM, "症状"), (S.REL_HAS_DRUG, "药物"),
                           (S.REL_BELONGS_TO_DEPT, "科室"), (S.REL_HAS_TREATMENT, "治疗")]:
            if _node_has_relation(G, node, rel):
                flags.append(label)
        mark = "✓" if "症状" in flags and ("药物" in flags or "治疗" in flags) else "△"
        if mark == "✓":
            hit += 1
        print(f"  {mark} {d}: 有[{', '.join(flags) or '无下游边'}]")
    print(f"  覆盖率: {hit}/{len(COMMON_DISEASES)} 个疾病同时具备症状+药物/治疗边")

    print("\n[A4] 噪声抽查（每关系随机 5 条，人工 eyeball）:")
    by_pred = defaultdict(list)
    for (h, p, t) in triples.keys():
        by_pred[p].append((h, t))
    for p, items in sorted(by_pred.items(), key=lambda x: -len(x[1])):
        shown = random.sample(items, min(5, len(items)))
        print(f"  [{p}]")
        for h, t in shown:
            print(f"    {h}  ->  {t}")


def part_b(G):
    print("\n" + "=" * 60)
    print("Part B — 多跳查询（症状 → 疾病 → 治疗/药物）")
    print("=" * 60)

    for symptom in SYMPTOM_QUERIES:
        res = query_by_symptom(G, symptom, top_k=3)
        if res is None or not res["diseases"]:
            print(f"\n症状「{symptom}」: 图中未命中（精确匹配局限，需第二阶段症状模糊归一）")
            continue
        print(f"\n症状「{symptom}」→ {len(res['diseases'])} 个疾病:")
        for d in res["diseases"]:
            print(f"  {d['name']} (score={d['score']})")
            print(f"    药物: {[x[0] for x in d['drugs'][:3]]}")
            print(f"    治疗: {[x[0] for x in d['treatments'][:3]]}")
            print(f"    症状: {[x[0] for x in d['symptoms'][:3]]}")
            print(f"    科室: {[x[0] for x in d['department'][:2]]}")

    print("\n" + "=" * 60)
    print("Part B2 — 硬断言（预期 top-1 疾病）")
    print("=" * 60)
    for symptom, expected in HARD_ASSERTS:
        res = query_by_symptom(G, symptom, top_k=1)
        if res is None or not res["diseases"]:
            print(f"  ? 症状「{symptom}」未命中图中")
            continue
        actual = res["diseases"][0]["name"]
        # 别名归一后比较：既认「全称=简称」（冠心病=冠状动脉粥样硬化性心脏病），
        # 也认包含关系（上消化道出血 ⊆ 急性上消化道出血）
        ok = "✓" if (canonicalize(actual) == canonicalize(expected)
                     or expected in actual or actual in expected) else "✗"
        print(f"  {ok} 「{symptom}」预期 {expected}，实际 top-1 {actual}")


def main():
    parser = argparse.ArgumentParser(description="验证知识图谱多跳推理")
    parser.add_argument("--sample", type=int, default=200000, help="抽样条数；0 表示全量")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    args = parser.parse_args()

    sample = args.sample if args.sample and args.sample > 0 else None
    G, disease_dict, triples = build_from_jsonl(args.input, sample)
    part_a(G, triples, disease_dict, args.sample)
    part_b(G)


if __name__ == "__main__":
    main()
