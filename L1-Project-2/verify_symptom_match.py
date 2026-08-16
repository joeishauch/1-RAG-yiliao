# -*- coding: utf-8 -*-
"""验证症状模糊匹配：口语症状名 → 图中规范节点（embedding 最近邻 + 阈值 0.6）。

用法：
    python verify_symptom_match.py

验证目标：
    1. 口语症状词（精确 miss）能被 fuzzy_match_symptom 归一到一个「语义等价」的规范节点；
    2. 阈值 0.6 能挡住「相似度不足」的候选（宁可漏、不可错）；
    3. 端到端：归一后的症状能继续走通「症状 → 疾病」第一跳。

判断口径：命中节点 ∈ 可接受等价集合（图里同一症状常有多个规范变体，
如「睡不着」→ 无法入睡/难以入睡/失眠，语义等价即算命中，不做字符串全等）。
"""
import logging

import numpy as np

from utils.kg_builder import load_graph
from utils.kg_query import query_by_symptom, resolve_entity
from utils.kg_symptom_match import (
    SYMPTOM_SIM_THRESHOLD, _get_embedding, fuzzy_match_symptom, get_symptom_index,
)

GRAPH_PATH = "kg_graph.pkl"

# 抑制 httpx/openai 的 INFO 请求日志（索引构建有 500+ 次请求，避免刷屏）
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# 口语 → 可接受等价节点集合（语义等价即可命中）。
# 注：「老想上厕所」为反向用例——口语偏「排便」，与「尿频」embedding 距离远，预期被阈值挡住。
CASES = [
    ("胸口闷",     {"胸闷", "胸闷不适", "胸闷胸痛"}),
    ("头疼",       {"头疼", "头痛", "头昏头痛"}),
    ("拉肚子",     {"腹泻", "腹胀腹泻"}),
    ("喘不上气",   {"气喘", "气促", "喘憋", "气短", "呼吸困难"}),
    ("心口疼",     {"心痛", "胸痛", "心前区疼痛", "心绞痛"}),
    ("睡不着",     {"失眠", "无法入睡", "难以入睡", "入睡困难"}),
    ("没力气",     {"乏力", "无力", "疲乏"}),
    ("流鼻涕",     {"流鼻涕", "流涕", "鼻塞流涕"}),
    ("打喷嚏",     {"打喷嚏", "喷嚏"}),
    ("老想上厕所", {"尿频", "小便频数", "尿次数增多"}),   # 预期 miss
    ("恶心反胃",   {"恶心", "反胃", "恶心与呕吐"}),
    ("浑身发冷",   {"寒战", "全身发冷", "发冷", "畏寒"}),
]


def _top_k_similar(names, mat, embedding, qname, k=3):
    """直接算口语词与全图症状的 top-k 余弦相似度（绕过阈值，看阈值裁掉了什么）。"""
    qv = np.asarray(embedding.embed_query(qname), dtype=np.float32)
    qn = np.linalg.norm(qv)
    if qn == 0:
        return []
    qv = qv / qn
    sims = mat @ qv
    idx = np.argsort(-sims)[:k]
    return [(names[i], float(sims[i])) for i in idx]


def main():
    G = load_graph(GRAPH_PATH)
    names, mat = get_symptom_index(G)   # 首次调用会构建全图症状向量索引（有磁盘缓存，二次秒开）
    emb = _get_embedding()

    print(f"图规模: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"症状节点数（索引规模）: {len(names)}")
    print(f"相似度阈值: {SYMPTOM_SIM_THRESHOLD}\n")

    hit = 0
    e2e_hit = 0
    for oral, acceptable in CASES:
        exact = resolve_entity(G, oral)
        fuzzy = fuzzy_match_symptom(G, oral)
        ok = fuzzy is not None and fuzzy in acceptable
        if ok:
            hit += 1

        top3 = _top_k_similar(names, mat, emb, oral)
        top3_str = " | ".join(f"{n}:{s:.3f}" for n, s in top3)
        exact_str = f"精确命中[{exact}]" if exact else "精确miss"

        # 端到端：口语词走完整 query_by_symptom（精确 miss → 模糊归一 → 第一跳）
        res = query_by_symptom(G, oral, top_k=1)
        e2e = res and res["diseases"] and res["diseases"][0]["name"]
        if e2e:
            e2e_hit += 1

        print(f"{'✓' if ok else '✗'} {oral} -> 模糊命中「{fuzzy}」")
        print(f"    {exact_str} | top-3: {top3_str}")
        if e2e:
            print(f"    端到端 top-1 疾病: {e2e}")
        else:
            print(f"    端到端: 未命中疾病（阈值挡住，或图中该症状无 symptom_of 边）")

    print(f"\n语义等价命中率: {hit}/{len(CASES)}")
    print(f"端到端走通: {e2e_hit}/{len(CASES)}")
    print("\n注：端到端 top-1 疾病多为罕见病，是 symptom_of 边 cnt≈1 无共现权重的固有问题，")
    print("    与症状模糊匹配无关；本脚本只验证「口语症状 → 规范节点」这一步的归一同源。")


if __name__ == "__main__":
    main()
