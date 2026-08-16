# -*- coding: utf-8 -*-
"""知识图谱多跳查询：症状 → 疾病 → 治疗/药物/症状/检查/科室 的链式推理。

只用 networkx 精确匹配 + 邻居遍历，不用 all_simple_paths（hub 节点会指数爆炸）、
不用 shortest_path（只给一条最优，丢失「一个症状对应多个疾病」的备选）。
采用自定义受限 BFS：第一跳走 symptom_of，第二跳走疾病的下游关系，
每层按边 count 排序、branch_limit 限制分支，防止 hub 疾病（如「发热」）爆炸。
"""
import math
import threading

from utils.kg_builder import load_graph, normalize_entity
from utils.kg_aliases import canonicalize
from utils.kg_symptom_match import fuzzy_match_symptom
from utils import kg_schema as S

# 图懒加载单例（双重检查锁，与 tools_config.get_triage_context 同款模式）
_kg = None
_kg_lock = threading.Lock()

BRANCH_LIMIT = 20   # 每层最多展开的分支数，防 hub 节点爆炸


def get_kg(path):
    """加载（并缓存）知识图谱，进程内只加载一次"""
    global _kg
    if _kg is None:
        with _kg_lock:
            if _kg is None:
                _kg = load_graph(path)
    return _kg


def resolve_entity(G, name):
    """把用户输入归一化（去空白 + 别名归一到规范疾病节点）后查图。

    别名归一放在「查图之前」，故「冠心病」这类被误标为 symptom 的简称节点
    会被纠正到「冠状动脉粥样硬化性心脏病」等真疾病节点。
    """
    n = normalize_entity(name)
    n = canonicalize(n)
    return n if n in G else None


def constrained_neighbors(G, node, allowed_relations):
    """返回 node 的出边中 relation ∈ allowed_relations 的 [(neighbor, relation, count)]"""
    out = []
    for _, nb, key in G.out_edges(node, keys=True):
        rel = G.edges[node, nb, key]["relation"]
        if rel in allowed_relations:
            out.append((nb, rel, G.edges[node, nb, key].get("count", 1)))
    return out


def query_by_symptom(G, symptom, top_k=5, branch_limit=BRANCH_LIMIT):
    """核心查询：症状 → 疾病 → 下游实体。

    返回：
        {"symptom": 规范化症状名,
         "diseases": [{name, score, drugs, treatments, symptoms,
                       examinations, complications, department, causes}, ...]}
        每个列表元素是 (实体名, 边 count)，按 count 降序；未命中返回 None。
    """
    s = resolve_entity(G, symptom)
    if s is None:
        # 精确 + 疾病别名 miss → 症状模糊匹配（口语名 → 规范症状节点）。
        # 只在这里做，不放进 resolve_entity，避免模糊匹配污染疾病名查询。
        s = fuzzy_match_symptom(G, normalize_entity(symptom))
    if s is None:
        return None

    # 第一跳：症状 -> 疾病（symptom_of）
    hops = [(nb, cnt) for nb, _rel, cnt in constrained_neighbors(G, s, S.FIRST_HOP_RELATIONS)]
    # hub 抑制（IDF 惩罚）：被极多症状指向的「泛化疾病」（如「糖尿病」连 25+ 高频症状）
    # 会在任意症状查询里冒出来，按入度稀释其边权。
    # 权衡：模板数据里 symptom_of 边 count 几乎全为 1（无共现频次信号），
    # 惩罚过重会把「多尿→糖尿病」这类正确映射也压下去。系数 0.5 是折中——
    # 保留抑制能力，又不过分误伤。根治要靠第二阶段（实体对齐 + 关联强度 + LLM）。
    scored = []
    for nb, cnt in hops:
        indeg = G.nodes[nb].get("symptom_indegree", 1)
        score = cnt / (1.0 + 0.5 * math.log1p(indeg))
        scored.append((nb, score))
    scored.sort(key=lambda x: -x[1])

    diseases = []
    seen = set()
    for dname, dscore in scored[:branch_limit]:
        if dname in seen:
            continue
        seen.add(dname)
        entry = {
            "name": dname, "score": dscore,
            "drugs": [], "treatments": [], "symptoms": [],
            "examinations": [], "complications": [], "department": [], "causes": [],
        }
        # 第二跳：疾病 -> 下游，按关系类型分桶
        for nb, rel, cnt in constrained_neighbors(G, dname, S.SECOND_HOP_RELATIONS):
            if rel == S.REL_HAS_DRUG:
                entry["drugs"].append((nb, cnt))
            elif rel == S.REL_HAS_TREATMENT:
                entry["treatments"].append((nb, cnt))
            elif rel == S.REL_HAS_SYMPTOM:
                entry["symptoms"].append((nb, cnt))
            elif rel == S.REL_HAS_EXAMINATION:
                entry["examinations"].append((nb, cnt))
            elif rel == S.REL_HAS_COMPLICATION:
                entry["complications"].append((nb, cnt))
            elif rel == S.REL_BELONGS_TO_DEPT:
                entry["department"].append((nb, cnt))
            elif rel == S.REL_HAS_CAUSE:
                entry["causes"].append((nb, cnt))
        # 各关系列表按 count 降序
        for key in ("drugs", "treatments", "symptoms", "examinations",
                    "complications", "department", "causes"):
            entry[key].sort(key=lambda x: -x[1])
            entry[key] = entry[key][:branch_limit]
        diseases.append(entry)
        if len(diseases) >= top_k:
            break

    return {"symptom": s, "diseases": diseases}
