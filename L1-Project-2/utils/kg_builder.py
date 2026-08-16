# -*- coding: utf-8 -*-
"""知识图谱构建：读 jsonl → 规则切分 → 疾病词典去噪 → 三元组 → networkx 图。

数据源：input/huatuo_knowledge_graph_qa/train_datasets.jsonl（79.8 万条）。
该数据集的 questions 是模板化问句（已编码关系类型）、answers 是「；」分隔的
实体枚举，因此只需规则切分即可产出 (实体, 关系, 实体) 三元组，无需 LLM 抽取。

关键去噪手段——疾病词典：
    belongs_to_department / has_complication 两个关系的 head 大多是「疾病」，
    症状 / 解剖部位 / 术式一般不会有「就诊科室」「并发症」，故用它们做疾病种子，
    第二遍遍历时只保留 head ∈ 疾病词典 的三元组，过滤掉「耳漏」「下肢动脉栓塞术」
    这类被误当主语的非疾病实体。
"""
import json
import re
import logging
import pickle
from collections import Counter

import networkx as nx

from utils import kg_schema as S

logger = logging.getLogger(__name__)

# 主语（head）末尾需要去掉的修饰/标点字符集
_HEAD_TRIM = "的。，？? 可能会是：:"

# 三元组关系规则表（复用 schema）
HEAD_TYPE = {pred: htype for pred, htype, _ttype, _pats in S.RELATION_RULES}
TAIL_TYPE = {pred: ttype for pred, _htype, ttype, _pats in S.RELATION_RULES}


# -------------------- 读取与基础解析 --------------------

def _read_records(path, sample=None):
    """逐行读 jsonl，返回 (record) 迭代器；sample 限制前 N 条"""
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if sample is not None and count >= sample:
                break
            yield json.loads(line)
            count += 1


def _first_question(record):
    """兼容 questions 是 [[..]] 或 [..] 两种结构，取第一个问题文本"""
    qs = record.get("questions", [])
    if not qs:
        return ""
    q0 = qs[0]
    if isinstance(q0, list):
        return q0[0] if q0 else ""
    return q0


def _first_answer(record):
    """取第一个答案文本"""
    ans = record.get("answers", [])
    return ans[0] if ans else ""


def parse_relation(question):
    """从问句模板识别关系类型，返回 (predicate, head)；未命中返回 (None, None)。

    采用「前缀取 head」：问句形如「<主语>的<关系短语>…」，命中关系短语后，
    取其前缀作为主语实体，并去掉末尾的「的/可能/会/是」等修饰。
    """
    for pred, _htype, _ttype, pats in S.RELATION_RULES:
        for p in pats:
            m = re.search(p, question)
            if m:
                head = question[:m.start()].rstrip(_HEAD_TRIM).strip()
                return pred, head
    return None, None


def split_answer(answer):
    """按「；」切分实体枚举，去空去重、去「病因：」前缀。

    注意：不用「，」「、」切分——has_cause/has_attribute 的长文本答案
    会被逗号切成碎片；只用「；」这一明确枚举分隔符。
    """
    parts = re.split(r"[；;]", answer)
    out = []
    for p in parts:
        p = p.strip().strip("。.，,；; ")
        p = re.sub(r"^病因[：:]\s*", "", p)
        # 过滤脏值 + 过长文本（实体枚举都很短，长文本是未移除关系的残留）
        if p in ("无", "无特殊", "无特定人群", "暂无", "不详"):
            continue
        if len(p) > 30:
            continue
        if p and p not in out:
            out.append(p)
    return out


def normalize_entity(s):
    """第一版实体归一化：只做字符串级清理（去空白），不做同义合并。

    同义对齐（糖尿病 vs 2型糖尿病）属第二阶段，届时替换本函数即可。
    """
    return re.sub(r"\s+", "", s.strip())


# -------------------- 症状词典去噪 --------------------

# 症状候选中的「非症状」字符（检查指标里的符号/单位/括号）
_SYM_BAD_CHARS = set("≥≤><%.．/\\+-℃°()（）~～=＝")
# 检查结果/变化方向常用后缀：「X改变 / X异常 / X升高 / X增加…」是检查结果句式，不是症状
_SYM_BAD_SUFFIX = (
    "改变", "异常", "升高", "增高", "降低", "减低", "增多", "减少", "增加", "下降",
    "增厚", "变薄", "狭窄", "扩张", "阳性", "阴性", "水平", "缺失", "损害", "减退",
    "减弱", "增强", "加快", "减慢", "差",
)
# 病理/影像/实验室术语特征词（含这些词 = 病理描述或检查所见，非患者主诉症状）
_SYM_PATHOLOGY_WORDS = (
    "病变", "沉积", "萎缩", "凋亡", "斑块", "坏死", "增生", "浸润", "纤维化",
    "钙化", "脱髓鞘", "变性", "退化", "缺损", "重建", "参数", "模型", "实验",
    "切片", "标本", "染色", "信号", "通路", "表达", "增殖", "分化", "受体",
    "基因", "因子", "含量", "浓度", "比值", "曲线", "测定", "检测", "范围",
)
# 含结果词但确是真症状的白名单（避免误杀神经系统/精神科症状）
_SYM_ALLOW = {"感觉异常", "行为异常", "月经异常"}


def is_valid_symptom(s):
    """判断是否为「真症状」（可作为「症状→疾病」入口）。

    过滤三类噪声（针对 has_symptom 答案里混入的检查指标/影像表现/病理术语）：
      1. 含英文/数字/符号（HbA1c、≥7.0mmol/L、150次/min）——检查指标
      2. 过短/过长（<2 或 >12 字）——不是症状
      3. 含病理/影像术语特征词（脱髓鞘、冠脉病变、神经元凋亡…）——病理描述
      4. 以「改变/异常/升高/增加…」结尾——检查结果句式
    白名单保留「感觉异常」等少数含结果词的真症状。
    """
    if not 2 <= len(s) <= 12:
        return False
    for ch in s:
        if ch in _SYM_BAD_CHARS:
            return False
        if ch.isascii() and (ch.isalpha() or ch.isdigit()):
            return False
    if s in _SYM_ALLOW:
        return True
    if any(w in s for w in _SYM_PATHOLOGY_WORDS):
        return False
    if s.endswith(_SYM_BAD_SUFFIX):
        return False
    return True


# -------------------- 疾病词典与三元组 --------------------

def _collect_disease_seeds(path, sample=None):
    """第一遍遍历：收集疾病种子（belongs_to_department 的 head + has_complication 的 head）。

    has_complication 的 head 可能混入术式（如「下肢动脉栓塞术的并发症」），
    用「含术字」启发式排除术式。
    """
    seeds = set()
    for rec in _read_records(path, sample):
        q = _first_question(rec)
        pred, head = parse_relation(q)
        if not head:
            continue
        head = normalize_entity(head)
        if pred == S.REL_BELONGS_TO_DEPT:
            seeds.add(head)
        elif pred == S.REL_HAS_COMPLICATION and "术" not in head:
            seeds.add(head)
    return seeds


def _add_triple(triples, pred, head, tail, disease_dict):
    """按关系类型分治过滤，把合格三元组计入 Counter；has_symptom 反转生成 symptom_of"""
    if pred == S.REL_SYMPTOM_OF:
        # head=症状, tail=疾病；要求 tail 是疾病
        if tail in disease_dict:
            triples[(head, pred, tail)] += 1
    elif pred == S.REL_HAS_COMPLICATION:
        # 两端都必须是疾病
        if head in disease_dict and tail in disease_dict:
            triples[(head, pred, tail)] += 1
    elif pred == S.REL_HAS_SYMPTOM:
        if head in disease_dict:
            triples[(head, pred, tail)] += 1
            # 反转：只有「真症状」才能作为「症状→疾病」入口，
            # 过滤 has_symptom 答案里混入的检查指标/影像表现（如「空腹血糖≥7.0mmol/L」）
            if is_valid_symptom(tail):
                triples[(tail, S.REL_SYMPTOM_OF, head)] += 1
    else:
        # has_drug / has_treatment / has_examination / has_cause / belongs_to_department / has_attribute
        if head in disease_dict:
            triples[(head, pred, tail)] += 1


def _build_triples(path, sample, disease_dict):
    """第二遍遍历：切分 + 过滤 + 归并三元组，返回 Counter[(head, pred, tail) -> count]"""
    triples = Counter()
    for rec in _read_records(path, sample):
        q = _first_question(rec)
        a = _first_answer(rec)
        pred, head = parse_relation(q)
        if pred is None or not head:
            continue
        head = normalize_entity(head)
        tails = [normalize_entity(t) for t in split_answer(a)]
        for tail in tails:
            if tail:
                _add_triple(triples, pred, head, tail, disease_dict)
    return triples


# -------------------- 建图与序列化 --------------------

def build_graph(triples):
    """把三元组 Counter 构建成 nx.MultiDiGraph（节点带 type/count/indegree，边带 relation/count）"""
    # 先统计 symptom_of 入度（distinct 症状数），用于查询端的 hub 抑制
    symptom_indegree = Counter()
    for (head, pred, tail) in triples:
        if pred == S.REL_SYMPTOM_OF:
            symptom_indegree[tail] += 1

    G = nx.MultiDiGraph()
    for (head, pred, tail), cnt in triples.items():
        htype = HEAD_TYPE.get(pred, S.NODE_DISEASE)
        ttype = TAIL_TYPE.get(pred, S.NODE_DISEASE)
        if G.has_node(head):
            G.nodes[head]["count"] += cnt
        else:
            G.add_node(head, type=htype, count=cnt)
        if G.has_node(tail):
            G.nodes[tail]["count"] += cnt
        else:
            G.add_node(tail, type=ttype, count=cnt)
        G.add_edge(head, tail, key=pred, relation=pred,
                   source="huatuo_knowledge_graph_qa", count=cnt)

    # 回填 symptom_indegree（hub 抑制用；缺失时查询端默认 1）
    for d, deg in symptom_indegree.items():
        if d in G:
            G.nodes[d]["symptom_indegree"] = deg
    return G


def build_from_jsonl(input_path, sample=None):
    """读 jsonl → 疾病词典 → 三元组 → 图，返回 (graph, disease_dict, triples)"""
    logger.info(f"开始构建知识图谱: {input_path}, sample={sample}")
    disease_dict = _collect_disease_seeds(input_path, sample)
    logger.info(f"疾病词典规模: {len(disease_dict)} 个疾病")
    triples = _build_triples(input_path, sample, disease_dict)
    logger.info(f"过滤后三元组数: {len(triples)} 条（去重前 {sum(triples.values())} 条实例）")
    G = build_graph(triples)
    logger.info(f"图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G, disease_dict, triples


def save_graph(G, path):
    """用标准库 pickle 序列化（networkx 3.6 已移除 gpickle；pickle 对
    MultiDiGraph + 中文节点 + 简单属性完全保真）。本地自产文件，安全。"""
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"图已序列化到 {path}")


def load_graph(path):
    logger.info(f"加载缓存图: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_or_load(input_path, out_path, sample=None, force=False):
    """有缓存且非 force 直接加载，否则重建并缓存"""
    if not force and out_path and _path_exists(out_path):
        return load_graph(out_path)
    G, disease_dict, triples = build_from_jsonl(input_path, sample)
    if out_path:
        save_graph(G, out_path)
    return G


def _path_exists(path):
    import os
    return os.path.exists(path)
