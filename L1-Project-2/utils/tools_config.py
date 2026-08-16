# -*- coding: utf-8 -*-
"""工具配置：注册智能分诊检索工具 + 混合检索（BM25 + 向量 + RRF + BGE-reranker 重排）。

检索来源从「纯向量」升级为「混合检索」：
- BM25 关键词（jieba 分词）与向量语义双路召回，互补关键词精确匹配与语义泛化；
- RRF 融合两路结果（对分数尺度不敏感，无需归一化）；
- 可选 cross-encoder（BGE-reranker）对融合候选集重排，进一步精筛。

科室统计 + 先验校准逻辑不变（_search_and_rank 仅换检索来源）。
"""
import json
import logging
import threading
from collections import Counter

import chromadb
import jieba
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi

from utils.config import Config
from utils.label_cleaner import INVALID_LABELS
from utils.kg_query import get_kg, query_by_symptom

logger = logging.getLogger(__name__)

# 分诊检索参数
TRIAGE_TOP_K = 20   # 检索相似病例条数（用于统计科室分布）
TRIAGE_TOP_N = 5    # 第一层返回的候选科室数量
CONSULT_TOP_K = 10  # 会诊时每个科室检索的病例条数
CONSULT_TOP_N = 3   # 参与深度会诊的科室数量（第二层）
QA_TOP_K = 5        # 科普问答兜底：检索返回的相似问答条数


class HybridRetriever:
    """BM25 关键词 + 向量语义双路召回，RRF 融合 + cross-encoder 重排。

    三档模式（供评估对比）：
        "vector"        纯向量（基线）
        "hybrid"        BM25 + 向量 + RRF 融合
        "hybrid_rerank" 在 hybrid 基础上加 BGE-reranker 重排（失败自动降级 hybrid）
    """

    def __init__(self):
        self.vectorstore = None       # Chroma 向量存储（build 时注入）
        self.bm25 = None              # BM25Okapi 索引
        self.docs = []                # 全库 Document 列表（含 metadata），与 bm25 corpus 对齐
        self.reranker = None          # CrossEncoder（懒加载）
        self.reranker_failed = False  # 重排器加载失败标记

    @staticmethod
    def _tokenize(text):
        """jieba 分词，过滤空白 token。"""
        return [w for w in jieba.cut(text) if w.strip()]

    def build(self, vectorstore):
        """从 ChromaDB 加载全库 question + metadata，构建 BM25 索引。"""
        self.vectorstore = vectorstore
        logger.info("构建 BM25 索引...")
        raw = vectorstore._collection.get(include=["documents", "metadatas"])
        documents = raw["documents"]
        metadatas = raw["metadatas"]
        self.docs = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(documents, metadatas)
        ]
        corpus = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 索引构建完成（{len(self.docs)} 条）")

    def _search_bm25(self, query, top_k):
        """BM25 关键词检索，返回 [(corpus_idx, score), ...] 按分数降序。"""
        if self.bm25 is None:
            return []
        tokenized = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(idx, score) for idx, score in ranked if score > 0]

    def _get_reranker(self):
        """懒加载 CrossEncoder（onnx 后端 CPU 加速，失败降级 torch / 无重排）。"""
        if self.reranker is None and not self.reranker_failed:
            # 优先 onnx（onnxruntime 已装 + 模型目录有 onnx/model.onnx，CPU 推理比 torch 快）；
            # onnx 加载失败回退 torch，两个都失败才标记降级为无重排。
            backends = [Config.RERANKER_BACKEND] + [b for b in ("onnx", "torch") if b != Config.RERANKER_BACKEND]
            for backend in backends:
                try:
                    from sentence_transformers import CrossEncoder
                    self.reranker = CrossEncoder(
                        Config.RERANKER_MODEL_PATH,
                        max_length=Config.RERANK_MAX_LENGTH,
                        backend=backend,
                    )
                    logger.info(f"BGE-reranker 重排器加载成功（backend={backend}）")
                    break
                except Exception as e:
                    logger.warning(f"重排器加载失败（backend={backend}），尝试下一个后端: {e}")
            else:
                self.reranker_failed = True
                logger.warning("所有后端加载失败，降级为无重排")
        return self.reranker

    def search(self, query, top_k=TRIAGE_TOP_K, mode="hybrid_rerank"):
        """混合检索，返回 Document 列表（含 label/answer metadata）。"""
        if mode == "vector":
            return self.vectorstore.similarity_search(query, k=top_k)

        # 双路召回
        bm25_hits = self._search_bm25(query, top_k=Config.HYBRID_BM25_TOP_K)
        vec_docs = self.vectorstore.similarity_search(query, k=Config.HYBRID_VEC_TOP_K)

        # RRF 融合（page_content 去重）
        doc_map = {}
        rrf_scores = {}
        for rank, (idx, _score) in enumerate(bm25_hits):
            doc = self.docs[idx]
            key = doc.page_content
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (Config.RRF_K + rank + 1)
        for rank, doc in enumerate(vec_docs):
            key = doc.page_content
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (Config.RRF_K + rank + 1)

        ranked_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        candidates = [doc_map[key] for key in ranked_keys]

        # 候选集不足 top_k 或不要求重排，直接返回
        if mode != "hybrid_rerank" or len(candidates) <= top_k:
            return candidates[:top_k]

        # cross-encoder 重排
        reranker = self._get_reranker()
        if reranker is None:
            return candidates[:top_k]
        pairs = [[query, doc.page_content] for doc in candidates]
        try:
            scores = reranker.predict(pairs)
            scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _s in scored[:top_k]]
        except Exception as e:
            logger.warning(f"重排失败，降级为 RRF 结果: {e}")
            return candidates[:top_k]


# 模块级缓存：vectorstore + 先验分布 + BM25 索引全进程只建一次
_triage_ctx = None
_ctx_lock = threading.Lock()

# 科普问答 vectorstore 的模块级缓存（medical_qa collection）
_qa_ctx = None
_qa_lock = threading.Lock()


def get_triage_context(llm_embedding):
    """懒加载并缓存分诊检索上下文。

    Args:
        llm_embedding: 嵌入模型实例。

    Returns:
        tuple: (vectorstore, prior_counter, prior_total, hybrid_retriever)
    """
    global _triage_ctx
    if _triage_ctx is None:
        with _ctx_lock:
            if _triage_ctx is None:
                vectorstore = Chroma(
                    persist_directory=Config.CHROMADB_DIRECTORY,
                    collection_name=Config.CHROMADB_COLLECTION_NAME,
                    embedding_function=llm_embedding,
                )
                _client = chromadb.PersistentClient(path=Config.CHROMADB_DIRECTORY)
                _col = _client.get_collection(Config.CHROMADB_COLLECTION_NAME)
                _metas = _col.get(include=["metadatas"])["metadatas"]
                prior_counter = Counter(m.get("label", "未知科室") for m in _metas)
                prior_total = len(_metas)
                hybrid = HybridRetriever()
                hybrid.build(vectorstore)
                _triage_ctx = (vectorstore, prior_counter, prior_total, hybrid)
    return _triage_ctx


def _search_and_rank(hybrid, prior_counter, prior_total, query, top_n=TRIAGE_TOP_N):
    """混合检索 → 统计科室 → 先验校准，返回 (scored, examples)。

    校准算法：score = cnt² / expected（expected = 检索条数 × 科室先验占比），
    类似 TF-IDF/卡方，同时抑制大样本科室基数优势 + 小样本 1 次命中噪声。

    Returns:
        tuple: (scored, examples)
            scored: [(label, cnt, score), ...] 按 score 降序
            examples: {label: 代表病例摘要}
    """
    docs = hybrid.search(query, top_k=TRIAGE_TOP_K, mode=Config.TRIAGE_RETRIEVAL_MODE)
    n_retrieved = len(docs)

    counter = Counter()
    examples = {}
    for d in docs:
        label = d.metadata.get("label", "未知科室")
        if label in INVALID_LABELS:  # 过滤非具体科室标签
            continue
        counter[label] += 1
        if label not in examples:
            examples[label] = d.metadata.get("answer", "")[:100]

    if not counter:
        return [], examples

    scored = []
    for label, cnt in counter.items():
        prior_cnt = prior_counter.get(label, 1)
        expected = n_retrieved * prior_cnt / prior_total
        score = (cnt * cnt) / expected if expected > 0 else 0.0
        scored.append((label, cnt, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_n], examples


def triage_rank(hybrid, prior_counter, prior_total, query, top_n=TRIAGE_TOP_N):
    """返回校准后的 Top-N 科室 [(label, cnt, score), ...]，供会诊节点复用。"""
    scored, _ = _search_and_rank(hybrid, prior_counter, prior_total, query, top_n=top_n)
    return scored


def get_qa_context(llm_embedding):
    """懒加载并缓存科普问答检索向量库（medical_qa collection）。

    与分诊检索（medical_triage）分离，供科普兜底工具 medical_qa 使用。
    """
    global _qa_ctx
    if _qa_ctx is None:
        with _qa_lock:
            if _qa_ctx is None:
                _qa_ctx = Chroma(
                    persist_directory=Config.CHROMADB_DIRECTORY,
                    collection_name=Config.CHROMADB_QA_COLLECTION_NAME,
                    embedding_function=llm_embedding,
                )
    return _qa_ctx


# 药物禁忌知识库的模块级缓存（drug_contraindications.json，纯 JSON 不依赖 embedding）
_drug_ctx = None
_drug_lock = threading.Lock()


def get_drug_context():
    """懒加载药物禁忌知识库：读 drug_contraindications.json → list[dict]。

    每个元素形如 {"drug_name": str, "contraindications": [str, ...], "source": str}。
    纯 JSON 数据源，不依赖 llm_embedding，故无参数。
    """
    global _drug_ctx
    if _drug_ctx is None:
        with _drug_lock:
            if _drug_ctx is None:
                with open(Config.DRUG_CONTRA_PATH, encoding="utf-8") as f:
                    _drug_ctx = json.load(f)
    return _drug_ctx


def search_drug_taboo(drug_query, records):
    """按药名匹配禁忌记录：精确 → 双向包含，返回命中记录列表（空则 []）。

    三层匹配（从严到宽）：
        1. 精确：drug_name == query
        2. 药名含查询：query in drug_name（「阿司匹林」→「阿司匹林栓」等）
        3. 查询含药名：drug_name in query（query 带了完整药名 + 别的字）
    """
    q = (drug_query or "").strip()
    if not q:
        return []
    # 1. 精确匹配
    exact = [r for r in records if r.get("drug_name", "").strip() == q]
    if exact:
        return exact
    # 2. 药名含查询（简称匹配完整药名）
    contained = [r for r in records if q in r.get("drug_name", "")]
    if contained:
        return contained
    # 3. 查询含药名（query 里带了完整药名 + 别的字）
    return [r for r in records if r.get("drug_name", "") and r["drug_name"] in q]


def get_tools(llm_embedding):
    """
    创建并返回工具列表

    Args:
        llm_embedding: 嵌入模型实例，用于初始化向量存储

    Returns:
        list: 工具列表（分诊 retrieve + 科普 medical_qa + 知识图谱 kg_query + 药物禁忌 drug_taboo）
    """
    # 复用模块级缓存的 vectorstore + 先验分布 + 混合检索器（与 consult 节点共享）
    _, prior_counter, prior_total, hybrid = get_triage_context(llm_embedding)
    # 科普问答检索向量库（medical_qa collection）
    qa_vectorstore = get_qa_context(llm_embedding)

    # 自定义分诊检索工具：混合检索相似病例 → 统计科室标签 → 先验校准 → 输出「科室 + 置信度」
    # 工具名保持 retrieve，ragAgent.py 的 route_after_tools 靠 "retrieve" in name 路由到会诊链路
    @tool
    def retrieve(query: str) -> str:
        """分诊查询工具：根据症状描述检索相似病例，返回候选科室及校准后的置信度分布。"""
        scored, examples = _search_and_rank(hybrid, prior_counter, prior_total, query)
        if not scored:
            return "未检索到相关分诊信息。"

        total_score = sum(s for _, _, s in scored) or 1.0

        # 组装「科室分布」字符串
        dist_lines = []
        for label, cnt, score in scored:
            conf = score / total_score
            dist_lines.append(f"{label}：{cnt}次（{conf:.0%}）")

        example_lines = [
            f"- {label}：{ans}"
            for label, ans in examples.items() if ans
        ]

        parts = ["候选科室分布（已按全库先验分布校准）：", "；".join(dist_lines)]
        if example_lines:
            parts.append("代表病例摘要：")
            parts.extend(example_lines)
        return "\n".join(parts)

    # 科普问答工具：检索医学知识库，返回相关问答的答案（供 generate_qa 节点综合）
    @tool
    def medical_qa(query: str) -> str:
        """医学知识问答工具：检索医学知识库，回答疾病、症状、用药、保健等科普类问题（非分诊/挂号咨询）。"""
        docs = qa_vectorstore.similarity_search(query, k=QA_TOP_K)
        if not docs:
            return "未检索到相关医学知识。"

        lines = []
        for i, d in enumerate(docs, 1):
            ans = d.metadata.get("answer", "").strip()
            if ans:
                lines.append(f"{i}. {ans}")
        return "\n".join(lines) if lines else "未检索到相关医学知识。"

    # 知识图谱工具：症状 → 可能疾病 + 治疗/药物/科室（多跳推理，与向量检索互补）
    @tool
    def kg_query(symptom: str) -> str:
        """症状疾病推理工具：根据症状查询医学知识图谱，返回可能的疾病及对应的治疗/药物/检查/科室等。用于「某症状可能是什么病、什么原因引起」类咨询。"""
        G = get_kg(Config.KG_GRAPH_PATH)
        res = query_by_symptom(G, symptom, top_k=5)
        if res is None or not res["diseases"]:
            return "知识图谱中未检索到该症状对应的疾病信息。"
        lines = [f"症状「{res['symptom']}」可能关联的疾病（按相关性排序）："]
        for i, d in enumerate(res["diseases"], 1):
            seg = [f"{i}. {d['name']}"]
            if d["department"]:
                seg.append(f"科室：{'、'.join(x[0] for x in d['department'][:3])}")
            if d["drugs"]:
                seg.append(f"药物：{'、'.join(x[0] for x in d['drugs'][:5])}")
            if d["treatments"]:
                seg.append(f"治疗：{'、'.join(x[0] for x in d['treatments'][:5])}")
            if d["symptoms"]:
                seg.append(f"伴随症状：{'、'.join(x[0] for x in d['symptoms'][:5])}")
            lines.append(" | ".join(seg))
        return "\n".join(lines)

    # 药物禁忌工具：按药名精确/包含匹配独立 JSON 知识库，返回禁忌症/禁忌人群
    @tool
    def drug_taboo(drug: str) -> str:
        """药物禁忌查询工具：根据药物名查询禁忌症/禁忌人群（如孕妇、过敏、肝肾功能不全等禁用/慎用情况）。用于「XX药有什么禁忌」「孕妇/儿童/肝肾功能不全者能吃XX药吗」类咨询。"""
        records = get_drug_context()
        hits = search_drug_taboo(drug, records)
        if not hits:
            return f"未在知识库中找到药物「{drug}」的禁忌信息。"
        lines = [f"药物「{drug}」的禁忌信息（共 {len(hits)} 条匹配）："]
        for r in hits[:5]:
            name = r.get("drug_name", "")
            items = "；".join(r.get("contraindications") or [])
            if not items:
                items = "（该药暂无明确禁忌记录）"
            lines.append(f"- 【{name}】{items}")
        return "\n".join(lines)

    # 返回工具列表（分诊 + 科普 + 知识图谱推理 + 药物禁忌）
    return [retrieve, medical_qa, kg_query, drug_taboo]
