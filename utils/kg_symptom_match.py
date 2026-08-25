# -*- coding: utf-8 -*-
"""症状模糊匹配：口语症状名 → 图中规范症状节点（embedding 最近邻 + 阈值）。

解决「用户口语 ≠ 图节点规范名」的 gap：胸口闷→胸闷、拉肚子→腹泻、睡不着→失眠。

设计要点：
  - 精确匹配优先（resolve_entity 先查图，miss 才走这里），故不污染疾病名查询；
  - 用通义 text-embedding-v1 把全图症状节点向量化，查询时余弦最近邻 + 阈值；
  - 阈值是「语义等价」的下界：低于阈值视为「图里没有对应症状」，返回 None
    （宁可漏、不可错——分诊场景误配比漏配更危险）。
"""
import os
import pickle
import threading
from pathlib import Path

import numpy as np

from utils.config import Config
from utils.llms import initialize_llm
from utils import kg_schema as S

# 相似度阈值（实测：正确配对 0.406~0.886、错误配对 0.002~0.624，取等价下界）
SYMPTOM_SIM_THRESHOLD = 0.6

# 症状向量索引磁盘缓存：图症状节点集合不变时直接复用，避免每次启动重建
# （全图约 1.4 万个症状 × chunk=25 ≈ 556 次 API 请求，首建约 3~4 分钟）
_INDEX_CACHE_PATH = Config.KG_SYMPTOM_INDEX_PATH

_embedding = None      # OpenAIEmbeddings 单例
_index = None          # (症状名列表, L2 归一化向量矩阵) 单例
_lock = threading.Lock()


def _get_embedding():
    """懒加载 embedding 实例（通义 text-embedding-v1）。"""
    global _embedding
    if _embedding is None:
        _, _embedding = initialize_llm("deepseek")
    return _embedding


def _collect_symptom_nodes(G):
    """收集图中所有症状节点（type=symptom），排序保证稳定顺序。"""
    return sorted(n for n in G.nodes if G.nodes[n].get("type") == S.NODE_SYMPTOM)


def _build_index(G, embedding):
    """把全图症状节点向量化，返回 (names, mat[n, dim] 已 L2 归一化)。"""
    names = _collect_symptom_nodes(G)
    # 通义 text-embedding-v1 单次 batch 上限 25，必须显式分小批（默认 chunk_size=1000 会 400）
    vecs = embedding.embed_documents(names, chunk_size=25)
    mat = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return names, mat


def _source_sha256(G):
    """读取图缓存携带的源版本；旧图没有版本时返回 None。"""
    return getattr(G, "graph", {}).get("source_sha256")


def _load_or_build_index(G, embedding):
    """按源版本和症状节点集合校验磁盘缓存。"""
    names = _collect_symptom_nodes(G)
    source_sha256 = _source_sha256(G)
    if os.path.exists(_INDEX_CACHE_PATH):
        try:
            with open(_INDEX_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict):
                cached_names = cached.get("names")
                mat = cached.get("matrix")
                if cached.get("source_sha256") == source_sha256 and cached_names == names:
                    return names, mat
            elif isinstance(cached, tuple) and len(cached) == 2:
                # 旧 tuple 无法证明来源版本，只兼容无版本旧图。
                cached_names, mat = cached
                if source_sha256 is None and cached_names == names:
                    return names, mat
        except Exception:
            pass  # 缓存损坏或不匹配 → 重建
    names, mat = _build_index(G, embedding)
    try:
        Path(_INDEX_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(_INDEX_CACHE_PATH, "wb") as f:
            pickle.dump({
                "source_sha256": source_sha256,
                "names": names,
                "matrix": mat,
            }, f)
    except Exception:
        pass  # 缓存写失败不影响功能，下次重建即可
    return names, mat


def invalidate_symptom_index():
    """清空进程内症状索引；磁盘索引由版本校验决定是否懒重建。"""
    global _index
    with _lock:
        _index = None


def get_symptom_index(G):
    """懒加载单例：首次调用时构建（或从磁盘缓存加载）全图症状向量索引。"""
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                _index = _load_or_build_index(G, _get_embedding())
    return _index


def fuzzy_match_symptom(G, name, threshold=SYMPTOM_SIM_THRESHOLD):
    """口语症状名 → 规范节点名；相似度不足阈值返回 None（视为未识别）。"""
    names, mat = get_symptom_index(G)
    qv = np.asarray(_get_embedding().embed_query(name), dtype=np.float32)
    qn = np.linalg.norm(qv)
    if qn == 0:
        return None
    qv = qv / qn
    sims = mat @ qv            # 余弦相似度
    idx = int(sims.argmax())
    return names[idx] if sims[idx] >= threshold else None
