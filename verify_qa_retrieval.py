# -*- coding: utf-8 -*-
"""科普索引方式 A/B 对照：问题-only vs 问题+答案（同一批数据，隔离变量）。

背景：第一次小样本对比把「问题-only 灌 1000 条」跟「问题+答案 灌 48600 条」直接比，
     混淆了「索引方式」和「样本量/数据分布」两个变量，结论不可信。
本脚本用**同一批前 1000 条 huatuo_encyclopedia**，分别以两种 document 构造灌两个临时库：
     medical_qa_v2  -> document = 问题（问题-only）
     medical_qa_v3  -> document = 问题 + 答案（旧方式）
再对相同 query 对比检索，才是真正隔离变量的 A/B 测试。

结论判定：若 v2 检索回的问题比 v3 更相关 → 方向 A 有效，全量重灌；
          若 v3 仍更强或打平 → 方向 A 无收益，保留旧库不动。
"""
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb
from langchain_chroma import Chroma

from utils.llms import initialize_llm
from jsonl2chroma import _first_question, _first_answer, BATCH_SIZE, MAX_DOC_LEN

TEST_QUERIES = [
    "什么是糖尿病",
    "高血压饮食要注意什么",
    "感冒了吃什么药",
    "儿童发烧如何护理",
    "失眠有什么改善方法",
]

SAMPLE_LIMIT = 1000
SRC_FILE = "input/huatuo_encyclopedia_qa/train_datasets.jsonl"


def _load_records(limit):
    """读前 limit 条百科记录，返回 [(question, answer), ...]"""
    pairs = []
    with open(SRC_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or len(pairs) >= limit:
                continue
            rec = json.loads(line)
            q = _first_question(rec).strip()
            a = _first_answer(rec).strip()
            if q:
                pairs.append((q, a))
    return pairs


def _build_documents(pairs, mode):
    """mode='q' -> 问题-only；mode='qa' -> 问题+答案拼接"""
    docs = []
    for q, a in pairs:
        if mode == "qa":
            text = f"{q} {a}".strip()
        else:
            text = q
        docs.append(text[:MAX_DOC_LEN])
    return docs


def _import_docs(docs, llm_embedding, collection_name):
    """把 document 列表灌入指定 collection（metadata 只存 source，答案不参与本次对比）"""
    client = chromadb.PersistentClient(path="chromaDB")
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.get_or_create_collection(name=collection_name)
    ids = [str(uuid.uuid4()) for _ in docs]
    metas = [{"source": "huatuo_encyclopedia"} for _ in docs]
    for i in range(0, len(docs), BATCH_SIZE):
        b_docs = docs[i:i + BATCH_SIZE]
        b_meta = metas[i:i + BATCH_SIZE]
        b_ids = ids[i:i + BATCH_SIZE]
        b_vecs = llm_embedding.embed_documents(b_docs)
        col.add(embeddings=b_vecs, documents=b_docs, metadatas=b_meta, ids=b_ids)
        print(f"  [{collection_name}] {min(i + BATCH_SIZE, len(docs))}/{len(docs)}")
    return col.count()


def main():
    _, llm_embedding = initialize_llm("deepseek")

    pairs = _load_records(SAMPLE_LIMIT)
    print(f"读取 {len(pairs)} 条百科问答")

    print("=== 灌 medical_qa_v2（问题-only） ===")
    _import_docs(_build_documents(pairs, "q"), llm_embedding, "medical_qa_v2")
    print("=== 灌 medical_qa_v3（问题+答案） ===")
    _import_docs(_build_documents(pairs, "qa"), llm_embedding, "medical_qa_v3")

    v2 = Chroma(persist_directory="chromaDB", collection_name="medical_qa_v2", embedding_function=llm_embedding)
    v3 = Chroma(persist_directory="chromaDB", collection_name="medical_qa_v3", embedding_function=llm_embedding)

    for q in TEST_QUERIES:
        print(f"\n{'=' * 60}\nquery: {q}")
        print("--- v2 问题-only ---")
        for i, d in enumerate(v2.similarity_search(q, k=5), 1):
            print(f"  {i}. {d.page_content[:55]}")
        print("--- v3 问题+答案 ---")
        for i, d in enumerate(v3.similarity_search(q, k=5), 1):
            print(f"  {i}. {d.page_content[:55]}")


if __name__ == "__main__":
    main()
