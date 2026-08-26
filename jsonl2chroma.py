# -*- coding: utf-8 -*-
"""
jsonl2chroma.py
功能：把医疗问答 JSON / JSONL 数据导入 ChromaDB，绕过 PDF 解析流程。

背景：早期基于 PDF 的导入脚本只吃 PDF，而开源的医疗问答数据集
（Huatuo-26M-Lite、huatuo_encyclopedia_qa 等）都是 JSON 格式。
本脚本把这些问答对组织成「可检索文本 + 结构化 metadata」灌入 ChromaDB，
复用 utils.llms 里同一套 embedding（通义 text-embedding-v1），
保证与 ragAgent.py 检索端的向量维度一致。

用法：
    python jsonl2chroma.py --dry-run              # 只预览字段解析结果，不灌库
    python jsonl2chroma.py --source huatuo_lite   # 只灌某个数据源
    python jsonl2chroma.py --limit 500            # 覆盖每个源的抽样条数
    python jsonl2chroma.py                        # 按下方 SOURCES 配置灌入全部源
"""
import json
import uuid
import logging
import argparse
from collections import defaultdict

import chromadb

from utils.llms import initialize_llm
from utils.config import Config
from utils.label_cleaner import is_template_question, correct_label, INVALID_LABELS

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 配置区 ====================

# LLM 类型：与 utils/config.py 保持一致；deepseek / qwen 的 embedding 都是通义 text-embedding-v1
LLM_TYPE = "deepseek"

# ChromaDB 持久化路径：与 Config.CHROMADB_DIRECTORY 保持一致
CHROMADB_DIRECTORY = "chromaDB"

# 单批 embedding 条数（控制请求频率）
BATCH_SIZE = 25

# 正文最大长度（字符）：通义 text-embedding-v1 输入上限为 2048 token，中文约 1 字 ≈ 1 token，
# 留安全余量取 1500。超长正文截断，但完整答案仍保留在 metadata（不参与 embedding，不受此限制）。
MAX_DOC_LEN = 1500


# -------------------- 字段解析函数 --------------------
# 每个数据集字段不同，各写一个解析函数，统一返回 {"document": str, "metadata": dict}
# 说明：document 是灌库的正文文本（检索时被匹配的对象），metadata 是附带的结构化信息。

def parse_lite(record):
    """Huatuo26M-Lite：question / answer / label(科室) / score / related_diseases
    分诊场景：用 question（症状描述）作为正文，query 匹配到 question 后，
    通过 metadata 里的 label 直接拿到科室，answer 拿到医生建议。

    清洗接入（只影响 medical_triage）：
    ① 过滤模板题（「X 的辅助治疗/判断依据」等非分诊主诉），返回 None 跳过；
    ② 纠正「头疼/头痛 + 失眠」被误标为「心理科学」的高置信度误标（→神经科学）。"""
    question = record.get("question", "")
    if is_template_question(question):
        return None
    label = correct_label(question, record.get("label", ""))
    if label in INVALID_LABELS:  # 过滤「其他/未知科室」兜底标签，不灌入分诊库
        return None
    return {
        "document": question,
        "metadata": {
            "label": label,                            # 科室标签（分诊答案，经清洗）
            "answer": record.get("answer", ""),        # 医生回答
            "score": int(record.get("score", 0) or 0), # 回答质量评分
            "related_diseases": record.get("related_diseases", "") or "",
            "source": "huatuo_lite",
        },
    }


def parse_encyclopedia(record):
    """huatuo_encyclopedia_qa：questions[[]] / answers[]
    科普问答：正文存「问题 + 答案」拼接。A/B 对照验证（同批 1000 条）表明问题-only 无检索增益，
    答案里的关键词反而能提升语义匹配，故保留与旧库 medical_qa 一致的做法。"""
    q = _first_question(record)
    a = _first_answer(record)
    return {
        "document": f"{q} {a}".strip(),
        "metadata": {"answer": a, "source": "huatuo_encyclopedia"},
    }


def parse_knowledge_graph(record):
    """huatuo_knowledge_graph_qa：questions[] / answers[]（多答案，用分号连接）"""
    q = _first_question(record)
    ans = record.get("answers", [])
    a = "；".join([x for x in ans if x]) if isinstance(ans, list) else ""
    return {
        "document": f"{q} {a}".strip(),
        "metadata": {"answer": a, "source": "huatuo_knowledge_graph"},
    }


def parse_cmd(record):
    """Chinese-medical-dialogue：instruction / input / output / history（alpaca 格式）"""
    q = f"{record.get('instruction', '')} {record.get('input', '')}".strip()
    a = record.get("output", "") or ""
    return {
        "document": f"{q} {a}".strip(),
        "metadata": {"answer": a, "source": "chinese_medical_dialogue"},
    }


# -------------------- 数据源配置 --------------------
# collection 命名策略：
#   medical_triage —— 分诊（Lite，带科室 label）
#   medical_qa     —— 科普问答兜底（百科 / 知识图谱 / 好大夫）
SOURCES = [
    {
        "name": "huatuo_lite",
        "file": "input/Huatuo26M-Lite/format_data.jsonl",
        "collection": "medical_triage",
        "parse": parse_lite,
        "per_label_limit": Config.PER_LABEL_LIMIT,   # 均衡抽样：每科最多 3000 条（小科室全取、大科室均匀间隔）
    },
    {
        "name": "huatuo_encyclopedia",
        "file": "input/huatuo_encyclopedia_qa/train_datasets.jsonl",
        "collection": "medical_qa",
        "parse": parse_encyclopedia,
        "limit": 20000,
    },
    {
        "name": "huatuo_knowledge_graph",
        "file": "input/huatuo_knowledge_graph_qa/train_datasets.jsonl",
        "collection": "medical_qa",
        "parse": parse_knowledge_graph,
        "limit": 10000,
    },
    {
        "name": "chinese_medical_dialogue",
        "file": "input/Chinese-medical-dialogue/data/train_0001_of_0001.json",
        "collection": "medical_qa",
        "parse": parse_cmd,
        "limit": 20000,
    },
]


# -------------------- 工具函数 --------------------

def _first_question(record):
    """兼容 questions 是 [[\"..\"]] 或 [\"..\"] 两种结构，取第一个问题文本"""
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


def _read_records(source):
    """按文件类型读取记录：.jsonl 逐行、.json 整读数组，返回迭代器"""
    path = source["file"]
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            yield item
    else:
        raise ValueError(f"不支持的文件类型: {path}")


def _clean_metadata(metadata):
    """ChromaDB 要求 metadata 值为 str/int/float/bool，剔除 None 值"""
    return {k: v for k, v in metadata.items() if v is not None}


def _parse_valid(source, record):
    """解析 + 清洗 + 有效性判断：返回 parsed dict，无效（被过滤）则返回 None"""
    parsed = source["parse"](record)
    if parsed is None:
        return None
    if not parsed["document"].strip():
        return None
    return parsed


def _balanced_parsed(source, per_label_limit):
    """按 label 均衡抽样：第一遍统计每科有效条数，第二遍按均匀间隔产出 parsed 记录。

    解决 head 截断（limit=20000）导致的「小科室（急诊/心理/中医/生殖健康）被淹没」问题：
    每科取 min(有效条数, per_label_limit)，小科室全取，大科室均匀间隔取样。
    """
    # 第一遍：统计每科有效条数（含清洗过滤）
    per_label = {}
    for record in _read_records(source):
        parsed = _parse_valid(source, record)
        if parsed is None:
            continue
        label = parsed["metadata"].get("label", "") or "其他"
        per_label[label] = per_label.get(label, 0) + 1

    take = {lb: min(cnt, per_label_limit) for lb, cnt in per_label.items()}

    # 第二遍：均匀间隔取样（整数 floor 除法，无浮点精度问题）
    seen = {}
    for record in _read_records(source):
        parsed = _parse_valid(source, record)
        if parsed is None:
            continue
        label = parsed["metadata"].get("label", "") or "其他"
        k = seen.get(label, 0) + 1
        seen[label] = k
        need = take.get(label, 0)
        total = per_label[label]
        if need >= total:
            yield parsed  # 小科室全取
        elif (k * need) // total > ((k - 1) * need) // total:
            yield parsed  # 大科室均匀间隔取 need 条


# -------------------- 灌库逻辑 --------------------

def _import_source_legacy(source, llm_embedding, dry_run=False, override_limit=None, override_collection=None):
    """把单个数据源灌入 ChromaDB"""
    import time
    name = source["name"]
    collection_name = override_collection if override_collection else source["collection"]
    start_ts = time.monotonic()
    metric_context = {"source": name, "collection": collection_name, "path": "jsonl2chroma"}

    # 均衡抽样（分诊源专用）：每科上限 per_label_limit；用 --limit 覆盖时退回 head 截断
    per_label_limit = source.get("per_label_limit")
    balanced = bool(per_label_limit) and override_limit is None
    if balanced:
        records_iter = _balanced_parsed(source, per_label_limit)
        limit = None
        logger.info(f"[{name}] 开始处理 → collection={collection_name}, 均衡抽样(每科上限 {per_label_limit})")
    else:
        records_iter = _read_records(source)
        limit = override_limit if override_limit is not None else source.get("limit")
        logger.info(f"[{name}] 开始处理 → collection={collection_name}, head 抽样 limit={limit}")

    documents, metadatas, ids = [], [], []
    count = 0
    records_processed = 0  # B.2 fix: limit 按 records 数（不是 chunks 数）

    # B.8：确定性质量门禁（仅本路径；doc_sync 走 QualityGate 共享同一规则）
    from quality_gate import QualityGate
    gate = QualityGate(name)

    # 同步方案 A.4：所有 chunk 统一打 doc_id（路径+大小规范化），doc_sync.py 用它做"doc 级全删全重建"
    # 这里计算一次，循环里 setdefault；老库无 doc_id 用 --migrate 回填
    from pathlib import Path as _Path
    from doc_sync import compute_doc_id as _compute_doc_id
    _file_path = _Path(source["file"])
    _file_size = _file_path.stat().st_size
    _doc_id = _compute_doc_id(_file_path, _file_size)

    for item in records_iter:
        if limit is not None and records_processed >= limit:
            break
        parsed = item if balanced else _parse_valid(source, item)
        if parsed is None:
            continue
        document = parsed["document"].strip()
        # B.8：先过确定性质量门禁；拒绝项不进入 embedding/入库
        accepted, _reason = gate.check_record(parsed, index=records_processed)
        if not accepted:
            continue
        # B.2 切片：CHUNKING_ENABLED 时按句切分，否则按 MAX_DOC_LEN 硬截断
        if Config.CHUNKING_ENABLED and len(document) > Config.CHUNK_SIZE:
            from chunking import chunk_text
            text_chunks = chunk_text(
                document,
                chunk_size=Config.CHUNK_SIZE,
                overlap=Config.CHUNK_OVERLAP,
                min_size=Config.CHUNK_MIN_SIZE,
            )
        elif len(document) > MAX_DOC_LEN:
            from chunking import Chunk
            text_chunks = [Chunk(text=document[:MAX_DOC_LEN], sentence_count=1)]
        else:
            from chunking import Chunk
            text_chunks = [Chunk(text=document, sentence_count=1)]
        records_processed += 1  # B.2 fix: 成功解析后 +1

        for tc in text_chunks:
            documents.append(tc.text)
            meta = _clean_metadata(parsed["metadata"])
            meta["doc_id"] = _doc_id  # 同步方案 A.4：doc 级定位标识
            meta["sentence_count"] = tc.sentence_count  # B.2：每 chunk 句子数（审计 + 追溯）
            meta["embedding_model"] = Config.EMBEDDING_MODEL_ID  # B.4：标记产生该向量的 embedding 模型
            metadatas.append(meta)
            ids.append(str(uuid.uuid4()))
            count += 1
            gate.stats.accepted += 1
    logger.info(f"[{name}] 解析完成，有效记录 {count} 条；质量门禁 accepted={gate.stats.accepted} rejected={gate.stats.rejected}")
    gate.write_report(extra={"path": "jsonl2chroma"})

    if dry_run:
        logger.info(f"[{name}] [dry-run] 前 2 条样例：")
        for i in range(min(2, len(documents))):
            logger.info(f"  document: {documents[i][:120]}...")
            logger.info(f"  metadata: {metadatas[i]}")
        return count

    # 连接 ChromaDB，创建/获取 collection
    client = chromadb.PersistentClient(path=CHROMADB_DIRECTORY)
    collection = client.get_or_create_collection(name=collection_name)

    # 分批 embedding + 灌库
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_meta = metadatas[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]
        from metrics import time_embedding
        with time_embedding(
            Config.EMBEDDING_MODEL_ID,
            batch_docs,
            context={**metric_context, "batch_items": len(batch_docs)},
        ):
            batch_vecs = llm_embedding.embed_documents(batch_docs)
        collection.add(
            embeddings=batch_vecs,
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids,
        )
        logger.info(f"[{name}] 已灌入 {min(i + BATCH_SIZE, len(documents))}/{len(documents)}")

    duration = time.monotonic() - start_ts
    try:
        from metrics import record_sync_done, record_collection_size
        record_sync_done(
            name, "ok", duration, inserted=count, deleted=0,
            collection=collection_name, context=metric_context,
            deleted_scope="none",
        )
        record_collection_size(collection_name, collection.count(), context=metric_context)
    except Exception as e:
        logger.warning(f"记录 import metrics 失败（不影响主流程）: {e}")
    return count
# -------------------- 批量导入（B.9 聚合入口） --------------------

def _collect_source_candidates(source, override_limit=None, override_collection=None):
    """Prepare Chroma-ready chunks without embedding or database access."""
    from pathlib import Path as _Path
    from doc_sync import compute_doc_id as _compute_doc_id
    from quality_gate import QualityGate

    name = source["name"]
    collection_name = override_collection or source["collection"]
    per_label_limit = source.get("per_label_limit")
    balanced = bool(per_label_limit) and override_limit is None
    records_iter = (
        _balanced_parsed(source, per_label_limit) if balanced else _read_records(source)
    )
    limit = None if balanced else (
        override_limit if override_limit is not None else source.get("limit")
    )
    file_path = _Path(source["file"])
    doc_id = _compute_doc_id(file_path, file_path.stat().st_size)
    gate = QualityGate(name)
    candidates = []
    records_processed = 0
    for item in records_iter:
        if limit is not None and records_processed >= limit:
            break
        parsed = item if balanced else _parse_valid(source, item)
        if parsed is None:
            continue
        document = parsed["document"].strip()
        accepted, _ = gate.check_record(parsed, index=records_processed, doc_id=doc_id)
        if not accepted:
            gate.stats.rejected += 1
            continue
        if Config.CHUNKING_ENABLED and len(document) > Config.CHUNK_SIZE:
            from chunking import chunk_text
            chunks = chunk_text(document, chunk_size=Config.CHUNK_SIZE,
                                overlap=Config.CHUNK_OVERLAP, min_size=Config.CHUNK_MIN_SIZE)
        elif len(document) > MAX_DOC_LEN:
            from chunking import Chunk
            chunks = [Chunk(text=document[:MAX_DOC_LEN], sentence_count=1)]
        else:
            from chunking import Chunk
            chunks = [Chunk(text=document, sentence_count=1)]
        records_processed += 1
        for chunk_index, chunk in enumerate(chunks):
            metadata = _clean_metadata(parsed["metadata"])
            metadata["doc_id"] = doc_id
            metadata["sentence_count"] = chunk.sentence_count
            metadata["embedding_model"] = Config.EMBEDDING_MODEL_ID
            candidates.append({
                "document": chunk.text,
                "metadata": metadata,
                "source": name,
                "score": metadata.get("score", 0),
                "collection": collection_name,
                "id": uuid.uuid4().hex,
                "record_index": records_processed - 1,
                "chunk_index": chunk_index,
            })
            gate.stats.accepted += 1
    gate.write_report(extra={"path": "jsonl2chroma", "dedup_candidates": len(candidates)})
    return candidates


def _record_dedup_metrics(result, context):
    from metrics import record_dedup_done
    record_dedup_done(
        kept_by_source=result.stats.get("kept_by_source", {}),
        dropped_by_source=result.stats.get("dropped_by_source", {}),
        context=context,
    )


def _write_candidates(candidates, llm_embedding, dry_run, collection_name, source_name):
    """Embed and write one source/collection group; dry-run performs no writes."""
    if dry_run:
        return len(candidates)
    client = chromadb.PersistentClient(path=CHROMADB_DIRECTORY)
    collection = client.get_or_create_collection(name=collection_name)
    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i:i + BATCH_SIZE]
        docs = [r["document"] for r in batch]
        with __import__("metrics").time_embedding(
            Config.EMBEDDING_MODEL_ID, docs,
            context={"source": source_name, "collection": collection_name,
                     "path": "jsonl2chroma", "batch_items": len(docs)},
        ):
            vectors = llm_embedding.embed_documents(docs)
        collection.add(
            embeddings=vectors,
            documents=docs,
            metadatas=[r["metadata"] for r in batch],
            ids=[r["id"] for r in batch],
        )
    return len(candidates)


def import_sources(sources, llm_embedding=None, *, dry_run=False,
                   override_limit=None, override_collection=None):
    """Import selected sources, optionally applying collection-scoped B.9 dedup."""
    candidates_by_source = []
    for source in sources:
        candidates_by_source.append((source, _collect_source_candidates(
            source, override_limit, override_collection)))
    active = Config.DEDUP_ENABLED and len({s["name"] for s, _ in candidates_by_source}) >= 2
    all_candidates = [r for _, rows in candidates_by_source for r in rows]
    kept_by_collection = defaultdict(list)
    dropped_total = 0
    if active:
        from dedup import dedup_and_write
        result = dedup_and_write(
            all_candidates,
            report_path=Config.DEDUP_REPORT_PATH,
            dropped_path=Config.DEDUP_DROPPED_PATH,
            context={
                "path": "jsonl2chroma",
                "sources": [s["name"] for s, _ in candidates_by_source],
                "scope": "collection",
            },
        )
        _record_dedup_metrics(result, {"path": "jsonl2chroma", "scope": "collection"})
        for row in result.kept:
            kept_by_collection[row["collection"]].append(row)
        dropped_total = result.dropped_count
        try:
            from utils.audit import build_record, write_audit
            write_audit(build_record(
                event="dedup",
                thread_id="import::dedup",
                user_id="jsonl2chroma",
                comment=json.dumps({
                    "sources": [s["name"] for s, _ in candidates_by_source],
                    "input": result.input_count,
                    "kept": result.kept_count,
                    "dropped": result.dropped_count,
                    "report_path": Config.DEDUP_REPORT_PATH,
                }, ensure_ascii=False),
            ))
        except Exception as e:
            logger.warning(f"去重审计写入失败（不影响主流程）: {e}")
    else:
        for row in all_candidates:
            kept_by_collection[row["collection"]].append(row)

    total = 0
    for source, _ in candidates_by_source:
        name = source["name"]
        collection_name = override_collection or source["collection"]
        rows = [r for r in kept_by_collection[collection_name] if r["source"] == name]
        total += _write_candidates(rows, llm_embedding, dry_run, collection_name, name)
    logger.info("批量处理完成：input=%s kept=%s dropped=%s%s", len(all_candidates), total,
                dropped_total, "（dry-run，未灌库）" if dry_run else "")
    return total


def import_source(source, llm_embedding, dry_run=False, override_limit=None, override_collection=None):
    """Compatibility wrapper; single-source calls intentionally bypass cross-source dedup."""
    return _import_source_legacy(source, llm_embedding, dry_run, override_limit, override_collection)


def main():
    parser = argparse.ArgumentParser(description="把医疗问答 JSON 数据导入 ChromaDB")
    parser.add_argument("--dry-run", action="store_true", help="只预览解析结果，不灌库")
    parser.add_argument("--source", type=str, default=None, help="只处理指定数据源名")
    parser.add_argument("--limit", type=int, default=None, help="覆盖所有源的抽样条数")
    parser.add_argument("--collection", type=str, default=None, help="覆盖目标 collection 名（小样本验证灌到临时库用）")
    parser.add_argument("--clear", action="store_true", help="灌库前先清空目标 collection（重灌用）")
    args = parser.parse_args()

    # 初始化 embedding（只需 embedding，LLM 聊天模型暂不用）
    # dry-run 模式不灌库，无需 embedding，避免无谓的 API key 校验
    if args.dry_run:
        llm_embedding = None
    else:
        _, llm_embedding = initialize_llm(LLM_TYPE)

    sources = [s for s in SOURCES if args.source is None or s["name"] == args.source]
    if not sources:
        logger.error(f"未找到数据源: {args.source}，可选: {[s['name'] for s in SOURCES]}")
        return

    # 灌库前清空涉及的 collection（去重，只清一次；配合 --collection 可清空临时库）
    if args.clear and not args.dry_run:
        client = chromadb.PersistentClient(path=CHROMADB_DIRECTORY)
        target_names = {args.collection} if args.collection else {s["collection"] for s in sources}
        for cname in target_names:
            try:
                client.delete_collection(cname)
                logger.info(f"已清空 collection「{cname}」")
            except Exception:
                logger.info(f"collection「{cname}」不存在，跳过清空")

    total = 0
    if Config.DEDUP_ENABLED and len({s["name"] for s in sources}) >= 2:
        total = import_sources(
            sources,
            llm_embedding,
            dry_run=args.dry_run,
            override_limit=args.limit,
            override_collection=args.collection,
        )
    else:
        for source in sources:
            total += import_source(source, llm_embedding, dry_run=args.dry_run,
                                   override_limit=args.limit, override_collection=args.collection)

    logger.info(f"全部处理完成，共 {total} 条记录" + ("（dry-run，未灌库）" if args.dry_run else ""))


if __name__ == "__main__":
    main()
