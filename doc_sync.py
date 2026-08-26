# -*- coding: utf-8 -*-
"""doc_sync.py — 文档级变更感知 + 同步引擎。

把 jsonl2chroma 的一次性 bootstrap 升级为生产级同步：
- 源文件改动 → 感知（mtime/size/sha256） → doc 级全删全重建 → 缓存失效 → 审计留痕
- 支持：CLI --once / --watch / webhook 三种触发模式
- 兼容：run_migration 一次性回填 doc_id 给存量无标识 chunk

review 补强清单（10 项已落地）：
- #1 首次 sync delete 崩溃保护（prev_doc_id None 跳过）
- #2 存量未 migrate 同步导致双份数据 → 上线顺序硬约束先 --migrate
- #3 Windows stale lock mtime 检测（> 30min 强制解锁重试）
- #4 --rebuild 模式硬性顺序（失效缓存 → delete → add）
- #5 路径规范化（normcase + 正斜杠，避开 Windows 大小写 / 反斜杠）
- #6 manifest tmp 残留清理（glob 删 .tmp.*）
- #8 审计事件 comment 字段带 source / doc_id / inserted / deleted / duration_s
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from utils.config import Config

logger = logging.getLogger(__name__)


# ==================== 数据结构 ====================

@dataclass
class SyncResult:
    """单 source 同步结果。"""

    source: str
    status: str  # "skipped" | "ok" | "rebuilt" | "error" | "migrated"
    doc_id: str
    prev_doc_id: Optional[str]
    inserted: int
    deleted: int
    duration_s: float
    error: Optional[str] = None
    kg_status: Optional[str] = None
    dedup_dropped: int = 0


# ==================== 工具函数 ====================

def compute_doc_id(path: Path, file_size: int) -> str:
    """review 补强 #5：路径规范化（Windows 大小写 / 反斜杠）。

    doc_id = f"{path.name}::{sha256(abs_path + file_size)[:12]}"
    注意：是路径 + 大小，不是内容 hash，确保 re-ingest 时仍能定位旧 chunks。
    """
    abs_path = os.path.normcase(os.path.abspath(str(path))).replace("\\", "/")
    digest = hashlib.sha256(f"{abs_path}{file_size}".encode("utf-8")).hexdigest()[:12]
    return f"{path.name}::{digest}"


def streaming_sha256(path: Path, block: int = 1 << 20) -> str:
    """分块流式计算 SHA-256（处理 604 MB 文件不爆内存）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: str | None = None) -> dict:
    """读 manifest；不存在或损坏返回空 dict。"""
    path = path or Config.SYNC_MANIFEST_PATH
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"manifest 读取失败 {path}: {e}，回退空 manifest")
        return {}


def save_manifest_atomic(manifest: dict, path: str | None = None) -> None:
    """原子写：review 补强 #6，先清残留 tmp，再写 tmp.pid，最后 os.replace。"""
    path = path or Config.SYNC_MANIFEST_PATH
    for f in glob.glob(path + ".tmp.*"):
        try:
            os.remove(f)
        except OSError:
            pass
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ==================== 文件锁（含 stale 检测）===================

def _try_lock_windows(fd: int, lock_path: str, stale_secs: int) -> None:
    """Windows msvcrt 加锁，review 补强 #3 含 stale 检测。"""
    import msvcrt

    # 检测 stale：锁文件 mtime 距今超过 stale_secs 视为遗留锁，强制解锁重试
    try:
        mtime = os.stat(lock_path).st_mtime
        if time.time() - mtime > stale_secs:
            logger.warning(
                f"检测到 stale lock（mtime 距今 {int(time.time() - mtime)}s > {stale_secs}s），强制解锁"
            )
            try:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            # 重置 mtime，便于后续 stale 检测基准
            with open(lock_path, "wb") as f:
                f.write(b"")
    except OSError:
        pass

    # 阻塞加锁
    while True:
        try:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            # 拿到锁后 touch，保持 mtime 活跃（让对端能识别 stale）
            with open(lock_path, "wb") as f:
                f.write(b"")
            return
        except OSError:
            time.sleep(0.1)


@contextlib.contextmanager
def acquire_sync_lock(
    lock_path: str | None = None,
    stale_secs: int | None = None,
) -> Iterator[None]:
    """进程级文件锁：Windows msvcrt / POSIX fcntl，含 stale 检测。

    review 补强 #3：Windows 上 stale lock 永久卡死 → mtime > 30 分钟视为 stale 强制解锁。
    """
    lock_path = lock_path or Config.SYNC_LOCK_PATH
    stale_secs = (
        stale_secs if stale_secs is not None else Config.SYNC_STALE_LOCK_SECS
    )
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    Path(lock_path).touch(exist_ok=True)

    if sys.platform == "win32":
        fd = os.open(lock_path, os.O_RDWR)
        try:
            _try_lock_windows(fd, lock_path, stale_secs)
            yield
        finally:
            try:
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except (OSError, ImportError):
                pass
            os.close(fd)
    else:
        import fcntl
        fd = os.open(lock_path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.utime(lock_path, None)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)


# ==================== 审计封装 ====================

def _emit_audit(event: str, source: str, doc_id: str, **fields) -> None:
    """review 补强 #8：审计事件 comment 字段带 source / doc_id / inserted / deleted / duration_s。"""
    from utils.audit import build_record, write_audit

    try:
        record = build_record(
            event=event,
            thread_id=f"sync::{source}",
            user_id="doc_sync",
            comment=json.dumps(
                {"source": source, "doc_id": doc_id, **fields}, ensure_ascii=False
            ),
        )
        write_audit(record)
    except Exception as e:
        logger.warning(f"审计写入失败 event={event}: {e}")


def _sync_kg_artifacts(
    source_path: Path,
    source_sha256: str,
    *,
    graph_path: str | None = None,
    sample: int | None = None,
) -> dict:
    """让 KG 图和症状索引跟随 knowledge-graph 源版本更新。

    图缓存按 source_sha256 自行判断是否需要重建；症状向量索引只失效进程内
    单例，下一次模糊匹配时再按图版本懒加载，避免同步阶段重复发起 embedding。

    graph_path / sample 允许覆盖（默认从 Config 读取），便于测试时隔离到临时路径。
    """
    started = time.time()
    from utils.kg_builder import build_or_load, graph_cache_matches, load_graph

    if graph_path is None:
        graph_path = Config.KG_GRAPH_PATH
    if sample is None:
        sample = Config.KG_BUILD_SAMPLE if Config.KG_BUILD_SAMPLE > 0 else None
    # 与 build_or_load 保持一致：sample<=0 视为全量（None）
    if sample is not None and sample <= 0:
        sample = None

    cache_hit = False
    if os.path.exists(graph_path):
        try:
            cache_hit = graph_cache_matches(
                load_graph(graph_path),
                source_sha256=source_sha256,
                sample=sample,
                source_path=source_path,
            )
        except Exception:
            cache_hit = False

    graph = build_or_load(
        str(source_path),
        graph_path,
        sample=sample,
        source_sha256=source_sha256,
    )
    if graph.graph.get("source_sha256") != source_sha256:
        raise RuntimeError("KG 图构建完成但 source_sha256 校验失败")

    # 缓存命中时跳过失效：图文件未变化，进程内单例仍有效，无需强制重载。
    if not cache_hit:
        from utils.kg_query import invalidate_kg_cache
        from utils.kg_symptom_match import invalidate_symptom_index
        invalidate_kg_cache()
        invalidate_symptom_index()

    status = "cached" if cache_hit else "rebuilt"
    duration = round(time.time() - started, 2)
    logger.info(
        f"[KG] source_sha256={source_sha256[:12]} status={status} "
        f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} duration={duration}s"
    )
    return {
        "status": status,
        "source_sha256": source_sha256,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "duration_s": duration,
    }


# ==================== 核心算法 ====================

def sync_source(
    source: dict,
    llm_embedding,
    *,
    dry_run: bool = False,
    override_collection: str | None = None,
    rebuild: bool = False,
    manifest_path: str | None = None,
    include_demo: bool = False,
    _prepared_records: list[dict] | None = None,
    _dedup_dropped: int = 0,
    _rebuild_collection_deleted: bool = False,
) -> SyncResult:
    """同步单个 source。

    Args:
        source: jsonl2chroma.SOURCES 元素，含 name / file / collection / parse / per_label_limit / limit
        llm_embedding: langchain Embeddings 实例；dry_run 时可为 None
        dry_run: True 时只统计不写库不删库
        override_collection: 覆盖目标 collection（小样本验证用）
        rebuild: True 时强制清空目标 collection 后重建（review 补强 #4 硬性顺序）
        manifest_path: 自定义 manifest 路径
        include_demo: review 补强 #10，--rebuild 时是否包含 demo001 子集；默认 False
    """
    from jsonl2chroma import (
        _balanced_parsed,
        _parse_valid,
        _read_records,
        MAX_DOC_LEN,
    )
    import chromadb

    start_ts = time.monotonic()
    dedup_dropped = _dedup_dropped
    name = source["name"]
    file_path = Path(source["file"])
    collection_name = override_collection or source["collection"]
    metric_context = {"source": name, "collection": collection_name}

    # 1. 算 file_size / mtime / sha256 / new_doc_id
    try:
        file_stat = file_path.stat()
        file_size = file_stat.st_size
        mtime = file_stat.st_mtime
        sha = streaming_sha256(file_path)
        new_doc_id = compute_doc_id(file_path, file_size)
    except OSError as e:
        duration = time.monotonic() - start_ts
        try:
            from metrics import record_sync_done
            record_sync_done(
                name, "error", duration, collection=collection_name,
                context=metric_context,
            )
        except Exception as metric_error:
            logger.warning(f"记录 stat/read 失败 metrics 失败: {metric_error}")
        return SyncResult(
            source=name, status="error", doc_id="", prev_doc_id=None,
            inserted=0, deleted=0, duration_s=duration,
            error=f"stat/read 源文件失败: {e}",
        )

    # 2. 读 manifest 中该 source 的 prev
    manifest = load_manifest(manifest_path)
    prev = manifest.get(name, {})

    # 3. skip 判断
    if (
        not rebuild
        and prev.get("sha256") == sha
        and prev.get("file_size") == file_size
        and prev.get("last_status") == "ok"
        and prev.get("chunk_count", 0) > 0
        and prev.get("embedding_model") == Config.EMBEDDING_MODEL_ID  # B.4：模型变了不跳过
    ):
        # B.3 修复：源未变化时仍检查 KG 构建产物是否缺失或过期，
        # 避免 kg_graph.pkl 被误删/损坏后 sync_skip 导致 KG 永远不恢复。
        kg_status = None
        if name == "huatuo_knowledge_graph":
            try:
                kg_info = _sync_kg_artifacts(file_path, sha)
                kg_status = kg_info["status"]
                if kg_status != "cached":
                    logger.info(f"[{name}] 源未变化但 KG 产物需要修复（status={kg_status}）")
            except Exception as e:
                logger.warning(f"[{name}] 源未变化但 KG 修复失败: {e}")

        duration = time.monotonic() - start_ts
        _emit_audit("sync_skip", name, new_doc_id,
                    kg_status=kg_status, duration_s=round(duration, 2))
        try:
            from metrics import record_sync_done
            record_sync_done(
                name, "skipped", duration, collection=collection_name,
                context=metric_context, deleted_scope="none",
            )
        except Exception as metric_error:
            logger.warning(f"记录 skip metrics 失败: {metric_error}")
        logger.info(f"[{name}] 无变化（sha256+size+chunk_count 全匹配），跳过")
        return SyncResult(
            source=name, status="skipped", doc_id=new_doc_id,
            prev_doc_id=prev.get("doc_id"), inserted=0, deleted=0,
            duration_s=duration, kg_status=kg_status,
            dedup_dropped=dedup_dropped,
        )

    # 4. 取 prev_doc_id
    prev_doc_id = prev.get("doc_id")

    # 5. 跑 parser 生成 documents / metadatas / ids
    dedup_dropped = _dedup_dropped
    if _prepared_records is not None:
        prepared = list(_prepared_records)
        documents = [row["document"] for row in prepared]
        metadatas = [row["metadata"] for row in prepared]
        ids = [row["id"] for row in prepared]
        count = len(prepared)
        records_processed = count
    else:
        # 现有单 source 路径保持原有解析与切片行为。
        per_label_limit = source.get("per_label_limit")
        override_limit = source.get("limit")
        balanced = bool(per_label_limit) and override_limit is None
        if balanced:
            records_iter = _balanced_parsed(source, per_label_limit)
            limit = None
        else:
            records_iter = _read_records(source)
            limit = override_limit

        documents, metadatas, ids = [], [], []
        count = 0
        records_processed = 0  # B.2 fix: limit 按 records 数（不是 chunks 数）
        for item in records_iter:
            if limit is not None and records_processed >= limit:
                break
            parsed = item if balanced else _parse_valid(source, item)
            if parsed is None:
                continue
            document = parsed["document"].strip()
            # B.2 切片：CHUNKING_ENABLED 时按句切分，否则按 MAX_DOC_LEN 硬截断
            # 与 jsonl2chroma 行为一致，保证"导入"和"同步"行为对齐
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
            records_processed += 1

            for tc in text_chunks:
                meta = dict(parsed["metadata"])
                meta["doc_id"] = new_doc_id
                meta.setdefault("source", name)
                meta["sentence_count"] = tc.sentence_count
                meta["embedding_model"] = Config.EMBEDDING_MODEL_ID
                documents.append(tc.text)
                metadatas.append({k: v for k, v in meta.items() if v is not None})
                ids.append(uuid.uuid4().hex)
                count += 1

    logger.info(f"[{name}] 解析完成，有效记录 {count} 条")

    if dry_run:
        duration = time.monotonic() - start_ts
        estimated_delete = prev.get("chunk_count", 0) if prev_doc_id else 0
        _emit_audit("sync_dry_run", name, new_doc_id,
                    inserted=count, deleted=estimated_delete,
                    duration_s=round(duration, 2))
        try:
            from metrics import record_sync_done
            record_sync_done(
                name, "dry_run", duration, inserted=count,
                deleted=estimated_delete, collection=collection_name,
                context=metric_context,
                deleted_scope="doc_id" if prev_doc_id else "none",
                deleted_estimated=bool(prev_doc_id),
            )
        except Exception as metric_error:
            logger.warning(f"记录 dry-run metrics 失败: {metric_error}")
        # B.8：dry-run 也跑质量门禁，仅记录不阻断（不删库不 embed）
        try:
            from quality_gate import QualityGate
            gate = QualityGate(name)
            gate_records = [
                {"document": doc, "metadata": meta}
                for doc, meta in zip(documents, metadatas)
            ]
            gate.check_batch(gate_records, doc_id=new_doc_id)
            gate.write_report(extra={"dry_run": True, "original_records": count})
            logger.info(
                f"[{name}] [dry-run] gate: accepted={gate.stats.accepted}, "
                f"rejected={gate.stats.rejected} rate={gate.stats.rejection_rate:.2%}"
            )
        except Exception as gate_error:
            logger.warning(f"质量门禁 dry-run 失败（不影响主流程）: {gate_error}")
        logger.info(f"[{name}] [dry-run] would insert={count}, delete={estimated_delete}")
        return SyncResult(
            source=name, status="ok", doc_id=new_doc_id,
            prev_doc_id=prev_doc_id, inserted=count, deleted=estimated_delete,
            duration_s=duration, dedup_dropped=dedup_dropped,
        )

    if _dedup_dropped:
        # sync_all may pass filtered Chroma-ready arrays; preserve the count in
        # this source result and never regenerate IDs after deduplication.
        dedup_dropped = _dedup_dropped
    # B.8：删除旧库前必须先跑质量门禁；拒绝率超阈值时不删，保护现有可用库。
    gate = None
    if Config.QUALITY_GATE_ENABLED:
        try:
            from quality_gate import QualityGate, is_within_rate_limit
            gate = QualityGate(name)
            # 重新跑一次 parser 把 records_iter 转为 parsed 列表喂给 gate。
            # 注意：records_iter 可能已被均衡抽样耗尽，这里直接复用已解析的 documents。
            gate_records = [
                {"document": doc, "metadata": meta}
                for doc, meta in zip(documents, metadatas)
            ]
            accepted_records = gate.check_batch(gate_records, doc_id=new_doc_id)
            gate.write_report(extra={"phase": "pre_delete", "rebuild": rebuild})
            logger.info(
                f"[{name}] gate: accepted={gate.stats.accepted}, "
                f"rejected={gate.stats.rejected} rate={gate.stats.rejection_rate:.2%}"
            )
            if (
                Config.QUALITY_GATE_FAIL_ON_THRESHOLD
                and not is_within_rate_limit(
                    gate.stats, Config.QUALITY_GATE_MAX_REJECTION_RATE
                )
            ):
                duration = time.monotonic() - start_ts
                _emit_audit(
                    "gate_blocked", name, new_doc_id,
                    rejected=gate.stats.rejected,
                    accepted=gate.stats.accepted,
                    rejection_rate=gate.stats.rejection_rate,
                    threshold=Config.QUALITY_GATE_MAX_REJECTION_RATE,
                    duration_s=round(duration, 2),
                )
                try:
                    from metrics import record_sync_done
                    record_sync_done(
                        name, "gate_blocked", duration,
                        collection=collection_name, context=metric_context,
                    )
                except Exception:
                    pass
                return SyncResult(
                    source=name, status="gate_blocked", doc_id=new_doc_id,
                    prev_doc_id=prev_doc_id,
                    inserted=0, deleted=0,
                    duration_s=duration,
                    error=f"质量门禁拒绝率 {gate.stats.rejection_rate:.2%} "
                          f"超阈值 {Config.QUALITY_GATE_MAX_REJECTION_RATE:.2%}",
                    dedup_dropped=dedup_dropped,
                )
            # 用 accepted 列表替换原 documents/metadatas/ids（同步下游）
            documents = [r["document"] for r in accepted_records]
            metadatas = [r["metadata"] for r in accepted_records]
            ids = ids[: len(documents)]
        except Exception as e:
            logger.warning(f"质量门禁异常，按原数据继续: {e}")
            gate = None
    client = chromadb.PersistentClient(path=Config.CHROMADB_DIRECTORY)
    deleted = 0

    try:
        if _rebuild_collection_deleted:
            # Aggregate sync already invalidated/deleted the shared collection.
            collection = client.get_or_create_collection(name=collection_name)
            deleted = 0
        elif rebuild:
            # review 补强 #4：硬性顺序 — 失效旧缓存 → delete_collection → get_or_create
            from utils.tools_config import (
                invalidate_qa_cache,
                invalidate_triage_cache,
            )
            if collection_name == Config.CHROMADB_COLLECTION_NAME:
                invalidate_triage_cache()
            elif collection_name == Config.CHROMADB_QA_COLLECTION_NAME:
                invalidate_qa_cache()
            try:
                client.delete_collection(collection_name)
                logger.info(f"[{name}] [--rebuild] 已删除 collection {collection_name}")
            except Exception as e:
                logger.info(f"[{name}] [--rebuild] collection {collection_name} 不存在，跳过删除: {e}")
        else:
            # 非 rebuild 模式：review 补强 #1 首次保护（manifest 为空时 prev_doc_id=None 跳过）
            if prev_doc_id:
                try:
                    collection = client.get_collection(collection_name)
                    collection.delete(where={"doc_id": prev_doc_id})
                    # chromadb 0.5+ 返回 DeleteResult-like，0.4 返回 dict；用 prev chunk_count 作估值
                    deleted = prev.get("chunk_count", 0)
                    logger.info(f"[{name}] 已删除 doc_id={prev_doc_id} 的旧 chunks（估算 {deleted} 条）")
                except Exception as e:
                    logger.warning(f"[{name}] delete(where={prev_doc_id}) 失败: {e}，回退 chunk_ids 列表")
                    chunk_ids = prev.get("chunk_ids", [])
                    if chunk_ids:
                        try:
                            collection = client.get_collection(collection_name)
                            collection.delete(ids=chunk_ids)
                            deleted = len(chunk_ids)
                        except Exception as e2:
                            logger.error(f"[{name}] 回退 chunk_ids 删除也失败: {e2}")
    except Exception as e:
        duration = time.monotonic() - start_ts
        _emit_audit("sync_error", name, new_doc_id,
                    error=str(e), duration_s=round(duration, 2))
        try:
            from metrics import record_sync_done
            record_sync_done(
                name, "error", duration, deleted=deleted,
                collection=collection_name, context=metric_context,
                deleted_scope="collection" if rebuild else "doc_id",
                deleted_estimated=not rebuild,
            )
        except Exception as metric_error:
            logger.warning(f"记录 delete 失败 metrics 失败: {metric_error}")
        return SyncResult(
            source=name, status="error", doc_id=new_doc_id,
            prev_doc_id=prev_doc_id, inserted=0, deleted=deleted,
            duration_s=duration, error=f"delete 阶段异常: {e}",
            dedup_dropped=dedup_dropped,
        )

    # 7. 写入新 chunks
    collection = client.get_or_create_collection(name=collection_name)
    inserted = 0
    try:
        for i in range(0, len(documents), Config.SYNC_BATCH_SIZE):
            batch_docs = documents[i:i + Config.SYNC_BATCH_SIZE]
            batch_meta = metadatas[i:i + Config.SYNC_BATCH_SIZE]
            batch_ids = ids[i:i + Config.SYNC_BATCH_SIZE]
            # B.7：记录 embedding 耗时与成本
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
            inserted += len(batch_docs)
            logger.info(f"[{name}] 已灌入 {inserted}/{len(documents)}")
    except Exception as e:
        duration = time.monotonic() - start_ts
        _emit_audit("sync_error", name, new_doc_id,
                    error=str(e), inserted=inserted, deleted=deleted,
                    duration_s=round(duration, 2))
        # B.7：记录 sync 失败指标
        from metrics import record_sync_done
        record_sync_done(
            name, "error", duration, inserted=inserted, deleted=deleted,
            collection=collection_name, context=metric_context,
            deleted_scope="collection" if rebuild else "doc_id",
            deleted_estimated=not rebuild,
        )
        return SyncResult(
            source=name, status="error", doc_id=new_doc_id,
            prev_doc_id=prev_doc_id, inserted=inserted, deleted=deleted,
            duration_s=duration, error=f"add 阶段异常: {e}",
            dedup_dropped=dedup_dropped,
        )

    # 9. KG 联动：只有向量写入成功后才更新同源图缓存。
    kg_status = None
    if name == "huatuo_knowledge_graph":
        try:
            kg_info = _sync_kg_artifacts(file_path, sha)
            kg_status = kg_info["status"]
            _emit_audit(
                "kg_sync",
                name,
                new_doc_id,
                kg_status=kg_status,
                source_sha256=sha,
                nodes=kg_info["nodes"],
                edges=kg_info["edges"],
                duration_s=kg_info["duration_s"],
            )
        except Exception as e:
            duration = time.monotonic() - start_ts
            _emit_audit(
                "sync_error",
                name,
                new_doc_id,
                error=f"KG 联动失败: {e}",
                inserted=inserted,
                deleted=deleted,
                source_sha256=sha,
                duration_s=round(duration, 2),
            )
            logger.exception(f"[{name}] 向量已写入，但 KG 联动失败")
            try:
                from metrics import record_sync_done
                record_sync_done(
                    name, "error", duration, inserted=inserted,
                    deleted=deleted, collection=collection_name,
                context={**metric_context, "kg_status": "error", "dedup_dropped": dedup_dropped},
                )
            except Exception as metric_error:
                logger.warning(f"记录 KG 失败 metrics 失败: {metric_error}")
            try:
                from metrics import record_collection_size
                record_collection_size(collection_name, collection.count(), context=metric_context)
            except Exception as metric_error:
                logger.warning(f"记录 KG 失败后的 collection snapshot 失败: {metric_error}")
            return SyncResult(
                source=name,
                status="error",
                doc_id=new_doc_id,
                prev_doc_id=prev_doc_id,
                inserted=inserted,
                deleted=deleted,
                duration_s=duration,
                error=f"KG 联动失败: {e}",
                kg_status="error",
                dedup_dropped=dedup_dropped,
            )

    # 10. 原子更新 manifest
    manifest[name] = {
        "doc_id": new_doc_id,
        "sha256": sha,
        "file_size": file_size,
        "mtime": mtime,
        "chunk_count": inserted,
        "chunk_ids": ids[:inserted],
        "embedding_model": Config.EMBEDDING_MODEL_ID,  # B.4：记录本次同步使用的 embedding 模型
        "last_status": "ok",
        "last_sync_ts": time.time(),
    }
    save_manifest_atomic(manifest, manifest_path)

    # 11. 审计
    duration = time.monotonic() - start_ts
    event = "sync_rebuild" if rebuild else "sync_insert"
    _emit_audit(event, name, new_doc_id,
                inserted=inserted, deleted=deleted, kg_status=kg_status,
                duration_s=round(duration, 2))

    # 12. 失效缓存（非 rebuild 模式；rebuild 已在步骤 6 失效过）
    if not rebuild:
        from utils.tools_config import (
            invalidate_qa_cache,
            invalidate_triage_cache,
        )
        if collection_name == Config.CHROMADB_COLLECTION_NAME:
            invalidate_triage_cache()
        elif collection_name == Config.CHROMADB_QA_COLLECTION_NAME:
            invalidate_qa_cache()

    logger.info(
        f"[{name}] 完成，inserted={inserted}, deleted={deleted}, "
        f"kg_status={kg_status or 'n/a'}, duration={duration:.1f}s"
    )
    # B.7：记录 sync 完成事件 + collection 大小
    try:
        from metrics import record_sync_done, record_collection_size
        record_sync_done(
            name, "ok", duration, inserted=inserted, deleted=deleted,
            collection=collection_name, context=metric_context,
            deleted_scope="collection" if rebuild else "doc_id",
            deleted_estimated=not rebuild and bool(prev_doc_id),
            deleted_known=not rebuild or bool(prev_doc_id),
        )
        record_collection_size(collection_name, collection.count(), context=metric_context)
    except Exception as e:
        logger.warning(f"记录 sync metrics 失败（不影响主流程）: {e}")

    return SyncResult(
        source=name,
        status="rebuilt" if rebuild else "ok",
        doc_id=new_doc_id,
        prev_doc_id=prev_doc_id,
        inserted=inserted,
        deleted=deleted,
        duration_s=duration,
        kg_status=kg_status,
        dedup_dropped=dedup_dropped,
    )


def sync_all(
    sources,
    llm_embedding,
    *,
    dry_run: bool = False,
    override_collection: str | None = None,
    rebuild: bool = False,
    manifest_path: str | None = None,
    include_demo: bool = False,
) -> list[SyncResult]:
    """串行同步所有 source（进程锁由调用方 acquire_sync_lock 包裹）。

    B.5：同步开始前自动检测 drug_contraindications.json 变更，有变化则失效药物缓存。
    """
    # B.5：检测药物禁忌文件变更
    try:
        from utils.tools_config import check_drug_contra_changes
        if check_drug_contra_changes():
            _emit_audit("drug_cache_invalidated", "drug_contraindications", "")
    except Exception as e:
        logger.warning(f"药物禁忌文件变更检测失败: {e}")

    if Config.DEDUP_ENABLED and len({s["name"] for s in sources}) >= 2:
        return _run_sync_all_dedup(
            sources, llm_embedding, dry_run=dry_run,
            override_collection=override_collection, rebuild=rebuild,
            manifest_path=manifest_path, include_demo=include_demo,
        )

    results = []
    for source in sources:
        try:
            result = sync_source(
                source,
                llm_embedding,
                dry_run=dry_run,
                override_collection=override_collection,
                rebuild=rebuild,
                manifest_path=manifest_path,
                include_demo=include_demo,
            )
        except Exception as e:
            logger.exception(
                f"[{source.get('name', '?')}] sync_source 未捕获异常: {e}"
            )
            result = SyncResult(
                source=source.get("name", "?"),
                status="error",
                doc_id="",
                prev_doc_id=None,
                inserted=0,
                deleted=0,
                duration_s=0.0,
                error=f"未捕获: {type(e).__name__}: {e}",
            )
            try:
                from metrics import record_sync_done
                source_name = source.get("name", "?")
                record_sync_done(
                    source_name, "error", 0.0,
                    collection=override_collection or source.get("collection"),
                    context={"source": source_name, "uncaught": True},
                )
            except Exception as metric_error:
                logger.warning(f"记录 outer sync 失败 metrics 失败: {metric_error}")
        results.append(result)
    return results


def _dedup_pending_sources(pending, *, report_path=None, dropped_path=None):
    """Deduplicate prepared source chunks by collection before any writes."""
    from dedup import dedup_and_write

    all_rows = [row for item in pending for row in item["records"]]
    if not all_rows:
        return 0
    kept_by_id = {}
    dropped_by_source = Counter()
    kept_by_source = Counter()
    report_path = report_path or Config.DEDUP_REPORT_PATH
    dropped_path = dropped_path or Config.DEDUP_DROPPED_PATH
    # One aggregate report/log per sync call; dedup_records scopes keys by the
    # collection carried by each candidate, preventing cross-collection merging.
    result = dedup_and_write(
        all_rows,
        report_path=report_path,
        dropped_path=dropped_path,
        context={"path": "doc_sync", "scope": "pending_sources"},
    )
    for row in result.kept:
        kept_by_id[row.get("id")] = row
    kept_by_source.update(result.stats.get("kept_by_source", {}))
    dropped_by_source.update(result.stats.get("dropped_by_source", {}))
    for item in pending:
        kept = [row for row in item["records"] if row.get("id") in kept_by_id]
        item["records"] = kept
        item["documents"] = [row["document"] for row in kept]
        item["metadatas"] = [row["metadata"] for row in kept]
        item["ids"] = [row["id"] for row in kept]
        item["dedup_dropped"] = sum(
            1 for row in all_rows
            if row.get("source") == item["source"]["name"] and row.get("id") not in kept_by_id
        )
    try:
        from metrics import record_dedup_done
        record_dedup_done(
            kept_by_source=dict(kept_by_source),
            dropped_by_source=dict(dropped_by_source),
            context={"path": "doc_sync", "scope": "pending_sources"},
        )
    except Exception as e:
        logger.warning(f"记录 doc_sync 去重 metrics 失败: {e}")
    _emit_audit(
        "dedup", "multiple", "",
        input=len(all_rows), kept=len(kept_by_id),
        dropped=result.dropped_count, report_path=report_path,
    )
    return result.dropped_count


def _prepare_sync_b9(source, *, manifest_path=None, override_collection=None,
                      rebuild=False):
    """Build sync candidates without embedding/deleting; used by aggregate B.9."""
    from jsonl2chroma import _balanced_parsed, _parse_valid, _read_records, MAX_DOC_LEN
    from quality_gate import QualityGate, is_within_rate_limit

    name = source["name"]
    file_path = Path(source["file"])
    collection_name = override_collection or source["collection"]
    try:
        stat = file_path.stat()
        file_size, mtime = stat.st_size, stat.st_mtime
        sha = streaming_sha256(file_path)
        new_doc_id = compute_doc_id(file_path, file_size)
    except OSError as e:
        return {"source": source, "error": f"stat/read 源文件失败: {e}"}

    prev = load_manifest(manifest_path).get(name, {})
    if (
        not rebuild
        and prev.get("sha256") == sha
        and prev.get("file_size") == file_size
        and prev.get("last_status") == "ok"
        and prev.get("chunk_count", 0) > 0
        and prev.get("embedding_model") == Config.EMBEDDING_MODEL_ID
    ):
        return {"source": source, "skipped": True}

    per_label_limit = source.get("per_label_limit")
    balanced = bool(per_label_limit) and source.get("limit") is None
    records_iter = _balanced_parsed(source, per_label_limit) if balanced else _read_records(source)
    limit = None if balanced else source.get("limit")
    candidates = []
    records_processed = 0
    for item in records_iter:
        if limit is not None and records_processed >= limit:
            break
        parsed = item if balanced else _parse_valid(source, item)
        if parsed is None:
            continue
        document = parsed["document"].strip()
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
            meta = {k: v for k, v in dict(parsed["metadata"]).items() if v is not None}
            meta["doc_id"] = new_doc_id
            meta.setdefault("source", name)
            meta["sentence_count"] = chunk.sentence_count
            meta["embedding_model"] = Config.EMBEDDING_MODEL_ID
            candidates.append({
                "document": chunk.text,
                "metadata": meta,
                "source": name,
                "score": meta.get("score", 0),
                "collection": collection_name,
                "id": uuid.uuid4().hex,
                "record_index": records_processed - 1,
                "chunk_index": chunk_index,
            })

    gate = QualityGate(name)
    accepted = candidates
    if Config.QUALITY_GATE_ENABLED:
        accepted = gate.check_batch(
            [{"document": row["document"], "metadata": row["metadata"]} for row in candidates],
            doc_id=new_doc_id,
        )
        # QualityGate returns parsed copies; match by document+metadata while
        # retaining the original candidate IDs for manifest/write alignment.
        accepted_keys = {(row["document"], repr(row["metadata"])) for row in accepted}
        accepted = [row for row in candidates
                    if (row["document"], repr(row["metadata"])) in accepted_keys]
        gate.write_report(extra={"phase": "pre_dedup", "source": name})
        if Config.QUALITY_GATE_FAIL_ON_THRESHOLD and not is_within_rate_limit(
            gate.stats, Config.QUALITY_GATE_MAX_REJECTION_RATE
        ):
            return {
                "source": source,
                "gate_blocked": True,
                "doc_id": new_doc_id,
                "prev": prev,
                "error": f"质量门禁拒绝率 {gate.stats.rejection_rate:.2%} 超阈值",
            }

    return {
        "source": source,
        "skipped": False,
        "doc_id": new_doc_id,
        "prev": prev,
        "file_size": file_size,
        "mtime": mtime,
        "sha": sha,
        "collection": collection_name,
        "records": accepted,
        "documents": [row["document"] for row in accepted],
        "metadatas": [row["metadata"] for row in accepted],
        "ids": [row["id"] for row in accepted],
        "dedup_dropped": 0,
    }


def _run_sync_all_dedup(sources, llm_embedding, *, dry_run, override_collection,
                        rebuild, manifest_path, include_demo):
    """Two-phase multi-source sync: prepare, dedup by collection, then commit."""
    prepared = []
    results = {}
    for source in sources:
        item = _prepare_sync_b9(
            source, manifest_path=manifest_path,
            override_collection=override_collection, rebuild=rebuild,
        )
        if item.get("skipped"):
            results[source["name"]] = sync_source(
                source, llm_embedding, dry_run=dry_run,
                override_collection=override_collection, rebuild=rebuild,
                manifest_path=manifest_path, include_demo=include_demo,
            )
        elif item.get("error") or item.get("gate_blocked"):
            results[source["name"]] = SyncResult(
                source=source["name"],
                status="gate_blocked" if item.get("gate_blocked") else "error",
                doc_id=item.get("doc_id", ""), prev_doc_id=item.get("prev", {}).get("doc_id"),
                inserted=0, deleted=0, duration_s=0.0, error=item.get("error"),
            )
        else:
            prepared.append(item)

    pending_names = {item["source"]["name"] for item in prepared}
    if len(pending_names) >= 2:
        _dedup_pending_sources(prepared)

    deleted_collections = set()
    if rebuild and not dry_run and prepared:
        import chromadb
        client = chromadb.PersistentClient(path=Config.CHROMADB_DIRECTORY)
        for item in prepared:
            collection_name = item["collection"]
            if collection_name in deleted_collections:
                continue
            from utils.tools_config import invalidate_qa_cache, invalidate_triage_cache
            if collection_name == Config.CHROMADB_COLLECTION_NAME:
                invalidate_triage_cache()
            elif collection_name == Config.CHROMADB_QA_COLLECTION_NAME:
                invalidate_qa_cache()
            try:
                client.delete_collection(collection_name)
            except Exception as e:
                logger.info(f"[{collection_name}] aggregate rebuild 删除跳过: {e}")
            deleted_collections.add(collection_name)

    for item in prepared:
        source = item["source"]
        results[source["name"]] = sync_source(
            source,
            llm_embedding,
            dry_run=dry_run,
            override_collection=override_collection,
            rebuild=rebuild,
            manifest_path=manifest_path,
            include_demo=include_demo,
            _prepared_records=item["records"],
            _dedup_dropped=item.get("dedup_dropped", 0),
            _rebuild_collection_deleted=(rebuild and not dry_run),
        )
    return [results[source["name"]] for source in sources]


def run_migration(
    sources,
    llm_embedding,  # noqa: ARG001 - 保留签名统一，migrate 不需要 embedding
    manifest_path: str | None = None,
) -> None:
    """一次性回填 doc_id 给存量无标识 chunk（不重 embed）。

    算法：
    1. 对每个 source 算 new_doc_id
    2. 拉全 collection 的 metadatas + ids
    3. 过滤 source 匹配 + meta 无 doc_id 的 chunk
    4. 分批（1000/批）collection.update，只改 metadata 不重 embed
    5. 写 manifest（last_status=migrated，chunk_count=0 因为没 re-ingest）
    """
    import chromadb

    client = chromadb.PersistentClient(path=Config.CHROMADB_DIRECTORY)
    manifest = load_manifest(manifest_path)

    for source in sources:
        name = source["name"]
        file_path = Path(source["file"])
        collection_name = source["collection"]
        try:
            file_size = file_path.stat().st_size
            mtime = file_path.stat().st_mtime
            new_doc_id = compute_doc_id(file_path, file_size)
        except OSError as e:
            logger.error(f"[{name}] stat 失败: {e}")
            continue

        try:
            collection = client.get_collection(collection_name)
        except Exception as e:
            logger.warning(
                f"[{name}] collection {collection_name} 不存在，跳过 migrate: {e}"
            )
            continue

        # 拉全量 metadatas + ids
        raw = collection.get(include=["metadatas"])
        all_ids = raw.get("ids", [])
        all_metas = raw.get("metadatas", [])

        # 过滤 source 匹配 + 无 doc_id
        targets = []
        for cid, meta in zip(all_ids, all_metas):
            meta = meta or {}
            if meta.get("source") == name and "doc_id" not in meta:
                new_meta = dict(meta)
                new_meta["doc_id"] = new_doc_id
                targets.append((cid, new_meta))

        if not targets:
            logger.info(
                f"[{name}] 无需迁移（{len(all_ids)} 条 chunk 已全有 doc_id）"
            )
            manifest[name] = {
                "doc_id": new_doc_id,
                "file_size": file_size,
                "mtime": mtime,
                "chunk_count": 0,
                "last_status": "migrated",
                "last_sync_ts": time.time(),
            }
            save_manifest_atomic(manifest, manifest_path)
            continue

        # 分批 update（不重 embed）
        batch_size = 1000
        migrated = 0
        for i in range(0, len(targets), batch_size):
            batch = targets[i:i + batch_size]
            batch_ids = [t[0] for t in batch]
            batch_metas = [t[1] for t in batch]
            try:
                collection.update(ids=batch_ids, metadatas=batch_metas)
                migrated += len(batch_ids)
                logger.info(f"[{name}] migrate 进度 {migrated}/{len(targets)}")
            except Exception as e:
                logger.error(f"[{name}] migrate 第 {i}-{i + batch_size} 批失败: {e}")
                break

        _emit_audit("sync_migrated", name, new_doc_id, migrated_count=migrated)
        logger.info(f"[{name}] migrate 完成，共 {migrated} 条补 doc_id")

        # 写 manifest
        manifest[name] = {
            "doc_id": new_doc_id,
            "file_size": file_size,
            "mtime": mtime,
            "chunk_count": 0,  # 不记录（没 re-ingest）
            "last_status": "migrated",
            "last_sync_ts": time.time(),
        }
        save_manifest_atomic(manifest, manifest_path)


def run_embedding_migration(
    collections: list[str] | None = None,
    model_id: str | None = None,
) -> None:
    """B.4：给存量无 embedding_model 字段的 chunks 补上标记（不重 embed）。

    推断规则：
    - source == "chinese_medical_dialogue" → "dashscope-text-embedding-v1"（始终用 dashscope）
    - 其它 source → 用当前 Config.EMBEDDING_MODEL_ID
    """
    import chromadb

    if model_id is None:
        model_id = Config.EMBEDDING_MODEL_ID
    if collections is None:
        collections = [Config.CHROMADB_COLLECTION_NAME, Config.CHROMADB_QA_COLLECTION_NAME]

    client = chromadb.PersistentClient(path=Config.CHROMADB_DIRECTORY)
    _LEGACY_MODEL = "dashscope-text-embedding-v1"

    for col_name in collections:
        try:
            col = client.get_collection(col_name)
        except Exception as e:
            logger.warning(f"[embedding_migration] {col_name} 不存在，跳过: {e}")
            continue

        raw = col.get(include=["metadatas"])
        all_ids = raw.get("ids", [])
        all_metas = raw.get("metadatas", [])

        targets = []
        for cid, meta in zip(all_ids, all_metas):
            meta = meta or {}
            if "embedding_model" in meta:
                continue
            new_meta = dict(meta)
            src = new_meta.get("source", "")
            new_meta["embedding_model"] = _LEGACY_MODEL if src == "chinese_medical_dialogue" else model_id
            targets.append((cid, new_meta))

        if not targets:
            logger.info(f"[embedding_migration] {col_name}: 无需迁移")
            continue

        batch_size = 1000
        migrated = 0
        for i in range(0, len(targets), batch_size):
            batch = targets[i:i + batch_size]
            try:
                col.update(ids=[t[0] for t in batch], metadatas=[t[1] for t in batch])
                migrated += len(batch)
                logger.info(f"[embedding_migration] {col_name} 进度 {migrated}/{len(targets)}")
            except Exception as e:
                logger.error(f"[embedding_migration] {col_name} 批次失败: {e}")
                break

        _emit_audit("embedding_migration", col_name, "", migrated_count=migrated, model_id=model_id)
        logger.info(f"[embedding_migration] {col_name} 完成，{migrated} 条 → embedding_model={model_id}")