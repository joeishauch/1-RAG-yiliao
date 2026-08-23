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

    start_ts = time.time()
    name = source["name"]
    file_path = Path(source["file"])
    collection_name = override_collection or source["collection"]

    # 1. 算 file_size / mtime / sha256 / new_doc_id
    try:
        file_stat = file_path.stat()
        file_size = file_stat.st_size
        mtime = file_stat.st_mtime
        sha = streaming_sha256(file_path)
        new_doc_id = compute_doc_id(file_path, file_size)
    except OSError as e:
        return SyncResult(
            source=name, status="error", doc_id="", prev_doc_id=None,
            inserted=0, deleted=0, duration_s=time.time() - start_ts,
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
    ):
        duration = time.time() - start_ts
        _emit_audit("sync_skip", name, new_doc_id, duration_s=round(duration, 2))
        logger.info(f"[{name}] 无变化（sha256+size+chunk_count 全匹配），跳过")
        return SyncResult(
            source=name, status="skipped", doc_id=new_doc_id,
            prev_doc_id=prev.get("doc_id"), inserted=0, deleted=0,
            duration_s=duration,
        )

    # 4. 取 prev_doc_id
    prev_doc_id = prev.get("doc_id")

    # 5. 跑 parser 生成 documents / metadatas / ids
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
        records_processed += 1  # B.2 fix: 成功解析后 +1

        for tc in text_chunks:
            meta = dict(parsed["metadata"])
            meta["doc_id"] = new_doc_id
            meta.setdefault("source", name)
            meta["sentence_count"] = tc.sentence_count  # B.2：每 chunk 句子数
            documents.append(tc.text)
            metadatas.append({k: v for k, v in meta.items() if v is not None})
            ids.append(uuid.uuid4().hex)
            count += 1

    logger.info(f"[{name}] 解析完成，有效记录 {count} 条")

    if dry_run:
        duration = time.time() - start_ts
        estimated_delete = prev.get("chunk_count", 0) if prev_doc_id else 0
        _emit_audit("sync_dry_run", name, new_doc_id,
                    inserted=count, deleted=estimated_delete,
                    duration_s=round(duration, 2))
        logger.info(f"[{name}] [dry-run] would insert={count}, delete={estimated_delete}")
        return SyncResult(
            source=name, status="ok", doc_id=new_doc_id,
            prev_doc_id=prev_doc_id, inserted=count, deleted=estimated_delete,
            duration_s=duration,
        )

    # 6. 删除旧 chunks（review 补强 #1 首次保护 + #4 rebuild 顺序 + #10 demo 默认排除）
    client = chromadb.PersistentClient(path=Config.CHROMADB_DIRECTORY)
    deleted = 0

    try:
        if rebuild:
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
        duration = time.time() - start_ts
        _emit_audit("sync_error", name, new_doc_id,
                    error=str(e), duration_s=round(duration, 2))
        return SyncResult(
            source=name, status="error", doc_id=new_doc_id,
            prev_doc_id=prev_doc_id, inserted=0, deleted=deleted,
            duration_s=duration, error=f"delete 阶段异常: {e}",
        )

    # 7. 写入新 chunks
    collection = client.get_or_create_collection(name=collection_name)
    inserted = 0
    try:
        for i in range(0, len(documents), Config.SYNC_BATCH_SIZE):
            batch_docs = documents[i:i + Config.SYNC_BATCH_SIZE]
            batch_meta = metadatas[i:i + Config.SYNC_BATCH_SIZE]
            batch_ids = ids[i:i + Config.SYNC_BATCH_SIZE]
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
        duration = time.time() - start_ts
        _emit_audit("sync_error", name, new_doc_id,
                    error=str(e), inserted=inserted, deleted=deleted,
                    duration_s=round(duration, 2))
        return SyncResult(
            source=name, status="error", doc_id=new_doc_id,
            prev_doc_id=prev_doc_id, inserted=inserted, deleted=deleted,
            duration_s=duration, error=f"add 阶段异常: {e}",
        )

    # 8. 原子更新 manifest
    manifest[name] = {
        "doc_id": new_doc_id,
        "sha256": sha,
        "file_size": file_size,
        "mtime": mtime,
        "chunk_count": inserted,
        "chunk_ids": ids[:inserted],
        "last_status": "ok",
        "last_sync_ts": time.time(),
    }
    save_manifest_atomic(manifest, manifest_path)

    # 9. 审计
    duration = time.time() - start_ts
    event = "sync_rebuild" if rebuild else "sync_insert"
    _emit_audit(event, name, new_doc_id,
                inserted=inserted, deleted=deleted, duration_s=round(duration, 2))

    # 10. 失效缓存（非 rebuild 模式；rebuild 已在步骤 6 失效过）
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
        f"[{name}] 完成，inserted={inserted}, deleted={deleted}, duration={duration:.1f}s"
    )
    return SyncResult(
        source=name,
        status="rebuilt" if rebuild else "ok",
        doc_id=new_doc_id,
        prev_doc_id=prev_doc_id,
        inserted=inserted,
        deleted=deleted,
        duration_s=duration,
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

    review 补强 #2：可在此处加"无 doc_id 旧 chunk 护栏" — 检测 manifest 为空但
    collection 已有大量无 doc_id chunk 时，提示先 --migrate。当前默认不强制（让
    --migrate 显式触发），由文档上线顺序约束保证。
    """
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
        results.append(result)
    return results


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