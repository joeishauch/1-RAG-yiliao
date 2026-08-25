# -*- coding: utf-8 -*-
"""cleanup_collections.py — 清理旧测试 collection，保留生产数据。

保留：medical_triage（分诊）、medical_qa（科普问答）
删除：其他所有 collection（历史实验残留）

用法：
    python cleanup_collections.py              # 预览模式（只列出不删除）
    python cleanup_collections.py --confirm    # 实际删除

需要在 xiangmu-2 conda 环境运行（chromadb 版本兼容）。
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import chromadb

PRODUCTION_COLLECTIONS = {"medical_triage", "medical_qa"}


def main():
    confirm = "--confirm" in sys.argv

    client = chromadb.PersistentClient(path="chromaDB")
    collections = client.list_collections()

    print(f"共 {len(collections)} 个 collection:\n")
    print(f"{'名称':<20} {'条数':>10} {'状态':>8}")
    print("-" * 42)

    to_delete = []
    to_keep = []

    for col in collections:
        name = col.name
        try:
            count = col.count()
        except Exception:
            count = "?"

        if name in PRODUCTION_COLLECTIONS:
            to_keep.append((name, count))
            print(f"  {name:<18} {count:>10}   保留")
        else:
            to_delete.append((name, count))
            print(f"  {name:<18} {count:>10}   待删除")

    print(f"\n保留: {len(to_keep)} 个")
    for name, count in to_keep:
        print(f"  {name} ({count} 条)")

    print(f"\n待删除: {len(to_delete)} 个")
    for name, count in to_delete:
        print(f"  {name} ({count} 条)")

    if not to_delete:
        print("\n没有需要清理的 collection")
        return

    if not confirm:
        print(f"\n预览模式：确认删除请加 --confirm")
        return

    print(f"\n开始删除 ...")
    deleted = 0
    for name, _ in to_delete:
        try:
            client.delete_collection(name)
            print(f"  已删除: {name}")
            deleted += 1
        except Exception as e:
            print(f"  删除失败: {name} - {e}")

    print(f"\n清理完成: 删除 {deleted}/{len(to_delete)} 个 collection")


if __name__ == "__main__":
    main()
