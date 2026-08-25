# -*- coding: utf-8 -*-
"""backup.py — ChromaDB 灾备快照。

把 chromaDB/ 目录打包为 tar.gz，按日期命名，自动清理过期备份。

保留策略：
  - 最近 7 天：每天保留
  - 最近 4 周：每周保留（周日）
  - 最近 6 月：每月保留（每月1号）

用法：
    python backup.py                        # 默认备份到 backups/
    python backup.py --out /path/to/backups # 自定义备份目录
    python backup.py --dry-run              # 只打印会做什么，不实际执行
    python backup.py --restore backups/chromadb_20260825_020000.tar.gz  # 恢复

定时任务（Windows 任务计划程序 / Linux cron）：
    0 2 * * * cd /path/to/project && python backup.py

恢复流程：
    1. 停止服务（python cli.py serve）
    2. python backup.py --restore backups/chromadb_YYYYMMDD_HHMMSS.tar.gz
    3. 重启服务
"""
import argparse
import os
import shutil
import sys
import tarfile
import time
from datetime import datetime, timedelta
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台中文编码
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# 配置
CHROMADB_DIR = "chromaDB"
DEFAULT_BACKUP_DIR = "backups"
BACKUP_PREFIX = "chromadb_"

# 保留策略阈值
KEEP_RECENT_DAYS = 7      # 最近 7 天全保留
KEEP_WEEKLY_WEEKS = 4     # 最近 4 周每周保留
KEEP_MONTHLY_MONTHS = 6   # 最近 6 月每月保留


def create_backup(backup_dir: str, dry_run: bool = False) -> str | None:
    """打包 chromaDB/ 为 tar.gz，返回备份文件路径。"""
    src = Path(CHROMADB_DIR)
    if not src.exists():
        print(f"❌ 源目录不存在: {src}")
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{BACKUP_PREFIX}{ts}.tar.gz"
    backup_path = Path(backup_dir) / backup_name

    # 计算源目录大小
    total_size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)

    if dry_run:
        print(f"[dry-run] 将备份 {src}/ → {backup_path} ({size_mb:.1f} MB)")
        return None

    Path(backup_dir).mkdir(parents=True, exist_ok=True)
    print(f"正在备份 {src}/ → {backup_path} ({size_mb:.1f} MB) ...")
    t0 = time.time()

    with tarfile.open(backup_path, "w:gz") as tar:
        tar.add(src, arcname=CHROMADB_DIR)

    duration = time.time() - t0
    actual_size = backup_path.stat().st_size / (1024 * 1024)
    print(f"✅ 备份完成: {backup_path} ({actual_size:.1f} MB, {duration:.1f}s)")
    return str(backup_path)


def parse_backup_date(name: str) -> datetime | None:
    """从备份文件名解析日期。chromadb_20260825_020000.tar.gz → datetime(2026,8,25,2,0,0)"""
    try:
        ts = name.replace(BACKUP_PREFIX, "").replace(".tar.gz", "")
        return datetime.strptime(ts, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def cleanup_old_backups(backup_dir: str, dry_run: bool = False) -> int:
    """按保留策略清理过期备份，返回删除数量。"""
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        return 0

    backups = []
    for f in backup_path.glob(f"{BACKUP_PREFIX}*.tar.gz"):
        dt = parse_backup_date(f.name)
        if dt:
            backups.append((f, dt))
    backups.sort(key=lambda x: x[1])

    if not backups:
        return 0

    now = datetime.now()
    keep = set()

    for f, dt in backups:
        age_days = (now - dt).days

        # 最近 7 天：全保留
        if age_days <= KEEP_RECENT_DAYS:
            keep.add(f)
            continue

        # 最近 4 周：每周日保留
        if age_days <= KEEP_WEEKLY_WEEKS * 7:
            if dt.weekday() == 6:  # 周日
                keep.add(f)
            continue

        # 最近 6 月：每月1号保留
        if age_days <= KEEP_MONTHLY_MONTHS * 30:
            if dt.day == 1:
                keep.add(f)
            continue

    to_delete = [f for f, _ in backups if f not in keep]
    if not to_delete:
        return 0

    if dry_run:
        print(f"[dry-run] 将删除 {len(to_delete)} 个过期备份:")
        for f in to_delete:
            print(f"  {f.name}")
        return 0

    for f in to_delete:
        f.unlink()
        print(f"  已删除: {f.name}")

    print(f"清理完成: 删除 {len(to_delete)} 个过期备份，保留 {len(backups) - len(to_delete)} 个")
    return len(to_delete)


def restore_backup(backup_file: str, dry_run: bool = False) -> bool:
    """从 tar.gz 恢复 chromaDB/ 目录。"""
    backup_path = Path(backup_file)
    if not backup_path.exists():
        print(f"❌ 备份文件不存在: {backup_path}")
        return False

    target = Path(CHROMADB_DIR)
    if target.exists():
        if dry_run:
            print(f"[dry-run] 将删除并覆盖 {target}/")
        else:
            print(f"⚠️  删除现有 {target}/ ...")
            shutil.rmtree(target)

    if dry_run:
        print(f"[dry-run] 将从 {backup_path} 恢复到 {target}/")
        return True

    print(f"正在从 {backup_path} 恢复 ...")
    t0 = time.time()
    with tarfile.open(backup_path, "r:gz") as tar:
        tar.extractall(path=".")
    duration = time.time() - t0

    count = sum(1 for _ in target.rglob("*") if _.is_file())
    print(f"✅ 恢复完成: {target}/ ({count} 文件, {duration:.1f}s)")
    return True


def list_backups(backup_dir: str):
    """列出所有备份。"""
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        print("暂无备份")
        return

    backups = []
    for f in sorted(backup_path.glob(f"{BACKUP_PREFIX}*.tar.gz")):
        dt = parse_backup_date(f.name)
        size_mb = f.stat().st_size / (1024 * 1024)
        backups.append((f.name, dt, size_mb))

    if not backups:
        print("暂无备份")
        return

    print(f"{'文件名':<40} {'日期':<20} {'大小':>8}")
    print("-" * 70)
    for name, dt, size in backups:
        date_str = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "未知"
        print(f"{name:<40} {date_str:<20} {size:>7.1f}MB")
    print(f"\n共 {len(backups)} 个备份")


def main():
    parser = argparse.ArgumentParser(description="ChromaDB 灾备快照")
    parser.add_argument("--out", type=str, default=DEFAULT_BACKUP_DIR, help="备份目录（默认 backups/）")
    parser.add_argument("--dry-run", action="store_true", help="只打印不执行")
    parser.add_argument("--restore", type=str, default=None, help="从指定备份文件恢复")
    parser.add_argument("--list", action="store_true", help="列出所有备份")
    parser.add_argument("--no-cleanup", action="store_true", help="跳过过期备份清理")
    args = parser.parse_args()

    if args.restore:
        ok = restore_backup(args.restore, dry_run=args.dry_run)
        sys.exit(0 if ok else 1)

    if args.list:
        list_backups(args.out)
        return

    # 备份
    result = create_backup(args.out, dry_run=args.dry_run)
    if result or args.dry_run:
        # 清理过期备份
        if not args.no_cleanup:
            cleanup_old_backups(args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
