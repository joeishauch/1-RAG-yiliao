# -*- coding: utf-8 -*-
"""生成后审计：所有 AI 输出与人工修改留痕到本地 JSONL。

落点放边界（main.py / cli.py），不进图 state：
- 图内写文件是副作用，与 interrupt 幂等性冲突（resume 会重跑节点）；
- 边界能拿到完整上下文（输入 / 草稿 / 决策 / 最终答案）。

隐私约定：只记录脱敏后的输入与 `redacted`/`redaction_count` 标记，绝不记录 PII 原文。
"""
import json
import os
import threading
import time
import logging

from utils.config import Config

logger = logging.getLogger(__name__)

# 写锁：保证多线程下追加不交错
_write_lock = threading.Lock()


def build_record(*, event, thread_id, user_id, risk_level=None, node=None,
                 redacted=False, redaction_count=0, user_input=None,
                 draft=None, action=None, revised_answer=None,
                 reviewer=None, final_answer=None, comment=None):
    """组装一条审计记录，统一字段与默认值。

    Args:
        event: 事件类型（block / draft / review_decision / final）。
        其余参数按事件类型选择性传入。

    Returns:
        dict: 含 timestamp 的统一审计记录。
    """
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        "thread_id": thread_id,
        "user_id": user_id,
        "risk_level": risk_level,
        "node": node,
        "redacted": redacted,
        "redaction_count": redaction_count,
        "user_input": user_input,        # 已脱敏
        "draft": draft,
        "action": action,
        "revised_answer": revised_answer,
        "reviewer": reviewer,
        "final_answer": final_answer,
        "comment": comment,
    }


def write_audit(record):
    """追加一条 JSON 到审计日志文件（线程安全，失败仅记日志不中断主流程）。

    Args:
        record: build_record 组装的审计记录 dict。
    """
    try:
        log_dir = os.path.dirname(Config.AUDIT_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _write_lock:
            with open(Config.AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.error(f"审计日志写入失败: {e}")
