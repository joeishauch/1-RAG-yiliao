# -*- coding: utf-8 -*-
"""chunking.py — 按中文标点切句的 splitter（B.2 真正切片）。

策略：
1. 主分隔符切句：。！？；\\n
2. 单句超长时按次分隔符（,）再切
3. 贪心合并句子到目标 chunk_size，带 overlap

不切：
- 总字数 <= chunk_size：返回单个 Chunk
- 单句超长且无次分隔符：硬截断到 chunk_size（最后兜底，丢失部分尾部内容）

jsonl2chroma.py 和 doc_sync.py 都复用本模块，保证"导入"和"同步"行为一致。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


# 主分隔符：中文句末 + 分号 + 换行（保留分隔符以维持语义完整）
PRIMARY_DELIM_PATTERN = re.compile(r"([。！？；\n])")


@dataclass
class Chunk:
    """切分后的一个 chunk。

    多个 chunk 共享同一份 metadata（label / answer / source / doc_id 等），
    上层 jsonl2chroma / doc_sync 在切分前解析一次 metadata，切分后复制到每个 chunk。
    """

    text: str
    sentence_count: int = 0  # 该 chunk 含几个原始句子


def split_sentences(text: str) -> List[str]:
    """按主分隔符切句，保留分隔符。

    例："你好。你好！" → ["你好。", "你好！"]
    """
    parts = PRIMARY_DELIM_PATTERN.split(text)
    sentences: List[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if PRIMARY_DELIM_PATTERN.fullmatch(p):
            buf += p
            sentences.append(buf)
            buf = ""
        else:
            buf += p
    if buf:
        sentences.append(buf)
    return sentences


def _split_long_sentence(sent: str, max_size: int) -> List[str]:
    """单句超长，按次分隔符 , 切分。

    "a,b,c,d,e" → ["a,", "b,", "c,", "d,e"]
    无 , 时硬截断到 max_size（兜底）。
    """
    if len(sent) <= max_size:
        return [sent]
    parts = sent.split(",")
    if len(parts) == 1:
        # 无次分隔符，硬截断
        return [sent[:max_size]]
    result: List[str] = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            if p:
                result.append(p + ",")
        else:
            if p:
                result.append(p)
    return result


def chunk_text(
    text: str,
    *,
    chunk_size: int = 384,
    overlap: int = 48,
    min_size: int = 64,
) -> List[Chunk]:
    """贪心合并句子到目标 chunk_size，带 overlap。

    Args:
        text: 原始文档
        chunk_size: 目标 chunk 大小（字），超过则切
        overlap: 与下一个 chunk 共享的尾部字数（避免边界语义断裂）
        min_size: 收尾时若残余 < 此值则合并到上一个 chunk，避免碎片化

    Returns:
        list[Chunk]: 切分后的 chunks，按原文档顺序
    """
    if not text:
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [Chunk(text=text, sentence_count=1)]

    # Step 1: 切句
    raw_sentences = split_sentences(text)

    # Step 2: 单句超长按 , 再切
    sentences: List[str] = []
    for s in raw_sentences:
        if len(s) > chunk_size:
            sentences.extend(_split_long_sentence(s, chunk_size))
        else:
            sentences.append(s)

    # Step 3: 贪心合并
    chunks: List[Chunk] = []
    cur: List[str] = []
    cur_len = 0

    for sent in sentences:
        # 单句加入会超 chunk_size 且 cur 非空 → flush
        if cur_len + len(sent) > chunk_size and cur:
            chunk_text_str = "".join(cur)
            chunks.append(
                Chunk(text=chunk_text_str, sentence_count=len(cur))
            )
            # overlap: 从 cur 末尾取 overlap 字数
            overlap_sents: List[str] = []
            overlap_chars = 0
            for s in reversed(cur):
                overlap_sents.insert(0, s)
                overlap_chars += len(s)
                if overlap_chars >= overlap:
                    break
            cur = overlap_sents
            cur_len = overlap_chars
        cur.append(sent)
        cur_len += len(sent)

    # Step 4: 收尾
    if cur:
        if chunks and cur_len < min_size:
            # 太短：合并到上一个
            last = chunks[-1]
            last.text += "".join(cur)
            last.sentence_count += len(cur)
        else:
            chunks.append(
                Chunk(text="".join(cur), sentence_count=len(cur))
            )

    return chunks


if __name__ == "__main__":
    # 简单演示
    samples = [
        "你好世界。今天天气不错，适合出门散步。注意防晒。",  # 短文本（不切）
        (
            "糖尿病是一组以高血糖为特征的代谢性疾病。"
            "胰岛素是由胰腺分泌的激素，可以降低血糖。"
            "糖尿病分为1型和2型。"
            "1型糖尿病多见于青少年，2型糖尿病多见于中老年人。"
            "妊娠糖尿病发生在妊娠期。"
            "糖尿病的典型症状是多饮、多食、多尿和体重下降。"
            "长期高血糖会损害多个器官系统，特别是眼、肾、心脏、血管和神经。"
            "糖尿病的诊断标准包括空腹血糖、餐后血糖和糖化血红蛋白。"
        ),  # 长答案（应切）
        "你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好",  # 无标点长句（按字数硬切）
        "",  # 空文本
    ]
    for i, s in enumerate(samples, 1):
        chunks = chunk_text(s, chunk_size=80, overlap=20, min_size=20)
        print(f"=== 样本 {i} (原长度={len(s)}) ===")
        print(f"  chunks 数量: {len(chunks)}")
        for j, c in enumerate(chunks, 1):
            print(f"  [{j}] 句子数={c.sentence_count} 长度={len(c.text)}: {c.text!r}")
        print()