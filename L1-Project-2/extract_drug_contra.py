# -*- coding: utf-8 -*-
"""从百科数据抽取药物禁忌（LLM 结构化抽取 → 独立 JSON 知识库）。

数据源：input/huatuo_encyclopedia_qa/train_datasets.jsonl（36.2 万条）。
只抽取「XX药/片/胶囊 + 禁忌症/用药禁忌」类药品说明书条目（全量约 2500 条，以脚本筛选规则实际命中为准）：
药物名从 question 正则提取，禁忌条目由 deepseek 结构化输出抽取。

用法：
    python extract_drug_contra.py                 # 默认抽前 100 条验证
    python extract_drug_contra.py --limit 0       # 全量（0=不限，确认质量后）
    python extract_drug_contra.py --limit 100 --out drug_contraindications_demo.json
"""
import argparse
import json
import logging
import os
import re
import sys
import time

# 切换到脚本所在目录，保证 input/ .env 等相对路径正确（与 e2e_regression.py 一致）
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

# Windows 下 stdout 可能是 GBK，统一转 UTF-8 避免中文乱码/报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("extract_drug_contra")

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from utils.config import Config
from utils.llms import get_llm

DEFAULT_INPUT = "input/huatuo_encyclopedia_qa/train_datasets.jsonl"
DEFAULT_OUT = "drug_contraindications.json"

# 药物提示词：question 里出现这些剂型/药物特征词才算药物禁忌（过滤「饮食禁忌」等疾病类禁忌）
DRUG_HINT = re.compile(r"药|片|胶囊|颗粒|丸|素|注射|口服|滴眼|软膏|溶液|糖浆|栓|喷雾|贴剂|混悬|糖衣|肠溶|缓释")
TABOO_HINT = re.compile(r"禁忌|禁忌症")

# 药名尾部需要去掉的修饰词（「禁忌」已用 find 切掉，这里只需清「的用药/有哪些/是什么/的」等尾缀；先长后短）
_TAIL_CLEAN = re.compile(r"(的用药|用药|有哪些|是什么|哪些|什么|是|有|的)+$")


class ContraindicationResult(BaseModel):
    """LLM 结构化输出：禁忌条目列表。"""
    contraindications: list[str] = Field(description="从答案中抽取的禁忌/禁用/忌服条目，每条一个独立禁忌人群或禁忌症")


EXTRACT_TEMPLATE = """你是药物说明书信息抽取助手。请从下面这条药品「禁忌」问答中，抽取所有禁忌条目。

=== 问题 ===
{question}

=== 答案 ===
{answer}

抽取要求：
1. 只抽取「禁忌 / 禁用 / 忌服 / 慎用」相关的人群或情况，每条单独列出，不合并。
2. 去掉「注意：本文仅供参考」「请遵医嘱」「具体请咨询医生」这类免责声明。
3. 保留原意、不编造；若答案没有实质禁忌内容（如只是泛泛说"遵医嘱"），返回空列表。
4. 每条禁忌保持简洁，直接描述禁忌人群或禁忌症（如「活动期溃疡病患者」「孕妇禁用」「对本品过敏者禁用」）。

只输出结构化结果，不要输出其他文字。"""


def _iter_records(path):
    """逐行读 jsonl，yield dict（复用 kg_builder 的读取逻辑，独立实现避免私有依赖）。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _first_question(record):
    """兼容 questions 是 [[..]] 或 [..] 两种结构，取第一个问题文本"""
    qs = record.get("questions", [])
    if not qs:
        return ""
    q0 = qs[0]
    return q0[0] if isinstance(q0, list) else q0


def _first_answer(record):
    ans = record.get("answers", [])
    return ans[0] if ans else ""


def is_drug_taboo(question):
    """药物禁忌候选：question 同时含药物剂型特征词 + 「禁忌」"""
    return bool(DRUG_HINT.search(question)) and bool(TABOO_HINT.search(question))


def extract_drug_name(question):
    """从 question 提取药物名：取「禁忌」之前的部分，去掉尾部修饰词。

    例：
        强力脑清素片禁忌症是什么？  → 强力脑清素片
        平地木的用药禁忌是什么?      → 平地木
        XX药有哪些禁忌               → XX药
    """
    q = question.strip()
    idx = q.find("禁忌")
    head = q[:idx] if idx > 0 else q
    # 去掉尾部「的用药 / 有哪些 / 是什么 / 的」等修饰词
    head = _TAIL_CLEAN.sub("", head)
    head = head.strip().strip("。，,；; ：:").strip()
    return head


def collect_candidates(input_path, limit):
    """流式扫描，筛出药物禁忌候选，返回 [(question, answer)]；limit<=0 表示不限（全量）"""
    candidates = []
    for rec in _iter_records(input_path):
        q = _first_question(rec)
        if not q:
            continue
        if is_drug_taboo(q):
            candidates.append((q, _first_answer(rec)))
            if 0 < limit <= len(candidates):
                break
    return candidates


def main():
    parser = argparse.ArgumentParser(description="从百科数据 LLM 抽取药物禁忌")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="百科数据 jsonl 路径")
    parser.add_argument("--limit", type=int, default=100, help="抽取条数上限（候选按文件顺序），0=不限（全量）")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="输出 JSON 路径")
    args = parser.parse_args()

    logger.info(f"扫描候选: {args.input}（limit={args.limit}）")
    candidates = collect_candidates(args.input, args.limit)
    logger.info(f"药物禁忌候选数: {len(candidates)}")

    llm_chat, _ = get_llm(Config.LLM_TYPE)
    chain = ChatPromptTemplate.from_messages([("human", EXTRACT_TEMPLATE)]) \
        | llm_chat.with_structured_output(ContraindicationResult, method="function_calling")

    results = []
    n_ok = n_fail = n_empty = 0
    total_items = 0

    for i, (question, answer) in enumerate(candidates, 1):
        drug_name = extract_drug_name(question)
        t0 = time.time()
        try:
            parsed = chain.invoke({"question": question, "answer": answer})
            items = parsed.contraindications or []
            if not items:
                n_empty += 1
            n_ok += 1
            total_items += len(items)
            results.append({
                "drug_name": drug_name,
                "contraindications": items,
                "source": question,
            })
            logger.info(f"[{i}/{len(candidates)}] {drug_name}: {len(items)} 条禁忌 "
                        f"({time.time() - t0:.1f}s)")
        except Exception as e:
            n_fail += 1
            logger.error(f"[{i}/{len(candidates)}] {drug_name}: 抽取失败 - {e}")
            results.append({"drug_name": drug_name, "error": str(e), "source": question})

    # 写 JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已写入 {args.out}")

    # 质量统计
    print(f"\n=== 药物禁忌抽取结果 ===")
    print(f"候选数:            {len(candidates)}")
    print(f"成功:              {n_ok}")
    print(f"失败:              {n_fail}")
    print(f"空禁忌（成功但无条目）: {n_empty}")
    print(f"平均禁忌条目数:     {total_items / n_ok if n_ok else 0:.2f}")

    # 逐条打印前 20 条成功结果供人工核对
    print(f"\n=== 前 20 条抽取结果 ===")
    shown = 0
    for r in results:
        if "error" in r:
            print(f"[失败] {r['drug_name']}: {r['error']}")
            continue
        if shown >= 20:
            break
        shown += 1
        items = "；".join(r["contraindications"]) if r["contraindications"] else "（空）"
        print(f"{shown}. 【{r['drug_name']}】 {items}")


if __name__ == "__main__":
    main()
