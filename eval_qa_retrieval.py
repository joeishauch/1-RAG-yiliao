# -*- coding: utf-8 -*-
"""医疗问答检索评估：B.2 新建，32 条用例评估 medical_qa collection 检索质量。

指标说明：
- Hit@K —— top-k 中任一文档 page_content 含 expected_keywords 之一的用例占比
- MRR   —— 首个命中关键词文档位置倒数的均值

依赖：
- DASHSCOPE_API_KEY（通义 text-embedding-v1 向量召回）
- medical_qa collection（huatuo_encyclopedia / huatuo_knowledge_graph / chinese_medical_dialogue）

与 eval_triage_retrieval.py 的区别：
- 分诊评估用 BM25+向量 hybrid（需要科室 label 字段）
- QA 评估只用纯向量（medical_qa 没科室 label，hybrid BM25 会引入无关医学词噪声）
- QA 期望用关键词包含判定（不像分诊有明确 label 字段）
"""
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_qa_retrieval")

from langchain_chroma import Chroma
from utils.config import Config


TEST_CASES = [
    # 慢性病（5 条）
    ("什么是糖尿病", ["糖尿病", "血糖", "胰岛素"]),
    ("高血压饮食要注意什么", ["低盐", "清淡", "降压"]),
    ("冠心病怎么治疗", ["冠状动脉", "心脏", "心绞痛"]),
    ("哮喘发作怎么办", ["哮喘", "气喘", "呼吸困难"]),
    ("慢性胃炎的症状", ["胃", "上腹", "消化"]),
    # 常见症状（5 条）
    ("头痛怎么办", ["头痛", "头部", "神经"]),
    ("咳嗽吃什么药", ["咳嗽", "止咳", "咽"]),
    ("长期失眠怎么调理", ["失眠", "睡眠", "入睡"]),
    ("便秘吃什么", ["便秘", "排便", "纤维"]),
    ("过敏性鼻炎", ["过敏", "鼻炎", "鼻塞"]),
    # 用药安全（4 条）
    ("抗生素怎么吃", ["抗生素", "细菌", "感染"]),
    ("止痛药副作用", ["止痛", "镇痛", "胃"]),
    ("降压药什么时候吃", ["降压", "血压", "高血压"]),
    ("感冒药能混着吃吗", ["感冒", "药物", "相互作用"]),
    # 母婴（4 条）
    ("怀孕初期注意什么", ["怀孕", "孕妇", "妊娠"]),
    ("哺乳期不能吃什么药", ["哺乳", "母乳", "婴儿"]),
    ("婴儿辅食怎么添加", ["辅食", "婴儿", "添加"]),
    ("孕妇感冒怎么办", ["孕妇", "怀孕", "感冒"]),
    # 营养（4 条）
    ("维生素C的作用", ["维生素", "维C", "抗氧化"]),
    ("蛋白质摄入多少", ["蛋白质", "氨基酸", "肌肉"]),
    ("膳食纤维的好处", ["膳食纤维", "纤维", "肠道"]),
    ("补钙吃什么", ["钙", "骨骼", "补钙"]),
    # 老年（3 条）
    ("骨质疏松怎么办", ["骨质疏松", "骨密度", "钙"]),
    ("老年痴呆前兆", ["痴呆", "记忆", "认知"]),
    ("康复训练怎么做", ["康复", "训练", "功能"]),
    # 季节性 / 急救（3 条）
    ("流感怎么预防", ["流感", "疫苗", "病毒"]),
    ("皮肤过敏怎么办", ["过敏", "皮肤", "瘙痒"]),
    ("中暑怎么处理", ["中暑", "高温", "脱水"]),
    # 检验（2 条）
    ("空腹血糖多少正常", ["血糖", "空腹", "正常值"]),
    ("肝功能指标怎么看", ["肝功能", "转氨酶", "肝脏"]),
    # 内分泌（2 条）
    ("甲状腺结节严重吗", ["甲状腺", "结节", "内分泌"]),
    ("痛风不能吃什么", ["痛风", "嘌呤", "尿酸"]),
]

TOP_K = 5
MODES = ["vector"]  # medical_qa 只用纯向量（hybrid BM25 噪声大）


def _init_embedding():
    """仅初始化 embedding，不依赖 DEEPSEEK_API_KEY。"""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v1",
        deployment="text-embedding-v1",
        check_embedding_ctx_length=False,
    )


def _keyword_hit(content: str, expected_kws: list) -> int:
    """返回首个命中的 keyword 在 expected_kws 列表里的索引，未命中返回 -1。"""
    for i, kw in enumerate(expected_kws):
        if kw in content:
            return i
    return -1


def eval_mode(vectorstore, mode: str):
    """对指定 mode 跑全部用例，返回 (指标, 明细)。"""
    hit = {1: 0, 3: 0, 5: 0}
    mrr_sum = 0.0
    miss = 0
    details = []

    for query, expected_kws in TEST_CASES:
        docs = vectorstore.similarity_search(query, k=TOP_K)
        rank = None
        retrieved_snippets = []
        for i, d in enumerate(docs):
            content = d.page_content
            retrieved_snippets.append(content[:60])
            if rank is None and _keyword_hit(content, expected_kws) >= 0:
                rank = i + 1  # 1-indexed

        if rank is None:
            miss += 1
        else:
            if rank <= 1:
                hit[1] += 1
            if rank <= 3:
                hit[3] += 1
            if rank <= 5:
                hit[5] += 1
            mrr_sum += 1.0 / rank

        details.append({
            "query": query,
            "expected_kws": expected_kws,
            "rank": rank,
            "top_snippets": retrieved_snippets,
        })

    n = len(TEST_CASES)
    metrics = {
        "hit@1": hit[1] / n,
        "hit@3": hit[3] / n,
        "hit@5": hit[5] / n,
        "mrr": mrr_sum / n,
        "miss_rate": miss / n,
    }
    return metrics, details


def main():
    print("=== 医疗问答检索评估（medical_qa）===")
    embedding = _init_embedding()
    vs = Chroma(
        persist_directory=Config.CHROMADB_DIRECTORY,
        collection_name=Config.CHROMADB_QA_COLLECTION_NAME,
        embedding_function=embedding,
    )
    print(f"medical_qa count = {vs._collection.count()}")

    for mode in MODES:
        metrics, details = eval_mode(vs, mode)
        print(
            f"\n{mode}: Hit@1={metrics['hit@1']:.2%} Hit@3={metrics['hit@3']:.2%} "
            f"Hit@5={metrics['hit@5']:.2%} MRR={metrics['mrr']:.3f} "
            f"miss={metrics['miss_rate']:.2%}"
        )
        # 打印 miss 详情（前 5 条）便于人工排查
        miss_cases = [d for d in details if d["rank"] is None]
        if miss_cases:
            print(f"\n  未命中 ({len(miss_cases)} 条，仅展示前 5):")
            for d in miss_cases[:5]:
                print(f"    query={d['query']!r} expected_kws={d['expected_kws']}")
                for j, snippet in enumerate(d["top_snippets"][:3], 1):
                    print(f"      [{j}] {snippet!r}")


if __name__ == "__main__":
    main()