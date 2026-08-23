# -*- coding: utf-8 -*-
"""分诊检索评估：对比 纯向量 / BM25+向量(RRF) / 混合+重排 三档的 Hit@K + MRR。

指标说明：
    Hit@K —— 期望科室出现在检索 Top-K 中的用例占比（rank 按原始返回位置计，含「其他」占位）
    MRR   —— 期望科室首次出现位置倒数的均值（rank=1 记 1，未命中记 0）

用法：
    python eval_triage_retrieval.py

依赖：
    DASHSCOPE_API_KEY（通义 text-embedding-v1 向量召回）
    重排需本地 BGE-reranker-large（utils/config.py 的 RERANKER_MODEL_PATH），加载失败自动降级为 hybrid
"""
import json
import logging
import os
import sys
import time

# 保证脚本在任意 cwd 下都能 import 项目内模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_triage_retrieval")

from utils.config import Config
from utils.tools_config import get_triage_context, INVALID_LABELS

# 测试用例：症状 → 期望科室（label 名与 ChromaDB metadata 完全一致）
# 覆盖 16 个科室中的 15 个；「其他」为非具体兜底标签、「急诊科」仅 3 条无法可靠召回，均不参与
TEST_CASES = [
    # 眼耳鼻喉科（5 条）
    ("鼻塞流鼻涕打喷嚏", ["眼耳鼻喉科"]),
    ("鼻中隔偏曲鼻堵", ["眼耳鼻喉科"]),
    ("听力下降耳鸣耳朵闷", ["眼耳鼻喉科"]),
    ("扁桃体发炎化脓", ["眼耳鼻喉科"]),
    ("眼睛红肿干涩疼痛", ["眼耳鼻喉科"]),
    # 妇产科（5 条）
    ("月经不调痛经", ["妇产科"]),
    ("白带异常外阴瘙痒", ["妇产科"]),
    ("宫颈筛查TCT异常", ["妇产科"]),
    ("更年期潮热盗汗", ["妇产科"]),
    ("子宫肌瘤", ["妇产科"]),
    # 内科（5 条）
    ("咳嗽发烧咳痰", ["内科"]),
    ("胃痛胃胀反酸", ["内科"]),
    ("高血压头晕", ["内科"]),
    ("糖尿病血糖控制", ["内科"]),
    ("心悸胸闷气短", ["内科"]),
    # 皮肤性病科（4 条）
    ("皮肤红疹瘙痒", ["皮肤性病科"]),
    ("痤疮痘痘粉刺", ["皮肤性病科"]),
    ("湿疹反复发作", ["皮肤性病科"]),
    ("尖锐湿疣", ["皮肤性病科"]),
    # 口腔科（3 条）
    ("牙痛牙龈肿痛", ["口腔科"]),
    ("龋齿蛀牙", ["口腔科"]),
    ("口腔溃疡反复", ["口腔科"]),
    # 神经科学（4 条）
    ("头晕耳鸣听力下降", ["眼耳鼻喉科", "神经科学"]),
    ("头痛偏头痛恶心", ["神经科学", "内科"]),
    ("面瘫口歪", ["神经科学"]),
    ("帕金森手抖", ["神经科学"]),
    # 心理科学（3 条）
    ("失眠焦虑情绪低落", ["心理科学", "神经科学"]),
    ("抑郁症兴趣减退", ["心理科学"]),
    ("社交恐惧紧张", ["心理科学"]),
    # 外科（3 条）
    ("骨折关节疼痛肿胀", ["外科"]),
    ("阑尾炎腹痛", ["外科"]),
    ("痔疮便血", ["外科"]),
    # 儿科（3 条）
    ("小孩发烧咳嗽流涕", ["儿科"]),
    ("新生儿黄疸", ["儿科"]),
    ("小儿腹泻", ["儿科"]),
    # 男性健康科（3 条）
    ("前列腺不适尿频尿急", ["男性健康科"]),
    ("阳痿早泄", ["男性健康科"]),
    ("前列腺增生", ["男性健康科"]),
    # 生殖健康科（3 条）
    ("避孕不孕备孕咨询", ["生殖健康科", "妇产科"]),
    ("试管婴儿咨询", ["生殖健康科"]),
    ("输卵管堵塞", ["生殖健康科"]),
    # 感染与免疫科（3 条）
    ("发热咳嗽咽痛全身酸痛", ["感染与免疫科", "内科"]),
    ("乙肝大三阳", ["感染与免疫科"]),
    ("艾滋病检测", ["感染与免疫科"]),
    # 肿瘤科（3 条）
    ("乳房肿块乳头溢液", ["肿瘤科", "外科"]),
    ("肺癌咳血", ["肿瘤科"]),
    ("化疗副作用", ["肿瘤科"]),
    # 中医科（3 条）
    ("腰酸乏力气血不足", ["中医科"]),
    ("中药调理", ["中医科"]),
    ("针灸治疗", ["中医科"]),
]

MODES = ["vector", "hybrid", "hybrid_rerank"]
TOP_K = 20  # 与分诊统计的 TRIAGE_TOP_K 一致


def _init_embedding():
    """仅初始化 embedding（通义 text-embedding-v1），不依赖 DEEPSEEK_API_KEY。"""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v1",
        deployment="text-embedding-v1",
        check_embedding_ctx_length=False,
    )


def eval_mode(hybrid, mode):
    """对指定 mode 跑全部用例，返回 (指标, 明细)。"""
    hit = {1: 0, 3: 0, 5: 0}
    mrr_sum = 0.0
    miss = 0
    details = []

    for query, expected in TEST_CASES:
        docs = hybrid.search(query, top_k=TOP_K, mode=mode)
        rank = None
        retrieved = []
        for i, d in enumerate(docs):
            label = d.metadata.get("label", "未知科室")
            retrieved.append(label)
            if rank is None and label in expected:
                rank = i + 1  # 期望科室首次出现的 rank（1-indexed）

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
            "expected": expected,
            "rank": rank,
            "top5": retrieved[:5],
        })

    n = len(TEST_CASES)
    return {
        "mode": mode,
        "hit@1": round(hit[1] / n, 4),
        "hit@3": round(hit[3] / n, 4),
        "hit@5": round(hit[5] / n, 4),
        "mrr": round(mrr_sum / n, 4),
        "miss_rate": round(miss / n, 4),
    }, details


def main():
    logger.info("初始化 embedding（通义 text-embedding-v1）...")
    llm_embedding = _init_embedding()

    logger.info("构建分诊上下文（Chroma + 先验分布 + BM25 索引，仅首次较慢）...")
    _vectorstore, prior_counter, prior_total, hybrid = get_triage_context(llm_embedding)
    logger.info(f"先验科室数：{len(prior_counter)}，总病例：{prior_total}")

    # 重排器状态提示
    if hybrid.reranker is not None:
        logger.info("重排器：已加载（CrossEncoder）")
    else:
        logger.warning("重排器：未加载（hybrid_rerank 将退化为 hybrid）")

    results = []
    all_details = {}
    for mode in MODES:
        logger.info(f"评估 mode={mode} ...")
        t0 = time.time()
        metrics, details = eval_mode(hybrid, mode)
        metrics["elapsed_s"] = round(time.time() - t0, 2)
        results.append(metrics)
        all_details[mode] = details
        logger.info(
            f"  {mode}: Hit@1={metrics['hit@1']:.2%} Hit@3={metrics['hit@3']:.2%} "
            f"Hit@5={metrics['hit@5']:.2%} MRR={metrics['mrr']:.3f} miss={metrics['miss_rate']:.2%}"
        )

    # 控制台对比表
    print("\n=== 检索评估对比（Hit@K + MRR）===")
    print(f"{'mode':<15}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}{'miss':>8}{'耗时':>8}")
    for m in results:
        print(
            f"{m['mode']:<15}{m['hit@1']:>8.2%}{m['hit@3']:>8.2%}{m['hit@5']:>8.2%}"
            f"{m['mrr']:>8.3f}{m['miss_rate']:>8.2%}{m['elapsed_s']:>7.1f}s"
        )

    # JSON 报告
    os.makedirs("eval_reports", exist_ok=True)
    report = {
        "config": {
            "top_k": TOP_K,
            "modes": MODES,
            "n_cases": len(TEST_CASES),
            "reranker_path": Config.RERANKER_MODEL_PATH,
            "reranker_loaded": hybrid.reranker is not None,
            "invalid_labels_filtered": sorted(INVALID_LABELS),
        },
        "summary": results,
        "details": all_details,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"eval_reports/triage_retrieval_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"报告已写入 {path}")


if __name__ == "__main__":
    main()
