# -*- coding: utf-8 -*-
"""知识图谱构建 CLI 入口。

用法：
    python build_kg.py                          # 默认抽 20 万条建图
    python build_kg.py --sample 50000           # 更小样本快速验证
    python build_kg.py --force                  # 强制重建（忽略缓存）
    python build_kg.py --sample 0               # 全量（sample=0 表示不抽样，内存需求大）
"""
import argparse
import logging

from utils.config import Config
from utils.kg_builder import build_or_load

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT = Config.KG_SOURCE_PATH
DEFAULT_OUT = Config.KG_GRAPH_PATH


def main():
    parser = argparse.ArgumentParser(description="从 huatuo_knowledge_graph_qa 构建 medical 知识图谱")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="数据源 jsonl 路径")
    parser.add_argument("--sample", type=int, default=Config.KG_BUILD_SAMPLE,
                        help="抽样条数；0 表示全量（内存需求大，谨慎）")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="图缓存文件路径")
    parser.add_argument("--force", action="store_true", help="忽略缓存强制重建")
    args = parser.parse_args()

    sample = args.sample if args.sample and args.sample > 0 else None
    G = build_or_load(args.input, args.out, sample=sample, force=args.force)
    print(f"\n图规模: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"源版本: {G.graph.get('source_sha256', 'unknown')}")
    print(f"缓存文件: {args.out}")


if __name__ == "__main__":
    main()
