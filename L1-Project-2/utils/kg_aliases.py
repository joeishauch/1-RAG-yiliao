# -*- coding: utf-8 -*-
"""疾病同义映射表：简称/俗称 → 图中真实存在的规范疾病节点。

数据源 huatuo_knowledge_graph_qa 的疾病节点名很脏，实体对齐要处理三类：
  1. 误分类：简称被当成了 symptom 节点（「冠心病」「类风湿关节炎」有
     「XX 是什么疾病的症状」模板，被误标为 symptom）——需纠正到真疾病节点
  2. 简称 vs 全称：甲亢/甲状腺功能亢进、哮喘/支气管哮喘——归一即可
  3. 纯脏变体：DES冠心病、早期胃癌、IABP冠心病——本表无法解决（无干净全称
     节点可映射），留第二阶段「修饰剥离」处理

本表只放「能映射到图中真实存在、且信息更完整的疾病节点」的别名。
"""
from collections import defaultdict

# 别名 → 规范疾病节点（canonical）
DISEASE_ALIASES = {
    # 误分类纠正：简称被标成 symptom，纠正到真疾病节点
    "冠心病": "冠状动脉粥样硬化性心脏病",
    # 简称 → 全称
    "甲亢": "甲状腺功能亢进",
    "哮喘": "支气管哮喘",
    "心梗": "心肌梗死",
    "脑梗": "脑梗死",
}

# 反向映射：canonical → [别名]（用于把查询结果归一成用户熟悉的简称）
CANONICAL_TO_ALIASES = defaultdict(list)
for _alias, _canon in DISEASE_ALIASES.items():
    CANONICAL_TO_ALIASES[_canon].append(_alias)


def canonicalize(name):
    """把名称归一到规范节点名；不在表内则原样返回。"""
    return DISEASE_ALIASES.get(name, name)
