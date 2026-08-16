# -*- coding: utf-8 -*-
"""知识图谱 schema：节点类型、关系规则表、查询用关系集合。

本模块是纯常量、无副作用，供 kg_builder（建图）与 kg_query（查询）共用。
关系识别采用「有序正则表」——外层按列表顺序遍历，先命中的规则胜出，
因此更具体 / 含反向语义的规则必须排在前面（见各注释）。
"""

# -------------------- 节点类型 --------------------
NODE_DISEASE = "disease"            # 疾病
NODE_SYMPTOM = "symptom"            # 症状
NODE_DRUG = "drug"                  # 药物
NODE_TREATMENT = "treatment"        # 治疗 / 术式
NODE_EXAMINATION = "examination"    # 检查
NODE_DEPARTMENT = "department"      # 就诊科室
NODE_CAUSE = "cause"                # 病因
NODE_ATTRIBUTE = "attribute_value"  # 属性值（发病率/发病部位/多发群体…）

# -------------------- 关系类型（predicate） --------------------
REL_SYMPTOM_OF = "symptom_of"                    # 症状 -> 疾病（多跳第一跳）
REL_HAS_SYMPTOM = "has_symptom"                  # 疾病 -> 症状
REL_HAS_DRUG = "has_drug"                        # 疾病 -> 药物
REL_HAS_TREATMENT = "has_treatment"              # 疾病 -> 治疗
REL_HAS_EXAMINATION = "has_examination"          # 疾病 -> 检查
REL_HAS_CAUSE = "has_cause"                      # 疾病 -> 病因
REL_HAS_COMPLICATION = "has_complication"        # 疾病 -> 疾病（并发症）
REL_BELONGS_TO_DEPT = "belongs_to_department"    # 疾病 -> 科室
REL_HAS_ATTRIBUTE = "has_attribute"              # 疾病 -> 属性值

# -------------------- 关系规则表 --------------------
# 每项 (predicate, head_type, tail_type, [patterns])
# 顺序即优先级（先匹配到的规则胜出）：
#   1. symptom_of 必须最前，避免「是什么疾病的症状」被 has_symptom 的「症状」误吞
#   2. has_drug 在 has_treatment 之前，避免「药物治疗」被归为治疗
#   3. has_symptom 的「临床表现」不会与 has_treatment 冲突（治疗规则不含「表现」）
RELATION_RULES = [
    (REL_SYMPTOM_OF, NODE_SYMPTOM, NODE_DISEASE,
     [r"是什么疾病的症状", r"哪些疾病.{0,3}症状"]),
    (REL_HAS_COMPLICATION, NODE_DISEASE, NODE_DISEASE,
     [r"并发症"]),
    (REL_HAS_DRUG, NODE_DISEASE, NODE_DRUG,
     [r"推荐药", r"用药", r"药物"]),
    (REL_HAS_TREATMENT, NODE_DISEASE, NODE_TREATMENT,
     [r"手术治疗", r"辅助治疗", r"放射治疗", r"化疗", r"治疗方式", r"治疗"]),
    (REL_HAS_SYMPTOM, NODE_DISEASE, NODE_SYMPTOM,
     [r"临床表现", r"症状"]),
    (REL_HAS_EXAMINATION, NODE_DISEASE, NODE_EXAMINATION,
     [r"影像学检查", r"检查"]),
    # 注：has_cause / has_attribute 因答案常为长文本（非实体枚举），规则切分质量差，MVP 排除
    (REL_BELONGS_TO_DEPT, NODE_DISEASE, NODE_DEPARTMENT,
     [r"就诊科室", r"科室"]),
]

# -------------------- 多跳查询用关系集合 --------------------
# 第一跳（从症状出发）：只走 symptom_of
FIRST_HOP_RELATIONS = {REL_SYMPTOM_OF}
# 第二跳（从疾病出发）：疾病能展开出的下游关系（含科室，供分诊参考）
SECOND_HOP_RELATIONS = {
    REL_HAS_DRUG, REL_HAS_TREATMENT, REL_HAS_SYMPTOM,
    REL_HAS_EXAMINATION, REL_HAS_COMPLICATION, REL_BELONGS_TO_DEPT,
}
