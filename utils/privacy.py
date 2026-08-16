# -*- coding: utf-8 -*-
"""病历脱敏：在入口边界对用户输入做单向掩码，防止 PII 进入持久化记忆与日志。

设计原则：
- 单向掩码：只替换为 ***，不保留原文、不落可逆映射（隐私不可逆）；
- 姓名仅做保守启发式（只匹配「先生/女士/小姐」等罕见称谓），
  刻意避开「患者/同学/小朋友」等医疗高频泛称，避免误伤正常主诉。
"""
import re
import logging

logger = logging.getLogger(__name__)

# 掩码占位符
_MASK = "***"

# 身份证号：18 位（末位可为 X/x）或 15 位纯数字。
# 用 (?<!\d)/(?!\d) 而非 \b 做边界——Python 里中文汉字属于 \w，\b 在中文与数字间不成立。
_ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)|(?<!\d)\d{15}(?!\d)")
# 手机号：1[3-9] 开头共 11 位
_MOBILE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# 姓名：2-3 个中文 + 罕见称谓（医疗语境中「先生/女士/小姐」多用于指代具体人名）
_NAME = re.compile(r"[一-龥]{2,3}(?:先生|女士|小姐)")

# 住址：聚焦「门牌/楼栋/小区」最小地址单元，避免贪婪吃掉大区划前缀
_ADDRESS_PATTERNS = [
    # 楼栋单元：数字 + 号楼/栋/单元/室/层（如 3号楼、2单元、101室）
    re.compile(r"\d{1,4}(?:号楼|栋|单元|室|层)"),
    # 道路门牌：路/街/巷/弄/大道/胡同 + 数字[号]（如 大街1号、中关村路88号）
    re.compile(r"(?:路|街|巷|弄|大道|胡同)\d{1,6}号?"),
    # 小区/公寓名称（如 阳光花园、幸福小区、青年公寓）
    re.compile(r"[一-龥]{2,8}(?:小区|花园|苑|公寓|大厦|广场|新村)"),
]

# 统一脱敏规则表：(正则, 类别)，按序依次替换
_PATTERNS = [
    (_ID_CARD, "身份证"),
    (_MOBILE, "手机号"),
    (_NAME, "姓名"),
]
_PATTERNS += [(p, "住址") for p in _ADDRESS_PATTERNS]


def desensitize(text):
    """对输入文本做单向脱敏。

    Args:
        text: 待脱敏文本。

    Returns:
        tuple: (脱敏后文本, 命中条数)
    """
    if not text:
        return text, 0
    count = 0
    for pattern, _kind in _PATTERNS:
        text, n = pattern.subn(_MASK, text)
        count += n
    if count:
        logger.info(f"入口脱敏：命中 {count} 处敏感信息")
    return text, count
