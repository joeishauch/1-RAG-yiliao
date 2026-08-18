# -*- coding: utf-8 -*-
"""模拟「医院信息系统」MCP Server（方向2：外部系统，演示 Agent 通过 MCP 接入第三方）。

提供查科室 / 查号源 / 查检查报告，全部 mock 数据，用于演示 MCP 协议接入。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastmcp import FastMCP

mcp = FastMCP("hospital-his")

_DEPARTMENTS = {
    "耳鼻喉科": {"医生数": 12, "剩余号源": 8, "坐诊时间": "周一至周六 8:00-17:00"},
    "内科": {"医生数": 20, "剩余号源": 15, "坐诊时间": "每日 8:00-17:00"},
    "儿科": {"医生数": 9, "剩余号源": 3, "坐诊时间": "每日 8:00-20:00"},
    "皮肤科": {"医生数": 7, "剩余号源": 0, "坐诊时间": "周一至周五 8:00-17:00"},
}


@mcp.tool()
def query_department(dept_name: str) -> str:
    """查询科室信息：医生数量、剩余号源、坐诊时间。"""
    info = _DEPARTMENTS.get(dept_name)
    if not info:
        return f"未找到科室「{dept_name}」"
    return f"科室「{dept_name}」：医生 {info['医生数']} 人，剩余号源 {info['剩余号源']}，{info['坐诊时间']}"


@mcp.tool()
def query_registration(dept_name: str) -> str:
    """查询科室挂号/号源情况。"""
    info = _DEPARTMENTS.get(dept_name)
    if not info:
        return f"未找到科室「{dept_name}」"
    n = info["剩余号源"]
    if n > 5:
        return f"「{dept_name}」当前剩余号源 {n} 个（号源充足，可预约）"
    if n > 0:
        return f"「{dept_name}」当前剩余号源 {n} 个（号源紧张，建议尽快）"
    return f"「{dept_name}」今日号源已满"


@mcp.tool()
def query_lab_report(patient_id: str) -> str:
    """根据患者ID查询最近一次检查报告状态。"""
    return f"患者 {patient_id} 最近检查报告：血常规（已出，2 项指标偏高）、CT（待审）"


if __name__ == "__main__":
    mcp.run(show_banner=False)  # 关闭 fastmcp 3.x 启动 banner