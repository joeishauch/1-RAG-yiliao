# -*- coding: utf-8 -*-
"""智能分诊系统 — 医生审核端（端口 7861）。

与患者端 webUI.py 分离：医生通过简单密码进入，查看待人工审核队列，
对高风险草稿做「通过 / 驳回 / 改写」，结果经后端 /v1/chat/review 续跑图，
最终回答会写回 checkpoint，患者端轮询自动拿到结果。

用法:
    python doctor_ui.py
"""
# 导入 Gradio 库，用于构建交互式前端界面
import gradio as gr
# 导入 requests 库，用于发送 HTTP 请求
import requests
# 导入 json 库，用于处理 JSON 数据
import json
# 导入 logging 库，用于记录日志
import logging
# 导入 re 库，用于正则表达式操作
import re

# 设置日志的基本配置，指定日志级别为 INFO，并定义日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 创建一个名为当前模块的日志记录器
logger = logging.getLogger(__name__)

# 后端服务端点 URL（后端 8012 是唯一权威，医生端不直接碰图）
url = "http://localhost:8012/v1/chat/completions"
review_url = "http://localhost:8012/v1/chat/review"
pending_url = "http://localhost:8012/v1/review/pending"
# 定义 HTTP 请求头，指定内容类型为 JSON
headers = {"Content-Type": "application/json"}

# 医生端简单访问密码（后续可挪到 .env 统一管理）
DOCTOR_PASSWORD = "admin"

# 待审队列缓存：thread_id -> 待审项（含 userId/conversationId/risk_level/draft/safety_hits）
queue_cache = {}


def format_response(full_text):
    """与 webUI.format_response 同款：把 <think> 标签转成加粗标题，便于阅读。"""
    text = re.sub(r'<think>', '**思考过程**：\n', full_text or "")
    text = re.sub(r'</think>', '\n\n**最终回复**：\n', text)
    return text.strip()


def _dropdown_choices():
    """由 queue_cache 生成下拉框选项列表 (label, thread_id)。"""
    return [
        (f"[{item.get('risk_level', 'unknown')}] 用户 {(item.get('userId') or '')[:8]} · 会话 {(item.get('conversationId') or '')[:8]}",
         tid)
        for tid, item in queue_cache.items()
    ]


def check_password(password):
    """校验访问密码：通过则显示主界面，否则提示错误。"""
    if password == DOCTOR_PASSWORD:
        return gr.update(visible=True), gr.update(visible=False), ""
    return gr.update(visible=False), gr.update(visible=True), "❌ 密码错误"


def refresh_queue():
    """拉取后端待审队列，更新缓存与下拉选项。"""
    try:
        resp = requests.get(pending_url, timeout=10)
        items = resp.json().get("items", [])
    except Exception as e:
        logger.error(f"刷新队列失败: {e}")
        return gr.update(choices=[], value=None), f"❌ 刷新失败：{e}"
    queue_cache.clear()
    for item in items:
        queue_cache[item.get("thread_id", "")] = item
    choices = _dropdown_choices()
    return gr.update(choices=choices, value=choices[0][1] if choices else None), f"共 {len(choices)} 条待审核"


def show_detail(thread_id):
    """根据选中的待审项，展示风险等级、安全命中与 AI 草稿。"""
    item = queue_cache.get(thread_id)
    if not item:
        return "### 无详情", ""
    risk = item.get("risk_level", "unknown")
    safety_hits = item.get("safety_hits", [])
    risk_md = f"**风险等级：** `{risk}`"
    if safety_hits:
        risk_md += f"\n\n**安全命中：** {', '.join(safety_hits)}"
    return risk_md, item.get("draft", "")


def submit_review(action, revised_answer, thread_id):
    """提交审核结果到后端，续跑图，返回最终回答并刷新队列。"""
    item = queue_cache.get(thread_id)
    if not item:
        return "❌ 请先选择一条待审核项", gr.update()
    payload = {
        "userId": item.get("userId"),
        "conversationId": item.get("conversationId"),
        "action": action,
        "revised_answer": revised_answer or None,
        "comment": None,
        "reviewer": "doctor",
    }
    try:
        resp = requests.post(review_url, headers=headers, data=json.dumps(payload), timeout=180)
        resp_json = resp.json()
        content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", f"审核异常：{resp_json}")
    except Exception as e:
        logger.error(f"审核提交失败: {e}")
        content = f"❌ 审核请求失败：{e}"
    formatted = format_response(content)
    # 审核完成，出缓存并刷新下拉列表
    queue_cache.pop(thread_id, None)
    choices = _dropdown_choices()
    return formatted, gr.update(choices=choices, value=choices[0][1] if choices else None)


# 使用 Gradio Blocks 创建医生端界面
with gr.Blocks(title="智能分诊 · 医生审核端") as demo:
    # 密码登录页（初始可见）
    with gr.Column(visible=True) as login_page:
        gr.Markdown("## 🔐 医生审核端登录")
        password_input = gr.Textbox(label="访问密码", type="password", placeholder="请输入医生端访问密码")
        login_btn = gr.Button("进入", variant="primary")
        login_msg = gr.Markdown("")

    # 主界面（初始隐藏）
    with gr.Column(visible=False) as main_page:
        gr.Markdown("## 🩺 待人工审核队列")
        with gr.Row():
            refresh_btn = gr.Button("刷新队列", variant="primary")
            queue_status = gr.Markdown("")
        # 待审项下拉框（label 显示风险等级 + 用户前缀）
        pending_dropdown = gr.Dropdown(label="待审核项（风险等级 · 用户）", choices=[], interactive=True)
        # 风险等级与安全命中
        risk_md = gr.Markdown()
        # AI 草稿（只读）
        draft_box = gr.Textbox(label="AI 草稿回复", interactive=False, lines=12)
        # 改写内容输入框
        revise_box = gr.Textbox(label="改写内容（选择「改写」时填写）", placeholder="输入改写后的回复…", lines=4)
        with gr.Row():
            approve_btn = gr.Button("✅ 通过", variant="primary")
            reject_btn = gr.Button("❌ 驳回", variant="secondary")
            revise_btn = gr.Button("✏️ 提交改写", variant="primary")
        # 审核结果（最终回答）
        result_box = gr.Textbox(label="审核结果（最终回答）", interactive=False, lines=8)

    # 绑定登录按钮：校验密码，切换页面显示
    login_btn.click(check_password, [password_input], [main_page, login_page, login_msg])
    # 绑定刷新队列按钮
    refresh_btn.click(refresh_queue, None, [pending_dropdown, queue_status])
    # 绑定下拉框选中事件：显示详情
    pending_dropdown.change(show_detail, [pending_dropdown], [risk_md, draft_box])
    # 绑定通过 / 驳回 / 改写按钮：提交审核并刷新队列
    approve_btn.click(lambda tid: submit_review("approve", None, tid), [pending_dropdown], [result_box, pending_dropdown])
    reject_btn.click(lambda tid: submit_review("reject", None, tid), [pending_dropdown], [result_box, pending_dropdown])
    revise_btn.click(lambda rv, tid: submit_review("revise", rv, tid), [revise_box, pending_dropdown], [result_box, pending_dropdown])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
