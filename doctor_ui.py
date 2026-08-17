# -*- coding: utf-8 -*-
"""智能分诊系统 — 医生审核端（端口 7861）。

与患者端 webUI.py 分离：医生用「手机号 + 密码」登录，按绑定科室查看待审核队列，
对高风险草稿做「通过 / 驳回 / 改写 / 移交」，结果经后端 /v1/chat/review 续跑图，
最终回答写回 checkpoint，患者端轮询自动拿到结果。

管理员（admin）额外拥有「账号管理」权限：创建医生账号（手机号 + 姓名 + 职称 + 科室），
医生不能自助注册。账号与密码哈希持久化在后端 output/doctor_accounts.json。

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
login_url = "http://localhost:8012/v1/doctor/login"
register_url = "http://localhost:8012/v1/doctor/register"
list_url = "http://localhost:8012/v1/doctor/list"
delete_url = "http://localhost:8012/v1/doctor/delete"
update_url = "http://localhost:8012/v1/doctor/update"
reset_request_url = "http://localhost:8012/v1/doctor/reset_request"
reset_confirm_url = "http://localhost:8012/v1/doctor/reset_confirm"
# 定义 HTTP 请求头，指定内容类型为 JSON
headers = {"Content-Type": "application/json"}

# 分诊系统 16 个科室（与后端 main.DEPARTMENTS 对齐）
DEPARTMENTS = [
    "妇产科", "内科", "皮肤性病科", "儿科", "眼耳鼻喉科", "肿瘤科", "神经科学", "外科",
    "男性健康科", "感染与免疫科", "口腔科", "心理科学", "中医科", "生殖健康科", "急诊科", "其他",
]
# 科室流转：最多移交次数（与后端 TRANSFER_MAX 对齐，用于展示）
TRANSFER_MAX = 2

# 当前登录会话（内存）：account/password 仅本次会话内用于二次鉴权与审核署名
session = {
    "account": "",
    "password": "",
    "is_admin": False,
    "department": None,
    "name": "",
    "title": "",
}

# 待审队列缓存：thread_id -> 待审项（含 userId/conversationId/risk_level/draft/safety_hits/current_department）
queue_cache = {}

# 医生名单缓存：phone -> {name, title, department}，管理员刷新名单时更新，选中时回填编辑表单
doctor_cache = {}


def format_response(full_text):
    """与 webUI.format_response 同款：把 <think> 标签转成加粗标题，便于阅读。"""
    text = re.sub(r'<think>', '**思考过程**：\n', full_text or "")
    text = re.sub(r'</think>', '\n\n**最终回复**：\n', text)
    return text.strip()


def _dropdown_choices():
    """由 queue_cache 生成下拉框选项列表 (label, thread_id)。"""
    return [
        (f"[{item.get('current_department') or '未定科室'}] {item.get('risk_level', 'unknown')} · 用户 {(item.get('userId') or '')[:8]}",
         tid)
        for tid, item in queue_cache.items()
    ]


def do_login(account, password):
    """登录：校验账号密码，成功后切换主界面并刷新身份显示。"""
    payload = {"account": account, "password": password}
    try:
        resp = requests.post(login_url, headers=headers, data=json.dumps(payload), timeout=10)
        data = resp.json()
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), "", f"❌ 登录失败：{e}"
    if not data.get("ok"):
        return gr.update(), gr.update(), gr.update(), "", f"❌ {data.get('detail', '登录失败')}"

    # 保存登录态
    session["account"] = data.get("account")
    session["password"] = password
    session["is_admin"] = bool(data.get("is_admin"))
    session["department"] = data.get("department")
    session["name"] = data.get("name") or ""
    session["title"] = data.get("title") or ""

    identity = "管理员（可管理账号 + 查看全科室队列）" if session["is_admin"] \
        else f"{session['department']} · {session['name']}"
    admin_visible = gr.update(visible=True) if session["is_admin"] else gr.update(visible=False)
    return (
        gr.update(visible=True),      # main_page
        gr.update(visible=False),     # login_page
        admin_visible,                # admin_page（仅管理员）
        f"**当前身份：** {identity}",
        "",                           # login_msg 清空
    )


def refresh_queue():
    """拉取待审队列：管理员看全部，医生只看自己绑定科室。"""
    if not session.get("account"):
        return gr.update(choices=[]), "❌ 请先登录"
    params = None
    if not session.get("is_admin") and session.get("department"):
        params = {"department": session["department"]}
    try:
        resp = requests.get(pending_url, params=params, timeout=10)
        items = resp.json().get("items", [])
    except Exception as e:
        logger.error(f"刷新队列失败: {e}")
        return gr.update(choices=[]), f"❌ 刷新失败：{e}"
    queue_cache.clear()
    for item in items:
        queue_cache[item.get("thread_id", "")] = item
    choices = _dropdown_choices()
    return gr.update(choices=choices, value=choices[0][1] if choices else None), f"共 {len(choices)} 条待审核"


def show_detail(thread_id):
    """根据选中的待审项，展示风险等级、当前科室、备选科室、安全命中与 AI 草稿。"""
    item = queue_cache.get(thread_id)
    if not item:
        return "### 无详情", ""
    risk = item.get("risk_level", "unknown")
    dept = item.get("current_department") or "未定"
    candidates = item.get("candidate_departments") or []
    transfer_count = item.get("transfer_count", 0)
    safety_hits = item.get("safety_hits", [])
    md = f"**风险等级：** `{risk}`\n\n**当前科室：** {dept}"
    if candidates:
        md += f"\n\n**备选科室：** {'、'.join(candidates)}"
    md += f"\n\n**移交次数：** {transfer_count} / {TRANSFER_MAX}"
    if safety_hits:
        md += f"\n\n**安全命中：** {', '.join(safety_hits)}"
    return md, item.get("draft", "")


def _review_payload(item, action, revised_answer=None, target_department=None):
    """组装审核请求体：携带真实审核医生身份（追责 + 患者端署名）。"""
    return {
        "userId": item.get("userId"),
        "conversationId": item.get("conversationId"),
        "action": action,
        "revised_answer": revised_answer or None,
        "comment": None,
        "target_department": target_department,
        "reviewer": session.get("account") or "doctor",
        "reviewer_name": session.get("name") or "",
        "reviewer_department": session.get("department") or "",
        "reviewer_title": session.get("title") or "",
    }


def submit_review(action, revised_answer, thread_id):
    """提交审核结果（通过/驳回/改写）到后端，续跑图，返回最终回答并刷新队列。"""
    item = queue_cache.get(thread_id)
    if not item:
        return "❌ 请先选择一条待审核项", gr.update()
    payload = _review_payload(item, action, revised_answer=revised_answer)
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


def transfer_item(target_department, thread_id):
    """移交给目标科室：不 resume 图，后端改队列当前科室并落纠错数据。"""
    item = queue_cache.get(thread_id)
    if not item:
        return "❌ 请先选择一条待审核项", gr.update()
    if not target_department:
        return "❌ 请选择移交目标科室", gr.update()
    payload = _review_payload(item, "transfer", target_department=target_department)
    try:
        resp = requests.post(review_url, headers=headers, data=json.dumps(payload), timeout=30)
        data = resp.json()
    except Exception as e:
        logger.error(f"移交失败: {e}")
        return f"❌ 移交失败：{e}", gr.update()
    if data.get("status") == "transferred":
        msg = f"✅ 已移交给「{data.get('to_department')}」"
    else:
        msg = f"❌ 移交失败：{data.get('detail', data)}"
    # 移出本端缓存（该科队列已不包含此项），刷新下拉列表
    queue_cache.pop(thread_id, None)
    choices = _dropdown_choices()
    return msg, gr.update(choices=choices, value=choices[0][1] if choices else None)


def register_doctor(name, phone, password, title, department):
    """管理员建号：手机号 + 姓名 + 职称 + 科室，绑定科室权限。"""
    if not (name and phone and password and department):
        return "❌ 姓名 / 手机号 / 密码 / 科室均必填"
    payload = {
        "admin_account": session.get("account"),
        "admin_password": session.get("password"),
        "phone": phone,
        "password": password,
        "name": name,
        "title": title or None,
        "department": department,
    }
    try:
        resp = requests.post(register_url, headers=headers, data=json.dumps(payload), timeout=10)
        data = resp.json()
    except Exception as e:
        return f"❌ 建号失败：{e}"
    if data.get("ok"):
        return f"✅ 已创建医生账号 {phone}（{department} · {name}）"
    return f"❌ {data.get('detail', '建号失败')}"


def list_doctors():
    """管理员拉取全部医生名单，返回 Dataframe 表格、下拉选项与状态。"""
    if not session.get("is_admin"):
        return gr.update(), gr.update(), "❌ 仅管理员可查看名单"
    params = {"admin_account": session.get("account"), "admin_password": session.get("password")}
    try:
        resp = requests.get(list_url, params=params, timeout=10)
        data = resp.json()
    except Exception as e:
        return gr.update(), gr.update(), f"❌ 拉取失败：{e}"
    if not data.get("ok"):
        return gr.update(), gr.update(), f"❌ {data.get('detail', '拉取失败')}"
    doctors = data.get("doctors", [])
    rows = [
        [d["phone"], d["name"], d["title"] or "—", d["department"] or "管理员", "是" if d["is_admin"] else "否"]
        for d in doctors
    ]
    # 管理员不参与增删改查；其余医生填入下拉 + 缓存（选中时回填编辑表单）
    doctor_cache.clear()
    phones = []
    for d in doctors:
        if not d["is_admin"]:
            phones.append(d["phone"])
            doctor_cache[d["phone"]] = {"name": d["name"], "title": d["title"], "department": d["department"]}
    return (
        gr.update(value=rows),
        gr.update(choices=phones, value=phones[0] if phones else None),
        f"共 {len(doctors)} 个账号（含管理员），医生 {len(phones)} 人",
    )


def fill_edit_form(phone):
    """选中医生时，把姓名/职称/科室回填到编辑表单。"""
    d = doctor_cache.get(phone)
    if not d:
        return gr.update(), gr.update(), gr.update()
    return gr.update(value=d["name"]), gr.update(value=d["title"]), gr.update(value=d["department"])


def update_doctor(phone, name, title, department):
    """管理员改医生资料（姓名/职称/科室）。"""
    if not phone:
        return "❌ 请先选择要编辑的医生"
    if not name or not department:
        return "❌ 姓名 / 科室必填"
    payload = {
        "admin_account": session.get("account"),
        "admin_password": session.get("password"),
        "phone": phone,
        "name": name,
        "title": title or None,
        "department": department,
    }
    try:
        resp = requests.post(update_url, headers=headers, data=json.dumps(payload), timeout=10)
        data = resp.json()
    except Exception as e:
        return f"❌ 更新失败：{e}"
    if data.get("ok"):
        return f"✅ 已更新 {phone}"
    return f"❌ {data.get('detail', '更新失败')}"


def delete_doctor(phone):
    """管理员删除医生账号。"""
    if not phone:
        return "❌ 请先选择要删除的医生"
    payload = {
        "admin_account": session.get("account"),
        "admin_password": session.get("password"),
        "phone": phone,
    }
    try:
        resp = requests.post(delete_url, headers=headers, data=json.dumps(payload), timeout=10)
        data = resp.json()
    except Exception as e:
        return f"❌ 删除失败：{e}"
    if data.get("ok"):
        return f"✅ 已删除 {phone}"
    return f"❌ {data.get('detail', '删除失败')}"


def request_reset(phone):
    """管理员发起密码重置：返回一次性链接（本地模拟短信下发）。"""
    if not phone:
        return "❌ 请先选择要重置密码的医生"
    payload = {
        "admin_account": session.get("account"),
        "admin_password": session.get("password"),
        "phone": phone,
    }
    try:
        resp = requests.post(reset_request_url, headers=headers, data=json.dumps(payload), timeout=10)
        data = resp.json()
    except Exception as e:
        return f"❌ 重置申请失败：{e}"
    if data.get("ok"):
        return (
            f"✅ 已生成一次性重置链接（{data.get('expires_in')} 秒内有效，用后即焚）：\n\n"
            f"🔗 完整链接：{data.get('link')}\n\n"
            f"🔑 重置 token（复制这串去登录页「忘记密码」输入）：\n`{data.get('reset_token')}`\n\n"
            f"请转发给医生本人，由其自行设定新密码。"
        )
    return f"❌ {data.get('detail', '重置申请失败')}"


def confirm_reset(token, new_password):
    """医生自助：用一次性 token 设定新密码（登录页「忘记密码」入口）。"""
    token = (token or "").strip()
    # 容错：用户可能直接粘贴了完整链接，自动提取 reset_token= 后面的那串
    if "reset_token=" in token:
        token = token.split("reset_token=", 1)[1].split("&")[0].strip()
    if not token:
        return "❌ 请输入重置 token（链接里 reset_token= 后面的那串）"
    if not new_password or len(new_password) < 6:
        return "❌ 新密码至少 6 位"
    payload = {"token": token, "new_password": new_password}
    try:
        resp = requests.post(reset_confirm_url, headers=headers, data=json.dumps(payload), timeout=10)
        data = resp.json()
    except Exception as e:
        return f"❌ 重置失败：{e}"
    if data.get("ok"):
        return f"✅ 密码已重置，请用手机号 + 新密码登录（账号 {data.get('phone')}）"
    return f"❌ {data.get('detail', '重置失败')}"


# 使用 Gradio Blocks 创建医生端界面
with gr.Blocks(title="智能分诊 · 医生审核端") as demo:
    # 登录页（初始可见）
    with gr.Column(visible=True) as login_page:
        gr.Markdown("## 🔐 医生审核端登录")
        account_input = gr.Textbox(label="账号", placeholder="手机号（医生）/ admin（管理员）")
        password_input = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
        login_btn = gr.Button("登录", variant="primary")
        login_msg = gr.Markdown("")
        # 忘记密码：管理员发起重置后，医生用一次性 token 自助设密
        with gr.Accordion("忘记密码？", open=False):
            reset_token_input = gr.Textbox(label="重置 token（管理员发起，由短信/线下转发）")
            reset_new_pw = gr.Textbox(label="新密码", type="password")
            reset_confirm_btn = gr.Button("设定新密码", variant="secondary")
            reset_confirm_msg = gr.Markdown("")

    # 主界面（初始隐藏）
    with gr.Column(visible=False) as main_page:
        gr.Markdown("## 🩺 待人工审核队列")
        identity_md = gr.Markdown("")

        # 账号管理（仅管理员可见）
        with gr.Column(visible=False) as admin_page:
            gr.Markdown("### 👤 账号管理（管理员建号）")
            with gr.Row():
                new_name = gr.Textbox(label="医生姓名", placeholder="王伟")
                new_phone = gr.Textbox(label="手机号（账号）", placeholder="13800138000")
            with gr.Row():
                new_password = gr.Textbox(label="初始密码", type="password")
                new_title = gr.Textbox(label="职称（可选）", placeholder="主治医师")
            new_department = gr.Dropdown(label="绑定科室（权限）", choices=DEPARTMENTS, interactive=True)
            register_btn = gr.Button("创建账号", variant="primary")
            register_msg = gr.Markdown("")

            # 医生名单（增删改查 + 密码重置）
            gr.Markdown("---\n### 📋 医生名单（增删改查）")
            doctor_list_btn = gr.Button("刷新名单", variant="secondary")
            doctor_table = gr.Dataframe(
                headers=["手机号", "姓名", "职称", "科室", "管理员"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False,
            )
            manage_phone = gr.Dropdown(label="选择医生（手机号）", choices=[], interactive=True)
            with gr.Row():
                edit_name = gr.Textbox(label="姓名")
                edit_title = gr.Textbox(label="职称")
                edit_department = gr.Dropdown(label="科室", choices=DEPARTMENTS, interactive=True)
            with gr.Row():
                save_edit_btn = gr.Button("💾 保存修改", variant="primary")
                delete_btn = gr.Button("🗑 删除账号", variant="secondary")
                reset_btn = gr.Button("🔗 重置密码", variant="secondary")
            manage_msg = gr.Markdown("")

        with gr.Row():
            refresh_btn = gr.Button("刷新队列", variant="primary")
            queue_status = gr.Markdown("")
        # 待审项下拉框（label 显示科室 + 风险等级 + 用户前缀）
        pending_dropdown = gr.Dropdown(label="待审核项（科室 · 风险等级 · 用户）", choices=[], interactive=True)
        # 风险等级 / 当前科室 / 备选科室 / 安全命中
        risk_md = gr.Markdown()
        # AI 草稿（只读）
        draft_box = gr.Textbox(label="AI 草稿回复", interactive=False, lines=12)
        # 改写内容输入框
        revise_box = gr.Textbox(label="改写内容（选择「改写」时填写）", placeholder="输入改写后的回复…", lines=4)
        # 移交：选目标科室（备选列表 + 手动下拉）
        with gr.Row():
            transfer_dropdown = gr.Dropdown(label="移交给科室", choices=DEPARTMENTS, interactive=True)
            transfer_btn = gr.Button("🔁 移交给该科室", variant="secondary")
        with gr.Row():
            approve_btn = gr.Button("✅ 通过", variant="primary")
            reject_btn = gr.Button("❌ 驳回", variant="secondary")
            revise_btn = gr.Button("✏️ 提交改写", variant="primary")
        # 审核结果（最终回答）
        result_box = gr.Textbox(label="审核结果（最终回答）", interactive=False, lines=8)

    # 绑定登录按钮：校验账号密码，切换页面显示，随后自动刷新队列
    login_btn.click(
        do_login, [account_input, password_input],
        [main_page, login_page, admin_page, identity_md, login_msg]
    ).then(refresh_queue, None, [pending_dropdown, queue_status])
    # 绑定刷新队列按钮
    refresh_btn.click(refresh_queue, None, [pending_dropdown, queue_status])
    # 绑定下拉框选中事件：显示详情
    pending_dropdown.change(show_detail, [pending_dropdown], [risk_md, draft_box])
    # 绑定通过 / 驳回 / 改写按钮：提交审核并刷新队列
    approve_btn.click(lambda tid: submit_review("approve", None, tid), [pending_dropdown], [result_box, pending_dropdown])
    reject_btn.click(lambda tid: submit_review("reject", None, tid), [pending_dropdown], [result_box, pending_dropdown])
    revise_btn.click(lambda rv, tid: submit_review("revise", rv, tid), [revise_box, pending_dropdown], [result_box, pending_dropdown])
    # 绑定移交按钮
    transfer_btn.click(transfer_item, [transfer_dropdown, pending_dropdown], [result_box, pending_dropdown])
    # 绑定管理员建号按钮
    register_btn.click(register_doctor, [new_name, new_phone, new_password, new_title, new_department], [register_msg])
    # 医生名单：刷新 / 选中回填 / 保存 / 删除 / 重置密码
    doctor_list_btn.click(list_doctors, None, [doctor_table, manage_phone, manage_msg])
    manage_phone.change(fill_edit_form, [manage_phone], [edit_name, edit_title, edit_department])
    save_edit_btn.click(update_doctor, [manage_phone, edit_name, edit_title, edit_department], [manage_msg])
    delete_btn.click(delete_doctor, [manage_phone], [manage_msg])
    reset_btn.click(request_reset, [manage_phone], [manage_msg])
    # 忘记密码：一次性 token 换新密码
    reset_confirm_btn.click(confirm_reset, [reset_token_input, reset_new_pw], [reset_confirm_msg])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
