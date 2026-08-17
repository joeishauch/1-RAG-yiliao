# 导入操作系统接口模块，用于处理文件路径和环境变量
import os
# 用于正则表达式匹配和处理字符串
import re
# 用于JSON数据的序列化和反序列化
import json
# 用于定义异步上下文管理器
from contextlib import asynccontextmanager
# 用于类型提示，定义列表和可选参数
from typing import List, Tuple
# 用于创建Web应用和处理HTTP异常
from fastapi import FastAPI, HTTPException, Depends
# 用于返回JSON和流式响应
from fastapi.responses import JSONResponse, StreamingResponse
# 用于运行FastAPI应用
import uvicorn
# 导入日志模块，用于记录程序运行时的信息
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
# 导入系统模块，用于处理系统相关的操作，如退出程序
import sys
import time
# 导入UUID模块，用于生成唯一标识符
import uuid
# 用于密码哈希（sha256）与随机盐生成（医生/患者账号持久化）
import hashlib
import secrets
# 从typing模块导入类型提示工具
from typing import Optional
# 导入Pydantic的基类和字段定义工具
from pydantic import BaseModel, Field
# 从自定义的库中引入函数
from ragAgent import (
    ToolConfig,
    create_graph,
    save_graph_visualization,
    get_llm,
    get_tools,
    Config,
    ConnectionPool,
    ConnectionPoolError,
    monitor_connection_pool,
)
from langgraph.types import Command
from utils.privacy import desensitize
from utils.safety import check_input_danger, check_output_diagnostic
from utils.audit import write_audit, build_record


# 设置LangSmith环境变量 进行应用跟踪，实时了解应用中的每一步发生了什么
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = ""


# 设置日志基本配置，级别为DEBUG或INFO
logger = logging.getLogger(__name__)
# 设置日志器级别为DEBUG
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
logger.handlers = []  # 清空默认处理器
# 使用ConcurrentRotatingFileHandler
handler = ConcurrentRotatingFileHandler(
    # 日志文件
    Config.LOG_FILE,
    # 日志文件最大允许大小为5MB，达到上限后触发轮转
    maxBytes = Config.MAX_BYTES,
    # 在轮转时，最多保留3个历史日志文件
    backupCount = Config.BACKUP_COUNT
)
# 设置处理器级别为DEBUG
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)


# 定义消息类，用于封装API接口返回数据
# 定义Message类
class Message(BaseModel):
    role: str
    content: str

# 定义ChatCompletionRequest类
class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    userId: Optional[str] = None
    conversationId: Optional[str] = None

# 定义人工审核结果请求类（resume 端点）
class ReviewRequest(BaseModel):
    userId: str
    conversationId: str
    action: str                       # approve / revise / reject / transfer
    revised_answer: Optional[str] = None
    comment: Optional[str] = None
    reviewer: Optional[str] = "human"  # 审核人账号（医生手机号）
    target_department: Optional[str] = None  # transfer 目标科室
    reviewer_name: Optional[str] = None      # 审核医生姓名（追责 + 患者端脱敏展示）
    reviewer_department: Optional[str] = None  # 审核医生所属科室
    reviewer_title: Optional[str] = None      # 审核医生职称（可选）

# 定义待审核响应类（interrupt 命中时返回）
class ReviewPendingResponse(BaseModel):
    status: str = "pending_review"
    thread_id: str
    user_id: str
    risk_level: Optional[str] = None
    draft: str
    safety_hits: List[str] = Field(default_factory=list)
    departments: List[str] = Field(default_factory=list)  # 推荐科室列表（首选为 [0]，供患者端展示审核科室）

# 医生登录请求（管理员填 admin，医生填手机号）
class DoctorLoginRequest(BaseModel):
    account: str
    password: str

# 管理员建号请求（医生不能自助注册）
class DoctorRegisterRequest(BaseModel):
    admin_account: str = "admin"     # 管理员账号（校验建号权限）
    admin_password: str              # 管理员密码
    phone: str                       # 新医生手机号
    password: str                    # 新医生密码
    name: str                        # 医生姓名（追责到人）
    title: Optional[str] = None      # 职称（可选）
    department: str                  # 绑定科室（权限）

# 定义ChatCompletionResponseChoice类
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

# 定义ChatCompletionResponse类
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None


def format_response(response):
    """对输入的文本进行段落分隔、添加适当的换行符，以及在代码块中增加标记，以便生成更具可读性的输出。

    Args:
        response: 输入的文本。

    Returns:
        具有清晰段落分隔的文本。
    """
    # 使用正则表达式 \n{2, }将输入的response按照两个或更多的连续换行符进行分割。这样可以将文本分割成多个段落，每个段落由连续的非空行组成
    paragraphs = re.split(r'\n{2,}', response)
    # 空列表，用于存储格式化后的段落
    formatted_paragraphs = []
    # 遍历每个段落进行处理
    for para in paragraphs:
        # 检查段落中是否包含代码块标记
        if '```' in para:
            # 将段落按照```分割成多个部分，代码块和普通文本交替出现
            parts = para.split('```')
            for i, part in enumerate(parts):
                # 检查当前部分的索引是否为奇数，奇数部分代表代码块
                if i % 2 == 1:  # 这是代码块
                    # 将代码块部分用换行符和```包围，并去除多余的空白字符
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            # 将分割后的部分重新组合成一个字符串
            para = ''.join(parts)
        else:
            # 否则，将句子中的句点后面的空格替换为换行符，以便句子之间有明确的分隔
            para = para.replace('. ', '.\n')
        # 将格式化后的段落添加到formatted_paragraphs列表
        # strip()方法用于移除字符串开头和结尾的空白字符（包括空格、制表符 \t、换行符 \n等）
        formatted_paragraphs.append(para.strip())
    # 将所有格式化后的段落用两个换行符连接起来，以形成一个具有清晰段落分隔的文本
    return '\n\n'.join(formatted_paragraphs)


# 分诊系统 16 个科室（与分诊库 label 完全对齐，含「其他」兜底标签）
DEPARTMENTS = [
    "妇产科", "内科", "皮肤性病科", "儿科", "眼耳鼻喉科", "肿瘤科", "神经科学", "外科",
    "男性健康科", "感染与免疫科", "口腔科", "心理科学", "中医科", "生殖健康科", "急诊科", "其他",
]

# 科室流转：最多移交次数（防踢皮球，达到上限后医生必须 approve/reject）
TRANSFER_MAX = 2

# 医生账号持久化文件（管理员 + 各科室医生，密码 sha256+盐哈希）
DOCTOR_ACCOUNTS_FILE = "output/doctor_accounts.json"
# 科室纠错数据文件（transfer 落盘，攒批重灌分诊库 → 数据飞轮）
TRIAGE_FEEDBACK_FILE = "output/triage_feedback.jsonl"
# 手机号正则（医生/患者账号统一：1 开头，第二位 3-9）
PHONE_RE = re.compile(r"^1[3-9]\d{9}$")

# 医生账号内存缓存：account -> {salt, password_hash, name, title, department, is_admin}
# 启动时从 doctor_accounts.json 加载；管理员建号后写回文件
doctor_accounts = {}


def _hash_password(password: str, salt: str) -> str:
    """密码哈希：sha256(salt + password)，盐为随机 hex。"""
    return hashlib.sha256(f"{salt}{password}".encode("utf-8")).hexdigest()


def _load_doctor_accounts() -> None:
    """启动时加载医生账号；不存在则初始化管理员 admin/admin123。"""
    global doctor_accounts
    if os.path.exists(DOCTOR_ACCOUNTS_FILE):
        try:
            with open(DOCTOR_ACCOUNTS_FILE, "r", encoding="utf-8") as f:
                doctor_accounts = json.load(f)
            return
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"读取医生账号文件失败，回退初始化：{e}")
    # 首启：初始化管理员 admin / admin123
    salt = secrets.token_hex(16)
    doctor_accounts = {
        "admin": {
            "salt": salt,
            "password_hash": _hash_password("admin123", salt),
            "name": "管理员",
            "title": "",
            "department": None,
            "is_admin": True,
        }
    }
    _save_doctor_accounts()


def _save_doctor_accounts() -> None:
    """医生账号写回磁盘（建号 / 改密后调用）。"""
    os.makedirs(os.path.dirname(DOCTOR_ACCOUNTS_FILE), exist_ok=True)
    with open(DOCTOR_ACCOUNTS_FILE, "w", encoding="utf-8") as f:
        json.dump(doctor_accounts, f, ensure_ascii=False, indent=2)


# 待人工审核队列：thread_id -> {thread_id, userId, conversationId, risk_level, draft, safety_hits,
#                               current_department, candidate_departments, transfer_count}
# 由 handle_non_stream_response 入队、review_resume 出队、GET /v1/review/pending 读取
pending_queue = {}


def _append_feedback(record: dict) -> None:
    """科室纠错数据追加到 triage_feedback.jsonl（数据飞轮，攒批重灌分诊库）。"""
    os.makedirs(os.path.dirname(TRIAGE_FEEDBACK_FILE), exist_ok=True)
    with open(TRIAGE_FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_last_ai_content(state: dict) -> Optional[str]:
    """逆序遍历 state 的消息，取最后一条有 content 的 AI 消息。

    resume 后最终 state 里既含草稿也含审核改写，取最后一条 AI 即最终回答。
    """
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
            return m.content
    return None


# 管理 FastAPI 应用生命周期的异步上下文管理器，负责启动和关闭时的初始化与清理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    管理 FastAPI 应用生命周期的异步上下文管理器，负责启动和关闭时的初始化与清理。

    Args:
        app (FastAPI): FastAPI 应用实例。

    Yields:
        None: 在 yield 前完成初始化，yield 后执行清理。

    Raises:
        ConnectionPoolError: 数据库连接池初始化或操作失败时抛出。
        Exception: 其他未预期的异常。
    """
    # 声明全局变量 graph 和 tool_config
    global graph, tool_config
    # 加载医生账号（首启初始化 admin/admin123）
    _load_doctor_accounts()
    # 初始化数据库连接池为 None
    db_connection_pool = None
    try:
        # 调用 get_llm 初始化聊天模型和嵌入模型
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

        # 获取工具列表，基于嵌入模型
        tools = get_tools(llm_embedding)

        # 创建工具配置实例
        tool_config = ToolConfig(tools)

        # 定义数据库连接参数：自动提交、无预准备阈值、5秒超时
        connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 5}
        # 创建数据库连接池：最大20个连接，最小2个活跃连接，超时10秒
        db_connection_pool = ConnectionPool(
            conninfo=Config.DB_URI,
            max_size=20,
            min_size=2,
            kwargs=connection_kwargs,
            timeout=10
        )

        # 尝试打开数据库连接池
        try:
            # 打开连接池以启用数据库连接
            db_connection_pool.open()
            # 记录连接池初始化成功的日志（INFO 级别）
            logger.info("Database connection pool initialized")
            # 记录详细调试日志（DEBUG 级别）
            logger.debug("Database connection pool initialized")
        except Exception as e:
            # 记录连接池打开失败的错误日志
            logger.error(f"Failed to open connection pool: {e}")
            # 抛出自定义连接池异常
            raise ConnectionPoolError(f"无法打开数据库连接池: {str(e)}")

        # 启动连接池监控线程，60秒检查一次，设置为守护线程
        monitor_thread = monitor_connection_pool(db_connection_pool, interval=60)

        # 尝试创建状态图
        try:
            # 使用数据库连接池和模型创建状态图
            graph = create_graph(db_connection_pool, llm_chat, llm_embedding, tool_config)
        except ConnectionPoolError as e:
            # 记录状态图创建失败的错误日志
            logger.error(f"Graph creation failed: {e}")
            # 退出程序，返回状态码 1
            sys.exit(1)

        # 保存状态图的可视化表示
        save_graph_visualization(graph)

    except ConnectionPoolError as e:
        # 捕获并记录连接池相关异常
        logger.error(f"Connection pool error: {e}")
        # 退出程序，返回状态码 1
        sys.exit(1)
    except Exception as e:
        # 捕获并记录其他未预期的异常
        logger.error(f"Unexpected error: {e}")
        # 退出程序，返回状态码 1
        sys.exit(1)

    # yield 表示应用运行期间，初始化完成后进入运行状态
    yield
    # 检查并关闭数据库连接池（清理资源）
    if db_connection_pool and not db_connection_pool.closed:
        # 关闭连接池
        db_connection_pool.close()
        # 记录连接池关闭的日志
        logger.info("Database connection pool closed")
    # 记录服务关闭的日志
    logger.info("The service has been shut down")

# 创建 FastAPI 实例, lifespan参数用于在应用程序生命周期的开始和结束时执行一些初始化或清理工作
app = FastAPI(lifespan=lifespan)


# 处理非流式响应的异步函数，生成并返回完整的响应内容
async def handle_non_stream_response(user_input, graph, tool_config, config, redaction_count=0):
    """
    处理非流式响应的异步函数，生成并返回完整的响应内容。

    Args:
        user_input (str): 用户输入的内容。
        graph: 图对象，用于处理消息流。
        tool_config: 工具配置对象，包含可用工具的名称和定义。
        config (dict): 配置参数，包含线程和用户标识。

    Returns:
        JSONResponse: 包含格式化响应的 JSON 响应对象。
    """
    # 初始化 content 变量，用于存储最终响应内容
    content = None
    # 从运行时配置提取 thread_id / user_id（审计留痕用）
    thread_id = config["configurable"]["thread_id"]
    user_id = config["configurable"]["user_id"]
    try:
        # 启动 graph.stream 处理用户输入，生成事件流
        events = graph.stream({"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0}, config)
        # 遍历事件流中的每个事件
        for event in events:
            # 高风险链路人工审核中断：识别 __interrupt__ 事件，返回 pending 状态
            if "__interrupt__" in event:
                interrupts = event["__interrupt__"]
                payload = interrupts[0].value if interrupts else {}
                write_audit(build_record(
                    event="draft", thread_id=thread_id, user_id=user_id,
                    risk_level=payload.get("risk_level"), draft=payload.get("draft"),
                    redacted=redaction_count > 0, redaction_count=redaction_count,
                ))
                # 推荐科室列表（review 节点 interrupt payload 传入，首选为 [0]）
                departments = payload.get("departments", []) or []
                # 入待审队列（供医生端 GET /v1/review/pending 拉取）
                pending_queue[thread_id] = {
                    "thread_id": thread_id,
                    "userId": config["configurable"]["user_id"],
                    "conversationId": thread_id.split("@@", 1)[1] if "@@" in thread_id else thread_id,
                    "risk_level": payload.get("risk_level"),
                    "draft": payload.get("draft", ""),
                    "safety_hits": payload.get("safety_hits", []),
                    "question": user_input,  # 脱敏后的用户主诉（transfer 纠错数据用）
                    # 科室流转维度：首选科室 + 备选科室 + 移交计数
                    "current_department": departments[0] if departments else None,
                    "candidate_departments": departments[1:],
                    "transfer_count": 0,
                }
                return JSONResponse(content=ReviewPendingResponse(
                    thread_id=thread_id,
                    user_id=user_id,
                    risk_level=payload.get("risk_level"),
                    draft=payload.get("draft", ""),
                    safety_hits=payload.get("safety_hits", []),
                    departments=departments,
                ).model_dump())
            # 遍历事件中的所有值
            for value in event.values():
                # 检查事件值是否包含有效消息列表
                if "messages" not in value or not isinstance(value["messages"], list):
                    # 记录警告日志，跳过无效消息
                    logger.warning("No valid messages in response")
                    continue

                # 获取消息列表中的最后一条消息
                last_message = value["messages"][-1]

                # 检查消息是否包含工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    # 遍历所有工具调用
                    for tool_call in last_message.tool_calls:
                        # 验证工具调用是否为字典且包含名称
                        if isinstance(tool_call, dict) and "name" in tool_call:
                            # 记录工具调用日志
                            logger.info(f"Calling tool: {tool_call['name']}")
                    # 跳过本次循环，继续处理下一事件
                    continue

                # 检查消息是否包含内容
                if hasattr(last_message, "content"):
                    # 将消息内容赋值给 content
                    content = last_message.content

                    # 检查是否为工具输出（基于工具名称）
                    if hasattr(last_message, "name") and last_message.name in tool_config.get_tool_names():
                        # 获取工具名称
                        tool_name = last_message.name
                        # 记录工具输出日志
                        logger.info(f"Tool Output [{tool_name}]: {content}")
                    # 处理大模型输出（非工具消息）
                    else:
                        # 记录最终响应日志
                        logger.info(f"Final Response is: {content}")
                else:
                    # 记录无内容的消息日志，跳过处理
                    logger.info("Message has no content, skipping")
    except ValueError as ve:
        # 捕获并记录值错误
        logger.error(f"Value error in response processing: {ve}")
    except Exception as e:
        # 捕获并记录其他未预期的异常
        logger.error(f"Error processing response: {e}")

    # 出口诊断性表述检测：命中则追加安全提示
    if content:
        diag_hint = check_output_diagnostic(content)
        if diag_hint:
            content = content + "\n\n" + diag_hint

    # 审计：记录最终回答（低/中风险直出，或高风险未开启审核时）
    write_audit(build_record(
        event="final", thread_id=thread_id, user_id=user_id, final_answer=content,
        redacted=redaction_count > 0, redaction_count=redaction_count,
    ))

    # 格式化响应内容，若无内容则返回默认值
    formatted_response = str(format_response(content)) if content else "No response generated"
    # 记录格式化后的响应日志
    logger.info(f"Results for Formatting: {formatted_response}")

    # 构造返回给客户端的响应对象
    try:
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=formatted_response),
                    finish_reason="stop"
                )
            ]
        )
    except Exception as resp_error:
        # 捕获并记录构造响应对象时的异常
        logger.error(f"Error creating response object: {resp_error}")
        # 构造错误响应对象
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content="Error generating response"),
                    finish_reason="error"
                )
            ]
        )

    # 记录发送给客户端的响应内容日志
    logger.info(f"Send response content: \n{response}")
    # 返回 JSON 格式的响应对象
    return JSONResponse(content=response.model_dump())


# 处理流式响应的异步函数，生成并返回流式数据
async def handle_stream_response(user_input, graph, config):
    """
    处理流式响应的异步函数，生成并返回流式数据。

    Args:
        user_input (str): 用户输入的内容。
        graph: 图对象，用于处理消息流。
        config (dict): 配置参数，包含线程和用户标识。

    Returns:
        StreamingResponse: 流式响应对象，媒体类型为 text/event-stream。
    """
    async def generate_stream():
        """
        内部异步生成器函数，用于产生流式响应数据。

        Yields:
            str: 流式数据块，格式为 SSE (Server-Sent Events)。

        Raises:
            Exception: 流生成过程中可能抛出的异常。
        """
        try:
            # 生成唯一的 chunk ID
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            # 调用 graph.stream 获取消息流
            stream_data = graph.stream(
                {"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0},
                config,
                stream_mode="messages"
            )
            # 遍历消息流中的每个数据块
            for message_chunk, metadata in stream_data:
                try:
                    # 获取当前节点名称
                    node_name = metadata.get("langgraph_node") if metadata else None
                    # 仅处理 generate 和 agent 节点
                    if node_name in ["generate", "agent"]:
                        # 获取消息内容，默认空字符串
                        chunk = getattr(message_chunk, 'content', '')
                        # 记录流式数据块日志
                        logger.info(f"Streaming chunk from {node_name}: {chunk}")
                        # 产出流式数据块
                        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                except Exception as chunk_error:
                    # 记录单个数据块处理异常
                    logger.error(f"Error processing stream chunk: {chunk_error}")
                    continue

            # 流结束：兜底检测是否有未处理的中断（HITL），有则产出 pending 事件（完整流式 HITL 归二期）
            try:
                graph_state = graph.get_state(config)
                interrupts = tuple(
                    it for task in (getattr(graph_state, "tasks", ()) or ())
                    for it in (getattr(task, "interrupts", ()) or ())
                )
                if interrupts:
                    payload = interrupts[0].value if interrupts else {}
                    yield f"data: {json.dumps({'status': 'pending_review', 'thread_id': config['configurable']['thread_id'], 'risk_level': payload.get('risk_level'), 'draft': payload.get('draft', '')})}\n\n"
                    return
            except Exception:
                pass

            # 产出流结束标记
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        except Exception as stream_error:
            # 记录流生成过程中的异常
            logger.error(f"Stream generation error: {stream_error}")
            # 产出错误提示
            yield f"data: {json.dumps({'error': 'Stream processing failed'})}\n\n"

    # 返回流式响应对象
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# 依赖注入函数，用于获取 graph 和 tool_config
async def get_dependencies() -> Tuple[any, any]:
    """
    依赖注入函数，用于获取 graph 和 tool_config。

    Returns:
        Tuple: 包含 (graph, tool_config) 的元组。

    Raises:
        HTTPException: 如果 graph 或 tool_config 未初始化，则抛出 500 错误。
    """
    if not graph or not tool_config:
        raise HTTPException(status_code=500, detail="Service not initialized")
    return graph, tool_config


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, dependencies: Tuple[any, any] = Depends(get_dependencies)):
    """接收来自前端的请求数据进行业务的处理。

    Args:
        request: 请求参数。

    Returns:
        标准的Python字典。
    """
    try:
        graph, tool_config = dependencies
        # 检查request是否有效
        if not request.messages or not request.messages[-1].content:
            logger.error("Invalid request: Empty or invalid messages")
            raise HTTPException(status_code=400, detail="Messages cannot be empty or invalid")
        user_input = request.messages[-1].content
        logger.info(f"The user's user_input is: {user_input}")

        # 入口脱敏：PII 永不进入持久化记忆/提示词/审计日志
        user_input, redaction_count = desensitize(user_input)

        # 入口危险信号拦截（脱敏后文本，危险词不受脱敏影响；不进图）
        blocked = check_input_danger(user_input)
        if blocked:
            write_audit(build_record(
                event="block",
                thread_id=f"{getattr(request, 'userId', 'unknown')}@@{getattr(request, 'conversationId', 'default')}",
                user_id=getattr(request, 'userId', 'unknown'),
                user_input=user_input,
                redacted=redaction_count > 0,
                redaction_count=redaction_count,
            ))
            return JSONResponse(content=ChatCompletionResponse(
                choices=[ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=blocked),
                    finish_reason="stop",
                )]
            ).model_dump())

        # 定义运行时配置，包含线程ID和用户ID，使用默认值防止未定义
        config = {
            "configurable": {
                "thread_id": f"{getattr(request, 'userId', 'unknown')}@@{getattr(request, 'conversationId', 'default')}",
                "user_id": getattr(request, 'userId', 'unknown')
            }
        }

        # 调用流式输出
        if request.stream:
            return await handle_stream_response(user_input, graph, config)
        # 调用非流式输出
        return await handle_non_stream_response(user_input, graph, tool_config, config, redaction_count=redaction_count)

    except Exception as e:
        logger.error(f"Error handling chat completion:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/review")
async def review_resume(request: ReviewRequest, dependencies: Tuple[any, any] = Depends(get_dependencies)):
    """接收医生审核结果，用 Command(resume=...) 续跑中断的图，返回最终回答。"""
    try:
        graph, tool_config = dependencies
        thread_id = f"{request.userId}@@{request.conversationId}"
        config = {"configurable": {"thread_id": thread_id, "user_id": request.userId}}

        # 科室移交（transfer）：不 resume 图，只改队列当前科室 + 落纠错数据，等目标科室再审
        if request.action == "transfer":
            item = pending_queue.get(thread_id)
            if not item:
                raise HTTPException(status_code=404, detail="待审项不存在或已处理")
            if item.get("transfer_count", 0) >= TRANSFER_MAX:
                raise HTTPException(status_code=400, detail=f"已达最大移交次数（{TRANSFER_MAX}），请直接通过或驳回")
            target = request.target_department
            if not target:
                raise HTTPException(status_code=400, detail="移交需指定 target_department")
            from_dept = item.get("current_department")
            item["transfer_count"] = item.get("transfer_count", 0) + 1
            item["current_department"] = target
            # 落科室纠错数据（数据飞轮）
            _append_feedback({
                "question": item.get("question", ""),
                "from_dept": from_dept,
                "to_dept": target,
                "doctor_phone": request.reviewer,
                "doctor_name": request.reviewer_name or "",
                "thread_id": thread_id,
                "ts": int(time.time()),
            })
            write_audit(build_record(
                event="transfer", thread_id=thread_id, user_id=request.userId,
                action="transfer", from_department=from_dept, to_department=target,
                reviewer=request.reviewer, comment=request.comment,
            ))
            return JSONResponse(content={"status": "transferred", "to_department": target})

        decision = {
            "action": request.action,
            "revised_answer": request.revised_answer,
            "comment": request.comment,
            "reviewer": request.reviewer,
            "reviewer_name": request.reviewer_name,
            "reviewer_department": request.reviewer_department,
            "reviewer_title": request.reviewer_title,
        }

        # 续跑：无需重传 input，checkpoint 已存 state；返回最终 state
        final_state = graph.invoke(Command(resume=decision), config)
        final_content = _extract_last_ai_content(final_state)
        # 审核完成，出待审队列
        pending_queue.pop(thread_id, None)

        # 出口诊断性表述检测
        if final_content:
            diag_hint = check_output_diagnostic(final_content)
            if diag_hint:
                final_content = final_content + "\n\n" + diag_hint

        write_audit(build_record(
            event="review_decision", thread_id=thread_id, user_id=request.userId,
            action=request.action, revised_answer=request.revised_answer,
            reviewer=request.reviewer, comment=request.comment,
        ))
        # 从最终 state 取风险等级，补进 final 审计记录，便于追溯高中低风险
        risk_level = final_state.get("risk_level") if isinstance(final_state, dict) else None
        write_audit(build_record(
            event="final", thread_id=thread_id, user_id=request.userId,
            risk_level=risk_level, final_answer=final_content,
        ))

        return JSONResponse(content=ChatCompletionResponse(
            choices=[ChatCompletionResponseChoice(
                index=0,
                message=Message(role="assistant", content=final_content or Config.REVIEW_REJECT_FALLBACK),
                finish_reason="stop",
            )]
        ).model_dump())
    except Exception as e:
        logger.error(f"Error handling review resume: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/review/pending")
async def review_pending(department: Optional[str] = None, dependencies: Tuple[any, any] = Depends(get_dependencies)):
    """返回待人工审核队列（医生端拉取）；department 非空时仅返回该科室待审项。"""
    items = list(pending_queue.values())
    if department:
        items = [it for it in items if it.get("current_department") == department]
    return JSONResponse(content={"items": items})


@app.post("/v1/doctor/login")
async def doctor_login(request: DoctorLoginRequest):
    """医生/管理员登录：校验密码，返回身份信息（is_admin/department/name/title）。"""
    acct = doctor_accounts.get(request.account)
    if not acct or _hash_password(request.password, acct["salt"]) != acct["password_hash"]:
        return JSONResponse(content={"ok": False, "detail": "账号或密码错误"})
    return JSONResponse(content={
        "ok": True,
        "account": request.account,
        "is_admin": acct.get("is_admin", False),
        "department": acct.get("department"),
        "name": acct.get("name"),
        "title": acct.get("title"),
    })


@app.post("/v1/doctor/register")
async def doctor_register(request: DoctorRegisterRequest):
    """管理员建号（医生不能自助注册）；账号为手机号，绑定科室 + 姓名（追责）。"""
    admin = doctor_accounts.get(request.admin_account)
    if not admin or not admin.get("is_admin"):
        return JSONResponse(content={"ok": False, "detail": "无建号权限"}, status_code=403)
    if _hash_password(request.admin_password, admin["salt"]) != admin["password_hash"]:
        return JSONResponse(content={"ok": False, "detail": "管理员密码错误"}, status_code=403)
    if not PHONE_RE.match(request.phone):
        return JSONResponse(content={"ok": False, "detail": "账号须为 11 位手机号（1 开头）"}, status_code=400)
    if not request.name.strip():
        return JSONResponse(content={"ok": False, "detail": "姓名必填"}, status_code=400)
    if request.department not in DEPARTMENTS:
        return JSONResponse(content={"ok": False, "detail": f"科室须为 {DEPARTMENTS} 之一"}, status_code=400)
    if request.phone in doctor_accounts:
        return JSONResponse(content={"ok": False, "detail": "该手机号已存在"}, status_code=400)
    salt = secrets.token_hex(16)
    doctor_accounts[request.phone] = {
        "salt": salt,
        "password_hash": _hash_password(request.password, salt),
        "name": request.name.strip(),
        "title": request.title or "",
        "department": request.department,
        "is_admin": False,
    }
    _save_doctor_accounts()
    return JSONResponse(content={"ok": True, "phone": request.phone})


@app.get("/v1/chat/state")
async def chat_state(userId: str, conversationId: str, dependencies: Tuple[any, any] = Depends(get_dependencies)):
    """查询会话最新状态：待审核（pending）或已完成（done，含最终回答）。"""
    graph, _ = dependencies
    thread_id = f"{userId}@@{conversationId}"
    config = {"configurable": {"thread_id": thread_id, "user_id": userId}}
    try:
        snap = graph.get_state(config)
        # langgraph 0.2.74 的 StateSnapshot 没有 interrupts 字段，interrupt 信息在 tasks[].interrupts 里
        interrupts = tuple(
            it for task in (getattr(snap, "tasks", ()) or ())
            for it in (getattr(task, "interrupts", ()) or ())
        )
        if interrupts:
            return JSONResponse(content={"status": "pending"})
        values = getattr(snap, "values", {}) or {}
        content = _extract_last_ai_content(values)
        if content:
            return JSONResponse(content={"status": "done", "content": content})
        return JSONResponse(content={"status": "done", "content": "（无回复）"})
    except Exception as e:
        logger.error(f"Error querying chat state: {e}")
        return JSONResponse(content={"status": "error", "detail": str(e)})


if __name__ == "__main__":
    logger.info(f"Start the server on port {Config.PORT}")
    # uvicorn是一个用于运行ASGI应用的轻量级、超快速的ASGI服务器实现
    # 用于部署基于FastAPI框架的异步PythonWeb应用程序
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)


