# -*- coding: utf-8 -*-
"""全局配置：pydantic-settings 从 .env / 环境变量加载，带类型校验。

加载优先级（pydantic-settings 默认）：环境变量 > .env 文件 > 字段默认值。
字段名即 .env 变量名（大小写不敏感），用法保持旧风格：
    from utils.config import Config
    Config.LLM_TYPE / Config.DB_URI / ...
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """集中管理所有配置。字段带类型，pydantic 自动校验/转换（如 int、str）。"""

    model_config = SettingsConfigDict(
        env_file=".env",            # 从项目根目录的 .env 加载
        env_file_encoding="utf-8",
        extra="ignore",             # 忽略 .env 中未在此定义的变量，避免报错
    )

    # ---- 大模型 ----
    LLM_TYPE: str = "deepseek"      # openai / qwen / oneapi / ollama / deepseek
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DASHSCOPE_API_KEY: str = ""     # 通义 text-embedding-v1 向量
    OPENAI_BASE_URL: str = ""
    OPENAI_API_KEY: str = ""

    # ---- Chroma 向量库 ----
    CHROMADB_DIRECTORY: str = "chromaDB"
    CHROMADB_COLLECTION_NAME: str = "medical_triage"   # 分诊（带科室 label）
    CHROMADB_QA_COLLECTION_NAME: str = "medical_qa"    # 科普问答兜底（百科/知识图谱/好大夫）

    # ---- 知识图谱（GraphRAG 多跳推理）----
    KG_GRAPH_PATH: str = "kg_graph.pkl"   # build_kg.py 规则切分构建的 medical 图缓存

    # ---- 药物禁忌（drug_taboo 工具数据源）----
    DRUG_CONTRA_PATH: str = "drug_contraindications.json"   # extract_drug_contra.py 抽取产物，list[{drug_name, contraindications, source}]

    # ---- 分诊数据清洗 ----
    PER_LABEL_LIMIT: int = 3000   # 均衡抽样：分诊数据每科最多灌入条数（小科室全取、大科室均匀间隔）

    # ---- 日志 ----
    LOG_FILE: str = "output/app.log"
    MAX_BYTES: int = 5 * 1024 * 1024
    BACKUP_COUNT: int = 3

    # ---- PostgreSQL 数据库 ----
    DB_URI: str = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

    # ---- 混合检索（BGE-reranker 重排）----
    RERANKER_MODEL_PATH: str = (
        "D:/ai/xtuner-env/大模型微调项目实战/demo_14/embedding_model/models/"
        "BAAI--bge-reranker-large/snapshots/master"
    )
    HYBRID_BM25_TOP_K: int = 20    # BM25 关键词召回条数
    HYBRID_VEC_TOP_K: int = 20     # 向量召回条数
    RRF_K: int = 60                # RRF 融合常数（rank 偏置）
    RERANK_MAX_LENGTH: int = 512   # cross-encoder 重排的最大输入长度
    RERANKER_BACKEND: str = "onnx" # cross-encoder 后端：onnx(onnxruntime CPU 加速) / torch
    # 分诊检索默认模式：vector(纯向量) / hybrid(BM25+向量RRF) / hybrid_rerank(再加本地BGE重排)
    # 评估结论(2026-08-14)：明确症状上纯向量最准且快，重排 CPU 单条 5-6s，故默认 vector
    TRIAGE_RETRIEVAL_MODE: str = "vector"

    # ---- API 服务 ----
    HOST: str = "0.0.0.0"
    PORT: int = 8012

    # ---- prompt 模板路径（固定，一般无需覆盖）----
    PROMPT_TEMPLATE_TXT_AGENT: str = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TXT_CONSULT: str = "prompts/prompt_template_consult.txt"
    PROMPT_TEMPLATE_TXT_SUMMARIZE: str = "prompts/prompt_template_summarize.txt"
    PROMPT_TEMPLATE_TXT_QA: str = "prompts/prompt_template_qa.txt"  # 科普兜底回答
    PROMPT_TEMPLATE_TXT_KG: str = "prompts/prompt_template_kg.txt"  # 知识图谱症状→疾病推理回答
    PROMPT_TEMPLATE_TXT_DRUG: str = "prompts/prompt_template_drug.txt"  # 药物禁忌回答
    PROMPT_TEMPLATE_TXT_HIS: str = "prompts/prompt_template_his.txt"  # 外部系统（HIS 号源/报告）查询回答

    # ---- MCP 外部系统接入（方向2）----
    MCP_HIS_ENABLED: bool = True   # 是否在 get_tools 时接入外部 HIS MCP 工具（默认开；.env 设 false 可关，加载失败自动降级）

    # ---- 安全合规 / HITL（人工审核）----
    HITL_ENABLED: bool = True               # 高风险链路人工审核开关（默认开启；.env 设 false 可关）
    REVIEW_REJECT_FALLBACK: str = "抱歉，该回复未通过安全审核，已拦截。如症状持续或加重，请及时就医。"
    AUDIT_LOG_PATH: str = "output/audit.jsonl"   # 审计日志（AI 输出 + 人工修改留痕）
    LOW_RISK_SAMPLE_RATE: float = 0.2       # 二期预留：低风险事后抽样审计比例


# 全局单例：模块 import 时即加载 .env 与环境变量
settings = Settings()

# 向后兼容别名：旧代码 `from utils.config import Config` + `Config.XXX` 零改动
# （Config 是实例而非类，实例属性访问同样满足 `Config.XXX` 的读取用法）
Config = settings
