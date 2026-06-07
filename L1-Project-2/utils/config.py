# config.py
import os

class Config:
    """统一的配置类，集中管理所有常量"""
    # prompt文件路径
    PROMPT_TEMPLATE_TXT_AGENT = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TXT_GRADE = "prompts/prompt_template_grade.txt"
    PROMPT_TEMPLATE_TXT_REWRITE = "prompts/prompt_template_rewrite.txt"
    PROMPT_TEMPLATE_TXT_GENERATE = "prompts/prompt_template_generate.txt"

    # Chroma 数据库配置
    CHROMADB_DIRECTORY = "chromaDB"
    CHROMADB_COLLECTION_NAME = "demo001"

    # 日志持久化存储
    LOG_FILE = "output/app.log"
    MAX_BYTES=5*1024*1024,
    BACKUP_COUNT=3

    # 数据库 URI，默认值（与 docker-compose.yml 中的 POSTGRES_DB 保持一致）
    DB_URI = os.getenv("DB_URI", "postgresql://postgres:postgres@localhost:5432/langgraph_db?sslmode=disable")
    # ("DB_URI", "postgresql://postgres（用户名）:postgres（密码）@localhost:5432/langgraph_db?sslmode=disable")

  	# 大模型选择：openai, qwen, oneapi, ollama, deepseek
    LLM_TYPE = "deepseek"

    # API服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8012