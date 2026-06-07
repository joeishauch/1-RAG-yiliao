import os
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型配置字典
MODEL_CONFIGS = {
    "openai": {
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "chat_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small"
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "chat_model": "qwen-max",
        "embedding_model": "text-embedding-v1"
    },
    "oneapi": {
        "base_url": "http://139.224.72.218:3000/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "chat_model": "qwen-max",
        "embedding_model": "text-embedding-v1"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "chat_model": "qwen2.5:32b",
        "embedding_model": "bge-m3:latest"
    },
    "deepseek": {
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "chat_model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "embedding_model": "text-embedding-v1",
        "embedding_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embedding_api_key": os.getenv("DASHSCOPE_API_KEY")
    }
}

# 默认配置
DEFAULT_LLM_TYPE = "deepseek"
DEFAULT_TEMPERATURE = 0.0


class LLMInitializationError(Exception):
    """自定义异常类用于LLM初始化错误"""
    pass


def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, OpenAIEmbeddings]:
    """
    初始化LLM实例

    Args:
        llm_type (str): LLM类型，可选值为 'openai', 'oneapi', 'qwen', 'ollama', 'deepseek'

    Returns:
        tuple[ChatOpenAI, OpenAIEmbeddings]: 初始化后的LLM实例和Embedding实例

    Raises:
        LLMInitializationError: 当LLM初始化失败时抛出
    """
    try:
        # 检查llm_type是否有效
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的LLM类型: {llm_type}. 可用的类型: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]

        if llm_type == "ollama":
            os.environ["OPENAI_API_KEY"] = "NA"

        api_key = config["api_key"]
        if not api_key:
            raise ValueError(f"{llm_type} 的 API Key 未设置，请检查环境变量")
# 创建llm实例
        llm_chat = ChatOpenAI(
            base_url=config["base_url"],
            api_key=api_key,
            model=config["chat_model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=30,
            max_retries=2
        )

        embedding_base_url = config.get("embedding_base_url", config["base_url"])
        embedding_api_key = config.get("embedding_api_key", api_key)
        embedding_model = config["embedding_model"]
        
        logger.info(f"使用 Embedding: base_url={embedding_base_url}, model={embedding_model}")
        
        llm_embedding = OpenAIEmbeddings(
            base_url=embedding_base_url,
            api_key=embedding_api_key,
            model=embedding_model,
            deployment=embedding_model,
            check_embedding_ctx_length=False
        )

        logger.info(f"成功初始化 {llm_type} LLM")
        return llm_chat, llm_embedding

    except ValueError as ve:
        logger.error(f"LLM配置错误: {str(ve)}")
        raise LLMInitializationError(f"LLM配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化LLM失败: {str(e)}")
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")


def get_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, OpenAIEmbeddings]:
    """
    获取LLM实例的封装函数，提供默认值和错误处理

    Args:
        llm_type (str): LLM类型

    Returns:
        tuple[ChatOpenAI, OpenAIEmbeddings]: LLM实例和Embedding实例
    """
    try:
        return initialize_llm(llm_type)
    except LLMInitializationError as e:
        logger.warning(f"使用默认配置重试: {str(e)}")
        if llm_type != DEFAULT_LLM_TYPE:
            return initialize_llm(DEFAULT_LLM_TYPE)
        raise


if __name__ == "__main__":
    try:
        # 测试 Qwen 模型（需要 DASHSCOPE_API_KEY）
        llm_qwen, embedding_qwen = get_llm("qwen")
        logger.info("Qwen LLM 初始化成功")
        
        # 测试 DeepSeek 模型（需要 DEEPSEEK_API_KEY）
        llm_deepseek, embedding_deepseek = get_llm("deepseek")
        logger.info("DeepSeek LLM 初始化成功")
        
        logger.info("所有模型测试通过！")
    except LLMInitializationError as e:
        logger.error(f"程序终止: {str(e)}")