"""
LLM 客户端工厂

根据配置创建 LLM 客户端实例
"""

from automl_react.config import get_config_loader


class LLMClientError(Exception):
    """LLM 客户端错误"""
    pass


def create_llm_client(model: str = None):
    """
    创建 LLM 客户端

    根据配置创建真实的 LLM 客户端，如果失败则抛出明确的错误
    """
    errors = []

    try:
        config_loader = get_config_loader()
        llm_config = config_loader.get_llm_config(model)

        provider = llm_config.get("provider", "openai")
        model_name = llm_config.get("model_name", model or "gpt-4")

        if provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
                api_key = llm_config.get("api_key")
                if not api_key:
                    raise LLMClientError(
                        f"Anthropic API 密钥未配置。\n"
                        f"请设置环境变量 ANTHROPIC_API_KEY 或在配置文件中指定 api_key。\n"
                        f"当前模型: {model_name}"
                    )
                return ChatAnthropic(
                    model=model_name,
                    temperature=llm_config.get("temperature", 0.1),
                    max_tokens=llm_config.get("max_tokens", 4096),
                    api_key=api_key
                )
            except ImportError as e:
                errors.append(f"langchain_anthropic 未安装: {e}")

        elif provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                api_key = llm_config.get("api_key")
                if not api_key:
                    raise LLMClientError(
                        f"OpenAI API 密钥未配置。\n"
                        f"请设置环境变量 OPENAI_API_KEY 或在配置文件中指定 api_key。\n"
                        f"当前模型: {model_name}"
                    )
                return ChatOpenAI(
                    model=model_name,
                    temperature=llm_config.get("temperature", 0.1),
                    max_tokens=llm_config.get("max_tokens", 4096),
                    api_key=api_key,
                    base_url=llm_config.get("base_url")
                )
            except ImportError as e:
                errors.append(f"langchain_openai 未安装: {e}")

    except LLMClientError:
        raise
    except Exception as e:
        errors.append(f"从配置创建 LLM 客户端失败: {e}")

    error_msg = "无法创建 LLM 客户端。\n\n"
    error_msg += "可能的原因:\n"
    error_msg += "1. 缺少必要的 Python 包:\n"
    error_msg += "   - pip install langchain-openai  # 使用 OpenAI\n"
    error_msg += "   - pip install langchain-anthropic  # 使用 Claude\n"
    error_msg += "2. API 密钥未配置:\n"
    error_msg += "   - 设置环境变量 OPENAI_API_KEY 或 ANTHROPIC_API_KEY\n"
    error_msg += "   - 或在 llm_config.yaml 中配置 api_key\n"
    error_msg += "3. 配置文件错误:\n"
    error_msg += "   - 检查 automl_react/config/llm_config.yaml 配置\n\n"
    error_msg += "详细错误:\n"
    for err in errors:
        error_msg += f"  - {err}\n"

    raise LLMClientError(error_msg)
