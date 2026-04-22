"""
OpenAI LLM Provider

支持 OpenAI 官方 API 及所有 OpenAI 兼容 API（通过 base_url 指定）。
"""

from typing import Any, Dict, List

from ..provider_factory import BaseProvider, LLMProviderFactory


class OpenAIProvider(BaseProvider):
    """OpenAI / OpenAI-Compatible 提供商"""

    @classmethod
    def create_client(cls, config: Dict[str, Any]) -> Any:
        from langchain_openai import ChatOpenAI

        api_key = config.get("api_key")
        if not api_key:
            from ...api.main import LLMClientError
            raise LLMClientError(
                f"OpenAI API 密钥未配置。\n"
                f"请设置环境变量 OPENAI_API_KEY 或在 llm_config.yaml 中指定 api_key。\n"
                f"当前模型: {config.get('model_name', 'unknown')}"
            )

        return ChatOpenAI(
            model=config.get("model_name", "gpt-4"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 4096),
            api_key=api_key,
            base_url=config.get("base_url"),
        )

    @classmethod
    def required_env_vars(cls) -> List[str]:
        return ["OPENAI_API_KEY"]


LLMProviderFactory.register("openai", OpenAIProvider)
