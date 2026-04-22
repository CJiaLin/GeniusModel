"""
Anthropic LLM Provider

支持 Anthropic Claude 系列模型。
"""

from typing import Any, Dict, List

from ..provider_factory import BaseProvider, LLMProviderFactory


class AnthropicProvider(BaseProvider):
    """Anthropic Claude 提供商"""

    @classmethod
    def create_client(cls, config: Dict[str, Any]) -> Any:
        from langchain_anthropic import ChatAnthropic

        api_key = config.get("api_key")
        if not api_key:
            from ...api.main import LLMClientError
            raise LLMClientError(
                f"Anthropic API 密钥未配置。\n"
                f"请设置环境变量 ANTHROPIC_API_KEY 或在 llm_config.yaml 中指定 api_key。\n"
                f"当前模型: {config.get('model_name', 'unknown')}"
            )

        return ChatAnthropic(
            model=config.get("model_name", "claude-sonnet-4-20250514"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 4096),
            api_key=api_key,
        )

    @classmethod
    def required_env_vars(cls) -> List[str]:
        return ["ANTHROPIC_API_KEY"]


LLMProviderFactory.register("anthropic", AnthropicProvider)
