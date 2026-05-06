"""
OpenAI Compatible LLM Provider

适配使用 OpenAI 兼容 API 的国产模型提供商：
- 阿里通义千问 (DashScope)
- Moonshot/Kimi
- 智谱 GLM (ZhipuAI)
- MiniMax
- DeepSeek
- 零一万物 (Yi)

这些提供商都支持 OpenAI 格式的 API，只需指定不同的 base_url 和 api_key。
"""

from typing import Any, Dict, List

from ..provider_factory import BaseProvider, LLMProviderFactory


# 已知提供商的默认 base_url
_KNOWN_PROVIDERS: Dict[str, str] = {
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4/",
    "deepseek": "https://api.deepseek.com/v1",
    "minimax": "https://api.minimax.chat/v1",
    "yi": "https://api.lingyiwanwu.com/v1",
}


class OpenAICompatibleProvider(BaseProvider):
    """OpenAI 兼容 API 提供商（主要面向国产大模型）"""

    @classmethod
    def create_client(cls, config: Dict[str, Any]) -> Any:
        from langchain_openai import ChatOpenAI

        api_key = config.get("api_key")
        if not api_key:
            from ...api.main import LLMClientError
            raise LLMClientError(
                f"API 密钥未配置。\n"
                f"请在 llm_config.yaml 中指定 api_key。\n"
                f"当前模型: {config.get('model_name', 'unknown')}"
            )

        # 如果指定了 sub_provider，使用预定义的 base_url
        base_url = config.get("base_url")
        sub_provider = config.get("sub_provider")
        if not base_url and sub_provider and sub_provider in _KNOWN_PROVIDERS:
            base_url = _KNOWN_PROVIDERS[sub_provider]

        if not base_url:
            from ...api.main import LLMClientError
            raise LLMClientError(
                f"OpenAI 兼容提供商需要指定 base_url 或 sub_provider。\n"
                f"已知的 sub_provider: {list(_KNOWN_PROVIDERS.keys())}"
            )

        # timeout: 支持秒(int)或毫秒(>1000的大数自动转换为秒)
        raw_timeout = config.get("timeout")
        if raw_timeout is not None:
            timeout_sec = raw_timeout / 1000 if raw_timeout > 1000 else raw_timeout
        else:
            timeout_sec = 120  # 默认 120 秒

        kwargs = {
            "model": config.get("model_name", "gpt-4"),
            "temperature": config.get("temperature", 0.1),
            "max_tokens": config.get("max_tokens", 4096),
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout_sec,
        }

        # 某些提供商需要额外的请求头
        extra_headers = config.get("extra_headers")
        if extra_headers:
            kwargs["default_headers"] = extra_headers

        return ChatOpenAI(**kwargs)

    @classmethod
    def required_env_vars(cls) -> List[str]:
        return []


LLMProviderFactory.register("openai_compatible", OpenAICompatibleProvider)
