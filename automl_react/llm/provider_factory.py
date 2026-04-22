"""
LLM Provider 工厂模块

基于注册表模式的 LLM 提供商抽象，支持按需注册和创建不同提供商的 LLM 客户端。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type


class BaseProvider(ABC):
    """LLM 提供商基类"""

    @classmethod
    @abstractmethod
    def create_client(cls, config: Dict[str, Any]) -> Any:
        """根据配置创建 LangChain Chat Model 实例。"""
        ...

    @classmethod
    def required_env_vars(cls) -> List[str]:
        """该提供商需要的环境变量列表（仅作文档用途）。"""
        return []


class LLMProviderFactory:
    """
    LLM 客户端工厂

    通过 ``register()`` 注册提供商，通过 ``create()`` 按 config 中的
    ``provider`` 字段查找并创建对应 LLM 客户端。
    """

    _registry: Dict[str, Type[BaseProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: Type[BaseProvider]):
        """注册一个 LLM 提供商。"""
        cls._registry[name] = provider_cls

    @classmethod
    def create(cls, config: Dict[str, Any]) -> Any:
        """
        根据配置创建 LLM 客户端。

        Args:
            config: 模型配置字典，必须包含 ``provider`` 字段。

        Returns:
            LangChain BaseChatModel 实例

        Raises:
            LLMClientError: 提供商未注册或创建失败
        """
        from ..api.main import LLMClientError

        provider_name = config.get("provider", "openai")
        if provider_name not in cls._registry:
            available = list(cls._registry.keys())
            raise LLMClientError(
                f"未知 LLM provider: '{provider_name}'。\n"
                f"已注册的 provider: {available}\n"
                f"请检查 llm_config.yaml 中的 provider 配置。"
            )
        try:
            return cls._registry[provider_name].create_client(config)
        except ImportError as e:
            raise LLMClientError(
                f"Provider '{provider_name}' 依赖的包未安装: {e}\n"
                f"请运行: pip install langchain-{provider_name}"
            )

    @classmethod
    def available_providers(cls) -> List[str]:
        """返回已注册的提供商名称列表。"""
        return list(cls._registry.keys())
