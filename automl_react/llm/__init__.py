"""
LLM 模块

提供多 LLM 提供商抽象层，通过注册表模式支持按需扩展。
"""

from .provider_factory import LLMProviderFactory, BaseProvider

# 导入 providers 包以触发自动注册
from . import providers  # noqa: F401

__all__ = ["LLMProviderFactory", "BaseProvider"]
