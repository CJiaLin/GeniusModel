"""Providers 包 — 导入即自动注册到 LLMProviderFactory。"""

from . import openai_provider       # noqa: F401
from . import anthropic_provider    # noqa: F401
from . import openai_compatible_provider  # noqa: F401
