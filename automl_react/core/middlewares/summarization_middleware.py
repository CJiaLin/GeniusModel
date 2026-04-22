"""
上下文摘要中间件

在 LLM 调用前检查上下文 token 数，超过阈值时自动摘要压缩旧条目。
借鉴 DeerFlow 的 SummarizationMiddleware 设计。
"""

from typing import Any, Callable, Optional

from ..middleware import IterationContext, Middleware
from ..memory import Memory


class SummarizationMiddleware(Middleware):
    """
    在 before_llm_call 中检查 memory token 数，
    超过阈值时用 LLM 生成历史摘要以压缩上下文。
    """

    def __init__(
        self,
        memory: Memory,
        llm: Any = None,
        max_context_tokens: int = 8000,
        summarization_threshold: float = 0.75,
        keep_recent: int = 5,
    ):
        self._memory = memory
        self._llm = llm
        self._max_tokens = max_context_tokens
        self._threshold = summarization_threshold
        self._keep_recent = keep_recent

    def _summarize(self, text: str) -> str:
        """使用 LLM 生成简短摘要。"""
        if not self._llm:
            # 无 LLM 时做简单截断
            return text[:500] + "..." if len(text) > 500 else text

        prompt = (
            "请将以下 AI Agent 的工作历史摘要为简洁的要点（不超过 300 字），"
            "保留关键发现、已执行的操作和重要结论：\n\n"
            f"{text}"
        )
        try:
            response = self._llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception:
            return text[:500] + "..." if len(text) > 500 else text

    def before_llm_call(self, ctx: IterationContext) -> IterationContext:
        tokens = self._memory.estimate_tokens()
        threshold_tokens = int(self._max_tokens * self._threshold)

        if tokens > threshold_tokens:
            self._memory.summarize(
                summarizer_fn=self._summarize,
                keep_recent=self._keep_recent,
            )
            new_tokens = self._memory.estimate_tokens()
            ctx.metadata["summarization"] = {
                "before_tokens": tokens,
                "after_tokens": new_tokens,
                "saved_tokens": tokens - new_tokens,
            }
        return ctx
