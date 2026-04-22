"""
Token 监控中间件

统计每次 LLM 调用的 token 消耗并写入 ctx.metadata，
在即将超限时发出警告。
"""

from ..middleware import IterationContext, Middleware


class TokenMonitorMiddleware(Middleware):
    """跟踪 token 消耗，写入 metadata 供其他中间件和日志使用。"""

    def __init__(self, max_tokens: int = 128000, encoding_name: str = "cl100k_base"):
        self._max_tokens = max_tokens
        self._encoding_name = encoding_name
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _count_tokens(self, text: str) -> int:
        """使用 tiktoken 统计 token 数量。"""
        try:
            import tiktoken
            enc = tiktoken.get_encoding(self._encoding_name)
            return len(enc.encode(text))
        except Exception:
            # tiktoken 不可用时降级为字符估算 (1 token ~ 4 chars)
            return len(text) // 4

    def before_llm_call(self, ctx: IterationContext) -> IterationContext:
        # 统计 prompt token 数
        prompt_text = ""
        if isinstance(ctx.prompt, list):
            prompt_text = "\n".join(
                getattr(msg, "content", str(msg)) for msg in ctx.prompt
            )
        else:
            prompt_text = str(ctx.prompt)

        input_tokens = self._count_tokens(prompt_text)
        self._total_input_tokens += input_tokens

        ctx.metadata["input_tokens"] = input_tokens
        ctx.metadata["total_input_tokens"] = self._total_input_tokens

        # 检查是否即将超限 (>80%)
        if input_tokens > self._max_tokens * 0.8:
            ctx.metadata["token_warning"] = (
                f"输入 token ({input_tokens}) 已接近模型上限 ({self._max_tokens})"
            )
        return ctx

    def after_llm_call(self, ctx: IterationContext) -> IterationContext:
        if ctx.llm_response:
            output_tokens = self._count_tokens(ctx.llm_response)
            self._total_output_tokens += output_tokens
            ctx.metadata["output_tokens"] = output_tokens
            ctx.metadata["total_output_tokens"] = self._total_output_tokens
            ctx.metadata["total_tokens"] = self._total_input_tokens + self._total_output_tokens
        return ctx
