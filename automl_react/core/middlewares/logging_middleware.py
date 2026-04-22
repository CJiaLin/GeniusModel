"""
日志中间件

统一记录 LLM 调用和工具调用的日志，从 react_agent._call_llm() 中提取而来。
"""

from datetime import datetime
from typing import Any

from ..middleware import IterationContext, Middleware


class LoggingMiddleware(Middleware):
    """在 LLM 调用和工具调用前后写入 LLMLogger 日志。"""

    def __init__(self, llm_logger: Any, config_loader: Any):
        self._logger = llm_logger
        self._config_loader = config_loader
        self._call_start: float = 0

    def before_llm_call(self, ctx: IterationContext) -> IterationContext:
        self._call_start = datetime.now().timestamp()
        return ctx

    def after_llm_call(self, ctx: IterationContext) -> IterationContext:
        elapsed_ms = int((datetime.now().timestamp() - self._call_start) * 1000)

        llm_config = self._config_loader.get_llm_config()
        model_name = llm_config.get("model_name", "unknown")
        provider = llm_config.get("provider", "unknown")

        input_text = ""
        if isinstance(ctx.prompt, list):
            parts = []
            for msg in ctx.prompt:
                role = getattr(msg, "type", "unknown")
                content = getattr(msg, "content", str(msg))
                parts.append(f"[{role}] {content}")
            input_text = "\n".join(parts)
        else:
            input_text = str(ctx.prompt)

        self._logger.log_call(
            model_name=model_name,
            provider=provider,
            input_content=input_text,
            output_content=ctx.llm_response or "",
            stage=ctx.stage,
            metadata={
                "iteration": ctx.iteration,
                "latency_ms": elapsed_ms,
                "prompt_scope": "final_actual_llm_input",
                "prompt_format": "chat_messages_system_user",
                **(ctx.metadata.get("llm_extra", {})),
            },
        )
        return ctx

    def before_tool_call(self, ctx: IterationContext) -> IterationContext:
        return ctx

    def after_tool_call(self, ctx: IterationContext) -> IterationContext:
        return ctx
