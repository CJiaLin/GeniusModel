"""
错误处理中间件

统一处理 LLM 调用和工具调用中的异常，提供可配置的重试逻辑。
"""

import time
import traceback
from typing import Any

from ..middleware import IterationContext, Middleware


class ErrorHandlingMiddleware(Middleware):
    """捕获异常并写入 ctx.error，防止单次失败终止整个循环。"""

    def __init__(self, max_retries: int = 0, retry_delay: float = 1.0, verbose: bool = False):
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._verbose = verbose

    def on_error(self, ctx: IterationContext, error: Exception) -> IterationContext:
        error_detail = f"{type(error).__name__}: {error}"
        if self._verbose:
            error_detail += f"\n{traceback.format_exc()}"

        ctx.error = error_detail
        ctx.metadata.setdefault("errors", []).append({
            "iteration": ctx.iteration,
            "error": str(error),
            "type": type(error).__name__,
        })
        return ctx
