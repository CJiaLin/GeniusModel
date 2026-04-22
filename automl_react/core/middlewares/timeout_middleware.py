"""
超时中间件

强制执行每次迭代和总流程的超时限制。
"""

import time

from ..middleware import IterationContext, Middleware


class TimeoutMiddleware(Middleware):
    """
    检测单次迭代和总流程超时。

    在 before_llm_call 中检测是否已超出 total_timeout，
    若超时则设置 ctx.should_stop = True。
    """

    def __init__(
        self,
        total_timeout: int = 600,
        per_iteration_timeout: int = 120,
    ):
        self._total_timeout = total_timeout
        self._per_iteration_timeout = per_iteration_timeout
        self._start_time: float = 0
        self._iteration_start: float = 0

    def before_llm_call(self, ctx: IterationContext) -> IterationContext:
        now = time.monotonic()

        # 首次调用时初始化起始时间
        if self._start_time == 0:
            self._start_time = now
        self._iteration_start = now

        # 检查总超时
        elapsed = now - self._start_time
        if elapsed > self._total_timeout:
            ctx.should_stop = True
            ctx.error = f"总流程超时: 已用 {elapsed:.0f}s，上限 {self._total_timeout}s"
            ctx.metadata["timeout_type"] = "total"
        return ctx

    def after_llm_call(self, ctx: IterationContext) -> IterationContext:
        now = time.monotonic()
        iteration_elapsed = now - self._iteration_start
        if iteration_elapsed > self._per_iteration_timeout:
            ctx.metadata["iteration_timeout_warning"] = (
                f"迭代 {ctx.iteration} 用时 {iteration_elapsed:.0f}s，"
                f"超出单次上限 {self._per_iteration_timeout}s"
            )
        ctx.metadata["iteration_elapsed_s"] = round(iteration_elapsed, 2)
        ctx.metadata["total_elapsed_s"] = round(now - self._start_time, 2)
        return ctx
