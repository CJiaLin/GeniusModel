"""
中间件框架模块

定义 ReAct Agent 执行管道的中间件协议和链式执行器。
借鉴 DeerFlow 的 18 层中间件链设计，为 Agent 的 LLM 调用和工具调用
提供可组合的横切关注点处理（日志、错误处理、超时、Token 监控等）。
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IterationContext:
    """
    中间件链中流转的迭代上下文。

    每次 ReAct 循环迭代创建一个新的 IterationContext，在中间件链中传递和修改。
    """
    iteration: int
    max_iterations: int
    stage: str
    user_input: str
    observation: str = ""

    # LLM 调用相关
    prompt: Any = None
    llm_response: Optional[str] = None

    # 工具调用相关
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None

    # 元数据（供中间件写入额外信息）
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 控制标志
    should_stop: bool = False
    error: Optional[str] = None


class Middleware(ABC):
    """
    中间件基类。

    子类可选择性覆盖以下钩子方法：
    - before_llm_call: 在 LLM 调用之前
    - after_llm_call: 在 LLM 调用之后
    - before_tool_call: 在工具调用之前
    - after_tool_call: 在工具调用之后
    - on_error: 在发生异常时
    """

    def before_llm_call(self, ctx: IterationContext) -> IterationContext:
        return ctx

    def after_llm_call(self, ctx: IterationContext) -> IterationContext:
        return ctx

    def before_tool_call(self, ctx: IterationContext) -> IterationContext:
        return ctx

    def after_tool_call(self, ctx: IterationContext) -> IterationContext:
        return ctx

    def on_error(self, ctx: IterationContext, error: Exception) -> IterationContext:
        return ctx


class MiddlewareChain:
    """
    中间件链——按注册顺序依次执行所有中间件的对应钩子。
    """

    def __init__(self, middlewares: Optional[List[Middleware]] = None):
        self._middlewares: List[Middleware] = list(middlewares or [])

    def add(self, middleware: Middleware):
        """追加一个中间件。"""
        self._middlewares.append(middleware)

    def run_before_llm(self, ctx: IterationContext) -> IterationContext:
        for mw in self._middlewares:
            ctx = mw.before_llm_call(ctx)
            if ctx.should_stop:
                break
        return ctx

    def run_after_llm(self, ctx: IterationContext) -> IterationContext:
        for mw in self._middlewares:
            ctx = mw.after_llm_call(ctx)
            if ctx.should_stop:
                break
        return ctx

    def run_before_tool(self, ctx: IterationContext) -> IterationContext:
        for mw in self._middlewares:
            ctx = mw.before_tool_call(ctx)
            if ctx.should_stop:
                break
        return ctx

    def run_after_tool(self, ctx: IterationContext) -> IterationContext:
        for mw in self._middlewares:
            ctx = mw.after_tool_call(ctx)
            if ctx.should_stop:
                break
        return ctx

    def run_on_error(self, ctx: IterationContext, error: Exception) -> IterationContext:
        for mw in self._middlewares:
            ctx = mw.on_error(ctx, error)
        return ctx
