"""
Stream Callback 机制

用于将 Agent 内部的 LLM 流式输出传递到外部（如 SSE 端点）。
Agent 层调用 callback 函数，不依赖 asyncio。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional


class StreamEventType(Enum):
    CONTENT = "content"
    PROGRESS = "progress"
    CODE_GENERATED = "code_generated"
    DONE = "done"
    ERROR = "error"


@dataclass
class StreamEvent:
    type: StreamEventType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


StreamCallbackFn = Optional[Callable[[StreamEvent], None]]
