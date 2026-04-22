"""
Memory 模块

管理 Agent 的记忆（对话历史、执行上下文）
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MemoryType(Enum):
    """记忆类型"""
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_MESSAGE = "system_message"
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    CODE = "code"


@dataclass
class MemoryEntry:
    """记忆条目"""
    type: MemoryType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class Memory:
    """
    记忆管理类
    
    管理 Agent 的对话历史和执行上下文
    
    Attributes:
        max_entries: 最大记忆条目数
        entries: 记忆条目列表
    """
    
    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []
        self.short_term: List[MemoryEntry] = []  # 短期记忆（当前会话）
        self.long_term: Dict[str, Any] = {}      # 长期记忆（关键信息）
    
    def add(self, entry_type: MemoryType, content: str, metadata: Dict[str, Any] = None):
        """添加记忆条目"""
        entry = MemoryEntry(
            type=entry_type,
            content=content,
            metadata=metadata or {}
        )
        self.entries.append(entry)
        self.short_term.append(entry)
        
        # 限制记忆大小
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
        if len(self.short_term) > 20:  # 短期记忆保留最近20条
            self.short_term.pop(0)
    
    def add_user_message(self, message: str):
        """添加用户消息"""
        self.add(MemoryType.USER_MESSAGE, message)
    
    def add_assistant_message(self, message: str):
        """添加助手消息"""
        self.add(MemoryType.ASSISTANT_MESSAGE, message)
    
    def add_thought(self, thought: str, step: int = None):
        """添加思考过程"""
        self.add(MemoryType.THOUGHT, thought, {"step": step})
    
    def add_action(self, action: str, action_input: Dict[str, Any] = None):
        """添加执行的动作"""
        self.add(MemoryType.ACTION, action, {"input": action_input})
    
    def add_observation(self, observation: str):
        """添加观察结果"""
        self.add(MemoryType.OBSERVATION, observation)
    
    def add_code(self, code: str, execution_result: Any = None):
        """添加执行的代码"""
        self.add(MemoryType.CODE, code, {"result": execution_result})
    
    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """获取最近的 n 条记忆"""
        return self.entries[-n:] if len(self.entries) > n else self.entries
    
    def get_short_term_context(
        self,
        include_user_messages: bool = True,
        include_assistant_messages: bool = True,
        skip_last_observation: bool = False,
    ) -> str:
        """获取短期记忆上下文

        Args:
            include_user_messages: 是否包含用户消息
            include_assistant_messages: 是否包含助手消息
            skip_last_observation: 是否跳过最后一条观察记录（避免与显式传入的 observation 重复）
        """
        if not self.short_term:
            return ""

        # 确定需要跳过的观察条目索引
        skip_idx = None
        if skip_last_observation:
            for i in range(len(self.short_term) - 1, -1, -1):
                if self.short_term[i].type == MemoryType.OBSERVATION:
                    skip_idx = i
                    break

        lines = ["## 对话历史"]
        for i, entry in enumerate(self.short_term):
            if i == skip_idx:
                continue
            if entry.type == MemoryType.USER_MESSAGE:
                if include_user_messages:
                    lines.append(f"用户: {entry.content}")
            elif entry.type == MemoryType.ASSISTANT_MESSAGE:
                if include_assistant_messages:
                    lines.append(f"助手: {entry.content}")
            elif entry.type == MemoryType.THOUGHT:
                lines.append(f"思考: {entry.content}")
            elif entry.type == MemoryType.ACTION:
                lines.append(f"动作: {entry.content}")
            elif entry.type == MemoryType.OBSERVATION:
                lines.append(f"观察: {entry.content}")

        if len(lines) == 1:
            return ""

        return "\n".join(lines)
    
    def get_full_context(self) -> str:
        """获取完整记忆上下文"""
        if not self.entries:
            return ""
        
        lines = ["## 完整执行历史"]
        for entry in self.entries:
            lines.append(f"[{entry.type.value}] {entry.content}")
        
        return "\n".join(lines)
    
    def to_messages(self) -> List[Dict[str, str]]:
        """转换为 LLM 消息格式"""
        messages = []
        
        for entry in self.entries:
            if entry.type == MemoryType.USER_MESSAGE:
                messages.append({"role": "user", "content": entry.content})
            elif entry.type == MemoryType.ASSISTANT_MESSAGE:
                messages.append({"role": "assistant", "content": entry.content})
            elif entry.type == MemoryType.SYSTEM_MESSAGE:
                messages.append({"role": "system", "content": entry.content})
            elif entry.type == MemoryType.THOUGHT:
                messages.append({"role": "assistant", "content": f"思考: {entry.content}"})
        
        return messages
    
    def clear(self):
        """清空记忆"""
        self.entries.clear()
        self.short_term.clear()
        self.long_term.clear()
    
    def set_long_term(self, key: str, value: Any):
        """设置长期记忆"""
        self.long_term[key] = value

    def get_long_term(self, key: str) -> Optional[Any]:
        """获取长期记忆"""
        return self.long_term.get(key)

    def estimate_tokens(self, encoding_name: str = "cl100k_base") -> int:
        """
        估算短期记忆的 token 数量。

        使用 tiktoken（若可用），否则降级为字符数 / 4 的粗略估算。
        """
        text = self.get_short_term_context()
        if not text:
            return 0
        try:
            import tiktoken
            enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4

    def summarize(
        self,
        summarizer_fn,
        keep_recent: int = 5,
    ):
        """
        压缩短期记忆：将较旧的条目摘要为单条 SYSTEM_MESSAGE。

        Args:
            summarizer_fn: 接受 str 返回 str 的摘要函数（通常由 LLM 完成）
            keep_recent: 保留最近 N 条条目不做摘要
        """
        if len(self.short_term) <= keep_recent:
            return  # 条目不够多，无需摘要

        old_entries = self.short_term[:-keep_recent]
        recent_entries = self.short_term[-keep_recent:]

        # 将旧条目格式化为文本
        lines = []
        for entry in old_entries:
            lines.append(f"[{entry.type.value}] {entry.content}")
        old_text = "\n".join(lines)

        # 调用摘要函数
        try:
            summary = summarizer_fn(old_text)
        except Exception:
            return  # 摘要失败时保持原样

        # 替换短期记忆：一条摘要 + 最近 N 条
        summary_entry = MemoryEntry(
            type=MemoryType.SYSTEM_MESSAGE,
            content=f"[历史摘要]\n{summary}",
        )
        self.short_term = [summary_entry] + recent_entries
