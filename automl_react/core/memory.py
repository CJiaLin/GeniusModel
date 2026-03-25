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
    ) -> str:
        """获取短期记忆上下文"""
        if not self.short_term:
            return ""
        
        lines = ["## 对话历史"]
        for entry in self.short_term:
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
