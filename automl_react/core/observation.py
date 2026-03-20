"""
Observation 模块

封装 Agent 对环境的观察结果
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ObservationType(Enum):
    """观察结果类型"""
    DATA = "data"              # 数据观察
    TOOL_RESULT = "tool_result"  # 工具执行结果
    CODE_OUTPUT = "code_output"  # 代码执行输出
    ERROR = "error"            # 错误信息
    USER_INPUT = "user_input"  # 用户输入
    SYSTEM = "system"          # 系统消息


@dataclass
class Observation:
    """
    观察结果类
    
    封装 Agent 从环境中观察到的信息
    
    Attributes:
        type: 观察结果类型
        content: 观察内容
        metadata: 元数据
        timestamp: 观察时间
    """
    type: ObservationType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_prompt_text(self) -> str:
        """转换为提示词文本格式"""
        lines = [f"【{self.type.value}】"]
        
        if isinstance(self.content, str):
            lines.append(self.content)
        elif isinstance(self.content, dict):
            for key, value in self.content.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append(str(self.content))
        
        return "\n".join(lines)
    
    @classmethod
    def from_tool_result(cls, tool_name: str, result: Any, success: bool = True) -> "Observation":
        """从工具执行结果创建观察"""
        return cls(
            type=ObservationType.TOOL_RESULT,
            content=result,
            metadata={
                "tool_name": tool_name,
                "success": success
            }
        )
    
    @classmethod
    def from_data(cls, data_info: Dict[str, Any]) -> "Observation":
        """从数据信息创建观察"""
        return cls(
            type=ObservationType.DATA,
            content=data_info
        )
    
    @classmethod
    def from_error(cls, error_message: str, error_type: str = "general") -> "Observation":
        """从错误信息创建观察"""
        return cls(
            type=ObservationType.ERROR,
            content=error_message,
            metadata={"error_type": error_type}
        )
    
    @classmethod
    def from_user_input(cls, user_message: str) -> "Observation":
        """从用户输入创建观察"""
        return cls(
            type=ObservationType.USER_INPUT,
            content=user_message
        )
