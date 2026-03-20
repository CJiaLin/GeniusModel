"""
工具基类模块

定义所有工具的接口和基础实现
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class ToolStatus(Enum):
    """工具执行状态"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class ToolResult:
    """
    工具执行结果
    
    Attributes:
        status: 执行状态
        data: 返回数据
        error: 错误信息
        metadata: 元数据
    """
    status: ToolStatus
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def success(cls, data: Any = None, metadata: Dict[str, Any] = None) -> "ToolResult":
        """创建成功结果"""
        return cls(
            status=ToolStatus.SUCCESS,
            data=data,
            metadata=metadata or {}
        )
    
    @classmethod
    def error(cls, error_message: str, metadata: Dict[str, Any] = None) -> "ToolResult":
        """创建错误结果"""
        return cls(
            status=ToolStatus.ERROR,
            error=error_message,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata
        }
    
    def to_observation_text(self) -> str:
        """转换为观察文本"""
        if self.status == ToolStatus.SUCCESS:
            if isinstance(self.data, dict):
                lines = ["执行结果:"]
                for key, value in self.data.items():
                    lines.append(f"  {key}: {value}")
                return "\n".join(lines)
            else:
                return f"执行结果: {self.data}"
        else:
            return f"执行失败: {self.error}"


class BaseTool(ABC):
    """
    工具基类
    
    所有工具的抽象基类，定义了工具的接口
    
    Attributes:
        name: 工具名称
        description: 工具描述
        parameters: 参数定义
    """
    
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    
    def __init__(self):
        self._validate_tool_definition()
    
    def _validate_tool_definition(self):
        """验证工具定义"""
        if not self.name:
            raise ValueError(f"工具 {self.__class__.__name__} 必须定义 name")
        if not self.description:
            raise ValueError(f"工具 {self.__class__.__name__} 必须定义 description")
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        执行工具
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            ToolResult: 执行结果
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """获取工具 schema（用于 LLM 理解）"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def __call__(self, **kwargs) -> ToolResult:
        """使工具可调用"""
        return self.execute(**kwargs)
