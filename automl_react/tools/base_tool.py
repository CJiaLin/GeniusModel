"""
工具基类模块

定义所有工具的接口和基础实现
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional, Type
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, ValidationError


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

    所有工具的抽象基类，定义了工具的接口。

    子类可设置 ``input_model`` 为一个 Pydantic BaseModel 子类，
    框架会自动从中生成 JSON Schema 并在执行前做参数校验。
    未设置 ``input_model`` 的工具保持原有行为（plain dict parameters）。

    Attributes:
        name: 工具名称
        description: 工具描述
        parameters: 参数定义（当 input_model 为 None 时使用）
        input_model: 可选的 Pydantic 输入模型类
    """

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    input_model: ClassVar[Optional[Type[BaseModel]]] = None

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

    def execute_validated(self, **kwargs) -> ToolResult:
        """验证参数后执行工具。若设置了 input_model 则先做 Pydantic 校验。"""
        if self.input_model is not None:
            try:
                validated = self.input_model(**kwargs)
                return self.execute(**validated.model_dump())
            except ValidationError as e:
                return ToolResult.error(f"参数验证失败: {e}")
        return self.execute(**kwargs)

    def get_schema(self) -> Dict[str, Any]:
        """获取工具 schema（用于 LLM 理解）"""
        if self.input_model is not None:
            json_schema = self.input_model.model_json_schema()
            params = json_schema.get("properties", {})
            return {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            }
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

    def __call__(self, **kwargs) -> ToolResult:
        """使工具可调用"""
        return self.execute_validated(**kwargs)
