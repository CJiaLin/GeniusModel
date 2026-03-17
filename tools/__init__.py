"""
MCP 工具定义 - Model Context Protocol 标准化工具接口
"""
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class MCPTool:
    """
    MCP 工具定义
    
    Attributes:
        name: 工具名称
        description: 工具描述
        func: 工具函数
        parameters: 参数定义
        returns: 返回值描述
        category: 工具类别
    """
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    returns: str = ""
    category: str = "general"
    
    def __call__(self, *args, **kwargs) -> Any:
        """调用工具"""
        return self.func(*args, **kwargs)
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "category": self.category
        }


class MCPToolRegistry:
    """MCP 工具注册表"""
    
    def __init__(self):
        """初始化工具注册表"""
        self.tools: Dict[str, MCPTool] = {}
        
    def register(self, tool: MCPTool):
        """
        注册工具
        
        Args:
            tool: MCPTool 实例
        """
        self.tools[tool.name] = tool
        
    def get(self, name: str) -> MCPTool:
        """获取工具"""
        if name not in self.tools:
            raise ValueError(f"工具不存在：{name}")
        return self.tools[name]
        
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """列出所有工具"""
        if category:
            return [name for name, tool in self.tools.items() 
                    if tool.category == category]
        return list(self.tools.keys())
        
    def has_tool(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self.tools
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {name: tool.to_dict() for name, tool in self.tools.items()}


# 全局工具注册表
_global_registry = MCPToolRegistry()


def get_registry() -> MCPToolRegistry:
    """获取全局工具注册表"""
    return _global_registry


def register_tool(name: str, description: str, 
                  category: str = "general",
                  parameters: Optional[Dict[str, Any]] = None,
                  returns: str = ""):
    """
    工具注册装饰器
    
    Usage:
        @register_tool("load_data", "加载数据文件", category="data")
        def load_data(filepath: str) -> pd.DataFrame:
            ...
    """
    def decorator(func: Callable) -> Callable:
        tool = MCPTool(
            name=name,
            description=description,
            func=func,
            parameters=parameters or {},
            returns=returns,
            category=category
        )
        _global_registry.register(tool)
        return func
    return decorator
