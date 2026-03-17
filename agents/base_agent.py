"""
基础 Agent 类 - 所有 Agent 的基类
"""
from typing import Optional, Dict, Any, List, Callable
from abc import ABC, abstractmethod
from langchain_core.language_models import BaseLanguageModel
import logging


class BaseAgent(ABC):
    """Agent 基类"""
    
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        name: Optional[str] = None,
        verbose: bool = False
    ):
        """
        初始化基础 Agent
        
        Args:
            llm: 语言模型实例
            name: Agent 名称
            verbose: 是否输出详细日志
        """
        self.llm = llm
        self.name = name or self.__class__.__name__
        self.verbose = verbose
        self.tools: Dict[str, Callable] = {}
        self.state: Optional[Any] = None
        self.logger = logging.getLogger(self.name)
        
        if verbose:
            logging.basicConfig(level=logging.INFO)
            
    def set_llm(self, llm: BaseLanguageModel):
        """设置语言模型"""
        self.llm = llm
        self.log(f"已设置语言模型：{type(llm).__name__}")
        
    def set_state(self, state: Any):
        """设置状态"""
        self.state = state
        self.log("已设置状态")
        
    def register_tool(self, name: str, func: Callable):
        """
        注册工具函数
        
        Args:
            name: 工具名称
            func: 工具函数
        """
        self.tools[name] = func
        self.log(f"已注册工具：{name}")
        
    def get_tool(self, name: str) -> Callable:
        """获取工具函数"""
        if name not in self.tools:
            raise ValueError(f"工具不存在：{name}")
        return self.tools[name]
        
    def list_tools(self) -> List[str]:
        """列出所有已注册的工具"""
        return list(self.tools.keys())
        
    def has_tool(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self.tools
        
    def invoke_llm(self, prompt: str, **kwargs) -> str:
        """
        调用 LLM
        
        Args:
            prompt: 提示词
            **kwargs: 其他参数
            
        Returns:
            LLM 响应
        """
        if self.llm is None:
            raise ValueError("未设置语言模型")
            
        self.log(f"调用 LLM: {prompt[:50]}...")
        response = self.llm.invoke(prompt, **kwargs)
        
        if hasattr(response, 'content'):
            result = response.content
        else:
            result = str(response)
            
        self.log(f"LLM 响应长度：{len(result)}")
        return result
        
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        log_msg = f"[{self.name}] {message}"
        if level == "ERROR":
            self.logger.error(log_msg)
        elif level == "WARNING":
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)
            
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        执行 Agent 的主要逻辑（必须由子类实现）
        """
        pass
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "tools": self.list_tools(),
            "has_llm": self.llm is not None,
            "verbose": self.verbose
        }