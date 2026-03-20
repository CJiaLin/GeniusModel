"""
AutoML ReAct Agent 框架

基于 ReAct (Reasoning + Acting) 架构的 AutoML 系统

核心概念:
- Observation: 观察环境状态
- Thought: LLM 推理下一步行动
- Action: 执行具体动作
- Memory: 保存对话历史和执行上下文

使用示例:
    from automl_react import ReActAgent, Tool
    
    agent = ReActAgent()
    agent.register_tool("load_data", DataLoaderTool())
    result = agent.run("分析数据并训练模型")
"""

from .core.react_agent import ReActAgent
from .core.memory import Memory
from .core.observation import Observation
from .tools.base_tool import BaseTool, ToolResult

__version__ = "1.0.0"
__all__ = [
    "ReActAgent",
    "Memory",
    "Observation",
    "BaseTool",
    "ToolResult",
]
