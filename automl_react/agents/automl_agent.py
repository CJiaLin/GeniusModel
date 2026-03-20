"""
AutoML Agent 模块

实现基于 ReAct 架构的 AutoML 业务 Agent
"""

from typing import Any, Dict

from ..core.react_agent import ReActAgent
from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
from ..tools.feature_tools import FeatureGeneratorTool
from ..tools.model_tools import ModelTrainerTool


class AutoMLAgent(ReActAgent):
    """
    AutoML Agent
    
    基于 ReAct 架构的自动化机器学习 Agent
    
    工作流程：
    1. 加载数据
    2. 分析数据
    3. 生成特征
    4. 训练模型
    
    Attributes:
        llm: 语言模型实例
        session_id: 会话ID
        data_path: 当前数据路径
        target_column: 目标列名
        task_type: 任务类型
    """
    
    def __init__(self, llm: Any = None, session_id: str = None, verbose: bool = False):
        super().__init__(llm=llm, max_iterations=15, verbose=verbose)
        self.session_id = session_id
        self.data_path: str = None
        self.target_column: str = None
        self.task_type: str = None
        
        # 注册工具
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认工具"""
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())
        self.register_tool("generate_features", FeatureGeneratorTool())
        self.register_tool("train_model", ModelTrainerTool())
    
    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一位专业的 AutoML 专家，专门帮助用户完成机器学习建模任务。

你的职责：
1. 理解用户的建模需求
2. 加载和分析数据
3. 生成有用的特征
4. 训练并评估模型

工作原则：
- 仔细分析数据质量和分布
- 根据数据特点选择合适的特征工程方法
- 明确区分分类和回归任务
- 提供清晰的执行步骤和结果解释

你可以使用以下工具来完成任务：
- load_data: 加载数据文件
- analyze_data: 分析数据
- generate_features: 生成特征
- train_model: 训练模型

请按照 ReAct 格式进行思考和行动。"""
    
    def set_data_context(self, data_path: str, target_column: str, task_type: str = None):
        """
        设置数据上下文
        
        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型（可选，自动检测）
        """
        self.data_path = data_path
        self.target_column = target_column
        self.task_type = task_type
        
        # 添加到长期记忆
        self.memory.set_long_term("data_path", data_path)
        self.memory.set_long_term("target_column", target_column)
        self.memory.set_long_term("task_type", task_type)
    
    def analyze(self, data_path: str = None) -> Dict[str, Any]:
        """
        分析数据
        
        Args:
            data_path: 数据文件路径（如果不指定则使用已设置的）
            
        Returns:
            分析结果
        """
        path = data_path or self.data_path
        if not path:
            raise ValueError("请提供数据文件路径")
        
        user_input = f"请分析数据文件: {path}"
        
        return self.run(user_input)
    
    def generate_features(self, data_path: str = None, target_column: str = None, task_type: str = "classification") -> Dict[str, Any]:
        """
        生成特征
        
        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型
            
        Returns:
            特征生成结果
        """
        path = data_path or self.data_path
        target = target_column or self.target_column
        
        if not path or not target:
            raise ValueError("请提供数据文件路径和目标列名")
        
        user_input = f"请为数据 {path} 生成特征，目标列是 {target}，任务类型是 {task_type}"
        
        return self.run(user_input)
    
    def train(self, data_path: str = None, target_column: str = None, task_type: str = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型
            
        Returns:
            训练结果
        """
        path = data_path or self.data_path
        target = target_column or self.target_column
        task = task_type or self.task_type or "classification"
        
        if not path or not target:
            raise ValueError("请提供数据文件路径和目标列名")
        
        user_input = f"请训练一个{task}模型，数据文件是 {path}，目标列是 {target}"
        
        return self.run(user_input)
    
    def full_pipeline(self, data_path: str, target_column: str, task_type: str = "classification") -> Dict[str, Any]:
        """
        执行完整建模流程
        
        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型
            
        Returns:
            完整流程结果
        """
        self.set_data_context(data_path, target_column, task_type)
        
        user_input = f"""请完成完整的建模流程：
1. 加载数据: {data_path}
2. 分析数据质量和分布
3. 生成特征
4. 训练{task_type}模型，目标列是 {target_column}

请按步骤执行，并在最后总结建模结果。"""
        
        return self.run(user_input)
