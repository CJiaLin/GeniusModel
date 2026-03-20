"""
AutoML 总控模块

协调多个 Agent 完成复杂任务
"""

from typing import Any, Dict, Optional
from enum import Enum

from .automl_agent import AutoMLAgent
from ..core.memory import Memory


class WorkflowStage(Enum):
    """工作流阶段"""
    IDLE = "idle"
    DATA_LOADING = "data_loading"
    DATA_ANALYSIS = "data_analysis"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    COMPLETED = "completed"
    ERROR = "error"


class AutoMLOrchestrator:
    """
    AutoML 总控器
    
    协调多个 Agent 完成复杂建模任务
    
    Attributes:
        llm: 语言模型实例
        agent: AutoML Agent 实例
        memory: 记忆管理器
        current_stage: 当前阶段
        data_path: 数据路径
        target_column: 目标列名
        task_type: 任务类型
    """
    
    def __init__(self, llm: Any = None, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        self.agent = AutoMLAgent(llm=llm, verbose=verbose)
        self.memory = Memory()
        self.current_stage = WorkflowStage.IDLE
        
        # 数据上下文
        self.data_path: Optional[str] = None
        self.target_column: Optional[str] = None
        self.task_type: str = "classification"
        
        # 执行历史
        self.execution_history: list = []
    
    def set_context(self, data_path: str, target_column: str, task_type: str = "classification"):
        """设置上下文"""
        self.data_path = data_path
        self.target_column = target_column
        self.task_type = task_type
        
        self.agent.set_data_context(data_path, target_column, task_type)
        
        if self.verbose:
            print(f"[Orchestrator] 设置上下文:")
            print(f"  数据路径: {data_path}")
            print(f"  目标列: {target_column}")
            print(f"  任务类型: {task_type}")
    
    def run_stage(self, stage: WorkflowStage, **kwargs) -> Dict[str, Any]:
        """
        执行特定阶段
        
        Args:
            stage: 工作流阶段
            **kwargs: 额外参数
            
        Returns:
            执行结果
        """
        self.current_stage = stage
        
        if self.verbose:
            print(f"\n[Orchestrator] 执行阶段: {stage.value}")
        
        if stage == WorkflowStage.DATA_ANALYSIS:
            result = self.agent.analyze(self.data_path)
        elif stage == WorkflowStage.FEATURE_ENGINEERING:
            result = self.agent.generate_features(
                self.data_path,
                self.target_column,
                self.task_type
            )
        elif stage == WorkflowStage.MODEL_TRAINING:
            result = self.agent.train(
                self.data_path,
                self.target_column,
                self.task_type
            )
        else:
            result = {"success": False, "error": f"未知阶段: {stage}"}
        
        # 记录执行历史
        self.execution_history.append({
            "stage": stage.value,
            "result": result
        })
        
        return result
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        执行完整流程
        
        Returns:
            完整流程结果
        """
        if not self.data_path or not self.target_column:
            raise ValueError("请先设置数据上下文")
        
        if self.verbose:
            print("\n" + "="*60)
            print("[Orchestrator] 开始完整建模流程")
            print("="*60)
        
        return self.agent.full_pipeline(
            self.data_path,
            self.target_column,
            self.task_type
        )
    
    def chat(self, message: str) -> Dict[str, Any]:
        """
        对话式交互
        
        Args:
            message: 用户消息
            
        Returns:
            响应结果
        """
        # 添加上下文信息
        context = ""
        if self.data_path:
            context += f"\n当前数据: {self.data_path}"
        if self.target_column:
            context += f"\n目标列: {self.target_column}"
        if self.task_type:
            context += f"\n任务类型: {self.task_type}"
        
        full_message = f"{message}{context}"
        
        return self.agent.run(full_message)
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "current_stage": self.current_stage.value,
            "data_path": self.data_path,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "execution_count": len(self.execution_history)
        }
    
    def reset(self):
        """重置状态"""
        self.agent.reset()
        self.memory.clear()
        self.current_stage = WorkflowStage.IDLE
        self.execution_history.clear()
        self.data_path = None
        self.target_column = None
        self.task_type = "classification"
