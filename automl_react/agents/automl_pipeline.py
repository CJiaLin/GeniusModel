"""
AutoML Pipeline Orchestrator

负责协调整个 AutoML 工作流程，调用各阶段的专门 Agent
"""

from typing import Any, Dict, Optional
from datetime import datetime

from .data_analysis_agent import DataAnalysisAgent
from .data_cleaning_agent import DataCleaningAgent
from .feature_engineering_agent import FeatureEngineeringAgent
from .model_training_agent import ModelTrainingAgent


class AutoMLPipeline:
    """
    AutoML Pipeline 编排器
    
    负责协调各阶段的 Agent，控制整个工作流程：
    1. 数据分析 → DataAnalysisAgent
    2. 数据清洗 → DataCleaningAgent
    3. 特征工程 → FeatureEngineeringAgent
    4. 模型训练 → ModelTrainingAgent
    
    Attributes:
        llm: 语言模型实例
        session_id: 会话ID
        agents: 各阶段的 Agent 实例
    """
    
    def __init__(self, llm: Any = None, session_id: str = None, verbose: bool = False):
        self.llm = llm
        self.session_id = session_id
        self.verbose = verbose
        
        # 初始化各阶段 Agent
        self.agents = {
            "analysis": DataAnalysisAgent(llm=llm, session_id=session_id, verbose=verbose),
            "cleaning": DataCleaningAgent(llm=llm, session_id=session_id, verbose=verbose),
            "feature": FeatureEngineeringAgent(llm=llm, session_id=session_id, verbose=verbose),
            "model": ModelTrainingAgent(llm=llm, session_id=session_id, verbose=verbose)
        }
        
        # 工作流状态
        self.current_stage = "data_analysis"
        self.completed_stages = []
        self.results = {}
    
    def run_stage(self, stage: str, **kwargs) -> Dict[str, Any]:
        """
        运行指定阶段
        
        Args:
            stage: 阶段名称 (data_analysis, data_cleaning, feature_engineering, model_training)
            **kwargs: 阶段参数
            
        Returns:
            阶段执行结果
        """
        stage_map = {
            "data_analysis": self._run_data_analysis,
            "data_cleaning": self._run_data_cleaning,
            "feature_engineering": self._run_feature_engineering,
            "model_training": self._run_model_training
        }
        
        if stage not in stage_map:
            return {"success": False, "error": f"未知阶段: {stage}"}
        
        return stage_map[stage](**kwargs)
    
    def _run_data_analysis(self, data_path: str, **kwargs) -> Dict[str, Any]:
        """运行数据分析阶段"""
        agent = self.agents["analysis"]
        result = agent.analyze(data_path)
        self.results["data_analysis"] = result
        self.completed_stages.append("data_analysis")
        self.current_stage = "data_cleaning"
        return result
    
    def _run_data_cleaning(self, data_path: str, **kwargs) -> Dict[str, Any]:
        """运行数据清洗阶段"""
        agent = self.agents["cleaning"]
        result = agent.generate_cleaning_plan(data_path)
        self.results["data_cleaning"] = result
        self.completed_stages.append("data_cleaning")
        self.current_stage = "feature_engineering"
        return result
    
    def _run_feature_engineering(self, data_path: str, target_column: str, task_type: str, **kwargs) -> Dict[str, Any]:
        """运行特征工程阶段"""
        agent = self.agents["feature"]
        result = agent.generate_feature_plan(data_path, target_column, task_type)
        self.results["feature_engineering"] = result
        self.completed_stages.append("feature_engineering")
        self.current_stage = "model_training"
        return result
    
    def _run_model_training(self, data_path: str, target_column: str, task_type: str, **kwargs) -> Dict[str, Any]:
        """运行模型训练阶段"""
        agent = self.agents["model"]
        result = agent.generate_model_plan(data_path, target_column, task_type)
        self.results["model_training"] = result
        self.completed_stages.append("model_training")
        return result
    
    def run_full_pipeline(self, data_path: str, target_column: str, task_type: str = "classification") -> Dict[str, Any]:
        """
        运行完整的 AutoML 流程
        
        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型
            
        Returns:
            完整流程结果
        """
        pipeline_results = {}
        
        # 1. 数据分析
        print("[Pipeline] 阶段 1/4: 数据分析")
        analysis_result = self._run_data_analysis(data_path)
        pipeline_results["data_analysis"] = analysis_result
        
        # 2. 数据清洗
        print("[Pipeline] 阶段 2/4: 数据清洗")
        cleaning_result = self._run_data_cleaning(data_path)
        pipeline_results["data_cleaning"] = cleaning_result
        
        # 3. 特征工程
        print("[Pipeline] 阶段 3/4: 特征工程")
        feature_result = self._run_feature_engineering(data_path, target_column, task_type)
        pipeline_results["feature_engineering"] = feature_result
        
        # 4. 模型训练
        print("[Pipeline] 阶段 4/4: 模型训练")
        model_result = self._run_model_training(data_path, target_column, task_type)
        pipeline_results["model_training"] = model_result
        
        return {
            "success": True,
            "pipeline": pipeline_results,
            "completed_stages": self.completed_stages,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_agent(self, stage: str):
        """获取指定阶段的 Agent"""
        return self.agents.get(stage)
    
    def get_status(self) -> Dict[str, Any]:
        """获取 Pipeline 状态"""
        return {
            "current_stage": self.current_stage,
            "completed_stages": self.completed_stages,
            "results": {k: v.get("success", False) for k, v in self.results.items()}
        }
