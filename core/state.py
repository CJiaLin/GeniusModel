"""
AutoML 流程状态管理
"""
from enum import Enum
from typing import Dict, Any, Optional, List
import pandas as pd
import json
from datetime import datetime


class StateStep(Enum):
    """流程步骤枚举"""
    INIT = "初始化"
    DATA_LOADING = "数据加载"
    DATA_PREPARATION = "数据预处理"
    FEATURE_ENGINEERING = "特征工程"
    MODEL_TRAINING = "模型训练"
    MODEL_EVALUATION = "模型评估"
    COMPLETED = "完成"
    FAILED = "失败"


class PipelineState:
    """AutoML 流程状态管理类"""
    
    def __init__(self, goal: str = "", target_column: str = ""):
        """
        初始化流程状态
        
        Args:
            goal: 用户建模目标
            target_column: 目标列
        """
        self.goal = goal
        self.target_column = target_column
        self.current_step = StateStep.INIT
        self.data: Optional[pd.DataFrame] = None
        self.features: List[str] = []
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.logs: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []  # 记录工具调用
        self.agent_actions: List[Dict[str, Any]] = []  # 记录 Agent 操作
        self.executed_code: List[str] = []  # 记录执行的代码
        self.data_path: Optional[str] = None  # 数据文件路径
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
    def update_step(self, step: StateStep):
        """更新当前步骤"""
        self.current_step = step
        self.updated_at = datetime.now()
        self.log(f"进入步骤: {step.value}")
        
    def set_data(self, data: pd.DataFrame):
        """设置数据"""
        self.data = data
        self.log(f"数据设置完成，形状: {data.shape}")
        
    def add_feature(self, feature_name: str):
        """添加特征"""
        if feature_name not in self.features:
            self.features.append(feature_name)
            self.log(f"新增特征: {feature_name}")
            
    def add_features(self, feature_names: List[str]):
        """批量添加特征"""
        for feature_name in feature_names:
            self.add_feature(feature_name)
            
    def set_model(self, model_name: str, model: Any):
        """设置模型"""
        self.models[model_name] = model
        self.log(f"模型已设置: {model_name}")
        
    def set_result(self, key: str, value: Any):
        """设置结果"""
        self.results[key] = value
        self.log(f"结果已设置: {key}")
        
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        log_entry = {
            "timestamp": datetime.now(),
            "level": level,
            "message": message,
            "step": self.current_step.value
        }
        self.logs.append(log_entry)
    
    def log_tool_call(self, tool_name: str, params: Dict[str, Any], result: Any = None):
        """记录工具调用"""
        tool_call = {
            "timestamp": datetime.now(),
            "tool_name": tool_name,
            "params": params,
            "result_type": type(result).__name__ if result is not None else "None",
            "step": self.current_step.value
        }
        self.tool_calls.append(tool_call)
    
    def log_agent_action(self, agent_name: str, action: str, details: Dict[str, Any] = None):
        """记录 Agent 操作"""
        agent_action = {
            "timestamp": datetime.now(),
            "agent_name": agent_name,
            "action": action,
            "details": details or {},
            "step": self.current_step.value
        }
        self.agent_actions.append(agent_action)
    
    def log_executed_code(self, code: str):
        """记录执行的代码"""
        self.executed_code.append(code)
        
    def get_log_messages(self) -> List[str]:
        """获取日志消息列表"""
        return [f"[{log['timestamp'].strftime('%H:%M:%S')}] {log['message']}" 
                for log in self.logs]
                
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "goal": self.goal,
            "target_column": self.target_column,
            "current_step": self.current_step.name,
            "features": self.features,
            "results": self.results,
            "logs": [
                {
                    "timestamp": log["timestamp"].isoformat(),
                    "level": log["level"],
                    "message": log["message"],
                    "step": log["step"]
                }
                for log in self.logs
            ],
            "tool_calls": [
                {
                    "timestamp": tc["timestamp"].isoformat(),
                    "tool_name": tc["tool_name"],
                    "params": tc["params"],
                    "result_type": tc["result_type"],
                    "step": tc["step"]
                }
                for tc in self.tool_calls
            ],
            "agent_actions": [
                {
                    "timestamp": aa["timestamp"].isoformat(),
                    "agent_name": aa["agent_name"],
                    "action": aa["action"],
                    "details": aa["details"],
                    "step": aa["step"]
                }
                for aa in self.agent_actions
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            # 注意：DataFrame 不直接序列化，只保存形状信息
            "data_shape": self.data.shape if self.data is not None else None,
            "data_path": self.data_path,
            "executed_code_count": len(self.executed_code)
        }
        
    def save_to_file(self, filepath: str):
        """保存状态到文件"""
        state_dict = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2, default=str)
            
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PipelineState':
        """从文件加载状态"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)
            
        state = cls(
            goal=state_dict.get("goal", ""),
            target_column=state_dict.get("target_column", "")
        )
        state.current_step = StateStep[state_dict["current_step"]]
        state.features = state_dict.get("features", [])
        state.results = state_dict.get("results", {})
        
        # 恢复日志
        for log_dict in state_dict.get("logs", []):
            log_entry = {
                "timestamp": datetime.fromisoformat(log_dict["timestamp"]),
                "level": log_dict["level"],
                "message": log_dict["message"],
                "step": log_dict["step"]
            }
            state.logs.append(log_entry)
            
        state.created_at = datetime.fromisoformat(state_dict["created_at"])
        state.updated_at = datetime.fromisoformat(state_dict["updated_at"])
        
        return state