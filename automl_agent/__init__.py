"""
AutoML Agent 核心包初始化模块

该模块定义了AutoML系统的公共API接口，导出主要的类和枚举类型，
方便外部模块直接导入使用。

导出内容:
- AutoMLEngine: 自动化机器学习引擎主类
- ModelingGoal: 建模目标数据模型
- ModelingPlan: 建模计划数据模型
- ModelingResult: 建模结果数据模型
- ProcessState: 流程状态数据模型
- ModelingTaskType: 建模任务类型枚举
- DataSourceType: 数据源类型枚举
- EvaluationMetric: 评估指标枚举
- ProcessStatus: 流程状态枚举
- FeatureSuggestion: 特征建议数据模型
- FeatureEngineeringResult: 特征工程结果数据模型
- LLMFeatureAnalyzer: LLM驱动的智能特征分析器

使用示例:
    from automl_agent import AutoMLEngine, ModelingGoal, ProcessStatus
    from automl_agent.enums import ModelingTaskType
    
    # 使用LLM特征生成
    from agents.feature_engineer import LLMFeatureAnalyzer, FeatureEngineeringResult

作者: AutoML Team
"""

# 导入并导出核心引擎类
from .engine import AutoMLEngine

# 导入并导出数据模型类
from .models import ModelingGoal, ModelingPlan, ModelingResult, ProcessState

# 导入并导出枚举类型
from .enums import ModelingTaskType, DataSourceType, EvaluationMetric, ProcessStatus
