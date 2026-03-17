"""
AutoML Agent 数据模型定义模块

本模块使用Pydantic定义了系统中所有的数据模型（Pydantic Models）。
这些模型用于：
1. 在各个Agent之间传递结构化的数据
2. API请求和响应的数据验证
3. 内部状态和结果的管理

每个模型都包含类型提示和默认值，确保数据的完整性和一致性。

注意：使用Optional[]保持Python 3.8兼容性
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from .enums import ModelingTaskType, DataSourceType, EvaluationMetric, ProcessStatus


class DataSource(BaseModel):
    """
    数据源配置模型
    
    用于描述数据的来源和连接信息。
    支持多种数据源类型，包括本地文件、URL和数据库。
    
    Attributes:
        source_type: 数据源类型（CSV、Excel、JSON等）
        file_path: 本地文件路径
        url: 网络数据URL
        db_config: 数据库连接配置
    """
    source_type: DataSourceType          # 数据源类型
    file_path: Optional[str] = None         # 本地文件路径
    url: Optional[str] = None               # 网络URL
    db_config: Optional[dict] = None        # 数据库连接配置


class DataProfile(BaseModel):
    """
    数据画像/概览模型
    
    存储数据集的基本统计信息和结构特征。
    用于数据探索阶段的初步分析结果展示。
    
    Attributes:
        shape: 数据集形状（行数，列数）
        columns: 列名列表
        dtypes: 每列的数据类型
        missing_values: 每列的缺失值数量
        numeric_columns: 数值型列名列表
        categorical_columns: 类别型列名列表
        target_column: 目标变量列名
    """
    shape: tuple = (0, 0)                    # 数据集形状 (行数, 列数)
    columns: list = []                        # 所有列名
    dtypes: dict = {}                         # 列名到数据类型的映射
    missing_values: dict = {}                 # 列名到缺失值数量的映射
    numeric_columns: list = []                # 数值型列名列表
    categorical_columns: list = []             # 类别型列名列表
    target_column: Optional[str] = None          # 目标变量列名


class DataQualityReport(BaseModel):
    """
    数据质量报告模型
    
    存储数据质量分析的详细结果。
    包含缺失值、重复值、异常值和数据分布等信息。
    
    Attributes:
        missing_analysis: 缺失值分析结果
        duplicate_count: 重复记录数量
        outlier_analysis: 异常值分析结果
        distribution_analysis: 数据分布分析结果
    """
    missing_analysis: dict = {}           # 缺失值分析
    duplicate_count: int = 0              # 重复记录数量
    outlier_analysis: dict = {}           # 异常值分析
    distribution_analysis: dict = {}      # 数据分布分析


class ModelingGoal(BaseModel):
    """
    建模目标模型
    
    存储用户的建模需求和目标。
    由总控Agent通过分析用户输入生成。
    
    Attributes:
        task_type: 建模任务类型（分类、回归等）
        target_column: 目标变量列名
        description: 建模目标的文字描述
        evaluation_metrics: 评估指标列表
        business_constraints: 业务约束条件
    """
    task_type: ModelingTaskType              # 建模任务类型
    target_column: str                       # 目标变量列名
    description: str                         # 建模目标描述
    evaluation_metrics: list = Field(
        default_factory=lambda: [EvaluationMetric.ACCURACY]  # 默认评估指标
    )
    business_constraints: dict = Field(
        default_factory=dict  # 业务约束条件
    )


class ModelingPlan(BaseModel):
    """
    建模计划模型
    
    存储完整的建模计划，包括数据要求、特征工程步骤和推荐模型等。
    由总控Agent根据建模目标生成。
    
    Attributes:
        goal: 建模目标对象
        required_data_quality: 所需的数据质量要求
        feature_engineering_steps: 特征工程步骤列表
        suggested_models: 推荐的模型列表
        estimated_complexity: 预估复杂度（低/中/高）
    """
    goal: ModelingGoal                       # 建模目标
    required_data_quality: dict = {}    # 数据质量要求
    feature_engineering_steps: list = []     # 特征工程步骤
    suggested_models: list = []              # 推荐的模型列表
    estimated_complexity: str = "medium"  # 预估复杂度


class ProcessState(BaseModel):
    """
    流程状态模型
    
    用于跟踪和管理AutoML建模流程的执行状态。
    记录当前步骤、进度百分比、消息和错误信息。
    
    Attributes:
        status: 当前流程状态（待处理/运行中/已完成/失败）
        current_step: 当前正在执行的步骤名称
        progress: 进度百分比（0.0-1.0）
        message: 状态消息
        artifacts: 流程产物（中间结果）
        errors: 错误信息列表
    """
    status: ProcessStatus = ProcessStatus.PENDING  # 流程状态
    current_step: str = ""                # 当前步骤名称
    progress: float = 0.0                 # 进度百分比 (0.0-1.0)
    message: str = ""                     # 状态消息
    artifacts: dict = Field(
        default_factory=dict  # 流程产物/中间结果
    )
    errors: list = Field(
        default_factory=list  # 错误信息列表
    )


class ModelingResult(BaseModel):
    """
    建模结果模型
    
    存储建模完成后的最终结果，包括最佳模型、评估指标和训练时间等。
    
    Attributes:
        best_model: 训练完成的最优模型对象
        metrics: 模型评估指标字典
        feature_importance: 特征重要性字典
        training_time: 模型训练耗时（秒）
        artifacts: 其他产物（模型文件、配置等）
    """
    best_model: Any = None                        # 最优模型对象
    metrics: dict = {}              # 评估指标
    feature_importance: Optional[dict] = None  # 特征重要性
    training_time: float = 0.0                   # 训练时间（秒）
    artifacts: dict = Field(
        default_factory=dict  # 其他产物
    )
