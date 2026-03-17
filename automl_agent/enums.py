"""
AutoML Agent 枚举类型定义模块

本模块定义了系统中使用的所有枚举类型，包括建模任务类型、数据源类型、模型类别、
评估指标和流程状态等。这些枚举类型用于确保代码的类型安全和一致性。
"""

from enum import Enum


class ModelingTaskType(str, Enum):
    """
    建模任务类型枚举
    
    定义了AutoML系统支持的不同类型的机器学习任务。
    每种任务类型对应不同的模型选择策略和评估指标。
    """
    CLASSIFICATION = "classification"          # 二分类/多分类任务
    REGRESSION = "regression"                  # 回归任务
    CLUSTERING = "clustering"                  # 聚类任务
    TIME_SERIES = "time_series"                # 时间序列预测
    ANOMALY_DETECTION = "anomaly_detection"   # 异常检测


class DataSourceType(str, Enum):
    """
    数据源类型枚举
    
    定义了系统支持的不同数据来源类型。
    用于数据加载模块识别和处理不同格式的数据文件。
    """
    CSV = "csv"            # CSV格式文件
    EXCEL = "excel"        # Excel格式文件
    JSON = "json"          # JSON格式文件
    DATABASE = "database"  # 数据库连接
    URL = "url"            # 网络URL数据源


class ModelCategory(str, Enum):
    """
    模型类别枚举
    
    按照算法原理对机器学习模型进行分类。
    用于模型选择和推荐时的参考。
    """
    LINEAR = "linear"                      # 线性模型（线性回归、逻辑回归等）
    TREE = "tree"                          # 树模型（决策树、随机森林等）
    ENSEMBLE = "ensemble"                  # 集成模型（Boosting、Bagging等）
    NEURAL_NETWORK = "neural_network"      # 神经网络模型
    OTHER = "other"                        # 其他类型模型


class EvaluationMetric(str, Enum):
    """
    模型评估指标枚举
    
    定义了不同任务类型常用的评估指标。
    这些指标用于衡量模型性能和进行模型比较。
    """
    ACCURACY = "accuracy"      # 准确率（分类任务）
    PRECISION = "precision"     # 精确率（分类任务）
    RECALL = "recall"          # 召回率（分类任务）
    F1 = "f1"                  # F1分数（分类任务）
    AUC = "auc"                # ROC-AUC（分类任务）
    RMSE = "rmse"              # 均方根误差（回归任务）
    MAE = "mae"                # 平均绝对误差（回归任务）
    R2 = "r2"                  # 决定系数R2（回归任务）
    SILHOUETTE = "silhouette"  # 轮廓系数（聚类任务）


class ProcessStatus(str, Enum):
    """
    流程状态枚举
    
    用于跟踪和管理AutoML建模流程的执行状态。
    帮助用户了解当前建模进度和识别问题。
    """
    PENDING = "pending"      # 等待执行
    RUNNING = "running"     # 正在执行
    COMPLETED = "completed" # 已完成
    FAILED = "failed"       # 执行失败
