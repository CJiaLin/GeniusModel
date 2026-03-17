"""
模型Agent模块

本模块是AutoML系统中的模型组件，负责模型选择、训练、调参和评估。
它是建模流程的最后阶段，在特征工程完成后执行模型相关操作。

主要组件：
1. ModelSelector - 模型选择器（根据任务类型和数据特征推荐模型）
2. HyperparameterTuner - 超参数调优器（提供各模型的参数网格）
3. ModelTrainer - 模型训练器（训练各种类型的机器学习模型）
4. ModelEvaluator - 模型评估器（计算各种评估指标）
5. ModelComparator - 模型比较器（比较多个模型的性能）
6. ModelAgent - 模型Agent主类

工作流程：
1. 根据任务类型选择合适的模型
2. 获取模型超参数网格
3. 训练多个候选模型
4. 评估模型性能
5. 选择最佳模型并返回结果
"""

import pandas as pd
import numpy as np
from typing import Any, Optional
from langchain_openai import ChatOpenAI
import time
import joblib

# 修复导入路径
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from automl_agent.models import ModelingGoal, ModelingResult, ProcessState
from automl_agent.enums import ModelingTaskType


class ModelSelector:
    """
    模型选择器
    
    根据建模任务类型和数据特征，选择合适的机器学习模型。
    提供基于任务类型和数据规模的推荐。
    
    Attributes:
        llm: 大语言模型实例（可选）
        CLASSIFICATION_MODELS: 支持的分类模型列表
        REGRESSION_MODELS: 支持的回归模型列表
        CLUSTERING_MODELS: 支持的聚类模型列表
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化模型选择器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm

    # 支持的分类模型
    CLASSIFICATION_MODELS = [
        "LogisticRegression",   # 逻辑回归
        "RandomForest",        # 随机森林
        "XGBoost",            # XGBoost
        "LightGBM",           # LightGBM
        "SVM",                # 支持向量机
        "KNN",               # K近邻
        "NaiveBayes",         # 朴素贝叶斯
        "GradientBoosting"    # 梯度提升
    ]

    # 支持的回归模型
    REGRESSION_MODELS = [
        "LinearRegression",    # 线性回归
        "Ridge",             # 岭回归
        "Lasso",             # Lasso回归
        "RandomForest",       # 随机森林
        "XGBoost",           # XGBoost
        "LightGBM",          # LightGBM
        "SVR"                # 支持向量回归
    ]

    # 支持的聚类模型
    CLUSTERING_MODELS = [
        "KMeans",             # K均值聚类
        "DBSCAN",            # 密度聚类
        "AgglomerativeClustering",  # 层次聚类
        "GaussianMixture"     # 高斯混合模型
    ]

    def select(self, task_type: ModelingTaskType, n_models: int = 3) -> list[str]:
        """
        根据任务类型选择模型
        
        Args:
            task_type: 建模任务类型
            n_models: 返回的模型数量，默认3
            
        Returns:
            list: 选中的模型名称列表
            
        Example:
            >>> selector = ModelSelector()
            >>> models = selector.select(ModelingTaskType.CLASSIFICATION, n_models=3)
            >>> print(models)
            ['LogisticRegression', 'RandomForest', 'XGBoost']
        """
        if task_type == ModelingTaskType.CLASSIFICATION:
            return self.CLASSIFICATION_MODELS[:n_models]
        elif task_type == ModelingTaskType.REGRESSION:
            return self.REGRESSION_MODELS[:n_models]
        elif task_type == ModelingTaskType.CLUSTERING:
            return self.CLUSTERING_MODELS[:n_models]
        else:
            return ["RandomForest"]

    def recommend_for_data(self, task_type: ModelingTaskType, n_samples: int, n_features: int) -> list[str]:
        """
        根据数据特征推荐模型
        
        根据样本数量和特征数量，推荐适合的模型。
        
        Args:
            task_type: 建模任务类型
            n_samples: 样本数量
            n_features: 特征数量
            
        Returns:
            list: 推荐的模型名称列表
            
        Example:
            >>> selector = ModelSelector()
            >>> models = selector.recommend_for_data(ModelingTaskType.CLASSIFICATION, 500, 100)
        """
        recommendations = []
        
        if task_type == ModelingTaskType.CLASSIFICATION:
            # 根据样本量选择
            if n_samples < 1000:
                # 小数据集适合简单模型
                recommendations.extend(["LogisticRegression", "NaiveBayes", "KNN"])
            elif n_features > 50:
                # 高维数据适合树模型
                recommendations.extend(["RandomForest", "XGBoost", "LightGBM"])
            else:
                # 中等规模数据
                recommendations.extend(["RandomForest", "XGBoost", "SVM"])
        
        elif task_type == ModelingTaskType.REGRESSION:
            # 根据特征数量选择
            if n_features > n_samples:
                # 高维数据适合正则化模型
                recommendations.extend(["Ridge", "Lasso"])
            else:
                # 标准情况
                recommendations.extend(["LinearRegression", "RandomForest", "XGBoost"])
        
        return recommendations


class HyperparameterTuner:
    """
    超参数调优器
    
    为各种机器学习模型提供超参数搜索网格。
    用于网格搜索或随机搜索调优。
    
    Attributes:
        llm: 大语言模型实例（可选）
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化超参数调优器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm

    def get_param_grid(self, model_name: str) -> dict[str, list[Any]]:
        """
        获取模型的超参数搜索网格
        
        Args:
            model_name: 模型名称
            
        Returns:
            dict: 超参数名称到候选值的映射字典
            
        Example:
            >>> tuner = HyperparameterTuner()
            >>> grid = tuner.get_param_grid("RandomForest")
            >>> print(grid)
            {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None], ...}
        """
        # 定义各模型的超参数网格
        param_grids = {
            "LogisticRegression": {
                "C": [0.01, 0.1, 1, 10],           # 正则化参数
                "penalty": ["l1", "l2"],             # 正则化类型
                "solver": ["liblinear"]              # 优化算法
            },
            "RandomForest": {
                "n_estimators": [50, 100, 200],     # 树的数量
                "max_depth": [3, 5, 10, None],      # 最大深度
                "min_samples_split": [2, 5, 10]     # 最小分裂样本数
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],     # 树的数量
                "max_depth": [3, 5, 7],             # 最大深度
                "learning_rate": [0.01, 0.1, 0.2],  # 学习率
                "subsample": [0.8, 1.0]             # 子采样比例
            },
            "LightGBM": {
                "n_estimators": [50, 100, 200],     # 树的数量
                "max_depth": [3, 5, 7],             # 最大深度
                "learning_rate": [0.01, 0.1, 0.2],  # 学习率
                "num_leaves": [15, 31, 63]          # 叶节点数量
            },
            "SVM": {
                "C": [0.1, 1, 10],                 # 正则化参数
                "kernel": ["rbf", "linear"],        # 核函数
                "gamma": ["scale", "auto"]          # 核系数
            },
            "KNN": {
                "n_neighbors": [3, 5, 7, 11],      # 近邻数量
                "weights": ["uniform", "distance"], # 权重类型
                "metric": ["euclidean", "manhattan"] # 距离度量
            },
            "LinearRegression": {
                "fit_intercept": [True, False]       # 是否拟合截距
            },
            "Ridge": {
                "alpha": [0.01, 0.1, 1, 10, 100]  # 正则化参数
            },
            "Lasso": {
                "alpha": [0.01, 0.1, 1, 10]       # 正则化参数
            },
            "KMeans": {
                "n_clusters": [2, 3, 5, 8],        # 聚类数量
                "init": ["k-means++", "random"],    # 初始化方法
                "max_iter": [100, 300, 500]        # 最大迭代次数
            }
        }
        
        # 返回模型的参数网格，如果不存在则返回空字典
        return param_grids.get(model_name, {})


class ModelTrainer:
    """
    模型训练器
    
    负责训练各种类型的机器学习模型。
    支持分类、回归和聚类任务。
    
    Attributes:
        llm: 大语言模型实例（可选）
        models: 训练好的模型字典
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化模型训练器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm
        self.models = {}  # 存储训练好的模型

    def train_model(self, model_name: str, X_train, y_train, params: Optional[dict[str, Any]] = None) -> Any:
        """
        训练单个模型
        
        Args:
            model_name: 模型名称
            X_train: 训练特征
            y_train: 训练标签
            params: 模型超参数字典
            
        Returns:
            训练好的模型对象
            
        Raises:
            ValueError: 如果模型不支持
        """
        # 导入sklearn模型
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        from sklearn.mixture import GaussianMixture
        
        # 尝试导入XGBoost
        try:
            import xgboost as xgb
            xgb_available = True
        except ImportError:
            xgb_available = False
        
        # 尝试导入LightGBM
        try:
            import lightgbm as lgb
            lgb_available = True
        except ImportError:
            lgb_available = False
        
        # 默认参数为空字典
        params = params or {}
        
        # 根据模型名称创建对应的模型
        if model_name == "LogisticRegression":
            model = LogisticRegression(**params)
        elif model_name == "RandomForest":
            # 根据目标变量类型选择分类或回归
            model = RandomForestClassifier(**params) if hasattr(y_train, 'nunique') else RandomForestRegressor(**params)
        elif model_name == "XGBoost" and xgb_available:
            model = xgb.XGBClassifier(**params) if hasattr(y_train, 'nunique') else xgb.XGBRegressor(**params)
        elif model_name == "LightGBM" and lgb_available:
            model = lgb.LGBMClassifier(**params) if hasattr(y_train, 'nunique') else lgb.LGBMRegressor(**params)
        elif model_name == "SVM":
            model = SVC(**params) if hasattr(y_train, 'nunique') else SVR(**params)
        elif model_name == "KNN":
            model = KNeighborsClassifier(**params)
        elif model_name == "NaiveBayes":
            model = GaussianNB(**params)
        elif model_name == "LinearRegression":
            model = LinearRegression(**params)
        elif model_name == "Ridge":
            model = Ridge(**params)
        elif model_name == "Lasso":
            model = Lasso(**params)
        elif model_name == "KMeans":
            model = KMeans(**params)
        elif model_name == "GradientBoosting":
            model = GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 训练模型
        model.fit(X_train, y_train)
        # 保存模型
        self.models[model_name] = model
        
        return model


class ModelEvaluator:
    """
    模型评估器
    
    负责评估模型性能，计算各种评估指标。
    支持分类、回归和聚类任务的评估。
    
    Attributes:
        llm: 大语言模型实例（可选）
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化模型评估器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm

    def evaluate_classification(self, model, X_test, y_test) -> dict[str, float]:
        """
        评估分类模型
        
        计算分类任务的多种评估指标。
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            dict: 评估指标字典
            
        Example:
            >>> evaluator = ModelEvaluator()
            >>> metrics = evaluator.evaluate_classification(model, X_test, y_test)
            >>> print(metrics['accuracy'])
            0.85
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算基础指标
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred)  # 准确率
        }
        
        # 计算精确率、召回率、F1
        try:
            metrics["precision"] = precision_score(y_test, y_pred, average="weighted")
            metrics["recall"] = recall_score(y_test, y_pred, average="weighted")
            metrics["f1"] = f1_score(y_test, y_pred, average="weighted")
        except:
            pass
        
        # 计算AUC（仅对二分类）
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    metrics["auc"] = roc_auc_score(y_test, y_proba[:, 1])
        except:
            pass
        
        return metrics

    def evaluate_regression(self, model, X_test, y_test) -> dict[str, float]:
        """
        评估回归模型
        
        计算回归任务的多种评估指标。
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            dict: 评估指标字典
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # 预测
        y_pred = model.predict(X_test)
        
        return {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),  # 均方根误差
            "mae": mean_absolute_error(y_test, y_pred),          # 平均绝对误差
            "r2": r2_score(y_test, y_pred)                     # 决定系数
        }

    def evaluate_clustering(self, model, X_test, y_test=None) -> dict[str, float]:
        """
        评估聚类模型
        
        计算聚类任务的评估指标。
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 可选的测试标签
            
        Returns:
            dict: 评估指标字典
        """
        from sklearn.metrics import silhouette_score
        
        # 获取聚类标签
        labels = model.predict(X_test) if hasattr(model, "predict") else model.labels_
        
        # 计算轮廓系数
        metrics = {
            "silhouette": silhouette_score(X_test, labels)
        }
        
        return metrics

    def evaluate(self, model, X_test, y_test, task_type: ModelingTaskType) -> dict[str, float]:
        """
        通用评估方法
        
        根据任务类型自动选择合适的评估方法。
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            task_type: 任务类型
            
        Returns:
            dict: 评估指标字典
        """
        if task_type == ModelingTaskType.CLASSIFICATION:
            return self.evaluate_classification(model, X_test, y_test)
        elif task_type == ModelingTaskType.REGRESSION:
            return self.evaluate_regression(model, X_test, y_test)
        elif task_type == ModelingTaskType.CLUSTERING:
            return self.evaluate_clustering(model, X_test, y_test)
        else:
            return {}


class ModelComparator:
    """
    模型比较器
    
    负责比较多个模型的性能，找出最佳模型。
    
    Attributes:
        llm: 大语言模型实例（可选）
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化模型比较器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm

    def compare(self, results: dict[str, dict[str, float]]) -> dict[str, Any]:
        """
        比较多个模型的性能
        
        Args:
            results: 模型名称到评估指标的映射字典
            
        Returns:
            dict: 比较结果，包含每个指标的最佳模型
            
        Example:
            >>> comparator = ModelComparator()
            >>> results = {
            ...     "RandomForest": {"accuracy": 0.85, "f1": 0.83},
            ...     "XGBoost": {"accuracy": 0.87, "f1": 0.85}
            ... }
            >>> comparison = comparator.compare(results)
        """
        if not results:
            return {}
        
        # 收集所有指标
        all_metrics = set()
        for model_result in results.values():
            all_metrics.update(model_result.keys())
        
        comparison = {}
        
        # 对每个指标进行比较
        for metric in all_metrics:
            metric_values = {}
            for model_name, model_result in results.items():
                if metric in model_result:
                    metric_values[model_name] = model_result[metric]
            
            if metric_values:
                # 对于误差类指标（越小越好），取负值比较
                # 对于其他指标（越大越好），直接比较
                if metric in ["rmse", "mae"]:
                    best_model = min(metric_values.items(), key=lambda x: x[1])
                else:
                    best_model = max(metric_values.items(), key=lambda x: x[1])
                
                comparison[metric] = {
                    "values": metric_values,          # 所有模型的指标值
                    "best_model": best_model[0],     # 最佳模型名称
                    "best_value": best_model[1]      # 最佳指标值
                }
        
        return comparison


class ModelAgent:
    """
    模型Agent
    
    这是建模阶段的主类，整合了模型选择、训练、调参和评估功能。
    在AutoML流程中负责训练和评估机器学习模型，选择最佳模型。
    
    Attributes:
        llm: 大语言模型实例
        selector: 模型选择器
        tuner: 超参数调优器
        trainer: 模型训练器
        evaluator: 模型评估器
        comparator: 模型比较器
        state: 流程状态
        best_model: 最佳模型
        model_results: 所有模型的评估结果
        
    Example:
        >>> agent = ModelAgent()
        >>> result = agent.run(X_train, y_train, X_test, y_test, goal)
        >>> print(result.metrics)
        {'accuracy': 0.87, 'precision': 0.86, 'f1': 0.85}
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化模型Agent
        
        Args:
            llm: 大语言模型实例
        """
        # 初始化LLM
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        
        # 初始化各组件
        self.selector = ModelSelector(self.llm)           # 模型选择
        self.tuner = HyperparameterTuner(self.llm)      # 超参数调优
        self.trainer = ModelTrainer(self.llm)           # 模型训练
        self.evaluator = ModelEvaluator(self.llm)       # 模型评估
        self.comparator = ModelComparator(self.llm)     # 模型比较
        self.state = ProcessState()                      # 流程状态
        self.best_model: Optional[Any] = None           # 最佳模型
        self.model_results: dict[str, dict[str, float]] = {}  # 模型结果
        
        # LLM驱动的模型训练相关
        self.conversation_history: list[dict] = []     # 对话记忆
        self.training_plan: Optional[dict] = None       # 当前训练方案
    
    def add_to_memory(self, role: str, content: str):
        """添加对话到记忆"""
        self.conversation_history.append({"role": role, "content": content})
    
    def get_memory_context(self) -> str:
        """获取记忆上下文"""
        if not self.conversation_history:
            return ""
        context_parts = ["## 对话历史\n"]
        for item in self.conversation_history[-10:]:
            context_parts.append(f"- {item['role']}: {item['content']}")
        return "\n".join(context_parts)

    def select_models(self, goal: ModelingGoal, n_samples: int, n_features: int) -> list[str]:
        """
        选择模型
        
        Args:
            goal: 建模目标
            n_samples: 样本数量
            n_features: 特征数量
            
        Returns:
            list: 选中的模型名称列表
        """
        self.state.current_step = "选择模型"
        return self.selector.select(goal.task_type, n_models=3)

    def train_models(self, X_train, y_train, model_names: list[str]) -> dict[str, Any]:
        """
        训练多个模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            model_names: 要训练的模型名称列表
            
        Returns:
            dict: 模型名称到训练好模型的映射
        """
        self.state.current_step = "训练模型"
        
        trained_models = {}
        
        # 依次训练每个模型
        for model_name in model_names:
            try:
                # 获取参数网格并使用默认参数
                param_grid = self.tuner.get_param_grid(model_name)
                default_params = {k: v[0] for k, v in param_grid.items()} if param_grid else {}
                
                # 训练模型
                model = self.trainer.train_model(model_name, X_train, y_train, default_params)
                trained_models[model_name] = model
            except Exception as e:
                print(f"训练 {model_name} 失败: {e}")
        
        return trained_models

    def evaluate_models(self, models: dict[str, Any], X_test, y_test, task_type: ModelingTaskType) -> dict[str, dict[str, float]]:
        """
        评估多个模型
        
        Args:
            models: 模型字典
            X_test: 测试特征
            y_test: 测试标签
            task_type: 任务类型
            
        Returns:
            dict: 模型名称到评估指标的映射
        """
        self.state.current_step = "评估模型"
        
        results = {}
        
        # 依次评估每个模型
        for model_name, model in models.items():
            try:
                metrics = self.evaluator.evaluate(model, X_test, y_test, task_type)
                results[model_name] = metrics
            except Exception as e:
                print(f"评估 {model_name} 失败: {e}")
        
        # 保存结果
        self.model_results = results
        return results

    def find_best_model(self) -> tuple[str, Any, dict[str, float]]:
        """
        找出最佳模型
        
        根据评估指标找出性能最好的模型。
        
        Returns:
            tuple: (最佳模型名称, 最佳模型对象, 评估指标)
            
        Raises:
            ValueError: 如果没有可用的模型结果
        """
        if not self.model_results:
            raise ValueError("没有可用的模型结果")
        
        best_model_name = None
        best_score = float('-inf')
        
        # 遍历所有模型结果
        for model_name, metrics in self.model_results.items():
            # 根据任务类型选择主要评估指标
            if "accuracy" in metrics:
                score = metrics["accuracy"]
            elif "r2" in metrics:
                score = metrics["r2"]
            elif "silhouette" in metrics:
                score = metrics["silhouette"]
            else:
                continue
            
            # 记录最佳分数
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        if best_model_name:
            # 尝试找到匹配的实际模型名称
            actual_model_name = self._find_matching_model_key(best_model_name)
            if actual_model_name:
                return actual_model_name, self.trainer.models[actual_model_name], self.model_results[best_model_name]
            elif best_model_name in self.trainer.models:
                return best_model_name, self.trainer.models[best_model_name], self.model_results[best_model_name]
        
        raise ValueError("无法确定最佳模型")
    
    def _find_matching_model_key(self, model_name: str) -> str:
        """查找匹配的实际模型键名"""
        model_name_lower = model_name.lower().replace("_", "").replace("-", "")
        
        for key in self.trainer.models.keys():
            key_lower = key.lower().replace("_", "").replace("-", "")
            if model_name_lower in key_lower or key_lower in model_name_lower:
                return key
        
        return None

    def compare_models(self) -> dict[str, Any]:
        """
        比较所有模型
        
        Returns:
            dict: 模型比较结果
        """
        return self.comparator.compare(self.model_results)

    def run(self, X_train, y_train, X_test, y_test, goal: ModelingGoal) -> ModelingResult:
        """
        运行模型流程
        
        执行完整的模型训练和评估流程。
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
            goal: 建模目标
            
        Returns:
            ModelingResult: 建模结果对象
            
        Example:
            >>> agent = ModelAgent()
            >>> result = agent.run(X_train, y_train, X_test, y_test, goal)
            >>> print(f"Best accuracy: {result.metrics['accuracy']}")
        """
        # 获取数据形状
        n_samples, n_features = X_train.shape
        
        # 步骤1: 选择模型
        model_names = self.select_models(goal, n_samples, n_features)
        
        # 步骤2: 训练模型
        trained_models = self.train_models(X_train, y_train, model_names)
        
        # 步骤3: 评估模型
        results = self.evaluate_models(trained_models, X_test, y_test, goal.task_type)
        
        # 步骤4: 找出最佳模型
        best_name, best_model, best_metrics = self.find_best_model()
        
        # 保存最佳模型
        self.best_model = best_model
        
        # 记录训练时间
        start_time = time.time()
        self.trainer.train_model(best_name, X_train, y_train)
        training_time = time.time() - start_time
        
        # 返回结果
        return ModelingResult(
            best_model=best_model,        # 最佳模型
            metrics=best_metrics,         # 最佳指标
            training_time=training_time    # 训练时间
        )
    
    def generate_training_plan(self, X, y, goal: ModelingGoal, data_info: str = "") -> dict:
        """
        生成模型训练方案（LLM驱动）
        
        Args:
            X: 特征数据
            y: 目标变量
            goal: 建模目标
            data_info: 数据信息描述
            
        Returns:
            dict: 训练方案
        """
        from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
        
        n_samples, n_features = X.shape if hasattr(X, 'shape') else (len(X), 1)
        
        # 直接构建提示词，避免模板问题
        task_desc = f"""你是一位AutoML专家。请根据以下信息设计模型训练方案。

## 建模任务
- 任务类型: {goal.task_type.value}
- 目标变量: {goal.target_column}
- 业务描述: {goal.description or ''}

## 数据信息
- 样本数量: {n_samples}
- 特征数量: {n_features}
- 数据信息: {data_info}

## 数据特征列名
{list(X.columns) if hasattr(X, 'columns') else '无'}

请设计一个完整的模型训练方案。返回JSON格式，包含以下字段：

```json
{{
    "model_choice": {{
        "selected_models": ["LightGBM", "RandomForest", "Ridge"]  // 模型名称列表
    }},
    "data_split": {{
        "train_ratio": 0.8,       // 训练集比例（浮点数）
        "test_ratio": 0.2,        // 测试集比例（浮点数）
        "random_state": 42         // 随机种子（整数）
    }},
    "evaluation_metrics": {{
        "primary_metric": "rmse",     // 主要指标（字符串）
        "secondary_metrics": ["mae", "r2"]  // 次要指标（字符串列表）
    }},
    "hyperparameters": {{
        "LightGBM": {{"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "num_leaves": 31}},
        "RandomForest": {{"n_estimators": 100, "max_depth": 10, "min_samples_split": 5}},
        "Ridge": {{"alpha": 1.0}},
        "XGBoost": {{"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}},
        "SVR": {{"C": 1.0, "kernel": "rbf"}}
    }}
}}
```

**重要约束**：
1. selected_models 必须是简单字符串列表，如 ["LightGBM", "RandomForest"]
2. hyperparameters 的值必须是具体数值/字符串，不能是列表或字典
3. 不要返回超参数搜索空间，只返回具体参数值
4. random_state 必须是整数，如 42
5. train_ratio 和 test_ratio 必须是浮点数，如 0.8

只返回JSON，不要其他内容。
"""
        
        response = self.llm.invoke(task_desc)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # 解析JSON
        import re
        import json
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                self.training_plan = plan
                self.add_to_memory("system", f"生成了训练方案: {plan.get('model_choice', {}).get('selected_models', [])}")
                return plan
        except Exception as e:
            print(f"解析训练方案失败: {e}")
        
        return {}
    
    def modify_training_plan(self, plan: dict, user_feedback: str) -> dict:
        """
        根据用户反馈修改训练方案
        
        Args:
            plan: 当前训练方案
            user_feedback: 用户反馈
            
        Returns:
            dict: 修改后的方案
        """
        memory_context = self.get_memory_context()
        
        from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
        
        prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(f"""你是一位AutoML专家。用户想要修改模型训练方案。

{memory_context}

当前训练方案:
{{plan}}

用户反馈: {{feedback}}

请根据用户反馈，生成修改后的训练方案。保持JSON格式。
""")
        ])
        
        formatted_prompt = prompt.format_messages(
            plan=plan,
            feedback=user_feedback
        )
        
        self.add_to_memory("user", user_feedback)
        
        response = self.llm.invoke(formatted_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        import re
        import json
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                modified_plan = json.loads(json_match.group())
                self.add_to_memory("assistant", f"根据反馈修改了训练方案")
                return modified_plan
        except Exception as e:
            print(f"解析修改后的方案失败: {e}")
        
        return plan
    
    def execute_training_plan(self, plan: dict, X, y) -> ModelingResult:
        """
        执行训练方案
        
        Args:
            plan: 训练方案
            X: 特征数据
            y: 目标变量
            
        Returns:
            ModelingResult: 训练结果
        """
        import json
        from sklearn.model_selection import train_test_split
        
        # 数据划分
        split_config = plan.get("data_split", {})
        train_ratio = split_config.get("train_ratio", 0.8)
        test_ratio = split_config.get("test_ratio", 0.2)
        random_state = split_config.get("random_state", 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state
        )
        
        print(f"\n📊 数据划分: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
        
        # 获取选中的模型
        model_configs = plan.get("hyperparameters", {})
        selected_models_raw = plan.get("model_choice", {}).get("selected_models", [])
        
        # 解析模型名称（可能是字典或字符串）
        selected_models = []
        for m in selected_models_raw:
            if isinstance(m, dict):
                selected_models.append(m.get("name", str(m)))
            else:
                selected_models.append(str(m))
        
        # 也检查hyperparameters的键
        if not selected_models:
            selected_models = list(model_configs.keys())
        
        # 解析model_configs的键
        parsed_configs = {}
        for model_name, params in model_configs.items():
            # 如果键是字典，取name字段
            if isinstance(model_name, dict):
                name = model_name.get("name", str(model_name))
            else:
                name = str(model_name)
            
            # 如果值是字典，取参数
            if isinstance(params, dict):
                parsed_configs[name] = params
            else:
                parsed_configs[name] = {}
        
        model_configs = parsed_configs
        
        self.model_results = {}
        trained_models = {}
        
        # 训练每个模型
        for model_name in selected_models:
            print(f"\n🤖 训练模型: {model_name}")
            try:
                params = model_configs.get(model_name, {})
                
                # 根据模型名称创建模型
                model = self._create_model(model_name, params)
                model.fit(X_train, y_train)
                
                # 评估
                y_pred = model.predict(X_test)
                metrics = self._calculate_metrics(y_test, y_pred, plan.get("evaluation_metrics", {}))
                
                trained_models[model_name] = model
                self.model_results[model_name] = metrics
                
                print(f"  ✓ 训练完成")
                for metric_name, value in metrics.items():
                    print(f"    - {metric_name}: {value:.4f}")
                    
            except Exception as e:
                print(f"  ✗ 训练失败: {e}")
        
        # 选择最佳模型
        if not self.model_results:
            raise ValueError("没有模型训练成功")
        
        best_name, best_model, best_metrics = self.find_best_model()
        
        self.best_model = best_model
        training_time = 0  # 简化的训练时间
        
        return ModelingResult(
            best_model=best_model,
            metrics=best_metrics,
            training_time=training_time
        )
    
    def _create_model(self, model_name: str, params: dict):
        """根据模型名称和参数创建模型"""
        # 转换参数：将列表转为单个值
        cleaned_params = self._clean_params(params)
        
        model_name_lower = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
        
        if "logistic" in model_name_lower:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**cleaned_params)
        elif "linear" in model_name_lower and "ridge" not in model_name_lower and "lasso" not in model_name_lower:
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**cleaned_params)
        elif "ridge" in model_name_lower:
            from sklearn.linear_model import Ridge
            return Ridge(**cleaned_params)
        elif "lasso" in model_name_lower:
            from sklearn.linear_model import Lasso
            return Lasso(**cleaned_params)
        elif "randomforest" in model_name_lower or "rf" in model_name_lower:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**cleaned_params)
        elif "xgboost" in model_name_lower or "xgb" in model_name_lower:
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(**cleaned_params)
            except:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(**cleaned_params)
        elif "lightgbm" in model_name_lower or "lgbm" in model_name_lower:
            try:
                import lightgbm as lgb
                return lgb.LGBMRegressor(**cleaned_params)
            except:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(**cleaned_params)
        elif "supportvector" in model_name_lower or "svr" in model_name_lower or "svm" in model_name_lower:
            from sklearn.svm import SVR
            return SVR(**cleaned_params)
        elif "knn" in model_name_lower:
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor(**cleaned_params)
        elif "gradient" in model_name_lower:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**cleaned_params)
        else:
            # 默认使用随机森林
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**cleaned_params)
    
    def _clean_params(self, params: dict) -> dict:
        """清理参数：将列表/字典转为单个值，处理特殊类型"""
        cleaned = {}
        for key, value in params.items():
            # 处理嵌套字典格式 {'type': 'log_uniform', 'low': 0.001, 'high': 100.0, 'default': 1.0}
            if isinstance(value, dict) and 'type' in value:
                # 取default值
                if 'default' in value:
                    cleaned[key] = value['default']
                # 或者根据type类型取范围值
                elif value.get('type') == 'log_uniform':
                    cleaned[key] = float(value.get('low', 0.1))
                elif value.get('type') == 'uniform':
                    cleaned[key] = float(value.get('low', 0.5))
                elif value.get('type') == 'int':
                    cleaned[key] = int(value.get('default', 5))
                elif value.get('type') == 'categorical':
                    choices = value.get('choices', [])
                    cleaned[key] = choices[0] if choices else 'rbf'
                continue
            
            # 处理普通列表格式
            if isinstance(value, list):
                if value:
                    # 根据参数类型选择合适的值
                    if key in ['n_estimators', 'max_iter', 'num_leaves', 'n_jobs']:
                        cleaned[key] = value[0] if isinstance(value[0], int) else 100
                    elif key in ['max_depth', 'min_samples_split', 'min_samples_leaf']:
                        cleaned[key] = value[0] if isinstance(value[0], int) else 5
                    elif key in ['learning_rate', 'C', 'alpha', 'epsilon', 'gamma']:
                        cleaned[key] = value[0] if isinstance(value[0], (int, float)) else 0.1
                    elif key in ['subsample', 'colsample_bytree', 'feature_fraction', 'bagging_fraction']:
                        cleaned[key] = value[0] if isinstance(value[0], (int, float)) else 0.8
                    elif key == 'kernel' and isinstance(value[0], str):
                        cleaned[key] = value[0]
                    else:
                        cleaned[key] = value[0]
            elif isinstance(value, bool):
                cleaned[key] = value
            elif isinstance(value, (int, float, str)):
                cleaned[key] = value
            else:
                cleaned[key] = value
        
        # 移除不支持的参数
        cleaned.pop('normalize', None)
        cleaned.pop('random_state', None)
        
        return cleaned
    
    def _calculate_metrics(self, y_true, y_pred, metrics_config: dict) -> dict:
        """计算评估指标"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        primary = metrics_config.get("primary_metric", "rmse")
        metrics = {}
        
        # 计算各种指标
        try:
            metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        except:
            pass
        
        try:
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
        except:
            pass
        
        try:
            metrics["r2"] = r2_score(y_true, y_pred)
        except:
            pass
        
        return metrics
