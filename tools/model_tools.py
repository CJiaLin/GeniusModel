"""
模型训练工具 - MCP 标准化工具
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from tools import register_tool


@register_tool(
    name="train_model",
    description="训练机器学习模型",
    category="model",
    parameters={
        "X_train": "pd.DataFrame - 训练集特征",
        "y_train": "pd.Series - 训练集目标",
        "model_type": "str - 模型类型 (rf, xgb, lgbm, lr, svc)",
        "task_type": "str - 任务类型 (classification, regression)",
        "params": "dict - 模型参数"
    },
    returns="object - 训练好的模型"
)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "rf",
    task_type: str = "classification",
    params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    训练模型
    
    Args:
        X_train: 训练集特征
        y_train: 训练集目标
        model_type: 模型类型
        task_type: 任务类型
        params: 模型参数
        
    Returns:
        训练好的模型
    """
    params = params or {}
    
    if model_type == "rf":
        # 随机森林
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if task_type == "classification":
            model = RandomForestClassifier(**params, random_state=42)
        else:
            model = RandomForestRegressor(**params, random_state=42)
            
    elif model_type == "xgb":
        # XGBoost
        try:
            import xgboost as xgb
            if task_type == "classification":
                model = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False)
            else:
                model = xgb.XGBRegressor(**params, random_state=42)
        except ImportError:
            raise ImportError("请安装 xgboost: pip install xgboost")
            
    elif model_type == "lgbm":
        # LightGBM
        try:
            import lightgbm as lgb
            if task_type == "classification":
                model = lgb.LGBMClassifier(**params, random_state=42)
            else:
                model = lgb.LGBMRegressor(**params, random_state=42)
        except ImportError:
            raise ImportError("请安装 lightgbm: pip install lightgbm")
            
    elif model_type == "lr":
        # 逻辑回归/线性回归
        from sklearn.linear_model import LogisticRegression, LinearRegression
        if task_type == "classification":
            model = LogisticRegression(**params, random_state=42, max_iter=1000)
        else:
            model = LinearRegression(**params)
            
    elif model_type == "svc":
        # 支持向量机
        from sklearn.svm import SVC, SVR
        if task_type == "classification":
            model = SVC(**params, random_state=42)
        else:
            model = SVR(**params)
            
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")
    
    # 训练模型
    model.fit(X_train, y_train)
    return model


@register_tool(
    name="predict",
    description="使用模型进行预测",
    category="model",
    parameters={
        "model": "object - 训练好的模型",
        "X": "pd.DataFrame - 要预测的特征数据"
    },
    returns="np.ndarray - 预测结果"
)
def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    模型预测
    
    Args:
        model: 训练好的模型
        X: 特征数据
        
    Returns:
        预测结果
    """
    return model.predict(X)


@register_tool(
    name="predict_proba",
    description="获取预测概率",
    category="model",
    parameters={
        "model": "object - 训练好的模型",
        "X": "pd.DataFrame - 要预测的特征数据"
    },
    returns="np.ndarray - 预测概率"
)
def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    获取预测概率
    
    Args:
        model: 训练好的模型
        X: 特征数据
        
    Returns:
        预测概率
    """
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    else:
        raise ValueError(f"模型 {type(model).__name__} 不支持 predict_proba")


@register_tool(
    name="get_feature_importance",
    description="获取特征重要性",
    category="model",
    parameters={
        "model": "object - 训练好的模型",
        "feature_names": "list - 特征名称列表"
    },
    returns="dict - 特征重要性字典"
)
def get_feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
    """
    获取特征重要性
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称
        
    Returns:
        特征重要性字典
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return {name: float(imp) for name, imp in zip(feature_names, importances)}
    elif hasattr(model, 'coef_'):
        coefs = model.coef_
        if len(coefs.shape) == 1:
            return {name: float(coef) for name, coef in zip(feature_names, coefs)}
        else:
            # 多分类
            return {name: float(np.mean(coefs[:, i])) for i, name in enumerate(feature_names)}
    else:
        raise ValueError(f"模型 {type(model).__name__} 不支持特征重要性")


@register_tool(
    name="tune_hyperparameters",
    description="超参数调优（网格搜索）",
    category="model",
    parameters={
        "model": "object - 基础模型",
        "X_train": "pd.DataFrame - 训练集特征",
        "y_train": "pd.Series - 训练集目标",
        "param_grid": "dict - 参数网格",
        "cv": "int - 交叉验证折数"
    },
    returns="dict - 最佳参数和最佳模型"
)
def tune_hyperparameters(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, List[Any]],
    cv: int = 5
) -> Dict[str, Any]:
    """
    超参数调优
    
    Args:
        model: 基础模型
        X_train: 训练集特征
        y_train: 训练集目标
        param_grid: 参数网格
        cv: 交叉验证折数
        
    Returns:
        包含最佳参数和最佳模型的字典
    """
    from sklearn.model_selection import GridSearchCV
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=None,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_model": grid_search.best_estimator_
    }


@register_tool(
    name="cross_validate",
    description="交叉验证",
    category="model",
    parameters={
        "model": "object - 模型",
        "X": "pd.DataFrame - 特征数据",
        "y": "pd.Series - 目标数据",
        "cv": "int - 交叉验证折数",
        "scoring": "str - 评估指标"
    },
    returns="dict - 交叉验证结果"
)
def cross_validate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: Optional[str] = None
) -> Dict[str, Any]:
    """
    交叉验证
    
    Args:
        model: 模型
        X: 特征数据
        y: 目标数据
        cv: 交叉验证折数
        scoring: 评估指标
        
    Returns:
        交叉验证结果字典
    """
    from sklearn.model_selection import cross_validate as cv_func
    
    scores = cv_func(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
    
    return {
        "test_scores": scores['test_score'].tolist(),
        "train_scores": scores['train_score'].tolist(),
        "mean_test_score": float(scores['test_score'].mean()),
        "std_test_score": float(scores['test_score'].std()),
        "mean_train_score": float(scores['train_score'].mean()),
        "fit_time": scores['fit_time'].tolist()
    }


@register_tool(
    name="save_model",
    description="保存模型到文件",
    category="model",
    parameters={
        "model": "object - 训练好的模型",
        "filepath": "str - 保存路径"
    },
    returns="str - 保存的文件路径"
)
def save_model(model: Any, filepath: str) -> str:
    """
    保存模型
    
    Args:
        model: 训练好的模型
        filepath: 保存路径
        
    Returns:
        保存的文件路径
    """
    import joblib
    joblib.dump(model, filepath)
    return filepath


@register_tool(
    name="load_model",
    description="从文件加载模型",
    category="model",
    parameters={
        "filepath": "str - 模型文件路径"
    },
    returns="object - 加载的模型"
)
def load_model(filepath: str) -> Any:
    """
    加载模型
    
    Args:
        filepath: 模型文件路径
        
    Returns:
        加载的模型
    """
    import joblib
    return joblib.load(filepath)


@register_tool(
    name="get_model_info",
    description="获取模型信息",
    category="model",
    parameters={
        "model": "object - 模型对象"
    },
    returns="dict - 模型信息字典"
)
def get_model_info(model: Any) -> Dict[str, Any]:
    """
    获取模型信息
    
    Args:
        model: 模型对象
        
    Returns:
        模型信息字典
    """
    return {
        "type": type(model).__name__,
        "parameters": model.get_params() if hasattr(model, 'get_params') else {},
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else None
    }
