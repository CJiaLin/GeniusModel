"""
模型评估工具 - MCP 标准化工具
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from tools import register_tool


@register_tool(
    name="evaluate_classification",
    description="评估分类模型性能",
    category="eval",
    parameters={
        "y_true": "pd.Series/np.ndarray - 真实标签",
        "y_pred": "pd.Series/np.ndarray - 预测标签",
        "y_proba": "np.ndarray - 预测概率 (可选)",
        "metrics": "list - 要计算的指标列表"
    },
    returns="dict - 评估指标字典"
)
def evaluate_classification(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_proba: Optional[np.ndarray] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    评估分类模型
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_proba: 预测概率
        metrics: 要计算的指标
        
    Returns:
        评估指标字典
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report, confusion_matrix
    )
    
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1"]
    
    results = {}
    
    if "accuracy" in metrics:
        results["accuracy"] = float(accuracy_score(y_true, y_pred))
    
    if "precision" in metrics:
        results["precision"] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    
    if "recall" in metrics:
        results["recall"] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    
    if "f1" in metrics:
        results["f1"] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    if "roc_auc" in metrics and y_proba is not None:
        if len(y_proba.shape) == 1:
            results["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        else:
            results["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted'))
    
    return results


@register_tool(
    name="evaluate_regression",
    description="评估回归模型性能",
    category="eval",
    parameters={
        "y_true": "pd.Series/np.ndarray - 真实值",
        "y_pred": "pd.Series/np.ndarray - 预测值"
    },
    returns="dict - 评估指标字典"
)
def evaluate_regression(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    评估回归模型
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        评估指标字典
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
    
    results = {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred))
    }
    
    return results


@register_tool(
    name="plot_confusion_matrix",
    description="计算混淆矩阵",
    category="eval",
    parameters={
        "y_true": "pd.Series/np.ndarray - 真实标签",
        "y_pred": "pd.Series/np.ndarray - 预测标签",
        "normalize": "bool - 是否归一化"
    },
    returns="dict - 混淆矩阵数据"
)
def plot_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    normalize: bool = False
) -> Dict[str, Any]:
    """
    计算混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        normalize: 是否归一化
        
    Returns:
        包含混淆矩阵数据的字典
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return {
        "matrix": cm.tolist(),
        "shape": cm.shape
    }


@register_tool(
    name="calculate_residuals",
    description="计算残差（回归）",
    category="eval",
    parameters={
        "y_true": "pd.Series/np.ndarray - 真实值",
        "y_pred": "pd.Series/np.ndarray - 预测值"
    },
    returns="dict - 残差分析结果"
)
def calculate_residuals(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, Any]:
    """
    计算残差
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        残差分析结果
    """
    residuals = np.array(y_true) - np.array(y_pred)
    
    return {
        "residuals": residuals.tolist(),
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "min_residual": float(np.min(residuals)),
        "max_residual": float(np.max(residuals))
    }


@register_tool(
    name="compare_models",
    description="比较多个模型的性能",
    category="eval",
    parameters={
        "models": "dict - 模型字典 {name: model}",
        "X_test": "pd.DataFrame - 测试集特征",
        "y_test": "pd.Series - 测试集目标",
        "task_type": "str - 任务类型 (classification, regression)"
    },
    returns="pd.DataFrame - 模型比较结果"
)
def compare_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str = "classification"
) -> pd.DataFrame:
    """
    比较多个模型
    
    Args:
        models: 模型字典
        X_test: 测试集特征
        y_test: 测试集目标
        task_type: 任务类型
        
    Returns:
        模型比较结果 DataFrame
    """
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        if task_type == "classification":
            metrics = evaluate_classification(y_test, y_pred)
        else:
            metrics = evaluate_regression(y_test, y_pred)
        
        results.append({
            "model": name,
            **metrics
        })
    
    return pd.DataFrame(results)


@register_tool(
    name="get_classification_report",
    description="生成详细的分类报告",
    category="eval",
    parameters={
        "y_true": "pd.Series/np.ndarray - 真实标签",
        "y_pred": "pd.Series/np.ndarray - 预测标签",
        "output_dict": "bool - 是否返回字典格式"
    },
    returns="str/dict - 分类报告"
)
def get_classification_report(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    output_dict: bool = False
) -> Union[str, Dict]:
    """
    生成分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        output_dict: 是否返回字典格式
        
    Returns:
        分类报告（字符串或字典）
    """
    from sklearn.metrics import classification_report
    
    return classification_report(y_true, y_pred, output_dict=output_dict, zero_division=0)


@register_tool(
    name="calculate_lift",
    description="计算提升度（分类模型）",
    category="eval",
    parameters={
        "y_true": "pd.Series/np.ndarray - 真实标签",
        "y_proba": "np.ndarray - 预测概率",
        "n_bins": "int - 分箱数量"
    },
    returns="dict - 提升度数据"
)
def calculate_lift(
    y_true: Union[pd.Series, np.ndarray],
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    计算提升度
    
    Args:
        y_true: 真实标签
        y_proba: 预测概率
        n_bins: 分箱数量
        
    Returns:
        提升度数据
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    # 如果是多分类，取正类的概率
    if len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1]
    
    # 按概率排序
    sorted_indices = np.argsort(y_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # 计算累积响应率
    n_total = len(y_true)
    n_positive = np.sum(y_true)
    baseline_rate = n_positive / n_total
    
    lift_data = {
        "decile": [],
        "cumulative_response_rate": [],
        "lift": []
    }
    
    for i in range(1, n_bins + 1):
        cutoff = int(n_total * i / n_bins)
        cumulative_positive = np.sum(y_true_sorted[:cutoff])
        cumulative_rate = cumulative_positive / cutoff
        lift = cumulative_rate / baseline_rate if baseline_rate > 0 else 0
        
        lift_data["decile"].append(i)
        lift_data["cumulative_response_rate"].append(cumulative_rate)
        lift_data["lift"].append(lift)
    
    return lift_data


@register_tool(
    name="get_prediction_distribution",
    description="获取预测值分布统计",
    category="eval",
    parameters={
        "y_pred": "pd.Series/np.ndarray - 预测值"
    },
    returns="dict - 分布统计信息"
)
def get_prediction_distribution(y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    获取预测值分布
    
    Args:
        y_pred: 预测值
        
    Returns:
        分布统计信息
    """
    y_pred = np.array(y_pred)
    
    return {
        "mean": float(np.mean(y_pred)),
        "std": float(np.std(y_pred)),
        "min": float(np.min(y_pred)),
        "max": float(np.max(y_pred)),
        "median": float(np.median(y_pred)),
        "q25": float(np.percentile(y_pred, 25)),
        "q75": float(np.percentile(y_pred, 75))
    }
