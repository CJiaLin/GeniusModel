"""
特征工程工具 - MCP 标准化工具
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from tools import register_tool


@register_tool(
    name="create_feature",
    description="创建新特征（基于表达式）",
    category="feature",
    parameters={
        "df": "pd.DataFrame - 数据对象",
        "feature_name": "str - 新特征名称",
        "expression": "str - 特征表达式（如 'col1 + col2'）"
    },
    returns="pd.DataFrame - 添加新特征后的数据"
)
def create_feature(df: pd.DataFrame, feature_name: str, expression: str) -> pd.DataFrame:
    """
    基于表达式创建新特征
    
    Args:
        df: DataFrame
        feature_name: 新特征名称
        expression: 表达式（可以使用列名）
        
    Returns:
        添加新特征后的 DataFrame
    """
    df_result = df.copy()
    df_result[feature_name] = df_result.eval(expression)
    return df_result


@register_tool(
    name="encode_categorical",
    description="编码分类变量",
    category="feature",
    parameters={
        "df": "pd.DataFrame - 数据对象",
        "columns": "list - 要编码的分类列",
        "method": "str - 编码方法 (onehot, label, ordinal)"
    },
    returns="pd.DataFrame - 编码后的数据"
)
def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "onehot"
) -> pd.DataFrame:
    """
    编码分类变量
    
    Args:
        df: DataFrame
        columns: 分类列
        method: 编码方法
        
    Returns:
        编码后的 DataFrame
    """
    df_result = df.copy()
    
    if method == "onehot":
        # One-Hot 编码
        df_result = pd.get_dummies(df_result, columns=columns, drop_first=False)
    elif method == "label":
        # Label 编码
        from sklearn.preprocessing import LabelEncoder
        for col in columns:
            le = LabelEncoder()
            df_result[col] = le.fit_transform(df_result[col].astype(str))
    elif method == "ordinal":
        # Ordinal 编码（按频率）
        for col in columns:
            value_counts = df_result[col].value_counts()
            mapping = {val: idx for idx, val in enumerate(value_counts.index)}
            df_result[col] = df_result[col].map(mapping)
    else:
        raise ValueError(f"不支持的编码方法：{method}")
    
    return df_result


@register_tool(
    name="scale_features",
    description="特征缩放（标准化、归一化）",
    category="feature",
    parameters={
        "df": "pd.DataFrame - 数据对象",
        "columns": "list - 要缩放的列",
        "method": "str - 缩放方法 (standard, minmax, robust)"
    },
    returns="pd.DataFrame - 缩放后的数据"
)
def scale_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard"
) -> pd.DataFrame:
    """
    特征缩放
    
    Args:
        df: DataFrame
        columns: 要缩放的列
        method: 缩放方法
        
    Returns:
        缩放后的 DataFrame
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    df_result = df.copy()
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"不支持的缩放方法：{method}")
    
    df_result[columns] = scaler.fit_transform(df_result[columns])
    return df_result


@register_tool(
    name="create_interaction_features",
    description="创建交互特征（列之间的乘积）",
    category="feature",
    parameters={
        "df": "pd.DataFrame - 数据对象",
        "columns": "list - 要创建交互特征的列",
        "operation": "str - 操作类型 (multiply, divide, add, subtract)"
    },
    returns="pd.DataFrame - 添加交互特征后的数据"
)
def create_interaction_features(
    df: pd.DataFrame,
    columns: List[str],
    operation: str = "multiply"
) -> pd.DataFrame:
    """
    创建交互特征
    
    Args:
        df: DataFrame
        columns: 列列表
        operation: 操作类型
        
    Returns:
        添加交互特征后的 DataFrame
    """
    df_result = df.copy()
    
    if len(columns) < 2:
        return df_result
    
    if operation == "multiply":
        feature_name = "_mul_".join(columns)
        df_result[feature_name] = df_result[columns[0]]
        for col in columns[1:]:
            df_result[feature_name] *= df_result[col]
    elif operation == "add":
        feature_name = "_add_".join(columns)
        df_result[feature_name] = df_result[columns[0]]
        for col in columns[1:]:
            df_result[feature_name] += df_result[col]
    elif operation == "subtract":
        feature_name = f"{columns[0]}_sub_{columns[1]}"
        df_result[feature_name] = df_result[columns[0]] - df_result[columns[1]]
    elif operation == "divide":
        feature_name = f"{columns[0]}_div_{columns[1]}"
        df_result[feature_name] = df_result[columns[0]] / (df_result[columns[1]] + 1e-8)
    else:
        raise ValueError(f"不支持的操作：{operation}")
    
    return df_result


@register_tool(
    name="create_polynomial_features",
    description="创建多项式特征",
    category="feature",
    parameters={
        "df": "pd.DataFrame - 数据对象",
        "columns": "list - 要创建多项式特征的列",
        "degree": "int - 多项式次数"
    },
    returns="pd.DataFrame - 添加多项式特征后的数据"
)
def create_polynomial_features(
    df: pd.DataFrame,
    columns: List[str],
    degree: int = 2
) -> pd.DataFrame:
    """
    创建多项式特征
    
    Args:
        df: DataFrame
        columns: 列列表
        degree: 多项式次数
        
    Returns:
        添加多项式特征后的 DataFrame
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    df_result = df.copy()
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df_result[columns])
    
    # 创建新列名
    feature_names = poly.get_feature_names_out(columns)
    
    # 只添加原始列没有的新特征
    new_features_mask = ~np.isin(feature_names, columns)
    new_features = poly_features[:, new_features_mask]
    new_names = [f"poly_{name.replace(' ', '_')}" for name in feature_names[new_features_mask]]
    
    for i, name in enumerate(new_names):
        df_result[name] = new_features[:, i]
    
    return df_result


@register_tool(
    name="binning",
    description="特征分箱（离散化）",
    category="feature",
    parameters={
        "df": "pd.DataFrame - 数据对象",
        "column": "str - 要分箱的列",
        "n_bins": "int - 分箱数量",
        "method": "str - 分箱方法 (quantile, uniform, kmeans)"
    },
    returns="pd.DataFrame - 添加分箱特征后的数据"
)
def binning(
    df: pd.DataFrame,
    column: str,
    n_bins: int = 5,
    method: str = "quantile"
) -> pd.DataFrame:
    """
    特征分箱
    
    Args:
        df: DataFrame
        column: 要分箱的列
        n_bins: 分箱数量
        method: 分箱方法
        
    Returns:
        添加分箱特征后的 DataFrame
    """
    df_result = df.copy()
    
    if method == "quantile":
        df_result[f"{column}_bin"] = pd.qcut(df_result[column], q=n_bins, duplicates='drop')
    elif method == "uniform":
        df_result[f"{column}_bin"] = pd.cut(df_result[column], bins=n_bins)
    elif method == "kmeans":
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_bins, random_state=42)
        df_result[f"{column}_bin"] = kmeans.fit_predict(df_result[[column]])
    else:
        raise ValueError(f"不支持的分箱方法：{method}")
    
    return df_result


@register_tool(
    name="select_features",
    description="特征选择（基于重要性、方差等）",
    category="feature",
    parameters={
        "df": "pd.DataFrame - 数据对象",
        "target_column": "str - 目标列",
        "n_features": "int - 选择的特征数量",
        "method": "str - 选择方法 (variance, correlation, mutual_info)"
    },
    returns="list - 选中的特征列名列表"
)
def select_features(
    df: pd.DataFrame,
    target_column: str,
    n_features: int = 10,
    method: str = "variance"
) -> List[str]:
    """
    特征选择
    
    Args:
        df: DataFrame
        target_column: 目标列
        n_features: 选择数量
        method: 选择方法
        
    Returns:
        选中的特征列名列表
    """
    from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    # 分离特征和目标
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 只处理数值列
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    
    if method == "variance":
        # 基于方差选择
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X_numeric)
        selected_mask = selector.get_support()
        selected_features = [numeric_cols[i] for i in range(len(numeric_cols)) if selected_mask[i]]
        
    elif method == "mutual_info":
        # 基于互信息选择
        if y.dtype == 'object' or len(y.unique()) < 10:
            scores = mutual_info_classif(X_numeric, y, random_state=42)
        else:
            scores = mutual_info_regression(X_numeric, y, random_state=42)
        
        # 选择 top N
        top_indices = np.argsort(scores)[-n_features:][::-1]
        selected_features = [numeric_cols[i] for i in top_indices]
        
    elif method == "importance":
        # 基于随机森林重要性
        if y.dtype == 'object' or len(y.unique()) < 10:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        rf.fit(X_numeric, y)
        importances = rf.feature_importances_
        
        # 选择 top N
        top_indices = np.argsort(importances)[-n_features:][::-1]
        selected_features = [numeric_cols[i] for i in top_indices]
    else:
        raise ValueError(f"不支持的特征选择方法：{method}")
    
    return selected_features[:n_features]


@register_tool(
    name="drop_features",
    description="删除指定的特征",
    category="feature",
    parameters={
        "df": "pd.DataFrame - 数据对象",
        "columns": "list - 要删除的列名"
    },
    returns="pd.DataFrame - 删除特征后的数据"
)
def drop_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    删除特征
    
    Args:
        df: DataFrame
        columns: 要删除的列
        
    Returns:
        删除特征后的 DataFrame
    """
    return df.drop(columns=columns, errors='ignore')


@register_tool(
    name="get_feature_correlation",
    description="获取特征相关性矩阵",
    category="feature",
    parameters={
        "df": "pd.DataFrame - 数据对象",
        "method": "str - 相关性计算方法 (pearson, spearman, kendall)"
    },
    returns="pd.DataFrame - 相关性矩阵"
)
def get_feature_correlation(
    df: pd.DataFrame,
    method: str = "pearson"
) -> pd.DataFrame:
    """
    获取特征相关性
    
    Args:
        df: DataFrame
        method: 相关性计算方法
        
    Returns:
        相关性矩阵
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr(method=method)
