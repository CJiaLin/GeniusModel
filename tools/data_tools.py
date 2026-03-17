"""
数据处理工具 - MCP 标准化工具
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from tools import register_tool, get_registry


@register_tool(
    name="load_data",
    description="加载数据文件（支持 CSV、Excel 等格式）",
    category="data",
    parameters={
        "filepath": "str - 文件路径",
        "file_type": "str - 文件类型 (csv, excel, auto)",
        "encoding": "str - 文件编码 (默认 utf-8)"
    },
    returns="pd.DataFrame - 加载的数据"
)
def load_data(filepath: str, file_type: str = "auto", encoding: str = "utf-8") -> pd.DataFrame:
    """
    加载数据文件
    
    Args:
        filepath: 文件路径
        file_type: 文件类型 (csv, excel, auto)
        encoding: 文件编码
        
    Returns:
        加载的 DataFrame
    """
    if file_type == "auto":
        if filepath.endswith('.csv'):
            file_type = "csv"
        elif filepath.endswith(('.xlsx', '.xls')):
            file_type = "excel"
        else:
            raise ValueError(f"无法自动检测文件类型：{filepath}")
    
    if file_type == "csv":
        df = pd.read_csv(filepath, encoding=encoding)
    elif file_type == "excel":
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"不支持的文件类型：{file_type}")
    
    return df


@register_tool(
    name="save_data",
    description="保存数据到文件",
    category="data",
    parameters={
        "df": "pd.DataFrame - 要保存的数据",
        "filepath": "str - 保存路径",
        "file_type": "str - 文件类型 (csv, excel)",
        "index": "bool - 是否保存索引"
    },
    returns="str - 保存的文件路径"
)
def save_data(df: pd.DataFrame, filepath: str, file_type: str = "csv", index: bool = False) -> str:
    """
    保存数据到文件
    
    Args:
        df: 要保存的 DataFrame
        filepath: 保存路径
        file_type: 文件类型
        index: 是否保存索引
        
    Returns:
        保存的文件路径
    """
    if file_type == "csv":
        df.to_csv(filepath, index=index)
    elif file_type == "excel":
        df.to_excel(filepath, index=index)
    else:
        raise ValueError(f"不支持的文件类型：{file_type}")
    
    return filepath


@register_tool(
    name="get_data_info",
    description="获取数据基本信息（形状、列名、数据类型等）",
    category="data",
    parameters={
        "df": "pd.DataFrame - 数据对象"
    },
    returns="dict - 数据信息字典"
)
def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    获取数据基本信息
    
    Args:
        df: DataFrame
        
    Returns:
        包含形状、列名、数据类型等信息的字典
    """
    info = {
        "shape": df.shape,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "has_nulls": df.isnull().any().any(),
        "null_counts": df.isnull().sum().to_dict()
    }
    return info


@register_tool(
    name="clean_data",
    description="清洗数据（处理缺失值、重复值等）",
    category="data",
    parameters={
        "df": "pd.DataFrame - 要清洗的数据",
        "handle_nulls": "str - 处理缺失值策略 (drop, fill_mean, fill_median, fill_zero)",
        "drop_duplicates": "bool - 是否删除重复值",
        "columns": "list - 指定处理的列，None 表示所有列"
    },
    returns="pd.DataFrame - 清洗后的数据"
)
def clean_data(
    df: pd.DataFrame,
    handle_nulls: str = "drop",
    drop_duplicates: bool = True,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    清洗数据
    
    Args:
        df: 要清洗的 DataFrame
        handle_nulls: 处理缺失值策略
        drop_duplicates: 是否删除重复值
        columns: 指定处理的列
        
    Returns:
        清洗后的 DataFrame
    """
    df_result = df.copy()
    
    # 处理缺失值
    if columns is None:
        columns = df_result.columns.tolist()
    
    for col in columns:
        if col not in df_result.columns:
            continue
            
        if handle_nulls == "drop":
            df_result = df_result.dropna(subset=[col])
        elif handle_nulls == "fill_mean":
            if pd.api.types.is_numeric_dtype(df_result[col]):
                df_result[col] = df_result[col].fillna(df_result[col].mean())
        elif handle_nulls == "fill_median":
            if pd.api.types.is_numeric_dtype(df_result[col]):
                df_result[col] = df_result[col].fillna(df_result[col].median())
        elif handle_nulls == "fill_zero":
            df_result[col] = df_result[col].fillna(0)
    
    # 删除重复值
    if drop_duplicates:
        df_result = df_result.drop_duplicates()
    
    return df_result


@register_tool(
    name="split_data",
    description="分割数据集为训练集和测试集",
    category="data",
    parameters={
        "df": "pd.DataFrame - 要分割的数据",
        "test_size": "float - 测试集比例 (0-1)",
        "target_column": "str - 目标列名",
        "random_state": "int - 随机种子",
        "shuffle": "bool - 是否打乱数据"
    },
    returns="tuple - (X_train, X_test, y_train, y_test)"
)
def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    target_column: str = "target",
    random_state: int = 42,
    shuffle: bool = True
) -> tuple:
    """
    分割数据集
    
    Args:
        df: DataFrame
        test_size: 测试集比例
        target_column: 目标列名
        random_state: 随机种子
        shuffle: 是否打乱
        
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # 分离特征和目标
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )
    
    return X_train, X_test, y_train, y_test


@register_tool(
    name="select_columns",
    description="选择或排除指定的列",
    category="data",
    parameters={
        "df": "pd.DataFrame - 数据对象",
        "columns": "list - 要选择的列名列表",
        "exclude": "bool - 是否为排除模式"
    },
    returns="pd.DataFrame - 处理后的数据"
)
def select_columns(
    df: pd.DataFrame,
    columns: List[str],
    exclude: bool = False
) -> pd.DataFrame:
    """
    选择或排除列
    
    Args:
        df: DataFrame
        columns: 列名列表
        exclude: 是否为排除模式
        
    Returns:
        处理后的 DataFrame
    """
    if exclude:
        selected_cols = [col for col in df.columns if col not in columns]
    else:
        selected_cols = columns
    
    return df[selected_cols]


@register_tool(
    name="filter_rows",
    description="根据条件过滤行",
    category="data",
    parameters={
        "df": "pd.DataFrame - 数据对象",
        "condition": "str - 过滤条件（如 'age > 18'）"
    },
    returns="pd.DataFrame - 过滤后的数据"
)
def filter_rows(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """
    根据条件过滤行
    
    Args:
        df: DataFrame
        condition: 过滤条件字符串
        
    Returns:
        过滤后的 DataFrame
    """
    return df.query(condition).reset_index(drop=True)


@register_tool(
    name="rename_columns",
    description="重命名列名",
    category="data",
    parameters={
        "df": "pd.DataFrame - 数据对象",
        "mapping": "dict - 列名映射字典 {旧名称：新名称}"
    },
    returns="pd.DataFrame - 重命名后的数据"
)
def rename_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    重命名列
    
    Args:
        df: DataFrame
        mapping: 列名映射字典
        
    Returns:
        重命名后的 DataFrame
    """
    return df.rename(columns=mapping)


@register_tool(
    name="get_column_stats",
    description="获取列的统计信息",
    category="data",
    parameters={
        "df": "pd.DataFrame - 数据对象",
        "columns": "list - 要统计的列，None 表示所有列"
    },
    returns="dict - 统计信息字典"
)
def get_column_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    获取列统计信息
    
    Args:
        df: DataFrame
        columns: 要统计的列
        
    Returns:
        统计信息字典
    """
    if columns is None:
        columns = df.columns.tolist()
    
    stats = {}
    for col in columns:
        if col not in df.columns:
            continue
            
        col_data = df[col]
        stats[col] = {
            "count": int(col_data.count()),
            "mean": float(col_data.mean()) if pd.api.types.is_numeric_dtype(col_data) else None,
            "std": float(col_data.std()) if pd.api.types.is_numeric_dtype(col_data) else None,
            "min": str(col_data.min()),
            "max": str(col_data.max()),
            "unique": int(col_data.nunique()),
            "null_count": int(col_data.isnull().sum())
        }
    
    return stats
