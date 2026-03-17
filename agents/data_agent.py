"""
数据处理Agent模块

本模块是AutoML系统中的数据处理组件，负责数据加载、探索、质量分析和清洗。
它是建模流程的第二阶段，在总控Agent确定建模计划后执行数据相关操作。

主要组件：
1. DataLoader - 数据加载器，支持多种文件格式
2. DataExplorer - 数据探索器，分析数据基本特征
3. DataQualityAnalyzer - 数据质量分析器
4. DataCleaner - 数据清洗器
5. DataAgent - 数据处理Agent主类

工作流程：
1. 加载数据（支持CSV、Excel、JSON等格式）
2. 探索数据（行列数、类型、分布等）
3. 分析数据质量（缺失值、重复值、异常值等）
4. 清洗数据（处理缺失值、异常值、重复值等）
"""

import pandas as pd
import numpy as np
from typing import Any, Optional
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import json
import re

# 修复导入路径
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from automl_agent.models import DataProfile, DataQualityReport, ProcessState
from automl_agent.core.executor import CodeExecutor


class DataLoader:
    """
    数据加载器
    
    负责从不同来源加载数据到DataFrame中。
    支持CSV、Excel、JSON等常见数据格式。
    
    Attributes:
        executor: 代码执行器，用于安全执行数据加载代码
        
    Example:
        >>> loader = DataLoader(executor)
        >>> df = loader.load("data.csv")
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>
    """
    
    def __init__(self, executor: CodeExecutor):
        """
        初始化数据加载器
        
        Args:
            executor: 代码执行器实例
        """
        self.executor = executor

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        加载CSV文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            pd.DataFrame: 加载的数据框
            
        Raises:
            Exception: 如果加载失败
        """
        # 构建加载CSV的代码
        code = f"""
import pandas as pd
result = pd.read_csv('{file_path}')
"""
        # 执行代码
        result = self.executor.execute(code)
        
        # 检查执行是否成功
        if not result["success"]:
            raise Exception(f"加载CSV失败: {result['error']}")
        
        # 获取返回的数据
        df_data = result.get("data")
        if df_data is None:
            raise Exception("未返回数据")
        
        # 将数据转换为DataFrame
        return pd.read_json(df_data) if isinstance(df_data, str) else df_data

    def load_excel(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        加载Excel文件
        
        Args:
            file_path: Excel文件路径
            sheet_name: 工作表名称，默认第一个
            
        Returns:
            pd.DataFrame: 加载的数据框
        """
        code = f"""
import pandas as pd
result = pd.read_excel('{file_path}', sheet_name={repr(sheet_name)})
"""
        result = self.executor.execute(code)
        if not result["success"]:
            raise Exception(f"加载Excel失败: {result['error']}")
        
        df_data = result.get("data")
        return pd.read_json(df_data) if isinstance(df_data, str) else df_data

    def load_json(self, file_path: str) -> pd.DataFrame:
        """
        加载JSON文件
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            pd.DataFrame: 加载的数据框
        """
        code = f"""
import pandas as pd
result = pd.read_json('{file_path}')
"""
        result = self.executor.execute(code)
        if not result["success"]:
            raise Exception(f"加载JSON失败: {result['error']}")
        
        df_data = result.get("data")
        return pd.read_json(df_data) if isinstance(df_data, str) else df_data

    def detect_file_type(self, file_path: str) -> str:
        """
        检测文件类型
        
        根据文件扩展名判断数据类型。
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件类型（csv/excel/json/unknown）
        """
        # 获取文件扩展名（小写）
        suffix = Path(file_path).suffix.lower()
        
        # 文件类型映射表
        type_map = {
            ".csv": "csv",
            ".xlsx": "excel",
            ".xls": "excel",
            ".json": "json"
        }
        return type_map.get(suffix, "unknown")

    def load(self, file_path: str) -> pd.DataFrame:
        """
        自动加载数据
        
        根据文件类型自动选择合适的加载方法。
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            pd.DataFrame: 加载的数据框
            
        Raises:
            ValueError: 如果文件类型不支持
        """
        # 检测文件类型
        file_type = self.detect_file_type(file_path)
        
        # 根据类型选择加载方法
        if file_type == "csv":
            return self.load_csv(file_path)
        elif file_type == "excel":
            return self.load_excel(file_path)
        elif file_type == "json":
            return self.load_json(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")


class DataExplorer:
    """
    数据探索器
    
    负责对数据进行初步探索，分析数据的基本特征。
    包括数据形状、列信息、数据类型等。
    
    Attributes:
        executor: 代码执行器
    """
    
    def __init__(self, executor: CodeExecutor):
        """
        初始化数据探索器
        
        Args:
            executor: 代码执行器实例
        """
        self.executor = executor

    def explore(self, df: pd.DataFrame) -> DataProfile:
        """
        探索数据
        
        分析数据的基本信息，返回数据画像。
        
        Args:
            df: 要探索的数据框
            
        Returns:
            DataProfile: 数据画像对象
            
        Example:
            >>> profile = explorer.explore(df)
            >>> print(profile.shape)
            (1000, 20)
        """
        # 构建数据探索代码
        code = f"""
import pandas as pd
import json

df = pd.read_json('{df.to_json()}')

shape = df.shape
columns = df.columns.tolist()
dtypes = df.dtypes.apply(str).to_dict()

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

missing = df.isnull().sum().to_dict()

result = {{
    'shape': shape,
    'columns': columns,
    'dtypes': dtypes,
    'numeric_columns': numeric_cols,
    'categorical_columns': categorical_cols,
    'missing_values': missing
}}
"""
        result = self.executor.execute(code)
        
        # 如果执行成功，解析结果
        if result["success"] and result.get("data"):
            data = result["data"]
            return DataProfile(
                shape=tuple(data.get("shape", [0, 0])),
                columns=data.get("columns", []),
                dtypes=data.get("dtypes", {}),
                missing_values=data.get("missing_values", {}),
                numeric_columns=data.get("numeric_columns", []),
                categorical_columns=data.get("categorical_columns", []),
                target_column=None
            )
        
        # 如果执行失败，使用本地方式
        return DataProfile(
            shape=df.shape,
            columns=df.columns.tolist(),
            dtypes=df.dtypes.apply(str).to_dict(),
            missing_values=df.isnull().sum().to_dict(),
            numeric_columns=df.select_dtypes(include=["number"]).columns.tolist(),
            categorical_columns=df.select_dtypes(include=["object", "category"]).columns.tolist(),
            target_column=None
        )


class DataQualityAnalyzer:
    """
    数据质量分析器
    
    负责分析数据质量，包括：
    - 缺失值分析
    - 重复值分析
    - 异常值分析
    - 数据分布分析
    
    Attributes:
        llm: 大语言模型实例（可选，用于高级分析）
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化数据质量分析器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm

    def analyze_missing(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        分析缺失值
        
        计算每个列的缺失值数量和百分比。
        
        Args:
            df: 要分析的数据框
            
        Returns:
            dict: 缺失值分析结果
        """
        # 计算缺失值数量
        missing_count = df.isnull().sum()
        # 计算缺失值百分比
        missing_pct = (missing_count / len(df) * 100).round(2)
        
        return {
            "missing_count": missing_count.to_dict(),              # 缺失值数量
            "missing_percentage": missing_pct.to_dict(),           # 缺失值百分比
            "total_missing": missing_count.sum(),                   # 总缺失值数
            "columns_with_missing": missing_count[missing_count > 0].index.tolist()  # 有缺失值的列
        }

    def analyze_duplicates(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        分析重复值
        
        计算重复记录的数量和百分比。
        
        Args:
            df: 要分析的数据框
            
        Returns:
            dict: 重复值分析结果
        """
        duplicate_count = df.duplicated().sum()
        
        return {
            "duplicate_count": int(duplicate_count),                     # 重复记录数量
            "duplicate_percentage": round(duplicate_count / len(df) * 100, 2)  # 重复百分比
        }

    def analyze_outliers(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> dict[str, Any]:
        """
        分析异常值
        
        使用IQR方法检测数值列中的异常值。
        
        Args:
            df: 要分析的数据框
            columns: 要分析的列，默认所有数值列
            
        Returns:
            dict: 异常值分析结果
        """
        # 获取数值列
        numeric_cols = columns or df.select_dtypes(include=["number"]).columns.tolist()
        
        outlier_info = {}
        
        # 对每个数值列进行异常值检测
        for col in numeric_cols:
            # 计算四分位数
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # 计算上下界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 找出异常值
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            # 记录异常值信息
            outlier_info[col] = {
                "count": len(outliers),                    # 异常值数量
                "percentage": round(len(outliers) / len(df) * 100, 2),  # 异常值百分比
                "lower_bound": lower_bound,               # 下界
                "upper_bound": upper_bound                # 上界
            }
        
        return outlier_info

    def analyze_distribution(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        分析数据分布
        
        计算数值列和类别列的分布统计信息。
        
        Args:
            df: 要分析的数据框
            
        Returns:
            dict: 数据分布分析结果
        """
        distribution = {}
        
        # 遍历所有列
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                # 数值列：计算统计量
                distribution[col] = {
                    "mean": float(df[col].mean()),                   # 均值
                    "median": float(df[col].median()),               # 中位数
                    "std": float(df[col].std()),                    # 标准差
                    "min": float(df[col].min()),                    # 最小值
                    "max": float(df[col].max()),                    # 最大值
                    "q25": float(df[col].quantile(0.25)),           # 25分位数
                    "q75": float(df[col].quantile(0.75))            # 75分位数
                }
            else:
                # 类别列：计算唯一值和频次
                distribution[col] = {
                    "unique_count": int(df[col].nunique()),          # 唯一值数量
                    "top_values": df[col].value_counts().head(5).to_dict()  # 前5个高频值
                }
        
        return distribution

    def generate_report(self, df: pd.DataFrame) -> DataQualityReport:
        """
        生成完整的数据质量报告
        
        综合分析数据质量的各个方面。
        
        Args:
            df: 要分析的数据框
            
        Returns:
            DataQualityReport: 数据质量报告对象
        """
        return DataQualityReport(
            missing_analysis=self.analyze_missing(df),         # 缺失值分析
            duplicate_count=int(df.duplicated().sum()),       # 重复值
            outlier_analysis=self.analyze_outliers(df),        # 异常值分析
            distribution_analysis=self.analyze_distribution(df)  # 数据分布分析
        )


class DataCleaner:
    """
    数据清洗器
    
    负责数据清洗操作，包括：
    - 处理缺失值
    - 去除重复记录
    - 处理异常值
    
    Attributes:
        llm: 大语言模型实例（可选，用于智能清洗策略）
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化数据清洗器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm

    def clean_missing(self, df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
        """
        清洗缺失值
        
        根据数据类型和策略填充缺失值。
        
        Args:
            df: 要清洗的数据框
            strategy: 填充策略，auto表示自动选择
            
        Returns:
            pd.DataFrame: 清洗后的数据框
        """
        df_clean = df.copy()
        
        # 遍历所有列
        for col in df_clean.columns:
            # 只处理有缺失值的列
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ["int64", "float64"]:
                    # 数值列：用中位数或0填充
                    if strategy == "auto":
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    else:
                        df_clean[col].fillna(0, inplace=True)
                else:
                    # 类别列：用众数或"Unknown"填充
                    df_clean[col].fillna(
                        df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else "Unknown", 
                        inplace=True
                    )
        
        return df_clean

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        去除重复记录
        
        Args:
            df: 要处理的数据框
            
        Returns:
            pd.DataFrame: 去重后的数据框
        """
        return df.drop_duplicates()

    def handle_outliers(self, df: pd.DataFrame, method: str = "remove") -> pd.DataFrame:
        """
        处理异常值
        
        使用IQR方法识别并处理异常值。
        
        Args:
            df: 要处理的数据框
            method: 处理方式，remove表示删除，cap表示截断
            
        Returns:
            pd.DataFrame: 处理后的数据框
        """
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns
        
        for col in numeric_cols:
            # 计算四分位数
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if method == "remove":
                # 删除异常值所在行
                df_clean = df_clean[
                    (df_clean[col] >= Q1 - 1.5 * IQR) & 
                    (df_clean[col] <= Q3 + 1.5 * IQR)
                ]
            elif method == "cap":
                # 截断到边界值
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        return df_clean


class DataAgent:
    """
    数据处理Agent
    
    这是数据处理阶段的主类，整合了数据加载、探索、质量分析和清洗功能。
    在AutoML流程中负责处理原始数据，为后续的特征工程和建模做准备。
    
    Attributes:
        llm: 大语言模型实例
        executor: 代码执行器
        loader: 数据加载器
        explorer: 数据探索器
        quality_analyzer: 数据质量分析器
        cleaner: 数据清洗器
        state: 流程状态
        current_data: 当前处理的数据
        
    Example:
        >>> agent = DataAgent()
        >>> profile = agent.load_data("data.csv")
        >>> report = agent.analyze_quality()
        >>> clean_df = agent.clean_data()
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化数据处理Agent
        
        Args:
            llm: 大语言模型实例
        """
        # 初始化LLM
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        
        # 初始化各组件
        self.executor = CodeExecutor()                # 代码执行器
        self.loader = DataLoader(self.executor)       # 数据加载器
        self.explorer = DataExplorer(self.executor)  # 数据探索器
        self.quality_analyzer = DataQualityAnalyzer(self.llm)  # 质量分析器
        self.cleaner = DataCleaner(self.llm)         # 数据清洗器
        self.state = ProcessState()                   # 流程状态
        self.current_data: Optional[pd.DataFrame] = None  # 当前数据

    def load_data(self, file_path: str) -> DataProfile:
        """
        加载数据
        
        从文件加载数据并返回数据画像。
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            DataProfile: 数据画像对象
        """
        self.state.current_step = "加载数据"
        self.current_data = self.loader.load(file_path)
        return self.explorer.explore(self.current_data)

    def analyze_quality(self) -> DataQualityReport:
        """
        分析数据质量
        
        对当前加载的数据进行全面的质量分析。
        
        Returns:
            DataQualityReport: 数据质量报告
            
        Raises:
            ValueError: 如果未先加载数据
        """
        self.state.current_step = "分析数据质量"
        
        if self.current_data is None:
            raise ValueError("请先加载数据")
        
        return self.quality_analyzer.generate_report(self.current_data)

    def clean_data(self, strategy: str = "auto") -> pd.DataFrame:
        """
        清洗数据
        
        对数据进行清洗，包括去重、处理缺失值、处理日期列等。
        
        Args:
            strategy: 缺失值填充策略
            
        Returns:
            pd.DataFrame: 清洗后的数据
            
        Raises:
            ValueError: 如果未先加载数据
        """
        self.state.current_step = "清洗数据"
        
        if self.current_data is None:
            raise ValueError("请先加载数据")
        
        # 执行清洗操作
        df_clean = self.current_data.copy()
        df_clean = self.cleaner.remove_duplicates(df_clean)  # 去重
        df_clean = self.cleaner.clean_missing(df_clean, strategy)  # 处理缺失值
        
        # 处理日期列：删除日期列或转换为数值特征
        date_cols = df_clean.select_dtypes(include=['datetime64', 'object']).columns
        for col in date_cols:
            # 尝试检测是否为日期列
            if '日期' in col or 'date' in col.lower() or 'time' in col.lower():
                # 尝试将日期转换为数值特征（如距离基准日期的天数）
                try:
                    date_series = pd.to_datetime(df_clean[col], errors='coerce')
                    if date_series.notna().sum() > 0:
                        # 转换为距离第一天的小时数
                        base_date = date_series.min()
                        df_clean[col + '_days'] = (date_series - base_date).dt.days
                        df_clean = df_clean.drop(columns=[col])
                        print(f"  转换日期列 '{col}' 为数值特征")
                except Exception as e:
                    # 如果转换失败，直接删除该列
                    df_clean = df_clean.drop(columns=[col])
                    print(f"  删除日期列 '{col}'")
        
        # 移除无法用于建模的列（如包含唯一ID的列）
        for col in df_clean.columns:
            if df_clean[col].nunique() == len(df_clean):
                df_clean = df_clean.drop(columns=[col])
                print(f"  删除唯一值列 '{col}'")
        
        # 更新当前数据
        self.current_data = df_clean
        return df_clean

    def get_data(self) -> pd.DataFrame:
        """
        获取当前数据
        
        Returns:
            pd.DataFrame: 当前处理的数据
            
        Raises:
            ValueError: 如果未先加载数据
        """
        if self.current_data is None:
            raise ValueError("请先加载数据")
        return self.current_data
