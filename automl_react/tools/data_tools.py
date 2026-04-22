"""
数据工具模块

提供数据加载和分析的工具
"""

import pandas as pd
from typing import Any, Dict, Optional, Literal
from pathlib import Path

from pydantic import BaseModel, Field

from .base_tool import BaseTool, ToolResult


class DataLoaderInput(BaseModel):
    file_path: str = Field(..., description="数据文件路径")
    file_type: Optional[Literal["csv", "excel", "json"]] = Field(
        None, description="文件类型 (csv/excel/json)，可选，自动检测"
    )


class DataLoaderTool(BaseTool):
    """
    数据加载工具

    支持加载 CSV、Excel、JSON 等格式的数据文件
    """

    name = "load_data"
    description = "加载数据文件，支持 CSV、Excel、JSON 格式"
    input_model = DataLoaderInput
    
    def execute(self, file_path: str, file_type: str = None) -> ToolResult:
        """执行数据加载"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return ToolResult.error(f"文件不存在: {file_path}")
            
            # 自动检测文件类型
            if file_type is None:
                suffix = path.suffix.lower()
                if suffix == ".csv":
                    file_type = "csv"
                elif suffix in [".xlsx", ".xls"]:
                    file_type = "excel"
                elif suffix == ".json":
                    file_type = "json"
                else:
                    return ToolResult.error(f"不支持的文件类型: {suffix}")
            
            # 加载数据
            if file_type == "csv":
                df = pd.read_csv(file_path)
            elif file_type == "excel":
                df = pd.read_excel(file_path)
            elif file_type == "json":
                df = pd.read_json(file_path)
            else:
                return ToolResult.error(f"不支持的文件类型: {file_type}")
            
            # 返回数据信息
            data_info = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isnull().sum().to_dict(),
                "preview": df.head(5).to_dict()
            }
            
            return ToolResult.success(
                data=data_info,
                metadata={"file_path": file_path, "file_type": file_type}
            )
            
        except Exception as e:
            return ToolResult.error(f"加载数据失败: {str(e)}")


class DataAnalyzerInput(BaseModel):
    file_path: str = Field(..., description="数据文件路径")


class DataAnalyzerTool(BaseTool):
    """
    数据分析工具

    分析数据的基本统计信息和质量
    """

    name = "analyze_data"
    description = "分析数据的基本统计信息和质量"
    input_model = DataAnalyzerInput
    
    def execute(self, file_path: str) -> ToolResult:
        """执行数据分析"""
        try:
            # 加载数据
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path)
            else:
                return ToolResult.error(f"不支持的文件格式")
            
            # 基础统计
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            
            analysis = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "missing_values": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
                "duplicate_count": int(df.duplicated().sum()),
            }
            
            # 数值列统计
            if numeric_cols:
                analysis["numeric_stats"] = df[numeric_cols].describe().to_dict()
            
            # 类别列统计
            if categorical_cols:
                analysis["categorical_stats"] = {
                    col: {
                        "unique_count": int(df[col].nunique()),
                        "top_values": df[col].value_counts().head(5).to_dict()
                    }
                    for col in categorical_cols[:5]  # 只统计前5个类别列
                }
            
            return ToolResult.success(data=analysis)
            
        except Exception as e:
            return ToolResult.error(f"数据分析失败: {str(e)}")
