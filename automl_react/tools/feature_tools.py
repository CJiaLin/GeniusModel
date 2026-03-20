"""
特征工程工具模块

提供特征生成的工具
"""

import pandas as pd
import numpy as np
from typing import Any, Dict

from .base_tool import BaseTool, ToolResult


class FeatureGeneratorTool(BaseTool):
    """
    特征生成工具
    
    自动生成数据特征
    """
    
    name = "generate_features"
    description = "自动生成数据特征，包括交互特征、统计特征等"
    parameters = {
        "file_path": {
            "type": "string",
            "description": "数据文件路径"
        },
        "target_column": {
            "type": "string",
            "description": "目标列名"
        },
        "task_type": {
            "type": "string",
            "description": "任务类型 (classification/regression)"
        }
    }
    
    def execute(self, file_path: str, target_column: str, task_type: str) -> ToolResult:
        """执行特征生成"""
        try:
            # 加载数据
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                return ToolResult.error("目前只支持 CSV 格式")
            
            if target_column not in df.columns:
                return ToolResult.error(f"目标列 '{target_column}' 不存在")
            
            original_cols = df.columns.tolist()
            
            # 获取数值列（排除目标列）
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != target_column]
            
            # 生成统计特征
            if len(numeric_cols) > 1:
                df["numeric_mean"] = df[numeric_cols].mean(axis=1)
                df["numeric_std"] = df[numeric_cols].std(axis=1)
                df["numeric_sum"] = df[numeric_cols].sum(axis=1)
            
            # 生成交互特征（前3个数值列）
            if len(numeric_cols) >= 2:
                for i in range(min(3, len(numeric_cols))):
                    for j in range(i+1, min(3, len(numeric_cols))):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
            
            new_cols = [c for c in df.columns if c not in original_cols]
            
            # 保存特征数据
            output_path = file_path.replace(".csv", "_features.csv")
            df.to_csv(output_path, index=False)
            
            return ToolResult.success(
                data={
                    "original_features": len(original_cols) - 1,
                    "new_features": len(new_cols),
                    "total_features": len(df.columns) - 1,
                    "new_columns": new_cols,
                    "output_path": output_path
                }
            )
            
        except Exception as e:
            return ToolResult.error(f"特征生成失败: {str(e)}")
