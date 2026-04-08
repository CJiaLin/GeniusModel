"""
数据画像工具模块

提供 DataProfileTool，对数据文件进行全面的数据质量分析
"""

import os
from typing import Any, Dict, List

import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult


class DataProfileTool(BaseTool):
    """数据质量画像工具"""

    name = "profile_data"
    description = (
        "对数据文件进行全面的数据质量分析，包括：缺失值分析、"
        "异常值检测（IQR方法）、重复值分析、数据类型和分布分析。"
        "返回结构化的质量报告。"
    )
    parameters = {
        "file_path": {
            "type": "string",
            "description": "数据文件路径（CSV格式）"
        }
    }

    def execute(self, file_path: str = "", **kwargs) -> ToolResult:
        """执行数据质量分析"""
        if not file_path:
            return ToolResult.error("必须指定 file_path 参数")

        if not os.path.isfile(file_path):
            return ToolResult.error(f"文件不存在: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return ToolResult.error(f"读取文件失败: {e}")

        try:
            profile = self._build_profile(df)
            report_text = self._format_report(profile, file_path)
            profile["report"] = report_text
            return ToolResult.success(data=profile)
        except Exception as e:
            return ToolResult.error(f"数据分析失败: {e}")

    def _build_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """构建数据画像"""
        profile: Dict[str, Any] = {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        }

        # 缺失值分析
        missing = {}
        for col in df.columns:
            miss_count = int(df[col].isnull().sum())
            if miss_count > 0:
                missing[col] = {
                    "count": miss_count,
                    "ratio": round(miss_count / len(df), 4),
                    "dtype": str(df[col].dtype),
                }
        profile["missing"] = missing

        # 异常值分析 (IQR)
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())
            if outlier_count > 0:
                outliers[col] = {
                    "count": outlier_count,
                    "ratio": round(outlier_count / len(series), 4),
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                }
        profile["outliers"] = outliers

        # 重复值分析
        dup_count = int(df.duplicated().sum())
        profile["duplicates"] = {
            "count": dup_count,
            "ratio": round(dup_count / len(df), 4) if len(df) > 0 else 0,
        }

        # 数据类型分析
        dtype_info = {}
        for col in df.columns:
            unique_count = int(df[col].nunique())
            samples = df[col].dropna().head(3).tolist()
            dtype_info[col] = {
                "dtype": str(df[col].dtype),
                "unique_count": unique_count,
                "sample_values": [str(s) for s in samples],
            }
        profile["dtypes"] = dtype_info

        return profile

    def _format_report(self, profile: Dict[str, Any], file_path: str) -> str:
        """格式化为可读报告"""
        lines = [
            f"## 数据质量报告",
            f"- 文件: {file_path}",
            f"- 行数: {profile['shape']['rows']}, 列数: {profile['shape']['columns']}",
            "",
        ]

        # 缺失值
        missing = profile.get("missing", {})
        if missing:
            lines.append(f"### 缺失值 ({len(missing)} 列有缺失)")
            for col, info in sorted(missing.items(), key=lambda x: -x[1]["ratio"]):
                lines.append(f"- {col}: {info['count']} ({info['ratio']:.1%})")
        else:
            lines.append("### 缺失值: 无")
        lines.append("")

        # 异常值
        outliers = profile.get("outliers", {})
        if outliers:
            lines.append(f"### 异常值 ({len(outliers)} 列有异常)")
            for col, info in sorted(outliers.items(), key=lambda x: -x[1]["ratio"])[:10]:
                lines.append(f"- {col}: {info['count']} ({info['ratio']:.1%})")
        else:
            lines.append("### 异常值: 无")
        lines.append("")

        # 重复值
        dup = profile["duplicates"]
        lines.append(f"### 重复值: {dup['count']} 行 ({dup['ratio']:.1%})")

        return "\n".join(lines)
