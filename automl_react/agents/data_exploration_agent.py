"""
数据探索性分析 Agent 模块

基于清洗后的数据进行探索性分析，分析数据分布、相关性等统计特征
"""

import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..core.react_agent import ReActAgent
from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
from ..skills_loader import get_skill_loader
from ..config import get_config_loader
from ..logger.llm_logger import LLMLogger


class DataExplorationAgent(ReActAgent):
    """
    数据探索性分析 Agent

    基于 ReAct 架构的数据探索性分析 Agent，支持：
    1. 分析清洗后数据的统计特征
    2. 分析特征相关性
    3. 分析目标变量分布
    4. 为特征工程提供建议

    Attributes:
        session_id: 会话ID
        data_path: 数据文件路径（清洗后的数据）
        exploration_result: 探索性分析结果
    """

    def __init__(self, llm: Any = None, session_id: str = None, verbose: bool = False):
        super().__init__(llm=llm, session_id=session_id, max_iterations=10, verbose=verbose)
        self.data_path: Optional[str] = None
        self.target_column: Optional[str] = None
        self.task_type: Optional[str] = None
        self.data_info: Optional[Dict] = None
        self.exploration_result: Optional[str] = None
        self.skill_loader = get_skill_loader()
        self.config_loader = get_config_loader()
        self.logger = LLMLogger(session_id=session_id)

    def _register_default_tools(self):
        """注册默认工具"""
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.config_loader.get_prompt("data_exploration", "system_prompt")

    def _get_exploration_skill_content(self) -> str:
        """获取探索性分析阶段使用的 skill 参考内容。"""
        techniques = self.skill_loader.get_skill_reference("data-analysis-1.0.2", "techniques.md")
        if not techniques:
            return ""
        return f"## 数据分析技术参考\n\n{techniques}\n"

    def explore(
        self,
        data_path: str,
        target_column: str = None,
        task_type: str = "classification",
        task_description: str = ""
    ) -> Dict[str, Any]:
        """
        对清洗后的数据进行探索性分析

        Args:
            data_path: 数据文件路径（清洗后的数据）
            target_column: 目标列名
            task_type: 任务类型
            task_description: 用户的建模背景和要求

        Returns:
            探索性分析结果
        """
        self.data_path = data_path
        self.target_column = target_column
        self.task_type = task_type

        # 加载并分析清洗后的数据
        try:
            df = pd.read_csv(data_path)

            # 收集数据统计信息
            self.data_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "numeric_columns": list(df.select_dtypes(include=['int64', 'float64']).columns),
                "categorical_columns": list(df.select_dtypes(include=['object']).columns),
                "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024
            }

            # 构建当前数据上下文
            current_data_context = self._build_data_summary(df)

        except Exception as e:
            current_data_context = f"无法加载数据文件: {data_path}\n错误: {str(e)}"

        # 加载 data-analysis skill 内容
        skill_content = self._get_exploration_skill_content()

        # 构建用户提示词
        task_context = ""
        if task_description:
            task_context = f"""
## 用户建模背景和要求

{task_description}

**重要：请在探索性分析中充分考虑用户的建模背景和要求。**

"""
        self.task_context = task_context

        cleaning_context = f"""
    ## 前序清洗说明

    数据清洗阶段已完成，当前传入的数据文件就是清洗结果数据。

    **重要：本阶段唯一有效的数据来源是当前传入的数据文件路径 `{data_path}`。如果任何历史文本与当前文件实际读取结果冲突，必须以当前文件实际读取结果为准。**

    """

        prompt_template = self.config_loader.get_prompt("data_exploration", "exploration_prompt")
        user_input = prompt_template.format(
            data_path=data_path,
            target_column=target_column or "",
            task_type=task_type,
            task_context=task_context,
            cleaning_context=cleaning_context,
            current_data_context=current_data_context,
            skill_content=skill_content,
        )

        # 调用 LLM 生成报告
        result = self.run(user_input, stage="data_exploration")

        self.exploration_result = result.get("answer", "")

        return {
            "success": True,
            "answer": self.exploration_result,
            "data_info": self.data_info
        }

    def _build_data_summary(self, df: pd.DataFrame) -> str:
        """
        构建数据摘要

        Args:
            df: 数据 DataFrame

        Returns:
            数据摘要字符串
        """
        summary_lines = ["## 清洗后数据基本信息\n"]
        summary_lines.append(f"- 数据形状: {df.shape[0]} 行 × {df.shape[1]} 列\n")
        summary_lines.append(f"- 数值列数量: {len(self.data_info['numeric_columns'])}\n")
        summary_lines.append(f"- 分类列数量: {len(self.data_info['categorical_columns'])}\n")
        summary_lines.append(f"- 内存占用: {self.data_info['memory_usage']:.2f} MB\n\n")

        # 数值列统计
        if self.data_info['numeric_columns']:
            summary_lines.append("## 数值列统计特征\n\n")
            numeric_df = df[self.data_info['numeric_columns']]
            
            # 计算统计特征
            stats = numeric_df.describe()
            stats.loc['skew'] = numeric_df.skew()
            stats.loc['kurt'] = numeric_df.kurtosis()
            
            summary_lines.append("| 列名 | 均值 | 标准差 | 最小值 | 最大值 | 偏度 | 峰度 |\n")
            summary_lines.append("|------|------|--------|--------|--------|------|------|\n")
            
            for col in self.data_info['numeric_columns'][:10]:
                mean = stats.loc['mean', col]
                std = stats.loc['std', col]
                min_val = stats.loc['min', col]
                max_val = stats.loc['max', col]
                skew = stats.loc['skew', col]
                kurt = stats.loc['kurt', col]
                summary_lines.append(f"| {col} | {mean:.2f} | {std:.2f} | {min_val:.2f} | {max_val:.2f} | {skew:.2f} | {kurt:.2f} |\n")
            
            summary_lines.append("\n")

        # 目标变量分析
        if self.target_column and self.target_column in df.columns:
            summary_lines.append(f"## 目标变量分析 ({self.target_column})\n\n")
            
            if self.target_column in self.data_info['numeric_columns']:
                target_series = df[self.target_column]
                summary_lines.append(f"- 类型: 数值型\n")
                summary_lines.append(f"- 均值: {target_series.mean():.2f}\n")
                summary_lines.append(f"- 标准差: {target_series.std():.2f}\n")
                summary_lines.append(f"- 最小值: {target_series.min():.2f}\n")
                summary_lines.append(f"- 最大值: {target_series.max():.2f}\n")
                summary_lines.append(f"- 偏度: {target_series.skew():.2f}\n")
                summary_lines.append(f"- 峰度: {target_series.kurtosis():.2f}\n")
            else:
                target_series = df[self.target_column]
                value_counts = target_series.value_counts()
                summary_lines.append(f"- 类型: 分类型\n")
                summary_lines.append(f"- 唯一值数量: {len(value_counts)}\n")
                summary_lines.append(f"- 分布:\n")
                for val, count in value_counts.head(10).items():
                    summary_lines.append(f"  - {val}: {count} ({count/len(df)*100:.1f}%)\n")
            
            summary_lines.append("\n")

        # 相关性分析（数值列）
        if len(self.data_info['numeric_columns']) > 1:
            summary_lines.append("## 高相关特征对（相关系数 > 0.7）\n\n")
            
            corr_matrix = df[self.data_info['numeric_columns']].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_val
                        ))
            
            if high_corr_pairs:
                summary_lines.append("| 特征1 | 特征2 | 相关系数 |\n")
                summary_lines.append("|-------|-------|----------|\n")
                for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                    summary_lines.append(f"| {f1} | {f2} | {corr:.3f} |\n")
            else:
                summary_lines.append("未发现高相关特征对\n")
            
            summary_lines.append("\n")

        return "".join(summary_lines)

    def generate_feature_suggestions(self) -> str:
        """
        基于探索性分析结果生成特征工程建议

        Returns:
            特征工程建议
        """
        if not self.exploration_result:
            return "请先进行探索性分析"

        prompt_template = self.config_loader.get_prompt("data_exploration", "feature_suggestions_prompt")
        user_input = prompt_template.format(
            exploration_result=self.exploration_result,
            skill_content=self._get_exploration_skill_content(),
            task_context=getattr(self, 'task_context', ''),
        )

        result = self.run(user_input, stage="feature_suggestions")

        return result.get("answer", "")
