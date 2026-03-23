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
        return """你是一位专业的数据探索性分析专家，专门分析清洗后的数据。

你的职责：
1. 分析数据分布特征（均值、方差、偏度、峰度等）
2. 分析特征相关性（相关系数矩阵）
3. 分析目标变量分布
4. 为特征工程提供建议

**重要原则**：
- 必须使用用户上传的实际数据文件进行分析
- 禁止使用示例数据或虚构数据
- 所有分析结果必须基于实际数据的统计信息
- 你分析的是已经清洗过的干净数据

工作原则：
- 专注于数据的统计特征和分布
- 分析特征之间的相关性
- 为特征工程提供有价值的建议
- 生成 Markdown 格式的分析报告

你可以使用以下工具来完成任务：
- load_data: 加载数据文件
- analyze_data: 分析数据的统计信息

请按照 ReAct 格式进行思考和行动。"""

    def explore(
        self,
        data_path: str,
        target_column: str = None,
        task_type: str = "classification",
        task_description: str = "",
        cleaning_report: str = None
    ) -> Dict[str, Any]:
        """
        对清洗后的数据进行探索性分析

        Args:
            data_path: 数据文件路径（清洗后的数据）
            target_column: 目标列名
            task_type: 任务类型
            task_description: 用户的建模背景和要求
            cleaning_report: 数据清洗报告（可选）

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

            # 构建数据摘要
            data_summary = self._build_data_summary(df)

        except Exception as e:
            data_summary = f"无法加载数据文件: {data_path}\n错误: {str(e)}"

        # 加载 data-analysis skill 内容
        techniques = self.skill_loader.get_skill_reference("data-analysis-1.0.2", "techniques.md")

        # 构建 skill 内容
        skill_content = ""
        if techniques:
            skill_content += f"## 数据分析技术参考\n\n{techniques[:1500]}\n\n"

        # 构建用户提示词
        task_context = ""
        if task_description:
            task_context = f"""
## 用户建模背景和要求

{task_description}

**重要：请在探索性分析中充分考虑用户的建模背景和要求。**

"""

        cleaning_context = ""
        if cleaning_report:
            cleaning_context = f"""
## 数据清洗报告（来自数据清洗阶段）

{cleaning_report[:2000]}

**重要：请结合清洗报告中的处理内容进行探索性分析。**

"""

        user_input = f"""请对以下清洗后的数据进行探索性分析：

{task_context}{cleaning_context}{data_summary}

{skill_content}

请生成 Markdown 格式的探索性分析报告，包括：
1. 数据分布特征分析（均值、方差、偏度、峰度等）
2. 特征相关性分析（相关系数矩阵，找出高相关特征对）
3. 目标变量分析（如果指定了目标列）
4. 特征重要性初步评估
5. **特征工程建议**

重要：分析完成后，必须明确指出下一步应该进行特征工程。
"""

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

        user_input = f"""基于以下探索性分析结果，生成具体的特征工程建议：

{self.exploration_result}

请生成具体的特征工程建议，包括：
1. 特征变换建议（标准化、归一化、对数变换等）
2. 特征组合建议
3. 特征选择建议
4. 编码建议（分类变量）
"""

        result = self.run(user_input, stage="feature_suggestions")

        return result.get("answer", "")
