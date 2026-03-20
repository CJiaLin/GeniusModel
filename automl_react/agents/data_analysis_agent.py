"""
数据分析 Agent 模块

实现数据分析、数据质量评估、生成分析报告
"""

import json
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..core.react_agent import ReActAgent
from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
from ..skills_loader import get_skill_loader
from ..config import get_config_loader
from ..logger.llm_logger import LLMLogger


class DataAnalysisAgent(ReActAgent):
    """
    数据分析 Agent

    基于 ReAct 架构的数据分析 Agent，支持：
    1. 数据加载和基本分析
    2. 数据质量评估
    3. 生成分析报告
    4. 引导下一步工作流程

    Attributes:
        session_id: 会话ID
        data_path: 数据文件路径
        analysis_result: 分析结果
    """

    def __init__(self, llm: Any = None, session_id: str = None, verbose: bool = False):
        super().__init__(llm=llm, session_id=session_id, max_iterations=10, verbose=verbose)
        self.data_path: Optional[str] = None
        self.data_info: Optional[Dict] = None
        self.analysis_result: Optional[str] = None
        self.skill_loader = get_skill_loader()
        self.config_loader = get_config_loader()
        self.logger = LLMLogger(session_id=session_id)

    def _register_default_tools(self):
        """注册默认工具"""
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一位专业的数据分析专家，专门帮助用户进行数据分析和质量评估。

你的职责：
1. 加载和分析数据
2. 评估数据质量（缺失值、异常值、重复值等）
3. 生成详细的分析报告
4. 引导用户进行下一步工作流程

**重要：标准工作流程顺序**
数据分析 → 数据清洗 → 特征工程 → 模型训练

工作原则：
- 仔细分析数据质量和分布
- 分析完成后，应引导用户进行数据清洗
- 提供清晰的执行步骤和结果解释
- 生成 Markdown 格式的分析报告

你可以使用以下工具来完成任务：
- load_data: 加载数据文件
- analyze_data: 分析数据的基本统计信息和质量

请按照 ReAct 格式进行思考和行动。"""

    def analyze(self, data_path: str, task_description: str = "") -> Dict[str, Any]:
        """
        分析数据并生成报告

        Args:
            data_path: 数据文件路径
            task_description: 用户的建模背景和要求

        Returns:
            分析结果
        """
        self.data_path = data_path

        # 首先加载并分析实际数据
        try:
            df = pd.read_csv(data_path)

            # 收集数据基本信息
            self.data_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "numeric_columns": list(df.select_dtypes(include=['int64', 'float64']).columns),
                "categorical_columns": list(df.select_dtypes(include=['object']).columns),
                "missing_values": df.isnull().sum().to_dict(),
                "missing_ratio": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
                "duplicate_rows": int(df.duplicated().sum())
            }

            # 构建数据摘要
            missing_sorted = sorted(
                [(k, v) for k, v in self.data_info["missing_values"].items() if v > 0],
                key=lambda x: x[1],
                reverse=True
            )[:10]

            data_summary = f"""
## 数据基本信息

- **文件路径**: {data_path}
- **数据形状**: {self.data_info['shape'][0]} 行 × {self.data_info['shape'][1]} 列
- **重复行数**: {self.data_info['duplicate_rows']}
- **数值列数量**: {len(self.data_info['numeric_columns'])}
- **分类列数量**: {len(self.data_info['categorical_columns'])}

## 缺失值情况 (Top 10)

"""
            for col, missing in missing_sorted:
                ratio = self.data_info["missing_ratio"][col]
                data_summary += f"- **{col}**: {missing} 个缺失 ({ratio:.1f}%)\n"

            if not missing_sorted:
                data_summary += "- 无缺失值\n"

            data_summary += f"""
## 数值列统计

主要数值列: {', '.join(self.data_info['numeric_columns'][:10])}

## 分类列

主要分类列: {', '.join(self.data_info['categorical_columns'][:10])}

重要：请基于上述实际数据生成分析报告。
"""

        except Exception as e:
            data_summary = f"无法加载数据文件: {data_path}\n错误: {str(e)}"

        # 加载 data-analysis skill 内容
        techniques = self.skill_loader.get_skill_reference("data-analysis-1.0.2", "techniques.md")
        pitfalls = self.skill_loader.get_skill_reference("data-analysis-1.0.2", "pitfalls.md")

        # 构建 skill 内容
        skill_content = ""
        if techniques:
            skill_content += f"## 数据分析技术参考\n\n{techniques[:1500]}\n\n"
        if pitfalls:
            skill_content += f"## 数据陷阱参考\n\n{pitfalls[:1500]}\n\n"

        # 构建用户提示词
        task_context = ""
        if task_description:
            task_context = f"""
## 用户建模背景和要求

{task_description}

**重要：请在分析过程中充分考虑用户的建模背景和要求。**

"""

        user_input = f"""请分析以下数据并生成详细的分析报告：

{task_context}{data_summary}

{skill_content}

请生成 Markdown 格式的分析报告，包括：
1. 数据概览
2. 数据质量评估
3. 关键特征分析
4. 建模建议（结合用户的建模背景和要求）
5. **下一步建议：引导进行数据清洗**

重要：分析完成后，必须明确指出下一步应该进行数据清洗。
"""

        # 调用 LLM 生成报告
        result = self.run(user_input, stage="data_analysis")

        self.analysis_result = result.get("answer", "")

        return result

    def get_analysis_summary(self) -> str:
        """获取分析结果摘要"""
        return self.analysis_result or ""
