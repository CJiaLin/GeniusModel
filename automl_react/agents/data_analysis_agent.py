"""
数据分析 Agent 模块

实现数据分析、数据质量评估、生成分析报告
"""

import json
import re
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..core.react_agent import ReActAgent
from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
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
        self.problem_definition_plan: Optional[str] = None
        self.problem_definition_payload: Optional[Dict[str, Any]] = None
        self.config_loader = get_config_loader()
        self.logger = LLMLogger(session_id=session_id)

    def _register_default_tools(self):
        """注册默认工具"""
        super()._register_default_tools()
        from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
        from ..tools.profile_tools import DataProfileTool
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())
        self.register_tool("profile_data", DataProfileTool())

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.config_loader.get_prompt("problem_definition", "system_prompt")

    def _collect_data_context(
        self,
        data_path: str,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> str:
        """读取数据并构建问题定义所需的数据事实。"""
        try:
            # 智能读取：检测文件格式和编码
            with open(data_path, "rb") as f_check:
                magic = f_check.read(4)
            if magic == b'PK\x03\x04':
                df = pd.read_excel(data_path)
            elif data_path and data_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(data_path)
            else:
                for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
                    try:
                        df = pd.read_csv(data_path, encoding=enc)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                else:
                    df = pd.read_csv(data_path, encoding="utf-8", errors="replace")

            self.data_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "numeric_columns": list(df.select_dtypes(include=["int64", "float64"]).columns),
                "categorical_columns": list(df.select_dtypes(include=["object", "category", "bool"]).columns),
                "missing_values": df.isnull().sum().to_dict(),
                "missing_ratio": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
                "duplicate_rows": int(df.duplicated().sum()),
            }

            missing_sorted = sorted(
                [(k, v) for k, v in self.data_info["missing_values"].items() if v > 0],
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            preview_columns = ", ".join(self.data_info["columns"][:20])
            numeric_preview = ", ".join(self.data_info["numeric_columns"][:12]) or "无"
            categorical_preview = ", ".join(self.data_info["categorical_columns"][:12]) or "无"

            target_fact = ""
            if target_column:
                target_fact = f"- **候选目标列**: {target_column}\n"
                if target_column in df.columns:
                    target_fact += f"- **候选目标列类型**: {df[target_column].dtype}\n"
                    target_fact += f"- **候选目标列唯一值数**: {df[target_column].nunique()}\n"

            task_type_fact = f"- **候选任务类型**: {task_type}\n" if task_type else ""

            current_data_context = f"""
## 数据事实快照

- **文件路径**: {data_path}
- **数据形状**: {self.data_info['shape'][0]} 行 × {self.data_info['shape'][1]} 列
- **总列数**: {len(self.data_info['columns'])}
- **重复行数**: {self.data_info['duplicate_rows']}
{target_fact}{task_type_fact}
## 列名预览

{preview_columns}

## 缺失值情况 (Top 10)

"""
            for col, missing in missing_sorted:
                ratio = self.data_info["missing_ratio"][col]
                current_data_context += f"- **{col}**: {missing} 个缺失 ({ratio:.1f}%)\n"

            if not missing_sorted:
                current_data_context += "- 无缺失值\n"

            current_data_context += f"""

## 字段类型概览

- **数值列数量**: {len(self.data_info['numeric_columns'])}
- **类别列数量**: {len(self.data_info['categorical_columns'])}
- **主要数值列**: {numeric_preview}
- **主要类别列**: {categorical_preview}

重要：你必须基于这些真实字段和用户任务描述明确建模问题定义，而不是泛泛讨论。
"""
            return current_data_context
        except Exception as exc:
            self.data_info = None
            return f"无法加载数据文件: {data_path}\n错误: {exc}"

    def _build_default_problem_definition(
        self,
        target_column: Optional[str],
        task_type: Optional[str],
        task_description: str,
    ) -> Dict[str, Any]:
        """当 LLM 未按要求输出 JSON 时，构造可用的兜底问题定义。"""
        return {
            "task_type": task_type or "unknown",
            "target_column": target_column or "unknown",
            "prediction_target": target_column or "unknown",
            "prediction_timing": "待补充，当前任务描述未明确说明预测时点",
            "primary_metric": "待确认",
            "secondary_metrics": [],
            "business_constraints": [],
            "success_criteria": [],
            "assumptions": [
                "问题定义由系统根据当前任务描述自动归纳，建议在确认时补充更精确的业务口径"
            ],
            "open_questions": [
                "预测时点是否与训练样本观测时点一致",
                "主评估指标是否应以业务原始尺度为准",
            ],
            "raw_task_description": task_description or "",
        }

    def _extract_problem_definition_payload(
        self,
        answer: str,
        target_column: Optional[str],
        task_type: Optional[str],
        task_description: str,
    ) -> Dict[str, Any]:
        """从 LLM 回答中提取结构化问题定义 JSON。"""
        matches = re.findall(r"```json\s*(\{.*?\})\s*```", answer, flags=re.DOTALL)
        for match in reversed(matches):
            try:
                payload = json.loads(match)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
        return self._build_default_problem_definition(target_column, task_type, task_description)

    def generate_problem_definition(
        self,
        data_path: str,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
        task_description: str = "",
        workflow_mode: str = "full",
    ) -> str:
        """基于用户任务描述和数据事实生成结构化问题定义。"""
        self.data_path = data_path

        task_context = ""
        if task_description:
            task_context = f"""
## 用户任务描述

{task_description}

"""

        # 根据工作流模式调整输出 schema 和提示
        if workflow_mode == "schema_only":
            output_schema = {
                "task_type": "classification/regression/other",
                "target_column": "目标列名（如适用）",
                "prediction_target": "预测目标的业务表述",
                "prediction_timing": "预测发生时点和可用信息边界",
                "feature_design_goals": ["特征设计目标1", "特征设计目标2"],
                "business_constraints": ["业务约束1"],
                "assumptions": ["关键假设1"],
                "open_questions": ["待确认问题1"],
            }
            mode_hint = """
注意：当前为 Schema-only 模式（仅有数据字典，无实际数据），目标是输出特征工程设计方案。
- 不需要输出评估指标（无数据无法计算）
- 重点关注：业务目标 → 特征设计方向的映射、特征构造逻辑、时间窗口策略
- prediction_timing 应明确特征可用的信息边界（防止数据穿越）
"""
        elif workflow_mode == "feature_only":
            output_schema = {
                "task_type": "classification/regression/other",
                "target_column": "目标列名",
                "prediction_target": "预测目标的业务表述",
                "prediction_timing": "预测发生时点和可用信息边界",
                "primary_metric": "主指标（特征质量指标，如 IV/PSI/VIF 等）",
                "secondary_metrics": ["辅助指标1（如 KS、相关系数、缺失率等）"],
                "business_constraints": ["业务约束1"],
                "success_criteria": ["成功标准1（如 IV>0.02 的特征数量、PSI<0.1 等）"],
                "assumptions": ["关键假设1"],
                "open_questions": ["待确认问题1"],
            }
            mode_hint = """
注意：当前为仅特征工程模式（有数据，目标是构建高质量特征，不训练完整模型）。
- 评估指标应聚焦于特征质量而非模型效果，例如：
  - IV (Information Value)：衡量特征对目标的区分能力
  - PSI (Population Stability Index)：衡量特征分布稳定性
  - VIF (Variance Inflation Factor)：衡量多重共线性
  - KS / AUC（单特征）：单变量区分度
  - 缺失率、覆盖率、零值率
- success_criteria 应针对特征质量设定（如"IV>0.02 的特征不少于 N 个"）
"""
        else:
            output_schema = {
                "task_type": "classification/regression/other",
                "target_column": "目标列名",
                "prediction_target": "预测目标的业务表述",
                "prediction_timing": "预测发生时点和可用信息边界",
                "primary_metric": "主指标",
                "secondary_metrics": ["辅助指标1", "辅助指标2"],
                "business_constraints": ["业务约束1"],
                "success_criteria": ["成功标准1"],
                "assumptions": ["关键假设1"],
                "open_questions": ["待确认问题1"],
            }
            mode_hint = ""

        prompt_template = self.config_loader.get_prompt("problem_definition", "problem_definition_prompt")
        user_input = prompt_template.format(
            task_context=task_context,
            current_data_context=self._collect_data_context(data_path, target_column, task_type),
            candidate_target_column=target_column or "未提供",
            candidate_task_type=task_type or "未提供",
            output_schema_json=json.dumps(output_schema, ensure_ascii=False, indent=2),
        )

        if mode_hint:
            user_input = mode_hint + "\n" + user_input

        result = self.run(user_input, stage="problem_definition")
        answer = result.get("answer", "")
        self.problem_definition_plan = answer
        self.problem_definition_payload = self._extract_problem_definition_payload(
            answer,
            target_column=target_column,
            task_type=task_type,
            task_description=task_description,
        )
        return answer

    def get_problem_definition_payload(self) -> Dict[str, Any]:
        """返回当前问题定义结构化结果。"""
        return self.problem_definition_payload or {}

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

            # 构建当前数据上下文
            missing_sorted = sorted(
                [(k, v) for k, v in self.data_info["missing_values"].items() if v > 0],
                key=lambda x: x[1],
                reverse=True
            )[:10]

            current_data_context = f"""
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
                current_data_context += f"- **{col}**: {missing} 个缺失 ({ratio:.1f}%)\n"

            if not missing_sorted:
                current_data_context += "- 无缺失值\n"

            current_data_context += f"""
## 数值列统计

主要数值列: {', '.join(self.data_info['numeric_columns'][:10])}

## 分类列

主要分类列: {', '.join(self.data_info['categorical_columns'][:10])}

重要：请基于上述实际数据生成分析报告。
"""

        except Exception as e:
            current_data_context = f"无法加载数据文件: {data_path}\n错误: {str(e)}"

        # 构建用户提示词
        task_context = ""
        if task_description:
            task_context = f"""
## 用户建模背景和要求

{task_description}

**重要：请在分析过程中充分考虑用户的建模背景和要求。**

"""

        prompt_template = self.config_loader.get_prompt("problem_definition", "analysis_prompt")
        user_input = prompt_template.format(
            task_context=task_context,
            current_data_context=current_data_context,
        )

        # 调用 LLM 生成报告
        result = self.run(user_input, stage="data_analysis")

        self.analysis_result = result.get("answer", "")

        return result

    def get_analysis_summary(self) -> str:
        """获取分析结果摘要"""
        return self.analysis_result or ""
