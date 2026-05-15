"""
数据聚合 Agent 模块

支持多表、多粒度数据聚合为一张建模就绪的宽表。
适用于反欺诈、电商推荐、工业预测、医疗等需要跨表聚合的场景。
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.react_agent import ReActAgent
from ..config import get_config_loader
from ..logger.llm_logger import LLMLogger


class DataAggregationAgent(ReActAgent):
    """
    通用多表聚合 Agent

    分析多张数据表的结构和关联关系，生成聚合方案并执行，
    将多表合并为一张建模就绪的宽表。
    """

    def __init__(self, llm: Any = None, session_id: str = None, verbose: bool = False):
        super().__init__(llm=llm, session_id=session_id, max_iterations=10, verbose=verbose)
        self.data_paths: List[str] = []
        self.target_column: Optional[str] = None
        self.task_type: str = "classification"
        self.task_description: str = ""
        self.aggregation_plan: Optional[str] = None
        self.aggregation_code: Optional[str] = None
        self.aggregated_data_path: Optional[str] = None
        self.config_loader = get_config_loader()
        self.logger = LLMLogger(session_id=session_id)

    def _register_default_tools(self):
        super()._register_default_tools()
        from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
        from ..tools.stage_tools import StageResultTool
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())
        self.register_tool("query_stage_result", StageResultTool(session_id=self.session_id))

    def get_system_prompt(self) -> str:
        return self.config_loader.get_prompt("data_aggregation", "system_prompt")

    def generate_aggregation_plan(
        self,
        data_paths: List[str],
        target_column: str = None,
        task_type: str = "classification",
        task_description: str = "",
    ) -> str:
        """
        分析多表关系并生成聚合方案。

        Args:
            data_paths: 所有数据文件路径列表
            target_column: 目标列名
            task_type: 任务类型
            task_description: 用户建模背景描述

        Returns:
            聚合方案 (Markdown)
        """
        self.data_paths = data_paths
        self.target_column = target_column
        self.task_type = task_type
        self.task_description = task_description

        import pandas as pd

        data_files_info_parts = []
        for i, path in enumerate(data_paths):
            try:
                df = pd.read_csv(path)
                info = (
                    f"### 表 {i}: {os.path.basename(path)}\n"
                    f"- 路径: {path}\n"
                    f"- 形状: {df.shape[0]} 行 × {df.shape[1]} 列\n"
                    f"- 列名: {', '.join(df.columns.tolist())}\n"
                    f"- 数据类型:\n"
                )
                for col in df.columns:
                    info += f"  - {col}: {df[col].dtype}"
                    if df[col].nunique() < 20:
                        info += f" (唯一值: {df[col].nunique()})"
                    info += "\n"
                info += f"- 前5行预览:\n```\n{df.head(5).to_string(index=False)}\n```\n"
                data_files_info_parts.append(info)
            except Exception as e:
                data_files_info_parts.append(
                    f"### 表 {i}: {os.path.basename(path)}\n- 读取失败: {e}\n"
                )

        data_files_info = "\n".join(data_files_info_parts)

        prompt_template = self.config_loader.get_prompt("data_aggregation", "plan_generation")
        user_input = prompt_template.format(
            data_files_info=data_files_info,
            target_column=target_column or "未指定",
            task_type=task_type,
            task_description=task_description or "未提供",
        )

        result = self.run(user_input, stage="data_aggregation_plan")
        self.aggregation_plan = result.get("answer", "")
        return self.aggregation_plan

    def revise_plan(self, current_plan: str, modifications: str, **kwargs) -> str:
        """基于用户反馈修订聚合方案。"""
        prompt = f"""# 数据聚合方案修订

## 当前方案

{current_plan}

## 用户修改要求

{modifications}

## 要求

请根据用户的修改要求，修订数据聚合方案。保持方案的完整性和可执行性。
输出完整的修订后方案（Markdown 格式）。
"""
        result = self.run(prompt, stage="data_aggregation_plan_revision")
        self.aggregation_plan = result.get("answer", "")
        return self.aggregation_plan

    def generate_aggregation_code(self, modifications: str = None) -> str:
        """
        基于方案生成聚合代码并执行。

        Args:
            modifications: 用户修改内容

        Returns:
            生成的聚合代码
        """
        if not self.aggregation_plan:
            raise ValueError("请先生成聚合方案")

        from ..utils.codeact_agent import CodeActAgent

        modifications_text = f"\n用户修改要求:\n{modifications}\n" if modifications else ""

        self.aggregated_data_path = str(
            self.asset_manager.session_dir / "data" / "aggregated_data.csv"
        )

        prompt_template = self.config_loader.get_prompt("data_aggregation", "code_generation_full")
        task_prompt = prompt_template.format(
            data_paths_json=json.dumps(self.data_paths, ensure_ascii=False),
            output_path=self.aggregated_data_path,
            plan=self.aggregation_plan,
            modifications=modifications_text,
        )

        codeact = CodeActAgent(
            llm=self.llm, max_iterations=5, timeout=300, session_id=self.session_id
        )
        if self._stream_callback:
            codeact.set_stream_callback(self._stream_callback)

        context = {
            "data_paths": self.data_paths,
            "output_path": self.aggregated_data_path,
            "target_column": self.target_column,
        }

        result = codeact.generate_and_execute(
            task_prompt=task_prompt,
            context=context,
            required_outputs=[],
            required_filepath=self.aggregated_data_path,
            output_validator=self._validate_aggregation_output,
            deterministic_fallback=self._deterministic_aggregation_fallback,
            stage="data_aggregation_code_generation",
        )

        if result.success:
            self.aggregation_code = result.code
            print(f"\n[CodeAct] 聚合代码生成成功，迭代次数: {result.iterations}")

            if self.aggregation_code:
                self.asset_manager.save_code(
                    code=self.aggregation_code,
                    filename="aggregation.py",
                    metadata={
                        "stage": "data_aggregation",
                        "data_paths": self.data_paths,
                        "target_column": self.target_column,
                        "task_type": self.task_type,
                        "execution_success": True,
                        "iterations": result.iterations,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            return self.aggregation_code
        else:
            print(f"\n[CodeAct] 聚合代码生成失败: {result.error}")
            raise ValueError(f"聚合代码生成失败: {result.error}")

    def _validate_aggregation_output(
        self, output_path: str, context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """验证聚合输出文件。"""
        import pandas as pd

        if not os.path.exists(output_path):
            return False, f"输出文件不存在: {output_path}"

        df = pd.read_csv(output_path)
        if df.shape[0] <= 0:
            return False, "聚合输出为空"

        target = context.get("target_column") or self.target_column
        if target and target not in df.columns:
            return False, f"聚合输出缺少目标列: {target}"

        all_nan_cols = [c for c in df.columns if df[c].isna().all()]
        if all_nan_cols:
            return False, f"存在全 NaN 列: {all_nan_cols[:5]}"

        return True, f"{df.shape[0]} 行 × {df.shape[1]} 列"

    def _deterministic_aggregation_fallback(
        self, context: Dict[str, Any], output_path: str
    ) -> Tuple[bool, str]:
        """确定性兜底：简单 merge 所有表。"""
        import pandas as pd

        data_paths = context.get("data_paths") or self.data_paths
        if not data_paths:
            return False, "无数据路径"

        try:
            dfs = [pd.read_csv(p) for p in data_paths]
        except Exception as e:
            return False, f"读取数据失败: {e}"

        if len(dfs) == 1:
            result_df = dfs[0]
        else:
            result_df = dfs[0]
            for df in dfs[1:]:
                common_cols = list(set(result_df.columns) & set(df.columns))
                if common_cols:
                    result_df = result_df.merge(df, on=common_cols, how="left")
                else:
                    result_df = pd.concat([result_df, df], axis=1)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        return True, f"兜底聚合完成: {result_df.shape[0]} 行 × {result_df.shape[1]} 列"

    def get_modifiable_aspects(self) -> list:
        return ["表角色划分", "关联键", "聚合粒度", "时间约束", "时间窗口", "聚合函数", "输出列"]
