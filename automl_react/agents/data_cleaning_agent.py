"""
数据清洗 Agent 模块

实现数据清洗思路生成、用户确认、代码生成与执行
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from ..core.react_agent import ReActAgent, ConfirmationRequired
from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
from ..config import get_config_loader


class DataCleaningAgent(ReActAgent):
    """
    数据清洗 Agent

    基于 ReAct 架构的数据清洗 Agent，支持：
    1. 数据质量分析
    2. 清洗思路生成（参考 data-analysis skill）
    3. 用户确认流程
    4. 代码生成与执行
    5. 清洗结果保存

    Attributes:
        session_id: 会话ID
        data_path: 数据文件路径
        cleaning_plan: 清洗方案
        cleaning_code: 清洗代码
    """

    def __init__(self, llm: Any = None, session_id: str = None, verbose: bool = False):
        super().__init__(llm=llm, session_id=session_id, max_iterations=10, verbose=verbose)
        self.data_path: Optional[str] = None
        self.data_info: Optional[Dict] = None
        self.cleaning_plan: Optional[str] = None
        self.cleaning_code: Optional[str] = None
        self.config_loader = get_config_loader()

    def _register_default_tools(self):
        """注册默认工具"""
        super()._register_default_tools()
        from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
        from ..tools.profile_tools import DataProfileTool
        from ..tools.stage_tools import StageResultTool
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())
        self.register_tool("profile_data", DataProfileTool())
        self.register_tool("query_stage_result", StageResultTool(session_id=self.session_id))

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.config_loader.get_prompt("data_cleaning", "system_prompt")

    def analyze_data(self, data_path: str) -> Dict[str, Any]:
        """
        分析数据质量

        Args:
            data_path: 数据文件路径

        Returns:
            数据分析结果
        """
        self.data_path = data_path

        # 构建分析提示词
        prompt_template = self.config_loader.get_prompt("data_cleaning", "quality_analysis_prompt")
        pitfalls_content = "如需参考数据陷阱指南，请使用 read_skill 工具读取 data-analysis-1.0.2 的 pitfalls 章节。"
        user_input = prompt_template.format(
            data_path=data_path,
            pitfalls_content=pitfalls_content,
        )

        result = self.run(user_input, stage="data_analysis")

        # 保存数据信息
        self.data_info = result.get("data_info", {})

        return result

    def analyze_data_quality(self, data_path: str, task_description: str = "") -> Dict[str, Any]:
        """
        分析数据质量，识别数据质量问题

        Args:
            data_path: 数据文件路径
            task_description: 用户的建模背景和要求

        Returns:
            数据质量分析结果
        """
        import pandas as pd
        import numpy as np

        self.data_path = data_path

        try:
            df = pd.read_csv(data_path)

            # 1. 数据完整性分析（缺失值）
            missing_analysis = {}
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_ratio = missing_count / len(df) * 100
                if missing_count > 0:
                    missing_analysis[col] = {
                        "missing_count": int(missing_count),
                        "missing_ratio": round(missing_ratio, 2),
                        "dtype": str(df[col].dtype)
                    }

            # 2. 数据一致性分析（异常值检测）
            outlier_analysis = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                if len(outliers) > 0:
                    outlier_analysis[col] = {
                        "outlier_count": len(outliers),
                        "outlier_ratio": round(len(outliers) / len(df) * 100, 2),
                        "lower_bound": round(lower_bound, 2),
                        "upper_bound": round(upper_bound, 2)
                    }

            # 3. 数据唯一性分析（重复值）
            duplicate_rows = df.duplicated().sum()
            duplicate_analysis = {
                "duplicate_rows": int(duplicate_rows),
                "duplicate_ratio": round(duplicate_rows / len(df) * 100, 2)
            }

            # 4. 数据类型分析
            dtype_analysis = {}
            for col in df.columns:
                dtype_analysis[col] = {
                    "dtype": str(df[col].dtype),
                    "unique_count": int(df[col].nunique()),
                    "sample_values": df[col].dropna().head(3).tolist()
                }

            # 保存数据信息
            self.data_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "missing_analysis": missing_analysis,
                "outlier_analysis": outlier_analysis,
                "duplicate_analysis": duplicate_analysis,
                "dtype_analysis": dtype_analysis,
                "numeric_columns": list(numeric_cols),
                "categorical_columns": list(df.select_dtypes(include=['object']).columns)
            }

            # 生成数据质量报告
            quality_report = self._generate_quality_report(task_description)

            return {
                "success": True,
                "data_info": self.data_info,
                "quality_report": quality_report
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _generate_quality_report(self, task_description: str = "") -> str:
        """
        生成数据质量报告

        Args:
            task_description: 用户的建模背景和要求

        Returns:
            数据质量报告（Markdown 格式）
        """
        if not self.data_info:
            return "未进行数据质量分析"

        report_lines = ["# 数据质量分析报告\n"]

        # 添加用户建模背景
        if task_description:
            report_lines.append("## 用户建模背景和要求\n")
            report_lines.append(f"{task_description}\n\n")

        # 数据基本信息
        report_lines.append("## 数据基本信息\n")
        report_lines.append(f"- 数据形状: {self.data_info['shape'][0]} 行 × {self.data_info['shape'][1]} 列\n")
        report_lines.append(f"- 数值列数量: {len(self.data_info['numeric_columns'])}\n")
        report_lines.append(f"- 分类列数量: {len(self.data_info['categorical_columns'])}\n\n")

        # 缺失值分析
        report_lines.append("## 数据完整性分析（缺失值）\n")
        if self.data_info['missing_analysis']:
            report_lines.append("| 列名 | 缺失数量 | 缺失比例 | 数据类型 | 建议处理方式 |\n")
            report_lines.append("|------|----------|----------|----------|--------------|\n")
            for col, info in sorted(self.data_info['missing_analysis'].items(), 
                                   key=lambda x: x[1]['missing_ratio'], reverse=True):
                suggestion = "删除列" if info['missing_ratio'] > 50 else "填充/插值" if info['missing_ratio'] > 5 else "简单填充"
                report_lines.append(f"| {col} | {info['missing_count']} | {info['missing_ratio']}% | {info['dtype']} | {suggestion} |\n")
        else:
            report_lines.append("无缺失值\n")
        report_lines.append("\n")

        # 异常值分析
        report_lines.append("## 数据一致性分析（异常值）\n")
        if self.data_info['outlier_analysis']:
            report_lines.append("| 列名 | 异常值数量 | 异常值比例 | 正常范围 | 建议处理方式 |\n")
            report_lines.append("|------|------------|------------|----------|--------------|\n")
            for col, info in self.data_info['outlier_analysis'].items():
                suggestion = "删除" if info['outlier_ratio'] > 10 else "Winsorize" if info['outlier_ratio'] > 1 else "保留"
                report_lines.append(f"| {col} | {info['outlier_count']} | {info['outlier_ratio']}% | [{info['lower_bound']}, {info['upper_bound']}] | {suggestion} |\n")
        else:
            report_lines.append("未检测到明显异常值\n")
        report_lines.append("\n")

        # 重复值分析
        report_lines.append("## 数据唯一性分析（重复值）\n")
        dup_info = self.data_info['duplicate_analysis']
        report_lines.append(f"- 重复行数: {dup_info['duplicate_rows']}\n")
        report_lines.append(f"- 重复比例: {dup_info['duplicate_ratio']}%\n")
        if dup_info['duplicate_rows'] > 0:
            report_lines.append("- 建议处理方式: 删除重复行\n")
        report_lines.append("\n")

        # 数据类型分析
        report_lines.append("## 数据类型分析\n")
        dtype_issues = []
        for col, info in self.data_info['dtype_analysis'].items():
            if info['dtype'] == 'object' and info['unique_count'] < 10:
                dtype_issues.append(f"- {col}: 可能是分类变量，建议转换为 category 类型")
            elif info['dtype'] == 'object' and info['unique_count'] > 100:
                dtype_issues.append(f"- {col}: 高基数文本列，可能需要文本处理或删除")

        if dtype_issues:
            report_lines.append("### 潜在数据类型问题\n")
            report_lines.extend([issue + "\n" for issue in dtype_issues])
        else:
            report_lines.append("数据类型无明显问题\n")
        report_lines.append("\n")

        # 总结和建议
        report_lines.append("## 数据质量总结和建议\n")
        total_issues = (len(self.data_info['missing_analysis']) + 
                       len(self.data_info['outlier_analysis']) + 
                       (1 if self.data_info['duplicate_analysis']['duplicate_rows'] > 0 else 0))
        report_lines.append(f"- 共发现 {total_issues} 类数据质量问题\n")
        report_lines.append("- 建议优先处理高缺失率列和重复行\n")
        report_lines.append("- 异常值处理需结合业务理解\n")

        return "".join(report_lines)

    def generate_cleaning_plan(self, data_path: str = None, task_description: str = "") -> str:
        """
        生成数据清洗方案（包含数据质量分析）

        Args:
            data_path: 数据文件路径
            task_description: 用户的建模背景和要求

        Returns:
            清洗方案（Markdown 格式）
        """
        path = data_path or self.data_path
        if not path:
            raise ValueError("请提供数据文件路径")

        self.data_path = path

        # 首先进行数据质量分析
        if not self.data_info:
            print(f"[DataCleaningAgent] 开始数据质量分析...")
            quality_result = self.analyze_data_quality(path, task_description)
            if not quality_result.get("success"):
                raise ValueError(f"数据质量分析失败: {quality_result.get('error')}")
            print(f"[DataCleaningAgent] 数据质量分析完成")

        # 使用数据质量报告作为方案输入上下文
        quality_report = self._generate_quality_report(task_description)
        
        # 打印数据质量报告摘要
        print(f"[DataCleaningAgent] 数据质量报告生成完成，长度: {len(quality_report)} 字符")
        print(f"[DataCleaningAgent] 数据形状: {self.data_info['shape']}")
        print(f"[DataCleaningAgent] 缺失值列数: {len(self.data_info['missing_analysis'])}")
        print(f"[DataCleaningAgent] 数据质量报告预览:\n{quality_report[:500]}...")

        # 从配置加载 Prompt
        prompt_template = self.config_loader.get_prompt("data_cleaning", "plan_generation")

        # 添加用户的建模背景
        task_context = ""
        if task_description:
            task_context = f"""
## 用户建模背景和要求

{task_description}

**重要：请在清洗方案中充分考虑用户的建模背景和要求。**

"""
        self.task_context = task_context

        user_input = prompt_template.format(
            data_path=path,
            quality_report=quality_report,
            task_context=task_context,
            shape=self.data_info['shape']
        )

        # 使用 ReAct 模式生成方案
        # LLM 可以调用 load_data 工具加载用户上传的数据（路径: {path}）
        result = self.run(user_input, stage="data_cleaning_plan")

        self.cleaning_plan = result.get("answer", "")

        return self.cleaning_plan

    def revise_plan(self, current_plan: str, modifications: str, **kwargs) -> str:
        """基于用户反馈修订清洗方案"""
        prompt_template = self.config_loader.get_prompt("data_cleaning", "plan_revision")
        user_input = prompt_template.format(
            current_plan=current_plan,
            user_modifications=modifications,
            data_path=self.data_path or "",
        )
        result = self.run(user_input, stage="data_cleaning_plan_revision")
        self.cleaning_plan = result.get("answer", "")
        return self.cleaning_plan

    def get_modifiable_aspects(self) -> list:
        return ["缺失值处理策略", "异常值处理方式", "重复行处理", "列删除/保留", "数据类型转换"]

    def request_user_confirmation(self) -> None:
        """
        请求用户确认清洗方案

        Raises:
            ConfirmationRequired: 需要用户确认时抛出
        """
        if not self.cleaning_plan:
            raise ValueError("请先生成清洗方案")

        # 参考的 skills
        skills_referenced = [
            {
                "name": "data-analysis-1.0.2",
                "files": ["techniques.md", "pitfalls.md"]
            }
        ]

        # 抛出确认异常
        raise ConfirmationRequired(
            stage="data_cleaning",
            proposal=self.cleaning_plan,
            skills_referenced=skills_referenced
        )

    def generate_cleaning_code(self, modifications: str = None) -> str:
        """
        生成清洗代码（使用 CodeAct 模式）

        Args:
            modifications: 用户修改内容

        Returns:
            清洗代码
        """
        if not self.cleaning_plan:
            raise ValueError("请先生成清洗方案")

        from ..utils.codeact_agent import CodeActAgent

        # 构建提示词
        modifications_text = f"\n用户修改要求:\n{modifications}\n" if modifications else ""

        # 构建数据信息摘要
        data_info_text = ""
        if self.data_info:
            # 获取缺失值列
            missing_cols = list(self.data_info.get('missing_analysis', {}).keys())[:10]
            
            data_info_text = f"""
## 数据信息

- 数据形状: {self.data_info['shape'][0]} 行 × {self.data_info['shape'][1]} 列
- 数值列: {', '.join(self.data_info['numeric_columns'][:15])}
- 分类列: {', '.join(self.data_info['categorical_columns'][:15])}
- 缺失值列: {', '.join(missing_cols)}

重要：请基于上述实际数据列名生成代码，不要使用示例数据中的列名。
"""

        self.cleaned_data_path = str(self.asset_manager.session_dir / "data" / "cleaned_data.csv")

        prompt_template = self.config_loader.get_prompt("data_cleaning", "code_generation_full")
        task_prompt = prompt_template.format(
            data_path=self.data_path,
            plan=self.cleaning_plan,
            modifications=modifications_text,
            data_info_text=data_info_text,
            cleaned_data_path=self.cleaned_data_path,
            task_context=getattr(self, 'task_context', ''),
        )

        # 使用 CodeActAgent 生成并执行代码
        codeact = CodeActAgent(llm=self.llm, max_iterations=5, timeout=300, session_id=self.session_id)

        # 准备执行上下文
        context = {
            "data_path": self.data_path,
            "cleaned_data_path": self.cleaned_data_path
        }

        # 生成代码并执行验证
        result = codeact.generate_and_execute(
            task_prompt=task_prompt,
            context=context,
            required_outputs=[],
            required_filepath=self.cleaned_data_path,
            output_validator=self._validate_cleaned_output,
            deterministic_fallback=self._deterministic_cleaning_fallback,
            stage="data_cleaning_code_generation",
        )

        if result.success:
            self.cleaning_code = result.code
            print(f"\n[CodeAct] 代码生成成功，迭代次数: {result.iterations}")
            
            # 保存代码到资产
            if self.cleaning_code:
                self.asset_manager.save_code(
                    code=self.cleaning_code,
                    filename="cleaning.py",
                    metadata={
                        "stage": "data_cleaning",
                        "data_path": self.data_path,
                        "execution_success": True,
                        "iterations": result.iterations,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            return self.cleaning_code
        else:
            print(f"\n[CodeAct] 代码生成失败: {result.error}")
            raise ValueError(f"代码生成失败: {result.error}")

    def _validate_cleaned_output(self, output_path: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """数据清洗输出校验：文件可读、非空、列结构有效。"""
        import pandas as pd

        df_out = pd.read_csv(output_path)
        if df_out.shape[0] <= 0:
            return False, "清洗输出为空"
        if df_out.shape[1] <= 0:
            return False, "清洗输出无列"

        in_path = context.get("data_path") or self.data_path
        if in_path:
            try:
                df_in = pd.read_csv(in_path)
                if df_out.shape[0] > max(df_in.shape[0] * 1.2, df_in.shape[0] + 1000):
                    return False, "清洗后行数异常膨胀"
            except Exception:
                pass

        return True, f"{df_out.shape[0]} 行 × {df_out.shape[1]} 列"

    def _deterministic_cleaning_fallback(self, context: Dict[str, Any], output_path: str) -> Tuple[bool, str]:
        """确定性兜底：无需 LLM，执行基础清洗并确保落盘。"""
        import pandas as pd

        in_path = context.get("data_path") or self.data_path
        if not in_path:
            return False, "缺少输入数据路径"

        try:
            df = pd.read_csv(in_path)
        except Exception as e:
            return False, f"读取输入数据失败: {e}"

        if df.empty:
            return False, "输入数据为空"

        # 基础清洗：去重 + 缺失值填充
        df = df.drop_duplicates()

        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                med = df[col].median()
                if pd.notna(med):
                    df[col] = df[col].fillna(med)

        other_cols = [c for c in df.columns if c not in numeric_cols]
        for col in other_cols:
            if df[col].isnull().any():
                mode_series = df[col].mode(dropna=True)
                fill_val = mode_series.iloc[0] if not mode_series.empty else "UNKNOWN"
                df[col] = df[col].fillna(fill_val)

        try:
            df.to_csv(output_path, index=False)
            return True, f"确定性清洗完成并保存到 {output_path}"
        except Exception as e:
            return False, f"保存清洗结果失败: {e}"

    def _append_cleaned_data_save_fallback(self, code: str) -> str:
        """为清洗代码补充结果落盘语句，避免因未保存文件导致流程失败。"""
        if not code or not code.strip():
            return code

        # 若代码已明确写入目标文件，则不重复追加。
        if self.cleaned_data_path in code and (
            "to_csv(" in code or "to_parquet(" in code or "to_pickle(" in code
        ):
            return code

        fallback_block = f"""

# Fallback save block injected by DataCleaningAgent
try:
    if 'df' in locals() and hasattr(df, 'to_csv'):
        df.to_csv(r'{self.cleaned_data_path}', index=False)
        print('Fallback save using df completed')
    elif 'cleaned_df' in locals() and hasattr(cleaned_df, 'to_csv'):
        cleaned_df.to_csv(r'{self.cleaned_data_path}', index=False)
        print('Fallback save using cleaned_df completed')
except Exception as _fallback_error:
    print(f'Fallback save failed: {{_fallback_error}}')
"""

        return code.rstrip() + "\n" + fallback_block

    def execute_cleaning(self, code: str = None) -> Dict[str, Any]:
        """
        执行清洗代码（使用 CodeActAgent）

        Args:
            code: 清洗代码，为 None 时使用已生成的代码

        Returns:
            执行结果
        """
        import os
        import pandas as pd

        cleaning_code = code or self.cleaning_code

        if not cleaning_code:
            raise ValueError("请先生成清洗代码")

        # 使用 CodeActAgent 执行代码
        from ..utils.codeact_agent import CodeActAgent
        
        codeact = CodeActAgent(llm=self.llm, max_iterations=1, timeout=300, session_id=self.session_id)
        
        context = {
            "data_path": self.data_path,
            "cleaned_data_path": self.cleaned_data_path
        }

        # 直接执行代码（不生成新代码）
        exec_result = codeact._execute_code(cleaning_code, context)

        # 检查是否成功生成了清洗后的数据文件
        file_exists = os.path.exists(self.cleaned_data_path)
        
        # 验证数据文件
        if file_exists:
            try:
                df = pd.read_csv(self.cleaned_data_path)
                print(f"[Agent] 清洗后数据验证成功: {df.shape[0]} 行 × {df.shape[1]} 列")
            except Exception as e:
                print(f"[Agent] 清洗后数据验证失败: {e}")
                file_exists = False

        # 构建结果
        result_info = {
            "success": file_exists,
            "cleaned_data_path": self.cleaned_data_path if file_exists else None,
            "original_path": self.data_path,
            "execution_output": exec_result.get("output", ""),
            "execution_error": exec_result.get("error") if not file_exists else None,
            "timestamp": datetime.now().isoformat()
        }

        # 保存结果信息到资产
        self.asset_manager.save_data(
            data=json.dumps(result_info, ensure_ascii=False, indent=2),
            filename="cleaning_result.json",
            asset_type="cleaning",
            metadata=result_info
        )

        return result_info

    def full_cleaning_workflow(
        self,
        data_path: str,
        skip_confirmation: bool = False
    ) -> Dict[str, Any]:
        """
        执行完整的数据清洗流程

        Args:
            data_path: 数据文件路径
            skip_confirmation: 是否跳过用户确认

        Returns:
            清洗结果
        """
        self.data_path = data_path

        # 1. 生成清洗方案
        plan = self.generate_cleaning_plan(data_path)

        # 2. 请求用户确认（如果不跳过）
        if not skip_confirmation:
            self.request_user_confirmation()

        # 3. 生成清洗代码
        code = self.generate_cleaning_code()

        # 4. 执行清洗
        result = self.execute_cleaning(code)

        return {
            "success": result.get("success", False),
            "plan": plan,
            "code": code,
            "result": result
        }
