"""
数据清洗 Agent 模块

实现数据清洗思路生成、用户确认、代码生成与执行
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..core.react_agent import ReActAgent, ConfirmationRequired
from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
from ..skills_loader import get_skill_loader
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
        self.skill_loader = get_skill_loader()
        self.config_loader = get_config_loader()

    def _register_default_tools(self):
        """注册默认工具"""
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        try:
            return self.config_loader.get_prompt("data_cleaning", "system_prompt")
        except KeyError:
            return """你是一位专业的数据清洗专家。

你的职责：
1. 分析数据质量问题（缺失值、异常值、重复值等）
2. 生成数据清洗方案
3. 编写清洗代码
4. 执行清洗并验证结果

**重要原则**：
- 必须使用用户上传的实际数据文件进行分析和清洗
- 禁止使用示例数据或虚构数据
- 所有分析结果必须基于实际数据的统计信息
- 清洗代码必须针对实际数据的列名和特征

请基于数据分析和最佳实践，生成详细的清洗方案。"""

    def analyze_data(self, data_path: str) -> Dict[str, Any]:
        """
        分析数据质量

        Args:
            data_path: 数据文件路径

        Returns:
            数据分析结果
        """
        self.data_path = data_path

        # 加载 data-analysis skill 的 pitfalls.md
        pitfalls = self.skill_loader.get_skill_reference("data-analysis-1.0.2", "pitfalls.md")

        # 构建分析提示词
        user_input = f"请分析数据文件: {data_path}"

        if pitfalls:
            user_input += f"\n\n参考以下数据陷阱指南:\n{pitfalls[:2000]}"

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

        # 使用数据质量报告作为数据摘要
        data_summary = self._generate_quality_report(task_description)
        
        # 打印数据质量报告摘要
        print(f"[DataCleaningAgent] 数据质量报告生成完成，长度: {len(data_summary)} 字符")
        print(f"[DataCleaningAgent] 数据形状: {self.data_info['shape']}")
        print(f"[DataCleaningAgent] 缺失值列数: {len(self.data_info['missing_analysis'])}")
        print(f"[DataCleaningAgent] 数据质量报告预览:\n{data_summary[:500]}...")

        # 加载 data-cleaning skill
        techniques = self.skill_loader.get_skill_reference("data-cleaning-1.0.0", "techniques.md")
        pitfalls = self.skill_loader.get_skill_reference("data-cleaning-1.0.0", "pitfalls.md")

        # 从配置加载 Prompt
        try:
            prompt_template = self.config_loader.get_prompt("data_cleaning", "plan_generation")
        except KeyError:
            prompt_template = """你是一位数据清洗专家。

================================================================================
📊 以下是用户上传的**实际数据**的质量分析报告（必须基于此生成方案）
================================================================================

{task_context}{data_summary}

================================================================================
📚 以下是数据清洗的技术参考（仅供参考，其中的示例数据不可用于方案）
================================================================================

{skill_content}

================================================================================

**🔴 极其重要的指示**：
1. **你必须基于上面的「实际数据质量分析报告」生成清洗方案**
2. **数据文件路径为: {data_path}**
3. **实际数据形状是 {shape}**
4. **上面的「技术参考」部分包含的是示例数据，仅供参考方法，不可用于方案**
5. **清洗方案中的所有列名必须与「实际数据质量分析报告」中的列名一致**
6. **禁止使用技术参考中的示例数据列名（如 sepal length、iris 等）**

请生成 Markdown 格式的清洗方案，方案必须基于「实际数据质量分析报告」。
"""

        # 构建 skill 内容（添加明显的边界标志）
        skill_content = ""
        if techniques or pitfalls:
            skill_content += "⚠️ 注意：以下内容为技术参考，包含示例数据，仅供参考方法\n\n"
            if techniques:
                skill_content += f"### 数据清洗技术参考\n\n{techniques[:1500]}\n\n"
            if pitfalls:
                skill_content += f"### 数据陷阱参考\n\n{pitfalls[:1500]}\n\n"
            skill_content += "⚠️ 技术参考结束\n"

        # 添加用户的建模背景
        task_context = ""
        if task_description:
            task_context = f"""
## 用户建模背景和要求

{task_description}

**重要：请在清洗方案中充分考虑用户的建模背景和要求。**

"""

        user_input = prompt_template.format(
            data_path=path,
            data_summary=data_summary,
            skill_content=skill_content,
            task_context=task_context,
            shape=self.data_info['shape']
        )

        # 使用 ReAct 模式生成方案
        # LLM 可以调用 load_data 工具加载用户上传的数据（路径: {path}）
        result = self.run(user_input, stage="data_cleaning_plan")

        self.cleaning_plan = result.get("answer", "")

        return self.cleaning_plan

    def request_user_confirmation(self) -> None:
        """
        请求用户确认清洗方案

        Raises:
            ConfirmationRequired: 需要用户确认时抛出
        """
        if not self.cleaning_plan:
            raise ValueError("请先生成清洗方案")

        # 生成代码预览
        code_preview = self._generate_code_preview()

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
            code_preview=code_preview,
            skills_referenced=skills_referenced
        )

    def _generate_code_preview(self) -> str:
        """生成代码预览"""
        if not self.cleaning_plan:
            return "# 代码将在确认后生成"

        # 从配置加载 Prompt
        try:
            prompt_template = self.config_loader.get_prompt("data_cleaning", "code_generation")
        except KeyError:
            prompt_template = """基于以下清洗方案，生成 Python 代码预览：

{plan}

请生成简洁的代码预览（仅展示主要步骤）。
"""

        user_input = prompt_template.format(plan=self.cleaning_plan[:1000])

        result = self.run(user_input, stage="data_cleaning_code_preview")

        return result.get("answer", "# 代码预览")

    def generate_cleaning_code(self, modifications: str = None) -> str:
        """
        生成清洗代码（使用结构化输出和迭代验证）

        Args:
            modifications: 用户修改内容

        Returns:
            清洗代码
        """
        if not self.cleaning_plan:
            raise ValueError("请先生成清洗方案")

        from ..utils.code_generator import CodeGenerator

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

        prompt = f"""基于以下清洗方案，生成完整的 Python 代码：

数据路径: {self.data_path}
{data_info_text}

清洗方案:
{self.cleaning_plan}

{modifications_text}

要求：
1. 使用 pandas 进行数据处理
2. 包含详细的注释
3. 保存清洗后的数据到指定路径: {self.data_path.replace('.csv', '_cleaned.csv')}
4. 返回清洗结果统计
5. 代码必须完整可执行，包含所有必要的导入语句
6. 必须使用上述实际数据的列名，不要使用示例数据

请生成完整的、可执行的 Python 代码。
"""

        # 使用代码生成器生成并验证代码
        code_gen = CodeGenerator(llm=self.llm)

        # 准备执行上下文
        context = {
            "data_path": self.data_path,
            "cleaned_data_path": self.data_path.replace('.csv', '_cleaned.csv')
        }

        # 生成代码并验证执行
        code, exec_result = code_gen.generate_code_with_validation(
            prompt=prompt,
            context=context,
            stage="data_cleaning_code",
            required_outputs=[]  # 数据清洗不强制要求特定输出变量
        )

        self.cleaning_code = code

        # 保存代码到资产
        if code:
            self.asset_manager.save_code(
                code=code,
                filename="cleaning.py",
                metadata={
                    "stage": "data_cleaning",
                    "data_path": self.data_path,
                    "execution_success": exec_result.success,
                    "execution_error": exec_result.error,
                    "timestamp": datetime.now().isoformat()
                }
            )

        return self.cleaning_code

    def execute_cleaning(self, code: str = None) -> Dict[str, Any]:
        """
        执行清洗代码

        Args:
            code: 清洗代码，为 None 时使用已生成的代码

        Returns:
            执行结果
        """
        from ..utils.code_generator import CodeGenerator
        import shutil

        cleaning_code = code or self.cleaning_code

        if not cleaning_code:
            raise ValueError("请先生成清洗代码")

        # 使用代码生成器执行代码
        code_gen = CodeGenerator()

        # 临时输出路径（在原始数据目录）
        temp_cleaned_path = self.data_path.replace('.csv', '_cleaned.csv')

        context = {
            "data_path": self.data_path,
            "cleaned_data_path": temp_cleaned_path
        }

        exec_result = code_gen.execute_code(cleaning_code, context)

        # 检查是否成功生成了清洗后的数据文件
        import os
        file_exists = os.path.exists(temp_cleaned_path)

        # 将清洗后的数据复制到 session 目录
        final_cleaned_path = None
        if file_exists:
            session_cleaned_path = self.asset_manager.session_dir / "data" / "cleaned_data.csv"
            session_cleaned_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_cleaned_path, session_cleaned_path)
            final_cleaned_path = str(session_cleaned_path)
            print(f"[Agent] 清洗后数据已复制到 session 目录: {final_cleaned_path}")
            # 删除临时文件
            os.remove(temp_cleaned_path)

        # 构建结果
        result_info = {
            "success": file_exists,  # 只要文件存在就认为成功
            "cleaned_data_path": final_cleaned_path,
            "original_path": self.data_path,
            "execution_output": exec_result.output,
            "execution_error": exec_result.error if not file_exists else None,
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
