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

    def generate_cleaning_plan(self, data_path: str = None, analysis_result: str = None, task_description: str = "") -> str:
        """
        生成数据清洗思路

        Args:
            data_path: 数据文件路径
            analysis_result: 数据分析报告（可选，如果提供则直接使用）
            task_description: 用户的建模背景和要求

        Returns:
            清洗方案（Markdown 格式）
        """
        path = data_path or self.data_path
        if not path:
            raise ValueError("请提供数据文件路径")

        self.data_path = path

        # 如果有分析报告，直接使用，不重复分析
        if analysis_result:
            data_summary = f"""
## 数据分析报告（来自上一阶段）

{analysis_result}

---
**请基于以上分析报告，直接生成数据清洗方案，不要重复分析数据质量问题。**

方案应包括：
1. 针对分析报告中识别的问题，制定具体的清洗策略
2. 每个清洗步骤的具体操作方法
3. 预期效果
"""
        else:
            # 没有分析报告，自行分析数据
            import pandas as pd
            import os

            try:
                df = pd.read_csv(path)

                # 收集数据基本信息
                data_info = {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "missing_values": df.isnull().sum().to_dict(),
                    "missing_ratio": (df.isnull().sum() / len(df) * 100).to_dict(),
                    "numeric_columns": list(df.select_dtypes(include=['int64', 'float64']).columns),
                    "categorical_columns": list(df.select_dtypes(include=['object']).columns),
                    "duplicate_rows": df.duplicated().sum(),
                    "sample_data": df.head(3).to_dict()
                }

                # 找出缺失值最多的列
                missing_sorted = sorted(
                    [(k, v) for k, v in data_info["missing_values"].items() if v > 0],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]

                # 找出数值列的统计信息
                numeric_stats = df.describe().to_dict() if data_info["numeric_columns"] else {}

                # 构建数据摘要
                data_summary = f"""
## 数据基本信息

- **文件路径**: {path}
- **数据形状**: {data_info['shape'][0]} 行 × {data_info['shape'][1]} 列
- **重复行数**: {data_info['duplicate_rows']}
- **数值列数量**: {len(data_info['numeric_columns'])}
- **分类列数量**: {len(data_info['categorical_columns'])}

## 缺失值情况 (Top 10)

"""
                for col, missing in missing_sorted:
                    ratio = data_info["missing_ratio"][col]
                    data_summary += f"- **{col}**: {missing} 个缺失 ({ratio:.1f}%)\n"

                if not missing_sorted:
                    data_summary += "- 无缺失值\n"

                data_summary += f"""
## 数值列统计

主要数值列: {', '.join(data_info['numeric_columns'][:10])}

## 分类列

主要分类列: {', '.join(data_info['categorical_columns'][:10])}

## 前3行数据预览

列名: {', '.join(data_info['columns'][:10])}...
"""

                self.data_info = data_info

            except Exception as e:
                data_summary = f"无法加载数据文件: {path}\n错误: {str(e)}"

        # 加载 data-analysis skill
        techniques = self.skill_loader.get_skill_reference("data-analysis-1.0.2", "techniques.md")
        pitfalls = self.skill_loader.get_skill_reference("data-analysis-1.0.2", "pitfalls.md")

        # 从配置加载 Prompt
        try:
            prompt_template = self.config_loader.get_prompt("data_cleaning", "plan_generation")
        except KeyError:
            if analysis_result:
                # 有分析报告时，直接基于报告生成清洗方案
                prompt_template = """你是一位数据清洗专家。以下是数据分析阶段生成的详细报告：

{task_context}{data_summary}

{skill_content}

**重要指示**：
1. 你必须基于上述分析报告中识别的数据质量问题，制定针对性的清洗方案
2. 不要重复分析数据质量问题，直接给出清洗策略
3. 清洗方案必须与上述报告中的数据特征一致（样本数、列名、缺失值情况等）
4. 如果报告显示有 PoolQC、MiscFeature 等高缺失率列，方案中必须处理这些列
5. 清洗方案应服务于用户的建模目标

请生成 Markdown 格式的清洗方案，包括：
1. 针对分析报告中识别的问题，制定具体的清洗策略
2. 每个清洗步骤的具体操作方法
3. 预期效果
"""
            else:
                # 没有分析报告时，需要先分析数据
                prompt_template = """请为以下数据生成详细的清洗方案：

{task_context}{data_summary}

{skill_content}

请生成 Markdown 格式的清洗方案，包括：
1. 数据质量问题分析
2. 针对性的清洗步骤（基于实际数据）
3. 预期效果

重要：方案必须基于上述实际数据分析结果，不要使用示例数据。
"""

        # 构建 skill 内容
        skill_content = ""
        if techniques:
            skill_content += f"## 数据清洗技术参考\n\n{techniques[:1500]}\n\n"
        if pitfalls:
            skill_content += f"## 数据陷阱参考\n\n{pitfalls[:1500]}\n\n"

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
            task_context=task_context
        )

        # 调用 LLM 生成方案
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
            data_info_text = f"""
## 数据信息

- 数据形状: {self.data_info['shape'][0]} 行 × {self.data_info['shape'][1]} 列
- 数值列: {', '.join(self.data_info['numeric_columns'][:15])}
- 分类列: {', '.join(self.data_info['categorical_columns'][:15])}
- 缺失值列: {', '.join([k for k, v in self.data_info['missing_values'].items() if v > 0][:10])}

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
