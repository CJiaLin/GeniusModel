"""
模型训练 Agent 模块

实现建模方案生成、用户确认、代码生成与执行
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..core.react_agent import ReActAgent, ConfirmationRequired
from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
from ..skills_loader import get_skill_loader
from ..config import get_config_loader


class ModelTrainingAgent(ReActAgent):
    """
    模型训练 Agent

    基于 ReAct 架构的模型训练 Agent，支持：
    1. 建模方案生成（参考 afrexai-ml-engineering skill）
    2. 用户确认流程
    3. 代码生成与执行
    4. 模型保存

    Attributes:
        session_id: 会话ID
        data_path: 数据文件路径
        target_column: 目标列名
        task_type: 任务类型
        model_plan: 建模方案
        model_code: 建模代码
    """

    def __init__(self, llm: Any = None, session_id: str = None, verbose: bool = False):
        super().__init__(llm=llm, session_id=session_id, max_iterations=10, verbose=verbose)
        self.data_path: Optional[str] = None
        self.target_column: Optional[str] = None
        self.task_type: str = "classification"
        self.data_info: Optional[Dict] = None
        self.model_plan: Optional[str] = None
        self.model_code: Optional[str] = None
        self.skill_loader = get_skill_loader()
        self.config_loader = get_config_loader()

    def _register_default_tools(self):
        """注册默认工具"""
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        try:
            return self.config_loader.get_prompt("model_training", "system_prompt")
        except KeyError:
            return """你是一位专业的机器学习专家。

你的职责：
1. 分析数据和任务类型
2. 推荐合适的模型和参数
3. 编写模型训练代码
4. 训练模型并评估性能

请基于数据特征和最佳实践，生成详细的建模方案。"""

    def analyze_data_for_modeling(
        self,
        data_path: str,
        target_column: str,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        分析数据以进行建模

        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型

        Returns:
            数据分析结果
        """
        self.data_path = data_path
        self.target_column = target_column
        self.task_type = task_type

        # 构建分析提示词
        user_input = f"请分析数据文件以进行建模: {data_path}, 目标列: {target_column}, 任务类型: {task_type}"

        result = self.run(user_input, stage="model_data_analysis")

        # 保存数据信息
        self.data_info = result.get("data_info", {})

        return result

    def generate_model_plan(
        self,
        data_path: str = None,
        target_column: str = None,
        task_type: str = "classification"
    ) -> str:
        """
        生成建模方案

        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型

        Returns:
            建模方案（Markdown 格式）
        """
        path = data_path or self.data_path
        target = target_column or self.target_column
        task = task_type or self.task_type

        if not path or not target:
            raise ValueError("请提供数据文件路径和目标列名")

        self.data_path = path
        self.target_column = target
        self.task_type = task

        # 首先加载并分析实际数据
        import pandas as pd

        try:
            df = pd.read_csv(path)

            # 收集数据基本信息
            self.data_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "numeric_columns": list(df.select_dtypes(include=['int64', 'float64']).columns),
                "categorical_columns": list(df.select_dtypes(include=['object']).columns),
                "target_dtype": str(df[target].dtype) if target in df.columns else "unknown",
                "target_unique": df[target].nunique() if target in df.columns else 0,
                "has_missing": df.isnull().any().any()
            }

            # 构建数据摘要
            data_summary = f"""
## 数据基本信息

- **文件路径**: {path}
- **数据形状**: {self.data_info['shape'][0]} 行 × {self.data_info['shape'][1]} 列
- **目标列**: {target}
- **目标列类型**: {self.data_info['target_dtype']}
- **目标列唯一值数**: {self.data_info['target_unique']}
- **数值列数量**: {len(self.data_info['numeric_columns'])}
- **分类列数量**: {len(self.data_info['categorical_columns'])}
- **是否有缺失值**: {self.data_info['has_missing']}

## 数值列

{', '.join(self.data_info['numeric_columns'][:20])}

## 分类列

{', '.join(self.data_info['categorical_columns'][:20])}

重要：请基于上述实际数据生成建模方案。
"""

        except Exception as e:
            data_summary = f"无法加载数据文件: {path}\n错误: {str(e)}"

        # 加载 afrexai-ml-engineering skill 的 Phase 3
        skill_content = self.skill_loader.get_skill_content("afrexai-ml-engineering-1.0.0")

        # 提取 Phase 3 相关内容
        phase3_content = ""
        if skill_content:
            import re
            phase3_match = re.search(
                r'##?\s*Phase\s*3[:：]\s*Model Selection.*?\n(.*?)(?=##?\s*Phase\s*4|\Z)',
                skill_content,
                re.DOTALL | re.IGNORECASE
            )
            if phase3_match:
                phase3_content = phase3_match.group(1)
            else:
                phase3_content = skill_content[:3000]

        # 从配置加载 Prompt
        try:
            prompt_template = self.config_loader.get_prompt("model_training", "plan_generation")
        except KeyError:
            prompt_template = """请为以下数据生成详细的建模方案：

{data_summary}

任务类型: {task_type}

{skill_content}

请生成 Markdown 格式的建模方案，包括：
1. 任务分析
2. 推荐的模型及理由
3. 模型参数设置
4. 评估指标
5. 预期效果

重要：方案必须基于上述实际数据分析结果。
"""

        user_input = prompt_template.format(
            data_path=path,
            target_column=target,
            task_type=task,
            data_summary=data_summary,
            skill_content=phase3_content
        )

        # 调用 LLM 生成方案
        result = self.run(user_input, stage="model_plan")

        self.model_plan = result.get("answer", "")

        return self.model_plan

    def request_user_confirmation(self) -> None:
        """
        请求用户确认建模方案

        Raises:
            ConfirmationRequired: 需要用户确认时抛出
        """
        if not self.model_plan:
            raise ValueError("请先生成建模方案")

        # 生成代码预览
        code_preview = self._generate_code_preview()

        # 参考的 skills
        skills_referenced = [
            {
                "name": "afrexai-ml-engineering-1.0.0",
                "files": ["SKILL.md (Phase 3: Model Selection)"]
            }
        ]

        # 抛出确认异常
        raise ConfirmationRequired(
            stage="model_training",
            proposal=self.model_plan,
            code_preview=code_preview,
            skills_referenced=skills_referenced
        )

    def _generate_code_preview(self) -> str:
        """生成代码预览"""
        if not self.model_plan:
            return "# 代码将在确认后生成"

        # 从配置加载 Prompt
        try:
            prompt_template = self.config_loader.get_prompt("model_training", "code_generation")
        except KeyError:
            prompt_template = """基于以下建模方案，生成 Python 代码预览：

{plan}

请生成简洁的代码预览（仅展示主要步骤）。
"""

        user_input = prompt_template.format(plan=self.model_plan[:1000])

        result = self.run(user_input, stage="model_code_preview")

        return result.get("answer", "# 代码预览")

    def generate_model_code(self, modifications: str = None) -> str:
        """
        生成建模代码（使用结构化输出和迭代验证）

        Args:
            modifications: 用户修改内容

        Returns:
            建模代码
        """
        if not self.model_plan:
            raise ValueError("请先生成建模方案")

        from ..utils.code_generator import CodeGenerator

        modifications_text = f"\n用户修改要求:\n{modifications}\n" if modifications else ""

        # 构建数据信息摘要
        data_info_text = ""
        if self.data_info:
            data_info_text = f"""
## 数据信息

- 数据形状: {self.data_info['shape'][0]} 行 × {self.data_info['shape'][1]} 列
- 目标列: {self.target_column}
- 任务类型: {self.task_type}
- 数值列: {', '.join(self.data_info['numeric_columns'][:15])}
- 分类列: {', '.join(self.data_info['categorical_columns'][:15])}

重要：请基于上述实际数据列名生成代码，不要使用示例数据中的列名。
"""

        prompt = f"""基于以下建模方案，生成完整的 Python 代码：

数据路径: {self.data_path}
目标列: {self.target_column}
任务类型: {self.task_type}
{data_info_text}

建模方案:
{self.model_plan}

{modifications_text}

要求：
1. 使用 scikit-learn 进行模型训练
2. 包含数据划分、模型训练、评估
3. 保存训练好的模型到: {self.data_path.replace('.csv', '_model.pkl')}
4. 返回模型性能指标
5. 包含详细的注释
6. 代码必须完整可执行，包含所有必要的导入语句
7. 必须使用上述实际数据的列名

请生成完整的、可执行的 Python 代码。
"""

        # 使用代码生成器生成并验证代码
        code_gen = CodeGenerator(llm=self.llm)

        context = {
            "data_path": self.data_path,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "model_path": self.data_path.replace('.csv', '_model.pkl')
        }

        # 生成代码并验证执行
        code, exec_result = code_gen.generate_code_with_validation(
            prompt=prompt,
            context=context,
            stage="model_code",
            required_outputs=[]
        )

        self.model_code = code

        # 保存代码到资产
        if code:
            self.asset_manager.save_code(
                code=code,
                filename="model_training.py",
                metadata={
                    "stage": "model_training",
                    "data_path": self.data_path,
                    "target_column": self.target_column,
                    "task_type": self.task_type,
                    "execution_success": exec_result.success,
                    "execution_error": exec_result.error,
                    "timestamp": datetime.now().isoformat()
                }
            )

        return self.model_code

    def execute_model_training(self, code: str = None) -> Dict[str, Any]:
        """
        执行模型训练代码

        Args:
            code: 建模代码，为 None 时使用已生成的代码

        Returns:
            执行结果
        """
        from ..utils.code_generator import CodeGenerator

        model_code = code or self.model_code

        if not model_code:
            raise ValueError("请先生成建模代码")

        # 使用代码生成器执行代码
        code_gen = CodeGenerator()

        context = {
            "data_path": self.data_path,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "model_path": self.data_path.replace('.csv', '_model.pkl')
        }

        exec_result = code_gen.execute_code(model_code, context)

        # 检查是否成功生成了模型文件
        import os
        model_path = self.data_path.replace('.csv', '_model.pkl')
        file_exists = os.path.exists(model_path)

        # 构建结果
        result_info = {
            "success": exec_result.success and file_exists,
            "model_path": model_path if file_exists else None,
            "data_path": self.data_path,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "metrics": exec_result.variables.get('metrics', {}) if exec_result.variables else {},
            "execution_output": exec_result.output,
            "execution_error": exec_result.error,
            "timestamp": datetime.now().isoformat()
        }

        # 保存结果信息到资产
        self.asset_manager.save_data(
            data=json.dumps(result_info, ensure_ascii=False, indent=2),
            filename="model_training_result.json",
            asset_type="models",
            metadata=result_info
        )

        return result_info

    def full_model_training_workflow(
        self,
        data_path: str,
        target_column: str,
        task_type: str = "classification",
        skip_confirmation: bool = False
    ) -> Dict[str, Any]:
        """
        执行完整的模型训练流程

        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型
            skip_confirmation: 是否跳过用户确认

        Returns:
            模型训练结果
        """
        self.data_path = data_path
        self.target_column = target_column
        self.task_type = task_type

        # 1. 生成建模方案
        plan = self.generate_model_plan(data_path, target_column, task_type)

        # 2. 请求用户确认（如果不跳过）
        if not skip_confirmation:
            self.request_user_confirmation()

        # 3. 生成建模代码
        code = self.generate_model_code()

        # 4. 执行模型训练
        result = self.execute_model_training(code)

        return {
            "success": result.get("success", False),
            "plan": plan,
            "code": code,
            "result": result
        }
