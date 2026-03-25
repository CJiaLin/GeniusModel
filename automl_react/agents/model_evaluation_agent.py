"""
模型评估 Agent 模块

封装模型评估阶段逻辑，统一通过 agents 层对外提供能力。
"""

import json
from typing import Any, Dict, Optional

from ..core.react_agent import ReActAgent, ConfirmationRequired
from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
from ..skills_loader import get_skill_loader
from ..evaluation import ModelEvaluator


class ModelEvaluationAgent(ReActAgent):
    """模型评估 Agent。"""

    def __init__(self, llm: Any = None, session_id: str = None, verbose: bool = False):
        super().__init__(llm=llm, session_id=session_id, max_iterations=8, verbose=verbose)
        self.evaluator = ModelEvaluator(session_id=self.session_id)
        self.skill_loader = get_skill_loader()
        self.evaluation_plan: Optional[str] = None

    def _register_default_tools(self):
        """注册默认工具。"""
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())

    def get_system_prompt(self) -> str:
        """获取系统提示词。"""
        return self.config_loader.get_prompt("model_evaluation", "system_prompt")

    def _get_benchmark_skill_content(self) -> str:
        """获取模型评估阶段参考内容。"""
        benchmark = self.skill_loader.get_skill_reference(
            "ml-model-eval-benchmark-0.1.0",
            "benchmarking-guide.md"
        )
        if not benchmark:
            return ""
        return "⚠️ 注意：以下内容为评估方法参考，仅供参考方法，不可替代当前会话的真实训练产物与测试数据。\n\n" + benchmark[:2500]

    def generate_evaluation_plan(
        self,
        target_column: str,
        task_type: str,
        model_result: Optional[Dict[str, Any]] = None,
        task_description: str = "",
    ) -> str:
        """生成模型评估方案。"""
        if model_result is None:
            model_result_json = self.asset_manager.read_asset("models", "model_training_result.json")
            if not model_result_json:
                raise ValueError("未找到模型训练结果，无法生成模型评估方案")

            try:
                model_result = json.loads(model_result_json)
            except Exception as exc:
                raise ValueError(f"模型训练结果解析失败: {exc}") from exc

        training_summary = {}
        training_summary_json = self.asset_manager.read_asset("models", "training_summary.json")
        if training_summary_json:
            try:
                training_summary = json.loads(training_summary_json)
            except Exception:
                training_summary = {}

        model_path = model_result.get("model_path")
        test_split_path = model_result.get("test_split_path")
        metrics = model_result.get("metrics", {})
        selected_feature_names = model_result.get("selected_feature_names", [])

        if not model_path or not test_split_path:
            raise ValueError("模型训练结果缺少评估所需的 model_path 或 test_split_path")

        task_context = ""
        if task_description:
            task_context = f"""
## 用户建模背景和要求

{task_description}

"""

        current_evaluation_context = f"""
## 当前评估事实

- **模型文件**: {model_path}
- **测试集文件**: {test_split_path}
- **目标列**: {target_column}
- **任务类型**: {task_type}
- **训练阶段记录的最佳模型**: {training_summary.get('best_model', '未知')}
- **训练阶段记录的目标变换**: {training_summary.get('target_transform', '未记录')}
- **训练阶段回收指标**: {json.dumps(metrics, ensure_ascii=False)}
- **训练阶段回收入模特征数**: {len(selected_feature_names)}

重要：评估阶段必须基于已保存的模型文件和测试集文件，验证训练摘要中的指标是否可复现，并补充形成标准化评估结论。
"""

        prompt_template = self.config_loader.get_prompt("model_evaluation", "plan_generation")
        user_input = prompt_template.format(
            target_column=target_column,
            task_type=task_type,
            task_context=task_context,
            current_evaluation_context=current_evaluation_context,
            skill_content=self._get_benchmark_skill_content(),
        )

        result = self.run(user_input, stage="model_evaluation_plan")
        self.evaluation_plan = result.get("answer", "")
        return self.evaluation_plan

    def request_user_confirmation(self) -> None:
        """请求用户确认评估方案。"""
        if not self.evaluation_plan:
            raise ValueError("请先生成模型评估方案")

        skills_referenced = [
            {
                "name": "ml-model-eval-benchmark-0.1.0",
                "files": ["benchmarking-guide.md"]
            }
        ]

        raise ConfirmationRequired(
            stage="model_evaluation",
            proposal=self.evaluation_plan,
            skills_referenced=skills_referenced,
        )

    def evaluate_from_training_result(
        self,
        target_column: str,
        task_type: str,
        model_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """基于模型训练结果执行模型评估。"""
        if model_result is None:
            model_result_json = self.asset_manager.read_asset("models", "model_training_result.json")
            if not model_result_json:
                raise ValueError("未找到模型训练结果，无法执行模型评估")

            try:
                model_result = json.loads(model_result_json)
            except Exception as exc:
                raise ValueError(f"模型训练结果解析失败: {exc}") from exc

        model_path = model_result.get("model_path")
        test_split_path = model_result.get("test_split_path")

        if not model_path:
            raise ValueError("模型训练结果缺少 model_path")
        if not test_split_path:
            raise ValueError("模型训练结果缺少 test_split_path")

        result = self.evaluator.evaluate_model(
            model_path=model_path,
            data_path=test_split_path,
            target_column=target_column,
            task_type=task_type,
        )

        result["stage"] = "model_evaluation"
        result["test_split_path"] = test_split_path
        return result