"""
PlanExecuteMixin

为 plan → confirm → code_execute 模式提供模板方法。
DataCleaningAgent、FeatureEngineeringAgent、ModelTrainingAgent 共享此模式。
"""

from typing import Dict, List, Optional, Tuple


class PlanExecuteMixin:
    """
    模板方法 mixin，为 plan→code 两阶段编排提供统一骨架。

    子类需实现：
    - _get_plan_prompt_key() -> Tuple[str, str]   # (section, key)
    - _get_code_prompt_key() -> Tuple[str, str]
    - _build_plan_context(**kwargs) -> dict        # prompt format kwargs
    - _build_code_context(modifications) -> dict
    - _get_codeact_config() -> dict               # max_iterations, timeout 等
    - _get_required_filepath() -> str             # CodeAct 必须生成的文件路径
    - _get_output_validator() -> callable | None  # 可选：输出校验器
    - _get_deterministic_fallback() -> callable | None  # 可选：确定性兜底
    - get_modifiable_aspects() -> List[str]
    """

    # 子类设置
    plan: Optional[str] = None
    generated_code: Optional[str] = None

    def _get_plan_prompt_key(self) -> Tuple[str, str]:
        raise NotImplementedError

    def _get_code_prompt_key(self) -> Tuple[str, str]:
        raise NotImplementedError

    def _build_plan_context(self, **kwargs) -> dict:
        raise NotImplementedError

    def _build_code_context(self, modifications: Optional[str] = None) -> dict:
        raise NotImplementedError

    def _get_codeact_config(self) -> dict:
        """返回 CodeActAgent 配置。默认值适用于大多数场景。"""
        return {
            "max_iterations": 5,
            "timeout": 300,
        }

    def _get_required_filepath(self) -> str:
        raise NotImplementedError

    def _get_output_validator(self):
        """返回输出校验函数，或 None 跳过校验。"""
        return None

    def _get_deterministic_fallback(self):
        """返回确定性兜底函数，或 None 不使用兜底。"""
        return None

    def get_modifiable_aspects(self) -> List[str]:
        """返回用户可修改的方面列表。"""
        raise NotImplementedError

    def generate_plan(self, **kwargs) -> str:
        """
        模板方法：生成方案。

        1. 从 config 加载 prompt 模板
        2. 调用子类 _build_plan_context 构建上下文
        3. 格式化 prompt
        4. 调用 LLM（通过 ReActAgent.run）
        5. 保存到 self.plan
        """
        section, key = self._get_plan_prompt_key()
        prompt_template = self.config_loader.get_prompt(section, key)
        context = self._build_plan_context(**kwargs)
        user_input = prompt_template.format(**context)

        # 使用 ReActAgent.run（子类继承自 ReActAgent）
        stage_name = f"{section}_plan"
        result = self.run(user_input, stage=stage_name)
        self.plan = result
        return result

    def generate_code(self, modifications: Optional[str] = None) -> str:
        """
        模板方法：生成并执行代码。

        1. 检查 plan 已存在
        2. 从 config 加载 code prompt 模板
        3. 调用子类 _build_code_context 构建上下文
        4. 创建 CodeActAgent 并执行
        5. 保存到 self.generated_code
        """
        if not self.plan:
            raise ValueError("请先调用 generate_plan() 生成方案")

        from automl_react.utils.codeact_agent import CodeActAgent

        section, key = self._get_code_prompt_key()
        prompt_template = self.config_loader.get_prompt(section, key)
        context = self._build_code_context(modifications=modifications)
        task_prompt = prompt_template.format(**context)

        config = self._get_codeact_config()
        codeact = CodeActAgent(
            llm=self.llm,
            max_iterations=config.get("max_iterations", 5),
            timeout=config.get("timeout", 300),
            session_id=getattr(self, "session_id", "default"),
        )

        required_filepath = self._get_required_filepath()
        output_validator = self._get_output_validator()
        deterministic_fallback = self._get_deterministic_fallback()

        result = codeact.generate_and_execute(
            task_prompt=task_prompt,
            context=context,
            required_filepath=required_filepath,
            output_validator=output_validator,
            deterministic_fallback=deterministic_fallback,
        )

        self.generated_code = result.get("code", "")
        return self.generated_code

    def revise_plan(self, current_plan: str, modifications: str) -> str:
        """
        修订方案：基于用户反馈重新生成方案。

        默认实现：将修改意见附加到原方案后重新调用 generate_plan。
        子类可 override 提供更精细的修订逻辑。
        """
        section, key = self._get_plan_prompt_key()
        prompt_template = self.config_loader.get_prompt(section, key)
        context = self._build_plan_context()

        revision_prompt = (
            f"以下是当前方案：\n\n{current_plan}\n\n"
            f"用户要求的修改：\n{modifications}\n\n"
            f"请基于上述反馈生成修订后的完整方案。"
        )

        stage_name = f"{section}_plan_revision"
        result = self.run(revision_prompt, stage=stage_name)
        self.plan = result
        return result
