"""
自主规划器 - 根据任务目标自主拆解任务并规划执行流程
"""
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from core.state import PipelineState, StateStep
from tools import get_registry
import json


class PlannerAgent(BaseAgent):
    """自主规划器 - 分析任务并制定执行计划"""
    
    def __init__(self, llm=None, name="PlannerAgent", verbose=False):
        super().__init__(llm, name, verbose)
        self.registry = get_registry()
        
    def plan_task(self, goal: str, state: PipelineState) -> Dict[str, Any]:
        """
        根据目标制定执行计划
        
        Args:
            goal: 用户建模目标
            state: 当前状态
            
        Returns:
            执行计划字典
        """
        # 获取可用工具信息
        all_tools = self.registry.to_dict()
        
        # 分析任务
        analysis_prompt = f"""
你是一位专业的 AutoML 专家，需要分析用户的目标并制定执行计划。

用户目标：{goal}

数据信息：
- 形状：{state.data.shape if state.data is not None else '未知'}
- 特征：{state.features if state.features else '未处理'}
- 目标列：{state.target_column}

可用工具类别：
- 数据处理工具：{self.registry.list_tools(category="data")}
- 特征工程工具：{self.registry.list_tools(category="feature")}
- 模型训练工具：{self.registry.list_tools(category="model")}
- 评估工具：{self.registry.list_tools(category="eval")}

请分析：
1. 任务类型（分类/回归）
2. 主要挑战
3. 建议的执行流程
4. 需要用到的关键工具

以 JSON 格式返回分析结果：
{{
    "task_type": "classification|regression|clustering",
    "challenges": ["challenge1", "challenge2"],
    "suggested_flow": ["step1", "step2", "step3"],
    "required_tools": ["tool1", "tool2"]
}}
"""
        
        analysis_response = self.invoke_llm(analysis_prompt)
        
        try:
            analysis = json.loads(analysis_response)
        except json.JSONDecodeError:
            # 如果 JSON 解析失败，尝试从文本中提取信息
            analysis = self._extract_analysis_from_text(analysis_response)
        
        # 制定详细计划
        plan_prompt = f"""
根据以下分析结果，制定详细的执行计划：

分析结果：
{json.dumps(analysis, ensure_ascii=False, indent=2)}

请制定具体的执行计划，包含每个步骤的详细信息：
- 步骤名称
- 执行工具
- 参数
- 依赖关系
- 预期输出

以 JSON 格式返回执行计划：
{{
    "goal": "{goal}",
    "task_type": "{analysis.get('task_type', 'unknown')}",
    "steps": [
        {{
            "name": "step_name",
            "description": "step description",
            "tool": "tool_name",
            "parameters": {{"param1": "value1"}},
            "dependencies": ["previous_step_name"],
            "category": "data|feature|model|eval"
        }}
    ]
}}
"""
        
        plan_response = self.invoke_llm(plan_prompt)
        
        try:
            plan = json.loads(plan_response)
        except json.JSONDecodeError:
            plan = self._extract_plan_from_text(plan_response)
        
        self.log(f"制定执行计划完成，共 {len(plan.get('steps', []))} 个步骤")
        return plan
    
    def _extract_analysis_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取分析结果"""
        # 简单的文本解析逻辑
        lines = text.split('\n')
        analysis = {
            "task_type": "unknown",
            "challenges": [],
            "suggested_flow": [],
            "required_tools": []
        }
        
        for line in lines:
            if "任务类型" in line or "task_type" in line:
                if "分类" in line or "classification" in line:
                    analysis["task_type"] = "classification"
                elif "回归" in line or "regression" in line:
                    analysis["task_type"] = "regression"
                elif "聚类" in line or "clustering" in line:
                    analysis["task_type"] = "clustering"
            
            if "挑战" in line or "challenge" in line:
                analysis["challenges"].append(line.strip())
        
        return analysis
    
    def _extract_plan_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取执行计划"""
        # 简单的文本解析逻辑
        return {
            "goal": "Unknown Goal",
            "task_type": "unknown",
            "steps": []
        }
    
    def execute(self, goal: str, state: PipelineState) -> Dict[str, Any]:
        """
        执行规划任务
        
        Args:
            goal: 用户目标
            state: 当前状态
            
        Returns:
            规划结果
        """
        self.log(f"开始规划任务：{goal}")
        plan = self.plan_task(goal, state)
        
        # 验证计划的有效性
        if not self._validate_plan(plan):
            raise ValueError("生成的执行计划无效")
        
        self.log(f"执行计划制定完成，共 {len(plan.get('steps', []))} 个步骤")
        return plan
    
    def _validate_plan(self, plan: Dict[str, Any]) -> bool:
        """验证计划的有效性"""
        if not isinstance(plan, dict):
            return False
        
        if "steps" not in plan:
            return False
        
        steps = plan["steps"]
        if not isinstance(steps, list):
            return False
        
        # 验证每个步骤的基本结构
        for step in steps:
            required_fields = ["name", "tool", "parameters"]
            for field in required_fields:
                if field not in step:
                    return False
        
        return True
    
    def get_available_tools_by_category(self) -> Dict[str, List[str]]:
        """获取按类别分组的可用工具"""
        categories = ["data", "feature", "model", "eval"]
        tools_by_category = {}
        
        for category in categories:
            tools_by_category[category] = self.registry.list_tools(category=category)
        
        return tools_by_category


class TaskDecomposer:
    """任务分解器 - 将复杂任务分解为子任务"""
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def decompose_task(self, goal: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        将复杂任务分解为子任务
        
        Args:
            goal: 原始目标
            context: 上下文信息
            
        Returns:
            子任务列表
        """
        context_str = json.dumps(context, ensure_ascii=False) if context else "{}"
        
        prompt = f"""
将以下复杂的 AutoML 任务分解为更小的子任务：

原始任务：{goal}

上下文信息：{context_str}

请将任务分解为以下类型的子任务：
- 数据准备（加载、清洗、探索）
- 特征工程（创建、选择、转换）
- 模型训练（选择、训练、调优）
- 模型评估（验证、测试）

以 JSON 格式返回分解结果：
{{
    "original_goal": "{goal}",
    "sub_tasks": [
        {{
            "name": "子任务名称",
            "description": "子任务描述",
            "category": "data|feature|model|eval",
            "priority": "high|medium|low",
            "estimated_duration": "minutes"
        }}
    ]
}}
"""
        
        response = self.llm.invoke(prompt) if self.llm else f"Sub-tasks for: {goal}"
        
        # 处理不同的响应类型（兼容不同版本的 LangChain）
        if hasattr(response, 'content'):
            response_content = response.content
        elif isinstance(response, str):
            response_content = response
        else:
            response_content = str(response)
        
        try:
            result = json.loads(response_content)
            return result.get("sub_tasks", [])
        except json.JSONDecodeError:
            # 如果解析失败，返回基本结构
            return [{"name": "default_task", "description": goal, "category": "data", "priority": "medium"}]