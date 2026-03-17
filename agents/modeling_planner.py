"""
总控Agent（建模计划器）模块

本模块是AutoML系统的核心协调组件，负责理解用户需求、制定建模计划。
它是整个建模流程的入口和调度中心。

主要组件：
1. IntentParser - 用户意图解析器
2. RequirementAnalyzer - 需求分析器
3. PlanGenerator - 建模计划生成器
4. ModelingPlanner - 总控Agent主类

工作流程：
1. 解析用户输入，提取建模目标
2. 分析数据需求和质量要求
3. 生成完整的建模计划
"""

from typing import Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 修复导入路径
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from automl_agent.models import ModelingGoal, ModelingPlan, ProcessState
from automl_agent.enums import ModelingTaskType, EvaluationMetric
from automl_agent.core.protocol import AgentMessage, create_message, create_response


class IntentParser:
    """
    用户意图解析器
    
    使用大语言模型解析用户的自然语言输入，提取建模目标和相关信息。
    将非结构化的用户描述转换为结构化的ModelingGoal对象。
    
    Attributes:
        llm: 大语言模型实例，用于生成解析结果
        
    Example:
        >>> parser = IntentParser(llm)
        >>> goal = parser.parse("我想预测用户是否会购买我们的产品")
        >>> print(goal.task_type)
        classification
    """
    
    def __init__(self, llm: ChatOpenAI):
        """
        初始化意图解析器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm
    
    def parse(self, user_input: str, data_description: str = "") -> ModelingGoal:
        """
        解析用户输入
        
        将用户的自然语言描述解析为结构化的建模目标对象。
        
        Args:
            user_input: 用户的自然语言输入
            data_description: 数据描述文件的内容（可选）
            
        Returns:
            ModelingGoal: 解析后的建模目标对象
            
        Example:
            >>> goal = parser.parse("预测股票价格走势")
            >>> goal.task_type
            regression
        """
        # 构建提示模板，引导LLM提取建模意图
        data_info = f"\n\n## 数据描述信息\n{data_description}" if data_description else ""
        
        prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template("""
你是一个建模意图分析专家。根据用户的描述，提取建模目标信息。

用户输入: {user_input}
{data_info}

请分析并返回以下信息（JSON格式）：
1. task_type: 任务类型，可以是 classification（分类）, regression（回归）, clustering（聚类）, time_series（时间序列）, anomaly_detection（异常检测）
   - 注意：
     * 如果用户说"预测涨跌"、"预测价格"、"预测收益"、"预测走势"、"预测未来"等，应该是回归(regression)
     * 如果用户说"预测是否"、"判断会不会"、"分类为"、"判断是"等，应该是分类(classification)
     * 如果用户说"预测...的涨跌"，这是回归问题，预测的是具体数值
2. target_column: 目标变量/标签列的名称（如果有的话）
3. description: 简短的建模描述
4. evaluation_metrics: 评估指标列表，可以是 accuracy, precision, recall, f1, auc, rmse, mae, r2, silhouette
   - 分类用: accuracy, precision, recall, f1, auc
   - 回归用: rmse, mae, r2

请直接返回JSON，不要其他内容。
""")
        ])
        
        # 格式化 prompt
        formatted_prompt = prompt.format(user_input=user_input, data_info=data_description)
        
        import json
        import re
        
        # 直接调用 LLM 的 invoke 方法，避免 LCEL 链的兼容性问题
        response = self.llm.invoke(formatted_prompt)
        
        # 处理不同的响应类型（兼容不同版本的 LangChain 和不同实现）
        if hasattr(response, 'content'):
            # ChatCompletionMessage 或 AIMessage 对象
            response_content = response.content
        elif isinstance(response, str):
            # 直接返回字符串
            response_content = response
        else:
            # 尝试转换为字符串
            response_content = str(response)
        
        # 从 LLM 响应中提取 JSON 数据
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            
            # 解析任务类型和评估指标
            task_type = ModelingTaskType(data.get("task_type", "classification"))
            metrics = [EvaluationMetric(m) for m in data.get("evaluation_metrics", ["accuracy"])]
            
            return ModelingGoal(
                task_type=task_type,
                target_column=data.get("target_column", ""),
                description=data.get("description", ""),
                evaluation_metrics=metrics
            )
        
        # 如果解析失败，返回默认值
        return ModelingGoal(
            task_type=ModelingTaskType.CLASSIFICATION,
            target_column="target",
            description=user_input,
            evaluation_metrics=[EvaluationMetric.ACCURACY]
        )


class RequirementAnalyzer:
    """
    需求分析器
    
    分析建模目标和数据概况，确定数据质量和处理要求。
    为后续的建模计划提供数据相关的约束条件。
    
    Attributes:
        llm: 大语言模型实例，用于生成分析结果
    """
    
    def __init__(self, llm: ChatOpenAI):
        """
        初始化需求分析器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm

    def analyze(self, goal: ModelingGoal, data_profile: dict[str, Any]) -> dict[str, Any]:
        """
        分析数据需求
        
        根据建模目标和数据概况，分析并返回数据需求和约束条件。
        
        Args:
            goal: 建模目标对象
            data_profile: 数据画像/概况信息
            
        Returns:
            dict: 包含数据需求、质量阈值和特殊考虑因素的字典
            
        Example:
            >>> requirements = analyzer.analyze(goal, {"shape": (1000, 20)})
            >>> requirements["data_requirements"]
            {"min_samples": 100}
        """
        # 构建提示模板
        prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template("""
你是数据分析专家。根据建模目标和数据信息，分析数据需求。

建模目标: {goal}
数据概况: {data_profile}

请分析并返回以下信息（JSON格式）：
1. data_requirements: 对数据的要求（如需要的特征类型、最小样本数等）
2. quality_thresholds: 数据质量阈值（如缺失率容忍度、异常值处理策略等）
3. special_considerations: 特殊考虑因素

请直接返回JSON，不要其他内容。
""")
        ])
        
        # 处理 goal 参数 - 可能是字符串或 Pydantic 模型
        try:
            goal_dict = goal.model_dump()
        except AttributeError:
            # goal 是字符串，没有 model_dump 方法
            goal_dict = {"description": str(goal), "task_type": "regression", "target_column": ""}
        
        # 格式化 prompt，避免 LCEL 链的兼容性问题
        formatted_prompt = prompt.format(
            goal=goal_dict,
            data_profile=data_profile
        )
        
        # 直接调用 LLM，避免 LCEL 链的兼容性问题
        response = self.llm.invoke(formatted_prompt)
        
        import json
        import re
        
        # 处理不同的响应类型
        if hasattr(response, 'content'):
            response_content = response.content
        elif isinstance(response, str):
            response_content = response
        else:
            response_content = str(response)
        
        # 提取 JSON 响应
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        
        # 返回默认值
        return {"data_requirements": {}, "quality_thresholds": {}, "special_considerations": []}


class PlanGenerator:
    """
    建模计划生成器
    
    根据建模目标、数据概况和需求分析结果，生成完整的建模计划。
    包括特征工程步骤、推荐模型列表和预估复杂度。
    
    Attributes:
        llm: 大语言模型实例，用于生成计划
    """
    
    def __init__(self, llm: ChatOpenAI):
        """
        初始化计划生成器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm

    def generate(self, goal: ModelingGoal, data_profile: dict[str, Any], requirements: dict[str, Any]) -> ModelingPlan:
        """
        生成建模计划
        
        创建完整的建模计划，包括目标、步骤和推荐模型。
        
        Args:
            goal: 建模目标对象
            data_profile: 数据概况信息
            requirements: 数据需求分析结果
            
        Returns:
            ModelingPlan: 完整的建模计划对象
            
        Example:
            >>> plan = generator.generate(goal, data_profile, requirements)
            >>> plan.suggested_models
            ['RandomForest', 'XGBoost', 'LightGBM']
        """
        # 构建提示模板
        prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template("""
你是建模计划制定专家。根据建模目标、数据概况和需求，制定详细的建模计划。

建模目标: {goal}
数据概况: {data_profile}
需求分析: {requirements}

请制定建模计划（JSON格式）：
1. goal: 完整的建模目标对象
2. required_data_quality: 所需的数据质量
3. feature_engineering_steps: 特征工程步骤列表
4. suggested_models: 推荐的模型列表（基于任务类型）
5. estimated_complexity: 预估复杂度（low/medium/high）

请直接返回JSON，不要其他内容。
""")
        ])
        
        # 处理 goal 参数
        try:
            goal_dict = goal.model_dump()
        except AttributeError:
            goal_dict = {"description": str(goal), "task_type": "regression", "target_column": ""}
        
        # 格式化 prompt，避免 LCEL 链的兼容性问题
        formatted_prompt = prompt.format(
            goal=json.dumps(goal_dict),
            data_profile=str(data_profile),
            requirements=str(requirements)
        )
        
        # 直接调用 LLM，避免 LCEL 链的兼容性问题
        response = self.llm.invoke(formatted_prompt)
        
        import json
        import re
        
        # 处理不同的响应类型
        if hasattr(response, 'content'):
            response_content = response.content
        elif isinstance(response, str):
            response_content = response
        else:
            response_content = str(response)
        
        # 提取 JSON 响应并创建 ModelingPlan 对象
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return ModelingPlan(
                goal=goal,
                required_data_quality=data.get("required_data_quality", {}),
                feature_engineering_steps=data.get("feature_engineering_steps", []),
                suggested_models=data.get("suggested_models", []),
                estimated_complexity=data.get("estimated_complexity", "medium")
            )
        
        # 返回默认计划
        return ModelingPlan(
            goal=goal,
            required_data_quality={},
            feature_engineering_steps=["数据清洗", "特征工程", "模型训练"],
            suggested_models=["RandomForest", "XGBoost"],
            estimated_complexity="medium"
        )


class ModelingPlanner:
    """
    总控Agent（建模计划器）
    
    这是整个AutoML系统的核心协调组件，负责：
    1. 理解用户建模需求
    2. 分析数据需求和质量要求
    3. 制定完整的建模计划
    4. 协调各子Agent的工作
    
    Attributes:
        llm: 大语言模型实例
        intent_parser: 意图解析器
        requirement_analyzer: 需求分析器
        plan_generator: 计划生成器
        state: 流程状态跟踪
        
    Example:
        >>> planner = ModelingPlanner()
        >>> plan = planner.run("预测用户购买意愿", "data.csv")
        >>> print(plan.suggested_models)
        ['RandomForest', 'XGBoost', 'LightGBM']
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化总控Agent
        
        Args:
            llm: 大语言模型实例，如果为None则使用默认的GPT-4
        """
        # 使用提供的LLM或创建默认的GPT-4实例
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        
        # 初始化各个子组件
        self.intent_parser = IntentParser(self.llm)           # 意图解析
        self.requirement_analyzer = RequirementAnalyzer(self.llm)  # 需求分析
        self.plan_generator = PlanGenerator(self.llm)          # 计划生成
        self.state = ProcessState()  # 流程状态跟踪

    def parse_intent(self, user_input: str, data_description: str = "") -> ModelingGoal:
        """
        解析用户意图
        
        Args:
            user_input: 用户的自然语言描述
            data_description: 数据描述文件的内容（可选）
            
        Returns:
            ModelingGoal: 建模目标对象
        """
        return self.intent_parser.parse(user_input, data_description)

    def analyze_requirements(self, goal: ModelingGoal, data_profile: dict[str, Any]) -> dict[str, Any]:
        """
        分析数据需求
        
        Args:
            goal: 建模目标对象
            data_profile: 数据概况
            
        Returns:
            dict: 需求分析结果
        """
        return self.requirement_analyzer.analyze(goal, data_profile)

    def create_plan(self, goal: ModelingGoal, data_profile: dict[str, Any], requirements: dict[str, Any]) -> ModelingPlan:
        """
        创建建模计划
        
        Args:
            goal: 建模目标
            data_profile: 数据概况
            requirements: 需求分析结果
            
        Returns:
            ModelingPlan: 建模计划
        """
        return self.plan_generator.generate(goal, data_profile, requirements)

    def run(self, user_goal: str, data_path: str, data_profile: Optional[dict[str, Any]] = None) -> ModelingPlan:
        """
        运行总控Agent，执行完整的建模计划生成流程
        
        这是主要入口方法，依次执行：
        1. 解析建模意图
        2. 分析数据需求
        3. 生成建模计划
        
        Args:
            user_goal: 用户的建模目标描述
            data_path: 数据文件路径
            data_profile: 可选的数据概况信息
            
        Returns:
            ModelingPlan: 生成的建模计划
            
        Example:
            >>> planner = ModelingPlanner()
            >>> plan = planner.run(
            ...     user_goal="预测用户是否会购买产品",
            ...     data_path="data/sales.csv"
            ... )
        """
        # 更新状态：开始运行
        self.state.status = "running"
        self.state.current_step = "解析建模意图"
        
        # 步骤1: 解析用户意图
        goal = self.parse_intent(user_goal)
        
        # 步骤2: 分析数据需求
        self.state.current_step = "分析数据需求"
        requirements = self.analyze_requirements(goal, data_profile or {})
        
        # 步骤3: 生成建模计划
        self.state.current_step = "生成建模计划"
        plan = self.create_plan(goal, data_profile or {}, requirements)
        
        # 更新状态：完成
        self.state.status = "completed"
        self.state.progress = 1.0
        
        return plan
