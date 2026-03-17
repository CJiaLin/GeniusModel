"""
AutoML Agent 对话式交互模块

本模块提供了AutoML系统的对话式交互能力，支持：
1. 多轮对话：与用户进行多轮交互，每轮根据上下文决定下一步
2. 步骤决策：利用LLM判断当前应该执行的步骤
3. 指令解析：解析用户的自然语言指令
4. 结果展示：向用户展示分析结果和中间结果

核心组件：
1. SessionManager - 会话管理器，管理用户会话和上下文
2. StepDecider - 步骤决策器，使用LLM判断下一步
3. InteractiveEngine - 对话式引擎，整合交互能力
4. UserInterface - 用户界面抽象

作者: AutoML Team
"""

import json
import uuid
from datetime import datetime
from typing import Any, Optional, Literal
from enum import Enum

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

# LangChain相关导入
try:
    from langchain.prompts import ChatPromptTemplate
except ImportError:
    ChatPromptTemplate = None


class StepType(str, Enum):
    """AutoML流程中的步骤类型枚举"""
    LOAD_DATA = "load_data"           # 加载数据
    EXPLORE_DATA = "explore_data"     # 探索数据
    ANALYZE_QUALITY = "analyze_quality"  # 分析质量
    CLEAN_DATA = "clean_data"         # 清洗数据
    FEATURE_ENGINEERING = "feature_engineering"  # 特征工程
    FEATURE_SELECTION = "feature_selection"  # 特征选择
    MODEL_SELECTION = "model_selection"  # 模型选择
    TRAIN_MODEL = "train_model"       # 训练模型
    EVALUATE_MODEL = "evaluate_model" # 评估模型
    GENERATE_REPORT = "generate_report"  # 生成报告
    ASK_USER = "ask_user"             # 询问用户
    COMPLETE = "complete"             # 完成


class UserIntent(str, Enum):
    """用户意图枚举"""
    CONTINUE = "continue"             # 继续下一步
    STOP = "stop"                     # 停止
    EXPLAIN = "explain"               # 解释当前结果
    MODIFY = "modify"                 # 修改参数
    SHOW_DATA = "show_data"           # 查看数据
    SHOW_CODE = "show_code"           # 查看代码
    SPECIFIC_STEP = "specific_step"  # 指定具体步骤
    ASK_QUESTION = "ask_question"    # 提问
    UNKNOWN = "unknown"               # 未知意图


class ConversationTurn(BaseModel):
    """对话轮次模型"""
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    user_input: str
    system_response: str
    step_executed: Optional[str] = None
    step_result: Optional[dict] = None


class StepContext(BaseModel):
    """当前步骤上下文"""
    current_step: str = "init"
    completed_steps: list[str] = Field(default_factory=list)
    data_info: dict[str, Any] = Field(default_factory=dict)
    quality_report: Optional[dict] = None
    feature_suggestions: list[dict] = Field(default_factory=list)
    model_results: list[dict] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)


class SessionManager:
    """
    会话管理器
    
    负责管理用户的会话状态，包括：
    - 会话ID和元数据
    - 完整的对话历史
    - 当前步骤上下文
    - 数据状态
    
    Attributes:
        session_id: 会话唯一标识
        created_at: 会话创建时间
        last_active: 最后活跃时间
        conversation_history: 对话历史
        context: 当前上下文
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """初始化会话管理器"""
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.conversation_history: list[ConversationTurn] = []
        self.context = StepContext()
    
    def add_turn(self, user_input: str, system_response: str, 
                 step: Optional[str] = None, result: Optional[dict] = None):
        """添加一轮对话"""
        turn = ConversationTurn(
            user_input=user_input,
            system_response=system_response,
            step_executed=step,
            step_result=result
        )
        self.conversation_history.append(turn)
        self.last_active = datetime.now()
        
        if step and step not in self.context.completed_steps:
            self.context.completed_steps.append(step)
    
    def get_history_summary(self) -> str:
        """获取对话历史摘要"""
        if not self.conversation_history:
            return "暂无对话历史"
        
        lines = ["对话历史:", "-" * 40]
        for i, turn in enumerate(self.conversation_history, 1):
            lines.append(f"\n【第{i}轮】")
            lines.append(f"用户: {turn.user_input}")
            lines.append(f"系统: {turn.system_response[:200]}...")
            if turn.step_executed:
                lines.append(f"执行步骤: {turn.step_executed}")
        
        return "\n".join(lines)
    
    def get_context_summary(self) -> str:
        """获取当前上下文摘要"""
        return f"""
当前步骤: {self.context.current_step}
已完成步骤: {', '.join(self.context.completed_steps) if self.context.completed_steps else '无'}
数据信息: {json.dumps(self.context.data_info, ensure_ascii=False, indent=2)}
"""


class StepDecider:
    """
    步骤决策器
    
    利用LLM的推理能力，根据当前上下文和用户输入，
    判断下一步应该执行什么操作。
    
    工作流程：
    1. 收集当前上下文信息
    2. 构建决策提示词
    3. 调用LLM进行推理
    4. 解析决策结果
    5. 返回决策结果
    
    Attributes:
        llm: 大语言模型实例
        available_steps: 可用的步骤列表
    """
    
    def __init__(self, llm: ChatOpenAI):
        """初始化步骤决策器"""
        self.llm = llm
        self.available_steps = [s.value for s in StepType]
    
    def decide(self, 
               user_input: str,
               context: StepContext,
               available_actions: Optional[list[str]] = None) -> dict[str, Any]:
        """
        根据用户输入和当前上下文决定下一步
        
        Args:
            user_input: 用户的最新输入
            context: 当前步骤上下文
            available_actions: 可用的动作列表
            
        Returns:
            dict: 包含决策结果的字典:
                - intent: 用户意图
                - next_step: 下一步应该执行的步骤
                - explanation: 决策解释
                - needs_user_confirmation: 是否需要用户确认
                - message: 返回给用户的消息
        """
        # 构建决策提示词
        prompt = self._build_decision_prompt(user_input, context, available_actions)
        
        # 调用LLM进行决策
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # 解析决策结果
        decision = self._parse_decision(content, context)
        
        return decision
    
    def _build_decision_prompt(self, 
                                user_input: str,
                                context: StepContext,
                                available_actions: Optional[list[str]] = None) -> str:
        """构建决策提示词"""
        
        # 可用动作列表
        actions = available_actions or [
            "continue: 继续执行下一步",
            "stop: 停止当前流程",
            "explain: 解释当前结果",
            "modify: 修改某个参数",
            "show_data: 展示数据或统计信息",
            "show_code: 展示生成的代码",
            "specific_step: 用户指定了具体步骤",
            "ask: 向用户提问"
        ]
        
        prompt = f"""你是一个AutoML系统的智能助手。你需要根据用户的输入和当前上下文，决定下一步应该做什么。

## 用户的最新输入
"{user_input}"

## 当前上下文信息
- 当前步骤: {context.current_step}
- 已完成的步骤: {', '.join(context.completed_steps) if context.completed_steps else '无'}
- 数据信息: {json.dumps(context.data_info, ensure_ascii=False)}
- 数据质量报告: {json.dumps(context.quality_report, ensure_ascii=False) if context.quality_report else '暂无'}
- 特征建议数量: {len(context.feature_suggestions)}
- 模型结果数量: {len(context.model_results)}

## 可用的动作
{chr(10).join(actions)}

## 你需要做的
1. 分析用户的输入，判断用户的意图
2. 根据当前上下文，决定下一步应该执行什么
3. 如果需要用户确认某个操作，给出确认请求

## 输出格式
请按以下JSON格式输出你的决策（不要有其他内容）:
{{
    "intent": "用户意图（continue/stop/explain/modify/show_data/show_code/specific_step/ask/unknown）",
    "next_step": "下一步应该执行的步骤",
    "explanation": "你的决策解释",
    "needs_user_confirmation": true或false,
    "message": "返回给用户的友好消息"
}}

注意：
- 如果用户没有明确指示，继续使用 "continue" 意图
- 如果当前步骤是 "init"，应该先加载数据
- 如果有重要结果需要用户确认，设置 needs_user_confirmation 为 true
"""
        return prompt
    
    def _parse_decision(self, response: str, context: StepContext) -> dict[str, Any]:
        """解析LLM的决策结果"""
        import re
        
        default_decision = {
            "intent": UserIntent.CONTINUE.value,
            "next_step": self._get_next_step(context.current_step),
            "explanation": "根据当前流程继续执行",
            "needs_user_confirmation": False,
            "message": "继续执行下一步..."
        }
        
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                decision = json.loads(json_str)
                
                # 验证必要字段
                if "intent" in decision and "next_step" in decision:
                    return decision
        except Exception as e:
            print(f"解析决策结果时出错: {e}")
        
        return default_decision
    
    def _get_next_step(self, current_step: str) -> str:
        """获取下一步的默认顺序"""
        step_order = [
            "init", "load_data", "explore_data", "analyze_quality",
            "clean_data", "feature_engineering", "feature_selection",
            "model_selection", "train_model", "evaluate_model", "complete"
        ]
        
        try:
            current_idx = step_order.index(current_step)
            if current_idx < len(step_order) - 1:
                return step_order[current_idx + 1]
        except ValueError:
            pass
        
        return "load_data"


class InteractiveEngine:
    """
    对话式AutoML引擎
    
    这是核心的交互式AutoML引擎，支持：
    1. 与用户进行多轮对话
    2. 根据上下文和用户输入决定下一步
    3. 执行各种AutoML步骤
    4. 展示结果并等待用户反馈
    
    Attributes:
        llm: 大语言模型实例
        session: 当前会话管理器
        decider: 步骤决策器
        data_agent: 数据处理Agent
        feature_agent: 特征工程Agent
        model_agent: 模型Agent
    """
    
    def __init__(self, llm: Optional[Any] = None):
        """初始化对话式引擎"""
        from .engine import AutoMLEngine
        
        # agents在顶层目录，需要特殊处理
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from agents.data_agent import DataAgent
        from agents.feature_engineer import FeatureEngineerAgent
        from agents.model_agent import ModelAgent
        
        # 优先使用传入的LLM，否则使用自定义客户端
        if llm is not None:
            self.llm = llm
        else:
            try:
                from llm_client import get_llm_client
                self.llm = get_llm_client()
            except ImportError:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # 初始化各个Agent
        self.data_agent = DataAgent(self.llm)
        self.feature_agent = FeatureEngineerAgent(self.llm)
        self.model_agent = ModelAgent(self.llm)
        
        # 初始化会话和决策器
        self.session = SessionManager()
        self.decider = StepDecider(self.llm)
        
        # 当前数据
        self.current_data = None
    
    def start(self, user_goal: str) -> str:
        """
        开始一个新的建模任务
        
        Args:
            user_goal: 用户的建模目标
            
        Returns:
            str: 系统欢迎消息
        """
        self.session = SessionManager()
        self.session.context.current_step = "init"
        
        welcome = f"""
🎯 **AutoML 智能建模助手**

您好！我将帮助您完成机器学习建模任务。

**您的建模目标**: {user_goal}

我将按照以下流程为您服务：
1. 📂 加载和探索数据
2. 📊 分析数据质量
3. 🧹 清洗数据
4. ⚙️ 特征工程
5. 🤖 选择和训练模型
6. 📈 评估模型

在整个过程中，您可以：
- 输入指令控制流程（如"继续"、"停止"、"解释"等）
- 随时查看当前数据和结果
- 修改参数或重新执行某个步骤
- 询问任何问题

**请问您的数据在哪里？** （请提供数据文件路径）
"""
        self.session.add_turn(user_goal, welcome, "init", {"goal": user_goal})
        return welcome
    
    def process_input(self, user_input: str) -> str:
        """
        处理用户输入并返回响应
        
        这是主要的交互方法，接收用户输入，处理后返回响应。
        
        Args:
            user_input: 用户的输入
            
        Returns:
            str: 系统响应消息
        """
        # 使用决策器判断下一步
        decision = self.decider.decide(user_input, self.session.context)
        
        intent = decision.get("intent", "continue")
        next_step = decision.get("next_step", "load_data")
        
        response = ""
        
        # 根据决策执行相应操作
        if intent == "continue" or intent == "specific_step":
            response = self._execute_step(next_step, user_input)
        elif intent == "explain":
            response = self._explain_current_step()
        elif intent == "show_data":
            response = self._show_data_info()
        elif intent == "show_code":
            response = self._show_generated_code()
        elif intent == "modify":
            response = self._handle_modify(user_input)
        elif intent == "ask":
            response = self._answer_question(user_input)
        elif intent == "stop":
            response = self._stop_process()
        else:
            response = decision.get("message", "我不明白您的意思，请重试。")
        
        # 记录对话
        self.session.add_turn(user_input, response, next_step, {"decision": decision})
        
        return response
    
    def _execute_step(self, step: str, user_input: str) -> str:
        """执行指定步骤"""
        step_handlers = {
            "load_data": self._handle_load_data,
            "explore_data": self._handle_explore_data,
            "analyze_quality": self._handle_analyze_quality,
            "clean_data": self._handle_clean_data,
            "feature_engineering": self._handle_feature_engineering,
            "model_selection": self._handle_model_selection,
            "train_model": self._handle_train_model,
            "evaluate_model": self._handle_evaluate_model,
        }
        
        handler = step_handlers.get(step)
        if handler:
            return handler(user_input)
        else:
            return f"未知步骤: {step}"
    
    def _handle_load_data(self, user_input: str) -> str:
        """处理数据加载"""
        # 尝试从用户输入中提取文件路径
        file_path = self._extract_file_path(user_input)
        
        if not file_path:
            return "请提供数据文件路径，例如：/path/to/data.csv"
        
        try:
            self.current_data = self.data_agent.load_data(file_path)
            self.session.context.data_info = {
                "shape": list(self.current_data.shape),
                "columns": self.current_data.columns.tolist(),
                "dtypes": {k: str(v) for k, v in self.current_data.dtypes.items()}
            }
            self.session.context.current_step = "load_data"
            
            return f"""
✅ **数据加载成功！**

📊 **数据概览**:
- 数据形状: {self.current_data.shape[0]} 行 × {self.current_data.shape[1]} 列
- 列名: {', '.join(self.current_data.columns.tolist()[:10])}{'...' if len(self.current_data.columns) > 10 else ''}

数据已准备好进行下一步分析。请问是否继续进行数据探索和分析？
"""
        except Exception as e:
            return f"加载数据失败: {str(e)}\n请检查文件路径是否正确。"
    
    def _handle_explore_data(self, user_input: str) -> str:
        """处理数据探索"""
        if self.current_data is None:
            return "请先加载数据。"
        
        # 获取数据统计信息
        numeric_stats = self.current_data.describe().to_dict()
        
        return f"""
📊 **数据探索结果**:

**数值列统计**:
{json.dumps(numeric_stats, indent=2, ensure_ascii=False)}

请问是否继续分析数据质量？
"""
    
    def _handle_analyze_quality(self, user_input: str) -> str:
        """处理数据质量分析"""
        if self.current_data is None:
            return "请先加载数据。"
        
        quality = self.data_agent.analyze_quality()
        self.session.context.quality_report = quality.dict() if hasattr(quality, 'dict') else {}
        
        return f"""
🔍 **数据质量分析报告**:

**缺失值**:
{json.dumps(self.session.context.quality_report.get('missing_analysis', {}), ensure_ascii=False)}

**重复记录**: {self.session.context.quality_report.get('duplicate_count', 0)}

请问是否继续清洗数据？
"""
    
    def _handle_clean_data(self, user_input: str) -> str:
        """处理数据清洗"""
        if self.current_data is None:
            return "请先加载数据。"
        
        self.current_data = self.data_agent.clean_data()
        
        return f"""
🧹 **数据清洗完成！**

清洗后的数据形状: {self.current_data.shape}

请问是否继续进行特征工程？
"""
    
    def _handle_feature_engineering(self, user_input: str) -> str:
        """处理特征工程"""
        if self.current_data is None:
            return "请先加载数据。"
        
        # 检查是否使用LLM特征生成
        use_llm = "llm" in user_input.lower() or "智能" in user_input
        
        if use_llm:
            # 使用LLM特征生成
            target = self.session.context.data_info.get("target_column", "target")
            result = self.feature_agent.generate_features_with_llm(
                target_column=target,
                task_type="classification",
                n_suggestions=10
            )
            
            self.session.context.feature_suggestions = [
                {"name": s.name, "category": s.category, "description": s.description}
                for s in result.suggestions
            ]
            
            return f"""
⚙️ **LLM智能特征工程完成！**

生成了 {len(result.new_columns)} 个新特征:
{', '.join(result.new_columns)}

**特征建议**:
{chr(10).join([f"- {s['name']}: {s['description']}" for s in self.session.context.feature_suggestions[:5]])}

请问是否继续选择模型？
"""
        else:
            # 传统方式
            self.feature_agent.set_data(self.current_data)
            self.current_data = self.feature_agent.generate_features()
            
            return f"""
⚙️ **特征工程完成！**

生成了 {len(self.current_data.columns)} 个特征。

请问是否继续选择模型？
"""
    
    def _handle_model_selection(self, user_input: str) -> str:
        """处理模型选择"""
        return """
🤖 **模型选择**

可用的模型类型:
- 分类: LogisticRegression, RandomForest, XGBoost, LightGBM
- 回归: LinearRegression, RandomForest, XGBoost, LightGBM

请问您想使用哪个模型？（或者让我自动选择最佳模型）
"""
    
    def _handle_train_model(self, user_input: str) -> str:
        """处理模型训练"""
        return "模型训练功能开发中..."
    
    def _handle_evaluate_model(self, user_input: str) -> str:
        """处理模型评估"""
        return "模型评估功能开发中..."
    
    def _explain_current_step(self) -> str:
        """解释当前步骤"""
        step_explanations = {
            "init": "这是AutoML流程的初始化阶段，我正在等待您的数据。",
            "load_data": "数据加载阶段，我正在读取您的数据文件。",
            "explore_data": "数据探索阶段，我正在分析数据的结构和分布。",
            "analyze_quality": "质量分析阶段，我正在检测缺失值、异常值等问题。",
            "clean_data": "数据清洗阶段，我正在处理数据中的问题。",
            "feature_engineering": "特征工程阶段，我正在生成新的特征。",
            "model_selection": "模型选择阶段，我正在选择适合的模型。",
            "train_model": "模型训练阶段，我正在训练模型。",
            "evaluate_model": "模型评估阶段，我正在评估模型性能。"
        }
        
        explanation = step_explanations.get(
            self.session.context.current_step, 
            "未知阶段"
        )
        
        return f"""
💡 **当前阶段说明**

{explanation}

已完成: {', '.join(self.session.context.completed_steps) if self.session.context.completed_steps else '无'}
"""
    
    def _show_data_info(self) -> str:
        """展示数据信息"""
        if self.current_data is None:
            return "暂无数据信息，请先加载数据。"
        
        return f"""
📊 **当前数据信息**:

- 形状: {self.current_data.shape}
- 列: {self.current_data.columns.tolist()}
- 前5行:
{self.current_data.head().to_string()}
"""
    
    def _show_generated_code(self) -> str:
        """展示生成的代码"""
        codes = self.feature_agent.get_generated_code()
        
        if not codes:
            return "暂无生成的代码。"
        
        return f"""
📝 **生成的特征工程代码**:

{'='*50}
{chr(10).join(f"--- 代码 {i+1} ---" + chr(10) + code for i, code in enumerate(codes))}
{'='*50}
"""
    
    def _handle_modify(self, user_input: str) -> str:
        """处理修改请求"""
        return """
✏️ **参数修改**

请告诉我您想修改什么参数？
- 数据路径
- 目标列
- 特征数量
- 模型类型
"""
    
    def _answer_question(self, user_input: str) -> str:
        """回答用户问题"""
        # 构建问答提示词
        prompt = f"""你是一个AutoML助手。请回答用户关于机器学习建模的问题。

当前上下文:
- 步骤: {self.session.context.current_step}
- 已完成: {', '.join(self.session.context.completed_steps)}

用户问题: {user_input}

请用友好的方式回答。
"""
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def _stop_process(self) -> str:
        """停止流程"""
        return """
🛑 **流程已停止**

感谢使用AutoML智能建模助手！
如果您有任何问题，欢迎随时咨询。
"""
    
    def _extract_file_path(self, user_input: str) -> Optional[str]:
        """从用户输入中提取文件路径"""
        import re
        
        # 常见模式
        patterns = [
            r'([/\w\-]+\.csv)',
            r'([/\w\-]+\.xlsx?)',
            r'([/\w\-]+\.json)',
            r'path[:\s]+([/\w\-]+)',
            r'文件[:\s]+([/\w\-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input)
            if match:
                return match.group(1)
        
        return None
    
    def get_session_summary(self) -> str:
        """获取会话摘要"""
        return self.session.get_context_summary()


class UserInterface:
    """
    用户界面抽象
    
    提供CLI和API两种交互方式。
    """
    
    @staticmethod
    def run_cli():
        """运行命令行界面"""
        from langchain_openai import ChatOpenAI
        from llm_client import configure_llm, get_llm_client, load_config_from_file
        
        file_config = load_config_from_file()
        api_key = file_config.get("api_key", "")
        
        if not api_key:
            api_key = input("请输入OpenAI API Key: ").strip()
        
        configure_llm(
            base_url=file_config.get("base_url", "https://fast.poloai.top"),
            api_key=api_key,
            model=file_config.get("model", "claude-sonnet-4-20250514-thinking")
        )
        llm = get_llm_client()
        engine = InteractiveEngine(llm)
        
        print("\n" + "="*50)
        print("  AutoML 智能建模助手")
        print("="*50 + "\n")
        
        # 获取用户目标
        user_goal = input("请描述您的建模目标: ").strip()
        
        if not user_goal:
            print("建模目标不能为空！")
            return
        
        # 开始会话
        response = engine.start(user_goal)
        print(response)
        
        # 对话循环
        while True:
            print("\n" + "-"*40)
            user_input = input("您: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit", "退出", "停止"]:
                print("\n感谢使用！再见！")
                break
            
            response = engine.process_input(user_input)
            print(f"\n🤖: {response}")
    
    @staticmethod
    def run_api():
        """运行API服务（待实现）"""
        print("API服务开发中...")
