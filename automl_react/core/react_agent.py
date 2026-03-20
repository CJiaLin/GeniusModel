"""
ReAct Agent 核心模块

实现 ReAct (Reasoning + Acting) 循环的核心逻辑，支持用户确认点
"""

import json
import re
import asyncio
from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod
from datetime import datetime

from .memory import Memory, MemoryType
from .observation import Observation, ObservationType
from ..tools.base_tool import BaseTool, ToolResult
from ..config import ConfigLoader, get_config_loader
from ..logger import LLMLogger, get_llm_logger
from ..assets import AssetManager, get_asset_manager


class ConfirmationRequired(Exception):
    """需要用户确认时抛出的异常"""
    
    def __init__(self, stage: str, proposal: str, code_preview: str = None, skills_referenced: List[Dict] = None):
        self.stage = stage
        self.proposal = proposal
        self.code_preview = code_preview
        self.skills_referenced = skills_referenced or []
        super().__init__(f"Stage '{stage}' requires user confirmation")


class ReActAgent(ABC):
    """
    ReAct Agent 基类
    
    实现 ReAct 循环的核心逻辑：
    Observation -> Thought -> Action -> Observation
    
    支持用户确认点、配置化 Prompt、LLM 日志记录
    
    Attributes:
        llm: 语言模型实例
        memory: 记忆管理器
        tools: 可用工具字典
        max_iterations: 最大迭代次数
        verbose: 是否输出详细日志
        session_id: 会话ID
        config_loader: 配置加载器
        llm_logger: LLM 日志记录器
        asset_manager: 资产管理器
    """
    
    def __init__(
        self,
        llm: Any = None,
        session_id: str = None,
        max_iterations: int = None,
        verbose: bool = False
    ):
        self.llm = llm
        self.session_id = session_id or "default"
        self.memory = Memory()
        self.tools: Dict[str, BaseTool] = {}
        self.verbose = verbose
        self._current_iteration = 0
        self._paused_for_confirmation = False
        self._confirmation_callback: Optional[Callable] = None
        
        # 初始化配置加载器
        self.config_loader = get_config_loader()
        
        # 从配置加载参数
        workflow_config = self.config_loader.get_workflow_config("react_agent") or {}
        self.max_iterations = max_iterations or workflow_config.get("max_iterations", 15)
        
        # 初始化日志记录器
        self.llm_logger = get_llm_logger(session_id=self.session_id)
        
        # 初始化资产管理器
        self.asset_manager = get_asset_manager(session_id=self.session_id)
        
        # 注册默认工具
        self._register_default_tools()
    
    def register_tool(self, name: str, tool: BaseTool):
        """注册工具"""
        self.tools[name] = tool
        if self.verbose:
            print(f"[ReActAgent] 注册工具: {name}")
    
    def register_tools(self, tools: Dict[str, BaseTool]):
        """批量注册工具"""
        for name, tool in tools.items():
            self.register_tool(name, tool)
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取所有工具的 schema"""
        return [tool.get_schema() for tool in self.tools.values()]
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """获取系统提示词（子类必须实现）"""
        pass
    
    def _get_prompt_from_config(self, section: str, key: str, **kwargs) -> str:
        """
        从配置获取 Prompt
        
        Args:
            section: 配置节
            key: 配置键
            **kwargs: 格式化参数
            
        Returns:
            Prompt 字符串
        """
        try:
            return self.config_loader.format_prompt(section, key, **kwargs)
        except (KeyError, ValueError) as e:
            if self.verbose:
                print(f"[ReActAgent] 从配置加载 Prompt 失败: {e}")
            # 返回空字符串，让子类处理
            return ""
    
    def _build_react_prompt(self, user_input: str, observation: str = "") -> str:
        """
        构建 ReAct 提示词
        
        包含：
        1. 系统提示词
        2. 可用工具
        3. 记忆上下文
        4. 当前观察
        5. ReAct 格式说明
        """
        # 系统提示词
        system_prompt = self.get_system_prompt()
        
        # 工具描述
        tool_descriptions = []
        for name, tool in self.tools.items():
            tool_descriptions.append(
                f"工具: {name}\n"
                f"  描述: {tool.description}\n"
                f"  参数: {json.dumps(tool.parameters, ensure_ascii=False)}"
            )
        tools_text = "\n\n".join(tool_descriptions) if tool_descriptions else "无可用工具"
        
        # 记忆上下文
        memory_context = self.memory.get_short_term_context()
        
        # 如果有观察结果，说明工具已执行，需要直接输出最终答案
        if observation and len(observation.strip()) > 0:
            # 工具执行后的提示词 - 要求直接输出最终答案
            prompt = f"""{system_prompt}

## 可用工具

{tools_text}

## 执行历史

{memory_context}

## 当前任务

用户输入: {user_input}

观察结果:
{observation}

基于以上观察结果，请直接输出最终答案。
你必须使用以下格式：

思考: 基于观察结果进行总结
最终答案: 你的完整回答

请输出最终答案：
"""
        else:
            # 初始提示词 - 使用完整 ReAct 格式
            prompt = f"""{system_prompt}

## 可用工具

{tools_text}

## ReAct 格式说明

你必须按照以下格式进行思考和行动：

思考: 分析当前情况，决定下一步行动
行动: 工具名称
行动输入: {{"参数名": "参数值"}}
观察: 等待工具执行结果

当任务完成时，输出：
思考: 任务已完成
最终答案: 你的回答

## 执行历史

{memory_context}

## 当前任务

用户输入: {user_input}

请按照 ReAct 格式进行思考和行动：
"""
        return prompt
    
    def _parse_thought(self, text: str) -> Optional[str]:
        """解析思考过程"""
        match = re.search(r'思考[:：]\s*(.+?)(?=\n(?:行动|最终答案)|$)', text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _parse_action(self, text: str) -> Optional[tuple]:
        """解析行动，返回 (工具名, 参数)"""
        # 匹配行动和行动输入
        action_match = re.search(r'行动[:：]\s*(\w+)', text)
        if not action_match:
            return None
        
        tool_name = action_match.group(1).strip()
        
        # 匹配行动输入（JSON格式）
        input_match = re.search(r'行动输入[:：]\s*(\{.*\})', text, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                action_input = {}
        else:
            action_input = {}
        
        return tool_name, action_input
    
    def _parse_final_answer(self, text: str) -> Optional[str]:
        """解析最终答案
        
        支持多种格式：
        1. ReAct 格式: "最终答案: xxx" 或 "Final Answer: xxx"
        2. 直接返回: 如果没有特定格式标记且不包含 ReAct 中间步骤，返回整个文本
        """
        # 尝试匹配 ReAct 格式
        match = re.search(r'(?:最终答案|Final Answer)[:：]\s*(.+)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 检查是否包含 ReAct 中间步骤标记
        # 如果包含这些标记，说明是中间状态，不是最终答案
        react_markers = ['思考:', '思考：', 'Thought:', '行动:', '行动：', 'Action:', '行动输入:', '行动输入：', 'Action Input:']
        for marker in react_markers:
            if marker in text:
                return None
        
        # 如果没有匹配到格式标记，但文本非空且不包含 ReAct 标记，返回整个文本
        # 这适用于直接生成内容的场景（如数据清洗方案）
        if text and len(text.strip()) > 10:
            return text.strip()
        
        return None
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Observation:
        """执行工具"""
        if tool_name not in self.tools:
            return Observation.from_error(
                f"工具 '{tool_name}' 不存在",
                error_type="tool_not_found"
            )
        
        tool = self.tools[tool_name]
        
        try:
            result = tool.execute(**tool_input)
            
            if result.status.value == "success":
                return Observation.from_tool_result(
                    tool_name=tool_name,
                    result=result.data,
                    success=True
                )
            else:
                return Observation.from_error(
                    result.error or "未知错误",
                    error_type="tool_execution"
                )
        except Exception as e:
            return Observation.from_error(
                str(e),
                error_type="tool_exception"
            )
    
    def _call_llm(self, prompt: str, stage: str = "") -> Any:
        """
        调用 LLM 并记录日志（使用流式输出）
        
        Args:
            prompt: 提示词
            stage: 工作流阶段
            
        Returns:
            LLM 响应
        """
        if self.llm is None:
            raise ValueError("LLM 未设置")
        
        start_time = datetime.now()
        
        # 使用流式输出
        full_response = ""
        print(f"[LLM] 开始流式输出 (stage: {stage})...")
        
        try:
            for chunk in self.llm.stream(prompt):
                if chunk.content:
                    content = chunk.content
                    full_response += content
                    # 实时打印输出
                    print(content, end="", flush=True)
        except Exception as e:
            # 如果流式输出失败，回退到同步调用
            print(f"\n[LLM] 流式输出失败，回退到同步调用: {e}")
            response = self.llm.invoke(prompt)
            full_response = response.content if hasattr(response, 'content') else str(response)
        
        print()  # 换行
        
        # 获取模型配置
        llm_config = self.config_loader.get_llm_config()
        model_name = llm_config.get("model_name", "unknown")
        provider = llm_config.get("provider", "unknown")
        
        # 记录日志
        self.llm_logger.log_call(
            model_name=model_name,
            provider=provider,
            input_content=prompt,
            output_content=full_response,
            stage=stage
        )
        
        # 返回与 invoke 相同格式的响应对象
        from langchain_core.messages import AIMessage
        return AIMessage(content=full_response)
    
    def _check_confirmation_required(self, stage: str) -> bool:
        """
        检查当前阶段是否需要用户确认
        
        Args:
            stage: 工作流阶段
            
        Returns:
            是否需要确认
        """
        try:
            stage_config = self.config_loader.get_workflow_config(f"stages.{stage}")
            if stage_config:
                return stage_config.get("require_confirmation", False)
        except KeyError:
            pass
        return False
    
    def _generate_proposal(self, stage: str, context: Dict[str, Any]) -> str:
        """
        生成方案（需要子类实现）
        
        Args:
            stage: 工作流阶段
            context: 上下文数据
            
        Returns:
            方案内容（Markdown 格式）
        """
        # 默认实现，子类应覆盖此方法
        return f"## {stage} 方案\n\n请确认此方案。"
    
    def _generate_code_preview(self, stage: str, proposal: str) -> str:
        """
        生成代码预览（需要子类实现）
        
        Args:
            stage: 工作流阶段
            proposal: 方案内容
            
        Returns:
            代码预览
        """
        # 默认实现，子类应覆盖此方法
        return "# 代码预览\n# 将在确认后生成"
    
    def request_confirmation(
        self,
        stage: str,
        proposal: str,
        code_preview: str = None,
        skills_referenced: List[Dict] = None
    ):
        """
        请求用户确认
        
        Args:
            stage: 工作流阶段
            proposal: 方案内容（Markdown 格式）
            code_preview: 代码预览
            skills_referenced: 参考的 Skills 列表
            
        Raises:
            ConfirmationRequired: 需要用户确认时抛出
        """
        self._paused_for_confirmation = True
        
        # 保存确认点信息到资产
        confirmation_data = {
            "stage": stage,
            "proposal": proposal,
            "code_preview": code_preview,
            "skills_referenced": skills_referenced,
            "timestamp": datetime.now().isoformat()
        }
        self.asset_manager.save_data(
            json.dumps(confirmation_data, ensure_ascii=False, indent=2),
            f"confirmation_{stage}.json",
            "code"
        )
        
        raise ConfirmationRequired(stage, proposal, code_preview, skills_referenced)
    
    def resume_after_confirmation(
        self,
        confirmed: bool = True,
        modifications: str = None
    ) -> Dict[str, Any]:
        """
        用户确认后恢复执行
        
        Args:
            confirmed: 用户是否确认
            modifications: 用户修改内容
            
        Returns:
            执行结果
        """
        self._paused_for_confirmation = False
        
        return {
            "success": confirmed,
            "modifications": modifications,
            "message": "已恢复执行" if confirmed else "用户取消"
        }
    
    def run(
        self,
        user_input: str,
        context: Dict[str, Any] = None,
        stage: str = ""
    ) -> Dict[str, Any]:
        """
        运行 ReAct 循环
        
        Args:
            user_input: 用户输入
            context: 额外上下文
            stage: 当前工作流阶段
            
        Returns:
            执行结果
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[ReActAgent] 开始执行: {user_input[:50]}...")
            print(f"{'='*60}\n")
        
        # 添加用户输入到记忆
        self.memory.add_user_message(user_input)
        
        # 初始化观察
        observation = ""
        
        for iteration in range(self.max_iterations):
            self._current_iteration = iteration + 1
            
            if self.verbose:
                print(f"\n--- 迭代 {iteration + 1}/{self.max_iterations} ---")
            
            # 构建提示词
            prompt = self._build_react_prompt(user_input, observation)
            
            # 调用 LLM（带日志记录）
            try:
                response = self._call_llm(prompt, stage=stage)
            except Exception as e:
                return {
                    "success": False,
                    "answer": f"LLM 调用失败: {str(e)}",
                    "iterations": iteration + 1
                }
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            if self.verbose:
                print(f"LLM 响应:\n{response_text}\n")
            
            # 解析思考
            thought = self._parse_thought(response_text)
            if thought:
                self.memory.add_thought(thought, step=iteration + 1)
                if self.verbose:
                    print(f"思考: {thought}")
            
            # 检查是否完成
            final_answer = self._parse_final_answer(response_text)
            if final_answer:
                self.memory.add_assistant_message(final_answer)
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"[ReActAgent] 任务完成")
                    print(f"{'='*60}\n")
                
                return {
                    "success": True,
                    "answer": final_answer,
                    "iterations": iteration + 1,
                    "memory": self.memory.to_messages()
                }
            
            # 解析行动
            action = self._parse_action(response_text)
            if action:
                tool_name, tool_input = action
                self.memory.add_action(f"使用工具: {tool_name}", tool_input)
                
                if self.verbose:
                    print(f"行动: {tool_name}")
                    print(f"行动输入: {tool_input}")
                
                # 执行工具
                obs = self._execute_tool(tool_name, tool_input)
                observation = obs.to_prompt_text()
                self.memory.add_observation(observation)
                
                if self.verbose:
                    print(f"观察: {observation[:200]}...")
            else:
                # 没有行动也没有最终答案，可能是格式问题
                observation = "请按照 ReAct 格式输出：思考 -> 行动/最终答案"
        
        # 达到最大迭代次数
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[ReActAgent] 达到最大迭代次数")
            print(f"{'='*60}\n")
        
        return {
            "success": False,
            "answer": "达到最大迭代次数，任务未完成",
            "iterations": self.max_iterations,
            "memory": self.memory.to_messages()
        }
    
    def _register_default_tools(self):
        """注册默认工具（子类可覆盖）"""
        pass
    
    def reset(self):
        """重置 Agent 状态"""
        self.memory.clear()
        self._current_iteration = 0
        self._paused_for_confirmation = False
