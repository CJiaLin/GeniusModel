# -*- coding: utf-8 -*-
"""
重构验证测试

验证 6 项重构（Pydantic 工具定义、Multi-LLM Provider、中间件链、
上下文摘要、沙盒执行增强、技能系统增强）在流程中是否生效。

运行方式: python -m pytest test_refactoring_points.py -v
"""

import json
import os
import time

import pytest

# =====================================================================
# Test 1: Pydantic 工具定义
# =====================================================================

from pydantic import BaseModel, Field
from automl_react.tools.base_tool import BaseTool, ToolResult, ToolStatus


class _DummyInput(BaseModel):
    name: str = Field(..., description="名称")
    count: int = Field(1, ge=0, description="数量")


class _DummyTool(BaseTool):
    name = "dummy_tool"
    description = "用于测试的虚拟工具"
    input_model = _DummyInput

    def execute(self, name: str = "", count: int = 1, **kwargs) -> ToolResult:
        return ToolResult.success(data={"name": name, "count": count})


class _LegacyTool(BaseTool):
    """无 input_model 的旧式工具"""
    name = "legacy_tool"
    description = "用于测试的旧式工具"
    parameters = {"msg": {"type": "string"}}

    def execute(self, msg: str = "hello", **kwargs) -> ToolResult:
        return ToolResult.success(data=msg)


class TestPydanticToolDefinition:
    """Test 1: Pydantic 工具定义验证"""

    def test_valid_params(self):
        tool = _DummyTool()
        result = tool.execute_validated(name="test", count=5)
        assert result.status == ToolStatus.SUCCESS
        assert result.data["name"] == "test"
        assert result.data["count"] == 5

    def test_invalid_params_type(self):
        tool = _DummyTool()
        result = tool.execute_validated(name=123, count="not_int")
        # name 被 Pydantic 强制转为 str "123"，但 count 非法
        # Pydantic v2 会对 "not_int" → int 抛 ValidationError
        assert result.status == ToolStatus.ERROR
        assert "参数验证失败" in result.error

    def test_invalid_params_constraint(self):
        tool = _DummyTool()
        result = tool.execute_validated(name="ok", count=-1)
        assert result.status == ToolStatus.ERROR
        assert "参数验证失败" in result.error

    def test_missing_required_param(self):
        tool = _DummyTool()
        # name 是必填项
        result = tool.execute_validated(count=1)
        assert result.status == ToolStatus.ERROR
        assert "参数验证失败" in result.error

    def test_schema_from_pydantic(self):
        tool = _DummyTool()
        schema = tool.get_schema()
        assert schema["name"] == "dummy_tool"
        assert "name" in schema["parameters"]
        assert "count" in schema["parameters"]

    def test_no_input_model_bypass(self):
        tool = _LegacyTool()
        result = tool.execute_validated(msg="world")
        assert result.status == ToolStatus.SUCCESS
        assert result.data == "world"

    def test_legacy_schema(self):
        tool = _LegacyTool()
        schema = tool.get_schema()
        assert schema["parameters"] == {"msg": {"type": "string"}}

    def test_callable_uses_validated(self):
        tool = _DummyTool()
        result = tool(name="callable_test", count=2)
        assert result.status == ToolStatus.SUCCESS
        assert result.data["name"] == "callable_test"


# =====================================================================
# Test 2: Multi-LLM Provider 抽象
# =====================================================================

from automl_react.llm.provider_factory import LLMProviderFactory, BaseProvider
from automl_react.llm.providers.openai_compatible_provider import _KNOWN_PROVIDERS


class TestLLMProviderFactory:
    """Test 2: Multi-LLM Provider 抽象验证"""

    def test_registered_providers(self):
        providers = LLMProviderFactory.available_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "openai_compatible" in providers

    def test_unknown_provider_raises(self):
        from automl_react.api.main import LLMClientError

        with pytest.raises(LLMClientError, match="未知 LLM provider"):
            LLMProviderFactory.create({"provider": "nonexistent_provider"})

    def test_register_custom_provider(self):
        class _MockProvider(BaseProvider):
            called = False

            @classmethod
            def create_client(cls, config):
                cls.called = True
                return "mock_client"

        LLMProviderFactory.register("test_mock", _MockProvider)
        try:
            assert "test_mock" in LLMProviderFactory.available_providers()
            client = LLMProviderFactory.create({"provider": "test_mock"})
            assert client == "mock_client"
            assert _MockProvider.called
        finally:
            # 清理注册表
            LLMProviderFactory._registry.pop("test_mock", None)

    def test_openai_compatible_known_providers(self):
        assert "dashscope" in _KNOWN_PROVIDERS
        assert "deepseek" in _KNOWN_PROVIDERS
        assert "moonshot" in _KNOWN_PROVIDERS
        assert "zhipu" in _KNOWN_PROVIDERS
        # 验证 base_url 格式
        for name, url in _KNOWN_PROVIDERS.items():
            assert url.startswith("https://"), f"{name} 的 base_url 应为 https"


# =====================================================================
# Test 3: 中间件链
# =====================================================================

from automl_react.core.middleware import Middleware, MiddlewareChain, IterationContext
from automl_react.core.middlewares.timeout_middleware import TimeoutMiddleware
from automl_react.core.middlewares.token_monitor_middleware import TokenMonitorMiddleware
from automl_react.core.middlewares.error_handling_middleware import ErrorHandlingMiddleware


def _make_ctx(**overrides):
    """创建测试用 IterationContext"""
    defaults = dict(
        iteration=1,
        max_iterations=5,
        stage="test",
        user_input="test input",
    )
    defaults.update(overrides)
    return IterationContext(**defaults)


class TestMiddlewareChain:
    """Test 3: 中间件链验证"""

    def test_chain_execution_order(self):
        order = []

        class MW1(Middleware):
            def before_llm_call(self, ctx):
                order.append("mw1")
                return ctx

        class MW2(Middleware):
            def before_llm_call(self, ctx):
                order.append("mw2")
                return ctx

        chain = MiddlewareChain([MW1(), MW2()])
        ctx = _make_ctx()
        chain.run_before_llm(ctx)
        assert order == ["mw1", "mw2"]

    def test_chain_should_stop(self):
        order = []

        class StopMW(Middleware):
            def before_llm_call(self, ctx):
                ctx.should_stop = True
                order.append("stop")
                return ctx

        class NeverReached(Middleware):
            def before_llm_call(self, ctx):
                order.append("never")
                return ctx

        chain = MiddlewareChain([StopMW(), NeverReached()])
        ctx = _make_ctx()
        ctx = chain.run_before_llm(ctx)
        assert order == ["stop"]
        assert ctx.should_stop is True

    def test_timeout_middleware_triggers(self):
        mw = TimeoutMiddleware(total_timeout=0, per_iteration_timeout=120)
        ctx = _make_ctx()
        # 先初始化 start_time
        mw._start_time = time.monotonic() - 1  # 已超时
        ctx = mw.before_llm_call(ctx)
        assert ctx.should_stop is True
        assert "超时" in ctx.error

    def test_token_monitor_counts(self):
        mw = TokenMonitorMiddleware(max_tokens=128000)
        ctx = _make_ctx(prompt="这是一段用来测试 token 计数的文本" * 10)
        ctx = mw.before_llm_call(ctx)
        assert ctx.metadata.get("input_tokens", 0) > 0

        ctx.llm_response = "LLM 回复内容"
        ctx = mw.after_llm_call(ctx)
        assert ctx.metadata.get("output_tokens", 0) > 0
        assert ctx.metadata.get("total_tokens", 0) > 0

    def test_error_handling_on_error(self):
        mw = ErrorHandlingMiddleware(verbose=False)
        ctx = _make_ctx()
        error = ValueError("test error message")
        ctx = mw.on_error(ctx, error)
        assert "ValueError" in ctx.error
        assert "test error message" in ctx.error
        assert len(ctx.metadata["errors"]) == 1
        assert ctx.metadata["errors"][0]["type"] == "ValueError"

    def test_all_hooks_exist(self):
        """确认所有 5 个 hook 点都可调用"""
        chain = MiddlewareChain([Middleware()])
        ctx = _make_ctx()
        chain.run_before_llm(ctx)
        chain.run_after_llm(ctx)
        chain.run_before_tool(ctx)
        chain.run_after_tool(ctx)
        chain.run_on_error(ctx, Exception("test"))


# =====================================================================
# Test 4: 上下文摘要/压缩
# =====================================================================

from automl_react.core.memory import Memory, MemoryType
from automl_react.core.middlewares.summarization_middleware import SummarizationMiddleware


class TestContextSummarization:
    """Test 4: 上下文摘要/压缩验证"""

    def test_estimate_tokens_positive(self):
        mem = Memory()
        for i in range(5):
            mem.add_user_message(f"这是第 {i} 条消息，包含足够的文本用于 token 估算")
        tokens = mem.estimate_tokens()
        assert tokens > 0

    def test_estimate_tokens_empty(self):
        mem = Memory()
        assert mem.estimate_tokens() == 0

    def test_summarize_compresses(self):
        mem = Memory()
        for i in range(10):
            mem.add_user_message(f"消息 {i}")
        assert len(mem.short_term) == 10

        mem.summarize(lambda x: "这是摘要内容", keep_recent=3)
        # 1 条摘要 + 3 条最近 = 4
        assert len(mem.short_term) == 4
        assert "[历史摘要]" in mem.short_term[0].content
        assert mem.short_term[0].type == MemoryType.SYSTEM_MESSAGE

    def test_summarize_keeps_recent_order(self):
        mem = Memory()
        for i in range(10):
            mem.add_user_message(f"消息 {i}")
        mem.summarize(lambda x: "摘要", keep_recent=3)
        # 最近 3 条应该是 消息7, 消息8, 消息9
        assert "消息 7" in mem.short_term[1].content
        assert "消息 8" in mem.short_term[2].content
        assert "消息 9" in mem.short_term[3].content

    def test_summarize_skip_when_few(self):
        mem = Memory()
        for i in range(3):
            mem.add_user_message(f"消息 {i}")
        original_len = len(mem.short_term)
        mem.summarize(lambda x: "摘要", keep_recent=5)
        assert len(mem.short_term) == original_len

    def test_summarize_fn_failure_no_crash(self):
        mem = Memory()
        for i in range(10):
            mem.add_user_message(f"消息 {i}")

        def bad_fn(x):
            raise RuntimeError("LLM failure")

        original_len = len(mem.short_term)
        mem.summarize(bad_fn, keep_recent=3)
        # 摘要失败，保持原样
        assert len(mem.short_term) == original_len

    def test_summarization_middleware_triggers(self):
        mem = Memory()
        # 添加大量内容使 token 超过阈值
        for i in range(50):
            mem.add_user_message(f"这是一段较长的测试消息内容，编号 {i}，用于撑大 token 数" * 5)

        mw = SummarizationMiddleware(
            memory=mem,
            llm=None,  # 无 LLM，用截断方式摘要
            max_context_tokens=100,  # 设置极低阈值
            summarization_threshold=0.5,
            keep_recent=3,
        )

        ctx = _make_ctx()
        ctx = mw.before_llm_call(ctx)

        # 如果 token 超过阈值，应该触发摘要
        if mem.estimate_tokens() > 0:
            # 摘要后 short_term 应被压缩
            assert len(mem.short_term) <= 4  # 1 摘要 + 3 最近
            assert "summarization" in ctx.metadata

    def test_summarization_middleware_no_trigger(self):
        mem = Memory()
        mem.add_user_message("短消息")

        mw = SummarizationMiddleware(
            memory=mem,
            max_context_tokens=100000,
            summarization_threshold=0.75,
        )
        ctx = _make_ctx()
        ctx = mw.before_llm_call(ctx)
        assert "summarization" not in ctx.metadata


# =====================================================================
# Test 5: 沙盒执行增强
# =====================================================================

from automl_react.utils.subprocess_executor import SubprocessCodeExecutor
from automl_react.utils.sandbox import SandboxExecutor


class TestSandboxExecution:
    """Test 5: 沙盒执行增强验证"""

    def test_resource_limits_in_wrapper(self):
        executor = SubprocessCodeExecutor(
            memory_limit_mb=1024, cpu_time_limit=60
        )
        script = executor._build_wrapper_script(
            "print('hello')", "/tmp/ctx.pkl", "/tmp/out.pkl", []
        )
        assert "resource.setrlimit" in script
        assert "1024 * 1024 * 1024" in script  # memory_limit_mb * 1024 * 1024
        assert "RLIMIT_CPU" in script
        assert "_cpu_limit = 60" in script

    def test_env_filtering(self):
        executor = SubprocessCodeExecutor()
        # 设置敏感环境变量
        os.environ["OPENAI_API_KEY"] = "test_key_12345"
        os.environ["ANTHROPIC_API_KEY"] = "test_key_67890"
        os.environ["DASHSCOPE_API_KEY"] = "test_dash"
        try:
            env = executor._build_env()
            assert "OPENAI_API_KEY" not in env
            assert "ANTHROPIC_API_KEY" not in env
            assert "DASHSCOPE_API_KEY" not in env
            # PATH 等正常变量应保留
            assert "PATH" in env
        finally:
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("DASHSCOPE_API_KEY", None)

    def test_sandbox_subprocess_mode(self):
        sandbox = SandboxExecutor(mode="subprocess")
        assert isinstance(sandbox._executor, SubprocessCodeExecutor)

    def test_sandbox_invalid_mode(self):
        with pytest.raises(ValueError, match="不支持的沙盒模式"):
            SandboxExecutor(mode="invalid_mode")

    def test_sandbox_execute_simple_code(self):
        sandbox = SandboxExecutor(mode="subprocess", timeout=30)
        result = sandbox.execute("x = 1 + 1\nprint(x)")
        assert result.success is True
        assert "2" in result.output

    def test_sandbox_execute_with_context(self):
        sandbox = SandboxExecutor(mode="subprocess", timeout=30)
        result = sandbox.execute(
            "print(greeting)",
            context={"greeting": "hello_world"},
        )
        assert result.success is True
        assert "hello_world" in result.output

    def test_sandbox_execute_collect_variables(self):
        sandbox = SandboxExecutor(mode="subprocess", timeout=30)
        result = sandbox.execute(
            "result = 42\nreport = 'done'",
            required_output_names=["result", "report"],
        )
        assert result.success is True
        assert result.variables.get("result") == 42
        assert result.variables.get("report") == "done"

    def test_sandbox_syntax_error(self):
        sandbox = SandboxExecutor(mode="subprocess", timeout=30)
        result = sandbox.execute("def bad(:\n  pass")
        assert result.success is False


# =====================================================================
# Test 6: 技能系统增强
# =====================================================================

from automl_react.tools.skill_tools import _STAGE_TAG_MAP, SkillSearchTool, SkillReadTool, SkillReadInput


class TestSkillSystemEnhancement:
    """Test 6: 技能系统增强验证"""

    def test_skill_packages_exist(self):
        base = os.path.join(os.path.dirname(__file__), "skills")
        fe_dir = os.path.join(base, "feature-engineering-patterns-1.0.0")
        ms_dir = os.path.join(base, "model-selection-heuristics-1.0.0")

        # feature-engineering
        assert os.path.isfile(os.path.join(fe_dir, "_meta.json"))
        assert os.path.isfile(os.path.join(fe_dir, "SKILL.md"))
        assert os.path.isfile(os.path.join(fe_dir, "numeric-transforms.md"))
        assert os.path.isfile(os.path.join(fe_dir, "categorical-encoding.md"))

        # model-selection
        assert os.path.isfile(os.path.join(ms_dir, "_meta.json"))
        assert os.path.isfile(os.path.join(ms_dir, "SKILL.md"))
        assert os.path.isfile(os.path.join(ms_dir, "classification-models.md"))
        assert os.path.isfile(os.path.join(ms_dir, "regression-models.md"))

    def test_meta_json_tags(self):
        base = os.path.join(os.path.dirname(__file__), "skills")

        fe_meta = os.path.join(base, "feature-engineering-patterns-1.0.0", "_meta.json")
        with open(fe_meta, "r") as f:
            meta = json.load(f)
        assert "feature-engineering" in meta["tags"]
        assert "ml-patterns" in meta["tags"]

        ms_meta = os.path.join(base, "model-selection-heuristics-1.0.0", "_meta.json")
        with open(ms_meta, "r") as f:
            meta = json.load(f)
        assert "model-selection" in meta["tags"]

    def test_stage_tag_map_keys(self):
        assert "data_cleaning" in _STAGE_TAG_MAP
        assert "feature_engineering" in _STAGE_TAG_MAP
        assert "model_training" in _STAGE_TAG_MAP
        assert "model_evaluation" in _STAGE_TAG_MAP

    def test_stage_tag_map_values(self):
        fe_tags = _STAGE_TAG_MAP["feature_engineering"]
        assert "feature-engineering" in fe_tags
        assert "ml-patterns" in fe_tags

        mt_tags = _STAGE_TAG_MAP["model_training"]
        assert "model-selection" in mt_tags

    def test_skill_search_by_stage(self):
        tool = SkillSearchTool()
        result = tool.execute(stage="feature_engineering")
        assert result.status == ToolStatus.SUCCESS
        assert isinstance(result.data, dict)
        assert len(result.data["skills"]) > 0
        # 应包含 feature-engineering-patterns
        names = [s["name"] for s in result.data["skills"]]
        assert any("feature-engineering" in n for n in names)

    def test_skill_search_by_stage_model_training(self):
        tool = SkillSearchTool()
        result = tool.execute(stage="model_training")
        assert result.status == ToolStatus.SUCCESS
        assert isinstance(result.data, dict)
        names = [s["name"] for s in result.data["skills"]]
        assert any("model-selection" in n for n in names)

    def test_skill_search_without_stage(self):
        tool = SkillSearchTool()
        result = tool.execute(query="feature")
        assert result.status == ToolStatus.SUCCESS

    def test_skill_search_no_args_returns_all(self):
        tool = SkillSearchTool()
        result = tool.execute()
        assert result.status == ToolStatus.SUCCESS

    def test_skill_read_tool(self):
        tool = SkillReadTool()
        # 先搜索获取名称
        search = SkillSearchTool()
        search_result = search.execute(stage="feature_engineering")
        if isinstance(search_result.data, dict):
            skill_name = search_result.data["skills"][0]["name"]
            sections = search_result.data["skills"][0]["sections"]
            if sections:
                read_result = tool.execute(
                    skill_name=skill_name, section=sections[0]
                )
                assert read_result.status == ToolStatus.SUCCESS
                assert len(read_result.data) > 0

    def test_skill_read_no_truncation(self):
        """验证 skill 内容不再被截断"""
        tool = SkillReadTool()
        search = SkillSearchTool()
        search_result = search.execute(stage="data_cleaning")
        if isinstance(search_result.data, dict):
            skill_name = search_result.data["skills"][0]["name"]
            sections = search_result.data["skills"][0]["sections"]
            if sections:
                result = tool.execute(skill_name=skill_name, section=sections[0])
                assert result.status == ToolStatus.SUCCESS
                # 确保内容没有截断标记
                assert "内容已截断" not in result.data
                assert "..." not in result.data[-20:]  # 末尾不应有截断省略号

    def test_skill_read_input_no_max_length(self):
        """验证 SkillReadInput 不再有 max_length 字段"""
        fields = SkillReadInput.model_fields
        assert "max_length" not in fields, "SkillReadInput 不应再包含 max_length 字段"


# =====================================================================
# Test 7: Prompt 去重 & SummarizationMiddleware 集成
# =====================================================================

from automl_react.core.memory import Memory, MemoryType


class TestPromptDeduplication:
    """验证 prompt 构建不会重复 observation"""

    def test_memory_skip_last_observation(self):
        """skip_last_observation=True 时最后一条 observation 不在上下文中"""
        mem = Memory()
        mem.add_thought("thought1", step=1)
        mem.add_action("action1", {"tool": "t"})
        mem.add_observation("obs_content_unique_marker")

        ctx_with = mem.get_short_term_context(
            include_user_messages=False,
            include_assistant_messages=False,
            skip_last_observation=False,
        )
        assert "obs_content_unique_marker" in ctx_with

        ctx_without = mem.get_short_term_context(
            include_user_messages=False,
            include_assistant_messages=False,
            skip_last_observation=True,
        )
        assert "obs_content_unique_marker" not in ctx_without

    def test_memory_skip_last_observation_keeps_older(self):
        """skip_last_observation 只跳过最后一条，保留更早的 observation"""
        mem = Memory()
        mem.add_observation("old_observation")
        mem.add_thought("thought2", step=2)
        mem.add_action("action2", {"tool": "t2"})
        mem.add_observation("new_observation")

        ctx = mem.get_short_term_context(
            include_user_messages=False,
            include_assistant_messages=False,
            skip_last_observation=True,
        )
        assert "old_observation" in ctx
        assert "new_observation" not in ctx

    def test_memory_skip_no_observation(self):
        """没有 observation 时 skip_last_observation=True 不影响结果"""
        mem = Memory()
        mem.add_thought("thought1", step=1)

        ctx = mem.get_short_term_context(
            include_user_messages=False,
            include_assistant_messages=False,
            skip_last_observation=True,
        )
        assert "thought1" in ctx

    def test_summarization_middleware_in_defaults(self):
        """SummarizationMiddleware 应在默认中间件列表中"""
        from automl_react.core.middlewares.summarization_middleware import SummarizationMiddleware

        # 构建一个 minimal Agent 来检查默认中间件
        # 使用 DataCleaningAgent 作为具体子类
        from automl_react.agents.data_cleaning_agent import DataCleaningAgent
        agent = DataCleaningAgent(llm=None, session_id="test_mw")
        mw_types = [type(m).__name__ for m in agent._middleware_chain._middlewares]
        assert "SummarizationMiddleware" in mw_types
