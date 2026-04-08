"""
P2 功能测试：ConfirmationPoint 扩展 + Agent revise_plan + SubprocessCodeExecutor 集成
"""

import os
import sys
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestConfirmationPointExtension(unittest.TestCase):
    """测试 ConfirmationPoint 扩展字段"""

    def test_revision_requested_status(self):
        """测试新增的 REVISION_REQUESTED 状态"""
        from automl_react.confirmation.confirmation_point import ConfirmationStatus
        status = ConfirmationStatus.REVISION_REQUESTED
        self.assertEqual(status.value, "revision_requested")

    def test_confirmation_point_new_fields(self):
        """测试 UserConfirmationPoint 新增字段"""
        from automl_react.confirmation.confirmation_point import UserConfirmationPoint
        point = UserConfirmationPoint(
            stage="data_cleaning",
            proposal_content="test plan",
        )
        self.assertEqual(point.revision_history, [])
        self.assertEqual(point.modifiable_aspects, [])

    def test_modifiable_aspects_set(self):
        """测试设置 modifiable_aspects"""
        from automl_react.confirmation.confirmation_point import UserConfirmationPoint
        point = UserConfirmationPoint(stage="data_cleaning", proposal_content="plan")
        point.modifiable_aspects = ["缺失值处理策略", "异常值处理方式"]
        self.assertEqual(len(point.modifiable_aspects), 2)

    def test_revision_history_append(self):
        """测试修订历史记录"""
        from automl_react.confirmation.confirmation_point import UserConfirmationPoint
        point = UserConfirmationPoint(stage="data_cleaning", proposal_content="plan v1")
        point.revision_history.append({
            "round": 1,
            "user_feedback": "请使用中位数填充",
            "previous_proposal": "plan v1",
            "timestamp": "2026-04-02T10:00:00",
        })
        self.assertEqual(len(point.revision_history), 1)
        self.assertEqual(point.revision_history[0]["round"], 1)

    def test_to_dict_includes_new_fields(self):
        """测试序列化包含新字段"""
        from automl_react.confirmation.confirmation_point import UserConfirmationPoint
        point = UserConfirmationPoint(stage="test", proposal_content="plan")
        point.modifiable_aspects = ["模型选择"]
        point.revision_history = [{"round": 1, "user_feedback": "test"}]
        d = point.to_dict()
        self.assertIn("modifiable_aspects", d)
        self.assertIn("revision_history", d)
        self.assertEqual(d["modifiable_aspects"], ["模型选择"])

    def test_from_dict_restores_new_fields(self):
        """测试反序列化恢复新字段"""
        from automl_react.confirmation.confirmation_point import UserConfirmationPoint
        point = UserConfirmationPoint(stage="test", proposal_content="plan")
        point.modifiable_aspects = ["编码策略"]
        point.revision_history = [{"round": 1, "user_feedback": "fix"}]
        d = point.to_dict()
        restored = UserConfirmationPoint.from_dict(d)
        self.assertEqual(restored.modifiable_aspects, ["编码策略"])
        self.assertEqual(len(restored.revision_history), 1)

    def test_confirmation_manager_revision_flow(self):
        """测试 ConfirmationManager 的修订工作流"""
        from automl_react.confirmation.confirmation_point import (
            ConfirmationManager, ConfirmationStatus
        )
        manager = ConfirmationManager()

        # 创建初始确认点
        point = manager.add_confirmation_point(
            stage="data_cleaning",
            proposal_content="original plan",
        )
        point.modifiable_aspects = ["缺失值策略"]

        # 用户请求修订
        point.set_user_response(
            status=ConfirmationStatus.REVISION_REQUESTED,
            modifications="请改用中位数填充",
        )
        self.assertEqual(point.user_response.status, ConfirmationStatus.REVISION_REQUESTED)

        # 创建修订后的确认点
        new_point = manager.add_confirmation_point(
            stage="data_cleaning",
            proposal_content="revised plan",
            metadata={"is_revision": True, "parent_id": point.id},
        )
        self.assertEqual(new_point.stage, "data_cleaning")
        self.assertIn("is_revision", new_point.metadata)


class TestAgentModifiableAspects(unittest.TestCase):
    """测试 Agent 的 get_modifiable_aspects"""

    def test_react_agent_base_default(self):
        """测试基类默认返回空列表"""
        from automl_react.core.react_agent import ReActAgent
        # ReActAgent 是抽象类，无法直接实例化，直接测试类方法
        self.assertEqual(ReActAgent.get_modifiable_aspects(None), [])

    def test_react_agent_base_revise_raises(self):
        """测试基类 revise_plan 抛出 NotImplementedError"""
        from automl_react.core.react_agent import ReActAgent
        with self.assertRaises(NotImplementedError):
            ReActAgent.revise_plan(None, "plan", "mods")


class TestPromptTemplates(unittest.TestCase):
    """测试 plan_revision prompt 模板"""

    def setUp(self):
        from automl_react.config import get_config_loader
        self.config = get_config_loader()

    def test_data_cleaning_plan_revision_exists(self):
        """测试 data_cleaning.plan_revision 模板存在"""
        tpl = self.config.get_prompt("data_cleaning", "plan_revision")
        self.assertIn("{current_plan}", tpl)
        self.assertIn("{user_modifications}", tpl)

    def test_feature_engineering_plan_revision_exists(self):
        tpl = self.config.get_prompt("feature_engineering", "plan_revision")
        self.assertIn("{current_plan}", tpl)
        self.assertIn("{user_modifications}", tpl)
        self.assertIn("{target_column}", tpl)

    def test_model_training_plan_revision_exists(self):
        tpl = self.config.get_prompt("model_training", "plan_revision")
        self.assertIn("{current_plan}", tpl)
        self.assertIn("{user_modifications}", tpl)

    def test_data_splitting_plan_revision_exists(self):
        tpl = self.config.get_prompt("data_splitting", "plan_revision")
        self.assertIn("{current_plan}", tpl)
        self.assertIn("{user_modifications}", tpl)


class TestCodeActSubprocessIntegration(unittest.TestCase):
    """测试 CodeActAgent 子进程模式集成"""

    def test_codeact_init_with_subprocess(self):
        """测试 CodeActAgent 默认开启子进程模式"""
        from automl_react.utils.codeact_agent import CodeActAgent
        agent = CodeActAgent(timeout=30)
        self.assertTrue(agent.use_subprocess)
        self.assertIsNotNone(agent.subprocess_executor)

    def test_codeact_init_without_subprocess(self):
        """测试 CodeActAgent 关闭子进程模式"""
        from automl_react.utils.codeact_agent import CodeActAgent
        agent = CodeActAgent(timeout=30, use_subprocess=False)
        self.assertFalse(agent.use_subprocess)

    def test_codeact_env_var_disable(self):
        """测试通过环境变量关闭子进程模式"""
        os.environ["CODEACT_USE_SUBPROCESS"] = "0"
        try:
            from automl_react.utils.codeact_agent import CodeActAgent
            agent = CodeActAgent(timeout=30)
            self.assertFalse(agent.use_subprocess)
        finally:
            del os.environ["CODEACT_USE_SUBPROCESS"]

    def test_execute_code_subprocess_mode(self):
        """测试子进程模式执行代码"""
        from automl_react.utils.codeact_agent import CodeActAgent
        agent = CodeActAgent(timeout=30)
        result = agent._execute_code(
            'x = 1 + 2\nprint(f"sum: {x}")',
            context={},
        )
        self.assertTrue(result["success"])
        self.assertIn("sum: 3", result["output"])

    def test_execute_code_inline_mode(self):
        """测试内联模式执行代码"""
        from automl_react.utils.codeact_agent import CodeActAgent
        agent = CodeActAgent(timeout=30, use_subprocess=False)
        result = agent._execute_code(
            'x = 1 + 2',
            context={},
        )
        self.assertTrue(result["success"])
        self.assertIn("x", result["variables"])
        self.assertEqual(result["variables"]["x"], 3)

    def test_execute_code_subprocess_error(self):
        """测试子进程模式错误捕获"""
        from automl_react.utils.codeact_agent import CodeActAgent
        agent = CodeActAgent(timeout=30)
        result = agent._execute_code('raise ValueError("test")', context={})
        self.assertFalse(result["success"])
        self.assertIn("ValueError", result["error"])


class TestPlanRevisionRequestModel(unittest.TestCase):
    """测试 API 请求模型"""

    def test_plan_revision_request_import(self):
        """测试 PlanRevisionRequest 可以导入"""
        from automl_react.api.main import PlanRevisionRequest
        req = PlanRevisionRequest(
            session_id="test",
            confirmation_id="abc",
            modifications="fix this",
        )
        self.assertEqual(req.session_id, "test")
        self.assertEqual(req.modifications, "fix this")


if __name__ == "__main__":
    unittest.main()
