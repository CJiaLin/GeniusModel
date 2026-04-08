"""
P3 功能测试：报告系统增强 + 会话管理加固
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestConfirmationManagerPersistence(unittest.TestCase):
    """测试 ConfirmationManager 持久化"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.save_path = os.path.join(self.tmpdir, "state", "confirmation_state.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_creates_file(self):
        from automl_react.confirmation.confirmation_point import ConfirmationManager
        mgr = ConfirmationManager(save_path=self.save_path)
        mgr.add_confirmation_point(stage="test", proposal_content="hello")
        self.assertTrue(os.path.exists(self.save_path))

    def test_save_and_load_roundtrip(self):
        from automl_react.confirmation.confirmation_point import (
            ConfirmationManager, ConfirmationStatus
        )
        mgr = ConfirmationManager(save_path=self.save_path)
        point = mgr.add_confirmation_point(
            stage="data_cleaning",
            proposal_content="clean plan v1",
            metadata={"key": "value"},
        )
        point.modifiable_aspects = ["策略1", "策略2"]

        # 提交响应
        mgr.submit_response(
            point_id=point.id,
            status=ConfirmationStatus.CONFIRMED,
            comment="looks good",
        )

        # 加载
        loaded = ConfirmationManager.load_from_disk(self.save_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded._save_path, self.save_path)

    def test_load_nonexistent_returns_none(self):
        from automl_react.confirmation.confirmation_point import ConfirmationManager
        result = ConfirmationManager.load_from_disk("/nonexistent/path.json")
        self.assertIsNone(result)

    def test_load_corrupt_json_returns_none(self):
        from automl_react.confirmation.confirmation_point import ConfirmationManager
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w") as f:
            f.write("not json {{{")
        result = ConfirmationManager.load_from_disk(self.save_path)
        self.assertIsNone(result)

    def test_auto_save_on_add(self):
        from automl_react.confirmation.confirmation_point import ConfirmationManager
        mgr = ConfirmationManager(save_path=self.save_path)
        mgr.add_confirmation_point(stage="test", proposal_content="p")
        # File should exist after add
        self.assertTrue(os.path.exists(self.save_path))
        with open(self.save_path) as f:
            data = json.load(f)
        self.assertEqual(len(data["queue"]), 1)

    def test_auto_save_on_submit(self):
        from automl_react.confirmation.confirmation_point import (
            ConfirmationManager, ConfirmationStatus
        )
        mgr = ConfirmationManager(save_path=self.save_path)
        point = mgr.add_confirmation_point(stage="test", proposal_content="p")
        mgr.submit_response(point.id, ConfirmationStatus.CONFIRMED)

        with open(self.save_path) as f:
            data = json.load(f)
        # After submit, point should have a response
        queue_point = data["queue"][0]
        self.assertEqual(queue_point["user_response"]["status"], "confirmed")

    def test_no_save_path_no_file(self):
        from automl_react.confirmation.confirmation_point import ConfirmationManager
        mgr = ConfirmationManager()  # No save_path
        mgr.add_confirmation_point(stage="test", proposal_content="p")
        # Should not raise and no file created
        self.assertFalse(os.path.exists(self.save_path))

    def test_save_explicit_path(self):
        from automl_react.confirmation.confirmation_point import ConfirmationManager
        mgr = ConfirmationManager()  # No default save_path
        mgr.add_confirmation_point(stage="test", proposal_content="p")
        explicit_path = os.path.join(self.tmpdir, "explicit.json")
        mgr.save(explicit_path)
        self.assertTrue(os.path.exists(explicit_path))


class TestReportGenerator(unittest.TestCase):
    """测试增强的报告生成器"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.session_id = "test_report_session"
        # Ensure assets directory
        self.assets_dir = os.path.join(self.tmpdir, self.session_id)
        os.makedirs(os.path.join(self.assets_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(self.assets_dir, "cleaning"), exist_ok=True)
        os.makedirs(os.path.join(self.assets_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.assets_dir, "features"), exist_ok=True)
        os.makedirs(os.path.join(self.assets_dir, "analysis"), exist_ok=True)
        os.makedirs(os.path.join(self.assets_dir, "exploration"), exist_ok=True)
        os.makedirs(os.path.join(self.assets_dir, "data"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_report_generator_import(self):
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="test")
        self.assertEqual(gen.session_id, "test")

    def test_load_json_asset_empty(self):
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="nonexistent")
        result = gen._load_cleaning_result()
        self.assertEqual(result, {})

    def test_load_problem_definition(self):
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="nonexistent")
        result = gen._load_problem_definition()
        self.assertEqual(result, {})

    def test_load_splitting_result(self):
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="nonexistent")
        result = gen._load_splitting_result()
        self.assertEqual(result, {})

    def test_load_exploration_result(self):
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="nonexistent")
        result = gen._load_exploration_result()
        self.assertIsNone(result)

    def test_generate_conclusions_classification(self):
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="test")
        lines = gen._generate_conclusions(
            task_type="classification",
            evaluation_result={"metrics": {"accuracy": 0.92, "f1": 0.89}},
            training_summary={"metrics": {"accuracy": 0.95}, "best_model": "XGBoost"},
            model_result={},
        )
        text = "\n".join(lines)
        self.assertIn("92", text)  # accuracy shown
        self.assertIn("XGBoost", text)

    def test_generate_conclusions_regression(self):
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="test")
        lines = gen._generate_conclusions(
            task_type="regression",
            evaluation_result={"metrics": {"r2": 0.75, "rmse": 12.3}},
            training_summary={"metrics": {"r2": 0.85}},
            model_result={},
        )
        text = "\n".join(lines)
        self.assertIn("0.75", text)
        self.assertIn("12.3", text)

    def test_generate_conclusions_empty_metrics(self):
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="test")
        lines = gen._generate_conclusions(
            task_type="classification",
            evaluation_result={},
            training_summary={},
            model_result={},
        )
        text = "\n".join(lines)
        self.assertIn("不可用", text)

    def test_overfit_detection(self):
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="test")
        lines = gen._generate_conclusions(
            task_type="classification",
            evaluation_result={"metrics": {"accuracy": 0.70}},
            training_summary={"metrics": {"accuracy": 0.99}},
            model_result={},
        )
        text = "\n".join(lines)
        self.assertIn("过拟合", text)

    def test_build_report_has_all_sections(self):
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="test")
        report = gen._build_report(
            data_path="/data/test.csv",
            target_column="target",
            task_type="classification",
            problem_definition={"task_type": "classification", "primary_metric": "accuracy"},
            splitting_result={"split_strategy": "stratified", "split_sizes": {"train": 800, "valid": 100, "test": 100}},
            cleaning_result={"success": True, "cleaned_data_path": "/cleaned.csv"},
            exploration_result="## 探索发现\n特征分布正常",
            feature_result={"success": True, "new_features": ["f1", "f2"], "total_features": 10},
            feature_evaluation_result={},
            model_result={"success": True, "metrics": {"accuracy": 0.9}},
            training_summary={"best_model": "XGBoost", "metrics": {"accuracy": 0.95}},
            evaluation_result={"success": True, "metrics": {"accuracy": 0.88}},
            chart_paths={},
        )
        self.assertIn("项目概述", report)
        self.assertIn("数据切分", report)
        self.assertIn("数据清洗", report)
        self.assertIn("数据探索分析", report)
        self.assertIn("特征工程", report)
        self.assertIn("模型训练", report)
        self.assertIn("模型评估", report)
        self.assertIn("可视化分析", report)
        self.assertIn("结论与建议", report)

    def test_generate_summary_json(self):
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id=self.session_id)
        # Use actual asset_manager to set up base_dir
        from automl_react.assets import get_asset_manager
        am = get_asset_manager(session_id=self.session_id)

        summary = gen.generate_summary_json(
            data_path="/test.csv",
            target_column="target",
            task_type="classification",
        )
        self.assertEqual(summary["session_id"], self.session_id)
        self.assertIn("problem_definition", summary)
        self.assertIn("data_summary", summary)
        self.assertIn("cleaning_summary", summary)
        self.assertIn("feature_summary", summary)
        self.assertIn("model_summary", summary)
        self.assertIn("evaluation_summary", summary)
        self.assertIn("generated_at", summary)

    def test_chart_generation_no_matplotlib(self):
        """测试无 matplotlib 时图表生成不报错"""
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="test")
        # Should return empty dict if matplotlib is unavailable or no data
        result = gen._generate_charts({}, {}, "classification")
        self.assertIsInstance(result, dict)


class TestSessionCRUDModels(unittest.TestCase):
    """测试 API 模型导入和 session helper"""

    def test_ensure_agent_helper_exists(self):
        """测试 _ensure_agent 辅助函数可导入"""
        # Just verify the module loads without error
        from automl_react.api.main import _ensure_agent
        self.assertTrue(callable(_ensure_agent))

    def test_ensure_agent_creates_new(self):
        """测试 _ensure_agent 创建新 Agent"""
        from automl_react.api.main import _ensure_agent

        class FakeAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        session = {"agents": {}}
        agent = _ensure_agent(session, "test", FakeAgent, x=1)
        self.assertIsNotNone(agent)
        self.assertEqual(agent.kwargs, {"x": 1})
        self.assertIn("test", session["agents"])

    def test_ensure_agent_returns_existing(self):
        """测试 _ensure_agent 返回已有 Agent"""
        from automl_react.api.main import _ensure_agent

        existing = object()
        session = {"agents": {"test": existing}}
        agent = _ensure_agent(session, "test", None)  # class won't be called
        self.assertIs(agent, existing)


class TestSessionTTLCleanup(unittest.TestCase):
    """测试会话 TTL 清理逻辑"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cleanup_expired_sessions(self):
        """测试清理过期会话"""
        from automl_react.api.main import _cleanup_expired_sessions

        # Create a fake expired session
        expired_dir = Path("assets") / "expired_session" / "state"
        expired_dir.mkdir(parents=True)
        old_time = (datetime.now() - timedelta(hours=100)).isoformat()
        with open(expired_dir / "workflow_state.json", "w") as f:
            json.dump({"last_updated": old_time, "current_stage": "COMPLETED"}, f)

        # Create a fresh session
        fresh_dir = Path("assets") / "fresh_session" / "state"
        fresh_dir.mkdir(parents=True)
        with open(fresh_dir / "workflow_state.json", "w") as f:
            json.dump({"last_updated": datetime.now().isoformat(), "current_stage": "DATA_CLEANING"}, f)

        _cleanup_expired_sessions()

        self.assertFalse((Path("assets") / "expired_session").exists())
        self.assertTrue((Path("assets") / "fresh_session").exists())

    def test_cleanup_no_assets_dir(self):
        """测试无 assets 目录不报错"""
        from automl_react.api.main import _cleanup_expired_sessions
        _cleanup_expired_sessions()  # Should not raise


class TestChartGeneration(unittest.TestCase):
    """测试图表生成功能"""

    def test_feature_importance_chart_with_dict(self):
        """测试特征重要性图生成（dict 格式）"""
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="test_charts")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib not installed")

        tmpdir = tempfile.mkdtemp()
        try:
            result = gen._chart_feature_importance(
                plt,
                {"feat_a": 0.3, "feat_b": 0.5, "feat_c": 0.1},
                ["feat_a", "feat_b", "feat_c"],
                Path(tmpdir),
            )
            self.assertTrue(result.endswith(".png"))
            self.assertTrue(os.path.exists(result))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_feature_importance_chart_with_list(self):
        """测试特征重要性图生成（list 格式）"""
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="test_charts")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib not installed")

        tmpdir = tempfile.mkdtemp()
        try:
            result = gen._chart_feature_importance(
                plt,
                [0.3, 0.5, 0.1],
                ["feat_a", "feat_b", "feat_c"],
                Path(tmpdir),
            )
            self.assertTrue(result.endswith(".png"))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_metrics_comparison_chart(self):
        """测试指标对比图生成"""
        from automl_react.report.report_generator import ReportGenerator
        gen = ReportGenerator(session_id="test_charts")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib not installed")

        tmpdir = tempfile.mkdtemp()
        try:
            result = gen._chart_metrics_comparison(
                plt,
                {"accuracy": 0.95, "f1": 0.92},
                {"accuracy": 0.88, "f1": 0.85},
                Path(tmpdir),
            )
            self.assertTrue(result.endswith(".png"))
            self.assertTrue(os.path.exists(result))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
