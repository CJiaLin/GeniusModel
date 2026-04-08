"""
新工具单元测试

测试 SkillSearchTool, SkillReadTool, DataProfileTool, StageResultTool
以及 SkillLoader 的增强方法
"""

import os
import sys
import json
import tempfile
import pytest
import pandas as pd
import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from automl_react.skills_loader.skill_loader import SkillLoader, Skill, get_skill_loader
from automl_react.tools.skill_tools import SkillSearchTool, SkillReadTool
from automl_react.tools.profile_tools import DataProfileTool


# ============================================================
# SkillLoader 增强方法测试
# ============================================================


class TestSkillLoaderEnhanced:
    """测试 SkillLoader 的新方法"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前重置全局 loader"""
        import automl_react.skills_loader.skill_loader as mod
        mod._skill_loader = None
        self.loader = get_skill_loader()

    def test_skill_has_tags(self):
        skill = self.loader.load_skill("data-analysis-1.0.2")
        assert isinstance(skill.tags, list)
        assert len(skill.tags) > 0
        assert "data-analysis" in skill.tags

    def test_skill_has_summary(self):
        skill = self.loader.load_skill("data-analysis-1.0.2")
        assert skill.summary
        assert len(skill.summary) > 10

    def test_skill_has_sections(self):
        skill = self.loader.load_skill("data-analysis-1.0.2")
        assert isinstance(skill.sections, dict)
        assert "techniques" in skill.sections
        assert "pitfalls" in skill.sections

    def test_get_skill_summary(self):
        summary = self.loader.get_skill_summary("data-analysis-1.0.2")
        assert summary is not None
        assert len(summary) > 0

    def test_get_skill_summary_nonexistent(self):
        summary = self.loader.get_skill_summary("nonexistent-skill")
        assert summary is None

    def test_search_by_tags(self):
        results = self.loader.search_by_tags(["data-analysis"])
        assert "data-analysis-1.0.2" in results

    def test_search_by_tags_ml(self):
        results = self.loader.search_by_tags(["ml-engineering"])
        assert "afrexai-ml-engineering-1.0.0" in results

    def test_search_by_tags_no_match(self):
        results = self.loader.search_by_tags(["nonexistent-tag"])
        assert len(results) == 0

    def test_list_sections(self):
        sections = self.loader.list_sections("data-analysis-1.0.2")
        assert sections is not None
        assert "techniques" in sections
        assert sections["techniques"] == "techniques.md"

    def test_list_sections_nonexistent(self):
        sections = self.loader.list_sections("nonexistent-skill")
        assert sections is None

    def test_get_section_content_file_ref(self):
        content = self.loader.get_section_content("data-analysis-1.0.2", "techniques")
        assert content is not None
        assert len(content) > 100

    def test_get_section_content_overview(self):
        content = self.loader.get_section_content("data-analysis-1.0.2", "overview")
        assert content is not None
        assert len(content) > 0

    def test_get_section_content_phase(self):
        content = self.loader.get_section_content(
            "afrexai-ml-engineering-1.0.0", "phase2-data-engineering"
        )
        # 可能为 None 如果 SKILL.md 中没有匹配的 Phase 2 标题
        # 但如果 SKILL.md 存在且包含 Phase 2，应该返回内容
        skill = self.loader.get_skill("afrexai-ml-engineering-1.0.0")
        if skill and "Phase 2" in skill.content:
            assert content is not None
            assert len(content) > 0

    def test_get_section_content_nonexistent(self):
        content = self.loader.get_section_content("data-analysis-1.0.2", "nonexistent")
        assert content is None

    def test_search_skills_enhanced(self):
        """测试增强后的 search_skills 能搜索 summary 和 tags"""
        results = self.loader.search_skills("数据分析")
        assert "data-analysis-1.0.2" in results

    def test_search_skills_by_tag_keyword(self):
        results = self.loader.search_skills("benchmarking")
        assert "ml-model-eval-benchmark-0.1.0" in results


# ============================================================
# SkillSearchTool 测试
# ============================================================


class TestSkillSearchTool:
    """测试 SkillSearchTool"""

    @pytest.fixture(autouse=True)
    def setup(self):
        import automl_react.skills_loader.skill_loader as mod
        mod._skill_loader = None
        self.tool = SkillSearchTool()

    def test_search_by_query(self):
        result = self.tool.execute(query="data analysis")
        assert result.status.value == "success"
        assert isinstance(result.data, list)
        names = [r["name"] for r in result.data]
        assert "data-analysis-1.0.2" in names

    def test_search_by_tags(self):
        result = self.tool.execute(tags=["model-evaluation"])
        assert result.status.value == "success"
        assert isinstance(result.data, list)
        names = [r["name"] for r in result.data]
        assert "ml-model-eval-benchmark-0.1.0" in names

    def test_search_no_match(self):
        result = self.tool.execute(query="zzz_nonexistent_zzz")
        assert result.status.value == "success"
        assert "未找到" in str(result.data)

    def test_search_no_args_returns_all(self):
        result = self.tool.execute()
        assert result.status.value == "success"
        assert isinstance(result.data, list)
        assert len(result.data) >= 3  # 至少 3 个技能包

    def test_result_has_sections(self):
        result = self.tool.execute(query="data-analysis")
        assert result.status.value == "success"
        for item in result.data:
            assert "sections" in item
            assert isinstance(item["sections"], list)


# ============================================================
# SkillReadTool 测试
# ============================================================


class TestSkillReadTool:
    """测试 SkillReadTool"""

    @pytest.fixture(autouse=True)
    def setup(self):
        import automl_react.skills_loader.skill_loader as mod
        mod._skill_loader = None
        self.tool = SkillReadTool()

    def test_read_overview(self):
        result = self.tool.execute(skill_name="data-analysis-1.0.2")
        assert result.status.value == "success"
        assert "⚠️" in str(result.data)

    def test_read_section(self):
        result = self.tool.execute(
            skill_name="data-analysis-1.0.2", section="techniques"
        )
        assert result.status.value == "success"
        assert len(str(result.data)) > 100

    def test_read_nonexistent_skill(self):
        result = self.tool.execute(skill_name="nonexistent")
        assert result.status.value == "error"

    def test_read_nonexistent_section(self):
        result = self.tool.execute(
            skill_name="data-analysis-1.0.2", section="nonexistent_section"
        )
        assert result.status.value == "error"
        assert "可用章节" in str(result.error)

    def test_max_length_truncation(self):
        result = self.tool.execute(
            skill_name="data-analysis-1.0.2", max_length=100
        )
        assert result.status.value == "success"
        # 内容 = 警告头 + 截断后的内容，总长度应受限
        # 由于有 header，实际输出会大于 max_length，但原始内容被截断
        assert "截断" in str(result.data) or len(str(result.data)) < 500

    def test_no_skill_name(self):
        result = self.tool.execute()
        assert result.status.value == "error"


# ============================================================
# DataProfileTool 测试
# ============================================================


class TestDataProfileTool:
    """测试 DataProfileTool"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tool = DataProfileTool()
        # 创建临时测试 CSV
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "test_data.csv")
        np.random.seed(42)
        df = pd.DataFrame({
            "numeric_col": np.random.randn(100),
            "with_missing": np.where(np.random.rand(100) > 0.7, np.nan, np.random.randn(100)),
            "category": np.random.choice(["A", "B", "C"], 100),
            "with_outlier": np.concatenate([np.random.randn(95), np.array([100, -100, 200, -200, 150])]),
        })
        # 添加一些重复行
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        df.to_csv(self.csv_path, index=False)

    def test_basic_profile(self):
        result = self.tool.execute(file_path=self.csv_path)
        assert result.status.value == "success"
        data = result.data
        assert "shape" in data
        assert data["shape"]["rows"] == 105
        assert data["shape"]["columns"] == 4

    def test_missing_detection(self):
        result = self.tool.execute(file_path=self.csv_path)
        data = result.data
        assert "missing" in data
        assert "with_missing" in data["missing"]
        assert data["missing"]["with_missing"]["count"] > 0

    def test_outlier_detection(self):
        result = self.tool.execute(file_path=self.csv_path)
        data = result.data
        assert "outliers" in data
        assert "with_outlier" in data["outliers"]
        assert data["outliers"]["with_outlier"]["count"] > 0

    def test_duplicate_detection(self):
        result = self.tool.execute(file_path=self.csv_path)
        data = result.data
        assert "duplicates" in data
        assert data["duplicates"]["count"] == 5

    def test_report_generated(self):
        result = self.tool.execute(file_path=self.csv_path)
        data = result.data
        assert "report" in data
        assert "数据质量报告" in data["report"]

    def test_nonexistent_file(self):
        result = self.tool.execute(file_path="/nonexistent/file.csv")
        assert result.status.value == "error"

    def test_no_file_path(self):
        result = self.tool.execute()
        assert result.status.value == "error"

    def test_with_train_csv(self):
        """用项目的 train.csv 测试（如果存在）"""
        train_path = os.path.join(
            os.path.dirname(__file__), "..", "train.csv"
        )
        if os.path.isfile(train_path):
            result = self.tool.execute(file_path=train_path)
            assert result.status.value == "success"
            assert result.data["shape"]["rows"] > 0


# ============================================================
# StageResultTool 测试（使用 mock 方式）
# ============================================================


class TestStageResultTool:
    """测试 StageResultTool"""

    def test_import(self):
        """验证导入正常"""
        from automl_react.tools.stage_tools import StageResultTool
        tool = StageResultTool(session_id="test")
        assert tool.name == "query_stage_result"
        assert tool._session_id == "test"

    def test_no_stage(self):
        from automl_react.tools.stage_tools import StageResultTool
        tool = StageResultTool(session_id="test")
        result = tool.execute()
        assert result.status.value == "error"
        assert "stage" in str(result.error)

    def test_invalid_stage(self):
        from automl_react.tools.stage_tools import StageResultTool
        tool = StageResultTool(session_id="test")
        result = tool.execute(stage="nonexistent_stage")
        assert result.status.value == "error"
        assert "未知阶段" in str(result.error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
