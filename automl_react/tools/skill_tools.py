"""
技能工具模块

提供 SkillSearchTool 和 SkillReadTool，让 LLM 在 ReAct 循环中自主检索和阅读技能知识
"""

from typing import Any, Dict, List

from .base_tool import BaseTool, ToolResult
from ..skills_loader import get_skill_loader


class SkillSearchTool(BaseTool):
    """搜索可用技能包的工具"""

    name = "search_skills"
    description = (
        "搜索可用的专业知识技能包。根据关键词或标签搜索，"
        "返回匹配的技能名称、摘要和可用章节列表。"
        "示例关键词: 'feature engineering', '数据清洗', 'model evaluation', 'benchmarking'"
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "搜索关键词（支持中英文）"
        },
        "tags": {
            "type": "array",
            "description": "可选的标签过滤（如 ['ml-engineering', 'data-analysis']）",
            "items": {"type": "string"}
        }
    }

    def execute(self, query: str = "", tags: List[str] = None, **kwargs) -> ToolResult:
        """执行技能搜索"""
        try:
            loader = get_skill_loader()
            loader.load_all_skills()

            matched_names = set()

            if query:
                matched_names.update(loader.search_skills(query))

            if tags:
                matched_names.update(loader.search_by_tags(tags))

            # 如果没有搜索条件，返回所有技能
            if not query and not tags:
                matched_names.update(loader.load_all_skills().keys())

            results = []
            for name in sorted(matched_names):
                skill = loader.get_skill(name)
                if not skill:
                    continue
                sections = loader.list_sections(name) or {}
                results.append({
                    "name": name,
                    "version": skill.version,
                    "summary": skill.summary or skill.description or "(无摘要)",
                    "tags": skill.tags,
                    "sections": list(sections.keys()),
                })

            if not results:
                return ToolResult.success(data="未找到匹配的技能包。请尝试其他关键词。")

            return ToolResult.success(
                data={
                    "note": "以上仅为技能包索引摘要，不包含实际内容。必须使用 read_skill 工具并指定 section 参数来读取具体章节内容。",
                    "skills": results,
                }
            )

        except Exception as e:
            return ToolResult.error(f"技能搜索失败: {e}")


class SkillReadTool(BaseTool):
    """读取指定技能包内容的工具"""

    name = "read_skill"
    description = (
        "读取指定技能包的详细内容。可以读取完整概述或指定章节。"
        "先用 search_skills 找到技能名和章节，再用此工具读取具体内容。"
    )
    parameters = {
        "skill_name": {
            "type": "string",
            "description": "技能包名称（如 'data-analysis-1.0.2'）"
        },
        "section": {
            "type": "string",
            "description": "可选的章节名称（如 'techniques', 'phase2-data-engineering'）。不指定则返回技能概述"
        },
        "max_length": {
            "type": "integer",
            "description": "返回内容最大字符数，默认 3000"
        }
    }

    def execute(
        self,
        skill_name: str = "",
        section: str = "",
        max_length: int = 3000,
        **kwargs
    ) -> ToolResult:
        """读取技能内容"""
        if not skill_name:
            return ToolResult.error("必须指定 skill_name 参数")

        try:
            loader = get_skill_loader()
            skill = loader.get_skill(skill_name)

            if not skill:
                available = loader.scan_skills()
                return ToolResult.error(
                    f"技能 '{skill_name}' 不存在。可用技能: {available}"
                )

            if section:
                content = loader.get_section_content(skill_name, section)
                if not content:
                    sections = loader.list_sections(skill_name) or {}
                    return ToolResult.error(
                        f"章节 '{section}' 不存在于技能 '{skill_name}' 中。"
                        f"可用章节: {list(sections.keys())}"
                    )
            else:
                content = skill.content or "(该技能无概述内容)"

            # 截断
            if len(content) > max_length:
                content = content[:max_length] + f"\n\n... [内容已截断，共 {len(content)} 字符]"

            # 添加参考警告
            header = (
                "⚠️ 以下内容为技术参考，仅供方法指导，"
                "所有决策必须基于当前会话的真实数据。\n\n"
            )

            return ToolResult.success(data=header + content)

        except Exception as e:
            return ToolResult.error(f"读取技能失败: {e}")
