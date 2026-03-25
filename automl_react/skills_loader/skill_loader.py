"""
Skill 加载器模块

加载和管理 skills 目录中的专业知识
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Skill:
    """
    Skill 数据类

    Attributes:
        name: Skill 名称
        version: Skill 版本
        description: Skill 描述
        path: Skill 目录路径
        meta: 元数据
        content: SKILL.md 内容
        references: 参考文件内容字典
    """
    name: str
    version: str
    description: str
    path: Path
    meta: Dict[str, Any] = field(default_factory=dict)
    content: str = ""
    references: Dict[str, str] = field(default_factory=dict)

    def get_reference(self, filename: str) -> Optional[str]:
        """获取指定参考文件内容"""
        return self.references.get(filename)

    def list_references(self) -> List[str]:
        """列出所有参考文件"""
        return list(self.references.keys())


class SkillLoader:
    """
    Skill 加载器

    负责扫描、加载和管理 skills 目录中的专业知识

    Attributes:
        skills_dir: skills 目录路径
        _skills: 已加载的 skill 字典
    """

    def __init__(self, skills_dir: str = None):
        """
        初始化 Skill 加载器

        Args:
            skills_dir: skills 目录路径，默认为项目根目录的 skills/
        """
        if skills_dir is None:
            # 默认 skills 目录：项目根目录/skills
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            skills_dir = project_root / "skills"

        self.skills_dir = Path(skills_dir)
        self._skills: Dict[str, Skill] = {}

    def scan_skills(self) -> List[str]:
        """
        扫描 skills 目录，返回所有 skill 名称列表

        Returns:
            Skill 名称列表
        """
        if not self.skills_dir.exists():
            return []

        skills = []
        for item in self.skills_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                # 检查是否包含 _meta.json 或 SKILL.md
                if (item / "_meta.json").exists() or (item / "SKILL.md").exists():
                    skills.append(item.name)

        return skills

    def _load_meta(self, skill_path: Path) -> Dict[str, Any]:
        """
        加载 skill 的元数据

        Args:
            skill_path: Skill 目录路径

        Returns:
            元数据字典
        """
        meta_file = skill_path / "_meta.json"

        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # 如果没有 _meta.json，尝试从目录名解析
        dir_name = skill_path.name
        parts = dir_name.rsplit("-", 2)

        if len(parts) >= 2:
            return {
                "name": parts[0],
                "version": parts[-1] if len(parts) > 2 else "1.0.0"
            }

        return {"name": dir_name, "version": "1.0.0"}

    def _load_content(self, skill_path: Path) -> str:
        """
        加载 SKILL.md 内容

        Args:
            skill_path: Skill 目录路径

        Returns:
            SKILL.md 内容
        """
        skill_md = skill_path / "SKILL.md"

        if skill_md.exists():
            with open(skill_md, "r", encoding="utf-8") as f:
                return f.read()

        return ""

    def _load_references(self, skill_path: Path) -> Dict[str, str]:
        """
        加载所有参考文件（.md 文件）

        Args:
            skill_path: Skill 目录路径

        Returns:
            参考文件内容字典
        """
        references = {}

        for md_file in skill_path.glob("*.md"):
            if md_file.name != "SKILL.md":
                with open(md_file, "r", encoding="utf-8") as f:
                    references[md_file.name] = f.read()

        # 递归加载子目录中的 .md 文件
        for subdir in skill_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                for md_file in subdir.rglob("*.md"):
                    rel_path = md_file.relative_to(skill_path)
                    with open(md_file, "r", encoding="utf-8") as f:
                        references[str(rel_path)] = f.read()

        return references

    def load_skill(self, skill_name: str) -> Skill:
        """
        加载指定 skill

        Args:
            skill_name: Skill 名称（目录名）

        Returns:
            Skill 对象
        """
        if skill_name in self._skills:
            return self._skills[skill_name]

        skill_path = self.skills_dir / skill_name

        if not skill_path.exists():
            raise FileNotFoundError(f"Skill '{skill_name}' 不存在")

        # 加载元数据
        meta = self._load_meta(skill_path)

        # 加载内容
        content = self._load_content(skill_path)

        # 加载参考文件
        references = self._load_references(skill_path)

        # 创建 Skill 对象
        skill = Skill(
            name=meta.get("name", skill_name),
            version=meta.get("version", "1.0.0"),
            description=meta.get("description", ""),
            path=skill_path,
            meta=meta,
            content=content,
            references=references
        )

        self._skills[skill_name] = skill
        return skill

    def load_all_skills(self) -> Dict[str, Skill]:
        """
        加载所有 skills

        Returns:
            Skill 字典
        """
        skill_names = self.scan_skills()

        for name in skill_names:
            if name not in self._skills:
                self.load_skill(name)

        return self._skills

    def get_skill(self, skill_name: str) -> Optional[Skill]:
        """
        获取已加载的 skill

        Args:
            skill_name: Skill 名称

        Returns:
            Skill 对象，如果不存在返回 None
        """
        if skill_name in self._skills:
            return self._skills.get(skill_name)

        skill_path = self.skills_dir / skill_name
        if not skill_path.exists():
            return None

        return self.load_skill(skill_name)

    def get_skill_content(self, skill_name: str) -> Optional[str]:
        """
        获取 skill 的 SKILL.md 内容

        Args:
            skill_name: Skill 名称

        Returns:
            SKILL.md 内容
        """
        skill = self.get_skill(skill_name)
        if skill:
            return skill.content
        return None

    def get_skill_reference(self, skill_name: str, filename: str) -> Optional[str]:
        """
        获取 skill 的指定参考文件内容

        Args:
            skill_name: Skill 名称
            filename: 参考文件名

        Returns:
            参考文件内容
        """
        skill = self.get_skill(skill_name)
        if skill:
            return skill.get_reference(filename)
        return None

    def search_skills(self, keyword: str) -> List[str]:
        """
        搜索包含关键词的 skill

        Args:
            keyword: 搜索关键词

        Returns:
            匹配的 skill 名称列表
        """
        results = []

        for name, skill in self._skills.items():
            # 检查名称
            if keyword.lower() in name.lower():
                results.append(name)
                continue

            # 检查描述
            if keyword.lower() in skill.description.lower():
                results.append(name)
                continue

            # 检查内容
            if keyword.lower() in skill.content.lower():
                results.append(name)
                continue

        return results

    def reload_skill(self, skill_name: str) -> Skill:
        """
        重新加载指定 skill

        Args:
            skill_name: Skill 名称

        Returns:
            Skill 对象
        """
        if skill_name in self._skills:
            del self._skills[skill_name]

        return self.load_skill(skill_name)

    def list_skills(self) -> List[str]:
        """
        列出所有已加载的 skill

        Returns:
            Skill 名称列表
        """
        return list(self._skills.keys())


# 全局 SkillLoader 实例
_skill_loader: Optional[SkillLoader] = None


def get_skill_loader(skills_dir: str = None) -> SkillLoader:
    """
    获取全局 SkillLoader 实例

    Args:
        skills_dir: skills 目录路径

    Returns:
        SkillLoader 实例
    """
    global _skill_loader

    if _skill_loader is None:
        _skill_loader = SkillLoader(skills_dir)

    return _skill_loader
