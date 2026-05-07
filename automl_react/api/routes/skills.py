"""
Skills 路由

/skills/* 端点
"""

from fastapi import APIRouter, HTTPException

from automl_react.skills_loader import get_skill_loader

router = APIRouter()


@router.get("/skills/list")
async def list_skills():
    """列出所有可用的 skills"""
    skill_loader = get_skill_loader()
    skills = skill_loader.scan_skills()
    return {"skills": skills}


@router.get("/skills/{skill_name}/content")
async def get_skill_content(skill_name: str):
    """获取 skill 内容"""
    skill_loader = get_skill_loader()

    try:
        skill = skill_loader.load_skill(skill_name)
        return {
            "name": skill.name,
            "version": skill.version,
            "description": skill.description,
            "references": skill.list_references()
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
