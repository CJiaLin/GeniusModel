"""
资产路由

/assets/* 端点
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from automl_react.assets import get_asset_manager
from automl_react.logger import get_llm_logger

from ..deps import validate_session_id

router = APIRouter()


@router.get("/assets/{session_id}/{asset_type}/{filename:path}")
async def download_asset(session_id: str, asset_type: str, filename: str):
    """下载资产文件"""
    validate_session_id(session_id)
    asset_manager = get_asset_manager(session_id=session_id)
    try:
        asset_path = asset_manager.get_asset(asset_type, filename)
    except ValueError:
        raise HTTPException(status_code=403, detail="非法文件路径")

    if not asset_path:
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        path=asset_path,
        filename=Path(filename).name,
        media_type="application/octet-stream"
    )


@router.get("/assets/{session_id}/list")
async def list_assets(session_id: str):
    """列出所有资产"""
    validate_session_id(session_id)
    asset_manager = get_asset_manager(session_id=session_id)
    urls = asset_manager.get_all_download_urls()

    return {
        "session_id": session_id,
        "assets": urls
    }


@router.get("/logs/{session_id}/llm")
async def get_llm_logs(session_id: str, limit: int = 100):
    """获取 LLM 调用日志"""
    llm_logger = get_llm_logger(session_id=session_id)
    logs = llm_logger.get_logs(limit=limit)

    return {
        "session_id": session_id,
        "logs": [log.to_dict() for log in logs]
    }
