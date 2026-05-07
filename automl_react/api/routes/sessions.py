"""
会话路由

/sessions/* 端点
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from automl_react.assets import get_asset_manager
from automl_react.confirmation import ConfirmationManager
from automl_react.confirmation.confirmation_point import UserConfirmationPoint

from ..deps import validate_session_id, get_registry
from ..registry import AppRegistry

router = APIRouter()


@router.get("/sessions")
async def list_sessions(registry: AppRegistry = Depends(get_registry)):
    """列出所有会话"""
    sessions_info = registry.get_all_sessions_info()
    return {
        "success": True,
        "sessions": list(sessions_info.values()),
        "count": len(sessions_info),
    }


@router.get("/sessions/{session_id}/status")
async def get_session_detail(session_id: str, registry: AppRegistry = Depends(get_registry)):
    """获取完整会话状态详情"""
    validate_session_id(session_id)
    session = await registry.get_session(session_id)
    ws = session.get("workflow_state")
    if not ws:
        raise HTTPException(status_code=404, detail="会话不存在")

    cm = session.get("confirmation_manager")
    agents_keys = list(session.get("agents", {}).keys())

    return {
        "success": True,
        "session_id": session_id,
        "current_stage": ws.current_stage.value if ws.current_stage else "unknown",
        "context": ws.context,
        "history": ws.history[-20:],
        "agents_loaded": agents_keys,
        "confirmation_state": cm.to_dict() if cm else None,
        "created_at": session.get("created_at", ""),
    }


@router.get("/sessions/{session_id}/restore")
async def restore_session(session_id: str, registry: AppRegistry = Depends(get_registry)):
    """恢复会话，返回前端重建 UI 所需的全部数据"""
    validate_session_id(session_id)
    session = await registry.get_session(session_id)
    ws = session.get("workflow_state")
    if not ws:
        raise HTTPException(status_code=404, detail="会话不存在")

    cm: Optional[ConfirmationManager] = session.get("confirmation_manager")
    asset_manager = get_asset_manager(session_id=session_id)

    # 阶段顺序
    stage_keys = [
        "data_upload", "problem_definition", "data_contract_check",
        "data_splitting", "data_cleaning", "data_exploration",
        "feature_engineering", "model_training",
    ]

    # 根据 confirmation history 重建 stage_data 和 stage_status
    stage_data: Dict[str, Any] = {}
    stage_status: Dict[str, str] = {}

    if cm:
        all_points = cm.history + ([cm.current] if cm.current else []) + cm._queue
        stage_points: Dict[str, UserConfirmationPoint] = {}
        for p in all_points:
            stage_points[p.stage] = p

        for stage_key, point in stage_points.items():
            sd: Dict[str, Any] = {
                "proposal": point.proposal_content,
                "confirmation_id": point.id,
                "modifiable_aspects": point.modifiable_aspects,
            }
            if point.metadata:
                for k, v in point.metadata.items():
                    if k not in ("stage",):
                        sd[k] = v
            if point.skills_referenced:
                sd["skills_referenced"] = [
                    {"skill_name": s.skill_name, "skill_path": s.skill_path, "reference_file": s.reference_file}
                    for s in point.skills_referenced
                ]

            if point.is_resolved():
                sd["requires_confirmation"] = False
                stage_status[stage_key] = "completed"
            elif point.is_rejected():
                sd["requires_confirmation"] = False
                stage_status[stage_key] = "error"
            else:
                sd["requires_confirmation"] = True
                stage_status[stage_key] = "pending"

            stage_data[stage_key] = sd

    if ws.get_context("data_path"):
        stage_status.setdefault("data_upload", "completed")

    # 重建 generated_code
    generated_code: Dict[str, str] = {}
    code_dir = asset_manager.session_dir / "code"
    code_file_map = {
        "data_cleaning": "cleaning.py",
        "feature_engineering": "feature_engineering.py",
        "model_training": "model_training.py",
    }
    for stage_key, filename in code_file_map.items():
        code_file = code_dir / filename
        if code_file.exists():
            try:
                generated_code[stage_key] = code_file.read_text(encoding="utf-8")
            except Exception:
                pass

    # 资产列表
    assets_data = []
    try:
        assets_data = asset_manager.get_all_download_urls()
    except Exception:
        pass

    # 当前阶段索引
    current_stage_value = ws.current_stage.value if ws.current_stage else "data_upload"
    current_stage_index = len(stage_keys) - 1
    for i, sk in enumerate(stage_keys):
        if sk == current_stage_value:
            current_stage_index = i
            break

    # 待处理的确认
    pending_confirmation = None
    if cm:
        current = cm.current
        if not current:
            pending = cm.get_pending_points()
            current = pending[0] if pending else None
        if current and not current.is_resolved():
            pending_confirmation = {
                "confirmation_id": current.id,
                "stage": current.stage,
            }

    return {
        "success": True,
        "session_id": session_id,
        "current_stage": current_stage_value,
        "context": ws.context,
        "history": ws.history[-20:],
        "stage_data": stage_data,
        "stage_status": stage_status,
        "generated_code": generated_code,
        "assets": assets_data,
        "current_stage_index": current_stage_index,
        "pending_confirmation": pending_confirmation,
        "created_at": session.get("created_at", ""),
    }


@router.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str, confirm: bool = False, registry: AppRegistry = Depends(get_registry)):
    """删除会话及所有关联资产"""
    validate_session_id(session_id)
    if not confirm:
        return {
            "success": False,
            "message": "请添加 ?confirm=true 确认删除",
            "session_id": session_id,
        }

    await registry.delete_session(session_id)

    return {
        "success": True,
        "message": f"会话 {session_id} 已删除",
        "session_id": session_id,
    }
