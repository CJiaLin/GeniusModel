"""
工作流路由

/workflow/* 端点
"""

import json
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

from automl_react.workflow import WorkflowState, WorkflowStage
from automl_react.workflow.workflow_state import WorkflowMode, get_stages_for_mode
from automl_react.confirmation import ConfirmationManager

from ..deps import validate_session_id, get_registry
from ..registry import AppRegistry
from ..helpers import (
    ensure_session_data_path,
    resolve_stage_alias,
    get_effective_target_column,
    get_effective_task_type,
)
from ..services.stage_runner import (
    run_problem_definition,
    run_data_aggregation,
    run_data_contract_check,
    run_data_splitting,
    run_data_exploration,
    run_data_cleaning,
    run_feature_engineering,
    run_model_training,
)

router = APIRouter()


class StartWorkflowRequest(BaseModel):
    session_id: str
    data_path: Optional[str] = None
    target_column: str = ""
    task_type: str = "classification"
    model: str = "claude-sonnet-4-20250514-thinking"
    task_description: str = ""
    workflow_mode: str = "full"  # "full" | "schema_only" | "feature_only"
    schema_file_path: Optional[str] = None  # for schema_only mode
    extra_data_paths: Optional[List[str]] = None  # 额外的数据文件路径列表（多表聚合）


@router.post("/workflow/start")
async def start_workflow(request: StartWorkflowRequest, registry: AppRegistry = Depends(get_registry)):
    """启动工作流"""
    session_id = request.session_id
    validate_session_id(session_id)
    session = await registry.get_session(session_id)

    # 根据 workflow_mode 处理数据路径
    if request.workflow_mode == "schema_only":
        # schema_only 模式：data_path 非必须，schema_file_path 或 task_description 即可
        data_path = ensure_session_data_path(session_id, request.data_path) if request.data_path else None
        schema_file_path = ensure_session_data_path(session_id, request.schema_file_path) if request.schema_file_path else None
    else:
        # full / feature_only 模式：需要实际数据
        data_path = ensure_session_data_path(session_id, request.data_path)
        schema_file_path = None
        if not data_path:
            raise HTTPException(status_code=400, detail="数据路径不存在")

    # 创建或获取工作流状态
    workflow_state = session.get("workflow_state")
    if not workflow_state:
        workflow_state = WorkflowState(session_id=session_id)
        session["workflow_state"] = workflow_state

    # 更新上下文
    workflow_state.update_context({
        "data_path": data_path,
        "target_column": request.target_column,
        "task_type": request.task_type,
        "model": request.model,
        "task_description": request.task_description,
        "workflow_mode": request.workflow_mode,
        "schema_file_path": schema_file_path,
        "extra_data_paths": request.extra_data_paths or [],
        "is_multi_table": bool(request.extra_data_paths),
    })
    workflow_state.save()

    # 创建确认管理器
    if not session.get("confirmation_manager"):
        cm_path = str(Path("assets") / session_id / "state" / "confirmation_state.json")
        session["confirmation_manager"] = ConfirmationManager(save_path=cm_path)

    return {
        "success": True,
        "session_id": session_id,
        "current_stage": workflow_state.current_stage.value,
        "data_path": data_path,
        "workflow_mode": request.workflow_mode,
    }


@router.get("/workflow/{session_id}/status")
async def get_workflow_status(session_id: str, registry: AppRegistry = Depends(get_registry)):
    """获取工作流状态"""
    validate_session_id(session_id)
    session = await registry.get_session(session_id)
    workflow_state = session.get("workflow_state")

    if not workflow_state:
        raise HTTPException(status_code=404, detail="会话不存在")

    return {
        "session_id": session_id,
        "current_stage": workflow_state.current_stage.value,
        "context": workflow_state.context,
        "history": workflow_state.history[-10:] if workflow_state.history else []
    }


@router.post("/workflow/{session_id}/stage/{stage}/run")
async def run_stage(session_id: str, stage: str, background_tasks: BackgroundTasks, registry: AppRegistry = Depends(get_registry)):
    """运行指定阶段"""
    validate_session_id(session_id)
    session = await registry.get_session(session_id)
    workflow_state = session.get("workflow_state")

    if not workflow_state:
        raise HTTPException(status_code=404, detail="会话不存在")

    requested_stage = stage
    stage = resolve_stage_alias(stage)

    # 模式守卫：检查该阶段是否在当前模式的有效阶段列表中
    mode = workflow_state.get_context("workflow_mode", WorkflowMode.FULL)
    mode_stages = get_stages_for_mode(mode)
    if stage not in mode_stages and stage not in ("completed", "error"):
        raise HTTPException(status_code=400, detail=f"阶段 '{stage}' 不在当前 '{mode}' 模式中可用")

    # 更新工作流状态
    target_stage = WorkflowStage(stage)
    if workflow_state.current_stage != target_stage:
        try:
            workflow_state.transition_to(target_stage)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的阶段转换: {e}")

    # 获取上下文
    data_path = workflow_state.get_context("data_path")
    target_column = get_effective_target_column(workflow_state)
    task_type = get_effective_task_type(workflow_state)

    # 获取或创建确认管理器
    confirmation_manager = session.get("confirmation_manager")
    if not confirmation_manager:
        cm_path = str(Path("assets") / session_id / "state" / "confirmation_state.json")
        confirmation_manager = ConfirmationManager(save_path=cm_path)
        session["confirmation_manager"] = confirmation_manager
        print(f"[API] 已重新创建确认管理器")

    # 根据阶段分发
    if stage == "problem_definition":
        return run_problem_definition(
            session, workflow_state, confirmation_manager,
            data_path, session_id, requested_stage,
        )
    elif stage == "data_aggregation":
        return run_data_aggregation(
            session, workflow_state, confirmation_manager,
            data_path, session_id,
        )
    elif stage == "data_contract_check":
        return run_data_contract_check(
            session, workflow_state, confirmation_manager,
            data_path, target_column, task_type, session_id,
        )
    elif stage == "data_splitting":
        return run_data_splitting(
            session, workflow_state, confirmation_manager,
            data_path, target_column, task_type, session_id,
        )
    elif stage == "data_exploration":
        return run_data_exploration(
            session, workflow_state,
            data_path, target_column, task_type, session_id,
        )
    elif stage == "data_cleaning":
        return run_data_cleaning(
            session, workflow_state, confirmation_manager,
            data_path, session_id,
        )
    elif stage == "feature_engineering":
        return run_feature_engineering(
            session, workflow_state, confirmation_manager,
            data_path, target_column, task_type, session_id,
        )
    elif stage == "model_training":
        return run_model_training(
            session, workflow_state, confirmation_manager,
            data_path, target_column, task_type, session_id,
        )

    return {"success": False, "error": "未知的阶段"}
