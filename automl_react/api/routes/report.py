"""
报告路由

/report/*, /pipeline/* 端点
"""

import json

from fastapi import APIRouter, Depends, HTTPException

from automl_react.assets import get_asset_manager
from automl_react.report import PipelineGenerator, ReportGenerator

from ..deps import validate_session_id, get_registry
from ..registry import AppRegistry
from ..helpers import get_effective_target_column, get_effective_task_type

router = APIRouter()


@router.post("/report/generate")
async def generate_report(session_id: str, registry: AppRegistry = Depends(get_registry)):
    """生成报告"""
    validate_session_id(session_id)
    session = await registry.get_session(session_id)
    workflow_state = session.get("workflow_state")

    if not workflow_state:
        raise HTTPException(status_code=404, detail="会话不存在")

    data_path = workflow_state.get_context("data_path")
    target_column = get_effective_target_column(workflow_state)
    task_type = get_effective_task_type(workflow_state)

    report_generator = ReportGenerator(session_id=session_id)

    try:
        report = report_generator.generate_report(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type
        )
        html_report = report_generator.export_to_html(report)
        summary = report_generator.generate_summary_json(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type
        )

        return {
            "success": True,
            "message": "报告已生成",
            "downloads": {
                "markdown": f"/assets/{session_id}/reports/modeling_report.md",
                "html": f"/assets/{session_id}/reports/modeling_report.html",
                "summary": f"/assets/{session_id}/reports/summary.json"
            },
            "summary": summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/generate")
async def generate_pipeline(session_id: str, registry: AppRegistry = Depends(get_registry)):
    """生成全流程脚本"""
    validate_session_id(session_id)
    session = await registry.get_session(session_id)
    workflow_state = session.get("workflow_state")

    if not workflow_state:
        raise HTTPException(status_code=404, detail="会话不存在")

    data_path = workflow_state.get_context("data_path")
    target_column = get_effective_target_column(workflow_state)
    task_type = get_effective_task_type(workflow_state)

    pipeline_generator = PipelineGenerator(session_id=session_id)

    try:
        pipeline_dir = pipeline_generator.generate_pipeline_package(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type
        )

        return {
            "success": True,
            "message": "全流程 pipeline 包已生成",
            "pipeline_dir": pipeline_dir,
            "download_url": f"/assets/{session_id}/code/pipeline.py"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/{session_id}/summary")
async def get_report_summary(session_id: str):
    """获取结构化 JSON 摘要"""
    validate_session_id(session_id)
    asset_manager = get_asset_manager(session_id=session_id)
    summary_json = asset_manager.read_asset("reports", "summary.json")
    if not summary_json:
        raise HTTPException(status_code=404, detail="摘要报告尚未生成")
    try:
        return json.loads(summary_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="摘要报告格式错误")
