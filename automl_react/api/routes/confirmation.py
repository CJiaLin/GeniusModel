"""
确认路由

/confirmation/* 端点
"""

import json
import queue
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from automl_react.workflow import WorkflowStage
from automl_react.confirmation import ConfirmationStatus
from automl_react.assets import get_asset_manager
from automl_react.report import PipelineGenerator, ReportGenerator
from automl_react.core.stream_callback import StreamEvent

from ..deps import validate_session_id, get_registry
from ..registry import AppRegistry
from ..helpers import (
    get_effective_target_column,
    get_effective_task_type,
    get_train_raw_data_path,
)
from ..services.stage_executor import (
    execute_problem_definition,
    execute_data_contract_check,
    execute_data_splitting,
    execute_data_cleaning,
    execute_feature_engineering,
    execute_feature_evaluation,
    execute_model_training,
    execute_model_evaluation,
)

router = APIRouter()


class UserConfirmationRequest(BaseModel):
    session_id: str
    confirmation_id: str
    status: str  # confirmed, modified, skipped, rejected
    modifications: Optional[str] = None


class PlanRevisionRequest(BaseModel):
    """方案修订请求"""
    session_id: str
    confirmation_id: str
    modifications: str


@router.post("/confirmation/submit")
async def submit_confirmation(request: UserConfirmationRequest, registry: AppRegistry = Depends(get_registry)):
    """提交用户确认并执行相应操作"""
    session_id = request.session_id
    validate_session_id(session_id)
    session = await registry.get_session(session_id)
    confirmation_manager = session.get("confirmation_manager")
    workflow_state = session.get("workflow_state")

    if not confirmation_manager:
        raise HTTPException(status_code=404, detail="确认管理器不存在")

    status_map = {
        "confirmed": ConfirmationStatus.CONFIRMED,
        "modified": ConfirmationStatus.MODIFIED,
        "skipped": ConfirmationStatus.SKIPPED,
        "rejected": ConfirmationStatus.REJECTED
    }

    status = status_map.get(request.status)
    if not status:
        raise HTTPException(status_code=400, detail="无效的确认状态")

    # 获取确认点
    current_confirmation = confirmation_manager._find_point_by_id(request.confirmation_id)
    if not current_confirmation:
        current_confirmation = confirmation_manager.current
        if not current_confirmation:
            pending_points = confirmation_manager.get_pending_points()
            if pending_points:
                current_confirmation = pending_points[0]
            else:
                raise HTTPException(status_code=404, detail="没有待处理的确认")

    stage = current_confirmation.stage
    data_path = workflow_state.get_context("data_path")
    target_column = get_effective_target_column(workflow_state)
    task_type = get_effective_task_type(workflow_state)

    execution_result = None
    next_confirmation_point = None

    if status in [ConfirmationStatus.CONFIRMED, ConfirmationStatus.MODIFIED]:
        try:
            if stage == "problem_definition":
                execution_result = await execute_problem_definition(
                    session, data_path, request.modifications
                )
            elif stage == "data_contract_check":
                execution_result = await execute_data_contract_check(
                    session, data_path, request.modifications
                )
            elif stage == "data_splitting":
                execution_result = await execute_data_splitting(
                    session, data_path, request.modifications
                )
            elif stage == "data_cleaning":
                execution_result = await execute_data_cleaning(
                    session, get_train_raw_data_path(workflow_state) or data_path, request.modifications
                )
            elif stage == "feature_engineering":
                execution_result = await execute_feature_engineering(
                    session, get_train_raw_data_path(workflow_state) or data_path, target_column, task_type, request.modifications
                )
                # 特征工程完成后提示可选特征评估
                if execution_result.get("success"):
                    evaluation_proposal = """## 是否进行特征评估（可选）

特征工程已执行完成。你可以选择继续进行特征评估，系统将：

1. 由 LLM 生成特征评估代码并执行
2. 基于执行结果生成分析报告，重点覆盖：
   - 特征可解释性（重要性、相关性、信息价值等）
   - 特征可靠性（缺失率、稳定性风险、低方差/高冗余等）
   - 特征筛选与优化建议

请选择：
- `confirmed`：执行特征评估
- `skipped`：跳过特征评估，继续后续流程
"""
                    next_confirmation_point = confirmation_manager.add_confirmation_point(
                        stage="feature_evaluation",
                        proposal_content=evaluation_proposal,
                        expected_outcome="生成特征可解释性与可靠性分析报告",
                        metadata={
                            "stage": "feature_evaluation",
                            "features_data_path": execution_result.get("features_data_path")
                        }
                    )
            elif stage == "model_training":
                execution_result = await execute_model_training(
                    session, get_train_raw_data_path(workflow_state) or data_path, target_column, task_type, request.modifications
                )
                # 训练成功后自动执行评估
                if execution_result.get("success"):
                    try:
                        eval_result = await execute_model_evaluation(session, target_column, task_type)
                        execution_result["evaluation"] = eval_result
                        print(f"[API] 模型评估执行完成: success={eval_result.get('success')}")
                    except Exception as eval_err:
                        import traceback as _tb
                        print(f"[API] 模型评估执行失败（不影响训练结果）: {eval_err}")
                        _tb.print_exc()
                        execution_result["evaluation"] = {"success": False, "error": str(eval_err)}

                    # 标记工作流完成：model_training → model_evaluation → completed
                    workflow_state.transition_to(
                        WorkflowStage.MODEL_EVALUATION,
                        message="Model evaluation completed"
                    )
                    workflow_state.transition_to(
                        WorkflowStage.COMPLETED,
                        message="Workflow completed"
                    )
                    try:
                        report_gen = ReportGenerator(session_id=session_id)
                        dp = workflow_state.get_context("data_path", "")
                        report = report_gen.generate_report(dp, target_column, task_type)
                        report_gen.export_to_html(report)
                        report_gen.generate_summary_json(dp, target_column, task_type)
                        print(f"[API] 自动生成报告完成")
                    except Exception as report_err:
                        import traceback as _tb
                        print(f"[API] 自动报告生成失败（不影响主流程）: {report_err}")
                        _tb.print_exc()
                    try:
                        pipeline_gen = PipelineGenerator(session_id=session_id)
                        dp = workflow_state.get_context("data_path", "")
                        pipeline_gen.generate_pipeline_package(
                            data_path=dp,
                            target_column=target_column,
                            task_type=task_type,
                        )
                        print(f"[API] 自动生成 pipeline 包完成")
                    except Exception as pipeline_err:
                        import traceback as _tb
                        print(f"[API] 自动 pipeline 脚本生成失败（不影响主流程）: {pipeline_err}")
                        _tb.print_exc()
                    try:
                        pipeline_gen = PipelineGenerator(session_id=session_id)
                        dp = workflow_state.get_context("data_path", "")
                        sklearn_path = pipeline_gen.generate_sklearn_pipeline(
                            data_path=dp,
                            target_column=target_column,
                            task_type=task_type,
                        )
                        print(f"[API] 自动生成 sklearn Pipeline 完成: {sklearn_path}")
                    except Exception as sklearn_err:
                        import traceback as _tb
                        print(f"[API] sklearn Pipeline 生成失败（不影响主流程）: {sklearn_err}")
                        _tb.print_exc()
            elif stage == "feature_evaluation":
                execution_result = await execute_feature_evaluation(
                    session, request.modifications
                )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"执行失败: {str(e)}",
                    "stage": stage
                }
            )

    # 提交响应
    try:
        confirmation_manager.submit_response(
            point_id=request.confirmation_id,
            status=status,
            modifications=request.modifications
        )

        response = {
            "success": True,
            "message": "确认已提交",
            "stage": stage,
            "status": request.status
        }

        if execution_result:
            response["execution"] = execution_result

        if next_confirmation_point:
            response["next_confirmation"] = {
                "requires_confirmation": True,
                "stage": "feature_evaluation",
                "confirmation_id": next_confirmation_point.id,
                "proposal": next_confirmation_point.proposal_content
            }

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _get_agent_for_stage(session: Dict, stage: str):
    """根据阶段名返回对应的 Agent 实例。"""
    stage_agent_map = {
        "problem_definition": "analysis",
        "data_contract_check": "analysis",
        "data_splitting": "splitting",
        "data_cleaning": "cleaning",
        "feature_engineering": "feature",
        "model_training": "model",
    }
    agent_key = stage_agent_map.get(stage)
    if not agent_key:
        raise HTTPException(status_code=400, detail=f"不支持方案修订的阶段: {stage}")

    agents = session.get("agents", {})
    agent = agents.get(agent_key)
    if not agent:
        raise HTTPException(status_code=404, detail=f"阶段 '{stage}' 对应的 Agent 未初始化")
    return agent


@router.post("/confirmation/revise")
async def revise_plan(request: PlanRevisionRequest, registry: AppRegistry = Depends(get_registry)):
    """修订当前方案。"""
    validate_session_id(request.session_id)
    session = await registry.get_session(request.session_id)
    confirmation_manager = session.get("confirmation_manager")

    if not confirmation_manager:
        raise HTTPException(status_code=404, detail="确认管理器不存在")

    current = confirmation_manager._find_point_by_id(request.confirmation_id)
    if not current:
        current = confirmation_manager.current
        if not current:
            raise HTTPException(status_code=404, detail="未找到对应的确认点")

    stage = current.stage
    agent = _get_agent_for_stage(session, stage)

    current.revision_history.append({
        "round": len(current.revision_history) + 1,
        "user_feedback": request.modifications,
        "previous_proposal": current.proposal_content,
        "timestamp": datetime.now().isoformat(),
    })

    try:
        revised_plan = agent.revise_plan(
            current_plan=current.proposal_content,
            modifications=request.modifications,
        )
    except NotImplementedError:
        raise HTTPException(
            status_code=400,
            detail=f"阶段 '{stage}' 不支持方案修订",
        )

    current.set_user_response(
        status=ConfirmationStatus.REVISION_REQUESTED,
        modifications=request.modifications,
    )

    new_point = confirmation_manager.add_confirmation_point(
        stage=stage,
        proposal_content=revised_plan,
        metadata={
            "stage": stage,
            "is_revision": True,
            "parent_id": current.id,
            "revision_round": len(current.revision_history),
        },
    )
    new_point.modifiable_aspects = agent.get_modifiable_aspects()
    new_point.revision_history = current.revision_history

    # 保存修订后方案到资产
    asset_manager = get_asset_manager(session_id=request.session_id)
    stage_asset_map = {
        "data_cleaning": ("cleaning", "cleaning_plan.md"),
        "feature_engineering": ("features", "feature_engineering_plan.md"),
        "model_training": ("models", "model_training_plan.md"),
        "data_splitting": ("analysis", "dataset_split_report.md"),
    }
    if stage in stage_asset_map:
        asset_type, filename = stage_asset_map[stage]
        asset_manager.save_data(revised_plan, filename, asset_type)

    return {
        "success": True,
        "stage": stage,
        "proposal": revised_plan,
        "modifiable_aspects": new_point.modifiable_aspects,
        "revision_round": len(new_point.revision_history),
        "requires_confirmation": True,
        "confirmation_id": new_point.id,
    }


@router.get("/confirmation/{session_id}/pending")
async def get_pending_confirmation(session_id: str, registry: AppRegistry = Depends(get_registry)):
    """获取待处理的确认点"""
    validate_session_id(session_id)
    session = await registry.get_session(session_id)
    confirmation_manager = session.get("confirmation_manager")

    if not confirmation_manager:
        raise HTTPException(status_code=404, detail="确认管理器不存在")

    current = confirmation_manager.current
    if not current:
        pending_points = confirmation_manager.get_pending_points()
        current = pending_points[0] if pending_points else None

    if not current:
        return {"has_pending": False}

    return {
        "has_pending": True,
        "confirmation": current.to_dict()
    }


@router.get("/confirmation/{session_id}/submit/stream")
async def submit_confirmation_stream(
    session_id: str,
    confirmation_id: str,
    status: str,
    registry: AppRegistry = Depends(get_registry),
):
    """流式确认执行 — 通过 SSE 推送代码生成、执行输出和最终结果"""
    import asyncio

    validate_session_id(session_id)

    async def event_generator():
        msg_queue: queue.Queue = queue.Queue()

        def stream_callback(event: StreamEvent):
            msg_queue.put(event)

        def _sse(event_type: str, content: str) -> str:
            return f"data: {json.dumps({'type': event_type, 'content': content}, ensure_ascii=False)}\n\n"

        session = await registry.get_session(session_id)
        confirmation_manager = session.get("confirmation_manager")
        workflow_state = session.get("workflow_state")

        if not confirmation_manager:
            yield _sse("error", "确认管理器不存在")
            return

        # 找到确认点
        current_confirmation = confirmation_manager._find_point_by_id(confirmation_id)
        if not current_confirmation:
            current_confirmation = confirmation_manager.current
            if not current_confirmation:
                pending_points = confirmation_manager.get_pending_points()
                if pending_points:
                    current_confirmation = pending_points[0]
                else:
                    yield _sse("error", "没有待处理的确认")
                    return

        stage = current_confirmation.stage
        data_path = workflow_state.get_context("data_path")
        target_column = get_effective_target_column(workflow_state)
        task_type = get_effective_task_type(workflow_state)

        # 设置 agent 的 stream callback
        agent = _get_agent_for_stage(session, stage)
        agent.set_stream_callback(stream_callback)

        yield _sse("progress", f"开始执行 {stage}...")

        # 在线程中运行同步的 execute 函数
        exec_result = {}
        exec_error = None

        async def _run_stage_execution():
            nonlocal exec_result, exec_error
            try:
                if stage == "data_cleaning":
                    exec_result = await execute_data_cleaning(
                        session, get_train_raw_data_path(workflow_state) or data_path, None
                    )
                elif stage == "feature_engineering":
                    exec_result = await execute_feature_engineering(
                        session, get_train_raw_data_path(workflow_state) or data_path, target_column, task_type, None
                    )
                elif stage == "model_training":
                    exec_result = await execute_model_training(
                        session, get_train_raw_data_path(workflow_state) or data_path, target_column, task_type, None
                    )
                    # 训练成功后自动执行评估
                    if exec_result.get("success"):
                        try:
                            eval_result = await execute_model_evaluation(session, target_column, task_type)
                            exec_result["evaluation"] = eval_result
                        except Exception as eval_err:
                            exec_result["evaluation"] = {"success": False, "error": str(eval_err)}

                        # 标记工作流完成
                        workflow_state.transition_to(
                            WorkflowStage.MODEL_EVALUATION,
                            message="Model evaluation completed"
                        )
                        workflow_state.transition_to(
                            WorkflowStage.COMPLETED,
                            message="Workflow completed"
                        )
                        # 自动生成报告
                        try:
                            report_gen = ReportGenerator(session_id=session_id)
                            dp = workflow_state.get_context("data_path", "")
                            report = report_gen.generate_report(dp, target_column, task_type)
                            report_gen.export_to_html(report)
                            report_gen.generate_summary_json(dp, target_column, task_type)
                        except Exception:
                            pass
                        try:
                            pipeline_gen = PipelineGenerator(session_id=session_id)
                            dp = workflow_state.get_context("data_path", "")
                            pipeline_gen.generate_pipeline_package(
                                data_path=dp,
                                target_column=target_column,
                                task_type=task_type,
                            )
                        except Exception:
                            pass
                        try:
                            pipeline_gen = PipelineGenerator(session_id=session_id)
                            dp = workflow_state.get_context("data_path", "")
                            pipeline_gen.generate_sklearn_pipeline(
                                data_path=dp,
                                target_column=target_column,
                                task_type=task_type,
                            )
                        except Exception:
                            pass
                else:
                    exec_error = Exception(f"阶段 '{stage}' 不支持流式执行")
            except Exception as e:
                exec_error = e
            finally:
                msg_queue.put(None)  # 哨兵值表示执行完成

        # 启动异步执行任务
        loop = asyncio.get_event_loop()
        task = loop.create_task(_run_stage_execution())

        # 从队列中读取事件并发送 SSE
        while True:
            try:
                event = await loop.run_in_executor(None, msg_queue.get, True, 1.0)
            except queue.Empty:
                if task.done():
                    break
                continue

            if event is None:  # 哨兵值，执行完成
                break

            yield _sse(event.type.value, event.content)

        # 确保 task 完成
        await task

        # 提交确认响应
        try:
            status_map = {
                "confirmed": ConfirmationStatus.CONFIRMED,
                "modified": ConfirmationStatus.MODIFIED,
            }
            confirmation_manager.submit_response(
                point_id=confirmation_id,
                status=status_map.get(status, ConfirmationStatus.CONFIRMED),
                modifications=None,
            )
        except Exception:
            pass

        # 发送最终结果
        if exec_error:
            yield _sse("error", str(exec_error))
        else:
            result_payload = {
                "success": exec_result.get("success", False),
                "stage": stage,
                "status": status,
                "execution": exec_result,
            }
            yield _sse("done", json.dumps(result_payload, ensure_ascii=False, default=str))

        # 清理 stream callback
        try:
            agent.set_stream_callback(None)
        except Exception:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
