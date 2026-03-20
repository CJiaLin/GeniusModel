"""
FastAPI 主应用

提供 AutoML 工作流的 API 接口
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import asyncio
from datetime import datetime

from automl_react.agents import (
    DataCleaningAgent,
    FeatureEngineeringAgent,
    ModelTrainingAgent
)
from automl_react.evaluation import ModelEvaluator
from automl_react.report import PipelineGenerator, ReportGenerator
from automl_react.workflow import WorkflowState, WorkflowStage
from automl_react.confirmation import ConfirmationManager, ConfirmationStatus
from automl_react.assets import get_asset_manager
from automl_react.logger import get_llm_logger
from automl_react.skills_loader import get_skill_loader
from automl_react.config import get_config_loader


# 创建 FastAPI 应用
app = FastAPI(
    title="AutoML API",
    description="交互式 AutoML 工作流 API",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储会话状态
_sessions: Dict[str, Dict[str, Any]] = {}


# 请求模型
class StartWorkflowRequest(BaseModel):
    session_id: str
    data_path: str
    target_column: str
    task_type: str = "classification"
    model: str = "claude-sonnet-4-20250514-thinking"


class UserConfirmationRequest(BaseModel):
    session_id: str
    confirmation_id: str
    status: str  # confirmed, modified, skipped, rejected
    modifications: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = "claude-sonnet-4-20250514-thinking"


# 辅助函数
def get_session(session_id: str) -> Dict[str, Any]:
    """获取会话状态"""
    if session_id not in _sessions:
        _sessions[session_id] = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "workflow_state": None,
            "confirmation_manager": None,
            "agents": {},
            "context": {}
        }
    return _sessions[session_id]


class LLMClientError(Exception):
    """LLM 客户端错误"""
    pass


def create_llm_client(model: str = None):
    """
    创建 LLM 客户端
    
    根据配置创建真实的 LLM 客户端，如果失败则抛出明确的错误
    """
    errors = []
    
    # 首先尝试从配置加载
    try:
        config_loader = get_config_loader()
        llm_config = config_loader.get_llm_config(model)
        
        provider = llm_config.get("provider", "openai")
        model_name = llm_config.get("model_name", model or "gpt-4")
        
        if provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
                api_key = llm_config.get("api_key")
                if not api_key:
                    raise LLMClientError(
                        f"Anthropic API 密钥未配置。\n"
                        f"请设置环境变量 ANTHROPIC_API_KEY 或在配置文件中指定 api_key。\n"
                        f"当前模型: {model_name}"
                    )
                return ChatAnthropic(
                    model=model_name,
                    temperature=llm_config.get("temperature", 0.1),
                    max_tokens=llm_config.get("max_tokens", 4096),
                    api_key=api_key
                )
            except ImportError as e:
                errors.append(f"langchain_anthropic 未安装: {e}")
        
        elif provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                api_key = llm_config.get("api_key")
                if not api_key:
                    raise LLMClientError(
                        f"OpenAI API 密钥未配置。\n"
                        f"请设置环境变量 OPENAI_API_KEY 或在配置文件中指定 api_key。\n"
                        f"当前模型: {model_name}"
                    )
                return ChatOpenAI(
                    model=model_name,
                    temperature=llm_config.get("temperature", 0.1),
                    max_tokens=llm_config.get("max_tokens", 4096),
                    api_key=api_key,
                    base_url=llm_config.get("base_url")
                )
            except ImportError as e:
                errors.append(f"langchain_openai 未安装: {e}")
        
    except LLMClientError:
        raise
    except Exception as e:
        errors.append(f"从配置创建 LLM 客户端失败: {e}")
    
    # 如果都失败了，抛出详细的错误信息
    error_msg = "无法创建 LLM 客户端。\n\n"
    error_msg += "可能的原因:\n"
    error_msg += "1. 缺少必要的 Python 包:\n"
    error_msg += "   - pip install langchain-openai  # 使用 OpenAI\n"
    error_msg += "   - pip install langchain-anthropic  # 使用 Claude\n"
    error_msg += "2. API 密钥未配置:\n"
    error_msg += "   - 设置环境变量 OPENAI_API_KEY 或 ANTHROPIC_API_KEY\n"
    error_msg += "   - 或在 llm_config.yaml 中配置 api_key\n"
    error_msg += "3. 配置文件错误:\n"
    error_msg += "   - 检查 automl_react/config/llm_config.yaml 配置\n\n"
    error_msg += "详细错误:\n"
    for err in errors:
        error_msg += f"  - {err}\n"
    
    raise LLMClientError(error_msg)


# API 路由
@app.get("/")
async def root():
    """根路径"""
    return {"message": "AutoML API", "version": "1.0.0"}


@app.post("/workflow/start")
async def start_workflow(request: StartWorkflowRequest):
    """
    启动工作流
    
    初始化工作流状态和相关组件
    """
    session = get_session(request.session_id)
    
    # 创建工作流状态
    workflow_state = WorkflowState(
        session_id=request.session_id,
        initial_stage=WorkflowStage.DATA_UPLOAD
    )
    workflow_state.set_context("data_path", request.data_path)
    workflow_state.set_context("target_column", request.target_column)
    workflow_state.set_context("task_type", request.task_type)
    workflow_state.set_context("model", request.model)
    
    session["workflow_state"] = workflow_state
    session["confirmation_manager"] = ConfirmationManager()
    
    # 创建 LLM 客户端
    try:
        llm = create_llm_client(request.model)
    except LLMClientError as e:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "LLM 服务不可用",
                "detail": str(e),
                "session_id": request.session_id
            }
        )
    
    # 创建 Agents
    session["agents"]["cleaning"] = DataCleaningAgent(
        llm=llm,
        session_id=request.session_id
    )
    session["agents"]["feature"] = FeatureEngineeringAgent(
        llm=llm,
        session_id=request.session_id
    )
    session["agents"]["model"] = ModelTrainingAgent(
        llm=llm,
        session_id=request.session_id
    )
    
    # 保存状态
    workflow_state.save()
    
    return {
        "success": True,
        "session_id": request.session_id,
        "current_stage": workflow_state.current_stage.value,
        "message": "工作流已启动"
    }


@app.get("/workflow/{session_id}/status")
async def get_workflow_status(session_id: str):
    """获取工作流状态"""
    session = get_session(session_id)
    workflow_state = session.get("workflow_state")
    
    if not workflow_state:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return {
        "session_id": session_id,
        "current_stage": workflow_state.current_stage.value,
        "context": workflow_state.context,
        "history": workflow_state.history[-10:] if workflow_state.history else []
    }


@app.post("/workflow/{session_id}/stage/{stage}/run")
async def run_stage(session_id: str, stage: str, background_tasks: BackgroundTasks):
    """
    运行指定阶段
    
    阶段: data_cleaning, feature_engineering, model_training
    """
    session = get_session(session_id)
    workflow_state = session.get("workflow_state")
    
    if not workflow_state:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 更新工作流状态
    try:
        workflow_state.transition_to(WorkflowStage(stage))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无效的阶段转换: {e}")
    
    # 获取上下文
    data_path = workflow_state.get_context("data_path")
    target_column = workflow_state.get_context("target_column")
    task_type = workflow_state.get_context("task_type")
    
    # 获取确认管理器
    confirmation_manager = session.get("confirmation_manager")
    if not confirmation_manager:
        raise HTTPException(status_code=500, detail="确认管理器未初始化")
    
    # 根据阶段执行相应操作
    if stage == "data_analysis":
        agent = session["agents"].get("automl")
        if agent:
            try:
                result = agent.analyze(data_path)
                
                # 保存分析结果到资产
                asset_manager = get_asset_manager(session_id=session_id)
                asset_manager.save_data(
                    data=str(result.get("answer", "")),
                    filename="data_analysis_result.md",
                    asset_type="analysis",
                    metadata={
                        "stage": "data_analysis",
                        "data_path": data_path,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                return {
                    "success": True,
                    "stage": stage,
                    "analysis": result.get("answer", ""),
                    "requires_confirmation": False
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    elif stage == "data_cleaning":
        agent = session["agents"].get("cleaning")
        if agent:
            try:
                result = agent.generate_cleaning_plan(data_path)
                
                # 创建确认点
                confirmation_point = confirmation_manager.add_confirmation_point(
                    stage="data_cleaning",
                    proposal_content=result
                )
                
                return {
                    "success": True,
                    "stage": stage,
                    "proposal": result,
                    "requires_confirmation": True,
                    "confirmation_id": confirmation_point.id
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    elif stage == "feature_engineering":
        agent = session["agents"].get("feature")
        if agent:
            try:
                result = agent.generate_feature_plan(data_path, target_column, task_type)
                
                # 创建确认点
                confirmation_point = confirmation_manager.add_confirmation_point(
                    stage="feature_engineering",
                    proposal_content=result
                )
                
                return {
                    "success": True,
                    "stage": stage,
                    "proposal": result,
                    "requires_confirmation": True,
                    "confirmation_id": confirmation_point.id
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

    elif stage == "model_training":
        agent = session["agents"].get("model")
        if agent:
            try:
                result = agent.generate_model_plan(data_path, target_column, task_type)

                # 创建确认点
                confirmation_point = confirmation_manager.add_confirmation_point(
                    stage="model_training",
                    proposal_content=result
                )

                return {
                    "success": True,
                    "stage": stage,
                    "proposal": result,
                    "requires_confirmation": True,
                    "confirmation_id": confirmation_point.id
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    return {"success": False, "error": "未知的阶段"}


@app.post("/confirmation/submit")
async def submit_confirmation(request: UserConfirmationRequest):
    """
    提交用户确认并执行相应操作
    
    根据确认的阶段执行代码生成和执行
    """
    from automl_react.utils import execute_code_safely
    
    session = get_session(request.session_id)
    confirmation_manager = session.get("confirmation_manager")
    workflow_state = session.get("workflow_state")
    
    if not confirmation_manager:
        raise HTTPException(status_code=404, detail="确认管理器不存在")
    
    # 转换状态字符串为枚举
    status_map = {
        "confirmed": ConfirmationStatus.CONFIRMED,
        "modified": ConfirmationStatus.MODIFIED,
        "skipped": ConfirmationStatus.SKIPPED,
        "rejected": ConfirmationStatus.REJECTED
    }
    
    status = status_map.get(request.status)
    if not status:
        raise HTTPException(status_code=400, detail="无效的确认状态")
    
    # 获取确认点信息
    current_confirmation = confirmation_manager.current
    if not current_confirmation:
        # 如果没有当前确认点，尝试从队列中获取第一个
        pending_points = confirmation_manager.get_pending_points()
        if pending_points:
            current_confirmation = pending_points[0]
        else:
            raise HTTPException(status_code=404, detail="没有待处理的确认")
    
    stage = current_confirmation.stage
    data_path = workflow_state.get_context("data_path")
    target_column = workflow_state.get_context("target_column")
    task_type = workflow_state.get_context("task_type")
    
    # 执行结果
    execution_result = None
    
    # 如果用户确认或修改，执行相应操作
    if status in [ConfirmationStatus.CONFIRMED, ConfirmationStatus.MODIFIED]:
        try:
            if stage == "data_cleaning":
                execution_result = await execute_data_cleaning(
                    session, data_path, request.modifications
                )
            elif stage == "feature_engineering":
                execution_result = await execute_feature_engineering(
                    session, data_path, target_column, task_type, request.modifications
                )
            elif stage == "model_training":
                execution_result = await execute_model_training(
                    session, data_path, target_column, task_type, request.modifications
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
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def execute_data_cleaning(session: Dict, data_path: str, modifications: Optional[str] = None) -> Dict:
    """执行数据清洗"""
    agent = session["agents"].get("cleaning")
    if not agent:
        raise ValueError("数据清洗 Agent 不存在")

    # 先生成清洗方案（如果还没有）
    if not agent.cleaning_plan:
        agent.generate_cleaning_plan(data_path)

    # 生成清洗代码并执行（新方法会自动验证和执行）
    code = agent.generate_cleaning_code(modifications)

    # 执行代码
    result = agent.execute_cleaning(code)

    # 构建执行结果
    execution_result = {
        "success": result.get("success", False),
        "cleaned_data_path": result.get("cleaned_data_path"),
        "original_path": result.get("original_path"),
        "timestamp": result.get("timestamp"),
        "stage": "data_cleaning"
    }

    # 保存结果到资产（用于报告生成）
    asset_manager = get_asset_manager(session_id=session.get("session_id", "default"))
    asset_manager.save_data(
        data=json.dumps(execution_result, ensure_ascii=False, indent=2),
        filename="cleaning_result.json",
        asset_type="cleaned_data",
        metadata=execution_result
    )

    return execution_result


async def execute_feature_engineering(
    session: Dict,
    data_path: str,
    target_column: str,
    task_type: str,
    modifications: Optional[str] = None
) -> Dict:
    """执行特征工程"""
    agent = session["agents"].get("feature")
    if not agent:
        raise ValueError("特征工程 Agent 不存在")

    # 先生成特征工程方案（如果还没有）
    if not agent.feature_plan:
        agent.generate_feature_plan(data_path, target_column, task_type)

    # 生成特征工程代码并执行
    code = agent.generate_feature_code(modifications)

    # 执行代码
    result = agent.execute_feature_engineering(code)

    # 构建执行结果
    execution_result = {
        "success": result.get("success", False),
        "features_data_path": result.get("features_data_path"),
        "original_path": result.get("original_path"),
        "timestamp": result.get("timestamp"),
        "stage": "feature_engineering"
    }

    # 保存结果到资产（用于报告生成）
    asset_manager = get_asset_manager(session_id=session.get("session_id", "default"))
    asset_manager.save_data(
        data=json.dumps(execution_result, ensure_ascii=False, indent=2),
        filename="feature_engineering_result.json",
        asset_type="features",
        metadata=execution_result
    )

    return execution_result


async def execute_model_training(
    session: Dict,
    data_path: str,
    target_column: str,
    task_type: str,
    modifications: Optional[str] = None
) -> Dict:
    """执行模型训练"""
    agent = session["agents"].get("model")
    if not agent:
        raise ValueError("模型训练 Agent 不存在")

    # 先生成建模方案（如果还没有）
    if not agent.model_plan:
        agent.generate_model_plan(data_path, target_column, task_type)

    # 生成训练代码并执行
    code = agent.generate_model_code(modifications)

    # 执行代码
    result = agent.execute_model_training(code)

    # 构建执行结果
    execution_result = {
        "success": result.get("success", False),
        "model_path": result.get("model_path"),
        "metrics": result.get("metrics", {}),
        "timestamp": result.get("timestamp"),
        "stage": "model_training"
    }

    # 保存结果到资产（用于报告生成）
    asset_manager = get_asset_manager(session_id=session.get("session_id", "default"))
    asset_manager.save_data(
        data=json.dumps(execution_result, ensure_ascii=False, indent=2),
        filename="model_training_result.json",
        asset_type="models",
        metadata=execution_result
    )

    return execution_result


@app.get("/confirmation/{session_id}/pending")
async def get_pending_confirmation(session_id: str):
    """获取待处理的确认点"""
    session = get_session(session_id)
    confirmation_manager = session.get("confirmation_manager")
    
    if not confirmation_manager:
        raise HTTPException(status_code=404, detail="确认管理器不存在")
    
    current = confirmation_manager.get_current_confirmation()
    
    if not current:
        return {"has_pending": False}
    
    return {
        "has_pending": True,
        "confirmation": current.to_dict()
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """对话接口"""
    session = get_session(request.session_id)
    
    # 创建 LLM 客户端
    llm = create_llm_client(request.model)
    
    if not llm:
        raise HTTPException(status_code=500, detail="LLM 客户端创建失败")
    
    # 这里可以实现更复杂的对话逻辑
    # 简化处理：直接返回 LLM 响应
    try:
        response = llm.invoke(request.message)
        return {
            "success": True,
            "response": response.content if hasattr(response, 'content') else str(response)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/stream")
async def chat_stream(session_id: str, message: str, model: str = None):
    """流式对话接口"""
    async def event_generator():
        try:
            # 发送开始消息
            yield f"data: {json.dumps({'type': 'start', 'content': '开始处理...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # 创建 LLM 客户端
            llm = create_llm_client(model)
            
            if not llm:
                yield f"data: {json.dumps({'type': 'error', 'content': 'LLM 客户端创建失败'})}\n\n"
                return
            
            # 调用 LLM
            response = llm.invoke(message)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 模拟流式输出
            chunks = content.split("\n")
            for chunk in chunks:
                if chunk.strip():
                    newline = "\n"
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk + newline})}\n\n"
                    await asyncio.sleep(0.05)
            
            # 发送完成消息
            done_msg = "处理完成"
            yield f"data: {json.dumps({'type': 'done', 'content': done_msg})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/assets/{session_id}/{asset_type}/{filename}")
async def download_asset(session_id: str, asset_type: str, filename: str):
    """下载资产文件"""
    asset_manager = get_asset_manager(session_id=session_id)
    asset_path = asset_manager.get_asset(asset_type, filename)
    
    if not asset_path:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=asset_path,
        filename=filename,
        media_type="application/octet-stream"
    )


@app.get("/assets/{session_id}/list")
async def list_assets(session_id: str):
    """列出所有资产"""
    asset_manager = get_asset_manager(session_id=session_id)
    urls = asset_manager.get_all_download_urls()
    
    return {
        "session_id": session_id,
        "assets": urls
    }


@app.get("/logs/{session_id}/llm")
async def get_llm_logs(session_id: str, limit: int = 100):
    """获取 LLM 调用日志"""
    llm_logger = get_llm_logger(session_id=session_id)
    logs = llm_logger.get_logs(limit=limit)
    
    return {
        "session_id": session_id,
        "logs": [log.to_dict() for log in logs]
    }


@app.get("/skills/list")
async def list_skills():
    """列出所有可用的 skills"""
    skill_loader = get_skill_loader()
    skills = skill_loader.scan_skills()
    
    return {
        "skills": skills
    }


@app.get("/skills/{skill_name}/content")
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


@app.post("/report/generate")
async def generate_report(session_id: str):
    """生成报告"""
    session = get_session(session_id)
    workflow_state = session.get("workflow_state")
    
    if not workflow_state:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 获取上下文
    data_path = workflow_state.get_context("data_path")
    target_column = workflow_state.get_context("target_column")
    task_type = workflow_state.get_context("task_type")
    
    # 生成报告
    report_generator = ReportGenerator(session_id=session_id)
    
    try:
        report = report_generator.generate_report(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type
        )
        
        # 同时生成 HTML 版本
        html_report = report_generator.export_to_html(report)
        
        return {
            "success": True,
            "message": "报告已生成",
            "downloads": {
                "markdown": f"/assets/{session_id}/reports/modeling_report.md",
                "html": f"/assets/{session_id}/reports/modeling_report.html"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/generate")
async def generate_pipeline(session_id: str):
    """生成全流程脚本"""
    session = get_session(session_id)
    workflow_state = session.get("workflow_state")
    
    if not workflow_state:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 获取上下文
    data_path = workflow_state.get_context("data_path")
    target_column = workflow_state.get_context("target_column")
    task_type = workflow_state.get_context("task_type")
    
    # 生成脚本
    pipeline_generator = PipelineGenerator(session_id=session_id)
    
    try:
        script = pipeline_generator.generate_pipeline_script(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type
        )
        
        return {
            "success": True,
            "message": "全流程脚本已生成",
            "download_url": f"/assets/{session_id}/code/pipeline.py"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
