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
from automl_react.agents.data_analysis_agent import DataAnalysisAgent
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
    task_description: str = ""  # 用户输入的建模背景和要求


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
    """获取会话状态（支持从文件恢复）"""
    if session_id not in _sessions:
        # 尝试从文件恢复 session
        session_dir = Path("assets") / session_id
        state_file = session_dir / "state" / "workflow_state.json"
        
        if state_file.exists():
            print(f"[API] 从文件恢复 session: {session_id}")
            try:
                # 使用 WorkflowState.load 方法加载状态
                workflow_state = WorkflowState.load(session_id)
                
                _sessions[session_id] = {
                    "session_id": session_id,
                    "created_at": workflow_state.history[0].get("timestamp", datetime.now().isoformat()) if workflow_state.history else datetime.now().isoformat(),
                    "workflow_state": workflow_state,
                    "confirmation_manager": None,  # 需要重新创建
                    "agents": {},  # 需要重新创建
                    "context": workflow_state.context
                }
                print(f"[API] Session 恢复成功，当前阶段: {workflow_state.current_stage}")
            except Exception as e:
                print(f"[API] Session 恢复失败: {e}，创建新 session")
                _sessions[session_id] = {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "workflow_state": None,
                    "confirmation_manager": None,
                    "agents": {},
                    "context": {}
                }
        else:
            # 创建新 session
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
    import shutil
    
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
    workflow_state.set_context("task_description", request.task_description)  # 保存用户的建模背景和要求
    
    session["workflow_state"] = workflow_state
    session["confirmation_manager"] = ConfirmationManager()
    
    # 将初始数据复制到 session 目录
    asset_manager = get_asset_manager(session_id=request.session_id)
    original_data_path = Path(request.data_path)
    if original_data_path.exists():
        session_data_path = asset_manager.session_dir / "data" / "original_data.csv"
        session_data_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_data_path, session_data_path)
        print(f"[API] 初始数据已复制到: {session_data_path}")
    
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
    from automl_react.agents import DataExplorationAgent
    session["agents"]["exploration"] = DataExplorationAgent(
        llm=llm,
        session_id=request.session_id
    )
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
    
    # 更新工作流状态（如果当前阶段不是目标阶段）
    target_stage = WorkflowStage(stage)
    if workflow_state.current_stage != target_stage:
        try:
            workflow_state.transition_to(target_stage)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的阶段转换: {e}")
    
    # 获取上下文
    data_path = workflow_state.get_context("data_path")
    target_column = workflow_state.get_context("target_column")
    task_type = workflow_state.get_context("task_type")
    
    # 获取或创建确认管理器
    confirmation_manager = session.get("confirmation_manager")
    if not confirmation_manager:
        # Session 恢复后需要重新创建确认管理器
        confirmation_manager = ConfirmationManager()
        session["confirmation_manager"] = confirmation_manager
        print(f"[API] 已重新创建确认管理器")
    
    # 根据阶段执行相应操作
    if stage == "data_exploration":
        agent = session["agents"].get("exploration")
        if not agent:
            return {"success": False, "error": "DataExploration Agent 未初始化"}
        
        try:
            # 获取用户的建模背景和要求
            task_description = workflow_state.get_context("task_description", "")
            
            print(f"[API] ========== 数据探索性分析阶段开始 ==========")
            
            # 读取清洗后的数据路径
            asset_manager = get_asset_manager(session_id=session_id)
            import json
            cleaning_result_json = asset_manager.read_asset("cleaning", "cleaning_result.json")
            cleaned_data_path = None
            if cleaning_result_json:
                try:
                    cleaning_data = json.loads(cleaning_result_json)
                    cleaned_data_path = cleaning_data.get("cleaned_data_path")
                except:
                    pass
            
            # 如果没有清洗后的数据，使用原始数据
            if not cleaned_data_path:
                cleaned_data_path = data_path
                print(f"[API] 使用原始数据: {cleaned_data_path}")
            else:
                print(f"[API] 使用清洗后的数据: {cleaned_data_path}")
            
            result = agent.explore(
                cleaned_data_path,
                target_column=target_column,
                task_type=task_type,
                task_description=task_description
            )
            print(f"[API] explore 返回结果: success={result.get('success')}, answer长度={len(result.get('answer', ''))}")
            
            # 保存探索性分析结果到资产
            asset_manager.save_data(
                data=str(result.get("answer", "")),
                filename="data_exploration_result.md",
                asset_type="exploration",
                metadata={
                    "stage": "data_exploration",
                    "data_path": cleaned_data_path,
                    "task_description": task_description,
                    "timestamp": datetime.now().isoformat()
                }
            )
            print(f"[API] 探索性分析结果已保存到: exploration/data_exploration_result.md")
            print(f"[API] ========== 数据探索性分析阶段完成 ==========")
            
            return {
                "success": True,
                "stage": stage,
                "exploration": result.get("answer", ""),
                "requires_confirmation": False
            }
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[API] 数据探索性分析错误: {error_detail}")
            return {"success": False, "error": error_detail}
    
    elif stage == "data_cleaning":
        agent = session["agents"].get("cleaning")
        if agent:
            try:
                print(f"[API] ========== 数据清洗阶段开始 ==========")
                
                # 获取用户的建模背景和要求
                task_description = workflow_state.get_context("task_description", "")
                if task_description:
                    print(f"[API] 用户建模背景: {task_description[:100]}...")
                
                # 数据清洗阶段自己进行数据质量分析
                # 生成清洗方案（包含数据质量分析）
                result = agent.generate_cleaning_plan(
                    data_path, 
                    task_description=task_description
                )
                print(f"[API] 清洗方案生成完成，长度: {len(result)} 字符")
                
                # 保存清洗方案到 session 目录
                asset_manager = get_asset_manager(session_id=session_id)
                asset_manager.save_data(
                    data=result,
                    filename="cleaning_plan.md",
                    asset_type="cleaning",
                    metadata={
                        "stage": "data_cleaning",
                        "data_path": data_path,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                print(f"[API] 清洗方案已保存到: cleaning/cleaning_plan.md")
                print(f"[API] ========== 数据清洗阶段完成 ==========")
                
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
        if not agent:
            # Session 恢复后需要重新创建 agent
            print(f"[API] 重新创建特征工程 Agent...")
            model = workflow_state.get_context("model", "kimi-k2.5")
            llm = create_llm_client(model)
            asset_manager = get_asset_manager(session_id=session_id)
            
            # 获取清洗后的数据路径
            cleaned_data_path = data_path
            cleaning_result_json = asset_manager.read_asset("cleaning", "cleaning_result.json")
            if cleaning_result_json:
                try:
                    cleaning_data = json.loads(cleaning_result_json)
                    cleaned_data_path = cleaning_data.get("cleaned_data_path", data_path)
                except:
                    pass
            
            agent = FeatureEngineeringAgent(
                llm=llm,
                asset_manager=asset_manager,
                data_path=cleaned_data_path,
                target_column=target_column,
                task_type=task_type
            )
            session["agents"]["feature"] = agent
            print(f"[API] 特征工程 Agent 已重新创建")
        
        if agent:
            try:
                print(f"[API] ========== 特征工程阶段开始 ==========")
                
                # 获取用户的建模背景和要求
                task_description = workflow_state.get_context("task_description", "")
                if task_description:
                    print(f"[API] 用户建模背景: {task_description[:100]}...")
                
                # 读取前一阶段的结果
                asset_manager = get_asset_manager(session_id=session_id)
                # 读取探索性分析报告（新流程：exploration 目录）
                exploration_result = asset_manager.read_asset("exploration", "data_exploration_result.md")
                # 读取清洗后的数据路径
                import json
                cleaning_result_json = asset_manager.read_asset("cleaning", "cleaning_result.json")
                cleaned_data_path = None
                if cleaning_result_json:
                    try:
                        cleaning_data = json.loads(cleaning_result_json)
                        cleaned_data_path = cleaning_data.get("cleaned_data_path")
                        if cleaned_data_path:
                            print(f"[API] 使用清洗后的数据: {cleaned_data_path}")
                    except:
                        pass
                
                result = agent.generate_feature_plan(
                    data_path, 
                    target_column, 
                    task_type,
                    analysis_result=exploration_result,  # 使用探索性分析报告
                    cleaned_data_path=cleaned_data_path,
                    task_description=task_description
                )
                print(f"[API] 特征工程方案生成完成，长度: {len(result)} 字符")
                
                # 保存特征工程方案到资产
                asset_manager.save_data(
                    data=result,
                    filename="feature_engineering_plan.md",
                    asset_type="features",
                    metadata={
                        "stage": "feature_engineering",
                        "data_path": data_path,
                        "has_exploration_input": exploration_result is not None,
                        "has_cleaned_data_input": cleaned_data_path is not None,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                print(f"[API] 特征工程方案已保存到: features/feature_engineering_plan.md")
                print(f"[API] ========== 特征工程阶段完成 ==========")
                
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
                print(f"[API] ========== 模型训练阶段开始 ==========")
                
                # 获取用户的建模背景和要求
                task_description = workflow_state.get_context("task_description", "")
                if task_description:
                    print(f"[API] 用户建模背景: {task_description[:100]}...")
                
                # 读取所有前一阶段的结果
                asset_manager = get_asset_manager(session_id=session_id)
                exploration_result = asset_manager.read_asset("exploration", "data_exploration_result.md")
                feature_metrics_report = asset_manager.read_asset("features", "feature_metrics_report.md")
                
                # 读取特征工程后的数据路径
                import json
                feature_result_json = asset_manager.read_asset("features", "feature_engineering_result.json")
                features_data_path = None
                if feature_result_json:
                    try:
                        feature_data = json.loads(feature_result_json)
                        features_data_path = feature_data.get("features_data_path")
                        if features_data_path:
                            print(f"[API] 使用特征工程后的数据: {features_data_path}")
                    except:
                        pass
                
                result = agent.generate_model_plan(
                    data_path, 
                    target_column, 
                    task_type,
                    exploration_report=exploration_result,
                    feature_metrics_report=feature_metrics_report,
                    features_data_path=features_data_path,
                    task_description=task_description
                )
                print(f"[API] 模型训练方案生成完成，长度: {len(result)} 字符")
                
                # 保存模型训练方案到资产
                asset_manager.save_data(
                    data=result,
                    filename="model_training_plan.md",
                    asset_type="models",
                    metadata={
                        "stage": "model_training",
                        "data_path": data_path,
                        "has_exploration_input": exploration_result is not None,
                        "has_feature_metrics_input": feature_metrics_report is not None,
                        "features_data_path": features_data_path,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                print(f"[API] 模型训练方案已保存到: models/model_training_plan.md")
                print(f"[API] ========== 模型训练阶段完成 ==========")

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
    next_confirmation_point = None
    
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
                # 特征工程完成后，提示用户可选执行特征评估（非强制）
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
                    session, data_path, target_column, task_type, request.modifications
                )
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


async def execute_data_cleaning(session: Dict, data_path: str, modifications: Optional[str] = None) -> Dict:
    """执行数据清洗（CodeAct 模式）"""
    import shutil
    
    print(f"[API] ========== 开始执行数据清洗 ==========")
    print(f"[API] 数据路径: {data_path}")
    
    agent = session["agents"].get("cleaning")
    if not agent:
        raise ValueError("数据清洗 Agent 不存在")

    session_id = session.get("session_id", "default")
    asset_manager = get_asset_manager(session_id=session_id)

    # 先生成清洗方案（如果还没有）
    if not agent.cleaning_plan:
        print(f"[API] 清洗方案未生成，开始生成...")
        agent.generate_cleaning_plan(data_path)
    else:
        print(f"[API] 清洗方案已存在，长度: {len(agent.cleaning_plan)} 字符")

    # 使用 CodeAct 模式生成并执行代码
    print(f"[API] 开始生成并执行清洗代码（CodeAct 模式）...")
    try:
        # CodeAct 模式：generate_cleaning_code 已经包含执行验证
        code = agent.generate_cleaning_code(modifications)
        print(f"[API] 清洗代码生成并执行完成，长度: {len(code) if code else 0} 字符")
    except Exception as e:
        print(f"[API] 清洗代码生成执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # 检查清洗后的数据文件
    final_cleaned_path = agent.cleaned_data_path
    import os
    file_exists = os.path.exists(final_cleaned_path)
    
    if not file_exists:
        print(f"[API] 警告: 清洗后的数据文件不存在: {final_cleaned_path}")
    
    # 构建执行结果
    execution_result = {
        "success": file_exists,
        "cleaned_data_path": final_cleaned_path,
        "original_path": data_path,
        "timestamp": datetime.now().isoformat(),
        "stage": "data_cleaning"
    }

    # 保存结果到资产（用于报告生成）
    asset_manager.save_data(
        data=json.dumps(execution_result, ensure_ascii=False, indent=2),
        filename="cleaning_result.json",
        asset_type="cleaning",
        metadata=execution_result
    )
    
    print(f"[API] ========== 数据清洗执行完成 ==========")

    return execution_result


async def execute_feature_engineering(
    session: Dict,
    data_path: str,
    target_column: str,
    task_type: str,
    modifications: Optional[str] = None
) -> Dict:
    """执行特征工程（CodeAct 模式）"""
    import shutil
    
    print(f"[API] ========== 开始执行特征工程 ==========")
    print(f"[API] 数据路径: {data_path}")
    
    agent = session["agents"].get("feature")
    if not agent:
        raise ValueError("特征工程 Agent 不存在")

    session_id = session.get("session_id", "default")
    asset_manager = get_asset_manager(session_id=session_id)

    # 先生成特征工程方案（如果还没有）
    if not agent.feature_plan:
        print(f"[API] 特征工程方案未生成，开始生成...")
        agent.generate_feature_plan(data_path, target_column, task_type)
    else:
        print(f"[API] 特征工程方案已存在，长度: {len(agent.feature_plan)} 字符")

    # 使用 CodeAct 模式生成并执行代码
    print(f"[API] 开始生成并执行特征工程代码（CodeAct 模式）...")
    try:
        # CodeAct 模式：generate_feature_code 已经包含执行验证
        code = agent.generate_feature_code(modifications)
        print(f"[API] 特征工程代码生成并执行完成，长度: {len(code) if code else 0} 字符")
    except Exception as e:
        print(f"[API] 特征工程代码生成执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # 检查特征工程后的数据文件（使用 agent 中定义的路径）
    features_data_path = agent.features_data_path if hasattr(agent, 'features_data_path') else None
    import os
    file_exists = features_data_path and os.path.exists(features_data_path)
    if not file_exists:
        print(f"[API] 警告: 特征工程后的数据文件不存在: {features_data_path}")

    # 构建执行结果
    execution_result = {
        "success": file_exists,
        "features_data_path": features_data_path if file_exists else None,
        "original_path": data_path,
        "timestamp": datetime.now().isoformat(),
        "stage": "feature_engineering",
        "evaluation_available": file_exists,
        "evaluation_required_confirmation": file_exists
    }

    # 保存结果到资产（用于报告生成）
    asset_manager.save_data(
        data=json.dumps(execution_result, ensure_ascii=False, indent=2),
        filename="feature_engineering_result.json",
        asset_type="features",
        metadata=execution_result
    )
    
    print(f"[API] ========== 特征工程执行完成 ==========")
    
    return execution_result


async def execute_feature_evaluation(session: Dict, modifications: Optional[str] = None) -> Dict:
    """执行特征评估（可解释性与可靠性分析）"""
    print(f"[API] ========== 开始执行特征评估 ==========")

    agent = session["agents"].get("feature")
    if not agent:
        raise ValueError("特征工程 Agent 不存在")

    # 执行特征评估（LLM 生成评估代码并执行 + 生成分析报告）
    result = agent.calculate_feature_metrics(modifications=modifications)

    # 保存结果到资产（用于报告生成）
    asset_manager = get_asset_manager(session_id=session.get("session_id", "default"))
    payload = {
        "success": result.get("success", False),
        "metrics_result_path": result.get("metrics_result_path"),
        "metrics_report_path": result.get("metrics_report_path"),
        "features_data_path": result.get("features_data_path"),
        "timestamp": result.get("timestamp", datetime.now().isoformat()),
        "stage": "feature_evaluation"
    }
    asset_manager.save_data(
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        filename="feature_evaluation_result.json",
        asset_type="features",
        metadata=payload
    )

    print(f"[API] ========== 特征评估执行完成 ==========")
    return payload


async def execute_model_training(
    session: Dict,
    data_path: str,
    target_column: str,
    task_type: str,
    modifications: Optional[str] = None
) -> Dict:
    """执行模型训练"""
    import json

    agent = session["agents"].get("model")
    if not agent:
        raise ValueError("模型训练 Agent 不存在")

    asset_manager = get_asset_manager(session_id=session.get("session_id", "default"))

    feature_result_json = asset_manager.read_asset("features", "feature_engineering_result.json")
    features_data_path = None
    if feature_result_json:
        try:
            feature_data = json.loads(feature_result_json)
            features_data_path = feature_data.get("features_data_path")
        except Exception:
            features_data_path = None

    exploration_report = asset_manager.read_asset("exploration", "data_exploration_result.md")
    feature_metrics_report = asset_manager.read_asset("features", "feature_metrics_report.md")

    # 先生成建模方案（如果还没有）
    if not agent.model_plan:
        agent.generate_model_plan(
            data_path,
            target_column,
            task_type,
            exploration_report=exploration_report,
            feature_metrics_report=feature_metrics_report,
            features_data_path=features_data_path
        )

    # 生成训练代码并执行
    code = agent.generate_model_code(modifications)

    # 执行代码
    result = agent.execute_model_training(code)

    # 构建执行结果
    execution_result = {
        "success": result.get("success", False),
        "model_path": result.get("model_path"),
        "train_split_path": result.get("train_split_path"),
        "test_split_path": result.get("test_split_path"),
        "training_summary_path": result.get("training_summary_path"),
        "metrics": result.get("metrics", {}),
        "selected_feature_names": result.get("selected_feature_names", []),
        "features_data_path": result.get("data_path"),
        "artifact_status": result.get("artifact_status", {}),
        "timestamp": result.get("timestamp"),
        "stage": "model_training"
    }

    # 保存结果到资产（用于报告生成）
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
    # 简化处理：使用流式输出
    try:
        print(f"[Chat] 开始流式对话...")
        full_response = ""
        
        try:
            for chunk in llm.stream(request.message):
                if chunk.content:
                    content = chunk.content
                    full_response += content
                    print(content, end="", flush=True)
        except Exception as e:
            # 如果流式输出失败，回退到同步调用
            print(f"\n[Chat] 流式输出失败，回退到同步调用: {e}")
            response = llm.invoke(request.message)
            full_response = response.content if hasattr(response, 'content') else str(response)
        
        print()  # 换行
        
        return {
            "success": True,
            "response": full_response
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
            
            # 使用真正的流式输出
            try:
                for chunk in llm.stream(message):
                    if chunk.content:
                        content = chunk.content
                        yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                        await asyncio.sleep(0.01)  # 小延迟确保数据发送
            except Exception as e:
                # 如果流式输出失败，回退到同步调用
                yield f"data: {json.dumps({'type': 'error', 'content': f'流式输出失败: {str(e)}'})}\n\n"
                return
            
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
