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
import shutil
from datetime import datetime, timedelta

from automl_react.agents import (
    DataCleaningAgent,
    DataExplorationAgent,
    DataSplittingAgent,
    FeatureEngineeringAgent,
    ModelEvaluationAgent,
    ModelTrainingAgent,
    run_dataset_split,
)
from automl_react.agents.data_analysis_agent import DataAnalysisAgent
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


class PlanRevisionRequest(BaseModel):
    """方案修订请求"""
    session_id: str
    confirmation_id: str
    modifications: str


# 辅助函数
def _get_session_original_data_path(session_id: str) -> Path:
    """获取会话内统一的原始数据资产路径。"""
    asset_manager = get_asset_manager(session_id=session_id)
    return asset_manager.session_dir / "data" / "original_data.csv"


def _ensure_session_data_path(session_id: str, source_data_path: Optional[str]) -> Optional[str]:
    """确保源数据已落入会话资产目录，并返回统一后的资产路径。"""
    if not source_data_path:
        return None

    session_data_path = _get_session_original_data_path(session_id)
    session_data_path.parent.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_data_path)
    if source_path.exists():
        try:
            if source_path.resolve() != session_data_path.resolve():
                import shutil
                shutil.copy2(source_path, session_data_path)
                print(f"[API] 初始数据已复制到: {session_data_path}")
        except FileNotFoundError:
            import shutil
            shutil.copy2(source_path, session_data_path)
            print(f"[API] 初始数据已复制到: {session_data_path}")

    if session_data_path.exists():
        _save_data_onboarding_artifacts(session_id, session_data_path, source_data_path)
        return str(session_data_path)

    return source_data_path


def _save_data_onboarding_artifacts(
    session_id: str,
    session_data_path: Path,
    original_source_path: str,
) -> None:
    """生成并持久化 schema 快照和上传元数据。"""
    import hashlib
    import pandas as pd

    asset_manager = get_asset_manager(session_id=session_id)
    metadata_path = asset_manager.session_dir / "data" / "data_metadata.json"
    schema_path = asset_manager.session_dir / "data" / "schema_snapshot.json"

    # 如果两个产物都已存在就跳过，避免重复覆盖
    if metadata_path.exists() and schema_path.exists():
        return

    # ---- 上传元数据 ----
    try:
        file_size = session_data_path.stat().st_size
        md5 = hashlib.md5()
        with open(session_data_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                md5.update(chunk)
        checksum = md5.hexdigest()
    except Exception:
        file_size = -1
        checksum = "unknown"

    data_metadata = {
        "session_id": session_id,
        "upload_timestamp": datetime.now().isoformat(),
        "original_source_path": str(original_source_path),
        "asset_path": str(session_data_path),
        "data_version": "1.0_original",
        "file_size_bytes": file_size,
        "checksum_md5": checksum,
    }
    asset_manager.save_data(
        data=json.dumps(data_metadata, ensure_ascii=False, indent=2),
        filename="data_metadata.json",
        asset_type="data",
    )
    print(f"[API] 上传元数据已保存: data/data_metadata.json")

    # ---- Schema 快照 ----
    try:
        df = pd.read_csv(session_data_path, nrows=0)  # 只读列名和类型
        df_full = pd.read_csv(session_data_path)
        schema_snapshot = {
            "session_id": session_id,
            "snapshot_timestamp": datetime.now().isoformat(),
            "shape": [int(df_full.shape[0]), int(df_full.shape[1])],
            "columns": list(df_full.columns),
            "dtypes": {col: str(dtype) for col, dtype in df_full.dtypes.items()},
            "missing_counts": {col: int(v) for col, v in df_full.isnull().sum().items()},
            "missing_ratios": {col: round(float(v / len(df_full) * 100), 2) for col, v in df_full.isnull().sum().items()},
            "numeric_columns": list(df_full.select_dtypes(include=["int64", "float64"]).columns),
            "categorical_columns": list(df_full.select_dtypes(include=["object", "category", "bool"]).columns),
            "duplicate_rows": int(df_full.duplicated().sum()),
            "memory_bytes": int(df_full.memory_usage(deep=True).sum()),
        }
    except Exception as exc:
        schema_snapshot = {
            "session_id": session_id,
            "snapshot_timestamp": datetime.now().isoformat(),
            "error": str(exc),
        }

    asset_manager.save_data(
        data=json.dumps(schema_snapshot, ensure_ascii=False, indent=2),
        filename="schema_snapshot.json",
        asset_type="data",
    )
    print(f"[API] Schema 快照已保存: data/schema_snapshot.json")


def _normalize_workflow_data_path(session_id: str, workflow_state: Optional[WorkflowState]) -> None:
    """将工作流中的源数据路径统一纠偏到会话资产目录。"""
    if not workflow_state:
        return

    current_data_path = workflow_state.get_context("data_path")
    normalized_data_path = _ensure_session_data_path(session_id, current_data_path)

    if normalized_data_path and normalized_data_path != current_data_path:
        workflow_state.set_context("data_path", normalized_data_path)
        workflow_state.save()
        print(f"[API] 已统一 data_path 到资产路径: {normalized_data_path}")


def _resolve_stage_alias(stage: str) -> str:
    """兼容旧阶段命名。"""
    if stage == "data_analysis":
        return "problem_definition"
    return stage


def _get_problem_definition_payload(workflow_state: Optional[WorkflowState]) -> Dict[str, Any]:
    """获取已确认的问题定义结构化结果。"""
    if not workflow_state:
        return {}
    payload = workflow_state.get_context("problem_definition", {})
    return payload if isinstance(payload, dict) else {}


def _get_effective_target_column(workflow_state: Optional[WorkflowState]) -> Optional[str]:
    """优先使用问题定义阶段确认后的目标列。"""
    payload = _get_problem_definition_payload(workflow_state)
    resolved_target = payload.get("target_column") or payload.get("prediction_target")
    if isinstance(resolved_target, str) and resolved_target.strip():
        return resolved_target.strip()
    if workflow_state:
        return workflow_state.get_context("target_column")
    return None


def _get_effective_task_type(workflow_state: Optional[WorkflowState]) -> Optional[str]:
    """优先使用问题定义阶段确认后的任务类型。"""
    payload = _get_problem_definition_payload(workflow_state)
    resolved_task_type = payload.get("task_type")
    if resolved_task_type in {"classification", "regression"}:
        return resolved_task_type
    if workflow_state:
        return workflow_state.get_context("task_type")
    return None


def _get_split_payload(workflow_state: Optional[WorkflowState]) -> Dict[str, Any]:
    """获取已确认的数据切分结果。"""
    if not workflow_state:
        return {}
    payload = workflow_state.get_context("data_split", {})
    return payload if isinstance(payload, dict) else {}


def _get_train_raw_data_path(workflow_state: Optional[WorkflowState]) -> Optional[str]:
    split_payload = _get_split_payload(workflow_state)
    train_path = split_payload.get("train_raw_path") or split_payload.get("split_paths", {}).get("train_raw_path")
    if train_path:
        return train_path
    if workflow_state:
        return workflow_state.get_context("data_path")
    return None


def _get_valid_raw_data_path(workflow_state: Optional[WorkflowState]) -> Optional[str]:
    split_payload = _get_split_payload(workflow_state)
    return split_payload.get("valid_raw_path") or split_payload.get("split_paths", {}).get("valid_raw_path")


def _get_test_raw_data_path(workflow_state: Optional[WorkflowState]) -> Optional[str]:
    split_payload = _get_split_payload(workflow_state)
    return split_payload.get("test_raw_path") or split_payload.get("split_paths", {}).get("test_raw_path")


def _normalize_list(value: Any) -> List[str]:
    """将问题定义中的条目统一转为字符串列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _compose_stage_task_description(workflow_state: Optional[WorkflowState]) -> str:
    """将原始任务描述和已确认问题定义合并为后续阶段上下文。"""
    if not workflow_state:
        return ""

    raw_description = workflow_state.get_context("task_description", "")
    payload = _get_problem_definition_payload(workflow_state)
    if not payload:
        return raw_description

    secondary_metrics = ", ".join(_normalize_list(payload.get("secondary_metrics"))) or "无"
    business_constraints = "\n".join(f"- {item}" for item in _normalize_list(payload.get("business_constraints"))) or "- 无"
    success_criteria = "\n".join(f"- {item}" for item in _normalize_list(payload.get("success_criteria"))) or "- 无"
    assumptions = "\n".join(f"- {item}" for item in _normalize_list(payload.get("assumptions"))) or "- 无"
    open_questions = "\n".join(f"- {item}" for item in _normalize_list(payload.get("open_questions"))) or "- 无"

    sections = [
        "## 已确认的问题定义",
        "",
        f"- 任务类型: {payload.get('task_type', '未确认')}",
        f"- 目标列: {payload.get('target_column', workflow_state.get_context('target_column', '未确认'))}",
        f"- 预测目标: {payload.get('prediction_target', '未确认')}",
        f"- 预测时点: {payload.get('prediction_timing', '待确认')}",
        f"- 主评估指标: {payload.get('primary_metric', '待确认')}",
        f"- 辅助指标: {secondary_metrics}",
        "",
        "### 业务约束",
        business_constraints,
        "",
        "### 成功标准",
        success_criteria,
        "",
        "### 关键假设",
        assumptions,
        "",
        "### 待确认问题",
        open_questions,
    ]

    if raw_description:
        sections.extend([
            "",
            "## 原始任务描述",
            "",
            raw_description,
        ])

    split_payload = _get_split_payload(workflow_state)
    if split_payload:
        split_paths = split_payload.get("split_paths", {})
        sections.extend([
            "",
            "## 已确认的数据切分",
            "",
            f"- 切分策略: {split_payload.get('split_strategy', '未确认')}",
            f"- train_raw: {split_paths.get('train_raw_path', '未生成')}",
            f"- valid_raw: {split_paths.get('valid_raw_path', '未生成')}",
            f"- test_raw: {split_paths.get('test_raw_path', '未生成')}",
            "- 原则: valid 用于选方案，test 只用于最终评估，不参与任何拟合与调参。",
        ])

    return "\n".join(sections)


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
                _normalize_workflow_data_path(session_id, workflow_state)

                # 尝试恢复 ConfirmationManager
                cm_path = str(session_dir / "state" / "confirmation_state.json")
                confirmation_manager = ConfirmationManager.load_from_disk(cm_path)
                if confirmation_manager is None:
                    confirmation_manager = ConfirmationManager(save_path=cm_path)

                _sessions[session_id] = {
                    "session_id": session_id,
                    "created_at": workflow_state.history[0].get("timestamp", datetime.now().isoformat()) if workflow_state.history else datetime.now().isoformat(),
                    "workflow_state": workflow_state,
                    "confirmation_manager": confirmation_manager,
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


def _ensure_agent(session: dict, agent_key: str, agent_class, **init_kwargs):
    """获取或创建 Agent 实例，确保缓存到 session。"""
    agent = session["agents"].get(agent_key)
    if agent is None:
        agent = agent_class(**init_kwargs)
        session["agents"][agent_key] = agent
    return agent


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
    normalized_data_path = _ensure_session_data_path(request.session_id, request.data_path)
    
    # 创建工作流状态
    workflow_state = WorkflowState(
        session_id=request.session_id,
        initial_stage=WorkflowStage.DATA_UPLOAD
    )
    workflow_state.set_context("data_path", normalized_data_path)
    workflow_state.set_context("target_column", request.target_column)
    workflow_state.set_context("task_type", request.task_type)
    workflow_state.set_context("model", request.model)
    workflow_state.set_context("task_description", request.task_description)  # 保存用户的建模背景和要求

    # 把上传元数据和 schema 快照写入 workflow context（路径 + 完整对象）
    asset_manager = get_asset_manager(session_id=request.session_id)
    _meta_path = asset_manager.session_dir / "data" / "data_metadata.json"
    _schema_path = asset_manager.session_dir / "data" / "schema_snapshot.json"
    if _meta_path.exists():
        workflow_state.set_context("data_metadata_path", str(_meta_path))
        try:
            with open(_meta_path, "r", encoding="utf-8") as fh:
                workflow_state.set_context("data_metadata", json.load(fh))
        except Exception:
            pass
    if _schema_path.exists():
        workflow_state.set_context("schema_snapshot_path", str(_schema_path))
        try:
            with open(_schema_path, "r", encoding="utf-8") as fh:
                workflow_state.set_context("schema_snapshot", json.load(fh))
        except Exception:
            pass
    
    session["workflow_state"] = workflow_state
    cm_path = str(Path("assets") / request.session_id / "state" / "confirmation_state.json")
    session["confirmation_manager"] = ConfirmationManager(save_path=cm_path)
    
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
    session["agents"]["exploration"] = DataExplorationAgent(
        llm=llm,
        session_id=request.session_id
    )
    session["agents"]["analysis"] = DataAnalysisAgent(
        llm=llm,
        session_id=request.session_id
    )
    session["agents"]["cleaning"] = DataCleaningAgent(
        llm=llm,
        session_id=request.session_id
    )
    session["agents"]["splitting"] = DataSplittingAgent(
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
    session["agents"]["evaluation"] = ModelEvaluationAgent(
        llm=llm,
        session_id=request.session_id
    )
    
    # 保存状态
    workflow_state.save()
    
    return {
        "success": True,
        "session_id": request.session_id,
        "current_stage": workflow_state.current_stage.value,
        "data_path": normalized_data_path,
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
    
    requested_stage = stage
    stage = _resolve_stage_alias(stage)

    # 更新工作流状态（如果当前阶段不是目标阶段）
    target_stage = WorkflowStage(stage)
    if workflow_state.current_stage != target_stage:
        try:
            workflow_state.transition_to(target_stage)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的阶段转换: {e}")
    
    # 获取上下文
    data_path = workflow_state.get_context("data_path")
    target_column = _get_effective_target_column(workflow_state)
    task_type = _get_effective_task_type(workflow_state)
    
    # 获取或创建确认管理器
    confirmation_manager = session.get("confirmation_manager")
    if not confirmation_manager:
        # Session 恢复后需要重新创建确认管理器
        cm_path = str(Path("assets") / session_id / "state" / "confirmation_state.json")
        confirmation_manager = ConfirmationManager(save_path=cm_path)
        session["confirmation_manager"] = confirmation_manager
        print(f"[API] 已重新创建确认管理器")
    
    # 根据阶段执行相应操作
    if stage == "problem_definition":
        model = workflow_state.get_context("model", "kimi-k2.5")
        llm = create_llm_client(model)
        agent = _ensure_agent(session, "analysis", DataAnalysisAgent, llm=llm, session_id=session_id)

        try:
            task_description = workflow_state.get_context("task_description", "")
            print(f"[API] ========== 问题定义阶段开始 ==========")

            result = agent.generate_problem_definition(
                data_path=data_path,
                target_column=workflow_state.get_context("target_column"),
                task_type=workflow_state.get_context("task_type"),
                task_description=task_description,
            )
            definition_payload = agent.get_problem_definition_payload()

            confirmation_point = confirmation_manager.add_confirmation_point(
                stage="problem_definition",
                proposal_content=result,
                metadata={
                    "stage": "problem_definition",
                    "target_column": workflow_state.get_context("target_column"),
                    "task_type": workflow_state.get_context("task_type"),
                    "problem_definition": definition_payload,
                }
            )

            print(f"[API] ========== 问题定义阶段完成 ==========")
            response_payload = {
                "success": True,
                "stage": stage,
                "proposal": result,
                "problem_definition": definition_payload,
                "requires_confirmation": True,
                "confirmation_id": confirmation_point.id,
            }
            if requested_stage == "data_analysis":
                response_payload["analysis"] = result
            return response_payload
        except Exception as e:
            return {"success": False, "error": str(e)}

    elif stage == "data_contract_check":
        try:
            print(f"[API] ========== 数据契约检查阶段开始 ==========")
            from automl_react.agents.data_contract_agent import run_data_contract_checks

            problem_def = _get_problem_definition_payload(workflow_state)
            contract_result = run_data_contract_checks(
                data_path=data_path,
                target_column=target_column or workflow_state.get_context("target_column", ""),
                task_type=task_type or workflow_state.get_context("task_type", "regression"),
                problem_definition=problem_def or None,
            )

            asset_manager = get_asset_manager(session_id=session_id)
            asset_manager.save_data(
                data=contract_result["summary"],
                filename="data_contract_report.md",
                asset_type="analysis",
                metadata={"stage": "data_contract_check", "timestamp": datetime.now().isoformat()},
            )
            asset_manager.save_data(
                data=json.dumps(contract_result, ensure_ascii=False, indent=2, default=str),
                filename="data_contract_result.json",
                asset_type="analysis",
                metadata={"stage": "data_contract_check", "timestamp": datetime.now().isoformat()},
            )

            # 创建确认点，让用户审阅风险清单并决定是否继续
            confirmation_point = confirmation_manager.add_confirmation_point(
                stage="data_contract_check",
                proposal_content=contract_result["summary"],
                metadata={
                    "stage": "data_contract_check",
                    "modelable": contract_result["modelable"],
                    "stats": contract_result["stats"],
                },
            )

            print(f"[API] 契约检查结论: {'可建模' if contract_result['modelable'] else '不可建模'}")
            print(f"[API] ========== 数据契约检查阶段完成 ==========")
            return {
                "success": True,
                "stage": stage,
                "modelable": contract_result["modelable"],
                "proposal": contract_result["summary"],
                "risk_list": contract_result["risk_list"],
                "questions_for_business": contract_result["questions_for_business"],
                "stats": contract_result["stats"],
                "requires_confirmation": True,
                "confirmation_id": confirmation_point.id,
            }
        except Exception as e:
            import traceback
            return {"success": False, "error": f"{e}\n{traceback.format_exc()}"}

    elif stage == "data_splitting":
        try:
            print(f"[API] ========== 数据集切分阶段开始 ==========")
            model = workflow_state.get_context("model", "kimi-k2.5")
            llm = create_llm_client(model)
            agent = _ensure_agent(session, "splitting", DataSplittingAgent, llm=llm, session_id=session_id)
            problem_def = _get_problem_definition_payload(workflow_state)
            split_plan = agent.generate_split_plan(
                data_path=data_path,
                target_column=target_column or workflow_state.get_context("target_column", ""),
                task_type=task_type or workflow_state.get_context("task_type", "classification"),
                problem_definition=problem_def or None,
                task_description=workflow_state.get_context("task_description", ""),
            )

            split_config = agent.split_config or {}
            preview_result = run_dataset_split(
                data_path=data_path,
                target_column=target_column or workflow_state.get_context("target_column", ""),
                task_type=task_type or workflow_state.get_context("task_type", "classification"),
                problem_definition=problem_def or None,
                output_dir=None,
                config=split_config,
            )

            asset_manager = get_asset_manager(session_id=session_id)
            asset_manager.save_data(
                data=split_plan,
                filename="dataset_split_report.md",
                asset_type="analysis",
                metadata={"stage": "data_splitting", "timestamp": datetime.now().isoformat()},
            )
            asset_manager.save_data(
                data=json.dumps(preview_result, ensure_ascii=False, indent=2),
                filename="dataset_split_result.json",
                asset_type="analysis",
                metadata={"stage": "data_splitting", "timestamp": datetime.now().isoformat()},
            )

            confirmation_point = confirmation_manager.add_confirmation_point(
                stage="data_splitting",
                proposal_content=split_plan,
                metadata={
                    "stage": "data_splitting",
                    "split_strategy": preview_result["split_strategy"],
                    "has_validation_split": preview_result["has_validation_split"],
                    "counts": preview_result["counts"],
                    "split_config": split_config,
                },
            )
            confirmation_point.modifiable_aspects = agent.get_modifiable_aspects()

            print(f"[API] ========== 数据集切分阶段完成 ==========")
            return {
                "success": True,
                "stage": stage,
                "proposal": split_plan,
                "split_strategy": preview_result["split_strategy"],
                "counts": preview_result["counts"],
                "has_validation_split": preview_result["has_validation_split"],
                "questions_for_business": preview_result["questions_for_business"],
                "warnings": preview_result["warnings"],
                "split_config": split_config,
                "requires_confirmation": True,
                "confirmation_id": confirmation_point.id,
                "modifiable_aspects": confirmation_point.modifiable_aspects,
            }
        except Exception as e:
            import traceback
            return {"success": False, "error": f"{e}\n{traceback.format_exc()}"}

    elif stage == "data_exploration":
        model = workflow_state.get_context("model", "kimi-k2.5")
        llm = create_llm_client(model)
        agent = _ensure_agent(session, "exploration", DataExplorationAgent, llm=llm, session_id=session_id)
        if not agent:
            return {"success": False, "error": "DataExploration Agent 未初始化"}
        
        try:
            # 获取用户的建模背景和要求
            task_description = _compose_stage_task_description(workflow_state)
            
            print(f"[API] ========== 数据探索性分析阶段开始 ==========")
            
            # 读取清洗后的数据路径
            asset_manager = get_asset_manager(session_id=session_id)
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
                cleaned_data_path = _get_train_raw_data_path(workflow_state) or data_path
                print(f"[API] 使用训练集原始数据: {cleaned_data_path}")
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
        model = workflow_state.get_context("model", "kimi-k2.5")
        llm = create_llm_client(model)
        agent = _ensure_agent(session, "cleaning", DataCleaningAgent, llm=llm, session_id=session_id)
        if agent:
            try:
                print(f"[API] ========== 数据清洗阶段开始 ==========")
                cleaning_input_path = _get_train_raw_data_path(workflow_state) or data_path
                
                # 获取用户的建模背景和要求
                task_description = _compose_stage_task_description(workflow_state)
                if task_description:
                    print(f"[API] 用户建模背景: {task_description[:100]}...")
                
                # 数据清洗阶段自己进行数据质量分析
                # 生成清洗方案（包含数据质量分析）
                result = agent.generate_cleaning_plan(
                    cleaning_input_path,
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
                        "data_path": cleaning_input_path,
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
                confirmation_point.modifiable_aspects = agent.get_modifiable_aspects()

                return {
                    "success": True,
                    "stage": stage,
                    "proposal": result,
                    "requires_confirmation": True,
                    "confirmation_id": confirmation_point.id,
                    "modifiable_aspects": confirmation_point.modifiable_aspects,
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

            agent = _ensure_agent(session, "feature", FeatureEngineeringAgent, llm=llm, session_id=session_id)
            print(f"[API] 特征工程 Agent 已重新创建")
        
        if agent:
            try:
                print(f"[API] ========== 特征工程阶段开始 ==========")
                
                # 获取用户的建模背景和要求
                task_description = _compose_stage_task_description(workflow_state)
                if task_description:
                    print(f"[API] 用户建模背景: {task_description[:100]}...")
                
                # 读取前一阶段的结果
                asset_manager = get_asset_manager(session_id=session_id)
                # 读取探索性分析报告（新流程：exploration 目录）
                exploration_result = asset_manager.read_asset("exploration", "data_exploration_result.md")
                # 读取清洗后的数据路径
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
                    _get_train_raw_data_path(workflow_state) or data_path,
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
                        "data_path": cleaned_data_path or _get_train_raw_data_path(workflow_state) or data_path,
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
                confirmation_point.modifiable_aspects = agent.get_modifiable_aspects()

                return {
                    "success": True,
                    "stage": stage,
                    "proposal": result,
                    "requires_confirmation": True,
                    "confirmation_id": confirmation_point.id,
                    "modifiable_aspects": confirmation_point.modifiable_aspects,
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

    elif stage == "model_training":
        model = workflow_state.get_context("model", "kimi-k2.5")
        llm = create_llm_client(model)
        agent = _ensure_agent(session, "model", ModelTrainingAgent, llm=llm, session_id=session_id)
        if agent:
            try:
                print(f"[API] ========== 模型训练阶段开始 ==========")
                
                # 获取用户的建模背景和要求
                task_description = _compose_stage_task_description(workflow_state)
                if task_description:
                    print(f"[API] 用户建模背景: {task_description[:100]}...")
                
                # 读取所有前一阶段的结果
                asset_manager = get_asset_manager(session_id=session_id)
                exploration_result = asset_manager.read_asset("exploration", "data_exploration_result.md")
                feature_metrics_report = asset_manager.read_asset("features", "feature_metrics_report.md")
                
                # 读取特征工程后的数据路径
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
                    _get_train_raw_data_path(workflow_state) or data_path,
                    target_column, 
                    task_type,
                    exploration_report=exploration_result,
                    feature_metrics_report=feature_metrics_report,
                    features_data_path=features_data_path,
                    task_description=task_description,
                    train_split_path=_get_train_raw_data_path(workflow_state),
                    valid_split_path=_get_valid_raw_data_path(workflow_state),
                    test_split_path=_get_test_raw_data_path(workflow_state),
                )
                print(f"[API] 模型训练方案生成完成，长度: {len(result)} 字符")
                
                # 保存模型训练方案到资产
                asset_manager.save_data(
                    data=result,
                    filename="model_training_plan.md",
                    asset_type="models",
                    metadata={
                        "stage": "model_training",
                        "data_path": features_data_path or _get_train_raw_data_path(workflow_state) or data_path,
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
                confirmation_point.modifiable_aspects = agent.get_modifiable_aspects()

                return {
                    "success": True,
                    "stage": stage,
                    "proposal": result,
                    "requires_confirmation": True,
                    "confirmation_id": confirmation_point.id,
                    "modifiable_aspects": confirmation_point.modifiable_aspects,
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

    elif stage == "model_evaluation":
        model = workflow_state.get_context("model", "kimi-k2.5")
        llm = create_llm_client(model)
        agent = _ensure_agent(session, "evaluation", ModelEvaluationAgent, llm=llm, session_id=session_id)

        try:
            print(f"[API] ========== 模型评估阶段开始 ==========")

            asset_manager = get_asset_manager(session_id=session_id)
            model_result_json = asset_manager.read_asset("models", "model_training_result.json")
            model_result = json.loads(model_result_json) if model_result_json else None

            task_description = _compose_stage_task_description(workflow_state)
            result = agent.generate_evaluation_plan(
                target_column=target_column,
                task_type=task_type,
                model_result=model_result,
                task_description=task_description,
            )

            asset_manager.save_data(
                data=result,
                filename="evaluation_plan.md",
                asset_type="reports",
                metadata={
                    "stage": "model_evaluation",
                    "timestamp": datetime.now().isoformat()
                }
            )

            confirmation_point = confirmation_manager.add_confirmation_point(
                stage="model_evaluation",
                proposal_content=result
            )

            print(f"[API] ========== 模型评估阶段完成 ==========")
            return {
                "success": True,
                "stage": stage,
                "proposal": result,
                "requires_confirmation": True,
                "confirmation_id": confirmation_point.id,
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
    target_column = _get_effective_target_column(workflow_state)
    task_type = _get_effective_task_type(workflow_state)
    
    # 执行结果
    execution_result = None
    next_confirmation_point = None
    
    # 如果用户确认或修改，执行相应操作
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
                    session, _get_train_raw_data_path(workflow_state) or data_path, request.modifications
                )
            elif stage == "feature_engineering":
                execution_result = await execute_feature_engineering(
                    session, _get_train_raw_data_path(workflow_state) or data_path, target_column, task_type, request.modifications
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
                    session, _get_train_raw_data_path(workflow_state) or data_path, target_column, task_type, request.modifications
                )
            elif stage == "feature_evaluation":
                execution_result = await execute_feature_evaluation(
                    session, request.modifications
                )
            elif stage == "model_evaluation":
                execution_result = await execute_model_evaluation(
                    session, target_column, task_type
                )
                if execution_result.get("success"):
                    workflow_state.transition_to(
                        WorkflowStage.COMPLETED,
                        message="Model evaluation completed"
                    )
                    # 自动生成报告和 JSON 摘要
                    try:
                        report_gen = ReportGenerator(session_id=session_id)
                        data_path = workflow_state.get_context("data_path", "")
                        report = report_gen.generate_report(data_path, target_column, task_type)
                        report_gen.export_to_html(report)
                        report_gen.generate_summary_json(data_path, target_column, task_type)
                        print(f"[API] 自动生成报告完成")
                    except Exception as report_err:
                        print(f"[API] 自动报告生成失败（不影响主流程）: {report_err}")
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"执行失败: {str(e)}",
                    "stage": stage
                }
            )
    
    if stage == "model_evaluation" and status == ConfirmationStatus.SKIPPED:
        workflow_state.transition_to(
            WorkflowStage.COMPLETED,
            message="Model evaluation skipped"
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
        "model_evaluation": "evaluation",
    }
    agent_key = stage_agent_map.get(stage)
    if not agent_key:
        raise HTTPException(status_code=400, detail=f"不支持方案修订的阶段: {stage}")

    agents = session.get("agents", {})
    agent = agents.get(agent_key)
    if not agent:
        raise HTTPException(status_code=404, detail=f"阶段 '{stage}' 对应的 Agent 未初始化")
    return agent


@app.post("/confirmation/revise")
async def revise_plan(request: PlanRevisionRequest):
    """
    修订当前方案。

    用户审阅方案后提供修改意见，系统基于反馈生成修订后方案，
    再返回新的 confirmation_id 供用户确认或继续修改。
    """
    session = get_session(request.session_id)
    confirmation_manager = session.get("confirmation_manager")

    if not confirmation_manager:
        raise HTTPException(status_code=404, detail="确认管理器不存在")

    # 查找当前确认点
    current = confirmation_manager._find_point_by_id(request.confirmation_id)
    if not current:
        # 也尝试当前活跃的确认点
        current = confirmation_manager.current
        if not current:
            raise HTTPException(status_code=404, detail="未找到对应的确认点")

    stage = current.stage

    # 获取对应 Agent
    agent = _get_agent_for_stage(session, stage)

    # 记录本轮修订
    current.revision_history.append({
        "round": len(current.revision_history) + 1,
        "user_feedback": request.modifications,
        "previous_proposal": current.proposal_content,
        "timestamp": datetime.now().isoformat(),
    })

    # 调用 Agent 修订方案
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

    # 标记旧确认点状态
    current.set_user_response(
        status=ConfirmationStatus.REVISION_REQUESTED,
        modifications=request.modifications,
    )

    # 创建新确认点
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
        "feature_engineering": ("features", "feature_plan.md"),
        "model_training": ("models", "model_plan.md"),
        "data_splitting": ("data", "split_plan.md"),
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


async def execute_data_splitting(
    session: Dict,
    data_path: str,
    modifications: Optional[str] = None,
) -> Dict:
    """根据用户确认后的方案生成切分代码并执行。"""
    session_id = session.get("session_id", "default")
    workflow_state = session.get("workflow_state")
    asset_manager = get_asset_manager(session_id=session_id)
    agent = session["agents"].get("splitting")

    if not agent:
        model = workflow_state.get_context("model", "kimi-k2.5") if workflow_state else "kimi-k2.5"
        llm = create_llm_client(model)
        agent = DataSplittingAgent(llm=llm, session_id=session_id)
        session["agents"]["splitting"] = agent

    target_column = _get_effective_target_column(workflow_state) or workflow_state.get_context("target_column")
    task_type = _get_effective_task_type(workflow_state) or workflow_state.get_context("task_type", "classification")
    problem_def = _get_problem_definition_payload(workflow_state)

    if not agent.split_plan:
        agent.generate_split_plan(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,
            problem_definition=problem_def or None,
            task_description=workflow_state.get_context("task_description", "") if workflow_state else "",
        )

    code = agent.generate_split_code(modifications)
    split_result = agent.execute_split(code)

    split_paths = split_result.get("split_paths", {})
    asset_manager.save_data(
        data=split_result.get("summary", ""),
        filename="dataset_split_report.md",
        asset_type="analysis",
        metadata={"stage": "data_splitting", "timestamp": datetime.now().isoformat()},
    )
    asset_manager.save_data(
        data=json.dumps(split_result, ensure_ascii=False, indent=2),
        filename="dataset_split_result.json",
        asset_type="analysis",
        metadata={"stage": "data_splitting", "timestamp": datetime.now().isoformat()},
    )

    if workflow_state:
        workflow_state.set_context("data_split", split_result)
        workflow_state.set_context("train_raw_path", split_paths.get("train_raw_path"))
        workflow_state.set_context("valid_raw_path", split_paths.get("valid_raw_path"))
        workflow_state.set_context("test_raw_path", split_paths.get("test_raw_path"))
        workflow_state.set_context("split_strategy", split_result.get("split_strategy"))
        workflow_state.set_context("split_config", split_result.get("config", agent.split_config or {}))
        workflow_state.save()

    return {
        "success": True,
        "stage": "data_splitting",
        "split_strategy": split_result.get("split_strategy"),
        "has_validation_split": split_result.get("has_validation_split"),
        "counts": split_result.get("counts", {}),
        "execution_mode": split_result.get("execution_mode"),
        "train_raw_path": split_paths.get("train_raw_path"),
        "valid_raw_path": split_paths.get("valid_raw_path"),
        "test_raw_path": split_paths.get("test_raw_path"),
    }


async def execute_problem_definition(
    session: Dict,
    data_path: str,
    modifications: Optional[str] = None,
) -> Dict:
    """固化用户确认后的问题定义结果。"""
    session_id = session.get("session_id", "default")
    workflow_state = session.get("workflow_state")
    asset_manager = get_asset_manager(session_id=session_id)

    agent = session["agents"].get("analysis")
    if not agent:
        raise ValueError("问题定义 Agent 不存在")

    original_task_description = workflow_state.get_context("task_description", "") if workflow_state else ""
    generation_task_description = original_task_description
    if modifications:
        generation_task_description = (
            f"{original_task_description}\n\n## 用户确认补充\n\n{modifications}"
            if original_task_description else modifications
        )

    if modifications or not getattr(agent, "problem_definition_plan", None):
        agent.generate_problem_definition(
            data_path=data_path,
            target_column=workflow_state.get_context("target_column") if workflow_state else None,
            task_type=workflow_state.get_context("task_type") if workflow_state else None,
            task_description=generation_task_description,
        )

    payload = dict(agent.get_problem_definition_payload())
    if modifications:
        payload["user_confirmed_modifications"] = modifications

    report_text = agent.problem_definition_plan or ""
    report_asset = asset_manager.save_data(
        data=report_text,
        filename="problem_definition.md",
        asset_type="analysis",
        metadata={
            "stage": "problem_definition",
            "timestamp": datetime.now().isoformat(),
        }
    )
    json_asset = asset_manager.save_data(
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        filename="problem_definition.json",
        asset_type="analysis",
        metadata={
            "stage": "problem_definition",
            "timestamp": datetime.now().isoformat(),
        }
    )

    if workflow_state:
        workflow_state.update_context({
            "problem_definition": payload,
            "problem_definition_report": report_text,
            "problem_definition_path": report_asset.path,
            "problem_definition_json_path": json_asset.path,
            "primary_metric": payload.get("primary_metric"),
            "prediction_timing": payload.get("prediction_timing"),
            "business_constraints": _normalize_list(payload.get("business_constraints")),
            "success_criteria": _normalize_list(payload.get("success_criteria")),
        })
        workflow_state.save()

    return {
        "success": True,
        "stage": "problem_definition",
        "problem_definition_path": report_asset.path,
        "problem_definition_json_path": json_asset.path,
        "problem_definition": payload,
    }


async def execute_data_contract_check(
    session: Dict,
    data_path: str,
    modifications: Optional[str] = None,
) -> Dict:
    """固化用户确认后的数据契约检查结果到 workflow context。"""
    session_id = session.get("session_id", "default")
    workflow_state = session.get("workflow_state")
    asset_manager = get_asset_manager(session_id=session_id)

    # 读取已经生成的契约检查结果
    contract_json = asset_manager.read_asset("analysis", "data_contract_result.json")
    if contract_json:
        contract_result = json.loads(contract_json)
    else:
        # 如果用户修改后重新检查
        from automl_react.agents.data_contract_agent import run_data_contract_checks
        target_column = _get_effective_target_column(workflow_state)
        task_type = _get_effective_task_type(workflow_state)
        problem_def = _get_problem_definition_payload(workflow_state)
        contract_result = run_data_contract_checks(
            data_path=data_path,
            target_column=target_column or "",
            task_type=task_type or "regression",
            problem_definition=problem_def or None,
        )
        asset_manager.save_data(
            data=contract_result["summary"],
            filename="data_contract_report.md",
            asset_type="analysis",
            metadata={"stage": "data_contract_check", "timestamp": datetime.now().isoformat()},
        )
        asset_manager.save_data(
            data=json.dumps(contract_result, ensure_ascii=False, indent=2, default=str),
            filename="data_contract_result.json",
            asset_type="analysis",
            metadata={"stage": "data_contract_check", "timestamp": datetime.now().isoformat()},
        )

    if workflow_state:
        workflow_state.update_context({
            "data_contract_modelable": contract_result.get("modelable", True),
            "data_contract_risk_list": contract_result.get("risk_list", []),
            "data_contract_questions": contract_result.get("questions_for_business", []),
        })
        if modifications:
            workflow_state.set_context("data_contract_user_notes", modifications)
        workflow_state.save()

    return {
        "success": True,
        "stage": "data_contract_check",
        "modelable": contract_result.get("modelable", True),
        "risk_count": len(contract_result.get("risk_list", [])),
    }


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

    workflow_state = session.get("workflow_state")

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
            features_data_path=features_data_path,
            train_split_path=workflow_state.get_context("train_raw_path") if workflow_state else None,
            valid_split_path=workflow_state.get_context("valid_raw_path") if workflow_state else None,
            test_split_path=workflow_state.get_context("test_raw_path") if workflow_state else None,
        )
    elif workflow_state:
        agent.train_split_path = workflow_state.get_context("train_raw_path")
        agent.valid_split_path = workflow_state.get_context("valid_raw_path")
        agent.test_split_path = workflow_state.get_context("test_raw_path")

    # 生成训练代码并执行
    code = agent.generate_model_code(modifications)

    # 执行代码
    result = agent.execute_model_training(code)

    # 构建执行结果
    execution_result = {
        "success": result.get("success", False),
        "model_path": result.get("model_path"),
        "train_split_path": result.get("train_split_path"),
        "valid_split_path": result.get("valid_split_path"),
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


async def execute_model_evaluation(
    session: Dict,
    target_column: str,
    task_type: str,
) -> Dict:
    """执行模型评估。"""
    agent = session["agents"].get("evaluation")
    if not agent:
        agent = ModelEvaluationAgent(session_id=session.get("session_id", "default"))
        session["agents"]["evaluation"] = agent

    return agent.evaluate_from_training_result(
        target_column=target_column,
        task_type=task_type,
    )


@app.get("/confirmation/{session_id}/pending")
async def get_pending_confirmation(session_id: str):
    """获取待处理的确认点"""
    session = get_session(session_id)
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
    target_column = _get_effective_target_column(workflow_state)
    task_type = _get_effective_task_type(workflow_state)
    
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

        # 生成 JSON 摘要
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


@app.post("/pipeline/generate")
async def generate_pipeline(session_id: str):
    """生成全流程脚本"""
    session = get_session(session_id)
    workflow_state = session.get("workflow_state")
    
    if not workflow_state:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 获取上下文
    data_path = workflow_state.get_context("data_path")
    target_column = _get_effective_target_column(workflow_state)
    task_type = _get_effective_task_type(workflow_state)
    
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


# ==================== Session CRUD ====================

SESSION_TTL_HOURS = int(os.environ.get("AUTOML_SESSION_TTL_HOURS", "72"))


@app.get("/sessions")
async def list_sessions():
    """列出所有会话（合并内存态和磁盘态）"""
    sessions_info = {}

    # 从磁盘扫描
    assets_dir = Path("assets")
    if assets_dir.exists():
        for d in assets_dir.iterdir():
            if d.is_dir() and (d / "state" / "workflow_state.json").exists():
                sid = d.name
                try:
                    with open(d / "state" / "workflow_state.json", "r") as f:
                        state_data = json.load(f)
                    sessions_info[sid] = {
                        "session_id": sid,
                        "current_stage": state_data.get("current_stage", "unknown"),
                        "last_updated": state_data.get("last_updated", ""),
                        "created_at": state_data.get("history", [{}])[0].get("timestamp", "") if state_data.get("history") else "",
                    }
                except Exception:
                    sessions_info[sid] = {
                        "session_id": sid,
                        "current_stage": "unknown",
                        "last_updated": "",
                        "created_at": "",
                    }

    # 合并内存态
    for sid, sess in _sessions.items():
        ws = sess.get("workflow_state")
        if ws:
            sessions_info[sid] = {
                "session_id": sid,
                "current_stage": ws.current_stage.value if ws.current_stage else "unknown",
                "last_updated": ws.last_updated if hasattr(ws, "last_updated") else "",
                "created_at": sess.get("created_at", ""),
            }
        elif sid not in sessions_info:
            sessions_info[sid] = {
                "session_id": sid,
                "current_stage": "unknown",
                "last_updated": "",
                "created_at": sess.get("created_at", ""),
            }

    return {
        "success": True,
        "sessions": list(sessions_info.values()),
        "count": len(sessions_info),
    }


@app.get("/sessions/{session_id}/status")
async def get_session_detail(session_id: str):
    """获取完整会话状态详情"""
    session = get_session(session_id)
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


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, confirm: bool = False):
    """删除会话及所有关联资产"""
    if not confirm:
        return {
            "success": False,
            "message": "请添加 ?confirm=true 确认删除",
            "session_id": session_id,
        }

    # 清除内存态
    if session_id in _sessions:
        del _sessions[session_id]

    # 清除磁盘文件
    session_dir = Path("assets") / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)

    return {
        "success": True,
        "message": f"会话 {session_id} 已删除",
        "session_id": session_id,
    }


# ==================== Report Summary ====================

@app.get("/report/{session_id}/summary")
async def get_report_summary(session_id: str):
    """获取结构化 JSON 摘要"""
    asset_manager = get_asset_manager(session_id=session_id)
    summary_json = asset_manager.read_asset("reports", "summary.json")
    if not summary_json:
        raise HTTPException(status_code=404, detail="摘要报告尚未生成")
    try:
        return json.loads(summary_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="摘要报告格式错误")


# ==================== TTL Cleanup ====================

def _cleanup_expired_sessions():
    """清理过期的会话目录"""
    assets_dir = Path("assets")
    if not assets_dir.exists():
        return

    cutoff = datetime.now() - timedelta(hours=SESSION_TTL_HOURS)
    cleaned = 0

    for d in assets_dir.iterdir():
        if not d.is_dir():
            continue
        state_file = d / "state" / "workflow_state.json"
        if not state_file.exists():
            continue
        try:
            with open(state_file, "r") as f:
                state_data = json.load(f)
            last_updated = state_data.get("last_updated", "")
            if last_updated and datetime.fromisoformat(last_updated) < cutoff:
                sid = d.name
                if sid in _sessions:
                    del _sessions[sid]
                shutil.rmtree(d)
                cleaned += 1
                print(f"[API] 清理过期会话: {sid}")
        except Exception:
            continue

    if cleaned:
        print(f"[API] TTL 清理完成，共清理 {cleaned} 个过期会话")


@app.on_event("startup")
async def on_startup():
    """应用启动时清理过期会话"""
    _cleanup_expired_sessions()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
