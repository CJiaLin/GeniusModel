"""
API 辅助函数模块

纯函数：路径获取、数据规范化、stage alias 等
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from automl_react.assets import get_asset_manager
from automl_react.workflow import WorkflowState


def get_session_original_data_path(session_id: str, ext: str = ".csv") -> Path:
    """获取会话内统一的原始数据资产路径。"""
    asset_manager = get_asset_manager(session_id=session_id)
    return asset_manager.session_dir / "data" / f"original_data{ext}"


def _detect_file_ext(file_path: Path) -> str:
    """检测文件真实类型（基于魔数和扩展名）。"""
    ext = file_path.suffix.lower()
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)
        if magic == b'PK\x03\x04':
            return ".xlsx"
    except Exception:
        pass
    return ext if ext else ".csv"


def _read_data_file(file_path: Path) -> "pd.DataFrame":
    """智能读取数据文件（支持 csv/xlsx/xls，自动检测编码）。"""
    import pandas as pd

    ext = _detect_file_ext(file_path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    # CSV: 尝试多种编码
    for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            return pd.read_csv(file_path, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return pd.read_csv(file_path, encoding="utf-8", errors="replace")


def ensure_session_data_path(session_id: str, source_data_path: Optional[str]) -> Optional[str]:
    """确保源数据已落入会话资产目录（统一转为 CSV），并返回统一后的资产路径。"""
    import pandas as pd

    asset_manager = get_asset_manager(session_id=session_id)
    data_dir = asset_manager.session_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "original_data.csv"

    # 如果已经存在转换好的 CSV，直接返回
    if not source_data_path:
        if csv_path.exists():
            return str(csv_path)
        # 兼容：查找任意 original_data.* 文件
        for f in data_dir.glob("original_data.*"):
            return str(f)
        return None

    source_path = Path(source_data_path)
    if not source_path.exists():
        if csv_path.exists():
            return str(csv_path)
        return source_data_path

    # 如果已转换过，直接返回
    if csv_path.exists() and source_path.resolve() != csv_path.resolve():
        return str(csv_path)

    # 检测源文件格式
    source_ext = _detect_file_ext(source_path)

    # 保存原始文件（保留原始格式）
    original_path = data_dir / f"original_data{source_ext}"
    if source_path.resolve() != original_path.resolve():
        import shutil
        shutil.copy2(source_path, original_path)
        print(f"[API] 原始文件已复制到: {original_path}")

    # 读取数据并统一保存为 CSV（供后续流程使用）
    try:
        df = _read_data_file(original_path)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[API] 数据已转换并保存为 CSV: {csv_path} ({len(df)} 行 × {len(df.columns)} 列)")
    except Exception as e:
        print(f"[API] 数据转换为 CSV 失败: {e}")
        # 回退：如果无法转换，直接使用原始文件
        if original_path.exists():
            save_data_onboarding_artifacts(session_id, original_path, source_data_path)
            return str(original_path)
        return source_data_path

    save_data_onboarding_artifacts(session_id, csv_path, source_data_path)
    return str(csv_path)


def save_data_onboarding_artifacts(
    session_id: str,
    session_data_path: Path,
    original_source_path: str,
) -> None:
    """生成并持久化 schema 快照和上传元数据。"""
    import pandas as pd

    asset_manager = get_asset_manager(session_id=session_id)
    metadata_path = asset_manager.session_dir / "data" / "data_metadata.json"
    schema_path = asset_manager.session_dir / "data" / "schema_snapshot.json"

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
        df_full = _read_data_file(session_data_path)
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


def normalize_workflow_data_path(session_id: str, workflow_state: Optional[WorkflowState]) -> None:
    """将工作流中的源数据路径统一纠偏到会话资产目录。"""
    if not workflow_state:
        return

    current_data_path = workflow_state.get_context("data_path")
    normalized_data_path = ensure_session_data_path(session_id, current_data_path)

    if normalized_data_path and normalized_data_path != current_data_path:
        workflow_state.set_context("data_path", normalized_data_path)
        workflow_state.save()
        print(f"[API] 已统一 data_path 到资产路径: {normalized_data_path}")


def resolve_stage_alias(stage: str) -> str:
    """兼容旧阶段命名。"""
    if stage == "data_analysis":
        return "problem_definition"
    return stage


def get_problem_definition_payload(workflow_state: Optional[WorkflowState]) -> Dict[str, Any]:
    """获取已确认的问题定义结构化结果。"""
    if not workflow_state:
        return {}
    payload = workflow_state.get_context("problem_definition", {})
    return payload if isinstance(payload, dict) else {}


def get_effective_target_column(workflow_state: Optional[WorkflowState]) -> Optional[str]:
    """优先使用问题定义阶段确认后的目标列。"""
    payload = get_problem_definition_payload(workflow_state)
    resolved_target = payload.get("target_column") or payload.get("prediction_target")
    if isinstance(resolved_target, str) and resolved_target.strip():
        return resolved_target.strip()
    if workflow_state:
        return workflow_state.get_context("target_column")
    return None


def get_effective_task_type(workflow_state: Optional[WorkflowState]) -> Optional[str]:
    """优先使用问题定义阶段确认后的任务类型。"""
    payload = get_problem_definition_payload(workflow_state)
    resolved_task_type = payload.get("task_type")
    if resolved_task_type in {"classification", "regression"}:
        return resolved_task_type
    if workflow_state:
        return workflow_state.get_context("task_type")
    return None


def get_split_payload(workflow_state: Optional[WorkflowState]) -> Dict[str, Any]:
    """获取已确认的数据切分结果。"""
    if not workflow_state:
        return {}
    payload = workflow_state.get_context("data_split", {})
    return payload if isinstance(payload, dict) else {}


def get_train_raw_data_path(workflow_state: Optional[WorkflowState]) -> Optional[str]:
    split_payload = get_split_payload(workflow_state)
    train_path = split_payload.get("train_raw_path") or split_payload.get("split_paths", {}).get("train_raw_path")
    if train_path:
        return train_path
    if workflow_state:
        return workflow_state.get_context("data_path")
    return None


def get_valid_raw_data_path(workflow_state: Optional[WorkflowState]) -> Optional[str]:
    split_payload = get_split_payload(workflow_state)
    return split_payload.get("valid_raw_path") or split_payload.get("split_paths", {}).get("valid_raw_path")


def get_test_raw_data_path(workflow_state: Optional[WorkflowState]) -> Optional[str]:
    split_payload = get_split_payload(workflow_state)
    return split_payload.get("test_raw_path") or split_payload.get("split_paths", {}).get("test_raw_path")


def normalize_list(value: Any) -> List[str]:
    """将问题定义中的条目统一转为字符串列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def compose_stage_task_description(workflow_state: Optional[WorkflowState]) -> str:
    """将原始任务描述和已确认问题定义合并为后续阶段上下文。"""
    if not workflow_state:
        return ""

    raw_description = workflow_state.get_context("task_description", "")
    payload = get_problem_definition_payload(workflow_state)
    if not payload:
        return raw_description

    secondary_metrics = ", ".join(normalize_list(payload.get("secondary_metrics"))) or "无"
    business_constraints = "\n".join(f"- {item}" for item in normalize_list(payload.get("business_constraints"))) or "- 无"
    success_criteria = "\n".join(f"- {item}" for item in normalize_list(payload.get("success_criteria"))) or "- 无"
    assumptions = "\n".join(f"- {item}" for item in normalize_list(payload.get("assumptions"))) or "- 无"
    open_questions = "\n".join(f"- {item}" for item in normalize_list(payload.get("open_questions"))) or "- 无"

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

    split_payload = get_split_payload(workflow_state)
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


def ensure_agent(session: dict, agent_key: str, agent_class, **init_kwargs):
    """获取或创建 Agent 实例，确保缓存到 session。"""
    agent = session["agents"].get(agent_key)
    if agent is None:
        agent = agent_class(**init_kwargs)
        session["agents"][agent_key] = agent
    return agent
