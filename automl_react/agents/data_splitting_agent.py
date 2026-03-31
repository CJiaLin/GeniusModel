"""数据集切分 Agent。

在数据清洗前输出可审阅的切分方案，允许用户修改后确认，
再基于确认后的方案生成切分代码并执行。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ..core.react_agent import ConfirmationRequired, ReActAgent
from ..config import get_config_loader
from ..skills_loader import get_skill_loader
from ..tools.data_tools import DataAnalyzerTool, DataLoaderTool
from ..utils.code_generator import CodeGenerator


RANDOM_STATE = 42


@dataclass
class SplitConfig:
    task_type: str
    split_method: str
    train_ratio: float
    valid_ratio: float
    test_ratio: float
    split_column: Optional[str] = None
    has_validation_split: bool = True
    strategy_detail: str = ""
    warnings: Optional[List[str]] = None
    questions_for_business: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "split_method": self.split_method,
            "train_ratio": self.train_ratio,
            "valid_ratio": self.valid_ratio,
            "test_ratio": self.test_ratio,
            "split_column": self.split_column,
            "has_validation_split": self.has_validation_split,
            "strategy_detail": self.strategy_detail,
            "warnings": self.warnings or [],
            "questions_for_business": self.questions_for_business or [],
        }


def _normalize_task_type(task_type: Optional[str]) -> str:
    value = (task_type or "classification").strip().lower()
    alias_map = {
        "binary": "classification",
        "multiclass": "classification",
        "time_series": "time_series",
        "timeseries": "time_series",
        "forecasting": "time_series",
    }
    return alias_map.get(value, value)


def _should_create_validation_split(n_rows: int) -> bool:
    return n_rows >= 100


def _infer_datetime_column(df: pd.DataFrame) -> Optional[str]:
    datetime_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetime64", "datetimetz"]).columns)
    if datetime_cols:
        return datetime_cols[0]

    for col in df.columns:
        lowered = col.lower()
        if not any(token in lowered for token in ("date", "time", "timestamp", "dt")):
            continue
        sample = df[col].dropna().head(50)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return col
    return None


def _infer_group_column(df: pd.DataFrame) -> Optional[str]:
    preferred_tokens = ("user", "device", "patient", "member", "customer", "account", "group", "entity", "case")
    candidates: List[str] = []
    for col in df.columns:
        lowered = col.lower()
        if not any(token in lowered for token in preferred_tokens):
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            continue
        unique_ratio = nunique / max(len(df), 1)
        if unique_ratio >= 0.95:
            continue
        candidates.append(col)
    return candidates[0] if candidates else None


def _build_regression_bins(target: pd.Series) -> Optional[pd.Series]:
    clean_target = target.dropna()
    if clean_target.nunique() < 8 or len(clean_target) < 80:
        return None

    bins = min(10, max(4, len(clean_target) // 80))
    try:
        bucketed = pd.qcut(clean_target, q=bins, duplicates="drop")
    except Exception:
        return None

    if getattr(bucketed, "nunique", lambda: 0)() < 2:
        return None

    bucketed = bucketed.astype(str)
    if bucketed.value_counts().min() < 2:
        return None

    aligned = pd.Series(index=target.index, dtype="object")
    aligned.loc[clean_target.index] = bucketed
    return aligned


def _can_stratify(labels: Optional[pd.Series]) -> bool:
    if labels is None:
        return False
    non_null = labels.dropna()
    if non_null.nunique() < 2:
        return False
    return bool((non_null.value_counts() >= 2).all())


def _validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    total = round(train_ratio + valid_ratio + test_ratio, 6)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"切分比例之和必须为 1.0，当前为 {total}")
    if train_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train/test 比例必须大于 0")
    if valid_ratio < 0:
        raise ValueError("valid 比例不能小于 0")


def _three_way_random_split(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    stratify_labels: Optional[pd.Series] = None,
) -> Dict[str, pd.DataFrame]:
    _validate_ratios(train_ratio, valid_ratio, test_ratio)
    stratify_full = stratify_labels if _can_stratify(stratify_labels) else None

    train_valid_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_full,
    )

    if valid_ratio <= 0:
        return {"train": train_valid_df.copy(), "valid": None, "test": test_df.copy()}

    relative_valid_ratio = valid_ratio / (train_ratio + valid_ratio)
    stratify_train_valid = None
    if stratify_full is not None:
        stratify_train_valid = stratify_labels.loc[train_valid_df.index]
        if not _can_stratify(stratify_train_valid):
            stratify_train_valid = None

    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=relative_valid_ratio,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_train_valid,
    )
    return {"train": train_df.copy(), "valid": valid_df.copy(), "test": test_df.copy()}


def _three_way_time_split(
    df: pd.DataFrame,
    datetime_column: str,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Dict[str, pd.DataFrame]:
    _validate_ratios(train_ratio, valid_ratio, test_ratio)
    ordered = df.copy()
    ordered[datetime_column] = pd.to_datetime(ordered[datetime_column], errors="coerce")
    ordered = ordered.sort_values(datetime_column, kind="stable").reset_index(drop=True)

    n_rows = len(ordered)
    train_end = max(int(round(n_rows * train_ratio)), 1)
    valid_end = max(int(round(n_rows * (train_ratio + valid_ratio))), train_end)

    train_df = ordered.iloc[:train_end].copy()
    valid_df = ordered.iloc[train_end:valid_end].copy() if valid_ratio > 0 else None
    test_df = ordered.iloc[valid_end:].copy()
    return {"train": train_df, "valid": valid_df, "test": test_df}


def _three_way_group_split(
    df: pd.DataFrame,
    group_column: str,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Dict[str, pd.DataFrame]:
    _validate_ratios(train_ratio, valid_ratio, test_ratio)
    groups = df[group_column]

    outer_splitter = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=RANDOM_STATE)
    train_valid_idx, test_idx = next(outer_splitter.split(df, groups=groups))
    train_valid_df = df.iloc[train_valid_idx].copy()
    test_df = df.iloc[test_idx].copy()

    if valid_ratio <= 0:
        return {"train": train_valid_df, "valid": None, "test": test_df}

    relative_valid_ratio = valid_ratio / (train_ratio + valid_ratio)
    inner_groups = train_valid_df[group_column]
    inner_splitter = GroupShuffleSplit(n_splits=1, test_size=relative_valid_ratio, random_state=RANDOM_STATE)
    train_idx, valid_idx = next(inner_splitter.split(train_valid_df, groups=inner_groups))
    train_df = train_valid_df.iloc[train_idx].copy()
    valid_df = train_valid_df.iloc[valid_idx].copy()
    return {"train": train_df, "valid": valid_df, "test": test_df}


def build_default_split_config(
    data_path: str,
    target_column: str,
    task_type: str = "classification",
    problem_definition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    df = pd.read_csv(data_path)
    normalized_task_type = _normalize_task_type(task_type)

    if target_column not in df.columns:
        raise ValueError(f"目标列不存在: {target_column}")

    has_validation_split = _should_create_validation_split(len(df))
    train_ratio = 0.6 if has_validation_split else 0.8
    valid_ratio = 0.2 if has_validation_split else 0.0
    test_ratio = 0.2

    split_method = "random"
    strategy_detail = "普通随机切分"
    split_column = None
    warnings: List[str] = []
    questions_for_business: List[str] = []

    datetime_column = _infer_datetime_column(df)
    group_column = _infer_group_column(df)

    if normalized_task_type == "time_series":
        if not datetime_column:
            raise ValueError("时间序列任务未检测到可排序的时间字段，无法生成时间顺序切分方案")
        split_method = "time_series"
        split_column = datetime_column
        strategy_detail = f"按时间字段 {datetime_column} 严格顺序切分"
        questions_for_business.append(f"请确认 {datetime_column} 是预测时点之前即可获得的时间字段。")
    elif group_column:
        split_method = "group"
        split_column = group_column
        strategy_detail = f"按分组字段 {group_column} 做 Group split，防止同实体泄漏"
        questions_for_business.append(f"请确认 {group_column} 代表同一实体粒度，适合作为分组切分字段。")
    elif normalized_task_type == "classification":
        split_method = "stratified"
        strategy_detail = "基于目标列做分层切分"
        if not _can_stratify(df[target_column]):
            warnings.append("分类标签分布过于稀疏，默认建议退化为普通随机切分。")
            split_method = "random"
            strategy_detail = "分类标签稀疏，建议采用普通随机切分"
    elif normalized_task_type == "regression":
        if _build_regression_bins(df[target_column]) is not None:
            split_method = "binned_regression"
            strategy_detail = "按目标列分桶后做近似分层切分"
        else:
            warnings.append("目标分布分桶条件不足，默认建议采用普通随机切分。")
            split_method = "random"
            strategy_detail = "回归任务采用普通随机切分"
    else:
        warnings.append(f"未识别的任务类型 {normalized_task_type}，默认建议采用普通随机切分。")

    config = SplitConfig(
        task_type=normalized_task_type,
        split_method=split_method,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        split_column=split_column,
        has_validation_split=has_validation_split,
        strategy_detail=strategy_detail,
        warnings=warnings,
        questions_for_business=questions_for_business,
    )
    return config.to_dict()


def apply_split_config(
    data_path: str,
    target_column: str,
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"目标列不存在: {target_column}")

    split_method = (config.get("split_method") or "random").strip().lower()
    task_type = _normalize_task_type(config.get("task_type"))
    train_ratio = float(config.get("train_ratio", 0.8))
    valid_ratio = float(config.get("valid_ratio", 0.0))
    test_ratio = float(config.get("test_ratio", 0.2))
    split_column = config.get("split_column")
    strategy_detail = config.get("strategy_detail") or split_method
    warnings = list(config.get("warnings") or [])
    questions_for_business = list(config.get("questions_for_business") or [])

    if valid_ratio <= 0:
        valid_ratio = 0.0
    _validate_ratios(train_ratio, valid_ratio, test_ratio)

    if split_method == "time_series":
        if not split_column:
            split_column = _infer_datetime_column(df)
        if not split_column:
            raise ValueError("时间切分需要有效的时间字段")
        split_frames = _three_way_time_split(df, split_column, train_ratio, valid_ratio, test_ratio)
    elif split_method == "group":
        if not split_column:
            split_column = _infer_group_column(df)
        if not split_column:
            raise ValueError("Group split 需要有效的分组字段")
        split_frames = _three_way_group_split(df, split_column, train_ratio, valid_ratio, test_ratio)
    elif split_method == "stratified":
        stratify_labels = df[target_column]
        if not _can_stratify(stratify_labels):
            raise ValueError("当前目标列分布不足以支持分层切分")
        split_frames = _three_way_random_split(df, train_ratio, valid_ratio, test_ratio, stratify_labels)
    elif split_method == "binned_regression":
        regression_bins = _build_regression_bins(df[target_column])
        if regression_bins is None:
            raise ValueError("当前目标列分布不足以支持分桶近似分层切分")
        split_frames = _three_way_random_split(df, train_ratio, valid_ratio, test_ratio, regression_bins)
    else:
        split_method = "random"
        split_frames = _three_way_random_split(df, train_ratio, valid_ratio, test_ratio, None)

    train_df = split_frames["train"]
    valid_df = split_frames["valid"]
    test_df = split_frames["test"]

    split_paths = {"train_raw_path": None, "valid_raw_path": None, "test_raw_path": None}
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_path = output_path / "train_raw.csv"
        test_path = output_path / "test_raw.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        split_paths["train_raw_path"] = str(train_path)
        split_paths["test_raw_path"] = str(test_path)

        if valid_df is not None and len(valid_df) > 0:
            valid_path = output_path / "valid_raw.csv"
            valid_df.to_csv(valid_path, index=False)
            split_paths["valid_raw_path"] = str(valid_path)

    result = {
        "success": True,
        "data_path": data_path,
        "target_column": target_column,
        "task_type": task_type,
        "split_strategy": split_method,
        "strategy_detail": strategy_detail,
        "split_column": split_column,
        "has_validation_split": valid_df is not None and len(valid_df) > 0,
        "ratios": {
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
        },
        "counts": {
            "total_rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)) if valid_df is not None else 0,
            "test_rows": int(len(test_df)),
        },
        "warnings": warnings,
        "questions_for_business": questions_for_business,
        "split_paths": split_paths,
        "config": dict(config),
    }
    result["summary"] = render_split_summary(result)
    return result


def render_split_summary(result: Dict[str, Any]) -> str:
    counts = result["counts"]
    ratios = result.get("ratios", {})
    lines = [
        "# 数据集切分方案",
        "",
        "## 推荐方案",
        "",
        f"- 切分策略: {result['split_strategy']}",
        f"- 训练集比例: {ratios.get('train_ratio', 0):.0%}",
        f"- 验证集比例: {ratios.get('valid_ratio', 0):.0%}",
        f"- 测试集比例: {ratios.get('test_ratio', 0):.0%}",
        f"- 训练集行数: {counts['train_rows']}",
        f"- 验证集行数: {counts['valid_rows']}",
        f"- 测试集行数: {counts['test_rows']}",
        f"- 目标列: {result['target_column']}",
        f"- 任务类型: {result['task_type']}",
        "",
        "## 原则",
        "",
        "- valid 用于选择方案与调参。",
        "- test 只做最终评估，不参与任何拟合和调参。",
        "- 切分应发生在数据清洗前，后续规则和拟合默认以 train 为主。",
        "",
        "## 选择理由",
        "",
        f"- {result['strategy_detail']}",
    ]

    if result.get("split_column"):
        lines.append(f"- 关键切分字段: {result['split_column']}")

    if result.get("warnings"):
        lines.extend(["", "## 风险提示", ""])
        lines.extend([f"- {item}" for item in result["warnings"]])

    if result.get("questions_for_business"):
        lines.extend(["", "## 需要用户确认", ""])
        lines.extend([f"- {item}" for item in result["questions_for_business"]])

    return "\n".join(lines)


class DataSplittingAgent(ReActAgent):
    """数据集切分 Agent。"""

    def __init__(self, llm: Any = None, session_id: str = None, verbose: bool = False):
        super().__init__(llm=llm, session_id=session_id, max_iterations=8, verbose=verbose)
        self.data_path: Optional[str] = None
        self.target_column: Optional[str] = None
        self.task_type: str = "classification"
        self.problem_definition: Optional[Dict[str, Any]] = None
        self.split_plan: Optional[str] = None
        self.split_code: Optional[str] = None
        self.split_config: Optional[Dict[str, Any]] = None
        self.data_info: Optional[Dict[str, Any]] = None
        self.config_loader = get_config_loader()
        self.skill_loader = get_skill_loader()

    def _register_default_tools(self):
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())

    def get_system_prompt(self) -> str:
        return self.config_loader.get_prompt("data_splitting", "system_prompt")

    def _collect_data_context(self, data_path: str, target_column: str, task_type: str) -> Tuple[str, Dict[str, Any]]:
        df = pd.read_csv(data_path)
        self.data_info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "target_exists": target_column in df.columns,
            "target_unique": df[target_column].nunique() if target_column in df.columns else 0,
            "datetime_column": _infer_datetime_column(df),
            "group_column": _infer_group_column(df),
        }

        default_config = build_default_split_config(data_path, target_column, task_type, self.problem_definition)
        regression_bins = _build_regression_bins(df[target_column]) if target_column in df.columns else None
        current_data_context = f"""
## 当前数据事实

- 数据路径: {data_path}
- 数据形状: {df.shape[0]} 行 × {df.shape[1]} 列
- 目标列: {target_column}
- 任务类型: {task_type}
- 是否检测到时间字段: {self.data_info['datetime_column'] or '否'}
- 是否检测到分组字段: {self.data_info['group_column'] or '否'}
- 默认建议切分方法: {default_config['split_method']}
- 默认建议比例: train={default_config['train_ratio']:.0%}, valid={default_config['valid_ratio']:.0%}, test={default_config['test_ratio']:.0%}
- 回归分桶分层是否可用: {'是' if regression_bins is not None else '否'}

## 字段预览

{', '.join(df.columns[:30])}

重要：你必须输出可执行前的切分方案，而不是直接执行切分。
"""
        return current_data_context, default_config

    def generate_split_plan(
        self,
        data_path: str,
        target_column: str,
        task_type: str = "classification",
        problem_definition: Optional[Dict[str, Any]] = None,
        task_description: str = "",
    ) -> str:
        self.data_path = data_path
        self.target_column = target_column
        self.task_type = _normalize_task_type(task_type)
        self.problem_definition = problem_definition or {}

        current_data_context, default_config = self._collect_data_context(data_path, target_column, self.task_type)
        self.split_config = default_config

        task_context = ""
        if task_description:
            task_context = f"""
## 用户任务描述

{task_description}

"""

        problem_context = ""
        if self.problem_definition:
            problem_context = f"""
## 已确认的问题定义

{json.dumps(self.problem_definition, ensure_ascii=False, indent=2)}

"""

        prompt_template = self.config_loader.get_prompt("data_splitting", "plan_generation")
        user_input = prompt_template.format(
            task_context=task_context,
            problem_context=problem_context,
            current_data_context=current_data_context,
            suggested_config_json=json.dumps(default_config, ensure_ascii=False, indent=2),
        )

        result = self.run(user_input, stage="data_splitting_plan")
        self.split_plan = result.get("answer", "")
        return self.split_plan

    def _parse_modifications(self, modifications: Optional[str]) -> Dict[str, Any]:
        if not modifications or not modifications.strip():
            return dict(self.split_config or {})

        merged = dict(self.split_config or {})
        text = modifications.strip()

        json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(1))
                if isinstance(payload, dict):
                    merged.update(payload)
                    return merged
            except Exception:
                pass

        raw_json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if raw_json_match:
            try:
                payload = json.loads(raw_json_match.group(0))
                if isinstance(payload, dict):
                    merged.update(payload)
                    return merged
            except Exception:
                pass

        ratio_patterns = {
            "train_ratio": [r"train\s*[:=]\s*(0?\.\d+|\d+%)", r"训练集(?:比例)?\s*[:：=]?\s*(0?\.\d+|\d+%)"],
            "valid_ratio": [r"valid\s*[:=]\s*(0?\.\d+|\d+%)", r"验证集(?:比例)?\s*[:：=]?\s*(0?\.\d+|\d+%)"],
            "test_ratio": [r"test\s*[:=]\s*(0?\.\d+|\d+%)", r"测试集(?:比例)?\s*[:：=]?\s*(0?\.\d+|\d+%)"],
        }
        for key, patterns in ratio_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    value = match.group(1)
                    merged[key] = float(value[:-1]) / 100.0 if value.endswith("%") else float(value)
                    break

        method_aliases = {
            "stratified": ["stratified", "stratify", "分层"],
            "binned_regression": ["binned", "分桶", "近似分层"],
            "time_series": ["time_series", "时间序列", "按时间"],
            "group": ["group split", "group", "分组", "按用户", "按设备", "按病例"],
            "random": ["random", "普通切分", "随机切分"],
        }
        lowered = text.lower()
        for method, aliases in method_aliases.items():
            if any(alias in lowered for alias in aliases):
                merged["split_method"] = method
                break

        column_match = re.search(r"(?:split_column|分组字段|时间字段)\s*[:：=]\s*([A-Za-z0-9_]+)", text)
        if column_match:
            merged["split_column"] = column_match.group(1)

        if merged.get("valid_ratio", 0) <= 0:
            merged["has_validation_split"] = False
        else:
            merged["has_validation_split"] = True
        return merged

    def _build_split_code(self, config: Dict[str, Any], output_dir: str) -> str:
        return f'''
import json
from automl_react.agents.data_splitting_agent import apply_split_config

data_path = {json.dumps(self.data_path, ensure_ascii=False)}
target_column = {json.dumps(self.target_column, ensure_ascii=False)}
output_dir = {json.dumps(output_dir, ensure_ascii=False)}
split_config = {json.dumps(config, ensure_ascii=False, indent=2)}

split_result = apply_split_config(
    data_path=data_path,
    target_column=target_column,
    config=split_config,
    output_dir=output_dir,
)

train_raw_path = split_result["split_paths"].get("train_raw_path")
valid_raw_path = split_result["split_paths"].get("valid_raw_path")
test_raw_path = split_result["split_paths"].get("test_raw_path")
split_strategy = split_result.get("split_strategy")
counts = split_result.get("counts", {{}})
'''

    def generate_split_code(self, modifications: Optional[str] = None) -> str:
        if not self.split_plan:
            raise ValueError("请先生成数据集切分方案")

        merged_config = self._parse_modifications(modifications)
        output_dir = str(self.asset_manager.session_dir / "data")
        self.split_config = merged_config
        self.split_code = self._build_split_code(merged_config, output_dir)

        self.asset_manager.save_code(
            code=self.split_code,
            filename="data_splitting.py",
            metadata={
                "stage": "data_splitting",
                "data_path": self.data_path,
                "target_column": self.target_column,
                "task_type": self.task_type,
                "split_config": merged_config,
                "timestamp": datetime.now().isoformat(),
            },
        )
        return self.split_code

    def execute_split(self, code: Optional[str] = None) -> Dict[str, Any]:
        split_code = code or self.split_code
        if not split_code:
            raise ValueError("请先生成切分代码")

        code_gen = CodeGenerator()
        output_dir = str(self.asset_manager.session_dir / "data")
        context = {
            "data_path": self.data_path,
            "target_column": self.target_column,
            "output_dir": output_dir,
        }
        exec_result = code_gen.execute_code(split_code, context)
        if not exec_result.success:
            fallback_result = apply_split_config(
                data_path=self.data_path,
                target_column=self.target_column,
                config=self.split_config or {},
                output_dir=output_dir,
            )
            fallback_result["execution_mode"] = "deterministic_fallback"
            return fallback_result

        result = apply_split_config(
            data_path=self.data_path,
            target_column=self.target_column,
            config=self.split_config or {},
            output_dir=output_dir,
        )
        result["execution_mode"] = "generated_code"
        result["execution_output"] = exec_result.output
        return result


def run_dataset_split(
    data_path: str,
    target_column: str,
    task_type: str = "classification",
    problem_definition: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    effective_config = config or build_default_split_config(
        data_path=data_path,
        target_column=target_column,
        task_type=task_type,
        problem_definition=problem_definition,
    )
    return apply_split_config(
        data_path=data_path,
        target_column=target_column,
        config=effective_config,
        output_dir=output_dir,
    )
