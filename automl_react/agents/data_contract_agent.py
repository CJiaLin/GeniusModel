"""
数据契约检查 Agent 模块

在进入清洗/建模之前，先做"能不能建模"的可行性检查。
纯确定性逻辑，不依赖 LLM。
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..assets import get_asset_manager


# ---------------------------------------------------------------------------
# 单项检查函数 —— 每个返回 (severity, title, detail)
#   severity: "blocker" | "warning" | "info"
# ---------------------------------------------------------------------------

def _check_target_exists(
    df: pd.DataFrame, target_column: str
) -> Tuple[str, str, str]:
    if target_column not in df.columns:
        return (
            "blocker",
            "目标列不存在",
            f"数据中不存在名为 '{target_column}' 的列。可用列: {', '.join(df.columns[:20])}",
        )
    return ("info", "目标列存在", f"目标列 '{target_column}' 已找到。")


def _check_sample_size(
    df: pd.DataFrame, min_rows: int = 30
) -> Tuple[str, str, str]:
    n = len(df)
    if n < min_rows:
        return (
            "blocker",
            "样本量不足",
            f"数据仅有 {n} 行，建议至少 {min_rows} 行。",
        )
    if n < 200:
        return (
            "warning",
            "样本量偏少",
            f"数据有 {n} 行，对复杂模型可能不够，建议 >= 200。",
        )
    return ("info", "样本量充足", f"数据有 {n} 行。")


def _check_primary_key(df: pd.DataFrame) -> Tuple[str, str, str]:
    """检查是否有疑似主键/唯一标识列。"""
    candidates = []
    for col in df.columns:
        if df[col].nunique() == len(df) and df[col].notna().all():
            candidates.append(col)
    if candidates:
        return (
            "info",
            "存在唯一标识列",
            f"以下列每行唯一: {', '.join(candidates[:5])}。建模时通常应排除。",
        )
    return (
        "info",
        "未检测到唯一标识列",
        "无列的唯一值数等于行数，不影响建模。",
    )


def _check_target_missing(
    df: pd.DataFrame, target_column: str
) -> Tuple[str, str, str]:
    if target_column not in df.columns:
        return ("info", "跳过标签缺失检查", "目标列不存在，已在前置检查覆盖。")
    missing_ratio = df[target_column].isna().mean()
    if missing_ratio > 0.5:
        return (
            "blocker",
            "标签严重缺失",
            f"目标列 '{target_column}' 缺失率 {missing_ratio:.1%}，超过 50%，无法建模。",
        )
    if missing_ratio > 0.1:
        return (
            "warning",
            "标签缺失偏多",
            f"目标列 '{target_column}' 缺失率 {missing_ratio:.1%}，可能影响模型质量。",
        )
    return (
        "info",
        "标签完整性良好",
        f"目标列 '{target_column}' 缺失率 {missing_ratio:.1%}。",
    )


def _check_target_distribution(
    df: pd.DataFrame, target_column: str, task_type: str
) -> Tuple[str, str, str]:
    if target_column not in df.columns:
        return ("info", "跳过目标分布检查", "目标列不存在。")
    series = df[target_column].dropna()
    if len(series) == 0:
        return ("blocker", "目标列全部缺失", "目标列无有效值。")

    if task_type == "classification":
        vc = series.value_counts()
        n_classes = len(vc)
        if n_classes < 2:
            return ("blocker", "目标列只有一个类别", f"类别数={n_classes}，无法训练分类器。")
        minority_ratio = vc.min() / vc.sum()
        if minority_ratio < 0.01:
            return (
                "warning",
                "目标严重不平衡",
                f"最少类别仅占 {minority_ratio:.2%}（共 {n_classes} 类），需要不平衡策略。",
            )
        if minority_ratio < 0.05:
            return (
                "warning",
                "目标存在不平衡",
                f"最少类别占 {minority_ratio:.2%}（共 {n_classes} 类），建议关注。",
            )
        return ("info", "目标分布正常", f"共 {n_classes} 类，最少类别占 {minority_ratio:.2%}。")
    else:
        # 回归
        if series.std() == 0:
            return ("blocker", "目标列方差为零", "所有值相同，无法训练回归模型。")
        skew = float(series.skew())
        detail = f"均值={series.mean():.4g}, 中位数={series.median():.4g}, 标准差={series.std():.4g}, 偏度={skew:.2f}"
        if abs(skew) > 5:
            return ("warning", "目标列分布极端偏斜", f"偏度={skew:.2f}，建议对数变换。{detail}")
        return ("info", "目标分布正常", detail)


def _check_datetime_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """检测疑似时间列，提示可能存在未来信息泄漏。"""
    datetime_cols = list(df.select_dtypes(include=["datetime64"]).columns)
    # 也检查字符串列中的日期模式
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(20).astype(str)
        date_like = sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
        if date_like.mean() > 0.5:
            datetime_cols.append(col)
    if datetime_cols:
        return (
            "warning",
            "检测到时间字段",
            f"以下列疑似时间字段: {', '.join(datetime_cols[:10])}。"
            "请确认这些字段在预测时点是否可用，避免未来信息泄漏。",
        )
    return ("info", "未检测到时间字段", "无明显日期格式列。")


def _check_id_columns(
    df: pd.DataFrame, target_column: str
) -> Tuple[str, str, str]:
    """检测疑似 ID 列（唯一值多、无预测意义），建模时应排除。"""
    id_candidates = []
    for col in df.columns:
        if col == target_column:
            continue
        nunique_ratio = df[col].nunique() / max(len(df), 1)
        col_lower = col.lower()
        if nunique_ratio > 0.9 and any(kw in col_lower for kw in ("id", "index", "key", "code", "no", "number")):
            id_candidates.append(col)
        elif nunique_ratio == 1.0 and df[col].dtype in ("int64", "float64", "object"):
            id_candidates.append(col)
    if id_candidates:
        return (
            "warning",
            "训练时不可用字段",
            f"以下列疑似唯一标识/ID 列，建模时应排除: {', '.join(id_candidates[:10])}",
        )
    return ("info", "未检测到 ID 列", "")


def _check_leakage(
    df: pd.DataFrame, target_column: str, task_type: str
) -> Tuple[str, str, str]:
    """检查与目标列相关性异常高的特征，提示可能泄漏。"""
    if target_column not in df.columns:
        return ("info", "跳过泄漏检查", "目标列不存在。")
    target = df[target_column]
    if not pd.api.types.is_numeric_dtype(target):
        return ("info", "跳过泄漏检查", "目标列非数值型，暂不做相关性泄漏检查。")

    high_corr = []
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if col == target_column:
            continue
        try:
            corr = df[col].corr(target)
            if abs(corr) > 0.95:
                high_corr.append((col, round(corr, 4)))
        except Exception:
            continue

    if high_corr:
        details = ", ".join(f"{c}(r={r})" for c, r in high_corr[:10])
        return (
            "warning",
            "疑似泄漏字段",
            f"以下字段与目标高度相关(|r|>0.95): {details}。请确认是否为合法特征。",
        )
    return ("info", "未检测到明显泄漏字段", "无特征与目标相关系数超过 0.95。")


# ---------------------------------------------------------------------------
# 汇总执行
# ---------------------------------------------------------------------------

def run_data_contract_checks(
    data_path: str,
    target_column: str,
    task_type: str = "regression",
    problem_definition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    执行全部数据契约检查，返回结构化结果。

    Returns:
        {
            "modelable": bool,
            "checks": [...],
            "risk_list": [...],
            "questions_for_business": [...],
            "summary": str,
        }
    """
    df = pd.read_csv(data_path)

    checkers = [
        _check_target_exists(df, target_column),
        _check_sample_size(df),
        _check_primary_key(df),
        _check_target_missing(df, target_column),
        _check_target_distribution(df, target_column, task_type),
        _check_datetime_columns(df),
        _check_id_columns(df, target_column),
        _check_leakage(df, target_column, task_type),
    ]

    checks = []
    for severity, title, detail in checkers:
        checks.append({"severity": severity, "title": title, "detail": detail})

    blockers = [c for c in checks if c["severity"] == "blocker"]
    warnings = [c for c in checks if c["severity"] == "warning"]
    infos = [c for c in checks if c["severity"] == "info"]

    modelable = len(blockers) == 0

    risk_list = [f"[{c['severity'].upper()}] {c['title']}: {c['detail']}" for c in blockers + warnings]

    questions = []
    for c in warnings:
        if "时间" in c["title"] or "泄漏" in c["title"]:
            questions.append(f"请确认: {c['detail']}")
        if "ID" in c["title"] or "不可用" in c["title"]:
            questions.append(f"请确认以下字段是否应该排除: {c['detail']}")
        if "不平衡" in c["title"]:
            questions.append(f"建议确认: {c['detail']}")
    # 如果问题定义里有 open_questions，也合并进来
    if problem_definition:
        for q in problem_definition.get("open_questions", []):
            if q and str(q).strip():
                questions.append(f"[问题定义待确认] {q}")

    # 生成 Markdown 报告
    lines = ["# 数据契约检查报告\n"]
    lines.append(f"**结论: {'✅ 可建模' if modelable else '❌ 不可建模'}**\n")
    lines.append(f"- 数据路径: `{data_path}`")
    lines.append(f"- 目标列: `{target_column}`")
    lines.append(f"- 任务类型: `{task_type}`")
    lines.append(f"- 样本数: {len(df)}")
    lines.append(f"- 特征数: {len(df.columns)}")
    lines.append(f"- 检查项数: {len(checks)} (阻断={len(blockers)}, 警告={len(warnings)}, 信息={len(infos)})\n")

    if blockers:
        lines.append("## ❌ 阻断项 (必须解决)\n")
        for c in blockers:
            lines.append(f"- **{c['title']}**: {c['detail']}")
        lines.append("")

    if warnings:
        lines.append("## ⚠️ 风险项 (建议确认)\n")
        for c in warnings:
            lines.append(f"- **{c['title']}**: {c['detail']}")
        lines.append("")

    if infos:
        lines.append("## ✅ 信息项\n")
        for c in infos:
            lines.append(f"- **{c['title']}**: {c['detail']}")
        lines.append("")

    if questions:
        lines.append("## ❓ 需要业务确认的问题\n")
        for i, q in enumerate(questions, 1):
            lines.append(f"{i}. {q}")
        lines.append("")

    lines.append("---\n")
    lines.append("*以上检查为确定性规则，不依赖模型。通过后可进入数据清洗阶段。*")

    summary = "\n".join(lines)

    return {
        "modelable": modelable,
        "checks": checks,
        "risk_list": risk_list,
        "questions_for_business": questions,
        "summary": summary,
        "stats": {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "n_blockers": len(blockers),
            "n_warnings": len(warnings),
            "n_infos": len(infos),
        },
    }
