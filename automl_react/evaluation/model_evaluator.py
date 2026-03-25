"""
模型评估模块

实现模型评估，参考 ml-model-eval-benchmark skill
"""

import json
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..skills_loader import get_skill_loader
from ..config import get_config_loader
from ..assets import get_asset_manager


class ModelEvaluator:
    """
    模型评估器

    基于 ml-model-eval-benchmark skill 进行模型评估

    Attributes:
        session_id: 会话ID
        skill_loader: Skill 加载器
        config_loader: 配置加载器
        asset_manager: 资产管理器
    """

    def __init__(self, session_id: str = None):
        self.session_id = session_id or "default"
        self.skill_loader = get_skill_loader()
        self.config_loader = get_config_loader()
        self.asset_manager = get_asset_manager(session_id=self.session_id)

    def load_benchmark_guide(self) -> str:
        """
        加载 ml-model-eval-benchmark skill 的 benchmarking-guide.md

        Returns:
            指南内容
        """
        return self.skill_loader.get_skill_reference(
            "ml-model-eval-benchmark-0.1.0",
            "benchmarking-guide.md"
        ) or ""

    def evaluate_model(
        self,
        model_path: str,
        data_path: str,
        target_column: str,
        task_type: str = "classification",
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        评估模型

        Args:
            model_path: 模型文件路径
            data_path: 测试数据路径
            target_column: 目标列名
            task_type: 任务类型
            metrics: 评估指标列表

        Returns:
            评估结果
        """
        # 加载评估指南
        benchmark_guide = self.load_benchmark_guide()

        # 从配置加载评估参数
        try:
            eval_config = self.config_loader.get_workflow_config("evaluation")
            default_metrics = eval_config.get("metrics", {})
        except KeyError:
            default_metrics = {}

        # 确定评估指标
        if metrics is None:
            if task_type == "classification":
                metrics = ["accuracy", "precision", "recall", "f1", "auc"]
            else:
                metrics = ["mse", "rmse", "mae", "r2"]

        training_context = self._load_training_context()
        expected_feature_names = self._extract_expected_feature_names(training_context)
        target_transform = self._extract_target_transform(training_context)

        # 构建评估代码
        eval_code = self._generate_evaluation_code(
            model_path=model_path,
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,
            metrics=metrics,
            expected_feature_names=expected_feature_names,
            target_transform=target_transform,
        )

        # 执行评估
        try:
            local_vars = {}
            exec(eval_code, local_vars)

            evaluation_results = local_vars.get("evaluation_results", {})
            diagnostics = local_vars.get("evaluation_diagnostics", {})

            # 添加元数据
            result = {
                "success": True,
                "model_path": model_path,
                "data_path": data_path,
                "target_column": target_column,
                "task_type": task_type,
                "metrics": evaluation_results,
                "diagnostics": diagnostics,
                "timestamp": datetime.now().isoformat()
            }

            # 保存评估结果
            self.asset_manager.save_data(
                data=json.dumps(result, ensure_ascii=False, indent=2),
                filename="evaluation.json",
                asset_type="reports",
                metadata=result
            )

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _load_training_context(self) -> Dict[str, Any]:
        """读取训练阶段产物，为评估恢复列契约和目标变换。"""
        context: Dict[str, Any] = {}

        training_summary = self.asset_manager.read_asset("models", "training_summary.json")
        if training_summary:
            try:
                context["training_summary"] = json.loads(training_summary)
            except Exception:
                context["training_summary"] = {}

        training_result = self.asset_manager.read_asset("models", "model_training_result.json")
        if training_result:
            try:
                context["training_result"] = json.loads(training_result)
            except Exception:
                context["training_result"] = {}

        return context

    def _extract_expected_feature_names(self, training_context: Dict[str, Any]) -> List[str]:
        """从训练阶段产物中恢复期望的特征顺序。"""
        training_result = training_context.get("training_result") or {}
        training_summary = training_context.get("training_summary") or {}

        feature_names = training_result.get("selected_feature_names")
        if isinstance(feature_names, list) and feature_names:
            return feature_names

        feature_names = training_summary.get("selected_feature_names")
        if isinstance(feature_names, list) and feature_names:
            return feature_names

        return []

    def _extract_target_transform(self, training_context: Dict[str, Any]) -> Optional[str]:
        """从训练阶段产物中恢复目标变换类型。"""
        training_summary = training_context.get("training_summary") or {}
        training_result = training_context.get("training_result") or {}

        for source in (training_summary, training_result):
            raw_transform = source.get("target_transform")
            normalized = self._normalize_target_transform(raw_transform)
            if normalized:
                return normalized

        return None

    @staticmethod
    def _normalize_target_transform(raw_transform: Any) -> Optional[str]:
        """规范化目标变换标记。"""
        if not raw_transform or not isinstance(raw_transform, str):
            return None

        lowered = raw_transform.lower()
        if "log1p" in lowered:
            return "log1p"
        return None

    def _generate_evaluation_code(
        self,
        model_path: str,
        data_path: str,
        target_column: str,
        task_type: str,
        metrics: List[str],
        expected_feature_names: List[str],
        target_transform: Optional[str],
    ) -> str:
        """
        生成评估代码

        Args:
            model_path: 模型文件路径
            data_path: 测试数据路径
            target_column: 目标列名
            task_type: 任务类型
            metrics: 评估指标列表

        Returns:
            评估代码
        """
        metrics_code = ", ".join([f"'{m}'" for m in metrics])
        expected_feature_names_json = json.dumps(expected_feature_names, ensure_ascii=False)
        target_transform_json = json.dumps(target_transform, ensure_ascii=False)

        code = f'''
import pandas as pd
import joblib
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
)
import numpy as np


EXPECTED_FEATURE_NAMES = {expected_feature_names_json}
INFERRED_TARGET_TRANSFORM = {target_transform_json}


def load_model_artifact(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def align_features(features, expected_feature_names, model, artifact=None):
    diagnostics = {{
        "feature_alignment_source": None,
        "missing_features": [],
        "extra_features": [],
    }}

    if isinstance(artifact, dict):
        packaged_feature_names = artifact.get("selected_feature_names")
        if isinstance(packaged_feature_names, list) and packaged_feature_names:
            diagnostics["feature_alignment_source"] = "artifact_package"
            diagnostics["missing_features"] = [name for name in packaged_feature_names if name not in features.columns]
            diagnostics["extra_features"] = [name for name in features.columns if name not in packaged_feature_names]
            aligned = features.reindex(columns=packaged_feature_names, fill_value=0)
            return aligned, diagnostics

    if isinstance(expected_feature_names, list) and expected_feature_names:
        diagnostics["feature_alignment_source"] = "training_artifacts"
        diagnostics["missing_features"] = [name for name in expected_feature_names if name not in features.columns]
        diagnostics["extra_features"] = [name for name in features.columns if name not in expected_feature_names]
        aligned = features.reindex(columns=expected_feature_names, fill_value=0)
        return aligned, diagnostics

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        diagnostics["feature_alignment_source"] = "model_artifact"
        diagnostics["missing_features"] = [name for name in expected if name not in features.columns]
        diagnostics["extra_features"] = [name for name in features.columns if name not in expected]
        aligned = features.reindex(columns=expected, fill_value=0)
        return aligned, diagnostics

    diagnostics["feature_alignment_source"] = "input_dataframe"
    return features, diagnostics


def resolve_target_transform(artifact, inferred_target_transform):
    if isinstance(artifact, dict):
        raw_transform = artifact.get("target_transform")
        if isinstance(raw_transform, str) and raw_transform.lower() == "log1p":
            return "log1p", "artifact_package"

    if inferred_target_transform:
        return inferred_target_transform, "training_artifacts"

    return None, None


def predict_from_artifact(artifact, features, expected_feature_names, inferred_target_transform):
    diagnostics = {{}}

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        preprocessor = artifact.get("preprocessor")
        aligned_features, alignment_info = align_features(features, expected_feature_names, model, artifact)
        target_transform, target_transform_source = resolve_target_transform(artifact, inferred_target_transform)

        pipeline_has_internal_preprocessor = (
            hasattr(model, "named_steps") and
            isinstance(getattr(model, "named_steps", None), dict) and
            "preprocessor" in model.named_steps
        )

        if preprocessor is not None and not pipeline_has_internal_preprocessor:
            model_input = preprocessor.transform(aligned_features)
            diagnostics["preprocessor_application"] = "artifact_preprocessor"
        else:
            model_input = aligned_features
            diagnostics["preprocessor_application"] = (
                "pipeline_internal" if pipeline_has_internal_preprocessor else "none"
            )

        predictions = model.predict(model_input)

        if target_transform == "log1p":
            predictions = np.expm1(predictions)

        diagnostics.update(alignment_info)
        diagnostics["target_transform_applied"] = target_transform
        diagnostics["target_transform_source"] = target_transform_source
        diagnostics["artifact_type"] = "package"
        return predictions, diagnostics

    aligned_features, alignment_info = align_features(features, expected_feature_names, artifact)
    target_transform, target_transform_source = resolve_target_transform(artifact, inferred_target_transform)
    predictions = artifact.predict(aligned_features)

    if target_transform == "log1p":
        predictions = np.expm1(predictions)

    diagnostics.update(alignment_info)
    diagnostics["target_transform_applied"] = target_transform
    diagnostics["target_transform_source"] = target_transform_source
    diagnostics["artifact_type"] = "raw_model"
    return predictions, diagnostics


def rmsle_score(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1e-6)
    y_true = np.maximum(y_true, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# 加载数据
df = pd.read_csv("{data_path}")
X = df.drop(columns=["{target_column}"])
y = df["{target_column}"]

# 加载模型
model_artifact = load_model_artifact("{model_path}")

# 预测
y_pred, evaluation_diagnostics = predict_from_artifact(
    model_artifact,
    X,
    EXPECTED_FEATURE_NAMES,
    INFERRED_TARGET_TRANSFORM,
)

# 计算评估指标
evaluation_results = {{}}

if "{task_type}" == "classification":
    if "accuracy" in [{metrics_code}]:
        evaluation_results["accuracy"] = float(accuracy_score(y, y_pred))
    if "precision" in [{metrics_code}]:
        evaluation_results["precision"] = float(precision_score(y, y_pred, average="weighted"))
    if "recall" in [{metrics_code}]:
        evaluation_results["recall"] = float(recall_score(y, y_pred, average="weighted"))
    if "f1" in [{metrics_code}]:
        evaluation_results["f1"] = float(f1_score(y, y_pred, average="weighted"))
    if "auc" in [{metrics_code}] and hasattr(model_artifact, "predict_proba"):
        y_pred_proba = model_artifact.predict_proba(X)
        if y_pred_proba.shape[1] == 2:
            evaluation_results["auc"] = float(roc_auc_score(y, y_pred_proba[:, 1]))
        else:
            evaluation_results["auc"] = float(roc_auc_score(y, y_pred_proba, multi_class="ovr"))
else:
    if "mse" in [{metrics_code}]:
        evaluation_results["mse"] = float(mean_squared_error(y, y_pred))
    if "rmse" in [{metrics_code}]:
        evaluation_results["rmse"] = float(np.sqrt(mean_squared_error(y, y_pred)))
    if "mae" in [{metrics_code}]:
        evaluation_results["mae"] = float(mean_absolute_error(y, y_pred))
    if "r2" in [{metrics_code}]:
        evaluation_results["r2"] = float(r2_score(y, y_pred))
    evaluation_results["rmsle"] = float(rmsle_score(y, y_pred))
    evaluation_results["mape"] = float(np.mean(np.abs((y - y_pred) / (y + 1e-6))) * 100)

print("评估完成:", evaluation_results)
'''
        return code

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        获取评估摘要

        Returns:
            评估摘要
        """
        # 读取评估结果
        eval_data = self.asset_manager.read_asset("reports", "evaluation.json")

        if eval_data:
            return json.loads(eval_data)

        return {"success": False, "message": "未找到评估结果"}
