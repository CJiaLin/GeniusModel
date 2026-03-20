"""
模型评估模块

实现模型评估，参考 ml-model-eval-benchmark skill
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

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

        # 构建评估代码
        eval_code = self._generate_evaluation_code(
            model_path=model_path,
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,
            metrics=metrics
        )

        # 执行评估
        try:
            local_vars = {}
            exec(eval_code, local_vars)

            evaluation_results = local_vars.get("evaluation_results", {})

            # 添加元数据
            result = {
                "success": True,
                "model_path": model_path,
                "data_path": data_path,
                "target_column": target_column,
                "task_type": task_type,
                "metrics": evaluation_results,
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

    def _generate_evaluation_code(
        self,
        model_path: str,
        data_path: str,
        target_column: str,
        task_type: str,
        metrics: List[str]
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

        code = f'''
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np

# 加载数据
df = pd.read_csv("{data_path}")
X = df.drop(columns=["{target_column}"])
y = df["{target_column}"]

# 加载模型
model = joblib.load("{model_path}")

# 预测
y_pred = model.predict(X)

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
    if "auc" in [{metrics_code}] and hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X)
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
