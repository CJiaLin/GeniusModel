"""
模型训练 Agent 模块

实现建模方案生成、用户确认、代码生成与执行
"""

import json
import os
import re
import pickle
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import joblib

from ..core.react_agent import ReActAgent, ConfirmationRequired
from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
from ..config import get_config_loader


class ModelTrainingAgent(ReActAgent):
    """
    模型训练 Agent

    基于 ReAct 架构的模型训练 Agent，支持：
    1. 建模方案生成（参考 afrexai-ml-engineering skill）
    2. 用户确认流程
    3. 代码生成与执行
    4. 模型保存

    Attributes:
        session_id: 会话ID
        data_path: 数据文件路径
        target_column: 目标列名
        task_type: 任务类型
        model_plan: 建模方案
        model_code: 建模代码
    """

    def __init__(self, llm: Any = None, session_id: str = None, verbose: bool = False):
        super().__init__(llm=llm, session_id=session_id, max_iterations=10, verbose=verbose)
        self.data_path: Optional[str] = None
        self.target_column: Optional[str] = None
        self.task_type: str = "classification"
        self.data_info: Optional[Dict] = None
        self.model_plan: Optional[str] = None
        self.model_code: Optional[str] = None
        self.model_result: Optional[Dict[str, Any]] = None
        self.evaluation_plan: Optional[str] = None
        self.evaluation_result: Optional[Dict[str, Any]] = None
        self.config_loader = get_config_loader()

    def _register_default_tools(self):
        """注册默认工具"""
        super()._register_default_tools()
        from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
        from ..tools.stage_tools import StageResultTool
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())
        self.register_tool("query_stage_result", StageResultTool(session_id=self.session_id))

    def _get_training_artifact_paths(self) -> Dict[str, str]:
        """获取模型训练阶段的标准资产路径。"""
        session_dir = self.asset_manager.session_dir
        return {
            "model_path": str(session_dir / "models" / "trained_model.pkl"),
            "train_split_path": str(session_dir / "data" / "features_train.csv"),
            "valid_split_path": str(session_dir / "data" / "features_valid.csv"),
            "test_split_path": str(session_dir / "data" / "features_test.csv"),
            "training_summary_path": str(session_dir / "models" / "training_summary.json")
        }

    def _summarize_report(self, report: str, max_chars: int = 4000, priority_keywords: List[str] = None) -> str:
        """从 Markdown 报告中按优先级提取章节，避免暴力截断。

        策略：按 ## 标题拆分章节，优先保留包含 priority_keywords 的章节，
        剩余预算填充其他章节。确保不会截断表格或列表中间。
        """
        if not report or len(report) <= max_chars:
            return report

        priority_keywords = priority_keywords or []

        # 按 ## 标题拆分
        parts = re.split(r'\n(##\s+[^\n]+)', report)
        # parts: [header, '## title1', 'body1', '## title2', 'body2', ...]
        header = parts[0].strip()
        sections = []
        for i in range(1, len(parts) - 1, 2):
            title = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections.append((title, body))

        kept = [header]
        remaining_budget = max_chars - len(header) - 10

        # 第一轮：纳入高优先级章节
        included_indices = set()
        for idx, (title, body) in enumerate(sections):
            if any(kw in title for kw in priority_keywords):
                section_text = f"\n\n{title}\n\n{body}"
                if len(section_text) <= remaining_budget:
                    kept.append(section_text)
                    remaining_budget -= len(section_text)
                    included_indices.add(idx)

        # 第二轮：剩余预算填充其他章节
        for idx, (title, body) in enumerate(sections):
            if idx not in included_indices:
                section_text = f"\n\n{title}\n\n{body}"
                if len(section_text) <= remaining_budget:
                    kept.append(section_text)
                    remaining_budget -= len(section_text)

        return "".join(kept)

    def _read_training_summary(self, summary_path: str) -> Dict[str, Any]:
        """读取训练摘要文件。"""
        if not os.path.exists(summary_path):
            return {}

        try:
            with open(summary_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _normalize_target_transform(raw_transform: Any) -> Optional[str]:
        """规范化目标变换字段。"""
        if not raw_transform:
            return None
        # 兼容 dict 格式: {"train": "log1p", "inference": "expm1"}
        if isinstance(raw_transform, dict):
            for val in raw_transform.values():
                if isinstance(val, str) and "log1p" in val.lower():
                    return "log1p"
            return None
        if not isinstance(raw_transform, str):
            return None
        if "log1p" in raw_transform.lower():
            return "log1p"
        return None

    def _load_model_artifact(self, model_path: str) -> Any:
        """读取模型产物。"""
        try:
            return joblib.load(model_path)
        except Exception:
            with open(model_path, "rb") as file:
                return pickle.load(file)

    def _save_model_artifact(self, artifact: Any, model_path: str) -> None:
        """统一保存模型产物。"""
        joblib.dump(artifact, model_path)

    def _build_packaged_model_artifact(
        self,
        raw_model_artifact: Any,
        summary_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """将模型产物标准化为统一打包结构。"""
        if isinstance(raw_model_artifact, dict) and "model" in raw_model_artifact:
            packaged = dict(raw_model_artifact)
        else:
            packaged = {
                "model": raw_model_artifact,
                "preprocessor": None,
            }

        selected_feature_names = summary_payload.get("selected_feature_names")
        if not isinstance(selected_feature_names, list):
            selected_feature_names = []

        packaged["selected_feature_names"] = selected_feature_names
        packaged["target_transform"] = self._normalize_target_transform(
            summary_payload.get("target_transform")
        )
        packaged.setdefault("preprocessor", None)
        packaged["target_column"] = summary_payload.get("target_column", self.target_column)
        packaged["task_type"] = summary_payload.get("task_type", self.task_type)
        packaged["artifact_format"] = "model_package_v1"
        return packaged

    def _normalize_training_artifacts(self) -> Dict[str, Any]:
        """将训练产物收敛为统一模型包格式。"""
        artifact_paths = self._get_training_artifact_paths()
        model_path = artifact_paths["model_path"]
        summary_path = artifact_paths["training_summary_path"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        summary_payload = self._read_training_summary(summary_path)
        raw_artifact = self._load_model_artifact(model_path)
        packaged_artifact = self._build_packaged_model_artifact(raw_artifact, summary_payload)
        self._save_model_artifact(packaged_artifact, model_path)
        return packaged_artifact

    def _collect_training_result(self, execution_output: str = "", execution_error: Optional[str] = None) -> Dict[str, Any]:
        """基于已生成资产收集统一的模型训练结果。"""
        artifact_paths = self._get_training_artifact_paths()
        artifact_status = {key: os.path.exists(path) for key, path in artifact_paths.items()}
        summary_payload = self._read_training_summary(artifact_paths["training_summary_path"])
        valid_split_path = summary_payload.get("valid_split_path")
        has_validation_split = bool(valid_split_path)
        success = (
            artifact_status["model_path"]
            and artifact_status["train_split_path"]
            and artifact_status["test_split_path"]
            and artifact_status["training_summary_path"]
            and (artifact_status["valid_split_path"] if has_validation_split else True)
        )

        result_info = {
            "success": success,
            "model_path": artifact_paths["model_path"] if artifact_status["model_path"] else None,
            "train_split_path": artifact_paths["train_split_path"] if artifact_status["train_split_path"] else None,
            "valid_split_path": valid_split_path if has_validation_split else None,
            "test_split_path": artifact_paths["test_split_path"] if artifact_status["test_split_path"] else None,
            "training_summary_path": artifact_paths["training_summary_path"] if artifact_status["training_summary_path"] else None,
            "data_path": self.data_path,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "metrics": summary_payload.get("metrics", {}),
            "selected_feature_names": summary_payload.get("selected_feature_names", []),
            "target_transform": self._normalize_target_transform(summary_payload.get("target_transform")),
            "artifact_status": artifact_status,
            "execution_output": execution_output,
            "execution_error": execution_error,
            "timestamp": datetime.now().isoformat()
        }
        return result_info

    def _validate_training_outputs(self, output_path: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """模型训练输出校验：模型、切分文件、训练摘要必须全部存在且可读。"""
        import pandas as pd

        artifact_paths = self._get_training_artifact_paths()
        required_paths = [
            artifact_paths["model_path"],
            artifact_paths["train_split_path"],
            artifact_paths["test_split_path"],
            artifact_paths["training_summary_path"],
        ]
        if context.get("valid_split_path"):
            required_paths.append(artifact_paths["valid_split_path"])
        missing_files = [path for path in required_paths if not os.path.exists(path)]
        if missing_files:
            return False, f"缺少训练产物: {missing_files}"

        try:
            train_df = pd.read_csv(artifact_paths["train_split_path"])
            valid_df = pd.read_csv(artifact_paths["valid_split_path"]) if os.path.exists(artifact_paths["valid_split_path"]) else None
            test_df = pd.read_csv(artifact_paths["test_split_path"])
        except Exception as error:
            return False, f"训练/测试切分文件不可读: {error}"

        if train_df.empty or test_df.empty or (valid_df is not None and valid_df.empty):
            return False, "训练集、验证集或测试集为空"

        target_column = context.get("target_column") or self.target_column
        valid_missing_target = valid_df is not None and target_column not in valid_df.columns
        if target_column and (target_column not in train_df.columns or target_column not in test_df.columns or valid_missing_target):
            return False, f"切分文件中缺少目标列: {target_column}"

        summary_payload = self._read_training_summary(artifact_paths["training_summary_path"])
        required_summary_keys = [
            "metrics",
            "selected_feature_names",
            "target_column",
            "task_type",
            "target_transform",
            "model_path",
            "train_split_path",
            "valid_split_path",
            "test_split_path"
        ]
        missing_summary_keys = [key for key in required_summary_keys if key not in summary_payload]
        if missing_summary_keys:
            return False, f"训练摘要缺少字段: {missing_summary_keys}"

        try:
            model_artifact = self._load_model_artifact(artifact_paths["model_path"])
        except Exception as error:
            return False, f"模型文件不可读: {error}"

        if not isinstance(model_artifact, dict) or "model" not in model_artifact:
            return False, "模型文件不是标准打包结构，缺少 model 字段"

        for required_key in ["selected_feature_names", "target_transform", "preprocessor"]:
            if required_key not in model_artifact:
                return False, f"模型文件缺少字段: {required_key}"

        valid_shape = valid_df.shape if valid_df is not None else None
        return True, f"train={train_df.shape}, valid={valid_shape}, test={test_df.shape}, target={target_column}"

    def _infer_model_from_plan(self, task_type: str):
        """从 model_plan 中推断用户期望的模型类型及参数。"""
        plan = getattr(self, "model_plan", None) or ""
        plan_lower = plan.lower()

        if "regression" in task_type:
            from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
            if "gradientboosting" in plan_lower or "梯度提升" in plan_lower:
                params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "random_state": 42}
                # 从方案中提取参数覆盖
                import re
                for p in ("n_estimators", "learning_rate", "max_depth", "subsample", "min_samples_leaf"):
                    m = re.search(rf"{p}\s*[=:]\s*([\d.]+)", plan)
                    if m:
                        params[p] = int(m.group(1)) if "." not in m.group(1) else float(m.group(1))
                return GradientBoostingRegressor(**params), "GradientBoostingRegressor"
            return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), "RandomForestRegressor"
        else:
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            if "gradientboosting" in plan_lower or "梯度提升" in plan_lower:
                params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "random_state": 42}
                import re
                for p in ("n_estimators", "learning_rate", "max_depth", "subsample", "min_samples_leaf"):
                    m = re.search(rf"{p}\s*[=:]\s*([\d.]+)", plan)
                    if m:
                        params[p] = int(m.group(1)) if "." not in m.group(1) else float(m.group(1))
                return GradientBoostingClassifier(**params), "GradientBoostingClassifier"
            return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), "RandomForestClassifier"

    def _deterministic_model_training_fallback(self, context: Dict[str, Any], output_path: str) -> Tuple[bool, str]:
        """确定性兜底：使用简单 sklearn 基线模型完成训练并落盘。"""
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        data_path = context.get("data_path") or self.data_path
        target_column = context.get("target_column") or self.target_column
        task_type = (context.get("task_type") or self.task_type or "classification").lower()
        artifact_paths = self._get_training_artifact_paths()
        train_split_path = context.get("train_split_path") or artifact_paths["train_split_path"]
        valid_split_path = context.get("valid_split_path") or artifact_paths["valid_split_path"]
        test_split_path = context.get("test_split_path") or artifact_paths["test_split_path"]

        if not data_path or not target_column:
            return False, "缺少训练输入路径或目标列"

        try:
            train_df = pd.read_csv(train_split_path) if os.path.exists(train_split_path) else pd.read_csv(data_path)
            valid_df = pd.read_csv(valid_split_path) if valid_split_path and os.path.exists(valid_split_path) else None
            test_df = pd.read_csv(test_split_path) if test_split_path and os.path.exists(test_split_path) else None
        except Exception as error:
            return False, f"读取训练数据失败: {error}"

        if target_column not in train_df.columns:
            return False, f"训练数据中不存在目标列: {target_column}"

        train_df = train_df.dropna(subset=[target_column]).copy()
        if valid_df is not None:
            valid_df = valid_df.dropna(subset=[target_column]).copy()
        if test_df is not None:
            test_df = test_df.dropna(subset=[target_column]).copy()

        if train_df.empty:
            return False, "去除目标列缺失后无可用样本"
        if test_df is None or test_df.empty:
            return False, "缺少可用的测试集资产"

        x_train_raw = train_df.drop(columns=[target_column])
        # 排除 Id/标识列，避免数据泄漏
        id_cols = [c for c in x_train_raw.columns if c.lower() in ("id", "index", "row_id", "row_number")]
        if id_cols:
            x_train_raw = x_train_raw.drop(columns=id_cols)
            print(f"[Fallback] 排除标识列: {id_cols}")
        evaluation_df = valid_df if valid_df is not None and not valid_df.empty else test_df
        evaluation_split_name = "valid" if valid_df is not None and not valid_df.empty else "test"
        x_valid_raw = evaluation_df.drop(columns=[target_column])
        if id_cols:
            x_valid_raw = x_valid_raw.drop(columns=[c for c in id_cols if c in x_valid_raw.columns])
        y_train = train_df[target_column]
        y_valid = evaluation_df[target_column]

        numeric_features = list(x_train_raw.select_dtypes(include=["int64", "float64", "int32", "float32", "bool"]).columns)
        categorical_features = [column for column in x_train_raw.columns if column not in numeric_features]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                    ]),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )

        estimator, estimator_name = self._infer_model_from_plan(task_type)
        print(f"[Fallback] 使用模型: {estimator_name}")

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        model.fit(x_train_raw, y_train)
        predictions = model.predict(x_valid_raw)

        if "regression" in task_type:
            metrics = {
                "primary_metric": "rmse",
                "r2": float(r2_score(y_valid, predictions)),
                "mae": float(mean_absolute_error(y_valid, predictions)),
                "rmse": float(mean_squared_error(y_valid, predictions) ** 0.5)
            }
        else:
            metrics = {
                "primary_metric": "f1_weighted",
                "accuracy": float(accuracy_score(y_valid, predictions)),
                "f1_weighted": float(f1_score(y_valid, predictions, average="weighted"))
            }

        selected_feature_names = list(x_train_raw.columns)
        packaged_artifact = {
            "model": model,
            "selected_feature_names": selected_feature_names,
            "target_transform": None,
            "preprocessor": None,
            "target_column": target_column,
            "task_type": self.task_type,
            "artifact_format": "model_package_v1"
        }
        self._save_model_artifact(packaged_artifact, artifact_paths["model_path"])

        summary_payload = {
            "metrics": metrics,
            "selected_feature_names": selected_feature_names,
            "target_column": target_column,
            "task_type": self.task_type,
            "target_transform": None,
            "data_path": data_path,
            "model_path": artifact_paths["model_path"],
            "train_split_path": train_split_path,
            "valid_split_path": valid_split_path if evaluation_split_name == "valid" else None,
            "test_split_path": test_split_path,
            "evaluation_split_name": evaluation_split_name,
            "timestamp": datetime.now().isoformat()
        }

        with open(artifact_paths["training_summary_path"], "w", encoding="utf-8") as file:
            json.dump(summary_payload, file, ensure_ascii=False, indent=2)

        # 保存函数式代码文件供 pipeline 使用
        fallback_code = f'''import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_model(train_path, valid_path, test_path, model_dir, target_column, summary_path=None):
    """训练模型（确定性兜底：RandomForest 基线）。"""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "trained_model.pkl")
    if summary_path is None:
        summary_path = os.path.join(model_dir, "training_summary.json")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path) if valid_path and os.path.exists(valid_path) else None
    test_df = pd.read_csv(test_path) if test_path and os.path.exists(test_path) else None

    train_df = train_df.dropna(subset=[target_column])
    if valid_df is not None:
        valid_df = valid_df.dropna(subset=[target_column])
    if test_df is not None:
        test_df = test_df.dropna(subset=[target_column])

    x_train = train_df.drop(columns=[target_column])
    id_cols = [c for c in x_train.columns if c.lower() in ("id", "index", "row_id")]
    if id_cols:
        x_train = x_train.drop(columns=id_cols)

    eval_df = valid_df if valid_df is not None and not valid_df.empty else test_df
    x_eval = eval_df.drop(columns=[target_column])
    if id_cols:
        x_eval = x_eval.drop(columns=[c for c in id_cols if c in x_eval.columns])
    y_train = train_df[target_column]
    y_eval = eval_df[target_column]

    numeric_features = list(x_train.select_dtypes(include=["number", "bool"]).columns)
    categorical_features = [c for c in x_train.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ],
        remainder="drop",
    )

    task_type = "{task_type}"
    if "regression" in task_type:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")

    model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)

    if "regression" in task_type:
        metrics = {{
            "r2": float(r2_score(y_eval, predictions)),
            "mae": float(mean_absolute_error(y_eval, predictions)),
            "rmse": float(mean_squared_error(y_eval, predictions) ** 0.5),
        }}
    else:
        metrics = {{
            "accuracy": float(accuracy_score(y_eval, predictions)),
            "f1_weighted": float(f1_score(y_eval, predictions, average="weighted")),
        }}

    selected_feature_names = list(x_train.columns)
    artifact = {{
        "model": model,
        "selected_feature_names": selected_feature_names,
        "target_transform": None,
        "preprocessor": None,
    }}
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    summary = {{
        "metrics": metrics,
        "selected_feature_names": selected_feature_names,
        "target_column": target_column,
        "task_type": task_type,
        "target_transform": None,
        "model_path": model_path,
    }}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"模型训练完成，指标: {{metrics}}")

    return {{
        "metrics": metrics,
        "selected_feature_names": selected_feature_names,
        "model_path": model_path,
        "train_split_path": train_path,
        "valid_split_path": valid_path,
        "test_split_path": test_path,
        "training_summary_path": summary_path,
    }}


if __name__ == "__main__":
    result = train_model(
        train_path="{train_split_path}",
        valid_path="{valid_split_path}",
        test_path="{test_split_path}",
        model_dir=os.path.dirname("{artifact_paths["model_path"]}"),
        target_column="{target_column}",
        summary_path="{artifact_paths["training_summary_path"]}",
    )
    metrics = result["metrics"]
    selected_feature_names = result["selected_feature_names"]
    model_path = result["model_path"]
    train_split_path = result["train_split_path"]
    valid_split_path = result["valid_split_path"]
    test_split_path = result["test_split_path"]
    training_summary_path = result["training_summary_path"]
'''
        self.asset_manager.save_code(
            code=fallback_code,
            filename="model_training.py",
            metadata={"stage": "model_training", "fallback": True}
        )

        return True, f"确定性基线训练完成，使用 {evaluation_split_name} 集做指标选择，模型已保存到 {output_path}"

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.config_loader.get_prompt("model_training", "system_prompt")

    def analyze_data_for_modeling(
        self,
        data_path: str,
        target_column: str,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        分析数据以进行建模

        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型

        Returns:
            数据分析结果
        """
        self.data_path = data_path
        self.target_column = target_column
        self.task_type = task_type

        prompt_template = self.config_loader.get_prompt("model_training", "analysis_prompt")
        user_input = prompt_template.format(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,
        )

        result = self.run(user_input, stage="model_data_analysis")

        # 保存数据信息
        self.data_info = result.get("data_info", {})

        return result

    def generate_model_plan(
        self,
        data_path: str = None,
        target_column: str = None,
        task_type: str = "classification",
        feature_metrics_report: str = None,
        exploration_report: str = None,
        features_data_path: str = None,
        task_description: str = "",
        train_split_path: str = None,
        valid_split_path: str = None,
        test_split_path: str = None,
    ) -> str:
        """
        生成建模方案

        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型
            feature_metrics_report: 特征分析报告（可选）
            exploration_report: 探索性分析报告（可选）
            features_data_path: 特征工程后的数据路径（可选，如果提供则使用此路径）
            task_description: 用户的建模背景和要求

        Returns:
            建模方案（Markdown 格式）
        """
        # 优先使用特征工程后的数据路径
        path = features_data_path or data_path or self.data_path
        target = target_column or self.target_column
        task = task_type or self.task_type

        if not path or not target:
            raise ValueError("请提供数据文件路径和目标列名")

        self.data_path = path
        self.target_column = target
        self.task_type = task

        artifact_paths = self._get_training_artifact_paths()
        self.train_split_path = train_split_path or getattr(self, "train_split_path", None) or artifact_paths["train_split_path"]
        self.valid_split_path = valid_split_path or getattr(self, "valid_split_path", None) or artifact_paths["valid_split_path"]
        self.test_split_path = test_split_path or getattr(self, "test_split_path", None) or artifact_paths["test_split_path"]

        task_context = ""
        
        # 添加用户的建模背景
        if task_description:
            task_context = f"""
## 用户建模背景和要求

{task_description}

**重要：请在建模方案中充分考虑用户的建模背景和要求。**

"""
        self.task_context = task_context

        # 从 workflow_config 读取摘要限额
        summarization_config = self.config_loader.load_workflow_config().get("summarization", {})
        exploration_max = summarization_config.get("exploration_report_max_chars", 4000)
        feature_max = summarization_config.get("feature_report_max_chars", 3500)

        exploration_report_context = ""
        if exploration_report:
            summarized = self._summarize_report(exploration_report, max_chars=exploration_max, priority_keywords=[
                "目标变量", "特征相关性", "相关性分析", "特征重要性", "特征工程建议", "下一步建议",
            ])
            exploration_report_context = f"""
    ## 探索性分析报告（来自数据探索阶段）

    {summarized}

    """

        feature_report_context = ""
        if feature_metrics_report:
            summarized = self._summarize_report(feature_metrics_report, max_chars=feature_max, priority_keywords=[
                "IV", "Information Value", "特征重要性", "相关性", "建议", "结论",
            ])
            feature_report_context = f"""
## 特征分析报告（来自特征评估阶段）

{summarized}

"""

        # 加载并分析实际数据
        import pandas as pd

        try:
            df = pd.read_csv(path)

            # 收集数据基本信息
            self.data_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "numeric_columns": list(df.select_dtypes(include=['int64', 'float64']).columns),
                "categorical_columns": list(df.select_dtypes(include=['object', 'category', 'bool']).columns),
                "target_dtype": str(df[target].dtype) if target in df.columns else "unknown",
                "target_unique": df[target].nunique() if target in df.columns else 0,
                "has_missing": df.isnull().any().any()
            }

            feature_columns = [column for column in df.columns if column != target]
            missing_counts = df.isnull().sum()
            columns_with_missing = missing_counts[missing_counts > 0]
            constant_columns = [
                column for column in feature_columns
                if df[column].nunique(dropna=False) <= 1
            ]
            high_missing_columns = columns_with_missing.sort_values(ascending=False).head(10)
            target_missing_count = int(df[target].isnull().sum()) if target in df.columns else -1

            missing_columns_text = "无" if high_missing_columns.empty else "\n".join(
                [f"- {column}: {int(count)}" for column, count in high_missing_columns.items()]
            )
            constant_columns_text = "无" if not constant_columns else ", ".join(constant_columns[:20])

            # 构建当前训练数据上下文
            current_training_data_context = f"""
## 当前数据基本信息

- **数据路径**: {path}
- **数据来源说明**: 当前训练输入文件为特征工程阶段输出的数据结果
- **数据形状**: {self.data_info['shape'][0]} 行 × {self.data_info['shape'][1]} 列
- **目标列**: {target}
- **目标列类型**: {self.data_info['target_dtype']}
- **目标列唯一值数**: {self.data_info['target_unique']}
- **目标列缺失值数**: {target_missing_count}
- **数值列数量**: {len(self.data_info['numeric_columns'])}
- **分类列数量**: {len(self.data_info['categorical_columns'])}
- **是否有缺失值**: {self.data_info['has_missing']}
- **候选入模特征数量**: {len(feature_columns)}
- **检测到的常量候选特征数量**: {len(constant_columns)}

## 数值列

{', '.join(self.data_info['numeric_columns'][:20])}

## 分类列

{', '.join(self.data_info['categorical_columns'][:20])}

## 候选入模字段

{', '.join(feature_columns[:40])}

## 缺失值概览（Top 10）

{missing_columns_text}

## 常量/低信息候选特征

{constant_columns_text}

## 已核实事实

- 当前唯一训练输入数据路径: {path}
- 训练阶段必须以特征工程后的数据文件为输入，不得回退到原始数据或清洗前数据
- 预测目标变量: {target}
- 应综合当前训练数据事实、探索性分析报告、特征分析报告决定最终建模策略
- 默认应保留除目标列外的全部特征工程产物，仅排除明确常量列或确认泄露特征
- 上游已提供训练集文件: {self.train_split_path}
- 上游已提供验证集文件: {self.valid_split_path if self.valid_split_path else '无，当前数据量不足时可退化为 train/test'}
- 上游已提供测试集文件: {self.test_split_path}
- 需要保存训练好的模型文件: {artifact_paths['model_path']}
- 需要保存训练摘要文件: {artifact_paths['training_summary_path']}
- valid 只用于选方案与调参，test 只用于最终评估，不参与任何拟合和调参

重要：请基于上述实际数据和前序阶段的分析结果生成建模方案，且所有字段、样本规模、目标变量判断都必须与当前读取到的数据一致。
"""

        except Exception as e:
            current_training_data_context = f"无法加载数据文件: {path}\n错误: {str(e)}"

        # 从配置加载 Prompt
        prompt_template = self.config_loader.get_prompt("model_training", "plan_generation")

        user_input = prompt_template.format(
            data_path=path,
            target_column=target,
            task_type=task,
            task_context=task_context,
            exploration_report_context=exploration_report_context,
            feature_report_context=feature_report_context,
            current_training_data_context=current_training_data_context,
        )

        # 调用 LLM 生成方案
        result = self.run(user_input, stage="model_plan")

        self.model_plan = result.get("answer", "")

        return self.model_plan

    def revise_plan(self, current_plan: str, modifications: str, **kwargs) -> str:
        """基于用户反馈修订建模方案"""
        prompt_template = self.config_loader.get_prompt("model_training", "plan_revision")
        user_input = prompt_template.format(
            current_plan=current_plan,
            user_modifications=modifications,
            target_column=getattr(self, "target_column", ""),
            task_type=getattr(self, "task_type", ""),
        )
        result = self.run(user_input, stage="model_training_plan_revision")
        self.model_plan = result.get("answer", "")
        return self.model_plan

    def get_modifiable_aspects(self) -> list:
        return ["模型选择", "超参搜索策略", "评估指标", "交叉验证方式", "目标变换"]

    def request_user_confirmation(self) -> None:
        """
        请求用户确认建模方案

        Raises:
            ConfirmationRequired: 需要用户确认时抛出
        """
        if not self.model_plan:
            raise ValueError("请先生成建模方案")

        # 参考的 skills
        skills_referenced = [
            {
                "name": "afrexai-ml-engineering-1.0.0",
                "files": ["SKILL.md (Phase 3: Model Selection)"]
            }
        ]

        # 抛出确认异常
        raise ConfirmationRequired(
            stage="model_training",
            proposal=self.model_plan,
            skills_referenced=skills_referenced
        )

    def generate_model_code(self, modifications: str = None) -> str:
        """
        生成建模代码（使用结构化输出和迭代验证）

        Args:
            modifications: 用户修改内容

        Returns:
            建模代码
        """
        if not self.model_plan:
            raise ValueError("请先生成建模方案")

        from ..utils.codeact_agent import CodeActAgent

        artifact_paths = self._get_training_artifact_paths()
        modifications_text = f"\n用户修改要求:\n{modifications}\n" if modifications else ""

        # 构建数据信息摘要
        data_info_text = ""
        if self.data_info:
            data_info_text = f"""
## 数据信息

- 数据形状: {self.data_info['shape'][0]} 行 × {self.data_info['shape'][1]} 列
- 目标列: {self.target_column}
- 任务类型: {self.task_type}
- 数值列: {', '.join(self.data_info['numeric_columns'][:15])}
- 分类列: {', '.join(self.data_info['categorical_columns'][:15])}

重要：请基于上述实际数据列名生成代码，不要使用示例数据中的列名。
"""

        prompt_template = self.config_loader.get_prompt("model_training", "code_generation_full")
        prompt = prompt_template.format(
            data_path=self.data_path,
            target_column=self.target_column,
            task_type=self.task_type,
            data_info_text=data_info_text,
            plan=self.model_plan,
            modifications=modifications_text,
            train_split_path=self.train_split_path or artifact_paths["train_split_path"],
            valid_split_path=self.valid_split_path or artifact_paths["valid_split_path"],
            test_split_path=self.test_split_path or artifact_paths["test_split_path"],
            model_path=artifact_paths["model_path"],
            training_summary_path=artifact_paths["training_summary_path"],
            task_context=getattr(self, 'task_context', ''),
        )

        # 使用 CodeActAgent 生成并执行代码
        codeact = CodeActAgent(llm=self.llm, max_iterations=5, timeout=300, session_id=self.session_id)
        if self._stream_callback:
            codeact.set_stream_callback(self._stream_callback)

        context = {
            "data_path": self.data_path,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "model_path": artifact_paths["model_path"],
            "train_split_path": self.train_split_path or artifact_paths["train_split_path"],
            "valid_split_path": self.valid_split_path or artifact_paths["valid_split_path"],
            "test_split_path": self.test_split_path or artifact_paths["test_split_path"],
            "training_summary_path": artifact_paths["training_summary_path"]
        }

        result = codeact.generate_and_execute(
            task_prompt=prompt,
            context=context,
            required_outputs=[
                "metrics",
                "selected_feature_names",
                "model_path",
                "train_split_path",
                "valid_split_path",
                "test_split_path",
                "training_summary_path"
            ],
            required_filepath=artifact_paths["model_path"],
            output_validator=self._validate_training_outputs,
            deterministic_fallback=self._deterministic_model_training_fallback,
            stage="model_training_code_generation",
        )

        if not result.success:
            raise ValueError(f"代码生成失败: {result.error}")

        self._normalize_training_artifacts()

        self.model_code = result.code
        self.model_result = self._collect_training_result(execution_output=result.output)

        # 保存代码到资产
        if self.model_code:
            self.asset_manager.save_code(
                code=self.model_code,
                filename="model_training.py",
                metadata={
                    "stage": "model_training",
                    "data_path": self.data_path,
                    "target_column": self.target_column,
                    "task_type": self.task_type,
                    "execution_success": result.success,
                    "execution_error": result.error,
                    "iterations": result.iterations,
                    "timestamp": datetime.now().isoformat()
                }
            )

        return self.model_code

    def execute_model_training(self, code: str = None) -> Dict[str, Any]:
        """
        执行模型训练代码

        Args:
            code: 建模代码，为 None 时使用已生成的代码

        Returns:
            执行结果
        """
        from ..utils.codeact_agent import CodeActAgent

        model_code = code or self.model_code

        if not model_code:
            raise ValueError("请先生成建模代码")

        if self.model_result and model_code == self.model_code:
            cached_result = self.model_result.copy()
            self.asset_manager.save_data(
                data=json.dumps(cached_result, ensure_ascii=False, indent=2),
                filename="model_training_result.json",
                asset_type="models",
                metadata=cached_result
            )
            return cached_result

        artifact_paths = self._get_training_artifact_paths()
        context = {
            "data_path": self.data_path,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "model_path": artifact_paths["model_path"],
            "train_split_path": self.train_split_path or artifact_paths["train_split_path"],
            "valid_split_path": self.valid_split_path or artifact_paths["valid_split_path"],
            "test_split_path": self.test_split_path or artifact_paths["test_split_path"],
            "training_summary_path": artifact_paths["training_summary_path"]
        }

        codeact = CodeActAgent(llm=self.llm, max_iterations=1, timeout=300, session_id=self.session_id)
        if self._stream_callback:
            codeact.set_stream_callback(self._stream_callback)
        exec_result = codeact._execute_code(model_code, context)
        if exec_result.get("success"):
            self._normalize_training_artifacts()
        result_info = self._collect_training_result(
            execution_output=exec_result.get("output", ""),
            execution_error=exec_result.get("error") if not exec_result.get("success") else None,
        )
        self.model_result = result_info

        # 保存结果信息到资产
        self.asset_manager.save_data(
            data=json.dumps(result_info, ensure_ascii=False, indent=2),
            filename="model_training_result.json",
            asset_type="models",
            metadata=result_info
        )

        return result_info

    # ------------------------------------------------------------------
    # 评估方法（从 ModelEvaluationAgent 融合）
    # ------------------------------------------------------------------

    def generate_evaluation_plan(self) -> str:
        """基于训练结果生成评估方案。"""
        training_summary = {}
        training_summary_json = self.asset_manager.read_asset("models", "training_summary.json")
        if training_summary_json:
            try:
                training_summary = json.loads(training_summary_json)
            except Exception:
                pass

        model_result = self.model_result or {}
        model_path = model_result.get("model_path", "未知")
        test_split_path = model_result.get("test_split_path", "未知")
        metrics = model_result.get("metrics", {})
        selected_feature_names = model_result.get("selected_feature_names", [])

        task_context = getattr(self, "task_context", "")

        current_evaluation_context = f"""
## 当前评估事实

- **模型文件**: {model_path}
- **测试集文件**: {test_split_path}
- **目标列**: {self.target_column}
- **任务类型**: {self.task_type}
- **训练阶段记录的最佳模型**: {training_summary.get('best_model_name', '未知')}
- **训练阶段记录的目标变换**: {training_summary.get('target_transform', '未记录')}
- **训练阶段回收指标**: {json.dumps(metrics, ensure_ascii=False)}
- **训练阶段回收入模特征数**: {len(selected_feature_names)}

重要：评估阶段必须基于已保存的模型文件和测试集文件，验证训练摘要中的指标是否可复现，并补充形成标准化评估结论。
"""
        prompt_template = self.config_loader.get_prompt("model_training", "evaluation_plan_generation")
        user_input = prompt_template.format(
            task_context=task_context,
            current_evaluation_context=current_evaluation_context,
        )

        result = self.run(user_input, stage="model_evaluation_plan")
        self.evaluation_plan = result.get("answer", "")
        return self.evaluation_plan

    def execute_evaluation(self) -> Dict[str, Any]:
        """执行模型评估，生成评估代码并运行。"""
        from ..evaluation import ModelEvaluator

        model_result = self.model_result or {}
        model_path = model_result.get("model_path")
        test_split_path = model_result.get("test_split_path")

        if not model_path:
            return {"success": False, "error": "缺少 model_path，无法评估"}
        if not test_split_path:
            return {"success": False, "error": "缺少 test_split_path，无法评估"}

        evaluator = ModelEvaluator(session_id=self.session_id)
        result = evaluator.evaluate_model(
            model_path=model_path,
            data_path=test_split_path,
            target_column=self.target_column,
            task_type=self.task_type,
        )

        result["stage"] = "model_evaluation"
        result["test_split_path"] = test_split_path
        self.evaluation_result = result
        return result

    def full_model_training_workflow(
        self,
        data_path: str,
        target_column: str,
        task_type: str = "classification",
        skip_confirmation: bool = False
    ) -> Dict[str, Any]:
        """
        执行完整的模型训练流程

        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型
            skip_confirmation: 是否跳过用户确认

        Returns:
            模型训练结果
        """
        self.data_path = data_path
        self.target_column = target_column
        self.task_type = task_type

        exploration_report = self.asset_manager.read_asset("exploration", "data_exploration_result.md")
        feature_metrics_report = self.asset_manager.read_asset("features", "feature_metrics_report.md")

        feature_result_json = self.asset_manager.read_asset("features", "feature_engineering_result.json")
        features_data_path = None
        if feature_result_json:
            try:
                feature_result = json.loads(feature_result_json)
                features_data_path = feature_result.get("features_data_path")
            except Exception:
                features_data_path = None

        # 1. 生成建模方案
        plan = self.generate_model_plan(
            data_path,
            target_column,
            task_type,
            feature_metrics_report=feature_metrics_report,
            exploration_report=exploration_report,
            features_data_path=features_data_path,
        )

        # 2. 请求用户确认（如果不跳过）
        if not skip_confirmation:
            self.request_user_confirmation()

        # 3. 生成建模代码
        code = self.generate_model_code()

        # 4. 执行模型训练
        result = self.execute_model_training(code)

        return {
            "success": result.get("success", False),
            "plan": plan,
            "code": code,
            "result": result
        }
