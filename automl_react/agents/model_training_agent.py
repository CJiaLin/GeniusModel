"""
模型训练 Agent 模块

实现建模方案生成、用户确认、代码生成与执行
"""

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import joblib

from ..core.react_agent import ReActAgent, ConfirmationRequired
from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
from ..skills_loader import get_skill_loader
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
        self.skill_loader = get_skill_loader()
        self.config_loader = get_config_loader()

    def _register_default_tools(self):
        """注册默认工具"""
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())

    def _get_training_artifact_paths(self) -> Dict[str, str]:
        """获取模型训练阶段的标准资产路径。"""
        session_dir = self.asset_manager.session_dir
        return {
            "model_path": str(session_dir / "models" / "trained_model.pkl"),
            "train_split_path": str(session_dir / "data" / "train_split.csv"),
            "test_split_path": str(session_dir / "data" / "test_split.csv"),
            "training_summary_path": str(session_dir / "models" / "training_summary.json")
        }

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
        if not raw_transform or not isinstance(raw_transform, str):
            return None

        lowered = raw_transform.lower()
        if "log1p" in lowered:
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

        result_info = {
            "success": all(artifact_status.values()),
            "model_path": artifact_paths["model_path"] if artifact_status["model_path"] else None,
            "train_split_path": artifact_paths["train_split_path"] if artifact_status["train_split_path"] else None,
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
        missing_files = [path for path in artifact_paths.values() if not os.path.exists(path)]
        if missing_files:
            return False, f"缺少训练产物: {missing_files}"

        try:
            train_df = pd.read_csv(artifact_paths["train_split_path"])
            test_df = pd.read_csv(artifact_paths["test_split_path"])
        except Exception as error:
            return False, f"训练/测试切分文件不可读: {error}"

        if train_df.empty or test_df.empty:
            return False, "训练集或测试集为空"

        target_column = context.get("target_column") or self.target_column
        if target_column and (target_column not in train_df.columns or target_column not in test_df.columns):
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

        return True, f"train={train_df.shape}, test={test_df.shape}, target={target_column}"

    def _deterministic_model_training_fallback(self, context: Dict[str, Any], output_path: str) -> Tuple[bool, str]:
        """确定性兜底：使用简单 sklearn 基线模型完成训练并落盘。"""
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        data_path = context.get("data_path") or self.data_path
        target_column = context.get("target_column") or self.target_column
        task_type = (context.get("task_type") or self.task_type or "classification").lower()
        artifact_paths = self._get_training_artifact_paths()

        if not data_path or not target_column:
            return False, "缺少训练输入路径或目标列"

        try:
            df = pd.read_csv(data_path)
        except Exception as error:
            return False, f"读取训练数据失败: {error}"

        if target_column not in df.columns:
            return False, f"训练数据中不存在目标列: {target_column}"

        df = df.dropna(subset=[target_column]).copy()
        if df.empty:
            return False, "去除目标列缺失后无可用样本"

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        x_train_raw = train_df.drop(columns=[target_column])
        x_test_raw = test_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        y_test = test_df[target_column]

        x_train = pd.get_dummies(x_train_raw, dummy_na=False)
        x_test = pd.get_dummies(x_test_raw, dummy_na=False)
        x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)

        if "regression" in task_type:
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            metrics = {
                "r2": float(r2_score(y_test, predictions)),
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(mean_squared_error(y_test, predictions) ** 0.5)
            }
        else:
            model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "f1_weighted": float(f1_score(y_test, predictions, average="weighted"))
            }

        selected_feature_names = list(x_train.columns)

        train_df.to_csv(artifact_paths["train_split_path"], index=False)
        test_df.to_csv(artifact_paths["test_split_path"], index=False)
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
            "train_split_path": artifact_paths["train_split_path"],
            "test_split_path": artifact_paths["test_split_path"],
            "timestamp": datetime.now().isoformat()
        }

        with open(artifact_paths["training_summary_path"], "w", encoding="utf-8") as file:
            json.dump(summary_payload, file, ensure_ascii=False, indent=2)

        return True, f"确定性基线训练完成，模型已保存到 {output_path}"

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.config_loader.get_prompt("model_training", "system_prompt")

    def _get_phase3_skill_content(self) -> str:
        """获取模型训练阶段使用的 skill 参考内容。"""
        skill_content = self.skill_loader.get_skill_content("afrexai-ml-engineering-1.0.0")
        if not skill_content:
            return ""

        import re

        phase3_match = re.search(
            r'##?\s*Phase\s*3[:：].*?\n(.*?)(?=##?\s*Phase\s*4|\Z)',
            skill_content,
            re.DOTALL | re.IGNORECASE
        )
        if phase3_match:
            return "## Phase 3: Experiment Management\n\n" + phase3_match.group(1).strip()
        return skill_content

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
        task_description: str = ""
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

        task_context = ""
        
        # 添加用户的建模背景
        if task_description:
            task_context = f"""
## 用户建模背景和要求

{task_description}

**重要：请在建模方案中充分考虑用户的建模背景和要求。**

"""
        self.task_context = task_context

        exploration_report_context = ""
        if exploration_report:
            exploration_report_context = f"""
    ## 探索性分析报告（来自数据探索阶段）

    {exploration_report[:2200]}

    """

        feature_report_context = ""
        if feature_metrics_report:
            feature_report_context = f"""
## 特征分析报告（来自特征评估阶段）

{feature_metrics_report[:2500]}

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
- 需要保留训练集文件: {artifact_paths['train_split_path']}
- 需要保留测试集文件: {artifact_paths['test_split_path']}
- 需要保存训练好的模型文件: {artifact_paths['model_path']}
- 需要保存训练摘要文件: {artifact_paths['training_summary_path']}

重要：请基于上述实际数据和前序阶段的分析结果生成建模方案，且所有字段、样本规模、目标变量判断都必须与当前读取到的数据一致。
"""

        except Exception as e:
            current_training_data_context = f"无法加载数据文件: {path}\n错误: {str(e)}"

        # 加载 afrexai-ml-engineering skill 的 Phase 3
        phase3_content = self._get_phase3_skill_content()

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
            skill_content=phase3_content
        )

        # 调用 LLM 生成方案
        result = self.run(user_input, stage="model_plan")

        self.model_plan = result.get("answer", "")

        return self.model_plan

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
            skill_content=self._get_phase3_skill_content(),
            modifications=modifications_text,
            train_split_path=artifact_paths["train_split_path"],
            test_split_path=artifact_paths["test_split_path"],
            model_path=artifact_paths["model_path"],
            training_summary_path=artifact_paths["training_summary_path"],
            task_context=getattr(self, 'task_context', ''),
        )

        # 使用 CodeActAgent 生成并执行代码
        codeact = CodeActAgent(llm=self.llm, max_iterations=5, timeout=300, session_id=self.session_id)

        context = {
            "data_path": self.data_path,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "model_path": artifact_paths["model_path"],
            "train_split_path": artifact_paths["train_split_path"],
            "test_split_path": artifact_paths["test_split_path"],
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
            "train_split_path": artifact_paths["train_split_path"],
            "test_split_path": artifact_paths["test_split_path"],
            "training_summary_path": artifact_paths["training_summary_path"]
        }

        codeact = CodeActAgent(llm=self.llm, max_iterations=1, timeout=300, session_id=self.session_id)
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
