"""
sklearn Pipeline 包装器模块

将各阶段的文件IO函数（clean_data, engineer_features）包装为 sklearn Transformer，
并将训练好的模型封装为最终 estimator，组装成可序列化的 sklearn Pipeline。
"""

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameStageTransformer(BaseEstimator, TransformerMixin):
    """将文件路径IO的阶段函数包装为 sklearn Transformer。

    fit(X): 运行阶段函数处理训练数据，存储训练数据引用供 transform 时使用。
    transform(X): 将输入写入临时文件，调用阶段函数，读取输出返回 DataFrame。
    """

    def __init__(self, stage_source: str, stage_fn_name: str, stage_name: str = "stage"):
        self.stage_source = stage_source
        self.stage_fn_name = stage_fn_name
        self.stage_name = stage_name

    def _get_stage_fn(self):
        """从源码字符串编译并返回阶段函数。"""
        namespace = {}
        exec(self.stage_source, namespace)
        fn = namespace.get(self.stage_fn_name)
        if fn is None:
            raise ValueError(
                f"函数 '{self.stage_fn_name}' 未在 {self.stage_name} 源码中找到"
            )
        return fn

    def fit(self, X, y=None):
        """运行阶段函数处理训练数据并存储训练引用。"""
        # 存储训练数据作为 transform 时的 train_path 参考
        self.train_reference_ = X.copy()
        # 运行一次确认函数可用，并存储训练输出作为下游 transformer 的 train_ref
        self.train_output_ = self._run_stage(X, X)
        return self

    def transform(self, X):
        """使用存储的训练引用对输入数据执行阶段函数。"""
        return self._run_stage(X, self.train_reference_)

    def _run_stage(self, input_df: pd.DataFrame, train_ref_df: pd.DataFrame) -> pd.DataFrame:
        """通过临时文件调用阶段函数。"""
        stage_fn = self._get_stage_fn()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            output_path = os.path.join(tmpdir, "output.csv")
            train_path = os.path.join(tmpdir, "train_ref.csv")

            input_df.to_csv(input_path, index=False)
            train_ref_df.to_csv(train_path, index=False)

            stage_fn(input_path, output_path, train_path)

            if not os.path.exists(output_path):
                raise RuntimeError(
                    f"{self.stage_name} 阶段函数未生成输出文件: {output_path}"
                )
            return pd.read_csv(output_path)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class TargetColumnSplitter(BaseEstimator, TransformerMixin):
    """在模型步骤之前分离目标列和ID列。"""

    def __init__(self, target_column: str, id_columns=None):
        self.target_column = target_column
        self.id_columns = id_columns or ["Id", "ID", "id", "index"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        drop_cols = [self.target_column] + self.id_columns
        cols_to_drop = [c for c in drop_cols if c in X.columns]
        return X.drop(columns=cols_to_drop) if cols_to_drop else X.copy()


class ModelStepWrapper(BaseEstimator):
    """包装训练好的模型 artifact 作为 sklearn Pipeline 的最终 estimator。

    内部复现 model_training 的全部预处理步骤（特征对齐、编码、缩放），
    确保 predict 时和训练时走完全一致的代码路径。
    """

    def __init__(self, model_artifact=None, target_column=None, task_type="classification"):
        self.model_artifact = model_artifact
        self.target_column = target_column
        self.task_type = task_type

    def fit(self, X, y=None):
        if self.model_artifact and isinstance(self.model_artifact, dict):
            self.feature_names_ = self.model_artifact.get("selected_feature_names", [])
        return self

    def _prepare_input(self, X):
        """对齐特征、处理字符串列、填充缺失值、应用 preprocessor。"""
        artifact = self.model_artifact
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError("model_artifact 格式不正确，缺少 'model' 键")

        feature_names = artifact.get("selected_feature_names", [])
        preprocessor = artifact.get("preprocessor")
        categorical_mappings = artifact.get("categorical_mappings")

        # 特征对齐
        if feature_names:
            X_aligned = X.reindex(columns=feature_names, fill_value=0)
        else:
            X_aligned = X.copy()

        # 应用 preprocessor（如 ColumnTransformer with OneHotEncoder）
        model = artifact["model"]
        pipeline_has_internal_preprocessor = (
            hasattr(model, "named_steps")
            and isinstance(getattr(model, "named_steps", None), dict)
            and "preprocessor" in model.named_steps
        )

        if preprocessor is not None and not pipeline_has_internal_preprocessor:
            # preprocessor（ColumnTransformer）内部已包含 OneHotEncoder 等编码器，
            # 会自行处理字符串分类列，不需要提前做 cat.codes 转换。
            # 只对数值列填充缺失值（分类列由 preprocessor 内的 SimpleImputer 处理）
            num_cols = X_aligned.select_dtypes(include=["number"]).columns.tolist()
            X_aligned[num_cols] = X_aligned[num_cols].fillna(0)
            model_input = preprocessor.transform(X_aligned)
        else:
            # 无外部 preprocessor 时，手动处理分类列
            if categorical_mappings and isinstance(categorical_mappings, dict):
                for col, mapping in categorical_mappings.items():
                    if col in X_aligned.columns:
                        X_aligned[col] = X_aligned[col].map(mapping).fillna(-1).astype(int)
            else:
                cat_cols = X_aligned.select_dtypes(include=["object", "category"]).columns.tolist()
                if cat_cols:
                    for col in cat_cols:
                        X_aligned[col] = X_aligned[col].astype("category").cat.codes
            X_aligned = X_aligned.fillna(0)
            model_input = X_aligned

        return model_input

    def predict(self, X):
        model_input = self._prepare_input(X)
        model = self.model_artifact["model"]
        predictions = model.predict(model_input)

        # 逆变换目标变量
        target_transform = self.model_artifact.get("target_transform")
        if isinstance(target_transform, str) and "log1p" in target_transform.lower():
            predictions = np.expm1(predictions)

        return predictions

    def predict_proba(self, X):
        model_input = self._prepare_input(X)
        model = self.model_artifact["model"]
        if hasattr(model, "predict_proba"):
            return model.predict_proba(model_input)
        raise AttributeError("模型不支持 predict_proba")
