"""
模型工具模块

提供模型训练和评估的工具
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from .base_tool import BaseTool, ToolResult


class ModelTrainerInput(BaseModel):
    file_path: str = Field(..., description="数据文件路径")
    target_column: str = Field(..., description="目标列名")
    task_type: Literal["classification", "regression"] = Field(
        ..., description="任务类型 (classification/regression)"
    )
    test_size: float = Field(0.2, description="测试集比例，默认 0.2", ge=0.0, le=1.0)


class ModelTrainerTool(BaseTool):
    """
    模型训练工具

    自动训练机器学习模型
    """

    name = "train_model"
    description = "训练机器学习模型，支持分类和回归任务"
    input_model = ModelTrainerInput
    
    def execute(
        self,
        file_path: str,
        target_column: str,
        task_type: str,
        test_size: float = 0.2
    ) -> ToolResult:
        """执行模型训练"""
        try:
            # 加载数据
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                return ToolResult.error("目前只支持 CSV 格式")
            
            if target_column not in df.columns:
                return ToolResult.error(f"目标列 '{target_column}' 不存在")
            
            # 数据预处理
            df = df.dropna()
            
            # 分离特征和目标
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # 处理类别特征
            categorical_cols = X.select_dtypes(include=["object"]).columns
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # 选择模型
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 评估
            if task_type == "classification":
                metrics = {
                    "accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4)
                }
            else:
                metrics = {
                    "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                    "r2_score": round(r2_score(y_test, y_pred), 4)
                }
            
            # 特征重要性
            feature_importance = dict(zip(
                X.columns,
                [round(float(x), 4) for x in model.feature_importances_]
            ))
            
            # 排序特征重要性
            feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])  # 只返回前10个
            
            return ToolResult.success(
                data={
                    "task_type": task_type,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "metrics": metrics,
                    "top_features": feature_importance
                }
            )
            
        except Exception as e:
            return ToolResult.error(f"模型训练失败: {str(e)}")
