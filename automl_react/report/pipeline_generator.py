"""
全流程脚本生成器模块

生成 pipeline.py 编排脚本，保存到 code/ 目录（与各阶段代码同目录）。
"""

import json
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline

from ..assets import get_asset_manager
from .sklearn_wrappers import DataFrameStageTransformer, TargetColumnSplitter, ModelStepWrapper


class PipelineGenerator:
    """
    全流程脚本生成器

    生成 pipeline.py 编排脚本到 code/ 目录，与各阶段代码文件同目录：
      - cleaning.py   — 数据清洗函数
      - feature_engineering.py — 特征工程函数
      - model_training.py — 模型训练函数
      - model_evaluation.py — 模型评估函数
      - pipeline.py   — 编排脚本（入口）

    使用说明:
      python code/pipeline.py --mode train --data raw_data.csv
      python code/pipeline.py --mode predict --data new_data.csv --model output/model.pkl
    """

    def __init__(self, session_id: str = None):
        self.session_id = session_id or "default"
        self.asset_manager = get_asset_manager(session_id=self.session_id)

    def collect_stage_codes(self) -> Dict[str, str]:
        """收集各阶段代码"""
        code_files = {
            "data_cleaning": "cleaning.py",
            "feature_engineering": "feature_engineering.py",
            "model_training": "model_training.py",
            "model_evaluation": "model_evaluation.py",
        }
        stages = {}
        for stage, filename in code_files.items():
            code = self.asset_manager.read_asset("code", filename)
            if code:
                stages[stage] = code
        return stages

    def _load_split_config(self) -> Dict[str, Any]:
        """从 session 资产读取切分配置。"""
        raw = self.asset_manager.read_asset("analysis", "dataset_split_result.json")
        if raw:
            try:
                result = json.loads(raw)
                return {
                    "train_ratio": result.get("ratios", {}).get("train_ratio", 0.6),
                    "valid_ratio": result.get("ratios", {}).get("valid_ratio", 0.2),
                    "test_ratio": result.get("ratios", {}).get("test_ratio", 0.2),
                }
            except json.JSONDecodeError:
                pass
        return {"train_ratio": 0.6, "valid_ratio": 0.2, "test_ratio": 0.2}

    # ------------------------------------------------------------------
    # 主生成方法
    # ------------------------------------------------------------------

    def generate_pipeline_package(
        self,
        data_path: str,
        target_column: str,
        task_type: str = "regression",
    ) -> str:
        """
        生成 pipeline.py 编排脚本，保存到 code/ 目录。

        Returns:
            code 目录路径
        """
        stage_codes = self.collect_stage_codes()
        split_config = self._load_split_config()

        # 生成 pipeline.py 编排器，保存到 code/ 目录
        orchestrator = self._generate_orchestrator(
            target_column=target_column,
            task_type=task_type,
            split_config=split_config,
            has_evaluation="model_evaluation" in stage_codes,
        )
        self.asset_manager.save_code(
            code=orchestrator,
            filename="pipeline.py",
            metadata={
                "stage": "pipeline",
                "data_path": data_path,
                "target_column": target_column,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat(),
            }
        )

        code_dir = str(self.asset_manager.session_dir / "code")
        return code_dir

    def _generate_orchestrator(
        self,
        target_column: str,
        task_type: str,
        split_config: Dict[str, float],
        has_evaluation: bool = True,
    ) -> str:
        """生成 pipeline.py 编排脚本。"""
        train_ratio = split_config.get("train_ratio", 0.6)
        valid_ratio = split_config.get("valid_ratio", 0.2)
        test_ratio = split_config.get("test_ratio", 0.2)

        eval_import = ""
        eval_train_block = ""
        eval_predict_block = ""

        if has_evaluation:
            eval_import = """from model_evaluation import load_model_artifact, predict_from_artifact, evaluate_predictions"""
            eval_train_block = f'''
    # ── 模型评估 (测试集) ──
    print("\\n" + "=" * 60)
    print("[5/5] 模型评估 (测试集)")
    print("=" * 60)
    try:
        eval_df = pd.read_csv(features_test)
        if target_column in eval_df.columns:
            y_true = eval_df[target_column]
            eval_X = eval_df.drop(columns=[target_column, "Id"], errors="ignore")
            artifact = pickle.load(open(model_path, "rb"))
            preds, diag = predict_from_artifact(artifact, eval_X, None, None)
            print(f"  预测诊断: {{diag}}")
            eval_metrics = evaluate_predictions(
                y_true=y_true, y_pred=preds, task_type="{task_type}",
                model_artifact=artifact, X=eval_X,
            )
            for k, v in eval_metrics.items():
                print(f"  {{k}}: {{v:.4f}}" if isinstance(v, float) else f"  {{k}}: {{v}}")
        else:
            print(f"  测试集不包含目标列 {{target_column}}，跳过评估")
    except Exception as e:
        print(f"  评估失败: {{e}}")'''

            eval_predict_block = f'''
    # ── 预测 ──
    print("\\n" + "=" * 60)
    print("[5/5] 模型预测")
    print("=" * 60)
    model_artifact = load_model_artifact(model_path)
    df = pd.read_csv(features_path)
    drop_cols = [c for c in [target_column, "Id"] if c in df.columns]
    X = df.drop(columns=drop_cols) if drop_cols else df.copy()
    predictions, diagnostics = predict_from_artifact(model_artifact, X, None, None)
    print(f"  预测诊断: {{diagnostics}}")

    result_df = pd.DataFrame({{"prediction": predictions}})
    if "Id" in df.columns:
        result_df.insert(0, "Id", df["Id"].values)
    output_path = str(output_dir / "predictions.csv")
    result_df.to_csv(output_path, index=False)
    print(f"  预测完成: {{len(predictions)}} 条记录")
    print(f"  结果保存至: {{output_path}}")

    # 评估（如果有真实标签）
    if target_column in df.columns:
        y_true = df[target_column]
        eval_metrics = evaluate_predictions(
            y_true=y_true, y_pred=predictions, task_type="{task_type}",
            model_artifact=model_artifact, X=X,
        )
        print("\\n  评估指标:")
        for k, v in eval_metrics.items():
            print(f"    {{k}}: {{v:.4f}}" if isinstance(v, float) else f"    {{k}}: {{v}}")'''

        script = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoML 全流程 Pipeline 脚本

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
会话ID: {self.session_id}

使用说明:
  训练模式:  python pipeline.py --mode train --data raw_data.csv
  推理模式:  python pipeline.py --mode predict --data new_data.csv --model output/model.pkl --train-data raw_data.csv
"""

import os
import sys
import json
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd
from pathlib import Path

# 将当前目录加入 path，以便 import 同目录的阶段模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cleaning import clean_data
from feature_engineering import engineer_features
from model_training import train_model
{eval_import}

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="AutoML Pipeline")
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--data", required=True, help="输入数据路径 (CSV)")
    parser.add_argument("--model", default=None, help="模型路径 (predict 模式必需)")
    parser.add_argument("--train-data", default=None, help="训练集路径 (predict 模式下用于计算统计量)")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--target", default="{target_column}", help="目标列名")
    return parser.parse_args()


# ============================================
# 训练模式
# ============================================
def run_training(args):
    from sklearn.model_selection import train_test_split

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    model_dir = output_dir / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    target_column = args.target

    # ── 数据切分 ──
    print("=" * 60)
    print("[1/5] 数据切分")
    print("=" * 60)
    df = pd.read_csv(args.data)
    print(f"原始数据形状: {{df.shape}}")

    train_df, temp_df = train_test_split(df, test_size={1 - train_ratio}, random_state=42)
    relative_valid = {valid_ratio} / ({valid_ratio} + {test_ratio})
    valid_df, test_df = train_test_split(temp_df, test_size=1 - relative_valid, random_state=42)

    train_raw_path = str(data_dir / "train_raw.csv")
    valid_raw_path = str(data_dir / "valid_raw.csv")
    test_raw_path = str(data_dir / "test_raw.csv")
    train_df.to_csv(train_raw_path, index=False)
    valid_df.to_csv(valid_raw_path, index=False)
    test_df.to_csv(test_raw_path, index=False)
    print(f"切分完成: train={{len(train_df)}}, valid={{len(valid_df)}}, test={{len(test_df)}}")

    # ── 数据清洗 ──
    print("\\n" + "=" * 60)
    print("[2/5] 数据清洗")
    print("=" * 60)
    cleaned_train = str(data_dir / "cleaned_train.csv")
    cleaned_valid = str(data_dir / "cleaned_valid.csv")
    cleaned_test = str(data_dir / "cleaned_test.csv")
    clean_data(train_raw_path, cleaned_train, train_raw_path)
    print("  训练集清洗完成")
    clean_data(valid_raw_path, cleaned_valid, train_raw_path)
    print("  验证集清洗完成")
    clean_data(test_raw_path, cleaned_test, train_raw_path)
    print("  测试集清洗完成")

    # ── 特征工程 ──
    print("\\n" + "=" * 60)
    print("[3/5] 特征工程")
    print("=" * 60)
    features_train = str(data_dir / "features_train.csv")
    features_valid = str(data_dir / "features_valid.csv")
    features_test = str(data_dir / "features_test.csv")
    engineer_features(cleaned_train, features_train, cleaned_train)
    print("  训练集特征工程完成")
    engineer_features(cleaned_valid, features_valid, cleaned_train)
    print("  验证集特征工程完成")
    engineer_features(cleaned_test, features_test, cleaned_train)
    print("  测试集特征工程完成")

    # ── 模型训练 ──
    print("\\n" + "=" * 60)
    print("[4/5] 模型训练")
    print("=" * 60)
    result = train_model(
        train_path=features_train,
        valid_path=features_valid,
        test_path=features_test,
        model_dir=str(model_dir),
        target_column=target_column,
        summary_path=str(model_dir / "training_summary.json"),
    )
    model_path = result["model_path"]
    print(f"  模型训练完成，保存至: {{model_path}}")
{eval_train_block}

    print("\\n" + "=" * 60)
    print("训练流程完成!")
    print("=" * 60)
    print(f"输出目录: {{output_dir}}")


# ============================================
# 推理模式
# ============================================
def run_predict(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    if not args.model:
        print("错误: 推理模式需要指定 --model 参数")
        sys.exit(1)

    model_path = args.model
    target_column = args.target

    # ── 解析训练集路径 ──
    train_data_path = args.train_data
    if train_data_path is None:
        print("WARNING: 未指定 --train-data，使用预测数据自身计算统计量（不推荐）")
        train_data_path = args.data

    # ── 数据清洗 ──
    print("=" * 60)
    print("[1/5] 数据清洗")
    print("=" * 60)
    cleaned_path = str(tmp_dir / "cleaned.csv")
    clean_data(args.data, cleaned_path, train_data_path)

    # 清洗训练集参考（供特征工程计算统计量）
    cleaned_train_ref = str(tmp_dir / "cleaned_train_ref.csv")
    if train_data_path != args.data:
        clean_data(train_data_path, cleaned_train_ref, train_data_path)
    else:
        cleaned_train_ref = cleaned_path

    # ── 特征工程 ──
    print("\\n" + "=" * 60)
    print("[2/5] 特征工程")
    print("=" * 60)
    features_path = str(tmp_dir / "features.csv")
    engineer_features(cleaned_path, features_path, cleaned_train_ref)
{eval_predict_block}


# ============================================
# 入口
# ============================================
if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_training(args)
    elif args.mode == "predict":
        run_predict(args)
'''
        return script

    # ------------------------------------------------------------------
    # 兼容旧接口
    # ------------------------------------------------------------------

    def generate_pipeline_script(
        self,
        data_path: str,
        target_column: str,
        task_type: str = "regression",
    ) -> str:
        """兼容旧接口：生成 pipeline 并返回编排脚本内容。"""
        self.generate_pipeline_package(data_path, target_column, task_type)
        return self.asset_manager.read_asset("code", "pipeline.py")

    def get_pipeline_script(self) -> Optional[str]:
        """获取已生成的全流程脚本"""
        return self.asset_manager.read_asset("code", "pipeline.py")

    def save_pipeline_script(self, script: str) -> Dict[str, Any]:
        """保存全流程脚本"""
        result = self.asset_manager.save_code(
            code=script,
            filename="pipeline.py",
            metadata={
                "stage": "pipeline",
                "timestamp": datetime.now().isoformat(),
            }
        )
        return {"success": True, "path": result.path, "size": result.size}

    # ------------------------------------------------------------------
    # sklearn Pipeline 生成
    # ------------------------------------------------------------------

    def generate_sklearn_pipeline(
        self,
        data_path: str,
        target_column: str,
        task_type: str = "regression",
    ) -> str:
        """
        将各阶段代码组装为 sklearn Pipeline 并序列化。

        将 cleaning、feature_engineering 函数包装为自定义 Transformer，
        加上训练好的模型，组装为一个可直接 predict 的 Pipeline 对象。

        Args:
            data_path: 原始训练数据路径（用于 fit 时 bake 训练引用）
            target_column: 目标列名
            task_type: 任务类型

        Returns:
            序列化后的 Pipeline 文件路径
        """
        # 1. 读取各阶段源码
        cleaning_source = self.asset_manager.read_asset("code", "cleaning.py")
        fe_source = self.asset_manager.read_asset("code", "feature_engineering.py")

        if not cleaning_source or not fe_source:
            raise ValueError("缺少 cleaning.py 或 feature_engineering.py，无法组装 Pipeline")

        # 2. 加载已训练模型 artifact
        model_path = self.asset_manager.session_dir / "models" / "trained_model.pkl"
        if not model_path.exists():
            raise ValueError(f"模型文件不存在: {model_path}")
        model_artifact = joblib.load(model_path)

        # 3. 组装 Pipeline
        steps = [
            ("cleaning", DataFrameStageTransformer(cleaning_source, "clean_data", "cleaning")),
            ("feature_engineering", DataFrameStageTransformer(fe_source, "engineer_features", "feature_engineering")),
            ("drop_target", TargetColumnSplitter(target_column)),
            ("model", ModelStepWrapper(model_artifact, target_column, task_type)),
        ]
        pipeline = SklearnPipeline(steps)

        # 4. 用原始训练数据 fit（bake 训练引用到各 transformer 中）
        raw_train_df = pd.read_csv(data_path)
        pipeline.fit(raw_train_df)

        # 5. 序列化
        output_path = self.asset_manager.session_dir / "models" / "sklearn_pipeline.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, output_path)

        return str(output_path)
