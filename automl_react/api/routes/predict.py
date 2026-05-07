"""
预测 & 上传路由

/predict, /upload 端点
"""

import os
import sys
import subprocess as sp
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from automl_react.utils.subprocess_executor import get_venv_python

from automl_react.assets import get_asset_manager

from ..deps import validate_session_id, get_registry
from ..registry import AppRegistry
from ..helpers import (
    get_session_original_data_path,
    save_data_onboarding_artifacts,
    get_effective_target_column,
    get_effective_task_type,
)

router = APIRouter()


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
):
    """上传数据文件到会话资产目录"""
    if not session_id:
        session_id = f"session_{int(datetime.now().timestamp() * 1000)}"

    validate_session_id(session_id)
    dest = get_session_original_data_path(session_id)
    dest.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    save_data_onboarding_artifacts(session_id, dest, file.filename or "uploaded_file")

    return {
        "success": True,
        "session_id": session_id,
        "filename": file.filename,
        "file_path": str(dest),
        "file_size": len(content),
    }


@router.post("/predict")
async def predict_run(session_id: str, data_path: str, output_dir: str = None, registry: AppRegistry = Depends(get_registry)):
    """使用已训练模型对新数据进行预测"""
    validate_session_id(session_id)
    session = await registry.get_session(session_id)
    workflow_state = session.get("workflow_state")
    if not workflow_state:
        raise HTTPException(status_code=404, detail="会话不存在")

    asset_manager = get_asset_manager(session_id=session_id)

    model_path = str(asset_manager.session_dir / "models" / "trained_model.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="trained_model.pkl 不存在，请先完成模型训练")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=400, detail=f"输入数据不存在: {data_path}")

    target_column = get_effective_target_column(workflow_state)

    if not output_dir:
        output_dir = str(asset_manager.session_dir / "predictions")
    os.makedirs(output_dir, exist_ok=True)

    predictions_path = os.path.join(output_dir, "predictions.csv")

    task_type = get_effective_task_type(workflow_state)

    # 优先使用 sklearn Pipeline pkl（保证训练/预测代码路径一致）
    sklearn_pipeline_path = asset_manager.session_dir / "models" / "sklearn_pipeline.pkl"
    if sklearn_pipeline_path.exists():
        try:
            import joblib
            import pandas as pd

            pipeline = joblib.load(sklearn_pipeline_path)
            raw_df = pd.read_csv(data_path)
            predictions = pipeline.predict(raw_df)

            result_df = pd.DataFrame({"prediction": predictions})

            # 分类模型同时输出预测得分和预测类别
            if task_type == "classification" and hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(raw_df)
                if proba.shape[1] == 2:
                    # 二分类：输出正类概率
                    result_df.insert(0, "prediction_score", proba[:, 1])
                else:
                    # 多分类：输出各类别概率
                    classes = pipeline.classes_ if hasattr(pipeline, "classes_") else range(proba.shape[1])
                    for i, cls in enumerate(classes):
                        result_df[f"score_class_{cls}"] = proba[:, i]
                # 将 prediction 列重命名为 prediction_label 以区分
                result_df.rename(columns={"prediction": "prediction_label"}, inplace=True)

            if "Id" in raw_df.columns:
                result_df.insert(0, "Id", raw_df["Id"].values)
            elif "ID" in raw_df.columns:
                result_df.insert(0, "ID", raw_df["ID"].values)
            result_df.to_csv(predictions_path, index=False)

            return {
                "success": True,
                "predictions_path": predictions_path,
                "records": len(result_df),
                "method": "sklearn_pipeline",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            # sklearn Pipeline 失败时 fallback 到脚本方式
            print(f"[predict] sklearn Pipeline 执行失败，fallback 到脚本: {e}")

    # Fallback: subprocess 执行 pipeline.py
    pipeline_path = str(asset_manager.session_dir / "code" / "pipeline.py")
    if not os.path.exists(pipeline_path):
        raise HTTPException(status_code=400, detail="pipeline.py 不存在，请先调用 /pipeline/generate")

    train_raw_path = workflow_state.get_context("train_raw_path") or workflow_state.get_context("data_path")

    cmd = [
        get_venv_python(), pipeline_path,
        "--mode", "predict",
        "--data", os.path.abspath(data_path),
        "--model", model_path,
        "--output-dir", output_dir,
        "--target", target_column,
    ]
    if train_raw_path and os.path.exists(train_raw_path):
        cmd.extend(["--train-data", train_raw_path])

    try:
        result = sp.run(cmd, capture_output=True, text=True, timeout=600, cwd=os.path.dirname(pipeline_path))
    except sp.TimeoutExpired:
        raise HTTPException(status_code=504, detail="预测超时 (600s)")

    success = result.returncode == 0 and os.path.exists(predictions_path)

    response = {
        "success": success,
        "predictions_path": predictions_path if success else None,
        "method": "script_fallback",
        "stdout": result.stdout,
        "timestamp": datetime.now().isoformat(),
    }

    if not success:
        response["error"] = result.stderr.strip() if result.stderr else "预测失败"
        raise HTTPException(status_code=500, detail=response)

    try:
        import pandas as pd
        pred_df = pd.read_csv(predictions_path)
        response["records"] = len(pred_df)
    except Exception:
        pass

    return response


@router.post("/predict/upload")
async def predict_upload(
    file: UploadFile = File(...),
    session_id: str = None,
    output_dir: str = None,
    registry: AppRegistry = Depends(get_registry),
):
    """上传数据文件并使用已训练模型进行预测"""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    validate_session_id(session_id)
    asset_manager = get_asset_manager(session_id=session_id)
    predict_dir = asset_manager.session_dir / "predictions"
    predict_dir.mkdir(parents=True, exist_ok=True)

    upload_path = predict_dir / (file.filename or "predict_input.csv")
    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    return await predict_run(
        session_id=session_id,
        data_path=str(upload_path),
        output_dir=output_dir,
        registry=registry,
    )
