"""
Stage 执行器

确认后的代码生成与执行逻辑（execute_* 函数）
"""

import os
import sys
import json
import subprocess as sp

from automl_react.utils.subprocess_executor import get_venv_python
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from automl_react.assets import get_asset_manager
from automl_react.agents import ModelEvaluationAgent

from ..helpers import (
    get_problem_definition_payload,
    get_effective_target_column,
    get_effective_task_type,
    get_train_raw_data_path,
    normalize_list,
)


def rerun_script_on_split(
    script_path: str,
    input_path: str,
    output_path: str,
    timeout: int = 300,
) -> Dict:
    """直接 subprocess 执行已保存的 .py 脚本，用于在 valid/test 上重跑清洗/特征工程代码。"""
    if not os.path.exists(script_path):
        return {"success": False, "error": f"脚本不存在: {script_path}"}
    if not os.path.exists(input_path):
        return {"success": False, "error": f"输入文件不存在: {input_path}"}

    try:
        result = sp.run(
            [get_venv_python(), script_path, input_path, output_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(script_path),
        )
        success = result.returncode == 0 and os.path.exists(output_path)
        return {
            "success": success,
            "output": result.stdout,
            "error": result.stderr.strip() if result.stderr and result.stderr.strip() else None,
            "return_code": result.returncode,
        }
    except sp.TimeoutExpired:
        return {"success": False, "error": f"执行超时: 超过 {timeout} 秒"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def execute_problem_definition(
    session: Dict,
    data_path: str,
    modifications: Optional[str] = None,
) -> Dict:
    """固化用户确认后的问题定义结果。"""
    session_id = session.get("session_id", "default")
    workflow_state = session.get("workflow_state")
    asset_manager = get_asset_manager(session_id=session_id)

    agent = session["agents"].get("analysis")
    if not agent:
        raise ValueError("问题定义 Agent 不存在")

    original_task_description = workflow_state.get_context("task_description", "") if workflow_state else ""
    generation_task_description = original_task_description
    if modifications:
        generation_task_description = (
            f"{original_task_description}\n\n## 用户确认补充\n\n{modifications}"
            if original_task_description else modifications
        )

    if modifications or not getattr(agent, "problem_definition_plan", None):
        agent.generate_problem_definition(
            data_path=data_path,
            target_column=workflow_state.get_context("target_column") if workflow_state else None,
            task_type=workflow_state.get_context("task_type") if workflow_state else None,
            task_description=generation_task_description,
        )

    payload = dict(agent.get_problem_definition_payload())
    if modifications:
        payload["user_confirmed_modifications"] = modifications

    report_text = agent.problem_definition_plan or ""
    report_asset = asset_manager.save_data(
        data=report_text,
        filename="problem_definition.md",
        asset_type="analysis",
        metadata={
            "stage": "problem_definition",
            "timestamp": datetime.now().isoformat(),
        }
    )
    json_asset = asset_manager.save_data(
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        filename="problem_definition.json",
        asset_type="analysis",
        metadata={
            "stage": "problem_definition",
            "timestamp": datetime.now().isoformat(),
        }
    )

    if workflow_state:
        workflow_state.update_context({
            "problem_definition": payload,
            "problem_definition_report": report_text,
            "problem_definition_path": report_asset.path,
            "problem_definition_json_path": json_asset.path,
            "primary_metric": payload.get("primary_metric"),
            "prediction_timing": payload.get("prediction_timing"),
            "business_constraints": normalize_list(payload.get("business_constraints")),
            "success_criteria": normalize_list(payload.get("success_criteria")),
        })
        workflow_state.save()

    return {
        "success": True,
        "stage": "problem_definition",
        "problem_definition_path": report_asset.path,
        "problem_definition_json_path": json_asset.path,
        "problem_definition": payload,
    }


async def execute_data_contract_check(
    session: Dict,
    data_path: str,
    modifications: Optional[str] = None,
) -> Dict:
    """固化用户确认后的数据契约检查结果到 workflow context。"""
    session_id = session.get("session_id", "default")
    workflow_state = session.get("workflow_state")
    asset_manager = get_asset_manager(session_id=session_id)

    contract_json = asset_manager.read_asset("analysis", "data_contract_result.json")
    if contract_json:
        contract_result = json.loads(contract_json)
    else:
        from automl_react.agents.data_contract_agent import run_data_contract_checks
        target_column = get_effective_target_column(workflow_state)
        task_type = get_effective_task_type(workflow_state)
        problem_def = get_problem_definition_payload(workflow_state)
        contract_result = run_data_contract_checks(
            data_path=data_path,
            target_column=target_column or "",
            task_type=task_type or "regression",
            problem_definition=problem_def or None,
        )
        asset_manager.save_data(
            data=contract_result["summary"],
            filename="data_contract_report.md",
            asset_type="analysis",
            metadata={"stage": "data_contract_check", "timestamp": datetime.now().isoformat()},
        )
        asset_manager.save_data(
            data=json.dumps(contract_result, ensure_ascii=False, indent=2, default=str),
            filename="data_contract_result.json",
            asset_type="analysis",
            metadata={"stage": "data_contract_check", "timestamp": datetime.now().isoformat()},
        )

    if workflow_state:
        workflow_state.update_context({
            "data_contract_modelable": contract_result.get("modelable", True),
            "data_contract_risk_list": contract_result.get("risk_list", []),
            "data_contract_questions": contract_result.get("questions_for_business", []),
        })
        if modifications:
            workflow_state.set_context("data_contract_user_notes", modifications)
        workflow_state.save()

    return {
        "success": True,
        "stage": "data_contract_check",
        "modelable": contract_result.get("modelable", True),
        "risk_count": len(contract_result.get("risk_list", [])),
    }


async def execute_data_splitting(
    session: Dict,
    data_path: str,
    modifications: Optional[str] = None,
) -> Dict:
    """根据用户确认后的方案生成切分代码并执行。"""
    from automl_react.agents import DataSplittingAgent

    session_id = session.get("session_id", "default")
    workflow_state = session.get("workflow_state")
    asset_manager = get_asset_manager(session_id=session_id)
    agent = session["agents"].get("splitting")

    if not agent:
        from .llm_factory import create_llm_client
        model = workflow_state.get_context("model", "kimi-k2.5") if workflow_state else "kimi-k2.5"
        llm = create_llm_client(model)
        agent = DataSplittingAgent(llm=llm, session_id=session_id)
        session["agents"]["splitting"] = agent

    target_column = get_effective_target_column(workflow_state) or (workflow_state.get_context("target_column") if workflow_state else "")
    task_type = get_effective_task_type(workflow_state) or (workflow_state.get_context("task_type", "classification") if workflow_state else "classification")
    problem_def = get_problem_definition_payload(workflow_state)

    if not agent.split_plan:
        agent.generate_split_plan(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,
            problem_definition=problem_def or None,
            task_description=workflow_state.get_context("task_description", "") if workflow_state else "",
        )

    code = agent.generate_split_code(modifications)
    split_result = agent.execute_split(code)

    split_paths = split_result.get("split_paths", {})
    asset_manager.save_data(
        data=split_result.get("summary", ""),
        filename="dataset_split_report.md",
        asset_type="analysis",
        metadata={"stage": "data_splitting", "timestamp": datetime.now().isoformat()},
    )
    asset_manager.save_data(
        data=json.dumps(split_result, ensure_ascii=False, indent=2),
        filename="dataset_split_result.json",
        asset_type="analysis",
        metadata={"stage": "data_splitting", "timestamp": datetime.now().isoformat()},
    )

    if workflow_state:
        workflow_state.set_context("data_split", split_result)
        workflow_state.set_context("train_raw_path", split_paths.get("train_raw_path"))
        workflow_state.set_context("valid_raw_path", split_paths.get("valid_raw_path"))
        workflow_state.set_context("test_raw_path", split_paths.get("test_raw_path"))
        workflow_state.set_context("split_strategy", split_result.get("split_strategy"))
        workflow_state.set_context("split_config", split_result.get("config", agent.split_config or {}))
        workflow_state.save()

    return {
        "success": True,
        "stage": "data_splitting",
        "split_strategy": split_result.get("split_strategy"),
        "has_validation_split": split_result.get("has_validation_split"),
        "counts": split_result.get("counts", {}),
        "execution_mode": split_result.get("execution_mode"),
        "train_raw_path": split_paths.get("train_raw_path"),
        "valid_raw_path": split_paths.get("valid_raw_path"),
        "test_raw_path": split_paths.get("test_raw_path"),
    }


async def execute_data_cleaning(session: Dict, data_path: str, modifications: Optional[str] = None) -> Dict:
    """执行数据清洗（CodeAct 模式）"""
    print(f"[API] ========== 开始执行数据清洗 ==========")
    print(f"[API] 数据路径: {data_path}")

    agent = session["agents"].get("cleaning")
    if not agent:
        raise ValueError("数据清洗 Agent 不存在")

    session_id = session.get("session_id", "default")
    asset_manager = get_asset_manager(session_id=session_id)

    if not agent.cleaning_plan:
        print(f"[API] 清洗方案未生成，开始生成...")
        agent.generate_cleaning_plan(data_path)
    else:
        print(f"[API] 清洗方案已存在，长度: {len(agent.cleaning_plan)} 字符")

    print(f"[API] 开始生成并执行清洗代码（CodeAct 模式）...")
    try:
        code = agent.generate_cleaning_code(modifications)
        print(f"[API] 清洗代码生成并执行完成，长度: {len(code) if code else 0} 字符")
    except Exception as e:
        print(f"[API] 清洗代码生成执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

    final_cleaned_path = agent.cleaned_data_path
    file_exists = os.path.exists(final_cleaned_path)

    if not file_exists:
        print(f"[API] 警告: 清洗后的数据文件不存在: {final_cleaned_path}")

    # ====== 在 valid/test 上重跑同一清洗脚本 ======
    workflow_state = session.get("workflow_state")
    cleaned_valid_path = None
    cleaned_test_path = None

    if file_exists and workflow_state:
        script_path = str(asset_manager.session_dir / "code" / "cleaning.py")
        valid_raw = workflow_state.get_context("valid_raw_path")
        test_raw = workflow_state.get_context("test_raw_path")
        data_dir = str(asset_manager.session_dir / "data")

        if valid_raw and os.path.exists(valid_raw):
            cleaned_valid_path = os.path.join(data_dir, "cleaned_valid.csv")
            r = rerun_script_on_split(script_path, valid_raw, cleaned_valid_path)
            if r["success"]:
                print(f"[API] valid 清洗完成: {cleaned_valid_path}")
                workflow_state.set_context("cleaned_valid_path", cleaned_valid_path)
            else:
                print(f"[API] valid 清洗失败: {r.get('error')}")
                cleaned_valid_path = None

        if test_raw and os.path.exists(test_raw):
            cleaned_test_path = os.path.join(data_dir, "cleaned_test.csv")
            r = rerun_script_on_split(script_path, test_raw, cleaned_test_path)
            if r["success"]:
                print(f"[API] test 清洗完成: {cleaned_test_path}")
                workflow_state.set_context("cleaned_test_path", cleaned_test_path)
            else:
                print(f"[API] test 清洗失败: {r.get('error')}")
                cleaned_test_path = None

        workflow_state.set_context("cleaned_train_path", final_cleaned_path)

    execution_result = {
        "success": file_exists,
        "cleaned_data_path": final_cleaned_path,
        "cleaned_valid_path": cleaned_valid_path,
        "cleaned_test_path": cleaned_test_path,
        "original_path": data_path,
        "generated_code": code,
        "timestamp": datetime.now().isoformat(),
        "stage": "data_cleaning"
    }

    asset_manager.save_data(
        data=json.dumps(execution_result, ensure_ascii=False, indent=2),
        filename="cleaning_result.json",
        asset_type="cleaning",
        metadata=execution_result
    )

    print(f"[API] ========== 数据清洗执行完成 ==========")
    return execution_result


async def execute_feature_engineering(
    session: Dict,
    data_path: str,
    target_column: str,
    task_type: str,
    modifications: Optional[str] = None
) -> Dict:
    """执行特征工程（CodeAct 模式）"""
    print(f"[API] ========== 开始执行特征工程 ==========")
    print(f"[API] 数据路径: {data_path}")

    agent = session["agents"].get("feature")
    if not agent:
        raise ValueError("特征工程 Agent 不存在")

    session_id = session.get("session_id", "default")
    asset_manager = get_asset_manager(session_id=session_id)

    if not agent.feature_plan:
        print(f"[API] 特征工程方案未生成，开始生成...")
        agent.generate_feature_plan(data_path, target_column, task_type)
    else:
        print(f"[API] 特征工程方案已存在，长度: {len(agent.feature_plan)} 字符")

    print(f"[API] 开始生成并执行特征工程代码（CodeAct 模式）...")
    try:
        code = agent.generate_feature_code(modifications)
        print(f"[API] 特征工程代码生成并执行完成，长度: {len(code) if code else 0} 字符")
    except Exception as e:
        print(f"[API] 特征工程代码生成执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

    features_data_path = agent.features_data_path if hasattr(agent, 'features_data_path') else None
    file_exists = features_data_path and os.path.exists(features_data_path)
    if not file_exists:
        print(f"[API] 警告: 特征工程后的数据文件不存在: {features_data_path}")

    # ====== 在 valid/test 上重跑同一特征工程脚本 ======
    workflow_state = session.get("workflow_state")
    features_valid_path = None
    features_test_path = None

    if file_exists and workflow_state:
        script_path = str(asset_manager.session_dir / "code" / "feature_engineering.py")
        cleaned_valid = workflow_state.get_context("cleaned_valid_path")
        cleaned_test = workflow_state.get_context("cleaned_test_path")
        data_dir = str(asset_manager.session_dir / "data")

        if cleaned_valid and os.path.exists(cleaned_valid):
            features_valid_path = os.path.join(data_dir, "features_valid.csv")
            r = rerun_script_on_split(script_path, cleaned_valid, features_valid_path)
            if r["success"]:
                print(f"[API] valid 特征工程完成: {features_valid_path}")
                workflow_state.set_context("features_valid_path", features_valid_path)
            else:
                print(f"[API] valid 特征工程失败: {r.get('error')}")
                features_valid_path = None

        if cleaned_test and os.path.exists(cleaned_test):
            features_test_path = os.path.join(data_dir, "features_test.csv")
            r = rerun_script_on_split(script_path, cleaned_test, features_test_path)
            if r["success"]:
                print(f"[API] test 特征工程完成: {features_test_path}")
                workflow_state.set_context("features_test_path", features_test_path)
            else:
                print(f"[API] test 特征工程失败: {r.get('error')}")
                features_test_path = None

        workflow_state.set_context("features_train_path", features_data_path)

    execution_result = {
        "success": file_exists,
        "features_data_path": features_data_path if file_exists else None,
        "features_valid_path": features_valid_path,
        "features_test_path": features_test_path,
        "original_path": data_path,
        "generated_code": code,
        "timestamp": datetime.now().isoformat(),
        "stage": "feature_engineering",
        "evaluation_available": file_exists,
        "evaluation_required_confirmation": file_exists
    }

    asset_manager.save_data(
        data=json.dumps(execution_result, ensure_ascii=False, indent=2),
        filename="feature_engineering_result.json",
        asset_type="features",
        metadata=execution_result
    )

    print(f"[API] ========== 特征工程执行完成 ==========")
    return execution_result


async def execute_feature_evaluation(session: Dict, modifications: Optional[str] = None) -> Dict:
    """执行特征评估（可解释性与可靠性分析）"""
    print(f"[API] ========== 开始执行特征评估 ==========")

    agent = session["agents"].get("feature")
    if not agent:
        raise ValueError("特征工程 Agent 不存在")

    result = agent.calculate_feature_metrics(modifications=modifications)

    asset_manager = get_asset_manager(session_id=session.get("session_id", "default"))
    payload = {
        "success": result.get("success", False),
        "metrics_result_path": result.get("metrics_result_path"),
        "metrics_report_path": result.get("metrics_report_path"),
        "features_data_path": result.get("features_data_path"),
        "timestamp": result.get("timestamp", datetime.now().isoformat()),
        "stage": "feature_evaluation"
    }
    asset_manager.save_data(
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        filename="feature_evaluation_result.json",
        asset_type="features",
        metadata=payload
    )

    print(f"[API] ========== 特征评估执行完成 ==========")
    return payload


async def execute_model_training(
    session: Dict,
    data_path: str,
    target_column: str,
    task_type: str,
    modifications: Optional[str] = None
) -> Dict:
    """执行模型训练"""
    workflow_state = session.get("workflow_state")

    agent = session["agents"].get("model")
    if not agent:
        raise ValueError("模型训练 Agent 不存在")

    asset_manager = get_asset_manager(session_id=session.get("session_id", "default"))

    feature_result_json = asset_manager.read_asset("features", "feature_engineering_result.json")
    features_data_path = None
    if feature_result_json:
        try:
            feature_data = json.loads(feature_result_json)
            features_data_path = feature_data.get("features_data_path")
            # 防御性恢复
            if features_data_path and workflow_state:
                if not workflow_state.get_context("features_train_path"):
                    workflow_state.set_context("features_train_path", features_data_path)
                fv = feature_data.get("features_valid_path")
                if fv and not workflow_state.get_context("features_valid_path"):
                    workflow_state.set_context("features_valid_path", fv)
                ft = feature_data.get("features_test_path")
                if ft and not workflow_state.get_context("features_test_path"):
                    workflow_state.set_context("features_test_path", ft)
        except Exception:
            features_data_path = None

    exploration_report = asset_manager.read_asset("exploration", "data_exploration_result.md")
    feature_metrics_report = asset_manager.read_asset("features", "feature_metrics_report.md")

    if not agent.model_plan:
        agent.generate_model_plan(
            data_path,
            target_column,
            task_type,
            exploration_report=exploration_report,
            feature_metrics_report=feature_metrics_report,
            features_data_path=features_data_path,
            train_split_path=(workflow_state.get_context("features_train_path") or workflow_state.get_context("train_raw_path")) if workflow_state else None,
            valid_split_path=(workflow_state.get_context("features_valid_path") or workflow_state.get_context("valid_raw_path")) if workflow_state else None,
            test_split_path=(workflow_state.get_context("features_test_path") or workflow_state.get_context("test_raw_path")) if workflow_state else None,
        )
    elif workflow_state:
        agent.train_split_path = workflow_state.get_context("features_train_path") or workflow_state.get_context("train_raw_path")
        agent.valid_split_path = workflow_state.get_context("features_valid_path") or workflow_state.get_context("valid_raw_path")
        agent.test_split_path = workflow_state.get_context("features_test_path") or workflow_state.get_context("test_raw_path")

    code = agent.generate_model_code(modifications)
    result = agent.execute_model_training(code)

    execution_result = {
        "success": result.get("success", False),
        "model_path": result.get("model_path"),
        "train_split_path": result.get("train_split_path"),
        "valid_split_path": result.get("valid_split_path"),
        "test_split_path": result.get("test_split_path"),
        "training_summary_path": result.get("training_summary_path"),
        "metrics": result.get("metrics", {}),
        "selected_feature_names": result.get("selected_feature_names", []),
        "features_data_path": result.get("data_path"),
        "artifact_status": result.get("artifact_status", {}),
        "generated_code": code,
        "timestamp": result.get("timestamp"),
        "stage": "model_training"
    }

    asset_manager.save_data(
        data=json.dumps(execution_result, ensure_ascii=False, indent=2),
        filename="model_training_result.json",
        asset_type="models",
        metadata=execution_result
    )

    return execution_result


async def execute_model_evaluation(
    session: Dict,
    target_column: str,
    task_type: str,
) -> Dict:
    """执行模型评估。"""
    agent = session["agents"].get("evaluation")
    if not agent:
        agent = ModelEvaluationAgent(session_id=session.get("session_id", "default"))
        session["agents"]["evaluation"] = agent

    return agent.evaluate_from_training_result(
        target_column=target_column,
        task_type=task_type,
    )
