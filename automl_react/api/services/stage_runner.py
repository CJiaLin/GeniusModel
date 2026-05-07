"""
Stage Runner

run_stage 中各阶段的 plan 生成编排逻辑
"""

import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from automl_react.agents import (
    DataCleaningAgent,
    DataExplorationAgent,
    DataSplittingAgent,
    FeatureEngineeringAgent,
    ModelTrainingAgent,
    run_dataset_split,
)
from automl_react.agents.data_analysis_agent import DataAnalysisAgent
from automl_react.assets import get_asset_manager
from automl_react.confirmation.confirmation_point import SkillReference

from ..helpers import (
    get_problem_definition_payload,
    get_train_raw_data_path,
    get_valid_raw_data_path,
    get_test_raw_data_path,
    compose_stage_task_description,
    ensure_agent,
)
from .llm_factory import create_llm_client


def run_problem_definition(
    session: Dict,
    workflow_state,
    confirmation_manager,
    data_path: str,
    session_id: str,
    requested_stage: str,
) -> Dict[str, Any]:
    """问题定义阶段：生成问题定义方案。"""
    model = workflow_state.get_context("model", "kimi-k2.5")
    llm = create_llm_client(model)
    agent = ensure_agent(session, "analysis", DataAnalysisAgent, llm=llm, session_id=session_id)

    try:
        task_description = workflow_state.get_context("task_description", "")
        print(f"[API] ========== 问题定义阶段开始 ==========")

        result = agent.generate_problem_definition(
            data_path=data_path,
            target_column=workflow_state.get_context("target_column"),
            task_type=workflow_state.get_context("task_type"),
            task_description=task_description,
        )
        definition_payload = agent.get_problem_definition_payload()

        confirmation_point = confirmation_manager.add_confirmation_point(
            stage="problem_definition",
            proposal_content=result,
            metadata={
                "stage": "problem_definition",
                "target_column": workflow_state.get_context("target_column"),
                "task_type": workflow_state.get_context("task_type"),
                "problem_definition": definition_payload,
            }
        )

        print(f"[API] ========== 问题定义阶段完成 ==========")
        response_payload = {
            "success": True,
            "stage": "problem_definition",
            "proposal": result,
            "problem_definition": definition_payload,
            "requires_confirmation": True,
            "confirmation_id": confirmation_point.id,
        }
        if requested_stage == "data_analysis":
            response_payload["analysis"] = result
        return response_payload
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_data_contract_check(
    session: Dict,
    workflow_state,
    confirmation_manager,
    data_path: str,
    target_column: Optional[str],
    task_type: Optional[str],
    session_id: str,
) -> Dict[str, Any]:
    """数据契约检查阶段。"""
    try:
        print(f"[API] ========== 数据契约检查阶段开始 ==========")
        from automl_react.agents.data_contract_agent import run_data_contract_checks

        problem_def = get_problem_definition_payload(workflow_state)
        contract_result = run_data_contract_checks(
            data_path=data_path,
            target_column=target_column or workflow_state.get_context("target_column", ""),
            task_type=task_type or workflow_state.get_context("task_type", "regression"),
            problem_definition=problem_def or None,
        )

        asset_manager = get_asset_manager(session_id=session_id)
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

        confirmation_point = confirmation_manager.add_confirmation_point(
            stage="data_contract_check",
            proposal_content=contract_result["summary"],
            metadata={
                "stage": "data_contract_check",
                "modelable": contract_result["modelable"],
                "stats": contract_result["stats"],
            },
        )

        print(f"[API] 契约检查结论: {'可建模' if contract_result['modelable'] else '不可建模'}")
        print(f"[API] ========== 数据契约检查阶段完成 ==========")
        return {
            "success": True,
            "stage": "data_contract_check",
            "modelable": contract_result["modelable"],
            "proposal": contract_result["summary"],
            "risk_list": contract_result["risk_list"],
            "questions_for_business": contract_result["questions_for_business"],
            "stats": contract_result["stats"],
            "requires_confirmation": True,
            "confirmation_id": confirmation_point.id,
        }
    except Exception as e:
        return {"success": False, "error": f"{e}\n{traceback.format_exc()}"}


def run_data_splitting(
    session: Dict,
    workflow_state,
    confirmation_manager,
    data_path: str,
    target_column: Optional[str],
    task_type: Optional[str],
    session_id: str,
) -> Dict[str, Any]:
    """数据集切分阶段。"""
    try:
        print(f"[API] ========== 数据集切分阶段开始 ==========")
        model = workflow_state.get_context("model", "kimi-k2.5")
        llm = create_llm_client(model)
        agent = ensure_agent(session, "splitting", DataSplittingAgent, llm=llm, session_id=session_id)
        problem_def = get_problem_definition_payload(workflow_state)
        split_plan = agent.generate_split_plan(
            data_path=data_path,
            target_column=target_column or workflow_state.get_context("target_column", ""),
            task_type=task_type or workflow_state.get_context("task_type", "classification"),
            problem_definition=problem_def or None,
            task_description=workflow_state.get_context("task_description", ""),
        )

        split_config = agent.split_config or {}
        preview_result = run_dataset_split(
            data_path=data_path,
            target_column=target_column or workflow_state.get_context("target_column", ""),
            task_type=task_type or workflow_state.get_context("task_type", "classification"),
            problem_definition=problem_def or None,
            output_dir=None,
            config=split_config,
        )

        asset_manager = get_asset_manager(session_id=session_id)
        asset_manager.save_data(
            data=split_plan,
            filename="dataset_split_report.md",
            asset_type="analysis",
            metadata={"stage": "data_splitting", "timestamp": datetime.now().isoformat()},
        )
        asset_manager.save_data(
            data=json.dumps(preview_result, ensure_ascii=False, indent=2),
            filename="dataset_split_result.json",
            asset_type="analysis",
            metadata={"stage": "data_splitting", "timestamp": datetime.now().isoformat()},
        )

        confirmation_point = confirmation_manager.add_confirmation_point(
            stage="data_splitting",
            proposal_content=split_plan,
            metadata={
                "stage": "data_splitting",
                "split_strategy": preview_result["split_strategy"],
                "has_validation_split": preview_result["has_validation_split"],
                "counts": preview_result["counts"],
                "split_config": split_config,
            },
        )
        confirmation_point.modifiable_aspects = agent.get_modifiable_aspects()

        print(f"[API] ========== 数据集切分阶段完成 ==========")
        return {
            "success": True,
            "stage": "data_splitting",
            "proposal": split_plan,
            "split_strategy": preview_result["split_strategy"],
            "counts": preview_result["counts"],
            "has_validation_split": preview_result["has_validation_split"],
            "questions_for_business": preview_result["questions_for_business"],
            "warnings": preview_result["warnings"],
            "split_config": split_config,
            "requires_confirmation": True,
            "confirmation_id": confirmation_point.id,
            "modifiable_aspects": confirmation_point.modifiable_aspects,
        }
    except Exception as e:
        return {"success": False, "error": f"{e}\n{traceback.format_exc()}"}


def run_data_exploration(
    session: Dict,
    workflow_state,
    data_path: str,
    target_column: Optional[str],
    task_type: Optional[str],
    session_id: str,
) -> Dict[str, Any]:
    """数据探索性分析阶段。"""
    model = workflow_state.get_context("model", "kimi-k2.5")
    llm = create_llm_client(model)
    agent = ensure_agent(session, "exploration", DataExplorationAgent, llm=llm, session_id=session_id)

    try:
        task_description = compose_stage_task_description(workflow_state)

        print(f"[API] ========== 数据探索性分析阶段开始 ==========")

        asset_manager = get_asset_manager(session_id=session_id)
        cleaning_result_json = asset_manager.read_asset("cleaning", "cleaning_result.json")
        cleaned_data_path = None
        if cleaning_result_json:
            try:
                cleaning_data = json.loads(cleaning_result_json)
                cleaned_data_path = cleaning_data.get("cleaned_data_path")
            except Exception:
                pass

        if not cleaned_data_path:
            cleaned_data_path = get_train_raw_data_path(workflow_state) or data_path
            print(f"[API] 使用训练集原始数据: {cleaned_data_path}")
        else:
            print(f"[API] 使用清洗后的数据: {cleaned_data_path}")

        result = agent.explore(
            cleaned_data_path,
            target_column=target_column,
            task_type=task_type,
            task_description=task_description
        )
        print(f"[API] explore 返回结果: success={result.get('success')}, answer长度={len(result.get('answer', ''))}")

        asset_manager.save_data(
            data=str(result.get("answer", "")),
            filename="data_exploration_result.md",
            asset_type="exploration",
            metadata={
                "stage": "data_exploration",
                "data_path": cleaned_data_path,
                "task_description": task_description,
                "timestamp": datetime.now().isoformat()
            }
        )
        print(f"[API] 探索性分析结果已保存到: exploration/data_exploration_result.md")
        print(f"[API] ========== 数据探索性分析阶段完成 ==========")

        return {
            "success": True,
            "stage": "data_exploration",
            "exploration": result.get("answer", ""),
            "requires_confirmation": False
        }
    except Exception as e:
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[API] 数据探索性分析错误: {error_detail}")
        return {"success": False, "error": error_detail}


def run_data_cleaning(
    session: Dict,
    workflow_state,
    confirmation_manager,
    data_path: str,
    session_id: str,
) -> Dict[str, Any]:
    """数据清洗阶段：生成清洗方案。"""
    model = workflow_state.get_context("model", "kimi-k2.5")
    llm = create_llm_client(model)
    agent = ensure_agent(session, "cleaning", DataCleaningAgent, llm=llm, session_id=session_id)

    try:
        print(f"[API] ========== 数据清洗阶段开始 ==========")
        cleaning_input_path = get_train_raw_data_path(workflow_state) or data_path

        task_description = compose_stage_task_description(workflow_state)
        if task_description:
            print(f"[API] 用户建模背景: {task_description[:100]}...")

        result = agent.generate_cleaning_plan(
            cleaning_input_path,
            task_description=task_description
        )
        print(f"[API] 清洗方案生成完成，长度: {len(result)} 字符")

        asset_manager = get_asset_manager(session_id=session_id)
        asset_manager.save_data(
            data=result,
            filename="cleaning_plan.md",
            asset_type="cleaning",
            metadata={
                "stage": "data_cleaning",
                "data_path": cleaning_input_path,
                "timestamp": datetime.now().isoformat()
            }
        )
        print(f"[API] 清洗方案已保存到: cleaning/cleaning_plan.md")
        print(f"[API] ========== 数据清洗阶段完成 ==========")

        skills_referenced = [
            SkillReference(
                skill_name="data-analysis-1.0.2",
                skill_path="skills/data-analysis-1.0.2",
                reference_file="techniques.md, pitfalls.md"
            )
        ]
        confirmation_point = confirmation_manager.add_confirmation_point(
            stage="data_cleaning",
            proposal_content=result,
            skills_referenced=skills_referenced
        )
        confirmation_point.modifiable_aspects = agent.get_modifiable_aspects()

        return {
            "success": True,
            "stage": "data_cleaning",
            "proposal": result,
            "requires_confirmation": True,
            "confirmation_id": confirmation_point.id,
            "modifiable_aspects": confirmation_point.modifiable_aspects,
            "skills_referenced": [{"name": s.skill_name, "files": [s.reference_file]} for s in skills_referenced],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_feature_engineering(
    session: Dict,
    workflow_state,
    confirmation_manager,
    data_path: str,
    target_column: Optional[str],
    task_type: Optional[str],
    session_id: str,
) -> Dict[str, Any]:
    """特征工程阶段：生成特征工程方案。"""
    agent = session["agents"].get("feature")
    if not agent:
        print(f"[API] 重新创建特征工程 Agent...")
        model = workflow_state.get_context("model", "kimi-k2.5")
        llm = create_llm_client(model)
        agent = ensure_agent(session, "feature", FeatureEngineeringAgent, llm=llm, session_id=session_id)
        print(f"[API] 特征工程 Agent 已重新创建")

    try:
        print(f"[API] ========== 特征工程阶段开始 ==========")

        task_description = compose_stage_task_description(workflow_state)
        if task_description:
            print(f"[API] 用户建模背景: {task_description[:100]}...")

        asset_manager = get_asset_manager(session_id=session_id)
        exploration_result = asset_manager.read_asset("exploration", "data_exploration_result.md")
        cleaning_result_json = asset_manager.read_asset("cleaning", "cleaning_result.json")
        cleaned_data_path = None
        if cleaning_result_json:
            try:
                cleaning_data = json.loads(cleaning_result_json)
                cleaned_data_path = cleaning_data.get("cleaned_data_path")
                if cleaned_data_path:
                    print(f"[API] 使用清洗后的数据: {cleaned_data_path}")
            except Exception:
                pass

        result = agent.generate_feature_plan(
            get_train_raw_data_path(workflow_state) or data_path,
            target_column,
            task_type,
            analysis_result=exploration_result,
            cleaned_data_path=cleaned_data_path,
            task_description=task_description
        )
        print(f"[API] 特征工程方案生成完成，长度: {len(result)} 字符")

        asset_manager.save_data(
            data=result,
            filename="feature_engineering_plan.md",
            asset_type="features",
            metadata={
                "stage": "feature_engineering",
                "data_path": cleaned_data_path or get_train_raw_data_path(workflow_state) or data_path,
                "has_exploration_input": exploration_result is not None,
                "has_cleaned_data_input": cleaned_data_path is not None,
                "timestamp": datetime.now().isoformat()
            }
        )
        print(f"[API] 特征工程方案已保存到: features/feature_engineering_plan.md")
        print(f"[API] ========== 特征工程阶段完成 ==========")

        skills_referenced = [
            SkillReference(
                skill_name="afrexai-ml-engineering-1.0.0",
                skill_path="skills/afrexai-ml-engineering-1.0.0",
                reference_file="SKILL.md (Phase 2: Data Engineering)"
            )
        ]
        confirmation_point = confirmation_manager.add_confirmation_point(
            stage="feature_engineering",
            proposal_content=result,
            skills_referenced=skills_referenced
        )
        confirmation_point.modifiable_aspects = agent.get_modifiable_aspects()

        return {
            "success": True,
            "stage": "feature_engineering",
            "proposal": result,
            "requires_confirmation": True,
            "confirmation_id": confirmation_point.id,
            "modifiable_aspects": confirmation_point.modifiable_aspects,
            "skills_referenced": [{"name": s.skill_name, "files": [s.reference_file]} for s in skills_referenced],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_model_training(
    session: Dict,
    workflow_state,
    confirmation_manager,
    data_path: str,
    target_column: Optional[str],
    task_type: Optional[str],
    session_id: str,
) -> Dict[str, Any]:
    """模型训练阶段：生成训练方案。"""
    model = workflow_state.get_context("model", "kimi-k2.5")
    llm = create_llm_client(model)
    agent = ensure_agent(session, "model", ModelTrainingAgent, llm=llm, session_id=session_id)

    try:
        print(f"[API] ========== 模型训练阶段开始 ==========")

        task_description = compose_stage_task_description(workflow_state)
        if task_description:
            print(f"[API] 用户建模背景: {task_description[:100]}...")

        asset_manager = get_asset_manager(session_id=session_id)
        exploration_result = asset_manager.read_asset("exploration", "data_exploration_result.md")
        feature_metrics_report = asset_manager.read_asset("features", "feature_metrics_report.md")

        feature_result_json = asset_manager.read_asset("features", "feature_engineering_result.json")
        features_data_path = None
        if feature_result_json:
            try:
                feature_data = json.loads(feature_result_json)
                features_data_path = feature_data.get("features_data_path")
                if features_data_path:
                    print(f"[API] 使用特征工程后的数据: {features_data_path}")
                # 防御性恢复
                if features_data_path and workflow_state:
                    if not workflow_state.get_context("features_train_path"):
                        workflow_state.set_context("features_train_path", features_data_path)
                        print(f"[API] 恢复 features_train_path: {features_data_path}")
                    fv = feature_data.get("features_valid_path")
                    if fv and not workflow_state.get_context("features_valid_path"):
                        workflow_state.set_context("features_valid_path", fv)
                        print(f"[API] 恢复 features_valid_path: {fv}")
                    ft = feature_data.get("features_test_path")
                    if ft and not workflow_state.get_context("features_test_path"):
                        workflow_state.set_context("features_test_path", ft)
                        print(f"[API] 恢复 features_test_path: {ft}")
            except Exception:
                pass

        result = agent.generate_model_plan(
            get_train_raw_data_path(workflow_state) or data_path,
            target_column,
            task_type,
            exploration_report=exploration_result,
            feature_metrics_report=feature_metrics_report,
            features_data_path=features_data_path,
            task_description=task_description,
            train_split_path=(workflow_state.get_context("features_train_path") if workflow_state else None) or get_train_raw_data_path(workflow_state),
            valid_split_path=(workflow_state.get_context("features_valid_path") if workflow_state else None) or get_valid_raw_data_path(workflow_state),
            test_split_path=(workflow_state.get_context("features_test_path") if workflow_state else None) or get_test_raw_data_path(workflow_state),
        )
        print(f"[API] 模型训练方案生成完成，长度: {len(result)} 字符")

        # 生成评估方案并合并
        try:
            eval_plan = agent.generate_evaluation_plan()
            if eval_plan:
                result = result + "\n\n---\n\n## 模型评估方案\n\n" + eval_plan
                print(f"[API] 评估方案已合并到训练方案")
        except Exception as eval_err:
            print(f"[API] 评估方案生成失败（不影响训练方案）: {eval_err}")

        asset_manager.save_data(
            data=result,
            filename="model_training_plan.md",
            asset_type="models",
            metadata={
                "stage": "model_training",
                "data_path": features_data_path or get_train_raw_data_path(workflow_state) or data_path,
                "has_exploration_input": exploration_result is not None,
                "has_feature_metrics_input": feature_metrics_report is not None,
                "features_data_path": features_data_path,
                "timestamp": datetime.now().isoformat()
            }
        )
        print(f"[API] 模型训练方案已保存到: models/model_training_plan.md")
        print(f"[API] ========== 模型训练阶段完成 ==========")

        skills_referenced = [
            SkillReference(
                skill_name="afrexai-ml-engineering-1.0.0",
                skill_path="skills/afrexai-ml-engineering-1.0.0",
                reference_file="SKILL.md (Phase 3: Model Selection)"
            )
        ]
        confirmation_point = confirmation_manager.add_confirmation_point(
            stage="model_training",
            proposal_content=result,
            skills_referenced=skills_referenced
        )
        confirmation_point.modifiable_aspects = agent.get_modifiable_aspects()

        return {
            "success": True,
            "stage": "model_training",
            "proposal": result,
            "requires_confirmation": True,
            "confirmation_id": confirmation_point.id,
            "modifiable_aspects": confirmation_point.modifiable_aspects,
            "skills_referenced": [{"name": s.skill_name, "files": [s.reference_file]} for s in skills_referenced],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
