#!/usr/bin/env python3
"""
从已有 session 继续执行 E2E 测试 — Phase 4 ~ Phase 8
Session: e2e_kimi_1776826448
"""

import json
import os
import sys
import time

import requests

# ─── 配置 ───
BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
SESSION_ID = "e2e_kimi_1776826448"
DATA_PATH = os.path.abspath("train.csv")
TIMEOUT = 10800  # 3h

# ─── 结果收集 ───
results = {}


def record(name: str, passed: bool, detail: str = ""):
    results[name] = {"passed": passed, "detail": detail}
    icon = "✅" if passed else "❌"
    print(f"  {icon} [{name}] {detail}")


def api(method, path, data=None, label=""):
    url = f"{BASE_URL}{path}"
    print(f"  ⏳ {label or path} ...", end="", flush=True)
    try:
        if method == "GET":
            resp = requests.get(url, timeout=TIMEOUT)
        else:
            resp = requests.post(url, json=data or {}, timeout=TIMEOUT)
        print(" done")
        if resp.status_code >= 400:
            print(f"    HTTP {resp.status_code}: {resp.text[:500]}")
            return None
        return resp.json()
    except requests.exceptions.Timeout:
        print(f" TIMEOUT ({TIMEOUT}s)")
        return None
    except Exception as e:
        print(f" ERROR: {e}")
        return None


def confirm(conf_id, status="confirmed", modifications=None):
    data = {"session_id": SESSION_ID, "confirmation_id": conf_id, "status": status}
    if modifications:
        data["modifications"] = modifications
    return api("POST", "/confirmation/submit", data, f"确认({status})")


def session_dir():
    return os.path.join("assets", SESSION_ID)


# ═══════════════════════════════════════════════════
# Phase 4: 模型训练
# ═══════════════════════════════════════════════════
def phase4_model_training():
    print("\n" + "=" * 70)
    print("Phase 4: 模型训练 — Skill 引用 + 代码保存 + 模型资产")
    print("=" * 70)

    result = api("POST", f"/workflow/{SESSION_ID}/stage/model_training/run", label="生成建模方案")
    if not result or not result.get("success"):
        record("model_plan", False, f"方案生成失败: {result}")
        return False

    proposal = result.get("proposal", "")
    conf_id = result.get("confirmation_id", "")
    skills_ref = result.get("skills_referenced", [])
    skill_names = [s.get("name", "") for s in skills_ref] if skills_ref else []

    record("model_plan", len(proposal) > 50, f"方案={len(proposal)} chars")
    record("model_skill_ref", bool(skills_ref), f"skills_referenced={skill_names}")

    # 确认并执行
    exec_result = confirm(conf_id)
    if not exec_result:
        record("model_exec", False, "确认执行失败")
        return False

    execution = exec_result.get("execution", {})
    model_path = execution.get("model_path", "")
    metrics = execution.get("metrics", {})

    record("model_exec", execution.get("success", False),
           f"model_path={os.path.basename(model_path or 'N/A')}, "
           f"metrics_keys={list(metrics.keys()) if metrics else 'none'}")

    # 验证代码保存
    code_file = os.path.join(session_dir(), "code", "model_training.py")
    record("model_code_saved", os.path.exists(code_file),
           f"model_training.py 存在={os.path.exists(code_file)}")

    # 验证模型文件
    model_pkl = os.path.join(session_dir(), "models", "trained_model.pkl")
    record("model_pkl_saved", os.path.exists(model_pkl),
           f"trained_model.pkl 存在={os.path.exists(model_pkl)}")

    # 验证 training_summary.json
    summary_file = os.path.join(session_dir(), "models", "training_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            summary = json.load(f)
        has_metrics = bool(summary.get("metrics") or summary.get("validation_metrics"))
        has_features = bool(summary.get("selected_feature_names") or summary.get("feature_names"))
        record("training_summary", has_metrics and has_features,
               f"has_metrics={has_metrics}, has_features={has_features}")
    else:
        record("training_summary", False, "training_summary.json 不存在")

    # 验证模型产物结构
    if os.path.exists(model_pkl):
        try:
            import joblib
            artifact = joblib.load(model_pkl)
            if isinstance(artifact, dict):
                keys = list(artifact.keys())
                has_model = "model" in keys
                has_features = "selected_feature_names" in keys
                record("model_artifact_struct", has_model and has_features,
                       f"keys={keys[:6]}")
            else:
                record("model_artifact_struct", False, f"类型={type(artifact).__name__}，非 dict")
        except Exception as e:
            record("model_artifact_struct", False, f"读取失败: {e}")

    return True


# ═══════════════════════════════════════════════════
# Phase 5: 模型评估
# ═══════════════════════════════════════════════════
def phase5_model_evaluation():
    print("\n" + "=" * 70)
    print("Phase 5: 模型评估")
    print("=" * 70)

    result = api("POST", f"/workflow/{SESSION_ID}/stage/model_evaluation/run", label="生成评估方案")
    if not result or not result.get("success"):
        record("eval_plan", False, f"评估方案生成失败: {result}")
        return False
    record("eval_plan", True, f"方案={len(result.get('proposal', ''))} chars")

    conf_id = result.get("confirmation_id", "")
    exec_result = confirm(conf_id)
    if not exec_result:
        record("eval_exec", False, "评估执行失败")
        return False

    evaluation = exec_result.get("execution", {})
    metrics = evaluation.get("metrics", {})
    record("eval_exec", evaluation.get("success", False) and bool(metrics),
           f"metrics={list(metrics.keys()) if metrics else 'none'}")
    return True


# ═══════════════════════════════════════════════════
# Phase 5.5: Pipeline 生成 + Predict 验证
# ═══════════════════════════════════════════════════
def phase5_5_pipeline_and_predict():
    print("\n" + "=" * 70)
    print("Phase 5.5: Pipeline 生成 + Predict 端点验证")
    print("=" * 70)

    result = api("POST", f"/pipeline/generate?session_id={SESSION_ID}", label="生成 pipeline.py")
    if not result or not result.get("success"):
        record("pipeline_generate", False, f"pipeline 生成失败: {result}")
        return False

    pipeline_path = os.path.join(session_dir(), "code", "pipeline.py")
    pipeline_exists = os.path.exists(pipeline_path)
    record("pipeline_generate", pipeline_exists, f"pipeline.py 存在={pipeline_exists}")

    if pipeline_exists:
        with open(pipeline_path) as f:
            content = f.read()
        has_train_path = "def clean_data(input_path, output_path, train_path" in content
        has_schema_validate = "def validate_schema(" in content
        has_train_data_arg = "--train-data" in content
        record("pipeline_p0_features",
               has_train_path and has_schema_validate and has_train_data_arg,
               f"train_path={has_train_path}, schema_validate={has_schema_validate}, --train-data={has_train_data_arg}")

    # 调用 /predict 端点
    result = api("POST", f"/predict?session_id={SESSION_ID}&data_path={DATA_PATH}", label="调用 /predict")
    if not result:
        record("predict_api", False, "predict 请求失败")
        return False

    success = result.get("success", False)
    predictions_path = result.get("predictions_path", "")
    records_count = result.get("records", 0)
    record("predict_api", success and records_count > 0,
           f"success={success}, records={records_count}, path={os.path.basename(predictions_path or 'N/A')}")

    if predictions_path and os.path.exists(predictions_path):
        import pandas as pd
        pred_df = pd.read_csv(predictions_path)
        has_prediction_col = "prediction" in pred_df.columns
        no_nan = not pred_df["prediction"].isna().any() if has_prediction_col else False
        record("predict_output_valid",
               has_prediction_col and no_nan and len(pred_df) > 0,
               f"rows={len(pred_df)}, has_prediction={has_prediction_col}, no_nan={no_nan}")
    else:
        record("predict_output_valid", False,
               f"predictions.csv 不存在: {predictions_path}")

    return True


# ═══════════════════════════════════════════════════
# Phase 6: 资产完整性校验
# ═══════════════════════════════════════════════════
def phase6_verify_assets():
    print("\n" + "=" * 70)
    print("Phase 6: 资产完整性校验")
    print("=" * 70)

    sd = session_dir()
    expected = {
        "data/original_data.csv": "原始数据",
        "data/train_raw.csv": "训练集",
        "cleaning/cleaning_plan.md": "清洗方案",
        "features/feature_engineering_plan.md": "特征方案",
        "models/model_training_plan.md": "建模方案",
        "models/trained_model.pkl": "训练模型",
        "models/training_summary.json": "训练摘要",
        "code/model_training.py": "训练代码",
        "state/workflow_state.json": "工作流状态",
    }

    missing = []
    for rel_path, desc in expected.items():
        if not os.path.exists(os.path.join(sd, rel_path)):
            missing.append(f"{desc}({rel_path})")

    record("asset_completeness",
           len(missing) == 0,
           f"缺失: {missing}" if missing else f"全部 {len(expected)} 项齐全")

    pipeline = os.path.join(sd, "code", "pipeline.py")
    if os.path.exists(pipeline):
        with open(pipeline) as f:
            content = f.read()
        record("pipeline_script", len(content) > 100,
               f"pipeline.py={len(content)} chars")
    else:
        record("pipeline_script", False, "pipeline.py 不存在")


# ═══════════════════════════════════════════════════
# Phase 7: 上下文压缩验证
# ═══════════════════════════════════════════════════
def phase7_verify_context_compression():
    print("\n" + "=" * 70)
    print("Phase 7: 上下文压缩验证")
    print("=" * 70)

    log_dir = os.path.join("logs", "llm_calls", SESSION_ID)
    if not os.path.exists(log_dir):
        record("context_compression_log_dir", False, f"日志目录不存在: {log_dir}")
        return

    stage_calls = {}
    total_calls = 0
    token_info_found = False
    summarization_found = False
    summarization_details = []

    for filename in sorted(os.listdir(log_dir)):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(log_dir, filename)
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_calls += 1
                stage = entry.get("stage", "unknown")
                stage_calls[stage] = stage_calls.get(stage, 0) + 1

                meta = entry.get("metadata", {})
                if meta.get("input_tokens") or meta.get("total_tokens"):
                    token_info_found = True

                if meta.get("summarization"):
                    summarization_found = True
                    summarization_details.append({
                        "stage": stage,
                        "before": meta["summarization"].get("before_tokens"),
                        "after": meta["summarization"].get("after_tokens"),
                        "saved": meta["summarization"].get("saved_tokens"),
                    })

    record("llm_call_count", total_calls > 0,
           f"总调用={total_calls}, 各阶段={stage_calls}")

    record("token_monitor_active", token_info_found,
           "LLM 日志中包含 token 统计" if token_info_found else "未发现 token 统计")

    record("context_summarization", summarization_found,
           f"摘要触发={len(summarization_details)}次, 详情={summarization_details}"
           if summarization_found else "未触发上下文摘要（可能 token 未超阈值）")

    state_file = os.path.join(session_dir(), "state", "workflow_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        history = state.get("history", [])
        stages_completed = [h.get("stage") for h in history if h.get("stage")]
        record("workflow_state_history", len(stages_completed) >= 4,
               f"已完成阶段={stages_completed}")
    else:
        record("workflow_state_history", False, "workflow_state.json 不存在")


# ═══════════════════════════════════════════════════
# Phase 8: Skill 工具调用日志验证
# ═══════════════════════════════════════════════════
def phase8_verify_skill_usage():
    print("\n" + "=" * 70)
    print("Phase 8: Skill 工具调用日志验证")
    print("=" * 70)

    log_dir = os.path.join("logs", "llm_calls", SESSION_ID)
    if not os.path.exists(log_dir):
        record("skill_log_dir", False, f"日志目录不存在: {log_dir}")
        return

    skill_tool_calls = []
    skill_content_reads = []
    new_skill_found = {"feature-engineering-patterns": False, "model-selection-heuristics": False}

    for filename in sorted(os.listdir(log_dir)):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(log_dir, filename)
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = json.dumps(entry, ensure_ascii=False)
                stage = entry.get("stage", "unknown")

                if any(kw in content for kw in ["read_skill", "search_skills"]):
                    skill_tool_calls.append(stage)

                if "以下内容为技术参考" in content:
                    skill_content_reads.append(stage)

                if "feature-engineering-patterns" in content:
                    new_skill_found["feature-engineering-patterns"] = True
                if "model-selection-heuristics" in content:
                    new_skill_found["model-selection-heuristics"] = True

    record("skill_tool_calls", bool(skill_tool_calls),
           f"skill 工具调用出现在阶段: {list(set(skill_tool_calls))}"
           if skill_tool_calls else "未发现 skill 工具调用")

    record("skill_content_read", bool(skill_content_reads),
           f"skill 内容实际读取的阶段: {list(set(skill_content_reads))}"
           if skill_content_reads else "未读取到 skill 实际内容")

    record("new_skill_fe_patterns", new_skill_found["feature-engineering-patterns"],
           "feature-engineering-patterns 技能包被引用"
           if new_skill_found["feature-engineering-patterns"] else "未引用 feature-engineering-patterns")

    record("new_skill_model_select", new_skill_found["model-selection-heuristics"],
           "model-selection-heuristics 技能包被引用"
           if new_skill_found["model-selection-heuristics"] else "未引用 model-selection-heuristics")


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════
def main():
    # 等待后端就绪
    print("等待后端就绪...")
    for _ in range(30):
        try:
            requests.get(f"{BASE_URL}/", timeout=2)
            print("✅ 后端已就绪\n")
            break
        except Exception:
            time.sleep(1)
    else:
        print("❌ 后端未就绪，退出")
        sys.exit(1)

    print(f"\n{'#' * 70}")
    print(f"# E2E 续跑测试 — 从 Phase 4 继续")
    print(f"# Session: {SESSION_ID}")
    print(f"{'#' * 70}\n")

    try:
        phase4_model_training()
        phase5_model_evaluation()
        phase5_5_pipeline_and_predict()
        phase6_verify_assets()
        phase7_verify_context_compression()
        phase8_verify_skill_usage()

    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 未预期异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ── 测试总结 ──
        print("\n\n" + "=" * 70)
        print("测试结果总结 (Phase 4~8)")
        print("=" * 70)

        passed = sum(1 for r in results.values() if r["passed"])
        failed = sum(1 for r in results.values() if not r["passed"])
        total = len(results)

        dimensions = {
            "模型训练": [
                "model_plan", "model_exec", "model_skill_ref",
                "model_code_saved", "model_pkl_saved",
                "training_summary", "model_artifact_struct",
            ],
            "模型评估": [
                "eval_plan", "eval_exec",
            ],
            "Pipeline + 预测": [
                "pipeline_generate", "pipeline_p0_features",
                "predict_api", "predict_output_valid",
            ],
            "资产完整性": [
                "asset_completeness", "pipeline_script",
            ],
            "上下文压缩": [
                "llm_call_count", "token_monitor_active",
                "context_summarization", "workflow_state_history",
            ],
            "Skill 应用": [
                "skill_tool_calls", "skill_content_read",
                "new_skill_fe_patterns", "new_skill_model_select",
            ],
        }

        for dim_name, keys in dimensions.items():
            dim_results = {k: results[k] for k in keys if k in results}
            dim_pass = sum(1 for r in dim_results.values() if r["passed"])
            dim_total = len(dim_results)
            status = "✅" if dim_pass == dim_total else "⚠️"
            print(f"\n{status} {dim_name}: {dim_pass}/{dim_total}")
            for k, r in dim_results.items():
                icon = "  ✅" if r["passed"] else "  ❌"
                print(f"  {icon} {k}: {r['detail']}")

        ungrouped_keys = set()
        for keys in dimensions.values():
            ungrouped_keys.update(keys)
        ungrouped = {k: v for k, v in results.items() if k not in ungrouped_keys}
        if ungrouped:
            print(f"\n  其他:")
            for k, r in ungrouped.items():
                icon = "  ✅" if r["passed"] else "  ❌"
                print(f"  {icon} {k}: {r['detail']}")

        print(f"\n{'=' * 70}")
        print(f"总计: {passed}/{total} 通过, {failed} 失败")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
