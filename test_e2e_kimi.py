#!/usr/bin/env python3
"""
完整建模流程 E2E 测试 — 使用 Kimi 2.5 (Moonshot)

验证维度：
1. 上下文压缩后关键内容是否正确提取
2. 是否正确应用 skill（含新增的 feature-engineering-patterns / model-selection-heuristics）
3. 是否正确保存模型资产
4. 各个环节运行结果是否正常
"""

import json
import os
import sys
import subprocess
import threading
import time

import requests

# ─── 配置 ───
BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
SESSION_ID = f"e2e_kimi_{int(time.time())}"
DATA_PATH = os.path.abspath("train.csv")
TARGET = "SalePrice"
TASK_TYPE = "regression"
MODEL = "kimi-k2.5"
TIMEOUT = 10800  # Kimi 2.5 响应较慢，给足超时 (3h)

AUTO_START_BACKEND = True  # 自动启动后端
_backend_process = None

# ─── 结果收集 ───
results = {}


def record(name: str, passed: bool, detail: str = ""):
    results[name] = {"passed": passed, "detail": detail}
    icon = "✅" if passed else "❌"
    print(f"  {icon} [{name}] {detail}")


# ─── 后端管理 ───
def _stream_logs(pipe):
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            print(f"  [BACKEND] {line.rstrip()}")
    except Exception:
        pass


def ensure_backend():
    global _backend_process
    try:
        requests.get(f"{BASE_URL}/", timeout=3)
        print("✅ 后端已运行\n")
        return True
    except Exception:
        pass
    if not AUTO_START_BACKEND:
        print("❌ 后端未运行")
        return False
    print("启动后端...")
    _backend_process = subprocess.Popen(
        [sys.executable, "-u", "-m", "uvicorn",
         "automl_react.api.main:app",
         "--host", "0.0.0.0",
         "--port", BASE_URL.rsplit(":", 1)[-1]],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    threading.Thread(target=_stream_logs, args=(_backend_process.stdout,), daemon=True).start()
    for _ in range(30):
        try:
            requests.get(f"{BASE_URL}/", timeout=2)
            print("✅ 后端启动成功\n")
            return True
        except Exception:
            time.sleep(1)
    print("❌ 后端启动超时")
    return False


def shutdown_backend():
    global _backend_process
    if _backend_process and _backend_process.poll() is None:
        _backend_process.terminate()
        try:
            _backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _backend_process.kill()
    _backend_process = None


# ─── API 工具 ───
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
# Phase 0: 前置阶段
# ═══════════════════════════════════════════════════
def phase0_prerequisites():
    print("=" * 70)
    print("Phase 0: 前置阶段 (start → problem_definition → contract → splitting)")
    print("=" * 70)

    # 启动
    result = api("POST", "/workflow/start", {
        "session_id": SESSION_ID,
        "data_path": DATA_PATH,
        "target_column": TARGET,
        "task_type": TASK_TYPE,
        "model": MODEL,
        "task_description": "房价预测回归任务，使用 Kaggle House Prices 数据集。",
    }, "启动工作流")
    if not result or not result.get("success"):
        record("workflow_start", False, f"启动失败: {result}")
        return False
    record("workflow_start", True, f"session={SESSION_ID}, model={MODEL}")

    # 问题定义
    result = api("POST", f"/workflow/{SESSION_ID}/stage/problem_definition/run", label="问题定义")
    if not result or not result.get("success"):
        record("problem_definition", False, f"失败: {result}")
        return False
    conf_id = result.get("confirmation_id", "")
    record("problem_definition", True, f"proposal={len(result.get('proposal', ''))} chars")
    if conf_id:
        confirm(conf_id)

    # 数据契约检查
    result = api("POST", f"/workflow/{SESSION_ID}/stage/data_contract_check/run", label="数据契约检查")
    if not result or not result.get("success"):
        record("data_contract_check", False, f"失败: {result}")
        return False
    conf_id = result.get("confirmation_id", "")
    record("data_contract_check", True, "通过")
    if conf_id:
        confirm(conf_id)

    # 数据集切分
    result = api("POST", f"/workflow/{SESSION_ID}/stage/data_splitting/run", label="数据集切分")
    if not result or not result.get("success"):
        record("data_splitting", False, f"失败: {result}")
        return False
    conf_id = result.get("confirmation_id", "")
    record("data_splitting", True, "通过")
    if conf_id:
        confirm(conf_id)

    print("  ✅ 前置阶段全部完成\n")
    return True


# ═══════════════════════════════════════════════════
# Phase 1: 数据清洗
# ═══════════════════════════════════════════════════
def phase1_data_cleaning():
    print("\n" + "=" * 70)
    print("Phase 1: 数据清洗 — Skill 引用 + 执行验证")
    print("=" * 70)

    result = api("POST", f"/workflow/{SESSION_ID}/stage/data_cleaning/run", label="生成清洗方案")
    if not result or not result.get("success"):
        record("cleaning_plan", False, f"方案生成失败: {result}")
        return False

    proposal = result.get("proposal", "")
    conf_id = result.get("confirmation_id", "")
    skills_ref = result.get("skills_referenced", [])
    skill_names = [s.get("name", "") for s in skills_ref] if skills_ref else []

    record("cleaning_plan", len(proposal) > 50, f"方案={len(proposal)} chars")
    record("cleaning_skill_ref", bool(skills_ref),
           f"skills_referenced={skill_names}")

    # 确认并执行
    exec_result = confirm(conf_id)
    if not exec_result:
        record("cleaning_exec", False, "确认执行失败")
        return False

    execution = exec_result.get("execution", {})
    cleaned_path = execution.get("cleaned_data_path", "")
    success = execution.get("success", False) and bool(cleaned_path) and os.path.exists(cleaned_path)
    record("cleaning_exec", success,
           f"success={execution.get('success')}, path_exists={os.path.exists(cleaned_path) if cleaned_path else False}")
    return True


# ═══════════════════════════════════════════════════
# Phase 2: 数据探索
# ═══════════════════════════════════════════════════
def phase2_exploration():
    print("\n" + "=" * 70)
    print("Phase 2: 数据探索性分析")
    print("=" * 70)

    result = api("POST", f"/workflow/{SESSION_ID}/stage/data_exploration/run", label="数据探索")
    if not result or not result.get("success"):
        record("exploration", False, f"失败: {result}")
        return False
    record("exploration", True,
           f"报告={len(result.get('exploration', result.get('report', '')))} chars")
    return True


# ═══════════════════════════════════════════════════
# Phase 3: 特征工程 — 重点验证新增 skill
# ═══════════════════════════════════════════════════
def phase3_feature_engineering():
    print("\n" + "=" * 70)
    print("Phase 3: 特征工程 — Skill 引用 + 执行验证")
    print("=" * 70)

    result = api("POST", f"/workflow/{SESSION_ID}/stage/feature_engineering/run", label="生成特征方案")
    if not result or not result.get("success"):
        record("feature_plan", False, f"方案生成失败: {result}")
        return False

    proposal = result.get("proposal", "")
    conf_id = result.get("confirmation_id", "")
    skills_ref = result.get("skills_referenced", [])
    skill_names = [s.get("name", "") for s in skills_ref] if skills_ref else []

    record("feature_plan", len(proposal) > 50, f"方案={len(proposal)} chars")
    record("feature_skill_ref", bool(skills_ref), f"skills_referenced={skill_names}")

    # 确认并执行
    exec_result = confirm(conf_id)
    if not exec_result:
        record("feature_exec", False, "确认执行失败")
        return False

    execution = exec_result.get("execution", {})
    features_path = execution.get("features_data_path", "")
    record("feature_exec", execution.get("success", False),
           f"features_path={os.path.basename(features_path or 'N/A')}")

    # 处理 feature_evaluation 子确认
    next_conf = exec_result.get("next_confirmation")
    if next_conf and next_conf.get("stage") == "feature_evaluation":
        eval_result = confirm(next_conf["confirmation_id"])
        if eval_result:
            eval_exec = eval_result.get("execution", {})
            record("feature_eval", eval_exec.get("success", False),
                   f"metrics={os.path.basename(eval_exec.get('metrics_result_path', 'N/A'))}")
        else:
            record("feature_eval", False, "确认执行失败")
    else:
        record("feature_eval", False, "未返回 feature_evaluation 确认点")

    return True


# ═══════════════════════════════════════════════════
# Phase 4: 模型训练 — 重点验证 skill + 资产
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
    code_dir = os.path.join(session_dir(), "code")
    code_file = os.path.join(code_dir, "model_training.py")
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

    # 生成 pipeline.py
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
        # 验证 P0 改动：train_path 支持 + schema 验证
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
    records = result.get("records", 0)
    record("predict_api", success and records > 0,
           f"success={success}, records={records}, path={os.path.basename(predictions_path or 'N/A')}")

    # 验证 predictions.csv 内容
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

    # pipeline.py 检查
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

    # 统计各阶段 LLM 调用次数和 token 信息
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

                # 检查 metadata 中的 token 信息
                meta = entry.get("metadata", {})
                if meta.get("input_tokens") or meta.get("total_tokens"):
                    token_info_found = True

                # 检查 summarization 信息
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

    # 检查 workflow_state 中的 memory 信息
    state_file = os.path.join(session_dir(), "state", "workflow_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        # 检查是否有 stage history（间接反映上下文管理在工作）
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

                # 检查 skill 工具调用
                if any(kw in content for kw in ["read_skill", "search_skills"]):
                    skill_tool_calls.append(stage)

                # 检查是否读取了 skill 实际内容
                if "以下内容为技术参考" in content:
                    skill_content_reads.append(stage)

                # 检查新增技能包是否被发现/引用
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
    if not ensure_backend():
        sys.exit(1)

    print(f"\n{'#' * 70}")
    print(f"# E2E 完整建模流程测试 — Kimi 2.5 (Moonshot)")
    print(f"# Session: {SESSION_ID}")
    print(f"# Data:    {DATA_PATH}")
    print(f"# Model:   {MODEL}")
    print(f"{'#' * 70}\n")

    try:
        if not phase0_prerequisites():
            print("\n❌ 前置阶段失败，终止")
            return

        phase1_data_cleaning()
        phase2_exploration()
        phase3_feature_engineering()
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
        print("测试结果总结")
        print("=" * 70)

        passed = sum(1 for r in results.values() if r["passed"])
        failed = sum(1 for r in results.values() if not r["passed"])
        total = len(results)

        # 按维度分组
        dimensions = {
            "流程运行": [
                "workflow_start", "problem_definition", "data_contract_check",
                "data_splitting", "cleaning_plan", "cleaning_exec",
                "exploration", "feature_plan", "feature_exec", "feature_eval",
                "model_plan", "model_exec", "eval_plan", "eval_exec",
            ],
            "Skill 应用": [
                "cleaning_skill_ref", "feature_skill_ref", "model_skill_ref",
                "skill_tool_calls", "skill_content_read",
                "new_skill_fe_patterns", "new_skill_model_select",
            ],
            "模型资产": [
                "model_code_saved", "model_pkl_saved",
                "training_summary", "model_artifact_struct",
                "asset_completeness", "pipeline_script",
                "pipeline_generate", "pipeline_p0_features",
            ],
            "生产预测": [
                "predict_api", "predict_output_valid",
            ],
            "上下文压缩": [
                "llm_call_count", "token_monitor_active",
                "context_summarization", "workflow_state_history",
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

        # 未分组的
        grouped_keys = set()
        for keys in dimensions.values():
            grouped_keys.update(keys)
        ungrouped = {k: v for k, v in results.items() if k not in grouped_keys}
        if ungrouped:
            print(f"\n  其他:")
            for k, r in ungrouped.items():
                icon = "  ✅" if r["passed"] else "  ❌"
                print(f"  {icon} {k}: {r['detail']}")

        print(f"\n{'=' * 70}")
        print(f"总计: {passed}/{total} 通过, {failed} 失败")
        print(f"{'=' * 70}")

        if AUTO_START_BACKEND:
            shutdown_backend()


if __name__ == "__main__":
    main()
