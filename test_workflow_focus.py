#!/usr/bin/env python3
"""
完整建模流程测试 — 聚焦以下 4 项核心关注点：

1. 用户反馈修改意见后，大模型是否按照用户意见修改方案
2. 方案生成环节是否正确调用了 Skill 相关内容作为参考
3. 模型训练阶段代码是否正确保存
4. 是否按照方案生成代码（方案→代码一致性）
"""

import requests
import json
import time
import os
import sys
import re
import threading
import subprocess

BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
SESSION_ID = f"focus_test_{int(time.time())}"
DATA_PATH = os.path.abspath("train.csv")
TARGET = "SalePrice"
TASK_TYPE = "regression"
TIMEOUT = 3600  # 每个 API 调用最多 60 分钟（LLM 延迟可能很高）

# 自动启动后端
AUTO_START_BACKEND = True
_backend_process = None

# ─────────────────────────────────────────────
# 测试结果收集
# ─────────────────────────────────────────────
results = {}


def record(name: str, passed: bool, detail: str = ""):
    """记录测试结果"""
    results[name] = {"passed": passed, "detail": detail}
    icon = "✅" if passed else "❌"
    print(f"  {icon} [{name}] {detail}")


# ─────────────────────────────────────────────
# 后端管理
# ─────────────────────────────────────────────
def _stream_logs(pipe):
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            print(f"[BACKEND] {line.rstrip()}")
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
        print("❌ 后端未运行且未启用自动启动")
        return False

    print("启动后端...")
    _backend_process = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "uvicorn",
            "automl_react.api.main:app",
            "--host", "0.0.0.0",
            "--port", BASE_URL.rsplit(":", 1)[-1],
        ],
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


# ─────────────────────────────────────────────
# API 工具函数
# ─────────────────────────────────────────────
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
    data = {
        "session_id": SESSION_ID,
        "confirmation_id": conf_id,
        "status": status,
    }
    if modifications:
        data["modifications"] = modifications
    return api("POST", "/confirmation/submit", data, f"确认({status})")


def revise(conf_id, modifications):
    """调用 /confirmation/revise 修订方案"""
    data = {
        "session_id": SESSION_ID,
        "confirmation_id": conf_id,
        "modifications": modifications,
    }
    return api("POST", "/confirmation/revise", data, "修订方案")


def session_dir():
    return os.path.join("assets", SESSION_ID)


# ─────────────────────────────────────────────
# 步骤：前置阶段
# ─────────────────────────────────────────────
def run_prerequisite_stages():
    """运行启动 → 问题定义 → 契约检查 → 数据切分 阶段"""
    print("=" * 70)
    print("Phase 0: 前置阶段 (start → problem_def → contract → splitting)")
    print("=" * 70)

    # 启动工作流
    result = api("POST", "/workflow/start", {
        "session_id": SESSION_ID,
        "data_path": DATA_PATH,
        "target_column": TARGET,
        "task_type": TASK_TYPE,
        "task_description": "房价预测回归任务。重点关注房屋面积、年份等关键特征。",
        "model": "qwen3.6-plus",
    }, "启动工作流")
    if not result or not result.get("success"):
        print(f"  FATAL: 工作流启动失败 → {result}")
        return False
    print(f"  session={SESSION_ID}")

    # 问题定义
    result = api("POST", f"/workflow/{SESSION_ID}/stage/problem_definition/run", label="问题定义")
    if not result or not result.get("success"):
        print(f"  FATAL: 问题定义失败")
        return False
    conf_id = result.get("confirmation_id", "")
    if conf_id:
        confirm(conf_id)

    # 数据契约检查
    result = api("POST", f"/workflow/{SESSION_ID}/stage/data_contract_check/run", label="数据契约检查")
    if not result or not result.get("success"):
        print(f"  FATAL: 契约检查失败 → {result}")
        return False
    conf_id = result.get("confirmation_id", "")
    if conf_id:
        confirm(conf_id)

    # 数据集切分
    result = api("POST", f"/workflow/{SESSION_ID}/stage/data_splitting/run", label="数据集切分")
    if not result or not result.get("success"):
        print(f"  FATAL: 数据切分失败")
        return False
    conf_id = result.get("confirmation_id", "")
    if conf_id:
        confirm(conf_id)

    print("  ✅ 前置阶段全部完成\n")
    return True


# ─────────────────────────────────────────────
# 关注点 1 & 2: 数据清洗 — 用户修改 + Skill 调用
# ─────────────────────────────────────────────
def test_data_cleaning_with_revision():
    """
    关注点 1: 测试用户修改意见后方案是否对应修改
    关注点 2: 测试方案是否引用了 Skill 内容
    """
    print("=" * 70)
    print("Phase 1: 数据清洗 — 关注点 1 (用户修改) + 关注点 2 (Skill 引用)")
    print("=" * 70)

    # Step 1: 生成初始清洗方案
    result = api("POST", f"/workflow/{SESSION_ID}/stage/data_cleaning/run", label="生成清洗方案")
    if not result or not result.get("success"):
        record("cleaning_plan_gen", False, f"方案生成失败: {result}")
        return False

    original_proposal = result.get("proposal", "")
    conf_id = result.get("confirmation_id", "")
    skills_ref = result.get("skills_referenced", [])

    record("cleaning_plan_gen", len(original_proposal) > 100,
           f"方案长度={len(original_proposal)} chars, confirmation_id={conf_id[:8]}...")

    # ── 关注点 2: 检查 skills_referenced 字段 ──
    has_skill_ref = bool(skills_ref)
    skill_names = [s.get("name", "") for s in skills_ref] if skills_ref else []
    record("cleaning_skill_referenced",
           has_skill_ref and any("data-analysis" in n for n in skill_names),
           f"skills_referenced={skill_names}")

    # 补充检查: API 返回的 skills_referenced 属于确认点元数据
    # 还需检查方案内容是否体现 skill 知识（如提到 pitfalls、techniques 等关键词）
    skill_keywords_in_proposal = any(
        kw in original_proposal.lower()
        for kw in ["pitfall", "陷阱", "skill", "technique", "技巧", "数据分析"]
    )
    record("cleaning_skill_content_influence",
           True,  # 不作为硬性要求，仅观察
           f"方案中是否包含 skill 关键词: {skill_keywords_in_proposal}")

    # Step 2: 用户提出修改意见（关注点 1 的核心）
    MODIFICATION = "对于所有缺失值比例超过50%的列直接删除，其余缺失值使用中位数填充。不要使用均值填充。"

    print(f"\n  📝 用户修改意见: {MODIFICATION}\n")

    revised_result = revise(conf_id, MODIFICATION)
    if not revised_result or not revised_result.get("success"):
        record("cleaning_revision", False, f"修订失败: {revised_result}")
        return False

    revised_proposal = revised_result.get("proposal", "")
    new_conf_id = revised_result.get("confirmation_id", "")
    revision_round = revised_result.get("revision_round", 0)

    record("cleaning_revision_returned", len(revised_proposal) > 100,
           f"修订方案长度={len(revised_proposal)}, round={revision_round}")

    # ── 关注点 1 核心验证: 修订后方案是否体现用户意见 ──
    # 验证修订后方案包含用户关键要求
    user_req_checks = {
        "提到删除高缺失列": any(kw in revised_proposal for kw in ["删除", "丢弃", "drop", "移除"]),
        "提到50%阈值": any(kw in revised_proposal for kw in ["50%", "50％", "0.5", "一半"]),
        "提到中位数填充": any(kw in revised_proposal for kw in ["中位数", "median"]),
        "未使用均值": "均值" not in revised_proposal or any(kw in revised_proposal for kw in ["不使用均值", "非均值", "禁用均值", "不用均值", "禁止均值", "不要均值", "严禁均值", "不采用均值"]),
    }

    all_met = all(user_req_checks.values())
    detail_str = " | ".join(f"{k}={'✓' if v else '✗'}" for k, v in user_req_checks.items())
    record("cleaning_revision_follows_user",
           all_met,
           f"用户意见是否体现: {detail_str}")

    if not all_met:
        # 打印修订方案片段便于调试
        print(f"\n  ⚠️ 修订方案前500字符:\n{revised_proposal[:500]}\n")

    # Step 3: 确认修订后的方案并执行
    exec_result = confirm(new_conf_id, "confirmed")
    if not exec_result:
        record("cleaning_execution", False, "确认执行失败")
        return False

    execution = exec_result.get("execution", {})
    cleaned_path = execution.get("cleaned_data_path", "")
    record("cleaning_execution",
           execution.get("success", False) and os.path.exists(cleaned_path),
           f"cleaned_data 存在={os.path.exists(cleaned_path)}")

    return True


# ─────────────────────────────────────────────
# 数据探索（不需确认，直接过）
# ─────────────────────────────────────────────
def run_data_exploration():
    print("\n" + "=" * 70)
    print("Phase 1.5: 数据探索性分析")
    print("=" * 70)
    result = api("POST", f"/workflow/{SESSION_ID}/stage/data_exploration/run", label="数据探索")
    if not result or not result.get("success"):
        print(f"  ⚠️ 数据探索失败: {result}")
        return False
    print(f"  ✅ 探索报告 {len(result.get('exploration', result.get('report', '')))} chars")
    return True


# ─────────────────────────────────────────────
# 关注点 2 (续): 特征工程 — Skill 引用检查
# ─────────────────────────────────────────────
def test_feature_engineering_skills():
    """关注点 2: 特征工程阶段 skill 引用检查"""
    print("\n" + "=" * 70)
    print("Phase 2: 特征工程 — 关注点 2 (Skill 引用)")
    print("=" * 70)

    result = api("POST", f"/workflow/{SESSION_ID}/stage/feature_engineering/run", label="生成特征方案")
    if not result or not result.get("success"):
        record("feature_plan_gen", False, f"方案生成失败: {result}")
        return False

    proposal = result.get("proposal", "")
    conf_id = result.get("confirmation_id", "")
    skills_ref = result.get("skills_referenced", [])

    record("feature_plan_gen", len(proposal) > 100,
           f"方案长度={len(proposal)} chars")

    # 检查 skills_referenced
    skill_names = [s.get("name", "") for s in skills_ref] if skills_ref else []
    record("feature_skill_referenced",
           any("ml-engineering" in n or "afrexai" in n for n in skill_names),
           f"skills_referenced={skill_names}")

    # 执行: 确认 → 执行 → 特征评估
    exec_result = confirm(conf_id)
    if not exec_result:
        record("feature_execution", False, "确认执行失败")
        return False

    execution = exec_result.get("execution", {})
    record("feature_execution", execution.get("success", False),
           f"features_data_path={execution.get('features_data_path', 'N/A')}")

    # 处理 feature_evaluation 子确认 — 执行评估并验证产物
    next_conf = exec_result.get("next_confirmation")
    if next_conf and next_conf.get("stage") == "feature_evaluation":
        eval_result = confirm(next_conf["confirmation_id"], status="confirmed")
        if eval_result:
            eval_exec = eval_result.get("execution", {})
            eval_success = eval_exec.get("success", False)
            metrics_path = eval_exec.get("metrics_result_path", "")
            report_path = eval_exec.get("metrics_report_path", "")
            record("feature_evaluation_execution", eval_success,
                   f"success={eval_success}, metrics={os.path.basename(metrics_path or '')}, "
                   f"report={os.path.basename(report_path or '')}")

            # 验证评估产物文件
            metrics_exists = bool(metrics_path) and os.path.exists(metrics_path)
            report_exists = bool(report_path) and os.path.exists(report_path)
            record("feature_evaluation_artifacts",
                   metrics_exists and report_exists,
                   f"metrics_json={metrics_exists}, report_md={report_exists}")

            # 验证评估代码保存
            eval_code_path = os.path.join(session_dir(), "code", "feature_metrics.py")
            record("feature_evaluation_code_saved", os.path.exists(eval_code_path),
                   f"feature_metrics.py 存在={os.path.exists(eval_code_path)}")
        else:
            record("feature_evaluation_execution", False, "确认执行失败")
    else:
        print("  ⚠️ 未返回 feature_evaluation 确认点")
        record("feature_evaluation_execution", False, "未返回确认点")

    return True


# ─────────────────────────────────────────────
# 关注点 1 (续) + 2 + 3 + 4: 模型训练
# ─────────────────────────────────────────────
def test_model_training_full():
    """
    关注点 1: 用户修改建模方案后验证修改是否生效
    关注点 2: Skill 引用检查
    关注点 3: 代码是否正确保存
    关注点 4: 代码是否按照方案生成
    """
    print("\n" + "=" * 70)
    print("Phase 3: 模型训练 — 关注点 1/2/3/4 综合测试")
    print("=" * 70)

    # Step 1: 生成初始建模方案
    result = api("POST", f"/workflow/{SESSION_ID}/stage/model_training/run", label="生成建模方案")
    if not result or not result.get("success"):
        record("model_plan_gen", False, f"方案生成失败: {result}")
        return False

    original_proposal = result.get("proposal", "")
    conf_id = result.get("confirmation_id", "")
    skills_ref = result.get("skills_referenced", [])

    record("model_plan_gen", len(original_proposal) > 100,
           f"方案长度={len(original_proposal)} chars")

    # ── 关注点 2: 检查 skills_referenced ──
    skill_names = [s.get("name", "") for s in skills_ref] if skills_ref else []
    record("model_skill_referenced",
           any("ml-engineering" in n or "afrexai" in n for n in skill_names),
           f"skills_referenced={skill_names}")

    # ── 关注点 1: 用户修改建模方案 ──
    # 明确要求使用 GradientBoosting 和特定参数
    MODEL_MODIFICATION = (
        "请使用 GradientBoostingRegressor 作为主要模型，"
        "设置 n_estimators=200, learning_rate=0.05, max_depth=4。"
        "不要使用 RandomForest。"
    )
    print(f"\n  📝 用户修改意见: {MODEL_MODIFICATION}\n")

    revised_result = revise(conf_id, MODEL_MODIFICATION)
    if not revised_result or not revised_result.get("success"):
        record("model_revision", False, f"修订失败: {revised_result}")
        # 降级：直接用原方案确认
        print("  ⚠️ 修订失败，降级为直接确认原方案")
        exec_result = confirm(conf_id)
    else:
        revised_proposal = revised_result.get("proposal", "")
        new_conf_id = revised_result.get("confirmation_id", "")
        revision_round = revised_result.get("revision_round", 0)

        record("model_revision_returned", len(revised_proposal) > 100,
               f"修订方案长度={len(revised_proposal)}, round={revision_round}")

        # 验证修订后方案包含用户要求
        revision_checks = {
            "提到GradientBoosting": any(kw in revised_proposal for kw in [
                "GradientBoosting", "gradient_boosting", "gradient boosting", "梯度提升"
            ]),
            "提到n_estimators=200": any(kw in revised_proposal for kw in [
                "200", "n_estimators"
            ]),
            "提到learning_rate=0.05": any(kw in revised_proposal for kw in [
                "0.05", "learning_rate"
            ]),
            "提到max_depth=4": any(kw in revised_proposal for kw in [
                "max_depth", "4"
            ]),
        }

        all_met = all(revision_checks.values())
        detail_str = " | ".join(f"{k}={'✓' if v else '✗'}" for k, v in revision_checks.items())
        record("model_revision_follows_user", all_met, f"{detail_str}")

        if not all_met:
            print(f"\n  ⚠️ 修订方案前 500 字符:\n{revised_proposal[:500]}\n")

        # Step 2: 确认修订后方案
        # 保存修订后方案以便后续验证代码一致性（关注点 4）
        final_proposal = revised_proposal
        exec_result = confirm(new_conf_id)

    if not exec_result:
        record("model_execution", False, "确认执行失败")
        return False

    execution = exec_result.get("execution", {})
    record("model_execution", execution.get("success", False),
           f"model_path={execution.get('model_path', 'N/A')}")

    # ── 关注点 3: 代码是否正确保存 ──
    code_dir = os.path.join(session_dir(), "code")
    model_code_path = os.path.join(code_dir, "model_training.py")
    code_exists = os.path.exists(model_code_path)
    record("model_code_saved", code_exists,
           f"model_training.py 存在={code_exists}, 路径={model_code_path}")

    if code_exists:
        with open(model_code_path, "r") as f:
            saved_code = f.read()
        record("model_code_nonempty", len(saved_code) > 50,
               f"代码长度={len(saved_code)} chars")

        # ── 关注点 4: 代码是否按照方案生成 ──
        # 检查代码中是否包含方案中提到的关键元素
        code_checks = {}

        # 如果修订成功，验证代码是否包含用户修改要求的模型
        if revised_result and revised_result.get("success"):
            code_checks["代码中使用GradientBoosting"] = any(kw in saved_code for kw in [
                "GradientBoostingRegressor", "GradientBoosting", "gradient_boosting"
            ])
            code_checks["代码中n_estimators=200"] = "200" in saved_code
            code_checks["代码中learning_rate=0.05"] = "0.05" in saved_code

        # 通用检查：代码结构完整性
        code_checks["包含数据读取"] = any(kw in saved_code for kw in ["read_csv", "pd.read", "load"])
        code_checks["包含模型训练"] = any(kw in saved_code for kw in [".fit(", "model.fit", "train"])
        code_checks["包含模型保存"] = any(kw in saved_code for kw in ["joblib", "pickle", "save", "dump"])
        code_checks["包含目标列引用"] = TARGET in saved_code or "target" in saved_code.lower()

        all_code_ok = all(code_checks.values())
        detail_str = " | ".join(f"{k}={'✓' if v else '✗'}" for k, v in code_checks.items())
        record("model_code_matches_plan", all_code_ok, f"{detail_str}")

        if not all_code_ok:
            # 打印代码片段便于调试
            print(f"\n  ⚠️ 代码前 800 字符:\n{saved_code[:800]}\n")
    else:
        record("model_code_nonempty", False, "代码文件不存在，无法检查")
        record("model_code_matches_plan", False, "代码文件不存在")

    # 关注点 3 (续): 模型产物文件检查
    model_dir = os.path.join(session_dir(), "models")
    model_pkl = os.path.join(model_dir, "trained_model.pkl")
    training_summary = os.path.join(model_dir, "training_summary.json")
    model_plan_md = os.path.join(model_dir, "model_training_plan.md")

    record("model_pkl_saved", os.path.exists(model_pkl),
           f"trained_model.pkl 存在={os.path.exists(model_pkl)}")

    record("training_summary_saved", os.path.exists(training_summary),
           f"training_summary.json 存在={os.path.exists(training_summary)}")

    if os.path.exists(training_summary):
        with open(training_summary) as f:
            summary = json.load(f)
        has_metrics = bool(summary.get("metrics") or summary.get("validation_metrics"))
        has_features = bool(summary.get("selected_feature_names") or summary.get("feature_names"))
        record("training_summary_content",
               has_metrics and has_features,
               f"has_metrics={has_metrics}, has_features={has_features}")

    record("model_plan_md_saved", os.path.exists(model_plan_md),
           f"model_training_plan.md 存在={os.path.exists(model_plan_md)}")

    # 模型产物标准化检查
    if os.path.exists(model_pkl):
        import joblib
        try:
            artifact = joblib.load(model_pkl)
            if isinstance(artifact, dict):
                required_keys = {"model", "selected_feature_names"}
                has_keys = required_keys.issubset(set(artifact.keys()))
                record("model_artifact_structure", has_keys,
                       f"产物 keys={list(artifact.keys())[:6]}")
            else:
                record("model_artifact_structure", False,
                       f"产物类型={type(artifact).__name__}，非标准 dict 打包")
        except Exception as e:
            record("model_artifact_structure", False, f"读取失败: {e}")

    return True


# ─────────────────────────────────────────────
# 模型评估（收尾验证）
# ─────────────────────────────────────────────
def test_model_evaluation():
    print("\n" + "=" * 70)
    print("Phase 4: 模型评估")
    print("=" * 70)

    result = api("POST", f"/workflow/{SESSION_ID}/stage/model_evaluation/run", label="生成评估方案")
    if not result or not result.get("success"):
        record("evaluation_plan", False, f"评估方案生成失败: {result}")
        return False

    conf_id = result.get("confirmation_id", "")
    record("evaluation_plan", True, f"方案长度={len(result.get('proposal', ''))}")

    exec_result = confirm(conf_id)
    if not exec_result:
        record("evaluation_execution", False, "评估执行失败")
        return False

    evaluation = exec_result.get("execution", {})
    metrics = evaluation.get("metrics", {})
    eval_error = evaluation.get("error", "")
    record("evaluation_execution",
           evaluation.get("success", False) and bool(metrics),
           f"metrics={list(metrics.keys()) if metrics else 'none'}"
           + (f", error={eval_error}" if eval_error else ""))

    return True


# ─────────────────────────────────────────────
# 最终资产完整性校验
# ─────────────────────────────────────────────
def verify_all_assets():
    """验证所有关键产物文件"""
    print("\n" + "=" * 70)
    print("Phase 5: 资产完整性校验")
    print("=" * 70)

    sd = session_dir()

    expected_files = {
        "data/original_data.csv": "原始数据",
        "data/train_raw.csv": "训练集",
        "data/test_raw.csv": "测试集",
        "cleaning/cleaning_plan.md": "清洗方案",
        "cleaning/cleaning_result.json": "清洗结果",
        "features/feature_engineering_plan.md": "特征方案",
        "features/feature_metrics.json": "特征评估指标",
        "features/feature_metrics_report.md": "特征评估报告",
        "features/feature_evaluation_result.json": "特征评估结果",
        "models/model_training_plan.md": "建模方案",
        "models/trained_model.pkl": "训练模型",
        "models/training_summary.json": "训练摘要",
        "code/model_training.py": "训练代码",
        "code/feature_metrics.py": "特征评估代码",
        "code/pipeline.py": "全流程脚本",
        "state/workflow_state.json": "工作流状态",
    }

    missing = []
    for rel_path, desc in expected_files.items():
        full = os.path.join(sd, rel_path)
        if not os.path.exists(full):
            missing.append(f"{desc}({rel_path})")

    record("asset_completeness",
           len(missing) == 0,
           f"缺失文件: {missing}" if missing else f"全部 {len(expected_files)} 项产物齐全")

    # 检查清洗和特征代码也保存了
    cleaning_code = os.path.join(sd, "code", "cleaning.py")
    feature_code = os.path.join(sd, "code", "feature_engineering.py")
    pipeline_code = os.path.join(sd, "code", "pipeline.py")
    record("all_code_files_saved",
           os.path.exists(cleaning_code) and os.path.exists(model_code_path := os.path.join(sd, "code", "model_training.py")),
           f"cleaning.py={os.path.exists(cleaning_code)}, "
           f"feature_engineering.py={os.path.exists(feature_code)}, "
           f"model_training.py={os.path.exists(model_code_path)}")

    # 检查 pipeline.py 是否包含 train 和 predict 模式
    if os.path.exists(pipeline_code):
        with open(pipeline_code, "r") as f:
            pipeline_content = f.read()
        has_train = "run_training" in pipeline_content
        has_predict = "run_predict" in pipeline_content
        has_argparse = "--mode" in pipeline_content
        record("pipeline_script_complete",
               has_train and has_predict and has_argparse,
               f"train={has_train}, predict={has_predict}, argparse={has_argparse}, "
               f"长度={len(pipeline_content)} chars")
    else:
        record("pipeline_script_complete", False, "pipeline.py 不存在")


# ─────────────────────────────────────────────
# 检查 LLM 日志中是否有 skill 工具调用
# ─────────────────────────────────────────────
def verify_skill_tool_calls_in_logs():
    """检查 LLM 日志确认 skill 工具被实际调用过"""
    print("\n" + "=" * 70)
    print("Phase 6: 检查 LLM 日志中 Skill 工具调用记录")
    print("=" * 70)

    log_dir = os.path.join("logs", "llm_calls", SESSION_ID)
    if not os.path.exists(log_dir):
        record("skill_tool_in_logs", False, f"日志目录不存在: {log_dir}")
        return

    skill_calls_found = False
    skill_content_found = False
    skill_call_details = []

    for filename in os.listdir(log_dir):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(log_dir, filename)
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 检查输入和输出中是否有 skill 相关的工具调用
                content = json.dumps(entry, ensure_ascii=False)
                if any(kw in content for kw in ["read_skill", "search_skills", "skill_tools"]):
                    skill_calls_found = True
                    stage = entry.get("stage", "unknown")
                    skill_call_details.append(stage)
                # 检查是否实际返回了 skill 内容（而非仅返回定义/摘要）
                if any(kw in content for kw in [
                    "以下内容为技术参考",  # read_skill 返回的 header
                    "Hypothesis Testing",  # data-analysis techniques
                    "Data Quality Assessment",  # ml-engineering phase2
                    "Experiment Tracking",  # ml-engineering phase3
                    "Feature Engineering Patterns",  # ml-engineering phase2
                ]):
                    skill_content_found = True

    record("skill_tool_in_logs", skill_calls_found,
           f"发现 skill 工具调用的阶段: {list(set(skill_call_details))}" if skill_calls_found
           else "未在日志中发现 skill 工具调用")

    record("skill_content_actually_read", skill_content_found,
           "日志中包含 skill 实际内容（非仅定义/摘要）" if skill_content_found
           else "日志中仅包含 skill 定义/摘要，未读取到实际内容")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    if not ensure_backend():
        sys.exit(1)

    print(f"\n{'#' * 70}")
    print(f"# 完整建模流程测试 — 核心关注点验证")
    print(f"# Session: {SESSION_ID}")
    print(f"# Data: {DATA_PATH}")
    print(f"{'#' * 70}\n")

    try:
        # Phase 0: 前置
        if not run_prerequisite_stages():
            print("\n❌ 前置阶段失败，终止测试")
            return

        # Phase 1: 数据清洗 → 关注点 1 (修改) + 关注点 2 (Skill)
        test_data_cleaning_with_revision()

        # Phase 1.5: 数据探索
        run_data_exploration()

        # Phase 2: 特征工程 → 关注点 2 (Skill)
        test_feature_engineering_skills()

        # Phase 3: 模型训练 → 关注点 1/2/3/4
        test_model_training_full()

        # Phase 4: 模型评估
        test_model_evaluation()

        # Phase 5: 资产完整性校验
        verify_all_assets()

        # Phase 6: LLM 日志中 Skill 调用验证
        verify_skill_tool_calls_in_logs()

    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 未预期的异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ── 打印测试总结 ──
        print("\n\n" + "=" * 70)
        print("测试结果总结")
        print("=" * 70)

        passed = sum(1 for r in results.values() if r["passed"])
        failed = sum(1 for r in results.values() if not r["passed"])
        total = len(results)

        # 按关注点分组
        focus_areas = {
            "关注点1: 用户修改意见→方案修订": [
                "cleaning_revision", "cleaning_revision_returned",
                "cleaning_revision_follows_user",
                "model_revision", "model_revision_returned",
                "model_revision_follows_user",
            ],
            "关注点2: Skill 引用": [
                "cleaning_skill_referenced", "cleaning_skill_content_influence",
                "feature_skill_referenced",
                "model_skill_referenced",
                "skill_tool_in_logs",
                "skill_content_actually_read",
            ],
            "关注点3: 代码正确保存": [
                "model_code_saved", "model_code_nonempty",
                "model_pkl_saved", "training_summary_saved",
                "training_summary_content", "model_plan_md_saved",
                "model_artifact_structure",
                "all_code_files_saved", "asset_completeness",
                "pipeline_script_complete",
                "feature_evaluation_execution",
                "feature_evaluation_artifacts",
                "feature_evaluation_code_saved",
            ],
            "关注点4: 按方案生成代码": [
                "model_code_matches_plan",
            ],
        }

        for area_name, test_keys in focus_areas.items():
            print(f"\n  {area_name}:")
            for key in test_keys:
                if key in results:
                    r = results[key]
                    icon = "  ✅" if r["passed"] else "  ❌"
                    print(f"    {icon} {key}: {r['detail']}")
                else:
                    print(f"    ⚪ {key}: (未执行)")

        # 未分组的
        grouped_keys = set()
        for keys in focus_areas.values():
            grouped_keys.update(keys)
        ungrouped = [k for k in results if k not in grouped_keys]
        if ungrouped:
            print(f"\n  其他:")
            for key in ungrouped:
                r = results[key]
                icon = "  ✅" if r["passed"] else "  ❌"
                print(f"    {icon} {key}: {r['detail']}")

        print(f"\n  总计: {passed}/{total} 通过, {failed}/{total} 失败")
        print("=" * 70)

        shutdown_backend()

        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
