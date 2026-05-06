#!/usr/bin/env python3
"""从已有 session 恢复测试：模型训练 → 模型评估 → 资产校验"""

import requests
import json
import time
import os
import sys
import threading
import subprocess

BASE_URL = "http://localhost:8000"
SESSION_ID = "focus_test_1776163003"
TARGET = "SalePrice"
TASK_TYPE = "regression"
TIMEOUT = 3600

_backend_process = None
results = {}


def record(name, passed, detail=""):
    results[name] = {"passed": passed, "detail": detail}
    icon = "✅" if passed else "❌"
    print(f"  {icon} [{name}] {detail}")


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

    print("启动后端...")
    _backend_process = subprocess.Popen(
        [sys.executable, "-u", "-m", "uvicorn",
         "automl_react.api.main:app",
         "--host", "0.0.0.0", "--port", "8000"],
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


def confirm(conf_id, status="confirmed"):
    return api("POST", "/confirmation/submit", {
        "session_id": SESSION_ID,
        "confirmation_id": conf_id,
        "status": status,
    }, f"确认({status})")


def revise(conf_id, modifications):
    return api("POST", "/confirmation/revise", {
        "session_id": SESSION_ID,
        "confirmation_id": conf_id,
        "modifications": modifications,
    }, "修订方案")


def session_dir():
    return os.path.join("assets", SESSION_ID)


# ─────────────────────────────────────────
# 恢复 session
# ─────────────────────────────────────────
def restore_session():
    print("=" * 70)
    print(f"恢复 Session: {SESSION_ID}")
    print("=" * 70)
    result = api("POST", f"/workflow/{SESSION_ID}/restore", label="恢复会话")
    if not result:
        # 尝试直接用 start 接口（带 session_id 复用）
        result = api("GET", f"/workflow/{SESSION_ID}/status", label="检查状态")
    if result:
        print(f"  当前阶段: {result.get('current_stage', result.get('stage', 'unknown'))}")
        return True
    print("  ⚠️ 恢复失败，尝试直接调用阶段 API")
    return True  # 继续尝试


# ─────────────────────────────────────────
# 模型训练
# ─────────────────────────────────────────
def test_model_training():
    print("\n" + "=" * 70)
    print("Phase 1: 模型训练")
    print("=" * 70)

    result = api("POST", f"/workflow/{SESSION_ID}/stage/model_training/run", label="生成建模方案")
    if not result or not result.get("success"):
        record("model_plan_gen", False, f"方案生成失败: {result}")
        return False

    proposal = result.get("proposal", "")
    conf_id = result.get("confirmation_id", "")
    record("model_plan_gen", len(proposal) > 100, f"方案长度={len(proposal)} chars")

    # 用户修改
    MODEL_MOD = "请使用 GradientBoostingRegressor 作为主要模型，设置 n_estimators=200, learning_rate=0.05, max_depth=4。不要使用 RandomForest。"
    print(f"\n  📝 用户修改: {MODEL_MOD}\n")

    revised = revise(conf_id, MODEL_MOD)
    if not revised or not revised.get("success"):
        record("model_revision", False, f"修订失败: {revised}")
        print("  ⚠️ 修订失败，降级确认原方案")
        exec_result = confirm(conf_id)
    else:
        revised_proposal = revised.get("proposal", "")
        new_conf_id = revised.get("confirmation_id", "")
        record("model_revision", True, f"修订方案长度={len(revised_proposal)}")

        checks = {
            "GradientBoosting": any(kw in revised_proposal for kw in ["GradientBoosting", "gradient_boosting", "梯度提升"]),
            "n_estimators=200": "200" in revised_proposal,
            "learning_rate=0.05": "0.05" in revised_proposal,
        }
        record("model_revision_follows_user", all(checks.values()),
               " | ".join(f"{k}={'✓' if v else '✗'}" for k, v in checks.items()))

        exec_result = confirm(new_conf_id)

    if not exec_result:
        record("model_execution", False, "确认执行失败")
        return False

    execution = exec_result.get("execution", {})
    record("model_execution", execution.get("success", False),
           f"model_path={execution.get('model_path', 'N/A')}")

    # 代码保存检查
    code_dir = os.path.join(session_dir(), "code")
    model_code = os.path.join(code_dir, "model_training.py")
    record("model_code_saved", os.path.exists(model_code),
           f"model_training.py={os.path.exists(model_code)}")

    model_pkl = os.path.join(session_dir(), "models", "trained_model.pkl")
    record("model_pkl_saved", os.path.exists(model_pkl),
           f"trained_model.pkl={os.path.exists(model_pkl)}")

    return True


# ─────────────────────────────────────────
# 模型评估
# ─────────────────────────────────────────
def test_model_evaluation():
    print("\n" + "=" * 70)
    print("Phase 2: 模型评估")
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

    # 评估代码保存检查
    eval_code_path = os.path.join(session_dir(), "code", "model_evaluation.py")
    record("evaluation_code_saved", os.path.exists(eval_code_path),
           f"model_evaluation.py={os.path.exists(eval_code_path)}")

    return True


# ─────────────────────────────────────────
# 资产完整性校验
# ─────────────────────────────────────────
def verify_assets():
    print("\n" + "=" * 70)
    print("Phase 3: 资产完整性校验")
    print("=" * 70)

    sd = session_dir()
    expected = {
        "code/cleaning.py": "清洗代码",
        "code/feature_engineering.py": "特征代码",
        "code/model_training.py": "训练代码",
        "code/model_evaluation.py": "评估代码",
        "code/pipeline.py": "全流程脚本",
        "models/trained_model.pkl": "训练模型",
        "models/training_summary.json": "训练摘要",
        "reports/evaluation.json": "评估结果",
    }

    missing = []
    for rel, desc in expected.items():
        if not os.path.exists(os.path.join(sd, rel)):
            missing.append(f"{desc}({rel})")

    record("asset_completeness",
           len(missing) == 0,
           f"缺失: {missing}" if missing else f"全部 {len(expected)} 项产物齐全")

    # pipeline.py 结构检查
    pipeline_path = os.path.join(sd, "code", "pipeline.py")
    if os.path.exists(pipeline_path):
        with open(pipeline_path) as f:
            content = f.read()
        has_train = "run_training" in content
        has_predict = "run_predict" in content
        has_argparse = "--mode" in content
        record("pipeline_complete",
               has_train and has_predict and has_argparse,
               f"train={has_train}, predict={has_predict}, argparse={has_argparse}, 长度={len(content)}")
    else:
        record("pipeline_complete", False, "pipeline.py 不存在")


# ─────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────
def main():
    if not ensure_backend():
        sys.exit(1)

    print(f"\n{'#' * 70}")
    print(f"# 恢复测试 — 从模型训练继续")
    print(f"# Session: {SESSION_ID}")
    print(f"{'#' * 70}\n")

    try:
        restore_session()
        test_model_training()
        test_model_evaluation()
        verify_assets()

    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被中断")
    except Exception as e:
        print(f"\n❌ 异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n\n" + "=" * 70)
        print("测试结果总结")
        print("=" * 70)

        passed = sum(1 for r in results.values() if r["passed"])
        failed = sum(1 for r in results.values() if not r["passed"])

        for name, r in results.items():
            icon = "✅" if r["passed"] else "❌"
            print(f"  {icon} {name}: {r['detail']}")

        print(f"\n  总计: {passed}/{len(results)} 通过, {failed}/{len(results)} 失败")
        print("=" * 70)

        shutdown_backend()
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
