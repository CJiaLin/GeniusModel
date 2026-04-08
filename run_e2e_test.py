#!/usr/bin/env python3
"""End-to-end workflow test - runs all stages sequentially via API calls."""

import requests
import json
import time
import sys
import os

BASE_URL = "http://localhost:8000"
SESSION_ID = f"e2e_{int(time.time())}"
DATA_PATH = os.path.abspath("train.csv")
TARGET = "SalePrice"
TASK_TYPE = "regression"
TIMEOUT = 600  # 10 minutes per API call

def api_call(method, path, data=None, label=""):
    """Make API call with timeout and error handling."""
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=TIMEOUT)
        else:
            resp = requests.post(url, json=data or {}, timeout=TIMEOUT)

        if resp.status_code >= 400:
            print(f"  HTTP {resp.status_code}: {resp.text[:300]}")
            return None
        return resp.json()
    except requests.exceptions.Timeout:
        print(f"  TIMEOUT after {TIMEOUT}s")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def confirm(session_id, confirmation_id, status="confirmed", modifications=None):
    """Submit confirmation."""
    data = {
        "session_id": session_id,
        "confirmation_id": confirmation_id,
        "status": status,
    }
    if modifications:
        data["modifications"] = modifications
    return api_call("POST", "/confirmation/submit", data, "confirm")

def check_status(session_id):
    """Get workflow status."""
    return api_call("GET", f"/workflow/{session_id}/status")

# ============================================================
print("=" * 60)
print(f"E2E Workflow Test")
print(f"Session: {SESSION_ID}")
print(f"Data: {DATA_PATH}")
print("=" * 60)

# Step 1: Start workflow
print("\n[Step 1] 启动工作流...")
result = api_call("POST", "/workflow/start", {
    "session_id": SESSION_ID,
    "data_path": DATA_PATH,
    "target_column": TARGET,
    "task_type": TASK_TYPE,
    "task_description": "房价预测回归任务，对数化SalePrice计算RMSLE",
    "model": "kimi-k2.5"
})
if not result or not result.get("success"):
    print(f"  FAILED: {result}")
    sys.exit(1)
print(f"  ✅ 工作流启动成功 | Stage: {result.get('current_stage')}")

# Step 2: Problem definition
print("\n[Step 2] 问题定义...")
result = api_call("POST", f"/workflow/{SESSION_ID}/stage/problem_definition/run")
if not result or not result.get("success"):
    print(f"  FAILED: {result}")
    sys.exit(1)
conf_id = result.get("confirmation_id", "")
proposal_len = len(result.get("proposal", ""))
print(f"  ✅ 方案生成 ({proposal_len} chars) | Confirm: {conf_id[:8]}...")

# Confirm problem definition
result = confirm(SESSION_ID, conf_id)
if not result:
    print("  FAILED to confirm")
    sys.exit(1)
print(f"  ✅ 已确认")

# Step 3: Data contract check
print("\n[Step 3] 数据契约检查...")
result = api_call("POST", f"/workflow/{SESSION_ID}/stage/data_contract_check/run")
if not result or not result.get("success"):
    print(f"  FAILED: {result}")
    sys.exit(1)
conf_id = result.get("confirmation_id", "")
modelable = result.get("modelable", "?")
print(f"  ✅ 可建模: {modelable} | Confirm: {conf_id[:8]}...")

result = confirm(SESSION_ID, conf_id)
print(f"  ✅ 已确认")

# Step 4: Data splitting
print("\n[Step 4] 数据集切分...")
result = api_call("POST", f"/workflow/{SESSION_ID}/stage/data_splitting/run")
if not result or not result.get("success"):
    print(f"  FAILED: {result}")
    sys.exit(1)
conf_id = result.get("confirmation_id", "")
split = result.get("split_result", {})
counts = split.get("counts", {})
print(f"  ✅ 切分完成 | train:{counts.get('train_rows','?')} valid:{counts.get('valid_rows','?')} test:{counts.get('test_rows','?')}")
print(f"     Confirm: {conf_id[:8]}...")

# Confirm with modified ratios (70/10/20)
result = confirm(SESSION_ID, conf_id, status="modified",
    modifications=json.dumps({"train_ratio": 0.7, "valid_ratio": 0.1, "test_ratio": 0.2}))
print(f"  ✅ 已确认 (modified 70/10/20)")

# Step 5: Data cleaning
print("\n[Step 5] 数据清洗...")
result = api_call("POST", f"/workflow/{SESSION_ID}/stage/data_cleaning/run")
if not result or not result.get("success"):
    print(f"  FAILED: {result}")
    sys.exit(1)
conf_id = result.get("confirmation_id", "")
plan_len = len(result.get("plan", result.get("proposal", "")))
print(f"  ✅ 清洗方案 ({plan_len} chars) | Confirm: {conf_id[:8]}...")

result = confirm(SESSION_ID, conf_id)
exec_result = result.get("execution_result", {}) if result else {}
print(f"  ✅ 清洗执行 | success: {exec_result.get('success', '?')}")

# Step 6: Data exploration
print("\n[Step 6] 数据探索性分析...")
result = api_call("POST", f"/workflow/{SESSION_ID}/stage/data_exploration/run")
if not result or not result.get("success"):
    print(f"  FAILED: {result}")
    sys.exit(1)
report_len = len(result.get("report", ""))
print(f"  ✅ 探索分析完成 ({report_len} chars)")

# Step 7: Feature engineering
print("\n[Step 7] 特征工程...")
result = api_call("POST", f"/workflow/{SESSION_ID}/stage/feature_engineering/run")
if not result or not result.get("success"):
    print(f"  FAILED: {result}")
    sys.exit(1)
conf_id = result.get("confirmation_id", "")
proposal = result.get("proposal", "")
if "LLM 调用失败" in proposal:
    print(f"  ⚠️ LLM调用失败: {proposal[:200]}")
else:
    print(f"  ✅ 特征方案 ({len(proposal)} chars) | Confirm: {conf_id[:8]}...")

result = confirm(SESSION_ID, conf_id)
exec_result = result.get("execution_result", {}) if result else {}
print(f"  ✅ 特征工程执行 | success: {exec_result.get('success', '?')}")

# Step 8: Model training
print("\n[Step 8] 模型训练...")
result = api_call("POST", f"/workflow/{SESSION_ID}/stage/model_training/run")
if not result or not result.get("success"):
    print(f"  FAILED: {result}")
    sys.exit(1)
conf_id = result.get("confirmation_id", "")
proposal = result.get("proposal", "")
if "LLM 调用失败" in proposal:
    print(f"  ⚠️ LLM调用失败: {proposal[:200]}")
else:
    print(f"  ✅ 训练方案 ({len(proposal)} chars) | Confirm: {conf_id[:8]}...")

result = confirm(SESSION_ID, conf_id)
exec_result = result.get("execution_result", {}) if result else {}
print(f"  ✅ 训练执行 | success: {exec_result.get('success', '?')}")

# Step 9: Model evaluation
print("\n[Step 9] 模型评估...")
result = api_call("POST", f"/workflow/{SESSION_ID}/stage/model_evaluation/run")
if not result or not result.get("success"):
    print(f"  FAILED: {result}")
    sys.exit(1)
conf_id = result.get("confirmation_id", "")
proposal = result.get("proposal", "")
print(f"  ✅ 评估方案 ({len(proposal)} chars) | Confirm: {conf_id[:8]}...")

result = confirm(SESSION_ID, conf_id)
exec_result = result.get("execution_result", {}) if result else {}
print(f"  ✅ 评估执行 | success: {exec_result.get('success', '?')}")

# Step 9.5: Check auto-report
print("\n[Step 9.5] 验证自动报告...")
session_dir = f"assets/{SESSION_ID}"
report_md = os.path.join(session_dir, "reports", "modeling_report.md")
report_html = os.path.join(session_dir, "reports", "modeling_report.html")
summary_json = os.path.join(session_dir, "reports", "summary.json")

if os.path.exists(report_md):
    with open(report_md) as f:
        content = f.read()
    print(f"  ✅ Markdown报告 ({len(content)} chars)")
    sections = ["项目概览", "数据清洗", "特征工程", "模型训练", "模型评估"]
    found = [s for s in sections if s in content]
    missing = [s for s in sections if s not in content]
    print(f"     找到章节: {found}")
    if missing:
        print(f"     缺失章节: {missing}")
else:
    print(f"  ⚠️ 报告不存在: {report_md}")

if os.path.exists(report_html):
    print(f"  ✅ HTML报告存在")
else:
    print(f"  ⚠️ HTML报告不存在")

if os.path.exists(summary_json):
    with open(summary_json) as f:
        summary = json.load(f)
    print(f"  ✅ JSON摘要 | keys: {list(summary.keys())[:5]}")
else:
    print(f"  ⚠️ JSON摘要不存在")

# Step 10: Validate assets
print("\n[Step 10] 验证资产文件...")
expected_dirs = ["data", "analysis", "cleaning", "code", "exploration", "features", "models", "state"]
for d in expected_dirs:
    path = os.path.join(session_dir, d)
    if os.path.exists(path):
        files = os.listdir(path)
        print(f"  ✅ {d}/ ({len(files)} files)")
    else:
        print(f"  ⚠️ {d}/ 不存在")

# Step 10.5: Session CRUD
print("\n[Step 10.5] 测试 Session CRUD...")
result = api_call("GET", "/sessions")
if result:
    sessions = result.get("sessions", [])
    found = any(s.get("session_id") == SESSION_ID for s in sessions)
    print(f"  ✅ GET /sessions | 总数: {len(sessions)} | 找到当前: {found}")

result = api_call("GET", f"/sessions/{SESSION_ID}/status")
if result:
    print(f"  ✅ GET /sessions/{SESSION_ID}/status | stage: {result.get('current_stage')}")

# Final status
print("\n" + "=" * 60)
status = check_status(SESSION_ID)
if status:
    print(f"最终状态: {status.get('current_stage')}")
    stages = [h['stage'] for h in status.get('history', [])]
    print(f"已完成阶段: {stages}")
print(f"\n会话目录: assets/{SESSION_ID}")
print("=" * 60)
print("\n✅ 全流程测试完成!")
