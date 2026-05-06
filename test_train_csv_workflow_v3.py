#!/usr/bin/env python3
"""
使用 train.csv 测试当前后端真实阶段顺序 (v3)

数据集: Kaggle House Prices (房价预测)
目标列: SalePrice
任务类型: regression (回归)
"""

import requests
import json
import time
import os
import sys
import threading
import subprocess

import joblib

# API 基础地址
BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8000")

# 测试会话 ID
SESSION_ID = f"train_csv_v3_{int(time.time())}"

# 测试数据路径
TRAIN_DATA_PATH = "/Users/cjialin/code/AutoMLByLLM/train.csv"

# 若本地后端未启动，则自动启动并透传后端日志到当前终端
AUTO_START_BACKEND = True
_backend_process = None
_backend_log_thread = None


def print_step(step_num, description):
    """打印步骤信息"""
    print(f"\n{'='*80}")
    print(f"步骤 {step_num}: {description}")
    print('='*80)


def _stream_backend_logs(pipe):
    """后台读取并转发后端日志，便于观察流式 LLM 输出。"""
    try:
        for line in iter(pipe.readline, ''):
            if not line:
                break
            print(f"[BACKEND] {line.rstrip()}")
    except Exception as e:
        print(f"[BACKEND] 日志流读取异常: {e}")


def ensure_backend_running():
    """确保后端服务可用；若不可用则自动启动并输出后端日志。"""
    global _backend_process, _backend_log_thread

    try:
        requests.get(f"{BASE_URL}/", timeout=3)
        print("\n✅ 后端服务正常（复用已有服务）")
        print("ℹ️ 当前复用外部后端进程时，后端日志不会自动透传到本测试终端")
        return True
    except Exception:
        if not AUTO_START_BACKEND:
            print("\n❌ 后端服务不可用，且未启用自动启动")
            return False

    print("\nℹ️ 检测到后端未运行，自动启动本地后端并透传日志...")
    cmd = [
        "/Users/cjialin/code/AutoMLByLLM/venv/bin/python",
        "-u",
        "-m",
        "uvicorn",
        "automl_react.api.main:app",
        "--host", "0.0.0.0",
        "--port", BASE_URL.rsplit(":", 1)[-1],
    ]
    _backend_process = subprocess.Popen(
        cmd,
        cwd="/Users/cjialin/code/AutoMLByLLM",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    _backend_log_thread = threading.Thread(
        target=_stream_backend_logs,
        args=(_backend_process.stdout,),
        daemon=True,
    )
    _backend_log_thread.start()

    for _ in range(30):
        try:
            requests.get(f"{BASE_URL}/", timeout=2)
            print("✅ 本地后端启动成功")
            return True
        except Exception:
            time.sleep(1)

    print("❌ 本地后端启动超时")
    return False


def shutdown_backend_if_started():
    """若由测试脚本启动后端，则在结束时回收进程。"""
    global _backend_process
    if _backend_process and _backend_process.poll() is None:
        _backend_process.terminate()
        try:
            _backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _backend_process.kill()
    _backend_process = None


def post_with_progress(url, json_body=None, timeout=600, label="请求"):
    """带简易进度提示的 POST 请求，避免长耗时阶段无输出。"""
    print(f"   ⏳ {label}中", end="", flush=True)
    try:
        response = requests.post(url, json=json_body, timeout=timeout)
    finally:
        print(" ... 完成")
    return response


def test_start_workflow():
    """步骤 1: 启动工作流"""
    print_step(1, "启动工作流 (train.csv - 房价预测)")
    
    url = f"{BASE_URL}/workflow/start"
    
    data = {
        "session_id": SESSION_ID,
        "data_path": TRAIN_DATA_PATH,
        "target_column": "SalePrice",
        "task_type": "regression",
        "model": "kimi-k2.5",
        "task_description": "我希望建立一个房价预测模型，用于预测二手房的销售价格。请重点关注房屋面积、地段、建造年份等关键特征，并尝试多种回归算法进行对比。采用对数化后的 SalePrice 计算评估指标。"
    }
    
    response = post_with_progress(url, json_body=data, timeout=600, label="启动工作流")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 工作流启动成功")
        print(f"   - 会话 ID: {result.get('session_id')}")
        print(f"   - 当前阶段: {result.get('current_stage')}")
        return True
    else:
        print(f"❌ 工作流启动失败: {response.status_code} - {response.text}")
        return False


def test_data_cleaning_stage():
    """步骤 5: 数据清洗阶段 - 生成方案并执行"""
    print_step(5, "数据清洗阶段 (生成方案 + 确认执行)")
    
    # 第一步：生成方案
    url = f"{BASE_URL}/workflow/{SESSION_ID}/stage/data_cleaning/run"
    
    data = {
        "data_path": TRAIN_DATA_PATH,
        "target_column": "SalePrice",
        "task_type": "regression"
    }
    
    response = post_with_progress(url, json_body=data, timeout=600, label="生成数据清洗方案")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 数据清洗方案生成成功")
        print(f"   - 方案长度: {len(result.get('proposal', ''))} 字符")
        
        # 第二步：确认并执行
        if result.get('requires_confirmation'):
            print(f"   提交确认并执行...")
            exec_result = test_submit_confirmation_and_execute(
                result.get('confirmation_id'), 
                "confirmed"
            )
            if exec_result:
                print(f"   ✅ 清洗执行完成")
                print(f"   - 执行结果: {exec_result.get('execution', {})}")
                return True
        return True
    else:
        print(f"❌ 数据清洗失败: {response.status_code} - {response.text}")
        return False


def test_data_exploration_stage():
    """步骤 6: 数据探索性分析阶段"""
    print_step(6, "数据探索性分析阶段")

    url = f"{BASE_URL}/workflow/{SESSION_ID}/stage/data_exploration/run"

    data = {
        "data_path": TRAIN_DATA_PATH,
        "target_column": "SalePrice",
        "task_type": "regression"
    }

    response = post_with_progress(url, json_body=data, timeout=600, label="执行数据探索性分析")

    if response.status_code == 200:
        result = response.json()
        exploration = result.get("exploration", "")
        print("✅ 数据探索性分析完成")
        print(f"   - 报告长度: {len(exploration)} 字符")
        return True
    else:
        print(f"❌ 数据探索性分析失败: {response.status_code} - {response.text}")
        return False


def test_feature_engineering_stage():
    """步骤 7: 特征工程阶段 - 生成方案并执行"""
    print_step(7, "特征工程阶段 (生成方案 + 确认执行)")
    
    url = f"{BASE_URL}/workflow/{SESSION_ID}/stage/feature_engineering/run"
    
    data = {
        "data_path": TRAIN_DATA_PATH,
        "target_column": "SalePrice",
        "task_type": "regression"
    }
    
    response = post_with_progress(url, json_body=data, timeout=1800, label="生成特征工程方案")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 特征工程方案生成成功")
        print(f"   - 方案长度: {len(result.get('proposal', ''))} 字符")
        
        # 确认并执行（应返回下一确认点：feature_evaluation）
        if result.get('requires_confirmation'):
            print(f"   提交确认并执行...")
            exec_result = test_submit_confirmation_and_execute(
                result.get('confirmation_id'), 
                "confirmed"
            )
            if exec_result:
                print(f"   ✅ 特征工程执行完成")
                print(f"   - 执行结果: {exec_result.get('execution', {})}")
                next_confirmation = exec_result.get("next_confirmation")
                if next_confirmation and next_confirmation.get("stage") == "feature_evaluation":
                    print("   ✅ 检测到特征评估确认点，开始执行特征评估")
                    eval_result = test_submit_confirmation_and_execute(
                        next_confirmation.get("confirmation_id"),
                        "confirmed"
                    )
                    if not eval_result:
                        print("   ❌ 特征评估确认执行失败")
                        return False
                    eval_execution = eval_result.get("execution", {})
                    print(f"   ✅ 特征评估执行完成")
                    print(f"   - 评估结果: {eval_execution}")

                    report_path = eval_execution.get("metrics_report_path")
                    if not report_path or not os.path.exists(report_path):
                        print("   ❌ 未生成特征分析报告 metrics_report_path")
                        return False
                    print(f"   ✅ 特征分析报告已生成: {report_path}")
                else:
                    print("   ❌ 未返回 feature_evaluation 确认点")
                    return False
                return True
            print("   ❌ 特征工程确认执行失败")
            return False
        print("   ❌ 特征工程阶段未返回 requires_confirmation，流程异常")
        return False
    else:
        print(f"❌ 特征工程失败: {response.status_code} - {response.text}")
        return False


def test_submit_confirmation_and_execute(confirmation_id, status="confirmed", modifications=None):
    """提交用户确认并执行"""
    url = f"{BASE_URL}/confirmation/submit"
    
    data = {
        "session_id": SESSION_ID,
        "confirmation_id": confirmation_id,
        "status": status,
        "modifications": modifications or ""
    }
    
    response = post_with_progress(url, json_body=data, timeout=1800, label=f"提交确认({status})")
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"   ❌ 确认提交失败: {response.status_code} - {response.text}")
        return None


def test_model_training_stage():
    """步骤 8: 模型训练阶段 - 生成方案并执行"""
    print_step(8, "模型训练阶段 (生成方案 + 确认执行)")

    # 第一步：生成方案
    url = f"{BASE_URL}/workflow/{SESSION_ID}/stage/model_training/run"

    data = {
        "data_path": TRAIN_DATA_PATH,
        "target_column": "SalePrice",
        "task_type": "regression"
    }

    response = post_with_progress(url, json_body=data, timeout=1800, label="生成模型训练方案")

    if response.status_code != 200:
        print(f"❌ 模型训练方案生成失败: {response.status_code} - {response.text}")
        return False

    result = response.json()
    if not result.get("success"):
        print(f"❌ 模型训练方案生成失败: {result.get('error', '未知错误')}")
        return False

    print(f"✅ 模型训练方案生成成功")
    print(f"   - 方案长度: {len(result.get('proposal', ''))} 字符")

    # 第二步：确认并执行
    if not result.get("requires_confirmation"):
        print("   ❌ 模型训练阶段未返回 requires_confirmation，流程异常")
        return False

    print(f"   提交确认并执行...")
    exec_result = test_submit_confirmation_and_execute(
        result.get("confirmation_id"),
        "confirmed"
    )
    if not exec_result:
        print("   ❌ 模型训练确认执行失败")
        return False

    execution = exec_result.get("execution", {})
    print(f"   ✅ 模型训练执行完成")
    print(f"   - 训练成功: {execution.get('success')}")
    print(f"   - 模型路径: {execution.get('model_path')}")
    print(f"   - 训练摘要: {execution.get('training_summary_path')}")

    # 验证关键指标
    metrics = execution.get("metrics", {})
    if metrics:
        print(f"   - 模型指标: {metrics}")

    # 验证关键产物文件
    model_path = execution.get("model_path")
    if not model_path or not os.path.exists(model_path):
        print(f"   ❌ 模型文件未生成: {model_path}")
        return False
    print(f"   ✅ 模型文件已生成: {model_path}")

    training_summary_path = execution.get("training_summary_path")
    if training_summary_path and os.path.exists(training_summary_path):
        print(f"   ✅ 训练摘要已生成: {training_summary_path}")

    return True


def test_model_evaluation_stage():
    """步骤 9: 模型评估阶段 - 生成方案并确认执行"""
    print_step(9, "模型评估阶段")

    url = f"{BASE_URL}/workflow/{SESSION_ID}/stage/model_evaluation/run"
    response = post_with_progress(url, json_body={}, timeout=1800, label="执行模型评估")

    if response.status_code != 200:
        print(f"❌ 模型评估失败: {response.status_code} - {response.text}")
        return False

    result = response.json()
    if not result.get("success"):
        print(f"❌ 模型评估方案生成失败: {result}")
        return False

    print("✅ 模型评估方案生成成功")
    print(f"   - 方案长度: {len(result.get('proposal', ''))} 字符")

    if not result.get("requires_confirmation"):
        print("   ❌ 模型评估阶段未返回 requires_confirmation，流程异常")
        return False

    exec_result = test_submit_confirmation_and_execute(
        result.get("confirmation_id"),
        "confirmed"
    )
    if not exec_result:
        print("   ❌ 模型评估确认执行失败")
        return False

    evaluation = exec_result.get("execution", {})
    if not evaluation.get("success"):
        print(f"❌ 模型评估执行失败: {evaluation.get('error', evaluation)}")
        return False

    metrics = evaluation.get("metrics", {})
    print("✅ 模型评估执行完成")
    print(f"   - 评估数据: {evaluation.get('data_path')}")
    print(f"   - 评估指标: {metrics}")

    if not metrics:
        print("   ❌ 未返回评估指标")
        return False

    required_metrics = {"rmse", "mae", "r2", "rmsle", "mape"}
    missing = [name for name in required_metrics if name not in metrics]
    if missing:
        print(f"   ❌ 缺少关键评估指标: {missing}")
        return False

    return True


def test_auto_report():
    """步骤 9.5: 验证模型评估后自动生成的报告"""
    print_step("9.5", "验证自动生成的报告和 JSON 摘要")

    import time
    time.sleep(1)  # 等待报告异步写入

    # 检查 Markdown 报告
    report_path = f"assets/{SESSION_ID}/reports/modeling_report.md"
    if not os.path.exists(report_path):
        print(f"❌ Markdown 报告未自动生成: {report_path}")
        return False
    with open(report_path) as f:
        report = f.read()
    print(f"✅ Markdown 报告已自动生成 ({len(report)} 字符)")

    # 验证报告包含所有章节
    required_sections = ["项目概述", "数据切分", "数据清洗", "数据探索分析",
                         "特征工程", "模型训练", "模型评估", "可视化分析", "结论与建议"]
    missing = [s for s in required_sections if s not in report]
    if missing:
        print(f"❌ 报告缺少章节: {missing}")
        return False
    print(f"✅ 报告包含全部 {len(required_sections)} 个章节")

    # 检查 HTML 报告
    html_path = f"assets/{SESSION_ID}/reports/modeling_report.html"
    if os.path.exists(html_path):
        print(f"✅ HTML 报告已生成")
    else:
        print("⚠️ HTML 报告未生成（可能缺少 markdown 库）")

    # 检查 JSON 摘要
    summary_path = f"assets/{SESSION_ID}/reports/summary.json"
    if not os.path.exists(summary_path):
        print(f"❌ JSON 摘要未自动生成: {summary_path}")
        return False
    with open(summary_path) as f:
        summary = json.load(f)
    required_keys = {"session_id", "problem_definition", "data_summary",
                     "cleaning_summary", "feature_summary", "model_summary",
                     "evaluation_summary", "generated_at"}
    missing = [k for k in required_keys if k not in summary]
    if missing:
        print(f"❌ JSON 摘要缺少字段: {missing}")
        return False
    print(f"✅ JSON 摘要已生成，包含 {len(summary)} 个顶层字段")

    # 检查图表
    charts_dir = f"assets/{SESSION_ID}/reports/charts"
    if os.path.exists(charts_dir):
        chart_files = os.listdir(charts_dir)
        print(f"✅ 已生成 {len(chart_files)} 个可视化图表: {chart_files}")
    else:
        print("⚠️ 未生成可视化图表目录")

    return True


def test_session_crud():
    """步骤 10.5: 测试会话 CRUD API"""
    print_step("10.5", "测试会话 CRUD API")

    # GET /sessions
    response = requests.get(f"{BASE_URL}/sessions")
    if response.status_code != 200:
        print(f"❌ GET /sessions 失败: {response.status_code}")
        return False
    sessions = response.json()
    session_ids = [s["session_id"] for s in sessions.get("sessions", [])]
    if SESSION_ID not in session_ids:
        print(f"❌ 当前会话 {SESSION_ID} 未出现在会话列表中")
        return False
    print(f"✅ GET /sessions 返回 {sessions['count']} 个会话")

    # GET /sessions/{id}/status
    response = requests.get(f"{BASE_URL}/sessions/{SESSION_ID}/status")
    if response.status_code != 200:
        print(f"❌ GET /sessions/{SESSION_ID}/status 失败: {response.status_code}")
        return False
    detail = response.json()
    print(f"✅ 会话详情: stage={detail.get('current_stage')}, agents={detail.get('agents_loaded')}")

    # GET /report/{session_id}/summary
    response = requests.get(f"{BASE_URL}/report/{SESSION_ID}/summary")
    if response.status_code != 200:
        print(f"❌ GET /report/{SESSION_ID}/summary 失败: {response.status_code}")
        return False
    summary = response.json()
    print(f"✅ 报告摘要 API: model={summary.get('model_summary', {}).get('best_model')}")

    return True


def test_list_assets():
    """步骤 10: 列出所有资产"""
    print_step(10, "列出所有生成的资产")
    
    url = f"{BASE_URL}/assets/{SESSION_ID}/list"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        assets = result.get('assets', {})
        print(f"✅ 资产列表获取成功")
        total_files = 0
        for asset_type, files in assets.items():
            print(f"   - {asset_type}: {len(files)} 个文件")
            total_files += len(files)
            for f in files[:5]:
                print(f"     * {f.get('name')} ({f.get('size', 0)} bytes)")
        print(f"   总计: {total_files} 个文件")

        analysis_files = [f.get('name') for f in assets.get('analysis', [])]
        required_analysis = {
            "problem_definition.md",
            "problem_definition.json",
            "data_contract_report.md",
            "data_contract_result.json",
            "dataset_split_report.md",
            "dataset_split_result.json",
        }
        missing = [x for x in required_analysis if x not in analysis_files]
        if missing:
            print(f"❌ 缺少关键问题定义产物: {missing}")
            return False
        print("✅ 问题定义产物校验通过")

        data_files = [f.get('name') for f in assets.get('data', [])]
        required_data_files = {"original_data.csv", "train_raw.csv", "test_raw.csv"}
        missing = [x for x in required_data_files if x not in data_files]
        if missing:
            print(f"❌ 缺少关键数据切分产物: {missing}")
            return False
        if "valid_raw.csv" in data_files:
            print("✅ 数据切分产物校验通过")
        else:
            print("⚠️ 当前会话未生成 valid_raw.csv，按最小 train/test 资源模式处理")

        # 关键产物校验 - 特征工程
        features_files = [f.get('name') for f in assets.get('features', [])]
        required_features = {
            "feature_engineering_plan.md",
            "feature_engineering_result.json",
            "feature_metrics.json",
            "feature_metrics_report.md",
            "feature_evaluation_result.json",
        }
        missing = [x for x in required_features if x not in features_files]
        if missing:
            print(f"❌ 缺少关键特征评估产物: {missing}")
            return False
        print("✅ 特征评估产物校验通过")

        # 关键产物校验 - 模型训练
        models_files = [f.get('name') for f in assets.get('models', [])]
        required_models = {
            "model_training_plan.md",
            "model_training_result.json",
            "trained_model.pkl",
            "training_summary.json",
        }
        missing = [x for x in required_models if x not in models_files]
        if missing:
            print(f"❌ 缺少关键模型训练产物: {missing}")
            return False
        print("✅ 模型训练产物校验通过")

        model_artifact_path = f"/Users/cjialin/code/AutoMLByLLM/assets/{SESSION_ID}/models/trained_model.pkl"
        try:
            model_artifact = joblib.load(model_artifact_path)
        except Exception as error:
            print(f"❌ 模型产物读取失败: {error}")
            return False

        if not isinstance(model_artifact, dict):
            print(f"❌ 模型产物不是标准打包结构: {type(model_artifact)}")
            return False

        required_artifact_keys = {"model", "selected_feature_names", "target_transform", "preprocessor"}
        missing_artifact_keys = [key for key in required_artifact_keys if key not in model_artifact]
        if missing_artifact_keys:
            print(f"❌ 模型产物缺少关键字段: {missing_artifact_keys}")
            return False

        print("✅ 模型包结构校验通过")

        reports_files = [f.get('name') for f in assets.get('reports', [])]
        required_reports = {"evaluation.json", "modeling_report.md", "summary.json"}
        missing = [x for x in required_reports if x not in reports_files]
        if missing:
            print(f"❌ 缺少关键报告产物: {missing}")
            return False
        print("✅ 模型评估和报告产物校验通过")
        return True
    else:
        print(f"❌ 资产列表获取失败: {response.status_code} - {response.text}")
        return False


def test_problem_definition_stage():
    """步骤 2: 问题定义阶段 - 生成方案并确认固化"""
    print_step(2, "问题定义阶段 (生成方案 + 确认固化)")

    url = f"{BASE_URL}/workflow/{SESSION_ID}/stage/problem_definition/run"

    response = post_with_progress(url, json_body={}, timeout=300, label="生成问题定义方案")

    if response.status_code != 200:
        print(f"❌ 问题定义失败: {response.status_code} - {response.text}")
        return False

    result = response.json()
    if not result.get("success"):
        print(f"❌ 问题定义方案生成失败: {result.get('error', '未知错误')}")
        return False

    print("✅ 问题定义方案生成成功")
    print(f"   - 方案长度: {len(result.get('proposal', ''))} 字符")

    definition = result.get("problem_definition", {})
    required_fields = {
        "task_type",
        "target_column",
        "prediction_target",
        "prediction_timing",
        "primary_metric",
        "business_constraints",
        "success_criteria",
    }
    missing_fields = [name for name in required_fields if name not in definition]
    if missing_fields:
        print(f"❌ 问题定义缺少关键字段: {missing_fields}")
        return False

    if not result.get("requires_confirmation"):
        print("❌ 问题定义阶段未返回 requires_confirmation，流程异常")
        return False

    exec_result = test_submit_confirmation_and_execute(
        result.get("confirmation_id"),
        "confirmed"
    )
    if not exec_result:
        print("❌ 问题定义确认执行失败")
        return False

    execution = exec_result.get("execution", {})
    if not execution.get("success"):
        print(f"❌ 问题定义固化失败: {execution}")
        return False

    print("✅ 问题定义已确认并固化")
    print(f"   - 报告路径: {execution.get('problem_definition_path')}")
    print(f"   - JSON 路径: {execution.get('problem_definition_json_path')}")
    return True


def test_data_contract_check_stage():
    """步骤 3: 数据契约检查 - 确认可建模性"""
    print_step(3, "数据契约检查 (可建模性审查 + 确认)")

    url = f"{BASE_URL}/workflow/{SESSION_ID}/stage/data_contract_check/run"

    response = post_with_progress(url, json_body={}, timeout=60, label="执行数据契约检查")

    if response.status_code != 200:
        print(f"❌ 数据契约检查失败: {response.status_code} - {response.text}")
        return False

    result = response.json()
    if not result.get("success"):
        print(f"❌ 数据契约检查失败: {result.get('error', '未知错误')}")
        return False

    modelable = result.get("modelable")
    stats = result.get("stats", {})
    risk_list = result.get("risk_list", [])
    questions = result.get("questions_for_business", [])

    print(f"✅ 数据契约检查完成")
    print(f"   - 结论: {'可建模' if modelable else '不可建模'}")
    print(f"   - 阻断项: {stats.get('n_blockers', 0)}")
    print(f"   - 警告项: {stats.get('n_warnings', 0)}")
    print(f"   - 风险数: {len(risk_list)}")
    print(f"   - 待确认问题数: {len(questions)}")

    if not modelable:
        print("   ⚠️ 数据不可建模，但仍继续测试流程")

    if not result.get("requires_confirmation"):
        print("❌ 数据契约检查未返回 requires_confirmation")
        return False

    exec_result = test_submit_confirmation_and_execute(
        result.get("confirmation_id"),
        "confirmed"
    )
    if not exec_result:
        print("❌ 数据契约检查确认失败")
        return False

    execution = exec_result.get("execution", {})
    if not execution.get("success"):
        print(f"❌ 数据契约检查固化失败: {execution}")
        return False

    print("✅ 数据契约检查已确认")
    return True


def test_data_splitting_stage():
    """步骤 4: 数据集切分 - 在清洗前固化 train/valid/test"""
    print_step(4, "数据集切分阶段 (生成方案 + 确认固化)")

    url = f"{BASE_URL}/workflow/{SESSION_ID}/stage/data_splitting/run"

    response = post_with_progress(url, json_body={}, timeout=600, label="执行数据集切分")

    if response.status_code != 200:
        print(f"❌ 数据集切分失败: {response.status_code} - {response.text}")
        return False

    result = response.json()
    if not result.get("success"):
        print(f"❌ 数据集切分失败: {result.get('error', '未知错误')}")
        return False

    counts = result.get("counts", {})
    print("✅ 数据集切分方案生成成功")
    print(f"   - 切分策略: {result.get('split_strategy')}")
    print(f"   - train: {counts.get('train_rows', 0)}")
    print(f"   - valid: {counts.get('valid_rows', 0)}")
    print(f"   - test: {counts.get('test_rows', 0)}")

    if not result.get("requires_confirmation"):
        print("❌ 数据集切分阶段未返回 requires_confirmation")
        return False

    split_modifications = json.dumps({
        "train_ratio": 0.7,
        "valid_ratio": 0.1,
        "test_ratio": 0.2,
        "split_method": "binned_regression",
    }, ensure_ascii=False)

    exec_result = test_submit_confirmation_and_execute(
        result.get("confirmation_id"),
        "modified",
        split_modifications,
    )
    if not exec_result:
        print("❌ 数据集切分确认失败")
        return False

    execution = exec_result.get("execution", {})
    if not execution.get("success"):
        print(f"❌ 数据集切分固化失败: {execution}")
        return False

    for key in ["train_raw_path", "test_raw_path"]:
        path = execution.get(key)
        if not path or not os.path.exists(path):
            print(f"❌ 缺少切分资产: {key} -> {path}")
            return False

    valid_raw_path = execution.get("valid_raw_path")
    if counts.get("valid_rows", 0) > 0 and (not valid_raw_path or not os.path.exists(valid_raw_path)):
        print(f"❌ 缺少验证集资产: {valid_raw_path}")
        return False

    status_response = requests.get(f"{BASE_URL}/workflow/{SESSION_ID}/status", timeout=60)
    if status_response.status_code != 200:
        print(f"❌ 无法读取 workflow 状态: {status_response.status_code} - {status_response.text}")
        return False

    status_payload = status_response.json()
    split_config = status_payload.get("context", {}).get("split_config", {})
    if split_config.get("train_ratio") != 0.7 or split_config.get("valid_ratio") != 0.1:
        print(f"❌ 用户修改后的切分配置未写入 workflow context: {split_config}")
        return False

    print(f"   - 修改后配置: {split_config}")
    print(f"   - 执行模式: {execution.get('execution_mode')}")

    print("✅ 数据集切分已确认并固化")
    return True


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("AutoML ReAct - train.csv 当前阶段顺序测试 (v3)")
    print("="*80)
    print(f"会话 ID: {SESSION_ID}")
    print(f"测试数据: {TRAIN_DATA_PATH}")
    print(f"数据集: Kaggle House Prices (房价预测)")
    print(f"目标列: SalePrice")
    print(f"任务类型: regression")
    print(f"后端地址: {BASE_URL}")
    
    # 检查服务是否可用（必要时自动启动并透传日志）
    if not ensure_backend_running():
        return
    
    results = []
    
    # 步骤 1: 启动工作流
    results.append(("启动工作流", test_start_workflow()))

    # 步骤 2: 问题定义
    results.append(("问题定义", test_problem_definition_stage()))

    # 步骤 3: 数据契约检查
    results.append(("数据契约检查", test_data_contract_check_stage()))
    
    # 步骤 4: 数据集切分
    results.append(("数据集切分", test_data_splitting_stage()))

    # 步骤 5: 数据清洗 (生成方案 + 执行)
    results.append(("数据清洗", test_data_cleaning_stage()))
    
    # 步骤 6: 数据探索性分析
    results.append(("数据探索性分析", test_data_exploration_stage()))

    # 步骤 7: 特征工程 (生成方案 + 执行)
    results.append(("特征工程", test_feature_engineering_stage()))

    # 步骤 8: 模型训练 (生成方案 + 执行)
    results.append(("模型训练", test_model_training_stage()))

    # 步骤 9: 模型评估
    results.append(("模型评估", test_model_evaluation_stage()))

    # 步骤 9.5: 验证自动报告
    results.append(("自动报告验证", test_auto_report()))

    # 步骤 10: 资产列表
    results.append(("列出资产", test_list_assets()))

    # 步骤 10.5: 会话 CRUD API
    results.append(("会话 CRUD API", test_session_crud()))
    
    print("\n" + "="*80)
    print("测试总结")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败")
    
    print(f"\n会话 ID: {SESSION_ID}")
    print(f"前端访问: http://localhost:8080")
    print(f"API 文档: http://localhost:8000/docs")

    shutdown_backend_if_started()


if __name__ == "__main__":
    main()
