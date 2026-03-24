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

# API 基础地址
BASE_URL = "http://localhost:8000"

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
        "automl_react.api.main"
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
        "model": "kimi-k2.5"
    }
    
    response = post_with_progress(url, json_body=data, timeout=120, label="启动工作流")
    
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
    """步骤 2: 数据清洗阶段 - 生成方案并执行"""
    print_step(2, "数据清洗阶段 (生成方案 + 确认执行)")
    
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
    """步骤 3: 数据探索性分析阶段"""
    print_step(3, "数据探索性分析阶段")

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
    """步骤 4: 特征工程阶段 - 生成方案并执行"""
    print_step(4, "特征工程阶段 (生成方案 + 确认执行)")
    
    url = f"{BASE_URL}/workflow/{SESSION_ID}/stage/feature_engineering/run"
    
    data = {
        "data_path": TRAIN_DATA_PATH,
        "target_column": "SalePrice",
        "task_type": "regression"
    }
    
    response = post_with_progress(url, json_body=data, timeout=900, label="生成特征工程方案")
    
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
                    reliability_path = eval_execution.get("feature_reliability_report_path")
                    if not report_path or not os.path.exists(report_path):
                        print("   ❌ 未生成特征分析报告 metrics_report_path")
                        return False
                    if not reliability_path or not os.path.exists(reliability_path):
                        print("   ❌ 未生成特征可解释性/可靠性报告 feature_reliability_report_path")
                        return False
                    print(f"   ✅ 特征分析报告已生成: {report_path}")
                    print(f"   ✅ 可解释性/可靠性报告已生成: {reliability_path}")
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
    
    response = post_with_progress(url, json_body=data, timeout=900, label=f"提交确认({status})")
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"   ❌ 确认提交失败: {response.status_code} - {response.text}")
        return None


def test_list_assets():
    """步骤 5: 列出所有资产"""
    print_step(5, "列出所有生成的资产")
    
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

        # 关键产物校验
        features_files = [f.get('name') for f in assets.get('features', [])]
        required = {
            "feature_engineering_plan.md",
            "feature_engineering_result.json",
            "feature_metrics.json",
            "feature_metrics_report.md",
            "feature_reliability_report.md",
            "feature_evaluation_result.json",
        }
        missing = [x for x in required if x not in features_files]
        if missing:
            print(f"❌ 缺少关键特征评估产物: {missing}")
            return False
        print("✅ 特征评估与可解释性报告产物校验通过")
        return True
    else:
        print(f"❌ 资产列表获取失败: {response.status_code} - {response.text}")
        return False


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
    
    # 步骤 2: 数据清洗 (生成方案 + 执行)
    results.append(("数据清洗", test_data_cleaning_stage()))
    
    # 步骤 3: 数据探索性分析
    results.append(("数据探索性分析", test_data_exploration_stage()))

    # 步骤 4: 特征工程 (生成方案 + 执行)
    results.append(("特征工程", test_feature_engineering_stage()))

    # 步骤 5: 列出资产
    results.append(("列出资产", test_list_assets()))
    
    # 打印测试总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
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
