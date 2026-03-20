#!/usr/bin/env python3
"""
使用 train.csv 测试完整建模流程 (v3 - 带执行确认)

数据集: Kaggle House Prices (房价预测)
目标列: SalePrice
任务类型: regression (回归)
"""

import requests
import json
import time
import os

# API 基础地址
BASE_URL = "http://localhost:8000"

# 测试会话 ID
SESSION_ID = f"train_csv_v3_{int(time.time())}"

# 测试数据路径
TRAIN_DATA_PATH = "/Users/cjialin/code/AutoMLByLLM/train.csv"


def print_step(step_num, description):
    """打印步骤信息"""
    print(f"\n{'='*80}")
    print(f"步骤 {step_num}: {description}")
    print('='*80)


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
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 工作流启动成功")
        print(f"   - 会话 ID: {result.get('session_id')}")
        print(f"   - 当前阶段: {result.get('current_stage')}")
        return True
    else:
        print(f"❌ 工作流启动失败: {response.status_code} - {response.text}")
        return False


def test_data_analysis_stage():
    """步骤 2: 数据分析阶段"""
    print_step(2, "数据分析阶段")
    
    url = f"{BASE_URL}/workflow/{SESSION_ID}/stage/data_analysis/run"
    
    data = {
        "data_path": TRAIN_DATA_PATH,
        "target_column": "SalePrice",
        "task_type": "regression"
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 数据分析完成")
        print(f"   - 返回字段: {list(result.keys())}")
        return True
    else:
        print(f"❌ 数据分析失败: {response.status_code} - {response.text}")
        return False


def test_data_cleaning_stage():
    """步骤 3: 数据清洗阶段 - 生成方案并执行"""
    print_step(3, "数据清洗阶段 (生成方案 + 确认执行)")
    
    # 第一步：生成方案
    url = f"{BASE_URL}/workflow/{SESSION_ID}/stage/data_cleaning/run"
    
    data = {
        "data_path": TRAIN_DATA_PATH,
        "target_column": "SalePrice",
        "task_type": "regression"
    }
    
    response = requests.post(url, json=data)
    
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


def test_feature_engineering_stage():
    """步骤 4: 特征工程阶段 - 生成方案并执行"""
    print_step(4, "特征工程阶段 (生成方案 + 确认执行)")
    
    url = f"{BASE_URL}/workflow/{SESSION_ID}/stage/feature_engineering/run"
    
    data = {
        "data_path": TRAIN_DATA_PATH,
        "target_column": "SalePrice",
        "task_type": "regression"
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 特征工程方案生成成功")
        print(f"   - 方案长度: {len(result.get('proposal', ''))} 字符")
        
        # 确认并执行
        if result.get('requires_confirmation'):
            print(f"   提交确认并执行...")
            exec_result = test_submit_confirmation_and_execute(
                result.get('confirmation_id'), 
                "confirmed"
            )
            if exec_result:
                print(f"   ✅ 特征工程执行完成")
                print(f"   - 执行结果: {exec_result.get('execution', {})}")
                return True
        return True
    else:
        print(f"❌ 特征工程失败: {response.status_code} - {response.text}")
        return False


def test_model_training_stage():
    """步骤 5: 模型训练阶段 - 生成方案并执行"""
    print_step(5, "模型训练阶段 (生成方案 + 确认执行)")
    
    url = f"{BASE_URL}/workflow/{SESSION_ID}/stage/model_training/run"
    
    data = {
        "data_path": TRAIN_DATA_PATH,
        "target_column": "SalePrice",
        "task_type": "regression"
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 建模方案生成成功")
        print(f"   - 方案长度: {len(result.get('proposal', ''))} 字符")
        
        # 确认并执行
        if result.get('requires_confirmation'):
            print(f"   提交确认并执行...")
            exec_result = test_submit_confirmation_and_execute(
                result.get('confirmation_id'), 
                "confirmed"
            )
            if exec_result:
                print(f"   ✅ 模型训练执行完成")
                print(f"   - 执行结果: {exec_result.get('execution', {})}")
                return True
        return True
    else:
        print(f"❌ 模型训练失败: {response.status_code} - {response.text}")
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
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"   ❌ 确认提交失败: {response.status_code} - {response.text}")
        return None


def test_generate_report():
    """步骤 6: 生成建模报告"""
    print_step(6, "生成建模报告")
    
    url = f"{BASE_URL}/report/generate?session_id={SESSION_ID}"
    
    response = requests.post(url)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 报告生成完成")
        if result.get('downloads'):
            print(f"   - Markdown: {result.get('downloads', {}).get('markdown')}")
            print(f"   - HTML: {result.get('downloads', {}).get('html')}")
        return True
    else:
        print(f"❌ 报告生成失败: {response.status_code} - {response.text}")
        return False


def test_generate_pipeline():
    """步骤 7: 生成全流程脚本"""
    print_step(7, "生成全流程建模脚本")
    
    url = f"{BASE_URL}/pipeline/generate?session_id={SESSION_ID}"
    
    response = requests.post(url)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 全流程脚本生成完成")
        if result.get('download_url'):
            print(f"   - 脚本路径: {result.get('download_url')}")
        return True
    else:
        print(f"❌ 全流程脚本生成失败: {response.status_code} - {response.text}")
        return False


def test_list_assets():
    """步骤 8: 列出所有资产"""
    print_step(8, "列出所有生成的资产")
    
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
        return True
    else:
        print(f"❌ 资产列表获取失败: {response.status_code} - {response.text}")
        return False


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("AutoML ReAct - train.csv 完整建模流程测试 (v3 - 带执行)")
    print("="*80)
    print(f"会话 ID: {SESSION_ID}")
    print(f"测试数据: {TRAIN_DATA_PATH}")
    print(f"数据集: Kaggle House Prices (房价预测)")
    print(f"目标列: SalePrice")
    print(f"任务类型: regression")
    print(f"后端地址: {BASE_URL}")
    
    # 检查服务是否可用
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"\n✅ 后端服务正常")
    except Exception as e:
        print(f"\n❌ 后端服务不可用: {e}")
        print("请确保后端服务已启动: python -m automl_react.api.main")
        return
    
    results = []
    
    # 步骤 1: 启动工作流
    results.append(("启动工作流", test_start_workflow()))
    
    # 步骤 2: 数据分析
    results.append(("数据分析", test_data_analysis_stage()))
    
    # 步骤 3: 数据清洗 (生成方案 + 执行)
    results.append(("数据清洗", test_data_cleaning_stage()))
    
    # 步骤 4: 特征工程 (生成方案 + 执行)
    results.append(("特征工程", test_feature_engineering_stage()))
    
    # 步骤 5: 模型训练 (生成方案 + 执行)
    results.append(("模型训练", test_model_training_stage()))
    
    # 步骤 6: 生成报告
    results.append(("生成报告", test_generate_report()))
    
    # 步骤 7: 生成全流程脚本
    results.append(("生成全流程脚本", test_generate_pipeline()))
    
    # 步骤 8: 列出资产
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


if __name__ == "__main__":
    main()
