#!/usr/bin/env python3
"""
测试特征工程代码生成和执行环节
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_feature_engineering():
    """测试特征工程代码生成和执行"""
    print("=" * 80)
    print("测试特征工程代码生成和执行环节")
    print("=" * 80)
    
    # 1. 启动工作流
    print("\n" + "=" * 80)
    print("1. 启动工作流")
    print("=" * 80)
    session_id = f"test_feature_eng_{int(time.time())}"
    url = f"{BASE_URL}/workflow/start"
    data = {
        "session_id": session_id,
        "data_path": "/Users/cjialin/code/AutoMLByLLM/train.csv",
        "target_column": "SalePrice",
        "task_type": "regression",
        "model": "kimi-k2.5",
        "task_description": "预测房价，使用RMSE评估"
    }
    response = requests.post(url, json=data)
    
    if response.status_code != 200:
        print(f"❌ 启动工作流失败: {response.text}")
        return
    
    print(f"✅ 工作流启动成功，会话 ID: {session_id}")
    
    # 2. 数据清洗阶段
    print("\n" + "=" * 80)
    print("2. 数据清洗阶段")
    print("=" * 80)
    
    # 2.1 生成清洗方案
    url = f"{BASE_URL}/workflow/{session_id}/stage/data_cleaning/run"
    response = requests.post(url, json=data, timeout=300)
    
    if response.status_code != 200:
        print(f"❌ 数据清洗方案生成失败: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    cleaning_confirmation_id = result.get("confirmation_id")
    print(f"✅ 数据清洗方案生成完成")
    
    # 2.2 执行清洗
    url = f"{BASE_URL}/confirmation/submit"
    data_submit = {
        "session_id": session_id,
        "confirmation_id": cleaning_confirmation_id,
        "status": "confirmed",
        "modifications": None
    }
    response = requests.post(url, json=data_submit, timeout=300)
    
    if response.status_code != 200:
        print(f"❌ 数据清洗执行失败: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    cleaned_data_path = result.get("execution", {}).get("cleaned_data_path")
    print(f"✅ 数据清洗执行完成: {cleaned_data_path}")
    
    # 3. 数据探索性分析阶段
    print("\n" + "=" * 80)
    print("3. 数据探索性分析阶段")
    print("=" * 80)
    
    url = f"{BASE_URL}/workflow/{session_id}/stage/data_exploration/run"
    response = requests.post(url, json=data, timeout=300)
    
    if response.status_code != 200:
        print(f"❌ 数据探索性分析失败: {response.status_code} - {response.text}")
        return
    
    print(f"✅ 数据探索性分析完成")
    
    # 4. 特征工程阶段
    print("\n" + "=" * 80)
    print("4. 特征工程阶段")
    print("=" * 80)
    
    # 4.1 生成特征工程方案
    url = f"{BASE_URL}/workflow/{session_id}/stage/feature_engineering/run"
    response = requests.post(url, json=data, timeout=300)
    
    if response.status_code != 200:
        print(f"❌ 特征工程方案生成失败: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    feature_confirmation_id = result.get("confirmation_id")
    print(f"✅ 特征工程方案生成完成")
    
    # 4.2 执行特征工程
    print("\n" + "=" * 80)
    print("4.2 执行特征工程代码")
    print("=" * 80)
    
    url = f"{BASE_URL}/confirmation/submit"
    data_submit = {
        "session_id": session_id,
        "confirmation_id": feature_confirmation_id,
        "status": "confirmed",
        "modifications": None
    }
    response = requests.post(url, json=data_submit, timeout=300)
    
    if response.status_code != 200:
        print(f"❌ 特征工程执行失败: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    execution = result.get("execution", {})
    features_data_path = execution.get("features_data_path")
    print(f"✅ 特征工程执行完成")
    print(f"   - success: {execution.get('success')}")
    print(f"   - features_data_path: {features_data_path}")
    
    # 5. 检查资产文件
    print("\n" + "=" * 80)
    print("5. 检查资产文件")
    print("=" * 80)
    
    url = f"{BASE_URL}/assets/{session_id}/list"
    response = requests.get(url)
    
    if response.status_code == 200:
        assets = response.json()
        print(f"\n资产列表:")
        for asset_type, files in assets.get("assets", {}).items():
            print(f"  - {asset_type}: {len(files)} 个文件")
            for f in files:
                print(f"    * {f.get('filename')} ({f.get('size')} bytes)")
    
    # 6. 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    if execution.get("success"):
        print("✅ 特征工程代码生成和执行成功！")
        print(f"   数据流: 原始数据 → 清洗后数据 → 特征工程后数据")
        print(f"   {features_data_path}")
    else:
        print("❌ 特征工程代码生成和执行失败")

if __name__ == "__main__":
    test_feature_engineering()
