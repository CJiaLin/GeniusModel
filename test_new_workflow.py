#!/usr/bin/env python3
"""
测试新的工作流程：数据清洗（含质量分析）→ 数据探索性分析 → 特征工程（方案+代码+执行）
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"
TRAIN_DATA_PATH = "/Users/cjialin/code/AutoMLByLLM/train.csv"

# 用户输入的建模背景和要求
TASK_DESCRIPTION = """
你的工作是预测每套房子的售价。对于测试集中的每个Id，你必须预测SalePrice变量的值。采用RMSE评估预测效果。

背景说明：
- 这是一个房价预测任务
- 目标变量是 SalePrice（房价）
- 评估指标是 RMSE（均方根误差）
- 需要预测测试集中每套房子的价格
"""

def test_new_workflow():
    """测试新的工作流程"""
    print("=" * 80)
    print("测试新工作流程：数据清洗（含质量分析）→ 数据探索性分析 → 特征工程（方案+代码+执行）")
    print("=" * 80)
    
    # 1. 启动工作流
    print("\n" + "=" * 80)
    print("1. 启动工作流")
    print("=" * 80)
    session_id = f"test_new_workflow_{int(time.time())}"
    url = f"{BASE_URL}/workflow/start"
    data = {
        "session_id": session_id,
        "data_path": TRAIN_DATA_PATH,
        "target_column": "SalePrice",
        "task_type": "regression",
        "model": "kimi-k2.5",
        "task_description": TASK_DESCRIPTION
    }
    response = requests.post(url, json=data)
    
    if response.status_code != 200:
        print(f"❌ 启动工作流失败: {response.text}")
        return
    
    print(f"✅ 工作流启动成功，会话 ID: {session_id}")
    
    # 2. 数据清洗阶段 - 生成方案（包含数据质量分析）
    print("\n" + "=" * 80)
    print("2. 数据清洗阶段 - 生成方案（包含数据质量分析）")
    print("=" * 80)
    url = f"{BASE_URL}/workflow/{session_id}/stage/data_cleaning/run"
    response = requests.post(url, json=data, timeout=300)
    
    if response.status_code != 200:
        print(f"❌ 数据清洗方案生成失败: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    cleaning_confirmation_id = result.get("confirmation_id")
    print(f"✅ 数据清洗方案生成完成，方案长度: {len(result.get('proposal', ''))} 字符")
    print(f"   确认 ID: {cleaning_confirmation_id}")
    
    # 3. 数据清洗阶段 - 提交确认并执行
    print("\n" + "=" * 80)
    print("3. 数据清洗阶段 - 提交确认并执行")
    print("=" * 80)
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
    execution = result.get("execution", {})
    cleaned_data_path = execution.get("cleaned_data_path")
    print(f"✅ 数据清洗执行完成")
    print(f"   - success: {execution.get('success')}")
    print(f"   - cleaned_data_path: {cleaned_data_path}")
    
    # 4. 数据探索性分析阶段
    print("\n" + "=" * 80)
    print("4. 数据探索性分析阶段（基于清洗后数据）")
    print("=" * 80)
    url = f"{BASE_URL}/workflow/{session_id}/stage/data_exploration/run"
    response = requests.post(url, json=data, timeout=300)
    
    if response.status_code != 200:
        print(f"❌ 数据探索性分析失败: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    exploration_result = result.get("exploration", "")
    print(f"✅ 数据探索性分析完成，报告长度: {len(exploration_result)} 字符")
    
    # 5. 特征工程阶段 - 生成方案
    print("\n" + "=" * 80)
    print("5. 特征工程阶段 - 生成方案")
    print("=" * 80)
    url = f"{BASE_URL}/workflow/{session_id}/stage/feature_engineering/run"
    response = requests.post(url, json=data, timeout=300)
    
    if response.status_code != 200:
        print(f"❌ 特征工程方案生成失败: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    feature_confirmation_id = result.get("confirmation_id")
    print(f"✅ 特征工程方案生成完成，方案长度: {len(result.get('proposal', ''))} 字符")
    print(f"   确认 ID: {feature_confirmation_id}")
    
    # 6. 特征工程阶段 - 提交确认并执行
    print("\n" + "=" * 80)
    print("6. 特征工程阶段 - 提交确认并执行")
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
    
    # 7. 检查资产文件
    print("\n" + "=" * 80)
    print("7. 检查资产文件")
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
    else:
        print(f"❌ 获取资产列表失败: {response.text}")
    
    # 8. 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print("✅ 数据清洗阶段完成（包含数据质量分析 + 执行）")
    print("✅ 数据探索性分析阶段完成（基于清洗后数据）")
    print("✅ 特征工程阶段完成（方案生成 + 代码生成 + 执行）")
    print(f"\n数据流:")
    print(f"  原始数据 → 清洗后数据 → 探索性分析 → 特征工程")
    print(f"  {TRAIN_DATA_PATH}")
    print(f"  → {cleaned_data_path}")
    print(f"  → {features_data_path}")
    print(f"\n会话 ID: {session_id}")
    print(f"资产目录: assets/{session_id}/")
    print("\n新工作流程测试通过！")

if __name__ == "__main__":
    test_new_workflow()
