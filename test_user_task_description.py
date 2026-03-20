#!/usr/bin/env python3
"""
测试用户建模背景输入功能
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

def test_user_task_description():
    """测试用户建模背景输入"""
    print("=" * 80)
    print("测试用户建模背景输入功能")
    print("=" * 80)
    
    # 1. 启动工作流（包含用户的建模背景）
    print("\n1. 启动工作流（包含用户的建模背景）...")
    session_id = f"test_task_desc_{int(time.time())}"
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
    
    result = response.json()
    print(f"✅ 工作流启动成功，会话 ID: {session_id}")
    
    # 2. 运行数据分析
    print("\n2. 运行数据分析...")
    url = f"{BASE_URL}/workflow/{session_id}/stage/data_analysis/run"
    response = requests.post(url, json=data, timeout=120)
    
    if response.status_code != 200:
        print(f"❌ 数据分析失败: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    print(f"✅ 数据分析完成")
    
    # 3. 检查分析报告是否包含用户的建模背景
    print("\n3. 检查分析报告...")
    analysis = result.get("analysis", "")
    
    print(f"\n分析报告长度: {len(analysis)} 字符")
    print(f"\n分析报告预览:")
    print("-" * 40)
    print(analysis[:500])
    print("-" * 40)
    
    # 检查是否包含关键信息
    checks = {
        "房价预测": "房价" in analysis or "SalePrice" in analysis,
        "RMSE指标": "RMSE" in analysis or "均方根误差" in analysis,
        "建模建议": "建模" in analysis or "模型" in analysis,
        "数据清洗引导": "清洗" in analysis or "cleaning" in analysis.lower()
    }
    
    print("\n检查结果:")
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {'通过' if passed else '未通过'}")
    
    # 4. 检查资产文件
    print("\n" + "=" * 80)
    print("4. 检查资产文件...")
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
    
    print(f"\n会话 ID: {session_id}")
    print(f"资产目录: assets/{session_id}/")

if __name__ == "__main__":
    test_user_task_description()
