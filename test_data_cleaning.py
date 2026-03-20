#!/usr/bin/env python3
"""
测试数据清洗流程
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"
TRAIN_DATA_PATH = "/Users/cjialin/code/AutoMLByLLM/train.csv"

def test_data_cleaning():
    """测试数据清洗流程"""
    print("=" * 80)
    print("测试数据清洗流程")
    print("=" * 80)
    
    # 1. 启动工作流
    print("\n1. 启动工作流...")
    session_id = f"test_cleaning_{int(time.time())}"
    url = f"{BASE_URL}/workflow/start"
    data = {
        "session_id": session_id,
        "data_path": TRAIN_DATA_PATH,
        "target_column": "SalePrice",
        "task_type": "regression",
        "model": "kimi-k2.5"
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
    else:
        result = response.json()
        print(f"✅ 数据分析完成")
    
    # 3. 运行数据清洗（生成方案）
    print("\n3. 运行数据清洗（生成方案）...")
    url = f"{BASE_URL}/workflow/{session_id}/stage/data_cleaning/run"
    response = requests.post(url, json=data, timeout=120)
    
    if response.status_code != 200:
        print(f"❌ 数据清洗方案生成失败: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    print(f"✅ 数据清洗方案生成成功")
    print(f"   - requires_confirmation: {result.get('requires_confirmation')}")
    print(f"   - confirmation_id: {result.get('confirmation_id')}")
    print(f"   - 方案长度: {len(result.get('proposal', ''))} 字符")
    
    # 4. 提交确认并执行
    print("\n4. 提交确认并执行...")
    confirmation_id = result.get("confirmation_id")
    url = f"{BASE_URL}/confirmation/submit"
    data_submit = {
        "session_id": session_id,
        "confirmation_id": confirmation_id,
        "status": "confirmed",
        "modifications": None
    }
    response = requests.post(url, json=data_submit, timeout=180)
    
    if response.status_code != 200:
        print(f"❌ 确认提交失败: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    print(f"✅ 数据清洗执行完成")
    if result.get("execution"):
        exec_result = result["execution"]
        print(f"   - success: {exec_result.get('success')}")
        print(f"   - cleaned_data_path: {exec_result.get('cleaned_data_path')}")
    
    # 5. 检查资产文件
    print("\n" + "=" * 80)
    print("5. 检查资产文件...")
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
    test_data_cleaning()
