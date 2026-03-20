#!/usr/bin/env python3
"""
测试数据分析阶段
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"
TRAIN_DATA_PATH = "/Users/cjialin/code/AutoMLByLLM/train.csv"

def test_data_analysis():
    """测试数据分析阶段"""
    print("=" * 80)
    print("测试数据分析阶段")
    print("=" * 80)
    
    # 1. 启动工作流
    print("\n1. 启动工作流...")
    session_id = f"test_analysis_{int(time.time())}"
    url = f"{BASE_URL}/workflow/start"
    data = {
        "session_id": session_id,
        "data_path": TRAIN_DATA_PATH,
        "target_column": "SalePrice",
        "task_type": "regression",
        "model": "kimi-k2.5"  # 使用配置文件中的模型
    }
    response = requests.post(url, json=data)
    
    if response.status_code != 200:
        print(f"❌ 启动工作流失败: {response.text}")
        return
    
    result = response.json()
    session_id = result.get("session_id")
    print(f"✅ 工作流启动成功，会话 ID: {session_id}")
    
    # 2. 运行数据分析
    print("\n2. 运行数据分析...")
    url = f"{BASE_URL}/workflow/{session_id}/stage/data_analysis/run"
    response = requests.post(url, json=data, timeout=120)
    
    if response.status_code != 200:
        print(f"❌ 数据分析失败: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    print(f"\n✅ 数据分析完成")
    print(f"   - 返回字段: {list(result.keys())}")
    print(f"   - success: {result.get('success')}")
    print(f"   - requires_confirmation: {result.get('requires_confirmation')}")
    
    # 打印分析结果
    analysis = result.get("analysis", "")
    if analysis:
        print(f"\n分析结果长度: {len(analysis)} 字符")
        print("\n" + "=" * 80)
        print("分析结果内容:")
        print("=" * 80)
        print(analysis[:2000] if len(analysis) > 2000 else analysis)
    else:
        print("\n⚠️ 没有返回分析结果")
    
    # 3. 检查资产文件
    print("\n" + "=" * 80)
    print("3. 检查资产文件...")
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
    test_data_analysis()
