#!/usr/bin/env python3
"""验证数据接入与资产固化功能"""
import requests, json, time, os

BASE = "http://localhost:8000"
SID = f"onboarding_check_{int(time.time())}"
DATA = "/Users/cjialin/code/AutoMLByLLM/train.csv"

# 1. 启动工作流
r = requests.post(f"{BASE}/workflow/start", json={
    "session_id": SID,
    "data_path": DATA,
    "target_column": "SalePrice",
    "task_type": "regression",
    "model": "kimi-k2.5",
    "task_description": "房价预测",
})
print("启动:", r.json().get("success"), "| data_path:", r.json().get("data_path"))

asset_dir = f"assets/{SID}/data"

# 2. 检查原始文件复制
original = os.path.join(asset_dir, "original_data.csv")
print(f"\n[1] 原始文件存在: {os.path.exists(original)}")
if os.path.exists(original):
    print(f"    大小: {os.path.getsize(original)} bytes")

# 3. 检查 schema 快照
schema_path = os.path.join(asset_dir, "schema_snapshot.json")
print(f"\n[2] Schema 快照存在: {os.path.exists(schema_path)}")
if os.path.exists(schema_path):
    with open(schema_path) as f:
        schema = json.load(f)
    print(f"    列数: {len(schema.get('columns', []))}")
    print(f"    行数: {schema.get('shape', ['?'])[0]}")
    print(f"    数值列: {len(schema.get('numeric_columns', []))}")
    print(f"    类别列: {len(schema.get('categorical_columns', []))}")
    print(f"    重复行: {schema.get('duplicate_rows')}")
    print(f"    时间戳: {schema.get('snapshot_timestamp')}")

# 4. 检查上传元数据
meta_path = os.path.join(asset_dir, "data_metadata.json")
print(f"\n[3] 上传元数据存在: {os.path.exists(meta_path)}")
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"    会话 ID:  {meta.get('session_id')}")
    print(f"    上传时间: {meta.get('upload_timestamp')}")
    print(f"    原始路径: {meta.get('original_source_path')}")
    print(f"    资产路径: {meta.get('asset_path')}")
    print(f"    数据版本: {meta.get('data_version')}")
    print(f"    文件大小: {meta.get('file_size_bytes')} bytes")
    print(f"    MD5:      {meta.get('checksum_md5')}")

# 5. 检查 workflow context 中的路径是否统一
r2 = requests.get(f"{BASE}/workflow/{SID}/status")
ctx = r2.json().get("context", {})
dp = ctx.get("data_path", "")
print(f"\n[4] workflow data_path 指向资产目录: {'assets/' in dp and 'original_data.csv' in dp}")
print(f"    实际值: {dp}")

# 6. 检查 schema_snapshot 也写入了 context
schema_ctx = ctx.get("schema_snapshot")
print(f"\n[5] schema_snapshot 写入 workflow context: {schema_ctx is not None}")
if schema_ctx:
    print(f"    context 内列数: {len(schema_ctx.get('columns', []))}")

# 汇总
checks = [
    os.path.exists(original),
    os.path.exists(schema_path),
    os.path.exists(meta_path),
    "assets/" in dp and "original_data.csv" in dp,
]
passed = sum(checks)
print(f"\n===== 资产固化验证: {passed}/{len(checks)} 通过 =====")
