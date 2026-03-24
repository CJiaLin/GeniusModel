"""
测试特征工程流程
"""
import os
import sys
sys.path.insert(0, '/Users/cjialin/code/AutoMLByLLM')

from automl_react.agents.feature_engineering_agent import FeatureEngineeringAgent
from automl_react.assets.asset_manager import AssetManager
from langchain_openai import ChatOpenAI

# 配置
session_id = "test_new_workflow_1774319996"
data_path = "/Users/cjialin/code/AutoMLByLLM/train.csv"
cleaned_data_path = f"/Users/cjialin/code/AutoMLByLLM/assets/{session_id}/data/cleaned_data.csv"
target_column = "SalePrice"
task_type = "regression"

# 初始化 LLM (使用配置文件中的设置)
llm = ChatOpenAI(
    model="kimi-k2.5",
    temperature=1.0,
    max_tokens=4096,
    base_url="https://api.moonshot.cn/v1",
    api_key="sk-owzJCshJL25ysNHOWcgwjND8I53AoiTarpB3XeHl33TYjSPt"
)

# 初始化 Asset Manager
asset_manager = AssetManager(session_id=session_id)

# 初始化 FeatureEngineeringAgent
agent = FeatureEngineeringAgent(
    llm=llm,
    asset_manager=asset_manager,
    data_path=cleaned_data_path,  # 使用清洗后的数据
    target_column=target_column,
    task_type=task_type
)

print("=" * 80)
print("测试特征工程流程")
print("=" * 80)

# 1. 生成特征工程方案
print("\n1. 生成特征工程方案...")
plan = agent.generate_feature_plan(cleaned_data_path, target_column, task_type)
print(f"特征工程方案长度: {len(plan)} 字符")

# 2. 生成特征工程代码并执行
print("\n2. 生成特征工程代码并执行...")
code = agent.generate_feature_code()
print(f"特征工程代码长度: {len(code) if code else 0} 字符")

# 3. 检查特征数据是否生成
features_data_path = agent.features_data_path
if features_data_path and os.path.exists(features_data_path):
    print(f"\n✅ 特征数据已生成: {features_data_path}")
    import pandas as pd
    df = pd.read_csv(features_data_path)
    print(f"数据形状: {df.shape}")
    
    # 4. 计算特征指标
    print("\n3. 计算特征指标...")
    metrics_result = agent.calculate_feature_metrics()
    print(f"特征指标计算结果: {metrics_result}")
else:
    print(f"\n❌ 特征数据未生成: {features_data_path}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
