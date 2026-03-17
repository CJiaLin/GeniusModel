"""
完整流程测试 - train.csv (房价预测)
测试: 数据清洗 → 特征工程 → 模型训练
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.dialog_pipeline import DialogPipeline
from llm_client import get_llm_client, configure_llm

def test_full_pipeline():
    """测试完整流程"""
    print("\n" + "="*60)
    print("完整流程测试 - train.csv (房价预测)")
    print("="*60)
    
    print("\n[1] 配置 LLM...")
    from llm_client import load_config_from_file
    _file_config = load_config_from_file()
    
    if not _file_config.get("api_key"):
        raise ValueError("请在 config.yaml 中配置 LLM API Key")
    
    configure_llm(
        base_url=_file_config.get("base_url", "https://fast.poloai.top"),
        api_key=_file_config.get("api_key"),
        model=_file_config.get("model", "claude-sonnet-4-20250514-thinking")
    )
    llm = get_llm_client()
    print("✓ LLM 配置完成")
    
    print("\n[2] 加载数据...")
    df = pd.read_csv("train.csv")
    print(f"✓ 数据: {df.shape[0]}行 × {df.shape[1]}列")
    
    print("\n[3] 初始化 Pipeline...")
    pipeline = DialogPipeline(llm)
    
    target_column = "SalePrice"
    modeling_scenario = "预测每套房子的售价，采用RMSE评估预测效果"
    
    result = pipeline.load_data(df, target_column, modeling_scenario)
    if not result["success"]:
        print(f"✗ 数据加载失败: {result}")
        return False
    print(f"✓ 数据加载完成")
    
    # ========== 数据清洗 ==========
    print("\n" + "-"*50)
    print("[4] 数据清洗")
    print("-"*50)
    
    cleaning_result = pipeline.generate_cleaning_thinking(
        "处理缺失值，删除高缺失率列，处理异常值，转换数据类型"
    )
    if not cleaning_result["success"]:
        print(f"✗ 清洗思路生成失败")
        return False
    print(f"✓ 清洗思路生成完成")
    
    pipeline.current_thinking = cleaning_result["thinking"]
    cleaning_code_result = pipeline.generate_cleaning_code()
    
    print(f"  清洗结果: {cleaning_code_result.get('success')}")
    if pipeline.data is not None:
        print(f"  数据变化: {df.shape[1]}列 → {pipeline.data.shape[1]}列")
    
    # ========== 特征工程 ==========
    print("\n" + "-"*50)
    print("[5] 特征工程")
    print("-"*50)
    
    feature_result = pipeline.generate_feature_thinking([
        "创建面积相关特征(总面积、庭院面积比)",
        "创建年份相关特征(房龄、翻新年限)",
        "编码类别变量"
    ])
    if not feature_result["success"]:
        print(f"✗ 特征思路生成失败")
        return False
    print(f"✓ 特征思路生成完成")
    
    pipeline.current_thinking = feature_result["thinking"]
    feature_code_result = pipeline.generate_feature_code()
    
    print(f"  特征工程结果: {feature_code_result.get('success')}")
    if pipeline.data is not None:
        print(f"  数据变化: {df.shape[1]}列 → {pipeline.data.shape[1]}列")
    
    # ========== 模型训练 ==========
    print("\n" + "-"*50)
    print("[6] 模型训练")
    print("-"*50)
    
    model_result = pipeline.generate_model_thinking(
        "使用回归模型预测房价，采用RMSE评估"
    )
    if not model_result["success"]:
        print(f"✗ 模型思路生成失败")
        return False
    print(f"✓ 模型思路生成完成")
    print(f"  思路: {model_result['thinking'][:200]}...")
    
    pipeline.current_thinking = model_result["thinking"]
    model_code_result = pipeline.generate_model_code()
    
    print(f"\n  模型训练结果:")
    print(f"    success: {model_code_result.get('success')}")
    print(f"    message: {model_code_result.get('message')}")
    
    if model_code_result.get("result"):
        result_data = model_code_result["result"]
        print(f"    output_data: {result_data.get('output_data')}")
    
    # ========== 总结 ==========
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"  原始数据: {df.shape}")
    if pipeline.data is not None:
        print(f"  最终数据: {pipeline.data.shape}")
    print(f"  代码块数量: {len(pipeline.code_blocks)}")
    
    for i, block in enumerate(pipeline.code_blocks):
        print(f"  - {i+1}. {block.name}: {'✓' if block.executed else '✗'}")
    
    print("\n" + "="*60)
    print("✓ 测试完成")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_full_pipeline()
