"""
使用真实数据集测试 - train.csv (房价预测)
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.dialog_pipeline import DialogPipeline
from llm_client import get_llm_client, configure_llm

def test_with_real_data():
    """使用真实数据测试"""
    print("\n" + "="*50)
    print("使用 train.csv 测试 - 房价预测")
    print("="*50)
    
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
    print(f"✓ 数据加载完成: {df.shape}")
    
    # 显示数据信息
    print(f"\n数据信息:")
    print(f"  - 行数: {df.shape[0]}")
    print(f"  - 列数: {df.shape[1]}")
    print(f"  - 数值列: {len(df.select_dtypes(include=['number']).columns)}")
    print(f"  - 类别列: {len(df.select_dtypes(include=['object']).columns)}")
    
    # 缺失值统计
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\n缺失值统计 (前10):")
    for col, cnt in missing.head(10).items():
        rate = cnt / df.shape[0] * 100
        print(f"  - {col}: {cnt} ({rate:.1f}%)")
    
    print("\n[3] 初始化 Pipeline...")
    pipeline = DialogPipeline(llm)
    
    target_column = "SalePrice"
    modeling_scenario = "预测每套房子的售价，采用RMSE评估预测效果"
    
    print("\n[4] 加载数据到 Pipeline...")
    result = pipeline.load_data(df, target_column, modeling_scenario)
    if not result["success"]:
        print(f"✗ 数据加载失败: {result}")
        return False
    print(f"✓ 数据加载完成")
    
    print("\n[5] 测试数据清洗思路生成...")
    cleaning_result = pipeline.generate_cleaning_thinking(
        "数据清洗：处理缺失值，删除高缺失率列，处理异常值"
    )
    if not cleaning_result["success"]:
        print(f"✗ 清洗思路生成失败: {cleaning_result}")
        return False
    print(f"✓ 清洗思路生成完成")
    print(f"  思路: {cleaning_result['thinking'][:300]}...")
    
    print("\n[6] 测试数据清洗代码生成与执行...")
    pipeline.current_thinking = cleaning_result["thinking"]
    code_result = pipeline.generate_cleaning_code()
    
    print(f"\n  代码生成结果:")
    print(f"    success: {code_result.get('success')}")
    print(f"    message: {code_result.get('message')}")
    
    if code_result.get("result"):
        print(f"    执行: {code_result['result'].get('success')}")
        if code_result['result'].get('output_data') is not None:
            output_df = code_result['result']['output_data']
            print(f"    输出数据: {output_df.shape}")
    
    print(f"\n  数据变化:")
    print(f"    原始: {df.shape}")
    if pipeline.data is not None:
        print(f"    清洗后: {pipeline.data.shape}")
        before_missing = df.isna().sum().sum()
        after_missing = pipeline.data.isna().sum().sum()
        print(f"    缺失值: {before_missing} → {after_missing}")
    
    print("\n" + "="*50)
    print("测试完成")
    print("="*50)
    
    return True

if __name__ == "__main__":
    test_with_real_data()
