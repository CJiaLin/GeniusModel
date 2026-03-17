"""
自动测试脚本 - 验证对话式 AutoML 流程
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.dialog_pipeline import DialogPipeline
from llm_client import get_llm_client, configure_llm

def create_test_data():
    """创建测试数据"""
    np.random.seed(42)
    n = 100
    
    data = {
        'id': range(1, n + 1),
        'age': np.random.randint(18, 70, n),
        'income': np.random.randint(30000, 150000, n),
        'score': np.random.randn(n),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'city': np.random.choice(['Beijing', 'Shanghai', 'Guangzhou'], n),
        'target': np.random.choice([0, 1], n)
    }
    
    df = pd.DataFrame(data)
    
    missing_idx = np.random.choice(n, 20, replace=False)
    df.loc[missing_idx[:10], 'age'] = np.nan
    df.loc[missing_idx[10:], 'income'] = np.nan
    
    df.loc[missing_idx[:5], 'score'] = np.nan
    
    for col in df.columns:
        missing_rate = df[col].isna().mean()
        print(f"  {col}: {missing_rate*100:.1f}% 缺失")
    
    return df

def test_pipeline():
    """测试整个流程"""
    print("\n" + "="*50)
    print("开始自动化测试")
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
    
    print("\n[2] 创建测试数据...")
    df = create_test_data()
    print(f"✓ 测试数据创建完成: {df.shape}")
    
    print("\n[3] 初始化 Pipeline...")
    pipeline = DialogPipeline(llm)
    
    print("\n[4] 加载数据...")
    result = pipeline.load_data(df, "target")
    if not result["success"]:
        print(f"✗ 数据加载失败: {result}")
        return False
    print(f"✓ 数据加载完成: {result['profile']['shape']}")
    
    print("\n[5] 测试数据清洗思路生成...")
    cleaning_result = pipeline.generate_cleaning_thinking("清洗数据，删除高缺失率列")
    if not cleaning_result["success"]:
        print(f"✗ 清洗思路生成失败: {cleaning_result}")
        return False
    print(f"✓ 清洗思路生成完成")
    print(f"  思路内容: {cleaning_result['thinking'][:200]}...")
    
    print("\n[6] 测试数据清洗代码生成与执行...")
    pipeline.current_thinking = cleaning_result["thinking"]
    code_result = pipeline.generate_cleaning_code()
    
    print(f"\n  代码生成结果:")
    print(f"    success: {code_result.get('success')}")
    print(f"    message: {code_result.get('message')}")
    
    if code_result.get("current_code"):
        print(f"\n  生成的代码:\n{code_result['current_code']}")
    
    if code_result.get("result"):
        print(f"\n  执行结果:")
        print(f"    success: {code_result['result'].get('success')}")
        print(f"    message: {code_result['result'].get('message')}")
        print(f"    error: {code_result['result'].get('error')}")
    
    print(f"\n  数据变化:")
    print(f"    原始: {df.shape}")
    print(f"    处理后: {pipeline.data.shape if pipeline.data is not None else 'None'}")
    
    if pipeline.data is not None:
        print(f"✓ 数据已更新: {pipeline.data.shape}")
        
        before_missing = df.isna().sum().sum()
        after_missing = pipeline.data.isna().sum().sum()
        print(f"  缺失值变化: {before_missing} → {after_missing}")
        
        if after_missing < before_missing:
            print(f"✓ 缺失值已处理！")
        else:
            print(f"✗ 缺失值未处理")
    else:
        print(f"✗ pipeline.data 为 None")
    
    print("\n" + "="*50)
    print("测试特征工程环节")
    print("="*50)
    
    print("\n[7] 测试特征工程思路生成...")
    feature_result = pipeline.generate_feature_thinking(["创建年龄分组特征", "编码类别变量"])
    if not feature_result["success"]:
        print(f"✗ 特征工程思路生成失败: {feature_result}")
        return False
    print(f"✓ 特征工程思路生成完成")
    print(f"  思路内容: {feature_result['thinking'][:200]}...")
    
    print("\n[8] 测试特征工程代码生成与执行...")
    pipeline.current_thinking = feature_result["thinking"]
    feature_code_result = pipeline.generate_feature_code()
    
    print(f"\n  代码生成结果:")
    print(f"    success: {feature_code_result.get('success')}")
    print(f"    message: {feature_code_result.get('message')}")
    
    if feature_code_result.get("current_code"):
        print(f"\n  当前代码:\n{feature_code_result['current_code'][:500]}...")
    else:
        print(f"\n  实际执行的代码（从 pipeline 获取）:\n{pipeline.current_code[:500]}...")
    
    if feature_code_result.get("result"):
        print(f"\n  执行结果:")
        print(f"    success: {feature_code_result['result'].get('success')}")
        print(f"    message: {feature_code_result['result'].get('message')}")
        print(f"    error: {feature_code_result['result'].get('error')}")
    
    print(f"\n  数据变化:")
    original_cols = 7
    if pipeline.data is not None:
        after_cleaning_cols = pipeline.data.shape[1]
    else:
        after_cleaning_cols = 0
    
    if pipeline.data is not None:
        after_cleaning_cols = pipeline.data.shape[1]
    else:
        after_cleaning_cols = 0
    
    print(f"\n  数据变化:")
    print(f"    原始: ({df.shape[0]}, {df.shape[1]})")
    print(f"    清洗后: {pipeline.data.shape if pipeline.data is not None else 'N/A'}")
    print(f"    特征工程后: {pipeline.data.shape if pipeline.data is not None else 'N/A'}")
    
    # 检查 output_data
    output_df = feature_code_result.get("result", {}).get("output_data")
    if output_df is not None:
        print(f"    output_data: {output_df.shape}")
        final_cols = output_df.shape[1]
    else:
        final_cols = pipeline.data.shape[1] if pipeline.data is not None else 0
    
    # 最终判断 - 和原始数据比较
    if final_cols > df.shape[1]:
        print(f"✓ 特征数量增加！原始 {df.shape[1]} 列 → {final_cols} 列 (增加 {final_cols - df.shape[1]} 列)")
    else:
        print(f"✗ 特征数量未变化")
    
    print("\n" + "="*50)
    print("测试完成")
    print("="*50)
    
    return True

if __name__ == "__main__":
    test_pipeline()
