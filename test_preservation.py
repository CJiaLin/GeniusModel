"""
测试代码和数据中间态保留
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.dialog_pipeline import DialogPipeline
from llm_client import get_llm_client, configure_llm

def test_data_and_code_preservation():
    """测试数据中间态和代码保留"""
    print("\n" + "="*60)
    print("测试: 数据中间态保留 & 代码汇总")
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
    print(f"✓ 原始数据: {df.shape}")
    
    print("\n[3] 初始化 Pipeline...")
    pipeline = DialogPipeline(llm)
    result = pipeline.load_data(df, "SalePrice", "房价预测")
    
    if not result["success"]:
        print(f"✗ 加载失败")
        return
    print(f"✓ 加载完成")
    
    # ========== 步骤1: 清洗 ==========
    print("\n" + "-"*50)
    print("[4] 步骤1: 数据清洗")
    print("-"*50)
    
    cleaning_result = pipeline.generate_cleaning_thinking("处理缺失值")
    pipeline.current_thinking = cleaning_result["thinking"]
    cleaning_code_result = pipeline.generate_cleaning_code()
    
    print(f"  清洗后数据: {pipeline.data.shape}")
    print(f"  代码块数: {len(pipeline.code_blocks)}")
    print(f"  最后代码块 executed: {pipeline.code_blocks[-1].executed if pipeline.code_blocks else 'N/A'}")
    
    # 检查 executor 中的数据
    print(f"\n  CodeExecutor local_vars 中的数据:")
    print(f"    keys: {list(pipeline.code_executor.local_vars.keys())}")
    if 'df' in pipeline.code_executor.local_vars:
        print(f"    df shape: {pipeline.code_executor.local_vars['df'].shape}")
    if 'df_clean' in pipeline.code_executor.local_vars:
        print(f"    df_clean shape: {pipeline.code_executor.local_vars['df_clean'].shape}")
    
    # ========== 步骤2: 特征工程 ==========
    print("\n" + "-"*50)
    print("[5] 步骤2: 特征工程")
    print("-"*50)
    
    feature_result = pipeline.generate_feature_thinking(["创建面积特征"])
    pipeline.current_thinking = feature_result["thinking"]
    feature_code_result = pipeline.generate_feature_code()
    
    print(f"  特征工程后数据: {pipeline.data.shape}")
    print(f"  代码块数: {len(pipeline.code_blocks)}")
    
    # 检查 executor 中的数据
    print(f"\n  CodeExecutor local_vars 中的数据:")
    print(f"    keys: {list(pipeline.code_executor.local_vars.keys())}")
    if 'df_featured' in pipeline.code_executor.local_vars:
        print(f"    df_featured shape: {pipeline.code_executor.local_vars['df_featured'].shape}")
    
    # ========== 步骤3: 模型训练 ==========
    print("\n" + "-"*50)
    print("[6] 步骤3: 模型训练")
    print("-"*50)
    
    model_result = pipeline.generate_model_thinking("使用随机森林回归")
    pipeline.current_thinking = model_result["thinking"]
    model_code_result = pipeline.generate_model_code()
    
    print(f"  模型训练后数据: {pipeline.data.shape}")
    print(f"  代码块数: {len(pipeline.code_blocks)}")
    
    # ========== 代码汇总测试 ==========
    print("\n" + "="*60)
    print("[7] 测试代码汇总导出")
    print("="*60)
    
    exported_code = pipeline.export_code()
    print(f"\n导出代码总长度: {len(exported_code)} 字符")
    print(f"代码块数量: {len(pipeline.code_blocks)}")
    
    print(f"\n各代码块:")
    for i, block in enumerate(pipeline.code_blocks):
        print(f"\n  --- 代码块 {i+1}: {block.name} ---")
        print(f"  executed: {block.executed}")
        print(f"  代码长度: {len(block.code)} 字符")
        print(f"  描述: {block.description[:80]}...")
    
    print(f"\n" + "="*60)
    print("导出代码预览:")
    print("="*60)
    print(exported_code[:1500])
    print("\n... (中间省略) ...")
    print(exported_code[-500:])
    
    # ========== 数据下载测试 ==========
    print("\n" + "="*60)
    print("[8] 测试数据下载")
    print("="*60)
    
    download_link = pipeline.get_data_download_link()
    if download_link:
        print(f"✓ 数据下载链接有效")
        print(f"  数据大小: {len(download_link)} 字符 (CSV)")
    else:
        print(f"✗ 数据下载链接无效")
    
    print("\n" + "="*60)
    print("✓ 测试完成")
    print("="*60)

if __name__ == "__main__":
    test_data_and_code_preservation()
