#!/usr/bin/env python3
"""
调试 Agent 实际运行情况
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from automl_react.agents import DataCleaningAgent
from automl_react.config import get_config_loader
from langchain_openai import ChatOpenAI

def test_data_cleaning_agent():
    """测试数据清洗 Agent"""
    
    # 创建 LLM 客户端
    config_loader = get_config_loader()
    llm_config = config_loader.get_llm_config()
    
    llm = ChatOpenAI(
        model=llm_config.get("model_name"),
        temperature=llm_config.get("temperature"),
        max_tokens=llm_config.get("max_tokens", 4096),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url")
    )
    
    # 创建 Agent（启用 verbose 模式查看详细日志）
    agent = DataCleaningAgent(
        llm=llm,
        session_id="debug_session",
        verbose=True
    )
    
    print("=" * 80)
    print("测试数据清洗 Agent")
    print("=" * 80)
    print()
    
    data_path = "/Users/cjialin/code/AutoMLByLLM/train.csv"
    
    try:
        print(f"调用 generate_cleaning_plan({data_path})...")
        print()
        
        result = agent.generate_cleaning_plan(data_path)
        
        print()
        print("=" * 80)
        print("生成结果:")
        print("=" * 80)
        print(f"结果长度: {len(result) if result else 0} 字符")
        print(f"结果内容前500字符:\n{result[:500] if result else 'None'}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def test_simple_run():
    """测试简单的 run 调用"""
    
    config_loader = get_config_loader()
    llm_config = config_loader.get_llm_config()
    
    llm = ChatOpenAI(
        model=llm_config.get("model_name"),
        temperature=llm_config.get("temperature"),
        max_tokens=llm_config.get("max_tokens", 4096),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url")
    )
    
    agent = DataCleaningAgent(
        llm=llm,
        session_id="debug_session_2",
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("测试简单 run 调用")
    print("=" * 80)
    print()
    
    # 简化提示词，要求直接输出
    simple_prompt = """请为 Kaggle House Prices 数据集生成数据清洗方案。

数据路径: /Users/cjialin/code/AutoMLByLLM/train.csv
目标列: SalePrice

请直接输出 Markdown 格式的清洗方案，包含：
1. 数据质量问题分析
2. 具体的清洗步骤
3. 预期效果

重要：直接输出方案内容，不要调用任何工具。"""

    try:
        result = agent.run(simple_prompt, stage="data_cleaning_test")
        
        print()
        print("=" * 80)
        print("Run 结果:")
        print("=" * 80)
        print(f"Success: {result.get('success')}")
        print(f"Answer 长度: {len(result.get('answer', ''))} 字符")
        print(f"Iterations: {result.get('iterations')}")
        print()
        print("Answer 前1000字符:")
        print(result.get('answer', '')[:1000])
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_data_cleaning_agent()
    test_simple_run()
