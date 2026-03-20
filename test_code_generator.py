#!/usr/bin/env python3
"""
测试新的代码生成器
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from automl_react.config import get_config_loader
from automl_react.utils.code_generator import CodeGenerator
from langchain_openai import ChatOpenAI

def test_code_generator():
    """测试代码生成器"""
    
    print("=" * 80)
    print("测试代码生成器")
    print("=" * 80)
    print()
    
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
    
    # 创建代码生成器
    code_gen = CodeGenerator(llm=llm)
    
    # 测试提示词
    prompt = """请生成一个简单的 Python 代码，实现以下功能：
1. 读取 CSV 文件: /Users/cjialin/code/AutoMLByLLM/train.csv
2. 显示数据的前5行
3. 显示数据的基本统计信息
4. 保存清洗后的数据到: /Users/cjialin/code/AutoMLByLLM/train_test_cleaned.csv

要求：
- 使用 pandas 库
- 包含必要的导入语句
- 代码必须完整可执行
"""
    
    print("测试代码生成...")
    print()
    
    # 准备上下文
    context = {
        "data_path": "/Users/cjialin/code/AutoMLByLLM/train.csv",
        "cleaned_data_path": "/Users/cjialin/code/AutoMLByLLM/train_test_cleaned.csv"
    }
    
    # 生成并验证代码
    code, exec_result = code_gen.generate_code_with_validation(
        prompt=prompt,
        context=context,
        stage="test",
        required_outputs=[]
    )
    
    print("=" * 80)
    print("生成的代码:")
    print("=" * 80)
    print(code[:1000] if len(code) > 1000 else code)
    print()
    
    print("=" * 80)
    print("执行结果:")
    print("=" * 80)
    print(f"Success: {exec_result.success}")
    print(f"Output: {exec_result.output[:500] if exec_result.output else 'None'}")
    if exec_result.error:
        print(f"Error: {exec_result.error[:500]}")
    print()
    
    # 检查文件是否生成
    import os
    output_file = "/Users/cjialin/code/AutoMLByLLM/train_test_cleaned.csv"
    if os.path.exists(output_file):
        print(f"✅ 文件已生成: {output_file}")
    else:
        print(f"❌ 文件未生成: {output_file}")


if __name__ == "__main__":
    test_code_generator()
