#!/usr/bin/env python3
"""
测试特征工程代码生成和提取
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from automl_react.config import get_config_loader
from automl_react.utils.code_generator import CodeGenerator
from langchain_openai import ChatOpenAI
import json


def test_feature_engineering_code_extraction():
    """测试特征工程代码提取"""
    
    print("=" * 80)
    print("测试特征工程代码提取")
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
    
    # 测试提示词（简化版）
    prompt = """基于以下特征工程方案，生成完整的 Python 代码：

数据路径: /Users/cjialin/code/AutoMLByLLM/train.csv
目标列: SalePrice
任务类型: regression

数据信息：
- 数据形状: 1460 行 × 81 列
- 目标列: SalePrice
- 数值列: Id, MSSubClass, LotFrontage, LotArea, OverallQual, OverallCond, YearBuilt...
- 分类列: MSZoning, Street, Alley, LotShape, LandContour...

特征工程方案:
1. 缺失值处理
2. 创建面积相关特征
3. 创建房龄特征
4. 质量评分编码

要求：
1. 使用 pandas 和 numpy
2. 保存结果到: /Users/cjialin/code/AutoMLByLLM/train_features.csv
3. 包含所有必要的导入语句

请生成完整的、可执行的 Python 代码。
"""
    
    print("调用 LLM 生成代码...")
    print()
    
    # 直接调用 generate_code 方法，查看原始返回
    result = code_gen.generate_code(prompt, context={}, stage="test")
    
    print("=" * 80)
    print("生成结果:")
    print("=" * 80)
    print(f"Success: {result.success}")
    print(f"Thinking 长度: {len(result.thinking)}")
    print(f"Code 长度: {len(result.code)}")
    if result.error:
        print(f"Error: {result.error}")
    print()
    
    print("=" * 80)
    print("提取的代码 (前500字符):")
    print("=" * 80)
    print(result.code[:500] if result.code else "空")
    print()
    
    # 检查代码是否以 JSON 格式开头
    if result.code.strip().startswith('{'):
        print("⚠️ 警告: 代码仍然以 JSON 格式开头！")
        print()
        print("完整代码内容:")
        print(result.code)
    else:
        print("✅ 代码提取成功，不是 JSON 格式")
    
    # 保存提取的代码到文件
    output_file = "/Users/cjialin/code/AutoMLByLLM/test_feature_code_extracted.py"
    with open(output_file, 'w') as f:
        f.write(result.code)
    print(f"\n代码已保存到: {output_file}")


if __name__ == "__main__":
    test_feature_engineering_code_extraction()
