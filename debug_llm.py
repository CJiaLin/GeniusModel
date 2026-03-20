#!/usr/bin/env python3
"""
调试 LLM 响应
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from automl_react.config import get_config_loader
from automl_react.skills_loader import get_skill_loader

def test_llm_direct():
    """直接测试 LLM 调用"""
    config_loader = get_config_loader()
    llm_config = config_loader.get_llm_config()
    
    print("LLM 配置:")
    print(f"  Provider: {llm_config.get('provider')}")
    print(f"  Model: {llm_config.get('model_name')}")
    print(f"  Temperature: {llm_config.get('temperature')}")
    print(f"  Base URL: {llm_config.get('base_url')}")
    print()
    
    # 创建 LLM 客户端
    provider = llm_config.get("provider", "openai")
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=llm_config.get("model_name"),
            temperature=llm_config.get("temperature"),
            max_tokens=llm_config.get("max_tokens", 4096),
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url")
        )
    
    # 测试直接调用
    test_prompt = """请为以下数据生成数据清洗方案：

数据路径: /Users/cjialin/code/AutoMLByLLM/train.csv

请生成 Markdown 格式的清洗方案，包括：
1. 数据质量问题分析
2. 清洗步骤
3. 预期效果

请直接输出方案内容，不需要使用 ReAct 格式。"""

    print("测试直接调用 LLM...")
    print(f"Prompt 长度: {len(test_prompt)} 字符")
    print()
    
    try:
        response = llm.invoke(test_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        print(f"响应长度: {len(content)} 字符")
        print()
        print("=" * 80)
        print("LLM 响应内容:")
        print("=" * 80)
        print(content[:2000] if len(content) > 2000 else content)
        print("=" * 80)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def test_react_format():
    """测试 ReAct 格式的调用"""
    config_loader = get_config_loader()
    llm_config = config_loader.get_llm_config()
    
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=llm_config.get("model_name"),
        temperature=llm_config.get("temperature"),
        max_tokens=llm_config.get("max_tokens", 4096),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url")
    )
    
    # ReAct 格式提示词
    react_prompt = """你是一位专业的数据清洗专家。

## 可用工具

工具: load_data
  描述: 加载数据文件
  参数: {"file_path": "string"}

工具: analyze_data
  描述: 分析数据质量
  参数: {"data": "object"}

## ReAct 格式说明

你必须按照以下格式进行思考和行动：

思考: 分析当前情况，决定下一步行动
行动: 工具名称
行动输入: {"参数名": "参数值"}
观察: 等待工具执行结果

当任务完成时，输出：
思考: 任务已完成
最终答案: 你的回答

## 执行历史



## 当前任务

用户输入: 请为以下数据生成详细的清洗方案：

数据路径: /Users/cjialin/code/AutoMLByLLM/train.csv



请按照 ReAct 格式进行思考和行动：
"""

    print("\n测试 ReAct 格式调用 LLM...")
    print(f"Prompt 长度: {len(react_prompt)} 字符")
    print()
    
    try:
        response = llm.invoke(react_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        print(f"响应长度: {len(content)} 字符")
        print()
        print("=" * 80)
        print("LLM 响应内容:")
        print("=" * 80)
        print(content[:2000] if len(content) > 2000 else content)
        print("=" * 80)
        
        # 测试解析
        import re
        match = re.search(r'(?:最终答案|Final Answer)[:：]\s*(.+)', content, re.DOTALL | re.IGNORECASE)
        if match:
            print("\n✅ 匹配到 '最终答案' 格式")
            print(f"解析结果长度: {len(match.group(1).strip())} 字符")
        else:
            print("\n❌ 未匹配到 '最终答案' 格式")
            if content and len(content.strip()) > 10:
                print(f"但内容非空 ({len(content.strip())} 字符)，可以作为直接返回")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 80)
    print("LLM 调试测试")
    print("=" * 80)
    print()
    
    test_llm_direct()
    print("\n" + "=" * 80 + "\n")
    test_react_format()
