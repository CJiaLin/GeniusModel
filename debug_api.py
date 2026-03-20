#!/usr/bin/env python3
"""
模拟 API 调用测试
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from automl_react.agents import DataCleaningAgent
from automl_react.config import get_config_loader
from automl_react.confirmation import ConfirmationManager
from langchain_openai import ChatOpenAI

def simulate_api_call():
    """模拟 API 的数据清洗阶段调用"""
    
    print("=" * 80)
    print("模拟 API 调用 - 数据清洗阶段")
    print("=" * 80)
    print()
    
    # 创建 LLM 客户端（与 API 相同的方式）
    config_loader = get_config_loader()
    llm_config = config_loader.get_llm_config()
    
    llm = ChatOpenAI(
        model=llm_config.get("model_name"),
        temperature=llm_config.get("temperature"),
        max_tokens=llm_config.get("max_tokens", 4096),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url")
    )
    
    # 创建 Agent（与 API 相同的方式）
    agents = {}
    agents["cleaning"] = DataCleaningAgent(
        llm=llm,
        session_id="test_session"
    )
    
    # 创建确认管理器
    confirmation_manager = ConfirmationManager()
    
    # 模拟 API 调用参数
    data_path = "/Users/cjialin/code/AutoMLByLLM/train.csv"
    
    print(f"数据路径: {data_path}")
    print(f"Agent 类型: {type(agents['cleaning'])}")
    print()
    
    # 调用 generate_cleaning_plan（与 API 相同的方式）
    agent = agents["cleaning"]
    
    try:
        print("调用 agent.generate_cleaning_plan()...")
        result = agent.generate_cleaning_plan(data_path)
        
        print()
        print(f"返回结果类型: {type(result)}")
        print(f"返回结果长度: {len(result) if result else 0} 字符")
        print()
        
        if result:
            print("✅ 成功获取清洗方案")
            print(f"前500字符:\n{result[:500]}")
            
            # 创建确认点
            confirmation_point = confirmation_manager.create_confirmation_point(
                stage="data_cleaning",
                proposal=result
            )
            
            print()
            print(f"确认点 ID: {confirmation_point.point_id}")
            print(f"确认点阶段: {confirmation_point.stage}")
            print(f"确认点方案长度: {len(confirmation_point.proposal)} 字符")
            
            # 模拟 API 返回
            api_response = {
                "success": True,
                "stage": "data_cleaning",
                "proposal": result,
                "requires_confirmation": True,
                "confirmation_id": confirmation_point.point_id
            }
            
            print()
            print("模拟 API 响应:")
            print(f"  success: {api_response['success']}")
            print(f"  stage: {api_response['stage']}")
            print(f"  proposal length: {len(api_response['proposal'])} 字符")
            print(f"  requires_confirmation: {api_response['requires_confirmation']}")
            print(f"  confirmation_id: {api_response['confirmation_id']}")
            
        else:
            print("❌ 返回结果为空！")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_calls():
    """测试多次调用，检查 Agent 状态"""
    
    print("\n" + "=" * 80)
    print("测试多次调用 Agent")
    print("=" * 80)
    print()
    
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
        session_id="test_session_2"
    )
    
    data_path = "/Users/cjialin/code/AutoMLByLLM/train.csv"
    
    # 第一次调用
    print("第一次调用 generate_cleaning_plan()...")
    result1 = agent.generate_cleaning_plan(data_path)
    print(f"  结果长度: {len(result1) if result1 else 0} 字符")
    print(f"  agent.cleaning_plan: {len(agent.cleaning_plan) if agent.cleaning_plan else 0} 字符")
    
    # 第二次调用（使用相同 agent）
    print("\n第二次调用 generate_cleaning_plan()...")
    result2 = agent.generate_cleaning_plan(data_path)
    print(f"  结果长度: {len(result2) if result2 else 0} 字符")
    print(f"  agent.cleaning_plan: {len(agent.cleaning_plan) if agent.cleaning_plan else 0} 字符")


if __name__ == "__main__":
    simulate_api_call()
    test_multiple_calls()
