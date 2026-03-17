"""
AutoML Agent 命令行入口程序（交互模式）

该模块提供交互式的命令行界面，允许用户与AutoML系统进行
多轮对话，控制建模流程的每一步。

主要功能:
- 交互式对话界面
- 多轮对话支持
- 用户指令解析
- 步骤决策
- 支持自定义LLM API端点

使用示例:
    # 方式1: 使用自定义API（推荐）
    export LLM_API_KEY="sk-xxx"
    export LLM_BASE_URL="https://poloai.top"
    python main_interactive.py --interactive
    
    # 方式2: 使用命令行参数
    python main_interactive.py --api-key "sk-xxx" --interactive

作者: AutoML Team
"""

import argparse
import sys
import os

from automl_agent.interactive import InteractiveEngine, UserInterface


def main():
    """
    AutoML交互式入口主函数
    
    支持两种模式:
    1. 交互模式: 与用户进行多轮对话
    2. 单次模式: 接收一次指令后执行并退出
    
    支持的LLM配置方式（按优先级）:
    1. 命令行参数: --api-key, --base-url, --model
    2. 环境变量: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    3. 配置文件: config.yaml (待实现)
    """
    parser = argparse.ArgumentParser(description="AutoML Agent - 交互式智能建模")
    parser.add_argument("--interactive", "-i", action="store_true", 
                        help="启动交互式对话模式")
    parser.add_argument("--goal", type=str, help="建模目标描述")
    parser.add_argument("--data", type=str, help="数据文件路径")
    parser.add_argument("--target", type=str, help="目标列名")
    
    # LLM相关参数
    parser.add_argument("--api-key", type=str, help="LLM API密钥")
    parser.add_argument("--base-url", type=str, default="https://poloai.top", 
                        help="LLM API基础URL")
    parser.add_argument("--model", type=str, 
                        default="claude-sonnet-4-20250514-thinking",
                        help="使用的模型名称")
    parser.add_argument("--temperature", type=float, default=0, 
                        help="采样温度")
    
    args = parser.parse_args()
    
    # 配置LLM
    from llm_client import configure_llm, get_llm_client
    
    # 优先使用命令行参数，否则使用环境变量
    api_key = args.api_key or os.environ.get("LLM_API_KEY", "")
    base_url = args.base_url or os.environ.get("LLM_BASE_URL", "https://poloai.top")
    model = args.model or os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514-thinking")
    
    if not api_key:
        print("错误: 请通过 --api-key 参数或设置 LLM_API_KEY 环境变量提供API Key")
        print("\n使用示例:")
        print("  # 方式1: 环境变量")
        print("  export LLM_API_KEY='sk-xxx'")
        print("  python main_interactive.py --interactive")
        print()
        print("  # 方式2: 命令行参数")
        print("  python main_interactive.py --api-key 'sk-xxx' --interactive")
        sys.exit(1)
    
    # 配置LLM客户端
    configure_llm(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=args.temperature
    )
    
    # 获取LLM客户端
    llm = get_llm_client()
    
    # 根据参数决定运行模式
    if args.interactive or not args.goal:
        # 交互模式
        UserInterface.run_cli()
    else:
        # 单次执行模式
        engine = InteractiveEngine(llm)
        
        print("=" * 60)
        print("AutoML Agent - 智能建模服务")
        print("=" * 60)
        
        print(f"\n[1] 建模目标: {args.goal}")
        print(f"[2] 数据文件: {args.data}")
        print(f"[3] 目标列: {args.target}")
        
        # 开始并执行
        engine.start(args.goal)
        
        # 处理数据加载
        response = engine.process_input(f"加载数据 {args.data}")
        print(f"\n🤖: {response}")
        
        # 询问是否继续
        while True:
            print("\n" + "-" * 40)
            user_input = input("继续下一步? (yes/no): ").strip().lower()
            
            if user_input in ["no", "n", "否", "不"]:
                print("\n建模流程已暂停。")
                break
            
            # 继续执行下一步
            response = engine.process_input("继续")
            print(f"\n🤖: {response}")
            
            if "完成" in response or "完成" in engine.session.context.current_step:
                break


if __name__ == "__main__":
    main()
