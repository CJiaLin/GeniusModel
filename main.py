"""
AutoML Agent 命令行入口程序

该模块是AutoML系统的命令行入口点，提供CLI界面供用户直接运行
自动化机器学习流程。用户可以通过命令行参数指定建模目标、数据路径
和目标列名，系统将自动完成从数据处理到模型评估的全流程。

主要功能:
- 命令行参数解析
- 环境配置(LLM API Key设置)
- AutoML引擎初始化和执行
- 结果展示和错误处理
- 支持自定义LLM API端点

使用示例:
    # 方式1: 使用环境变量
    export LLM_API_KEY="sk-xxx"
    python main.py --goal "预测用户是否流失" --data data.csv --target churn
    
    # 方式2: 使用命令行参数
    python main.py --goal "预测房价" --data house.csv --target price --api-key "sk-xxx"

作者: AutoML Team
"""

import argparse
import sys
import os
from typing import Optional

from automl_agent.engine import AutoMLEngine
from automl_agent.models import ModelingResult

# 导入LLM客户端
from llm_client import configure_llm, get_llm_client


def main():
    """
    AutoML Agent 命令行主函数
    
    该函数是整个程序的入口点，负责:
    1. 解析命令行参数
    2. 配置LLM（支持自定义API端点）
    3. 打印配置信息
    4. 初始化AutoML引擎
    5. 执行完整的建模流程
    6. 展示建模结果或处理错误
    
    参数:
        无(通过sys.argv获取命令行参数)
    
    返回:
        无(正常退出返回0，异常退出返回1)
    
    命令行参数:
        --goal (必填): 用户描述的建模目标，如"预测用户是否流失"
        --data (必填): 数据文件路径，支持CSV格式
        --target (必填): 目标变量列名
        --api-key (可选): LLM API密钥
        --base-url (可选): LLM API基础URL，默认 https://poloai.top
        --model (可选): 使用的模型名称
        --use-llm-features (可选): 是否使用LLM特征生成
        --n-feature-suggestions (可选): LLM特征建议数量
    
    环境变量:
        LLM_API_KEY: LLM API密钥
        LLM_BASE_URL: LLM API基础URL
        LLM_MODEL: 模型名称
    
    异常处理:
        捕获所有异常并打印错误信息，然后以退出码1终止程序
    """
    # 创建命令行参数解析器，设置程序描述信息
    parser = argparse.ArgumentParser(description="AutoML Agent - 智能建模服务")
    
    # 添加必填参数：建模目标描述
    parser.add_argument("--goal", type=str, required=True, help="建模目标描述")
    # 添加必填参数：数据文件路径
    parser.add_argument("--data", type=str, required=True, help="数据文件路径")
    # 添加必填参数：目标列名
    parser.add_argument("--target", type=str, required=True, help="目标列名")
    
    # LLM相关参数
    parser.add_argument("--api-key", type=str, help="LLM API密钥")
    parser.add_argument("--base-url", type=str, default="https://poloai.top", 
                        help="LLM API基础URL")
    parser.add_argument("--model", type=str, 
                        default="claude-sonnet-4-20250514-thinking",
                        help="使用的模型名称")
    parser.add_argument("--temperature", type=float, default=0, 
                        help="采样温度")
    
    # AutoML相关参数
    parser.add_argument("--use-llm-features", action="store_true",
                        help="使用LLM驱动的特征生成")
    parser.add_argument("--n-feature-suggestions", type=int, default=10,
                        help="LLM特征建议数量")
    parser.add_argument("--no-interactive", action="store_true",
                        help="禁用交互模式，不等待用户确认")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查API Key
    api_key = args.api_key or os.environ.get("LLM_API_KEY", "")
    if not api_key:
        print("错误: 请通过 --api-key 参数或设置 LLM_API_KEY 环境变量提供API Key")
        print("\n使用示例:")
        print("  # 方式1: 环境变量")
        print("  export LLM_API_KEY='sk-xxx'")
        print("  python main.py --goal '预测流失' --data data.csv --target churn")
        print()
        print("  # 方式2: 命令行参数")
        print("  python main.py --goal '预测流失' --data data.csv --target churn --api-key 'sk-xxx'")
        sys.exit(1)
    
    # 配置LLM
    configure_llm(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature
    )
    
    # 获取LLM客户端
    llm = get_llm_client()
    
    # 打印程序标题和分隔线
    print("=" * 60)
    print("AutoML Agent - 智能建模服务")
    print("=" * 60)
    
    # 打印用户配置信息
    print(f"\n[1] 建模目标: {args.goal}")
    print(f"[2] 数据文件: {args.data}")
    print(f"[3] 目标列: {args.target}")
    print(f"[4] 使用LLM特征生成: {args.use_llm_features}")
    print(f"[5] 模型: {args.model}")
    
    # 创建AutoML引擎实例
    engine = AutoMLEngine(llm)
    
    print("\n开始建模流程...\n")
    
    try:
        # 执行完整的AutoML流程
        result = engine.run(
            args.goal, 
            args.data, 
            args.target,
            use_llm_features=args.use_llm_features,
            n_feature_suggestions=args.n_feature_suggestions,
            interactive=not args.no_interactive
        )
        
        # 打印完成信息和分隔线
        print("\n" + "=" * 60)
        print("建模完成!")
        print("=" * 60)
        
        # 打印模型评估指标
        print(f"\n模型评估指标:")
        for metric, value in result.metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 打印训练时间
        print(f"\n训练时间: {result.training_time:.2f}秒")
        
        # 如果使用了LLM特征生成，打印生成的特征代码
        if args.use_llm_features:
            codes = result.artifacts.get("generated_feature_code", [])
            if codes:
                print(f"\n生成了 {len(codes)} 个特征")
                print("\n特征生成代码:")
                for i, code in enumerate(codes, 1):
                    print(f"\n--- 特征 {i} ---")
                    print(code[:500] + "..." if len(code) > 500 else code)
        
    except Exception as e:
        # 捕获异常并打印错误信息
        print(f"\n建模失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # 当直接运行此脚本时，调用main函数
    main()
