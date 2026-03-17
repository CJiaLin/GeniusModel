"""
AutoMLEngine - AutoML自动化机器学习引擎

该模块提供了AutoML系统的核心编排引擎，负责协调各个Agent完成
从数据加载到模型训练评估的完整机器学习流程。

主要功能:
- 意图解析与建模规划
- 数据加载、探索与质量分析
- 数据清洗与预处理
- 特征工程与编码（支持传统方式和LLM驱动方式）
- 模型训练与评估

依赖:
- langchain: 语言模型接口
- pandas: 数据处理
- sklearn: 机器学习工具

作者: AutoML Team
"""

from typing import Any, Optional
import pandas as pd

from .models import ModelingGoal, ModelingPlan, ModelingResult, ProcessState
from .enums import ModelingTaskType, ProcessStatus

# agents在顶层目录，需要特殊处理
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.modeling_planner import ModelingPlanner
from agents.data_agent import DataAgent
from agents.feature_engineer import FeatureEngineerAgent, FeatureEngineeringResult
from agents.model_agent import ModelAgent

# 导入LLM客户端（支持自定义API端点）
try:
    from llm_client import get_llm_client, configure_llm, CustomLLMClient
except ImportError:
    from langchain_openai import ChatOpenAI
    CustomLLMClient = None


class AutoMLEngine:
    """
    AutoML自动化机器学习引擎主类
    
    负责整个机器学习流水线的编排和执行，协调ModelingPlanner、
    DataAgent、FeatureEngineerAgent和ModelAgent四个核心组件，
    实现从用户意图到模型输出的端到端自动化。
    
    属性:
        llm: OpenAI语言模型实例，用于Agent的推理和决策
        planner: 建模规划器，负责意图解析和任务规划
        data_agent: 数据处理Agent，负责数据加载、探索、清洗
        feature_agent: 特征工程Agent，负责特征生成、编码、选择
        model_agent: 模型Agent，负责模型选择、训练、评估
        state: 流程状态追踪对象
        plan: 当前执行的建模计划
    
    示例:
        >>> from langchain_openai import ChatOpenAI
        >>> from automl_agent.engine import AutoMLEngine
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> engine = AutoMLEngine(llm)
        >>> result = engine.run("预测用户是否流失", "data.csv", "churn")
    """

    def __init__(self, llm: Optional[Any] = None):
        """
        初始化AutoML引擎
        
        参数:
            llm: 语言模型实例，如果为None则使用自定义LLM客户端
        
        创建各个Agent实例并初始化流程状态:
        - ModelingPlanner: 解析用户建模意图
        - DataAgent: 处理数据加载和清洗
        - FeatureEngineerAgent: 执行特征工程
        - ModelAgent: 进行模型训练和评估
        """
        # 优先使用传入的LLM，否则使用自定义客户端
        if llm is not None:
            self.llm = llm
        elif CustomLLMClient is not None:
            # 使用自定义LLM客户端（默认配置）
            self.llm = get_llm_client()
        else:
            # 后备到langchain_openai
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        self.planner = ModelingPlanner(self.llm)
        self.data_agent = DataAgent(self.llm)
        self.feature_agent = FeatureEngineerAgent(self.llm)
        self.model_agent = ModelAgent(self.llm)
        self.state = ProcessState()
        self.plan: Optional[ModelingPlan] = None
        self.data_path = ""
    
    def _save_feature_code_to_file(self, executed_code: list, output_dir: str = "testsample"):
        """
        保存特征生成代码到.py文件
        
        Args:
            executed_code: 执行的代码列表
            output_dir: 输出目录
        """
        import os
        import ast
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/generated_features_{timestamp}.py"
        
        header = f'''"""
AutoML Generated Feature Engineering Code
Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import pandas as pd
import numpy as np

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    自动生成的特征工程函数
    
    Args:
        df: 输入数据框
        
    Returns:
        pd.DataFrame: 添加新特征后的数据框
    """
    df = df.copy()
'''
        
        code_lines = [header]
        
        for i, code in enumerate(executed_code):
            if code and code.strip():
                # 格式化代码
                try:
                    ast.parse(code)
                    formatted = ast.unparse(ast.parse(code)) if hasattr(ast, 'unparse') else code
                except:
                    formatted = code
                
                code_lines.append(f"\n    # Feature {i+1}")
                # 缩进代码
                for line in formatted.split('\n'):
                    code_lines.append(f"    {line}")
        
        code_lines.append("\n    return df")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(code_lines))
        
        print(f"\n💾 特征生成代码已保存到: {filename}")

    def run(self, user_goal: str, data_path: str, target_column: str, 
            use_llm_features: bool = False, n_feature_suggestions: int = 10,
            interactive: bool = True, data_description_path: str = None,
            confirm_callback = None) -> ModelingResult:
        """
        执行完整的 AutoML 流程
        
        参数:
            user_goal: 用户建模目标
            data_path: 数据文件路径
            target_column: 目标列名
            use_llm_features: 是否使用 LLM 驱动的特征工程
            n_feature_suggestions: LLM 特征建议数量
            interactive: 是否启用交互模式（等待用户确认）
            data_description_path: 数据描述文件路径
            confirm_callback: 确认回调函数，签名 (step_name, info) -> bool，返回 False 表示取消
        流程包括:
        1. 解析用户建模意图
        2. 加载和探索数据
        3. 分析数据质量
        4. 清洗数据
        5. 特征工程(生成特征和编码) - 支持传统方式或LLM驱动方式
        6. 模型训练和评估
        
        参数:
            user_goal: 用户描述的建模目标，如"预测用户是否流失"
            data_path: 数据文件的路径（可以是目录或文件路径）
            target_column: 目标变量列名
            use_llm_features: 是否使用LLM驱动的特征生成，默认False使用传统方式
            n_feature_suggestions: LLM特征生成时生成的特征建议数量，默认10
            interactive: 是否启用交互模式，每步等待用户确认，默认True
            data_description_path: 数据描述文件路径（可选），用于提供数据集的详细说明
        
        返回:
            ModelingResult: 包含模型训练结果、评估指标等信息的对象
        
        异常:
            ValueError: 当目标列不存在于数据中时抛出
            RuntimeError: 当流程中任何步骤失败时抛出
        """
        
        def wait_user_confirm(step_name: str, info: str = ""):
            """等待用户确认"""
            if not interactive:
                # 非交互模式，直接返回
                return
            
            # 交互模式下的确认逻辑
            if confirm_callback:
                # 使用回调（Web 模式）
                if not confirm_callback(step_name, info):
                    raise RuntimeError(f"用户取消了流程：{step_name}")
            else:
                # 使用终端输入（终端模式）
                print(f"\n{'='*60}")
                print(f"📌 步骤：{step_name}")
                if info:
                    print(f"📊 {info}")
                print(f"{'='*60}")
                user_input = input("按回车键继续，或输入 'q' 退出...")
                if user_input.lower() == 'q':
                    raise RuntimeError("用户取消了流程")
        
        self.state.status = ProcessStatus.RUNNING
        self.state.current_step = "解析建模意图"
        
        # 读取数据描述文件（如果提供）
        data_description = ""
        if data_description_path:
            import os
            if os.path.exists(data_description_path):
                try:
                    with open(data_description_path, 'r', encoding='utf-8') as f:
                        data_description = f.read()
                    print(f"\n📄 已加载数据描述文件: {data_description_path}")
                except Exception as e:
                    print(f"\n⚠️ 读取数据描述文件失败: {e}")
        
        # 第一步：解析用户建模意图，确定任务类型(分类/回归/聚类等)
        print("\n📌 步骤1: 解析建模意图")
        goal = self.planner.parse_intent(user_goal, data_description=data_description)
        goal.target_column = target_column
        print(f"  任务类型: {goal.task_type.value}")
        print(f"  目标列: {target_column}")
        
        wait_user_confirm("加载数据", "正在加载数据...")
        
        self.state.current_step = "加载和探索数据"
        # 第二步：加载数据并进行初步探索，获取数据基本信息
        data_profile = self.data_agent.load_data(data_path)
        print(f"\n✅ 数据加载完成!")
        print(f"  数据形状: {data_profile.shape}")
        print(f"  列数: {len(data_profile.columns)}")
        
        wait_user_confirm("分析数据质量", "正在分析...")
        
        self.state.current_step = "分析数据质量"
        # 第三步：分析数据质量，检测缺失值、异常值、重复值等
        quality_report = self.data_agent.analyze_quality()
        print(f"\n✅ 数据质量分析完成!")
        # 修复：处理嵌套字典的情况
        try:
            if quality_report.missing_analysis:
                if isinstance(list(quality_report.missing_analysis.values())[0], dict):
                    total_missing = sum(sum(v.values()) for v in quality_report.missing_analysis.values())
                else:
                    total_missing = sum(quality_report.missing_analysis.values())
            else:
                total_missing = 0
        except:
            total_missing = "未知"
        print(f"  缺失值总数: {total_missing}")
        print(f"  重复行数: {quality_report.duplicate_count}")
        
        wait_user_confirm("清洗数据", "正在清洗...")
        
        self.state.current_step = "清洗数据"
        # 第四步：根据质量报告清洗数据，处理缺失值和异常值
        df_clean = self.data_agent.clean_data()
        print(f"\n✅ 数据清洗完成!")
        print(f"  清洗后数据形状: {df_clean.shape}")
        print(f"  可用列: {list(df_clean.columns)}")
        
        wait_user_confirm("特征工程", "正在进行特征工程...")
        
        # 第五步：特征工程
        self.state.current_step = "特征工程"
        self.feature_agent.set_data(df_clean)
        
        # 根据参数选择特征生成方式
        if use_llm_features:
            # LLM驱动的智能特征生成 - 新流程
            print(f"\n🤖 正在使用LLM分析数据并生成特征方向...")
            
            # 步骤1: 生成特征方向（不生成具体代码）
            feature_directions = self.feature_agent.generate_feature_directions(
                target_column=target_column,
                task_type=goal.task_type.value,
                business_description=user_goal,
                n_directions=5
            )
            
            # 展示特征方向供用户修改/确认
            print(f"\n{'='*60}")
            print("📋 LLM 特征工程方向建议")
            print(f"{'='*60}")
            for i, d in enumerate(feature_directions, 1):
                print(f"\n【方向 {i}】{d.direction_name}")
                print(f"  类别: {d.category}")
                print(f"  描述: {d.description}")
                print(f"  建议特征:")
                for f in d.suggested_features[:3]:
                    if isinstance(f, dict):
                        print(f"    - {f.get('name', '')}: {f.get('meaning', '')}")
                    else:
                        print(f"    - {f}")
            
            # 步骤1: 循环修改特征方向，直到用户确认
            while True:
                print(f"\n{'='*60}")
                print("操作说明:")
                print("  - 直接回车: 确认当前方向，生成具体特征")
                print("  - 输入修改意见: 如'增加星期特征'、'去掉统计特征'、'改为回归任务'等")
                print("  - 输入'q': 退出")
                print(f"{'='*60}")
                user_input = input("请输入指令: ").strip()
                
                if user_input.lower() == 'q':
                    print("用户退出")
                    return None
                
                if not user_input:
                    # 用户确认，退出循环，继续生成具体特征
                    break
                
                # 用户有修改意见
                print(f"\n🤖 正在根据您的反馈修改特征方向...")
                feature_directions = self.feature_agent.modify_directions_with_feedback(
                    feature_directions, user_input
                )
                print("✅ 方向已修改")
                
                # 展示修改后的方向
                print(f"\n{'='*60}")
                print("📋 修改后的特征工程方向")
                print(f"{'='*60}")
                for i, d in enumerate(feature_directions, 1):
                    print(f"\n【方向 {i}】{d.direction_name}")
                    print(f"  类别: {d.category}")
                    print(f"  描述: {d.description}")
                    print(f"  建议特征:")
                    for f in d.suggested_features[:3]:
                        if isinstance(f, dict):
                            print(f"    - {f.get('name', '')}: {f.get('meaning', '')}")
                        else:
                            print(f"    - {f}")
            
            # 步骤2: 根据确认的方向生成具体代码
            print(f"\n🤖 正在根据方向生成具体特征...")
            feature_result = self.feature_agent.generate_features_from_directions(
                feature_directions, df_clean
            )
            
            # 展示生成的特征及物理意义
            print(f"\n{'='*60}")
            print("✅ 特征生成完成！")
            print(f"{'='*60}")
            print(f"\n生成的新特征及其物理意义:")
            for i, s in enumerate(feature_result.suggestions, 1):
                print(f"\n【特征 {i}】{s.name}")
                print(f"  描述: {s.description}")
                print(f"  类别: {s.category}")
                print(f"  推理: {s.reasoning}")
            
            # 展示数据样例
            print(f"\n{'='*60}")
            print("📊 数据样例（前5行）:")
            print(f"{'='*60}")
            if feature_result.final_data is not None:
                # 只显示新生成的特征列
                new_cols = feature_result.new_columns[:5]  # 最多显示5个新特征
                if new_cols:
                    preview_cols = new_cols + (["目标变量"] if "目标变量" in feature_result.final_data.columns else [])
                    print(feature_result.final_data[preview_cols].head().to_string())
                else:
                    print(feature_result.final_data.head().to_string())
            print(f"\n数据形状: {feature_result.final_data.shape if feature_result.final_data is not None else 'N/A'}")
            
            # 步骤3: 循环确认/重新生成特征，直到用户确认
            while True:
                print(f"\n{'='*60}")
                print("操作说明:")
                print("  - 直接回车: 确认特征，进入质量报告")
                print("  - 输入'n': 重新生成特征")
                print("  - 输入'q': 退出")
                print(f"{'='*60}")
                confirm = input("确认这些特征? (回车确认/n重新生成/q退出): ").strip().lower()
                
                if confirm == 'q':
                    print("用户退出")
                    return None
                
                if confirm == 'n':
                    print("正在重新生成特征...")
                    feature_result = self.feature_agent.generate_features_from_directions(
                        feature_directions, df_clean
                    )
                    # 重新展示
                    print(f"\n{'='*60}")
                    print("✅ 重新生成完成！")
                    print(f"{'='*60}")
                    print(f"\n生成的新特征及其物理意义:")
                    for i, s in enumerate(feature_result.suggestions, 1):
                        print(f"\n【特征 {i}】{s.name}")
                        print(f"  描述: {s.description}")
                    
                    # 展示数据样例
                    print(f"\n{'='*60}")
                    print("📊 数据样例（前5行）:")
                    print(f"{'='*60}")
                    if feature_result.final_data is not None:
                        new_cols = feature_result.new_columns[:5]
                        if new_cols:
                            print(feature_result.final_data[new_cols].head().to_string())
                        else:
                            print(feature_result.final_data.head().to_string())
                    print(f"\n数据形状: {feature_result.final_data.shape if feature_result.final_data is not None else 'N/A'}")
                    continue
                
                # 用户确认，回车
                break
            
            print(f"\n📊 正在计算特征质量报告...")
            df_features = feature_result.final_data
            quality_report = self.feature_agent.calculate_feature_quality(
                df_features, target_column, goal.task_type.value
            )
            
            # 展示特征质量报告
            print(f"\n{'='*60}")
            print("📋 特征质量报告")
            print(f"{'='*60}")
            print(f"\n总特征数: {quality_report['total_features']}")
            
            # IV值统计
            if quality_report.get('summary', {}).get('max_iv'):
                print(f"\nIV值统计:")
                print(f"  平均IV: {quality_report['summary']['avg_iv']}")
                print(f"  最大IV: {quality_report['summary']['max_iv']}")
            
            print(f"\n各特征质量详情:")
            for col, metrics in quality_report['feature_metrics'].items():
                iv_info = f", IV={metrics.get('iv', 'N/A')}" if metrics.get('iv') else ""
                corr_info = f", 相关系数={metrics.get('correlation', 'N/A')}" if metrics.get('correlation') else ""
                print(f"  - {col}: 缺失率={metrics['missing_rate']}%{iv_info}{corr_info}")
            
            # 高相关特征警告
            if quality_report.get('summary', {}).get('high_correlation_features'):
                print(f"\n⚠️ 高相关特征（可能存在多重共线性）:")
                for f in quality_report['summary']['high_correlation_features']:
                    print(f"  - {f}")
            
            # 步骤4: 推荐特征选择
            recommended = self.feature_agent.recommend_feature_selection(quality_report)
            print(f"\n🎯 推荐保留的特征（前10个）:")
            print(f"  {recommended}")
            
            # 等待用户确认
            print(f"\n{'='*60}")
            modify_confirm = input("是否进行特征筛选? (y筛选/n继续建模/q退出): ").strip().lower()
            
            if modify_confirm == 'q':
                print("用户退出")
                return None
            elif modify_confirm == 'y':
                # 进行特征筛选
                feature_cols = [c for c in df_features.columns if c != target_column]
                keep_cols = [c for c in recommended if c in feature_cols]
                if target_column in df_features.columns:
                    df_features = df_features[keep_cols + [target_column]]
                else:
                    df_features = df_features[keep_cols]
                print(f"✅ 特征筛选完成，保留 {len(keep_cols)} 个特征")
            else:
                print("跳过特征筛选，使用全部特征")
            
            # 保存结果
            self.state.artifacts["feature_generation_code"] = feature_result.executed_code
            self.state.artifacts["feature_quality_report"] = quality_report
            self.state.artifacts["feature_suggestions"] = [
                {"name": s.name, "category": s.category, "description": s.description}
                for s in feature_result.suggestions
            ]
            
            # 保存特征生成代码为.py文件
            self._save_feature_code_to_file(feature_result.executed_code)
            
            print(f"\n✅ 特征工程完成!")
        else:
            # 传统方式：特征衍生
            df_features = self.feature_agent.generate_features()
            print(f"\n✅ 特征工程完成!")
            print(f"  总特征数: {len(df_features.columns)}")
        
        # 对类别特征进行编码(标签编码)
        self.feature_agent.set_data(df_features)
        df_features = self.feature_agent.encode_categorical(method="label")
        
        # 分离特征和目标变量（确保target_column存在）
        if target_column not in df_features.columns:
            # 如果target_column不存在，尝试从原始清洗后的数据获取
            if target_column in df_clean.columns:
                df_features[target_column] = df_clean[target_column].values[:len(df_features)]
            else:
                raise ValueError(f"目标列 {target_column} 不在数据中")
        
        X = df_features.drop(columns=[target_column])
        y = df_features[target_column]
        
        print(f"\n📊 最终数据集:")
        print(f"  特征矩阵形状: {X.shape}")
        print(f"  目标变量: {target_column}")
        
        # 第六步：LLM驱动的交互式模型训练
        wait_user_confirm("模型训练", "正在设计训练方案...")
        
        self.state.current_step = "模型训练"
        
        # 步骤1: 生成训练方案
        print(f"\n🤖 正在分析数据并生成训练方案...")
        
        # 准备数据信息
        data_info = f"特征数: {X.shape[1]}, 样本数: {X.shape[0]}"
        
        training_plan = self.model_agent.generate_training_plan(
            X, y, goal, data_info
        )
        
        # 展示训练方案
        print(f"\n{'='*60}")
        print("📋 模型训练方案")
        print(f"{'='*60}")
        
        # 模型选择
        model_choice = training_plan.get("model_choice", {})
        print(f"\n【模型选择】")
        print(f"  选择的模型: {model_choice.get('selected_models', [])}")
        print(f"  选择理由: {model_choice.get('reasoning', 'N/A')}")
        
        # 数据划分
        split_config = training_plan.get("data_split", {})
        print(f"\n【数据划分】")
        print(f"  训练集比例: {split_config.get('train_ratio', 0.8)}")
        print(f"  测试集比例: {split_config.get('test_ratio', 0.2)}")
        print(f"  划分方法: {split_config.get('split_method', 'random')}")
        print(f"  随机种子: {split_config.get('random_state', 42)}")
        
        # 评估指标 - 处理LLM返回的不同格式
        eval_config = training_plan.get("evaluation_metrics", {})
        print(f"\n【评估指标】")
        
        # 处理不同的返回格式
        if isinstance(eval_config, dict):
            print(f"  主要指标: {eval_config.get('primary_metric', eval_config.get('name', 'N/A'))}")
            secondary = eval_config.get("secondary_metrics", [])
            if isinstance(secondary, list):
                secondary_str = ", ".join([s.get('name', str(s)) if isinstance(s, dict) else str(s) for s in secondary[:3]])
                print(f"  次要指标: [{secondary_str}]")
            else:
                print(f"  次要指标: {secondary}")
        elif isinstance(eval_config, list):
            # 如果是列表格式
            if eval_config:
                first = eval_config[0]
                if isinstance(first, dict):
                    print(f"  主要指标: {first.get('name', 'N/A')}")
                    others = eval_config[1:4]
                    others_str = ", ".join([e.get('name', str(e)) if isinstance(e, dict) else str(e) for e in others])
                    print(f"  次要指标: [{others_str}]")
                else:
                    print(f"  评估指标: {eval_config[:5]}")
        else:
            print(f"  评估指标: {eval_config}")
        
        # 超参数
        hyperparams = training_plan.get("hyperparameters", {})
        if hyperparams:
            print(f"\n【超参数设置】")
            for model_name, params in hyperparams.items():
                print(f"  {model_name}: {params}")
        
        # 交叉验证
        cv_config = training_plan.get("cross_validation", {})
        if cv_config.get("enabled"):
            print(f"\n【交叉验证】")
            print(f"  启用: 是")
            print(f"  Folds: {cv_config.get('n_folds', 5)}")
        
        # 步骤2: 循环修改训练方案
        while True:
            print(f"\n{'='*60}")
            print("操作说明:")
            print("  - 直接回车: 确认方案，开始训练")
            print("  - 输入修改意见: 如'增加XGBoost'、'改用时间序列划分'等")
            print("  - 输入'q': 退出")
            print(f"{'='*60}")
            user_input = input("请输入指令: ").strip()
            
            if user_input.lower() == 'q':
                print("用户退出")
                return None
            
            if not user_input:
                # 用户确认，退出循环，开始训练
                break
            
            # 用户有修改意见
            print(f"\n🤖 正在根据您的反馈修改训练方案...")
            training_plan = self.model_agent.modify_training_plan(training_plan, user_input)
            print("✅ 方案已修改")
            
            # 展示修改后的方案
            print(f"\n{'='*60}")
            print("📋 修改后的训练方案")
            print(f"{'='*60}")
            
            model_choice = training_plan.get("model_choice", {})
            selected = model_choice.get("selected_models", [])
            if isinstance(selected, list) and selected:
                if isinstance(selected[0], dict):
                    names = [s.get("model_name", s.get("name", str(s))) for s in selected[:3]]
                else:
                    names = selected[:3]
                print(f"\n【模型选择】: {', '.join(names)}")
            else:
                print(f"\n【模型选择】: {selected}")
            
            split_config = training_plan.get("data_split", {})
            print(f"【数据划分】: 训练{split_config.get('train_ratio', 0.8)}, 测试{split_config.get('test_ratio', 0.2)}")
            
            eval_config = training_plan.get("evaluation_metrics", {})
            if isinstance(eval_config, dict):
                print(f"【评估指标】: {eval_config.get('primary_metric', eval_config.get('name', 'N/A'))}")
            else:
                print(f"【评估指标】: {eval_config}")
        
        # 步骤3: 执行训练
        print(f"\n🤖 正在执行模型训练...")
        result = self.model_agent.execute_training_plan(training_plan, X, y)
        
        # 如果使用了LLM特征生成，将特征代码也保存到结果中
        if use_llm_features:
            result.artifacts["generated_feature_code"] = self.feature_agent.get_generated_code()
            result.artifacts["generated_feature_count"] = len(self.feature_agent.get_generated_code())
        
        # 更新状态为完成
        self.state.status = ProcessStatus.COMPLETED
        self.state.progress = 1.0
        self.state.current_step = "完成"
        
        return result

    def get_progress(self) -> dict[str, Any]:
        """
        获取当前流程执行进度
        
        返回当前AutoML流程的状态信息，包括:
        - 当前执行状态(运行中/完成/失败)
        - 当前正在执行的步骤名称
        - 整体进度百分比
        - 状态消息
        
        返回:
            dict: 包含status、current_step、progress、message的字典
        """
        return {
            "status": self.state.status,
            "current_step": self.state.current_step,
            "progress": self.state.progress,
            "message": self.state.message
        }

    def run_step_by_step(self, step: str, **kwargs) -> Any:
        """
        逐步执行AutoML流程的单个步骤
        
        该方法支持分步执行，允许用户按需调用特定的流程步骤。
        适用于需要中间结果或需要自定义流程的场景。
        
        参数:
            step: 要执行的步骤名称，支持以下值:
                - "load_data": 加载数据
                - "analyze_quality": 分析数据质量
                - "clean_data": 清洗数据
                - "feature_engineering": 特征工程
                - "encode_features": 特征编码
                - "select_features": 特征选择
                - "train_model": 模型训练
            **kwargs: 步骤所需的参数，如data_path、data、method等
        
        返回:
            Any: 步骤执行结果，类型取决于所执行的步骤
        
        异常:
            ValueError: 当提供的step名称不支持时抛出
        
        示例:
            >>> # 只加载数据
            >>> profile = engine.run_step_by_step("load_data", data_path="data.csv")
            >>> # 只执行特征工程
            >>> features = engine.run_step_by_step("feature_engineering", data=df_clean)
        """
        if step == "load_data":
            return self.data_agent.load_data(kwargs["data_path"])
        elif step == "analyze_quality":
            return self.data_agent.analyze_quality()
        elif step == "clean_data":
            return self.data_agent.clean_data(kwargs.get("strategy", "auto"))
        elif step == "feature_engineering":
            self.feature_agent.set_data(kwargs["data"])
            return self.feature_agent.generate_features()
        elif step == "encode_features":
            self.feature_agent.set_data(kwargs["data"])
            return self.feature_agent.encode_categorical(kwargs.get("method", "label"))
        elif step == "select_features":
            return self.feature_agent.select_features(kwargs.get("method", "importance"), kwargs.get("target"))
        elif step == "train_model":
            return self.model_agent.run(
                kwargs["X_train"], 
                kwargs["y_train"], 
                kwargs["X_test"], 
                kwargs["y_test"], 
                kwargs["goal"]
            )
        else:
            raise ValueError(f"未知步骤: {step}")
