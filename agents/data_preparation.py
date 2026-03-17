"""
数据准备 Agent 集群 - 负责数据加载、清洗、探索等任务
"""
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from core.state import PipelineState, StateStep
import pandas as pd


class DataPreparationAgent(BaseAgent):
    """数据准备 Agent - 协调数据准备相关工作"""
    
    def __init__(self, llm=None, name="DataPreparationAgent", verbose=False):
        super().__init__(llm, name, verbose)
        self.sub_agents = {
            "loader": DataLoaderAgent(llm, verbose),
            "cleaner": DataCleanerAgent(llm, verbose),
            "explorer": DataExplorerAgent(llm, verbose),
            "validator": DataValidatorAgent(llm, verbose)
        }
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """
        执行数据准备任务
        
        Args:
            state: 当前状态
            task: 任务描述
            
        Returns:
            更新后的状态
        """
        self.log(f"开始执行数据准备任务")
        state.update_step(StateStep.DATA_PREPARATION)
        
        # 根据任务类型调用相应的子 Agent
        task_type = task.get("type", "prepare")
        
        if task_type == "load":
            state = self.sub_agents["loader"].execute(state, task)
        elif task_type == "clean":
            state = self.sub_agents["cleaner"].execute(state, task)
        elif task_type == "explore":
            state = self.sub_agents["explorer"].execute(state, task)
        elif task_type == "validate":
            state = self.sub_agents["validator"].execute(state, task)
        else:
            # 执行完整的数据准备流程
            state = self._execute_full_pipeline(state)
        
        self.log(f"数据准备任务完成")
        return state
    
    def _execute_full_pipeline(self, state: PipelineState) -> PipelineState:
        """执行完整的数据准备流程"""
        # 1. 加载数据
        state = self.sub_agents["loader"].execute(state, {"action": "load"})
        
        # 2. 验证数据
        state = self.sub_agents["validator"].execute(state, {"action": "validate"})
        
        # 3. 探索数据
        state = self.sub_agents["explorer"].execute(state, {"action": "explore"})
        
        # 4. 清洗数据
        state = self.sub_agents["cleaner"].execute(state, {"action": "clean"})
        
        return state


class DataLoaderAgent(BaseAgent):
    """数据加载 Agent"""
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """加载数据"""
        self.log("开始加载数据")
        
        from tools.data_tools import load_data
        
        # 从任务或状态中获取数据路径
        data_path = task.get("data_path") or getattr(state, 'data_path', None)
        
        if not data_path:
            raise ValueError("未指定数据路径")
        
        # 加载数据
        df = load_data(data_path)
        state.set_data(df)
        
        # 记录数据路径
        state.data_path = data_path
        
        self.log(f"数据加载完成：{df.shape}")
        return state


class DataCleanerAgent(BaseAgent):
    """数据清洗 Agent"""
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """清洗数据"""
        self.log("开始清洗数据")
        
        from tools.data_tools import clean_data
        
        if state.data is None:
            raise ValueError("数据未加载")
        
        # 获取清洗策略
        strategy = task.get("strategy", "auto")
        handle_nulls = task.get("handle_nulls", "fill_mean")
        drop_duplicates = task.get("drop_duplicates", True)
        
        # 自动分析数据质量
        if strategy == "auto":
            quality_report = self._analyze_data_quality(state.data)
            handle_nulls = quality_report.get("recommended_null_strategy", "fill_mean")
        
        # 执行清洗
        df_clean = clean_data(
            state.data,
            handle_nulls=handle_nulls,
            drop_duplicates=drop_duplicates
        )
        
        state.set_data(df_clean)
        self.log(f"数据清洗完成：{state.data.shape}")
        return state
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析数据质量"""
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        total_cells = df.size
        
        null_ratio = total_nulls / total_cells if total_cells > 0 else 0
        
        # 推荐策略
        if null_ratio > 0.5:
            recommended = "drop"
        elif null_ratio > 0.1:
            recommended = "fill_mean"
        else:
            recommended = "fill_mean"
        
        return {
            "null_ratio": null_ratio,
            "total_nulls": int(total_nulls),
            "recommended_null_strategy": recommended
        }


class DataExplorerAgent(BaseAgent):
    """数据探索 Agent"""
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """探索数据"""
        self.log("开始探索数据")
        
        from tools.data_tools import get_data_info, get_column_stats
        
        if state.data is None:
            raise ValueError("数据未加载")
        
        # 获取数据基本信息
        info = get_data_info(state.data)
        
        # 获取统计信息
        stats = get_column_stats(state.data)
        
        # 将探索结果保存到状态
        state.set_result("data_info", info)
        state.set_result("column_stats", stats)
        
        # 生成探索报告
        report = self._generate_exploration_report(state.data, info, stats)
        state.set_result("data_exploration_report", report)
        
        self.log(f"数据探索完成，生成报告")
        return state
    
    def _generate_exploration_report(self, df: pd.DataFrame, info: Dict, stats: Dict) -> str:
        """生成探索报告"""
        report = []
        report.append("=" * 60)
        report.append("数据探索报告")
        report.append("=" * 60)
        report.append(f"\n数据形状：{info['shape']}")
        report.append(f"列数：{info['columns']}")
        report.append(f"行数：{info['rows']}")
        report.append(f"\n缺失值统计：")
        report.append(f"  总缺失值：{info['null_counts']}")
        report.append(f"\n列数据类型：")
        for col, dtype in info['dtypes'].items():
            report.append(f"  {col}: {dtype}")
        report.append(f"\n列统计信息：")
        for col, col_stats in stats.items():
            report.append(f"\n  {col}:")
            report.append(f"    均值：{col_stats['mean']}")
            report.append(f"    标准差：{col_stats['std']}")
            report.append(f"    最小值：{col_stats['min']}")
            report.append(f"    最大值：{col_stats['max']}")
            report.append(f"    唯一值数量：{col_stats['unique']}")
        
        return "\n".join(report)


class DataValidatorAgent(BaseAgent):
    """数据验证 Agent"""
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """验证数据"""
        self.log("开始验证数据")
        
        if state.data is None:
            raise ValueError("数据未加载")
        
        # 执行验证
        validation_result = self._validate_data(state.data, state.target_column)
        
        # 保存验证结果
        state.set_result("data_validation", validation_result)
        
        if not validation_result.get("is_valid", False):
            self.log(f"数据验证失败：{validation_result.get('errors', [])}", level="WARNING")
        else:
            self.log("数据验证通过")
        
        return state
    
    def _validate_data(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """验证数据质量"""
        errors = []
        warnings = []
        
        # 1. 检查数据是否为空
        if df.empty:
            errors.append("数据为空")
        
        # 2. 检查目标列是否存在
        if target_column not in df.columns:
            errors.append(f"目标列不存在：{target_column}")
        
        # 3. 检查是否有过多的缺失值
        null_ratio = df.isnull().sum().sum() / df.size
        if null_ratio > 0.8:
            errors.append(f"缺失值比例过高：{null_ratio:.2%}")
        elif null_ratio > 0.5:
            warnings.append(f"缺失值比例较高：{null_ratio:.2%}")
        
        # 4. 检查重复值
        duplicate_ratio = df.duplicated().sum() / len(df)
        if duplicate_ratio > 0.5:
            warnings.append(f"重复值比例较高：{duplicate_ratio:.2%}")
        
        # 5. 检查数据类型
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.9:
                    warnings.append(f"列 '{col}' 可能是标识符而非特征")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "null_ratio": null_ratio,
            "duplicate_ratio": duplicate_ratio
        }
