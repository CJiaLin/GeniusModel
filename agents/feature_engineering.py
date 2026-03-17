"""
特征工程 Agent 集群 - 负责特征创建、选择、转换等任务
"""
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from core.state import PipelineState, StateStep
import pandas as pd
import numpy as np


class FeatureEngineeringAgent(BaseAgent):
    """特征工程 Agent - 协调特征工程相关工作"""
    
    def __init__(self, llm=None, name="FeatureEngineeringAgent", verbose=False):
        super().__init__(llm, name, verbose)
        self.sub_agents = {
            "creator": FeatureCreatorAgent(llm, verbose),
            "selector": FeatureSelectorAgent(llm, verbose),
            "transformer": FeatureTransformerAgent(llm, verbose),
            "encoder": FeatureEncoderAgent(llm, verbose)
        }
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """
        执行特征工程任务
        
        Args:
            state: 当前状态
            task: 任务描述
            
        Returns:
            更新后的状态
        """
        self.log(f"开始执行特征工程任务")
        state.update_step(StateStep.FEATURE_ENGINEERING)
        
        # 根据任务类型调用相应的子 Agent
        task_type = task.get("type", "engineer")
        
        if task_type == "create":
            state = self.sub_agents["creator"].execute(state, task)
        elif task_type == "select":
            state = self.sub_agents["selector"].execute(state, task)
        elif task_type == "transform":
            state = self.sub_agents["transformer"].execute(state, task)
        elif task_type == "encode":
            state = self.sub_agents["encoder"].execute(state, task)
        else:
            # 执行完整的特征工程流程
            state = self._execute_full_pipeline(state, task)
        
        self.log(f"特征工程任务完成，当前特征数：{len(state.features)}")
        return state
    
    def _execute_full_pipeline(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """执行完整的特征工程流程"""
        # 1. 编码分类变量
        state = self.sub_agents["encoder"].execute(state, {"action": "encode"})
        
        # 2. 创建新特征
        state = self.sub_agents["creator"].execute(state, {"action": "create"})
        
        # 3. 转换特征
        state = self.sub_agents["transformer"].execute(state, {"action": "transform"})
        
        # 4. 选择特征
        state = self.sub_agents["selector"].execute(state, {"action": "select"})
        
        return state


class FeatureCreatorAgent(BaseAgent):
    """特征创建 Agent - 基于 LLM 或规则创建新特征"""
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """创建新特征"""
        self.log("开始创建特征")
        
        from tools.feature_tools import create_feature, create_interaction_features
        
        if state.data is None:
            raise ValueError("数据未加载")
        
        # 获取创建策略
        strategy = task.get("strategy", "auto")
        
        if strategy == "auto":
            # 基于 LLM 生成特征建议
            suggestions = self._generate_feature_suggestions(state)
        elif strategy == "interaction":
            # 创建交互特征
            suggestions = self._suggest_interactions(state)
        else:
            suggestions = task.get("suggestions", [])
        
        # 执行特征创建
        created_features = []
        for suggestion in suggestions:
            try:
                if "expression" in suggestion:
                    feature_name = suggestion["name"]
                    df_new = create_feature(state.data, feature_name, suggestion["expression"])
                    state.set_data(df_new)
                    state.add_feature(feature_name)
                    created_features.append(feature_name)
                    self.log(f"创建特征：{feature_name}")
            except Exception as e:
                self.log(f"创建特征失败：{suggestion.get('name', 'unknown')} - {str(e)}", level="WARNING")
        
        # 保存创建的特征列表
        state.set_result("created_features", created_features)
        self.log(f"创建完成，共 {len(created_features)} 个新特征")
        
        return state
    
    def _generate_feature_suggestions(self, state: PipelineState) -> List[Dict[str, str]]:
        """基于 LLM 生成特征建议"""
        if self.llm is None:
            return []
        
        # 获取数据信息
        info = state.data.describe(include='all')
        columns = state.data.columns.tolist()
        
        prompt = f"""
基于以下数据信息，建议 5-10 个可能有用的特征：

数据列：{columns}
数据统计信息：
{info.to_string()}

建模目标：{state.goal}
目标列：{state.target_column}

请为每个建议的特征提供：
- 特征名称
- 特征表达式（使用现有列名）
- 特征说明

以 JSON 格式返回：
[
    {{"name": "feature_name", "expression": "col1 + col2", "description": "说明"}},
    ...
]
"""
        
        try:
            response = self.invoke_llm(prompt)
            import json
            suggestions = json.loads(response)
            return suggestions if isinstance(suggestions, list) else []
        except Exception as e:
            self.log(f"生成特征建议失败：{str(e)}", level="WARNING")
            return []
    
    def _suggest_interactions(self, state: PipelineState) -> List[Dict[str, str]]:
        """建议交互特征"""
        numeric_cols = state.data.select_dtypes(include=[np.number]).columns.tolist()
        suggestions = []
        
        # 简单的前 3 个数值列的交互
        for i, col1 in enumerate(numeric_cols[:3]):
            for col2 in numeric_cols[i+1:4]:
                suggestions.append({
                    "name": f"{col1}_mul_{col2}",
                    "expression": f"{col1} * {col2}",
                    "description": f"{col1} 和 {col2} 的交互"
                })
        
        return suggestions


class FeatureSelectorAgent(BaseAgent):
    """特征选择 Agent - 选择最重要的特征"""
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """选择特征"""
        self.log("开始选择特征")
        
        from tools.feature_tools import select_features, drop_features
        
        if state.data is None:
            raise ValueError("数据未加载")
        
        # 获取选择策略
        n_features = task.get("n_features", 20)
        method = task.get("method", "importance")
        
        # 执行特征选择
        try:
            selected = select_features(
                state.data,
                state.target_column,
                n_features=n_features,
                method=method
            )
            
            # 确定要删除的特征
            current_features = [col for col in state.data.columns if col != state.target_column]
            features_to_drop = [f for f in current_features if f not in selected]
            
            # 删除不重要的特征
            if features_to_drop:
                df_new = drop_features(state.data, features_to_drop)
                state.set_data(df_new)
                state.features = selected
            
            # 保存选择结果
            state.set_result("selected_features", selected)
            state.set_result("dropped_features", features_to_drop)
            
            self.log(f"特征选择完成，保留 {len(selected)} 个特征")
            
        except Exception as e:
            self.log(f"特征选择失败：{str(e)}", level="WARNING")
        
        return state


class FeatureTransformerAgent(BaseAgent):
    """特征转换 Agent - 缩放、分箱等转换"""
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """转换特征"""
        self.log("开始转换特征")
        
        from tools.feature_tools import scale_features, binning
        
        if state.data is None:
            raise ValueError("数据未加载")
        
        # 获取转换策略
        transform_type = task.get("type", "scale")
        columns = task.get("columns", None)
        
        # 自动选择数值列
        if columns is None:
            columns = state.data.select_dtypes(include=[np.number]).columns.tolist()
            if state.target_column in columns:
                columns.remove(state.target_column)
        
        # 执行转换
        try:
            if transform_type == "scale":
                method = task.get("method", "standard")
                df_new = scale_features(state.data, columns, method=method)
                state.set_data(df_new)
                self.log(f"特征缩放完成：{method}")
            
            elif transform_type == "binning":
                n_bins = task.get("n_bins", 5)
                for col in columns[:3]:  # 只对前 3 个列分箱
                    df_new = binning(state.data, col, n_bins=n_bins)
                    state.set_data(df_new)
                    state.add_feature(f"{col}_bin")
                self.log(f"特征分箱完成")
            
            # 保存转换结果
            state.set_result("transformed_features", columns)
            
        except Exception as e:
            self.log(f"特征转换失败：{str(e)}", level="WARNING")
        
        return state


class FeatureEncoderAgent(BaseAgent):
    """特征编码 Agent - 编码分类变量"""
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """编码分类变量"""
        self.log("开始编码分类变量")
        
        from tools.feature_tools import encode_categorical
        
        if state.data is None:
            raise ValueError("数据未加载")
        
        # 自动识别分类列
        categorical_cols = state.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 排除目标列
        if state.target_column in categorical_cols:
            categorical_cols.remove(state.target_column)
        
        # 获取编码策略
        method = task.get("method", "onehot")
        columns = task.get("columns", categorical_cols)
        
        # 执行编码
        if columns:
            try:
                df_encoded = encode_categorical(state.data, columns, method=method)
                state.set_data(df_encoded)
                
                # 更新特征列表
                new_cols = [col for col in df_encoded.columns if col != state.target_column]
                state.features = new_cols
                
                self.log(f"编码完成，使用 {method} 方法，处理 {len(columns)} 个分类列")
                state.set_result("encoded_columns", columns)
                state.set_result("encoding_method", method)
                
            except Exception as e:
                self.log(f"特征编码失败：{str(e)}", level="WARNING")
        else:
            self.log("未发现需要编码的分类列")
        
        return state
