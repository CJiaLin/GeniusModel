"""
特征工程Agent模块

本模块是AutoML系统中的特征工程组件，负责数据的特征构建、衍生、编码和筛选。
它是建模流程的第三阶段，在数据清洗完成后执行特征相关操作。

主要组件：
1. FeatureGenerator - 特征生成器（交互特征、聚合特征）
2. FeatureDeriver - 特征衍生器（多项式特征、统计特征）
3. FeatureEncoder - 特征编码器（标签编码、OneHot编码、目标编码、频率编码）
4. FeatureSelector - 特征选择器（重要性筛选、相关性筛选、方差筛选）
5. FeatureTransformer - 特征转换器（标准化、归一化、分箱）
6. LLMFeatureAnalyzer - LLM驱动的智能特征分析器（基于大模型的特征生成）
7. FeatureEngineerAgent - 特征工程Agent主类

工作流程（传统方式）：
1. 生成新特征（交互特征、聚合特征）
2. 衍生特征（多项式、统计特征）
3. 编码类别特征
4. 选择重要特征
5. 转换特征（标准化、归一化）

工作流程（LLM驱动方式）：
1. LLM分析数据场景和特征
2. LLM自主思考生成什么特征
3. LLM生成特征加工代码
4. 执行代码并返回结果
"""

import pandas as pd
import numpy as np
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 修复导入路径
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from automl_agent.models import ProcessState


class FeatureGenerator:
    """
    特征生成器
    
    负责生成新的特征，包括：
    - 基于业务逻辑的领域特征
    - 交互特征（两个特征的乘积、商）
    - 聚合特征（基于分组的统计特征）
    
    Attributes:
        llm: 大语言模型实例（可选，用于智能特征生成）
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化特征生成器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm

    def generate_from_domain(self, df: pd.DataFrame, target: str, business_description: str) -> pd.DataFrame:
        """
        基于业务描述生成领域特征
        
        Args:
            df: 输入数据框
            target: 目标变量名
            business_description: 业务描述
            
        Returns:
            pd.DataFrame: 添加了新特征的数据框
        """
        df_new = df.copy()
        
        return df_new


class FeatureDirection(BaseModel):
    """
    特征方向模型
    
    用于存储LLM生成的特征方向建议，不包含具体代码，仅包含方向描述。
    用户确认方向后，再生成具体的特征加工代码。
    
    Attributes:
        direction_name: 方向名称
        description: 方向描述和为什么要做这个方向
        category: 特征类别（时间特征、统计特征、交叉特征、领域特征等）
        suggested_features: 建议的具体特征列表
    """
    direction_name: str = Field(description="方向名称")
    description: str = Field(description="方向描述")
    category: str = Field(description="特征类别")
    suggested_features: list = Field(default_factory=list, description="建议的具体特征列表")


class FeatureSuggestion(BaseModel):
    """
    特征建议模型
    
    用于存储LLM生成的单个特征建议，包含特征名称、生成逻辑和预期效果。
    
    Attributes:
        name: 特征名称
        description: 特征描述和生成逻辑
        code: 生成该特征的Python代码
        category: 特征类别（交互、聚合、统计、领域特定等）
        reasoning: LLM生成该特征建议的推理过程
    """
    name: str = Field(description="特征名称")
    description: str = Field(description="特征描述和生成逻辑")
    code: str = Field(description="生成该特征的Python代码")
    category: str = Field(description="特征类别")
    reasoning: str = Field(description="LLM生成该特征建议的推理过程")


class FeatureEngineeringResult(BaseModel):
    """
    特征工程结果模型
    
    存储LLM驱动的特征生成完整结果，包括生成的特征、执行的代码和执行日志。
    
    Attributes:
        suggestions: 特征建议列表
        executed_code: 已执行的Python代码列表
        execution_logs: 执行日志列表
        new_columns: 新生成的特征列名列表
        final_data: 最终处理后的数据（使用Any避免Pydantic序列化问题）
        error: 如果执行失败，存储错误信息
    """
    model_config = {'arbitrary_types_allowed': True}
    
    suggestions: list = Field(default_factory=list)
    executed_code: list = Field(default_factory=list)
    execution_logs: list = Field(default_factory=list)
    new_columns: list = Field(default_factory=list)
    final_data: Any = None
    error: Optional[str] = None


class LLMFeatureAnalyzer:
    """
    LLM驱动的智能特征分析器
    
    利用大语言模型的推理能力，分析数据场景和特征，自主思考
    生成什么类型的特征，并生成相应的特征加工代码。
    
    该类实现了从数据分析到代码生成的完整流程：
    1. 分析数据结构和统计特征
    2. 根据建模目标和业务场景思考特征方向
    3. 生成特征建议和加工代码
    4. 执行代码并返回结果
    
    Attributes:
        llm: 大语言模型实例
        executor: 代码执行器实例
        data_summary: 数据摘要信息
        suggestions: 生成的特征建议列表
        execution_history: 执行历史记录
    """
    
    def __init__(self, llm: ChatOpenAI, executor: Optional[Any] = None):
        """
        初始化LLM特征分析器
        
        Args:
            llm: 大语言模型实例，用于分析和生成代码
            executor: 代码执行器实例，如果为None则创建新的
        """
        self.llm = llm
        self.executor = executor
        self.data_summary: Optional[dict] = None
        self.suggestions: list[FeatureSuggestion] = []
        self.execution_history: list[dict] = []
        self.directions: list[FeatureDirection] = []
        self.conversation_history: list[dict] = []  # 对话历史记忆
    
    def add_to_memory(self, role: str, content: str):
        """
        添加对话到记忆
        
        Args:
            role: 角色 (user/system/assistant)
            content: 内容
        """
        self.conversation_history.append({"role": role, "content": content})
    
    def get_memory_context(self) -> str:
        """获取记忆上下文"""
        if not self.conversation_history:
            return ""
        
        context_parts = ["## 对话历史\n"]
        for item in self.conversation_history[-10:]:  # 保留最近10轮
            context_parts.append(f"- {item['role']}: {item['content']}")
        return "\n".join(context_parts)
    
    def analyze_data(self, df: pd.DataFrame, target_column: str, task_type: str, 
                     business_description: Optional[str] = None) -> dict:
        """
        分析数据并生成摘要
        
        对数据进行全面分析，提取关键统计信息，为后续特征生成提供基础。
        
        Args:
            df: 输入数据框
            target_column: 目标变量列名
            task_type: 任务类型（分类/回归）
            business_description: 可选的业务描述
            
        Returns:
            dict: 包含数据分析结果的字典
        """
        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.apply(str).to_dict(),
            "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_stats": {},
            "categorical_stats": {},
            "target_column": target_column,
            "task_type": task_type,
            "business_description": business_description or "无特定业务描述"
        }
        
        # 计算数值列的统计信息
        numeric_cols = summary["numeric_columns"]
        for col in numeric_cols:
            if col != target_column:
                summary["numeric_stats"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median()),
                    "null_count": int(df[col].isnull().sum()),
                    "unique_count": int(df[col].nunique())
                }
        
        # 计算类别列的统计信息
        cat_cols = summary["categorical_columns"]
        for col in cat_cols:
            if col != target_column:
                summary["categorical_stats"][col] = {
                    "unique_values": df[col].nunique(),
                    "null_count": int(df[col].isnull().sum()),
                    "top_values": df[col].value_counts().head(5).to_dict()
                }
        
        # 目标变量的统计信息
        if target_column in df.columns:
            if task_type == "classification":
                summary["target_stats"] = {
                    "distribution": df[target_column].value_counts().to_dict(),
                    "unique_count": int(df[target_column].nunique())
                }
            else:
                summary["target_stats"] = {
                    "mean": float(df[target_column].mean()),
                    "std": float(df[target_column].std()),
                    "min": float(df[target_column].min()),
                    "max": float(df[target_column].max())
                }
        
        self.data_summary = summary
        return summary
    
    def generate_feature_suggestions(self, df: pd.DataFrame, target_column: str,
                                     task_type: str, n_suggestions: int = 10) -> list[FeatureSuggestion]:
        """
        生成特征建议
        
        利用LLM分析数据后，自主思考并生成适合当前场景的特征建议。
        每个建议包含特征名称、描述、生成代码和推理过程。
        
        Args:
            df: 输入数据框
            target_column: 目标变量列名
            task_type: 任务类型（classification/regression）
            n_suggestions: 要生成的建议数量，默认10个
            
        Returns:
            list[FeatureSuggestion]: 特征建议列表
        """
        # 如果还没有分析数据，先进行分析
        if self.data_summary is None:
            self.analyze_data(df, target_column, task_type)
        
        # 构建提示词
        prompt = self._build_feature_suggestion_prompt(
            self.data_summary, n_suggestions
        )
        
        # 调用LLM生成特征建议
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # 解析LLM响应，提取特征建议
        suggestions = self._parse_suggestions(content)
        
        self.suggestions = suggestions
        return suggestions
    
    def _build_feature_suggestion_prompt(self, data_summary: dict, n_suggestions: int) -> str:
        """
        构建特征建议生成的提示词
        
        Args:
            data_summary: 数据摘要字典
            n_suggestions: 需要生成的建议数量
            
        Returns:
            str: 格式化的提示词
        """
        numeric_cols = data_summary.get("numeric_columns", [])
        categorical_cols = data_summary.get("categorical_columns", [])
        numeric_stats = data_summary.get("numeric_stats", {})
        target_column = data_summary.get("target_column", "")
        task_type = data_summary.get("task_type", "")
        business_desc = data_summary.get("business_description", "")
        
        # 构建数值列的详细信息
        numeric_details = []
        for col in numeric_cols:
            if col != target_column and col in numeric_stats:
                stats = numeric_stats[col]
                numeric_details.append(
                    f"- {col}: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}, "
                    f"范围=[{stats['min']:.2f}, {stats['max']:.2f}], 唯一值={stats['unique_count']}"
                )
        
        # 构建类别列的详细信息
        categorical_details = []
        cat_stats = data_summary.get("categorical_stats", {})
        for col in categorical_cols:
            if col != target_column and col in cat_stats:
                stats = cat_stats[col]
                top_vals = ", ".join([f"{k}({v})" for k, v in list(stats['top_values'].items())[:3]])
                categorical_details.append(
                    f"- {col}: 唯一值={stats['unique_values']}, 常见值=[{top_vals}]"
                )
        
        prompt = f"""你是一位资深的数据科学家和特征工程专家。请分析以下数据集的场景和特征，
自主思考并生成适合的特征工程建议。

## 建模场景
- 任务类型: {task_type}
- 目标变量: {target_column}
- 业务描述: {business_desc}

## 数据概览
- 数据形状: {data_summary['shape'][0]}行 x {data_summary['shape'][1]}列
- 数值特征 ({len(numeric_cols)}个):
{chr(10).join(numeric_details) if numeric_details else "无"}
- 类别特征 ({len(categorical_cols)}个):
{chr(10).join(categorical_details) if categorical_details else "无"}

## 你的任务
请生成{n_suggestions}个特征工程建议。对于每个建议，请提供：
1. 特征名称 (name): 简洁明了的特征名
2. 特征描述 (description): 说明这个特征是什么以及为什么有用
3. 生成代码 (code): 完整的Python代码，使用pandas实现
4. 特征类别 (category): 特征类型（交互特征、聚合特征、统计特征、领域特征、时间特征、文本特征等）
5. 推理过程 (reasoning): 你为什么认为这个特征有价值

## 输出格式
请按以下JSON格式输出（不要有其他内容）：
[
  {{
    "name": "特征名称",
    "description": "特征描述",
    "code": "pandas代码",
    "category": "特征类别",
    "reasoning": "推理过程"
  }},
  ...
]

请确保：
1. 代码可以直接运行，使用df作为DataFrame变量名
2. 代码最后将新特征赋值给df，例如: df['new_feature'] = ...
3. 考虑特征的实际价值，优先选择对预测目标有帮助的特征
4. 代码要处理可能的缺失值和异常值
"""
        return prompt
    
    def _parse_suggestions(self, response: str) -> list[FeatureSuggestion]:
        """
        解析LLM响应，提取特征建议
        
        Args:
            response: LLM返回的响应内容
            
        Returns:
            list[FeatureSuggestion]: 特征建议列表
        """
        suggestions = []
        
        try:
            import re
            import json
            
            # 1. 尝试提取JSON数组
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    items = json.loads(json_str)
                except json.JSONDecodeError:
                    # 尝试修复常见的JSON格式问题
                    json_str = self._fix_json_string(json_str)
                    items = json.loads(json_str)
                
                for item in items:
                    if isinstance(item, dict):
                        suggestion = FeatureSuggestion(
                            name=item.get("name", ""),
                            description=item.get("description", ""),
                            code=item.get("code", ""),
                            category=item.get("category", "通用"),
                            reasoning=item.get("reasoning", "")
                        )
                        suggestions.append(suggestion)
        except Exception as e:
            print(f"解析特征建议时出错: {e}")
            # 如果JSON解析失败，尝试从文本中提取
        
        return suggestions
    
    def _fix_json_string(self, json_str: str) -> str:
        """修复常见的JSON格式问题"""
        import re
        
        # 移除行首行尾的空白
        json_str = json_str.strip()
        
        # 修复未闭合的引号
        # 查找形如 "key": "value 的模式（缺少闭合引号）
        json_str = re.sub(r'":\s*"([^"]*?)(?<!\\)"(?:\s*[,}\]])', r'": "\1"\2', json_str)
        
        # 修复未转义的双引号
        # 在字符串值内部的未转义引号
        def fix_unescaped_quotes(match):
            content = match.group(1)
            # 转义内部的引号
            content = content.replace('": "', '": \\"').replace('", "', '", \\"')
            return f'"{content}"'
        
        # 修复未闭合的括号
        if json_str.count('[') > json_str.count(']'):
            json_str += ']' * (json_str.count('[') - json_str.count(']'))
        if json_str.count('{') > json_str.count('}'):
            json_str += '}' * (json_str.count('{') - json_str.count('}'))
        
        return json_str
    
    def generate_and_execute_features(self, df: pd.DataFrame, 
                                       target_column: str,
                                       task_type: str,
                                       business_description: Optional[str] = None,
                                       n_suggestions: int = 10,
                                       execute: bool = True) -> FeatureEngineeringResult:
        """
        生成并执行特征
        
        这是主要的方法，完整实现LLM驱动的特征生成流程：
        1. 分析数据
        2. 生成特征建议
        3. 执行特征代码（可选）
        4. 返回结果
        
        Args:
            df: 输入数据框
            target_column: 目标变量列名
            task_type: 任务类型（classification/regression）
            business_description: 可选的业务描述
            n_suggestions: 要生成的特征建议数量
            execute: 是否执行生成的代码，默认True
            
        Returns:
            FeatureEngineeringResult: 包含执行结果的FeatureEngineeringResult对象
        """
        result = FeatureEngineeringResult()
        
        try:
            # 步骤1: 分析数据
            self.analyze_data(df, target_column, task_type, business_description)
            result.execution_logs.append("步骤1: 数据分析完成")
            
            # 步骤2: 生成特征建议
            suggestions = self.generate_feature_suggestions(
                df, target_column, task_type, n_suggestions
            )
            result.suggestions = suggestions
            result.execution_logs.append(f"步骤2: 生成了{len(suggestions)}个特征建议")
            
            if not execute:
                return result
            
            # 步骤3: 执行特征代码
            df_result = df.copy()
            executed_columns = []
            
            for i, suggestion in enumerate(suggestions):
                if not suggestion.code.strip():
                    continue
                
                try:
                    # 准备执行上下文
                    context = {"df": df_result, "pd": pd, "np": np}
                    
                    # 如果有executor则使用，否则创建临时的
                    if self.executor:
                        exec_result = self.executor.execute_with_context(
                            suggestion.code, context
                        )
                    else:
                        # 直接执行
                        local_ns = {}
                        exec(suggestion.code, {"df": df_result, "pd": pd, "np": np}, local_ns)
                        exec_result = {"success": True, "data": df_result}
                    
                    if exec_result.get("success", False):
                        # 获取执行后的DataFrame
                        if self.executor:
                            df_result = self.executor.get_variable("df")
                        result.execution_logs.append(
                            f"  - 特征 '{suggestion.name}' 执行成功"
                        )
                        executed_columns.append(suggestion.name)
                        result.executed_code.append(suggestion.code)
                    else:
                        error_msg = exec_result.get("error", {}).get("message", "未知错误")
                        result.execution_logs.append(
                            f"  - 特征 '{suggestion.name}' 执行失败: {error_msg}"
                        )
                        
                except Exception as e:
                    result.execution_logs.append(
                        f"  - 特征 '{suggestion.name}' 执行异常: {str(e)}"
                    )
            
            result.new_columns = executed_columns
            result.final_data = df_result
            result.execution_logs.append(
                f"步骤3: 成功生成{len(executed_columns)}个新特征: {executed_columns}"
            )
            
        except Exception as e:
            result.error = str(e)
            result.execution_logs.append(f"错误: {str(e)}")
        
        return result
    
    def generate_feature_directions(self, df: pd.DataFrame, 
                                     target_column: str,
                                     task_type: str,
                                     business_description: Optional[str] = None,
                                     n_directions: int = 5) -> list[FeatureDirection]:
        """
        生成特征方向（不生成具体代码）
        
        这个方法让LLM分析数据后提供特征构建方向，供用户确认后再生成代码。
        
        Args:
            df: 输入数据框
            target_column: 目标变量列名
            task_type: 任务类型
            business_description: 业务描述
            n_directions: 方向数量
            
        Returns:
            list[FeatureDirection]: 特征方向列表
        """
        # 分析数据
        self.analyze_data(df, target_column, task_type, business_description)
        
        # 构建提示词
        prompt = self._build_feature_direction_prompt(
            self.data_summary, n_directions
        )
        
        # 调用LLM生成特征方向
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # 解析方向
        directions = self._parse_directions(content)
        
        self.directions = directions
        return directions
    
    def _build_feature_direction_prompt(self, data_summary: dict, n_directions: int) -> str:
        """构建特征方向生成的提示词"""
        numeric_cols = data_summary.get("numeric_columns", [])
        categorical_cols = data_summary.get("categorical_columns", [])
        numeric_stats = data_summary.get("numeric_stats", {})
        target_column = data_summary.get("target_column", "")
        task_type = data_summary.get("task_type", "")
        business_desc = data_summary.get("business_description", "")
        
        # 构建数值列的详细信息
        numeric_details = []
        for col in numeric_cols[:8]:  # 限制数量
            if col != target_column and col in numeric_stats:
                stats = numeric_stats[col]
                numeric_details.append(
                    f"- {col}: 均值={stats['mean']:.2f}, 范围=[{stats['min']:.2f}, {stats['max']:.2f}]"
                )
        
        # 构建类别列的详细信息
        categorical_details = []
        cat_stats = data_summary.get("categorical_stats", {})
        for col in categorical_cols[:5]:
            if col != target_column and col in cat_stats:
                stats = cat_stats[col]
                categorical_details.append(f"- {col}: 唯一值={stats['unique_values']}")
        
        prompt = f"""你是一位资深的数据科学家和特征工程专家。请分析以下数据集的场景，
为特征工程提供几个可能的方向建议。

## 建模场景
- 任务类型: {task_type}
- 目标变量: {target_column}
- 业务描述: {business_desc}

## 数据概览
- 数值特征: {', '.join(numeric_cols[:8])}
- 类别特征: {', '.join(categorical_cols[:5])}

## 你的任务
请根据已有的数据生成{n_directions}个特征工程方向。对于每个方向，请提供：
1. 方向名称 (direction_name): 简洁的方向名
2. 方向描述 (description): 为什么要做这个方向，有什么价值
3. 特征类别 (category): 时间特征/统计特征/交叉特征/领域特征/变换特征等
4. 建议的具体特征列表 (suggested_features): 建议生成哪些具体特征（列出3-5个特征名和物理意义）

## 输出格式
请按以下JSON格式输出（不要有其他内容）：
[
  {{
    "direction_name": "方向名称",
    "description": "方向描述",
    "category": "特征类别",
    "suggested_features": [
      {{"name": "特征名", "meaning": "物理意义"}},
      ...
    ]
  }},
  ...
]
"""
        return prompt
    
    def _parse_directions(self, response: str) -> list[FeatureDirection]:
        """解析LLM响应，提取特征方向"""
        import re
        import json
        
        directions = []
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                items = json.loads(json_str)
                
                for item in items:
                    direction = FeatureDirection(
                        direction_name=item.get("direction_name", ""),
                        description=item.get("description", ""),
                        category=item.get("category", "通用"),
                        suggested_features=item.get("suggested_features", [])
                    )
                    directions.append(direction)
        except Exception as e:
            print(f"解析特征方向时出错: {e}")
        
        return directions
    
    def generate_features_from_directions(self, directions: list[FeatureDirection],
                                          df: pd.DataFrame) -> FeatureEngineeringResult:
        """
        根据确认的特征方向生成具体代码并执行
        
        Args:
            directions: 确认的特征方向列表
            df: 输入数据框
            
        Returns:
            FeatureEngineeringResult: 执行结果
        """
        # 构建提示词生成具体代码
        prompt = self._build_code_from_directions_prompt(directions, df)
        
        # 调用LLM生成代码
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # 解析生成的代码
        suggestions = self._parse_suggestions(content)
        
        # 执行代码
        result = self._execute_suggestions(suggestions, df)
        
        return result
    
    def _build_code_from_directions_prompt(self, directions: list[FeatureDirection], df: pd.DataFrame) -> str:
        """根据方向构建生成代码的提示词"""
        columns = df.columns.tolist()
        
        direction_info = []
        for d in directions:
            features = []
            for f in d.suggested_features:
                if isinstance(f, dict):
                    features.append(f"{f.get('name', '')}: {f.get('meaning', '')}")
                else:
                    features.append(str(f))
            direction_info.append(
                f"- {d.direction_name} ({d.category}): {d.description}\n  建议特征: {', '.join(features)}"
            )
        
        prompt = f"""你是一位数据科学家。请根据以下特征方向，生成具体的特征加工代码。

## 数据列名
{columns}

## 特征方向
{chr(10).join(direction_info)}

## 你的任务
请为每个方向生成具体的特征加工代码。每个特征需要：
1. name: 特征名称（英文变量名）
2. description: 特征描述
3. code: 完整的Python代码（使用pandas）

## 输出格式
请按以下JSON格式输出：
[
  {{
    "name": "特征名",
    "description": "特征描述（包含物理意义）",
    "code": "pandas代码",
    "category": "特征类别",
    "reasoning": "为什么这个特征有价值"
  }},
  ...
]

注意：
- 代码最后要将新特征赋值给df: df['新特征名'] = ...
- 要处理可能的缺失值
- 特征名要简洁明了
"""
        return prompt
    
    def _execute_suggestions(self, suggestions: list[FeatureSuggestion], 
                           df: pd.DataFrame) -> FeatureEngineeringResult:
        """执行特征建议"""
        result = FeatureEngineeringResult()
        result.suggestions = suggestions
        
        df_result = df.copy()
        executed_columns = []
        
        for suggestion in suggestions:
            if not suggestion.code.strip():
                continue
            
            try:
                context = {"df": df_result, "pd": pd, "np": np}
                
                if self.executor:
                    exec_result = self.executor.execute_with_context(suggestion.code, context)
                else:
                    # 直接执行代码
                    local_ns = {"df": df_result.copy(), "pd": pd, "np": np}
                    exec(suggestion.code, {}, local_ns)
                    df_result = local_ns.get("df", df_result)
                    exec_result = {"success": True, "data": df_result}
                
                if exec_result.get("success", False):
                    if self.executor:
                        df_result = self.executor.get_variable("df")
                    result.execution_logs.append(f"✓ {suggestion.name}: {suggestion.description}")
                    executed_columns.append(suggestion.name)
                    result.executed_code.append(suggestion.code)
                else:
                    error_msg = exec_result.get("error", {}).get("message", "未知错误")
                    result.execution_logs.append(f"✗ {suggestion.name}: {error_msg}")
                    
            except Exception as e:
                result.execution_logs.append(f"✗ {suggestion.name}: {str(e)}")
        
        result.new_columns = executed_columns
        result.final_data = df_result
        result.execution_logs.append(f"\n共生成 {len(executed_columns)} 个特征")
        
        return result
    
    def generate_feature_ideas_only(self, df: pd.DataFrame, 
                                     target_column: str,
                                     task_type: str,
                                     business_description: Optional[str] = None,
                                     n_suggestions: int = 10) -> list[FeatureSuggestion]:
        """
        仅生成特征思路，不执行代码
        
        这个方法让LLM分析数据后提供特征构建思路，供用户确认后再生成代码。
        
        Args:
            df: 输入数据框
            target_column: 目标变量列名
            task_type: 任务类型
            business_description: 业务描述
            n_suggestions: 建议数量
            
        Returns:
            list[FeatureSuggestion]: 特征建议列表（不含执行的代码）
        """
        # 分析数据
        self.analyze_data(df, target_column, task_type, business_description)
        
        # 生成特征建议
        suggestions = self.generate_feature_suggestions(
            df, target_column, task_type, n_suggestions
        )
        
        self.suggestions = suggestions
        return suggestions
    
    def generate_code_for_suggestions(self, suggestions: list[FeatureSuggestion], 
                                       df: pd.DataFrame) -> FeatureEngineeringResult:
        """
        根据确认的特征建议生成代码并执行
        
        Args:
            suggestions: 确认要执行的特征建议列表
            df: 输入数据框
            
        Returns:
            FeatureEngineeringResult: 执行结果
        """
        result = FeatureEngineeringResult()
        result.suggestions = suggestions
        
        df_result = df.copy()
        executed_columns = []
        
        for suggestion in suggestions:
            if not suggestion.code.strip():
                continue
            
            try:
                context = {"df": df_result, "pd": pd, "np": np}
                
                if self.executor:
                    exec_result = self.executor.execute_with_context(suggestion.code, context)
                else:
                    local_ns = {}
                    exec(suggestion.code, {"df": df_result, "pd": pd, "np": np}, local_ns)
                    exec_result = {"success": True, "data": df_result}
                
                if exec_result.get("success", False):
                    if self.executor:
                        df_result = self.executor.get_variable("df")
                    result.execution_logs.append(f"✓ 特征 '{suggestion.name}' 执行成功")
                    executed_columns.append(suggestion.name)
                    result.executed_code.append(suggestion.code)
                else:
                    error_msg = exec_result.get("error", {}).get("message", "未知错误")
                    result.execution_logs.append(f"✗ 特征 '{suggestion.name}' 执行失败: {error_msg}")
                    
            except Exception as e:
                result.execution_logs.append(f"✗ 特征 '{suggestion.name}' 执行异常: {str(e)}")
        
        result.new_columns = executed_columns
        result.final_data = df_result
        result.execution_logs.append(f"成功生成 {len(executed_columns)} 个新特征")
        
        return result
    
    def calculate_feature_quality(self, df: pd.DataFrame, target_column: str, 
                                  task_type: str) -> dict:
        """
        计算特征质量指标
        
        包括：
        - IV值（信息价值）
        - 特征重要性
        - 相关性分析
        - 缺失率统计
        
        Args:
            df: 特征数据框
            target_column: 目标列名
            task_type: 任务类型
            
        Returns:
            dict: 特征质量报告
        """
        report = {
            "total_features": len(df.columns) - 1,
            "feature_metrics": {},
            "summary": {}
        }
        
        # 排除目标列
        feature_cols = [c for c in df.columns if c != target_column]
        
        for col in feature_cols:
            metrics = {}
            
            # 1. 缺失率
            missing_rate = df[col].isnull().sum() / len(df) * 100
            metrics["missing_rate"] = round(missing_rate, 2)
            
            # 2. 唯一值数量
            metrics["unique_count"] = int(df[col].nunique())
            
            # 3. 与目标的相关性（数值型）
            if df[col].dtype in ['int64', 'float64'] and df[target_column].dtype in ['int64', 'float64']:
                corr = df[col].corr(df[target_column])
                metrics["correlation"] = round(corr, 4) if not pd.isna(corr) else None
            else:
                metrics["correlation"] = None
            
            # 4. 对于分类任务，计算IV值
            if task_type == "classification" and target_column in df.columns:
                iv_value = self._calculate_iv(df, col, target_column)
                metrics["iv"] = round(iv_value, 4) if iv_value else None
                # IV值解读
                if iv_value:
                    if iv_value < 0.02:
                        metrics["iv_interpretation"] = "无预测力"
                    elif iv_value < 0.1:
                        metrics["iv_interpretation"] = "弱预测力"
                    elif iv_value < 0.3:
                        metrics["iv_interpretation"] = "中等预测力"
                    elif iv_value < 0.5:
                        metrics["iv_interpretation"] = "强预测力"
                    else:
                        metrics["iv_interpretation"] = "极强预测力"
            
            report["feature_metrics"][col] = metrics
        
        # 汇总统计
        valid_ivs = [m.get("iv") for m in report["feature_metrics"].values() if m.get("iv")]
        if valid_ivs:
            report["summary"]["avg_iv"] = round(sum(valid_ivs) / len(valid_ivs), 4)
            report["summary"]["max_iv"] = round(max(valid_ivs), 4)
        
        high_corr = [c for c, m in report["feature_metrics"].items() 
                     if m.get("correlation") and abs(m["correlation"]) > 0.7]
        report["summary"]["high_correlation_features"] = high_corr
        
        high_missing = [c for c, m in report["feature_metrics"].items() 
                        if m.get("missing_rate", 0) > 50]
        report["summary"]["high_missing_features"] = high_missing
        
        return report
    
    def _calculate_iv(self, df: pd.DataFrame, feature_col: str, target_col: str) -> float:
        """计算IV值（信息价值）"""
        try:
            # 只处理类别型特征
            if df[feature_col].dtype not in ['object', 'category', 'int64']:
                return None
            
            # 创建交叉表
            cross_tab = pd.crosstab(df[feature_col], df[target_col])
            
            if cross_tab.shape[0] < 2 or cross_tab.shape[1] < 2:
                return None
            
            # 计算分布
            total = cross_tab.sum().sum()
            iv = 0
            
            for col in cross_tab.columns:
                col_dist = cross_tab[col] / total
                total_dist = cross_tab.sum(axis=1) / total
                
                for val in col_dist.index:
                    if col_dist[val] > 0 and total_dist[val] > 0:
                        woe = np.log(col_dist[val] / total_dist[val])
                        iv += (col_dist[val] - total_dist[val]) * woe
            
            return iv if not pd.isna(iv) else None
            
        except Exception:
            return None
    
    def recommend_feature_selection(self, quality_report: dict, top_n: int = 10) -> list:
        """
        基于质量报告推荐特征选择
        
        Args:
            quality_report: 特征质量报告
            top_n: 推荐保留的特征数量
            
        Returns:
            list: 推荐保留的特征列表
        """
        feature_scores = []
        
        for col, metrics in quality_report.get("feature_metrics", {}).items():
            score = 0
            
            # IV值评分（越高越好）
            if metrics.get("iv"):
                score += min(metrics["iv"] * 10, 5)  # 最多5分
            
            # 相关性评分（绝对值越高越好）
            if metrics.get("correlation"):
                score += abs(metrics["correlation"]) * 3  # 最多3分
            
            # 缺失率评分（越低越好）
            missing_rate = metrics.get("missing_rate", 0)
            score += (100 - missing_rate) / 100 * 2  # 最多2分
            
            feature_scores.append((col, score))
        
        # 按分数排序
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前N个特征
        recommended = [f[0] for f in feature_scores[:top_n]]
        
        return recommended
    
    def get_feature_code_summary(self) -> str:
        """
        获取所有特征生成代码的汇总
        
        Returns:
            str: 格式化的代码汇总
        """
        if not self.suggestions:
            return "暂无特征建议"
        
        lines = ["=" * 60, "特征工程代码汇总", "=" * 60, ""]
        
        for i, suggestion in enumerate(self.suggestions, 1):
            lines.append(f"【特征 {i}】{suggestion.name}")
            lines.append(f"  类别: {suggestion.category}")
            lines.append(f"  描述: {suggestion.description}")
            lines.append(f"  推理: {suggestion.reasoning}")
            lines.append(f"  代码:")
            # 缩进代码
            for line in suggestion.code.split('\n'):
                lines.append(f"    {line}")
            lines.append("")
        
        return "\n".join(lines)

    def generate_interaction_features(self, df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
        """
        生成交互特征
        
        为数值列生成交互特征，包括乘积和商。
        
        Args:
            df: 输入数据框
            numeric_cols: 数值列名列表
            
        Returns:
            pd.DataFrame: 添加了交互特征的数据框
            
        Example:
            >>> df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
            >>> df_new = generator.generate_interaction_features(df, ['a', 'b'])
            >>> 'a_x_b' in df_new.columns
            True
        """
        df_new = df.copy()
        
        # 遍历所有数值列对
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # 生成乘积特征
                df_new[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                # 生成除法特征（避免除零）
                if df[col2].min() != 0:
                    df_new[f"{col1}_div_{col2}"] = df[col1] / df[col2]
        
        return df_new

    def generate_aggregation_features(self, df: pd.DataFrame, group_cols: list[str], agg_col: str) -> pd.DataFrame:
        """
        生成聚合特征
        
        基于分组列对目标列进行聚合操作，生成统计特征。
        
        Args:
            df: 输入数据框
            group_cols: 分组列名列表
            agg_col: 要聚合的列名
            
        Returns:
            pd.DataFrame: 添加了聚合特征的数据框
            
        Example:
            >>> df = pd.DataFrame({'group': ['A','A','B','B'], 'value': [1,2,3,4]})
            >>> df_new = generator.generate_aggregation_features(df, ['group'], 'value')
            >>> 'value_mean_by_group' in df_new.columns
            True
        """
        df_new = df.copy()
        
        # 聚合函数列表
        agg_funcs = ["mean", "std", "min", "max", "sum"]
        
        # 对每个聚合函数生成特征
        for func in agg_funcs:
            # 使用transform保持原始行数
            grouped = df.groupby(group_cols)[agg_col].transform(func)
            # 创建新列名
            df_new[f"{agg_col}_{func}_by_{'_'.join(group_cols)}"] = grouped
        
        return df_new


class FeatureDeriver:
    """
    特征衍生器
    
    负责从现有特征衍生新的特征，包括：
    - 多项式特征
    - 统计特征
    
    Attributes:
        llm: 大语言模型实例（可选）
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化特征衍生器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm

    def derive_polynomial_features(self, df: pd.DataFrame, columns: list[str], degree: int = 2) -> pd.DataFrame:
        """
        衍生多项式特征
        
        为指定列生成指定次数的多项式特征。
        
        Args:
            df: 输入数据框
            columns: 要生成多项式特征的列名列表
            degree: 多项式的次数，默认2
            
        Returns:
            pd.DataFrame: 添加了多项式特征的数据框
            
        Example:
            >>> df = pd.DataFrame({'x': [1,2,3]})
            >>> df_new = deriver.derive_polynomial_features(df, ['x'], degree=3)
            >>> 'x_pow_2' in df_new.columns
            True
        """
        df_new = df.copy()
        
        # 为每列生成指定次数的多项式
        for col in columns:
            for d in range(2, degree + 1):
                df_new[f"{col}_pow_{d}"] = df[col] ** d
        
        return df_new

    def derive_statistical_features(self, df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
        """
        衍生统计特征
        
        为多个数值列生成统计聚合特征。
        
        Args:
            df: 输入数据框
            numeric_cols: 数值列名列表
            
        Returns:
            pd.DataFrame: 添加了统计特征的数据框
            
        Example:
            >>> df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
            >>> df_new = deriver.derive_statistical_features(df, ['a', 'b'])
            >>> 'numeric_mean' in df_new.columns
            True
        """
        df_new = df.copy()
        
        # 只有多于一列时才生成统计特征
        if len(numeric_cols) > 1:
            # 计算每行的统计量
            df_new["numeric_mean"] = df[numeric_cols].mean(axis=1)   # 均值
            df_new["numeric_std"] = df[numeric_cols].std(axis=1)    # 标准差
            df_new["numeric_sum"] = df[numeric_cols].sum(axis=1)     # 求和
            df_new["numeric_max"] = df[numeric_cols].max(axis=1)     # 最大值
            df_new["numeric_min"] = df[numeric_cols].min(axis=1)     # 最小值
            df_new["numeric_range"] = df_new["numeric_max"] - df_new["numeric_min"]  # 极差
        
        return df_new


class FeatureEncoder:
    """
    特征编码器
    
    负责对类别特征进行编码，包括：
    - 标签编码
    - OneHot编码
    - 目标编码
    - 频率编码
    
    Attributes:
        llm: 大语言模型实例（可选）
        encoding_maps: 编码映射字典，用于存储编码关系
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化特征编码器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm
        self.encoding_maps = {}  # 存储编码映射关系

    def label_encode(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        标签编码
        
        将类别值映射为从0开始的整数。
        
        Args:
            df: 输入数据框
            columns: 要编码的列名列表
            
        Returns:
            pd.DataFrame: 编码后的数据框
            
        Example:
            >>> df = pd.DataFrame({'color': ['red', 'blue', 'red']})
            >>> df_new = encoder.label_encode(df, ['color'])
            >>> df_new['color'].tolist()
            [1, 0, 1]
        """
        df_new = df.copy()
        
        # 对每列进行标签编码
        for col in columns:
            # 获取唯一值并创建映射
            unique_vals = df_new[col].unique()
            encoding_map = {val: idx for idx, val in enumerate(unique_vals)}
            # 保存映射关系
            self.encoding_maps[col] = encoding_map
            # 应用编码
            df_new[col] = df_new[col].map(encoding_map)
        
        return df_new

    def one_hot_encode(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        OneHot编码
        
        将类别特征转换为独热编码形式。
        
        Args:
            df: 输入数据框
            columns: 要编码的列名列表
            
        Returns:
            pd.DataFrame: OneHot编码后的数据框
            
        Example:
            >>> df = pd.DataFrame({'color': ['red', 'blue', 'green']})
            >>> df_new = encoder.one_hot_encode(df, ['color'])
            >>> 'color_red' in df_new.columns
            True
        """
        return pd.get_dummies(df, columns=columns, drop_first=True)

    def target_encode(self, df: pd.DataFrame, columns: list[str], target: pd.Series) -> pd.DataFrame:
        """
        目标编码
        
        使用目标变量的均值对类别特征进行编码。
        
        Args:
            df: 输入数据框
            columns: 要编码的列名列表
            target: 目标变量序列
            
        Returns:
            pd.DataFrame: 目标编码后的数据框
            
        Example:
            >>> df = pd.DataFrame({'city': ['NY', 'LA', 'NY'], 'target': [1, 0, 1]})
            >>> df_new = encoder.target_encode(df, ['city'], df['target'])
        """
        df_new = df.copy()
        
        # 对每个类别列进行目标编码
        for col in columns:
            # 计算每个类别对应的目标均值
            target_mean = df.groupby(col)[target.name].mean()
            # 应用编码
            df_new[f"{col}_target_enc"] = df[col].map(target_mean)
        
        return df_new

    def frequency_encode(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        频率编码
        
        使用每个类别值的频率进行编码。
        
        Args:
            df: 输入数据框
            columns: 要编码的列名列表
            
        Returns:
            pd.DataFrame: 频率编码后的数据框
            
        Example:
            >>> df = pd.DataFrame({'color': ['red', 'blue', 'red']})
            >>> df_new = encoder.frequency_encode(df, ['color'])
        """
        df_new = df.copy()
        
        # 对每个类别列进行频率编码
        for col in columns:
            # 计算每个类别的频率
            freq = df[col].value_counts(normalize=True)
            # 应用编码
            df_new[f"{col}_freq_enc"] = df[col].map(freq)
        
        return df_new


class FeatureSelector:
    """
    特征选择器
    
    负责从多个特征中选择最重要的特征，包括：
    - 基于模型重要性的选择
    - 基于相关性的选择
    - 基于方差的选择
    
    Attributes:
        llm: 大语言模型实例（可选）
        selected_features: 选中的特征列表
        feature_importance: 特征重要性字典
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化特征选择器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm
        self.selected_features = []  # 选中的特征
        self.feature_importance = {}  # 特征重要性

    def select_by_importance(self, X: pd.DataFrame, y: pd.Series, n_features: int = 10) -> list[str]:
        """
        基于特征重要性选择特征
        
        使用随机森林模型计算特征重要性，选择最重要的前n个特征。
        
        Args:
            X: 特征数据框
            y: 目标变量
            n_features: 要选择的特征数量，默认10
            
        Returns:
            list: 选中的特征名列表
            
        Example:
            >>> X = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
            >>> y = pd.Series([0, 1, 0])
            >>> selected = selector.select_by_importance(X, y, n_features=1)
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # 根据目标变量类型选择模型
        if y.dtype == "object" or y.nunique() <= 10:
            # 分类任务
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            # 回归任务
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # 训练模型
        model.fit(X, y)
        
        # 获取特征重要性并排序
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        
        # 保存结果
        self.feature_importance = importances.to_dict()
        self.selected_features = importances.head(n_features).index.tolist()
        
        return self.selected_features

    def select_by_correlation(self, df: pd.DataFrame, target: str, threshold: float = 0.8) -> list[str]:
        """
        基于相关性选择特征
        
        移除与目标变量高度相关的特征，避免多重共线性。
        
        Args:
            df: 输入数据框
            target: 目标变量列名
            threshold: 相关系数阈值，默认0.8
            
        Returns:
            list: 选中的特征名列表
            
        Example:
            >>> df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6], 'target': [1,0,1]})
            >>> selected = selector.select_by_correlation(df, 'target', threshold=0.8)
        """
        # 获取数值列（排除目标列）
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target]
        
        # 计算相关性矩阵
        corr_matrix = df[numeric_cols].corr().abs()
        
        # 获取上三角矩阵
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 找出高相关特征
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # 保留低相关特征
        selected = [c for c in numeric_cols if c not in to_drop]
        self.selected_features = selected
        
        return selected

    def select_by_variance(self, df: pd.DataFrame, threshold: float = 0.0) -> list[str]:
        """
        基于方差选择特征
        
        移除方差低于阈值的特征，这些特征通常信息量较少。
        
        Args:
            df: 输入数据框
            threshold: 方差阈值，默认0.0
            
        Returns:
            list: 选中的特征名列表
            
        Example:
            >>> df = pd.DataFrame({'a': [1,1,1], 'b': [1,2,3]})
            >>> selected = selector.select_by_variance(df, threshold=0.5)
        """
        from sklearn.feature_selection import VarianceThreshold
        
        # 使用方差阈值选择器
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df.select_dtypes(include=["number"]))
        
        # 获取选中的特征
        selected = df.select_dtypes(include=["number"]).columns[selector.get_support()].tolist()
        self.selected_features = selected
        
        return selected


class FeatureTransformer:
    """
    特征转换器
    
    负责对特征进行数值转换，包括：
    - 标准化（Z-score）
    - 归一化（Min-Max）
    - 分箱（离散化）
    
    Attributes:
        llm: 大语言模型实例（可选）
        transformers: 转换器字典，用于保存训练好的转换器
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化特征转换器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm
        self.transformers = {}  # 保存转换器

    def standardize(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        标准化特征（Z-score）
        
        将特征转换为均值为0、标准差为1的分布。
        
        Args:
            df: 输入数据框
            columns: 要标准化的列名列表
            
        Returns:
            pd.DataFrame: 标准化后的数据框
            
        Example:
            >>> df = pd.DataFrame({'a': [1,2,3,4,5]})
            >>> df_new = transformer.standardize(df, ['a'])
            >>> abs(df_new['a'].mean()) < 0.01
            True
        """
        from sklearn.preprocessing import StandardScaler
        
        df_new = df.copy()
        
        # 创建并应用标准化器
        scaler = StandardScaler()
        df_new[columns] = scaler.fit_transform(df[columns])
        
        # 保存转换器以便后续使用
        self.transformers["standardize"] = scaler
        
        return df_new

    def normalize(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        归一化特征（Min-Max）
        
        将特征缩放到[0, 1]范围内。
        
        Args:
            df: 输入数据框
            columns: 要归一化的列名列表
            
        Returns:
            pd.DataFrame: 归一化后的数据框
            
        Example:
            >>> df = pd.DataFrame({'a': [1,2,3,4,5]})
            >>> df_new = transformer.normalize(df, ['a'])
            >>> df_new['a'].min() == 0 and df_new['a'].max() == 1
            True
        """
        from sklearn.preprocessing import MinMaxScaler
        
        df_new = df.copy()
        
        # 创建并应用归一化器
        scaler = MinMaxScaler()
        df_new[columns] = scaler.fit_transform(df[columns])
        
        # 保存转换器
        self.transformers["normalize"] = scaler
        
        return df_new

    def bin_numerical(self, df: pd.DataFrame, columns: list[str], n_bins: int = 5) -> pd.DataFrame:
        """
        数值分箱
        
        将连续数值特征离散化为多个箱。
        
        Args:
            df: 输入数据框
            columns: 要分箱的列名列表
            n_bins: 分箱数量，默认5
            
        Returns:
            pd.DataFrame: 分箱后的数据框
            
        Example:
            >>> df = pd.DataFrame({'a': [1,2,3,4,5,6,7,8,9,10]})
            >>> df_new = transformer.bin_numerical(df, ['a'], n_bins=3)
            >>> df_new['a_binned'].unique()
            array([0, 1, 2])
        """
        df_new = df.copy()
        
        # 对每列进行分箱
        for col in columns:
            df_new[f"{col}_binned"] = pd.cut(df[col], bins=n_bins, labels=False)
        
        return df_new


class FeatureEngineerAgent:
    """
    特征工程Agent
    
    这是特征工程阶段的主类，整合了特征生成、衍生、编码、选择和转换功能。
    在AutoML流程中负责将原始数据转换为适合建模的特征。
    
    该Agent支持两种特征生成方式：
    1. 传统方式（硬编码）：使用预定义的特征工程方法
    2. LLM驱动方式：利用大语言模型的推理能力自主生成特征
    
    Attributes:
        llm: 大语言模型实例
        generator: 特征生成器
        deriver: 特征衍生器
        encoder: 特征编码器
        selector: 特征选择器
        transformer: 特征转换器
        llm_analyzer: LLM驱动的智能特征分析器
        state: 流程状态
        current_data: 当前处理的特征数据
        llm_result: 最近一次LLM特征生成的结果
        
    Example:
        >>> agent = FeatureEngineerAgent()
        >>> agent.set_data(df)
        >>> 
        >>> # 传统方式
        >>> df_features = agent.generate_features()
        >>> 
        >>> # LLM驱动方式
        >>> result = agent.generate_features_with_llm(target_column="target", task_type="classification")
        >>> print(result.new_columns)
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None, executor: Optional[Any] = None):
        """
        初始化特征工程Agent
        
        Args:
            llm: 大语言模型实例
            executor: 代码执行器实例
        """
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from automl_agent.core.executor import CodeExecutor
        
        # 初始化LLM
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        
        # 初始化代码执行器
        self.executor = executor or CodeExecutor()
        
        # 初始化各组件
        self.generator = FeatureGenerator(self.llm)       # 特征生成
        self.deriver = FeatureDeriver(self.llm)          # 特征衍生
        self.encoder = FeatureEncoder(self.llm)           # 特征编码
        self.selector = FeatureSelector(self.llm)         # 特征选择
        self.transformer = FeatureTransformer(self.llm)   # 特征转换
        self.llm_analyzer = LLMFeatureAnalyzer(self.llm, self.executor)  # LLM特征分析器
        
        self.state = ProcessState()                       # 流程状态
        self.current_data: Optional[pd.DataFrame] = None # 当前数据
        self.llm_result: Optional[FeatureEngineeringResult] = None  # LLM生成结果
        self.generated_code: list[str] = []  # 保存生成的特征加工代码

    def set_data(self, df: pd.DataFrame):
        """
        设置要处理的特征数据
        
        Args:
            df: 输入数据框
        """
        self.current_data = df.copy()
        self.generated_code = []

    def generate_features_with_llm(
        self,
        target_column: str,
        task_type: str,
        business_description: Optional[str] = None,
        n_suggestions: int = 10,
        execute: bool = True
    ) -> FeatureEngineeringResult:
        """
        使用LLM驱动生成特征
        
        这是核心的LLM驱动特征生成方法，利用大模型的推理能力：
        1. 分析数据的统计特征和结构
        2. 根据建模目标和业务场景自主思考特征方向
        3. 生成特征建议和加工代码
        4. 执行代码并返回结果
        
        Args:
            target_column: 目标变量列名
            task_type: 任务类型（classification/regression）
            business_description: 可选的业务描述，用于更精准的特征生成
            n_suggestions: 要生成的特征建议数量，默认10个
            execute: 是否执行生成的代码，默认True
            
        Returns:
            FeatureEngineeringResult: 包含特征建议、执行的代码、执行日志和新特征的结果对象
            
        Example:
            >>> agent = FeatureEngineerAgent()
            >>> agent.set_data(df)
            >>> 
            >>> # LLM驱动的特征生成
            >>> result = agent.generate_features_with_llm(
            ...     target_column="churn",
            ...     task_type="classification",
            ...     business_description="预测用户是否流失",
            ...     n_suggestions=15
            ... )
            >>> 
            >>> # 查看生成的特征
            >>> print(f"生成了 {len(result.new_columns)} 个新特征")
            >>> print(result.new_columns)
            >>> 
            >>> # 获取最终数据
            >>> df_with_features = result.final_data
        """
        self.state.current_step = "LLM智能特征生成"
        
        if self.current_data is None:
            raise ValueError("请先设置数据")
        
        # 调用LLM分析器生成特征
        result = self.llm_analyzer.generate_and_execute_features(
            df=self.current_data,
            target_column=target_column,
            task_type=task_type,
            business_description=business_description,
            n_suggestions=n_suggestions,
            execute=execute
        )
        
        # 保存结果和代码
        self.llm_result = result
        self.generated_code = result.executed_code
        
        # 如果执行成功，更新当前数据
        if result.final_data is not None:
            self.current_data = result.final_data
        
        return result

    def generate_features(self, business_description: Optional[str] = None, target: Optional[str] = None) -> pd.DataFrame:
        """
        生成新特征
        
        从现有特征生成新的衍生特征。
        
        Args:
            business_description: 可选的业务描述
            target: 可选的目标变量名
            
        Returns:
            pd.DataFrame: 添加了新特征的数据框
            
        Raises:
            ValueError: 如果未先设置数据
        """
        self.state.current_step = "生成特征"
        
        if self.current_data is None:
            raise ValueError("请先设置数据")
        
        df = self.current_data
        # 获取数值列
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        
        # 衍生统计特征
        df = self.deriver.derive_statistical_features(df, numeric_cols)
        
        # 更新当前数据
        self.current_data = df
        return df

    def encode_categorical(self, method: str = "label", columns: Optional[list[str]] = None) -> pd.DataFrame:
        """
        编码类别特征
        
        Args:
            method: 编码方法（label/onehot/frequency），默认label
            columns: 要编码的列名列表，默认所有类别列
            
        Returns:
            pd.DataFrame: 编码后的数据框
            
        Raises:
            ValueError: 如果未先设置数据
        """
        self.state.current_step = "编码类别特征"
        
        if self.current_data is None:
            raise ValueError("请先设置数据")
        
        # 如果未指定列，默认选择所有类别列
        if columns is None:
            columns = self.current_data.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # 根据方法选择编码方式
        if method == "label":
            df = self.encoder.label_encode(self.current_data, columns)
        elif method == "onehot":
            df = self.encoder.one_hot_encode(self.current_data, columns)
        elif method == "frequency":
            df = self.encoder.frequency_encode(self.current_data, columns)
        else:
            df = self.current_data
        
        # 更新当前数据
        self.current_data = df
        return df

    def select_features(self, method: str = "importance", target: Optional[pd.Series] = None, n_features: int = 10) -> list[str]:
        """
        选择重要特征
        
        Args:
            method: 选择方法（importance/correlation/variance）
            target: 目标变量（部分方法需要）
            n_features: 要选择的特征数量
            
        Returns:
            list: 选中的特征名列表
            
        Raises:
            ValueError: 如果未先设置数据
        """
        self.state.current_step = "选择特征"
        
        if self.current_data is None:
            raise ValueError("请先设置数据")
        
        # 根据方法选择特征
        if method == "importance" and target is not None:
            selected = self.selector.select_by_importance(self.current_data, target, n_features)
        elif method == "correlation":
            if target is None:
                raise ValueError("需要指定target列进行相关性筛选")
            selected = self.selector.select_by_correlation(self.current_data, target.name)
        elif method == "variance":
            selected = self.selector.select_by_variance(self.current_data)
        else:
            selected = self.current_data.columns.tolist()
        
        return selected

    def transform_features(self, method: str = "standardize", columns: Optional[list[str]] = None) -> pd.DataFrame:
        """
        转换特征
        
        Args:
            method: 转换方法（standardize/normalize/binning）
            columns: 要转换的列名列表
            
        Returns:
            pd.DataFrame: 转换后的数据框
            
        Raises:
            ValueError: 如果未先设置数据
        """
        self.state.current_step = "转换特征"
        
        if self.current_data is None:
            raise ValueError("请先设置数据")
        
        # 如果未指定列，默认选择所有数值列
        if columns is None:
            columns = self.current_data.select_dtypes(include=["number"]).columns.tolist()
        
        # 根据方法进行转换
        if method == "standardize":
            df = self.transformer.standardize(self.current_data, columns)
        elif method == "normalize":
            df = self.transformer.normalize(self.current_data, columns)
        elif method == "binning":
            df = self.transformer.bin_numerical(self.current_data, columns)
        else:
            df = self.current_data
        
        # 更新当前数据
        self.current_data = df
        return df

    def get_data(self) -> pd.DataFrame:
        """
        获取当前处理的特征数据
        
        Returns:
            pd.DataFrame: 当前数据
            
        Raises:
            ValueError: 如果未先设置数据
        """
        if self.current_data is None:
            raise ValueError("请先设置数据")
        return self.current_data

    def get_generated_code(self) -> list[str]:
        """
        获取生成的特征加工代码
        
        返回最近一次LLM特征生成过程中执行的所有Python代码。
        
        Returns:
            list[str]: 生成的代码列表
            
        Example:
            >>> agent.generate_features_with_llm(target_column="churn", task_type="classification")
            >>> codes = agent.get_generated_code()
            >>> for code in codes:
            ...     print(code)
            ...     print("-" * 40)
        """
        return self.generated_code
    
    def generate_feature_ideas_only(
        self,
        target_column: str,
        task_type: str,
        business_description: Optional[str] = None,
        n_suggestions: int = 10
    ) -> list:
        """
        仅生成特征思路，不执行代码
        
        委托给llm_analyzer执行
        
        Returns:
            list: 特征建议列表
        """
        return self.llm_analyzer.generate_feature_ideas_only(
            df=self.current_data,
            target_column=target_column,
            task_type=task_type,
            business_description=business_description,
            n_suggestions=n_suggestions
        )
    
    def generate_code_for_suggestions(self, suggestions: list, df: pd.DataFrame) -> FeatureEngineeringResult:
        """
        根据确认的特征建议生成代码并执行
        
        委托给llm_analyzer执行
        
        Returns:
            FeatureEngineeringResult: 执行结果
        """
        result = self.llm_analyzer.generate_code_for_suggestions(suggestions, df)
        self.generated_code = result.executed_code
        self.llm_result = result
        return result
    
    def calculate_feature_quality(self, df: pd.DataFrame, target_column: str, task_type: str) -> dict:
        """
        计算特征质量指标
        
        委托给llm_analyzer执行
        
        Returns:
            dict: 特征质量报告
        """
        return self.llm_analyzer.calculate_feature_quality(df, target_column, task_type)
    
    def generate_feature_directions(self, target_column: str, task_type: str,
                                   business_description: Optional[str] = None,
                                   n_directions: int = 5) -> list:
        """
        生成特征方向
        
        Returns:
            list: 特征方向列表
        """
        return self.llm_analyzer.generate_feature_directions(
            df=self.current_data,
            target_column=target_column,
            task_type=task_type,
            business_description=business_description,
            n_directions=n_directions
        )
    
    def generate_features_from_directions(self, directions: list, df: pd.DataFrame) -> FeatureEngineeringResult:
        """
        根据确认的方向生成特征
        
        Returns:
            FeatureEngineeringResult: 执行结果
        """
        result = self.llm_analyzer.generate_features_from_directions(directions, df)
        self.generated_code = result.executed_code
        self.llm_result = result
        return result
    
    def modify_directions_with_feedback(self, directions: list, user_feedback: str) -> list:
        """
        根据用户反馈修改特征方向（带记忆功能）
        
        会将前文生成的情况和用户修改的情况融合到上下文中。
        
        Args:
            directions: 原始方向列表
            user_feedback: 用户反馈（如"增加星期特征"）
            
        Returns:
            list: 修改后的方向列表
        """
        # 获取记忆上下文
        memory_context = self.get_memory_context()
        
        # 构建提示词，包含记忆
        prompt = f"""你是一位数据科学家。用户想要修改特征方向。

{memory_context}

原始方向:
{directions}

用户最新反馈: {user_feedback}

请根据用户反馈，生成修改后的特征方向列表。确保：
1. 保留之前有效的方向
2. 根据用户反馈添加新方向或修改
3. 保持JSON格式输出
"""
        # 保存用户反馈到记忆
        self.add_to_memory("user", user_feedback)
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        modified = self.llm_analyzer._parse_directions(content)
        
        # 保存修改结果到记忆
        self.add_to_memory("assistant", f"根据反馈修改为{len(modified)}个方向")
        
        return modified
    
    def get_memory_context(self) -> str:
        """获取记忆上下文"""
        return self.llm_analyzer.get_memory_context()
    
    def add_to_memory(self, role: str, content: str):
        """添加对话到记忆"""
        self.llm_analyzer.add_to_memory(role, content)
    
    def recommend_feature_selection(self, quality_report: dict, top_n: int = 10) -> list:
        """
        基于质量报告推荐特征选择
        
        委托给llm_analyzer执行
        
        Returns:
            list: 推荐保留的特征列表
        """
        return self.llm_analyzer.recommend_feature_selection(quality_report, top_n)
    
    def get_code_summary(self) -> str:
        """
        获取特征生成代码的汇总报告
        
        Returns:
            str: 格式化的代码汇总报告
        """
        if not self.llm_result:
            return "暂无特征生成结果"
        
        return self.llm_analyzer.get_feature_code_summary()
    
    def get_llm_result(self) -> Optional[FeatureEngineeringResult]:
        """
        获取最近一次LLM特征生成的结果
        
        Returns:
            Optional[FeatureEngineeringResult]: 特征工程结果，包含建议、代码、执行日志等
        """
        return self.llm_result
