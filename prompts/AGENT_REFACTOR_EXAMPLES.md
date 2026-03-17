# Agent 改造示例 - 使用配置文件中的 Prompt

本文件展示如何修改现有的 Agent 类，使用配置文件中的 Prompt 而不是硬编码在代码中。

## 改造前的代码（planner_agent.py）

```python
# 原来的代码 - prompt 硬编码在方法中
def plan_task(self, goal: str, state: PipelineState) -> Dict[str, Any]:
    analysis_prompt = f"""
你是一位专业的 AutoML 专家，需要分析用户的目标并制定执行计划。

用户目标：{goal}

数据信息：
- 形状：{state.data.shape if state.data is not None else '未知'}
- 特征：{state.features if state.features else '未处理'}
- 目标列：{state.target_column}

可用工具类别：
- 数据处理工具：{self.registry.list_tools(category="data")}
- 特征工程工具：{self.registry.list_tools(category="feature")}
- 模型训练工具：{self.registry.list_tools(category="model")}
- 评估工具：{self.registry.list_tools(category="eval")}

请分析：
1. 任务类型（分类/回归）
2. 主要挑战
3. 建议的执行流程
4. 需要用到的关键工具

以 JSON 格式返回分析结果：
{{
    "task_type": "classification|regression|clustering",
    "challenges": ["challenge1", "challenge2"],
    "suggested_flow": ["step1", "step2", "step3"],
    "required_tools": ["tool1", "tool2"]
}}
"""
    
    analysis_response = self.invoke_llm(analysis_prompt)
    # ... 后续代码
```

## 改造后的代码（planner_agent.py）

```python
# 改造后的代码 - 使用配置文件中的 prompt
from core.prompt_loader import load_prompt

class PlannerAgent(BaseAgent):
    """自主规划器 - 分析任务并制定执行计划"""
    
    def __init__(self, llm=None, name="PlannerAgent", verbose=False):
        super().__init__(llm, name, verbose)
        self.registry = get_registry()
    
    def plan_task(self, goal: str, state: PipelineState) -> Dict[str, Any]:
        # 使用配置文件中的 prompt
        analysis_prompt = load_prompt(
            "planner",
            "task_analysis",
            goal=goal,
            data_shape=str(state.data.shape) if state.data is not None else "未知",
            features=", ".join(state.features) if state.features else "未处理",
            target_column=state.target_column,
            data_tools=", ".join(self.registry.list_tools(category="data")),
            feature_tools=", ".join(self.registry.list_tools(category="feature")),
            model_tools=", ".join(self.registry.list_tools(category="model")),
            eval_tools=", ".join(self.registry.list_tools(category="eval"))
        )
        
        analysis_response = self.invoke_llm(analysis_prompt)
        # ... 后续代码
        
        # 生成执行计划
        plan_prompt = load_prompt(
            "planner",
            "plan_generation",
            analysis_result=json.dumps(analysis, ensure_ascii=False, indent=2),
            goal=goal,
            task_type=analysis.get("task_type", "unknown")
        )
        
        plan_response = self.invoke_llm(plan_prompt)
        # ... 后续代码
```

## 改造 feature_engineer.py 示例

### 改造前

```python
# 硬编码在方法中
prompt = f"""你是一位资深的数据科学家和特征工程专家。请分析以下数据集的场景和特征，
自主思考并生成适合的特征工程建议。

## 建模场景
- 任务类型：{task_type}
- 目标变量：{target_column}
- 业务描述：{business_description}

## 数据概览
- 数据形状：{data_summary['shape'][0]}行 x {data_summary['shape'][1]}列
- 数值特征 ({len(numeric_cols)}个):
{chr(10).join(numeric_details) if numeric_details else "无"}
- 类别特征 ({len(categorical_cols)}个):
{chr(10).join(categorical_details) if categorical_details else "无"}

## 你的任务
请生成{n_suggestions}个特征工程建议...
"""
```

### 改造后

```python
# 使用配置文件
from core.prompt_loader import load_prompt

class LLMFeatureAnalyzer:
    def generate_feature_suggestions(self, df: pd.DataFrame, target_column: str,
                                     task_type: str, n_suggestions: int = 10):
        # 准备数据
        numeric_details = self._build_numeric_details(numeric_cols, numeric_stats, target_column)
        categorical_details = self._build_categorical_details(categorical_cols, cat_stats)
        
        # 使用配置文件中的 prompt
        prompt = load_prompt(
            "feature",
            "feature_suggestion_generation",
            task_type=task_type,
            target_column=target_column,
            business_description=business_description,
            data_shape=f"{data_summary['shape'][0]}行 x {data_summary['shape'][1]}列",
            data_columns=data_summary['shape'][1],
            len_numeric_cols=len(numeric_cols),
            numeric_details=numeric_details,
            len_categorical_cols=len(categorical_cols),
            categorical_details=categorical_details,
            n_suggestions=n_suggestions
        )
        
        response = self.llm.invoke(prompt)
        # ... 后续代码
```

## 改造 model_agent.py 示例

### 改造前

```python
task_desc = f"""你是一位 AutoML 专家。请根据以下信息设计模型训练方案。

## 建模任务
- 任务类型：{goal.task_type.value}
- 目标变量：{goal.target_column}
- 业务描述：{goal.description or ''}

## 数据信息
- 样本数量：{n_samples}
- 特征数量：{n_features}
- 数据信息：{data_info}

## 数据特征列名
{list(X.columns) if hasattr(X, 'columns') else '无'}

请设计一个完整的模型训练方案...
"""
```

### 改造后

```python
from core.prompt_loader import load_prompt

class ModelAgent:
    def generate_training_plan(self, X, y, goal: ModelingGoal, data_info: str = ""):
        # 使用配置文件中的 prompt
        prompt = load_prompt(
            "model",
            "training_plan_generation",
            task_type=goal.task_type.value,
            target_column=goal.target_column,
            description=goal.description or "",
            n_samples=n_samples,
            n_features=n_features,
            data_info=data_info,
            feature_columns=list(X.columns) if hasattr(X, 'columns') else "无"
        )
        
        response = self.llm.invoke(prompt)
        # ... 后续代码
```

## 改造 data_agent.py 示例

### 改造前

```python
# 硬编码在方法中（如果有）
prompt = f"""你是一位数据质量分析专家。请分析以下数据的质量情况..."""
```

### 改造后

```python
from core.prompt_loader import load_prompt

class DataQualityAnalyzer:
    def analyze(self, df: pd.DataFrame):
        prompt = load_prompt(
            "data",
            "data_quality_analysis",
            data_shape=str(df.shape),
            columns=", ".join(df.columns.tolist()),
            dtypes=str(df.dtypes.to_dict()),
            missing_info=str(df.isnull().sum().to_dict()),
            duplicate_info=str(df.duplicated().sum()),
            outlier_info="待检测"
        )
        
        response = self.llm.invoke(prompt)
        # ... 后续代码
```

## 关键要点

1. **导入工具**: 在每个需要使用的文件中导入 `load_prompt`
   ```python
   from core.prompt_loader import load_prompt
   ```

2. **替换硬编码的 prompt**: 找到所有硬编码的 prompt 字符串，用 `load_prompt()` 调用替换

3. **参数传递**: 将原来 f-string 中的变量作为参数传递给 `load_prompt()`

4. **测试**: 修改后充分测试，确保 prompt 格式化正确

5. **模块化**: 每个模块的 prompt 放在对应的 YAML 文件中
   - planner -> planner_prompts.yaml
   - feature -> feature_prompts.yaml
   - model -> model_prompts.yaml
   - data -> data_prompts.yaml

## 完整的文件改造清单

需要改造的文件：
- [ ] `agents/planner_agent.py` - 使用 planner_prompts.yaml
- [ ] `agents/feature_engineer.py` - 使用 feature_prompts.yaml
- [ ] `agents/model_agent.py` - 使用 model_prompts.yaml
- [ ] `agents/data_agent.py` - 使用 data_prompts.yaml
- [ ] `automl_agent/interactive.py` - 根据需要添加 prompt

## 优势

1. **易于维护**: 修改 prompt 不需要改动代码
2. **集中管理**: 所有 prompt 在一个地方，便于查找和修改
3. **版本控制**: 可以单独跟踪 prompt 的变更
4. **可测试性**: 可以独立测试不同的 prompt 版本
5. **可配置性**: 可以根据不同场景加载不同的 prompt 配置
