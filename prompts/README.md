# Prompts 配置目录

本目录包含 AutoML Agent 系统中所有 LLM 调用的 Prompt 配置。

## 文件结构

```
prompts/
├── planner_prompts.yaml    # 任务规划相关的 Prompts
├── data_prompts.yaml       # 数据处理相关的 Prompts
├── feature_prompts.yaml    # 特征工程相关的 Prompts
├── model_prompts.yaml      # 模型训练相关的 Prompts
└── README.md              # 使用说明
```

## 使用方法

### 1. 使用 PromptLoader 加载 Prompt

```python
from core.prompt_loader import PromptLoader

# 初始化加载器
loader = PromptLoader()

# 获取并格式化 Prompt
prompt = loader.get_prompt(
    "planner", 
    "task_analysis",
    goal="预测用户是否流失",
    data_shape="(1000, 20)",
    features="age, income, gender",
    target_column="churn",
    data_tools=["load_data", "clean_data"],
    feature_tools=["generate_features", "encode_features"],
    model_tools=["train_model", "evaluate_model"],
    eval_tools=["calculate_metrics"]
)
```

### 2. 使用便捷函数

```python
from core.prompt_loader import load_prompt, load_raw_prompt

# 加载并格式化 Prompt
prompt = load_prompt("feature", "feature_suggestion_generation",
                    task_type="classification",
                    target_column="churn",
                    business_description="预测用户流失",
                    n_suggestions=10)

# 加载原始 Prompt 模板（不格式化）
template = load_raw_prompt("model", "training_plan_generation")
```

### 3. 在 Agent 中使用

```python
from core.prompt_loader import load_prompt

class PlannerAgent:
    def plan_task(self, goal: str, state: PipelineState):
        # 使用配置文件中的 Prompt
        prompt = load_prompt(
            "planner",
            "task_analysis",
            goal=goal,
            data_shape=str(state.data.shape),
            features=", ".join(state.features or []),
            target_column=state.target_column,
            data_tools=", ".join(self.registry.list_tools(category="data")),
            feature_tools=", ".join(self.registry.list_tools(category="feature")),
            model_tools=", ".join(self.registry.list_tools(category="model")),
            eval_tools=", ".join(self.registry.list_tools(category="eval"))
        )
        
        response = self.llm.invoke(prompt)
        return self._parse_response(response)
```

## 添加新的 Prompt

1. 在对应的 YAML 文件中添加新的 Prompt 条目
2. 使用有意义的名称，格式为 `action_target`（如 `feature_suggestion_generation`）
3. 使用 `{parameter_name}` 格式的参数占位符
4. 在 YAML 中使用 `|` 保持多行文本格式

示例：

```yaml
new_feature: |
  你是一位特征工程专家。
  
  ## 任务
  {task_description}
  
  ## 数据
  {data_description}
  
  请生成特征建议...
```

## 修改现有 Prompt

直接编辑对应的 YAML 文件即可。所有使用 Prompt 的地方会自动加载最新配置。

## 最佳实践

1. **参数化**: 使用 `{parameter}` 格式的参数，避免硬编码
2. **模块化**: 按功能模块组织 Prompt，便于管理
3. **文档化**: 为每个 Prompt 添加注释说明用途
4. **版本控制**: 对 Prompt 的修改进行版本控制
5. **测试**: 修改 Prompt 后进行充分测试

## 配置说明

### planner_prompts.yaml
- `task_analysis`: 任务分析 Prompt
- `plan_generation`: 执行计划生成 Prompt
- `task_decomposition`: 任务分解 Prompt

### data_prompts.yaml
- `data_quality_analysis`: 数据质量分析 Prompt
- `cleaning_strategy_generation`: 清洗策略生成 Prompt
- `data_exploration_summary`: 数据探索总结 Prompt

### feature_prompts.yaml
- `feature_suggestion_generation`: 特征建议生成 Prompt
- `feature_direction_generation`: 特征方向生成 Prompt
- `code_generation_from_directions`: 从方向生成代码 Prompt

### model_prompts.yaml
- `training_plan_generation`: 训练方案生成 Prompt
- `training_plan_modification`: 训练方案修改 Prompt
- `model_selection_recommendation`: 模型选择推荐 Prompt
