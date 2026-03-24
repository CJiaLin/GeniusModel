# AutoML ReAct Core

AutoML ReAct 核心模块，提供交互式建模的完整功能。

## 模块说明

### agents/
Agent 实现模块：
- `data_analysis_agent.py` - 数据分析 Agent（可选能力）
- `data_cleaning_agent.py` - 数据清洗 Agent
- `data_exploration_agent.py` - 探索性数据分析 Agent
- `feature_engineering_agent.py` - 特征工程 Agent
- `model_training_agent.py` - 模型训练 Agent

### api/
FastAPI 后端服务：
- `main.py` - API 主入口，提供分阶段工作流与确认执行

### assets/
资产管理模块：
- `asset_manager.py` - 资产保存、加载和下载

### config/
配置管理模块：
- `config_loader.py` - 配置加载器
- `prompts.yaml` - Prompt 模板配置
- `llm_config.yaml` - LLM 模型配置

### confirmation/
用户确认机制：
- `confirmation_point.py` - 确认点实现

### core/
核心组件：
- `react_agent.py` - ReAct Agent 基类
- `memory.py` - 记忆管理
- `observation.py` - 观察结果

### evaluation/
模型评估：
- `model_evaluator.py` - 评估指标计算

### logger/
日志记录：
- `llm_logger.py` - LLM 调用日志（JSONL 格式）

### report/
报告生成：
- `report_generator.py` - Markdown 报告生成
- `pipeline_generator.py` - 全流程 Python 脚本生成

### skills_loader/
Skills 加载：
- `skill_loader.py` - Skill 内容加载器

### tools/
工具集：
- `base_tool.py` - 工具基类
- `data_tools.py` - 数据处理工具
- `feature_tools.py` - 特征工程工具
- `model_tools.py` - 模型工具

### workflow/
工作流管理：
- `workflow_state.py` - 工作流状态管理

### utils/
代码生成与执行：
- `codeact_agent.py` - CodeAct 迭代生成与执行
- `code_generator.py` - 代码生成与执行验证

## 当前主流程

1. 启动工作流（/workflow/start）
2. 数据清洗（生成方案并确认执行）
3. 数据探索性分析（基于清洗后数据）
4. 特征工程（生成方案并确认执行）
5. 模型训练（生成方案并确认执行）
6. 生成报告与全流程脚本

说明：旧版单体编排链路模块已移除，当前统一以 api/main.py 的分阶段接口为准。

## 使用示例

```python
from automl_react.agents.data_cleaning_agent import DataCleaningAgent
from automl_react.config import get_config_loader

# 创建 Agent
agent = DataCleaningAgent(session_id="session-123")

# 生成清洗方案
plan = agent.generate_cleaning_plan("data.csv")
print(plan)  # Markdown 格式

# 用户确认后生成代码
code = agent.generate_cleaning_code()

# 执行清洗
result = agent.execute_cleaning(code)
```

## 配置加载

```python
from automl_react.config import get_config_loader

config_loader = get_config_loader()

# 获取 Prompt
system_prompt = config_loader.get_prompt("data_cleaning", "system_prompt")
plan_template = config_loader.get_prompt("data_cleaning", "plan_generation")

# 获取 LLM 配置
llm_config = config_loader.get_llm_config("gpt-4")

# 获取阶段专用模型配置
stage_config = config_loader.get_stage_model("data_cleaning")
```
