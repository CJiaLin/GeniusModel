# AutoML ReAct - 交互式智能建模系统

## 项目简介

基于 **ReAct (Reasoning + Acting)** Agent 架构开发的交互式 AutoML 系统，提供从数据接入到模型训练与报告输出的自动化建模流程。系统支持用户在关键阶段进行确认和干预，确保建模过程符合业务需求。

**核心设计理念**：
- **人在回路 (Human-in-the-loop)**：每个关键阶段都支持用户确认、修改或跳过
- **可审计与可复现**：完整的 LLM 调用日志、代码版本控制和资产保存
- **配置驱动**：通过 YAML 文件管理所有 Prompt 和模型配置，零代码修改即可调整行为
- **数据事实优先**：所有方案必须基于实际数据，禁止臆造字段或统计值

## 核心功能

### 1. 完整的建模工作流

系统遵循标准的机器学习工程流程：

```
数据上传 → 问题定义 → 数据契约检查 → 数据集切分 → 数据清洗 → 数据探索 → 特征工程 → 模型训练 → 模型评估
```

各阶段功能说明：

| 阶段 | 功能描述 |
|------|----------|
| **数据上传** | 自动保存数据到会话目录，生成 Schema 快照和上传元数据 |
| **问题定义** | 明确预测目标、评估指标、业务约束、成功标准和关键假设 |
| **数据契约检查** | 检查数据是否满足建模前提条件，识别风险并给出建议 |
| **数据集切分** | 设计 train/valid/test 切分方案，支持随机、分层、时间顺序、分组切分 |
| **数据清洗** | 自动数据质量分析 + 清洗方案生成 → 用户确认 → 代码生成与执行 |
| **数据探索** | 基于清洗后数据进行统计分布、相关性、目标变量分析 |
| **特征工程** | 生成特征工程方案 → 用户确认 → 代码生成与执行 → 可选特征评估 |
| **模型训练** | 生成建模方案 → 用户确认 → 代码生成与训练执行 |
| **模型评估** | 基于测试集进行最终评估，复核训练阶段指标 |

### 2. 用户确认机制

每个关键阶段都支持用户干预：
- ✅ **确认 (confirmed)**：按方案执行
- ✏️ **修改 (modified)**：提供修改意见后重新生成方案
- ⏭️ **跳过 (skipped)**：跳过当前阶段
- ❌ **拒绝 (rejected)**：拒绝当前方案

### 3. Skills 知识集成

系统自动调用 Skills 目录中的专业知识：

| Skill | 版本 | 用途 |
|-------|------|------|
| **data-analysis** | 1.0.2 | 数据分析方法论、图表选择、指标契约、陷阱识别 |
| **afrexai-ml-engineering** | 1.0.0 | ML 工程最佳实践、模型选择、实验管理 |
| **ml-model-eval-benchmark** | 0.1.0 | 模型评估基准、对比实验设计 |

### 4. 资产管理系统

自动保存所有生成的资产，按会话组织：

```
assets/{session_id}/
├── data/                    # 数据资产
│   ├── original_data.csv    # 原始上传数据
│   ├── data_metadata.json   # 上传元数据
│   └── schema_snapshot.json # Schema 快照
├── cleaning/                # 数据清洗产物
│   ├── cleaning_plan.md     # 清洗方案
│   └── cleaning_result.json # 清洗结果
├── exploration/             # 探索性分析产物
│   └── data_exploration_result.md
├── features/                # 特征工程产物
│   ├── feature_engineering_plan.md
│   ├── feature_engineering_result.json
│   └── feature_metrics_report.md
├── models/                  # 模型产物
│   ├── model_training_plan.md
│   ├── model_training_result.json
│   └── trained_model.pkl
├── analysis/                # 分析报告
│   ├── problem_definition.md
│   ├── data_contract_report.md
│   └── dataset_split_report.md
├── reports/                 # 最终报告
│   ├── modeling_report.md
│   └── modeling_report.html
└── state/                   # 工作流状态
    └── workflow_state.json
```

### 5. LLM 调用日志

记录每次 LLM 调用的完整信息：
- 输入（System Prompt + User Prompt）
- 输出（Response）
- 时间戳和元数据
- 存储格式：JSONL

### 6. 配置驱动架构

所有配置通过 YAML 文件管理：

- `automl_react/config/prompts.yaml`：所有 Agent 的 Prompt 模板
- `automl_react/config/llm_config.yaml`：LLM 模型配置、API 密钥、重试策略
- `automl_react/config/workflow_config.yaml`：工作流阶段配置

## 项目架构

```
AutoMLByLLM/
├── automl_react/                 # 核心代码库
│   ├── agents/                   # Agent 模块
│   │   ├── data_analysis_agent.py      # 问题定义 Agent
│   │   ├── data_contract_agent.py      # 数据契约检查 Agent
│   │   ├── data_splitting_agent.py     # 数据集切分 Agent
│   │   ├── data_cleaning_agent.py      # 数据清洗 Agent
│   │   ├── data_exploration_agent.py   # 探索性分析 Agent
│   │   ├── feature_engineering_agent.py # 特征工程 Agent
│   │   ├── model_training_agent.py      # 模型训练 Agent
│   │   └── model_evaluation_agent.py    # 模型评估 Agent
│   ├── api/
│   │   └── main.py               # FastAPI 主入口
│   ├── config/
│   │   ├── config_loader.py      # 配置加载器
│   │   ├── prompts.yaml          # Prompt 配置
│   │   └── llm_config.yaml       # LLM 配置
│   ├── confirmation/
│   │   └── confirmation_point.py # 用户确认机制
│   ├── core/
│   │   ├── react_agent.py        # ReAct Agent 基类
│   │   ├── memory.py             # 记忆管理
│   │   └── observation.py        # 观察结果
│   ├── evaluation/
│   │   └── model_evaluator.py    # 评估指标计算
│   ├── logger/
│   │   └── llm_logger.py         # LLM 调用日志
│   ├── report/
│   │   ├── report_generator.py   # Markdown 报告
│   │   └── pipeline_generator.py # 全流程脚本
│   ├── skills_loader/
│   │   └── skill_loader.py       # Skill 内容加载
│   ├── tools/
│   │   ├── base_tool.py          # 工具基类
│   │   ├── data_tools.py         # 数据处理工具
│   │   ├── feature_tools.py      # 特征工程工具
│   │   └── model_tools.py        # 模型工具
│   ├── utils/
│   │   ├── code_generator.py     # 代码生成与执行验证
│   │   └── codeact_agent.py      # CodeAct 迭代生成执行
│   └── workflow/
│       └── workflow_state.py     # 工作流状态管理
│
├── frontend/
│   └── index.html               # 单页应用（含 CSS/JS）
│
├── skills/                       # Skills 目录
│   ├── afrexai-ml-engineering-1.0.0/
│   ├── data-analysis-1.0.2/
│   └── ml-model-eval-benchmark-0.1.0/
│
├── logs/                         # 日志目录
│   └── llm_calls/               # LLM 调用日志
│
├── assets/                       # 生成的资产
│   └── {session_id}/            # 按会话组织
│
├── requirements.txt              # 依赖
└── README.md                     # 本文件
```

## 技术栈

- **Agent 架构**: ReAct (Reasoning + Acting)
- **Web 框架**: FastAPI + Uvicorn
- **流式通信**: Server-Sent Events (SSE)
- **前端**: 原生 HTML/JS + Marked.js (Markdown 渲染) + Highlight.js (代码高亮)
- **数据处理**: Pandas + NumPy
- **机器学习**: Scikit-learn + XGBoost
- **配置管理**: PyYAML
- **LLM 支持**: OpenAI 兼容 API（GPT、Claude、Kimi、DeepSeek、通义千问等）

## 快速开始

### 环境要求

- Python 3.9+
- 支持 OpenAI 兼容 API 的 LLM 服务

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 LLM

编辑 `automl_react/config/llm_config.yaml`：

```yaml
default_model: "kimi-k2.5"

models:
  kimi-k2.5:
    provider: "openai"
    model_name: "kimi-k2.5"
    temperature: 1.0
    max_tokens: 65536
    api_key: "${MOONSHOT_API_KEY}"
    base_url: "https://api.moonshot.cn/v1"
```

支持的环境变量：
- `OPENAI_API_KEY`: OpenAI API 密钥
- `ANTHROPIC_API_KEY`: Claude API 密钥
- `MOONSHOT_API_KEY`: Kimi API 密钥
- `DEEPSEEK_API_KEY`: DeepSeek API 密钥
- `DASHSCOPE_API_KEY`: 通义千问 API 密钥

### 启动服务

**1. 启动后端 API**：

```bash
cd /Users/cjialin/code/AutoMLByLLM
python -m automl_react.api.main
```

服务将在 http://localhost:8000 启动

**2. 启动前端**（可选，也可直接用浏览器打开）：

```bash
cd /Users/cjialin/code/AutoMLByLLM/frontend
python -m http.server 8080
```

访问 http://localhost:8080

### 直接使用浏览器打开

前端是独立的单页应用，可以直接用浏览器打开：

```bash
open /Users/cjialin/code/AutoMLByLLM/frontend/index.html
```

## API 接口

### 主要端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/workflow/start` | POST | 启动新工作流 |
| `/workflow/{session_id}/stage/{stage}/run` | POST | 执行指定阶段 |
| `/confirmation/submit` | POST | 提交用户确认 |
| `/confirmation/{session_id}/pending` | GET | 获取当前待确认点 |
| `/workflow/{session_id}/status` | GET | 获取工作流状态 |
| `/chat` | POST | 同步对话 |
| `/chat/stream` | GET | 流式对话（SSE） |
| `/assets/{session_id}/{asset_type}/{filename}` | GET | 下载资产文件 |
| `/assets/{session_id}/list` | GET | 列出会话资产 |
| `/report/generate` | POST | 生成建模报告 |
| `/pipeline/generate` | POST | 生成全流程脚本 |

### 支持的工作流阶段

- `problem_definition` - 问题定义
- `data_contract_check` - 数据契约检查
- `data_splitting` - 数据集切分
- `data_cleaning` - 数据清洗
- `data_exploration` - 数据探索性分析
- `feature_engineering` - 特征工程
- `model_training` - 模型训练
- `model_evaluation` - 模型评估

### 流式对话示例

```javascript
const response = await fetch(
  'http://localhost:8000/chat/stream?session_id=session-123&message=' + encodeURIComponent('分析这个数据文件')
);

const reader = response.body.getReader();
while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    console.log(new TextDecoder().decode(value));
}
```

## 配置说明

### Prompts 配置 (`automl_react/config/prompts.yaml`)

```yaml
data_cleaning:
  system_prompt: "..."
  plan_generation: "..."
  code_generation_full: "..."

feature_engineering:
  system_prompt: "..."
  plan_generation: "..."
  ...
```

### LLM 配置 (`automl_react/config/llm_config.yaml`)

```yaml
default_model: "kimi-k2.5"

models:
  kimi-k2.5:
    provider: "openai"
    model_name: "kimi-k2.5"
    temperature: 1.0
    max_tokens: 65536
    api_key: "${MOONSHOT_API_KEY}"
    base_url: "https://api.moonshot.cn/v1"

stage_models:
  data_analysis: "kimi-k2.5"
  data_cleaning: "kimi-k2.5"
  feature_engineering: "kimi-k2.5"
  model_training: "kimi-k2.5"

logging:
  enabled: true
  log_dir: "logs/llm_calls"
  log_format: "jsonl"
```

## 使用流程

1. **启动工作流**: 输入会话 ID、数据路径、目标列、任务类型、模型
2. **问题定义**: 明确预测目标、评估指标、业务约束
3. **数据契约检查**: 检查数据是否满足建模前提条件
4. **数据集切分**: 设计 train/valid/test 切分方案
5. **数据清洗**: 
   - 系统生成清洗方案（Markdown 格式）
   - 用户确认或修改方案
   - 执行清洗并保存结果
6. **数据探索性分析**:
   - 基于清洗后数据生成探索报告
   - 为特征工程提供统计依据
7. **特征工程**:
   - 系统生成特征工程方案
   - 用户确认或修改方案
   - 执行特征工程并保存结果
   - 可选：执行特征评估（可解释性与可靠性分析）
8. **模型训练**:
   - 系统生成建模方案
   - 用户确认或修改方案
   - 训练模型并评估
9. **模型评估**:
   - 基于测试集进行最终评估
   - 复核训练阶段指标
10. **结果下载**:
    - 下载训练好的模型
    - 下载全流程 Python 脚本
    - 下载建模分析报告

## 核心设计原则

### 1. 数据事实优先
- 所有方案必须基于实际数据的统计信息
- 禁止使用示例数据或虚构数据
- 禁止臆造当前数据中不存在的字段

### 2. 人在回路
- 每个关键阶段都支持用户确认和干预
- 用户可以修改方案或提供额外上下文
- 系统尊重用户的业务知识和需求

### 3. 可审计与可复现
- 完整的 LLM 调用日志（JSONL 格式）
- 所有中间产物和最终结果都保存为资产
- 支持从任意阶段恢复工作流

### 4. 配置驱动
- 零代码修改即可调整 Prompt 和模型
- 支持为不同阶段配置不同的模型
- Prompt 模板支持动态参数替换

### 5. 工程最佳实践
- 严格区分训练集、验证集和测试集
- 防止数据泄露（目标变量变换、特征选择等）
- 支持增量数据的鲁棒处理

## 项目特色

1. **ReAct Agent 架构**: 结合推理和行动，实现智能决策
2. **全程用户参与**: 每个关键阶段都可确认和干预
3. **Skills 知识集成**: 自动调用专业知识库
4. **配置驱动**: 零代码修改即可调整 Prompt 和模型
5. **完整资产保存**: 所有中间产物和最终结果都可下载
6. **详细日志记录**: 完整的 LLM 调用链路可追溯
7. **流式响应**: 实时展示 LLM 生成内容
8. **Markdown 渲染**: 美观的方案展示和代码高亮

## 许可证

MIT License
