# AutoML ReAct - 交互式智能建模系统

## 项目简介

基于 **ReAct (Reasoning + Acting)** Agent 架构开发的交互式 AutoML 系统，提供从数据接入到模型训练与报告输出的完整自动化建模流程。系统在每个关键阶段支持用户确认和干预，确保建模过程符合业务需求。

**核心设计理念**：

- **人在回路 (Human-in-the-Loop)**：每个关键阶段都支持用户确认、修改或跳过
- **可审计与可复现**：完整的 LLM 调用日志、代码版本控制和资产保存
- **配置驱动**：通过 YAML 文件管理所有 Prompt 和模型配置，零代码修改即可调整行为
- **数据事实优先**：所有方案必须基于实际数据，禁止臆造字段或统计值

## 系统架构

### ReAct Agent 核心循环

```
┌──────────────┐
│  Observation  │  ← 观察环境（数据事实、前序结果）
└──────┬───────┘
       ▼
┌──────────────┐
│   Thought    │  ← LLM 推理下一步行动
└──────┬───────┘
       ▼
┌──────────────┐
│    Action    │  ← 执行工具调用（数据分析、代码生成等）
└──────┬───────┘
       ▼
┌──────────────┐
│  Observation  │  ← 捕获执行结果，继续循环或完成
└──────────────┘
```

### 核心组件

| 组件 | 位置 | 职责 |
|------|------|------|
| **ReActAgent** | `core/react_agent.py` | ReAct 循环引擎，工具管理，LLM 消息构建与交互 |
| **Memory** | `core/memory.py` | 对话历史、思考过程、动作和观察的记忆管理 |
| **Observation** | `core/observation.py` | 工具执行结果的统一表示（数据、错误、状态） |
| **ConfigLoader** | `config/config_loader.py` | 从 YAML 加载 Prompt、LLM 配置、工作流配置 |
| **AssetManager** | `assets/asset_manager.py` | 会话资产管理（数据、代码、模型、报告）按目录组织 |
| **WorkflowState** | `workflow/workflow_state.py` | 工作流阶段跟踪、状态转移验证、持久化 |
| **LLMLogger** | `logger/llm_logger.py` | 记录 LLM 调用（输入/输出/Token/延迟）为 JSONL |
| **ConfirmationPoint** | `confirmation/confirmation_point.py` | 用户确认机制（确认/修改/跳过/拒绝） |
| **SubprocessExecutor** | `utils/subprocess_executor.py` | 隔离子进程执行 LLM 生成的代码，超时控制与崩溃隔离 |
| **CodeActAgent** | `utils/codeact_agent.py` | CodeAct 模式：迭代式代码生成与执行 |
| **CodeGenerator** | `utils/code_generator.py` | 代码生成与执行验证 |

## 完整建模工作流

系统遵循标准的机器学习工程流程，共 9 个阶段：

```
数据上传 → 问题定义 → 数据契约检查 → 数据集切分 → 数据清洗 → 数据探索 → 特征工程 → 模型训练 → 模型评估
```

### 各阶段详细说明

| 阶段 | Agent | 需确认 | 功能描述 |
|------|-------|--------|----------|
| **数据上传** | — | ❌ | 自动保存数据到会话目录，生成 Schema 快照和上传元数据（MD5、时间戳） |
| **问题定义** | DataAnalysisAgent | ❌ | 加载数据收集事实，LLM 生成结构化问题定义（目标列、任务类型、评估指标、业务约束、成功标准） |
| **数据契约检查** | DataContractAgent | ❌ | 纯逻辑检查（无 LLM）：样本量 ≥ 30、目标列存在、无严重缺失、数据泄漏风险、ID 列识别 |
| **数据集切分** | DataSplittingAgent | ✅ | 自动推断切分方式（随机/分层/时间序列/分组），生成可审阅的切分方案和代码 |
| **数据清洗** | DataCleaningAgent | ✅ | 质量分析 → 方案生成 → 用户确认 → 代码生成与执行（仅无状态方法） |
| **数据探索** | DataExplorationAgent | ❌ | 统计分布、相关性矩阵、目标变量分析，输出特征工程建议 |
| **特征工程** | FeatureEngineeringAgent | ✅ | 特征变换/交互/编码/选择方案 → 用户确认 → 代码生成与执行（不变换目标变量） |
| **模型训练** | ModelTrainingAgent | ✅ | 建模方案 → 用户确认 → 训练执行，输出标准化 model_package_v1 模型包 |
| **模型评估** | ModelEvaluationAgent | ❌ | 测试集评估 + 训练指标复核 + 最终评估结论 |

### 用户确认机制

在标记为"需确认"的阶段，系统会暂停等待用户决策：

- ✅ **确认 (confirmed)**：按当前方案执行
- ✏️ **修改 (modified)**：提供修改意见，系统重新生成方案（支持多轮修订）
- ⏭️ **跳过 (skipped)**：跳过当前阶段
- ❌ **拒绝 (rejected)**：拒绝当前方案

## 工具系统

所有工具继承自 `BaseTool` 抽象基类，提供统一的 Schema 和执行接口。

| 工具 | 位置 | 功能 |
|------|------|------|
| `load_data` | `tools/data_tools.py` | 加载 CSV/Excel/JSON 文件，返回数据基本信息和预览 |
| `analyze_data` | `tools/data_tools.py` | 分析数据统计特性（缺失值、唯一值、数值/类别列统计） |
| `profile_data` | `tools/profile_tools.py` | 数据质量分析（缺失值、异常值 IQR 检测、重复值、分布） |
| `generate_features` | `tools/feature_tools.py` | 自动生成统计特征和交互特征 |
| `train_model` | `tools/model_tools.py` | 训练随机森林分类/回归模型 |
| `query_stage_result` | `tools/stage_tools.py` | 查询前序阶段结果（清洗、探索、特征、模型等） |
| `search_skills` / `read_skill` | `tools/skill_tools.py` | 搜索和读取 Skills 专业知识包 |

## Skills 知识系统

Agent 在需要时自动调用 `skills/` 目录下的专业知识包，Markdown 内容注入 LLM Prompt 指导决策：

| Skill | 版本 | 用途 | 关联阶段 |
|-------|------|------|----------|
| **data-analysis** | 1.0.2 | 数据分析方法论、图表选择、指标契约、陷阱识别 | 数据清洗、数据探索 |
| **afrexai-ml-engineering** | 1.0.0 | ML 工程最佳实践、模型选择、实验管理 | 特征工程、模型训练 |
| **ml-model-eval-benchmark** | 0.1.0 | 模型评估基准、对比实验设计 | 模型训练、模型评估 |

## 资产管理系统

所有生成的资产按会话自动组织，支持中间结果保存和断点恢复：

```
assets/{session_id}/
├── data/                           # 数据资产
│   ├── original_data.csv           # 原始上传数据
│   ├── data_metadata.json          # 上传元数据（MD5、时间戳）
│   ├── schema_snapshot.json        # Schema 快照（列名、类型、缺失率）
│   ├── train_raw.csv               # 训练集
│   ├── valid_raw.csv               # 验证集
│   ├── test_raw.csv                # 测试集
│   └── splitting_result.json       # 切分方案结果
├── cleaning/                       # 数据清洗产物
│   ├── cleaning_plan.md            # 用户确认的清洗方案
│   ├── cleaning_code.py            # 生成的清洗代码
│   ├── cleaned_data.csv            # 清洗后数据
│   └── cleaning_result.json        # 清洗结果统计
├── exploration/                    # 探索性分析产物
│   └── data_exploration_result.md  # 探索报告
├── features/                       # 特征工程产物
│   ├── feature_engineering_plan.md # 特征工程方案
│   ├── feature_engineering_code.py # 特征工程代码
│   ├── feat_dataset.csv            # 工程后数据
│   └── feature_engineering_result.json
├── models/                         # 模型产物
│   ├── model_training_plan.md      # 建模方案
│   ├── model_training_code.py      # 训练代码
│   ├── trained_model.pkl           # 标准化模型包 (model_package_v1)
│   ├── training_summary.json       # 训练摘要（特征名、指标、参数）
│   └── model_training_result.json
├── analysis/                       # 分析报告
│   ├── problem_definition.json     # 结构化问题定义
│   └── data_contract_report.md     # 契约检查报告
├── reports/                        # 最终报告
│   ├── modeling_report.md          # Markdown 报告
│   ├── modeling_report.html        # HTML 报告
│   └── charts/                     # 可视化图表 (PNG)
├── prompts/                        # LLM Prompt 快照
│   └── {stage}_prompt_*.md
├── code/                           # 生成的代码存档
└── state/                          # 工作流状态
    └── workflow_state.json
```

## 项目结构

```
AutoMLByLLM/
├── automl_react/                    # 核心代码库
│   ├── agents/                      # 各阶段 Agent 实现
│   │   ├── data_analysis_agent.py   #   问题定义
│   │   ├── data_contract_agent.py   #   数据契约检查（纯逻辑）
│   │   ├── data_splitting_agent.py  #   数据集切分
│   │   ├── data_cleaning_agent.py   #   数据清洗
│   │   ├── data_exploration_agent.py#   探索性分析
│   │   ├── feature_engineering_agent.py # 特征工程
│   │   ├── model_training_agent.py  #   模型训练
│   │   └── model_evaluation_agent.py#   模型评估
│   ├── api/
│   │   └── main.py                  # FastAPI 主入口（20 个端点）
│   ├── assets/
│   │   └── asset_manager.py         # 会话资产管理
│   ├── config/
│   │   ├── config_loader.py         # YAML 配置加载器（支持环境变量解析）
│   │   ├── prompts.yaml             # 所有 Agent 的 Prompt 模板
│   │   ├── llm_config.yaml          # LLM 模型配置
│   │   └── workflow_config.yaml     # 工作流阶段配置
│   ├── confirmation/
│   │   └── confirmation_point.py    # 用户确认/修改/跳过/拒绝
│   ├── core/
│   │   ├── react_agent.py           # ReAct Agent 核心循环
│   │   ├── memory.py                # 对话记忆管理
│   │   └── observation.py           # 观察结果表示
│   ├── evaluation/
│   │   └── model_evaluator.py       # 评估指标计算
│   ├── logger/
│   │   └── llm_logger.py            # LLM 调用日志（JSONL）
│   ├── report/
│   │   ├── report_generator.py      # Markdown/HTML 可视化报告
│   │   └── pipeline_generator.py    # 全流程可复现 Python 脚本
│   ├── skills_loader/
│   │   └── skill_loader.py          # Skills 知识包加载
│   ├── tools/
│   │   ├── base_tool.py             # 工具抽象基类
│   │   ├── data_tools.py            # 数据加载与分析
│   │   ├── feature_tools.py         # 特征生成
│   │   ├── model_tools.py           # 模型训练
│   │   ├── profile_tools.py         # 数据质量分析
│   │   ├── stage_tools.py           # 前序阶段结果查询
│   │   └── skill_tools.py           # Skills 搜索与读取
│   ├── utils/
│   │   ├── code_generator.py        # 代码生成与验证
│   │   ├── codeact_agent.py         # CodeAct 迭代生成执行
│   │   ├── code_executor.py         # 代码执行器
│   │   └── subprocess_executor.py   # 隔离子进程执行（超时/崩溃隔离）
│   └── workflow/
│       └── workflow_state.py        # 工作流状态机（11 个状态）
│
├── frontend/
│   └── index.html                   # 单页应用（原生 HTML/CSS/JS）
│
├── skills/                          # 专业知识包
│   ├── data-analysis-1.0.2/
│   ├── afrexai-ml-engineering-1.0.0/
│   └── ml-model-eval-benchmark-0.1.0/
│
├── tests/                           # 测试文件
│   ├── test_new_tools.py
│   ├── test_p3.py
│   ├── test_plan_revision.py
│   └── test_subprocess_executor.py
│
├── logs/                            # 运行时日志
│   └── llm_calls/                   # LLM 调用日志（per session）
│
├── assets/                          # 运行时生成的会话资产
│
├── pyproject.toml                   # 项目元数据与依赖
├── requirements.txt                 # Python 依赖
└── README.md
```

## 技术栈

| 领域 | 技术 |
|------|------|
| **Agent 架构** | ReAct (Reasoning + Acting) + CodeAct（迭代代码生成执行） |
| **Web 框架** | FastAPI + Uvicorn |
| **流式通信** | Server-Sent Events (SSE) |
| **前端** | 原生 HTML/CSS/JS + marked.js (Markdown) + highlight.js (代码高亮) |
| **数据处理** | Pandas + NumPy |
| **机器学习** | Scikit-learn |
| **可视化** | Matplotlib + Seaborn |
| **配置管理** | PyYAML（支持 `${ENV_VAR}` / `${ENV_VAR:default}` 环境变量引用） |
| **LLM 支持** | OpenAI 兼容 API（Kimi、DeepSeek、GPT、Claude、通义千问等） |
| **代码执行** | 隔离子进程（pickle 序列化，超时控制，崩溃隔离） |
| **序列化** | Joblib（模型持久化）、JSON（状态/配置） |

## 快速开始

### 环境要求

- Python 3.9+
- 支持 OpenAI 兼容 API 的 LLM 服务

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 LLM

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

stage_models:                        # 可为不同阶段指定不同模型
  data_analysis: "kimi-k2.5"
  data_cleaning: "kimi-k2.5"
  feature_engineering: "kimi-k2.5"
  model_training: "kimi-k2.5"
```

设置对应的环境变量：

```bash
export MOONSHOT_API_KEY="your-api-key"
# 或
export OPENAI_API_KEY="your-api-key"
export DEEPSEEK_API_KEY="your-api-key"
export DASHSCOPE_API_KEY="your-api-key"
```

### 3. 启动服务

**启动后端 API**：

```bash
python -m automl_react.api.main
```

服务在 http://localhost:8000 启动。

**启动前端**（可选）：

```bash
# 方式一：用 http server
cd frontend && python -m http.server 8080
# 访问 http://localhost:8080

# 方式二：直接打开
open frontend/index.html
```

## API 接口

### 完整端点列表

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | API 信息 |
| **工作流控制** | | |
| `/workflow/start` | POST | 启动新工作流（session_id, data_path, target_column, task_type） |
| `/workflow/{session_id}/status` | GET | 获取工作流状态 |
| `/workflow/{session_id}/stage/{stage}/run` | POST | 执行指定阶段 |
| **确认管理** | | |
| `/confirmation/submit` | POST | 提交确认决策 |
| `/confirmation/revise` | POST | 修订方案（多轮修订） |
| `/confirmation/{session_id}/pending` | GET | 获取当前待确认项 |
| **对话接口** | | |
| `/chat` | POST | 同步对话 |
| `/chat/stream` | GET | 流式对话（SSE） |
| **资产管理** | | |
| `/assets/{session_id}/{asset_type}/{filename}` | GET | 下载资产文件 |
| `/assets/{session_id}/list` | GET | 列出会话所有资产 |
| **会话管理** | | |
| `/sessions` | GET | 列出所有会话 |
| `/sessions/{session_id}/status` | GET | 获取会话状态 |
| `/sessions/{session_id}` | DELETE | 删除会话 |
| **Skills** | | |
| `/skills/list` | GET | 列出可用技能包 |
| `/skills/{skill_name}/content` | GET | 获取技能内容 |
| **报告与脚本** | | |
| `/report/generate` | POST | 生成建模分析报告（Markdown + HTML） |
| `/report/{session_id}/summary` | GET | 获取报告摘要 |
| `/pipeline/generate` | POST | 生成全流程可复现 Python 脚本 |
| **日志** | | |
| `/logs/{session_id}/llm` | GET | 获取 LLM 调用日志 |

### 工作流阶段标识

| 标识 | 说明 |
|------|------|
| `data_upload` | 数据上传 |
| `problem_definition` | 问题定义 |
| `data_contract_check` | 数据契约检查 |
| `data_splitting` | 数据集切分 |
| `data_cleaning` | 数据清洗 |
| `data_exploration` | 数据探索性分析 |
| `feature_engineering` | 特征工程 |
| `model_training` | 模型训练 |
| `model_evaluation` | 模型评估 |

## 配置说明

### 三个核心配置文件

#### 1. Prompt 配置 (`automl_react/config/prompts.yaml`)

为每个 Agent 定义完整的 Prompt 模板，包括：
- `system_prompt`：角色定义、职责、推理方法、数据完整性约束
- `plan_generation`：方案生成提示词
- `code_generation_full`：代码生成提示词
- `plan_revision`：方案修订提示词（支持多轮修订）

**关键特点**：所有 Prompt 强制要求"当前数据事实快照"，防止 LLM 被历史文本污染。

#### 2. LLM 配置 (`automl_react/config/llm_config.yaml`)

```yaml
default_model: "kimi-k2.5"

models:
  kimi-k2.5:
    provider: "openai"
    model_name: "kimi-k2.5"
    temperature: 1.0
    max_tokens: 65536
    api_key: "${MOONSHOT_API_KEY}"       # 支持环境变量引用
    base_url: "https://api.moonshot.cn/v1"

stage_models:                            # 阶段级模型覆盖
  data_analysis: "kimi-k2.5"
  model_training: "kimi-k2.5"

logging:
  enabled: true
  log_dir: "logs/llm_calls"
  log_format: "jsonl"
```

#### 3. 工作流配置 (`automl_react/config/workflow_config.yaml`)

```yaml
react_agent:
  max_iterations: 10                     # ReAct 最大迭代次数
  timeout: 300                           # 超时秒数

stages:
  data_cleaning:
    required: true
    confirmation_required: true
    skills: ["data-analysis-1.0.2"]

confirmation:
  timeout: 3600
  auto_skip: false
  default_action: "confirm"

assets:
  save_code: true
  save_models: true
```

## LLM 调用日志

每次 LLM 调用记录为 JSONL 格式，存储在 `logs/llm_calls/{session_id}/YYYYMMDD.jsonl`：

```json
{
  "session_id": "session_123",
  "timestamp": "2024-04-08T10:30:00",
  "model_name": "kimi-k2.5",
  "provider": "openai",
  "stage": "data_cleaning",
  "input_content": "...",
  "output_content": "...",
  "input_tokens": 2000,
  "output_tokens": 1500,
  "latency_ms": 3500,
  "metadata": {}
}
```

## 前端功能

单页应用（SPA），无需构建工具，直接浏览器打开：

- **工作流可视化**：9 个阶段进度状态实时展示（未开始/进行中/完成）
- **方案展示**：Markdown 渲染 + Python 代码高亮
- **用户确认面板**：Confirm / Modify / Skip / Reject 按钮
- **流式输出**：SSE 实时显示 LLM 生成的长内容
- **资产管理**：下载数据、模型、报告等生成文件
- **聊天界面**：与 Agent 实时对话交互
- **Skills 参考**：展示当前阶段引用的专业知识

**界面布局**：左侧为控制面板（上传数据、配置任务、阶段切换），右侧为内容区（方案展示、确认交互、资产列表）。

## 使用流程

1. **启动工作流**：上传数据文件，指定目标列和任务类型
2. **问题定义**：系统自动分析数据事实，生成结构化问题定义
3. **数据契约检查**：自动检查建模可行性（样本量、数据质量、泄漏风险）
4. **数据集切分**：审阅和确认 train/valid/test 切分方案
5. **数据清洗**：审阅质量分析报告和清洗方案 → 确认/修改 → 执行
6. **数据探索**：自动生成统计分析报告，为后续阶段提供依据
7. **特征工程**：审阅特征方案 → 确认/修改 → 执行
8. **模型训练**：审阅建模方案 → 确认/修改 → 训练执行
9. **模型评估**：测试集最终评估，复核训练指标
10. **结果下载**：下载模型包（`.pkl`）、全流程脚本、分析报告

## 核心设计原则

| 原则 | 说明 |
|------|------|
| **数据事实优先** | 所有方案基于实际数据统计，禁止臆造字段或示例数据 |
| **人在回路** | 关键阶段支持确认/修改/跳过，尊重用户业务知识 |
| **可审计与可复现** | JSONL 日志 + 资产保存 + 状态持久化，支持断点恢复 |
| **配置驱动** | YAML 管理 Prompt 和模型，支持阶段级模型覆盖 |
| **工程最佳实践** | 严格区分数据集，防止泄漏，代码隔离执行 |
| **崩溃隔离** | LLM 生成的代码在子进程执行，主进程不受影响 |

## 许可证

MIT License
