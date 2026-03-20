# AutoML ReAct - 交互式智能建模系统

## 项目简介

基于 **ReAct (Reasoning + Acting)** Agent 架构开发的交互式 AutoML 系统，提供从数据上传到模型部署的完整自动化机器学习建模流程。系统支持用户在每个关键阶段（数据清洗、特征工程、模型训练）进行确认和干预，确保建模过程符合业务需求。

## 核心功能

### 1. 交互式建模流程
- **数据上传**: 支持 CSV、Excel、JSON 等多种格式
- **数据分析**: 自动生成数据质量分析报告
- **数据清洗**: 生成清洗方案 → 用户确认 → 执行清洗
- **特征工程**: 生成特征方案 → 用户确认 → 执行特征工程
- **模型训练**: 生成建模方案 → 用户确认 → 训练模型
- **结果输出**: 模型文件、全流程脚本、可视化报告

### 2. 用户确认机制
每个关键阶段都支持用户确认：
- ✅ **确认**: 按方案执行
- ✏️ **修改**: 提供修改意见后重新生成方案
- ⏭️ **跳过**: 跳过当前阶段

### 3. Skills 集成
系统自动调用 Skills 目录中的专业知识：
- **data-analysis-1.0.2**: 数据分析方法论、数据陷阱识别
- **afrexai-ml-engineering-1.0.0**: ML 工程最佳实践
- **ml-model-eval-benchmark-0.1.0**: 模型评估基准

### 4. 资产管理系统
自动保存所有生成的资产：
- 清洗后的数据文件
- 特征工程后的数据文件
- 生成的 Python 代码
- 训练好的模型文件
- 建模流程报告

### 5. LLM 调用日志
记录每次 LLM 调用的完整信息：
- 输入（System Prompt + User Prompt）
- 输出（Response）
- 时间戳和元数据
- 存储格式：JSONL

### 6. 配置驱动
所有 Prompt 和 LLM 配置都通过 YAML 文件管理：
- `prompts.yaml`: 所有 Agent 的 Prompt 模板
- `llm_config.yaml`: LLM 模型配置、API 密钥、重试策略

## 项目架构

```
AutoMLByLLM/
├── automl_react/                 # 核心代码库
│   ├── agents/                   # Agent 模块
│   │   ├── data_cleaning_agent.py      # 数据清洗 Agent
│   │   ├── feature_engineering_agent.py # 特征工程 Agent
│   │   ├── model_training_agent.py      # 模型训练 Agent
│   │   ├── automl_agent.py              # AutoML 主 Agent
│   │   └── orchestrator.py              # 流程编排器
│   ├── api/                      # FastAPI 后端
│   │   └── main.py               # API 主入口（含 SSE 流式输出）
│   ├── assets/                   # 资产管理
│   │   └── asset_manager.py      # 资产保存和下载
│   ├── config/                   # 配置管理
│   │   ├── config_loader.py      # 配置加载器
│   │   ├── prompts.yaml          # Prompt 配置
│   │   └── llm_config.yaml       # LLM 配置
│   ├── confirmation/             # 用户确认机制
│   │   └── confirmation_point.py # 确认点实现
│   ├── core/                     # 核心组件
│   │   ├── react_agent.py        # ReAct Agent 基类
│   │   ├── memory.py             # 记忆管理
│   │   └── observation.py        # 观察结果
│   ├── evaluation/               # 模型评估
│   │   └── model_evaluator.py    # 评估指标计算
│   ├── logger/                   # 日志记录
│   │   └── llm_logger.py         # LLM 调用日志
│   ├── report/                   # 报告生成
│   │   ├── report_generator.py   # Markdown 报告
│   │   └── pipeline_generator.py # 全流程脚本
│   ├── skills_loader/            # Skills 加载
│   │   └── skill_loader.py       # Skill 内容加载
│   ├── tools/                    # 工具集
│   │   ├── base_tool.py          # 工具基类
│   │   ├── data_tools.py         # 数据处理工具
│   │   ├── feature_tools.py      # 特征工程工具
│   │   └── model_tools.py        # 模型工具
│   ├── workflow/                 # 工作流管理
│   │   └── workflow_state.py     # 工作流状态
│   └── README.md
│
├── frontend/                     # 前端界面
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
default_model: "gpt-4"

models:
  gpt-4:
    provider: "openai"
    model_name: "gpt-4"
    temperature: 0.1
    max_tokens: 4096
    api_key: "${OPENAI_API_KEY}"  # 从环境变量读取
    base_url: null
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
| `/chat/stream` | POST | 流式对话（SSE） |
| `/assets/{session_id}/{asset_type}/{filename}` | GET | 下载资产文件 |
| `/report/generate` | POST | 生成建模报告 |
| `/pipeline/generate` | POST | 生成全流程脚本 |

### 流式对话示例

```javascript
const response = await fetch('http://localhost:8000/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        session_id: 'session-123',
        message: '分析这个数据文件'
    })
});

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
  code_generation: "..."
  code_generation_full: "..."

feature_engineering:
  system_prompt: "..."
  plan_generation: "..."
  ...
```

### LLM 配置 (`automl_react/config/llm_config.yaml`)

```yaml
default_model: "gpt-4"

models:
  gpt-4:
    provider: "openai"
    model_name: "gpt-4"
    temperature: 0.1
    max_tokens: 4096
    api_key: "${OPENAI_API_KEY}"
    base_url: null

stage_models:
  data_analysis: "gpt-4"
  data_cleaning: "gpt-4"
  feature_engineering: "gpt-4"
  model_training: "gpt-4"

logging:
  enabled: true
  log_dir: "logs/llm_calls"
  log_format: "jsonl"
```

## 使用流程

1. **上传数据**: 在网页上选择数据文件上传
2. **数据分析**: 系统自动分析数据质量
3. **数据清洗**: 
   - 系统生成清洗方案（Markdown 格式）
   - 用户确认或修改方案
   - 执行清洗并保存结果
4. **特征工程**:
   - 系统生成特征工程方案
   - 用户确认或修改方案
   - 执行特征工程并保存结果
5. **模型训练**:
   - 系统生成建模方案
   - 用户确认或修改方案
   - 训练模型并评估
6. **结果下载**:
   - 下载训练好的模型
   - 下载全流程 Python 脚本
   - 下载建模分析报告

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
