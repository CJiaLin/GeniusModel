# AutoML Agent - 智能建模服务系统

## 项目简介

基于 LangChain + LangGraph 开发的综合 AutoML Agent 系统，提供从数据导入到模型评估的完整自动化机器学习建模流程。系统支持传统特征工程和 LLM 驱动的智能特征生成两种模式，并通过交互式对话方式让用户全程参与建模决策。

## 核心功能

### 1. 智能建模流程编排
- 自动解析用户建模意图（分类/回归/聚类等任务类型）
- 完整的端到端建模流程：数据加载 → 质量分析 → 数据清洗 → 特征工程 → 模型训练 → 模型评估
- 支持交互式对话式建模，每步等待用户确认

### 2. 数据自动探索与清洗
- 支持多种数据格式加载（CSV、Excel、JSON）
- 自动检测数据类型（数值型、类别型）
- 全面数据质量分析（缺失值、重复值、异常值检测）
- 智能数据清洗策略（自动填充缺失值、去重、异常值处理）

### 3. 特征工程自动化

**传统模式**：
- 交互特征生成（乘积、商）
- 聚合特征生成（分组统计）
- 多项式特征衍生
- 统计特征生成

**LLM 驱动模式（核心特色）**：
- LLM 分析数据结构和业务场景
- 自主思考生成特征方向
- 生成具体特征加工代码并执行
- 特征质量评估（IV 值、相关性分析）
- 基于质量报告的智能特征选择

### 4. 模型自动选择与调参
- 根据任务类型和数据特征自动推荐模型
- 支持多种模型：LogisticRegression、RandomForest、XGBoost、LightGBM、SVM 等
- 自动超参数配置
- 多模型性能比较，选出最佳模型

### 5. 安全代码执行环境
- 沙箱环境执行用户生成的代码
- 防止恶意代码执行
- 支持代码执行结果验证

### 6. API 服务
- FastAPI 提供 RESTful API
- 支持 WebSocket 实时流式输出
- 跨域支持，便于集成

## 项目架构

```
AutoMLByLLM/
├── agents/                    # 核心 Agent 模块
│   ├── base_agent.py         # Agent 基类
│   ├── modeling_planner.py   # 建模计划器（总控 Agent）
│   ├── data_agent.py         # 数据处理 Agent
│   ├── feature_engineer.py   # 特征工程 Agent（支持 LLM 驱动）
│   └── model_agent.py        # 模型训练 Agent
│
├── automl_agent/             # AutoML 核心引擎
│   ├── engine.py            # 主引擎
│   ├── interactive.py       # 对话式交互引擎
│   ├── models.py            # 数据模型定义
│   ├── enums.py             # 枚举类型定义
│   ├── core/                # 核心组件
│   │   ├── executor.py     # 代码执行器
│   │   └── protocol.py     # 通信协议
│   └── mcp/                 # MCP 协议实现
│
├── api/                      # FastAPI 服务
│   ├── main.py             # API 主入口
│   └── routes/             # API 路由
│       └── chat_api.py     # 对话 API
│
├── tools/                    # MCP 标准化工具
│   ├── data_tools.py       # 数据处理工具
│   ├── model_tools.py      # 模型工具
│   ├── feature_tools.py    # 特征工程工具
│   └── eval_tools.py       # 评估工具
│
├── core/                     # 核心功能
│   ├── pipeline.py          # 流程编排
│   ├── state.py            # 状态管理
│   └── prompt_loader.py    # 提示词加载
│
├── ui/                       # 前端界面
│   ├── frontend/           # Web 前端（HTML/JS）
│   └── dialog_app.py       # Streamlit 应用
│
├── skills/                   # 技能模块
│   ├── afrexai-ml-engineering-1.0.0/    # ML 工程方法论
│   ├── data-analysis-1.0.2/              # 数据分析方法论
│   └── ml-model-eval-benchmark-0.1.0/   # 模型评估基准
│
├── prompts/                  # 提示词模板
│   ├── planner_prompts.yaml
│   ├── data_prompts.yaml
│   ├── feature_prompts.yaml
│   └── model_prompts.yaml
│
└── config.yaml              # 配置文件
```

## 技术栈

- **语言模型**: LangChain + LangGraph
- **Web 框架**: FastAPI + Uvicorn
- **数据处理**: Pandas + NumPy
- **机器学习**: Scikit-learn + XGBoost + LightGBM
- **数据验证**: Pydantic
- **可视化**: Matplotlib + Seaborn
- **前端**: HTML/JS + Streamlit

## 快速开始

### 环境要求

- Python 3.9+
- API Key（支持 OpenAI 兼容 API，如 Kimi、Claude、GPT 等）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 LLM

编辑 `config.yaml` 文件：

```yaml
llm:
  base_url: "https://api.moonshot.cn"    # API 端点
  api_key: "your-api-key"                 # 你的 API Key
  model: "kimi-k2-0905-preview"           # 模型名称
  temperature: 0
  timeout: 60
  max_retries: 3
```

支持的 API 端点：
- Kimi: `https://api.moonshot.cn`
- OpenAI: `https://api.openai.com`
- Claude: `https://api.anthropic.com`
- 其他兼容 OpenAI API 的服务

---

## 服务启动

### 方式一：后端 API 服务

```bash
# 启动后端 API 服务（默认端口 8000）
python -m api.main

# 或使用 uvicorn 指定端口
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

后端服务启动后访问：
- API 文档：http://localhost:8000/docs
- ReDoc 文档：http://localhost:8000/redoc

### 方式二：Web 前端 + 后端 API

**终端 1 - 启动后端**：
```bash
python -m api.main
```

**终端 2 - 启动前端**：
```bash
python -m http.server 8080 --directory ui/frontend
```

访问：http://localhost:8080

### 方式三：Streamlit 应用

```bash
streamlit run ui/dialog_app.py
```

访问：http://localhost:8501

### 方式四：命令行建模

```bash
# 交互式建模
python main.py --goal "预测房价" --data house.csv --target price --use-llm-features

# 非交互模式
python main.py --goal "预测流失" --data churn.csv --target churn --no-interactive
```

---

## API 接口说明

### 主要接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/chat/{session_id}` | WebSocket | 对话式建模（流式输出） |
| `/upload/{session_id}` | POST | 上传数据文件 |
| `/model/download/{session_id}` | GET | 下载训练好的模型 |
| `/report/download/{session_id}` | GET | 下载建模报告 |
| `/data/download/cleaned/{session_id}` | GET | 下载清洗后的数据 |
| `/data/download/featured/{session_id}` | GET | 下载特征工程后的数据 |

### WebSocket 对话示例

```javascript
const ws = new WebSocket('ws://localhost:8000/chat/session-123');

// 发送消息
ws.send(JSON.stringify({
    type: 'message',
    content: '我想预测用户是否流失'
}));

// 接收流式响应
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.content);
};
```

---

## 编程使用

```python
from automl_agent.engine import AutoMLEngine
from llm_client import configure_llm, get_llm_client, set_system_prompt

# 配置 LLM（可选，默认从 config.yaml 读取）
configure_llm(
    base_url="https://api.moonshot.cn",
    api_key="your-api-key",
    model="kimi-k2-0905-preview"
)

# 自定义 System Prompt（可选）
set_system_prompt("你是一位专业的AutoML专家...")

# 获取 LLM 客户端
llm = get_llm_client()

# 创建引擎并运行
engine = AutoMLEngine(llm)
result = engine.run(
    user_goal="预测用户是否流失",
    data_path="data.csv",
    target_column="churn",
    use_llm_features=True  # 使用 LLM 特征生成
)

print(f"模型准确率: {result.metrics['accuracy']}")
```

---

## 配置说明

### 配置文件 (config.yaml)

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `base_url` | API 端点 | `https://api.moonshot.cn` |
| `api_key` | API 密钥 | - |
| `model` | 模型名称 | `kimi-k2-0905-preview` |
| `temperature` | 温度参数 | `0` |
| `timeout` | 超时时间（秒） | `60` |
| `max_retries` | 最大重试次数 | `3` |

### 命令行参数

```
--goal              建模目标描述（必填）
--data              数据文件路径（必填）
--target            目标列名（必填）
--use-llm-features  使用 LLM 特征生成
--no-interactive    非交互模式
```

---

## 项目特色

1. **LLM 驱动的特征工程**: 利用大语言模型的推理能力，自动分析数据场景，生成有业务意义的特征
2. **全程交互式建模**: 用户可以参与每个决策步骤，确保模型符合业务需求
3. **代码执行安全**: 沙箱环境执行生成的代码，保证系统安全
4. **灵活的 API 集成**: FastAPI 提供完整的 RESTful 接口，便于二次开发
5. **多前端支持**: 提供 Web 前端和 Streamlit 两种界面
6. **技能模块集成**: 内置 ML 工程方法论和数据分析最佳实践

---

## 许可证

MIT License
