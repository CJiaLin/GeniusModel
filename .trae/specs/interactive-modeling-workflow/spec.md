# 交互式建模工作流规格

## Why
当前 AutoML 系统虽然实现了基础建模功能，但缺乏用户交互和确认机制。用户需要在每个关键节点（数据清洗、特征工程、建模方案）进行确认和修改，以确保建模结果符合业务需求。本规格定义一个支持用户交互确认的完整建模工作流，并充分利用 skills 目录中的专业知识。

## What Changes

### 1. 交互式工作流架构

引入用户确认节点到 ReAct Agent 循环中：

```
用户输入 → Agent思考 → 生成方案 → 用户确认/修改 → 生成代码 → 执行 → 观察结果 → (循环)
```

### 2. Skills 集成设计

每个环节都可以调用或参考 skills 内的对应内容：

| 工作流阶段 | 使用的 Skill | 参考内容 |
|-----------|-------------|---------|
| 数据分析 | `data-analysis-1.0.2` | metric-contracts.md, chart-selection.md, pitfalls.md |
| 数据清洗 | `data-analysis-1.0.2` | techniques.md, decision-briefs.md |
| 特征工程 | `afrexai-ml-engineering-1.0.0` | Phase 2: Data Engineering for ML |
| 模型选型 | `afrexai-ml-engineering-1.0.0` | Phase 3: Model Selection Guide |
| 模型评估 | `ml-model-eval-benchmark-0.1.0` | benchmarking-guide.md |
| 模型部署 | `afrexai-ml-engineering-1.0.0` | Phase 5: Model Deployment |

### 3. 配置化设计

所有 Prompt 和大模型配置从配置文件读取：
- `config/prompts.yaml` - Prompt 模板配置
- `config/llm_config.yaml` - 大模型参数配置
- `config/workflow_config.yaml` - 工作流参数配置

**禁止在代码中硬编码任何 prompt 或大模型配置。**

### 4. 日志与资产保存

每次大模型调用的输入输出日志必须记录：
- 请求时间、模型名称、输入内容
- 响应时间、输出内容、Token 消耗
- 保存路径：`logs/llm_calls/{session_id}/{timestamp}.jsonl`

用户交互过程中生成的资产必须保存并提供下载：
- 原始数据：`assets/{session_id}/data/`
- 清洗后数据：`assets/{session_id}/cleaned_data/`
- 特征数据：`assets/{session_id}/features/`
- 生成代码：`assets/{session_id}/code/`
- 训练好的模型：`assets/{session_id}/models/`
- 可视化报告：`assets/{session_id}/reports/`
- 全流程脚本：`assets/{session_id}/pipeline.py`

### 5. 新增工作流阶段

#### 阶段1: 数据上传与分析
- 数据上传
- 自动数据分析（参考 `data-analysis` skill）
- 生成数据质量报告

#### 阶段2: 数据清洗（含用户确认）
- Agent生成数据清洗思路（参考 `data-analysis` skill 的 techniques.md）
- 用户查看并确认/修改清洗思路
- 根据确认的思路生成清洗代码
- 执行代码并返回清洗结果
- 保存清洗后的数据

#### 阶段3: 特征工程（含用户确认）
- 基于清洗后数据生成特征工程思路（参考 `afrexai-ml-engineering` skill 的 Phase 2）
- 用户查看并确认/修改特征思路
- 根据确认的思路生成特征代码
- 执行代码并返回特征结果
- 保存特征数据

#### 阶段4: 建模方案设计（含用户确认）
- 基于特征结果生成建模方案（参考 `afrexai-ml-engineering` skill 的 Phase 3）
- 用户查看并确认/修改建模方案
- 根据确认的方案生成建模代码
- 执行代码并训练模型
- 保存模型文件

#### 阶段5: 模型评估与结果输出
- 模型评估（参考 `ml-model-eval-benchmark` skill）
- 返回模型训练指标
- 保存模型文件
- 生成全流程建模脚本
- 生成可视化分析报告（参考 `data-analysis` skill 的 chart-selection.md）
- 提供所有资产下载

### 6. 前后端交互架构

#### 后端架构
- **ReAct Agent**: 核心决策引擎
- **SkillLoader**: Skills 加载和调用
- **WorkflowState**: 工作流状态管理
- **UserConfirmationPoint**: 用户确认点管理
- **CodeGenerator**: 代码生成器
- **CodeExecutor**: 代码执行器
- **LLMLogger**: 大模型调用日志记录器
- **AssetManager**: 资产文件管理器
- **ConfigLoader**: 配置文件加载器

#### 前端交互
- 流式响应 (SSE) 展示 Agent 思考过程
- 方案展示面板（支持 Markdown 渲染）
  - 大模型生成的内容（思路、方案、说明）使用 Markdown 格式渲染
  - 支持代码块、表格、列表等 Markdown 元素
  - 支持语法高亮
- 方案编辑界面（用户修改）
- 代码预览界面（支持语法高亮）
- 执行结果可视化
- 资产下载界面

### 7. 确认点设计

每个确认点包含：
- **方案内容**: 结构化方案描述
- **参考的 Skill**: 显示参考了哪些专业知识
- **代码预览**: 将要执行的代码
- **预期效果**: 执行后的预期结果
- **用户操作**: 确认 / 修改 / 跳过

### 8. 代码清理

实施完成后，必须执行代码清理：
- 删除历史无关代码
- 删除未使用的文件
- 删除硬编码的 prompt（迁移到配置文件）
- 删除硬编码的配置（迁移到配置文件）
- 统一代码风格

## Impact
- Affected specs: automl-agent-design
- Affected code: automl_react/ 目录下所有模块
- 新增模块: workflow/, confirmation/, report/, skills_loader/, logger/, assets/
- 删除: 历史无关代码文件

## ADDED Requirements

### Requirement: Skills 加载与调用机制
系统 SHALL 提供统一的 Skills 加载和调用机制，让每个工作流环节都可以访问 skills 目录内的专业知识。

#### Scenario: 加载 data-analysis skill
- **GIVEN** 系统需要进行数据分析
- **WHEN** Agent 调用 SkillLoader
- **THEN** 返回 data-analysis skill 的内容和参考文件

#### Scenario: 参考 ML Engineering 最佳实践
- **GIVEN** 系统需要进行特征工程
- **WHEN** Agent 查询特征工程相关知识
- **THEN** 返回 afrexai-ml-engineering skill 中 Phase 2 的内容

### Requirement: 配置化 Prompt 和模型配置
系统 SHALL 从配置文件读取所有 Prompt 和大模型配置，禁止在代码中硬编码。

#### Scenario: 加载 Prompt 配置
- **GIVEN** 系统需要生成数据清洗思路
- **WHEN** Agent 请求数据清洗 Prompt
- **THEN** 从 `config/prompts.yaml` 加载对应的 Prompt 模板

#### Scenario: 加载 LLM 配置
- **GIVEN** 系统需要调用大模型
- **WHEN** Agent 初始化 LLM 客户端
- **THEN** 从 `config/llm_config.yaml` 加载模型参数

### Requirement: 大模型调用日志记录
系统 SHALL 记录每次大模型调用的输入输出日志。

#### Scenario: 记录 LLM 调用
- **GIVEN** Agent 调用大模型
- **WHEN** 收到模型响应
- **THEN** 记录请求时间、模型名称、输入、输出、Token 消耗到 `logs/llm_calls/`

#### Scenario: 查询 LLM 调用历史
- **GIVEN** 用户需要查看调用历史
- **WHEN** 请求查询接口
- **THEN** 返回该会话的所有 LLM 调用记录

### Requirement: 用户确认机制
系统 SHALL 在每个关键阶段提供用户确认机制，允许用户查看、修改或跳过 Agent 生成的方案，并显示参考的 Skill 来源。大模型生成的内容使用 Markdown 格式，前端进行渲染。

#### Scenario: 数据清洗确认
- **GIVEN** Agent 已生成数据清洗思路
- **WHEN** 系统展示清洗思路给用户
- **THEN** 用户可以确认执行、修改思路或跳过此步骤，并看到参考的 skill 来源，思路内容以 Markdown 格式渲染

#### Scenario: 特征工程确认
- **GIVEN** Agent 已生成特征工程思路
- **WHEN** 系统展示特征思路给用户
- **THEN** 用户可以确认执行、修改思路或跳过此步骤，并看到参考的 skill 来源，思路内容以 Markdown 格式渲染

#### Scenario: 建模方案确认
- **GIVEN** Agent 已生成建模方案
- **WHEN** 系统展示建模方案给用户
- **THEN** 用户可以确认执行、修改方案或跳过此步骤，并看到参考的 skill 来源，方案内容以 Markdown 格式渲染

### Requirement: 方案修改与重新生成
系统 SHALL 支持用户修改方案后，Agent 根据修改后的要求重新生成代码，并可选择参考不同的 skills。

#### Scenario: 修改数据清洗思路
- **GIVEN** 用户正在查看数据清洗思路
- **WHEN** 用户提出修改要求（如"增加异常值处理"）
- **THEN** Agent 根据新要求重新生成清洗思路和代码，可引用 data-analysis skill 的 pitfalls.md

#### Scenario: 修改特征工程思路
- **GIVEN** 用户正在查看特征工程思路
- **WHEN** 用户提出修改要求（如"增加交叉特征"）
- **THEN** Agent 根据新要求重新生成特征思路和代码，可引用 afrexai-ml-engineering skill

### Requirement: 代码执行与结果反馈
系统 SHALL 在用户确认后执行生成的代码，并将执行结果反馈给用户。

#### Scenario: 执行数据清洗代码
- **GIVEN** 用户已确认数据清洗思路
- **WHEN** 系统执行生成的清洗代码
- **THEN** 返回清洗后的数据摘要和执行日志

#### Scenario: 执行特征工程代码
- **GIVEN** 用户已确认特征工程思路
- **WHEN** 系统执行生成的特征代码
- **THEN** 返回新生成的特征列表和统计信息

### Requirement: 资产保存与下载
系统 SHALL 保存用户交互过程中生成的所有资产，并提供下载接口。

#### Scenario: 保存清洗后的数据
- **GIVEN** 数据清洗已完成
- **WHEN** 系统保存清洗结果
- **THEN** 数据保存到 `assets/{session_id}/cleaned_data/`

#### Scenario: 保存训练好的模型
- **GIVEN** 模型训练已完成
- **WHEN** 系统保存模型
- **THEN** 模型保存到 `assets/{session_id}/models/`

#### Scenario: 下载全流程脚本
- **GIVEN** 用户请求下载脚本
- **WHEN** 调用下载接口
- **THEN** 返回 `assets/{session_id}/pipeline.py` 文件

#### Scenario: 下载可视化报告
- **GIVEN** 用户请求下载报告
- **WHEN** 调用下载接口
- **THEN** 返回 `assets/{session_id}/reports/` 中的报告文件

### Requirement: 全流程建模脚本生成
系统 SHALL 在模型训练完成后，生成可独立运行的全流程建模脚本。

#### Scenario: 生成建模脚本
- **GIVEN** 模型训练已完成
- **WHEN** 用户请求生成全流程脚本
- **THEN** 系统生成包含数据加载、清洗、特征工程、模型训练的完整 Python 脚本，保存到 `assets/{session_id}/pipeline.py`

### Requirement: Markdown 内容渲染
系统 SHALL 确保大模型生成的所有内容（思路、方案、说明、报告）使用 Markdown 格式，前端使用 Markdown 渲染器正确渲染。

#### Scenario: 渲染数据清洗思路
- **GIVEN** Agent 生成了数据清洗思路
- **WHEN** 前端展示思路内容
- **THEN** 使用 Markdown 渲染器渲染，支持代码块、表格、列表等元素

#### Scenario: 渲染建模方案
- **GIVEN** Agent 生成了建模方案
- **WHEN** 前端展示方案内容
- **THEN** 使用 Markdown 渲染器渲染，支持代码块、表格、列表等元素

### Requirement: 可视化分析报告生成
系统 SHALL 生成包含数据分布、特征重要性、模型评估指标的可视化分析报告，参考 data-analysis skill 的 chart-selection.md。报告使用 Markdown 格式，前端渲染。

#### Scenario: 生成分析报告
- **GIVEN** 建模流程已完成
- **WHEN** 系统生成分析报告
- **THEN** 报告包含数据概览、清洗记录、特征工程记录、模型性能图表，使用 chart-selection.md 中的图表选择指南，以 Markdown 格式保存到 `assets/{session_id}/reports/`，前端使用 Markdown 渲染

### Requirement: 代码清理
系统 SHALL 在功能实现完成后，清理历史无关代码。

#### Scenario: 删除历史代码
- **GIVEN** 新功能已实现
- **WHEN** 执行代码清理
- **THEN** 删除未使用的文件、硬编码的 prompt、历史遗留代码

## MODIFIED Requirements

### Requirement: ReAct Agent 扩展
扩展 ReAct Agent 以支持：
- 用户确认点
- Skills 调用
- 配置化 Prompt 加载
- LLM 调用日志记录
- 暂停等待用户输入
- 从用户修改中恢复执行

## REMOVED Requirements

无
