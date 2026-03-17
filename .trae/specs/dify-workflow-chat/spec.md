# Dify风格工作流对话前端规格

## Why

当前对话式AutoML应用采用Streamlit实现，缺乏类似Dify的流式输出和工作流可视化能力。用户无法直观地看到Agent思维过程、节点执行状态和实时输出，体验不够直观。

## What Changes

### 前端技术栈升级

- **现有方案**: Streamlit原生对话
- **升级方案**: React/Vue.js + SSE流式输出 + 工作流可视化

### 核心功能特性

1. **流式输出（Streaming）**
   - Server-Sent Events (SSE) 实时推送
   - 支持打字机效果逐字显示
   - 支持Markdown渲染

2. **工作流可视化**
   - 节点式流程图展示
   - 节点状态实时更新（pending/running/completed/failed）
   - 节点执行时间显示
   - 可展开查看节点详情

3. **对话界面优化**
   - 类似ChatGPT的左侧会话列表
   - 右侧对话+工作流双栏布局
   - 支持代码块语法高亮
   - 支持图片/图表展示

4. **后端API重构**
   - RESTful API + SSE端点
   - 会话管理
   - 工作流状态追踪

## Impact

- Affected specs: 需要新增前端项目，现有后端需要适配
- Affected code:
  - 新增: `ui/frontend/` - React/Vue前端项目
  - 新增: `api/routes/chat.py` - 对话API
  - 新增: `api/routes/workflow.py` - 工作流API
  - 修改: `streaming/` - 流式输出模块

## ADDED Requirements

### Requirement: 流式输出系统

系统 SHALL 支持通过Server-Sent Events进行实时流式输出。

#### Scenario: 用户发送消息触发流式响应
- **WHEN** 用户在输入框发送消息
- **THEN** 系统逐块返回响应内容，前端实时渲染，无需等待完整响应

#### Scenario: 流式输出中断
- **WHEN** 网络中断或服务器错误
- **THEN** 前端显示错误提示，用户可重试

### Requirement: 工作流节点可视化

系统 SHALL在对话界面右侧显示工作流执行状态。

#### Scenario: 完整工作流执行
- **WHEN** Agent执行数据处理→特征工程→建模流程
- **THEN** 右侧显示节点流程图，每个节点状态实时更新

#### Scenario: 节点执行失败
- **WHEN** 某节点执行出错
- **THEN** 节点变为红色失败状态，展开可查看错误详情

### Requirement: 会话管理

系统 SHALL 支持多会话管理和历史记录。

#### Scenario: 用户新建会话
- **WHEN** 用户点击新建会话按钮
- **THEN** 创建新会话，生成会话标题，可切换历史会话

#### Scenario: 用户查看历史会话
- **WHEN** 用户从左侧会话列表选择历史会话
- **THEN** 加载完整对话历史和工作流状态

### Requirement: 代码块展示

系统 SHALL 支持美化的代码块展示。

#### Scenario: Agent返回代码
- **WHEN** Agent生成Python代码
- **THEN** 前端显示带语法高亮的代码块，支持复制按钮

## MODIFIED Requirements

### Requirement: 对话流程
[完整需求见上文]

## REMOVED Requirements

### Requirement: 无
当前为新增功能，无移除需求
