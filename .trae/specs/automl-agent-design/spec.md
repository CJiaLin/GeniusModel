# AutoML Agent 综合建模服务规格

## Why
当前机器学习建模流程复杂，需要专业数据科学家参与。本系统旨在创建一个智能Agent，让非专业用户也能通过自然语言描述建模目标和场景，获得完整的从0到1的建模服务。

## What Changes

### 系统架构设计

核心架构包含三个层次：
1. **顶层：总控Agent (ModelingPlanner)** - 协调整个建模流程
2. **中层：领域子Agent** - 负责特定阶段的建模任务
3. **底层：工具/Skill层** - 提供原子化能力

### 核心子Agent设计

#### 1. ModelingPlanner (总控Agent)
- **职责**：理解用户建模需求，制定建模计划，协调各子Agent执行
- **输入**：用户描述的建模场景和目标
- **输出**：建模计划（包含数据要求、建模类型、评估指标等）
- **技能**：
  - IntentParser - 解析用户意图，提取建模目标
  - RequirementAnalyzer - 分析数据需求和业务约束
  - PlanGenerator - 生成阶段性建模计划

#### 2. DataAgent (数据处理Agent)
- **职责**：数据获取、探索、质量分析、清洗
- **技能**：
  - DataLoader - 支持CSV、Excel、JSON、数据库等数据源
  - DataExplorer - 数据基本信息探索（行列数、类型、分布）
  - DataQualityAnalyzer - 缺失值、异常值、重复值检测
  - DataCleaner - 数据清洗逻辑生成
  - DataVisualizer - 生成数据可视化图表

#### 3. FeatureEngineerAgent (特征工程Agent)
- **职责**：特征构建、衍生、筛选
- **技能**：
  - FeatureGenerator - 基于业务逻辑生成特征
  - FeatureDeriver - 数值特征衍生（多项式、交叉、统计特征）
  - FeatureEncoder - 类别特征编码（Label/OneHot/Target Encoding）
  - FeatureSelector - 特征重要性分析、相关性筛选、递归特征消除
  - FeatureTransformer - 标准化、归一化、分箱处理

#### 4. ModelAgent (模型Agent)
- **职责**：模型选型、训练、调参、评估
- **技能**：
  - ModelSelector - 根据问题类型推荐模型（分类/回归/聚类）
  - HyperparameterTuner - 网格搜索/贝叶斯优化/随机搜索
  - ModelTrainer - 模型训练执行
  - ModelEvaluator - 多维度评估指标计算
  - ModelComparator - 模型对比分析

#### 5. CodeExecutor (代码执行器 - MCP Server)
- **职责**：安全执行大模型生成的Python代码
- **功能**：
  - 沙箱环境隔离
  - 依赖包管理
  - 执行超时控制
  - 结果序列化返回
  - 错误捕获和日志记录

### 工具层设计

#### MCP Servers
1. **FileMCP** - 文件操作（上传、读取、下载）
2. **ExecutionMCP** - Python代码安全执行
3. **DatabaseMCP** - 数据库连接和查询
4. **VisualizationMCP** - 数据可视化生成

#### LangChain Tools
1. **Python REPL Tool** - 代码执行
2. **Search Tool** - 查找相似案例/最佳实践
3. **File Management Tools** - 文件操作

### 工作流程设计

```
用户输入 → ModelingPlanner → 建模计划
                            ↓
                      DataAgent (数据上传/探索/清洗)
                            ↓
                      FeatureEngineerAgent (特征工程)
                            ↓
                      ModelAgent (模型选型/训练/调参)
                            ↓
                      结果报告 + 模型输出
```

### 交互模式

1. **引导式交互**：Agent逐步引导用户完成每个阶段
2. **自动决策**：用户可选择让Agent自动决定最佳方案
3. **人工确认**：关键节点需要用户确认后再继续

## Impact
- Affected specs: 整个AutoML系统
- Affected code: 需要开发完整的Agent框架

## ADDED Requirements

### Requirement: 多数据源支持
系统 SHALL 支持从多种数据源获取数据，包括本地文件、URL、数据库

#### Scenario: 用户上传CSV文件
- **WHEN** 用户上传CSV文件并指定建模目标
- **THEN** Agent自动识别列类型，生成初步数据探索报告

### Requirement: 建模流程编排
系统 SHALL 按照标准建模流程逐步执行，每个阶段可配置

#### Scenario: 完整建模流程
- **WHEN** 用户提供完整数据和明确目标
- **THEN** Agent依次执行数据处理→特征工程→建模→评估全流程

### Requirement: 代码安全执行
系统 SHALL 在隔离环境中执行生成的代码，防止危险操作

#### Scenario: 执行大模型生成的代码
- **WHEN** 大模型生成Python代码请求执行
- **THEN** 代码在沙箱环境中执行，结果安全返回

### Requirement: 建模结果可解释
系统 SHALL 提供模型决策解释和可视化

#### Scenario: 分类模型结果展示
- **WHEN** 分类模型训练完成
- **THEN** 展示特征重要性、混淆矩阵、分类报告等

## MODIFIED Requirements

### Requirement: 总控Agent
[完整需求见上文]

## REMOVED Requirements

### Requirement: 无
当前为全新设计，无移除需求
