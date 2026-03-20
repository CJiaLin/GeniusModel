# 工作流阶段间数据传递优化 Spec

## Why
当前各阶段（数据分析、数据清洗、特征工程、模型训练）比较割裂和独立，前一阶段的分析结果和输出数据没有作为后续阶段的输入。这导致后续 Agent 需要重复进行分析，浪费资源且可能导致不一致。

## What Changes
- 数据清洗阶段应接收数据分析报告作为输入
- 特征工程阶段应接收数据分析报告、数据清洗报告和清洗后的数据作为输入
- 模型训练阶段应接收数据分析报告、数据清洗报告、特征工程报告和特征工程后的数据作为输入
- 各阶段 Agent 应接收所有前一阶段的结果作为上下文

## Impact
- Affected code: 
  - `automl_react/agents/data_cleaning_agent.py`
  - `automl_react/agents/feature_engineering_agent.py`
  - `automl_react/agents/model_training_agent.py`
  - `automl_react/api/main.py`

## ADDED Requirements

### Requirement: 阶段间上下文传递
系统 SHALL 在各阶段之间传递完整的上下文数据，确保后续阶段可以访问所有前一阶段的分析结果和处理后的数据。

#### Scenario: 数据清洗阶段接收分析报告
- **WHEN** 数据清洗阶段开始
- **THEN** 系统 SHALL 读取数据分析报告
- **AND** 将分析报告作为上下文传递给数据清洗 Agent
- **AND** 数据清洗 Agent 基于分析报告制定清洗方案，不重复分析数据

#### Scenario: 特征工程阶段接收完整上下文
- **WHEN** 特征工程阶段开始
- **THEN** 系统 SHALL 读取数据分析报告
- **AND** 系统 SHALL 读取数据清洗报告
- **AND** 系统 SHALL 获取清洗后的数据路径
- **AND** 将以上内容作为上下文传递给特征工程 Agent
- **AND** 特征工程 Agent 基于清洗后的数据和分析结果制定特征方案

#### Scenario: 模型训练阶段接收完整上下文
- **WHEN** 模型训练阶段开始
- **THEN** 系统 SHALL 读取数据分析报告
- **AND** 系统 SHALL 读取数据清洗报告
- **AND** 系统 SHALL 读取特征工程报告
- **AND** 系统 SHALL 获取特征工程后的数据路径
- **AND** 将以上内容作为上下文传递给模型训练 Agent
- **AND** 模型训练 Agent 基于特征数据和所有分析结果制定建模方案

## MODIFIED Requirements

### Requirement: DataCleaningAgent.generate_cleaning_plan
方法 SHALL 接收数据分析报告作为可选参数：
- 如果提供了分析报告，直接基于报告制定清洗方案
- 如果没有提供分析报告，自行分析数据

### Requirement: FeatureEngineeringAgent.generate_feature_plan
方法 SHALL 接收完整上下文作为可选参数：
- `analysis_result`: 数据分析报告
- `cleaning_result`: 数据清洗报告
- `cleaned_data_path`: 清洗后的数据路径

### Requirement: ModelTrainingAgent.generate_model_plan
方法 SHALL 接收完整上下文作为可选参数：
- `analysis_result`: 数据分析报告
- `cleaning_result`: 数据清洗报告
- `feature_result`: 特征工程报告
- `features_data_path`: 特征工程后的数据路径

### Requirement: API 阶段调用
API 的 `run_stage` 函数 SHALL 在调用各阶段 Agent 时：
- 读取所有前一阶段的结果文件
- 将结果作为上下文传递给当前阶段的 Agent
