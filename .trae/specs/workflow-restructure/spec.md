# 工作流程重构 Spec

## Why
当前的工作流程将数据分析放在第一步，但实际上应该先进行数据质量分析，确定数据质量问题后才能制定合理的清洗方案。清洗完成后的数据才是干净的数据，此时进行探索性数据分析才能得到准确的统计特征和相关性分析结果。当前流程不够科学合理，可能导致分析结果不准确。

## What Changes
- **BREAKING**: 将数据分析阶段拆分为两个阶段：数据清洗（包含数据质量分析）和数据探索性分析
- **BREAKING**: 调整工作流程顺序：数据清洗（含质量分析）→ 数据探索性分析 → 特征工程 → 模型训练
- 数据清洗阶段首先进行数据质量分析，识别数据质量问题（缺失值、异常值、重复值等），生成清洗报告，用户确认后执行清洗
- 数据探索性分析阶段移到数据清洗之后，分析清洗后数据的分布、相关性等统计特征
- 更新工作流状态枚举和阶段转换逻辑
- 更新 API 端点以支持新的工作流程

## Impact
- Affected specs: 
  - `workflow-context-passing` - 需要更新阶段间数据传递逻辑
- Affected code:
  - `automl_react/core/workflow_state.py` - 更新工作流阶段枚举
  - `automl_react/agents/data_cleaning_agent.py` - 重构为包含数据质量分析
  - `automl_react/agents/data_analysis_agent.py` - 重命名为数据探索性分析 Agent
  - `automl_react/api/main.py` - 更新 API 端点和阶段调用逻辑
  - `automl_react/config/workflow_config.yaml` - 更新工作流配置

## ADDED Requirements

### Requirement: 数据清洗阶段包含数据质量分析
系统 SHALL 在数据清洗阶段首先进行数据质量分析，然后生成清洗报告供用户确认。

#### Scenario: 数据清洗阶段完整流程
- **WHEN** 用户启动数据清洗阶段
- **THEN** 系统 SHALL 首先进行数据质量分析：
  - 分析数据的完整性（缺失值统计和比例）
  - 分析数据的一致性（异常值、离群点检测）
  - 分析数据的唯一性（重复值检测）
  - 分析数据类型的正确性
- **AND** 系统 SHALL 基于质量分析生成数据清洗报告：
  - 各列的缺失值统计和建议处理方式
  - 异常值检测结果和建议处理方式
  - 重复值检测结果和建议
  - 数据类型问题及建议
  - 完整的清洗策略
- **AND** 系统 SHALL 请求用户确认清洗报告
- **WHEN** 用户确认清洗报告
- **THEN** 系统 SHALL 生成清洗代码
- **AND** 系统 SHALL 执行清洗代码
- **AND** 系统 SHALL 保存清洗后的数据

### Requirement: 数据探索性分析阶段
系统 SHALL 提供独立的数据探索性分析阶段，基于清洗后的数据进行分析。

#### Scenario: 探索性分析基于清洗后数据
- **WHEN** 数据清洗阶段完成
- **AND** 用户启动数据探索性分析阶段
- **THEN** 系统 SHALL 使用清洗后的数据
- **AND** 系统 SHALL 分析数据分布特征（均值、方差、偏度、峰度等）
- **AND** 系统 SHALL 分析特征相关性（相关系数矩阵）
- **AND** 系统 SHALL 分析目标变量分布
- **AND** 系统 SHALL 生成可视化建议
- **AND** 系统 SHALL 提供特征重要性初步评估
- **AND** 系统 SHALL 为特征工程提供建议

### Requirement: 新的工作流程顺序
系统 SHALL 按照以下顺序执行工作流：

#### Scenario: 标准工作流程
- **GIVEN** 用户上传数据并输入建模背景
- **WHEN** 用户启动工作流
- **THEN** 系统 SHALL 按以下顺序执行：
  1. 数据清洗阶段：
     - 数据质量分析 → 生成清洗报告
     - 用户确认清洗报告
     - 生成并执行清洗代码
  2. 数据探索性分析阶段：
     - 分析清洗后数据分布
     - 分析特征相关性
     - 生成探索性分析报告
  3. 特征工程阶段：
     - 生成特征方案
     - 用户确认特征方案
     - 生成并执行特征工程代码
  4. 模型训练阶段：
     - 生成建模方案
     - 用户确认建模方案
     - 生成并执行模型训练代码

## MODIFIED Requirements

### Requirement: WorkflowStage 枚举
枚举 SHALL 更新为以下阶段：
```python
class WorkflowStage(Enum):
    DATA_UPLOAD = "data_upload"
    DATA_CLEANING = "data_cleaning"  # 包含数据质量分析
    DATA_EXPLORATION = "data_exploration"  # 新增（原 data_analysis，移到清洗后）
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    COMPLETED = "completed"
```

### Requirement: DataCleaningAgent 重构
DataCleaningAgent SHALL 包含以下方法：
- `analyze_data_quality()`: 分析数据质量，识别问题
- `generate_cleaning_plan()`: 基于质量分析生成清洗报告（原方法，增强）
- `generate_cleaning_code()`: 生成清洗代码
- `execute_cleaning()`: 执行清洗代码

### Requirement: DataAnalysisAgent 重命名和重构
DataAnalysisAgent SHALL 重命名为 DataExplorationAgent，专注于：
- 分析清洗后数据的统计特征
- 分析特征相关性
- 分析目标变量分布
- 为特征工程提供建议

### Requirement: 阶段间数据传递
系统 SHALL 更新阶段间数据传递逻辑：
- 数据清洗阶段：接收原始数据和用户建模背景
- 数据探索性分析阶段：接收清洗后的数据、数据清洗报告、用户建模背景
- 特征工程阶段：接收数据探索性分析报告、数据清洗报告、清洗后的数据、用户建模背景
- 模型训练阶段：接收所有前一阶段的结果

## REMOVED Requirements

### Requirement: 原数据分析阶段位置
**Reason**: 原数据分析阶段在数据清洗之前，分析的是脏数据，结果不准确。
**Migration**: 将探索性分析移到数据清洗之后，数据质量分析合并到数据清洗阶段。
