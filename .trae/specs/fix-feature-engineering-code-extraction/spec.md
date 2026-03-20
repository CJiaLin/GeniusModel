# 特征工程代码提取修复 Spec

## Why
特征工程阶段的代码生成后保存的文件包含完整的 JSON 响应（`{"thinking": "...", "code": "..."}`），而不是纯 Python 代码。数据清洗阶段的代码提取是成功的，需要参考数据清洗的实现方式修复特征工程和模型训练阶段。

## What Changes
- 修复 `CodeGenerator._sanitize_code` 方法，确保正确提取 JSON 中的 `code` 字段
- 确保特征工程 Agent 和模型训练 Agent 使用与数据清洗 Agent 相同的代码提取逻辑
- 统一所有代码生成环节的处理方式

## Impact
- Affected code: 
  - `automl_react/utils/code_generator.py`
  - `automl_react/agents/feature_engineering_agent.py`
  - `automl_react/agents/model_training_agent.py`

## ADDED Requirements

### Requirement: 代码提取一致性
所有 Agent（数据清洗、特征工程、模型训练）的代码生成方法 SHALL 使用相同的代码提取逻辑，确保保存的代码文件只包含纯 Python 代码。

#### Scenario: 特征工程代码生成
- **WHEN** 特征工程 Agent 生成代码
- **THEN** 保存的 `feature_engineering.py` 文件 SHALL 只包含纯 Python 代码，不包含 JSON 格式或 Markdown 标记

#### Scenario: 模型训练代码生成
- **WHEN** 模型训练 Agent 生成代码
- **THEN** 保存的 `model_training.py` 文件 SHALL 只包含纯 Python 代码，不包含 JSON 格式或 Markdown 标记

## MODIFIED Requirements

### Requirement: CodeGenerator._sanitize_code 方法
方法 SHALL 正确处理以下情况：
1. LLM 返回完整的 JSON 字符串（包含 `thinking` 和 `code` 字段）
2. JSON 中的 `code` 字段值包含转义字符（如 `\n`）
3. 代码被 Markdown 代码块（` ```python `）包裹

处理逻辑：
1. 如果内容以 `{` 开头且包含 `"code"` 或 `'code'`，尝试解析 JSON
2. 提取 `code` 字段的值
3. JSON 解析后，`\n` 等转义字符自动转换为实际换行符
4. 移除 Markdown 代码块围栏行
