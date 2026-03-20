# 交互式建模工作流验证清单

## 架构验证

* [ ] ConfigLoader 能正确加载所有配置

* [ ] SkillLoader 能正确加载所有 skills

* [ ] LLMLogger 正确记录每次调用

* [ ] AssetManager 正确管理资产文件

* [ ] ReAct Agent 支持可暂停循环

* [ ] WorkflowState 状态管理完整

* [ ] UserConfirmationPoint 确认点机制可用

* [ ] 前后端交互协议定义清晰

## 配置化验证

* [ ] 所有 Prompt 从配置文件读取

* [ ] 所有 LLM 配置从配置文件读取

* [ ] 代码中无硬编码 Prompt

* [ ] 代码中无硬编码模型配置

* [ ] 配置热更新正常工作

## Skills 集成验证

* [ ] data-analysis skill 正确加载

* [ ] afrexai-ml-engineering skill 正确加载

* [ ] ml-model-eval-benchmark skill 正确加载

* [ ] Skill 内容能正确注入到提示词中

* [ ] 确认点显示参考的 skill 来源

## 日志记录验证

* [ ] 每次 LLM 调用都被记录

* [ ] 日志包含请求时间、模型、输入、输出、Token

* [ ] 日志保存到 `logs/llm_calls/{session_id}/`

* [ ] 日志查询接口正常工作

## 资产管理验证

* [ ] 原始数据保存到 `assets/{session_id}/data/`

* [ ] 清洗后数据保存到 `assets/{session_id}/cleaned_data/`

* [ ] 特征数据保存到 `assets/{session_id}/features/`

* [ ] 生成代码保存到 `assets/{session_id}/code/`

* [ ] 模型文件保存到 `assets/{session_id}/models/`

* [ ] 报告保存到 `assets/{session_id}/reports/`

* [ ] 全流程脚本保存到 `assets/{session_id}/pipeline.py`

* [ ] 资产下载接口正常工作

## 数据清洗流程验证

* [ ] 从配置加载数据清洗 Prompt

* [ ] 正确加载 data-analysis skill 的 techniques.md

* [ ] 正确加载 data-analysis skill 的 pitfalls.md

* [ ] 数据清洗思路生成正确

* [ ] 清洗思路展示格式清晰

* [ ] 用户确认/修改/跳过功能正常

* [ ] 根据用户反馈重新生成思路正确

* [ ] 清洗代码生成正确

* [ ] 清洗代码执行成功

* [ ] 清洗结果摘要返回完整

* [ ] 记录 LLM 调用日志

## 特征工程流程验证

* [ ] 从配置加载特征工程 Prompt

* [ ] 正确加载 afrexai-ml-engineering skill 的 Phase 2

* [ ] 特征工程思路生成正确

* [ ] 特征思路展示格式清晰

* [ ] 用户确认/修改/跳过功能正常

* [ ] 根据用户反馈重新生成思路正确

* [ ] 特征代码生成正确

* [ ] 特征代码执行成功

* [ ] 特征结果摘要返回完整

* [ ] 记录 LLM 调用日志

## 建模方案流程验证

* [ ] 从配置加载建模方案 Prompt

* [ ] 正确加载 afrexai-ml-engineering skill 的 Phase 3

* [ ] 建模方案生成正确

* [ ] 建模方案展示格式清晰

* [ ] 用户确认/修改/跳过功能正常

* [ ] 根据用户反馈重新生成方案正确

* [ ] 建模代码生成正确

* [ ] 建模代码执行成功

* [ ] 模型训练结果返回完整

* [ ] 记录 LLM 调用日志

## 模型评估与结果输出验证

* [ ] 正确加载 ml-model-eval-benchmark skill

* [ ] 模型训练指标返回正确

* [ ] 模型文件保存成功

* [ ] 全流程建模脚本生成正确

* [ ] 脚本可独立运行

* [ ] 正确加载 data-analysis skill 的 chart-selection.md

* [ ] 可视化分析报告生成正确

* [ ] 报告包含所有必要图表

## API 验证

* [ ] 工作流启动接口正常

* [ ] 确认点等待接口正常

* [ ] 用户响应接口正常

* [ ] 状态查询接口正常

* [ ] Skill 内容查询接口正常

* [ ] 资产下载接口正常

* [ ] 日志查询接口正常

* [ ] SSE 流式响应正常

## 前端验证

* [ ] 方案展示组件正常

* [ ] 方案编辑组件正常

* [ ] 代码预览组件正常

* [ ] 结果可视化组件正常

* [ ] Skill 参考展示组件正常

* [ ] 资产下载组件正常

* [ ] 用户交互流程顺畅

## 代码清理验证

* [ ] 删除历史无关代码

* [ ] 删除未使用的文件

* [ ] 删除旧版 backend/ 代码

* [ ] 删除旧版 agents/ 代码

* [ ] 删除旧版 tools/ 代码

* [ ] 代码中无硬编码 Prompt

* [ ] 代码中无硬编码配置

* [ ] 代码风格统一

## 集成验证

* [ ] 完整工作流可正常运行

* [ ] 用户可在任意确认点修改方案

* [ ] 系统可从修改中正确恢复

* [ ] Skill 知识正确应用到各阶段

* [ ] 日志记录完整

* [ ] 资产保存完整

* [ ] 错误处理机制完善

* [ ] 状态持久化正确

