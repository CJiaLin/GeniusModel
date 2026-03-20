# 交互式建模工作流任务清单

## 阶段一：核心架构扩展（含 Skills、配置、日志支持）
- [x] Task 1.1: 实现 ConfigLoader 配置加载器
  - [x] 实现 prompts.yaml 加载
  - [x] 实现 llm_config.yaml 加载
  - [x] 实现 workflow_config.yaml 加载
  - [x] 实现配置热更新
- [x] Task 1.2: 实现 SkillLoader Skills 加载器
  - [x] 实现 skills 目录扫描
  - [x] 实现 skill 元数据解析（_meta.json, SKILL.md）
  - [x] 实现 skill 内容检索接口
  - [x] 实现 skill 参考文件加载（.md 文件）
- [x] Task 1.3: 实现 LLMLogger 大模型调用日志记录器
  - [x] 定义日志数据结构（请求时间、模型、输入、输出、Token）
  - [x] 实现日志写入（JSONL 格式）
  - [x] 实现日志查询接口
  - [x] 配置日志保存路径 `logs/llm_calls/{session_id}/`
- [x] Task 1.4: 实现 AssetManager 资产文件管理器
  - [x] 定义资产目录结构
  - [x] 实现资产保存接口
  - [x] 实现资产查询接口
  - [x] 实现资产下载接口
  - [x] 配置资产保存路径 `assets/{session_id}/`
- [x] Task 1.5: 扩展 ReAct Agent 支持用户确认点
  - [x] 实现可暂停的 ReAct 循环
  - [x] 添加确认点检查机制
  - [x] 实现状态保存和恢复
- [x] Task 1.6: 实现 WorkflowState 状态管理
  - [x] 定义工作流状态枚举
  - [x] 实现状态持久化
  - [x] 实现状态转换验证
- [x] Task 1.7: 实现 UserConfirmationPoint 确认点管理
  - [x] 定义确认点数据结构（含 skill 引用信息）
  - [x] 实现确认点队列
  - [x] 实现用户响应处理

## 阶段二：数据清洗交互流程（集成 data-analysis skill）
- [x] Task 2.1: 实现数据清洗思路生成
  - [x] 从配置加载数据清洗 Prompt
  - [x] 加载 data-analysis skill 的 techniques.md
  - [x] 加载 data-analysis skill 的 pitfalls.md
  - [x] 分析数据质量问题
  - [x] 基于 skill 知识生成清洗策略建议
  - [x] 记录 LLM 调用日志
  - [x] 格式化清洗思路展示（含 skill 引用）
- [x] Task 2.2: 实现数据清洗用户确认流程
  - [x] 展示清洗思路给用户（显示参考的 skill）
  - [x] 接收用户确认/修改/跳过
  - [x] 根据反馈重新生成思路（可加载其他 skill 参考）
- [x] Task 2.3: 实现数据清洗代码生成与执行
  - [x] 从配置加载代码生成 Prompt
  - [x] 根据确认思路生成代码
  - [x] 保存代码到 `assets/{session_id}/code/cleaning.py`
  - [x] 执行清洗代码
  - [x] 保存清洗后数据到 `assets/{session_id}/cleaned_data/`
  - [x] 返回清洗结果摘要
  - [x] 记录 LLM 调用日志

## 阶段三：特征工程交互流程（集成 afrexai-ml-engineering skill）
- [x] Task 3.1: 实现特征工程思路生成
  - [x] 从配置加载特征工程 Prompt
  - [x] 加载 afrexai-ml-engineering skill 的 Phase 2 内容
  - [x] 分析清洗后数据特征
  - [x] 基于 skill 最佳实践生成特征工程建议
  - [x] 记录 LLM 调用日志
  - [x] 格式化特征思路展示（含 skill 引用）
- [x] Task 3.2: 实现特征工程用户确认流程
  - [x] 展示特征思路给用户（显示参考的 skill）
  - [x] 接收用户确认/修改/跳过
  - [x] 根据反馈重新生成思路
- [x] Task 3.3: 实现特征工程代码生成与执行
  - [x] 从配置加载代码生成 Prompt
  - [x] 根据确认思路生成代码
  - [x] 保存代码到 `assets/{session_id}/code/feature_engineering.py`
  - [x] 执行特征代码
  - [x] 保存特征数据到 `assets/{session_id}/features/`
  - [x] 返回特征结果摘要
  - [x] 记录 LLM 调用日志

## 阶段四：建模方案交互流程（集成 afrexai-ml-engineering skill）
- [x] Task 4.1: 实现建模方案生成
  - [x] 从配置加载建模方案 Prompt
  - [x] 加载 afrexai-ml-engineering skill 的 Phase 3 内容
  - [x] 分析特征数据
  - [x] 基于 skill 推荐模型和参数
  - [x] 记录 LLM 调用日志
  - [x] 格式化建模方案展示（含 skill 引用）
- [x] Task 4.2: 实现建模方案用户确认流程
  - [x] 展示建模方案给用户（显示参考的 skill）
  - [x] 接收用户确认/修改/跳过
  - [x] 根据反馈重新生成方案
- [x] Task 4.3: 实现建模代码生成与执行
  - [x] 从配置加载代码生成 Prompt
  - [x] 根据确认方案生成代码
  - [x] 保存代码到 `assets/{session_id}/code/model_training.py`
  - [x] 执行建模代码
  - [x] 保存模型文件到 `assets/{session_id}/models/`
  - [x] 返回模型训练结果
  - [x] 记录 LLM 调用日志

## 阶段五：模型评估与结果输出（集成 ml-model-eval-benchmark skill）
- [x] Task 5.1: 实现模型评估
  - [x] 加载 ml-model-eval-benchmark skill 的 benchmarking-guide.md
  - [x] 基于 skill 方法计算评估指标
  - [x] 返回训练指标
  - [x] 保存评估结果到 `assets/{session_id}/reports/evaluation.json`
- [x] Task 5.2: 实现全流程脚本生成
  - [x] 收集各阶段代码
  - [x] 组装完整脚本
  - [x] 保存到 `assets/{session_id}/pipeline.py`
  - [x] 提供下载接口
- [x] Task 5.3: 实现可视化分析报告（集成 data-analysis skill）
  - [x] 加载 data-analysis skill 的 chart-selection.md
  - [x] 基于 skill 指南生成数据分布图表
  - [x] 基于 skill 指南生成特征重要性图表
  - [x] 基于 skill 指南生成模型评估图表
  - [x] 组装 HTML/PDF 报告
  - [x] 保存到 `assets/{session_id}/reports/`
  - [x] 提供下载接口

## 阶段六：前后端集成
- [ ] Task 6.1: 实现后端 API
  - [ ] 实现工作流启动接口
  - [ ] 实现确认点等待接口
  - [ ] 实现用户响应接口
  - [ ] 实现状态查询接口
  - [ ] 实现 skill 内容查询接口
  - [ ] 实现资产下载接口
  - [ ] 实现日志查询接口
- [ ] Task 6.2: 实现前端交互界面
  - [ ] 方案展示组件（显示 skill 引用）
  - [ ] 方案编辑组件
  - [ ] 代码预览组件
  - [ ] 结果可视化组件
  - [ ] Skill 参考展示组件
  - [ ] 资产下载组件

## 阶段七：配置迁移
- [ ] Task 7.1: 迁移 Prompt 配置
  - [ ] 将所有硬编码 Prompt 迁移到 `config/prompts.yaml`
  - [ ] 按工作流阶段组织 Prompt
  - [ ] 添加 Prompt 版本管理
- [ ] Task 7.2: 迁移 LLM 配置
  - [ ] 将所有硬编码模型配置迁移到 `config/llm_config.yaml`
  - [ ] 支持多模型配置
  - [ ] 添加模型参数配置

## 阶段八：代码清理
- [ ] Task 8.1: 删除历史无关代码
  - [ ] 识别并删除未使用的文件
  - [ ] 删除旧版 backend/ 代码
  - [ ] 删除旧版 agents/ 代码
  - [ ] 删除旧版 tools/ 代码
- [ ] Task 8.2: 清理硬编码配置
  - [ ] 检查并删除代码中的硬编码 Prompt
  - [ ] 检查并删除代码中的硬编码模型配置
  - [ ] 检查并删除代码中的硬编码路径
- [ ] Task 8.3: 统一代码风格
  - [ ] 统一导入格式
  - [ ] 统一命名规范
  - [ ] 添加必要的类型注解

## 阶段九：测试与优化
- [ ] Task 9.1: 单元测试
  - [ ] 测试 ConfigLoader 配置加载
  - [ ] 测试 SkillLoader 加载逻辑
  - [ ] 测试 LLMLogger 日志记录
  - [ ] 测试 AssetManager 资产管理
  - [ ] 测试确认点逻辑
  - [ ] 测试状态管理
  - [ ] 测试代码生成
- [ ] Task 9.2: 集成测试
  - [ ] 测试完整工作流
  - [ ] 测试用户交互场景
  - [ ] 测试 skill 调用
  - [ ] 测试资产保存和下载
  - [ ] 测试日志记录
  - [ ] 测试错误恢复

# 任务依赖关系
- Task 1.1 → Task 1.2 → Task 1.3 → Task 1.4 → Task 1.5 → Task 1.6 → Task 1.7
- Task 1.x → Task 2.1, Task 3.1, Task 4.1
- Task 2.1 → Task 2.2 → Task 2.3
- Task 3.1 → Task 3.2 → Task 3.3
- Task 4.1 → Task 4.2 → Task 4.3
- Task 2.3, Task 3.3, Task 4.3 → Task 5.1 → Task 5.2 → Task 5.3
- Task 5.x → Task 6.1 → Task 6.2
- Task 6.x → Task 7.1 → Task 7.2 → Task 8.1 → Task 8.2 → Task 8.3 → Task 9.1 → Task 9.2

# 可并行任务
- Task 2.1, Task 3.1, Task 4.1 可并行
- Task 5.2, Task 5.3 可并行
- Task 6.1 和 Task 6.2 可部分并行
- Task 7.1 和 Task 7.2 可并行
- Task 8.1, Task 8.2, Task 8.3 可部分并行
