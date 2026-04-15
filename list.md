总流程

业务问题定义
数据接入与资产固化
数据契约检查
最小规则清洗
数据集切分
训练集内数据分析
训练集内特征工程设计
训练集内预处理拟合
不平衡处理与训练策略选择
模型训练与验证
最终测试评估
交付与监控
1. 问题定义
先明确这 6 件事，不然后面都容易跑偏：

任务类型
二分类、多分类、回归、排序、时间序列
预测目标
目标列是什么，预测时是否真实可得
预测时点
你在什么时刻做预测，哪些字段那时可用
评价指标
主指标一个，辅助指标若干
业务约束
可解释性、时延、成本、误报漏报代价
成功标准
至少要比什么基线好多少

2. 数据接入与资产固化
这一步只做一件事：把用户上传的数据固化成会话资产，后续全程只认资产路径。

保存原始文件
记录 schema 快照
记录上传时间、会话 ID、数据版本
后续所有阶段都引用资产目录内路径
目的：

可审计
可复现
不再混用外部路径和资产路径

3. 数据契约检查
先做“能不能建模”的检查，而不是先做“怎么建模”。

至少检查：

目标列是否存在
样本数是否够
主键/唯一标识是否存在
标签是否严重缺失
目标分布是否异常
时间字段和未来信息是否混入
训练时不可用字段是否存在
明显泄漏字段是否存在
输出应是：

可建模 / 不可建模
风险清单
需要业务确认的问题

4. 最小规则清洗
这一步只做“不会引入统计泄漏”的清洗。

适合在切分前做：

删除完全重复行
列名标准化
类型格式修正
单位统一
明显非法值修正
空字符串、特殊占位符归一
业务规则可判定的异常值处理
不建议在切分前做：

全局均值/中位数/众数填补
全局标准化
全局类别编码
依赖目标的清洗规则
任何需要“拟合参数”的处理
一句话：
切分前可以做 rule-based cleaning，不能做 fit-based preprocessing。

5. 数据集切分
这是整个科学性最关键的一步，必须前移。

推荐：
train / valid / test
没资源时至少 train / test

按任务类型选择切分策略：
分类
分层切分
回归
普通切分，或按目标分桶近似分层
时间序列
严格按时间切分
用户/设备/病例等分组问题
Group split，防止同实体泄漏

原则：
valid 用来选方案
test 只做最终评估
test 不参与任何拟合和调参
切分后应固化资产：

train_raw.csv
valid_raw.csv
test_raw.csv
6. 训练集内数据分析
EDA 不该再默认看整表，而应该以训练集为主。

分析内容：

缺失情况
类别分布
数值分布
目标分布
特征与目标的关系
训练集内异常值模式
类别不平衡
共线性和冗余
可以辅助查看 valid/test 的地方：

只做分布漂移对比
不据此重新设计针对 test 的规则
7. 训练集内特征工程设计
这里要把特征工程拆成两类。

可在切分前做的纯规则特征：

日期拆分
字段相加减乘除
规则标签
文本长度
布尔标记
必须切分后、只在训练集拟合的特征处理：

缺失值填补器
编码器
标准化器
降维器
监督式特征选择
目标编码
WOE/IV
任何基于统计量的转换
最合理的产物不是“全量 features_data.csv”优先，而是：

feature plan
fitted preprocessor
transformed train / valid / test
feature metadata
8. 训练集内预处理拟合
这一步是正式的 fit 阶段。

训练集上拟合：

imputer
encoder
scaler
selector
feature generator with learned params
然后：

transform(train)
transform(valid)
transform(test)
不能做：

fit(valid)
fit(test)
用 test 重新估计均值、众数、类别空间
最好的工程形式：

sklearn Pipeline / ColumnTransformer
或统一的可序列化 preprocessor 包
9. 不平衡处理与训练策略选择
这一步只对分类任务开启，且只在训练集上做。

先判断是否不平衡：

少数类占比
类别数
样本总量
每类最小样本数
误分类成本是否对称
常见策略：

class_weight
focal loss
RandomOverSampler
RandomUnderSampler
SMOTE
SMOTENC
BorderlineSMOTE
原则：

采样只作用于训练集
valid/test 保持真实分布
采样是训练策略，不是数据真相
采样结果不要覆盖原始 split 资产
记录内容：

是否检测到不平衡
采用了什么采样器
参数是什么
仅训练集使用
10. 模型训练与验证
这一步做的是实验管理，不只是“训一个模型”。

至少应该有：

基线模型
2 到 4 个候选模型
统一验证策略
主指标选型
可复现随机种子
参数搜索记录
早停与防过拟合策略
推荐输出：

best_model
all_models_comparison
selected_feature_names
target_transform
fitted preprocessor
split strategy
training summary
packaged model artifact
模型包建议统一结构：

model
preprocessor
selected_feature_names
target_transform
target_column
task_type
artifact_format
11. 最终测试评估
这是最后一次，也是唯一一次正式测 test。

测试阶段只做：

load packaged artifact
transform(test)
predict(test)
inverse transform if needed
metric calculation
diagnostics
不能做：

在 test 上重新 fit 任何处理器
在 test 上选模型
在 test 上调阈值
在 test 上决定是否采样
评估应包含：

主指标
辅助指标
confusion matrix / PR / ROC（分类）
residual / error analysis（回归）
slice analysis
calibration
drift / stability notes
failure cases
12. 交付与监控
一个完整 0 到 1 建模 Agent，不应止步于“指标出来了”。

应交付：

模型包
训练摘要
评估报告
推理脚本或服务接口
特征清单
输入契约
版本信息
依赖环境
重训条件
监控建议
上线后应监控：

输入缺失率
类别新值比例
特征分布漂移
预测分布漂移
标签延迟回流后的效果衰减
分类任务阈值稳定性
回归任务误差分桶表现
最科学合理的“阶段边界”
如果要把它压缩成你项目里的主流程，我建议是这个版本：

数据上传
固化原始资产
数据契约检查
标签、泄漏、任务可行性
最小规则清洗
不做全局拟合
数据切分
train/valid/test 原始切分资产
训练集数据分析
以 train 为主
特征工程设计
区分 rule-based 和 fit-based
预处理拟合
fit on train, transform all
类别不平衡策略判断
train only
模型训练与验证
valid 选方案
模型打包
model + preprocessor + metadata
最终测试评估
test only once
报告与交付


测试完整的建模流程，核心关注：
1. 要测试用户反馈修改意见后大模型修改方案是否符合用户意见
2. 在方案生产的环节是否正确调用了skill相关内容作为参考
3. 模型训练阶段代码是否正确保存
4. 是否有按照方案生成代码

测试阶段应改造：
1. 模型测试阶段应该复用训练数据的完整处理过程
2. 涉及用数据分布做处理的（如标准化、归一化、正则化、缺失值填充等）应该复用训练集处理过程中的模型或者尺度
3. 没有正确读取到skill的内容，而是只返回了skill的定义
4. 