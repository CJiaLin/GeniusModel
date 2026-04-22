# Feature Engineering Patterns

特征工程常用模式与最佳实践技能包。

## 适用场景

- 表格数据的特征工程阶段
- 数值/类别/时间/文本特征处理
- 特征交互与高阶特征构造

## 包含章节

| 章节 | 内容 |
|------|------|
| numeric-transforms | 数值特征变换（对数、分箱、标准化、多项式等） |
| categorical-encoding | 类别特征编码（OneHot、Target Encoding、频率编码等） |
| datetime-features | 时间特征提取（周期、滞后、滑动窗口等） |
| interaction-features | 特征交互与组合（乘法交互、分组聚合、比率特征等） |

## 使用原则

- 所有特征工程操作必须**仅基于训练集**拟合，然后 transform 到验证/测试集
- 避免数据泄露：不使用未来信息、不使用目标变量做特征
- 优先选择可解释、业务有意义的特征
