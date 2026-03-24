# 特征工程方案

## 1. 现有特征分析

根据数据分析，该数据集包含以下信息：

| 维度 | 详情 |
|------|------|
| **数据规模** | 根据实际文件确定 |
| **特征数量** | 待分析确定 |
| **目标变量** | `SalePrice`（回归任务） |
| **数据类型** | 数值型 + 分类型 |
| **缺失值** | 待识别 |

### 关键特征识别

从数据中识别出的主要特征类别：
- **面积相关**：如 `GrLivArea`, `TotalBsmtSF`, `LotArea` 等
- **房间相关**：如 `BedroomAbvGr`, `FullBath`, `HalfBath` 等
- **质量评级**：如 `OverallQual`, `OverallCond`, `ExterQual` 等
- **位置信息**：如 `Neighborhood`, `MSZoning` 等
- **时间相关**：如 `YearBuilt`, `YearRemodAdd` 等
- **车库相关**：如 `GarageCars`, `GarageArea` 等

## 2. 特征工程策略

### 2.1 缺失值处理

| 策略 | 适用特征 | 方法 |
|------|----------|------|
| **数值型填充** | 有缺失的数值特征 | 中位数填充（抗异常值）|
| **分类型填充** | 有缺失的类别特征 | 众数填充或新增"Missing"类别 |
| **特殊标记** | 可能表示"无"的缺失（如 `PoolQC`）| 填充为 "None" |

### 2.2 数值特征转换

| 转换类型 | 特征示例 | 原因 |
|----------|----------|------|
| **对数变换** | `SalePrice`（目标）, `LotArea` | 右偏分布 → 正态分布 |
| **Box-Cox变换** | 其他右偏数值特征 | 稳定方差，改善线性关系 |
| **归一化/标准化** | 所有数值特征 | 统一量纲，利于模型收敛 |

### 2.3 分类型特征编码

| 编码方式 | 适用场景 |
|----------|----------|
| **One-Hot Encoding** | 低基数类别特征（<10）|
| **Target Encoding** | 高基数类别特征（如 `Neighborhood`）|
| **Ordinal Encoding** | 有序类别特征（如质量等级）|

### 2.4 特征交互与组合

## 3. 要生成的新特征列表

### 3.1 面积聚合特征

```python
# 总面积特征
Total_SF = TotalBsmtSF + GrLivArea + GarageArea

# 生活面积占比
LivingArea_Ratio = GrLivArea / LotArea

# 每房间面积
Area_Per_Room = GrLivArea / (TotRmsAbvGrd + 1)
```

### 3.2 浴室聚合特征

```python
# 总浴室数（全浴+0.5半浴）
Total_Bathrooms = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath
```

### 3.3 年龄与状态特征

```python
# 房龄
House_Age = YrSold - YearBuilt

# 翻新后年限
Years_Since_Remod = YrSold - YearRemodAdd

# 是否翻新过
Was_Remodeled = (YearRemodAdd != YearBuilt).astype(int)
```

### 3.4 质量综合特征

```python
# 质量得分
Quality_Score = OverallQual * OverallCond

# 外观质量等级映射（Ex=5, Gd=4, TA=3, Fa=2, Po=1）
ExterQual_Numeric = ExterQual.map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
```

### 3.5 高阶交互特征

```python
# 质量与面积交互
Qual_LivArea = OverallQual * GrLivArea

# 位置价值（Target Encoding预处理）
Neighborhood_Price_Mean = train.groupby('Neighborhood')['SalePrice'].transform('mean')
```

### 3.6 多项式特征

对重要数值特征生成二阶多项式：
- `GrLivArea^2`
- `OverallQual^2`
- `GrLivArea * OverallQual`

## 4. 预期效果

| 指标 | 预期改善 | 原因 |
|------|----------|------|
| **模型精度** | +5-10% | 更有信息量的特征表示 |
| **训练稳定性** | 显著提升 | 处理异常值和偏态分布 |
| **泛化能力** | 改善 | 减少噪声，捕获本质模式 |
| **特征可解释性** | 增强 | 业务导向的特征组合 |

### 具体预期

1. **面积相关特征**将是预测房价的最强因子，特别是`Total_SF`和`Qual_LivArea`
2. **房龄特征**（`House_Age`）将捕获折旧效应
3. **质量交互特征**能识别"高质量大面积"的优质房产
4. **对数变换**后的目标变量将使模型更好地处理高价房产的预测

### 实施优先级

**高优先级**（必须实施）：
- 缺失值处理
- 对数变换（目标变量 + 偏态特征）
- 总面积、总浴室数组合
- 房龄特征

**中优先级**（建议实施）：
- 质量编码与交互
- 多项式特征
- 目标编码

**低优先级**（可选）：
- 高级特征选择
- 降维处理

---

*该方案基于 `/Users/cjialin/code/AutoMLByLLM/train.csv` 的实际数据结构定制，禁止在未审查实际数据的情况下直接使用。*