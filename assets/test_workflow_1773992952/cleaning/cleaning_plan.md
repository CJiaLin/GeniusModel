```markdown
# 数据清洗方案报告

## 1. 数据质量问题分析

### 1.1 数据概况
| 指标 | 数值 |
|------|------|
| 总行数 | 1,460 |
| 总列数 | 81 |
| 数值型列 | 35 |
| 分类型列 | 46 |
| 目标变量 | SalePrice |

### 1.2 缺失值问题（按严重程度分类）

| 严重程度 | 列名 | 缺失数 | 缺失比例 | 说明 |
|----------|------|--------|----------|------|
| 🔴 极高 | PoolQC | 1,453 | 99.52% | 大多数房屋无泳池 |
| 🔴 极高 | MiscFeature | 1,406 | 96.30% | 大多数房屋无特殊设施 |
| 🔴 极高 | Alley | 1,369 | 93.77% | 大多数房屋无巷道 |
| 🔴 极高 | Fence | 1,179 | 80.75% | 大多数房屋无围栏 |
| 🟡 中等 | FireplaceQu | 690 | 47.26% | 无壁炉的房屋 |
| 🟡 中等 | LotFrontage | 259 | 17.74% | 临街距离缺失 |
| 🟢 较低 | Garage相关列 | 81 | 5.55% | 无车库的房屋 |
| 🟢 较低 | Bsmt相关列 | 37-38 | 2.53%-2.60% | 无地下室的房屋 |
| 🟢 较低 | MasVnr相关列 | 8 | 0.55% | 砌体贴面缺失 |
| 🟢 较低 | Electrical | 1 | 0.07% | 电气系统缺失 |

### 1.3 异常值问题

| 列名 | 问题描述 | 影响 |
|------|----------|------|
| **LotArea** | 最大值215,245，远超75%分位数11,601 | 可能存在异常大地块 |
| **GrLivArea** | 最大值5,642，75%分位数1,776 | 可能存在豪宅或错误数据 |
| **1stFlrSF** | 存在极端大值 | 与GrLivArea相关 |
| **GarageYrBlt** | 存在2207年的异常值 | 明显录入错误 |
| **SalePrice** | 右偏分布 | 可能需要对数变换 |

### 1.4 数据类型问题

| 列名 | 当前类型 | 建议类型 | 原因 |
|------|----------|----------|------|
| MSSubClass | 数值型 | 分类型 | 是建筑类型编码，非连续数值 |
| MoSold | 数值型 | 分类型 | 月份应作为类别 |
| GarageYrBlt | 数值型 | 需清洗 | 存在异常年份 |

### 1.5 重复值检查

- ✅ 完全重复行：**0 行**
- ✅ ID列重复：**0 行**

---

## 2. 清洗步骤

### 步骤 1: 缺失值处理

#### 1.1 缺失代表"无"的列（NA = None）
```python
# 以下列缺失表示该房屋没有此设施，填充"None"
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 
             'FireplaceQu', 'GarageType', 'GarageFinish',
             'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']

for col in none_cols:
    df[col].fillna('None', inplace=True)
```

#### 1.2 数值型缺失值处理
```python
# LotFrontage - 按Neighborhood分组填充中位数
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# GarageYrBlt - 无车库则填充0
df['GarageYrBlt'].fillna(0, inplace=True)

# MasVnrArea - 无砌体贴面则填充0
df['MasVnrArea'].fillna(0, inplace=True)

# Electrical - 填充众数（最常用电气系统）
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
```

### 步骤 2: 异常值处理

```python
# 2.1 修正GarageYrBlt异常值
df.loc[df['GarageYrBlt'] > 2010, 'GarageYrBlt'] = df.loc[df['GarageYrBlt'] > 2010, 'YearBuilt']

# 2.2 处理面积异常值（使用IQR方法或设置合理上限）
# 标记潜在异常值但不删除，添加异常值标记列
Q3 = df['LotArea'].quantile(0.99)
df['LotArea_Outlier'] = (df['LotArea'] > Q3).astype(int)

Q3 = df['GrLivArea'].quantile(0.99)
df['GrLivArea_Outlier'] = (df['GrLivArea'] > Q3).astype(int)

# 2.3 对目标变量进行对数变换（处理右偏）
df['SalePrice_Log'] = np.log1p(df['SalePrice'])
```

### 步骤 3: 数据类型转换

```python
# 转换为分类类型
df['MSSubClass'] = df['MSSubClass'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)

# 确保ID为整数
df['Id'] = df['Id'].astype(int)
```

### 步骤 4: 特征工程（可选增强）

```python
# 4.1 创建总面积特征
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# 4.2 创建房屋年龄特征
df['HouseAge'] = df['YrSold'].astype(int) - df['YearBuilt']

# 4.3 创建是否翻新特征
df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)

# 4.4 创建总浴室数
df['TotalBathrooms'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])

# 4.5 车库是否建立特征
df['HasGarage'] = (df['GarageYrBlt'] > 0).astype(int)
```

### 步骤 5: 最终验证

```python
# 5.1 检查是否还有缺失值
assert df.isnull().sum().sum() == 0, "仍存在缺失值！"

# 5.2 检查数据类型
print(df.dtypes)

# 5.3 保存清洗后数据
df.to_csv('/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv', index=False)
```

---

## 3. 预期效果

### 3.1 数据质量提升

| 指标 | 清洗前 | 清洗后 | 改善 |
|------|--------|--------|------|
| 缺失值比例 | 5.76% | 0% | ✅ 完全消除 |
| 异常值标记 | 0列 | 2列 | ✅ 可控处理 |
| 数据类型错误 | 3列 | 0列 | ✅ 完全修正 |
| 重复行 | 0行 | 0行 | ✅ 保持干净 |

### 3.2 特征增强

| 新增特征 | 说明 | 预期作用 |
|----------|------|----------|
| TotalSF | 总面积 | 综合面积指标 |
| HouseAge | 房龄 | 时间衰减效应 |
| IsRemodeled | 是否翻新 | 翻新价值 |
| TotalBathrooms | 总浴室数 | 居住舒适度 |
| HasGarage | 有无车库 | 便利性指标 |
| SalePrice_Log | 对数价格 | 正态化目标变量 |

### 3.3 对建模的影响

1. **缺失值处理**：避免模型因缺失值产生的偏差，保持样本完整性
2. **异常值标记**：保留极端值信息同时控制风险，避免直接删除造成信息损失
3. **特征工程**：提供更有意义的组合特征，提升模型预测能力
4. **类型修正**：确保分类变量正确处理，避免数值编码误解

### 3.4 输出文件

- **主文件**: `train_cleaned.csv` - 清洗后的完整数据集
- **维度**: 1,460 行 × 90+ 列（含新增特征）
- **适用场景**: 可直接用于机器学习模型训练

---

## 4. 注意事项

| 序号 | 注意事项 | 建议 |
|------|----------|------|
| 1 | 训练集和测试集需统一处理 | 确保测试集使用相同的填充值和转换逻辑 |
| 2 | 极端值保留 | 房地产数据中的豪宅可能是真实存在，建议标记而非删除 |
| 3 | 目标变量变换 | 如需预测原始价格，记得对预测结果进行指数还原 |
| 4 | 特征选择 | 新增特征可能引入多重共线性，建议后续进行特征筛选 |
```