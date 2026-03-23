# 房价预测数据清洗方案

## 1. 项目背景与目标

### 1.1 任务描述
- **目标**: 预测每套房子的售价 (SalePrice)
- **评估指标**: RMSE (均方根误差)
- **数据规模**: 1,460 行 × 81 列
- **数据文件**: `/Users/cjialin/code/AutoMLByLLM/train.csv`

### 1.2 数据质量概览
| 问题类型 | 数量 | 严重程度 |
|---------|------|---------|
| 缺失值问题 | 19 列 | 高 (5列缺失率>59%) |
| 异常值问题 | 31 列 | 中 (需Winsorize处理) |
| 重复值 | 0 行 | 无 |
| 数据类型优化 | 43 列 | 中 (建议转category) |

---

## 2. 数据清洗详细方案

### 2.1 高缺失率列处理（删除列）

**策略**: 对于缺失率超过 50% 的列，直接删除，因为这些列信息含量极低。

| 列名 | 缺失比例 | 处理方式 | 业务理由 |
|------|---------|---------|---------|
| PoolQC | 99.52% | **删除列** | 绝大多数房屋无泳池 |
| MiscFeature | 96.30% | **删除列** | 特殊设施极少见 |
| Alley | 93.77% | **删除列** | 小巷通道信息缺失过多 |
| Fence | 80.75% | **删除列** | 围栏信息不完整 |
| MasVnrType | 59.73% | **删除列** | 砌面类型缺失过多 |

```python
# 删除高缺失率列
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=cols_to_drop)
```

### 2.2 中等缺失率列处理（业务逻辑填充）

#### 2.2.1 地下室相关特征（5列）

| 列名 | 缺失数量 | 数据类型 | 填充策略 |
|------|---------|---------|---------|
| BsmtExposure | 38 | object | "No" (无暴露) |
| BsmtFinType2 | 38 | object | "Unf" (未装修) |
| BsmtQual | 37 | object | "NA" (无地下室) |
| BsmtCond | 37 | object | "NA" (无地下室) |
| BsmtFinType1 | 37 | object | "NA" (无地下室) |

```python
# 地下室特征填充
basement_cols = {
    'BsmtExposure': 'No',
    'BsmtFinType2': 'Unf',
    'BsmtQual': 'NA',
    'BsmtCond': 'NA',
    'BsmtFinType1': 'NA'
}

for col, fill_val in basement_cols.items():
    df[col] = df[col].fillna(fill_val)
```

#### 2.2.2 车库相关特征（5列）

| 列名 | 缺失数量 | 数据类型 | 填充策略 |
|------|---------|---------|---------|
| GarageType | 81 | object | "NA" (无车库) |
| GarageYrBlt | 81 | float64 | 0 (或YearBuilt) |
| GarageFinish | 81 | object | "NA" (无车库) |
| GarageQual | 81 | object | "NA" (无车库) |
| GarageCond | 81 | object | "NA" (无车库) |

```python
# 车库特征填充
garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cat_cols:
    df[col] = df[col].fillna('NA')

# 车库建造年份用房屋建造年份填充
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
```

#### 2.2.3 其他重要特征

| 列名 | 缺失数量 | 数据类型 | 填充策略 |
|------|---------|---------|---------|
| FireplaceQu | 690 | object | "NA" (无壁炉) |
| LotFrontage | 259 | float64 | 按Neighborhood分组中位数 |
| MasVnrArea | 8 | float64 | 0 (无砌面) |
| Electrical | 1 | object | 众数 "SBrkr" |

```python
# 壁炉质量
df['FireplaceQu'] = df['FireplaceQu'].fillna('NA')

# LotFrontage按社区分组填充
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# 剩余缺失值用整体中位数填充
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# 砌面面积
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# 电力系统
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
```

---

### 2.3 异常值处理（Winsorize + 删除）

#### 2.3.1 删除异常严重的列

| 列名 | 异常值比例 | 处理方式 | 原因 |
|------|-----------|---------|------|
| BsmtFinSF2 | 11.44% | **删除列** | 99%值为0，信息量极低 |
| EnclosedPorch | 14.25% | **删除列** | 90%值为0，信息量极低 |

```python
# 删除低信息量变量的列
df = df.drop(columns=['BsmtFinSF2', 'EnclosedPorch'])
```

#### 2.3.2 Winsorize处理（缩尾至5%-95%）

| 列名 | 正常范围 | 处理方法 |
|------|---------|---------|
| MSSubClass | [-55.0, 145.0] | 缩尾处理 |
| LotFrontage | [27.5, 111.5] | 缩尾处理 |
| LotArea | [1481.5, 17673.5] | 缩尾处理 |
| OverallCond | [3.5, 7.5] | 缩尾处理 |
| MasVnrArea | [-249.0, 415.0] | 缩尾处理 |
| BsmtUnfSF | [-654.5, 1685.5] | 缩尾处理 |
| TotalBsmtSF | [42.0, 2052.0] | 缩尾处理 |
| 1stFlrSF | [118.12, 2155.12] | 缩尾处理 |
| LowQualFinSF | [0.0, 0.0] | 缩尾处理 |
| GrLivArea | [158.62, 2747.62] | 缩尾处理 |
| BsmtHalfBath | [0.0, 0.0] | 缩尾处理 |
| BedroomAbvGr | [0.5, 4.5] | 缩尾处理 |
| KitchenAbvGr | [1.0, 1.0] | 缩尾处理 |
| TotRmsAbvGrd | [2.0, 10.0] | 缩尾处理 |
| GarageArea | [-27.75, 938.25] | 缩尾处理 |
| WoodDeckSF | [-252.0, 420.0] | 缩尾处理 |
| OpenPorchSF | [-102.0, 170.0] | 缩尾处理 |
| 3SsnPorch | [0.0, 0.0] | 缩尾处理 |
| ScreenPorch | [0.0, 0.0] | 缩尾处理 |
| MiscVal | [0.0, 0.0] | 缩尾处理 |
| SalePrice | [3937.5, 340037.5] | 缩尾处理 |

```python
from scipy.stats import mstats

# 需要Winsorize的数值列
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
]

# 应用Winsorize (5% - 95%)
for col in winsorize_cols:
    if col in df.columns:
        df[col] = mstats.winsorize(df[col], limits=[0.05, 0.05])
```

#### 2.3.3 保留但检查的异常值

以下列异常值比例低且可能包含真实极端值信息，建议保留：

| 列名 | 异常值比例 | 处理方式 |
|------|-----------|---------|
| OverallQual | 0.14% | 保留 |
| YearBuilt | 0.48% | 保留 |
| BsmtFinSF1 | 0.48% | 保留 |
| 2ndFlrSF | 0.14% | 保留 |
| BsmtFullBath | 0.07% | 保留 |
| Fireplaces | 0.34% | 保留 |
| GarageCars | 0.34% | 保留 |
| PoolArea | 0.48% | 保留 |

---

### 2.4 数据类型优化

将分类变量转换为 `category` 类型，减少内存占用并提高模型效率。

```python
# 分类变量列表（基于数据质量报告）
categorical_cols = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual',
    'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
    'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
    'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
    'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'MSSubClass'
]

# 转换为category类型（仅存在于df中的列）
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
```

---

### 2.5 特征工程（房价预测专用）

#### 2.5.1 创建新特征

```python
# 总面积特征（重要价格因子）
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# 房屋年龄
df['HouseAge'] = df['YrSold'] - df['YearBuilt']

# 翻新年龄
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# 总浴室数
df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']

# 总门廊面积
df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']

# 是否有泳池
df['HasPool'] = (df['PoolArea'] > 0).astype(int)

# 是否有壁炉
df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

# 是否有车库
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)

# 是否有地下室
df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)

# 是否有2楼
df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
```

#### 2.5.2 对数变换（针对RMSE优化）

由于RMSE对大误差敏感，对目标变量和右偏特征进行对数变换：

```python
import numpy as np

# 对目标变量进行对数变换（必须在训练集上fit，测试集上transform）
df['LogSalePrice'] = np.log1p(df['SalePrice'])

# 对右偏的数值特征进行对数变换
skewed_features = ['LotArea', 'LotFrontage', 'MasVnrArea', 'TotalBsmtSF', 
                   '1stFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                   'OpenPorchSF', 'TotalSF', 'TotalPorchSF']

for col in skewed_features:
    if col in df.columns and df[col].min() >= 0:
        df[f'Log_{col}'] = np.log1p(df[col])
```

---

### 2.6 最终验证

```python
# 1. 检查缺失值
print("剩余缺失值:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 2. 检查数据形状
print(f"\n清洗后数据形状: {df.shape}")

# 3. 检查数据类型
print(f"\n数值列数: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"分类列数: {df.select_dtypes(include=['category']).shape[1]}")

# 4. 目标变量统计
print(f"\nSalePrice统计:\n{df['SalePrice'].describe()}")

# 5. 保存清洗后的数据
df.to_csv('/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv', index=False)
```

---

## 3. 清洗流程总结

```
原始数据 (1460, 81)
    ↓
删除高缺失率列 (5列) → (1460, 76)
    ↓
删除低信息列 (2列) → (1460, 74)
    ↓
缺失值填充 (14列) → 无缺失
    ↓
异常值Winsorize (21列) → 缩尾处理
    ↓
特征工程 → 新增10+特征 → (1460, 85+)
    ↓
数据类型优化 → 43列转category
    ↓
对数变换 → 优化RMSE评估
    ↓
清洗后数据 (1460, 85+) 保存
```

---

## 4. 关键注意事项

### 4.1 训练-测试一致性
- 所有填充