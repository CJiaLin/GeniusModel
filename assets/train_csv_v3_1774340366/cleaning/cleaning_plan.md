# 数据清洗方案

## 📊 数据概览

| 项目 | 信息 |
|------|------|
| 数据文件 | `/Users/cjialin/code/AutoMLByLLM/train.csv` |
| 数据形状 | 1460 行 × 81 列 |
| 数值列 | 38 列 |
| 分类列 | 43 列 |
| 重复行 | 0 行 |

---

## 🔍 数据质量问题诊断

### 1. 缺失值问题（5个等级）

| 等级 | 列名 | 缺失比例 | 处理策略 |
|------|------|----------|----------|
| **极高** | `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `MasVnrType` | >59% | **删除整列** |
| **高** | `FireplaceQu` | 47.26% | 填充为"None"（无壁炉） |
| **中** | `LotFrontage` | 17.74% | 按Neighborhood分组中位数填充 |
| **低** | `GarageType`, `GarageYrBlt`, `GarageFinish`, `GarageQual`, `GarageCond` | 5.55% | 条件填充（无车库） |
| **极低** | `BsmtExposure`, `BsmtFinType2`, `BsmtQual`, `BsmtCond`, `BsmtFinType1`, `MasVnrArea`, `Electrical` | <3% | 众数/中位数填充 |

### 2. 异常值问题（需Winsorize的列）

| 列名 | 异常值数量 | 正常范围 | 处理方式 |
|------|------------|----------|----------|
| `MSSubClass` | 103 | [-55.0, 145.0] | Winsorize |
| `LotFrontage` | 88 | [27.5, 111.5] | Winsorize |
| `LotArea` | 69 | [1481.5, 17673.5] | Winsorize |
| `OverallCond` | 125 | [3.5, 7.5] | Winsorize |
| `MasVnrArea` | 96 | [-249.0, 415.0] | Winsorize |
| `BsmtUnfSF` | 29 | [-654.5, 1685.5] | Winsorize |
| `TotalBsmtSF` | 61 | [42.0, 2052.0] | Winsorize |
| `1stFlrSF` | 20 | [118.12, 2155.12] | Winsorize |
| `LowQualFinSF` | 26 | [0.0, 0.0] | Winsorize |
| `GrLivArea` | 31 | [158.62, 2747.62] | Winsorize |
| `BsmtHalfBath` | 82 | [0.0, 0.0] | Winsorize |
| `BedroomAbvGr` | 35 | [0.5, 4.5] | Winsorize |
| `KitchenAbvGr` | 68 | [1.0, 1.0] | Winsorize |
| `TotRmsAbvGrd` | 30 | [2.0, 10.0] | Winsorize |
| `GarageArea` | 21 | [-27.75, 938.25] | Winsorize |
| `WoodDeckSF` | 32 | [-252.0, 420.0] | Winsorize |
| `OpenPorchSF` | 77 | [-102.0, 170.0] | Winsorize |
| `3SsnPorch` | 24 | [0.0, 0.0] | Winsorize |
| `ScreenPorch` | 116 | [0.0, 0.0] | Winsorize |
| `MiscVal` | 52 | [0.0, 0.0] | Winsorize |
| `SalePrice` | 61 | [3937.5, 340037.5] | Winsorize |

### 3. 数据类型优化

43个分类变量（如 `MSZoning`, `Street`, `Alley`, `LotShape` 等）建议从 `object` 转换为 `category` 类型以节省内存。

---

## 🛠️ 数据清洗实施方案

### 阶段一：删除高缺失率列

```python
import pandas as pd
import numpy as np
from scipy.stats import mstats

# 加载数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 阶段1: 删除缺失率超过50%的列
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df_cleaned = df.drop(columns=cols_to_drop)
print(f"删除 {len(cols_to_drop)} 列，剩余 {df_cleaned.shape[1]} 列")
```

### 阶段二：缺失值填充

```python
# 阶段2: 智能填充缺失值

# 2.1 FireplaceQu - 填充为"None"表示无壁炉
df_cleaned['FireplaceQu'] = df_cleaned['FireplaceQu'].fillna('None')

# 2.2 LotFrontage - 按Neighborhood分组中位数填充
df_cleaned['LotFrontage'] = df_cleaned.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
# 如果仍有缺失，用全局中位数填充
df_cleaned['LotFrontage'] = df_cleaned['LotFrontage'].fillna(df_cleaned['LotFrontage'].median())

# 2.3 Garage相关列 - 根据GarageArea或GarageCars判断是否有车库
garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cols:
    df_cleaned[col] = df_cleaned[col].fillna('None')

# GarageYrBlt - 用YearBuilt填充（无车库则认为与建造年份相同）
df_cleaned['GarageYrBlt'] = df_cleaned['GarageYrBlt'].fillna(df_cleaned['YearBuilt'])

# 2.4 Bsmt相关列 - 根据TotalBsmtSF判断是否有地下室
bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_cat_cols:
    df_cleaned[col] = df_cleaned[col].fillna('None')

# 2.5 MasVnrArea - 用0填充（无砌体贴面）
df_cleaned['MasVnrArea'] = df_cleaned['MasVnrArea'].fillna(0)

# 2.6 Electrical - 用众数填充
df_cleaned['Electrical'] = df_cleaned['Electrical'].fillna(df_cleaned['Electrical'].mode()[0])

print(f"缺失值填充完成，剩余缺失值: {df_cleaned.isnull().sum().sum()}")
```

### 阶段三：异常值处理（Winsorize）

```python
# 阶段3: Winsorize异常值（缩尾处理）
# 使用1%和99%分位数进行缩尾

winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
]

for col in winsorize_cols:
    lower = df_cleaned[col].quantile(0.01)
    upper = df_cleaned[col].quantile(0.99)
    df_cleaned[col] = df_cleaned[col].clip(lower, upper)

print(f"Winsorize处理完成，处理了 {len(winsorize_cols)} 列")
```

### 阶段四：数据类型优化

```python
# 阶段4: 将分类变量转换为category类型
categorical_cols = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
    'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
]

# 只转换实际存在的列
existing_cat_cols = [col for col in categorical_cols if col in df_cleaned.columns]
for col in existing_cat_cols:
    df_cleaned[col] = df_cleaned[col].astype('category')

print(f"数据类型优化完成，转换了 {len(existing_cat_cols)} 列")
```

### 阶段五：验证与保存

```python
# 阶段5: 验证清洗结果
print("=" * 50)
print("清洗后数据质量报告")
print("=" * 50)
print(f"数据形状: {df_cleaned.shape}")
print(f"缺失值总数: {df_cleaned.isnull().sum().sum()}")
print(f"重复行数: {df_cleaned.duplicated().sum()}")
print(f"内存使用: {df_cleaned.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 保存清洗后的数据
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df_cleaned.to_csv(output_path, index=False)
print(f"\n清洗后的数据已保存至: {output_path}")
```

---

## 📈 清洗效果验证指标

| 验证项 | 清洗前 | 清洗后 | 目标 |
|--------|--------|--------|------|
| 缺失值总数 | >3000 | 0 | ✅ 完全消除 |
| 高缺失率列 | 5列 | 0列 | ✅ 已删除 |
| 异常值处理 | 30+列 | 已Winsorize | ✅ 已处理 |
| 重复行 | 0 | 0 | ✅ 保持清洁 |
| 数据类型优化 | 0列 | ~38列 | ✅ 内存优化 |

---

## ⚠️ 注意事项

1. **业务理解**：`FireplaceQu`、`Garage`相关列、`Bsmt`相关列的缺失值实际上表示"无该设施"，填充为"None"是合理的业务解释。

2. **LotFrontage处理**：采用按`Neighborhood`分组填充，因为同一街区的房屋通常具有相似的前 lot 宽度。

3. **异常值保留列**：`OverallQual`、`YearBuilt`、`BsmtFinSF1`、`2ndFlrSF`、`BsmtFullBath`、`Fireplaces`、`GarageCars`、`PoolArea`的异常值比例极低(<0.5%)或具有业务合理性，予以保留。

4. **目标变量**：`SalePrice`的异常值采用Winsorize处理，避免极端房价影响模型训练。

5. **Id列**：保留`Id`列用于后续数据追踪和合并操作。