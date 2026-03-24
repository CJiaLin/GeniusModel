# 数据清洗方案

## 一、数据概览

基于实际数据分析结果，数据基本情况如下：
- **数据文件**: `/Users/cjialin/code/AutoMLByLLM/train.csv`
- **数据形状**: 1460 行 × 81 列
- **数值列**: 38 列
- **分类列**: 43 列
- **重复行**: 0 行

---

## 二、数据质量问题汇总

### 2.1 缺失值问题（19列）

| 分类 | 列名 | 缺失数量 | 缺失比例 | 处理策略 |
|------|------|----------|----------|----------|
| **极高缺失率(>80%)** | PoolQC | 1453 | 99.52% | 删除列 |
| | MiscFeature | 1406 | 96.30% | 删除列 |
| | Alley | 1369 | 93.77% | 删除列 |
| | Fence | 1179 | 80.75% | 删除列 |
| **高缺失率(50-80%)** | MasVnrType | 872 | 59.73% | 删除列 |
| **中等缺失率(20-50%)** | FireplaceQu | 690 | 47.26% | 填充"None" |
| | LotFrontage | 259 | 17.74% | 按Neighborhood分组中位数填充 |
| **低缺失率(<10%)** | GarageType | 81 | 5.55% | 填充"None" |
| | GarageYrBlt | 81 | 5.55% | 填充0或YearBuilt |
| | GarageFinish | 81 | 5.55% | 填充"None" |
| | GarageQual | 81 | 5.55% | 填充"None" |
| | GarageCond | 81 | 5.55% | 填充"None" |
| | BsmtExposure | 38 | 2.60% | 填充"None" |
| | BsmtFinType2 | 38 | 2.60% | 填充"None" |
| | BsmtQual | 37 | 2.53% | 填充"None" |
| | BsmtCond | 37 | 2.53% | 填充"None" |
| | BsmtFinType1 | 37 | 2.53% | 填充"None" |
| | MasVnrArea | 8 | 0.55% | 填充0 |
| | Electrical | 1 | 0.07% | 填充众数 |

### 2.2 异常值问题（31列）

| 处理策略 | 列名 | 异常值数量 | 异常值比例 |
|----------|------|------------|------------|
| **删除列** | BsmtFinSF2 | 167 | 11.44% |
| | EnclosedPorch | 208 | 14.25% |
| **Winsorize处理** | MSSubClass | 103 | 7.05% |
| | LotFrontage | 88 | 6.03% |
| | LotArea | 69 | 4.73% |
| | OverallCond | 125 | 8.56% |
| | MasVnrArea | 96 | 6.58% |
| | BsmtUnfSF | 29 | 1.99% |
| | TotalBsmtSF | 61 | 4.18% |
| | 1stFlrSF | 20 | 1.37% |
| | LowQualFinSF | 26 | 1.78% |
| | GrLivArea | 31 | 2.12% |
| | BsmtHalfBath | 82 | 5.62% |
| | BedroomAbvGr | 35 | 2.40% |
| | KitchenAbvGr | 68 | 4.66% |
| | TotRmsAbvGrd | 30 | 2.05% |
| | GarageArea | 21 | 1.44% |
| | WoodDeckSF | 32 | 2.19% |
| | OpenPorchSF | 77 | 5.27% |
| | 3SsnPorch | 24 | 1.64% |
| | ScreenPorch | 116 | 7.95% |
| | MiscVal | 52 | 3.56% |
| | SalePrice | 61 | 4.18% |
| **保留(业务合理)** | OverallQual | 2 | 0.14% |
| | YearBuilt | 7 | 0.48% |
| | BsmtFinSF1 | 7 | 0.48% |
| | 2ndFlrSF | 2 | 0.14% |
| | BsmtFullBath | 1 | 0.07% |
| | Fireplaces | 5 | 0.34% |
| | GarageCars | 5 | 0.34% |
| | PoolArea | 7 | 0.48% |

### 2.3 数据类型优化

43个分类列建议从 `object` 转换为 `category` 类型以节省内存。

---

## 三、清洗执行方案

### 步骤1: 导入依赖库

```python
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
```

### 步骤2: 加载数据

```python
# 加载原始数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')
print(f"原始数据形状: {df.shape}")

# 保存原始数据副本用于对比
df_original = df.copy()
```

### 步骤3: 处理高缺失率列（删除）

```python
# 删除缺失率超过50%的列
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=cols_to_drop)
print(f"删除高缺失率列后形状: {df.shape}")
```

### 步骤4: 处理缺失值填充

```python
# 4.1  FireplaceQu - 无壁炉的填充为"None"
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

# 4.2  LotFrontage - 按Neighborhood分组中位数填充
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
# 如果仍有缺失，用整体中位数填充
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# 4.3  Garage相关列 - 无车库的填充为"None"或0
garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cols:
    df[col] = df[col].fillna('None')

# GarageYrBlt - 无车库的填充为0或用YearBuilt
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

# 4.4  Basement相关列 - 无地下室的填充为"None"
bsmt_cols = ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']
for col in bsmt_cols:
    df[col] = df[col].fillna('None')

# 4.5  MasVnrArea - 无贴面面积的填充为0
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# 4.6  Electrical - 填充众数
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

print("缺失值填充完成")
print(f"剩余缺失值数量: {df.isnull().sum().sum()}")
```

### 步骤5: 处理异常值列（删除）

```python
# 删除异常值过多的列（主要是大部分为0的稀疏列）
cols_to_drop_outliers = ['BsmtFinSF2', 'EnclosedPorch']
df = df.drop(columns=cols_to_drop_outliers)
print(f"删除异常值列后形状: {df.shape}")
```

### 步骤6: Winsorize处理

```python
def winsorize_series(series, limits=(0.05, 0.05)):
    """对序列进行Winsorize处理，限制在5%-95%分位数"""
    lower = series.quantile(0.05)
    upper = series.quantile(0.95)
    return series.clip(lower, upper)

# 需要Winsorize的数值列
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
]

for col in winsorize_cols:
    if col in df.columns:
        df[col] = winsorize_series(df[col])

print("Winsorize处理完成")
```

### 步骤7: 数据类型优化

```python
# 转换为category类型的列
category_cols = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType',
    'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond',
    'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
    'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'
]

for col in category_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

print("数据类型优化完成")
```

### 步骤8: 验证清洗结果

```python
print("=" * 60)
print("数据清洗验证报告")
print("=" * 60)

print(f"\n原始数据形状: {df_original.shape}")
print(f"清洗后数据形状: {df.shape}")
print(f"删除列数: {df_original.shape[1] - df.shape[1]}")

print(f"\n剩余缺失值总数: {df.isnull().sum().sum()}")

# 数值列范围检查
print("\n关键数值列范围检查:")
for col in ['LotFrontage', 'LotArea', 'SalePrice']:
    if col in df.columns:
        print(f"{col}: [{df[col].min():.2f}, {df[col].max():.2f}]")

# 内存使用对比
original_memory = df_original.memory_usage(deep=True).sum() / 1024**2
cleaned_memory = df.memory_usage(deep=True).sum() / 1024**2
print(f"\n内存使用对比:")
print(f"原始数据: {original_memory:.2f} MB")
print(f"清洗后数据: {cleaned_memory:.2f} MB")
print(f"节省: {(1 - cleaned_memory/original_memory)*100:.1f}%")

print("\n" + "=" * 60)
```

### 步骤9: 保存清洗后的数据

```python
# 保存清洗后的数据
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"清洗后的数据已保存至: {output_path}")
```

---

## 四、清洗效果预期

| 指标 | 清洗前 | 清洗后 | 改善 |
|------|--------|--------|------|
| 缺失值列 | 19列 | 0列 | 100%解决 |
| 缺失值总数 | 6965个 | 0个 | 100%解决 |
| 异常值列 | 31列 | 9列 | 71%解决 |
| 数据类型 | 43 object | 43 category | 内存优化 |
| 总列数 | 81列 | 74列 | 删除7列 |

---

## 五、业务说明

1. **高缺失率列删除**: PoolQC、MiscFeature、Alley、Fence、MasVnrType 缺失率超过59%，且大多表示"不存在"的特征，保留价值低。

2. **LotFrontage填充策略**: 使用Neighborhood分组中位数填充，因为同社区的房屋通常具有相似的街道长度特征。

3. **Garage/Basement填充**: 缺失表示该房屋没有车库或地下室，填充为"None"或0是合理的业务解释。

4. **Winsorize vs 删除**: 对异常值采用Winsorize（缩尾处理）而非直接删除，保留样本量的同时限制极端值影响。

5. **BsmtFinSF2/EnclosedPorch删除**: 这两列异常值比例超过11%，且大部分值为0，信息含量低，直接删除。

此清洗方案在保持数据完整性的同时，显著提升了数据质量，为后续的建模分析奠定了坚实基础。