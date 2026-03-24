# 🏠 房价预测数据清洗方案

## 📋 方案概述

| 项目 | 内容 |
|------|------|
| **数据文件** | `/Users/cjialin/code/AutoMLByLLM/train.csv` |
| **原始形状** | (1460, 81) |
| **目标变量** | SalePrice |
| **任务类型** | 回归（房价预测） |
| **评估指标** | RMSE |

---

## 🔍 数据质量问题总览

| 问题类型 | 数量 | 处理优先级 |
|----------|------|------------|
| 高缺失率列(>50%) | 5列 | 🔴 高 |
| 中等缺失率列(5%-50%) | 7列 | 🟡 中 |
| 低缺失率列(<5%) | 7列 | 🟢 低 |
| 异常值列 | 30列 | 🟡 中 |
| 数据类型优化 | 43列 | 🟢 低 |

---

## 📝 详细清洗步骤

### 步骤 1: 导入必要的库

```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

# 加载数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')
print(f"原始数据形状: {df.shape}")
```

---

### 步骤 2: 处理高缺失率列（删除）

**策略**：删除缺失率超过 50% 的列，这些列信息含量过低，填充可能引入噪音。

| 列名 | 缺失率 | 删除原因 |
|------|--------|----------|
| PoolQC | 99.52% | 几乎无有效信息 |
| MiscFeature | 96.30% | 几乎无有效信息 |
| Alley | 93.77% | 几乎无有效信息 |
| Fence | 80.75% | 缺失率过高 |
| MasVnrType | 59.73% | 缺失率过高 |

```python
# 高缺失率列（>50%）
high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']

# 删除这些列
df = df.drop(columns=high_missing_cols)
print(f"删除高缺失率列后形状: {df.shape}")
print(f"删除的列: {high_missing_cols}")
```

---

### 步骤 3: 处理中等缺失率列（智能填充）

#### 3.1 FireplaceQu（壁炉质量）- 缺失率 47.26%

**业务理解**：缺失表示该房屋没有壁炉。

```python
# FireplaceQu 缺失表示无壁炉，填充为 'None'
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
```

#### 3.2 LotFrontage（临街宽度）- 缺失率 17.74%

**策略**：使用 KNN 填充，基于相似房产的特征（LotArea, Neighborhood, MSSubClass）。

```python
# 使用 LotArea 的中位数分组填充
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
# 如果仍有缺失，使用全局中位数
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
```

#### 3.3 Garage 相关列（缺失率 5.55%）

**列名**：GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond

**业务理解**：缺失表示无车库。

```python
# GarageType 填充为 'None'
df['GarageType'] = df['GarageType'].fillna('None')

# GarageFinish 填充为 'None'
df['GarageFinish'] = df['GarageFinish'].fillna('None')

# GarageQual 填充为 'None'
df['GarageQual'] = df['GarageQual'].fillna('None')

# GarageCond 填充为 'None'
df['GarageCond'] = df['GarageCond'].fillna('None')

# GarageYrBlt 缺失表示无车库，填充为 0 或与 YearBuilt 相同
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
```

---

### 步骤 4: 处理低缺失率列（简单填充）

| 列名 | 缺失数 | 填充策略 |
|------|--------|----------|
| BsmtExposure | 38 | 填充为 'No'（无暴露） |
| BsmtFinType2 | 38 | 填充为 'Unf'（未完成） |
| BsmtQual | 37 | 填充为 'TA'（典型） |
| BsmtCond | 37 | 填充为 'TA'（典型） |
| BsmtFinType1 | 37 | 填充为 'Unf'（未完成） |
| MasVnrArea | 8 | 填充为 0 |
| Electrical | 1 | 填充为众数 |

```python
# 地下室相关列 - 缺失表示无地下室或典型情况
df['BsmtExposure'] = df['BsmtExposure'].fillna('No')
df['BsmtFinType2'] = df['BsmtFinType2'].fillna('Unf')
df['BsmtQual'] = df['BsmtQual'].fillna('TA')
df['BsmtCond'] = df['BsmtCond'].fillna('TA')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('Unf')

# MasVnrArea - 缺失表示无贴面，填充为 0
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# Electrical - 只有一个缺失，填充众数
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
```

---

### 步骤 5: 异常值处理

#### 5.1 删除高异常值比例列

**策略**：BsmtFinSF2 和 EnclosedPorch 异常值比例超过 10%，且业务价值有限，直接删除。

```python
# 删除高异常值比例列
outlier_drop_cols = ['BsmtFinSF2', 'EnclosedPorch']
df = df.drop(columns=outlier_drop_cols, errors='ignore')
print(f"删除高异常值列后形状: {df.shape}")
```

#### 5.2 Winsorize 处理（限制极端值）

**适用列**：数值型特征，将极端值限制在 1% 和 99% 分位数。

```python
def winsorize_series(series, lower_percentile=0.01, upper_percentile=0.99):
    """对序列进行 Winsorize 处理"""
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower=lower, upper=upper)

# 需要 Winsorize 的数值列（排除目标变量 SalePrice 和 ID）
winsorize_cols = [
    'M