```markdown
# 数据清洗方案

## 1. 数据概述

| 项目 | 详情 |
|------|------|
| 数据源 | `/Users/cjialin/code/AutoMLByLLM/train.csv` |
| 数据形状 | 1460 行 × 81 列 |
| 数值列数量 | 38 |
| 分类列数量 | 43 |
| 重复行数 | 0 |

---

## 2. 缺失值处理方案

### 2.1 删除高缺失率列（缺失率 > 50%）

以下列缺失率过高，信息含量极低，建议直接删除：

| 列名 | 缺失数量 | 缺失比例 | 处理方式 |
|------|----------|----------|----------|
| PoolQC | 1453 | 99.52% | 删除列 |
| MiscFeature | 1406 | 96.3% | 删除列 |
| Alley | 1369 | 93.77% | 删除列 |
| Fence | 1179 | 80.75% | 删除列 |
| MasVnrType | 872 | 59.73% | 删除列 |

**清洗代码：**
```python
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=cols_to_drop)
```

### 2.2 智能填充中等缺失率列（缺失率 5%-50%）

#### A. FireplaceQu（壁炉质量）- 缺失率 47.26%
**业务逻辑**：缺失表示该房屋没有壁炉，应填充为"None"或"NA"

```python
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
```

#### B. LotFrontage（临街宽度）- 缺失率 17.74%
**填充策略**：按社区（Neighborhood）分组使用中位数填充，因为同一社区的房屋通常具有相似的临街特征

```python
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
# 如果仍有缺失，使用整体中位数
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
```

#### C. Garage相关列（缺失率 5.55%）
涉及列：GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond

**业务逻辑**：缺失表示没有车库，分类变量填充"None"，数值变量填充0或合理默认值

```python
# 分类变量
garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cat_cols:
    df[col] = df[col].fillna('None')

# 数值变量 - 建造年份（使用房屋建造年份或0）
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
```

### 2.3 简单填充低缺失率列（缺失率 < 5%）

#### A. 地下室相关列（缺失率 ~2.6%）
涉及列：BsmtExposure, BsmtFinType2, BsmtQual, BsmtCond, BsmtFinType1

**填充策略**：缺失表示没有地下室，分类变量填充"None"

```python
bsmt_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
for col in bsmt_cols:
    df[col] = df[col].fillna('None')
```

#### B. MasVnrArea（砌体饰面面积）- 缺失率 0.55%

```python
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
```

#### C. Electrical（电力系统）- 缺失率 0.07%

```python
# 使用众数填充
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
```

---

## 3. 异常值处理方案

### 3.1 删除异常值严重的列

以下列异常值比例过高，且信息价值有限，建议删除：

| 列名 | 异常值比例 | 处理方式 |
|------|------------|----------|
| BsmtFinSF2 | 11.44% | 删除列 |
| EnclosedPorch | 14.25% | 删除列 |

```python
df = df.drop(columns=['BsmtFinSF2', 'EnclosedPorch'])
```

### 3.2 Winsorize处理（缩尾处理）

对以下数值列进行Winsorize处理（限制在5%-95%分位数或IQR范围内）：

| 列名 | 正常范围 | 处理方式 |
|------|----------|----------|
| MSSubClass | [-55.0, 145.0] | Winsorize (5%-95%) |
| LotFrontage | [27.5, 111.5] | Winsorize |
| LotArea | [1481.5, 17673.5] | Winsorize |
| OverallCond | [3.5, 7.5] | Winsorize |
| MasVnrArea | [-249.0, 415.0] | Winsorize |
| BsmtUnfSF | [-654.5, 1685.5] | Winsorize |
| TotalBsmtSF | [42.0, 2052.0] | Winsorize |
| 1stFlrSF | [118.12, 2155.12] | Winsorize |
| LowQualFinSF | [0.0, 0.0] | Winsorize |
| GrLivArea | [158.62, 2747.62] | Winsorize |
| BsmtHalfBath | [0.0, 0.0] | Winsorize |
| BedroomAbvGr | [0.5, 4.5] | Winsorize |
| KitchenAbvGr | [1.0, 1.0] | Winsorize |
| TotRmsAbvGrd | [2.0, 10.0] | Winsorize |
| GarageArea | [-27.75, 938.25] | Winsorize |
| WoodDeckSF | [-252.0, 420.0] | Winsorize |
| OpenPorchSF | [-102.0, 170.0] | Winsorize |
| 3SsnPorch | [0.0, 0.0] | Winsorize |
| ScreenPorch | [0.0, 0.0] | Winsorize |
| MiscVal | [0.0, 0.0] | Winsorize |
| SalePrice | [3937.5, 340037.5] | Winsorize |

**清洗代码：**
```python
from scipy.stats import mstats

winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'LowQualFinSF', 'GrLivArea',
    'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch',
    'MiscVal', 'SalePrice'
]

# 使用IQR方法或5%-95%分位数进行Winsorize
for col in winsorize_cols:
    if col in df.columns:
        lower = df[col].quantile(0.05)
        upper = df[col].quantile(0.95)
        df[col] = df[col].clip(lower=lower, upper=upper)
```

### 3.3 保留的列（异常值比例极低或有业务意义）

以下列异常值比例较低或属于合理业务范围，予以保留：
- OverallQual (0.14%)
- YearBuilt (0.48%)
- BsmtFinSF1 (0.48%)
- 2ndFlrSF (0.14%)
- BsmtFullBath (0.07%)
- Fireplaces (0.34%)
- GarageCars (0.34%)
- PoolArea (0.48%)

---

## 4. 数据类型优化方案

### 4.1 分类变量转换为Category类型

将以下43个分类列转换为category类型以节省内存：

```python
categorical_cols = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
    'SaleType', 'SaleCondition', 'MSSubClass'
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
```

### 4.2 数值类型优化

```python
# 将整数列转换为Int64（可处理缺失值）
int_cols = df.select_dtypes(include=['int64']).columns
for col in int_cols:
    df[col] = df[col].astype('Int64')

# 将浮点数列转换为float32以节省内存
float_cols = df.select_dtypes(include=['float64']).columns
for col in float_cols:
    df[col] = df[col].astype('float32')
```

---

## 5. 重复值处理

数据集中无重复行（重复率 0.0%），无需处理。

如需额外检查：
```python
df = df.drop_duplicates()
```

---

## 6. 完整清洗流程代码

```python
import pandas as pd
import numpy as np
from scipy.stats import mstats

def clean_housing_data(file_path):
    """
    房屋数据清洗函数
    """
    # 1. 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # 2. 删除高缺失率列
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df = df.drop(columns=cols_to_drop)
    
    # 3. 删除异常值严重的列
    df = df.drop(columns=['BsmtFinSF2', 'EnclosedPorch'])
    
    # 4. 填充缺失值
    # 4.1 FireplaceQu
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
    
    # 4.2 LotFrontage
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # 4.3 Garage相关列
    garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cat_cols:
        df[col] = df[col].fillna('None')
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    
    # 4.4 地下室相关列
    bsmt_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
    for col in bsmt_cols:
        df[col] = df[col].fillna('None')
    
    # 4.5 其他列
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    # 5. 异常值处理（Winsorize）
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'LowQualFinSF', 'GrLivArea',
        'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'GarageArea', 'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch',
        'MiscVal', 'SalePrice'
    ]
    
    for col in winsorize_cols:
        if col in df.columns:
            lower = df[col].quantile(0.05)
            upper = df[col].quantile(0.95)
            df[col] = df[col].clip(lower=lower, upper=upper)
    
    # 6. 数据类型优化
    categorical_cols = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
        'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'PavedDrive', 'SaleType', 'SaleCondition'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # 7. 删除重复值（如有）
    df = df.drop_duplicates()
    
    print(f"清洗后数据形状: {df.shape}")
    print(f"剩余缺失值总数: {df.isnull().sum().sum()}")
    
    return df

# 执行清洗
df_cleaned = clean_housing_data('/Users/cjialin/code/AutoMLByLLM/train.csv')
df_cleaned.to_csv('/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv', index=False)
```

---

## 7. 清洗效果验证指标

| 验证项 | 目标值 | 验证方法 |
|--------|--------|----------|
| 缺失值总数 | 0 | `df.isnull().sum().sum()` |
| 重复行数 | 0 | `df.duplicated().sum()` |
| 异常值比例 | < 1% | IQR方法重新检测 |
| 数据类型一致性 | 100% | 分类列均为category |
| 行数保留率 | > 95% | 清洗后行数/原始行数 |

---

## 8. 注意事项

1. **业务理解**：本方案基于房屋数据的业务逻辑（如"NA"表示没有该设施），在应用前请确认业务场景是否匹配
2. **顺序重要**：必须先删除列，再填充缺失值，最后处理异常值
3. **Winsorize参数**：可根据实际业务需求调整分位数阈值（当前使用5%-95%）
4. **备份数据**：清洗前务必备份原始数据
```