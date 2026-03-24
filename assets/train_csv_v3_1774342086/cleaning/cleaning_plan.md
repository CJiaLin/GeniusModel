# 数据清洗方案报告

## 1. 数据源信息

| 项目 | 详情 |
|------|------|
| 文件路径 | `/Users/cjialin/code/AutoMLByLLM/train.csv` |
| 数据形状 | 1,460 行 × 81 列 |
| 数值列数 | 38 |
| 分类列数 | 43 |
| 重复行数 | 0 |

---

## 2. 数据质量问题概览

基于实际数据分析，发现 **50 类数据质量问题**：
- **高缺失率列**: 5 列缺失率 > 50%
- **中等缺失率列**: 14 列需要填充处理
- **异常值列**: 26 列存在异常值（其中 2 列建议删除）
- **数据类型优化**: 43 列可转换为 category 类型

---

## 3. 详细清洗方案

### 3.1 缺失值处理方案

#### 🔴 第一阶段：删除高缺失率列（缺失率 > 50%）
基于业务理解，这些列缺失率过高，填充价值低，建议直接删除。

```python
# 待删除列列表
cols_to_drop = [
    'PoolQC',      # 缺失率 99.52% (1453/1460)
    'MiscFeature', # 缺失率 96.30% (1406/1460)
    'Alley',       # 缺失率 93.77% (1369/1460)
    'Fence',       # 缺失率 80.75% (1179/1460)
    'MasVnrType'   # 缺失率 59.73% (872/1460)
]
```

#### 🟡 第二阶段：智能填充（缺失率 5%-50%）

** FireplaceQu (缺失率 47.26%) **
- 业务逻辑：无 Fireplace 的住宅无此字段，应填充 "NoFireplace"
- 填充策略：分类值填充

** LotFrontage (缺失率 17.74%) **
- 业务逻辑：临街距离与社区相关
- 填充策略：按 Neighborhood 分组中位数填充

** Garage 相关列 (缺失率 5.55%) **
- 涉及列：GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond
- 业务逻辑：无车库的住宅无这些字段
- 填充策略：
  - GarageType: 填充 "NoGarage"
  - GarageYrBlt: 填充 0 或建筑年份
  - GarageFinish, GarageQual, GarageCond: 填充 "NoGarage"

```python
# Garage 填充逻辑
garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cols:
    df[col].fillna('NoGarage', inplace=True)

# GarageYrBlt 填充为 YearBuilt（假设无车库记录时与建筑年份相同）
df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)
```

#### 🟢 第三阶段：简单填充（缺失率 < 5%）

** 地下室相关列 (缺失率 ~2.5%) **
- 涉及列：BsmtExposure, BsmtFinType2, BsmtQual, BsmtCond, BsmtFinType1
- 填充策略：填充 "NoBasement"

** 其他低缺失列 **
- MasVnrArea (0.55%): 中位数填充 或 0
- Electrical (0.07%): 众数填充

```python
# 地下室列填充
bsmt_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
for col in bsmt_cols:
    df[col].fillna('NoBasement', inplace=True)

# MasVnrArea 填充 0（表示无饰面）
df['MasVnrArea'].fillna(0, inplace=True)

# Electrical 众数填充
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
```

---

### 3.2 异常值处理方案

#### 🔴 删除异常值比例过高的列
这些列异常值比例过高，数据质量差，建议删除。

```python
# 删除列列表（异常值比例 > 10% 且业务价值低）
cols_drop_outlier = [
    'BsmtFinSF2',      # 异常值 11.44%（大部分为0）
    'EnclosedPorch'    # 异常值 14.25%（大部分为0）
]
```

#### 🟡 Winsorize 处理（缩尾处理）
对以下列进行 1%-99% 分位数缩尾，保留极端值但限制影响：

```python
# 需要 Winsorize 的数值列列表
winsorize_cols = [
    'MSSubClass',      # 正常范围: [-55.0, 145.0]
    'LotFrontage',     # 正常范围: [27.5, 111.5]
    'LotArea',         # 正常范围: [1481.5, 17673.5]
    'OverallCond',     # 正常范围: [3.5, 7.5]
    'MasVnrArea',      # 正常范围: [-249.0, 415.0]
    'BsmtUnfSF',       # 正常范围: [-654.5, 1685.5]
    'TotalBsmtSF',     # 正常范围: [42.0, 2052.0]
    '1stFlrSF',        # 正常范围: [118.12, 2155.12]
    'LowQualFinSF',    # 正常范围: [0.0, 0.0]
    'GrLivArea',       # 正常范围: [158.62, 2747.62]
    'BsmtHalfBath',    # 正常范围: [0.0, 0.0]
    'BedroomAbvGr',    # 正常范围: [0.5, 4.5]
    'KitchenAbvGr',    # 正常范围: [1.0, 1.0]
    'TotRmsAbvGrd',    # 正常范围: [2.0, 10.0]
    'GarageArea',      # 正常范围: [-27.75, 938.25]
    'WoodDeckSF',      # 正常范围: [-252.0, 420.0]
    'OpenPorchSF',     # 正常范围: [-102.0, 170.0]
    '3SsnPorch',       # 正常范围: [0.0, 0.0]
    'ScreenPorch',     # 正常范围: [0.0, 0.0]
    'MiscVal',         # 正常范围: [0.0, 0.0]
    'SalePrice'        # 正常范围: [3937.5, 340037.5]
]

# Winsorize 函数实现
def winsorize_series(series, lower_percentile=0.01, upper_percentile=0.99):
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower, upper)
```

#### 🟢 保留的异常值列
以下列虽然检测到异常值，但属于正常业务范围，予以保留：

```python
# 保留列列表（业务正常值）
keep_outlier_cols = [
    'OverallQual',     # 评分 1-10，超出范围样本少
    'YearBuilt',       # 老建筑和新建筑均为正常历史数据
    'BsmtFinSF1',      # 大面积装修属正常情况
    '2ndFlrSF',        # 无二层或大面积二层均正常
    'BsmtFullBath',    # 数量差异属正常
    'Fireplaces',      # 多个壁炉属正常
    'GarageCars',      # 多车位属正常
    'PoolArea'         # 无泳池或大面积泳池均正常
]
```

---

### 3.3 数据类型优化方案

将 43 个分类列从 object 转换为 category 类型，减少内存占用并提高处理效率。

```python
# 分类列列表（基于实际数据）
categorical_cols = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
    'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType',
    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
    'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
    'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
]

# 转换为 category 类型
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
```

---

## 4. 完整清洗代码实现

```python
import pandas as pd
import numpy as np
from scipy.stats import mstats

def clean_housing_data(file_path):
    """
    房价数据清洗完整流程
    基于实际数据质量报告 (1460行 × 81列)
    """
    
    # 1. 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # 2. 删除高缺失率列 (缺失率 > 50%)
    cols_high_missing = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df.drop(columns=cols_high_missing, inplace=True, errors='ignore')
    print(f"删除高缺失列后: {df.shape}")
    
    # 3. 删除异常值比例过高的列
    cols_high_outlier = ['BsmtFinSF2', 'EnclosedPorch']
    df.drop(columns=cols_high_outlier, inplace=True, errors='ignore')
    print(f"删除高异常值列后: {df.shape}")
    
    # 4. 缺失值填充
    
    # 4.1 FireplaceQu - 无壁炉标记
    df['FireplaceQu'].fillna('NoFireplace', inplace=True)
    
    # 4.2 LotFrontage - 按社区分组中位数填充
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # 4.3 Garage 相关列
    garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cat_cols:
        df[col].fillna('NoGarage', inplace=True)
    df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)
    
    # 4.4 Basement 相关列
    bsmt_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
    for col in bsmt_cols:
        df[col].fillna('NoBasement', inplace=True)
    
    # 4.5 其他数值/分类列
    df['MasVnrArea'].fillna(0, inplace=True)
    df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
    
    print(f"缺失值填充完成，剩余缺失值: {df.isnull().sum().sum()}")
    
    # 5. 异常值处理 (Winsorize)
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'LowQualFinSF', 'GrLivArea',
        'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'GarageArea', 'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch',
        'MiscVal', 'SalePrice'
    ]
    
    for col in winsorize_cols:
        if col in df.columns:
            lower, upper = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower, upper)
    
    print("异常值 Winsorize 处理完成")
    
    # 6. 数据类型优化
    categorical_cols = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond',
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    print("数据类型转换完成")
    
    # 7. 验证
    print(f"\n清洗后数据形状: {df.shape}")
    print(f"剩余缺失值总数: {df.isnull().sum().sum()}")
    print(f"重复行数: {df.duplicated().sum()}")
    
    return df

# 执行清洗
df_cleaned = clean_housing_data('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 保存清洗后数据
df_cleaned.to_csv('/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv', index=False)
```

---

## 5. 清洗效果验证指标

| 验证项目 | 清洗前 | 清洗后 | 目标 |
|---------|--------|--------|------|
| 数据形状 | (1460, 81) | (1460, 74) | 删除 7 列 |
| 缺失值总数 | >3,000 | 0 | 完全填充 |
| 重复行数 | 0 | 0 | 保持无重复 |
| 内存占用 | 高 (object) | 低 (category) | 优化存储 |
| 异常值影响 | 高 | 可控 | Winsorize |

---

## 6. 注意事项

1. **业务理解**：FireplaceQu、Garage 等列的缺失表示"无该设施"，而非数据缺失，已用特定值标记
2. **Winsorize 边界**：使用 1%-99% 分位数，既保留数据分布特征，又控制极端值影响
3. **分类变量**：转换为 category 类型可显著减少内存使用（预估减少 50% 以上）
4. **后续建模**：建议对 SalePrice 进行对数变换（通常房价呈右偏分布）

---

**方案基于实际数据质量报告生成，所有列名与实际数据一致。**