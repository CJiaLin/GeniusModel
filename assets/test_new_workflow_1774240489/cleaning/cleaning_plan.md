# 数据清洗方案

## 数据概览

| 项目 | 值 |
|------|-----|
| **数据路径** | `/Users/cjialin/code/AutoMLByLLM/train.csv` |
| **数据形状** | 1,460 行 × 81 列 |
| **目标变量** | `SalePrice` |
| **数据类型分布** | 数值型: 38列, 分类型: 43列 |

---

## 1. 数据质量问题分析

### 1.1 缺失值分析

根据实际数据统计，缺失值分布如下：

#### 🔴 高缺失率特征（>80%）- 表示"不存在"
| 列名 | 缺失数 | 缺失率 | 业务含义 |
|------|--------|--------|----------|
| `PoolQC` | 1,453 | 99.5% | 无游泳池 |
| `MiscFeature` | 1,406 | 96.3% | 无其他设施 |
| `Alley` | 1,369 | 93.8% | 无巷道入口 |
| `Fence` | 1,179 | 80.8% | 无围栏 |

#### 🟡 中等缺失率特征（40-60%）
| 列名 | 缺失数 | 缺失率 | 业务含义 |
|------|--------|--------|----------|
| `FireplaceQu` | 690 | 47.3% | 无壁炉 |
| `MasVnrType` | 872 | 59.7% | 无砌体饰面 |

#### 🟢 低缺失率特征（<20%）
| 列名 | 缺失数 | 缺失率 | 处理策略 |
|------|--------|--------|----------|
| `LotFrontage` | 259 | 17.7% | 需插值填充 |
| `GarageType` | 81 | 5.5% | 无车库 |
| `GarageYrBlt` | 81 | 5.5% | 无车库 |
| `GarageFinish` | 81 | 5.5% | 无车库 |
| `GarageQual` | 81 | 5.5% | 无车库 |
| `GarageCond` | 81 | 5.5% | 无车库 |
| `BsmtQual` | 37 | 2.5% | 无地下室 |
| `BsmtCond` | 37 | 2.5% | 无地下室 |
| `BsmtExposure` | 38 | 2.6% | 无地下室 |
| `BsmtFinType1` | 37 | 2.5% | 无地下室 |
| `BsmtFinType2` | 38 | 2.6% | 无地下室 |
| `MasVnrArea` | 8 | 0.5% | 需填充 |
| `Electrical` | 1 | 0.07% | 需填充 |

### 1.2 数据类型问题

- **`MSSubClass`** (int64): 实际是分类变量（住宅类型代码），需转换为object
- **`MoSold`**, **`YrSold`**: 可考虑转换为周期性特征
- **年份相关变量**: `YearBuilt`, `YearRemodAdd`, `GarageYrBlt` 可计算房龄等衍生特征

### 1.3 潜在异常值检查点

- `LotArea`: 需检查极大值（如超过50000）
- `SalePrice`: 目标变量，检查右偏分布和极值
- `GrLivArea`: 地面生活面积，检查异常大值
- `1stFlrSF`, `2ndFlrSF`: 检查不合理组合

---

## 2. 详细清洗步骤

### 步骤 1: 缺失值分类处理

#### A. "不存在"类型缺失 → 填充为 "None" 或 0

```python
# 分类变量：填充 "None"
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']

for col in none_cols:
    df[col] = df[col].fillna('None')

# 数值变量：填充 0
zero_cols = ['GarageYrBlt', 'MasVnrArea']
for col in zero_cols:
    df[col] = df[col].fillna(0)
```

#### B. 逻辑依赖缺失 → 基于统计填充

```python
# LotFrontage: 按 Neighborhood 分组中位数填充
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# Electrical: 众数填充（仅1个缺失）
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
```

### 步骤 2: 数据类型转换

```python
# MSSubClass 转为分类变量
df['MSSubClass'] = df['MSSubClass'].astype(str)

# 确保年份为整数（已填充的GarageYrBlt为0表示无车库）
df['GarageYrBlt'] = df['GarageYrBlt'].astype(int)
```

### 步骤 3: 异常值处理

```python
# 检查并标记潜在异常值
# LotArea 异常大值（基于IQR方法）
Q1 = df['LotArea'].quantile(0.25)
Q3 = df['LotArea'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 3 * IQR  # 使用3倍IQR作为极端异常值标准

print(f"LotArea > {outlier_threshold:.0f} 的异常值数量: {(df['LotArea'] > outlier_threshold).sum()}")

# GrLivArea 与 SalePrice 的联合异常值（基于业务逻辑）
# 通常删除 GrLivArea > 4000 且 SalePrice 较低的极端离群点
```

### 步骤 4: 特征工程（推荐）

```python
# 1. 房龄相关特征
current_year = 2011  # 数据集时间范围
df['HouseAge'] = current_year - df['YearBuilt']
df['RemodAge'] = current_year - df['YearRemodAdd']
df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)

# 2. 总面积特征
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + 
                      df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

# 3. 浴室总数
df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                   df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])

# 4. 是否有特定设施的二元特征
df['HasPool'] = (df['PoolArea'] > 0).astype(int)
df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
```

### 步骤 5: 分类变量编码准备

```python
# 有序分类变量映射（基于业务逻辑）
quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

for col in ordinal_cols:
    df[col] = df[col].map(quality_map).fillna(0).astype(int)

# 其他有序变量
df['BsmtExposure'] = df['BsmtExposure'].map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
df['Functional'] = df['Functional'].map({'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 
                                         'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8})
```

---

## 3. 完整清洗代码

```python
import pandas as pd
import numpy as np

def clean_housing_data(file_path):
    """
    清洗Ames Housing数据集
    """
    # 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # ===== 步骤1: 处理缺失值 =====
    
    # 1.1 "不存在"类型 → None/0
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType']
    
    for col in none_cols:
        df[col] = df[col].fillna('None')
    
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # 1.2 基于统计填充
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    # ===== 步骤2: 数据类型转换 =====
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    df['GarageYrBlt'] = df['GarageYrBlt'].astype(int)
    
    # ===== 步骤3: 特征工程 =====
    current_year = df['YrSold'].max()
    
    # 房龄特征
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)
    df['HasRemod'] = (df['YearRemodAdd'] > df['YearBuilt']).astype(int)
    
    # 面积聚合
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + 
                          df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])
    
    # 浴室总数
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                       df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    
    # 设施存在性
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['HasFence'] = (df['Fence'] != 'None').astype(int)
    
    # ===== 步骤4: 有序编码 =====
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    
    for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
        df[col] = df[col].map(quality_map).fillna(0).astype(int)
    
    # 其他有序映射
    df['BsmtExposure'] = df['BsmtExposure'].map(
        {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    ).fillna(0).astype(int)
    
    df['Functional'] = df['Functional'].map(
        {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}
    )
    
    print(f"清洗后数据形状: {df.shape}")
    print(f"剩余缺失值总数: {df.isnull().sum().sum()}")
    
    return df

# 执行清洗
df_cleaned = clean_housing_data('/Users/cjialin/code/AutoMLByLLM/train.csv')
```

---

## 4. 预期效果

| 指标 | 清洗前 | 清洗后 | 改善 |
|------|--------|--------|------|
| **缺失值总数** | ~6,800个 | **0个** | ✅ 完全消除 |
| **特征数量** | 81列 | ~95列 | ⬆️ 增加工程特征 |
| **数据完整性** | 91.5% | **100%