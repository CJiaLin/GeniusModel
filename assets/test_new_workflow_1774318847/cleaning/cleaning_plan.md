# 房价预测数据清洗方案

## 1. 数据质量概览

| 指标 | 数值 |
|------|------|
| 数据形状 | 1460行 × 81列 |
| 数值列 | 38个 |
| 分类列 | 43个 |
| 缺失值列 | 19列存在缺失 |
| 重复行 | 0行 |
| 目标变量 | SalePrice |

---

## 2. 清洗策略设计

### 2.1 高缺失率列删除（>50%）

根据房价预测的业务理解，以下列缺失率过高，信息不足，直接删除：

| 列名 | 缺失率 | 删除原因 |
|------|--------|----------|
| PoolQC | 99.52% | 绝大多数房屋无泳池 |
| MiscFeature | 96.30% | 杂项功能，缺失过多 |
| Alley | 93.77% | 巷道类型，非主要因素 |
| Fence | 80.75% | 围栏质量，缺失过多 |
| MasVnrType | 59.73% | 砌体饰面类型，高缺失 |

### 2.2 中等缺失率列填充（5%-50%）

| 列名 | 缺失率 | 填充策略 | 理由 |
|------|--------|----------|------|
| FireplaceQu | 47.26% | "NoFireplace" | 缺失表示无壁炉 |
| LotFrontage | 17.74% | 按Neighborhood中位数 | 同社区地块特征相似 |
| GarageType | 5.55% | "NoGarage" | 缺失表示无车库 |
| GarageYrBlt | 5.55% | 房屋建造年份 | 无车库时设为YearBuilt |
| GarageFinish | 5.55% | "NoGarage" | 缺失表示无车库 |
| GarageQual | 5.55% | "NoGarage" | 缺失表示无车库 |
| GarageCond | 5.55% | "NoGarage" | 缺失表示无车库 |

### 2.3 低缺失率列填充（<5%）

| 列名 | 缺失率 | 填充策略 |
|------|--------|----------|
| BsmtExposure | 2.60% | "NoBsmt" |
| BsmtFinType2 | 2.60% | "NoBsmt" |
| BsmtQual | 2.53% | "NoBsmt" |
| BsmtCond | 2.53% | "NoBsmt" |
| BsmtFinType1 | 2.53% | "NoBsmt" |
| MasVnrArea | 0.55% | 0 |
| Electrical | 0.07% | 众数"SBrkr" |

### 2.4 异常值处理策略

#### 删除异常列（零方差或接近零方差）

| 列名 | 异常比例 | 处理方式 |
|------|----------|----------|
| BsmtFinSF2 | 11.44% | 删除列（绝大多数为0） |
| EnclosedPorch | 14.25% | 删除列（绝大多数为0） |

#### Winsorize处理（保留信息但限制极端值）

| 列名 | 正常范围 | 处理方法 |
|------|----------|----------|
| MSSubClass | [20, 190] | 截断至1%-99%分位数 |
| LotFrontage | [21, 141] | 截断至1%-99%分位数 |
| LotArea | [1300, 21500] | 截断至1%-99%分位数 |
| OverallCond | [2, 9] | 截断至1%-99%分位数 |
| MasVnrArea | [0, 800] | 截断上限至99%分位数 |
| BsmtUnfSF | [0, 2000] | 截断至1%-99%分位数 |
| TotalBsmtSF | [0, 2150] | 截断至1%-99%分位数 |
| 1stFlrSF | [300, 2150] | 截断至1%-99%分位数 |
| LowQualFinSF | 删除 | 绝大多数为0，信息价值低 |
| GrLivArea | [400, 2800] | 截断至1%-99%分位数 |
| BsmtHalfBath | 删除 | 变异系数极低 |
| BedroomAbvGr | [0, 5] | 截断至99%分位数 |
| KitchenAbvGr | 删除 | 99%为1，无区分度 |
| TotRmsAbvGrd | [3, 11] | 截断至1%-99%分位数 |
| GarageArea | [0, 1000] | 截断至1%-99%分位数 |
| WoodDeckSF | [0, 600] | 截断上限至99%分位数 |
| OpenPorchSF | [0, 300] | 截断上限至99%分位数 |
| 3SsnPorch | 删除 | 绝大多数为0 |
| ScreenPorch | [0, 400] | 截断上限至99%分位数 |
| MiscVal | 删除 | 绝大多数为0 |
| SalePrice | [35000, 400000] | 截断至1%-99%分位数（目标变量特殊处理）|

**注**：SalePrice作为目标变量，若使用对数变换可缓解异常值影响，考虑保留原始分布或进行对数变换。

### 2.5 数据类型转换

将以下43个分类列转换为`category`类型以节省内存并确保正确编码：

- MSZoning, Street, Alley, LotShape, LandContour, Utilities, LotConfig
- LandSlope, Neighborhood, Condition1, Condition2, BldgType, HouseStyle
- RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, ExterQual
- ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1
- BsmtFinType2, Heating, HeatingQC, CentralAir, Electrical, KitchenQual
- Functional, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond
- PavedDrive, PoolQC, Fence, MiscFeature, SaleType, SaleCondition

---

## 3. 特征工程建议（房价预测专项）

### 3.1 面积特征组合
```python
# 创建总面积特征
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['ScreenPorch']
```

### 3.2 质量评分组合
```python
# 综合质量评分
df['OverallScore'] = df['OverallQual'] * df['OverallCond']
```

### 3.3 房屋年龄特征
```python
# 房龄和翻新状态
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)
```

### 3.4 缺失值指示特征
为高缺失率但已删除的列创建指示变量可能保留信息：
```python
df['HasPool'] = (df['PoolArea'] > 0).astype(int)
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
```

---

## 4. 完整清洗代码

```python
"""
房价预测数据清洗脚本
数据路径: /Users/cjialin/code/AutoMLByLLM/train.csv
目标: 为房价预测模型准备高质量数据
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """加载数据"""
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    return df

def remove_high_missing_columns(df):
    """删除高缺失率列（>50%）"""
    high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df = df.drop(columns=high_missing_cols, errors='ignore')
    print(f"删除高缺失列后: {df.shape}")
    return df

def remove_low_variance_columns(df):
    """删除低方差列（接近零方差）"""
    low_variance_cols = ['BsmtFinSF2', 'EnclosedPorch', 'LowQualFinSF', 
                         'BsmtHalfBath', 'KitchenAbvGr', '3SsnPorch', 'MiscVal']
    df = df.drop(columns=low_variance_cols, errors='ignore')
    print(f"删除低方差列后: {df.shape}")
    return df

def fill_missing_values(df):
    """填充缺失值"""
    
    # 分类变量：表示"无此特征"的缺失
    none_cols = ['FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # GarageYrBlt: 无车库时填充为房屋建造年份
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    
    # LotFrontage: 按Neighborhood分组填充中位数
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 若仍有缺失，用全局中位数填充
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # MasVnrArea: 缺失视为无砌体饰面，填充0
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # Electrical: 用众数填充
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    # 确保没有剩余缺失值
    missing_after = df.isnull().sum().sum()
    print(f"填充后剩余缺失值: {missing_after}")
    
    return df

def winsorize_outliers(df):
    """Winsorize处理异常值（限制在1%-99%分位数）"""
    
    # 需要Winsorize的数值列
    winsorize_cols = {
        'MSSubClass': (0.01, 0.99),
        'LotFrontage': (0.01, 0.99),
        'LotArea': (0.01, 0.99),
        'OverallCond': (0.01, 0.99),
        'MasVnrArea': (0, 0.99),
        'BsmtUnfSF': (0.01, 0.99),
        'TotalBsmtSF': (0.01, 0.99),
        '1stFlrSF': (0.01, 0.99),
        'GrLivArea': (0.01, 0.99),
        'TotRmsAbvGrd': (0.01, 0.99),
        'GarageArea': (0.01, 0.99),
        'WoodDeckSF': (0, 0.99),
        'OpenPorchSF': (0, 0.99),
        'ScreenPorch': (0, 0.99),
        'BedroomAbvGr': (0, 0.99),
        'SalePrice': (0.01, 0.99)  # 目标变量
    }
    
    for col, (lower, upper) in winsorize_cols.items():
        if col in df.columns:
            lower_val = df[col].quantile(lower)
            upper_val = df[col].quantile(upper)
            df[col] = df[col].clip(lower=lower_val, upper=upper_val)
    
    print("Winsorize处理完成")
    return df

def create_features(df):
    """创建新特征"""
    
    # 总面积
    df['TotalSF'] = df['