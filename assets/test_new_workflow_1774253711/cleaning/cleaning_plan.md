# 房价预测数据清洗方案

## 方案概述

**任务背景**：房价预测（回归任务）  
**目标变量**：`SalePrice`  
**评估指标**：RMSE（均方根误差）  
**数据规模**：1460 行 × 81 列  
**清洗策略**：保留信息完整性，处理缺失值与异常值，优化模型性能

---

## 一、缺失值处理方案

### 1.1 删除高缺失率列（缺失率 > 50%）

根据分析，以下5列缺失率过高，信息含量极低，直接删除：

| 列名 | 缺失比例 | 删除原因 |
|------|----------|----------|
| `PoolQC` | 99.52% | 几乎全为缺失 |
| `MiscFeature` | 96.30% | 几乎全为缺失 |
| `Alley` | 93.77% | 几乎全为缺失 |
| `Fence` | 80.75% | 缺失率过高 |
| `MasVnrType` | 59.73% | 缺失率过高 |

**业务逻辑**：这些特征在房价预测中属于稀有设施，缺失表示不存在，但高缺失率导致方差过小，对模型贡献有限。

### 1.2 填充中等缺失率列（5% < 缺失率 < 50%）

| 列名 | 缺失比例 | 填充策略 | 业务逻辑 |
|------|----------|----------|----------|
| `FireplaceQu` | 47.26% | 填充为 `"None"` | 缺失表示无壁炉 |
| `LotFrontage` | 17.74% | 按 `Neighborhood` 分组中位数填充 | 同社区地块临路长度相似 |
| `GarageType` | 5.55% | 填充为 `"None"` | 缺失表示无车库 |
| `GarageYrBlt` | 5.55% | 用 `YearBuilt` 填充 | 无车库时假设与房屋同年建造 |
| `GarageFinish` | 5.55% | 填充为 `"None"` | 缺失表示无车库 |
| `GarageQual` | 5.55% | 填充为 `"None"` | 缺失表示无车库 |
| `GarageCond` | 5.55% | 填充为 `"None"` | 缺失表示无车库 |

### 1.3 填充低缺失率列（缺失率 < 5%）

| 列名 | 缺失比例 | 填充策略 |
|------|----------|----------|
| `BsmtExposure` | 2.60% | 填充为 `"No"`（无暴露） |
| `BsmtFinType2` | 2.60% | 填充为 `"Unf"`（未完工） |
| `BsmtQual` | 2.53% | 填充为 `"TA"`（典型/平均） |
| `BsmtCond` | 2.53% | 填充为 `"TA"`（典型/平均） |
| `BsmtFinType1` | 2.53% | 填充为 `"Unf"`（未完工） |
| `MasVnrArea` | 0.55% | 填充为 0 |
| `Electrical` | 0.07% | 填充为众数 `"SBrkr"` |

---

## 二、异常值处理方案

### 2.1 删除异常值占比过高的列

| 列名 | 异常值比例 | 处理方式 | 原因 |
|------|------------|----------|------|
| `BsmtFinSF2` | 11.44% | 删除列 | 异常值比例过高 |
| `EnclosedPorch` | 14.25% | 删除列 | 异常值比例过高 |

### 2.2 Winsorize 缩尾处理（限制在 1.5×IQR 范围内）

对以下列进行上下限缩尾处理，保留极端值但限制其影响：

| 列名 | 处理方式 |
|------|----------|
| `MSSubClass` | Winsorize (下限: -55.0, 上限: 145.0) |
| `LotFrontage` | Winsorize (下限: 27.5, 上限: 111.5) |
| `LotArea` | Winsorize (下限: 1481.5, 上限: 17673.5) |
| `OverallCond` | Winsorize (下限: 3.5, 上限: 7.5) |
| `MasVnrArea` | Winsorize (下限: 0, 上限: 415.0) |
| `BsmtUnfSF` | Winsorize (下限: 0, 上限: 1685.5) |
| `TotalBsmtSF` | Winsorize (下限: 42.0, 上限: 2052.0) |
| `1stFlrSF` | Winsorize (下限: 118.12, 上限: 2155.12) |
| `LowQualFinSF` | Winsorize (下限: 0, 上限: 0) → 实际不处理 |
| `GrLivArea` | Winsorize (下限: 158.62, 上限: 2747.62) |
| `BsmtHalfBath` | Winsorize |
| `BedroomAbvGr` | Winsorize |
| `KitchenAbvGr` | Winsorize |
| `TotRmsAbvGrd` | Winsorize |
| `GarageArea` | Winsorize |
| `WoodDeckSF` | Winsorize |
| `OpenPorchSF` | Winsorize |
| `3SsnPorch` | Winsorize |
| `ScreenPorch` | Winsorize |
| `MiscVal` | Winsorize |
| `SalePrice` | **不处理**（目标变量保持原始分布） |

**注意**：`SalePrice` 作为目标变量，不进行异常值处理，让模型学习真实的价格分布。

### 2.3 保留的异常值列

以下列的异常值比例极低或具有业务意义，予以保留：

| 列名 | 异常值比例 | 保留原因 |
|------|------------|----------|
| `OverallQual` | 0.14% | 评分等级本身有限 |
| `YearBuilt` | 0.48% | 历史建筑有实际意义 |
| `BsmtFinSF1` | 0.48% | 地下室面积差异正常 |
| `2ndFlrSF` | 0.14% | 两层房面积差异正常 |
| `BsmtFullBath` | 0.07% | 比例极低 |
| `Fireplaces` | 0.34% | 多壁炉豪宅存在 |
| `GarageCars` | 0.34% | 多车位豪宅存在 |
| `PoolArea` | 0.48% | 带泳池豪宅存在 |

---

## 三、数据类型优化

### 3.1 分类变量转换

将以下 43 个 object 类型列转换为 `category` 类型，优化内存和模型处理：

```python
categorical_cols = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
    'SaleType', 'SaleCondition', 'MSSubClass'
]
```

**特殊处理**：`MSSubClass` 虽为数值型，但实际表示建筑类型，应作为分类变量。

---

## 四、特征工程（房价预测专用）

### 4.1 创建衍生特征

| 新特征 | 计算公式 | 业务意义 |
|--------|----------|----------|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | 房屋总使用面积 |
| `TotalBath` | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` | 总浴室当量 |
| `HouseAge` | `YrSold - YearBuilt` | 房屋年龄 |
| `RemodAge` | `YrSold - YearRemodAdd` | 翻新距今时间 |
| `HasPool` | `PoolArea > 0` | 是否有泳池 |
| `Has2ndFloor` | `2ndFlrSF > 0` | 是否有二楼 |
| `HasGarage` | `GarageArea > 0` | 是否有车库 |
| `HasBasement` | `TotalBsmtSF > 0` | 是否有地下室 |
| `HasFireplace` | `Fireplaces > 0` | 是否有壁炉 |

### 4.2 对数变换

房价预测中，面积和价格通常呈右偏分布，进行对数变换：

| 特征 | 变换方式 | 原因 |
|------|----------|------|
| `SalePrice` | `np.log1p` | 目标变量右偏，对数变换后更接近正态 |
| `LotArea` | `np.log1p` | 面积分布右偏 |
| `TotalSF` | `np.log1p` | 总面积右偏 |

---

## 五、数据清洗 Python 代码

```python
import pandas as pd
import numpy as np
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')

def clean_housing_data(file_path):
    """
    房价数据清洗函数
    输入: 原始数据文件路径
    输出: 清洗后的 DataFrame
    """
    
    # 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # 保存 Id 列
    ids = df['Id'].copy() if 'Id' in df.columns else None
    
    # ========== 1. 删除高缺失率列 ==========
    high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df = df.drop(columns=high_missing_cols, errors='ignore')
    print(f"删除高缺失列后: {df.shape}")
    
    # ========== 2. 处理缺失值 ==========
    
    # 2.1 分类变量填充 - 无该设施
    none_fill_cols = ['FireplaceQu', 'GarageType', 'GarageFinish', 
                      'GarageQual', 'GarageCond']
    for col in none_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # 2.2 地下室相关填充
    basement_cols = {
        'BsmtExposure': 'No',
        'BsmtFinType2': 'Unf',
        'BsmtQual': 'TA',
        'BsmtCond': 'TA',
        'BsmtFinType1': 'Unf'
    }
    for col, val in basement_cols.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    
    # 2.3 LotFrontage - 按 Neighborhood 分组中位数填充
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果仍有缺失，用整体中位数填充
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # 2.4 GarageYrBlt - 用 YearBuilt 填充
    if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    
    # 2.5 其他数值列填充
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna('SBrkr')  # 众数
    
    # ========== 3. 删除异常值过多的列 ==========
    df = df.drop(columns=['BsmtFinSF2', 'EnclosedPorch'], errors='ignore')
    
    # ========== 4. 异常值处理（Winsorize）==========
    
    # 需要缩尾处理的数值列（排除目标变量和Id）
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
        'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
        'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        '3SsnPorch', 'ScreenPorch', 'MiscVal'
    ]
    
    # 对训练集进行缩尾（限制在1%和99%分位数）
    for col in winsorize_cols:
        if col in df.columns and col != 'SalePrice':
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
    
    # ========== 5. 特征工程 ==========
    
    # 5.1 创建总面积特征
    if all(col in df.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # 5.2