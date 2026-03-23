# 房价预测数据清洗方案

## 一、方案概述

本方案针对 **Ames Housing 房价预测数据集**（1460条样本，81个特征）设计，目标是通过数据清洗提升房价预测模型（SalePrice）的RMSE表现。

### 清洗策略总览
| 问题类型 | 处理策略 | 涉及列数 |
|---------|---------|---------|
| 高缺失率列 | 删除（>50%） | 5列 |
| 中等缺失率 | 业务逻辑填充 | 12列 |
| 低缺失率 | 统计填充 | 6列 |
| 异常值 | Winsorize处理 | 20列 |
| 零方差/近零方差 | 删除 | 2列 |
| 数据类型优化 | Category转换 | 43列 |

---

## 二、详细清洗步骤

### 步骤1：高缺失率列删除（完整性修复）

**处理依据**：缺失率超过50%的列缺乏足够信息，填充会引入噪声。

| 列名 | 缺失率 | 处理动作 |
|------|--------|---------|
| PoolQC | 99.52% | 删除 |
| MiscFeature | 96.30% | 删除 |
| Alley | 93.77% | 删除 |
| Fence | 80.75% | 删除 |
| MasVnrType | 59.73% | 删除 |

**业务解释**：这些特征代表泳池质量、杂项功能、小巷通道、围栏和砌体饰面类型，大多数房屋不具备这些特征，保留会稀释模型注意力。

---

### 步骤2：缺失值智能填充（完整性修复）

#### 2.1 数值型缺失填充

| 列名 | 缺失率 | 填充策略 | 业务依据 |
|------|--------|---------|---------|
| LotFrontage | 17.74% | 按Neighborhood分组中位数 | 同街区房屋临街距离相似 |
| GarageYrBlt | 5.55% | 用YearBuilt填充 | 无车库时假设与房屋同建 |
| MasVnrArea | 0.55% | 填充0 | 缺失表示无砌体饰面 |

#### 2.2 分类型缺失填充（地下室相关）

| 列名 | 缺失率 | 填充值 | 业务依据 |
|------|--------|-------|---------|
| BsmtQual | 2.53% | 'None' | 缺失=无地下室 |
| BsmtCond | 2.53% | 'None' | 缺失=无地下室 |
| BsmtExposure | 2.60% | 'None' | 缺失=无地下室 |
| BsmtFinType1 | 2.53% | 'None' | 缺失=无地下室 |
| BsmtFinType2 | 2.60% | 'None' | 缺失=无地下室 |

#### 2.3 分类型缺失填充（车库相关）

| 列名 | 缺失率 | 填充值 | 业务依据 |
|------|--------|-------|---------|
| GarageType | 5.55% | 'None' | 缺失=无车库 |
| GarageFinish | 5.55% | 'None' | 缺失=无车库 |
| GarageQual | 5.55% | 'None' | 缺失=无车库 |
| GarageCond | 5.55% | 'None' | 缺失=无车库 |

#### 2.4 其他分类填充

| 列名 | 缺失率 | 填充策略 |
|------|--------|---------|
| FireplaceQu | 47.26% | 'None'（缺失=无壁炉） |
| Electrical | 0.07% | 众数（SBrkr） |

---

### 步骤3：异常值处理（一致性修复）

采用 **Winsorize（缩尾处理）**，将极值限制在1%-99%分位数，保留数据分布形态的同时消除极端异常影响。

| 列名 | 异常比例 | 处理方式 | 下限 | 上限 |
|------|---------|---------|------|------|
| MSSubClass | 7.05% | Winsorize | 1%分位 | 99%分位 |
| LotFrontage | 6.03% | Winsorize | 1%分位 | 99%分位 |
| LotArea | 4.73% | Winsorize | 1%分位 | 99%分位 |
| OverallCond | 8.56% | Winsorize | 1%分位 | 99%分位 |
| MasVnrArea | 6.58% | Winsorize | 1%分位 | 99%分位 |
| BsmtUnfSF | 1.99% | Winsorize | 1%分位 | 99%分位 |
| TotalBsmtSF | 4.18% | Winsorize | 1%分位 | 99%分位 |
| 1stFlrSF | 1.37% | Winsorize | 1%分位 | 99%分位 |
| LowQualFinSF | 1.78% | Winsorize | 1%分位 | 99%分位 |
| GrLivArea | 2.12% | Winsorize | 1%分位 | 99%分位 |
| BsmtHalfBath | 5.62% | Winsorize | 1%分位 | 99%分位 |
| BedroomAbvGr | 2.40% | Winsorize | 1%分位 | 99%分位 |
| KitchenAbvGr | 4.66% | Winsorize | 1%分位 | 99%分位 |
| TotRmsAbvGrd | 2.05% | Winsorize | 1%分位 | 99%分位 |
| GarageArea | 1.44% | Winsorize | 1%分位 | 99%分位 |
| WoodDeckSF | 2.19% | Winsorize | 1%分位 | 99%分位 |
| OpenPorchSF | 5.27% | Winsorize | 1%分位 | 99%分位 |
| 3SsnPorch | 1.64% | Winsorize | 1%分位 | 99%分位 |
| ScreenPorch | 7.95% | Winsorize | 1%分位 | 99%分位 |
| MiscVal | 3.56% | Winsorize | 1%分位 | 99%分位 |
| SalePrice | 4.18% | Winsorize | 1%分位 | 99%分位 |

**注意**：OverallQual、YearBuilt、BsmtFinSF1、2ndFlrSF、BsmtFullBath、Fireplaces、GarageCars、PoolArea 异常比例<1%或属于合理业务范围，予以保留。

---

### 步骤4：零方差特征删除（有效性修复）

| 列名 | 异常比例 | 处理动作 | 原因 |
|------|---------|---------|------|
| BsmtFinSF2 | 11.44% | 删除列 | 91.44%为0，方差极低 |
| EnclosedPorch | 14.25% | 删除列 | 85.75%为0，方差极低 |

---

### 步骤5：数据类型优化（存储与效率）

将43个分类型列转换为 `category` 类型，减少内存占用并提升模型训练效率。

**转换列表**：MSZoning, Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Condition1, Condition2, BldgType, HouseStyle, RoofStyle, RoofMatl, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, Heating, HeatingQC, CentralAir, Electrical, KitchenQual, Functional, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PavedDrive, PoolQC, Fence, MiscFeature, SaleType, SaleCondition 等。

---

### 步骤6：房价预测特征工程（业务增强）

基于房地产领域知识，创建以下衍生特征：

| 新特征 | 计算公式 | 业务含义 |
|--------|---------|---------|
| TotalSF | GrLivArea + TotalBsmtSF | 房屋总使用面积 |
| Total_Bathrooms | BsmtFullBath + 0.5*BsmtHalfBath + FullBath + 0.5*HalfBath | 等效总浴室数 |
| Total_PorchSF | OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch | 总门廊面积 |
| HasPool | PoolArea > 0 | 是否有泳池 |
| Has2ndFloor | 2ndFlrSF > 0 | 是否有二层 |
| HasGarage | GarageArea > 0 | 是否有车库 |
| HasBsmt | TotalBsmtSF > 0 | 是否有地下室 |
| HasFireplace | Fireplaces > 0 | 是否有壁炉 |
| HouseAge | YrSold - YearBuilt | 房龄 |
| RemodAge | YrSold - YearRemodAdd | 翻新后年数 |
| IsNewHouse | YrSold == YearBuilt | 是否新房 |

---

## 三、Python清洗代码

```python
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """加载数据"""
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    return df

def drop_high_missing_columns(df):
    """删除高缺失率列（>50%）"""
    high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df = df.drop(columns=high_missing_cols, errors='ignore')
    print(f"删除高缺失列后形状: {df.shape}")
    return df

def drop_low_variance_columns(df):
    """删除低方差列"""
    low_var_cols = ['BsmtFinSF2', 'EnclosedPorch']
    df = df.drop(columns=low_var_cols, errors='ignore')
    print(f"删除低方差列后形状: {df.shape}")
    return df

def fill_missing_values(df):
    """智能填充缺失值"""
    
    # 数值型填充
    # LotFrontage: 按Neighborhood分组中位数
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # GarageYrBlt: 用YearBuilt填充
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    
    # MasVnrArea: 填充0
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # 分类型填充 - 地下室相关（缺失=无地下室）
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        df[col] = df[col].fillna('None')
    
    # 地下室数值列填充0（如果有的话）
    basement_num_cols = ['BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1']
    for col in basement_num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 分类型填充 - 车库相关（缺失=无车库）
    garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cols:
        df[col] = df[col].fillna('None')
    
    # GarageCars填充0
    if 'GarageCars' in df.columns:
        df['GarageCars'] = df['GarageCars'].fillna(0)
    
    # 分类型填充 - 其他
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    print("缺失值填充完成")
    print(f"剩余缺失值总数: {df.isnull().sum().sum()}")
    return df

def winsorize_outliers(df):
    """对数值列进行Winsorize处理（1%-99%）"""
    
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
        'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal'
    ]
    
    # 对目标变量也进行Winsorize（训练集）
    if 'SalePrice' in df.columns:
        winsorize_cols.append('SalePrice')
    
    for col in winsorize_cols:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
    
    print("Winsorize异常值处理完成")
    return df

def optimize_data_types(df):
    """优化数据类型为category"""
    
    categorical_cols = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond',
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
        'MiscFeature', 'SaleType', 'SaleCondition'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # 特殊处理：MSSubClass实际是分类变量（建筑类型代码）
    df['MSSubClass'] = df['MSSubClass'].astype('category')
    
    print("数据类型优化完成")
    return df

def create_features(df):
    """创建房价预测相关特征"""
    
    # 总面积特征
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
    
    # 总浴室数（全浴室算1，半浴室算0.5）
    df['Total_Bathrooms'] = (df['BsmtFullBath'].fillna(0) + 
                              0.5 * df['BsmtHalfBath'].fillna(0) + 
                              df['FullBath'].fillna(0) + 
                              0.5 * df['HalfBath'].fillna(0))
    
    # 总门廊面积
    df['Total_PorchSF'] = (df['OpenPorchSF'].fillna(0) + 
                           df['EnclosedPorch'].fillna(0) + 
                           df['3SsnPorch'].fillna(0) + 
                           df['ScreenPorch'].fillna(0))
    
    # 二元特征
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    
    # 年龄特征