# 🏠 房价预测数据清洗方案

## 1. 方案概述

**数据文件**: `/Users/cjialin/code/AutoMLByLLM/train.csv`  
**数据规模**: 1,460 行 × 81 列  
**目标变量**: `SalePrice`  
**任务类型**: 回归任务（RMSE评估）

**核心策略**:
- 删除缺失率>50%的低价值特征（5列）
- 基于业务逻辑填充缺失值（区分"无该设施"vs"数据缺失"）
- Winsorize处理连续型异常值（保护RMSE指标）
- 删除零方差或近零方差特征（2列）
- 优化数据类型减少内存占用

---

## 2. 详细清洗步骤

### 步骤 1: 高缺失率列删除（缺失率>50%）

| 列名 | 缺失率 | 删除原因 |
|------|--------|----------|
| `PoolQC` | 99.52% | 仅7套有泳池，信息不足 |
| `MiscFeature` | 96.30% | 仅54套有杂项特征 |
| `Alley` | 93.77% | 仅91套有巷子通道 |
| `Fence` | 80.75% | 仅281套有围栏 |
| `MasVnrType` | 59.73% | 缺失率过高且难以可靠插补 |

**处理逻辑**: 这些特征缺失率过高，填充会引入过多噪声，对房价预测贡献有限。

### 步骤 2: 缺失值智能填充

#### 2.1 分类变量（表示"无此设施"）

| 列名 | 缺失率 | 填充值 | 业务逻辑 |
|------|--------|--------|----------|
| `FireplaceQu` | 47.26% | `"None"` | 无壁炉 |
| `GarageType` | 5.55% | `"NoGarage"` | 无车库 |
| `GarageFinish` | 5.55% | `"NoGarage"` | 无车库 |
| `GarageQual` | 5.55% | `"NoGarage"` | 无车库 |
| `GarageCond` | 5.55% | `"NoGarage"` | 无车库 |
| `BsmtExposure` | 2.60% | `"NoBasement"` | 无地下室 |
| `BsmtFinType2` | 2.60% | `"NoBasement"` | 无地下室 |
| `BsmtQual` | 2.53% | `"NoBasement"` | 无地下室 |
| `BsmtCond` | 2.53% | `"NoBasement"` | 无地下室 |
| `BsmtFinType1` | 2.53% | `"NoBasement"` | 无地下室 |

#### 2.2 数值变量

| 列名 | 缺失率 | 填充策略 | 说明 |
|------|--------|----------|------|
| `LotFrontage` | 17.74% | 按`Neighborhood`分组中位数 | 街区前长度与社区相关 |
| `GarageYrBlt` | 5.55% | 填充`YearBuilt` | 无车库时设为房屋建造年份 |
| `MasVnrArea` | 0.55% | 填充`0` | 无砌体饰面 |
| `Electrical` | 0.07% | 填充众数`"SBrkr"` | 唯一缺失值用最常见值 |

### 步骤 3: 异常值处理

#### 3.1 删除近零方差列

| 列名 | 异常值比例 | 处理方式 | 原因 |
|------|------------|----------|------|
| `BsmtFinSF2` | 11.44% | **删除列** | 96.6%值为0，无区分度 |
| `EnclosedPorch` | 14.25% | **删除列** | 85.75%值为0，无区分度 |

#### 3.2 Winsorize缩尾处理（5%-95%分位数）

对以下连续变量进行上下限缩尾，减少极端值对RMSE的影响：

**房屋面积相关**:
- `LotFrontage`, `LotArea`, `MasVnrArea`, `BsmtUnfSF`, `TotalBsmtSF`
- `1stFlrSF`, `GrLivArea`, `GarageArea`

**房间数量相关**:
- `MSSubClass`, `OverallCond`, `LowQualFinSF`, `BsmtHalfBath`
- `BedroomAbvGr`, `KitchenAbvGr`, `TotRmsAbvGrd`

**外部设施**:
- `WoodDeckSF`, `OpenPorchSF`, `3SsnPorch`, `ScreenPorch`, `MiscVal`

**目标变量**:
- `SalePrice`（训练集Winsorize，测试集不处理）

**保留的异常值**（业务合理性）:
- `OverallQual`, `YearBuilt`, `BsmtFinSF1`, `2ndFlrSF`, `BsmtFullBath`, `Fireplaces`, `GarageCars` - 这些异常值代表真实的极端房产，具有预测价值。

### 步骤 4: 数据类型优化

将43个object类型分类变量转换为`category`类型，减少内存占用并加速模型训练：

**转换列表**: `MSZoning`, `Street`, `Alley`（如保留）, `LotShape`, `LandContour`, `Utilities`, `LotConfig`, `LandSlope`, `Neighborhood`, `Condition1`, `Condition2`, `BldgType`, `HouseStyle`, `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`, `MasVnrType`（如保留）, `ExterQual`, `ExterCond`, `Foundation`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `Heating`, `HeatingQC`, `CentralAir`, `Electrical`, `KitchenQual`, `Functional`, `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `PavedDrive`, `PoolQC`（如保留）, `Fence`（如保留）, `MiscFeature`（如保留）, `SaleType`, `SaleCondition`

---

## 3. Python清洗代码

```python
import pandas as pd
import numpy as np
from scipy import stats

# ==================== 1. 数据加载 ====================
file_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
df = pd.read_csv(file_path)
print(f"原始数据形状: {df.shape}")

# ==================== 2. 删除高缺失率列 ====================
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 
                'BsmtFinSF2', 'EnclosedPorch']
df = df.drop(columns=cols_to_drop)
print(f"删除高缺失/零方差列后: {df.shape}")

# ==================== 3. 缺失值填充 ====================

# 3.1 分类变量：填充"None"表示无该设施
none_fill_cols = ['FireplaceQu', 'GarageType', 'GarageFinish', 
                  'GarageQual', 'GarageCond']
for col in none_fill_cols:
    df[col] = df[col].fillna('None')

# 3.2 地下室相关：填充"NoBasement"
basement_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 
                 'BsmtCond', 'BsmtFinType1']
for col in basement_cols:
    df[col] = df[col].fillna('NoBasement')

# 3.3 LotFrontage：按Neighborhood分组中位数填充
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())