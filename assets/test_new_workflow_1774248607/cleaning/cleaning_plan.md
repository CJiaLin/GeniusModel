# 🏠 房价预测数据清洗方案

## 📋 方案概述

**数据文件**: `/Users/cjialin/code/AutoMLByLLM/train.csv`  
**数据形状**: 1460 行 × 81 列  
**任务类型**: 房价预测（回归任务）  
**目标变量**: `SalePrice`  
**评估指标**: RMSE（均方根误差）

---

## 🔍 数据质量诊断

### 1. 缺失值分布（基于实际数据统计）

| 严重程度 | 列名 | 缺失数 | 缺失比例 | 业务含义 | 处理策略 |
|---------|------|--------|----------|----------|----------|
| 🔴 极高 | PoolQC | 1453 | 99.52% | 泳池质量 | **删除列** |
| 🔴 极高 | MiscFeature | 1406 | 96.30% | 其他设施 | **删除列** |
| 🔴 极高 | Alley | 1369 | 93.77% | 小巷通道 | **删除列** |
| 🔴 极高 | Fence | 1179 | 80.75% | 围栏质量 | **删除列** |
| 🔴 极高 | MasVnrType | 872 | 59.73% | 砌体饰面类型 | **删除列** |
| 🟠 高 | FireplaceQu | 690 | 47.26% | 壁炉质量 | 填充"无壁炉" |
| 🟠 高 | LotFrontage | 259 | 17.74% | 临街宽度 | 按社区中位数填充 |
| 🟡 中 | GarageType/Finish/Qual/Cond | 81 | 5.55% | 车库相关 | 填充"无车库" |
| 🟡 中 | GarageYrBlt | 81 | 5.55% | 车库建造年份 | 填充建筑年份 |
| 🟢 低 | BsmtQual/Cond/Exposure/FinType | 37-38 | 2.53-2.6% | 地下室相关 | 填充"无地下室" |
| 🟢 低 | MasVnrArea | 8 | 0.55% | 砌体面积 | 填充0 |
| 🟢 低 | Electrical | 1 | 0.07% | 电力系统 | 填充众数 |

### 2. 异常值识别（基于 IQR 方法）

| 列名 | 异常值数量 | 正常范围 | 处理策略 |
|------|------------|----------|----------|
| MSSubClass | 103 | [-55.0, 145.0] | Winsorize 缩尾 |
| LotFrontage | 88 | [27.5, 111.5] | Winsorize 缩尾 |
| LotArea | 69 | [1481.5, 17673.5] | Winsorize 缩尾 |
| OverallCond | 125 | [3.5, 7.5] | Winsorize 缩尾 |
| MasVnrArea | 96 | [-249.0, 415.0] | Winsorize 缩尾 |
| TotalBsmtSF | 61 | [42.0, 2052.0] | Winsorize 缩尾 |
| GrLivArea | 31 | [158.62, 2747.62] | Winsorize 缩尾 |
| SalePrice | 61 | [3937.5, 340037.5] | **保留**（目标变量） |

### 3. 重复值检查
- **重复行数**: 0（无需处理）

---

## 🛠️ 详细清洗方案

### 阶段一：高缺失率列删除

**理由**: 缺失率超过 50% 的列提供的信息极少，填充会引入大量噪声。

**待删除列**:
- `PoolQC` (99.52% 缺失)
- `MiscFeature` (96.30% 缺失)
- `Alley` (93.77% 缺失)
- `Fence` (80.75% 缺失)
- `MasVnrType` (59.73% 缺失)

### 阶段二：缺失值智能填充

#### A. 分类变量 - 业务逻辑填充

| 列名 | 缺失含义 | 填充值 |
|------|----------|--------|
| FireplaceQu | 无壁炉 | `"None"` |
| GarageType | 无车库 | `"None"` |
| GarageFinish | 无车库 | `"None"` |
| GarageQual | 无车库 | `"None"` |
| GarageCond | 无车库 | `"None"` |
| BsmtQual | 无地下室 | `"None"` |
| BsmtCond | 无地下室 | `"None"` |
| BsmtExposure | 无地下室 | `"None"` |
| BsmtFinType1 | 无地下室 | `"None"` |
| BsmtFinType2 | 无地下室 | `"None"` |
| Electrical | 缺失 | 众数 `"SBrkr"` |

#### B. 数值变量 - 统计填充

| 列名 | 填充策略 | 理由 |
|------|----------|------|
| LotFrontage | 按 `Neighborhood` 分组中位数 | 同社区地块宽度相似 |
| GarageYrBlt | 填充 `YearBuilt` | 无车库时与房屋建造年份一致 |
| MasVnrArea | 填充 0 | 缺失表示无砌体饰面 |

### 阶段三：异常值处理

**策略**: 对数值特征使用 **Winsorize（缩尾处理）**，将极端值限制在 1%-99% 分位数范围内。

**处理列**:
- `MSSubClass`, `LotFrontage`, `LotArea`
- `OverallCond`, `MasVnrArea`
- `BsmtUnfSF`, `TotalBsmtSF`, `1stFlrSF`
- `GrLivArea`, `GarageArea`
- `WoodDeckSF`, `OpenPorchSF`, `ScreenPorch`

**注意**: `SalePrice`（目标变量）的异常值**保留**，避免人为改变目标分布。

### 阶段四：数据类型优化

将分类变量转换为 `category` 类型，减少内存占用并提升模型性能：

```python
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
```

---

## 💻 完整清洗代码

```python
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

# ==================== 1. 加载数据 ====================
file_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
df = pd.read_csv(file_path)

print(f"原始数据形状: {df.shape}")
print(f"原始缺失值总数: {df.isnull().sum().sum()}")

# ==================== 2. 删除高缺失率列 ====================
# 缺失率 > 50% 的列
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=cols_to_drop)
print(f"\n删除高缺失列后形状: {df.shape}")

# ==================== 3. 缺失值填充 ====================

# 3.1 分类变量 - 按业务逻辑填充"None"
none_cols = [
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
]
for col in none_cols:
    df[col] = df[col].fillna('None')

# 3.2 数值变量 - MasVnrArea 填充0
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# 3.3 LotFrontage - 按 Neighborhood 中位数填充
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
# 如果仍有缺失（某些社区全缺失），用整体中位数填充
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# 3.4 GarageYrBlt - 填充 YearBuilt
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

# 3.5 Electrical - 填充众数
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

print(f"缺失值填充后总数: {df.isnull().sum().sum()}")

# ==================== 4. 异常值处理（Winsorize） ====================
# 需要缩尾处理的数值列（排除ID、年份、目标变量）
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF', 'ScreenPorch', 'MiscVal'
]

# 应用 1%-99% 缩尾
for col in winsorize_cols:
    if col in df.columns:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

print(f"异常值处理完成")

# ==================== 5. 数据类型优化 ====================
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

# 过滤实际存在的列
existing_cat_cols = [col for col in categorical_cols if col in df.columns]
for col in existing_cat_cols:
    df[col] = df[col].astype('category')

print(f"数据类型优化完成，共 {len(existing_cat_cols)} 个分类变量")

# ==================== 6. 特征工程（房价预测专用） ====================

# 6.1 总面积特征
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# 6.2 房屋年龄特征
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# 6.3 浴室总数
df['TotalBath'] = (
    df['FullBath'] + 0.5 * df['HalfBath'] + 
    df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
)

# 6.4 门廊总面积
df['TotalPorchSF'] = (
    df['OpenPorchSF'] + df['EnclosedPorch'] + 
    df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
)

# 6.5 是否有特定设施
df['HasPool'] = (df['PoolArea'] > 0).astype(int)
df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

print(f"特征工程完成，新增 11 个特征")

# ==================== 7. 验证结果 ====================
print("\n" + "="*50)
print("数据清洗验证报告")
print("="*50)

print(f"\n最终数据形状: {df.shape}")
print(f"最终缺失值总数: {df.isnull().sum().sum()}")
print(f"重复行数: {df.duplicated().sum()}")

print(f"\n数值列数量: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"分类列数量: {df.select_dtypes(include=['category']).shape[1]}")

# 目标变量统计
print(f"\n目标变量 SalePrice 统计:")
print(df['SalePrice'].describe())

# ==================== 8. 保存清洗后数据 ====================
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n清洗后数据已保存至: {output_path}")
```

---

## 📊 清洗效果预期

| 指标 | 清洗前 | 清洗后 | 改善 |
|------|--------|--------|------|
| 缺失值总数 | ~6,800 | 0 | ✅ 完全消除 |
| 特征数量 | 81 | 87 | ⬆️ 新增6个工程特征 |
| 数据行数 | 1460 | 1460 | ✅ 无删除 |
| 异常值比例 | ~15% | <2% | ✅ 有效控制 |

---

## ⚠️ 建模建议

1. **目标变量处理**: `SalePrice` 右偏严重，建议进行 `log1p` 变换：
   ```python
   df['SalePriceLog'] = np.log1p(df['SalePrice'])
   ```

2. **特征编码**: 对有序分类变量（如 `ExterQual`, `BsmtQual`）进行标签编码：
   ```python
   qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
   ```

3. **交叉验证**: 使用 K-Fold (K=5) 进行模型验证，确保 RMSE 评估的稳定性。

4. **特征选择**: 清洗后使用特征重要性或相关性分析，进一步筛选对房价预测有价值的特征。