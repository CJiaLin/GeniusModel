```markdown
# 房价预测数据清洗方案

## 1. 项目概述

**任务类型**: 房价预测（回归任务）  
**目标变量**: `SalePrice`  
**评估指标**: RMSE（均方根误差）  
**数据规模**: 1460 行 × 81 列  
**数据路径**: `/Users/cjialin/code/AutoMLByLLM/train.csv`

---

## 2. 数据质量问题汇总

### 2.1 缺失值分布
| 类别 | 列名 | 缺失比例 | 处理策略 |
|------|------|----------|----------|
| **极高缺失** (>80%) | PoolQC, MiscFeature, Alley, Fence, MasVnrType | 59%-99.5% | **删除整列** |
| **中等缺失** (5%-50%) | FireplaceQu, LotFrontage, GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond | 5.5%-47.3% | **业务逻辑填充** |
| **低缺失** (<3%) | BsmtExposure, BsmtFinType2, BsmtQual, BsmtCond, BsmtFinType1, MasVnrArea, Electrical | 0.07%-2.6% | **众数/中位数填充** |

### 2.2 异常值处理策略
| 处理类型 | 列名 | 异常值比例 | 处理方式 |
|----------|------|------------|----------|
| **删除整列** | BsmtFinSF2, EnclosedPorch | 11.4%, 14.3% | 零值占比过高，信息量少 |
| **Winsorize限幅** | MSSubClass, LotFrontage, LotArea, OverallCond, MasVnrArea, BsmtUnfSF, TotalBsmtSF, 1stFlrSF, LowQualFinSF, GrLivArea, BsmtHalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, GarageArea, WoodDeckSF, OpenPorchSF, 3SsnPorch, ScreenPorch, MiscVal, SalePrice | 1.4%-8.6% | 上下限5%-95%分位数截断 |
| **保留原值** | OverallQual, YearBuilt, BsmtFinSF1, 2ndFlrSF, BsmtFullBath, Fireplaces, GarageCars, PoolArea | 0.07%-0.48% | 业务合理或比例极低 |

---

## 3. 详细清洗流程

### 3.1 数据加载与初始设置

```python
import pandas as pd
import numpy as np
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')

# 加载数据
file_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
df = pd.read_csv(file_path)

print(f"原始数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
```

### 3.2 高缺失率列删除（信息增益低）

```python
# 缺失率超过50%的列直接删除
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=cols_to_drop)
print(f"删除高缺失列后形状: {df.shape}")
```

### 3.3 缺失值智能填充（基于业务理解）

#### 3.3.1 地下室相关特征（缺失表示无地下室）

```python
# 地下室相关列：缺失表示没有地下室，填为"None"或0
bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
bsmt_num_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

for col in bsmt_cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

for col in bsmt_num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)
```

#### 3.3.2 车库相关特征（缺失表示无车库）

```python
# 车库相关列：缺失表示没有车库
garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
garage_num_cols = ['GarageYrBlt', 'GarageCars', 'GarageArea']

for col in garage_cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

# GarageYrBlt用建筑年份填充（无车库时与房屋建造年份一致或设为0）
if 'GarageYrBlt' in df.columns:
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

for col in ['GarageCars', 'GarageArea']:
    if col in df.columns:
        df[col] = df[col].fillna(0)
```

#### 3.3.3 其他特征填充

```python
# LotFrontage：用同区域(Neighborhood)的中位数填充
if 'LotFrontage' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    # 如果仍有缺失，用全局中位数填充
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# FireplaceQu：缺失表示无壁炉
if 'FireplaceQu' in df.columns:
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

# MasVnrArea：缺失填0（无砌体饰面面积）
if 'MasVnrArea' in df.columns:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# Electrical：唯一缺失值用众数填充
if 'Electrical' in df.columns:
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

print(f"缺失值填充后总缺失数: {df.isnull().sum().sum()}")
```

### 3.4 异常值处理

#### 3.4.1 删除低信息列（零值占比过高）

```python
# 删除零值占比过高的列
cols_drop_outlier = ['BsmtFinSF2', 'EnclosedPorch']
df = df.drop(columns=[col for col in cols_drop_outlier if col in df.columns])
```

#### 3.4.2 Winsorize限幅处理（保留5%-95%分位数范围）

```python
# 需要Winsorize的数值列（基于数据质量报告）
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr', 
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF', 
    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
]

# 对每个列进行Winsorize处理（限制在5%-95%分位数）
for col in winsorize_cols:
    if col in df.columns and df[col].dtype in ['int64', 'float64']:
        lower = df[col].quantile(0.05)
        upper = df[col].quantile(0.95)
        df[col] = np.clip(df[col], lower, upper)

print(f"异常值处理后形状: {df.shape}")
```

### 3.5 数据类型优化（分类变量转换）

```python
# 根据数据质量报告，以下列应转换为category类型
cat_cols = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType',
    'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond',
    'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
    'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'
]

# 只转换存在的列
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

print(f"分类变量转换完成，内存使用优化")
```

### 3.6 房价预测特征工程（针对RMSE优化）

```python
# 创建新的衍生特征（提升预测性能）

# 1. 总面积特征（地下室+地上+车库+其他）
if all(col in df.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea']):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GarageArea']

# 2. 总浴室数（全浴室+0.5*半浴室）
if all(col in df.columns for col in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
    df['TotalBathrooms'] = (
        df['FullBath'] + 0.5 * df['HalfBath'] + 
        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    )

# 3. 房龄和翻新年龄
if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
if 'YearRemodAdd' in df.columns and 'YrSold' in df.columns:
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# 4. 是否有泳池（PoolArea > 0）
if 'PoolArea' in df.columns:
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)

# 5. 是否有2层
if '2ndFlrSF' in df.columns:
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)

# 6. 车库质量评分（将GarageQual转为数值）
qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
if 'GarageQual' in df.columns:
    df['GarageQualScore'] = df['GarageQual'].map(qual_map).fillna(0)

# 7. 地下室质量评分
if 'BsmtQual' in df.columns:
    df['BsmtQualScore'] = df['BsmtQual'].map(qual_map).fillna(0)

print(f"特征工程后列数: {df.shape[1]}")
```

### 3.7 数据验证与保存

```python
# 验证清洗结果
print("=== 数据清洗验证报告 ===")
print(f"最终数据形状: {df.shape}")
print(f"剩余缺失值总数: {df.isnull().sum().sum()}")
print(f"数值列数量: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"分类列数量: {len(df.select_dtypes(include=['category']).columns)}")

# 检查目标变量SalePrice的统计信息
if 'SalePrice' in df.columns:
    print(f"\n目标变量SalePrice统计:")
    print(df['SalePrice'].describe())

# 保存清洗后的数据
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n清洗后数据已保存至: {output_path}")
```

---

## 4. 清洗方案总结

### 4.1 执行清单

| 步骤 | 操作 | 影响列数 | 说明 |
|------|------|----------|------|
| 1 | 删除高缺失列 | 5列 | PoolQC, MiscFeature, Alley, Fence, MasVnrType |
| 2 | 删除低信息列 | 2列 | BsmtFinSF2, EnclosedPorch（零值占比>10%） |
| 3 | 缺失值填充 | 17列 | 按业务逻辑分组填充 |
| 4 | 异常值Winsorize | 21列 | 5%-95%分位数限幅 |
| 5 | 类型转换 | 35列 | 转为category类型优化内存 |
| 6 | 特征工程 | +7列 | 总面积、总浴室、房龄等衍生特征 |

### 4.2 针对房价预测的特殊考虑

1. **对数变换准备**: `SalePrice`通常右偏，建议在建模前进行`log1p`变换以降低RMSE
2. **特征交互**: 质量评分（OverallQual）与面积类特征的交互对房价影响显著
3. **时间特征**: YearBuilt和YrSold的差值（房龄）比绝对年份更有预测力
4. **类别编码**: 有序分类变量（如ExterQual, BsmtQual）已映射为数值评分，无序分类变量保留为category待One-Hot编码

### 4.3 验证指标

- **缺失值**: 清洗后应为0
- **异常值**: 保留合理范围内的极端值（Winsorize后）
- **数据类型**: 数值型38列 → 保留关键数值特征，分类型转为category
- **行数**: 保持1460行（无重复行删除）

---

## 5. 后续建模建议

1. **特征