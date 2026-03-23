# 🏠 房价预测数据清洗方案

## 一、方案概述

**数据文件**: `/Users/cjialin/code/AutoMLByLLM/train.csv`  
**数据形状**: 1460 行 × 81 列  
**任务类型**: 房价预测（回归任务）  
**目标变量**: `SalePrice`  
**评估指标**: RMSE（均方根误差）

---

## 二、数据质量问题汇总

| 问题类型 | 数量 | 主要影响列 |
|---------|------|-----------|
| 高缺失率列（>50%） | 5 列 | PoolQC, MiscFeature, Alley, Fence, MasVnrType |
| 中缺失率列（5%-50%） | 7 列 | FireplaceQu, LotFrontage, Garage相关列 |
| 低缺失率列（<5%） | 7 列 | Bsmt相关列, MasVnrArea, Electrical |
| 异常值列 | 21 列 | LotArea, GrLivArea, SalePrice 等 |
| 高异常值比例列（>10%） | 2 列 | BsmtFinSF2, EnclosedPorch |

---

## 三、详细清洗步骤

### 步骤 1: 高缺失率列删除

**策略**: 删除缺失率超过 50% 的列，这些列信息含量过低，填充可能引入噪声。

```python
import pandas as pd
import numpy as np
from scipy.stats import mstats

# 加载数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 步骤 1: 删除高缺失率列（>50%）
high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=high_missing_cols)
print(f"删除高缺失率列后形状: {df.shape}")
```

**处理列说明**:
| 列名 | 缺失率 | 删除原因 |
|-----|-------|---------|
| PoolQC | 99.52% | 泳池质量，绝大多数房屋无泳池 |
| MiscFeature | 96.3% | 其他杂项特征，信息稀疏 |
| Alley | 93.77% | 小巷通道类型，极少房屋有小巷 |
| Fence | 80.75% | 围栏质量，大部分无围栏 |
| MasVnrType | 59.73% | 砌体饰面类型，缺失过多 |

---

### 步骤 2: 异常值列处理

**策略 2.1**: 删除异常值比例过高（>10%）且业务意义不大的列

```python
# 步骤 2.1: 删除高异常值比例列
high_outlier_cols = ['BsmtFinSF2', 'EnclosedPorch']
df = df.drop(columns=high_outlier_cols)
print(f"删除高异常值列后形状: {df.shape}")
```

**策略 2.2**: 对数值型特征进行 Winsorize 处理（缩尾至 1%-99%）

```python
# 步骤 2.2: Winsorize 处理异常值
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
]

for col in winsorize_cols:
    if col in df.columns:
        df[col] = mstats.winsorize(df[col], limits=[0.01, 0.01])
```

**Winsorize 处理列说明**:

| 列名 | 异常值比例 | 业务含义 | 处理方式 |
|-----|----------|---------|---------|
| LotArea | 4.73% | 地块面积 | 缩尾至 1%-99% |
| GrLivArea | 2.12% | 地上居住面积 | 缩尾至 1%-99% |
| SalePrice | 4.18% | 销售价格（目标变量） | 缩尾至 1%-99% |
| OverallCond | 8.56% | 整体状况评分 | 缩尾至 1%-99% |
| GarageArea | 1.44% | 车库面积 | 缩尾至 1%-99% |

---

### 步骤 3: 缺失值填充

**策略 3.1**: 类别型变量 - 使用 "None" 或众数填充

```python
# 步骤 3.1: 类别型变量缺失值填充
#  basement 相关特征（无地下室设为"None"）
bsmt_cat_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
for col in bsmt_cat_cols:
    df[col] = df[col].fillna('None')

# garage 相关特征（无车库设为"None"）
garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cat_cols:
    df[col] = df[col].fillna('None')

# 其他类别型变量
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')  # 无壁炉
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])  # 用电系统使用众数
```

**策略 3.2**: 数值型变量 - 使用 0 或中位数填充

```python
# 步骤 3.2: 数值型变量缺失值填充
# LotFrontage - 使用按邻域分组的中位数填充（同街区房屋临街距离相似）
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
# 若仍有缺失，使用整体中位数
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# MasVnrArea - 无砌体饰面设为 0
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# GarageYrBlt - 无车库设为 0 或与房屋建造年份相同
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
```

---

### 步骤 4: 数据类型优化

**策略**: 将类别型变量从 object 转换为 category 类型，优化内存和模型性能

```python
# 步骤 4: 数据类型转换
# 有序分类变量（可编码为数值）
ordinal_cols = [
    'LotShape', 'LandSlope', 'OverallQual', 'OverallCond',
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu',
    'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive'
]

# 名义分类变量
nominal_cols = [
    'MSZoning', 'Street', 'LandContour', 'Utilities',
    'LotConfig', 'Condition1', 'Condition2', 'BldgType',
    'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'Foundation', 'Heating', 'CentralAir',
    'Electrical', 'GarageType', 'SaleType', 'SaleCondition'
]

# 转换为 category 类型
for col in ordinal_cols + nominal_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

print(f"数值列: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"类别列: {df.select_dtypes(include=['category']).shape[1]}")
```

---

### 步骤 5: 特征工程（房价预测专用）

**策略**: 基于业务理解创建新特征，提升预测性能

```python
# 步骤 5: 特征工程
# 5.1 总面积特征（房屋总面积是房价的重要预测因子）
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# 5.2 总面积（包含车库、门廊等）
df['TotalArea'] = (
    df['TotalSF'] + df['GarageArea'] + df['WoodDeckSF'] + 
    df['OpenPorchSF'] + df['EnclosedPorch'] if 'EnclosedPorch' in df.columns else 0 +
    df['3SsnPorch'] + df['ScreenPorch']
)

# 5.3 房屋年龄和翻新年份
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)

# 5.4 浴室总数
df['TotalBath'] = (
    df['FullBath'] + 0.5 * df['HalfBath'] +
    df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
)

# 5.5 质量综合评分
quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
df['OverallQualScore'] = (
    df['ExterQual'].map(quality_map) + 
    df['KitchenQual'].map(quality_map) +
    df['BsmtQual'].map(quality_map).fillna(0)
)

# 5.6 每平方英尺价格（用于验证，实际预测时不需要）
# df['PricePerSF'] = df['SalePrice'] / df['GrLivArea']
```

---

### 步骤 6: 最终验证

```python
# 步骤 6: 最终验证
print("=" * 50)
print("清洗后数据质量报告")
print("=" * 50)
print(f"数据形状: {df.shape}")
print(f"\n剩余缺失值:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\n数据类型分布:\n{df.dtypes.value_counts()}")

# 保存清洗后的数据
df.to_csv('/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv', index=False)
print("\n清洗完成！数据已保存至 train_cleaned.csv")
```

---

## 四、完整清洗代码

```python
"""
房价预测数据清洗脚本
数据路径: /Users/cjialin/code/AutoMLByLLM/train.csv
任务: 预测 SalePrice
"""

import pandas as pd
import numpy as np
from scipy.stats import mstats

def clean_housing_data(file_path):
    """
    清洗房价数据
    """
    # 1. 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # 2. 删除高缺失率列（>50%）
    high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df = df.drop(columns=high_missing_cols)
    
    # 3. 删除高异常值比例列（>10%）
    high_outlier_cols = ['BsmtFinSF2', 'EnclosedPorch']
    df = df.drop(columns=high_outlier_cols, errors='ignore')
    
    # 4. Winsorize 数值型异常值
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
        'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
    ]
    for col in winsorize_cols:
        if col in df.columns:
            df[col] = mstats.winsorize(df[col].astype(float), limits=[0.01, 0.01])
    
    # 5. 填充缺失值 - 类别型
    bsmt_cat_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
    for col in bsmt_cat_cols:
        df[col] = df[col].fillna('None')
    
    garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cat_cols:
        df[col] = df[col].fillna('None')
    
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    # 6. 填充缺失值 - 数值型
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    
    # 7. 特征工程
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)
    df['TotalBath'] = (
        df['FullBath'] + 0.5 * df['HalfBath'] +
        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    )
    
    # 8. 数据类型优化
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].astype('category')
    
    print(f"清洗后数据形状: {df.shape}")
    print(f"剩余缺失值总数: {df.isnull().sum().sum()}")
    
    return df

# 执行清洗
if __name__ == "__main__":
    df_cleaned = clean_housing_data('/Users/cjialin/code/AutoMLByLLM/train.csv')
    df_cleaned.to_csv('/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv', index=False)
```

---

## 五、清洗前后对比

| 指标 | 清洗前 | 清洗后 |
|-----|-------|-------|
| 行数 | 1460 | 1460