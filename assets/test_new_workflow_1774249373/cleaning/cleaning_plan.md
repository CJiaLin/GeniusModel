```markdown
# 房价预测数据清洗方案

## 方案概述

**数据文件**: `/Users/cjialin/code/AutoMLByLLM/train.csv`  
**数据形状**: (1460, 81)  
**任务类型**: 房价预测（回归任务）  
**目标变量**: `SalePrice`  
**评估指标**: RMSE

本方案基于数据质量分析报告，针对高缺失率特征、异常值和数据类型问题进行系统性清洗，以提升房价预测模型的性能。

---

## 一、缺失值处理方案

### 1.1 高缺失率特征删除（缺失率 > 50%）

以下特征缺失率过高，信息含量不足，建议直接删除：

| 列名 | 缺失比例 | 删除原因 |
|------|----------|----------|
| PoolQC | 99.52% | 仅7个非缺失值，无预测价值 |
| MiscFeature | 96.30% | 仅54个非缺失值 |
| Alley | 93.77% | 仅91个非缺失值 |
| Fence | 80.75% | 缺失率过高 |
| MasVnrType | 59.73% | 缺失率过高，且与MasVnrArea重复 |

### 1.2 中等缺失率特征填充（缺失率 5%-50%）

```python
# FireplaceQu: 填充为"None"（表示无壁炉）
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

# LotFrontage: 按邻居组（Neighborhood）的中位数填充
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# Garage相关特征（缺失表示无车库）
garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cols:
    df[col] = df[col].fillna('None')

# GarageYrBlt: 无车库的填充0，表示无建造年份
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
```

### 1.3 低缺失率特征填充（缺失率 < 5%）

```python
# Basement相关特征（缺失表示无地下室）
bsmt_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
for col in bsmt_cols:
    df[col] = df[col].fillna('None')

# MasVnrArea: 填充0（无砌体 veneer）
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# Electrical: 填充众数
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
```

---

## 二、异常值处理方案

### 2.1 删除异常值过多的特征

以下特征异常值比例过高，且对房价预测价值有限：

```python
# 删除异常值比例过高的列
cols_to_drop_outliers = ['BsmtFinSF2', 'EnclosedPorch']
df = df.drop(columns=cols_to_drop_outliers, errors='ignore')
```

### 2.2 Winsorize处理（缩尾处理）

对以下数值特征进行1%-99%分位数缩尾处理，保留边界值：

```python
def winsorize_series(series, lower_percentile=0.01, upper_percentile=0.99):
    """对序列进行缩尾处理"""
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower, upper)

# 需要Winsorize的列（基于业务合理性）
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
]

for col in winsorize_cols:
    if col in df.columns:
        df[col] = winsorize_series(df[col])
```

### 2.3 保留的异常值

以下特征的"异常值"实际上具有业务意义，予以保留：

```python
# 这些异常值代表真实的极端情况，对房价预测有重要价值
keep_outliers = ['OverallQual', 'YearBuilt', 'BsmtFinSF1', '2ndFlrSF', 
                 'BsmtFullBath', 'Fireplaces', 'GarageCars', 'PoolArea']
# 无需处理，保留原值
```

---

## 三、数据类型优化

### 3.1 分类变量转换

将43个分类变量转换为`category`类型，减少内存占用并明确数据类型：

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
    'SaleType', 'SaleCondition', 'Alley', 'MSSubClass'
]

# 转换为category类型
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
```

### 3.2 数值类型优化

```python
# MSSubClass实际是分类变量（建筑类型编码）
df['MSSubClass'] = df['MSSubClass'].astype('category')

# 确保年份相关列为整数
year_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']
for col in year_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)
```

---

## 四、特征工程（针对房价预测）

### 4.1 创建衍生特征

```python
# 房屋总面积特征（多维度聚合）
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + 
                      df['ScreenPorch'] + df['WoodDeckSF'])

# 房屋年龄特征（相对于售出时间）
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# 是否有游泳池
df['HasPool'] = (df['PoolArea'] > 0).astype(int)

# 是否有地下室
df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)

# 车库容量分类
df['GarageCapacity'] = pd.cut(df['GarageCars'], 
                               bins=[-1, 0, 1, 2, 10], 
                               labels=['NoGarage', 'Small', 'Medium', 'Large'])
```

### 4.2 对数变换（针对RMSE优化）

由于评估指标是RMSE，且房价通常呈右偏分布，对目标变量和偏态特征进行对数变换：

```python
import numpy as np

# 对目标变量进行对数变换（必须在训练集上拟合，测试集上应用）
df['SalePrice_Log'] = np.log1p(df['SalePrice'])

# 对高度右偏的数值特征进行对数变换
skewed_cols = ['LotArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 
               'MiscVal', 'TotalSF', 'TotalPorchSF']

for col in skewed_cols:
    if col in df.columns:
        # 加1避免log(0)
        df[f'{col}_Log'] = np.log1p(df[col])
```

---

## 五、完整清洗代码

```python
import pandas as pd
import numpy as np
from scipy import stats

def clean_housing_data(file_path):
    """
    房价数据清洗主函数
    """
    # 1. 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # 2. 删除高缺失率列
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"删除高缺失率列后: {df.shape}")
    
    # 3. 删除异常值过多的列
    df = df.drop(columns=['BsmtFinSF2', 'EnclosedPorch'], errors='ignore')
    
    # 4. 填充缺失值
    # 4.1 分类变量填充
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
    
    garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cols:
        df[col] = df[col].fillna('None')
    
    bsmt_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
    for col in bsmt_cols:
        df[col] = df[col].fillna('None')
    
    # 4.2 数值变量填充
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    # 5. 异常值处理（Winsorize）
    def winsorize_series(series, lower=0.01, upper=0.99):
        return series.clip(series.quantile(lower), series.quantile(upper))
    
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
        'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
    ]
    
    for col in winsorize_cols:
        if col in df.columns:
            df[col] = winsorize_series(df[col])
    
    # 6. 数据类型转换
    categorical_cols = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
        'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # MSSubClass转为分类变量
    df['MSSubClass'] = df['MSSubClass'].astype('category')
    
    # 7. 特征工程
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + 
                          df['ScreenPorch'] + df['WoodDeckSF'])
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    
    # 8. 对数变换（针对RMSE优化）
    df['SalePrice_Log'] = np.log1p(df['SalePrice'])
    
    skewed_cols = ['LotArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 
                   'MiscVal', 'TotalSF', 'TotalPorchSF']
    for col in skewed_cols:
        if col in df.columns:
            df[f'{col}_Log'] = np.log1p(df[col])
    
    print(f"清洗后数据形状: {df.shape}")
    print(f"缺失值总数: {df.isnull().sum().sum()}")
    
    return df

# 执行清洗
df_cleaned = clean_housing_data('/Users/cjialin/code/AutoMLByLLM/train.csv')
```

---

## 六、数据质量验证

### 6.1 验证检查清单

```python
def validate_cleaned_data(df):
    """验证清洗后的数据质量"""
    report = {}
    
    # 1. 检查缺失值
    missing = df.isnull().sum()
    report['remaining_missing'] = missing[missing > 0].to_dict()
    report['total_missing'] = missing.sum()
    
    # 2. 检查目标变量
    report['target_stats'] = {
        'SalePrice_mean': df['SalePrice'].mean(),
        'SalePrice_std': df['SalePrice'].std(),
        'SalePrice_min': df['SalePrice'].min(),
        'SalePrice_max': df['SalePrice'].max()
    }
    
    # 3. 检查数据类型分布
    report['dtypes'] = df.dtypes.value_counts().to_dict()
    
    # 4. 检查重复行
    report['duplicate_rows'] = df.duplicated