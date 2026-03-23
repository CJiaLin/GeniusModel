# 房价预测数据清洗方案

## 1. 方案概述

**任务背景**: 房价预测（回归任务）  
**目标变量**: `SalePrice`  
**评估指标**: RMSE（均方根误差）  
**数据规模**: 1,460 行 × 81 列  
**数据文件**: `/Users/cjialin/code/AutoMLByLLM/train.csv`

### 清洗策略总览
- **缺失值处理**: 删除5个极高缺失率列（>59%），填充其余缺失值
- **异常值处理**: Winsorize缩尾处理21个数值列，删除2个高异常率列
- **数据类型优化**: 43个分类变量转换为category类型
- **特征工程**: 基于业务理解创建新特征

---

## 2. 详细清洗步骤

### 步骤 1: 数据加载与初步检查

```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer

# 加载数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 基础信息检查
print(f"数据形状: {df.shape}")
print(f"\n目标变量统计:\n{df['SalePrice'].describe()}")

# 保存Id列（用于最终预测提交）
ids = df['Id'].copy()
```

### 步骤 2: 高缺失率列删除

根据分析报告，以下5列缺失率超过59%，直接删除：

| 列名 | 缺失率 | 删除原因 |
|------|--------|----------|
| PoolQC | 99.52% | 几乎全缺失 |
| MiscFeature | 96.30% | 几乎全缺失 |
| Alley | 93.77% | 几乎全缺失 |
| Fence | 80.75% | 高缺失率 |
| MasVnrType | 59.73% | 高缺失率 |

```python
# 删除高缺失率列
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=cols_to_drop)
print(f"删除后数据形状: {df.shape}")
```

### 步骤 3: 缺失值填充

#### 3.1 数值型缺失值填充

| 列名 | 缺失数 | 填充策略 | 业务逻辑 |
|------|--------|----------|----------|
| LotFrontage | 259 | 中位数填充 | 与街区相关，中位数更稳健 |
| GarageYrBlt | 81 | 中位数填充 | 建造年份，中位数合理 |
| MasVnrArea | 8 | 0填充 | 缺失表示无砌体贴面 |

```python
# 数值列填充
df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
df['GarageYrBlt'].fillna(df['GarageYrBlt'].median(), inplace=True)
df['MasVnrArea'].fillna(0, inplace=True)
```

#### 3.2 分类型缺失值填充

| 列名 | 缺失数 | 填充策略 | 业务逻辑 |
|------|--------|----------|----------|
| FireplaceQu | 690 | "None" | 缺失表示无壁炉 |
| GarageType | 81 | "None" | 缺失表示无车库 |
| GarageFinish | 81 | "None" | 缺失表示无车库 |
| GarageQual | 81 | "None" | 缺失表示无车库 |
| GarageCond | 81 | "None" | 缺失表示无车库 |
| BsmtExposure | 38 | "None" | 缺失表示无地下室 |
| BsmtFinType2 | 38 | "None" | 缺失表示无地下室 |
| BsmtQual | 37 | "None" | 缺失表示无地下室 |
| BsmtCond | 37 | "None" | 缺失表示无地下室 |
| BsmtFinType1 | 37 | "None" | 缺失表示无地下室 |
| Electrical | 1 | 众数填充 | 仅1个缺失，用最常见值 |

```python
# 地下室相关列填充
bsmt_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
for col in bsmt_cols:
    df[col].fillna('None', inplace=True)

# 车库相关列填充
garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cols:
    df[col].fillna('None', inplace=True)

# 壁炉质量填充
df['FireplaceQu'].fillna('None', inplace=True)

# Electrical用众数填充
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
```

### 步骤 4: 异常值处理

#### 4.1 高异常率列删除

| 列名 | 异常率 | 处理 |
|------|--------|------|
| BsmtFinSF2 | 11.44% | 删除 |
| EnclosedPorch | 14.25% | 删除 |

```python
# 删除高异常率列
df = df.drop(columns=['BsmtFinSF2', 'EnclosedPorch'])
```

#### 4.2 Winsorize缩尾处理

对以下21个数值列进行5%-95%分位数缩尾处理：

```python
def winsorize_series(s, lower_percentile=0.05, upper_percentile=0.95):
    """对序列进行缩尾处理"""
    lower = s.quantile(lower_percentile)
    upper = s.quantile(upper_percentile)
    return s.clip(lower, upper)

# 需要Winsorize的列（排除目标变量SalePrice，在验证集上单独处理）
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal'
]

# 注意：SalePrice在训练集上需要处理，但通常不对目标变量做Winsorize
# 这里仅对特征变量处理

for col in winsorize_cols:
    if col in df.columns:
        df[col] = winsorize_series(df[col])
```

### 步骤 5: 数据类型转换

将43个分类变量转换为category类型，减少内存占用并优化模型性能：

```python
# 分类变量列表
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

# 转换为category类型（仅存在于当前数据中的列）
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
```

### 步骤 6: 特征工程（房价预测专用）

基于房价预测的业务理解，创建以下新特征：

```python
# 6.1 总面积特征（重要房价指标）
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# 6.2 房屋年龄和翻新年龄
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# 6.3 总浴室数
df['TotalBathrooms'] = (df['FullBath'] + df['BsmtFullBath'] + 
                        0.5 * (df['HalfBath'] + df['BsmtHalfBath']))

# 6.4 总门廊面积
df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + 
                      df['ScreenPorch'] + df['WoodDeckSF'])

# 6.5 是否有游泳池
df['HasPool'] = (df['PoolArea'] > 0).astype(int)

# 6.6 是否有车库
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)

# 6.7 是否有地下室
df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)

# 6.8 是否有壁炉
df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

# 6.9 质量评分组合
df['QualCond'] = df['OverallQual'] * df['OverallCond']
```

### 步骤 7: 目标变量处理

对于房价预测任务，目标变量`SalePrice`通常呈现右偏分布，建议进行对数变换以降低RMSE：

```python
# 对目标变量进行对数变换（使分布更接近正态）
df['SalePriceLog'] = np.log1p(df['SalePrice'])

# 原始SalePrice保留用于参考，建模时使用SalePriceLog
```

---

## 3. 清洗验证

### 3.1 验证代码

```python
def validate_cleaning(df):
    """验证清洗结果"""
    report = {}
    
    # 1. 检查缺失值
    missing = df.isnull().sum()
    report['remaining_missing'] = missing[missing > 0].to_dict()
    
    # 2. 检查数据形状
    report['final_shape'] = df.shape
    
    # 3. 检查数值列范围
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # 4. 检查分类列唯一值
    cat_cols = df.select_dtypes(include=['category']).columns
    report['cat_unique_counts'] = {col: df[col].nunique() for col in cat_cols}
    
    return report

# 执行验证
validation_report = validate_cleaning(df)
print("清洗验证报告:")
print(f"最终数据形状: {validation_report['final_shape']}")
print(f"剩余缺失值: {validation_report['remaining_missing']}")
```

### 3.2 预期验证标准

| 检查项 | 预期结果 |
|--------|----------|
| 剩余缺失值 | 0（或极少数） |
| 数据行数 | 1,460（保持不变） |
| 列数 | 约75列（删除7列，新增9列特征） |
| 异常值比例 | <2%（Winsorize后） |

---

## 4. 完整清洗代码

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_housing_data(file_path, is_train=True):
    """
    房价数据清洗函数
    
    Parameters:
    -----------
    file_path : str
        数据文件路径
    is_train : bool
        是否为训练集（训练集需要对SalePrice处理）
    
    Returns:
    --------
    pd.DataFrame
        清洗后的数据
    """
    # 1. 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # 2. 删除高缺失率列
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 3. 删除高异常率列
    df = df.drop(columns=['BsmtFinSF2', 'EnclosedPorch'], errors='ignore')
    
    # 4. 数值缺失值填充
    if 'LotFrontage' in df.columns:
        df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'].fillna(df['GarageYrBlt'].median(), inplace=True)
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'].fillna(0, inplace=True)
    
    # 5. 分类缺失值填充
    none_cols = ['FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
                 'GarageCond', 'BsmtExposure', 'BsmtFinType2', 'BsmtQual',
                 'BsmtCond', 'BsmtFinType1']
    for col in none_cols:
        if col in df.columns:
            df[col].fillna('None', inplace=True)
    
    if 'Electrical' in df.columns:
        df['Electrical'].fillna('SBrkr', inplace=True)  # 最常见值
    
    # 6. 异常值Winsorize处理
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
        'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal'
    ]
    
    for col in winsorize_cols:
        if col in df.columns:
            lower = df[col].quantile(0.05)
            upper = df[col].quantile(0.95)
            df[col] = df[col].clip(lower, upper)
    
    # 7. 特征工程
    if all(col in df.columns for col in ['TotalBsmtSF', '1stFl