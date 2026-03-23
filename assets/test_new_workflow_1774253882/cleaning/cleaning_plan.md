# 🧹 房价预测数据清洗方案

## 📋 方案概述

**数据集**: `/Users/cjialin/code/AutoMLByLLM/train.csv`  
**数据形状**: (1460, 81)  
**任务类型**: 房价预测（回归任务）  
**目标变量**: `SalePrice`  
**评估指标**: RMSE

---

## 1️⃣ 数据质量概览

| 问题类型 | 数量 | 优先级 |
|---------|------|--------|
| 高缺失率列（>50%） | 5列 | 🔴 高 |
| 中等缺失率列（5%-50%） | 6列 | 🟡 中 |
| 低缺失率列（<5%） | 8列 | 🟢 低 |
| 异常值列 | 29列 | 🟡 中 |
| 重复行 | 0行 | ✅ 无问题 |
| 数据类型优化 | 43列 | 🟢 低 |

---

## 2️⃣ 清洗步骤详解

### 步骤 1: 删除高缺失率列（缺失率 > 50%）

**依据**: 这些列缺失率过高，填充会引入大量噪声，且对房价预测贡献有限。

```python
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
```

| 列名 | 缺失率 | 删除原因 |
|------|--------|----------|
| PoolQC | 99.52% | 几乎无泳池质量信息 |
| MiscFeature | 96.3% | 几乎无杂项特征信息 |
| Alley | 93.77% | 几乎无巷道信息 |
| Fence | 80.75% | 围栏信息缺失严重 |
| MasVnrType | 59.73% | 砌体 veneer 类型缺失过半 |

**清洗操作**: 直接删除这些列

---

### 步骤 2: 缺失值填充策略

#### 2.1 数值型缺失值填充

| 列名 | 缺失数 | 填充策略 | 理由 |
|------|--------|----------|------|
| LotFrontage | 259 | 中位数 | 街道长度，受社区影响大 |
| GarageYrBlt | 81 | 中位数 | 建造年份，用整体中位数 |
| MasVnrArea | 8 | 0 | 缺失表示无砌体 veneer |

#### 2.2 分类型缺失值填充（车库相关）

| 列名 | 缺失数 | 填充值 | 理由 |
|------|--------|--------|------|
| GarageType | 81 | 'NoGarage' | 缺失表示无车库 |
| GarageFinish | 81 | 'NoGarage' | 缺失表示无车库 |
| GarageQual | 81 | 'NoGarage' | 缺失表示无车库 |
| GarageCond | 81 | 'NoGarage' | 缺失表示无车库 |

#### 2.3 分类型缺失值填充（地下室相关）

| 列名 | 缺失数 | 填充值 | 理由 |
|------|--------|--------|------|
| BsmtQual | 37 | 'NoBsmt' | 缺失表示无地下室 |
| BsmtCond | 37 | 'NoBsmt' | 缺失表示无地下室 |
| BsmtExposure | 38 | 'NoBsmt' | 缺失表示无地下室 |
| BsmtFinType1 | 37 | 'NoBsmt' | 缺失表示无地下室 |
| BsmtFinType2 | 38 | 'NoBsmt' | 缺失表示无地下室 |

#### 2.4 其他分类型缺失值

| 列名 | 缺失数 | 填充策略 | 理由 |
|------|--------|----------|------|
| FireplaceQu | 690 | 'NoFireplace' | 缺失表示无壁炉 |
| Electrical | 1 | 众数 | 仅1个缺失，用最常见的 |

---

### 步骤 3: 异常值处理（Winsorize 策略）

**业务背景**: 房价预测任务中，极端值可能是真实存在的豪宅或特殊情况，直接删除会损失信息。采用 Winsorize（缩尾处理）保留极端值但限制其影响。

**需要 Winsorize 的列**（基于 IQR 方法）:

```python
winsorize_cols = {
    'MSSubClass': (0.05, 0.95),      # 建筑类型
    'LotFrontage': (0.05, 0.95),     # 街道长度
    'LotArea': (0.05, 0.95),         # 地块面积
    'OverallCond': (0.05, 0.95),     # 总体条件
    'MasVnrArea': (0.05, 0.95),      # 砌体面积
    'BsmtUnfSF': (0.05, 0.95),       # 未完工地下室面积
    'TotalBsmtSF': (0.05, 0.95),     # 总地下室面积
    '1stFlrSF': (0.05, 0.95),        # 第一层面积
    'LowQualFinSF': (0.05, 0.95),    # 低质量完成面积
    'GrLivArea': (0.05, 0.95),       # 地面生活面积
    'BsmtHalfBath': (0.05, 0.95),    # 地下室半卫
    'BedroomAbvGr': (0.05, 0.95),    # 卧室数
    'KitchenAbvGr': (0.05, 0.95),    # 厨房数
    'TotRmsAbvGrd': (0.05, 0.95),    # 总房间数
    'GarageArea': (0.05, 0.95),      # 车库面积
    'WoodDeckSF': (0.05, 0.95),      # 木甲板面积
    'OpenPorchSF': (0.05, 0.95),     # 开放式门廊
    '3SsnPorch': (0.05, 0.95),       # 三季门廊
    'ScreenPorch': (0.05, 0.95),     # 纱门门廊
    'MiscVal': (0.05, 0.95),         # 杂项价值
    'SalePrice': (0.05, 0.95)        # 目标变量（训练集）
}
```

**保留的列**（异常值比例低或有业务意义）:
- `OverallQual`: 仅为2个异常值，且是重要质量指标
- `YearBuilt`: 老房子可能是历史建筑，有价值
- `BsmtFinSF1`, `BsmtFinSF2`: 地下室面积可为0
- `2ndFlrSF`: 单层房为0是正常情况
- `Fireplaces`, `GarageCars`: 离散值，范围合理
- `PoolArea`: 泳池面积可为0

---

### 步骤 4: 删除无信息量变列

**依据**: 报告中 `BsmtFinSF2` 和 `EnclosedPorch` 的异常值比例分别为 11.44% 和 14.25%，且正常范围为 [0.0, 0.0]，说明这些列大部分值为0，方差极低。

```python
# 删除方差极低的列
low_variance_cols = ['BsmtFinSF2', 'EnclosedPorch']
```

---

### 步骤 5: 数据类型优化

**将分类变量转换为 category 类型**，减少内存占用并明确数据类型：

```python
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
```

**注意**: 经过步骤1删除后，部分列（PoolQC, Fence, MiscFeature）已不存在。

---

## 3️⃣ 完整清洗代码

```python
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def clean_housing_data(file_path, is_train=True):
    """
    清洗房价预测数据集
    
    Parameters:
    -----------
    file_path : str
        数据文件路径
    is_train : bool
        是否为训练集（训练集需要处理目标变量SalePrice）
    
    Returns:
    --------
    df : pd.DataFrame
        清洗后的数据
    """
    # 1. 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # 2. 删除高缺失率列（>50%）
    cols_to_drop_high_missing = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df = df.drop(columns=[col for col in cols_to_drop_high_missing if col in df.columns])
    print(f"删除高缺失率列后: {df.shape}")
    
    # 3. 删除低方差列
    low_variance_cols = ['BsmtFinSF2', 'EnclosedPorch']
    df = df.drop(columns=[col for col in low_variance_cols if col in df.columns])
    
    # 4. 缺失值填充
    
    # 4.1 数值型缺失值
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
    
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # 4.2 车库相关分类型变量
    garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cols:
        if col in df.columns:
            df[col] = df[col].fillna('NoGarage')
    
    # 4.3 地下室相关分类型变量
    bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in bsmt_cols:
        if col in df.columns:
            df[col] = df[col].fillna('NoBsmt')
    
    # 4.4 其他分类型变量
    if 'FireplaceQu' in df.columns:
        df['FireplaceQu'] = df['FireplaceQu'].fillna('NoFireplace')
    
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    # 5. 异常值处理（Winsorize）
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
        'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal'
    ]
    
    # 训练集需要处理目标变量
    if is_train and 'SalePrice' in df.columns:
        winsorize_cols.append('SalePrice')
    
    for col in winsorize_cols:
        if col in df.columns:
            # 只对数值型且非空值进行 winsorize
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = winsorize(df[col].values, limits=[0.05, 0.05])
    
    # 6. 数据类型优化
    # 定义分类变量（删除已删除的列）
    categorical_cols = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond',
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # 7. 最终检查
    remaining_missing = df.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    if len(remaining_missing) > 0:
        print("警告：仍有缺失值存在:")
        print(remaining_missing)
    else:
        print("✅ 所有缺失值已处理完成")
    
    print(f"清洗后数据形状: {df.shape}")
    return df


# 执行清洗
if __name__ == "__main__":
    # 训练集清洗
    train_cleaned = clean_housing_data('/Users/cjialin/code/AutoMLByLLM/train.csv', is_train=True)
    
    # 保存清洗后的数据
    train_cleaned.to_csv('/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv', index=False)
    print("清洗完成，数据已保存")
```

---

## 4️⃣ 清洗效果验证

### 验证清单

| 检查项 | 期望结果 | 验证方法 |
|--------|---------|---------|
| 缺失值 | 0 | `df.isnull().sum().sum() == 0` |
| 重复行 | 0 | `df.duplicated().sum() == 0` |
| 目标变量分布 | 接近正态 | 绘制直方图 |
| 分类变量类型 | category | `df.dtypes` |
| 数值变量范围 | 合理范围内 | 描述性统计 |

### 清洗前后对比

| 指标 | 清洗前 | 清洗后 |
|------|--------|--------|
| 列数 | 81 | 74（删除7列） |
| 缺失值总数 | ~6,000+ | 0 |
| 异常值比例 | ~30% | <5% |
| 内存占用 | 高 | 优化后降低 |

---

## 5️⃣ 针对房价预测的特殊考虑

### 5.1 特征工程建议（清洗后）

基于清洗后的数据，建议创建以下新特征以提升RMSE：

```python
# 总面积特征
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# 房屋年龄
df['HouseAge'] = df['YrSold'] - df['YearBuilt']

# 翻新年龄
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# 每平方英尺价格（仅训练集）
if 'SalePrice' in df.columns:
    df['PricePerSF'] = df['SalePrice'] / df['GrLivArea']
```

### 5.2 RMSE优化策略

1. **对数变换**: 房价通常右偏，对 `SalePrice` 取对数可改善RMSE
2. **标准化**: Winsorize 后的数值特征进行标准化
3. **编码**: 分类变量使用 One-Hot 或 Target Encoding

---

## 6️⃣ 总结

本清洗方案针对房价预测任务的特点，采用以下核心策略：

1. **保守删除**: 仅删除缺失率>50%且业务价值低的列
2. **业务导向填充**: 根据"缺失=不存在"的业务逻辑填充分类变量
3. **Winsorize处理异常值**: 保留极端值信息但限制影响
4. **类型优化**: 减少内存并提升模型训练效率

清洗后的数据将更适合回归模型训练