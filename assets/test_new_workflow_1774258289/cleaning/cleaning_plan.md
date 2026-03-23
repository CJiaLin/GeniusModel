# 房价预测数据清洗方案

## 一、数据概况与建模背景

| 项目 | 说明 |
|------|------|
| **数据路径** | `/Users/cjialin/code/AutoMLByLLM/train.csv` |
| **数据形状** | (1460, 81) |
| **目标变量** | SalePrice（房价） |
| **评估指标** | RMSE（均方根误差） |
| **任务类型** | 回归预测 |

---

## 二、数据质量问题总览

### 2.1 问题分类统计

| 问题类型 | 数量 | 优先级 |
|----------|------|--------|
| 高缺失率列（>50%） | 5列 | 🔴 高 |
| 中缺失率列（5%-50%） | 7列 | 🟡 中 |
| 低缺失率列（<5%） | 7列 | 🟢 低 |
| 异常值需处理列 | 19列 | 🟡 中 |
| 数据类型优化列 | 43列 | 🟢 低 |
| 重复行 | 0 | - |

---

## 三、详细清洗策略

### 3.1 缺失值处理策略

#### 🔴 高缺失率列：直接删除（缺失率>50%）

| 列名 | 缺失率 | 删除原因 |
|------|--------|----------|
| `PoolQC` | 99.52% | 几乎无有效信息 |
| `MiscFeature` | 96.30% | 几乎无有效信息 |
| `Alley` | 93.77% | 几乎无有效信息 |
| `Fence` | 80.75% | 缺失过多，难以可靠填充 |
| `MasVnrType` | 59.73% | 缺失过半，且与MasVnrArea相关 |

#### 🟡 中缺失率列：智能填充（缺失率5%-50%）

| 列名 | 缺失率 | 填充策略 | 业务逻辑 |
|------|--------|----------|----------|
| `FireplaceQu` | 47.26% | 填充"None" | 无Fireplace则无质量评级 |
| `LotFrontage` | 17.74% | 按Neighborhood分组中位数 | 同社区地块宽度相似 |
| `GarageType` | 5.55% | 填充"None" | 无Garage则无类型 |
| `GarageYrBlt` | 5.55% | 填充YearBuilt | 无Garage则与房屋同年 |
| `GarageFinish` | 5.55% | 填充"None" | 无Garage则无装修等级 |
| `GarageQual` | 5.55% | 填充"None" | 无Garage则无质量评级 |
| `GarageCond` | 5.55% | 填充"None" | 无Garage则无条件评级 |

#### 🟢 低缺失率列：简单填充（缺失率<5%）

| 列名 | 缺失率 | 填充策略 |
|------|--------|----------|
| `BsmtExposure` | 2.60% | 填充"No"（无地下室暴露） |
| `BsmtFinType2` | 2.60% | 填充"No"（无地下室） |
| `BsmtQual` | 2.53% | 填充"No"（无地下室） |
| `BsmtCond` | 2.53% | 填充"No"（无地下室） |
| `BsmtFinType1` | 2.53% | 填充"No"（无地下室） |
| `MasVnrArea` | 0.55% | 填充0（无贴面面积） |
| `Electrical` | 0.07% | 填充众数（SBrkr） |

### 3.2 异常值处理策略

#### 需删除的列（过多零值/异常分布）

| 列名 | 异常率 | 处理方式 | 原因 |
|------|--------|----------|------|
| `BsmtFinSF2` | 11.44% | **删除整列** | 绝大多数为0，方差极低 |
| `EnclosedPorch` | 14.25% | **删除整列** | 绝大多数为0，方差极低 |

#### 需Winsorize缩尾处理的列（19列）

使用1.5×IQR规则确定上下界，极端值缩尾至边界：

| 列名 | 异常率 | 处理边界 | 策略 |
|------|--------|----------|------|
| `MSSubClass` | 7.05% | [-55.0, 145.0] | 上下界缩尾 |
| `LotFrontage` | 6.03% | [27.5, 111.5] | 上界缩尾 |
| `LotArea` | 4.73% | [1481.5, 17673.5] | 上下界缩尾 |
| `OverallCond` | 8.56% | [3.5, 7.5] | 上下界缩尾 |
| `MasVnrArea` | 6.58% | [-249.0, 415.0] | 上界缩尾（面积≥0） |
| `BsmtUnfSF` | 1.99% | [-654.5, 1685.5] | 上下界缩尾 |
| `TotalBsmtSF` | 4.18% | [42.0, 2052.0] | 上下界缩尾 |
| `1stFlrSF` | 1.37% | [118.12, 2155.12] | 上下界缩尾 |
| `LowQualFinSF` | 1.78% | [0.0, 0.0] | 上界缩尾 |
| `GrLivArea` | 2.12% | [158.62, 2747.62] | 上下界缩尾 |
| `BsmtHalfBath` | 5.62% | [0.0, 0.0] | 上界缩尾 |
| `BedroomAbvGr` | 2.40% | [0.5, 4.5] | 上下界缩尾 |
| `KitchenAbvGr` | 4.66% | [1.0, 1.0] | 上下界缩尾（众数为1） |
| `TotRmsAbvGrd` | 2.05% | [2.0, 10.0] | 上下界缩尾 |
| `GarageArea` | 1.44% | [-27.75, 938.25] | 上下界缩尾（面积≥0） |
| `WoodDeckSF` | 2.19% | [-252.0, 420.0] | 上下界缩尾（面积≥0） |
| `OpenPorchSF` | 5.27% | [-102.0, 170.0] | 上下界缩尾（面积≥0） |
| `3SsnPorch` | 1.64% | [0.0, 0.0] | 上界缩尾 |
| `ScreenPorch` | 7.95% | [0.0, 0.0] | 上界缩尾 |
| `MiscVal` | 3.56% | [0.0, 0.0] | 上界缩尾 |
| `SalePrice` | 4.18% | [3937.5, 340037.5] | **目标变量：上下界缩尾** |

#### 保留的列（异常值合理或比例极低）

| 列名 | 异常率 | 保留原因 |
|------|--------|----------|
| `OverallQual` | 0.14% | 比例极低，且可能为真实高/低品质房屋 |
| `YearBuilt` | 0.48% | 可能为真实历史建筑 |
| `BsmtFinSF1` | 0.48% | 比例极低 |
| `2ndFlrSF` | 0.14% | 比例极低 |
| `BsmtFullBath` | 0.07% | 比例极低 |
| `Fireplaces` | 0.34% | 可能为真实多壁炉房屋 |
| `GarageCars` | 0.34% | 可能为真实多车位车库 |
| `PoolArea` | 0.48% | 游泳池面积，真实存在 |

### 3.3 数据类型优化

将43个object类型分类变量转换为`category`类型，减少内存占用并提升建模效率。

---

## 四、Python清洗代码

```python
"""
房价预测数据清洗脚本
数据路径: /Users/cjialin/code/AutoMLByLLM/train.csv
目标: 清洗数据用于房价预测模型（RMSE评估）
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. 加载数据
# ============================================
file_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
df = pd.read_csv(file_path)
print(f"原始数据形状: {df.shape}")

# 保存Id列（预测时需要）
if 'Id' in df.columns:
    id_col = df['Id'].copy()

# ============================================
# 2. 删除高缺失率列（>50%）
# ============================================
high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=high_missing_cols)
print(f"删除高缺失列后: {df.shape}")

# ============================================
# 3. 删除低方差/异常分布列
# ============================================
low_variance_cols = ['BsmtFinSF2', 'EnclosedPorch']
df = df.drop(columns=low_variance_cols)
print(f"删除低方差列后: {df.shape}")

# ============================================
# 4. 缺失值填充 - 地下室相关（中缺失率+低缺失率）
# ============================================

# 地下室分类变量：缺失表示无地下室，填充"None"
bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 
                 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_cat_cols:
    df[col] = df[col].fillna('None')

# ============================================
# 5. 缺失值填充 - 车库相关
# ============================================

# 车库分类变量：缺失表示无车库，填充"None"
garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cat_cols:
    df[col] = df[col].fillna('None')

# GarageYrBlt：无车库时填充YearBuilt（房屋建造年份）
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

# ============================================
# 6. 缺失值填充 - 其他特征
# ============================================

# FireplaceQu：无壁炉填充"None"
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

# LotFrontage：按Neighborhood分组填充中位数（同社区地块相似）
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
# 若仍有缺失（如新社区），填充整体中位数
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# MasVnrArea：无贴面填充0
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# Electrical：填充众数（标准断路器SBrkr）
df['Electrical'] = df['Electrical'].fillna('SBrkr')

print(f"缺失值填充完成，剩余缺失值: {df.isnull().sum().sum()}")

# ============================================
# 7. 异常值处理 - Winsorize缩尾
# ============================================

def winsorize_series(series, lower_quantile=0.01, upper_quantile=0.99):
    """对序列进行缩尾处理"""
    lower_bound = series.quantile(lower_quantile)
    upper_bound = series.quantile(upper_quantile)
    return series.clip(lower=lower_bound, upper=upper_bound)

# 需Winsorize的列及其边界（基于IQR分析）
winsorize_config = {
    'MSSubClass': (0.01, 0.99),
    'LotFrontage': (0.00, 0.99),
    'LotArea': (0.01, 0.99),
    'OverallCond': (0.01, 0.99),
    'MasVnrArea': (0.00, 0.99),  # 面积下限为0
    'BsmtUnfSF': (0.01, 0.99),
    'TotalBsmtSF': (0.01, 0.99),
    '1stFlrSF': (0.01, 0.99),
    'LowQualFinSF': (0.00, 0.90),  # 大量0值
    'GrLivArea': (0.01, 0.99),
    'BsmtHalfBath': (0.00, 0.95),
    'BedroomAbvGr': (0.01, 0.99),
    'KitchenAbvGr': (0.01, 0.99),
    'TotRmsAbvGrd': (0.01, 0.99),
    'GarageArea': (0.00, 0.99),
    'WoodDeckSF': (0.00, 0.99),
    'OpenPorchSF': (0.00, 0.99),
    '3SsnPorch': (0.00, 0.95),
    'ScreenPorch': (0.00, 0.95),
    'MiscVal': (0.00, 0.95),
    'SalePrice': (0.01, 0.99)  # 目标变量也要处理
}

for col, (lower_q, upper_q) in winsorize_config.items():
    if col in df.columns:
        before_outliers = ((df[col] < df[col].quantile(0.01)) | 
                          (df[col] > df[col].quantile(0.99))).sum()
        df[col] = winsorize_series(df[col], lower_q, upper_q)
        after_outliers = ((df[col] < df[col].quantile(0.01)) | 
                         (df[col] > df[col].quantile(0.99))).sum()
        print(f"{col}: 缩尾处理完成，极端值从 {before_outliers} 降至 {after_outliers}")

print("异常值处理完成")

# ============================================
# 8. 数据类型优化
# ============================================

# 将object类型转换为category（提升效率）
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    df[col] = df[col].astype('category')

print(f"分类变量已优化: {len(categorical_cols)} 列")

# ============================================
# 9. 最终验证
# ============================================

print("\n" + "="*50)
print("清洗后数据质量报告")
print("="*50)
print(f"数据形状: {df.shape}")
print(f"缺失值总数: {df.isnull().sum().sum()}")
print(f"重复行数: {df.duplicated().sum()}")
print(f"内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 数值列统计
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\n数值列: {len(numeric_cols)} 个")
print(f"分类列: {len(categorical_cols)} 个")

# ============================================