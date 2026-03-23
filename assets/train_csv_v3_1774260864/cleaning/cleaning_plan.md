# 数据清洗方案

## 1. 数据概览与评估

### 1.1 基本信息
- **数据文件**: `/Users/cjialin/code/AutoMLByLLM/train.csv`
- **数据形状**: 1460 行 × 81 列
- **数值列**: 38 个
- **分类列**: 43 个
- **重复行**: 0

### 1.2 主要质量问题总结
| 问题类型 | 数量 | 严重程度 |
|---------|------|---------|
| 高缺失率列(>50%) | 5列 | 严重 |
| 中等缺失率列(5%-50%) | 7列 | 中等 |
| 低缺失率列(<5%) | 7列 | 轻微 |
| 异常值需处理 | 20列 | 中等 |
| 异常值建议删除的列 | 2列 | 严重 |
| 数据类型优化 | 43列 | 轻微 |

---

## 2. 数据清洗策略

### 2.1 缺失值处理策略

#### 2.1.1 删除高缺失率列（缺失率>50%）
| 列名 | 缺失率 | 原因 |
|------|--------|------|
| `PoolQC` | 99.52% | 几乎所有房屋无泳池 |
| `MiscFeature` | 96.3% | 杂项特征极少 |
| `Alley` | 93.77% | 小巷通道信息缺失严重 |
| `Fence` | 80.75% | 围栏信息缺失严重 |
| `MasVnrType` | 59.73% | 砌体贴面类型缺失严重 |

#### 2.1.2 中等缺失率列处理策略（5%-50%）
| 列名 | 缺失率 | 填充策略 | 理由 |
|------|--------|---------|------|
| `FireplaceQu` | 47.26% | 填充为'None' | 缺失表示无壁炉 |
| `LotFrontage` | 17.74% | 按Neighborhood分组中位数 | 临街距离与社区相关 |
| `GarageType` | 5.55% | 填充为'None' | 缺失表示无车库 |
| `GarageYrBlt` | 5.55% | 用YearBuilt填充 | 无车库时假设与房屋同年建造 |
| `GarageFinish` | 5.55% | 填充为'None' | 缺失表示无车库 |
| `GarageQual` | 5.55% | 填充为'None' | 缺失表示无车库 |
| `GarageCond` | 5.55% | 填充为'None' | 缺失表示无车库 |

#### 2.1.3 低缺失率列处理策略（<5%）
| 列名 | 缺失率 | 填充策略 |
|------|--------|---------|
| `BsmtExposure` | 2.6% | 填充为'No'（无曝光）|
| `BsmtFinType2` | 2.6% | 填充为'Unf'（未完成）|
| `BsmtQual` | 2.53% | 填充为'Ta'（典型/平均）|
| `BsmtCond` | 2.53% | 填充为'Ta'（典型/平均）|
| `BsmtFinType1` | 2.53% | 填充为'Unf'（未完成）|
| `MasVnrArea` | 0.55% | 填充为0 |
| `Electrical` | 0.07% | 填充为众数'SBrkr' |

### 2.2 异常值处理策略

#### 2.2.1 删除异常值过多的列
| 列名 | 异常值比例 | 处理方式 |
|------|-----------|---------|
| `BsmtFinSF2` | 11.44% | 删除整列 |
| `EnclosedPorch` | 14.25% | 删除整列 |

#### 2.2.2 Winsorize处理（缩尾处理）
对以下列进行1%-99%分位数缩尾处理：
- `MSSubClass`, `LotFrontage`, `LotArea`, `OverallCond`
- `MasVnrArea`, `BsmtUnfSF`, `TotalBsmtSF`, `1stFlrSF`
- `LowQualFinSF`, `GrLivArea`, `BsmtHalfBath`
- `BedroomAbvGr`, `KitchenAbvGr`, `TotRmsAbvGrd`
- `GarageArea`, `WoodDeckSF`, `OpenPorchSF`
- `3SsnPorch`, `ScreenPorch`, `MiscVal`, `SalePrice`

#### 2.2.3 保留异常值
以下列异常值具有业务意义，予以保留：
- `OverallQual`, `YearBuilt`, `BsmtFinSF1`, `2ndFlrSF`
- `BsmtFullBath`, `Fireplaces`, `GarageCars`, `PoolArea`

### 2.3 数据类型优化
将43个object类型的分类变量转换为`category`类型，减少内存占用并提高处理效率。

---

## 3. 数据清洗代码实现

```python
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 步骤1: 加载数据
# ==========================================
def load_data(file_path):
    """加载原始数据"""
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    return df

# ==========================================
# 步骤2: 处理缺失值 - 删除高缺失率列
# ==========================================
def drop_high_missing_columns(df):
    """删除缺失率超过50%的列"""
    columns_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    
    # 检查列是否存在
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    df_cleaned = df.drop(columns=existing_cols)
    
    print(f"删除的列: {existing_cols}")
    print(f"删除后形状: {df_cleaned.shape}")
    return df_cleaned

# ==========================================
# 步骤3: 处理缺失值 - 分类变量填充
# ==========================================
def fill_categorical_missing(df):
    """填充分类变量的缺失值"""
    
    # 填充为'None'（表示无该设施）
    none_columns = ['FireplaceQu', 'GarageType', 'GarageFinish', 
                    'GarageQual', 'GarageCond']
    for col in none_columns:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # 地下室相关特征填充
    if 'BsmtExposure' in df.columns:
        df['BsmtExposure'] = df['BsmtExposure'].fillna('No')
    if 'BsmtFinType2' in df.columns:
        df['BsmtFinType2'] = df['BsmtFinType2'].fillna('Unf')
    if 'BsmtQual' in df.columns:
        df['BsmtQual'] = df['BsmtQual'].fillna('TA')  # Typical/Average
    if 'BsmtCond' in df.columns:
        df['BsmtCond'] = df['BsmtCond'].fillna('TA')
    if 'BsmtFinType1' in df.columns:
        df['BsmtFinType1'] = df['BsmtFinType1'].fillna('Unf')
    
    # Electrical填充为众数
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna('SBrkr')
    
    print("分类变量缺失值填充完成")
    return df

# ==========================================
# 步骤4: 处理缺失值 - 数值变量填充
# ==========================================
def fill_numerical_missing(df):
    """填充数值变量的缺失值"""
    
    # LotFrontage按Neighborhood分组中位数填充
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果仍有缺失，用整体中位数填充
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # GarageYrBlt用YearBuilt填充（假设无车库时与房屋建造年份相同）
    if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    
    # MasVnrArea填充为0
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    print("数值变量缺失值填充完成")
    return df

# ==========================================
# 步骤5: 处理异常值 - 删除异常值过多的列
# ==========================================
def drop_high_outlier_columns(df):
    """删除异常值比例过高的列"""
    columns_to_drop = ['BsmtFinSF2', 'EnclosedPorch']
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    df_cleaned = df.drop(columns=existing_cols)
    
    print(f"删除的列(异常值过多): {existing_cols}")
    return df_cleaned

# ==========================================
# 步骤6: 处理异常值 - Winsorize缩尾处理
# ==========================================
def winsorize_outliers(df):
    """对指定列进行Winsorize缩尾处理(1%-99%)"""
    
    columns_to_winsorize = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
        'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath',
        'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
    ]
    
    for col in columns_to_winsorize:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            # 计算1%和99%分位数
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            
            # 缩尾处理
            df[col] = df[col].clip(lower=lower, upper=upper)
    
    print("Winsorize缩尾处理完成")
    return df

# ==========================================
# 步骤7: 数据类型优化
# ==========================================
def optimize_data_types(df):
    """将分类变量转换为category类型"""
    
    categorical_columns = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
        'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'PavedDrive', 'SaleType', 'SaleCondition'
    ]
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # 优化整数类型
    int_columns = df.select_dtypes(include=['int64']).columns
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # 优化浮点类型
    float_columns = df.select_dtypes(include=['float64']).columns
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    print("数据类型优化完成")
    return df

# ==========================================
# 步骤8: 验证清洗结果
# ==========================================
def validate_cleaning(df_original, df_cleaned):
    """验证数据清洗效果"""
    
    print("\n" + "="*50)
    print("数据清洗验证报告")
    print("="*50)
    
    # 形状对比
    print(f"\n原始数据形状: {df_original.shape}")
    print(f"清洗后形状: {df_cleaned.shape}")
    print(f"删除列数: {df_original.shape[1] - df_cleaned.shape[1]}")
    
    # 缺失值检查
    missing_after = df_cleaned.isnull().sum()
    missing_cols = missing_after[missing_after > 0]
    
    if len(missing_cols) > 0:
        print(f"\n剩余缺失值列({len(missing_cols)}个):")
        for col, count in missing_cols.items():
            print(f"  - {col}: {count} ({count/len(df_cleaned)*100:.2f}%)")
    else:
        print("\n✓ 所有缺失值已处理完毕")
    
    # 数据类型检查
    print(f"\n数据类型分布:")
    print(df_cleaned.dtypes.value_counts())
    
    # 内存使用对比
    mem_original = df_original.memory_usage(deep=True).sum() / 1024**2
    mem_cleaned = df_cleaned.memory_usage(deep=True).sum() / 1024**2
    
    print(f"\n内存使用:")
    print(f"  原始: {mem_original:.2f} MB")
    print(f"  清洗后: {mem_cleaned:.2f} MB")
    print(f"  减少: {(1-mem_cleaned/mem_original)*100:.2f}%")
    
    return True

# ==========================================
# 主执行流程
# ==========================================
def main_cleaning_pipeline(file_path, output_path=None):
    """主清洗流程"""
    
    print("开始数据清洗流程...")
    print("="*50)
    
    # 1. 加载数据
    df = load_data(file_path)
    df_original = df.copy()
    
    # 2. 删除高缺失率列
    df = drop_high_missing_columns(df)
    
    # 3. 删除异常值过多的列
    df = drop_high_outlier_columns(df)
    
    # 4. 填充分类变量缺失值
    df = fill_categorical_missing(df)
    
    # 5. 填充数值变量缺失值
    df = fill_numerical_missing(df)
    
    # 6. 异常值处理(Winsorize)
    df = winsorize_outliers(df)
    
    # 7. 数据类型优化
    df = optimize_data_types(df)
    
    # 8. 验证结果
    validate_cleaning(df_original, df)
    
    # 9. 保存清洗后的数据
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n清洗后的数据已保存至: {output_path}")
    
    print("\n数据清洗完成!")
    return df

# 执行清洗
if __name__ == "__main__":
    FILE_PATH = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    OUTPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    df_cleaned = main_cleaning_pipeline(FILE_PATH, OUTPUT_PATH)
```

---

## 4. 清洗步骤详解

### 4.1 第一步：删除高缺失率列
**原因**：缺失率超过50%的列提供的信息极其有限，填充会引入大量噪声。
- 删除 `PoolQC` (99.52%)、`MiscFeature` (96.3%)、`Alley` (93.77%)、`Fence` (80.75