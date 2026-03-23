# 房价预测数据清洗方案

## 一、数据概述

| 项目 | 详情 |
|------|------|
| **数据文件** | `/Users/cjialin/code/AutoMLByLLM/train.csv` |
| **数据形状** | (1460, 81) |
| **任务类型** | 房价预测回归任务 |
| **目标变量** | `SalePrice` |
| **评估指标** | RMSE（均方根误差） |

## 二、数据质量问题诊断

### 2.1 缺失值分布

| 优先级 | 列名 | 缺失数量 | 缺失比例 | 处理策略 |
|:------:|------|:--------:|:--------:|----------|
| 🔴 高 | `PoolQC` | 1453 | 99.52% | **删除列**（信息极少） |
| 🔴 高 | `MiscFeature` | 1406 | 96.30% | **删除列**（信息极少） |
| 🔴 高 | `Alley` | 1369 | 93.77% | **删除列**（信息极少） |
| 🔴 高 | `Fence` | 1179 | 80.75% | **删除列**（信息极少） |
| 🔴 高 | `MasVnrType` | 872 | 59.73% | **删除列**（缺失过多） |
| 🟡 中 | `FireplaceQu` | 690 | 47.26% | 填充为"None"（无壁炉） |
| 🟡 中 | `LotFrontage` | 259 | 17.74% | 按Neighborhood分组中位数填充 |
| 🟡 中 | `GarageType` | 81 | 5.55% | 填充为"None"（无车库） |
| 🟡 中 | `GarageYrBlt` | 81 | 5.55% | 填充为0（无车库） |
| 🟡 中 | `GarageFinish` | 81 | 5.55% | 填充为"None"（无车库） |
| 🟡 中 | `GarageQual` | 81 | 5.55% | 填充为"None"（无车库） |
| 🟡 中 | `GarageCond` | 81 | 5.55% | 填充为"None"（无车库） |
| 🟢 低 | `BsmtExposure` | 38 | 2.60% | 填充为"None"（无地下室） |
| 🟢 低 | `BsmtFinType2` | 38 | 2.60% | 填充为"None"（无地下室） |
| 🟢 低 | `BsmtQual` | 37 | 2.53% | 填充为"None"（无地下室） |
| 🟢 低 | `BsmtCond` | 37 | 2.53% | 填充为"None"（无地下室） |
| 🟢 低 | `BsmtFinType1` | 37 | 2.53% | 填充为"None"（无地下室） |
| 🟢 低 | `MasVnrArea` | 8 | 0.55% | 填充为0（无砌体贴面） |
| 🟢 低 | `Electrical` | 1 | 0.07% | 填充为众数 |

### 2.2 异常值处理策略

| 列名 | 处理方式 | 说明 |
|------|----------|------|
| `MSSubClass` | Winsorize (5%-95%) | 建筑类型编码，极端值影响有限 |
| `LotFrontage` | Winsorize (1%-99%) | 临街距离，保留合理范围 |
| `LotArea` | Winsorize (1%-99%) | 地块面积，去除极端大面积 |
| `OverallCond` | Winsorize (5%-95%) | 整体状况评分 |
| `MasVnrArea` | Winsorize (1%-99%) | 砌体贴面面积 |
| `BsmtUnfSF` | Winsorize (1%-99%) | 地下室未完工面积 |
| `TotalBsmtSF` | Winsorize (1%-99%) | 地下室总面积 |
| `1stFlrSF` | Winsorize (1%-99%) | 首层面积 |
| `GrLivArea` | Winsorize (1%-99%) | 地上生活面积（关键特征）|
| `GarageArea` | Winsorize (1%-99%) | 车库面积 |
| `WoodDeckSF` | Winsorize (1%-99%) | 木质甲板面积 |
| `OpenPorchSF` | Winsorize (1%-99%) | 开放门廊面积 |
| `SalePrice` | **对数变换** | 目标变量右偏，取log1p改善分布 |

### 2.3 数据类型优化

所有object类型列应转换为category类型以减少内存占用：
- `MSZoning`, `Street`, `Alley`, `LotShape`, `LandContour`, `Utilities`, `LotConfig`, `LandSlope`
- `Neighborhood`, `Condition1`, `Condition2`, `BldgType`, `HouseStyle`
- `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`, `MasVnrType`
- `ExterQual`, `ExterCond`, `Foundation`, `BsmtQual`, `BsmtCond`, `BsmtExposure`
- `BsmtFinType1`, `BsmtFinType2`, `Heating`, `HeatingQC`, `CentralAir`, `Electrical`
- `KitchenQual`, `Functional`, `FireplaceQu`, `GarageType`, `GarageFinish`
- `GarageQual`, `GarageCond`, `PavedDrive`, `PoolQC`, `Fence`, `MiscFeature`
- `SaleType`, `SaleCondition`

## 三、详细清洗代码

```python
import pandas as pd
import numpy as np
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 数据加载 ====================
def load_data(file_path):
    """加载数据文件"""
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    return df

# ==================== 2. 缺失值处理 ====================
def handle_missing_values(df):
    """处理缺失值"""
    df_clean = df.copy()
    
    # 2.1 删除高缺失率列（>50%）
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df_clean = df_clean.drop(columns=cols_to_drop)
    print(f"删除高缺失率列: {cols_to_drop}")
    
    # 2.2 填充分类变量 -  basement相关（缺失表示无地下室）
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        df_clean[col] = df_clean[col].fillna('None')
    
    # 2.3 填充分类变量 - garage相关（缺失表示无车库）
    garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cat_cols:
        df_clean[col] = df_clean[col].fillna('None')
    
    # 2.4 填充数值变量 - GarageYrBlt（缺失表示无车库）
    df_clean['GarageYrBlt'] = df_clean['GarageYrBlt'].fillna(0)
    
    # 2.5 填充分类变量 - FireplaceQu（缺失表示无壁炉）
    df_clean['FireplaceQu'] = df_clean['FireplaceQu'].fillna('None')
    
    # 2.6 填充数值变量 - MasVnrArea（缺失表示无砌体贴面）
    df_clean['MasVnrArea'] = df_clean['MasVnrArea'].fillna(0)
    
    # 2.7 填充Electrical（单一缺失值用众数）
    df_clean['Electrical'] = df_clean['Electrical'].fillna(df_clean['Electrical'].mode()[0])
    
    # 2.8 智能填充LotFrontage（按Neighborhood分组中位数）
    df_clean['LotFrontage'] = df_clean.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    # 如果仍有缺失（某些Neighborhood全是NaN），用整体中位数
    df_clean['LotFrontage'] = df_clean['LotFrontage'].fillna(df_clean['LotFrontage'].median())
    
    return df_clean

# ==================== 3. 异常值处理 ====================
def handle_outliers(df):
    """处理异常值"""
    df_clean = df.copy()
    
    # 需要Winsorize的列及分位数
    winsorize_cols = {
        'MSSubClass': (0.05, 0.95),
        'LotFrontage': (0.01, 0.99),
        'LotArea': (0.01, 0.99),
        'OverallCond': (0.05, 0.95),
        'MasVnrArea': (0.01, 0.99),
        'BsmtUnfSF': (0.01, 0.99),
        'TotalBsmtSF': (0.01, 0.99),
        '1stFlrSF': (0.01, 0.99),
        'GrLivArea': (0.01, 0.99),
        'GarageArea': (0.01, 0.99),
        'WoodDeckSF': (0.01, 0.99),
        'OpenPorchSF': (0.01, 0.99),
    }
    
    for col, (lower, upper) in winsorize_cols.items():
        if col in df_clean.columns:
            df_clean[col] = mstats.winsorize(df_clean[col], limits=(lower, upper))
    
    # 目标变量对数变换（重要：改善右偏分布，降低RMSE）
    df_clean['SalePrice'] = np.log1p(df_clean['SalePrice'])
    
    return df_clean

# ==================== 4. 数据类型转换 ====================
def optimize_dtypes(df):
    """优化数据类型"""
    df_clean = df.copy()
    
    # 获取所有object列
    object_cols = df_clean.select_dtypes(include=['object']).columns
    
    # 转换为category类型
    for col in object_cols:
        df_clean[col] = df_clean[col].astype('category')
    
    # 转换特定数值列为更节省内存的类型
    int_cols = df_clean.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if col != 'Id':  # 保留Id为int64
            df_clean[col] = df_clean[col].astype('int32')
    
    float_cols = df_clean.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df_clean[col] = df_clean[col].astype('float32')
    
    return df_clean

# ==================== 5. 特征工程 ====================
def feature_engineering(df):
    """创建新特征（针对房价预测优化）"""
    df_fe = df.copy()
    
    # 5.1 总面积特征（关键特征组合）
    df_fe['TotalSF'] = df_fe['TotalBsmtSF'] + df_fe['1stFlrSF'] + df_fe['2ndFlrSF']
    df_fe['TotalPorchSF'] = df_fe['OpenPorchSF'] + df_fe['EnclosedPorch'] + \
                            df_fe['3SsnPorch'] + df_fe['ScreenPorch'] + df_fe['WoodDeckSF']
    
    # 5.2 房屋年龄及改造特征
    df_fe['HouseAge'] = df_fe['YrSold'] - df_fe['YearBuilt']
    df_fe['RemodAge'] = df_fe['YrSold'] - df_fe['YearRemodAdd']
    df_fe['IsNew'] = (df_fe['YrSold'] == df_fe['YearBuilt']).astype(int)
    
    # 5.3 卫生间总数
    df_fe['TotalBath'] = df_fe['FullBath'] + 0.5 * df_fe['HalfBath'] + \
                         df_fe['BsmtFullBath'] + 0.5 * df_fe['BsmtHalfBath']
    
    # 5.4 总面积与房间数比率
    df_fe['SFPerRoom'] = df_fe['GrLivArea'] / (df_fe['TotRmsAbvGrd'] + 1)
    
    # 5.5 是否有特定设施
    df_fe['HasPool'] = (df_fe['PoolArea'] > 0).astype(int)
    df_fe['Has2ndFloor'] = (df_fe['2ndFlrSF'] > 0).astype(int)
    df_fe['HasGarage'] = (df_fe['GarageArea'] > 0).astype(int)
    df_fe['HasBsmt'] = (df_fe['TotalBsmtSF'] > 0).astype(int)
    df_fe['HasFireplace'] = (df_fe['Fireplaces'] > 0).astype(int)
    
    return df_fe

# ==================== 6. 主清洗流程 ====================
def clean_data(file_path, output_path=None):
    """完整数据清洗流程"""
    print("=" * 60)
    print("开始数据清洗流程")
    print("=" * 60)
    
    # 1. 加载数据
    df = load_data(file_path)
    
    # 2. 检查缺失值
    print(f"\n原始数据缺失值统计:")
    missing_stats = df.isnull().sum()
    print(missing_stats[missing_stats > 0].sort_values(ascending=False))
    
    # 3. 处理缺失值
    print("\n" + "-" * 40)
    print("步骤1: 处理缺失值")
    df = handle_missing_values(df)
    
    # 4. 处理异常值
    print("\n" + "-" * 40)
    print("步骤2: 处理异常值")
    df = handle_outliers(df)
    
    # 5. 数据类型优化
    print("\n" + "-" * 40)
    print("步骤3: 优化数据类型")
    df = optimize_dtypes(df)
    
    # 6. 特征工程
    print("\n" + "-" * 40)
    print("步骤4: 特征工程")
    df = feature_engineering(df)
    
    # 7. 最终验证
    print("\n" + "-" * 40)
    print("步骤5: 最终验证")
    print(f"清洗后数据形状: {df.shape}")
    print(f"剩余缺失值: {df.isnull().sum().sum()}")
    print(f"重复行数: {df.duplicated().sum()}")
    
    # 8. 保存清洗后数据
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n清洗后数据已保存至: {output_path}")
    
    print("=" * 60)
    print("数据清洗完成")
    print("=" * 60)
    
    return df

# ==================== 执行 ====================
if __name__ == "__main__":
    INPUT_FILE = "/Users/cjialin/code/AutoMLByLLM/train.csv"
    OUTPUT_FILE = "/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv"
    
    df_cleaned = clean_data(INPUT_FILE, OUTPUT_FILE)
```

## 四、清洗效果验证

### 4.1 预期清洗效果

| 指标 | 清洗前 | 清洗后 | 改善 |
|------|:------:|:------:|------|
| 缺失值总数 | >6,000 | 0 | ✅ 完全消除 |
| 高缺失率列 | 5列 | 0列 | ✅ 删除5列 |
| 数据列数 | 81 | 76+5新特征 | 优化后81列 |
| 内存占用 | ~900KB | ~600KB | ✅ 减少33% |
| SalePrice分布 | 右偏 | 近似正态 | ✅ 改善预测效果 |

### 4.2 验证代码

```python
def validate_cleaning(df_original, df_cleaned):
    """验证清洗效果"""
    print("数据清洗验证报告")
    print("-" * 50)
    
    # 1. 缺失值检查
    missing_before = df_original.isnull().sum().sum()
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"缺失值: {missing_before} → {missing_after} {'✅' if missing_after == 0 else '❌'}")