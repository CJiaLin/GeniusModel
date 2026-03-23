# 房价预测数据清洗方案

## 1. 项目背景

**任务类型**: 房价预测回归任务  
**目标变量**: `SalePrice` (房屋售价)  
**评估指标**: RMSE (均方根误差)  
**数据规模**: 1,460 行 × 81 列

## 2. 数据质量问题总结

基于实际数据分析，发现以下主要质量问题：

| 问题类别 | 具体问题 | 影响程度 |
|---------|---------|---------|
| **高缺失率列** | PoolQC(99.52%)、MiscFeature(96.3%)、Alley(93.77%)、Fence(80.75%)、MasVnrType(59.73%) | 🔴 严重 |
| **中等缺失率** | FireplaceQu(47.26%)、LotFrontage(17.74%)、Garage相关列(5.55%) | 🟡 中等 |
| **低缺失率** | Bsmt相关列(~2.5%)、MasVnrArea(0.55%)、Electrical(0.07%) | 🟢 轻微 |
| **异常值** | MSSubClass、LotFrontage、LotArea、OverallCond等30个数值列 | 🟡 中等 |
| **数据类型** | 43个分类变量未转换为category类型 | 🟢 轻微 |

## 3. 清洗策略

### 3.1 缺失值处理策略

| 列名 | 缺失率 | 处理策略 | 理由 |
|-----|-------|---------|-----|
| PoolQC, MiscFeature, Alley, Fence, MasVnrType | >50% | **删除列** | 缺失率过高，信息含量低，填充会引入噪声 |
| FireplaceQu | 47.26% | 填充为"None" | 缺失表示无壁炉，属于有意义的缺失 |
| LotFrontage | 17.74% | 按Neighborhood分组中位数填充 | 同街区的房屋临街距离相似 |
| GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond | 5.55% | 类型列填充"None"，年份填充0 | 缺失表示无车库 |
| BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2 | ~2.5% | 填充为"None" | 缺失表示无地下室 |
| MasVnrArea | 0.55% | 填充为0 | 缺失表示无砌体贴面 |
| Electrical | 0.07% | 填充为众数"SBrkr" | 仅1个缺失，标准接线方式最常见 |

### 3.2 异常值处理策略

采用 **Winsorize（缩尾处理）**，将异常值限制在 [Q1-1.5×IQR, Q3+1.5×IQR] 范围内：

**需要Winsorize的列**:
- `MSSubClass`, `LotFrontage`, `LotArea`, `OverallCond`
- `MasVnrArea`, `BsmtUnfSF`, `TotalBsmtSF`, `1stFlrSF`
- `LowQualFinSF`, `GrLivArea`, `BsmtHalfBath`, `BedroomAbvGr`
- `KitchenAbvGr`, `TotRmsAbvGrd`, `GarageArea`
- `WoodDeckSF`, `OpenPorchSF`, `3SsnPorch`, `ScreenPorch`, `MiscVal`
- `SalePrice` (目标变量)

**需要删除的列** (异常值比例过高或信息价值低):
- `BsmtFinSF2` (11.44%异常值且多数为0)
- `EnclosedPorch` (14.25%异常值且多数为0)

**保留的列** (异常值具有业务意义):
- `OverallQual`, `YearBuilt`, `BsmtFinSF1`, `2ndFlrSF`, `BsmtFullBath`
- `Fireplaces`, `GarageCars`, `PoolArea`

### 3.3 数据类型优化

将43个object类型的分类变量转换为 `category` 类型，减少内存占用并明确数据语义。

## 4. 清洗代码实现

```python
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 配置参数
# ============================================
DATA_PATH = '/Users/cjialin/code/AutoMLByLLM/train.csv'
OUTPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'

# 高缺失率列（需要删除）
COLS_TO_DROP = [
    'PoolQC',      # 99.52% 缺失
    'MiscFeature', # 96.30% 缺失  
    'Alley',       # 93.77% 缺失
    'Fence',       # 80.75% 缺失
    'MasVnrType'   # 59.73% 缺失
]

# 异常值比例过高需要删除的列
COLS_DROP_OUTLIER = [
    'BsmtFinSF2',    # 11.44% 异常值
    'EnclosedPorch'  # 14.25% 异常值
]

# 需要Winsorize的数值列（基于实际数据异常值分析）
NUM_COLS_TO_WINSORIZE = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch',
    'MiscVal', 'SalePrice'
]

# 分类变量列表（基于实际数据列名）
CATEGORICAL_COLS = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
    'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PavedDrive', 'SaleType', 'SaleCondition'
]

# ============================================
# 主清洗流程
# ============================================

def load_and_inspect_data(filepath):
    """加载并检查数据"""
    df = pd.read_csv(filepath)
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    return df

def remove_high_missing_columns(df):
    """删除高缺失率列"""
    df_clean = df.drop(columns=COLS_TO_DROP, errors='ignore')
    print(f"删除高缺失率列后形状: {df_clean.shape}")
    return df_clean

def remove_outlier_columns(df):
    """删除异常值过多的列"""
    df_clean = df.drop(columns=COLS_DROP_OUTLIER, errors='ignore')
    print(f"删除高异常值列后形状: {df_clean.shape}")
    return df_clean

def impute_missing_values(df):
    """
    智能填充缺失值
    基于房价数据的业务理解
    """
    df_imputed = df.copy()
    
    # 1. FireplaceQu: 缺失表示无壁炉
    if 'FireplaceQu' in df_imputed.columns:
        df_imputed['FireplaceQu'] = df_imputed['FireplaceQu'].fillna('None')
    
    # 2. LotFrontage: 按Neighborhood分组中位数填充
    if 'LotFrontage' in df_imputed.columns:
        df_imputed['LotFrontage'] = df_imputed.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果仍有缺失，用全局中位数填充
        df_imputed['LotFrontage'] = df_imputed['LotFrontage'].fillna(df_imputed['LotFrontage'].median())
    
    # 3. Garage相关列
    garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cat_cols:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna('None')
    
    if 'GarageYrBlt' in df_imputed.columns:
        # 缺失表示无车库，填充为0或与房屋建造年份相同
        df_imputed['GarageYrBlt'] = df_imputed['GarageYrBlt'].fillna(0)
    
    # 4. Bsmt（地下室）相关列
    bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in bsmt_cat_cols:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna('None')
    
    # 5. MasVnrArea: 缺失表示无砌体贴面
    if 'MasVnrArea' in df_imputed.columns:
        df_imputed['MasVnrArea'] = df_imputed['MasVnrArea'].fillna(0)
    
    # 6. Electrical: 用众数填充
    if 'Electrical' in df_imputed.columns:
        df_imputed['Electrical'] = df_imputed['Electrical'].fillna('SBrkr')
    
    return df_imputed

def winsorize_outliers(df):
    """
    对数值列进行Winsorize处理
    限制极端值的影响，但保留数据点
    """
    df_winsorized = df.copy()
    
    for col in NUM_COLS_TO_WINSORIZE:
        if col in df_winsorized.columns and pd.api.types.is_numeric_dtype(df_winsorized[col]):
            # 计算IQR边界
            Q1 = df_winsorized[col].quantile(0.25)
            Q3 = df_winsorized[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Winsorize: 将极值限制在边界内
            df_winsorized[col] = df_winsorized[col].clip(lower=lower_bound, upper=upper_bound)
            
            # 记录处理信息
            original_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            print(f"  {col}: 处理了 {original_outliers} 个异常值")
    
    return df_winsorized

def optimize_data_types(df):
    """优化数据类型"""
    df_optimized = df.copy()
    
    # 将分类变量转换为category类型
    for col in CATEGORICAL_COLS:
        if col in df_optimized.columns:
            df_optimized[col] = df_optimized[col].astype('category')
    
    # 将整数类型优化为更小的类型
    int_cols = df_optimized.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if col != 'Id':  # 保留Id的原始类型
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            if c_min >= 0:
                if c_max < 255:
                    df_optimized[col] = df_optimized[col].astype(np.uint8)
                elif c_max < 65535:
                    df_optimized[col] = df_optimized[col].astype(np.uint16)
    
    return df_optimized

def create_additional_features(df):
    """
    创建派生特征（特征工程）
    基于房价数据的领域知识
    """
    df_featured = df.copy()
    
    # 1. 房屋总面积
    if all(col in df_featured.columns for col in ['1stFlrSF', '2ndFlrSF', 'LowQualFinSF']):
        df_featured['TotalSF'] = df_featured['1stFlrSF'] + df_featured['2ndFlrSF'] + df_featured['LowQualFinSF']
    
    # 2. 总浴室数
    if all(col in df_featured.columns for col in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
        df_featured['TotalBath'] = df_featured['FullBath'] + 0.5 * df_featured['HalfBath'] + \
                                   df_featured['BsmtFullBath'] + 0.5 * df_featured['BsmtHalfBath']
    
    # 3. 房屋年龄
    if all(col in df_featured.columns for col in ['YrSold', 'YearBuilt']):
        df_featured['HouseAge'] = df_featured['YrSold'] - df_featured['YearBuilt']
    
    # 4. 装修后年数
    if all(col in df_featured.columns for col in ['YrSold', 'YearRemodAdd']):
        df_featured['RemodAge'] = df_featured['YrSold'] - df_featured['YearRemodAdd']
    
    # 5. 是否有车库
    if 'GarageArea' in df_featured.columns:
        df_featured['HasGarage'] = (df_featured['GarageArea'] > 0).astype(int)
    
    # 6. 是否有地下室
    if 'TotalBsmtSF' in df_featured.columns:
        df_featured['HasBsmt'] = (df_featured['TotalBsmtSF'] > 0).astype(int)
    
    # 7. 是否有壁炉
    if 'Fireplaces' in df_featured.columns:
        df_featured['HasFireplace'] = (df_featured['Fireplaces'] > 0).astype(int)
    
    # 8. 是否有泳池
    if 'PoolArea' in df_featured.columns:
        df_featured['HasPool'] = (df_featured['PoolArea'] > 0).astype(int)
    
    # 9. 是否有2楼
    if '2ndFlrSF' in df_featured.columns:
        df_featured['Has2ndFloor'] = (df_featured['2ndFlrSF'] > 0).astype(int)
    
    # 10. 户外空间总面积
    porch_cols = ['WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch']
    available_porch_cols = [col for col in porch_cols if col in df_featured.columns]
    if available_porch_cols:
        df_featured['TotalPorchSF'] = df_featured[available_porch_cols].sum(axis=1)
    
    return df_featured

def validate_cleaning(df_original, df_cleaned):
    """验证清洗效果"""
    print("\n" + "="*50)
    print("数据清洗验证报告")
    print("="*50)
    
    # 1. 形状对比
    print(f"\n原始数据形状: {df_original.shape}")
    print(f"清洗后形状: {df_cleaned.shape}")
    
    # 2. 缺失值对比
    missing_before = df_original.isnull().sum().sum()
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"\n缺失值总数: {missing_before} -> {missing_after}")
    
    # 3. 数值列统计对比
    num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    print(f"\n数值列数量: {len(num_cols)}")
    
    # 4. 目标变量SalePrice分布
    if 'SalePrice' in df_cleaned.columns:
        print(f"\n目标变量SalePrice统计:")
        print(f"  均值: ${df_cleaned['SalePrice'].mean():,.2f}")
        print(f"  中位数: ${df_cleaned['SalePrice'].median():,.2f}")
        print(f"  标准差: ${df_cleaned['SalePrice'].std():,.2f}")
        print(f"  最小值: ${df_cleaned['SalePrice'].min():,.2f}")
        print(f"  最大值: ${df_cleaned['SalePrice'].max():,.2f}")
    
    # 5. 内存使用对比
    mem_before = df_original.memory_usage(deep=True).sum() / 1024**2
    mem_after = df_cleaned.memory_usage(deep=True).sum() / 1024**2
    print(f"\n内存使用: {mem_before:.2f