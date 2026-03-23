import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 数据加载 ====================
file_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
output_path = '/Users/cjialin/code/AutoMLByLLM/assets/test_new_workflow_1774258600/data/cleaned_data.csv'

df = pd.read_csv(file_path)
print(f"原始数据形状: {df.shape}")

# ==================== 2. 删除高缺失率列 (>50%) ====================
high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=[col for col in high_missing_cols if col in df.columns])
print(f"删除高缺失列后: {df.shape}")

# ==================== 3. 缺失值填充 ====================

# 3.1 分类变量填充（基于业务逻辑）
if 'FireplaceQu' in df.columns:
    df['FireplaceQu'] = df['FireplaceQu'].fillna('NoFireplace')

# Garage相关列
garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('NoGarage')

# Basement相关列
bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('NoBasement')

# Electrical填充众数
if 'Electrical' in df.columns:
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# 3.2 数值变量填充
# LotFrontage按Neighborhood分组中位数填充
if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# GarageYrBlt填充0（表示无车库）
if 'GarageYrBlt' in df.columns:
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

# MasVnrArea填充0
if 'MasVnrArea' in df.columns:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

print(f"缺失值填充后剩余缺失值: {df.isnull().sum().sum()}")

# ==================== 4. 异常值处理 ====================

# 4.1 删除异常值过多的列
outlier_cols_drop = ['BsmtFinSF2', 'EnclosedPorch']
df = df.drop(columns=[col for col in outlier_cols_drop if col in df.columns], errors='ignore')

# 4.2 Winsorize处理（限制在1%-99%分位数）
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal'
]

for col in winsorize_cols:
    if col in df.columns:
        df[col] = winsorize(df[col], limits=[0.01, 0.01])

# ==================== 5. 数据类型转换 ====================
# 将分类列转换为category类型（基于数据信息中的分类列）
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

existing_cat_cols = [col for col in categorical_cols if col in df.columns]
df[existing_cat_cols] = df[existing_cat_cols].astype('category')

# ==================== 6. 特征工程 ====================

# 6.1 总面积特征
if all(col in df.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# 6.2 房龄特征
if all(col in df.columns for col in ['YrSold', 'YearBuilt', 'YearRemodAdd']):
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# 6.3 浴室总数
bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
if all(col in df.columns for col in bath_cols):
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                       df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])

# 6.4 设施二元特征
if 'PoolArea' in df.columns:
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
if 'GarageArea' in df.columns:
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
if 'TotalBsmtSF' in df.columns:
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
if 'Fireplaces' in df.columns:
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
if '2ndFlrSF' in df.columns:
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)

# ==================== 7. 保存与验证 ====================
df.to_csv(output_path, index=False)

# 输出统计信息
print(f"\n清洗完成！")
print(f"最终数据形状: {df.shape}")
print(f"数值列数量: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"分类列数量: {len(df.select_dtypes(include=['category']).columns)}")
print(f"新生成特征: {[col for col in df.columns if col in ['TotalSF', 'HouseAge', 'RemodAge', 'TotalBath', 'HasPool', 'HasGarage', 'HasBasement', 'HasFireplace', 'Has2ndFloor']]}")
print(f"数据已保存至: {output_path}")

# 检查剩余缺失值
final_missing = df.isnull().sum()
final_missing = final_missing[final_missing > 0]
if len(final_missing) > 0:
    print(f"\n剩余缺失值列:\n{final_missing}")
else:
    print("\n✅ 所有缺失值已处理完毕")