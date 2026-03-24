import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import os

# 路径设置
input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
output_path = '/Users/cjialin/code/AutoMLByLLM/assets/train_csv_v3_1774344333/data/cleaned_data.csv'

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 加载数据
df = pd.read_csv(input_path)
original_shape = df.shape
original_missing = df.isnull().sum().sum()

print(f"原始数据形状: {original_shape}")
print(f"原始缺失值总数: {original_missing}")

# =================== 步骤 1: 删除高缺失率列 ===================
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df = df.drop(columns=cols_to_drop)
print(f"删除高缺失率列: {cols_to_drop}")

# =================== 步骤 2: 处理缺失值 ===================
# 2.1 分类变量填充 - 业务含义填充
fill_none_cols = ['FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in fill_none_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

# 地下室相关分类变量填充
fill_basement = {
    'BsmtExposure': 'No',
    'BsmtFinType2': 'Unf',
    'BsmtQual': 'TA',
    'BsmtCond': 'TA',
    'BsmtFinType1': 'Unf'
}
for col, val in fill_basement.items():
    if col in df.columns:
        df[col] = df[col].fillna(val)

# Electrical 用众数填充
if 'Electrical' in df.columns:
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# 2.2 数值变量填充
if 'LotFrontage' in df.columns:
    # 按 Neighborhood 中位数填充，剩余用整体中位数
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

if 'GarageYrBlt' in df.columns:
    # 用 YearBuilt 填充（假设车库与房屋同年建造）
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

if 'MasVnrArea' in df.columns:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# =================== 步骤 3: 处理异常值 ===================
# 3.1 删除低信息列（绝大多数为0）
low_info_cols = ['BsmtFinSF2', 'EnclosedPorch']
low_info_cols = [col for col in low_info_cols if col in df.columns]
df = df.drop(columns=low_info_cols)

# 3.2 Winsorize 处理 (5%-95% 缩尾)
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'LowQualFinSF', 'GrLivArea',
    'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch',
    'MiscVal', 'SalePrice'
]

for col in winsorize_cols:
    if col in df.columns:
        df[col] = winsorize(df[col], limits=[0.05, 0.05])

# =================== 步骤 4: 数据类型优化 ===================
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

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# =================== 步骤 5: 验证重复值 ===================
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()

# =================== 保存数据 ===================
df.to_csv(output_path, index=False)

# =================== 清洗结果统计 ===================
final_shape = df.shape
final_missing = df.isnull().sum().sum()

print("\n" + "="*60)
print("数据清洗完成统计")
print("="*60)
print(f"原始数据形状:       {original_shape}")
print(f"清洗后数据形状:     {final_shape}")
print(f"删除列数:           {original_shape[1] - final_shape[1]} 列")
print(f"原始缺失值总数:     {original_missing}")
print(f"剩余缺失值总数:     {final_missing}")
print(f"重复行数:           {duplicates} (已删除)")
print(f"清洗后数据已保存至: {output_path}")
print("="*60)