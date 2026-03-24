import pandas as pd
import numpy as np
import os

# 设置路径
input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
output_path = '/Users/cjialin/code/AutoMLByLLM/assets/train_csv_v3_1774340366/data/cleaned_data.csv'

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 加载数据
df = pd.read_csv(input_path)
original_shape = df.shape
original_memory = df.memory_usage(deep=True).sum() / 1024**2

print(f"原始数据形状: {original_shape}")
print(f"原始缺失值总数: {df.isnull().sum().sum()}")

# 阶段1: 删除缺失率超过50%的列
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df = df.drop(columns=existing_cols_to_drop)
print(f"阶段1完成: 删除 {len(existing_cols_to_drop)} 列，剩余 {df.shape[1]} 列")

# 阶段2: 缺失值填充

# 2.1 FireplaceQu - 填充为"None"表示无壁炉
if 'FireplaceQu' in df.columns:
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

# 2.2 LotFrontage - 按Neighborhood分组中位数填充，剩余用全局中位数
if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# 2.3 Garage相关列 - 分类列填充为"None"，年份列用YearBuilt填充
garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

if 'GarageYrBlt' in df.columns:
    if 'YearBuilt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    else:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())

# 2.4 Bsmt相关列 - 填充为"None"表示无地下室
bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

# 2.5 MasVnrArea - 用0填充（无砌体贴面）
if 'MasVnrArea' in df.columns:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# 2.6 Electrical - 用众数填充
if 'Electrical' in df.columns:
    mode_val = df['Electrical'].mode()
    if len(mode_val) > 0:
        df['Electrical'] = df['Electrical'].fillna(mode_val[0])

print(f"阶段2完成: 缺失值填充后剩余缺失值 {df.isnull().sum().sum()}")

# 阶段3: Winsorize异常值（1%和99%分位数缩尾）
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
]

for col in winsorize_cols:
    if col in df.columns:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

print(f"阶段3完成: Winsorize处理了 {len([col for col in winsorize_cols if col in df.columns])} 列")

# 阶段4: 数据类型优化 - 将分类变量转换为category类型
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

# 只转换实际存在的列
existing_cat_cols = [col for col in categorical_cols if col in df.columns]
for col in existing_cat_cols:
    df[col] = df[col].astype('category')

print(f"阶段4完成: 转换了 {len(existing_cat_cols)} 列为category类型")

# 阶段5: 保存清洗后的数据
df.to_csv(output_path, index=False)

# 验证与统计
final_memory = df.memory_usage(deep=True).sum() / 1024**2
memory_saved = original_memory - final_memory

print("\n" + "=" * 50)
print("数据清洗完成 - 统计报告")
print("=" * 50)
print(f"原始数据形状: {original_shape}")
print(f"清洗后数据形状: {df.shape}")
print(f"删除的列数: {len(existing_cols_to_drop)} ({existing_cols_to_drop})")
print(f"剩余缺失值总数: {df.isnull().sum().sum()}")
print(f"重复行数: {df.duplicated().sum()}")
print(f"原始内存使用: {original_memory:.2f} MB")
print(f"清洗后内存使用: {final_memory:.2f} MB")
print(f"内存节省: {memory_saved:.2f} MB")
print(f"\n清洗后的数据已保存至: {output_path}")