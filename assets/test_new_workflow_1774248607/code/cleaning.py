import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

# ==================== 1. 加载数据 ====================
file_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
df = pd.read_csv(file_path)

print(f"原始数据形状: {df.shape}")
print(f"原始缺失值总数: {df.isnull().sum().sum()}")

# ==================== 2. 删除高缺失率列 ====================
# 缺失率 > 50% 的列：PoolQC, MiscFeature, Alley, Fence, MasVnrType
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=cols_to_drop)
print(f"\n删除高缺失列后形状: {df.shape}")

# ==================== 3. 缺失值智能填充 ====================

# 3.1 分类变量 - 按业务逻辑填充"None"（表示无该设施）
none_cols = [
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
]
for col in none_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

# 3.2 数值变量 - MasVnrArea 缺失表示无砌体饰面，填充0
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# 3.3 LotFrontage - 按 Neighborhood（社区）分组中位数填充
# 同社区的地块宽度通常相似
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
# 如果仍有缺失（某些社区全缺失），用整体中位数填充
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# 3.4 GarageYrBlt - 缺失表示无车库，填充房屋建造年份
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

# 3.5 Electrical - 缺失值极少，填充众数（标准断路器）
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

print(f"缺失值填充后总数: {df.isnull().sum().sum()}")

# ==================== 4. 异常值处理（Winsorize 缩尾） ====================
# 对数值特征使用 1%-99% 缩尾，将极端值限制在合理范围内
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF', 'ScreenPorch', 'MiscVal'
]

for col in winsorize_cols:
    if col in df.columns:
        # 使用 1% 和 99% 分位数进行缩尾
        df[col] = winsorize(df[col], limits=[0.01, 0.01])

print(f"异常值处理完成（Winsorize 1%-99%）")

# ==================== 5. 数据类型优化 ====================
# 将分类变量转换为 category 类型，减少内存占用
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

# 过滤实际存在的列（避免删除列后报错）
existing_cat_cols = [col for col in categorical_cols if col in df.columns]
for col in existing_cat_cols:
    df[col] = df[col].astype('category')

print(f"数据类型优化完成，共 {len(existing_cat_cols)} 个分类变量")

# ==================== 6. 特征工程（房价预测专用） ====================

# 6.1 总面积特征（地下室 + 一楼 + 二楼）
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# 6.2 房屋年龄特征
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# 6.3 浴室总数（全浴室计1，半浴室计0.5，包含地下室）
df['TotalBath'] = (
    df['FullBath'] + 0.5 * df['HalfBath'] + 
    df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
)

# 6.4 门廊总面积（所有室外空间）
df['TotalPorchSF'] = (
    df['OpenPorchSF'] + df['EnclosedPorch'] + 
    df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
)

# 6.5 二元特征：是否有特定设施（0/1）
df['HasPool'] = (df['PoolArea'] > 0).astype(int)
df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
df['HasFireplace'] = (df['Fireplaces'] > 0).astype