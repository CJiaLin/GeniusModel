import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
output_path = '/Users/cjialin/code/AutoMLByLLM/assets/test_new_workflow_1774263262/data/cleaned_data.csv'

df = pd.read_csv(input_path)
original_shape = df.shape
print(f"原始数据形状: {original_shape}")

# 2. 删除高缺失率列（>50%）
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# 3. 删除低信息列（零值占比过高）
cols_drop_outlier = ['BsmtFinSF2', 'EnclosedPorch']
df = df.drop(columns=[col for col in cols_drop_outlier if col in df.columns])

# 4. 缺失值智能填充

# 4.1 地下室相关特征（缺失表示无地下室）
bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
bsmt_num_cols = ['BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

for col in bsmt_cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

for col in bsmt_num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# 4.2 车库相关特征（缺失表示无车库）
garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
garage_num_cols = ['GarageYrBlt', 'GarageCars', 'GarageArea']

for col in garage_cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

if 'GarageYrBlt' in df.columns:
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

for col in ['GarageCars', 'GarageArea']:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# 4.3 其他特征填充
if 'LotFrontage' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

if 'FireplaceQu' in df.columns:
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

if 'MasVnrArea' in df.columns:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

if 'Electrical' in df.columns:
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# 5. 异常值处理 - Winsorize限幅（5%-95%分位数）
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr', 
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF', 
    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
]

for col in winsorize_cols:
    if col in df.columns and df[col].dtype in ['int64', 'float64']:
        lower = df[col].quantile(0.05)
        upper = df[col].quantile(0.95)
        df[col] = np.clip(df[col], lower, upper)

# 6. 数据类型优化（分类变量转换）
cat_cols = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond',
    'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
    'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'
]

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# 7. 保存清洗后的数据
df.to_csv(output_path, index=False)

# 8. 输出清洗结果统计
print("\n=== 数据清洗完成 ===")
print(f"原始数据形状: {original_shape}")
print(f"清洗后数据形状: {df.shape}")
print(f"删除列数: {original_shape[1] - df.shape[1]}")
print(f"剩余缺失值总数: {df.isnull().sum().sum()}")
print(f"数值列数量: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"分类列数量: {len(df.select_dtypes(include=['category']).columns)}")
print(f"行数变化: {original_shape[0]} -> {df.shape[0]} (无变化)")
print(f"\n清洗后数据已保存至: {output_path}")