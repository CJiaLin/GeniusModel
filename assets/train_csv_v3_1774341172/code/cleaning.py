import pandas as pd
import numpy as np
import os

# 设置数据路径
input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
output_path = '/Users/cjialin/code/AutoMLByLLM/assets/train_csv_v3_1774341172/data/cleaned_data.csv'

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 1. 加载数据
df = pd.read_csv(input_path)
original_shape = df.shape
print(f"原始数据形状: {original_shape}")

# 2. 删除高缺失率列（>50%）
high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
high_missing_cols = [col for col in high_missing_cols if col in df.columns]
df = df.drop(columns=high_missing_cols)
print(f"删除高缺失率列后: {df.shape}")

# 3. 中等缺失率列智能填充
if 'FireplaceQu' in df.columns:
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

if 'LotFrontage' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

if 'GarageYrBlt' in df.columns:
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

# 4. 低缺失率列填充
bsmt_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
for col in bsmt_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

if 'MasVnrArea' in df.columns:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

if 'Electrical' in df.columns:
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# 5. 删除异常值列（零方差/高异常值）
cols_to_drop_outlier = ['BsmtFinSF2', 'EnclosedPorch']
cols_to_drop_outlier = [col for col in cols_to_drop_outlier if col in df.columns]
df = df.drop(columns=cols_to_drop_outlier)
print(f"删除异常值列后: {df.shape}")

# 6. Winsorize处理（5%-95%分位数截断）
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr', 
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF', 
    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal'
]

for col in winsorize_cols:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        lower_bound = df[col].quantile(0.05)
        upper_bound = df[col].quantile(0.95)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

print("Winsorize处理完成")

# 7. 数据类型优化（分类变量转category）
categorical_cols = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 
    'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 
    'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 
    'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 
    'SaleCondition'
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

print("数据类型转换完成")

# 8. 重复值检查与处理
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"删除重复行后: {df.shape}")

# 9. 保存清洗后的数据
df.to_csv(output_path, index=False)
print(f"\n清洗后的数据已保存至: {output_path}")

# 10. 清洗结果统计
final_shape = df.shape
remaining_missing = df.isnull().sum().sum()
missing_cols = df.columns[df.isnull().any()].tolist()
num_numeric = df.select_dtypes(include=[np.number]).shape[1]
num_categorical = df.select_dtypes(include=['category']).shape[1]
memory_usage = df.memory_usage(deep=True).sum() / 1024**2

print("\n" + "="*50)
print("数据清洗结果统计")
print("="*50)
print(f"原始数据形状: {original_shape}")
print(f"清洗后数据形状: {final_shape}")
print(f"删除列数: {original_shape[1] - final_shape[1]}")
print(f"剩余缺失值总数: {remaining_missing}")
if missing_cols:
    print(f"仍有缺失值的列: {missing_cols}")
else:
    print("所有缺失值已处理完毕")
print(f"数值列数量: {num_numeric}")
print(f"分类列数量: {num_categorical}")
print(f"内存使用: {memory_usage:.2f} MB")
print(f"重复行数: {duplicate_count}")
print("="*50)