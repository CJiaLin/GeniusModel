import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os
import warnings
warnings.filterwarnings('ignore')

# 数据路径配置
input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
output_path = '/Users/cjialin/code/AutoMLByLLM/assets/test_new_workflow_1774319996/data/cleaned_data.csv'

# 步骤1：加载数据
df = pd.read_csv(input_path)
original_shape = df.shape
print(f"原始数据形状: {original_shape}")

# 步骤2：删除高缺失率列（>50%）
high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
cols_to_drop = [col for col in high_missing_cols if col in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"删除高缺失率列: {cols_to_drop}")

# 步骤3：删除近零方差列
near_zero_variance_cols = ['BsmtFinSF2', 'EnclosedPorch']
cols_to_drop = [col for col in near_zero_variance_cols if col in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"删除近零方差列: {cols_to_drop}")

# 步骤4：缺失值处理

# 4.1 分类变量缺失填充（NA表示"无此设施"）
bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_cols:
    if col in df.columns:
        df[col] = df[col].fillna('NoBsmt')

garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('NoGarage')

if 'FireplaceQu' in df.columns:
    df['FireplaceQu'] = df['FireplaceQu'].fillna('NoFireplace')

if 'Electrical' in df.columns:
    mode_val = df['Electrical'].mode()
    if len(mode_val) > 0:
        df['Electrical'] = df['Electrical'].fillna(mode_val[0])

# 4.2 数值变量缺失填充
if 'MasVnrArea' in df.columns:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

if 'GarageYrBlt' in df.columns:
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

# LotFrontage使用KNN插值（基于LotArea, OverallQual, OverallCond）
if 'LotFrontage' in df.columns:
    imputer_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond']
    available_cols = [col for col in imputer_cols if col in df.columns]
    if len(available_cols) > 1 and 'LotFrontage' in available_cols:
        imputer_df = df[available_cols].copy()
        knn_imputer = KNNImputer(n_neighbors=5)
        imputed_values = knn_imputer.fit_transform(imputer_df)
        lot_frontage_idx = available_cols.index('LotFrontage')
        df['LotFrontage'] = imputed_values[:, lot_frontage_idx]
        print(f"使用KNN插值填充LotFrontage")

# 步骤5：异常值处理（Winsorize，5%-95%分位数）
def winsorize_series(series, lower_percentile=0.05, upper_percentile=0.95):
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower, upper)

winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
]

winsorized_count = 0
for col in winsorize_cols:
    if col in df.columns:
        df[col] = winsorize_series(df[col])
        winsorized_count += 1
print(f"完成{winsorized_count}列的异常值Winsorize处理")

# 步骤6：数据类型优化（分类变量转category）
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    df[col] = df[col].astype('category')

if 'MSSubClass' in df.columns:
    df['MSSubClass'] = df['MSSubClass'].astype('category')

# 步骤7：保存清洗后的数据
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

# 步骤8：清洗结果统计
print("\n" + "="*50)
print("数据清洗完成统计")
print("="*50)
print(f"原始数据形状: {original_shape}")
print(f"清洗后数据形状: {df.shape}")
print(f"列数变化: {original_shape[1]} -> {df.shape[1]} (减少{original_shape[1] - df.shape[1]}列)")

# 检查剩余缺失值
missing_after = df.isnull().sum()
missing_cols = missing_after[missing_after > 0]
print(f"\n剩余缺失值列数: {len(missing_cols)}")
if len(missing_cols) > 0:
    print("剩余缺失值分布:")
    print(missing_cols)
else:
    print("所有缺失值已处理完毕")

print(f"\n数据类型分布:")
print(df.dtypes.value_counts())

print(f"\n清洗后数据已保存至: {output_path}")