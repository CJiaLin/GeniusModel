import pandas as pd
import numpy as np
import os
from pathlib import Path

# 设置数据路径
input_path = "/Users/cjialin/code/AutoMLByLLM/train.csv"
output_path = "/Users/cjialin/code/AutoMLByLLM/assets/test_new_workflow_1774318847/data/cleaned_data.csv"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 加载数据
print("正在加载数据...")
df = pd.read_csv(input_path)
original_shape = df.shape
print(f"原始数据形状: {original_shape}")

# 1. 删除高缺失率列 (>50%)
high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=[col for col in high_missing_cols if col in df.columns])
print(f"删除高缺失列后: {df.shape}")

# 2. 删除低方差列 (接近零方差)
low_variance_cols = ['BsmtFinSF2', 'EnclosedPorch', 'LowQualFinSF', 
                     'BsmtHalfBath', 'KitchenAbvGr', '3SsnPorch', 'MiscVal']
df = df.drop(columns=[col for col in low_variance_cols if col in df.columns])
print(f"删除低方差列后: {df.shape}")

# 3. 填充缺失值
# 3.1 分类变量：表示"无此特征"
no_feature_mappings = {
    'FireplaceQu': 'NoFireplace',
    'GarageType': 'NoGarage',
    'GarageFinish': 'NoGarage',
    'GarageQual': 'NoGarage',
    'GarageCond': 'NoGarage',
    'BsmtExposure': 'NoBsmt',
    'BsmtFinType2': 'NoBsmt',
    'BsmtQual': 'NoBsmt',
    'BsmtCond': 'NoBsmt',
    'BsmtFinType1': 'NoBsmt'
}

for col, fill_val in no_feature_mappings.items():
    if col in df.columns:
        df[col] = df[col].fillna(fill_val)

# 3.2 GarageYrBlt: 无车库时填充为房屋建造年份
if 'GarageYrBlt' in df.columns:
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

# 3.3 LotFrontage: 按Neighborhood分组填充中位数
if 'LotFrontage' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    # 若仍有缺失，用全局中位数填充
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# 3.4 MasVnrArea: 缺失视为无砌体饰面，填充0
if 'MasVnrArea' in df.columns:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# 3.5 Electrical: 用众数填充
if 'Electrical' in df.columns:
    df['Electrical'] = df['Electrical'].fillna('SBrkr')

# 4. Winsorize处理异常值 (限制在1%-99%分位数)
winsorize_cols = {
    'MSSubClass': (0.01, 0.99),
    'LotFrontage': (0.01, 0.99),
    'LotArea': (0.01, 0.99),
    'OverallCond': (0.01, 0.99),
    'MasVnrArea': (0, 0.99),
    'BsmtUnfSF': (0.01, 0.99),
    'TotalBsmtSF': (0.01, 0.99),
    '1stFlrSF': (0.01, 0.99),
    'GrLivArea': (0.01, 0.99),
    'TotRmsAbvGrd': (0.01, 0.99),
    'GarageArea': (0.01, 0.99),
    'WoodDeckSF': (0, 0.99),
    'OpenPorchSF': (0, 0.99),
    'ScreenPorch': (0, 0.99),
    'BedroomAbvGr': (0, 0.99),
    'SalePrice': (0.01, 0.99)
}

for col, (lower, upper) in winsorize_cols.items():
    if col in df.columns:
        lower_val = df[col].quantile(lower)
        upper_val = df[col].quantile(upper)
        df[col] = df[col].clip(lower=lower_val, upper=upper_val)

# 保存清洗后的数据
df.to_csv(output_path, index=False)

# 输出清洗结果统计
print("\n" + "="*50)
print("数据清洗完成统计")
print("="*50)
print(f"原始数据形状: {original_shape}")
print(f"清洗后数据形状: {df.shape}")
print(f"删除的列数: {original_shape[1] - df.shape[1]}")
print(f"剩余缺失值总数: {df.isnull().sum().sum()}")
print(f"数值列数量: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"分类列数量: {df.select_dtypes(include=['object']).shape[1]}")
print(f"清洗后数据已保存至: {output_path}")
print("="*50)