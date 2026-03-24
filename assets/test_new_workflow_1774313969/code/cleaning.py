import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# 数据路径
input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
output_path = '/Users/cjialin/code/AutoMLByLLM/assets/test_new_workflow_1774313969/data/cleaned_data.csv'

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 加载数据
df = pd.read_csv(input_path)
print(f"原始数据形状: {df.shape}")

# 步骤 1: 删除高缺失率列（>50%）
high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=[col for col in high_missing_cols if col in df.columns])
print(f"删除高缺失率列后形状: {df.shape}")

# 步骤 2: 处理中等缺失率列

# FireplaceQu - 缺失表示无壁炉
if 'FireplaceQu' in df.columns:
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

# LotFrontage - 按Neighborhood分组填充中位数，剩余用全局中位数
if 'LotFrontage' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# Garage相关列
garage_cols_cat = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cols_cat:
    if col in df.columns:
        df[col] = df[col].fillna('None')

if 'GarageYrBlt' in df.columns:
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

# 步骤 3: 处理低缺失率列
if 'BsmtExposure' in df.columns:
    df['BsmtExposure'] = df['BsmtExposure'].fillna('No')
if 'BsmtFinType2' in df.columns:
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna('Unf')
if 'BsmtQual' in df.columns:
    df['BsmtQual'] = df['BsmtQual'].fillna('TA')
if 'BsmtCond' in df.columns:
    df['BsmtCond'] = df['BsmtCond'].fillna('TA')
if 'BsmtFinType1' in df.columns:
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna('Unf')
if 'MasVnrArea' in df.columns:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
if 'Electrical' in df.columns:
    mode_val = df['Electrical'].mode()
    df['Electrical'] = df['Electrical'].fillna(mode_val[0] if not mode_val.empty else 'SBrkr')

# 步骤 4: 异常值处理

# 删除高异常值比例列
outlier_drop_cols = ['BsmtFinSF2', 'EnclosedPorch']
df = df.drop(columns=[col for col in outlier_drop_cols if col in df.columns])
print(f"删除高异常值列后形状: {df.shape}")

# Winsorize处理数值列（排除Id和SalePrice）
def winsorize_series(series, lower_percentile=0.01, upper_percentile=0.99):
    """对序列进行 Winsorize 处理"""
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower=lower, upper=upper)

# 获取数值列（排除Id和SalePrice）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['Id', 'SalePrice']
winsorize_cols = [col for col in numeric_cols if col not in exclude_cols]

for col in winsorize_cols:
    df[col] = winsorize_series(df[col])

# 保存清洗后的数据
df.to_csv(output_path, index=False)
print(f"清洗后数据已保存至: {output_path}")
print(f"清洗后数据形状: {df.shape}")

# 返回清洗结果统计
print("\n清洗结果统计:")
print(f"- 最终数据形状: {df.shape}")
print(f"- 剩余缺失值总数: {df.isnull().sum().sum()}")
print(f"- 数值列数量: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"- 分类列数量: {len(df.select_dtypes(include=['object']).columns)}")

# 检查是否还有缺失值
if df.isnull().sum().sum() > 0:
    print("\n剩余缺失值详情:")
    missing_info = df.isnull().sum()
    print(missing_info[missing_info > 0])
else:
    print("\n✓ 所有缺失值已处理完毕")