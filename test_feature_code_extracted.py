import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 分离目标变量
target = df['SalePrice'].copy()
df = df.drop('SalePrice', axis=1)

# 1. 缺失值处理
# 数值列用中位数填充
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# 分类列用'None'填充
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna('None', inplace=True)

# 2. 创建面积相关特征
# 总面积（地下室 + 第一层 + 第二层）
if 'TotalBsmtSF' in df.columns and '1stFlrSF' in df.columns and '2ndFlrSF' in df.columns:
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# 总门廊面积
porch_cols = ['OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF']
existing_porch_cols = [col for col in porch_cols if col in df.columns]
if existing_porch_cols:
    df['TotalPorchSF'] = df[existing_porch_cols].sum(axis=1)

# 是否有车库
df['HasGarage'] = (df['GarageArea'] > 0).astype(int) if 'GarageArea' in df.columns else 0

# 是否有泳池
df['HasPool'] = (df['PoolArea'] > 0).astype(int) if 'PoolArea' in df.columns else 0

# 是否有地下室
df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int) if 'TotalBsmtSF' in df.columns else 0

# 是否有壁炉
df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int) if 'Fireplaces' in df.columns else 0

# 3. 创建房龄特征
if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)

if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['HasRemod'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int) if 'YearBuilt' in df.columns else 0

# 4. 质量评分编码（将分类质量等级转换为数值）
quality_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, 'NA': 0}

quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

for col in quality_cols:
    if col in df.columns:
        df[col] = df[col].map(quality_mapping).fillna(0)

# 其他有用的特征
# 每平方英尺价格（在训练集中使用实际价格，这里先不计算，保留原始特征）
# 卫生间总数
if 'FullBath' in df.columns and 'HalfBath' in df.columns:
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']

if 'BsmtFullBath' in df.columns and 'BsmtHalfBath' in df.columns:
    df['TotalBsmtBath'] = df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']

# 将所有分类变量转换为数值（独热编码）
df = pd.get_dummies(df, drop_first=True)

# 添加目标变量回数据集
df['SalePrice'] = target

# 保存结果
df.to_csv('/Users/cjialin/code/AutoMLByLLM/train_features.csv', index=False)

print(f"特征工程完成！")
print(f"原始特征数: 80（不含目标变量）")
print(f"处理后特征数: {df.shape[1] - 1}（不含目标变量）")
print(f"数据形状: {df.shape}")
print(f"结果已保存至: /Users/cjialin/code/AutoMLByLLM/train_features.csv")