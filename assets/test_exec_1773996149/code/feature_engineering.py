import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 数据路径配置
data_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned_features.csv'

# 加载数据
print('正在加载数据...')
df = pd.read_csv(data_path)
original_cols = list(df.columns)
print(f'原始数据形状: {df.shape}')
print(f'原始列数: {len(original_cols)}')

# 记录新生成的特征
new_features = []

# ==================== 1. 缺失值处理 ====================
print('\n正在处理缺失值...')

# LotFrontage: 按Neighborhood分组中位数填充
if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    print('  - LotFrontage已按Neighborhood中位数填充')

# GarageYrBlt: 用YearBuilt填充（假设无车库则与房屋同年建造）
if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    print('  - GarageYrBlt已用YearBuilt填充')

# 高缺失类别特征: 创建二元指示特征并填充"None"
high_missing_categorical = {
    'PoolQC': 'HasPool_QC',
    'MiscFeature': 'HasMiscFeature', 
    'Alley': 'HasAlley',
    'Fence': 'HasFence',
    'FireplaceQu': 'HasFireplace_Q'
}

for col, indicator_name in high_missing_categorical.items():
    if col in df.columns:
        df[indicator_name] = df[col].notna().astype(int)
        new_features.append(indicator_name)
        df[col] = df[col].fillna('None')
        print(f'  - {col}已创建指示特征{indicator_name}并填充None')

# MasVnr相关特征填充
if 'MasVnrArea' in df.columns:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
if 'MasVnrType' in df.columns:
    df['MasVnrType'] = df['MasVnrType'].fillna('None')

# Bsmt相关特征填充（如果存在）
bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

# Garage相关特征填充
garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

# GarageArea, GarageCars填充0
if 'GarageArea' in df.columns:
    df['GarageArea'] = df['GarageArea'].fillna(0)
if 'GarageCars' in df.columns:
    df['GarageCars'] = df['GarageCars'].fillna(0)

# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF填充0
bsmt_sf_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
for col in bsmt_sf_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# ==================== 2. 面积特征工程 ====================
print('\n正在创建面积特征...')

# TotalSF: 总居住面积 = 地下室 + 一层 + 二层
if 'TotalBsmtSF' in df.columns and '1stFlrSF' in df.columns and '2ndFlrSF' in df.columns:
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    new_features.append('TotalSF')
    print('  - 已创建TotalSF')

# TotalPorchSF: 总室外面积
porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']
available_porch = [c for c in porch_cols if c in df.columns]
if available_porch:
    df['TotalPorchSF'] = df[available_porch].sum(axis=1)
    new_features.append('TotalPorchSF')
    print('  - 已创建TotalPorchSF')

# 二元指示特征
if 'PoolArea' in df.columns:
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    new_features.append('HasPool')
    print('  - 已创建HasPool')

if '2ndFlrSF' in df.columns:
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    new_features.append('Has2ndFloor')
    print('  - 已创建Has2ndFloor')

if 'GarageArea' in df.columns:
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    new_features.append('HasGarage')
    print('  - 已创建HasGarage')

if 'TotalBsmtSF' in df.columns:
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    new_features.append('HasBasement')
    print('  - 已创建HasBasement')

if 'Fireplaces' in df.columns:
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    new_features.append('HasFireplace')
    print('  - 已创建HasFireplace')

# AvgRoomSize: 平均房间面积
if 'GrLivArea' in df.columns and 'TotRmsAbvGrd' in df.columns:
    df['AvgRoomSize'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)
    new_features.append('AvgRoomSize')
    print('  - 已创建AvgRoomSize')

# BathToRoomRatio: 卫生间房间比
if 'FullBath' in df.columns and 'HalfBath' in df.columns and 'TotRmsAbvGrd' in df.columns:
    total_bath = df['FullBath'] + 0.5 * df['HalfBath']
    df['BathToRoomRatio'] = total_bath / (df['TotRmsAbvGrd'] + 1)
    new_features.append('BathToRoomRatio')
    print('  - 已创建BathToRoomRatio')

# BedroomRatio: 卧室占比
if 'BedroomAbvGr' in df.columns and 'TotRmsAbvGrd' in df.columns:
    df['BedroomRatio'] = df['BedroomAbvGr'] / (df['TotRmsAbvGrd'] + 1)
    new_features.append('BedroomRatio')
    print('  - 已创建BedroomRatio')

# LotUtilization: 土地利用率
if 'GrLivArea' in df.columns and 'LotArea' in df.columns:
    df['LotUtilization'] = df['GrLivArea'] / (df['LotArea'] + 1)
    new_features.append('LotUtilization')
    print('  - 已创建LotUtilization')

# BsmtFinRatio: 地下室完成度
if 'BsmtFinSF1' in df.columns and 'TotalBsmtSF' in df.columns:
    df['BsmtFinRatio'] = df['BsmtFinSF1'] / (df['TotalBsmtSF'] + 1)
    new_features.append('BsmtFinRatio')
    print('  - 已创建BsmtFinRatio')

# ==================== 3. 时间特征工程 ====================
print('\n正在创建时间特征...')

if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    new_features.append('HouseAge')
    print('  - 已创建HouseAge')

if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    new_features.append('RemodAge')
    print('  - 已创建RemodAge')

if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
    df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)
    new_features.append('IsNew')
    print('  - 已创建IsNew')

if 'YrSold' in df.columns and 'GarageYrBlt' in df.columns:
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    new_features.append('GarageAge')
    print('  - 已创建GarageAge')

# ==================== 4. 质量评分特征 ====================
print('\n正在创建质量评分特征...')

if 'OverallQual' in df.columns and 'OverallCond' in df.columns:
    df['OverallScore'] = df['OverallQual'] * df['OverallCond']
    new_features.append('OverallScore')
    print('  - 已创建OverallScore')

# 质量等级映射
quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

# 对质量等级特征进行编码
qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
for col in qual_cols:
    if col in df.columns:
        encoded_col = col + '_Encoded'
        df[encoded_col] = df[col].map(quality_map).fillna(0)
        if encoded_col not in original_cols:
            new_features.append(encoded_col)
        print(f'  - 已创建{encoded_col}')

# BsmtExposure编码
if 'BsmtExposure' in df.columns:
    exposure_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    df['BsmtExposure_Encoded'] = df['BsmtExposure'].map(exposure_map).fillna(0)
    new_features.append('BsmtExposure_Encoded')
    print('  - 已创建BsmtExposure_Encoded')

# CentralAir二元编码
if 'CentralAir' in df.columns:
    df['CentralAir_Encoded'] = (df['CentralAir'] == 'Y').astype(int)
    new_features.append('CentralAir_Encoded')
    print('  - 已创建CentralAir_Encoded')

# 交互特征
if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
    df['Qual_LivArea'] = df['OverallQual'] * df['GrLivArea']
    new_features.append('Qual_LivArea')
    print('  - 已创建Qual_LivArea')

if 'HouseAge' in df.columns and 'OverallQual' in df.columns:
    df['Age_Qual_Interact'] = df['HouseAge'] * df['OverallQual']
    new_features.append('Age_Qual_Interact')
    print('  - 已创建Age_Qual_Interact')

# ==================== 5. 对数变换 ====================
print('\n正在执行对数变换...')

# 对右偏的数值特征进行对数变换
skewed_cols = ['LotArea', 'GrLivArea', 'TotalSF']
if 'TotalSF' not in df.columns:
    skewed_cols.remove('TotalSF')

for col in skewed_cols:
    if col in df.columns:
        log_col = 'Log_' + col
        df[log_col] = np.log1p(df[col].clip(lower=0))
        if log_col not in original_cols:
            new_features.append(log_col)
        print(f'  - 已创建{log_col}')

# ==================== 6. 保存结果 ====================
print('\n正在保存特征工程后的数据...')
df.to_csv(output_path, index=False)
print(f'数据已保存到: {output_path}')

print('\n' + '='*50)
print(f'特征工程完成！')
print(f'原始特征数: {len(original_cols)}')
print(f'新生成特征数: {len(new_features)}')
print(f'总特征数: {len(df.columns)}')
print('\n新生成的特征列表:')
for feat in new_features:
    print(f'  - {feat}')
print('='*50)