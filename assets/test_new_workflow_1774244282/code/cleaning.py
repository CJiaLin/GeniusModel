import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. 加载数据
# ============================================
file_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
df = pd.read_csv(file_path)

print(f"原始数据形状: {df.shape}")
print(f"原始数据列数: {len(df.columns)}")

# 记录原始信息用于对比
original_shape = df.shape
original_missing = df.isnull().sum().sum()

# ============================================
# 2. 删除高缺失率列 (>50%)
# ============================================
# 根据方案删除：PoolQC, MiscFeature, Alley, Fence, MasVnrType
high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
# 只删除实际存在的列
cols_to_drop = [col for col in high_missing_cols if col in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"删除高缺失列: {cols_to_drop}")
print(f"删除高缺失列后: {df.shape}")

# ============================================
# 3. 缺失值智能填充
# ============================================

# 3.1 类别特征填充为"None"（表示无该设施）
categorical_none_cols = [
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
]
for col in categorical_none_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')
        print(f"填充 {col}: None")

# 3.2 数值特征填充
# LotFrontage: 按Neighborhood分组填充中位数
if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    # 如果仍有缺失，用整体中位数填充
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    print("填充 LotFrontage: 按Neighborhood中位数")

# GarageYrBlt: 无车库则使用YearBuilt
if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    print("填充 GarageYrBlt: YearBuilt")

# MasVnrArea: 无砌体贴面填0（如果还存在该列）
if 'MasVnrArea' in df.columns:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    print("填充 MasVnrArea: 0")

# Electrical: 用众数填充
if 'Electrical' in df.columns:
    mode_val = df['Electrical'].mode()
    if len(mode_val) > 0:
        df['Electrical'] = df['Electrical'].fillna(mode_val[0])
        print(f"填充 Electrical: {mode_val[0]}")

print(f"缺失值填充后总缺失数: {df.isnull().sum().sum()}")

# ============================================
# 4. 异常值处理 - Winsorize (5%-95%分位数截断)
# ============================================
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch',
    'MiscVal', 'SalePrice'
]

winsorize_count = 0
for col in winsorize_cols:
    if col in df.columns:
        lower = df[col].quantile(0.05)
        upper = df[col].quantile(0.95)
        df[col] = df[col].clip(lower, upper)
        winsorize_count += 1

print(f"异常值Winsorize处理完成: {winsorize_count} 列")

# ============================================
# 5. 数据类型优化 (object -> category)
# ============================================
object_cols = df.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    df[object_cols] = df[object_cols].astype('category')
    print(f"转换 {len(object_cols)} 列为category类型")

# ============================================
# 6. 特征工程（房价预测专用）
# ============================================

# 6.1 房屋年龄相关特征
current_year = 2024
if 'YearBuilt' in df.columns:
    df['HouseAge'] = current_year - df['YearBuilt']
if 'YearRemodAdd' in df.columns:
    df['RemodAge'] = current_year - df['YearRemodAdd']
if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
    df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)

# 6.2 总面积特征
area_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
if all(col in df.columns for col in area_cols):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']
existing_porch_cols = [col for col in porch_cols if col in df.columns]
if existing_porch_cols:
    df['TotalPorchSF'] = df[existing_porch_cols].sum(axis=1)

# 6.3 房间密度
if 'TotRmsAbvGrd' in df.columns and 'GrLivArea' in df.columns:
    df['RoomDensity'] = df['TotRmsAbvGrd'] / (df['GrLivArea'] + 1)

# 6.4 浴室总数
bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
existing_bath_cols = [col for col in bath_cols if col in df.columns]
if len(existing_bath_cols) >= 4:
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                       df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
elif 'FullBath' in df.columns and 'HalfBath' in df.columns:
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']

# 6.5 车库比例
if 'GarageArea' in df.columns and 'LotArea' in df.columns:
    df['GarageAreaRatio'] = df['GarageArea'] / (df['LotArea'] + 1)

print("特征工程完成")

# ============================================
# 7. 最终验证与统计
# ============================================
print("\n" + "="*50)
print("清洗后数据质量报告")
print("="*50)
print(f"原始数据形状: {original_shape}")
print(f"清洗后数据形状: {df.shape}")
print(f"删除列数: {original_shape[1] - df.shape[1]}")
print(f"新增特征数: {df.shape[1] - (original_shape[1] - len(cols_to_drop))}")
print(f"原始缺失值总数: {original_missing}")
print(f"清洗后缺失值总数: {df.isnull().sum().sum()}")
print(f"重复行数: {df.duplicated().sum()}")
print(f"数值列数: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"分类列数: {len(df.select_dtypes(include=['category']).columns)}")

# 显示各列缺失值情况（如有）
remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]
if len(remaining_missing) > 0:
    print(f"\n仍有缺失值的列:\n{remaining_missing}")
else:
    print("\n所有缺失值已处理完毕")

# ============================================
# 8. 保存清洗后数据
# ============================================
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n清洗后数据已保存至: {output_path}")
print(f"文件大小: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")