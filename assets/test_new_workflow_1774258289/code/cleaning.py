"""
房价预测数据清洗脚本
数据路径: /Users/cjialin/code/AutoMLByLLM/train.csv
目标: 清洗数据用于房价预测模型（RMSE评估）
"""

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

# 保存Id列（预测时需要）
if 'Id' in df.columns:
    id_col = df['Id'].copy()

# ============================================
# 2. 删除高缺失率列（>50%）
# ============================================
high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
df = df.drop(columns=[col for col in high_missing_cols if col in df.columns])
print(f"删除高缺失列后: {df.shape}")

# ============================================
# 3. 删除低方差/异常分布列
# ============================================
low_variance_cols = ['BsmtFinSF2', 'EnclosedPorch']
df = df.drop(columns=[col for col in low_variance_cols if col in df.columns])
print(f"删除低方差列后: {df.shape}")

# ============================================
# 4. 缺失值填充 - 地下室相关
# ============================================
# 地下室分类变量：缺失表示无地下室，填充"None"
bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

# ============================================
# 5. 缺失值填充 - 车库相关
# ============================================
# 车库分类变量：缺失表示无车库，填充"None"
garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

# GarageYrBlt：无车库时填充YearBuilt（房屋建造年份）
if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

# ============================================
# 6. 缺失值填充 - 其他特征
# ============================================
# FireplaceQu：无壁炉填充"None"
if 'FireplaceQu' in df.columns:
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

# LotFrontage：按Neighborhood分组填充中位数（同社区地块相似）
if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    # 若仍有缺失（如新社区），填充整体中位数
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# MasVnrArea：无贴面填充0
if 'MasVnrArea' in df.columns:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# Electrical：填充众数（标准断路器SBrkr）
if 'Electrical' in df.columns:
    df['Electrical'] = df['Electrical'].fillna('SBrkr')

print(f"缺失值填充完成，剩余缺失值: {df.isnull().sum().sum()}")

# ============================================
# 7. 异常值处理 - Winsorize缩尾（1%-99%分位数）
# ============================================
def winsorize_series(series, lower_quantile=0.01, upper_quantile=0.99):
    """对序列进行缩尾处理"""
    lower_bound = series.quantile(lower_quantile)
    upper_bound = series.quantile(upper_quantile)
    return series.clip(lower=lower_bound, upper=upper_bound)

# 需Winsorize的数值列配置（列名: (下限分位, 上限分位)）
winsorize_config = {
    'MSSubClass': (0.01, 0.99),
    'LotFrontage': (0.00, 0.99),
    'LotArea': (0.01, 0.99),
    'OverallCond': (0.01, 0.99),
    'MasVnrArea': (0.00, 0.99),
    'BsmtUnfSF': (0.01, 0.99),
    'TotalBsmtSF': (0.01, 0.99),
    '1stFlrSF': (0.01, 0.99),
    'LowQualFinSF': (0.00, 0.90),
    'GrLivArea': (0.01, 0.99),
    'BsmtHalfBath': (0.00, 0.95),
    'BedroomAbvGr': (0.01, 0.99),
    'KitchenAbvGr': (0.01, 0.99),
    'TotRmsAbvGrd': (0.01, 0.99),
    'GarageArea': (0.00, 0.99),
    'WoodDeckSF': (0.00, 0.99),
    'OpenPorchSF': (0.00, 0.99),
    '3SsnPorch': (0.00, 0.95),
    'ScreenPorch': (0.00, 0.95),
    'MiscVal': (0.00, 0.95),
    'SalePrice': (0.01, 0.99)
}

winsorize_count = 0
for col, (lower_q, upper_q) in winsorize_config.items():
    if col in df.columns:
        df[col] = winsorize_series(df[col], lower_q, upper_q)
        winsorize_count += 1

print(f"异常值处理完成，已处理 {winsorize_count} 个数值列")

# ============================================
# 8. 数据类型优化
# ============================================
# 将object类型转换为category（提升效率）
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    df[col] = df[col].astype('category')

print(f"分类变量已优化: {len(categorical_cols)} 列")

# ============================================
# 9. 保存清洗后的数据
# ============================================
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n清洗后数据已保存至: {output_path}")

# ============================================
# 10. 最终验证与统计
# ============================================
print("\n" + "="*50)
print("清洗后数据质量报告")
print("="*50)
print(f"数据形状: {df.shape}")
print(f"缺失值总数: {df.isnull().sum().sum()}")
print(f"重复行数: {df.duplicated().sum()}")
print(f"内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 数据类型统计
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n数值列数量: {len(numeric_cols)}")
print(f"分类列数量: {len(categorical_cols)}")

# 目标变量统计
if 'SalePrice' in df.columns:
    print(f"\n目标变量SalePrice统计:")
    print(f"  均值: {df['SalePrice'].mean():.2f}")
    print(f"  标准差: {df['SalePrice'].std():.2f}")
    print(f"  最小值: {df['SalePrice'].min():.2f}")
    print(f"  最大值: {df['SalePrice'].max():.2f}")

print("="*50)
print("数据清洗完成!")