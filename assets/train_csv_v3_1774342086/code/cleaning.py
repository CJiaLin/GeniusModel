import pandas as pd
import numpy as np
import os

def clean_housing_data():
    """
    房价数据清洗完整流程
    基于数据质量报告处理 1460 行 × 81 列的数据
    """
    
    # 数据路径
    input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_path = '/Users/cjialin/code/AutoMLByLLM/assets/train_csv_v3_1774342086/data/cleaned_data.csv'
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. 加载数据
    df = pd.read_csv(input_path)
    original_shape = df.shape
    print(f"原始数据形状: {original_shape}")
    
    # 2. 删除高缺失率列 (缺失率 > 50%)
    cols_high_missing = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df.drop(columns=[col for col in cols_high_missing if col in df.columns], inplace=True)
    
    # 3. 删除异常值比例过高的列
    cols_high_outlier = ['BsmtFinSF2', 'EnclosedPorch']
    df.drop(columns=[col for col in cols_high_outlier if col in df.columns], inplace=True)
    
    # 4. 缺失值填充
    
    # 4.1 FireplaceQu - 无壁炉标记
    if 'FireplaceQu' in df.columns:
        df['FireplaceQu'].fillna('NoFireplace', inplace=True)
    
    # 4.2 LotFrontage - 按 Neighborhood 分组中位数填充
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
    
    # 4.3 Garage 相关列
    garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cat_cols:
        if col in df.columns:
            df[col].fillna('NoGarage', inplace=True)
    
    if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
        df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)
    
    # 4.4 Basement 相关列
    bsmt_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
    for col in bsmt_cols:
        if col in df.columns:
            df[col].fillna('NoBasement', inplace=True)
    
    # 4.5 其他数值/分类列
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'].fillna(0, inplace=True)
    
    if 'Electrical' in df.columns:
        df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
    
    # 5. 异常值处理 (Winsorize 1%-99%)
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'LowQualFinSF', 'GrLivArea',
        'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'GarageArea', 'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch',
        'MiscVal', 'SalePrice'
    ]
    
    for col in winsorize_cols:
        if col in df.columns:
            lower, upper = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower, upper)
    
    # 6. 数据类型优化 - 转换为 category
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
    
    # 7. 保存清洗后的数据
    df.to_csv(output_path, index=False)
    
    # 8. 清洗结果统计
    final_shape = df.shape
    dropped_cols = original_shape[1] - final_shape[1]
    remaining_missing = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
    
    # 统计信息字典
    stats = {
        '原始数据形状': original_shape,
        '清洗后数据形状': final_shape,
        '删除列数': dropped_cols,
        '剩余缺失值总数': remaining_missing,
        '重复行数': duplicate_rows,
        '内存占用(MB)': round(memory_usage, 2),
        '保存路径': output_path
    }
    
    # 打印统计结果
    print("\n" + "="*50)
    print("数据清洗结果统计")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("="*50)
    
    return df, stats

# 执行清洗
if __name__ == "__main__":
    df_cleaned, cleaning_stats = clean_housing_data()
    print("\n数据清洗完成！")