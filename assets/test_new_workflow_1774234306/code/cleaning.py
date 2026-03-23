import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def clean_house_prices_data(input_path, output_path):
    """
    房价数据清洗流程
    基于实际数据列名进行处理
    """
    print(f"开始加载数据: {input_path}")
    
    # 1. 加载数据
    df = pd.read_csv(input_path)
    original_shape = df.shape
    original_missing = df.isnull().sum().sum()
    
    print(f"原始数据形状: {original_shape}")
    print(f"原始缺失值总数: {original_missing}")
    print(f"列名: {list(df.columns)}")
    
    # 2. 处理缺失值 - 按照不同策略分类处理
    
    # 2.1 Alley（巷子类型）- 缺失表示没有巷子
    if 'Alley' in df.columns:
        df['Alley'] = df['Alley'].fillna('None')
    
    # 2.2 LotFrontage（临街宽度）- 基于Neighborhood分组中位数填充
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果仍有缺失（某些Neighborhood全缺失），用整体中位数填充
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # 2.3 MasVnrType（砌体饰面类型）和 MasVnrArea（砌体饰面面积）
    if 'MasVnrType' in df.columns:
        df['MasVnrType'] = df['MasVnrType'].fillna('None')
    
    if 'MasVnrArea' in df.columns:
        # 如果类型为None，面积应为0；否则用中位数填充
        mask = (df['MasVnrType'] == 'None') | (df['MasVnrType'].isna())
        df.loc[mask, 'MasVnrArea'] = df.loc[mask, 'MasVnrArea'].fillna(0)
        df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].median())
    
    # 2.4 地下室相关字段 - 缺失表示没有地下室
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        if col in df.columns:
            df[col] = df[col].fillna('No Basement')
    
    # 地下室面积字段（如果有缺失）填0
    basement_area_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
    for col in basement_area_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 2.5 Electrical（电力系统）- 用众数填充
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0] if not df['Electrical'].mode().empty else 'SBrkr')
    
    # 2.6 其他常见缺失值处理（ FireplaceQu, PoolQC, Fence, MiscFeature 等如果存在）
    other_categorical_cols = ['FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'GarageType', 
                              'GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt']
    for col in other_categorical_cols:
        if col in df.columns:
            if col == 'GarageYrBlt':
                # 年份字段特殊处理：缺失表示无车库，可用0或房屋建造年份
                df[col] = df[col].fillna(df['YearBuilt'] if 'YearBuilt' in df.columns else 0)
            else:
                df[col] = df[col].fillna('None')
    
    # 车库面积字段
    garage_area_cols = ['GarageCars', 'GarageArea']
    for col in garage_area_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 3. 异常值处理
    print("开始异常值处理...")
    
    # 3.1 确保所有面积字段非负
    area_cols = ['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'WoodDeckSF', 
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']
    
    for col in area_cols:
        if col in df.columns:
            # 将负值设为0（如果有的话）
            negative_mask = df[col] < 0
            if negative_mask.any():
                print(f"发现 {col} 中有 {negative_mask.sum()} 个负值，已修正为0")
                df.loc[negative_mask, col] = 0
    
    # 3.2 年份字段合理性检查
    year_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
    current_year = 2024
    for col in year_cols:
        if col in df.columns:
            # 将未来年份设为当前年份
            future_mask = df[col] > current_year
            if future_mask.any():
                print(f"发现 {col} 中有 {future_mask.sum()} 个未来年份，已修正")
                df.loc[future_mask, col] = current_year
            # 将过早的年份（<1800）设为最小合理值
            early_mask = df[col] < 1800
            if early_mask.any():
                print(f"发现 {col} 中有 {early_mask.sum()} 个过早年份，已修正")
                df.loc[early_mask, col] = 1800
    
    # 4. 特征工程
    print("开始特征工程...")
    
    # 4.1 计算房屋总居住面积（如果相关列存在）
    sf_cols = ['1stFlrSF', '2ndFlrSF', 'LowQualFinSF'] if 'LowQualFinSF' in df.columns else ['1stFlrSF', '2ndFlrSF']
    if all(col in df.columns for col in sf_cols):
        df['TotalSF'] = df[sf_cols].sum(axis=1)
    
    # 4.2 计算房屋年龄和改造年龄
    if 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YearBuilt'].apply(lambda x: max(0, current_year - x))
    
    if 'YearRemodAdd' in df.columns and 'YearBuilt' in df.columns:
        df['RemodAge'] = df['YearRemodAdd'] - df['YearBuilt']
        df['RemodAge'] = df['RemodAge'].apply(lambda x: max(0, x))  # 确保非负
    
    # 4.3 总浴室数（如果相关列存在）
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    available_bath_cols = [col for col in bath_cols if col in df.columns]
    if available_bath_cols:
        # Half bath算作0.5个浴室
        df['TotalBath'] = df['FullBath'] if 'FullBath' in df.columns else 0
        if 'HalfBath' in df.columns:
            df['TotalBath'] += 0.5 * df['HalfBath']
        if 'BsmtFullBath' in df.columns:
            df['TotalBath'] += df['BsmtFullBath']
        if 'BsmtHalfBath' in df.columns:
            df['TotalBath'] += 0.5 * df['BsmtHalfBath']
    
    # 4.4 总门廊面积
    porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    available_porch_cols = [col for col in porch_cols if col in df.columns]
    if available_porch_cols:
        df['TotalPorchSF'] = df[available_porch_cols].sum(axis=1)
    
    # 4.5 是否有泳池、是否有2楼、是否有地下室等二元特征
    if 'PoolArea' in df.columns:
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    
    if '2ndFlrSF' in df.columns:
        df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    
    if 'TotalBsmtSF' in df.columns:
        df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    
    if 'GarageArea' in df.columns:
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    
    # 5. 数据类型优化
    # 将某些数值型但实际上是分类的字段转换为字符串
    categorical_as_numeric = ['MSSubClass', 'OverallQual', 'OverallCond']
    for col in categorical_as_numeric:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # 6. 最终缺失值检查和处理
    remaining_missing = df.isnull().sum()
    cols_with_missing = remaining_missing[remaining_missing > 0]
    
    if len(cols_with_missing) > 0:
        print(f"仍有缺失值的列: \n{cols_with_missing}")
        # 对剩余数值列用中位数填充，分类列用众数填充
        for col in cols_with_missing.index:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # 7. 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"\n清洗后数据已保存至: {output_path}")
    
    # 8. 生成统计报告
    final_missing = df.isnull().sum().sum()
    final_shape = df.shape
    
    report = {
        'original_shape': original_shape,
        'final_shape': final_shape,
        'original_missing': int(original_missing),
        'final_missing': int(final_missing),
        'rows_processed': original_shape[0],
        'columns_processed': original_shape[1],
        'new_features_added': final_shape[1] - original_shape[1],
        'missing_fixed': int(original_missing - final_missing)
    }
    
    print("\n" + "="*50)
    print("数据清洗完成报告")
    print("="*50)
    print(f"原始数据维度: {original_shape}")
    print(f"清洗后维度: {final_shape}")
    print(f"新增特征数: {report['new_features_added']}")
    print(f"处理前缺失值: {original_missing}")
    print(f"处理后缺失值: {final_missing}")
    print(f"修复缺失值数: {report['missing_fixed']}")
    print("="*50)
    
    return df, report

# 执行清洗
input_file = '/Users/cjialin/code/AutoMLByLLM/train.csv'
output_file = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'

try:
    df_cleaned, cleaning_report = clean_house_prices_data(input_file, output_file)
    print("\n数据清洗成功完成！")
except Exception as e:
    print(f"清洗过程中出现错误: {str(e)}")
    import traceback
    traceback.print_exc()