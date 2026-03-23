import pandas as pd
import numpy as np
import os

def clean_housing_data(file_path, output_path):
    """
    房价数据清洗主函数
    针对Ames Housing数据集进行全面的数据清洗和特征工程
    
    参数:
        file_path: 原始数据文件路径
        output_path: 清洗后数据保存路径
    
    返回:
        df: 清洗后的数据框
        report: 清洗过程统计报告
    """
    print(f"开始加载数据: {file_path}")
    
    # 1. 加载数据
    df = pd.read_csv(file_path)
    original_shape = df.shape
    print(f"原始数据形状: {original_shape}")
    
    # 初始化报告
    report = {
        'original_shape': original_shape,
        'dropped_columns': [],
        'filled_missing': {},
        'created_features': []
    }
    
    # 2. 删除高缺失率列（缺失率 > 50%）
    # 根据实际数据信息，这些列几乎全是缺失值
    cols_to_drop_high_missing = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    existing_cols_to_drop = [col for col in cols_to_drop_high_missing if col in df.columns]
    
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        report['dropped_columns'].extend(existing_cols_to_drop)
        print(f"删除高缺失率列: {existing_cols_to_drop}")
    
    # 3. 删除异常值过多的列（基于业务判断）
    cols_to_drop_outliers = ['BsmtFinSF2', 'EnclosedPorch']
    existing_outlier_cols = [col for col in cols_to_drop_outliers if col in df.columns]
    
    if existing_outlier_cols:
        df = df.drop(columns=existing_outlier_cols)
        report['dropped_columns'].extend(existing_outlier_cols)
        print(f"删除异常值过多列: {existing_outlier_cols}")
    
    print(f"删除列后数据形状: {df.shape}")
    
    # 4. 缺失值处理
    
    # 4.1 分类变量缺失值填充（缺失表示不存在）
    # FireplaceQu: 填充为"None"（表示无壁炉）
    if 'FireplaceQu' in df.columns:
        missing_count = df['FireplaceQu'].isnull().sum()
        if missing_count > 0:
            df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
            report['filled_missing']['FireplaceQu'] = f'填充"None"({missing_count}个)'
            print(f"填充 FireplaceQu: {missing_count}个缺失值设为'None'")
    
    # Garage相关特征（缺失表示无车库）
    garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna('None')
                report['filled_missing'][col] = f'填充"None"({missing_count}个)'
                print(f"填充 {col}: {missing_count}个缺失值设为'None'")
    
    # Basement相关特征（缺失表示无地下室）
    bsmt_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
    for col in bsmt_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna('None')
                report['filled_missing'][col] = f'填充"None"({missing_count}个)'
                print(f"填充 {col}: {missing_count}个缺失值设为'None'")
    
    # 4.2 数值变量缺失值填充
    # LotFrontage: 按邻居组（Neighborhood）的中位数填充
    if 'LotFrontage' in df.columns:
        missing_count = df['LotFrontage'].isnull().sum()
        if missing_count > 0 and 'Neighborhood' in df.columns:
            df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median())
            )
            # 如果仍有缺失（某些Neighborhood全为NaN），用全局中位数填充
            if df['LotFrontage'].isnull().sum() > 0:
                global_median = df['LotFrontage'].median()
                df['LotFrontage'] = df['LotFrontage'].fillna(global_median)
            report['filled_missing']['LotFrontage'] = f'按Neighborhood中位数填充({missing_count}个)'
            print(f"填充 LotFrontage: {missing_count}个缺失值按Neighborhood中位数填充")
    
    # GarageYrBlt: 无车库的填充0，表示无建造年份
    if 'GarageYrBlt' in df.columns:
        missing_count = df['GarageYrBlt'].isnull().sum()
        if missing_count > 0:
            df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
            report['filled_missing']['GarageYrBlt'] = f'填充0({missing_count}个)'
            print(f"填充 GarageYrBlt: {missing_count}个缺失值设为0")
    
    # MasVnrArea: 填充0（无砌体veneer）
    if 'MasVnrArea' in df.columns:
        missing_count = df['MasVnrArea'].isnull().sum()
        if missing_count > 0:
            df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
            report['filled_missing']['MasVnrArea'] = f'填充0({missing_count}个)'
            print(f"填充 MasVnrArea: {missing_count}个缺失值设为0")
    
    # Electrical: 填充众数
    if 'Electrical' in df.columns:
        missing_count = df['Electrical'].isnull().sum()
        if missing_count > 0:
            mode_value = df['Electrical'].mode()[0]
            df['Electrical'] = df['Electrical'].fillna(mode_value)
            report['filled_missing']['Electrical'] = f'填充众数"{mode_value}"({missing_count}个)'
            print(f"填充 Electrical: {missing_count}个缺失值设为众数'{mode_value}'")
    
    # 5. 异常值处理（Winsorize缩尾处理）
    # 对数值特征进行1%-99%分位数缩尾处理
    def winsorize_series(series, lower_percentile=0.01, upper_percentile=0.99):
        """对序列进行缩尾处理，保留边界值"""
        if series.isnull().all():
            return series
        lower = series.quantile(lower_percentile)
        upper = series.quantile(upper_percentile)
        return series.clip(lower, upper)
    
    # 需要Winsorize的列（基于业务合理性和实际数据中的数值列）
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
        'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal'
    ]
    
    # 如果存在SalePrice，也对目标变量进行Winsorize
    if 'SalePrice' in df.columns:
        winsorize_cols.append('SalePrice')
    
    winsorized_count = 0
    for col in winsorize_cols:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            original_min = df[col].min()
            original_max = df[col].max()
            df[col] = winsorize_series(df[col])
            if df[col].min() != original_min or df[col].max() != original_max:
                winsorized_count += 1
                print(f"Winsorize处理: {col} (原范围:[{original_min:.2f}, {original_max:.2f}], 新范围:[{df[col].min():.2f}, {df[col].max():.2f}])")
    
    report['winsorized_columns_count'] = winsorized_count
    print(f"共完成 {winsorized_count} 列的Winsorize处理")
    
    # 6. 数据类型优化
    
    # 6.1 分类变量转换为category类型（减少内存占用）
    categorical_cols = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
        'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'
    ]
    
    # 只转换实际存在的列
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # 6.2 MSSubClass实际是分类变量（建筑类型编码），转换为category
    if 'MSSubClass' in df.columns:
        df['MSSubClass'] = df['MSSubClass'].astype('category')
    
    # 6.3 确保年份相关列为整数类型
    year_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']
    for col in year_cols:
        if col in df.columns:
            # 先填充可能存在的缺失值，再转换类型
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(0)
            df[col] = df[col].astype(int)
    
    print("数据类型转换完成")
    
    # 7. 特征工程（针对房价预测优化）
    
    # 7.1 创建总面积特征（多维度聚合）
    # 注意：确保所有需要的列都存在
    sf_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    existing_sf_cols = [col for col in sf_cols if col in df.columns]
    if len(existing_sf_cols) >= 2:  # 至少有两个列才能求和
        df['TotalSF'] = df[existing_sf_cols].sum(axis=1)
        report['created_features'].append('TotalSF')
        print(f"创建特征 TotalSF = {' + '.join(existing_sf_cols)}")
    
    # 门廊总面积
    porch_cols = ['OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']
    existing_porch_cols = [col for col in porch_cols if col in df.columns]
    if existing_porch_cols:
        df['TotalPorchSF'] = df[existing_porch_cols].sum(axis=1)
        report['created_features'].append('TotalPorchSF')
        print(f"创建特征 TotalPorchSF = {' + '.join(existing_porch_cols)}")
    
    # 7.2 房屋年龄特征（相对于售出时间）
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        report['created_features'].append('HouseAge')
        print("创建特征 HouseAge = YrSold - YearBuilt")
    
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        report['created_features'].append('RemodAge')
        print("创建特征 RemodAge = YrSold - YearRemodAdd")
    
    # 7.3 二元指示特征
    if 'PoolArea' in df.columns:
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        report['created_features'].append('HasPool')
        print("创建特征 HasPool = (PoolArea > 0)")
    
    if 'TotalBsmtSF' in df.columns:
        df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
        report['created_features'].append('HasBsmt')
        print("创建特征 HasBsmt = (TotalBsmtSF > 0)")
    
    # 8. 对数变换（针对RMSE优化）
    # 房价通常呈右偏分布，对数变换可使其更接近正态分布
    
    if 'SalePrice' in df.columns:
        # 对目标变量进行对数变换
        df['SalePrice_Log'] = np.log1p(df['SalePrice'])
        report['created_features'].append('SalePrice_Log')
        print("创建特征 SalePrice_Log = log1p(SalePrice)")
    
    # 对高度右偏的数值特征进行对数变换
    skewed_cols = ['LotArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 
                   'MiscVal']
    
    # 如果TotalSF存在，也加入对数变换
    if 'TotalSF' in df.columns:
        skewed_cols.append('TotalSF')
    if 'TotalPorchSF' in df.columns:
        skewed_cols.append('TotalPorchSF')
    
    for col in skewed_cols:
        if col in df.columns:
            # 确保没有负值（对数变换要求）
            if (df[col] < 0).sum() == 0:
                new_col_name = f'{col}_Log'
                df[new_col_name] = np.log1p(df[col])
                report['created_features'].append(new_col_name)
                print(f"创建特征 {new_col_name} = log1p({col})")
    
    # 9. 最终数据质量检查
    final_missing = df.isnull().sum().sum()
    report['final_shape'] = df.shape
    report['total_missing_remaining'] = final_missing
    
    print(f"\n清洗完成!")
    print(f"最终数据形状: {df.shape}")
    print(f"剩余缺失值总数: {final_missing}")
    
    # 10. 保存清洗后的数据
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"数据已保存至: {output_path}")
    
    return df, report

def validate_cleaned_data(df):
    """
    验证清洗后的数据质量
    返回详细的验证报告
    """
    print("\n开始数据质量验证...")
    
    report = {}
    
    #