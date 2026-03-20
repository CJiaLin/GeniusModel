import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_and_clean_data():
    """
    加载并清洗数据
    """
    # 1. 加载数据
    file_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    print(f"正在加载数据: {file_path}")
    df = pd.read_csv(file_path)
    
    # 原始数据信息
    original_shape = df.shape
    print(f"原始数据形状: {original_shape}")
    print(f"原始数据列数: {len(df.columns)}")
    
    # 2. 处理缺失值
    print("\n开始处理缺失值...")
    
    # 数值列缺失值处理（基于提供的数值列信息）
    numeric_cols_with_na = ['LotFrontage', 'MasVnrArea']
    for col in numeric_cols_with_na:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if col == 'MasVnrArea':
                    # MasVnrArea 缺失可能表示没有砌体 veneer，用0填充
                    df[col].fillna(0, inplace=True)
                    print(f"  {col}: 填充 {missing_count} 个缺失值为 0")
                else:
                    # 其他数值列用中位数填充
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"  {col}: 填充 {missing_count} 个缺失值为中位数 {median_val}")
    
    # 分类列缺失值处理（基于提供的分类列和缺失值列信息）
    # Alley: 缺失表示没有巷道，用'NA'填充
    if 'Alley' in df.columns:
        missing_count = df['Alley'].isnull().sum()
        if missing_count > 0:
            df['Alley'].fillna('NA', inplace=True)
            print(f"  Alley: 填充 {missing_count} 个缺失值为 'NA'")
    
    # MasVnrType: 缺失可能表示没有砌体 veneer
    if 'MasVnrType' in df.columns:
        missing_count = df['MasVnrType'].isnull().sum()
        if missing_count > 0:
            df['MasVnrType'].fillna('None', inplace=True)
            print(f"  MasVnrType: 填充 {missing_count} 个缺失值为 'None'")
    
    # 地下室相关列: 缺失表示没有地下室
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col].fillna('NA', inplace=True)
                print(f"  {col}: 填充 {missing_count} 个缺失值为 'NA'")
    
    # Electrical: 用众数填充
    if 'Electrical' in df.columns:
        missing_count = df['Electrical'].isnull().sum()
        if missing_count > 0:
            mode_val = df['Electrical'].mode()[0]
            df['Electrical'].fillna(mode_val, inplace=True)
            print(f"  Electrical: 填充 {missing_count} 个缺失值为众数 '{mode_val}'")
    
    # 检查其他可能的缺失值列（数据中可能有更多缺失值列）
    other_missing_cols = df.columns[df.isnull().sum() > 0].tolist()
    if other_missing_cols:
        print(f"\n  发现其他缺失值列: {other_missing_cols}")
        for col in other_missing_cols:
            missing_count = df[col].isnull().sum()
            if df[col].dtype in ['int64', 'float64']:
                # 数值列用中位数填充
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"    {col}: 填充 {missing_count} 个缺失值为中位数 {median_val}")
            else:
                # 分类列用众数填充
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"    {col}: 填充 {missing_count} 个缺失值为众数 '{mode_val}'")
    
    # 3. 数据类型优化
    print("\n优化数据类型...")
    
    # 将某些数值列转换为整数（如果它们是整数类型）
    int_candidates = ['Id', 'MSSubClass', 'OverallQual', 'OverallCond', 
                      'YearBuilt', 'YearRemodAdd']
    for col in int_candidates:
        if col in df.columns and df[col].dtype == 'float64':
            df[col] = df[col].astype('int64')
            print(f"  {col}: 转换为 int64")
    
    # 将分类列转换为category类型以节省内存
    categorical_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
                       'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
                       'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
                       'RoofStyle', 'RoofMatl', 'MasVnrType', 'BsmtQual', 
                       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                       'Electrical']
    
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype('category')
            print(f"  {col}: 转换为 category")
    
    # 4. 异常值检测和处理
    print("\n检测异常值...")
    
    # 基于提供的数值列检查异常值
    numeric_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 
                   'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    
    outlier_stats = {}
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_stats[col] = outlier_count
            
            if outlier_count > 0:
                print(f"  {col}: 发现 {outlier_count} 个异常值 (范围: [{lower_bound:.2f}, {upper_bound:.2f}])")
                # 使用截断法处理极端异常值
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 5. 验证年份数据的合理性
    year_cols = ['YearBuilt', 'YearRemodAdd']
    current_year = 2024
    for col in year_cols:
        if col in df.columns:
            invalid_years = df[(df[col] < 1800) | (df[col] > current_year)]
            if len(invalid_years) > 0:
                print(f"  {col}: 发现 {len(invalid_years)} 个无效年份，使用众数替换")
                mode_year = df[col].mode()[0]
                df.loc[(df[col] < 1800) | (df[col] > current_year), col] = mode_year
    
    # 6. 确保面积数据的一致性
    area_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
    if all(col in df.columns for col in area_cols + ['TotalBsmtSF']):
        # 检查地下室总面积是否等于各部分之和
        calculated_total = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['BsmtUnfSF']
        inconsistent = (df['TotalBsmtSF'] - calculated_total).abs() > 1
        if inconsistent.sum() > 0:
            print(f"\n  发现 {inconsistent.sum()} 行地下室面积不一致，重新计算 TotalBsmtSF")
            df.loc[inconsistent, 'TotalBsmtSF'] = calculated_total[inconsistent]
    
    # 7. 保存清洗后的数据
    output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    df.to_csv(output_path, index=False)
    print(f"\n清洗后的数据已保存到: {output_path}")
    
    # 8. 生成清洗报告
    cleaned_shape = df.shape
    total_missing_original = sum([df[col].isnull().sum() for col in df.columns])
    
    # 统计信息
    stats = {
        '原始数据形状': original_shape,
        '清洗后数据形状': cleaned_shape,
        '总行数': cleaned_shape[0],
        '总列数': cleaned_shape[1],
        '数值列数量': len(df.select_dtypes(include=[np.number]).columns),
        '分类列数量': len(df.select_dtypes(include=['category', 'object']).columns),
        '处理后缺失值总数': df.isnull().sum().sum(),
        '异常值处理统计': outlier_stats,
        '内存使用优化': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    
    print("\n" + "="*50)
    print("数据清洗完成!")
    print("="*50)
    print(f"原始数据: {original_shape[0]} 行 × {original_shape[1]} 列")
    print(f"清洗后数据: {cleaned_shape[0]} 行 × {cleaned_shape[1]} 列")
    print(f"数值列数量: {stats['数值列数量']}")
    print(f"分类列数量: {stats['分类列数量']}")
    print(f"处理后缺失值总数: {stats['处理后缺失值总数']}")
    print(f"内存使用: {stats['内存使用优化']}")
    
    return df, stats

# 执行清洗
if __name__ == "__main__":
    cleaned_df, statistics = load_and_clean_data()
    
    # 显示前几行数据
    print("\n清洗后数据预览 (前5行):")
    print(cleaned_df.head())
    
    print("\n数据基本信息:")
    print(cleaned_df.info())