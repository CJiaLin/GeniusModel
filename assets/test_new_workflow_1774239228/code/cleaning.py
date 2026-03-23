import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def clean_house_prices_data():
    """
    房价数据清洗主函数
    """
    # 1. 加载数据
    input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    print("=" * 60)
    print("开始数据清洗流程")
    print("=" * 60)
    
    try:
        df = pd.read_csv(input_path)
        original_shape = df.shape
        print(f"✓ 成功加载数据: {original_shape[0]} 行 × {original_shape[1]} 列")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return
    
    # 创建清洗副本
    df_cleaned = df.copy()
    
    # 记录清洗统计
    cleaning_stats = {
        'original_rows': original_shape[0],
        'original_cols': original_shape[1],
        'duplicates_removed': 0,
        'missing_filled': {},
        'outliers_capped': {}
    }
    
    # 2. 处理重复值
    print("\n[步骤 1] 处理重复值...")
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    cleaning_stats['duplicates_removed'] = initial_rows - len(df_cleaned)
    print(f"✓ 删除重复行: {cleaning_stats['duplicates_removed']} 行")
    
    # 3. 处理缺失值
    print("\n[步骤 2] 处理缺失值...")
    
    # 3.1 数值型特征缺失值处理
    numeric_missing_cols = ['LotFrontage', 'MasVnrArea']
    
    # LotFrontage - 使用同组（Neighborhood）的中位数填充，若无则用全局中位数
    if 'LotFrontage' in df_cleaned.columns and df_cleaned['LotFrontage'].isnull().sum() > 0:
        missing_count = df_cleaned['LotFrontage'].isnull().sum()
        if 'Neighborhood' in df_cleaned.columns:
            df_cleaned['LotFrontage'] = df_cleaned.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median())
            )
        # 如果仍有缺失（某些Neighborhood全是NaN），用全局中位数
        df_cleaned['LotFrontage'].fillna(df_cleaned['LotFrontage'].median(), inplace=True)
        cleaning_stats['missing_filled']['LotFrontage'] = missing_count
        print(f"✓ LotFrontage: 填充 {missing_count} 个缺失值（基于Neighborhood中位数）")
    
    # MasVnrArea - 砌体饰面面积，缺失通常表示面积为0
    if 'MasVnrArea' in df_cleaned.columns and df_cleaned['MasVnrArea'].isnull().sum() > 0:
        missing_count = df_cleaned['MasVnrArea'].isnull().sum()
        df_cleaned['MasVnrArea'].fillna(0, inplace=True)
        cleaning_stats['missing_filled']['MasVnrArea'] = missing_count
        print(f"✓ MasVnrArea: 填充 {missing_count} 个缺失值为 0")
    
    # 3.2 分类特征缺失值处理
    # Alley - 巷道通道，NA表示无通道
    if 'Alley' in df_cleaned.columns and df_cleaned['Alley'].isnull().sum() > 0:
        missing_count = df_cleaned['Alley'].isnull().sum()
        df_cleaned['Alley'].fillna('None', inplace=True)
        cleaning_stats['missing_filled']['Alley'] = missing_count
        print(f"✓ Alley: 填充 {missing_count} 个缺失值为 'None'")
    
    # MasVnrType - 砌体饰面类型
    if 'MasVnrType' in df_cleaned.columns and df_cleaned['MasVnrType'].isnull().sum() > 0:
        missing_count = df_cleaned['MasVnrType'].isnull().sum()
        df_cleaned['MasVnrType'].fillna('None', inplace=True)
        cleaning_stats['missing_filled']['MasVnrType'] = missing_count
        print(f"✓ MasVnrType: 填充 {missing_count} 个缺失值为 'None'")
    
    # 地下室相关特征 - 缺失表示无地下室
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
            missing_count = df_cleaned[col].isnull().sum()
            df_cleaned[col].fillna('No Basement', inplace=True)
            cleaning_stats['missing_filled'][col] = missing_count
            print(f"✓ {col}: 填充 {missing_count} 个缺失值为 'No Basement'")
    
    # Electrical - 电力系统，使用众数填充
    if 'Electrical' in df_cleaned.columns and df_cleaned['Electrical'].isnull().sum() > 0:
        missing_count = df_cleaned['Electrical'].isnull().sum()
        mode_val = df_cleaned['Electrical'].mode()[0]
        df_cleaned['Electrical'].fillna(mode_val, inplace=True)
        cleaning_stats['missing_filled']['Electrical'] = missing_count
        print(f"✓ Electrical: 填充 {missing_count} 个缺失值为众数 '{mode_val}'")
    
    # 检查是否还有缺失值
    remaining_missing = df_cleaned.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"! 警告: 仍有 {remaining_missing} 个缺失值未处理")
        # 对其他缺失值进行通用处理
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                else:
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown', inplace=True)
    else:
        print("✓ 所有缺失值已处理完成")
    
    # 4. 处理异常值（IQR方法封顶）
    print("\n[步骤 3] 处理异常值（IQR封顶法）...")
    
    # 定义需要处理异常值的数值列（排除ID类和时间类）
    outlier_numeric_cols = [
        'LotFrontage', 'LotArea', 'MasVnrArea', 
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        '1stFlrSF', '2ndFlrSF'
    ]
    
    def cap_outliers_iqr(df, column):
        """使用IQR方法封顶异常值"""
        if column not in df.columns:
            return df, 0
            
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 计算异常值数量
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        # 封顶处理
        df[column] = df[column].clip(lower_bound, upper_bound)
        
        return df, outlier_count
    
    for col in outlier_numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned, count = cap_outliers_iqr(df_cleaned, col)
            if count > 0:
                cleaning_stats['outliers_capped'][col] = count
                print(f"✓ {col}: 封顶处理 {count} 个异常值")
    
    if not cleaning_stats['outliers_capped']:
        print("✓ 未发现显著异常值或所有值均在正常范围内")
    
    # 5. 数据类型优化
    print("\n[步骤 4] 优化数据类型...")
    
    # 将分类列转换为category类型（节省内存）
    categorical_cols = [
        'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
        'RoofStyle', 'RoofMatl', 'MasVnrType', 'BsmtQual', 
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical'
    ]
    
    for col in categorical_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype('category')
    
    # 确保整数列类型正确（如OverallQual, OverallCond等）
    int_cols = ['Id', 'MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd']
    for col in int_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(int)
    
    print("✓ 数据类型优化完成")
    
    # 6. 保存清洗后的数据
    print("\n[步骤 5] 保存清洗结果...")
    try:
        df_cleaned.to_csv(output_path, index=False)
        print(f"✓ 清洗后数据已保存至: {output_path}")
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        return
    
    # 7. 生成统计报告
    print("\n" + "=" * 60)
    print("数据清洗完成报告")
    print("=" * 60)
    print(f"原始数据: {cleaning_stats['original_rows']} 行 × {cleaning_stats['original_cols']} 列")
    print(f"清洗后数据: {df_cleaned.shape[0]} 行 × {df_cleaned.shape[1]} 列")
    print(f"删除重复行: {cleaning_stats['duplicates_removed']} 行")
    
    print(f"\n缺失值处理详情:")
    total_missing_filled = sum(cleaning_stats['missing_filled'].values())
    print(f"  总计处理缺失值: {total_missing_filled} 个")
    for col, count in cleaning_stats['missing_filled'].items():
        print(f"    - {col}: {count} 个")
    
    print(f"\n异常值处理详情:")
    total_outliers = sum(cleaning_stats['outliers_capped'].values())
    print(f"  总计处理异常值: {total_outliers} 个")
    for col, count in cleaning_stats['outliers_capped'].items():
        print(f"    - {col}: {count} 个")
    
    # 验证
    print(f"\n数据质量验证:")
    print(f"  缺失值检查: {'通过 ✓' if df_cleaned.isnull().sum().sum() == 0 else '未通过 ✗'}")
    print(f"  重复值检查: {'通过 ✓' if not df_cleaned.duplicated().any() else '未通过 ✗'}")
    
    print("\n清洗后的数据前5行预览:")
    print(df_cleaned.head())
    
    return df_cleaned, cleaning_stats

if __name__ == "__main__":
    df_result, stats = clean_house_prices_data()