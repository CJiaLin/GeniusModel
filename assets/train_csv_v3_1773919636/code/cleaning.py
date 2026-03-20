import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_house_prices_data():
    """
    清洗House Prices数据集
    数据路径: /Users/cjialin/code/AutoMLByLLM/train.csv
    保存路径: /Users/cjialin/code/AutoMLByLLM/train_cleaned.csv
    """
    
    # ==================== 1. 加载数据 ====================
    print("=" * 60)
    print("开始数据清洗流程")
    print("=" * 60)
    
    file_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    # 加载原始数据
    df = pd.read_csv(file_path)
    original_shape = df.shape
    print(f"\n原始数据形状: {original_shape}")
    print(f"总行数: {original_shape[0]}, 总列数: {original_shape[1]}")
    
    # ==================== 2. 初始数据质量分析 ====================
    print("\n" + "=" * 60)
    print("初始数据质量报告")
    print("=" * 60)
    
    # 统计缺失值
    missing_stats = df.isnull().sum()
    missing_cols = missing_stats[missing_stats > 0].sort_values(ascending=False)
    
    print(f"\n包含缺失值的列数: {len(missing_cols)}")
    if len(missing_cols) > 0:
        print("\n缺失值统计:")
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"  - {col}: {count} ({pct:.2f}%)")
    
    # 数据类型统计
    print(f"\n数据类型分布:")
    print(f"  - 数值型列: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  - 分类型列: {len(df.select_dtypes(include=['object']).columns)}")
    
    # ==================== 3. 缺失值处理 ====================
    print("\n" + "=" * 60)
    print("开始处理缺失值")
    print("=" * 60)
    
    # 定义数据列（基于实际提供的信息）
    numeric_cols = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 
                    'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 
                    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                    '1stFlrSF', '2ndFlrSF']
    
    categorical_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
                        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
                        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
                        'RoofStyle', 'RoofMatl']
    
    missing_cols_list = ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 
                         'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                         'BsmtFinType2', 'Electrical']
    
    # 3.1 处理LotFrontage（街道正面长度）- 数值型
    # 使用Neighborhood分组的中位数填充，同社区房屋通常有相似的LotFrontage
    if 'LotFrontage' in df.columns and df['LotFrontage'].isnull().sum() > 0:
        print("\n处理 LotFrontage 缺失值...")
        missing_before = df['LotFrontage'].isnull().sum()
        
        # 按Neighborhood分组计算中位数并填充
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # 如果仍有缺失（某些Neighborhood可能全为空），使用整体中位数
        if df['LotFrontage'].isnull().sum() > 0:
            df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
        
        print(f"  填充了 {missing_before} 个缺失值")
    
    # 3.2 处理Alley（巷道类型）- 分类变量
    # NA表示没有巷道接入
    if 'Alley' in df.columns and df['Alley'].isnull().sum() > 0:
        print("\n处理 Alley 缺失值...")
        missing_before = df['Alley'].isnull().sum()
        df['Alley'].fillna('None', inplace=True)
        print(f"  填充了 {missing_before} 个缺失值为 'None' (表示无巷道)")
    
    # 3.3 处理MasVnrType（砌体饰面类型）和MasVnrArea（砌体饰面面积）
    if 'MasVnrType' in df.columns and df['MasVnrType'].isnull().sum() > 0:
        print("\n处理 MasVnrType 缺失值...")
        missing_before = df['MasVnrType'].isnull().sum()
        df['MasVnrType'].fillna('None', inplace=True)
        print(f"  填充了 {missing_before} 个缺失值为 'None'")
    
    if 'MasVnrArea' in df.columns and df['MasVnrArea'].isnull().sum() > 0:
        print("\n处理 MasVnrArea 缺失值...")
        missing_before = df['MasVnrArea'].isnull().sum()
        # 如果MasVnrType为None，则面积应为0
        df.loc[df['MasVnrType'] == 'None', 'MasVnrArea'] = 0
        # 其他情况用中位数填充
        df['MasVnrArea'].fillna(df['MasVnrArea'].median(), inplace=True)
        print(f"  填充了 {missing_before} 个缺失值")
    
    # 3.4 处理地下室相关变量（Bsmt开头）
    # NA表示没有地下室
    bsmt_categorical = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    
    for col in bsmt_categorical:
        if col in df.columns and df[col].isnull().sum() > 0:
            print(f"\n处理 {col} 缺失值...")
            missing_before = df[col].isnull().sum()
            df[col].fillna('None', inplace=True)
            print(f"  填充了 {missing_before} 个缺失值为 'None' (表示无地下室)")
    
    # 3.5 处理Electrical（电气系统）- 分类变量
    if 'Electrical' in df.columns and df['Electrical'].isnull().sum() > 0:
        print("\n处理 Electrical 缺失值...")
        missing_before = df['Electrical'].isnull().sum()
        mode_value = df['Electrical'].mode()[0]
        df['Electrical'].fillna(mode_value, inplace=True)
        print(f"  使用众数 '{mode_value}' 填充了 {missing_before} 个缺失值")
    
    # ==================== 4. 数据类型转换 ====================
    print("\n" + "=" * 60)
    print("数据类型检查与转换")
    print("=" * 60)
    
    # 确保数值列为正确的数据类型
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 确保分类列为字符串类型
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    print("数据类型转换完成")
    
    # ==================== 5. 异常值检测与处理 ====================
    print("\n" + "=" * 60)
    print("异常值检测")
    print("=" * 60)
    
    # 检查数值列的异常值（使用IQR方法）
    numeric_data = df.select_dtypes(include=[np.number])
    outlier_report = []
    
    for col in numeric_data.columns:
        if col == 'Id':  # 跳过ID列
            continue
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        if len(outliers) > 0:
            outlier_report.append(f"  - {col}: {len(outliers)} 个异常值")
    
    if outlier_report:
        print("\n检测到的异常值（基于IQR方法）:")
        for report in outlier_report[:5]:  # 只显示前5个
            print(report)
        if len(outlier_report) > 5:
            print(f"  ... 还有 {len(outlier_report) - 5} 列存在异常值")
        print("\n注意: 异常值未被删除，建议在建模时根据业务逻辑处理")
    else:
        print("未检测到明显异常值")
    
    # ==================== 6. 最终验证 ====================
    print("\n" + "=" * 60)
    print("清洗结果验证")
    print("=" * 60)
    
    # 检查是否还有缺失值
    remaining_missing = df.isnull().sum().sum()
    print(f"\n剩余缺失值总数: {remaining_missing}")
    
    if remaining_missing > 0:
        remaining_cols = df.isnull().sum()
        remaining_cols = remaining_cols[remaining_cols > 0]
        print("仍有缺失值的列:")
        for col, count in remaining_cols.items():
            print(f"  - {col}: {count}")
    else:
        print("所有缺失值已处理完毕!")
    
    # ==================== 7. 保存清洗后的数据 ====================
    print("\n" + "=" * 60)
    print("保存清洗后的数据")
    print("=" * 60)
    
    df.to_csv(output_path, index=False)
    print(f"数据已保存至: {output_path}")
    
    # ==================== 8. 生成最终统计报告 ====================
    print("\n" + "=" * 60)
    print("数据清洗完成报告")
    print("=" * 60)
    
    cleaned_shape = df.shape
    print(f"\n清洗后数据形状: {cleaned_shape}")
    print(f"总行数: {cleaned_shape[0]}, 总列数: {cleaned_shape[1]}")
    print(f"处理记录数: {original_shape[0]}")
    print(f"保留特征数: {original_shape[1]}")
    
    print(f"\n数值型特征数: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"分类型特征数: {len(df.select_dtypes(include=['object']).columns)}")
    
    print("\n主要清洗操作:")
    print("  1. LotFrontage: 使用Neighborhood分组中位数填充")
    print("  2. Alley: 填充为'None' (表示无巷道)")
    print("  3. MasVnrType: 填充为'None' (表示无砌体饰面)")
    print("  4. MasVnrArea: 根据MasVnrType填充0或中位数")
    print("  5. 地下室相关特征: 填充为'None' (表示无地下室)")
    print("  6. Electrical: 使用众数填充")
    print("  7. 数据类型标准化")
    
    print("\n" + "=" * 60)
    print("数据清洗流程完成!")
    print("=" * 60)
    
    return df

# 执行清洗流程
if __name__ == "__main__":
    cleaned_data = clean_house_prices_data()