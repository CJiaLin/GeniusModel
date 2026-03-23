import pandas as pd
import numpy as np

def clean_housing_data(input_path, output_path):
    """
    Ames Housing 数据集清洗函数
    包含缺失值处理、类型转换和特征工程
    """
    # 读取原始数据
    print(f"正在读取数据: {input_path}")
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    missing_before = df.isnull().sum().sum()
    
    print(f"原始数据形状: {original_shape}")
    print(f"原始缺失值数量: {missing_before}")
    
    # 步骤 1: 高缺失率分类特征处理（缺失表示设施不存在）
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 
                 'FireplaceQu', 'MasVnrType']
    
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # 步骤 2: 车库相关特征处理
    # 车库字符串类型特征（无车库填充'None'）
    garage_str_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_str_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # 车库建造年份（无车库填充0）
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    
    # 步骤 3: 地下室相关特征处理（无地下室填充'None'）
    bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 
                 'BsmtFinType1', 'BsmtFinType2']
    
    for col in bsmt_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # 步骤 4: 数值型缺失值处理
    # LotFrontage：按社区（Neighborhood）的中位数填充
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 若仍有缺失（可能是新社区），用全局中位数填充
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # MasVnrArea：无砌体饰面的填充0
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # Electrical：仅1条缺失，用众数填充
    if 'Electrical' in df.columns:
        mode_electrical = df['Electrical'].mode()
        if len(mode_electrical) > 0:
            df['Electrical'] = df['Electrical'].fillna(mode_electrical[0])
    
    # 步骤 5: 数据类型转换
    # MSSubClass转为分类变量（字符串类型）
    if 'MSSubClass' in df.columns:
        df['MSSubClass'] = df['MSSubClass'].astype(str)
    
    # 步骤 6: 衍生特征创建
    new_features = []
    
    # 房屋年龄特征
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        new_features.append('HouseAge')
    
    # 翻新年龄特征
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        new_features.append('RemodAge')
    
    # 是否新房（当年建造当年出售）
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)
        new_features.append('IsNew')
    
    # 总面积特征（地下室 + 一层 + 二层 + 低质量面积）
    area_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF']
    if all(col in df.columns for col in area_cols):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['LowQualFinSF']
        new_features.append('TotalSF')
    
    # 车库标志（基于面积）
    if 'GarageArea' in df.columns:
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        new_features.append('HasGarage')
    
    # 泳池标志（基于面积）
    if 'PoolArea' in df.columns:
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        new_features.append('HasPool')
    
    # 壁炉标志
    if 'Fireplaces' in df.columns:
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
        new_features.append('HasFireplace')
    
    # 大房子标志（地上居住面积 > 4000 平方英尺）
    if 'GrLivArea' in df.columns:
        df['IsLargeHouse'] = (df['GrLivArea'] > 4000).astype(int)
        new_features.append('IsLargeHouse')
    
    # 步骤 7: 目标变量对数转换（处理右偏分布）
    if 'SalePrice' in df.columns:
        df['LogSalePrice'] = np.log1p(df['SalePrice'])
        new_features.append('LogSalePrice')
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False)
    
    # 验证和统计
    missing_after = df.isnull().sum().sum()
    cleaned_shape = df.shape
    
    # 验证检查点
    assert missing_after == 0, f"仍存在 {missing_after} 个缺失值"
    if 'MSSubClass' in df.columns:
        assert df['MSSubClass'].dtype == 'object', "MSSubClass 应转换为字符串类型"
    if 'HouseAge' in df.columns:
        assert (df['HouseAge'] >= 0).all(), "HouseAge 出现负值"
    
    return {
        'original_shape': original_shape,
        'cleaned_shape': cleaned_shape,
        'missing_before': missing_before,
        'missing_after': missing_after,
        'new_features': new_features
    }

# 主执行逻辑
if __name__ == '__main__':
    input_file = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_file = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    try:
        stats = clean_housing_data(input_file, output_file)
        
        print("\n=== 数据清洗完成 ===")
        print(f"原始数据形状: {stats['original_shape']}")
        print(f"清洗后数据形状: {stats['cleaned_shape']}")
        print(f"清洗前缺失值总数: {stats['missing_before']}")
        print(f"清洗后缺失值总数: {stats['missing_after']}")
        print(f"新增特征数量: {len(stats['new_features'])}")
        print(f"新增特征列表: {', '.join(stats['new_features'])}")
        print(f"清洗后数据已保存至: {output_file}")
        
    except Exception as e:
        print(f"清洗过程发生错误: {str(e)}")
        raise