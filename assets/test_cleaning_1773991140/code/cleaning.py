import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_house_prices_data(input_path, output_path):
    """
    房价数据完整清洗流程
    基于数据质量报告和清洗方案实现
    """
    print("=" * 60)
    print("开始数据清洗流程")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/7] 加载原始数据...")
    df = pd.read_csv(input_path)
    original_shape = df.shape
    print(f"原始数据形状: {original_shape}")
    print(f"数值特征数: {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"分类特征数: {df.select_dtypes(include=['object']).shape[1]}")
    
    # 2. 处理极高缺失率特征（>80%，直接删除）
    print("\n[2/7] 处理极高缺失率特征...")
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
    df = df.drop(columns=cols_to_drop)
    print(f"已删除特征: {cols_to_drop}")
    print(f"当前数据形状: {df.shape}")
    
    # 3. 分类特征缺失填充（基于业务逻辑）
    print("\n[3/7] 处理分类特征缺失值...")
    
    # 地下室相关特征（缺失表示无地下室）
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        df[col] = df[col].fillna('No_Basement')
    print(f"地下室特征填充完成: {len(basement_cols)}个字段")
    
    # 车库相关特征（缺失表示无车库）
    garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cols:
        df[col] = df[col].fillna('No_Garage')
    
    # 车库建造年份（无车库时用房屋建造年份填充）
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    print(f"车库特征填充完成: {len(garage_cols)}个字段 + GarageYrBlt")
    
    # 壁炉质量（缺失表示无壁炉）
    df['FireplaceQu'] = df['FireplaceQu'].fillna('No_Fireplace')
    
    # 砖石饰面类型（缺失表示无饰面）
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    print("其他分类特征填充完成: FireplaceQu, MasVnrType")
    
    # 4. 数值特征缺失填充
    print("\n[4/7] 处理数值特征缺失值...")
    
    # 砖石饰面面积（无砖石时填充0）
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # 临街距离（按街区分组填充中位数）
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # 电气系统（用众数填充）
    electrical_mode = df['Electrical'].mode()[0]
    df['Electrical'] = df['Electrical'].fillna(electrical_mode)
    print(f"Electrical用众数填充: {electrical_mode}")
    print("数值特征缺失填充完成")
    
    # 5. 异常值处理（IQR方法，使用3倍IQR更宽松）
    print("\n[5/7] 处理异常值...")
    
    def remove_outliers_iqr(df, column, multiplier=3):
        """使用IQR方法截断异常值"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        df[column] = df[column].clip(lower_bound, upper_bound)
        return outliers_count
    
    outlier_cols = ['LotArea', 'GrLivArea', '1stFlrSF', 'BsmtFinSF1']
    for col in outlier_cols:
        count = remove_outliers_iqr(df, col)
        print(f"  {col}: 处理 {count} 个异常值")
    
    # 6. 特征工程（数据增强）
    print("\n[6/7] 特征工程...")
    
    # 创建总面积特征（1楼+2楼+地下室）
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    
    # 创建房屋年龄特征
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    
    # 创建质量综合评分
    df['OverallQualCond'] = df['OverallQual'] * df['OverallCond']
    
    # 创建二元标记特征
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    
    print("新增特征: TotalSF, HouseAge, RemodAge, OverallQualCond")
    print("新增标记: HasBasement, HasGarage, Has2ndFloor, HasFireplace")
    
    # 7. 序数编码（将质量等级转换为数值）
    print("\n[7/7] 序数编码...")
    
    ordinal_mappings = {
        'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtQual': {'No_Basement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtCond': {'No_Basement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'FireplaceQu': {'No_Fireplace': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageQual': {'No_Garage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageCond': {'No_Garage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    }
    
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    print(f"完成 {len(ordinal_mappings)} 个序数特征编码")
    
    # 8. 最终验证
    print("\n" + "=" * 60)
    print("数据验证")
    print("=" * 60)
    
    # 检查缺失值
    remaining_missing = df.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    
    if len(remaining_missing) > 0:
        print(f"警告: 仍存在 {len(remaining_missing)} 个特征有缺失值:")
        print(remaining_missing)
    else:
        print("✓ 所有缺失值已处理完毕")
    
    # 验证数据一致性
    try:
        # 验证地下室面积一致性
        bsmt_check = (df['TotalBsmtSF'] == df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['BsmtUnfSF']).all()
        print(f"✓ 地下室面积一致性检查: {'通过' if bsmt_check else '失败'}")
        
        # 验证年龄非负
        age_check = (df['HouseAge'] >= 0).all()
        print(f"✓ 房屋年龄非负检查: {'通过' if age_check else '失败'}")
        
        # 验证车库年份合理性
        garage_year_check = (df['GarageYrBlt'] >= 1900).all()
        print(f"✓ 车库年份合理性检查: {'通过' if garage_year_check else '失败'}")
        
    except Exception as e:
        print(f"验证过程出错: {e}")
    
    # 9. 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"\n清洗后数据已保存至: {output_path}")
    
    # 返回统计信息
    stats = {
        'original_shape': original_shape,
        'cleaned_shape': df.shape,
        'dropped_columns': len(cols_to_drop),
        'new_features': 8,  # TotalSF, HouseAge, RemodAge, OverallQualCond, HasBasement, HasGarage, Has2ndFloor, HasFireplace
        'remaining_missing': len(remaining_missing),
        'features_count': df.shape[1]
    }
    
    print("\n" + "=" * 60)
    print("清洗完成统计")
    print("=" * 60)
    print(f"原始数据形状: {stats['original_shape']}")
    print(f"清洗后形状: {stats['cleaned_shape']}")
    print(f"删除特征数: {stats['dropped_columns']}")
    print(f"新增特征数: {stats['new_features']}")
    print(f"最终特征数: {stats['features_count']}")
    print(f"剩余缺失值: {stats['remaining_missing']}")
    
    return df, stats

# 主执行入口
if __name__ == "__main__":
    input_file = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_file = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    try:
        cleaned_df, statistics = clean_house_prices_data(input_file, output_file)
        print("\n数据清洗流程成功完成！")
    except Exception as e:
        print(f"\n数据清洗过程出错: {str(e)}")
        raise