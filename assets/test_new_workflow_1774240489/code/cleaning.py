import pandas as pd
import numpy as np
import os

def clean_housing_data(input_path, output_path):
    """
    清洗Ames Housing数据集
    参数:
        input_path: 原始数据路径
        output_path: 清洗后数据保存路径
    返回:
        清洗后的DataFrame
    """
    print(f"正在加载数据: {input_path}")
    
    # 加载数据
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")
    
    df = pd.read_csv(input_path)
    original_shape = df.shape
    print(f"原始数据形状: {original_shape}")
    print(f"原始缺失值总数: {df.isnull().sum().sum()}")
    
    # ===== 步骤1: 处理缺失值 =====
    
    # 1.1 '不存在'类型缺失值 → 填充为 'None'（分类变量）
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType']
    
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # 1.2 '不存在'类型缺失值 → 填充为 0（数值变量）
    zero_cols = ['GarageYrBlt', 'MasVnrArea']
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 1.3 基于统计的填充
    # LotFrontage: 按Neighborhood分组中位数填充
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果仍有缺失（某些Neighborhood全是NaN），用全局中位数填充
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # Electrical: 众数填充（仅1个缺失）
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    # ===== 步骤2: 数据类型转换 =====
    # MSSubClass是分类变量（住宅类型代码），转为字符串
    if 'MSSubClass' in df.columns:
        df['MSSubClass'] = df['MSSubClass'].astype(str)
    
    # GarageYrBlt转为整数（已填充0表示无车库）
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].astype(int)
    
    # ===== 步骤3: 特征工程 =====
    
    # 房龄相关特征
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)
        df['HasRemod'] = (df['YearRemodAdd'] > df['YearBuilt']).astype(int)
    
    # 总面积特征
    area_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    if all(col in df.columns for col in area_cols):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # 门廊总面积
    porch_cols = ['OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF']
    available_porch_cols = [col for col in porch_cols if col in df.columns]
    if available_porch_cols:
        df['TotalPorchSF'] = df[available_porch_cols].sum(axis=1)
    
    # 浴室总数（全浴+半浴*0.5，包括地下室）
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in df.columns for col in bath_cols):
        df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                           df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    
    # 设施存在性二元特征
    if 'PoolArea' in df.columns:
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    
    if '2ndFlrSF' in df.columns:
        df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    
    if 'GarageArea' in df.columns:
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    
    if 'TotalBsmtSF' in df.columns:
        df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    
    if 'Fireplaces' in df.columns:
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    
    if 'Fence' in df.columns:
        df['HasFence'] = (df['Fence'] != 'None').astype(int)
    
    # ===== 步骤4: 有序分类变量编码 =====
    
    # 质量等级映射
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    
    ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    
    for col in ordinal_cols:
        if col in df.columns:
            df[col] = df[col].map(quality_map).fillna(0).astype(int)
    
    # BsmtExposure有序编码
    if 'BsmtExposure' in df.columns:
        exposure_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
        df['BsmtExposure'] = df['BsmtExposure'].map(exposure_map).fillna(0).astype(int)
    
    # Functional有序编码
    if 'Functional' in df.columns:
        functional_map = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 
                          'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}
        df['Functional'] = df['Functional'].map(functional_map).fillna(8).astype(int)
    
    # ===== 保存结果 =====
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(output_path, index=False)
    print(f"清洗后数据已保存至: {output_path}")
    
    # ===== 统计信息 =====
    final_shape = df.shape
    remaining_missing = df.isnull().sum().sum()
    
    print("\n" + "="*50)
    print("数据清洗统计报告")
    print("="*50)
    print(f"原始数据形状: {original_shape}")
    print(f"清洗后数据形状: {final_shape}")
    print(f"新增特征数量: {final_shape[1] - original_shape[1]}")
    print(f"剩余缺失值总数: {remaining_missing}")
    
    if remaining_missing > 0:
        print("\n各列剩余缺失值:")
        missing_cols = df.columns[df.isnull().any()].tolist()
        for col in missing_cols:
            print(f"  {col}: {df[col].isnull().sum()}")
    else:
        print("数据完整性: 100% (无缺失值)")
    
    print("="*50)
    
    return df

# 执行清洗
if __name__ == "__main__":
    input_file = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_file = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    try:
        df_cleaned = clean_housing_data(input_file, output_file)
        print("\n数据清洗成功完成！")
    except Exception as e:
        print(f"\n数据清洗失败: {str(e)}")
        raise