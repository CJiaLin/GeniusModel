import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def clean_housing_data(file_path, output_path):
    """
    房屋数据清洗完整流程
    """
    # 1. 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    print(f"原始数据列数: {len(df.columns)}")
    
    # 2. 处理缺失值
    # 2.1 LotFrontage (数值型) - 按Neighborhood分组使用中位数填充，因为同邻域房屋临街距离相似
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果仍有缺失（某些Neighborhood全缺失），用全局中位数填充
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # 2.2 Alley (分类型) - NA表示'没有巷道'，填充为'None'
    if 'Alley' in df.columns:
        df['Alley'] = df['Alley'].fillna('None')
    
    # 2.3 MasVnrType (分类型) - 填充为'None'表示无砌体饰面
    if 'MasVnrType' in df.columns:
        df['MasVnrType'] = df['MasVnrType'].fillna('None')
    
    # 2.4 MasVnrArea (数值型) - 如果类型为None或缺失，面积应为0
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # 2.5 地下室相关列 (BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2)
    # NA表示'无地下室'，填充为'No'
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        if col in df.columns:
            df[col] = df[col].fillna('No')
    
    # 2.6 Electrical (分类型) - 用众数填充
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0] if not df['Electrical'].mode().empty else 'SBrkr')
    
    # 3. 特征工程
    # 3.1 房屋年龄和改造年龄
    if 'YearBuilt' in df.columns and 'YearRemodAdd' in df.columns:
        current_year = 2024
        df['HouseAge'] = current_year - df['YearBuilt']
        df['RemodAge'] = current_year - df['YearRemodAdd']
        df['IsNew'] = (df['YearBuilt'] >= 2000).astype(int)
        df['HasRemod'] = (df['YearRemodAdd'] > df['YearBuilt']).astype(int)
    
    # 3.2 总面积计算（如果存在相关列）
    area_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    if all(col in df.columns for col in area_cols):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # 3.3 地下室完成比例
    if 'BsmtFinSF1' in df.columns and 'TotalBsmtSF' in df.columns:
        df['BsmtFinRatio'] = (df['BsmtFinSF1'] + df.get('BsmtFinSF2', 0)) / (df['TotalBsmtSF'] + 1)
    
    # 4. 异常值处理
    # 4.1 LotArea极端值（使用IQR方法）
    if 'LotArea' in df.columns:
        Q1 = df['LotArea'].quantile(0.25)
        Q3 = df['LotArea'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR  # 使用3倍IQR保留更多数据
        df['LotArea'] = np.where(df['LotArea'] > upper_bound, upper_bound, df['LotArea'])
    
    # 4.2 确保所有面积值为非负
    area_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
                   'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    for col in area_columns:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    
    # 5. 删除无关列
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # 6. 分类变量编码
    # 获取所有分类列（object类型）
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 对有序分类变量使用标签编码，无序使用One-Hot
    ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                   'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    
    # 实际存在的有序列
    existing_ordinal = [col for col in ordinal_cols if col in df.columns]
    
    # 质量等级映射
    quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'No': 0, 'None': 0}
    
    for col in existing_ordinal:
        df[col] = df[col].map(quality_map).fillna(0)
    
    # 对其他分类变量进行One-Hot编码
    nominal_cols = [col for col in categorical_cols if col not in existing_ordinal]
    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    # 7. 数据类型优化
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # 8. 统计信息
    missing_after = df.isnull().sum().sum()
    print(f"清洗后数据形状: {df.shape}")
    print(f"处理后缺失值总数: {missing_after}")
    print(f"新增特征数量: {df.shape[1] - 81 + 1}")  # +1因为删除了Id
    
    # 9. 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"清洗后数据已保存至: {output_path}")
    
    return df

if __name__ == "__main__":
    input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    cleaned_df = clean_housing_data(input_path, output_path)
    
    # 打印详细统计
    print("\n=== 清洗完成统计 ===")
    print(f"最终数据集形状: {cleaned_df.shape}")
    print(f"数值型特征数: {len(cleaned_df.select_dtypes(include=[np.number]).columns)}")
    print(f"是否存在缺失值: {cleaned_df.isnull().sum().sum() > 0}")