import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

def main():
    """主函数：执行完整的数据清洗流程"""
    
    # 定义文件路径
    input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    print("=" * 60)
    print("开始数据清洗流程")
    print("=" * 60)
    
    # Step 1: 加载数据
    print(f"\n步骤 1: 加载数据 from {input_path}")
    try:
        df = pd.read_csv(input_path)
        original_shape = df.shape
        print(f"✓ 原始数据加载成功: {original_shape[0]} 行 × {original_shape[1]} 列")
    except FileNotFoundError:
        print(f"✗ 错误: 文件 {input_path} 不存在，请检查路径")
        return
    except Exception as e:
        print(f"✗ 加载数据时发生错误: {str(e)}")
        return
    
    # 记录原始信息用于对比
    original_missing = df.isnull().sum().sum()
    
    # Step 2: 数据类型转换
    print("\n步骤 2: 数据类型转换与特征工程...")
    
    # 将 MSSubClass 转换为类别型（建筑类型标识码，非数值意义）
    if 'MSSubClass' in df.columns:
        df['MSSubClass'] = df['MSSubClass'].astype(str)
        print("  ✓ MSSubClass 转换为类别型")
    
    # 时间相关特征转为类别型
    if 'MoSold' in df.columns:
        df['MoSold'] = df['MoSold'].astype(str)
        print("  ✓ MoSold 转换为类别型")
    
    if 'YrSold' in df.columns:
        df['YrSold'] = df['YrSold'].astype(str)
        print("  ✓ YrSold 转换为类别型")
    
    # 创建房屋年龄特征（特征工程）
    if all(col in df.columns for col in ['YrSold', 'YearBuilt']):
        df['HouseAge'] = df['YrSold'].astype(int) - df['YearBuilt']
        print("  ✓ 创建 HouseAge 特征")
    
    if all(col in df.columns for col in ['YrSold', 'YearRemodAdd']):
        df['RemodAge'] = df['YrSold'].astype(int) - df['YearRemodAdd']
        print("  ✓ 创建 RemodAge 特征")
    
    # Step 3: 缺失值处理
    print("\n步骤 3: 缺失值处理...")
    
    # 3.1 极高缺失率列处理（PoolQC, MiscFeature）
    cols_to_drop = []
    if 'PoolQC' in df.columns:
        cols_to_drop.append('PoolQC')
    if 'MiscFeature' in df.columns:
        cols_to_drop.append('MiscFeature')
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  ✓ 删除极高缺失率列: {cols_to_drop}")
    
    # 3.2 填充"无此设施"类别
    none_cols = ['Alley', 'Fence', 'FireplaceQu', 
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    
    filled_count = 0
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
            filled_count += 1
    if filled_count > 0:
        print(f"  ✓ 填充 'None' 到 {filled_count} 个类别型特征")
    
    # 3.3 数值型缺失处理 - LotFrontage 按社区中位数填充
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果仍有缺失（某些社区可能全为NaN），用总体中位数填充
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
        print("  ✓ LotFrontage 按 Neighborhood 中位数填充")
    
    # 3.4 其余少量缺失值填充
    if 'Electrical' in df.columns and df['Electrical'].isnull().sum() > 0:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
        print("  ✓ Electrical 填充为众数")
    
    if 'MasVnrType' in df.columns:
        df['MasVnrType'] = df['MasVnrType'].fillna('None')
        print("  ✓ MasVnrType 填充为 'None'")
    
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
        print("  ✓ MasVnrArea 填充为 0")
    
    # 处理 GarageYrBlt 等数值型但表示缺失的列
    garage_num_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars']
    for col in garage_num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 处理地下室数值型特征
    bsmt_num_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                     'BsmtFullBath', 'BsmtHalfBath']
    for col in bsmt_num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    print("  ✓ 车库和地下室数值特征缺失值填充为 0")
    
    # Step 4: 异常值处理
    print("\n步骤 4: 异常值处理...")
    
    # 4.1 检测并处理 GrLivArea 异常值（删除 > 4000 的记录）
    if 'GrLivArea' in df.columns:
        outliers = df[df['GrLivArea'] > 4000].index
        outlier_count = len(outliers)
        if outlier_count > 0:
            df = df.drop(outliers)
            print(f"  ✓ 删除 {outlier_count} 个 GrLivArea > 4000 的异常值")
        else:
            print("  ✓ 未发现 GrLivArea 异常值")
    
    # 4.2 对目标变量进行对数变换（处理右偏）
    if 'SalePrice' in df.columns:
        df['SalePrice_Log'] = np.log1p(df['SalePrice'])
        print("  ✓ 创建 SalePrice_Log 对数变换特征")
    
    # Step 5: 特征工程
    print("\n步骤 5: 特征工程...")
    
    # 5.1 合并车库相关特征
    if all(col in df.columns for col in ['GarageArea', 'GarageCars']):
        df['TotalGarageSF'] = df['GarageArea'] * df['GarageCars']
        print("  ✓ 创建 TotalGarageSF 特征")
    
    # 5.2 合并地下室面积
    bsmt_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
    if all(col in df.columns for col in bsmt_cols):
        df['TotalBsmtSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['BsmtUnfSF']
        print("  ✓ 创建 TotalBsmtSF 特征")
    
    # 5.3 总居住面积
    sf_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    if all(col in df.columns for col in sf_cols):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        print("  ✓ 创建 TotalSF 特征")
    elif all(col in df.columns for col in ['1stFlrSF', '2ndFlrSF']):
        # 如果没有 TotalBsmtSF，使用 1stFlrSF + 2ndFlrSF
        df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF']
        print("  ✓ 创建 TotalSF 特征（不含地下室）")
    
    # 5.4 总浴室数
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in df.columns for col in bath_cols):
        df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                          df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
        print("  ✓ 创建 TotalBath 特征")
    
    # 5.5 整体质量评分
    if all(col in df.columns for col in ['OverallQual', 'OverallCond']):
        df['OverallScore'] = df['OverallQual'] * df['OverallCond']
        print("  ✓ 创建 OverallScore 特征")
    
    # Step 6: 类别型变量编码
    print("\n步骤 6: 类别型变量编码...")
    
    # 6.1 有序类别编码（保持顺序关系）
    ordinal_mappings = {
        'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    }
    
    ordinal_encoded = 0
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            ordinal_encoded += 1
    
    if ordinal_encoded > 0:
        print(f"  ✓ 完成 {ordinal_encoded} 个有序类别特征的映射编码")
    
    # 6.2 剩余类别型变量 Label Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
        print(f"  ✓ 完成 {len(categorical_cols)} 个类别特征的 Label Encoding")
    else:
        print("  ✓ 无剩余类别型特征需要编码")
    
    # Step 7: 数据验证与保存
    print("\n步骤 7: 数据验证与保存...")
    
    # 7.1 验证无缺失值
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing == 0:
        print("  ✓ 验证通过：数据中无缺失值")
    else:
        print(f"  ⚠ 警告：仍存在 {remaining_missing} 个缺失值")
        missing_cols = df.isnull().sum()[df.isnull().sum() > 0]
        print(f"    缺失值分布: {missing_cols.to_dict()}")
        # 对于任何剩余的缺失值，用中位数或0填充
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        print("  ✓ 自动填充剩余缺失值")
    
    # 7.2 保存清洗后数据
    try:
        df.to_csv(output_path, index=False)
        print(f"  ✓ 清洗后数据已保存至: {output_path}")
    except Exception as e:
        print(f"  ✗ 保存文件时发生错误: {str(e)}")
        return
    
    # 打印清洗结果统计
    print("\n" + "=" * 60)
    print("数据清洗完成 - 结果统计")
    print("=" * 60)
    print(f"原始数据形状: {original_shape[0]} 行 × {original_shape[1]} 列")
    print(f"清洗后数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"删除行数: {original_shape[0] - df.shape[0]} 行 (异常值)")
    print(f"删除/新增列数: {df.shape[1] - original_shape[1]} 列")
    print(f"原始缺失值总数: {original_missing}")
    print(f"当前缺失值总数: {df.isnull().sum().sum()}")
    print(f"数值型特征数: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"目标变量: SalePrice 和 SalePrice_Log")
    print("=" * 60)

if __name__ == "__main__":
    main()