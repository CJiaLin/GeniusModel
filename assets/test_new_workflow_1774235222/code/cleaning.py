import pandas as pd
import numpy as np
import os

def clean_housing_data(file_path, output_path):
    """
    Ames Housing数据集清洗函数
    
    参数:
        file_path: 原始数据路径
        output_path: 清洗后数据保存路径
    
    返回:
        df: 清洗后的DataFrame
        report: 清洗报告字典
    """
    print(f"开始加载数据: {file_path}")
    
    # 1. 加载数据
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    df = pd.read_csv(file_path)
    original_shape = df.shape
    original_missing = df.isnull().sum().sum()
    
    print(f"原始数据形状: {original_shape}")
    print(f"原始缺失值总数: {original_missing}")
    
    # 2. 检查ID唯一性（标识符处理）
    if 'Id' in df.columns:
        if not df['Id'].is_unique:
            print("警告: ID列存在重复值")
        else:
            print("验证通过: ID列无重复")
    
    # 3. 高缺失率分类特征处理 - 填充'None'表示无此设施
    # 这些特征的NA表示'没有该设施'，需要保留这一信息
    none_features = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 
        'FireplaceQu', 'MasVnrType', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'
    ]
    
    # 只处理实际存在的列（避免KeyError）
    existing_none_features = [col for col in none_features if col in df.columns]
    if existing_none_features:
        df[existing_none_features] = df[existing_none_features].fillna('None')
        print(f"已填充为'None'的特征: {len(existing_none_features)}个")
    
    # 4. 数值型缺失值处理
    # 4.1 MasVnrArea: 无贴面则面积为0
    if 'MasVnrArea' in df.columns:
        missing_before = df['MasVnrArea'].isnull().sum()
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
        print(f"MasVnrArea: 已填充{missing_before}个缺失值为0")
    
    # 4.2 GarageYrBlt: 无车库时填充为YearBuilt（假设与房子同时建造）
    if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
        missing_before = df['GarageYrBlt'].isnull().sum()
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
        print(f"GarageYrBlt: 已填充{missing_before}个缺失值为YearBuilt")
    
    # 4.3 LotFrontage: 按Neighborhood分组填充中位数
    # 街区(Neighborhood)相似的房屋通常有相似的街道正面长度
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        missing_before = df['LotFrontage'].isnull().sum()
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果某些Neighborhood全是NaN，用整体中位数填充剩余缺失值
        remaining_missing = df['LotFrontage'].isnull().sum()
        if remaining_missing > 0:
            overall_median = df['LotFrontage'].median()
            df['LotFrontage'] = df['LotFrontage'].fillna(overall_median)
            print(f"LotFrontage: 分组填充后剩余{remaining_missing}个，已用整体中位数{overall_median}填充")
        else:
            print(f"LotFrontage: 已按Neighborhood分组填充{missing_before}个缺失值")
    
    # 5. 电气系统填充 - 只有一个缺失，用众数SBrkr（标准断路器）填充
    if 'Electrical' in df.columns:
        missing_before = df['Electrical'].isnull().sum()
        if missing_before > 0:
            mode_value = df['Electrical'].mode()[0] if not df['Electrical'].mode().empty else 'SBrkr'
            df['Electrical'] = df['Electrical'].fillna(mode_value)
            print(f"Electrical: 已填充{missing_before}个缺失值为'{mode_value}'")
    
    # 6. 数据类型修正
    # MSSubClass是建筑类型代码，应视为分类变量而非连续数值
    if 'MSSubClass' in df.columns:
        df['MSSubClass'] = df['MSSubClass'].astype(str)
        print("MSSubClass: 已转换为字符串类型")
    
    # 7. 验证清洗结果
    final_missing = df.isnull().sum().sum()
    filled_count = original_missing - final_missing
    
    print(f"\n=== 清洗结果统计 ===")
    print(f"清洗后数据形状: {df.shape}")
    print(f"填充缺失值数量: {filled_count}")
    print(f"剩余缺失值数量: {final_missing}")
    
    # 8. 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"清洗后数据已保存至: {output_path}")
    
    # 9. 生成详细报告
    report = {
        'original_shape': original_shape,
        'final_shape': df.shape,
        'original_missing_values': int(original_missing),
        'final_missing_values': int(final_missing),
        'filled_missing_values': int(filled_count),
        'none_filled_features': existing_none_features,
        'processing_details': {
            'MasVnrArea': 'filled with 0 (no veneer)',
            'GarageYrBlt': 'filled with YearBuilt (built with house)',
            'LotFrontage': 'filled with Neighborhood median',
            'Electrical': 'filled with mode (SBrkr)',
            'MSSubClass': 'converted from int to string'
        }
    }
    
    return df, report

# 主执行程序
if __name__ == "__main__":
    # 定义输入输出路径
    INPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    OUTPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    try:
        # 执行数据清洗
        df_cleaned, report = clean_housing_data(INPUT_PATH, OUTPUT_PATH)
        
        # 打印详细报告
        print("\n" + "="*50)
        print("数据清洗完成报告")
        print("="*50)
        print(f"输入文件: {INPUT_PATH}")
        print(f"输出文件: {OUTPUT_PATH}")
        print(f"原始数据: {report['original_shape'][0]}行 × {report['original_shape'][1]}列")
        print(f"缺失值处理: {report['filled_missing_values']}个已填充")
        print(f"数据质量: 剩余缺失值{report['final_missing_values']}个")
        print(f"分类特征处理: {len(report['none_filled_features'])}个高缺失率特征标记为'None'")
        print("="*50)
        
    except Exception as e:
        print(f"清洗过程出错: {str(e)}")
        raise