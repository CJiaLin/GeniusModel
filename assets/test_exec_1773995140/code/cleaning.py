import pandas as pd
import numpy as np
import os
from datetime import datetime

def clean_housing_data(input_path, output_path):
    """
    Ames Housing 数据清洗函数
    基于数据清洗方案报告实现完整的清洗流程
    """
    
    print(f"正在读取数据: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")
    
    # 读取数据
    df = pd.read_csv(input_path)
    original_shape = df.shape
    original_columns = set(df.columns)
    print(f"原始数据形状: {original_shape}")
    
    # 复制数据避免修改原数据
    data = df.copy()
    
    # 统计清洗前的缺失值
    missing_before = data.isnull().sum().sum()
    missing_stats_before = data.isnull().sum()[data.isnull().sum() > 0].to_dict()
    
    # 步骤1: 处理"不存在"型缺失（分类变量）- 用"None"填充
    # 这些字段缺失表示该设施不存在（如PoolQC缺失表示无游泳池）
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 
                 'FireplaceQu', 'MasVnrType', 'BsmtQual', 'BsmtCond',
                 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    
    existing_none_cols = [col for col in none_cols if col in data.columns]
    if existing_none_cols:
        data[existing_none_cols] = data[existing_none_cols].fillna('None')
        print(f"已填充'不存在'型分类变量: {len(existing_none_cols)}列")
    
    # 步骤2: 处理"不存在"型缺失（数值变量）
    # GarageYrBlt缺失表示无车库，填0
    if 'GarageYrBlt' in data.columns:
        data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)
        print("已填充GarageYrBlt为0")
    
    # MasVnrArea缺失表示无砌体饰面，填0
    if 'MasVnrArea' in data.columns:
        data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
        print("已填充MasVnrArea为0")
    
    # 步骤3: 智能填充LotFrontage（按Neighborhood中位数）
    # 同一街区的临街距离通常相似，使用分组中位数更合理
    if 'LotFrontage' in data.columns and 'Neighborhood' in data.columns:
        data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果仍有缺失（新社区），用总体中位数填充
        if data['LotFrontage'].isnull().sum() > 0:
            data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
        print("已填充LotFrontage（按Neighborhood中位数）")
    
    # 步骤4: 单值填充（Electrical用众数）
    if 'Electrical' in data.columns:
        mode_val = data['Electrical'].mode()
        if len(mode_val) > 0:
            data['Electrical'] = data['Electrical'].fillna(mode_val[0])
            print("已填充Electrical为众数")
    
    # 步骤5: 数据类型转换（MSSubClass转为字符串，虽然是数字但代表类别）
    if 'MSSubClass' in data.columns:
        data['MSSubClass'] = data['MSSubClass'].astype(str)
        print("已转换MSSubClass为字符串类型")
    
    # 步骤6: 有序分类变量映射为数值（保持顺序关系）
    # 质量等级从差到好：None(0) < Po(1) < Fa(2) < TA(3) < Gd(4) < Ex(5)
    quality_mapping = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
    
    mapped_count = 0
    for col in quality_cols:
        if col in data.columns:
            data[col] = data[col].map(quality_mapping)
            # 如果有未映射的值（如原本就是NaN），填充为0
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(0)
            mapped_count += 1
    print(f"已映射{mapped_count}个质量等级列为数值")
    
    # 步骤7: 特征工程（创建新特征）
    # 7.1 房屋年龄特征（比年份更直观）
    if all(col in data.columns for col in ['YrSold', 'YearBuilt']):
        data['HouseAge'] = data['YrSold'] - data['YearBuilt']
        print("已创建HouseAge特征")
    
    # 7.2 翻新年龄（距离售出的年数）
    if all(col in data.columns for col in ['YrSold', 'YearRemodAdd']):
        data['RemodAge'] = data['YrSold'] - data['YearRemodAdd']
        print("已创建RemodAge特征")
    
    # 7.3 是否新房（售出年份等于建造年份）
    if all(col in data.columns for col in ['YrSold', 'YearBuilt']):
        data['IsNew'] = (data['YrSold'] == data['YearBuilt']).astype(int)
        print("已创建IsNew特征")
    
    # 7.4 总面积特征（地下室+一层+二层）
    sf_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    if all(col in data.columns for col in sf_cols):
        data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
        print("已创建TotalSF特征")
    
    # 7.5 门廊总面积（所有室外平台面积之和）
    porch_cols = ['OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF']
    existing_porch_cols = [col for col in porch_cols if col in data.columns]
    if existing_porch_cols:
        data['TotalPorchSF'] = data[existing_porch_cols].sum(axis=1)
        print("已创建TotalPorchSF特征")
    
    # 7.6 浴室总数（全浴室按1算，半浴室按0.5算）
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in data.columns for col in bath_cols):
        data['TotalBath'] = (data['FullBath'] + 0.5 * data['HalfBath'] + 
                             data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath'])
        print("已创建TotalBath特征")
    
    # 步骤8: 检查并处理其他可能的缺失值（如GarageCars, GarageArea等数值型车库字段）
    garage_numeric_cols = ['GarageCars', 'GarageArea']
    for col in garage_numeric_cols:
        if col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                data[col] = data[col].fillna(0)
                print(f"已填充{col}为0（{missing_count}个缺失值）")
    
    # 检查地下室相关数值字段
    bsmt_numeric_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
    for col in bsmt_numeric_cols:
        if col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                data[col] = data[col].fillna(0)
                print(f"已填充{col}为0（{missing_count}个缺失值）")
    
    # 检查地下室浴室字段
    bsmt_bath_cols = ['BsmtFullBath', 'BsmtHalfBath']
    for col in bsmt_bath_cols:
        if col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                data[col] = data[col].fillna(0)
                print(f"已填充{col}为0（{missing_count}个缺失值）")
    
    # 步骤9: 保存清洗后的数据
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    data.to_csv(output_path, index=False)
    print(f"清洗后的数据已保存到: {output_path}")
    
    # 步骤10: 生成清洗统计报告
    missing_after = data.isnull().sum().sum()
    new_columns = set(data.columns) - original_columns
    removed_columns = original_columns - set(data.columns)
    
    stats = {
        '原始数据形状': original_shape,
        '清洗后数据形状': data.shape,
        '新增列数': len(new_columns),
        '新增列名': list(new_columns),
        '删除列数': len(removed_columns),
        '清洗前缺失值总数': int(missing_before),
        '清洗后缺失值总数': int(missing_after),
        '缺失值减少率': f"{((missing_before - missing_after) / max(missing_before, 1) * 100):.2f}%",
        '清洗前含缺失值的列': len(missing_stats_before),
        '清洗后含缺失值的列': int((data.isnull().sum() > 0).sum()),
        '数据类型转换': 'MSSubClass转为字符串，质量等级列转为数值',
        '特征工程': f"创建了{len(new_columns)}个新特征"
    }
    
    return data, stats

if __name__ == "__main__":
    # 定义输入输出路径
    input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    print("=" * 60)
    print("Ames Housing 数据清洗开始")
    print("=" * 60)
    
    try:
        # 执行清洗
        cleaned_df, statistics = clean_housing_data(input_path, output_path)
        
        print("\n" + "=" * 60)
        print("清洗结果统计报告")
        print("=" * 60)
        
        for key, value in statistics.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"{key}: {value[:5]}... 等{len(value)}个")
            else:
                print(f"{key}: {value}")
        
        print("=" * 60)
        print("数据清洗完成！")
        print("=" * 60)
        
        # 显示前几行数据预览
        print("\n清洗后数据预览（前5行）:")
        print(cleaned_df.head())
        
        # 显示数据类型信息
        print("\n数据类型概览:")
        print(f"数值型列: {len(cleaned_df.select_dtypes(include=[np.number]).columns)}")
        print(f"分类型列: {len(cleaned_df.select_dtypes(include=['object']).columns)}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()