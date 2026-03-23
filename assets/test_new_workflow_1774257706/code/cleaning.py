import pandas as pd
import numpy as np
import os
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')

def main():
    # 设置数据路径
    input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_path = '/Users/cjialin/code/AutoMLByLLM/assets/test_new_workflow_1774257706/data/cleaned_data.csv'
    
    print("=" * 60)
    print("开始数据清洗流程")
    print("=" * 60)
    print(f"输入文件: {input_path}")
    
    # 1. 加载数据
    df = pd.read_csv(input_path)
    print(f"原始数据形状: {df.shape}")
    
    # 检查初始缺失值
    missing_before = df.isnull().sum().sum()
    print(f"初始缺失值总数: {missing_before}")
    
    # 2. 缺失值处理
    print("\n" + "-" * 40)
    print("步骤1: 处理缺失值")
    
    # 2.1 删除高缺失率列（>50%）
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        print(f"删除高缺失率列: {existing_cols_to_drop}")
    
    # 2.2 填充分类变量 - basement相关（缺失表示无地下室）
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # 2.3 填充分类变量 - garage相关（缺失表示无车库）
    garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # 2.4 填充数值变量 - GarageYrBlt（缺失表示无车库）
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    
    # 2.5 填充分类变量 - FireplaceQu（缺失表示无壁炉）
    if 'FireplaceQu' in df.columns:
        df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
    
    # 2.6 填充数值变量 - MasVnrArea（缺失表示无砌体贴面）
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # 2.7 填充Electrical（单一缺失值用众数）
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    # 2.8 智能填充LotFrontage（按Neighborhood分组中位数）
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果仍有缺失（某些Neighborhood全是NaN），用整体中位数
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    print("缺失值处理完成")