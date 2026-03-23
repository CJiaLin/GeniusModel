import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_housing_data(file_path):
    """
    房价预测数据清洗函数
    基于清洗方案实现：删除高缺失率列、近零方差列、智能填充缺失值、
    Winsorize异常值处理、数据类型转换
    
    参数: file_path - 原始数据文件路径
    返回: 清洗后的DataFrame
    """
    
    # 加载数据
    print(f"正在加载数据: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
        return None
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None
        
    print(f"原始数据形状: {df.shape}")
    
    # 记录原始信息用于对比
    original_shape = df.shape
    original_missing = df.isnull().sum().sum()
    
    # ==========================================
    # 步骤1: 删除高缺失率列(>50%)
    # ==========================================
    # PoolQC(99.52%), MiscFeature(96.3%), Alley(93.77%), Fence(80.75%), MasVnrType(59.73%)
    high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    
    # 只删除实际存在的列
    cols_to_drop = [col for col in high_missing_cols if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"删除高缺失率列(>50%): {cols_to_drop}")
        print(f"删除后数据形状: {df.shape}")
    
    # ==========================================
    # 步骤2: 删除近零方差列
    # ==========================================
    # BsmtFinSF2(绝大多数为0), EnclosedPorch(绝大多数为0)
    near_zero_variance_cols = ['BsmtFinSF2', 'EnclosedPorch']
    
    cols_to_drop = [col for col in near_zero_variance_cols if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"删除近零方差列: {cols_to_drop}")
        print(f"删除后数据形状: {df.shape}")
    
    # ==========================================
    # 步骤3: 智能填充缺失值
    # ==========================================
    
    # 3.1 分类变量 - 填充"None"表示该设施不存在
    # FireplaceQu(47.26%缺失): 无壁炉
    # BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2: 无地下室
    # GarageType, GarageFinish, GarageQual, GarageCond: 无车库
    none_fill_cols = [
        'FireplaceQu', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
        'BsmtFinType1', 'BsmtFinType2', 'GarageType', 
        'GarageFinish', 'GarageQual', 'GarageCond'
    ]
    
    for col in none_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # 3.2 GarageYrBlt - 无车库填充为房屋建造年份(YearBuilt)
    if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
        print("GarageYrBlt 缺失值已用 YearBuilt 填充")
    
    # 3.3 LotFrontage(17.74%缺失) - 按Neighborhood分组用中位数填充
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果还有缺失（新社区），用整体中位数
        remaining_missing = df['LotFrontage'].isnull().sum()
        if remaining_missing > 0:
            overall_median = df['LotFrontage'].median()
            df['LotFrontage'] = df['LotFrontage'].fillna(overall_median)
            print(f"LotFrontage 剩余{remaining_missing}个缺失值已用整体中位数填充")
        else:
            print("LotFrontage 已按Neighborhood分组中位数填充完成")
    
    #