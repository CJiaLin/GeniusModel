import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# 文件路径配置
INPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/assets/test_new_workflow_1774258600/data/cleaned_data.csv'
OUTPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/assets/test_new_workflow_1774258600/data/cleaned_data_features.csv'
TARGET_COL = 'SalePrice'

def safe_divide(numerator, denominator, fill_value=0):
    """安全除法，避免除以零"""
    return np.where(denominator != 0, numerator / denominator, fill_value)

def feature_engineering_pipeline(input_path, output_path, target_col):
    """
    完整的特征工程流程
    基于实际数据列名：Id, MSSubClass, LotFrontage, LotArea, OverallQual, OverallCond, 
    YearBuilt, YearRemodAdd, MasVnrArea, BsmtFinSF1, BsmtUnfSF, TotalBsmtSF, 
    1stFlrSF, 2ndFlrSF, LowQualFinSF, MSZoning, Street, LotShape, LandContour, 
    Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2, 
    BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st
    """
    print("=" * 50)
    print("开始特征工程流程...")
    print("=" * 50)
    
    # 1. 加载数据
    print(f"\n[1/6] 加载数据: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    df = pd.read_csv(input_path)
    original_shape = df.shape
    print(f"原始数据形状: {original_shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查目标列
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不在数据中。可用列: {df.columns.tolist()}")
    
    #