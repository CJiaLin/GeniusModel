import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================
# 配置参数
# ============================================
DATA_PATH = '/Users/cjialin/code/AutoMLByLLM/train.csv'
OUTPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/assets/test_new_workflow_1774256909/data/cleaned_data.csv'

# 高缺失率列（需要删除）- 缺失率超过50%
COLS_TO_DROP = [
    'PoolQC',      # 99.52% 缺失
    'MiscFeature', # 96.30% 缺失  
    'Alley',       # 93.77% 缺失
    'Fence',       # 80.75% 缺失
    'MasVnrType'   # 59.73% 缺失
]

# 异常值比例过高且信息价值低的列（需要删除）
COLS_DROP_OUTLIER = [
    'BsmtFinSF2',    # 11.44% 异常值且多数为0
    'EnclosedPorch'  # 14.25% 异常值且多数为0
]

# 需要Winsorize（缩尾处理）的数值列
# 将极值限制在 [Q1-1.5×IQR, Q3+1.5×IQR] 范围内
NUM_COLS_TO_WINSORIZE = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch',
    'MiscVal', 'SalePrice'
]

# 分类变量列表（基于实际数据中的分类列）
CATEGORICAL_COLS = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
    'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PavedDrive', 'SaleType', 'SaleCondition'
]

# ============================================
# 数据加载与检查
# ============================================

def load_and_inspect_data(filepath):
    """加载数据并显示基本信息"""
    print("="*60)
    print("步骤1: 加载原始数据")
    print("="*60)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"原始数据形状: {df.shape}")
    print(f"总列数: {len(df.columns)}")
    print(f"\n前5列: {list(df.columns[:5])}")
    print(f"后5列: {list(df.columns[-5:])}")
    
    # 显示缺失值统计
    missing_stats = df.isnull().sum()
    missing_cols = missing_stats[missing_stats > 0].sort_values(ascending=False)
    if len(missing_cols) > 0:
        print(f"\n缺失值列数: {len(missing_cols)}")
        print("Top 5 缺失值列:")
        print(missing_cols.head())
    
    return df

# ============================================
# 缺失值处理
# ============================================

def remove_high_missing_columns(df):
    """删除高缺失率列（缺失率>50%）"""
    print("\n" + "="*60)
    print("步骤2: 删除高缺失率列")
    print("="*60)
    
    cols_to_drop = [col for col in COLS_TO_DROP if col in df.columns]
    df_clean = df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"删除的列: {cols_to_drop}")
    print(f"删除后形状: {df_clean.shape}")
    return df_clean

def remove_outlier_columns(df):
    """删除异常值比例过高且信息价值低的列"""
    print("\n" + "="*60)
    print("步骤3: 删除高异常值比例列")
    print("="*60)
    
    cols_to_drop = [col for col in COLS_DROP_OUTLIER if col in df.columns]
    df_clean = df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"删除的列: {cols_to_drop}")
    print(f"删除后形状: {df_clean.shape}")
    return df_clean

def impute_missing_values(df):
    """
    智能填充缺失值 - 基于房价数据的业务理解
    """
    print("\n" + "="*60)
    print("步骤4: 缺失值填充")
    print("="*60)
    
    df_imputed = df.copy()
    fill_stats = []
    
    # 1. FireplaceQu: 缺失表示无壁炉，填充为"None"
    if 'FireplaceQu' in df_imputed.columns:
        missing_count = df_imputed['FireplaceQu'].isnull().sum()
        df_imputed['FireplaceQu'] = df_imputed['FireplaceQu'].fillna('None')
        fill_stats.append(f"FireplaceQu: 填充 {missing_count} 个缺失值为 'None'")
    
    # 2. LotFrontage: 按Neighborhood分组中位数填充（同街区房屋临街距离相似）
    if 'LotFrontage' in df_imputed.columns:
        missing_before = df_imputed['LotFrontage'].isnull().sum()
        
        # 按Neighborhood分组填充
        df_imputed['LotFrontage'] = df_imputed.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # 如果仍有缺失（某些Neighborhood全为NA），用全局中位数填充
        global_median = df_imputed['LotFrontage'].median()
        df_imputed['LotFrontage'] = df_imputed['LotFrontage'].fillna(global_median)
        
        missing_after = df_imputed['LotFrontage'].isnull().sum()
        fill_stats.append(f"LotFrontage: 填充 {missing_before} 个缺失值（按Neighborhood中位数）")
    
    # 3. Garage相关列：缺失表示无车库
    garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cat_cols:
        if col in df_imputed.columns:
            missing_count = df_imputed[col].isnull().sum()
            df_imputed[col] = df_imputed[col].fillna('None')
            if missing_count > 0:
                fill_stats.append(f"{col}: 填充 {missing_count} 个缺失值为 'None'")
    
    # GarageYrBlt: 缺失表示无车库，填充为0
    if 'GarageYrBlt' in df_imputed.columns:
        missing_count = df_imputed['GarageYrBlt'].isnull().sum()
        df_imputed['GarageYrBlt'] = df_imputed['GarageYrBlt'].fillna(0)
        if missing_count > 0:
            fill_stats.append(f"GarageYrBlt: 填充 {missing_count} 个缺失值为 0")
    
    # 4. Bsmt（地下室）相关列：缺失表示无地下室
    bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in bsmt_cat_cols:
        if col in df_imputed.columns:
            missing_count = df_imputed[col].isnull().sum()
            df_imputed[col] = df_imputed[col].fillna('None')
            if missing_count > 0:
                fill_stats.append(f"{col}: 填充 {missing_count} 个缺失值为 'None'")
    
    # 5. MasVnrArea: 缺失表示无砌体贴面，填充为0
    if 'MasVnrArea' in df_imputed.columns:
        missing_count = df_imputed['MasVnrArea'].isnull().sum()
        df_imputed['MasVnrArea'] = df_imputed['MasVnrArea'].fillna(0)
        if missing_count > 0:
            fill_stats.append(f"MasVnrArea: 填充 {missing_count} 个缺失值为 0")
    
    # 6. Electrical: 用众数"SBrkr"（标准接线方式）填充
    if 'Electrical' in df_imputed.columns:
        missing_count = df_imputed['Electrical'].isnull().sum()
        df_imputed['Electrical'] = df_imputed['Electrical'].fillna('SBrkr')
        if missing_count > 0:
            fill_stats.append(f"Electrical: 填充 {missing_count} 个缺失值为 'SBrkr'（众数）")
    
    # 打印填充统计
    for stat in fill_stats:
        print(f"  ✓ {stat}")
    
    # 检查是否还有缺失值
    remaining_missing = df_imputed.isnull().sum().sum()
    print(f"\n剩余缺失值总数: {remaining_missing}")
    
    return df_imputed

# ============================================
# 异常值处理
# ============================================

def winsorize_outliers(df):
    """
    Winsorize（缩尾处理）：将极值限制在 [Q1-1.5×IQR, Q3+1.5×IQR] 范围内
    保留数据点但限制极端值的影响
    """
    print("\n" + "="*60)
    print("步骤5: 异常值处理（Winsorize）")
    print("="*60)
    
    df_winsorized = df.copy()
    outlier_stats = []
    
    for col in NUM_COLS_TO_WINSORIZE:
        if col in df_winsorized.columns and pd.api.types.is_numeric_dtype(df_winsorized[col]):
            # 计算IQR边界
            Q1 = df_winsorized[col].quantile(0.25)
            Q3 = df_winsorized[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 统计原始异常值数量
            original_outliers = ((df_winsorized[col] < lower_bound) | (df_winsorized[col] > upper_bound)).sum()
            
            # Winsorize: 将极值裁剪到边界内
            df_winsorized[col] = df_winsorized[col].clip(lower=lower_bound, upper=upper_bound)
            
            if original_outliers > 0:
                outlier_stats.append(f"{col}: 处理 {original_outliers} 个异常值 [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # 打印处理统计（只显示有异常值的列）
    for stat in outlier_stats[:10]:  # 只显示前10条避免输出过长
        print(f"  ✓ {stat}")
    if len(outlier_stats) > 10:
        print(f"  ... 还有 {len(outlier_stats)-10} 列已处理")
    
    print(f"总共处理 {len(outlier_stats)} 列的异常值")
    return df_winsorized

# ============================================
# 数据类型优化
# ============================================

def optimize_data_types(df):
    """优化内存使用：分类变量转category，整数类型降级"""
    print("\n" + "="*60)
    print("步骤6: 数据类型优化")
    print("="*60)
    
    df_optimized = df.copy()
    
    # 1. 将分类变量转换为category类型
    cat_converted = 0
    for col in CATEGORICAL_COLS:
        if col in df_optimized.columns and df_optimized[col].dtype == 'object':
            df_optimized[col] = df_optimized[col].astype('category')
            cat_converted += 1
    
    print(f"转换 {cat_converted} 个分类列为 category 类型")
    
    # 2. 将整数类型优化为更小的类型（减少内存）
    int_cols = df_optimized.select_dtypes(include=['int64']).columns
    int_optimized = 0
    for col in int_cols:
        if col != 'Id':  # 保留Id的原始类型
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            if c_min >= 0:  # 只处理无符号数
                if c_max < 255:
                    df_optimized[col] = df_optimized[col].astype(np.uint8)
                    int_optimized += 1
                elif c_max < 65535:
                    df_optimized[col] = df_optimized[col].astype(np.uint16)
                    int_optimized += 1
    
    print(f"优化 {int_optimized} 个整数列为更小的存储类型")
    
    # 计算内存节省
    mem_before = df.memory_usage(deep=True).sum() / 1024**2
    mem_after = df_optimized.memory_usage(deep=True).sum() / 1024**2
    print(f"内存使用: {mem_before:.2f} MB -> {mem_after:.2f} MB (节省 {mem_before-mem_after:.2f} MB)")
    
    return df_optimized

# ============================================
# 特征工程
# ============================================

def create_additional_features(df):
    """
    创建派生特征 - 基于房价数据的领域知识
    """
    print("\n" + "="*60)
    print("步骤7: 特征工程")
    print("="*60)
    
    df_featured = df.copy()
    features_created = []
    
    # 1. 房屋总面积（地上+地下室）
    area_cols = ['1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'TotalBsmtSF']
    available_area_cols = [col for col in area_cols if col in df_featured.columns]
    if len(available_area_cols) >= 2:
        df_featured['TotalSF'] = df_featured[available_area_cols].sum(axis=1)
        features_created.append(f"TotalSF ({', '.join(available_area_cols)})")
    
    # 2. 总浴室数（全浴室+半浴室*0.5，包括地下室）
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in df_featured.columns for col in bath_cols):
        df_featured['TotalBath'] = (df_featured['FullBath'] + 0.5 * df_featured['HalfBath'] + 
                                   df_featured['BsmtFullBath'] + 0.5 * df_featured['BsmtHalfBath'])
        features_created.append("TotalBath (全浴室+半浴室*0.5)")
    
    # 3. 房屋年龄（销售年份-建造年份）
    if all(col in df_featured.columns for col in ['YrSold', 'YearBuilt']):
        df_featured['HouseAge'] = df_featured['YrSold'] - df_featured['YearBuilt']
        features_created.append("HouseAge (销售年份-建造年份)")
    
    # 4. 装修后年数（销售年份-装修年份）
    if all(col in df_featured.columns for col in ['YrSold', 'YearRemodAdd']):
        df_featured['RemodAge'] = df_featured['YrSold'] - df_featured['YearRemodAdd']
        features_created.append("RemodAge (销售年份-装修年份)")
    
    # 5. 是否有车库（二值特征）
    if 'GarageArea' in df_featured.columns:
        df_featured