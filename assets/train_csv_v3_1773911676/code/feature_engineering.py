以下是完整的 Python 特征工程代码：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征工程脚本 - 房价预测数据集
数据路径: /Users/cjialin/code/AutoMLByLLM/train.csv
目标列: SalePrice
任务类型: regression
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_and_initial_clean(file_path):
    """
    加载数据并进行初步清理
    """
    print("=" * 60)
    print("步骤 1: 加载数据")
    print("=" * 60)
    
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    print(f"特征数量: {df.shape[1] - 1}")
    print(f"样本数量: {df.shape[0]}")
    
    # 分离目标变量
    target_col = 'SalePrice'
    if target_col in df.columns:
        y = df[target_col].copy()
        df = df.drop(columns=[target_col])
        print(f"目标变量 {target_col} 已分离")
    else:
        y = None
    
    # 保存 ID 列（如果有）
    id_col = 'Id' if 'Id' in df.columns else None
    if id_col:
        ids = df[id_col].copy()
        df = df.drop(columns=[id_col])
    else:
        ids = None
    
    return df, y, ids, id_col

def analyze_features(df):
    """
    分析特征类型
    """
    print("\n" + "=" * 60)
    print("步骤 2: 特征分析")
    print("=" * 60)
    
    # 区分数值型和分类型特征
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"数值型特征数量: {len(numeric_features)}")
    print(f"分类型特征数量: {len(categorical_features)}")
    
    # 检查缺失值
    missing_info = df.isnull().sum()
    missing_features = missing_info[missing_info > 0]
    print(f"\n包含缺失值的特征数量: {len(missing_features)}")
    if len(missing_features) > 0:
        print("缺失值详情:")
        for feat, count in missing_features.items():
            print(f"  - {feat}: {count} ({count/len(df)*100:.1f}%)")
    
    return numeric_features, categorical_features

def create_domain_features(df):
    """
    基于领域知识创建新特征（房价预测特定）
    """
    print("\n" + "=" * 60)
    print("步骤 3: 领域特征工程")
    print("=" * 60)
    
    new_features = []
    df_new = df.copy()
    
    # 1. 总面积相关特征
    area_cols = ['LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 
                 'GarageArea', 'PoolArea', 'OpenPorchSF', 'EnclosedPorch', 
                 '3SsnPorch', 'ScreenPorch', 'WoodDeckSF', 'MasVnrArea']
    
    existing_area_cols = [col for col in area_cols if col in df.columns]
    if len(existing_area_cols) > 0:
        # 总居住面积
        if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
            df_new['Total_SF'] = df['GrLivArea'] + df['TotalBsmtSF']
            new_features.append('Total_SF')
        
        # 总室外面积
        outdoor_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
                       'ScreenPorch', 'WoodDeckSF', 'PoolArea']
        existing_outdoor = [col for col in outdoor_cols if col in df.columns]
        if existing_outdoor:
            df_new['Total_Outdoor_SF'] = df[existing_outdoor].sum(axis=1)
            new_features.append('Total_Outdoor_SF')
        
        # 总面积（包含车库和地下室）
        total_area_cols = ['GrLivArea', 'TotalBsmtSF', 'GarageArea']
        existing_total = [col for col in total_area_cols if col in df.columns]
        if len(existing_total) == 3:
            df_new['Total_Area_Including_Garage'] = df[existing_total].sum(axis=1)
            new_features.append('Total_Area_Including_Garage')
        
        #  lot  Frontage 比率
        if 'LotFrontage' in df.columns and 'LotArea' in df.columns:
            df_new['Lot_Frontage_Ratio'] = df['LotFrontage'] / (df['LotArea'] + 1)
            new_features.append('Lot_Frontage_Ratio')
    
    # 2. 房间和面积比率
    if 'GrLivArea' in df.columns and 'TotRmsAbvGrd' in df.columns:
        df_new['Avg_Room_Size'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)
        new_features.append('Avg_Room_Size')
    
    if 'GrLivArea' in df.columns and 'BedroomAbvGr' in df.columns:
        df_new['Avg_Bedroom_Size'] = df['GrLivArea'] / (df['BedroomAbvGr'] + 1)
        new_features.append('Avg_Bedroom_Size')
    
    # 3. 年龄和改造相关特征
    current_year = 2024
    
    if 'YearBuilt' in df.columns:
        df_new['House_Age'] = current_year - df['YearBuilt']
        new_features.append('House_Age')
        
        # 是否是新房
        df_new['Is_New_House'] = (df_new['House_Age'] <= 5).astype(int)
        new_features.append('Is_New_House')
    
    if 'YearRemodAdd' in df.columns and 'YearBuilt' in df.columns:
        df_new['Years_Since_Remod'] = current_year - df['YearRemodAdd']
        df_new['Is_Remodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
        df_new['Remod_Age'] = df['YearRemodAdd'] - df['YearBuilt']
        new_features.extend(['Years_Since_Remod', 'Is_Remodeled', 'Remod_Age'])
    
    if 'GarageYrBlt' in df.columns:
        df_new['Garage_Age'] = current_year - df['GarageYrBlt']
        df_new['Has_Garage'] = df['GarageYrBlt'].notna().astype(int)
        new_features.extend(['Garage_Age', 'Has_Garage'])
    
    # 4. 浴室相关特征
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    existing_baths = [col for col in bath_cols if col in df.columns]
    if existing_baths:
        # 总浴室数（半浴算作0.5）
        full_bath = df.get('FullBath', 0) + df.get('BsmtFullBath', 0)
        half_bath = df.get('HalfBath', 0) + df.get('BsmtHalfBath', 0)
        df_new['Total_Bathrooms'] = full_bath + 0.5 * half_bath
        new_features.append('Total_Bathrooms')
    
    # 5. 质量和条件组合
    quality_cols = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 
                   'BsmtQual', 'BsmtCond', 'KitchenQual', 'GarageQual', 'FireplaceQu']
    
    # 将质量标签转换为数值
    qual_map = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    
    for col in ['ExterQual', 'ExterCond', 'KitchenQual', 'BsmtQual', 'GarageQual']:
        if col in df.columns:
            df_new[f'{col}_Num'] = df[col].map(qual_map).fillna(0)
            new_features.append(f'{col}_Num')
    
    if 'OverallQual' in df.columns and 'OverallCond' in df.columns:
        df_new['Qual_Cond_Ratio'] = df['OverallQual'] / (df['OverallCond'] + 1)
        df_new['Qual_Cond_Diff'] = df['OverallQual'] - df['OverallCond']
        df_new['Qual_Cond_Product'] = df['OverallQual'] * df['OverallCond']
        new_features.extend(['Qual_Cond_Ratio', 'Qual_Cond_Diff', 'Qual_Cond_Product'])
    
    # 6. 车库相关特征
    if 'GarageCars' in df.columns and 'GarageArea' in df.columns:
        df_new['Garage_Area_Per_Car'] = df['GarageArea'] / (df['GarageCars'] + 1)
        new_features.append('Garage_Area_Per_Car')
    
    # 7. 壁炉相关
    if 'Fireplaces' in df.columns:
        df_new['Has_Fireplace'] = (df['Fireplaces'] > 0).astype(int)
        df_new['Multiple_Fireplaces'] = (df['Fireplaces'] > 1).astype(int)
        new_features.extend(['Has_Fireplace', 'Multiple_Fireplaces'])
    
    # 8. 泳池相关
    if 'PoolArea' in df.columns:
        df_new['Has_Pool'] = (df['PoolArea'] > 0).astype(int)
        new_features.append('Has_Pool')
    
    # 9. 厨房占比
    if 'KitchenAbvGr' in df.columns and 'TotRmsAbvGrd' in df.columns:
        df_new['Kitchen_Ratio'] = df['KitchenAbvGr'] / (df['TotRmsAbvGrd'] + 1)
        new_features.append('Kitchen_Ratio')
    
    print(f"创建了 {len(new_features)} 个新特征:")
    for feat in new_features:
        print(f"  - {feat}")
    
    return df_new, new_features

def handle_missing_values(df, numeric_features, categorical_features):
    """
    处理缺失值
    """
    print("\n" + "=" * 60)
    print("步骤 4: 缺失值处理")
    print("=" * 60)
    
    df_clean = df.copy()
    
    # 数值型特征：使用中位数填充
    numeric_imputer = SimpleImputer(strategy='median')
    df_clean[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])
    
    # 分类型特征：使用众数填充或创建"Missing"类别
    for col in categorical_features:
        if df_clean[col].isnull().sum() > 0:
            # 如果缺失值较多，创建"Missing"类别
            if df_clean[col].isnull().sum() / len(df_clean) > 0.5:
                df_clean[col] = df_clean[col].fillna('Missing')
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    print("缺失值处理完成")
    return df_clean

def encode_categorical_features(df, categorical_features, y=None):
    """
    对分类特征进行编码
    """
    print("\n" + "=" * 60)
    print("步骤 5: 分类特征编码")
    print("=" * 60)
    
    df_encoded = df.copy()
    high_cardinality_features = []
    low_cardinality_features = []
    
    for col in categorical_features:
        unique_count = df[col].nunique()
        if unique_count > 10:
            high_cardinality_features.append(col)
        else:
            low_cardinality_features.append(col)
    
    print(f"高基数特征 ({len(high_cardinality_features)}): {high_cardinality_features}")
    print(f"低基数特征 ({len(low_cardinality_features)}): {low_cardinality_features}")
    
    # 低基数特征：One-Hot 编码
    if low_cardinality_features:
        df_encoded = pd.get_dummies(df_encoded, columns=low_cardinality_features, drop_first=True)
        print(f"One-Hot 编码完成，新增 {len(df_encoded.columns) - len(df.columns)} 列")
    
    # 高基数特征：目标编码或 Label 编码
    for col in high_cardinality_features:
        if y is not None:
            # 目标编码：使用目标变量的均值
            target_mean = y.groupby(df[col]).mean()
            df_encoded[f'{col}_TargetEnc'] = df[col].map(target_mean)
        else:
            # Label 编码
            le = LabelEncoder()
            df_encoded[f'{col}_LabelEnc'] = le.fit_transform(df[col].astype(str))
    
    return df_encoded

def transform_numeric_features(df, numeric_features):
    """
    数值特征转换（对数变换、归一化等）
    """
    print("\n" + "=" * 60)
    print("步骤 6: 数值特征转换")
    print("=" * 60)
    
    df_transformed = df.copy()
    skewed_features = []
    
    # 识别高度偏斜的特征（|skew| > 0.75）
    for col in numeric_features:
        if col in df.columns:
            skewness = df[col].skew()
            if abs(skewness) > 0.75 and df[col].min() >= 0:
                skewed_features.append((col, skewness))
    
    print(f"发现 {len(skewed_features)} 个偏斜特征，将进行对数变换")
    
    # 对偏斜特征进行对数变换
    for col, skew in skewed_features:
        # 添加常数避免 log(0)
        df_transformed[f'{col}_Log'] = np.log1p(df[col])
    
    print(f"对数变换完成，新增 {len(skewed_features)} 个对数特征")
    
    return df_transformed

def create_polynomial_features(df, important_features, degree=2):
    """
    创建多项式特征（针对重要特征）
    """
    print("\n" + "=" * 60)
    print("步骤 7: 多项式特征")
    print("=" * 60)
    
    poly_features = []
    df_poly = df.copy()
    
    # 选择最重要的几个数值特征进行多项式扩展
    selected = [f for f in important_features if f in df.columns][:5]  # 限制数量避免维度爆炸
    
    for i, feat1 in enumerate(selected):
        # 平方项
        df_poly[f'{feat1}_Squared'] = df[feat1] ** 2
        poly_features.append(f'{feat1}_Squared')
        
        # 交叉项
        for feat2 in selected[i+1:]:
            df_poly[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            poly_features.append(f'{feat1}_x_{feat2}')
    
    print(f"创建了 {len(poly_features)} 个多项式特征")
    return df_poly, poly_features

def scale_features(df, exclude_cols=None):
    """
    标准化数值特征
    """
    print("\n" + "=" * 60)
    print("步骤 8: 特征标准化")
    print("=" * 60)
    
    if exclude_cols is None:
        exclude_cols = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    print(f"标准化了 {len(cols