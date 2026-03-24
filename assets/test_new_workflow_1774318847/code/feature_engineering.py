import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """加载数据"""
    try:
        df = pd.read_csv(filepath)
        print(f"数据加载成功，形状: {df.shape}")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        raise

def create_area_features(df):
    """创建面积聚合特征"""
    print("创建面积特征...")
    new_features = []
    
    # TotalSF: 房屋总使用面积
    if all(col in df.columns for col in ['GrLivArea', 'TotalBsmtSF']):
        df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
        new_features.append('TotalSF')
    
    # TotalPorchSF: 总门廊/甲板面积
    porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'ScreenPorch']
    if all(col in df.columns for col in porch_cols):
        df['TotalPorchSF'] = df[porch_cols].sum(axis=1)
        new_features.append('TotalPorchSF')
    
    # TotalArea: 建筑总面积（含车库）
    if 'TotalSF' in df.columns and 'GarageArea' in df.columns:
        df['TotalArea'] = df['TotalSF'] + df['GarageArea']
        new_features.append('TotalArea')
    
    # LotRatio: 建筑密度
    if all(col in df.columns for col in ['GrLivArea', 'LotArea']):
        df['LotRatio'] = df['GrLivArea'] / (df['LotArea'] + 1e-6)  # 防止除零
        new_features.append('LotRatio')
    
    # BsmtFinRatio: 地下室完工比例
    if all(col in df.columns for col in ['BsmtFinSF1', 'TotalBsmtSF']):
        df['BsmtFinRatio'] = df['BsmtFinSF1'] / (df['TotalBsmtSF'] + 1e-6)
        new_features.append('BsmtFinRatio')
    
    # 2ndFloorRatio: 二层面积占比
    if all(col in df.columns for col in ['2ndFlrSF', 'GrLivArea']):
        df['2ndFloorRatio'] = df['2ndFlrSF'] / (df['GrLivArea'] + 1e-6)
        new_features.append('2ndFloorRatio')
    
    # OutdoorSF: 户外设施总面积
    outdoor_cols = ['WoodDeckSF', 'OpenPorchSF', 'ScreenPorch', 'PoolArea']
    available_outdoor = [col for col in outdoor_cols if col in df.columns]
    if available_outdoor:
        df['OutdoorSF'] = df[available_outdoor].sum(axis=1)
        new_features.append('OutdoorSF')
    
    # AvgRoomSize: 平均房间大小
    if all(col in df.columns for col in ['GrLivArea', 'TotRmsAbvGrd']):
        df['AvgRoomSize'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1e-6)
        new_features.append('AvgRoomSize')
    
    return df, new_features

def create_time_features(df):
    """创建时间特征"""
    print("创建时间特征...")
    new_features = []
    
    # HouseAge: 房龄
    if all(col in df.columns for col in ['YrSold', 'YearBuilt']):
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        new_features.append('HouseAge')
    
    # RemodAge: 翻新后年数
    if all(col in df.columns for col in ['YrSold', 'YearRemodAdd']):
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        new_features.append('RemodAge')
    
    # IsNew: 是否新房
    if all(col in df.columns for col in ['YrSold', 'YearBuilt']):
        df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)
        new_features.append('IsNew')
    
    # HasRemod: 是否翻新过
    if all(col in df.columns for col in ['YearRemodAdd', 'YearBuilt']):
        df['HasRemod'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
        new_features.append('HasRemod')
    
    # SeasonSold: 销售季节
    if 'MoSold' in df.columns:
        season_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        df['SeasonSold'] = df['MoSold'].map(season_map)
        # 对季节进行独热编码
        season_dummies = pd.get_dummies(df['SeasonSold'], prefix='Season')
        df = pd.concat([df, season_dummies], axis=1)
        new_features.extend(season_dummies.columns.tolist())
    
    return df, new_features

def encode_quality_features(df):
    """质量特征编码"""
    # 定义质量映射
    quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None': 0}
    exposure_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0, 'None': 0}
    finish_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0, 'None': 0}
    
    # 对质量列进行编码
    qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                 'KitchenQual', 'GarageQual', 'GarageCond', 'FireplaceQu', 
                 'PoolQC', 'HeatingQC']
    
    for col in qual_cols:
        if col in df.columns:
            df[col + '_encoded'] = df[col].map(quality_map).fillna(0)
    
    # BsmtExposure编码
    if 'BsmtExposure' in df.columns:
        df['BsmtExposure_encoded'] = df['BsmtExposure'].map(exposure_map).fillna(0)
    
    # GarageFinish编码
    if 'GarageFinish' in df.columns:
        df['GarageFinish_encoded'] = df['GarageFinish'].map(finish_map).fillna(0)
    
    return df

def create_quality_features(df):
    """创建质量综合特征"""
    print("创建质量特征...")
    new_features = []
    
    # 先进行编码
    df = encode_quality_features(df)
    
    # QualScore: 综合质量分
    if all(col in df.columns for col in ['OverallQual', 'OverallCond']):
        df['QualScore'] = df['OverallQual'] * df['OverallCond']
        new_features.append('QualScore')
    
    # ExterScore: 外部质量评分
    if all(col in df.columns for col in ['ExterQual_encoded', 'ExterCond_encoded']):
        df['ExterScore'] = df['ExterQual_encoded'] + df['ExterCond_encoded']
        new_features.append('ExterScore')
    
    # BsmtScore: 地下室质量评分
    bsmt_cols = ['BsmtQual_encoded', 'BsmtCond_encoded', 'BsmtExposure_encoded']
    if all(col in df.columns for col in bsmt_cols):
        df['BsmtScore'] = df[bsmt_cols].sum(axis=1)
        new_features.append('BsmtScore')
    
    # KitchenScore: 厨房质量加权
    if 'KitchenQual_encoded' in df.columns:
        df['KitchenScore'] = df['KitchenQual_encoded'] * 2
        new_features.append('KitchenScore')
    
    # GarageScore: 车库综合质量
    garage_cols = ['GarageQual_encoded', 'GarageCond_encoded', 'GarageFinish_encoded']
    if all(col in df.columns for col in garage_cols):
        df['GarageScore'] = df[garage_cols].sum(axis=1)
        new_features.append('GarageScore')
    
    return df, new_features

def create_density_features(df):
    """创建功能密度特征"""
    print("创建密度特征...")
    new_features = []
    
    # BathPerRoom: 每房间浴室数
    if all(col in df.columns for col in ['FullBath', 'HalfBath', 'TotRmsAbvGrd']):
        df['BathPerRoom'] = (df['FullBath'] + 0.5 * df['HalfBath']) / (df['TotRmsAbvGrd'] + 1e-6)
        new_features.append('BathPerRoom')
    
    # RoomDensity: 房间密度
    if all(col in df.columns for col in ['TotRmsAbvGrd', 'GrLivArea']):
        df['RoomDensity'] = df['TotRmsAbvGrd'] / (df['GrLivArea'] + 1e-6) * 1000
        new_features.append('RoomDensity')
    
    # BedroomRatio: 卧室占比
    if all(col in df.columns for col in ['BedroomAbvGr', 'TotRmsAbvGrd']):
        df['BedroomRatio'] = df['BedroomAbvGr'] / (df['TotRmsAbvGrd'] + 1e-6)
        new_features.append('BedroomRatio')
    
    # GarageEfficiency: 车库停车效率
    if all(col in df.columns for col in ['GarageCars', 'GarageArea']):
        df['GarageEfficiency'] = df['GarageCars'] / (df['GarageArea'] + 1e-6)
        new_features.append('GarageEfficiency')
    
    return df, new_features

def create_interaction_features(df):
    """创建交互特征"""
    print("创建交互特征...")
    new_features = []
    
    # Qual_LivArea: 质量加权面积
    if all(col in df.columns for col in ['OverallQual', 'GrLivArea']):
        df['Qual_LivArea'] = df['OverallQual'] * df['GrLivArea']
        new_features.append('Qual_LivArea')
    
    # Qual_BsmtSF: 质量加权地下室
    if all(col in df.columns for col in ['OverallQual', 'TotalBsmtSF']):
        df['Qual_BsmtSF'] = df['OverallQual'] * df['TotalBsmtSF']
        new_features.append('Qual_BsmtSF')
    
    # Area_GarageCars: 面积与车库容量交互
    if all(col in df.columns for col in ['GrLivArea', 'GarageCars']):
        df['Area_GarageCars'] = df['GrLivArea'] * df['GarageCars']
        new_features.append('Area_GarageCars')
    
    # Age_Qual: 房龄与质量交互
    if all(col in df.columns for col in ['HouseAge', 'OverallQual']):
        df['Age_Qual'] = df['HouseAge'] * (10 - df['OverallQual'])
        new_features.append('Age_Qual')
    
    return df, new_features

def create_target_encoding(df, target_col):
    """目标编码（针对高基数类别特征）"""
    print("创建目标编码特征...")
    new_features = []
    
    # Neighborhood目标编码
    if 'Neighborhood' in df.columns and target_col in df.columns:
        neighborhood_mean = df.groupby('Neighborhood')[target_col].mean()
        df['Neighborhood_PriceLevel'] = df['Neighborhood'].map(neighborhood_mean)
        new_features.append('Neighborhood_PriceLevel')
    
    # MSZoning目标编码
    if 'MSZoning' in df.columns and target_col in df.columns:
        zoning_mean = df.groupby('MSZoning')[target_col].mean()
        df['MSZoning_PriceLevel'] = df['MSZoning'].map(zoning_mean)
        new_features.append('MSZoning_PriceLevel')
    
    return df, new_features

def create_binary_features(df):
    """创建其他派生二元特征"""
    print("创建二元特征...")
    new_features = []
    
    # HasPool: 是否有泳池
    if 'PoolArea' in df.columns:
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        new_features.append('HasPool')
    
    # Has2ndFloor: 是否有二层
    if '2ndFlrSF' in df.columns:
        df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
        new_features.append('Has2ndFloor')
    
    # HasBasement: 是否有地下室
    if 'TotalBsmtSF' in df.columns:
        df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
        new_features.append('HasBasement')
    
    # HasFireplace: 是否有壁炉
    if 'Fireplaces' in df.columns:
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
        new_features.append('HasFireplace')
    
    # HasGarage: 是否有车库
    if 'GarageArea' in df.columns:
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        new_features.append('HasGarage')
    
    return df, new_features

def handle_missing_values(df):
    """处理缺失值"""
    print("处理缺失值...")
    # 数值列用中位数填充
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # 类别列用众数填充
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'None', inplace=True)
    
    return df

def main():
    """主函数"""
    # 数据路径
    input_path = '/Users/cjialin/code/AutoMLByLLM/assets/test_new_workflow_1774318847/data/cleaned_data.csv'
    output_path = '/Users/cjialin/code/AutoMLByLLM/assets/test_new_workflow_1774318847/data/features_data.csv'
    target_col = 'SalePrice'
    
    # 加载数据
    df = load_data(input_path)
    
    # 保存原始特征列表（排除ID和目标）
    original_features = [col for col in df.columns if col not in ['Id', target_col]]
    
    # 逐步创建特征
    all_new_features = []
    
    # 1. 面积特征
    df, area_features = create_area_features(df)
    all_new_features.extend(area_features)
    
    # 2. 时间特征
    df, time_features = create_time_features(df)
    all_new_features.extend(time_features)
    
    # 3. 质量特征
    df