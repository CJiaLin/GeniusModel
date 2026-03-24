import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

def feature_engineering():
    """
    基于房屋价格预测数据的完整特征工程流程
    包含面积聚合、时间特征、质量交互、目标编码等42+个新特征
    """
    
    # 1. 数据加载
    input_path = "/Users/cjialin/code/AutoMLByLLM/assets/train_csv_v3_1774344333/data/cleaned_data.csv"
    output_path = "/Users/cjialin/code/AutoMLByLLM/assets/train_csv_v3_1774344333/data/features_data.csv"
    
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"Original data shape: {df.shape}")
    
    # 保存原始列名（除目标变量外）
    original_features = [col for col in df.columns if col != 'SalePrice']
    new_features = []
    
    # 2. 面积聚合特征 (8个)
    print("Creating area aggregation features...")
    
    # TotalSF: 总使用面积（地上居住面积+地下室面积）
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
    new_features.append('TotalSF')
    
    # TotalPorchSF: 户外休闲空间总面积
    porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'ScreenPorch']
    df['TotalPorchSF'] = df[porch_cols].sum(axis=1)
    new_features.append('TotalPorchSF')
    
    # Has2ndFloor: 是否双层住宅
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    new_features.append('Has2ndFloor')
    
    # HasBasement: 是否有地下室
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    new_features.append('HasBasement')
    
    # HasGarage: 是否有车库
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    new_features.append('HasGarage')
    
    # HasFireplace: 是否有壁炉
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    new_features.append('HasFireplace')
    
    # HasPool: 是否有泳池
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    new_features.append('HasPool')
    
    # HasPorch: 是否有门廊
    df['HasPorch'] = (df['TotalPorchSF'] > 0).astype(int)
    new_features.append('HasPorch')
    
    # 3. 面积比例特征 (4个)
    print("Creating area ratio features...")
    
    # BasementRatio: 地下室占比（避免除零）
    df['BasementRatio'] = np.where(df['TotalSF'] > 0, df['TotalBsmtSF'] / df['TotalSF'], 0)
    new_features.append('BasementRatio')
    
    # 2ndFloorRatio: 二层面积占比
    df['2ndFloorRatio'] = np.where(df['GrLivArea'] > 0, df['2ndFlrSF'] / df['GrLivArea'], 0)
    new_features.append('2ndFloorRatio')
    
    # GarageRatio: 车库占地比
    df['GarageRatio'] = np.where(df['LotArea'] > 0, df['GarageArea'] / df['LotArea'], 0)
    new_features.append('GarageRatio')
    
    # LivingAreaRatio: 建筑密度（居住面积占地块比例）
    df['LivingAreaRatio'] = np.where(df['LotArea'] > 0, df['GrLivArea'] / df['LotArea'], 0)
    new_features.append('LivingAreaRatio')
    
    # 4. 时间衍生特征 (5个)
    print("Creating temporal features...")
    
    # HouseAge: 房龄（销售年份-建造年份）
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    new_features.append('HouseAge')
    
    # RemodAge: 翻新后年数
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    new_features.append('RemodAge')
    
    # IsNew: 是否新房
    df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)
    new_features.append('IsNew')
    
    # HasRemod: 是否翻新过
    df['HasRemod'] = (df['YearRemodAdd'] > df['YearBuilt']).astype(int)
    new_features.append('HasRemod')
    
    # GarageAge: 车库年龄（处理缺失值）
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    # 对于没有车库的样本，车库年龄设为-1或特定标记
    df.loc[df['GarageYrBlt'].isna(), 'GarageAge'] = -1
    new_features.append('GarageAge')
    
    # 5. 有序类别数值映射 (6个类别)
    print("Encoding ordinal categorical features...")
    
    # 定义质量等级映射字典
    quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    quality_map_na = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, np.nan: 0}
    
    # ExterQual, ExterCond: 外部质量和条件
    df['ExterQual_num'] = df['ExterQual'].map(quality_map)
    df['ExterCond_num'] = df['ExterCond'].map(quality_map)
    new_features.extend(['ExterQual_num', 'ExterCond_num'])
    
    # BsmtQual, BsmtCond: 地下室质量和条件（含NA）
    df['BsmtQual_num'] = df['BsmtQual'].map(quality_map_na)
    df['BsmtCond_num'] = df['BsmtCond'].map(quality_map_na)
    new_features.extend(['BsmtQual_num', 'BsmtCond_num'])
    
    # KitchenQual: 厨房质量
    df['KitchenQual_num'] = df['KitchenQual'].map(quality_map)
    new_features.append('KitchenQual_num')
    
    # FireplaceQu: 壁炉质量（含NA）
    df['FireplaceQu_num'] = df['FireplaceQu'].map(quality_map_na)
    new_features.append('FireplaceQu_num')
    
    # GarageQual, GarageCond: 车库质量和条件（含NA）
    df['GarageQual_num'] = df['GarageQual'].map(quality_map_na)
    df['GarageCond_num'] = df['GarageCond'].map(quality_map_na)
    new_features.extend(['GarageQual_num', 'GarageCond_num'])
    
    # 6. 质量交互特征 (6个)
    print("Creating quality interaction features...")
    
    # QualSF: 质量加权居住面积
    df['QualSF'] = df['OverallQual'] * df['GrLivArea']
    new_features.append('QualSF')
    
    # QualTotalSF: 质量加权总面积
    df['QualTotalSF'] = df['OverallQual'] * df['TotalSF']
    new_features.append('QualTotalSF')
    
    # QualCond: 质量×状态综合评分
    df['QualCond'] = df['OverallQual'] * df['OverallCond']
    new_features.append('QualCond')
    
    # ExterScore: 外部状态综合（质量×条件）
    df['ExterScore'] = df['ExterQual_num'] * df['ExterCond_num']
    new_features.append('ExterScore')
    
    # BsmtScore: 地下室状态综合（质量×条件）
    df['BsmtScore'] = df['BsmtQual_num'] * df['BsmtCond_num']
    new_features.append('BsmtScore')
    
    # KitchenScore: 厨房质量评分（质量×数量，这里KitchenAbvGr通常为1）
    df['KitchenScore'] = df['KitchenQual_num'] * df['KitchenAbvGr']
    new_features.append('KitchenScore')
    
    # 7. 房间配置特征 (5个)
    print("Creating room configuration features...")
    
    # TotalBath: 总浴室当量（全浴室+0.5半浴室，包括地下室）
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                       df['BsmtFullBath'].fillna(0) + 0.5 * df['BsmtHalfBath'].fillna(0))
    new_features.append('TotalBath')
    
    # BedroomRatio: 卧室占比
    df['BedroomRatio'] = np.where(df['TotRmsAbvGrd'] > 0, 
                                  df['BedroomAbvGr'] / df['TotRmsAbvGrd'], 0)
    new_features.append('BedroomRatio')
    
    # RoomDensity: 房间密度（每平方英尺房间数）
    df['RoomDensity'] = np.where(df['GrLivArea'] > 0, 
                                 df['TotRmsAbvGrd'] / df['GrLivArea'], 0)
    new_features.append('RoomDensity')
    
    # BathBedroomRatio: 浴卧比（避免除零）
    df['BathBedroomRatio'] = df['TotalBath'] / (df['BedroomAbvGr'] + 1)
    new_features.append('BathBedroomRatio')
    
    # FamilySize: 假设家庭规模（卧室数+2）
    df['FamilySize'] = df['BedroomAbvGr'] + 2
    new_features.append('FamilySize')
    
    # 8. 多项式与对数特征 (6个)
    print("Creating polynomial and log transformation features...")
    
    # GrLivAreaLog: 对数变换消除右偏
    df['GrLivAreaLog'] = np.log1p(df['GrLivArea'])
    new_features.append('GrLivAreaLog')
    
    # LotAreaLog: 地块面积对数变换
    df['LotAreaLog'] = np.log1p(df['LotArea'])
    new_features.append('LotAreaLog')
    
    # GrLivAreaSq: 居住面积平方项
    df['GrLivAreaSq'] = df['GrLivArea'] ** 2
    new_features.append('GrLivAreaSq')
    
    # OverallQualSq: 质量评分平方项（捕捉边际效应递减）
    df['OverallQualSq'] = df['OverallQual'] ** 2
    new_features.append('OverallQualSq')
    
    # HouseAgeSq: 房龄平方项（折旧非线性）
    df['HouseAgeSq'] = df['HouseAge'] ** 2
    new_features.append('HouseAgeSq')
    
    # QualAreaInt: 质量-面积对数交互
    df['QualAreaInt'] = df['OverallQual'] * df['GrLivAreaLog']
    new_features.append('QualAreaInt')
    
    # 9. 目标编码特征 (2个) - 使用5折交叉验证避免数据泄露
    print("Creating target encoding features with CV...")
    
    def target_encode_cv(df, col, target, n_splits=5):
        """
        使用交叉验证进行目标编码，避免数据泄露
        """
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        encoded = np.zeros(len(df))
        
        for train_idx, val_idx in kfold.split(df):
            # 在训练折上计算均值
            target_mean = df.iloc[train_idx].groupby(col)[target].mean()
            # 映射到验证折
            encoded[val_idx] = df.iloc[val_idx][col].map(target_mean)
        
        # 填充缺失值（使用全局均值）
        global_mean = df[target].mean()
        encoded[np.isnan(encoded)] = global_mean
        
        return encoded
    
    # NeighborhoodPrice: 区域价格水平（中位数更稳健，但这里用均值）
    df['NeighborhoodPrice'] = target_encode_cv(df, 'Neighborhood', 'SalePrice')
    new_features.append('NeighborhoodPrice')
    
    # MSSubClassPrice: 建筑类型价格水平
    df['MSSubClassPrice'] = target_encode_cv(df, 'MSSubClass', 'SalePrice')
    new_features.append('MSSubClassPrice')
    
    # 10. 数据保存
    print(f"\nSaving engineered data to {output_path}")
    df.to_csv(output_path, index=False)
    
    # 11. 输出特征工程报告
    print("\n" + "="*60)
    print("特征工程完成报告")
    print("="*60)
    print(f"原始特征数量: {len(original_features)}")
    print(f"新生成特征数量: {len(new_features)}")
    print(f"总特征数量: {df.shape[1] - 1} (不含目标变量)")
    print(f"数据形状: {df.shape}")
    print("-"*60)
    print("新生成的特征列表:")
    for i, feat in enumerate(new_features, 1):
        print(f"{i:2d}. {feat}")
    print("="*60)
    
    return new_features, df

if __name__ == "__main__":
    new_features_list, engineered_df = feature_engineering()
    
    # 输出结果摘要
    print(f"\n特征工程数据已保存至指定路径")
    print(f"新增特征共 {len(new_features_list)} 个")