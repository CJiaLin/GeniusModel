import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置路径 ====================
input_path = '/Users/cjialin/code/AutoMLByLLM/assets/train_csv_v3_1774342086/data/cleaned_data.csv'
output_path = '/Users/cjialin/code/AutoMLByLLM/assets/train_csv_v3_1774342086/data/features_data.csv'

# ==================== 加载数据 ====================
print("正在加载数据...")
df = pd.read_csv(input_path)
original_cols = list(df.columns)
existing_cols = set(df.columns)
print(f"原始数据形状: {df.shape}")
print(f"原始特征数: {len(original_cols)}")

# 记录新生成的特征
new_features = []

# ==================== 1. 面积聚合特征 (FE-01) ====================
print("正在生成面积聚合特征...")

# TotalSF = GrLivArea + TotalBsmtSF (房屋总使用面积)
if 'GrLivArea' in existing_cols and 'TotalBsmtSF' in existing_cols:
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
    new_features.append('TotalSF')

# TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch (总门廊面积)
porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
available_porch_cols = [col for col in porch_cols if col in existing_cols]
if available_porch_cols:
    df['TotalPorchSF'] = df[available_porch_cols].sum(axis=1)
    new_features.append('TotalPorchSF')

# TotalOutdoorSF = WoodDeckSF + TotalPorchSF + PoolArea (总户外活动面积)
if 'WoodDeckSF' in existing_cols and 'PoolArea' in existing_cols:
    if 'TotalPorchSF' in df.columns:
        df['TotalOutdoorSF'] = df['WoodDeckSF'] + df['TotalPorchSF'] + df['PoolArea']
    else:
        df['TotalOutdoorSF'] = df['WoodDeckSF'] + df['PoolArea']
    new_features.append('TotalOutdoorSF')

# LotUtilization = (GrLivArea / LotArea) × 100 (土地利用率)
if 'GrLivArea' in existing_cols and 'LotArea' in existing_cols:
    df['LotUtilization'] = (df['GrLivArea'] / (df['LotArea'] + 1)) * 100
    new_features.append('LotUtilization')

# BasementFinRatio = BsmtFinSF1 / (TotalBsmtSF + 1) (地下室完成比例)
if 'BsmtFinSF1' in existing_cols and 'TotalBsmtSF' in existing_cols:
    df['BasementFinRatio'] = df['BsmtFinSF1'] / (df['TotalBsmtSF'] + 1)
    new_features.append('BasementFinRatio')

# ==================== 2. 房间比例特征 (FE-02) ====================
print("正在生成房间比例特征...")

# TotalBathrooms = FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath (等效总浴室数)
bath_components = []
if 'FullBath' in existing_cols:
    bath_components.append(df['FullBath'])
if 'HalfBath' in existing_cols:
    bath_components.append(0.5 * df['HalfBath'])
if 'BsmtFullBath' in existing_cols:
    bath_components.append(df['BsmtFullBath'])
if 'BsmtHalfBath' in existing_cols:
    bath_components.append(0.5 * df['BsmtHalfBath'])

if bath_components:
    df['TotalBathrooms'] = sum(bath_components)
    new_features.append('TotalBathrooms')

# BedroomToBathRatio = BedroomAbvGr / (TotalBathrooms + 0.1) (卧室浴室比)
if 'BedroomAbvGr' in existing_cols and 'TotalBathrooms' in df.columns:
    df['BedroomToBathRatio'] = df['BedroomAbvGr'] / (df['TotalBathrooms'] + 0.1)
    new_features.append('BedroomToBathRatio')

# RoomsPerSF = TotRmsAbvGrd / (GrLivArea + 1) (房间密度)
if 'TotRmsAbvGrd' in existing_cols and 'GrLivArea' in existing_cols:
    df['RoomsPerSF'] = df['TotRmsAbvGrd'] / (df['GrLivArea'] + 1)
    new_features.append('RoomsPerSF')

# KitchenToRoomRatio = KitchenAbvGr / (TotRmsAbvGrd + 1) (厨房占比)
if 'KitchenAbvGr' in existing_cols and 'TotRmsAbvGrd' in existing_cols:
    df['KitchenToRoomRatio'] = df['KitchenAbvGr'] / (df['TotRmsAbvGrd'] + 1)
    new_features.append('KitchenToRoomRatio')

# ==================== 3. 建筑年龄特征 (FE-03) ====================
print("正在生成建筑年龄特征...")

if 'YrSold' in existing_cols:
    # HouseAge = YrSold - YearBuilt (房屋年龄)
    if 'YearBuilt' in existing_cols:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['HouseAge'] = df['HouseAge'].clip(lower=0)  # 确保非负
        new_features.append('HouseAge')
        
        # IsNewHouse = 1 if HouseAge <= 1 else 0 (是否新房)
        df['IsNewHouse'] = (df['HouseAge'] <= 1).astype(int)
        new_features.append('IsNewHouse')
    
    # RemodAge = YrSold - YearRemodAdd (翻新后年数)
    if 'YearRemodAdd' in existing_cols:
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        df['RemodAge'] = df['RemodAge'].clip(lower=0)
        new_features.append('RemodAge')
        
        # HasRemod = 1 if YearRemodAdd != YearBuilt else 0 (是否翻新)
        if 'YearBuilt' in existing_cols:
            df['HasRemod'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
            new_features.append('HasRemod')
    
    # GarageAge = YrSold - GarageYrBlt (车库年龄)
    if 'GarageYrBlt' in existing_cols:
        df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
        df['GarageAge'] = df['GarageAge'].clip(lower=0)
        # 处理缺失值：如果GarageYrBlt为NaN，则GarageAge设为NaN或中位数
        new_features.append('GarageAge')

# ==================== 4. 质量交互特征 (FE-04) ====================
print("正在生成质量交互特征...")

if 'OverallQual' in existing_cols:
    # Qual_LivArea = OverallQual × GrLivArea (质量面积交互)
    if 'GrLivArea' in existing_cols:
        df['Qual_LivArea'] = df['OverallQual'] * df['GrLivArea']
        new_features.append('Qual_LivArea')
    
    # Qual_Cond = OverallQual - OverallCond (质量与状况差距)
    if 'OverallCond' in existing_cols:
        df['Qual_Cond'] = df['OverallQual'] - df['OverallCond']
        new_features.append('Qual_Cond')
        
        # Qual_Cond_Combined = (OverallQual + OverallCond) / 2 (平均质量状况)
        df['Qual_Cond_Combined'] = (df['OverallQual'] + df['OverallCond']) / 2
        new_features.append('Qual_Cond_Combined')
    
    # Qual_TotalSF = OverallQual × TotalSF (质量总面交互)
    if 'TotalSF' in df.columns:
        df['Qual_TotalSF'] = df['OverallQual'] * df['TotalSF']
        new_features.append('Qual_TotalSF')

# ==================== 5. 对数变换特征 (FE-05) ====================
print("正在生成对数变换特征...")

# LogSalePrice = log1p(SalePrice) (目标变量对数变换)
if 'SalePrice' in existing_cols:
    df['LogSalePrice'] = np.log1p(df['SalePrice'])
    new_features.append('LogSalePrice')

# 面积相关对数变换
log_transform_cols = {
    'GrLivArea': 'LogGrLivArea',
    'LotArea': 'LogLotArea',
    'TotalBsmtSF': 'LogTotalBsmtSF'
}

for orig_col, new_col in log_transform_cols.items():
    if orig_col in existing_cols:
        df[new_col] = np.log1p(df[orig_col])
        new_features.append(new_col)

# LogTotalSF (如果TotalSF已生成)
if 'TotalSF' in df.columns:
    df['LogTotalSF'] = np.log1p(df['TotalSF'])
    new_features.append('LogTotalSF')

# ==================== 6. 类别编码优化 (FE-06) ====================
print("正在生成类别编码特征...")

# Neighborhood 目标编码 (使用训练集均值避免数据泄漏)
if 'Neighborhood' in existing_cols and 'SalePrice' in existing_cols:
    neighborhood_mean = df.groupby('Neighborhood')['SalePrice'].mean()
    df['Neighborhood_MeanPrice'] = df['Neighborhood'].map(neighborhood_mean)
    new_features.append('Neighborhood_MeanPrice')

# MSSubClass 转换为类别并分箱 (分箱编码)
if 'MSSubClass' in existing_cols:
    # 将MSSubClass视为类别型
    df['MSSubClass_Category'] = df['MSSubClass'].astype(str)
    new_features.append('MSSubClass_Category')

# OverallQual 分箱 (等级合并)
if 'OverallQual' in existing_cols:
    # 将质量分为低(1-4)、中(5-6)、高(7-8)、极高(9-10)
    bins = [0, 4, 6, 8, 10]
    labels = ['Low', 'Medium', 'High', 'VeryHigh']
    df['QualityCategory'] = pd.cut(df['OverallQual'], bins=bins, labels=labels)
    new_features.append('QualityCategory')

# ==================== 7. 特殊标志特征 ====================
print("正在生成特殊标志特征...")

# HasPool = PoolArea > 0 (是否有泳池)
if 'PoolArea' in existing_cols:
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    new_features.append('HasPool')

# Has2ndFloor = 2ndFlrSF > 0 (是否有二楼)
if '2ndFlrSF' in existing_cols:
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    new_features.append('Has2ndFloor')

# HasGarage = GarageArea > 0 (是否有车库)
if 'GarageArea' in existing_cols:
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    new_features.append('HasGarage')

# HasBasement = TotalBsmtSF > 0 (是否有地下室)
if 'TotalBsmtSF' in existing_cols:
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    new_features.append('HasBasement')

# HasFireplace = Fireplaces > 0 (是否有壁炉)
if 'Fireplaces' in existing_cols:
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    new_features.append('HasFireplace')

# HasDeck = WoodDeckSF > 0 (是否有露台)
if 'WoodDeckSF' in existing_cols:
    df['HasDeck'] = (df['WoodDeckSF'] > 0).astype(int)
    new_features.append('HasDeck')

# HasFence = Fence != 'None' (是否有围栏)
if 'Fence' in existing_cols:
    df['HasFence'] = (~df['Fence'].isin(['None', 'NA', '']) & df['Fence'].notna()).astype(int)
    new_features.append('HasFence')

# HasAirCond = CentralAir == 'Y' (是否有中央空调)
if 'CentralAir' in existing_cols:
    df['HasAirCond'] = (df['CentralAir'] == 'Y').astype(int)
    new_features.append('HasAirCond')

# ==================== 8. 高级组合特征 ====================
print("正在生成高级组合特征...")

# PricePerSF = SalePrice / (GrLivArea + 1) (单位面积价格，用于验证)
if 'SalePrice' in existing_cols and 'GrLivArea' in existing_cols:
    df['PricePerSF'] = df['SalePrice'] / (df['GrLivArea'] + 1)
    new_features.append('PricePerSF')

# Qual_Utilization = OverallQual × LotUtilization (质量利用率交互)
if 'OverallQual' in existing_cols and 'LotUtilization' in df.columns:
    df['Qual_Utilization'] = df['OverallQual'] * df['LotUtilization']
    new_features.append('Qual_Utilization')

# Age_Qual_Interact = HouseAge × OverallQual (年龄质量交互)
if 'HouseAge' in df.columns and 'OverallQual' in existing_cols:
    df['Age_Qual_Interact'] = df['HouseAge'] * df['OverallQual']
    new_features.append('Age_Qual_Interact')

# LuxuryScore = HasPool + (Fireplaces > 0) + (GarageCars >= 2) + (OverallQual >= 8) (豪华度评分)
luxury_components = []
if 'PoolArea' in existing_cols:
    luxury_components.append((df['PoolArea'] > 0).astype(int))
if 'Fireplaces' in existing_cols:
    luxury_components.append((df['Fireplaces'] > 0).astype(int))
if 'GarageCars' in existing_cols:
    luxury_components.append((df['GarageCars'] >= 2).astype(int))
if 'OverallQual' in existing_cols:
    luxury_components.append((df['OverallQual'] >= 8).astype(int))

if len(luxury_components) >= 2:  # 至少需要2个组件
    df['LuxuryScore'] = sum(luxury_components)
    new_features.append('LuxuryScore')

# SF_per_Room = TotalSF / (TotRmsAbvGrd + 1) (总面积与房间数比)
if 'TotalSF' in df.columns and 'TotRmsAbvGrd' in existing_cols:
    df['SF_per_Room'] = df['TotalSF'] / (df['TotRmsAbvGrd'] + 1)
    new_features.append('SF_per_Room')

# GarageToHouseRatio = GarageArea / (GrLivArea + 1) (车库与房屋比例)
if 'GarageArea' in existing_cols and 'GrLivArea' in existing_cols:
    df['GarageToHouseRatio'] = df['GarageArea'] / (df['GrLivArea'] + 1)
    new_features.append('GarageToHouseRatio')

# ==================== 保存结果 ====================
print("\n" + "="*50)
print(f"特征工程完成统计:")
print(f"原始特征数: {len(original_cols)}")
print(f"新生成特征数: {len(new_features)}")
print(f"总特征数: {df.shape[1]}")
print("="*50)

# 保存到指定路径
print(f"\n正在保存特征工程后的数据到:")
print(f"{output_path}")
df.to_csv(output_path, index=False)
print("数据保存成功！")

# 输出新生成的特征列表
print("\n新生成的特征列表:")
for i, feat in enumerate(new_features, 1):
    print(f"{i:2d}. {feat}")

# 显示部分特征统计
print("\n部分数值特征统计预览:")
numeric_new_features = df[new_features].select_dtypes(include=[np.number]).columns[:5]
if len(numeric_new_features) > 0:
    print(df[numeric_new_features].describe())

# 返回新生成的特征列表
new_features_list = new_features
print(f"\n执行完毕，共生成 {len(new_features_list)} 个新特征。")