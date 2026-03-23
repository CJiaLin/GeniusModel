# 🏠 房价预测特征工程方案

## 1. 数据概况

| 指标 | 数值 |
|------|------|
| 样本数量 | 1,460 |
| 特征数量 | 80 (不含目标列) |
| 目标列 | SalePrice |
| 任务类型 | 回归 |
| 数值特征 | 38列 |
| 类别特征 | 43列 |

---

## 2. 现有特征分析

### 2.1 目标变量分析
- **SalePrice**: 房价范围 $34,900 - $755,000，均值 $180,921
- 分布呈右偏，需要进行对数变换

### 2.2 高缺失率特征（>80%）
| 特征 | 缺失率 | 说明 |
|------|--------|------|
| PoolQC | 99.5% | 游泳池质量（NA=无游泳池） |
| MiscFeature | 96.3% | 其他杂项特征 |
| Alley | 93.8% | 小巷通道类型（NA=无小巷） |
| Fence | 80.8% | 围栏质量（NA=无围栏） |

### 2.3 中等缺失率特征（5%-50%）
| 特征 | 缺失率 | 说明 |
|------|--------|------|
| FireplaceQu | 47.3% | 壁炉质量（NA=无壁炉） |
| LotFrontage | 17.7% | 临街宽度 |
| GarageYrBlt | 5.5% | 车库建造年份 |
| Garage系列 | 5.5% | 车库相关特征 |

### 2.4 关键数值特征
- **面积类**: GrLivArea, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GarageArea, LotArea
- **质量类**: OverallQual, OverallCond
- **时间类**: YearBuilt, YearRemodAdd, GarageYrBlt, YrSold, MoSold
- **房间类**: TotRmsAbvGrd, BedroomAbvGr, FullBath, HalfBath

### 2.5 关键类别特征
- **位置类**: Neighborhood (25个区域), MSZoning, Condition1, Condition2
- **类型类**: BldgType, HouseStyle, RoofStyle, Foundation
- **质量等级类**: ExterQual, ExterCond, BsmtQual, KitchenQual, GarageQual, FireplaceQu

---

## 3. 特征工程策略

### 3.1 缺失值处理策略

```python
# 策略1: NA表示"无此特征"的列
na_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
               'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
               'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
               'MasVnrType']
# 填充为'None'

# 策略2: 数值型缺失值
# LotFrontage: 按Neighborhood分组中位数填充
# GarageYrBlt: 填充为YearBuilt（假设车库与房屋同时建造）
# MasVnrArea: 填充为0
# BsmtFinSF系列: 填充为0

# 策略3: 单一缺失值
# Electrical: 填充众数
```

### 3.2 特征变换策略

| 策略 | 适用特征 | 方法 |
|------|----------|------|
| 对数变换 | SalePrice, LotArea, GrLivArea | `np.log1p` |
| 标签编码 | 有序类别特征 (ExterQual, BsmtQual等) | 自定义映射 |
| One-Hot编码 | 名义类别特征 (Neighborhood, MSZoning等) | `pd.get_dummies` |
| 归一化 | 所有数值特征 | MinMaxScaler或StandardScaler |

---

## 4. 要生成的新特征列表

### 4.1 面积组合特征 (9个)

| 新特征名 | 计算公式 | 业务含义 |
|----------|----------|----------|
| TotalSF | GrLivArea + TotalBsmtSF | 房屋总使用面积 |
| TotalPorchSF | OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF | 总门廊/甲板面积 |
| Has2ndFloor | (2ndFlrSF > 0).astype(int) | 是否有二楼 |
| HasBasement | (TotalBsmtSF > 0).astype(int) | 是否有地下室 |
| HasGarage | (GarageArea > 0).astype(int) | 是否有车库 |
| HasPool | (PoolArea > 0).astype(int) | 是否有游泳池 |
| HasFireplace | (Fireplaces > 0).astype(int) | 是否有壁炉 |
| LowQualSF_Ratio | LowQualFinSF / GrLivArea | 低质量面积占比 |
| OutdoorSF | WoodDeckSF + OpenPorchSF + PoolArea | 户外设施总面积 |

### 4.2 时间特征 (6个)

| 新特征名 | 计算公式 | 业务含义 |
|----------|----------|----------|
| HouseAge | YrSold - YearBuilt | 房龄 |
| RemodelAge | YrSold - YearRemodAdd | 翻新后年数 |
| IsNewHouse | (YrSold == YearBuilt).astype(int) | 是否新房 |
| HasRemodeled | (YearRemodAdd != YearBuilt).astype(int) | 是否翻新过 |
| GarageAge | YrSold - GarageYrBlt | 车库年龄 |
| SoldSeason | 基于MoSold的季度分类 | 销售季节 |

### 4.3 质量评分特征 (4个)

| 新特征名 | 计算公式 | 业务含义 |
|----------|----------|----------|
| OverallScore | OverallQual + OverallCond | 综合质量得分 |
| ExterScore | ExterQual编码 + ExterCond编码 | 外部质量得分 |
| BsmtScore | BsmtQual编码 + BsmtCond编码 + BsmtExposure编码 | 地下室质量得分 |
| KitchenScore | KitchenQual编码 + KitchenAbvGr | 厨房质量得分 |

### 4.4 房间密度/比例特征 (5个)

| 新特征名 | 计算公式 | 业务含义 |
|----------|----------|----------|
| AvgRoomSize | GrLivArea / TotRmsAbvGrd | 平均房间大小 |
| BedroomRatio | BedroomAbvGr / TotRmsAbvGrd | 卧室占比 |
| BathRatio | (FullBath + 0.5 * HalfBath) / TotRmsAbvGrd | 浴室占比 |
| TotalBath | FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath | 总浴室数 |
| RoomsPerFloor | TotRmsAbvGrd / (Has2ndFloor + 1) | 每层房间数 |

### 4.5 价值效率特征 (3个)

| 新特征名 | 计算公式 | 业务含义 |
|----------|----------|----------|
| PricePerSF | SalePrice / TotalSF | 每平方英尺价格（仅训练集） |
| GarageCarsDensity | GarageCars / (GarageArea + 1) | 车库停车密度 |
| LotFrontageRatio | LotFrontage / LotArea | 临街宽度占比 |

### 4.6 交互特征 (6个)

| 新特征名 | 计算公式 | 业务含义 |
|----------|----------|----------|
| QualArea | OverallQual * GrLivArea | 质量×面积 |
| QualBsmtArea | OverallQual * TotalBsmtSF | 质量×地下室面积 |
| QualGarageArea | OverallQual * GarageArea | 质量×车库面积 |
| YearBuiltQual | YearBuilt * OverallQual | 建造年份×质量 |
| NeighborhoodQual | Neighborhood + OverallQual | 区域质量组合 |
| MSZoningQual | MSZoning + OverallQual | 区域分类×质量 |

---

## 5. 特征工程代码

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def feature_engineering(train_path, test_path=None):
    """
    特征工程主函数
    """
    # 加载数据
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path else None
    
    # 合并处理（保持训练测试一致性）
    if test is not None:
        test['SalePrice'] = 0
        all_data = pd.concat([train, test], ignore_index=True)
    else:
        all_data = train.copy()
    
    # ==================== 缺失值处理 ====================
    
    # NA表示"无此特征"的类别特征
    na_none_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                        'BsmtFinType2', 'MasVnrType']
    for col in na_none_features:
        all_data[col] = all_data[col].fillna('None')
    
    # 数值型缺失值
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))
    all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(all_data['YearBuilt'])
    all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
    
    # 地下室面积缺失
    bsmt_num_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                     'BsmtFullBath', 'BsmtHalfBath']
    for col in bsmt_num_cols:
        all_data[col] = all_data[col].fillna(0)
    
    # 其他缺失值
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    
    # ==================== 特征变换 ====================
    
    # 对数变换（目标变量）
    all_data['SalePrice'] = np.log1p(all_data['SalePrice'])
    
    # 右偏数值特征对数变换
    skewed_features = ['LotArea', 'GrLivArea', '1stFlrSF', 'TotalBsmtSF']
    for col in skewed_features:
        all_data[col] = np.log1p(all_data[col])
    
    # ==================== 生成新特征 ====================
    
    # 面积组合特征
    all_data['TotalSF'] = all_data['GrLivArea'] + all_data['TotalBsmtSF']
    all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] + all_data['EnclosedPorch'] +
                                all_data['3SsnPorch'] + all_data['ScreenPorch'] +
                                all_data['WoodDeckSF'])
    all_data['Has2ndFloor'] = (all_data['2ndFlrSF'] > 0).astype(int)
    all_data['HasBasement'] = (all_data['TotalBsmtSF'] > 0).astype(int)
    all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
    all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
    all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)
    all_data['LowQualSF_Ratio'] = all_data['LowQualFinSF'] / (all_data['GrLivArea'] + 1)
    all_data['OutdoorSF'] = all_data['WoodDeckSF'] + all_data['OpenPorchSF'] + all_data['PoolArea']
    
    # 时间特征
    all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
    all_data['RemodelAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
    all_data['IsNewHouse'] = (all_data['YrSold'] == all_data['YearBuilt']).astype(int)
    all_data['HasRemodeled'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)
    all_data['GarageAge'] = all_data['YrSold'] - all_data['GarageYrBlt']
    all_data['GarageAge'] = all_data['GarageAge'].clip(lower=0)  # 处理异常值
    
    # 销售季节
    all_data['SoldSeason'] = all_data['MoSold'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # 质量评分特征（先编码有序类别）
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']:
        all_data[col + '_Enc'] = all_data[col].map(quality_map).fillna(0)
    
    exposure_map = {'None': 0, 'No': 1, 'Mn