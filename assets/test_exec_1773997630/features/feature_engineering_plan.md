# 🏠 房价预测特征工程方案

## 1. 现有特征分析

### 1.1 数据概况
| 指标 | 数值 |
|------|------|
| 样本数量 | 1,460 |
| 特征总数 | 87 |
| 数值特征 | 约52个 |
| 类别特征 | 约35个 |
| 目标变量 | SalePrice |

### 1.2 特征分组分析

| 特征类别 | 包含特征 | 分析 |
|---------|---------|------|
| **面积类** | `LotArea`, `TotalSF`, `GrLivArea`, `1stFlrSF`, `2ndFlrSF`, `TotalBsmtSF`, `GarageArea` | 核心预测因子，已存在`TotalSF`聚合特征 |
| **质量类** | `OverallQual`, `ExterQual`, `BsmtQual`, `KitchenQual`, `GarageQual` | 序数编码已完成，需构建质量综合评分 |
| **时间类** | `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`, `YrSold` | 已生成`HouseAge`, `RemodAge` |
| **房间类** | `BedroomAbvGr`, `KitchenAbvGr`, `TotRmsAbvGrd`, `TotalBath` | 已存在`TotalBath`综合特征 |
| **外部设施** | `WoodDeckSF`, `OpenPorchSF`, `PoolArea`, `TotalPorchSF` | 已存在`TotalPorchSF`聚合特征 |
| **高缺失特征** | `PoolQC`(99.5%), `MiscFeature`(96.4%), `Alley`(93.8%), `Fence`(80.8%) | 需创建存在性指示特征 |

### 1.3 数据质量评估
- ✅ **优点**: 大部分核心特征无缺失，已完成部分特征工程
- ⚠️ **注意**: 4个特征缺失率>80%，需特殊处理
- ✅ **编码状态**: 质量特征已完成序数编码(1-5/1-10)

---

## 2. 特征工程策略

### 2.1 策略矩阵

| 策略类型 | 适用场景 | 优先级 |
|---------|---------|--------|
| 🎯 **特征组合** | 面积、质量评分的加权聚合 | P0 |
| 📐 **比例/密度特征** | 单位面积指标、房间密度 | P0 |
| 🔗 **交互特征** | 质量×面积、年龄×质量 | P1 |
| 📊 **多项式特征** | 核心数值特征的非线性变换 | P1 |
| 🏷️ **分类特征编码** | 高基数类别特征的目标编码 | P1 |
| 📦 **分箱特征** | 连续特征的离散化 | P2 |
| ➗ **降维特征** | 高度相关特征的PCA | P2 |

### 2.2 特征工程原则
1. **业务导向**: 基于房地产领域知识构建特征
2. **可解释性**: 优先保持特征含义清晰
3. **避免泄露**: 不使用未来信息进行特征构建
4. **维度控制**: 防止特征爆炸，保持合理维度

---

## 3. 要生成的新特征列表

### 3.1 面积相关特征 (6个)

| 特征名 | 计算公式/说明 | 预期作用 |
|-------|-------------|---------|
| `LivingAreaRatio` | `GrLivArea / LotArea` | 土地利用效率 |
| `BsmtRatio` | `TotalBsmtSF / GrLivArea` | 地下室相对大小 |
| `GarageRatio` | `GarageArea / LotArea` | 车库占地比例 |
| `FloorAreaRatio` | `(1stFlrSF + 2ndFlrSF) / LotArea` | 容积率近似 |
| `AvgRoomSize` | `GrLivArea / TotRmsAbvGrd` | 平均房间大小 |
| `Has2ndFloor` | `1 if 2ndFlrSF > 0 else 0` | 是否有二楼（二进制） |

### 3.2 质量综合评分 (4个)

| 特征名 | 计算公式 | 预期作用 |
|-------|---------|---------|
| `WeightedQual` | `OverallQual * 0.4 + ExterQual * 0.2 + KitchenQual * 0.2 + BsmtQual * 0.1 + GarageQual * 0.1` | 加权质量评分 |
| `QualCondDiff` | `OverallQual - OverallCond` | 质量与状况差异 |
| `ExtQualCond` | `ExterQual * ExterCond` | 外部质量状况组合 |
| `BsmtScore` | `BsmtQual * BsmtCond` | 地下室综合评分 |

### 3.3 时间与年龄特征 (5个)

| 特征名 | 计算公式 | 预期作用 |
|-------|---------|---------|
| `GarageAge` | `YrSold - GarageYrBlt` | 车库年龄 |
| `IsRecentlyRemod` | `1 if RemodAge <= 5 else 0` | 近期是否改造 |
| `HouseAgeDecade` | `HouseAge // 10` | 房屋年龄分箱（十年） |
| `YrBuiltRemodDiff` | `YearRemodAdd - YearBuilt` | 建造到改造间隔 |
| `SeasonSold` | `MoSold`映射为季节(1-4) | 销售季节 |

### 3.4 房间与设施特征 (5个)

| 特征名 | 计算公式 | 预期作用 |
|-------|---------|---------|
| `BedroomRatio` | `BedroomAbvGr / TotRmsAbvGrd` | 卧室占比 |
| `BathPerBedroom` | `TotalBath / (BedroomAbvGr + 1)` | 卧室卫浴比 |
| `HasFireplace` | `1 if Fireplaces > 0 else 0` | 是否有壁炉 |
| `HasPool` | `1 if PoolArea > 0 else 0` | 是否有泳池 |
| `Has2ndBath` | `1 if HalfBath > 0 else 0` | 是否有半卫 |

### 3.5 高缺失特征存在性指示 (4个)

| 特征名 | 说明 | 预期作用 |
|-------|------|---------|
| `HasAlley` | `1 if Alley not null else 0` | 是否有小巷通道 |
| `HasPoolFacility` | `1 if PoolQC not null else 0` | 是否有泳池设施 |
| `HasFence` | `1 if Fence not null else 0` | 是否有围栏 |
| `HasMiscFeature` | `1 if MiscFeature not null else 0` | 是否有其他设施 |

### 3.6 交互特征 (6个)

| 特征名 | 计算公式 | 预期作用 |
|-------|---------|---------|
| `QualXArea` | `OverallQual * GrLivArea` | 质量与面积交互 |
| `AgeXQual` | `HouseAge * OverallQual` | 年龄与质量交互 |
| `QualXAgeInv` | `OverallQual / (HouseAge + 1)` | 质量年龄比 |
| `TotalSF_X_Qual` | `TotalSF * OverallQual` | 总面积与质量 |
| `NeighborhoodPriceLevel` | 按`Neighborhood`分组的`SalePrice`中位数（训练集） | 社区价格等级 |
| `MSZoningPriceLevel` | 按`MSZoning`分组的`SalePrice`中位数（训练集） | 分区价格等级 |

### 3.7 多项式/非线性特征 (4个)

| 特征名 | 计算公式 | 预期作用 |
|-------|---------|---------|
| `GrLivArea_Sq` | `GrLivArea ** 2` | 居住面积平方 |
| `GrLivArea_Log` | `log1p(GrLivArea)` | 居住面积对数 |
| `LotArea_Log` | `log1p(LotArea)` | 地块面积对数 |
| `SalePrice_Log` | `log1p(SalePrice)` | 目标变量对数（用于偏态修正）|

### 3.8 分类特征聚合 (2个)

| 特征名 | 计算公式 | 预期作用 |
|-------|---------|---------|
| `BldgType_HouseStyle` | `BldgType + '_' + HouseStyle` | 建筑类型与风格组合 |
| `OverallQual_Cat` | `OverallQual`分箱为Low/Med/High | 质量等级分类 |

---

## 4. 特征工程实施计划

### 4.1 实施阶段

```python
阶段1: 基础特征生成（优先级P0）
├── 面积比例特征: LivingAreaRatio, BsmtRatio, GarageRatio
├── 质量综合评分: WeightedQual, QualCondDiff
└── 房间设施特征: BedroomRatio, BathPerBedroom, HasFireplace, HasPool

阶段2: 高级特征生成（优先级P1）
├── 时间特征扩展: GarageAge, IsRecentlyRemod, SeasonSold
├── 存在性指示特征: HasAlley, HasPoolFacility, HasFence, HasMiscFeature
├── 交互特征: QualXArea, AgeXQual, TotalSF_X_Qual
└── 目标编码: NeighborhoodPriceLevel, MSZoningPriceLevel

阶段3: 变换与优化（优先级P2）
├── 多项式特征: GrLivArea_Sq, GrLivArea_Log
├── 分箱特征: HouseAgeDecade, OverallQual_Cat
└── 组合分类: BldgType_HouseStyle
```

### 4.2 特征工程代码框架

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_features(df, is_train=True, price_mapping=None):
    """
    特征工程主函数
    """
    df = df.copy()
    
    # ========== 阶段1: 面积比例特征 ==========
    df['LivingAreaRatio'] = df['GrLivArea'] / df['LotArea']
    df['BsmtRatio'] = df['TotalBsmtSF'] / (df['GrLivArea'] + 1)
    df['GarageRatio'] = df['GarageArea'] / (df['LotArea'] + 1)
    df['FloorAreaRatio'] = (df['1stFlrSF'] + df['2ndFlrSF']) / (df['LotArea'] + 1)
    df['AvgRoomSize'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    
    # ========== 阶段2: 质量综合评分 ==========
    df['WeightedQual'] = (
        df['OverallQual'] * 0.4 + 
        df['ExterQual'] * 0.2 + 
        df['KitchenQual'] * 0.2 + 
        df['BsmtQual'] * 0.1 + 
        df['GarageQual'] * 0.1
    )
    df['QualCondDiff'] = df['OverallQual'] - df['OverallCond']
    df['ExtQualCond'] = df['ExterQual'] * df['ExterCond']
    df['BsmtScore'] = df['BsmtQual'] * df['BsmtCond']
    
    # ========== 阶段3: 时间与年龄特征 ==========
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    df['IsRecentlyRemod'] = (df['RemodAge'] <= 5).astype(int)
    df['HouseAgeDecade'] = (df['HouseAge'] // 10).astype(int)
    df['YrBuiltRemodDiff'] = df['YearRemodAdd'] - df['YearBuilt']
    df['SeasonSold'] = df['MoSold'].map({12:1, 1:1, 2:1, 
                                         3:2, 4:2, 5:2, 
                                         6:3, 7:3, 8:3, 
                                         9:4, 10:4, 11:4})
    
    # ========== 阶段4: 房间与设施特征 ==========
    df['BedroomRatio'] = df['BedroomAbvGr'] / (df['TotRmsAbvGrd'] + 1)
    df['BathPerBedroom'] = df['TotalBath'] / (df['BedroomAbvGr'] + 1)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['Has2ndBath'] = (df['HalfBath'] > 0).astype(int)
    
    # ========== 阶段5: 存在性指示特征 ==========
    df['HasAlley'] = df['Alley'].notna().astype(int)
    df['HasPoolFacility'] = df['PoolQC'].notna().astype(int)
    df['HasFence'] = df['Fence'].notna().astype(int)
    df['HasMiscFeature'] = df['MiscFeature'].notna().astype(int)
    
    # ========== 阶段6: 交互特征 ==========
    df['QualXArea'] = df['OverallQual'] * df['GrLivArea']
    df['AgeXQual'] = df['HouseAge'] * df['OverallQual']
    df['QualXAgeInv'] = df['OverallQual'] / (df['HouseAge'] + 1)
    df['TotalSF_X_Qual'] = df['TotalSF'] * df['OverallQual']
    
    # 目标编码（防止数据泄露）
    if is_train:
        price_mapping = {
            'Neighborhood': df.groupby('Neighborhood')['SalePrice'].median(),
            'MSZoning': df.groupby('MSZoning')['SalePrice'].median()
        }
        df['NeighborhoodPriceLevel'] = df['Neighborhood'].map(price_mapping['Neighborhood'])
        df['MSZoningPriceLevel'] = df['MSZoning'].map(price_mapping['MSZoning'])
    else:
        df['NeighborhoodPriceLevel'] = df['Neighborhood'].map(price_mapping['Neighborhood'])
        df['MSZoningPriceLevel'] = df['MSZoning'].map(price_mapping['MSZoning'])
    
    # ========== 阶段7: 多项式/非线性特征 ==========
    df['GrLivArea_Sq'] = df['GrLivArea'] ** 2
    df['GrLivArea_Log'] = np.log1p(df['GrLivArea'])
    df['LotArea_Log'] = np.log1p(df['LotArea'])
    
    # ========== 阶段8: 分箱与组合特征 ==========
    df['OverallQual_Cat'] = pd.cut(df['OverallQual'], 
                                   bins=[0, 4, 6, 10], 
                                   labels=['Low', 'Medium', 'High'])
    df['BldgType_HouseStyle'] = df['BldgType'].astype(str) + '_' + df['HouseStyle'].astype(str)
    
    # 处理无穷值和缺失值
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df, price_mapping
```

---

## 5. 预期效果

### 5.1 特征维度变化

| 阶段 | 特征数量 | 新增特征 | 累积特征 |
|------|---------|---------|---------|
| 原始数据 | 87 | - | 87 |
| 阶段1: 面积比例 | 87 + 6 | 6 | 93 |
| 阶段2: 质量评分 | 93 + 4 | 4 | 97 |
| 阶段3: 时间扩展 | 97 + 5 | 5 | 102 |
| 阶段4: 房间设施 | 102 + 5 | 5 | 107 |
| 阶段5: 存在性指示 | 107 + 4 | 4 | 111 |
| 阶段6