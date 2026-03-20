# 特征工程方案报告

## 1. 现有特征分析

### 1.1 数据概况
- **数据集**: House Prices - Advanced Regression Techniques
- **样本数量**: 1460
- **特征数量**: 81（包含目标变量 SalePrice）
- **任务类型**: 回归任务（预测房价）
- **目标变量**: SalePrice

### 1.2 特征分类

#### 数值型特征（38个）
**面积相关特征（16个）**:
- LotFrontage: 街道连接距离
- LotArea: 地块面积
- MasVnrArea: 砖石贴面面积
- BsmtFinSF1/BsmtFinSF2: 地下室完成面积
- BsmtUnfSF: 地下室未完成面积
- TotalBsmtSF: 地下室总面积
- 1stFlrSF/2ndFlrSF: 一/二层面积
- LowQualFinSF: 低质量完成面积
- GrLivArea: 地上生活面积
- GarageArea: 车库面积
- WoodDeckSF/OpenPorchSF/EnclosedPorch/3SsnPorch/ScreenPorch: 室外空间面积
- PoolArea: 泳池面积

**数量/年份特征（14个）**:
- MSSubClass: 建筑类型
- OverallQual/OverallCond: 整体质量/状况（1-10）
- YearBuilt: 建造年份
- YearRemodAdd: 翻新年份
- BsmtFullBath/BsmtHalfBath: 地下室卫生间数量
- FullBath/HalfBath: 地上卫生间数量
- BedroomAbvGr: 卧室数量
- KitchenAbvGr: 厨房数量
- TotRmsAbvGrd: 地上总房间数
- Fireplaces: 壁炉数量
- GarageYrBlt: 车库建造年份
- GarageCars: 车库容量
- MoSold/YrSold: 销售月份/年份

#### 类别型特征（43个）
**建筑属性**:
- MSZoning: 区域分类
- Street/Alley: 道路类型
- LotShape: 地块形状
- LandContour/LandSlope: 土地平整度/坡度
- Neighborhood: 社区
- Condition1/Condition2: 周边条件
- BldgType/HouseStyle: 建筑类型/房屋风格
- RoofStyle/RoofMatl: 屋顶样式/材料
- Exterior1st/Exterior2nd: 外墙材料
- MasVnrType: 砖石贴面类型
- Foundation: 基础类型
- Heating/HeatingQC: 供暖系统/质量
- CentralAir: 中央空调
- Electrical: 电力系统
- GarageType/GarageFinish: 车库类型/完成度
- PavedDrive: 车道铺设
- Fence: 围栏
- MiscFeature: 其他特性
- SaleType/SaleCondition: 销售类型/条件

**质量等级（Ordinal）**:
- ExterQual/ExterCond: 外部质量/状况
- BsmtQual/BsmtCond: 地下室高度/状况
- BsmtExposure: 地下室采光
- BsmtFinType1/BsmtFinType2: 地下室完成类型
- KitchenQual: 厨房质量
- FireplaceQu: 壁炉质量
- GarageQual/GarageCond: 车库质量/状况
- PoolQC: 泳池质量
- Functional: 房屋功能性

### 1.3 数据质量问题
- **缺失值特征**: PoolQC, MiscFeature, Alley, Fence, FireplaceQu, LotFrontage, GarageYrBlt等
- **分布偏斜**: 目标变量SalePrice右偏，部分面积特征右偏
- **年份特征**: 存在时间序列特性，需考虑房龄

---

## 2. 特征工程策略

### 2.1 缺失值处理策略

| 特征 | 缺失类型 | 处理策略 |
|------|---------|---------|
| PoolQC/Fence/MiscFeature/Alley | 高缺失（>80%） | 创建"是否有该设施"的二元特征 |
| FireplaceQu | 中等缺失（~50%） | 填充"None"，保留原始特征 |
| LotFrontage | 中等缺失（~17%） | 按Neighborhood分组中位数填充 |
| GarageYrBlt | 低缺失（~5%） | 用YearBuilt填充（无车库视为同建房年） |
| MasVnrArea/Type | 低缺失（<1%） | 填充0或"None" |

### 2.2 数值特征工程

#### A. 面积聚合特征
```
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF          # 总居住面积
TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF  # 总室外面积
HasPool = (PoolArea > 0).astype(int)                 # 是否有泳池
Has2ndFloor = (2ndFlrSF > 0).astype(int)             # 是否有二层
HasGarage = (GarageArea > 0).astype(int)             # 是否有车库
HasBasement = (TotalBsmtSF > 0).astype(int)          # 是否有地下室
HasFireplace = (Fireplaces > 0).astype(int)          # 是否有壁炉
```

#### B. 房间比例特征
```
AvgRoomSize = GrLivArea / TotRmsAbvGrd               # 平均房间面积
BathRatio = (FullBath + 0.5*HalfBath) / TotRmsAbvGrd # 卫生间房间比
BedroomRatio = BedroomAbvGr / TotRmsAbvGrd           # 卧室占比
```

#### C. 时间特征
```
HouseAge = YrSold - YearBuilt                        # 房龄
RemodAge = YrSold - YearRemodAdd                     # 翻新后年数
IsNew = (YrSold == YearBuilt).astype(int)            # 是否新房
GarageAge = YrSold - GarageYrBlt                     # 车库年龄
```

#### D. 对数变换（处理右偏分布）
```
Log_SalePrice = log(SalePrice)                       # 目标变量对数化
Log_LotArea = log(LotArea + 1)
Log_GrLivArea = log(GrLivArea)
```

### 2.3 类别特征工程

#### A. 有序编码（Ordinal Encoding）
将质量等级映射为数值：
```
quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ExterQual → ExterQual_Encoded
KitchenQual → KitchenQual_Encoded
...
```

#### B. 有序组合特征
```
OverallScore = OverallQual * OverallCond             # 整体质量得分
QualLivArea = GrLivArea * OverallQual               # 加权居住面积
```

#### C. 独热编码（One-Hot Encoding）
- 对低基数类别特征（<10类）: Neighborhood, MSSubClass, MSZoning
- 对高基数类别: 考虑目标编码（Target Encoding）或聚类

#### D. 稀有类别处理
对出现频率<1%的类别合并为"Other"

### 2.4 交互特征

#### A. 面积-质量交互
```
Qual_TotalSF = TotalSF * OverallQual
Qual_BsmtSF = TotalBsmtSF * BsmtQual_Encoded
```

#### B. 位置-面积交互
```
Neighborhood_GrLivArea = Neighborhood × GrLivArea（分组统计后合并）
```

#### C. 时间-质量交互
```
Age_Quality = HouseAge * OverallQual
```

---

## 3. 新特征生成清单

### 3.1 面积相关新特征（8个）

| 特征名 | 计算公式 | 预期作用 |
|-------|---------|---------|
| TotalSF | Bsmt + 1stFlr + 2ndFlr | 综合居住面积，强预测力 |
| TotalPorchSF | 所有室外面积之和 | 室外活动空间指标 |
| AvgRoomSize | GrLivArea / TotRms | 房间宽敞度 |
| BathToRoomRatio | 总卫生间数 / 总房间数 | 便利性指标 |
| BsmtFinRatio | BsmtFinSF1 / TotalBsmtSF | 地下室完成度 |
| LotUtilization | GrLivArea / LotArea | 土地利用率 |
| OutdoorSpaceRatio | TotalPorchSF / LotArea | 户外空间占比 |
| GarageLivRatio | GarageArea / GrLivArea | 车库相对规模 |

### 3.2 二元指示特征（6个）

| 特征名 | 条件 | 预期作用 |
|-------|------|---------|
| HasPool | PoolArea > 0 | 高端设施指示 |
| Has2ndFloor | 2ndFlrSF > 0 | 复式结构指示 |
| HasGarage | GarageArea > 0 | 停车设施指示 |
| HasBasement | TotalBsmtSF > 0 | 地下室指示 |
| HasFireplace | Fireplaces > 0 | 壁炉设施指示 |
| HasFence | Fence != 'None' | 围栏指示 |

### 3.3 时间相关新特征（5个）

| 特征名 | 计算公式 | 预期作用 |
|-------|---------|---------|
| HouseAge | YrSold - YearBuilt | 房龄，折旧因素 |
| RemodAge | YrSold - YearRemodAdd | 翻新程度 |
| IsNew | YrSold == YearBuilt | 新房溢价 |
| GarageAge | YrSold - GarageYrBlt | 车库新旧 |
| YearsSinceRemod | 同RemodAge | 时间衰减特征 |

### 3.4 质量评分特征（4个）

| 特征名 | 计算公式 | 预期作用 |
|-------|---------|---------|
| OverallScore | Qual × Cond | 综合质量评分 |
| ExterScore | ExterQual × ExterCond | 外部状况评分 |
| BsmtScore | BsmtQual × BsmtCond × BsmtExposure | 地下室综合评分 |
| KitchenScore | KitchenQual × KitchenAbvGr | 厨房质量评分 |

### 3.5 交互特征（4个）

| 特征名 | 计算公式 | 预期作用 |
|-------|---------|---------|
| Qual_LivArea | OverallQual × GrLivArea | 质量加权面积 |
| Age_Qual_Interact | HouseAge × OverallQual | 质量-折旧交互 |
| Neighborhood_PriceLevel | 社区中位数房价（目标编码） | 位置价值 |
| MSSubClass_AvgPrice | 建筑类型均价（目标编码） | 建筑类型价值 |

### 3.6 统计特征（3个）

| 特征名 | 计算方法 | 预期作用 |
|-------|---------|---------|
| Neighborhood_GrLivArea_Mean | 社区平均居住面积 | 对比基准 |
| Neighborhood_SalePrice_Mean | 社区平均房价（训练集） | 位置价值参考 |
| MSSubClass_YearBuilt_Mode | 建筑类型典型建造年代 | 年代特征 |

---

## 4. 预期效果

### 4.1 模型性能提升

| 评估指标 | 基线 | 预期提升 | 说明 |
|---------|------|---------|------|
| RMSE | ~0.15 | -15%~25% | 对数变换减少异常值影响 |
| R² Score | ~0.85 | +5%~10% | 新特征增强解释力 |
| CV Score Std | ~0.02 | -30% | 特征工程减少过拟合 |

### 4.2 关键特征重要性预测

基于房产评估理论，预期Top10重要特征：
1. **OverallQual** - 整体质量（核心）
2. **TotalSF** - 总居住面积（新特征）
3. **GrLivArea** - 地上生活面积
4. **Qual_LivArea** - 质量加权面积（新特征）
5. **GarageCars/Area** - 车库容量
6. **HouseAge** - 房龄（新特征）
7. **Neighborhood** - 社区位置
8. **TotalBsmtSF** - 地下室面积
9. **YearBuilt** - 建造年份
10. **KitchenQual** - 厨房质量

### 4.3 业务洞察增强

- **面积效率**: AvgRoomSize揭示空间利用效率
- **折旧模型**: HouseAge × OverallQual量化质量折旧
- **设施溢价**: HasPool/HasFireplace量化设施价值
- **位置价值**: Neighborhood编码量化地段溢价
- **翻新回报**: RemodAge与SalePrice关系评估翻新ROI

### 4.4 风险控制

| 风险点 | 缓解措施 |
|-------|---------|
| 过拟合（高维特征） | 使用正则化（L1/L2），交叉验证 |
| 数据泄露（目标编码） | 训练集独立计算，验证/测试集映射 |
| 多重共线性 | VIF检验，PCA降维备选 |
| 异常值敏感 | RobustScaler，分位数变换 |

### 4.5 实施优先级

```
高优先级（必须）:
├── 缺失值处理（特别是LotFrontage, GarageYrBlt）
├── 对数变换（目标变量+右偏特征）
├── 基础面积聚合（TotalSF）
├── 质量等级编码（Ordinal）
└── 时间特征（HouseAge, IsNew）

中优先级（推荐）:
├── 二元指示特征（HasPool等）
├── 房间比例特征（AvgRoomSize等）
├── 质量评分特征（OverallScore等）
└── 独热编码（低基数类别）

低优先级（优化）:
├── 复杂交互特征
├── 目标编码（高基数类别）
├── 统计聚合特征
└── 非线性变换（Box-Cox等）
```

---

## 5. 代码实现框架

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        
    def fit(self, X, y=None):
        # 计算训练集统计量（用于目标编码）
        return self
        
    def transform(self, X):
        df = X.copy()
        
        # 1. 缺失值处理
        df = self._handle_missing(df)
        
        # 2. 创建面积特征
        df = self._create_area_features(df)
        
        # 3. 创建时间特征
        df = self._create_time_features(df)
        
        # 4. 创建质量特征
        df = self._create_quality_features(df)
        
        # 5. 编码
        df = self._encode_features(df)
        
        return df
    
    def _handle_missing(self, df):
        # 实现缺失值填充逻辑
        pass
    
    def _create_area_features(self, df):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        # ... 其他特征
        return df
    
    def _create_time_features(self, df):
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)
        return df
    
    def _create_quality_features(self, df):
        df['OverallScore'] = df['OverallQual'] * df['OverallCond']
        return df
    
    def _encode_features(self, df):
        # Ordinal + One-Hot Encoding
        return df
```

---

**总结**: 本特征工程方案通过系统性地处理缺失值、聚合面积信息、提取时间特征、编码质量等级和创建交互特征，预期显著提升房价预测模型的准确性和鲁棒性。建议分阶段实施，优先处理高优先级特征，逐步验证效果。