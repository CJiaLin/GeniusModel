# 特征工程方案

## 1. 现有特征分析

### 1.1 数据概况
| 指标 | 数值 |
|------|------|
| 样本数量 | 1,460 |
| 特征总数 | 81 |
| 数值特征 | 37 |
| 类别特征 | 43 |
| 目标列 | SalePrice |

### 1.2 目标变量分析
- **类型**: 连续数值型（回归任务）
- **分布**: 右偏分布（skewness > 0）
- **范围**: $34,900 - $755,000
- **均值**: $180,921
- **处理建议**: 建议对目标变量进行对数变换（log1p）以改善分布

### 1.3 特征质量分析

#### 高缺失率特征（>90%）
| 特征名 | 缺失率 | 建议处理 |
|--------|--------|----------|
| PoolQC | 99.5% | 删除或转为"无泳池"类别 |
| MiscFeature | 96.3% | 删除或合并 |
| Alley | 93.7% | 删除或转为"无通道"类别 |
| Fence | 80.8% | 填充为"无围栏" |
| FireplaceQu | 47.3% | 填充为"无壁炉" |

#### 高相关性数值特征（与SalePrice > 0.5）
- **GrLivArea** (0.71): 地面生活面积
- **TotalBsmtSF** (0.61): 地下室总面积
- **GarageArea** (0.62): 车库面积
- **GarageCars** (0.64): 车库容量
- **OverallQual** (0.79): 整体质量评分

---

## 2. 特征工程策略

### 2.1 数据清洗
```python
# 删除或处理高缺失率特征
drop_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
fill_none_features = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                      'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
```

### 2.2 数值特征工程

#### A. 面积相关特征
| 策略 | 新特征名 | 计算方法 |
|------|---------|---------|
| 总面积 | TotalArea | GrLivArea + TotalBsmtSF + GarageArea |
| 总面积对数 | LogTotalArea | log1p(TotalArea) |
| 生活面积比 | LivingAreaRatio | GrLivArea / LotArea |
| 地下室完成率 | BsmtFinRatio | BsmtFinSF1 / TotalBsmtSF |

#### B. 时间相关特征
| 策略 | 新特征名 | 计算方法 |
|------|---------|---------|
| 房屋年龄 | HouseAge | YrSold - YearBuilt |
| 翻新年数 | RemodelAge | YrSold - YearRemodAdd |
| 是否新建 | IsNewHouse | (YrSold == YearBuilt).astype(int) |
| 车库年龄 | GarageAge | YrSold - GarageYrBlt |

#### C. 比率特征
| 策略 | 新特征名 | 计算方法 |
|------|---------|---------|
| 卧室密度 | BedroomDensity | BedroomAbvGr / GrLivArea |
| 浴室比例 | BathRatio | (FullBath + 0.5*HalfBath) / GrLivArea |
| 价值密度 | ValuePerSqFt | SalePrice / GrLivArea |

### 2.3 类别特征工程

#### A. 有序编码（具有内在顺序的特征）
```python
quality_mapping = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
```

#### B. 独热编码（无序类别特征）
- Neighborhood (25个类别)
- HouseStyle (8个类别)
- MSSubClass (15个类别)
- GarageType (6个类别)

#### C. 目标编码（高基数类别特征）
```python
# 对Neighborhood进行目标编码
neighborhood_mean_price = df.groupby('Neighborhood')['SalePrice'].mean()
df['Neighborhood_Encoded'] = df['Neighborhood'].map(neighborhood_mean_price)
```

#### D. 特征组合
| 新特征名 | 组合方式 |
|---------|---------|
| QualCond_Score | OverallQual * OverallCond |
| Exter_Score | ExterQual_num * ExterCond_num |
| Kitchen_Grade | KitchenQual_num * KitchenAbvGr |
| Qual_Area_Interact | OverallQual * GrLivArea |

### 2.4 特征变换

#### A. 对数变换（右偏数值特征）
```python
log_features = ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF']
for feat in log_features:
    df[f'{feat}_log'] = np.log1p(df[feat])
```

#### B. 箱式变换（Box-Cox）
```python
from scipy.stats import boxcox
df['SalePrice_boxcox'], _ = boxcox(df['SalePrice'] + 1)
```

#### C. 多项式特征（重要数值特征）
```python
from sklearn.preprocessing import PolynomialFeatures
poly_features = ['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'GarageCars']
```

---

## 3. 要生成的新特征列表

### 3.1 核心新特征（推荐优先级：高）

| 序号 | 特征名 | 类型 | 描述 | 预期效果 |
|-----|--------|------|------|---------|
| 1 | TotalArea | 数值 | 房屋总面积（居住+地下室+车库） | 综合衡量房屋规模 |
| 2 | HouseAge | 数值 | 房屋年龄 | 反映折旧程度 |
| 3 | IsNewHouse | 二元 | 是否为新房 | 捕捉新房溢价 |
| 4 | QualCond_Score | 数值 | 质量与状况乘积 | 综合质量指标 |
| 5 | LogGrLivArea | 数值 | 生活面积对数 | 改善分布偏度 |
| 6 | TotalBathrooms | 数值 | 全浴+0.5*半浴 | 统一浴室指标 |
| 7 | HasBasement | 二元 | 是否有地下室 | 二元特征 |
| 8 | HasGarage | 二元 | 是否有车库 | 二元特征 |
| 9 | HasPool | 二元 | 是否有泳池 | 稀有特征指示 |
| 10 | Has2ndFloor | 二元 | 是否有二楼 | 楼层特征 |

### 3.2 进阶新特征（推荐优先级：中）

| 序号 | 特征名 | 类型 | 描述 | 预期效果 |
|-----|--------|------|------|---------|
| 11 | Neighborhood_MeanPrice | 数值 | 社区平均房价 | 位置价值量化 |
| 12 | LivingAreaRatio | 数值 | 生活面积/地块面积 | 土地利用效率 |
| 13 | BsmtFinRatio | 数值 | 地下室完成比例 | 地下室质量 |
| 14 | RemodelAge | 数值 | 距离翻新年数 | 翻新效果 |
| 15 | Qual_Area_Interact | 数值 | 质量*面积 | 交互特征 |
| 16 | BedroomDensity | 数值 | 卧室数/生活面积 | 空间密度 |
| 17 | PorchArea | 数值 | 所有门廊面积之和 | 户外空间 |
| 18 | SeasonSold | 类别 | 销售季节 | 时间周期性 |
| 19 | MSSubClass_Grouped | 类别 | 住宅类型分组 | 降维处理 |
| 20 | Functional_Simple | 类别 | 功能简化分类 | 降维处理 |

### 3.3 高级特征（推荐优先级：低，用于模型优化）

| 序号 | 特征名 | 类型 | 描述 | 预期效果 |
|-----|--------|------|------|---------|
| 21 | PCA_Area_Components | 数值(2) | 面积特征PCA | 降维去噪 |
| 22 | Cluster_Neighborhood | 数值 | 社区聚类标签 | 社区分层 |
| 23 | PricePerRoom | 数值 | 每房间价格 | 价值密度 |
| 24 | LuxuryScore | 数值 | 豪华设施评分 | 高端市场指示 |
| 25 | UtilityEfficiency | 数值 | 公用设施效率 | 功能性指标 |

---

## 4. 预期效果

### 4.1 模型性能提升预测

| 评估指标 | 基线模型 | 特征工程后 | 提升幅度 |
|---------|---------|-----------|---------|
| RMSE | ~45,000 | ~35,000 | 22%↓ |
| RMSLE | ~0.15 | ~0.12 | 20%↓ |
| R² Score | 0.78 | 0.88 | 13%↑ |
| Kaggle Score | ~0.15 | ~0.12 | 20%↓ |

### 4.2 各策略预期贡献

```
特征工程贡献度预测:
├── 目标对数变换: +15% 稳定性
├── 面积特征组合: +12% 预测力
├── 时间特征提取: +8% 解释力
├── 质量编码优化: +10% 准确性
├── 比率特征构建: +7% 区分度
└── 类别特征编码: +8% 表达能力
```

### 4.3 风险与注意事项

| 风险点 | 说明 | 缓解措施 |
|--------|------|---------|
| 过拟合 | 新特征过多可能导致过拟合 | 使用交叉验证，特征选择 |
| 数据泄露 | 目标编码需防止泄露 | 仅使用训练集统计量 |
| 分布偏移 | 测试集分布可能不同 | 稳健的特征变换 |
| 共线性 | 新特征可能与原特征高度相关 | VIF检验，PCA降维 |

### 4.4 实施建议

1. **阶段1**: 实施核心新特征（1-10），预计可获得80%的性能提升
2. **阶段2**: 添加进阶特征（11-20），精细调优模型
3. **阶段3**: 测试高级特征（21-25），针对特定模型优化
4. **验证**: 使用5折交叉验证评估每阶段效果

---

## 附录：完整代码框架

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import boxcox

class FeatureEngineer:
    def __init__(self):
        self.le_dict = {}
        self.target_means = {}
        
    def fit(self, df, target_col='SalePrice'):
        # 学习编码参数
        pass
        
    def transform(self, df):
        # 应用特征工程
        df = df.copy()
        
        # 1. 基础清洗
        df = self._clean_data(df)
        
        # 2. 数值特征工程
        df = self._create_area_features(df)
        df = self._create_time_features(df)
        df = self._create_ratio_features(df)
        
        # 3. 类别特征工程
        df = self._encode_ordinal(df)
        df = self._encode_categorical(df)
        
        # 4. 特征变换
        df = self._transform_features(df)
        
        return df
    
    def _clean_data(self, df):
        # 缺失值处理
        return df
    
    def _create_area_features(self, df):
        df['TotalArea'] = df['GrLivArea'] + df['TotalBsmtSF'].fillna(0) + df['GarageArea'].fillna(0)
        df['LivingAreaRatio'] = df['GrLivArea'] / df['LotArea']
        return df
    
    def _create_time_features(self, df):
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['IsNewHouse'] = (df['YrSold'] == df['YearBuilt']).astype(int)
        return df
    
    # ... 其他方法
```

---

**总结**: 本方案针对房价预测任务的特点，从面积、时间、质量、类别四个维度构建25个新特征，预期可显著提升模型性能。建议按阶段实施，并在每阶段验证效果。