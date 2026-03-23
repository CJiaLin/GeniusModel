# 🏠 房屋价格预测特征工程方案

## 1. 现有特征分析

### 1.1 数据概览
| 属性 | 数值 |
|------|------|
| 样本数量 | 1,460 |
| 特征总数 | 81 |
| 数值特征 | 38 |
| 分类特征 | 43 |
| 目标变量 | SalePrice |

### 1.2 目标变量分析
- **分布特征**: 右偏分布（均值 $180,921 > 中位数 $163,000）
- **标准差**: $79,442，价格波动较大
- **建议**: 需要进行对数变换处理偏态

### 1.3 特征质量评估
**高缺失率特征（考虑删除或特殊处理）**:
| 特征 | 缺失率 | 建议 |
|------|--------|------|
| PoolQC | 99.5% | 创建"是否有泳池"二元特征 |
| MiscFeature | 96.3% | 删除或二元化 |
| Fence | 80.8% | 创建"是否有围栏"二元特征 |
| FireplaceQu | 47.3% | 填补为"无壁炉"类别 |

**高相关性数值特征（与SalePrice）**:
- `GrLivArea`: 0.71 (地上居住面积)
- `GarageCars`: 0.64 (车库容量)
- `GarageArea`: 0.62 (车库面积)
- `TotalBsmtSF`: 0.61 (地下室总面积)
- `1stFlrSF`: 0.61 (首层面积)

---

## 2. 特征工程策略

### 2.1 数值特征处理

#### A. 偏态分布变换
```python
# 对高度偏态的数值特征进行对数变换
log_transform_features = ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'SalePrice']
```

#### B. 面积特征组合
| 新特征 | 计算公式 | 业务含义 |
|--------|----------|----------|
| `TotalArea` | GrLivArea + TotalBsmtSF | 房屋总使用面积 |
| `AvgRoomSize` | GrLivArea / TotRmsAbvGrd | 平均房间大小 |
| `OutdoorArea` | LotArea - 1stFlrSF | 室外可用面积 |
| `BsmtRatio` | TotalBsmtSF / 1stFlrSF | 地下室占比 |

#### C. 时间特征工程
```python
# 从YrSold和YearBuilt提取
Age = YrSold - YearBuilt                    # 房屋年龄
RemodAge = YrSold - YearRemodAdd            # 翻新后年数
IsNew = 1 if YrSold == YearBuilt else 0     # 是否新房
```

#### D. 比率特征
| 新特征 | 计算公式 | 业务含义 |
|--------|----------|----------|
| `PricePerSqFt` | SalePrice / GrLivArea | 每平方英尺价格 |
| `LotRatio` | GrLivArea / LotArea | 房屋占地比例 |
| `BathroomRatio` | FullBath + 0.5*HalfBath | 有效浴室数 |

### 2.2 分类特征处理

#### A. 有序编码（质量等级特征）
```python
quality_mapping = {
    'Ex': 5,  # Excellent
    'Gd': 4,  # Good
    'TA': 3,  # Typical/Average
    'Fa': 2,  # Fair
    'Po': 1,  # Poor
    'NA': 0   # No Feature
}
# 适用特征: ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond, PoolQC
```

#### B. 目标编码（高基数分类特征）
```python
# 对Neighborhood等基数高的特征进行目标均值编码
# 注意：使用交叉验证防止过拟合
```

#### C. 独热编码（低基数分类特征）
```python
# 对基数 ≤ 5 的特征进行One-Hot编码
# 如: MSZoning, Street, Alley, LotShape, LandContour, Utilities
```

#### D. 分组聚合特征
```python
# 按Neighborhood聚合数值特征
Neighborhood_MeanPrice = df.groupby('Neighborhood')['SalePrice'].transform('mean')
Neighborhood_MeanArea = df.groupby('Neighborhood')['GrLivArea'].transform('mean')
```

### 2.3 特征交互

#### A. 多项式特征
```python
# 对关键面积特征创建二次项
poly_features = ['GrLivArea', 'TotalBsmtSF', 'GarageArea']
# 生成平方项和交互项
```

#### B. 条件特征
| 新特征 | 条件 | 业务含义 |
|--------|------|----------|
| `HasBasement` | TotalBsmtSF > 0 | 是否有地下室 |
| `HasGarage` | GarageArea > 0 | 是否有车库 |
| `Has2ndFloor` | 2ndFlrSF > 0 | 是否有二层 |
| `HasPool` | PoolArea > 0 | 是否有泳池 |
| `HasFireplace` | Fireplaces > 0 | 是否有壁炉 |
| `HasDeck` | WoodDeckSF > 0 | 是否有木制甲板 |

### 2.4 降维策略
```python
# 对高度相关的面积特征进行PCA
pca_features = ['GrLivArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'TotalBsmtSF']
```

---

## 3. 要生成的新特征列表

### 3.1 数值衍生特征（15个）
| 序号 | 特征名 | 类型 | 来源 | 优先级 |
|------|--------|------|------|--------|
| 1 | `TotalSF` | 数值 | GrLivArea + TotalBsmtSF | ⭐⭐⭐ |
| 2 | `HouseAge` | 数值 | YrSold - YearBuilt | ⭐⭐⭐ |
| 3 | `RemodAge` | 数值 | YrSold - YearRemodAdd | ⭐⭐⭐ |
| 4 | `AvgRoomSize` | 数值 | GrLivArea / TotRmsAbvGrd | ⭐⭐⭐ |
| 5 | `BsmtRatio` | 数值 | TotalBsmtSF / 1stFlrSF | ⭐⭐ |
| 6 | `LotRatio` | 数值 | GrLivArea / LotArea | ⭐⭐ |
| 7 | `BathScore` | 数值 | FullBath + 0.5*HalfBath | ⭐⭐ |
| 8 | `TotalPorchSF` | 数值 | 所有门廊面积之和 | ⭐⭐ |
| 9 | `OutdoorSpace` | 数值 | LotArea - 1stFlrSF | ⭐ |
| 10 | `QualityScore` | 数值 | OverallQual * OverallCond | ⭐⭐⭐ |
| 11 | `PricePerSqFt` | 数值 | SalePrice / GrLivArea | ⭐⭐⭐ |
| 12 | `GrLivArea_log` | 数值 | log(GrLivArea) | ⭐⭐⭐ |
| 13 | `LotArea_log` | 数值 | log(LotArea) | ⭐⭐ |
| 14 | `SalePrice_log` | 数值 | log(SalePrice) | ⭐⭐⭐ |
| 15 | `Neighborhood_PriceMean` | 数值 | 按Neighborhood聚合 | ⭐⭐ |

### 3.2 二元指示特征（7个）
| 序号 | 特征名 | 条件 | 优先级 |
|------|--------|------|--------|
| 1 | `HasBasement` | TotalBsmtSF > 0 | ⭐⭐ |
| 2 | `HasGarage` | GarageArea > 0 | ⭐⭐ |
| 3 | `Has2ndFloor` | 2ndFlrSF > 0 | ⭐⭐ |
| 4 | `HasPool` | PoolArea > 0 | ⭐ |
| 5 | `HasFireplace` | Fireplaces > 0 | ⭐⭐ |
| 6 | `HasDeck` | WoodDeckSF > 0 | ⭐ |
| 7 | `IsNewHouse` | YrSold == YearBuilt | ⭐⭐⭐ |

### 3.3 编码后特征
| 特征组 | 编码方式 | 预期数量 |
|--------|----------|----------|
| 质量等级特征(10个) | 有序编码: Ex=5→NA=0 | 10 |
| Neighborhood | 目标编码 | 1 |
| MSSubClass等(15个) | One-Hot编码 | ~50 |
| 二元分类特征(10个) | Label编码 | 10 |

### 3.4 特征交互（多项式）
| 组合 | 新特征 | 预期数量 |
|------|--------|----------|
| GrLivArea × TotalBsmtSF | 面积交互项 | 1 |
| GrLivArea², TotalBsmtSF² | 平方项 | 2 |

---

## 4. 预期效果

### 4.1 性能提升预估

| 指标 | 基准模型 | 特征工程后 | 提升 |
|------|----------|------------|------|
| **RMSE** | ~35,000 | ~25,000 | **-28.6%** |
| **R² Score** | ~0.80 | ~0.90 | **+12.5%** |
| **MAE** | ~25,000 | ~18,000 | **-28.0%** |

### 4.2 关键改进点

1. **目标变量对数变换**: 将右偏分布转为正态分布，显著提升线性模型性能
2. **面积特征组合**: TotalSF预期成为最强预测因子之一
3. **年龄特征**: HouseAge捕捉房屋折旧效应
4. **质量编码**: 将有序分类转为数值，保留质量层次信息
5. **邻域目标编码**: 捕捉地理位置对价格的非线性影响

### 4.3 模型适配建议

| 模型类型 | 推荐特征处理方式 |
|----------|------------------|
| 线性回归/Ridge/Lasso | 全部数值特征 + 对数变换 + One-Hot编码 |
| 随机森林/XGBoost | 保留分类特征原始形式 + 重点使用交互特征 |
| 神经网络 | 标准化/归一化所有数值特征 + Embedding层处理分类特征 |

### 4.4 特征重要性预期TOP10
```
1. OverallQual        (原始质量评分)
2. TotalSF            (总面积 - 新特征)
3. GrLivArea_log      (地上面积对数)
4. HouseAge           (房屋年龄)
5. GarageCars         (车库容量)
6. QualityScore       (质量综合评分)
7. Neighborhood       (地理位置编码)
8. FullBath           (完整浴室数)
9. YearBuilt          (建造年份)
10. BsmtRatio         (地下室比例)
```

---

## 5. 实施路线图

```
阶段1: 数据清洗与准备
   ├── 缺失值处理
   ├── 异常值检测
   └── 数据类型转换

阶段2: 核心特征工程
   ├── 数值特征变换（对数、多项式）
   ├── 面积特征组合
   ├── 时间特征提取
   └── 比率特征计算

阶段3: 分类特征编码
   ├── 有序特征映射
   ├── 目标编码（高基数）
   └── One-Hot编码（低基数）

阶段4: 高级特征
   ├── 特征交互项
   ├── 分组聚合特征
   └── 特征选择（基于重要性）

阶段5: 验证与优化
   ├── 交叉验证
   ├── 特征重要性分析
   └── 迭代优化
```

---

## 附录: 关键代码模板

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ========== 1. 数值特征工程 ==========
def create_numeric_features(df):
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['AvgRoomSize'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)
    df['QualityScore'] = df['OverallQual'] * df['OverallCond']
    
    # 对数变换
    for col in ['GrLivArea', 'LotArea', 'TotalSF', 'SalePrice']:
        df[f'{col}_log'] = np.log1p(df[col])
    
    return df

# ========== 2. 二元特征工程 ==========
def create_boolean_features(df):
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['IsNewHouse'] = (df['YrSold'] == df['YearBuilt']).astype(int)
    return df

# ========== 3. 有序编码 ==========
quality_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}
quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual']

def encode_quality(df):
    for col in quality_cols:
        df[f'{col}_enc'] = df[col].map(quality_map).fillna(0)
    return df
```

此方案预计可生成 **100+ 个特征**，通过特征选择后保留 **30-50 个核心特征** 用于最终建模。