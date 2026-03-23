# 特征工程方案报告

## 1. 现有特征分析

### 1.1 数据概述
| 属性 | 值 |
|------|-----|
| 样本数量 | 1,460 |
| 特征数量 | 81 (含Id和目标列) |
| 目标变量 | `SalePrice` (房价，单位美元) |
| 任务类型 | 回归 |
| 目标分布 | 右偏分布 (均值$180,921，范围$34,900-$755,000) |

### 1.2 特征分类

| 类别 | 列数 | 特征示例 |
|------|------|----------|
| **房屋质量评分** | 2 | `OverallQual`, `OverallCond` |
| **面积相关** | 12 | `LotArea`, `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `GarageArea` 等 |
| **时间相关** | 3 | `YearBuilt`, `YearRemodAdd`, `GarageYrBlt` |
| **房间数量** | 7 | `BedroomAbvGr`, `KitchenAbvGr`, `TotRmsAbvGrd`, `FullBath`, `HalfBath` 等 |
| **地理位置** | 2 | `Neighborhood`, `MSZoning` |
| **建筑材料/质量** | 15 | `ExterQual`, `BsmtQual`, `KitchenQual`, `HeatingQC` 等 |
| **便利设施** | 8 | `PoolArea`, `Fireplaces`, `GarageCars`, `WoodDeckSF` 等 |
| **销售信息** | 4 | `MoSold`, `YrSold`, `SaleType`, `SaleCondition` |

### 1.3 数据质量问题

| 问题类型 | 特征 | 缺失比例 | 处理建议 |
|----------|------|----------|----------|
| 高缺失率 | `PoolQC` | 99.5% | 缺失表示无泳池，需填充"None" |
| 高缺失率 | `MiscFeature` | 96.3% | 缺失表示无特殊设施 |
| 高缺失率 | `Alley` | 93.8% | 缺失表示无小巷通道 |
| 高缺失率 | `Fence` | 80.8% | 缺失表示无围栏 |
| 中等缺失率 | `FireplaceQu` | 47.3% | 缺失表示无壁炉 |
| 中等缺失率 | `LotFrontage` | 17.7% | 需中位数/均值填充 |
| 低缺失率 | 其他地下室/车库特征 | 2-6% | 模式填充 |

---

## 2. 特征工程策略

### 2.1 缺失值处理策略

```python
# 缺失值编码策略（基于实际数据特征）
NONE_FEATURES = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType']
ZERO_FEATURES = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
                 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
                 'BsmtHalfBath', 'MasVnrArea']
```

### 2.2 特征转换策略

| 策略 | 目标特征 | 原因 |
|------|----------|------|
| 对数变换 | `SalePrice`, `LotArea` | 目标右偏，大数值特征长尾分布 |
| 类别编码 | `ExterQual`, `BsmtQual` 等 | 有序类别(Ex>Gd>TA>Fa>Po) |
| 独热编码 | `Neighborhood`, `HouseStyle` 等 | 名义类别特征 |

### 2.3 特征创建策略

| 策略类别 | 具体方法 | 适用场景 |
|----------|----------|----------|
| **面积聚合** | 总面积、平均面积 | 合并分散的面积特征 |
| **时间特征** | 房龄、翻新后年数 | 提取时间隐含信息 |
| **比率特征** | 卧室/浴室比、价格/面积比 | 捕获相对关系 |
| **质量聚合** | 加权质量评分 | 综合多个质量指标 |
| **设施计数** | 高级设施数量 | 综合便利设施 |

---

## 3. 要生成的新特征列表

### 3.1 面积相关特征 (6个)

| 新特征名 | 计算公式 | 预期效果 |
|----------|----------|----------|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | 房屋总使用面积，强预测因子 |
| `TotalPorchSF` | `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF` | 户外空间总面积 |
| `Has2ndFloor` | `1 if 2ndFlrSF > 0 else 0` | 是否两层建筑 |
| `HasBasement` | `1 if TotalBsmtSF > 0 else 0` | 是否有地下室 |
| `HasGarage` | `1 if GarageArea > 0 else 0` | 是否有车库 |
| `HasPool` | `1 if PoolArea > 0 else 0` | 是否有泳池 |

### 3.2 时间相关特征 (3个)

| 新特征名 | 计算公式 | 预期效果 |
|----------|----------|----------|
| `HouseAge` | `YrSold - YearBuilt` | 房龄，负相关因子 |
| `RemodAge` | `YrSold - YearRemodAdd` | 翻新后年数 |
| `IsNewHouse` | `1 if YrSold == YearBuilt else 0` | 是否新房，通常溢价 |

### 3.3 质量聚合特征 (2个)

| 新特征名 | 计算公式 | 预期效果 |
|----------|----------|----------|
| `OverallQualCond` | `OverallQual * OverallCond` | 质量与状况交互，高价值标志 |
| `QualScore` | 加权:`OverallQual*3 + ExterQual_enc + BsmtQual_enc + KitchenQual_enc + GarageQual_enc + FireplaceQu_enc` | 综合质量评分 |

### 3.4 房间与设施特征 (4个)

| 新特征名 | 计算公式 | 预期效果 |
|----------|----------|----------|
| `TotalBath` | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` | 总浴室当量 |
| `BedroomToBathRatio` | `BedroomAbvGr / TotalBath` | 卧室浴室比 |
| `HighQualFacilityCount` | `HasPool + HasFireplace + (GarageCars>=2) + (OverallQual>=8)` | 高端设施计数 |
| `HasFireplace` | `1 if Fireplaces > 0 else 0` | 是否有壁炉 |

### 3.5 价格效率特征 (1个)

| 新特征名 | 计算公式 | 预期效果 |
|----------|----------|----------|
| `PricePerSF` | `SalePrice / TotalSF` (仅在训练后分析) | 单位面积价格，识别异常值 |

---

## 4. 预期效果

### 4.1 特征维度变化

| 阶段 | 特征数量 | 说明 |
|------|----------|------|
| 原始特征 | 80 (不含Id) | 原始数据 |
| 缺失值处理后 | 80 | 统一编码策略 |
| 新特征创建后 | **~95** | 新增15个特征 |
| 编码后 | **~250** | 独热编码扩展 |

### 4.2 模型性能预期

| 指标 | 基线模型 | 特征工程后 | 提升 |
|------|----------|------------|------|
| RMSLE | ~0.15 | ~0.12 | 20% ↓ |
| R² | ~0.85 | ~0.92 | 8% ↑ |
| CV稳定性 | ±0.02 | ±0.01 | 50% ↑ |

### 4.3 关键特征重要性预测

基于领域知识，预期Top10重要特征：

1. `OverallQual` - 综合质量评分
2. `TotalSF` - 总面积（新创建）
3. `GrLivArea` - 地上居住面积
4. `GarageCars` / `GarageArea` - 车库容量
5. `HouseAge` - 房龄（新创建）
6. `TotalBsmtSF` - 地下室面积
7. `FullBath` / `TotalBath` - 浴室数量
8. `YearBuilt` / `YearRemodAdd` - 建造/翻新年份
9. `Neighborhood` - 社区（独热编码后）
10. `QualScore` - 综合质量评分（新创建）

### 4.4 实施建议

```python
# 推荐的特征工程流水线
pipeline_steps = [
    "1. 缺失值处理 (None/0/中位数填充)",
    "2. 异常值处理 (GrLivArea > 4000, SalePrice异常值)",
    "3. 对数变换 (LotArea, 可选SalePrice)",
    "4. 有序类别编码 (质量评级 Ex->5, Gd->4...)",
    "5. 创建聚合特征 (面积、时间、比率)",
    "6. 独热编码 (名义类别)",
    "7. 特征缩放 (标准化/归一化)"
]
```

---

**总结**: 该特征工程方案针对Kaggle房价预测数据集的81个原始特征，通过系统性的缺失值处理、15个新特征创建、以及合理的编码策略，预计可将模型性能提升15-25%，同时增强模型的可解释性和稳定性。