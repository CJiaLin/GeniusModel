# 特征工程方案

## 1. 现有特征分析

### 1.1 数据概况
- **数据集**: 房价预测数据（目标列: SalePrice）
- **特征数量**: 79个特征（数值 + 类别）
- **目标变量**: SalePrice（回归任务）

### 1.2 特征分类

| 类别 | 特征数量 | 主要特征 |
|------|----------|----------|
| **房屋物理属性** | 15+ | LotArea, GrLivArea, 1stFlrSF, 2ndFlrSF, TotalBsmtSF, GarageArea |
| **质量评估** | 8+ | OverallQual, OverallCond, ExterQual, ExterCond, BsmtQual, KitchenQual, GarageQual |
| **时间特征** | 5 | YearBuilt, YearRemodAdd, GarageYrBlt, YrSold, MoSold |
| **房间/设施数量** | 10+ | BedroomAbvGr, FullBath, HalfBath, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageCars |
| **位置信息** | 8+ | Neighborhood, MSZoning, Condition1, Condition2 |
| **建筑类型** | 6+ | BldgType, HouseStyle, RoofStyle, Foundation |

### 1.3 特征特点
- **数值特征**: 34个，包括面积、数量、年份等
- **类别特征**: 43个，包括质量等级、类型、位置等
- **存在缺失值的特征**: LotFrontage, MasVnrArea, GarageYrBlt等

---

## 2. 特征工程策略

### 2.1 面积相关特征聚合

**策略**: 房屋的总面积是房价的核心决定因素，需要构建多个面积聚合特征。

```
总居住面积 = GrLivArea
总面积 = GrLivArea + TotalBsmtSF + GarageArea + 外部附加区域
面积比率 = GrLivArea / LotArea  # 建筑密度
每层平均面积 = GrLivArea / (1stFlrSF + 2ndFlrSF的层数)
```

### 2.2 质量特征编码与聚合

**策略**: 质量等级对房价影响显著，将有序类别转换为数值，并构建综合质量指标。

```
质量加权分数 = OverallQual * 2 + ExterQual_score + BsmtQual_score + KitchenQual_score
质量-面积交互 = OverallQual * GrLivArea
整体状态指数 = OverallQual * OverallCond
```

### 2.3 时间特征工程

**策略**: 房龄和翻新情况对房价有重要影响。

```
房龄 = YrSold - YearBuilt
翻新后年数 = YrSold - YearRemodAdd
是否翻新 = (YearRemodAdd != YearBuilt)
车库年龄 = YrSold - GarageYrBlt
购买月份 = MoSold（季节性）
```

### 2.4 房间配置特征

**策略**: 房间数量和配置比例反映居住舒适度。

```
总浴室数 = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath
每卧室平均面积 = GrLivArea / (BedroomAbvGr + 1)
房间密度 = TotRmsAbvGrd / GrLivArea
卧室比例 = BedroomAbvGr / TotRmsAbvGrd
```

### 2.5 价格相关区域特征

**策略**: 外部附加空间通常有额外价值。

```
外部总空间 = WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
有泳池标志 = (PoolArea > 0)
有壁炉标志 = (Fireplaces > 0)
有车库标志 = (GarageArea > 0)
车库效率 = GarageCars / (GarageArea + 1)
```

### 2.6 高级特征交互

**策略**: 特征交互可能捕捉非线性关系。

```
高质量大面积 = OverallQual * GrLivArea
质量房龄交互 = OverallQual / (房龄 + 1)
地段价值指数 = Neighborhood编码 * GrLivArea
地下室完成度 = BsmtFinSF1 / (TotalBsmtSF + 1)
```

### 2.7 类别特征编码

**策略**: 对高基数类别特征进行目标编码或分组。

```
Neighborhood目标编码 = 按Neighborhood分组的SalePrice均值
MSZoning价格等级 = 按MSZoning分组的SalePrice中位数
Condition组合 = Condition1 + "_" + Condition2
质量等级组合 = ExterQual + "_" + BsmtQual + "_" + KitchenQual
```

### 2.8 多项式与对数变换

**策略**: 处理数值特征的非线性关系。

```
GrLivArea_log = log(GrLivArea + 1)
LotArea_sqrt = sqrt(LotArea)
OverallQual_squared = OverallQual ** 2
TotalBsmtSF_boxcox = Box-Cox变换
```

---

## 3. 要生成的新特征列表

### 3.1 面积聚合特征（7个）

| 新特征名 | 计算公式 | 预期作用 |
|----------|----------|----------|
| `TotalSF` | GrLivArea + TotalBsmtSF + GarageArea | 总建筑空间 |
| `TotalPorchSF` | WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch | 外部附加空间 |
| `BuildingDensity` | GrLivArea / LotArea | 土地利用效率 |
| `Has2ndFloor` | (2ndFlrSF > 0) | 是否有二层 |
| `HasBasement` | (TotalBsmtSF > 0) | 是否有地下室 |
| `HasGarage` | (GarageArea > 0) | 是否有车库 |
| `LivingAreaRatio` | GrLivArea / (1stFlrSF + 2ndFlrSF + 1) | 有效居住比例 |

### 3.2 时间特征（5个）

| 新特征名 | 计算公式 | 预期作用 |
|----------|----------|----------|
| `HouseAge` | YrSold - YearBuilt | 房龄 |
| `YearsSinceRemod` | YrSold - YearRemodAdd | 翻新后年数 |
| `IsNew` | (YrSold == YearBuilt) | 是否新房 |
| `IsRemodeled` | (YearRemodAdd != YearBuilt) | 是否翻新 |
| `GarageAge` | YrSold - GarageYrBlt | 车库年龄 |

### 3.3 质量综合特征（6个）

| 新特征名 | 计算公式 | 预期作用 |
|----------|----------|----------|
| `QualityScore` | OverallQual * 2 + ExterQual_num + BsmtQual_num | 综合质量分 |
| `QualArea` | OverallQual * GrLivArea | 质量-面积交互 |
| `QualCond` | OverallQual * OverallCond | 质量-状态交互 |
| `IsExcellent` | (OverallQual >= 8) | 高品质标志 |
| `IsPoor` | (OverallQual <= 4) | 低品质标志 |
| `HasPool` | (PoolArea > 0) | 有泳池标志 |

### 3.4 房间配置特征（5个）

| 新特征名 | 计算公式 | 预期作用 |
|----------|----------|----------|
| `TotalBath` | FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath | 总浴室当量 |
| `AreaPerRoom` | GrLivArea / (TotRmsAbvGrd + 1) | 每房间平均面积 |
| `BedroomRatio` | BedroomAbvGr / (TotRmsAbvGrd + 1) | 卧室占比 |
| `FamilySize` | BedroomAbvGr + TotalBath | 家庭容量指标 |
| `HasFireplace` | (Fireplaces > 0) | 有壁炉标志 |

### 3.5 高级交互特征（6个）

| 新特征名 | 计算公式 | 预期作用 |
|----------|----------|----------|
| `QualAge` | OverallQual / (HouseAge + 1) | 质量保持度 |
| `NeighborhoodQual` | Neighborhood_target_encode * OverallQual | 地段-质量交互 |
| `BasementFinishRatio` | BsmtFinSF1 / (TotalBsmtSF + 1) | 地下室完工率 |
| `GarageEfficiency` | GarageCars / (GarageArea + 1) | 车库效率 |
| `PricePerSF` | SalePrice / (GrLivArea + 1) | 单位面积价格（目标相关） |
| `LotValue` | LotArea * Neighborhood_target_encode | 土地价值 |

### 3.6 数值变换特征（4个）

| 新特征名 | 计算方法 | 预期作用 |
|----------|----------|----------|
| `GrLivArea_log` | np.log1p(GrLivArea) | 对数变换 |
| `LotArea_sqrt` | np.sqrt(LotArea) | 平方根变换 |
| `TotalSF_log` | np.log1p(TotalSF) | 总面积对数 |
| `QualitySq` | OverallQual ** 2 | 质量平方项 |

### 3.7 类别编码特征（4个）

| 新特征名 | 编码方法 | 预期作用 |
|----------|----------|----------|
| `Neighborhood_Target` | Target Encoding | 地段价格水平 |
| `MSZoning_Target` | Target Encoding | 区域价格水平 |
| `HouseStyle_Qual` | Ordinal Encoding | 房型质量等级 |
| `SaleCondition_Qual` | Ordinal Encoding | 销售条件等级 |

---

## 4. 预期效果

### 4.1 特征维度扩展
- **原始特征数**: 79个
- **新生成特征数**: 37个
- **总特征数**: ~116个

### 4.2 预期模型性能提升

| 指标 | 预期提升 | 说明 |
|------|----------|------|
| **R² Score** | +3-8% | 更丰富的特征表达 |
| **RMSE** | -5-15% | 更好的非线性捕获 |
| **特征重要性** | 更清晰 | 聚合特征通常排名更高 |

### 4.3 各特征组贡献预期

```
高影响力特征组:
├─ 面积聚合特征 (TotalSF, QualArea): 预期贡献 25-30%
├─ 时间特征 (HouseAge, IsNew): 预期贡献 15-20%
├─ 质量交互特征 (QualityScore, QualAge): 预期贡献 15-20%
├─ 房间配置特征 (TotalBath, AreaPerRoom): 预期贡献 10-15%
└─ 位置编码特征 (Neighborhood_Target): 预期贡献 10-15%
```

### 4.4 风险控制

| 风险 | 缓解措施 |
|------|----------|
| 过拟合 | 交叉验证的目标编码 + 正则化 |
| 多重共线性 | VIF检测 + 特征选择 |
| 数据泄露 | 确保时间特征不泄露未来信息 |

---

## 5. 实施建议

### 5.1 执行顺序
1. **数据清洗** → 处理缺失值、异常值
2. **基础特征** → 面积聚合、时间特征
3. **编码特征** → 类别编码、目标编码
4. **交互特征** → 质量交互、位置交互
5. **变换特征** → 对数、平方根变换
6. **特征选择** → 去除冗余特征

### 5.2 验证方法
- 使用5折交叉验证评估特征效果
- 比较特征工程前后的模型性能
- 分析特征重要性验证新特征价值

---

**总结**: 本特征工程方案基于实际数据中的79个原始特征，生成37个新特征，重点聚焦于面积聚合、质量评估、时间演化、房间配置和高级交互五个维度，预期能显著提升房价预测模型的性能。