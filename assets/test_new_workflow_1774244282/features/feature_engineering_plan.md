# 特征工程方案报告

## 1. 现有特征分析

### 1.1 数据概览
- **数据规模**: 1,460 行 × 84 列
- **目标变量**: SalePrice (回归任务)
- **已存在的工程特征**: HouseAge, RemodAge, IsNew, TotalSF, TotalPorchSF, RoomDensity, TotalBath, GarageAreaRatio

### 1.2 特征分类

| 类别 | 特征数量 | 主要特征 |
|------|---------|---------|
| **地块信息** | 7 | LotFrontage, LotArea, LotShape, LotConfig, LandSlope, LandContour, MSZoning |
| **建筑结构** | 12 | BldgType, HouseStyle, MSSubClass, 1stFlrSF, 2ndFlrSF, GrLivArea, TotalBsmtSF |
| **质量评估** | 6 | OverallQual, OverallCond, ExterQual, ExterCond, BsmtQual, BsmtCond |
| **时间相关** | 5 | YearBuilt, YearRemodAdd, MoSold, YrSold, GarageYrBlt |
| **功能设施** | 15 | FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageCars, PoolArea |
| **外部设施** | 6 | WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PavedDrive |
| **分类特征** | 20+ | Neighborhood, Condition1, Condition2, Exterior1st, Exterior2nd, Foundation, RoofStyle 等 |

### 1.3 数据质量问题
- **缺失值**: 
  - Bsmt相关: 37-38个 (约2.5%)
  - FireplaceQu: 690个 (约47%)
  - Garage相关: 81个 (约5.5%)
- **已处理**: 已有特征无缺失值，说明数据已初步清洗

---

## 2. 特征工程策略

### 2.1 策略一：面积比率特征 (Ratio Features)
**原理**: 相对比例比绝对值更能反映房屋结构特征

### 2.2 策略二：质量聚合特征 (Quality Aggregation)
**原理**: 综合多个质量指标形成总体质量评分

### 2.3 策略三：交互特征 (Interaction Features)
**原理**: 捕捉特征间的协同效应

### 2.4 策略四：时间衍生特征 (Temporal Features)
**原理**: 挖掘销售时间和建造时间的模式

### 2.5 策略五：分类特征编码 (Categorical Encoding)
**原理**: 将高基数分类特征转换为数值表示

---

## 3. 要生成的新特征列表

### 3.1 面积比率特征 (Area Ratios)

| 新特征名 | 计算公式 | 预期作用 |
|---------|---------|---------|
| `BsmtFinRatio` | BsmtFinSF1 / TotalBsmtSF | 地下室完成比例 |
| `BsmtUnfRatio` | BsmtUnfSF / TotalBsmtSF | 地下室未完成比例 |
| `1stFlrRatio` | 1stFlrSF / TotalSF | 一层面积占比 |
| `2ndFlrRatio` | 2ndFlrSF / TotalSF | 二层面积占比 |
| `LotFrontageRatio` | LotFrontage / LotArea | 地块前宽占比 |
| `LivingAreaRatio` | GrLivArea / TotalSF | 生活面积占比 |
| `GarageCarsRatio` | GarageCars / (BedroomAbvGr + 1) | 车位与卧室比 |

### 3.2 质量聚合特征 (Quality Features)

| 新特征名 | 计算公式 | 预期作用 |
|---------|---------|---------|
| `OverallScore` | OverallQual × OverallCond | 总体质量评分 |
| `ExterScore` | ExterQual编码 × ExterCond编码 | 外部质量综合 |
| `BsmtScore` | BsmtQual编码 + BsmtCond编码 + BsmtExposure编码 | 地下室质量综合 |
| `QualityIndex` | (OverallQual + ExterQual编码 + KitchenQual编码 + BsmtQual编码) / 4 | 平均质量指数 |

### 3.3 交互特征 (Interaction Features)

| 新特征名 | 计算公式 | 预期作用 |
|---------|---------|---------|
| `Qual_LivArea` | OverallQual × GrLivArea | 质量与生活面积交互 |
| `Age_Qual` | HouseAge × OverallQual | 年龄与质量交互 |
| `TotalSF_Bedroom` | TotalSF / (BedroomAbvGr + 1) | 每卧室平均面积 |
| `Qual_Garage` | OverallQual × GarageArea | 质量与车库交互 |
| `Remod_Effect` | RemodAge × (YearRemodAdd - YearBuilt) | 改造效果评估 |

### 3.4 时间衍生特征 (Temporal Features)

| 新特征名 | 计算公式 | 预期作用 |
|---------|---------|---------|
| `SeasonSold` | 根据MoSold分箱(春夏秋冬) | 销售季节 |
| `IsRecentRemod` | (YrSold - YearRemodAdd) <= 5 | 近期改造标志 |
| `SaleYearCategory` | YrSold分箱 | 销售年份类别 |
| `HouseAgeDecade` | HouseAge // 10 | 房屋年龄 decade |

### 3.5 功能组合特征 (Functional Features)

| 新特征名 | 计算公式 | 预期作用 |
|---------|---------|---------|
| `Has2ndFloor` | (2ndFlrSF > 0).astype(int) | 是否有二层 |
| `HasBasement` | (TotalBsmtSF > 0).astype(int) | 是否有地下室 |
| `HasGarage` | (GarageArea > 0).astype(int) | 是否有车库 |
| `HasFireplace` | (Fireplaces > 0).astype(int) | 是否有壁炉 |
| `HasPool` | (PoolArea > 0).astype(int) | 是否有泳池 |
| `HasPorch` | (TotalPorchSF > 0).astype(int) | 是否有门廊 |
| `HasMultipleFeatures` | Has2ndFloor + HasBasement + HasGarage + HasFireplace | 多功能设施计数 |

### 3.6 空间效率特征 (Efficiency Features)

| 新特征名 | 计算公式 | 预期作用 |
|---------|---------|---------|
| `RoomsPerBedroom` | TotRmsAbvGrd / (BedroomAbvGr + 1) | 房间与卧室比 |
| `SFPerRoom` | GrLivArea / (TotRmsAbvGrd + 1) | 每房间面积 |
| `BathPerBedroom` | TotalBath / (BedroomAbvGr + 1) | 每卧室浴室数 |
| `GarageEfficiency` | GarageCars / (GarageArea + 1) | 车库空间效率 |

---

## 4. 预期效果

### 4.1 特征重要性预期

| 特征类别 | 预期重要性 | 理由 |
|---------|-----------|------|
| 质量聚合特征 | ⭐⭐⭐⭐⭐ | 房屋质量是房价的核心决定因素 |
| 面积比率特征 | ⭐⭐⭐⭐ | 反映房屋结构合理性 |
| 交互特征 | ⭐⭐⭐⭐ | 捕捉非线性关系 |
| 功能组合特征 | ⭐⭐⭐ | 指示房屋设施完善度 |
| 时间衍生特征 | ⭐⭐⭐ | 反映市场趋势和房屋新旧程度 |

### 4.2 模型性能预期

| 指标 | 预期改进 |
|-----|---------|
| R² Score | 提升 3-7% |
| RMSE | 降低 5-10% |
| MAE | 降低 4-8% |

### 4.3 特征工程价值

1. **可解释性增强**: 质量聚合和面积比率特征具有明确的业务含义
2. **非线性关系捕捉**: 交互特征帮助模型学习特征协同效应
3. **维度扩展**: 从84维扩展至约110维，提供更丰富的信息
4. **稳健性提升**: 多重编码和分箱处理增强模型泛化能力

---

## 5. 实施建议

### 5.1 优先级排序
1. **高优先级**: 质量聚合特征、交互特征（Qual_LivArea, OverallScore）
2. **中优先级**: 面积比率特征、功能组合特征
3. **低优先级**: 时间衍生特征（部分已存在于基础特征中）

### 5.2 验证方法
- 使用特征重要性分析（如Random Forest的feature_importances_）
- 递归特征消除（RFE）筛选有效特征
- 交叉验证评估特征增量的实际贡献

### 5.3 注意事项
- 避免多重共线性：检查新生成特征与现有特征的相关系数
- 防止过拟合：高阶交互特征需谨慎使用
- 编码一致性：分类特征编码需保持训练集和测试集一致