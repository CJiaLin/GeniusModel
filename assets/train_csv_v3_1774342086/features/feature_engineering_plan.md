# 🏠 特征工程方案：房价预测 (House Price Prediction)

## 1. 现有特征分析

### 1.1 数据概览

| 指标 | 值 |
|------|-----|
| **样本数量** | 1,460 |
| **特征总数** | 81 (含目标列) |
| **目标列** | SalePrice |
| **任务类型** | 回归 |
| **数据质量** | 已清洗 (cleaned_data) |

### 1.2 数值型特征 (38个)

| 特征类别 | 具体特征 |
|---------|---------|
| **面积相关** | LotFrontage, LotArea, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea |
| **房间数量** | BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageCars |
| **建筑时间** | YearBuilt, YearRemodAdd, GarageYrBlt, MoSold, YrSold |
| **评分等级** | OverallQual, OverallCond |
| **其他** | MiscVal |

**关键数值特征统计:**

| 特征 | 均值 | 标准差 | 最小值 | 最大值 | 缺失率 |
|------|------|--------|--------|--------|--------|
| SalePrice | $180,921 | $79,443 | $34,900 | $755,000 | 0% |
| GrLivArea | 1,515 sqft | 525 sqft | 334 | 5,642 | 0% |
| TotalBsmtSF | 1,057 sqft | 438 sqft | 0 | 6,110 | 0% |
| LotArea | 10,516 sqft | 9,982 sqft | 1,300 | 215,245 | 0% |
| OverallQual | 6.1 | 1.4 | 1 | 10 | 0% |

### 1.3 类别型特征 (43个)

| 特征类别 | 具体特征 |
|---------|---------|
| **建筑类型** | MSSubClass, MSZoning, BldgType, HouseStyle, RoofStyle, RoofMatl, Foundation |
| **外部特征** | Exterior1st, Exterior2nd, MasVnrType |
| **公共设施** | Street, Alley, Utilities, Neighborhood, Condition1, Condition2 |
| **内部特征** | Heating, HeatingQC, CentralAir, Electrical, Functional |
| **房间质量** | KitchenQual, FireplaceQu, PoolQC |
| **地下室** | BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2 |
| **车库** | GarageType, GarageFinish, GarageQual, GarageCond |
| **销售相关** | SaleType, SaleCondition, LandContour, LandSlope, LotConfig, Fence, PavedDrive, MiscFeature |

### 1.4 数据特征分析

```
📊 目标变量 SalePrice 分布特征:
   ├── 均值: $180,921
   ├── 中位数: $163,000
   ├── 标准差: $79,443
   ├── 偏度: 1.88 (右偏，建议对数变换)
   └── 价格范围: $34,900 - $755,000

📈 数值特征相关性分析 (与SalePrice):
   ├── OverallQual: 0.79 (质量评分 - 强相关)
   ├── GrLivArea: 0.71 (地上生活面积 - 强相关)
   ├── GarageCars: 0.64 (车库容量)
   ├── GarageArea: 0.62 (车库面积)
   ├── TotalBsmtSF: 0.61 (地下室总面积)
   └── 1stFlrSF: 0.61 (一楼面积)

⚠️ 缺失值特征 (如存在):
   ├── GarageYrBlt, LotFrontage, MasVnrType 等
   └── 已处理为 "None" 或中位数填充
```

---

## 2. 特征工程策略

### 2.1 策略矩阵

| 策略编号 | 策略名称 | 适用特征 | 优先级 |
|---------|---------|---------|--------|
| FE-01 | 面积聚合特征 | 所有SF (平方英尺) 特征 | 🔴 高 |
| FE-02 | 房间比例特征 | 浴室/卧室比例 | 🔴 高 |
| FE-03 | 建筑年龄特征 | 时间相关特征 | 🔴 高 |
| FE-04 | 质量交互特征 | OverallQual × Area | 🟡 中 |
| FE-05 | 对数变换 | 右偏数值特征 | 🟡 中 |
| FE-06 | 类别编码优化 | 高基数的类别特征 | 🟡 中 |
| FE-07 | 邻域价格编码 | Neighborhood | 🟢 低 |
| FE-08 | 多项式特征 | 核心面积特征 | 🟢 低 |

---

## 3. 要生成的新特征列表

### 3.1 面积聚合特征 (FE-01)

| 新特征名 | 计算公式 | 预期作用 |
|---------|---------|---------|
| `TotalSF` | GrLivArea + TotalBsmtSF | 房屋总使用面积 |
| `TotalPorchSF` | OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch | 总门廊面积 |
| `TotalOutdoorSF` | WoodDeckSF + TotalPorchSF + PoolArea | 总户外活动面积 |
| `LotUtilization` | (GrLivArea / LotArea) × 100 | 土地利用率 |
| `BasementFinRatio` | BsmtFinSF1 / (TotalBsmtSF + 1) | 地下室完成比例 |

### 3.2 房间比例特征 (FE-02)

| 新特征名 | 计算公式 | 预期作用 |
|---------|---------|---------|
| `TotalBathrooms` | FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath | 等效总浴室数 |
| `BedroomToBathRatio` | BedroomAbvGr / (TotalBathrooms + 0.1) | 卧室浴室比 |
| `RoomsPerSF` | TotRmsAbvGrd / (GrLivArea + 1) | 房间密度 |
| `KitchenToRoomRatio` | KitchenAbvGr / (TotRmsAbvGrd + 1) | 厨房占比 |

### 3.3 建筑年龄特征 (FE-03)

| 新特征名 | 计算公式 | 预期作用 |
|---------|---------|---------|
| `HouseAge` | YrSold - YearBuilt | 房屋年龄 |
| `RemodAge` | YrSold - YearRemodAdd | 翻新后年数 |
| `IsNewHouse` | 1 if HouseAge <= 1 else 0 | 是否新房 |
| `HasRemod` | 1 if YearRemodAdd != YearBuilt else 0 | 是否翻新 |
| `GarageAge` | YrSold - GarageYrBlt | 车库年龄 |

### 3.4 质量交互特征 (FE-04)

| 新特征名 | 计算公式 | 预期作用 |
|---------|---------|---------|
| `Qual_LivArea` | OverallQual × GrLivArea | 质量面积交互 |
| `Qual_Cond` | OverallQual - OverallCond | 质量与状况差距 |
| `Qual_TotalSF` | OverallQual × TotalSF | 质量总面交互 |
| `Qual_Cond_Combined` | (OverallQual + OverallCond) / 2 | 平均质量状况 |

### 3.5 对数变换特征 (FE-05)

| 新特征名 | 变换公式 | 原特征 |
|---------|---------|--------|
| `LogSalePrice` | log1p(SalePrice) | 目标变量对数变换 |
| `LogGrLivArea` | log1p(GrLivArea) | 地上面积对数 |
| `LogLotArea` | log1p(LotArea) | 土地面积对数 |
| `LogTotalSF` | log1p(TotalSF) | 总面积对数 |
| `LogTotalBsmtSF` | log1p(TotalBsmtSF) | 地下室面积对数 |

### 3.6 类别编码优化 (FE-06)

| 新特征名 | 编码方式 | 原特征 |
|---------|---------|--------|
| `Neighborhood_MeanPrice` | 目标编码 | Neighborhood |
| `MSSubClass_Category` | 分箱编码 | MSSubClass |
| `QualityCategory` | 等级合并 | OverallQual |

### 3.7 特殊标志特征

| 新特征名 | 条件 | 预期作用 |
|---------|------|---------|
| `HasPool` | PoolArea > 0 | 是否有泳池 |
| `Has2ndFloor` | 2ndFlrSF > 0 | 是否有二楼 |
| `HasGarage` | GarageArea > 0 | 是否有车库 |
| `HasBasement` | TotalBsmtSF > 0 | 是否有地下室 |
| `HasFireplace` | Fireplaces > 0 | 是否有壁炉 |
| `HasDeck` | WoodDeckSF > 0 | 是否有露台 |
| `HasFence` | Fence != 'None' | 是否有围栏 |
| `HasAirCond` | CentralAir == 'Y' | 是否有中央空调 |

### 3.8 高级组合特征

| 新特征名 | 计算公式 | 预期作用 |
|---------|---------|---------|
| `PricePerSF` | SalePrice / (GrLivArea + 1) | 单位面积价格 (用于验证) |
| `Qual_Utilization` | OverallQual × LotUtilization | 质量利用率交互 |
| `Age_Qual_Interact` | HouseAge × OverallQual | 年龄质量交互 |
| `LuxuryScore` | HasPool + (Fireplaces > 0) + (GarageCars >= 2) + (OverallQual >= 8) | 豪华度评分 |

---

## 4. 预期效果

### 4.1 特征重要性预测

```
🎯 预期高重要性特征:
   ├── TotalSF (聚合面积)
   ├── Qual_LivArea (质量面积交互)
   ├── HouseAge (房屋年龄)
   ├── TotalBathrooms (总浴室数)
   └── LogGrLivArea (对数面积)

📈 模型性能提升预测:
   ├── 基线模型 (原始特征): ~0.15 RMSE
   ├── 增加面积聚合特征: -15% RMSE
   ├── 增加质量交互特征: -10% RMSE
   ├── 对数变换目标变量: -20% RMSE
   └── 综合预期提升: 25-35%
```

### 4.2 特征工程实施路线图

```
阶段 1 (核心特征):
├── TotalSF, TotalBathrooms
├── HouseAge, HasRemod
└── Log变换目标变量

阶段 2 (交互特征):
├── Qual_LivArea, Qual_TotalSF
├── LotUtilization
└── BedroomToBathRatio

阶段 3 (高级特征):
├── 目标编码 (Neighborhood)
├── 多项式特征 (面积平方项)
└── LuxuryScore 综合评分
```

### 4.3 验证策略

| 验证方法 | 目的 |
|---------|------|
| 特征重要性分析 | 确认新特征的有效性 |
| 相关性热力图 | 检测多重共线性 |
| 交叉验证 | 评估泛化性能 |
| 残差分析 | 验证模型假设 |

---

## 5. 实施注意事项

### 5.1 数据泄漏防护
- 目标编码需使用训练集统计量
- 避免在验证/测试集上拟合编码器

### 5.2 特征缩放
- 面积类特征建议标准化
- 对数变换后特征接近正态分布

### 5.3 类别特征处理
- 低基数: One-Hot Encoding
- 高基数 (Neighborhood): Target Encoding
- 有序类别: Label Encoding

---

**总结**: 本特征工程方案基于实际数据的81个原始特征，通过面积聚合、比例计算、时间变换、质量交互、对数变换和类别编码等策略，预期可生成30+个新特征，显著提升房价预测模型的性能和稳定性。