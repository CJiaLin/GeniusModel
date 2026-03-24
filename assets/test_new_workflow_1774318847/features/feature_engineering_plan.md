# 房价预测特征工程方案

## 1. 现有特征分析

### 数据概览
| 指标 | 数值 |
|------|------|
| 样本数量 | 1,460 |
| 特征数量 | 69 (含目标变量) |
| 目标变量 | SalePrice |
| 数据质量 | 优秀，无缺失值 |

### 特征分类

| 类别 | 特征示例 | 数量 |
|------|----------|------|
| **面积类** | LotArea, GrLivArea, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GarageArea | 12+ |
| **质量评分** | OverallQual, OverallCond, ExterQual, BsmtQual, KitchenQual | 8+ |
| **房间数量** | BedroomAbvGr, TotRmsAbvGrd, FullBath, HalfBath, Fireplaces | 8+ |
| **时间特征** | YearBuilt, YearRemodAdd, GarageYrBlt, YrSold, MoSold | 5 |
| **地理区位** | Neighborhood, MSZoning, Condition1, Condition2 | 6 |
| **建筑特征** | BldgType, HouseStyle, RoofStyle, Foundation | 8+ |
| **外部设施** | WoodDeckSF, OpenPorchSF, PoolArea, PavedDrive | 6+ |

---

## 2. 特征工程策略

### 策略一：面积特征组合与转换
- **原理**：房价与使用面积强相关，需构建总面积、面积比例等特征
- **方法**：累加各功能区域面积，计算面积占比，生成交互特征

### 策略二：时间特征工程
- **原理**：房龄、翻新程度、销售时机影响房价
- **方法**：计算房龄、翻新后年数、销售季节等

### 策略三：质量综合评分
- **原理**：房屋整体质量是多维度评分的综合体现
- **方法**：加权/平均多项质量指标，生成综合质量分

### 策略四：功能密度特征
- **原理**：单位面积的 room/bath 数量反映空间利用效率
- **方法**：房间数/面积比率计算

### 策略五：类别特征编码
- **原理**：Neighborhood等类别对房价影响显著
- **方法**：目标编码、序数编码高基数类别特征

### 策略六：交互特征
- **原理**：特征组合产生非线性效应（如高质量+大面积）
- **方法**：质量×面积、车库容量×质量等

---

## 3. 要生成的新特征列表

### 📐 面积聚合特征 (8个)

| 新特征名 | 计算公式 | 业务含义 |
|----------|----------|----------|
| `TotalSF` | GrLivArea + TotalBsmtSF | 房屋总使用面积 |
| `TotalPorchSF` | WoodDeckSF + OpenPorchSF + ScreenPorch | 总门廊/甲板面积 |
| `TotalArea` | TotalSF + GarageArea | 建筑总面积（含车库） |
| `LotRatio` | GrLivArea / LotArea | 建筑密度 |
| `BsmtFinRatio` | BsmtFinSF1 / TotalBsmtSF | 地下室完工比例 |
| `2ndFloorRatio` | 2ndFlrSF / GrLivArea | 二层面积占比 |
| `OutdoorSF` | WoodDeckSF + OpenPorchSF + ScreenPorch + PoolArea | 户外设施总面积 |
| `AvgRoomSize` | GrLivArea / TotRmsAbvGrd | 平均房间大小 |

### 📅 时间特征 (5个)

| 新特征名 | 计算公式 | 业务含义 |
|----------|----------|----------|
| `HouseAge` | YrSold - YearBuilt | 房龄 |
| `RemodAge` | YrSold - YearRemodAdd | 翻新后年数 |
| `IsNew` | (YrSold == YearBuilt).astype(int) | 是否新房 |
| `HasRemod` | (YearRemodAdd != YearBuilt).astype(int) | 是否翻新过 |
| `SeasonSold` | MoSold.map({12,1,2:'Winter', 3,4,5:'Spring',...}) | 销售季节 |

### ⭐ 质量综合特征 (5个)

| 新特征名 | 计算方法 | 业务含义 |
|----------|----------|----------|
| `QualScore` | OverallQual × OverallCond | 综合质量分 |
| `ExterScore` | ExterQual编码 + ExterCond编码 | 外部质量评分 |
| `BsmtScore` | BsmtQual编码 + BsmtCond编码 + BsmtExposure编码 | 地下室质量评分 |
| `KitchenScore` | KitchenQual编码 × 2 | 厨房质量加权 |
| `GarageScore` | GarageQual编码 + GarageCond编码 + GarageFinish编码 | 车库综合质量 |

### 🏠 功能密度特征 (4个)

| 新特征名 | 计算公式 | 业务含义 |
|----------|----------|----------|
| `BathPerRoom` | (FullBath + 0.5×HalfBath) / TotRmsAbvGrd | 每房间浴室数 |
| `RoomDensity` | TotRmsAbvGrd / GrLivArea × 1000 | 房间密度 |
| `BedroomRatio` | BedroomAbvGr / TotRmsAbvGrd | 卧室占比 |
| `GarageEfficiency` | GarageCars / (GarageArea + 1) | 车库停车效率 |

### 🔧 交互特征 (6个)

| 新特征名 | 计算公式 | 业务含义 |
|----------|----------|----------|
| `Qual_LivArea` | OverallQual × GrLivArea | 质量加权面积 |
| `Qual_BsmtSF` | OverallQual × TotalBsmtSF | 质量加权地下室 |
| `Area_GarageCars` | GrLivArea × GarageCars | 面积与车库容量交互 |
| `Age_Qual` | HouseAge × (10 - OverallQual) | 房龄与质量交互 |
| `Neighborhood_PriceLevel` | Neighborhood的目标编码 | 区域价格水平 |
| `MSZoning_PriceLevel` | MSZoning的目标编码 |  zoning价格水平 |

### 🎯 其他派生特征 (4个)

| 新特征名 | 计算方法 | 业务含义 |
|----------|----------|----------|
| `HasPool` | (PoolArea > 0).astype(int) | 是否有泳池 |
| `Has2ndFloor` | (2ndFlrSF > 0).astype(int) | 是否有二层 |
| `HasBasement` | (TotalBsmtSF > 0).astype(int) | 是否有地下室 |
| `HasFireplace` | (Fireplaces > 0).astype(int) | 是否有壁炉 |
| `HasGarage` | (GarageArea > 0).astype(int) | 是否有车库 |

---

## 4. 预期效果

### 预期新增特征数量
- **原始特征**：68个（不含Id和SalePrice）
- **新生成特征**：32个
- **最终特征维度**：约100个

### 预期改进效果

| 评估维度 | 预期改进 |
|----------|----------|
| **模型性能** | R²提升 3-8%，RMSE降低 5-15% |
| **特征重要性** | TotalSF, QualScore, HouseAge预计成为Top5重要特征 |
| **非线性捕获** | 质量×面积交互项捕获高价值房产溢价效应 |
| **类别特征** | Neighborhood目标编码减少高维稀疏问题 |
| **时间趋势** | HouseAge帮助模型捕获折旧效应 |

### 关键成功因素
1. **TotalSF**（总使用面积）预计成为最强预测因子
2. **QualScore**（质量综合分）能更好量化"装修品质溢价"
3. **HouseAge + HasRemod** 组合有效区分新旧房价值
4. **交互特征**捕获高价值房产的非线性定价规律

### 后续验证建议
1. 使用SHAP值分析新特征的实际贡献度
2. 对比特征工程前后的交叉验证分数
3. 检查新特征的多重共线性（特别是面积相关特征）
4. 对高度 skewed 的特征进行对数变换（如LotArea, GrLivArea）