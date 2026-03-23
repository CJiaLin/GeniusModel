# 特征工程方案：Ames Housing 房价预测

## 1. 现有特征分析

### 1.1 数据概况
| 项目 | 详情 |
|------|------|
| 样本量 | 1,460 |
| 特征数 | 80 (不含Id) |
| 目标变量 | SalePrice (连续型，回归任务) |
| 数据类型 | 数值型(36个) / 类别型(44个) |

### 1.2 特征分类

**面积相关特征 (连续数值)**
- `LotArea`, `LotFrontage` - 地块面积
- `1stFlrSF`, `2ndFlrSF`, `LowQualFinSF`, `GrLivArea` - 楼层面积
- `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF` - 地下室面积
- `GarageArea` - 车库面积
- `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch` - 室外空间
- `PoolArea`, `MasVnrArea` - 其他面积

**房间与设施 (离散数值)**
- `BedroomAbvGr`, `KitchenAbvGr`, `TotRmsAbvGrd` - 房间数
- `FullBath`, `HalfBath`, `BsmtFullBath`, `BsmtHalfBath` - 浴室数
- `Fireplaces`, `GarageCars` - 其他设施数量

**时间特征**
- `YearBuilt` - 建造年份
- `YearRemodAdd` - 改造年份
- `GarageYrBlt` - 车库建造年份
- `YrSold`, `MoSold` - 销售时间

**质量评级 (有序类别)**
- `OverallQual`, `OverallCond` - 整体质量(1-10)
- `ExterQual`, `ExterCond` - 外部质量
- `BsmtQual`, `BsmtCond` - 地下室质量
- `HeatingQC`, `KitchenQual` - 设备质量
- `FireplaceQu`, `GarageQual`, `GarageCond`, `PoolQC` - 其他质量

**地理位置 (类别)**
- `Neighborhood` - 社区(25个不同值)
- `MSZoning` - 区域划分
- `Condition1`, `Condition2` - 位置条件
- `LandContour`, `LandSlope`, `LotConfig` - 地形特征

**建筑特征 (类别)**
- `BldgType`, `HouseStyle` - 建筑类型
- `MSSubClass` - 住宅类型(应作为类别处理)
- `RoofStyle`, `RoofMatl` - 屋顶
- `Exterior1st`, `Exterior2nd` - 外墙材料
- `Foundation` - 基础类型

**缺失值严重特征**
| 特征 | 缺失数 | 缺失率 | 处理策略 |
|------|--------|--------|----------|
| `PoolQC` | 1,453 | 99.5% | 与`PoolArea`关联填充 |
| `MiscFeature` | 1,406 | 96.3% | 缺失=无该设施 |
| `Alley` | 1,369 | 93.8% | 缺失=无小巷通道 |
| `Fence` | 1,179 | 80.8% | 缺失=无围栏 |
| `FireplaceQu` | 690 | 47.3% | 与`Fireplaces`关联 |
| `LotFrontage` | 259 | 17.7% | 按`Neighborhood`中位数填充 |
| `Garage相关` | 81 | 5.5% | 缺失=无车库 |
| `Bsmt相关` | 37-38 | 2.6% | 缺失=无地下室 |
| `MasVnrType/Area` | 8-872 | 0.5-59.7% | 缺失=无饰面 |

---

## 2. 特征工程策略

### 2.1 第一阶段：数据清洗与缺失值处理

**策略A：缺失值指示器（Missing Indicator）**
对高缺失率特征创建二元指示变量，缺失本身可能是信号。

**策略B：领域知识填充**
- 车库相关：`GarageYrBlt`用`YearBuilt`填充（同时建造），类别特征填"None"
- 地下室相关：数值填0，类别填"None"
- 砌体饰面：`MasVnrArea`为0时`MasVnrType`="None"

### 2.2 第二阶段：特征聚合与构造

**策略C：总面积特征**
合并分散的面积指标，创建更稳健的特征。

**策略D：时间特征工程**
从年份构造房龄、改造年龄等衍生特征。

**策略E：质量聚合**
将多个质量评级整合为综合质量指标。

**策略F：房间密度与比例**
计算单位面积的房间数、浴室比例等。

### 2.3 第三阶段：特征编码与转换

**策略G：有序类别编码**
将质量等级（Ex>Gd>TA>Fa>Po）映射为数值(5-1)。

**策略H：目标编码**
对高基数类别（如`Neighborhood`）使用目标均值编码。

**策略I：对数变换**
对右偏的数值特征（面积、价格）进行对数变换。

---

## 3. 要生成的新特征列表

### 3.1 基础聚合特征 (15个)

| 新特征名 | 计算公式 | 说明 |
|---------|---------|------|
| `TotalSF` | `GrLivArea + TotalBsmtSF` | 房屋总使用面积 |
| `TotalPorchSF` | `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF` | 室外活动空间总面积 |
| `TotalBath` | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` | 等效总浴室数（半浴按0.5计） |
| `Has2ndFloor` | `1 if 2ndFlrSF > 0 else 0` | 是否有二楼 |
| `HasBasement` | `1 if TotalBsmtSF > 0 else 0` | 是否有地下室 |
| `HasGarage` | `1 if GarageArea > 0 else 0` | 是否有车库 |
| `HasPool` | `1 if PoolArea > 0 else 0` | 是否有泳池 |
| `HasFireplace` | `1 if Fireplaces > 0 else 0` | 是否有壁炉 |
| `HasFence` | `1 if Fence not null else 0` | 是否有围栏 |
| `HasAlley` | `1 if Alley not null else 0` | 是否有小巷通道 |
| `HasMiscFeature` | `1 if MiscFeature not null else 0` | 是否有其他设施 |
| `HasMasVnr` | `1 if MasVnrArea > 0 else 0` | 是否有砌体饰面 |
| `HasWoodDeck` | `1 if WoodDeckSF > 0 else 0` | 是否有木质甲板 |
| `HasOpenPorch` | `1 if OpenPorchSF > 0 else 0` | 是否有开放式门廊 |
| `HasEnclosedPorch` | `1 if EnclosedPorch > 0 else 0` | 是否有封闭式门廊 |

### 3.2 时间衍生特征 (6个)

| 新特征名 | 计算公式 | 说明 |
|---------|---------|------|
| `HouseAge` | `YrSold - YearBuilt` | 房龄（销售时） |
| `RemodAge` | `YrSold - YearRemodAdd` | 改造后年数 |
| `IsNew` | `1 if YrSold == YearBuilt else 0` | 是否新房 |
| `HasRemod` | `1 if YearRemodAdd != YearBuilt else 0` | 是否经过改造 |
| `GarageAge` | `YrSold - GarageYrBlt` | 车库年龄 |
| `SeasonSold` | 根据`MoSold`映射：1-3冬,4-6春,7-9夏,10-12秋 | 销售季节 |

### 3.3 面积比例与密度特征 (8个)

| 新特征名 | 计算公式 | 说明 |
|---------|---------|------|
| `LotFrontageRatio` | `LotFrontage / LotArea` | 临街面比例 |
| `BasementFinRatio` | `(BsmtFinSF1 + BsmtFinSF2) / TotalBsmtSF` | 地下室完成比例 |
| `LivingAreaRatio` | `GrLivArea / LotArea` | 建筑面积占比 |
| `2ndFloorRatio` | `2ndFlrSF / GrLivArea` | 二楼面积占比 |
| `LowQualRatio` | `LowQualFinSF / GrLivArea` | 低质量面积占比 |
| `RoomsPerSF` | `TotRmsAbvGrd / GrLivArea` | 每平方米房间数 |
| `BedroomRatio` | `BedroomAbvGr / TotRmsAbvGrd` | 卧室占比 |
| `GarageAreaPerCar` | `GarageArea / GarageCars` | 单车位面积（处理除零） |

### 3.4 质量综合特征 (4个)

| 新特征名 | 计算公式 | 说明 |
|---------|---------|------|
| `OverallQualityScore` | `(OverallQual + OverallCond) / 2` | 整体质量平均分 |
| `ExteriorScore` | 将`ExterQual`和`ExterCond`编码后平均 | 外部质量综合 |
| `KitchenScore` | `KitchenAbvGr * KitchenQual编码` | 厨房质量加权 |
| `TotalQualScore` | 多质量特征的加权聚合 | 综合质量指数 |

### 3.5 缺失值指示特征 (8个)

| 新特征名 | 说明 |
|---------|------|
| `LotFrontageMissing` | LotFrontage是否缺失 |
| `GarageMissing` | 车库信息是否缺失 |
| `BasementMissing` | 地下室信息是否缺失 |
| `FireplaceQuMissing` | 壁炉质量是否缺失 |
| `FenceMissing` | 围栏信息是否缺失 |
| `AlleyMissing` | 小巷信息是否缺失 |
| `PoolQCMissing` | 泳池质量是否缺失 |
| `MiscFeatureMissing` | 其他设施是否缺失 |

### 3.6 交互特征 (6个)

| 新特征名 | 计算公式 | 说明 |
|---------|---------|------|
| `QualLivingArea` | `OverallQual * GrLivArea` | 质量与面积的交互 |
| `QualBasement` | `BsmtQual编码 * TotalBsmtSF` | 地下室质量×面积 |
| `QualGarage` | `GarageQual编码 * GarageArea` | 车库质量×面积 |
| `QualExterArea` | `ExterQual编码 * GrLivArea` | 外部质量×面积 |
| `NeighborhoodQuality` | 按`Neighborhood`分组的`OverallQual`均值 | 社区平均质量 |
| `YearQuality` | `YearBuilt * OverallQual` | 年份与质量的交互 |

---

## 4. 预期效果

### 4.1 模型性能提升预期

| 指标 | 基线模型 | 预期提升 | 优化后模型 |
|------|---------|---------|-----------|
| RMSE | ~0.15 | -15%~-25% | ~0.11-0.13 |
| MAE | ~0.12 | -10%~-20% | ~0.09-0.11 |
| R² Score | ~0.85 | +3%~+8% | ~0.88-0.92 |

### 4.2 特征重要性预期

**高重要性特征（Top 10）**
1. `TotalSF` - 总使用面积（综合GrLivArea和地下室）
2. `OverallQual` - 整体质量（原始特征，极强预测力）
3. `QualLivingArea` - 质量与面积交互
4. `GarageArea` / `GarageCars` - 车库信息
5. `HouseAge` / `YearBuilt` - 房龄相关
6. `TotalBsmtSF` - 地下室面积
7. `Neighborhood` - 社区位置（目标编码后）
8. `TotalBath` - 等效浴室数
9. `1stFlrSF` - 一楼面积
10. `ExterQual` - 外部质量

### 4.3 各策略预期贡献

| 策略类别 | 贡献度 | 主要收益 |
|---------|-------|---------|
| 缺失值处理 | ⭐⭐⭐ | 保留完整信息，创建缺失指示信号 |
| 面积聚合 | ⭐⭐⭐⭐⭐ | 降低维度，创建更稳健的总面积指标 |
| 时间特征 | ⭐⭐⭐⭐ | 捕捉房龄效应，比原始年份更有解释力 |
| 质量聚合 | ⭐⭐⭐⭐ | 综合多个质量维度，减少冗余 |
| 类别编码 | ⭐⭐⭐⭐ | 将有序类别转为数值，目标编码高基类特征 |
| 特征交互 | ⭐⭐⭐ | 捕捉质量与面积的非线性关系 |

### 4.4 实施建议

**阶段1（必做）**：缺失值处理 + 基础聚合特征（TotalSF, TotalBath等）
**阶段2（推荐）**：时间特征 + 质量编码
**阶段3（进阶）**：交互特征 + 目标编码

**注意事项**：
- 避免过度工程化，总特征数建议控制在150以内
- 对创建的比例特征进行平滑处理（加1除法避免除零）
- 对目标编码使用交叉验证防止数据泄漏
- 最终对`SalePrice`进行对数变换（`log1p`）以处理右偏分布

---

**总结**：本方案基于Ames Housing数据的实际特征结构，通过面积聚合、时间衍生、质量综合、缺失值编码和特征交互等策略，预计可生成约50个新特征，显著提升房价预测模型的性能和稳定性。