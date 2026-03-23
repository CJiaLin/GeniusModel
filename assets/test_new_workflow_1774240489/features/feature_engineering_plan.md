# 🏠 房价预测特征工程方案

## 1. 现有特征分析

### 1.1 数据概览
| 维度 | 详情 |
|------|------|
| 样本量 | 1,460 |
| 特征数 | 94 |
| 目标变量 | SalePrice |
| 任务类型 | 回归 |

### 1.2 特征分类

**已编码的质量特征（数值型）：**
- `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`
- `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`, `PoolQC`

**面积类特征：**
- `LotArea`, `LotFrontage`, `MasVnrArea`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`
- `GrLivArea`, `GarageArea`, `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`
- **已创建:** `TotalSF`, `TotalPorchSF`

**房间与设施：**
- `FullBath`, `HalfBath`, `BsmtFullBath`, `BsmtHalfBath`, `BedroomAbvGr`, `KitchenAbvGr`
- `TotRmsAbvGrd`, `Fireplaces`, `GarageCars`
- **已创建:** `TotalBath`

**时间特征：**
- `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`, `MoSold`, `YrSold`
- **已创建:** `HouseAge`, `RemodAge`, `IsNew`, `HasRemod`

**类别特征（需编码）：**
- `MSZoning`, `Street`, `Alley`, `LotShape`, `LandContour`, `LotConfig`
- `Neighborhood`, `Condition1`, `Condition2`, `BldgType`, `HouseStyle`
- `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`, `MasVnrType`
- `Foundation`, `BsmtFinType1`, `BsmtFinType2`, `Heating`, `CentralAir`
- `Electrical`, `GarageType`, `GarageFinish`, `PavedDrive`, `Fence`, `MiscFeature`
- `SaleType`, `SaleCondition`

**二值特征（已创建）：**
- `HasPool`, `Has2ndFloor`, `HasGarage`, `HasBasement`, `HasFireplace`, `HasFence`

### 1.3 缺失值分析
| 特征 | 缺失数 | 缺失率 | 处理策略 |
|------|--------|--------|----------|
| MiscFeature | 1,406 | 96.3% | 填充"No"或创建IsMiscFeature |
| Alley | 1,369 | 93.8% | 填充"NoAlley"或创建HasAlley |
| Fence | 1,179 | 80.8% | 填充"NoFence" |
| MasVnrType | 872 | 59.7% | 填充"None" |
| GarageType | 81 | 5.5% | 与GarageArea联合处理 |
| BsmtFinType1/2 | 37-38 | 2.6% | 填充"NoBsmt" |

---

## 2. 特征工程策略

### 2.1 策略一：交互特征创建
基于房价领域知识，创建关键交互特征：
- 面积 × 质量评分
- 房间密度（面积/房间数）
- 车库效率（面积/车位）

### 2.2 策略二：比率特征
- 各层面积占比
- 装修面积比例
- 土地利用率

### 2.3 策略三：聚合统计特征
- 质量综合评分
- 外部设施总分
- 功能完整性评分

### 2.4 策略四：非线性变换
- 面积特征的对数变换（处理右偏）
- 年龄特征的平方项（捕捉非线性折旧）

### 2.5 策略五：类别特征编码
- 高基数类别（Neighborhood, Exterior）: Target Encoding
- 低基数类别: One-Hot Encoding
- 有序类别: 保持现有Label Encoding

---

## 3. 要生成的新特征列表

### 3.1 面积交互特征（8个）

| 特征名 | 计算公式 | 业务含义 |
|--------|----------|----------|
| `LivingAreaPerRoom` | `GrLivArea / TotRmsAbvGrd` | 平均房间面积，衡量空间舒适度 |
| `LotUtilization` | `GrLivArea / LotArea` | 土地利用效率 |
| `BasementFinishRatio` | `BsmtFinSF1 / TotalBsmtSF` | 地下室装修比例 |
| `FirstFloorRatio` | `1stFlrSF / TotalSF` | 一层面积占比 |
| `SecondFloorRatio` | `2ndFlrSF / TotalSF` | 二层面积占比 |
| `GarageEfficiency` | `GarageArea / (GarageCars + 1)` | 单车位平均面积 |
| `OutdoorLivingSpace` | `WoodDeckSF + OpenPorchSF + EnclosedPorch` | 室外生活空间总面积 |
| `TotalFinishedSF` | `TotalSF - BsmtUnfSF - LowQualFinSF` | 总装修完成面积 |

### 3.2 质量综合特征（6个）

| 特征名 | 计算公式 | 业务含义 |
|--------|----------|----------|
| `OverallScore` | `OverallQual * OverallCond` | 房屋综合质量得分 |
| `ExteriorScore` | `ExterQual * ExterCond` | 外部质量综合评分 |
| `BsmtScore` | `BsmtQual * BsmtCond * BsmtExposure` | 地下室综合评分 |
| `GarageScore` | `GarageQual * GarageCond` | 车库质量评分 |
| `QualityIndex` | `(ExterQual + BsmtQual + KitchenQual + HeatingQC) / 4` | 平均质量指数 |
| `IsHighQuality` | `(OverallQual >= 8) & (ExterQual >= 4)` | 高品质房屋标识 |

### 3.3 时间相关特征（5个）

| 特征名 | 计算公式 | 业务含义 |
|--------|----------|----------|
| `SeasonSold` | `MoSold` 映射为季节 | 销售季节 |
| `YearsSinceRemod` | `YrSold - YearRemodAdd` | 距上次装修年数 |
| `GarageAge` | `YrSold - GarageYrBlt` | 车库年龄 |
| `IsVintage` | `HouseAge > 50` | 是否为老房（50年以上） |
| `IsRecentRemod` | `YearsSinceRemod <= 5` | 近期装修标识 |

### 3.4 功能密度特征（4个）

| 特征名 | 计算公式 | 业务含义 |
|--------|----------|----------|
| `BathPerBedroom` | `TotalBath / (BedroomAbvGr + 1)` | 卧室-浴室比 |
| `RoomDensity` | `TotRmsAbvGrd / GrLivArea * 100` | 房间密度 |
| `BedroomRatio` | `BedroomAbvGr / TotRmsAbvGrd` | 卧室占比 |
| `FunctionalScore` | `Functional` 加权 | 功能性评分 |

### 3.5 价值特征（4个）

| 特征名 | 计算公式 | 业务含义 |
|--------|----------|----------|
| `PricePerSF` | `SalePrice / GrLivArea` | 单位面积价格（仅用于分析）|
| `LuxuryIndex` | `PoolArea + MiscVal + (Fireplaces * 1000)` | 豪华设施指数 |
| `ValueAddFeatures` | `HasPool + HasFireplace + HasFence + (GarageCars > 1)` | 增值设施计数 |
| `NeighborhoodPriceLevel` | Neighborhood分组编码 | 社区价格等级 |

### 3.6 缺失值指示特征（4个）

| 特征名 | 说明 |
|--------|------|
| `HasAlley` | Alley是否缺失（0/1） |
| `HasFence` | Fence是否缺失（0/1） |
| `HasMiscFeature` | MiscFeature是否缺失（0/1） |
| `HasMasVnr` | MasVnrType是否缺失（0/1） |

### 3.7 非线性变换特征（6个）

| 特征名 | 变换方式 | 应用特征 |
|--------|----------|----------|
| `LogGrLivArea` | `log1p(GrLivArea)` | 居住面积对数 |
| `LogLotArea` | `log1p(LotArea)` | 土地面积对数 |
| `LogTotalSF` | `log1p(TotalSF)` | 总面积对数 |
| `SqrtHouseAge` | `sqrt(HouseAge)` | 房龄平方根 |
| `HouseAgeSq` | `HouseAge ** 2` | 房龄平方（折旧加速） |
| `LogSalePrice` | `log1p(SalePrice)` | 目标变量对数（建议） |

### 3.8 高级交互特征（5个）

| 特征名 | 计算公式 | 业务含义 |
|--------|----------|----------|
| `QualityArea` | `OverallQual * GrLivArea` | 质量-面积交互 |
| `QualAgeInteraction` | `OverallQual / (HouseAge + 1)` | 质量-年龄交互 |
| `BsmtLivingRatio` | `TotalBsmtSF / TotalSF` | 地下室生活空间占比 |
| `PorchRatio` | `TotalPorchSF / LotArea` | 门廊占地比 |
| `RemodImpact` | `HasRemod * OverallQual` | 装修与质量交互 |

---

## 4. 特征工程实施建议

### 4.1 实施顺序
```
步骤1: 处理缺失值 → 创建缺失值指示特征
步骤2: 创建面积交互特征 → 计算比率
步骤3: 创建质量综合评分 → 加权聚合
步骤4: 时间特征提取 → 季节、年龄计算
步骤5: 非线性变换 → 对数、平方根
步骤6: 类别特征编码 → Target/One-Hot Encoding
步骤7: 特征选择 → 删除冗余/高相关特征
```

### 4.2 编码策略

**Target Encoding（目标均值编码）：**
- `Neighborhood`, `Exterior1st`, `Exterior2nd`（高基数）
- 使用5折交叉验证避免过拟合

**One-Hot Encoding：**
- `MSZoning`, `LotShape`, `LandContour`, `LotConfig`
- `BldgType`, `HouseStyle`, `RoofStyle`, `Foundation`
- `CentralAir`, `PavedDrive`

**保留Label Encoding：**
- `OverallQual`, `OverallCond`（已有序）
- 所有已数值化的质量特征

### 4.3 需要删除的特征
- `Id`（标识符）
- `SalePrice`（目标变量，需分离）
- `Utilities`（单一值，无方差）
- `Street`（几乎单一值）
- 原始日期特征（保留派生特征后）

---

## 5. 预期效果

### 5.1 特征重要性预期
| 特征类别 | 预期重要性 | 原因 |
|----------|------------|------|
| Quality × Area 交互 | ⭐⭐⭐⭐⭐ | 质量与面积的乘积是价值核心 |
| OverallScore | ⭐⭐⭐⭐⭐ | 综合质量直接决定房价 |
| LogGrLivArea | ⭐⭐⭐⭐⭐ | 对数变换更符合价格非线性 |
| Neighborhood编码 | ⭐⭐⭐⭐⭐ | 地段是房价首要因素 |
| HouseAge + Age² | ⭐⭐⭐⭐☆ | 捕捉折旧非线性模式 |
| TotalSF | ⭐⭐⭐⭐☆ | 已有综合面积特征 |
| Garage相关特征 | ⭐⭐⭐☆☆ | 停车设施影响便利性 |
| 外部设施 | ⭐⭐⭐☆☆ | 附加值但非核心 |

### 5.2 模型性能提升预期

| 评估指标 | 基准模型 | 预期提升 | 说明 |
|----------|----------|----------|------|
| RMSE | ~50,000 | -15%~25% | 特征交互降低偏差 |
| MAE | ~35,000 | -15%~20% | 质量评分更精准 |
| R² | ~0.85 | +0.05~0.10 | 解释方差增加 |

### 5.3 关键洞察特征
1. **`QualityArea`**: 高质量大户型房屋价格溢价显著
2. **`HouseAgeSq`**: 老房子折旧加速，新房保值
3. **`LivingAreaPerRoom`**: 房间大小舒适度影响单价
4. **`QualAgeInteraction`**: 老房高质量=经典建筑，可能溢价

---

## 6. 最终特征数量预测

| 阶段 | 特征数 | 说明 |
|------|--------|------|
| 原始特征 | 94 | 含已创建特征 |
| 清理后 | 85 | 删除ID、常量特征 |
| 衍生数值特征 | +35 | 面积、质量、时间交互 |
| 缺失指示特征 | +4 | HasXXX特征 |
| 非线性变换 | +5 | 对数、平方变换 |
| 类别编码 | +50~80 | One-Hot/Target编码 |
| **最终预计** | **170~200** | 经过特征选择后约120~150 |

---

## 7. 注意事项

1. **数据泄漏避免**: 目标编码时严格使用训练集统计，验证集/测试集仅映射
2. **异常值处理**: `LotArea`, `GrLivArea`存在极端大值，建议先进行分位数截断或对数变换
3. **共线性检测**: `TotalSF`与`GrLivArea`高度相关，考虑PCA或选择其一
4. **领域知识**: 房价评估中"位置"最重要，确保`Neighborhood`特征充分挖掘

---

**总结**: 本方案基于实际数据的94个现有特征，通过面积交互、质量聚合、时间变换、类别编码四大策略，预计生成**35+个新数值特征**和**50+个编码特征**，最终形成170-200维的特征空间，可显著提升房价预测模型的准确性。