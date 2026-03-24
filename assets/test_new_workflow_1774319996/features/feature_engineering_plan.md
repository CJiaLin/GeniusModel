# 特征工程方案

## 数据集基本信息

| 属性 | 值 |
|------|-----|
| 样本数量 | 1,460 |
| 特征数量 | 80 (含目标列) |
| 目标列 | SalePrice |
| 任务类型 | 回归 (Regression) |

---

## 1. 现有特征分析

### 1.1 特征分类

| 类别 | 特征数量 | 特征列表 |
|------|---------|---------|
| 数值型特征 | 36 | Id, MSSubClass, LotFrontage, LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageYrBlt, GarageCars, GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, MiscVal, MoSold, YrSold |
| 类别型特征 | 43 | MSZoning, Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2, BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, Heating, HeatingQC, CentralAir, Electrical, KitchenQual, Functional, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PavedDrive, PoolQC, Fence, MiscFeature, SaleType, SaleCondition |

### 1.2 目标变量分析

**SalePrice (销售价格)**
- 这是一个典型的右偏分布数据
- 房价预测通常需要对其做对数变换以满足线性模型假设

---

## 2. 特征工程策略

### 2.1 缺失值处理策略

| 策略 | 适用特征 | 处理方式 |
|------|---------|---------|
| 零填充 | GarageYrBlt, MasVnrArea | 用 0 填充（表示无该车库/饰面） |
| 众数填充 | BsmtQual, GarageType 等 | 用最常见类别填充 |
| 新类别标记 | PoolQC, Fence, MiscFeature | 创建"无此设施"新类别 |
| 中位数填充 | LotFrontage | 按 Neighborhood 分组填充 |

### 2.2 特征变换策略

| 变换类型 | 目标特征 | 原因 |
|---------|---------|------|
| 对数变换 | SalePrice | 右偏分布，需要正态化 |
| 对数变换 | GrLivArea, TotalBsmtSF | 面积类特征通常右偏 |
| Box-Cox 变换 | 其他数值特征 | 自动寻找最佳变换 |

### 2.3 特征编码策略

| 编码方式 | 适用特征 | 原因 |
|---------|---------|------|
| 有序编码 | ExterQual, BsmtQual, KitchenQual | 质量等级有内在顺序 (Ex > Gd > TA > Fa > Po) |
| 目标编码 | Neighborhood | 类别数量较多 (25个)，与房价关系密切 |
| One-Hot 编码 | 其他低基数类别特征 | 无内在顺序关系 |

---

## 3. 要生成的新特征列表

### 3.1 面积组合特征

| 新特征名 | 计算公式 | 业务含义 |
|---------|---------|---------|
| `Total_SF` | GrLivArea + TotalBsmtSF | 房屋总使用面积 |
| `Total_BsmtFinSF` | BsmtFinSF1 + BsmtFinSF2 | 地下室总装修面积 |
| `Total_PorchSF` | OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch | 门廊总面积 |
| `Total_OutdoorSF` | WoodDeckSF + Total_PorchSF | 户外空间总面积 |
| `LivingRatio` | GrLivArea / LotArea | 建筑密度 |
| `2ndFloorRatio` | 2ndFlrSF / GrLivArea | 二层占比 |

### 3.2 浴室组合特征

| 新特征名 | 计算公式 | 业务含义 |
|---------|---------|---------|
| `Total_Bath` | FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath | 等效总浴室数 |
| `FullBathRatio` | FullBath / (FullBath + HalfBath + 0.001) | 全浴室占比 |

### 3.3 年龄与状况特征

| 新特征名 | 计算公式 | 业务含义 |
|---------|---------|---------|
| `HouseAge` | YrSold - YearBuilt | 房屋年龄 |
| `RemodAge` | YrSold - YearRemodAdd | 翻新距今时间 |
| `IsNew` | (YrSold == YearBuilt).astype(int) | 是否新建房 |
| `IsRemodeled` | (YearRemodAdd != YearBuilt).astype(int) | 是否翻新过 |
| `GarageAge` | YrSold - GarageYrBlt | 车库年龄 |

### 3.4 质量组合特征

| 新特征名 | 计算公式 | 业务含义 |
|---------|---------|---------|
| `OverallScore` | OverallQual * OverallCond | 综合质量评分 |
| `QualCondDiff` | OverallQual - OverallCond | 质量状况差异 |

### 3.5 房间密度特征

| 新特征名 | 计算公式 | 业务含义 |
|---------|---------|---------|
| `AvgRoomSize` | GrLivArea / (TotRmsAbvGrd + 1) | 平均房间大小 |
| `BedroomRatio` | BedroomAbvGr / (TotRmsAbvGrd + 1) | 卧室占比 |

### 3.6 设施计数特征

| 新特征名 | 计算公式 | 业务含义 |
|---------|---------|---------|
| `HasPool` | (PoolArea > 0).astype(int) | 是否有泳池 |
| `Has2ndFloor` | (2ndFlrSF > 0).astype(int) | 是否有二层 |
| `HasGarage` | (GarageArea > 0).astype(int) | 是否有车库 |
| `HasBsmt` | (TotalBsmtSF > 0).astype(int) | 是否有地下室 |
| `HasFireplace` | (Fireplaces > 0).astype(int) | 是否有壁炉 |
| `HasDeck` | (WoodDeckSF > 0).astype(int) | 是否有木甲板 |
| `Total_Facilities` | HasPool + Has2ndFloor + HasGarage + HasBsmt + HasFireplace + HasDeck | 设施总数 |

### 3.7 季节性特征

| 新特征名 | 计算公式 | 业务含义 |
|---------|---------|---------|
| `SeasonSold` | 将 MoSold 映射为季节 | 销售季节 (春/夏/秋/冬) |
| `IsSummer` | (MoSold in [6,7,8]).astype(int) | 是否夏季销售 |

---

## 4. 预期效果

### 4.1 特征重要性预期

| 特征类别 | 预期重要性 | 原因 |
|---------|-----------|------|
| 面积相关特征 (Total_SF, GrLivArea) | ⭐⭐⭐⭐⭐ | 房屋面积是房价最直接的决定因素 |
| 质量特征 (OverallScore, OverallQual) | ⭐⭐⭐⭐⭐ | 质量等级直接影响房屋价值 |
| 位置特征 (Neighborhood) | ⭐⭐⭐⭐ | 地段是房价的关键因素 |
| 年龄特征 (HouseAge, IsNew) | ⭐⭐⭐⭐ | 新房通常价格更高 |
| 浴室/卧室特征 | ⭐⭐⭐ | 影响居住舒适度 |
| 设施特征 | ⭐⭐⭐ | 增值设施提升房价 |

### 4.2 模型性能预期

| 指标 | 基准 (原始特征) | 预期提升 |
|------|----------------|---------|
| RMSE | ~0.15 (log尺度) | 15-25% 降低 |
| R² | ~0.85 | 0.88-0.92 |
| MAE | ~0.10 (log尺度) | 15-20% 降低 |

### 4.3 特征数量变化

| 阶段 | 特征数量 |
|------|---------|
| 原始特征 (去除Id) | 79 |
| 特征工程后 | 150+ |
| 选择后 (经特征选择) | 80-100 |

---

## 5. 执行建议

### 5.1 执行顺序

```
1. 数据清洗 → 缺失值处理
2. 特征变换 → 对数/Box-Cox变换
3. 特征创建 → 生成新特征
4. 特征编码 → 类别特征编码
5. 特征选择 → 去除冗余特征
6. 标准化 → 数值特征标准化
```

### 5.2 验证方法

- 使用交叉验证评估特征工程效果
- 对比特征工程前后的模型性能
- 分析特征重要性验证新特征的有效性

---

此特征工程方案专门针对房价预测任务设计，充分考虑了房地产市场的业务逻辑和数据特点，预期能显著提升模型预测性能。