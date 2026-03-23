# 特征工程方案报告

## 1. 现有特征分析

### 1.1 数据概览
| 指标 | 数值 |
|------|------|
| 样本数量 | 1,460 |
| 特征数量 | 80 (不含Id和SalePrice) |
| 目标变量 | SalePrice (回归任务) |

### 1.2 特征分类

**数值特征 (36个):**
- `MSSubClass`, `LotFrontage`, `LotArea`, `OverallQual`, `OverallCond`
- `YearBuilt`, `YearRemodAdd`, `MasVnrArea`
- `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`
- `1stFlrSF`, `2ndFlrSF`, `LowQualFinSF`, `GrLivArea`
- `BsmtFullBath`, `BsmtHalfBath`, `FullBath`, `HalfBath`
- `BedroomAbvGr`, `KitchenAbvGr`, `TotRmsAbvGrd`, `Fireplaces`
- `GarageYrBlt`, `GarageCars`, `GarageArea`
- `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch`
- `PoolArea`, `MiscVal`, `MoSold`, `YrSold`

**类别特征 (43个):**
- `MSZoning`, `Street`, `Alley`, `LotShape`, `LandContour`, `Utilities`
- `LotConfig`, `LandSlope`, `Neighborhood`, `Condition1`, `Condition2`
- `BldgType`, `HouseStyle`, `RoofStyle`, `RoofMatl`
- `Exterior1st`, `Exterior2nd`, `MasVnrType`
- `ExterQual`, `ExterCond`, `Foundation`
- `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`
- `Heating`, `HeatingQC`, `CentralAir`, `Electrical`, `KitchenQual`
- `Functional`, `FireplaceQu`, `GarageType`, `GarageFinish`
- `GarageQual`, `GarageCond`, `PavedDrive`, `PoolQC`, `Fence`
- `MiscFeature`, `SaleType`, `SaleCondition`

### 1.3 缺失值分析
| 特征 | 缺失数量 | 缺失比例 | 处理建议 |
|------|---------|---------|---------|
| `Alley` | 1,369 | 93.8% | 填充为"None" |
| `MasVnrType` | 872 | 59.7% | 填充为"None" |
| `BsmtQual` | 37 | 2.5% | 填充为"NoBasement" |
| `BsmtCond` | 37 | 2.5% | 填充为"NoBasement" |
| `BsmtExposure` | 38 | 2.6% | 填充为"NoBasement" |
| `BsmtFinType1` | 37 | 2.5% | 填充为"NoBasement" |
| `BsmtFinType2` | 38 | 2.6% | 填充为"NoBasement" |

---

## 2. 特征工程策略

### 2.1 缺失值处理策略
```python
# 缺失值映射策略
missing_fill_strategy = {
    # 类别特征 - 无该设施
    'Alley': 'NoAlley',
    'MasVnrType': 'None',
    'BsmtQual': 'NoBasement',
    'BsmtCond': 'NoBasement',
    'BsmtExposure': 'NoBasement',
    'BsmtFinType1': 'NoBasement',
    'BsmtFinType2': 'NoBasement',
    'FireplaceQu': 'NoFireplace',
    'GarageType': 'NoGarage',
    'GarageFinish': 'NoGarage',
    'GarageQual': 'NoGarage',
    'GarageCond': 'NoGarage',
    'PoolQC': 'NoPool',
    'Fence': 'NoFence',
    'MiscFeature': 'None',
    
    # 数值特征 - 0或众数
    'MasVnrArea': 0,
    'GarageYrBlt': lambda x: x['YearBuilt']  # 用建筑年份填充
}
```

### 2.2 特征转换策略

**A. 对数变换 (针对右偏分布)**
- `SalePrice` (目标变量)
- `LotArea`, `GrLivArea`, `TotalBsmtSF`
- `1stFlrSF`, `2ndFlrSF`

**B. 有序类别编码**
```python
# 质量等级映射
quality_map = {
    'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,
    'NoBasement': 0, 'NoFireplace': 0, 'NoPool': 0, 'NoGarage': 0, 'None': 0
}
# 适用特征: ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond, PoolQC
```

---

## 3. 要生成的新特征列表

### 3.1 面积聚合特征 (8个)

| 新特征名 | 计算公式 | 业务含义 |
|---------|---------|---------|
| `TotalSF` | `TotalBsmtSF + GrLivArea` | 房屋总使用面积 |
| `TotalPorchSF` | `WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch` | 总门廊/露台面积 |
| `TotalBath` | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` | 等效总浴室数 |
| `TotalFinSF` | `BsmtFinSF1 + BsmtFinSF2` | 地下室完工面积 |
| `Has2ndFloor` | `2ndFlrSF > 0` | 是否有二楼 |
| `HasBasement` | `TotalBsmtSF > 0` | 是否有地下室 |
| `HasGarage` | `GarageArea > 0` | 是否有车库 |
| `HasPool` | `PoolArea > 0` | 是否有泳池 |
| `HasFireplace` | `Fireplaces > 0` | 是否有壁炉 |

### 3.2 时间衍生特征 (6个)

| 新特征名 | 计算公式 | 业务含义 |
|---------|---------|---------|
| `HouseAge` | `YrSold - YearBuilt` | 房屋年龄 |
| `RemodAge` | `YrSold - YearRemodAdd` | 改造后年数 |
| `IsNew` | `YrSold == YearBuilt` | 是否新房 |
| `GarageAge` | `YrSold - GarageYrBlt` | 车库年龄 |
| `YearsSinceRemod` | `YearRemodAdd - YearBuilt` | 建筑到改造间隔 |
| `SeasonSold` | `MoSold` 映射为季节 | 销售季节 |

### 3.3 质量综合特征 (4个)

| 新特征名 | 计算公式 | 业务含义 |
|---------|---------|---------|
| `OverallScore` | `OverallQual * OverallCond` | 整体质量综合得分 |
| `ExterScore` | `ExterQual(编码) * ExterCond(编码)` | 外部质量得分 |
| `KitchenScore` | `KitchenAbvGr * KitchenQual(编码)` | 厨房质量得分 |
| `BsmtScore` | `BsmtQual(编码) * BsmtCond(编码)` | 地下室质量得分 |

### 3.4 面积比例特征 (5个)

| 新特征名 | 计算公式 | 业务含义 |
|---------|---------|---------|
| `BsmtFinRatio` | `TotalFinSF / TotalBsmtSF` | 地下室完工比例 |
| `2ndFloorRatio` | `2ndFlrSF / GrLivArea` | 二楼面积占比 |
| `LowQualRatio` | `LowQualFinSF / GrLivArea` | 低质量面积占比 |
| `LotFrontageRatio` | `LotFrontage / LotArea` | 临街面比例 |
| `GarageAreaRatio` | `GarageArea / LotArea` | 车库占地比例 |

### 3.5 类别特征编码 (重点)

**高基数类别特征 (目标编码):**
- `Neighborhood` (25个类别)
- `Exterior1st` (15个类别)
- `Exterior2nd` (16个类别)

**二元特征提取:**
- `Is1Story`: `HouseStyle == '1Story'`
- `Is2Story`: `HouseStyle == '2Story'`
- `HasCentralAir`: `CentralAir == 'Y'`
- `IsPaved`: `Street == 'Pave'`

### 3.6 交互特征 (4个)

| 新特征名 | 计算公式 | 业务含义 |
|---------|---------|---------|
| `QualArea` | `OverallQual * GrLivArea` | 质量与面积交互 |
| `QualLot` | `OverallQual * LotArea` | 质量与地块交互 |
| `YearBuiltQual` | `YearBuilt * OverallQual` | 年份与质量交互 |
| `BsmtQualArea` | `BsmtQual(编码) * TotalBsmtSF` | 地下室质量与面积交互 |

---

## 4. 预期效果

### 4.1 特征维度变化
- **原始特征**: 80个
- **新增特征**: 约25-30个
- **最终特征**: 约105-110个（编码后可能更多）

### 4.2 预期收益

| 方面 | 预期效果 |
|------|---------|
| **模型性能** | RMSE降低15-25%，R²提升5-10% |
| **非线性捕捉** | 通过交互特征和多项式特征捕捉非线性关系 |
| **领域知识** | 面积聚合和质量综合特征反映房屋价值核心因素 |
| **时间趋势** | 时间特征帮助模型识别折旧和装修价值 |

### 4.3 关键成功因素

1. **面积特征**: `TotalSF`, `TotalPorchSF`, `QualArea` 预计是最重要的预测因子
2. **质量特征**: `OverallScore`, `ExterScore` 对价格有强解释力
3. **时间特征**: `HouseAge` 帮助模型理解折旧效应
4. **类别编码**: `Neighborhood`的目标编码能捕捉地理位置溢价

### 4.4 验证计划

```python
# 特征重要性验证
- 使用随机森林/GBDT输出特征重要性
- 验证Top 10特征是否包含新生成的聚合特征

# 相关性分析
- 检查新特征与SalePrice的相关系数
- 剔除与现有特征高度冗余(r>0.95)的特征

# 交叉验证
- 5折交叉验证对比特征工程前后的模型性能
- 监控过拟合情况（特别是目标编码特征）
```

---

## 5. 实施建议

### 阶段1: 数据清洗 (优先级: 高)
- 执行缺失值填充策略
- 处理异常值（如LotArea极端值）

### 阶段2: 基础特征 (优先级: 高)
- 生成面积聚合特征 (`TotalSF`, `TotalBath`等)
- 时间衍生特征 (`HouseAge`, `IsNew`)

### 阶段3: 高级特征 (优先级: 中)
- 质量综合评分
- 交互特征

### 阶段4: 编码优化 (优先级: 中)
- 有序类别标签编码
- 高基数特征目标编码（需防止过拟合）

此方案基于实际数据的特征分布和业务含义，预计能显著提升房价预测模型的准确性。