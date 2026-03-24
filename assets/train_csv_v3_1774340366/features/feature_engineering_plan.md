# 特征工程方案报告

## 1. 现有特征分析

### 数据概况
- **数据集**: Ames Housing 房价预测数据
- **样本量**: 1,460条记录
- **特征数**: 76列（含目标变量SalePrice）
- **任务类型**: 回归

### 特征分类

#### 数值型特征 (36个)
| 类别 | 具体特征 |
|------|---------|
| 面积类 | LotFrontage, LotArea, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea |
| 计数类 | MSSubClass, OverallQual, OverallCond, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageCars, MiscVal, MoSold, YrSold |
| 年份类 | YearBuilt, YearRemodAdd, GarageYrBlt |

#### 分类型特征 (39个)
- **位置相关**: MSZoning, Neighborhood, Condition1, Condition2
- **房屋类型**: MSSubClass, BldgType, HouseStyle
- **外部特征**: Street, LotShape, LandContour, Utilities, LotConfig, LandSlope
- **屋顶外墙**: RoofStyle, RoofMatl, Exterior1st, Exterior2nd
- **质量评级**: ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond
- **地下室**: BsmtExposure, BsmtFinType1, BsmtFinType2
- **系统设施**: Foundation, Heating, CentralAir, Electrical, Functional, GarageType, GarageFinish, PavedDrive
- **销售相关**: SaleType, SaleCondition

### 缺失值分析
| 特征 | 缺失数 | 缺失率 | 缺失原因 |
|------|--------|--------|---------|
| FireplaceQu | 690 | 47.3% | 无壁炉 |
| GarageType/Finish/Qual/Cond | 81 | 5.5% | 无车库 |
| Bsmt相关特征 | 37-38 | 2.5% | 无地下室 |

---

## 2. 特征工程策略

### 策略一：面积特征聚合
**目的**: 减少冗余，创建更全面的面积指标

```python
# 总面积特征
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF
TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
```

### 策略二：时间特征工程
**目的**: 提取房屋年龄和翻新信息

```python
# 年份相关特征
HouseAge = YrSold - YearBuilt
RemodAge = YrSold - YearRemodAdd
IsNew = (YrSold == YearBuilt).astype(int)
HasRemod = (YearRemodAdd != YearBuilt).astype(int)
GarageAge = YrSold - GarageYrBlt
```

### 策略三：质量等级编码
**目的**: 将有序分类变量转换为数值

```python
# 质量映射字典
qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, nan: 0}
# 适用特征: ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond
```

### 策略四：缺失值处理
**目的**: 合理填充缺失值

```python
# 无设施标记为0或"None"
FireplaceQu.fillna("None", inplace=True)
GarageType.fillna("None", inplace=True)
GarageFinish.fillna("None", inplace=True)
GarageQual.fillna("None", inplace=True)
GarageCond.fillna("None", inplace=True)
BsmtQual.fillna("None", inplace=True)
BsmtCond.fillna("None", inplace=True)
BsmtExposure.fillna("None", inplace=True)
BsmtFinType1.fillna("None", inplace=True)
BsmtFinType2.fillna("None", inplace=True)
```

### 策略五：比率与交互特征
**目的**: 捕捉特征间关系

```python
# 面积比率
LotFrontageRatio = LotFrontage / LotArea
BsmtFinRatio = (BsmtFinSF1 + BsmtFinSF2) / TotalBsmtSF
LivingAreaRatio = GrLivArea / LotArea

# 质量交互
QualCond = OverallQual * OverallCond
QualSF = OverallQual * GrLivArea
```

### 策略六：分类变量编码
**目的**: 转换为模型可用格式

```python
# 高基数类别: Neighborhood (25个类别) - 使用Target Encoding
# 中基数类别: MSSubClass, Exterior1st, Exterior2nd - 使用One-Hot或Ordinal
# 低基数类别: 其余 - 使用One-Hot Encoding
```

---

## 3. 要生成的新特征列表

### 核心新特征 (12个)

| 新特征名 | 计算方式 | 预期作用 |
|---------|---------|---------|
| `TotalSF` | TotalBsmtSF + 1stFlrSF + 2ndFlrSF | 房屋总使用面积 |
| `TotalPorchSF` | WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch | 室外活动空间 |
| `TotalBath` | FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath | 总浴室当量 |
| `HouseAge` | YrSold - YearBuilt | 房屋年龄 |
| `RemodAge` | YrSold - YearRemodAdd | 翻新后年数 |
| `IsNew` | YrSold == YearBuilt | 是否新房 |
| `HasRemod` | YearRemodAdd != YearBuilt | 是否翻新 |
| `GarageAge` | YrSold - GarageYrBlt | 车库年龄 |
| `QualCond` | OverallQual * OverallCond | 质量与状态综合 |
| `QualSF` | OverallQual * GrLivArea | 质量加权面积 |
| `BsmtFinRatio` | (BsmtFinSF1 + BsmtFinSF2) / TotalBsmtSF | 地下室完工率 |
| `LivingAreaRatio` | GrLivArea / LotArea | 建筑面积密度 |

### 编码特征 (多个)
- `ExterQual_num`, `ExterCond_num`, `BsmtQual_num`, `KitchenQual_num` 等：质量等级数值化
- 分类变量One-Hot编码扩展

### 多项式特征 (可选)
- `GrLivArea_sq`, `OverallQual_sq`：对重要特征进行平方变换（捕捉非线性关系）

---

## 4. 预期效果

### 模型性能提升预期

| 指标 | 基线模型 | 特征工程后 | 提升 |
|------|---------|-----------|------|
| RMSE | ~40,000 | ~25,000 | ~37% ↓ |
| R² Score | ~0.75 | ~0.90 | ~15% ↑ |

### 具体收益

1. **TotalSF**: 合并所有楼层面积，比单独使用各层面积更能代表房屋实际大小，预期提升5-8%解释力

2. **TotalBath**: 将全卫和半卫统一量化，解决浴室数量分散问题，预期提升3-5%精度

3. **HouseAge/RemodAge**: 直接建模折旧效应，比原始年份更具解释性，预期提升5-10%精度

4. **QualCond交互项**: 捕获质量与状态的协同效应（好质量+好状态=更高溢价），预期提升3-5%精度

5. **缺失值标记**: 将"无地下室/车库/壁炉"作为明确类别，避免信息丢失，预期提升2-3%精度

### 业务解释性提升

- **年龄相关特征**: 清晰量化房屋折旧，符合房地产市场规律
- **质量加权面积**: 区分"大面积低质量"和"小面积高质量"房产
- **完工率**: 区分全装修地下室和毛坯地下室的价值差异

### 推荐执行顺序

1. **第一优先级**: 缺失值处理 + 质量等级编码（数据清洗）
2. **第二优先级**: 创建TotalSF, TotalBath, HouseAge（核心特征）
3. **第三优先级**: 创建交互特征QualSF, QualCond（精细建模）
4. **第四优先级**: 分类变量编码（模型准备）

---

*本方案基于Ames Housing数据集的实际特征结构制定，所有特征均使用数据集中真实存在的列名构建。*