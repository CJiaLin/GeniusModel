# 房屋价格预测特征工程方案

## 1. 现有特征分析

### 1.1 数据概况
| 指标 | 数值 |
|------|------|
| 样本数量 | 1,460 |
| 特征总数 | 80 (不含ID和目标) |
| 数值型特征 | 35个 |
| 类别型特征 | 43个 |
| 目标变量 | SalePrice (回归任务) |

### 1.2 目标变量分析
- **SalePrice**: 连续型数值，范围 34,900 ~ 755,000
- **分布特征**: 右偏分布，需要进行对数转换
- **均值**: 180,921 | **中位数**: 163,000

### 1.3 数值型特征分类

| 类别 | 特征示例 | 处理策略 |
|------|----------|----------|
| **面积特征** | LotArea, GrLivArea, TotalBsmtSF, GarageArea | 标准化/归一化，创建面积比率 |
| **质量评分** | OverallQual, OverallCond | 有序数值，保持原值 |
| **计数特征** | BedroomAbvGr, FullBath, GarageCars | 离散数值，考虑分箱 |
| **年份特征** | YearBuilt, YearRemodAdd, GarageYrBlt | 转换为房龄，提取时间特征 |
| **缺失值较多** | LotFrontage, GarageYrBlt | 需要填充策略 |

### 1.4 类别型特征分类

| 类别 | 特征数量 | 示例 | 编码策略 |
|------|----------|------|----------|
| **有序等级** | 10+ | ExterQual, BsmtQual, KitchenQual | 标签编码 (Ex>Gd>TA>Fa>Po) |
| **无序类别** | 20+ | Neighborhood, HouseStyle, MSSoning | One-Hot / Target编码 |
| **二元特征** | 3 | Street, CentralAir, Alley | 二元编码 |
| **高基数** | 5+ | Neighborhood(25类) | Target编码 / 降维 |

### 1.5 缺失值情况
| 特征 | 缺失率 | 缺失处理建议 |
|------|--------|--------------|
| PoolQC | 99.5% | 填充"None"（无泳池） |
| MiscFeature | 96.3% | 填充"None" |
| Alley | 93.8% | 填充"None"（无巷道） |
| Fence | 80.8% | 填充"None" |
| FireplaceQu | 47.3% | 填充"None"（无壁炉） |
| LotFrontage | 17.7% | 按Neighborhood分组中位数填充 |
| GarageYrBlt | 5.5% | 填充YearBuilt（同建房年份） |

---

## 2. 特征工程策略

### 2.1 目标变量转换
```python
# 对数转换处理右偏分布
y_log = np.log1p(SalePrice)
```

### 2.2 数值型特征工程

#### 2.2.1 面积相关特征
- **总面积聚合**: 创建总居住面积、总外部面积
- **面积比率**: 各层面积占比、地下室完工比率
- **单位面积价格 proxy**: 用于特征交互

#### 2.2.2 时间特征提取
- **房龄**: 销售年份 - 建造年份
- **翻新后年限**: 销售年份 - 翻新年份
- **是否为新房**: 建造年份 = 销售年份

#### 2.2.3 质量-面积交互
- 质量评分 × 面积（如 OverallQual × GrLivArea）

### 2.3 类别型特征工程

#### 2.3.1 有序类别编码
将质量等级转换为数值：
```
Ex (Excellent) = 5
Gd (Good) = 4
TA (Typical/Average) = 3
Fa (Fair) = 2
Po (Poor) = 1
NA = 0
```

#### 2.3.2 高基数类别处理
- **Neighborhood**: Target Encoding 或分组编码
- **按目标变量均值分组**，减少维度

### 2.4 特征组合

| 组合类型 | 新特征名 | 计算方式 |
|----------|----------|----------|
| 总居住面积 | TotalSF | GrLivArea + TotalBsmtSF |
| 总浴室数 | TotalBathrooms | FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath |
| 总门廊面积 | TotalPorchSF | OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch |
| 房龄 | HouseAge | YrSold - YearBuilt |
| 翻新状态 | IsRemodeled | (YearRemodAdd != YearBuilt) |
| 质量面积交互 | QualSF | OverallQual × TotalSF |

---

## 3. 要生成的新特征列表

### 3.1 数值型新特征 (12个)

| # | 特征名 | 类型 | 计算公式/来源 | 预期作用 |
|---|--------|------|---------------|----------|
| 1 | `TotalSF` | 连续 | GrLivArea + TotalBsmtSF | 综合面积指标 |
| 2 | `TotalBathrooms` | 连续 | FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath | 总卫浴能力 |
| 3 | `TotalPorchSF` | 连续 | 所有门廊面积之和 | 户外空间 |
| 4 | `HouseAge` | 离散 | YrSold - YearBuilt | 房屋新旧程度 |
| 5 | `RemodAge` | 离散 | YrSold - YearRemodAdd | 翻新后时间 |
| 6 | `IsNew` | 二元 | HouseAge == 0 | 新房标识 |
| 7 | `HasBasement` | 二元 | TotalBsmtSF > 0 | 地下室有无 |
| 8 | `Has2ndFloor` | 二元 | 2ndFlrSF > 0 | 二层有无 |
| 9 | `HasGarage` | 二元 | GarageArea > 0 | 车库有无 |
| 10 | `HasPool` | 二元 | PoolArea > 0 | 泳池有无 |
| 11 | `HasFireplace` | 二元 | Fireplaces > 0 | 壁炉有无 |
| 12 | `QualSF` | 连续 | OverallQual × TotalSF | 质量-面积交互 |

### 3.2 比率型新特征 (8个)

| # | 特征名 | 计算公式 | 业务含义 |
|---|--------|----------|----------|
| 13 | `LotFrontageRatio` | LotFrontage / LotArea | 临街面占比 |
| 14 | `LivingAreaRatio` | GrLivArea / LotArea | 居住密度 |
| 15 | `BasementFinishRatio` | BsmtFinSF1 / TotalBsmtSF | 地下室完工率 |
| 16 | `2ndFloorRatio` | 2ndFlrSF / GrLivArea | 二层占比 |
| 17 | `GarageAreaRatio` | GarageArea / LotArea | 车库占地比 |
| 18 | `RoomsPerBedroom` | TotRmsAbvGrd / BedroomAbvGr | 房间-卧室比 |
| 19 | `AreaPerRoom` | GrLivArea / TotRmsAbvGrd | 平均房间面积 |

### 3.3 聚合统计特征 (6个)

| # | 特征名 | 描述 | 计算方法 |
|---|--------|------|----------|
| 20 | `Neighborhood_MedianPrice` | 社区房价中位数 | GroupBy Neighborhood |
| 21 | `Neighborhood_PriceStd` | 社区房价标准差 | GroupBy Neighborhood |
| 22 | `MSZoning_MedianPrice` | 区域类型房价中位数 | GroupBy MSZoning |
| 23 | `HouseStyle_MedianPrice` | 房型房价中位数 | GroupBy HouseStyle |

### 3.4 多项式特征 (可选，高阶)

| 特征组合 | 说明 |
|----------|------|
| `OverallQual^2` | 质量评分的非线性效应 |
| `GrLivArea^2` | 面积的非线性效应 |
| `log(TotalSF)` | 面积的对数转换 |

---

## 4. 预期效果

### 4.1 特征维度变化
| 阶段 | 特征数量 |
|------|----------|
| 原始数值特征 | 35 |
| 原始类别特征 (One-Hot后) | ~150 |
| 新增特征 | 23+ |
| **总计** | **~200+** |

### 4.2 各策略预期收益

| 策略 | 预期提升 | 原因 |
|------|----------|------|
| **对数转换目标** | RMSE ↓ 5-10% | 处理右偏，稳定方差 |
| **面积聚合特征** | R² ↑ 2-5% | 综合面积是房价核心决定因素 |
| **时间特征** | R² ↑ 1-3% | 房龄对折旧和价值的直接影响 |
| **质量-面积交互** | R² ↑ 2-4% | 高质量大房子的溢价效应 |
| **社区统计特征** | R² ↑ 3-5% | 引入位置信息的外部效应 |

### 4.3 模型适用性

| 模型类型 | 推荐特征处理 |
|----------|--------------|
| **线性回归/Ridge/Lasso** | 必须标准化，使用对数目标，多项式特征 |
| **树模型 (Random Forest/XGBoost)** | 类别标签编码即可，自动处理非线性 |
| **神经网络** | 标准化/归一化，Embedding处理类别 |

### 4.4 潜在风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 多重共线性 | 线性模型不稳定 | VIF检测，PCA降维 |
| 目标泄漏 | 过拟合 | 确保时间顺序正确分割 |
| 高维灾难 | 计算成本增加 | 特征选择（RFE/Boruta） |

---

## 5. 实施建议

### 阶段一：基础特征（高优先级）
1. 缺失值处理
2. 目标变量对数转换
3. 生成 TotalSF, HouseAge, TotalBathrooms

### 阶段二：交互特征（中优先级）
4. 质量-面积交互项
5. 社区统计聚合
6. 比率特征

### 阶段三：高级特征（低优先级，验证后）
7. 多项式特征
8. PCA降维
9. 特征选择

---

*本方案基于Ames Housing数据集结构制定，可根据实际建模效果迭代优化。*