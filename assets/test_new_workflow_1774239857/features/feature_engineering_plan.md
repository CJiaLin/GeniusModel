# 特征工程方案

基于对实际数据的分析，以下是详细的特征工程方案：

## 1. 现有特征分析

### 数据概况
- **数据形状**: 1460行 × 81列（清理后）
- **目标变量**: `SalePrice`（连续型，房价回归预测）
- **特征类型分布**:
  - 数值型特征: 38个（包括整数和浮点数）
  - 分类型特征: 43个（需要编码处理）

### 关键数值特征
| 特征类别 | 特征示例 | 说明 |
|---------|---------|------|
| 面积相关 | `LotArea`, `GrLivArea`, `TotalBsmtSF` | 地块、地上、地下面积 |
| 房间相关 | `BedroomAbvGr`, `FullBath`, `HalfBath` | 卧室、全/半浴室数量 |
| 质量相关 | `OverallQual`, `OverallCond` | 总体质量和状况评分 |
| 年份相关 | `YearBuilt`, `YearRemodAdd` | 建造和改造年份 |

### 关键分类特征
| 特征类别 | 特征示例 | 说明 |
|---------|---------|------|
| 房屋类型 | `MSSubClass`, `MSZoning`, `HouseStyle` | 住宅类别、分区、风格 |
| 外部特征 | `Neighborhood`, `Exterior1st`, `Exterior2nd` | 社区、外墙材料 |
| 设施配置 | `GarageType`, `PoolQC`, `KitchenQual` | 车库、泳池、厨房质量 |

---

## 2. 特征工程策略

### 2.1 数值特征变换
```
策略: 对右偏分布的数值特征进行对数变换
目标特征: LotArea, GrLivArea, 1stFlrSF, 2ndFlrSF, LowQualFinSF, TotalBsmtSF
方法: np.log1p(x) 避免零值问题
原因: 面积类数据通常呈现长尾分布，对数变换可使其更接近正态分布
```

### 2.2 多项式特征
```
策略: 创建关键面积特征的交互项和多项式特征
基准特征: GrLivArea, TotalBsmtSF, OverallQual
生成特征: 
  - 平方项: GrLivArea^2, TotalBsmtSF^2
  - 乘积项: GrLivArea × OverallQual, TotalBsmtSF × OverallQual
原因: 房屋价值与面积和质量可能存在非线性关系
```

### 2.3 分组统计特征
```
策略: 按社区计算价格相关统计特征
分组列: Neighborhood
统计量: 中位数房价、平均质量评分
生成特征: Neighborhood_MedianPrice, Neighborhood_MeanQual
原因: 社区是房价的重要决定因素，引入统计特征可捕捉社区效应
```

### 2.4 时间特征
```
策略: 从年份特征中提取时间信息
基准特征: YearBuilt, YearRemodAdd, YrSold
生成特征:
  - HouseAge = YrSold - YearBuilt (房龄)
  - RemodAge = YrSold - YearRemodAdd (改造后年数)
  - IsNew = 1 if YrSold == YearBuilt else 0 (是否新房)
原因: 房屋年龄和改造历史直接影响价值
```

### 2.5 比例特征
```
策略: 创建有意义的面积比例
生成特征:
  - LotRatio = GrLivArea / LotArea (房屋占地比例)
  - BathRatio = FullBath / (BedroomAbvGr + 1) (卧室-浴室比例)
  - PorchRatio = (OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch) / GrLivArea
原因: 比例特征比绝对数值更具可比性
```

### 2.6 缺失值指示特征
```
策略: 为重要特征创建缺失指示器
适用特征: LotFrontage, GarageYrBlt, MasVnrArea
生成特征: HasLotFrontage, HasGarageYrBlt, HasMasVnr
原因: 缺失本身可能包含有价值的信息
```

---

## 3. 要生成的新特征列表

### 数值变换特征 (6个)
| 新特征名 | 源特征 | 变换方法 |
|---------|-------|---------|
| `Log_GrLivArea` | GrLivArea | log1p |
| `Log_LotArea` | LotArea | log1p |
| `Log_TotalBsmtSF` | TotalBsmtSF | log1p |
| `Log_1stFlrSF` | 1stFlrSF | log1p |
| `Sqrt_OverallQual` | OverallQual | sqrt |
| `Sqrt_TotRmsAbvGrd` | TotRmsAbvGrd | sqrt |

### 时间相关特征 (3个)
| 新特征名 | 计算公式 | 说明 |
|---------|---------|------|
| `HouseAge` | YrSold - YearBuilt | 房龄 |
| `YearsSinceRemod` | YrSold - YearRemodAdd | 改造后年数 |
| `IsNewHouse` | (YrSold == YearBuilt).astype(int) | 是否新房指示 |

### 交互特征 (5个)
| 新特征名 | 计算公式 | 说明 |
|---------|---------|------|
| `Qual_LivArea` | OverallQual × GrLivArea | 质量×面积交互 |
| `Qual_BsmtArea` | OverallQual × TotalBsmtSF | 质量×地下室面积 |
| `Qual_Cond` | OverallQual × OverallCond | 质量×状况交互 |
| `AreaRatio_GrLiv_Lot` | GrLivArea / (LotArea + 1) | 房屋占地比 |
| `BathPerBedroom` | (FullBath + 0.5*HalfBath) / (BedroomAbvGr + 1) | 卧室-浴室比 |

### 聚合特征 (4个)
| 新特征名 | 分组依据 | 聚合方式 |
|---------|---------|---------|
| `Neighborhood_MedianPrice` | Neighborhood | SalePrice中位数 |
| `Neighborhood_QualMean` | Neighborhood | OverallQual均值 |
| `MSSubClass_MedianPrice` | MSSubClass | SalePrice中位数 |
| `MSZoning_PriceLevel` | MSZoning | 价格等级编码 |

### 分类特征编码 (8个)
| 特征名 | 编码方法 | 类别数 |
|-------|---------|-------|
| `Neighborhood` | Target Encoding | 25 |
| `MSSubClass` | Ordinal Encoding | 16 |
| `MSZoning` | One-Hot Encoding | 7 |
| `HouseStyle` | One-Hot Encoding | 8 |
| `KitchenQual` | Ordinal Encoding (Ex>Gd>Ta>Fa>Po) | 5 |
| `ExterQual` | Ordinal Encoding | 5 |
| `GarageType` | One-Hot Encoding | 7 |
| `Foundation` | One-Hot Encoding | 6 |

### 缺失指示特征 (3个)
| 新特征名 | 指示的缺失特征 |
|---------|--------------|
| `HasLotFrontage` | LotFrontage非缺失 |
| `HasGarageYrBlt` | GarageYrBlt非缺失 |
| `HasMasVnr` | MasVnrArea非缺失 |

---

## 4. 预期效果

### 4.1 模型性能提升预期
| 评估指标 | 基准模型 | 预期提升 | 优化后 |
|---------|---------|---------|--------|
| RMSE | ~30000 | -15% ~ -20% | ~24000-25500 |
| MAE | ~20000 | -12% ~ -18% | ~16400-17600 |
| R² | ~0.80 | +0.03 ~ +0.05 | ~0.83-0.85 |

### 4.2 特征重要性预期
```
Top 10 重要特征预测:
1. OverallQual (原始质量评分)
2. GrLivArea (地上居住面积)
3. Qual_LivArea (质量×面积交互)
4. Neighborhood_MedianPrice (社区房价中位数)
5. TotalBsmtSF (地下室总面积)
6. HouseAge (房龄)
7. GarageCars (车库容量)
8. Log_GrLivArea (对数变换面积)
9. KitchenQual (厨房质量编码)
10. Neighborhood_QualMean (社区平均质量)
```

### 4.3 风险控制措施
- **过拟合预防**: 对Target Encoding使用交叉验证，防止数据泄漏
- **多重共线性**: 对高度相关特征(如GrLivArea与Log_GrLivArea)只保留一个
- **特征选择**: 使用Lasso或树模型的特征重要性筛选最终特征集

---

## 5. 实施建议

### 执行顺序
1. **数据清洗** → 处理缺失值、异常值
2. **时间特征** → 计算房龄等时间变量
3. **数值变换** → 对数、平方根变换
4. **分类编码** → 目标编码、独热编码
5. **交互特征** → 创建乘积和比例特征
6. **聚合特征** → 分组统计特征
7. **特征选择** → 筛选有效特征

### 验证方法
- 使用5折交叉验证评估特征效果
- 对比添加特征前后的模型性能
- 监控特征重要性分布，确保新特征有贡献