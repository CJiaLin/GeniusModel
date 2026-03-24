# 特征工程方案报告

## 1. 现有特征分析

### 数据概况
| 指标 | 数值 |
|------|------|
| 数据规模 | 1,460 行 × 74 列 |
| 数值特征 | 36 个 |
| 类别特征 | 38 个 |
| 目标变量 | SalePrice (int64) |
| 重复样本 | 0 |

### 数值特征分布分析
| 特征类别 | 具体列名 | 特点 |
|---------|---------|------|
| **面积特征** | LotArea, GrLivArea, 1stFlrSF, 2ndFlrSF, TotalBsmtSF, GarageArea, WoodDeckSF, OpenPorchSF, ScreenPorch, PoolArea | GrLivArea均值1499±451，LotArea均值9682±3469 |
| **质量评分** | OverallQual(6.1±1.38), OverallCond(5.6±1.02) | OverallQual 1-10分，与房价强相关 |
| **时间特征** | YearBuilt(1971±30), YearRemodAdd(1984±21), GarageYrBlt(1977±26) | 房龄跨度138年(1872-2010) |
| **房间数量** | BedroomAbvGr(2.9), FullBath(1.6), TotRmsAbvGrd(6.5), Fireplaces(0.6), GarageCars(1.8) | KitchenAbvGr恒为1，LowQualFinSF/3SsnPorch/MiscVal全为0 |
| **销售时间** | MoSold(6.3), YrSold(2007.8) | 销售周期2006-2010年 |

### 类别特征分布分析
| 特征 | 唯一值 | 主要类别 |
|------|--------|---------|
| Neighborhood | - | 位置核心特征 |
| MSZoning | 5 | RL(1151), RM(218)为主 |
| HouseStyle/BldgType | - | 房屋类型 |
| ExterQual/ExterCond | - | 外部质量 |
| BsmtQual/BsmtCond | - | 地下室质量 |
| KitchenQual | - | 厨房质量 |
| GarageType/Finish | - | 车库特征 |
| Functional | - | 功能性评级 |

### 缺失值分析
| 特征 | 缺失率 | 说明 |
|------|--------|------|
| FireplaceQu | 47.26% | 无壁炉的样本 |
| GarageType/Finish/Qual/Cond | 5.55% | 无车库的样本 |
| 其余特征 | 0% | 数据清洗已完成 |

---

## 2. 特征工程策略

### 策略一：面积聚合与衍生特征
基于房屋各区域面积构建聚合指标，捕捉空间利用效率。

### 策略二：时间特征工程
利用年份数据构建房龄、翻新时长等时间衍生特征。

### 策略三：质量-面积交互特征
质量评分与面积的乘积，反映"高品质空间"的价值。

### 策略四：房间配置特征
构建浴室/卧室比例、房间密度等居住舒适度指标。

### 策略五：多项式与对数变换
对关键数值特征进行非线性变换，捕捉复杂关系。

### 策略六：类别特征编码
对高基数类别特征采用目标编码，对有序类别特征进行数值映射。

---

## 3. 要生成的新特征列表

### 3.1 面积聚合特征（8个）

| 新特征名 | 计算公式 | 预期效果 |
|---------|---------|---------|
| `TotalSF` | GrLivArea + TotalBsmtSF | 总使用面积，预期为最强预测因子 |
| `TotalPorchSF` | WoodDeckSF + OpenPorchSF + ScreenPorch | 户外休闲空间总面积 |
| `Has2ndFloor` | (2ndFlrSF > 0).astype(int) | 是否双层住宅 |
| `HasBasement` | (TotalBsmtSF > 0).astype(int) | 是否有地下室 |
| `HasGarage` | (GarageArea > 0).astype(int) | 是否有车库 |
| `HasFireplace` | (Fireplaces > 0).astype(int) | 是否有壁炉 |
| `HasPool` | (PoolArea > 0).astype(int) | 是否有泳池 |
| `HasPorch` | (TotalPorchSF > 0).astype(int) | 是否有门廊 |

### 3.2 面积比例特征（4个）

| 新特征名 | 计算公式 | 预期效果 |
|---------|---------|---------|
| `BasementRatio` | TotalBsmtSF / TotalSF | 地下室占比 |
| `2ndFloorRatio` | 2ndFlrSF / GrLivArea | 二层面积占比 |
| `GarageRatio` | GarageArea / LotArea | 车库占地比 |
| `LivingAreaRatio` | GrLivArea / LotArea | 建筑密度 |

### 3.3 时间衍生特征（5个）

| 新特征名 | 计算公式 | 预期效果 |
|---------|---------|---------|
| `HouseAge` | YrSold - YearBuilt | 房龄，折旧因子 |
| `RemodAge` | YrSold - YearRemodAdd | 翻新后年数 |
| `IsNew` | (YrSold == YearBuilt).astype(int) | 是否新房 |
| `HasRemod` | (YearRemodAdd > YearBuilt).astype(int) | 是否翻新过 |
| `GarageAge` | YrSold - GarageYrBlt | 车库年龄 |

### 3.4 质量交互特征（6个）

| 新特征名 | 计算公式 | 预期效果 |
|---------|---------|---------|
| `QualSF` | OverallQual * GrLivArea | 质量加权居住面积 |
| `QualTotalSF` | OverallQual * TotalSF | 质量加权总面积 |
| `QualCond` | OverallQual * OverallCond | 质量×状态综合评分 |
| `ExterScore` | ExterQual编码 × ExterCond编码 | 外部状态综合 |
| `BsmtScore` | BsmtQual编码 × BsmtCond编码 | 地下室状态综合 |
| `KitchenScore` | KitchenQual编码 × KitchenAbvGr | 厨房质量评分 |

### 3.5 房间配置特征（5个）

| 新特征名 | 计算公式 | 预期效果 |
|---------|---------|---------|
| `TotalBath` | FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath | 总浴室当量 |
| `BedroomRatio` | BedroomAbvGr / TotRmsAbvGrd | 卧室占比 |
| `RoomDensity` | TotRmsAbvGrd / GrLivArea | 房间密度 |
| `BathBedroomRatio` | TotalBath / (BedroomAbvGr + 1) | 浴卧比 |
| `FamilySize` | BedroomAbvGr + 2 | 假设家庭规模 |

### 3.6 多项式与对数特征（6个）

| 新特征名 | 变换方式 | 原特征 |
|---------|---------|--------|
| `GrLivAreaLog` | np.log1p(GrLivArea) | 对数变换消除右偏 |
| `LotAreaLog` | np.log1p(LotArea) | 对数变换 |
| `GrLivAreaSq` | GrLivArea ** 2 | 平方项捕捉非线性 |
| `OverallQualSq` | OverallQual ** 2 | 质量边际效应递减 |
| `HouseAgeSq` | HouseAge ** 2 | 折旧非线性 |
| `QualAreaInt` | OverallQual * GrLivAreaLog | 质量-面积对数交互 |

### 3.7 目标编码特征（2个）

| 新特征名 | 编码方式 | 原特征 |
|---------|---------|--------|
| `NeighborhoodPrice` | Neighborhood的SalePrice中位数 | 区域价格水平 |
| `MSSubClassPrice` | MSSubClass的SalePrice中位数 | 建筑类型价格水平 |

### 3.8 有序类别数值映射（6个）

| 特征 | 映射规则 |
|------|---------|
| ExterQual/ExterCond | Ex→5, Gd→4, TA→3, Fa→2, Po→1 |
| BsmtQual/BsmtCond | Ex→5, Gd→4, TA→3, Fa→2, Po→1, NA→0 |
| KitchenQual | Ex→5, Gd→4, TA→3, Fa→2, Po→1 |
| FireplaceQu | Ex→5, Gd→4, TA→3, Fa→2, Po→1, NA→0 |
| GarageQual/GarageCond | Ex→5, Gd→4, TA→3, Fa→2, Po→1, NA→0 |

---

## 4. 预期效果

### 4.1 维度扩展
- 原始特征：73个（含Id）
- 新增特征：约42个
- 最终维度：约115个

### 4.2 模型性能预期提升

| 改进维度 | 预期效果 |
|---------|---------|
| **非线性捕捉** | 多项式项和对数变换可捕捉房价与面积的非线性关系 |
| **交互效应** | 质量×面积交互项可区分"大但质量差"与"小但精致"的住宅 |
| **房龄折旧** | 时间特征可建模折旧效应和翻新溢价 |
| **区域价值** | Neighborhood目标编码直接注入区域价格水平信息 |
| **空间效率** | 比例特征可识别高密度/低密度住宅的价值差异 |

### 4.3 关键特征重要性预期

根据房屋定价理论，预期最重要的特征包括：
1. **TotalSF** - 总使用面积
2. **QualTotalSF / QualSF** - 质量加权面积
3. **NeighborhoodPrice** - 区域价格水平
4. **OverallQual** - 整体质量评级
5. **GrLivArea** - 地上居住面积
6. **HouseAge** - 房龄折旧
7. **GarageCars × GarageArea** - 车库价值

### 4.4 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 目标编码泄露 | 使用交叉验证进行目标编码 |
| 多重共线性 | VIF检测，相关系数>0.9的特征剔除 |
| 维度灾难 | 特征重要性筛选，保留Top 50-80特征 |

---

## 5. 执行优先级建议

| 优先级 | 特征组 | 理由 |
|-------|--------|------|
| P0 | 面积聚合、质量交互、时间特征 | 理论支撑强，预期收益高 |
| P1 | 目标编码、有序类别映射 | 类别特征信息提取 |
| P2 | 多项式变换、对数变换 | 非线性建模 |
| P3 | 比例特征、房间配置 | 细化空间利用信息 |

**推荐初始特征集**：原始特征 + P0/P1级别特征，约90-100个特征进行初步建模。