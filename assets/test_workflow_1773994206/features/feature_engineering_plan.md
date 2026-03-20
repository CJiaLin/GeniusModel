# 🏠 房价预测特征工程方案

## 1. 现有特征分析

### 数据概览
| 属性 | 值 |
|------|-----|
| 样本数 | 1,460 |
| 特征数 | 76 |
| 目标变量 | SalePrice (连续型) |
| 缺失值 | 0（已清洗） |

### 特征分类

| 类别 | 特征数量 | 典型特征 |
|------|---------|---------|
| **面积特征** | 18个 | LotArea, GrLivArea, GarageArea, TotalBsmtSF 等 |
| **时间特征** | 5个 | YearBuilt, YearRemodAdd, GarageYrBlt, YrSold, MoSold |
| **质量评分** | 2个 | OverallQual, OverallCond (1-10分) |
| **房间/浴室** | 8个 | BedroomAbvGr, FullBath, HalfBath, TotRmsAbvGrd 等 |
| **有序分类** | 12个 | ExterQual, KitchenQual, BsmtQual 等 (Ex>Gd>Ta>Fa>Po) |
| **名义分类** | 30个 | Neighborhood, MSZoning, HouseStyle 等 |

### 数据特点
- **长尾分布**：房价通常呈右偏分布，需对数变换
- **多维度评估**：房屋价值由位置、质量、面积、新旧程度共同决定
- **层次结构**：地下室→一层→二层→车库→外部设施

---

## 2. 特征工程策略

### 2.1 时间维度特征 ⏱️
```python
# 房屋年龄相关
HouseAge = YrSold - YearBuilt
RemodAge = YrSold - YearRemodAdd
GarageAge = YrSold - GarageYrBlt
IsNew = (YrSold == YearBuilt).astype(int)
```

### 2.2 面积聚合特征 📐
```python
# 总面积指标
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF
LotRatio = GrLivArea / LotArea  # 建筑密度

# 空间效率
AvgRoomSize = GrLivArea / (TotRmsAbvGrd + 1)
BedroomRatio = BedroomAbvGr / (TotRmsAbvGrd + 1)
```

### 2.3 浴室整合特征 🛁
```python
# 等效全浴室数（半浴室按0.5计算）
TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
BathPerBed = TotalBath / (BedroomAbvGr + 1)
```

### 2.4 质量交互特征 ⭐
```python
# 质量与面积交互
QualSF = OverallQual * GrLivArea
QualCond = OverallQual * OverallCond

# 地下室完成率
BsmtFinRatio = (BsmtFinSF1 + BsmtFinSF2) / (TotalBsmtSF + 1)
```

### 2.5 有序分类编码 📝
```python
# 质量等级映射字典
quality_map = {'ex': 5, 'gd': 4, 'ta': 3, 'fa': 2, 'po': 1, 'na': 0}

# 应用映射
ExterQualScore = ExterQual.map(quality_map)
KitchenQualScore = KitchenQual.map(quality_map)
BsmtQualScore = BsmtQual.map(quality_map)
```

### 2.6 类别聚合特征 🏘️
```python
# 按邻域聚合房价统计（训练集）
Neighborhood_MedianPrice = df.groupby('Neighborhood')['SalePrice'].transform('median')
Neighborhood_PriceRatio = SalePrice / Neighborhood_MedianPrice  # 相对价格水平
```

### 2.7 功能区存在性特征 🚗
```python
# 二进制特征
HasPool = (PoolArea > 0).astype(int)
HasGarage = (GarageArea > 0).astype(int)
HasBasement = (TotalBsmtSF > 0).astype(int)
HasFireplace = (Fireplaces > 0).astype(int)
Has2ndFloor = (2ndFlrSF > 0).astype(int)
```

---

## 3. 要生成的新特征列表

### 新增数值特征（12个）

| 特征名 | 计算公式 | 业务含义 |
|--------|---------|---------|
| `HouseAge` | YrSold - YearBuilt | 房龄 |
| `RemodAge` | YrSold - YearRemodAdd | 翻新后年数 |
| `GarageAge` | YrSold - GarageYrBlt | 车库年龄 |
| `IsNew` | YrSold == YearBuilt | 是否新房 |
| `TotalSF` | Bsmt + 1F + 2F | 总居住面积 |
| `TotalPorchSF` | 门廊总和 | 户外活动空间 |
| `LotRatio` | GrLivArea/LotArea | 土地利用率 |
| `AvgRoomSize` | GrLivArea/Rooms | 平均房间大小 |
| `TotalBath` | 全浴+0.5半浴 | 等效浴室数 |
| `QualSF` | Qual × GrLivArea | 质量加权面积 |
| `BsmtFinRatio` | 完成面积/总面积 | 地下室完工率 |
| `QualCond` | Qual × Cond | 质量状态综合 |

### 新增分类特征（8个）

| 特征名 | 类型 | 说明 |
|--------|------|------|
| `HasPool` | 二值 | 是否有游泳池 |
| `HasGarage` | 二值 | 是否有车库 |
| `HasBasement` | 二值 | 是否有地下室 |
| `HasFireplace` | 二值 | 是否有壁炉 |
| `Has2ndFloor` | 二值 | 是否有二层 |
| `ExterQualScore` | 有序(1-5) | 外部质量量化 |
| `KitchenQualScore` | 有序(1-5) | 厨房质量量化 |
| `BsmtQualScore` | 有序(1-5) | 地下室质量量化 |

### 新增交互特征（6个）

| 特征名 | 说明 |
|--------|------|
| `Neighborhood_Qual` | 邻域×质量交互 |
| `HouseStyle_Age` | 房型×房龄交互 |
| `MSZoning_LotRatio` | 区域类型×密度交互 |
| `GarageQual_Area` | 车库质量×面积交互 |
| `MoSold_YrSold` | 月份×年份（捕捉季节性趋势） |
| `QualCond_TotalSF` | 综合质量×总面积 |

---

## 4. 预期效果

### 4.1 性能提升预期

| 指标 | 基准模型 | 特征工程后 | 提升幅度 |
|------|---------|-----------|---------|
| **R² Score** | 0.82-0.85 | 0.88-0.92 | +5~8% |
| **RMSE** | 35,000-40,000 | 25,000-30,000 | -20~30% |
| **MAE** | 25,000-28,000 | 18,000-22,000 | -20~25% |

### 4.2 特征重要性预期

**高重要性特征（Top 10）**：
1. OverallQual / QualSF（质量是房价核心）
2. GrLivArea / TotalSF（面积决定基础价值）
3. Neighborhood（地段决定价格水平）
4. HouseAge / GarageAge（折旧因素）
5. TotalBath（功能性指标）
6. GarageCars / GarageArea（停车便利性）
7. BsmtFinSF1（额外可用空间）
8. YearRemodAdd（现代化程度）
9. FullBath（生活便利性）
10. LotArea（土地价值）

### 4.3 关键业务洞察

| 特征组 | 洞察 | 应用场景 |
|--------|------|---------|
| **时间特征** | 房龄与价格呈非线性关系（新房溢价、老房复古价值） | 定价策略、翻修决策 |
| **空间效率** | 平均房间面积比总房间数更重要 | 户型设计优化 |
| **质量交互** | 高质量小面积 > 低质量大面积 | 装修投资建议 |
| **邻域效应** | 位置对价格的影响呈区域性聚集 | 区域市场分析 |

### 4.4 风险与注意事项

⚠️ **潜在问题**：
1. **多重共线性**：TotalSF与GrLivArea高度相关，建议使用PCA或剔除
2. **数据泄露**：Neighborhood聚合统计需确保仅使用训练集计算
3. **异常值敏感**：LotRatio可能出现极端值（超大地块或小房大院子），需截断处理
4. **类别不平衡**：部分Neighborhood样本量小，聚合统计可能不稳定

---

## 5. 实施建议

### 执行顺序
```
Step 1: 时间特征 → Step 2: 面积聚合 → Step 3: 浴室整合 
    ↓
Step 4: 质量编码 → Step 5: 二值特征 → Step 6: 交互特征
    ↓
Step 7: 验证相关性 → Step 8: 剔除冗余 → Step 9: 标准化/归一化
```

### 代码框架
```python
# 伪代码示意
def feature_engineering(df):
    # 1. 时间特征
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    # 2. 面积聚合
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # 3. 浴室整合
    df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + \
                      df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    
    # 4. 质量编码
    quality_map = {'ex':5, 'gd':4, 'ta':3, 'fa':2, 'po':1}
    for col in ['ExterQual', 'KitchenQual', 'BsmtQual']:
        df[f'{col}Score'] = df[col].str.lower().map(quality_map)
    
    # 5. 二值特征
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    # ... 其他二值特征
    
    # 6. 交互特征
    df['QualSF'] = df['OverallQual'] * df['GrLivArea']
    
    return df
```

此方案预计可将特征维度从**76扩展至约100+**，通过更丰富的特征表示显著提升模型对房价内在规律的捕捉能力。