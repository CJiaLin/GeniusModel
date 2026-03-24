# 房价预测数据清洗方案

## 一、数据概况

| 项目 | 详情 |
|------|------|
| 数据文件 | `/Users/cjialin/code/AutoMLByLLM/train.csv` |
| 数据形状 | (1460, 81) |
| 数值列 | 38 列 |
| 分类列 | 43 列 |
| 目标变量 | `SalePrice` |
| 评估指标 | RMSE |

---

## 二、数据质量问题总结

### 2.1 缺失值分布

| 严重程度 | 列名 | 缺失比例 | 建议处理 |
|---------|------|---------|---------|
| 🔴 极高(>80%) | PoolQC, MiscFeature, Alley, Fence, MasVnrType | 59%-99% | 删除列 |
| 🟡 中等(15%-50%) | FireplaceQu, LotFrontage | 17%-47% | 智能填充 |
| 🟢 较低(<10%) | Garage相关列(5类), Basement相关列(5类), MasVnrArea, Electrical | 0.07%-5.55% | 简单填充 |

### 2.2 异常值问题

- **31个数值列**存在统计异常值
- **BsmtFinSF2** 和 **EnclosedPorch** 存在极端零值比例（建议删除）
- **SalePrice** 目标变量存在4.18%异常值（需Winsorize处理）

### 2.3 数据类型问题

- 43个分类变量当前为 `object` 类型，建议转换为 `category` 类型优化内存

---

## 三、详细清洗步骤

### 步骤1：数据加载与备份

```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 加载数据
file_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
df = pd.read_csv(file_path)

# 保存原始数据备份
df_original = df.copy()
print(f"原始数据形状: {df.shape}")
```

### 步骤2：删除高缺失率列（>50%）

基于业务理解，这些列缺失率过高，提供的信息有限，且可能引入噪声：

```python
# 定义高缺失率列（根据报告）
high_missing_cols = [
    'PoolQC',      # 99.52% 缺失
    'MiscFeature', # 96.30% 缺失
    'Alley',       # 93.77% 缺失
    'Fence',       # 80.75% 缺失
    'MasVnrType'   # 59.73% 缺失
]

# 删除列
df = df.drop(columns=high_missing_cols)
print(f"删除高缺失率列后形状: {df.shape}")
```

### 步骤3：删除零方差/近零方差列

```python
# 根据报告，这些列异常值比例极高（主要为0值）
near_zero_variance_cols = ['BsmtFinSF2', 'EnclosedPorch']

df = df.drop(columns=near_zero_variance_cols)
print(f"删除近零方差列后形状: {df.shape}")
```

### 步骤4：缺失值处理

#### 4.1 分类变量缺失填充（NA表示"无此设施"）

```python
# 地下室相关列（NA表示无地下室）
bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_cols:
    df[col] = df[col].fillna('NoBsmt')

# 车库相关列（NA表示无车库）
garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cat_cols:
    df[col] = df[col].fillna('NoGarage')

# 壁炉质量（NA表示无壁炉）
df['FireplaceQu'] = df['FireplaceQu'].fillna('NoFireplace')

# 电力系统（仅1条缺失，用众数填充）
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
```

#### 4.2 数值变量缺失填充

```python
# MasVnrArea（砖石饰面面积）- 8条缺失，用0填充表示无饰面
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# GarageYrBlt（车库建造年份）- 81条缺失，用0表示无车库
# 注意：建模时可将0作为特殊类别处理，或与房屋建造年份关联
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

# LotFrontage（临街面长度）- 259条缺失，使用KNN插值
# 基于相似房产的LotArea和Neighborhood进行预测
imputer_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond']
imputer_df = df[imputer_cols].copy()

# 使用KNN插值（基于相关特征）
knn_imputer = KNNImputer(n_neighbors=5)
imputed_values = knn_imputer.fit_transform(imputer_df)
df['LotFrontage'] = imputed_values[:, 0]
```

### 步骤5：异常值处理（Winsorize）

对报告中的异常值列进行缩尾处理（保留5%-95%分位数范围）：

```python
def winsorize_series(series, lower_percentile=0.05, upper_percentile=0.95):
    """对序列进行缩尾处理"""
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower, upper)

# 需要Winsorize的列（基于报告中的异常值分析）
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
]

# 执行Winsorize（保留目标变量SalePrice用于建模，但处理极端异常）
for col in winsorize_cols:
    if col in df.columns:
        df[col] = winsorize_series(df[col])

print(f"完成{len(winsorize_cols)}列的异常值处理")
```

### 步骤6：数据类型优化

```python
# 将分类变量转换为category类型（节省内存，提升性能）
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# 排除已删除的列
categorical_cols = [col for col in categorical_cols if col in df.columns]

for col in categorical_cols:
    df[col] = df[col].astype('category')

# MSSubClass虽然是数值，但本质是分类变量（建筑类型代码）
df['MSSubClass'] = df['MSSubClass'].astype('category')

print(f"转换了{len(categorical_cols)}列为category类型")
```

### 步骤7：特征工程（房价预测专用）

```python
# 1. 总面积特征（地下室 + 地上生活面积）
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# 2. 总面积（不含地下室）
df['TotalAboveGroundSF'] = df['1stFlrSF'] + df['2ndFlrSF']

# 3. 房屋年龄和翻新年限
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# 4. 浴室总数（全浴室权重为1，半浴室权重为0.5）
df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + \
                  df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']

# 5. 门廊总面积
df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + \
                     df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']

# 6. 是否有车库（二值特征）
df['HasGarage'] = (df['GarageYrBlt'] > 0).astype(int)

# 7. 是否有地下室
df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)

# 8. 是否有壁炉
df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

# 9. 是否有泳池
df['HasPool'] = (df['PoolArea'] > 0).astype(int)

print("特征工程完成，新增9个特征")
```

### 步骤8：验证清洗结果

```python
# 8.1 检查缺失值
missing_after = df.isnull().sum()
missing_after = missing_after[missing_after > 0]
print(f"清洗后剩余缺失值列数: {len(missing_after)}")
if len(missing_after) > 0:
    print(missing_after)

# 8.2 检查数据形状
print(f"\n最终数据形状: {df.shape}")

# 8.3 检查数据类型分布
print(f"\n数据类型分布:")
print(df.dtypes.value_counts())

# 8.4 目标变量统计
print(f"\n目标变量SalePrice统计:")
print(df['SalePrice'].describe())
```

### 步骤9：保存清洗后的数据

```python
# 保存清洗后的数据
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"清洗后的数据已保存至: {output_path}")

# 保存清洗报告
report = {
    'original_shape': (1460, 81),
    'cleaned_shape': df.shape,
    'dropped_columns': high_missing_cols + near_zero_variance_cols,
    'imputed_columns': ['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'Electrical'] + 
                       bsmt_cols + garage_cat_cols + ['FireplaceQu'],
    'winsorized_columns': winsorize_cols,
    'engineered_features': ['TotalSF', 'TotalAboveGroundSF', 'HouseAge', 'RemodAge', 
                           'TotalBath', 'TotalPorchSF', 'HasGarage', 'HasBsmt', 
                           'HasFireplace', 'HasPool']
}

print("\n清洗摘要:")
for key, value in report.items():
    print(f"{key}: {value}")
```

---

## 四、清洗方案要点说明

### 4.1 针对房价预测的特殊考虑

| 问题 | 处理策略 | 理由 |
|------|---------|------|
| **高缺失率特征** | 删除PoolQC等5列 | 缺失率>60%，且与房价关联性有限 |
| **设施类NA值** | 填充为"NoXXX"类别 | NA表示"无此设施"，是有效信息 |
| **LotFrontage缺失** | KNN插值 | 与LotArea强相关，可用邻居信息推断 |
| **异常值处理** | Winsorize(5%-95%) | 保留极端房价信息但限制异常影响 |
| **时间特征** | 构造Age特征 | 房龄和翻新年限对房价影响显著 |

### 4.2 保留的重要信息

- **Id列**：保留用于最终提交
- **SalePrice异常值**：仅做Winsorize，不删除（RMSE对异常值敏感，但需保留分布信息）
- **Neighborhood**：保留作为关键区位特征
- **年份信息**：转换为年龄特征，保留时间趋势

### 4.3 新增特征解释

| 特征名 | 计算公式 | 业务含义 |
|-------|---------|---------|
| TotalSF | 地下室+地上面积 | 房屋总使用面积 |
| HouseAge | 销售年-建造年 | 房屋年龄 |
| TotalBath | 全浴+0.5×半浴 | 浴室等效数量 |
| HasGarage | GarageYrBlt>0 | 是否有车库（二值） |

---

## 五、预期效果

- **数据完整性**：缺失值从19列降至0列
- 特征维度：从81列优化至**约75列**（删除6列+新增9列）
- 数据质量：消除极端异常值影响，提升模型稳定性
- 建模就绪：所有特征数值化/类别化完毕，可直接输入模型

---

**执行说明**：按顺序执行步骤1-9即可完成全部清洗流程。建议在执行前备份原始数据，并检查每步的输出信息。