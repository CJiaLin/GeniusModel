```markdown
# 数据清洗方案

## 1. 方案概述

**数据文件路径**: `/Users/cjialin/code/AutoMLByLLM/train.csv`  
**原始数据形状**: (1460, 81)  
**目标**: 通过系统性的数据清洗，提升数据质量，为后续建模分析奠定基础

### 清洗策略总览
| 问题类型 | 处理策略 | 涉及列数 |
|---------|---------|---------|
| 高缺失率列 | 删除列（缺失率>50%） | 5列 |
| 中等缺失率列 | 智能填充（基于业务逻辑） | 6列 |
| 低缺失率列 | 简单填充（众数/中位数） | 8列 |
| 极端异常值 | Winsorize处理 | 21列 |
| 零方差/高异常列 | 删除列 | 2列 |
| 数据类型优化 | 转换为category类型 | 43列 |

---

## 2. 详细清洗步骤

### 步骤1: 数据加载与初步检查

```python
import pandas as pd
import numpy as np
from scipy import stats

# 加载原始数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 记录原始数据信息
original_shape = df.shape
print(f"原始数据形状: {original_shape}")
print(f"数值列数量: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"分类列数量: {df.select_dtypes(include=['object']).shape[1]}")
```

---

### 步骤2: 缺失值处理

#### 2.1 删除高缺失率列（缺失率>50%）
基于业务理解，这些列缺失率过高，填充会引入过多噪声，建议直接删除。

```python
# 定义高缺失率列
high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']

# 删除前记录信息
print("删除高缺失率列:")
for col in high_missing_cols:
    missing_pct = df[col].isnull().sum() / len(df) * 100
    print(f"  - {col}: {missing_pct:.2f}% 缺失")

# 执行删除
df = df.drop(columns=high_missing_cols)
print(f"删除后数据形状: {df.shape}")
```

#### 2.2 中等缺失率列的智能填充（5%-50%）
这些列具有业务逻辑关联，需要根据具体情况填充。

**A. FireplaceQu（壁炉质量）- 缺失率47.26%**
```python
# 假设：缺失表示无壁炉，填充为"None"或"NA"
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
```

**B. LotFrontage（临街距离）- 缺失率17.74%**
```python
# 使用同Neighborhoood组的中位数填充（假设地理位置相似的房屋临街距离相近）
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
# 如果仍有缺失，使用整体中位数
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
```

**C. Garage相关列（5.55%缺失）- GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond**
```python
# 假设：缺失表示无车库
garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cols:
    df[col] = df[col].fillna('None')

# GarageYrBlt（车库建造年份）- 无车库的填充为0或特定标记
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
```

#### 2.3 低缺失率列的简单填充（<5%）
```python
# 地下室相关列（约2.5%缺失）
bsmt_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
for col in bsmt_cols:
    df[col] = df[col].fillna('None')  # 假设缺失表示无地下室

# MasVnrArea（砌体饰面面积）- 0.55%缺失，填充0（表示无砌体饰面）
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# Electrical（电力系统）- 0.07%缺失，使用众数填充
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
```

---

### 步骤3: 异常值处理

#### 3.1 删除零方差/高异常值列
这些列异常值比例过高或缺乏变异，对建模无贡献。

```python
# 删除BsmtFinSF2和EnclosedPorch（异常值比例>10%且正常范围为0）
cols_to_drop_outlier = ['BsmtFinSF2', 'EnclosedPorch']
df = df.drop(columns=cols_to_drop_outlier)
print(f"删除异常值列后形状: {df.shape}")
```

#### 3.2 Winsorize处理（缩尾处理）
对数值型异常值进行上下限截断，保留5%-95%分位数范围内的值。

```python
def winsorize_column(df, column, lower_quantile=0.05, upper_quantile=0.95):
    """对指定列进行Winsorize处理"""
    lower_bound = df[column].quantile(lower_quantile)
    upper_bound = df[column].quantile(upper_quantile)
    
    original_count = df.shape[0]
    outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"{column}: 截断范围 [{lower_bound:.2f}, {upper_bound:.2f}], 处理异常值 {outliers_count} 个")
    return df

# 定义需要Winsorize的列
winsorize_cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 
    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
    'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr', 
    'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF', 
    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
]

print("执行Winsorize处理:")
for col in winsorize_cols:
    if col in df.columns:
        df = winsorize_column(df, col)
```

#### 3.3 保留特定异常值的列
以下列虽有异常值，但属于合理业务范围，予以保留：
- `OverallQual` (0.14%异常): 评分1-10的极端值可能是合理的
- `YearBuilt` (0.48%异常): 极早年份可能是历史建筑
- `BsmtFinSF1` (0.48%异常): 地下室面积极端值可能合理
- `2ndFlrSF` (0.14%异常): 二层面积为0表示单层建筑
- `BsmtFullBath` (0.07%异常): 地下室全浴室数量
- `Fireplaces` (0.34%异常): 壁炉数量
- `GarageCars` (0.34%异常): 车库容量
- `PoolArea` (0.48%异常): 泳池面积为0表示无泳池

---

### 步骤4: 数据类型优化

将分类变量从`object`类型转换为`category`类型，节省内存并提升建模效率。

```python
# 定义分类变量列表（基于数据质量报告）
categorical_cols = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType',
    'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond',
    'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
    'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'
]

# 注意：Alley, PoolQC, Fence, MiscFeature, MasVnrType 已在步骤2.1中删除

# 转换数据类型
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

print(f"数据类型转换完成，内存使用优化")
```

---

### 步骤5: 重复值检查与处理

```python
# 检查重复行（基于报告，原始数据无重复，但清洗后需再次确认）
duplicate_count = df.duplicated().sum()
print(f"重复行数量: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"删除重复行后形状: {df.shape}")
```

---

### 步骤6: 最终验证

```python
def validate_cleaning(df):
    """验证清洗结果"""
    report = {
        '最终数据形状': df.shape,
        '剩余缺失值总数': df.isnull().sum().sum(),
        '数值列数量': df.select_dtypes(include=[np.number]).shape[1],
        '分类列数量': df.select_dtypes(include=['category']).shape[1],
        '对象列数量': df.select_dtypes(include=['object']).shape[1],
        '内存使用(MB)': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # 检查每列的缺失情况
    missing_cols = df.columns[df.isnull().any()].tolist()
    report['仍有缺失值的列'] = missing_cols
    
    return report

validation_report = validate_cleaning(df)
print("\n" + "="*50)
print("数据清洗验证报告")
print("="*50)
for key, value in validation_report.items():
    print(f"{key}: {value}")

# 保存清洗后的数据
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n清洗后的数据已保存至: {output_path}")
```

---

## 3. 清洗效果预期

| 指标 | 清洗前 | 清洗后 | 改善 |
|-----|-------|-------|-----|
| 数据形状 | (1460, 81) | (1460, 73) | 删除8个低质量列 |
| 缺失值列数 | 19列 | 0列 | 完全填充 |
| 异常值比例 | ~21列有异常 | 控制在5%以内 | Winsorize处理 |
| 内存使用 | 较高 | 降低30-40% | Category类型优化 |
| 重复行 | 0 | 0 | 保持无重复 |

---

## 4. 注意事项

1. **业务理解**: 本方案假设缺失值表示"不存在"（如缺失PoolQC表示无泳池），填充为"None"或0。如业务逻辑不同，需调整填充策略。

2. **LotFrontage填充**: 当前使用Neighborhood分组中位数填充，如地理位置信息不准确，可考虑使用整体中位数或基于LotArea的回归填充。

3. **异常值处理**: Winsorize使用5%-95%分位数，可根据模型敏感度调整为1%-99%或3σ原则。

4. **目标变量SalePrice**: 如这是监督学习任务，建议对SalePrice进行对数变换（`np.log1p`）以处理右偏分布。

5. **特征工程机会**: 清洗后可考虑创建新特征，如：
   - `TotalSF` = 1stFlrSF + 2ndFlrSF + BsmtFinSF1
   - `HouseAge` = YrSold - YearBuilt
   - `RemodAge` = YrSold - YearRemodAdd
```