# 数据清洗方案

## 数据集信息

| 项目       | 详情                                          |
| :------- | :------------------------------------------ |
| **文件路径** | `/Users/cjialin/code/AutoMLByLLM/train.csv` |
| **数据规模** | 1,460 行 × 81 列                              |
| **目标变量** | `SalePrice`                                 |
| **特征类型** | 数值型: 38个 / 类别型: 43个                         |

***

## 1. 数据质量问题分析

### 1.1 缺失值问题

| 严重程度      | 特征名称          | 缺失率    | 处理策略         |
| :-------- | :------------ | :----- | :----------- |
| 🔴 **极高** | `PoolQC`      | 99.5%  | 删除列或标记为"无泳池" |
| 🔴 **极高** | `MiscFeature` | 96.3%  | 删除列          |
| 🔴 **极高** | `Alley`       | 93.8%  | 填充"无小巷"      |
| 🔴 **极高** | `Fence`       | 80.8%  | 填充"无围栏"      |
| 🟡 **中等** | `FireplaceQu` | 47.3%  | 填充"无壁炉"      |
| 🟡 **中等** | `LotFrontage` | 17.7%  | 按社区中位数填充     |
| 🟢 **较低** | `Garage*` 系列  | \~5.5% | 填充"无车库"      |
| 🟢 **较低** | `Bsmt*` 系列    | \~2.5% | 填充"无地下室"     |

### 1.2 异常值问题

| 特征          | 问题描述               | 检测方法           |
| :---------- | :----------------- | :------------- |
| `GrLivArea` | 2个极端高值（>4000 sqft） | 箱线图/IQR方法      |
| `SalePrice` | 少数高房价异常点           | 与GrLivArea联合分析 |

### 1.3 数据类型问题

| 特征           | 当前类型 | 建议类型 | 原因        |
| :----------- | :--- | :--- | :-------- |
| `MSSubClass` | 数值型  | 类别型  | 建筑类型标识码   |
| `MoSold`     | 数值型  | 类别型  | 月份应为分类变量  |
| `YrSold`     | 数值型  | 类别型  | 年份作为时间段分类 |

***

## 2. 清洗步骤

### Step 1: 环境准备与数据加载

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置显示选项
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# 加载数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')
print(f"原始数据形状: {df.shape}")
```

### Step 2: 处理数据类型转换

```python
# 转换MSSubClass为类别型
df['MSSubClass'] = df['MSSubClass'].astype(str)

# 时间相关特征转为类别型
df['MoSold'] = df['MoSold'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)

# 创建房屋年龄特征（特征工程）
df['HouseAge'] = df['YrSold'].astype(int) - df['YearBuilt']
df['RemodAge'] = df['YrSold'].astype(int) - df['YearRemodAdd']
```

### Step 3: 缺失值处理

```python
# 3.1 极高缺失率列处理
df.drop(['PoolQC', 'MiscFeature'], axis=1, inplace=True)

# 3.2 填充"无此设施"类别
none_cols = ['Alley', 'Fence', 'FireplaceQu', 
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for col in none_cols:
    df[col] = df[col].fillna('None')

# 3.3 数值型缺失处理
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# 3.4 其余少量缺失值填充
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
```

### Step 4: 异常值处理

```python
# 4.1 检测并处理GrLivArea异常值
Q1 = df['GrLivArea'].quantile(0.25)
Q3 = df['GrLivArea'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 标记异常值（建议删除GrLivArea > 4000的记录）
outliers = df[(df['GrLivArea'] > 4000)].index
print(f"检测到 {len(outliers)} 个异常值")
df = df.drop(outliers)

# 4.2 对目标变量进行对数变换（处理右偏）
df['SalePrice_Log'] = np.log1p(df['SalePrice'])
```

### Step 5: 特征工程

```python
# 5.1 合并车库相关特征
df['TotalGarageSF'] = df['GarageArea'] * df['GarageCars']

# 5.2 合并地下室面积
df['TotalBsmtSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['BsmtUnfSF']

# 5.3 总居住面积
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# 5.4 总浴室数
df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + \
                  df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']

# 5.5 整体质量评分
df['OverallScore'] = df['OverallQual'] * df['OverallCond']
```

### Step 6: 类别型变量编码

```python
from sklearn.preprocessing import LabelEncoder

# 6.1 有序类别编码（保持顺序关系）
ordinal_mappings = {
    'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
}

for col, mapping in ordinal_mappings.items():
    df[col] = df[col].map(mapping)

# 6.2 剩余类别型变量Label Encoding
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))
```

### Step 7: 数据验证与保存

```python
# 7.1 验证无缺失值
assert df.isnull().sum().sum() == 0, "仍存在缺失值！"

# 7.2 保存清洗后数据
df.to_csv('/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv', index=False)

print(f"清洗后数据形状: {df.shape}")
print("数据清洗完成！")
```

***

## 3. 预期效果

### 3.1 数据质量提升

| 指标        | 清洗前     | 清洗后      | 改善     |
| :-------- | :------ | :------- | :----- |
| **缺失值比例** | 最高99.5% | 0%       | ✅ 完全消除 |
| **异常值**   | 2个极端值   | 已处理      | ✅ 消除干扰 |
| **特征数量**  | 81列     | \~75列    | 精简高效   |
| **可用特征**  | 基础特征    | +10个衍生特征 | 增强表达能力 |

### 3.2 模型性能预期

| 方面        | 预期效果                  |
| :-------- | :-------------------- |
| **稳定性**   | 消除缺失值和异常值导致的预测不稳定     |
| **准确性**   | 有序编码保留信息，提升树模型和线性模型表现 |
| **特征丰富度** | 面积聚合、年龄特征增强房价预测能力     |
| **训练效率**  | 删除极高缺失列，减少噪声维度        |

### 3.3 清洗后数据特征

```
最终数据维度: [约1458行 × 约90列]
├── 数值特征: 原始38个 + 衍生特征
├── 类别特征: 全部编码完成
├── 目标变量: SalePrice (及Log变换版)
└── 数据质量: 无缺失、无异常、可直接建模
```

***

## 附录: 执行检查清单

- [ ] 代码可在目标环境运行
- [ ] 所有缺失值已处理
- [ ] 异常值已标记/删除
- [ ] 数据类型正确转换
- [ ] 特征工程逻辑验证
- [ ] 输出文件保存成功
- [ ] 与原始数据行数核对

