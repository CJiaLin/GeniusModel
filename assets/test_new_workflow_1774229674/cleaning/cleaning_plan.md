# 数据清洗方案

## 一、数据质量问题分析

### 1. 数据概览
| 指标 | 数值 |
|------|------|
| 数据规模 | 891 行 × 12 列 |
| 目标变量 | Survived（生存状态） |
| 任务类型 | 二分类问题 |

### 2. 数据质量问题清单

| 问题类型 | 涉及列 | 严重程度 | 详情 |
|---------|--------|---------|------|
| 缺失值 | Age | ⚠️ 中等 | 177 个缺失（19.87%），需要填充 |
| 缺失值 | Cabin | 🔴 严重 | 687 个缺失（77.10%），缺失率过高 |
| 缺失值 | Embarked | 🟢 轻微 | 2 个缺失（0.22%），可直接删除或填充 |
| 异常值 | Fare | ⚠️ 中等 | 最小值为 0，可能存在免费票或数据错误 |
| 格式问题 | Ticket | 🟡 提示 | 格式不统一（字母+数字混合），可考虑特征提取 |
| 格式问题 | Name | 🟡 提示 | 包含称谓信息，可提取为单独特征 |

---

## 二、详细清洗步骤

### 步骤 1: 基础数据预处理

```python
import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 设置 PassengerId 为索引
df.set_index('PassengerId', inplace=True)

# 查看基础信息
print(f"数据维度: {df.shape}")
print(f"\n列名: {df.columns.tolist()}")
```

### 步骤 2: 缺失值处理

#### 2.1 Embarked 列（轻微缺失）
```python
# 查看缺失值行
print(df[df['Embarked'].isnull()])

# 方案: 使用众数填充（S 港最常见）
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

#### 2.2 Age 列（中等缺失）
```python
# 方案: 按 Pclass 和 Sex 分组，使用中位数填充
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
    lambda x: x.fillna(x.median())
)

# 如果仍有缺失，使用整体中位数填充
df['Age'].fillna(df['Age'].median(), inplace=True)
```

#### 2.3 Cabin 列（严重缺失）
```python
# 方案: 提取舱位等级字母，缺失值标记为 'Unknown'
df['Cabin_Level'] = df['Cabin'].str[0].fillna('Unknown')

# 原 Cabin 列删除（缺失率过高，信息不足）
df.drop('Cabin', axis=1, inplace=True)
```

### 步骤 3: 异常值处理

#### 3.1 Fare 列
```python
# 查看 Fare 为 0 的记录
print(f"Fare 为 0 的记录数: {(df['Fare'] == 0).sum()}")

# 方案: 用 Pclass 分组的中位数替换 0 值
df.loc[df['Fare'] == 0, 'Fare'] = np.nan
df['Fare'] = df.groupby('Pclass')['Fare'].transform(
    lambda x: x.fillna(x.median())
)
```

#### 3.2 Age 列边界检查
```python
# 检查是否有不合理的年龄值
print(f"Age < 1: {(df['Age'] < 1).sum()} 条")
print(f"Age > 100: {(df['Age'] > 100).sum()} 条")

# 婴儿年龄是正常的（有 0.42 岁的记录）
```

### 步骤 4: 特征工程（提升数据质量）

```python
# 4.1 从 Name 提取称谓
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# 合并稀有称谓
rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# 4.2 创建家庭规模特征
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 4.3 创建是否独自旅行特征
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# 4.4 对 Fare 进行分箱
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Medium', 'High', 'VeryHigh'])

# 4.5 对 Age 进行分箱
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100], 
                      labels=['Child', 'Teenager', 'Adult', 'MiddleAge', 'Senior'])

# 删除原始 Name 和 Ticket 列
df.drop(['Name', 'Ticket'], axis=1, inplace=True)
```

### 步骤 5: 数据类型转换

```python
# 转换类别型变量
categorical_cols = ['Sex', 'Embarked', 'Title', 'Cabin_Level', 'FareBin', 'AgeBin']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# 查看最终数据类型
print(df.dtypes)
```

### 步骤 6: 重复值检查

```python
# 检查重复行
duplicate_count = df.duplicated().sum()
print(f"重复行数量: {duplicate_count}")

# 如有重复则删除
if duplicate_count > 0:
    df.drop_duplicates(inplace=True)
```

---

## 三、预期效果

### 1. 数据质量提升指标

| 指标 | 清洗前 | 清洗后 | 改善 |
|------|--------|--------|------|
| 缺失值比例 | 19.87% (Age) / 77.10% (Cabin) | 0% | ✅ 完全消除 |
| 异常值比例 | ~0.3% (Fare=0) | 0% | ✅ 完全消除 |
| 特征数量 | 11 个 | 13 个 | ✅ 增加 2 个有效特征 |
| 数据完整性 | 81% | 100% | ✅ 显著提升 |

### 2. 特征工程收益

| 新特征 | 说明 | 预期价值 |
|--------|------|---------|
| Title | 乘客称谓（Mr/Mrs/Miss/Master/Rare） | 反映社会地位，与生存率相关 |
| FamilySize | 家庭成员总数 | 中等规模家庭生存率更高 |
| IsAlone | 是否独自旅行 | 独自旅行者生存率较低 |
| FareBin | 票价分箱 | 处理 Fare 的偏态分布 |
| AgeBin | 年龄分箱 | 反映不同年龄段生存差异 |
| Cabin_Level | 舱位等级（A/B/C/D/E/F/G/T/Unknown） | 替代高缺失率的原 Cabin |

### 3. 对模型训练的积极影响

1. **减少过拟合风险**：合理的缺失值填充降低模型学习到错误模式的可能性
2. **提升特征表达能力**：Title、FamilySize 等特征比原始特征更具预测力
3. **处理类别不平衡**：FareBin 和 AgeBin 的分箱处理减少极端值影响
4. **消除噪声数据**：Fare=0 的异常值修复确保模型训练稳定

### 4. 最终数据快照

```python
# 清洗后的数据概况
print(f"最终数据维度: {df.shape}")
print(f"\n各列缺失值统计:\n{df.isnull().sum()}")
print(f"\n前 5 行数据预览:\n{df.head()}")
```

**预期输出维度**: 891 行 × 13 列（特征更丰富，无缺失值）