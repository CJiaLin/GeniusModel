# 数据清洗方案

## 📊 数据集概览

| 属性 | 值 |
|------|-----|
| 文件路径 | `/Users/cjialin/code/AutoMLByLLM/train.csv` |
| 数据形状 | (178, 14) |
| 总记录数 | 178 |
| 特征数量 | 14 |

---

## 1. 数据质量问题分析

### 1.1 缺失值分析

| 列名 | 缺失数量 | 缺失比例 | 数据类型 | 严重程度 |
|------|---------|---------|---------|---------|
| `is_outlier` | 12 | 6.74% | object | 🔴 中等 |
| `has_nan` | 10 | 5.62% | object | 🔴 中等 |
| `created_at` | 8 | 4.49% | object | 🟡 低 |
| `petal width category` | 6 | 3.37% | object | 🟡 低 |
| `sepal length (cm)` | 5 | 2.81% | float64 | 🟡 低 |
| `petal length (cm)` | 4 | 2.25% | float64 | 🟡 低 |
| `sepal width (cm)` | 3 | 1.69% | float64 | 🟢 轻微 |

**总计**: 约 24.16% 的列存在缺失值，最大缺失率为 6.74%。

### 1.2 重复值分析

- **重复行数**: 3 行 (1.69%)
- **影响**: 可能导致模型过拟合或评估偏差

### 1.3 异常值分析（基于 IQR 方法）

| 特征 | 异常值数量 | 异常比例 |
|------|----------|---------|
| `sepal width (cm)` | 4 | 2.25% |
| `sepal length (cm)` | 2 | 1.12% |
| `petal length (cm)` | 3 | 1.69% |
| `feature_sum` | 3 | 1.69% |

### 1.4 数据类型问题

| 问题 | 列名 | 建议处理 |
|------|------|---------|
| 日期时间字符串 | `created_at` | 转换为 datetime 类型 |
| 二元标记列 | `has_nan`, `is_outlier` | 映射为布尔值或 0/1 |
| 类别型特征 | `*_category` 列 | 标签编码或独热编码 |

---

## 2. 清洗步骤

### 步骤 1: 加载数据与环境准备

```python
import pandas as pd
import numpy as np
from datetime import datetime

# 加载原始数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 创建清洗后的数据副本
df_cleaned = df.copy()

print(f"原始数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
```

### 步骤 2: 处理重复行

```python
# 删除完全重复的行
initial_rows = len(df_cleaned)
df_cleaned = df_cleaned.drop_duplicates()
removed_duplicates = initial_rows - len(df_cleaned)

print(f"删除重复行: {removed_duplicates} 行")
```

### 步骤 3: 处理缺失值

#### 3.1 数值型特征缺失值（均值填充）

```python
# 定义数值型特征
numeric_cols = ['sepal length (cm)', 'sepal width (cm)', 
                'petal length (cm)', 'sepal length (cm)']

# 使用均值填充
for col in numeric_cols:
    if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
        mean_val = df_cleaned[col].mean()
        df_cleaned[col].fillna(mean_val, inplace=True)
        print(f"{col}: 使用均值 {mean_val:.4f} 填充")
```

#### 3.2 类别型特征缺失值（众数填充）

```python
# 类别型特征
categorical_cols = ['petal width category']

for col in categorical_cols:
    if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
        mode_val = df_cleaned[col].mode()[0]
        df_cleaned[col].fillna(mode_val, inplace=True)
        print(f"{col}: 使用众数 '{mode_val}' 填充")
```

#### 3.3 标记列缺失值（填充为"否"）

```python
# 二元标记列，假设缺失表示"否"
flag_cols = ['has_nan', 'is_outlier']

for col in flag_cols:
    if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
        df_cleaned[col].fillna('否', inplace=True)
        print(f"{col}: 使用 '否' 填充")
```

#### 3.4 日期时间缺失值

```python
# 使用当前时间或数据中的最近日期填充
if 'created_at' in df_cleaned.columns:
    # 转换为 datetime
    df_cleaned['created_at'] = pd.to_datetime(df_cleaned['created_at'], errors='coerce')
    # 使用中位数日期填充
    median_date = df_cleaned['created_at'].median()
    df_cleaned['created_at'].fillna(median_date, inplace=True)
```

### 步骤 4: 处理异常值

```python
def handle_outliers_iqr(df, column, method='cap'):
    """
    使用 IQR 方法处理异常值
    
    method: 'cap' - 封顶处理, 'remove' - 删除
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    
    if method == 'cap':
        df[column] = df[column].clip(lower_bound, upper_bound)
        print(f"{column}: 封顶处理 {outlier_count} 个异常值")
    elif method == 'remove':
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        print(f"{column}: 删除 {outlier_count} 个异常值")
    
    return df

# 对数值特征应用异常值处理
outlier_cols = ['sepal length (cm)', 'sepal width (cm)', 
                'petal length (cm)', 'feature_sum']

for col in outlier_cols:
    if col in df_cleaned.columns:
        df_cleaned = handle_outliers_iqr(df_cleaned, col, method='cap')
```

### 步骤 5: 数据类型转换

```python
# 5.1 转换布尔标记列
df_cleaned['has_nan'] = df_cleaned['has_nan'].map({'是': 1, '否': 0})
df_cleaned['is_outlier'] = df_cleaned['is_outlier'].map({'是': 1, '否': 0})

# 5.2 类别编码（标签编码示例）
from sklearn.preprocessing import LabelEncoder

category_cols = ['sepal length category', 'sepal width category', 
                 'petal length category', 'petal width category']

for col in category_cols:
    if col in df_cleaned.columns:
        le = LabelEncoder()
        df_cleaned[col + '_encoded'] = le.fit_transform(df_cleaned[col])
        print(f"{col}: 映射关系 {dict(zip(le.classes_, le.transform(le.classes_)))}")
```

### 步骤 6: 特征工程（可选）

```python
# 提取时间特征
if 'created_at' in df_cleaned.columns:
    df_cleaned['year'] = df_cleaned['created_at'].dt.year
    df_cleaned['month'] = df_cleaned['created_at'].dt.month
    df_cleaned['day'] = df_cleaned['created_at'].dt.day
    df_cleaned['dayofweek'] = df_cleaned['created_at'].dt.dayofweek
```

### 步骤 7: 保存清洗后的数据

```python
# 保存清洗结果
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df_cleaned.to_csv(output_path, index=False)

print(f"\n清洗完成！")
print(f"原始数据: {df.shape}")
print(f"清洗后数据: {df_cleaned.shape}")
print(f"保存路径: {output_path}")
```

---

## 3. 预期效果

### 3.1 质量提升指标

| 指标 | 清洗前 | 清洗后 | 改善 |
|------|--------|--------|------|
| 缺失值比例 | 24.16% | 0% | ✅ 完全消除 |
| 重复行比例 | 1.69% | 0% | ✅ 完全消除 |
| 异常值影响 | 12个异常点 | 已处理 | ✅ 风险降低 |
| 数据一致性 | 存在格式问题 | 统一标准 | ✅ 标准化 |

### 3.2 数据可用性提升

1. **完整性**: 所有记录的缺失值被合理填充，数据完整性达到 100%
2. **准确性**: 异常值通过封顶处理保留数据分布特征，避免信息损失
3. **一致性**: 日期格式统一，类别编码标准化
4. **可用性**: 生成可直接用于机器学习模型的数值特征

### 3.3 模型训练预期收益

- **减少过拟合**: 删除重复行降低记忆效应
- **提升稳定性**: 异常值处理减少极端值影响
- **加速收敛**: 标准化数据分布有助于梯度下降优化

---

## 4. 验证清单

- [ ] 运行清洗代码无错误
- [ ] 检查 `df_cleaned.isnull().sum().sum() == 0`
- [ ] 验证 `df_cleaned.duplicated().sum() == 0`
- [ ] 确认所有数值列在合理范围内
- [ ] 保存文件并能正常读取验证