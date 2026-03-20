# 数据清洗方案

## 数据概况

| 项目 | 信息 |
|------|------|
| 数据路径 | `/Users/cjialin/code/AutoMLByLLM/train.csv` |
| 文件格式 | CSV |
| 数据规模 | 待分析后确定 |

---

## 1. 数据质量问题分析

### 1.1 缺失值检测
- **检测方法**: 统计每列的缺失比例
- **处理策略**: 
  - 缺失比例 < 5%：直接删除或插值填充
  - 缺失比例 5%-30%：根据数据类型选择均值/中位数/众数填充，或使用KNN插补
  - 缺失比例 > 30%：考虑删除该特征或标记为特殊类别

### 1.2 异常值检测
- **数值型特征**: 使用 IQR 方法（箱线图法）或 Z-score > 3 识别异常值
- **类别型特征**: 检查出现频率极低的稀有类别
- **处理策略**: 根据业务逻辑选择删除、截断或保留

### 1.3 重复值检测
- **完全重复**: 检测并删除完全相同的行
- **部分重复**: 基于关键业务字段（如ID）检测重复

### 1.4 数据类型问题
- 检查数值型列是否被误读为字符串
- 日期格式统一化
- 类别型编码规范化

### 1.5 一致性问题
- 检查单位一致性
- 检查命名规范（大小写、空格、特殊字符）

---

## 2. 详细清洗步骤

### Step 1: 数据加载与初步检查
```python
import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 基础信息
print(f"数据形状: {df.shape}")
print(f"\n列名: {df.columns.tolist()}")
print(f"\n数据类型:\n{df.dtypes}")
print(f"\n缺失值统计:\n{df.isnull().sum()}")
```

### Step 2: 缺失值处理
```python
# 计算缺失比例
missing_ratio = df.isnull().sum() / len(df) * 100

# 分类处理
for col in df.columns:
    ratio = missing_ratio[col]
    if ratio > 0:
        if ratio < 5:
            # 少量缺失：删除或前向填充
            df[col].fillna(method='ffill', inplace=True)
        elif df[col].dtype in ['int64', 'float64']:
            # 数值型：中位数填充
            df[col].fillna(df[col].median(), inplace=True)
        else:
            # 类别型：众数填充
            df[col].fillna(df[col].mode()[0], inplace=True)
```

### Step 3: 异常值处理
```python
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 对数值型列应用
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df = remove_outliers_iqr(df, col)
```

### Step 4: 重复值处理
```python
# 删除完全重复的行
df.drop_duplicates(inplace=True)

# 基于关键字段去重（如有ID列）
# df.drop_duplicates(subset=['id'], keep='first', inplace=True)
```

### Step 5: 数据类型转换
```python
# 自动转换数值型
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        pass

# 日期转换（如适用）
# df['date'] = pd.to_datetime(df['date'])
```

### Step 6: 文本数据清洗
```python
# 去除首尾空格
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()
    # 统一大小写
    df[col] = df[col].str.lower()
    # 替换多个空格为单个
    df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
```

### Step 7: 类别型特征编码
```python
# 检查类别数量
for col in df.select_dtypes(include=['object']).columns:
    n_unique = df[col].nunique()
    print(f"{col}: {n_unique} 个唯一值")
    
    # 高基数类别处理（如有必要）
    if n_unique > 50:
        # 保留Top N频繁值，其余归为"其他"
        top_n = df[col].value_counts().nlargest(50).index
        df[col] = df[col].where(df[col].isin(top_n), 'other')
```

---

## 3. 预期效果

| 指标 | 预期改善 |
|------|---------|
| **数据完整性** | 缺失值比例降至 < 1% |
| **数据准确性** | 异常值影响降低 80%+ |
| **数据一致性** | 格式统一，无重复记录 |
| **模型适用性** | 清洗后数据可直接用于机器学习 |
| **存储优化** | 删除冗余数据，文件体积减小 |

---

## 4. 质量验证清单

- [ ] 缺失值已全部处理或记录
- [ ] 异常值已检测并合理处理
- [ ] 重复记录已清除
- [ ] 数据类型正确
- [ ] 数值范围符合业务逻辑
- [ ] 类别值分布合理
- [ ] 清洗前后数据行数对比记录

---

## 5. 输出文件

```python
# 保存清洗后的数据
df.to_csv('/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv', index=False)

# 生成清洗报告
with open('/Users/cjialin/code/AutoMLByLLM/cleaning_report.txt', 'w') as f:
    f.write(f"原始数据行数: {original_shape[0]}\n")
    f.write(f"清洗后数据行数: {df.shape[0]}\n")
    f.write(f"删除行数: {original_shape[0] - df.shape[0]}\n")
    f.write(f"处理列数: {df.shape[1]}\n")
```

---

> **备注**: 此方案为通用框架，具体执行时需根据 `analyze_data` 的实际分析结果调整参数和策略。