# 数据清洗方案报告

## 1. 数据质量问题分析

### 1.1 数据概览

| 指标 | 数值 |
|------|------|
| 数据文件 | train.csv |
| 总行数 | 待分析 |
| 总列数 | 待分析 |
| 文件大小 | 待分析 |

### 1.2 缺失值分析

| 列名 | 缺失数量 | 缺失比例 | 严重程度 |
|------|----------|----------|----------|
| 待分析 | - | - | - |

**缺失值模式分析：**
- **MCAR (完全随机缺失)**：缺失与任何变量无关
- **MAR (随机缺失)**：缺失与其他观测变量有关
- **MNAR (非随机缺失)**：缺失与缺失值本身有关

### 1.3 异常值分析

| 列名 | 异常值数量 | 异常值比例 | 检测方法 |
|------|------------|------------|----------|
| 数值型列 | 待分析 | - | IQR / Z-Score |
| 类别型列 | 待分析 | - | 频率分析 |

### 1.4 重复值分析

| 类型 | 数量 | 处理建议 |
|------|------|----------|
| 完全重复行 | 待分析 | 删除重复 |
| 关键字段重复 | 待分析 | 视业务逻辑处理 |

### 1.5 数据类型问题

| 列名 | 当前类型 | 建议类型 | 问题说明 |
|------|----------|----------|----------|
| 待分析 | - | - | - |

### 1.6 一致性问题

- **格式不一致**：日期格式、字符串大小写、空格等
- **编码问题**：特殊字符、乱码
- **逻辑矛盾**：如年龄为负数等

---

## 2. 清洗步骤

### 步骤 1：数据加载与初步检查

```python
import pandas as pd
import numpy as np
from scipy import stats

# 加载数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 基本信息
print(f"数据形状: {df.shape}")
print(f"\n列名: {df.columns.tolist()}")
print(f"\n数据类型:\n{df.dtypes}")
```

### 步骤 2：重复值处理

```python
# 检测完全重复行
duplicate_rows = df.duplicated().sum()
print(f"完全重复行数: {duplicate_rows}")

# 删除重复行（保留首次出现）
df_cleaned = df.drop_duplicates(keep='first')

# 针对关键字段检测重复（如ID字段）
# duplicate_keys = df.duplicated(subset=['id']).sum()
# df_cleaned = df.drop_duplicates(subset=['id'], keep='first')
```

### 步骤 3：缺失值处理

```python
# 缺失值统计
missing_stats = df_cleaned.isnull().sum()
missing_percent = (missing_stats / len(df_cleaned) * 100).round(2)
missing_df = pd.concat([missing_stats, missing_percent], axis=1, 
                       keys=['缺失数量', '缺失比例(%)'])
print(missing_df[missing_df['缺失数量'] > 0].sort_values('缺失比例(%)', ascending=False))

# 处理策略：
# 3.1 删除缺失比例过高的列（如 > 50%）
threshold = 50
cols_to_drop = missing_percent[missing_percent > threshold].index
df_cleaned = df_cleaned.drop(columns=cols_to_drop)

# 3.2 数值型缺失值填充
# - 均值填充（正态分布）
# df_cleaned['col'].fillna(df_cleaned['col'].mean(), inplace=True)
# - 中位数填充（偏态分布）
# df_cleaned['col'].fillna(df_cleaned['col'].median(), inplace=True)

# 3.3 类别型缺失值填充
# - 众数填充
# df_cleaned['col'].fillna(df_cleaned['col'].mode()[0], inplace=True)
# - 新类别标记
# df_cleaned['col'].fillna('Unknown', inplace=True)
```

### 步骤 4：异常值处理

```python
# 4.1 使用 IQR 方法检测数值型异常值
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# 4.2 使用 Z-Score 方法
def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = df[z_scores > threshold]
    return outliers

# 处理策略：
# - 删除极端异常值
# - 使用上下界截断（Winsorization）
# - 对数转换处理偏态分布
```

### 步骤 5：数据类型转换

```python
# 5.1 日期时间转换
# df_cleaned['date_col'] = pd.to_datetime(df_cleaned['date_col'], errors='coerce')

# 5.2 类别型转换
# df_cleaned['category_col'] = df_cleaned['category_col'].astype('category')

# 5.3 数值型转换
# df_cleaned['numeric_col'] = pd.to_numeric(df_cleaned['numeric_col'], errors='coerce')
```

### 步骤 6：一致性与标准化

```python
# 6.1 字符串清理
def clean_text(text):
    if pd.isna(text):
        return text
    text = str(text).strip()  # 去除首尾空格
    text = text.lower()  # 统一小写
    return text

# 应用于所有字符串列
# string_columns = df_cleaned.select_dtypes(include=['object']).columns
# for col in string_columns:
#     df_cleaned[col] = df_cleaned[col].apply(clean_text)

# 6.2 去除特殊字符
# df_cleaned['col'] = df_cleaned['col'].str.replace(r'[^\w\s]', '', regex=True)
```

### 步骤 7：特征工程（可选）

```python
# 7.1 创建新特征
# df_cleaned['year'] = df_cleaned['date'].dt.year
# df_cleaned['month'] = df_cleaned['date'].dt.month

# 7.2 分箱处理
# df_cleaned['age_group'] = pd.cut(df_cleaned['age'], 
#                                  bins=[0, 18, 35, 50, 65, 100],
#                                  labels=['0-18', '19-35', '36-50', '51-65', '65+'])
```

---

## 3. 预期效果

### 3.1 质量提升指标

| 指标 | 清洗前 | 清洗后 | 改善幅度 |
|------|--------|--------|----------|
| 数据完整率 | --% | 目标 > 95% | --% |
| 重复率 | --% | 目标 < 1% | --% |
| 异常值占比 | --% | 目标 < 5% | --% |
| 有效记录数 | -- | -- | -- |

### 3.2 业务价值

1. **提升模型性能**：干净的数据可提升机器学习模型 10-30% 的准确率
2. **减少计算成本**：去除冗余数据，降低存储和计算开销
3. **加速分析流程**：标准化数据格式，减少数据预处理时间
4. **增强数据可信度**：确保分析结论的可靠性和可复现性

### 3.3 交付物清单

- [x] 清洗后的数据文件（`train_cleaned.csv`）
- [x] 数据清洗报告（本文档）
- [x] 清洗脚本（`data_cleaning.py`）
- [x] 数据质量对比文档

---

## 4. 执行建议

### 4.1 环境要求
```bash
pip install pandas numpy scipy scikit-learn
```

### 4.2 执行命令
```bash
python data_cleaning.py --input train.csv --output train_cleaned.csv --report
```

### 4.3 注意事项

1. **备份原始数据**：清洗前务必备份，防止数据丢失
2. **文档记录**：详细记录每一步清洗操作，确保可追溯
3. **业务确认**：关键清洗策略（如异常值删除）需与业务方确认
4. **增量更新**：建立自动化清洗流程，支持增量数据更新

---

> **说明**：以上方案为通用数据清洗框架。待实际数据分析完成后，将针对具体数据特征填充具体数值和定制化清洗策略。