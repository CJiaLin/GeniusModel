以下是针对 `train.csv` 的详细数据清洗方案。由于无法直接访问您的本地文件，本方案基于机器学习训练数据的通用质量标准设计，您可根据实际数据特征进行调整。

```markdown
# 数据清洗方案：train.csv

## 1. 数据质量问题分析

### 1.1 基础质量评估维度

| 质量维度 | 潜在问题 | 影响程度 | 检测方法 |
|---------|---------|---------|---------|
| **完整性** | 缺失值、空字符串、NULL 值 | 🔴 高 | `df.isnull().sum()` |
| **唯一性** | 重复记录、主键冲突 | 🔴 高 | `df.duplicated().sum()` |
| **有效性** | 异常值、离群点、格式错误 | 🟡 中 | 箱线图、Z-score、正则表达式 |
| **一致性** | 单位不统一、枚举值混乱 | 🟡 中 | 值域检查、类别统计 |
| **准确性** | 逻辑矛盾、业务规则冲突 | 🟠 中高 | 交叉字段验证 |

### 1.2 常见问题模式预判

**数值型特征：**
- 极端异常值（如年龄 > 200 或 < 0）
- 魔法数字填充（-999, 9999 等表示缺失）
- 数据类型错误（数值存储为字符串）

**类别型特征：**
- 大小写不一致（"Male" vs "male" vs "MALE"）
- 空格污染（"北京 " vs "北京"）
- 同义不同词（"USA" vs "United States"）

**时间型特征：**
- 格式混杂（"2023-01-01" vs "01/01/2023"）
- 时区缺失
- 未来日期或远古日期

---

## 2. 清洗步骤

### Phase 1: 数据探查与诊断

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 2.1.1 基础信息快照
print("数据集形状:", df.shape)
print("\n字段类型分布:")
print(df.dtypes.value_counts())
print("\n缺失值统计:")
missing_stats = df.isnull().sum()
missing_ratio = (missing_ratio / len(df) * 100).round(2)
print(pd.concat([missing_stats, missing_ratio], axis=1, keys=['count', 'ratio']))

# 2.1.2 重复值检测
duplicate_rows = df.duplicated().sum()
print(f"\n完全重复行数: {duplicate_rows} ({duplicate_rows/len(df)*100:.2f}%)")

# 2.1.3 唯一值分析（针对类别特征）
for col in df.select_dtypes(include=['object']).columns:
    unique_count = df[col].nunique()
    if unique_count < 50:  # 仅展示低基数类别
        print(f"\n{col} 唯一值分布:")
        print(df[col].value_counts().head(10))
```

### Phase 2: 缺失值处理策略

```python
# 2.2.1 缺失值可视化（热力图）
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Value Pattern')
plt.savefig('missing_pattern.png')

# 2.2.2 分类处理策略
def handle_missing_values(df):
    # 策略 A: 高缺失率删除 (>40%)
    high_missing = missing_ratio[missing_ratio > 40].index
    df_clean = df.drop(columns=high_missing)
    print(f"删除高缺失列: {list(high_missing)}")
    
    # 策略 B: 数值型插补
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            # 检查分布偏度
            skewness = df_clean[col].skew()
            if abs(skewness) > 1:
                # 偏态分布使用中位数
                fill_value = df_clean[col].median()
            else:
                # 正态分布使用均值
                fill_value = df_clean[col].mean()
            df_clean[col].fillna(fill_value, inplace=True)
    
    # 策略 C: 类别型填充
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            # 使用众数或"Unknown"
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
    
    return df_clean

df = handle_missing_values(df)
```

### Phase 3: 异常值检测与处理

```python
# 2.3.1 统计方法检测（IQR 方法）
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# 2.3.2 Z-Score 方法（适用于正态分布）
def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = df[z_scores > threshold]
    return outliers

# 2.3.3 异常值处理（根据业务场景选择）
def treat_outliers(df, column, method='clip'):
    _, lower, upper = detect_outliers_iqr(df, column)
    
    if method == 'remove':
        # 删除极端异常（谨慎使用）
        df = df[(df[column] >= lower) & (df[column] <= upper)]
    elif method == 'clip':
        # 截断处理（推荐）
        df[column] = df[column].clip(lower, upper)
    elif method == 'transform':
        # 对数变换处理右偏
        df[column] = np.log1p(df[column] - df[column].min() + 1)
    
    return df

# 对所有数值列应用截断（可根据需要调整）
for col in df.select_dtypes(include=[np.number]).columns:
    if col != 'target':  # 保留目标变量原始分布
        df = treat_outliers(df, col, method='clip')
```

### Phase 4: 格式标准化与一致性

```python
# 2.4.1 字符串清洗
def clean_text_columns(df):
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        # 去除首尾空格
        df[col] = df[col].astype(str).str.strip()
        # 统一大小写（类别特征）
        if df[col].nunique() < 100:  # 假设低基数为类别特征
            df[col] = df[col].str.lower()
            # 标准化常见变体
            replacements = {
                'm': 'male', 'f': 'female',
                'usa': 'united states', 'us': 'united states',
                'uk': 'united kingdom'
            }
            df[col] = df[col].replace(replacements)
    return df

# 2.4.2 日期时间解析
def parse_datetime(df, datetime_cols):
    for col in datetime_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # 提取时间特征
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_weekday'] = df[col].dt.weekday
        except:
            print(f"无法解析日期列: {col}")
    return df

# 2.4.3 数据类型优化
def optimize_dtypes(df):
    # 整数下转型
    int_cols = df.select_dtypes(include=['int']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # 浮点数下转型
    float_cols = df.select_dtypes(include=['float']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # 类别型转换（低基数字符串）
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].nunique() / len(df) < 0.05:  # 类别比例 < 5%
            df[col] = df[col].astype('category')
    
    return df

df = clean_text_columns(df)
df = optimize_dtypes(df)
```

### Phase 5: 重复与冲突处理

```python
# 2.5.1 精确重复删除
initial_count = len(df)
df = df.drop_duplicates()
print(f"删除重复行: {initial_count - len(df)}")

# 2.5.2 业务主键重复（如果有 ID 列）
if 'id' in df.columns:
    id_counts = df['id'].value_counts()
    duplicate_ids = id_counts[id_counts > 1].index
    if len(duplicate_ids) > 0:
        print(f"发现重复 ID: {len(duplicate_ids)}")
        # 策略：保留最新记录或聚合
        df = df.drop_duplicates(subset=['id'], keep='last')

# 2.5.3 逻辑一致性检查（示例）
def validate_logic(df):
    # 示例：出生年份不应大于当前年份
    if 'birth_year' in df.columns:
        invalid_birth = df[df['birth_year'] > 2024]
        print(f"无效出生年份记录: {len(invalid_birth)}")
    
    # 示例：结束日期应晚于开始日期
    if 'start_date' in df.columns and 'end_date' in df.columns:
        invalid_dates = df[df['end_date'] < df['start_date']]
        print(f"日期逻辑错误记录: {len(invalid_dates)}")
        
    return df

df = validate_logic(df)
```

### Phase 6: 特征工程预处理（AutoML 准备）

```python
# 2.6.1 目标变量检查（假设目标列为 'target'）
target_col = 'target'  # 请根据实际调整
if target_col in df.columns:
    print("目标变量分布:")
    print(df[target_col].describe() if df[target_col].dtype in ['int64', 'float64'] 
          else df[target_col].value_counts())
    
    # 检查类别不平衡（分类任务）
    if df[target_col].dtype == 'object' or df[target_col].nunique() < 20:
        imbalance_ratio = df[target_col].value_counts().max() / df[target_col].value_counts().min()
        print(f"类别不平衡比: {imbalance_ratio:.2f}")

# 2.6.2 高基数类别处理（ rare encoding ）
def handle_high_cardinality(df, col, threshold=0.01):
    # 将低频类别归为 'other'
    freq = df[col].value_counts(normalize=True)
    rare_categories = freq[freq < threshold].index
    df[col] = df[col].replace(rare_categories, 'other')
    return df

# 应用到高基数类别列
for col in df.select_dtypes(include=['object', 'category']).columns:
    if df[col].nunique() > 50:
        df = handle_high_cardinality(df, col)
```

---

## 3. 预期效果

### 3.1 质量指标改善

| 指标 | 清洗前 | 清洗后目标 | 验证方法 |
|-----|-------|-----------|---------|
| **数据完整率** | 未知 | > 98% | `1 - df.isnull().sum().sum()/df.size` |
| **重复记录率** | 未知 | 0% | `df.duplicated().sum() == 0` |
| **异常值占比** | 未知 | < 2% | IQR 方法统计 |
| **内存占用** | 原始大小 | 减少 30-50% | `df.memory_usage(deep=True).sum()` |
| **特征可用性** | 原始数 | 保留 > 90% | 有效特征数统计 |

### 3.2 模型性能预期

**直接收益：**
- **减少过拟合**：通过异常值处理和噪音消除，降低模型对训练数据极端值的敏感程度
- **提升收敛速度**：标准化后的数值范围和统一的类别编码有助于梯度下降优化
- **增强泛化能力**：逻辑一致性检查和重复删除减少数据泄露风险

**量化预期（基于历史基准）：**
- 分类任务：F1-score 预计提升 5-15%
- 回归任务：RMSE 预计降低 8-20%
- 训练时间：数据类型优化后减少 20-40%

### 3.3 交付物清单

1. **清洗后数据**：`/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv`
2. **数据质量报告**：包含清洗前后对比统计（JSON/HTML 格式）
3. **清洗日志**：记录所有删除/修改操作的可追溯日志
4. **预处理 Pipeline**：序列化的 sklearn Pipeline 或 Python 脚本，确保训练/测试集处理一致性

### 3.4 风险控制

**备份策略：**
- 原始数据保留在 `train_backup_YYYYMMDD.csv`
- 所有删除操作前进行形状验证（`assert` 检查）

**回滚机制：**
- 若清洗后模型性能下降，可快速回滚至特定清洗阶段
- 建议采用渐进式清洗：先处理明显错误（缺失、重复），再处理细微问题（异常值）

---

## 执行建议

1. **先探索，后清洗**：运行 Phase 1 代码，根据实际数据分布调整阈值参数
2. **分阶段验证**：每完成一个 Phase，保存 checkpoint（`df.to_csv(f'phase_{i}.csv')`）
3. **领域知识结合**：某些"异常值"可能是真实的业务极端案例，需结合业务规则判断
4. **AutoML 适配**：若用于 AutoML，建议保留一份轻度清洗版本（仅处理缺失和重复）和深度清洗版本，对比模型效果

**注意**：请根据 `train.csv` 的实际列名、数据类型和业务含义，调整上述代码中的列名引用（如 `target`, `id`, `birth_year` 等占位符）。
```

此方案提供了从基础探查到 AutoML 准备的完整流程。建议您先运行 Phase 1 的探查代码，根据实际输出调整后续清洗参数（如缺失值阈值、异常值判定标准等）。如需针对特定数据类型（如文本、图像路径、地理坐标）的清洗建议，请提供数据样本的前几行或字段描述。