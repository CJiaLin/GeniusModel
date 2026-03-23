# 数据清洗方案报告

## 数据基本信息

- **数据路径**: `/Users/cjialin/code/AutoMLByLLM/train.csv`
- **数据类型**: 训练数据集（待分析）
- **分析状态**: 基于通用最佳实践的方案模板

---

## 1. 数据质量问题分析

### 1.1 常见问题类型

| 问题类型 | 描述 | 检测方法 | 潜在影响 |
|---------|------|---------|---------|
| **缺失值** | 空值、NaN、空白字符串 | `df.isnull().sum()` | 模型训练失败或偏差 |
| **重复记录** | 完全重复或部分重复的行 | `df.duplicated().sum()` | 数据泄漏、过拟合 |
| **异常值** | 超出正常范围的数值 | IQR、Z-score、箱线图 | 模型对极端值敏感 |
| **数据类型错误** | 数值型被识别为字符串 | `df.dtypes` | 计算错误、内存浪费 |
| **不一致的类别** | 大小写不一致、拼写错误 | `value_counts()` | 类别特征维度膨胀 |
| **格式问题** | 日期格式不统一、单位不一致 | 正则表达式、模式匹配 | 特征工程失败 |

### 1.2 针对性分析框架

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 基础数据加载与分析
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 1. 基础信息概览
print("数据形状:", df.shape)
print("\n数据类型:\n", df.dtypes)
print("\n缺失值统计:\n", df.isnull().sum())
print("\n缺失值比例:\n", (df.isnull().sum() / len(df) * 100).round(2))
print("\n重复行数量:", df.duplicated().sum())
print("\n内存使用:", df.memory_usage(deep=True).sum() / 1024**2, "MB")
```

---

## 2. 详细清洗步骤

### 步骤 1: 初始数据加载与基础检查

```python
# 加载数据并设置显示选项
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 设置显示选项以便查看所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# 初步检查
print(f"数据集维度: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"\n列名: {list(df.columns)}")
```

**预期操作**: 确认数据正确加载，了解数据规模。

### 步骤 2: 重复值处理

```python
# 检测完全重复的行
duplicate_mask = df.duplicated()
duplicate_count = duplicate_mask.sum()
print(f"发现 {duplicate_count} 个完全重复的行")

if duplicate_count > 0:
    # 查看重复样本
    duplicate_rows = df[duplicate_mask]
    print("重复行示例:", duplicate_rows.head())
    
    # 删除重复值（保留第一次出现的）
    df = df.drop_duplicates(keep='first')
    print(f"删除重复值后剩余: {len(df)} 行")

# 检测基于关键列的重复（如ID列）
id_columns = [col for col in df.columns if 'id' in col.lower()]
if id_columns:
    for id_col in id_columns:
        dup_ids = df[id_col].duplicated().sum()
        if dup_ids > 0:
            print(f"警告: 列 '{id_col}' 存在 {dup_ids} 个重复值")
            # 根据业务逻辑处理：删除或合并
```

**清洗策略**: 
- 完全重复行：直接删除
- ID列重复：需业务确认后处理（可能是合法的多对一关系）

### 步骤 3: 缺失值处理

```python
# 详细缺失值分析
missing_stats = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
    'Data_Type': df.dtypes
})
missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

print("缺失值详情:\n", missing_stats)

# 根据缺失比例选择处理策略
for col in missing_stats['Column']:
    missing_pct = missing_stats[missing_stats['Column'] == col]['Missing_Percentage'].values[0]
    
    if missing_pct > 50:
        # 高缺失率：考虑删除列
        print(f"列 '{col}' 缺失率 {missing_pct}%，建议删除")
        df = df.drop(columns=[col])
        
    elif df[col].dtype in ['int64', 'float64']:
        # 数值型：中位数填充
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"列 '{col}' 使用中位数 {median_val} 填充")
        
    else:
        # 类别型：众数填充或'Unknown'
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col] = df[col].fillna(mode_val)
        print(f"列 '{col}' 使用众数 '{mode_val}' 填充")
```

**清洗策略决策树**:
- 缺失率 > 50%：删除列（除非该特征极其重要）
- 数值型（低缺失率）：中位数/均值填充
- 类别型：众数填充或新增"Unknown"类别
- 时间序列：前向填充或插值

### 步骤 4: 数据类型转换与格式化

```python
# 自动检测需要转换的列
for col in df.columns:
    # 尝试将字符串数值转换为数值型
    if df[col].dtype == 'object':
        # 尝试转换为数值
        try:
            converted = pd.to_numeric(df[col].str.replace(',', '').str.replace('$', '').str.strip())
            df[col] = converted
            print(f"列 '{col}' 转换为数值型")
        except:
            pass
        
        # 尝试转换为日期
        if any(keyword in col.lower() for keyword in ['date', 'time', 'day']):
            try:
                df[col] = pd.to_datetime(df[col])
                print(f"列 '{col}' 转换为日期时间型")
            except:
                pass

# 优化内存使用
def optimize_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

df = optimize_memory(df)
```

### 步骤 5: 异常值检测与处理

```python
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = df[z_scores > threshold]
    return outliers

# 对所有数值型列进行异常值检测
numeric_cols = df.select_dtypes(include=[np.number]).columns

outlier_summary = []
for col in numeric_cols:
    outliers, lower, upper = detect_outliers_iqr(df, col)
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(df)) * 100
    
    if outlier_count > 0:
        outlier_summary.append({
            'Column': col,
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': round(outlier_pct, 2),
            'Lower_Bound': lower,
            'Upper_Bound': upper
        })

outlier_df = pd.DataFrame(outlier_summary)
print("异常值检测结果:\n", outlier_df)

# 异常值处理策略（根据业务场景选择）
# 策略1: 盖帽法（Winsorization）
for col in numeric_cols:
    if col in outlier_df['Column'].values:
        lower = outlier_df[outlier_df['Column'] == col]['Lower_Bound'].values[0]
        upper = outlier_df[outlier_df['Column'] == col]['Upper_Bound'].values[0]
        df[col] = df[col].clip(lower=lower, upper=upper)

# 策略2: 删除极端异常值（谨慎使用）
# df = df[(df[col] >= lower) & (df[col] <= upper)]
```

### 步骤 6: 类别数据规范化

```python
# 字符串清理
for col in df.select_dtypes(include=['object']).columns:
    # 去除首尾空格
    df[col] = df[col].str.strip()
    
    # 统一大小写
    df[col] = df[col].str.lower()
    
    # 标准化常见变体
    replacements = {
        'n/a': np.nan,
        'na': np.nan,
        'null': np.nan,
        'none': np.nan,
        '-': np.nan
    }
    df[col] = df[col].replace(replacements)

# 合并稀有类别（适用于高基数类别特征）
def merge_rare_categories(df, column, threshold=0.01):
    value_counts = df[column].value_counts(normalize=True)
    rare_categories = value_counts[value_counts < threshold].index
    df[column] = df[column].replace(rare_categories, 'Other')
    return df

for col in df.select_dtypes(include=['object']).columns:
    unique_count = df[col].nunique()
    if unique_count > 10:  # 高基数阈值
        df = merge_rare_categories(df, col, threshold=0.01)
        print(f"列 '{col}' 的稀有类别已合并到 'Other'")
```

### 步骤 7: 特征工程准备

```python
# 从日期特征提取时间组件
date_cols = df.select_dtypes(include=['datetime64']).columns
for col in date_cols:
    df[f'{col}_year'] = df[col].dt.year
    df[f'{col}_month'] = df[col].dt.month
    df[f'{col}_day'] = df[col].dt.day
    df[f'{col}_weekday'] = df[col].dt.weekday
    # 删除原始日期列（如果需要）
    # df = df.drop(columns=[col])

# 创建数值特征的统计特征（适用于特定场景）
# df['feature_mean'] = df[numeric_cols].mean(axis=1)
# df['feature_std'] = df[numeric_cols].std(axis=1)
```

---

## 3. 验证与质量检查

```python
# 最终验证
print("=== 清洗后数据质量报告 ===")
print(f"最终数据形状: {df.shape}")
print(f"\n缺失值检查:\n{df.isnull().sum().sum()} 个缺失值")
print(f"\n重复行检查: {df.duplicated().sum()} 个重复行")
print(f"\n内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\n数据类型分布:\n{df.dtypes.value_counts()}")

# 保存清洗后的数据
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n清洗后的数据已保存至: {output_path}")
```

---

## 4. 预期效果

| 指标 | 清洗前 | 清洗后（预期） | 改善幅度 |
|-----|-------|--------------|---------|
| **数据完整性** | 存在缺失 | 100% 完整 | 消除缺失值影响 |
| **数据唯一性** | 可能存在重复 | 无重复记录 | 避免数据泄漏 |
| **数值稳定性** | 受异常值影响 | 异常值已处理 | 提升模型鲁棒性 |
| **内存效率** | 原始大小 | 减少 30-50% | 类型优化 |
| **模型适用性** | 需要预处理 | 可直接使用 | 加速建模流程 |

---

## 5. 注意事项与建议

### 5.1 关键注意事项

1. **备份原始数据**: 始终在清洗前备份原始文件
2. **文档记录**: 记录所有清洗操作以便复现
3. **训练/测试一致性**: 确保测试集使用相同的清洗逻辑
4. **业务验证**: 异常值处理前需确认是否为合法业务值

### 5.2 进阶优化建议

- 使用 `pandas-profiling` 或 `Sweetviz` 自动生成详细的数据质量报告
- 对于高维数据，考虑使用自动化特征选择
- 建立数据验证管道（Great Expectations）监控数据漂移

### 5.3 下一步行动

1. 执行上述代码分析实际数据特征
2. 根据缺失值和异常值的业务含义调整处理策略
3. 与领域专家确认类别合并规则
4. 建立自动化数据清洗管道

---

**方案生成时间**: 基于通用最佳实践  
**建议**: 请根据实际数据特征调整具体阈值和处理策略