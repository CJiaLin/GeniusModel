# 数据清洗方案报告

**数据路径**: `/Users/cjialin/code/AutoMLByLLM/train.csv`  
**生成日期**: 2024年  
**方案版本**: v1.0

---

## 1. 数据质量问题分析

### 1.1 常见问题类型

基于机器学习项目数据的常见特征，预期可能存在以下质量问题：

| 问题类型 | 描述 | 影响程度 |
|---------|------|---------|
| **缺失值** | 特征列中存在空值（NaN、None、空字符串） | 🔴 高 |
| **重复记录** | 完全重复或部分重复的行 | 🔴 高 |
| **异常值** | 数值型数据中的极端值、离群点 | 🟡 中 |
| **格式不一致** | 日期格式混乱、字符串大小写不统一 | 🟡 中 |
| **类型错误** | 数值列存储为字符串、类别编码混乱 | 🟡 中 |
| **高基数类别** | 类别特征的唯一值过多 | 🟢 低 |
| **常量特征** | 所有行取值相同的列 | 🟢 低 |

### 1.2 诊断代码

```python
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 加载数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 基础信息报告
def data_quality_report(df):
    report = {
        '数据形状': df.shape,
        '内存使用': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        '重复行数': df.duplicated().sum(),
        '完全缺失列': df.columns[df.isnull().all()].tolist()
    }
    
    # 缺失值分析
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        '缺失数量': missing,
        '缺失比例(%)': missing_pct
    }).sort_values('缺失比例(%)', ascending=False)
    
    # 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_summary = df[numeric_cols].describe()
    
    # 类别列分析
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    cardinality = {col: df[col].nunique() for col in categorical_cols}
    
    return report, missing_df, stats_summary, cardinality

# 执行诊断
report, missing_df, stats_summary, cardinality = data_quality_report(df)

print("=== 基础信息 ===")
for k, v in report.items():
    print(f"{k}: {v}")

print("\n=== 缺失值分析 (Top 10) ===")
print(missing_df.head(10))
```

---

## 2. 详细清洗步骤

### 步骤 1: 重复值处理

**策略**: 删除完全重复的行，保留首次出现

```python
# 删除完全重复行
initial_rows = len(df)
df_clean = df.drop_duplicates(keep='first')

# 处理部分重复（基于关键业务字段）
# 例如：如果'id'列存在，确保id唯一
if 'id' in df_clean.columns:
    df_clean = df_clean.drop_duplicates(subset=['id'], keep='first')

print(f"删除重复行: {initial_rows - len(df_clean)} 行")
```

### 步骤 2: 缺失值处理

**策略**: 根据缺失比例选择不同策略

```python
def handle_missing_values(df, strategy_dict=None):
    """
    智能缺失值处理
    - 缺失比例 < 5%: 删除行
    - 5% <= 缺失比例 < 50%: 填充（数值用中位数，类别用众数）
    - 缺失比例 >= 50%: 删除列或标记为'Missing'类别
    """
    df_result = df.copy()
    
    for col in df_result.columns:
        missing_pct = df_result[col].isnull().sum() / len(df_result)
        
        if missing_pct == 0:
            continue
            
        if missing_pct < 0.05:
            # 删除缺失行
            df_result = df_result.dropna(subset=[col])
            print(f"[{col}] 缺失{missing_pct:.1%}: 删除相关行")
            
        elif missing_pct < 0.5:
            if df_result[col].dtype in ['int64', 'float64']:
                # 数值型：中位数填充（对异常值更鲁棒）
                fill_value = df_result[col].median()
                df_result[col].fillna(fill_value, inplace=True)
                print(f"[{col}] 缺失{missing_pct:.1%}: 中位数填充 ({fill_value})")
            else:
                # 类别型：众数填充
                fill_value = df_result[col].mode()[0]
                df_result[col].fillna(fill_value, inplace=True)
                print(f"[{col}] 缺失{missing_pct:.1%}: 众数填充 ({fill_value})")
        else:
            # 高缺失比例：创建指示变量或删除
            if df_result[col].dtype == 'object':
                df_result[col].fillna('Missing', inplace=True)
                print(f"[{col}] 缺失{missing_pct:.1%}: 标记为'Missing'")
            else:
                df_result.drop(columns=[col], inplace=True)
                print(f"[{col}] 缺失{missing_pct:.1%}: 删除列")
    
    return df_result

df_clean = handle_missing_values(df_clean)
```

### 步骤 3: 异常值检测与处理

**策略**: 使用IQR方法和Z-Score检测异常值

```python
def detect_outliers(df, method='iqr', threshold=3):
    """
    异常值检测
    method: 'iqr' 或 'zscore'
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = {}
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df[np.abs(stats.zscore(df[col])) > threshold]
        
        outlier_count = len(outliers)
        outlier_pct = outlier_count / len(df) * 100
        
        if outlier_count > 0:
            outlier_summary[col] = {
                '数量': outlier_count,
                '比例': f"{outlier_pct:.2f}%",
                '下限': lower_bound if method == 'iqr' else None,
                '上限': upper_bound if method == 'iqr' else None
            }
    
    return outlier_summary

# 检测异常值
outliers = detect_outliers(df_clean, method='iqr')
print("异常值检测结果:", outliers)

# 处理异常值（盖帽法/ Winsorization）
def winsorize_column(series, lower_percentile=0.01, upper_percentile=0.99):
    """将极端值限制在指定百分位数"""
    lower_bound = series.quantile(lower_percentile)
    upper_bound = series.quantile(upper_percentile)
    return series.clip(lower=lower_bound, upper=upper_bound)

# 对数值列应用Winsorization（可选）
# for col in ['income', 'age', 'price']:  # 根据实际列名调整
#     if col in df_clean.columns:
#         df_clean[col] = winsorize_column(df_clean[col])
```

### 步骤 4: 数据类型优化与格式标准化

```python
def optimize_dtypes(df):
    """优化内存使用，转换数据类型"""
    df_result = df.copy()
    
    # 数值类型优化
    for col in df_result.select_dtypes(include=['int64']).columns:
        c_min, c_max = df_result[col].min(), df_result[col].max()
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            df_result[col] = df_result[col].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            df_result[col] = df_result[col].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df_result[col] = df_result[col].astype(np.int32)
    
    for col in df_result.select_dtypes(include=['float64']).columns:
        c_min, c_max = df_result[col].min(), df_result[col].max()
        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            df_result[col] = df_result[col].astype(np.float32)
    
    # 类别类型优化（低基数字符串）
    for col in df_result.select_dtypes(include=['object']).columns:
        if df_result[col].nunique() / len(df_result) < 0.5:  # 基数小于50%
            df_result[col] = df_result[col].astype('category')
    
    return df_result

df_clean = optimize_dtypes(df_clean)

# 字符串标准化
def standardize_text(df, text_cols):
    """标准化文本数据：去除空格、统一大小写"""
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            # 合并重复类别（如'usa', 'us', 'united states'）
            # 需要定义映射字典
    return df
```

### 步骤 5: 特征工程预处理

```python
# 删除常量特征（所有值相同）
constant_cols = [col for col in df_clean.columns if df_clean[col].nunique() == 1]
df_clean = df_clean.drop(columns=constant_cols)
print(f"删除常量特征: {constant_cols}")

# 处理高基数类别特征（可选）
high_cardinality_cols = [
    col for col in df_clean.select_dtypes(include=['category', 'object']).columns
    if df_clean[col].nunique() > 100
]
print(f"高基数特征（需特殊处理）: {high_cardinality_cols}")

# 日期解析（如果有日期列）
# date_cols = ['date', 'timestamp']  # 根据实际列名调整
# for col in date_cols:
#     if col in df_clean.columns:
#         df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
#         # 提取时间特征
#         df_clean[f'{col}_year'] = df_clean[col].dt.year
#         df_clean[f'{col}_month'] = df_clean[col].dt.month
#         df_clean[f'{col}_day'] = df_clean[col].dt.day
```

---

## 3. 预期效果

### 3.1 质量指标对比

| 指标 | 清洗前（预期） | 清洗后（目标） | 改善幅度 |
|------|--------------|--------------|---------|
| 数据完整率 | 85-95% | 99-100% | +5-15% |
| 重复记录数 | 0-5% | 0% | -100% |
| 异常值比例 | 1-10% | <1% 或标记 | 可控 |
| 内存占用 | 基准 | 减少30-50% | -30-50% |
| 特征可用性 | 基准 | 100%有效特征 | 标准化 |

### 3.2 下游任务收益

1. **模型训练**: 减少过拟合风险，提升泛化能力
2. **特征工程**: 确保统计特征计算准确（如均值、方差不受异常值影响）
3. **计算效率**: 优化后的数据类型减少内存使用，加快训练速度
4. **可解释性**: 标准化的类别标签和清晰的数值范围便于业务理解

### 3.3 验证代码

```python
# 清洗后验证
def validate_cleaning(df_original, df_cleaned):
    print("=== 清洗效果验证 ===")
    print(f"原始数据形状: {df_original.shape}")
    print(f"清洗后形状: {df_cleaned.shape}")
    print(f"数据保留率: {len(df_cleaned)/len(df_original)*100:.1f}%")
    print(f"\n剩余缺失值: {df_cleaned.isnull().sum().sum()}")
    print(f"重复行数: {df_cleaned.duplicated().sum()}")
    print(f"内存优化: {df_original.memory_usage(deep=True).sum()/1024**2:.2f} MB → "
          f"{df_cleaned.memory_usage(deep=True).sum()/1024**2:.2f} MB")

validate_cleaning(df, df_clean)

# 保存清洗后的数据
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df_clean.to_csv(output_path, index=False)
print(f"\n清洗完成！数据已保存至: {output_path}")
```

---

## 4. 执行建议

### 执行顺序
1. 首先运行**诊断代码**，了解实际数据质量状况
2. 根据诊断结果，选择适用的清洗步骤（并非所有步骤都必需）
3. 对每个清洗步骤，先在小样本上测试，确认无误后全量执行
4. 保存清洗前后的对比报告，便于追踪数据 lineage

### 注意事项
- 🔴 **备份原始数据**: 清洗前务必备份 `train.csv`
- 🟡 **业务知识**: 异常值处理需结合业务逻辑（如年龄>150显然是错误，但收入>100万可能是合理的）
- 🟢 **文档记录**: 记录所有清洗操作，确保可复现性

---

**方案生成完成**。请根据实际数据特征调整代码中的列名和参数阈值。如需针对特定数据类型的深度清洗（如文本数据、时间序列、地理坐标等），请提供数据样本以生成更精确的清洗方案。