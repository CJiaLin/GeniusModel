# 特征交互与组合

## 1. 乘法交互

```python
# 两个数值特征的交互
df['price_x_quantity'] = df['price'] * df['quantity']

# 选择性创建交互（基于相关性或领域知识）
interaction_pairs = [('height', 'weight'), ('income', 'age')]
for col1, col2 in interaction_pairs:
    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
```

## 2. 比率特征

```python
# 除法比率（注意分母为零）
df['price_per_sqft'] = df['price'] / df['sqft'].replace(0, np.nan)
df['income_to_debt'] = df['income'] / df['debt'].clip(lower=1)
```

## 3. 分组聚合特征

```python
# 组内均值 / 组内排名
for agg_func in ['mean', 'std', 'median', 'count']:
    df[f'category_{agg_func}_price'] = df.groupby('category')['price'].transform(agg_func)

# 与组均值的差异
df['price_diff_from_cat_mean'] = df['price'] - df['category_mean_price']

# 组内排名
df['rank_in_category'] = df.groupby('category')['price'].rank(pct=True)
```

## 4. 差分与变化率

```python
# 一阶差分
df['value_diff'] = df.groupby('entity_id')['value'].diff()

# 变化率
df['value_pct_change'] = df.groupby('entity_id')['value'].pct_change()
```

## 5. 统计聚合特征

```python
num_cols = ['feature_a', 'feature_b', 'feature_c']

df['row_mean'] = df[num_cols].mean(axis=1)
df['row_std'] = df[num_cols].std(axis=1)
df['row_max'] = df[num_cols].max(axis=1)
df['row_min'] = df[num_cols].min(axis=1)
df['row_range'] = df['row_max'] - df['row_min']
df['row_skew'] = df[num_cols].skew(axis=1)
```

## 6. 计数与布尔聚合

```python
# 非零特征计数
df['nonzero_count'] = (df[num_cols] != 0).sum(axis=1)

# 缺失值计数
df['missing_count'] = df[num_cols].isnull().sum(axis=1)

# 满足条件的特征计数
df['high_value_count'] = (df[num_cols] > threshold).sum(axis=1)
```

## 特征交互的最佳实践

1. **基于领域知识**：优先创建业务上有意义的交互特征
2. **控制数量**：避免盲目创建所有两两组合，维度爆炸
3. **验证有效性**：通过特征重要性或相关性验证新特征是否有用
4. **注意泄露**：分组聚合需确保只使用训练集统计量
