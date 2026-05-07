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

1. **基于领域知识优先**：优先创建业务上有意义的交互特征
2. **系统性覆盖**：不要只挑 1-2 对，对所有与目标相关的数值列系统性地创建比率、差值和乘积特征
3. **分组聚合是高价值来源**：按每个有业务含义的分类列做 groupby 统计（mean/std/count），这类特征通常信息增益大
4. **先生成再筛选**：宁可多生成交互特征，后续通过特征重要性筛选无效特征，不要在生成阶段过于保守
5. **注意泄露**：分组聚合需确保只使用训练集统计量
6. **行级统计不要遗漏**：row_mean/row_std/row_max/row_min/nonzero_count 等往往是强特征

## 7. 系统性交叉策略

当数据有 K 个数值列和 M 个分类列时，推荐的交叉生成策略：

```python
import itertools

# 策略 A：数值×数值（选择与目标相关性 top-K 的列做两两交叉）
top_num_cols = corr_with_target.abs().nlargest(8).index.tolist()
for col1, col2 in itertools.combinations(top_num_cols, 2):
    df[f'{col1}_div_{col2}'] = df[col1] / df[col2].clip(lower=1e-8)
    df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
    df[f'{col1}_mul_{col2}'] = df[col1] * df[col2]

# 策略 B：分类×数值（每个分类列对 top 数值列做分组聚合）
for cat_col in cat_cols:
    for num_col in top_num_cols[:5]:
        grp = df.groupby(cat_col)[num_col]
        df[f'{cat_col}_{num_col}_mean'] = grp.transform('mean')
        df[f'{cat_col}_{num_col}_std'] = grp.transform('std')
        # 与组均值的偏差（强特征）
        df[f'{cat_col}_{num_col}_diff'] = df[num_col] - df[f'{cat_col}_{num_col}_mean']

# 策略 C：分类×分类（组合后频率编码）
for cat1, cat2 in itertools.combinations(cat_cols[:5], 2):
    combined = df[cat1].astype(str) + '_' + df[cat2].astype(str)
    freq = combined.value_counts(normalize=True)
    df[f'{cat1}_{cat2}_freq'] = combined.map(freq)
```

### 生成数量参考

| 原始特征数 | 推荐新增特征数 | 覆盖策略 |
|-----------|--------------|---------|
| 5~10 | 15~30 | 全量两两交叉 + 分组聚合 + 行统计 |
| 10~30 | 30~60 | Top-K 交叉 + 全分类聚合 + 行统计 + 分箱 |
| 30~100 | 50~100 | Top-K 交叉 + 选择性聚合 + 行统计 + 编码 |
| 100+ | 80~150 | 分组主题 + 选择性交叉 + PCA 降维后交叉 |
