# 类别特征编码

## 1. One-Hot Encoding

**适用场景：** 低基数类别特征（<= 10~20 个唯一值）

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded = ohe.fit_transform(X_train[cat_cols])
# 测试集: ohe.transform(X_test[cat_cols])
```

**注意：** `handle_unknown='ignore'` 可处理测试集中出现训练集未见过的类别。

## 2. Label Encoding / Ordinal Encoding

**适用场景：** 有序类别（如教育水平、评分等级）

```python
from sklearn.preprocessing import OrdinalEncoder

categories = [['low', 'medium', 'high']]  # 指定顺序
oe = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
```

## 3. Target Encoding

**适用场景：** 高基数类别特征（如城市、邮编）

```python
from sklearn.model_selection import KFold
import numpy as np

def target_encode(train, test, col, target, n_splits=5):
    """K-Fold Target Encoding 防止数据泄露"""
    train[f'{col}_te'] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(train):
        means = train.iloc[train_idx].groupby(col)[target].mean()
        train.iloc[val_idx, train.columns.get_loc(f'{col}_te')] = \
            train.iloc[val_idx][col].map(means)

    global_mean = train[target].mean()
    train[f'{col}_te'] = train[f'{col}_te'].fillna(global_mean)

    # 测试集使用训练集全量均值
    full_means = train.groupby(col)[target].mean()
    test[f'{col}_te'] = test[col].map(full_means).fillna(global_mean)

    return train, test
```

**关键：** 必须使用 K-Fold 方式防止目标泄露。

## 4. 频率编码 (Frequency Encoding)

**适用场景：** 类别频率本身包含信息

```python
freq = X_train[col].value_counts(normalize=True)
X_train[f'{col}_freq'] = X_train[col].map(freq)
X_test[f'{col}_freq'] = X_test[col].map(freq).fillna(0)
```

## 5. 二值编码 (Binary Encoding)

**适用场景：** 中等基数类别特征（10~100 个唯一值），比 OneHot 节省维度

```python
import category_encoders as ce

binary_enc = ce.BinaryEncoder(cols=cat_cols)
X_train_encoded = binary_enc.fit_transform(X_train)
X_test_encoded = binary_enc.transform(X_test)
```

## 选择指南

| 方法 | 基数范围 | 优点 | 缺点 |
|------|---------|------|------|
| OneHot | <= 20 | 简单、无信息损失 | 高维、稀疏 |
| Ordinal | 有序类别 | 紧凑 | 引入隐含序关系 |
| Target | > 20 | 保留目标相关性 | 需防泄露 |
| Frequency | 任意 | 简单、保留分布 | 不同类别可能同频 |
| Binary | 10~100 | 折衷维度 | 需额外库 |
