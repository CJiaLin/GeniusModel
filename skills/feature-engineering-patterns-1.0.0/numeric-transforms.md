# 数值特征变换

## 1. 对数变换 (Log Transform)

**适用场景：** 右偏分布的数值特征（如收入、金额、面积）

```python
import numpy as np

# log1p 变换（处理零值）
df['feature_log'] = np.log1p(df['feature'])

# 逆变换
# original = np.expm1(df['feature_log'])
```

**注意：** 负值无法取对数，需先偏移。

## 2. 分箱 (Binning)

**适用场景：** 非线性关系、减少异常值影响

```python
# 等距分箱
df['age_bin'] = pd.cut(df['age'], bins=5, labels=False)

# 等频分箱
df['income_qbin'] = pd.qcut(df['income'], q=5, labels=False, duplicates='drop')

# 业务规则分箱
bins = [0, 18, 35, 55, 100]
labels = ['youth', 'young_adult', 'middle', 'senior']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
```

## 3. 标准化与归一化

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: 均值=0, 方差=1（适合高斯分布）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[num_cols])
X_test_scaled = scaler.transform(X_test[num_cols])  # 只 transform，不 fit

# RobustScaler: 对异常值更鲁棒
robust_scaler = RobustScaler()

# MinMaxScaler: 映射到 [0, 1]（适合有界特征）
minmax_scaler = MinMaxScaler()
```

**关键原则：** 必须在训练集上 `fit`，在测试集上仅 `transform`。

## 4. 多项式特征

```python
from sklearn.preprocessing import PolynomialFeatures

# 二阶多项式（含交互项）
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X_train[['feature_a', 'feature_b']])
```

**注意：** 高阶多项式容易过拟合，建议配合正则化使用。

## 5. 幂变换 (Power Transform)

```python
from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson 变换（支持负值）
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X_train[num_cols])
```

## 6. 缺失值指示器

```python
# 为有缺失的列创建二值指示特征
for col in cols_with_missing:
    df[f'{col}_is_missing'] = df[col].isnull().astype(int)
```

有时"是否缺失"本身就是一个有价值的信号。
