# 回归模型选择指南

## 快速选择矩阵

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 基线 / 可解释性 | LinearRegression / Ridge | 简单、快速 |
| 特征多、需特征选择 | Lasso / ElasticNet | L1 正则自动稀疏化 |
| 非线性关系 | RandomForestRegressor | 无需手动构造非线性特征 |
| 追求最高精度 | XGBRegressor / LGBMRegressor | 表格数据通常最优 |
| 有异常值 | HuberRegressor / RANSAC | 对异常值鲁棒 |

## 1. 线性回归与正则化

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# 基线
lr = LinearRegression()

# Ridge (L2 正则) - 处理多重共线性
ridge = Ridge(alpha=1.0)

# Lasso (L1 正则) - 特征选择
lasso = Lasso(alpha=0.1)

# ElasticNet (L1 + L2) - 兼顾两者
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

**选择指南：**
- 特征间高度相关 → Ridge
- 需要特征选择 / 稀疏解 → Lasso
- 两者兼需 → ElasticNet

## 2. 鲁棒回归

```python
from sklearn.linear_model import HuberRegressor, RANSACRegressor

# Huber 回归 - 对异常值鲁棒
huber = HuberRegressor(epsilon=1.35)

# RANSAC - 忽略异常值子集
ransac = RANSACRegressor(random_state=42)
```

**适用：** 数据中存在异常值或离群点。

## 3. KNN 回归

```python
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',   # 'uniform' 或 'distance'
    metric='minkowski'
)
```

**注意：** 必须先对特征做标准化，否则距离计算被大数值特征主导。

## 4. 随机森林回归

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
```

## 5. XGBoost / LightGBM 回归

```python
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

xgb_reg = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

lgbm_reg = LGBMRegressor(
    n_estimators=300,
    max_depth=-1,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
```

## 6. 回归评估指标

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

# MSE / RMSE
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)

# MAE
mae = mean_absolute_error(y_true, y_pred)

# R²
r2 = r2_score(y_true, y_pred)

# MAPE（目标值不含零时使用）
mape = mean_absolute_percentage_error(y_true, y_pred)
```

**指标选择：**
- 对大误差敏感 → RMSE
- 需要可解释的绝对误差 → MAE
- 需要比例误差 → MAPE
- 需要整体拟合度 → R²
