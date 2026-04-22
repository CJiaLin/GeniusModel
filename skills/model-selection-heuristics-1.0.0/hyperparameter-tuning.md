# 超参数调优策略

## 方法对比

| 方法 | 搜索效率 | 实现复杂度 | 适用场景 |
|------|---------|-----------|---------|
| GridSearchCV | 低（穷举） | 低 | 参数空间小（< 100 组合） |
| RandomizedSearchCV | 中 | 低 | 参数空间中等 |
| Bayesian (Optuna) | 高 | 中 | 参数空间大、计算昂贵 |
| Halving | 高 | 低 | 大数据集、快速筛选 |

## 1. Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1_weighted',      # 或 'roc_auc', 'neg_mean_squared_error'
    cv=5,
    n_jobs=-1,
    verbose=1,
    refit=True                  # 自动用最佳参数在全量数据上重训
)
grid.fit(X_train, y_train)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳分数: {grid.best_score_:.4f}")
best_model = grid.best_estimator_
```

## 2. Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=50,                  # 随机采样 50 组
    scoring='f1_weighted',
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search.fit(X_train, y_train)
```

**优势：** 相同计算预算下，RandomSearch 通常优于 GridSearch（Bergstra & Bengio, 2012）。

## 3. Bayesian Optimization (Optuna)

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    model = XGBClassifier(**params, random_state=42, verbosity=0, n_jobs=-1)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"最佳参数: {study.best_params}")
print(f"最佳分数: {study.best_value:.4f}")
```

**优势：** 智能探索参数空间，效率远高于网格/随机搜索。

## 4. Halving Search（逐步淘汰）

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

halving = HalvingRandomSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_candidates=100,           # 初始候选数
    factor=3,                   # 每轮淘汰 2/3
    resource='n_samples',       # 逐步增加样本量
    scoring='f1_weighted',
    cv=5,
    random_state=42,
    n_jobs=-1
)
halving.fit(X_train, y_train)
```

**原理：** 先用少量数据评估大量候选，逐步增加数据量并淘汰表现差的组合。

## 5. 早停 (Early Stopping)

```python
# XGBoost 早停
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2)

model = XGBClassifier(
    n_estimators=1000,          # 设较大值
    learning_rate=0.1,
    random_state=42
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print(f"最佳迭代次数: {model.best_iteration}")

# LightGBM 早停
from lightgbm import LGBMClassifier, early_stopping

lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.1, verbose=-1)
lgbm.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(stopping_rounds=50)]
)
```

## 常用参数搜索范围参考

### XGBoost / LightGBM

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| n_estimators | 100 ~ 1000 | 配合早停 |
| max_depth | 3 ~ 10 | 越深越容易过拟合 |
| learning_rate | 0.01 ~ 0.3 | 越小需要更多树 |
| subsample | 0.6 ~ 1.0 | 行抽样比例 |
| colsample_bytree | 0.6 ~ 1.0 | 列抽样比例 |
| reg_alpha | 1e-8 ~ 10 | L1 正则 |
| reg_lambda | 1e-8 ~ 10 | L2 正则 |
| min_child_weight | 1 ~ 10 | 叶节点最小权重 |

### 调参经验

1. **先固定 learning_rate=0.1**，调其他参数
2. **树结构参数** (max_depth, min_child_weight) 优先调
3. **正则化参数** (reg_alpha, reg_lambda) 其次
4. **采样参数** (subsample, colsample_bytree) 最后
5. **最后降低 learning_rate**，增加 n_estimators，配合早停
