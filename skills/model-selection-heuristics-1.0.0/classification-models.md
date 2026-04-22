# 分类模型选择指南

## 快速选择矩阵

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 基线 / 可解释性要求高 | LogisticRegression | 简单、快速、概率输出 |
| 高维稀疏特征（文本） | LogisticRegression / LinearSVC | 高维下仍高效 |
| 中小数据集、非线性 | RandomForest | 无需太多调参、鲁棒 |
| 追求最高精度 | XGBoost / LightGBM | GBDT 在表格数据上通常最优 |
| 大规模数据 + GPU | LightGBM / CatBoost | 训练速度快，原生类别支持 |
| 类别不平衡 | XGBoost (scale_pos_weight) / SMOTE + 任意模型 | 内置不平衡处理 |

## 1. 逻辑回归 (Logistic Regression)

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,               # 正则化强度（越小越强）
    penalty='l2',        # 'l1' 可做特征选择
    solver='lbfgs',      # 大数据集用 'saga'
    max_iter=1000,
    class_weight='balanced',  # 类别不平衡时使用
    random_state=42
)
```

**适用：** 线性可分、需要概率输出、特征维度高（文本分类）、基线模型。

## 2. 支持向量机 (SVM)

```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',        # 'linear' 高维, 'rbf' 非线性, 'poly' 多项式
    C=1.0,
    gamma='scale',       # 'auto' 或具体值
    probability=True,    # 需要概率输出时开启（会变慢）
    class_weight='balanced',
    random_state=42
)
```

**适用：** 中小数据集、高维数据、二分类。
**不适用：** 样本量 > 10 万（训练太慢）。

## 3. 随机森林 (Random Forest)

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,       # None 不限制（通常效果好）
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',  # 每棵树随机选 sqrt(n) 个特征
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=42
)
```

**适用：** 通用、无需太多特征预处理、需要特征重要性。

## 4. XGBoost

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,        # L1 正则
    reg_lambda=1.0,       # L2 正则
    scale_pos_weight=1,   # 正负样本比例（不平衡时调整）
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
```

## 5. LightGBM

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=300,
    max_depth=-1,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    is_unbalance=True,    # 类别不平衡
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
```

**优势：** 比 XGBoost 更快、内存更少、原生支持类别特征。

## 6. CatBoost

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    l2_leaf_reg=3.0,
    auto_class_weights='Balanced',
    cat_features=cat_col_indices,  # 直接传入类别列索引
    verbose=0,
    random_state=42
)
```

**优势：** 原生类别特征处理、无需 OneHot、对有序类别表现好。

## 模型选择决策流

1. **先跑基线：** LogisticRegression + 简单特征
2. **特征多且非线性：** RandomForest（快速获得不错的结果）
3. **追求精度：** XGBoost / LightGBM + 超参数调优
4. **有大量类别特征：** CatBoost
5. **比较多个模型，选交叉验证分数最高的**
