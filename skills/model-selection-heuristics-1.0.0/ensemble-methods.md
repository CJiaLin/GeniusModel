# 集成学习方法

## 概览

| 方法 | 代表算法 | 核心思想 | 适用场景 |
|------|---------|---------|---------|
| Bagging | RandomForest | 并行训练 + 投票/平均 | 减小方差 |
| Boosting | XGBoost, LightGBM, CatBoost | 串行纠错 | 减小偏差 |
| Stacking | StackingClassifier | 元学习器融合 | 最大化精度 |

## 1. Bagging

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=50,
    max_samples=0.8,      # 每棵树使用 80% 的样本
    max_features=0.8,     # 每棵树使用 80% 的特征
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
```

**原理：** 通过 Bootstrap 抽样训练多个模型，预测时投票（分类）或平均（回归）。
**效果：** 降低方差，对过拟合的模型特别有效。

## 2. Boosting

### 2.1 AdaBoost

```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.1,
    algorithm='SAMME.R',
    random_state=42
)
```

### 2.2 Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=3,          # Boosting 通常用浅树
    learning_rate=0.1,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42
)
```

### 2.3 GBDT 三剑客对比

| 特性 | XGBoost | LightGBM | CatBoost |
|------|---------|----------|----------|
| 树生长方式 | Level-wise | Leaf-wise | Symmetric |
| 类别特征 | 需编码 | 原生支持 | 原生支持（最优） |
| 训练速度 | 中 | 快 | 慢（但无需预处理） |
| 内存占用 | 中 | 低 | 高 |
| GPU 支持 | 是 | 是 | 是 |
| 缺失值处理 | 原生 | 原生 | 原生 |
| 默认效果 | 需调参 | 较好 | 开箱即用最好 |

## 3. Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42, verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
    ],
    final_estimator=LogisticRegression(),  # 元学习器
    cv=5,                 # 使用 5-fold 生成元特征
    stack_method='predict_proba',
    n_jobs=-1
)
```

**关键点：**
- 基模型应该**多样化**（不同类型的模型）
- 元学习器通常用简单模型（如 LogisticRegression）
- `cv` 参数防止数据泄露

## 4. Voting

```python
from sklearn.ensemble import VotingClassifier

# 软投票（使用概率平均，通常更好）
voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', XGBClassifier(n_estimators=100, verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=100, verbose=-1)),
    ],
    voting='soft',        # 'hard' 用多数投票, 'soft' 用概率平均
    weights=[1, 2, 2],    # 可选：给不同模型不同权重
    n_jobs=-1
)
```

## 选择建议

1. **单模型足够好** → 不需要集成
2. **模型过拟合** → Bagging（降方差）
3. **模型欠拟合** → Boosting（降偏差）
4. **竞赛 / 追求极致精度** → Stacking
5. **快速提升且代码简单** → Voting
