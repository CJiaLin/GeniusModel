# AutoML 建模流程报告

**生成时间**: 2026-03-19 18:07:54
**会话ID**: train_csv_v3_1773914239

---

## 1. 数据概览

- **数据路径**: `/Users/cjialin/code/AutoMLByLLM/train.csv`
- **目标列**: `SalePrice`
- **任务类型**: regression

## 2. 数据清洗

⚠️ **清洗状态**: 跳过或失败

## 3. 特征工程

⚠️ **特征工程状态**: 跳过或失败

## 4. 模型训练

❌ **训练状态**: 失败
- **错误**: 未知错误

## 5. 模型评估

⚠️ **评估状态**: 未执行或失败

## 6. 可视化图表

### 6.1 数据分布

```python
# 数据分布可视化代码
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 数值特征分布
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} 分布')
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[col])
    plt.title(f'{col} 箱线图')
    plt.tight_layout()
    plt.savefig(f'distribution_{col}.png')
    plt.close()
```

### 6.2 特征重要性

```python
# 特征重要性可视化
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_names = X.columns
    
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1][:20]  # Top 20
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.title('Top 20 特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
```

### 6.3 模型评估图表

```python
# 混淆矩阵（分类任务）
if task_type == 'classification':
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.close()

# ROC 曲线（二分类）
if task_type == 'classification' and len(np.unique(y)) == 2:
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC 曲线')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.close()

# 残差图（回归任务）
if task_type == 'regression':
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差图')
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.title('残差分布')
    plt.tight_layout()
    plt.savefig('residuals.png')
    plt.close()
```

## 7. 结论与建议

### 7.1 模型性能总结

### 7.2 改进建议

1. **数据层面**: 考虑收集更多数据或进行数据增强
2. **特征层面**: 尝试更多特征组合和特征选择方法
3. **模型层面**: 尝试集成模型或深度学习模型
4. **调优层面**: 进行超参数调优以提升模型性能

---

*报告由 AutoML 系统自动生成*