import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# ==========================================
# 1. 数据加载与配置
# ==========================================
DATA_PATH = '/Users/cjialin/code/AutoMLByLLM/train.csv'
MODEL_PATH = '/Users/cjialin/code/AutoMLByLLM/train_model.pkl'
TARGET_COL = 'SalePrice'
ID_COL = 'Id'

print("正在加载数据...")
df = pd.read_csv(DATA_PATH)
print(f"数据加载完成，形状: {df.shape}")

# ==========================================
# 2. 特征列定义（基于用户提供的实际列名）
# ==========================================
# 数值列（用户明确列出的，注意：MSSubClass 实际为类别，移至分类列）
NUMERIC_FEATURES = [
    'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
    'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'
]

# 分类列（用户明确列出的，添加 MSSubClass）
CATEGORICAL_FEATURES = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
    'RoofStyle', 'RoofMatl', 'MSSubClass'
]

# 自动检测并补充数据集中存在的其他列
existing_cols = set(df.columns)
known_cols = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL, ID_COL])
remaining_cols = existing_cols - known_cols

# 根据数据类型自动分类剩余列
for col in remaining_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        NUMERIC_FEATURES.append(col)
    else:
        CATEGORICAL_FEATURES.append(col)

print(f"数值特征数量: {len(NUMERIC_FEATURES)}")
print(f"分类特征数量: {len(CATEGORICAL_FEATURES)}")

# ==========================================
# 3. 数据准备
# ==========================================
# 分离特征和目标变量
X = df.drop([TARGET_COL, ID_COL], axis=1)
y = df[TARGET_COL]

# 划分训练集和测试集（80/20 分割）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# ==========================================
# 4. 预处理管道构建
# ==========================================
# 数值特征处理：中位数填充缺失值 + 标准化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 分类特征处理：常数填充缺失值 + One-Hot 编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

# 组合预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ],
    remainder='drop'  # 丢弃未明确指定的列（如有）
)

# ==========================================
# 5. 模型定义（堆叠集成）
# ==========================================
# 基学习器 1：梯度提升（对应方案中的 LightGBM/XGBoost 替代品）
gbr = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=50,
    verbose=0
)

# 基学习器 2：随机森林
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# 基学习器 3：岭回归
ridge = Ridge(alpha=10.0, random_state=42)

# 堆叠集成配置
estimators = [
    ('gbr', gbr),
    ('rf', rf),
    ('ridge', ridge)
]

# 元学习器：使用 RidgeCV 自动选择最佳 alpha
final_estimator = RidgeCV(
    alphas=[0.1, 1.0, 10.0, 100.0, 1000.0],
    cv=5,
    scoring='neg_root_mean_squared_error'
)

stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5,
    passthrough=False,  # 不传递原始特征到元学习器，防止过拟合
    n_jobs=-1,
    verbose=0
)

# ==========================================
# 6. 完整管道构建（含目标变量对数变换）
# ==========================================
# 使用 TransformedTargetRegressor 对 SalePrice 进行 log1p 变换（处理右偏分布）
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', TransformedTargetRegressor(
        regressor=stacking_regressor,
        func=np.log1p,           # 训练前对目标取 log(1+x)
        inverse_func=np.expm1    # 预测后取 exp(x)-1 还原
    ))
])

# ==========================================
# 7. 模型训练
# ==========================================
print("\n开始训练堆叠集成模型（含目标变量对数变换）...")
print("基学习器: GradientBoosting + RandomForest + Ridge")
print("元学习器: RidgeCV")

model_pipeline.fit(X_train, y_train)
print("模型训练完成！")

# ==========================================
# 8. 模型评估
# ==========================================
print("\n正在评估模型性能...")

# 在测试集上预测
y_pred = model_pipeline.predict(X_test)

# 计算评估指标
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# 5折交叉验证评估（使用原始特征 X 和标签 y）
print("正在进行5折交叉验证（这可能需要几分钟）...")
cv_scores = cross_val_score(
    model_pipeline, X, y, 
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=0
)
cv_rmse = -cv_scores.mean()
cv_std = cv_scores.std()

# ==========================================
# 9. 性能指标汇总
# ==========================================
results = {
    'Test_RMSE': round(rmse, 2),
    'Test_R2': round(r2, 4),
    'Test_MAE': round(mae, 2),
    'Test_MAPE(%)': round(mape, 2),
    'CV_RMSE_Mean': round(cv_rmse, 2),
    'CV_RMSE_Std': round(cv_std, 2)
}

print("\n" + "="*50)
print("模型性能评估结果")
print("="*50)
print(f"测试集 RMSE:        ${results['Test_RMSE']:,.2f}")
print(f"测试集 R²:          {results['Test_R2']:.4f}")
print(f"测试集 MAE:         ${results['Test_MAE']:,.2f}")
print(f"测试集 MAPE:        {results['Test_MAPE(%)']:.2f}%")
print(f"5折交叉验证 RMSE:   ${results['CV_RMSE_Mean']:,.2f} (±{results['CV_RMSE_Std']:,.2f})")
print("="*50)

# ==========================================
# 10. 保存模型
# ==========================================
print(f"\n正在保存模型到: {MODEL_PATH}")
joblib.dump(model_pipeline, MODEL_PATH)
print("模型保存成功！")

# 输出最终性能指标字典
print("\n返回性能指标:")
print(results)