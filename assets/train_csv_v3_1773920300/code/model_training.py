import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. 数据加载与基本配置
# ============================================

# 数据文件路径
DATA_PATH = '/Users/cjialin/code/AutoMLByLLM/train.csv'
MODEL_SAVE_PATH = '/Users/cjialin/code/AutoMLByLLM/train_model.pkl'

# 目标列名称
TARGET_COL = 'SalePrice'

# 基于提供信息定义的数值列（15个数值特征）
NUMERICAL_FEATURES = [
    'Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
    'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'
]

# 基于提供信息定义的分类列（15个分类特征）
CATEGORICAL_FEATURES = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl'
]

print("开始加载数据...")

# 加载数据集
df = pd.read_csv(DATA_PATH)
print(f"数据加载完成，形状: {df.shape}")
print(f"目标列: {TARGET_COL}")

# 检查并筛选实际存在的列
available_num_features = [col for col in NUMERICAL_FEATURES if col in df.columns]
available_cat_features = [col for col in CATEGORICAL_FEATURES if col in df.columns]

print(f"可用数值特征数量: {len(available_num_features)}")
print(f"可用分类特征数量: {len(available_cat_features)}")

# ============================================
# 2. 数据准备
# ============================================

# 分离特征和目标变量
X = df[available_num_features + available_cat_features].copy()
y = df[TARGET_COL].copy()

print(f"特征矩阵形状: {X.shape}")
print(f"目标向量形状: {y.shape}")

# ============================================
# 3. 构建预处理Pipeline
# ============================================

# 数值特征处理流程：填充缺失值（中位数）-> 标准化（Z-score）
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 分类特征处理流程：填充缺失值（众数）-> One-Hot编码
# handle_unknown='ignore' 确保测试集中出现训练集未见过的类别时不会报错
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

# 组合预处理流程：对数值和分类特征分别应用不同的转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, available_num_features),
        ('cat', categorical_transformer, available_cat_features)
    ],
    remainder='drop'
)

print("预处理Pipeline构建完成")

# ============================================
# 4. 构建完整模型Pipeline
# ============================================

# 使用GradientBoostingRegressor作为回归模型
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

print("模型Pipeline构建完成")

# ============================================
# 5. 数据划分
# ============================================

# 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"训练集大小: {X_train.shape[0]} 样本")
print(f"测试集大小: {X_test.shape[0]} 样本")

# ============================================
# 6. 模型训练
# ============================================

print("开始训练模型...")
model.fit(X_train, y_train)
print("模型训练完成")

# ============================================
# 7. 模型评估
# ============================================

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算回归评估指标
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("模型性能评估结果")
print("="*50)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print("="*50)

# ============================================
# 8. 保存模型
# ============================================

joblib.dump(model, MODEL_SAVE_PATH)
print(f"\n模型已保存到: {MODEL_SAVE_PATH}")

# 返回性能指标字典
performance_metrics = {
    'mae': float(mae),
    'mse': float(mse),
    'rmse': float(rmse),
    'r2_score': float(r2)
}

print("\n最终性能指标:")
print(performance_metrics)