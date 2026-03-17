"""
AutoML 完整建模流程代码
目标列: SalePrice
生成时间: 2026-03-16T19:16:34.887790
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib

# ========== 1. 数据加载 ==========
df = pd.read_csv('your_data.csv')
print(f"原始数据形状: {df.shape}")

# ========== 2. 数据清洗 ==========
# 缺失值处理
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print(f"清洗后数据形状: {df.shape}")

# ========== 3. 特征工程 ==========
# 创建新特征
if 'GrLivArea' in df.columns and 'LotArea' in df.columns:
    df['GrLivArea_LotArea_ratio'] = df['GrLivArea'] / (df['LotArea'] + 1)

if '1stFlrSF' in df.columns and '2ndFlrSF' in df.columns:
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF']

if 'FullBath' in df.columns and 'HalfBath' in df.columns:
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']

if 'YearBuilt' in df.columns:
    df['Age'] = 2024 - df['YearBuilt']

# 类别特征编码
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() < 15:
        df[col] = pd.factorize(df[col])[0]

print(f"特征工程后数据形状: {df.shape}")

# ========== 4. 模型训练 ==========
X = df.drop(columns=['SalePrice', 'Id'] if 'Id' in df.columns else ['SalePrice'])
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n模型评估结果:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# ========== 5. 保存模型 ==========
joblib.dump(model, 'model_SalePrice.joblib')
print("\n模型已保存至: model_SalePrice.joblib")
