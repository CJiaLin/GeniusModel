```python
"""
房价预测回归模型训练脚本
任务类型: 回归 (Regression)
目标列: SalePrice
使用模型: Random Forest + Gradient Boosting (Scikit-Learn)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==========================================
# 1. 数据加载与探索
# ==========================================
def load_and_explore_data(file_path):
    """
    加载数据并进行初步探索
    """
    print("=" * 50)
    print("步骤 1: 数据加载与探索")
    print("=" * 50)
    
    # 加载数据
    df = pd.read_csv(file_path)
    print(f"数据形状: {df.shape}")
    print(f"\n列名: {list(df.columns)}")
    print(f"\n前5行预览:\n{df.head()}")
    
    # 目标变量统计
    print(f"\n目标变量 SalePrice 统计信息:")
    print(df['SalePrice'].describe())
    
    # 检查缺失值
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(f"\n缺失值统计 (前10):\n{missing.head(10)}")
    else:
        print("\n无缺失值")
    
    return df

# ==========================================
# 2. 数据预处理
# ==========================================
def preprocess_data(df, target_col='SalePrice'):
    """
    数据预处理：处理缺失值、编码分类变量、特征工程
    """
    print("\n" + "=" * 50)
    print("步骤 2: 数据预处理")
    print("=" * 50)
    
    # 分离特征和目标变量
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    
    # 对目标变量进行对数变换（处理右偏分布）
    print("对目标变量进行对数变换...")
    y_log = np.log1p(y)
    
    # 识别数值型和类别型特征
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"数值型特征数量: {len(numeric_features)}")
    print(f"类别型特征数量: {len(categorical_features)}")
    
    # 处理数值型特征的缺失值（用中位数填充）
    numeric_imputer = SimpleImputer(strategy='median')
    X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
    
    # 处理类别型特征的缺失值（用"Missing"填充）并进行标签编码
    X[categorical_features] = X[categorical_features].fillna('Missing')
    
    # 对类别型特征进行标签编码
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    print(f"预处理完成，特征矩阵形状: {X.shape}")
    
    return X, y_log, label_encoders, numeric_imputer

# ==========================================
# 3. 特征工程（可选的高级特征）
# ==========================================
def feature_engineering(X):
    """
    创建新特征以提高模型性能
    """
    print("\n" + "=" * 50)
    print("步骤 3: 特征工程")
    print("=" * 50)
    
    X_engineered = X.copy()
    
    # 创建总面积特征（如果有相关特征）
    area_cols = [col for col in X.columns if 'Area' in col or 'SF' in col]
    if len(area_cols) > 1:
        X_engineered['TotalArea'] = X[area_cols].sum(axis=1)
        print(f"创建 TotalArea 特征，基于: {area_cols[:3]}...")
    
    # 创建房屋年龄特征（如果有 YearBuilt）
    if 'YearBuilt' in X.columns and 'YrSold' in X.columns:
        X_engineered['HouseAge'] = X['YrSold'] - X['YearBuilt']
        X_engineered['IsNew'] = (X_engineered['HouseAge'] <= 1).astype(int)
        print("创建 HouseAge 和 IsNew 特征")
    
    # 创建质量得分组合特征（如果有相关质量特征）
    quality_cols = [col for col in X.columns if 'Qual' in col or 'Cond' in col]
    if len(quality_cols) > 1:
        X_engineered['OverallQuality'] = X[quality_cols].mean(axis=1)
        print(f"创建 OverallQuality 特征，基于: {quality_cols[:3]}...")
    
    print(f"特征工程后形状: {X_engineered.shape}")
    return X_engineered

# ==========================================
# 4. 数据划分
# ==========================================
def split_data(X, y, test_size=0.2):
    """
    划分训练集和测试集
    """
    print("\n" + "=" * 50)
    print("步骤 4: 数据划分")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"特征数量: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

# ==========================================
# 5. 模型训练
# ==========================================
def train_models(X_train, y_train):
    """
    训练多个回归模型
    """
    print("\n" + "=" * 50)
    print("步骤 5: 模型训练")
    print("=" * 50)
    
    models = {}
    
    # 模型 1: Random Forest
    print("\n训练 Random Forest...")
    rf_params = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    print("Random Forest 训练完成")
    
    # 模型 2: Gradient Boosting
    print("\n训练 Gradient Boosting...")
    gb_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 4,
        'min_samples_split': 5,
        'min_samples_leaf': 3,
        'subsample': 0.8,
        'random_state': RANDOM_STATE
    }
    gb_model = GradientBoostingRegressor(**gb_params)
    gb_model.fit(X_train, y_train)
    models['GradientBoosting'] = gb_model
    print("Gradient Boosting 训练完成")
    
    # 交叉验证评估
    print("\n交叉验证评估 (5折):")
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        print(f"{name}: RMSE = {rmse_scores.mean():.4f} (+/- {rmse_scores.std()*2:.4f})")
    
    return models

# ==========================================
# 6. 模型评估
# ==========================================
def evaluate_models(models, X_test, y_test):
    """
    评估模型性能（在对数空间和平凡空间）
    """
    print("\n" + "=" * 50)
    print("步骤 6: 模型评估")
    print("=" * 50)
    
    results = {}
    
    for name, model in models.items():
        # 预测（对数空间）
        y_pred_log = model.predict(X_test)
        
        # 转换回原始空间
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)
        
        # 计算指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MSE': mse
        }
        
        print(f"\n{name} 性能:")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE: {mae:,.2f}")
        print(f"  R²: {r2:.4f}")
    
    # 选择最佳模型（基于 RMSE）
    best_model_name = min(results, key=lambda x: results[x]['RMSE'])
    print(f"\n最佳模型: {best_model_name}")
    print(f"最佳 RMSE: {results[best_model_name]['RMSE']:,.2f}")
    
    return results, best_model_name

# ==========================================
# 7. 特征重要性分析
# ==========================================
def analyze_feature_importance(model, feature_names, top_n=20):
    """
    分析特征重要性
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        print(f"\n前 {top_n} 个重要特征:")
        for i in range(top_n):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 8))
        plt.title(f"Top {top_n} Feature Importances")
        plt.barh(range(top_n), importances[indices], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n特征重要性图已保存为 'feature_importance.png'")
        plt.close()

# ==========================================
# 8. 保存模型
# ==========================================
def save_model(model, model_name, save_dir='./models'):
    """
    保存训练好的模型和相关对象
    """
    print("\n" + "=" * 50)
    print("步骤 8: 保存模型")
    print("=" * 50)
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建目录: {save_dir}")
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.pkl"
    model_path = os.path.join(save_dir, model_filename)
    
    # 保存模型
    joblib.dump(model, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存模型信息
    info = {
        'model_name': model_name,
        'timestamp': timestamp,
        'scikit_learn_version': joblib.__version__,
        'model_path': model_path
    }
    
    info_path = os.path.join(save_dir, f"model_info_{timestamp}.txt")
    with open(info_path, 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    return model_path

# ==========================================
# 9. 主函数
# ==========================================
def main():
    """
    主执行函数
    """
    # 配置
    DATA_PATH = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    TARGET_COL = 'SalePrice'
    MODEL_SAVE_DIR = './trained_models'
    
    print("开始房价预测模型训练流程...")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 加载数据
        df = load_and_explore_data(DATA_PATH)
        
        # 2. 预处理
        X, y_log, label_encoders, imputer = preprocess_data(df, TARGET_COL)
        
        # 3. 特征工程
        X = feature_engineering(X)
        
        # 4. 划分数据
        X_train, X_test, y_train, y_test = split_data(X, y_log)
        
        # 5. 训练模型
        models = train_models(X_train, y_train)
        
        # 6. 评估模型
        results, best_model_name = evaluate_models(models, X_test, y_test)
        
        # 7. 特征重要性分析（使用最佳模型）
        best_model = models[best_model_name]
        analyze_feature_importance(best_model, X.columns.tolist())
        
        # 8. 保存最佳模型
        model_path = save_model(best_model, best_model_name, MODEL_SAVE_DIR)
        
        # 9. 保存预处理对象（用于后续推理）
        preprocessing_objects = {
            'label_encoders': label_encoders,
            'imputer': imputer,
            'feature_names': X.columns.tolist()
        }
        prep_path = os.path.join(MODEL_SAVE_DIR, f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        joblib.dump(preprocessing_objects, prep_path)
        print(f"预处理对象已保存到: {prep_path}")
        
        # 10. 输出总结
        print("\n" + "=" * 50)
        print("训练完成总结")
        print("=" * 50)
        print(f"最佳模型: {best_model_name}")
        print(f"测试集 RMSE: {results[best_model_name]['RMSE']:,.2f}")
        print(f"测试集 R²: {results[best