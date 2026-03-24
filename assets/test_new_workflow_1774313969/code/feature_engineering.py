import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 配置路径
DATA_PATH = '/Users/cjialin/code/AutoMLByLLM/train.csv'
OUTPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/assets/test_new_workflow_1774313969/data/features_data.csv'
TARGET_COL = 'SalePrice'

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 加载数据
print("正在加载数据...")
df = pd.read_csv(DATA_PATH)
print(f"原始数据形状: {df.shape}")

# 记录新生成的特征
new_features = []

# ==================== 1. 缺失值处理 ====================
print("处理缺失值...")

# 数值型列：中位数填充
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove(TARGET_COL) if TARGET_COL in numeric_cols else None
if 'Id' in numeric_cols:
    numeric_cols.remove('Id')

# 对于可能表示"无"的分类特征，填充为"None"
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 
             'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']

for col in none_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

# 数值型缺失值填充（中位数）
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  数值列 {col}: 使用中位数 {median_val:.2f} 填充")

# 分类型缺失值填充（众数）
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"  分类列 {col}: 使用众数 {mode_val} 填充")

# ==================== 2. 目标变量转换 ====================
print("转换目标变量...")
df[f'{TARGET_COL}_log'] = np.log1p(df[TARGET_COL])
new_features.append(f'{TARGET_COL}_log')

# ==================== 3. 面积聚合特征 ====================
print("创建面积聚合特征...")

# 总面积特征
if all(col in df.columns for col in ['TotalBsmtSF', 'GrLivArea', 'GarageArea']):
    df['Total_SF'] = df['TotalBsmtSF'] + df['GrLivArea'] + df['GarageArea']
    new_features.append('Total_SF')
    print("  创建 Total_SF")

# 生活面积占比
if all(col in df.columns for col in ['GrLivArea', 'LotArea']):
    df['LivingArea_Ratio'] = df['GrLivArea'] / (df['LotArea'] + 1)  # +1 避免除零
    new_features.append('LivingArea_Ratio')
    print("  创建 LivingArea_Ratio")

# 每房间面积
if all(col in df.columns for col in ['GrLivArea', 'TotRmsAbvGrd']):
    df['Area_Per_Room'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)
    new_features.append('Area_Per_Room')
    print("  创建 Area_Per_Room")

# ==================== 4. 浴室聚合特征 ====================
print("创建浴室聚合特征...")

bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
if all(col in df.columns for col in bath_cols):
    df['Total_Bathrooms'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                             df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    new_features.append('Total_Bathrooms')
    print("  创建 Total_Bathrooms")

# ==================== 5. 年龄与状态特征 ====================
print("创建年龄与状态特征...")

if all(col in df.columns for col in ['YrSold', 'YearBuilt']):
    df['House_Age'] = df['YrSold'] - df['YearBuilt']
    new_features.append('House_Age')
    print("  创建 House_Age")

if all(col in df.columns for col in ['YrSold', 'YearRemodAdd']):
    df['Years_Since_Remod'] = df['YrSold'] - df['YearRemodAdd']
    new_features.append('Years_Since_Remod')
    print("  创建 Years_Since_Remod")

if all(col in df.columns for col in ['YearRemodAdd', 'YearBuilt']):
    df['Was_Remodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    new_features.append('Was_Remodeled')
    print("  创建 Was_Remodeled")

# ==================== 6. 质量综合特征 ====================
print("创建质量综合特征...")

if all(col in df.columns for col in ['OverallQual', 'OverallCond']):
    df['Quality_Score'] = df['OverallQual'] * df['OverallCond']
    new_features.append('Quality_Score')
    print("  创建 Quality_Score")

# 外观质量等级映射
if 'ExterQual' in df.columns:
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    df['ExterQual_Numeric'] = df['ExterQual'].map(qual_map).fillna(3)
    new_features.append('ExterQual_Numeric')
    print("  创建 ExterQual_Numeric")

# ==================== 7. 高阶交互特征 ====================
print("创建高阶交互特征...")

if all(col in df.columns for col in ['OverallQual', 'GrLivArea']):
    df['Qual_LivArea'] = df['OverallQual'] * df['GrLivArea']
    new_features.append('Qual_LivArea')
    print("  创建 Qual_LivArea")

# 位置价值（Target Encoding）
if 'Neighborhood' in df.columns:
    neighborhood_mean = df.groupby('Neighborhood')[TARGET_COL].mean()
    df['Neighborhood_Price_Mean'] = df['Neighborhood'].map(neighborhood_mean)
    new_features.append('Neighborhood_Price_Mean')
    print("  创建 Neighborhood_Price_Mean")

# ==================== 8. 其他重要特征 ====================
print("创建其他业务特征...")

# 是否为新房（建造年份等于销售年份）
if all(col in df.columns for col in ['YearBuilt', 'YrSold']):
    df['Is_New_House'] = (df['YearBuilt'] == df['YrSold']).astype(int)
    new_features.append('Is_New_House')
    print("  创建 Is_New_House")

# 车库与房屋比率
if all(col in df.columns for col in ['GarageArea', 'GrLivArea']):
    df['Garage_Living_Ratio'] = df['GarageArea'] / (df['GrLivArea'] + 1)
    new_features.append('Garage_Living_Ratio')
    print("  创建 Garage_Living_Ratio")

# 总门廊面积
porch_cols = ['OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF']
available_porch_cols = [col for col in porch_cols if col in df.columns]
if available_porch_cols:
    df['Total_Porch_SF'] = df[available_porch_cols].sum(axis=1)
    new_features.append('Total_Porch_SF')
    print("  创建 Total_Porch_SF")

# ==================== 9. 数值特征对数变换 ====================
print("对右偏数值特征进行对数变换...")

# 识别右偏的数值特征（偏度 > 0.75）
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [col for col in numeric_features if col not in [TARGET_COL, f'{TARGET_COL}_log', 'Id']]

skewed_features = []
for col in numeric_features:
    if df[col].min() >= 0:  # 确保非负才能进行log1p
        skewness = df[col].skew()
        if skewness > 0.75:
            skewed_features.append(col)

print(f"  对 {len(skewed_features)} 个特征进行对数变换")
for col in skewed_features[:5]:  # 限制数量避免特征爆炸
    new_col_name = f'{col}_log'
    df[new_col_name] = np.log1p(df[col])
    new_features.append(new_col_name)

# ==================== 10. 多项式特征 ====================
print("创建多项式特征...")

# 对重要特征创建二阶多项式
important_num_cols = ['GrLivArea', 'OverallQual', 'TotalBsmtSF']
available_important_cols = [col for col in important_num_cols if col in df.columns]

if len(available_important_cols) >= 2:
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = poly.fit_transform(df[available_important_cols])
    poly_feature_names = poly.get_feature_names_out(available_important_cols)
    
    # 只保留交互项和平方项（跳过原始特征）
    for i, name in enumerate(poly_feature_names):
        if name not in available_important_cols and ' ' in name:  # 交互项或平方项
            clean_name = name.replace(' ', '_x_').replace('^2', '_sq')
            if clean_name not in df.columns:
                df[clean_name] = poly_features[:, i]
                new_features.append(clean_name)
    
    print(f"  创建 {len(new_features) - len([f for f in new_features if '_sq' not in f and '_x_' not in f])} 个多项式特征")

# ==================== 11. 分类变量编码 ====================
print("编码分类变量...")

# 低基数分类变量：One-Hot Encoding
low_cardinality_cols = [col for col in categorical_cols if df[col].nunique() < 10]

if low_cardinality_cols:
    df = pd.get_dummies(df, columns=low_cardinality_cols, drop_first=True)
    print(f"  对 {len(low_cardinality_cols)} 个低基数分类变量进行One-Hot编码")

# 高基数分类变量：Label Encoding（保留原值用于Tree模型）
high_cardinality_cols = [col for col in categorical_cols if col in df.columns and df[col].nunique() >= 10]
for col in high_cardinality_cols:
    df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
    new_features.append(f'{col}_encoded')
    print(f"  对 {col} 进行Label编码")

# ==================== 12. 最终处理和保存 ====================
print("保存特征工程后的数据...")

# 处理可能产生的无穷值或NaN
df = df.replace([np.inf, -np.inf], np.nan)
# 用中位数填充新生成的数值特征中的NaN
for col in new_features:
    if col in df.columns and df[col].dtype in [np.float64, np.int64]:
        df[col] = df[col].fillna(df[col].median())

# 保存数据
df.to_csv(OUTPUT_PATH, index=False)
print(f"数据已保存到: {OUTPUT_PATH}")
print(f"最终数据形状: {df.shape}")

# 输出新生成的特征列表
print("\n" + "="*50)
print("新生成的特征列表:")
print("="*50)
for i, feat in enumerate(new_features, 1):
    print(f"{i:2d}. {feat}")

print(f"\n总共生成 {len(new_features)} 个新特征")
print(f"原始特征数: {df.shape[1] - len(new_features)}")
print(f"总特征数: {df.shape[1]}")