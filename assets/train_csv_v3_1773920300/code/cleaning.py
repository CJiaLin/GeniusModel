import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据加载 ====================
print("=" * 50)
print("开始数据清洗流程")
print("=" * 50)

# 加载原始数据
file_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
df = pd.read_csv(file_path)

print(f"原始数据形状: {df.shape}")
print(f"总列数: {df.shape[1]}")
print(f"总行数: {df.shape[0]}")

# ==================== 初始数据质量分析 ====================
print("\n" + "=" * 50)
print("初始数据质量分析")
print("=" * 50)

# 检查缺失值情况
missing_info = df.isnull().sum()
missing_cols = missing_info[missing_info > 0].sort_values(ascending=False)
print(f"\n存在缺失值的列数: {len(missing_cols)}")
if len(missing_cols) > 0:
    print("\n缺失值统计:")
    for col, count in missing_cols.items():
        print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")

# 检查数据类型
print(f"\n数值列数量: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"分类列数量: {df.select_dtypes(include=['object']).shape[1]}")

# ==================== 数据清洗 ====================
print("\n" + "=" * 50)
print("开始数据清洗")
print("=" * 50)

# 创建清洗后的数据副本
df_cleaned = df.copy()

# 1. 处理LotFrontage（房屋前街道长度）- 数值型，用中位数填充
if 'LotFrontage' in df_cleaned.columns:
    median_lot_frontage = df_cleaned['LotFrontage'].median()
    missing_count = df_cleaned['LotFrontage'].isnull().sum()
    df_cleaned['LotFrontage'].fillna(median_lot_frontage, inplace=True)
    print(f"LotFrontage: 用中位数 {median_lot_frontage} 填充 {missing_count} 个缺失值")

# 2. 处理Alley（小巷类型）- 分类型，缺失表示没有小巷，填'NA'
if 'Alley' in df_cleaned.columns:
    missing_count = df_cleaned['Alley'].isnull().sum()
    df_cleaned['Alley'].fillna('NA', inplace=True)
    print(f"Alley: 用'NA'填充 {missing_count} 个缺失值（表示无小巷）")

# 3. 处理MasVnrType（砌体饰面类型）- 分类型，缺失表示没有，填'None'
if 'MasVnrType' in df_cleaned.columns:
    missing_count = df_cleaned['MasVnrType'].isnull().sum()
    df_cleaned['MasVnrType'].fillna('None', inplace=True)
    print(f"MasVnrType: 用'None'填充 {missing_count} 个缺失值（表示无砌体饰面）")

# 4. 处理MasVnrArea（砌体饰面面积）- 数值型
if 'MasVnrArea' in df_cleaned.columns:
    missing_count = df_cleaned['MasVnrArea'].isnull().sum()
    # 如果MasVnrType为'None'，则MasVnrArea填0，否则用中位数
    masvnr_median = df_cleaned[df_cleaned['MasVnrType'] != 'None']['MasVnrArea'].median()
    df_cleaned.loc[df_cleaned['MasVnrType'] == 'None', 'MasVnrArea'] = 0
    df_cleaned.loc[(df_cleaned['MasVnrType'] != 'None') & (df_cleaned['MasVnrArea'].isnull()), 'MasVnrArea'] = masvnr_median
    print(f"MasVnrArea: 根据MasVnrType填充 {missing_count} 个缺失值（None类型填0，其他填中位数{masvnr_median}）")

# 5. 处理地下室相关特征 - 缺失表示没有地下室，填'NA'或0
bsmt_categorical = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_categorical:
    if col in df_cleaned.columns:
        missing_count = df_cleaned[col].isnull().sum()
        df_cleaned[col].fillna('NA', inplace=True)
        print(f"{col}: 用'NA'填充 {missing_count} 个缺失值（表示无地下室）")

# 地下室数值型特征（如果有缺失也填0）
bsmt_numerical = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
for col in bsmt_numerical:
    if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
        missing_count = df_cleaned[col].isnull().sum()
        df_cleaned[col].fillna(0, inplace=True)
        print(f"{col}: 用0填充 {missing_count} 个缺失值")

# 6. 处理Electrical（电力系统）- 分类型，用众数填充
if 'Electrical' in df_cleaned.columns:
    missing_count = df_cleaned['Electrical'].isnull().sum()
    mode_electrical = df_cleaned['Electrical'].mode()[0]
    df_cleaned['Electrical'].fillna(mode_electrical, inplace=True)
    print(f"Electrical: 用众数 '{mode_electrical}' 填充 {missing_count} 个缺失值")

# 7. 检查并处理其他可能的缺失值（ FireplaceQu, PoolQC, Fence, MiscFeature等）
other_na_cols = ['FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'GarageType', 'GarageFinish', 
                 'GarageQual', 'GarageCond', 'GarageYrBlt']
for col in other_na_cols:
    if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
        missing_count = df_cleaned[col].isnull().sum()
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col].fillna('NA', inplace=True)
            print(f"{col}: 用'NA'填充 {missing_count} 个缺失值")
        else:
            df_cleaned[col].fillna(0, inplace=True)
            print(f"{col}: 用0填充 {missing_count} 个缺失值")

# ==================== 数据类型优化 ====================
print("\n" + "=" * 50)
print("数据类型优化")
print("=" * 50)

# MSSubClass是分类变量（建筑类型代码），应转为字符串
df_cleaned['MSSubClass'] = df_cleaned['MSSubClass'].astype(str)
print("MSSubClass: 转换为字符串类型（分类变量）")

# 检查年份相关列的合理性
year_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
for col in year_cols:
    if col in df_cleaned.columns:
        # 将不合理的年份（如0或过大）设为缺失，然后用YearBuilt填充
        invalid_years = df_cleaned[(df_cleaned[col] < 1800) | (df_cleaned[col] > 2025)][col].count()
        if invalid_years > 0:
            df_cleaned.loc[(df_cleaned[col] < 1800) | (df_cleaned[col] > 2025), col] = df_cleaned.loc[(df_cleaned[col] < 1800) | (df_cleaned[col] > 2025), 'YearBuilt']
            print(f"{col}: 修正 {invalid_years} 个异常年份值")

# ==================== 异常值检测与处理 ====================
print("\n" + "=" * 50)
print("异常值检测")
print("=" * 50)

# 检查数值列的异常值（使用IQR方法）
numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
outlier_summary = {}

for col in numerical_cols:
    if col == 'Id':  # 跳过ID列
        continue
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # 使用3倍IQR，更宽松
    upper_bound = Q3 + 3 * IQR
    outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)][col]
    if len(outliers) > 0:
        outlier_summary[col] = len(outliers)

if outlier_summary:
    print(f"检测到 {len(outlier_summary)} 个列存在潜在异常值（使用3*IQR规则）:")
    for col, count in sorted(outlier_summary.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {col}: {count} 个异常值")
else:
    print("未检测到显著异常值")

# ==================== 保存清洗后的数据 ====================
print("\n" + "=" * 50)
print("保存清洗后的数据")
print("=" * 50)

output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df_cleaned.to_csv(output_path, index=False)
print(f"清洗后的数据已保存至: {output_path}")

# ==================== 清洗结果统计 ====================
print("\n" + "=" * 50)
print("清洗结果统计")
print("=" * 50)

# 检查清洗后是否还有缺失值
remaining_missing = df_cleaned.isnull().sum().sum()
print(f"清洗后总缺失值数量: {remaining_missing}")

# 清洗前后对比
print(f"\n数据形状: {df.shape} -> {df_cleaned.shape}")
print(f"数值列数量: {len(df.select_dtypes(include=[np.number]).columns)} -> {len(df_cleaned.select_dtypes(include=[np.number]).columns)}")
print(f"分类列数量: {len(df.select_dtypes(include=['object']).columns)} -> {len(df_cleaned.select_dtypes(include=['object']).columns)}")

# 各列缺失值对比（仅显示有变化的）
print("\n各列缺失值处理情况:")
for col in df.columns:
    before = df[col].isnull().sum()
    after = df_cleaned[col].isnull().sum()
    if before > 0 or after > 0:
        print(f"  {col}: {before} -> {after}")

print("\n" + "=" * 50)
print("数据清洗完成!")
print("=" * 50)