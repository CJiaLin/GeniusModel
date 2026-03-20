import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 1. 读取数据
data_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
df = pd.read_csv(data_path)
print(f"原始数据形状: {df.shape}")

# 基于用户提供的实际列名
numeric_cols = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
                'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
                'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
categorical_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
                    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
                    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl']

# 确保所有列都存在，保留目标列
all_cols = numeric_cols + categorical_cols + ['SalePrice']
df = df[[col for col in all_cols if col in df.columns]].copy()

print(f"使用的列数: {len(df.columns)}")

# 2. 缺失值处理
# 数值列用中位数填充
for col in numeric_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# 分类列用'None'填充缺失值
for col in categorical_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col].fillna('None', inplace=True)

# 确保数值列为float类型
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].astype(float)

# 3. 特征工程 - 创建27个新特征
new_features = []

# 参考年份设定为2010（房价预测常用年份）
reference_year = 2010

# ===== 类别1: 面积组合特征 (9个) =====
# 1. 房屋总面积（地下室+一层+二层）
df['TotalHouseSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
new_features.append('TotalHouseSF')

# 2. 地上总面积（一层+二层）
df['AboveGroundSF'] = df['1stFlrSF'] + df['2ndFlrSF']
new_features.append('AboveGroundSF')

# 3. 地下室总完成面积
df['BsmtFinTotal'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
new_features.append('BsmtFinTotal')

# 4. 总完成面积（地下室完成+地上面积）
df['TotalFinSF'] = df['BsmtFinTotal'] + df['1stFlrSF'] + df['2ndFlrSF']
new_features.append('TotalFinSF')

# 5. 地下室完成比例（避免除零）
df['BsmtFinRatio'] = df['BsmtFinTotal'] / (df['TotalBsmtSF'] + 1)
new_features.append('BsmtFinRatio')

# 6. 地下室未完成比例
df['BsmtUnfRatio'] = df['BsmtUnfSF'] / (df['TotalBsmtSF'] + 1)
new_features.append('BsmtUnfRatio')

# 7. 一层面积占比
df['1stFlrRatio'] = df['1stFlrSF'] / (df['TotalHouseSF'] + 1)
new_features.append('1stFlrRatio')

# 8. 二层面积占比
df['2ndFlrRatio'] = df['2ndFlrSF'] / (df['TotalHouseSF'] + 1)
new_features.append('2ndFlrRatio')

# 9. 土地利用率（地上建筑面积/土地面积）
df['LotUtilization'] = df['AboveGroundSF'] / (df['LotArea'] + 1)
new_features.append('LotUtilization')

# ===== 类别2: 时间特征 (5个) =====
# 10. 房龄
df['HouseAge'] = reference_year - df['YearBuilt']
new_features.append('HouseAge')

# 11. 改造后年数
df['YearsSinceRemod'] = reference_year - df['YearRemodAdd']
new_features.append('YearsSinceRemod')

# 12. 是否经过改造（改造年份与建造年份不同）
df['IsRemod'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
new_features.append('IsRemod')

# 13. 是否新房（2000年后建造）
df['IsNew'] = (df['YearBuilt'] >= 2000).astype(int)
new_features.append('IsNew')

# 14. 改造时房龄
df['RemodAge'] = df['YearRemodAdd'] - df['YearBuilt']
new_features.append('RemodAge')

# ===== 类别3: 质量交互特征 (5个) =====
# 15. 质量-状况综合得分
df['QualCondScore'] = df['OverallQual'] * df['OverallCond']
new_features.append('QualCondScore')

# 16. 质量-面积交互
df['QualArea'] = df['OverallQual'] * df['TotalHouseSF']
new_features.append('QualArea')

# 17. 质量-土地面积交互（使用对数变换）
df['QualLot'] = df['OverallQual'] * np.log1p(df['LotArea'])
new_features.append('QualLot')

# 18. 质量-年龄交互（质量越高，房龄折旧影响越小）
df['QualAge'] = df['OverallQual'] / (df['HouseAge'] + 1)
new_features.append('QualAge')

# 19. 状况-面积交互
df['CondArea'] = df['OverallCond'] * df['TotalHouseSF']
new_features.append('CondArea')

# ===== 类别4: 房间/设施指示器 (4个) =====
# 20. 是否有地下室
df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
new_features.append('HasBasement')

# 21. 是否有二层
df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
new_features.append('Has2ndFloor')

# 22. 是否有砌体贴面
df['HasMasVnr'] = (df['MasVnrArea'] > 0).astype(int)
new_features.append('HasMasVnr')

# 23. 是否有临街面
df['HasLotFrontage'] = (df['LotFrontage'] > 0).astype(int)
new_features.append('HasLotFrontage')

# ===== 类别5: 统计聚合特征 (4个) =====
# 24. 按Neighborhood计算平均质量
df['Neighborhood_Qual_Mean'] = df.groupby('Neighborhood')['OverallQual'].transform('mean')
new_features.append('Neighborhood_Qual_Mean')

# 25. 按Neighborhood计算平均房龄
df['Neighborhood_Age_Mean'] = df.groupby('Neighborhood')['HouseAge'].transform('mean')
new_features.append('Neighborhood_Age_Mean')

# 26. 按MSSubClass计算平均质量
df['MSSubClass_Qual_Mean'] = df.groupby('MSSubClass')['OverallQual'].transform('mean')
new_features.append('MSSubClass_Qual_Mean')

# 27. 按MSZoning计算平均质量
df['MSZoning_Qual_Mean'] = df.groupby('MSZoning')['OverallQual'].transform('mean')
new_features.append('MSZoning_Qual_Mean')

# 4. 分类变量编码
# 对分类变量进行Label Encoding
for col in categorical_cols:
    if col in df.columns and df[col].dtype == 'object':
        le = LabelEncoder()
        df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))

# 5. 处理无穷值和异常值
df = df.replace([np.inf, -np.inf], np.nan)

# 用中位数填充新生成特征中的缺失值
for feat in new_features:
    if df[feat].isnull().sum() > 0:
        df[feat].fillna(df[feat].median(), inplace=True)

# 6. 保存特征工程后的数据
output_path = '/Users/cjialin/code/AutoMLByLLM/train_features.csv'
df.to_csv(output_path, index=False)

print(f"特征工程完成！")
print(f"新数据形状: {df.shape}")
print(f"新生成的特征数量: {len(new_features)}")
print(f"新生成的特征列表: {new_features}")
print(f"数据已保存到: {output_path}")

# 返回新生成的特征列表
result = {
    'new_features': new_features,
    'shape': df.shape,
    'output_path': output_path
}

print("\n特征工程执行成功！")
print(f"结果: {result}")