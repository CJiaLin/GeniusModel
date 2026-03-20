#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoML 全流程建模脚本

生成时间: 2026-03-20 09:26:39
会话ID: train_csv_v3_1773969197

使用说明:
1. 确保已安装依赖: pip install pandas scikit-learn joblib
2. 运行脚本: python pipeline.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# 配置
DATA_PATH = "/Users/cjialin/code/AutoMLByLLM/train.csv"
TARGET_COLUMN = "SalePrice"
TASK_TYPE = "regression"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("AutoML 全流程建模")
print("=" * 60)
print(f"数据路径: {DATA_PATH}")
print(f"目标列: {TARGET_COLUMN}")
print(f"任务类型: {TASK_TYPE}")
print()


# ============================================
# 阶段 1: 数据清洗
# ============================================
print("\n[阶段 1] 数据清洗...")

import pandas as pd
import numpy as np
from pathlib import Path

def clean_data():
    """
    数据清洗主函数
    """
    # 设置数据路径
    input_path = "/Users/cjialin/code/AutoMLByLLM/train.csv"
    output_path = "/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv"
    
    print("=" * 60)
    print("开始数据清洗流程")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n步骤 1: 加载数据...")
    df = pd.read_csv(input_path)
    print(f"原始数据形状: {df.shape}")
    print(f"原始数据列数: {df.shape[1]}")
    
    # 2. 查看原始缺失值情况
    print("\n步骤 2: 分析缺失值情况...")
    missing_cols = df.columns[df.isnull().any()].tolist()
    print(f"包含缺失值的列: {missing_cols}")
    missing_stats = df[missing_cols].isnull().sum()
    print("缺失值统计:")
    for col, count in missing_stats.items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    
    # 3. 处理数值型缺失值
    print("\n步骤 3: 处理数值型缺失值...")
    
    # LotFrontage: 用中位数填充（街道到房产的直线距离）
    if 'LotFrontage' in df.columns:
        lot_frontage_median = df['LotFrontage'].median()
        df['LotFrontage'].fillna(lot_frontage_median, inplace=True)
        print(f"  LotFrontage: 使用中位数 {lot_frontage_median} 填充")
    
    # MasVnrArea: 用0填充（砌体贴面面积，缺失表示无砌体贴面）
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'].fillna(0, inplace=True)
        print(f"  MasVnrArea: 使用0填充")
    
    # 4. 处理分类型缺失值
    print("\n步骤 4: 处理分类型缺失值...")
    
    # Alley: 用'None'填充（巷子通道，NA表示没有巷子）
    if 'Alley' in df.columns:
        df['Alley'].fillna('None', inplace=True)
        print(f"  Alley: 使用'None'填充")
    
    # 地下室相关字段：用'None'填充表示没有地下室
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        if col in df.columns:
            df[col].fillna('None', inplace=True)
            print(f"  {col}: 使用'None'填充")
    
    # MasVnrType: 用众数填充
    if 'MasVnrType' in df.columns:
        mas_vnr_mode = df['MasVnrType'].mode()[0]
        df['MasVnrType'].fillna(mas_vnr_mode, inplace=True)
        print(f"  MasVnrType: 使用众数 '{mas_vnr_mode}' 填充")
    
    # Electrical: 用众数填充
    if 'Electrical' in df.columns:
        electrical_mode = df['Electrical'].mode()[0]
        df['Electrical'].fillna(electrical_mode, inplace=True)
        print(f"  Electrical: 使用众数 '{electrical_mode}' 填充")
    
    # 5. 数据类型优化
    print("\n步骤 5: 优化数据类型...")
    
    # 定义分类列（基于提供的信息）
    categorical_columns = [
        'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
        'RoofStyle', 'RoofMatl', 'MasVnrType', 'BsmtQual', 
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Electrical'
    ]
    
    # 将分类列转换为category类型
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    print(f"  已将 {len([c for c in categorical_columns if c in df.columns])} 列转换为category类型")
    
    # 6. 删除不必要的列
    print("\n步骤 6: 删除标识符列...")
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        print("  已删除 'Id' 列")
    
    # 7. 检查重复值
    print("\n步骤 7: 检查重复值...")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"  发现 {duplicate_count} 行重复数据，已删除")
    else:
        print("  未发现重复数据")
    
    # 8. 验证缺失值是否全部处理
    print("\n步骤 8: 验证缺失值处理...")
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing == 0:
        print("  所有缺失值已处理完毕")
    else:
        print(f"  警告：仍有 {remaining_missing} 个缺失值")
        remaining_cols = df.columns[df.isnull().any()].tolist()
        print(f"  剩余缺失值列: {remaining_cols}")
    
    # 9. 保存清洗后的数据
    print("\n步骤 9: 保存清洗后的数据...")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  数据已保存到: {output_path}")
    
    # 10. 生成统计报告
    print("\n" + "=" * 60)
    print("数据清洗完成 - 统计报告")
    print("=" * 60)
    
    # 内存使用统计
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    report = {
        "原始数据行数": 1460,
        "原始数据列数": 81,
        "清洗后数据行数": len(df),
        "清洗后数据列数": len(df.columns),
        "删除重复行数": duplicate_count,
        "删除列数": 1 if 'Id' not in df.columns else 0,
        "处理后缺失值总数": remaining_missing,
        "内存使用": f"{original_memory:.2f} MB"
    }
    
    print("\n清洗结果统计:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # 各列数据类型统计
    print(f"\n数据类型分布:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} 列")
    
    print("\n数值列样例（前5列）:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols[:5]:
        print(f"  {col}: 均值={df[col].mean():.2f}, 标准差={df[col].std():.2f}")
    
    print("\n分类列样例（前5列）:")
    cat_cols = df.select_dtypes(include=['category']).columns.tolist()
    for col in cat_cols[:5]:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count} 个唯一值")
    
    return df, report

# 执行清洗
if __name__ == "__main__":
    cleaned_df, statistics = clean_data()
    print("\n清洗流程全部完成！")

print("✓ 数据清洗完成")


# ============================================
# 阶段 2: 特征工程
# ============================================
print("\n[阶段 2] 特征工程...")

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

print("✓ 特征工程完成")


# ============================================
# 保存结果
# ============================================
print("\n" + "=" * 60)
print("建模完成!")
print("=" * 60)

# 保存结果摘要
summary = {
    "data_path": DATA_PATH,
    "target_column": TARGET_COLUMN,
    "task_type": TASK_TYPE,
    "output_dir": str(OUTPUT_DIR),
    "timestamp": pd.Timestamp.now().isoformat()
}

with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存到: {{OUTPUT_DIR}}")
print("\n文件列表:")
for file in OUTPUT_DIR.iterdir():
    print(f"  - {{file.name}}")
