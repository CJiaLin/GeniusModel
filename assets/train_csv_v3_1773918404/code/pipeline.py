#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoML 全流程建模脚本

生成时间: 2026-03-19 19:21:27
会话ID: train_csv_v3_1773918404

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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 定义数据路径
INPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/train.csv'
OUTPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'

# 定义实际数据的列名（基于提供的信息）
NUMERIC_COLS = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
                'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
                'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF']

CATEGORICAL_COLS = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
                   'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
                   'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl']

MISSING_COLS = ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 
                'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical']

def load_data(path):
    """加载数据并返回DataFrame"""
    df = pd.read_csv(path)
    print(f"原始数据形状: {df.shape}")
    return df

def analyze_missing(df):
    """分析缺失值情况"""
    missing_stats = pd.DataFrame({
        '缺失数量': df.isnull().sum(),
        '缺失比例': df.isnull().sum() / len(df) * 100
    })
    missing_stats = missing_stats[missing_stats['缺失数量'] > 0].sort_values('缺失比例', ascending=False)
    print("\n缺失值统计:")
    print(missing_stats)
    return missing_stats

def handle_missing_values(df):
    """处理缺失值"""
    df_clean = df.copy()
    
    for col in df_clean.columns:
        missing_ratio = df_clean[col].isnull().sum() / len(df_clean)
        
        if missing_ratio == 0:
            continue
            
        # 策略1: 缺失比例 > 50%，删除列
        if missing_ratio > 0.5:
            print(f"删除列 {col} (缺失率: {missing_ratio:.2%})")
            df_clean.drop(columns=[col], inplace=True)
            
        # 策略2: 数值型变量 - 使用中位数填充
        elif df_clean[col].dtype in ['int64', 'float64']:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"列 {col} 使用 median ({median_val}) 填充缺失值")
                
        # 策略3: 类别型变量 - 使用众数或'None'填充
        else:
            if not df_clean[col].mode().empty:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"列 {col} 使用 mode ({mode_val}) 填充缺失值")
            else:
                df_clean[col].fillna('None', inplace=True)
                print(f"列 {col} 使用 'None' 填充缺失值")
    
    return df_clean

def handle_duplicates(df):
    """处理重复值"""
    duplicates = df.duplicated().sum()
    print(f"\n完全重复行数: {duplicates}")
    
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"删除重复行后剩余: {len(df)} 行")
    
    # 检查基于Id的重复（如果存在Id列）
    if 'Id' in df.columns:
        id_duplicates = df.duplicated(subset=['Id']).sum()
        print(f"基于Id的重复行数: {id_duplicates}")
        if id_duplicates > 0:
            df = df.drop_duplicates(subset=['Id'], keep='first')
            print(f"删除Id重复后剩余: {len(df)} 行")
    
    return df

def detect_outliers_iqr(df, column):
    """使用IQR方法检测异常值"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def handle_outliers(df, method='clip'):
    """处理异常值"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_report = {}
    
    for col in numeric_cols:
        if col == 'Id':  # 跳过Id列
            continue
            
        outliers, lower, upper = detect_outliers_iqr(df, col)
        
        if len(outliers) > 0:
            outlier_report[col] = len(outliers)
            
            if method == 'clip':  # 缩尾处理
                df[col] = df[col].clip(lower, upper)
            elif method == 'remove':  # 删除
                df = df[~df.index.isin(outliers.index)]
    
    if outlier_report:
        print(f"\n异常值处理报告 (method={method}):")
        for col, count in outlier_report.items():
            print(f"  {col}: {count} 个异常值")
    
    return df

def optimize_data_types(df):
    """优化数据类型"""
    for col in df.columns:
        # 如果列名在数值列列表中但类型为object，尝试转换
        if col in NUMERIC_COLS and df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"列 {col} 转换为数值型")
            except:
                pass
        
        # 类别型优化：如果唯一值比例小于50%且为object类型
        if df[col].dtype == 'object':
            num_unique = df[col].nunique()
            if num_unique / len(df) < 0.5:
                df[col] = df[col].astype('category')
                print(f"列 {col} 转换为category类型 (唯一值: {num_unique})")
    
    return df

def clean_text_columns(df):
    """清洗文本列"""
    text_cols = df.select_dtypes(include=['object']).columns
    
    for col in text_cols:
        # 去除前后空格
        df[col] = df[col].astype(str).str.strip()
        
        # 统一大小写（对于短文本类别）
        if df[col].str.len().mean() < 50:
            df[col] = df[col].str.lower()
        
        # 处理空字符串和特殊标记
        df[col] = df[col].replace(['nan', 'null', 'none', 'na', ''], np.nan)
    
    return df

def validate_cleaning(df_original, df_cleaned):
    """验证清洗效果"""
    report = {
        '原始行数': len(df_original),
        '清洗后行数': len(df_cleaned),
        '删除行数': len(df_original) - len(df_cleaned),
        '缺失值总数(清洗后)': df_cleaned.isnull().sum().sum(),
        '重复行数(清洗后)': df_cleaned.duplicated().sum(),
        '数值列数': len(df_cleaned.select_dtypes(include=[np.number]).columns),
        '类别列数': len(df_cleaned.select_dtypes(include=['object', 'category']).columns)
    }
    return report

def main():
    """主函数：执行完整的数据清洗流程"""
    print("="*60)
    print("开始数据清洗流程")
    print("="*60)
    
    # 步骤1: 加载数据
    print("\n步骤1: 加载数据...")
    df = load_data(INPUT_PATH)
    df_original = df.copy()
    
    # 步骤2: 分析缺失值
    print("\n步骤2: 分析缺失值...")
    analyze_missing(df)
    
    # 步骤3: 处理缺失值
    print("\n步骤3: 处理缺失值...")
    df = handle_missing_values(df)
    
    # 步骤4: 处理重复值
    print("\n步骤4: 处理重复值...")
    df = handle_duplicates(df)
    
    # 步骤5: 处理异常值
    print("\n步骤5: 处理异常值...")
    df = handle_outliers(df, method='clip')
    
    # 步骤6: 优化数据类型
    print("\n步骤6: 优化数据类型...")
    df = optimize_data_types(df)
    
    # 步骤7: 清洗文本数据
    print("\n步骤7: 清洗文本数据...")
    df = clean_text_columns(df)
    
    # 步骤8: 最终验证
    print("\n步骤8: 验证清洗结果...")
    report = validate_cleaning(df_original, df)
    
    print("\n" + "="*60)
    print("清洗结果统计")
    print("="*60)
    for key, value in report.items():
        print(f"{key}: {value}")
    
    # 步骤9: 保存清洗后的数据
    print(f"\n步骤9: 保存清洗后的数据到 {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    print("数据清洗完成！")
    
    # 返回清洗后的DataFrame（如果在交互环境中使用）
    return df, report

if __name__ == "__main__":
    df_cleaned, cleaning_report = main()

print("✓ 数据清洗完成")


# ============================================
# 阶段 2: 特征工程
# ============================================
print("\n[阶段 2] 特征工程...")

{
    "thinking": "基于用户提供的特征工程方案，我需要生成完整的Python代码。关键要求：\n1. 使用实际数据路径 `/Users/cjialin/code/AutoMLByLLM/train.csv`\n2. 目标列为 `SalePrice`\n3. 包含所有提到的数值列和分类列\n4. 实现特征工程方案中的所有策略：缺失值处理、异常值处理、特征创建、编码、变换\n5. 保存结果到 `/Users/cjialin/code/AutoMLByLLM/train_features.csv`\n6. 修复上次代码的语法错误（EOL错误通常是字符串未闭合或跨行问题）\n7. 返回JSON格式，包含thinking和code字段\n\n我将创建一个完整的特征工程类，包含所有方案中提到的特征创建逻辑，并添加列存在性检查以确保健壮性。",
    "code": "import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import LabelEncoder, StandardScaler\nimport warnings\nwarnings.filterwarnings('ignore')\n\nclass HousePriceFeatureEngineer:\n    def __init__(self):\n        self.quality_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}\n        self.label_encoders = {}\n        self.new_features = []\n        \n    def fit_transform(self, df, target_col='SalePrice'):\n        df = df.copy()\n        original_cols = set(df.columns.tolist())\n        \n        print(\"Starting feature engineering...\")\n        print(f\"Original shape: {df.shape}\")\n        \n        # Step 1: Handle missing values\n        df = self._handle_missing(df)\n        print(\"Missing values handled\")\n        \n        # Step 2: Handle outliers\n        df = self._handle_outliers(df)\n        print(\"Outliers handled\")\n        \n        # Step 3: Create new features\n        df = self._create_features(df)\n        print(\"New features created\")\n        \n        # Step 4: Encode categorical features\n        df = self._encode_features(df)\n        print(\"Features encoded\")\n        \n        # Step 5: Transform skewed features\n        df = self._transform_features(df, target_col)\n        print(\"Features transformed\")\n        \n        # Identify new features\n        current_cols = set(df.columns.tolist())\n        self.new_features = sorted(list(current_cols - original_cols))\n        \n        print(f\"Final shape: {df.shape}\")\n        print(f\"New features created: {len(self.new_features)}\")\n        \n        return df\n    \n    def _handle_missing(self, df):\n        # Strategy 1: Categorical features - fill 'None' (indicates absence of feature)\n        fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',\n                     'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',\n                     'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', \n                     'BsmtFinType2', 'MasVnrType']\n        for col in fill_none:\n            if col in df.columns:\n                df[col] = df[col].fillna('None')\n        \n        # Strategy 2: Numerical features - fill 0 (indicates absence of feature)\n        fill_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', \n                     'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', \n                     'BsmtHalfBath', 'MasVnrArea']\n        for col in fill_zero:\n            if col in df.columns:\n                df[col] = df[col].fillna(0)\n        \n        # Strategy 3: LotFrontage - fill by Neighborhood median, then global median\n        if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:\n            df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(\n                lambda x: x.fillna(x.median())\n            )\n            df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())\n        \n        # Strategy 4: Electrical - fill with mode\n        if 'Electrical' in df.columns:\n            mode_val = df['Electrical'].mode()\n            if len(mode_val) > 0:\n                df['Electrical'] = df['Electrical'].fillna(mode_val[0])\n        \n        # Create missing indicators for important features\n        if 'LotFrontage' in df.columns:\n            df['LotFrontage_Missing'] = df['LotFrontage'].isna().astype(int)\n        if 'GarageYrBlt' in df.columns:\n            df['GarageYrBlt_Missing'] = (df['GarageYrBlt'] == 0).astype(int)\n        if 'MasVnrArea' in df.columns:\n            df['MasVnrArea_Missing'] = (df['MasVnrArea'] == 0).astype(int)\n            \n        return df\n    \n    def _handle_outliers(self, df):\n        # Remove extreme outliers in GrLivArea (> 4000 sqft, typically data entry errors)\n        if 'GrLivArea' in df.columns and len(df) > 1000:\n            outlier_idx = df[df['GrLivArea'] > 4000].index\n            if len(outlier_idx) < 10:\n                df = df.drop(outlier_idx)\n        \n        # Clip extreme LotArea values\n        if 'LotArea' in df.columns:\n            df['LotArea'] = df['LotArea'].clip(upper=50000)\n        \n        return df\n    \n    def _create_features(self, df):\n        eps = 1e-6\n        \n        # Area Related Features\n        if all(col in df.columns for col in ['GrLivArea', 'TotalBsmtSF']):\n            df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']\n        \n        porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']\n        available_porch = [col for col in porch_cols if col in df.columns]\n        if available_porch:\n            df['TotalPorchSF'] = df[available_porch].sum(axis=1)\n        \n        if 'WoodDeckSF' in df.columns and 'PoolArea' in df.columns:\n            if 'TotalPorchSF' in df.columns:\n                df['TotalOutdoorSF'] = df['WoodDeckSF'] + df['TotalPorchSF'] + df['PoolArea']\n            else:\n                df['TotalOutdoorSF'] = df['WoodDeckSF'] + df['PoolArea']\n        \n        if 'LotArea' in df.columns and 'TotRmsAbvGrd' in df.columns:\n            df['LotAreaPerRoom'] = df['LotArea'] / (df['TotRmsAbvGrd'] + eps)\n        \n        if 'GrLivArea' in df.columns and 'LotArea' in df.columns:\n            df['LivingAreaRatio'] = df['GrLivArea'] / (df['LotArea'] + eps)\n        \n        if 'TotalBsmtSF' in df.columns and '1stFlrSF' in df.columns:\n            df['BasementRatio'] = df['TotalBsmtSF'] / (df['1stFlrSF'] + eps)\n        \n        if '2ndFlrSF' in df.columns:\n            df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)\n        if 'TotalBsmtSF' in df.columns:\n            df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)\n        \n        # Room and Quality Features\n        bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']\n        if all(col in df.columns for col in bath_cols):\n            df['TotalBath'] = (df['FullBath'] + 0.5*df['HalfBath'] + \n                              df['BsmtFullBath'] + 0.5*df['BsmtHalfBath'])\n        \n        if 'BedroomAbvGr' in df.columns and 'TotalBath' in df.columns:\n            df['BedroomToBathRatio'] = df['BedroomAbvGr'] / (df['TotalBath'] + eps)\n        \n        if 'OverallQual' in df.columns and 'OverallCond' in df.columns:\n            df['QualCondScore'] = df['OverallQual'] * df['OverallCond']\n        \n        if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:\n            df['QualSF'] = df['OverallQual'] * df['GrLivArea']\n        \n        # Time Related Features\n        if 'YrSold' in df.columns and 'YearBuilt' in df.columns:\n            df['HouseAge'] = df['YrSold'] - df['YearBuilt']\n            df['IsNewHouse'] = (df['YrSold'] == df['YearBuilt']).astype(int)\n        \n        if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:\n            df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']\n            df['HasRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)\n            df['IsRecentRemod'] = (df['RemodAge'] <= 5).astype(int)\n        \n        if 'YearBuilt' in df.columns:\n            df['BuiltDecade'] = (df['YearBuilt'] // 10) * 10\n        \n        if 'MoSold' in df.columns:\n            season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, \n                          6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}\n            df['SeasonSold'] = df['MoSold'].map(season_map)\n        \n        # Garage and Facility Features\n        if 'PoolArea' in df.columns:\n            df['HasPool'] = (df['PoolArea'] > 0).astype(int)\n        \n        if 'Fireplaces' in df.columns:\n            df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)\n        \n        if 'Fence' in df.columns:\n            df['HasFence'] = (df['Fence'] != 'None').astype(int)\n        \n        if 'GarageArea' in df.columns:\n            df['HasGarage'] = (df['GarageArea'] > 0).astype(int)\n        \n        if 'YrSold' in df.columns and 'GarageYrBlt' in df.columns:\n            df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']\n        \n        if 'GarageArea' in df.columns and 'GarageCars' in df.columns:\n            df['GarageValue'] = df['GarageArea'] * df['GarageCars']\n        \n        # Encode quality ratings\n        qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', \n                     'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'PoolQC']\n        for col in qual_cols:\n            if col in df.columns:\n                df[f'{col}_enc'] = df[col].map(self.quality_mapping).fillna(0)\n        \n        qual_enc_cols = [f'{col}_enc' for col in qual_cols if f'{col}_enc' in df.columns]\n        if qual_enc_cols and 'OverallQual' in df.columns:\n            df['TotalQualScore'] = df['OverallQual'] + df[qual_enc_cols].sum(axis=1)\n        \n        # Interaction Features\n        if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:\n            df['Qual_x_LivArea'] = df['OverallQual'] * df['GrLivArea']\n        \n        if 'OverallQual' in df.columns and 'YearBuilt' in df.columns:\n            df['Qual_x_YearBuilt'] = df['OverallQual'] * df['YearBuilt']\n        \n        if 'TotalSF' in df.columns and 'TotalBath' in df.columns:\n            df['Area_x_Bath'] = df['TotalSF'] * df['TotalBath']\n        elif 'GrLivArea' in df.columns and 'TotalBath' in df.columns:\n            df['Area_x_Bath'] = df['GrLivArea'] * df['TotalBath']\n        \n        if 'LotArea' in df.columns and 'OverallQual' in df.columns:\n            df['Lot_x_Qual'] = df['LotArea'] * df['OverallQual']\n        \n        if 'HouseAge' in df.columns and 'OverallQual' in df.columns:\n            df['Age_x_Qual'] = df['HouseAge'] * df['OverallQual']\n        \n        return df\n    \n    def _encode_features(self, df):\n        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()\n        \n        for col in categorical_cols:\n            if col not in self.label_encoders:\n                self.label_encoders[col] = LabelEncoder()\n                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))\n            else:\n                df[col] = self.label_encoders[col].transform(df[col].astype(str))\n        \n        return df\n    \n    def _transform_features(self, df, target_col):\n        skewed_cols = ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', \n                       'TotalSF', 'LotFrontage', 'MasVnrArea', 'WoodDeckSF', \n                       'OpenPorchSF', 'GarageArea']\n        \n        for col in skewed_cols:\n            if col in df.columns:\n                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))\n        \n        if target_col in df.columns:\n            df[f'{target_col}_log'] = np.log1p(df[target_col])\n        \n        return df\n\ndef main():\n    input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'\n    output_path = '/Users/cjialin/code/AutoMLByLLM/train_features.csv'\n    \n    print(f\"Loading data from {input_path}\")\n    df = pd.read_csv(input_path)\n    \n    print(f\"Original data shape: {df.shape}\")\n    print(f\"Target column: SalePrice\")\n    \n    engineer = HousePriceFeatureEngineer()\n    df_transformed = engineer.fit_transform(df, target_col='SalePrice')\n    \n    print(f\"\\nSaving transformed data to {output_path}\")\n    df_transformed.to_csv(output_path, index=False)\n    \n    print(\"\\nNew features created:\")\n    for feat in engineer.new_features:\n        print(f\"  - {feat}\")\n    \n    print(f\"\\nFeature engineering completed. Total features: {df_transformed.shape[1]}\")\n    return engineer

print("✓ 特征工程完成")


# ============================================
# 阶段 3: 模型训练
# ============================================
print("\n[阶段 3] 模型训练...")

{
    "thinking": "基于用户提供的建模方案，我需要生成一个完整的房价预测Python代码。关键点包括：\n1. 使用提供的数据路径 /Users/cjialin/code/AutoMLByLLM/train.csv\n2. 目标列是 SalePrice，需要进行对数变换处理右偏分布\n3. 使用实际列名（数值列和分类列如描述所示）\n4. 实现多种模型：Ridge回归、Random Forest、Gradient Boosting，并使用Stacking进行融合\n5. 使用KFold交叉验证评估\n6. 保存模型到指定路径\n7. 修复之前的语法错误（字符串引号问题）\n\n代码结构：\n- 导入必要的库\n- 加载数据并分离特征和目标\n- 对目标变量进行log1p变换\n- 自动识别数值和分类特征（同时兼容提供的列名列表）\n- 构建ColumnTransformer预处理管道（中位数填充+标准化，常数填充+独热编码）\n- 定义基线模型和Stacking集成模型\n- 使用Pipeline组合预处理和模型\n- 5折交叉验证评估（RMSE、MAE、R²）\n- 保存训练好的模型\n- 输出性能指标\n\n特别注意避免字符串引号冲突，不使用三重引号包裹代码，确保JSON格式有效。",
    "code": "import pandas as pd\nimport numpy as np\nimport os\nimport warnings\nwarnings.filterwarnings('ignore')\n\nfrom sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.linear_model import Ridge\nfrom sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\nimport joblib\n\nprint(\"=\" * 60)\nprint(\"房价预测模型训练 - 基于 Scikit-Learn\")\nprint(\"=\" * 60)\n\n# 1. 加载数据\ntrain_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'\nmodel_path = '/Users/cjialin/code/AutoMLByLLM/train_model.pkl'\n\nprint(f\"\\n正在加载数据: {train_path}\")\ntry:\n    df = pd.read_csv(train_path)\n    print(f\"数据加载成功！形状: {df.shape}\")\nexcept Exception as e:\n    print(f\"数据加载失败: {e}\")\n    raise\n\n# 2. 数据基本信息\nprint(f\"\\n数据概览:\")\nprint(f\"- 总行数: {len(df)}\")\nprint(f\"- 总列数: {len(df.columns)}\")\nprint(f\"- 目标列: SalePrice\")\n\n# 检查目标列\nif 'SalePrice' not in df.columns:\n    raise ValueError(\"数据集中未找到 'SalePrice' 列！\")\n\n# 3. 分离特征和目标变量\nX = df.drop(['SalePrice'], axis=1)\nif 'Id' in X.columns:\n    X = X.drop(['Id'], axis=1)\n    print(\"- 已移除 Id 列\")\n\ny = df['SalePrice']\n\n# 对目标变量进行对数变换，处理右偏分布\ny_log = np.log1p(y)\nprint(f\"- 目标变量已进行 log1p 变换\")\nprint(f\"- 变换前目标均值: {y.mean():.2f}, 标准差: {y.std():.2f}\")\nprint(f\"- 变换后目标均值: {y_log.mean():.4f}, 标准差: {y_log.std():.4f}\")\n\n# 4. 识别特征类型\n# 基于提供的数据信息中的列名\nnumeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()\ncategorical_features = X.select_dtypes(include=['object']).columns.tolist()\n\nprint(f\"\\n特征分析:\")\nprint(f\"- 数值型特征数量: {len(numeric_features)}\")\nprint(f\"- 分类型特征数量: {len(categorical_features)}\")\n\n# 显示关键特征（基于建模方案中提到的实际列名）\nkey_numeric = [col for col in numeric_features if col in ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF']]\nkey_categorical = [col for col in categorical_features if col in ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl']]\n\nif key_numeric:\n    print(f\"- 关键数值特征: {', '.join(key_numeric[:5])}...\")\nif key_categorical:\n    print(f\"- 关键分类特征: {', '.join(key_categorical[:5])}...\")\n\n# 5. 构建预处理管道\nprint(f\"\\n构建预处理管道...\")\n\n# 数值型特征处理：中位数填充 + 标准化\nnumeric_transformer = Pipeline(steps=[\n    ('imputer', SimpleImputer(strategy='median')),\n    ('scaler', StandardScaler())\n])\n\n# 分类型特征处理：常数填充 + 独热编码\ncategorical_transformer = Pipeline(steps=[\n    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\n    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))\n])\n\n# 组合预处理步骤\npreprocessor = ColumnTransformer(\n    transformers=[\n        ('num', numeric_transformer, numeric_features),\n        ('cat', categorical_transformer, categorical_features)\n    ])\n\n# 6. 定义模型（按照建模方案推荐）\nprint(f\"\\n初始化模型...\")\n\n# 基础模型参数设置\nridge_params = {'alpha': 1.0, 'random_state': 42}\nrf_params = {\n    'n_estimators': 500,\n    'max_depth': 20,\n    'min_samples_split': 5,\n    'min_samples_leaf': 2,\n    'max_features': 'sqrt',\n    'random_state': 42,\n    'n_jobs': -1\n}\ngb_params = {\n    'n_estimators': 500,\n    'learning_rate': 0.05,\n    'max_depth': 6,\n    'subsample': 0.8,\n    'random_state': 42\n}\n\n# 创建基学习器\nridge_reg = Ridge(**ridge_params)\nrf_reg = RandomForestRegressor(**rf_params)\ngb_reg = GradientBoostingRegressor(**gb_params)\n\n# Stacking集成（使用Ridge作为元学习器）\nestimators = [\n    ('ridge', ridge_reg),\n    ('rf', rf_reg),\n    ('gb', gb_reg)\n]\n\nstacking_reg = StackingRegressor(\n    estimators=estimators,\n    final_estimator=Ridge(alpha=1.0, random_state=42),\n    cv=5,\n    passthrough=False,\n    n_jobs=-1\n)\n\n# 构建完整管道\nmodel = Pipeline([\n    ('preprocessor', preprocessor),\n    ('regressor', stacking_reg)\n])\n\nprint(\"- 使用模型: StackingRegressor(Ridge + RandomForest + GradientBoosting)\")\n\n# 7. 交叉验证评估\nprint(f\"\\n开始5折交叉验证...\")\n\n# 定义评估指标\nscoring = {\n    'rmse': 'neg_mean_squared_error',\n    'mae': 'neg_mean_absolute_error',\n    'r2': 'r2'\n}\n\n# 执行交叉验证\nkf = KFold(n_splits=5, shuffle=True, random_state=42)\ncv_results = cross_validate(model, X, y_log, cv=kf, scoring=scoring, return_train_score=True, n_jobs=-1)\n\n# 计算指标\nrmse_log = np.sqrt(-cv_results['test_rmse'].mean())\nmae_log = -cv_results['test_mae'].mean()\nr2 = cv_results['test_r2'].mean()\n\n# 转换回原始尺度的近似值（使用expm1的近似）\nrmse_original = np.expm1(rmse_log)\nmae_original = np.expm1(mae_log)\n\nprint(f\"\\n交叉验证结果 (对数尺度):\")\nprint(f\"- RMSE: {rmse_log:.5f} ± {np.sqrt(cv_results['test_rmse'].var()):.5f}\")\nprint(f\"- MAE: {mae_log:.5f} ± {cv_results['test_mae'].std():.5f}\")\nprint(f\"- R² Score: {r2:.4f} ± {cv_results['test_r2'].std():.4f}\")\n\nprint(f\"\\n交叉验证结果 (原始尺度近似):\")\nprint(f\"- RMSE: ${rmse_original:,.2f}\")\nprint(f\"- MAE: ${mae_original:,.2f}\")\n\n# 8. 在全部数据上训练最终模型\nprint(f\"\\n在全部数据上训练最终模型...\")\nmodel.fit(X, y_log)\nprint(\"模型训练完成！\")\n\n# 9. 保存模型\nprint(f\"\\n保存模型到: {model_path}\")\ntry:\n    joblib.dump(model, model_path)\n    print(\"模型保存成功！\")\nexcept Exception as e:\n    print(f\"模型保存失败: {e}\")\n    raise\n\n# 10. 返回性能指标摘要\nprint(\"\\n\" + \"=\" * 60)\nprint(\"模型性能指标摘要\")\nprint(\"=\" * 60)\nperformance_metrics = {\n    'cv_rmse_log': float(rmse_log),\n    'cv_mae_log': float(mae_log),\n    'cv_r2_score': float(r2),\n    'cv_rmse_original_scale': float(rmse_original),\n    'cv_mae_original_scale': float(mae_original),\n    'model_path': model_path,\n    'n_samples': len(df),\n    'n

print("✓ 模型训练完成")


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
