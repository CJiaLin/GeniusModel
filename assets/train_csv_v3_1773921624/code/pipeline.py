#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoML 全流程建模脚本

生成时间: 2026-03-20 08:46:22
会话ID: train_csv_v3_1773921624

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
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_and_clean_data():
    """
    加载并清洗数据
    """
    # 1. 加载数据
    file_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    print(f"正在加载数据: {file_path}")
    df = pd.read_csv(file_path)
    
    # 原始数据信息
    original_shape = df.shape
    print(f"原始数据形状: {original_shape}")
    print(f"原始数据列数: {len(df.columns)}")
    
    # 2. 处理缺失值
    print("\n开始处理缺失值...")
    
    # 数值列缺失值处理（基于提供的数值列信息）
    numeric_cols_with_na = ['LotFrontage', 'MasVnrArea']
    for col in numeric_cols_with_na:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if col == 'MasVnrArea':
                    # MasVnrArea 缺失可能表示没有砌体 veneer，用0填充
                    df[col].fillna(0, inplace=True)
                    print(f"  {col}: 填充 {missing_count} 个缺失值为 0")
                else:
                    # 其他数值列用中位数填充
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"  {col}: 填充 {missing_count} 个缺失值为中位数 {median_val}")
    
    # 分类列缺失值处理（基于提供的分类列和缺失值列信息）
    # Alley: 缺失表示没有巷道，用'NA'填充
    if 'Alley' in df.columns:
        missing_count = df['Alley'].isnull().sum()
        if missing_count > 0:
            df['Alley'].fillna('NA', inplace=True)
            print(f"  Alley: 填充 {missing_count} 个缺失值为 'NA'")
    
    # MasVnrType: 缺失可能表示没有砌体 veneer
    if 'MasVnrType' in df.columns:
        missing_count = df['MasVnrType'].isnull().sum()
        if missing_count > 0:
            df['MasVnrType'].fillna('None', inplace=True)
            print(f"  MasVnrType: 填充 {missing_count} 个缺失值为 'None'")
    
    # 地下室相关列: 缺失表示没有地下室
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col].fillna('NA', inplace=True)
                print(f"  {col}: 填充 {missing_count} 个缺失值为 'NA'")
    
    # Electrical: 用众数填充
    if 'Electrical' in df.columns:
        missing_count = df['Electrical'].isnull().sum()
        if missing_count > 0:
            mode_val = df['Electrical'].mode()[0]
            df['Electrical'].fillna(mode_val, inplace=True)
            print(f"  Electrical: 填充 {missing_count} 个缺失值为众数 '{mode_val}'")
    
    # 检查其他可能的缺失值列（数据中可能有更多缺失值列）
    other_missing_cols = df.columns[df.isnull().sum() > 0].tolist()
    if other_missing_cols:
        print(f"\n  发现其他缺失值列: {other_missing_cols}")
        for col in other_missing_cols:
            missing_count = df[col].isnull().sum()
            if df[col].dtype in ['int64', 'float64']:
                # 数值列用中位数填充
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"    {col}: 填充 {missing_count} 个缺失值为中位数 {median_val}")
            else:
                # 分类列用众数填充
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"    {col}: 填充 {missing_count} 个缺失值为众数 '{mode_val}'")
    
    # 3. 数据类型优化
    print("\n优化数据类型...")
    
    # 将某些数值列转换为整数（如果它们是整数类型）
    int_candidates = ['Id', 'MSSubClass', 'OverallQual', 'OverallCond', 
                      'YearBuilt', 'YearRemodAdd']
    for col in int_candidates:
        if col in df.columns and df[col].dtype == 'float64':
            df[col] = df[col].astype('int64')
            print(f"  {col}: 转换为 int64")
    
    # 将分类列转换为category类型以节省内存
    categorical_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
                       'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
                       'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
                       'RoofStyle', 'RoofMatl', 'MasVnrType', 'BsmtQual', 
                       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                       'Electrical']
    
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype('category')
            print(f"  {col}: 转换为 category")
    
    # 4. 异常值检测和处理
    print("\n检测异常值...")
    
    # 基于提供的数值列检查异常值
    numeric_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 
                   'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    
    outlier_stats = {}
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_stats[col] = outlier_count
            
            if outlier_count > 0:
                print(f"  {col}: 发现 {outlier_count} 个异常值 (范围: [{lower_bound:.2f}, {upper_bound:.2f}])")
                # 使用截断法处理极端异常值
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 5. 验证年份数据的合理性
    year_cols = ['YearBuilt', 'YearRemodAdd']
    current_year = 2024
    for col in year_cols:
        if col in df.columns:
            invalid_years = df[(df[col] < 1800) | (df[col] > current_year)]
            if len(invalid_years) > 0:
                print(f"  {col}: 发现 {len(invalid_years)} 个无效年份，使用众数替换")
                mode_year = df[col].mode()[0]
                df.loc[(df[col] < 1800) | (df[col] > current_year), col] = mode_year
    
    # 6. 确保面积数据的一致性
    area_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
    if all(col in df.columns for col in area_cols + ['TotalBsmtSF']):
        # 检查地下室总面积是否等于各部分之和
        calculated_total = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['BsmtUnfSF']
        inconsistent = (df['TotalBsmtSF'] - calculated_total).abs() > 1
        if inconsistent.sum() > 0:
            print(f"\n  发现 {inconsistent.sum()} 行地下室面积不一致，重新计算 TotalBsmtSF")
            df.loc[inconsistent, 'TotalBsmtSF'] = calculated_total[inconsistent]
    
    # 7. 保存清洗后的数据
    output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    df.to_csv(output_path, index=False)
    print(f"\n清洗后的数据已保存到: {output_path}")
    
    # 8. 生成清洗报告
    cleaned_shape = df.shape
    total_missing_original = sum([df[col].isnull().sum() for col in df.columns])
    
    # 统计信息
    stats = {
        '原始数据形状': original_shape,
        '清洗后数据形状': cleaned_shape,
        '总行数': cleaned_shape[0],
        '总列数': cleaned_shape[1],
        '数值列数量': len(df.select_dtypes(include=[np.number]).columns),
        '分类列数量': len(df.select_dtypes(include=['category', 'object']).columns),
        '处理后缺失值总数': df.isnull().sum().sum(),
        '异常值处理统计': outlier_stats,
        '内存使用优化': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    
    print("\n" + "="*50)
    print("数据清洗完成!")
    print("="*50)
    print(f"原始数据: {original_shape[0]} 行 × {original_shape[1]} 列")
    print(f"清洗后数据: {cleaned_shape[0]} 行 × {cleaned_shape[1]} 列")
    print(f"数值列数量: {stats['数值列数量']}")
    print(f"分类列数量: {stats['分类列数量']}")
    print(f"处理后缺失值总数: {stats['处理后缺失值总数']}")
    print(f"内存使用: {stats['内存使用优化']}")
    
    return df, stats

# 执行清洗
if __name__ == "__main__":
    cleaned_df, statistics = load_and_clean_data()
    
    # 显示前几行数据
    print("\n清洗后数据预览 (前5行):")
    print(cleaned_df.head())
    
    print("\n数据基本信息:")
    print(cleaned_df.info())

print("✓ 数据清洗完成")


# ============================================
# 阶段 2: 特征工程
# ============================================
print("\n[阶段 2] 特征工程...")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

class HousePriceFeatureEngineer:
    """
    房价预测特征工程类
    实现基于领域知识的特征工程，包括缺失值处理、特征聚合、交互特征创建和编码
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.target_encoders = {}
        self.ordinal_mappings = {
            'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 
            'None': 0, 'NoBasement': 0, 'NoGarage': 0
        }
        self.new_features = []
        
    def fit_transform(self, df, target_col='SalePrice'):
        """
        执行完整的特征工程流程
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始输入数据
        target_col : str
            目标变量列名
            
        Returns:
        --------
        pd.DataFrame
            特征工程后的数据
        list
            新生成的特征名称列表
        """
        print(f"原始数据形状: {df.shape}")
        df = df.copy()
        
        # 步骤1: 处理缺失值
        print("步骤1: 处理缺失值...")
        df = self._handle_missing(df)
        
        # 步骤2: 创建缺失值指示特征
        print("步骤2: 创建缺失值指示特征...")
        df = self._create_missing_indicators(df)
        
        # 步骤3: 创建面积聚合特征
        print("步骤3: 创建面积聚合特征...")
        df = self._create_area_features(df)
        
        # 步骤4: 创建时间衍生特征
        print("步骤4: 创建时间衍生特征...")
        df = self._create_time_features(df)
        
        # 步骤5: 创建质量交互特征
        print("步骤5: 创建质量交互特征...")
        df = self._create_quality_features(df)
        
        # 步骤6: 创建高级交互特征（包含目标编码）
        print("步骤6: 创建高级交互特征...")
        df = self._create_advanced_features(df, target_col)
        
        # 步骤7: 编码分类特征
        print("步骤7: 编码分类特征...")
        df = self._encode_categorical(df, target_col)
        
        # 步骤8: 对数变换处理偏态特征
        print("步骤8: 应用对数变换...")
        df = self._apply_log_transform(df, target_col)
        
        print(f"特征工程完成！")
        print(f"最终数据形状: {df.shape}")
        print(f"新生成特征数量: {len(self.new_features)}")
        
        return df, self.new_features
    
    def _handle_missing(self, df):
        """
        智能缺失值处理
        - 极高缺失率特征：保留用于创建指示器，填充'None'
        - 地下室/车库特征：填充'NoBasement'/'NoGarage'
        - LotFrontage：按Neighborhood分组填充中位数
        - GarageYrBlt：使用YearBuilt填充
        - 其他数值特征：填充0，分类特征填充'None'
        """
        # 极高缺失率特征处理
        high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
        for col in high_missing_cols:
            if col in df.columns:
                df[col] = df[col].fillna('None')
        
        # 地下室相关特征
        bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        for col in bsmt_cols:
            if col in df.columns:
                df[col] = df[col].fillna('NoBasement')
        
        # 车库相关特征
        garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
        for col in garage_cols:
            if col in df.columns:
                df[col] = df[col].fillna('NoGarage')
        
        # 数值型特征特殊处理
        if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
            # 按社区分组填充中位数（同社区的房屋临街面宽度相似）
            df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median())
            )
            # 如果仍有缺失，使用整体中位数
            df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
        
        if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
            # 无车库则使用建房年份（表示从建房起就没有车库）
            df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
        
        if 'MasVnrArea' in df.columns:
            df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
        
        if 'MasVnrType' in df.columns:
            df['MasVnrType'] = df['MasVnrType'].fillna('None')
        
        if 'Electrical' in df.columns:
            df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0] if not df['Electrical'].mode().empty else 'SBrkr')
        
        # 通用填充策略
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna('None')
        
        return df
    
    def _create_missing_indicators(self, df):
        """
        创建缺失值指示特征（二值特征）
        将高缺失率特征转换为"有无"指示器
        """
        indicators = {
            'HasPool': ('PoolQC', ['None']),
            'HasMiscFeature': ('MiscFeature', ['None']),
            'HasAlley': ('Alley', ['None']),
            'HasFence': ('Fence', ['None']),
            'HasFireplace': ('FireplaceQu', ['None']),
            'HasBasement': ('BsmtQual', ['NoBasement', 'None']),
            'HasGarage': ('GarageType', ['NoGarage', 'None']),
            'Has2ndFloor': ('2ndFlrSF', None)  # 数值型特殊处理
        }
        
        for new_col, (orig_col, none_values) in indicators.items():
            if orig_col in df.columns:
                if none_values is None:
                    # 数值型：大于0则为有
                    df[new_col] = (df[orig_col] > 0).astype(int)
                else:
                    # 分类型：不在none_values列表中则为有
                    df[new_col] = (~df[orig_col].isin(none_values)).astype(int)
                self.new_features.append(new_col)
        
        return df
    
    def _create_area_features(self, df):
        """
        创建面积相关聚合特征
        包括总面积、门廊面积、浴室计算、密度指标等
        """
        # 总居住面积（地上+地下）
        if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
            df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
            self.new_features.append('TotalSF')
        
        # 总门廊面积（所有户外平台/门廊）
        porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        available_porch = [col for col in porch_cols if col in df.columns]
        if available_porch:
            df['TotalPorchSF'] = df[available_porch].sum(axis=1)
            self.new_features.append('TotalPorchSF')
        
        # 总浴室数（加权计算：全浴室=1，半浴室=0.5，包含地下室）
        full_bath = df.get('FullBath', 0) + df.get('BsmtFullBath', 0)
        half_bath = 0.5 * (df.get('HalfBath', 0) + df.get('BsmtHalfBath', 0))
        df['TotalBath'] = full_bath + half_bath
        self.new_features.append('TotalBath')
        
        # 房屋占地比例（居住面积/地块面积）
        if 'GrLivArea' in df.columns and 'LotArea' in df.columns:
            df['HouseToLotRatio'] = df['GrLivArea'] / (df['LotArea'] + 1)
            self.new_features.append('HouseToLotRatio')
        
        # 每房间平均面积（空间利用率指标）
        if 'GrLivArea' in df.columns and 'TotRmsAbvGrd' in df.columns:
            df['LivingAreaPerRoom'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)
            self.new_features.append('LivingAreaPerRoom')
        
        # 卧室占比（卧室数量/总房间数）
        if 'BedroomAbvGr' in df.columns and 'TotRmsAbvGrd' in df.columns:
            df['BedroomAbvGrRatio'] = df['BedroomAbvGr'] / (df['TotRmsAbvGrd'] + 1)
            self.new_features.append('BedroomAbvGrRatio')
        
        # 是否有开放式门廊（二值）
        if 'OpenPorchSF' in df.columns:
            df['HasOpenPorch'] = (df['OpenPorchSF'] > 0).astype(int)
            self.new_features.append('HasOpenPorch')
        
        # 是否有木质甲板（二值）
        if 'WoodDeckSF' in df.columns:
            df['HasWoodDeck'] = (df['WoodDeckSF'] > 0).astype(int)
            self.new_features.append('HasWoodDeck')
        
        return df
    
    def _create_time_features(self, df):
        """
        创建时间衍生特征
        利用YearBuilt, YearRemodAdd, GarageYrBlt, YrSold, MoSold创建年龄和时机特征
        """
        if 'YrSold' not in df.columns:
            return df
        
        # 房屋年龄（销售时）
        if 'YearBuilt' in df.columns:
            df['HouseAge'] = df['YrSold'] - df['YearBuilt']
            self.new_features.append('HouseAge')
        
        # 是否新房（销售年份=建造年份）
        if 'YearBuilt' in df.columns:
            df['IsNewHouse'] = (df['YrSold'] == df['YearBuilt']).astype(int)
            self.new_features.append('IsNewHouse')
        
        # 翻新后年数（销售时距翻新的年数）
        if 'YearRemodAdd' in df.columns:
            df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
            self.new_features.append('RemodAge')
        
        # 车库年龄
        if 'GarageYrBlt' in df.columns:
            df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
            self.new_features.append('GarageAge')
        
        # 销售季节（将月份映射为季节）
        if 'MoSold' in df.columns:
            season_map = {
                12: 0, 1: 0, 2: 0,   # 冬季
                3: 1, 4: 1, 5: 1,    # 春季
                6: 2, 7: 2, 8: 2,    # 夏季
                9: 3, 10: 3, 11: 3   # 秋季
            }
            df['SeasonSold'] = df['MoSold'].map(season_map)
            self.new_features.append('SeasonSold')
        
        # 是否翻新过（翻新年份不等于建造年份）
        if 'YearRemodAdd' in df.columns and 'YearBuilt' in df.columns:
            df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
            self.new_features.append('IsRemodeled')
        
        return df
    
    def _create_quality_features(self, df):
        """
        创建质量相关交互特征和序数编码
        质量评级特征（Ex, Gd, TA, Fa, Po）转换为数值
        """
        # 质量与面积的交互（关键特征：质量好的大房子更值钱）
        if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
            df['QualSF'] = df['OverallQual'] * df['GrLivArea']
            self.new_features.append('QualSF')
        
        # 质量与状态的交互
        if 'OverallQual' in df.columns and 'OverallCond' in df.columns:
            df['QualCond'] = df['OverallQual'] * df['OverallCond']
            self.new_features.append('QualCond')
        
        # 对质量评级特征进行序数编码
        qual_cols = [
            'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
            'HeatingQC', 'KitchenQual', 'FireplaceQu', 
            'GarageQual', 'GarageCond', 'PoolQC'
        ]
        
        for col in qual_cols:
            if col in df.columns:
                encoded_col = f'{col}_encoded'
                df[encoded_col] = df[col].map(self.ordinal_mappings).fillna(0)
                self.new_features.append(encoded_col)
        
        # 外部总评分（质量+状况）
        if 'ExterQual_encoded' in df.columns and 'ExterCond_encoded' in df.columns:
            df['ExterScore'] = df['ExterQual_encoded'] + df['ExterCond_encoded']
            self.new_features.append('ExterScore')
        
        # 地下室总评分
        if 'BsmtQual_encoded' in df.columns and 'BsmtCond_encoded' in df.columns:
            df['BsmtScore'] = df['BsmtQual_encoded'] + df['BsmtCond_encoded']
            self.new_features.append('BsmtScore')
        
        # 厨房综合评分（质量×数量）
        if 'KitchenQual_encoded' in df.columns and 'KitchenAbvGr' in df.columns:
            df['KitchenScore'] = df['KitchenQual_encoded'] * df['KitchenAbvGr']
            self.new_features.append('KitchenScore')
        
        # 车库综合评分（质量+状况+完成度）
        if 'GarageFinish' in df.columns:
            finish_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NoGarage': 0, 'None': 0}
            df['GarageFinish_encoded'] = df['GarageFinish'].map(finish_map).fillna(0)
            
            if 'GarageQual_encoded' in df.columns and 'GarageCond_encoded' in df.columns:
                df['GarageScore'] = (
                    df['GarageQual_encoded'] + 
                    df['GarageCond_encoded'] + 
                    df['GarageFinish_encoded']
                )
                self.new_features.append('GarageScore')
                self.new_features.append('GarageFinish_encoded')
        
        return df
    
    def _create_advanced_features(self, df, target_col):
        """
        创建高级交互特征
        包括目标编码（Target Encoding）和复杂交互
        """
        # Neighborhood目标编码（使用KFold防止泄露）
        if 'Neighborhood' in df.columns and target_col in df.columns:
            df['NeighborhoodPrice'] = self._target_encode(df, 'Neighborhood', target_col)
            self.new_features.append('NeighborhoodPrice')
        
        # 建筑类型目标编码
        if 'MSSubClass' in df.columns and target_col in df.columns:
            df['MSSubClassPrice'] = self._target_encode(df, 'MSSubClass', target_col)
            self.new_features.append('MSSubClassPrice')
        
        # 高质量房屋面积（质量>=8的居住面积）
        if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
            df['HighQualSF'] = np.where(df['OverallQual'] >= 8, df['GrLivArea'], 0)
            self.new_features.append('HighQualSF')
        
        # 低质量房屋面积（质量<=4的居住面积）
        if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
            df['LowQualSF'] = np.where(df['OverallQual'] <= 4, df['GrLivArea'], 0)
            self.new_features.append('LowQualSF')
        
        # 空间宽敞度（每卧室平均面积）
        if 'GrLivArea' in df.columns and 'BedroomAbvGr' in df.columns:
            df['Spaciousness'] = df['GrLivArea'] / (df['BedroomAbvGr'] + 1)
            self.new_features.append('Spaciousness')
        
        # 便利设施指数（浴室数+车位数）
        if 'FullBath' in df.columns and 'GarageCars' in df.columns:
            df['Convenience'] = df['FullBath'] + df['GarageCars']
            self.new_features.append('Convenience')
        
        return df
    
    def _target_encode(self, df, col, target_col, n_splits=5):
        """
        使用KFold进行目标编码，防止数据泄露和过拟合
        
        Parameters:
        -----------
        col : str
            要编码的分类列
        target_col : str
            目标变量列
        n_splits : int
            KFold折数
            
        Returns:
        --------
        pd.Series
            编码后的数值
        """
        if col not in df.columns or target_col not in df.columns:
            return pd.Series([0] * len(df), index=df.index)
        
        global_mean = df[target_col].mean()
        encoded = pd.Series(index=df.index, dtype=float)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(df):
            train_data = df.iloc[train_idx]
            means = train_data.groupby(col)[target_col].mean()
            encoded.iloc[val_idx] = df.iloc[val_idx][col].map(means)
        
        # 填充缺失值（未出现的类别用全局均值）
        encoded = encoded.fillna(global_mean)
        
        return encoded
    
    def _encode_categorical(self, df, target_col):
        """
        分类特征编码
        - 低基数（<=10）：One-Hot编码
        - 高基数（>10）：Label编码（保留原始列用于解释，添加编码列）
        """
        exclude_cols = [target_col, 'Id']
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        # 分离高基数和低基数特征
        low_cardinality = [col for col in categorical_cols if df[col].nunique() <= 10]
        high_cardinality = [col for col in categorical_cols if df[col].nunique() > 10]
        
        # One-Hot编码低基数特征
        if low_cardinality:
            df = pd.get_dummies(df, columns=low_cardinality, drop_first=True)
        
        # Label编码高基数特征（保留原始列）
        for col in high_cardinality:
            if col not in ['Neighborhood', 'MSSubClass']:  # 这两个已做目标编码
                le = LabelEncoder()
                encoded_values = le.fit_transform(df[col].astype(str))
                df[f'{col}_encoded'] = encoded_values
                self.label_encoders[col] = le
        
        return df
    
    def _apply_log_transform(self, df, target_col):
        """
        对偏态分布的特征进行对数变换
        特别是目标变量SalePrice（右偏分布）
        """
        # 目标变量对数变换（关键：房价通常是对数正态分布）
        if target_col in df.columns:
            df[f'{target_col}_log'] = np.log1p(df[target_col])
        
        # 对高度偏态的面积特征进行对数变换
        skewed_features = [
            'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'LotArea', 
            'TotalSF', 'TotalPorchSF'
        ]
        
        for col in skewed_features:
            if col in df.columns:
                # 确保没有负值
                if (df[col] >= 0).all():
                    log_col = f'{col}_log'
                    df[log_col] = np.log1p(df[col])
                    self.new_features.append(log_col)
        
        return df


def main():
    """
    主执行函数：读取数据、执行特征工程、保存结果
    """
    # 配置路径
    input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_path = '/Users/cjialin/code/AutoMLByLLM/train_features.csv'
    target_col = 'SalePrice'
    
    print("="*60)
    print("房价预测特征工程 - 开始执行")
    print("="*60)
    
    # 读取数据
    print(f"\n正在读取数据: {input_path}")
    try:
        df = pd.read_csv(input_path)
        print(f"数据读取成功！形状: {df.shape}")
        print(f"列名: {list(df.columns[:10])}... (共{len(df.columns)}列)")
    except Exception as e:
        print(f"读取数据失败: {e}")
        return
    
    # 检查目标列
    if target_col not in df.columns:
        print(f"错误: 目标列 '{target_col}' 不存在于数据中")
        return
    
    # 执行特征工程
    print("\n" + "="*60)
    print("开始特征工程处理")
    print("="*60)
    
    engineer = HousePriceFeatureEngineer()
    df_processed, new_features = engineer.fit_transform(df, target_col=target_col)
    
    # 保存结果
    print(f"\n正在保存特征工程后的数据...")
    try:
        df_processed.to_csv(output_path, index=False)
        print(f"保存成功: {output_path}")
    except Exception as e:
        print(f"保存失败: {e}")
        return
    
    # 输出报告
    print("\n" + "="*60)
    print("特征工程报告")
    print("="*60)
    print(f"原始特征数: {len(df.columns)}")
    print(f"新生成特征数: {len(new_features)}")
    print(f"最终总特征数: {len(df_processed.columns)}")
    print(f"\n新生成的特征列表:")
    print("-"*60)
    
    for i, feature in enumerate(new_features, 1):
        print(f"{i:2d}. {feature:<30}", end="")
        if i % 2 == 0:
            print()
    if len(new_features) % 2 != 0:
        print()
    
    print("-"*60)
    print(f"\n处理完成！数据已保存至: {output_path}")
    print("="*60)
    
    return df_processed, new_features


if __name__ == "__main__":
    main()

print("✓ 特征工程完成")


# ============================================
# 阶段 3: 模型训练
# ============================================
print("\n[阶段 3] 模型训练...")

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
