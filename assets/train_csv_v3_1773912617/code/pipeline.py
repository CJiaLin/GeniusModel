#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoML 全流程建模脚本

生成时间: 2026-03-19 17:41:09
会话ID: train_csv_v3_1773912617

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
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 定义数据路径
input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'

# 1. 读取原始数据
df = pd.read_csv(input_path)
original_shape = df.shape
original_missing = df.isnull().sum().sum()

print("开始数据清洗...")
print(f"原始数据形状: {original_shape}")
print(f"原始缺失值总数: {original_missing}")

# 2. 缺失值处理

# 2.1 Cabin - 缺失率77%，转换为二元特征（有/无船舱）
df['Has_Cabin'] = df['Cabin'].notna().astype(int)
df.drop('Cabin', axis=1, inplace=True)
print("✓ Cabin处理完成：转换为Has_Cabin二元特征")

# 2.2 Age - 按Pclass和Sex分组填充中位数
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
    lambda x: x.fillna(x.median())
)
print("✓ Age处理完成：使用Pclass和Sex分组中位数填充")

# 2.3 Embarked - 众数填充（仅2个缺失值）
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace=True)
print(f"✓ Embarked处理完成：使用众数'{mode_embarked}'填充")

# 3. 异常值处理

# 3.1 Fare - 使用对数转换缓解右偏分布和极值影响
df['Fare_log'] = np.log1p(df['Fare'])
print("✓ Fare处理完成：添加对数转换特征Fare_log")

# 4. 特征工程

# 4.1 从Name提取Title（称谓）
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.',
                                     expand=False)

# 统一稀有称谓
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
               'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
print("✓ Title提取完成：统一称谓类别")

# 4.2 创建家庭规模特征
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 4.3 创建是否独行特征
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
print("✓ 家庭特征创建完成：FamilySize和IsAlone")

# 4.4 从Ticket提取前缀
df['Ticket_Prefix'] = df['Ticket'].str.extract('([A-Za-z]+)', expand=False)
df['Ticket_Prefix'] = df['Ticket_Prefix'].fillna('None')
print("✓ Ticket前缀提取完成")

# 5. 数据类型转换与编码

# 5.1 Sex和Embarked映射编码
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
print("✓ 类别编码完成：Sex和Embarked已映射为数值")

# 5.2 Title标签编码
le = LabelEncoder()
df['Title_Encoded'] = le.fit_transform(df['Title'])
print(f"✓ Title编码完成：共{len(le.classes_)}个类别")

# 6. 删除冗余列
drop_columns = ['PassengerId', 'Name', 'Ticket', 'Title', 'Ticket_Prefix']
df_cleaned = df.drop(columns=drop_columns)
print(f"✓ 删除冗余列完成：移除了{len(drop_columns)}个原始列")

# 7. 保存清洗后的数据
df_cleaned.to_csv(output_path, index=False)
print(f"✓ 数据已保存至: {output_path}")

# 8. 生成清洗结果统计
cleaned_shape = df_cleaned.shape
cleaned_missing = df_cleaned.isnull().sum().sum()
features_added = cleaned_shape[1] - original_shape[1]

# 打印详细统计报告
print("\n" + "="*50)
print("数据清洗统计报告")
print("="*50)
print(f"原始数据形状: {original_shape}")
print(f"清洗后数据形状: {cleaned_shape}")
print(f"原始缺失值总数: {original_missing}")
print(f"清洗后缺失值总数: {cleaned_missing}")
print(f"新增特征数: {features_added}")
print(f"处理样本数: {cleaned_shape[0]}")
print(f"最终特征数: {cleaned_shape[1]}")
print("="*50)

# 返回统计字典（供下游使用）
cleaning_stats = {
    'original_shape': original_shape,
    'cleaned_shape': cleaned_shape,
    'original_missing': int(original_missing),
    'cleaned_missing': int(cleaned_missing),
    'features_added': features_added,
    'rows_processed': cleaned_shape[0],
    'final_features': cleaned_shape[1],
    'output_path': output_path
}

print("\n清洗后的特征列表:")
for i, col in enumerate(df_cleaned.columns, 1):
    missing_count = df_cleaned[col].isnull().sum()
    print(f"{i:2d}. {col:15s} (缺失值: {missing_count})")

print("\n数据清洗流程执行完毕！")

print("✓ 数据清洗完成")


# ============================================
# 阶段 2: 特征工程
# ============================================
print("\n[阶段 2] 特征工程...")

```python
"""
房屋价格预测特征工程完整实现
数据路径: /Users/cjialin/code/AutoMLByLLM/train.csv
目标列: SalePrice
任务类型: Regression
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
import json

# 忽略警告
warnings.filterwarnings('ignore')


class HousePriceFeatureEngineer:
    """
    房价预测特征工程类
    基于Kaggle House Prices竞赛的最佳实践
    """
    
    def __init__(self):
        # 质量等级映射 (Excellent -> Poor -> NA)
        self.quality_mapping = {
            'Ex': 5,  # Excellent
            'Gd': 4,  # Good
            'TA': 3,  # Typical/Average
            'Fa': 2,  # Fair
            'Po': 1,  # Poor
            'NA': 0,  # No feature
            np.nan: 0  # 缺失值也映射为0
        }
        
        # 功能完整性评分映射
        self.functional_mapping = {
            'Typ': 8,   # Typical Functionality
            'Min1': 7,  # Minor Deductions 1
            'Min2': 6,  # Minor Deductions 2
            'Mod': 5,   # Moderate Deductions
            'Maj1': 4,  # Major Deductions 1
            'Maj2': 3,  # Major Deductions 2
            'Sev': 2,   # Severely Damaged
            'Sal': 1,   # Salvage only
            np.nan: 0
        }
        
        # 目标编码映射（需要在fit阶段计算）
        self.neighborhood_mean_price: Optional[Dict] = None
        self.mszoning_mean_price: Optional[Dict] = None
        self.neighborhood_std_price: Optional[Dict] = None
        
        # 记录新生成的特征
        self.new_features: List[str] = []
        
        # 记录缺失值填充策略
        self.fill_values: Dict = {}
        
    def fit(self, df: pd.DataFrame, target_col: str = 'SalePrice') -> 'HousePriceFeatureEngineer':
        """
        拟合特征工程器，计算目标编码等依赖训练集的统计量
        
        Parameters:
        -----------
        df : pd.DataFrame
            训练数据
        target_col : str
            目标列名
            
        Returns:
        --------
        self : HousePriceFeatureEngineer
            返回自身以支持链式调用
        """
        print(f"[Fit] 开始拟合特征工程器，数据形状: {df.shape}")
        
        # 1. 计算目标编码 - Neighborhood
        self.neighborhood_mean_price = df.groupby('Neighborhood')[target_col].mean().to_dict()
        self.neighborhood_std_price = df.groupby('Neighborhood')[target_col].std().to_dict()
        
        # 2. 计算目标编码 - MSZoning
        self.mszoning_mean_price = df.groupby('MSZoning')[target_col].mean().to_dict()
        
        # 3. 记录LotFrontage按Neighborhood的填充值
        self.fill_values['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].median().to_dict()
        
        # 4. 记录数值型特征的众数/中位数（用于测试集填充）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != target_col:
                if df[col].isnull().sum() > 0:
                    self.fill_values[col] = df[col].median()
        
        # 5. 记录类别型特征的众数
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                self.fill_values[col] = df[col].mode()[0] if not df[col].mode().empty else 'None'
        
        print(f"[Fit] 拟合完成，记录了 {len(self.fill_values)} 个填充值")
        return self
    
    def transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        执行特征工程转换
        
        Parameters:
        -----------
        df : pd.DataFrame
            输入数据
        is_train : bool
            是否为训练集（影响某些特征的计算方式）
            
        Returns:
        --------
        df : pd.DataFrame
            特征工程后的数据
        """
        print(f"[Transform] 开始特征工程，原始数据形状: {df.shape}")
        df = df.copy()
        self.new_features = []  # 重置新特征列表
        
        # ==================== 阶段 1: 缺失值处理 ====================
        df = self._handle_missing_values(df)
        
        # ==================== 阶段 2: 异常值处理 ====================
        df = self._handle_outliers(df)
        
        # ==================== 阶段 3: 质量特征有序编码 ====================
        df = self._encode_quality_features(df)
        
        # ==================== 阶段 4: 面积相关特征 ====================
        df = self._create_area_features(df)
        
        # ==================== 阶段 5: 房间结构特征 ====================
        df = self._create_room_features(df)
        
        # ==================== 阶段 6: 时间相关特征 ====================
        df = self._create_time_features(df)
        
        # ==================== 阶段 7: 质量评分特征 ====================
        df = self._create_quality_score_features(df)
        
        # ==================== 阶段 8: 功能与条件特征 ====================
        df = self._create_functional_features(df)
        
        # ==================== 阶段 9: 目标编码 ====================
        df = self._create_target_encoding_features(df)
        
        # ==================== 阶段 10: 交互特征 ====================
        df = self._create_interaction_features(df)
        
        # ==================== 阶段 11: 多项式与变换特征 ====================
        df = self._create_polynomial_features(df)
        
        # ==================== 阶段 12: 对数变换（针对右偏特征） ====================
        df = self._apply_log_transform(df)
        
        print(f"[Transform] 特征工程完成，最终数据形状: {df.shape}")
        print(f"[Transform] 新生成特征数量: {len(self.new_features)}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        策略基于特征工程方案第2.1节
        """
        print("  [缺失值处理] 开始...")
        
        # 1. 缺失表示"无该设施"的特征 - 填充"None"或"NA"
        none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
        for col in none_cols:
            if col in df.columns:
                df[col] = df[col].fillna('None')
        
        # 2. 车库相关特征 - 缺失表示无车库
        garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
        for col in garage_cols:
            if col in df.columns:
                df[col] = df[col].fillna('None')
        
        # GarageYrBlt 填充0（表示无车库）
        if 'GarageYrBlt' in df.columns:
            df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
        
        # GarageArea, GarageCars 填充0
        for col in ['GarageArea', 'GarageCars']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # 3. 地下室相关特征 - 缺失表示无地下室
        bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        for col in bsmt_cols:
            if col in df.columns:
                df[col] = df[col].fillna('None')
        
        # 地下室数值特征填充0
        bsmt_numeric = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
        for col in bsmt_numeric:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # 4. 砌体贴面特征
        if 'MasVnrType' in df.columns:
            df['MasVnrType'] = df['MasVnrType'].fillna('None')
        if 'MasVnrArea' in df.columns:
            df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
        
        # 5. LotFrontage - 按Neighborhood中位数填充
        if 'LotFrontage' in df.columns and self.fill_values.get('LotFrontage'):
            df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median() if not x.median() != x.median() else self.fill_values.get('LotFrontage', {}).get(x.name, 0))
            )
            # 如果仍有缺失，用全局中位数填充
            df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
        
        # 6. Electrical - 填充众数
        if 'Electrical' in df.columns:
            df['Electrical'] = df['Electrical'].fillna('SBrkr')  # Standard Circuit Breakers
        
        # 7. 其他类别型特征填充"None"或众数
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna('None')
        
        # 8. 其他数值型特征填充0或中位数
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(0)
        
        print(f"    完成，剩余缺失值: {df.isnull().sum().sum()}")
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理异常值
        基于面积与价格的关系识别异常
        """
        print("  [异常值处理] 开始...")
        
        # 标记异常值（但不删除，仅标记供后续使用）
        # 基于GrLivArea的异常（超过4000平方英尺可能是异常）
        if 'GrLivArea' in df.columns:
            df['IsGrLivAreaOutlier'] = (df['GrLivArea'] > 4000).astype(int)
            self.new_features.append('IsGrLivAreaOutlier')
        
        # 基于LotArea的异常（超过100000可能是异常）
        if 'LotArea' in df.columns:
            df['IsLotAreaOutlier'] = (df['LotArea'] > 100000).astype(int)
            self.new_features.append('IsLotAreaOutlier')
        
        print(f"    完成，标记了 {df['IsGrLivAreaOutlier'].sum() if 'IsGrLivAreaOutlier' in df.columns else 0} 个GrLivArea异常")
        return df
    
    def _encode_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        质量特征有序编码
        将类别质量等级转换为数值评分
        """
        print("  [质量特征编码] 开始...")
        
        quality_cols = [
            'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
            'HeatingQC', 'KitchenQual', 'FireplaceQu', 
            'GarageQual', 'GarageCond', 'PoolQC'
        ]
        
        for col in quality_cols:
            if col in df.columns:
                # 应用映射，未匹配的值默认为0
                df[col] = df[col].map(self.quality_mapping).fillna(0).astype(int)
        
        # Functional 特殊处理
        if 'Functional' in df.columns:
            df['Functional'] = df['Functional'].map(self.functional_mapping).fillna(5)
        
        print(f"    完成，编码了 {len([c for c in quality_cols if c in df.columns])} 个质量特征")
        return df
    
    def _create_area_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建面积相关新特征（方案第3.1节）
        """
        print("  [面积特征] 开始创建...")
        
        # 1. TotalSF: 房屋总平方英尺（地上+地下）
        if all(col in df.columns for col in ['TotalBsmtSF', 'GrLivArea']):
            df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']
            self.new_features.append('TotalSF')
        
        # 2. TotalArea: 包含附属设施的总面积
        area_cols = ['TotalSF', 'GarageArea', 'OpenPorchSF', 'EnclosedPorch', 
                     '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']
        available_cols = [c for c in area_cols if c in df.columns]
        if available_cols:
            df['TotalArea'] = df[available_cols].sum(axis=1)
            self.new_features.append('TotalArea')
        
        # 3. LivingAreaRatio: 居住面积占比
        if all(col in df.columns for col in ['GrLivArea', 'LotArea']):
            df['LivingAreaRatio'] = df['GrLivArea'] / (df['LotArea'] + 1)  # +1避免除0
            self.new_features.append('LivingAreaRatio')
        
        # 4-10. 二元特征：是否有某设施
        binary_features = {
            'Has2ndFloor': ('2ndFlrSF', lambda x: x > 0),
            'HasBasement': ('TotalBsmtSF', lambda x: x > 0),
            'HasGarage': ('GarageArea', lambda x: x > 0),
            'HasPool': ('PoolArea', lambda x: x > 0),
            'HasFireplace': ('Fireplaces', lambda x: x > 0),
            'HasDeck': ('WoodDeckSF', lambda x: x > 0),
        }
        
        for new_col, (src_col, condition) in binary_features.items():
            if src_col in df.columns:
                df[new_col] = condition(df[src_col]).astype(int)
                self.new_features.append(new_col)
        
        # HasPorch: 是否有门廊（任何类型）
        porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        available_porch = [c for c in porch_cols if c in df.columns]
        if available_porch:
            df['HasPorch'] = (df[available_porch].sum(axis=1) > 0).astype(int)
            self.new_features.append('HasPorch')
        
        # 11. TotalPorchSF: 门廊和甲板总面积
        porch_deck_cols = ['OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 
                          'ScreenPorch', 'WoodDeckSF']
        available_porch_deck = [c for c in porch_deck_cols if c in df.columns]
        if available_porch_deck:
            df['TotalPorchSF'] = df[available_porch_deck].sum(axis=1)
            self.new_features.append('TotalPorchSF')
        
        # 12. LotFrontageRatio: 临街距离比例
        if all(col in df.columns for col in ['LotFrontage', 'LotArea']):
            df['LotFrontageRatio'] = df['LotFrontage'] / (df['LotArea'] + 1)
            self.new_features.append('LotFrontageRatio')
        
        print(f"    完成，创建了 {len([f for f in self.new_features if 'Has' in f or 'SF' in f or 'Area' in f])} 个面积相关特征")
        return df
    
    def _create_room_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建房间结构新特征（方案第3.2节）
        """
        print("  [房间特征] 开始创建...")
        
        # 1. TotalBathrooms: 总浴室当量数
        bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        available_bath = [c for c in bath_cols if c in df.columns]
        if available_bath:
            df['TotalBathrooms'] = (
                df.get('FullBath', 0) + 
                0.5 * df.get('HalfBath', 0) + 
                df.get('BsmtFullBath', 0) + 
                0.5 * df.get('BsmtHalfBath',
print("✓ 特征工程完成")


# ============================================
# 阶段 3: 模型训练
# ============================================
print("\n[阶段 3] 模型训练...")

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
