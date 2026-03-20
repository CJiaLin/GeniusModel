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