# 房价预测数据清洗方案

## 1. 任务背景与目标

### 1.1 任务描述
- **目标**: 预测每套房子的售价（SalePrice）
- **评估指标**: RMSE（均方根误差）
- **数据规模**: 1,460 行 × 81 列
- **目标变量**: SalePrice（连续型数值）

### 1.2 清洗目标
确保数据质量，提升模型预测性能，降低RMSE误差。

---

## 2. 数据质量问题概览

根据数据分析，发现以下主要质量问题：

| 问题类型 | 数量 | 严重程度 |
|---------|------|---------|
| 高缺失率列（>50%） | 5列 | 🔴 严重 |
| 中等缺失率列（5%-50%） | 7列 | 🟡 中等 |
| 低缺失率列（<5%） | 7列 | 🟢 轻微 |
| 异常值列 | 25列 | 🟡 中等 |
| 重复行 | 0行 | 🟢 无问题 |
| 类型优化 | 38列 | 🟢 可优化 |

---

## 3. 详细清洗方案

### 3.1 缺失值处理策略

#### 3.1.1 删除高缺失率列（缺失率 > 50%）
这些列缺失过多，填充可能引入噪音，建议直接删除。

| 列名 | 缺失率 | 处理方式 |
|------|--------|---------|
| PoolQC | 99.52% | 删除列 |
| MiscFeature | 96.30% | 删除列 |
| Alley | 93.77% | 删除列 |
| Fence | 80.75% | 删除列 |
| MasVnrType | 59.73% | 删除列 |

**业务解释**: 这些特征（泳池质量、杂项功能、小巷通道、围栏、砌体饰面类型）在大多数房屋中不存在，属于"不存在即缺失"类型，保留价值低。

#### 3.1.2 中等缺失率列处理（5% < 缺失率 < 50%）

| 列名 | 缺失率 | 数据类型 | 填充策略 | 填充值 |
|------|--------|---------|---------|--------|
| FireplaceQu | 47.26% | object | 常量填充 | "None"（表示无壁炉） |
| LotFrontage | 17.74% | float64 | 分组中位数填充 | 按Neighborhood分组的中位数 |
| GarageType | 5.55% | object | 常量填充 | "None"（表示无车库） |
| GarageYrBlt | 5.55% | float64 | 派生填充 | 用YearBuilt（房屋建造年份）填充 |
| GarageFinish | 5.55% | object | 常量填充 | "None" |
| GarageQual | 5.55% | object | 常量填充 | "None" |
| GarageCond | 5.55% | object | 常量填充 | "None" |

**业务解释**:
- FireplaceQu缺失表示房屋没有壁炉
- Garage相关列缺失表示房屋没有车库
- LotFrontage使用邻里（Neighborhood）分组中位数填充，因为同区域的房屋通常有相似的临街面宽度

#### 3.1.3 低缺失率列处理（缺失率 < 5%）

| 列名 | 缺失率 | 数据类型 | 填充策略 | 填充值 |
|------|--------|---------|---------|--------|
| BsmtExposure | 2.60% | object | 常量填充 | "No"（无暴露） |
| BsmtFinType2 | 2.60% | object | 常量填充 | "Unf"（未装修） |
| BsmtQual | 2.53% | object | 众数填充 | 众数（TA - Typical） |
| BsmtCond | 2.53% | object | 众数填充 | 众数（TA - Typical） |
| BsmtFinType1 | 2.53% | object | 众数填充 | 众数 |
| MasVnrArea | 0.55% | float64 | 常量填充 | 0（表示无砌体饰面） |
| Electrical | 0.07% | object | 众数填充 | 众数（SBrkr） |

---

### 3.2 异常值处理策略

#### 3.2.1 删除异常特征列
以下列几乎全为0值，方差极低，对预测无贡献：

| 列名 | 异常值比例 | 处理方式 | 原因 |
|------|-----------|---------|------|
| BsmtFinSF2 | 11.44% | 删除列 | 地下室第二区域面积，绝大多数为0 |
| EnclosedPorch | 14.25% | 删除列 | 封闭式门廊面积，绝大多数房屋没有 |

#### 3.2.2 Winsorize异常值处理（缩尾处理）
对以下数值列进行上下1%缩尾处理（或根据业务范围截断）：

| 列名 | 异常值比例 | 处理范围 | 业务解释 |
|------|-----------|---------|---------|
| MSSubClass | 7.05% | [20, 190] | 住宅类型代码，截断极值 |
| LotFrontage | 6.03% | [0, Q3+1.5IQR] | 临街面宽度，不能为负 |
| LotArea | 4.73% | [1%, 99%] | 地块面积，缩尾处理 |
| OverallCond | 8.56% | [1, 9] | 整体状况评分，1-10范围 |
| MasVnrArea | 6.58% | [0, Q3+1.5IQR] | 砌体饰面面积，不能为负 |
| BsmtUnfSF | 1.99% | [0, Q3+1.5IQR] | 地下室未完工面积 |
| TotalBsmtSF | 4.18% | [0, Q3+1.5IQR] | 地下室总面积 |
| 1stFlrSF | 1.37% | [0, Q3+1.5IQR] | 第一层面积 |
| LowQualFinSF | 1.78% | [0, Q3+1.5IQR] | 低质量装修面积 |
| GrLivArea | 2.12% | [0, Q3+1.5IQR] | 地上生活面积 |
| BsmtHalfBath | 5.62% | [0, 2] | 地下室半浴室数量 |
| BedroomAbvGr | 2.40% | [0, 6] | 地上卧室数量 |
| KitchenAbvGr | 4.66% | [1, 3] | 厨房数量，至少1个 |
| TotRmsAbvGrd | 2.05% | [3, 12] | 总房间数 |
| GarageArea | 1.44% | [0, Q3+1.5IQR] | 车库面积 |
| WoodDeckSF | 2.19% | [0, Q3+1.5IQR] | 木质甲板面积 |
| OpenPorchSF | 5.27% | [0, Q3+1.5IQR] | 开放式门廊面积 |
| 3SsnPorch | 1.64% | [0, Q3+1.5IQR] | 三季门廊面积 |
| ScreenPorch | 7.95% | [0, Q3+1.5IQR] | 纱门门廊面积 |
| MiscVal | 3.56% | [0, Q3+1.5IQR] | 杂项价值 |
| **SalePrice** | **4.18%** | **[1%, 99%]** | **目标变量，需处理** |

#### 3.2.3 保留的异常值
以下列虽有异常值，但属于合理业务范围，保留：

| 列名 | 原因 |
|------|------|
| OverallQual | 1-10评分，范围合理 |
| YearBuilt | 包含历史建筑，合理 |
| BsmtFinSF1 | 地下室装修面积差异大，合理 |
| 2ndFlrSF | 部分房屋无二层，0值合理 |
| BsmtFullBath | 计数变量，合理 |
| Fireplaces | 壁炉数量差异，合理 |
| GarageCars | 车位数量差异，合理 |
| PoolArea | 少数有泳池，合理 |

---

### 3.3 数据类型优化

将以下43个object列转换为category类型，优化内存和模型性能：

```python
# 需要转换为category的列
category_columns = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType',
    'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterQual',
    'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
    'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
    'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
    'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
]
```

---

### 3.4 特征工程（可选但推荐）

#### 3.4.1 创建新特征
| 新特征名 | 计算方式 | 业务含义 |
|---------|---------|---------|
| TotalSF | GrLivArea + TotalBsmtSF | 房屋总面积 |
| HouseAge | YearSold - YearBuilt | 房龄 |
| TotalBathrooms | FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath | 总浴室数 |
| HasPool | PoolArea > 0 | 是否有泳池 |
| HasGarage | GarageArea > 0 | 是否有车库 |
| HasBasement | TotalBsmtSF > 0 | 是否有地下室 |
| HasFireplace | Fireplaces > 0 | 是否有壁炉 |

---

## 4. 清洗执行代码

```python
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def clean_house_price_data(file_path):
    """
    房价预测数据清洗函数
    参数: file_path - 原始数据路径
    返回: 清洗后的DataFrame
    """
    # 1. 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # 2. 删除高缺失率列（>50%）
    high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df = df.drop(columns=high_missing_cols)
    print(f"删除高缺失列后: {df.shape}")
    
    # 3. 删除低方差列（几乎全为0）
    low_variance_cols = ['BsmtFinSF2', 'EnclosedPorch']
    df = df.drop(columns=low_variance_cols, errors='ignore')
    
    # 4. 处理中等缺失率列
    # FireplaceQu - 缺失表示无壁炉
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
    
    # LotFrontage - 按Neighborhood分组中位数填充
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    # 如果仍有缺失，用整体中位数填充
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # Garage相关列 - 缺失表示无车库
    garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cols:
        df[col] = df[col].fillna('None')
    
    # GarageYrBlt - 用YearBuilt填充
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    
    # 5. 处理低缺失率列
    # Basement相关列
    df['BsmtExposure'] = df['BsmtExposure'].fillna('No')
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna('Unf')
    df['BsmtQual'] = df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
    df['BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
    
    # MasVnrArea - 缺失表示无砌体饰面
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # Electrical - 用众数填充
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    # 6. 异常值处理 - Winsorize
    # 需要缩尾的数值列
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'LowQualFinSF', 'GrLivArea',
        'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'GarageArea', 'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 
        'ScreenPorch', 'MiscVal', 'SalePrice'
    ]
    
    for col in winsorize_cols:
        if col in df.columns:
            # 使用1%和99%分位数进行缩尾
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
    
    # 7. 确保非负的列
    non_negative_cols = [
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF',
        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea',
        'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
    ]
    
    for col in non_negative_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    
    # 8. 数据类型转换
    category_columns = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond',
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'
    ]
    
    # 过滤掉已删除的列
    category_columns = [col for col in category_columns if col in df.columns]
    df[category_columns] = df[category_columns].astype('category')
    
    # 9. 特征工程（可选）
    # 总面积
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
    
    # 是否有地下室
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    
    # 是否有车库
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    
    # 是否有泳池
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)