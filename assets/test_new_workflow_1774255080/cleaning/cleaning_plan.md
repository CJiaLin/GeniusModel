# 房价预测数据清洗方案

## 1. 方案概述

### 1.1 任务背景
- **目标**: 预测房价（SalePrice）
- **评估指标**: RMSE（均方根误差）
- **数据规模**: 1460 行 × 81 列
- **数据特点**: 包含38个数值列和43个分类列

### 1.2 清洗策略总览

| 问题类型 | 处理数量 | 主要策略 |
|---------|---------|---------|
| 高缺失率列(>50%) | 5列 | 直接删除 |
| 中等缺失率列(5%-50%) | 8列 | 基于业务逻辑填充 |
| 低缺失率列(<5%) | 6列 | 众数/中位数填充 |
| 异常值处理 | 20列 | Winsorize（缩尾处理） |
| 零方差/近零方差列 | 2列 | 删除 |
| 数据类型转换 | 43列 | 转换为category类型 |

---

## 2. 详细清洗步骤

### 步骤1: 高缺失率列删除
删除缺失比例超过50%的列，这些列信息含量过低：

- **PoolQC** (99.52%缺失) - 游泳池质量，绝大多数房屋无游泳池
- **MiscFeature** (96.3%缺失) - 其他杂项功能
- **Alley** (93.77%缺失) - 小巷通道类型
- **Fence** (80.75%缺失) - 围栏质量
- **MasVnrType** (59.73%缺失) - 砌体饰面类型

### 步骤2: 近零方差列删除
删除几乎无变化的列：

- **BsmtFinSF2** (11.44%异常值，实际为绝大多数为0)
- **EnclosedPorch** (14.25%异常值，实际为绝大多数为0)

### 步骤3: 缺失值智能填充

#### 3.1 基于业务逻辑的填充
- **FireplaceQu** (47.26%缺失): 缺失表示无壁炉，填充为"None"
- **LotFrontage** (17.74%缺失): 按社区(Neighborhood)分组，用中位数填充
- **Garage相关列** (5.55%缺失): GarageType/YrBlt/Finish/Qual/Cond，缺失表示无车库
  - 分类列填充"None"
  - GarageYrBlt填充为0或房屋建造年份
- **地下室相关列** (~2.5%缺失): BsmtQual/Cond/Exposure/FinType1/2，缺失表示无地下室，填充"None"

#### 3.2 简单填充
- **MasVnrArea** (0.55%缺失): 填充0（无砌体饰面）
- **Electrical** (0.07%缺失): 填充众数（最常见电气系统）

### 步骤4: 异常值处理 - Winsorize
对以下列进行1%-99%分位数缩尾处理（保留边界值而非删除）：

| 列名 | 处理理由 |
|-----|---------|
| MSSubClass | 建筑类型编码，极端值影响模型 |
| LotFrontage | 街道英尺数，过大/过小为异常 |
| LotArea | 地块面积，极端大值为异常 |
| OverallCond | 整体状况评分 |
| MasVnrArea | 砌体面积 |
| BsmtUnfSF | 未完工地下室面积 |
| TotalBsmtSF | 总地下室面积 |
| 1stFlrSF | 一层面积 |
| LowQualFinSF | 低质量完工面积 |
| GrLivArea | 地上生活面积（关键预测因子） |
| BsmtHalfBath | 地下室半卫数量 |
| BedroomAbvGr | 地上卧室数 |
| KitchenAbvGr | 地上厨房数 |
| TotRmsAbvGrd | 地上总房间数 |
| GarageArea | 车库面积 |
| WoodDeckSF | 木甲板面积 |
| OpenPorchSF | 开放式门廊面积 |
| 3SsnPorch | 三季门廊面积 |
| ScreenPorch | 纱门门廊面积 |
| MiscVal | 其他价值 |
| **SalePrice** | **目标变量，必须处理** |

### 步骤5: 数据类型优化
将43个object类型列转换为category类型，减少内存占用并优化模型性能。

---

## 3. Python清洗代码

```python
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def clean_housing_data(file_path):
    """
    房价预测数据清洗函数
    参数: file_path - 原始数据文件路径
    返回: 清洗后的DataFrame
    """
    
    # 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # ==========================================
    # 步骤1: 删除高缺失率列(>50%)
    # ==========================================
    high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df = df.drop(columns=high_missing_cols)
    print(f"删除高缺失列后: {df.shape}")
    
    # ==========================================
    # 步骤2: 删除近零方差列
    # ==========================================
    near_zero_variance_cols = ['BsmtFinSF2', 'EnclosedPorch']
    df = df.drop(columns=near_zero_variance_cols)
    print(f"删除近零方差列后: {df.shape}")
    
    # ==========================================
    # 步骤3: 智能填充缺失值
    # ==========================================
    
    # 3.1 分类变量 - 填充"None"表示该设施不存在
    none_fill_cols = {
        'FireplaceQu': 'None',  # 无壁炉
        'BsmtQual': 'None',     # 无地下室
        'BsmtCond': 'None',
        'BsmtExposure': 'None',
        'BsmtFinType1': 'None',
        'BsmtFinType2': 'None',
        'GarageType': 'None',   # 无车库
        'GarageFinish': 'None',
        'GarageQual': 'None',
        'GarageCond': 'None'
    }
    
    for col, fill_value in none_fill_cols.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    
    # 3.2 GarageYrBlt - 无车库填充0，或用车库建造年份的中位数
    if 'GarageYrBlt' in df.columns:
        # 用房屋建造年份填充（无车库的年份设为建造年份）
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    
    # 3.3 LotFrontage - 按Neighborhood分组用中位数填充
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果还有缺失（新社区），用整体中位数
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # 3.4 数值型简单填充
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    print(f"缺失值填充完成，剩余缺失值: {df.isnull().sum().sum()}")
    
    # ==========================================
    # 步骤4: 异常值处理 - Winsorize (1%-99%)
    # ==========================================
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
        'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
    ]
    
    for col in winsorize_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            # 计算1%和99%分位数
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            # 缩尾处理
            df[col] = df[col].clip(lower=lower, upper=upper)
    
    print("异常值Winsorize处理完成")
    
    # ==========================================
    # 步骤5: 数据类型转换
    # ==========================================
    # 识别所有object列并转换为category
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 排除已经删除的列
    object_cols = [col for col in object_cols if col in df.columns]
    
    for col in object_cols:
        df[col] = df[col].astype('category')
    
    print(f"转换{len(object_cols)}列为category类型")
    
    # ==========================================
    # 验证
    # ==========================================
    print(f"\n清洗后数据形状: {df.shape}")
    print(f"总缺失值数量: {df.isnull().sum().sum()}")
    print(f"重复行数: {df.duplicated().sum()}")
    
    return df

# 使用示例
if __name__ == "__main__":
    # 清洗训练数据
    train_cleaned = clean_housing_data('/Users/cjialin/code/AutoMLByLLM/train.csv')
    
    # 保存清洗后数据
    train_cleaned.to_csv('/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv', index=False)
    print("清洗完成，数据已保存")
```

---

## 4. 特征工程建议（可选）

为提升房价预测效果，建议在清洗后进行以下特征构造：

```python
# 总面积特征（重要预测因子）
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# 房屋年龄和翻新年龄
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# 浴室总数
df['TotalBath'] = (df['FullBath'] + df['BsmtFullBath'] * 0.5 + 
                   df['HalfBath'] * 0.5 + df['BsmtHalfBath'] * 0.25)

# 车库质量得分（分类变量编码）
garage_quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['GarageScore'] = df['GarageQual'].map(garage_quality_map).fillna(0)
```

---

## 5. 验证清单

清洗完成后，请验证以下项目：

- [ ] 数据形状为 (1460, 74) 左右（删除了7列）
- [ ] 无缺失值（或仅有个别合理缺失）
- [ ] SalePrice分布更集中（无极端异常值）
- [ ] 所有分类变量已转为category类型
- [ ] GrLivArea、TotalBsmtSF等关键特征无异常大值
- [ ] Id列完整保留（用于预测提交）

---

## 6. 注意事项

1. **目标变量保护**: SalePrice的异常值采用Winsorize而非删除，保留所有样本
2. **测试集一致性**: 测试集需应用相同的清洗逻辑（使用训练集的统计量填充）
3. **内存优化**: Category类型可减少内存占用约50-70%
4. **RMSE优化**: 对数变换SalePrice可进一步优化RMSE（建议在模型阶段进行）