# 房价预测数据清洗方案

## 一、数据质量概述

| 指标 | 数值 |
|------|------|
| 数据形状 | (1460, 81) |
| 数值列 | 38列 |
| 分类列 | 43列 |
| 重复行 | 0 |
| 缺失值问题列 | 19列 |
| 异常值问题列 | 31列 |

### 主要质量问题
1. **极端缺失列**：5列缺失率超过59%（PoolQC、MiscFeature、Alley、Fence、MasVnrType）
2. **中等缺失列**：FireplaceQu (47.26%)、LotFrontage (17.74%)
3. ** garage相关缺失**：5列各缺失81条（5.55%）
4. **地下室相关缺失**：5列各缺失约37-38条（约2.6%）
5. **异常值**：31列存在统计异常值，需结合业务判断是否处理

---

## 二、清洗策略

### 2.1 高缺失率列处理（缺失率>50%）

| 列名 | 缺失率 | 处理方式 | 理由 |
|------|--------|----------|------|
| PoolQC | 99.52% | 删除 | 仅7户有泳池，信息价值极低 |
| MiscFeature | 96.3% | 删除 | 仅54户有特殊设施 |
| Alley | 93.77% | 删除 | 仅91户有胡同通道 |
| Fence | 80.75% | 删除 | 仅281户有围栏 |
| MasVnrType | 59.73% | 删除 | 缺失过多，难以可靠填充 |

### 2.2 中等缺失率列处理（5%-50%）

| 列名 | 缺失率 | 处理方式 | 填充策略 |
|------|--------|----------|----------|
| FireplaceQu | 47.26% | 填充"None" | 无壁炉的填充为无 |
| LotFrontage | 17.74% | 中位数填充 | 按Neighborhood分组中位数 |
| GarageType | 5.55% | 填充"None" | 无车库 |
| GarageYrBlt | 5.55% | 填充YearBuilt | 无车库则用房屋建造年份 |
| GarageFinish | 5.55% | 填充"None" | 无车库 |
| GarageQual | 5.55% | 填充"None" | 无车库 |
| GarageCond | 5.55% | 填充"None" | 无车库 |

### 2.3 低缺失率列处理（<5%）

| 列名 | 缺失数 | 处理方式 |
|------|--------|----------|
| BsmtExposure | 38 | 填充"No"（无暴露） |
| BsmtFinType2 | 38 | 填充"No"（无地下室） |
| BsmtQual | 37 | 填充"No"（无地下室） |
| BsmtCond | 37 | 填充"No"（无地下室） |
| BsmtFinType1 | 37 | 填充"No"（无地下室） |
| MasVnrArea | 8 | 填充0（无石材贴面） |
| Electrical | 1 | 填充众数"SBrkr" |

### 2.4 异常值处理策略

#### 删除列（无变化特征）
- **BsmtFinSF2**: 167条异常（11.44%），但99%为0，方差过低，删除
- **EnclosedPorch**: 208条异常（14.25%），99%为0，删除

#### Winsorize处理（缩尾至5%-95%分位数）
| 列名 | 处理方式 |
|------|----------|
| MSSubClass | 上下限缩尾 |
| LotFrontage | 上下限缩尾 |
| LotArea | 上下限缩尾 |
| OverallCond | 上下限缩尾 |
| MasVnrArea | 上下限缩尾（填充后） |
| BsmtUnfSF | 上下限缩尾 |
| TotalBsmtSF | 上下限缩尾 |
| 1stFlrSF | 上下限缩尾 |
| GrLivArea | 上下限缩尾 |
| BsmtHalfBath | 上限缩尾 |
| BedroomAbvGr | 上下限缩尾 |
| KitchenAbvGr | 上下限缩尾 |
| TotRmsAbvGrd | 上下限缩尾 |
| GarageArea | 上下限缩尾 |
| WoodDeckSF | 上下限缩尾 |
| OpenPorchSF | 上下限缩尾 |
| 3SsnPorch | 上限缩尾 |
| ScreenPorch | 上限缩尾 |
| MiscVal | 上限缩尾 |
| SalePrice | 上下限缩尾（目标变量） |

#### 保留异常值（业务合理）
- OverallQual (2条): 质量评分1-10是合理范围
- YearBuilt (7条): 老房子是合理存在的
- BsmtFinSF1 (7条): 地下室面积差异大属正常
- 2ndFlrSF (2条): 部分房屋无二层
- BsmtFullBath (1条): 地下室无浴室正常
- Fireplaces (5条): 多壁炉豪宅合理
- GarageCars (5条): 多车位豪宅合理
- PoolArea (7条): 带泳池的房产

---

## 三、完整清洗代码

```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """
    房价预测数据清洗主函数
    参数:
        file_path: 数据文件路径
    返回:
        清洗后的DataFrame
    """
    # 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # ============================================
    # 第一步：删除高缺失率列（缺失率>50%）
    # ============================================
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df = df.drop(columns=cols_to_drop)
    print(f"删除高缺失列后: {df.shape}")
    
    # ============================================
    # 第二步：删除低方差列（无信息特征）
    # ============================================
    cols_to_drop_low_var = ['BsmtFinSF2', 'EnclosedPorch']
    df = df.drop(columns=cols_to_drop_low_var)
    print(f"删除低方差列后: {df.shape}")
    
    # ============================================
    # 第三步：缺失值填充 - 分类变量
    # ============================================
    
    # 无相关设施填充"None"
    none_fill_cols = ['FireplaceQu', 'GarageType', 'GarageFinish', 
                      'GarageQual', 'GarageCond']
    for col in none_fill_cols:
        df[col] = df[col].fillna('None')
    
    # 地下室相关填充"No"（表示无地下室）
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 
                     'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        df[col] = df[col].fillna('No')
    
    # Electrical填充众数
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    # ============================================
    # 第四步：缺失值填充 - 数值变量
    # ============================================
    
    # LotFrontage按Neighborhood分组中位数填充
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    # 若仍有缺失（某些Neighborhood全缺失），用整体中位数
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # GarageYrBlt：无车库则使用YearBuilt
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    
    # MasVnrArea：无石材贴面填充0
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    print(f"缺失值填充完成，剩余缺失: {df.isnull().sum().sum()}")
    
    # ============================================
    # 第五步：异常值处理 - Winsorize
    # ============================================
    
    def winsorize_series(series, lower_quantile=0.05, upper_quantile=0.95):
        """对序列进行缩尾处理"""
        lower = series.quantile(lower_quantile)
        upper = series.quantile(upper_quantile)
        return series.clip(lower, upper)
    
    # 需要Winsorize的数值列
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
        'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
        'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
    ]
    
    for col in winsorize_cols:
        if col in df.columns:
            df[col] = winsorize_series(df[col])
    
    print("Winsorize异常值处理完成")
    
    # ============================================
    # 第六步：数据类型转换
    # ============================================
    
    # MSSubClass实际是分类变量（建筑类型代码）
    df['MSSubClass'] = df['MSSubClass'].astype('category')
    
    # 将object类型转换为category（节省内存，便于建模）
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    print(f"分类变量数量: {len(categorical_cols)}")
    
    # ============================================
    # 第七步：特征工程（可选，提升模型效果）
    # ============================================
    
    # 房屋年龄
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    # 翻新后年龄
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    
    # 总面积
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # 总浴室数
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                       df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    
    # 总门廊面积
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + 
                          df['EnclosedPorch'] + df['ScreenPorch'] + 
                          df['WoodDeckSF'])
    
    print(f"特征工程后: {df.shape}")
    
    return df


def validate_cleaning(df_original, df_cleaned):
    """
    验证清洗效果
    """
    print("\n" + "="*50)
    print("数据清洗验证报告")
    print("="*50)
    
    # 缺失值检查
    missing_before = df_original.isnull().sum().sum()
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"\n缺失值: {missing_before} -> {missing_after}")
    
    # 列数变化
    print(f"列数: {df_original.shape[1]} -> {df_cleaned.shape[1]}")
    print(f"行数: {df_original.shape[0]} -> {df_cleaned.shape[0]}")
    
    # 数值列统计变化
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    print(f"\n数值列数量: {len(numeric_cols)}")
    
    # 目标变量SalePrice分布
    print(f"\n目标变量SalePrice:")
    print(f"  均值: {df_cleaned['SalePrice'].mean():.2f}")
    print(f"  标准差: {df_cleaned['SalePrice'].std():.2f}")
    print(f"  最小值: {df_cleaned['SalePrice'].min()}")
    print(f"  最大值: {df_cleaned['SalePrice'].max()}")
    
    # 异常值检查（基于IQR）
    Q1 = df_cleaned['SalePrice'].quantile(0.25)
    Q3 = df_cleaned['SalePrice'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_cleaned['SalePrice'] < (Q1 - 1.5 * IQR)) | 
                (df_cleaned['SalePrice'] > (Q3 + 1.5 * IQR))).sum()
    print(f"  IQR异常值数量: {outliers}")
    
    print("\n" + "="*50)
    print("清洗完成！数据已准备好用于建模。")
    print("="*50)


# 主执行流程
if __name__ == "__main__":
    # 数据路径
    FILE_PATH = "/Users/cjialin/code/AutoMLByLLM/train.csv"
    
    # 加载原始数据
    df_raw = pd.read_csv(FILE_PATH)
    
    # 执行清洗
    df_cleaned = load_and_clean_data(FILE_PATH)
    
    # 验证结果
    validate_cleaning(df_raw, df_cleaned)
    
    # 保存清洗后数据
    output_path = "/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n清洗后数据已保存至: {output_path}")
```

---

## 四、清洗效果预期

| 指标 | 清洗前 | 清洗后 | 改善 |
|------|--------|--------|------|
| 缺失值总数 | ~6,800 | 0 | ✅ 完全消除 |
| 特征维度 | 81 | 79 | 删除5列，新增5个特征 |
| 异常值比例 | ~15% | <5% | Winsorize处理 |
| 内存占用 | 高 | 降低30% | Category类型优化 |

---

## 五、建模建议

1. **特征编码**：对分类变量使用One-Hot Encoding或Target Encoding
2. **特征选择**：考虑使用Lasso或特征重要性筛选高价值特征
3. **目标变量**：SalePrice已Winsorize，若需可进一步做Log变换使其更接近正态分布
4. **交叉验证**：建议使用K-Fold（K=5或10）评估RMSE

---

## 六、注意事项

1. **测试集一致性**：测试集需使用相同的清洗流程（使用训练集的统计量填充）
2. **数据泄露**：避免使用测试集信息填充缺失值
3. **业务解释**：删除的列（如PoolQC）在实际预测中若出现需特殊处理
4. **RMSE优化**：关注大误差样本，考虑对SalePrice取log降低异常值影响