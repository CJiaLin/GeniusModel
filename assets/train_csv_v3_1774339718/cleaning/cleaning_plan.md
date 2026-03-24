# 🧹 数据清洗方案

## 📋 数据概况

| 项目 | 详情 |
|------|------|
| **数据文件** | `/Users/cjialin/code/AutoMLByLLM/train.csv` |
| **数据形状** | 1,460 行 × 81 列 |
| **数值列** | 38 列 |
| **分类列** | 43 列 |
| **数据质量问题** | 50 类 |

---

## 🎯 清洗策略总览

### 1️⃣ 高缺失率列删除（缺失率 > 50%）
- **PoolQC** (99.52%) - 游泳池质量，绝大多数房屋没有游泳池
- **MiscFeature** (96.30%) - 其他杂项特征
- **Alley** (93.77%) - 巷子类型
- **Fence** (80.75%) - 围栏质量
- **MasVnrType** (59.73%) - 砖石饰面类型

### 2️⃣ 中等缺失率列填充（缺失率 5%-50%）
- **FireplaceQu** (47.26%) - 壁炉质量，用 "No Fireplace" 填充
- **LotFrontage** (17.74%) - 临街距离，用中位数或基于 Neighborhood 分组填充
- **Garage 相关列** (5.55%) - 车库信息，用 "No Garage" 或 0 填充

### 3️⃣ 低缺失率列填充（缺失率 < 5%）
- **地下室相关列** - 用众数或 "No Basement" 填充
- **MasVnrArea** - 用 0 填充（表示无砖石饰面）
- **Electrical** - 用众数填充

### 4️⃣ 异常值处理
- **删除列**：BsmtFinSF2 (11.44% 异常值)、EnclosedPorch (14.25% 异常值)
- **Winsorize 处理**：MSSubClass, LotFrontage, LotArea, OverallCond 等数值列
- **保留列**：OverallQual, YearBuilt 等具有业务合理性的异常值

### 5️⃣ 数据类型优化
- 将 39 个分类列转换为 `category` 类型以节省内存

---

## 🐍 Python 清洗代码

```python
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 加载数据
# ==========================================
def load_data(file_path):
    """加载原始数据"""
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    return df

# ==========================================
# 2. 高缺失率列删除
# ==========================================
def drop_high_missing_columns(df):
    """删除缺失率超过50%的列"""
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
    df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"删除高缺失率列后形状: {df_cleaned.shape}")
    return df_cleaned

# ==========================================
# 3. 缺失值填充
# ==========================================
def fill_missing_values(df):
    """智能填充缺失值"""
    df_filled = df.copy()
    
    # 3.1 FireplaceQu - 无壁炉的用 "No Fireplace" 填充
    if 'FireplaceQu' in df_filled.columns:
        df_filled['FireplaceQu'] = df_filled['FireplaceQu'].fillna('No Fireplace')
    
    # 3.2 LotFrontage - 按 Neighborhood 分组填充中位数
    if 'LotFrontage' in df_filled.columns:
        df_filled['LotFrontage'] = df_filled.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果仍有缺失，用整体中位数填充
        df_filled['LotFrontage'] = df_filled['LotFrontage'].fillna(df_filled['LotFrontage'].median())
    
    # 3.3 Garage 相关列 - 无车库的用 "No Garage" 或 0 填充
    garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cols:
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].fillna('No Garage')
    
    if 'GarageYrBlt' in df_filled.columns:
        # 无车库的用 0 或房屋建造年份填充
        df_filled['GarageYrBlt'] = df_filled['GarageYrBlt'].fillna(0)
    
    # 3.4 地下室相关列 - 无地下室的用 "No Basement" 填充
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].fillna('No Basement')
    
    # 3.5 MasVnrArea - 用 0 填充（表示无砖石饰面）
    if 'MasVnrArea' in df_filled.columns:
        df_filled['MasVnrArea'] = df_filled['MasVnrArea'].fillna(0)
    
    # 3.6 Electrical - 用众数填充
    if 'Electrical' in df_filled.columns:
        df_filled['Electrical'] = df_filled['Electrical'].fillna(df_filled['Electrical'].mode()[0])
    
    # 验证无缺失值
    remaining_missing = df_filled.isnull().sum().sum()
    print(f"填充后剩余缺失值数量: {remaining_missing}")
    
    return df_filled

# ==========================================
# 4. 异常值处理
# ==========================================
def winsorize_column(series, lower_percentile=0.05, upper_percentile=0.95):
    """对数值列进行 Winsorize 处理"""
    lower_bound = series.quantile(lower_percentile)
    upper_bound = series.quantile(upper_percentile)
    return series.clip(lower=lower_bound, upper=upper_bound)

def handle_outliers(df):
    """处理异常值"""
    df_cleaned = df.copy()
    
    # 4.1 删除高异常值比例的列
    cols_to_drop = ['BsmtFinSF2', 'EnclosedPorch']
    df_cleaned = df_cleaned.drop(columns=[col for col in cols_to_drop if col in df_cleaned.columns], errors='ignore')
    
    # 4.2 Winsorize 处理数值列
    winsorize_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
        'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice'
    ]
    
    for col in winsorize_cols:
        if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]):
            df_cleaned[col] = winsorize_column(df_cleaned[col])
    
    print(f"异常值处理后形状: {df_cleaned.shape}")
    return df_cleaned

# ==========================================
# 5. 数据类型转换
# ==========================================
def optimize_data_types(df):
    """优化数据类型以节省内存"""
    df_optimized = df.copy()
    
    # 分类列列表（基于数据质量报告）
    categorical_columns = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
        'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'PavedDrive', 'SaleType', 'SaleCondition'
    ]
    
    for col in categorical_columns:
        if col in df_optimized.columns:
            df_optimized[col] = df_optimized[col].astype('category')
    
    # 将 MSSubClass 转换为分类变量（虽然是数字，但代表类别）
    if 'MSSubClass' in df_optimized.columns:
        df_optimized['MSSubClass'] = df_optimized['MSSubClass'].astype('category')
    
    # 内存优化信息
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
    print(f"原始内存占用: {original_memory:.2f} MB")
    print(f"优化后内存占用: {optimized_memory:.2f} MB")
    print(f"内存节省: {(1 - optimized_memory/original_memory)*100:.1f}%")
    
    return df_optimized

# ==========================================
# 6. 主清洗流程
# ==========================================
def clean_data(file_path, output_path=None):
    """完整的数据清洗流程"""
    print("=" * 50)
    print("开始数据清洗流程")
    print("=" * 50)
    
    # 步骤1: 加载数据
    df = load_data(file_path)
    
    # 步骤2: 删除高缺失率列
    df = drop_high_missing_columns(df)
    
    # 步骤3: 填充缺失值
    df = fill_missing_values(df)
    
    # 步骤4: 处理异常值
    df = handle_outliers(df)
    
    # 步骤5: 优化数据类型
    df = optimize_data_types(df)
    
    # 保存清洗后的数据
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n清洗后的数据已保存至: {output_path}")
    
    print("=" * 50)
    print("数据清洗完成！")
    print("=" * 50)
    
    return df

# ==========================================
# 7. 数据验证
# ==========================================
def validate_cleaned_data(df):
    """验证清洗后的数据质量"""
    print("\n" + "=" * 50)
    print("数据验证报告")
    print("=" * 50)
    
    # 检查缺失值
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"总缺失值数量: {total_missing}")
    
    if total_missing > 0:
        print("仍有缺失值的列:")
        print(missing_counts[missing_counts > 0])
    else:
        print("✅ 无缺失值")
    
    # 检查重复行
    duplicates = df.duplicated().sum()
    print(f"重复行数: {duplicates}")
    
    # 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n数值列数量: {len(numeric_cols)}")
    
    # 分类列统计
    categorical_cols = df.select_dtypes(include=['category']).columns
    print(f"分类列数量: {len(categorical_cols)}")
    
    # 数据形状
    print(f"最终数据形状: {df.shape}")
    
    return {
        'missing_values': total_missing,
        'duplicates': duplicates,
        'shape': df.shape
    }

# ==========================================
# 执行清洗
# ==========================================
if __name__ == "__main__":
    INPUT_FILE = "/Users/cjialin/code/AutoMLByLLM/train.csv"
    OUTPUT_FILE = "/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv"
    
    # 执行清洗
    df_cleaned = clean_data(INPUT_FILE, OUTPUT_FILE)
    
    # 验证结果
    validation_results = validate_cleaned_data(df_cleaned)
```

---

## 📊 清洗效果对比

| 检查项 | 清洗前 | 清洗后 | 状态 |
|--------|--------|--------|------|
| 缺失值总数 | 13,960+ | 0 | ✅ 完全清除 |
| 重复行 | 0 | 0 | ✅ 无需处理 |
| 高缺失率列 | 5 列 | 0 列 | ✅ 已删除 |
| 异常值列 | 2 列 | 0 列 | ✅ 已处理 |
| 数据类型优化 | 0 列 | 39 列 | ✅ 已转换 |
| 最终特征数 | 81 | 74 | 优化后 |

---

## ⚠️ 注意事项

1. **业务理解**：LotFrontage 的填充使用了 Neighborhood 分组，假设同街区的房屋临街距离相似
2. **GarageYrBlt**：无车库的房屋使用 0 填充，分析时需注意区分
3. **Winsorize 边界**：使用 5%-95% 分位数，如需更严格可调整为 1%-99%
4. **MSSubClass**：虽然为数字类型，但本质是分类变量，已转换为 category

---

## 🔍 后续建议

1. **特征工程**：可基于现有特征创建新特征（如总居住面积、房屋年龄等）
2. **编码处理**：分类变量需进行 One-Hot 或 Label Encoding
3. **目标变量**：SalePrice 已进行 Winsorize，如需对数转换可进一步处理
4. **验证集**：确保训练集和测试集使用相同的清洗逻辑