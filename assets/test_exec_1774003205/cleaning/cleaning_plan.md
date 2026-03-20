# 数据清洗方案

## 数据概览

| 项目 | 详情 |
|------|------|
| **数据路径** | `/Users/cjialin/code/AutoMLByLLM/train.csv` |
| **文件格式** | CSV |
| **分析时间** | 2024年 |
| **目标** | 生成高质量训练数据 |

---

## 一、数据质量问题分析

### 1.1 缺失值检测
```python
import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')

# 缺失值统计
missing_stats = pd.DataFrame({
    '列名': df.columns,
    '缺失数量': df.isnull().sum(),
    '缺失比例(%)': (df.isnull().sum() / len(df) * 100).round(2),
    '数据类型': df.dtypes
})
print(missing_stats[missing_stats['缺失数量'] > 0])
```

**常见问题类型：**
- 🔴 **高风险**：缺失比例 > 50% 的列
- 🟡 **中风险**：缺失比例 10%-50% 的列  
- 🟢 **低风险**：缺失比例 < 10% 的列

### 1.2 异常值检测
```python
# 数值型列的统计描述
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 使用 IQR 方法检测异常值
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: 异常值数量 = {len(outliers)}")
```

### 1.3 重复值检测
```python
# 完全重复的行
duplicate_rows = df.duplicated().sum()
print(f"完全重复行数: {duplicate_rows}")

# 基于关键列的重复（如ID列）
if 'id' in df.columns:
    id_duplicates = df['id'].duplicated().sum()
    print(f"ID重复数量: {id_duplicates}")
```

### 1.4 数据类型问题
| 问题类型 | 检测方法 | 示例 |
|---------|---------|------|
| 类型不匹配 | `df.dtypes` | 数值列被识别为 object |
| 格式不一致 | 正则表达式 | 日期格式混乱 |
| 编码问题 | 查看乱码字符 | UTF-8/GBK 编码错误 |

### 1.5 一致性检查
- **分类变量**：检查是否有拼写不一致（如 "Male"/"male"/"M"）
- **数值范围**：检查是否有超出合理范围的值
- **逻辑一致性**：如"年龄"与"出生日期"是否匹配

---

## 二、清洗步骤

### 步骤 1：初始设置与数据加载
```python
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# 加载原始数据
df_raw = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')
df = df_raw.copy()

print(f"原始数据形状: {df.shape}")
print(f"\n列名: {list(df.columns)}")
```

### 步骤 2：缺失值处理

#### 策略 A：删除高缺失率列
```python
# 删除缺失率超过 50% 的列
threshold = 0.5
cols_to_drop = df.columns[df.isnull().mean() > threshold]
df = df.drop(columns=cols_to_drop)
print(f"删除的列: {list(cols_to_drop)}")
```

#### 策略 B：数值型缺失值填充
```python
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        # 根据分布选择填充策略
        skewness = df[col].skew()
        
        if abs(skewness) < 1:  # 近似正态分布
            fill_value = df[col].mean()
            method = "均值"
        else:  # 偏态分布
            fill_value = df[col].median()
            method = "中位数"
        
        df[col].fillna(fill_value, inplace=True)
        print(f"{col}: 使用{method}填充 ({fill_value:.2f})")
```

#### 策略 C：分类型缺失值填充
```python
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        # 使用众数填充
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        
        # 或者创建"Unknown"类别
        # df[col].fillna('Unknown', inplace=True)
        
        print(f"{col}: 使用众数 '{mode_value}' 填充")
```

### 步骤 3：异常值处理

#### 策略 A：盖帽法（Winsorization）
```python
def winsorize_series(series, limits=(0.05, 0.05)):
    """将极端值限制在指定百分位数"""
    lower = series.quantile(limits[0])
    upper = series.quantile(1 - limits[1])
    return series.clip(lower, upper)

# 对关键数值列应用盖帽法
for col in numeric_cols:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = winsorize_series(df[col])
        print(f"{col}: 应用 5%-95% 盖帽法")
```

#### 策略 B：IQR 方法删除异常值
```python
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    return df[mask]

# 谨慎使用：仅在确定异常值为错误数据时使用
# df = remove_outliers_iqr(df, 'target_column')
```

#### 策略 C：Z-Score 方法
```python
# 对近似正态分布的列使用 Z-Score
for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    outlier_mask = z_scores > 3
    if outlier_mask.sum() > 0:
        print(f"{col}: 检测到 {outlier_mask.sum()} 个 Z-Score > 3 的异常值")
```

### 步骤 4：重复值处理
```python
# 记录清洗前数量
before_count = len(df)

# 删除完全重复的行
df = df.drop_duplicates()

# 基于关键列删除重复（保留第一条）
if 'id' in df.columns:
    df = df.drop_duplicates(subset=['id'], keep='first')

after_count = len(df)
print(f"删除重复行: {before_count - after_count} 行")
print(f"剩余数据: {after_count} 行")
```

### 步骤 5：数据类型转换
```python
# 自动转换数据类型
df = df.convert_dtypes()

# 手动转换示例
# 日期列转换
date_cols = ['date', 'created_at', 'timestamp']  # 根据实际列名调整
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# 类别型转换（降低内存）
for col in categorical_cols:
    if df[col].nunique() / len(df) < 0.5:  # 基数较小的列
        df[col] = df[col].astype('category')

print("数据类型转换完成")
print(df.dtypes)
```

### 步骤 6：特征工程（清洗阶段）
```python
# 统一文本格式
text_cols = df.select_dtypes(include=['object']).columns
for col in text_cols:
    df[col] = df[col].str.strip()  # 去除首尾空格
    df[col] = df[col].str.lower()  # 统一小写（根据需要）

# 标准化分类值（示例）
if 'gender' in df.columns:
    gender_mapping = {
        'male': 'Male', 'm': 'Male', 'M': 'Male',
        'female': 'Female', 'f': 'Female', 'F': 'Female'
    }
    df['gender'] = df['gender'].map(gender_mapping).fillna(df['gender'])
```

### 步骤 7：验证与保存
```python
# 最终验证
print("=== 清洗后数据质量报告 ===")
print(f"数据形状: {df.shape}")
print(f"\n缺失值统计:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\n数据类型:\n{df.dtypes}")

# 保存清洗后的数据
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n清洗后数据已保存至: {output_path}")

# 生成清洗报告
cleaning_report = {
    '原始数据行数': len(df_raw),
    '清洗后数据行数': len(df),
    '删除行数': len(df_raw) - len(df),
    '删除列数': len(df_raw.columns) - len(df.columns),
    '处理缺失值列数': len([c for c in df.columns if df_raw[c].isnull().sum() > 0]),
}
print(f"\n清洗报告: {cleaning_report}")
```

---

## 三、预期效果

### 3.1 数据质量指标

| 指标 | 清洗前 | 清洗后 | 目标值 |
|------|--------|--------|--------|
| 完整率 (Completeness) | ?% | >95% | >95% |
| 唯一性 (Uniqueness) | ?% | 100% | 100% |
| 有效性 (Validity) | ?% | >98% | >98% |
| 一致性 (Consistency) | ?% | >99% | >99% |
| 准确性 (Accuracy) | ?% | >95% | >95% |

### 3.2 模型训练收益

| 收益类型 | 具体效果 |
|---------|---------|
| **减少过拟合** | 去除异常值和噪声，提高泛化能力 |
| **提升收敛速度** | 数据标准化后，梯度下降更稳定 |
| **改善模型性能** | 完整且一致的数据提升预测精度 |
| **降低内存占用** | 类型优化（category）减少 50%+ 内存 |
| **减少训练时间** | 删除重复行和无关列，加速训练 |

### 3.3 风险与注意事项

| 风险点 | 缓解措施 |
|--------|---------|
| 过度清洗导致信息损失 | 保留原始数据备份，记录所有清洗操作 |
| 填充值引入偏差 | 使用多种填充策略对比验证 |
| 异常值可能是真实信号 | 业务专家审核后再删除 |
| 数据泄露风险 | 确保训练集和测试集分开清洗 |

---

## 四、执行脚本（完整版）

```python
#!/usr/bin/env python3
"""
数据清洗完整脚本
用法: python data_cleaning.py --input train.csv --output train_cleaned.csv
"""

import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataCleaner:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.report = {}
        
    def load(self):
        """加载数据"""
        logging.info(f"加载数据: {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        self.report['original_shape'] = self.df.shape
        return self
    
    def analyze(self):
        """分析数据质量"""
        logging.info("分析数据质量...")
        self.report['missing'] = self.df.isnull().sum().to_dict()
        self.report['dtypes'] = self.df.dtypes.to_dict()
        self.report['duplicates'] = self.df.duplicated().sum()
        return self
    
    def clean(self):
        """执行清洗流程"""
        logging.info("开始数据清洗...")
        
        # 1. 删除高缺失率列
        missing_ratio = self.df.isnull().mean()
        cols_to_keep = missing_ratio[missing_ratio < 0.5].index
        self.df = self.df[cols_to_keep]
        
        # 2. 填充缺失值
        for col in self.df.select_dtypes(include=[np.number]).columns:
            self.df[col].fillna(self.df[col].median(), inplace=True)
        
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown', inplace=True)
        
        # 3. 删除重复行
        self.df = self.df.drop_duplicates()
        
        # 4. 处理异常值（盖帽法）
        for col in self.df.select_dtypes(include=[np.number]).columns:
            lower = self.df[col].quantile(0.01)
            upper = self.df[col].quantile(0.99)
            self.df[col] = self.df[col].clip(lower, upper)
        
        # 5. 类型优化
        self.df = self.df.convert_dtypes()
        
        self.report['cleaned_shape'] = self.df.shape
        return self
    
    def save(self):
        """保存结果"""
        self.df.to_csv(self.output_path, index=False)
        logging.info(f"清洗完成，保存至: {self.output_path}")
        logging.info(f"清洗报告: {self.report}")
        return self

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据清洗工具')
    parser.add_argument('--input', default='/Users/cjialin/code/AutoMLByLLM/train.csv')
    parser.add_argument('--output', default='/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv')
    args = parser.parse_args()
    
    cleaner = DataCleaner(args.input, args.output)
    cleaner.load().analyze().clean().save()
```

---

## 五、下一步建议

1. **数据探索 (EDA)**：清洗后进行深入的数据分析，发现特征间关系
2. **特征工程**：基于清洗后的数据创建新特征
3. **数据验证**：使用统计测试验证清洗效果
4. **自动化监控**：建立数据质量监控 pipeline，持续跟踪数据质量

---

> **备注**：以上方案为通用模板，请根据 `analyze_data` 和 `load_data` 的实际返回结果调整具体参数和策略。建议在实际执行前，先在小样本数据上测试清洗代码。