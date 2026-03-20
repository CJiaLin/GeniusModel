import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=== 开始数据清洗流程 ===")

# 步骤 1: 加载数据
file_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
df = pd.read_csv(file_path)
print(f"成功加载数据，原始形状: {df.shape}")
print(f"原始列名: {df.columns.tolist()}")

# 创建清洗副本
df_clean = df.copy()

# 步骤 2: 缺失值处理

# 2.1 Cabin 列处理（高缺失率77.1%）
# 策略：提取甲板信息，缺失标记为'Unknown'，创建是否有舱位标记，然后删除原列
print("处理 Cabin 列...")
df_clean['Deck'] = df_clean['Cabin'].str[0]  # 提取首字母作为甲板信息
df_clean['Deck'] = df_clean['Deck'].fillna('Unknown')
df_clean['Has_Cabin'] = df_clean['Cabin'].notna().astype(int)
df_clean = df_clean.drop('Cabin', axis=1)
print("  - 已提取 Deck 信息并删除 Cabin 列")

# 2.2 Age 列处理（中等缺失率19.9%）
# 策略：使用 Pclass 和 Sex 分组的中位数填充
print("处理 Age 列...")
age_fill = df_clean.groupby(['Pclass', 'Sex'])['Age'].median()

def fill_age(row):
    if pd.isna(row['Age']):
        # 尝试获取对应分组的中位数，如果不存在则使用全局中位数
        group_median = age_fill.get((row['Pclass'], row['Sex']))
        if pd.isna(group_median):
            return df_clean['Age'].median()
        return group_median
    return row['Age']

df_clean['Age'] = df_clean.apply(fill_age, axis=1)
df_clean['Age'] = df_clean['Age'].astype(float)
print(f"  - 已填充 Age 缺失值，当前缺失: {df_clean['Age'].isnull().sum()}")

# 2.3 Embarked 列处理（低缺失率0.2%）
# 策略：众数填充
print("处理 Embarked 列...")
mode_value = df_clean['Embarked'].mode()[0]
df_clean['Embarked'] = df_clean['Embarked'].fillna(mode_value)
print(f"  - 已使用众数 '{mode_value}' 填充 Embarked")

# 步骤 3: 异常值处理

# 3.1 Fare 异常值处理（存在极端值512.33）
print("处理 Fare 异常值...")
Q1 = df_clean['Fare'].quantile(0.25)
Q3 = df_clean['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 标记异常值
df_clean['Fare_Outlier'] = ((df_clean['Fare'] < lower_bound) | (df_clean['Fare'] > upper_bound)).astype(int)

# 对数转换（处理右偏分布，使用log1p处理0值）
df_clean['Fare_Log'] = np.log1p(df_clean['Fare'])

# 对极端值进行封顶处理（保留原始Fare供参考）
df_clean['Fare_Capped'] = df_clean['Fare'].clip(upper=upper_bound)
print(f"  - 检测到 {df_clean['Fare_Outlier'].sum()} 个异常值，上限设置为 {upper_bound:.2f}")

# 步骤 4: 特征工程

# 4.1 创建家庭规模特征
print("创建家庭相关特征...")
df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1

# 4.2 创建是否独自旅行特征
df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)

# 4.3 从 Name 提取称谓
print("提取称谓特征...")
df_clean['Title'] = df_clean['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# 合并稀有称谓
title_mapping = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Dr': 'Rare',
    'Rev': 'Rare',
    'Col': 'Rare',
    'Major': 'Rare',
    'Mlle': 'Miss',
    'Countess': 'Rare',
    'Ms': 'Miss',
    'Lady': 'Rare',
    'Jonkheer': 'Rare',
    'Don': 'Rare',
    'Dona': 'Rare',
    'Mme': 'Mrs',
    'Capt': 'Rare',
    'Sir': 'Rare'
}
df_clean['Title'] = df_clean['Title'].map(title_mapping).fillna('Rare')

# 4.4 年龄分箱
print("创建年龄分箱...")
df_clean['AgeBin'] = pd.cut(df_clean['Age'], 
                            bins=[0, 12, 18, 35, 60, 100], 
                            labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

# 4.5 票价分箱
print("创建票价分箱...")
df_clean['FareBin'] = pd.qcut(df_clean['Fare_Capped'], q=4, labels=['Low', 'Mid', 'High', 'Premium'])

# 步骤 5: 数据类型转换
print("转换数据类型...")
categorical_cols = ['Survived', 'Pclass', 'Sex', 'Embarked', 'Deck', 'Title', 'AgeBin', 'FareBin']
for col in categorical_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype('category')

# 5.2 删除冗余列（保留原始Fare供参考，删除Name和Ticket）
columns_to_drop = ['Name', 'Ticket']
df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')

# 步骤 6: 验证与保存
print("\n=== 清洗后数据质量报告 ===")
print(f"数据形状: {df_clean.shape}")
print(f"\n缺失值统计:")
missing_stats = df_clean.isnull().sum()
print(missing_stats[missing_stats > 0] if len(missing_stats[missing_stats > 0]) > 0 else "无缺失值")

print(f"\n数据类型:")
print(df_clean.dtypes)

print(f"\n特征列表:")
print(df_clean.columns.tolist())

# 6.2 保存清洗后的数据
output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
df_clean.to_csv(output_path, index=False)
print(f"\n清洗后数据已保存至: {output_path}")

# 返回统计信息
print("\n=== 清洗结果统计 ===")
print(f"原始数据形状: {df.shape}")
print(f"清洗后数据形状: {df_clean.shape}")
print(f"新增特征: {len(df_clean.columns) - len(df.columns)}")
print(f"总缺失值: {df_clean.isnull().sum().sum()}")
print("数据处理完成！")