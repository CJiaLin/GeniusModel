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
