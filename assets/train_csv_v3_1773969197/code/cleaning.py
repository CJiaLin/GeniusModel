import pandas as pd
import numpy as np
from pathlib import Path

def clean_data():
    """
    数据清洗主函数
    """
    # 设置数据路径
    input_path = "/Users/cjialin/code/AutoMLByLLM/train.csv"
    output_path = "/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv"
    
    print("=" * 60)
    print("开始数据清洗流程")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n步骤 1: 加载数据...")
    df = pd.read_csv(input_path)
    print(f"原始数据形状: {df.shape}")
    print(f"原始数据列数: {df.shape[1]}")
    
    # 2. 查看原始缺失值情况
    print("\n步骤 2: 分析缺失值情况...")
    missing_cols = df.columns[df.isnull().any()].tolist()
    print(f"包含缺失值的列: {missing_cols}")
    missing_stats = df[missing_cols].isnull().sum()
    print("缺失值统计:")
    for col, count in missing_stats.items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    
    # 3. 处理数值型缺失值
    print("\n步骤 3: 处理数值型缺失值...")
    
    # LotFrontage: 用中位数填充（街道到房产的直线距离）
    if 'LotFrontage' in df.columns:
        lot_frontage_median = df['LotFrontage'].median()
        df['LotFrontage'].fillna(lot_frontage_median, inplace=True)
        print(f"  LotFrontage: 使用中位数 {lot_frontage_median} 填充")
    
    # MasVnrArea: 用0填充（砌体贴面面积，缺失表示无砌体贴面）
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'].fillna(0, inplace=True)
        print(f"  MasVnrArea: 使用0填充")
    
    # 4. 处理分类型缺失值
    print("\n步骤 4: 处理分类型缺失值...")
    
    # Alley: 用'None'填充（巷子通道，NA表示没有巷子）
    if 'Alley' in df.columns:
        df['Alley'].fillna('None', inplace=True)
        print(f"  Alley: 使用'None'填充")
    
    # 地下室相关字段：用'None'填充表示没有地下室
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        if col in df.columns:
            df[col].fillna('None', inplace=True)
            print(f"  {col}: 使用'None'填充")
    
    # MasVnrType: 用众数填充
    if 'MasVnrType' in df.columns:
        mas_vnr_mode = df['MasVnrType'].mode()[0]
        df['MasVnrType'].fillna(mas_vnr_mode, inplace=True)
        print(f"  MasVnrType: 使用众数 '{mas_vnr_mode}' 填充")
    
    # Electrical: 用众数填充
    if 'Electrical' in df.columns:
        electrical_mode = df['Electrical'].mode()[0]
        df['Electrical'].fillna(electrical_mode, inplace=True)
        print(f"  Electrical: 使用众数 '{electrical_mode}' 填充")
    
    # 5. 数据类型优化
    print("\n步骤 5: 优化数据类型...")
    
    # 定义分类列（基于提供的信息）
    categorical_columns = [
        'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
        'RoofStyle', 'RoofMatl', 'MasVnrType', 'BsmtQual', 
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Electrical'
    ]
    
    # 将分类列转换为category类型
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    print(f"  已将 {len([c for c in categorical_columns if c in df.columns])} 列转换为category类型")
    
    # 6. 删除不必要的列
    print("\n步骤 6: 删除标识符列...")
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        print("  已删除 'Id' 列")
    
    # 7. 检查重复值
    print("\n步骤 7: 检查重复值...")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"  发现 {duplicate_count} 行重复数据，已删除")
    else:
        print("  未发现重复数据")
    
    # 8. 验证缺失值是否全部处理
    print("\n步骤 8: 验证缺失值处理...")
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing == 0:
        print("  所有缺失值已处理完毕")
    else:
        print(f"  警告：仍有 {remaining_missing} 个缺失值")
        remaining_cols = df.columns[df.isnull().any()].tolist()
        print(f"  剩余缺失值列: {remaining_cols}")
    
    # 9. 保存清洗后的数据
    print("\n步骤 9: 保存清洗后的数据...")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  数据已保存到: {output_path}")
    
    # 10. 生成统计报告
    print("\n" + "=" * 60)
    print("数据清洗完成 - 统计报告")
    print("=" * 60)
    
    # 内存使用统计
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    report = {
        "原始数据行数": 1460,
        "原始数据列数": 81,
        "清洗后数据行数": len(df),
        "清洗后数据列数": len(df.columns),
        "删除重复行数": duplicate_count,
        "删除列数": 1 if 'Id' not in df.columns else 0,
        "处理后缺失值总数": remaining_missing,
        "内存使用": f"{original_memory:.2f} MB"
    }
    
    print("\n清洗结果统计:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # 各列数据类型统计
    print(f"\n数据类型分布:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} 列")
    
    print("\n数值列样例（前5列）:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols[:5]:
        print(f"  {col}: 均值={df[col].mean():.2f}, 标准差={df[col].std():.2f}")
    
    print("\n分类列样例（前5列）:")
    cat_cols = df.select_dtypes(include=['category']).columns.tolist()
    for col in cat_cols[:5]:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count} 个唯一值")
    
    return df, report

# 执行清洗
if __name__ == "__main__":
    cleaned_df, statistics = clean_data()
    print("\n清洗流程全部完成！")