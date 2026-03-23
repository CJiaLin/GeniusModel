import pandas as pd
import numpy as np
import os

def clean_housing_data():
    """
    清洗房价数据集
    处理缺失值、重复值，并保存清洗后的数据
    """
    
    # 设置文件路径
    input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    print("开始数据清洗流程...")
    print(f"读取数据: {input_path}")
    
    # 读取原始数据
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"错误: 文件未找到 {input_path}")
        return None
    except Exception as e:
        print(f"读取数据时发生错误: {str(e)}")
        return None
    
    # 记录原始数据信息
    original_shape = df.shape
    original_missing = df.isnull().sum().sum()
    
    print(f"\n原始数据信息:")
    print(f"  - 数据形状: {original_shape[0]} 行 × {original_shape[1]} 列")
    print(f"  - 缺失值总数: {original_missing}")
    
    # ==========================================
    # 1. 处理缺失值
    # ==========================================
    
    # 1.1 LotFrontage - 使用基于Neighborhood的中位数填充
    # 同社区的地块通常具有相似的临街长度特征
    if 'LotFrontage' in df.columns:
        print("\n处理 LotFrontage 缺失值...")
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果仍有缺失（某些Neighborhood可能全是NaN），用整体中位数填充
        overall_median = df['LotFrontage'].median()
        df['LotFrontage'] = df['LotFrontage'].fillna(overall_median)
    
    # 1.2 Alley - NA表示"没有巷子"，用'None'填充
    if 'Alley' in df.columns:
        print("处理 Alley 缺失值...")
        df['Alley'] = df['Alley'].fillna('None')
    
    # 1.3 MasVnrType（砖石贴面类型）和 MasVnrArea（砖石贴面面积）
    if 'MasVnrType' in df.columns:
        print("处理 MasVnrType 缺失值...")
        df['MasVnrType'] = df['MasVnrType'].fillna('None')
    
    if 'MasVnrArea' in df.columns:
        print("处理 MasVnrArea 缺失值...")
        # 如果没有贴面类型，面积应为0
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # 1.4 地下室相关列 - NA表示"没有地下室"
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_cols:
        if col in df.columns:
            print(f"处理 {col} 缺失值...")
            df[col] = df[col].fillna('None')
    
    # 1.5 Electrical（电气系统）- 用众数填充
    if 'Electrical' in df.columns:
        print("处理 Electrical 缺失值...")
        mode_value = df['Electrical'].mode()
        if not mode_value.empty:
            df['Electrical'] = df['Electrical'].fillna(mode_value[0])
        else:
            df['Electrical'] = df['Electrical'].fillna('Unknown')
    
    # ==========================================
    # 2. 处理剩余缺失值（如果有）
    # ==========================================
    remaining_missing = df.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    
    if len(remaining_missing) > 0:
        print(f"\n处理其他 {len(remaining_missing)} 个列的剩余缺失值...")
        for col in remaining_missing.index:
            if df[col].dtype in ['int64', 'float64']:
                # 数值型：用中位数填充
                df[col] = df[col].fillna(df[col].median())
            else:
                # 分类型：用众数填充，如果没有众数则用'Unknown'
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna('Unknown')
    
    # ==========================================
    # 3. 删除重复行
    # ==========================================
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"\n删除 {duplicate_count} 个重复行...")
        df = df.drop_duplicates()
    
    # ==========================================
    # 4. 保存清洗后的数据
    # ==========================================
    try:
        # 确保目录存在
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df.to_csv(output_path, index=False)
        print(f"\n清洗后数据已保存至: {output_path}")
    except Exception as e:
        print(f"保存数据时发生错误: {str(e)}")
        return None
    
    # ==========================================
    # 5. 生成清洗统计报告
    # ==========================================
    final_shape = df.shape
    final_missing = df.isnull().sum().sum()
    
    cleaning_stats = {
        '原始数据行数': original_shape[0],
        '原始数据列数': original_shape[1],
        '清洗后行数': final_shape[0],
        '清洗后列数': final_shape[1],
        '删除重复行数': duplicate_count,
        '填充缺失值总数': original_missing - final_missing,
        '剩余缺失值数': final_missing,
        '输出文件路径': output_path
    }
    
    print("\n" + "="*50)
    print("数据清洗完成！统计信息:")
    print("="*50)
    for key, value in cleaning_stats.items():
        print(f"{key}: {value}")
    print("="*50)
    
    # 显示各列的缺失值情况（用于验证）
    print("\n各列缺失值统计（Top 10）:")
    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
    if len(missing_stats) > 0:
        print(missing_stats.head(10))
    else:
        print("所有列均无缺失值！")
    
    return cleaning_stats

# 执行清洗
if __name__ == "__main__":
    stats = clean_housing_data()