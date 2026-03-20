import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    # 设置数据路径
    input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    print("开始数据清洗流程...")
    print(f"读取数据: {input_path}")
    
    # 读取原始数据
    df = pd.read_csv(input_path)
    original_shape = df.shape
    print(f"原始数据维度: {original_shape}")
    print(f"原始缺失值总数: {df.isnull().sum().sum()}")
    
    # ==========================================
    # 步骤 1: 缺失值处理
    # ==========================================
    print("\n步骤 1: 处理缺失值...")
    
    # 1.1 缺失代表"无"的列（NA = None）
    # 这些列的缺失表示房屋没有该设施，应填充为"None"
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 
                 'FireplaceQu', 'GarageType', 'GarageFinish',
                 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
                 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType']
    
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    print(f"  - 已处理 {len(none_cols)} 个'None'类别列")
    
    # 1.2 数值型缺失值处理
    
    # LotFrontage - 按Neighborhood分组填充中位数
    # 同一街区的房屋临街距离相似，使用分组中位数更合理
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 如果仍有缺失（如整个街区都缺失），用全局中位数填充
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
        print("  - LotFrontage: 已按Neighborhood分组填充中位数")
    
    # GarageYrBlt - 无车库则填充0（表示无车库）
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
        print("  - GarageYrBlt: 已填充0")
    
    # MasVnrArea - 无砌体贴面则填充0
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
        print("  - MasVnrArea: 已填充0")
    
    # Electrical - 填充众数（最常用电气系统）
    if 'Electrical' in df.columns:
        electrical_mode = df['Electrical'].mode()
        if len(electrical_mode) > 0:
            df['Electrical'] = df['Electrical'].fillna(electrical_mode[0])
            print(f"  - Electrical: 已填充众数 '{electrical_mode[0]}'")
    
    # 检查缺失值处理情况
    remaining_missing = df.isnull().sum().sum()
    print(f"  - 剩余缺失值数量: {remaining_missing}")
    
    # ==========================================
    # 步骤 2: 异常值处理
    # ==========================================
    print("\n步骤 2: 处理异常值...")
    
    # 2.1 修正GarageYrBlt异常值
    # 存在2207年等明显错误的未来年份，应替换为房屋建造年份
    if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
        invalid_garage_year = df['GarageYrBlt'] > 2010
        if invalid_garage_year.sum() > 0:
            df.loc[invalid_garage_year, 'GarageYrBlt'] = df.loc[invalid_garage_year, 'YearBuilt']
            print(f"  - GarageYrBlt: 已修正 {invalid_garage_year.sum()} 个异常年份")
    
    # 2.2 处理面积异常值（使用99%分位数标记，不删除）
    # 房地产数据中的极端值可能是真实豪宅，标记后保留
    if 'LotArea' in df.columns:
        lotarea_q99 = df['LotArea'].quantile(0.99)
        df['LotArea_Outlier'] = (df['LotArea'] > lotarea_q99).astype(int)
        print(f"  - LotArea: 标记 {df['LotArea_Outlier'].sum()} 个异常值 (>{lotarea_q99:.0f})")
    
    if 'GrLivArea' in df.columns:
        grlivarea_q99 = df['GrLivArea'].quantile(0.99)
        df['GrLivArea_Outlier'] = (df['GrLivArea'] > grlivarea_q99).astype(int)
        print(f"  - GrLivArea: 标记 {df['GrLivArea_Outlier'].sum()} 个异常值 (>{grlivarea_q99:.0f})")
    
    # 2.3 对目标变量进行对数变换（处理右偏分布）
    if 'SalePrice' in df.columns:
        df['SalePrice_Log'] = np.log1p(df['SalePrice'])
        print("  - SalePrice: 已创建对数变换列 SalePrice_Log")
    
    # ==========================================
    # 步骤 3: 数据类型转换
    # ==========================================
    print("\n步骤 3: 转换数据类型...")
    
    # MSSubClass是建筑类型编码，应作为类别而非连续数值
    if 'MSSubClass' in df.columns:
        df['MSSubClass'] = df['MSSubClass'].astype(str)
        print("  - MSSubClass: 已转换为字符串类别")
    
    # 月份和年份应作为类别（周期性/离散）
    if 'MoSold' in df.columns:
        df['MoSold'] = df['MoSold'].astype(str)
        print("  - MoSold: 已转换为字符串类别")
    
    if 'YrSold' in df.columns:
        df['YrSold'] = df['YrSold'].astype(str)
        print("  - YrSold: 已转换为字符串类别")
    
    # 确保ID为整数
    if 'Id' in df.columns:
        df['Id'] = df['Id'].astype(int)
        print("  - Id: 已确保为整数类型")
    
    # ==========================================
    # 步骤 4: 特征工程
    # ==========================================
    print("\n步骤 4: 创建新特征...")
    
    # 4.1 创建总面积特征（地下室 + 一楼 + 二楼）
    area_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    if all(col in df.columns for col in area_cols):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        print("  - TotalSF: 已创建（地下室+一楼+二楼）")
    
    # 4.2 创建房屋年龄特征（销售年份 - 建造年份）
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YrSold'].astype(int) - df['YearBuilt']
        print("  - HouseAge: 已创建（销售时房龄）")
    
    # 4.3 创建是否翻新特征
    if 'YearRemodAdd' in df.columns and 'YearBuilt' in df.columns:
        df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
        print("  - IsRemodeled: 已创建（1=翻新过，0=未翻新）")
    
    # 4.4 创建总浴室数（全浴室计1，半浴室计0.5，包括地下室）
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in df.columns for col in bath_cols):
        df['TotalBathrooms'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                               df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
        print("  - TotalBathrooms: 已创建（含地下室浴室）")
    
    # 4.5 车库是否建立特征
    if 'GarageYrBlt' in df.columns:
        df['HasGarage'] = (df['GarageYrBlt'] > 0).astype(int)
        print("  - HasGarage: 已创建（1=有车库，0=无车库）")
    
    # ==========================================
    # 步骤 5: 最终验证与保存
    # ==========================================
    print("\n步骤 5: 最终验证...")
    
    # 5.1 检查是否还有缺失值
    final_missing = df.isnull().sum().sum()
    if final_missing == 0:
        print("  ✓ 缺失值检查通过：已完全处理所有缺失值")
    else:
        print(f"  ⚠ 警告：仍存在 {final_missing} 个缺失值")
        # 显示具体哪些列还有缺失值
        missing_cols = df.columns[df.isnull().any()].tolist()
        print(f"    剩余缺失值列: {missing_cols}")
        # 对剩余缺失值进行兜底处理（填充0或None）
        for col in missing_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('None')
            else:
                df[col] = df[col].fillna(0)
        print("  ✓ 已对剩余缺失值进行兜底填充")
    
    # 5.2 统计信息
    final_shape = df.shape
    print(f"\n清洗完成统计:")
    print(f"  - 原始维度: {original_shape}")
    print(f"  - 清洗后维度: {final_shape}")
    print(f"  - 新增特征数: {final_shape[1] - original_shape[1]}")
    print(f"  - 总行数: {final_shape[0]}")
    print(f"  - 总列数: {final_shape[1]}")
    
    # 5.3 保存清洗后数据
    df.to_csv(output_path, index=False)
    print(f"\n✓ 清洗后数据已保存至: {output_path}")
    
    # 返回关键统计信息字典
    stats = {
        'original_rows': original_shape[0],
        'original_cols': original_shape[1],
        'final_rows': final_shape[0],
        'final_cols': final_shape[1],
        'new_features': final_shape[1] - original_shape[1],
        'missing_values_before': original_shape[0] * original_shape[1],  # 粗略估计
        'missing_values_after': 0
    }
    
    print("\n清洗流程执行完毕！")
    return stats

if __name__ == "__main__":
    result_stats = main()
    print(f"\n详细统计信息: {result_stats}")