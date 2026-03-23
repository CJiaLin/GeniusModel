import pandas as pd
import numpy as np

def clean_housing_data():
    """
    清洗房价数据集，处理缺失值并保存结果
    """
    # 定义输入输出路径
    input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    print(f"开始读取数据: {input_path}")
    
    try:
        # 读取原始数据
        df = pd.read_csv(input_path)
        
        # 记录原始数据信息
        original_shape = df.shape
        original_missing = df.isnull().sum()
        total_original_missing = original_missing.sum()
        
        print(f"原始数据形状: {original_shape}")
        print(f"原始缺失值总数: {total_original_missing}")
        
        # ==================== 缺失值处理 ====================
        
        # 1. LotFrontage - 数值型（临街距离），使用整体中位数填充
        if 'LotFrontage' in df.columns:
            median_lot_frontage = df['LotFrontage'].median()
            df['LotFrontage'].fillna(median_lot_frontage, inplace=True)
            print(f"LotFrontage 缺失值已用中位数 {median_lot_frontage} 填充")
        
        # 2. Alley - 分类变量（小巷通道），用'NA'表示没有小巷
        if 'Alley' in df.columns:
            df['Alley'].fillna('NA', inplace=True)
            print("Alley 缺失值已用 'NA' 填充")
        
        # 3. MasVnrType - 分类变量（砌体类型），用'None'表示无砌体
        if 'MasVnrType' in df.columns:
            df['MasVnrType'].fillna('None', inplace=True)
            print("MasVnrType 缺失值已用 'None' 填充")
        
        # 4. MasVnrArea - 数值型（砌体面积），用0填充（与'None'类型对应）
        if 'MasVnrArea' in df.columns:
            df['MasVnrArea'].fillna(0, inplace=True)
            print("MasVnrArea 缺失值已用 0 填充")
        
        # 5. 地下室相关分类变量 - 用'NA'表示没有地下室
        basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        for col in basement_cols:
            if col in df.columns:
                df[col].fillna('NA', inplace=True)
                print(f"{col} 缺失值已用 'NA' 填充")
        
        # 6. Electrical - 分类变量（电力系统），使用众数填充
        if 'Electrical' in df.columns:
            mode_electrical = df['Electrical'].mode()[0]
            df['Electrical'].fillna(mode_electrical, inplace=True)
            print(f"Electrical 缺失值已用众数 '{mode_electrical}' 填充")
        
        # ==================== 数据类型优化 ====================
        # 确保数值列类型正确
        numeric_cols = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 
                       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 
                       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                       '1stFlrSF', '2ndFlrSF']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 确保分类列为category类型以节省内存
        categorical_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
                           'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
                           'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
                           'RoofStyle', 'RoofMatl']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # ==================== 保存清洗后的数据 ====================
        df.to_csv(output_path, index=False)
        print(f"\n清洗后的数据已保存至: {output_path}")
        
        # ==================== 生成统计报告 ====================
        cleaned_missing = df.isnull().sum().sum()
        
        result_stats = {
            'status': 'success',
            'input_path': input_path,
            'output_path': output_path,
            'original_shape': original_shape,
            'cleaned_shape': df.shape,
            'original_missing_count': int(total_original_missing),
            'cleaned_missing_count': int(cleaned_missing),
            'processed_columns': [
                {'column': 'LotFrontage', 'strategy': 'median', 'value': float(median_lot_frontage) if 'LotFrontage' in df.columns else None},
                {'column': 'Alley', 'strategy': 'constant', 'value': 'NA'},
                {'column': 'MasVnrType', 'strategy': 'constant', 'value': 'None'},
                {'column': 'MasVnrArea', 'strategy': 'constant', 'value': 0},
                {'column': 'BsmtQual', 'strategy': 'constant', 'value': 'NA'},
                {'column': 'BsmtCond', 'strategy': 'constant', 'value': 'NA'},
                {'column': 'BsmtExposure', 'strategy': 'constant', 'value': 'NA'},
                {'column': 'BsmtFinType1', 'strategy': 'constant', 'value': 'NA'},
                {'column': 'BsmtFinType2', 'strategy': 'constant', 'value': 'NA'},
                {'column': 'Electrical', 'strategy': 'mode', 'value': str(mode_electrical) if 'Electrical' in df.columns else None}
            ]
        }
        
        print("\n========== 清洗结果统计 ==========")
        print(f"原始数据行数: {original_shape[0]}")
        print(f"原始数据列数: {original_shape[1]}")
        print(f"清洗前缺失值总数: {total_original_missing}")
        print(f"清洗后缺失值总数: {cleaned_missing}")
        print(f"处理列数: {len([c for c in df.columns if c in ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical']])}")
        print("=====================================")
        
        return result_stats
        
    except FileNotFoundError:
        error_msg = f"错误：找不到文件 {input_path}"
        print(error_msg)
        return {'status': 'error', 'message': error_msg}
    except Exception as e:
        error_msg = f"清洗过程发生错误: {str(e)}"
        print(error_msg)
        return {'status': 'error', 'message': error_msg}

if __name__ == '__main__':
    result = clean_housing_data()
    
    if result['status'] == 'success':
        print("\n数据清洗成功完成！")
        print(f"清洗后文件路径: {result['output_path']}")
    else:
        print(f"\n数据清洗失败: {result['message']}")