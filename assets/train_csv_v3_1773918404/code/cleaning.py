import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 定义数据路径
INPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/train.csv'
OUTPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'

# 定义实际数据的列名（基于提供的信息）
NUMERIC_COLS = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
                'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
                'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF']

CATEGORICAL_COLS = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
                   'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
                   'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl']

MISSING_COLS = ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 
                'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical']

def load_data(path):
    """加载数据并返回DataFrame"""
    df = pd.read_csv(path)
    print(f"原始数据形状: {df.shape}")
    return df

def analyze_missing(df):
    """分析缺失值情况"""
    missing_stats = pd.DataFrame({
        '缺失数量': df.isnull().sum(),
        '缺失比例': df.isnull().sum() / len(df) * 100
    })
    missing_stats = missing_stats[missing_stats['缺失数量'] > 0].sort_values('缺失比例', ascending=False)
    print("\n缺失值统计:")
    print(missing_stats)
    return missing_stats

def handle_missing_values(df):
    """处理缺失值"""
    df_clean = df.copy()
    
    for col in df_clean.columns:
        missing_ratio = df_clean[col].isnull().sum() / len(df_clean)
        
        if missing_ratio == 0:
            continue
            
        # 策略1: 缺失比例 > 50%，删除列
        if missing_ratio > 0.5:
            print(f"删除列 {col} (缺失率: {missing_ratio:.2%})")
            df_clean.drop(columns=[col], inplace=True)
            
        # 策略2: 数值型变量 - 使用中位数填充
        elif df_clean[col].dtype in ['int64', 'float64']:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"列 {col} 使用 median ({median_val}) 填充缺失值")
                
        # 策略3: 类别型变量 - 使用众数或'None'填充
        else:
            if not df_clean[col].mode().empty:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"列 {col} 使用 mode ({mode_val}) 填充缺失值")
            else:
                df_clean[col].fillna('None', inplace=True)
                print(f"列 {col} 使用 'None' 填充缺失值")
    
    return df_clean

def handle_duplicates(df):
    """处理重复值"""
    duplicates = df.duplicated().sum()
    print(f"\n完全重复行数: {duplicates}")
    
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"删除重复行后剩余: {len(df)} 行")
    
    # 检查基于Id的重复（如果存在Id列）
    if 'Id' in df.columns:
        id_duplicates = df.duplicated(subset=['Id']).sum()
        print(f"基于Id的重复行数: {id_duplicates}")
        if id_duplicates > 0:
            df = df.drop_duplicates(subset=['Id'], keep='first')
            print(f"删除Id重复后剩余: {len(df)} 行")
    
    return df

def detect_outliers_iqr(df, column):
    """使用IQR方法检测异常值"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def handle_outliers(df, method='clip'):
    """处理异常值"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_report = {}
    
    for col in numeric_cols:
        if col == 'Id':  # 跳过Id列
            continue
            
        outliers, lower, upper = detect_outliers_iqr(df, col)
        
        if len(outliers) > 0:
            outlier_report[col] = len(outliers)
            
            if method == 'clip':  # 缩尾处理
                df[col] = df[col].clip(lower, upper)
            elif method == 'remove':  # 删除
                df = df[~df.index.isin(outliers.index)]
    
    if outlier_report:
        print(f"\n异常值处理报告 (method={method}):")
        for col, count in outlier_report.items():
            print(f"  {col}: {count} 个异常值")
    
    return df

def optimize_data_types(df):
    """优化数据类型"""
    for col in df.columns:
        # 如果列名在数值列列表中但类型为object，尝试转换
        if col in NUMERIC_COLS and df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"列 {col} 转换为数值型")
            except:
                pass
        
        # 类别型优化：如果唯一值比例小于50%且为object类型
        if df[col].dtype == 'object':
            num_unique = df[col].nunique()
            if num_unique / len(df) < 0.5:
                df[col] = df[col].astype('category')
                print(f"列 {col} 转换为category类型 (唯一值: {num_unique})")
    
    return df

def clean_text_columns(df):
    """清洗文本列"""
    text_cols = df.select_dtypes(include=['object']).columns
    
    for col in text_cols:
        # 去除前后空格
        df[col] = df[col].astype(str).str.strip()
        
        # 统一大小写（对于短文本类别）
        if df[col].str.len().mean() < 50:
            df[col] = df[col].str.lower()
        
        # 处理空字符串和特殊标记
        df[col] = df[col].replace(['nan', 'null', 'none', 'na', ''], np.nan)
    
    return df

def validate_cleaning(df_original, df_cleaned):
    """验证清洗效果"""
    report = {
        '原始行数': len(df_original),
        '清洗后行数': len(df_cleaned),
        '删除行数': len(df_original) - len(df_cleaned),
        '缺失值总数(清洗后)': df_cleaned.isnull().sum().sum(),
        '重复行数(清洗后)': df_cleaned.duplicated().sum(),
        '数值列数': len(df_cleaned.select_dtypes(include=[np.number]).columns),
        '类别列数': len(df_cleaned.select_dtypes(include=['object', 'category']).columns)
    }
    return report

def main():
    """主函数：执行完整的数据清洗流程"""
    print("="*60)
    print("开始数据清洗流程")
    print("="*60)
    
    # 步骤1: 加载数据
    print("\n步骤1: 加载数据...")
    df = load_data(INPUT_PATH)
    df_original = df.copy()
    
    # 步骤2: 分析缺失值
    print("\n步骤2: 分析缺失值...")
    analyze_missing(df)
    
    # 步骤3: 处理缺失值
    print("\n步骤3: 处理缺失值...")
    df = handle_missing_values(df)
    
    # 步骤4: 处理重复值
    print("\n步骤4: 处理重复值...")
    df = handle_duplicates(df)
    
    # 步骤5: 处理异常值
    print("\n步骤5: 处理异常值...")
    df = handle_outliers(df, method='clip')
    
    # 步骤6: 优化数据类型
    print("\n步骤6: 优化数据类型...")
    df = optimize_data_types(df)
    
    # 步骤7: 清洗文本数据
    print("\n步骤7: 清洗文本数据...")
    df = clean_text_columns(df)
    
    # 步骤8: 最终验证
    print("\n步骤8: 验证清洗结果...")
    report = validate_cleaning(df_original, df)
    
    print("\n" + "="*60)
    print("清洗结果统计")
    print("="*60)
    for key, value in report.items():
        print(f"{key}: {value}")
    
    # 步骤9: 保存清洗后的数据
    print(f"\n步骤9: 保存清洗后的数据到 {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    print("数据清洗完成！")
    
    # 返回清洗后的DataFrame（如果在交互环境中使用）
    return df, report

if __name__ == "__main__":
    df_cleaned, cleaning_report = main()