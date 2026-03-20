```python
"""
数据清洗完整脚本
数据路径: /Users/cjialin/code/AutoMLByLLM/train.csv
输出路径: /Users/cjialin/code/AutoMLByLLM/train_cleaned.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os

# 忽略警告信息以保持输出整洁
warnings.filterwarnings('ignore')

# ============================================================
# 配置参数（可根据实际数据调整）
# ============================================================
CONFIG = {
    'input_path': '/Users/cjialin/code/AutoMLByLLM/train.csv',
    'output_path': '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv',
    'report_path': '/Users/cjialin/code/AutoMLByLLM/cleaning_report.txt',
    'missing_threshold': 0.5,          # 高缺失率阈值（50%）
    'outlier_method': 'percentile',    # 异常值处理方法：percentile, iqr, zscore
    'outlier_bounds': (0.01, 0.99),    # 百分位边界（1% - 99%）
    'protected_columns': ['id', 'target', 'label', 'target'],  # 不处理的保护列
    'cardinality_threshold': 50,       # 类别型转换阈值
    'convert_numeric_threshold': 0.9,  # 自动转数值阈值（90%可转换）
}

# ============================================================
# 第一部分：数据质量分析函数
# ============================================================

def analyze_missing_values(df):
    """
    分析缺失值情况
    
    参数:
        df: pandas DataFrame
    返回:
        缺失值统计DataFrame
    """
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2),
        'Dtype': df.dtypes.values
    })
    missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )
    return missing_stats

def detect_outliers_iqr(df, column):
    """
    使用IQR方法检测异常值
    
    参数:
        df: pandas DataFrame
        column: 列名
    返回:
        (异常值数量, 下界, 上界)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

def analyze_outliers(df):
    """
    分析所有数值型列的异常值
    
    参数:
        df: pandas DataFrame
    返回:
        异常值报告DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_report = []
    
    for col in numeric_cols:
        count, lower, upper = detect_outliers_iqr(df, col)
        if count > 0:
            outlier_report.append({
                'Column': col,
                'Outlier_Count': count,
                'Outlier_Percentage': round(count / len(df) * 100, 2),
                'Lower_Bound': round(lower, 2),
                'Upper_Bound': round(upper, 2)
            })
    
    return pd.DataFrame(outlier_report) if outlier_report else pd.DataFrame()

def generate_quality_report(df):
    """
    生成完整的数据质量报告
    
    参数:
        df: pandas DataFrame
    返回:
        质量报告字典
    """
    report = {
        '基本信息': {
            '总行数': len(df),
            '总列数': len(df.columns),
            '内存使用(MB)': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        },
        '缺失值': {
            '有缺失值的列数': df.isnull().any().sum(),
            '完全缺失的列': df.columns[df.isnull().all()].tolist(),
            '高缺失率列(>50%)': df.columns[df.isnull().mean() > 0.5].tolist()
        },
        '重复值': {
            '完全重复行数': df.duplicated().sum(),
            '重复率(%)': round(df.duplicated().sum() / len(df) * 100, 2)
        },
        '数据类型': {
            '数值型列数': len(df.select_dtypes(include=[np.number]).columns),
            '类别型列数': len(df.select_dtypes(include=['object']).columns),
            '其他类型列数': len(df.columns) - len(df.select_dtypes(include=[np.number, 'object']).columns)
        }
    }
    return report

# ============================================================
# 第二部分：数据清洗函数
# ============================================================

def load_and_backup_data(file_path):
    """
    加载数据并创建备份
    
    参数:
        file_path: 数据文件路径
    返回:
        (原始DataFrame, 清洗用DataFrame, 清洗日志列表)
    """
    print(f"正在加载数据: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 加载数据
    df_original = pd.read_csv(file_path)
    df_cleaned = df_original.copy()
    
    # 初始化清洗日志
    cleaning_log = []
    cleaning_log.append(f"[{datetime.now()}] 原始数据加载成功")
    cleaning_log.append(f"[{datetime.now()}] 原始数据形状: {df_original.shape}")
    cleaning_log.append(f"[{datetime.now()}] 原始数据内存使用: {df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"数据加载完成，形状: {df_original.shape}")
    return df_original, df_cleaned, cleaning_log

def standardize_column_names(df, cleaning_log):
    """
    标准化列名
    
    参数:
        df: pandas DataFrame
        cleaning_log: 清洗日志列表
    返回:
        处理后的DataFrame
    """
    print("\n步骤1: 标准化列名...")
    original_columns = df.columns.tolist()
    
    # 标准化列名：小写、去除空格、替换特殊字符为下划线
    df.columns = (
        df.columns
        .str.lower()                    # 转为小写
        .str.strip()                    # 去除首尾空格
        .str.replace(' ', '_', regex=False)     # 空格替换为下划线
        .str.replace('-', '_', regex=False)     # 连字符替换为下划线
        .str.replace('[^a-z0-9_]', '', regex=True)  # 去除其他特殊字符
    )
    
    # 记录列名变化
    changed_columns = [f"{old} -> {new}" for old, new in zip(original_columns, df.columns) if old != new]
    if changed_columns:
        cleaning_log.append(f"[{datetime.now()}] 列名标准化: 修改了 {len(changed_columns)} 个列名")
        for change in changed_columns[:5]:  # 只记录前5个
            cleaning_log.append(f"  {change}")
        if len(changed_columns) > 5:
            cleaning_log.append(f"  ... 还有 {len(changed_columns) - 5} 个列名被修改")
    else:
        cleaning_log.append(f"[{datetime.now()}] 列名标准化: 无需修改")
    
    print(f"列名标准化完成，共 {len(df.columns)} 列")
    return df

def handle_missing_values(df, cleaning_log, config):
    """
    处理缺失值
    
    策略：
    1. 删除完全缺失的列
    2. 高缺失率列(>50%)标记为指示变量
    3. 数值型列根据偏度选择填充方法（均值/中位数）
    4. 类别型列使用众数填充
    
    参数:
        df: pandas DataFrame
        cleaning_log: 清洗日志列表
        config: 配置字典
    返回:
        处理后的DataFrame
    """
    print("\n步骤2: 处理缺失值...")
    missing_before = df.isnull().sum().sum()
    
    # 策略1: 删除完全缺失的列
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        df = df.drop(columns=empty_cols)
        cleaning_log.append(f"[{datetime.now()}] 删除完全缺失列: {empty_cols}")
        print(f"  删除完全缺失列: {empty_cols}")
    
    # 策略2: 高缺失率列(>50%)处理 - 添加缺失指示变量
    high_missing_cols = df.columns[df.isnull().mean() > config['missing_threshold']].tolist()
    if high_missing_cols:
        for col in high_missing_cols:
            df[f'{col}_is_missing'] = df[col].isnull().astype(int)
        cleaning_log.append(f"[{datetime.now()}] 高缺失率列处理(>{config['missing_threshold']*100}%): {high_missing_cols}")
        print(f"  高缺失率列添加指示变量: {high_missing_cols}")
    
    # 策略3: 数值型列填充
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    filled_numeric = 0
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            # 根据偏度选择填充策略
            skewness = df[col].skew()
            if abs(skewness) > 1:  # 高度偏斜，使用中位数（对异常值更稳健）
                fill_value = df[col].median()
                method = "中位数"
            else:  # 近似正态，使用均值
                fill_value = df[col].mean()
                method = "均值"
            
            df[col] = df[col].fillna(fill_value)
            cleaning_log.append(f"[{datetime.now()}] {col}: {method}填充 {missing_count}个缺失值 ({fill_value:.4f}), 偏度={skewness:.2f}")
            filled_numeric += 1
    
    # 策略4: 类别型列填充
    categorical_cols = df.select_dtypes(include=['object']).columns
    filled_categorical = 0
    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_value)
            cleaning_log.append(f"[{datetime.now()}] {col}: 众数填充 {missing_count}个缺失值 ({mode_value})")
            filled_categorical += 1
    
    missing_after = df.isnull().sum().sum()
    print(f"  缺失值处理完成: 数值型列填充 {filled_numeric} 个，类别型列填充 {filled_categorical} 个")
    print(f"  总缺失值: {missing_before} -> {missing_after}")
    
    return df

def winsorize_series(series, lower_percentile=0.01, upper_percentile=0.99):
    """
    Winsorization缩尾处理：将极端值替换为指定分位数边界值
    
    参数:
        series: pandas Series
        lower_percentile: 下百分位
        upper_percentile: 上百分位
    返回:
        处理后的Series
    """
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower, upper)

def handle_outliers(df, cleaning_log, config):
    """
    处理异常值（Winsorization方法）
    
    参数:
        df: pandas DataFrame
        cleaning_log: 清洗日志列表
        config: 配置字典
    返回:
        处理后的DataFrame
    """
    print("\n步骤3: 处理异常值...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    protected = config['protected_columns']
    outlier_lower, outlier_upper = config['outlier_bounds']
    processed_count = 0
    
    for col in numeric_cols:
        # 跳过保护列（如ID、目标变量等）
        if col in protected:
            continue
        
        # 统计处理前的极端值数量（基于百分位）
        outlier_count_before = ((df[col] < df[col].quantile(outlier_lower)) | 
                               (df[col] > df[col].quantile(outlier_upper))).sum()
        
        if outlier_count_before > 0:
            # 执行Winsorization
            df[col] = winsorize_series(df[col], outlier_lower, outlier_upper)
            cleaning_log.append(
                f"[{datetime.now()}] {col}: Winsorization处理 {outlier_count_before} 个极端值 "
                f"([{outlier_lower*100}%, {outlier_upper*100}%] 百分位)"
            )
            processed_count += 1
    
    print(f"  异常值处理完成: 处理了 {processed_count} 个数值型列")
    return df

def remove_duplicates(df, cleaning_log):
    """
    删除重复行
    
    参数:
        df: pandas DataFrame
        cleaning_log: 清洗日志列表
    返回:
        处理后的DataFrame
    """
    print("\n步骤4: 删除重复值...")
    duplicate_count = df.duplicated().sum()
    
    if duplicate_count > 0:
        df = df.drop_duplicates()
        cleaning_log.append(f"[{datetime.now()}] 删除完全重复行: {duplicate_count} 行")
        print(f"  删除重复行: {duplicate_count} 行")
    else:
        cleaning_log.append(f"[{datetime.now()}] 未发现重复行")
        print("  未发现重复行")
    
    return df

def optimize_data_types(df, cleaning_log, config):
    """
    优化数据类型
    
    策略：
    1. 尝试将可转换的字符串列转为数值型
    2. 低基数(<50)的类别型列转为category类型以节省内存
    
    参数:
        df: pandas DataFrame
        cleaning_log: 清洗日志列表
        config: 配置字典
    返回:
        处理后的DataFrame
    """
    print("\n步骤5: 优化数据类型...")
    memory_before = df.memory_usage(deep=True).sum()
    
    converted_numeric = 0
    converted_category = 0
    
    # 策略1: 自动检测并转换数值列
    for col in df.select_dtypes(include=['object']).columns:
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            conversion_rate = converted.notna().sum() / len(df)
            
            if conversion_rate > config['convert_numeric_threshold']:
                df[col] = converted
                cleaning_log.append(f"[{datetime.now()}] {col}: 转换为数值类型 (转换率 {conversion_rate*100:.1f}%)")
                converted_numeric += 1
        except Exception:
            pass
    
    # 策略2: 低基数类别型列转为category
    for col in df.select_dtypes(include=['object']).columns:
        unique_count = df[col].nunique()
        if unique_count < config['cardinality_threshold']:
            df[col] = df[col].astype('category')
            cleaning_log.append(f"[{datetime.now()}] {col}: 转换为category类型 (基数 {unique_count})")
            converted_category += 1
    
    memory_after = df.memory_usage(deep=True).sum()
    memory_saved = memory_before - memory_after
    
    print(f"  数据类型优化完成:")
    print(f"    转换为数值型: {converted_numeric} 列")
    print(f"    转换为Category: {converted_category} 列")
    print(f"    内存节省: {memory_saved / 1024**2:.2f} MB ({memory_saved/memory_before*100:.1f}%)")
    
    return df

def clean_string_values(df, cleaning_log):
    """
    清洗字符串值
    
    策略：
    1. 去除首尾空格
    2. 标准化空白字符（多个空格转为单个）
    
    参数:
        df: pandas DataFrame
        cleaning_log: 清洗日志列表
    返回:
        处理后的DataFrame
    """
    print("\n步骤6: 清洗字符串格式...")
    string_cols = df.select_dtypes(include=['object']).columns
    cleaned_count = 0
    
    for col in string_cols:
        # 去除首尾空格
        df[col] = df[col].str.strip()
        # 标准化空白字符（多个空格、制表符等转为单个空格）
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        cleaned_count += 1
    
    cleaning_log.append(f"[{datetime.now()}] 字符串格式标准化: 处理 {cleaned_count} 列")
    print(f"  字符串清洗完成: 处理了 {cleaned_count} 个列")
    return df

def validate_and_save(df, df_original, cleaning_log, config):
    """
    验证清洗结果并保存
    
    参数:
        df