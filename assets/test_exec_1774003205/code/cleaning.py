#!/usr/bin/env python3
"""
数据清洗完整脚本
基于AutoMLByLLM项目的数据清洗方案实现
"""

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

# 忽略警告以保持输出整洁
warnings.filterwarnings('ignore')

# 设置pandas显示选项，便于查看完整数据信息
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)

# 定义输入输出路径
INPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/train.csv'
OUTPUT_PATH = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'


def analyze_data_quality(df):
    """
    分析数据质量，生成详细报告
    """
    print("=" * 60)
    print("数据质量分析报告")
    print("=" * 60)
    
    # 基础信息
    print(f"\n【基础信息】")
    print(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 缺失值分析
    print(f"\n【缺失值分析】")
    missing_stats = pd.DataFrame({
        '列名': df.columns,
        '缺失数量': df.isnull().sum(),
        '缺失比例(%)': (df.isnull().sum() / len(df) * 100).round(2),
        '数据类型': df.dtypes
    })
    missing_stats = missing_stats[missing_stats['缺失数量'] > 0].sort_values('缺失比例(%)', ascending=False)
    
    if len(missing_stats) > 0:
        print(missing_stats.to_string(index=False))
        high_risk = missing_stats[missing_stats['缺失比例(%)'] > 50]['列名'].tolist()
        medium_risk = missing_stats[(missing_stats['缺失比例(%)'] >= 10) & (missing_stats['缺失比例(%)'] <= 50)]['列名'].tolist()
        low_risk = missing_stats[missing_stats['缺失比例(%)'] < 10]['列名'].tolist()
        
        if high_risk:
            print(f"\n🔴 高风险列(缺失>50%): {high_risk}")
        if medium_risk:
            print(f"🟡 中风险列(缺失10%-50%): {medium_risk}")
        if low_risk:
            print(f"🟢 低风险列(缺失<10%): {low_risk}")
    else:
        print("✓ 未发现缺失值")
    
    # 重复值分析
    print(f"\n【重复值分析】")
    duplicate_rows = df.duplicated().sum()
    print(f"完全重复行数: {duplicate_rows}")
    
    # 检测是否有明显的ID列
    id_like_cols = [col for col in df.columns if 'id' in str(col).lower()]
    for col in id_like_cols:
        if col in df.columns:
            id_dups = df[col].duplicated().sum()
            if id_dups > 0:
                print(f"列 '{col}' 重复值: {id_dups}")
    
    # 数值型列统计
    print(f"\n【数值型列统计】")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print(f"数值型列数量: {len(numeric_cols)}")
        desc = df[numeric_cols].describe().T
        print(desc[['count', 'mean', 'std', 'min', 'max']].round(2).to_string())
    else:
        print("无数值型列")
    
    # 分类型列统计
    print(f"\n【分类型列统计】")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        print(f"分类型列数量: {len(categorical_cols)}")
        for col in categorical_cols[:5]:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} 个唯一值")
            if unique_count <= 10:
                print(f"    取值分布: {df[col].value_counts().head(3).to_dict()}")
    else:
        print("无分类型列")
    
    print("=" * 60)
    return missing_stats


def clean_data():
    """
    执行完整的数据清洗流程
    """
    print(f"开始加载数据: {INPUT_PATH}")
    
    # 检查文件是否存在
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")
    
    # 加载原始数据
    df_raw = pd.read_csv(INPUT_PATH)
    df = df_raw.copy()
    
    print(f"原始数据加载完成: {df.shape}")
    
    # 记录清洗前的统计信息
    original_shape = df.shape
    original_memory = df.memory_usage(deep=True).sum()
    
    # 第一步：数据质量分析
    analyze_data_quality(df)
    
    print("\n" + "=" * 60)
    print("开始数据清洗流程")
    print("=" * 60)
    
    # 第二步：删除高缺失率列（缺失率 > 50%）
    print("\n【步骤1】删除高缺失率列(>50%)...")
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  删除列: {cols_to_drop}")
    else:
        print("  无高缺失率列需要删除")
    
    # 第三步：识别数值型和分类型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\n【步骤2】处理缺失值...")
    print(f"  数值型列({len(numeric_cols)}个): {numeric_cols}")
    print(f"  分类型列({len(categorical_cols)}个): {categorical_cols}")
    
    # 处理数值型缺失值
    numeric_filled = 0
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            skewness = df[col].skew()
            if abs(skewness) < 1:
                fill_value = df[col].mean()
                method = "均值"
            else:
                fill_value = df[col].median()
                method = "中位数"
            
            df[col] = df[col].fillna(fill_value)
            numeric_filled += 1
            print(f"  {col}: 使用{method}填充 {missing_count} 个缺失值 ({fill_value:.4f})")
    
    if numeric_filled == 0:
        print("  数值型列无缺失值")
    
    # 处理分类型缺失值
    categorical_filled = 0
    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            mode_val = df[col].mode()
            if not mode_val.empty:
                fill_value = mode_val[0]
                df[col] = df[col].fillna(fill_value)
                categorical_filled += 1
                print(f"  {col}: 使用众数填充 {missing_count} 个缺失值 ('{fill_value}')")
            else:
                df[col] = df[col].fillna('Unknown')
                print(f"  {col}: 使用'Unknown'填充 {missing_count} 个缺失值")
    
    if categorical_filled == 0:
        print("  分类型列无缺失值")
    
    # 第四步：异常值处理（盖帽法 - Winsorization）
    print(f"\n【步骤3】异常值处理(盖帽法1%-99%)...")
    outlier_processed = 0
    for col in numeric_cols:
        if col in df.columns:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            
            if (df[col] < lower).any() or (df[col] > upper).any():
                outliers_before = ((df[col] < lower) | (df[col] > upper)).sum()
                df[col] = df[col].clip(lower, upper)
                outlier_processed += 1
                print(f"  {col}: 处理 {outliers_before} 个异常值 -> 范围[{lower:.4f}, {upper:.4f}]")
    
    if outlier_processed == 0:
        print("  未检测到需要处理的极端异常值")
    
    # 第五步：删除重复行
    print(f"\n【步骤4】删除重复行...")
    before_dedup = len(df)
    
    # 删除完全重复的行
    df = df.drop_duplicates()
    
    # 如果有ID列，基于ID列去重（保留第一条）
    id_cols = [col for col in df.columns if 'id' in str(col).lower()]
    for col in id_cols:
        if col in df.columns and df[col].duplicated().sum() > 0:
            df = df.drop_duplicates(subset=[col], keep='first')
            print(f"  基于列'{col}'去重")
    
    after_dedup = len(df)
    removed_dups = before_dedup - after_dedup
    print(f"  删除重复行: {removed_dups} 行")
    print(f"  剩余数据: {after_dedup} 行")
    
    # 第六步：数据类型优化
    print(f"\n【步骤5】数据类型优化...")
    
    # 转换基数较小的分类列为category以节省内存
    for col in categorical_cols:
        if col in df.columns and df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    # 自动转换数据类型（pandas推断）
    df = df.convert_dtypes()
    
    # 尝试将数值型列降级为更节省内存的类型
    for col in numeric_cols:
        if col in df.columns:
            col_type = df[col].dtype
            if col_type == 'int64':
                if df[col].min() >= -128 and df[col].max() <= 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() >= -32768 and df[col].max() <= 32767:
                    df[col] = df[col].astype('int16')
                elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                    df[col] = df[col].astype('int32')
            elif col_type == 'float64':
                df[col] = df[col].astype('float32')
    
    new_memory = df.memory_usage(deep=True).sum()
    memory_saved = (original_memory - new_memory) / original_memory * 100
    print(f"  内存优化: 从 {original_memory/1024**2:.2f}MB 降至 {new_memory/1024**2:.2f}MB (节省 {memory_saved:.1f}%)")
    
    # 第七步：文本清理（去除首尾空格）
    print(f"\n【步骤6】文本标准化...")
    text_cleaned = 0
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if df[col].dtype == 'object':
            # 仅对字符串类型的数据进行处理
            if df[col].apply(lambda x: isinstance(x, str) if pd.notna(x) else True).all():
                df[col] = df[col].str.strip()
                text_cleaned += 1
    
    print(f"  处理了 {text_cleaned} 个文本列的空白字符")
    
    # 保存清洗后的数据
    print(f"\n【步骤7】保存清洗后数据...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  已保存至: {OUTPUT_PATH}")
    
    # 生成最终报告
    print("\n" + "=" * 60)
    print("数据清洗完成报告")
    print("=" * 60)
    
    final_shape = df.shape
    report = {
        '原始数据行数': original_shape[0],
        '原始数据列数': original_shape[1],
        '清洗后行数': final_shape[0],
        '清洗后列数': final_shape[1],
        '删除列数': original_shape[1] - final_shape[1],
        '删除行数': original_shape[0] - final_shape[0],
        '删除重复行': removed_dups,
        '处理缺失值列': numeric_filled + categorical_filled,
        '处理异常值列': outlier_processed,
        '原始内存MB': round(original_memory / 1024**2, 2),
        '清洗后内存MB': round(new_memory / 1024**2, 2),
        '内存节省%': round(memory_saved, 2)
    }
    
    for key, value in report.items():
        print(f"{key}: {value}")
    
    print("=" * 60)
    
    return df, report


if __name__ == '__main__':
    try:
        cleaned_df, cleaning_report = clean_data()
        print("\n✓ 数据清洗流程成功完成！")
    except Exception as e:
        print(f"\n✗ 数据清洗过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()