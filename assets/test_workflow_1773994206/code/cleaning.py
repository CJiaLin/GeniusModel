import pandas as pd
import numpy as np
from scipy import stats
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df_original = None
        self.df_cleaned = None
        self.report = {}
        
    def load_data(self):
        """加载数据并记录初始状态"""
        print(f"正在加载数据: {self.input_path}")
        self.df_original = pd.read_csv(self.input_path)
        self.df_cleaned = self.df_original.copy()
        
        self.report['原始数据形状'] = self.df_original.shape
        self.report['原始列名'] = self.df_original.columns.tolist()
        self.report['原始数据类型'] = self.df_original.dtypes.to_dict()
        
        print(f"数据加载完成，形状: {self.df_original.shape}")
        print(f"列数: {len(self.df_original.columns)}")
        return self
    
    def analyze_missing_values(self):
        """分析缺失值情况"""
        missing_stats = self.df_cleaned.isnull().sum()
        missing_percent = (missing_stats / len(self.df_cleaned) * 100).round(2)
        
        missing_df = pd.DataFrame({
            '缺失数量': missing_stats,
            '缺失比例(%)': missing_percent
        })
        
        self.report['缺失值分析'] = missing_df[missing_df['缺失数量'] > 0].to_dict()
        return missing_df
    
    def remove_duplicates(self):
        """处理重复值"""
        initial_rows = len(self.df_cleaned)
        duplicate_count = self.df_cleaned.duplicated().sum()
        
        self.df_cleaned = self.df_cleaned.drop_duplicates(keep='first')
        removed_rows = initial_rows - len(self.df_cleaned)
        
        self.report['重复值处理'] = {
            '原始重复行数': int(duplicate_count),
            '删除行数': int(removed_rows),
            '剩余行数': int(len(self.df_cleaned))
        }
        
        print(f"删除重复行: {removed_rows} 行")
        return self
    
    def handle_missing_values(self, drop_threshold=50):
        """处理缺失值：高比例删除，数值型中位数填充，类别型众数填充"""
        missing_before = self.df_cleaned.isnull().sum().sum()
        
        # 分析每列的缺失比例
        missing_percent = (self.df_cleaned.isnull().sum() / len(self.df_cleaned) * 100)
        
        # 删除缺失比例过高的列
        cols_to_drop = missing_percent[missing_percent > drop_threshold].index.tolist()
        if cols_to_drop:
            self.df_cleaned = self.df_cleaned.drop(columns=cols_to_drop)
            print(f"删除高缺失列(>{drop_threshold}%): {cols_to_drop}")
        
        self.report['删除的高缺失列'] = cols_to_drop
        
        # 分别处理数值型和类别型缺失值
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df_cleaned.select_dtypes(include=['object']).columns
        
        # 数值型：中位数填充
        for col in numeric_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                median_val = self.df_cleaned[col].median()
                self.df_cleaned[col].fillna(median_val, inplace=True)
                print(f"数值列 '{col}' 使用中位数 {median_val} 填充")
        
        # 类别型：众数填充，如果没有众数则使用'Unknown'
        for col in categorical_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                mode_val = self.df_cleaned[col].mode()
                if len(mode_val) > 0:
                    self.df_cleaned[col].fillna(mode_val[0], inplace=True)
                    print(f"类别列 '{col}' 使用众数 '{mode_val[0]}' 填充")
                else:
                    self.df_cleaned[col].fillna('Unknown', inplace=True)
                    print(f"类别列 '{col}' 使用 'Unknown' 填充")
        
        missing_after = self.df_cleaned.isnull().sum().sum()
        self.report['缺失值处理'] = {
            '处理前缺失值总数': int(missing_before),
            '处理后缺失值总数': int(missing_after),
            '填充策略': '数值型-中位数，类别型-众数或Unknown'
        }
        
        return self
    
    def detect_and_handle_outliers(self, method='iqr', threshold=1.5):
        """检测并处理异常值，使用IQR方法进行边界截断"""
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns
        outlier_stats = {}
        
        for col in numeric_cols:
            if self.df_cleaned[col].nunique() < 10:  # 跳过类别型数值列（如0/1）
                continue
                
            Q1 = self.df_cleaned[col].quantile(0.25)
            Q3 = self.df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = self.df_cleaned[(self.df_cleaned[col] < lower_bound) | 
                                      (self.df_cleaned[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                # Winsorization: 截断到边界值
                self.df_cleaned[col] = self.df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                outlier_stats[col] = {
                    '异常值数量': int(outlier_count),
                    '异常值比例(%)': round(outlier_count / len(self.df_cleaned) * 100, 2),
                    '下界': float(lower_bound),
                    '上界': float(upper_bound)
                }
                print(f"列 '{col}': 截断 {outlier_count} 个异常值到 [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        self.report['异常值处理'] = outlier_stats
        return self
    
    def convert_data_types(self):
        """自动转换数据类型：尝试识别日期列，优化内存使用"""
        # 尝试自动识别日期列
        for col in self.df_cleaned.select_dtypes(include=['object']).columns:
            try:
                # 如果列名包含日期关键词，尝试转换
                if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'month', 'year']):
                    self.df_cleaned[col] = pd.to_datetime(self.df_cleaned[col], errors='ignore')
                    print(f"列 '{col}' 转换为日期时间类型")
            except:
                pass
        
        # 优化数值类型内存使用
        for col in self.df_cleaned.select_dtypes(include=['int']).columns:
            self.df_cleaned[col] = pd.to_numeric(self.df_cleaned[col], downcast='integer')
        
        for col in self.df_cleaned.select_dtypes(include=['float']).columns:
            self.df_cleaned[col] = pd.to_numeric(self.df_cleaned[col], downcast='float')
        
        self.report['清洗后数据类型'] = self.df_cleaned.dtypes.to_dict()
        return self
    
    def standardize_text(self):
        """标准化文本数据：去除空格、统一小写、去除特殊字符"""
        string_cols = self.df_cleaned.select_dtypes(include=['object']).columns
        
        for col in string_cols:
            # 去除首尾空格并统一小写
            self.df_cleaned[col] = self.df_cleaned[col].astype(str).str.strip().str.lower()
            # 将'nan'字符串转回真正的NaN（如果有的话）
            self.df_cleaned[col] = self.df_cleaned[col].replace('nan', np.nan)
            # 填充NaN
            if self.df_cleaned[col].isnull().sum() > 0:
                self.df_cleaned[col].fillna('unknown', inplace=True)
        
        self.report['文本标准化'] = {
            '处理的列': string_cols.tolist(),
            '操作': ['去除首尾空格', '转换为小写', '填充缺失值为unknown']
        }
        print(f"文本标准化完成，处理列: {list(string_cols)}")
        return self
    
    def generate_report(self):
        """生成最终清洗报告"""
        self.report['清洗后数据形状'] = self.df_cleaned.shape
        self.report['数据保留率(%)'] = round(
            len(self.df_cleaned) / len(self.df_original) * 100, 2
        )
        
        print("\n" + "="*50)
        print("数据清洗报告")
        print("="*50)
        print(f"原始数据形状: {self.report['原始数据形状']}")
        print(f"清洗后数据形状: {self.report['清洗后数据形状']}")
        print(f"数据保留率: {self.report['数据保留率(%)']}%")
        print(f"删除的列: {self.report.get('删除的高缺失列', [])}")
        print("="*50)
        
        return self.report
    
    def save_cleaned_data(self):
        """保存清洗后的数据"""
        # 确保目录存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.df_cleaned.to_csv(self.output_path, index=False)
        print(f"清洗后的数据已保存至: {self.output_path}")
        return self
    
    def run_full_pipeline(self):
        """执行完整的数据清洗流程"""
        print("开始数据清洗流程...")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.load_data()
        print("\n步骤1: 分析缺失值...")
        self.analyze_missing_values()
        
        print("\n步骤2: 处理重复值...")
        self.remove_duplicates()
        
        print("\n步骤3: 处理缺失值...")
        self.handle_missing_values()
        
        print("\n步骤4: 检测并处理异常值...")
        self.detect_and_handle_outliers()
        
        print("\n步骤5: 转换数据类型...")
        self.convert_data_types()
        
        print("\n步骤6: 标准化文本数据...")
        self.standardize_text()
        
        print("\n步骤7: 保存数据...")
        self.save_cleaned_data()
        
        print("\n步骤8: 生成报告...")
        report = self.generate_report()
        
        print("\n数据清洗完成！")
        return report, self.df_cleaned

# 主执行函数
def main():
    # 配置路径
    input_path = '/Users/cjialin/code/AutoMLByLLM/train.csv'
    output_path = '/Users/cjialin/code/AutoMLByLLM/train_cleaned.csv'
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在 {input_path}")
        return None, None
    
    # 创建清洗器实例并执行
    cleaner = DataCleaner(input_path, output_path)
    report, df_cleaned = cleaner.run_full_pipeline()
    
    # 返回清洗结果统计
    result_stats = {
        'input_shape': report['原始数据形状'],
        'output_shape': report['清洗后数据形状'],
        'retention_rate': report['数据保留率(%)'],
        'columns_dropped': report.get('删除的高缺失列', []),
        'rows_before': report['原始数据形状'][0],
        'rows_after': report['清洗后数据形状'][0],
        'columns_before': report['原始数据形状'][1],
        'columns_after': report['清洗后数据形状'][1]
    }
    
    print("\n清洗结果统计:")
    for key, value in result_stats.items():
        print(f"  {key}: {value}")
    
    return report, df_cleaned

if __name__ == "__main__":
    report, df_cleaned = main()