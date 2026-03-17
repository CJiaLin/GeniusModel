def format_analysis_report(analysis: dict, target: str = None) -> str:
    """
    格式化数据分析报告为 Markdown，包含详细的图表解读
    """
    report = f"""# 📊 数据分析报告

## 1. 数据概览

| 指标 | 值 |
|------|-----|
| 样本数 | {analysis['shape'][0]} |
| 特征数 | {analysis['shape'][1]} |
| 目标列 | {target or '未指定'} |

## 2. 数据质量评估

### 缺失值情况
"""
    
    # 添加缺失值信息
    missing_cols = {k: v for k, v in analysis['missing'].items() if v > 0}
    if missing_cols:
        report += "\n| 列名 | 缺失数 | 缺失率 |\n|------|--------|--------|\n"
        for col, count in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = analysis['missing_pct'][col]
            report += f"| {col} | {count} | {pct:.1f}% |\n"
    else:
        report += "\n✅ 数据完整，无缺失值\n"
    
    # 数值特征统计
    if analysis['numeric_summary']:
        report += """\n## 3. 数值特征统计

| 特征 | 均值 | 中位数 | 标准差 | 最小值 | 最大值 |
|------|------|--------|--------|--------|--------|
"""
        for col, stats in list(analysis['numeric_summary'].items())[:10]:
            report += f"| {col} | {stats['mean']:.2f} | {stats['median']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |\n"
    
    # 类别特征统计
    if analysis['categorical_summary']:
        report += """\n## 4. 类别特征统计

| 特征 | 唯一值数量 | 最常见值 |
|------|------------|----------|
"""
        for col, stats in list(analysis['categorical_summary'].items())[:10]:
            top_value = list(stats['top_values'].keys())[0] if stats['top_values'] else 'N/A'
            report += f"| {col} | {stats['unique_count']} | {top_value} |\n"
    
    # 可视化部分 - 添加详细的总结性描述
    report += """\n## 5. 数据可视化与解读\n\n"""
    
    # 1. 缺失值分布
    if "missing_values" in analysis['visualizations']:
        missing_summary = []
        for col, count in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = analysis['missing_pct'][col]
            missing_summary.append(f"- **{col}**: {count} 个缺失 ({pct:.1f}%)")
        
        report += f"""### 5.1 缺失值分布

![Missing Values](data:image/png;base64,{analysis['visualizations']['missing_values']})

**图表解读**:
上述柱状图展示了各特征的缺失值数量。从图中可以看出：
{chr(10).join(missing_summary) if missing_summary else '- 数据完整，无明显缺失值'}

**对数据清洗的指导**:
- 高缺失率(>50%)的特征建议删除
- 中等缺失率(10-50%)的特征需要填充策略
- 低缺失率(<10%)的特征可直接填充
- 类别特征和数值特征需要采用不同的填充方法

---

"""
    
    # 2. 数值特征分布
    if "numeric_distributions" in analysis['visualizations']:
        distribution_summary = []
        for col, stats in list(analysis['numeric_summary'].items())[:4]:
            skew = stats.get('skewness', 0)
            if abs(skew) > 1:
                dist_type = "右偏分布" if skew > 0 else "左偏分布"
                suggestion = "建议对数变换"
            else:
                dist_type = "近似正态分布"
                suggestion = "可直接标准化"
            distribution_summary.append(f"- **{col}**: {dist_type}，{suggestion}")
        
        report += f"""### 5.2 数值特征分布

![Numeric Distributions](data:image/png;base64,{analysis['visualizations']['numeric_distributions']})

**图表解读**:
上述直方图展示了主要数值特征的分布情况：
{chr(10).join(distribution_summary)}

**对数据清洗和特征工程的指导**:
- **右偏分布**（如收入、价格）: 使用对数变换(log1p)使其更接近正态分布
- **正态分布**: 使用z-score标准化
- **多峰分布**: 考虑分箱或创建类别特征
- **异常值**: 查看分布尾部，确定是否需要截断

---

"""
    
    # 3. 目标变量分布
    if "target_distribution" in analysis['visualizations'] and target:
        target_stats = analysis['numeric_summary'].get(target, {})
        if target_stats:
            skew = target_stats.get('skewness', 0)
            if abs(skew) > 1:
                target_suggestion = f"目标变量呈现{'右偏' if skew > 0 else '左偏'}分布，建议在特征工程阶段进行对数变换"
            else:
                target_suggestion = "目标变量分布相对均匀，可直接用于建模"
        else:
            target_suggestion = "目标变量为类别型，适合分类任务"
        
        report += f"""### 5.3 目标变量分布

![Target Distribution](data:image/png;base64,{analysis['visualizations']['target_distribution']})

**图表解读**:
{target_suggestion}

**对建模的指导**:
- **回归任务**: 检查目标变量是否需要进行变换（如对数变换）
- **分类任务**: 检查类别是否平衡，不平衡需要采用相应策略
- **异常目标值**: 检查是否存在极端值需要处理

---

"""
    
    # 4. 相关性矩阵
    if "correlation_matrix" in analysis['visualizations']:
        high_corr_pairs = ["- 请查看热力图中的深色区域（相关性>0.8或<-0.8）"]
        
        report += f"""### 5.4 特征相关性矩阵

![Correlation Matrix](data:image/png;base64,{analysis['visualizations']['correlation_matrix']})

**图表解读**:
上述热力图展示了数值特征之间的相关性：
{chr(10).join(high_corr_pairs)}

**对特征工程的指导**:
- **高相关性特征** (>0.8): 存在多重共线性，考虑删除或PCA降维
- **与目标变量高相关**: 重要特征，优先保留
- **负相关特征**: 同样重要，不要误删
- **交互特征**: 高相关性特征可考虑创建比率或差值特征

---

"""
    
    # 6. 分析结论与建议
    report += """## 6. 数据清洗方案建议

基于以上数据分析，建议按以下优先级进行数据清洗：

### 6.1 缺失值处理
"""
    
    if missing_cols:
        high_missing = {k: v for k, v in missing_cols.items() if analysis['missing_pct'][k] > 50}
        med_missing = {k: v for k, v in missing_cols.items() if 10 <= analysis['missing_pct'][k] <= 50}
        low_missing = {k: v for k, v in missing_cols.items() if analysis['missing_pct'][k] < 10}
        
        if high_missing:
            report += f"**高缺失率(>50%) - 建议删除**: {', '.join(list(high_missing.keys())[:3])}\n\n"
        if med_missing:
            report += f"**中等缺失率(10-50%) - 需要填充策略**: {', '.join(list(med_missing.keys())[:3])}\n\n"
        if low_missing:
            report += f"**低缺失率(<10%) - 简单填充**: {', '.join(list(low_missing.keys())[:3])}\n\n"
    else:
        report += "✅ 数据完整，无需缺失值处理\n\n"
    
    report += """### 6.2 异常值处理
- 使用IQR方法检测异常值
- 轻微异常: Winsorization（分位数截断）
- 严重异常: 标记为缺失后填充
- 业务相关异常: 保留并创建异常标记特征

### 6.3 特征变换
- 右偏分布特征: 对数变换 (log1p)
- 正态分布特征: z-score标准化
- 高基数类别特征: 目标编码或频数编码
- 低基数类别特征: One-hot编码

### 6.4 数据泄漏预防
- [ ] 训练/测试划分在特征工程之前完成
- [ ] 目标编码仅使用训练集统计信息
- [ ] 无目标变量派生特征

---

**请回复 "确认分析结果" 继续生成详细的数据清洗代码，或提出您的修改意见**
"""
    
    return report
