"""
可视化分析报告生成器模块

生成包含数据分布、特征重要性、模型评估指标的可视化分析报告
参考 data-analysis skill 的 chart-selection.md
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from ..skills_loader import get_skill_loader
from ..assets import get_asset_manager


class ReportGenerator:
    """
    报告生成器

    生成 Markdown 格式的可视化分析报告

    Attributes:
        session_id: 会话ID
        skill_loader: Skill 加载器
        asset_manager: 资产管理器
    """

    def __init__(self, session_id: str = None):
        self.session_id = session_id or "default"
        self.skill_loader = get_skill_loader()
        self.asset_manager = get_asset_manager(session_id=self.session_id)

    def load_chart_selection_guide(self) -> str:
        """
        加载 data-analysis skill 的 chart-selection.md

        Returns:
            图表选择指南内容
        """
        return self.skill_loader.get_skill_reference(
            "data-analysis-1.0.2",
            "chart-selection.md"
        ) or ""

    def generate_report(
        self,
        data_path: str,
        target_column: str,
        task_type: str = "classification"
    ) -> str:
        """
        生成可视化分析报告

        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型

        Returns:
            Markdown 格式的报告
        """
        # 加载图表选择指南
        chart_guide = self.load_chart_selection_guide()

        # 读取各阶段结果
        cleaning_result = self._load_cleaning_result()
        feature_result = self._load_feature_result()
        model_result = self._load_model_result()
        evaluation_result = self._load_evaluation_result()

        # 生成报告
        report = self._build_report(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,
            cleaning_result=cleaning_result,
            feature_result=feature_result,
            model_result=model_result,
            evaluation_result=evaluation_result,
            chart_guide=chart_guide
        )

        # 保存报告
        self.asset_manager.save_report(
            report=report,
            filename="modeling_report.md",
            metadata={
                "stage": "report",
                "data_path": data_path,
                "target_column": target_column,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat()
            }
        )

        return report

    def _load_cleaning_result(self) -> Dict[str, Any]:
        """加载数据清洗结果"""
        data = self.asset_manager.read_asset("cleaned_data", "cleaning_result.json")
        if data:
            return json.loads(data)
        return {}

    def _load_feature_result(self) -> Dict[str, Any]:
        """加载特征工程结果"""
        data = self.asset_manager.read_asset("features", "feature_engineering_result.json")
        if data:
            return json.loads(data)
        return {}

    def _load_model_result(self) -> Dict[str, Any]:
        """加载模型训练结果"""
        data = self.asset_manager.read_asset("models", "model_training_result.json")
        if data:
            return json.loads(data)
        return {}

    def _load_evaluation_result(self) -> Dict[str, Any]:
        """加载评估结果"""
        data = self.asset_manager.read_asset("reports", "evaluation.json")
        if data:
            return json.loads(data)
        return {}

    def _build_report(
        self,
        data_path: str,
        target_column: str,
        task_type: str,
        cleaning_result: Dict,
        feature_result: Dict,
        model_result: Dict,
        evaluation_result: Dict,
        chart_guide: str
    ) -> str:
        """
        构建报告

        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型
            cleaning_result: 清洗结果
            feature_result: 特征工程结果
            model_result: 模型训练结果
            evaluation_result: 评估结果
            chart_guide: 图表选择指南

        Returns:
            Markdown 格式的报告
        """
        report_lines = []

        # 报告标题
        report_lines.extend([
            "# AutoML 建模流程报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**会话ID**: {self.session_id}",
            "",
            "---",
            ""
        ])

        # 1. 数据概览
        report_lines.extend([
            "## 1. 数据概览",
            "",
            f"- **数据路径**: `{data_path}`",
            f"- **目标列**: `{target_column}`",
            f"- **任务类型**: {task_type}",
            ""
        ])

        # 2. 数据清洗
        report_lines.extend([
            "## 2. 数据清洗",
            ""
        ])

        if cleaning_result.get("success"):
            report_lines.extend([
                "✅ **清洗状态**: 成功",
                f"- **原始数据**: {cleaning_result.get('original_path', 'N/A')}",
                f"- **清洗后数据**: {cleaning_result.get('cleaned_data_path', 'N/A')}",
                ""
            ])
        else:
            report_lines.extend([
                "⚠️ **清洗状态**: 跳过或失败",
                ""
            ])

        # 3. 特征工程
        report_lines.extend([
            "## 3. 特征工程",
            ""
        ])

        if feature_result.get("success"):
            new_features = feature_result.get('new_features', [])
            report_lines.extend([
                "✅ **特征工程状态**: 成功",
                f"- **原始特征数**: {feature_result.get('original_features', 'N/A')}",
                f"- **新生成特征数**: {len(new_features)}",
                f"- **总特征数**: {feature_result.get('total_features', 'N/A')}",
                ""
            ])

            if new_features:
                report_lines.extend([
                    "**新生成的特征**:",
                    ""
                ])
                for feature in new_features[:10]:  # 最多显示10个
                    report_lines.append(f"- {feature}")
                if len(new_features) > 10:
                    report_lines.append(f"- ... 等共 {len(new_features)} 个特征")
                report_lines.append("")
        else:
            report_lines.extend([
                "⚠️ **特征工程状态**: 跳过或失败",
                ""
            ])

        # 4. 模型训练
        report_lines.extend([
            "## 4. 模型训练",
            ""
        ])

        if model_result.get("success"):
            report_lines.extend([
                "✅ **训练状态**: 成功",
                f"- **任务类型**: {model_result.get('task_type', 'N/A')}",
                f"- **模型路径**: `{model_result.get('model_path', 'N/A')}`",
                ""
            ])
        else:
            report_lines.extend([
                "❌ **训练状态**: 失败",
                f"- **错误**: {model_result.get('error', '未知错误')}",
                ""
            ])

        # 5. 模型评估
        report_lines.extend([
            "## 5. 模型评估",
            ""
        ])

        if evaluation_result.get("success"):
            metrics = evaluation_result.get('metrics', {})
            report_lines.extend([
                "✅ **评估状态**: 成功",
                "",
                "### 评估指标",
                "",
                "| 指标 | 数值 |",
                "|------|------|"
            ])

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    report_lines.append(f"| {metric_name} | {metric_value:.4f} |")
                else:
                    report_lines.append(f"| {metric_name} | {metric_value} |")

            report_lines.append("")
        else:
            report_lines.extend([
                "⚠️ **评估状态**: 未执行或失败",
                ""
            ])

        # 6. 可视化图表
        report_lines.extend([
            "## 6. 可视化图表",
            "",
            "### 6.1 数据分布",
            "",
            "```python",
            "# 数据分布可视化代码",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            f"df = pd.read_csv('{data_path}')",
            "",
            "# 数值特征分布",
            "numeric_cols = df.select_dtypes(include=[np.number]).columns",
            "for col in numeric_cols:",
            "    plt.figure(figsize=(10, 4))",
            "    plt.subplot(1, 2, 1)",
            "    sns.histplot(df[col], kde=True)",
            "    plt.title(f'{col} 分布')",
            "    plt.subplot(1, 2, 2)",
            "    sns.boxplot(y=df[col])",
            "    plt.title(f'{col} 箱线图')",
            "    plt.tight_layout()",
            "    plt.savefig(f'distribution_{col}.png')",
            "    plt.close()",
            "```",
            "",
            "### 6.2 特征重要性",
            "",
            "```python",
            "# 特征重要性可视化",
            "if hasattr(model, 'feature_importances_'):",
            "    importances = model.feature_importances_",
            "    feature_names = X.columns",
            "    ",
            "    plt.figure(figsize=(10, 6))",
            "    indices = np.argsort(importances)[::-1][:20]  # Top 20",
            "    plt.bar(range(len(indices)), importances[indices])",
            "    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)",
            "    plt.title('Top 20 特征重要性')",
            "    plt.tight_layout()",
            "    plt.savefig('feature_importance.png')",
            "    plt.close()",
            "```",
            "",
            "### 6.3 模型评估图表",
            "",
            "```python",
            "# 混淆矩阵（分类任务）",
            "if task_type == 'classification':",
            "    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay",
            "    cm = confusion_matrix(y_test, y_pred)",
            "    disp = ConfusionMatrixDisplay(confusion_matrix=cm)",
            "    disp.plot()",
            "    plt.title('混淆矩阵')",
            "    plt.savefig('confusion_matrix.png')",
            "    plt.close()",
            "",
            "# ROC 曲线（二分类）",
            "if task_type == 'classification' and len(np.unique(y)) == 2:",
            "    from sklearn.metrics import roc_curve, auc",
            "    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])",
            "    roc_auc = auc(fpr, tpr)",
            "    ",
            "    plt.figure()",
            "    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')",
            "    plt.plot([0, 1], [0, 1], 'k--')",
            "    plt.xlabel('False Positive Rate')",
            "    plt.ylabel('True Positive Rate')",
            "    plt.title('ROC 曲线')",
            "    plt.legend()",
            "    plt.savefig('roc_curve.png')",
            "    plt.close()",
            "",
            "# 残差图（回归任务）",
            "if task_type == 'regression':",
            "    residuals = y_test - y_pred",
            "    plt.figure(figsize=(10, 4))",
            "    plt.subplot(1, 2, 1)",
            "    plt.scatter(y_pred, residuals, alpha=0.5)",
            "    plt.axhline(y=0, color='r', linestyle='--')",
            "    plt.xlabel('预测值')",
            "    plt.ylabel('残差')",
            "    plt.title('残差图')",
            "    plt.subplot(1, 2, 2)",
            "    sns.histplot(residuals, kde=True)",
            "    plt.title('残差分布')",
            "    plt.tight_layout()",
            "    plt.savefig('residuals.png')",
            "    plt.close()",
            "```",
            ""
        ])

        # 7. 结论与建议
        report_lines.extend([
            "## 7. 结论与建议",
            "",
            "### 7.1 模型性能总结",
            ""
        ])

        if evaluation_result.get("success"):
            metrics = evaluation_result.get('metrics', {})
            if task_type == "classification":
                accuracy = metrics.get('accuracy', 0)
                report_lines.extend([
                    f"- **准确率**: {accuracy:.2%}",
                    ""
                ])
            else:
                r2 = metrics.get('r2', 0)
                report_lines.extend([
                    f"- **R² 分数**: {r2:.4f}",
                    ""
                ])

        report_lines.extend([
            "### 7.2 改进建议",
            "",
            "1. **数据层面**: 考虑收集更多数据或进行数据增强",
            "2. **特征层面**: 尝试更多特征组合和特征选择方法",
            "3. **模型层面**: 尝试集成模型或深度学习模型",
            "4. **调优层面**: 进行超参数调优以提升模型性能",
            "",
            "---",
            "",
            "*报告由 AutoML 系统自动生成*"
        ])

        return "\n".join(report_lines)

    def get_report(self) -> Optional[str]:
        """
        获取已生成的报告

        Returns:
            报告内容，如果不存在返回 None
        """
        return self.asset_manager.read_asset("reports", "modeling_report.md")

    def export_to_html(self, markdown_report: str) -> str:
        """
        将 Markdown 报告转换为 HTML

        Args:
            markdown_report: Markdown 格式的报告

        Returns:
            HTML 格式的报告
        """
        try:
            import markdown

            html = markdown.markdown(
                markdown_report,
                extensions=['tables', 'fenced_code', 'toc']
            )

            # 添加样式
            styled_html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AutoML 建模报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        pre code {{ background-color: transparent; padding: 0; }}
        blockquote {{ border-left: 4px solid #3498db; margin: 0; padding-left: 20px; color: #666; }}
    </style>
</head>
<body>
{html}
</body>
</html>'''

            # 保存 HTML 报告
            self.asset_manager.save_report(
                report=styled_html,
                filename="modeling_report.html",
                metadata={
                    "stage": "report",
                    "format": "html",
                    "timestamp": datetime.now().isoformat()
                }
            )

            return styled_html

        except ImportError:
            # 如果没有 markdown 库，返回简单 HTML
            return f"<pre>{markdown_report}</pre>"
