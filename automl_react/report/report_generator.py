"""
可视化分析报告生成器模块

生成包含数据分布、特征重要性、模型评估指标的可视化分析报告
"""

import json
import os
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
        """加载 data-analysis skill 的 chart-selection.md"""
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
        # 读取各阶段结果
        problem_definition = self._load_problem_definition()
        splitting_result = self._load_splitting_result()
        cleaning_result = self._load_cleaning_result()
        exploration_result = self._load_exploration_result()
        feature_result = self._load_feature_result()
        feature_evaluation_result = self._load_feature_evaluation_result()
        model_result = self._load_model_result()
        training_summary = self._load_training_summary()
        evaluation_result = self._load_evaluation_result()

        # 生成可视化图表
        chart_paths = self._generate_charts(training_summary, evaluation_result, task_type)

        # 生成报告
        report = self._build_report(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,
            problem_definition=problem_definition,
            splitting_result=splitting_result,
            cleaning_result=cleaning_result,
            exploration_result=exploration_result,
            feature_result=feature_result,
            feature_evaluation_result=feature_evaluation_result,
            model_result=model_result,
            training_summary=training_summary,
            evaluation_result=evaluation_result,
            chart_paths=chart_paths,
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

    # ==================== Data Loaders ====================

    def _load_json_asset(self, asset_type: str, filename: str) -> Dict[str, Any]:
        """通用 JSON 资产加载"""
        data = self.asset_manager.read_asset(asset_type, filename)
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                pass
        return {}

    def _load_cleaning_result(self) -> Dict[str, Any]:
        return self._load_json_asset("cleaning", "cleaning_result.json")

    def _load_feature_result(self) -> Dict[str, Any]:
        return self._load_json_asset("features", "feature_engineering_result.json")

    def _load_model_result(self) -> Dict[str, Any]:
        return self._load_json_asset("models", "model_training_result.json")

    def _load_training_summary(self) -> Dict[str, Any]:
        return self._load_json_asset("models", "training_summary.json")

    def _load_feature_evaluation_result(self) -> Dict[str, Any]:
        return self._load_json_asset("features", "feature_evaluation_result.json")

    def _load_evaluation_result(self) -> Dict[str, Any]:
        return self._load_json_asset("reports", "evaluation.json")

    def _load_problem_definition(self) -> Dict[str, Any]:
        return self._load_json_asset("analysis", "problem_definition.json")

    def _load_splitting_result(self) -> Dict[str, Any]:
        return self._load_json_asset("data", "splitting_result.json")

    def _load_exploration_result(self) -> Optional[str]:
        """加载探索性分析结果（Markdown 文本）"""
        return self.asset_manager.read_asset("exploration", "data_exploration_result.md")

    # ==================== Chart Generation ====================

    def _generate_charts(
        self,
        training_summary: Dict,
        evaluation_result: Dict,
        task_type: str,
    ) -> Dict[str, str]:
        """生成可视化图表并保存为 PNG，返回 {chart_name: relative_path}"""
        chart_paths = {}
        charts_dir = self.asset_manager.session_dir / "reports" / "charts"
        os.makedirs(charts_dir, exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return chart_paths

        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # 1. 特征重要性图
        feature_importances = training_summary.get("feature_importances", {})
        feature_names = training_summary.get("selected_feature_names", [])
        if feature_importances and feature_names:
            chart_paths["feature_importance"] = self._chart_feature_importance(
                plt, feature_importances, feature_names, charts_dir
            )

        # 2. 指标对比图（训练 vs 测试）
        train_metrics = training_summary.get("metrics", {})
        eval_metrics = evaluation_result.get("metrics", {})
        if train_metrics and eval_metrics:
            chart_paths["metrics_comparison"] = self._chart_metrics_comparison(
                plt, train_metrics, eval_metrics, charts_dir
            )

        return chart_paths

    def _chart_feature_importance(self, plt, importances, feature_names, charts_dir) -> str:
        """生成特征重要性图"""
        try:
            # importances 可以是 dict {name: score} 或 list
            if isinstance(importances, dict):
                names = list(importances.keys())[:20]
                values = [importances[n] for n in names]
            elif isinstance(importances, list) and feature_names:
                pairs = sorted(zip(feature_names, importances), key=lambda x: abs(x[1]), reverse=True)[:20]
                names = [p[0] for p in pairs]
                values = [p[1] for p in pairs]
            else:
                return ""

            fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.3)))
            ax.barh(range(len(names)), values)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel("Importance")
            ax.set_title("Top Feature Importance")
            ax.invert_yaxis()
            fig.tight_layout()
            path = str(charts_dir / "feature_importance.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)
            return path
        except Exception:
            return ""

    def _chart_metrics_comparison(self, plt, train_metrics, eval_metrics, charts_dir) -> str:
        """生成训练 vs 测试指标对比图"""
        try:
            common_keys = [k for k in train_metrics if k in eval_metrics
                          and isinstance(train_metrics[k], (int, float))
                          and isinstance(eval_metrics[k], (int, float))]
            if not common_keys:
                return ""

            import numpy as np
            x = np.arange(len(common_keys))
            width = 0.35
            fig, ax = plt.subplots(figsize=(max(6, len(common_keys) * 1.5), 5))
            ax.bar(x - width/2, [train_metrics[k] for k in common_keys], width, label="Train")
            ax.bar(x + width/2, [eval_metrics[k] for k in common_keys], width, label="Test")
            ax.set_xticks(x)
            ax.set_xticklabels(common_keys, rotation=45, ha="right")
            ax.set_title("Train vs Test Metrics")
            ax.legend()
            fig.tight_layout()
            path = str(charts_dir / "metrics_comparison.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)
            return path
        except Exception:
            return ""

    # ==================== Report Builder ====================

    def _build_report(
        self,
        data_path: str,
        target_column: str,
        task_type: str,
        problem_definition: Dict,
        splitting_result: Dict,
        cleaning_result: Dict,
        exploration_result: Optional[str],
        feature_result: Dict,
        feature_evaluation_result: Dict,
        model_result: Dict,
        training_summary: Dict,
        evaluation_result: Dict,
        chart_paths: Dict[str, str],
    ) -> str:
        """构建完整的 Markdown 报告"""
        lines = []

        # 报告标题
        lines.extend([
            "# AutoML 建模流程报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**会话ID**: {self.session_id}",
            "",
            "---",
            ""
        ])

        section = 1

        # 1. 项目概述
        lines.extend([f"## {section}. 项目概述", ""])
        lines.extend([
            f"- **数据路径**: `{data_path}`",
            f"- **目标列**: `{target_column}`",
            f"- **任务类型**: {task_type}",
        ])
        if problem_definition:
            pd_data = problem_definition
            if pd_data.get("task_type"):
                lines.append(f"- **确认任务类型**: {pd_data['task_type']}")
            if pd_data.get("primary_metric"):
                lines.append(f"- **主评估指标**: {pd_data['primary_metric']}")
            if pd_data.get("secondary_metrics"):
                lines.append(f"- **辅助指标**: {', '.join(pd_data['secondary_metrics'])}")
            if pd_data.get("business_constraints"):
                lines.append(f"- **业务约束**: {pd_data['business_constraints']}")
            if pd_data.get("success_criteria"):
                lines.append(f"- **成功标准**: {pd_data['success_criteria']}")
        lines.append("")
        section += 1

        # 2. 数据切分
        lines.extend([f"## {section}. 数据切分", ""])
        if splitting_result:
            lines.extend([
                f"- **切分策略**: {splitting_result.get('split_strategy', 'N/A')}",
                f"- **随机种子**: {splitting_result.get('random_seed', 'N/A')}",
            ])
            split_paths = splitting_result.get("split_paths", {})
            split_sizes = splitting_result.get("split_sizes", {})
            if split_sizes:
                lines.extend([
                    f"- **训练集样本数**: {split_sizes.get('train', 'N/A')}",
                    f"- **验证集样本数**: {split_sizes.get('valid', 'N/A')}",
                    f"- **测试集样本数**: {split_sizes.get('test', 'N/A')}",
                ])
            elif split_paths:
                lines.extend([
                    f"- **训练集**: `{split_paths.get('train_raw_path', 'N/A')}`",
                    f"- **验证集**: `{split_paths.get('valid_raw_path', 'N/A')}`",
                    f"- **测试集**: `{split_paths.get('test_raw_path', 'N/A')}`",
                ])
        else:
            lines.append("*数据切分结果未找到*")
        lines.append("")
        section += 1

        # 3. 数据清洗
        lines.extend([f"## {section}. 数据清洗", ""])
        if cleaning_result.get("success"):
            lines.extend([
                "**清洗状态**: 成功",
                f"- **原始数据**: {cleaning_result.get('original_path', 'N/A')}",
                f"- **清洗后数据**: {cleaning_result.get('cleaned_data_path', 'N/A')}",
            ])
        else:
            lines.append("*数据清洗未执行或失败*")
        lines.append("")
        section += 1

        # 4. 数据探索分析
        lines.extend([f"## {section}. 数据探索分析", ""])
        if exploration_result:
            # 截取探索报告的关键部分（避免报告过长）
            exploration_lines = exploration_result.strip().split("\n")
            if len(exploration_lines) > 80:
                lines.append("*以下为探索性分析的关键发现摘要：*")
                lines.append("")
                lines.extend(exploration_lines[:80])
                lines.append("")
                lines.append(f"*... 完整探索报告共 {len(exploration_lines)} 行，详见 exploration/data_exploration_result.md*")
            else:
                lines.extend(exploration_lines)
        else:
            lines.append("*数据探索分析未执行*")
        lines.append("")
        section += 1

        # 5. 特征工程
        lines.extend([f"## {section}. 特征工程", ""])
        if feature_result.get("success"):
            new_features = feature_result.get('new_features', [])
            lines.extend([
                "**特征工程状态**: 成功",
                f"- **特征数据路径**: {feature_result.get('features_data_path', 'N/A')}",
                f"- **原始特征数**: {feature_result.get('original_features', 'N/A')}",
                f"- **新生成特征数**: {len(new_features)}",
                f"- **总特征数**: {feature_result.get('total_features', 'N/A')}",
            ])
            if feature_evaluation_result.get("success"):
                lines.extend([
                    f"- **特征评估报告**: {feature_evaluation_result.get('metrics_report_path', 'N/A')}",
                ])
            if new_features:
                lines.extend(["", "**新生成的特征**:", ""])
                for feature in new_features[:10]:
                    lines.append(f"- {feature}")
                if len(new_features) > 10:
                    lines.append(f"- ... 等共 {len(new_features)} 个特征")
        else:
            lines.append("*特征工程未执行或失败*")
        lines.append("")
        section += 1

        # 6. 模型训练
        lines.extend([f"## {section}. 模型训练", ""])
        if model_result.get("success"):
            model_metrics = model_result.get('metrics', {})
            lines.extend([
                "**训练状态**: 成功",
                f"- **模型路径**: `{model_result.get('model_path', 'N/A')}`",
            ])
            if training_summary:
                lines.extend([
                    f"- **最佳模型**: {training_summary.get('best_model', 'N/A')}",
                    f"- **目标变换**: {training_summary.get('target_transform', 'None')}",
                    f"- **入模特征数**: {len(training_summary.get('selected_feature_names', []))}",
                ])
            if model_metrics:
                lines.extend(["", "### 训练阶段指标", "", "| 指标 | 数值 |", "|------|------|"])
                for k, v in model_metrics.items():
                    val = f"{v:.4f}" if isinstance(v, float) else str(v)
                    lines.append(f"| {k} | {val} |")
        else:
            error = model_result.get('error', '未执行')
            lines.append(f"*模型训练未执行或失败: {error}*")
        lines.append("")
        section += 1

        # 7. 模型评估
        lines.extend([f"## {section}. 模型评估", ""])
        if evaluation_result.get("success"):
            metrics = evaluation_result.get('metrics', {})
            lines.extend([
                "**评估状态**: 成功",
                "",
                "### 评估指标",
                "",
                "| 指标 | 数值 |",
                "|------|------|"
            ])
            for k, v in metrics.items():
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                lines.append(f"| {k} | {val} |")
        else:
            lines.append("*模型评估未执行或失败*")
        lines.append("")
        section += 1

        # 8. 可视化分析
        lines.extend([f"## {section}. 可视化分析", ""])
        if chart_paths:
            for chart_name, chart_path in chart_paths.items():
                if chart_path:
                    lines.extend([
                        f"### {chart_name.replace('_', ' ').title()}",
                        "",
                        f"![{chart_name}]({chart_path})",
                        "",
                    ])
        else:
            lines.append("*可视化图表未能生成（可能缺少 matplotlib 或无足够数据）*")
        lines.append("")
        section += 1

        # 9. 结论与建议
        lines.extend([f"## {section}. 结论与建议", ""])
        lines.extend(self._generate_conclusions(
            task_type, evaluation_result, training_summary, model_result
        ))
        lines.extend(["", "---", "", "*报告由 AutoML 系统自动生成*"])

        return "\n".join(lines)

    def _generate_conclusions(
        self,
        task_type: str,
        evaluation_result: Dict,
        training_summary: Dict,
        model_result: Dict,
    ) -> List[str]:
        """基于实际结果生成结论与建议"""
        lines = ["### 模型性能总结", ""]

        eval_metrics = evaluation_result.get("metrics", {})
        train_metrics = training_summary.get("metrics", {}) or model_result.get("metrics", {})

        if not eval_metrics:
            lines.append("*评估结果不可用，无法生成定量结论。*")
            return lines

        # 定量总结
        if task_type == "classification":
            acc = eval_metrics.get("accuracy")
            f1 = eval_metrics.get("f1") or eval_metrics.get("f1_score")
            if acc is not None:
                lines.append(f"- **测试集准确率**: {acc:.2%}")
            if f1 is not None:
                lines.append(f"- **测试集 F1**: {f1:.4f}")
        elif task_type == "regression":
            r2 = eval_metrics.get("r2")
            rmse = eval_metrics.get("rmse")
            mae = eval_metrics.get("mae")
            if r2 is not None:
                lines.append(f"- **测试集 R\u00b2**: {r2:.4f}")
            if rmse is not None:
                lines.append(f"- **测试集 RMSE**: {rmse:.4f}")
            if mae is not None:
                lines.append(f"- **测试集 MAE**: {mae:.4f}")

        # 过拟合检查
        if train_metrics and eval_metrics:
            lines.extend(["", "### 过拟合分析", ""])
            overfit_detected = False
            for key in train_metrics:
                if key in eval_metrics and isinstance(train_metrics[key], (int, float)) and isinstance(eval_metrics[key], (int, float)):
                    train_val = train_metrics[key]
                    eval_val = eval_metrics[key]
                    if train_val != 0:
                        gap = abs(train_val - eval_val) / abs(train_val)
                        if gap > 0.1:
                            lines.append(
                                f"- **{key}**: 训练 {train_val:.4f} vs 测试 {eval_val:.4f}"
                                f" (差距 {gap:.1%}，可能过拟合)"
                            )
                            overfit_detected = True
            if not overfit_detected:
                lines.append("- 训练与测试指标差距在合理范围内，未检测到明显过拟合。")

        # 改进建议
        lines.extend(["", "### 改进建议", ""])
        suggestions = []
        if task_type == "classification":
            acc = eval_metrics.get("accuracy", 1.0)
            if acc < 0.7:
                suggestions.append("当前准确率较低，建议检查特征质量或尝试更复杂的模型")
            elif acc < 0.85:
                suggestions.append("可通过超参调优或特征选择进一步提升性能")
        elif task_type == "regression":
            r2 = eval_metrics.get("r2", 1.0)
            if r2 < 0.5:
                suggestions.append("R\u00b2 较低，建议检查特征工程或考虑更复杂的非线性模型")
            elif r2 < 0.8:
                suggestions.append("可通过特征组合或超参调优提升拟合效果")

        best_model = training_summary.get("best_model", "")
        if best_model:
            suggestions.append(f"当前最佳模型为 {best_model}，可尝试集成方法进一步提升")

        if not suggestions:
            suggestions.append("模型整体表现良好，可考虑部署或进行更多场景验证")

        for i, s in enumerate(suggestions, 1):
            lines.append(f"{i}. {s}")

        return lines

    # ==================== Summary JSON ====================

    def generate_summary_json(
        self,
        data_path: str,
        target_column: str,
        task_type: str,
    ) -> Dict[str, Any]:
        """生成结构化 JSON 摘要并保存"""
        problem_def = self._load_problem_definition()
        splitting = self._load_splitting_result()
        cleaning = self._load_cleaning_result()
        feature = self._load_feature_result()
        model = self._load_model_result()
        training = self._load_training_summary()
        evaluation = self._load_evaluation_result()

        summary = {
            "session_id": self.session_id,
            "generated_at": datetime.now().isoformat(),
            "problem_definition": {
                "task_type": problem_def.get("task_type", task_type),
                "target_column": problem_def.get("target_column", target_column),
                "primary_metric": problem_def.get("primary_metric", ""),
            },
            "data_summary": {
                "original_data_path": data_path,
                "split_strategy": splitting.get("split_strategy", "N/A"),
                "split_sizes": splitting.get("split_sizes", {}),
            },
            "cleaning_summary": {
                "success": cleaning.get("success", False),
                "cleaned_data_path": cleaning.get("cleaned_data_path", ""),
            },
            "feature_summary": {
                "success": feature.get("success", False),
                "original_features": feature.get("original_features", 0),
                "new_features_count": len(feature.get("new_features", [])),
                "total_features": feature.get("total_features", 0),
            },
            "model_summary": {
                "success": model.get("success", False),
                "best_model": training.get("best_model", ""),
                "target_transform": training.get("target_transform", "None"),
                "selected_features_count": len(training.get("selected_feature_names", [])),
                "training_metrics": training.get("metrics", {}),
            },
            "evaluation_summary": {
                "success": evaluation.get("success", False),
                "metrics": evaluation.get("metrics", {}),
            },
            "pipeline_available": bool(
                self.asset_manager.get_asset("code", "pipeline.py")
            ),
        }

        # 保存
        self.asset_manager.save_data(
            data=json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            filename="summary.json",
            asset_type="reports",
            metadata={"stage": "report", "timestamp": datetime.now().isoformat()},
        )

        return summary

    # ==================== Accessors ====================

    def get_report(self) -> Optional[str]:
        """获取已生成的报告"""
        return self.asset_manager.read_asset("reports", "modeling_report.md")

    def export_to_html(self, markdown_report: str) -> str:
        """将 Markdown 报告转换为 HTML"""
        try:
            import markdown

            html = markdown.markdown(
                markdown_report,
                extensions=['tables', 'fenced_code', 'toc']
            )

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
        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
    </style>
</head>
<body>
{html}
</body>
</html>'''

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
            return f"<pre>{markdown_report}</pre>"
