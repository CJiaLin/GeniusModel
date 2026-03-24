"""
特征工程 Agent 模块

实现特征工程思路生成、用户确认、代码生成与执行
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..core.react_agent import ReActAgent, ConfirmationRequired
from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
from ..skills_loader import get_skill_loader
from ..config import get_config_loader
from ..logger.llm_logger import LLMLogger


class FeatureEngineeringAgent(ReActAgent):
    """
    特征工程 Agent

    基于 ReAct 架构的特征工程 Agent，支持：
    1. 特征分析
    2. 特征工程思路生成（参考 afrexai-ml-engineering skill）
    3. 用户确认流程
    4. 代码生成与执行
    5. 特征数据保存

    Attributes:
        session_id: 会话ID
        data_path: 数据文件路径
        target_column: 目标列名
        task_type: 任务类型
        feature_plan: 特征工程方案
        feature_code: 特征工程代码
    """

    def __init__(self, llm: Any = None, session_id: str = None, verbose: bool = False):
        super().__init__(llm=llm, session_id=session_id, max_iterations=10, verbose=verbose)
        self.data_path: Optional[str] = None
        self.target_column: Optional[str] = None
        self.task_type: str = "classification"
        self.data_info: Optional[Dict] = None
        self.feature_plan: Optional[str] = None
        self.feature_code: Optional[str] = None
        self.skill_loader = get_skill_loader()
        self.config_loader = get_config_loader()
        self.logger = LLMLogger(session_id=session_id)

    def _register_default_tools(self):
        """注册默认工具"""
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        try:
            return self.config_loader.get_prompt("feature_engineering", "system_prompt")
        except KeyError:
            return """你是一位专业的特征工程专家。

你的职责：
1. 分析数据特征
2. 生成特征工程方案
3. 编写特征工程代码
4. 执行特征工程并验证结果

**重要原则**：
- 必须使用用户上传的实际数据文件进行特征工程
- 禁止使用示例数据或虚构数据
- 所有特征工程方案必须基于实际数据的列名和特征
- 特征工程代码必须针对实际数据的列名和特征

请基于数据特征和最佳实践，生成详细的特征工程方案。"""

    def analyze_features(self, data_path: str, target_column: str) -> Dict[str, Any]:
        """
        分析特征

        Args:
            data_path: 数据文件路径
            target_column: 目标列名

        Returns:
            特征分析结果
        """
        self.data_path = data_path
        self.target_column = target_column

        # 构建分析提示词
        user_input = f"请分析数据文件的特征: {data_path}, 目标列: {target_column}"

        result = self.run(user_input, stage="feature_analysis")

        # 保存数据信息
        self.data_info = result.get("data_info", {})

        return result

    def generate_feature_plan(
        self,
        data_path: str = None,
        target_column: str = None,
        task_type: str = "classification",
        analysis_result: str = None,
        cleaning_result: str = None,
        cleaned_data_path: str = None,
        task_description: str = ""
    ) -> str:
        """
        生成特征工程思路

        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型
            analysis_result: 数据分析报告（可选）
            cleaning_result: 数据清洗报告（可选）
            cleaned_data_path: 清洗后的数据路径（可选，如果提供则使用此路径）
            task_description: 用户的建模背景和要求

        Returns:
            特征工程方案（Markdown 格式）
        """
        # 优先使用清洗后的数据路径
        path = cleaned_data_path or data_path or self.data_path
        target = target_column or self.target_column

        if not path or not target:
            raise ValueError("请提供数据文件路径和目标列名")

        self.data_path = path
        self.target_column = target
        self.task_type = task_type

        # 构建上下文摘要
        context_summary = ""
        
        # 添加用户的建模背景
        if task_description:
            context_summary += f"""
## 用户建模背景和要求

{task_description}

**重要：请在特征工程方案中充分考虑用户的建模背景和要求。**

"""
        
        # 如果有分析报告，添加到上下文
        if analysis_result:
            context_summary += f"""
## 数据分析报告（来自数据分析阶段）

{analysis_result[:3000]}

"""
        
        # 如果有清洗报告，添加到上下文
        if cleaning_result:
            context_summary += f"""
## 数据清洗报告（来自数据清洗阶段）

{cleaning_result[:2000]}

"""

        # 加载数据基本信息
        import pandas as pd

        try:
            df = pd.read_csv(path)

            # 收集数据基本信息
            self.data_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "numeric_columns": list(df.select_dtypes(include=['int64', 'float64']).columns),
                "categorical_columns": list(df.select_dtypes(include=['object']).columns),
                "target_dtype": str(df[target].dtype) if target in df.columns else "unknown",
                "target_unique": df[target].nunique() if target in df.columns else 0
            }

            # 构建数据摘要
            data_summary = f"""
{context_summary}
## 当前数据基本信息

- **数据路径**: {path}
- **数据形状**: {self.data_info['shape'][0]} 行 × {self.data_info['shape'][1]} 列
- **目标列**: {target}
- **目标列类型**: {self.data_info['target_dtype']}
- **目标列唯一值数量**: {self.data_info['target_unique']}
- **任务类型**: {task_type}

## 数值列

{', '.join(self.data_info['numeric_columns'][:20])}

## 分类列

{', '.join(self.data_info['categorical_columns'][:20])}

重要：请基于上述实际数据列名和前序阶段的分析结果生成方案。
"""

        except Exception as e:
            data_summary = f"无法加载数据文件: {path}\n错误: {str(e)}"

        # 加载 afrexai-ml-engineering skill 的 Phase 2
        skill_content = self.skill_loader.get_skill_content("afrexai-ml-engineering-1.0.0")

        # 提取 Phase 2 相关内容
        phase2_content = ""
        if skill_content:
            import re
            phase2_match = re.search(
                r'##?\s*Phase\s*2[:：]\s*Data Engineering.*?\n(.*?)(?=##?\s*Phase\s*3|\Z)',
                skill_content,
                re.DOTALL | re.IGNORECASE
            )
            if phase2_match:
                phase2_content = phase2_match.group(1)
            else:
                phase2_content = skill_content[:3000]

        # 从配置加载 Prompt
        try:
            prompt_template = self.config_loader.get_prompt("feature_engineering", "plan_generation")
        except KeyError:
            prompt_template = """请为以下数据生成详细的特征工程方案：

{data_summary}

任务类型: {task_type}

{skill_content}

请生成 Markdown 格式的特征工程方案，包括：
1. 现有特征分析
2. 特征工程策略
3. 要生成的新特征列表
4. 预期效果

重要：方案必须基于上述实际数据分析结果。
"""

        user_input = prompt_template.format(
            data_path=path,
            target_column=target,
            task_type=task_type,
            data_summary=data_summary,
            skill_content=phase2_content
        )

        # 调用 LLM 生成方案
        result = self.run(user_input, stage="feature_engineering_plan")

        self.feature_plan = result.get("answer", "")

        return self.feature_plan

    def request_user_confirmation(self) -> None:
        """
        请求用户确认特征工程方案

        Raises:
            ConfirmationRequired: 需要用户确认时抛出
        """
        if not self.feature_plan:
            raise ValueError("请先生成特征工程方案")

        # 生成代码预览
        code_preview = self._generate_code_preview()

        # 参考的 skills
        skills_referenced = [
            {
                "name": "afrexai-ml-engineering-1.0.0",
                "files": ["SKILL.md (Phase 2: Data Engineering)"]
            }
        ]

        # 抛出确认异常
        raise ConfirmationRequired(
            stage="feature_engineering",
            proposal=self.feature_plan,
            code_preview=code_preview,
            skills_referenced=skills_referenced
        )

    def _generate_code_preview(self) -> str:
        """生成代码预览"""
        if not self.feature_plan:
            return "# 代码将在确认后生成"

        # 从配置加载 Prompt
        try:
            prompt_template = self.config_loader.get_prompt("feature_engineering", "code_generation")
        except KeyError:
            prompt_template = """基于以下特征工程方案，生成 Python 代码预览：

{plan}

请生成简洁的代码预览（仅展示主要步骤）。
"""

        user_input = prompt_template.format(plan=self.feature_plan[:1000])

        result = self.run(user_input, stage="feature_engineering_code_preview")

        return result.get("answer", "# 代码预览")

    def generate_feature_code(self, modifications: str = None) -> str:
        """
        生成特征工程代码（使用 CodeAct 模式）

        Args:
            modifications: 用户修改内容

        Returns:
            特征工程代码
        """
        if not self.feature_plan:
            raise ValueError("请先生成特征工程方案")

        from ..utils.codeact_agent import CodeActAgent

        modifications_text = f"\n用户修改要求:\n{modifications}\n" if modifications else ""

        # 构建数据信息摘要
        data_info_text = ""
        if self.data_info:
            data_info_text = f"""
## 数据信息

- 数据形状: {self.data_info['shape'][0]} 行 × {self.data_info['shape'][1]} 列
- 目标列: {self.target_column}
- 数值列: {', '.join(self.data_info['numeric_columns'][:15])}
- 分类列: {', '.join(self.data_info['categorical_columns'][:15])}

重要：请基于上述实际数据列名生成代码，不要使用示例数据中的列名。
"""
        self.features_data_path = str(self.asset_manager.session_dir / "data" / "features_data.csv")

        task_prompt = f"""基于以下特征工程方案，生成完整的 Python 代码：

数据路径: {self.data_path}
目标列: {self.target_column}
任务类型: {self.task_type}
{data_info_text}

特征工程方案:
{self.feature_plan}

{modifications_text}

要求：
1. 使用 pandas 和 scikit-learn 进行特征工程
2. 包含详细的注释
3. 保存特征工程后的数据到: {self.features_data_path}
4. 返回新生成的特征列表
5. 代码必须完整可执行，包含所有必要的导入语句
6. 必须使用上述实际数据的列名

请生成完整的、可执行的 Python 代码。
"""

        # 使用 CodeActAgent 生成并执行代码
        codeact = CodeActAgent(llm=self.llm, max_iterations=5, timeout=300)

        context = {
            "data_path": self.data_path,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "feature_data_path": self.features_data_path
        }

        # 生成代码并执行验证
        result = codeact.generate_and_execute(
            task_prompt=task_prompt,
            context=context,
            required_outputs=[],
            required_filepath=self.features_data_path
        )

        if result.success:
            self.feature_code = result.code
            print(f"\n[CodeAct] 代码生成成功，迭代次数: {result.iterations}")
            
            # 保存代码到资产
            if self.feature_code:
                self.asset_manager.save_code(
                    code=self.feature_code,
                    filename="feature_engineering.py",
                    metadata={
                        "stage": "feature_engineering",
                        "data_path": self.data_path,
                        "target_column": self.target_column,
                        "task_type": self.task_type,
                        "execution_success": True,
                        "iterations": result.iterations,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            return self.feature_code
        else:
            print(f"\n[CodeAct] 代码生成失败: {result.error}")
            raise ValueError(f"代码生成失败: {result.error}")

    def calculate_feature_metrics(self) -> Dict[str, Any]:
        """
        计算特征指标（使用 CodeAct 模式）
        
        分两步执行：
        1. 生成并执行指标计算代码
        2. 根据执行结果生成特征分析报告
        
        Returns:
            特征指标结果
        """
        import os
        import json
        
        if not self.features_data_path or not os.path.exists(self.features_data_path):
            raise ValueError("请先执行特征工程代码生成特征数据")

        metrics_result_path = str(self.asset_manager.session_dir / "features" / "feature_metrics.json")
        metrics_report_path = str(self.asset_manager.session_dir / "features" / "feature_metrics_report.md")
        
        # ========== 第一步：生成并执行指标计算代码 ==========
        print(f"\n[CodeAct] 第一步：计算特征指标...")
        
        metrics_task_prompt = f"""基于以下特征数据，计算特征指标：

特征数据路径: {self.features_data_path}
目标列: {self.target_column}
任务类型: {self.task_type}
指标结果保存路径: {metrics_result_path}

请计算以下指标并保存为 JSON 文件：
1. **IV 值（Information Value）**：评估特征对目标的预测能力
   - 对于分类任务：计算每个特征的 IV 值
   - 对于回归任务：将目标离散化后计算 IV 值
2. **相关性分析**：计算特征与目标的相关系数
3. **特征重要性**：使用随机森林计算特征重要性
4. **方差分析**：计算特征的方差，识别低方差特征
5. **缺失率统计**：统计每个特征的缺失率

输出要求：
1. 将所有指标结果保存为 JSON 文件: {metrics_result_path}
2. JSON 格式示例：
{{
    "iv_values": {{"feature1": 0.5, "feature2": 0.3, ...}},
    "correlations": {{"feature1": 0.8, "feature2": 0.6, ...}},
    "feature_importance": {{"feature1": 0.15, "feature2": 0.12, ...}},
    "variances": {{"feature1": 1.5, "feature2": 0.8, ...}},
    "missing_rates": {{"feature1": 0.0, "feature2": 0.05, ...}}
}}

请生成完整的、可执行的 Python 代码。
"""

        # 使用 CodeActAgent 生成并执行代码
        from ..utils.codeact_agent import CodeActAgent
        
        codeact = CodeActAgent(llm=self.llm, max_iterations=5, timeout=300)
        
        context = {
            "features_data_path": self.features_data_path,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "metrics_result_path": metrics_result_path
        }

        # 生成代码并执行验证
        result = codeact.generate_and_execute(
            task_prompt=metrics_task_prompt,
            context=context,
            required_outputs=[]
        )

        if not result.success:
            print(f"\n[CodeAct] 特征指标计算失败: {result.error}")
            raise ValueError(f"特征指标计算失败: {result.error}")

        print(f"\n[CodeAct] 特征指标计算成功，迭代次数: {result.iterations}")
        
        # 检查指标结果文件是否生成
        if not os.path.exists(metrics_result_path):
            print(f"[Agent] 警告: 特征指标结果文件未生成")
            return {"success": False, "error": "特征指标结果文件未生成"}
        
        print(f"[Agent] 特征指标结果已保存到: {metrics_result_path}")
        
        # ========== 第二步：生成特征分析报告 ==========
        print(f"\n[CodeAct] 第二步：生成特征分析报告...")
        
        # 读取指标结果
        with open(metrics_result_path, 'r') as f:
            metrics_data = json.load(f)
        
        report_task_prompt = f"""基于以下特征指标数据，生成详细的特征分析报告：

特征数据路径: {self.features_data_path}
目标列: {self.target_column}
任务类型: {self.task_type}
指标数据: {json.dumps(metrics_data, indent=2, ensure_ascii=False)}
报告保存路径: {metrics_report_path}

请生成详细的特征分析报告（Markdown 格式），包含以下内容：

## 1. 指标概览
- 各指标计算结果汇总表格
- 指标分布统计

## 2. 特征评估
- 高预测能力特征（IV > 0.3）
- 中等预测能力特征（0.1 < IV <= 0.3）
- 低预测能力特征（IV <= 0.1）
- 高相关性特征（|corr| > 0.7）
- 低方差特征（方差接近 0）
- 高缺失率特征（缺失率 > 10%）

## 3. 特征筛选建议
- 建议保留的特征列表
- 建议删除的特征列表及原因
- 需要进一步处理的特征

## 4. 特征优化建议
- 特征工程优化方向
- 模型选择建议
- 后续改进方向

请生成完整的 Markdown 格式报告，保存到: {metrics_report_path}
"""

        # 生成报告
        report_result = codeact.generate_and_execute(
            task_prompt=report_task_prompt,
            context={
                "metrics_data": metrics_data,
                "metrics_report_path": metrics_report_path
            },
            required_outputs=[]
        )

        if report_result.success:
            print(f"\n[CodeAct] 特征分析报告生成成功")
            
            # 检查报告文件是否生成
            if os.path.exists(metrics_report_path):
                print(f"[Agent] 特征分析报告已保存到: {metrics_report_path}")
                
                return {
                    "success": True,
                    "metrics_result_path": metrics_result_path,
                    "metrics_report_path": metrics_report_path,
                    "features_data_path": self.features_data_path,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                print(f"[Agent] 警告: 特征分析报告未生成")
                return {
                    "success": True,
                    "metrics_result_path": metrics_result_path,
                    "metrics_report_path": None,
                    "features_data_path": self.features_data_path,
                    "timestamp": datetime.now().isoformat()
                }
        else:
            print(f"\n[CodeAct] 特征分析报告生成失败: {report_result.error}")
            return {
                "success": True,
                "metrics_result_path": metrics_result_path,
                "metrics_report_path": None,
                "error": report_result.error,
                "features_data_path": self.features_data_path,
                "timestamp": datetime.now().isoformat()
            }

    def execute_feature_engineering(self, code: str = None) -> Dict[str, Any]:
        """
        执行特征工程代码

        Args:
            code: 特征工程代码，为 None 时使用已生成的代码

        Returns:
            执行结果
        """
        from ..utils.code_generator import CodeGenerator
        import shutil

        feature_code = code or self.feature_code

        if not feature_code:
            raise ValueError("请先生成特征工程代码")

        # 使用代码生成器执行代码
        code_gen = CodeGenerator()

        # 临时输出路径（在原始数据目录）
        temp_features_path = self.data_path.replace('.csv', '_features.csv')

        context = {
            "data_path": self.data_path,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "feature_data_path": temp_features_path
        }

        exec_result = code_gen.execute_code(feature_code, context)

        # 检查是否成功生成了特征数据文件
        import os
        file_exists = os.path.exists(temp_features_path)

        # 将特征工程后的数据复制到 session 目录
        final_features_path = None
        if file_exists:
            session_features_path = self.asset_manager.session_dir / "data" / "features_data.csv"
            session_features_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_features_path, session_features_path)
            final_features_path = str(session_features_path)
            print(f"[Agent] 特征工程后数据已复制到 session 目录: {final_features_path}")
            # 删除临时文件
            os.remove(temp_features_path)

        # 构建结果
        result_info = {
            "success": file_exists,  # 只要文件存在就认为成功
            "features_data_path": final_features_path,
            "original_path": self.data_path,
            "target_column": self.target_column,
            "execution_output": exec_result.output,
            "execution_error": exec_result.error if not file_exists else None,
            "timestamp": datetime.now().isoformat()
        }

        # 保存结果信息到资产
        self.asset_manager.save_data(
            data=json.dumps(result_info, ensure_ascii=False, indent=2),
            filename="feature_engineering_result.json",
            asset_type="features",
            metadata=result_info
        )

        return result_info

    def full_feature_engineering_workflow(
        self,
        data_path: str,
        target_column: str,
        task_type: str = "classification",
        skip_confirmation: bool = False
    ) -> Dict[str, Any]:
        """
        执行完整的特征工程流程

        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型
            skip_confirmation: 是否跳过用户确认

        Returns:
            特征工程结果
        """
        self.data_path = data_path
        self.target_column = target_column
        self.task_type = task_type

        # 1. 生成特征工程方案
        plan = self.generate_feature_plan(data_path, target_column, task_type)

        # 2. 请求用户确认（如果不跳过）
        if not skip_confirmation:
            self.request_user_confirmation()

        # 3. 生成特征工程代码
        code = self.generate_feature_code()

        # 4. 执行特征工程
        result = self.execute_feature_engineering(code)

        return {
            "success": result.get("success", False),
            "plan": plan,
            "code": code,
            "result": result
        }
