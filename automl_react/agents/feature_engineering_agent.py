"""
特征工程 Agent 模块

实现特征工程思路生成、用户确认、代码生成与执行
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.react_agent import ReActAgent, ConfirmationRequired
from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
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
        self.config_loader = get_config_loader()
        self.logger = LLMLogger(session_id=session_id)

    def _register_default_tools(self):
        """注册默认工具"""
        super()._register_default_tools()
        from ..tools.data_tools import DataLoaderTool, DataAnalyzerTool
        from ..tools.stage_tools import StageResultTool
        self.register_tool("load_data", DataLoaderTool())
        self.register_tool("analyze_data", DataAnalyzerTool())
        self.register_tool("query_stage_result", StageResultTool(session_id=self.session_id))

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.config_loader.get_prompt("feature_engineering", "system_prompt")

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

        prompt_template = self.config_loader.get_prompt("feature_engineering", "analysis_prompt")
        user_input = prompt_template.format(data_path=data_path, target_column=target_column)

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
        cleaned_data_path: str = None,
        task_description: str = ""
    ) -> str:
        """
        生成特征工程思路

        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            task_type: 任务类型
            analysis_result: 探索性分析报告（可选）
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

        task_context = ""
        
        # 添加用户的建模背景
        if task_description:
            task_context = f"""
## 用户建模背景和要求

{task_description}

**重要：请在特征工程方案中充分考虑用户的建模背景和要求。**

"""
        self.task_context = task_context
        
        exploration_context = ""
        if analysis_result:
            exploration_context = f"""
## 探索性分析报告（来自数据探索阶段）

{analysis_result}

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

            # 构建当前数据上下文
            current_data_context = f"""
## 当前数据基本信息

- **数据路径**: {path}
- **数据来源说明**: 数据清洗阶段已完成，当前输入文件为清洗结果数据
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

            # 用于强约束 LLM 的“事实快照”，避免被前序文本污染。
            verified_facts = {
                "data_path": path,
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "total_columns": int(len(df.columns)),
                "numeric_columns_count": int(len(self.data_info['numeric_columns'])),
                "categorical_columns_count": int(len(self.data_info['categorical_columns'])),
                "target_column": target,
                "target_dtype": self.data_info['target_dtype'],
            }

        except Exception as e:
            current_data_context = f"无法加载数据文件: {path}\n错误: {str(e)}"
            verified_facts = {
                "data_path": path,
                "error": str(e)
            }

        # 从配置加载 Prompt
        prompt_template = self.config_loader.get_prompt("feature_engineering", "plan_generation")

        user_input = prompt_template.format(
            data_path=path,
            target_column=target,
            task_type=task_type,
            task_context=task_context,
            exploration_context=exploration_context,
            current_data_context=current_data_context,
            verified_facts_json=json.dumps(verified_facts, ensure_ascii=False, indent=2),
        )

        # 调用 LLM 生成方案
        result = self.run(user_input, stage="feature_engineering_plan")

        self.feature_plan = result.get("answer", "")

        return self.feature_plan

    def revise_plan(self, current_plan: str, modifications: str, **kwargs) -> str:
        """基于用户反馈修订特征工程方案"""
        prompt_template = self.config_loader.get_prompt("feature_engineering", "plan_revision")
        user_input = prompt_template.format(
            current_plan=current_plan,
            user_modifications=modifications,
            data_path=getattr(self, "data_path", ""),
            target_column=getattr(self, "target_column", ""),
        )
        result = self.run(user_input, stage="feature_engineering_plan_revision")
        self.feature_plan = result.get("answer", "")
        return self.feature_plan

    def get_modifiable_aspects(self) -> list:
        return ["特征构造方法", "编码策略", "特征选择", "交叉特征", "缺失值填充策略"]

    def request_user_confirmation(self) -> None:
        """
        请求用户确认特征工程方案

        Raises:
            ConfirmationRequired: 需要用户确认时抛出
        """
        if not self.feature_plan:
            raise ValueError("请先生成特征工程方案")

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
            skills_referenced=skills_referenced
        )

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
        self.features_data_path = str(self.asset_manager.session_dir / "data" / "features_train.csv")

        prompt_template = self.config_loader.get_prompt("feature_engineering", "code_generation_full")
        task_prompt = prompt_template.format(
            data_path=self.data_path,
            target_column=self.target_column,
            task_type=self.task_type,
            data_info_text=data_info_text,
            plan=self.feature_plan,
            modifications=modifications_text,
            features_data_path=self.features_data_path,
            task_context=getattr(self, 'task_context', ''),
        )

        # 使用 CodeActAgent 生成并执行代码
        codeact = CodeActAgent(llm=self.llm, max_iterations=5, timeout=300, session_id=self.session_id)
        if self._stream_callback:
            codeact.set_stream_callback(self._stream_callback)

        context = {
            "data_path": self.data_path,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "feature_data_path": self.features_data_path,
            "input_data_path": self.data_path,
            "output_data_path": self.features_data_path,
        }

        # 生成代码并执行验证
        result = codeact.generate_and_execute(
            task_prompt=task_prompt,
            context=context,
            required_outputs=[],
            required_filepath=self.features_data_path,
            output_validator=self._validate_features_output,
            deterministic_fallback=self._deterministic_feature_fallback,
            stage="feature_engineering_code_generation",
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

    def _validate_features_output(self, output_path: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """特征工程输出校验：文件可读、非空、包含有效特征列。"""
        import pandas as pd

        df_out = pd.read_csv(output_path)
        if df_out.shape[0] <= 0:
            return False, "特征工程输出为空"
        if df_out.shape[1] <= 0:
            return False, "特征工程输出无列"

        target = context.get("target_column") or self.target_column
        non_target_cols = [c for c in df_out.columns if c != target]
        if len(non_target_cols) <= 0:
            return False, "特征工程输出缺少可用特征列"

        # 行数校验：特征工程仅处理训练集（data_path 即 train_raw），
        # valid/test 应在模型训练/评估阶段复用同样的变换逻辑处理。
        in_path = context.get("data_path") or self.data_path
        if in_path:
            try:
                df_in = pd.read_csv(in_path)
                if df_out.shape[0] != df_in.shape[0]:
                    return False, f"特征工程输出行数({df_out.shape[0]})与输入行数({df_in.shape[0]})不一致"
            except Exception:
                pass

        return True, f"{df_out.shape[0]} 行 × {df_out.shape[1]} 列"

    def _deterministic_feature_fallback(self, context: Dict[str, Any], output_path: str) -> Tuple[bool, str]:
        """确定性兜底：无需 LLM，执行基础可复现特征处理并确保落盘。"""
        import pandas as pd

        in_path = context.get("data_path") or self.data_path
        target = context.get("target_column") or self.target_column
        if not in_path:
            return False, "缺少输入数据路径"

        try:
            df = pd.read_csv(in_path)
        except Exception as e:
            return False, f"读取输入数据失败: {e}"

        if df.empty:
            return False, "输入数据为空"

        target_series = None
        if target and target in df.columns:
            target_series = df[target].copy()
            X = df.drop(columns=[target]).copy()
        else:
            X = df.copy()

        # 基础处理：缺失填充 + 类别编码
        numeric_cols = X.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if X[col].isnull().any():
                med = X[col].median()
                if pd.notna(med):
                    X[col] = X[col].fillna(med)

        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
        for col in cat_cols:
            X[col] = X[col].fillna("UNKNOWN").astype(str)
            X[col] = pd.factorize(X[col])[0]

        out_df = X.copy()
        if target_series is not None:
            out_df[target] = target_series.values

        try:
            out_df.to_csv(output_path, index=False)
        except Exception as e:
            return False, f"保存特征工程结果失败: {e}"

        # 保存函数式代码文件供 pipeline 使用
        fallback_code = f'''import os
import pandas as pd


def engineer_features(input_path, output_path, train_path=None):
    """基础特征工程：缺失填充 + 类别编码（确定性兜底）。"""
    if train_path is None:
        train_path = input_path

    train_df = pd.read_csv(train_path)
    df = pd.read_csv(input_path)

    target_column = "{target}"
    target_series = None
    if target_column and target_column in df.columns:
        target_series = df[target_column].copy()
        X = df.drop(columns=[target_column]).copy()
        train_X = train_df.drop(columns=[target_column], errors="ignore").copy()
    else:
        X = df.copy()
        train_X = train_df.drop(columns=[target_column], errors="ignore").copy()

    # 数值列用训练集中位数填充
    numeric_cols = X.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if X[col].isnull().any():
            med = train_X[col].median() if col in train_X.columns else X[col].median()
            if pd.notna(med):
                X[col] = X[col].fillna(med)

    # 类别列编码
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
    for col in cat_cols:
        X[col] = X[col].fillna("UNKNOWN").astype(str)
        # 用训练集建立映射
        if col in train_X.columns:
            categories = train_X[col].fillna("UNKNOWN").astype(str).unique()
            cat_map = {{cat: i for i, cat in enumerate(sorted(categories))}}
            X[col] = X[col].map(cat_map).fillna(-1).astype(int)
        else:
            X[col] = pd.factorize(X[col])[0]

    new_features = list(cat_cols)
    print(f"新生成/变换的特征: {{new_features}}")

    out_df = X.copy()
    if target_series is not None:
        out_df[target_column] = target_series.values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import sys
    _input = sys.argv[1] if len(sys.argv) > 1 else "{in_path}"
    _output = sys.argv[2] if len(sys.argv) > 2 else "{output_path}"
    _train = sys.argv[3] if len(sys.argv) > 3 else _input
    engineer_features(_input, _output, _train)
'''
        self.asset_manager.save_code(
            code=fallback_code,
            filename="feature_engineering.py",
            metadata={"stage": "feature_engineering", "fallback": True}
        )

        return True, f"确定性特征工程完成并保存到 {output_path}"

    def _generate_markdown_text(self, prompt: str, stage: str = "") -> str:
        """直接生成 Markdown 文本，不经过 CodeAct 执行链路。"""
        if not self.llm:
            raise ValueError("LLM 未初始化，无法生成报告")

        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        full_response = ""
        start_time = datetime.now()

        try:
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    full_response += chunk.content
        except Exception:
            response = self.llm.invoke(messages)
            full_response = response.content if hasattr(response, "content") else str(response)

        llm_config = self.config_loader.get_llm_config()
        model_name = llm_config.get("model_name", "unknown")
        provider = llm_config.get("provider", "unknown")
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        self.llm_logger.log_call(
            model_name=model_name,
            provider=provider,
            input_content=self._serialize_llm_input(messages),
            output_content=full_response,
            latency_ms=latency_ms,
            stage=stage,
            metadata={
                "call_type": "direct_generation",
                "prompt_scope": "final_actual_llm_input",
                "prompt_format": "chat_messages_system_user",
            }
        )

        content = full_response.strip()
        if content.startswith("```markdown"):
            content = content[len("```markdown"):]
        elif content.startswith("```"):
            content = content[len("```"):]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()
        if not content:
            raise ValueError("模型未生成有效报告内容")

        return content

    def _write_report_file(self, file_path: str, content: str) -> Tuple[bool, str]:
        """将报告内容写入本地文件并进行存在性校验。"""
        import os

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content.rstrip() + "\n")
        except Exception as e:
            return False, f"写入报告失败: {e}"

        if not os.path.exists(file_path):
            return False, f"报告文件未生成: {file_path}"

        if os.path.getsize(file_path) <= 0:
            return False, f"报告文件为空: {file_path}"

        return True, file_path

    def calculate_feature_metrics(self, modifications: str = None) -> Dict[str, Any]:
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
        modifications_text = f"\n用户关注点/补充要求:\n{modifications}\n" if modifications else ""
        
        # ========== 第一步：生成并执行指标计算代码 ==========
        print(f"\n[CodeAct] 第一步：计算特征指标...")
        
        prompt_template = self.config_loader.get_prompt("feature_engineering", "metrics_code_generation")
        metrics_task_prompt = prompt_template.format(
            features_data_path=self.features_data_path,
            target_column=self.target_column,
            task_type=self.task_type,
            metrics_result_path=metrics_result_path,
            modifications_text=modifications_text,
            task_context=getattr(self, 'task_context', ''),
        )

        # 使用 CodeActAgent 生成并执行代码
        from ..utils.codeact_agent import CodeActAgent
        
        codeact = CodeActAgent(llm=self.llm, max_iterations=5, timeout=300, session_id=self.session_id)
        if self._stream_callback:
            codeact.set_stream_callback(self._stream_callback)

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
            required_outputs=[],
            stage="feature_metrics_code_generation"
        )

        if not result.success:
            print(f"\n[CodeAct] 特征指标计算失败: {result.error}")
            raise ValueError(f"特征指标计算失败: {result.error}")

        print(f"\n[CodeAct] 特征指标计算成功，迭代次数: {result.iterations}")
        if result.code:
            self.asset_manager.save_code(
                code=result.code,
                filename="feature_metrics.py",
                metadata={
                    "stage": "feature_evaluation",
                    "data_path": self.features_data_path,
                    "target_column": self.target_column,
                    "task_type": self.task_type,
                    "execution_success": result.success,
                    "execution_error": result.error,
                    "iterations": result.iterations,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        
        # 检查指标结果文件是否生成
        if not os.path.exists(metrics_result_path):
            print(f"[Agent] 警告: 特征指标结果文件未生成")
            return {"success": False, "error": "特征指标结果文件未生成"}
        
        print(f"[Agent] 特征指标结果已保存到: {metrics_result_path}")
        
        # ========== 第二步：直接生成特征分析报告 ==========
        print(f"\n[LLM] 第二步：直接生成特征分析报告...")
        
        # 读取指标结果
        with open(metrics_result_path, 'r') as f:
            metrics_data = json.load(f)
        
        prompt_template = self.config_loader.get_prompt("feature_engineering", "metrics_report_generation")
        report_task_prompt = prompt_template.format(
            features_data_path=self.features_data_path,
            target_column=self.target_column,
            task_type=self.task_type,
            feature_plan=self.feature_plan or "无",
            metrics_data=json.dumps(metrics_data, indent=2, ensure_ascii=False),
            metrics_report_path=metrics_report_path,
            modifications_text=modifications_text,
            task_context=getattr(self, 'task_context', ''),
        )

        try:
            report_content = self._generate_markdown_text(
                report_task_prompt,
                stage="feature_metrics_report_generation"
            )
            report_ok, report_msg = self._write_report_file(metrics_report_path, report_content)
        except Exception as e:
            print(f"\n[LLM] 特征分析报告生成失败: {e}")
            return {
                "success": True,
                "metrics_result_path": metrics_result_path,
                "metrics_report_path": None,
                "error": str(e),
                "features_data_path": self.features_data_path,
                "timestamp": datetime.now().isoformat()
            }

        if report_ok:
            print(f"\n[LLM] 特征分析报告生成成功")
            
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
            print(f"\n[LLM] 特征分析报告生成失败: {report_msg}")
            return {
                "success": True,
                "metrics_result_path": metrics_result_path,
                "metrics_report_path": None,
                "error": report_msg,
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
