"""
对话式 AutoML 核心模块 - 改进版

流程：
1. 思路生成 → 用户确认
2. 生成代码 → 自动执行
3. 语法检查 → 自动修复（最多 3 次）
4. 执行结果展示 → 数据下载
"""

import os
import sys
import io
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import ast
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DialogAutoML')


class PipelineStage(Enum):
    """建模流程阶段"""
    IDLE = "idle"
    DATA_LOADED = "data_loaded"
    DATA_EXPLORED = "data_explored"
    DATA_CLEANING = "data_cleaning"  # 清洗思路
    DATA_CLEANED = "data_cleaned"    # 清洗完成
    FEATURE_PLANNING = "feature_planning"  # 特征思路
    FEATURES_GENERATED = "features_generated"
    MODEL_PLANNING = "model_planning"  # 模型思路
    MODEL_TRAINED = "model_trained"
    COMPLETED = "completed"


@dataclass
class CodeBlock:
    """代码块"""
    stage: str
    name: str
    code: str
    description: str
    thinking: str = ""  # 思路说明
    executed: bool = False
    execution_result: Optional[str] = None
    error: Optional[str] = None
    output_data: Optional[pd.DataFrame] = None


class CodeExecutor:
    """代码执行器"""
    
    def __init__(self):
        self.local_vars = {}
        self.globals = {}
    
    def execute(self, code: str, description: str = "", data: pd.DataFrame = None) -> Dict[str, Any]:
        """执行代码，包含语法检查和自动修复"""
        logger.info(f"执行代码: {description[:50]}...")
        
        # 添加数据到执行环境
        if data is not None:
            self.local_vars['df'] = data.copy()
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            # 语法检查
            try:
                ast.parse(code)
                logger.info("✓ 语法检查通过")
            except SyntaxError as e:
                last_error = f"语法错误: {e}"
                logger.warning(f"语法错误 (尝试 {attempt+1}/{max_retries}): {last_error}")
                code = self._fix_syntax_error(code, str(e))
                continue
            
            # 执行代码
            try:
                # 捕获 print 输出
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()
                
                exec_globals = {
                    'pd': pd,
                    'np': np,
                    **self.local_vars
                }
                
                exec(code, exec_globals)
                self.local_vars.update(exec_globals)
                
                # 恢复 stdout 并获取输出
                sys.stdout = old_stdout
                output_text = captured_output.getvalue()
                
                # 获取输出数据 - 按优先级检查
                output_data = None
                for var_name in ['df_featured', 'df_clean', 'df', 'result', 'results']:
                    if var_name in self.local_vars:
                        output_data = self.local_vars[var_name]
                        logger.info(f"找到输出变量: {var_name}, shape={getattr(output_data, 'shape', 'N/A')}")
                        break
                
                result = {
                    "success": True,
                    "message": "代码执行成功",
                    "output": output_text if output_text else "执行完成",
                    "output_data": output_data
                }
                
                logger.info(f"✓ 代码执行成功: {description[:50]}")
                return result
                
            except Exception as e:
                # 恢复 stdout
                sys.stdout = old_stdout
                last_error = str(e)
                logger.warning(f"执行错误 (尝试 {attempt+1}/{max_retries}): {last_error}")
        
        # 所有尝试都失败
        return {
            "success": False,
            "message": "代码执行失败",
            "error": last_error,
            "output": traceback.format_exc()
        }
    
    def _fix_syntax_error(self, code: str, error_msg: str) -> str:
        """尝试修复语法错误（简单版本）"""
        # 常见的简单语法错误修复
        fixes = [
            # 修复缺失的括号
            ("(", "(\n"),
            # 修复缺少的引号
            ('"', '"'),
            # 移除可能的不可见字符
            (r'\x00', ''),
        ]
        
        for old, new in fixes:
            code = code.replace(old, new)
        
        return code
    
    def get_data(self, name: str) -> Any:
        return self.local_vars.get(name)


class ThoughtGenerator:
    """思路生成器 - 使用 LLM 生成处理思路"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_cleaning_thinking(self, data_profile: Dict, user_requirements: str = "") -> str:
        """生成数据清洗思路"""
        missing = data_profile.get('missing', {})
        missing_str = self._summarize_missing(missing)
        
        prompt = f"""
你是一个数据科学家，请为以下数据集提供数据清洗思路。

数据集信息:
- 形状: {data_profile.get('shape', 'N/A')}
- 数值列: {len(data_profile.get('numeric_columns', []))}
- 类别列: {len(data_profile.get('categorical_columns', []))}
- 缺失值: {missing_str}

用户需求:
{user_requirements if user_requirements else "无特定需求"}

请简洁地描述：
1. 数据存在的主要问题
2. 计划如何处理
3. 预期的处理效果

请用中文回复，直接给出思路描述，不要代码。
"""
        return self._generate_thought(prompt)
    
    def generate_feature_thinking(self, data_profile: Dict, directions: List[str]) -> str:
        """生成特征工程思路"""
        prompt = f"""
你是一个数据科学家，请为以下数据集提供特征工程思路。

数据集信息:
- 形状: {data_profile.get('shape', 'N/A')}
- 数值列: {len(data_profile.get('numeric_columns', []))}
- 类别列: {len(data_profile.get('categorical_columns', []))}

用户希望的方向:
{chr(10).join(f"- {d}" for d in directions)}

请简洁地描述：
1. 计划创建哪些新特征
2. 如何编码类别变量
3. 预期的效果

请用中文回复，直接给出思路描述，不要代码。
"""
        return self._generate_thought(prompt)
    
    def generate_model_thinking(self, data_profile: Dict, user_requirements: str = "") -> str:
        """生成模型训练思路"""
        prompt = f"""
你是一个数据科学家，请为以下数据集提供模型训练思路。

数据集信息:
- 形状: {data_profile.get('shape', 'N/A')}
- 目标列: {data_profile.get('target_column', 'N/A')}

用户需求:
{user_requirements if user_requirements else "预测目标变量"}

请简洁地描述：
1. 建议使用的模型
2. 数据划分策略
3. 评估指标

请用中文回复，直接给出思路描述，不要代码。
"""
        return self._generate_thought(prompt)
    
    def _generate_thought(self, prompt: str) -> str:
        """调用 LLM 生成思路"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            if not isinstance(content, str):
                content = str(content)
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"思路生成失败: {e}")
            return f"思路生成失败: {e}"
    
    def _summarize_missing(self, missing: Dict) -> str:
        """总结缺失值"""
        if not missing:
            return "无"
        
        has_missing = {k: v for k, v in missing.items() if v > 0}
        if not has_missing:
            return "无"
        
        items = list(has_missing.items())[:5]
        return ", ".join(f"{k}: {v}" for k, v in items)


class CodeGenerator:
    """代码生成器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_code(self, thinking: str, stage: str, data_profile: Dict = None) -> str:
        """根据思路生成代码"""
        
        if stage == "data_cleaning":
            return self._generate_cleaning_code(thinking, data_profile)
        elif stage == "feature_engineering":
            return self._generate_feature_code(thinking, data_profile)
        elif stage == "model_training":
            return self._generate_model_code(thinking, data_profile)
        else:
            return "# 未知阶段"
    
    def _generate_cleaning_code(self, thinking: str, data_profile: Dict = None) -> str:
        """生成清洗代码"""
        shape = data_profile.get('shape', 'N/A') if data_profile else 'N/A'
        missing = data_profile.get('missing', {}) if data_profile else {}
        missing_str = self._summarize_missing(missing)
        
        prompt = f"""
请根据以下思路生成数据清洗代码。

数据信息:
- 形状: {shape}
- 缺失值: {missing_str}

处理思路:
{thinking}

要求:
1. 使用 pandas
2. 代码完整可运行
3. 结果保存到 df_clean 变量
4. 直接返回 Python 代码，用 ```python ... ``` 包裹
"""
        return self._generate_from_prompt(prompt)
    
    def _generate_feature_code(self, thinking: str, data_profile: Dict = None) -> str:
        """生成特征代码"""
        cols = data_profile.get('columns', []) if data_profile else []
        
        prompt = f"""
请根据以下思路生成特征工程代码。

重要提示：
- 输入数据在 `df` 变量中
- 数据当前包含的列: {cols}
- 请使用当前数据中实际存在的列，不要假设任何列存在！
- 如果某列不存在，跳过或用其他列替代

处理思路:
{thinking}

要求:
1. 使用 pandas 和 numpy
2. 直接使用 df 变量中的数据，不要创建新数据！
3. 代码完整可运行，处理列不存在的情况
4. 结果保存到 df_featured 变量
5. 直接返回 Python 代码，用 ```python ... ``` 包裹
"""
        return self._generate_from_prompt(prompt)
    
    def _generate_model_code(self, thinking: str, data_profile: Dict = None) -> str:
        """生成模型代码"""
        prompt = f"""
请根据以下思路生成模型训练代码。

处理思路:
{thinking}

要求:
1. 使用 sklearn
2. 代码完整可运行
3. 训练模型并评估，结果保存到 results 变量
4. 直接返回 Python 代码，用 ```python ... ``` 包裹
"""
        return self._generate_from_prompt(prompt)
    
    def fix_code_error(self, code: str, error: str, thinking: str) -> str:
        """根据错误修复代码"""
        prompt = f"""
请修复以下代码的错误。

原始思路:
{thinking}

代码:
{code}

错误:
{error}

请修复代码中的问题，直接返回修复后的 Python 代码，用 ```python ... ``` 包裹。
"""
        return self._generate_from_prompt(prompt)
    
    def _generate_from_prompt(self, prompt: str) -> str:
        """调用 LLM 生成代码"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            if not isinstance(content, str):
                content = str(content)
            
            # 提取代码
            import re
            code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"代码生成失败: {e}")
            return f"# 代码生成失败: {e}"
    
    def _summarize_missing(self, missing: Dict) -> str:
        """总结缺失值"""
        if not missing:
            return "无"
        
        # 只显示有缺失值的列
        has_missing = {k: v for k, v in missing.items() if v > 0}
        if not has_missing:
            return "无"
        
        # 显示前 5 个
        items = list(has_missing.items())[:5]
        return ", ".join(f"{k}: {v}" for k, v in items)


class DialogPipeline:
    """对话式 AutoML 流程"""
    
    def __init__(self, llm):
        self.llm = llm
        self.thought_generator = ThoughtGenerator(llm)
        self.code_generator = CodeGenerator(llm)
        self.code_executor = CodeExecutor()
        
        # 状态
        self.data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.modeling_scenario: str = ""
        self.data_profile: Optional[Dict] = None
        self.current_stage: Optional[str] = None
        self.code_blocks: List[CodeBlock] = []
        
        # 当前进度
        self.current_thinking: str = ""
        self.current_code: str = ""
    
    def load_data(self, data: pd.DataFrame, target_column: str, modeling_scenario: str = "") -> Dict:
        """加载数据"""
        self.data = data
        self.target_column = target_column
        self.modeling_scenario = modeling_scenario
        self.data_profile = self._generate_profile(data)
        
        return {
            "success": True,
            "message": f"数据加载完成: {data.shape[0]}行 x {data.shape[1]}列",
            "profile": self.data_profile
        }
    
    def generate_cleaning_thinking(self, user_requirements: str = "") -> Dict:
        """生成清洗思路"""
        self.current_stage = "data_cleaning"
        
        thinking = self.thought_generator.generate_cleaning_thinking(
            self.data_profile,
            user_requirements
        )
        self.current_thinking = thinking
        
        return {
            "success": True,
            "thinking": thinking,
            "message": "数据清洗思路已生成，请确认"
        }
    
    def generate_cleaning_code(self) -> Dict:
        """生成清洗代码并执行"""
        logger.info("=== 开始生成清洗代码 ===")
        
        code = self.code_generator.generate_code(
            self.current_thinking,
            "data_cleaning",
            self.data_profile
        )
        
        logger.info(f"生成的代码:\n{code[:200]}...")
        self.current_code = code
        
        # 执行代码
        logger.info("开始执行代码...")
        result = self.code_executor.execute(code, "数据清洗", self.data)
        
        logger.info(f"执行结果: success={result.get('success')}, message={result.get('message')}")
        
        # 保存代码块
        code_block = CodeBlock(
            stage="data_cleaning",
            name="数据清洗",
            code=code,
            description=self.current_thinking,
            executed=result["success"]
        )
        self.code_blocks.append(code_block)
        
        if result["success"]:
            # 更新数据
            if result.get("output_data") is not None:
                self.data = result["output_data"]
            
            return {
                "success": True,
                "message": "数据清洗完成！",
                "result": result,
                "data_shape": self.data.shape
            }
        else:
            # 尝试修复
            return self._fix_and_retry(code, result, "data_cleaning")
    
    def generate_feature_thinking(self, directions: List[str]) -> Dict:
        """生成特征工程思路"""
        self.current_stage = "feature_engineering"
        
        thinking = self.thought_generator.generate_feature_thinking(
            self.data_profile,
            directions
        )
        self.current_thinking = thinking
        
        return {
            "success": True,
            "thinking": thinking,
            "message": "特征工程思路已生成，请确认"
        }
    
    def generate_feature_code(self) -> Dict:
        """生成特征代码并执行"""
        code = self.code_generator.generate_code(
            self.current_thinking,
            "feature_engineering",
            self.data_profile
        )
        self.current_code = code
        
        result = self.code_executor.execute(code, "特征工程", self.data)
        
        code_block = CodeBlock(
            stage="feature_engineering",
            name="特征工程",
            code=code,
            description=self.current_thinking,
            executed=result["success"]
        )
        self.code_blocks.append(code_block)
        
        if result["success"]:
            if result.get("output_data") is not None:
                self.data = result["output_data"]
            
            return {
                "success": True,
                "message": "特征工程完成！",
                "result": result,
                "data_shape": self.data.shape
            }
        else:
            return self._fix_and_retry(code, result, "feature_engineering")
    
    def generate_model_thinking(self, user_requirements: str = "") -> Dict:
        """生成模型训练思路"""
        self.current_stage = "model_training"
        
        thinking = self.thought_generator.generate_model_thinking(
            self.data_profile,
            user_requirements
        )
        self.current_thinking = thinking
        
        return {
            "success": True,
            "thinking": thinking,
            "message": "模型训练思路已生成，请确认"
        }
    
    def generate_model_code(self) -> Dict:
        """生成模型代码并执行"""
        code = self.code_generator.generate_code(
            self.current_thinking,
            "model_training",
            self.data_profile
        )
        self.current_code = code
        
        result = self.code_executor.execute(code, "模型训练", self.data)
        
        code_block = CodeBlock(
            stage="model_training",
            name="模型训练",
            code=code,
            description=self.current_thinking,
            executed=result["success"]
        )
        self.code_blocks.append(code_block)
        
        if result["success"]:
            return {
                "success": True,
                "message": "模型训练完成！",
                "result": result
            }
        else:
            return self._fix_and_retry(code, result, "model_training")
    
    def _fix_and_retry(self, code: str, result: Dict, stage: str) -> Dict:
        """修复代码错误并重试"""
        max_fixes = 3
        
        for i in range(max_fixes):
            fixed_code = self.code_generator.fix_code_error(
                code,
                result.get("error", ""),
                self.current_thinking
            )
            
            retry_result = self.code_executor.execute(
                fixed_code,
                f"{stage} (修复{i+1})",
                self.data
            )
            
            if retry_result["success"]:
                # 更新代码块
                self.code_blocks[-1].code = fixed_code
                self.code_blocks[-1].executed = True
                
                if retry_result.get("output_data") is not None:
                    self.data = retry_result["output_data"]
                
                return {
                    "success": True,
                    "message": f"代码已修复并执行成功！(尝试 {i+1})",
                    "result": retry_result,
                    "data_shape": self.data.shape
                }
        
        return {
            "success": False,
            "message": "代码修复失败",
            "error": result.get("error"),
            "result": result
        }
    
    def get_data_download_link(self) -> str:
        """获取数据下载链接"""
        if self.data is None:
            return None
        
        csv = self.data.to_csv(index=False)
        return csv
    
    def export_code(self) -> str:
        """导出所有代码"""
        lines = [
            "# 对话式 AutoML 生成的代码",
            f"# 目标列: {self.target_column}",
            f"# 生成时间: {pd.Timestamp.now()}",
            ""
        ]
        
        for block in self.code_blocks:
            if block.executed:
                lines.append(f"\n# {'='*60}")
                lines.append(f"# {block.name}")
                lines.append(f"# {'='*60}")
                lines.append(block.code)
        
        return "\n".join(lines)
    
    def _generate_profile(self, data: pd.DataFrame) -> Dict:
        """生成数据概览"""
        return {
            "shape": data.shape,
            "columns": list(data.columns),
            "numeric_columns": list(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(data.select_dtypes(include=['object']).columns),
            "missing": data.isnull().sum().to_dict(),
            "target_column": self.target_column
        }
