"""
CodeAct Agent 模块

基于 CodeAct 模式的代码生成和执行框架。
CodeAct 模式：LLM 生成代码 → 执行代码 → 观察执行结果 → 根据结果调整代码（循环）
"""

import json
import os
import re
import ast
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..config import get_config_loader
from ..logger import get_llm_logger
from ..core.stream_callback import StreamCallbackFn, StreamEvent, StreamEventType


@dataclass
class CodeActResult:
    """CodeAct 执行结果"""
    success: bool
    code: str
    output: str
    error: Optional[str] = None
    iterations: int = 0
    execution_time: float = 0.0
    execution_error: Optional[str] = None  # 代码执行错误（与生成错误区分）


class CodeActAgent:
    """
    CodeAct Agent
    
    基于 CodeAct 模式的代码生成和执行框架。
    通过迭代的方式生成、执行、观察、修正代码。
    """
    
    def __init__(self, llm: Any = None, max_iterations: int = 5, timeout: int = 300,
                 session_id: str = None, use_subprocess: bool = True):
        self.llm = llm
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.session_id = session_id or "default"
        self.config_loader = get_config_loader()
        self.llm_logger = get_llm_logger(session_id=self.session_id)
        self._stream_callback: StreamCallbackFn = None
        # 子进程执行模式（可通过环境变量 CODEACT_USE_SUBPROCESS=0 关闭）
        self.use_subprocess = use_subprocess and os.environ.get("CODEACT_USE_SUBPROCESS", "1") != "0"
        if self.use_subprocess:
            from .sandbox import SandboxExecutor
            # 从配置读取沙盒参数
            try:
                exec_config = self.config_loader.get_workflow_config("execution") or {}
            except KeyError:
                exec_config = {}
            self.subprocess_executor = SandboxExecutor(
                mode=exec_config.get("sandbox_mode", "subprocess"),
                timeout=timeout,
                memory_limit_mb=exec_config.get("memory_limit_mb", 2048),
                cpu_time_limit=exec_config.get("cpu_time_limit", timeout),
            )

    def set_stream_callback(self, callback: StreamCallbackFn):
        """设置流式输出回调函数"""
        self._stream_callback = callback

    def generate_and_execute(
        self,
        task_prompt: str,
        context: Dict[str, Any] = None,
        required_outputs: List[str] = None,
        required_filepath: str = None,
        output_validator: Optional[Callable[[str, Dict[str, Any]], Tuple[bool, str]]] = None,
        deterministic_fallback: Optional[Callable[[Dict[str, Any], str], Tuple[bool, str]]] = None,
        syntax_check: bool = True,
        stage: str = "",
    ) -> CodeActResult:
        """
        生成并执行代码（CodeAct 模式）
        
        Args:
            task_prompt: 任务提示词
            context: 执行上下文变量
            required_outputs: 需要验证的输出变量名
            required_filepath: 需要验证的输出文件路径
            output_validator: 输出文件校验函数，返回 (是否通过, 说明)
            deterministic_fallback: 确定性兜底函数，返回 (是否成功, 说明)
            syntax_check: 是否进行代码语法完整性校验
            
        Returns:
            CodeActResult
        """
        start_time = time.time()
        
        conversation_history = []
        current_code = ""
        last_error = None
        
        for iteration in range(self.max_iterations):
            print(f"\n[CodeAct] 迭代 {iteration + 1}/{self.max_iterations}")
            if self._stream_callback:
                self._stream_callback(StreamEvent(StreamEventType.PROGRESS, f"代码生成迭代 {iteration + 1}/{self.max_iterations}"))

            # 构建提示词
            if iteration == 0:
                prompt = self._build_initial_messages(task_prompt)
            else:
                print(last_error)
                prompt = self._build_retry_messages(task_prompt, current_code, last_error)

            # 生成代码
            print(f"[CodeAct] 生成代码...")
            code_result = self._generate_code(prompt, stage=stage, iteration=iteration + 1)

            if not code_result["success"]:
                last_error = code_result.get("error", "代码生成失败")
                continue

            current_code = code_result["code"]

            # L1: 代码完整性校验（语法/结构）
            if syntax_check:
                syntax_ok, syntax_msg = self._validate_code_syntax(current_code, stage=stage)
                if not syntax_ok:
                    last_error = syntax_msg
                    continue

            # 执行代码
            print(f"[CodeAct] 执行代码...")
            if self._stream_callback:
                self._stream_callback(StreamEvent(StreamEventType.PROGRESS, "正在执行代码..."))
            exec_result = self._execute_code(current_code, context, required_outputs)
            
            if exec_result["success"]:
                # 检查必需的输出变量
                if required_outputs:
                    missing = [var for var in required_outputs if var not in exec_result.get("variables", {})]
                    if missing:
                        last_error = f"缺少必需的输出变量: {missing}"
                        continue
                
                # 检查必需的输出文件
                if required_filepath:
                    file_ok, file_msg = self._validate_required_file(
                        required_filepath,
                        context or {},
                        output_validator,
                    )
                    if not file_ok:
                        last_error = file_msg
                        continue
            

                # 成功 
                execution_time = time.time() - start_time
                return CodeActResult(
                    success=True,
                    code=current_code,
                    output=exec_result.get("output", ""),
                    iterations=iteration + 1,
                    execution_time=execution_time,
                    execution_error=None
                )
            else:
                last_error = exec_result.get("error", "代码执行失败")
        
        # L3: 确定性兜底（仅在需要落盘且提供兜底函数时触发）
        if deterministic_fallback and required_filepath:
            try:
                ok, msg = deterministic_fallback(context or {}, required_filepath)
                if ok and os.path.isfile(required_filepath):
                    execution_time = time.time() - start_time
                    return CodeActResult(
                        success=True,
                        code=current_code,
                        output=f"Deterministic fallback applied: {msg}",
                        iterations=self.max_iterations,
                        execution_time=execution_time,
                        execution_error=None,
                    )
                last_error = f"确定性兜底失败: {msg}"
            except Exception as e:
                last_error = f"确定性兜底异常: {e}"

        # 所有迭代都失败
        execution_time = time.time() - start_time
        return CodeActResult(
            success=False,
            code=current_code,
            output="",
            error=f"经过 {self.max_iterations} 次迭代仍无法生成可执行代码。最后错误: {last_error}",
            iterations=self.max_iterations,
            execution_time=execution_time,
            execution_error=last_error
        )

    def _validate_code_syntax(self, code: str, stage: str = "") -> Tuple[bool, str]:
        """校验代码是否可解析，并检查函数封装格式。"""
        if not code or not code.strip():
            return False, "代码为空"

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"代码语法不完整: {e}"

        # R1: 禁止 globals().update()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'update'
                    and isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Name)
                    and node.func.value.func.id == 'globals'):
                return False, "代码格式违规: 不要使用 globals().update()，直接赋值变量即可"

        # R4: 禁止多行 import（括号换行）
        if re.search(r'from\s+\S+\s+import\s*\(', code):
            return False, "代码格式违规: import 语句必须写在单行，不要用括号换行（from X import a, b, c）"

        # 正向校验：必须有指定的主函数定义和 if __name__ 块
        expected_func = None
        if "cleaning" in stage:
            expected_func = "clean_data"
        elif "feature" in stage:
            expected_func = "engineer_features"
        elif "training" in stage or "model_training" in stage:
            expected_func = "train_model"

        if expected_func:
            func_names = [
                node.name for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if expected_func not in func_names:
                return False, f"代码格式违规: 必须定义主函数 {expected_func}()，将核心逻辑封装在函数内"

            # 检查必须有 if __name__ == "__main__" 块
            has_main_guard = False
            for node in tree.body:
                if isinstance(node, ast.If):
                    test = node.test
                    if (isinstance(test, ast.Compare)
                            and isinstance(test.left, ast.Name) and test.left.id == '__name__'
                            and any(isinstance(c, ast.Constant) and c.value == '__main__' for c in test.comparators)):
                        has_main_guard = True
                        break
            if not has_main_guard:
                return False, "代码格式违规: 必须包含 if __name__ == '__main__': 块，在其中解析参数并调用主函数"

        return True, "ok"

    def _validate_required_file(
        self,
        required_filepath: str,
        context: Dict[str, Any],
        output_validator: Optional[Callable[[str, Dict[str, Any]], Tuple[bool, str]]] = None,
    ) -> Tuple[bool, str]:
        """统一输出文件验证：存在性 + 可选业务校验。"""
        if not os.path.isfile(required_filepath):
            return (
                False,
                f"输出文件未生成: {required_filepath}\n"
                f"请确保代码末尾有保存数据的语句，例如: df.to_csv('{required_filepath}', index=False)",
            )

        print(f"[CodeAct] 输出文件已生成: {required_filepath}")

        if output_validator:
            ok, msg = output_validator(required_filepath, context)
            if not ok:
                return False, f"输出文件业务校验失败: {msg}"
            print(f"[CodeAct] 输出业务校验通过: {msg}")
            return True, msg

        try:
            df = pd.read_csv(required_filepath)
            if df.shape[0] <= 0 or df.shape[1] <= 0:
                return False, "输出文件为空或无有效列"
            print(f"[CodeAct] 数据验证成功: {df.shape[0]} 行 × {df.shape[1]} 列")
            return True, "ok"
        except Exception as e:
            return False, f"输出文件格式错误: {e}"
    
    def _build_initial_messages(self, task_prompt: str) -> List[BaseMessage]:
        """构建首轮代码生成消息。"""
        system_content = """你是一位专业的 Python 工程师，擅长生成健壮、可直接执行的数据处理和机器学习代码。

你的职责：
1. 根据用户任务生成完整、可执行的 Python 代码
2. 严格使用用户提供的真实数据路径、真实字段和真实约束
3. 输出的代码必须可以直接运行，且包含所有必要导入

代码生成要求：
1. 代码必须完整、可执行，所有括号和引号正确闭合
2. 使用绝对路径处理文件，保存前用 os.makedirs 确保目录存在
3. 在代码末尾输出执行结果
4. 只输出 Python 代码块，用 ```python 和 ``` 包围

常见错误预防：
- 将 numpy 类型转为 Python 原生类型后再做 JSON 序列化
- 使用 .copy() 避免 SettingWithCopyWarning
- 处理可能的空 DataFrame 或缺失列情况

代码格式约束（必须严格遵守）：
- 不要使用 globals().update()
- 核心逻辑必须封装在指定的主函数中（如 clean_data、engineer_features、train_model）
- 文件末尾必须有 if __name__ == "__main__": 块，在其中解析 sys.argv 参数并调用主函数
- import 写在文件顶部，所有 import 语句必须写在单行，不要用括号换行
  （正确: from sklearn.metrics import accuracy_score, f1_score, r2_score）
  （错误: from sklearn.metrics import (\\n    accuracy_score,\\n    f1_score\\n)）"""

        human_content = f"""## 当前任务

{task_prompt}"""

        return [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

    def _build_retry_messages(self, task_prompt: str, code: str, error: str) -> List[BaseMessage]:
        """构建重试代码生成消息。"""
        system_content = """你是一位专业的 Python 工程师，擅长诊断和修复代码错误。

你的职责：
1. 分析上一轮代码的错误原因，按错误类型针对性修复
2. 保留上一轮代码中正常工作的部分，仅修改出错的部分
3. 严格保留用户任务中的真实数据路径、字段和输出要求

错误诊断指南：
- ImportError/ModuleNotFoundError → 检查导入语句，使用标准库替代
- KeyError/列名不存在 → 先打印 df.columns 确认实际列名
- FileNotFoundError → 确认路径正确，用 os.makedirs 创建目录
- TypeError（numpy 序列化）→ 用 int()/float()/str() 转换后再序列化
- SyntaxError → 检查括号、引号、缩进是否正确闭合

输出要求：
1. 只输出 Python 代码块，用 ```python 和 ``` 包围
2. 输出完整修复后代码，不是补丁片段

代码格式约束（必须严格遵守）：
- 不要使用 globals().update()
- 核心逻辑必须封装在指定的主函数中（如 clean_data、engineer_features、train_model）
- 文件末尾必须有 if __name__ == "__main__": 块，在其中解析 sys.argv 参数并调用主函数
- 所有 import 语句必须写在单行，不要用括号换行"""

        human_content = f"""## 原始任务

{task_prompt}

## 上次生成的代码

```python
{code}
```

## 执行错误

{error}

请分析错误原因并重新生成修复后的完整代码。"""

        return [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

    def _serialize_llm_input(self, llm_input: Any) -> str:
        """将最终发送给 LLM 的输入序列化为可审阅文本。"""
        if isinstance(llm_input, str):
            return llm_input

        if isinstance(llm_input, list):
            lines = ["## Final Chat Messages", ""]
            for index, message in enumerate(llm_input, start=1):
                role = getattr(message, "type", message.__class__.__name__).replace("Message", "").lower()
                content = getattr(message, "content", str(message))
                lines.extend([
                    f"### Message {index}",
                    f"- Role: {role}",
                    "",
                    str(content).strip(),
                    "",
                ])
            return "\n".join(lines).strip()

        return str(llm_input)

    def _generate_code(self, prompt: Any, stage: str = "", iteration: int = 1) -> Dict[str, Any]:
        """生成代码"""
        try:
            import time as _time
            full_response = ""
            start_time = datetime.now()
            _t0 = _time.time()
            _ttft = None

            # 尝试流式输出
            try:
                for chunk in self.llm.stream(prompt):
                    if chunk.content:
                        if _ttft is None:
                            _ttft = _time.time() - _t0
                            print(f"\n[CodeAct] 首 token 响应时间 (TTFT): {_ttft:.2f}s")
                        content = chunk.content
                        full_response += content
                        print(content, end="", flush=True)
                        if self._stream_callback:
                            self._stream_callback(StreamEvent(StreamEventType.CONTENT, content))
            except Exception as e:
                # 流式输出失败，回退到同步调用
                print(f"\n[CodeAct] 流式输出失败，回退到同步调用: {e}")
                try:
                    response = self.llm.invoke(prompt)
                    full_response = response.content if hasattr(response, 'content') else str(response)
                    if self._stream_callback:
                        self._stream_callback(StreamEvent(StreamEventType.CONTENT, full_response))
                except Exception as e2:
                    return {"success": False, "error": str(e2)}

            _total = _time.time() - _t0
            _gen_time = _total - (_ttft or 0)
            print(f"\n[CodeAct] 耗时统计 — TTFT: {_ttft:.2f}s | 生成: {_gen_time:.2f}s | 总计: {_total:.2f}s"
                  if _ttft else f"\n[CodeAct] 总耗时: {_total:.2f}s")

            llm_config = self.config_loader.get_llm_config()
            model_name = llm_config.get("model_name", "unknown")
            provider = llm_config.get("provider", "unknown")
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.llm_logger.log_call(
                model_name=model_name,
                provider=provider,
                input_content=self._serialize_llm_input(prompt),
                output_content=full_response,
                latency_ms=latency_ms,
                stage=stage,
                metadata={
                    "call_type": "codeact",
                    "prompt_scope": "final_actual_llm_input",
                    "prompt_format": "chat_messages_system_user",
                    "iteration": iteration,
                }
            )
            
            # 提取代码
            code = self._extract_code(full_response)

            if not code.strip():
                return {"success": False, "error": "无法提取有效代码"}

            # 将生成的代码发送给前端展示
            if self._stream_callback:
                self._stream_callback(StreamEvent(StreamEventType.CODE_GENERATED, code))

            return {"success": True, "code": code}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_code(self, code: str, context: Dict[str, Any] = None, required_output_names: List[str] = None) -> Dict[str, Any]:
        """执行代码（根据配置选择子进程或内联模式）"""
        if self.use_subprocess:
            return self._execute_code_subprocess(code, context, required_output_names)
        return self._execute_code_inline(code, context)

    def _execute_code_subprocess(self, code: str, context: Dict[str, Any] = None, required_output_names: List[str] = None) -> Dict[str, Any]:
        """在子进程中执行代码"""
        result = self.subprocess_executor.execute(
            code=code,
            context=context or {},
            required_output_names=required_output_names or [],
        )

        if result.success:
            return {
                "success": True,
                "output": result.output or "代码执行成功",
                "variables": result.variables,
            }
        else:
            error_msg = result.error or "代码执行失败"
            if result.timed_out:
                error_msg = f"执行超时（{self.timeout}秒）: {error_msg}"
            # 包含 stdout 信息辅助调试
            if result.output:
                error_msg = f"{error_msg}\n\nStdout:\n{result.output}"
            return {
                "success": False,
                "error": error_msg,
                "variables": result.variables,
            }

    def _execute_code_inline(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """在当前进程中执行代码（回退模式，带 stdout/stderr 捕获）"""
        import io
        import contextlib

        local_vars = context.copy() if context else {}
        local_vars['__name__'] = '__main__'
        local_vars['__builtins__'] = __builtins__

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buf):
                with contextlib.redirect_stderr(stderr_buf):
                    exec(code, local_vars)

            for key in ['__name__', '__builtins__']:
                if key in local_vars:
                    del local_vars[key]

            output = stdout_buf.getvalue() or "代码执行成功"
            return {
                "success": True,
                "output": output,
                "variables": local_vars
            }

        except Exception as e:
            stderr_output = stderr_buf.getvalue()
            error_detail = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
            if stderr_output:
                error_detail = f"{error_detail}\n\nStderr:\n{stderr_output}"
            return {
                "success": False,
                "error": error_detail,
                "variables": {}
            }
    
    def _extract_code(self, content: str) -> str:
        """从响应中提取代码"""
        # 尝试匹配 ```python ... ```
        code_match = re.search(r'```python\s*\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 尝试匹配 ```python ... (不完整的代码块)
        code_match = re.search(r'```python\s*\n(.*?)(?:\n```|$)', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 尝试匹配 ``` ... ```
        code_match = re.search(r'```\s*\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 尝试匹配 ```...``` (没有换行的代码块)
        code_match = re.search(r'```(.*?)```', content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            # 如果代码以 python 开头，去掉它
            if code.startswith('python'):
                code = code[6:].strip()
            return code
        
        # 如果包含 import 语句，提取从 import 开始的内容
        if 'import ' in content or 'def ' in content or 'class ' in content:
            # 找到第一个 import、def 或 class 的位置
            import_pos = content.find('import ')
            def_pos = content.find('def ')
            class_pos = content.find('class ')
            
            positions = [p for p in [import_pos, def_pos, class_pos] if p >= 0]
            if positions:
                start_pos = min(positions)
                # 检查是否是 from ... import
                if start_pos > 0 and content[start_pos-5:start_pos] == 'from ':
                    start_pos -= 5
                return content[start_pos:].strip()
        
        return ""
