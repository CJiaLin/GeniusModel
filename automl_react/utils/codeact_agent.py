"""
CodeAct Agent 模块

基于 CodeAct 模式的代码生成和执行框架。
CodeAct 模式：LLM 生成代码 → 执行代码 → 观察执行结果 → 根据结果调整代码（循环）
"""

import json
import os
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd


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
    
    def __init__(self, llm: Any = None, max_iterations: int = 5, timeout: int = 300):
        self.llm = llm
        self.max_iterations = max_iterations
        self.timeout = timeout
    
    def generate_and_execute(
        self,
        task_prompt: str,
        context: Dict[str, Any] = None,
        required_outputs: List[str] = None,
        required_filepath: str = None,
    ) -> CodeActResult:
        """
        生成并执行代码（CodeAct 模式）
        
        Args:
            task_prompt: 任务提示词
            context: 执行上下文变量
            required_outputs: 需要验证的输出变量名
            required_filepath: 需要验证的输出文件路径
            
        Returns:
            CodeActResult
        """
        start_time = time.time()
        
        conversation_history = []
        current_code = ""
        last_error = None
        
        for iteration in range(self.max_iterations):
            print(f"\n[CodeAct] 迭代 {iteration + 1}/{self.max_iterations}")
            
            # 构建提示词
            if iteration == 0:
                prompt = self._build_initial_prompt(task_prompt)
            else:
                print(last_error)
                prompt = self._build_retry_prompt(task_prompt, current_code, last_error)
            
            # 生成代码
            print(f"[CodeAct] 生成代码...")
            code_result = self._generate_code(prompt)
            
            if not code_result["success"]:
                last_error = code_result.get("error", "代码生成失败")
                continue
            
            current_code = code_result["code"]
            
            # 执行代码
            print(f"[CodeAct] 执行代码...")
            exec_result = self._execute_code(current_code, context)
            
            if exec_result["success"]:
                # 检查必需的输出变量
                if required_outputs:
                    missing = [var for var in required_outputs if var not in exec_result.get("variables", {})]
                    if missing:
                        last_error = f"缺少必需的输出变量: {missing}"
                        continue
                
                # 检查必需的输出文件
                if required_filepath:
                    if os.path.isfile(required_filepath):
                        print(f"[CodeAct] 输出文件已生成: {required_filepath}")
                        try:
                            df = pd.read_csv(required_filepath)
                            print(f"[CodeAct] 数据验证成功: {df.shape[0]} 行 × {df.shape[1]} 列")
                        except Exception as e:
                            last_error = f"输出文件格式错误: {e}"
                            continue
                    else:
                        last_error = f"输出文件未生成: {required_filepath}\n请确保代码末尾有保存数据的语句，例如: df.to_csv('{required_filepath}', index=False)"
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
    
    def _build_initial_prompt(self, task_prompt: str) -> str:
        """构建初始提示词"""
        return f"""{task_prompt}

请生成完整的、可执行的 Python 代码来完成任务。

要求：
1. 分析问题并确定需要编写什么代码
2. 编写能解决问题的Python代码
3. 代码必须完整、可执行
4. 包含所有必要的 import 语句
5. 使用绝对路径处理文件
6. 确保所有括号、引号正确闭合
7. 在代码末尾输出执行结果

直接输出 Python 代码，用 ```python 和 ``` 包围。"""
    
    def _build_retry_prompt(self, task_prompt: str, code: str, error: str) -> str:
        """构建重试提示词"""
        return f"""{task_prompt}

上次生成的代码执行失败，请修复问题。

## 上次生成的代码：
```python
{code}
```

## 执行错误：
{error}

请分析错误原因，修复代码中的问题，重新生成完整的、可执行的 Python 代码。

要求：
1. 仔细检查代码语法
2. 确保所有导入的库都已导入
3. 确保文件路径正确
4. 确保变量名一致
5. 处理可能的异常情况

直接输出修复后的 Python 代码，用 ```python 和 ``` 包围。"""
    
    def _generate_code(self, prompt: str) -> Dict[str, Any]:
        """生成代码"""
        try:
            full_response = ""
            
            # 尝试流式输出
            try:
                for chunk in self.llm.stream(prompt):
                    if chunk.content:
                        content = chunk.content
                        full_response += content
                        print(content, end="", flush=True)
            except Exception as e:
                # 流式输出失败，回退到同步调用
                print(f"\n[CodeAct] 流式输出失败，回退到同步调用: {e}")
                try:
                    response = self.llm.invoke(prompt)
                    full_response = response.content if hasattr(response, 'content') else str(response)
                except Exception as e2:
                    return {"success": False, "error": str(e2)}
            
            print()  # 换行
            
            # 提取代码
            code = self._extract_code(full_response)
            
            if not code.strip():
                return {"success": False, "error": "无法提取有效代码"}
            
            return {"success": True, "code": code}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_code(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行代码"""
        # 准备执行环境
        local_vars = context.copy() if context else {}
        local_vars['__name__'] = '__executor__'
        local_vars['__builtins__'] = __builtins__
        
        try:
            # 使用 exec 执行代码
            exec(code, local_vars)
            
            # 删除内部变量
            for key in ['__Name__', '__builtins__', '__executor__']:
                if key in local_vars:
                    del local_vars[key]
            
            return {
                "success": True,
                "output": "代码执行成功",
                "variables": local_vars
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}",
                "variables": {}
            }
    
    def _extract_code(self, content: str) -> str:
        """从响应中提取代码"""
        # 尝试匹配 ```python ... ```
        code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 尝试匹配 ```python ... (不完整的代码块)
        code_match = re.search(r'```python\n(.*?)(?:\n```|$)', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 尝试匹配 ``` ... ```
        code_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 如果包含 import 语句，提取从 import 开始的内容
        if 'import ' in content:
            import_start = content.find('import ')
            if import_start > 0 and content[import_start-5:import_start] == 'from ':
                import_start -= 5
            return content[import_start:].strip()
        
        return ""
