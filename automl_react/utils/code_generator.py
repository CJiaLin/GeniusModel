"""
代码生成器模块

提供结构化代码生成和执行验证功能
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CodeGenerationResult:
    """代码生成结果"""
    thinking: str
    code: str
    success: bool
    error: Optional[str] = None


@dataclass
class CodeExecutionResult:
    """代码执行结果"""
    success: bool
    output: str
    error: Optional[str] = None
    variables: Dict[str, Any] = None
    execution_time: float = 0.0


class CodeGenerator:
    """
    代码生成器
    
    提供结构化代码生成和执行验证功能
    """
    
    def __init__(self, llm: Any = None):
        self.llm = llm
        self.max_retries = 3
    
    def generate_code(
        self,
        prompt: str,
        context: Dict[str, Any] = None,
        stage: str = ""
    ) -> CodeGenerationResult:
        """
        生成代码（结构化输出）
        
        Args:
            prompt: 生成提示词
            context: 上下文信息
            stage: 工作流阶段
            
        Returns:
            CodeGenerationResult 包含 thinking 和 code
        """
        # 构建结构化输出提示词
        structured_prompt = f"""{prompt}

重要：你必须以 JSON 格式返回结果，格式如下：
{{
    "thinking": "你的思考过程，包括分析、设计思路等",
    "code": "完整的、可执行的 Python 代码，不要包含 markdown 代码块标记"
}}

要求：
1. code 字段必须包含完整的、可执行的 Python 代码
2. 不要包含 ```python 或 ``` 标记
3. 代码必须能够直接执行，不要有任何说明文字
4. 如果代码需要保存文件，请使用绝对路径
"""
        
        try:
            # 使用流式输出
            print(f"[CodeGenerator] 开始流式生成代码...")
            full_response = ""
            
            try:
                for chunk in self.llm.stream(structured_prompt):
                    if chunk.content:
                        content = chunk.content
                        full_response += content
                        print(content, end="", flush=True)
            except Exception as e:
                # 如果流式输出失败，回退到同步调用
                print(f"\n[CodeGenerator] 流式输出失败，回退到同步调用: {e}")
                response = self.llm.invoke(structured_prompt)
                full_response = response.content if hasattr(response, 'content') else str(response)
            
            print()  # 换行
            
            # 尝试解析 JSON
            result = self._parse_json_response(full_response)
            
            if result:
                raw_code = result.get('code', '')
                clean_code = self._sanitize_code(raw_code)
                return CodeGenerationResult(
                    thinking=result.get('thinking', ''),
                    code=clean_code,
                    success=bool(clean_code.strip())
                )
            else:
                # 如果 JSON 解析失败，尝试多种方法提取代码
                # 方法1: 尝试从部分 JSON 中提取 code 字段
                import re
                code_match = re.search(r'"code"\s*:\s*"(.*?)"\s*,?\s*\}?', full_response, re.DOTALL)
                if code_match:
                    extracted = code_match.group(1)
                    # 处理转义字符
                    extracted = extracted.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
                    clean_code = self._sanitize_code(extracted)
                    if clean_code.strip():
                        return CodeGenerationResult(
                            thinking='从部分 JSON 中提取代码',
                            code=clean_code,
                            success=True
                        )
                
                # 方法2: 尝试提取 markdown 代码块
                code = self._sanitize_code(self._extract_code_fallback(full_response))
                if code.strip():
                    return CodeGenerationResult(
                        thinking='JSON 解析失败，使用备用提取方法',
                        code=code,
                        success=True
                    )
                
                # 方法3: 如果响应中包含 Python 代码特征，直接返回
                if 'import pandas' in full_response or 'import numpy' in full_response:
                    # 尝试提取从 import 开始到文件结束的内容
                    import_start = full_response.find('import ')
                    if import_start > 0:
                        potential_code = full_response[import_start-7:]  # 包含 'import' 前面的 'from ' 或 'import '
                        clean_code = self._sanitize_code(potential_code)
                        if clean_code.strip():
                            return CodeGenerationResult(
                                thinking='直接提取代码内容',
                                code=clean_code,
                                success=True
                            )
                
                return CodeGenerationResult(
                    thinking='无法提取有效代码',
                    code='',
                    success=False
                )
                
        except Exception as e:
            return CodeGenerationResult(
                thinking='',
                code='',
                success=False,
                error=str(e)
            )
    
    def generate_code_with_validation(
        self,
        prompt: str,
        context: Dict[str, Any] = None,
        stage: str = "",
        required_outputs: List[str] = None
    ) -> Tuple[str, CodeExecutionResult]:
        """
        生成代码并验证执行
        
        Args:
            prompt: 生成提示词
            context: 上下文信息
            stage: 工作流阶段
            required_outputs: 需要验证的输出变量名
            
        Returns:
            (代码, 执行结果)
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            # 生成代码
            if attempt == 0:
                gen_prompt = prompt
            else:
                # 重试时添加上次错误信息
                gen_prompt = f"""{prompt}

上次生成的代码执行失败，错误信息：
{last_error}

请修复代码中的问题，重新生成完整的、可执行的 Python 代码。
注意：
1. 仔细检查代码语法
2. 确保所有导入的库都已导入
3. 确保文件路径正确
4. 确保变量名一致
"""
            
            result = self.generate_code(gen_prompt, context, stage)
            
            if not result.success:
                last_error = result.error
                continue
            
            # 执行代码验证
            exec_result = self.execute_code(result.code, context)
            
            if exec_result.success:
                # 检查必需的输出变量
                if required_outputs:
                    missing = [var for var in required_outputs if var not in exec_result.variables]
                    if missing:
                        last_error = f"缺少必需的输出变量: {missing}"
                        continue
                
                return result.code, exec_result
            else:
                last_error = exec_result.error
        
        # 所有重试都失败
        return result.code if result.success else "", CodeExecutionResult(
            success=False,
            output="",
            error=f"经过 {self.max_retries} 次尝试仍无法生成可执行代码。最后一次错误: {last_error}",
            variables={}
        )
    
    def execute_code(
        self,
        code: str,
        context: Dict[str, Any] = None,
        timeout: int = 60
    ) -> CodeExecutionResult:
        """
        执行 Python 代码
        
        Args:
            code: Python 代码
            context: 执行上下文变量
            timeout: 超时时间（秒）
            
        Returns:
            CodeExecutionResult
        """
        import traceback
        import time
        
        start_time = time.time()
        
        # 准备执行环境
        local_vars = context.copy() if context else {}
        local_vars['__name__'] = '__main__'
        
        # 捕获输出
        import io
        import sys
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # 重定向输出
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # 执行代码
            exec(code, local_vars)
            
            # 恢复输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            execution_time = time.time() - start_time
            
            return CodeExecutionResult(
                success=True,
                output=stdout_capture.getvalue(),
                error=None,
                variables={k: v for k, v in local_vars.items() if not k.startswith('__')},
                execution_time=execution_time
            )
            
        except Exception as e:
            # 恢复输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            execution_time = time.time() - start_time
            
            return CodeExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=f"{str(e)}\n{traceback.format_exc()}",
                variables={k: v for k, v in local_vars.items() if not k.startswith('__')},
                execution_time=execution_time
            )
    
    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """解析 JSON 响应"""
        try:
            # 尝试直接解析
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 JSON 块
        try:
            # 匹配 ```json ... ``` 格式
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # 匹配 ``` ... ``` 格式
            json_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # 匹配 { ... } 格式
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
                
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _extract_code_fallback(self, content: str) -> str:
        """备用代码提取方法"""
        # 尝试匹配 ```python ... ```
        code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 尝试匹配 ``` ... ```
        code_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 如果无法提取代码，返回空字符串
        return ""

    def _sanitize_code(self, code: str) -> str:
        """
        清理代码中的 Markdown 标记和多余包装，确保是纯 Python 源码。
        """
        if not code:
            return ""

        cleaned = code.strip()

        # 如果代码看起来像是 JSON 格式（以 { 开头），尝试提取其中的 code 字段
        if cleaned.startswith('{') and ('"code"' in cleaned or "'code'" in cleaned):
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and 'code' in parsed:
                    extracted_code = parsed['code']
                    if isinstance(extracted_code, str):
                        return extracted_code
                    return str(extracted_code)
            except json.JSONDecodeError:
                # JSON 解析失败，尝试使用正则表达式提取 code 字段
                import re
                code_match = re.search(r'"code"\s*:\s*"(.*?)"\s*,?\s*\}', cleaned, re.DOTALL)
                if code_match:
                    extracted = code_match.group(1)
                    # 处理转义字符
                    extracted = extracted.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
                    return extracted

        # 移除常见的 Markdown 代码块围栏行
        lines = []
        for line in cleaned.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                continue
            lines.append(line)

        cleaned = "\n".join(lines).strip()

        return cleaned
