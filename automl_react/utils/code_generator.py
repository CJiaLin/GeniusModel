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
        self.max_retries = 5  # 添加重试次数
        self.retry_delay = 10  # 添加重试延迟（秒）
        self.timeout = 600  # 添加超时时间（秒）
    
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
        # 构建代码生成提示词 - 只输出代码，不输出思考
        structured_prompt = f"""{prompt}

重要：你只需要输出 Python 代码，不要输出任何思考内容或解释。

直接输出代码，用 ```python 和 ``` 包围。

要求：
1. 只输出代码，不要有任何文字说明
2. 代码必须完整、可执行，不要省略任何部分
3. 代码块必须用 ```python 和 ``` 包围
4. 确保所有括号、引号都正确闭合
5. 如果代码需要保存文件，请使用绝对路径
6. 代码末尾必须有完整的结束
"""
        
        last_error = None
        
        for retry in range(self.max_retries):
            try:
                print(f"[CodeGenerator] 开始流式生成代码 (尝试 {retry + 1}/{self.max_retries})...")
                
                try:
                    # 使用流式输出
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
                        try:
                            response = self.llm.invoke(structured_prompt)
                            full_response = response.content if hasattr(response, 'content') else str(response)
                        except Exception as e2:
                            last_error = str(e2)
                            continue
                    
                    # 检查是否有响应
                    if not full_response.strip():
                        print("[CodeGenerator] 未收到响应")
                        if retry < self.max_retries - 1:
                            print(f"[CodeGenerator] 等待 {self.retry_delay} 秒后重试...")
                            continue
                        else:
                            print("[CodeGenerator] 达到最大重试次数")
                            return CodeGenerationResult(
                                thinking='',
                                code="",
                                success=False,
                                error="达到最大重试次数，请稍后重试"
                            )
                    
                    print()  # 换行
                    
                    # 提取代码
                    code = self._extract_code_fallback(full_response)
                    
                    if code.strip():
                        return CodeGenerationResult(
                            thinking='',
                            code=code,
                            success=True
                        )
                    
                    # 如果 markdown 提取失败，尝试 JSON 解析
                    result = self._parse_json_response(full_response)
                    
                    if result:
                        raw_code = result.get('code', '')
                        clean_code = self._sanitize_code(raw_code)
                        return CodeGenerationResult(
                            thinking=result.get('thinking', ''),
                            code=clean_code,
                            success=bool(clean_code.strip())
                        )
                    
                    # 最后尝试从部分 JSON 中提取
                    code_match = re.search(r'"code"\s*:\s*"(.*?)"\s*,?\s*\}?', full_response, re.DOTALL)
                    if code_match:
                        extracted = code_match.group(1)
                        extracted = extracted.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
                        clean_code = self._sanitize_code(extracted)
                        if clean_code.strip():
                            return CodeGenerationResult(
                                thinking='从部分 JSON 中提取代码',
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
        # 尝试匹配 ```python ... ``` (完整代码块)
        code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 尝试匹配 ```python ... (不完整的代码块，没有结束的 ```)
        code_match = re.search(r'```python\n(.*?)(?:\n```|$)', content, re.DOTALL)
        if code_match:
            extracted = code_match.group(1).strip()
            # 如果提取的代码看起来不完整，尝试获取更多内容
            if extracted and not self._is_code_complete(extracted):
                # 尝试获取从 ```python 开始到文件结束的所有内容
                python_start = content.find('```python')
                if python_start >= 0:
                    extracted = content[python_start + 10:].strip()
                    # 移除可能的结束标记
                    if extracted.endswith('```'):
                        extracted = extracted[:-3].strip()
            return extracted
        
        # 尝试匹配 ``` ... ```
        code_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 尝试匹配 ``` ... (不完整的代码块)
        code_match = re.search(r'```\n(.*?)(?:\n```|$)', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 如果包含 import 语句，尝试提取从 import 开始的所有内容
        if 'import ' in content:
            import_start = content.find('import ')
            # 找到 import 前面可能的 'from ' 或直接从 import 开始
            if import_start > 0 and content[import_start-5:import_start] == 'from ':
                import_start -= 5
            return content[import_start:].strip()
        
        # 返回空字符串
        return ""
    
    def _is_code_complete(self, code: str) -> bool:
        """检查代码是否完整（简单的括号匹配检查）"""
        # 统计括号数量
        open_parens = code.count('(') - code.count(')')
        open_brackets = code.count('[') - code.count(']')
        open_braces = code.count('{') - code.count('}')
        
        # 如果有未闭合的括号，代码不完整
        if open_parens > 0 or open_brackets > 0 or open_braces > 0:
            return False
        
        # 检查是否有未闭合的字符串
        # 简单检查：如果代码以奇数个引号结尾，可能不完整
        lines = code.strip().split('\n')
        if lines:
            last_line = lines[-1]
            # 检查是否以不完整的表达式结尾
            if last_line.rstrip().endswith(('(', '[', '{', ',', '+', '-', '*', '/', '=', ':')):
                return False
        
        return True

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
