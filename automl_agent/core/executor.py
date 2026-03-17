"""
代码执行器模块

本模块提供了在隔离环境中安全执行Python代码的功能。
主要用于大模型生成的代码在后台建模环境中的执行。

主要功能：
1. 代码编译和执行
2. stdout/stderr 捕获
3. 执行结果序列化返回
4. 变量空间管理
5. 错误处理和堆栈跟踪

注意：此执行器使用基础的exec()实现，生产环境中应考虑使用
更安全的沙箱环境（如Docker容器或专门的代码沙箱服务）。
"""

import io
import sys
import json
import traceback
import functools
from typing import Any, Callable
from contextlib import redirect_stdout, redirect_stderr


class CodeExecutor:
    """
    Python代码执行器类
    
    用于在当前Python进程中安全地执行动态生成的代码。
    提供了输出捕获、变量管理和错误处理功能。
    
    Attributes:
        timeout: 代码执行超时时间（秒），默认300秒
        max_output_size: 最大输出大小（字节），默认100KB
        _namespace: 代码执行的变量命名空间，用于存储变量状态
    
    Example:
        >>> executor = CodeExecutor()
        >>> result = executor.execute("import pandas as pd\\nresult = pd.DataFrame({'a': [1,2,3]})")
        >>> print(result['success'])
        True
    """
    
    def __init__(self, timeout: int = 300, max_output_size: int = 100000):
        """
        初始化代码执行器
        
        Args:
            timeout: 执行超时时间（秒），默认300秒
            max_output_size: 最大输出大小（字节），默认100KB
        """
        self.timeout = timeout
        self.max_output_size = max_output_size
        self._namespace = {}  # 变量命名空间，用于在多次执行间保持状态
    
    def execute(self, code: str) -> dict[str, Any]:
        """
        执行Python代码
        
        在隔离的变量空间中执行给定的Python代码，
        捕获stdout/stderr输出，并返回执行结果。
        
        Args:
            code: 要执行的Python代码字符串
            
        Returns:
            dict: 执行结果字典，包含以下键：
                - success: bool，执行是否成功
                - output: str，捕获的标准输出
                - error: dict，错误信息（如果执行失败）
                - data: Any，执行结果（如果代码中定义了result变量）
        
        Example:
            >>> result = executor.execute("x = 1 + 1\\nresult = x * 2")
            >>> print(result['data'])
            4
        """
        # 初始化结果字典
        result = {
            "success": False,      # 执行是否成功
            "output": "",          # 标准输出
            "error": None,         # 错误信息
            "data": None           # 返回数据
        }
        
        # 保存原始的stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        # 创建字符串IO对象用于捕获输出
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # 使用context manager重定向输出到捕获对象
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # 编译代码以提前发现语法错误
                compiled = compile(code, "<string>", "exec")
                # 在命名空间中执行代码
                exec(compiled, self._namespace)
            
            # 获取捕获的输出
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            # 将输出添加到结果中，截断到最大大小
            if stdout_output:
                result["output"] = stdout_output[:self.max_output_size]
            if stderr_output:
                result["output"] += "\n[STDERR]: " + stderr_output[:self.max_output_size]
            
            # 如果代码中定义了result变量，将其作为返回数据
            if "result" in self._namespace:
                result["data"] = self._namespace["result"]
            
            # 标记执行成功
            result["success"] = True
            
        except Exception as e:
            # 捕获并记录执行过程中的错误
            result["error"] = {
                "type": type(e).__name__,      # 错误类型
                "message": str(e),              # 错误消息
                "traceback": traceback.format_exc()  # 完整堆栈跟踪
            }
            
        finally:
            # 恢复原始的stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        return result
    
    def execute_with_context(self, code: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        使用预定义上下文执行代码
        
        在执行代码前，先将提供的上下文变量添加到命名空间，
        使这些变量可以在代码中直接使用。
        
        Args:
            code: 要执行的Python代码字符串
            context: 上下文变量字典，会被添加到执行命名空间
            
        Returns:
            dict: execute()方法的返回结果
            
        Example:
            >>> context = {"df": pd.DataFrame({'a': [1,2,3]})}
            >>> result = executor.execute_with_context("result = df['a'].sum()", context)
        """
        # 更新命名空间，添加上下文变量
        self._namespace.update(context)
        # 执行代码
        return self.execute(code)
    
    def reset(self):
        """
        重置执行器状态
        
        清空所有变量，释放内存，返回到初始状态。
        适用于开始新的建模任务时清理之前的状态。
        """
        self._namespace = {}
    
    def get_variable(self, name: str) -> Any:
        """
        获取命名空间中的变量
        
        Args:
            name: 变量名
            
        Returns:
            Any: 变量的值，如果不存在则返回None
        """
        return self._namespace.get(name)
    
    def set_variable(self, name: str, value: Any):
        """
        设置命名空间中的变量
        
        Args:
            name: 变量名
            value: 变量的值
        """
        self._namespace[name] = value
    
    def clear_variable(self, name: str):
        """
        清除命名空间中的指定变量
        
        Args:
            name: 要清除的变量名
        """
        if name in self._namespace:
            del self._namespace[name]
