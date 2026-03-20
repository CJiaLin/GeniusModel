"""
代码执行器模块

安全地执行生成的 Python 代码，并捕获输出和结果
"""

import os
import sys
import io
import contextlib
import traceback
from typing import Dict, Any, Optional
from pathlib import Path


class CodeExecutionResult:
    """代码执行结果"""
    
    def __init__(self):
        self.success: bool = False
        self.output: str = ""
        self.error: Optional[str] = None
        self.variables: Dict[str, Any] = {}
        self.execution_time: float = 0.0


class CodeExecutor:
    """
    代码执行器
    
    安全地执行 Python 代码，捕获输出和变量
    """
    
    def __init__(self, timeout: int = 300):
        self.timeout = timeout
    
    def execute(self, code: str, context: Dict[str, Any] = None) -> CodeExecutionResult:
        """
        执行代码
        
        Args:
            code: 要执行的 Python 代码
            context: 预定义的上下文变量
            
        Returns:
            CodeExecutionResult: 执行结果
        """
        import time
        
        result = CodeExecutionResult()
        start_time = time.time()
        
        # 创建执行环境
        exec_globals = {
            '__builtins__': __builtins__,
            'pd': None,
            'np': None,
            'sklearn': None,
        }
        
        # 添加常用导入
        try:
            import pandas as pd
            import numpy as np
            exec_globals['pd'] = pd
            exec_globals['np'] = np
            
            # sklearn 相关
            from sklearn import preprocessing, model_selection, metrics
            exec_globals['preprocessing'] = preprocessing
            exec_globals['model_selection'] = model_selection
            exec_globals['metrics'] = metrics
            
        except ImportError as e:
            result.error = f"缺少必要的依赖: {e}"
            return result
        
        # 添加上下文变量
        if context:
            exec_globals.update(context)
        
        # 捕获输出
        output_buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(output_buffer):
                with contextlib.redirect_stderr(output_buffer):
                    exec(code, exec_globals)
            
            result.success = True
            result.output = output_buffer.getvalue()
            
            # 提取关键变量
            key_vars = ['df', 'df_cleaned', 'df_features', 'model', 'metrics', 'X', 'y', 'X_train', 'X_test', 'y_train', 'y_test']
            for var in key_vars:
                if var in exec_globals:
                    result.variables[var] = exec_globals[var]
                    
        except Exception as e:
            result.success = False
            result.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result.output = output_buffer.getvalue()
        
        result.execution_time = time.time() - start_time
        return result


def execute_code_safely(code: str, context: Dict[str, Any] = None, timeout: int = 300) -> Dict[str, Any]:
    """
    便捷函数：安全执行代码
    
    Args:
        code: Python 代码
        context: 上下文变量
        timeout: 超时时间
        
    Returns:
        执行结果字典
    """
    executor = CodeExecutor(timeout=timeout)
    result = executor.execute(code, context)
    
    return {
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "variables": {k: type(v).__name__ for k, v in result.variables.items()},
        "execution_time": result.execution_time
    }
