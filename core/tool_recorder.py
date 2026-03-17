"""
MCP 工具增强 - 添加调用记录功能
"""
from functools import wraps
from typing import Any, Callable, Optional
from core.state import PipelineState


def record_tool_call(tool_name: str, state_attr: str = "state"):
    """
    装饰器：记录工具调用到 PipelineState
    
    Args:
        tool_name: 工具名称
        state_attr: state 对象的属性名（默认为 "state"）
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取 state 对象
            state = None
            if args:  # 如果有位置参数，第一个参数可能是 state
                # 尝试找到 PipelineState 对象
                for arg in args:
                    if isinstance(arg, PipelineState):
                        state = arg
                        break
            
            if state is None and state_attr in kwargs:
                state = kwargs[state_attr]
            
            # 执行原函数
            result = func(*args, **kwargs)
            
            # 记录工具调用
            if state is not None:
                # 获取参数信息
                param_info = {
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                    "function_name": func.__name__
                }
                
                # 尝试获取更详细的参数信息
                try:
                    import inspect
                    sig = inspect.signature(func)
                    bound_args = sig.bind_partial(*args, **kwargs)
                    bound_args.apply_defaults()
                    param_info["params"] = dict(bound_args.arguments)
                except:
                    pass  # 如果获取参数失败，使用基本参数信息
                
                state.log_tool_call(tool_name, param_info, result)
            
            return result
        return wrapper
    return decorator


# 为了与现有工具注册系统兼容，提供一个包装函数
def wrap_tool_with_recording(tool_func: Callable, tool_name: str):
    """
    包装工具函数以添加记录功能
    
    Args:
        tool_func: 原始工具函数
        tool_name: 工具名称
        
    Returns:
        包装后的工具函数
    """
    return record_tool_call(tool_name)(tool_func)
