"""
代码执行MCP服务器模块

本模块提供了一个基于FastAPI的MCP（Model Context Protocol）服务器，
用于通过HTTP API远程执行Python代码。

主要功能：
1. 提供RESTful API接口执行Python代码
2. 支持代码执行上下文管理
3. 支持变量空间管理（设置、获取、重置）
4. 异步请求处理

使用方式：
    启动服务器：python -m automl_agent.mcp.execution_mcp
    API端点：http://localhost:8001

API接口：
    POST /execute - 执行Python代码
    POST /execute/variable - 设置变量
    GET /execute/variable/{name} - 获取变量值
    POST /execute/reset - 重置执行器状态
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Optional
import uvicorn

from ..core.executor import CodeExecutor


# 创建FastAPI应用实例
app = FastAPI(
    title="Code Execution MCP Server",  # API标题
    description="用于远程执行Python代码的MCP服务器",  # API描述
    version="1.0.0"  # API版本
)

# 创建代码执行器实例（全局单例）
executor = CodeExecutor()


class ExecuteRequest(BaseModel):
    """
    代码执行请求模型
    
    定义了执行代码所需的请求参数。
    
    Attributes:
        code: 要执行的Python代码字符串
        context: 可选的上下文变量字典，会被添加到执行环境
        timeout: 执行超时时间（秒），默认300秒
    """
    code: str                                    # 要执行的代码
    context: Optional[dict[str, Any]] = None      # 执行上下文变量
    timeout: Optional[int] = 300                 # 超时时间（秒）


class ExecuteResponse(BaseModel):
    """
    代码执行响应模型
    
    定义了执行结果的返回格式。
    
    Attributes:
        success: 执行是否成功
        output: 捕获的标准输出
        error: 错误信息（如果执行失败）
        data: 返回数据（如果代码中定义了result变量）
    """
    success: bool                              # 是否成功
    output: str                                # 标准输出
    error: Optional[dict[str, Any]] = None     # 错误信息
    data: Optional[Any] = None                 # 返回数据


@app.post("/execute", response_model=ExecuteResponse)
async def execute_code(request: ExecuteRequest):
    """
    执行Python代码的API端点
    
    接收Python代码并执行，返回执行结果。
    
    Args:
        request: ExecuteRequest对象，包含代码和上下文
        
    Returns:
        ExecuteResponse对象，包含执行结果
        
    Raises:
        HTTPException: 如果执行过程中发生错误
        
    Example:
        请求:
        {
            "code": "result = 1 + 1",
            "context": null
        }
        
        响应:
        {
            "success": true,
            "output": "",
            "error": null,
            "data": 2
        }
    """
    try:
        # 根据是否有上下文选择执行方式
        if request.context:
            # 使用上下文变量执行代码
            result = executor.execute_with_context(request.code, request.context)
        else:
            # 直接执行代码
            result = executor.execute(request.code)
        
        # 返回执行结果
        return ExecuteResponse(**result)
        
    except Exception as e:
        # 捕获并处理异常
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute/variable")
async def set_variable(name: str, value: Any):
    """
    设置变量的API端点
    
    在执行器的命名空间中设置一个变量。
    
    Args:
        name: 变量名
        value: 变量值
        
    Returns:
        状态信息字典
        
    Example:
        请求: POST /execute/variable?name=df&value={"data": [...]}
    """
    # 在执行器中设置变量
    executor.set_variable(name, value)
    return {"status": "ok"}


@app.get("/execute/variable/{name}")
async def get_variable(name: str):
    """
    获取变量的API端点
    
    从执行器的命名空间中获取指定变量的值。
    
    Args:
        name: 变量名
        
    Returns:
        包含变量名和值的字典
        
    Example:
        响应: {"name": "df", "value": {...}}
    """
    # 从执行器获取变量
    value = executor.get_variable(name)
    return {"name": name, "value": value}


@app.post("/execute/reset")
async def reset_executor():
    """
    重置执行器的API端点
    
    清空执行器的所有变量和状态，返回初始状态。
    
    Returns:
        状态信息字典
        
    Example:
        响应: {"status": "reset"}
    """
    # 重置执行器
    executor.reset()
    return {"status": "reset"}


# 主程序入口
if __name__ == "__main__":
    """
    启动MCP服务器
    
    使用uvicorn运行FastAPI应用，监听0.0.0.0:8001
    
    使用方式:
        python execution_mcp.py
    """
    uvicorn.run(
        app,                    # FastAPI应用实例
        host="0.0.0.0",        # 监听地址
        port=8001              # 监听端口
    )
