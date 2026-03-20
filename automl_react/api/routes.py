"""
API 路由模块

提供 FastAPI 路由
"""

import json
import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..agents.orchestrator import AutoMLOrchestrator


# 请求模型
class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = "claude-sonnet-4-20250514-thinking"
    data_path: str = None
    target_column: str = None
    task_type: str = "classification"


# 存储会话
_sessions: dict = {}


def create_chat_router(get_llm_client) -> APIRouter:
    """创建聊天路由"""
    
    router = APIRouter()
    
    @router.post("/chat/stream")
    async def chat_stream(request: ChatRequest):
        """流式聊天接口"""
        
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                # 获取或创建会话
                if request.session_id not in _sessions:
                    llm = get_llm_client(
                        session_id=request.session_id,
                        stage="chat"
                    )
                    orchestrator = AutoMLOrchestrator(llm=llm, verbose=False)
                    
                    # 设置上下文
                    if request.data_path and request.target_column:
                        orchestrator.set_context(
                            request.data_path,
                            request.target_column,
                            request.task_type
                        )
                    
                    _sessions[request.session_id] = orchestrator
                else:
                    orchestrator = _sessions[request.session_id]
                
                # 发送开始消息
                yield f"data: {json.dumps({'type': 'start', 'content': '开始处理...'})}\n\n"
                await asyncio.sleep(0.1)
                
                # 执行对话
                result = orchestrator.chat(request.message)
                
                # 发送结果
                if result.get("success"):
                    answer = result.get("answer", "")
                    
                    # 模拟流式输出
                    chunks = answer.split("\n")
                    for chunk in chunks:
                        if chunk.strip():
                            newline = "\n"
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk + newline})}\n\n"
                            await asyncio.sleep(0.05)
                    
                    # 发送完成消息
                    yield f"data: {json.dumps({'type': 'done', 'content': '处理完成'})}\n\n"
                else:
                    error = result.get("answer", "处理失败")
                    yield f"data: {json.dumps({'type': 'error', 'content': error})}\n\n"
                    
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        
        from fastapi.responses import StreamingResponse
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    @router.post("/pipeline")
    async def run_pipeline(request: ChatRequest):
        """执行完整建模流程"""
        
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                # 创建 LLM 客户端
                llm = get_llm_client(
                    session_id=request.session_id,
                    stage="pipeline"
                )
                
                # 创建总控器
                orchestrator = AutoMLOrchestrator(llm=llm, verbose=False)
                
                if not request.data_path or not request.target_column:
                    yield f"data: {json.dumps({'type': 'error', 'content': '请提供数据路径和目标列'})}\n\n"
                    return
                
                # 设置上下文
                orchestrator.set_context(
                    request.data_path,
                    request.target_column,
                    request.task_type
                )
                
                # 存储会话
                _sessions[request.session_id] = orchestrator
                
                # 发送开始消息
                yield f"data: {json.dumps({'type': 'start', 'content': '开始完整建模流程...'})}\n\n"
                await asyncio.sleep(0.1)
                
                # 执行完整流程
                result = orchestrator.run_full_pipeline()
                
                # 发送结果
                if result.get("success"):
                    answer = result.get("answer", "")
                    
                    # 模拟流式输出
                    chunks = answer.split("\n")
                    for chunk in chunks:
                        if chunk.strip():
                            newline = "\n"
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk + newline})}\n\n"
                            await asyncio.sleep(0.05)
                    
                    done_msg = "建模完成"
                    yield f"data: {json.dumps({'type': 'done', 'content': done_msg})}\n\n"
                else:
                    error = result.get("answer", "建模失败")
                    yield f"data: {json.dumps({'type': 'error', 'content': error})}\n\n"
                    
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        
        from fastapi.responses import StreamingResponse
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    @router.get("/sessions/{session_id}/status")
    async def get_session_status(session_id: str):
        """获取会话状态"""
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        orchestrator = _sessions[session_id]
        return orchestrator.get_status()
    
    return router
