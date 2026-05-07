"""
Chat 路由

/chat 端点
"""

import json
import asyncio

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from ..deps import get_registry
from ..registry import AppRegistry
from ..services.llm_factory import create_llm_client

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = "claude-sonnet-4-20250514-thinking"


@router.post("/chat")
async def chat(request: ChatRequest, registry: AppRegistry = Depends(get_registry)):
    """对话接口"""
    session = await registry.get_session(request.session_id)
    llm = create_llm_client(request.model)

    if not llm:
        raise HTTPException(status_code=500, detail="LLM 客户端创建失败")

    try:
        full_response = ""
        try:
            for chunk in llm.stream(request.message):
                if chunk.content:
                    full_response += chunk.content
        except Exception as e:
            response = llm.invoke(request.message)
            full_response = response.content if hasattr(response, 'content') else str(response)

        return {
            "success": True,
            "response": full_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/stream")
async def chat_stream(session_id: str, message: str, model: str = None):
    """流式对话接口"""
    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'start', 'content': '开始处理...'})}\n\n"
            await asyncio.sleep(0.1)

            llm = create_llm_client(model)
            if not llm:
                yield f"data: {json.dumps({'type': 'error', 'content': 'LLM 客户端创建失败'})}\n\n"
                return

            try:
                for chunk in llm.stream(message):
                    if chunk.content:
                        yield f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"
                        await asyncio.sleep(0.01)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'流式输出失败: {str(e)}'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'done', 'content': '处理完成'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
