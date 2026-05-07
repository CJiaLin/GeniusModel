"""
FastAPI 主应用

提供 AutoML 工作流的 API 接口。
本文件仅负责：app 工厂、中间件注册、router 挂载、启动事件。
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from .deps import verify_api_key
from .registry import AppRegistry, _default_registry
from .services.llm_factory import LLMClientError  # re-export for backwards compat

from .routes import (
    workflow,
    confirmation,
    assets,
    sessions,
    chat,
    skills,
    report,
    predict,
)


_CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "AUTOML_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080",
    ).split(",")
    if o.strip()
]


def create_app(registry: AppRegistry = None) -> FastAPI:
    """
    App 工厂。

    Args:
        registry: 注入的 AppRegistry 实例。None 则使用默认全局实例。
                  测试时传入独立实例可隔离状态。
    """
    reg = registry or _default_registry

    application = FastAPI(
        title="GeniusModel API",
        description="基于 ReAct 模式的智能建模 API",
        version="1.0.0",
        dependencies=[Depends(verify_api_key)],
    )
    application.state.registry = reg

    application.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 路由注册
    application.include_router(workflow.router, tags=["workflow"])
    application.include_router(confirmation.router, tags=["confirmation"])
    application.include_router(assets.router, tags=["assets"])
    application.include_router(sessions.router, tags=["sessions"])
    application.include_router(chat.router, tags=["chat"])
    application.include_router(skills.router, tags=["skills"])
    application.include_router(report.router, tags=["report"])
    application.include_router(predict.router, tags=["predict"])

    # 根路由
    @application.get("/")
    async def root():
        return {
            "service": "GeniusModel API",
            "status": "running",
            "version": "1.0.0",
        }

    # 启动事件
    @application.on_event("startup")
    async def on_startup():
        await reg.cleanup_expired()

        async def _periodic_cleanup():
            while True:
                await asyncio.sleep(1800)
                try:
                    await reg.cleanup_expired()
                except Exception as e:
                    print(f"[API] 定期清理异常: {e}")

        asyncio.create_task(_periodic_cleanup())

    return application


# 默认 app 实例（uvicorn / gunicorn 入口）
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
