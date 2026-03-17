"""
AutoML API 主入口

整合所有路由，提供统一的API服务。
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import chat_api

app = FastAPI(title="AutoML API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_api.app.router, tags=["chat"])


@app.get("/")
async def root():
    return {"message": "AutoML API Server", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
