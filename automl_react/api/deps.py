"""
API 公共依赖模块

认证、session_id 校验、registry 注入等 FastAPI 依赖
"""

import os
import re as _re

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader, APIKeyQuery


# ==================== API Key 认证 ====================

_API_KEY = os.environ.get("AUTOML_API_KEY")
_header_scheme = APIKeyHeader(name="Authorization", auto_error=False)
_query_scheme = APIKeyQuery(name="api_key", auto_error=False)


async def verify_api_key(
    header: str = Security(_header_scheme),
    query: str = Security(_query_scheme),
):
    """API Key 认证依赖。未设置 AUTOML_API_KEY 环境变量时放行所有请求。"""
    if not _API_KEY:
        return
    token = None
    if header:
        token = header.removeprefix("Bearer ").strip()
    if not token:
        token = query
    if token != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ==================== Session ID 校验 ====================

_SESSION_ID_RE = _re.compile(r'^[a-zA-Z0-9_\-]+$')


def validate_session_id(session_id: str):
    """校验 session_id 格式，防止路径遍历。"""
    if not session_id or not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="非法 session_id")


# ==================== Registry 注入 ====================

def get_registry(request: Request):
    """FastAPI 依赖：从 app.state 获取 AppRegistry 实例。"""
    return request.app.state.registry
