"""
会话状态管理模块 — 模块级便捷入口

本模块保留 get_session / delete_session / cleanup_expired_sessions 等函数签名，
内部委托到 AppRegistry 默认实例。agent / service 层可继续以原方式 import 使用。
"""

from typing import Dict, Any


def get_session(session_id: str) -> Dict[str, Any]:
    """同步获取会话状态（供 agent/service 层使用）。"""
    from .registry import _default_registry
    return _default_registry.get_session_sync(session_id)


def delete_session(session_id: str):
    """同步删除会话（内存+磁盘+全局注册表）。"""
    import shutil
    from pathlib import Path
    from .registry import _default_registry

    _default_registry.sessions.pop(session_id, None)
    _default_registry.asset_managers.pop(session_id, None)
    _default_registry.llm_loggers.pop(session_id, None)
    session_dir = Path("assets") / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)


def cleanup_expired_sessions():
    """同步清理过期会话。"""
    from .registry import _default_registry
    _default_registry._cleanup_unlocked()


def get_all_sessions_info() -> Dict[str, Any]:
    """获取所有会话信息。"""
    from .registry import _default_registry
    return _default_registry.get_all_sessions_info()
