"""
集中式注册表 — sessions + asset_managers + llm_loggers，并发安全。

路由层通过 async 方法访问（asyncio.Lock 保护），
agent 层通过 sync 方法或模块级 get_* 便捷函数访问。
测试时创建独立 AppRegistry 实例即可隔离状态。
"""

import asyncio
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from automl_react.workflow import WorkflowState
from automl_react.confirmation import ConfirmationManager
from automl_react.assets.asset_manager import AssetManager
from automl_react.logger.llm_logger import LLMLogger


class AppRegistry:
    """
    管理所有 per-session 状态的并发安全注册表。

    - async 方法（get_session / delete_session / cleanup_expired）供路由层使用
    - sync 方法（get_session_sync / get_asset_manager / get_llm_logger）供 agent 层使用
    """

    def __init__(self, max_sessions: int = 50, ttl_hours: int = 72):
        self._lock = asyncio.Lock()
        self.max_sessions = max_sessions
        self.ttl_hours = ttl_hours
        self.sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.asset_managers: Dict[str, AssetManager] = {}
        self.llm_loggers: Dict[str, LLMLogger] = {}

    # ==================== async 方法（路由层） ====================

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        async with self._lock:
            return self._get_session_unlocked(session_id)

    async def delete_session(self, session_id: str) -> None:
        async with self._lock:
            self.sessions.pop(session_id, None)
            self.asset_managers.pop(session_id, None)
            self.llm_loggers.pop(session_id, None)
        session_dir = Path("assets") / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)

    async def cleanup_expired(self) -> int:
        async with self._lock:
            return self._cleanup_unlocked()

    # ==================== sync 方法（agent 层） ====================

    def get_session_sync(self, session_id: str) -> Dict[str, Any]:
        """同步获取会话。仅在已知单线程上下文中使用。"""
        return self._get_session_unlocked(session_id)

    def get_asset_manager(self, session_id: str = "default",
                          base_dir: str = None) -> AssetManager:
        if session_id not in self.asset_managers:
            self.asset_managers[session_id] = AssetManager(base_dir, session_id)
        return self.asset_managers[session_id]

    def get_llm_logger(self, session_id: str = "default",
                       log_dir: str = None) -> LLMLogger:
        if session_id not in self.llm_loggers:
            self.llm_loggers[session_id] = LLMLogger(log_dir, session_id)
        return self.llm_loggers[session_id]

    def remove_asset_manager(self, session_id: str) -> None:
        self.asset_managers.pop(session_id, None)

    def remove_llm_logger(self, session_id: str) -> None:
        self.llm_loggers.pop(session_id, None)

    # ==================== 内部方法 ====================

    def _get_session_unlocked(self, session_id: str) -> Dict[str, Any]:
        if session_id in self.sessions:
            self.sessions.move_to_end(session_id)
            return self.sessions[session_id]
        session = self._restore_or_create(session_id)
        self.sessions[session_id] = session
        self._enforce_cap()
        return session

    def _restore_or_create(self, session_id: str) -> Dict[str, Any]:
        """尝试从磁盘恢复会话，失败则创建空会话。"""
        from .helpers import normalize_workflow_data_path

        session_dir = Path("assets") / session_id
        state_file = session_dir / "state" / "workflow_state.json"

        if state_file.exists():
            try:
                workflow_state = WorkflowState.load(session_id)
                normalize_workflow_data_path(session_id, workflow_state)

                cm_path = str(session_dir / "state" / "confirmation_state.json")
                confirmation_manager = ConfirmationManager.load_from_disk(cm_path)
                if confirmation_manager is None:
                    confirmation_manager = ConfirmationManager(save_path=cm_path)

                return {
                    "session_id": session_id,
                    "created_at": (
                        workflow_state.history[0].get("timestamp", datetime.now().isoformat())
                        if workflow_state.history
                        else datetime.now().isoformat()
                    ),
                    "workflow_state": workflow_state,
                    "confirmation_manager": confirmation_manager,
                    "agents": {},
                    "context": workflow_state.context,
                }
            except Exception as e:
                print(f"[Registry] Session 恢复失败: {e}")

        return {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "workflow_state": None,
            "confirmation_manager": None,
            "agents": {},
            "context": {},
        }

    def _enforce_cap(self) -> None:
        """淘汰最久未访问的会话，直到满足容量限制。"""
        while len(self.sessions) > self.max_sessions:
            evicted_sid, _ = self.sessions.popitem(last=False)
            self.asset_managers.pop(evicted_sid, None)
            self.llm_loggers.pop(evicted_sid, None)
            print(f"[Registry] LRU 淘汰: {evicted_sid}")

    def _cleanup_unlocked(self) -> int:
        """清理超过 TTL 的会话（磁盘+内存）。"""
        assets_dir = Path("assets")
        if not assets_dir.exists():
            return 0

        cutoff = datetime.now() - timedelta(hours=self.ttl_hours)
        cleaned = 0

        for d in assets_dir.iterdir():
            if not d.is_dir():
                continue
            state_file = d / "state" / "workflow_state.json"
            if not state_file.exists():
                continue
            try:
                with open(state_file, "r") as f:
                    state_data = json.load(f)
                last_updated = state_data.get("last_updated", "")
                if last_updated and datetime.fromisoformat(last_updated) < cutoff:
                    sid = d.name
                    self.sessions.pop(sid, None)
                    self.asset_managers.pop(sid, None)
                    self.llm_loggers.pop(sid, None)
                    shutil.rmtree(d)
                    cleaned += 1
            except Exception:
                continue

        # 清理内存中无磁盘状态且超过 TTL 的空会话
        stale_sids = []
        for sid, sess in self.sessions.items():
            created_str = sess.get("created_at", "")
            if not created_str:
                continue
            try:
                created_at = datetime.fromisoformat(created_str)
                if created_at < cutoff and sess.get("workflow_state") is None:
                    stale_sids.append(sid)
            except (ValueError, TypeError):
                continue
        for sid in stale_sids:
            self.sessions.pop(sid, None)
            self.asset_managers.pop(sid, None)
            self.llm_loggers.pop(sid, None)
            cleaned += 1

        if cleaned:
            print(f"[Registry] TTL 清理完成，共清理 {cleaned} 个过期会话")
        return cleaned

    def get_all_sessions_info(self) -> Dict[str, Any]:
        """获取所有会话信息（合并内存态和磁盘态）。"""
        sessions_info = {}

        assets_dir = Path("assets")
        if assets_dir.exists():
            for d in assets_dir.iterdir():
                if d.is_dir() and (d / "state" / "workflow_state.json").exists():
                    sid = d.name
                    try:
                        with open(d / "state" / "workflow_state.json", "r") as f:
                            state_data = json.load(f)
                        sessions_info[sid] = {
                            "session_id": sid,
                            "current_stage": state_data.get("current_stage", "unknown"),
                            "last_updated": state_data.get("last_updated", ""),
                            "created_at": (
                                state_data.get("history", [{}])[0].get("timestamp", "")
                                if state_data.get("history")
                                else ""
                            ),
                        }
                    except Exception:
                        sessions_info[sid] = {
                            "session_id": sid,
                            "current_stage": "unknown",
                            "last_updated": "",
                            "created_at": "",
                        }

        for sid, sess in self.sessions.items():
            ws = sess.get("workflow_state")
            if ws:
                sessions_info[sid] = {
                    "session_id": sid,
                    "current_stage": ws.current_stage.value if ws.current_stage else "unknown",
                    "last_updated": ws.last_updated if hasattr(ws, "last_updated") else "",
                    "created_at": sess.get("created_at", ""),
                }
            elif sid not in sessions_info:
                sessions_info[sid] = {
                    "session_id": sid,
                    "current_stage": "unknown",
                    "last_updated": "",
                    "created_at": sess.get("created_at", ""),
                }

        return sessions_info


# ==================== 默认全局实例 ====================

_MAX_SESSIONS = int(os.environ.get("AUTOML_MAX_SESSIONS", "50"))
_TTL_HOURS = int(os.environ.get("AUTOML_SESSION_TTL_HOURS", "72"))

_default_registry = AppRegistry(max_sessions=_MAX_SESSIONS, ttl_hours=_TTL_HOURS)
