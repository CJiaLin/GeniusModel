"""
LLM 调用日志记录器模块

记录每次大模型调用的输入输出日志
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field, asdict

from ..assets import get_asset_manager


@dataclass
class LLMCallRecord:
    """
    LLM 调用记录数据类

    Attributes:
        session_id: 会话ID
        timestamp: 请求时间
        model_name: 模型名称
        provider: 提供商
        input_content: 输入内容
        output_content: 输出内容
        input_tokens: 输入 Token 数
        output_tokens: 输出 Token 数
        total_tokens: 总 Token 数
        latency_ms: 响应延迟（毫秒）
        stage: 工作流阶段
        metadata: 元数据
    """
    session_id: str
    timestamp: str
    model_name: str
    provider: str
    input_content: str
    output_content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    stage: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class LLMLogger:
    """
    LLM 调用日志记录器

    记录每次大模型调用的输入输出日志到 JSONL 文件

    Attributes:
        log_dir: 日志保存根目录
        session_id: 当前会话ID
    """

    def __init__(self, log_dir: str = None, session_id: str = None):
        """
        初始化 LLM 日志记录器

        Args:
            log_dir: 日志保存根目录，默认为 logs/llm_calls
            session_id: 会话ID
        """
        if log_dir is None:
            # 默认日志目录：项目根目录/logs/llm_calls
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            log_dir = project_root / "logs" / "llm_calls"

        self.log_dir = Path(log_dir)
        self.session_id = session_id or "default"
        self.asset_manager = get_asset_manager(session_id=self.session_id)

        # 创建会话日志目录
        self.session_log_dir = self.log_dir / self.session_id
        self.session_log_dir.mkdir(parents=True, exist_ok=True)

        # 日志文件路径
        self.log_file = self.session_log_dir / f"{datetime.now().strftime('%Y%m%d')}.jsonl"

    def _write_record(self, record: LLMCallRecord):
        """
        写入日志记录

        Args:
            record: LLM 调用记录
        """
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")

    def log_call(
        self,
        model_name: str,
        provider: str,
        input_content: str,
        output_content: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: int = 0,
        stage: str = "",
        metadata: Dict[str, Any] = None
    ) -> LLMCallRecord:
        """
        记录 LLM 调用

        Args:
            model_name: 模型名称
            provider: 提供商
            input_content: 输入内容
            output_content: 输出内容
            input_tokens: 输入 Token 数
            output_tokens: 输出 Token 数
            latency_ms: 响应延迟（毫秒）
            stage: 工作流阶段
            metadata: 元数据

        Returns:
            LLMCallRecord 记录对象
        """
        prompt_metadata = dict(metadata or {})
        save_prompt_asset = prompt_metadata.pop("_save_prompt_asset", True)
        prompt_asset_content = prompt_metadata.pop("_prompt_asset_content", input_content)
        prompt_asset = None
        if save_prompt_asset:
            prompt_asset = self.asset_manager.save_prompt_for_stage(
                prompt=prompt_asset_content,
                stage=stage,
                metadata={
                    "stage": stage,
                    "model_name": model_name,
                    "provider": provider,
                    **prompt_metadata,
                }
            )
        if prompt_asset is not None:
            prompt_metadata["prompt_asset_path"] = prompt_asset.path
            prompt_metadata["prompt_asset_type"] = prompt_asset.type

        record = LLMCallRecord(
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            provider=provider,
            input_content=input_content,
            output_content=output_content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            stage=stage,
            metadata=prompt_metadata
        )

        self._write_record(record)
        return record

    def log_call_from_response(
        self,
        model_name: str,
        provider: str,
        input_content: str,
        response: Any,
        start_time: datetime,
        stage: str = "",
        metadata: Dict[str, Any] = None
    ) -> LLMCallRecord:
        """
        从响应对象记录 LLM 调用

        Args:
            model_name: 模型名称
            provider: 提供商
            input_content: 输入内容
            response: LLM 响应对象
            start_time: 请求开始时间
            stage: 工作流阶段
            metadata: 元数据

        Returns:
            LLMCallRecord 记录对象
        """
        end_time = datetime.now()
        latency_ms = int((end_time - start_time).total_seconds() * 1000)

        # 提取输出内容
        output_content = ""
        input_tokens = 0
        output_tokens = 0

        if hasattr(response, 'content'):
            output_content = response.content
        elif hasattr(response, 'text'):
            output_content = response.text
        else:
            output_content = str(response)

        # 提取 Token 信息（如果可用）
        if hasattr(response, 'usage'):
            usage = response.usage
            if hasattr(usage, 'input_tokens'):
                input_tokens = usage.input_tokens
            if hasattr(usage, 'output_tokens'):
                output_tokens = usage.output_tokens
            if hasattr(usage, 'prompt_tokens'):
                input_tokens = usage.prompt_tokens
            if hasattr(usage, 'completion_tokens'):
                output_tokens = usage.completion_tokens

        return self.log_call(
            model_name=model_name,
            provider=provider,
            input_content=input_content,
            output_content=output_content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            stage=stage,
            metadata=metadata
        )

    def get_logs(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        stage: str = None,
        limit: int = 100
    ) -> List[LLMCallRecord]:
        """
        查询日志记录

        Args:
            start_time: 开始时间
            end_time: 结束时间
            stage: 工作流阶段筛选
            limit: 返回记录数限制

        Returns:
            LLMCallRecord 列表
        """
        records = []

        if not self.log_file.exists():
            return records

        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    record = LLMCallRecord(**data)

                    # 时间筛选
                    if start_time:
                        record_time = datetime.fromisoformat(record.timestamp)
                        if record_time < start_time:
                            continue

                    if end_time:
                        record_time = datetime.fromisoformat(record.timestamp)
                        if record_time > end_time:
                            continue

                    # 阶段筛选
                    if stage and record.stage != stage:
                        continue

                    records.append(record)

                    if len(records) >= limit:
                        break

                except (json.JSONDecodeError, TypeError):
                    continue

        return records

    def get_logs_by_session(self, session_id: str, limit: int = 100) -> List[LLMCallRecord]:
        """
        获取指定会话的日志

        Args:
            session_id: 会话ID
            limit: 返回记录数限制

        Returns:
            LLMCallRecord 列表
        """
        records = []
        session_log_file = self.log_dir / session_id / f"{datetime.now().strftime('%Y%m%d')}.jsonl"

        if not session_log_file.exists():
            return records

        with open(session_log_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    record = LLMCallRecord(**data)
                    records.append(record)

                    if len(records) >= limit:
                        break

                except (json.JSONDecodeError, TypeError):
                    continue

        return records

    def get_stats(self) -> Dict[str, Any]:
        """
        获取日志统计信息

        Returns:
            统计信息字典
        """
        records = self.get_logs(limit=10000)

        if not records:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "avg_latency_ms": 0,
                "calls_by_stage": {},
                "calls_by_model": {}
            }

        total_tokens = sum(r.total_tokens for r in records)
        total_latency = sum(r.latency_ms for r in records)

        # 按阶段统计
        calls_by_stage = {}
        for r in records:
            stage = r.stage or "unknown"
            calls_by_stage[stage] = calls_by_stage.get(stage, 0) + 1

        # 按模型统计
        calls_by_model = {}
        for r in records:
            model = r.model_name or "unknown"
            calls_by_model[model] = calls_by_model.get(model, 0) + 1

        return {
            "total_calls": len(records),
            "total_tokens": total_tokens,
            "avg_latency_ms": total_latency / len(records) if records else 0,
            "calls_by_stage": calls_by_stage,
            "calls_by_model": calls_by_model
        }


# 全局 LLMLogger 实例
_llm_loggers: Dict[str, LLMLogger] = {}


def get_llm_logger(log_dir: str = None, session_id: str = None) -> LLMLogger:
    """
    获取 LLMLogger 实例

    Args:
        log_dir: 日志保存根目录
        session_id: 会话ID

    Returns:
        LLMLogger 实例
    """
    if session_id is None:
        session_id = "default"

    if session_id not in _llm_loggers:
        _llm_loggers[session_id] = LLMLogger(log_dir, session_id)

    return _llm_loggers[session_id]
