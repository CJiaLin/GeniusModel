"""
阶段结果查询工具模块

提供 StageResultTool，让 LLM 查询前序工作流阶段的结果
"""

import os
from typing import Any, Dict, Optional, Literal

from pydantic import BaseModel, Field

from .base_tool import BaseTool, ToolResult


# 阶段名到 AssetManager 资产类型的映射
STAGE_ASSET_MAP = {
    "cleaning": "cleaning",
    "exploration": "exploration",
    "features": "features",
    "models": "models",
    "analysis": "analysis",
    "reports": "reports",
    "data": "data",
    "code": "code",
}


class StageResultInput(BaseModel):
    stage: str = Field(
        ..., description="阶段名称: cleaning / exploration / features / models / analysis / reports / data / code"
    )
    filename: str = Field(
        "", description="可选的文件名（如 'cleaning_result.json', 'feature_metrics_report.md'）。不指定则列出可用文件"
    )


class StageResultTool(BaseTool):
    """查询前序工作流阶段结果的工具"""

    name = "query_stage_result"
    description = (
        "查询前序工作流阶段的结果。支持查询数据清洗(cleaning)、"
        "数据探索(exploration)、特征工程(features)、模型训练(models)、"
        "分析(analysis)等阶段的结果文件。不指定文件名时列出该阶段所有可用文件。"
    )
    input_model = StageResultInput

    def __init__(self, session_id: str = "default"):
        self._session_id = session_id
        super().__init__()

    def execute(self, stage: str = "", filename: str = "", **kwargs) -> ToolResult:
        """查询阶段结果"""
        if not stage:
            return ToolResult.error(
                f"必须指定 stage 参数。可选值: {list(STAGE_ASSET_MAP.keys())}"
            )

        if stage not in STAGE_ASSET_MAP:
            return ToolResult.error(
                f"未知阶段 '{stage}'。可选值: {list(STAGE_ASSET_MAP.keys())}"
            )

        try:
            from ..assets import get_asset_manager
            am = get_asset_manager(session_id=self._session_id)
        except Exception as e:
            return ToolResult.error(f"无法获取资产管理器: {e}")

        asset_type = STAGE_ASSET_MAP[stage]

        # 未指定文件名时，列出可用文件
        if not filename:
            try:
                assets = am.list_assets(asset_type)
                if not assets:
                    return ToolResult.success(data=f"阶段 '{stage}' 暂无结果文件。")
                file_list = [
                    {"name": a.name, "size": a.size}
                    for a in assets
                ]
                return ToolResult.success(data={
                    "stage": stage,
                    "files": file_list,
                })
            except Exception as e:
                return ToolResult.error(f"列出文件失败: {e}")

        # 读取指定文件
        try:
            content = am.read_asset(asset_type, filename)
            if content is None:
                # 列出可用文件作为提示
                try:
                    assets = am.list_assets(asset_type)
                    available = [a.name for a in assets] if assets else []
                except Exception:
                    available = []
                return ToolResult.error(
                    f"文件 '{filename}' 不存在于 '{stage}' 阶段。"
                    f"可用文件: {available}"
                )

            # 截断过长内容
            if isinstance(content, str) and len(content) > 5000:
                content = content[:5000] + f"\n\n... [内容已截断，共 {len(content)} 字符]"

            return ToolResult.success(data=content)

        except Exception as e:
            return ToolResult.error(f"读取文件失败: {e}")
