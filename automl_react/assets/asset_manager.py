"""
资产管理器模块

管理用户交互过程中生成的所有资产文件
"""

import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AssetInfo:
    """
    资产信息数据类

    Attributes:
        name: 资产名称
        type: 资产类型
        path: 资产路径
        size: 文件大小（字节）
        created_at: 创建时间
        metadata: 元数据
    """
    name: str
    type: str
    path: str
    size: int = 0
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class AssetManager:
    """
    资产管理器

    管理用户交互过程中生成的数据、代码、模型、报告等资产

    Attributes:
        base_dir: 资产保存根目录
        session_id: 当前会话ID
    """

    def __init__(self, base_dir: str = None, session_id: str = None):
        """
        初始化资产管理器

        Args:
            base_dir: 资产保存根目录，默认为 assets
            session_id: 会话ID
        """
        if base_dir is None:
            # 默认资产目录：项目根目录/assets
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            base_dir = project_root / "assets"

        self.base_dir = Path(base_dir)
        self.session_id = session_id or "default"

        # 创建会话资产目录
        self.session_dir = self.base_dir / self.session_id
        self._create_directory_structure()

    def _create_directory_structure(self):
        """创建资产目录结构"""
        directories = [
            "data",           # 原始数据和处理后的数据
            "analysis",       # 数据分析结果
            "cleaning",       # 数据清洗方案和结果
            "features",       # 特征工程方案和结果
            "code",           # 生成代码
            "models",         # 训练好的模型
            "reports",        # 可视化报告
        ]

        for dir_name in directories:
            (self.session_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def _get_asset_path(self, asset_type: str, filename: str) -> Path:
        """
        获取资产文件路径

        Args:
            asset_type: 资产类型
            filename: 文件名

        Returns:
            资产文件路径
        """
        return self.session_dir / asset_type / filename

    def save_data(
        self,
        data: Any,
        filename: str,
        asset_type: str = "data",
        metadata: Dict[str, Any] = None
    ) -> AssetInfo:
        """
        保存数据资产

        Args:
            data: 数据内容
            filename: 文件名
            asset_type: 资产类型
            metadata: 元数据

        Returns:
            AssetInfo 资产信息
        """
        asset_path = self._get_asset_path(asset_type, filename)
        
        # 确保目录存在
        asset_path.parent.mkdir(parents=True, exist_ok=True)

        # 根据数据类型选择保存方式
        if isinstance(data, str):
            with open(asset_path, "w", encoding="utf-8") as f:
                f.write(data)
        elif isinstance(data, bytes):
            with open(asset_path, "wb") as f:
                f.write(data)
        else:
            # 尝试使用 pandas 保存
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    if filename.endswith(".csv"):
                        data.to_csv(asset_path, index=False)
                    elif filename.endswith(".parquet"):
                        data.to_parquet(asset_path, index=False)
                    else:
                        data.to_csv(asset_path, index=False)
                else:
                    # 默认使用 JSON
                    import json
                    with open(asset_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
            except ImportError:
                import json
                with open(asset_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

        return self._create_asset_info(asset_path, asset_type, metadata)

    def save_code(
        self,
        code: str,
        filename: str,
        metadata: Dict[str, Any] = None
    ) -> AssetInfo:
        """
        保存代码文件

        Args:
            code: 代码内容
            filename: 文件名
            metadata: 元数据

        Returns:
            AssetInfo 资产信息
        """
        return self.save_data(code, filename, "code", metadata)

    def save_model(
        self,
        model: Any,
        filename: str,
        metadata: Dict[str, Any] = None
    ) -> AssetInfo:
        """
        保存模型文件

        Args:
            model: 模型对象
            filename: 文件名
            metadata: 元数据

        Returns:
            AssetInfo 资产信息
        """
        asset_path = self._get_asset_path("models", filename)

        # 使用 joblib 或 pickle 保存模型
        try:
            import joblib
            joblib.dump(model, asset_path)
        except ImportError:
            import pickle
            with open(asset_path, "wb") as f:
                pickle.dump(model, f)

        return self._create_asset_info(asset_path, "models", metadata)

    def save_report(
        self,
        report: str,
        filename: str,
        metadata: Dict[str, Any] = None
    ) -> AssetInfo:
        """
        保存报告文件

        Args:
            report: 报告内容（Markdown 或 HTML）
            filename: 文件名
            metadata: 元数据

        Returns:
            AssetInfo 资产信息
        """
        return self.save_data(report, filename, "reports", metadata)

    def save_pipeline_script(
        self,
        script: str,
        metadata: Dict[str, Any] = None
    ) -> AssetInfo:
        """
        保存全流程建模脚本

        Args:
            script: Python 脚本内容
            metadata: 元数据

        Returns:
            AssetInfo 资产信息
        """
        return self.save_data(script, "pipeline.py", "code", metadata)

    def _create_asset_info(
        self,
        asset_path: Path,
        asset_type: str,
        metadata: Dict[str, Any] = None
    ) -> AssetInfo:
        """
        创建资产信息对象

        Args:
            asset_path: 资产路径
            asset_type: 资产类型
            metadata: 元数据

        Returns:
            AssetInfo 资产信息
        """
        stat = asset_path.stat()

        return AssetInfo(
            name=asset_path.name,
            type=asset_type,
            path=str(asset_path),
            size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime).isoformat(),
            metadata=metadata or {}
        )

    def get_asset(self, asset_type: str, filename: str) -> Optional[Path]:
        """
        获取资产文件路径

        Args:
            asset_type: 资产类型
            filename: 文件名

        Returns:
            资产文件路径，如果不存在返回 None
        """
        asset_path = self._get_asset_path(asset_type, filename)

        if asset_path.exists():
            return asset_path

        return None

    def read_asset(self, asset_type: str, filename: str) -> Optional[str]:
        """
        读取资产文件内容

        Args:
            asset_type: 资产类型
            filename: 文件名

        Returns:
            资产文件内容，如果不存在返回 None
        """
        asset_path = self.get_asset(asset_type, filename)

        if asset_path is None:
            return None

        with open(asset_path, "r", encoding="utf-8") as f:
            return f.read()

    def list_assets(self, asset_type: str = None) -> List[AssetInfo]:
        """
        列出所有资产

        Args:
            asset_type: 资产类型筛选，为 None 时列出所有类型

        Returns:
            AssetInfo 列表
        """
        assets = []

        if asset_type:
            asset_dirs = [self.session_dir / asset_type]
        else:
            asset_dirs = [d for d in self.session_dir.iterdir() if d.is_dir()]

        for asset_dir in asset_dirs:
            if not asset_dir.exists():
                continue

            for asset_file in asset_dir.iterdir():
                if asset_file.is_file():
                    assets.append(self._create_asset_info(
                        asset_file,
                        asset_dir.name,
                        {}
                    ))

        return assets

    def delete_asset(self, asset_type: str, filename: str) -> bool:
        """
        删除资产文件

        Args:
            asset_type: 资产类型
            filename: 文件名

        Returns:
            是否删除成功
        """
        asset_path = self._get_asset_path(asset_type, filename)

        if asset_path.exists():
            asset_path.unlink()
            return True

        return False

    def clear_assets(self, asset_type: str = None):
        """
        清空资产

        Args:
            asset_type: 资产类型，为 None 时清空所有资产
        """
        if asset_type:
            asset_dir = self.session_dir / asset_type
            if asset_dir.exists():
                shutil.rmtree(asset_dir)
                asset_dir.mkdir(parents=True, exist_ok=True)
        else:
            # 清空整个会话目录
            if self.session_dir.exists():
                shutil.rmtree(self.session_dir)
            self._create_directory_structure()

    def get_download_url(self, asset_type: str, filename: str) -> str:
        """
        获取资产下载 URL

        Args:
            asset_type: 资产类型
            filename: 文件名

        Returns:
            下载 URL
        """
        return f"/api/assets/{self.session_id}/{asset_type}/{filename}"

    def get_all_download_urls(self) -> Dict[str, List[Dict[str, str]]]:
        """
        获取所有资产的下载 URL

        Returns:
            资产类型到下载 URL 列表的映射
        """
        urls = {}

        for asset_type in ["data", "analysis", "cleaning", "features", "code", "models", "reports"]:
            assets = self.list_assets(asset_type)
            if assets:
                urls[asset_type] = [
                    {
                        "name": a.name,
                        "url": self.get_download_url(asset_type, a.name),
                        "size": a.size,
                        "created_at": a.created_at
                    }
                    for a in assets
                ]

        return urls


# 全局 AssetManager 实例
_asset_managers: Dict[str, AssetManager] = {}


def get_asset_manager(base_dir: str = None, session_id: str = None) -> AssetManager:
    """
    获取 AssetManager 实例

    Args:
        base_dir: 资产保存根目录
        session_id: 会话ID

    Returns:
        AssetManager 实例
    """
    if session_id is None:
        session_id = "default"

    if session_id not in _asset_managers:
        _asset_managers[session_id] = AssetManager(base_dir, session_id)

    return _asset_managers[session_id]
