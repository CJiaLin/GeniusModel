"""
全流程脚本生成器模块

生成可独立运行的全流程建模脚本
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from ..assets import get_asset_manager


class PipelineGenerator:
    """
    全流程脚本生成器

    收集各阶段代码，组装完整的建模脚本

    Attributes:
        session_id: 会话ID
        asset_manager: 资产管理器
    """

    def __init__(self, session_id: str = None):
        self.session_id = session_id or "default"
        self.asset_manager = get_asset_manager(session_id=self.session_id)

    def collect_stage_codes(self) -> Dict[str, str]:
        """
        收集各阶段代码

        Returns:
            阶段代码字典
        """
        stages = {}

        # 读取各阶段代码
        code_files = {
            "data_cleaning": "cleaning.py",
            "feature_engineering": "feature_engineering.py",
            "model_training": "model_training.py"
        }

        for stage, filename in code_files.items():
            code = self.asset_manager.read_asset("code", filename)
            if code:
                stages[stage] = code

        return stages

    def generate_pipeline_script(
        self,
        data_path: str,
        target_column: str,
        task_type: str = "classification"
    ) -> str:
        """
        生成全流程建模脚本

        Args:
            data_path: 原始数据路径
            target_column: 目标列名
            task_type: 任务类型

        Returns:
            完整的 Python 脚本
        """
        # 收集各阶段代码
        stage_codes = self.collect_stage_codes()

        # 生成脚本头部
        script_header = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoML 全流程建模脚本

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
会话ID: {self.session_id}

使用说明:
1. 确保已安装依赖: pip install pandas scikit-learn joblib
2. 运行脚本: python pipeline.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# 配置
DATA_PATH = "{data_path}"
TARGET_COLUMN = "{target_column}"
TASK_TYPE = "{task_type}"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("AutoML 全流程建模")
print("=" * 60)
print(f"数据路径: {{DATA_PATH}}")
print(f"目标列: {{TARGET_COLUMN}}")
print(f"任务类型: {{TASK_TYPE}}")
print()

'''

        # 生成各阶段代码
        script_body = ""

        # 1. 数据清洗阶段
        if "data_cleaning" in stage_codes:
            script_body += '''
# ============================================
# 阶段 1: 数据清洗
# ============================================
print("\\n[阶段 1] 数据清洗...")

'''
            # 提取数据清洗的核心代码
            cleaning_code = self._extract_core_code(stage_codes["data_cleaning"])
            script_body += cleaning_code
            script_body += '''
print("✓ 数据清洗完成")

'''

        # 2. 特征工程阶段
        if "feature_engineering" in stage_codes:
            script_body += '''
# ============================================
# 阶段 2: 特征工程
# ============================================
print("\\n[阶段 2] 特征工程...")

'''
            # 提取特征工程的核心代码
            feature_code = self._extract_core_code(stage_codes["feature_engineering"])
            script_body += feature_code
            script_body += '''
print("✓ 特征工程完成")

'''

        # 3. 模型训练阶段
        if "model_training" in stage_codes:
            script_body += '''
# ============================================
# 阶段 3: 模型训练
# ============================================
print("\\n[阶段 3] 模型训练...")

'''
            # 提取模型训练的核心代码
            model_code = self._extract_core_code(stage_codes["model_training"])
            script_body += model_code
            script_body += '''
print("✓ 模型训练完成")

'''

        # 生成脚本尾部
        script_footer = '''
# ============================================
# 保存结果
# ============================================
print("\\n" + "=" * 60)
print("建模完成!")
print("=" * 60)

# 保存结果摘要
summary = {
    "data_path": DATA_PATH,
    "target_column": TARGET_COLUMN,
    "task_type": TASK_TYPE,
    "output_dir": str(OUTPUT_DIR),
    "timestamp": pd.Timestamp.now().isoformat()
}

with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\\n结果已保存到: {{OUTPUT_DIR}}")
print("\\n文件列表:")
for file in OUTPUT_DIR.iterdir():
    print(f"  - {{file.name}}")
'''

        # 组装完整脚本
        full_script = script_header + script_body + script_footer

        # 保存脚本到资产
        self.asset_manager.save_code(
            code=full_script,
            filename="pipeline.py",
            metadata={
                "stage": "pipeline",
                "data_path": data_path,
                "target_column": target_column,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat()
            }
        )

        return full_script

    def _extract_core_code(self, code: str) -> str:
        """
        从代码中提取核心逻辑

        Args:
            code: 原始代码

        Returns:
            核心代码
        """
        import re

        # 优先提取 ```python ... ``` 代码块
        code_blocks = re.findall(r'```python\n(.*?)\n```', code, re.DOTALL)

        if not code_blocks:
            # 其次尝试普通 ``` ... ``` 代码块
            code_blocks = re.findall(r'```\n(.*?)\n```', code, re.DOTALL)

        core = code_blocks[0] if code_blocks else code

        # 统一清理掉可能残留的 Markdown 围栏行
        lines = []
        for line in core.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                continue
            lines.append(line)

        return "\n".join(lines).strip() + "\n"

    def get_pipeline_script(self) -> Optional[str]:
        """
        获取已生成的全流程脚本

        Returns:
            脚本内容，如果不存在返回 None
        """
        return self.asset_manager.read_asset("code", "pipeline.py")

    def save_pipeline_script(self, script: str) -> Dict[str, Any]:
        """
        保存全流程脚本

        Args:
            script: 脚本内容

        Returns:
            保存结果
        """
        result = self.asset_manager.save_code(
            code=script,
            filename="pipeline.py",
            metadata={
                "stage": "pipeline",
                "timestamp": datetime.now().isoformat()
            }
        )

        return {
            "success": True,
            "path": result.path,
            "size": result.size
        }
