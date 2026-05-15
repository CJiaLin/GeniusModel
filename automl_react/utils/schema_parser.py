"""
Schema 解析工具

支持从 JSON/YAML/CSV/XLSX 文件或纯文本描述中解析表结构和数据字典。
"""

import json
import os
from typing import Any, Dict, List, Optional


def _read_text_file(file_path: str) -> str:
    """读取文本文件，自动检测编码（UTF-8 → GBK → latin-1）。"""
    for encoding in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    # latin-1 不会失败，但以防万一
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_schema(file_path: str = None, text: str = None) -> Dict[str, Any]:
    """
    解析 schema 文件或文本描述，返回标准化结构。

    Args:
        file_path: schema 文件路径（json/yaml/csv/xlsx）
        text: 纯文本描述（如 taskDescription 中包含的表结构说明）

    Returns:
        {
            "columns": [{"name": str, "dtype": str, "description": str}, ...],
            "source_type": "json" | "yaml" | "csv" | "xlsx" | "text",
            "raw_text": str (仅 text 模式),
        }
    """
    if file_path and os.path.isfile(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".json":
            return _parse_json(file_path)
        elif ext in (".yaml", ".yml"):
            return _parse_yaml(file_path)
        elif ext == ".csv":
            return _parse_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            return _parse_excel(file_path)
        else:
            # 尝试作为文本读取
            content = _read_text_file(file_path)
            return {"columns": [], "source_type": "text", "raw_text": content}

    if text:
        return {"columns": [], "source_type": "text", "raw_text": text}

    raise ValueError("请提供 schema 文件路径或文本描述")


def _parse_json(file_path: str) -> Dict[str, Any]:
    """解析 JSON schema 文件。"""
    content = _read_text_file(file_path)
    data = json.loads(content)

    columns = _normalize_columns(data)
    return {"columns": columns, "source_type": "json", "raw_text": json.dumps(data, ensure_ascii=False, indent=2)}


def _parse_yaml(file_path: str) -> Dict[str, Any]:
    """解析 YAML schema 文件。"""
    try:
        import yaml
    except ImportError:
        # fallback: 以文本形式读取
        content = _read_text_file(file_path)
        return {"columns": [], "source_type": "text", "raw_text": content}

    content = _read_text_file(file_path)
    data = yaml.safe_load(content)

    columns = _normalize_columns(data)
    return {"columns": columns, "source_type": "yaml", "raw_text": json.dumps(data, ensure_ascii=False, indent=2)}


def _parse_csv(file_path: str) -> Dict[str, Any]:
    """解析 CSV 数据字典文件。"""
    import pandas as pd
    df = pd.read_csv(file_path)
    columns = _extract_columns_from_dataframe(df)
    return {"columns": columns, "source_type": "csv", "raw_text": df.to_string(index=False)}


def _parse_excel(file_path: str) -> Dict[str, Any]:
    """解析 Excel 数据字典文件。"""
    import pandas as pd
    df = pd.read_excel(file_path)
    columns = _extract_columns_from_dataframe(df)
    return {"columns": columns, "source_type": "xlsx", "raw_text": df.to_string(index=False)}


def _extract_columns_from_dataframe(df) -> List[Dict[str, str]]:
    """
    从 DataFrame 中提取字段信息。

    支持两种格式：
    1. 数据字典格式：列名包含 name/字段名, type/类型, description/描述 等
    2. 原始数据格式：直接用列名和 dtype 推断
    """
    col_lower = [c.lower() for c in df.columns]

    # 检测是否是数据字典格式
    name_col = _find_col(col_lower, df.columns, ["name", "字段名", "column", "field", "列名", "column_name"])
    type_col = _find_col(col_lower, df.columns, ["type", "dtype", "类型", "数据类型", "data_type"])
    desc_col = _find_col(col_lower, df.columns, ["description", "描述", "说明", "备注", "comment", "desc"])

    if name_col:
        # 数据字典格式
        columns = []
        for _, row in df.iterrows():
            col_info = {"name": str(row[name_col]).strip()}
            if type_col:
                col_info["dtype"] = str(row[type_col]).strip()
            if desc_col:
                col_info["description"] = str(row[desc_col]).strip()
            columns.append(col_info)
        return columns
    else:
        # 原始数据格式 — 用列名和 dtype
        return [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]


def _find_col(col_lower: list, original_cols, candidates: list) -> Optional[str]:
    """在列名中查找匹配的字段。"""
    for candidate in candidates:
        for i, cl in enumerate(col_lower):
            if candidate in cl:
                return original_cols[i]
    return None


def _normalize_columns(data: Any) -> List[Dict[str, str]]:
    """
    从结构化数据中提取列信息。

    支持多种 JSON/YAML 格式：
    - {"columns": [{"name": ..., "type": ...}, ...]}
    - {"fields": [{"name": ..., "type": ...}, ...]}
    - [{"name": ..., "type": ...}, ...]
    - {"col_name": "type", ...}
    """
    if isinstance(data, list):
        # 列表格式
        return [_normalize_col_entry(item) for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        # 检查常见 key
        for key in ("columns", "fields", "schema", "table", "字段"):
            if key in data:
                val = data[key]
                if isinstance(val, list):
                    return [_normalize_col_entry(item) for item in val if isinstance(item, dict)]
                elif isinstance(val, dict):
                    return [{"name": k, "dtype": str(v)} for k, v in val.items()]

        # 没有标准 key，尝试将 dict 作为 {col_name: type_or_desc}
        if all(isinstance(v, str) for v in data.values()):
            return [{"name": k, "dtype": v} for k, v in data.items()]

    return []


def _normalize_col_entry(item: dict) -> Dict[str, str]:
    """标准化单个列条目。"""
    result = {}
    # name
    for key in ("name", "column", "field", "字段名", "列名", "column_name"):
        if key in item:
            result["name"] = str(item[key])
            break
    if "name" not in result:
        result["name"] = str(list(item.values())[0]) if item else "unknown"

    # dtype
    for key in ("type", "dtype", "data_type", "类型", "数据类型"):
        if key in item:
            result["dtype"] = str(item[key])
            break

    # description
    for key in ("description", "desc", "描述", "说明", "comment", "备注"):
        if key in item:
            result["description"] = str(item[key])
            break

    return result


def schema_to_text(schema_info: Dict[str, Any]) -> str:
    """将解析后的 schema 转换为 LLM 可读的文本描述。"""
    columns = schema_info.get("columns", [])
    raw_text = schema_info.get("raw_text", "")

    if not columns and raw_text:
        return raw_text

    lines = ["| 字段名 | 数据类型 | 描述 |", "| --- | --- | --- |"]
    for col in columns:
        name = col.get("name", "")
        dtype = col.get("dtype", "-")
        desc = col.get("description", "-")
        lines.append(f"| {name} | {dtype} | {desc} |")

    return "\n".join(lines)
