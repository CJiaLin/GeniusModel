"""
文件管理MCP服务器模块

本模块提供了一个基于FastAPI的MCP（Model Context Protocol）服务器，
用于通过HTTP API管理文件的上传、下载、删除和类型检测。

主要功能：
1. 文件上传和存储
2. 文件列表查看
3. 文件读取/元数据获取
4. 文件删除
5. 文件类型自动检测

使用方式：
    启动服务器：python -m automl_agent.mcp.file_mcp
    API端点：http://localhost:8002

API接口：
    POST /upload - 上传文件
    GET /files - 获取文件列表
    GET /files/{filename} - 获取文件信息
    DELETE /files/{filename} - 删除文件
    GET /detect/{filename} - 检测文件类型
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn


# 创建FastAPI应用实例
app = FastAPI(
    title="File Management MCP Server",  # API标题
    description="用于文件管理的MCP服务器",  # API描述
    version="1.0.0"  # API版本
)

# 定义文件存储目录
STORAGE_DIR = Path("data/uploads")
# 确保存储目录存在，不存在则创建
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class FileInfo(BaseModel):
    """
    文件信息模型
    
    存储文件的基本信息。
    
    Attributes:
        filename: 文件名
        path: 文件完整路径
        size: 文件大小（字节）
        content_type: MIME类型
        hash: 文件MD5哈希值
    """
    filename: str     # 文件名
    path: str         # 完整路径
    size: int         # 文件大小（字节）
    content_type: str # MIME类型
    hash: str         # MD5哈希值


class FileListResponse(BaseModel):
    """
    文件列表响应模型
    
    用于返回文件列表的响应结构。
    
    Attributes:
        files: 文件信息列表
    """
    files: list[FileInfo]  # 文件列表


@app.post("/upload", response_model=FileInfo)
async def upload_file(file: UploadFile = File(...)):
    """
    文件上传API端点
    
    接收客户端上传的文件，并保存到本地存储目录。
    
    Args:
        file: 上传的文件对象（FastAPI自动解析）
        
    Returns:
        FileInfo: 上传文件的详细信息
        
    Raises:
        HTTPException: 如果保存文件时发生错误
        
    Example:
        请求: POST /upload (multipart/form-data)
        响应: {
            "filename": "data.csv",
            "path": "data/uploads/data.csv",
            "size": 1024,
            "content_type": "text/csv",
            "hash": "5d41402abc4b2a76b9719d911017c592"
        }
    """
    # 构建文件保存路径
    file_path = STORAGE_DIR / file.filename
    
    # 读取文件内容
    content = await file.read()
    
    # 将内容写入磁盘
    with open(file_path, "wb") as f:
        f.write(content)
    
    # 计算文件MD5哈希值
    file_hash = hashlib.md5(content).hexdigest()
    
    # 返回文件信息
    return FileInfo(
        filename=file.filename,              # 文件名
        path=str(file_path),                 # 完整路径
        size=len(content),                   # 文件大小
        content_type=file.content_type,      # MIME类型
        hash=file_hash                       # MD5哈希
    )


@app.get("/files", response_model=FileListResponse)
async def list_files():
    """
    获取文件列表API端点
    
    返回存储目录中所有文件的列表信息。
    
    Returns:
        FileListResponse: 包含所有文件信息的响应
        
    Example:
        响应: {
            "files": [
                {"filename": "data.csv", "path": "...", "size": 1024, ...}
            ]
        }
    """
    files = []
    
    # 遍历存储目录中的所有文件
    for f in STORAGE_DIR.iterdir():
        if f.is_file():
            # 获取文件统计信息
            stat = f.stat()
            
            # 创建FileInfo对象并添加到列表
            files.append(FileInfo(
                filename=f.name,                                   # 文件名
                path=str(f),                                       # 完整路径
                size=stat.st_size,                                 # 文件大小
                content_type="application/octet-stream",           # 默认MIME类型
                hash=hashlib.md5(f.read_bytes()).hexdigest()      # MD5哈希
            ))
    
    return FileListResponse(files=files)


@app.get("/files/{filename}")
async def read_file(filename: str):
    """
    获取文件信息API端点
    
    根据文件名获取文件的元数据信息。
    
    Args:
        filename: 要查询的文件名
        
    Returns:
        包含文件路径和大小的字典
        
    Raises:
        HTTPException: 如果文件不存在
        
    Example:
        响应: {"path": "data/uploads/data.csv", "size": 1024}
    """
    file_path = STORAGE_DIR / filename
    
    # 检查文件是否存在
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return {
        "path": str(file_path),                        # 文件路径
        "size": file_path.stat().st_size              # 文件大小
    }


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """
    删除文件API端点
    
    根据文件名删除指定的文件。
    
    Args:
        filename: 要删除的文件名
        
    Returns:
        删除状态信息
        
    Raises:
        HTTPException: 如果文件不存在
        
    Example:
        响应: {"status": "deleted", "filename": "data.csv"}
    """
    file_path = STORAGE_DIR / filename
    
    # 检查文件是否存在
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # 删除文件
    file_path.unlink()
    
    return {
        "status": "deleted",     # 删除状态
        "filename": filename     # 被删除的文件名
    }


@app.get("/detect/{filename}")
async def detect_file_type(filename: str):
    """
    检测文件类型API端点
    
    根据文件扩展名自动识别文件的MIME类型。
    
    Args:
        filename: 要检测的文件名
        
    Returns:
        包含文件名、扩展名和MIME类型的字典
        
    Raises:
        HTTPException: 如果文件不存在
        
    Example:
        响应: {
            "filename": "data.csv",
            "extension": ".csv",
            "content_type": "text/csv"
        }
    """
    file_path = STORAGE_DIR / filename
    
    # 检查文件是否存在
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # 获取文件扩展名（小写）
    ext = file_path.suffix.lower()
    
    # MIME类型映射表
    content_type_map = {
        ".csv": "text/csv",                                              # CSV文件
        ".json": "application/json",                                     # JSON文件
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # Excel文件
        ".xls": "application/vnd.ms-excel",                              # 老版本Excel
        ".txt": "text/plain",                                            # 文本文件
        ".parquet": "application/apache.parquet",                        # Parquet文件
    }
    
    return {
        "filename": filename,                                        # 文件名
        "extension": ext,                                             # 扩展名
        "content_type": content_type_map.get(ext, "application/octet-stream")  # MIME类型
    }


# 主程序入口
if __name__ == "__main__":
    """
    启动MCP服务器
    
    使用uvicorn运行FastAPI应用，监听0.0.0.0:8002
    
    使用方式:
        python file_mcp.py
    """
    uvicorn.run(
        app,                    # FastAPI应用实例
        host="0.0.0.0",        # 监听地址
        port=8002              # 监听端口
    )
