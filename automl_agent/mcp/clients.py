"""
MCP客户端模块

本模块提供了用于与MCP服务器进行通信的客户端类。
包括代码执行服务器客户端和文件管理服务器客户端。

主要功能：
1. ExecutionMCPClient - 代码执行服务器的HTTP客户端
2. FileMCPClient - 文件管理服务器的HTTP客户端

这些客户端类封装了HTTP请求的细节，提供了更友好的Python API。
"""

from typing import Any, Optional
from pydantic import BaseModel
import requests


class ExecutionMCPClient:
    """
    代码执行MCP服务器客户端
    
    用于与代码执行MCP服务器进行通信的客户端类。
    封装了所有HTTP请求，提供简洁的Python方法调用。
    
    Attributes:
        base_url: MCP服务器的基地址，默认 http://localhost:8001
        
    Example:
        >>> client = ExecutionMCPClient()
        >>> result = client.execute("print('Hello')")
        >>> print(result['success'])
        True
    """
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        初始化客户端
        
        Args:
            base_url: MCP服务器的基地址，默认 http://localhost:8001
        """
        self.base_url = base_url
    
    def execute(self, code: str, context: Optional[dict[str, Any]] = None, timeout: int = 300) -> dict[str, Any]:
        """
        执行Python代码
        
        向MCP服务器发送代码执行请求。
        
        Args:
            code: 要执行的Python代码字符串
            context: 可选的上下文变量字典
            timeout: 请求超时时间（秒），默认300秒
            
        Returns:
            dict: 执行结果字典，包含 success、output、error、data 键
            
        Raises:
            requests.HTTPError: 如果HTTP请求失败
            
        Example:
            >>> result = client.execute("result = 1 + 1")
            >>> result['data']
            2
        """
        response = requests.post(
            f"{self.base_url}/execute",      # API端点
            json={"code": code, "context": context, "timeout": timeout}  # 请求体
        )
        response.raise_for_status()  # 检查HTTP错误
        return response.json()       # 返回JSON响应
    
    def set_variable(self, name: str, value: Any):
        """
        设置变量
        
        在MCP服务器的命名空间中设置一个变量。
        
        Args:
            name: 变量名
            value: 变量值
            
        Raises:
            requests.HTTPError: 如果HTTP请求失败
            
        Example:
            >>> client.set_variable("df", {"data": [1,2,3]})
        """
        response = requests.post(
            f"{self.base_url}/execute/variable",  # API端点
            params={"name": name, "value": value}   # 查询参数
        )
        response.raise_for_status()
    
    def get_variable(self, name: str) -> Any:
        """
        获取变量
        
        从MCP服务器的命名空间中获取指定变量的值。
        
        Args:
            name: 变量名
            
        Returns:
            Any: 变量的值
            
        Raises:
            requests.HTTPError: 如果HTTP请求失败
            
        Example:
            >>> value = client.get_variable("df")
        """
        response = requests.get(
            f"{self.base_url}/execute/variable/{name}"  # API端点（路径参数）
        )
        response.raise_for_status()
        return response.json()["value"]
    
    def reset(self):
        """
        重置执行器
        
        清空MCP服务器的执行器状态和所有变量。
        
        Raises:
            requests.HTTPError: 如果HTTP请求失败
            
        Example:
            >>> client.reset()
        """
        response = requests.post(f"{self.base_url}/execute/reset")  # POST请求重置
        response.raise_for_status()


class FileMCPClient:
    """
    文件管理MCP服务器客户端
    
    用于与文件管理MCP服务器进行通信的客户端类。
    提供了文件上传、列表、读取、删除和类型检测等功能。
    
    Attributes:
        base_url: MCP服务器的基地址，默认 http://localhost:8002
        
    Example:
        >>> client = FileMCPClient()
        >>> client.upload("data.csv")
        >>> files = client.list_files()
    """
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        """
        初始化客户端
        
        Args:
            base_url: MCP服务器的基地址，默认 http://localhost:8002
        """
        self.base_url = base_url
    
    def upload(self, file_path: str) -> dict[str, Any]:
        """
        上传文件
        
        将本地文件上传到MCP服务器的存储目录。
        
        Args:
            file_path: 要上传的本地文件路径
            
        Returns:
            dict: 文件信息字典，包含 filename、path、size、content_type、hash
            
        Raises:
            requests.HTTPError: 如果HTTP请求失败
            
        Example:
            >>> info = client.upload("/path/to/data.csv")
            >>> print(info['filename'])
            data.csv
        """
        with open(file_path, "rb") as f:  # 以二进制模式打开文件
            response = requests.post(
                f"{self.base_url}/upload",     # API端点
                files={"file": f}              # 上传文件
            )
        response.raise_for_status()
        return response.json()
    
    def list_files(self) -> list[dict[str, Any]]:
        """
        获取文件列表
        
        获取服务器存储目录中的所有文件列表。
        
        Returns:
            list: 文件信息字典列表
            
        Raises:
            requests.HTTPError: 如果HTTP请求失败
            
        Example:
            >>> files = client.list_files()
            >>> for f in files:
            ...     print(f['filename'])
        """
        response = requests.get(f"{self.base_url}/files")  # GET请求获取列表
        response.raise_for_status()
        return response.json()["files"]
    
    def read_file(self, filename: str) -> dict[str, Any]:
        """
        读取文件信息
        
        获取指定文件的元数据信息。
        
        Args:
            filename: 要查询的文件名
            
        Returns:
            dict: 包含文件路径和大小的字典
            
        Raises:
            requests.HTTPError: 如果HTTP请求失败
            
        Example:
            >>> info = client.read_file("data.csv")
            >>> print(info['size'])
            1024
        """
        response = requests.get(f"{self.base_url}/files/{filename}")
        response.raise_for_status()
        return response.json()
    
    def delete_file(self, filename: str):
        """
        删除文件
        
        删除服务器存储目录中的指定文件。
        
        Args:
            filename: 要删除的文件名
            
        Raises:
            requests.HTTPError: 如果HTTP请求失败
            
        Example:
            >>> client.delete_file("data.csv")
        """
        response = requests.delete(f"{self.base_url}/files/{filename}")
        response.raise_for_status()
    
    def detect_type(self, filename: str) -> dict[str, Any]:
        """
        检测文件类型
        
        自动识别文件的MIME类型。
        
        Args:
            filename: 要检测的文件名
            
        Returns:
            dict: 包含 filename、extension、content_type 的字典
            
        Raises:
            requests.HTTPError: 如果HTTP请求失败
            
        Example:
            >>> info = client.detect_type("data.csv")
            >>> print(info['content_type'])
            text/csv
        """
        response = requests.get(f"{self.base_url}/detect/{filename}")
        response.raise_for_status()
        return response.json()
