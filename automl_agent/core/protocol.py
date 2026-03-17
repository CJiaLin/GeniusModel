"""
Agent间通信协议模块

本模块定义了AutoML系统中各个Agent之间进行通信的标准协议和数据格式。

主要功能：
1. 定义消息类型枚举
2. Agent消息数据结构
3. Agent响应数据结构
4. 消息创建辅助函数

设计原则：
- 每个消息都有唯一的消息ID用于追踪
- 消息包含发送者、接收者、动作和载荷
- 支持请求、响应、错误、进度和完成等多种消息类型
- 响应消息与原始请求通过message_id关联

注意：使用Optional[]保持Python 3.8兼容性
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class MessageType(str, Enum):
    """
    消息类型枚举
    
    定义了Agent之间通信的消息类型。
    每种类型对应不同的通信场景和含义。
    """
    REQUEST = "request"     # 请求消息：发起方请求接收方执行某个动作
    RESPONSE = "response"  # 响应消息：对请求的回复
    ERROR = "error"        # 错误消息：执行过程中发生错误
    PROGRESS = "progress"  # 进度消息：报告任务执行进度
    COMPLETE = "complete"  # 完成消息：任务执行完成


class AgentMessage(BaseModel):
    """
    Agent消息模型
    
    用于在Agent之间传递信息的基本消息结构。
    包含消息元数据和消息载荷。
    
    Attributes:
        message_id: 唯一标识符，用于追踪请求-响应配对
        timestamp: 消息创建时间
        sender: 发送方Agent名称
        receiver: 接收方Agent名称
        message_type: 消息类型
        action: 要执行的动作名称
        payload: 消息载荷，包含动作所需的参数
        status: 消息状态（pending/success/failed）
        error: 错误信息（如果状态为failed）
    
    Example:
        >>> msg = AgentMessage(
        ...     message_id="123-456",
        ...     sender="planner",
        ...     receiver="data_agent",
        ...     message_type=MessageType.REQUEST,
        ...     action="load_data",
        ...     payload={"file_path": "data.csv"}
        ... )
    """
    message_id: str                              # 消息唯一标识符
    timestamp: datetime = Field(default_factory=datetime.now)  # 创建时间
    sender: str                                   # 发送方Agent名称
    receiver: str                                 # 接收方Agent名称
    message_type: MessageType                    # 消息类型
    action: str                                   # 要执行的动作
    payload: dict = Field(default_factory=dict)  # 消息载荷
    status: str = "pending"  # 状态
    error: Optional[str] = None                    # 错误信息


class AgentResponse(BaseModel):
    """
    Agent响应模型
    
    用于响应AgentMessage的回复结构。
    包含原始请求的ID和执行结果。
    
    Attributes:
        message_id: 对应请求消息的ID
        timestamp: 响应创建时间
        status: 执行状态（success/failed）
        data: 响应数据，包含执行结果
        error: 错误信息（如果状态为failed）
    
    Example:
        >>> response = AgentResponse(
        ...     message_id="123-456",
        ...     status="success",
        ...     data={"profile": {"shape": (100, 10)}}
        ... )
    """
    message_id: str                              # 对应请求的消息ID
    timestamp: datetime = Field(default_factory=datetime.now)  # 创建时间
    status: str = "success"        # 执行状态
    data: dict = Field(default_factory=dict)  # 响应数据
    error: Optional[str] = None                   # 错误信息


def create_message(
    sender: str,
    receiver: str,
    action: str,
    payload: Optional[dict] = None,
    message_type: MessageType = MessageType.REQUEST
) -> AgentMessage:
    """
    创建Agent消息的辅助函数
    
    简化消息创建过程，自动生成UUID和时间戳。
    
    Args:
        sender: 发送方Agent名称
        receiver: 接收方Agent名称
        action: 要执行的动作名称
        payload: 消息载荷字典
        message_type: 消息类型，默认REQUEST
        
    Returns:
        AgentMessage: 创建的消息对象
        
    Example:
        >>> msg = create_message(
        ...     sender="planner",
        ...     receiver="data_agent",
        ...     action="load_data",
        ...     payload={"path": "data.csv"}
        ... )
    """
    import uuid
    return AgentMessage(
        message_id=str(uuid.uuid4()),  # 生成唯一ID
        sender=sender,                   # 发送方
        receiver=receiver,               # 接收方
        message_type=message_type,       # 消息类型
        action=action,                   # 动作名称
        payload=payload or {}            # 载荷数据
    )


def create_response(
    original_message: AgentMessage, 
    status: str, 
    data: Optional[dict] = None, 
    error: Optional[str] = None
) -> AgentResponse:
    """
    创建Agent响应的辅助函数
    
    根据原始请求消息创建对应的响应消息。
    自动复制原始消息的ID以建立关联。
    
    Args:
        original_message: 原始请求消息
        status: 执行状态（success/failed）
        data: 响应数据
        error: 错误信息
        
    Returns:
        AgentResponse: 创建的响应对象
        
    Example:
        >>> response = create_response(
        ...     original_message=msg,
        ...     status="success",
        ...     data={"result": "ok"}
        ... )
    """
    return AgentResponse(
        message_id=original_message.message_id,  # 使用原始消息ID建立关联
        status=status,                            # 执行状态
        data=data or {},                         # 响应数据
        error=error                              # 错误信息
    )
