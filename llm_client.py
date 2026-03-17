"""
LLM客户端配置模块

本模块提供自定义LLM API端点的支持，允许用户配置自己的大模型服务。

支持的配置方式：
1. 配置文件（config.yaml）- 优先级最低
2. 显式参数 - 优先级最高

配置项：
- API基础URL
- API密钥
- 默认模型
- 超时设置
- 重试策略
- System Prompt

使用示例:
    from llm_client import get_llm_client, configure_llm, set_system_prompt
    
    # 配置（方式1: 直接配置）
    configure_llm(
        base_url="https://poloai.top",
        api_key="sk-YourKey",
        model="claude-sonnet-4-20250514-thinking"
    )
    
    # 设置 System Prompt（用于建模场景）
    set_system_prompt("你是一位专业的AutoML专家，专门帮助用户完成机器学习建模任务。")
    
    # 获取客户端
    llm = get_llm_client()

作者: AutoML Team
"""

import os
import yaml
from typing import Any, Optional

# 默认的建模场景 System Prompt
DEFAULT_SYSTEM_PROMPT = """你是一位专业的 AutoML 专家，专门帮助用户完成机器学习建模任务。

你的主要职责包括：
1. 数据分析和预处理
2. 特征工程（传统方法和LLM驱动的方法）
3. 模型选择和训练
4. 模型评估和优化

重要规则：
- 只回答与机器学习建模相关的问题
- 如果用户问的问题与建模无关，礼貌地拒绝并引导用户回到建模主题
- 提供专业、准确、可执行的建议
- 在生成代码时，确保代码安全、可靠

请始终围绕建模任务来回答问题。"""

# 全局 System Prompt
_global_system_prompt = DEFAULT_SYSTEM_PROMPT


def load_config_from_file(config_path: str = "config.yaml") -> dict:
    """从配置文件加载LLM配置"""
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    
    if not os.path.exists(config_file):
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config and 'llm' in config:
                return config['llm']
    except Exception as e:
        print(f"警告: 加载配置文件失败: {e}")
    
    return {}


def set_system_prompt(prompt: str) -> None:
    """设置全局 System Prompt"""
    global _global_system_prompt
    _global_system_prompt = prompt


def get_system_prompt() -> str:
    """获取当前 System Prompt"""
    return _global_system_prompt


def reset_system_prompt() -> None:
    """重置为默认 System Prompt"""
    global _global_system_prompt
    _global_system_prompt = DEFAULT_SYSTEM_PROMPT


# 优先尝试使用langchain_openai（更好的LangChain兼容性）
try:
    from langchain_openai import ChatOpenAI as _ChatOpenAI
    
    # 从配置文件加载默认配置
    _file_config = load_config_from_file()
    
    # 全局配置
    _global_config = {
        "base_url": _file_config.get("base_url", "https://fast.poloai.top/v1"),
        "api_key": _file_config.get("api_key", ""),
        "model": _file_config.get("model", "claude-sonnet-4-20250514-thinking"),
        "temperature": _file_config.get("temperature", 0),
    }
    
    def configure_llm(
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        max_retries: int = 3
    ) -> dict:
        """配置全局LLM（显式参数优先级最高）"""
        global _global_config
        
        # 优先使用配置文件
        file_config = load_config_from_file()
        
        # 确定最终配置（显式参数 > 配置文件 > 默认值）
        final_base_url = base_url or file_config.get("base_url", "https://fast.poloai.top")
        final_api_key = api_key or file_config.get("api_key", "")
        final_model = model or file_config.get("model", "claude-sonnet-4-20250514-thinking")
        
        # 确保base_url包含/v1路径
        if final_base_url and "/v1" not in final_base_url:
            final_base_url = final_base_url.rstrip("/") + "/v1"
        
        _global_config.update({
            "base_url": final_base_url,
            "api_key": final_api_key,
            "model": final_model,
            "temperature": temperature,
        })
        
        return _global_config
    
    def get_llm_client(
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> _ChatOpenAI:
        """获取LLM客户端实例
        
        Args:
            base_url: API 基础 URL
            api_key: API 密钥
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            system_prompt: System Prompt（如果为None则使用全局设置）
        
        Returns:
            配置好的 LLM 客户端
        """
        global _global_config, _global_system_prompt
        
        # 确定使用的 system prompt
        used_system_prompt = system_prompt if system_prompt else _global_system_prompt
        
        # 优先使用配置文件
        file_config = load_config_from_file()
        
        # 合并配置：显式参数 > 配置文件 > 全局配置
        cfg = {**_global_config}
        
        # 如果有显式参数则覆盖
        if base_url:
            cfg["base_url"] = base_url if "/v1" in base_url else base_url + "/v1"
        if api_key:
            cfg["api_key"] = api_key
        if model:
            cfg["model"] = model
        
        # 如果全局配置为空，从配置文件补充
        if not cfg.get("api_key") and file_config.get("api_key"):
            cfg["api_key"] = file_config.get("api_key")
        if not cfg.get("base_url") or cfg.get("base_url") == "https://fast.poloai.top/v1":
            base_from_file = file_config.get("base_url", "https://fast.poloai.top")
            if "/v1" not in base_from_file:
                base_from_file = base_from_file.rstrip("/") + "/v1"
            cfg["base_url"] = base_from_file
        if not cfg.get("model") or cfg.get("model") == "claude-sonnet-4-20250514-thinking":
            cfg["model"] = file_config.get("model", "claude-sonnet-4-20250514-thinking")
        
        # 创建客户端
        llm = _ChatOpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # 将 system prompt 存储在 llm 对象上，供后续使用
        if used_system_prompt:
            llm._automl_system_prompt = used_system_prompt
        
        return llm
    
    def get_config() -> dict:
        """获取当前配置"""
        return _global_config.copy()

except ImportError:
    # 如果没有langchain_openai，使用自定义实现
    import http.client
    import json
    from typing import Any, Optional
    from pydantic import BaseModel, Field
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    
    class LLMConfig(BaseModel):
        """LLM配置模型"""
        base_url: str = "https://fast.poloai.top"
        api_key: str = ""
        model: str = "claude-sonnet-4-20250514-thinking"
        temperature: float = 0
        max_tokens: Optional[int] = None
        timeout: int = 60
        max_retries: int = 3
    
    class CustomLLMClient:
        """自定义LLM客户端（兼容LangChain）"""
        
        def __init__(
            self,
            base_url: str = "https://fast.poloai.top",
            api_key: str = "",
            model: str = "claude-sonnet-4-20250514-thinking",
            temperature: float = 0,
            max_tokens: Optional[int] = None,
            timeout: int = 60,
            max_retries: int = 3,
            system_prompt: Optional[str] = None,
            **kwargs
        ):
            global _global_system_prompt
            
            self.config = LLMConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries
            )
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.model_name = model
            self.callback_manager = None
            self.verbose = False
            # 使用传入的 system_prompt 或全局 system_prompt
            self.system_prompt = system_prompt if system_prompt else _global_system_prompt
        
        @property
        def _identifying_params(self) -> dict:
            return {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
        
        @property
        def _llm_type(self) -> str:
            return "custom_chat"
        
        def _convert_message(self, message: BaseMessage) -> dict:
            if isinstance(message, HumanMessage):
                return {"role": "user", "content": message.content}
            elif isinstance(message, AIMessage):
                return {"role": "assistant", "content": message.content}
            elif isinstance(message, SystemMessage):
                return {"role": "system", "content": message.content}
            return {"role": "user", "content": str(message)}
        
        def _call(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            **kwargs
        ) -> str:
            # 将 messages 转换为 API 格式
            api_messages = [self._convert_message(msg) for msg in messages]
            
            # 如果存在 system prompt 且 messages 中没有，则添加到最前面
            if self.system_prompt:
                has_system = any(msg["role"] == "system" for msg in api_messages)
                if not has_system:
                    api_messages.insert(0, {"role": "system", "content": self.system_prompt})
            
            payload = {
                "model": self.config.model,
                "messages": api_messages,
                "temperature": self.config.temperature,
            }
            
            if self.config.max_tokens:
                payload["max_tokens"] = self.config.max_tokens
            if stop:
                payload["stop"] = stop
            
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            from urllib.parse import urlparse
            parsed = urlparse(self.config.base_url)
            host = parsed.netloc
            path = parsed.path if parsed.path else "/v1/chat/completions"
            
            conn = http.client.HTTPSConnection(host, timeout=self.config.timeout)
            conn.request("POST", path, body=json.dumps(payload), headers=headers)
            response = conn.getresponse()
            data = response.read().decode("utf-8")
            conn.close()
            
            result = json.loads(data)
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            elif "error" in result:
                raise Exception(f"API错误: {result['error']}")
            raise Exception(f"未知响应: {data}")
        
        def invoke(self, input: Any, **kwargs) -> AIMessage:
            if isinstance(input, str):
                messages = [HumanMessage(content=input)]
            elif isinstance(input, list):
                messages = input
            else:
                messages = [HumanMessage(content=str(input))]
            
            content = self._call(messages, **kwargs)
            return AIMessage(content=content)
        
        def __repr__(self) -> str:
            return f"CustomLLMClient(model={self.config.model}, temperature={self.config.temperature})"
    
    # 全局配置
    _global_config = None
    
    def configure_llm(
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        max_retries: int = 3
    ) -> LLMConfig:
        global _global_config
        
        # 从配置文件加载
        file_config = load_config_from_file()
        
        # 确定最终配置（显式参数 > 配置文件 > 默认值）
        final_base_url = base_url or file_config.get("base_url", "https://fast.poloai.top")
        final_api_key = api_key or file_config.get("api_key", "")
        final_model = model or file_config.get("model", "claude-sonnet-4-20250514-thinking")
        
        _global_config = LLMConfig(
            base_url=final_base_url,
            api_key=final_api_key,
            model=final_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries
        )
        
        return _global_config
    
    def get_llm_client(
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> CustomLLMClient:
        global _global_config
        
        # 从配置文件加载
        file_config = load_config_from_file()
        
        # 合并配置：显式参数 > 配置文件 > 全局配置
        cfg_dict = {}
        if _global_config:
            cfg_dict = {
                "base_url": _global_config.base_url,
                "api_key": _global_config.api_key,
                "model": _global_config.model,
                "temperature": _global_config.temperature,
            }
        
        # 如果有显式参数则覆盖
        if base_url:
            cfg_dict["base_url"] = base_url
        if api_key:
            cfg_dict["api_key"] = api_key
        if model:
            cfg_dict["model"] = model
        
        # 如果全局配置为空，从配置文件补充
        if not cfg_dict.get("api_key") and file_config.get("api_key"):
            cfg_dict["api_key"] = file_config.get("api_key")
        if not cfg_dict.get("base_url"):
            cfg_dict["base_url"] = file_config.get("base_url", "https://fast.poloai.top") + "/v1"
        if not cfg_dict.get("model"):
            cfg_dict["model"] = file_config.get("model", "claude-sonnet-4-20250514-thinking")
        
        # 传递 system_prompt
        if system_prompt:
            cfg_dict["system_prompt"] = system_prompt
        
        return CustomLLMClient(**cfg_dict)
    
    def get_config() -> Optional[LLMConfig]:
        return _global_config

# 便捷函数
def create_llm(model: str = "claude-sonnet4-thinking", temperature: float = 0, **kwargs) -> Any:
    """创建LLM实例的便捷函数"""
    return get_llm_client(model=model, temperature=temperature, **kwargs)


if __name__ == "__main__":
    # 测试
    print("测试LLM客户端...")
    
    llm = get_llm_client()
    print(f"LLM: {llm}")
    
    response = llm.invoke("你好")
    print(f"响应: {response.content}")
