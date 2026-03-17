"""
Prompt 加载和管理模块

该模块提供统一的 Prompt 加载和管理功能，支持从 YAML 配置文件中加载
预定义的 Prompt 模板，并提供参数替换功能。

使用示例:
    from core.prompt_loader import PromptLoader
    
    # 初始化加载器
    loader = PromptLoader()
    
    # 获取并格式化 Prompt
    prompt = loader.get_prompt("planner", "task_analysis", 
                              goal="预测销量", 
                              data_shape="(1000, 20)")
"""

import yaml
import os
from typing import Dict, Any, Optional


class PromptLoader:
    """
    Prompt 加载器
    
    负责从配置文件中加载预定义的 Prompt 模板，并提供格式化功能。
    """
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        初始化 Prompt 加载器
        
        Args:
            prompts_dir: prompts 配置文件所在的目录
        """
        self.prompts_dir = prompts_dir
        self._cache = {}
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """加载所有 Prompt 配置文件到缓存"""
        if not os.path.exists(self.prompts_dir):
            raise FileNotFoundError(f"Prompts 目录不存在: {self.prompts_dir}")
        
        for filename in os.listdir(self.prompts_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                module_name = filename.replace('_prompts.yaml', '').replace('_prompts.yml', '')
                filepath = os.path.join(self.prompts_dir, filename)
                self._cache[module_name] = self._load_yaml(filepath)
    
    def _load_yaml(self, filepath: str) -> Dict[str, str]:
        """
        从 YAML 文件加载 Prompt 配置
        
        Args:
            filepath: YAML 文件路径
            
        Returns:
            Dict: Prompt 名称到内容的映射
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_prompt(self, module: str, prompt_name: str, **kwargs) -> str:
        """
        获取并格式化 Prompt
        
        Args:
            module: 模块名 (如 'planner', 'feature', 'model', 'data')
            prompt_name: Prompt 名称
            **kwargs: 用于格式化 Prompt 的参数
            
        Returns:
            str: 格式化后的 Prompt 字符串
        """
        if module not in self._cache:
            raise KeyError(f"模块 '{module}' 的 Prompt 配置不存在")
        
        if prompt_name not in self._cache[module]:
            raise KeyError(f"模块 '{module}' 中不存在名为 '{prompt_name}' 的 Prompt")
        
        prompt_template = self._cache[module][prompt_name]
        
        if kwargs:
            try:
                return prompt_template.format(**kwargs)
            except KeyError as e:
                raise KeyError(f"Prompt 格式化失败，缺少参数: {e}")
        else:
            return prompt_template
    
    def get_raw_prompt(self, module: str, prompt_name: str) -> str:
        """
        获取原始 Prompt 模板（不进行格式化）
        
        Args:
            module: 模块名
            prompt_name: Prompt 名称
            
        Returns:
            str: 原始 Prompt 模板字符串
        """
        if module not in self._cache:
            raise KeyError(f"模块 '{module}' 的 Prompt 配置不存在")
        
        if prompt_name not in self._cache[module]:
            raise KeyError(f"模块 '{module}' 中不存在名为 '{prompt_name}' 的 Prompt")
        
        return self._cache[module][prompt_name]


# 全局 Prompt 加载器实例
_prompt_loader = None


def get_prompt_loader() -> PromptLoader:
    """
    获取全局 Prompt 加载器实例
    
    Returns:
        PromptLoader: 全局 Prompt 加载器实例
    """
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader


def load_prompt(module: str, prompt_name: str, **kwargs) -> str:
    """
    便捷函数：加载并格式化 Prompt
    
    Args:
        module: 模块名
        prompt_name: Prompt 名称
        **kwargs: 用于格式化 Prompt 的参数
        
    Returns:
        str: 格式化后的 Prompt 字符串
    """
    loader = get_prompt_loader()
    return loader.get_prompt(module, prompt_name, **kwargs)


def load_raw_prompt(module: str, prompt_name: str) -> str:
    """
    便捷函数：加载原始 Prompt 模板
    
    Args:
        module: 模块名
        prompt_name: Prompt 名称
        
    Returns:
        str: 原始 Prompt 模板字符串
    """
    loader = get_prompt_loader()
    return loader.get_raw_prompt(module, prompt_name)