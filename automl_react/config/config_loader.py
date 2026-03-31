"""
配置加载器模块

从配置文件加载 Prompt、LLM 配置和工作流配置
"""

import os
import re
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """
    配置加载器

    负责从 YAML 配置文件加载所有配置

    Attributes:
        config_dir: 配置文件目录
        _prompts: Prompt 配置缓存
        _llm_config: LLM 配置缓存
        _workflow_config: 工作流配置缓存
    """

    def __init__(self, config_dir: str = None):
        """
        初始化配置加载器

        Args:
            config_dir: 配置文件目录，默认为项目根目录的 config/
        """
        if config_dir is None:
            # 默认配置目录：项目根目录/config
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            config_dir = project_root / "automl_react" / "config"

        self.config_dir = Path(config_dir)
        self._prompts: Dict[str, Any] = {}
        self._llm_config: Dict[str, Any] = {}
        self._workflow_config: Dict[str, Any] = {}

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        加载 YAML 文件

        Args:
            filename: 文件名

        Returns:
            配置字典
        """
        filepath = self.config_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"配置文件不存在: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_prompts(self, reload: bool = False) -> Dict[str, Any]:
        """
        加载 Prompt 配置

        Args:
            reload: 是否强制重新加载

        Returns:
            Prompt 配置字典
        """
        if not self._prompts or reload:
            self._prompts = self._load_yaml("prompts.yaml")
        return self._prompts

    def load_llm_config(self, reload: bool = False) -> Dict[str, Any]:
        """
        加载 LLM 配置

        Args:
            reload: 是否强制重新加载

        Returns:
            LLM 配置字典
        """
        if not self._llm_config or reload:
            self._llm_config = self._load_yaml("llm_config.yaml")
        return self._llm_config

    def load_workflow_config(self, reload: bool = False) -> Dict[str, Any]:
        """
        加载工作流配置

        Args:
            reload: 是否强制重新加载

        Returns:
            工作流配置字典
        """
        if not self._workflow_config or reload:
            self._workflow_config = self._load_yaml("workflow_config.yaml")
        return self._workflow_config

    def get_prompt(self, section: str, key: str = None) -> str:
        """
        获取指定 Prompt

        Args:
            section: 配置节（如 data_cleaning, feature_engineering）
            key: 配置键（如 system_prompt, user_prompt），为 None 时返回整个 section

        Returns:
            Prompt 字符串或配置字典
        """
        prompts = self.load_prompts()

        section_aliases = {
            "data_analysis": "problem_definition",
        }
        key_aliases = {
            ("data_cleaning", "quality_analysis_prompt"): "quality_analysis_prompt",
            ("data_cleaning", "data_analysis_prompt"): "quality_analysis_prompt",
        }

        resolved_section = section_aliases.get(section, section)
        resolved_key = key_aliases.get((section, key), key)

        if resolved_section not in prompts:
            raise KeyError(f"Prompt section '{section}' 不存在")

        if key is None:
            return prompts[resolved_section]

        if resolved_key not in prompts[resolved_section]:
            raise KeyError(f"Prompt key '{key}' 在 section '{section}' 中不存在")

        return prompts[resolved_section][resolved_key]

    def get_llm_config(self, model_name: str = None) -> Dict[str, Any]:
        """
        获取 LLM 配置（自动解析环境变量）

        Args:
            model_name: 模型名称，为 None 时返回默认配置

        Returns:
            LLM 配置字典
        """
        config = self.load_llm_config()

        if model_name is None:
            # 返回默认模型配置
            default_model = config.get("default_model", "gpt-4")
            model_config = config.get("models", {}).get(default_model, {})
        elif model_name in config.get("models", {}):
            model_config = config["models"][model_name].copy()
        else:
            raise KeyError(f"模型 '{model_name}' 配置不存在")

        # 解析环境变量引用
        return self._resolve_env_vars(model_config)

    def _resolve_env_vars(self, config: Any) -> Any:
        """
        递归解析配置中的环境变量引用

        支持格式：
        - ${ENV_VAR} - 必需的环境变量
        - ${ENV_VAR:default} - 带默认值的环境变量

        Args:
            config: 配置值（可以是字典、列表或字符串）

        Returns:
            解析后的配置值
        """
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._resolve_env_var_string(config)
        return config

    def _resolve_env_var_string(self, value: str) -> str:
        """
        解析字符串中的环境变量引用

        Args:
            value: 可能包含环境变量引用的字符串

        Returns:
            解析后的字符串
        """
        pattern = r'\$\{([^}]+)\}'

        def replace_env_var(match):
            env_expr = match.group(1)
            if ':' in env_expr:
                env_name, default_value = env_expr.split(':', 1)
                return os.environ.get(env_name, default_value)
            else:
                return os.environ.get(env_expr, match.group(0))

        return re.sub(pattern, replace_env_var, value)

    def get_stage_model(self, stage: str) -> Dict[str, Any]:
        """
        获取指定阶段使用的模型配置

        Args:
            stage: 阶段名称（如 data_cleaning, feature_engineering）

        Returns:
            模型配置字典
        """
        config = self.load_llm_config()
        stage_models = config.get("stage_models", {})

        # 获取阶段指定的模型名称，或使用默认模型
        model_name = stage_models.get(stage)
        if model_name:
            return self.get_llm_config(model_name)

        # 返回默认模型配置
        return self.get_llm_config()

    def get_logging_config(self) -> Dict[str, Any]:
        """
        获取日志配置

        Returns:
            日志配置字典
        """
        config = self.load_llm_config()
        return config.get("logging", {
            "enabled": True,
            "log_dir": "logs/llm_calls",
            "log_format": "jsonl"
        })

    def get_retry_config(self) -> Dict[str, Any]:
        """
        获取重试配置

        Returns:
            重试配置字典
        """
        config = self.load_llm_config()
        return config.get("retry", {
            "max_retries": 3,
            "retry_delay": 1.0,
            "exponential_backoff": True
        })

    def get_workflow_config(self, key: str = None) -> Any:
        """
        获取工作流配置

        Args:
            key: 配置键，为 None 时返回全部配置

        Returns:
            配置值或配置字典
        """
        config = self.load_workflow_config()

        if key is None:
            return config

        if key not in config:
            raise KeyError(f"工作流配置 '{key}' 不存在")

        return config[key]

    def reload_all(self):
        """重新加载所有配置"""
        self.load_prompts(reload=True)
        self.load_llm_config(reload=True)
        self.load_workflow_config(reload=True)

    def format_prompt(self, section: str, key: str, **kwargs) -> str:
        """
        格式化 Prompt（支持变量替换）

        Args:
            section: 配置节
            key: 配置键
            **kwargs: 变量值

        Returns:
            格式化后的 Prompt 字符串
        """
        prompt_template = self.get_prompt(section, key)

        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Prompt 模板变量 {e} 未提供")


# 全局配置加载器实例
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: str = None) -> ConfigLoader:
    """
    获取全局配置加载器实例

    Args:
        config_dir: 配置文件目录

    Returns:
        ConfigLoader 实例
    """
    global _config_loader

    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)

    return _config_loader
