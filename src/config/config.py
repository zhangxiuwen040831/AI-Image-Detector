import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """配置管理类，用于加载和管理不同环境的配置"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.current_config: Optional[Dict[str, Any]] = None
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """加载指定的配置文件"""
        config_path = self.config_dir / f"{config_name}.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        self.configs[config_name] = config
        self.current_config = config
        return config
    
    def get_config(self, config_name: str = None) -> Dict[str, Any]:
        """获取指定的配置，如果没有指定则返回当前配置"""
        if config_name:
            if config_name not in self.configs:
                return self.load_config(config_name)
            return self.configs[config_name]
        elif self.current_config:
            return self.current_config
        else:
            raise ValueError("No config loaded")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的路径"""
        if not self.current_config:
            raise ValueError("No config loaded")
        
        keys = key.split(".")
        value = self.current_config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, updates: Dict[str, Any]):
        """更新当前配置"""
        if not self.current_config:
            raise ValueError("No config loaded")
        
        def _update_recursive(config: Dict[str, Any], updates: Dict[str, Any]):
            for key, value in updates.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    _update_recursive(config[key], value)
                else:
                    config[key] = value
        
        _update_recursive(self.current_config, updates)


# 创建全局配置管理器实例
config_manager = ConfigManager()


def load_config(config_name: str) -> Dict[str, Any]:
    """加载配置的便捷函数"""
    return config_manager.load_config(config_name)


def get_config(config_name: str = None) -> Dict[str, Any]:
    """获取配置的便捷函数"""
    return config_manager.get_config(config_name)


def get(key: str, default: Any = None) -> Any:
    """获取配置值的便捷函数"""
    return config_manager.get(key, default)


def update_config(updates: Dict[str, Any]):
    """更新配置的便捷函数"""
    config_manager.update(updates)