import yaml
from pathlib import Path
from .paths import CONFIG_DIR

class ConfigManager:
    def __init__(self):
        self.configs = {}
    
    def load_config(self, config_type, config_name="default"):
        """
        加载配置文件
        config_type: base, train, eval, infer, deploy
        config_name: 配置文件名（不含.yml后缀）
        """
        config_path = CONFIG_DIR / config_type / f"{config_name}.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 加载基础配置
        if config_type != "base":
            base_config = self.load_config("base")
            # 合并配置
            config = self._merge_configs(base_config, config)
        
        self.configs[config_type] = config
        return config
    
    def _merge_configs(self, base, override):
        """
        合并配置字典
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_config(self, config_type):
        """
        获取已加载的配置
        """
        if config_type not in self.configs:
            return self.load_config(config_type)
        return self.configs[config_type]

# 全局配置管理器实例
config_manager = ConfigManager()

# 便捷函数
def load_config(config_type, config_name="default"):
    return config_manager.load_config(config_type, config_name)

def get_config(config_type):
    return config_manager.get_config(config_type)
