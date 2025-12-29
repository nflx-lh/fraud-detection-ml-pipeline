import yaml
import os

class ConfigReader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file '{self.config_path}' not found.")
        
        with open(self.config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
        
    def get(self, key, default=None):
        keys = key.split(".")
        value = self.config
        for k in keys:
            value = value.get(k, default) if isinstance(value, dict) else default
        return value

config = ConfigReader("monitoring/monitor_conf.yaml")