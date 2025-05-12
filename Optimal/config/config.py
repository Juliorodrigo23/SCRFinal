# config/config.py
import os
import json
from typing import Dict, Any

class Config:
    """Configuration management for the project."""
    
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize configuration from file or create default."""
        self.config_path = config_path
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._create_default_config()
            self._save_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            "api_keys": {
                "openai": os.environ.get("OPENAI_API_KEY", ""),
                "elevenlabs": os.environ.get("ELEVENLABS_API_KEY", "")
            },
            "voice_mapping": {
                "Jack": "older-male-1",
                "Maria": "older-female-1",
                "Chen": "older-male-2",
                "Fatima": "older-female-2",
                "Robert": "older-male-3",
                "robot": "robot-assistant"
            },
            "sentiment_thresholds": {
                "positive": 0.3,
                "negative": -0.3
            },
            "approach_threshold": 0.5
        }
    
    def _save_config(self):
        """Save configuration to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default=None):
        """Get a configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value):
        """Set a configuration value."""
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._save_config()