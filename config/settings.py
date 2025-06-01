"""
Main configuration settings for Agentix framework.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AgentixConfig(BaseModel):
    """Main configuration for Agentix framework."""
    
    # Framework settings
    framework_version: str = "0.1.0"
    environment: str = "development"  # development, staging, production
    
    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Agent settings
    default_agent_timeout: int = 300  # seconds
    max_concurrent_agents: int = 10
    agent_memory_limit: int = 1024 * 1024 * 1024  # 1GB
    
    # Memory system configuration
    memory_config: Dict[str, Any] = Field(default_factory=lambda: {
        "enable_temporal_graph": True,
        "enable_episodic_memory": True,
        "enable_semantic_memory": True,
        "working_memory_size": 1000,
        "auto_consolidation": True,
        "consolidation_interval": 3600
    })
    
    # Tool system configuration
    tool_config: Dict[str, Any] = Field(default_factory=lambda: {
        "default_timeout": 30,
        "max_retries": 3,
        "enable_validation": True,
        "require_approval": False
    })
    
    # LLM configuration
    llm_config: Dict[str, Any] = Field(default_factory=lambda: {
        "default_provider": "openai",
        "default_model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "timeout": 60
    })
    
    # Security settings
    security_config: Dict[str, Any] = Field(default_factory=lambda: {
        "enable_guardrails": True,
        "validate_inputs": True,
        "validate_outputs": True,
        "blocked_domains": [],
        "allowed_file_extensions": [".txt", ".json", ".csv", ".md"],
        "max_file_size": 10 * 1024 * 1024  # 10MB
    })
    
    # Performance settings
    performance_config: Dict[str, Any] = Field(default_factory=lambda: {
        "enable_caching": True,
        "cache_ttl": 3600,
        "max_cache_size": 1000,
        "enable_metrics": True,
        "metrics_interval": 60
    })
    
    # API settings
    api_config: Dict[str, Any] = Field(default_factory=lambda: {
        "host": "0.0.0.0",
        "port": 8000,
        "enable_cors": True,
        "enable_docs": True,
        "rate_limit": 100,
        "rate_limit_window": 60
    })
    
    # Database settings (if using external storage)
    database_config: Optional[Dict[str, Any]] = None
    
    # External service configurations
    external_services: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f"environment must be one of {valid_envs}")
        return v
    
    @validator('max_concurrent_agents')
    def validate_max_concurrent_agents(cls, v):
        if v < 1 or v > 100:
            raise ValueError("max_concurrent_agents must be between 1 and 100")
        return v
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory system configuration."""
        return self.memory_config.copy()
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool system configuration."""
        return self.tool_config.copy()
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.llm_config.copy()
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self.security_config.copy()
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)


def load_config(config_path: Optional[str] = None) -> AgentixConfig:
    """
    Load configuration from file or environment variables.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        
    Returns:
        AgentixConfig instance
    """
    config_data = {}
    
    # Load from file if provided
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_file.suffix}")
    
    # Override with environment variables
    env_overrides = _load_from_environment()
    config_data.update(env_overrides)
    
    return AgentixConfig(**config_data)


def save_config(config: AgentixConfig, config_path: str, format: str = "yaml"):
    """
    Save configuration to file.
    
    Args:
        config: AgentixConfig instance to save
        config_path: Path where to save the configuration
        format: File format ('yaml' or 'json')
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    config_data = config.dict()
    
    with open(config_file, 'w') as f:
        if format.lower() == 'yaml':
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            json.dump(config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


def _load_from_environment() -> Dict[str, Any]:
    """Load configuration overrides from environment variables."""
    env_config = {}
    
    # Framework settings
    if os.getenv('AGENTIX_ENVIRONMENT'):
        env_config['environment'] = os.getenv('AGENTIX_ENVIRONMENT')
    
    if os.getenv('AGENTIX_LOG_LEVEL'):
        env_config['log_level'] = os.getenv('AGENTIX_LOG_LEVEL')
    
    if os.getenv('AGENTIX_LOG_FILE'):
        env_config['log_file'] = os.getenv('AGENTIX_LOG_FILE')
    
    # Agent settings
    if os.getenv('AGENTIX_MAX_CONCURRENT_AGENTS'):
        env_config['max_concurrent_agents'] = int(os.getenv('AGENTIX_MAX_CONCURRENT_AGENTS'))
    
    if os.getenv('AGENTIX_AGENT_TIMEOUT'):
        env_config['default_agent_timeout'] = int(os.getenv('AGENTIX_AGENT_TIMEOUT'))
    
    # LLM settings
    llm_config = {}
    if os.getenv('AGENTIX_LLM_PROVIDER'):
        llm_config['default_provider'] = os.getenv('AGENTIX_LLM_PROVIDER')
    
    if os.getenv('AGENTIX_LLM_MODEL'):
        llm_config['default_model'] = os.getenv('AGENTIX_LLM_MODEL')
    
    if os.getenv('AGENTIX_LLM_TEMPERATURE'):
        llm_config['temperature'] = float(os.getenv('AGENTIX_LLM_TEMPERATURE'))
    
    if llm_config:
        env_config['llm_config'] = llm_config
    
    # API settings
    api_config = {}
    if os.getenv('AGENTIX_API_HOST'):
        api_config['host'] = os.getenv('AGENTIX_API_HOST')
    
    if os.getenv('AGENTIX_API_PORT'):
        api_config['port'] = int(os.getenv('AGENTIX_API_PORT'))
    
    if api_config:
        env_config['api_config'] = api_config
    
    # External service API keys
    external_services = {}
    
    # OpenAI
    if os.getenv('OPENAI_API_KEY'):
        external_services['openai'] = {'api_key': os.getenv('OPENAI_API_KEY')}
    
    # Anthropic
    if os.getenv('ANTHROPIC_API_KEY'):
        external_services['anthropic'] = {'api_key': os.getenv('ANTHROPIC_API_KEY')}
    
    # Google
    if os.getenv('GOOGLE_API_KEY'):
        external_services['google'] = {'api_key': os.getenv('GOOGLE_API_KEY')}
    
    # Bing
    if os.getenv('BING_API_KEY'):
        external_services['bing'] = {'api_key': os.getenv('BING_API_KEY')}
    
    if external_services:
        env_config['external_services'] = external_services
    
    return env_config


def get_default_config() -> AgentixConfig:
    """Get default configuration."""
    return AgentixConfig()


def create_config_template(output_path: str, format: str = "yaml"):
    """
    Create a configuration template file.
    
    Args:
        output_path: Path where to save the template
        format: File format ('yaml' or 'json')
    """
    default_config = get_default_config()
    save_config(default_config, output_path, format)


# Global configuration instance
_global_config: Optional[AgentixConfig] = None


def get_global_config() -> AgentixConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_global_config(config: AgentixConfig):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def reset_global_config():
    """Reset the global configuration to default."""
    global _global_config
    _global_config = None
