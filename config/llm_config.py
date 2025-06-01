"""
LLM configuration for Agentix framework.

This module provides configuration for different LLM providers and models.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OPENROUTER = "openrouter"
    LOCAL = "local"


class LLMConfig(BaseModel):
    """Configuration for LLM providers and models."""

    # Provider settings
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=100000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

    # Request settings
    timeout: int = Field(default=60, ge=1, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0)

    # Streaming settings
    enable_streaming: bool = True
    stream_chunk_size: int = 1024

    # Tool calling settings
    enable_function_calling: bool = True
    max_function_calls: int = 10

    # Safety settings
    enable_content_filter: bool = True
    content_filter_threshold: float = 0.8

    # Caching settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds

    # Provider-specific settings
    provider_config: Dict[str, Any] = Field(default_factory=dict)

    @validator('model')
    def validate_model(cls, v, values):
        """Validate model name based on provider."""
        provider = values.get('provider')

        if provider == LLMProvider.OPENAI:
            valid_models = [
                'gpt-4', 'gpt-4-turbo', 'gpt-4-turbo-preview',
                'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'
            ]
            if v not in valid_models:
                # Allow custom models for flexibility
                pass

        elif provider == LLMProvider.ANTHROPIC:
            valid_models = [
                'claude-3-opus-20240229', 'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307', 'claude-2.1', 'claude-2.0'
            ]
            if v not in valid_models:
                # Allow custom models for flexibility
                pass

        elif provider == LLMProvider.OPENROUTER:
            # OpenRouter supports many models, allow any model name
            # Popular models include:
            # - anthropic/claude-3-opus
            # - openai/gpt-4-turbo
            # - google/gemini-pro
            # - meta-llama/llama-2-70b-chat
            # - mistralai/mixtral-8x7b-instruct
            pass

        return v

    def get_provider_config(self) -> Dict[str, Any]:
        """Get provider-specific configuration."""
        base_config = {
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'timeout': self.timeout
        }

        if self.provider == LLMProvider.OPENAI:
            base_config.update({
                'frequency_penalty': self.frequency_penalty,
                'presence_penalty': self.presence_penalty,
                'stream': self.enable_streaming
            })

        elif self.provider == LLMProvider.ANTHROPIC:
            base_config.update({
                'stream': self.enable_streaming
            })

        elif self.provider == LLMProvider.OPENROUTER:
            base_config.update({
                'stream': self.enable_streaming,
                'frequency_penalty': self.frequency_penalty,
                'presence_penalty': self.presence_penalty,
                # OpenRouter-specific parameters
                'top_k': self.provider_config.get('top_k', 40),
                'repetition_penalty': self.provider_config.get('repetition_penalty', 1.1),
                'min_p': self.provider_config.get('min_p', 0.0),
                'top_a': self.provider_config.get('top_a', 0.0)
            })

        # Add provider-specific overrides
        base_config.update(self.provider_config)

        return base_config

    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Agentix/0.1.0'
        }

        if self.api_key:
            if self.provider == LLMProvider.OPENAI:
                headers['Authorization'] = f'Bearer {self.api_key}'
            elif self.provider == LLMProvider.ANTHROPIC:
                headers['x-api-key'] = self.api_key
                headers['anthropic-version'] = '2023-06-01'
            elif self.provider == LLMProvider.GOOGLE:
                headers['Authorization'] = f'Bearer {self.api_key}'
            elif self.provider == LLMProvider.COHERE:
                headers['Authorization'] = f'Bearer {self.api_key}'
            elif self.provider == LLMProvider.OPENROUTER:
                headers['Authorization'] = f'Bearer {self.api_key}'
                headers['HTTP-Referer'] = 'https://agentix.dev'  # Optional: for analytics
                headers['X-Title'] = 'Agentix Framework'  # Optional: for analytics

        return headers

    def get_api_url(self) -> str:
        """Get API URL for the provider."""
        if self.api_base:
            return self.api_base

        if self.provider == LLMProvider.OPENAI:
            return "https://api.openai.com/v1"
        elif self.provider == LLMProvider.ANTHROPIC:
            return "https://api.anthropic.com"
        elif self.provider == LLMProvider.GOOGLE:
            return "https://generativelanguage.googleapis.com/v1"
        elif self.provider == LLMProvider.COHERE:
            return "https://api.cohere.ai/v1"
        elif self.provider == LLMProvider.HUGGINGFACE:
            return "https://api-inference.huggingface.co"
        elif self.provider == LLMProvider.OPENROUTER:
            return "https://openrouter.ai/api/v1"
        else:
            return "http://localhost:8000"  # Local provider default


class LLMConfigManager:
    """Manager for multiple LLM configurations."""

    def __init__(self):
        self.configs: Dict[str, LLMConfig] = {}
        self.default_config_name: Optional[str] = None

    def add_config(self, name: str, config: LLMConfig, set_as_default: bool = False):
        """Add an LLM configuration."""
        self.configs[name] = config

        if set_as_default or not self.default_config_name:
            self.default_config_name = name

    def get_config(self, name: Optional[str] = None) -> Optional[LLMConfig]:
        """Get an LLM configuration by name."""
        if name is None:
            name = self.default_config_name

        return self.configs.get(name)

    def list_configs(self) -> List[str]:
        """List all configuration names."""
        return list(self.configs.keys())

    def remove_config(self, name: str):
        """Remove an LLM configuration."""
        if name in self.configs:
            del self.configs[name]

            if self.default_config_name == name:
                self.default_config_name = next(iter(self.configs.keys()), None)

    def set_default(self, name: str):
        """Set the default configuration."""
        if name in self.configs:
            self.default_config_name = name
        else:
            raise ValueError(f"Configuration '{name}' not found")

    def create_openai_config(self, name: str, model: str = "gpt-4",
                           api_key: Optional[str] = None, **kwargs) -> LLMConfig:
        """Create and add an OpenAI configuration."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model=model,
            api_key=api_key,
            **kwargs
        )
        self.add_config(name, config)
        return config

    def create_anthropic_config(self, name: str, model: str = "claude-3-sonnet-20240229",
                              api_key: Optional[str] = None, **kwargs) -> LLMConfig:
        """Create and add an Anthropic configuration."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model=model,
            api_key=api_key,
            **kwargs
        )
        self.add_config(name, config)
        return config

    def create_google_config(self, name: str, model: str = "gemini-pro",
                           api_key: Optional[str] = None, **kwargs) -> LLMConfig:
        """Create and add a Google configuration."""
        config = LLMConfig(
            provider=LLMProvider.GOOGLE,
            model=model,
            api_key=api_key,
            **kwargs
        )
        self.add_config(name, config)
        return config

    def create_openrouter_config(self, name: str, model: str = "openai/gpt-4-turbo",
                               api_key: Optional[str] = None, **kwargs) -> LLMConfig:
        """Create and add an OpenRouter configuration."""
        config = LLMConfig(
            provider=LLMProvider.OPENROUTER,
            model=model,
            api_key=api_key,
            **kwargs
        )
        self.add_config(name, config)
        return config

    def create_local_config(self, name: str, model: str = "local-model",
                          api_base: str = "http://localhost:8000", **kwargs) -> LLMConfig:
        """Create and add a local LLM configuration."""
        config = LLMConfig(
            provider=LLMProvider.LOCAL,
            model=model,
            api_base=api_base,
            **kwargs
        )
        self.add_config(name, config)
        return config


# Global LLM config manager
_global_llm_manager: Optional[LLMConfigManager] = None


def get_global_llm_manager() -> LLMConfigManager:
    """Get the global LLM configuration manager."""
    global _global_llm_manager
    if _global_llm_manager is None:
        _global_llm_manager = LLMConfigManager()

        # Add default configurations if API keys are available
        import os

        if os.getenv('OPENAI_API_KEY'):
            _global_llm_manager.create_openai_config(
                'default_openai',
                api_key=os.getenv('OPENAI_API_KEY'),
                set_as_default=True
            )

        if os.getenv('ANTHROPIC_API_KEY'):
            _global_llm_manager.create_anthropic_config(
                'default_anthropic',
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )

        if os.getenv('GOOGLE_API_KEY'):
            _global_llm_manager.create_google_config(
                'default_google',
                api_key=os.getenv('GOOGLE_API_KEY')
            )

        if os.getenv('OPENROUTER_API_KEY'):
            _global_llm_manager.create_openrouter_config(
                'default_openrouter',
                api_key=os.getenv('OPENROUTER_API_KEY')
            )

    return _global_llm_manager


def get_default_llm_config() -> Optional[LLMConfig]:
    """Get the default LLM configuration."""
    manager = get_global_llm_manager()
    return manager.get_config()


def set_default_llm_config(config: LLMConfig, name: str = "default"):
    """Set the default LLM configuration."""
    manager = get_global_llm_manager()
    manager.add_config(name, config, set_as_default=True)
