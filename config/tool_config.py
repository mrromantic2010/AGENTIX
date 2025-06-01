"""
Tool configuration management for Agentix framework.
"""

from typing import Dict, Any, Optional, List, Type
from pydantic import BaseModel, Field
import json
import yaml
from pathlib import Path

from ..tools.base import ToolConfig
from ..tools.web_search import WebSearchConfig
from ..tools.database import DatabaseConfig
from ..tools.file_operations import FileOperationsConfig
from ..tools.api_client import APIClientConfig


class ToolConfigManager:
    """Manager for tool configurations."""
    
    def __init__(self):
        self.configs: Dict[str, ToolConfig] = {}
        self.config_types: Dict[str, Type[ToolConfig]] = {
            'web_search': WebSearchConfig,
            'database': DatabaseConfig,
            'file_operations': FileOperationsConfig,
            'api_client': APIClientConfig,
            'base': ToolConfig
        }
    
    def register_config_type(self, tool_type: str, config_class: Type[ToolConfig]):
        """Register a new tool configuration type."""
        self.config_types[tool_type] = config_class
    
    def create_config(self, tool_type: str, name: str, **kwargs) -> ToolConfig:
        """Create a tool configuration."""
        if tool_type not in self.config_types:
            raise ValueError(f"Unknown tool type: {tool_type}")
        
        config_class = self.config_types[tool_type]
        config = config_class(name=name, **kwargs)
        self.configs[name] = config
        
        return config
    
    def add_config(self, name: str, config: ToolConfig):
        """Add a tool configuration."""
        self.configs[name] = config
    
    def get_config(self, name: str) -> Optional[ToolConfig]:
        """Get a tool configuration by name."""
        return self.configs.get(name)
    
    def list_configs(self) -> List[str]:
        """List all configuration names."""
        return list(self.configs.keys())
    
    def remove_config(self, name: str):
        """Remove a tool configuration."""
        if name in self.configs:
            del self.configs[name]
    
    def load_from_file(self, file_path: str):
        """Load tool configurations from file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Load configurations
        for tool_name, tool_data in data.get('tools', {}).items():
            tool_type = tool_data.get('type', 'base')
            
            if tool_type in self.config_types:
                config_class = self.config_types[tool_type]
                config = config_class(**tool_data)
                self.configs[tool_name] = config
    
    def save_to_file(self, file_path: str, format: str = 'yaml'):
        """Save tool configurations to file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        data = {
            'tools': {}
        }
        
        for name, config in self.configs.items():
            # Determine tool type
            tool_type = 'base'
            for type_name, type_class in self.config_types.items():
                if isinstance(config, type_class) and type_name != 'base':
                    tool_type = type_name
                    break
            
            config_data = config.dict()
            config_data['type'] = tool_type
            data['tools'][name] = config_data
        
        # Save to file
        with open(path, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def create_web_search_config(self, name: str, **kwargs) -> WebSearchConfig:
        """Create a web search tool configuration."""
        config = WebSearchConfig(name=name, **kwargs)
        self.configs[name] = config
        return config
    
    def create_database_config(self, name: str, **kwargs) -> DatabaseConfig:
        """Create a database tool configuration."""
        config = DatabaseConfig(name=name, **kwargs)
        self.configs[name] = config
        return config
    
    def create_file_operations_config(self, name: str, **kwargs) -> FileOperationsConfig:
        """Create a file operations tool configuration."""
        config = FileOperationsConfig(name=name, **kwargs)
        self.configs[name] = config
        return config
    
    def create_api_client_config(self, name: str, **kwargs) -> APIClientConfig:
        """Create an API client tool configuration."""
        config = APIClientConfig(name=name, **kwargs)
        self.configs[name] = config
        return config
    
    def get_configs_by_type(self, tool_type: str) -> List[ToolConfig]:
        """Get all configurations of a specific type."""
        if tool_type not in self.config_types:
            return []
        
        config_class = self.config_types[tool_type]
        return [config for config in self.configs.values() if isinstance(config, config_class)]
    
    def validate_all_configs(self) -> Dict[str, List[str]]:
        """Validate all configurations and return any errors."""
        errors = {}
        
        for name, config in self.configs.items():
            config_errors = []
            
            try:
                # Basic validation through Pydantic
                config.dict()
            except Exception as e:
                config_errors.append(f"Validation error: {str(e)}")
            
            # Tool-specific validation
            if hasattr(config, 'validate_config'):
                try:
                    config.validate_config()
                except Exception as e:
                    config_errors.append(f"Tool validation error: {str(e)}")
            
            if config_errors:
                errors[name] = config_errors
        
        return errors


# Global tool config manager
_global_tool_manager: Optional[ToolConfigManager] = None


def get_global_tool_manager() -> ToolConfigManager:
    """Get the global tool configuration manager."""
    global _global_tool_manager
    if _global_tool_manager is None:
        _global_tool_manager = ToolConfigManager()
        _setup_default_tool_configs(_global_tool_manager)
    
    return _global_tool_manager


def _setup_default_tool_configs(manager: ToolConfigManager):
    """Setup default tool configurations."""
    import os
    
    # Web search configuration
    google_api_key = os.getenv('GOOGLE_API_KEY')
    google_search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    bing_api_key = os.getenv('BING_API_KEY')
    
    if google_api_key and google_search_engine_id:
        manager.create_web_search_config(
            'google_search',
            description='Google Custom Search',
            search_engine='google',
            api_key=google_api_key,
            search_engine_id=google_search_engine_id
        )
    
    if bing_api_key:
        manager.create_web_search_config(
            'bing_search',
            description='Bing Web Search',
            search_engine='bing',
            api_key=bing_api_key
        )
    
    # DuckDuckGo (no API key required)
    manager.create_web_search_config(
        'duckduckgo_search',
        description='DuckDuckGo Search',
        search_engine='duckduckgo'
    )
    
    # File operations configuration
    manager.create_file_operations_config(
        'safe_file_ops',
        description='Safe file operations',
        allowed_directories=['/tmp', '/var/tmp'],
        max_file_size=10 * 1024 * 1024,  # 10MB
        create_directories=True,
        overwrite_files=False
    )
    
    # API client configuration
    manager.create_api_client_config(
        'general_api_client',
        description='General purpose API client',
        timeout_seconds=30,
        max_retries=3,
        verify_ssl=True
    )
    
    # Database configurations (examples)
    if os.getenv('DATABASE_URL'):
        # Parse database URL and create config
        # This is a simplified example
        manager.create_database_config(
            'default_database',
            description='Default database connection',
            database_type='postgresql',
            host='localhost',
            port=5432,
            database='agentix',
            username=os.getenv('DB_USER', 'agentix'),
            password=os.getenv('DB_PASSWORD', '')
        )


def create_tool_config_template(output_path: str, format: str = 'yaml'):
    """Create a tool configuration template file."""
    manager = ToolConfigManager()
    
    # Add example configurations
    manager.create_web_search_config(
        'example_google_search',
        description='Example Google search configuration',
        search_engine='google',
        api_key='your_google_api_key',
        search_engine_id='your_search_engine_id',
        max_results=10
    )
    
    manager.create_database_config(
        'example_database',
        description='Example database configuration',
        database_type='postgresql',
        host='localhost',
        port=5432,
        database='your_database',
        username='your_username',
        password='your_password'
    )
    
    manager.create_file_operations_config(
        'example_file_ops',
        description='Example file operations configuration',
        allowed_directories=['/safe/directory'],
        max_file_size=10 * 1024 * 1024
    )
    
    manager.create_api_client_config(
        'example_api_client',
        description='Example API client configuration',
        base_url='https://api.example.com',
        auth_type='bearer',
        auth_token='your_api_token'
    )
    
    manager.save_to_file(output_path, format)
