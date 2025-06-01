"""
Base tool system for Agentix agents.

This module provides the foundation for all tools used by agents,
including registration, validation, and execution frameworks.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class ToolStatus(str, Enum):
    """Tool execution status."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ToolConfig(BaseModel):
    """Base configuration for tools."""
    
    name: str
    description: str
    version: str = "1.0.0"
    
    # Execution settings
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Validation settings
    validate_input: bool = True
    validate_output: bool = True
    
    # Security settings
    require_approval: bool = False
    allowed_domains: List[str] = Field(default_factory=list)
    
    # Tool-specific configuration
    config: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result from tool execution."""
    
    status: ToolStatus
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Tool information
    tool_name: str
    tool_version: str
    timestamp: datetime = Field(default_factory=datetime.now)


class BaseTool(ABC):
    """
    Abstract base class for all Agentix tools.
    
    Tools enable agents to perform external actions such as:
    - Web searches and API calls
    - Database operations
    - File system interactions
    - Communication with external services
    """
    
    def __init__(self, config: ToolConfig):
        self.config = config
        self.name = config.name
        self.description = config.description
        self.version = config.version
        
        self.logger = logging.getLogger(f"agentix.tools.{self.name}")
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        self.logger.info(f"Tool '{self.name}' initialized")
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with the given parameters.
        
        Args:
            parameters: Tool-specific parameters
            
        Returns:
            ToolResult containing the execution result
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate the parameters for tool execution.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        pass
    
    async def run(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Run the tool with validation, error handling, and retries.
        
        Args:
            parameters: Tool parameters
            
        Returns:
            ToolResult with execution outcome
        """
        start_time = datetime.now()
        self.execution_count += 1
        
        try:
            # Validate input parameters
            if self.config.validate_input:
                if not self.validate_parameters(parameters):
                    self.failure_count += 1
                    return ToolResult(
                        status=ToolStatus.FAILURE,
                        error="Parameter validation failed",
                        tool_name=self.name,
                        tool_version=self.version,
                        execution_time=0.0
                    )
            
            # Execute with retries
            last_error = None
            for attempt in range(self.config.retry_attempts):
                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        self.execute(parameters),
                        timeout=self.config.timeout_seconds
                    )
                    
                    # Validate output if configured
                    if self.config.validate_output:
                        if not self.validate_result(result):
                            raise ValueError("Output validation failed")
                    
                    # Success
                    execution_time = (datetime.now() - start_time).total_seconds()
                    result.execution_time = execution_time
                    result.tool_name = self.name
                    result.tool_version = self.version
                    
                    self.success_count += 1
                    self.logger.debug(f"Tool '{self.name}' executed successfully")
                    
                    return result
                    
                except asyncio.TimeoutError:
                    last_error = "Tool execution timed out"
                    self.logger.warning(f"Tool '{self.name}' timed out on attempt {attempt + 1}")
                    
                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"Tool '{self.name}' failed on attempt {attempt + 1}: {str(e)}")
                
                # Wait before retry (except on last attempt)
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)
            
            # All attempts failed
            execution_time = (datetime.now() - start_time).total_seconds()
            self.failure_count += 1
            
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=f"Tool failed after {self.config.retry_attempts} attempts: {last_error}",
                tool_name=self.name,
                tool_version=self.version,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.failure_count += 1
            self.logger.error(f"Unexpected error in tool '{self.name}': {str(e)}")
            
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=f"Unexpected error: {str(e)}",
                tool_name=self.name,
                tool_version=self.version,
                execution_time=execution_time
            )
    
    def validate_result(self, result: ToolResult) -> bool:
        """
        Validate the tool execution result.
        
        Args:
            result: Result to validate
            
        Returns:
            True if result is valid, False otherwise
        """
        # Basic validation - can be overridden by specific tools
        return result.status in [ToolStatus.SUCCESS, ToolStatus.FAILURE]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics."""
        success_rate = self.success_count / self.execution_count if self.execution_count > 0 else 0
        
        return {
            'name': self.name,
            'version': self.version,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': success_rate
        }


class ToolRegistry:
    """
    Registry for managing and discovering tools.
    
    The registry provides:
    - Tool registration and discovery
    - Tool lifecycle management
    - Tool execution coordination
    - Tool statistics and monitoring
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_configs: Dict[str, ToolConfig] = {}
        self.logger = logging.getLogger("agentix.tools.registry")
        
        self.logger.info("Tool registry initialized")
    
    def register_tool(self, tool: BaseTool):
        """Register a tool in the registry."""
        if tool.name in self.tools:
            self.logger.warning(f"Tool '{tool.name}' already registered, replacing")
        
        self.tools[tool.name] = tool
        self.tool_configs[tool.name] = tool.config
        
        self.logger.info(f"Registered tool: {tool.name} v{tool.version}")
    
    def unregister_tool(self, tool_name: str):
        """Unregister a tool from the registry."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            del self.tool_configs[tool_name]
            self.logger.info(f"Unregistered tool: {tool_name}")
        else:
            self.logger.warning(f"Tool '{tool_name}' not found for unregistration")
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool."""
        tool = self.tools.get(tool_name)
        if not tool:
            return None
        
        return {
            'name': tool.name,
            'description': tool.description,
            'version': tool.version,
            'config': tool.config.dict(),
            'stats': tool.get_stats()
        }
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name."""
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=f"Tool '{tool_name}' not found",
                tool_name=tool_name,
                tool_version="unknown"
            )
        
        return await tool.run(parameters)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get overall registry statistics."""
        total_executions = sum(tool.execution_count for tool in self.tools.values())
        total_successes = sum(tool.success_count for tool in self.tools.values())
        total_failures = sum(tool.failure_count for tool in self.tools.values())
        
        overall_success_rate = total_successes / total_executions if total_executions > 0 else 0
        
        return {
            'total_tools': len(self.tools),
            'total_executions': total_executions,
            'total_successes': total_successes,
            'total_failures': total_failures,
            'overall_success_rate': overall_success_rate,
            'tools': {name: tool.get_stats() for name, tool in self.tools.items()}
        }


# Global tool registry instance
global_tool_registry = ToolRegistry()


def register_tool(tool_class: Type[BaseTool], config: ToolConfig):
    """Convenience function to register a tool."""
    tool_instance = tool_class(config)
    global_tool_registry.register_tool(tool_instance)
    return tool_instance


def get_tool(tool_name: str) -> Optional[BaseTool]:
    """Convenience function to get a tool."""
    return global_tool_registry.get_tool(tool_name)


async def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """Convenience function to execute a tool."""
    return await global_tool_registry.execute_tool(tool_name, parameters)
