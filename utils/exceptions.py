"""
Custom exceptions for Agentix framework.
"""

from typing import Optional, Dict, Any


class AgentixError(Exception):
    """Base exception for all Agentix-related errors."""

    def __init__(self, message: str, error_code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context
        }


class NodeExecutionError(AgentixError):
    """Exception raised when a node execution fails."""

    def __init__(self, message: str, node_id: Optional[str] = None,
                 node_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.node_id = node_id
        self.node_type = node_type

        if node_id:
            self.context['node_id'] = node_id
        if node_type:
            self.context['node_type'] = node_type


class ValidationError(AgentixError):
    """Exception raised when validation fails."""

    def __init__(self, message: str, field: Optional[str] = None,
                 value: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value

        if field:
            self.context['field'] = field
        if value is not None:
            self.context['value'] = str(value)


class ToolExecutionError(AgentixError):
    """Exception raised when a tool execution fails."""

    def __init__(self, message: str, tool_name: Optional[str] = None,
                 tool_operation: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.tool_operation = tool_operation

        if tool_name:
            self.context['tool_name'] = tool_name
        if tool_operation:
            self.context['tool_operation'] = tool_operation


class MemoryError(AgentixError):
    """Exception raised when memory operations fail."""

    def __init__(self, message: str, memory_type: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.memory_type = memory_type
        self.operation = operation

        if memory_type:
            self.context['memory_type'] = memory_type
        if operation:
            self.context['operation'] = operation


class LLMError(AgentixError):
    """Exception raised when LLM operations fail."""

    def __init__(self, message: str, provider: Optional[str] = None,
                 model: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.model = model

        if provider:
            self.context['provider'] = provider
        if model:
            self.context['model'] = model


class RateLimitError(AgentixError):
    """Exception raised when rate limits are exceeded."""

    def __init__(self, message: str, limit: Optional[int] = None,
                 window: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.limit = limit
        self.window = window

        if limit:
            self.context['limit'] = limit
        if window:
            self.context['window'] = window


class AuthenticationError(AgentixError):
    """Exception raised when authentication fails."""

    def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.provider = provider

        if provider:
            self.context['provider'] = provider


class ConfigurationError(AgentixError):
    """Exception raised when configuration is invalid."""

    def __init__(self, message: str, config_key: Optional[str] = None,
                 config_value: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_value = config_value

        if config_key:
            self.context['config_key'] = config_key
        if config_value is not None:
            self.context['config_value'] = str(config_value)


class GraphExecutionError(AgentixError):
    """Exception raised when graph execution fails."""

    def __init__(self, message: str, graph_id: Optional[str] = None,
                 current_node: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.graph_id = graph_id
        self.current_node = current_node

        if graph_id:
            self.context['graph_id'] = graph_id
        if current_node:
            self.context['current_node'] = current_node


class AuthenticationError(AgentixError):
    """Exception raised when authentication fails."""

    def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.provider = provider

        if provider:
            self.context['provider'] = provider


class RateLimitError(AgentixError):
    """Exception raised when rate limits are exceeded."""

    def __init__(self, message: str, limit: Optional[int] = None,
                 window: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.limit = limit
        self.window = window

        if limit:
            self.context['limit'] = limit
        if window:
            self.context['window'] = window


class SecurityError(AgentixError):
    """Exception raised when security validation fails."""

    def __init__(self, message: str, security_check: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.security_check = security_check

        if security_check:
            self.context['security_check'] = security_check


class TimeoutError(AgentixError):
    """Exception raised when operations timeout."""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds

        if timeout_seconds:
            self.context['timeout_seconds'] = timeout_seconds


class ResourceError(AgentixError):
    """Exception raised when resource limits are exceeded."""

    def __init__(self, message: str, resource_type: Optional[str] = None,
                 limit: Optional[Any] = None, current: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.limit = limit
        self.current = current

        if resource_type:
            self.context['resource_type'] = resource_type
        if limit is not None:
            self.context['limit'] = str(limit)
        if current is not None:
            self.context['current'] = str(current)


def handle_exception(func):
    """Decorator to handle and log exceptions."""
    import functools
    import logging

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AgentixError:
            # Re-raise Agentix errors as-is
            raise
        except Exception as e:
            # Convert other exceptions to AgentixError
            logger = logging.getLogger(func.__module__)
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise AgentixError(f"Unexpected error in {func.__name__}: {str(e)}") from e

    return wrapper


def handle_async_exception(func):
    """Decorator to handle and log exceptions in async functions."""
    import functools
    import logging

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AgentixError:
            # Re-raise Agentix errors as-is
            raise
        except Exception as e:
            # Convert other exceptions to AgentixError
            logger = logging.getLogger(func.__module__)
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise AgentixError(f"Unexpected error in {func.__name__}: {str(e)}") from e

    return wrapper


class ErrorHandler:
    """Centralized error handling and reporting."""

    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.max_history = 1000

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Handle and record an error."""
        import logging
        from datetime import datetime

        logger = logging.getLogger("agentix.error_handler")

        # Record error
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context or {}
        }

        if isinstance(error, AgentixError):
            error_info.update(error.to_dict())

        # Add to history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

        # Update counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Log error
        if isinstance(error, (SecurityError, AuthenticationError)):
            logger.error(f"Security error: {error}")
        elif isinstance(error, (ValidationError, ConfigurationError)):
            logger.warning(f"Validation error: {error}")
        else:
            logger.error(f"Error: {error}")

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }

    def clear_history(self):
        """Clear error history."""
        self.error_history.clear()
        self.error_counts.clear()


# Global error handler
_global_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler."""
    return _global_error_handler
