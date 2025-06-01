"""
Logging configuration and utilities for Agentix framework.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class AgentixFormatter(logging.Formatter):
    """Custom formatter for Agentix logs."""
    
    def __init__(self, include_context: bool = True):
        self.include_context = include_context
        super().__init__()
    
    def format(self, record):
        # Base format
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record.levelname
        name = record.name
        message = record.getMessage()
        
        # Build log line
        log_line = f"{timestamp} | {level:8} | {name:30} | {message}"
        
        # Add context if available and enabled
        if self.include_context and hasattr(record, 'context'):
            context_str = " | ".join(f"{k}={v}" for k, v in record.context.items())
            log_line += f" | {context_str}"
        
        # Add exception info if present
        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)
        
        return log_line


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record):
        # Add context to record
        if not hasattr(record, 'context'):
            record.context = {}
        
        record.context.update(self.context)
        return True


def setup_logging(name: Optional[str] = None, 
                 level: str = "INFO",
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None,
                 include_context: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """
    Setup logging configuration for Agentix.
    
    Args:
        name: Logger name (defaults to 'agentix')
        level: Logging level
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)
        include_context: Whether to include context in logs
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    
    logger_name = name or "agentix"
    logger = logging.getLogger(logger_name)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create formatter
    if log_format:
        formatter = logging.Formatter(log_format)
    else:
        formatter = AgentixFormatter(include_context=include_context)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name
        context: Context to add to all log messages
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if context:
        # Add context filter
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)
    
    return logger


class LogContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.filter = None
    
    def __enter__(self):
        self.filter = ContextFilter(self.context)
        self.logger.addFilter(self.filter)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.filter:
            self.logger.removeFilter(self.filter)


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_execution_time(self, operation: str, execution_time: float, 
                          context: Optional[Dict[str, Any]] = None):
        """Log execution time for an operation."""
        log_context = context or {}
        log_context.update({
            'operation': operation,
            'execution_time': execution_time,
            'metric_type': 'execution_time'
        })
        
        with LogContext(self.logger, **log_context):
            self.logger.info(f"Operation '{operation}' completed in {execution_time:.3f}s")
    
    def log_memory_usage(self, operation: str, memory_mb: float,
                        context: Optional[Dict[str, Any]] = None):
        """Log memory usage for an operation."""
        log_context = context or {}
        log_context.update({
            'operation': operation,
            'memory_mb': memory_mb,
            'metric_type': 'memory_usage'
        })
        
        with LogContext(self.logger, **log_context):
            self.logger.info(f"Operation '{operation}' used {memory_mb:.2f}MB memory")
    
    def log_api_call(self, provider: str, model: str, tokens_used: int,
                    response_time: float, context: Optional[Dict[str, Any]] = None):
        """Log API call metrics."""
        log_context = context or {}
        log_context.update({
            'provider': provider,
            'model': model,
            'tokens_used': tokens_used,
            'response_time': response_time,
            'metric_type': 'api_call'
        })
        
        with LogContext(self.logger, **log_context):
            self.logger.info(f"API call to {provider}/{model}: {tokens_used} tokens, {response_time:.3f}s")


class StructuredLogger:
    """Logger for structured data."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_agent_execution(self, agent_name: str, execution_id: str,
                           status: str, duration: float, 
                           nodes_executed: int, context: Optional[Dict[str, Any]] = None):
        """Log agent execution summary."""
        log_data = {
            'event_type': 'agent_execution',
            'agent_name': agent_name,
            'execution_id': execution_id,
            'status': status,
            'duration': duration,
            'nodes_executed': nodes_executed,
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            log_data.update(context)
        
        with LogContext(self.logger, **log_data):
            self.logger.info(f"Agent '{agent_name}' execution {status}: {duration:.3f}s, {nodes_executed} nodes")
    
    def log_tool_execution(self, tool_name: str, operation: str,
                          status: str, duration: float,
                          context: Optional[Dict[str, Any]] = None):
        """Log tool execution."""
        log_data = {
            'event_type': 'tool_execution',
            'tool_name': tool_name,
            'operation': operation,
            'status': status,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            log_data.update(context)
        
        with LogContext(self.logger, **log_data):
            self.logger.info(f"Tool '{tool_name}' {operation} {status}: {duration:.3f}s")
    
    def log_memory_operation(self, memory_type: str, operation: str,
                           status: str, duration: float,
                           context: Optional[Dict[str, Any]] = None):
        """Log memory operation."""
        log_data = {
            'event_type': 'memory_operation',
            'memory_type': memory_type,
            'operation': operation,
            'status': status,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            log_data.update(context)
        
        with LogContext(self.logger, **log_data):
            self.logger.info(f"Memory {memory_type} {operation} {status}: {duration:.3f}s")


def configure_third_party_logging():
    """Configure logging for third-party libraries."""
    
    # Reduce verbosity of common third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # Set specific levels for database libraries
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    logging.getLogger('alembic').setLevel(logging.WARNING)


def setup_development_logging():
    """Setup logging configuration for development."""
    return setup_logging(
        level="DEBUG",
        include_context=True,
        log_file="logs/agentix_dev.log"
    )


def setup_production_logging():
    """Setup logging configuration for production."""
    return setup_logging(
        level="INFO",
        include_context=False,
        log_file="logs/agentix_prod.log",
        max_file_size=50 * 1024 * 1024,  # 50MB
        backup_count=10
    )


# Global logger instances
_main_logger: Optional[logging.Logger] = None
_performance_logger: Optional[PerformanceLogger] = None
_structured_logger: Optional[StructuredLogger] = None


def get_main_logger() -> logging.Logger:
    """Get the main Agentix logger."""
    global _main_logger
    if _main_logger is None:
        _main_logger = setup_logging()
    return _main_logger


def get_performance_logger() -> PerformanceLogger:
    """Get the performance logger."""
    global _performance_logger
    if _performance_logger is None:
        main_logger = get_main_logger()
        _performance_logger = PerformanceLogger(main_logger)
    return _performance_logger


def get_structured_logger() -> StructuredLogger:
    """Get the structured logger."""
    global _structured_logger
    if _structured_logger is None:
        main_logger = get_main_logger()
        _structured_logger = StructuredLogger(main_logger)
    return _structured_logger
