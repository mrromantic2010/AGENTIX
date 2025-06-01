"""
Performance monitoring and metrics collection for Agentix framework.
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .logging import get_performance_logger


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation."""
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    memory_start: Optional[float] = None
    memory_end: Optional[float] = None
    memory_peak: Optional[float] = None
    cpu_usage: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor performance of operations and system resources."""
    
    def __init__(self, enable_system_monitoring: bool = True):
        self.enable_system_monitoring = enable_system_monitoring
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.completed_operations: deque = deque(maxlen=1000)
        
        # System monitoring
        self.system_metrics: Dict[str, deque] = {
            'cpu_percent': deque(maxlen=100),
            'memory_percent': deque(maxlen=100),
            'memory_used_mb': deque(maxlen=100),
            'disk_usage_percent': deque(maxlen=100)
        }
        
        self.logger = get_performance_logger()
        
        # Start system monitoring thread
        if self.enable_system_monitoring:
            self._start_system_monitoring()
    
    def start_operation(self, operation_id: str, operation_name: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start monitoring an operation."""
        
        metrics = PerformanceMetrics(
            operation=operation_name,
            start_time=datetime.now(),
            memory_start=self._get_memory_usage(),
            metadata=metadata or {}
        )
        
        self.active_operations[operation_id] = metrics
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     error: Optional[str] = None):
        """End monitoring an operation."""
        
        if operation_id not in self.active_operations:
            return
        
        metrics = self.active_operations[operation_id]
        metrics.end_time = datetime.now()
        metrics.duration = (metrics.end_time - metrics.start_time).total_seconds()
        metrics.memory_end = self._get_memory_usage()
        metrics.success = success
        metrics.error = error
        
        # Calculate memory usage
        if metrics.memory_start and metrics.memory_end:
            metrics.memory_peak = max(metrics.memory_start, metrics.memory_end)
        
        # Log performance
        self.logger.log_execution_time(
            metrics.operation,
            metrics.duration,
            {
                'success': success,
                'memory_start': metrics.memory_start,
                'memory_end': metrics.memory_end,
                **metrics.metadata
            }
        )
        
        # Move to completed operations
        self.completed_operations.append(metrics)
        del self.active_operations[operation_id]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def _start_system_monitoring(self):
        """Start background system monitoring."""
        
        def monitor_system():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.system_metrics['cpu_percent'].append(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.system_metrics['memory_percent'].append(memory.percent)
                    self.system_metrics['memory_used_mb'].append(memory.used / 1024 / 1024)
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    self.system_metrics['disk_usage_percent'].append(disk.percent)
                    
                    time.sleep(5)  # Monitor every 5 seconds
                    
                except Exception:
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for operations."""
        
        # Filter operations
        if operation_name:
            operations = [op for op in self.completed_operations if op.operation == operation_name]
        else:
            operations = list(self.completed_operations)
        
        if not operations:
            return {}
        
        # Calculate statistics
        durations = [op.duration for op in operations if op.duration]
        success_count = sum(1 for op in operations if op.success)
        
        stats = {
            'total_operations': len(operations),
            'successful_operations': success_count,
            'failed_operations': len(operations) - success_count,
            'success_rate': success_count / len(operations) if operations else 0,
        }
        
        if durations:
            stats.update({
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_duration': sum(durations)
            })
        
        return stats
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        
        stats = {}
        
        for metric_name, values in self.system_metrics.items():
            if values:
                stats[metric_name] = {
                    'current': values[-1],
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        return stats
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get currently active operations."""
        
        active = []
        current_time = datetime.now()
        
        for op_id, metrics in self.active_operations.items():
            duration = (current_time - metrics.start_time).total_seconds()
            
            active.append({
                'operation_id': op_id,
                'operation': metrics.operation,
                'duration': duration,
                'start_time': metrics.start_time.isoformat(),
                'metadata': metrics.metadata
            })
        
        return active


class MetricsCollector:
    """Collect and aggregate metrics over time."""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.aggregations: Dict[str, Dict[str, float]] = {}
        self.last_aggregation = datetime.now()
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        
        self.metrics[name].append(point)
    
    def record_counter(self, name: str, increment: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        self.record_metric(f"{name}_count", increment, tags)
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        self.record_metric(f"{name}_gauge", value, tags)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        self.record_metric(f"{name}_histogram", value, tags)
    
    def get_metric_values(self, name: str, since: Optional[datetime] = None) -> List[MetricPoint]:
        """Get metric values, optionally filtered by time."""
        
        if name not in self.metrics:
            return []
        
        points = list(self.metrics[name])
        
        if since:
            points = [p for p in points if p.timestamp >= since]
        
        return points
    
    def aggregate_metrics(self, window_minutes: int = 5) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics over a time window."""
        
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=window_minutes)
        
        aggregations = {}
        
        for metric_name, points in self.metrics.items():
            # Filter points in window
            window_points = [p for p in points if p.timestamp >= window_start]
            
            if not window_points:
                continue
            
            values = [p.value for p in window_points]
            
            aggregations[metric_name] = {
                'count': len(values),
                'sum': sum(values),
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1] if values else 0
            }
        
        self.aggregations = aggregations
        self.last_aggregation = current_time
        
        return aggregations
    
    def get_top_metrics(self, metric_type: str = "avg", limit: int = 10) -> List[tuple]:
        """Get top metrics by a specific aggregation type."""
        
        if not self.aggregations:
            self.aggregate_metrics()
        
        metric_values = []
        
        for name, agg in self.aggregations.items():
            if metric_type in agg:
                metric_values.append((name, agg[metric_type]))
        
        # Sort by value (descending)
        metric_values.sort(key=lambda x: x[1], reverse=True)
        
        return metric_values[:limit]
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        
        if format == "json":
            import json
            
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'aggregations': self.aggregations,
                'metric_counts': {name: len(points) for name, points in self.metrics.items()}
            }
            
            return json.dumps(export_data, indent=2)
        
        elif format == "prometheus":
            # Basic Prometheus format
            lines = []
            
            for metric_name, agg in self.aggregations.items():
                for agg_type, value in agg.items():
                    lines.append(f"{metric_name}_{agg_type} {value}")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.monitor = monitor
        self.operation_name = operation_name
        self.metadata = metadata
        self.operation_id = None
    
    def __enter__(self):
        import uuid
        self.operation_id = str(uuid.uuid4())
        self.monitor.start_operation(self.operation_id, self.operation_name, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        self.monitor.end_operation(self.operation_id, success, error)


# Global instances
_global_monitor: Optional[PerformanceMonitor] = None
_global_metrics: Optional[MetricsCollector] = None


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def get_global_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def time_operation(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator to time function execution."""
    
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            monitor = get_global_monitor()
            with OperationTimer(monitor, operation_name, metadata):
                return func(*args, **kwargs)
        return wrapper
    
    return decorator
