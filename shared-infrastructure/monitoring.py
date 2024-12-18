"""
Production monitoring and observability for GenAI applications
Comprehensive logging, metrics, alerting, and performance tracking
"""

import os
import json
import time
import logging
import threading
import asyncio
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from functools import wraps
from contextlib import contextmanager
from collections import defaultdict, deque
import re
import hashlib

import psutil
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from google.cloud import monitoring_v3, logging as gcp_logging
from google.cloud.logging_v2 import Client as LoggingClient

from .config import get_config

config = get_config()

# Security patterns for log sanitization
SENSITIVE_PATTERNS = [
    r'password["\s]*[:=]["\s]*[^,}\s]+',
    r'token["\s]*[:=]["\s]*[^,}\s]+',
    r'key["\s]*[:=]["\s]*[^,}\s]+',
    r'secret["\s]*[:=]["\s]*[^,}\s]+',
    r'api_key["\s]*[:=]["\s]*[^,}\s]+',
    r'bearer\s+[a-zA-Z0-9\-._~+/]+=*',
    r'sk-[a-zA-Z0-9]{20,}',
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
    r'\b\d{3}-\d{2}-\d{4}\b'  # SSN
]

@dataclass
class LogEvent:
    """Structured log event with security and context"""
    timestamp: datetime
    level: str
    message: str
    service: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    duration_ms: Optional[float] = None
    status_code: Optional[int] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "service": self.service,
            "operation": self.operation,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "duration_ms": self.duration_ms,
            "status_code": self.status_code,
            "error_type": self.error_type,
            "metadata": self.metadata
        }

@dataclass
class MetricEvent:
    """Performance metric event"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = "count"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str
    threshold: float
    window_minutes: int
    cooldown_minutes: int
    severity: str = "warning"
    enabled: bool = True

class LogSanitizer:
    """Secure log sanitization to prevent data leaks"""
    
    def __init__(self):
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SENSITIVE_PATTERNS]
        self.replacement = "[REDACTED]"
    
    def sanitize(self, text: str) -> str:
        """Sanitize sensitive information from logs"""
        if not isinstance(text, str):
            text = str(text)
        
        for pattern in self.patterns:
            text = pattern.sub(self.replacement, text)
        
        return text
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary data recursively"""
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize keys
            clean_key = self.sanitize(key)
            
            # Sanitize values
            if isinstance(value, dict):
                sanitized[clean_key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[clean_key] = [
                    self.sanitize_dict(item) if isinstance(item, dict) else self.sanitize(str(item))
                    for item in value
                ]
            else:
                sanitized[clean_key] = self.sanitize(str(value))
        
        return sanitized

class StructuredLogger:
    """Thread-safe structured logging with security"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.sanitizer = LogSanitizer()
        self.logger = logging.getLogger(f"genai.{service_name}")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Setup formatters
        self._setup_formatters()
        
        # Thread-local storage for context
        self._local = threading.local()
        
        # GCP logging client
        self.gcp_client = None
        if config.gcp_project_id:
            try:
                self.gcp_client = LoggingClient(project=config.gcp_project_id)
            except Exception as e:
                self.logger.warning(f"Failed to initialize GCP logging: {e}")
    
    def _setup_formatters(self):
        """Setup log formatters"""
        # JSON formatter for structured logs
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "service": "%(name)s", "message": "%(message)s"}'
        )
        
        # Console formatter for development
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter if config.debug else json_formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler if configured
        if hasattr(config, 'log_file') and config.log_file:
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setFormatter(json_formatter)
            self.logger.addHandler(file_handler)
    
    def set_context(self, **context):
        """Set logging context for current thread"""
        self._local.context = context
    
    def get_context(self) -> Dict[str, Any]:
        """Get current logging context"""
        return getattr(self._local, 'context', {})
    
    def clear_context(self):
        """Clear logging context"""
        self._local.context = {}
    
    def log_event(self, event: LogEvent):
        """Log structured event"""
        try:
            # Sanitize sensitive data
            sanitized_metadata = self.sanitizer.sanitize_dict(event.metadata)
            sanitized_message = self.sanitizer.sanitize(event.message)
            
            # Merge with thread context
            context = self.get_context()
            merged_metadata = {**context, **sanitized_metadata}
            
            # Create log record
            log_data = {
                "timestamp": event.timestamp.isoformat(),
                "service": event.service,
                "operation": event.operation,
                "user_id": event.user_id,
                "session_id": event.session_id,
                "request_id": event.request_id,
                "duration_ms": event.duration_ms,
                "status_code": event.status_code,
                "error_type": event.error_type,
                "metadata": merged_metadata
            }
            
            # Log to local logger
            log_level = getattr(logging, event.level.upper())
            self.logger.log(log_level, sanitized_message, extra=log_data)
            
            # Send to GCP if available
            if self.gcp_client and config.environment == "production":
                self._send_to_gcp(event, log_data)
        
        except Exception as e:
            # Fallback logging to prevent log failures from breaking the application
            self.logger.error(f"Failed to log event: {e}", exc_info=True)
    
    def _send_to_gcp(self, event: LogEvent, log_data: Dict[str, Any]):
        """Send log to Google Cloud Logging"""
        try:
            severity = event.level.upper()
            if severity == "WARNING":
                severity = "WARN"
            elif severity == "CRITICAL":
                severity = "FATAL"
            
            self.gcp_client.logger(self.service_name).log_struct(
                log_data,
                severity=severity,
                trace=event.request_id,
                span_id=event.session_id
            )
        except Exception as e:
            self.logger.error(f"Failed to send log to GCP: {e}")
    
    def info(self, message: str, operation: str = "", **kwargs):
        """Log info message"""
        self._log_simple("INFO", message, operation, **kwargs)
    
    def warning(self, message: str, operation: str = "", **kwargs):
        """Log warning message"""
        self._log_simple("WARNING", message, operation, **kwargs)
    
    def error(self, message: str, operation: str = "", error: Exception = None, **kwargs):
        """Log error message"""
        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)
            kwargs["traceback"] = traceback.format_exc()
        
        self._log_simple("ERROR", message, operation, **kwargs)
    
    def critical(self, message: str, operation: str = "", **kwargs):
        """Log critical message"""
        self._log_simple("CRITICAL", message, operation, **kwargs)
    
    def _log_simple(self, level: str, message: str, operation: str, **kwargs):
        """Helper for simple logging"""
        event = LogEvent(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            service=self.service_name,
            operation=operation,
            metadata=kwargs
        )
        self.log_event(event)

class MetricsCollector:
    """Thread-safe metrics collection with Prometheus integration"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.registry = CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        # Standard metrics
        self._init_standard_metrics()
        
        # Custom metrics storage
        self._custom_metrics = defaultdict(list)
        
        # Alert rules
        self.alert_rules: List[AlertRule] = []
    
    def _init_standard_metrics(self):
        """Initialize standard metrics"""
        with self._lock:
            # Request metrics
            self._metrics["request_count"] = Counter(
                'genai_requests_total',
                'Total requests',
                ['service', 'operation', 'status'],
                registry=self.registry
            )
            
            self._metrics["request_duration"] = Histogram(
                'genai_request_duration_seconds',
                'Request duration',
                ['service', 'operation'],
                registry=self.registry
            )
            
            # Model metrics
            self._metrics["model_load_count"] = Counter(
                'genai_model_loads_total',
                'Total model loads',
                ['service', 'model_name'],
                registry=self.registry
            )
            
            self._metrics["model_inference_count"] = Counter(
                'genai_model_inferences_total',
                'Total model inferences',
                ['service', 'model_name'],
                registry=self.registry
            )
            
            self._metrics["model_inference_duration"] = Histogram(
                'genai_model_inference_duration_seconds',
                'Model inference duration',
                ['service', 'model_name'],
                registry=self.registry
            )
            
            # Resource metrics
            self._metrics["memory_usage"] = Gauge(
                'genai_memory_usage_bytes',
                'Memory usage',
                ['service'],
                registry=self.registry
            )
            
            self._metrics["cpu_usage"] = Gauge(
                'genai_cpu_usage_percent',
                'CPU usage percentage',
                ['service'],
                registry=self.registry
            )
            
            # Error metrics
            self._metrics["error_count"] = Counter(
                'genai_errors_total',
                'Total errors',
                ['service', 'error_type'],
                registry=self.registry
            )
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0):
        """Increment counter metric"""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if labels:
                    metric.labels(**labels).inc(value)
                else:
                    metric.inc(value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge metric value"""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if labels:
                    metric.labels(**labels).set(value)
                else:
                    metric.set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe histogram metric"""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)
    
    def record_request(self, operation: str, duration: float, status_code: int):
        """Record request metrics"""
        status = "success" if 200 <= status_code < 400 else "error"
        
        self.increment_counter("request_count", {
            "service": self.service_name,
            "operation": operation,
            "status": status
        })
        
        self.observe_histogram("request_duration", duration, {
            "service": self.service_name,
            "operation": operation
        })
    
    def record_model_load(self, model_name: str, duration: float):
        """Record model loading metrics"""
        self.increment_counter("model_load_count", {
            "service": self.service_name,
            "model_name": model_name
        })
    
    def record_model_inference(self, model_name: str, duration: float):
        """Record model inference metrics"""
        self.increment_counter("model_inference_count", {
            "service": self.service_name,
            "model_name": model_name
        })
        
        self.observe_histogram("model_inference_duration", duration, {
            "service": self.service_name,
            "model_name": model_name
        })
    
    def record_error(self, error_type: str):
        """Record error metrics"""
        self.increment_counter("error_count", {
            "service": self.service_name,
            "error_type": error_type
        })
    
    def update_resource_metrics(self):
        """Update resource usage metrics"""
        try:
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            self.set_gauge("memory_usage", memory_info.rss, {
                "service": self.service_name
            })
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            self.set_gauge("cpu_usage", cpu_percent, {
                "service": self.service_name
            })
            
        except Exception as e:
            # Don't let metrics collection break the application
            pass
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        with self._lock:
            return generate_latest(self.registry).decode('utf-8')
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule"""
        with self._lock:
            self.alert_rules.append(rule)
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check alert conditions"""
        alerts = []
        
        # This is a simplified implementation
        # In production, you'd integrate with a proper alerting system
        
        return alerts

class PerformanceTracker:
    """Thread-safe performance tracking with statistical analysis"""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self._data = defaultdict(lambda: deque(maxlen=max_samples))
        self._lock = threading.RLock()
    
    def record(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Record performance data"""
        with self._lock:
            self._data[operation].append({
                "timestamp": datetime.utcnow(),
                "duration": duration,
                "metadata": metadata or {}
            })
    
    def get_stats(self, operation: str) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            data = list(self._data[operation])
            
            if not data:
                return {"operation": operation, "sample_count": 0}
            
            durations = [d["duration"] for d in data]
            
            return {
                "operation": operation,
                "sample_count": len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "avg_duration": sum(durations) / len(durations),
                "p50_duration": self._percentile(durations, 50),
                "p95_duration": self._percentile(durations, 95),
                "p99_duration": self._percentile(durations, 99),
                "recent_samples": data[-10:]  # Last 10 samples
            }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile / 100
        f = int(k)
        c = k - f
        
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all performance statistics"""
        with self._lock:
            return {operation: self.get_stats(operation) for operation in self._data.keys()}

class MonitoringSystem:
    """Comprehensive monitoring orchestrator"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = StructuredLogger(service_name)
        self.metrics = MetricsCollector(service_name)
        self.performance = PerformanceTracker()
        
        # Background monitoring
        self._monitoring_thread = None
        self._monitoring_active = False
        
        # Start background monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name=f"monitor_{self.service_name}"
        )
        self._monitoring_thread.start()
        
        self.logger.info("Started monitoring system", operation="start_monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped monitoring system", operation="stop_monitoring")
    
    def _monitoring_worker(self):
        """Background monitoring worker"""
        while self._monitoring_active:
            try:
                # Update resource metrics
                self.metrics.update_resource_metrics()
                
                # Check alerts
                alerts = self.metrics.check_alerts()
                for alert in alerts:
                    self.logger.warning(
                        f"Alert triggered: {alert['name']}",
                        operation="alert",
                        **alert
                    )
                
                # Sleep for monitoring interval
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(
                    "Error in monitoring worker",
                    operation="monitoring_worker",
                    error=e
                )
                time.sleep(30)  # Wait before retrying
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "service": self.service_name,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "healthy",
                "uptime_seconds": time.time() - psutil.boot_time(),
                "memory": {
                    "rss_mb": memory_info.rss / 1024 / 1024,
                    "vms_mb": memory_info.vms / 1024 / 1024,
                    "percent": process.memory_percent()
                },
                "cpu": {
                    "percent": process.cpu_percent(),
                    "num_threads": process.num_threads()
                },
                "monitoring": {
                    "active": self._monitoring_active,
                    "thread_alive": self._monitoring_thread.is_alive() if self._monitoring_thread else False
                }
            }
            
        except Exception as e:
            return {
                "service": self.service_name,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_metrics_endpoint(self) -> str:
        """Get Prometheus metrics"""
        return self.metrics.get_metrics()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        return {
            "service": self.service_name,
            "timestamp": datetime.utcnow().isoformat(),
            "performance": self.performance.get_all_stats()
        }

# Monitoring decorators
def monitor_request(monitoring_system: MonitoringSystem, operation: str):
    """Monitor request performance and errors"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = hashlib.md5(f"{operation}_{time.time()}".encode()).hexdigest()[:8]
            
            # Set logging context
            monitoring_system.logger.set_context(
                request_id=request_id,
                operation=operation
            )
            
            try:
                monitoring_system.logger.info(
                    f"Starting {operation}",
                    operation=operation,
                    request_id=request_id
                )
                
                result = await func(*args, **kwargs)
                
                duration = time.time() - start_time
                monitoring_system.metrics.record_request(operation, duration, 200)
                monitoring_system.performance.record(operation, duration)
                
                monitoring_system.logger.info(
                    f"Completed {operation}",
                    operation=operation,
                    request_id=request_id,
                    duration_ms=duration * 1000,
                    status_code=200
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitoring_system.metrics.record_request(operation, duration, 500)
                monitoring_system.metrics.record_error(type(e).__name__)
                
                monitoring_system.logger.error(
                    f"Error in {operation}",
                    operation=operation,
                    request_id=request_id,
                    duration_ms=duration * 1000,
                    status_code=500,
                    error=e
                )
                
                raise
            
            finally:
                monitoring_system.logger.clear_context()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = hashlib.md5(f"{operation}_{time.time()}".encode()).hexdigest()[:8]
            
            monitoring_system.logger.set_context(
                request_id=request_id,
                operation=operation
            )
            
            try:
                monitoring_system.logger.info(
                    f"Starting {operation}",
                    operation=operation,
                    request_id=request_id
                )
                
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                monitoring_system.metrics.record_request(operation, duration, 200)
                monitoring_system.performance.record(operation, duration)
                
                monitoring_system.logger.info(
                    f"Completed {operation}",
                    operation=operation,
                    request_id=request_id,
                    duration_ms=duration * 1000,
                    status_code=200
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitoring_system.metrics.record_request(operation, duration, 500)
                monitoring_system.metrics.record_error(type(e).__name__)
                
                monitoring_system.logger.error(
                    f"Error in {operation}",
                    operation=operation,
                    request_id=request_id,
                    duration_ms=duration * 1000,
                    status_code=500,
                    error=e
                )
                
                raise
            
            finally:
                monitoring_system.logger.clear_context()
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def monitor_model_operation(monitoring_system: MonitoringSystem, model_name: str, operation_type: str):
    """Monitor model operations"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                duration = time.time() - start_time
                
                if operation_type == "load":
                    monitoring_system.metrics.record_model_load(model_name, duration)
                elif operation_type == "inference":
                    monitoring_system.metrics.record_model_inference(model_name, duration)
                
                monitoring_system.logger.info(
                    f"Model {operation_type} completed",
                    operation=f"model_{operation_type}",
                    model_name=model_name,
                    duration_ms=duration * 1000
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitoring_system.metrics.record_error(type(e).__name__)
                
                monitoring_system.logger.error(
                    f"Model {operation_type} failed",
                    operation=f"model_{operation_type}",
                    model_name=model_name,
                    duration_ms=duration * 1000,
                    error=e
                )
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                
                if operation_type == "load":
                    monitoring_system.metrics.record_model_load(model_name, duration)
                elif operation_type == "inference":
                    monitoring_system.metrics.record_model_inference(model_name, duration)
                
                monitoring_system.logger.info(
                    f"Model {operation_type} completed",
                    operation=f"model_{operation_type}",
                    model_name=model_name,
                    duration_ms=duration * 1000
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitoring_system.metrics.record_error(type(e).__name__)
                
                monitoring_system.logger.error(
                    f"Model {operation_type} failed",
                    operation=f"model_{operation_type}",
                    model_name=model_name,
                    duration_ms=duration * 1000,
                    error=e
                )
                
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# Global monitoring instances
_monitoring_instances: Dict[str, MonitoringSystem] = {}
_monitoring_lock = threading.RLock()

def get_monitoring_system(service_name: str) -> MonitoringSystem:
    """Get or create monitoring system for service"""
    with _monitoring_lock:
        if service_name not in _monitoring_instances:
            _monitoring_instances[service_name] = MonitoringSystem(service_name)
        
        return _monitoring_instances[service_name]

def shutdown_monitoring():
    """Shutdown all monitoring systems"""
    with _monitoring_lock:
        for monitoring_system in _monitoring_instances.values():
            monitoring_system.stop_monitoring()
        
        _monitoring_instances.clear()

# Cleanup on exit
def _cleanup_monitoring():
    """Cleanup monitoring on exit"""
    shutdown_monitoring()

import atexit
atexit.register(_cleanup_monitoring) 