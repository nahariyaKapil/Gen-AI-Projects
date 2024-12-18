"""
Cost Optimization Module for GenAI Portfolio
Thread-safe lazy loading, resource cleanup, and usage-based scaling
"""

import os
import gc
import time
import psutil
import asyncio
import threading
import weakref
import logging
from typing import Dict, List, Optional, Any, Callable, ContextManager
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import tracemalloc

from .config import get_config

config = get_config()
logger = logging.getLogger(__name__)

# Enable memory tracing
if config.debug:
    tracemalloc.start()

@dataclass
class ResourceUsage:
    """Track resource usage for cost optimization"""
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: float
    active_models: int
    last_request_time: datetime
    request_count: int
    thread_count: int = 0
    memory_growth_rate: float = 0.0

@dataclass
class CostThresholds:
    """Cost monitoring thresholds"""
    daily_limit: float = 10.0
    monthly_limit: float = 100.0
    idle_timeout_minutes: int = 5
    cleanup_interval_minutes: int = 30
    memory_limit_mb: int = 2048
    cpu_limit_percent: float = 80.0

@dataclass
class ModelMetrics:
    """Model performance and usage metrics"""
    load_count: int = 0
    total_requests: int = 0
    total_latency: float = 0.0
    memory_usage: float = 0.0
    last_used: datetime = field(default_factory=datetime.now)
    error_count: int = 0


class ModelReference:
    """Thread-safe weak reference wrapper for models"""
    
    def __init__(self, model: Any, name: str, cleanup_callback: Callable):
        self._ref = weakref.ref(model, cleanup_callback)
        self.name = name
        self._lock = threading.RLock()
        self.created_at = datetime.now()
        self.metrics = ModelMetrics()
    
    def get_model(self) -> Optional[Any]:
        """Get model if still alive"""
        with self._lock:
            return self._ref()
    
    def is_alive(self) -> bool:
        """Check if model is still alive"""
        return self._ref() is not None
    
    def record_usage(self, latency: float = 0.0):
        """Record model usage"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.total_latency += latency
            self.metrics.last_used = datetime.now()


class ThreadSafeLazyModelLoader:
    """Thread-safe lazy loading manager for AI models"""
    
    def __init__(self):
        self._models: Dict[str, ModelReference] = {}
        self._model_factories: Dict[str, Callable] = {}
        self._loading_locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()
        self._cleanup_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cleanup")
        self._memory_monitor = None
        self._running = True
        
        # Start background monitoring
        self._start_monitoring()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.shutdown()
    
    def _start_monitoring(self):
        """Start background monitoring threads"""
        self._memory_monitor = threading.Thread(
            target=self._memory_monitor_worker,
            daemon=True,
            name="memory_monitor"
        )
        self._memory_monitor.start()
        logger.info("Started memory monitoring thread")
    
    def _memory_monitor_worker(self):
        """Background memory monitoring worker"""
        while self._running:
            try:
                # Monitor every 30 seconds
                time.sleep(30)
                if not self._running:
                    break
                
                # Check memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_mb > config.memory_limit_mb * 0.8:  # 80% threshold
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                    self._cleanup_idle_models(force=True)
                
                # Cleanup dead references
                self._cleanup_dead_references()
                
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
    
    def register_model(self, name: str, factory: Callable):
        """Register a model factory for lazy loading"""
        with self._global_lock:
            if not callable(factory):
                raise ValueError(f"Factory for {name} must be callable")
            
            self._model_factories[name] = factory
            self._loading_locks[name] = threading.RLock()
            
            logger.info(f"Registered model factory: {name}")
    
    @contextmanager
    def get_model(self, name: str, timeout: float = 30.0) -> ContextManager[Any]:
        """Get model with context manager for automatic cleanup"""
        model = None
        try:
            model = self._load_model_if_needed(name, timeout)
            yield model
        finally:
            if model and hasattr(model, '__exit__'):
                try:
                    model.__exit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error during model cleanup: {e}")
    
    def _load_model_if_needed(self, name: str, timeout: float) -> Any:
        """Load model if needed with thread safety"""
        if name not in self._model_factories:
            raise ValueError(f"Model {name} not registered")
        
        with self._global_lock:
            # Check if model exists and is alive
            if name in self._models and self._models[name].is_alive():
                model_ref = self._models[name]
                model = model_ref.get_model()
                if model:
                    model_ref.record_usage()
                    logger.debug(f"Using cached model: {name}")
                    return model
        
        # Load model with specific lock
        with self._loading_locks[name]:
            # Double-check after acquiring specific lock
            with self._global_lock:
                if name in self._models and self._models[name].is_alive():
                    model_ref = self._models[name]
                    model = model_ref.get_model()
                    if model:
                        model_ref.record_usage()
                        return model
            
            # Load model
            logger.info(f"Loading model: {name}")
            start_time = time.time()
            
            try:
                factory = self._model_factories[name]
                model = factory()
                
                if model is None:
                    raise RuntimeError(f"Factory returned None for model {name}")
                
                load_time = time.time() - start_time
                
                # Create cleanup callback
                def cleanup_callback(ref):
                    with self._global_lock:
                        if name in self._models and self._models[name]._ref is ref:
                            del self._models[name]
                            logger.info(f"Cleaned up model reference: {name}")
                
                # Store model reference
                with self._global_lock:
                    model_ref = ModelReference(model, name, cleanup_callback)
                    model_ref.metrics.load_count = 1
                    self._models[name] = model_ref
                
                logger.info(f"Model {name} loaded in {load_time:.2f}s")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")
                raise
    
    def unload_model(self, name: str, force: bool = False):
        """Unload specific model"""
        with self._global_lock:
            if name not in self._models:
                logger.warning(f"Model {name} not found for unloading")
                return
            
            model_ref = self._models[name]
            model = model_ref.get_model()
            
            if model and not force:
                # Check if model is in use (simple heuristic)
                time_since_use = datetime.now() - model_ref.metrics.last_used
                if time_since_use.total_seconds() < 60:  # Used in last minute
                    logger.info(f"Model {name} recently used, skipping unload")
                    return
            
            # Remove from cache
            del self._models[name]
            
            # Force garbage collection
            del model_ref
            del model
            gc.collect()
            
            # Clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            
            logger.info(f"Unloaded model: {name}")
    
    def _cleanup_idle_models(self, idle_timeout_minutes: int = 5, force: bool = False):
        """Cleanup idle models with memory pressure consideration"""
        current_time = datetime.now()
        idle_threshold = current_time - timedelta(minutes=idle_timeout_minutes)
        
        # Get memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_pressure = memory_mb > config.memory_limit_mb * 0.7
        
        models_to_unload = []
        
        with self._global_lock:
            for name, model_ref in list(self._models.items()):
                should_unload = False
                
                if force or memory_pressure:
                    should_unload = True
                elif model_ref.metrics.last_used < idle_threshold:
                    should_unload = True
                elif not model_ref.is_alive():
                    should_unload = True
                
                if should_unload:
                    models_to_unload.append(name)
        
        # Unload models outside the global lock
        for name in models_to_unload:
            try:
                self.unload_model(name, force=force)
            except Exception as e:
                logger.error(f"Error unloading model {name}: {e}")
        
        if models_to_unload:
            logger.info(f"Unloaded {len(models_to_unload)} idle models")
        
        return len(models_to_unload)
    
    def _cleanup_dead_references(self):
        """Clean up dead weak references"""
        with self._global_lock:
            dead_refs = [
                name for name, ref in self._models.items()
                if not ref.is_alive()
            ]
            
            for name in dead_refs:
                del self._models[name]
                logger.debug(f"Cleaned up dead reference: {name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status with thread safety"""
        with self._global_lock:
            alive_models = {
                name: {
                    "is_alive": ref.is_alive(),
                    "load_count": ref.metrics.load_count,
                    "total_requests": ref.metrics.total_requests,
                    "avg_latency": (
                        ref.metrics.total_latency / ref.metrics.total_requests
                        if ref.metrics.total_requests > 0 else 0
                    ),
                    "last_used": ref.metrics.last_used.isoformat(),
                    "age_minutes": (datetime.now() - ref.created_at).total_seconds() / 60
                }
                for name, ref in self._models.items()
                if ref.is_alive()
            }
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "loaded_models": list(alive_models.keys()),
            "registered_models": list(self._model_factories.keys()),
            "model_details": alive_models,
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(),
            "thread_count": threading.active_count(),
            "is_running": self._running
        }
    
    def shutdown(self):
        """Shutdown with proper cleanup"""
        if not self._running:
            return
        
        logger.info("Shutting down lazy model loader...")
        self._running = False
        
        # Unload all models
        with self._global_lock:
            model_names = list(self._models.keys())
        
        for name in model_names:
            try:
                self.unload_model(name, force=True)
            except Exception as e:
                logger.error(f"Error unloading model {name} during shutdown: {e}")
        
        # Shutdown thread pool
        try:
            self._cleanup_executor.shutdown(wait=True, timeout=10)
        except Exception as e:
            logger.error(f"Error shutting down thread pool: {e}")
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Lazy model loader shutdown complete")


class ResourceMonitor:
    """Thread-safe resource monitoring with alerting"""
    
    def __init__(self, thresholds: CostThresholds):
        self.thresholds = thresholds
        self.usage_history: List[ResourceUsage] = []
        self.alerts_sent: Dict[str, datetime] = {}
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.RLock()
        self._alert_cooldown = timedelta(minutes=15)  # Prevent spam
    
    def start_monitoring(self):
        """Start resource monitoring"""
        with self._lock:
            if self._monitoring:
                return
            
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_worker,
                daemon=True,
                name="resource_monitor"
            )
            self._monitor_thread.start()
            logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        with self._lock:
            self._monitoring = False
            
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        logger.info("Stopped resource monitoring")
    
    def _monitor_worker(self):
        """Background monitoring worker"""
        while self._monitoring:
            try:
                usage = self._collect_usage()
                
                with self._lock:
                    self.usage_history.append(usage)
                    
                    # Keep only last 100 entries
                    if len(self.usage_history) > 100:
                        self.usage_history = self.usage_history[-100:]
                
                self._check_alerts(usage)
                
                # Sleep for monitoring interval
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _collect_usage(self) -> ResourceUsage:
        """Collect current resource usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # GPU memory (if available)
            gpu_memory = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            except ImportError:
                pass
            
            # Calculate memory growth rate
            memory_growth_rate = 0.0
            with self._lock:
                if len(self.usage_history) > 0:
                    prev_memory = self.usage_history[-1].memory_mb
                    current_memory = memory_info.rss / 1024 / 1024
                    memory_growth_rate = current_memory - prev_memory
            
            return ResourceUsage(
                cpu_percent=process.cpu_percent(),
                memory_mb=memory_info.rss / 1024 / 1024,
                gpu_memory_mb=gpu_memory,
                active_models=threading.active_count(),  # Approximate
                last_request_time=datetime.now(),
                request_count=0,  # Would be tracked by application
                thread_count=threading.active_count(),
                memory_growth_rate=memory_growth_rate
            )
            
        except Exception as e:
            logger.error(f"Error collecting usage: {e}")
            return ResourceUsage(
                cpu_percent=0, memory_mb=0, gpu_memory_mb=0,
                active_models=0, last_request_time=datetime.now(),
                request_count=0, thread_count=0, memory_growth_rate=0
            )
    
    def _check_alerts(self, usage: ResourceUsage):
        """Check for alert conditions"""
        alerts = []
        
        # Memory alerts
        if usage.memory_mb > self.thresholds.memory_limit_mb * 0.9:
            alerts.append(("memory_critical", f"Memory usage critical: {usage.memory_mb:.1f}MB"))
        elif usage.memory_mb > self.thresholds.memory_limit_mb * 0.8:
            alerts.append(("memory_warning", f"Memory usage high: {usage.memory_mb:.1f}MB"))
        
        # CPU alerts
        if usage.cpu_percent > self.thresholds.cpu_limit_percent:
            alerts.append(("cpu_high", f"CPU usage high: {usage.cpu_percent:.1f}%"))
        
        # Memory growth alerts
        if usage.memory_growth_rate > 50:  # 50MB growth
            alerts.append(("memory_leak", f"Rapid memory growth: +{usage.memory_growth_rate:.1f}MB"))
        
        # Send alerts with cooldown
        for alert_type, message in alerts:
            if self._should_send_alert(alert_type):
                self._send_alert(alert_type, message)
    
    def _should_send_alert(self, alert_type: str) -> bool:
        """Check if alert should be sent (cooldown check)"""
        with self._lock:
            last_sent = self.alerts_sent.get(alert_type)
            if last_sent is None:
                return True
            
            return datetime.now() - last_sent > self._alert_cooldown
    
    def _send_alert(self, alert_type: str, message: str):
        """Send alert (log for now, could integrate with alerting system)"""
        logger.warning(f"ALERT [{alert_type}]: {message}")
        
        with self._lock:
            self.alerts_sent[alert_type] = datetime.now()
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary with thread safety"""
        with self._lock:
            if not self.usage_history:
                return {"status": "no_data"}
            
            recent = self.usage_history[-10:]  # Last 10 readings
            
            return {
                "current": {
                    "cpu_percent": recent[-1].cpu_percent,
                    "memory_mb": recent[-1].memory_mb,
                    "gpu_memory_mb": recent[-1].gpu_memory_mb,
                    "thread_count": recent[-1].thread_count,
                    "memory_growth_rate": recent[-1].memory_growth_rate
                },
                "averages": {
                    "cpu_percent": sum(r.cpu_percent for r in recent) / len(recent),
                    "memory_mb": sum(r.memory_mb for r in recent) / len(recent),
                    "gpu_memory_mb": sum(r.gpu_memory_mb for r in recent) / len(recent)
                },
                "thresholds": {
                    "memory_limit_mb": self.thresholds.memory_limit_mb,
                    "cpu_limit_percent": self.thresholds.cpu_limit_percent
                },
                "history_count": len(self.usage_history),
                "monitoring": self._monitoring
            }


class AutoScaler:
    """Intelligent auto-scaling based on resource usage"""
    
    def __init__(self, lazy_loader: ThreadSafeLazyModelLoader, monitor: ResourceMonitor):
        self.lazy_loader = lazy_loader
        self.monitor = monitor
        self.request_count = 0
        self.last_scale_action = datetime.now()
        self._scaling = False
        self._scale_thread = None
        self._lock = threading.RLock()
        self.min_scale_interval = timedelta(minutes=2)  # Prevent rapid scaling
    
    def start_autoscaling(self):
        """Start autoscaling"""
        with self._lock:
            if self._scaling:
                return
            
            self._scaling = True
            self._scale_thread = threading.Thread(
                target=self._scaling_worker,
                daemon=True,
                name="autoscaler"
            )
            self._scale_thread.start()
            logger.info("Started autoscaling")
    
    def stop_autoscaling(self):
        """Stop autoscaling"""
        with self._lock:
            self._scaling = False
            
        if self._scale_thread and self._scale_thread.is_alive():
            self._scale_thread.join(timeout=5)
        
        logger.info("Stopped autoscaling")
    
    def record_request(self):
        """Record a request for scaling decisions"""
        with self._lock:
            self.request_count += 1
    
    def _scaling_worker(self):
        """Background scaling worker"""
        while self._scaling:
            try:
                time.sleep(120)  # Check every 2 minutes
                
                if not self._scaling:
                    break
                
                # Check if enough time has passed since last action
                with self._lock:
                    if datetime.now() - self.last_scale_action < self.min_scale_interval:
                        continue
                
                usage_summary = self.monitor.get_usage_summary()
                if usage_summary.get("status") == "no_data":
                    continue
                
                current = usage_summary["current"]
                
                # Scale down if low usage
                if (current["memory_mb"] > 1000 and  # Has models loaded
                    current["cpu_percent"] < 10 and     # Low CPU
                    datetime.now().hour in [2, 3, 4]):  # Off-peak hours
                    
                    self._scale_down()
                
            except Exception as e:
                logger.error(f"Autoscaling error: {e}")
                time.sleep(60)
    
    def _scale_down(self):
        """Scale down by unloading idle models"""
        try:
            unloaded = self.lazy_loader._cleanup_idle_models(idle_timeout_minutes=2)
            
            if unloaded > 0:
                with self._lock:
                    self.last_scale_action = datetime.now()
                
                logger.info(f"Auto-scaled down: unloaded {unloaded} models")
            
        except Exception as e:
            logger.error(f"Error during scale down: {e}")


class CostOptimizer:
    """Main cost optimization orchestrator"""
    
    def __init__(self, thresholds: Optional[CostThresholds] = None):
        self.thresholds = thresholds or CostThresholds()
        self.lazy_loader = ThreadSafeLazyModelLoader()
        self.monitor = ResourceMonitor(self.thresholds)
        self.autoscaler = AutoScaler(self.lazy_loader, self.monitor)
        self._initialized = False
        
        # Register cleanup on exit
        import atexit
        atexit.register(self.shutdown)
    
    def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
        
        logger.info("Initializing cost optimizer...")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Start autoscaling
        self.autoscaler.start_autoscaling()
        
        self._initialized = True
        logger.info("Cost optimizer initialized")
    
    def shutdown(self):
        """Shutdown all components"""
        if not self._initialized:
            return
        
        logger.info("Shutting down cost optimizer...")
        
        # Stop autoscaling
        self.autoscaler.stop_autoscaling()
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Shutdown lazy loader
        self.lazy_loader.shutdown()
        
        self._initialized = False
        logger.info("Cost optimizer shutdown complete")
    
    def register_model(self, name: str, factory: Callable):
        """Register a model factory"""
        self.lazy_loader.register_model(name, factory)
    
    @contextmanager
    def get_model(self, name: str, timeout: float = 30.0):
        """Get model with automatic resource management"""
        start_time = time.time()
        
        try:
            with self.lazy_loader.get_model(name, timeout) as model:
                self.autoscaler.record_request()
                yield model
        finally:
            # Record latency
            latency = time.time() - start_time
            if name in self.lazy_loader._models:
                self.lazy_loader._models[name].record_usage(latency)
    
    def get_cost_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive cost and performance dashboard"""
        loader_status = self.lazy_loader.get_status()
        usage_summary = self.monitor.get_usage_summary()
        
        # Memory trace info if available
        memory_trace = {}
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            memory_trace = {
                "traced_current_mb": current / 1024 / 1024,
                "traced_peak_mb": peak / 1024 / 1024
            }
        
        return {
            "status": "operational" if self._initialized else "not_initialized",
            "models": loader_status,
            "resources": usage_summary,
            "memory_trace": memory_trace,
            "thresholds": {
                "daily_limit": self.thresholds.daily_limit,
                "monthly_limit": self.thresholds.monthly_limit,
                "memory_limit_mb": self.thresholds.memory_limit_mb,
                "cpu_limit_percent": self.thresholds.cpu_limit_percent
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        loader_status = self.lazy_loader.get_status()
        usage_summary = self.monitor.get_usage_summary()
        
        if usage_summary.get("status") != "no_data":
            current = usage_summary["current"]
            
            # Memory recommendations
            memory_usage_percent = (current["memory_mb"] / self.thresholds.memory_limit_mb) * 100
            if memory_usage_percent > 80:
                recommendations.append("Consider unloading unused models to reduce memory usage")
            
            # Model recommendations
            if len(loader_status["loaded_models"]) > 3:
                recommendations.append("Multiple models loaded - consider implementing model rotation")
            
            # Growth rate recommendations
            if current.get("memory_growth_rate", 0) > 20:
                recommendations.append("Memory growth detected - check for potential memory leaks")
        
        if not recommendations:
            recommendations.append("System is optimally configured")
        
        return recommendations


def track_request(cost_optimizer: CostOptimizer):
    """Decorator to track requests and optimize resource usage"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                cost_optimizer.autoscaler.record_request()
                return await func(*args, **kwargs)
            finally:
                # Track request completion
                duration = time.time() - start_time
                logger.debug(f"Request completed in {duration:.2f}s")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                cost_optimizer.autoscaler.record_request()
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                logger.debug(f"Request completed in {duration:.2f}s")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global cost optimizer instance
_global_cost_optimizer: Optional[CostOptimizer] = None
_optimizer_lock = threading.RLock()

def get_cost_optimizer() -> CostOptimizer:
    """Get global cost optimizer instance"""
    global _global_cost_optimizer
    
    with _optimizer_lock:
        if _global_cost_optimizer is None:
            _global_cost_optimizer = CostOptimizer()
            _global_cost_optimizer.initialize()
        
        return _global_cost_optimizer

def shutdown_cost_optimizer():
    """Shutdown global cost optimizer"""
    global _global_cost_optimizer
    
    with _optimizer_lock:
        if _global_cost_optimizer is not None:
            _global_cost_optimizer.shutdown()
            _global_cost_optimizer = None

# Cleanup on module exit
def _cleanup_on_exit():
    """Cleanup function called on module exit"""
    shutdown_cost_optimizer()

import atexit
atexit.register(_cleanup_on_exit) 