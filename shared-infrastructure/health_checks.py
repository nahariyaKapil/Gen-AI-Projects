"""
Comprehensive Health Check System
Production-ready health monitoring for all services
Built for reliability and early problem detection
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib

import psutil
import redis
import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from google.cloud import storage, monitoring_v3
from google.cloud.exceptions import GoogleCloudError

from .config import get_config
from .monitoring import get_monitoring_system

config = get_config()
monitoring = get_monitoring_system("health_checks")


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    service: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class HealthThresholds:
    """Configurable thresholds for health checks"""
    response_time_warning: float = 1000  # ms
    response_time_critical: float = 5000  # ms
    memory_usage_warning: float = 80  # percentage
    memory_usage_critical: float = 95  # percentage
    cpu_usage_warning: float = 80  # percentage
    cpu_usage_critical: float = 95  # percentage
    disk_usage_warning: float = 80  # percentage
    disk_usage_critical: float = 90  # percentage


class BaseHealthCheck:
    """Base class for health checks"""
    
    def __init__(self, name: str, thresholds: Optional[HealthThresholds] = None):
        self.name = name
        self.thresholds = thresholds or HealthThresholds()
        self.last_result: Optional[HealthCheckResult] = None
    
    async def check(self) -> HealthCheckResult:
        """Perform health check"""
        start_time = time.time()
        
        try:
            monitoring.logger.debug(f"Starting health check: {self.name}")
            
            # Perform the actual check
            status, message, details = await self._perform_check()
            
            duration_ms = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                service=self.name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details=details
            )
            
            self.last_result = result
            
            monitoring.logger.debug(
                f"Health check completed: {self.name}",
                status=status.value,
                duration_ms=duration_ms
            )
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            result = HealthCheckResult(
                service=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {error_msg}",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                error=error_msg
            )
            
            self.last_result = result
            
            monitoring.logger.error(
                f"Health check failed: {self.name}",
                error=e,
                duration_ms=duration_ms
            )
            
            return result
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Override this method in subclasses"""
        raise NotImplementedError


class SystemResourcesHealthCheck(BaseHealthCheck):
    """Check system resources (CPU, memory, disk)"""
    
    def __init__(self, thresholds: Optional[HealthThresholds] = None):
        super().__init__("system_resources", thresholds)
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": (disk.used / disk.total) * 100,
            "disk_free_gb": disk.free / (1024**3),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
        
        # Determine status based on thresholds
        status = HealthStatus.HEALTHY
        issues = []
        
        if cpu_percent >= self.thresholds.cpu_usage_critical:
            status = HealthStatus.UNHEALTHY
            issues.append(f"Critical CPU usage: {cpu_percent:.1f}%")
        elif cpu_percent >= self.thresholds.cpu_usage_warning:
            status = HealthStatus.DEGRADED
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if memory.percent >= self.thresholds.memory_usage_critical:
            status = HealthStatus.UNHEALTHY
            issues.append(f"Critical memory usage: {memory.percent:.1f}%")
        elif memory.percent >= self.thresholds.memory_usage_warning:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            issues.append(f"High memory usage: {memory.percent:.1f}%")
        
        disk_percent = (disk.used / disk.total) * 100
        if disk_percent >= self.thresholds.disk_usage_critical:
            status = HealthStatus.UNHEALTHY
            issues.append(f"Critical disk usage: {disk_percent:.1f}%")
        elif disk_percent >= self.thresholds.disk_usage_warning:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            issues.append(f"High disk usage: {disk_percent:.1f}%")
        
        if issues:
            message = "; ".join(issues)
        else:
            message = "System resources are healthy"
        
        return status, message, details


class DatabaseHealthCheck(BaseHealthCheck):
    """Check database connectivity and performance"""
    
    def __init__(self, database_url: str, thresholds: Optional[HealthThresholds] = None):
        super().__init__("database", thresholds)
        self.database_url = database_url
        self.engine = None
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        try:
            if not self.engine:
                self.engine = create_engine(self.database_url, pool_pre_ping=True)
            
            # Test connection with a simple query
            start_time = time.time()
            
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                row = result.fetchone()
                
                if not row or row[0] != 1:
                    raise Exception("Database query returned unexpected result")
            
            query_time_ms = (time.time() - start_time) * 1000
            
            details = {
                "query_time_ms": query_time_ms,
                "pool_size": self.engine.pool.size(),
                "checked_out": self.engine.pool.checkedout(),
                "overflow": self.engine.pool.overflow(),
                "checked_in": self.engine.pool.checkedin()
            }
            
            # Check response time
            if query_time_ms >= self.thresholds.response_time_critical:
                status = HealthStatus.UNHEALTHY
                message = f"Database response time critical: {query_time_ms:.1f}ms"
            elif query_time_ms >= self.thresholds.response_time_warning:
                status = HealthStatus.DEGRADED
                message = f"Database response time high: {query_time_ms:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database is healthy (response: {query_time_ms:.1f}ms)"
            
            return status, message, details
            
        except SQLAlchemyError as e:
            return HealthStatus.UNHEALTHY, f"Database error: {str(e)}", {"error_type": "sqlalchemy_error"}
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Database connection failed: {str(e)}", {"error_type": "connection_error"}


class RedisHealthCheck(BaseHealthCheck):
    """Check Redis connectivity and performance"""
    
    def __init__(self, redis_url: str, thresholds: Optional[HealthThresholds] = None):
        super().__init__("redis", thresholds)
        self.redis_url = redis_url
        self.redis_client = None
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        try:
            if not self.redis_client:
                self.redis_client = redis.from_url(self.redis_url)
            
            # Test connection with ping and info
            start_time = time.time()
            
            ping_result = self.redis_client.ping()
            if not ping_result:
                raise Exception("Redis ping failed")
            
            info = self.redis_client.info()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            details = {
                "response_time_ms": response_time_ms,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "used_memory_percentage": info.get("used_memory_rss", 0) / info.get("maxmemory", 1) * 100 if info.get("maxmemory") else 0,
                "redis_version": info.get("redis_version", "unknown"),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }
            
            # Check response time
            if response_time_ms >= self.thresholds.response_time_critical:
                status = HealthStatus.UNHEALTHY
                message = f"Redis response time critical: {response_time_ms:.1f}ms"
            elif response_time_ms >= self.thresholds.response_time_warning:
                status = HealthStatus.DEGRADED
                message = f"Redis response time high: {response_time_ms:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Redis is healthy (response: {response_time_ms:.1f}ms)"
            
            return status, message, details
            
        except redis.RedisError as e:
            return HealthStatus.UNHEALTHY, f"Redis error: {str(e)}", {"error_type": "redis_error"}
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Redis connection failed: {str(e)}", {"error_type": "connection_error"}


class GoogleCloudHealthCheck(BaseHealthCheck):
    """Check Google Cloud services connectivity"""
    
    def __init__(self, project_id: str, thresholds: Optional[HealthThresholds] = None):
        super().__init__("google_cloud", thresholds)
        self.project_id = project_id
        self.storage_client = None
        self.monitoring_client = None
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        details = {}
        issues = []
        overall_status = HealthStatus.HEALTHY
        
        # Test Cloud Storage
        try:
            if not self.storage_client:
                self.storage_client = storage.Client(project=self.project_id)
            
            start_time = time.time()
            
            # List buckets to test connectivity
            buckets = list(self.storage_client.list_buckets(max_results=1))
            
            storage_time_ms = (time.time() - start_time) * 1000
            details["storage_response_ms"] = storage_time_ms
            details["storage_accessible"] = True
            
            if storage_time_ms >= self.thresholds.response_time_warning:
                issues.append(f"Cloud Storage slow: {storage_time_ms:.1f}ms")
                overall_status = HealthStatus.DEGRADED
            
        except GoogleCloudError as e:
            details["storage_accessible"] = False
            details["storage_error"] = str(e)
            issues.append(f"Cloud Storage error: {str(e)}")
            overall_status = HealthStatus.UNHEALTHY
        except Exception as e:
            details["storage_accessible"] = False
            details["storage_error"] = str(e)
            issues.append(f"Cloud Storage connection failed: {str(e)}")
            overall_status = HealthStatus.UNHEALTHY
        
        # Test Cloud Monitoring
        try:
            if not self.monitoring_client:
                self.monitoring_client = monitoring_v3.MetricServiceClient()
            
            start_time = time.time()
            
            # List metric descriptors to test connectivity
            project_name = f"projects/{self.project_id}"
            descriptors = list(self.monitoring_client.list_metric_descriptors(
                name=project_name,
                page_size=1
            ))
            
            monitoring_time_ms = (time.time() - start_time) * 1000
            details["monitoring_response_ms"] = monitoring_time_ms
            details["monitoring_accessible"] = True
            
            if monitoring_time_ms >= self.thresholds.response_time_warning:
                issues.append(f"Cloud Monitoring slow: {monitoring_time_ms:.1f}ms")
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            
        except GoogleCloudError as e:
            details["monitoring_accessible"] = False
            details["monitoring_error"] = str(e)
            issues.append(f"Cloud Monitoring error: {str(e)}")
            overall_status = HealthStatus.UNHEALTHY
        except Exception as e:
            details["monitoring_accessible"] = False
            details["monitoring_error"] = str(e)
            issues.append(f"Cloud Monitoring connection failed: {str(e)}")
            overall_status = HealthStatus.UNHEALTHY
        
        if issues:
            message = "; ".join(issues)
        else:
            message = "Google Cloud services are healthy"
        
        return overall_status, message, details


class ExternalServiceHealthCheck(BaseHealthCheck):
    """Check external HTTP service connectivity"""
    
    def __init__(self, name: str, url: str, timeout: float = 10.0, 
                 thresholds: Optional[HealthThresholds] = None):
        super().__init__(f"external_service_{name}", thresholds)
        self.url = url
        self.timeout = timeout
        self.http_client = None
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        try:
            if not self.http_client:
                self.http_client = httpx.AsyncClient(timeout=self.timeout)
            
            start_time = time.time()
            
            response = await self.http_client.get(self.url)
            
            response_time_ms = (time.time() - start_time) * 1000
            
            details = {
                "url": self.url,
                "status_code": response.status_code,
                "response_time_ms": response_time_ms,
                "headers": dict(response.headers)
            }
            
            # Check status code
            if response.status_code >= 500:
                status = HealthStatus.UNHEALTHY
                message = f"Service error: HTTP {response.status_code}"
            elif response.status_code >= 400:
                status = HealthStatus.DEGRADED
                message = f"Service issue: HTTP {response.status_code}"
            elif response_time_ms >= self.thresholds.response_time_critical:
                status = HealthStatus.UNHEALTHY
                message = f"Service response time critical: {response_time_ms:.1f}ms"
            elif response_time_ms >= self.thresholds.response_time_warning:
                status = HealthStatus.DEGRADED
                message = f"Service response time high: {response_time_ms:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Service is healthy (HTTP {response.status_code}, {response_time_ms:.1f}ms)"
            
            return status, message, details
            
        except httpx.TimeoutException:
            return HealthStatus.UNHEALTHY, f"Service timeout after {self.timeout}s", {"url": self.url, "error_type": "timeout"}
        except httpx.RequestError as e:
            return HealthStatus.UNHEALTHY, f"Service connection failed: {str(e)}", {"url": self.url, "error_type": "connection_error"}
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Service check failed: {str(e)}", {"url": self.url, "error_type": "unknown_error"}


class HealthCheckManager:
    """Manages and coordinates all health checks"""
    
    def __init__(self):
        self.checks: Dict[str, BaseHealthCheck] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.check_interval = 30  # seconds
        self.background_task = None
        self.is_running = False
    
    def register_check(self, check: BaseHealthCheck):
        """Register a health check"""
        self.checks[check.name] = check
        monitoring.logger.info(f"Registered health check: {check.name}")
    
    def unregister_check(self, name: str):
        """Unregister a health check"""
        if name in self.checks:
            del self.checks[name]
            if name in self.last_results:
                del self.last_results[name]
            monitoring.logger.info(f"Unregistered health check: {name}")
    
    async def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check"""
        if name not in self.checks:
            return None
        
        result = await self.checks[name].check()
        self.last_results[name] = result
        
        # Log significant status changes
        if result.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
            monitoring.logger.warning(
                f"Health check issue detected",
                check_name=name,
                status=result.status.value,
                message=result.message
            )
        
        return result
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        if not self.checks:
            return results
        
        # Run checks concurrently
        tasks = {name: self.run_check(name) for name in self.checks.keys()}
        
        for name, task in tasks.items():
            try:
                result = await task
                if result:
                    results[name] = result
            except Exception as e:
                monitoring.logger.error(f"Failed to run health check {name}", error=e)
                
                # Create error result
                results[name] = HealthCheckResult(
                    service=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check execution failed: {str(e)}",
                    duration_ms=0,
                    timestamp=datetime.utcnow(),
                    error=str(e)
                )
        
        self.last_results.update(results)
        return results
    
    def get_overall_status(self) -> Tuple[HealthStatus, str]:
        """Get overall system health status"""
        if not self.last_results:
            return HealthStatus.UNKNOWN, "No health checks have been performed"
        
        unhealthy_count = sum(1 for r in self.last_results.values() if r.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for r in self.last_results.values() if r.status == HealthStatus.DEGRADED)
        healthy_count = sum(1 for r in self.last_results.values() if r.status == HealthStatus.HEALTHY)
        
        total_checks = len(self.last_results)
        
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY, f"{unhealthy_count}/{total_checks} checks unhealthy"
        elif degraded_count > 0:
            return HealthStatus.DEGRADED, f"{degraded_count}/{total_checks} checks degraded"
        elif healthy_count == total_checks:
            return HealthStatus.HEALTHY, f"All {total_checks} checks healthy"
        else:
            return HealthStatus.UNKNOWN, "Mixed health check results"
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        overall_status, overall_message = self.get_overall_status()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall_status.value,
            "overall_message": overall_message,
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                    "timestamp": result.timestamp.isoformat(),
                    "details": result.details,
                    "error": result.error
                }
                for name, result in self.last_results.items()
            },
            "summary": {
                "total_checks": len(self.last_results),
                "healthy": sum(1 for r in self.last_results.values() if r.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for r in self.last_results.values() if r.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for r in self.last_results.values() if r.status == HealthStatus.UNHEALTHY),
                "unknown": sum(1 for r in self.last_results.values() if r.status == HealthStatus.UNKNOWN)
            }
        }
    
    async def start_background_checks(self):
        """Start background health checks"""
        if self.is_running:
            return
        
        self.is_running = True
        self.background_task = asyncio.create_task(self._background_check_loop())
        monitoring.logger.info("Started background health checks")
    
    async def stop_background_checks(self):
        """Stop background health checks"""
        self.is_running = False
        
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
        
        monitoring.logger.info("Stopped background health checks")
    
    async def _background_check_loop(self):
        """Background loop for running health checks"""
        while self.is_running:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                monitoring.logger.error("Error in background health check loop", error=e)
                await asyncio.sleep(10)  # Wait before retrying


# Global health check manager
health_manager = HealthCheckManager()


def setup_default_health_checks():
    """Setup default health checks for the application"""
    # System resources check
    health_manager.register_check(SystemResourcesHealthCheck())
    
    # Database check if configured
    if hasattr(config, 'database_url') and config.database_url:
        health_manager.register_check(DatabaseHealthCheck(config.database_url))
    
    # Redis check if configured
    if hasattr(config, 'redis_url') and config.redis_url:
        health_manager.register_check(RedisHealthCheck(config.redis_url))
    
    # Google Cloud check if configured
    if hasattr(config, 'gcp_project_id') and config.gcp_project_id:
        health_manager.register_check(GoogleCloudHealthCheck(config.gcp_project_id))


async def get_health_status() -> Dict[str, Any]:
    """Get current health status"""
    return health_manager.get_health_report()


async def run_health_checks() -> Dict[str, Any]:
    """Run all health checks and return results"""
    await health_manager.run_all_checks()
    return health_manager.get_health_report() 