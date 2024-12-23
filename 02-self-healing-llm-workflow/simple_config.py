"""
Simple configuration for Self-Healing LLM Workflow
Local configuration to avoid shared infrastructure dependencies
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class WorkflowConfig:
    """Configuration for workflow system"""
    
    # Server settings
    host: str = "localhost"
    port: int = 8001
    debug: bool = True
    log_level: str = "INFO"
    
    # Security
    allowed_origins: List[str] = None
    
    # Workflow settings
    max_workflows: int = 10
    workflow_timeout: int = 3600  # 1 hour
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["*"]


class SimpleMonitoring:
    """Simple monitoring system"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = SimpleLogger()


class SimpleLogger:
    """Simple logger replacement"""
    
    def info(self, message: str, **kwargs):
        print(f"INFO: {message}")
        if kwargs:
            print(f"  Details: {kwargs}")
    
    def warning(self, message: str, **kwargs):
        print(f"WARNING: {message}")
        if kwargs:
            print(f"  Details: {kwargs}")
    
    def error(self, message: str, **kwargs):
        print(f"ERROR: {message}")
        if kwargs:
            print(f"  Details: {kwargs}")
    
    def debug(self, message: str, **kwargs):
        if os.getenv("DEBUG", "false").lower() == "true":
            print(f"DEBUG: {message}")
            if kwargs:
                print(f"  Details: {kwargs}")


class AuthenticatedUser:
    """Simple authenticated user"""
    
    def __init__(self, user_id: str = "demo_user", is_admin: bool = False):
        self.user_id = user_id
        self.is_admin = is_admin


def get_workflow_config() -> WorkflowConfig:
    """Get workflow configuration"""
    return WorkflowConfig()


def get_monitoring_system(service_name: str) -> SimpleMonitoring:
    """Get monitoring system"""
    return SimpleMonitoring(service_name)


def get_current_user() -> AuthenticatedUser:
    """Get current authenticated user (demo mode)"""
    return AuthenticatedUser()


class WorkflowError(Exception):
    """Exception for workflow operations"""
    pass 