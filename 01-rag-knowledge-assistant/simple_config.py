"""
Simple configuration for RAG Knowledge Assistant
Local configuration to avoid shared infrastructure dependencies
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    
    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Server settings
    host: str = "localhost"
    port: int = 8000
    debug: bool = True
    log_level: str = "INFO"
    
    # Security
    allowed_origins: List[str] = None
    
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


def get_rag_config() -> RAGConfig:
    """Get RAG configuration"""
    return RAGConfig()


def get_monitoring_system(service_name: str) -> SimpleMonitoring:
    """Get monitoring system"""
    return SimpleMonitoring(service_name)


def get_current_user() -> AuthenticatedUser:
    """Get current authenticated user (demo mode)"""
    return AuthenticatedUser() 