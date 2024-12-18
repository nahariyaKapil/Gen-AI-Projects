"""
Shared configuration for all GenAI projects
Secure configuration management with proper secret handling
"""

import os
import hashlib
import secrets
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings
from cryptography.fernet import Fernet


class SecretManager:
    """Secure secret management with encryption"""
    
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key"""
        key_file = os.environ.get('ENCRYPTION_KEY_FILE', '.key')
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Read-only for owner
            return key
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()


class SharedConfig(BaseSettings):
    """Base configuration with security-first approach"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Security
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=1, env="JWT_EXPIRATION_HOURS")  # Reduced from 24
    jwt_refresh_days: int = Field(default=7, env="JWT_REFRESH_DAYS")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    # Google Cloud Platform
    gcp_project_id: str = Field(..., env="GOOGLE_CLOUD_PROJECT")
    gcp_region: str = Field(default="us-central1", env="GCP_REGION")
    gcp_credentials_path: Optional[str] = Field(default=None, env="GOOGLE_APPLICATION_CREDENTIALS")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_ssl_mode: str = Field(default="require", env="DATABASE_SSL_MODE")
    
    # Redis
    redis_url: str = Field(..., env="REDIS_URL")
    redis_ssl: bool = Field(default=True, env="REDIS_SSL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # API Configuration
    api_rate_limit: int = Field(default=100, env="API_RATE_LIMIT")
    max_request_size: int = Field(default=10485760, env="MAX_REQUEST_SIZE")  # 10MB
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    default_model: str = Field(default="gpt-3.5-turbo", env="DEFAULT_MODEL")
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # File Upload Security
    max_file_size: int = Field(default=52428800, env="MAX_FILE_SIZE")  # 50MB
    allowed_file_types: List[str] = Field(
        default=["pdf", "txt", "docx", "md", "csv", "json"],
        env="ALLOWED_FILE_TYPES"
    )
    upload_dir: str = Field(default="/tmp/uploads", env="UPLOAD_DIR")
    
    # Monitoring
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=8080, env="METRICS_PORT")
    
    # Resource Limits
    max_concurrent_requests: int = Field(default=50, env="MAX_CONCURRENT_REQUESTS")
    memory_limit_mb: int = Field(default=2048, env="MEMORY_LIMIT_MB")
    cpu_limit: float = Field(default=1.0, env="CPU_LIMIT")
    
    @validator('jwt_secret_key')
    def validate_jwt_secret(cls, v):
        if not v or v == "your-secret-key":
            raise ValueError("JWT secret key must be set and cannot be default value")
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v
    
    @validator('environment')
    def validate_environment(cls, v):
        if v not in ['development', 'staging', 'production']:
            raise ValueError("Environment must be development, staging, or production")
        return v
    
    @validator('openai_api_key', 'anthropic_api_key')
    def validate_api_keys(cls, v):
        if v and (len(v) < 10 or not v.startswith(('sk-', 'anthropic-'))):
            raise ValueError("Invalid API key format")
        return v
    
    @validator('allowed_file_types')
    def validate_file_types(cls, v):
        safe_types = ["pdf", "txt", "docx", "md", "csv", "json", "py", "js", "ts"]
        for file_type in v:
            if file_type not in safe_types:
                raise ValueError(f"File type {file_type} not allowed")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from environment with fallback to secret manager"""
        secret = os.environ.get(key)
        if not secret:
            return None
        
        # Decrypt if encrypted
        if secret.startswith('enc:'):
            secret_manager = SecretManager()
            return secret_manager.decrypt(secret[4:])
        
        return secret


class RAGConfig(SharedConfig):
    """Configuration for RAG applications"""
    
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=5, env="RETRIEVAL_TOP_K")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # Vector DB
    vector_db_type: str = Field(default="faiss", env="VECTOR_DB_TYPE")
    vector_db_persist_dir: str = Field(default="./vector_db", env="VECTOR_DB_PERSIST_DIR")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    @validator('chunk_size')
    def validate_chunk_size(cls, v):
        if v < 100 or v > 2000:
            raise ValueError("Chunk size must be between 100 and 2000")
        return v


class CodeLLMConfig(SharedConfig):
    """Configuration for Code LLM applications"""
    
    code_model: str = Field(default="codellama/CodeLlama-7b-Instruct-hf", env="CODE_MODEL")
    max_code_length: int = Field(default=8192, env="MAX_CODE_LENGTH")
    supported_languages: List[str] = Field(
        default=["python", "javascript", "typescript", "java", "cpp", "go", "rust"],
        env="SUPPORTED_LANGUAGES"
    )
    enable_code_execution: bool = Field(default=False, env="ENABLE_CODE_EXECUTION")
    
    # Security settings
    code_execution_timeout: int = Field(default=10, env="CODE_EXECUTION_TIMEOUT")
    sandbox_enabled: bool = Field(default=True, env="SANDBOX_ENABLED")


class BenchmarkConfig(SharedConfig):
    """Configuration for LLM benchmarking"""
    
    benchmark_models: List[str] = Field(
        default=["gpt-3.5-turbo", "gpt-4"],
        env="BENCHMARK_MODELS"
    )
    
    target_latency_ms: int = Field(default=2000, env="TARGET_LATENCY_MS")
    target_throughput_qps: int = Field(default=10, env="TARGET_THROUGHPUT_QPS")
    
    # Load testing
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    test_duration_minutes: int = Field(default=5, env="TEST_DURATION_MINUTES")


class MultilingualConfig(SharedConfig):
    """Configuration for multilingual applications"""
    
    supported_languages: List[str] = Field(
        default=["en", "es", "fr", "de", "ja", "zh"],
        env="SUPPORTED_LANGUAGES"
    )
    default_language: str = Field(default="en", env="DEFAULT_LANGUAGE")
    
    translation_model: str = Field(
        default="Helsinki-NLP/opus-mt-en-hi",
        env="TRANSLATION_MODEL"
    )
    min_confidence_threshold: float = Field(default=0.8, env="MIN_CONFIDENCE_THRESHOLD")


# Secure configuration instances
def get_config() -> SharedConfig:
    """Get validated configuration instance"""
    return SharedConfig()

def get_rag_config() -> RAGConfig:
    """Get RAG configuration instance"""
    return RAGConfig()

def get_code_llm_config() -> CodeLLMConfig:
    """Get Code LLM configuration instance"""
    return CodeLLMConfig()

def get_benchmark_config() -> BenchmarkConfig:
    """Get benchmark configuration instance"""
    return BenchmarkConfig()

def get_multilingual_config() -> MultilingualConfig:
    """Get multilingual configuration instance"""
    return MultilingualConfig()

# Default instance
config = get_config() 