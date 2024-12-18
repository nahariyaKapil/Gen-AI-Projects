"""
Production-ready authentication and security infrastructure
Secure JWT auth with RBAC, rate limiting, and session management
"""

import jwt
import bcrypt
import redis
import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set
from functools import wraps
from dataclasses import dataclass
from enum import Enum

from fastapi import HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, select
from sqlalchemy.exc import SQLAlchemyError
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter

from .config import get_config

config = get_config()


class UserRole(str, Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    USER = "user"
    GUEST = "guest"


@dataclass
class User:
    id: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = None


@dataclass
class TokenData:
    user_id: str
    email: str
    role: UserRole
    token_type: str
    expires_at: datetime


class SecurityManager:
    """Comprehensive security management"""
    
    def __init__(self):
        self.security = HTTPBearer()
        self.redis_client = self._get_redis_client()
        self.db = self._get_firestore_client()
        self.token_blacklist: Set[str] = set()
        
    def _get_redis_client(self) -> Optional[redis.Redis]:
        """Get Redis client with proper error handling"""
        try:
            client = redis.from_url(
                config.redis_url,
                ssl=config.redis_ssl,
                ssl_cert_reqs="required",
                decode_responses=True
            )
            client.ping()
            return client
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return None
    
    def _get_firestore_client(self) -> Optional[firestore.Client]:
        """Get Firestore client with proper error handling"""
        try:
            if config.gcp_project_id:
                return firestore.Client(project=config.gcp_project_id)
        except Exception as e:
            print(f"Firestore connection failed: {e}")
        return None
    
    def hash_password(self, password: str) -> str:
        """Hash password with secure bcrypt settings"""
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        # Use 12 rounds for proper security
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False
    
    def create_tokens(self, user: User) -> Dict[str, str]:
        """Create access and refresh tokens"""
        access_token = self._create_token(user, "access")
        refresh_token = self._create_token(user, "refresh")
        
        # Store refresh token
        if self.redis_client:
            self.redis_client.setex(
                f"refresh:{user.id}",
                timedelta(days=config.jwt_refresh_days),
                refresh_token
            )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": config.jwt_expiration_hours * 3600
        }
    
    def _create_token(self, user: User, token_type: str) -> str:
        """Create JWT token with proper claims"""
        now = datetime.utcnow()
        
        if token_type == "access":
            expires_delta = timedelta(hours=config.jwt_expiration_hours)
        else:  # refresh
            expires_delta = timedelta(days=config.jwt_refresh_days)
        
        payload = {
            "user_id": user.id,
            "email": user.email,
            "role": user.role.value,
            "token_type": token_type,
            "iat": now,
            "exp": now + expires_delta,
            "jti": secrets.token_urlsafe(32)  # JWT ID for blacklisting
        }
        
        return jwt.encode(payload, config.jwt_secret_key, algorithm=config.jwt_algorithm)
    
    def verify_token(self, token: str) -> TokenData:
        """Verify and decode JWT token"""
        try:
            # Check blacklist
            if token in self.token_blacklist:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            payload = jwt.decode(
                token,
                config.jwt_secret_key,
                algorithms=[config.jwt_algorithm]
            )
            
            # Check if token is blacklisted in Redis
            if self.redis_client and payload.get("jti"):
                if self.redis_client.get(f"blacklist:{payload['jti']}"):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has been revoked"
                    )
            
            return TokenData(
                user_id=payload["user_id"],
                email=payload["email"],
                role=UserRole(payload["role"]),
                token_type=payload["token_type"],
                expires_at=datetime.fromtimestamp(payload["exp"])
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def revoke_token(self, token: str) -> None:
        """Revoke a token"""
        try:
            payload = jwt.decode(
                token,
                config.jwt_secret_key,
                algorithms=[config.jwt_algorithm],
                options={"verify_exp": False}
            )
            
            jti = payload.get("jti")
            if jti and self.redis_client:
                # Add to blacklist with expiration
                exp = payload.get("exp", 0)
                ttl = max(0, exp - datetime.utcnow().timestamp())
                self.redis_client.setex(f"blacklist:{jti}", int(ttl), "1")
            
            # Add to memory blacklist as fallback
            self.token_blacklist.add(token)
            
        except jwt.JWTError:
            pass  # Token invalid, no need to blacklist
    
    def refresh_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token"""
        token_data = self.verify_token(refresh_token)
        
        if token_data.token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Get user and create new tokens
        user = self.get_user_by_id(token_data.user_id)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Revoke old tokens
        self.revoke_token(refresh_token)
        
        return self.create_tokens(user)


class RoleBasedAccessControl:
    """Role-based access control system"""
    
    PERMISSIONS = {
        UserRole.ADMIN: [
            "read", "write", "delete", "manage_users", "view_analytics",
            "manage_system", "view_logs", "manage_roles"
        ],
        UserRole.DEVELOPER: [
            "read", "write", "view_analytics", "execute_code", "manage_workflows"
        ],
        UserRole.USER: ["read", "write"],
        UserRole.GUEST: ["read"]
    }
    
    PROJECT_PERMISSIONS = {
        "rag-knowledge-assistant": ["read", "write", "upload_documents"],
        "self-healing-llm": ["read", "write", "manage_workflows"],
        "code-llm-assistant": ["read", "write", "execute_code"],
        "llm-benchmark": ["read", "view_analytics"],
        "multilingual-enterprise": ["read", "write", "manage_translations"]
    }
    
    @classmethod
    def has_permission(cls, user_role: UserRole, permission: str, 
                      project: Optional[str] = None) -> bool:
        """Check if user role has specific permission"""
        role_permissions = cls.PERMISSIONS.get(user_role, [])
        
        if permission in role_permissions:
            return True
        
        if project and project in cls.PROJECT_PERMISSIONS:
            project_permissions = cls.PROJECT_PERMISSIONS[project]
            return permission in project_permissions and "write" in role_permissions
        
        return False
    
    @classmethod
    def get_user_permissions(cls, user_role: UserRole) -> List[str]:
        """Get all permissions for a user role"""
        return cls.PERMISSIONS.get(user_role, [])


class RateLimiter:
    """Redis-based rate limiting with sliding window"""
    
    def __init__(self, redis_client: Optional[redis.Redis]):
        self.redis = redis_client
    
    def is_allowed(self, identifier: str, limit: int, 
                  window: int = 60, burst_limit: int = None) -> bool:
        """Check if request is allowed based on rate limit"""
        if not self.redis:
            return True
        
        now = datetime.utcnow().timestamp()
        key = f"rate_limit:{identifier}"
        
        # Sliding window rate limiting
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, now - window)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiration
        pipe.expire(key, window)
        
        results = pipe.execute()
        current_count = results[1]
        
        # Check burst limit first
        if burst_limit and current_count >= burst_limit:
            return False
        
        return current_count < limit
    
    def get_remaining(self, identifier: str, limit: int, window: int = 60) -> int:
        """Get remaining requests in current window"""
        if not self.redis:
            return limit
        
        now = datetime.utcnow().timestamp()
        key = f"rate_limit:{identifier}"
        
        # Clean old entries and count
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zcard(key)
        
        results = pipe.execute()
        current_count = results[1]
        
        return max(0, limit - current_count)


class UserManager:
    """User management with secure database operations"""
    
    def __init__(self, firestore_client: Optional[firestore.Client]):
        self.db = firestore_client
        self.collection = "users"
    
    def create_user(self, email: str, password: str, role: UserRole = UserRole.USER,
                   metadata: Optional[Dict] = None) -> User:
        """Create new user with validation"""
        if not self.db:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database not available"
            )
        
        # Validate inputs
        if not self._validate_email(email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        if len(password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        
        # Check if user exists using proper query
        existing_query = self.db.collection(self.collection).where(
            filter=FieldFilter("email", "==", email)
        )
        existing_users = existing_query.get()
        
        if existing_users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already exists"
            )
        
        # Hash password
        security = SecurityManager()
        hashed_password = security.hash_password(password)
        
        # Create user document
        user_id = secrets.token_urlsafe(32)
        user_data = {
            "id": user_id,
            "email": email,
            "password_hash": hashed_password,
            "role": role.value,
            "created_at": datetime.utcnow(),
            "is_active": True,
            "last_login": None,
            "metadata": metadata or {}
        }
        
        try:
            self.db.collection(self.collection).document(user_id).set(user_data)
            
            return User(
                id=user_id,
                email=email,
                role=role,
                is_active=True,
                created_at=user_data["created_at"],
                metadata=metadata or {}
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create user: {str(e)}"
            )
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with secure password verification"""
        if not self.db:
            return None
        
        try:
            # Secure query using proper filtering
            users_query = self.db.collection(self.collection).where(
                filter=FieldFilter("email", "==", email)
            ).limit(1)
            
            users = users_query.get()
            
            if not users:
                return None
            
            user_doc = users[0]
            user_data = user_doc.to_dict()
            
            if not user_data.get("is_active", False):
                return None
            
            # Verify password
            security = SecurityManager()
            if not security.verify_password(password, user_data["password_hash"]):
                return None
            
            # Update last login
            self.db.collection(self.collection).document(user_doc.id).update({
                "last_login": datetime.utcnow()
            })
            
            return User(
                id=user_data["id"],
                email=user_data["email"],
                role=UserRole(user_data["role"]),
                is_active=user_data["is_active"],
                created_at=user_data["created_at"],
                last_login=datetime.utcnow(),
                metadata=user_data.get("metadata", {})
            )
            
        except Exception as e:
            print(f"Authentication error: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID with proper error handling"""
        if not self.db:
            return None
        
        try:
            doc = self.db.collection(self.collection).document(user_id).get()
            
            if not doc.exists:
                return None
            
            user_data = doc.to_dict()
            
            return User(
                id=user_data["id"],
                email=user_data["email"],
                role=UserRole(user_data["role"]),
                is_active=user_data["is_active"],
                created_at=user_data["created_at"],
                last_login=user_data.get("last_login"),
                metadata=user_data.get("metadata", {})
            )
            
        except Exception as e:
            print(f"Error fetching user: {e}")
            return None
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None


class AuthenticationMiddleware:
    """Authentication middleware with comprehensive security"""
    
    def __init__(self):
        self.security_manager = SecurityManager()
        self.rate_limiter = RateLimiter(self.security_manager.redis_client)
        self.user_manager = UserManager(self.security_manager.db)
        self.rbac = RoleBasedAccessControl()
    
    def get_current_user(self, 
                        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> User:
        """Get current authenticated user"""
        token_data = self.security_manager.verify_token(credentials.credentials)
        user = self.user_manager.get_user_by_id(token_data.user_id)
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        return user
    
    def require_permission(self, permission: str, project: Optional[str] = None):
        """Require specific permission"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract user from dependencies
                user = None
                for arg in args:
                    if isinstance(arg, User):
                        user = arg
                        break
                
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                if not self.rbac.has_permission(user.role, permission, project):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def rate_limit(self, limit: int = 100, window: int = 60, burst_limit: int = None):
        """Apply rate limiting"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                request = None
                user = None
                
                # Extract request and user from arguments
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                    elif isinstance(arg, User):
                        user = arg
                
                # Use user ID or IP for rate limiting
                if user:
                    identifier = f"user:{user.id}"
                elif request:
                    identifier = f"ip:{request.client.host}"
                else:
                    identifier = "unknown"
                
                if not self.rate_limiter.is_allowed(identifier, limit, window, burst_limit):
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded"
                    )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator


# Global instances
auth_middleware = AuthenticationMiddleware()
security_manager = SecurityManager()
user_manager = UserManager(security_manager.db)
rbac = RoleBasedAccessControl()

# Common dependencies
get_current_user = auth_middleware.get_current_user
require_permission = auth_middleware.require_permission
rate_limit = auth_middleware.rate_limit 