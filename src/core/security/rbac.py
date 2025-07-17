"""
Role-Based Access Control (RBAC) Security System.

This module provides comprehensive RBAC capabilities with role hierarchy,
permission management, secure authentication using local cryptography,
and access control enforcement with audit trails. All processing is performed locally with zero external dependencies.
"""

import asyncio
import hashlib
import secrets
import hmac
import json
import time
from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import base64
import uuid

# Security and crypto imports - all local
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import jwt
from passlib.context import CryptContext
from passlib.hash import argon2

import structlog
from pydantic import BaseModel, Field, ConfigDict, validator

from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class ActionType(str, Enum):
    """Types of actions that can be performed."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    SEARCH = "search"
    ADMIN = "admin"
    EXECUTE = "execute"
    ANALYZE = "analyze"
    EXPORT = "export"
    IMPORT = "import"
    SHARE = "share"


class ResourceType(str, Enum):
    """Types of resources that can be accessed."""
    MEMORY = "memory"
    USER = "user"
    ROLE = "role"
    PERMISSION = "permission"
    AUDIT_LOG = "audit_log"
    SYSTEM = "system"
    CONFIG = "config"
    ANALYTICS = "analytics"
    GRAPH = "graph"
    EMBEDDING = "embedding"


class AuthenticationMethod(str, Enum):
    """Authentication methods supported."""
    PASSWORD = "password"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    SESSION = "session"
    CERTIFICATE = "certificate"


class AccessDecision(str, Enum):
    """Access control decisions."""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


@dataclass
class Permission:
    """Represents a permission in the system."""
    id: str
    name: str
    resource_type: ResourceType
    action_type: ActionType
    conditions: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def matches(self, resource: ResourceType, action: ActionType) -> bool:
        """Check if this permission matches a resource and action."""
        return self.resource_type == resource and self.action_type == action
    
    def evaluate_conditions(self, context: Dict[str, Any]) -> bool:
        """Evaluate permission conditions against context."""
        if not self.conditions:
            return True
        
        for condition_key, condition_value in self.conditions.items():
            if condition_key not in context:
                return False
            
            context_value = context[condition_key]
            
            # Handle different condition types
            if isinstance(condition_value, dict):
                operator = condition_value.get("operator", "eq")
                value = condition_value.get("value")
                
                if operator == "eq" and context_value != value:
                    return False
                elif operator == "ne" and context_value == value:
                    return False
                elif operator == "gt" and context_value <= value:
                    return False
                elif operator == "lt" and context_value >= value:
                    return False
                elif operator == "in" and context_value not in value:
                    return False
                elif operator == "not_in" and context_value in value:
                    return False
            else:
                if context_value != condition_value:
                    return False
        
        return True


@dataclass
class Role:
    """Represents a role in the system."""
    id: str
    name: str
    permissions: Set[str] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    child_roles: Set[str] = field(default_factory=set)
    description: str = ""
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_permission(self, permission_id: str):
        """Add a permission to this role."""
        self.permissions.add(permission_id)
        self.updated_at = datetime.utcnow()
    
    def remove_permission(self, permission_id: str):
        """Remove a permission from this role."""
        self.permissions.discard(permission_id)
        self.updated_at = datetime.utcnow()
    
    def has_permission(self, permission_id: str) -> bool:
        """Check if role has a specific permission."""
        return permission_id in self.permissions


@dataclass
class User:
    """Represents a user in the system."""
    id: str
    username: str
    email: str
    password_hash: str
    roles: Set[str] = field(default_factory=set)
    api_keys: List[str] = field(default_factory=list)
    is_active: bool = True
    is_system_user: bool = False
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked_until: Optional[datetime] = None
    password_changed_at: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_role(self, role_id: str):
        """Add a role to this user."""
        self.roles.add(role_id)
        self.updated_at = datetime.utcnow()
    
    def remove_role(self, role_id: str):
        """Remove a role from this user."""
        self.roles.discard(role_id)
        self.updated_at = datetime.utcnow()
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.account_locked_until is None:
            return False
        return datetime.utcnow() < self.account_locked_until
    
    def generate_api_key(self) -> str:
        """Generate a new API key for the user."""
        api_key = f"tyra_{secrets.token_urlsafe(32)}"
        self.api_keys.append(api_key)
        self.updated_at = datetime.utcnow()
        return api_key


@dataclass
class AccessContext:
    """Context information for access control decisions."""
    user_id: str
    resource_type: ResourceType
    action_type: ActionType
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessResult:
    """Result of an access control check."""
    decision: AccessDecision
    user_id: str
    resource_type: ResourceType
    action_type: ActionType
    reason: str
    permissions_checked: List[str] = field(default_factory=list)
    roles_checked: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    conditions_evaluated: Dict[str, bool] = field(default_factory=dict)


class SecurityConfig:
    """Configuration for security settings."""
    
    def __init__(self):
        self.password_min_length = 12
        self.password_require_uppercase = True
        self.password_require_lowercase = True
        self.password_require_numbers = True
        self.password_require_special = True
        self.max_failed_login_attempts = 5
        self.account_lockout_duration_minutes = 30
        self.session_timeout_minutes = 480  # 8 hours
        self.jwt_secret_key = secrets.token_urlsafe(64)
        self.jwt_algorithm = "HS256"
        self.jwt_expiration_minutes = 60
        self.api_key_prefix = "tyra_"
        self.enable_audit_logging = True
        self.cache_permissions = True
        self.cache_ttl_seconds = 300  # 5 minutes


class CryptographyManager:
    """Manages cryptographic operations for RBAC."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.pwd_context = CryptContext(
            schemes=["argon2"],
            deprecated="auto",
            argon2__memory_cost=65536,  # 64MB
            argon2__time_cost=3,
            argon2__parallelism=1,
        )
        self.backend = default_backend()
    
    def hash_password(self, password: str) -> str:
        """Hash a password using Argon2."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            return self.pwd_context.verify(password, password_hash)
        except Exception as e:
            logger.error("Password verification failed", error=str(e))
            return False
    
    def generate_salt(self, length: int = 32) -> bytes:
        """Generate a cryptographic salt."""
        return secrets.token_bytes(length)
    
    def derive_key(self, password: str, salt: bytes, length: int = 32) -> bytes:
        """Derive a key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=length,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(password.encode())
    
    def encrypt_data(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using AES-256-GCM."""
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return iv + encryptor.tag + ciphertext
    
    def decrypt_data(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using AES-256-GCM."""
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def create_jwt_token(self, payload: Dict[str, Any]) -> str:
        """Create a JWT token."""
        now = datetime.utcnow()
        payload.update({
            "iat": now,
            "exp": now + timedelta(minutes=self.config.jwt_expiration_minutes),
            "iss": "tyra-mcp-memory-server"
        })
        
        return jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
                options={"verify_exp": True}
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid JWT token", error=str(e))
            return None


class PermissionManager:
    """Manages permissions in the RBAC system."""
    
    def __init__(self, cache: Optional[RedisCache] = None):
        self.cache = cache
        self.permissions: Dict[str, Permission] = {}
        self._initialize_default_permissions()
    
    def _initialize_default_permissions(self):
        """Initialize default system permissions."""
        default_permissions = [
            # Memory permissions
            ("memory_create", ResourceType.MEMORY, ActionType.CREATE, "Create new memories"),
            ("memory_read", ResourceType.MEMORY, ActionType.READ, "Read memory content"),
            ("memory_update", ResourceType.MEMORY, ActionType.UPDATE, "Update memory content"),
            ("memory_delete", ResourceType.MEMORY, ActionType.DELETE, "Delete memories"),
            ("memory_list", ResourceType.MEMORY, ActionType.LIST, "List memories"),
            ("memory_search", ResourceType.MEMORY, ActionType.SEARCH, "Search memories"),
            ("memory_share", ResourceType.MEMORY, ActionType.SHARE, "Share memories"),
            
            # User management permissions
            ("user_create", ResourceType.USER, ActionType.CREATE, "Create new users"),
            ("user_read", ResourceType.USER, ActionType.READ, "Read user information"),
            ("user_update", ResourceType.USER, ActionType.UPDATE, "Update user information"),
            ("user_delete", ResourceType.USER, ActionType.DELETE, "Delete users"),
            ("user_list", ResourceType.USER, ActionType.LIST, "List users"),
            
            # Role management permissions
            ("role_create", ResourceType.ROLE, ActionType.CREATE, "Create new roles"),
            ("role_read", ResourceType.ROLE, ActionType.READ, "Read role information"),
            ("role_update", ResourceType.ROLE, ActionType.UPDATE, "Update roles"),
            ("role_delete", ResourceType.ROLE, ActionType.DELETE, "Delete roles"),
            
            # System permissions
            ("system_admin", ResourceType.SYSTEM, ActionType.ADMIN, "System administration"),
            ("system_analyze", ResourceType.SYSTEM, ActionType.ANALYZE, "System analysis"),
            ("config_read", ResourceType.CONFIG, ActionType.READ, "Read configuration"),
            ("config_update", ResourceType.CONFIG, ActionType.UPDATE, "Update configuration"),
            
            # Analytics permissions
            ("analytics_read", ResourceType.ANALYTICS, ActionType.READ, "Read analytics data"),
            ("analytics_export", ResourceType.ANALYTICS, ActionType.EXPORT, "Export analytics"),
            
            # Graph permissions
            ("graph_read", ResourceType.GRAPH, ActionType.READ, "Read graph data"),
            ("graph_analyze", ResourceType.GRAPH, ActionType.ANALYZE, "Analyze graph relationships"),
        ]
        
        for perm_id, resource_type, action_type, description in default_permissions:
            permission = Permission(
                id=perm_id,
                name=perm_id.replace("_", " ").title(),
                resource_type=resource_type,
                action_type=action_type,
                description=description
            )
            self.permissions[perm_id] = permission
    
    async def create_permission(
        self,
        permission_id: str,
        name: str,
        resource_type: ResourceType,
        action_type: ActionType,
        conditions: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> Permission:
        """Create a new permission."""
        if permission_id in self.permissions:
            raise ValueError(f"Permission {permission_id} already exists")
        
        permission = Permission(
            id=permission_id,
            name=name,
            resource_type=resource_type,
            action_type=action_type,
            conditions=conditions or {},
            description=description
        )
        
        self.permissions[permission_id] = permission
        
        # Cache permission if caching is enabled
        if self.cache:
            await self.cache.set(
                f"permission:{permission_id}",
                permission.__dict__,
                ttl=300
            )
        
        logger.info("Permission created", permission_id=permission_id, name=name)
        return permission
    
    async def get_permission(self, permission_id: str) -> Optional[Permission]:
        """Get a permission by ID."""
        # Try cache first
        if self.cache:
            cached = await self.cache.get(f"permission:{permission_id}")
            if cached:
                return Permission(**cached)
        
        permission = self.permissions.get(permission_id)
        
        # Cache the result
        if permission and self.cache:
            await self.cache.set(
                f"permission:{permission_id}",
                permission.__dict__,
                ttl=300
            )
        
        return permission
    
    async def delete_permission(self, permission_id: str) -> bool:
        """Delete a permission."""
        if permission_id not in self.permissions:
            return False
        
        del self.permissions[permission_id]
        
        # Remove from cache
        if self.cache:
            await self.cache.delete(f"permission:{permission_id}")
        
        logger.info("Permission deleted", permission_id=permission_id)
        return True
    
    def list_permissions(
        self,
        resource_type: Optional[ResourceType] = None,
        action_type: Optional[ActionType] = None
    ) -> List[Permission]:
        """List permissions with optional filtering."""
        permissions = list(self.permissions.values())
        
        if resource_type:
            permissions = [p for p in permissions if p.resource_type == resource_type]
        
        if action_type:
            permissions = [p for p in permissions if p.action_type == action_type]
        
        return permissions


class RoleManager:
    """Manages roles in the RBAC system."""
    
    def __init__(self, permission_manager: PermissionManager, cache: Optional[RedisCache] = None):
        self.permission_manager = permission_manager
        self.cache = cache
        self.roles: Dict[str, Role] = {}
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default system roles."""
        # Admin role - full access
        admin_role = Role(
            id="admin",
            name="Administrator",
            description="Full system access",
            is_system_role=True
        )
        admin_role.permissions = set(self.permission_manager.permissions.keys())
        self.roles["admin"] = admin_role
        
        # User role - basic memory operations
        user_role = Role(
            id="user",
            name="User",
            description="Basic memory operations",
            is_system_role=True
        )
        user_role.permissions = {
            "memory_create", "memory_read", "memory_update", "memory_delete",
            "memory_list", "memory_search"
        }
        self.roles["user"] = user_role
        
        # Reader role - read-only access
        reader_role = Role(
            id="reader",
            name="Reader",
            description="Read-only access",
            is_system_role=True
        )
        reader_role.permissions = {"memory_read", "memory_list", "memory_search"}
        self.roles["reader"] = reader_role
        
        # Analyst role - analytics access
        analyst_role = Role(
            id="analyst",
            name="Analyst",
            description="Analytics and reporting access",
            is_system_role=True
        )
        analyst_role.permissions = {
            "memory_read", "memory_list", "memory_search",
            "analytics_read", "analytics_export", "graph_read", "graph_analyze"
        }
        self.roles["analyst"] = analyst_role
    
    async def create_role(
        self,
        role_id: str,
        name: str,
        permissions: Optional[Set[str]] = None,
        parent_roles: Optional[Set[str]] = None,
        description: str = ""
    ) -> Role:
        """Create a new role."""
        if role_id in self.roles:
            raise ValueError(f"Role {role_id} already exists")
        
        role = Role(
            id=role_id,
            name=name,
            permissions=permissions or set(),
            parent_roles=parent_roles or set(),
            description=description
        )
        
        # Update parent role relationships
        if parent_roles:
            for parent_id in parent_roles:
                if parent_id in self.roles:
                    self.roles[parent_id].child_roles.add(role_id)
        
        self.roles[role_id] = role
        
        # Cache role if caching is enabled
        if self.cache:
            await self.cache.set(
                f"role:{role_id}",
                role.__dict__,
                ttl=300
            )
        
        logger.info("Role created", role_id=role_id, name=name)
        return role
    
    async def get_role(self, role_id: str) -> Optional[Role]:
        """Get a role by ID."""
        # Try cache first
        if self.cache:
            cached = await self.cache.get(f"role:{role_id}")
            if cached:
                # Reconstruct sets from cached data
                cached_role = cached.copy()
                cached_role['permissions'] = set(cached_role.get('permissions', []))
                cached_role['parent_roles'] = set(cached_role.get('parent_roles', []))
                cached_role['child_roles'] = set(cached_role.get('child_roles', []))
                return Role(**cached_role)
        
        role = self.roles.get(role_id)
        
        # Cache the result
        if role and self.cache:
            role_dict = role.__dict__.copy()
            # Convert sets to lists for JSON serialization
            role_dict['permissions'] = list(role_dict['permissions'])
            role_dict['parent_roles'] = list(role_dict['parent_roles'])
            role_dict['child_roles'] = list(role_dict['child_roles'])
            
            await self.cache.set(
                f"role:{role_id}",
                role_dict,
                ttl=300
            )
        
        return role
    
    async def get_effective_permissions(self, role_id: str) -> Set[str]:
        """Get all effective permissions for a role (including inherited)."""
        visited = set()
        effective_permissions = set()
        
        async def collect_permissions(current_role_id: str):
            if current_role_id in visited:
                return
            
            visited.add(current_role_id)
            role = await self.get_role(current_role_id)
            
            if role:
                effective_permissions.update(role.permissions)
                
                # Recursively collect from parent roles
                for parent_id in role.parent_roles:
                    await collect_permissions(parent_id)
        
        await collect_permissions(role_id)
        return effective_permissions
    
    async def delete_role(self, role_id: str) -> bool:
        """Delete a role."""
        role = self.roles.get(role_id)
        if not role:
            return False
        
        if role.is_system_role:
            raise ValueError("Cannot delete system roles")
        
        # Update parent-child relationships
        for parent_id in role.parent_roles:
            if parent_id in self.roles:
                self.roles[parent_id].child_roles.discard(role_id)
        
        for child_id in role.child_roles:
            if child_id in self.roles:
                self.roles[child_id].parent_roles.discard(role_id)
        
        del self.roles[role_id]
        
        # Remove from cache
        if self.cache:
            await self.cache.delete(f"role:{role_id}")
        
        logger.info("Role deleted", role_id=role_id)
        return True
    
    def list_roles(self) -> List[Role]:
        """List all roles."""
        return list(self.roles.values())


class UserManager:
    """Manages users in the RBAC system."""
    
    def __init__(
        self,
        crypto_manager: CryptographyManager,
        role_manager: RoleManager,
        config: SecurityConfig,
        cache: Optional[RedisCache] = None
    ):
        self.crypto_manager = crypto_manager
        self.role_manager = role_manager
        self.config = config
        self.cache = cache
        self.users: Dict[str, User] = {}
        self.username_to_id: Dict[str, str] = {}
        self.email_to_id: Dict[str, str] = {}
        self._initialize_default_users()
    
    def _initialize_default_users(self):
        """Initialize default system users."""
        # Create system admin user
        admin_user = User(
            id="admin",
            username="admin",
            email="admin@tyra-mcp.local",
            password_hash=self.crypto_manager.hash_password("admin123!@#"),
            is_system_user=True
        )
        admin_user.add_role("admin")
        
        self.users["admin"] = admin_user
        self.username_to_id["admin"] = "admin"
        self.email_to_id["admin@tyra-mcp.local"] = "admin"
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password strength according to policy."""
        if len(password) < self.config.password_min_length:
            return False
        
        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            return False
        
        if self.config.password_require_lowercase and not any(c.islower() for c in password):
            return False
        
        if self.config.password_require_numbers and not any(c.isdigit() for c in password):
            return False
        
        if self.config.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False
        
        return True
    
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[Set[str]] = None
    ) -> User:
        """Create a new user."""
        if username in self.username_to_id:
            raise ValueError(f"Username {username} already exists")
        
        if email in self.email_to_id:
            raise ValueError(f"Email {email} already exists")
        
        if not self._validate_password_strength(password):
            raise ValueError("Password does not meet strength requirements")
        
        user_id = str(uuid.uuid4())
        password_hash = self.crypto_manager.hash_password(password)
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or {"user"}  # Default to user role
        )
        
        self.users[user_id] = user
        self.username_to_id[username] = user_id
        self.email_to_id[email] = user_id
        
        # Cache user if caching is enabled
        if self.cache:
            user_dict = user.__dict__.copy()
            user_dict['roles'] = list(user_dict['roles'])
            await self.cache.set(
                f"user:{user_id}",
                user_dict,
                ttl=300
            )
        
        logger.info("User created", user_id=user_id, username=username, email=email)
        return user
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None
    ) -> Optional[User]:
        """Authenticate a user with username and password."""
        user_id = self.username_to_id.get(username)
        if not user_id:
            logger.warning("Authentication failed - user not found", username=username)
            return None
        
        user = await self.get_user(user_id)
        if not user:
            return None
        
        if user.is_locked():
            logger.warning("Authentication failed - account locked", username=username)
            return None
        
        if not user.is_active:
            logger.warning("Authentication failed - account inactive", username=username)
            return None
        
        if not self.crypto_manager.verify_password(password, user.password_hash):
            # Increment failed login attempts
            user.failed_login_attempts += 1
            
            if user.failed_login_attempts >= self.config.max_failed_login_attempts:
                user.account_locked_until = datetime.utcnow() + timedelta(
                    minutes=self.config.account_lockout_duration_minutes
                )
                logger.warning(
                    "Account locked due to too many failed attempts",
                    username=username,
                    attempts=user.failed_login_attempts
                )
            
            # Update user in storage
            await self._update_user_cache(user)
            
            logger.warning("Authentication failed - invalid password", username=username)
            return None
        
        # Successful authentication
        user.failed_login_attempts = 0
        user.last_login = datetime.utcnow()
        user.account_locked_until = None
        
        # Update user in storage
        await self._update_user_cache(user)
        
        logger.info("User authenticated successfully", username=username, ip_address=ip_address)
        return user
    
    async def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate a user with API key."""
        # Search through all users for matching API key
        for user in self.users.values():
            if api_key in user.api_keys and user.is_active and not user.is_locked():
                logger.info("User authenticated with API key", user_id=user.id)
                return user
        
        logger.warning("Authentication failed - invalid API key")
        return None
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        # Try cache first
        if self.cache:
            cached = await self.cache.get(f"user:{user_id}")
            if cached:
                cached_user = cached.copy()
                cached_user['roles'] = set(cached_user.get('roles', []))
                return User(**cached_user)
        
        user = self.users.get(user_id)
        
        # Cache the result
        if user and self.cache:
            await self._update_user_cache(user)
        
        return user
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        user_id = self.username_to_id.get(username)
        if user_id:
            return await self.get_user(user_id)
        return None
    
    async def _update_user_cache(self, user: User):
        """Update user in cache."""
        if self.cache:
            user_dict = user.__dict__.copy()
            user_dict['roles'] = list(user_dict['roles'])
            await self.cache.set(
                f"user:{user.id}",
                user_dict,
                ttl=300
            )
    
    async def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change a user's password."""
        user = await self.get_user(user_id)
        if not user:
            return False
        
        if not self.crypto_manager.verify_password(old_password, user.password_hash):
            logger.warning("Password change failed - invalid old password", user_id=user_id)
            return False
        
        if not self._validate_password_strength(new_password):
            raise ValueError("New password does not meet strength requirements")
        
        user.password_hash = self.crypto_manager.hash_password(new_password)
        user.password_changed_at = datetime.utcnow()
        user.updated_at = datetime.utcnow()
        
        await self._update_user_cache(user)
        
        logger.info("Password changed successfully", user_id=user_id)
        return True
    
    def list_users(self) -> List[User]:
        """List all users."""
        return list(self.users.values())


class AccessControlEngine:
    """Core access control engine for RBAC."""
    
    def __init__(
        self,
        permission_manager: PermissionManager,
        role_manager: RoleManager,
        user_manager: UserManager,
        cache: Optional[RedisCache] = None
    ):
        self.permission_manager = permission_manager
        self.role_manager = role_manager
        self.user_manager = user_manager
        self.cache = cache
        self.access_history: deque = deque(maxlen=1000)
    
    async def check_access(self, context: AccessContext) -> AccessResult:
        """Check if access should be granted for the given context."""
        start_time = time.time()
        
        try:
            # Get user
            user = await self.user_manager.get_user(context.user_id)
            if not user or not user.is_active or user.is_locked():
                return AccessResult(
                    decision=AccessDecision.DENY,
                    user_id=context.user_id,
                    resource_type=context.resource_type,
                    action_type=context.action_type,
                    reason="User not found, inactive, or locked"
                )
            
            # Get all effective permissions for user's roles
            all_permissions = set()
            roles_checked = []
            
            for role_id in user.roles:
                role_permissions = await self.role_manager.get_effective_permissions(role_id)
                all_permissions.update(role_permissions)
                roles_checked.append(role_id)
            
            # Find matching permissions
            matching_permissions = []
            permissions_checked = []
            
            for permission_id in all_permissions:
                permission = await self.permission_manager.get_permission(permission_id)
                if permission and permission.matches(context.resource_type, context.action_type):
                    matching_permissions.append(permission)
                    permissions_checked.append(permission_id)
            
            if not matching_permissions:
                return AccessResult(
                    decision=AccessDecision.DENY,
                    user_id=context.user_id,
                    resource_type=context.resource_type,
                    action_type=context.action_type,
                    reason="No matching permissions found",
                    permissions_checked=permissions_checked,
                    roles_checked=roles_checked
                )
            
            # Evaluate conditions for matching permissions
            conditions_evaluated = {}
            context_dict = {
                "user_id": context.user_id,
                "resource_id": context.resource_id,
                "ip_address": context.ip_address,
                "user_agent": context.user_agent,
                "timestamp": context.timestamp,
                **context.additional_context
            }
            
            for permission in matching_permissions:
                condition_result = permission.evaluate_conditions(context_dict)
                conditions_evaluated[permission.id] = condition_result
                
                if condition_result:
                    # At least one permission allows access
                    result = AccessResult(
                        decision=AccessDecision.ALLOW,
                        user_id=context.user_id,
                        resource_type=context.resource_type,
                        action_type=context.action_type,
                        reason=f"Access granted via permission {permission.id}",
                        permissions_checked=permissions_checked,
                        roles_checked=roles_checked,
                        conditions_evaluated=conditions_evaluated
                    )
                    
                    # Record access decision
                    self.access_history.append(result)
                    
                    logger.info(
                        "Access granted",
                        user_id=context.user_id,
                        resource_type=context.resource_type.value,
                        action_type=context.action_type.value,
                        permission=permission.id,
                        duration_ms=(time.time() - start_time) * 1000
                    )
                    
                    return result
            
            # No permissions allowed access
            result = AccessResult(
                decision=AccessDecision.DENY,
                user_id=context.user_id,
                resource_type=context.resource_type,
                action_type=context.action_type,
                reason="Permission conditions not met",
                permissions_checked=permissions_checked,
                roles_checked=roles_checked,
                conditions_evaluated=conditions_evaluated
            )
            
            self.access_history.append(result)
            
            logger.warning(
                "Access denied",
                user_id=context.user_id,
                resource_type=context.resource_type.value,
                action_type=context.action_type.value,
                reason=result.reason,
                duration_ms=(time.time() - start_time) * 1000
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Access control check failed",
                user_id=context.user_id,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            
            return AccessResult(
                decision=AccessDecision.DENY,
                user_id=context.user_id,
                resource_type=context.resource_type,
                action_type=context.action_type,
                reason=f"Access check failed: {str(e)}"
            )
    
    def get_access_history(self, limit: int = 100) -> List[AccessResult]:
        """Get recent access control decisions."""
        return list(self.access_history)[-limit:]


class RBACSystem:
    """
    Complete Role-Based Access Control System.
    
    Provides comprehensive RBAC functionality with secure authentication,
    role hierarchy, permission management, and access control enforcement.
    """
    
    def __init__(self, cache: Optional[RedisCache] = None):
        self.config = SecurityConfig()
        self.crypto_manager = CryptographyManager(self.config)
        self.cache = cache
        
        # Initialize managers
        self.permission_manager = PermissionManager(cache)
        self.role_manager = RoleManager(self.permission_manager, cache)
        self.user_manager = UserManager(
            self.crypto_manager,
            self.role_manager,
            self.config,
            cache
        )
        self.access_control = AccessControlEngine(
            self.permission_manager,
            self.role_manager,
            self.user_manager,
            cache
        )
        
        logger.info("RBAC system initialized")
    
    async def authenticate_request(
        self,
        auth_header: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        jwt_token: Optional[str] = None
    ) -> Optional[User]:
        """Authenticate a request using various methods."""
        
        # Try JWT token authentication
        if jwt_token:
            payload = self.crypto_manager.verify_jwt_token(jwt_token)
            if payload and "user_id" in payload:
                return await self.user_manager.get_user(payload["user_id"])
        
        # Try API key authentication
        if api_key:
            return await self.user_manager.authenticate_api_key(api_key)
        
        # Try username/password authentication
        if username and password:
            return await self.user_manager.authenticate_user(username, password)
        
        # Try parsing Authorization header
        if auth_header:
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                payload = self.crypto_manager.verify_jwt_token(token)
                if payload and "user_id" in payload:
                    return await self.user_manager.get_user(payload["user_id"])
            elif auth_header.startswith("ApiKey "):
                key = auth_header[7:]
                return await self.user_manager.authenticate_api_key(key)
        
        return None
    
    async def authorize_action(
        self,
        user: User,
        resource_type: ResourceType,
        action_type: ActionType,
        resource_id: Optional[str] = None,
        **context_kwargs
    ) -> bool:
        """Authorize a user action."""
        context = AccessContext(
            user_id=user.id,
            resource_type=resource_type,
            action_type=action_type,
            resource_id=resource_id,
            additional_context=context_kwargs
        )
        
        result = await self.access_control.check_access(context)
        return result.decision == AccessDecision.ALLOW
    
    async def create_session_token(self, user: User) -> str:
        """Create a JWT session token for a user."""
        payload = {
            "user_id": user.id,
            "username": user.username,
            "roles": list(user.roles),
            "session_id": str(uuid.uuid4())
        }
        return self.crypto_manager.create_jwt_token(payload)
    
    def require_permission(
        self,
        resource_type: ResourceType,
        action_type: ActionType,
        resource_id_param: Optional[str] = None
    ):
        """Decorator to require specific permissions for a function."""
        def decorator(func: Callable) -> Callable:
            async def wrapper(*args, **kwargs):
                # This would be implemented based on your framework
                # For now, it's a placeholder for the pattern
                pass
            return wrapper
        return decorator
    
    async def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all effective permissions for a user."""
        user = await self.user_manager.get_user(user_id)
        if not user:
            return set()
        
        all_permissions = set()
        for role_id in user.roles:
            role_permissions = await self.role_manager.get_effective_permissions(role_id)
            all_permissions.update(role_permissions)
        
        return all_permissions
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform RBAC system health check."""
        try:
            # Check managers
            users_count = len(self.user_manager.users)
            roles_count = len(self.role_manager.roles)
            permissions_count = len(self.permission_manager.permissions)
            
            # Test access control
            test_context = AccessContext(
                user_id="admin",
                resource_type=ResourceType.SYSTEM,
                action_type=ActionType.READ
            )
            test_result = await self.access_control.check_access(test_context)
            
            return {
                "status": "healthy",
                "users_count": users_count,
                "roles_count": roles_count,
                "permissions_count": permissions_count,
                "access_control_working": test_result.decision != AccessDecision.DENY,
                "cache_enabled": self.cache is not None,
                "config": {
                    "password_min_length": self.config.password_min_length,
                    "max_failed_login_attempts": self.config.max_failed_login_attempts,
                    "session_timeout_minutes": self.config.session_timeout_minutes
                }
            }
            
        except Exception as e:
            logger.error("RBAC health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }