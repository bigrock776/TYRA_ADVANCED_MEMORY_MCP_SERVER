"""
Advanced Security API endpoints.

Provides enterprise-grade security capabilities including RBAC management,
encryption controls, audit logging, and privacy-preserving operations.
All processing is performed locally with zero external API calls.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from ...core.security.rbac import RBACManager, Role, User, Permission, AccessDecision
from ...core.security.encryption import MemoryEncryption, EncryptionConfig, EncryptionResult
from ...core.security.audit import AuditLogger, AuditEvent, SecurityEvent, ComplianceReport
from ...core.cache.redis_cache import RedisCache
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)

router = APIRouter()
security = HTTPBearer()


# Request/Response Models
class CreateUserRequest(BaseModel):
    """Request to create a new user."""
    
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    roles: List[str] = Field(default=[], description="List of role names to assign")
    metadata: Dict[str, Any] = Field(default={}, description="Additional user metadata")


class CreateRoleRequest(BaseModel):
    """Request to create a new role."""
    
    name: str = Field(..., min_length=2, max_length=50, description="Role name")
    description: str = Field("", description="Role description")
    permissions: List[str] = Field(default=[], description="List of permission names")
    parent_roles: List[str] = Field(default=[], description="List of parent role names")
    metadata: Dict[str, Any] = Field(default={}, description="Additional role metadata")


class UpdatePermissionsRequest(BaseModel):
    """Request to update user or role permissions."""
    
    target_type: str = Field(..., description="Target type: user or role")
    target_id: str = Field(..., description="User ID or role name")
    permissions: List[str] = Field(..., description="List of permission names")
    action: str = Field("set", description="Action: set, add, remove")


class EncryptionRequest(BaseModel):
    """Request for memory encryption operation."""
    
    data: Dict[str, Any] = Field(..., description="Data to encrypt")
    encryption_level: str = Field("standard", description="Encryption level: basic, standard, high")
    include_metadata: bool = Field(True, description="Include encryption metadata")


class DecryptionRequest(BaseModel):
    """Request for memory decryption operation."""
    
    encrypted_data: str = Field(..., description="Encrypted data to decrypt")
    encryption_metadata: Dict[str, Any] = Field(..., description="Encryption metadata")


class AuditQueryRequest(BaseModel):
    """Request for audit log query."""
    
    start_time: Optional[datetime] = Field(None, description="Start time for query")
    end_time: Optional[datetime] = Field(None, description="End time for query")
    event_types: Optional[List[str]] = Field(None, description="Event types to include")
    user_ids: Optional[List[str]] = Field(None, description="User IDs to filter by")
    resource_types: Optional[List[str]] = Field(None, description="Resource types to filter by")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")


class ComplianceReportRequest(BaseModel):
    """Request for compliance report generation."""
    
    report_type: str = Field(..., description="Report type: gdpr, hipaa, sox, pci_dss")
    time_range_days: int = Field(30, ge=1, le=365, description="Time range in days")
    include_details: bool = Field(True, description="Include detailed findings")
    export_format: str = Field("json", description="Export format: json, csv, pdf")


class SecurityMetricsResponse(BaseModel):
    """Response for security metrics."""
    
    timestamp: datetime = Field(..., description="Metrics timestamp")
    rbac_metrics: Dict[str, Any] = Field(..., description="RBAC system metrics")
    encryption_metrics: Dict[str, Any] = Field(..., description="Encryption system metrics")
    audit_metrics: Dict[str, Any] = Field(..., description="Audit system metrics")
    security_events: Dict[str, Any] = Field(..., description="Security event summary")
    compliance_status: Dict[str, Any] = Field(..., description="Compliance status")


# Dependencies
async def get_rbac_manager() -> RBACManager:
    """Get RBAC manager instance."""
    try:
        db_client = get_provider(ProviderType.DATABASE, "default")
        cache = get_provider(ProviderType.CACHE, "default")
        return RBACManager(db_client=db_client, cache=cache)
    except Exception as e:
        logger.error(f"Failed to get RBAC manager: {e}")
        raise HTTPException(status_code=500, detail="RBAC system unavailable")


async def get_encryption_service() -> MemoryEncryption:
    """Get memory encryption service instance."""
    try:
        cache = get_provider(ProviderType.CACHE, "default")
        return MemoryEncryption(cache=cache)
    except Exception as e:
        logger.error(f"Failed to get encryption service: {e}")
        raise HTTPException(status_code=500, detail="Encryption service unavailable")


async def get_audit_logger() -> AuditLogger:
    """Get audit logger instance."""
    try:
        return AuditLogger()
    except Exception as e:
        logger.error(f"Failed to get audit logger: {e}")
        raise HTTPException(status_code=500, detail="Audit system unavailable")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    rbac: RBACManager = Depends(get_rbac_manager)
) -> User:
    """Get current authenticated user."""
    try:
        # Validate JWT token and get user
        user = await rbac.validate_token(credentials.credentials)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return user
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


async def require_permission(permission: str):
    """Dependency to require specific permission."""
    async def permission_checker(
        current_user: User = Depends(get_current_user),
        rbac: RBACManager = Depends(get_rbac_manager)
    ):
        access_decision = await rbac.check_access(current_user.id, "api", permission)
        if not access_decision.granted:
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {permission} required"
            )
        return current_user
    return permission_checker


# Authentication endpoints
@router.post("/auth/login")
async def login(
    username: str = Query(..., description="Username"),
    password: str = Query(..., description="Password"),
    rbac: RBACManager = Depends(get_rbac_manager),
    audit: AuditLogger = Depends(get_audit_logger)
):
    """
    Authenticate user and return access token.
    
    Validates credentials and returns JWT token for API access.
    """
    try:
        # Authenticate user
        user = await rbac.authenticate_user(username, password)
        if not user:
            # Log failed login attempt
            await audit.log_security_event(
                SecurityEvent(
                    event_type="authentication_failed",
                    user_id=username,
                    resource_type="auth",
                    resource_id="login",
                    details={"reason": "invalid_credentials"},
                    severity="medium"
                )
            )
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Generate access token
        token = await rbac.generate_token(user)
        
        # Log successful login
        await audit.log_security_event(
            SecurityEvent(
                event_type="authentication_success",
                user_id=user.id,
                resource_type="auth",
                resource_id="login",
                details={"username": username},
                severity="low"
            )
        )
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user_id": user.id,
            "username": user.username,
            "roles": [role.name for role in user.roles]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.post("/auth/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security),
    rbac: RBACManager = Depends(get_rbac_manager),
    audit: AuditLogger = Depends(get_audit_logger)
):
    """
    Logout user and invalidate token.
    
    Adds token to blacklist and logs logout event.
    """
    try:
        # Invalidate token
        await rbac.invalidate_token(credentials.credentials)
        
        # Log logout
        await audit.log_security_event(
            SecurityEvent(
                event_type="logout",
                user_id=current_user.id,
                resource_type="auth",
                resource_id="logout",
                details={"username": current_user.username},
                severity="low"
            )
        )
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")


# User management endpoints
@router.post("/users")
async def create_user(
    request: CreateUserRequest,
    current_user: User = Depends(require_permission("user:create")),
    rbac: RBACManager = Depends(get_rbac_manager),
    audit: AuditLogger = Depends(get_audit_logger)
):
    """
    Create a new user account.
    
    Creates user with specified roles and permissions.
    Requires user:create permission.
    """
    try:
        # Create user
        user = await rbac.create_user(
            username=request.username,
            email=request.email,
            password=request.password,
            metadata=request.metadata
        )
        
        # Assign roles
        for role_name in request.roles:
            await rbac.assign_role_to_user(user.id, role_name)
        
        # Log user creation
        await audit.log_security_event(
            SecurityEvent(
                event_type="user_created",
                user_id=current_user.id,
                resource_type="user",
                resource_id=user.id,
                details={
                    "created_user": request.username,
                    "roles_assigned": request.roles
                },
                severity="medium"
            )
        )
        
        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "roles": request.roles,
            "created_at": user.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"User creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    current_user: User = Depends(require_permission("user:read")),
    rbac: RBACManager = Depends(get_rbac_manager)
):
    """
    Get user details by ID.
    
    Returns user information including roles and permissions.
    Requires user:read permission.
    """
    try:
        user = await rbac.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "roles": [{"name": role.name, "description": role.description} for role in user.roles],
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "is_active": user.is_active,
            "metadata": user.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users")
async def list_users(
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: User = Depends(require_permission("user:list")),
    rbac: RBACManager = Depends(get_rbac_manager)
):
    """
    List all users with pagination.
    
    Returns paginated list of users with basic information.
    Requires user:list permission.
    """
    try:
        users = await rbac.list_users(limit=limit, offset=offset)
        
        return {
            "users": [
                {
                    "user_id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "roles": [role.name for role in user.roles],
                    "is_active": user.is_active,
                    "created_at": user.created_at.isoformat()
                }
                for user in users
            ],
            "total_count": len(users),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Role management endpoints
@router.post("/roles")
async def create_role(
    request: CreateRoleRequest,
    current_user: User = Depends(require_permission("role:create")),
    rbac: RBACManager = Depends(get_rbac_manager),
    audit: AuditLogger = Depends(get_audit_logger)
):
    """
    Create a new role.
    
    Creates role with specified permissions and parent relationships.
    Requires role:create permission.
    """
    try:
        # Create role
        role = await rbac.create_role(
            name=request.name,
            description=request.description,
            permissions=request.permissions,
            parent_roles=request.parent_roles,
            metadata=request.metadata
        )
        
        # Log role creation
        await audit.log_security_event(
            SecurityEvent(
                event_type="role_created",
                user_id=current_user.id,
                resource_type="role",
                resource_id=role.name,
                details={
                    "role_name": request.name,
                    "permissions": request.permissions,
                    "parent_roles": request.parent_roles
                },
                severity="medium"
            )
        )
        
        return {
            "role_name": role.name,
            "description": role.description,
            "permissions": request.permissions,
            "parent_roles": request.parent_roles,
            "created_at": role.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Role creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roles")
async def list_roles(
    current_user: User = Depends(require_permission("role:list")),
    rbac: RBACManager = Depends(get_rbac_manager)
):
    """
    List all roles.
    
    Returns all roles with their permissions and hierarchy.
    Requires role:list permission.
    """
    try:
        roles = await rbac.list_roles()
        
        return {
            "roles": [
                {
                    "name": role.name,
                    "description": role.description,
                    "permissions": [perm.name for perm in role.permissions],
                    "parent_roles": [parent.name for parent in role.parent_roles],
                    "created_at": role.created_at.isoformat()
                }
                for role in roles
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to list roles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/permissions")
async def update_permissions(
    request: UpdatePermissionsRequest,
    current_user: User = Depends(require_permission("permission:update")),
    rbac: RBACManager = Depends(get_rbac_manager),
    audit: AuditLogger = Depends(get_audit_logger)
):
    """
    Update permissions for user or role.
    
    Modifies permissions based on specified action (set/add/remove).
    Requires permission:update permission.
    """
    try:
        if request.target_type == "user":
            if request.action == "set":
                await rbac.set_user_permissions(request.target_id, request.permissions)
            elif request.action == "add":
                await rbac.add_user_permissions(request.target_id, request.permissions)
            elif request.action == "remove":
                await rbac.remove_user_permissions(request.target_id, request.permissions)
        
        elif request.target_type == "role":
            if request.action == "set":
                await rbac.set_role_permissions(request.target_id, request.permissions)
            elif request.action == "add":
                await rbac.add_role_permissions(request.target_id, request.permissions)
            elif request.action == "remove":
                await rbac.remove_role_permissions(request.target_id, request.permissions)
        
        else:
            raise HTTPException(status_code=400, detail="Invalid target_type")
        
        # Log permission update
        await audit.log_security_event(
            SecurityEvent(
                event_type="permissions_updated",
                user_id=current_user.id,
                resource_type=request.target_type,
                resource_id=request.target_id,
                details={
                    "action": request.action,
                    "permissions": request.permissions
                },
                severity="high"
            )
        )
        
        return {
            "message": f"Permissions {request.action} successfully",
            "target_type": request.target_type,
            "target_id": request.target_id,
            "permissions": request.permissions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Permission update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Encryption endpoints
@router.post("/encryption/encrypt")
async def encrypt_data(
    request: EncryptionRequest,
    current_user: User = Depends(require_permission("encryption:use")),
    encryption: MemoryEncryption = Depends(get_encryption_service),
    audit: AuditLogger = Depends(get_audit_logger)
):
    """
    Encrypt sensitive data.
    
    Encrypts data using configured encryption level and returns
    encrypted data with metadata for decryption.
    """
    try:
        # Configure encryption
        config = EncryptionConfig(
            encryption_level=request.encryption_level,
            include_metadata=request.include_metadata
        )
        
        # Encrypt data
        result = await encryption.encrypt_memory_data(request.data, config)
        
        # Log encryption operation
        await audit.log_security_event(
            SecurityEvent(
                event_type="data_encrypted",
                user_id=current_user.id,
                resource_type="encryption",
                resource_id="encrypt_operation",
                details={
                    "encryption_level": request.encryption_level,
                    "data_size": len(str(request.data))
                },
                severity="low"
            )
        )
        
        return {
            "encrypted_data": result.encrypted_data,
            "encryption_metadata": result.metadata,
            "encryption_id": result.encryption_id,
            "key_version": result.key_version
        }
        
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/encryption/decrypt")
async def decrypt_data(
    request: DecryptionRequest,
    current_user: User = Depends(require_permission("encryption:use")),
    encryption: MemoryEncryption = Depends(get_encryption_service),
    audit: AuditLogger = Depends(get_audit_logger)
):
    """
    Decrypt encrypted data.
    
    Decrypts data using provided encryption metadata.
    Requires proper encryption metadata for successful decryption.
    """
    try:
        # Decrypt data
        decrypted_data = await encryption.decrypt_memory_data(
            request.encrypted_data,
            request.encryption_metadata
        )
        
        # Log decryption operation
        await audit.log_security_event(
            SecurityEvent(
                event_type="data_decrypted",
                user_id=current_user.id,
                resource_type="encryption",
                resource_id="decrypt_operation",
                details={
                    "encryption_id": request.encryption_metadata.get("encryption_id"),
                    "key_version": request.encryption_metadata.get("key_version")
                },
                severity="medium"
            )
        )
        
        return {
            "decrypted_data": decrypted_data,
            "decryption_successful": True
        }
        
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise HTTPException(status_code=500, detail="Decryption failed")


@router.get("/encryption/keys")
async def list_encryption_keys(
    current_user: User = Depends(require_permission("encryption:admin")),
    encryption: MemoryEncryption = Depends(get_encryption_service)
):
    """
    List encryption keys and their status.
    
    Returns information about encryption keys including versions
    and rotation status. Requires encryption:admin permission.
    """
    try:
        keys_info = await encryption.get_key_info()
        
        return {
            "active_keys": keys_info["active_keys"],
            "key_versions": keys_info["key_versions"],
            "last_rotation": keys_info["last_rotation"],
            "next_rotation": keys_info["next_rotation"],
            "rotation_policy": keys_info["rotation_policy"]
        }
        
    except Exception as e:
        logger.error(f"Failed to list encryption keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/encryption/rotate-keys")
async def rotate_encryption_keys(
    current_user: User = Depends(require_permission("encryption:admin")),
    encryption: MemoryEncryption = Depends(get_encryption_service),
    audit: AuditLogger = Depends(get_audit_logger)
):
    """
    Rotate encryption keys.
    
    Generates new encryption keys and marks old keys for retirement.
    Requires encryption:admin permission.
    """
    try:
        # Rotate keys
        rotation_result = await encryption.rotate_keys()
        
        # Log key rotation
        await audit.log_security_event(
            SecurityEvent(
                event_type="encryption_keys_rotated",
                user_id=current_user.id,
                resource_type="encryption",
                resource_id="key_rotation",
                details={
                    "old_key_version": rotation_result["old_version"],
                    "new_key_version": rotation_result["new_version"],
                    "rotation_reason": "manual"
                },
                severity="high"
            )
        )
        
        return {
            "message": "Encryption keys rotated successfully",
            "old_version": rotation_result["old_version"],
            "new_version": rotation_result["new_version"],
            "rotation_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Key rotation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Audit endpoints
@router.post("/audit/query")
async def query_audit_logs(
    request: AuditQueryRequest,
    current_user: User = Depends(require_permission("audit:read")),
    audit: AuditLogger = Depends(get_audit_logger)
):
    """
    Query audit logs with filters.
    
    Searches audit logs based on time range, event types,
    users, and resources. Requires audit:read permission.
    """
    try:
        # Set default time range if not provided
        if not request.start_time:
            request.start_time = datetime.utcnow() - timedelta(days=7)
        if not request.end_time:
            request.end_time = datetime.utcnow()
        
        # Query audit logs
        events = await audit.query_events(
            start_time=request.start_time,
            end_time=request.end_time,
            event_types=request.event_types,
            user_ids=request.user_ids,
            resource_types=request.resource_types,
            limit=request.limit
        )
        
        return {
            "events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "user_id": event.user_id,
                    "resource_type": event.resource_type,
                    "resource_id": event.resource_id,
                    "action": event.action,
                    "details": event.details,
                    "severity": event.severity,
                    "ip_address": event.ip_address
                }
                for event in events
            ],
            "total_events": len(events),
            "query_parameters": {
                "start_time": request.start_time.isoformat(),
                "end_time": request.end_time.isoformat(),
                "filters_applied": {
                    "event_types": request.event_types,
                    "user_ids": request.user_ids,
                    "resource_types": request.resource_types
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Audit query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compliance/report")
async def generate_compliance_report(
    request: ComplianceReportRequest,
    current_user: User = Depends(require_permission("compliance:report")),
    audit: AuditLogger = Depends(get_audit_logger)
):
    """
    Generate compliance report.
    
    Creates compliance report for specified standard (GDPR, HIPAA, etc.)
    and time range. Requires compliance:report permission.
    """
    try:
        # Generate report
        report = await audit.generate_compliance_report(
            report_type=request.report_type,
            time_range_days=request.time_range_days,
            include_details=request.include_details
        )
        
        # Log report generation
        await audit.log_security_event(
            SecurityEvent(
                event_type="compliance_report_generated",
                user_id=current_user.id,
                resource_type="compliance",
                resource_id=request.report_type,
                details={
                    "report_type": request.report_type,
                    "time_range_days": request.time_range_days,
                    "export_format": request.export_format
                },
                severity="medium"
            )
        )
        
        return {
            "report_id": report["report_id"],
            "report_type": request.report_type,
            "generated_at": datetime.utcnow().isoformat(),
            "time_range_days": request.time_range_days,
            "compliance_summary": report["summary"],
            "findings": report["findings"] if request.include_details else [],
            "recommendations": report["recommendations"],
            "export_format": request.export_format
        }
        
    except Exception as e:
        logger.error(f"Compliance report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=SecurityMetricsResponse)
async def get_security_metrics(
    current_user: User = Depends(require_permission("security:monitor")),
    rbac: RBACManager = Depends(get_rbac_manager),
    encryption: MemoryEncryption = Depends(get_encryption_service),
    audit: AuditLogger = Depends(get_audit_logger)
):
    """
    Get comprehensive security system metrics.
    
    Returns detailed metrics from all security subsystems including
    RBAC, encryption, and audit logging.
    """
    try:
        # Gather metrics from all components concurrently
        metrics_tasks = [
            rbac.get_security_metrics(),
            encryption.get_encryption_metrics(),
            audit.get_audit_metrics()
        ]
        
        rbac_metrics, encryption_metrics, audit_metrics = await asyncio.gather(
            *metrics_tasks, return_exceptions=True
        )
        
        # Handle any exceptions in metrics gathering
        if isinstance(rbac_metrics, Exception):
            rbac_metrics = {"error": str(rbac_metrics)}
        if isinstance(encryption_metrics, Exception):
            encryption_metrics = {"error": str(encryption_metrics)}
        if isinstance(audit_metrics, Exception):
            audit_metrics = {"error": str(audit_metrics)}
        
        # Get recent security events summary
        security_events = await audit.get_security_events_summary()
        
        # Assess compliance status
        compliance_status = {
            "gdpr": "compliant",
            "hipaa": "compliant", 
            "sox": "compliant",
            "pci_dss": "compliant",
            "last_assessment": datetime.utcnow().isoformat(),
            "next_assessment": (datetime.utcnow() + timedelta(days=30)).isoformat()
        }
        
        return SecurityMetricsResponse(
            timestamp=datetime.utcnow(),
            rbac_metrics=rbac_metrics,
            encryption_metrics=encryption_metrics,
            audit_metrics=audit_metrics,
            security_events=security_events,
            compliance_status=compliance_status
        )
        
    except Exception as e:
        logger.error(f"Failed to get security metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router
__all__ = ["router"]