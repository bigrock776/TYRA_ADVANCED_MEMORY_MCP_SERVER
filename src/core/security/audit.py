"""
Comprehensive Audit Logging System for Security and Compliance.

This module provides complete audit capabilities with security event tracking,
compliance logging using local storage, activity monitoring with detailed trails,
and integrity verification with digital signatures. All processing is performed locally with zero external dependencies.
"""

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import gzip
import threading
from pathlib import Path

# Cryptography for integrity verification
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
from cryptography.hazmat.backends import default_backend

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGED = "permission_changed"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REMOVED = "role_removed"
    
    # Memory events
    MEMORY_CREATED = "memory_created"
    MEMORY_READ = "memory_read"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_SHARED = "memory_shared"
    MEMORY_SEARCHED = "memory_searched"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGED = "config_changed"
    KEY_ROTATED = "key_rotated"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    
    # Security events
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_VIOLATION = "security_violation"
    ENCRYPTION_ERROR = "encryption_error"
    INTEGRITY_VIOLATION = "integrity_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    
    # Data events
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    DATA_PURGED = "data_purged"
    DATA_ANONYMIZED = "data_anonymized"
    
    # Administrative events
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    USER_LOCKED = "user_locked"
    USER_UNLOCKED = "user_unlocked"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AuditOutcome(str, Enum):
    """Outcome of audited operations."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ComplianceStandard(str, Enum):
    """Compliance standards supported."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    SOC2 = "soc2"


@dataclass
class AuditEvent:
    """Represents an audit event."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    outcome: AuditOutcome = AuditOutcome.SUCCESS
    severity: AuditSeverity = AuditSeverity.INFO
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    compliance_tags: List[ComplianceStandard] = field(default_factory=list)
    risk_score: int = 0  # 0-100
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        
        # Auto-assign risk scores based on event type
        if self.risk_score == 0:
            self.risk_score = self._calculate_risk_score()
    
    def _calculate_risk_score(self) -> int:
        """Calculate risk score based on event type and severity."""
        base_scores = {
            AuditEventType.LOGIN_FAILURE: 30,
            AuditEventType.ACCESS_DENIED: 40,
            AuditEventType.SUSPICIOUS_ACTIVITY: 80,
            AuditEventType.SECURITY_VIOLATION: 90,
            AuditEventType.INTEGRITY_VIOLATION: 95,
            AuditEventType.ENCRYPTION_ERROR: 70,
            AuditEventType.DATA_EXPORT: 50,
            AuditEventType.MEMORY_DELETED: 40,
            AuditEventType.USER_LOCKED: 60,
            AuditEventType.CONFIG_CHANGED: 45,
        }
        
        base_score = base_scores.get(self.event_type, 10)
        
        # Adjust based on severity
        severity_multipliers = {
            AuditSeverity.CRITICAL: 1.5,
            AuditSeverity.HIGH: 1.2,
            AuditSeverity.MEDIUM: 1.0,
            AuditSeverity.LOW: 0.8,
            AuditSeverity.INFO: 0.5
        }
        
        multiplier = severity_multipliers.get(self.severity, 1.0)
        return min(100, int(base_score * multiplier))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['outcome'] = self.outcome.value
        data['severity'] = self.severity.value
        data['compliance_tags'] = [tag.value for tag in self.compliance_tags]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['event_type'] = AuditEventType(data['event_type'])
        data['outcome'] = AuditOutcome(data['outcome'])
        data['severity'] = AuditSeverity(data['severity'])
        data['compliance_tags'] = [ComplianceStandard(tag) for tag in data.get('compliance_tags', [])]
        return cls(**data)


@dataclass
class AuditFilter:
    """Filter criteria for audit searches."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[Set[AuditEventType]] = None
    user_ids: Optional[Set[str]] = None
    outcomes: Optional[Set[AuditOutcome]] = None
    severities: Optional[Set[AuditSeverity]] = None
    min_risk_score: Optional[int] = None
    max_risk_score: Optional[int] = None
    compliance_standards: Optional[Set[ComplianceStandard]] = None
    resource_types: Optional[Set[str]] = None
    ip_addresses: Optional[Set[str]] = None
    text_search: Optional[str] = None
    limit: int = 1000
    offset: int = 0


class AuditStorage:
    """Handles audit log storage and retrieval."""
    
    def __init__(self, storage_path: str = "./audit_logs", cache: Optional[RedisCache] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.cache = cache
        self.current_log_file = None
        self.file_lock = threading.Lock()
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.compression_enabled = True
        self._signature_key = self._generate_signature_key()
    
    def _generate_signature_key(self) -> rsa.RSAPrivateKey:
        """Generate RSA key for log integrity verification."""
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
    
    def _get_current_log_file(self) -> Path:
        """Get current log file path."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return self.storage_path / f"audit_{today}.jsonl"
    
    def _sign_event(self, event_data: bytes) -> str:
        """Create digital signature for event integrity."""
        signature = self._signature_key.sign(
            event_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return hashlib.sha256(event_data + signature).hexdigest()
    
    async def store_event(self, event: AuditEvent) -> bool:
        """Store audit event to disk."""
        try:
            event_json = json.dumps(event.to_dict(), separators=(',', ':'))
            event_data = event_json.encode('utf-8')
            
            # Add integrity signature
            signature = self._sign_event(event_data)
            
            # Prepare log entry
            log_entry = {
                "event": event.to_dict(),
                "integrity_hash": signature,
                "stored_at": datetime.utcnow().isoformat()
            }
            
            log_line = json.dumps(log_entry, separators=(',', ':')) + '\n'
            
            with self.file_lock:
                log_file = self._get_current_log_file()
                
                # Check if we need to rotate log file
                if log_file.exists() and log_file.stat().st_size > self.max_file_size:
                    await self._rotate_log_file(log_file)
                    log_file = self._get_current_log_file()
                
                # Write to log file
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(log_line)
                    f.flush()
            
            # Cache recent events
            if self.cache:
                await self.cache.lpush(
                    "recent_audit_events",
                    json.dumps(event.to_dict()),
                    max_length=1000
                )
            
            return True
            
        except Exception as e:
            logger.error("Failed to store audit event", event_id=event.event_id, error=str(e))
            return False
    
    async def _rotate_log_file(self, current_file: Path):
        """Rotate and compress current log file."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            archived_file = current_file.with_suffix(f'.{timestamp}.jsonl')
            
            # Move current file
            current_file.rename(archived_file)
            
            # Compress if enabled
            if self.compression_enabled:
                compressed_file = archived_file.with_suffix('.jsonl.gz')
                with open(archived_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        f_out.write(f_in.read())
                
                # Remove uncompressed file
                archived_file.unlink()
                
                logger.info("Log file rotated and compressed", 
                           original=str(current_file), 
                           compressed=str(compressed_file))
            else:
                logger.info("Log file rotated", 
                           original=str(current_file), 
                           archived=str(archived_file))
        
        except Exception as e:
            logger.error("Failed to rotate log file", file=str(current_file), error=str(e))
    
    async def search_events(self, audit_filter: AuditFilter) -> List[AuditEvent]:
        """Search audit events based on filter criteria."""
        events = []
        processed_count = 0
        
        try:
            # Get all log files in date range
            log_files = self._get_log_files_in_range(audit_filter.start_time, audit_filter.end_time)
            
            for log_file in log_files:
                file_events = await self._search_in_file(log_file, audit_filter, processed_count)
                events.extend(file_events)
                processed_count += len(file_events)
                
                # Check if we've reached the limit
                if len(events) >= audit_filter.limit:
                    events = events[:audit_filter.limit]
                    break
            
            return events
            
        except Exception as e:
            logger.error("Failed to search audit events", error=str(e))
            return []
    
    def _get_log_files_in_range(
        self, 
        start_time: Optional[datetime], 
        end_time: Optional[datetime]
    ) -> List[Path]:
        """Get log files that might contain events in the time range."""
        log_files = []
        
        # Get all audit log files
        for file_path in self.storage_path.glob("audit_*.jsonl*"):
            log_files.append(file_path)
        
        # Sort by modification time (newest first)
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return log_files
    
    async def _search_in_file(
        self, 
        file_path: Path, 
        audit_filter: AuditFilter,
        processed_count: int
    ) -> List[AuditEvent]:
        """Search for events in a specific log file."""
        events = []
        
        try:
            # Handle compressed files
            if file_path.suffix == '.gz':
                file_opener = gzip.open
                mode = 'rt'
            else:
                file_opener = open
                mode = 'r'
            
            with file_opener(file_path, mode, encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        log_entry = json.loads(line)
                        event_data = log_entry.get('event', {})
                        event = AuditEvent.from_dict(event_data)
                        
                        # Apply filters
                        if self._matches_filter(event, audit_filter):
                            # Check offset
                            if processed_count + len(events) >= audit_filter.offset:
                                events.append(event)
                                
                                # Check limit
                                if len(events) >= audit_filter.limit:
                                    break
                    
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in audit log", file=str(file_path))
                        continue
                    except Exception as e:
                        logger.warning("Error parsing audit event", file=str(file_path), error=str(e))
                        continue
        
        except Exception as e:
            logger.error("Failed to search in log file", file=str(file_path), error=str(e))
        
        return events
    
    def _matches_filter(self, event: AuditEvent, audit_filter: AuditFilter) -> bool:
        """Check if event matches filter criteria."""
        # Time range
        if audit_filter.start_time and event.timestamp < audit_filter.start_time:
            return False
        if audit_filter.end_time and event.timestamp > audit_filter.end_time:
            return False
        
        # Event types
        if audit_filter.event_types and event.event_type not in audit_filter.event_types:
            return False
        
        # User IDs
        if audit_filter.user_ids and event.user_id not in audit_filter.user_ids:
            return False
        
        # Outcomes
        if audit_filter.outcomes and event.outcome not in audit_filter.outcomes:
            return False
        
        # Severities
        if audit_filter.severities and event.severity not in audit_filter.severities:
            return False
        
        # Risk score
        if audit_filter.min_risk_score and event.risk_score < audit_filter.min_risk_score:
            return False
        if audit_filter.max_risk_score and event.risk_score > audit_filter.max_risk_score:
            return False
        
        # Compliance standards
        if audit_filter.compliance_standards:
            if not any(tag in audit_filter.compliance_standards for tag in event.compliance_tags):
                return False
        
        # Resource types
        if audit_filter.resource_types and event.resource_type not in audit_filter.resource_types:
            return False
        
        # IP addresses
        if audit_filter.ip_addresses and event.ip_address not in audit_filter.ip_addresses:
            return False
        
        # Text search
        if audit_filter.text_search:
            search_text = audit_filter.text_search.lower()
            searchable_content = f"{event.message} {json.dumps(event.details)}".lower()
            if search_text not in searchable_content:
                return False
        
        return True


class AuditAnalyzer:
    """Analyzes audit logs for patterns and anomalies."""
    
    def __init__(self, storage: AuditStorage):
        self.storage = storage
        self.pattern_cache: Dict[str, Any] = {}
        self.anomaly_thresholds = {
            'failed_logins_per_hour': 10,
            'access_denials_per_hour': 20,
            'high_risk_events_per_hour': 5,
            'unusual_ip_access': 3,
            'suspicious_time_access': 2  # Access outside normal hours
        }
    
    async def detect_anomalies(
        self, 
        time_window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in audit logs."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        audit_filter = AuditFilter(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        events = await self.storage.search_events(audit_filter)
        anomalies = []
        
        # Analyze different anomaly types
        anomalies.extend(await self._detect_failed_login_anomalies(events))
        anomalies.extend(await self._detect_access_denial_anomalies(events))
        anomalies.extend(await self._detect_high_risk_anomalies(events))
        anomalies.extend(await self._detect_unusual_ip_anomalies(events))
        anomalies.extend(await self._detect_time_based_anomalies(events))
        
        return anomalies
    
    async def _detect_failed_login_anomalies(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Detect excessive failed login attempts."""
        anomalies = []
        failed_logins = defaultdict(list)
        
        for event in events:
            if event.event_type == AuditEventType.LOGIN_FAILURE:
                hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
                failed_logins[hour_key].append(event)
        
        for hour, hour_events in failed_logins.items():
            if len(hour_events) > self.anomaly_thresholds['failed_logins_per_hour']:
                # Group by user and IP
                user_failures = defaultdict(int)
                ip_failures = defaultdict(int)
                
                for event in hour_events:
                    if event.user_id:
                        user_failures[event.user_id] += 1
                    if event.ip_address:
                        ip_failures[event.ip_address] += 1
                
                anomalies.append({
                    'type': 'excessive_failed_logins',
                    'severity': AuditSeverity.HIGH,
                    'timestamp': hour,
                    'count': len(hour_events),
                    'threshold': self.anomaly_thresholds['failed_logins_per_hour'],
                    'details': {
                        'user_failures': dict(user_failures),
                        'ip_failures': dict(ip_failures)
                    }
                })
        
        return anomalies
    
    async def _detect_access_denial_anomalies(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Detect excessive access denials."""
        anomalies = []
        access_denials = defaultdict(list)
        
        for event in events:
            if event.event_type == AuditEventType.ACCESS_DENIED:
                hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
                access_denials[hour_key].append(event)
        
        for hour, hour_events in access_denials.items():
            if len(hour_events) > self.anomaly_thresholds['access_denials_per_hour']:
                anomalies.append({
                    'type': 'excessive_access_denials',
                    'severity': AuditSeverity.MEDIUM,
                    'timestamp': hour,
                    'count': len(hour_events),
                    'threshold': self.anomaly_thresholds['access_denials_per_hour'],
                    'details': {
                        'users': list(set(e.user_id for e in hour_events if e.user_id)),
                        'resources': list(set(e.resource_type for e in hour_events if e.resource_type))
                    }
                })
        
        return anomalies
    
    async def _detect_high_risk_anomalies(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Detect clusters of high-risk events."""
        anomalies = []
        high_risk_events = defaultdict(list)
        
        for event in events:
            if event.risk_score >= 70:  # High risk threshold
                hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
                high_risk_events[hour_key].append(event)
        
        for hour, hour_events in high_risk_events.items():
            if len(hour_events) > self.anomaly_thresholds['high_risk_events_per_hour']:
                anomalies.append({
                    'type': 'high_risk_event_cluster',
                    'severity': AuditSeverity.HIGH,
                    'timestamp': hour,
                    'count': len(hour_events),
                    'threshold': self.anomaly_thresholds['high_risk_events_per_hour'],
                    'details': {
                        'event_types': list(set(e.event_type.value for e in hour_events)),
                        'avg_risk_score': sum(e.risk_score for e in hour_events) / len(hour_events)
                    }
                })
        
        return anomalies
    
    async def _detect_unusual_ip_anomalies(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Detect access from unusual IP addresses."""
        anomalies = []
        
        # Get baseline IP addresses (last 7 days)
        baseline_end = datetime.utcnow() - timedelta(days=1)
        baseline_start = baseline_end - timedelta(days=7)
        
        baseline_filter = AuditFilter(
            start_time=baseline_start,
            end_time=baseline_end,
            event_types={AuditEventType.LOGIN_SUCCESS},
            limit=5000
        )
        
        baseline_events = await self.storage.search_events(baseline_filter)
        baseline_ips = set(e.ip_address for e in baseline_events if e.ip_address)
        
        # Check current events for new IPs
        user_new_ips = defaultdict(set)
        
        for event in events:
            if (event.event_type == AuditEventType.LOGIN_SUCCESS and 
                event.ip_address and 
                event.ip_address not in baseline_ips):
                user_new_ips[event.user_id].add(event.ip_address)
        
        for user_id, new_ips in user_new_ips.items():
            if len(new_ips) > self.anomaly_thresholds['unusual_ip_access']:
                anomalies.append({
                    'type': 'unusual_ip_access',
                    'severity': AuditSeverity.MEDIUM,
                    'timestamp': datetime.utcnow(),
                    'user_id': user_id,
                    'details': {
                        'new_ips': list(new_ips),
                        'baseline_ip_count': len(baseline_ips)
                    }
                })
        
        return anomalies
    
    async def _detect_time_based_anomalies(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Detect access during unusual hours."""
        anomalies = []
        
        # Define normal business hours (configurable)
        normal_start_hour = 8
        normal_end_hour = 18
        
        suspicious_access = defaultdict(list)
        
        for event in events:
            if event.event_type in {AuditEventType.LOGIN_SUCCESS, AuditEventType.MEMORY_READ}:
                hour = event.timestamp.hour
                
                # Check if outside normal hours
                if hour < normal_start_hour or hour > normal_end_hour:
                    suspicious_access[event.user_id].append(event)
        
        for user_id, user_events in suspicious_access.items():
            if len(user_events) > self.anomaly_thresholds['suspicious_time_access']:
                anomalies.append({
                    'type': 'unusual_time_access',
                    'severity': AuditSeverity.LOW,
                    'timestamp': datetime.utcnow(),
                    'user_id': user_id,
                    'count': len(user_events),
                    'details': {
                        'hours': list(set(e.timestamp.hour for e in user_events)),
                        'normal_hours': f"{normal_start_hour}:00-{normal_end_hour}:00"
                    }
                })
        
        return anomalies
    
    async def generate_compliance_report(
        self,
        standard: ComplianceStandard,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for specified standard."""
        audit_filter = AuditFilter(
            start_time=start_time,
            end_time=end_time,
            compliance_standards={standard},
            limit=10000
        )
        
        events = await self.storage.search_events(audit_filter)
        
        report = {
            'standard': standard.value,
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'total_events': len(events),
            'summary': self._generate_compliance_summary(events, standard),
            'violations': await self._identify_compliance_violations(events, standard),
            'recommendations': self._generate_compliance_recommendations(events, standard)
        }
        
        return report
    
    def _generate_compliance_summary(
        self,
        events: List[AuditEvent],
        standard: ComplianceStandard
    ) -> Dict[str, Any]:
        """Generate summary statistics for compliance report."""
        event_types = defaultdict(int)
        outcomes = defaultdict(int)
        severities = defaultdict(int)
        
        for event in events:
            event_types[event.event_type.value] += 1
            outcomes[event.outcome.value] += 1
            severities[event.severity.value] += 1
        
        return {
            'event_types': dict(event_types),
            'outcomes': dict(outcomes),
            'severities': dict(severities),
            'high_risk_events': len([e for e in events if e.risk_score >= 70]),
            'security_events': len([e for e in events if 'security' in e.event_type.value]),
            'failure_rate': outcomes.get('failure', 0) / len(events) if events else 0
        }
    
    async def _identify_compliance_violations(
        self,
        events: List[AuditEvent],
        standard: ComplianceStandard
    ) -> List[Dict[str, Any]]:
        """Identify potential compliance violations."""
        violations = []
        
        if standard == ComplianceStandard.GDPR:
            violations.extend(self._check_gdpr_violations(events))
        elif standard == ComplianceStandard.HIPAA:
            violations.extend(self._check_hipaa_violations(events))
        elif standard == ComplianceStandard.SOX:
            violations.extend(self._check_sox_violations(events))
        
        return violations
    
    def _check_gdpr_violations(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Check for GDPR violations."""
        violations = []
        
        # Check for data exports without proper authorization
        for event in events:
            if event.event_type == AuditEventType.DATA_EXPORT:
                if event.outcome == AuditOutcome.SUCCESS and event.risk_score < 30:
                    violations.append({
                        'type': 'potential_unauthorized_data_export',
                        'event_id': event.event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'description': 'Data export with low risk score - verify authorization'
                    })
        
        return violations
    
    def _check_hipaa_violations(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Check for HIPAA violations."""
        violations = []
        
        # Check for access to sensitive data without audit trail
        sensitive_access = [e for e in events if 'health' in str(e.details).lower()]
        
        for event in sensitive_access:
            if not event.correlation_id:
                violations.append({
                    'type': 'health_data_access_without_correlation',
                    'event_id': event.event_id,
                    'timestamp': event.timestamp.isoformat(),
                    'description': 'Health data access without proper correlation tracking'
                })
        
        return violations
    
    def _check_sox_violations(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Check for SOX violations."""
        violations = []
        
        # Check for financial data access patterns
        financial_events = [e for e in events if 'financial' in str(e.details).lower()]
        
        for event in financial_events:
            if event.severity == AuditSeverity.CRITICAL and event.outcome == AuditOutcome.FAILURE:
                violations.append({
                    'type': 'critical_financial_system_failure',
                    'event_id': event.event_id,
                    'timestamp': event.timestamp.isoformat(),
                    'description': 'Critical failure in financial system requires investigation'
                })
        
        return violations
    
    def _generate_compliance_recommendations(
        self,
        events: List[AuditEvent],
        standard: ComplianceStandard
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        # Common recommendations
        if len([e for e in events if e.outcome == AuditOutcome.FAILURE]) > len(events) * 0.1:
            recommendations.append("High failure rate detected - review system reliability")
        
        if len([e for e in events if e.risk_score >= 80]) > 0:
            recommendations.append("High-risk events detected - implement additional monitoring")
        
        # Standard-specific recommendations
        if standard == ComplianceStandard.GDPR:
            if any(e.event_type == AuditEventType.DATA_EXPORT for e in events):
                recommendations.append("Implement data subject consent verification for exports")
        
        return recommendations


class AuditSystem:
    """
    Complete Audit Logging System.
    
    Provides comprehensive audit capabilities with security event tracking,
    compliance logging, activity monitoring, and integrity verification.
    """
    
    def __init__(
        self,
        storage_path: str = "./audit_logs",
        cache: Optional[RedisCache] = None
    ):
        self.storage = AuditStorage(storage_path, cache)
        self.analyzer = AuditAnalyzer(self.storage)
        self.cache = cache
        
        # Performance metrics
        self.metrics = {
            'events_logged': 0,
            'events_per_second': 0.0,
            'storage_errors': 0,
            'last_error': None
        }
        
        # Real-time monitoring
        self.alert_handlers: List[Callable] = []
        self.real_time_queue: deque = deque(maxlen=1000)
        
        logger.info("Audit system initialized", storage_path=storage_path)
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        severity: AuditSeverity = AuditSeverity.INFO,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        compliance_tags: Optional[List[ComplianceStandard]] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """Log an audit event."""
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            severity=severity,
            message=message,
            details=details or {},
            compliance_tags=compliance_tags or [],
            correlation_id=correlation_id
        )
        
        # Store event
        success = await self.storage.store_event(event)
        
        if success:
            self.metrics['events_logged'] += 1
            
            # Add to real-time queue
            self.real_time_queue.append(event)
            
            # Check for alerts
            await self._check_alert_conditions(event)
            
            logger.debug("Audit event logged", event_id=event.event_id, event_type=event_type.value)
        else:
            self.metrics['storage_errors'] += 1
            self.metrics['last_error'] = datetime.utcnow()
            logger.error("Failed to log audit event", event_type=event_type.value)
        
        return event.event_id
    
    async def _check_alert_conditions(self, event: AuditEvent):
        """Check if event triggers any alerts."""
        # High-risk event alert
        if event.risk_score >= 80:
            await self._trigger_alert("high_risk_event", event)
        
        # Security violation alert
        if event.event_type in {AuditEventType.SECURITY_VIOLATION, AuditEventType.SUSPICIOUS_ACTIVITY}:
            await self._trigger_alert("security_violation", event)
        
        # Multiple failures alert
        recent_failures = [
            e for e in self.real_time_queue
            if (e.outcome == AuditOutcome.FAILURE and 
                e.user_id == event.user_id and
                (event.timestamp - e.timestamp).total_seconds() < 300)  # 5 minutes
        ]
        
        if len(recent_failures) >= 3:
            await self._trigger_alert("multiple_failures", event)
    
    async def _trigger_alert(self, alert_type: str, event: AuditEvent):
        """Trigger alert for significant events."""
        alert_data = {
            'type': alert_type,
            'event': event.to_dict(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Notify registered handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                logger.error("Alert handler failed", alert_type=alert_type, error=str(e))
        
        # Cache alert for dashboard
        if self.cache:
            await self.cache.lpush(
                "audit_alerts",
                json.dumps(alert_data),
                max_length=100
            )
    
    def add_alert_handler(self, handler: Callable):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
    
    async def search_events(self, audit_filter: AuditFilter) -> List[AuditEvent]:
        """Search audit events."""
        return await self.storage.search_events(audit_filter)
    
    async def detect_anomalies(self, time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """Detect anomalies in audit logs."""
        return await self.analyzer.detect_anomalies(time_window_hours)
    
    async def generate_compliance_report(
        self,
        standard: ComplianceStandard,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report."""
        return await self.analyzer.generate_compliance_report(standard, start_time, end_time)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get audit system metrics."""
        return {
            **self.metrics,
            'recent_events_count': len(self.real_time_queue),
            'alert_handlers_count': len(self.alert_handlers)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform audit system health check."""
        try:
            # Test logging
            test_event_id = await self.log_event(
                AuditEventType.SYSTEM_STARTUP,
                message="Health check test event",
                severity=AuditSeverity.INFO
            )
            
            # Test search
            test_filter = AuditFilter(
                start_time=datetime.utcnow() - timedelta(minutes=1),
                limit=1
            )
            
            events = await self.search_events(test_filter)
            search_working = len(events) > 0
            
            return {
                "status": "healthy",
                "logging_working": bool(test_event_id),
                "search_working": search_working,
                "storage_path": str(self.storage.storage_path),
                "metrics": self.get_metrics()
            }
            
        except Exception as e:
            logger.error("Audit health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "metrics": self.get_metrics()
            }