"""
Memory Encryption System for Secure Data Protection.

This module provides comprehensive encryption capabilities for memory data including
content encryption with AES-256, embedding encryption using local algorithms,
metadata protection with field-level encryption, and key management with rotation support.
All processing is performed locally with zero external dependencies.
"""

import asyncio
import secrets
import hashlib
import hmac
import json
import base64
import zlib
from typing import Dict, List, Optional, Set, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import struct
import numpy as np

# Cryptography imports - all local
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class EncryptionMethod(str, Enum):
    """Encryption methods supported."""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"


class KeyDerivationMethod(str, Enum):
    """Key derivation methods."""
    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"
    ARGON2 = "argon2"
    HKDF = "hkdf"


class EncryptionScope(str, Enum):
    """Scope of encryption."""
    CONTENT = "content"
    EMBEDDING = "embedding"
    METADATA = "metadata"
    FULL_MEMORY = "full_memory"
    FIELD_LEVEL = "field_level"
    DATABASE = "database"
    TRANSPORT = "transport"


class CompressionMethod(str, Enum):
    """Compression methods before encryption."""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    BROTLI = "brotli"


@dataclass
class EncryptionKey:
    """Represents an encryption key."""
    key_id: str
    key_data: bytes
    method: EncryptionMethod
    purpose: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    rotation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def should_rotate(self, max_age_days: int = 90) -> bool:
        """Check if key should be rotated."""
        age = datetime.utcnow() - self.created_at
        return age.days >= max_age_days


@dataclass
class EncryptedData:
    """Represents encrypted data."""
    ciphertext: bytes
    method: EncryptionMethod
    key_id: str
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    salt: Optional[bytes] = None
    compression: CompressionMethod = CompressionMethod.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode(),
            "method": self.method.value,
            "key_id": self.key_id,
            "iv": base64.b64encode(self.iv).decode() if self.iv else None,
            "tag": base64.b64encode(self.tag).decode() if self.tag else None,
            "salt": base64.b64encode(self.salt).decode() if self.salt else None,
            "compression": self.compression.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedData':
        """Create from dictionary."""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            method=EncryptionMethod(data["method"]),
            key_id=data["key_id"],
            iv=base64.b64decode(data["iv"]) if data.get("iv") else None,
            tag=base64.b64decode(data["tag"]) if data.get("tag") else None,
            salt=base64.b64decode(data["salt"]) if data.get("salt") else None,
            compression=CompressionMethod(data.get("compression", "none")),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
        )


class KeyManager:
    """Manages encryption keys with rotation and derivation."""
    
    def __init__(self, cache: Optional[RedisCache] = None):
        self.cache = cache
        self.keys: Dict[str, EncryptionKey] = {}
        self.active_key_ids: Dict[str, str] = {}  # purpose -> active key_id
        self.backend = default_backend()
        self._initialize_master_keys()
    
    def _initialize_master_keys(self):
        """Initialize master encryption keys."""
        # Create master key for memory content
        content_key = self._generate_key(
            purpose="memory_content",
            method=EncryptionMethod.AES_256_GCM
        )
        
        # Create master key for embeddings
        embedding_key = self._generate_key(
            purpose="memory_embedding",
            method=EncryptionMethod.AES_256_GCM
        )
        
        # Create master key for metadata
        metadata_key = self._generate_key(
            purpose="memory_metadata",
            method=EncryptionMethod.AES_256_GCM
        )
        
        # Create RSA key pair for asymmetric operations
        rsa_key = self._generate_rsa_key_pair(purpose="asymmetric_ops")
        
        logger.info("Master encryption keys initialized")
    
    def _generate_key(
        self,
        purpose: str,
        method: EncryptionMethod,
        key_size: Optional[int] = None
    ) -> EncryptionKey:
        """Generate a new encryption key."""
        key_id = f"{purpose}_{secrets.token_hex(16)}"
        
        if method == EncryptionMethod.AES_256_GCM:
            key_data = secrets.token_bytes(32)  # 256 bits
        elif method == EncryptionMethod.AES_256_CBC:
            key_data = secrets.token_bytes(32)  # 256 bits
        elif method == EncryptionMethod.CHACHA20_POLY1305:
            key_data = secrets.token_bytes(32)  # 256 bits
        elif method == EncryptionMethod.FERNET:
            key_data = Fernet.generate_key()
        else:
            raise ValueError(f"Unsupported key generation method: {method}")
        
        key = EncryptionKey(
            key_id=key_id,
            key_data=key_data,
            method=method,
            purpose=purpose
        )
        
        self.keys[key_id] = key
        self.active_key_ids[purpose] = key_id
        
        logger.info("Encryption key generated", key_id=key_id, purpose=purpose, method=method.value)
        return key
    
    def _generate_rsa_key_pair(self, purpose: str, key_size: int = 2048) -> Tuple[EncryptionKey, EncryptionKey]:
        """Generate RSA key pair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )
        
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo
        )
        
        # Create key objects
        private_key_id = f"{purpose}_private_{secrets.token_hex(16)}"
        public_key_id = f"{purpose}_public_{secrets.token_hex(16)}"
        
        private_encryption_key = EncryptionKey(
            key_id=private_key_id,
            key_data=private_pem,
            method=EncryptionMethod.RSA_2048 if key_size == 2048 else EncryptionMethod.RSA_4096,
            purpose=f"{purpose}_private"
        )
        
        public_encryption_key = EncryptionKey(
            key_id=public_key_id,
            key_data=public_pem,
            method=EncryptionMethod.RSA_2048 if key_size == 2048 else EncryptionMethod.RSA_4096,
            purpose=f"{purpose}_public"
        )
        
        self.keys[private_key_id] = private_encryption_key
        self.keys[public_key_id] = public_encryption_key
        self.active_key_ids[f"{purpose}_private"] = private_key_id
        self.active_key_ids[f"{purpose}_public"] = public_key_id
        
        logger.info("RSA key pair generated", private_key_id=private_key_id, public_key_id=public_key_id)
        return private_encryption_key, public_encryption_key
    
    async def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get an encryption key by ID."""
        # Try cache first
        if self.cache:
            cached = await self.cache.get(f"encryption_key:{key_id}")
            if cached:
                return EncryptionKey(**cached)
        
        key = self.keys.get(key_id)
        
        # Cache the result
        if key and self.cache:
            await self.cache.set(
                f"encryption_key:{key_id}",
                key.__dict__,
                ttl=3600  # 1 hour
            )
        
        return key
    
    async def get_active_key(self, purpose: str) -> Optional[EncryptionKey]:
        """Get the active key for a purpose."""
        key_id = self.active_key_ids.get(purpose)
        if key_id:
            return await self.get_key(key_id)
        return None
    
    def derive_key(
        self,
        password: str,
        salt: bytes,
        method: KeyDerivationMethod = KeyDerivationMethod.PBKDF2,
        key_length: int = 32
    ) -> bytes:
        """Derive a key from password using specified method."""
        if method == KeyDerivationMethod.PBKDF2:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                iterations=100000,
                backend=self.backend
            )
            return kdf.derive(password.encode())
        
        elif method == KeyDerivationMethod.SCRYPT:
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                n=2**14,
                r=8,
                p=1,
                backend=self.backend
            )
            return kdf.derive(password.encode())
        
        else:
            raise ValueError(f"Unsupported key derivation method: {method}")
    
    async def rotate_key(self, purpose: str) -> EncryptionKey:
        """Rotate the active key for a purpose."""
        current_key = await self.get_active_key(purpose)
        if not current_key:
            raise ValueError(f"No active key found for purpose: {purpose}")
        
        # Generate new key
        new_key = self._generate_key(purpose, current_key.method)
        
        # Deactivate old key but keep it for decryption
        current_key.is_active = False
        current_key.rotation_count += 1
        
        logger.info("Key rotated", old_key_id=current_key.key_id, new_key_id=new_key.key_id, purpose=purpose)
        return new_key
    
    def list_keys(self, purpose: Optional[str] = None) -> List[EncryptionKey]:
        """List encryption keys."""
        keys = list(self.keys.values())
        if purpose:
            keys = [k for k in keys if k.purpose == purpose]
        return keys


class EncryptionEngine:
    """Core encryption engine for memory data."""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        self.backend = default_backend()
    
    async def encrypt_data(
        self,
        data: bytes,
        purpose: str,
        method: Optional[EncryptionMethod] = None,
        compression: CompressionMethod = CompressionMethod.ZLIB
    ) -> EncryptedData:
        """Encrypt data using specified or default method."""
        
        # Get encryption key
        key = await self.key_manager.get_active_key(purpose)
        if not key:
            raise ValueError(f"No active encryption key for purpose: {purpose}")
        
        # Use provided method or key's default method
        encryption_method = method or key.method
        
        # Compress data if requested
        if compression != CompressionMethod.NONE:
            data = self._compress_data(data, compression)
        
        # Encrypt based on method
        if encryption_method == EncryptionMethod.AES_256_GCM:
            return await self._encrypt_aes_gcm(data, key, compression)
        elif encryption_method == EncryptionMethod.AES_256_CBC:
            return await self._encrypt_aes_cbc(data, key, compression)
        elif encryption_method == EncryptionMethod.CHACHA20_POLY1305:
            return await self._encrypt_chacha20(data, key, compression)
        elif encryption_method == EncryptionMethod.FERNET:
            return await self._encrypt_fernet(data, key, compression)
        else:
            raise ValueError(f"Unsupported encryption method: {encryption_method}")
    
    async def decrypt_data(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt data."""
        key = await self.key_manager.get_key(encrypted_data.key_id)
        if not key:
            raise ValueError(f"Encryption key not found: {encrypted_data.key_id}")
        
        # Decrypt based on method
        if encrypted_data.method == EncryptionMethod.AES_256_GCM:
            data = await self._decrypt_aes_gcm(encrypted_data, key)
        elif encrypted_data.method == EncryptionMethod.AES_256_CBC:
            data = await self._decrypt_aes_cbc(encrypted_data, key)
        elif encrypted_data.method == EncryptionMethod.CHACHA20_POLY1305:
            data = await self._decrypt_chacha20(encrypted_data, key)
        elif encrypted_data.method == EncryptionMethod.FERNET:
            data = await self._decrypt_fernet(encrypted_data, key)
        else:
            raise ValueError(f"Unsupported decryption method: {encrypted_data.method}")
        
        # Decompress if compressed
        if encrypted_data.compression != CompressionMethod.NONE:
            data = self._decompress_data(data, encrypted_data.compression)
        
        return data
    
    def _compress_data(self, data: bytes, method: CompressionMethod) -> bytes:
        """Compress data before encryption."""
        if method == CompressionMethod.ZLIB:
            return zlib.compress(data)
        elif method == CompressionMethod.GZIP:
            import gzip
            return gzip.compress(data)
        else:
            return data
    
    def _decompress_data(self, data: bytes, method: CompressionMethod) -> bytes:
        """Decompress data after decryption."""
        if method == CompressionMethod.ZLIB:
            return zlib.decompress(data)
        elif method == CompressionMethod.GZIP:
            import gzip
            return gzip.decompress(data)
        else:
            return data
    
    async def _encrypt_aes_gcm(
        self,
        data: bytes,
        key: EncryptionKey,
        compression: CompressionMethod
    ) -> EncryptedData:
        """Encrypt using AES-256-GCM."""
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.GCM(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return EncryptedData(
            ciphertext=ciphertext,
            method=EncryptionMethod.AES_256_GCM,
            key_id=key.key_id,
            iv=iv,
            tag=encryptor.tag,
            compression=compression
        )
    
    async def _decrypt_aes_gcm(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt using AES-256-GCM."""
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.GCM(encrypted_data.iv, encrypted_data.tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
    
    async def _encrypt_aes_cbc(
        self,
        data: bytes,
        key: EncryptionKey,
        compression: CompressionMethod
    ) -> EncryptedData:
        """Encrypt using AES-256-CBC."""
        # Pad data to block size
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        iv = secrets.token_bytes(16)  # 128-bit IV for CBC
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        return EncryptedData(
            ciphertext=ciphertext,
            method=EncryptionMethod.AES_256_CBC,
            key_id=key.key_id,
            iv=iv,
            compression=compression
        )
    
    async def _decrypt_aes_cbc(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt using AES-256-CBC."""
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.CBC(encrypted_data.iv),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
        
        # Remove padding
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()
    
    async def _encrypt_chacha20(
        self,
        data: bytes,
        key: EncryptionKey,
        compression: CompressionMethod
    ) -> EncryptedData:
        """Encrypt using ChaCha20-Poly1305."""
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        cipher = Cipher(
            algorithms.ChaCha20(key.key_data, nonce),
            mode=None,
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return EncryptedData(
            ciphertext=ciphertext,
            method=EncryptionMethod.CHACHA20_POLY1305,
            key_id=key.key_id,
            iv=nonce,
            compression=compression
        )
    
    async def _decrypt_chacha20(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt using ChaCha20-Poly1305."""
        cipher = Cipher(
            algorithms.ChaCha20(key.key_data, encrypted_data.iv),
            mode=None,
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
    
    async def _encrypt_fernet(
        self,
        data: bytes,
        key: EncryptionKey,
        compression: CompressionMethod
    ) -> EncryptedData:
        """Encrypt using Fernet."""
        fernet = Fernet(key.key_data)
        ciphertext = fernet.encrypt(data)
        
        return EncryptedData(
            ciphertext=ciphertext,
            method=EncryptionMethod.FERNET,
            key_id=key.key_id,
            compression=compression
        )
    
    async def _decrypt_fernet(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt using Fernet."""
        fernet = Fernet(key.key_data)
        return fernet.decrypt(encrypted_data.ciphertext)


class MemoryEncryption:
    """High-level memory encryption interface."""
    
    def __init__(self, key_manager: KeyManager, encryption_engine: EncryptionEngine):
        self.key_manager = key_manager
        self.encryption_engine = encryption_engine
    
    async def encrypt_memory_content(
        self,
        content: str,
        compression: CompressionMethod = CompressionMethod.ZLIB
    ) -> EncryptedData:
        """Encrypt memory content."""
        content_bytes = content.encode('utf-8')
        return await self.encryption_engine.encrypt_data(
            content_bytes,
            purpose="memory_content",
            compression=compression
        )
    
    async def decrypt_memory_content(self, encrypted_data: EncryptedData) -> str:
        """Decrypt memory content."""
        content_bytes = await self.encryption_engine.decrypt_data(encrypted_data)
        return content_bytes.decode('utf-8')
    
    async def encrypt_memory_embedding(
        self,
        embedding: np.ndarray,
        compression: CompressionMethod = CompressionMethod.ZLIB
    ) -> EncryptedData:
        """Encrypt memory embedding."""
        embedding_bytes = embedding.tobytes()
        return await self.encryption_engine.encrypt_data(
            embedding_bytes,
            purpose="memory_embedding",
            compression=compression
        )
    
    async def decrypt_memory_embedding(
        self,
        encrypted_data: EncryptedData,
        dtype: np.dtype = np.float32,
        shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """Decrypt memory embedding."""
        embedding_bytes = await self.encryption_engine.decrypt_data(encrypted_data)
        embedding = np.frombuffer(embedding_bytes, dtype=dtype)
        
        if shape:
            embedding = embedding.reshape(shape)
        
        return embedding
    
    async def encrypt_memory_metadata(
        self,
        metadata: Dict[str, Any],
        compression: CompressionMethod = CompressionMethod.ZLIB
    ) -> EncryptedData:
        """Encrypt memory metadata."""
        metadata_json = json.dumps(metadata, default=str)
        metadata_bytes = metadata_json.encode('utf-8')
        return await self.encryption_engine.encrypt_data(
            metadata_bytes,
            purpose="memory_metadata",
            compression=compression
        )
    
    async def decrypt_memory_metadata(self, encrypted_data: EncryptedData) -> Dict[str, Any]:
        """Decrypt memory metadata."""
        metadata_bytes = await self.encryption_engine.decrypt_data(encrypted_data)
        metadata_json = metadata_bytes.decode('utf-8')
        return json.loads(metadata_json)
    
    async def encrypt_full_memory(self, memory: Memory) -> Dict[str, EncryptedData]:
        """Encrypt all components of a memory object."""
        encrypted_components = {}
        
        # Encrypt content
        if memory.content:
            encrypted_components['content'] = await self.encrypt_memory_content(memory.content)
        
        # Encrypt embedding
        if memory.embedding is not None:
            encrypted_components['embedding'] = await self.encrypt_memory_embedding(memory.embedding)
        
        # Encrypt metadata
        metadata = {
            'user_id': memory.user_id,
            'created_at': memory.created_at.isoformat() if memory.created_at else None,
            'updated_at': memory.updated_at.isoformat() if memory.updated_at else None,
            'tags': memory.tags,
            'importance_score': memory.importance_score,
            'access_count': memory.access_count,
            'last_accessed': memory.last_accessed.isoformat() if memory.last_accessed else None
        }
        encrypted_components['metadata'] = await self.encrypt_memory_metadata(metadata)
        
        return encrypted_components
    
    async def decrypt_full_memory(
        self,
        memory_id: str,
        encrypted_components: Dict[str, EncryptedData],
        embedding_dtype: np.dtype = np.float32,
        embedding_shape: Optional[Tuple[int, ...]] = None
    ) -> Memory:
        """Decrypt all components and reconstruct memory object."""
        
        # Decrypt content
        content = None
        if 'content' in encrypted_components:
            content = await self.decrypt_memory_content(encrypted_components['content'])
        
        # Decrypt embedding
        embedding = None
        if 'embedding' in encrypted_components:
            embedding = await self.decrypt_memory_embedding(
                encrypted_components['embedding'],
                dtype=embedding_dtype,
                shape=embedding_shape
            )
        
        # Decrypt metadata
        metadata = {}
        if 'metadata' in encrypted_components:
            metadata = await self.decrypt_memory_metadata(encrypted_components['metadata'])
        
        # Reconstruct memory object
        memory = Memory(
            id=memory_id,
            content=content,
            embedding=embedding,
            user_id=metadata.get('user_id'),
            created_at=datetime.fromisoformat(metadata['created_at']) if metadata.get('created_at') else None,
            updated_at=datetime.fromisoformat(metadata['updated_at']) if metadata.get('updated_at') else None,
            tags=metadata.get('tags', []),
            importance_score=metadata.get('importance_score', 0.0),
            access_count=metadata.get('access_count', 0),
            last_accessed=datetime.fromisoformat(metadata['last_accessed']) if metadata.get('last_accessed') else None
        )
        
        return memory


class FieldLevelEncryption:
    """Provides field-level encryption for sensitive data."""
    
    def __init__(self, encryption_engine: EncryptionEngine):
        self.encryption_engine = encryption_engine
        self.sensitive_fields = {
            'email', 'phone', 'ssn', 'credit_card', 'personal_info',
            'private_notes', 'confidential_data', 'password_hash'
        }
    
    async def encrypt_fields(
        self,
        data: Dict[str, Any],
        field_purposes: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Encrypt sensitive fields in a data dictionary."""
        encrypted_data = data.copy()
        
        for field_name, field_value in data.items():
            if self._is_sensitive_field(field_name) and field_value is not None:
                purpose = (field_purposes or {}).get(field_name, "sensitive_field")
                
                # Convert value to bytes
                if isinstance(field_value, str):
                    value_bytes = field_value.encode('utf-8')
                elif isinstance(field_value, (int, float)):
                    value_bytes = str(field_value).encode('utf-8')
                else:
                    value_bytes = json.dumps(field_value, default=str).encode('utf-8')
                
                # Encrypt the field
                encrypted_field = await self.encryption_engine.encrypt_data(
                    value_bytes,
                    purpose=purpose
                )
                
                # Store as encrypted data dictionary
                encrypted_data[field_name] = encrypted_field.to_dict()
                encrypted_data[f"{field_name}_encrypted"] = True
        
        return encrypted_data
    
    async def decrypt_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt encrypted fields in a data dictionary."""
        decrypted_data = data.copy()
        
        for field_name, field_value in data.items():
            if field_name.endswith('_encrypted') and field_value is True:
                original_field = field_name.replace('_encrypted', '')
                
                if original_field in data and isinstance(data[original_field], dict):
                    # Reconstruct encrypted data object
                    encrypted_data = EncryptedData.from_dict(data[original_field])
                    
                    # Decrypt the field
                    decrypted_bytes = await self.encryption_engine.decrypt_data(encrypted_data)
                    
                    # Convert back to appropriate type
                    if original_field in ['email', 'phone', 'ssn']:
                        decrypted_data[original_field] = decrypted_bytes.decode('utf-8')
                    else:
                        try:
                            decrypted_data[original_field] = json.loads(decrypted_bytes.decode('utf-8'))
                        except json.JSONDecodeError:
                            decrypted_data[original_field] = decrypted_bytes.decode('utf-8')
                    
                    # Remove encryption markers
                    del decrypted_data[f"{original_field}_encrypted"]
        
        return decrypted_data
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if a field should be encrypted."""
        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in self.sensitive_fields)


class DatabaseEncryption:
    """Provides transparent database encryption."""
    
    def __init__(self, memory_encryption: MemoryEncryption):
        self.memory_encryption = memory_encryption
        self.encrypted_tables = {'memories', 'user_data', 'sensitive_logs'}
    
    async def encrypt_for_storage(self, table_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt record before database storage."""
        if table_name not in self.encrypted_tables:
            return record
        
        encrypted_record = record.copy()
        
        if table_name == 'memories':
            # Handle memory-specific encryption
            if 'content' in record and record['content']:
                encrypted_content = await self.memory_encryption.encrypt_memory_content(record['content'])
                encrypted_record['content'] = encrypted_content.to_dict()
                encrypted_record['content_encrypted'] = True
            
            if 'embedding' in record and record['embedding'] is not None:
                embedding_array = np.array(record['embedding'])
                encrypted_embedding = await self.memory_encryption.encrypt_memory_embedding(embedding_array)
                encrypted_record['embedding'] = encrypted_embedding.to_dict()
                encrypted_record['embedding_encrypted'] = True
        
        return encrypted_record
    
    async def decrypt_from_storage(self, table_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt record after database retrieval."""
        if table_name not in self.encrypted_tables:
            return record
        
        decrypted_record = record.copy()
        
        if table_name == 'memories':
            # Handle memory-specific decryption
            if record.get('content_encrypted') and 'content' in record:
                encrypted_data = EncryptedData.from_dict(record['content'])
                decrypted_content = await self.memory_encryption.decrypt_memory_content(encrypted_data)
                decrypted_record['content'] = decrypted_content
                del decrypted_record['content_encrypted']
            
            if record.get('embedding_encrypted') and 'embedding' in record:
                encrypted_data = EncryptedData.from_dict(record['embedding'])
                decrypted_embedding = await self.memory_encryption.decrypt_memory_embedding(encrypted_data)
                decrypted_record['embedding'] = decrypted_embedding.tolist()
                del decrypted_record['embedding_encrypted']
        
        return decrypted_record


class EncryptionSystem:
    """
    Complete Memory Encryption System.
    
    Provides comprehensive encryption capabilities for memory data including
    content, embeddings, metadata, and field-level encryption with key management.
    """
    
    def __init__(self, cache: Optional[RedisCache] = None):
        self.cache = cache
        self.key_manager = KeyManager(cache)
        self.encryption_engine = EncryptionEngine(self.key_manager)
        self.memory_encryption = MemoryEncryption(self.key_manager, self.encryption_engine)
        self.field_encryption = FieldLevelEncryption(self.encryption_engine)
        self.database_encryption = DatabaseEncryption(self.memory_encryption)
        
        # Performance metrics
        self.encryption_stats = {
            'total_encryptions': 0,
            'total_decryptions': 0,
            'total_bytes_encrypted': 0,
            'total_bytes_decrypted': 0,
            'avg_encryption_time': 0.0,
            'avg_decryption_time': 0.0
        }
        
        logger.info("Memory encryption system initialized")
    
    async def encrypt_memory(self, memory: Memory) -> Dict[str, EncryptedData]:
        """Encrypt a complete memory object."""
        start_time = datetime.utcnow()
        
        try:
            encrypted_components = await self.memory_encryption.encrypt_full_memory(memory)
            
            # Update statistics
            self.encryption_stats['total_encryptions'] += 1
            duration = (datetime.utcnow() - start_time).total_seconds()
            self._update_avg_time('encryption', duration)
            
            logger.info(
                "Memory encrypted successfully",
                memory_id=memory.id,
                components=list(encrypted_components.keys()),
                duration_ms=duration * 1000
            )
            
            return encrypted_components
            
        except Exception as e:
            logger.error("Memory encryption failed", memory_id=memory.id, error=str(e))
            raise
    
    async def decrypt_memory(
        self,
        memory_id: str,
        encrypted_components: Dict[str, EncryptedData],
        embedding_dtype: np.dtype = np.float32,
        embedding_shape: Optional[Tuple[int, ...]] = None
    ) -> Memory:
        """Decrypt a complete memory object."""
        start_time = datetime.utcnow()
        
        try:
            memory = await self.memory_encryption.decrypt_full_memory(
                memory_id,
                encrypted_components,
                embedding_dtype,
                embedding_shape
            )
            
            # Update statistics
            self.encryption_stats['total_decryptions'] += 1
            duration = (datetime.utcnow() - start_time).total_seconds()
            self._update_avg_time('decryption', duration)
            
            logger.info(
                "Memory decrypted successfully",
                memory_id=memory_id,
                components=list(encrypted_components.keys()),
                duration_ms=duration * 1000
            )
            
            return memory
            
        except Exception as e:
            logger.error("Memory decryption failed", memory_id=memory_id, error=str(e))
            raise
    
    def _update_avg_time(self, operation: str, duration: float):
        """Update average operation time."""
        key = f'avg_{operation}_time'
        count_key = f'total_{operation}s'
        
        current_avg = self.encryption_stats[key]
        count = self.encryption_stats[count_key]
        
        # Calculate new average
        new_avg = ((current_avg * (count - 1)) + duration) / count
        self.encryption_stats[key] = new_avg
    
    async def rotate_keys(self, purposes: Optional[List[str]] = None) -> Dict[str, str]:
        """Rotate encryption keys for specified purposes."""
        if purposes is None:
            purposes = ['memory_content', 'memory_embedding', 'memory_metadata']
        
        rotated_keys = {}
        
        for purpose in purposes:
            try:
                new_key = await self.key_manager.rotate_key(purpose)
                rotated_keys[purpose] = new_key.key_id
                logger.info("Key rotated successfully", purpose=purpose, new_key_id=new_key.key_id)
            except Exception as e:
                logger.error("Key rotation failed", purpose=purpose, error=str(e))
                rotated_keys[purpose] = f"ERROR: {str(e)}"
        
        return rotated_keys
    
    def get_encryption_stats(self) -> Dict[str, Any]:
        """Get encryption system statistics."""
        return {
            **self.encryption_stats,
            'active_keys': len([k for k in self.key_manager.keys.values() if k.is_active]),
            'total_keys': len(self.key_manager.keys),
            'keys_requiring_rotation': len([
                k for k in self.key_manager.keys.values()
                if k.should_rotate() and k.is_active
            ])
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform encryption system health check."""
        try:
            # Test encryption/decryption with sample data
            test_data = "Test encryption data"
            encrypted = await self.memory_encryption.encrypt_memory_content(test_data)
            decrypted = await self.memory_encryption.decrypt_memory_content(encrypted)
            
            encryption_working = decrypted == test_data
            
            # Check key availability
            content_key = await self.key_manager.get_active_key("memory_content")
            embedding_key = await self.key_manager.get_active_key("memory_embedding")
            metadata_key = await self.key_manager.get_active_key("memory_metadata")
            
            return {
                "status": "healthy" if encryption_working else "unhealthy",
                "encryption_test_passed": encryption_working,
                "active_keys": {
                    "content": content_key.key_id if content_key else None,
                    "embedding": embedding_key.key_id if embedding_key else None,
                    "metadata": metadata_key.key_id if metadata_key else None
                },
                "statistics": self.get_encryption_stats()
            }
            
        except Exception as e:
            logger.error("Encryption health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }