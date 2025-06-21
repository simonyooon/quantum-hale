"""
Cryptographic Utilities for Quantum HALE Drone System

This module provides cryptographic utilities and key management functions
for the quantum-secured communications system.
"""

import hashlib
import hmac
import secrets
import base64
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import time

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    logging.warning("Cryptography library not available, using basic implementation")
    CRYPTOGRAPHY_AVAILABLE = False


class KeyType(Enum):
    """Types of cryptographic keys"""
    SESSION = "session"
    IDENTITY = "identity"
    QUANTUM = "quantum"
    MASTER = "master"


@dataclass
class KeyInfo:
    """Information about a cryptographic key"""
    key_id: str
    key_type: KeyType
    algorithm: str
    key_size: int
    created_at: float
    expires_at: Optional[float]
    usage_count: int = 0


class KeyManager:
    """
    Manages cryptographic keys for the quantum communications system
    """
    
    def __init__(self):
        self.keys: Dict[str, bytes] = {}
        self.key_info: Dict[str, KeyInfo] = {}
        self.master_key: Optional[bytes] = None
        
        logging.info("Key Manager initialized")
    
    def generate_master_key(self) -> bytes:
        """Generate a new master key"""
        self.master_key = secrets.token_bytes(32)
        key_id = self._generate_key_id()
        
        self.keys[key_id] = self.master_key
        self.key_info[key_id] = KeyInfo(
            key_id=key_id,
            key_type=KeyType.MASTER,
            algorithm="AES-256",
            key_size=256,
            created_at=time.time(),
            expires_at=None
        )
        
        logging.info(f"Master key generated: {key_id}")
        return self.master_key
    
    def derive_session_key(self, shared_secret: bytes, session_id: str, 
                          key_size: int = 256) -> bytes:
        """
        Derive a session key from a shared secret using HKDF
        
        Args:
            shared_secret: The shared secret from key exchange
            session_id: Unique session identifier
            key_size: Size of the derived key in bits
            
        Returns:
            Derived session key
        """
        if CRYPTOGRAPHY_AVAILABLE:
            return self._derive_key_cryptography(shared_secret, session_id, key_size)
        else:
            return self._derive_key_basic(shared_secret, session_id, key_size)
    
    def _derive_key_cryptography(self, shared_secret: bytes, session_id: str, 
                                key_size: int) -> bytes:
        """Derive key using cryptography library"""
        salt = session_id.encode()
        info = b"quantum-hale-session-key"
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_size // 8,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = kdf.derive(shared_secret)
        return key
    
    def _derive_key_basic(self, shared_secret: bytes, session_id: str, 
                         key_size: int) -> bytes:
        """Derive key using basic cryptographic functions"""
        salt = session_id.encode()
        info = b"quantum-hale-session-key"
        
        # Simple HKDF-like derivation
        prk = hashlib.pbkdf2_hmac('sha256', shared_secret, salt, 1)
        key = hashlib.pbkdf2_hmac('sha256', prk, info, 1)
        
        # Ensure correct key size
        if len(key) < key_size // 8:
            # Extend key if needed
            extended_key = key
            counter = 1
            while len(extended_key) < key_size // 8:
                counter_bytes = counter.to_bytes(4, 'big')
                next_block = hashlib.pbkdf2_hmac('sha256', prk, info + counter_bytes, 1)
                extended_key += next_block
                counter += 1
            key = extended_key[:key_size // 8]
        else:
            key = key[:key_size // 8]
        
        return key
    
    def store_key(self, key: bytes, key_type: KeyType, algorithm: str, 
                  expires_at: Optional[float] = None) -> str:
        """
        Store a cryptographic key
        
        Args:
            key: The key to store
            key_type: Type of the key
            algorithm: Algorithm used with this key
            expires_at: Expiration timestamp (None for no expiration)
            
        Returns:
            Key ID
        """
        key_id = self._generate_key_id()
        
        self.keys[key_id] = key
        self.key_info[key_id] = KeyInfo(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            key_size=len(key) * 8,
            created_at=time.time(),
            expires_at=expires_at
        )
        
        logging.info(f"Key stored: {key_id} ({key_type.value}, {algorithm})")
        return key_id
    
    def get_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve a key by ID"""
        if key_id not in self.keys:
            return None
        
        key_info = self.key_info[key_id]
        
        # Check expiration
        if key_info.expires_at and time.time() > key_info.expires_at:
            logging.warning(f"Key expired: {key_id}")
            self.delete_key(key_id)
            return None
        
        # Update usage count
        key_info.usage_count += 1
        
        return self.keys[key_id]
    
    def delete_key(self, key_id: str) -> bool:
        """Delete a key"""
        if key_id in self.keys:
            del self.keys[key_id]
            del self.key_info[key_id]
            logging.info(f"Key deleted: {key_id}")
            return True
        return False
    
    def cleanup_expired_keys(self) -> int:
        """Remove all expired keys"""
        current_time = time.time()
        expired_keys = []
        
        for key_id, key_info in self.key_info.items():
            if key_info.expires_at and current_time > key_info.expires_at:
                expired_keys.append(key_id)
        
        for key_id in expired_keys:
            self.delete_key(key_id)
        
        logging.info(f"Cleaned up {len(expired_keys)} expired keys")
        return len(expired_keys)
    
    def get_key_info(self, key_id: str) -> Optional[KeyInfo]:
        """Get information about a key"""
        return self.key_info.get(key_id)
    
    def list_keys(self, key_type: Optional[KeyType] = None) -> Dict[str, KeyInfo]:
        """List all keys, optionally filtered by type"""
        if key_type is None:
            return self.key_info.copy()
        
        return {k: v for k, v in self.key_info.items() if v.key_type == key_type}
    
    def _generate_key_id(self) -> str:
        """Generate a unique key ID"""
        return secrets.token_hex(16)


class CryptoUtils:
    """
    Cryptographic utility functions for the quantum communications system
    """
    
    def __init__(self):
        self.key_manager = KeyManager()
        
        logging.info("Crypto Utils initialized")
    
    def encrypt_message(self, message: bytes, key: bytes) -> bytes:
        """
        Encrypt a message using AES-256-GCM
        
        Args:
            message: Message to encrypt
            key: Encryption key
            
        Returns:
            Encrypted message with authentication tag
        """
        if CRYPTOGRAPHY_AVAILABLE:
            return self._encrypt_cryptography(message, key)
        else:
            return self._encrypt_basic(message, key)
    
    def _encrypt_cryptography(self, message: bytes, key: bytes) -> bytes:
        """Encrypt using cryptography library"""
        # Generate random IV
        iv = secrets.token_bytes(12)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Encrypt and get tag
        ciphertext = encryptor.update(message) + encryptor.finalize()
        tag = encryptor.tag
        
        # Combine IV, ciphertext, and tag
        return iv + ciphertext + tag
    
    def _encrypt_basic(self, message: bytes, key: bytes) -> bytes:
        """Basic encryption using XOR and HMAC"""
        # Generate random IV
        iv = secrets.token_bytes(16)
        
        # Pad message to block size
        block_size = 16
        padding_length = block_size - (len(message) % block_size)
        padded_message = message + bytes([padding_length] * padding_length)
        
        # Simple XOR encryption with key
        encrypted = bytearray()
        for i in range(0, len(padded_message), block_size):
            block = padded_message[i:i+block_size]
            # XOR with key (repeated if necessary)
            encrypted_block = bytes(a ^ b for a, b in zip(block, key[:len(block)]))
            encrypted.extend(encrypted_block)
        
        # Add HMAC for integrity
        h = hmac.new(key, iv + bytes(encrypted), hashlib.sha256)
        tag = h.digest()[:16]
        
        return iv + bytes(encrypted) + tag
    
    def decrypt_message(self, encrypted_message: bytes, key: bytes) -> Optional[bytes]:
        """
        Decrypt a message using AES-256-GCM
        
        Args:
            encrypted_message: Encrypted message
            key: Decryption key
            
        Returns:
            Decrypted message or None if decryption fails
        """
        if CRYPTOGRAPHY_AVAILABLE:
            return self._decrypt_cryptography(encrypted_message, key)
        else:
            return self._decrypt_basic(encrypted_message, key)
    
    def _decrypt_cryptography(self, encrypted_message: bytes, key: bytes) -> Optional[bytes]:
        """Decrypt using cryptography library"""
        try:
            # Extract IV, ciphertext, and tag
            iv = encrypted_message[:12]
            tag = encrypted_message[-16:]
            ciphertext = encrypted_message[12:-16]
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Decrypt
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return plaintext
            
        except Exception as e:
            logging.error(f"Decryption failed: {e}")
            return None
    
    def _decrypt_basic(self, encrypted_message: bytes, key: bytes) -> Optional[bytes]:
        """Basic decryption"""
        try:
            # Extract IV, ciphertext, and tag
            iv = encrypted_message[:16]
            tag = encrypted_message[-16:]
            ciphertext = encrypted_message[16:-16]
            
            # Verify HMAC
            h = hmac.new(key, iv + ciphertext, hashlib.sha256)
            expected_tag = h.digest()[:16]
            
            if not hmac.compare_digest(tag, expected_tag):
                logging.error("HMAC verification failed")
                return None
            
            # Decrypt
            decrypted = bytearray()
            for i in range(0, len(ciphertext), 16):
                block = ciphertext[i:i+16]
                # XOR with key
                decrypted_block = bytes(a ^ b for a, b in zip(block, key[:len(block)]))
                decrypted.extend(decrypted_block)
            
            # Remove padding
            padding_length = decrypted[-1]
            if padding_length > 16:
                logging.error("Invalid padding")
                return None
            
            return bytes(decrypted[:-padding_length])
            
        except Exception as e:
            logging.error(f"Decryption failed: {e}")
            return None
    
    def generate_random_bytes(self, length: int) -> bytes:
        """Generate cryptographically secure random bytes"""
        return secrets.token_bytes(length)
    
    def hash_message(self, message: bytes, algorithm: str = "sha256") -> bytes:
        """Hash a message using the specified algorithm"""
        if algorithm.lower() == "sha256":
            return hashlib.sha256(message).digest()
        elif algorithm.lower() == "sha3_256":
            return hashlib.sha3_256(message).digest()
        elif algorithm.lower() == "blake2b":
            return hashlib.blake2b(message).digest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    def verify_message_integrity(self, message: bytes, expected_hash: bytes, 
                                algorithm: str = "sha256") -> bool:
        """Verify message integrity using hash comparison"""
        actual_hash = self.hash_message(message, algorithm)
        return hmac.compare_digest(actual_hash, expected_hash)
    
    def create_secure_token(self, data: Dict[str, Any], secret_key: bytes, 
                           expires_in: int = 3600) -> str:
        """
        Create a secure token (JWT-like)
        
        Args:
            data: Data to include in token
            secret_key: Secret key for signing
            expires_in: Token expiration time in seconds
            
        Returns:
            Base64-encoded secure token
        """
        # Create payload
        payload = {
            "data": data,
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in
        }
        
        # Encode payload
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode()
        
        # Create signature
        signature = hmac.new(secret_key, payload_bytes, hashlib.sha256).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode()
        
        # Combine payload and signature
        token = f"{payload_b64}.{signature_b64}"
        
        return token
    
    def verify_secure_token(self, token: str, secret_key: bytes) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a secure token
        
        Args:
            token: Token to verify
            secret_key: Secret key for verification
            
        Returns:
            Decoded data or None if verification fails
        """
        try:
            # Split token
            parts = token.split('.')
            if len(parts) != 2:
                return None
            
            payload_b64, signature_b64 = parts
            
            # Decode payload
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            payload = json.loads(payload_bytes.decode())
            
            # Verify signature
            expected_signature = hmac.new(secret_key, payload_bytes, hashlib.sha256).digest()
            actual_signature = base64.urlsafe_b64decode(signature_b64)
            
            if not hmac.compare_digest(expected_signature, actual_signature):
                return None
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                return None
            
            return payload.get("data")
            
        except Exception as e:
            logging.error(f"Token verification failed: {e}")
            return None
    
    def derive_key_from_password(self, password: str, salt: bytes, 
                                key_size: int = 256) -> bytes:
        """
        Derive a key from a password using PBKDF2
        
        Args:
            password: Password to derive key from
            salt: Salt for key derivation
            key_size: Size of derived key in bits
            
        Returns:
            Derived key
        """
        if CRYPTOGRAPHY_AVAILABLE:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_size // 8,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            return kdf.derive(password.encode())
        else:
            # Basic PBKDF2 implementation
            return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, key_size // 8)
    
    def get_key_manager(self) -> KeyManager:
        """Get the key manager instance"""
        return self.key_manager 