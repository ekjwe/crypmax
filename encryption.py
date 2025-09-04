"""
Encryption utilities for QuantumTrade Bot
Handles data encryption and decryption for sensitive information
"""

import os
import base64
import hashlib
import logging
from typing import Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class EncryptionManager:
    """Manages encryption and decryption of sensitive data"""
    
    def __init__(self, password: Optional[str] = None):
        self.password = password or os.getenv("ENCRYPTION_PASSWORD", "default_password_change_me")
        self.salt = b'quantumtrade_salt_2024'  # In production, use random salt per user
        self._cipher = None
        self._initialize_cipher()
    
    def _initialize_cipher(self):
        """Initialize the Fernet cipher with derived key"""
        try:
            # Derive key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
            self._cipher = Fernet(key)
            logger.info("Encryption cipher initialized")
        except Exception as e:
            logger.error(f"Error initializing encryption cipher: {e}")
            # Fallback to a simpler approach if cryptography library is not available
            self._cipher = self._create_simple_cipher()
    
    def _create_simple_cipher(self):
        """Create a simple cipher if cryptography library is not available"""
        try:
            # Simple base64 encoding as fallback (not secure, for demo only)
            class SimpleCipher:
                def encrypt(self, data: bytes) -> bytes:
                    return base64.b64encode(data)
                
                def decrypt(self, data: bytes) -> bytes:
                    return base64.b64decode(data)
            
            logger.warning("Using simple base64 encoding (not secure) - install cryptography library for proper encryption")
            return SimpleCipher()
        except Exception as e:
            logger.error(f"Error creating simple cipher: {e}")
            return None
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt data and return as base64 string"""
        try:
            if self._cipher is None:
                logger.warning("No cipher available, returning data as-is")
                return data if isinstance(data, str) else data.decode('utf-8', errors='ignore')
            
            # Convert to bytes if string
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Encrypt data
            encrypted_data = self._cipher.encrypt(data)
            
            # Return as base64 string
            return base64.b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            # Return original data if encryption fails
            return data if isinstance(data, str) else data.decode('utf-8', errors='ignore')
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt base64 encoded encrypted data"""
        try:
            if self._cipher is None:
                logger.warning("No cipher available, returning data as-is")
                return encrypted_data
            
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            
            # Decrypt data
            decrypted_data = self._cipher.decrypt(encrypted_bytes)
            
            # Return as string
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            # Return original data if decryption fails
            return encrypted_data
    
    def encrypt_dict(self, data: dict) -> dict:
        """Encrypt sensitive fields in a dictionary"""
        try:
            encrypted_dict = data.copy()
            
            # List of sensitive fields to encrypt
            sensitive_fields = [
                'password', 'api_key', 'secret_key', 'private_key',
                'token', 'auth_token', 'access_token', 'refresh_token',
                'email_password', 'telegram_bot_token', 'webhook_url'
            ]
            
            for field in sensitive_fields:
                if field in encrypted_dict and encrypted_dict[field]:
                    encrypted_dict[field] = self.encrypt(str(encrypted_dict[field]))
            
            return encrypted_dict
            
        except Exception as e:
            logger.error(f"Dictionary encryption error: {e}")
            return data
    
    def decrypt_dict(self, encrypted_data: dict) -> dict:
        """Decrypt sensitive fields in a dictionary"""
        try:
            decrypted_dict = encrypted_data.copy()
            
            # List of sensitive fields to decrypt
            sensitive_fields = [
                'password', 'api_key', 'secret_key', 'private_key',
                'token', 'auth_token', 'access_token', 'refresh_token',
                'email_password', 'telegram_bot_token', 'webhook_url'
            ]
            
            for field in sensitive_fields:
                if field in decrypted_dict and decrypted_dict[field]:
                    decrypted_dict[field] = self.decrypt(str(decrypted_dict[field]))
            
            return decrypted_dict
            
        except Exception as e:
            logger.error(f"Dictionary decryption error: {e}")
            return encrypted_data
    
    def generate_key(self) -> str:
        """Generate a new Fernet key"""
        try:
            key = Fernet.generate_key()
            return base64.urlsafe_b64encode(key).decode('utf-8')
        except Exception as e:
            logger.error(f"Key generation error: {e}")
            return base64.b64encode(os.urandom(32)).decode('utf-8')
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> tuple:
        """Hash password with salt"""
        try:
            if salt is None:
                salt = os.urandom(32)
            
            # Use PBKDF2 for password hashing
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = kdf.derive(password.encode())
            
            return base64.b64encode(key).decode('utf-8'), base64.b64encode(salt).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Password hashing error: {e}")
            # Fallback to simple hash
            salted_password = password + (salt.decode('utf-8', errors='ignore') if salt else 'default_salt')
            hashed = hashlib.sha256(salted_password.encode()).hexdigest()
            return hashed, base64.b64encode(salt).decode('utf-8') if salt else 'default_salt'
    
    def verify_password(self, password: str, hashed_password: str, salt_b64: str) -> bool:
        """Verify password against hash"""
        try:
            salt = base64.b64decode(salt_b64.encode('utf-8'))
            new_hash, _ = self.hash_password(password, salt)
            return new_hash == hashed_password
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> bool:
        """Encrypt a file"""
        try:
            if output_path is None:
                output_path = file_path + '.encrypted'
            
            with open(file_path, 'rb') as f:
                data = f.read()
            
            encrypted_data = self._cipher.encrypt(data)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info(f"File encrypted: {file_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"File encryption error: {e}")
            return False
    
    def decrypt_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> bool:
        """Decrypt a file"""
        try:
            if output_path is None:
                output_path = encrypted_file_path.replace('.encrypted', '')
            
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self._cipher.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            logger.info(f"File decrypted: {encrypted_file_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"File decryption error: {e}")
            return False
    
    def secure_delete(self, file_path: str, passes: int = 3) -> bool:
        """Securely delete a file by overwriting it multiple times"""
        try:
            if not os.path.exists(file_path):
                return True
            
            file_size = os.path.getsize(file_path)
            
            with open(file_path, 'r+b') as f:
                for _ in range(passes):
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            os.remove(file_path)
            logger.info(f"File securely deleted: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Secure delete error: {e}")
            return False
    
    def create_checksum(self, data: Union[str, bytes]) -> str:
        """Create SHA256 checksum of data"""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            return hashlib.sha256(data).hexdigest()
            
        except Exception as e:
            logger.error(f"Checksum creation error: {e}")
            return ""
    
    def verify_checksum(self, data: Union[str, bytes], expected_checksum: str) -> bool:
        """Verify data against expected checksum"""
        try:
            actual_checksum = self.create_checksum(data)
            return actual_checksum == expected_checksum
        except Exception as e:
            logger.error(f"Checksum verification error: {e}")
            return False
    
    def get_encryption_info(self) -> dict:
        """Get information about the encryption setup"""
        return {
            "cipher_available": self._cipher is not None,
            "encryption_method": "Fernet" if hasattr(self._cipher, 'encrypt') else "Simple Base64",
            "is_secure": hasattr(self._cipher, 'encrypt') and not isinstance(self._cipher, type(None)),
            "key_derivation": "PBKDF2-SHA256",
            "iterations": 100000
        }

# Create singleton instance
encryption_manager = EncryptionManager()
