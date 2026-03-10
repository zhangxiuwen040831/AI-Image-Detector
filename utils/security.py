import os
import bcrypt
import base64
from typing import Optional
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class SecurityUtils:
    def __init__(self, key: str = None):
        # 优先使用环境变量中的密钥，如果未提供则使用默认值（仅用于开发）
        # 生产环境必须强制通过 AES_SECRET_KEY 注入
        env_key = os.environ.get("AES_SECRET_KEY")
        if env_key:
            self.key = base64.b64decode(env_key) if len(env_key) > 32 else env_key.encode('utf-8')
        elif key:
            self.key = key.encode('utf-8')
        else:
            # 默认开发密钥 (32 bytes for AES-256)
            self.key = b'dev_secret_key_must_be_32_bytes!'
            
        if len(self.key) not in [16, 24, 32]:
            raise ValueError("AES key must be 16, 24, or 32 bytes long")

    @staticmethod
    def hash_password(password: str) -> str:
        """使用 bcrypt (cost=12) 对密码进行加盐哈希"""
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """验证密码是否匹配"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def encrypt_field(self, plaintext: str) -> str:
        """使用 AES-256-GCM 加密敏感字段"""
        if not plaintext:
            return ""
        
        iv = get_random_bytes(12) # GCM 推荐 IV 长度为 12 字节
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=iv)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode('utf-8'))
        
        # 格式: iv + tag + ciphertext (Base64 encoded)
        combined = iv + tag + ciphertext
        return base64.b64encode(combined).decode('utf-8')

    def decrypt_field(self, encrypted_text: str) -> str:
        """解密敏感字段"""
        if not encrypted_text:
            return ""
            
        try:
            combined = base64.b64decode(encrypted_text)
            # 解析: IV (12) + Tag (16) + Ciphertext
            iv = combined[:12]
            tag = combined[12:28]
            ciphertext = combined[28:]
            
            cipher = AES.new(self.key, AES.MODE_GCM, nonce=iv)
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            return plaintext.decode('utf-8')
        except Exception as e:
            # 在生产环境中应记录审计日志
            print(f"[SECURITY ALERT] Decryption failed: {e}")
            raise ValueError("Decryption failed: Invalid key or corrupted data")

# 单例实例
security_manager = SecurityUtils()
