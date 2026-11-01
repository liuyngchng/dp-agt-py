"""
敏感配置加密工具（AES-256-GCM）

- 密钥独立存放在 data/secrets.key（首次启动自动生成，权限 600）
- app.db 中的密文与 secrets.key 分离，单独偷走任一文件都无法解密
- 加密的是 CSV 上传的 prompt_config 值（接口地址、访问密钥等）
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_key: bytes | None = None


def init_crypto(data_dir: str) -> bytes:
    """初始化加密密钥（幂等）。返回当前密钥。"""
    global _key
    key_file = Path(data_dir) / "secrets.key"
    if key_file.exists():
        _key = bytes.fromhex(key_file.read_text(encoding="utf-8").strip())
        return _key
    _key = AESGCM.generate_key(bit_length=256)
    key_file.write_text(_key.hex(), encoding="utf-8")
    key_file.chmod(0o600)
    return _key


def encrypt_value(plain: str) -> "EncryptedValue":
    """加密一段明文。"""
    if _key is None:
        raise RuntimeError("crypto 未初始化")
    nonce = os.urandom(12)
    aesgcm = AESGCM(_key)
    ciphertext = aesgcm.encrypt(nonce, plain.encode("utf-8"), None)
    return EncryptedValue(
        enc=base64.b64encode(ciphertext).decode("ascii"),
        iv=base64.b64encode(nonce).decode("ascii"),
        tag="",
    )


def decrypt_value(enc: str, iv: str, tag: str) -> str | None:
    """解密一段密文，失败返回 None（密钥不匹配等场景）。"""
    if _key is None:
        raise RuntimeError("crypto 未初始化")
    try:
        nonce = base64.b64decode(iv.encode("ascii"))
        ciphertext = base64.b64decode(enc.encode("ascii"))
        aesgcm = AESGCM(_key)
        plain = aesgcm.decrypt(nonce, ciphertext, None)
        return plain.decode("utf-8")
    except Exception:
        return None


class EncryptedValue:
    def __init__(self, enc: str, iv: str, tag: str):
        self.enc = enc  # base64 密文
        self.iv = iv    # base64 IV
        self.tag = tag  # auth tag（AESGCM 内置，保留字段兼容）
