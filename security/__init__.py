"""Site-side image confidentiality for the PPE pipeline (KVKK).

Envelope encryption using libsodium sealed boxes (X25519 + XSalsa20-Poly1305):
photos are encrypted *at capture* to HQ's PUBLIC key, so a site device never
holds a key that can decrypt anything. Only the HQ PRIVATE key can recover an
image. See security/README.md for the design and key-management rules.
"""

from .envelope import (
    ENC_SUFFIX,
    decrypt_blob,
    decrypt_file_to_bytes,
    encrypt_bytes,
    is_encrypted,
    load_private_key,
    load_public_key,
)

__all__ = [
    "ENC_SUFFIX",
    "decrypt_blob",
    "decrypt_file_to_bytes",
    "encrypt_bytes",
    "is_encrypted",
    "load_private_key",
    "load_public_key",
]
