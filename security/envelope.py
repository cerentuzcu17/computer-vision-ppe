"""Envelope encryption for site images.

Wraps libsodium *sealed boxes* (PyNaCl) so the rest of the codebase never has
to touch raw crypto. A sealed box:

  * encrypts to a recipient PUBLIC key only (anonymous sender),
  * generates a fresh ephemeral keypair per message (so every image gets its
    own one-time symmetric key under the hood — envelope encryption),
  * can only be opened with the matching PRIVATE key.

That is exactly the "encrypt on-site, decrypt only at HQ" property we need:
the şantiye device carries the public key, and losing that device leaks
nothing.

File format (.enc)
------------------
    MAGIC (6 bytes = b"PPEENC")
    VERSION (1 byte)
    EXT_LEN (1 byte)              # length of the original extension string
    EXT (EXT_LEN bytes, utf-8)    # e.g. "jpg" — used only to name/route output
    CIPHERTEXT (rest)             # sealed box: ephemeral pubkey || nonce || box

The header is authenticated indirectly: any tampering makes the sealed box
fail to open (Poly1305 MAC), so decryption raises rather than returning
corrupted pixels.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from nacl.public import PrivateKey, PublicKey, SealedBox

MAGIC = b"PPEENC"
VERSION = 1
ENC_SUFFIX = ".enc"

_HEADER_MIN = len(MAGIC) + 2  # magic + version + ext_len


# --- key loading ------------------------------------------------------------
def load_public_key(path: str | os.PathLike[str]) -> PublicKey:
    """Load HQ's public key (base64 text file). Safe to ship to site devices."""
    raw = _read_key_material(path)
    return PublicKey(raw)


def load_private_key(source: str | os.PathLike[str] | None = None) -> PrivateKey:
    """Load HQ's private key.

    Resolution order:
      1. explicit ``source`` (a file path, or a raw base64 string),
      2. the ``PPE_PRIVATE_KEY`` env var (base64 — good for KMS/secret injection),
      3. ``keys/ppe_hq.key`` relative to the repo root.

    The private key must never live on a site device or in version control.
    In production, prefer injecting it via env/secret manager over a file.
    """
    if source is not None:
        candidate = Path(str(source))
        if candidate.exists():
            return PrivateKey(_read_key_material(candidate))
        # Treat the string itself as base64 key material.
        return PrivateKey(_b64decode(str(source)))

    env = os.environ.get("PPE_PRIVATE_KEY")
    if env:
        return PrivateKey(_b64decode(env))

    default = Path(__file__).resolve().parent.parent / "keys" / "ppe_hq.key"
    if default.exists():
        return PrivateKey(_read_key_material(default))

    raise FileNotFoundError(
        "No private key found. Pass a path, set PPE_PRIVATE_KEY, or place "
        "keys/ppe_hq.key at the repo root. (Generate with security/keygen.py.)"
    )


def _read_key_material(path: str | os.PathLike[str]) -> bytes:
    """Read a key file. Lines starting with '#' are treated as comments."""
    text = Path(path).read_text(encoding="utf-8")
    body = "".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    ).strip()
    return _b64decode(body)


def _b64decode(s: str) -> bytes:
    raw = base64.b64decode(s.strip())
    if len(raw) != 32:
        raise ValueError(
            f"Expected a 32-byte X25519 key, got {len(raw)} bytes. "
            "Is the key file corrupted or the wrong format?"
        )
    return raw


# --- encrypt / decrypt ------------------------------------------------------
def encrypt_bytes(plaintext: bytes, public_key: PublicKey, ext: str = "") -> bytes:
    """Encrypt raw image bytes to ``public_key``. Returns a full .enc blob."""
    ext_b = ext.lstrip(".").encode("utf-8")[:255]
    ciphertext = SealedBox(public_key).encrypt(plaintext)
    header = MAGIC + bytes([VERSION, len(ext_b)]) + ext_b
    return header + ciphertext


def decrypt_blob(blob: bytes, private_key: PrivateKey) -> tuple[bytes, str]:
    """Decrypt an in-memory .enc blob. Returns (plaintext_bytes, original_ext).

    Raises ValueError on a bad header and nacl.exceptions.CryptoError if the
    ciphertext was tampered with or the wrong key is used.
    """
    if not is_encrypted(blob):
        raise ValueError("Not a PPEENC blob (bad magic).")
    version = blob[len(MAGIC)]
    if version != VERSION:
        raise ValueError(f"Unsupported .enc version: {version}")
    ext_len = blob[len(MAGIC) + 1]
    ext_start = _HEADER_MIN
    ext = blob[ext_start : ext_start + ext_len].decode("utf-8", "replace")
    ciphertext = blob[ext_start + ext_len :]
    plaintext = SealedBox(private_key).decrypt(ciphertext)
    return plaintext, ext


def decrypt_file_to_bytes(
    path: str | os.PathLike[str], private_key: PrivateKey
) -> tuple[bytes, str]:
    """Read a .enc file and decrypt it in memory. Plaintext never hits disk."""
    return decrypt_blob(Path(path).read_bytes(), private_key)


def is_encrypted(data: bytes | str | os.PathLike[str]) -> bool:
    """True if ``data`` (bytes blob or a file path) is a PPEENC container."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data[: len(MAGIC)]) == MAGIC
    p = Path(data)
    if not p.is_file():
        return False
    with p.open("rb") as fh:
        return fh.read(len(MAGIC)) == MAGIC
