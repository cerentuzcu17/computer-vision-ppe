"""Security tests: the properties KVKK actually depends on.

Run:  uv run pytest tests/test_security.py -v
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nacl.exceptions import CryptoError  # noqa: E402
from nacl.public import PrivateKey  # noqa: E402

from security.encrypt import strip_exif  # noqa: E402
from security.envelope import (  # noqa: E402
    MAGIC,
    decrypt_blob,
    encrypt_bytes,
    is_encrypted,
)


@pytest.fixture
def keypair():
    priv = PrivateKey.generate()
    return priv, priv.public_key


IMG = b"\xff\xd8\xff\xe0not-a-real-jpeg-but-arbitrary-bytes\x00\x01\x02\xff\xd9"


# --- core guarantee: only the private key recovers the plaintext ------------
def test_roundtrip_recovers_exact_bytes(keypair):
    priv, pub = keypair
    blob = encrypt_bytes(IMG, pub, ext="jpg")
    plaintext, ext = decrypt_blob(blob, priv)
    assert plaintext == IMG
    assert ext == "jpg"


def test_ciphertext_is_not_plaintext(keypair):
    _, pub = keypair
    blob = encrypt_bytes(IMG, pub, ext="jpg")
    # The raw image bytes must not appear in the encrypted container.
    assert IMG not in blob


def test_wrong_key_cannot_decrypt(keypair):
    _, pub = keypair
    blob = encrypt_bytes(IMG, pub, ext="jpg")
    attacker = PrivateKey.generate()  # an outsider's key
    with pytest.raises(CryptoError):
        decrypt_blob(blob, attacker)


def test_tampered_ciphertext_is_rejected(keypair):
    priv, pub = keypair
    blob = bytearray(encrypt_bytes(IMG, pub, ext="jpg"))
    blob[-1] ^= 0x01  # flip one bit in the ciphertext
    with pytest.raises(CryptoError):
        decrypt_blob(bytes(blob), priv)


def test_two_encryptions_differ(keypair):
    """Sealed boxes use a fresh ephemeral key each time -> no ciphertext reuse."""
    _, pub = keypair
    assert encrypt_bytes(IMG, pub, ext="jpg") != encrypt_bytes(IMG, pub, ext="jpg")


# --- .enc container format ---------------------------------------------------
def test_is_encrypted_detects_magic(keypair):
    _, pub = keypair
    assert is_encrypted(encrypt_bytes(IMG, pub))
    assert not is_encrypted(IMG)


def test_bad_header_raises(keypair):
    priv, _ = keypair
    with pytest.raises(ValueError):
        decrypt_blob(b"NOTPPE" + b"\x00" * 40, priv)


def test_unsupported_version_raises(keypair):
    priv, pub = keypair
    blob = bytearray(encrypt_bytes(IMG, pub, ext="jpg"))
    blob[len(MAGIC)] = 99  # bogus version byte
    with pytest.raises(ValueError):
        decrypt_blob(bytes(blob), priv)


# --- EXIF stripping (data minimization) -------------------------------------
def _jpeg_with_app1(payload: bytes) -> bytes:
    """Minimal JPEG: SOI + APP1(EXIF-like) + SOS + data + EOI."""
    app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    sos = b"\xff\xda\x00\x03\x01\x00"  # tiny scan header
    return b"\xff\xd8" + app1 + sos + b"scandata" + b"\xff\xd9"


def test_strip_exif_removes_app1():
    secret = b"Exif\x00\x00GPS:41.0,29.0;Make:SiteCam"
    jpeg = _jpeg_with_app1(secret)
    out = strip_exif(jpeg, ".jpg")
    assert secret not in out          # identifying metadata gone
    assert b"scandata" in out         # pixel/scan data preserved
    assert out.startswith(b"\xff\xd8") and out.endswith(b"\xff\xd9")


def test_strip_exif_passes_through_non_jpeg():
    png = b"\x89PNG\r\n\x1a\n" + b"whatever"
    assert strip_exif(png, ".png") == png


# --- ImageEncryptor facade (core/security.py) -------------------------------
def _tmp_keys(tmp_path):
    """Write a real keypair to disk and return (pub_path, key_path)."""
    priv = PrivateKey.generate()
    import base64
    pub_p = tmp_path / "hq.pub"
    key_p = tmp_path / "hq.key"
    pub_p.write_text(base64.b64encode(bytes(priv.public_key)).decode())
    key_p.write_text(base64.b64encode(bytes(priv)).decode())
    return str(pub_p), str(key_p)


def test_imageencryptor_frame_roundtrip(tmp_path):
    import numpy as np

    from core.security import ImageEncryptor

    pub_p, key_p = _tmp_keys(tmp_path)
    frame = np.full((16, 24, 3), 127, dtype=np.uint8)

    site = ImageEncryptor.for_site(pub_p)          # public key only
    blob = site.encrypt_frame(frame)
    assert blob[:6] == MAGIC

    hq = ImageEncryptor.for_hq(key_p)              # private key
    out = hq.decrypt_frame(blob)
    assert out.shape == frame.shape


def test_site_encryptor_cannot_decrypt(tmp_path):
    """The core KVKK property: a site device (public key only) can't decrypt."""
    import numpy as np

    from core.security import ImageEncryptor

    pub_p, _ = _tmp_keys(tmp_path)
    site = ImageEncryptor.for_site(pub_p)
    blob = site.encrypt_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError):
        site.decrypt_frame(blob)
