"""ImageEncryptor — in-memory frame encryption for the PPE pipeline.

Keeps the ``encrypt_frame`` / ``decrypt_frame`` API from the first draft, but
swaps the engine from symmetric Fernet to an ASYMMETRIC sealed-box envelope
(see ``security/envelope.py``).

Why the change: to encrypt at the şantiye, the encryption key has to live on the
site device. With a symmetric key (Fernet) that *same* key also decrypts, so a
lost or stolen device leaks every image ever captured. With a sealed box the
site holds only the PUBLIC key and cannot decrypt anything — only HQ's PRIVATE
key can. That is the KVKK property we need: a site-device compromise exposes no
images. The blobs produced here use the same ``.enc`` format as
``security/encrypt.py`` and ``security/ingest.py``, so the frame-level API and
the file-level tools are interoperable.

Usage
-----
    # Site device — encrypt only (holds the public key, cannot decrypt)
    enc = ImageEncryptor.for_site("keys/ppe_hq.pub")
    blob = enc.encrypt_frame(frame)            # -> bytes, safe to store/transmit

    # HQ — decrypt (needs the private key)
    dec = ImageEncryptor.for_hq("keys/ppe_hq.key")   # or PPE_PRIVATE_KEY env
    frame = dec.decrypt_frame(blob)            # -> np.ndarray (BGR)

Generate the keypair once, at HQ:  ``uv run security/keygen.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Make the repo root importable so `security` resolves when this module is used
# from scripts under core/ (e.g. the pipeline) or via `uv run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from security.envelope import (  # noqa: E402
    decrypt_blob,
    encrypt_bytes,
    load_private_key,
    load_public_key,
)


class ImageEncryptor:
    """Encrypt/decrypt OpenCV frames using the asymmetric sealed-box envelope.

    Construct via :meth:`for_site` (public key, encrypt-only) or :meth:`for_hq`
    (private key, decrypt + encrypt). You can also pass keys directly.
    """

    def __init__(self, public_key=None, private_key=None):
        # public_key  -> can encrypt (site side)
        # private_key -> can decrypt (HQ side); its public key can also encrypt
        self._public_key = public_key
        self._private_key = private_key

    # --- constructors --------------------------------------------------------
    @classmethod
    def for_site(cls, public_key_path: str) -> "ImageEncryptor":
        """Site device: can encrypt only. Give it HQ's .pub file."""
        return cls(public_key=load_public_key(public_key_path))

    @classmethod
    def for_hq(cls, private_key_source: str | None = None) -> "ImageEncryptor":
        """HQ: can decrypt (and encrypt). Key from a path, base64, or PPE_PRIVATE_KEY."""
        priv = load_private_key(private_key_source)
        return cls(public_key=priv.public_key, private_key=priv)

    # --- API (compatible with the original draft) ---------------------------
    def encrypt_frame(self, frame_np: np.ndarray, ext: str = "jpg") -> bytes:
        """Encode an OpenCV frame to image bytes in memory and encrypt it.

        Returns a ``.enc`` blob (header + sealed box). Needs a public key.
        """
        if self._public_key is None:
            raise RuntimeError(
                "This encryptor has no public key and cannot encrypt. "
                "Build it with ImageEncryptor.for_site('keys/ppe_hq.pub')."
            )
        ext = ext.lstrip(".") or "jpg"
        success, encoded = cv2.imencode(f".{ext}", frame_np)
        if not success:
            raise ValueError("Failed to encode frame to byte format.")
        return encrypt_bytes(encoded.tobytes(), self._public_key, ext=ext)

    def decrypt_frame(self, encrypted_bytes: bytes) -> np.ndarray:
        """Decrypt a ``.enc`` blob back into an OpenCV frame (BGR ndarray).

        HQ only — needs the private key.
        """
        if self._private_key is None:
            raise RuntimeError(
                "This encryptor has no private key and cannot decrypt (by design, "
                "so site devices can't). Build it with ImageEncryptor.for_hq(key) at HQ."
            )
        plaintext, _ext = decrypt_blob(encrypted_bytes, self._private_key)
        frame = cv2.imdecode(np.frombuffer(plaintext, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Decrypted data is not a valid image.")
        return frame
