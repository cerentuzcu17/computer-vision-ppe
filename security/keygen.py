"""Generate the HQ keypair for site-image encryption.

Run ONCE, at HQ, in a controlled environment:

    uv run security/keygen.py

Produces two files in keys/ :
    ppe_hq.pub   -> PUBLIC key. Distribute freely to every şantiye device.
    ppe_hq.key   -> PRIVATE key. HQ ONLY. Never commit, never copy to a site.

If keys already exist, the script refuses to overwrite them (regenerating a
keypair makes every previously-encrypted image undecryptable). Delete the old
keys deliberately if you truly mean to rotate.
"""

from __future__ import annotations

import argparse
import base64
import os
import stat
from pathlib import Path

from nacl.public import PrivateKey

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KEYDIR = REPO_ROOT / "keys"


def _write(path: Path, comment: str, key_bytes: bytes, private: bool) -> None:
    b64 = base64.b64encode(key_bytes).decode("ascii")
    path.write_text(f"# {comment}\n{b64}\n", encoding="utf-8")
    if private:
        # chmod 600 — owner read/write only.
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the HQ encryption keypair")
    parser.add_argument(
        "--out", default=str(DEFAULT_KEYDIR), help="output directory for the keys"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing keys (DANGER)"
    )
    args = parser.parse_args()

    keydir = Path(args.out)
    keydir.mkdir(parents=True, exist_ok=True)
    pub_path = keydir / "ppe_hq.pub"
    priv_path = keydir / "ppe_hq.key"

    if (pub_path.exists() or priv_path.exists()) and not args.force:
        raise SystemExit(
            f"Keys already exist in {keydir}. Refusing to overwrite.\n"
            "Rotating keys makes all existing .enc images undecryptable. "
            "Re-run with --force only if you are certain."
        )

    private_key = PrivateKey.generate()
    public_key = private_key.public_key

    _write(pub_path, "PPE HQ PUBLIC key (X25519) — safe to distribute to sites",
           bytes(public_key), private=False)
    _write(priv_path, "PPE HQ PRIVATE key (X25519) — HQ ONLY, never commit",
           bytes(private_key), private=True)

    print(f"[keygen] public  -> {pub_path}  (ship to site devices)")
    print(f"[keygen] private -> {priv_path}  (HQ only, chmod 600)")
    print("\nNext: distribute ONLY the .pub file to şantiye devices.")
    print("Keep ppe_hq.key in a KMS/secret manager if you can; never in git.")


if __name__ == "__main__":
    main()
