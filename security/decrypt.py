"""HQ-side decryption (export). Run ONLY at HQ, where the private key lives.

Most of the time you do NOT need this: the inference pipeline decrypts images
in memory (see core/pose_prototype.py), so plaintext never touches disk. Use
this tool only when you deliberately need plaintext files back — e.g. building
a training set inside a controlled, access-logged environment.

    uv run security/decrypt.py --key keys/ppe_hq.key photo.jpg.enc --out ./cleartext
    PPE_PRIVATE_KEY=<base64> uv run security/decrypt.py *.enc --out ./cleartext

Writing plaintext to disk re-creates identifiable personal data. Only do it in
a secure location and delete it when done — that is a KVKK obligation, not just
good hygiene.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from security.envelope import (  # noqa: E402
    ENC_SUFFIX,
    decrypt_file_to_bytes,
    is_encrypted,
    load_private_key,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decrypt .enc site images (HQ only)")
    parser.add_argument("paths", nargs="+", help=".enc file(s) or directory(ies)")
    parser.add_argument("--key", default=None,
                        help="private key file or base64 (else PPE_PRIVATE_KEY / keys/ppe_hq.key)")
    parser.add_argument("--out", default=".", help="output directory for plaintext")
    parser.add_argument("--recursive", action="store_true", help="walk sub-directories")
    args = parser.parse_args()

    private_key = load_private_key(args.key)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            it = p.rglob("*") if args.recursive else p.iterdir()
            targets.extend(f for f in it if f.is_file() and is_encrypted(f))
        elif p.is_file():
            targets.append(p)
        else:
            print(f"[skip] not found: {p}", file=sys.stderr)

    if not targets:
        raise SystemExit("No .enc files to decrypt.")

    print("[warn] writing plaintext (identifiable) images to disk — secure this folder.",
          file=sys.stderr)
    count = 0
    for src in targets:
        plaintext, ext = decrypt_file_to_bytes(src, private_key)
        # photo.jpg.enc -> photo.jpg ; fall back to stored ext if needed.
        name = src.name[: -len(ENC_SUFFIX)] if src.name.endswith(ENC_SUFFIX) else src.name
        if not Path(name).suffix and ext:
            name = f"{name}.{ext}"
        dest = out_dir / name
        dest.write_bytes(plaintext)
        count += 1
        print(f"[dec] {src.name} -> {dest}")
    print(f"[done] decrypted {count} file(s) into {out_dir}")


if __name__ == "__main__":
    main()
