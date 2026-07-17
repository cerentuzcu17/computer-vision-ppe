"""Site-side encryption: run this where photos are captured (the şantiye).

Encrypts each image to HQ's PUBLIC key and writes a <name>.enc file. By
default it strips EXIF first (GPS, timestamps, device serials — all of which
can re-identify a worker or site) and then securely removes the plaintext
original, so nothing identifiable is left on the device.

    # one file
    uv run security/encrypt.py --pub keys/ppe_hq.pub photo.jpg

    # a whole capture folder, in place
    uv run security/encrypt.py --pub keys/ppe_hq.pub --recursive /capture/incoming

Flags:
    --pub            path to HQ public key (required)
    --out            output dir (default: alongside each source file)
    --recursive      walk sub-directories
    --keep-original  do NOT delete the plaintext after encrypting (default: delete)
    --keep-metadata  do NOT strip EXIF before encrypting (default: strip)

This tool only ever needs the PUBLIC key. It cannot decrypt anything — by
design, so a lost/stolen site device exposes no images.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as a plain script (adds repo root so `security` imports work).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from security.envelope import ENC_SUFFIX, encrypt_bytes, load_public_key  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
JPEG_EXTS = {".jpg", ".jpeg"}


def strip_exif(data: bytes, ext: str) -> bytes:
    """Losslessly strip identifying metadata (EXIF/XMP) from JPEG bytes.

    Works directly on the byte stream — removes every APP1 segment (which is
    where EXIF's GPS/timestamp/device fields and XMP live) while leaving the
    image scan data untouched, so pixels are bit-for-bit preserved. Non-JPEG
    inputs pass through unchanged (PNG/BMP/WebP carry little identifying EXIF;
    add handling here if your source format embeds it).
    """
    if ext.lower() not in JPEG_EXTS:
        return data
    if data[:2] != b"\xff\xd8":  # not a JPEG despite the extension
        return data

    out = bytearray(data[:2])  # SOI
    i, n = 2, len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            out.extend(data[i:])  # malformed; keep the rest verbatim
            break
        marker = data[i + 1]
        if marker == 0xDA:  # Start Of Scan — image data follows to EOI
            out.extend(data[i:])
            break
        if 0xD0 <= marker <= 0xD9 or marker == 0x01:  # standalone, no length
            out.extend(data[i : i + 2])
            i += 2
            continue
        seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
        if marker != 0xE1:  # drop APP1 (EXIF/XMP); keep everything else
            out.extend(data[i : i + 2 + seg_len])
        i += 2 + seg_len
    return bytes(out)


def secure_delete(path: Path) -> None:
    """Best-effort secure delete: overwrite then unlink.

    On SSDs/journaled filesystems overwrite-in-place is not guaranteed to
    destroy the data, but it removes the obvious plaintext copy. For strong
    guarantees, capture directly onto an encrypted volume.
    """
    try:
        length = path.stat().st_size
        with path.open("r+b", buffering=0) as fh:
            fh.write(os.urandom(length))
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        # Don't abort a batch over one undeletable file — but make it loud, since
        # a leftover plaintext original is a KVKK problem the operator must fix.
        print(f"[warn] could not delete plaintext {path}: {exc}. "
              "Remove it manually / capture onto an encrypted volume.",
              file=sys.stderr)


def encrypt_one(
    src: Path, public_key, out_dir: Path | None, keep_original: bool, strip: bool
) -> Path:
    ext = src.suffix.lower().lstrip(".")
    data = src.read_bytes()
    if strip:
        data = strip_exif(data, src.suffix)
    blob = encrypt_bytes(data, public_key, ext=ext)

    dest_dir = out_dir if out_dir is not None else src.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (src.name + ENC_SUFFIX)
    dest.write_bytes(blob)

    if not keep_original:
        secure_delete(src)
    return dest


def iter_images(root: Path, recursive: bool):
    it = root.rglob("*") if recursive else root.iterdir()
    for p in it:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypt site images to HQ's public key")
    parser.add_argument("paths", nargs="+", help="image file(s) or directory(ies)")
    parser.add_argument("--pub", default="keys/ppe_hq.pub",
                        help="path to HQ public key (.pub) [default: keys/ppe_hq.pub]")
    parser.add_argument("--out", default=None, help="output dir (default: next to source)")
    parser.add_argument("--recursive", action="store_true", help="walk sub-directories")
    parser.add_argument("--keep-original", action="store_true",
                        help="do not delete plaintext after encrypting")
    parser.add_argument("--keep-metadata", action="store_true",
                        help="do not strip EXIF before encrypting")
    args = parser.parse_args()

    if not Path(args.pub).exists():
        raise SystemExit(
            f"Public key not found: {args.pub}\n"
            "Generate the keypair first (once, at HQ):  uv run security/keygen.py"
        )
    public_key = load_public_key(args.pub)
    out_dir = Path(args.out) if args.out else None
    strip = not args.keep_metadata

    targets: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            targets.extend(iter_images(p, args.recursive))
        elif p.is_file():
            targets.append(p)
        else:
            print(f"[skip] not found: {p}", file=sys.stderr)

    if not targets:
        raise SystemExit("No images to encrypt.")

    count = 0
    for src in targets:
        dest = encrypt_one(src, public_key, out_dir, args.keep_original, strip)
        count += 1
        print(f"[enc] {src.name} -> {dest.name}")
    print(f"[done] encrypted {count} image(s).")


if __name__ == "__main__":
    main()
