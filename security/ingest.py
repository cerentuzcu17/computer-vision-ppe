"""Encrypt-on-ingest: seal site images the moment they arrive.

Point this at the directory where photos land from the şantiye. Every image is
encrypted to HQ's public key, EXIF-stripped, and the plaintext original is
securely wiped — so a raw identifiable photo never lingers in the ingest area.

    # one-shot sweep of whatever is already there
    uv run security/ingest.py --pub keys/ppe_hq.pub --dir /capture/incoming

    # keep watching and encrypt new arrivals as they land
    uv run security/ingest.py --pub keys/ppe_hq.pub --dir /capture/incoming --watch

    # encrypt into a separate output dir instead of in place
    uv run security/ingest.py --pub keys/ppe_hq.pub --dir /capture/incoming \
        --out /capture/encrypted

Watch mode uses the `watchdog` package if installed, and otherwise falls back
to a simple polling loop (no extra dependency required).

This tool needs only the PUBLIC key — it cannot decrypt anything.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from security.encrypt import IMAGE_EXTS, encrypt_one  # noqa: E402
from security.envelope import ENC_SUFFIX, load_public_key  # noqa: E402


def _is_target(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and not path.name.endswith(ENC_SUFFIX)
    )


def _stable(path: Path, settle: float = 0.5) -> bool:
    """True once a file has stopped growing — avoids grabbing a half-written upload."""
    try:
        first = path.stat().st_size
        time.sleep(settle)
        return path.stat().st_size == first
    except OSError:
        return False


def _seal(path: Path, public_key, out_dir: Path | None, keep_original: bool, strip: bool) -> None:
    try:
        if not _stable(path):
            return  # still being written; catch it on the next pass
        dest = encrypt_one(path, public_key, out_dir, keep_original, strip)
        print(f"[ingest] sealed {path.name} -> {dest.name}", flush=True)
    except FileNotFoundError:
        pass  # file vanished (moved/deleted) between detection and sealing
    except Exception as exc:  # never let one bad file kill the watcher
        print(f"[ingest][error] {path.name}: {exc}", file=sys.stderr, flush=True)


def sweep(directory: Path, public_key, out_dir, keep_original, strip, recursive) -> int:
    it = directory.rglob("*") if recursive else directory.iterdir()
    count = 0
    for p in sorted(it):
        if _is_target(p):
            _seal(p, public_key, out_dir, keep_original, strip)
            count += 1
    return count


def watch(directory, public_key, out_dir, keep_original, strip, recursive, interval) -> None:
    print(f"[ingest] watching {directory} (Ctrl-C to stop)", flush=True)
    seal = lambda p: _seal(p, public_key, out_dir, keep_original, strip)  # noqa: E731

    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        class Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    p = Path(event.src_path)
                    if _is_target(p):
                        seal(p)

            on_moved = lambda self, e: (  # noqa: E731
                None if e.is_directory or not _is_target(Path(e.dest_path)) else seal(Path(e.dest_path))
            )

        observer = Observer()
        observer.schedule(Handler(), str(directory), recursive=recursive)
        observer.start()
        try:
            while True:
                time.sleep(1)
        finally:
            observer.stop()
            observer.join()
    except ImportError:
        # No watchdog — poll. Track what we've already sealed by filename.
        print("[ingest] watchdog not installed; using polling. "
              "(`uv add watchdog` for event-driven watching.)", flush=True)
        seen: set[str] = set()
        try:
            while True:
                for p in (directory.rglob("*") if recursive else directory.iterdir()):
                    if _is_target(p) and str(p) not in seen:
                        seal(p)
                        seen.add(str(p))
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[ingest] stopped.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypt site images on ingest")
    parser.add_argument("--pub", default="keys/ppe_hq.pub",
                        help="HQ public key (.pub) [default: keys/ppe_hq.pub]")
    parser.add_argument("--dir", required=True, help="ingest directory to seal")
    parser.add_argument("--out", default=None, help="output dir (default: in place)")
    parser.add_argument("--watch", action="store_true", help="keep watching for new files")
    parser.add_argument("--recursive", action="store_true", help="include sub-directories")
    parser.add_argument("--keep-original", action="store_true",
                        help="do not delete plaintext after sealing")
    parser.add_argument("--keep-metadata", action="store_true",
                        help="do not strip EXIF before sealing")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="poll interval seconds (polling fallback only)")
    args = parser.parse_args()

    directory = Path(args.dir)
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")
    if not Path(args.pub).exists():
        raise SystemExit(
            f"Public key not found: {args.pub}\n"
            "Generate the keypair first (once, at HQ):  uv run security/keygen.py"
        )
    public_key = load_public_key(args.pub)
    out_dir = Path(args.out) if args.out else None
    strip = not args.keep_metadata

    n = sweep(directory, public_key, out_dir, args.keep_original, strip, args.recursive)
    print(f"[ingest] initial sweep sealed {n} image(s).", flush=True)

    if args.watch:
        watch(directory, public_key, out_dir, args.keep_original, strip,
              args.recursive, args.interval)


if __name__ == "__main__":
    main()
