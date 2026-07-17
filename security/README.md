# security/ — site image confidentiality (KVKK)

Photos from the şantiye contain workers' faces (personal data under KVKK). This
module encrypts each image **at the point of capture**, so a raw identifiable
photo never travels the network or sits on a server — or on the capture device —
in the clear.

## Quick start (local / demo) — read this first

Four commands. Do them in order and you can't really get it wrong.

```bash
# 0. one-time: install deps
uv sync

# 1. one-time: make the keypair (creates keys/ — already gitignored)
uv run security/keygen.py

# 2. encrypt an image  (defaults to keys/ppe_hq.pub, so no flags needed)
uv run security/encrypt.py dataset/photo.jpg
#    -> makes dataset/photo.jpg.enc  and removes the plaintext original

# 3. run PPE detection straight on the ENCRYPTED file — no manual decrypt step
uv run core/pose_prototype.py --source dataset/photo.jpg.enc --show
```

That's the whole loop. Step 3 is the good demo moment: the model runs directly
on an encrypted file because the pipeline decrypts it in memory using
`keys/ppe_hq.key`. Nothing identifiable is written to disk.

### Showcasing without destroying your test images

`encrypt.py` deletes the plaintext by default (that's the KVKK-correct
behavior). When you're just testing or presenting and want to keep your
originals, encrypt into a separate folder and keep them:

```bash
uv run security/encrypt.py dataset/*.jpg --out demo_encrypted --keep-original
# originals stay in dataset/, encrypted copies go to demo_encrypted/
```

To show the decrypted image explicitly (e.g. "see, it's the same photo back"):

```bash
uv run security/decrypt.py demo_encrypted/photo.jpg.enc --out demo_cleartext
```

### The two rules that keep you out of trouble

1. **Never commit `keys/`.** It's gitignored already — just don't force it in.
   If `git status` ever shows a `.key`, stop and remove it before committing.
2. **Back up `keys/ppe_hq.key` once**, somewhere off your laptop (a password
   manager entry is perfect). Lose that file and the encrypted images are gone
   for good — there is no recovery. This is the one mistake that actually hurts.

If a command says `Public key not found`, you skipped step 1 — run
`uv run security/keygen.py`.

## The design in one line

Encrypt on-site to HQ's **public** key; only HQ's **private** key can decrypt.

We use **libsodium sealed boxes** (via PyNaCl): X25519 key agreement +
XSalsa20-Poly1305 authenticated encryption. A sealed box generates a fresh
ephemeral keypair for every image, so each photo effectively gets its own
one-time symmetric key wrapped to HQ's public key — textbook envelope
encryption, without us hand-rolling any crypto.

```
  ŞANTİYE (site device)                         HQ (controlled env)
  ─────────────────────                         ───────────────────
  capture photo                                 keys/ppe_hq.key  (PRIVATE)
       │                                                 ▲
       ▼                                                 │
  strip EXIF (GPS/time/serial)                           │
       │                                                 │
       ▼                                                 │
  sealed_box.encrypt(img, HQ_PUBLIC) ──►  photo.jpg.enc ─┘
       │                                    (safe to move / store)
       ▼
  securely delete plaintext
```

**Why asymmetric matters here:** the site device holds *only the public key*.
If a device is lost, stolen, or seized on-site, nobody can decrypt anything —
not even the images that device captured. A symmetric key on the device would
be a single point of failure.

## Files

| File          | Runs at | Purpose                                                        |
|---------------|---------|----------------------------------------------------------------|
| `keygen.py`   | HQ once | Generate the X25519 keypair (`keys/ppe_hq.pub` + `ppe_hq.key`). |
| `encrypt.py`  | Site    | Encrypt images to the public key; strip EXIF; wipe plaintext.   |
| `ingest.py`   | Site    | Seal images automatically as they land in a capture dir (`--watch`).|
| `decrypt.py`  | HQ      | Export plaintext back (use sparingly — recreates personal data).|
| `envelope.py` | both    | The crypto core + `.enc` file format (import, don't run).       |

Tests live in `tests/test_security.py` (`uv run pytest`): round-trip, wrong-key
rejection, tamper rejection, EXIF stripping, and the `.enc` format.

The inference pipeline (`core/pose_prototype.py`) imports `envelope` directly
and decrypts `.enc` images **in memory** — plaintext never hits disk during
normal processing.

## Usage

```bash
uv sync                                   # installs pynacl

# 1. HQ: make the keypair (once). Ship ONLY the .pub to sites.
uv run security/keygen.py

# 2. Site: encrypt captured images (deletes plaintext, strips EXIF by default)
uv run security/encrypt.py --pub keys/ppe_hq.pub --recursive /capture/incoming

#    …or seal images automatically as they arrive:
uv run security/ingest.py --pub keys/ppe_hq.pub --dir /capture/incoming --watch

# 3. HQ: run inference straight on the encrypted files — no manual decrypt step
uv run core/pose_prototype.py --source dataset/photo.jpg.enc --save observe/out.jpg

# (optional) HQ: export plaintext into a secured folder, if you must
uv run security/decrypt.py --key keys/ppe_hq.key dataset/photo.jpg.enc --out ./cleartext
```

The private key can also come from `PPE_PRIVATE_KEY` (base64) instead of a file,
which is the right way to inject it from a secret manager / KMS on the HQ box.

## `.enc` file format

```
MAGIC "PPEENC" (6B) | VERSION (1B) | EXT_LEN (1B) | EXT (utf-8) | SEALED_BOX
```

The sealed box is `ephemeral_pubkey || nonce || ciphertext+MAC`. Any tampering
fails the Poly1305 MAC, so decryption raises instead of returning corrupt pixels.

## Porting the encrypt step to other capture devices

The capture device only needs libsodium's `crypto_box_seal` and HQ's public
key — no repo code required. libsodium is available on every platform:

- **Android/Kotlin/Java** — Lazysodium: `SealedBox.cryptoBoxSeal(...)`
- **iOS/Swift** — Swift-Sodium: `sodium.box.seal(message:recipientPublicKey:)`
- **C/embedded** — libsodium: `crypto_box_seal()`
- **JS/Node** — libsodium-wrappers: `crypto_box_seal()`

Produce the sealed-box bytes on-device, then prepend the same 8-byte+ext header
so HQ tooling recognizes the file (or ship raw sealed-box bytes and add the
header on ingest). Keep the format version in sync with `envelope.VERSION`.

## KVKK notes (encryption is necessary, not sufficient)

**One-line summary for the supervisor:** worker photos are encrypted the moment
they're captured, using a key that lives only at HQ, so raw identifiable images
never sit unprotected in transit or storage; EXIF (location/time/device) is
stripped; and only HQ can recover the originals. Encryption is the technical
safeguard KVKK asks for — the items below are the surrounding process controls
that make it a complete answer, not just a crypto library.


- **Lawful basis + notice.** Encryption protects the data; you still need a
  lawful basis to process worker images and must inform workers (aydınlatma
  metni).
- **Data minimization.** EXIF stripping (default on) removes GPS, timestamps,
  and device serials that could re-identify a worker or pinpoint the site.
- **Key custody.** The private key is the whole ballgame. Keep it in a KMS/HSM
  or secret manager, never in git, never on a site device. `.gitignore` already
  blocks `keys/`, `*.key`, `*.pub`, and `*.enc`.
- **Access logging.** Log every decryption/export (who, when, which images).
  KVKK expects an access trail for personal data.
- **Retention.** Delete originals when the lawful basis expires. If you don't
  need recoverable originals at all, consider irreversibly anonymizing (face
  blur) instead of encrypting — that takes the data out of KVKK scope entirely.
- **Key rotation** re-encrypts nothing automatically: old `.enc` files still
  need the old private key, so archive retired private keys securely rather than
  destroying them while encrypted data still exists.
