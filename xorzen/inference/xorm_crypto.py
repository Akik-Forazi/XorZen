"""
.xorm Encryption & Licensing — XorZen Model Format v1.1
=========================================================

Adds AES-256-GCM weight encryption + HMAC-SHA256 tamper detection +
license file support to the .xorm format.

THREAT MODEL — what this protects against:
  ✅ Casual extraction: someone unzipping the .xorm and running
     ``torch.load('weights.pt')`` to get the model weights. After this
     change, weights.pt is AES-256-GCM encrypted and unreadable without
     the key embedded in the xorzen runtime.
  ✅ Tampering: modifying manifest.json, arch.json, or the encrypted
     weights to change model identity or bypass safety policies. The
     HMAC-SHA256 signature covers all of these and is verified at load
     time.
  ✅ License violation: loading a model in a context that violates the
     embedded license (e.g. past expiry, wrong owner, disallowed use).
     The runtime checks the license before decrypting weights.
  ⚠️ Determined extraction: a reverse engineer with Python expertise can
     extract the AES key from the xorzen runtime source code (the key
     must live somewhere in the code to be usable). This raises the bar
     significantly but is not a mathematical guarantee.
  ❌ Uncensoring / de-safety-training: once weights are decrypted into
     RAM for inference, a determined attacker can dump them to a plain
     file and fine-tune away any safety training. This is fundamentally
     impossible to prevent for any software-based DRM — the only real
     defense is keeping the model behind an API.

FORMAT v1.1 (encrypted) vs v1.0 (plain):
  v1.0 .xorm contains:
    .xorm_magic          — "XORMV001\n"
    manifest.json
    arch.json
    features.json
    weights_manifest.json
    weights.pt           — raw torch.save() pickle
    meta.json

  v1.1 .xorm adds (alongside v1.0 files, backwards-compatible):
    .xorm_magic          — "XORMV002\n"  (version bump)
    manifest.json
    arch.json
    features.json
    weights_manifest.json
    weights.pt.enc       — AES-256-GCM encrypted weights.pt
    weights.pt.nonce     — 12-byte AES-GCM nonce
    weights.pt.tag       — 16-byte AES-GCM authentication tag
    license.json         — owner, allowed_uses, expiry, model_id
    signature.bin        — HMAC-SHA256 over (manifest+arch+weights.pt.enc+
                           weights.pt.nonce+weights.pt.tag+license.json)
    meta.json

  The reader auto-detects v1 vs v2 by reading .xorm_magic. v1 files load
  without encryption; v2 files require the decryption key.

KEY MANAGEMENT:
  The AES-256 key is derived from a master secret embedded in the xorzen
  runtime via HKDF-SHA256. The master secret is split across multiple
  constants in this file to make casual grep-based extraction harder
  (NOT a real security measure — just obscurity). For real product-grade
  security, use a hardware security module (HSM) or a license server.

USAGE:
  # Save an encrypted .xorm
  writer = XormWriter(model_name="zero_50M_v1", family="zero", role="lm",
                      arch_config={...}, feature_schema={...},
                      encrypt=True, license={
                          "owner": "Akik Forazi",
                          "allowed_uses": ["inference", "research"],
                          "expiry": "2027-12-31",
                      })
  writer.save(model, "model.xorm")

  # Load (auto-detects v1 vs v2, verifies signature + license)
  reader = XormReader("model.xorm")
  state_dict = reader.load_state_dict()  # raises if signature/license fails
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Crypto dependencies
# ---------------------------------------------------------------------------
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    AESGCM = None
    HKDF = None
    hashes = None


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------
XORM_MAGIC_V1 = "XORMV001\n"   # unencrypted (legacy)
XORM_MAGIC_V2 = "XORMV002\n"   # encrypted + signed + licensed
XORM_EXTENSION = ".xorm"


# ---------------------------------------------------------------------------
# Master secret — split across constants for light obfuscation.
# NOTE: This is NOT real security. It just makes casual grep harder.
# To rotate: change all three _KEY_PART_* constants and re-encrypt all models.
# ---------------------------------------------------------------------------
_KEY_PART_A = b"\x42\x8f\xa1\xe3\x56\x7c\x09\xde"
_KEY_PART_B = b"\xab\x13\x67\xf0\x8e\x2d\xc5\x91"
_KEY_PART_C = b"\x5a\x3e\xb8\xc4\x07\xf9\xd2\x6e"
_HKDF_SALT = b"xorzen-xorm-v2-salt"
_HKDF_INFO = b"xorzen-xorm-v2-aes-256-gcm"


def _derive_aes_key(model_id: str) -> bytes:
    """Derive a 32-byte AES-256 key from the master secret + model_id.

    The model_id is mixed into the key derivation so that even if two models
    share the same master secret, they get different per-model AES keys.
    This means extracting one model's key doesn't immediately decrypt all
    other models.
    """
    if not CRYPTO_AVAILABLE:
        raise ImportError(
            "cryptography library required for .xorm v2 encryption. "
            "Install with: pip install cryptography"
        )
    master = _KEY_PART_A + _KEY_PART_B + _KEY_PART_C  # 24 bytes
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_HKDF_SALT,
        info=_HKDF_INFO + b":" + model_id.encode("utf-8"),
    )
    return hkdf.derive(master)


# ---------------------------------------------------------------------------
# License
# ---------------------------------------------------------------------------
@dataclass
class License:
    """Model license embedded in the .xorm file."""
    owner: str = "Unknown"
    model_id: str = "unnamed"
    allowed_uses: list = field(default_factory=lambda: ["inference"])
    expiry: Optional[str] = None       # ISO date string "2027-12-31"
    issued_at: Optional[str] = None    # ISO timestamp
    issuer: str = "FRAZIYM / Akik Forazi"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "owner": self.owner,
            "model_id": self.model_id,
            "allowed_uses": self.allowed_uses,
            "expiry": self.expiry,
            "issued_at": self.issued_at,
            "issuer": self.issuer,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "License":
        return cls(
            owner=d.get("owner", "Unknown"),
            model_id=d.get("model_id", "unnamed"),
            allowed_uses=d.get("allowed_uses", ["inference"]),
            expiry=d.get("expiry"),
            issued_at=d.get("issued_at"),
            issuer=d.get("issuer", "FRAZIYM / Akik Forazi"),
            notes=d.get("notes", ""),
        )

    def validate(self, intended_use: str = "inference") -> Tuple[bool, str]:
        """Validate the license against current time + intended use.

        Returns (ok, reason). reason is empty string if ok.
        """
        # Check expiry
        if self.expiry:
            try:
                expiry_date = time.strptime(self.expiry, "%Y-%m-%d")
                now = time.gmtime()
                if now > expiry_date:
                    return False, f"License expired on {self.expiry}"
            except ValueError:
                return False, f"Invalid expiry format: {self.expiry}"

        # Check allowed uses
        if intended_use not in self.allowed_uses:
            return False, (f"License does not allow '{intended_use}'. "
                           f"Allowed: {self.allowed_uses}")

        return True, ""


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------
def _compute_signature(
    manifest_bytes: bytes,
    arch_bytes: bytes,
    encrypted_weights: bytes,
    nonce: bytes,
    tag: bytes,
    license_bytes: bytes,
    model_id: str,
) -> bytes:
    """Compute HMAC-SHA256 over all critical .xorm contents.

    The HMAC key is derived from the same master secret as the AES key
    (via HKDF with a different info string) so that signature verification
    requires the same runtime key. This means an attacker can't forge a
    valid signature without the runtime key, even if they can read all
    the other files.
    """
    if not CRYPTO_AVAILABLE:
        raise ImportError("cryptography library required for .xorm v2 signing")

    master = _KEY_PART_A + _KEY_PART_B + _KEY_PART_C
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_HKDF_SALT,
        info=b"xorzen-xorm-v2-hmac:" + model_id.encode("utf-8"),
    )
    hmac_key = hkdf.derive(master)

    h = hmac.new(hmac_key, digestmod=hashlib.sha256)
    h.update(b"manifest:")
    h.update(manifest_bytes)
    h.update(b"\narch:")
    h.update(arch_bytes)
    h.update(b"\nweights:")
    h.update(encrypted_weights)
    h.update(b"\nnonce:")
    h.update(nonce)
    h.update(b"\ntag:")
    h.update(tag)
    h.update(b"\nlicense:")
    h.update(license_bytes)
    return h.digest()


# ---------------------------------------------------------------------------
# Encryption / decryption helpers
# ---------------------------------------------------------------------------
def _encrypt_weights(state_dict_bytes: bytes, model_id: str) -> Tuple[bytes, bytes, bytes]:
    """Encrypt raw state_dict bytes with AES-256-GCM.

    Returns (ciphertext, nonce, tag).
    """
    if not CRYPTO_AVAILABLE:
        raise ImportError("cryptography library required for .xorm v2 encryption")

    key = _derive_aes_key(model_id)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce, standard for GCM
    # AESGCM.encrypt returns ciphertext+tag concatenated; we split them
    ct_and_tag = aesgcm.encrypt(nonce, state_dict_bytes, associated_data=None)
    ciphertext = ct_and_tag[:-16]
    tag = ct_and_tag[-16:]
    return ciphertext, nonce, tag


def _decrypt_weights(
    ciphertext: bytes,
    nonce: bytes,
    tag: bytes,
    model_id: str,
) -> bytes:
    """Decrypt AES-256-GCM encrypted state_dict bytes.

    Raises ValueError if the tag doesn't verify (tampering detected).
    """
    if not CRYPTO_AVAILABLE:
        raise ImportError("cryptography library required for .xorm v2 decryption")

    key = _derive_aes_key(model_id)
    aesgcm = AESGCM(key)
    # AESGCM.decrypt expects ciphertext+tag concatenated
    ct_and_tag = ciphertext + tag
    try:
        return aesgcm.decrypt(nonce, ct_and_tag, associated_data=None)
    except Exception as e:
        raise ValueError(
            f".xorm weight decryption failed (tampering detected or wrong key): {e}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def is_encrypted_xorm(path: str) -> bool:
    """Quick check: is this a v2 (encrypted) .xorm file?"""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            magic = zf.read(".xorm_magic").decode("utf-8")
        return magic.startswith("XORMV002")
    except Exception:
        return False


def encrypt_existing_xorm(
    input_path: str,
    output_path: str,
    license: License,
) -> str:
    """Convert a v1 (plain) .xorm to a v2 (encrypted+signed+licensed) .xorm.

    Useful for re-packaging existing models without re-training.
    """
    if not CRYPTO_AVAILABLE:
        raise ImportError("cryptography library required: pip install cryptography")

    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.suffix != XORM_EXTENSION:
        output_path = output_path.with_suffix(XORM_EXTENSION)

    with zipfile.ZipFile(input_path, "r") as zf_in:
        names = zf_in.namelist()
        if ".xorm_magic" not in names:
            raise ValueError(f"Not a .xorm file: {input_path}")
        magic = zf_in.read(".xorm_magic").decode("utf-8")
        if magic.startswith("XORMV002"):
            raise ValueError(f"{input_path} is already v2 (encrypted)")
        if not magic.startswith("XORMV001"):
            raise ValueError(f"Unknown .xorm version: {magic.strip()}")

        manifest_bytes = zf_in.read("manifest.json")
        arch_bytes = zf_in.read("arch.json")
        features_bytes = zf_in.read("features.json")
        weights_manifest_bytes = zf_in.read("weights_manifest.json")
        weights_bytes = zf_in.read("weights.pt")
        meta_bytes = zf_in.read("meta.json")

    manifest = json.loads(manifest_bytes)
    model_id = manifest.get("model_name", "unnamed")
    license.model_id = model_id
    if not license.issued_at:
        license.issued_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Encrypt weights
    ciphertext, nonce, tag = _encrypt_weights(weights_bytes, model_id)

    # Build license JSON
    license_bytes = json.dumps(license.to_dict(), indent=2).encode("utf-8")

    # Compute signature
    signature = _compute_signature(
        manifest_bytes=manifest_bytes,
        arch_bytes=arch_bytes,
        encrypted_weights=ciphertext,
        nonce=nonce,
        tag=tag,
        license_bytes=license_bytes,
        model_id=model_id,
    )

    # Update manifest to mark as v2 + record license
    manifest_v2 = dict(manifest)
    manifest_v2["xorm_version"] = "2.0"
    manifest_v2["encrypted"] = True
    manifest_v2["license_owner"] = license.owner
    manifest_v2["license_expiry"] = license.expiry
    manifest_bytes_v2 = json.dumps(manifest_v2, indent=2).encode("utf-8")

    # Re-sign with the updated manifest
    signature = _compute_signature(
        manifest_bytes=manifest_bytes_v2,
        arch_bytes=arch_bytes,
        encrypted_weights=ciphertext,
        nonce=nonce,
        tag=tag,
        license_bytes=license_bytes,
        model_id=model_id,
    )

    # Write v2 .xorm
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf_out:
        zf_out.writestr(".xorm_magic", XORM_MAGIC_V2)
        zf_out.writestr("manifest.json", manifest_bytes_v2)
        zf_out.writestr("arch.json", arch_bytes)
        zf_out.writestr("features.json", features_bytes)
        zf_out.writestr("weights_manifest.json", weights_manifest_bytes)
        zf_out.writestr("weights.pt.enc", ciphertext)
        zf_out.writestr("weights.pt.nonce", nonce)
        zf_out.writestr("weights.pt.tag", tag)
        zf_out.writestr("license.json", license_bytes)
        zf_out.writestr("signature.bin", signature)
        zf_out.writestr("meta.json", meta_bytes)

    return str(output_path)


def verify_xorm_signature(path: str) -> Tuple[bool, str, Optional[License]]:
    """Verify the HMAC signature of a v2 .xorm file.

    Returns (ok, reason, license). If ok=False, reason explains why.
    """
    if not CRYPTO_AVAILABLE:
        return False, "cryptography library not installed", None

    try:
        with zipfile.ZipFile(path, "r") as zf:
            magic = zf.read(".xorm_magic").decode("utf-8")
            if not magic.startswith("XORMV002"):
                return False, f"Not a v2 .xorm (magic: {magic.strip()})", None

            manifest_bytes = zf.read("manifest.json")
            arch_bytes = zf.read("arch.json")
            ciphertext = zf.read("weights.pt.enc")
            nonce = zf.read("weights.pt.nonce")
            tag = zf.read("weights.pt.tag")
            license_bytes = zf.read("license.json")
            stored_signature = zf.read("signature.bin")

        manifest = json.loads(manifest_bytes)
        model_id = manifest.get("model_name", "unnamed")

        expected_signature = _compute_signature(
            manifest_bytes=manifest_bytes,
            arch_bytes=arch_bytes,
            encrypted_weights=ciphertext,
            nonce=nonce,
            tag=tag,
            license_bytes=license_bytes,
            model_id=model_id,
        )

        if not hmac.compare_digest(stored_signature, expected_signature):
            return False, "HMAC signature mismatch — file has been tampered with", None

        license_obj = License.from_dict(json.loads(license_bytes))
        return True, "", license_obj

    except KeyError as e:
        return False, f"Missing required file in .xorm: {e}", None
    except Exception as e:
        return False, f"Signature verification error: {e}", None


def load_encrypted_state_dict(
    path: str,
    intended_use: str = "inference",
    map_location: str = "cpu",
) -> dict:
    """Load + decrypt + verify a v2 .xorm file's state_dict.

    Args:
        path: Path to the .xorm file.
        intended_use: Must match one of the license's allowed_uses.
        map_location: torch.load map_location for the decrypted state_dict.

    Raises:
        ValueError: If signature verification fails, license is invalid,
                    or decryption fails (tampering).
        ImportError: If cryptography library is not installed.

    Returns:
        The decrypted state_dict.
    """
    import torch  # local import — only needed at load time

    # Step 1: verify signature + license
    ok, reason, license_obj = verify_xorm_signature(path)
    if not ok:
        raise ValueError(f".xorm signature verification failed: {reason}")

    # Step 2: check license
    license_ok, license_reason = license_obj.validate(intended_use=intended_use)
    if not license_ok:
        raise ValueError(f".xorm license violation: {license_reason}")

    # Step 3: decrypt weights
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        model_id = manifest.get("model_name", "unnamed")
        ciphertext = zf.read("weights.pt.enc")
        nonce = zf.read("weights.pt.nonce")
        tag = zf.read("weights.pt.tag")

    plaintext_bytes = _decrypt_weights(ciphertext, nonce, tag, model_id)

    # Step 4: deserialize
    buf = io.BytesIO(plaintext_bytes)
    return torch.load(buf, map_location=map_location, weights_only=True)
