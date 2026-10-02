"""
.xorm Format Specification — XorZen Model Format v1 & v2
=========================================================
A .xorm file is a ZIP archive with a validated internal structure.

v1 (plain, legacy):
  .xorm_magic         — Version sentinel "XORMV001"
  manifest.json       — Model identity, params, dtype, timestamps
  arch.json           — Architecture hyperparameters (config overrides)
  features.json       — Per-turn feature schema (names, indices, descriptions)
  weights.pt          — PyTorch state_dict (pickle, PLAINTEXT)
  weights_manifest.json — Layer names, shapes, dtypes, param counts (JSON, for tooling)
  meta.json           — Training metrics (accuracy, loss, etc.)

v2 (encrypted + signed + licensed — see xorm_crypto.py for details):
  .xorm_magic         — Version sentinel "XORMV002"
  manifest.json       — Model identity (includes encrypted=true, license fields)
  arch.json
  features.json
  weights_manifest.json
  weights.pt.enc      — AES-256-GCM encrypted weights.pt
  weights.pt.nonce    — 12-byte AES-GCM nonce
  weights.pt.tag      — 16-byte AES-GCM authentication tag
  license.json        — owner, allowed_uses, expiry, model_id
  signature.bin       — HMAC-SHA256 over (manifest+arch+weights.pt.enc+
                         nonce+tag+license.json)
  meta.json

The XormReader auto-detects v1 vs v2 by reading .xorm_magic. v2 files
require the `cryptography` library and the decryption key embedded in
xorm_crypto.py.
"""

import zipfile
import json
import io
import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch

from xorzen.inference.xorm_crypto import (
    XORM_MAGIC_V1,
    XORM_MAGIC_V2,
    XORM_EXTENSION,
    License,
    _encrypt_weights,
    _decrypt_weights,
    _compute_signature,
    verify_xorm_signature,
    CRYPTO_AVAILABLE,
)

# Backwards-compat: re-export the v1 magic for any code that imports it
XORM_MAGIC = XORM_MAGIC_V1


def _param_suffix(n: int) -> str:
    """Convert raw param count to a human-readable suffix: 598234 → '598K'."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B".replace(".0B", "B")
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{round(n / 1000)}K"
    return str(n)


def _build_weights_manifest(state_dict: dict) -> dict:
    """
    Build a JSON-serialisable manifest of every tensor in a state_dict.
    Used by the VS Code .xorm viewer to show layers without parsing pickle.
    """
    layers = []
    total_params = 0
    total_bytes = 0

    for name, tensor in state_dict.items():
        numel = tensor.numel()
        dtype_str = str(tensor.dtype).replace("torch.", "")
        bytes_per_el = tensor.element_size()
        size_bytes = numel * bytes_per_el

        layers.append({
            "name": name,
            "shape": list(tensor.shape),
            "dtype": dtype_str,
            "params": numel,
            "size_kb": round(size_bytes / 1024, 2),
        })
        total_params += numel
        total_bytes += size_bytes

    return {
        "layers": layers,
        "total_params": total_params,
        "total_size_mb": round(total_bytes / (1024 * 1024), 3),
        "param_suffix": _param_suffix(total_params),
    }


class XormWriter:
    """Build and save a .xorm file (v1 plain or v2 encrypted)."""

    def __init__(
        self,
        model_name: str,
        family: str,
        role: str,
        arch_config: Dict[str, Any],
        feature_schema: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
        encrypt: bool = False,
        license: Optional[License] = None,
    ):
        self.model_name = model_name
        self.family = family
        self.role = role
        self.arch_config = arch_config
        self.feature_schema = feature_schema
        self.meta = meta or {}
        self.encrypt = encrypt
        self.license = license

        if encrypt and not CRYPTO_AVAILABLE:
            raise ImportError(
                "cryptography library required for encrypted .xorm v2. "
                "Install with: pip install cryptography"
            )
        if encrypt and license is None:
            # Default license: owner-only, inference+research, no expiry
            self.license = License(
                owner="FRAZIYM / Akik Forazi",
                model_id=model_name,
                allowed_uses=["inference", "research"],
                expiry=None,
                issued_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                issuer="FRAZIYM / Akik Forazi",
            )

    def save(self, model: torch.nn.Module, output_path: str) -> str:
        path = Path(output_path)
        if path.suffix != XORM_EXTENSION:
            path = path.with_suffix(XORM_EXTENSION)

        state_dict = model.state_dict()
        total_params = sum(p.numel() for p in model.parameters())
        dtype_str = str(next(model.parameters()).dtype).replace("torch.", "")

        manifest = {
            "xorm_version": "2.0" if self.encrypt else "1.0",
            "model_name": self.model_name,
            "family": self.family,
            "role": self.role,
            "total_params": total_params,
            "param_suffix": _param_suffix(total_params),
            "dtype": dtype_str,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "author": "FRAZIYM / AKIK FARAJI",
            "project": "XorVec/XorZen",
            "encrypted": self.encrypt,
        }

        weights_manifest = _build_weights_manifest(state_dict)

        weights_buf = io.BytesIO()
        torch.save(state_dict, weights_buf)
        weights_bytes = weights_buf.getvalue()

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if self.encrypt:
                # v2: encrypted + signed + licensed
                # Update manifest with license metadata
                self.license.model_id = self.model_name
                manifest["license_owner"] = self.license.owner
                manifest["license_expiry"] = self.license.expiry
                manifest["license_allowed_uses"] = self.license.allowed_uses

                manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
                arch_bytes = json.dumps(self.arch_config, indent=2).encode("utf-8")
                features_bytes = json.dumps(self.feature_schema, indent=2).encode("utf-8")
                weights_manifest_bytes = json.dumps(weights_manifest, indent=2).encode("utf-8")
                meta_bytes = json.dumps(self.meta, indent=2).encode("utf-8")
                license_bytes = json.dumps(self.license.to_dict(), indent=2).encode("utf-8")

                # Encrypt weights
                ciphertext, nonce, tag = _encrypt_weights(weights_bytes, self.model_name)

                # Compute HMAC signature
                signature = _compute_signature(
                    manifest_bytes=manifest_bytes,
                    arch_bytes=arch_bytes,
                    encrypted_weights=ciphertext,
                    nonce=nonce,
                    tag=tag,
                    license_bytes=license_bytes,
                    model_id=self.model_name,
                )

                zf.writestr(".xorm_magic", XORM_MAGIC_V2)
                zf.writestr("manifest.json", manifest_bytes)
                zf.writestr("arch.json", arch_bytes)
                zf.writestr("features.json", features_bytes)
                zf.writestr("weights_manifest.json", weights_manifest_bytes)
                zf.writestr("weights.pt.enc", ciphertext)
                zf.writestr("weights.pt.nonce", nonce)
                zf.writestr("weights.pt.tag", tag)
                zf.writestr("license.json", license_bytes)
                zf.writestr("signature.bin", signature)
                zf.writestr("meta.json", meta_bytes)
            else:
                # v1: plain (legacy)
                zf.writestr(".xorm_magic", XORM_MAGIC_V1)
                zf.writestr("manifest.json", json.dumps(manifest, indent=2))
                zf.writestr("arch.json", json.dumps(self.arch_config, indent=2))
                zf.writestr("features.json", json.dumps(self.feature_schema, indent=2))
                zf.writestr("weights_manifest.json", json.dumps(weights_manifest, indent=2))
                zf.writestr("weights.pt", weights_bytes)
                zf.writestr("meta.json", json.dumps(self.meta, indent=2))

        return str(path)


class XormReader:
    """Load a .xorm file (v1 plain or v2 encrypted)."""

    def __init__(self, path: str, intended_use: str = "inference"):
        self.path = Path(path)
        self.intended_use = intended_use
        if not self.path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self._zf = zipfile.ZipFile(self.path, "r")
        self._validate()
        self._is_v2 = self._magic.startswith("XORMV002")
        self._license: Optional[License] = None
        if self._is_v2:
            self._verify_v2()

    def _validate(self):
        names = self._zf.namelist()
        if ".xorm_magic" not in names:
            raise ValueError("Not a valid .xorm file: missing magic sentinel")
        self._magic = self._zf.read(".xorm_magic").decode("utf-8")
        if not (self._magic.startswith("XORMV001") or self._magic.startswith("XORMV002")):
            raise ValueError(f"Unknown .xorm version: {self._magic.strip()}")

        if self._magic.startswith("XORMV001"):
            for required in ("manifest.json", "arch.json", "weights.pt"):
                if required not in names:
                    raise ValueError(f"Corrupt .xorm v1: missing {required}")
        elif self._magic.startswith("XORMV002"):
            for required in ("manifest.json", "arch.json", "weights.pt.enc",
                              "weights.pt.nonce", "weights.pt.tag",
                              "license.json", "signature.bin"):
                if required not in names:
                    raise ValueError(f"Corrupt .xorm v2: missing {required}")

    def _verify_v2(self):
        """Verify HMAC signature + license for a v2 .xorm file."""
        ok, reason, license_obj = verify_xorm_signature(str(self.path))
        if not ok:
            raise ValueError(f".xorm v2 signature verification failed: {reason}")
        self._license = license_obj

        # Validate license for the intended use
        license_ok, license_reason = license_obj.validate(intended_use=self.intended_use)
        if not license_ok:
            raise ValueError(f".xorm v2 license violation: {license_reason}")

    @property
    def is_encrypted(self) -> bool:
        return self._is_v2

    @property
    def license(self) -> Optional[License]:
        if not self._is_v2:
            return None
        if self._license is None:
            try:
                self._license = License.from_dict(
                    json.loads(self._zf.read("license.json"))
                )
            except Exception:
                pass
        return self._license

    @property
    def manifest(self) -> Dict[str, Any]:
        return json.loads(self._zf.read("manifest.json"))

    @property
    def arch_config(self) -> Dict[str, Any]:
        return json.loads(self._zf.read("arch.json"))

    @property
    def feature_schema(self) -> Dict[str, Any]:
        try:
            return json.loads(self._zf.read("features.json"))
        except KeyError:
            return {}

    @property
    def weights_manifest(self) -> Dict[str, Any]:
        """Layer names, shapes, dtypes, param counts — safe for non-Python tooling."""
        try:
            return json.loads(self._zf.read("weights_manifest.json"))
        except KeyError:
            return {"layers": [], "total_params": 0, "total_size_mb": 0}

    @property
    def meta(self) -> Dict[str, Any]:
        try:
            return json.loads(self._zf.read("meta.json"))
        except KeyError:
            return {}

    def load_state_dict(self, map_location="cpu") -> dict:
        if self._is_v2:
            # Decrypt + verify (verification already done in __init__)
            from xorzen.inference.xorm_crypto import _decrypt_weights
            manifest = json.loads(self._zf.read("manifest.json"))
            model_id = manifest.get("model_name", "unnamed")
            ciphertext = self._zf.read("weights.pt.enc")
            nonce = self._zf.read("weights.pt.nonce")
            tag = self._zf.read("weights.pt.tag")
            try:
                plaintext_bytes = _decrypt_weights(ciphertext, nonce, tag, model_id)
            except Exception as e:
                raise ValueError(f".xorm decryption failed (tampering or wrong key): {e}")
            buf = io.BytesIO(plaintext_bytes)
            return torch.load(buf, map_location=map_location, weights_only=True)
        else:
            # v1: plain
            weights_bytes = self._zf.read("weights.pt")
            buf = io.BytesIO(weights_bytes)
            return torch.load(buf, map_location=map_location, weights_only=True)

    def close(self):
        self._zf.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
