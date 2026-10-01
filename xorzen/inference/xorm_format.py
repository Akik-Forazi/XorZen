"""
.xorm Format Specification — XorZen Model Format v1
=====================================================
A .xorm file is a ZIP archive with a validated internal structure.

Internal files:
  .xorm_magic         — Version sentinel "XORMV001"
  manifest.json       — Model identity, params, dtype, timestamps
  arch.json           — Architecture hyperparameters (config overrides)
  features.json       — Per-turn feature schema (names, indices, descriptions)
  weights.pt          — PyTorch state_dict (pickle)
  weights_manifest.json — Layer names, shapes, dtypes, param counts (JSON, for tooling)
  meta.json           — Training metrics (accuracy, loss, etc.)
"""

import zipfile
import json
import io
import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch

XORM_MAGIC = "XORMV001\n"
XORM_EXTENSION = ".xorm"


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
    """Build and save a .xorm file."""

    def __init__(
        self,
        model_name: str,
        family: str,
        role: str,
        arch_config: Dict[str, Any],
        feature_schema: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.family = family
        self.role = role
        self.arch_config = arch_config
        self.feature_schema = feature_schema
        self.meta = meta or {}

    def save(self, model: torch.nn.Module, output_path: str) -> str:
        path = Path(output_path)
        if path.suffix != XORM_EXTENSION:
            path = path.with_suffix(XORM_EXTENSION)

        state_dict = model.state_dict()
        total_params = sum(p.numel() for p in model.parameters())
        dtype_str = str(next(model.parameters()).dtype).replace("torch.", "")

        manifest = {
            "xorm_version": "1.0",
            "model_name": self.model_name,
            "family": self.family,
            "role": self.role,
            "total_params": total_params,
            "param_suffix": _param_suffix(total_params),
            "dtype": dtype_str,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "author": "FRAZIYM / AKIK FARAJI",
            "project": "XorVec/XorZen",
        }

        weights_manifest = _build_weights_manifest(state_dict)

        weights_buf = io.BytesIO()
        torch.save(state_dict, weights_buf)
        weights_bytes = weights_buf.getvalue()

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(".xorm_magic", XORM_MAGIC)
            zf.writestr("manifest.json",         json.dumps(manifest,         indent=2))
            zf.writestr("arch.json",             json.dumps(self.arch_config, indent=2))
            zf.writestr("features.json",         json.dumps(self.feature_schema, indent=2))
            zf.writestr("weights_manifest.json", json.dumps(weights_manifest, indent=2))
            zf.writestr("weights.pt",            weights_bytes)
            zf.writestr("meta.json",             json.dumps(self.meta,        indent=2))

        return str(path)


class XormReader:
    """Load a .xorm file."""

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self._zf = zipfile.ZipFile(self.path, "r")
        self._validate()

    def _validate(self):
        names = self._zf.namelist()
        if ".xorm_magic" not in names:
            raise ValueError("Not a valid .xorm file: missing magic sentinel")
        magic = self._zf.read(".xorm_magic").decode("utf-8")
        if not magic.startswith("XORMV001"):
            raise ValueError(f"Unknown .xorm version: {magic.strip()}")
        for required in ("manifest.json", "arch.json", "weights.pt"):
            if required not in names:
                raise ValueError(f"Corrupt .xorm: missing {required}")

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
        weights_bytes = self._zf.read("weights.pt")
        buf = io.BytesIO(weights_bytes)
        return torch.load(buf, map_location=map_location, weights_only=True)

    def close(self):
        self._zf.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
