"""
xorzen.model.ssm — DEAD CODE (kept as placeholder for API compatibility).

Historical note
---------------
This file previously contained a `S4DKernel` class (complex-diagonal S4D SSM)
and an `SSMBlock` wrapper. As of v1.0.1 it is **dead code** — no module in the
codebase imports `S4DKernel` or `SSMBlock`. The actual SSM used by the HASS
block is `SSMPathway` in `xorzen/model/components/hass_block.py`, which uses a
**Mamba-style real diagonal A** (not complex) via the helpers in
`xorzen/model/components/ssm_scan.py` (ZOH discretization + chunked/parallel
scan).

Why keep this file at all?
--------------------------
We keep a minimal stub so that `from xorzen.model.ssm import S4DKernel` raises
an **actionable ImportError** instead of a confusing `ModuleNotFoundError`.
If you were depending on `S4DKernel`, switch to:

    from xorzen.model.components.hass_block import SSMPathway
    from xorzen.model.components.ssm_scan import discretize_zoh, select_scan

See `docs/parity/SSM_INVESTIGATION.md` for the architectural reasoning.
"""

from __future__ import annotations
import warnings

__all__: list[str] = []

# Emit a DeprecationWarning on import so stale users notice, but do NOT raise —
# we don't want to break `import xorzen.model` just because this stub exists.
warnings.warn(
    "xorzen.model.ssm is dead code (S4DKernel / SSMBlock removed in v1.0.1). "
    "The production SSM is xorzen.model.components.hass_block.SSMPathway. "
    "See docs/parity/SSM_INVESTIGATION.md.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    """Attribute lookup hook — only triggers if someone actually tries to use
    a removed name. Lets `import xorzen.model.ssm` succeed (with warning)
    while `from xorzen.model.ssm import S4DKernel` raises ImportError."""
    if name in ("S4DKernel", "SSMBlock"):
        raise ImportError(
            f"{name} has been removed from xorzen.model.ssm (dead code in v1.0.1). "
            f"The production SSM is xorzen.model.components.hass_block.SSMPathway "
            f"(Mamba-style real diagonal A via xorzen.model.components.ssm_scan). "
            f"See docs/parity/SSM_INVESTIGATION.md."
        )
    raise AttributeError(f"module 'xorzen.model.ssm' has no attribute {name!r}")
