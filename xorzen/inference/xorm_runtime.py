"""
XorZen Inference Runtime
========================
Production-grade inference engine for .xorm models.

Features:
  - Single / batch conversation inference
  - Per-turn attribution (which turns drove the grade)
  - Confidence calibration via temperature scaling
  - Half-precision (fp16/bf16) and CPU/GPU device routing
  - Model registry — never loads the same .xorm twice
  - torch.compile() opt-in for max throughput
  - Model warmup for consistent first-call latency
  - Streaming turn-by-turn grading
  - Detailed GradeResult with reasoning trace
"""

import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from xorzen.inference.xorm_format import XormReader
from xorzen.models.greed import GreedModel
from xorzen.config import ConfigFactory, ModelSize

# ──────────────────────────────────────────────
# Grade taxonomy
# ──────────────────────────────────────────────
GRADE_MAP: Dict[int, str] = {
    0: "Basic",
    1: "Standard",
    2: "Premium",
    3: "MUTANT",
    4: "SOTA",
}

# Reverse map
GRADE_IDX: Dict[str, int] = {v: k for k, v in GRADE_MAP.items()}

# ──────────────────────────────────────────────
# Registry — model singleton cache
# ──────────────────────────────────────────────
_REGISTRY: Dict[str, "XormSession"] = {}
_REGISTRY_LOCK = threading.Lock()

# ──────────────────────────────────────────────
# Model family loaders
# ──────────────────────────────────────────────
XORM_FAMILY_REGISTRY = {
    "greed": GreedModel,
}


@dataclass
class TurnAttribution:
    """Per-turn contribution to the final grade."""
    turn_idx: int
    role: str
    content_preview: str          # first 60 chars
    gate_value: float             # greedy CoT gate strength for this turn
    is_key_turn: bool             # True if gate_value > 0.5


@dataclass
class GradeResult:
    """Complete grading output from XorZen-GREED."""
    grade: str                                    # e.g. "SOTA"
    class_idx: int
    confidence: float                             # softmax prob of predicted class
    probabilities: Dict[str, float]               # all 5 class probabilities
    temperature: float                            # calibration temperature used

    # Attribution (populated if explain=True)
    turn_attributions: List[TurnAttribution] = field(default_factory=list)

    # Performance
    latency_ms: float = 0.0
    model_name: str = ""

    def summary(self) -> str:
        """Human-readable one-liner."""
        bar = "█" * int(self.confidence * 20)
        return (
            f"[{self.grade}] {self.confidence:.1%}  {bar}  "
            f"({self.latency_ms:.1f}ms)"
        )

    def probabilities_table(self) -> str:
        lines = []
        for grade in ["Basic", "Standard", "Premium", "MUTANT", "SOTA"]:
            p = self.probabilities.get(grade, 0.0)
            bar = "█" * int(p * 20)
            marker = " ◄" if grade == self.grade else ""
            lines.append(f"  {grade:<10} {p:5.1%}  {bar}{marker}")
        return "\n".join(lines)

    def key_turns(self) -> List[TurnAttribution]:
        return [t for t in self.turn_attributions if t.is_key_turn]


# ──────────────────────────────────────────────
# Session
# ──────────────────────────────────────────────
class XormSession:
    """Active inference session for a loaded .xorm model."""

    def __init__(
        self,
        model: torch.nn.Module,
        reader: XormReader,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        temperature: float = 1.0,
        compiled: bool = False,
    ):
        self.model = model
        self.reader = reader
        self.manifest = reader.manifest
        self.arch_config = reader.arch_config
        self.feature_schema = reader.feature_schema
        self.meta = reader.meta
        self.device = torch.device(device)
        self.dtype = dtype
        self.temperature = temperature

        self.model.to(self.device)
        if dtype in (torch.float16, torch.bfloat16):
            self.model = self.model.to(dtype)

        self.model.eval()

        if compiled:
            try:
                self.model = torch.compile(self.model)
                print("[xorm] Model compiled with torch.compile()")
            except Exception as e:
                print(f"[xorm] torch.compile() not available: {e}")

    # ── Warmup ──────────────────────────────────────────────────────────────
    def warmup(self, num_turns: int = 4, num_runs: int = 3) -> float:
        """
        Run dummy forward passes to warm up PyTorch JIT / CUDA kernels.
        Returns average warmup latency in ms.
        """
        dummy = torch.zeros(1, num_turns, self.arch_config.get("input_dim", 24),
                            device=self.device, dtype=self.dtype)
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                self.model(dummy)
                times.append((time.perf_counter() - t0) * 1000)
        avg_ms = sum(times) / len(times)
        print(f"[xorm] Warmed up in {avg_ms:.1f}ms avg over {num_runs} runs")
        return avg_ms

    # ── Core inference ───────────────────────────────────────────────────────
    @torch.no_grad()
    def _forward(
        self, features: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """Run model forward and return calibrated probabilities."""
        logits = self.model(features)              # [B, num_classes]
        if temperature != 1.0:
            logits = logits / temperature
        return F.softmax(logits, dim=-1)            # [B, num_classes]

    # ── Single conversation ──────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        features: Union[List[List[float]], torch.Tensor],
        temperature: Optional[float] = None,
        explain: bool = False,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> GradeResult:
        """
        Grade a single conversation.

        Args:
            features:    Per-turn features [T, F] as list or tensor.
            temperature: Calibration temperature (lower = more confident).
                         Defaults to session temperature.
            explain:     If True, compute per-turn attributions.
            messages:    Original message dicts (needed for explain previews).

        Returns:
            GradeResult
        """
        t0 = time.perf_counter()
        temp = temperature if temperature is not None else self.temperature

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=self.dtype, device=self.device)
        else:
            features = features.to(device=self.device, dtype=self.dtype)

        if features.dim() == 2:
            features = features.unsqueeze(0)       # [1, T, F]

        probs = self._forward(features, temp).squeeze(0)  # [num_classes]

        class_idx  = int(probs.argmax().item())
        confidence = float(probs[class_idx].item())
        grade      = GRADE_MAP.get(class_idx, "Basic")
        all_probs  = {GRADE_MAP[i]: float(probs[i].item()) for i in range(len(probs))}
        latency_ms = (time.perf_counter() - t0) * 1000

        attributions = []
        if explain:
            attributions = self._explain_turns(features, messages)

        return GradeResult(
            grade=grade,
            class_idx=class_idx,
            confidence=confidence,
            probabilities=all_probs,
            temperature=temp,
            turn_attributions=attributions,
            latency_ms=latency_ms,
            model_name=self.manifest.get("model_name", "xorzen-greed"),
        )

    # ── Batch inference ──────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_batch(
        self,
        batch: List[Union[List[List[float]], torch.Tensor]],
        temperature: Optional[float] = None,
        batch_size: int = 32,
    ) -> List[GradeResult]:
        """
        Grade a list of conversations efficiently.

        Sequences are padded to the longest in each mini-batch so shorter
        conversations don't waste compute on padding.

        Args:
            batch:      List of per-turn feature arrays, each [Ti, F].
            temperature: Calibration temperature.
            batch_size: Mini-batch size.

        Returns:
            List of GradeResult, one per conversation.
        """
        temp = temperature if temperature is not None else self.temperature
        results = []
        t0 = time.perf_counter()

        for i in range(0, len(batch), batch_size):
            chunk = batch[i : i + batch_size]
            # Convert to tensors
            tensors = []
            for f in chunk:
                if not isinstance(f, torch.Tensor):
                    f = torch.tensor(f, dtype=self.dtype)
                tensors.append(f)

            # Dynamic padding to longest in chunk
            max_t = max(t.shape[0] for t in tensors)
            feat_dim = tensors[0].shape[-1]
            padded = torch.zeros(len(tensors), max_t, feat_dim,
                                 dtype=self.dtype, device=self.device)
            for j, t in enumerate(tensors):
                padded[j, : t.shape[0]] = t.to(self.device)

            probs_batch = self._forward(padded, temp)  # [B, num_classes]

            for k in range(len(chunk)):
                probs = probs_batch[k]
                class_idx  = int(probs.argmax().item())
                confidence = float(probs[class_idx].item())
                results.append(GradeResult(
                    grade=GRADE_MAP.get(class_idx, "Basic"),
                    class_idx=class_idx,
                    confidence=confidence,
                    probabilities={GRADE_MAP[i]: float(probs[i].item())
                                   for i in range(len(probs))},
                    temperature=temp,
                    latency_ms=0.0,  # filled below
                    model_name=self.manifest.get("model_name", ""),
                ))

        total_ms = (time.perf_counter() - t0) * 1000
        per_ms   = total_ms / max(len(batch), 1)
        for r in results:
            r.latency_ms = per_ms
        return results

    # ── Turn attribution ─────────────────────────────────────────────────────
    @torch.no_grad()
    def _explain_turns(
        self,
        features: torch.Tensor,        # [1, T, F]
        messages: Optional[List[Dict[str, Any]]],
    ) -> List[TurnAttribution]:
        """
        Estimate which turns most influenced the grade using ablation.
        For each turn, zeroes out its feature vector and measures
        how much the predicted probability drops.
        """
        base_probs   = self._forward(features, self.temperature)    # [1, NC]
        base_cls     = int(base_probs.argmax(dim=-1).item())
        base_p       = float(base_probs[0, base_cls].item())
        num_turns    = features.shape[1]
        attributions = []

        for t in range(num_turns):
            ablated = features.clone()
            ablated[0, t] = 0.0
            abl_probs = self._forward(ablated, self.temperature)
            abl_p     = float(abl_probs[0, base_cls].item())
            # How much the class probability drops when this turn is zeroed
            gate_val  = max(0.0, base_p - abl_p)

            role    = ""
            preview = ""
            if messages and t < len(messages):
                role    = messages[t].get("role", "")
                content = messages[t].get("content") or ""
                preview = content[:60].replace("\n", " ")

            attributions.append(TurnAttribution(
                turn_idx=t,
                role=role,
                content_preview=preview,
                gate_value=gate_val,
                is_key_turn=gate_val > 0.02,
            ))

        # Normalise gate values to [0, 1]
        max_gate = max((a.gate_value for a in attributions), default=1.0) or 1.0
        for a in attributions:
            a.gate_value = round(a.gate_value / max_gate, 4)
            a.is_key_turn = a.gate_value > 0.4

        return attributions

    # ── Streaming grading ────────────────────────────────────────────────────
    @torch.no_grad()
    def stream_grade(
        self,
        features: List[List[float]],
        min_turns: int = 2,
        temperature: Optional[float] = None,
    ) -> List[GradeResult]:
        """
        Grade the conversation at each turn prefix.
        Returns a list of GradeResult where result[i] is the grade
        after seeing turns 0..i.

        Useful for live evaluation as a conversation unfolds.
        """
        temp    = temperature if temperature is not None else self.temperature
        results = []

        for end in range(min_turns, len(features) + 1):
            prefix = features[:end]
            feat   = torch.tensor(
                prefix, dtype=self.dtype, device=self.device
            ).unsqueeze(0)
            probs     = self._forward(feat, temp).squeeze(0)
            class_idx = int(probs.argmax().item())
            results.append(GradeResult(
                grade=GRADE_MAP.get(class_idx, "Basic"),
                class_idx=class_idx,
                confidence=float(probs[class_idx].item()),
                probabilities={GRADE_MAP[i]: float(probs[i].item())
                               for i in range(len(probs))},
                temperature=temp,
            ))
        return results

    # ── Calibration ──────────────────────────────────────────────────────────
    def set_temperature(self, temperature: float) -> None:
        """Set the default calibration temperature for all predictions."""
        if temperature <= 0:
            raise ValueError("Temperature must be > 0")
        self.temperature = temperature
        print(f"[xorm] Temperature set to {temperature}")

    def auto_calibrate(
        self,
        features_list: List[List[List[float]]],
        true_grades: List[str],
        candidates: Optional[List[float]] = None,
    ) -> float:
        """
        Search for the best calibration temperature using accuracy on a
        held-out validation set.

        Args:
            features_list: Per-conversation feature sequences.
            true_grades:   Ground truth grade strings.
            candidates:    Temperatures to search over.

        Returns:
            Best temperature found (also sets self.temperature).
        """
        if candidates is None:
            candidates = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]

        best_temp = 1.0
        best_acc  = -1.0

        for temp in candidates:
            preds = self.predict_batch(features_list, temperature=temp)
            correct = sum(
                1 for p, g in zip(preds, true_grades) if p.grade == g
            )
            acc = correct / len(true_grades)
            if acc > best_acc:
                best_acc  = acc
                best_temp = temp

        self.temperature = best_temp
        print(f"[xorm] Auto-calibrated: temperature={best_temp}  accuracy={best_acc:.1%}")
        return best_temp

    # ── Utilities ────────────────────────────────────────────────────────────
    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def info(self) -> str:
        m = self.manifest
        lines = [
            f"Model      : {m.get('model_name', '?')}",
            f"Family     : {m.get('family', '?')}",
            f"Role       : {m.get('role', '?')}",
            f"Params     : {m.get('total_params', self.param_count()):,}",
            f"Dtype      : {m.get('dtype', str(self.dtype))}",
            f"Device     : {self.device}",
            f"Temperature: {self.temperature}",
            f"Created    : {m.get('created_at', '?')}",
        ]
        meta = self.meta
        if meta.get("accuracy"):
            lines.append(f"Train Acc  : {meta['accuracy']:.1%}")
        return "\n".join(lines)

    def close(self) -> None:
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return (
            f"XormSession({self.manifest.get('model_name', '?')}  "
            f"device={self.device}  temperature={self.temperature})"
        )


# ──────────────────────────────────────────────
# Runtime
# ──────────────────────────────────────────────
class XormRuntime:
    """
    Global inference engine for .xorm models.

    Maintains a registry of loaded sessions so the same model is
    never loaded from disk twice.
    """

    @staticmethod
    def load(
        path: str,
        device: str = "cpu",
        dtype: Optional[str] = None,
        temperature: float = 1.0,
        compile: bool = False,
        use_registry: bool = True,
    ) -> XormSession:
        """
        Load a .xorm model and return an active XormSession.

        Args:
            path:         Path to the .xorm file.
            device:       "cpu" | "cuda" | "cuda:0" etc.
            dtype:        "fp32" | "fp16" | "bf16" (default fp32).
            temperature:  Default calibration temperature.
            compile:      If True, run torch.compile() on the model.
            use_registry: If True, return cached session for same path.

        Returns:
            XormSession
        """
        abs_path = str(Path(path).resolve())

        if use_registry:
            with _REGISTRY_LOCK:
                if abs_path in _REGISTRY:
                    return _REGISTRY[abs_path]

        # Map dtype string to torch.dtype
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            None:   torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.float32)

        reader   = XormReader(abs_path)
        manifest = reader.manifest
        family   = manifest.get("family")

        if family not in XORM_FAMILY_REGISTRY:
            reader.close()
            raise ValueError(
                f"Unsupported model family '{family}'. "
                f"Supported: {list(XORM_FAMILY_REGISTRY.keys())}"
            )

        model_cls   = XORM_FAMILY_REGISTRY[family]
        arch_config = reader.arch_config

        if family == "greed":
            base_overrides = {
                k: v for k, v in arch_config.items()
                if k not in ("input_dim", "num_classes")
            }
            config = ConfigFactory.get_config(ModelSize.TINY_23K, **base_overrides)
            config.vocab_size = arch_config.get("num_classes", 5)
            model = model_cls(
                config,
                input_dim=arch_config.get("input_dim", 24),
                num_classes=arch_config.get("num_classes", 5),
                test_mode=True,
            )
        else:
            raise ValueError(f"No loader registered for family '{family}'")

        state_dict = reader.load_state_dict(map_location=device)
        model.load_state_dict(state_dict)

        session = XormSession(
            model=model,
            reader=reader,
            device=device,
            dtype=torch_dtype,
            temperature=temperature,
            compiled=compile,
        )

        if use_registry:
            with _REGISTRY_LOCK:
                _REGISTRY[abs_path] = session

        print(f"[xorm] Loaded {manifest.get('model_name', path)}")
        return session

    @staticmethod
    def load_from_dir(
        directory: str,
        family: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, XormSession]:
        """
        Load all .xorm files from a directory.

        Args:
            directory: Directory to scan.
            family:    If set, only load models of this family.
            **kwargs:  Passed to XormRuntime.load().

        Returns:
            Dict mapping filename → XormSession.
        """
        sessions = {}
        for p in Path(directory).glob("*.xorm"):
            try:
                session = XormRuntime.load(str(p), **kwargs)
                if family is None or session.manifest.get("family") == family:
                    sessions[p.name] = session
            except Exception as e:
                print(f"[xorm] Skipping {p.name}: {e}")
        return sessions

    @staticmethod
    def clear_registry() -> None:
        """Close and remove all cached sessions."""
        with _REGISTRY_LOCK:
            for s in _REGISTRY.values():
                try:
                    s.close()
                except Exception:
                    pass
            _REGISTRY.clear()
        print("[xorm] Registry cleared")

    @staticmethod
    def registry_info() -> str:
        """List all models currently loaded in the registry."""
        with _REGISTRY_LOCK:
            if not _REGISTRY:
                return "[xorm] Registry is empty"
            lines = [f"[xorm] Registry ({len(_REGISTRY)} model(s)):"]
            for path, s in _REGISTRY.items():
                name = s.manifest.get("model_name", Path(path).name)
                lines.append(f"  {name}  ({s.device})")
            return "\n".join(lines)
