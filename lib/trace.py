"""trace.py

Diagnostics / tracing data structures for GraphIDyOM.

Goal
- Capture (optionally) everything you want to inspect at each time step t:
  - per-order entropies, relative entropies, and the weights used in the order merge
  - per-order probability assigned to the observed symbol and its surprisal
  - merged (per-model) entropy/prob/surprisal
  - LTM-vs-STM merge weights (and their entropies)

This module is intentionally *pure data* + serialization helpers.
It contains no graph/model logic.

You can:
- store full distributions (can be large) or omit them
- export traces as JSON-friendly dicts

PPM notes
- For PPM, "weights" are not entropy weights; they are the *mass allocated* at each
  stage k (escape_before * total/(total+unique)). We store escape_before/after and
  counts stats to make this transparent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from graph_types import Dist, Symbol


# -------------------------
# Configuration
# -------------------------


@dataclass(frozen=True)
class TraceConfig:
    """What to store in traces.

    Defaults are conservative to keep JSON size reasonable.
    """

    store_full_dists: bool = False
    store_per_order_dists: bool = False
    store_counts_stats: bool = True


# -------------------------
# Per-order tracing
# -------------------------


@dataclass
class OrderTrace:
    """Diagnostics for one order component at time t."""

    order: int

    # Uncertainty
    entropy: Optional[float] = None
    rel_entropy: Optional[float] = None

    # Weight used in the merge (entropy merge) OR allocated mass (PPM)
    weight: float = 0.0

    # Observed symbol stats under this component
    p_obs: Optional[float] = None
    surprisal_obs: Optional[float] = None

    # Optional: store the component distribution
    dist: Optional[Dict[Symbol, float]] = None

    # Optional: PPM/count stats
    total: Optional[float] = None
    unique: Optional[int] = None
    escape_before: Optional[float] = None
    escape_after: Optional[float] = None
    unseen_context: Optional[bool] = None

    # Optional: debug context node label used to compute counts (stringified)
    context_node: Optional[str] = None

    def to_dict(self, *, cfg: TraceConfig = TraceConfig()) -> Dict[str, Any]:
        d = {
            "order": int(self.order),
            "entropy": self.entropy,
            "rel_entropy": self.rel_entropy,
            "weight": float(self.weight),
            "p_obs": self.p_obs,
            "surprisal_obs": self.surprisal_obs,
        }

        if cfg.store_counts_stats:
            d.update(
                {
                    "total": self.total,
                    "unique": self.unique,
                    "escape_before": self.escape_before,
                    "escape_after": self.escape_after,
                    "unseen_context": self.unseen_context,
                }
            )
            # include debug label when present
            if self.context_node is not None:
                d["context_node"] = self.context_node

        if cfg.store_per_order_dists and self.dist is not None:
            d["dist"] = dict(self.dist)

        return d


# -------------------------
# Model-level trace (LTM or STM)
# -------------------------


@dataclass
class ModelTrace:
    """Diagnostics for a model (LTM or STM) at time t."""

    name: str  # "ltm" or "stm" or other

    # Merge strategy label: "entropy_geometric" / "entropy_arithmetic" / "ppm" ...
    merge_strategy: str

    # Per-order traces
    per_order: Dict[int, OrderTrace] = field(default_factory=dict)

    # Merged distribution stats
    merged_entropy: Optional[float] = None
    merged_rel_entropy: Optional[float] = None

    merged_p_obs: Optional[float] = None
    merged_surprisal_obs: Optional[float] = None

    # Optional: store merged dist
    merged_dist: Optional[Dict[Symbol, float]] = None

    # Extra strategy-specific info (e.g. alphabet_size for PPM)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, cfg: TraceConfig = TraceConfig()) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "merge_strategy": self.merge_strategy,
            "merged_entropy": self.merged_entropy,
            "merged_rel_entropy": self.merged_rel_entropy,
            "merged_p_obs": self.merged_p_obs,
            "merged_surprisal_obs": self.merged_surprisal_obs,
            "per_order": {str(k): v.to_dict(cfg=cfg) for k, v in sorted(self.per_order.items())},
            "extra": dict(self.extra) if self.extra else {},
        }

        if cfg.store_full_dists and self.merged_dist is not None:
            d["merged_dist"] = dict(self.merged_dist)

        return d


# -------------------------
# LTM-vs-STM merge tracing
# -------------------------


@dataclass
class MergeTrace:
    """Diagnostics for a merge between components (e.g. LTM vs STM)."""

    merge_strategy: str

    # weights per component label
    weights: Dict[str, float] = field(default_factory=dict)

    # entropies per component label (optional)
    entropies: Dict[str, float] = field(default_factory=dict)
    rel_entropies: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "merge_strategy": self.merge_strategy,
            "weights": dict(self.weights),
            "entropies": dict(self.entropies),
            "rel_entropies": dict(self.rel_entropies),
        }


# -------------------------
# Step trace
# -------------------------


@dataclass
class StepTrace:
    """All diagnostics for a single time step."""

    index: int
    observed: Optional[Symbol]

    # model traces
    ltm: Optional[ModelTrace] = None
    stm: Optional[ModelTrace] = None

    # how LTM/STM were combined
    ltm_stm_merge: Optional[MergeTrace] = None

    # final combined stats
    final_entropy: Optional[float] = None
    final_rel_entropy: Optional[float] = None
    final_p_obs: Optional[float] = None
    final_surprisal_obs: Optional[float] = None

    # optional final dist
    final_dist: Optional[Dict[Symbol, float]] = None

    def to_dict(self, *, cfg: TraceConfig = TraceConfig()) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "index": int(self.index),
            "observed": self.observed,
            "final_entropy": self.final_entropy,
            "final_rel_entropy": self.final_rel_entropy,
            "final_p_obs": self.final_p_obs,
            "final_surprisal_obs": self.final_surprisal_obs,
        }

        if self.ltm is not None:
            d["ltm"] = self.ltm.to_dict(cfg=cfg)
        if self.stm is not None:
            d["stm"] = self.stm.to_dict(cfg=cfg)
        if self.ltm_stm_merge is not None:
            d["ltm_stm_merge"] = self.ltm_stm_merge.to_dict()

        if cfg.store_full_dists and self.final_dist is not None:
            d["final_dist"] = dict(self.final_dist)

        return d


# -------------------------
# Full trace
# -------------------------


@dataclass
class Trace:
    """A trace over a whole sequence/file."""

    steps: Tuple[StepTrace, ...]
    cfg: TraceConfig = TraceConfig()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_config": {
                "store_full_dists": self.cfg.store_full_dists,
                "store_per_order_dists": self.cfg.store_per_order_dists,
                "store_counts_stats": self.cfg.store_counts_stats,
            },
            "steps": [s.to_dict(cfg=self.cfg) for s in self.steps],
        }


def dist_entropy_bits(dist: Mapping[Symbol, float], *, min_prob: float = 1e-15) -> float:
    """Shannon entropy in bits for a distribution."""

    import math

    h = 0.0
    for p in dist.values():
        p = float(p)
        if p <= 0.0:
            continue
        p = max(p, float(min_prob))
        h -= p * (math.log(p, 2))
    return float(h)


def dist_relative_entropy(dist: Mapping[Symbol, float], *, min_prob: float = 1e-15) -> float:
    """Relative entropy H/Hmax where Hmax=log2(|support|)."""

    import math

    support = sum(1 for p in dist.values() if float(p) > 0.0)
    if support <= 1:
        return 1.0
    h = dist_entropy_bits(dist, min_prob=min_prob)
    hmax = math.log(float(support), 2)
    if hmax <= 0:
        return 1.0
    r = h / hmax
    if r < 0.0:
        return 0.0
    if r > 1.0:
        return 1.0
    return float(r)


def dict_for_json(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable structures."""

    if isinstance(obj, Trace):
        return obj.to_dict()
    if isinstance(obj, StepTrace):
        return obj.to_dict(cfg=TraceConfig())
    if isinstance(obj, ModelTrace):
        return obj.to_dict(cfg=TraceConfig())
    if isinstance(obj, OrderTrace):
        return obj.to_dict(cfg=TraceConfig())
    if isinstance(obj, MergeTrace):
        return obj.to_dict()

    if isinstance(obj, dict):
        return {k: dict_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dict_for_json(v) for v in obj]

    return obj


def write_trace_json(path: str, trace: Trace, *, indent: int = 2) -> None:
    """Write a Trace to a JSON file."""

    import json
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace.to_dict(), f, ensure_ascii=False, indent=int(indent))
