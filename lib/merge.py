"""merge.py

Merge strategies for combining multiple probability distributions.

This module is model-agnostic: it only operates on distributions of the form
    {symbol: probability}

We start with entropy-weighted merges (faithful to your previous design):
- Entropy-weighted geometric mean
- Entropy-weighted arithmetic mean

Later we can add:
- PPM-style backoff distribution as a separate strategy/class.

Notes on faithfulness
- "Relative entropy" here means normalized Shannon entropy H(p)/log(|A|) in [0,1]
  when |A|>1. Low entropy => peaky => more confident.
- We convert confidence into a weight. A common faithful mapping is:
    w = 1 - rel_entropy
  (more confident => higher weight).
- If an alphabet is provided, it is used for the log(|A|) normalization; otherwise
  the distribution's support size is used.

All functions avoid returning NaNs and handle empty inputs gracefully.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from graph_types import Dist, MergeStrategy, Symbol


# -------------------------
# Entropy utilities
# -------------------------


def shannon_entropy(dist: Mapping[Symbol, float], *, min_prob: float = 1e-15) -> float:
    """Compute Shannon entropy (nats) for a distribution."""

    h = 0.0
    for p in dist.values():
        p = float(p)
        if p <= 0.0:
            continue
        p = max(p, min_prob)
        h -= p * math.log(p)
    return float(h)


def relative_entropy(
    dist: Mapping[Symbol, float],
    *,
    alphabet_size: Optional[int] = None,
    min_prob: float = 1e-15,
) -> float:
    """Normalized entropy in [0,1] when possible.
    
    IMPORTANT: To match IDyOM behavior, we normalize by the distribution's support size
    (number of non-zero probabilities) rather than a global alphabet size.
    IDyOM's getEntropyMax(state) uses len(getPrediction(state).keys()), which is the
    support of that specific context's distribution.
    
    The alphabet_size parameter is kept for backwards compatibility but is NOT used
    for Hmax computation to maintain IDyOM faithfulness.
    """

    if not dist:
        # Empty distribution = maximum uncertainty (IDyOM returns 1.0 for unseen states)
        return 1.0

    # CRITICAL: Use distribution support size (like IDyOM), not global alphabet
    a = len(dist)
    # Faithful to IDyOM: when Hmax == 0 (support <= 1), return 1.0
    if a <= 1:
        return 1.0

    h = shannon_entropy(dist, min_prob=min_prob)
    denom = math.log(float(a))
    if denom <= 0:
        return 0.0

    r = h / denom
    # clamp for numerical safety
    if r < 0.0:
        return 0.0
    if r > 1.0:
        return 1.0
    return float(r)



def confidence_weight_one_minus(
    dist: Mapping[Symbol, float],
    *,
    alphabet_size: Optional[int] = None,
) -> float:
    """Legacy/simple mapping: w = 1 - rel_entropy."""

    r = relative_entropy(dist, alphabet_size=alphabet_size)
    w = 1.0 - float(r)
    if w < 0.0:
        return 0.0
    if w > 1.0:
        return 1.0
    return float(w)


def entropy_weights(
    dists: Sequence[Mapping[Symbol, float]],
    *,
    alphabet_size: Optional[int] = None,
    mode: str = "inverse_power",
    b: float = 1.0,
    weight_offset: float = 0.01,
    eps: float = 1e-15,
) -> Tuple[float, ...]:
    """Compute per-distribution weights from normalized entropy.

    This is meant to be coherent with the (Open)IDyOM merging logic.

    Note: alphabet_size is passed to relative_entropy but is NOT used for Hmax
    computation (to match IDyOM's per-context support-based normalization).
    It's kept for API compatibility.

    Supported modes
    - "inverse_power" (IDyOM-style): w_i ∝ (rel_entropy_i + eps)^(-b)
      * Lower entropy => higher weight
      * `b` controls how strongly you prefer lower entropy
    - "one_minus": w_i = 1 - rel_entropy_i

    The returned weights are normalized to sum to 1 (unless dists is empty).
    """

    if not dists:
        return tuple()

    rels = [relative_entropy(d, alphabet_size=alphabet_size) for d in dists]

    if mode == "one_minus":
        ws = [max(0.0, 1.0 - float(r)) for r in rels]
    elif mode == "inverse_power":
        # Faithful to IDyOM/OpenIDyOM snippets:
        #   base = rel_entropy + 0.01
        #   w_i ∝ (base + eps)^(-b)
        b = float(b)
        off = float(weight_offset)
        ws = [float((float(r) + off + float(eps)) ** (-b)) for r in rels]
    else:
        raise ValueError(f"Unknown entropy weight mode: {mode}")

    # "Doomy normalization" as in your IDyOM snippet: ensure non-negative and non-zero.
    if any(w < 0.0 for w in ws):
        shift = abs(min(ws))
        ws = [w + shift for w in ws]

    s = float(sum(ws))
    if s <= 0.0:
        ws = [1.0 for _ in ws]
        s = float(len(ws))

    return tuple(float(w) / s for w in ws)


# -------------------------
# Merge primitives
# -------------------------


def _normalize(dist: Dict[Symbol, float], *, min_total: float = 1e-15) -> Dist:
    total = float(sum(dist.values()))
    if total <= min_total:
        return {}
    return {s: float(v) / total for s, v in dist.items()}


def weighted_arithmetic_mean(
    dists: Sequence[Mapping[Symbol, float]],
    *,
    weights: Sequence[float],
    min_total: float = 1e-15,
) -> Dist:
    """Weighted arithmetic mean of distributions."""

    out: Dict[Symbol, float] = {}
    for dist, w in zip(dists, weights):
        w = float(w)
        if w <= 0.0:
            continue
        for s, p in dist.items():
            out[s] = out.get(s, 0.0) + w * float(p)
    return _normalize(out, min_total=min_total)


def weighted_geometric_mean(
    dists: Sequence[Mapping[Symbol, float]],
    *,
    weights: Sequence[float],
    epsilon: float = 1e-15,
) -> Dist:
    """Weighted geometric mean of distributions.

    For each symbol s:
      log p(s) = sum_i w_i * log(max(p_i(s), epsilon))
    """

    symbols = set()
    for d in dists:
        symbols.update(d.keys())

    if not symbols:
        return {}

    out: Dict[Symbol, float] = {}
    for s in symbols:
        acc = 0.0
        wsum = 0.0
        for dist, w in zip(dists, weights):
            w = float(w)
            if w <= 0.0:
                continue
            p = float(dist.get(s, 0.0))
            p = max(p, float(epsilon))
            acc += w * math.log(p)
            wsum += w

        if wsum <= 0.0:
            continue

        out[s] = math.exp(acc / wsum)

    return _normalize(out, min_total=epsilon)


# -------------------------
# Strategy classes
# -------------------------


@dataclass(frozen=True)
class EntropyGeometricMerge(MergeStrategy):
    """Entropy-weighted geometric merge."""

    name: str = "entropy_geometric"
    alphabet_size: Optional[int] = None
    weight_mode: str = "inverse_power"
    b: float = 1.0

    def merge(self, dists: Sequence[Dist], *, weights: Optional[Sequence[float]] = None) -> Dist:
        dists = [d for d in dists if d]
        if not dists:
            return {}

        if weights is None:
            ws = list(
                entropy_weights(
                    dists,
                    alphabet_size=self.alphabet_size,
                    mode=self.weight_mode,
                    b=self.b,
                )
            )
        else:
            ws = [float(w) for w in weights]

        if sum(ws) <= 0.0:
            ws = [1.0 for _ in dists]

        return weighted_geometric_mean(dists, weights=ws)


@dataclass(frozen=True)
class EntropyArithmeticMerge(MergeStrategy):
    """Entropy-weighted arithmetic merge."""

    name: str = "entropy_arithmetic"
    alphabet_size: Optional[int] = None
    weight_mode: str = "inverse_power"
    b: float = 1.0

    def merge(self, dists: Sequence[Dist], *, weights: Optional[Sequence[float]] = None) -> Dist:
        dists = [d for d in dists if d]
        if not dists:
            return {}

        if weights is None:
            ws = list(
                entropy_weights(
                    dists,
                    alphabet_size=self.alphabet_size,
                    mode=self.weight_mode,
                    b=self.b,
                )
            )
        else:
            ws = [float(w) for w in weights]

        if sum(ws) <= 0.0:
            ws = [1.0 for _ in dists]

        return weighted_arithmetic_mean(dists, weights=ws)


# -------------------------
# PPM-style backoff merge
# -------------------------


@dataclass(frozen=True)
class PPMMerge:
    """PPM-style backoff merge.

    This is a *count-based* backoff, matching the logic in your provided IDyOM snippet.

    IMPORTANT: PPM is not well-defined from probability dists alone; it needs counts.
    Therefore this class provides `dist_from_counts(...)` / `prob_from_counts(...)`.

    How to use
    - For a given context, compute `counts_k` for each order k (including order 0),
      where counts_k maps symbol -> count.
    - Pass them from highest order to lowest order.

    Order -1 fallback
    - If `alphabet` is provided, we use a uniform distribution over the alphabet.
    - Otherwise we use a uniform distribution over the union of observed symbols.

    Escape mechanism (IDyOM-style)
    - At each order, let:
        total = sum(counts)
        unique = number of symbols with nonzero count
        weight = total / (total + unique)
      Then:
        P += escape * (count(symbol)/total)
        escape *= (1 - weight)
    """

    name: str = "ppm"
    stop_escape_threshold: float = 1e-10
    # IDyOM/OpenIDyOM escape methods: a, b, c, d, x.
    # Default "c" matches the existing GraphIDYOM behavior.
    escape_method: str = "c"

    def _method_params(self) -> tuple[float, float, bool]:
        m = str(self.escape_method).strip().lower()
        if m == "a":
            return 0.0, 1.0, True
        if m == "b":
            return -1.0, 1.0, False
        if m == "c":
            return 0.0, 1.0, False
        if m == "d":
            return -0.5, 2.0, False
        if m == "x":
            return 0.0, 1.0, False
        raise ValueError(f"Unknown PPM escape method: {self.escape_method!r}. Expected one of: a,b,c,d,x")

    def _type_count(self, counts: Mapping[Symbol, float]) -> float:
        m = str(self.escape_method).strip().lower()
        if m == "x":
            # Method X: number of singleton symbols + 1.
            singletons = 0
            for v in counts.values():
                if float(v) == 1.0:
                    singletons += 1
            return float(singletons + 1)
        return float(sum(1 for v in counts.values() if float(v) > 0.0))

    def prob_from_counts(
        self,
        counts_by_order_high_to_low: Sequence[Mapping[Symbol, float]],
        symbol: Symbol,
        *,
        alphabet: Optional[Sequence[Symbol]] = None,
        excluded_count: int = 1,
        exclusion: bool = False,
        update_exclusion: bool = True,
    ) -> float:
        dist = self.dist_from_counts(
            counts_by_order_high_to_low,
            alphabet=alphabet,
            excluded_count=excluded_count,
            exclusion=exclusion,
            update_exclusion=update_exclusion,
        )
        return float(dist.get(symbol, 0.0))

    def dist_from_counts(
        self,
        counts_by_order_high_to_low: Sequence[Mapping[Symbol, float]],
        *,
        alphabet: Optional[Sequence[Symbol]] = None,
        excluded_count: int = 1,
        exclusion: bool = False,
        update_exclusion: bool = True,
    ) -> Dist:
        # Determine alphabet for order -1 fallback
        if alphabet is None:
            symset = set()
            for c in counts_by_order_high_to_low:
                symset.update(c.keys())
            alphabet_list = sorted(symset)
        else:
            alphabet_list = list(alphabet)

        if not alphabet_list:
            return {}

        out: Dict[Symbol, float] = {s: 0.0 for s in alphabet_list}
        use_exclusion = bool(exclusion)
        root_counts: Mapping[Symbol, float] = (
            counts_by_order_high_to_low[-1] if counts_by_order_high_to_low else {}
        )

        escape = 1.0
        k, d, method_a = self._method_params()

        for counts in counts_by_order_high_to_low:
            raw_counts = {s: float(v) for s, v in counts.items() if float(v) > 0.0}
            if not raw_counts:
                # In your IDyOM snippet: reset escape to 1.0 when no observations
                escape = 1.0
                continue

            # IDyOM-style adjusted transition counts: c + k (for seen symbols).
            adj_counts: Dict[Symbol, float] = {}
            for s, c in raw_counts.items():
                adj = float(c + k)
                if adj > 0.0:
                    adj_counts[s] = adj

            state_count = float(sum(adj_counts.values()))
            if state_count <= 0.0:
                escape = 1.0
                continue

            type_count = float(self._type_count(raw_counts))
            if method_a:
                denom = state_count + 1.0
            else:
                denom = state_count + (type_count / float(d))
            weight = float(state_count / denom) if denom > 0.0 else 0.0

            # Add contribution from this order
            for s in alphabet_list:
                c = float(adj_counts.get(s, 0.0))
                if c <= 0.0:
                    continue
                out[s] += escape * (weight * (c / state_count))

            # Update escape for next (lower) order
            escape *= (1.0 - weight)
            if escape < float(self.stop_escape_threshold):
                break

        # Order -1 uniform fallback (matches your snippet)
        denom = float(len(alphabet_list) + 1 - int(excluded_count))
        if denom > 0.0 and escape > 0.0:
            uni = escape * (1.0 / denom)
            if use_exclusion:
                root_symbols = {s for s, v in root_counts.items() if float(v) > 0.0}
                for s in alphabet_list:
                    if s in root_symbols:
                        continue
                    out[s] += uni
            else:
                for s in alphabet_list:
                    out[s] += uni

        # Final normalization guard
        total_p = float(sum(out.values()))
        if total_p <= 0.0:
            # fallback: uniform
            u = 1.0 / float(len(alphabet_list))
            return {s: u for s in alphabet_list}

        return {s: float(p) / total_p for s, p in out.items()}
