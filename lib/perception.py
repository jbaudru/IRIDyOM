"""perception.py

Perception expansion utilities for smoothed transition distributions.

Key formula:
    P_hat = (1 - η) Σ_{n=1}^{D} η^{n-1} P^n

Where:
    - P is the transition matrix at a given order
    - η is the geometric decay parameter
    - D is the max depth of the power-series expansion
    - P_hat is the smoothed/perceived transition distribution

Dead-end states (no successors) act as sinks and absorb mass.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import networkx as nx

from graph_types import Dist, Symbol, TokenCodec
from label_utils import TERMINAL_SYMBOLS, fallback_symbol_from_label

# Default parameters
DEFAULT_ETA = 0.8
DEFAULT_MAX_DEPTH = 15


def _clamp_eta(eta: float) -> float:
    e = float(eta)
    if e < 0.0:
        return 0.0
    if e >= 1.0:
        # Keep geometric weights valid and non-degenerate.
        return 0.999999
    return e
def _symbol_from_state(
    state: str,
    *,
    order_k: Optional[int] = None,
    codec: Optional[TokenCodec] = None,
) -> Optional[Symbol]:
    """Extract the model symbol represented by a destination state label.

    If a codec is provided, this uses `codec.symbol_from_dest_label(...)` so
    extraction is exactly aligned with model semantics. Otherwise it falls back
    to robust JSON-aware parsing.
    """
    if codec is not None and order_k is not None:
        try:
            sym = codec.symbol_from_dest_label(state, int(order_k))
            if sym is not None:
                return sym
        except Exception:
            # Fall back to robust generic extraction.
            pass
    return fallback_symbol_from_label(state)


def _get_transition_dist(g: nx.DiGraph, state: str) -> Dict[str, float]:
    """Get normalized transition distribution over successor states."""
    if not g.has_node(state):
        return {}
    
    dist: Dict[str, float] = {}
    total = 0.0
    for _, dst, data in g.out_edges(state, data=True):
        w = float(data.get('weight', 1.0))
        dist[dst] = w
        total += w
    
    if total <= 0:
        return {}
    
    return {k: v / total for k, v in dist.items()}


def _get_symbol_dist(
    g: nx.DiGraph,
    state: str,
    *,
    order_k: Optional[int] = None,
    codec: Optional[TokenCodec] = None,
) -> Dist:
    """Get transition distribution over output symbols (not states)."""
    state_dist = _get_transition_dist(g, state)
    
    symbol_dist: Dict[Symbol, float] = {}
    for succ_state, prob in state_dist.items():
        sym = _symbol_from_state(succ_state, order_k=order_k, codec=codec)
        if sym is None:
            continue
        symbol_dist[sym] = symbol_dist.get(sym, 0.0) + prob
    
    return symbol_dist


def _is_terminal(state: str) -> bool:
    """Check if state ends with a terminal symbol."""
    sym = fallback_symbol_from_label(state)
    return bool(sym) and sym in TERMINAL_SYMBOLS


def compute_phat(
    g: nx.DiGraph,
    state: str,
    *,
    eta: float = DEFAULT_ETA,
    max_depth: int = DEFAULT_MAX_DEPTH,
    order_k: Optional[int] = None,
    codec: Optional[TokenCodec] = None,
) -> Dist:
    """Compute perceived distribution P_hat from a state.
    
    Uses sink behavior: dead-end states absorb mass (no backoff).
    
    Args:
        g: The graph for this order
        state: Current context state label
        eta: Geometric decay parameter (default 0.8)
        max_depth: Maximum power series depth (default 15)
    
    Returns:
        Perceived distribution over symbols
    """
    if not g.has_node(state) or g.out_degree(state) == 0:
        return {}
    eta_eff = _clamp_eta(float(eta))
    depth = max(1, int(max_depth))
    
    p_hat: Dict[Symbol, float] = {}
    frontier: Dict[str, float] = {state: 1.0}
    
    for n in range(1, depth + 1):
        next_frontier: Dict[str, float] = {}
        step_symbols: Dict[Symbol, float] = {}
        
        for src_state, mass in frontier.items():
            dist = _get_transition_dist(g, src_state)
            
            if not dist:
                # Sink: absorb mass, no propagation
                continue
            
            for succ_state, prob in dist.items():
                sym = _symbol_from_state(succ_state, order_k=order_k, codec=codec)
                if sym is None:
                    continue
                step_symbols[sym] = step_symbols.get(sym, 0.0) + mass * prob
                
                # Only propagate non-terminal states
                if not _is_terminal(succ_state):
                    next_frontier[succ_state] = next_frontier.get(succ_state, 0.0) + mass * prob
        
        # Accumulate with geometric weighting
        coeff = (1.0 - eta_eff) * (eta_eff ** (n - 1))
        for sym, prob in step_symbols.items():
            p_hat[sym] = p_hat.get(sym, 0.0) + coeff * prob
        
        frontier = next_frontier
        if not frontier:
            break
    
    # Normalize
    total = sum(p_hat.values())
    if total > 0:
        return {k: v / total for k, v in p_hat.items()}
    return _get_symbol_dist(g, state, order_k=order_k, codec=codec)
