"""graph_stats.py

Graph statistics utilities.

This module is viewpoint-agnostic:
- It does not parse or construct node labels itself.
- Instead, it relies on a TokenCodec to map a destination node label into a Symbol
  (what the model predicts next).

Faithfulness notes
- The original code computed distributions from outgoing edges by summing edge weights
  and normalizing.
- Some destinations may not map to a symbol (codec returns None); those edges are skipped.

All returned distributions are plain dicts: {symbol: prob}.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Set, Tuple

import networkx as nx

from graph_types import Counts, Dist, NodeLabel, Order, Symbol, TokenCodec


def _is_no_event(sym: Symbol) -> bool:
    # Sentinel used in interval-only mode; never treat as a predictive symbol.
    return sym == "NO_EVENT"


def counts_from_out_edges(
    g: nx.DiGraph,
    context_node: NodeLabel,
    *,
    order_k: Order,
    codec: TokenCodec,
    use_update_exclusion: bool = False,
) -> Counts:
    """Aggregate outgoing edge weights into symbol counts.

    Args:
        g: directed graph for a particular order.
        context_node: the node label representing the context.
        order_k: the order of the graph (needed to decode destination labels).
        codec: used to map destination node labels -> predicted symbol.

    Returns:
        counts: dict mapping symbol -> summed edge weight.
        Empty dict {} if context_node not in graph (unseen context).
    """

    if not g.has_node(context_node):
        # Unseen context: return empty (matches IDyOM's None return for unseen states)
        return {}

    counts: Dict[Symbol, float] = {}

    weight_key = "weight_ux" if bool(use_update_exclusion) else "weight"
    for _, dst, data in g.out_edges(context_node, data=True):
        w = float(data.get(weight_key, data.get("weight", 1.0)))
        sym = codec.symbol_from_dest_label(dst, order_k)
        if sym is None or _is_no_event(sym):
            continue
        counts[sym] = counts.get(sym, 0.0) + w

    return counts


def dist_from_out_edges(
    g: nx.DiGraph,
    context_node: NodeLabel,
    *,
    order_k: Order,
    codec: TokenCodec,
    min_total: float = 1e-12,
    use_update_exclusion: bool = False,
) -> Dist:
    """Convert outgoing edge weights to a probability distribution.
    
    Returns empty dict {} when context is unseen (no node or no counts).
    This matches IDyOM behavior: unseen contexts are excluded from merges,
    and if needed their relative entropy is computed as 1.0 (max uncertainty).
    """

    counts = counts_from_out_edges(
        g,
        context_node,
        order_k=order_k,
        codec=codec,
        use_update_exclusion=use_update_exclusion,
    )
    total = float(sum(counts.values()))
    if total <= min_total:
        # Unseen context or no data: return empty (will be excluded from merge)
        return {}
    return {s: float(c) / total for s, c in counts.items()}


def alphabet_from_graphs(
    graphs_by_order: Mapping[int, nx.DiGraph],
    *,
    orders: Sequence[int],
    codec: TokenCodec,
) -> Tuple[Symbol, ...]:
    """Compute the alphabet (set of symbols) represented in the graphs.

    We extract symbols from all destination node labels in each order graph.
    """

    alphabet: Set[Symbol] = set()

    for k in orders:
        g = graphs_by_order.get(int(k))
        if g is None:
            continue
        for _, dst in g.edges():
            sym = codec.symbol_from_dest_label(dst, int(k))
            if sym is not None and not _is_no_event(sym):
                alphabet.add(sym)

    return tuple(sorted(alphabet))


def order0_counts_from_graphs(
    graphs_by_order: Mapping[int, nx.DiGraph],
    *,
    orders: Sequence[int],
    codec: TokenCodec,
    prefer_order: int = 1,
    use_update_exclusion: bool = False,
) -> Counts:
    """Compute order-0 symbol counts from node weights.

    Why node weights (faithful)
    - In your original pipeline, nodes are added for every observed window and node
      attribute `weight` is incremented. This naturally counts occurrences even for
      terminal nodes (ending events) that may have no outgoing edges.

    Which graph is used
    - Prefer the order-1 graph when available because its nodes correspond most directly
      to individual events (or NO_EVENT in interval-only mode).
    - Otherwise fall back to the smallest available order from `orders`.

    Note
    - For order>1 graphs, node weights count windows (contexts) and therefore omit the
      first (order-1) events of each sequence. This is why we prefer order-1.
    """

    if not orders:
        return {}

    # Choose which order graph to use for order-0 estimation.
    #
    # Normal mode: prefer order-1 nodes (events).
    # Derived-only modes (interval-only, bioi-ratio-only, or both): low-order graphs
    # collapse to the NO_EVENT sentinel, so the closest analogue to an order-0 marginal
    # is the smallest graph whose node labels carry a meaningful predictive symbol.
    min_predictive_order_fn = getattr(codec, "min_predictive_order", None)
    min_predictive_order = int(min_predictive_order_fn()) if callable(min_predictive_order_fn) else 1
    if min_predictive_order > 1:
        if min_predictive_order in graphs_by_order:
            chosen_k = min_predictive_order
        else:
            eligible = [int(o) for o in orders if int(o) >= min_predictive_order and int(o) in graphs_by_order]
            if not eligible:
                return {}
            chosen_k = min(eligible)
    else:
        chosen_k = int(prefer_order) if int(prefer_order) in graphs_by_order else min(int(o) for o in orders)

    g = graphs_by_order.get(chosen_k)
    if g is None:
        return {}

    counts: Dict[Symbol, float] = {}

    # Node labels are encoded similarly to destination labels, so we reuse
    # `symbol_from_dest_label` to extract the event symbol represented by the node.
    weight_key = "weight_ux" if bool(use_update_exclusion) else "weight"
    cache_key = "_order0_counts_cache_ux" if bool(use_update_exclusion) else "_order0_counts_cache"
    cache_version_key = f"{cache_key}_version"
    graph_version = int(g.graph.get("_graph_cache_version", 0))
    cached = g.graph.get(cache_key)
    if isinstance(cached, dict) and int(g.graph.get(cache_version_key, -1)) == graph_version:
        return dict(cached)

    for node, attrs in g.nodes(data=True):
        sym = codec.symbol_from_dest_label(node, chosen_k)
        if sym is None or _is_no_event(sym):
            continue
        w = float(attrs.get(weight_key, attrs.get("weight", 1.0)))
        counts[sym] = counts.get(sym, 0.0) + w

    g.graph[cache_key] = dict(counts)
    g.graph[cache_version_key] = graph_version
    return counts


def normalize_counts(counts: Mapping[Symbol, float], *, min_total: float = 1e-12) -> Dist:
    """Utility to normalize counts into a distribution."""

    total = float(sum(float(v) for v in counts.values()))
    if total <= min_total:
        return {}
    return {s: float(c) / total for s, c in counts.items()}
