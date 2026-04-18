"""model.py

High-level GraphIDyOM model orchestrator.

This module provides a modular, extensible interface:
- Train a Long-Term Model (LTM) from a folder of files (graphs per order)
- Maintain a Short-Term Model (STM) online while processing a file
- Predict next-symbol distributions from a context
- Compute per-event likelihoods / surprisals over a file

Design principles
- MIDI parsing is delegated to a MidiParser.
- Viewpoint encoding/decoding is delegated to a TokenCodec.
- Order merging is delegated to merge strategies in `merge.py`.
- Graph stats are delegated to `graph_stats.py`.

Faithfulness targets (relative to the original IDyOM-like implementation)
- Orders are weighted by (normalized) entropy using the inverse-power rule:
    w_i ∝ (rel_entropy_i + eps)^(-b)
  implemented via `merge.entropy_weights(...)`.
- LTM and STM are combined using the same entropy-weighted merge (geometric by default).
- PPM is supported via a count-based backoff distribution (needs counts, not only dists).

Note
- This implementation currently supports a *single* viewpoint codec instance.
  Extending to multiple viewpoints is straightforward by combining multiple model
  instances or creating a MultiViewpointCodec and a product-of-experts combiner.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx

from graph_build import GraphBuildConfig, GraphBuilder
from graph_stats import (
    alphabet_from_graphs,
    counts_from_out_edges,
    dist_from_out_edges,
    normalize_counts,
    order0_counts_from_graphs,
)
from interaction_history import midi_history_to_noteinfos
from merge import (
    EntropyGeometricMerge,
    EntropyArithmeticMerge,
    PPMMerge,
    entropy_weights,
)
from graph_types import Dist, MidiParser, NoteInfo, ProcessedSequence, StepPrediction, Symbol, TokenCodec
from trace import (
    MergeTrace,
    ModelTrace,
    OrderTrace,
    StepTrace,
    Trace,
    TraceConfig,
    dist_entropy_bits,
    dist_relative_entropy,
)
from pretrained_models_manager import PretrainedModelsManager, ModelConfig
from target_projection import TargetProjectionMixin
from token_codec import bioi_ratio_symbol_at
from viewpoint_system import ViewpointSpec, classify_viewpoint_config


# -------------------------
# STM internal state
# -------------------------


@dataclass
class _STMOrderState:
    """Per-order incremental graph-building state."""

    order: int
    prev_node: Optional[str] = None
    end_time: Optional[float] = None
    # store the last `order` NoteInfo for window construction
    window: List[NoteInfo] = None  # will be treated as a ring buffer manually


class _STMState:
    def __init__(self, *, orders: Sequence[int]):
        self.graphs: Dict[int, nx.DiGraph] = {int(o): nx.DiGraph() for o in orders}
        for g in self.graphs.values():
            g.graph["_graph_cache_version"] = 0
        self.orders: Tuple[int, ...] = tuple(int(o) for o in orders)
        self.per_order: Dict[int, _STMOrderState] = {
            int(o): _STMOrderState(order=int(o), window=[]) for o in orders
        }


# -------------------------
# Main model
# -------------------------


class GraphIDYOMModel(TargetProjectionMixin):
    """A single-viewpoint GraphIDyOM model (LTM + optional STM).

    Public usage
    - fit_folder(...)
    - reset_stm()
    - process_file(path)
    - predict_next_dist(history)

    The model predicts *symbols* defined by the TokenCodec:
    - normal mode: note token JSON string
    - interval-only mode: interval token JSON string (and NO_EVENT for the first event)
    """

    def __init__(
        self,
        *,
        parser: MidiParser,
        codec: TokenCodec,
        orders: Sequence[int] = (1, 2, 3, 4),
        graph_build_config: GraphBuildConfig = GraphBuildConfig(),
        use_stm: bool = True,
        # Order-merging strategy within a model (across orders)
        order_merge=None,
        # How to combine LTM and STM distributions
        ltm_stm_merge=None,
        # PPM settings
        use_ppm: bool = False,
        ppm_excluded_count: int = 1,
        ppm_escape_method: str = "c",
        ppm_reset_escape_on_unseen: bool = False,
        ppm_ltm_exclusion: bool = True,
        ppm_stm_exclusion: bool = True,
        ppm_ltm_update_exclusion: bool = False,
        ppm_stm_update_exclusion: bool = True,
        verbosity: int = 1,
        # Target viewpoint for projection
        target_viewpoint: Optional[str] = None,
    ):
        self.parser = parser
        self.codec = codec
        self.viewpoint_spec: Optional[ViewpointSpec] = None
        if getattr(self.codec, "cfg", None) is not None:
            self.viewpoint_spec = classify_viewpoint_config(self.codec.cfg)

        # If the midi viewpoint is requested by the codec, ensure the parser provides the required data
        if getattr(self.codec, "cfg", None) and getattr(self.codec.cfg, "midi", False):
            # No need for idyom_compatible flag; the parser now always provides the required viewpoints
            pass

        self.orders: Tuple[int, ...] = tuple(sorted({int(o) for o in orders if int(o) >= 1}))
        if not self.orders:
            raise ValueError("orders must contain at least one integer >= 1")
        
        # Target viewpoint configuration
        self.target_viewpoint = target_viewpoint
        if self.target_viewpoint is not None:
            self._validate_target_viewpoint()

        self.graph_build_config = graph_build_config
        self.use_stm = bool(use_stm)
        self.use_ppm = bool(use_ppm)
        self.ppm_excluded_count = int(ppm_excluded_count)
        self.ppm_escape_method = str(ppm_escape_method).strip().lower()
        self.ppm_reset_escape_on_unseen = bool(ppm_reset_escape_on_unseen)
        self.ppm_ltm_exclusion = bool(ppm_ltm_exclusion)
        self.ppm_stm_exclusion = bool(ppm_stm_exclusion)
        self.ppm_ltm_update_exclusion = bool(ppm_ltm_update_exclusion)
        self.ppm_stm_update_exclusion = bool(ppm_stm_update_exclusion)
        self.verbosity = int(verbosity)

        # Merging strategies
        self.order_merge = order_merge or EntropyArithmeticMerge()
        self.ltm_stm_merge = ltm_stm_merge or EntropyArithmeticMerge()
        
        # PPMMerge initialization - handle different API versions
        try:
            # Try with both parameters (newer version)
            self.ppm = PPMMerge(
                escape_method=self.ppm_escape_method,
                exclusion=self.ppm_ltm_exclusion,
            )
        except TypeError:
            try:
                # Try with only escape_method (middle version)
                self.ppm = PPMMerge(
                    escape_method=self.ppm_escape_method,
                )
            except TypeError:
                # Fallback: try with no parameters
                self.ppm = PPMMerge()
        
        # Set additional attributes if they exist
        if hasattr(self.ppm, 'reset_escape_on_unseen'):
            self.ppm.reset_escape_on_unseen = self.ppm_reset_escape_on_unseen
        if hasattr(self.ppm, 'exclusion'):
            self.ppm.exclusion = self.ppm_ltm_exclusion
        if isinstance(self.order_merge, PPMMerge):
            self.ppm = self.order_merge
        self.ppm_reset_escape_on_unseen = bool(
            getattr(self.ppm, "reset_escape_on_unseen", self.ppm_reset_escape_on_unseen)
        )

        # Graph builder used only for LTM and optional offline STM construction
        self.graph_builder = GraphBuilder(
            parser=self.parser,
            codec=self.codec,
            config=self.graph_build_config,
            verbosity=self.verbosity,
        )

        # Trained LTM graphs
        self.ltm_graphs: Dict[int, nx.DiGraph] = {}

        # Alphabet (symbols) inferred from LTM graphs; used for normalization and fallbacks
        self.alphabet: Tuple[Symbol, ...] = ()
        
        # Target viewpoint alphabet (if target_viewpoint is specified)
        self.target_alphabet: Tuple[Symbol, ...] = ()
        
        # Projection cache for performance (maps source symbol -> target symbols with probs)
        self._projection_cache: Dict[Symbol, Dict[Symbol, float]] = {}

        # Online STM state
        self._stm_state: Optional[_STMState] = None
        if self.use_stm:
            self.reset_stm()

    def _ppm_exclusion_for_graphs(self, graphs: Mapping[int, nx.DiGraph]) -> bool:
        if graphs is self.ltm_graphs:
            return bool(self.ppm_ltm_exclusion)
        if self._stm_state is not None and graphs is self._stm_state.graphs:
            return bool(self.ppm_stm_exclusion)
        return bool(self.ppm_ltm_exclusion)

    def _ppm_update_exclusion_for_graphs(self, graphs: Mapping[int, nx.DiGraph]) -> bool:
        if graphs is self.ltm_graphs:
            return bool(self.ppm_ltm_update_exclusion)
        if self._stm_state is not None and graphs is self._stm_state.graphs:
            return bool(self.ppm_stm_update_exclusion)
        return bool(self.ppm_ltm_update_exclusion)

    # -------------------------
    # Training / setup
    # -------------------------

    def _refresh_merge_alphabet_size(self) -> None:
        """Sync merge strategy alphabet size after alphabet updates."""
        if self.target_viewpoint is not None:
            a_size = len(self.target_alphabet) if self.target_alphabet else None
        else:
            a_size = len(self.alphabet) if self.alphabet else None
        if hasattr(self.order_merge, "alphabet_size"):
            object.__setattr__(self.order_merge, "alphabet_size", a_size)  # type: ignore
        if hasattr(self.ltm_stm_merge, "alphabet_size"):
            object.__setattr__(self.ltm_stm_merge, "alphabet_size", a_size)  # type: ignore

    def _prime_ltm_order0_cache(self) -> None:
        """Precompute order-0 marginals for static LTM graphs."""
        if not self.ltm_graphs:
            return
        order0_counts_from_graphs(self.ltm_graphs, orders=self.orders, codec=self.codec)
        order0_counts_from_graphs(
            self.ltm_graphs,
            orders=self.orders,
            codec=self.codec,
            use_update_exclusion=True,
        )

    def _observed_symbol_for_lookup(
        self,
        *,
        history: Sequence[NoteInfo],
        current_note: Optional[NoteInfo],
        observed: Optional[Symbol],
    ) -> Optional[Symbol]:
        """Return the symbol that should be scored against the final output dist."""
        if self.target_viewpoint is None or current_note is None:
            return observed
        seq = list(history) + [current_note]
        return self._target_symbol_from_note(seq=seq, index=len(history), note=current_note)

    def _merge_trace_for_components(
        self,
        *,
        labels: Sequence[str],
        dists: Sequence[Dist],
        strategy,
    ) -> Tuple[List[float], MergeTrace]:
        """Return entropy-based merge weights plus a serializable trace."""
        labels_list = [str(label) for label in labels]
        dists_list = [dict(dist) for dist in dists]
        if not dists_list:
            return [], MergeTrace(merge_strategy=getattr(strategy, "name", "merge"))

        if len(dists_list) == 1:
            ws = [1.0]
        else:
            ws = list(
                entropy_weights(
                    dists_list,
                    alphabet_size=getattr(strategy, "alphabet_size", None),
                    mode=str(getattr(strategy, "weight_mode", "inverse_power")),
                    b=float(getattr(strategy, "b", 1.0)),
                    eps=1e-15,
                )
            )
            if sum(ws) <= 0.0:
                ws = [1.0 / float(len(dists_list))] * len(dists_list)

        trace = MergeTrace(
            merge_strategy=getattr(strategy, "name", "merge"),
            weights={label: float(w) for label, w in zip(labels_list, ws)},
            entropies={
                label: float(dist_entropy_bits(dist)) for label, dist in zip(labels_list, dists_list)
            },
            rel_entropies={
                label: float(dist_relative_entropy(dist)) for label, dist in zip(labels_list, dists_list)
            },
        )
        return ws, trace

    def _target_symbol_from_note(
        self,
        *,
        seq: Sequence[NoteInfo],
        index: int,
        note: NoteInfo,
    ) -> Optional[str]:
        """Map an observed note to the corresponding target-viewpoint symbol."""
        tv = self.target_viewpoint
        if tv is None:
            return None
        if tv == "pitchOctave":
            return str(note.pitch)
        if tv == "pitchClass":
            return str(note.pitch_class)
        if tv == "midi_number":
            return str(int(note.midi))
        if tv == "interval":
            if int(index) <= 0:
                return None
            prev = seq[int(index) - 1]
            return str(int(note.midi) - int(prev.midi))
        if tv == "length":
            return str(int(note.length))
        if tv == "duration":
            ndigits = getattr(getattr(self.codec, "cfg", None), "token_round_ndigits", None)
            val = float(note.duration)
            if ndigits is not None:
                val = float(round(val, int(ndigits)))
            return str(val)
        if tv == "offset":
            ndigits = getattr(getattr(self.codec, "cfg", None), "token_round_ndigits", None)
            val = float(note.offset)
            if ndigits is not None:
                val = float(round(val, int(ndigits)))
            return str(val)
        if tv == "bioi_ratio":
            ratio_symbol = bioi_ratio_symbol_at(seq, int(index))
            if ratio_symbol is not None:
                return str(ratio_symbol)
            return None
        return None

    def extend_alphabet_from_sequences(self, sequences: Sequence[Sequence[NoteInfo]]) -> Dict[str, int]:
        """Extend source/target alphabets with symbols present in evaluation sequences.

        This aligns with IDyOM's behavior where basic viewpoint alphabets are
        initialized from dataset + pretraining data, not only model-training files.
        """
        src = set(self.alphabet)
        tgt = set(self.target_alphabet) if self.target_viewpoint is not None else set()
        src_before = len(src)
        tgt_before = len(tgt)

        for seq in sequences:
            if not seq:
                continue
            for i, note in enumerate(seq):
                sym = self.codec.symbol_at_index(seq, int(i))
                if sym is not None and sym != "NO_EVENT":
                    src.add(sym)
                t_sym = self._target_symbol_from_note(seq=seq, index=int(i), note=note)
                if t_sym is not None and self.target_viewpoint is not None:
                    tgt.add(t_sym)

        self.alphabet = tuple(sorted(src))
        if self.target_viewpoint is not None:
            self.target_alphabet = tuple(sorted(tgt))
            self._projection_cache.clear()
        self._refresh_merge_alphabet_size()
        return {
            "source_added": int(len(src) - src_before),
            "target_added": int(len(tgt) - tgt_before),
            "source_size": int(len(self.alphabet)),
            "target_size": int(len(self.target_alphabet)),
        }

    def extend_alphabet_from_files(self, files: Sequence[Union[str, os.PathLike]]) -> Dict[str, int]:
        """Parse files and extend alphabets using observed evaluation symbols."""
        seqs: List[List[NoteInfo]] = []
        for f in files:
            try:
                seq = self.parser.parse_file(str(f))
            except Exception:
                continue
            if seq:
                seqs.append(seq)
        return self.extend_alphabet_from_sequences(seqs)

    def fit_folder(self, folder: str, *, export_graphml: bool = True) -> Dict[int, nx.DiGraph]:
        """Train LTM graphs from all supported files in a folder."""

        self.ltm_graphs = self.graph_builder.build_folder_graphs(
            folder,
            orders=self.orders,
            export_graphml=export_graphml,
        )

        # Build alphabet from LTM; set alphabet_size on merges for stable relative entropy
        self.alphabet = alphabet_from_graphs(self.ltm_graphs, orders=self.orders, codec=self.codec)
        
        # Build target alphabet if target_viewpoint is specified
        if self.target_viewpoint is not None:
            self.target_alphabet = self._build_target_alphabet_from_folder(folder)
            # Clear projection cache when alphabet changes
            self._projection_cache.clear()
        self._refresh_merge_alphabet_size()
        self._prime_ltm_order0_cache()

        return self.ltm_graphs

    def fit_files(
        self,
        files: Sequence[Union[str, os.PathLike]],
        *,
        export_graphml: bool = False,
        export_base: str = "files",
    ) -> Dict[int, nx.DiGraph]:
        """Train LTM graphs from an explicit list of files."""

        file_paths = [str(Path(f)) for f in files]
        if not file_paths:
            raise ValueError("files must contain at least one path")

        self.ltm_graphs = self.graph_builder.build_file_graphs(
            file_paths,
            orders=self.orders,
            export_graphml=export_graphml,
            export_base=str(export_base),
        )

        self.alphabet = alphabet_from_graphs(self.ltm_graphs, orders=self.orders, codec=self.codec)

        if self.target_viewpoint is not None:
            self.target_alphabet = self._build_target_alphabet_from_files(file_paths)
            self._projection_cache.clear()
        self._refresh_merge_alphabet_size()
        self._prime_ltm_order0_cache()

        return self.ltm_graphs

    def export_graphs(
        self,
        graphs: Mapping[int, nx.DiGraph],
        *,
        base: str,
        export_dir: Union[str, os.PathLike],
    ) -> None:
        """Write GraphML files for the provided graphs."""
        export_dir_path = Path(export_dir)
        export_dir_path.mkdir(parents=True, exist_ok=True)
        self.graph_builder._export_graphs(  # type: ignore[attr-defined]
            graphs,
            base=str(base),
            export_dir=str(export_dir_path),
        )

    def filtered_ltm_subgraphs_for_nodes(
        self,
        visited_nodes_by_order: Mapping[int, Sequence[str]],
        *,
        include_missing_visited: bool = False,
    ) -> Dict[int, nx.DiGraph]:
        """Return LTM subgraphs containing visited nodes and their outgoing neighborhoods."""

        out: Dict[int, nx.DiGraph] = {}
        for order in self.orders:
            ltm_graph = self.ltm_graphs.get(int(order))
            if ltm_graph is None:
                continue

            visited_nodes = tuple(dict.fromkeys(str(node) for node in visited_nodes_by_order.get(int(order), ())))
            subgraph = nx.DiGraph()
            matched_count = 0

            for node in visited_nodes:
                if not ltm_graph.has_node(node):
                    if include_missing_visited:
                        subgraph.add_node(node, missing_from_ltm=True, visited_in_stm=True)
                    continue

                matched_count += 1
                subgraph.add_node(node, **dict(ltm_graph.nodes[node]))
                subgraph.nodes[node]["visited_in_stm"] = True

                for succ in ltm_graph.successors(node):
                    if not subgraph.has_node(succ):
                        subgraph.add_node(succ, **dict(ltm_graph.nodes[succ]))
                    subgraph.add_edge(node, succ, **dict(ltm_graph.edges[node, succ]))

            subgraph.graph["order"] = int(order)
            subgraph.graph["source"] = "ltm_filtered"
            subgraph.graph["visited_nodes"] = int(len(visited_nodes))
            subgraph.graph["matched_visited_nodes"] = int(matched_count)
            out[int(order)] = subgraph

        return out

    def filtered_ltm_subgraphs_for_stm(
        self,
        stm_graphs: Optional[Mapping[int, nx.DiGraph]] = None,
        *,
        include_missing_visited: bool = False,
    ) -> Dict[int, nx.DiGraph]:
        """Return filtered LTM subgraphs using STM node visits as the filter."""

        graphs = stm_graphs
        if graphs is None:
            if self._stm_state is None:
                raise RuntimeError("STM graphs are unavailable. Process a file first or provide stm_graphs.")
            graphs = self._stm_state.graphs

        visited_nodes_by_order = {
            int(order): tuple(str(node) for node in graph.nodes())
            for order, graph in graphs.items()
        }
        return self.filtered_ltm_subgraphs_for_nodes(
            visited_nodes_by_order,
            include_missing_visited=include_missing_visited,
        )

    def save_ltm(self, save_dir: str = None, *, dataset_name: str = None, 
                 source_viewpoint: str = None, augmented: bool = None,
                 manager: PretrainedModelsManager = None) -> None:
        """Save trained LTM graphs, alphabets, and configuration to disk.
        
        Two modes:
        1. Explicit path: save_ltm("/path/to/dir")
        2. Managed structure: save_ltm(dataset_name="elsass", source_viewpoint="pitch", augmented=True)
        
        Args:
            save_dir: Direct path. If provided, other args are ignored.
            dataset_name: Dataset name for managed structure (e.g., "largeWestern_elsass")
            source_viewpoint: Source viewpoint name (e.g., "pitch", "interval")
            augmented: Whether augmentation was used in training
            manager: PretrainedModelsManager instance (uses default if None)
        
        Saves:
            - LTM graphs (one pickle file per order)
            - Source alphabet
            - Target alphabet (if target_viewpoint is set)
            - Metadata (orders, target_viewpoint, etc.)
        """
        if not self.ltm_graphs:
            raise RuntimeError("No LTM graphs to save. Call fit_folder() first.")
        
        # Determine save directory
        if save_dir:
            # Explicit path mode
            save_path = Path(save_dir)
        elif dataset_name and source_viewpoint and augmented is not None:
            # Managed structure mode
            if manager is None:
                manager = PretrainedModelsManager()
            
            config = ModelConfig(
                dataset_name=dataset_name,
                source_viewpoint=source_viewpoint,
                augmented=augmented,
                target_viewpoint=self.target_viewpoint,
            )
            save_path = manager.get_model_dir(config)
            
            if self.verbosity >= 1:
                print(f"Using managed structure: {config}")
        else:
            raise ValueError(
                "Must provide either: (1) save_dir, or (2) dataset_name, "
                "source_viewpoint, and augmented parameters"
            )
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save each order's graph
        graphs_dir = save_path / "graphs"
        graphs_dir.mkdir(exist_ok=True)
        
        for order, graph in self.ltm_graphs.items():
            graph_file = graphs_dir / f"order_{order}.gpickle"
            with open(graph_file, "wb") as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save alphabets
        alphabet_file = save_path / "alphabet.json"
        with open(alphabet_file, "w", encoding="utf-8") as f:
            json.dump(list(self.alphabet), f, indent=2)
        
        if self.target_alphabet:
            target_alphabet_file = save_path / "target_alphabet.json"
            with open(target_alphabet_file, "w", encoding="utf-8") as f:
                json.dump(list(self.target_alphabet), f, indent=2)
        
        # Save metadata (including viewpoint configuration)
        from dataclasses import asdict
        viewpoint_config_dict = asdict(self.codec.cfg)
        
        metadata = {
            "orders": list(self.orders),
            "target_viewpoint": self.target_viewpoint,
            "use_ppm": self.use_ppm,
            "ppm_excluded_count": self.ppm_excluded_count,
            "ppm_escape_method": self.ppm_escape_method,
            "ppm_reset_escape_on_unseen": self.ppm_reset_escape_on_unseen,
            "ppm_ltm_exclusion": self.ppm_ltm_exclusion,
            "ppm_stm_exclusion": self.ppm_stm_exclusion,
            "ppm_ltm_update_exclusion": self.ppm_ltm_update_exclusion,
            "ppm_stm_update_exclusion": self.ppm_stm_update_exclusion,
            "viewpoint_config": viewpoint_config_dict,
        }
        if self.viewpoint_spec is not None:
            metadata["viewpoint_spec"] = {
                "name": self.viewpoint_spec.name,
                "kind": str(self.viewpoint_spec.kind.value),
                "components": list(self.viewpoint_spec.components),
                "typeset": list(self.viewpoint_spec.typeset),
                "links": list(self.viewpoint_spec.links),
            }
        metadata_file = save_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbosity >= 1:
            print(f"[+] Saved LTM to {save_path}")
            print(f"    - {len(self.ltm_graphs)} order graphs")
            print(f"    - Source alphabet: {len(self.alphabet)} symbols")
            if self.target_alphabet:
                print(f"    - Target alphabet: {len(self.target_alphabet)} symbols")

    def load_ltm(self, load_dir: str = None, *, dataset_name: str = None,
                 source_viewpoint: str = None, augmented: bool = None,
                 manager: PretrainedModelsManager = None) -> None:
        """Load pretrained LTM graphs, alphabets, and configuration from disk.
        
        Two modes:
        1. Explicit path: load_ltm("/path/to/dir")
        2. Managed structure: load_ltm(dataset_name="elsass", source_viewpoint="pitch", augmented=True)
        
        Args:
            load_dir: Direct path. If provided, other args are ignored.
            dataset_name: Dataset name for managed structure
            source_viewpoint: Source viewpoint name
            augmented: Whether augmentation was used in training
            manager: PretrainedModelsManager instance (uses default if None)
        
        Loads:
            - LTM graphs (one pickle file per order)
            - Source alphabet
            - Target alphabet (if present)
            - Metadata (validates consistency with current model config)
        """
        # Determine load directory
        if load_dir:
            # Explicit path mode
            load_path = Path(load_dir)
        elif dataset_name and source_viewpoint and augmented is not None:
            # Managed structure mode
            if manager is None:
                manager = PretrainedModelsManager()
            
            config = ModelConfig(
                dataset_name=dataset_name,
                source_viewpoint=source_viewpoint,
                augmented=augmented,
                target_viewpoint=self.target_viewpoint,
            )
            load_path = manager.get_model_dir(config)
            
            if self.verbosity >= 1:
                print(f"Using managed structure: {config}")
        else:
            raise ValueError(
                "Must provide either: (1) load_dir, or (2) dataset_name, "
                "source_viewpoint, and augmented parameters"
            )
        
        if not load_path.exists():
            raise FileNotFoundError(f"LTM directory not found: {load_path}")
        
        # Load metadata and validate
        metadata_file = load_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        saved_orders = tuple(sorted(metadata["orders"]))
        if saved_orders != self.orders:
            raise ValueError(
                f"Order mismatch: saved={saved_orders}, current={self.orders}. "
                "Create model with matching orders."
            )
        
        saved_target_vp = metadata.get("target_viewpoint")
        if saved_target_vp != self.target_viewpoint:
            raise ValueError(
                f"Target viewpoint mismatch: saved={saved_target_vp}, current={self.target_viewpoint}. "
                "Create model with matching target_viewpoint."
            )

        saved_ppm_escape = str(metadata.get("ppm_escape_method", self.ppm_escape_method)).strip().lower()
        if saved_ppm_escape != self.ppm_escape_method:
            raise ValueError(
                f"PPM escape method mismatch: saved={saved_ppm_escape}, current={self.ppm_escape_method}. "
                "Create model with matching ppm_escape_method."
            )
        if "ppm_reset_escape_on_unseen" in metadata:
            self.ppm_reset_escape_on_unseen = bool(metadata.get("ppm_reset_escape_on_unseen"))
            if hasattr(self.ppm, "reset_escape_on_unseen"):
                try:
                    object.__setattr__(self.ppm, "reset_escape_on_unseen", self.ppm_reset_escape_on_unseen)  # type: ignore[arg-type]
                except Exception:
                    pass
        if "ppm_ltm_exclusion" in metadata:
            self.ppm_ltm_exclusion = bool(metadata.get("ppm_ltm_exclusion"))
        if "ppm_stm_exclusion" in metadata:
            self.ppm_stm_exclusion = bool(metadata.get("ppm_stm_exclusion"))
        if "ppm_ltm_update_exclusion" in metadata:
            self.ppm_ltm_update_exclusion = bool(metadata.get("ppm_ltm_update_exclusion"))
        if "ppm_stm_update_exclusion" in metadata:
            self.ppm_stm_update_exclusion = bool(metadata.get("ppm_stm_update_exclusion"))
        if hasattr(self.ppm, "exclusion"):
            try:
                object.__setattr__(self.ppm, "exclusion", self.ppm_ltm_exclusion)  # type: ignore[arg-type]
            except Exception:
                pass
        
        # Validate viewpoint configuration
        saved_vp_config = metadata.get("viewpoint_config")
        if saved_vp_config:
            from dataclasses import asdict
            current_vp_config = asdict(self.codec.cfg)
            # Compare relevant fields (exclude token_round_ndigits as it's just numeric precision)
            important_fields = [
                'pitch',
                'octave',
                'midi_number',
                'duration',
                'length',
                'offset',
                'interval',
                'bioi_ratio',
            ]
            for field in important_fields:
                saved_val = saved_vp_config.get(field)
                current_val = current_vp_config.get(field)
                if saved_val != current_val:
                    raise ValueError(
                        f"Viewpoint config mismatch for field '{field}': "
                        f"saved={saved_val}, current={current_val}. "
                        "Create model with matching viewpoint configuration."
                    )
        
        # Load graphs
        graphs_dir = load_path / "graphs"
        self.ltm_graphs = {}
        
        for order in self.orders:
            graph_file = graphs_dir / f"order_{order}.gpickle"
            if not graph_file.exists():
                raise FileNotFoundError(f"Graph file not found: {graph_file}")
            
            with open(graph_file, "rb") as f:
                self.ltm_graphs[order] = pickle.load(f)
        
        # Load alphabets
        alphabet_file = load_path / "alphabet.json"
        if not alphabet_file.exists():
            raise FileNotFoundError(f"Alphabet file not found: {alphabet_file}")
        
        with open(alphabet_file, "r", encoding="utf-8") as f:
            self.alphabet = tuple(json.load(f))
        
        # Load target alphabet if present
        target_alphabet_file = load_path / "target_alphabet.json"
        if target_alphabet_file.exists():
            with open(target_alphabet_file, "r", encoding="utf-8") as f:
                self.target_alphabet = tuple(json.load(f))
        else:
            self.target_alphabet = ()
        
        # Clear projection cache
        self._projection_cache.clear()
        
        self._refresh_merge_alphabet_size()
        self._prime_ltm_order0_cache()
        
        if self.verbosity >= 1:
            print(f"[+] Loaded LTM from {load_path}")
            print(f"    - {len(self.ltm_graphs)} order graphs")
            print(f"    - Source alphabet: {len(self.alphabet)} symbols")
            if self.target_alphabet:
                print(f"    - Target alphabet: {len(self.target_alphabet)} symbols")

    def reset_stm(self) -> None:
        """Reset the online STM state."""

        self._stm_state = _STMState(orders=self.orders)

    def prime_stm(
        self,
        history: Sequence[NoteInfo],
        *,
        reset: bool = True,
        maxlen: Optional[int] = None,
    ) -> None:
        """Prime STM with a history sequence before a prediction step."""

        if not self.use_stm:
            return

        if reset or self._stm_state is None:
            self.reset_stm()

        seq = list(history)
        if maxlen is not None and int(maxlen) > 0:
            seq = seq[-int(maxlen):]

        for note in seq:
            self._stm_update_with_note(note)

    def observe_notes(self, notes: Sequence[NoteInfo]) -> None:
        """Append observed notes to the current STM once each."""

        if not self.use_stm:
            return

        if self._stm_state is None:
            self.reset_stm()

        for note in notes:
            self._stm_update_with_note(note)

    def history_from_midi_numbers(
        self,
        midi_history: Sequence[int],
        *,
        default_duration: float = 1.0,
        default_length: int = 24,
    ) -> List[NoteInfo]:
        """Convert user-provided MIDI numbers into NoteInfo history.

        This is the interactive counterpart of parser output and should be used
        when priming STM/predicting without reading a MIDI file.
        """
        return midi_history_to_noteinfos(
            midi_history,
            default_duration=float(default_duration),
            default_length=int(default_length),
            beat_duration=float(self.codec.beat_duration),
        )

    def source_symbols_from_midi_numbers(
        self,
        midi_history: Sequence[int],
        *,
        default_duration: float = 1.0,
        default_length: int = 24,
    ) -> Tuple[Optional[Symbol], ...]:
        """Project MIDI-number history into this model's source viewpoint symbols.

        This gives the same symbol sequence that parser-derived notes would produce
        through `codec.symbol_at_index(...)`.
        """
        history = self.history_from_midi_numbers(
            midi_history,
            default_duration=default_duration,
            default_length=default_length,
        )
        return tuple(self.codec.symbol_at_index(history, i) for i in range(len(history)))

    def prime_stm_from_midi_numbers(
        self,
        midi_history: Sequence[int],
        *,
        reset: bool = True,
        maxlen: Optional[int] = None,
        default_duration: float = 1.0,
        default_length: int = 24,
    ) -> None:
        """Prime STM directly from interactive MIDI-number history."""
        history = self.history_from_midi_numbers(
            midi_history,
            default_duration=default_duration,
            default_length=default_length,
        )
        self.prime_stm(history, reset=bool(reset), maxlen=maxlen)

    # -------------------------
    # Public prediction API
    # -------------------------

    def predict_next_dist(
        self,
        history: Sequence[NoteInfo],
        *,
        short_term_only: bool = False,
        long_term_only: bool = False,
    ) -> Dist:
        """Predict the next-symbol distribution given history notes."""

        if long_term_only and short_term_only:
            raise ValueError("Cannot set both short_term_only and long_term_only")

        # If no LTM has been trained, we can only use STM.
        has_ltm = bool(self.ltm_graphs)

        if long_term_only or not self.use_stm or self._stm_state is None:
            if not has_ltm:
                return self._fallback_uniform_dist()
            return self._predict_with_model_graphs(self.ltm_graphs, history)

        if short_term_only:
            return self._predict_with_model_graphs(self._stm_state.graphs, history)

        # Combine LTM and STM (if LTM exists)
        dist_stm = self._predict_with_model_graphs(self._stm_state.graphs, history)
        if not has_ltm:
            return dist_stm

        dist_ltm = self._predict_with_model_graphs(self.ltm_graphs, history)

        # If one side is empty, return the other.
        if not dist_ltm:
            return dist_stm
        if not dist_stm:
            return dist_ltm

        return self.ltm_stm_merge.merge([dist_ltm, dist_stm])

    def process_file(
        self,
        path: str,
        *,
        reset_stm: bool = True,
        short_term_only: bool = False,
        long_term_only: bool = False,
        prob_floor: float = 1e-15,
        export_stm_graphml: bool = False,
        stm_graphml_base: Optional[str] = None,
        return_trace: bool = False,
        trace_config: Optional[TraceConfig] = None,
    ) -> Union[ProcessedSequence, Tuple[ProcessedSequence, Trace]]:
        """Run the model over a file.

        - Always returns per-step probabilities/surprisals as `ProcessedSequence`.
        - If `return_trace=True`, also returns a `Trace` containing rich diagnostics
          (per-order entropies/weights, per-model merge info, and final merge info).
        """

        if long_term_only and short_term_only:
            raise ValueError("Cannot set both short_term_only and long_term_only")

        if reset_stm and self.use_stm:
            self.reset_stm()

        cfg = trace_config or TraceConfig()

        # Parse the file to retrieve all viewpoints, including the new MIDI-based ones
        seq = self.parser.parse_file(path)
        if seq:
            # Align default behavior with IDyOM: allow test-sequence symbols to be
            # part of the predictive support (dataset + pretraining alphabets).
            self.extend_alphabet_from_sequences([seq])
        steps: List[StepPrediction] = []
        trace_steps: List[StepTrace] = []

        # history is the sequence of previous notes already "heard" by the STM
        history: List[NoteInfo] = []

        for i in range(len(seq)):
            observed = self.codec.symbol_at_index(seq, i)

            # --- build distributions + diagnostics ---
            dist_final: Dist = {}
            ltm_trace: Optional[ModelTrace] = None
            stm_trace: Optional[ModelTrace] = None
            merge_trace: Optional[MergeTrace] = None

            has_ltm = bool(self.ltm_graphs)
            has_stm = bool(self.use_stm and (self._stm_state is not None))

            if long_term_only or not has_stm:
                if has_ltm:
                    dist_final, ltm_trace = self._predict_with_model_graphs_trace(
                        name="ltm", graphs=self.ltm_graphs, history=history, observed=observed, current_note=seq[i], cfg=cfg
                    )
                else:
                    dist_final = self._fallback_uniform_dist()
            elif short_term_only:
                dist_final, stm_trace = self._predict_with_model_graphs_trace(
                    name="stm", graphs=self._stm_state.graphs, history=history, observed=observed, current_note=seq[i], cfg=cfg
                )
            else:
                # both enabled: compute each then merge
                dist_stm, stm_trace = self._predict_with_model_graphs_trace(
                    name="stm", graphs=self._stm_state.graphs, history=history, observed=observed, current_note=seq[i], cfg=cfg
                )

                if not has_ltm:
                    dist_final = dist_stm
                else:
                    dist_ltm, ltm_trace = self._predict_with_model_graphs_trace(
                        name="ltm", graphs=self.ltm_graphs, history=history, observed=observed, current_note=seq[i], cfg=cfg
                    )

                    if not dist_ltm:
                        dist_final = dist_stm
                    elif not dist_stm:
                        dist_final = dist_ltm
                    else:
                        merge_ws, merge_trace = self._merge_trace_for_components(
                            labels=("ltm", "stm"),
                            dists=(dist_ltm, dist_stm),
                            strategy=self.ltm_stm_merge,
                        )
                        dist_final = self.ltm_stm_merge.merge([dist_ltm, dist_stm], weights=merge_ws)

            # --- compute probability + surprisal for observed ---
            observed_lookup = self._observed_symbol_for_lookup(
                history=history,
                current_note=seq[i],
                observed=observed,
            )
            p = float(dist_final.get(observed_lookup, 0.0)) if observed_lookup is not None else 0.0
            
            if p <= 0.0:
                p = max(float(self._fallback_prob()), float(prob_floor))
            p = max(p, float(prob_floor))
            surprisal = -math.log(p, 2)

            steps.append(StepPrediction(index=i, observed=observed, prob=p, surprisal=surprisal))

            if return_trace:
                st = StepTrace(
                    index=i,
                    observed=observed,
                    ltm=ltm_trace,
                    stm=stm_trace,
                    ltm_stm_merge=merge_trace,
                    final_entropy=(float(dist_entropy_bits(dist_final)) if dist_final else None),
                    final_rel_entropy=(float(dist_relative_entropy(dist_final)) if dist_final else None),
                    final_p_obs=float(p) if observed is not None else None,
                    final_surprisal_obs=float(surprisal) if observed is not None else None,
                    final_dist=(dict(dist_final) if (cfg.store_full_dists and dist_final) else None),
                )
                trace_steps.append(st)

            # Update STM with the newly observed note
            history.append(seq[i])
            if self.use_stm and self._stm_state is not None:
                self._stm_update_with_note(seq[i])

        processed = ProcessedSequence(steps=tuple(steps), alphabet=self.alphabet)

        # Export STM graphs only after the complete file has been processed.
        if export_stm_graphml and self.use_stm and self._stm_state is not None:
            base = stm_graphml_base
            if not base:
                base = os.path.splitext(os.path.basename(path))[0] or "stm"
            self.graph_builder._export_graphs(  # type: ignore[attr-defined]
                self._stm_state.graphs,
                base=base,
                export_dir=self.graph_builder.export_dir_stm,
            )

        if not return_trace:
            return processed

        return processed, Trace(steps=tuple(trace_steps), cfg=cfg)

    # -------------------------
    # Internal: prediction from one set of graphs (LTM or STM)
    # -------------------------

    def _predict_with_model_graphs_trace(
        self,
        *,
        name: str,
        graphs: Mapping[int, nx.DiGraph],
        history: Sequence[NoteInfo],
        observed: Optional[Symbol],
        current_note: Optional[NoteInfo],
        cfg: TraceConfig,
    ) -> Tuple[Dist, ModelTrace]:
        """Return (merged_dist, ModelTrace) for either LTM or STM."""

        if not graphs:
            mt = ModelTrace(name=name, merge_strategy="none")
            return self._fallback_uniform_dist(), mt

        # --- PPM path (count-based) ---
        if self.use_ppm:
            # Collect counts for each order in high->low order.
            counts_by_order: List[Tuple[int, Mapping[Symbol, float]]] = []
            use_update_exclusion = bool(self._ppm_update_exclusion_for_graphs(graphs))

            # Track PPM escape allocation to produce coherent "weights".
            escape = 1.0
            per_order_trace: Dict[int, OrderTrace] = {}

            for k in sorted(self.orders, reverse=True):
                ctx = self._context_node_label(history, k)
                if ctx is None:
                    continue
                gk = graphs.get(k)
                if gk is None:
                    continue

                ck = counts_from_out_edges(
                    gk,
                    ctx,
                    order_k=k,
                    codec=self.codec,
                    use_update_exclusion=use_update_exclusion,
                )
                counts_by_order.append((k, ck))

                total = float(sum(float(v) for v in ck.values()))
                unique = sum(1 for v in ck.values() if float(v) > 0.0)
                dk: Dist = {}
                esc_after = float(escape)

                ot = OrderTrace(order=int(k))

                if total > 0.0:
                    # Conditional dist for entropy diagnostics
                    dk = normalize_counts(dict(ck))
                    ot.dist = dict(dk) if (cfg.store_per_order_dists and dk) else None
                    ot.entropy = float(dist_entropy_bits(dk)) if dk else None
                    ot.rel_entropy = float(dist_relative_entropy(dk)) if dk else None

                    denom = float(total + unique)
                    mass = (escape * (total / denom)) if denom > 0.0 else 0.0
                    esc_after = (escape * (unique / denom)) if denom > 0.0 else escape

                    ot.weight = float(mass)
                    ot.escape_before = float(escape)
                    ot.escape_after = float(esc_after)
                    ot.total = float(total)
                    ot.unique = int(unique)
                    ot.unseen_context = False
                    # record the context node label used for counts (debug)
                    ot.context_node = ctx

                if observed is not None and dk:
                    p_obs = float(dk.get(observed, 0.0))
                    ot.p_obs = p_obs if p_obs > 0.0 else float(self._fallback_prob())
                    ot.surprisal_obs = -math.log(max(ot.p_obs, 1e-15), 2)

                    escape = esc_after
                else:
                    # Unseen context handling is configurable for IDyOM-compat experiments.
                    ot.weight = 0.0
                    ot.escape_before = float(escape)
                    if self.ppm_reset_escape_on_unseen:
                        ot.escape_after = 1.0
                    else:
                        ot.escape_after = float(escape)
                    ot.total = float(total)
                    ot.unique = int(unique)
                    ot.unseen_context = True
                    # still record ctx for debugging
                    ot.context_node = ctx
                    if self.ppm_reset_escape_on_unseen:
                        escape = 1.0

                per_order_trace[int(k)] = ot

            # Order 0 counts
            c0 = order0_counts_from_graphs(
                graphs,
                orders=self.orders,
                codec=self.codec,
                use_update_exclusion=use_update_exclusion,
            )
            total0 = float(sum(float(v) for v in c0.values()))
            unique0 = sum(1 for v in c0.values() if float(v) > 0.0)

            if total0 > 0.0:
                d0 = normalize_counts(dict(c0))
                counts_by_order.append((0, c0))

                ot0 = OrderTrace(order=0)
                ot0.dist = dict(d0) if (cfg.store_per_order_dists and d0) else None
                ot0.entropy = float(dist_entropy_bits(d0)) if d0 else None
                ot0.rel_entropy = float(dist_relative_entropy(d0)) if d0 else None

                denom0 = float(total0 + unique0)
                mass0 = (escape * (total0 / denom0)) if denom0 > 0.0 else 0.0
                esc_after0 = (escape * (unique0 / denom0)) if denom0 > 0.0 else escape

                ot0.weight = float(mass0)
                ot0.escape_before = float(escape)
                ot0.escape_after = float(esc_after0)
                ot0.total = float(total0)
                ot0.unique = int(unique0)
                ot0.unseen_context = False
                # order0 has no context node
                ot0.context_node = None

                if observed is not None and d0:
                    p_obs = float(d0.get(observed, 0.0))
                    ot0.p_obs = p_obs if p_obs > 0.0 else float(self._fallback_prob())
                    ot0.surprisal_obs = -math.log(max(ot0.p_obs, 1e-15), 2)

                escape = esc_after0
                per_order_trace[0] = ot0

            # Order -1 uniform mass
            if self.alphabet:
                uni_p = 1.0 / float(len(self.alphabet))
            else:
                uni_p = float(self._fallback_prob())

            otm1 = OrderTrace(order=-1)
            otm1.weight = float(escape)
            otm1.entropy = None
            otm1.rel_entropy = 1.0
            if observed is not None:
                otm1.p_obs = float(uni_p)
                otm1.surprisal_obs = -math.log(max(float(uni_p), 1e-15), 2)
            per_order_trace[-1] = otm1

            # Build PPM distribution from counts in high->low order (+ order0 at the end)
            counts_list: List[Mapping[Symbol, float]] = [
                c for order, c in sorted(counts_by_order, key=lambda x: -x[0]) if order != 0
            ]
            counts_list.append(c0)

            dist = self.ppm.dist_from_counts(
                counts_list,
                alphabet=self.alphabet or None,
                excluded_count=self.ppm_excluded_count,
                exclusion=self._ppm_exclusion_for_graphs(graphs),
                update_exclusion=self._ppm_update_exclusion_for_graphs(graphs),
            )
            
            # Project distribution to target viewpoint if specified
            if self.target_viewpoint is not None:
                dist = self._project_dist_to_target(dist, history)
            observed_lookup = self._observed_symbol_for_lookup(
                history=history,
                current_note=current_note,
                observed=observed,
            )
            observed_p = dist.get(observed_lookup, 0.0) if observed_lookup is not None else 0.0

            mt = ModelTrace(
                name=name,
                merge_strategy=getattr(self.ppm, "name", "ppm"),
                per_order=per_order_trace,
                merged_entropy=float(dist_entropy_bits(dist)) if dist else None,
                merged_rel_entropy=float(dist_relative_entropy(dist)) if dist else None,
                merged_p_obs=(float(observed_p) if observed is not None else None),
                merged_surprisal_obs=(
                    (-math.log(max(float(observed_p), 1e-15), 2)) if observed is not None else None),
                merged_dist=(dict(dist) if (cfg.store_full_dists and dist) else None),
                extra={"alphabet_size": int(len(self.alphabet)) if self.alphabet else None},
            )

            return dist, mt

        # --- Entropy-weighted order merge path ---
        per_order_dists: List[Tuple[int, Dist]] = []

        for k in self.orders:
            ctx = self._context_node_label(history, k)
            if ctx is None:
                continue
            gk = graphs.get(k)
            if gk is None:
                continue
            dk = dist_from_out_edges(gk, ctx, order_k=k, codec=self.codec)
            if dk:
                per_order_dists.append((int(k), dk))

        # Order 0
        c0 = order0_counts_from_graphs(graphs, orders=self.orders, codec=self.codec)
        d0 = normalize_counts(c0)
        if d0:
            per_order_dists.append((0, d0))

        if not per_order_dists:
            mt = ModelTrace(name=name, merge_strategy=getattr(self.order_merge, "name", "order_merge"))
            return self._fallback_uniform_dist(), mt

        orders_with_dists = [k for k, _ in per_order_dists]
        dists_for_merge = [d for _, d in per_order_dists]

        # Compute rel entropies and weights coherently with merge.py
        w_mode = getattr(self.order_merge, "weight_mode", "inverse_power")
        b = float(getattr(self.order_merge, "b", 1.0))
        ws = entropy_weights(
            dists_for_merge,
            alphabet_size=getattr(self.order_merge, "alphabet_size", None),
            mode=str(w_mode),
            b=b,
            eps=1e-15,
        )

        per_order_trace: Dict[int, OrderTrace] = {}
        for k, d, w in zip(orders_with_dists, dists_for_merge, ws):
            ot = OrderTrace(order=int(k))
            ot.dist = dict(d) if (cfg.store_per_order_dists and d) else None
            ot.entropy = float(dist_entropy_bits(d)) if d else None
            ot.rel_entropy = float(dist_relative_entropy(d)) if d else None
            ot.weight = float(w)

            if observed is not None and d:
                p_obs = float(d.get(observed, 0.0))
                ot.p_obs = p_obs if p_obs > 0.0 else float(self._fallback_prob())
                ot.surprisal_obs = -math.log(max(ot.p_obs, 1e-15), 2)

            per_order_trace[int(k)] = ot

        merged = self.order_merge.merge(dists_for_merge, weights=ws)
        
        # Project distribution to target viewpoint if specified
        if self.target_viewpoint is not None:
            merged = self._project_dist_to_target(merged, history)
        observed_lookup = self._observed_symbol_for_lookup(
            history=history,
            current_note=current_note,
            observed=observed,
        )
        observed_p = merged.get(observed_lookup, 0.0) if observed_lookup is not None else 0.0

        mt = ModelTrace(
            name=name,
            merge_strategy=getattr(self.order_merge, "name", "order_merge"),
            per_order=per_order_trace,
            merged_entropy=float(dist_entropy_bits(merged)) if merged else None,
            merged_rel_entropy=float(dist_relative_entropy(merged)) if merged else None,
            merged_p_obs=(float(observed_p) if observed is not None else None),
            merged_surprisal_obs=(
                (-math.log(max(float(observed_p), 1e-15), 2)) if observed is not None else None),
            merged_dist=(dict(merged) if (cfg.store_full_dists and merged) else None),
        )

        return merged, mt

    def _predict_with_model_graphs(self, graphs: Mapping[int, nx.DiGraph], history: Sequence[NoteInfo]) -> Dist:
        """Predict distribution using a single model's graphs (LTM or STM).

        - If self.use_ppm is True: compute counts per order and build PPM dist.
        - Else: compute per-order dists and merge across orders using self.order_merge.
        """

        if not graphs:
            return self._fallback_uniform_dist()

        if self.use_ppm:
            # Build counts from highest order to lowest order, then include order-0 counts.
            counts_list: List[Mapping[Symbol, float]] = []
            use_update_exclusion = bool(self._ppm_update_exclusion_for_graphs(graphs))

            for k in sorted(self.orders, reverse=True):
                ctx = self._context_node_label(history, k)
                if ctx is None:
                    continue
                gk = graphs.get(k)
                if gk is None:
                    continue
                ck = counts_from_out_edges(
                    gk,
                    ctx,
                    order_k=k,
                    codec=self.codec,
                    use_update_exclusion=use_update_exclusion,
                )
                counts_list.append(ck)

            # Order 0 counts from node weights (preferred)
            c0 = order0_counts_from_graphs(
                graphs,
                orders=self.orders,
                codec=self.codec,
                use_update_exclusion=use_update_exclusion,
            )
            counts_list.append(c0)

            dist = self.ppm.dist_from_counts(
                counts_list,
                alphabet=self.alphabet or None,
                excluded_count=self.ppm_excluded_count,
                exclusion=self._ppm_exclusion_for_graphs(graphs),
                update_exclusion=self._ppm_update_exclusion_for_graphs(graphs),
            )
            
            # Project distribution to target viewpoint if specified
            if self.target_viewpoint is not None:
                dist = self._project_dist_to_target(dist, history)
            
            return dist

        # Non-PPM: compute per-order dists and merge
        per_order_dists: List[Dist] = []

        for k in self.orders:
            ctx = self._context_node_label(history, k)
            if ctx is None:
                continue
            gk = graphs.get(k)
            if gk is None:
                continue
            dk = dist_from_out_edges(gk, ctx, order_k=k, codec=self.codec)
            if dk:
                per_order_dists.append(dk)

        # Order 0
        c0 = order0_counts_from_graphs(graphs, orders=self.orders, codec=self.codec)
        d0 = normalize_counts(c0)
        if d0:
            per_order_dists.append(d0)

        if not per_order_dists:
            return self._fallback_uniform_dist()

        merged = self.order_merge.merge(per_order_dists)
        
        # Project distribution to target viewpoint if specified
        if self.target_viewpoint is not None:
            merged = self._project_dist_to_target(merged, history)
        
        return merged

    def _window_size_for_order(self, order: int) -> int:
        fn = getattr(self.codec, "window_size_for_order", None)
        if callable(fn):
            return max(0, int(fn(int(order))))
        return max(0, int(order))

    def _context_node_label(self, history: Sequence[NoteInfo], order: int) -> Optional[str]:
        window_size = self._window_size_for_order(order)
        if window_size <= 0:
            return None
        if len(history) < window_size:
            return None
        window = history[-window_size:]
        return self.codec.window_label(window)

    # -------------------------
    # Internal: STM incremental update (faithful to GraphBuilder online logic)
    # -------------------------

    def _stm_update_with_note(self, note: NoteInfo) -> None:
        if self._stm_state is None:
            return

        if self.graph_build_config.group_by_beat:
            # Group-by-beat online updates are not implemented yet.
            # We can add them later if needed.
            raise NotImplementedError("Online STM updates with group_by_beat=True are not implemented")

        ready_updates: Dict[int, Tuple[_STMOrderState, nx.DiGraph, List[NoteInfo], str]] = {}
        for k in self.orders:
            state = self._stm_state.per_order[k]
            g = self._stm_state.graphs[k]
            window_size = self._window_size_for_order(k)

            # update ring buffer
            state.window.append(note)
            if len(state.window) > window_size:
                state.window = state.window[-window_size:]

            if len(state.window) < window_size:
                continue

            window = list(state.window)
            node = self.codec.window_label(window)
            ready_updates[int(k)] = (state, g, window, node)

        # Evaluate update-exclusion against *pre-update* higher-order contexts.
        update_excluded_increment: Dict[int, bool] = {}
        orders_ready = sorted(ready_updates.keys())
        for k in orders_ready:
            _, _, _, node_k = ready_updates[k]
            sym_k = self.codec.symbol_from_dest_label(node_k, int(k))
            predicted_higher = False
            if sym_k is not None:
                for j in orders_ready:
                    if int(j) <= int(k):
                        continue
                    state_j, g_j, _, _ = ready_updates[j]
                    ctx_j = state_j.prev_node
                    if ctx_j is None:
                        continue
                    counts_j = counts_from_out_edges(
                        g_j,
                        ctx_j,
                        order_k=int(j),
                        codec=self.codec,
                        use_update_exclusion=False,
                    )
                    if float(counts_j.get(sym_k, 0.0)) > 0.0:
                        predicted_higher = True
                        break
            update_excluded_increment[int(k)] = not predicted_higher

        # Apply updates for each order, always incrementing standard counts.
        for k in self.orders:
            payload = ready_updates.get(int(k))
            if payload is None:
                continue
            state, g, window, node = payload
            inc_update_exclusion = bool(update_excluded_increment.get(int(k), True))

            # add node/update weights
            self._add_or_update_node(
                g,
                node,
                window,
                update_exclusion=inc_update_exclusion,
            )

            # add edge if allowed
            if state.end_time is not None:
                window_size = self._window_size_for_order(k)
                time_diff = 0.0 if window_size > 1 else (window[0].timestamp - state.end_time)
                if time_diff <= self.graph_build_config.max_link_time_diff:
                    self._add_or_update_edge(
                        g,
                        state.prev_node,
                        node,
                        weight=1.0,
                        update_exclusion=inc_update_exclusion,
                    )

            state.prev_node = node
            state.end_time = window[-1].timestamp + window[-1].duration

        # STM graphs mutate online, so bump a cheap cache version after each note.
        for g in self._stm_state.graphs.values():
            g.graph["_graph_cache_version"] = int(g.graph.get("_graph_cache_version", 0)) + 1

    def _add_or_update_node(
        self,
        net: nx.DiGraph,
        node: str,
        window_infos: Sequence[NoteInfo],
        *,
        update_exclusion: bool = True,
    ) -> None:
        total_duration = float(sum(info.duration for info in window_infos))
        if not net.has_node(node):
            net.add_node(
                node,
                weight=1,
                duration_weight=total_duration,
                weight_ux=(1.0 if update_exclusion else 0.0),
            )
        else:
            prev_weight = float(net.nodes[node].get("weight", 0.0))
            net.nodes[node]["weight"] += 1
            net.nodes[node]["duration_weight"] += total_duration
            if "weight_ux" not in net.nodes[node]:
                net.nodes[node]["weight_ux"] = prev_weight
            if update_exclusion:
                net.nodes[node]["weight_ux"] += 1

    def _add_or_update_edge(
        self,
        net: nx.DiGraph,
        from_node: Optional[str],
        to_node: Optional[str],
        *,
        weight: float = 1.0,
        update_exclusion: bool = True,
    ) -> None:
        if from_node is None or to_node is None:
            return
        if net.has_edge(from_node, to_node):
            prev_weight = float(net[from_node][to_node].get("weight", 0.0))
            net[from_node][to_node]["weight"] += weight
            if "weight_ux" not in net[from_node][to_node]:
                net[from_node][to_node]["weight_ux"] = prev_weight
            if update_exclusion:
                net[from_node][to_node]["weight_ux"] += weight
        else:
            net.add_edge(
                from_node,
                to_node,
                weight=weight,
                weight_ux=(float(weight) if update_exclusion else 0.0),
            )

    # Target projection/validation methods are provided by TargetProjectionMixin.

    def _fallback_prob(self) -> float:
        # IDyOM code often uses 1/30 as a fallback. We emulate that by using the
        # known alphabet size when available.
        if self.target_viewpoint is not None and self.target_alphabet:
            return 1.0 / float(len(self.target_alphabet))
        if self.alphabet:
            return 1.0 / float(len(self.alphabet))
        return 1.0 / 30.0

    def _fallback_uniform_dist(self) -> Dist:
        if self.target_viewpoint is not None and self.target_alphabet:
            u = 1.0 / float(len(self.target_alphabet))
            return {s: u for s in self.target_alphabet}
        if self.alphabet:
            u = 1.0 / float(len(self.alphabet))
            return {s: u for s in self.alphabet}
        return {}
