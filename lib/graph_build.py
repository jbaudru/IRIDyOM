"""graph_build.py

Graph construction utilities (LTM folder graphs + STM graphs) for GraphIDyOM.

This module is intentionally viewpoint/token-agnostic:
- MIDI parsing is delegated to a MidiParser.
- Node label encoding is delegated to a TokenCodec.

It aims to mirror the behavior of the original `SimpleMidiFolderGraphs` graph
construction (minus augmentation, which we will reintroduce later in a dedicated
augmentation module).

Key behaviors preserved
- Supports order-n context windows (offline) and deque-based online building.
- Optional grouping by beat bucket (group_by_beat).
- Edge linking rule: for order==1 only, link if inter-onset gap <= max_link_time_diff;
  for order>1, always link successive windows (time_diff treated as 0.0).
- Node weights and duration_weight accumulation.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
from tqdm import tqdm

from graph_types import MidiParser, NoteInfo, TokenCodec


@dataclass(frozen=True)
class GraphBuildConfig:
    """Graph construction settings (not viewpoints)."""

    max_link_time_diff: float = 4.0
    group_by_beat: bool = False
    augment: bool = False
    augment_transposition: bool = True
    transpose_range: Tuple[int, int] = (-6, 6)
    augment_rhythm: bool = True
    threshold_fast: float = 10.0
    threshold_slow: float = 24.0
    rhythm_factors_fast: Tuple[float, ...] = (2.0, 4.0)
    rhythm_factors_slow: Tuple[float, ...] = (0.5, 0.25)


class GraphBuilder:
    """Builds directed graphs for different Markov orders from NoteInfo sequences."""

    def __init__(
        self,
        *,
        parser: MidiParser,
        codec: TokenCodec,
        config: GraphBuildConfig = GraphBuildConfig(),
        verbosity: int = 1,
        outfolder: str = "results",
        export_subdir: str = "graphs",
    ):
        self.parser = parser
        self.codec = codec
        self.config = config
        self.verbosity = int(verbosity)

        export_root = str(outfolder)
        if os.path.exists(export_root) and not os.path.isdir(export_root):
            fallback_root = os.path.join(tempfile.gettempdir(), "graphidyom_results")
            if self.verbosity >= 1:
                print(
                    f"[warn] GraphBuilder outfolder is not a directory: {export_root}. "
                    f"Using fallback: {fallback_root}"
                )
            export_root = fallback_root

        self.export_dir = os.path.join(export_root, export_subdir)
        self.export_dir_ltm = os.path.join(self.export_dir, "LTM")
        self.export_dir_stm = os.path.join(self.export_dir, "STM")
        try:
            os.makedirs(self.export_dir_ltm, exist_ok=True)
            os.makedirs(self.export_dir_stm, exist_ok=True)
        except OSError:
            fallback_root = os.path.join(tempfile.gettempdir(), "graphidyom_results")
            self.export_dir = os.path.join(fallback_root, export_subdir)
            self.export_dir_ltm = os.path.join(self.export_dir, "LTM")
            self.export_dir_stm = os.path.join(self.export_dir, "STM")
            os.makedirs(self.export_dir_ltm, exist_ok=True)
            os.makedirs(self.export_dir_stm, exist_ok=True)

    # -------------------------
    # Public API
    # -------------------------

    def build_folder_graphs(
        self,
        folder: str,
        *,
        orders: Sequence[int] = (1,),
        export_graphml: bool = True,
        extensions: Tuple[str, ...] = (".mid", ".midi", ".xml", ".mxl", ".musicxml"),
        include_augmentation: Optional[bool] = None,
    ) -> Dict[int, nx.DiGraph]:
        """Build LTM graphs from all supported files in a folder."""

        orders = self._validate_orders(orders)
        use_aug = self.config.augment if include_augmentation is None else bool(include_augmentation)

        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in extensions
        ]
        files.sort()
        if not files:
            raise RuntimeError(f"No supported music files found in: {folder}")

        graphs: Dict[int, nx.DiGraph] = {o: nx.DiGraph() for o in orders}

        skipped_empty = 0
        skipped_read = 0
        skipped_other = 0

        self._print_if_useful(f"[+] Found {len(files)} files in {folder}", 1)
        self._print_if_useful(f"[+] Export dir (LTM): {self.export_dir_ltm}", 1)

        for file_path in tqdm(files, desc="Parsing music files (LTM)"):
            try:
                seq = self.parser.parse_file(file_path)
            except Exception:
                skipped_read += 1
                continue

            if not seq:
                skipped_empty += 1
                continue

            try:
                seqs = self._all_augmented_sequences(seq) if use_aug else [seq]
                for seq_i in seqs:
                    for order in orders:
                        self._process_sequence_offline(graphs[order], seq_i, order)
            except Exception:
                skipped_other += 1
                continue

        self._print_if_useful(
            f"[+] Done LTM. Skipped: empty={skipped_empty}, unreadable={skipped_read}, other={skipped_other}",
            1,
        )

        if export_graphml:
            base = os.path.basename(os.path.abspath(folder)) or "folder"
            self._export_graphs(graphs, base=base, export_dir=self.export_dir_ltm)

        return graphs

    def build_stm_graphs_for_file(
        self,
        midi_file: str,
        *,
        orders: Sequence[int] = (1,),
        export_graphml: bool = True,
        online: bool = True,
    ) -> Dict[int, nx.DiGraph]:
        """Build STM graphs for a single file.

        online=True uses a deque window (streaming-friendly) but yields identical results
        to offline mode for monophonic sequences when group_by_beat=False.
        """

        orders = self._validate_orders(orders)

        seq = self.parser.parse_file(midi_file)
        if not seq:
            raise RuntimeError(f"Empty file for STM: {midi_file}")

        graphs: Dict[int, nx.DiGraph] = {o: nx.DiGraph() for o in orders}

        for order in orders:
            if online:
                self._process_sequence_online(graphs[order], seq, order)
            else:
                self._process_sequence_offline(graphs[order], seq, order)

        if export_graphml:
            base = os.path.splitext(os.path.basename(midi_file))[0] or "file"
            self._export_graphs(graphs, base=base, export_dir=self.export_dir_stm)

        return graphs

    # -------------------------
    # Internal helpers
    # -------------------------

    def _validate_orders(self, orders: Sequence[int]) -> Tuple[int, ...]:
        out = tuple(int(o) for o in orders)
        for o in out:
            if o < 1:
                raise ValueError("Orders must be >= 1")
        return out

    def _print_if_useful(self, msg: str, level: int = 1) -> None:
        if level <= self.verbosity:
            print(msg)

    def _export_graphs(self, graphs: Mapping[int, nx.DiGraph], *, base: str, export_dir: str) -> None:
        for order, g in graphs.items():
            out_name = f"{base}_order{order}.graphml"
            out_path = os.path.join(export_dir, out_name)
            self._print_if_useful(f"[+] Writing graph: {out_path}", 1)
            nx.write_graphml(self._graph_with_export_attrs(g), out_path)

    def _graph_with_export_attrs(self, graph: nx.DiGraph) -> nx.DiGraph:
        export_graph = nx.DiGraph()
        export_graph.graph.update(
            {
                key: value
                for key, value in graph.graph.items()
                if not str(key).startswith("_")
            }
        )

        for node, attrs in graph.nodes(data=True):
            export_graph.add_node(node, **dict(attrs))

        for src, dst, attrs in graph.edges(data=True):
            export_graph.add_edge(src, dst, **dict(attrs))

        return export_graph

    def _add_or_update_node(self, net: nx.DiGraph, node: str, window_infos: Sequence[NoteInfo]) -> None:
        total_duration = float(sum(info.duration for info in window_infos))
        if not net.has_node(node):
            net.add_node(node, weight=1, duration_weight=total_duration)
        else:
            net.nodes[node]["weight"] += 1
            net.nodes[node]["duration_weight"] += total_duration

    def _add_or_update_edge(
        self,
        net: nx.DiGraph,
        from_node: Optional[str],
        to_node: Optional[str],
        *,
        weight: float = 1.0,
    ) -> None:
        if from_node is None or to_node is None:
            return
        if net.has_edge(from_node, to_node):
            net[from_node][to_node]["weight"] += weight
        else:
            net.add_edge(from_node, to_node, weight=weight)

    def _augment_transpositions(self, seq: Sequence[NoteInfo]) -> List[List[NoteInfo]]:
        if not (self.config.augment and self.config.augment_transposition):
            return [list(seq)]
        tmin, tmax = self.config.transpose_range
        out: List[List[NoteInfo]] = []
        for t in range(int(tmin), int(tmax)):
            new_seq: List[NoteInfo] = []
            for info in seq:
                if not hasattr(self.codec, "pitch_str_transpose") or not callable(getattr(self.codec, "pitch_str_transpose")):
                    raise RuntimeError("Transposition augmentation requires codec.pitch_str_transpose(...)")
                new_pitch = self.codec.pitch_str_transpose(info.pitch, t, octave=True)
                new_pc = self.codec.pitch_str_transpose(info.pitch_class, t, octave=False)
                # Transpose midi number by semitones as well
                new_midi = int(info.midi) + int(t)
                new_info = NoteInfo(
                    timestamp=info.timestamp,
                    duration=info.duration,
                    offset=info.offset,
                    pitch=new_pitch,
                    pitch_class=new_pc,
                    midi=new_midi,
                    length=info.length,
                )
                new_seq.append(new_info)
            out.append(new_seq)
        return out

    def _augment_rhythm(self, seq: Sequence[NoteInfo]) -> List[List[NoteInfo]]:
        if not (self.config.augment and self.config.augment_rhythm):
            return [list(seq)]
        if not seq:
            return [list(seq)]
        mean_d = sum(info.duration for info in seq) / len(seq)
        threshold_fast = self.config.threshold_fast
        threshold_slow = self.config.threshold_slow
        rhythm_factors_fast = self.config.rhythm_factors_fast
        rhythm_factors_slow = self.config.rhythm_factors_slow

        if mean_d > threshold_slow:
            factors = rhythm_factors_slow
        elif mean_d < threshold_fast:
            factors = rhythm_factors_fast
        else:
            mid = (threshold_slow + threshold_fast) / 2.0
            if mean_d > mid:
                factors = rhythm_factors_slow[:1]
            else:
                factors = rhythm_factors_fast[:1]

        out: List[List[NoteInfo]] = [list(seq)]
        for f in factors:
            new_seq: List[NoteInfo] = []
            for info in seq:
                new_dur = max(1e-6, info.duration * f)
                # Scale length proportionally (quantized integers, rounded like in IDyOM's data.py)
                # When factor < 1 (slow to fast), use round; when factor > 1 (fast to slow), direct multiply
                if info.length > 0:
                    if f < 1.0:
                        new_length = int(round(info.length * f))
                    else:
                        new_length = int(info.length * f)
                    new_length = max(1, new_length)
                else:
                    new_length = 0
                    
                new_info = NoteInfo(
                    timestamp=info.timestamp,
                    duration=new_dur,
                    offset=info.offset,
                    pitch=info.pitch,
                    pitch_class=info.pitch_class,
                    midi=info.midi,
                    length=new_length,
                )
                new_seq.append(new_info)
            out.append(new_seq)
        return out

    def _all_augmented_sequences(self, seq: Sequence[NoteInfo]) -> List[List[NoteInfo]]:
        if not self.config.augment:
            return [list(seq)]
        seqs = self._augment_transpositions(seq)
        out: List[List[NoteInfo]] = []
        for s in seqs:
            out.extend(self._augment_rhythm(s))
        return out

    # -------------------------
    # Core graph construction logic (faithful)
    # -------------------------

    def _window_size_for_order(self, order: int) -> int:
        fn = getattr(self.codec, "window_size_for_order", None)
        if callable(fn):
            return max(0, int(fn(int(order))))
        return max(0, int(order))

    def _process_sequence_offline(self, net: nx.DiGraph, seq: Sequence[NoteInfo], order: int) -> None:
        window_size = self._window_size_for_order(order)
        if len(seq) < window_size:
            return

        if self.config.group_by_beat:
            self._process_sequence_grouped_by_beat(net, seq)
            return

        prev_node: Optional[str] = None
        end_time: Optional[float] = None

        for i in range(len(seq) + 1 - window_size):
            window = seq[i : i + window_size]
            node = self.codec.window_label(window)

            self._add_or_update_node(net, node, window)

            if end_time is not None:
                time_diff = 0.0 if window_size > 1 else (window[0].timestamp - end_time)
                if time_diff <= self.config.max_link_time_diff:
                    self._add_or_update_edge(net, prev_node, node, weight=1.0)

            prev_node = node
            end_time = window[-1].timestamp + window[-1].duration

    def _process_sequence_online(self, net: nx.DiGraph, seq: Sequence[NoteInfo], order: int) -> None:
        window_size = self._window_size_for_order(order)
        if len(seq) < window_size:
            return

        if self.config.group_by_beat:
            # grouping-by-beat is inherently offline over timestamps
            self._process_sequence_grouped_by_beat(net, seq)
            return

        win: Deque[NoteInfo] = deque(maxlen=window_size)
        prev_node: Optional[str] = None
        end_time: Optional[float] = None

        for info in seq:
            win.append(info)
            if len(win) < window_size:
                continue

            window = list(win)
            node = self.codec.window_label(window)

            self._add_or_update_node(net, node, window)

            if end_time is not None:
                time_diff = 0.0 if window_size > 1 else (window[0].timestamp - end_time)
                if time_diff <= self.config.max_link_time_diff:
                    self._add_or_update_edge(net, prev_node, node, weight=1.0)

            prev_node = node
            end_time = window[-1].timestamp + window[-1].duration

    def _process_sequence_grouped_by_beat(self, net: nx.DiGraph, seq: Sequence[NoteInfo]) -> None:
        # Mirrors previous behavior: one node per beat bucket.
        if not seq:
            return

        beat_dur = float(self.codec.beat_duration)
        if beat_dur <= 0:
            raise ValueError("beat_duration must be > 0")

        time = math.floor(float(seq[0].timestamp))
        start_idx = 0
        prev_node: Optional[str] = None

        while start_idx < len(seq):
            end_idx = start_idx
            while end_idx < len(seq) and float(seq[end_idx].timestamp) < time + beat_dur:
                end_idx += 1

            window = seq[start_idx:end_idx]
            node = self.codec.window_label(window) if window else "empty"

            self._add_or_update_node(net, node, window)
            if prev_node is not None:
                self._add_or_update_edge(net, prev_node, node, weight=1.0)

            prev_node = node
            start_idx = end_idx
            time += beat_dur
