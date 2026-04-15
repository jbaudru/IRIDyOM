

"""token_codec.py

TokenCodec implementation responsible for:
- Encoding windows of NoteInfo into node label strings
- Providing deterministic JSON encoding for viewpoint tokens
- Interval viewpoint support (including interval-only mode)
- Decoding node labels back into the "predicted symbol" used for distributions

This aims to be faithful to the behavior in the original `SimpleMidiFolderGraphs`.

Label format (faithful)
- Normal mode: comma-joined JSON objects representing note viewpoint tokens,
  optionally followed by interval JSON objects if interval=True and order>=2.
- Interval-only mode: for k<2 => "NO_EVENT"; else comma-joined interval JSON objects ONLY.

Determinism
- json.dumps(..., sort_keys=True, separators=(",", ":"))
- Optional rounding of duration/offset via token_round_ndigits
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence

import music21 as ms

from graph_types import NodeLabel, NoteInfo, Order, Symbol, TokenCodec
from label_utils import extract_json_objects as _extract_json_objects


JSON_DUMPS_KW: Dict[str, Any] = {"sort_keys": True, "separators": (",", ":")}
BIOI_TICKS_PER_QUARTER = 24


def canonical_bioi_ratio_symbol(current_ticks: int, previous_ticks: int) -> Optional[str]:
    """Return a Lisp-like exact ratio label for consecutive BIOIs."""

    prev = int(previous_ticks)
    curr = int(current_ticks)
    if prev <= 0:
        return None
    ratio = Fraction(curr, prev)
    if ratio.denominator == 1:
        return str(ratio.numerator)
    return f"{ratio.numerator}/{ratio.denominator}"


def quantized_bioi_ticks_at(seq: Sequence[NoteInfo], i: int) -> Optional[int]:
    """Return the BIOI ending at event `i`, quantized to IDyOM's 24 ticks."""

    if i <= 0 or i >= len(seq):
        return None
    delta = float(seq[i].timestamp) - float(seq[i - 1].timestamp)
    ticks = int(round(BIOI_TICKS_PER_QUARTER * delta))
    return max(0, ticks)


def bioi_ratio_symbol_at(seq: Sequence[NoteInfo], i: int) -> Optional[str]:
    """Return the ratio between the current and previous BIOI at event `i`."""

    if i <= 1 or i >= len(seq):
        return None
    prev_ticks = quantized_bioi_ticks_at(seq, i - 1)
    curr_ticks = quantized_bioi_ticks_at(seq, i)
    if prev_ticks is None or curr_ticks is None:
        return None
    return canonical_bioi_ratio_symbol(curr_ticks, prev_ticks)


@dataclass(frozen=True)
class ViewpointConfig:
    """Which viewpoints to encode into note tokens."""

    pitch: bool = True
    octave: bool = False  # if pitch=True: False => pitch class, True => pitch+octave
    midi_number: bool = False     # If True, include integer MIDI pitch number (raw mido value)
    duration: bool = False
    length: bool = False           # If True, include IDyOM-style length token (quantized ticks)
    offset: bool = False
    interval: bool = False
    bioi_ratio: bool = False       # If True, include IDyOM-style bioi-ratio viewpoint

    beat_duration: float = 1.0

    # pitch normalization
    enharmony: bool = True

    # numeric stability for duration/offset tokens
    token_round_ndigits: Optional[int] = None


class JsonTokenCodec(TokenCodec):
    """Faithful codec mirroring the original node label construction."""

    def __init__(self, cfg: ViewpointConfig):
        self.cfg = cfg
        bd = float(cfg.beat_duration)
        if bd <= 0:
            raise ValueError("beat_duration must be > 0")

    # -------------------------
    # TokenCodec API
    # -------------------------

    @property
    def beat_duration(self) -> float:
        return float(self.cfg.beat_duration)

    def _has_basic_note_components(self) -> bool:
        return bool(
            self.cfg.pitch
            or self.cfg.midi_number
            or self.cfg.duration
            or self.cfg.length
            or self.cfg.offset
        )

    def derived_only_mode(self) -> bool:
        return bool(self.cfg.interval or self.cfg.bioi_ratio) and (not self._has_basic_note_components())

    def min_predictive_order(self) -> int:
        return 1

    def window_size_for_order(self, order: int) -> int:
        order = int(order)
        if order <= 0:
            return 0
        if not self.derived_only_mode():
            return order
        lag = 0
        if self.cfg.interval:
            lag = max(lag, 1)
        if self.cfg.bioi_ratio:
            lag = max(lag, 2)
        return order + lag

    def interval_only_mode(self) -> bool:
        return self.derived_only_mode() and bool(self.cfg.interval) and (not self.cfg.bioi_ratio)

    def window_label(self, window: Sequence[NoteInfo]) -> NodeLabel:
        """Encode a window of NoteInfo into a node label."""

        k = len(window)

        # Derived-only mode: ONLY derived tokens, no note placeholders.
        if self.derived_only_mode():
            if k < self.min_predictive_order():
                return "NO_EVENT"
            derived_tokens: List[str] = []
            if self.cfg.interval:
                for j in range(k - 1):
                    semitones = self._chromatic_interval_semitones(window[j].pitch, window[j + 1].pitch)
                    derived_tokens.append(self._interval_token(semitones))
            if self.cfg.bioi_ratio:
                for j in range(2, k):
                    ratio_symbol = bioi_ratio_symbol_at(window, j)
                    if ratio_symbol is not None:
                        derived_tokens.append(self._bioi_ratio_token(ratio_symbol))
            return ",".join(derived_tokens) if derived_tokens else "NO_EVENT"

        # Normal mode: note tokens + optionally interval tokens.
        tokens = [self._note_token_at(window, i) for i in range(k)]
        if self.cfg.interval and k >= 2:
            for j in range(k - 1):
                semitones = self._chromatic_interval_semitones(window[j].pitch, window[j + 1].pitch)
                tokens.append(self._interval_token(semitones))
        return ",".join(tokens)

    def extract_json_objects(self, label: NodeLabel) -> List[str]:
        """Extract JSON object substrings from a comma-joined label.

        The original code used a brace-matching parser; we implement the same idea
        so we are robust to commas within the stream of multiple JSON objects.
        """
        return _extract_json_objects(label)

    def symbol_from_dest_label(self, dest_label: NodeLabel, order_k: Order) -> Optional[Symbol]:
        """Extract the predicted symbol from the destination node label.

        Faithful intent:
        - In interval-only mode:
            * order_k == 1 -> "NO_EVENT" (same sentinel as label)
            * order_k >= 2 -> the last interval token JSON string
        - In normal mode:
            * If interval_only_mode is False, we return the last NOTE token JSON string
              (not interval), since those represent the 'event' at the end of the window.
              This matches how your original model used note token dicts as symbols.
        """

        if dest_label == "NO_EVENT":
            return "NO_EVENT" if self.derived_only_mode() else None

        objs = self.extract_json_objects(dest_label)
        if not objs:
            return None

        if self.derived_only_mode():
            return self._derived_symbol_from_objects(objs, int(order_k))

        # Normal mode: label begins with k note tokens, then optional (k-1) interval tokens.
        # We want the note token corresponding to the last event in the window.
        # If interval viewpoint is enabled, merge the last interval token into that note.
        # BIOI-ratio is already embedded into the note token in normal mode.
        k = int(order_k)
        if k <= 0:
            return None

        if len(objs) < k:
            # Unexpected; fallback to last object
            return objs[-1]

        note_token = objs[k - 1]

        if self.cfg.interval and k >= 2 and len(objs) >= (2 * k - 1):
            try:
                note_obj = json.loads(note_token)
                interval_obj = json.loads(objs[-1])
                if isinstance(note_obj, dict) and isinstance(interval_obj, dict) and "interval" in interval_obj:
                    merged = dict(note_obj)
                    merged["interval"] = int(interval_obj["interval"])
                    return json.dumps(merged, **JSON_DUMPS_KW)
            except (json.JSONDecodeError, TypeError, ValueError):
                return note_token

        return note_token

    def symbol_at_index(self, seq: Sequence[NoteInfo], i: int) -> Optional[Symbol]:
        """Return the symbol for the i-th event in a sequence.

        Faithful mapping:
        - interval-only: symbol is the interval from i-1 -> i (i==0 => NO_EVENT)
        - normal: symbol is the NOTE token JSON string for event i
        """

        if i < 0 or i >= len(seq):
            return None

        if self.derived_only_mode():
            token: Dict[str, Any] = {}
            if self.cfg.interval and i > 0:
                semitones = self._chromatic_interval_semitones(seq[i - 1].pitch, seq[i].pitch)
                token["interval"] = int(semitones)
            if self.cfg.bioi_ratio:
                ratio_symbol = bioi_ratio_symbol_at(seq, i)
                if ratio_symbol is not None:
                    token["bioi_ratio"] = ratio_symbol
            if not token:
                return "NO_EVENT"
            return json.dumps(token, **JSON_DUMPS_KW)

        note_token = self._note_token_at(seq, i)
        if self.cfg.interval and i > 0:
            try:
                semitones = self._chromatic_interval_semitones(seq[i - 1].pitch, seq[i].pitch)
                note_obj = json.loads(note_token)
                if isinstance(note_obj, dict):
                    note_obj["interval"] = int(semitones)
                    return json.dumps(note_obj, **JSON_DUMPS_KW)
            except (json.JSONDecodeError, TypeError, ValueError):
                return note_token
        return note_token

    # -------------------------
    # Deterministic note/interval tokens (faithful)
    # -------------------------

    def _stable_float(self, x: float) -> float:
        if self.cfg.token_round_ndigits is None:
            return float(x)
        return float(round(float(x), int(self.cfg.token_round_ndigits)))

    def _note_token_at(self, seq: Sequence[NoteInfo], i: int) -> str:
        info = seq[i]
        token: Dict[str, Any] = {}
        if self.cfg.pitch:
            token["pitch"] = info.pitch if self.cfg.octave else info.pitch_class
        if self.cfg.midi_number:
            token["midi_number"] = int(info.midi)
        if self.cfg.duration:
            token["duration"] = self._stable_float(info.duration)
        if self.cfg.length:
            token["length"] = int(info.length)
        if self.cfg.offset:
            token["offset"] = self._stable_float(info.offset)
        if self.cfg.bioi_ratio:
            ratio_symbol = bioi_ratio_symbol_at(seq, i)
            if ratio_symbol is not None:
                token["bioi_ratio"] = ratio_symbol
        return json.dumps(token, **JSON_DUMPS_KW)

    def _interval_token(self, semitones: int) -> str:
        return json.dumps({"interval": int(semitones)}, **JSON_DUMPS_KW)

    def _bioi_ratio_token(self, ratio_symbol: str) -> str:
        return json.dumps({"bioi_ratio": str(ratio_symbol)}, **JSON_DUMPS_KW)

    def _derived_symbol_from_objects(self, objs: Sequence[str], order_k: int) -> Optional[Symbol]:
        if not objs:
            return None
        if self.cfg.interval and not self.cfg.bioi_ratio:
            return objs[-1]
        if self.cfg.bioi_ratio and not self.cfg.interval:
            return objs[-1]
        if not (self.cfg.interval and self.cfg.bioi_ratio):
            return objs[-1]

        note_count = self.window_size_for_order(int(order_k))
        interval_count = max(int(note_count) - 1, 0) if self.cfg.interval else 0
        ratio_count = max(int(note_count) - 2, 0) if self.cfg.bioi_ratio else 0
        expected = interval_count + ratio_count
        if expected <= 0 or len(objs) < expected:
            return objs[-1]
        try:
            merged: Dict[str, Any] = {}
            interval_obj = json.loads(objs[interval_count - 1]) if interval_count > 0 else None
            ratio_obj = json.loads(objs[-1]) if ratio_count > 0 else None
            if isinstance(interval_obj, dict):
                merged.update(interval_obj)
            if isinstance(ratio_obj, dict):
                merged.update(ratio_obj)
            if merged:
                return json.dumps(merged, **JSON_DUMPS_KW)
        except (json.JSONDecodeError, TypeError, ValueError):
            return objs[-1]
        return objs[-1]

    # -------------------------
    # Pitch/interval helpers (faithful)
    # -------------------------

    def _pitch_to_str(self, pitch_obj: ms.pitch.Pitch, octave: bool) -> str:
        if self.cfg.enharmony:
            pitch_obj = ms.pitch.Pitch(pitch_obj.midi)
        return str(pitch_obj) if octave else pitch_obj.name

    def pitch_str_transpose(self, pitch_str: str, semitones: int, octave: bool = True) -> str:
        """Utility that matches the old helper; used later for augmentation/sampling."""

        p = ms.pitch.Pitch(pitch_str)
        p.midi += int(semitones)
        return self._pitch_to_str(p, octave)

    def _chromatic_interval_semitones(self, prev_pitch_str: str, next_pitch_str: str) -> int:
        p1 = ms.pitch.Pitch(prev_pitch_str)
        p2 = ms.pitch.Pitch(next_pitch_str)
        return int(ms.interval.Interval(p1, p2).chromatic.semitones)
