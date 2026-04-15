"""Target-viewpoint projection logic for GraphIDYOM models.

This module isolates target-viewpoint concerns from `GraphIDYOMModel`:
- target viewpoint validation
- target alphabet construction
- symbol-level projection
- distribution-level projection

`GraphIDYOMModel` can inherit `TargetProjectionMixin` to keep the public/private
method surface stable while moving heavy logic out of the core model class.
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import music21 as ms

from graph_types import Dist, NoteInfo, Symbol
from token_codec import bioi_ratio_symbol_at


class TargetProjectionMixin:
    """Mixin implementing target-viewpoint projection behavior.

    Required attributes on the host class:
    - `target_viewpoint`
    - `target_alphabet`
    - `alphabet`
    - `codec`
    - `parser`
    - `graph_builder`
    - `graph_build_config`
    - `_projection_cache`
    """

    # -------------------------
    # Validation / alphabet
    # -------------------------

    def _validate_target_viewpoint(self) -> None:
        """Validate that target viewpoint is compatible with selected viewpoints."""
        if self.target_viewpoint is None:
            return
        supported = {
            "pitchOctave",
            "pitchClass",
            "midi_number",
            "interval",
            "length",
            "duration",
            "offset",
            "bioi_ratio",
        }
        if self.target_viewpoint not in supported:
            raise ValueError(
                f"Unsupported target_viewpoint: {self.target_viewpoint}. "
                f"Supported: {sorted(supported)}"
            )

        cfg = getattr(self.codec, "cfg", None)
        if cfg is None:
            raise ValueError("Codec must have a 'cfg' attribute for target viewpoint validation")

        # IDyOM-style typeset compatibility: target basic viewpoint must be
        # predictable by the source viewpoint's components.
        if self.target_viewpoint in {"pitchOctave", "pitchClass", "midi_number", "interval"}:
            has_pitch_viewpoint = bool(cfg.pitch or cfg.midi_number or cfg.interval)
            if not has_pitch_viewpoint:
                raise ValueError(
                    f"target_viewpoint={self.target_viewpoint!r} requires at least one pitch-related "
                    "source component (pitch, midi_number, or interval)"
                )
        elif self.target_viewpoint == "length":
            if not (bool(cfg.length) or bool(getattr(cfg, "bioi_ratio", False))):
                raise ValueError(
                    "target_viewpoint='length' requires source component 'length' or 'bioi_ratio'"
                )
        elif self.target_viewpoint == "duration":
            if not bool(cfg.duration):
                raise ValueError(
                    "target_viewpoint='duration' requires source component 'duration'"
                )
        elif self.target_viewpoint == "offset":
            if not bool(cfg.offset):
                raise ValueError(
                    "target_viewpoint='offset' requires source component 'offset'"
                )
        elif self.target_viewpoint == "bioi_ratio":
            if not bool(getattr(cfg, "bioi_ratio", False)):
                raise ValueError(
                    "target_viewpoint='bioi_ratio' requires source component 'bioi_ratio'"
                )

    def _build_target_alphabet(self, source_alphabet: Tuple[Symbol, ...]) -> Tuple[Symbol, ...]:
        """Legacy helper: infer target alphabet directly from source alphabet."""
        if self.target_viewpoint != "pitchOctave":
            return source_alphabet

        target_pitches = set()
        for sym in source_alphabet:
            try:
                obj = json.loads(sym)
                if isinstance(obj, dict) and "pitch" in obj:
                    pitch = obj["pitch"]
                    if any(c.isdigit() or c == "-" for c in pitch):
                        target_pitches.add(pitch)
            except (json.JSONDecodeError, TypeError):
                pass
        return tuple(sorted(target_pitches))

    def _build_target_alphabet_from_folder(self, folder: str) -> Tuple[Symbol, ...]:
        """Build target viewpoint alphabet by parsing dataset files."""
        if self.target_viewpoint is None:
            return self.alphabet

        extensions = (".mid", ".midi", ".xml", ".mxl", ".musicxml")
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in extensions
        ]
        return self._build_target_alphabet_from_files(files)

    def _build_target_alphabet_from_files(self, files: Sequence[os.PathLike | str]) -> Tuple[Symbol, ...]:
        """Build target viewpoint alphabet by parsing an explicit set of files."""
        if self.target_viewpoint is None:
            return self.alphabet

        target_symbols = set()
        use_aug = bool(getattr(self.graph_build_config, "augment", False))

        for file_path in files:
            try:
                notes = self.parser.parse_file(str(Path(file_path)))
                sequences = [notes]
                if use_aug and hasattr(self.graph_builder, "_all_augmented_sequences"):
                    sequences = self.graph_builder._all_augmented_sequences(notes)  # type: ignore[attr-defined]
                ndigits = getattr(getattr(self.codec, "cfg", None), "token_round_ndigits", None)
                for seq in sequences:
                    for idx, note in enumerate(seq):
                        if self.target_viewpoint == "pitchOctave":
                            target_symbols.add(note.pitch)
                        elif self.target_viewpoint == "pitchClass":
                            target_symbols.add(note.pitch_class)
                        elif self.target_viewpoint == "midi_number":
                            target_symbols.add(str(int(note.midi)))
                        elif self.target_viewpoint == "interval":
                            if idx > 0:
                                target_symbols.add(str(int(note.midi - seq[idx - 1].midi)))
                        elif self.target_viewpoint == "length":
                            target_symbols.add(str(int(note.length)))
                        elif self.target_viewpoint == "duration":
                            dur = float(note.duration)
                            if ndigits is not None:
                                dur = float(round(dur, int(ndigits)))
                            target_symbols.add(str(dur))
                        elif self.target_viewpoint == "offset":
                            off = float(note.offset)
                            if ndigits is not None:
                                off = float(round(off, int(ndigits)))
                            target_symbols.add(str(off))
                        elif self.target_viewpoint == "bioi_ratio":
                            ratio_symbol = bioi_ratio_symbol_at(seq, idx)
                            if ratio_symbol is not None:
                                target_symbols.add(str(ratio_symbol))
            except Exception:
                # Skip files that cannot be parsed by this backend.
                pass

        return tuple(sorted(target_symbols))

    # -------------------------
    # Symbol / distribution projection
    # -------------------------

    def _project_symbol_to_target(
        self,
        symbol: Symbol,
        context: Optional[Sequence[NoteInfo]] = None,
        current_note: Optional[NoteInfo] = None,
    ) -> List[Symbol]:
        """Project a single source symbol to target viewpoint symbol(s)."""
        if self.target_viewpoint is None or symbol == "NO_EVENT":
            return [symbol]

        if self.target_viewpoint == "pitchOctave":
            return self._project_symbol_to_pitch_octave(symbol, context, current_note)
        if self.target_viewpoint == "pitchClass":
            return self._project_symbol_to_pitch_class(symbol, context, current_note)
        if self.target_viewpoint == "midi_number":
            return self._project_symbol_to_midi_number(symbol, context, current_note)
        if self.target_viewpoint == "interval":
            return self._project_symbol_to_interval(symbol, context, current_note)
        if self.target_viewpoint == "length":
            return self._project_symbol_to_length(symbol, context, current_note)
        if self.target_viewpoint in {"duration", "offset", "bioi_ratio"}:
            return self._project_symbol_to_scalar_field(symbol, self.target_viewpoint, current_note)
        return [symbol]

    def _project_dist_to_target(self, dist: Dist, context: Optional[Sequence[NoteInfo]] = None) -> Dist:
        """Project a distribution from source symbols to target-viewpoint symbols."""
        if self.target_viewpoint is None or not dist:
            return dist

        if self.target_viewpoint == "pitchOctave":
            return self._project_to_pitch_octave(dist, context)
        if self.target_viewpoint == "pitchClass":
            return self._project_dist_via_symbol_projection(
                dist,
                context=context,
                projector=self._project_symbol_to_pitch_class,
                uniform_on_empty=True,
            )
        if self.target_viewpoint == "midi_number":
            return self._project_dist_via_symbol_projection(
                dist,
                context=context,
                projector=self._project_symbol_to_midi_number,
                uniform_on_empty=True,
            )
        if self.target_viewpoint == "interval":
            return self._project_dist_via_symbol_projection(
                dist,
                context=context,
                projector=self._project_symbol_to_interval,
                uniform_on_empty=True,
            )
        if self.target_viewpoint == "length":
            return self._project_dist_via_symbol_projection(
                dist,
                context=context,
                projector=self._project_symbol_to_length,
                uniform_on_empty=True,
            )
        if self.target_viewpoint in {"duration", "offset", "bioi_ratio"}:
            return self._project_dist_via_symbol_projection(
                dist,
                context=context,
                projector=lambda s, _, c: self._project_symbol_to_scalar_field(
                    s, self.target_viewpoint, c
                ),
                uniform_on_empty=False,
            )
        return dist

    def _project_dist_via_symbol_projection(
        self,
        dist: Dist,
        *,
        context: Optional[Sequence[NoteInfo]],
        projector,
        uniform_on_empty: bool,
    ) -> Dist:
        out: Dict[str, float] = {}
        for symbol, prob in dist.items():
            p = float(prob)
            if p <= 0.0:
                continue
            mapped = projector(symbol, context, None)
            if not mapped:
                continue
            share = p / float(len(mapped))
            for m in mapped:
                out[str(m)] = out.get(str(m), 0.0) + share
        total = float(sum(out.values()))
        if total > 0.0:
            return {k: (v / total) for k, v in out.items()}
        if uniform_on_empty and self.target_alphabet:
            u = 1.0 / float(len(self.target_alphabet))
            return {str(s): u for s in self.target_alphabet}
        return {}

    def _target_pitch_candidates_for_midi(self, midi_val: int) -> List[Symbol]:
        """Return target-alphabet symbols enharmonically equivalent to `midi_val`."""
        if not self.target_alphabet:
            return []
        try:
            midi_i = max(0, min(127, int(midi_val)))
        except Exception:
            return []

        out: List[Symbol] = []
        for target_pitch in self.target_alphabet:
            try:
                if int(ms.pitch.Pitch(str(target_pitch)).midi) == midi_i:
                    out.append(str(target_pitch))
            except Exception:
                continue
        return out

    def _target_pitch_candidates_for_name(self, pitch_name: str) -> List[Symbol]:
        """Return target-alphabet symbols enharmonically equivalent to `pitch_name`."""
        if not self.target_alphabet:
            return []
        p = str(pitch_name)
        if p in self.target_alphabet:
            return [p]
        try:
            midi_i = int(ms.pitch.Pitch(p).midi)
        except Exception:
            return []
        return self._target_pitch_candidates_for_midi(midi_i)

    def _project_symbol_to_pitch_octave(
        self,
        symbol: Symbol,
        context: Optional[Sequence[NoteInfo]] = None,
        current_note: Optional[NoteInfo] = None,
    ) -> List[Symbol]:
        if not self.target_alphabet:
            return [symbol]
        try:
            obj = json.loads(symbol)
            if not isinstance(obj, dict):
                return [symbol]

            obj_no_duration = {k: v for k, v in obj.items() if k not in ["duration", "offset", "length"]}
            has_pitch = "pitch" in obj_no_duration
            pitch_with_octave = False
            pitch_class_only = False
            if has_pitch:
                pitch_str = obj_no_duration["pitch"]
                pitch_with_octave = any(c.isdigit() for c in pitch_str)
                pitch_class_only = not pitch_with_octave

            has_interval = "interval" in obj_no_duration
            has_midi_number = "midi_number" in obj_no_duration

            if pitch_with_octave:
                return self._target_pitch_candidates_for_name(str(pitch_str))

            if has_midi_number:
                try:
                    midi_val = int(obj_no_duration["midi_number"])
                    return self._target_pitch_candidates_for_midi(midi_val)
                except Exception:
                    return []
                return []

            if pitch_class_only and has_interval:
                if context and len(context) > 0:
                    last_pitch = context[-1].pitch
                    interval_semitones = obj_no_duration["interval"]
                    try:
                        last_note = ms.pitch.Pitch(last_pitch)
                        new_midi = last_note.midi + interval_semitones
                        return self._target_pitch_candidates_for_midi(int(new_midi))
                    except Exception:
                        pass
                return []

            if has_interval:
                if context and len(context) > 0:
                    last_pitch = context[-1].pitch
                    interval_semitones = obj_no_duration["interval"]
                    try:
                        last_note = ms.pitch.Pitch(last_pitch)
                        new_midi = last_note.midi + interval_semitones
                        return self._target_pitch_candidates_for_midi(int(new_midi))
                    except Exception:
                        pass
                return []

            if pitch_class_only:
                pitch_class = obj_no_duration["pitch"]
                matching = [p for p in self.target_alphabet if p.startswith(pitch_class)]
                return matching if matching else []

            if current_note is not None:
                actual_pitch = str(current_note.pitch)
                cands = self._target_pitch_candidates_for_name(actual_pitch)
                if cands:
                    return cands
                try:
                    return self._target_pitch_candidates_for_midi(int(current_note.midi))
                except Exception:
                    pass
            return []
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _project_symbol_to_pitch_class(
        self,
        symbol: Symbol,
        context: Optional[Sequence[NoteInfo]] = None,
        current_note: Optional[NoteInfo] = None,
    ) -> List[Symbol]:
        allowed = set(self.target_alphabet) if self.target_alphabet else None
        try:
            obj = json.loads(symbol)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            if "pitch" in obj:
                p = str(obj["pitch"])
                try:
                    pitch_class = ms.pitch.Pitch(p).name
                except Exception:
                    pitch_class = p
                if allowed is None or pitch_class in allowed:
                    return [pitch_class]
                return []
            if "midi_number" in obj:
                try:
                    pc = ms.pitch.Pitch(midi=int(obj["midi_number"])).name
                    if allowed is None or pc in allowed:
                        return [pc]
                except Exception:
                    pass
                return []
            if "interval" in obj and context and len(context) > 0:
                try:
                    base = ms.pitch.Pitch(context[-1].pitch)
                    p = ms.pitch.Pitch(midi=int(base.midi + int(obj["interval"]))).name
                    if allowed is None or p in allowed:
                        return [p]
                except Exception:
                    pass
                return []
        if current_note is not None:
            pc = current_note.pitch_class
            if allowed is None or pc in allowed:
                return [pc]
        return []

    def _project_symbol_to_midi_number(
        self,
        symbol: Symbol,
        context: Optional[Sequence[NoteInfo]] = None,
        current_note: Optional[NoteInfo] = None,
    ) -> List[Symbol]:
        allowed = set(str(s) for s in self.target_alphabet) if self.target_alphabet else None
        try:
            obj = json.loads(symbol)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            if "midi_number" in obj:
                s = str(int(obj["midi_number"]))
                if allowed is None or s in allowed:
                    return [s]
                return []
            if "pitch" in obj:
                try:
                    midi = str(int(ms.pitch.Pitch(str(obj["pitch"])).midi))
                    if allowed is None or midi in allowed:
                        return [midi]
                except Exception:
                    pass
                return []
            if "interval" in obj and context and len(context) > 0:
                try:
                    base = ms.pitch.Pitch(context[-1].pitch).midi
                    midi = str(int(base + int(obj["interval"])))
                    if allowed is None or midi in allowed:
                        return [midi]
                except Exception:
                    pass
                return []
        if current_note is not None:
            s = str(int(current_note.midi))
            if allowed is None or s in allowed:
                return [s]
        return []

    def _project_symbol_to_interval(
        self,
        symbol: Symbol,
        context: Optional[Sequence[NoteInfo]] = None,
        current_note: Optional[NoteInfo] = None,
    ) -> List[Symbol]:
        allowed = set(str(s) for s in self.target_alphabet) if self.target_alphabet else None

        def _accept(v: int) -> List[Symbol]:
            s = str(int(v))
            if allowed is None or s in allowed:
                return [s]
            return []

        def _interval_from_midi(midi_value: int) -> List[Symbol]:
            if not context or len(context) == 0:
                return []
            base = int(context[-1].midi)
            return _accept(int(midi_value) - base)

        try:
            obj = json.loads(symbol)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            if "interval" in obj:
                try:
                    return _accept(int(obj["interval"]))
                except Exception:
                    return []
            if "midi_number" in obj:
                try:
                    return _interval_from_midi(int(obj["midi_number"]))
                except Exception:
                    return []
            if "pitch" in obj:
                pitch_str = str(obj["pitch"])
                has_octave = any(ch.isdigit() for ch in pitch_str)
                if has_octave:
                    try:
                        return _interval_from_midi(int(ms.pitch.Pitch(pitch_str).midi))
                    except Exception:
                        return []
                if current_note is not None and context and len(context) > 0:
                    return _accept(int(current_note.midi) - int(context[-1].midi))
                return []
        if current_note is not None and context and len(context) > 0:
            return _accept(int(current_note.midi) - int(context[-1].midi))
        return []

    def _project_symbol_to_scalar_field(
        self,
        symbol: Symbol,
        field: str,
        current_note: Optional[NoteInfo] = None,
    ) -> List[Symbol]:
        allowed = set(str(s) for s in self.target_alphabet) if self.target_alphabet else None
        ndigits = getattr(getattr(self.codec, "cfg", None), "token_round_ndigits", None)

        def _fmt_scalar(x) -> str:
            if field in {"duration", "offset"}:
                v = float(x)
                if ndigits is not None:
                    v = float(round(v, int(ndigits)))
                return str(v)
            if field == "length":
                return str(int(x))
            return str(x)

        try:
            obj = json.loads(symbol)
        except Exception:
            obj = None
        if isinstance(obj, dict) and field in obj:
            s = _fmt_scalar(obj[field])
            if allowed is None or s in allowed:
                return [s]
            return []
        if current_note is not None:
            val = getattr(current_note, field, None)
            if val is not None:
                s = _fmt_scalar(val)
                if allowed is None or s in allowed:
                    return [s]
        return []

    def _project_symbol_to_length(
        self,
        symbol: Symbol,
        context: Optional[Sequence[NoteInfo]] = None,
        current_note: Optional[NoteInfo] = None,
    ) -> List[Symbol]:
        allowed = set(str(s) for s in self.target_alphabet) if self.target_alphabet else None

        def _accept(v: int) -> List[Symbol]:
            s = str(int(v))
            if allowed is None or s in allowed:
                return [s]
            return []

        try:
            obj = json.loads(symbol)
        except Exception:
            obj = None

        if isinstance(obj, dict):
            if "length" in obj:
                try:
                    return _accept(int(obj["length"]))
                except Exception:
                    return []
            if "bioi_ratio" in obj:
                if not context or len(context) == 0:
                    if self.target_alphabet:
                        return [str(s) for s in self.target_alphabet]
                    return []
                try:
                    prev_length = int(getattr(context[-1], "length", 0))
                    if prev_length == 0:
                        if self.target_alphabet:
                            return [str(s) for s in self.target_alphabet]
                        return []
                    ratio = Fraction(str(obj["bioi_ratio"]))
                    projected = Fraction(prev_length) * ratio
                    if projected.denominator == 1:
                        return _accept(int(projected.numerator))
                except Exception:
                    return []
                return []

        if current_note is not None:
            try:
                return _accept(int(getattr(current_note, "length")))
            except Exception:
                return []
        return []

    def _project_to_pitch_octave(self, dist: Dist, context: Optional[Sequence[NoteInfo]] = None) -> Dist:
        """Project distribution to `pitchOctave` with caching for stable symbols."""
        if not self.target_alphabet:
            return dist
        cfg = getattr(self.codec, "cfg", None)
        if cfg is None:
            return dist

        dist_no_duration: Dict[str, float] = {}
        for symbol, prob in dist.items():
            try:
                obj = json.loads(symbol)
                if isinstance(obj, dict) and "duration" in obj:
                    obj_no_duration = {
                        k: v for k, v in obj.items() if k not in {"duration", "offset", "length"}
                    }
                    symbol_no_duration = json.dumps(obj_no_duration, sort_keys=True)
                    dist_no_duration[symbol_no_duration] = dist_no_duration.get(symbol_no_duration, 0.0) + prob
                else:
                    dist_no_duration[symbol] = dist_no_duration.get(symbol, 0.0) + prob
            except (json.JSONDecodeError, TypeError):
                dist_no_duration[symbol] = dist_no_duration.get(symbol, 0.0) + prob

        has_any_pitch_info = False
        for symbol in dist_no_duration.keys():
            try:
                obj = json.loads(symbol)
                if isinstance(obj, dict) and ("pitch" in obj or "interval" in obj or "midi_number" in obj):
                    has_any_pitch_info = True
                    break
            except (json.JSONDecodeError, TypeError):
                pass

        if not has_any_pitch_info:
            uniform_prob = 1.0 / len(self.target_alphabet)
            return {pitch: uniform_prob for pitch in self.target_alphabet}

        projected: Dict[str, float] = {}
        target_pitches = set(self.target_alphabet)

        for symbol, prob in dist_no_duration.items():
            if symbol in self._projection_cache:
                cached_projection = self._projection_cache[symbol]
                for target_pitch, target_prob in cached_projection.items():
                    projected[target_pitch] = projected.get(target_pitch, 0.0) + (prob * target_prob)
                continue

            try:
                obj = json.loads(symbol)
                if not isinstance(obj, dict):
                    continue

                symbol_projection: Dict[str, float] = {}
                has_pitch = "pitch" in obj
                pitch_with_octave = False
                pitch_class_only = False
                if has_pitch:
                    pitch_str = obj["pitch"]
                    pitch_with_octave = any(c.isdigit() for c in pitch_str)
                    pitch_class_only = not pitch_with_octave
                has_interval = "interval" in obj
                has_midi_number = "midi_number" in obj

                if pitch_with_octave:
                    pitch_candidates = self._target_pitch_candidates_for_name(str(pitch_str))
                    if pitch_candidates:
                        each = 1.0 / float(len(pitch_candidates))
                        for cand in pitch_candidates:
                            symbol_projection[cand] = symbol_projection.get(cand, 0.0) + each
                elif has_midi_number:
                    try:
                        midi_val = int(obj["midi_number"])
                        pitch_candidates = self._target_pitch_candidates_for_midi(midi_val)
                        if pitch_candidates:
                            each = 1.0 / float(len(pitch_candidates))
                            for cand in pitch_candidates:
                                symbol_projection[cand] = symbol_projection.get(cand, 0.0) + each
                    except Exception:
                        pass
                elif has_interval and pitch_class_only and context is not None and len(context) > 0:
                    last_pitch = context[-1].pitch
                    interval = obj["interval"]
                    try:
                        last_note = ms.pitch.Pitch(last_pitch)
                        new_midi = last_note.midi + interval
                        pitch_candidates = self._target_pitch_candidates_for_midi(int(new_midi))
                        if pitch_candidates:
                            share = float(prob) / float(len(pitch_candidates))
                            for cand in pitch_candidates:
                                projected[cand] = projected.get(cand, 0.0) + share
                    except Exception:
                        pass
                    continue
                elif has_interval and context is not None and len(context) > 0:
                    last_pitch = context[-1].pitch
                    interval = obj["interval"]
                    try:
                        last_note = ms.pitch.Pitch(last_pitch)
                        new_midi = last_note.midi + interval
                        pitch_candidates = self._target_pitch_candidates_for_midi(int(new_midi))
                        if pitch_candidates:
                            share = float(prob) / float(len(pitch_candidates))
                            for cand in pitch_candidates:
                                projected[cand] = projected.get(cand, 0.0) + share
                    except Exception:
                        pass
                    continue
                elif pitch_class_only:
                    pitch_class = obj["pitch"]
                    matching_pitches = []
                    for target_pitch in target_pitches:
                        idx = 0
                        while idx < len(target_pitch) and (
                            target_pitch[idx].isalpha() or target_pitch[idx] in ["#", "b", "-"]
                        ):
                            idx += 1
                            if (
                                idx > 0
                                and target_pitch[idx - 1] == "-"
                                and idx < len(target_pitch)
                                and not target_pitch[idx].isdigit()
                            ):
                                idx -= 1
                                break
                        if target_pitch[:idx] == pitch_class:
                            matching_pitches.append(target_pitch)
                    if matching_pitches:
                        prob_per_octave = 1.0 / len(matching_pitches)
                        for pitch_octave in matching_pitches:
                            symbol_projection[pitch_octave] = prob_per_octave

                if symbol_projection and not has_interval:
                    self._projection_cache[symbol] = symbol_projection
                for target_pitch, target_prob in symbol_projection.items():
                    projected[target_pitch] = projected.get(target_pitch, 0.0) + (prob * target_prob)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

        total = sum(projected.values())
        if total > 0:
            projected = {k: v / total for k, v in projected.items()}
            return projected
        if self.target_alphabet:
            uniform_prob = 1.0 / len(self.target_alphabet)
            return {pitch: uniform_prob for pitch in self.target_alphabet}
        return projected
