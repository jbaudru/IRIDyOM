"""midi_parse.py

Music21-based monophonic MIDI parser with mido-based quantized note length extraction.

This module provides faithful monophonic parsing consistent with the original 
`SimpleMidiFolderGraphs` behavior:

Core parsing features:
- Uses `music21.converter.parse(..., quarterLengthDivisors=(16,))` for note extraction
- Selects a single part with notes; optionally discards if multiple parts exist
- Optionally uses first part only when multiple parts are available
- Flattens notes and sorts by offset
- Detects and rejects polyphony/overlapping notes with RuntimeError

Quantized note length extraction:
- Primary path: independent mido-based parsing to extract raw MIDI note timings
- Fallback path: if mido is unavailable/fails, quantize from music21 inter-onset durations
  using the same IDyOM quantization standard (24 ticks per beat)

NoteInfo fields emitted:
- timestamp: note.offset (absolute time in beats)
- duration: (next_start - start) for non-last notes; else note.duration.quarterLength
- offset: start % beat_duration (position within beat cycle)
- pitch: normalized pitch with octave (enharmony-aware)
- pitch_class: normalized pitch without octave (enharmony-aware)
- midi: MIDI note number (normalized if enharmony=True)
- length: IDyOM-style length viewpoint token (quantized ticks). This uses the
  inter-onset duration for all notes except the final note, for which we emit 0
  (IDyOM drops the terminal duration via duration[:-1]).

Pitch normalization:
 - If enharmony=True, normalize using Pitch(midi) before extracting pitch/midi values
- Ensures consistent enharmonic spelling across different input formats
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import List, Optional

import music21 as ms

from graph_types import MidiParser, NoteInfo


# -------------------------
# Internal mido-based note extraction (independent from IDyOM)
# -------------------------

@dataclass
class _MidiNote:
    """Simple note representation from mido parsing."""
    velocity: int
    pitch: int
    start: int  # absolute time in ticks
    end: int    # absolute time in ticks


def _parse_mido_notes_impl(path: str) -> tuple[List[_MidiNote], int]:
    """Parse a MIDI file using mido to extract notes and ticks_per_beat.
    
    This is an independent reimplementation based on IDyOM's myMidi.getNotesFromMidi,
    but self-contained to avoid external dependencies.
    
    Returns:
        (notes, ticks_per_beat) where notes is a list of _MidiNote objects
        and ticks_per_beat is the MIDI quantization unit.
    """
    try:
        import mido
    except ImportError:
        raise ImportError("mido is required for MIDI parsing. Install it with: pip install mido")
    
    mido_obj = mido.MidiFile(path)
    
    # Convert delta times to cumulative times
    for track in mido_obj.tracks:
        tick = int(0)
        for event in track:
            event.time += tick
            tick = event.time
    
    NOTES = []
    
    # Extract notes from all tracks
    for track_idx, track in enumerate(mido_obj.tracks):
        # Track note-on events by (channel, note) to match with note-offs
        last_note_on = collections.defaultdict(list)
        notes = []
        
        for event in track:
            if event.type == 'note_on' and event.velocity > 0:
                # Store this as the last note-on location
                note_on_index = (event.channel, event.note)
                last_note_on[note_on_index].append((event.time, event.velocity))
            
            # Note offs can also be note on events with 0 velocity
            elif event.type == 'note_off' or (event.type == 'note_on' and event.velocity == 0):
                # Check that a note-on exists (ignore spurious note-offs)
                key = (event.channel, event.note)
                if key in last_note_on:
                    end_tick = event.time
                    open_notes = last_note_on[key]
                    
                    # Separate notes that are ending and notes that continue
                    notes_to_close = [
                        (start_tick, velocity)
                        for start_tick, velocity in open_notes
                        if start_tick != end_tick
                    ]
                    notes_to_keep = [
                        (start_tick, velocity)
                        for start_tick, velocity in open_notes
                        if start_tick == end_tick
                    ]
                    
                    # Create note objects for closing notes
                    for start_tick, velocity in notes_to_close:
                        note = _MidiNote(
                            velocity=velocity,
                            pitch=event.note,
                            start=start_tick,
                            end=end_tick
                        )
                        notes.append(note)
                    
                    if len(notes_to_close) > 0 and len(notes_to_keep) > 0:
                        # Note-on on the same tick but we already closed previous notes
                        last_note_on[key] = notes_to_keep
                    else:
                        # Remove the last note on for this instrument
                        del last_note_on[key]
        
        NOTES.append(notes)
    
    # Find the track with notes (ensure monophonic)
    isPoly = False
    for N in NOTES:
        if len(N) > 0:
            if isPoly:
                raise RuntimeError(f"This file is polyphonic, cannot handle: {path}")
            notes = N
            isPoly = True
    
    if not isPoly:
        raise RuntimeError(f"This file is empty, cannot handle: {path}")
    
    return notes, mido_obj.ticks_per_beat


def _compute_quantized_lengths(notes: List[_MidiNote], ticks_per_beat: int, quantization: int = 24) -> dict[int, int]:
    """Compute quantized note lengths from mido note objects.
    
    For each note except the last: duration until next note (in quantized ticks)
    For the last note: actual note duration (in quantized ticks)
    
    Returns:
        Dict mapping note index to quantized duration in ticks.
    """
    if not notes:
        return {}
    
    result = {}
    
    # For each note except the last: duration until next note
    for i in range(len(notes) - 1):
        duration_until_next = notes[i + 1].start - notes[i].start
        quantized_dur = int(round(quantization * duration_until_next / ticks_per_beat))
        result[i] = int(quantized_dur)
    
    # For the last note: actual note duration
    last_note = notes[-1]
    last_duration = last_note.end - last_note.start
    quantized_last = int(round(quantization * last_duration / ticks_per_beat))
    result[len(notes) - 1] = int(quantized_last)
    
    return result


def _midi_to_pitch_name(midi_note: int) -> str:
    """Convert MIDI note number to pitch+octave (sharp-based spelling)."""
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    pc = int(midi_note) % 12
    octave = int(midi_note) // 12 - 1
    return f"{names[pc]}{octave}"


def _midi_to_pitch_class(midi_note: int) -> str:
    """Convert MIDI note number to pitch class (sharp-based spelling)."""
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return names[int(midi_note) % 12]


class Music21MidiParser(MidiParser):
    """Faithful monophonic parser based on music21."""

    def __init__(
        self,
        *,
        beat_duration: float = 1.0,
        enharmony: bool = True,
        discard_if_multiple_parts: bool = False,
        use_first_part_only: bool = True,
        quarter_length_divisors: tuple[int, ...] = (16,),
    ):
        self.beat_duration = float(beat_duration)
        if self.beat_duration <= 0:
            raise ValueError("beat_duration must be > 0")

        self.enharmony = bool(enharmony)
        self.discard_if_multiple_parts = bool(discard_if_multiple_parts)
        self.use_first_part_only = bool(use_first_part_only)
        self.quarter_length_divisors = tuple(int(x) for x in quarter_length_divisors)

    # -------------------------
    # Public API
    # -------------------------

    def _parse_mido_notes(self, path: str) -> dict:
        """Parse a MIDI file using mido to extract raw note lengths (durations in quantized ticks).
        
        Returns a dict mapping note index (in the order they appear) to their quantized tick duration,
        consistent with IDyOM's standard quantization.
        
        This is now self-contained and does not depend on IDyOM.
        """
        try:
            mido_notes, ticks_per_beat = _parse_mido_notes_impl(path)
            quantization = 24  # IDyOM standard
            return _compute_quantized_lengths(mido_notes, ticks_per_beat, quantization)
        except Exception:
            # If parsing fails (e.g., mido unavailable), fall back to music21-based
            # quantization in _parse_monophonic_notes.
            return {}

    def parse_file(self, path: str) -> List[NoteInfo]:
        """Parse a MIDI/MusicXML file into a monophonic sequence of NoteInfo."""

        try:
            piece = ms.converter.parse(path, quarterLengthDivisors=self.quarter_length_divisors)
        except Exception as e:
            raise RuntimeError(f"Could not parse file: {path}\n{e}")

        # Get mido-based raw note lengths
        mido_data = self._parse_mido_notes(path)

        part = self._choose_part_stream(piece)
        if part is None:
            raise RuntimeError(f"No valid monophonic part found (or discarded multipart): {path}")

        # Parse monophonic notes using music21
        monophonic_notes = self._parse_monophonic_notes(part, mido_data=mido_data)

        return monophonic_notes

    # -------------------------
    # Multi-part selection (faithful to original behavior)
    # -------------------------

    def _choose_part_stream(self, piece) -> Optional[ms.stream.Part]:
        parts_with_notes: List[ms.stream.Part] = []
        for part in getattr(piece, "parts", []):
            if len(list(part.recurse().notes)) > 0:
                parts_with_notes.append(part)

        if len(parts_with_notes) == 0:
            return None

        if len(parts_with_notes) > 1:
            if self.discard_if_multiple_parts:
                return None
            if self.use_first_part_only:
                return parts_with_notes[0]
            # Original code returns parts_with_notes[0] in all non-discard cases
            return parts_with_notes[0]

        return parts_with_notes[0]

    # -------------------------
    # Pitch normalization helpers (faithful)
    # -------------------------

    def _pitch_to_str(self, pitch_obj: ms.pitch.Pitch, octave: bool) -> str:
        if self.enharmony:
            pitch_obj = ms.pitch.Pitch(pitch_obj.midi)
        return str(pitch_obj) if octave else pitch_obj.name

    def _parse_pitch_note(self, note: ms.note.Note, octave: bool) -> str:
        return self._pitch_to_str(note.pitch, octave)

    # -------------------------
    # Monophonic parsing (faithful)
    # -------------------------

    def _parse_monophonic_notes(self, part_stream, mido_data: dict = None) -> List[NoteInfo]:
        if mido_data is None:
            mido_data = {}
            
        notes = list(part_stream.flatten().notes)
        if len(notes) == 0:
            return []

        # Faithful: sort by note offset
        notes.sort(key=lambda n: float(n.offset))

        parsed: List[NoteInfo] = []
        for i, n in enumerate(notes):
            start = float(n.offset)
            end = start + float(n.duration.quarterLength)

            if i < len(notes) - 1:
                next_start = float(notes[i + 1].offset)
                if next_start < end:
                    # Faithful error message
                    raise RuntimeError("Polyphonic/overlapping notes detected")
                dur_until_next = next_start - start
            else:
                dur_until_next = float(n.duration.quarterLength)

            # Midi number: mirror IDyOM behaviour. If enharmony normalization is
            # enabled, normalize via Pitch(midi) before taking the midi value.
            midi_val = int(ms.pitch.Pitch(n.pitch.midi).midi) if self.enharmony else int(n.pitch.midi)
            
            # IDyOM-compatible "length" viewpoint:
            # - use inter-onset quantized durations for non-final notes
            # - prefer mido-extracted quantization when available
            # - otherwise fall back to music21 inter-onset durations
            # - drop the terminal duration (IDyOM uses duration[:-1]), represented here as 0
            if i < len(notes) - 1:
                if i in mido_data:
                    length_val = int(mido_data.get(i, 0))
                else:
                    length_val = int(round(24.0 * float(dur_until_next)))
                    if length_val < 0:
                        length_val = 0
            else:
                length_val = 0
            
            info = NoteInfo(
                timestamp=start,
                duration=float(dur_until_next),
                offset=float(start % self.beat_duration),
                pitch=self._parse_pitch_note(n, octave=True),
                pitch_class=self._parse_pitch_note(n, octave=False),
                midi=midi_val,
                length=length_val,
            )
            parsed.append(info)

        return parsed


class MidoMidiParser(MidiParser):
    """IDyOM-compatible parser based purely on mido track-note extraction.

    This parser avoids music21 parsing overhead and follows the same core note
    extraction logic as IDyOM's myMidi parser:
    - parse note-on/off events from mido
    - keep a single note-bearing track (reject multi-track polyphonic files)
    - emit inter-onset duration for non-final notes
    - emit length tokens quantized at 24 ticks/beat (terminal note length=0)
    """

    def __init__(self, *, beat_duration: float = 1.0, enharmony: bool = True):
        self.beat_duration = float(beat_duration)
        if self.beat_duration <= 0:
            raise ValueError("beat_duration must be > 0")
        self.enharmony = bool(enharmony)

    def parse_file(self, path: str) -> List[NoteInfo]:
        notes, ticks_per_beat = _parse_mido_notes_impl(path)
        if not notes:
            return []

        q_lengths = _compute_quantized_lengths(notes, ticks_per_beat, quantization=24)
        parsed: List[NoteInfo] = []
        for i, n in enumerate(notes):
            start_beats = float(n.start) / float(ticks_per_beat)
            end_beats = float(n.end) / float(ticks_per_beat)

            if i < len(notes) - 1:
                next_start_beats = float(notes[i + 1].start) / float(ticks_per_beat)
                dur_until_next = next_start_beats - start_beats
            else:
                dur_until_next = end_beats - start_beats

            midi_val = int(n.pitch)
            if self.enharmony:
                # Preserve behavior parity with the music21 parser normalization.
                midi_val = int(ms.pitch.Pitch(midi_val).midi)

            parsed.append(
                NoteInfo(
                    timestamp=float(start_beats),
                    duration=float(dur_until_next),
                    offset=float(start_beats % self.beat_duration),
                    pitch=_midi_to_pitch_name(midi_val),
                    pitch_class=_midi_to_pitch_class(midi_val),
                    midi=midi_val,
                    length=int(q_lengths.get(i, 0)) if i < len(notes) - 1 else 0,
                )
            )
        return parsed


def create_midi_parser(
    mode: str,
    *,
    beat_duration: float = 1.0,
    enharmony: bool = True,
    discard_if_multiple_parts: bool = False,
    use_first_part_only: bool = True,
    quarter_length_divisors: tuple[int, ...] = (16,),
) -> MidiParser:
    """Factory for parser backend selection.

    mode:
    - "music21": existing Music21MidiParser
    - "mido": new MidoMidiParser
    """
    m = str(mode).strip().lower()
    if m == "music21":
        return Music21MidiParser(
            beat_duration=beat_duration,
            enharmony=enharmony,
            discard_if_multiple_parts=discard_if_multiple_parts,
            use_first_part_only=use_first_part_only,
            quarter_length_divisors=quarter_length_divisors,
        )
    if m == "mido":
        return MidoMidiParser(
            beat_duration=beat_duration,
            enharmony=enharmony,
        )
    raise ValueError(f"Unknown parser mode: {mode!r}. Expected one of: music21, mido")
