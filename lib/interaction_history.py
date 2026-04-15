"""Helpers for interactive history input (e.g., user-provided MIDI numbers)."""

from __future__ import annotations

from typing import List, Sequence

import music21 as ms

from graph_types import NoteInfo


def midi_history_to_noteinfos(
    midi_history: Sequence[int],
    *,
    default_duration: float = 1.0,
    default_length: int = 24,
    beat_duration: float = 1.0,
    start_timestamp: float = 0.0,
) -> List[NoteInfo]:
    """Convert a MIDI-number history into NoteInfo objects.

    This mirrors parser output semantics used by GraphIDYOM:
    - monotonic synthetic timestamps,
    - duration/offset/length defaults,
    - both pitch and pitch_class filled from MIDI number.
    """
    out: List[NoteInfo] = []
    t = float(start_timestamp)
    for m in midi_history:
        p = ms.pitch.Pitch(midi=int(max(0, min(127, int(m)))))
        out.append(
            NoteInfo(
                timestamp=float(t),
                duration=float(default_duration),
                offset=float(t % beat_duration),
                pitch=str(p),
                pitch_class=p.name,
                midi=int(p.midi),
                length=int(default_length),
            )
        )
        t += float(default_duration)
    return out
