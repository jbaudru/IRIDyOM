

"""types.py

Shared types and minimal protocols used across the modular GraphIDyOM codebase.

Why this file exists
- Avoid circular imports by keeping the foundational dataclasses, typing aliases,
  and Protocol interfaces in one place.
- Keep this file lightweight: no heavy dependencies (music21, networkx) beyond
  standard library typing.

Conventions
- Note/event information is represented with the NoteInfo dataclass.
- Graph nodes/labels are strings produced by a TokenCodec.
- Distributions are plain dicts mapping symbols -> probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, TypeAlias


# -------------------------
# Core aliases
# -------------------------

Symbol: TypeAlias = str
NodeLabel: TypeAlias = str
Order: TypeAlias = int

Counts: TypeAlias = Dict[Symbol, float]
Dist: TypeAlias = Dict[Symbol, float]


# -------------------------
# Data structures
# -------------------------


@dataclass(frozen=True)
class NoteInfo:
    """A minimal, model-agnostic representation of a monophonic note event.

    Fields:
    - timestamp: onset time in quarterLength units (float)
    - duration: duration until next onset in quarterLength units (float)
    - offset: onset % beat_duration (float)
    - pitch: pitch with octave (e.g., 'C#4')
    - pitch_class: pitch class only (e.g., 'C#')
    - midi: integer MIDI pitch number (0-127)
    - length: IDyOM-style length token in ticks (inter-onset duration; terminal note uses 0)
    """

    timestamp: float  # onset time in quarterLength units
    duration: float   # duration until next onset (or note duration for last note)
    offset: float     # onset % beat_duration
    pitch: str        # pitch with octave (e.g., 'C#4')
    pitch_class: str  # pitch class only (e.g., 'C#')
    midi: int         # MIDI pitch number (integer)
    length: int       # IDyOM-style length token in ticks


# -------------------------
# Protocol interfaces
# -------------------------


class MidiParser(Protocol):
    """Parses a file into a monophonic sequence of NoteInfo."""

    def parse_file(self, path: str) -> List[NoteInfo]:
        """Parse a single MIDI/MusicXML file into a monophonic sequence.

        Implementations may raise exceptions for unreadable files or polyphony.
        """

        ...


class TokenCodec(Protocol):
    """Viewpoint/token encoder/decoder used by both graph building and the model.

    Responsibilities
    - Given a window of NoteInfo, produce a *node label* string.
    - Optionally decode a label to extract the "next symbol" used by the model
      (e.g., pitch class, interval token, combined viewpoint token, etc.).

    Graph construction treats labels as opaque; the model may need decoding.
    """

    @property
    def beat_duration(self) -> float:
        ...

    def interval_only_mode(self) -> bool:
        """True if labels represent interval-only streams with NO note tokens."""

        ...

    def window_size_for_order(self, order: Order) -> int:
        """Return the note-window size needed to realize a logical model order."""

        ...

    # ---- Encoding ----
    def window_label(self, window: Sequence[NoteInfo]) -> NodeLabel:
        """Encode a window (order-n context) into a node label string."""

        ...

    # ---- Decoding (for model/stats) ----
    def extract_json_objects(self, label: NodeLabel) -> List[str]:
        """Extract JSON object substrings from a node label.

        This supports labels that are composed by joining JSON objects with commas.
        For interval-only mode, this returns a list of interval JSON objects.
        """

        ...

    def symbol_from_dest_label(self, dest_label: NodeLabel, order_k: Order) -> Optional[Symbol]:
        """Return the predicted symbol associated with arriving at `dest_label`.

        Used by graph_stats to convert outgoing edges into a distribution.

        Examples
        - If order_k == 3 and labels are "{note},{note},{note},...", return
          the last note's viewpoint token (or an interval token, depending on setup).
        - In interval-only mode, order_k==1 may map to a sentinel like "NO_EVENT".
        """

        ...

    def symbol_at_index(self, seq: Sequence[NoteInfo], i: int) -> Optional[Symbol]:
        """Return the target symbol for the i-th note event in a sequence.

        This is used for evaluation (prob of the actually observed next event).
        """

        ...


class MergeStrategy(Protocol):
    """Combines multiple component distributions (e.g., orders, STM/LTM) into one."""

    name: str

    def merge(
        self,
        dists: Sequence[Dist],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> Dist:
        ...


class Sampler(Protocol):
    """Sampling strategy for drawing a symbol from a probability distribution."""

    name: str

    def sample(self, dist: Mapping[Symbol, float]) -> Symbol:
        ...


# -------------------------
# Optional small result container
# -------------------------


@dataclass(frozen=True)
class StepPrediction:
    """A minimal container for per-step evaluation outputs.

    Keep this small and stable; downstream code can extend with richer diagnostics.
    """

    index: int
    observed: Optional[Symbol]
    prob: float
    surprisal: float


@dataclass(frozen=True)
class ProcessedSequence:
    """Represents the result of running a model over a sequence/file."""

    steps: Tuple[StepPrediction, ...]
    alphabet: Tuple[Symbol, ...] = ()

    def mean_surprisal(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.surprisal for s in self.steps) / float(len(self.steps))
