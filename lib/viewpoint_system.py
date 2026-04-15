"""viewpoint_system.py

IDyOM-style viewpoint hierarchy helpers for GraphIDYOM.

This module provides a semantic layer over the existing token codec:
- basic viewpoints
- derived viewpoints
- linked viewpoints
- threaded viewpoints
- test viewpoints

Only the viewpoints currently implemented in GraphIDYOM are modeled here,
but in an extensible shape so additional derived/threaded/test viewpoints
can be added with minimal changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from token_codec import ViewpointConfig


class ViewpointKind(str, Enum):
    BASIC = "basic"
    DERIVED = "derived"
    LINKED = "linked"
    THREADED = "threaded"
    TEST = "test"


@dataclass(frozen=True)
class ViewpointDefinition:
    """Definition of a single viewpoint in the IDyOM hierarchy."""

    name: str
    kind: ViewpointKind
    # Basic viewpoints this viewpoint can predict (IDyOM typeset concept).
    typeset: Tuple[str, ...]
    # Source components needed to instantiate this viewpoint in GraphIDYOM.
    components: Tuple[str, ...]
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewpointSpec:
    """Semantic description of a configured viewpoint system."""

    name: str
    kind: ViewpointKind
    components: Tuple[str, ...]
    typeset: Tuple[str, ...]
    links: Tuple[str, ...] = ()
    metadata: Dict[str, str] = field(default_factory=dict)


_BASIC_DEFINITIONS: Dict[str, ViewpointDefinition] = {
    "pitchClass": ViewpointDefinition(
        name="pitchClass",
        kind=ViewpointKind.BASIC,
        typeset=("pitch",),
        components=("pitch",),
    ),
    "pitchOctave": ViewpointDefinition(
        name="pitchOctave",
        kind=ViewpointKind.BASIC,
        typeset=("pitch",),
        components=("pitch",),
    ),
    "midi_number": ViewpointDefinition(
        name="midi_number",
        kind=ViewpointKind.BASIC,
        typeset=("pitch",),
        components=("midi_number",),
    ),
    "duration": ViewpointDefinition(
        name="duration",
        kind=ViewpointKind.BASIC,
        typeset=("duration",),
        components=("duration",),
    ),
    "length": ViewpointDefinition(
        name="length",
        kind=ViewpointKind.BASIC,
        typeset=("length",),
        components=("length",),
    ),
    "offset": ViewpointDefinition(
        name="offset",
        kind=ViewpointKind.BASIC,
        typeset=("offset",),
        components=("offset",),
    ),
}

_DERIVED_DEFINITIONS: Dict[str, ViewpointDefinition] = {
    "interval": ViewpointDefinition(
        name="interval",
        kind=ViewpointKind.DERIVED,
        typeset=("pitch",),
        components=("interval",),
    ),
    "bioi_ratio": ViewpointDefinition(
        name="bioi_ratio",
        kind=ViewpointKind.DERIVED,
        typeset=("bioi_ratio",),
        components=("bioi_ratio",),
        metadata={"lisp_equivalent": "bioi-ratio"},
    ),
}


class ViewpointConfigRegistry:
    """Central registry for source viewpoint presets/components.

    This keeps preset and component parsing in one place so adding new viewpoints
    does not require duplicating mappings in CLI/app scripts.
    """

    _PRESETS: Mapping[str, ViewpointConfig] = {
        "midi": ViewpointConfig(
            pitch=False, octave=False, midi_number=True, duration=False, length=False, offset=False, interval=False
        ),
        "pitch_octave": ViewpointConfig(
            pitch=True, octave=True, midi_number=False, duration=False, length=False, offset=False, interval=False
        ),
        "pitch_class": ViewpointConfig(
            pitch=True, octave=False, midi_number=False, duration=False, length=False, offset=False, interval=False
        ),
        "interval": ViewpointConfig(
            pitch=False,
            octave=False,
            midi_number=False,
            duration=False,
            length=False,
            offset=False,
            interval=True,
            bioi_ratio=False,
        ),
        "length": ViewpointConfig(
            pitch=False,
            octave=False,
            midi_number=False,
            duration=False,
            length=True,
            offset=False,
            interval=False,
            bioi_ratio=False,
        ),
        "duration": ViewpointConfig(
            pitch=False,
            octave=False,
            midi_number=False,
            duration=True,
            length=False,
            offset=False,
            interval=False,
            bioi_ratio=False,
        ),
        "offset": ViewpointConfig(
            pitch=False,
            octave=False,
            midi_number=False,
            duration=False,
            length=False,
            offset=True,
            interval=False,
            bioi_ratio=False,
        ),
        "bioi_ratio": ViewpointConfig(
            pitch=False,
            octave=False,
            midi_number=False,
            duration=False,
            length=False,
            offset=False,
            interval=False,
            bioi_ratio=True,
        ),
    }

    _ALIASES: Mapping[str, str] = {
        "midi_number": "midi",
        "pitch": "pitch_class",
        "pitchoctave": "pitch_octave",
        "pitch+octave": "pitch_octave",
        "bioi-ratio": "bioi_ratio",
        "bioiratio": "bioi_ratio",
    }

    @classmethod
    def preset_choices(cls) -> Tuple[str, ...]:
        return tuple(cls._PRESETS.keys())

    @classmethod
    def preset(cls, name: str) -> ViewpointConfig:
        key = str(name).strip().lower()
        key = cls._ALIASES.get(key, key)
        if key not in cls._PRESETS:
            raise ValueError(f"Unknown viewpoint preset: {name!r}")
        return replace(cls._PRESETS[key])

    @classmethod
    def from_components(
        cls,
        components: Sequence[str],
        *,
        octave: bool = False,
        beat_duration: float = 1.0,
        enharmony: bool = True,
        token_round_ndigits: int | None = None,
    ) -> ViewpointConfig:
        flags = {
            "pitch": False,
            "octave": False,
            "midi_number": False,
            "duration": False,
            "length": False,
            "offset": False,
            "interval": False,
            "bioi_ratio": False,
        }
        for raw in components:
            comp = str(raw).strip().lower()
            if comp in {"pitch", "pitch_class"}:
                flags["pitch"] = True
            elif comp in {"pitch_octave", "pitch+octave", "pitchoctave"}:
                flags["pitch"] = True
                flags["octave"] = True
            elif comp in {"midi", "midi_number"}:
                flags["midi_number"] = True
            elif comp in {"duration", "length", "offset", "interval", "bioi_ratio", "bioi-ratio", "bioiratio"}:
                if comp in {"bioi-ratio", "bioiratio"}:
                    comp = "bioi_ratio"
                flags[comp] = True
            else:
                raise ValueError(f"Unknown viewpoint component: {raw!r}")

        if flags["pitch"] and bool(octave):
            flags["octave"] = True

        return ViewpointConfig(
            pitch=bool(flags["pitch"]),
            octave=bool(flags["octave"]),
            midi_number=bool(flags["midi_number"]),
            duration=bool(flags["duration"]),
            length=bool(flags["length"]),
            offset=bool(flags["offset"]),
            interval=bool(flags["interval"]),
            bioi_ratio=bool(flags["bioi_ratio"]),
            beat_duration=float(beat_duration),
            enharmony=bool(enharmony),
            token_round_ndigits=token_round_ndigits,
        )


def enabled_components(cfg: ViewpointConfig) -> Tuple[str, ...]:
    """Return enabled source components in deterministic order."""

    components = []
    if cfg.pitch:
        components.append("pitch")
    if cfg.midi_number:
        components.append("midi_number")
    if cfg.duration:
        components.append("duration")
    if cfg.length:
        components.append("length")
    if cfg.offset:
        components.append("offset")
    if cfg.interval:
        components.append("interval")
    if getattr(cfg, "bioi_ratio", False):
        components.append("bioi_ratio")
    return tuple(components)


def component_viewpoint_name(component: str, cfg: ViewpointConfig) -> str:
    """Return the viewpoint name corresponding to one codec component."""

    if component == "pitch":
        return "pitchOctave" if bool(cfg.octave) else "pitchClass"
    return str(component)


def definitions_for_config(cfg: ViewpointConfig) -> Tuple[ViewpointDefinition, ...]:
    """Resolve configured source components to viewpoint definitions."""

    defs = []
    for comp in enabled_components(cfg):
        if comp == "pitch":
            defs.append(_BASIC_DEFINITIONS[component_viewpoint_name(comp, cfg)])
            continue
        if comp in _BASIC_DEFINITIONS:
            defs.append(_BASIC_DEFINITIONS[comp])
            continue
        if comp in _DERIVED_DEFINITIONS:
            defs.append(_DERIVED_DEFINITIONS[comp])
            continue
        raise ValueError(f"Unknown viewpoint component in config: {comp!r}")
    return tuple(defs)


def _ordered_unique(items: Iterable[str]) -> Tuple[str, ...]:
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return tuple(out)


def viewpoint_name_from_config(cfg: ViewpointConfig) -> str:
    """Build deterministic source viewpoint name (existing GraphIDYOM format)."""

    defs = definitions_for_config(cfg)
    if not defs:
        return "empty"
    if len(defs) == 1:
        return defs[0].name
    return "__".join(d.name for d in defs)


def viewpoint_typeset_from_config(cfg: ViewpointConfig) -> Tuple[str, ...]:
    """Return merged typeset for configured components."""

    defs = definitions_for_config(cfg)
    return _ordered_unique(t for d in defs for t in d.typeset)


def classify_viewpoint_config(cfg: ViewpointConfig) -> ViewpointSpec:
    """Classify a ViewpointConfig into an IDyOM-style viewpoint spec."""

    defs = definitions_for_config(cfg)
    if not defs:
        raise ValueError("ViewpointConfig enables no components")

    name = viewpoint_name_from_config(cfg)
    components = tuple(d.name for d in defs)
    typeset = viewpoint_typeset_from_config(cfg)

    if len(defs) == 1:
        kind = defs[0].kind
        links: Tuple[str, ...] = ()
    else:
        # Multi-component codec tokens are linked viewpoints in IDyOM terms.
        kind = ViewpointKind.LINKED
        links = tuple(d.name for d in defs)

    return ViewpointSpec(
        name=name,
        kind=kind,
        components=components,
        typeset=typeset,
        links=links,
        metadata={"octave": str(bool(cfg.octave))},
    )


def supports_target_viewpoint(cfg: ViewpointConfig, target_viewpoint: str) -> bool:
    """Whether the configured source components can predict target_viewpoint."""

    target = str(target_viewpoint)
    typeset = set(viewpoint_typeset_from_config(cfg))
    if target in {"pitchOctave", "pitchClass", "midi_number", "interval"}:
        return "pitch" in typeset
    if target == "bioi_ratio":
        return "bioi_ratio" in typeset
    if target == "length":
        return ("length" in typeset) or ("bioi_ratio" in typeset)
    if target == "duration":
        return "duration" in typeset
    if target == "offset":
        return "offset" in typeset
    return False
