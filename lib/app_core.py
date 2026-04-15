"""Core service layer for a lightweight GraphIDYOM app and local API."""

from __future__ import annotations

from collections import deque
import json
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Optional, Sequence

import music21 as ms

from graph_build import GraphBuildConfig
from graph_types import Dist, NoteInfo
from interaction_history import midi_history_to_noteinfos
from midi_parse import create_midi_parser
from model import GraphIDYOMModel
from multi_model import ModelInstance, MultiModelIDYOM
from pretrained_models_manager import PretrainedModelsManager
from token_codec import JsonTokenCodec, ViewpointConfig
from viewpoint_system import ViewpointConfigRegistry


def _midi_to_pitch(midi: int) -> str:
    return str(ms.pitch.Pitch(midi=int(max(0, min(127, int(midi))))))


def _pitch_to_midi(pitch: str) -> int:
    return int(ms.pitch.Pitch(pitch).midi)


def _project_dist_to_pitch_octave_fallback(
    dist: Mapping[str, float],
    *,
    last_pitch: Optional[str],
    target_alphabet: Optional[Sequence[str]],
) -> Dist:
    projected: Dict[str, float] = {}
    allowed = set(target_alphabet) if target_alphabet else None

    for symbol, prob in dist.items():
        if prob <= 0.0:
            continue

        # Already a plain pitch+octave string.
        try:
            p = ms.pitch.Pitch(symbol)
            pitch_candidate = str(p)
            if allowed is None or pitch_candidate in allowed:
                projected[pitch_candidate] = projected.get(pitch_candidate, 0.0) + float(prob)
                continue
        except Exception:
            pass

        # JSON symbol from viewpoint codecs.
        try:
            obj = json.loads(symbol)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        if "pitch" in obj:
            pitch_str = str(obj["pitch"])
            has_octave = any(c.isdigit() for c in pitch_str)
            if has_octave:
                if allowed is None or pitch_str in allowed:
                    projected[pitch_str] = projected.get(pitch_str, 0.0) + float(prob)
            elif allowed is not None:
                matches = [x for x in allowed if x.startswith(pitch_str)]
                if matches:
                    each = float(prob) / float(len(matches))
                    for m in matches:
                        projected[m] = projected.get(m, 0.0) + each
            continue

        if "midi_number" in obj:
            try:
                pitch_str = _midi_to_pitch(int(obj["midi_number"]))
                if allowed is None or pitch_str in allowed:
                    projected[pitch_str] = projected.get(pitch_str, 0.0) + float(prob)
            except Exception:
                pass
            continue

        if "interval" in obj and last_pitch is not None:
            try:
                base = ms.pitch.Pitch(last_pitch)
                new_pitch = ms.pitch.Pitch(midi=int(base.midi + int(obj["interval"])))
                pitch_str = str(new_pitch)
                if allowed is None or pitch_str in allowed:
                    projected[pitch_str] = projected.get(pitch_str, 0.0) + float(prob)
            except Exception:
                pass

    s = float(sum(projected.values()))
    if s > 0:
        return {k: (v / s) for k, v in projected.items()}
    return {}


def preset_viewpoint_config(name: str) -> ViewpointConfig:
    return ViewpointConfigRegistry.preset(name)


def _normalize_target_viewpoint(target_viewpoint: Optional[str]) -> Optional[str]:
    if target_viewpoint is None:
        return None
    t = str(target_viewpoint).strip()
    if not t or t.lower() in {"none", "null"}:
        return None
    return t


@dataclass
class _InteractiveSession:
    session_id: str
    base_model_id: str
    model: GraphIDYOMModel
    context_limit: int
    recent_history: Deque[NoteInfo] = field(default_factory=deque)
    next_timestamp: float = 0.0
    default_duration: float = 1.0
    default_length: int = 24
    total_notes_observed: int = 0


class GraphIDYOMAppService:
    def __init__(self) -> None:
        self.manager = PretrainedModelsManager()
        self._models: Dict[str, GraphIDYOMModel] = {}
        self._meta: Dict[str, Dict] = {}
        self._sessions: Dict[str, _InteractiveSession] = {}

    def list_pretrained_models(self, dataset_name: Optional[str] = None) -> List[Dict]:
        raw = self.manager.list_available_models(dataset_name=dataset_name)
        rows: List[Dict] = []
        if dataset_name:
            for folder_name, metadata in raw.items():
                rows.append(
                    {
                        "dataset": dataset_name,
                        "folder": folder_name,
                        "path": str(self.manager.datasets_dir / dataset_name / folder_name),
                        "metadata": metadata,
                    }
                )
            return sorted(rows, key=lambda x: x["folder"])

        for ds, items in raw.items():
            for folder_name, metadata in items.items():
                rows.append(
                    {
                        "dataset": ds,
                        "folder": folder_name,
                        "path": str(self.manager.datasets_dir / ds / folder_name),
                        "metadata": metadata,
                    }
                )
        return sorted(rows, key=lambda x: (x["dataset"], x["folder"]))

    def _make_model_from_metadata(self, metadata: Mapping) -> GraphIDYOMModel:
        viewpoint_config = metadata.get("viewpoint_config")
        if not isinstance(viewpoint_config, dict):
            raise RuntimeError(
                "Invalid model metadata: missing 'viewpoint_config'. "
                "Use app_cli train/save-managed flow or provide legacy metadata-compatible model folders."
            )
        vp_cfg = ViewpointConfig(**viewpoint_config)
        parser_mode = str(metadata.get("parser_mode", "mido"))
        parser = create_midi_parser(
            parser_mode,
            beat_duration=float(vp_cfg.beat_duration),
            enharmony=bool(vp_cfg.enharmony),
            discard_if_multiple_parts=False,
            use_first_part_only=True,
            quarter_length_divisors=(16,),
        )
        model = GraphIDYOMModel(
            parser=parser,
            codec=JsonTokenCodec(vp_cfg),
            orders=tuple(metadata["orders"]),
            graph_build_config=GraphBuildConfig(),
            use_stm=True,
            use_ppm=bool(metadata.get("use_ppm", False)),
            ppm_excluded_count=int(metadata.get("ppm_excluded_count", 1)),
            ppm_escape_method=str(metadata.get("ppm_escape_method", "c")),
            eta_ltm=float(metadata.get("eta_ltm", 0.0)),
            eta_stm=float(metadata.get("eta_stm", metadata.get("eta_ltm", 0.0))),
            eta_max_depth=int(metadata.get("eta_max_depth", 15)),
            verbosity=0,
            target_viewpoint=_normalize_target_viewpoint(metadata.get("target_viewpoint")),
        )
        return model

    def _infer_legacy_viewpoint_config(self, model_path: Path, metadata: Mapping) -> Dict[str, object]:
        # Backward-compatible inference for older saved models that predate
        # explicit viewpoint_config/parser_mode fields in metadata.json.
        flags: Dict[str, bool] = {
            "pitch": False,
            "octave": False,
            "midi_number": False,
            "duration": False,
            "length": False,
            "offset": False,
            "interval": False,
            "bioi_ratio": False,
        }

        alphabet_file = model_path / "alphabet.json"
        if alphabet_file.exists():
            try:
                symbols = json.loads(alphabet_file.read_text(encoding="utf-8"))
                if isinstance(symbols, list):
                    for symbol in symbols[:256]:
                        if not isinstance(symbol, str):
                            continue
                        try:
                            obj = json.loads(symbol)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        if "pitch" in obj:
                            flags["pitch"] = True
                            pitch_value = str(obj.get("pitch", ""))
                            if any(ch.isdigit() for ch in pitch_value):
                                flags["octave"] = True
                        if "midi_number" in obj:
                            flags["midi_number"] = True
                        if "duration" in obj:
                            flags["duration"] = True
                        if "length" in obj:
                            flags["length"] = True
                        if "offset" in obj:
                            flags["offset"] = True
                        if "interval" in obj:
                            flags["interval"] = True
                        if "bioi_ratio" in obj:
                            flags["bioi_ratio"] = True
            except Exception:
                pass

        if not any(
            (
                flags["pitch"],
                flags["midi_number"],
                flags["duration"],
                flags["length"],
                flags["offset"],
                flags["interval"],
                flags["bioi_ratio"],
            )
        ):
            name = model_path.name.lower()
            if "bioi_ratio" in name or "bioi-ratio" in name:
                flags["bioi_ratio"] = True
            elif "interval" in name:
                flags["interval"] = True
            elif "midi" in name:
                flags["midi_number"] = True
            elif "duration" in name:
                flags["duration"] = True
            elif "length" in name:
                flags["length"] = True
            elif "offset" in name:
                flags["offset"] = True
            else:
                flags["pitch"] = True

        return {
            **flags,
            "beat_duration": float(metadata.get("beat_duration", 1.0)),
            "enharmony": bool(metadata.get("enharmony", True)),
            "token_round_ndigits": metadata.get("token_round_ndigits"),
        }

    def _normalize_model_metadata(self, model_path: Path, metadata: Mapping) -> Dict:
        normalized = dict(metadata)
        if not isinstance(normalized.get("viewpoint_config"), dict):
            normalized["viewpoint_config"] = self._infer_legacy_viewpoint_config(model_path, normalized)
        if "parser_mode" not in normalized:
            normalized["parser_mode"] = "mido"
        return normalized

    def load_model_from_dir(self, model_dir: str) -> Dict:
        model_path = Path(model_dir).resolve()
        metadata_file = model_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"metadata.json not found in {model_path}")

        metadata_raw = json.loads(metadata_file.read_text(encoding="utf-8"))
        metadata = self._normalize_model_metadata(model_path, metadata_raw)
        model = self._make_model_from_metadata(metadata)
        model.load_ltm(str(model_path))

        model_id = uuid.uuid4().hex[:12]
        self._models[model_id] = model
        self._meta[model_id] = {
            "model_dir": str(model_path),
            "metadata": metadata,
        }
        return {"model_id": model_id, "model_dir": str(model_path), "metadata": metadata}

    def load_pretrained_model(self, dataset_name: str, model_folder_name: str) -> Dict:
        model_dir = self.manager.datasets_dir / dataset_name / model_folder_name
        return self.load_model_from_dir(str(model_dir))

    def train_model(
        self,
        *,
        input_folder: str,
        orders: Sequence[int],
        viewpoint_preset: str = "midi",
        viewpoint_config: Optional[Mapping[str, object]] = None,
        target_viewpoint: Optional[str] = "pitchOctave",
        parser_mode: str = "mido",
        use_ppm: bool = False,
        ppm_escape_method: str = "c",
        augmented: bool = False,
        transpose_range: Sequence[int] = (-6, 6),
        save_dir: Optional[str] = None,
        save_managed_dataset: Optional[str] = None,
        save_managed_viewpoint_name: Optional[str] = None,
    ) -> Dict:
        if viewpoint_config is not None:
            vp_cfg = ViewpointConfig(**dict(viewpoint_config))
        else:
            vp_cfg = preset_viewpoint_config(viewpoint_preset)
        target_viewpoint_norm = _normalize_target_viewpoint(target_viewpoint)
        parser = create_midi_parser(
            parser_mode,
            beat_duration=float(vp_cfg.beat_duration),
            enharmony=bool(vp_cfg.enharmony),
            discard_if_multiple_parts=False,
            use_first_part_only=True,
            quarter_length_divisors=(16,),
        )
        model = GraphIDYOMModel(
            parser=parser,
            codec=JsonTokenCodec(vp_cfg),
            orders=tuple(int(o) for o in orders),
            graph_build_config=GraphBuildConfig(
                augment=bool(augmented),
                augment_transposition=True,
                transpose_range=(int(transpose_range[0]), int(transpose_range[1])),
                augment_rhythm=True,
            ),
            use_stm=True,
            use_ppm=bool(use_ppm),
            ppm_excluded_count=1,
            ppm_escape_method=str(ppm_escape_method),
            verbosity=1,
            target_viewpoint=target_viewpoint_norm,
        )
        model.fit_folder(str(Path(input_folder).resolve()), export_graphml=False)

        saved_to = None
        if save_dir:
            model.save_ltm(str(Path(save_dir).resolve()))
            saved_to = str(Path(save_dir).resolve())
        elif save_managed_dataset and save_managed_viewpoint_name:
            model.save_ltm(
                dataset_name=str(save_managed_dataset),
                source_viewpoint=str(save_managed_viewpoint_name),
                augmented=bool(augmented),
                manager=self.manager,
            )
            cfg_folder = f"{save_managed_viewpoint_name}_augmented_{str(bool(augmented)).lower()}"
            saved_to = str(self.manager.datasets_dir / save_managed_dataset / cfg_folder)

        model_id = uuid.uuid4().hex[:12]
        self._models[model_id] = model
        self._meta[model_id] = {
            "trained": True,
            "input_folder": str(Path(input_folder).resolve()),
            "viewpoint_config": asdict(vp_cfg),
            "orders": [int(o) for o in orders],
            "target_viewpoint": target_viewpoint_norm,
            "use_ppm": bool(use_ppm),
            "ppm_escape_method": str(ppm_escape_method),
            "parser_mode": str(parser_mode),
            "saved_to": saved_to,
        }
        return {"model_id": model_id, "saved_to": saved_to, "metadata": self._meta[model_id]}

    def get_loaded_models(self) -> Dict[str, Dict]:
        return dict(self._meta)

    def _metadata_for_model(self, model_id: str) -> Mapping:
        if model_id not in self._meta:
            raise KeyError(f"Unknown model_id: {model_id}")
        meta = self._meta[model_id]
        if isinstance(meta.get("metadata"), Mapping):
            return meta["metadata"]
        return meta

    def _clone_model_for_session(self, model_id: str) -> GraphIDYOMModel:
        if model_id not in self._models:
            raise KeyError(f"Unknown model_id: {model_id}")

        base_model = self._models[model_id]
        session_model = self._make_model_from_metadata(self._metadata_for_model(model_id))
        session_model.ltm_graphs = base_model.ltm_graphs
        session_model.alphabet = tuple(base_model.alphabet)
        session_model.target_alphabet = tuple(base_model.target_alphabet)
        session_model._projection_cache.clear()
        session_model._refresh_merge_alphabet_size()
        if session_model.use_stm:
            session_model.reset_stm()
        return session_model

    def _session_summary(self, session: _InteractiveSession) -> Dict:
        return {
            "session_id": session.session_id,
            "model_id": session.base_model_id,
            "context_size": len(session.recent_history),
            "context_limit": int(session.context_limit),
            "notes_observed": int(session.total_notes_observed),
            "history_midi": [int(note.midi) for note in session.recent_history],
        }

    def start_session(
        self,
        *,
        model_id: str,
        midi_history: Optional[Sequence[int]] = None,
        default_duration: float = 1.0,
        default_length: int = 24,
    ) -> Dict:
        session_model = self._clone_model_for_session(model_id)
        context_limit = max(session_model.orders) if session_model.orders else 0
        session_id = uuid.uuid4().hex[:12]
        session = _InteractiveSession(
            session_id=session_id,
            base_model_id=model_id,
            model=session_model,
            context_limit=int(context_limit),
            recent_history=deque(maxlen=(int(context_limit) if context_limit > 0 else None)),
            default_duration=float(default_duration),
            default_length=int(default_length),
        )
        self._sessions[session_id] = session
        if midi_history:
            self.observe_in_session(session_id=session_id, midi_notes=midi_history)
        return self._session_summary(session)

    def reset_session(self, *, session_id: str) -> Dict:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session_id: {session_id}")
        session = self._sessions[session_id]
        session.model.reset_stm()
        session.recent_history.clear()
        session.next_timestamp = 0.0
        session.total_notes_observed = 0
        return self._session_summary(session)

    def close_session(self, *, session_id: str) -> Dict:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session_id: {session_id}")
        session = self._sessions.pop(session_id)
        return self._session_summary(session)

    def observe_in_session(self, *, session_id: str, midi_notes: Sequence[int]) -> Dict:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session_id: {session_id}")
        session = self._sessions[session_id]
        notes = midi_history_to_noteinfos(
            midi_notes,
            default_duration=float(session.default_duration),
            default_length=int(session.default_length),
            beat_duration=float(session.model.codec.beat_duration),
            start_timestamp=float(session.next_timestamp),
        )
        if notes:
            session.model.observe_notes(notes)
            for note in notes:
                session.recent_history.append(note)
            session.total_notes_observed += len(notes)
            session.next_timestamp = float(notes[-1].timestamp + notes[-1].duration)
        return self._session_summary(session)

    def _format_prediction_rows(
        self,
        *,
        model: GraphIDYOMModel,
        history: Sequence[NoteInfo],
        dist: Mapping[str, float],
        top_k: int,
        output: str,
    ) -> List[Dict]:
        output_mode = str(output).strip().lower()
        ranked_target = sorted(
            ((str(k), float(v)) for k, v in dist.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )[: int(top_k)]

        if output_mode in {"target", "raw"}:
            return [{"symbol": s, "prob": float(prob)} for s, prob in ranked_target]
        if output_mode in {"midi", "midi_number"} and model.target_viewpoint == "midi_number":
            return [{"midi": int(float(s)), "prob": float(prob)} for s, prob in ranked_target]
        if output_mode == "pitch" and model.target_viewpoint == "pitchOctave":
            return [{"pitch": s, "prob": float(prob)} for s, prob in ranked_target]

        if model.target_viewpoint == "pitchOctave":
            pitch_dist = {str(k): float(v) for k, v in dist.items()}
        elif model.target_viewpoint == "midi_number":
            pitch_dist = {}
            for s, p in ranked_target:
                try:
                    pitch_dist[_midi_to_pitch(int(float(s)))] = float(p)
                except Exception:
                    continue
        else:
            pitch_dist = _project_dist_to_pitch_octave_fallback(
                dist,
                last_pitch=history[-1].pitch if history else None,
                target_alphabet=model.target_alphabet if model.target_alphabet else None,
            )

        if not pitch_dist:
            raise RuntimeError(
                "Could not map model output to pitch+octave. "
                "Use output='target' for non-pitch targets or a model with pitch-compatible target/source."
            )

        ranked = sorted(pitch_dist.items(), key=lambda kv: kv[1], reverse=True)[: int(top_k)]
        if output_mode in ("midi", "midi_number"):
            return [{"midi": _pitch_to_midi(p), "prob": float(prob)} for p, prob in ranked]
        return [{"pitch": p, "prob": float(prob)} for p, prob in ranked]

    def predict_next_session(
        self,
        *,
        session_id: str,
        top_k: int = 32,
        output: str = "pitch",
        long_term_only: bool = False,
        short_term_only: bool = False,
    ) -> Dict:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session_id: {session_id}")
        session = self._sessions[session_id]
        history = list(session.recent_history)
        dist = session.model.predict_next_dist(
            history,
            long_term_only=bool(long_term_only),
            short_term_only=bool(short_term_only),
        )
        out_rows = self._format_prediction_rows(
            model=session.model,
            history=history,
            dist=dist,
            top_k=int(top_k),
            output=str(output),
        )
        return {
            **self._session_summary(session),
            "output": str(output),
            "target_viewpoint": session.model.target_viewpoint,
            "top_k": int(top_k),
            "long_term_only": bool(long_term_only),
            "short_term_only": bool(short_term_only),
            "predictions": out_rows,
        }

    def predict_next(
        self,
        *,
        model_id: str,
        midi_history: Sequence[int],
        top_k: int = 32,
        output: str = "pitch",
        long_term_only: bool = False,
        short_term_only: bool = False,
        reset_stm: bool = True,
        stm_context_maxlen: Optional[int] = None,
    ) -> Dict:
        if model_id not in self._models:
            raise KeyError(f"Unknown model_id: {model_id}")
        if not midi_history:
            raise ValueError("midi_history must contain at least one note")

        model = self._models[model_id]
        history = midi_history_to_noteinfos(midi_history, beat_duration=float(model.codec.beat_duration))

        if not long_term_only and getattr(model, "use_stm", False):
            model.prime_stm(
                history,
                reset=bool(reset_stm),
                maxlen=(int(stm_context_maxlen) if stm_context_maxlen is not None else None),
            )

        dist = model.predict_next_dist(
            history,
            long_term_only=bool(long_term_only),
            short_term_only=bool(short_term_only),
        )
        out_rows = self._format_prediction_rows(
            model=model,
            history=history,
            dist=dist,
            top_k=int(top_k),
            output=str(output),
        )

        return {
            "model_id": model_id,
            "history_midi": [int(x) for x in midi_history],
            "output": str(output),
            "target_viewpoint": model.target_viewpoint,
            "top_k": int(top_k),
            "long_term_only": bool(long_term_only),
            "short_term_only": bool(short_term_only),
            "predictions": out_rows,
        }

    def predict_next_composite(
        self,
        *,
        model_ids: Sequence[str],
        midi_history: Sequence[int],
        top_k: int = 64,
        output: str = "target",
        model_merge: str = "arith",
        long_term_only: bool = False,
        short_term_only: bool = False,
        reset_stm: bool = True,
        stm_context_maxlen: Optional[int] = None,
        max_symbols_per_target: Optional[int] = None,
    ) -> Dict:
        if not model_ids:
            raise ValueError("model_ids must contain at least one loaded model id")
        if not midi_history:
            raise ValueError("midi_history must contain at least one note")

        instances: List[ModelInstance] = []
        beat_duration_ref: Optional[float] = None
        target_viewpoints: List[str] = []

        for model_id in model_ids:
            if model_id not in self._models:
                raise KeyError(f"Unknown model_id: {model_id}")
            model = self._models[model_id]
            if model.target_viewpoint is None:
                raise ValueError(
                    f"Model {model_id!r} has target_viewpoint=None. "
                    "Composite target prediction requires explicit model targets."
                )
            target_viewpoints.append(str(model.target_viewpoint))

            bd = float(model.codec.beat_duration)
            if beat_duration_ref is None:
                beat_duration_ref = bd
            elif abs(beat_duration_ref - bd) > 1e-12:
                raise ValueError(
                    "All models must use the same beat_duration for composite prediction. "
                    f"Got {beat_duration_ref} and {bd}."
                )

            instances.append(ModelInstance(name=str(model_id), model=model))

        history = midi_history_to_noteinfos(
            midi_history,
            beat_duration=float(beat_duration_ref if beat_duration_ref is not None else 1.0),
        )

        if not long_term_only:
            for inst in instances:
                if getattr(inst.model, "use_stm", False):
                    inst.model.prime_stm(
                        history,
                        reset=bool(reset_stm),
                        maxlen=(int(stm_context_maxlen) if stm_context_maxlen is not None else None),
                    )

        mm = MultiModelIDYOM(
            model_instances=instances,
            model_merge=str(model_merge),
            weight_mode="inverse_power",
            b=1.0,
            target_viewpoint=None,
            verbosity=0,
        )
        dist_joint = mm.predict_next_joint_dist(
            history,
            long_term_only=bool(long_term_only),
            short_term_only=bool(short_term_only),
            max_symbols_per_target=max_symbols_per_target,
        )

        ranked = sorted(
            ((str(k), float(v)) for k, v in dist_joint.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )[: int(top_k)]

        output_mode = str(output).strip().lower()
        out_rows = []
        for symbol, prob in ranked:
            targets = MultiModelIDYOM.decode_joint_symbol(symbol)
            row = {"targets": targets, "prob": float(prob)}

            pitch_value: Optional[str] = None
            midi_value: Optional[int] = None
            if "pitchOctave" in targets:
                pitch_value = str(targets["pitchOctave"])
                try:
                    midi_value = _pitch_to_midi(pitch_value)
                except Exception:
                    midi_value = None
            elif "midi_number" in targets:
                try:
                    midi_value = int(float(targets["midi_number"]))
                    pitch_value = _midi_to_pitch(midi_value)
                except Exception:
                    midi_value = None
                    pitch_value = None

            if pitch_value is not None:
                row["pitch"] = pitch_value
            if midi_value is not None:
                row["midi"] = int(midi_value)

            if output_mode in {"target", "raw"}:
                out_rows.append(row)
            elif output_mode in {"pitch"}:
                if "pitch" in row:
                    out_rows.append(row)
            elif output_mode in {"midi", "midi_number"}:
                if "midi" in row:
                    out_rows.append(row)
            else:
                out_rows.append(row)

        unique_targets = sorted(set(target_viewpoints))
        return {
            "model_ids": [str(x) for x in model_ids],
            "history_midi": [int(x) for x in midi_history],
            "output": str(output),
            "model_merge": str(model_merge),
            "target_viewpoints": unique_targets,
            "top_k": int(top_k),
            "long_term_only": bool(long_term_only),
            "short_term_only": bool(short_term_only),
            "max_symbols_per_target": (
                int(max_symbols_per_target) if max_symbols_per_target is not None else None
            ),
            "predictions": out_rows,
        }

    def robot_select_note(
        self,
        *,
        model_id: str,
        robot_midi_history: Sequence[int],
        global_monomelody_history: Optional[Sequence[int]] = None,
        collisions_active: bool = False,
        stm_order_max: int = 5,
        top_k: int = 32,
        output: str = "midi",
        selection: str = "argmax",
        long_term_only: bool = False,
        short_term_only: bool = False,
    ) -> Dict:
        if model_id not in self._models:
            raise KeyError(f"Unknown model_id: {model_id}")

        if collisions_active:
            return {
                "model_id": model_id,
                "collisions_active": True,
                "selected": None,
                "predictions": [],
                "reason": "Collisions still active; robot note selection deferred.",
            }

        source_history = list(global_monomelody_history or robot_midi_history)
        if not source_history:
            raise ValueError("Provide robot_midi_history or global_monomelody_history with at least one MIDI note")

        context_len = max(1, min(5, int(stm_order_max)))
        context = source_history[-context_len:]

        pred = self.predict_next(
            model_id=model_id,
            midi_history=context,
            top_k=int(top_k),
            output=output,
            long_term_only=bool(long_term_only),
            short_term_only=bool(short_term_only),
            reset_stm=True,
            stm_context_maxlen=context_len,
        )

        rows = list(pred.get("predictions", []))
        if not rows:
            raise RuntimeError("No prediction candidates available for robot selection")

        mode = str(selection).strip().lower()
        if mode not in ("argmax", "sample"):
            mode = "argmax"

        if mode == "sample":
            weights = [max(0.0, float(r.get("prob", 0.0))) for r in rows]
            total = sum(weights)
            if total <= 0.0:
                chosen = rows[0]
            else:
                thresh = random.random() * total
                acc = 0.0
                chosen = rows[-1]
                for row, w in zip(rows, weights):
                    acc += w
                    if acc >= thresh:
                        chosen = row
                        break
        else:
            chosen = rows[0]

        return {
            "model_id": model_id,
            "collisions_active": False,
            "selection_mode": mode,
            "context_midi": context,
            "context_size": len(context),
            "long_term_only": bool(long_term_only),
            "short_term_only": bool(short_term_only),
            "selected": chosen,
            "predictions": rows,
        }
