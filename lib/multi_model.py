"""multi_model.py

Multi-model instance orchestrator for GraphIDyOM.

This module enables combining multiple GraphIDYOMModel instances with different viewpoints.
The merging logic implements a two-layer distribution mixing strategy:

1. First layer: Merge all LTM distributions from different models
2. First layer: Merge all STM distributions from different models
3. Second layer: Merge the combined LTM and STM using relative entropy weights

The merging is arithmetic (as specified in the framework design).

Design principles
- Each model instance operates independently with its own viewpoint, codec, and graphs.
- All models project their final distributions to a common target viewpoint.
- The two-layer merging avoids direct mixing of per-order distributions across models.
- Weights are computed using entropy-weighted strategies (same as single-model case).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from graph_types import Dist, NoteInfo, ProcessedSequence, StepPrediction, Symbol
from merge import EntropyArithmeticMerge, EntropyGeometricMerge, entropy_weights
from model import GraphIDYOMModel
from pretrained_models_manager import ModelConfig, PretrainedModelsManager
from trace import (
    MergeTrace,
    ModelTrace,
    StepTrace,
    Trace,
    TraceConfig,
    dist_entropy_bits,
    dist_relative_entropy,
)


@dataclass
class ModelInstance:
    """A named instance of GraphIDYOMModel."""

    name: str
    model: GraphIDYOMModel


class MultiModelIDYOM:
    """Combine multiple GraphIDYOMModel instances with different viewpoints.

    Each model contributes an LTM and STM distribution. These are merged in two layers:
    1. Merge all LTMs together -> combined_ltm_dist
    2. Merge all STMs together -> combined_stm_dist
    3. Merge combined_ltm_dist and combined_stm_dist using entropy weights

    The merging strategy is arithmetic by default (as per framework design).
    """

    def __init__(
        self,
        model_instances: Sequence[Union[GraphIDYOMModel, ModelInstance]],
        *,
        model_merge: Optional[str] = "arith",
        weight_mode: str = "inverse_power",
        b: float = 1.0,
        target_viewpoint: Optional[str] = None,
        verbosity: int = 1,
    ):
        """Initialize multi-model orchestrator.

        Args:
            model_instances: List of GraphIDYOMModel or ModelInstance objects.
            model_merge: "arith" (default) or "geom" for merging models.
            weight_mode: "inverse_power" or "one_minus" for entropy weighting.
            b: Power parameter for inverse_power mode.
            target_viewpoint: Common target viewpoint for all model outputs.
            verbosity: Logging verbosity level.
        """
        self.model_instances: List[ModelInstance] = []
        for inst in model_instances:
            if isinstance(inst, ModelInstance):
                self.model_instances.append(inst)
            elif isinstance(inst, GraphIDYOMModel):
                # Auto-generate name from codec if available
                codec = inst.codec
                codec_name = getattr(codec, "cfg", None)
                viewpoint_str = str(codec_name) if codec_name else "model"
                name = f"model_{len(self.model_instances)}_{viewpoint_str}"
                self.model_instances.append(ModelInstance(name=name, model=inst))
            else:
                raise TypeError(f"Expected GraphIDYOMModel or ModelInstance, got {type(inst)}")

        if not self.model_instances:
            raise ValueError("model_instances must contain at least one model")

        self.model_merge_mode = str(model_merge).lower().strip()
        if self.model_merge_mode not in ("arith", "geom", "product"):
            raise ValueError(
                f"model_merge must be 'arith', 'geom', or 'product', got {self.model_merge_mode!r}"
            )

        self.weight_mode = str(weight_mode)
        self.b = float(b)
        self.target_viewpoint = target_viewpoint
        self.verbosity = int(verbosity)

        # Create merge strategy for combining model instances
        # (Default to arithmetic as per framework design)
        if self.model_merge_mode == "arith":
            self._model_merge = EntropyArithmeticMerge(
                alphabet_size=None,
                weight_mode=self.weight_mode,
                b=self.b,
            )
        elif self.model_merge_mode == "geom":
            self._model_merge = EntropyGeometricMerge(
                alphabet_size=None,
                weight_mode=self.weight_mode,
                b=self.b,
            )
        else:
            # Product is only meaningful for joint prediction across distinct targets.
            # For single-target merges we fall back to arithmetic (IDyOM-style).
            self._model_merge = EntropyArithmeticMerge(
                alphabet_size=None,
                weight_mode=self.weight_mode,
                b=self.b,
            )

    def _can_use_product_merge(self, instances: Sequence[ModelInstance]) -> bool:
        """True iff product merge is valid for this instance set.

        Product merge is only used when all models expose explicit targets and
        there are at least two distinct target viewpoints to predict jointly.
        """
        if self.model_merge_mode != "product":
            return False

        if not instances:
            return False

        targets: List[str] = []
        for inst in instances:
            tv = inst.model.target_viewpoint
            if tv is None:
                return False
            targets.append(str(tv))

        return len(set(targets)) >= 2

    def fit_all(self, folder: str, *, export_graphml: bool = True) -> None:
        """Train LTM for all model instances from a dataset folder."""
        if self.verbosity >= 1:
            print(f"Training {len(self.model_instances)} models from folder: {folder}")

        for inst in self.model_instances:
            if self.verbosity >= 1:
                print(f"  Training model: {inst.name}")
            inst.model.fit_folder(folder, export_graphml=export_graphml)

    def save_all_ltm(self, save_dir: str = None, *, dataset_name: str = None,
                     manager: PretrainedModelsManager = None) -> None:
        """Save all trained LTM models to disk.
        
        Two modes:
        1. Explicit path: save_all_ltm("/path/to/dir")
           Each model saved as: /path/to/dir/{model_name}/...
           Multi-model config: /path/to/dir/multi_model_config.json
        
        2. Managed structure: save_all_ltm(dataset_name="elsass")
           Creates organized structure for each model in dataset subdirectory
        
        Args:
            save_dir: Direct path for models
            dataset_name: Dataset name for managed structure (applies to all models)
            manager: PretrainedModelsManager instance
        
        Structure (managed mode):
            pretrained_models/datasets/{dataset_name}/{model_name}_augmented_*/
                graphs/, alphabet.json, target_alphabet.json, metadata.json
            multi_model_config.json (at dataset level)
        """
        if save_dir:
            # Explicit path mode
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save each model's LTM
            for inst in self.model_instances:
                model_dir = save_path / inst.name
                inst.model.save_ltm(str(model_dir))
        
        elif dataset_name:
            # Managed structure mode
            if manager is None:
                manager = PretrainedModelsManager()
            
            # Get base dataset directory
            dataset_dir = manager.datasets_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each model using its own config
            for inst in self.model_instances:
                # Use the model's graph_build_config to determine augmentation status
                augmented = inst.model.graph_build_config.augment
                
                config = ModelConfig(
                    dataset_name=dataset_name,
                    source_viewpoint=inst.name,  # Use instance name as viewpoint identifier
                    augmented=augmented,
                    target_viewpoint=self.target_viewpoint,
                )
                model_dir = manager.get_model_dir(config)
                inst.model.save_ltm(str(model_dir))
        else:
            raise ValueError(
                "Must provide either save_dir or dataset_name"
            )
        
        # Save multi-model configuration
        if save_dir:
            config_file = Path(save_dir) / "multi_model_config.json"
        else:
            config_file = dataset_dir / "multi_model_config.json"
        
        config = {
            "model_names": [inst.name for inst in self.model_instances],
            "model_merge": self.model_merge_mode,
            "weight_mode": self.weight_mode,
            "b": self.b,
            "target_viewpoint": self.target_viewpoint,
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        if self.verbosity >= 1:
            print(f"[+] Saved all {len(self.model_instances)} models")
            if save_dir:
                print(f"    Location: {save_dir}")
            else:
                print(f"    Location: {dataset_dir}")

    def load_all_ltm(self, load_dir: str = None, *, dataset_name: str = None,
                     manager: PretrainedModelsManager = None) -> None:
        """Load all pretrained LTM models from disk.
        
        Two modes:
        1. Explicit path: load_all_ltm("/path/to/dir")
           Loads from: /path/to/dir/{model_name}/...
        
        2. Managed structure: load_all_ltm(dataset_name="elsass")
           Loads from organized structure
        
        Args:
            load_dir: Direct path for models
            dataset_name: Dataset name for managed structure
            manager: PretrainedModelsManager instance
        
        Validates:
            - Number of models matches
            - Model names match
            - Configuration matches
        """
        if load_dir:
            # Explicit path mode
            load_path = Path(load_dir)
        elif dataset_name:
            # Managed structure mode
            if manager is None:
                manager = PretrainedModelsManager()
            load_path = manager.datasets_dir / dataset_name
        else:
            raise ValueError("Must provide either load_dir or dataset_name")
        
        if not load_path.exists():
            raise FileNotFoundError(f"Load directory not found: {load_path}")
        
        # Load multi-model configuration
        config_file = load_path / "multi_model_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Multi-model config not found: {config_file}")
        
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Validate configuration
        saved_names = config["model_names"]
        current_names = [inst.name for inst in self.model_instances]
        
        if len(saved_names) != len(current_names):
            raise ValueError(
                f"Model count mismatch: saved={len(saved_names)}, current={len(current_names)}"
            )
        
        if saved_names != current_names:
            raise ValueError(
                f"Model names mismatch: saved={saved_names}, current={current_names}"
            )
        
        if config.get("target_viewpoint") != self.target_viewpoint:
            raise ValueError(
                f"Target viewpoint mismatch: saved={config.get('target_viewpoint')}, "
                f"current={self.target_viewpoint}"
            )
        
        # Load each model's LTM
        for inst in self.model_instances:
            # Construct full directory name with augmentation status (used for managed mode)
            augmented = inst.model.graph_build_config.augment
            if dataset_name:
                # Managed structure: use full name with augmentation status
                model_subdir = f"{inst.name}_augmented_{str(augmented).lower()}"
                model_dir = load_path / model_subdir
            else:
                # Explicit path mode: just use instance name
                model_dir = load_path / inst.name
            inst.model.load_ltm(str(model_dir))
        
        if self.verbosity >= 1:
            print(f"[+] Loaded all {len(self.model_instances)} models from {load_path}")

    def reset_stm_all(self) -> None:
        """Reset STM for all model instances."""
        for inst in self.model_instances:
            if inst.model.use_stm:
                inst.model.reset_stm()

    @staticmethod
    def encode_joint_symbol(target_values: Mapping[str, str]) -> Symbol:
        """Encode a joint target assignment as a deterministic JSON symbol."""
        return json.dumps({str(k): str(v) for k, v in sorted(target_values.items())}, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def decode_joint_symbol(symbol: Symbol) -> Dict[str, str]:
        """Decode a joint target assignment symbol."""
        obj = json.loads(symbol)
        if not isinstance(obj, dict):
            raise ValueError(f"Joint symbol is not a JSON object: {symbol!r}")
        return {str(k): str(v) for k, v in obj.items()}

    @staticmethod
    def _top_k_normalized(dist: Dist, k: Optional[int]) -> Dist:
        if not dist:
            return {}
        if k is None or int(k) <= 0 or len(dist) <= int(k):
            return dict(dist)
        top = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[: int(k)]
        z = float(sum(float(v) for _, v in top))
        if z <= 0.0:
            return {}
        return {str(s): (float(p) / z) for s, p in top}

    def _predict_combined_dist_for_instances(
        self,
        instances: Sequence[ModelInstance],
        history: Sequence[NoteInfo],
        *,
        short_term_only: bool = False,
        long_term_only: bool = False,
    ) -> Dist:
        if long_term_only and short_term_only:
            raise ValueError("Cannot set both short_term_only and long_term_only")

        if self._can_use_product_merge(instances):
            per_model_dists: List[Dist] = []
            for inst in instances:
                d = inst.model.predict_next_dist(
                    history,
                    short_term_only=short_term_only,
                    long_term_only=long_term_only,
                )
                if d:
                    per_model_dists.append(d)
            merged = self._product_merge_dists(per_model_dists)
            if per_model_dists and not merged:
                raise ValueError(
                    "Product merge produced incompatible symbols for next-distribution. "
                    "Use predict_next_joint_dist() for multi-target product prediction."
                )
            return merged

        ltm_dists: List[Dist] = []
        stm_dists: List[Dist] = []

        for inst in instances:
            if not short_term_only:
                ltm_dist = inst.model.predict_next_dist(history, long_term_only=True)
                if ltm_dist:
                    ltm_dists.append(ltm_dist)

            if not long_term_only:
                stm_dist = inst.model.predict_next_dist(history, short_term_only=True)
                if stm_dist:
                    stm_dists.append(stm_dist)

        if short_term_only:
            combined_ltm = {}
        elif ltm_dists:
            combined_ltm = self._merge_layer_entropy_weighted(ltm_dists, "arithmetic")
        else:
            combined_ltm = {}

        if long_term_only:
            combined_stm = {}
        elif stm_dists:
            combined_stm = self._merge_layer_entropy_weighted(stm_dists, "arithmetic")
        else:
            combined_stm = {}

        if not combined_ltm:
            return combined_stm
        if not combined_stm:
            return combined_ltm

        return self._model_merge.merge([combined_ltm, combined_stm])

    def predict_next_dist(
        self,
        history: Sequence[NoteInfo],
        *,
        short_term_only: bool = False,
        long_term_only: bool = False,
    ) -> Dist:
        """Predict combined distribution from all models.

        Uses two-layer merging:
        1. All LTMs -> combined_ltm (entropy-weighted arithmetic)
        2. All STMs -> combined_stm (entropy-weighted arithmetic)
        3. combined_ltm + combined_stm -> final using entropy weights
        """

        return self._predict_combined_dist_for_instances(
            self.model_instances,
            history,
            short_term_only=short_term_only,
            long_term_only=long_term_only,
        )

    def predict_next_joint_dist(
        self,
        history: Sequence[NoteInfo],
        *,
        short_term_only: bool = False,
        long_term_only: bool = False,
        max_symbols_per_target: Optional[int] = None,
    ) -> Dist:
        """Predict a joint distribution over multiple target viewpoints.

        Models are grouped by each model's `target_viewpoint`. Each group is merged
        internally using the configured multi-model strategy. The final joint
        distribution is the independent product across target groups.
        """
        by_target: Dict[str, List[ModelInstance]] = defaultdict(list)
        for inst in self.model_instances:
            tv = inst.model.target_viewpoint
            if tv is None:
                raise ValueError(
                    f"Model {inst.name!r} has target_viewpoint=None. "
                    "Joint target prediction requires explicit target viewpoints."
                )
            by_target[str(tv)].append(inst)

        if not by_target:
            return {}

        per_target: Dict[str, Dist] = {}
        for target_name, instances in by_target.items():
            d_target = self._predict_combined_dist_for_instances(
                instances,
                history,
                short_term_only=short_term_only,
                long_term_only=long_term_only,
            )
            d_target = self._top_k_normalized(d_target, max_symbols_per_target)
            if d_target:
                per_target[target_name] = d_target

        if not per_target:
            return {}

        targets = sorted(per_target.keys())
        joint_states: Dict[Tuple[Tuple[str, str], ...], float] = {tuple(): 1.0}

        for t in targets:
            d = per_target[t]
            if not d:
                continue
            nxt: Dict[Tuple[Tuple[str, str], ...], float] = {}
            for key, p_key in joint_states.items():
                base_map = dict(key)
                for sym, p_sym in d.items():
                    m = dict(base_map)
                    m[str(t)] = str(sym)
                    new_key = tuple(sorted((str(k), str(v)) for k, v in m.items()))
                    nxt[new_key] = nxt.get(new_key, 0.0) + (float(p_key) * float(p_sym))
            joint_states = nxt
            if not joint_states:
                break

        if not joint_states:
            return {}

        out: Dict[Symbol, float] = {}
        for key, p in joint_states.items():
            symbol = self.encode_joint_symbol({k: v for k, v in key})
            out[symbol] = float(p)

        z = float(sum(out.values()))
        if z <= 0.0:
            return {}
        return {s: (p / z) for s, p in out.items()}

    def process_file(
        self,
        path: str,
        *,
        reset_stm: bool = True,
        short_term_only: bool = False,
        long_term_only: bool = False,
        prob_floor: float = 1e-15,
        return_trace: bool = False,
        trace_config: Optional[TraceConfig] = None,
    ) -> Union[ProcessedSequence, Tuple[ProcessedSequence, Trace]]:
        """Process a file with all models, combining predictions at each step.

        Returns:
            ProcessedSequence with combined per-step probabilities/surprisals.
            If return_trace=True, also returns a Trace with diagnostics.
        """

        if reset_stm:
            self.reset_stm_all()

        cfg = trace_config or TraceConfig()

        # Use first model's parser to parse the file
        first_model = self.model_instances[0].model
        seq = first_model.parser.parse_file(path)

        steps: List[StepPrediction] = []
        trace_steps: List[StepTrace] = []
        history: List[NoteInfo] = []

        for i in range(len(seq)):
            # Get observed symbol from first model (all should agree on encoding)
            observed = first_model.codec.symbol_at_index(seq, i)
            current_note = seq[i]

            if self._can_use_product_merge(self.model_instances):
                per_model_dists: List[Dist] = []
                p_joint = 1.0
                active_views = 0

                for inst in self.model_instances:
                    dist_i = inst.model.predict_next_dist(
                        history,
                        short_term_only=short_term_only,
                        long_term_only=long_term_only,
                    )
                    if dist_i:
                        per_model_dists.append(dist_i)

                    observed_i = inst.model.codec.symbol_at_index(seq, i)
                    if observed_i is None:
                        continue

                    if inst.model.target_viewpoint is not None:
                        projected_symbols = inst.model._project_symbol_to_target(
                            observed_i, history, current_note
                        )
                        p_i = sum(dist_i.get(sym, 0.0) for sym in projected_symbols)
                    else:
                        p_i = float(dist_i.get(observed_i, 0.0))

                    if p_i <= 0.0:
                        p_i = max(float(inst.model._fallback_prob()), float(prob_floor))
                    p_joint *= float(p_i)
                    active_views += 1

                if active_views == 0:
                    p_joint = 1.0
                p_joint = max(float(p_joint), float(prob_floor))
                surprisal = -math.log(p_joint, 2)
                dist_final = self._product_merge_dists(per_model_dists)

                steps.append(
                    StepPrediction(index=i, observed=observed, prob=float(p_joint), surprisal=float(surprisal))
                )

                if return_trace:
                    st = StepTrace(
                        index=i,
                        observed=observed,
                        ltm=None,
                        stm=None,
                        final_entropy=(float(dist_entropy_bits(dist_final)) if dist_final else None),
                        final_rel_entropy=(float(dist_relative_entropy(dist_final)) if dist_final else None),
                        final_p_obs=float(p_joint) if observed is not None else None,
                        final_surprisal_obs=float(surprisal) if observed is not None else None,
                        final_dist=(dict(dist_final) if (cfg.store_full_dists and dist_final) else None),
                    )
                    trace_steps.append(st)

                history.append(current_note)
                for inst in self.model_instances:
                    if inst.model.use_stm and inst.model._stm_state is not None:
                        inst.model._stm_update_with_note(current_note)
                continue

            # Collect per-model traces and distributions
            model_traces: Dict[str, ModelTrace] = {}
            ltm_dists_with_trace: List[Tuple[Dist, ModelTrace]] = []
            stm_dists_with_trace: List[Tuple[Dist, ModelTrace]] = []

            for inst in self.model_instances:
                # Get LTM dist with trace
                if inst.model.ltm_graphs:
                    ltm_dist, ltm_trace = inst.model._predict_with_model_graphs_trace(
                        name=f"{inst.name}_ltm",
                        graphs=inst.model.ltm_graphs,
                        history=history,
                        observed=observed,
                        current_note=current_note,
                        cfg=cfg,
                    )
                    if ltm_dist:
                        ltm_dists_with_trace.append((ltm_dist, ltm_trace))
                    model_traces[f"{inst.name}_ltm"] = ltm_trace

                # Get STM dist with trace
                if inst.model.use_stm and inst.model._stm_state:
                    stm_dist, stm_trace = inst.model._predict_with_model_graphs_trace(
                        name=f"{inst.name}_stm",
                        graphs=inst.model._stm_state.graphs,
                        history=history,
                        observed=observed,
                        current_note=current_note,
                        cfg=cfg,
                    )
                    if stm_dist:
                        stm_dists_with_trace.append((stm_dist, stm_trace))
                    model_traces[f"{inst.name}_stm"] = stm_trace

            # Layer 1: Merge all LTMs (entropy-weighted)
            ltm_dists = [d for d, _ in ltm_dists_with_trace]
            if ltm_dists:
                combined_ltm = self._merge_layer_entropy_weighted(ltm_dists, "arithmetic")
            else:
                combined_ltm = {}

            # Layer 1: Merge all STMs (entropy-weighted)
            stm_dists = [d for d, _ in stm_dists_with_trace]
            if stm_dists:
                combined_stm = self._merge_layer_entropy_weighted(stm_dists, "arithmetic")
            else:
                combined_stm = {}

            # Layer 2: Merge combined LTM and STM
            if combined_ltm and combined_stm:
                dist_final = self._model_merge.merge([combined_ltm, combined_stm])
            elif combined_ltm:
                dist_final = combined_ltm
            elif combined_stm:
                dist_final = combined_stm
            else:
                # Fallback: uniform distribution
                if first_model.alphabet:
                    a = float(len(first_model.alphabet))
                    dist_final = {s: 1.0 / a for s in first_model.alphabet}
                else:
                    dist_final = {}

            # Compute probability and surprisal for observed
            # Project source observed symbol to target space if target_viewpoint is set
            if self.target_viewpoint is not None and observed is not None:
                projected_symbols = first_model._project_symbol_to_target(observed, history, current_note)
                p = sum(dist_final.get(sym, 0.0) for sym in projected_symbols)
            else:
                p = float(dist_final.get(observed, 0.0)) if observed is not None else 0.0
            if p <= 0.0:
                p = max(1e-10, prob_floor)
            p = max(p, float(prob_floor))
            surprisal = -math.log(p, 2)

            steps.append(StepPrediction(index=i, observed=observed, prob=p, surprisal=surprisal))

            if return_trace:
                # Store per-model traces as extra info for multi-model analysis
                extra_traces = {}
                for model_name, trace in model_traces.items():
                    extra_traces[model_name] = trace.to_dict(cfg=cfg) if trace else None

                st = StepTrace(
                    index=i,
                    observed=observed,
                    ltm=model_traces.get(f"{self.model_instances[0].name}_ltm"),
                    stm=model_traces.get(f"{self.model_instances[0].name}_stm"),
                    final_entropy=(float(dist_entropy_bits(dist_final)) if dist_final else None),
                    final_rel_entropy=(float(dist_relative_entropy(dist_final)) if dist_final else None),
                    final_p_obs=float(p) if observed is not None else None,
                    final_surprisal_obs=float(surprisal) if observed is not None else None,
                    final_dist=(dict(dist_final) if (cfg.store_full_dists and dist_final) else None),
                )
                # Store additional per-model traces
                if not hasattr(st, 'extra'):
                    st.extra = {}
                st.extra['per_model_traces'] = extra_traces
                
                trace_steps.append(st)

            # Update history and all model STMs
            history.append(current_note)
            for inst in self.model_instances:
                if inst.model.use_stm and inst.model._stm_state is not None:
                    inst.model._stm_update_with_note(current_note)

        # Use first model's alphabet for ProcessedSequence
        processed = ProcessedSequence(steps=tuple(steps), alphabet=first_model.alphabet)

        if not return_trace:
            return processed

        return processed, Trace(steps=tuple(trace_steps), cfg=cfg)

    def _product_merge_dists(self, dists: Sequence[Dist]) -> Dist:
        """Independent product combination over same target symbol space."""
        dists = [d for d in dists if d]
        if not dists:
            return {}

        keyset = set(dists[0].keys())
        for d in dists[1:]:
            if set(d.keys()) != keyset:
                return {}
        if not keyset:
            return {}

        out: Dict[Symbol, float] = {}
        for s in keyset:
            p = 1.0
            for d in dists:
                p *= float(d.get(s, 0.0))
            out[s] = float(p)

        z = float(sum(out.values()))
        if z <= 0.0:
            return {}
        return {k: (v / z) for k, v in out.items()}

    def _merge_layer_entropy_weighted(self, dists: Sequence[Dist], mode: str = "arithmetic") -> Dist:
        """Merge a layer of distributions using entropy-weighted merging.

        This is used for Layer 1 merging to respect the confidence (entropy) of each model's
        projected distributions.

        Args:
            dists: Sequence of distributions to merge.
            mode: "arithmetic" (default) or "geometric".

        Returns:
            Merged distribution with entropy-based weighting.
        """
        if not dists:
            return {}

        dists = [d for d in dists if d]
        if not dists:
            return {}

        if len(dists) == 1:
            return dict(dists[0])

        # Compute entropy-weighted merging (same as layer 2)
        ws = list(
            entropy_weights(
                dists,
                alphabet_size=None,
                mode=self.weight_mode,
                b=self.b,
            )
        )

        if sum(ws) <= 0.0:
            ws = [1.0 / float(len(dists))] * len(dists)

        if mode == "arithmetic":
            from merge import weighted_arithmetic_mean
            return weighted_arithmetic_mean(dists, weights=ws)
        else:
            from merge import weighted_geometric_mean
            return weighted_geometric_mean(dists, weights=ws)

    def _merge_layer(self, dists: Sequence[Dist], mode: str = "arithmetic") -> Dist:
        """Merge a layer of distributions using uniform weighting (deprecated).

        Use _merge_layer_entropy_weighted for entropy-based merging.

        Args:
            dists: Sequence of distributions to merge.
            mode: "arithmetic" (default) or "geometric".

        Returns:
            Merged distribution.
        """
        if not dists:
            return {}

        dists = [d for d in dists if d]
        if not dists:
            return {}

        if len(dists) == 1:
            return dict(dists[0])

        # Use arithmetic merging with uniform weights
        weights = [1.0 / float(len(dists))] * len(dists)
        
        if mode == "arithmetic":
            from merge import weighted_arithmetic_mean
            return weighted_arithmetic_mean(dists, weights=weights)
        else:
            from merge import weighted_geometric_mean
            return weighted_geometric_mean(dists, weights=weights)
