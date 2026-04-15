"""product_model.py

Independent multi-viewpoint product model (IDyOM-style across viewpoints).

This module combines multiple GraphIDYOMModel instances by multiplying the
per-viewpoint observed-event probabilities at each event index:

    p_total(event_i) = Π_v p_v(observed_v_i | history_v_i)

This mirrors the independent product strategy used in the local IDyOMpy code
for multi-viewpoint likelihood computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from graph_types import Dist, MidiParser, NoteInfo, ProcessedSequence, StepPrediction
from model import GraphIDYOMModel


@dataclass
class ProductModelInstance:
    name: str
    model: GraphIDYOMModel


class IndependentViewpointProductModel:
    """Combine viewpoint models by independent probability product."""

    def __init__(
        self,
        model_instances: Sequence[Union[ProductModelInstance, GraphIDYOMModel]],
        *,
        parser: Optional[MidiParser] = None,
        verbosity: int = 1,
    ) -> None:
        instances: List[ProductModelInstance] = []
        for i, inst in enumerate(model_instances):
            if isinstance(inst, ProductModelInstance):
                instances.append(inst)
            elif isinstance(inst, GraphIDYOMModel):
                name = f"vp_{i}"
                if inst.viewpoint_spec is not None:
                    name = inst.viewpoint_spec.name
                instances.append(ProductModelInstance(name=name, model=inst))
            else:
                raise TypeError(f"Unsupported instance type: {type(inst)}")
        if not instances:
            raise ValueError("model_instances must not be empty")
        self.model_instances = instances
        self.parser = parser or instances[0].model.parser
        self.verbosity = int(verbosity)

    def reset_stm_all(self) -> None:
        for inst in self.model_instances:
            if inst.model.use_stm:
                inst.model.reset_stm()

    def process_file(
        self,
        path: str,
        *,
        reset_stm: bool = True,
        short_term_only: bool = False,
        long_term_only: bool = False,
        prob_floor: float = (1.0 / 30.0),
    ) -> ProcessedSequence:
        """Process a file by multiplying per-viewpoint probabilities."""

        if long_term_only and short_term_only:
            raise ValueError("Cannot set both short_term_only and long_term_only")

        if reset_stm:
            self.reset_stm_all()

        seq = self.parser.parse_file(path)
        history: List[NoteInfo] = []
        steps: List[StepPrediction] = []

        for i in range(len(seq)):
            p_total = 1.0
            for inst in self.model_instances:
                m = inst.model
                observed = m.codec.symbol_at_index(seq, i)
                dist = m.predict_next_dist(
                    history,
                    short_term_only=short_term_only,
                    long_term_only=long_term_only,
                )
                p = float(dist.get(observed, 0.0)) if observed is not None else 0.0
                if p <= 0.0:
                    p = max(float(m._fallback_prob()), float(prob_floor))
                p_total *= p

            p_total = max(float(p_total), float(prob_floor))
            surprisal = -math.log(p_total, 2)
            steps.append(
                StepPrediction(
                    index=i,
                    observed=None,
                    prob=float(p_total),
                    surprisal=float(surprisal),
                )
            )

            # Update shared history and all STMs.
            history.append(seq[i])
            for inst in self.model_instances:
                m = inst.model
                if m.use_stm and m._stm_state is not None:
                    m._stm_update_with_note(seq[i])

        # Expose alphabet from the first model as a convenience.
        return ProcessedSequence(steps=tuple(steps), alphabet=self.model_instances[0].model.alphabet)

    def predict_next_dist(
        self,
        history: Sequence[NoteInfo],
        *,
        short_term_only: bool = False,
        long_term_only: bool = False,
    ) -> Dist:
        """Return product distribution when models share the same target space.

        If symbol spaces differ, this raises ValueError.
        """
        dists = [
            inst.model.predict_next_dist(
                history,
                short_term_only=short_term_only,
                long_term_only=long_term_only,
            )
            for inst in self.model_instances
        ]
        dists = [d for d in dists if d]
        if not dists:
            return {}

        keys0 = set(dists[0].keys())
        for d in dists[1:]:
            if set(d.keys()) != keys0:
                raise ValueError(
                    "IndependentViewpointProductModel.predict_next_dist requires models "
                    "to share the same symbol space (same target viewpoint/alphabet)."
                )

        out: Dict[str, float] = {}
        for s in keys0:
            p = 1.0
            for d in dists:
                p *= float(d.get(s, 0.0))
            out[s] = p

        z = float(sum(out.values()))
        if z <= 0.0:
            return {}
        return {k: (v / z) for k, v in out.items()}
