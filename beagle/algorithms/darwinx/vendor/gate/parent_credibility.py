"""Parent-credibility weighting for the self_evolve archive.

When ``atelier.predictions`` records per-node credibility (jaccard
accuracy of the predicted_impact vs the actual next-iter flips), we
can use that to penalize lineages whose proposer chronically
over-predicts: lower a node's parent-sampling probability when its
own predictions were wrong.

This module is the read-side of the predictions system. It's
intentionally a thin layer:

1. ``load_node_credibilities(campaign)`` scans the campaign's
   ``atelier/predictions/`` directory and returns a list of
   ``PredictionCredibility`` records keyed by ``child_node_id``.
2. ``credibility_weight(credibility)`` maps a single record to a
   parent-sampling weight in ``[floor, 1.0]``.
3. ``weights_for_campaign(campaign)`` returns a ``{node_id: weight}``
   dict the parent picker can consume.

The picker (``self_evolve.parent_selection.MixedHighScoreStrategy``)
opts in via env var; default behavior is unchanged.

Weight formula (chosen for clarity over sophistication):

  weight = max(floor, jaccard_accuracy)            when has_credibility
  weight = empty_penalty                            when is_empty_prediction
  weight = 1.0                                      when no record (root, fresh node)

This means:
- A node with a perfect prediction (jaccard=1.0) keeps full weight.
- A node that over-predicted (jaccard=0.3) gets weight 0.3 → its
  effective parent score is 0.3× its raw score.
- A node that abdicated falsifiability (empty prediction) gets a flat
  ``empty_penalty`` (default 0.5 — half-weight, equivalent to a 50 %
  over-predictor).
- New campaigns start neutral (weight=1.0) since no credibility data
  exists yet.

The picker uses ``score * weight`` as the sort key. With our typical
score range [0.4, 0.8] on smoke-10 and weight range [floor, 1.0],
this naturally favors lineages with both high scores AND honest
predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .predictions import PredictionCredibility, load_credibility


__all__ = [
    "DEFAULT_FLOOR",
    "DEFAULT_EMPTY_PENALTY",
    "credibility_weight",
    "load_node_credibilities",
    "weights_for_campaign",
]


logger = logging.getLogger("atelier.parent_credibility")


DEFAULT_FLOOR = 0.1
"""Minimum weight for a chronic over-predictor. We never zero out a
lineage entirely — a strict zero would mean a single bad iteration
permanently removes the node from the archive, which is too brittle
for early campaigns where the proposer is still calibrating."""


DEFAULT_EMPTY_PENALTY = 0.5
"""Weight applied to nodes whose proposer emitted an EMPTY
predicted_impact block. Half-weight = roughly equivalent to a
50 %-over-predictor. Tuned to discourage abdication without killing
exploration."""


def credibility_weight(
    credibility: PredictionCredibility | None,
    *,
    floor: float = DEFAULT_FLOOR,
    empty_penalty: float = DEFAULT_EMPTY_PENALTY,
) -> float:
    """Map one credibility record to a parent-sampling weight.

    ``None`` → 1.0 (no data → don't penalize).
    Empty prediction → ``empty_penalty``.
    Otherwise → max(floor, jaccard_accuracy).
    """
    if credibility is None:
        return 1.0
    if credibility.is_empty_prediction:
        return empty_penalty
    return max(floor, credibility.jaccard_accuracy)


def load_node_credibilities(
    *, reports_root: Path | str, campaign: str
) -> dict[str, PredictionCredibility]:
    """Scan the campaign's predictions directory for all credibility
    sidecars and return them as a ``{child_node_id: credibility}`` dict.

    Missing directory → empty dict (the campaign has no L5 data yet).
    """
    pred_dir = (
        Path(reports_root) / campaign / "atelier" / "predictions"
    )
    if not pred_dir.is_dir():
        return {}

    out: dict[str, PredictionCredibility] = {}
    for path in pred_dir.glob("*.credibility.json"):
        node_id = path.name[: -len(".credibility.json")]
        cred = load_credibility(
            reports_root=reports_root,
            campaign=campaign,
            child_node_id=node_id,
        )
        if cred is not None:
            out[node_id] = cred
    return out


def weights_for_campaign(
    *,
    reports_root: Path | str,
    campaign: str,
    floor: float = DEFAULT_FLOOR,
    empty_penalty: float = DEFAULT_EMPTY_PENALTY,
) -> dict[str, float]:
    """Compute a ``{node_id: parent_sampling_weight}`` for every node
    that has a credibility record. Nodes without a record are absent
    from the dict (the picker treats absence as weight=1.0).

    Reads at most one file per node (no LLM calls, no I/O beyond
    JSON parses), so safe to call on every parent-selection cycle.
    """
    creds = load_node_credibilities(
        reports_root=reports_root, campaign=campaign
    )
    return {
        node_id: credibility_weight(
            c, floor=floor, empty_penalty=empty_penalty
        )
        for node_id, c in creds.items()
    }
