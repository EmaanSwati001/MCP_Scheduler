"""Fairness and tail metrics used by E4 and E7 (the core summary has avg/P95/P99/max)."""
from __future__ import annotations

from typing import Dict, Sequence

from ..models import Job


def fairness_metrics(
    finished: Sequence[Job], w_ref: float, multiples: Sequence[float] = (1, 3)
) -> Dict[str, float]:
    """Starvation counts and per-size waits.

    A job is "starved at m" when wait > m * w_ref. w_ref must be the same
    reference for every scheduler being compared (the trace's mean duration).
    """
    waits = [j.wait_time for j in finished]
    n = len(waits) or 1
    out: Dict[str, float] = {}
    for m in multiples:
        c = sum(1 for w in waits if w > m * w_ref)
        out[f"starved_{m:g}x"] = c
        out[f"starved_{m:g}x_frac"] = c / n
    eight = [j.wait_time for j in finished if j.gpus >= 8]
    small = [j.wait_time for j in finished if j.gpus < 8]
    out["avg_wait_8gpu"] = sum(eight) / len(eight) if eight else 0.0
    out["avg_wait_small"] = sum(small) / len(small) if small else 0.0
    return out
