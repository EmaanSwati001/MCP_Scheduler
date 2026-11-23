from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, NamedTuple, Sequence

from ..models import Job


class Sample(NamedTuple):
    """Cluster snapshot taken after each scheduling round. The values hold
    until the next sample (piecewise-constant)."""

    t: float
    used_gpus: int
    lb_index: float
    queue_len: int
    running: int


def percentile(values: Sequence[float], q: float) -> float:
    """q in [0, 100], linear interpolation (same as numpy's default)."""
    if not values:
        return 0.0
    data = sorted(values)
    pos = (len(data) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def _integrate(
    timeline: Sequence[Sample], value: Callable[[Sample], float], until: float
) -> float:
    total = 0.0
    for cur, nxt in zip(timeline, timeline[1:]):
        if cur.t >= until:
            break
        total += value(cur) * (min(nxt.t, until) - cur.t)
    return total


def summarize(
    finished: Sequence[Job],
    timeline: Sequence[Sample],
    total_capacity: int,
    rejected: int = 0,
    unscheduled: int = 0,
) -> Dict[str, object]:
    """Common metrics for one run. All times are in trace time units."""
    waits = [j.wait_time for j in finished]
    by_size: Dict[int, List[float]] = defaultdict(list)
    for j in finished:
        by_size[j.gpus].append(j.wait_time)

    t0 = timeline[0].t if timeline else 0.0
    t_end = timeline[-1].t if timeline else 0.0
    horizon = t_end - t0

    def time_avg(value, until=t_end):
        span = until - t0
        if span <= 0:
            return value(timeline[0]) if timeline else 0.0
        return _integrate(timeline, value, until) / span

    # Paper Table I averages unallocated GPUs only while jobs are still being
    # allocated, i.e. until the last job has started.
    last_start = max((j.start_time for j in finished), default=t0)

    return {
        "jobs_completed": len(finished),
        "jobs_rejected": rejected,
        "jobs_unscheduled": unscheduled,
        "avg_wait": sum(waits) / len(waits) if waits else 0.0,
        "p95_wait": percentile(waits, 95),
        "p99_wait": percentile(waits, 99),
        "max_wait": max(waits, default=0.0),
        "avg_wait_by_gpus": {
            g: sum(w) / len(w) for g, w in sorted(by_size.items())
        },
        "gpu_utilization": time_avg(lambda s: s.used_gpus / total_capacity),
        "avg_lb_index": time_avg(lambda s: s.lb_index),
        "avg_unallocated_ratio_during_allocation": time_avg(
            lambda s: 1 - s.used_gpus / total_capacity, last_start
        ),
        "makespan": horizon,
        "throughput": len(finished) / horizon if horizon > 0 else 0.0,
    }
