from __future__ import annotations

import csv
import random
from dataclasses import replace
from typing import Dict, List, Sequence

from ..models import Job

# Placeholder mix for development only; not measured from any trace.
DEFAULT_DEMAND: Dict[int, float] = {1: 0.60, 2: 0.20, 4: 0.12, 8: 0.08}

CSV_FIELDS = ["job_id", "arrival_time", "gpus", "duration"]


def synthetic_jobs(
    n: int,
    seed: int = 0,
    mean_interarrival: float = 60.0,
    mean_duration: float = 3600.0,
    demand: Dict[int, float] = DEFAULT_DEMAND,
) -> List[Job]:
    """Poisson arrivals, exponential durations, GPU demand drawn from `demand`."""
    rng = random.Random(seed)
    sizes, weights = zip(*sorted(demand.items()))
    jobs: List[Job] = []
    t = 0.0
    for i in range(n):
        jobs.append(
            Job(
                job_id=i,
                arrival_time=t,
                gpus=rng.choices(sizes, weights)[0],
                duration=rng.expovariate(1.0 / mean_duration),
            )
        )
        t += rng.expovariate(1.0 / mean_interarrival)
    return jobs


def compress_arrivals(jobs: Sequence[Job], eta: float) -> List[Job]:
    """Arrival-rate compression: t' = t0 + (t - t0) / eta, t0 = first arrival.

    eta > 1 makes jobs arrive eta times more densely. Durations are unchanged.
    """
    if eta <= 0:
        raise ValueError("eta must be positive")
    if not jobs:
        return []
    t0 = min(j.arrival_time for j in jobs)
    return [replace(j, arrival_time=t0 + (j.arrival_time - t0) / eta) for j in jobs]


def make_burst(jobs: Sequence[Job], at: float = 0.0) -> List[Job]:
    """All jobs arrive together (paper's 800-job burst test). Queue order is
    then by job_id."""
    return [replace(j, arrival_time=at) for j in jobs]


def save_jobs_csv(jobs: Sequence[Job], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_FIELDS)
        for j in jobs:
            writer.writerow([j.job_id, repr(j.arrival_time), j.gpus, repr(j.duration)])


def load_jobs_csv(path: str) -> List[Job]:
    """Read the common job schema. Trace loaders (Helios, Philly) should
    write this format so every experiment reads the same files."""
    with open(path, newline="", encoding="utf-8") as fh:
        jobs = [
            Job(
                job_id=int(row["job_id"]),
                arrival_time=float(row["arrival_time"]),
                gpus=int(row["gpus"]),
                duration=float(row["duration"]),
            )
            for row in csv.DictReader(fh)
        ]
    jobs.sort(key=lambda j: (j.arrival_time, j.job_id))
    return jobs
