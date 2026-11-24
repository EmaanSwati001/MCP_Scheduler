"""Comparison schedulers from the paper: Random, Consolidate (Best Fit), FGD, and a simplified MAST.

None of them waits for the head of the queue: a job that does not fit is
skipped and the next job is tried. Every policy plans on state.clone().
"""
from __future__ import annotations

import random
from typing import Dict, List

from ..models import ClusterState, Job
from .base import Placement, Scheduler


class RandomPlacement(Scheduler):
    """Random: each job goes to a random server that can host it. Seeded."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def setup(self, jobs, state: ClusterState) -> None:
        # Reset so every run with the same seed makes the same choices.
        self.rng = random.Random(self.seed)

    def schedule(self, queue: List[Job], state: ClusterState, now: float) -> List[Placement]:
        plan = state.clone()
        placements: List[Placement] = []
        for job in queue:
            options = list(plan.feasible_servers(job.gpus))
            if not options:
                continue
            server = self.rng.choice(options)
            plan.allocate(job.job_id, job.gpus, server.cluster_id, server.server_id)
            placements.append(Placement(job.job_id, server.cluster_id, server.server_id))
        return placements


class Consolidate(Scheduler):
    """Consolidate, also called Best Fit: put each job on the server with the
    fewest free GPUs that still fits, so big empty servers stay empty."""

    name = "consolidate"

    def schedule(self, queue: List[Job], state: ClusterState, now: float) -> List[Placement]:
        plan = state.clone()
        placements: List[Placement] = []
        for job in queue:
            best = None
            for server in plan.feasible_servers(job.gpus):
                if best is None or server.free < best.free:
                    best = server  # strict '<' keeps the first server on ties
            if best is None:
                continue
            plan.allocate(job.job_id, job.gpus, best.cluster_id, best.server_id)
            placements.append(Placement(job.job_id, best.cluster_id, best.server_id))
        return placements


class FGD(Scheduler):
    """FGD (fragmentation gradient descent, Weng et al.): place each job where
    the fragmentation measure rises least. Uses the global mix of GPU request
    sizes from the whole workload (paper Eq. 3 with no queue term, Eq. 5 and 10).

    F(r) = sum over sizes d > r of p_d * r   (r = free GPUs on a server)
    G    = F(free - gpus) - F(free)          (smaller is better)
    """

    name = "fgd"

    def __init__(self) -> None:
        self.popularity: Dict[int, float] = {}

    def setup(self, jobs, state: ClusterState) -> None:
        counts: Dict[int, int] = {}
        for job in jobs:
            counts[job.gpus] = counts.get(job.gpus, 0) + 1
        total = sum(counts.values()) or 1
        self.popularity = {d: c / total for d, c in counts.items()}

    def _F(self, r: int) -> float:
        return sum(p * r for d, p in self.popularity.items() if d > r)

    def schedule(self, queue: List[Job], state: ClusterState, now: float) -> List[Placement]:
        plan = state.clone()
        placements: List[Placement] = []
        for job in queue:
            best = None
            best_gradient = None
            for server in plan.feasible_servers(job.gpus):
                gradient = self._F(server.free - job.gpus) - self._F(server.free)
                if best_gradient is None or gradient < best_gradient:
                    best, best_gradient = server, gradient
            if best is None:
                continue
            plan.allocate(job.job_id, job.gpus, best.cluster_id, best.server_id)
            placements.append(Placement(job.job_id, best.cluster_id, best.server_id))
        return placements


class MASTLike(Scheduler):
    """Simplified MAST. Order the queue by priority = age / GPUs (older and
    smaller first), then put each job on the server with the most free GPUs
    that fits. The paper's MAST also uses tenant quotas; we have one tenant."""

    name = "mast"

    def schedule(self, queue: List[Job], state: ClusterState, now: float) -> List[Placement]:
        plan = state.clone()
        ordered = sorted(queue, key=lambda j: (-(now - j.arrival_time) / j.gpus, j.job_id))
        placements: List[Placement] = []
        for job in ordered:
            options = list(plan.feasible_servers(job.gpus))
            if not options:
                continue
            server = max(options, key=lambda s: s.free)  # first server on ties
            plan.allocate(job.job_id, job.gpus, server.cluster_id, server.server_id)
            placements.append(Placement(job.job_id, server.cluster_id, server.server_id))
        return placements