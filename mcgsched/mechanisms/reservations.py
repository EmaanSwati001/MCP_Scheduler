"""Drain reservation (extension N1).

A protected job that cannot start picks one server and reserves it. No new job is
placed there, running jobs finish normally (no preemption), and when enough GPUs
are free the protected job starts. No runtime estimates are used.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple

from ..models import ClusterState, Job

Key = Tuple[int, int]  # (cluster_id, server_id)


def completions_needed(free: int, running: Dict[int, int], gpus: int) -> int:
    """Fewest running jobs that must finish before `gpus` are free on a server.

    Without runtime knowledge the best case is that the largest jobs finish
    first, so take running jobs from the biggest down.
    """
    if free >= gpus:
        return 0
    need, count = gpus - free, 0
    for g in sorted(running.values(), reverse=True):
        need -= g
        count += 1
        if need <= 0:
            return count
    return 10 ** 9  # capacity too small


def choose_drain_server(job: Job, state: ClusterState, taken: Iterable[Key]) -> Optional[Key]:
    """Server whose capacity can hold `job` and needs the fewest completions.

    Ties: more GPUs already free, then smaller capacity (keep big servers for
    big jobs), then id order. Servers already reserved are skipped.
    """
    taken = set(taken)
    best, best_rank = None, None
    for s in state.servers():
        if s.key in taken or s.capacity < job.gpus:
            continue
        rank = (completions_needed(s.free, s.running, job.gpus), -s.free, s.capacity, s.key)
        if best_rank is None or rank < best_rank:
            best, best_rank = s.key, rank
    return best


class ReservationTable:
    """job_id -> reserved server. At most one reservation per server."""

    def __init__(self) -> None:
        self.by_job: Dict[int, Key] = {}

    def __len__(self) -> int:
        return len(self.by_job)

    def servers(self) -> Set[Key]:
        return set(self.by_job.values())

    def others(self, job_id: int) -> Set[Key]:
        return {k for j, k in self.by_job.items() if j != job_id}

    def add(self, job_id: int, key: Key) -> None:
        if key in self.servers():
            raise ValueError(f"server {key} is already reserved")
        self.by_job[job_id] = key

    def drop(self, job_id: int) -> Optional[Key]:
        return self.by_job.pop(job_id, None)

    def keep_only(self, waiting_ids: Iterable[int]) -> None:
        live = set(waiting_ids)
        self.by_job = {j: k for j, k in self.by_job.items() if j in live}
