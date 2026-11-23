from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, NamedTuple, Sequence

from ..models import ClusterState, Job


class Placement(NamedTuple):
    """Decision: start `job_id` now on server (`cluster_id`, `server_id`)."""

    job_id: int
    cluster_id: int
    server_id: int


class Scheduler(ABC):
    """Interface every policy implements. See README for the full contract.

    The simulator owns the clock and applies allocations. A scheduler only
    reads the queue and the state and returns the placements it wants.
    """

    name = "base"

    def setup(self, jobs: Sequence[Job], state: ClusterState) -> None:
        """Called once before the run with the whole workload and the empty
        topology. Use it for trace-wide statistics such as the global demand
        popularity p_d. Default: nothing."""

    @abstractmethod
    def schedule(
        self, queue: List[Job], state: ClusterState, now: float
    ) -> List[Placement]:
        """Called after every batch of arrivals/completions while jobs wait.

        queue: waiting jobs in arrival order (a new list each call; do not
               modify the Job objects).
        state: live cluster state, read-only. Plan on `state.clone()`.
        Returns placements to start at `now`, applied in list order. Every
        placement must be feasible when applied or the simulator raises.
        Jobs not returned keep waiting.
        """
