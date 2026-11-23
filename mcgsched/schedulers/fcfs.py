from __future__ import annotations

from typing import List

from ..models import ClusterState, Job
from .base import Placement, Scheduler


class FCFSFirstFit(Scheduler):
    """Strict arrival order, first feasible server.

    Stops at the first job that does not fit (head-of-line blocking), so no
    job ever overtakes an earlier one.
    """

    name = "fcfs"

    def schedule(
        self, queue: List[Job], state: ClusterState, now: float
    ) -> List[Placement]:
        plan = state.clone()
        placements: List[Placement] = []
        for job in queue:
            server = next(plan.feasible_servers(job.gpus), None)
            if server is None:
                break
            plan.allocate(job.job_id, job.gpus, server.cluster_id, server.server_id)
            placements.append(Placement(job.job_id, server.cluster_id, server.server_id))
        return placements
