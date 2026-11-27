"""MCG-Sched with capacity-aware fragmentation F_C (N2 – Person 5).

Extends MCGSched so that, for each candidate server, the fragmentation
gradient uses only the demand sizes that physically fit on that server's
capacity.  On a homogeneous topology every server has the same capacity
so the behaviour is identical to the original MCGSched (tested).

Two variants are registered:
  mcg_fc          – truncated F_C (p_tilde_d set to 0 for d > server cap; not renormed)
  mcg_fc_norm     – truncated + re-normalised (p_tilde(d|C) = p_tilde_d / sum_{q<=C} p_tilde_q)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..mechanisms.fragmentation import CapacityAwareDemandModel
from ..mechanisms.load_balance import LoadBalance
from ..models import ClusterState, Job
from .base import Placement, Scheduler
from .mcg import MCGSched

Choice = Optional[Tuple[float, int, int]]


class MCGHetero(MCGSched):
    """MCG-Sched using the capacity-aware fragmentation gradient G_C.

    Every call to _select() now passes the candidate server's capacity
    to the demand model so the fragmentation score only considers demands
    that the server can actually host.
    """

    name = "mcg_fc"

    def __init__(
        self,
        lam: float = 1.0,
        w: float = 0.5,
        window: Union[str, int] = "dynamic",
        l_star: float = 10.0,
        alpha: float = 0.05,
        use_queue: bool = True,
        skip_blocked: bool = False,
        renormalize: bool = False,
    ) -> None:
        # Build with the standard MCGSched args; the base __init__ creates a
        # base DemandModel which we then immediately replace.
        super().__init__(
            lam=lam, w=w, window=window, l_star=l_star, alpha=alpha,
            use_queue=use_queue, skip_blocked=skip_blocked,
        )
        self._renormalize = renormalize
        self._lam_arg = lam
        self._use_queue_arg = use_queue
        # Replace base DemandModel with the capacity-aware version.
        self.model = CapacityAwareDemandModel(
            lam=lam, use_queue=use_queue, renormalize=renormalize
        )

    def setup(self, jobs: Sequence[Job], state: ClusterState) -> None:
        """Override setup so it always re-creates a CapacityAwareDemandModel.

        MCGSched.setup() calls `self.model = DemandModel(...)`, which would
        overwrite our capacity-aware model.  We replicate the same logic but
        keep the right model type.
        """
        self.model = CapacityAwareDemandModel(
            lam=self._lam_arg, use_queue=self._use_queue_arg,
            renormalize=self._renormalize
        )
        self.model.fit(jobs, state.max_server_capacity)

    # Override _select so it passes server capacity to gradient_for_server.
    def _select(self, job: Job, plan: ClusterState, lb: LoadBalance) -> Choice:
        # We cannot cache by GPU count alone because different servers have
        # different capacities, so skip the cache.
        best: Choice = None
        for cluster in plan.clusters:
            sigma = None
            for server in cluster.servers:
                if server.free < job.gpus:
                    continue
                if sigma is None:
                    sigma = lb.after(cluster.cluster_id, job.gpus)
                # Use capacity-aware gradient
                g = self.model.gradient_for_server(server.free, job.gpus, server.capacity)
                v = (1 - self.w) * g + self.w * sigma
                if best is None or v < best[0]:
                    best = (v, cluster.cluster_id, server.server_id)
        return best

    # _order, schedule, etc. are inherited from MCGSched unchanged.


class MCGHeteroNorm(MCGHetero):
    """MCG_FC with the renormalised capacity-aware fragmentation."""

    name = "mcg_fc_norm"

    def __init__(self, **kw) -> None:
        kw["renormalize"] = True
        super().__init__(**kw)
