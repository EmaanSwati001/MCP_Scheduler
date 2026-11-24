from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..mechanisms.fragmentation import DemandModel
from ..mechanisms.load_balance import LoadBalance
from ..models import ClusterState, Job
from .base import Placement, Scheduler

Choice = Optional[Tuple[float, int, int]]


class MCGSched(Scheduler):
    """MCG-Sched: Select() (Algorithm 1) plus window prioritization (Algorithm 2).

    window: "dynamic", a fixed window size L, or 0 for no prioritization.
    skip_blocked: False follows the paper (stop at the first job that cannot start);
    True skips it and tries the next job.
    """

    name = "mcg"

    def __init__(
        self,
        lam: float = 1.0,
        w: float = 0.5,
        window: Union[str, int] = "dynamic",
        l_star: float = 10.0,
        alpha: float = 0.05,
        use_queue: bool = True,
        skip_blocked: bool = False,
    ) -> None:
        self.lam = lam
        self.w = w
        self.window = window
        self.l_star = l_star
        self.alpha = alpha
        self.use_queue = use_queue
        self.skip_blocked = skip_blocked
        self.model = DemandModel(lam, use_queue)
        self._cache: Dict[int, Choice] = {}

    def setup(self, jobs: Sequence[Job], state: ClusterState) -> None:
        self.model = DemandModel(self.lam, self.use_queue)
        self.model.fit(jobs, state.max_server_capacity)

    def _select(self, job: Job, plan: ClusterState, lb: LoadBalance) -> Choice:
        if job.gpus in self._cache:
            return self._cache[job.gpus]
        best: Choice = None
        for cluster in plan.clusters:
            sigma = None
            for server in cluster.servers:
                if server.free < job.gpus:
                    continue
                if sigma is None:
                    sigma = lb.after(cluster.cluster_id, job.gpus)
                g = self.model.gradient(server.free, job.gpus)
                v = (1 - self.w) * g + self.w * sigma
                if best is None or v < best[0]:
                    best = (v, cluster.cluster_id, server.server_id)
        self._cache[job.gpus] = best
        return best

    def _score_dynamic(self, queue, plan, lb) -> List[Tuple[float, Job]]:
        budget = self.l_star
        smin = float("inf")
        scored: List[Tuple[float, Job]] = []
        for job in queue:
            choice = self._select(job, plan, lb)
            if choice is None:
                scored.append((float("inf"), job))
                budget -= 1
            else:
                v = choice[0]
                if v < smin:
                    budget += 1
                elif v > smin:
                    budget -= 1
                else:
                    budget -= self.alpha
                smin = min(smin, v)
                scored.append((v, job))
            if budget < 0:
                break
        return scored

    def _score_fixed(self, queue, plan, lb, size: int) -> List[Tuple[float, Job]]:
        scored = []
        for job in queue[:size]:
            choice = self._select(job, plan, lb)
            scored.append((float("inf") if choice is None else choice[0], job))
        return scored

    def _order(self, queue: List[Job], plan: ClusterState, lb: LoadBalance) -> List[Job]:
        if self.window == 0:
            return list(queue)
        if self.window == "dynamic":
            scored = self._score_dynamic(queue, plan, lb)
        else:
            scored = self._score_fixed(queue, plan, lb, int(self.window))
        head = [job for _, job in sorted(scored, key=lambda x: x[0])]
        return head + list(queue[len(scored):])

    def schedule(self, queue: List[Job], state: ClusterState, now: float) -> List[Placement]:
        plan = state.clone()
        lb = LoadBalance(plan)
        self.model.set_queue(queue)
        self._cache = {}
        placements: List[Placement] = []
        for job in self._order(queue, plan, lb):
            choice = self._select(job, plan, lb)
            if choice is None:
                if self.skip_blocked:
                    continue
                break
            _, cluster_id, server_id = choice
            plan.allocate(job.job_id, job.gpus, cluster_id, server_id)
            lb.commit(cluster_id, job.gpus)
            self._cache = {}
            placements.append(Placement(job.job_id, cluster_id, server_id))
        return placements


class MCGNoWindow(MCGSched):
    name = "mcg_nowindow"

    def __init__(self, **kw) -> None:
        kw["window"] = 0
        super().__init__(**kw)


class MCGNoQueue(MCGSched):
    name = "mcg_noqueue"

    def __init__(self, **kw) -> None:
        kw["use_queue"] = False
        super().__init__(**kw)


class MCGPlain(MCGSched):
    name = "mcg_plain"

    def __init__(self, **kw) -> None:
        kw["window"] = 0
        kw["use_queue"] = False
        super().__init__(**kw)