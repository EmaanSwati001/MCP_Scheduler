"""MCG-Sched with the fairness (N1) and normalisation (N4) extensions.

N1  aging + starvation guard + drain reservation
N4  min-max normalised score (optional)

All of it is switched on by constructor arguments. With the defaults
(beta=0, k=None, reserve=False, normalize=False) MCGFair makes exactly the same
decisions as MCGSched, which tests/test_fairness_and_normalisation.py checks.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..mechanisms.load_balance import LoadBalance
from ..mechanisms.reservations import Key, ReservationTable, choose_drain_server
from ..models import ClusterState, Job
from .base import Placement
from .mcg import MCGSched, Choice

INF = float("inf")


def _norm(x: float, lo: float, hi: float) -> float:
    """Min-max scale to [0, 1]; 0 when the pool has a single value."""
    return (x - lo) / (hi - lo) if hi > lo else 0.0


class MCGFair(MCGSched):
    """MCG-Sched plus aging, a starvation guard, drain reservation and score normalisation.

    beta, aging_cap  v_age = v - beta * min(wait / W_ref, aging_cap); used only to order the window.
    k                a job waiting >= k * W_ref is protected and is tried before ordinary jobs.
                     None switches the guard off.
    reserve          True: a protected job that cannot start reserves a draining server and
                     everything else keeps running on the other servers.
                     False: the guard blocks (nothing else starts until the protected job does).
    max_reservations how many protected jobs may hold a reservation at once.
    normalize        min-max scale G and sigma^2 over one candidate pool per round.
    w_ref            reference wait. Default: mean job duration of the whole trace, fixed for the run.
    """

    name = "mcg_fair"

    def __init__(
        self,
        beta: float = 0.0,
        aging_cap: float = 5.0,
        k: Optional[float] = None,
        reserve: bool = False,
        max_reservations: int = 1,
        normalize: bool = False,
        w_ref: Optional[float] = None,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.beta = beta
        self.aging_cap = aging_cap
        self.k = k
        self.reserve = reserve
        self.max_reservations = max_reservations
        self.normalize = normalize
        self._w_ref_arg = w_ref
        self.w_ref = w_ref if w_ref else 1.0
        self.table = ReservationTable()
        self.protected_ids: set = set()
        self.log: List[Tuple[float, str, int, Optional[Key]]] = []
        self._excluded: FrozenSet[Key] = frozenset()
        self._sizes: Sequence[int] = ()
        self._bounds: Dict[FrozenSet[Key], Optional[Tuple[float, float, float, float]]] = {}

    # ---- setup / introspection ----------------------------------------

    def setup(self, jobs: Sequence[Job], state: ClusterState) -> None:
        super().setup(jobs, state)
        if self._w_ref_arg:
            self.w_ref = self._w_ref_arg
        else:
            mean = sum(j.duration for j in jobs) / len(jobs) if jobs else 0.0
            self.w_ref = mean if mean > 0 else 1.0
        self.table = ReservationTable()
        self.protected_ids = set()
        self.log = []

    def reserved_servers(self) -> Dict[int, Key]:
        """job_id -> reserved (cluster, server). For the dashboard."""
        return dict(self.table.by_job)

    # ---- candidate scoring --------------------------------------------

    def _candidates(self, gpus: int, plan: ClusterState, lb: LoadBalance, excluded):
        for cluster in plan.clusters:
            sigma = None
            for s in cluster.servers:
                if s.free < gpus or s.key in excluded:
                    continue
                if sigma is None:
                    sigma = lb.after(cluster.cluster_id, gpus)
                yield self.model.gradient(s.free, gpus), sigma, cluster.cluster_id, s.server_id

    def _pool_bounds(self, plan, lb, excluded):
        """min/max of G and sigma^2 over every feasible (size, server) candidate in the
        current round, so all jobs in the window are scored on one common scale."""
        key = frozenset(excluded)
        if key not in self._bounds:
            gs: List[float] = []
            ss: List[float] = []
            for size in self._sizes:
                for g, sg, _, _ in self._candidates(size, plan, lb, excluded):
                    gs.append(g)
                    ss.append(sg)
            self._bounds[key] = (min(gs), max(gs), min(ss), max(ss)) if gs else None
        return self._bounds[key]

    def _select_ex(self, job: Job, plan, lb, excluded) -> Choice:
        bounds = self._pool_bounds(plan, lb, excluded) if self.normalize else None
        best: Choice = None
        for g, sg, cid, sid in self._candidates(job.gpus, plan, lb, excluded):
            if bounds is not None:
                g, sg = _norm(g, bounds[0], bounds[1]), _norm(sg, bounds[2], bounds[3])
            v = (1 - self.w) * g + self.w * sg
            if best is None or v < best[0]:
                best = (v, cid, sid)
        return best

    def _select(self, job: Job, plan, lb) -> Choice:
        """Hook used by the parent's window code: ordinary jobs avoid reserved servers."""
        if job.gpus not in self._cache:
            self._cache[job.gpus] = self._select_ex(job, plan, lb, self._excluded)
        return self._cache[job.gpus]

    def _invalidate(self) -> None:
        self._cache = {}
        self._bounds = {}
        self._excluded = frozenset(self.table.servers())

    # ---- ordering ------------------------------------------------------

    def _bonus(self, job: Job, now: float) -> float:
        if self.beta == 0:
            return 0.0
        return self.beta * min(job.waited(now) / self.w_ref, self.aging_cap)

    def _order_fair(self, queue: List[Job], plan, lb, now: float) -> List[Job]:
        if self.window == 0:
            return list(queue)
        if self.window == "dynamic":
            scored = self._score_dynamic(queue, plan, lb)
        else:
            scored = self._score_fixed(queue, plan, lb, int(self.window))
        head = [j for _, j in sorted(scored, key=lambda x: x[0] - self._bonus(x[1], now))]
        return head + list(queue[len(scored):])

    # ---- one scheduling round -------------------------------------------

    def schedule(self, queue: List[Job], state: ClusterState, now: float) -> List[Placement]:
        plan = state.clone()
        lb = LoadBalance(plan)
        self.model.set_queue(queue)
        self.table.keep_only(j.job_id for j in queue)
        window_jobs = queue if self.window == "dynamic" else queue[: int(self.window) or 0]
        self._sizes = sorted({j.gpus for j in window_jobs})
        self._invalidate()
        placements: List[Placement] = []
        tried = set()
        failed = set()  # (gpus, own reservation) pairs that cannot start this round; free GPUs only shrink

        def start(job: Job, cid: int, sid: int) -> None:
            plan.allocate(job.job_id, job.gpus, cid, sid)
            lb.commit(cid, job.gpus)
            placements.append(Placement(job.job_id, cid, sid))
            if self.table.drop(job.job_id) is not None:
                self.log.append((now, "release", job.job_id, (cid, sid)))
            self._invalidate()

        # 1. protected jobs, oldest first
        if self.k is not None:
            threshold = self.k * self.w_ref
            for job in queue:
                if job.waited(now) < threshold:
                    continue
                if job.job_id not in self.protected_ids:
                    self.protected_ids.add(job.job_id)
                    self.log.append((now, "protect", job.job_id, None))
                own = self.table.by_job.get(job.job_id)
                if (job.gpus, own) in failed:
                    tried.add(job.job_id)
                    continue
                choice = self._select_ex(job, plan, lb, self.table.others(job.job_id))
                if choice is None:
                    failed.add((job.gpus, own))
                if choice is not None:
                    start(job, choice[1], choice[2])
                    tried.add(job.job_id)
                    continue
                if not self.reserve:
                    return placements  # blocking guard: nothing behind it may start
                if job.job_id not in self.table.by_job and len(self.table) < self.max_reservations:
                    key = choose_drain_server(job, plan, self.table.servers())
                    if key is not None:
                        self.table.add(job.job_id, key)
                        self.log.append((now, "reserve", job.job_id, key))
                        self._invalidate()
                tried.add(job.job_id)

        # 2. everyone else: MCG window ordering, reserved servers excluded
        rest = [j for j in queue if j.job_id not in tried]
        for job in self._order_fair(rest, plan, lb, now):
            choice = self._select(job, plan, lb)
            if choice is None:
                if self.skip_blocked:
                    continue
                break
            start(job, choice[1], choice[2])
        return placements


class MCGAging(MCGFair):
    name = "mcg_aging"

    def __init__(self, **kw) -> None:
        kw.setdefault("beta", 1.0)
        super().__init__(**kw)


class MCGGuard(MCGFair):
    """Blocking guard, no reservation: shows why reservation is needed."""

    name = "mcg_guard"

    def __init__(self, **kw) -> None:
        kw.setdefault("k", 3.0)
        super().__init__(**kw)


class MCGFairFull(MCGFair):
    """Aging + guard + drain reservation (the full N1 policy)."""

    name = "mcg_fair"

    def __init__(self, **kw) -> None:
        kw.setdefault("beta", 1.0)
        kw.setdefault("k", 3.0)
        kw.setdefault("reserve", True)
        super().__init__(**kw)


class MCGNorm(MCGFair):
    name = "mcg_norm"

    def __init__(self, **kw) -> None:
        kw.setdefault("normalize", True)
        super().__init__(**kw)
