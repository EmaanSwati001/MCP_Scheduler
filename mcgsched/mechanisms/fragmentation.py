from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

from ..models import Job


class DemandModel:
    """Demand popularity (Eq. 3-4), fragmentation F(r) (Eq. 5) and gradient G (Eq. 10).

    This is the original homogeneous model used by MCG-Sched.  All demands are
    included regardless of server capacity.
    """

    def __init__(self, lam: float = 1.0, use_queue: bool = True) -> None:
        self.lam = lam
        self.use_queue = use_queue
        self.global_pop: Dict[int, float] = {}
        self.max_r = 0
        self._queue_counts: Counter = Counter()
        self._queue_len = 0
        self._tables: Dict[int, List[float]] = {}

    def fit(self, jobs: Sequence[Job], max_capacity: int) -> None:
        counts = Counter(j.gpus for j in jobs)
        total = sum(counts.values()) or 1
        self.global_pop = {d: c / total for d, c in counts.items()}
        self.max_r = max_capacity
        self.set_queue([])

    def set_queue(self, queue: Sequence[Job]) -> None:
        self._queue_counts = Counter(j.gpus for j in queue)
        self._queue_len = len(queue)
        self._tables = {}

    def popularity(self, cur_gpus: int) -> Dict[int, float]:
        m = self._queue_len
        if not self.use_queue or m <= 1:
            return dict(self.global_pop)
        sizes = set(self.global_pop) | set(self._queue_counts)
        pop = {}
        for d in sizes:
            others = self._queue_counts[d] - (1 if d == cur_gpus else 0)
            p_queue = others / (m - 1)
            pop[d] = (self.global_pop.get(d, 0.0) + self.lam * p_queue) / (1 + self.lam)
        return pop

    def table(self, cur_gpus: int) -> List[float]:
        t = self._tables.get(cur_gpus)
        if t is None:
            pop = self.popularity(cur_gpus)
            t = [sum(p * r for d, p in pop.items() if d > r) for r in range(self.max_r + 1)]
            self._tables[cur_gpus] = t
        return t

    def F(self, r: int, cur_gpus: int) -> float:
        return self.table(cur_gpus)[r]

    def gradient(self, free: int, gpus: int) -> float:
        t = self.table(gpus)
        return t[free - gpus] - t[free]


# ---------------------------------------------------------------------------
# N2: Capacity-aware fragmentation (Person 5)
# ---------------------------------------------------------------------------

class CapacityAwareDemandModel(DemandModel):
    """Capacity-aware fragmentation F_C (N2 extension).

    For a server with capacity C, only demands d <= C are relevant. The
    popularity distribution is re-normalised over those feasible demands so
    the fragmentation score reflects what can *actually* run on the server,
    rather than jobs that would be rejected on arrival anyway.

    Two modes controlled by `renormalize`:
      True  – p_tilde(d | C) = p_tilde_d / sum_{q: D_q <= C} p_tilde_q
              (pure capacity filter, normalised to 1 over feasible demands)
      False – p_tilde(d | C) = p_tilde_d  for d <= C, 0 otherwise
              (truncated, not renormalised; still uses the blended global+queue mix)

    On a cluster where every server has the same max capacity, F_C with
    renormalize=False is identical to the original F (tested in test_heterogeneous.py).
    """

    def __init__(self, lam: float = 1.0, use_queue: bool = True,
                 renormalize: bool = False) -> None:
        super().__init__(lam=lam, use_queue=use_queue)
        self.renormalize = renormalize
        # Per-capacity table cache: (cur_gpus, server_capacity) -> List[float]
        self._cap_tables: Dict[tuple, List[float]] = {}

    def set_queue(self, queue: Sequence[Job]) -> None:
        super().set_queue(queue)
        self._cap_tables = {}   # invalidate capacity-specific cache too

    def _capacity_table(self, cur_gpus: int, server_cap: int) -> List[float]:
        """F_C table for a server whose maximum capacity is `server_cap`."""
        key = (cur_gpus, server_cap)
        t = self._cap_tables.get(key)
        if t is not None:
            return t

        pop = self.popularity(cur_gpus)
        # Keep only demands that the server can physically host.
        feasible = {d: p for d, p in pop.items() if d <= server_cap}
        if not feasible:
            # No demand fits: assign a constant (non-informative) table of 0.
            t = [0.0] * (self.max_r + 1)
            self._cap_tables[key] = t
            return t

        if self.renormalize:
            total = sum(feasible.values()) or 1.0
            feasible = {d: p / total for d, p in feasible.items()}

        t = [
            sum(p * r for d, p in feasible.items() if d > r)
            for r in range(self.max_r + 1)
        ]
        self._cap_tables[key] = t
        return t

    def gradient_for_server(self, free: int, gpus: int, server_cap: int) -> float:
        """G_C = F_C(free - gpus, cap) - F_C(free, cap).

        Falls back to the original gradient when server_cap >= max_r
        (homogeneous case; produces identical numbers).
        """
        if server_cap >= self.max_r:
            return self.gradient(free, gpus)
        t = self._capacity_table(gpus, server_cap)
        return t[free - gpus] - t[free]

    # Keep the original gradient() available so MCGSched can call it without
    # knowing about capacity; call gradient_for_server() for N2.
    def gradient(self, free: int, gpus: int) -> float:  # type: ignore[override]
        """Original F-based gradient (ignores capacity).  Preserved for compatibility."""
        return super().gradient(free, gpus)