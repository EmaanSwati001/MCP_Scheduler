from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from ..models import Job


class DemandModel:
    """Demand popularity (Eq. 3-4), fragmentation F(r) (Eq. 5) and gradient G (Eq. 10)."""

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