from __future__ import annotations

from ..models import ClusterState


class LoadBalance:
    """Load-balance index (paper Eq. 6-9) with constant-time what-if evaluation."""

    def __init__(self, state: ClusterState) -> None:
        self.caps = [c.capacity for c in state.clusters]
        self.utils = state.utilizations()
        self.n = len(self.utils)
        self.total = sum(self.utils)
        self.total_sq = sum(u * u for u in self.utils)

    def index(self) -> float:
        mean = self.total / self.n
        return max(self.total_sq / self.n - mean * mean, 0.0)

    def after(self, cluster_id: int, gpus: int) -> float:
        d = gpus / self.caps[cluster_id]
        u = self.utils[cluster_id]
        total = self.total + d
        total_sq = self.total_sq + 2 * u * d + d * d
        mean = total / self.n
        return max(total_sq / self.n - mean * mean, 0.0)

    def commit(self, cluster_id: int, gpus: int) -> None:
        d = gpus / self.caps[cluster_id]
        u = self.utils[cluster_id]
        self.total += d
        self.total_sq += 2 * u * d + d * d
        self.utils[cluster_id] = u + d