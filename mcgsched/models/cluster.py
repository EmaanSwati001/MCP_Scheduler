from __future__ import annotations

import json
from typing import Iterator, List, Optional, Sequence

from .server import Server


class Cluster:
    """A group of servers. `used` is maintained by ClusterState only."""

    __slots__ = ("cluster_id", "servers", "capacity", "used")

    def __init__(self, cluster_id: int, capacities: Sequence[int]) -> None:
        self.cluster_id = cluster_id
        self.servers: List[Server] = [
            Server(cluster_id, i, c) for i, c in enumerate(capacities)
        ]
        self.capacity = sum(capacities)
        self.used = 0

    @property
    def utilization(self) -> float:
        """Fraction of this cluster's GPUs that are allocated (0..1)."""
        return self.used / self.capacity

    def __repr__(self) -> str:
        return f"Cluster{self.cluster_id}(used={self.used}/{self.capacity})"


def _variance(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


class ClusterState:
    """All clusters. The only object allowed to change GPU allocations.

    Schedulers receive the live state and must treat it as read-only. To plan
    several placements in one round, work on `state.clone()`.
    """

    def __init__(self, clusters: List[Cluster]) -> None:
        if not clusters:
            raise ValueError("topology needs at least one cluster")
        self.clusters = clusters
        self.total_capacity = sum(c.capacity for c in clusters)
        self.max_server_capacity = max(s.capacity for s in self.servers())

    # ---- construction -------------------------------------------------

    @classmethod
    def from_spec(cls, spec: dict) -> "ClusterState":
        """Build from a topology spec.

        {"clusters": [
            {"count": 8, "servers": 16, "gpus_per_server": 8},   # 8 equal clusters
            {"capacities": [4, 4, 8, 8]}                          # one mixed cluster
        ]}
        """
        clusters: List[Cluster] = []
        for entry in spec["clusters"]:
            if "capacities" in entry:
                caps = list(entry["capacities"])
            else:
                caps = [entry["gpus_per_server"]] * entry["servers"]
            if not caps or any((not isinstance(c, int)) or c < 1 for c in caps):
                raise ValueError(f"bad cluster entry: {entry}")
            for _ in range(entry.get("count", 1)):
                clusters.append(Cluster(len(clusters), caps))
        return cls(clusters)

    @classmethod
    def from_json(cls, path: str) -> "ClusterState":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_spec(json.load(fh))

    def clone(self) -> "ClusterState":
        """Independent deep copy (for scheduler planning or a fresh run)."""
        new = object.__new__(ClusterState)
        new.clusters = []
        for c in self.clusters:
            nc = object.__new__(Cluster)
            nc.cluster_id = c.cluster_id
            nc.capacity = c.capacity
            nc.used = c.used
            nc.servers = [
                Server(s.cluster_id, s.server_id, s.capacity, s.free, dict(s.running))
                for s in c.servers
            ]
            new.clusters.append(nc)
        new.total_capacity = self.total_capacity
        new.max_server_capacity = self.max_server_capacity
        return new

    # ---- queries ------------------------------------------------------

    def servers(self) -> Iterator[Server]:
        """All servers, cluster by cluster, in fixed id order."""
        for c in self.clusters:
            yield from c.servers

    def server(self, cluster_id: int, server_id: int) -> Server:
        if cluster_id < 0 or server_id < 0:
            raise IndexError((cluster_id, server_id))
        return self.clusters[cluster_id].servers[server_id]

    def feasible_servers(self, gpus: int) -> Iterator[Server]:
        """Servers that can host `gpus` right now, in fixed id order."""
        return (s for s in self.servers() if s.free >= gpus)

    @property
    def used_gpus(self) -> int:
        return sum(c.used for c in self.clusters)

    @property
    def free_gpus(self) -> int:
        return self.total_capacity - self.used_gpus

    def utilizations(self) -> List[float]:
        return [c.utilization for c in self.clusters]

    def load_balance_index(self) -> float:
        """Paper Eq. (6): variance of cluster utilization. Smaller = more balanced."""
        return _variance(self.utilizations())

    def lb_index_after(self, cluster_id: int, gpus: int) -> float:
        """Paper Eq. (7)-(9): the index if `gpus` more were allocated in `cluster_id`."""
        utils = self.utilizations()
        utils[cluster_id] += gpus / self.clusters[cluster_id].capacity
        return _variance(utils)

    # ---- mutation -----------------------------------------------------

    def allocate(self, job_id: int, gpus: int, cluster_id: int, server_id: int) -> None:
        self.server(cluster_id, server_id).allocate(job_id, gpus)
        self.clusters[cluster_id].used += gpus

    def release(self, job_id: int, cluster_id: int, server_id: int) -> None:
        gpus = self.server(cluster_id, server_id).release(job_id)
        self.clusters[cluster_id].used -= gpus

    def __repr__(self) -> str:
        return (
            f"ClusterState({len(self.clusters)} clusters, "
            f"{self.used_gpus}/{self.total_capacity} GPUs used)"
        )
