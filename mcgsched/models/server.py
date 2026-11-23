from __future__ import annotations

from typing import Dict, Optional


class Server:
    """One node with a fixed integer GPU capacity. GPUs are exclusive."""

    __slots__ = ("cluster_id", "server_id", "capacity", "free", "running")

    def __init__(
        self,
        cluster_id: int,
        server_id: int,
        capacity: int,
        free: Optional[int] = None,
        running: Optional[Dict[int, int]] = None,
    ) -> None:
        self.cluster_id = cluster_id
        self.server_id = server_id
        self.capacity = capacity
        self.free = capacity if free is None else free
        self.running: Dict[int, int] = {} if running is None else running  # job_id -> gpus

    def can_fit(self, gpus: int) -> bool:
        return self.free >= gpus

    def allocate(self, job_id: int, gpus: int) -> None:
        if job_id in self.running:
            raise ValueError(f"job {job_id} already runs on server {self.key}")
        if gpus > self.free:
            raise ValueError(
                f"server {self.key}: job {job_id} wants {gpus} GPUs, only {self.free} free"
            )
        self.free -= gpus
        self.running[job_id] = gpus

    def release(self, job_id: int) -> int:
        gpus = self.running.pop(job_id)
        self.free += gpus
        return gpus

    @property
    def key(self):
        return (self.cluster_id, self.server_id)

    def __repr__(self) -> str:
        return f"Server{self.key}(free={self.free}/{self.capacity})"
