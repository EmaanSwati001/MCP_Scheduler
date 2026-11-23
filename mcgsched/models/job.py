from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass
class Job:
    """One GPU job. The first four fields are the common trace schema.

    start_time, end_time and placement are written by the simulator only.
    """

    job_id: int
    arrival_time: float
    gpus: int
    duration: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    placement: Optional[Tuple[int, int]] = None  # (cluster_id, server_id)

    def __post_init__(self) -> None:
        if self.gpus < 1:
            raise ValueError(f"job {self.job_id}: gpus must be >= 1, got {self.gpus}")
        if self.duration < 0:
            raise ValueError(f"job {self.job_id}: negative duration")
        if self.arrival_time < 0:
            raise ValueError(f"job {self.job_id}: negative arrival time")

    @property
    def wait_time(self) -> Optional[float]:
        """Queueing delay; None until the job has started."""
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time

    def waited(self, now: float) -> float:
        """How long the job has been in the queue at time `now`."""
        return now - self.arrival_time

    def fresh(self) -> "Job":
        """Copy with all simulator-written fields cleared."""
        return replace(self, start_time=None, end_time=None, placement=None)
