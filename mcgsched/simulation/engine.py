from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from ..models import ClusterState, Job
from ..schedulers.base import Scheduler
from .metrics import Sample, summarize

# Event kinds. At equal timestamps completions are handled before arrivals so
# released GPUs are visible to the scheduling round at that instant.
_FINISH, _ARRIVAL = 0, 1


class SchedulerError(RuntimeError):
    """A scheduler returned a placement the simulator cannot apply."""


@dataclass
class SimulationResult:
    scheduler: str
    finished: List[Job]      # in completion order
    rejected: List[Job]      # larger than any server, can never run
    unscheduled: List[Job]   # still waiting when no events were left
    timeline: List[Sample]
    summary: Dict[str, object]


class Simulator:
    """Discrete-event simulator: single-server, exclusive, non-preemptive jobs.

    The caller's jobs and state are copied, so the same inputs can be fed to
    every scheduler. `listener(event, sim, job)` is called for "arrival",
    "reject", "start" and "finish" (used by the dashboard).
    """

    def __init__(
        self,
        jobs: Sequence[Job],
        state: ClusterState,
        scheduler: Scheduler,
        listener: Optional[Callable[[str, "Simulator", Job], None]] = None,
    ) -> None:
        self.state = state.clone()
        self.scheduler = scheduler
        self.listener = listener
        self.jobs: Dict[int, Job] = {}
        for job in jobs:
            if job.job_id in self.jobs:
                raise ValueError(f"duplicate job_id {job.job_id}")
            self.jobs[job.job_id] = job.fresh()

        self.queue: List[Job] = []          # waiting, arrival order
        self.running: Dict[int, Job] = {}
        self.finished: List[Job] = []
        self.rejected: List[Job] = []
        self.timeline: List[Sample] = []
        self._heap: list = []
        self._seq = 0
        for job in sorted(self.jobs.values(), key=lambda j: (j.arrival_time, j.job_id)):
            self._push(job.arrival_time, _ARRIVAL, job.job_id)
        self.now = self._heap[0][0] if self._heap else 0.0

        scheduler.setup([j.fresh() for j in self.jobs.values()], self.state.clone())

    # ---- public -------------------------------------------------------

    def step(self) -> bool:
        """Process every event at the next timestamp, then run one scheduling
        round. Returns False when no events are left."""
        if not self._heap:
            return False
        self.now = self._heap[0][0]
        while self._heap and self._heap[0][0] == self.now:
            _, kind, _, job_id = heapq.heappop(self._heap)
            job = self.jobs[job_id]
            if kind == _FINISH:
                self._finish(job)
            else:
                self._arrive(job)
        self._schedule_round()
        self.timeline.append(
            Sample(
                self.now,
                self.state.used_gpus,
                self.state.load_balance_index(),
                len(self.queue),
                len(self.running),
            )
        )
        return True

    def run(self) -> SimulationResult:
        while self.step():
            pass
        return self.result()

    def result(self) -> SimulationResult:
        return SimulationResult(
            scheduler=self.scheduler.name,
            finished=list(self.finished),
            rejected=list(self.rejected),
            unscheduled=list(self.queue),
            timeline=list(self.timeline),
            summary=summarize(
                self.finished,
                self.timeline,
                self.state.total_capacity,
                rejected=len(self.rejected),
                unscheduled=len(self.queue),
            ),
        )

    # ---- internals ----------------------------------------------------

    def _push(self, time: float, kind: int, job_id: int) -> None:
        heapq.heappush(self._heap, (time, kind, self._seq, job_id))
        self._seq += 1

    def _emit(self, event: str, job: Job) -> None:
        if self.listener is not None:
            self.listener(event, self, job)

    def _arrive(self, job: Job) -> None:
        if job.gpus > self.state.max_server_capacity:
            self.rejected.append(job)
            self._emit("reject", job)
            return
        self.queue.append(job)
        self._emit("arrival", job)

    def _finish(self, job: Job) -> None:
        self.state.release(job.job_id, *job.placement)
        del self.running[job.job_id]
        self.finished.append(job)
        self._emit("finish", job)

    def _schedule_round(self) -> None:
        if not self.queue:
            return
        placements = self.scheduler.schedule(list(self.queue), self.state, self.now)
        if not placements:
            return
        waiting = {j.job_id: j for j in self.queue}
        for p in placements:
            job = waiting.pop(p.job_id, None)
            if job is None:
                raise SchedulerError(
                    f"{self.scheduler.name}: job {p.job_id} is not waiting "
                    f"(unknown, already started, or placed twice)"
                )
            try:
                self.state.allocate(job.job_id, job.gpus, p.cluster_id, p.server_id)
            except (IndexError, ValueError) as exc:
                raise SchedulerError(f"{self.scheduler.name}: bad placement {p}: {exc}") from exc
            job.start_time = self.now
            job.end_time = self.now + job.duration
            job.placement = (p.cluster_id, p.server_id)
            self.running[job.job_id] = job
            self._push(job.end_time, _FINISH, job.job_id)
            self._emit("start", job)
        self.queue = [j for j in self.queue if j.job_id in waiting]


def simulate(jobs: Sequence[Job], state: ClusterState, scheduler: Scheduler) -> SimulationResult:
    """Run one scheduler over a workload and topology."""
    return Simulator(jobs, state, scheduler).run()
