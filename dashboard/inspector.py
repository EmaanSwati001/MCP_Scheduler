"""Simulation inspection, live decision telemetry, and experiment helpers
for the MCG-Sched Control Plane dashboard.

Runs exclusively on top of the real `mcgsched` simulator and schedulers
without modifying any core simulation logic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mcgsched.evaluation.fairness import fairness_metrics
from mcgsched.mechanisms.fragmentation import CapacityAwareDemandModel, DemandModel
from mcgsched.mechanisms.load_balance import LoadBalance
from mcgsched.models import ClusterState, Job
from mcgsched.schedulers import REGISTRY, get_scheduler
from mcgsched.schedulers.base import Placement, Scheduler
from mcgsched.simulation import simulate
from mcgsched.simulation.engine import Simulator
from mcgsched.workloads import (
    compress_arrivals,
    load_jobs_csv,
    make_burst,
    synthetic_jobs,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "clean"
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"

TRACE_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "venus",
        "name": "Venus (Helios)",
        "file": str(DATA_DIR / "venus.csv"),
        "description": "SenseTime Helios Venus cluster (15-day window, 6,941 GPU jobs)",
    },
    {
        "id": "philly",
        "name": "Philly (Microsoft)",
        "file": str(DATA_DIR / "philly.csv"),
        "description": "Microsoft Philly GPU cluster trace (7-day window, 5,043 jobs)",
    },
    {
        "id": "earth",
        "name": "Earth (Helios)",
        "file": str(DATA_DIR / "earth.csv"),
        "description": "Additional Helios cluster trace for generalization testing",
    },
    {
        "id": "saturn",
        "name": "Saturn (Helios)",
        "file": str(DATA_DIR / "saturn.csv"),
        "description": "Additional Helios cluster trace for generalization testing",
    },
    {
        "id": "uranus",
        "name": "Uranus (Helios)",
        "file": str(DATA_DIR / "uranus.csv"),
        "description": "Additional Helios cluster trace for generalization testing",
    },
    {
        "id": "synthetic",
        "name": "Synthetic (Poisson)",
        "file": None,
        "description": "Controlled Poisson arrivals, exponential durations, seeded",
    },
]

TOPOLOGY_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "venus15",
        "name": "Venus 15-Cluster (Homogeneous 8-GPU)",
        "file": str(CONFIG_DIR / "venus15.json"),
        "tag": "Paper Standard",
    },
    {
        "id": "venus15_mixed",
        "name": "Venus 15-Cluster (Mixed 4+8-GPU)",
        "file": str(CONFIG_DIR / "venus15_mixed.json"),
        "tag": "Heterogeneous",
    },
    {
        "id": "paper_8x16",
        "name": "Paper 8x16 (1,024 GPUs)",
        "file": str(CONFIG_DIR / "paper_8x16.json"),
        "tag": "Paper Synthetic",
    },
    {
        "id": "paper_16x32",
        "name": "Paper 16x32 (4,096 GPUs)",
        "file": str(CONFIG_DIR / "paper_16x32.json"),
        "tag": "Large Scale",
    },
    {
        "id": "earth",
        "name": "Earth Cluster",
        "file": str(CONFIG_DIR / "earth.json"),
        "tag": "Helios",
    },
    {
        "id": "saturn",
        "name": "Saturn Cluster",
        "file": str(CONFIG_DIR / "saturn.json"),
        "tag": "Helios",
    },
    {
        "id": "uranus",
        "name": "Uranus Cluster",
        "file": str(CONFIG_DIR / "uranus.json"),
        "tag": "Helios",
    },
    {
        "id": "toy",
        "name": "Toy (2x2x8 = 32 GPUs)",
        "file": str(CONFIG_DIR / "toy.json"),
        "tag": "Micro Demo",
    },
]

SCHEDULER_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "mcg",
        "name": "MCG-Sched (Full)",
        "category": "MCG-Sched Core",
        "description": "Demand-aware fragmentation G + cluster load balance sigma^2 + dynamic window.",
        "params": ["w", "lam", "l_star", "alpha"],
    },
    {
        "id": "mcg_fair",
        "name": "MCG-Fair (Aging + Guard + Drain Reserve)",
        "category": "Fairness & Normalization",
        "description": "Full starvation protection: score aging, hard wait guard (k*W_ref), and non-preemptive drain reservations.",
        "params": ["w", "lam", "beta", "k", "max_reservations"],
    },
    {
        "id": "mcg_aging",
        "name": "MCG-Aging",
        "category": "Fairness & Normalization",
        "description": "Window prioritization with wait-time aging bonus (v - beta * wait/W_ref).",
        "params": ["w", "lam", "beta"],
    },
    {
        "id": "mcg_guard",
        "name": "MCG-Guard (Blocking)",
        "category": "Fairness & Normalization",
        "description": "Hard starvation guard without drain reservation (blocks queue until protected job starts).",
        "params": ["w", "lam", "k"],
    },
    {
        "id": "mcg_norm",
        "name": "MCG-Norm (Min-Max Scaled)",
        "category": "Fairness & Normalization",
        "description": "Min-max scales G and sigma^2 across each round's shared candidate pool before weighting by w.",
        "params": ["w", "lam"],
    },
    {
        "id": "mcg_fc",
        "name": "MCG-F_C (Capacity-Aware Fragmentation)",
        "category": "Heterogeneous",
        "description": "Filters demand distribution p_tilde to only job sizes d <= server capacity C.",
        "params": ["w", "lam"],
    },
    {
        "id": "mcg_fc_norm",
        "name": "MCG-F_C Norm (Renormalised)",
        "category": "Heterogeneous",
        "description": "Capacity-aware fragmentation with p_tilde(d|C) renormalised over feasible demands.",
        "params": ["w", "lam"],
    },
    {
        "id": "fgd",
        "name": "FGD (Weng et al.)",
        "category": "Baselines",
        "description": "Fragmentation Gradient Descent using global demand mix only (no cluster load balancing).",
        "params": [],
    },
    {
        "id": "consolidate",
        "name": "Consolidate (Best Fit)",
        "category": "Baselines",
        "description": "Places each job on the feasible server with fewest free GPUs to preserve large empty nodes.",
        "params": [],
    },
    {
        "id": "fcfs",
        "name": "FCFS + First Fit",
        "category": "Baselines",
        "description": "Classic First-Come First-Served with First-Fit server allocation.",
        "params": [],
    },
    {
        "id": "mast",
        "name": "MAST-Like",
        "category": "Baselines",
        "description": "Prioritizes jobs by age/GPUs and selects the server with maximum free GPUs.",
        "params": [],
    },
    {
        "id": "random",
        "name": "Random Placement",
        "category": "Baselines",
        "description": "Seeded uniform random choice across feasible servers.",
        "params": [],
    },
    {
        "id": "mcg_nowindow",
        "name": "MCG (No Window)",
        "category": "Ablations",
        "description": "Ablation: disables dynamic window prioritization (L=0, strict arrival order).",
        "params": ["w", "lam"],
    },
    {
        "id": "mcg_noqueue",
        "name": "MCG (No Queue Blend)",
        "category": "Ablations",
        "description": "Ablation: uses global demand popularity only (ignores current queue mix).",
        "params": ["w"],
    },
    {
        "id": "mcg_plain",
        "name": "MCG (Plain)",
        "category": "Ablations",
        "description": "Ablation: disables both window prioritization and queue popularity blend.",
        "params": ["w"],
    },
]


def get_catalog_meta() -> Dict[str, Any]:
    """Return trace, topology, and scheduler metadata for the UI."""
    traces = []
    for item in TRACE_CATALOG:
        path = item["file"]
        exists = path is None or os.path.exists(path)
        count = None
        if path and exists:
            # Fast line count minus CSV header
            with open(path, "r", encoding="utf-8") as fh:
                count = max(sum(1 for _ in fh) - 1, 0)
        traces.append({
            "id": item["id"],
            "name": item["name"],
            "description": item["description"],
            "available": exists,
            "total_jobs": count,
        })

    topologies = []
    for item in TOPOLOGY_CATALOG:
        path = item["file"]
        if not os.path.exists(path):
            continue
        state = ClusterState.from_json(path)
        cap_counts: Dict[int, int] = {}
        for s in state.servers():
            cap_counts[s.capacity] = cap_counts.get(s.capacity, 0) + 1
        topologies.append({
            "id": item["id"],
            "name": item["name"],
            "tag": item["tag"],
            "clusters": len(state.clusters),
            "servers": sum(len(c.servers) for c in state.clusters),
            "total_gpus": state.total_capacity,
            "max_server_gpus": state.max_server_capacity,
            "server_mix": {str(k): v for k, v in sorted(cap_counts.items())},
        })

    schedulers = [s for s in SCHEDULER_CATALOG if s["id"] in REGISTRY]
    return {
        "traces": traces,
        "topologies": topologies,
        "schedulers": schedulers,
    }


def load_workload(
    trace_id: str,
    n_jobs: int = 300,
    eta: float = 10.0,
    burst: bool = False,
    seed: int = 0,
) -> List[Job]:
    """Load and preprocess a workload slice."""
    entry = next((t for t in TRACE_CATALOG if t["id"] == trace_id), None)
    if entry is None:
        raise ValueError(f"Unknown trace: {trace_id}")
    path = entry["file"]
    if path and os.path.exists(path):
        jobs = load_jobs_csv(path)
    else:
        jobs = synthetic_jobs(n_jobs if n_jobs > 0 else 200, seed=seed)

    if n_jobs > 0 and len(jobs) > n_jobs:
        jobs = jobs[:n_jobs]
    if eta != 1.0:
        jobs = compress_arrivals(jobs, eta)
    if burst:
        jobs = make_burst(jobs)
    return jobs


def load_topology(topo_id: str) -> ClusterState:
    entry = next((t for t in TOPOLOGY_CATALOG if t["id"] == topo_id), None)
    if entry is None:
        raise ValueError(f"Unknown topology: {topo_id}")
    return ClusterState.from_json(entry["file"])


def build_scheduler(sched_id: str, params: Optional[Dict[str, Any]] = None) -> Scheduler:
    """Instantiate a scheduler with optional safe hyperparameter overrides."""
    if not params:
        return get_scheduler(sched_id)
    allowed = {}
    info = next((s for s in SCHEDULER_CATALOG if s["id"] == sched_id), None)
    supported = set(info["params"]) if info else set()
    for k, v in params.items():
        if k in supported and v is not None:
            if k in ("max_reservations",):
                allowed[k] = int(v)
            else:
                allowed[k] = float(v)
    return get_scheduler(sched_id, **allowed)


class ObservedScheduler:
    """Transparent proxy around any Scheduler that captures pre-round state
    and live candidate scoring breakdowns for the dashboard inspector."""

    def __init__(self, inner: Scheduler, session: "SimulationSession") -> None:
        self._inner = inner
        self._session = session
        self.name = inner.name

    def setup(self, jobs: Sequence[Job], state: ClusterState) -> None:
        self._inner.setup(jobs, state)
        # Also prepare a reference CapacityAwareDemandModel for N2 side-by-side inspection
        self._session.ref_model_orig = DemandModel(
            lam=getattr(self._inner, "lam", 1.0),
            use_queue=getattr(self._inner, "use_queue", True),
        )
        self._session.ref_model_orig.fit(jobs, state.max_server_capacity)
        self._session.ref_model_fc = CapacityAwareDemandModel(
            lam=getattr(self._inner, "lam", 1.0),
            use_queue=getattr(self._inner, "use_queue", True),
            renormalize=True,
        )
        self._session.ref_model_fc.fit(jobs, state.max_server_capacity)

    def schedule(self, queue: List[Job], state: ClusterState, now: float) -> List[Placement]:
        pre_state = state.clone() if self._session.capture_details else None
        placements = self._inner.schedule(queue, state, now)
        if self._session.capture_details and queue:
            self._session._record_decision(pre_state, queue, placements, now)
        return placements

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class SimulationSession:
    """Stateful wrapper around `Simulator` for interactive step/play/run."""

    def __init__(
        self,
        trace_id: str,
        topo_id: str,
        sched_id: str,
        n_jobs: int = 300,
        eta: float = 10.0,
        burst: bool = False,
        seed: int = 0,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = {
            "trace_id": trace_id,
            "topo_id": topo_id,
            "sched_id": sched_id,
            "n_jobs": n_jobs,
            "eta": eta,
            "burst": burst,
            "seed": seed,
            "params": params or {},
        }
        jobs = load_workload(trace_id, n_jobs=n_jobs, eta=eta, burst=burst, seed=seed)
        self.total_jobs = len(jobs)
        durs = [j.duration for j in jobs]
        self.w_ref = (sum(durs) / len(durs)) if durs else 1.0
        state = load_topology(topo_id)
        inner_sched = build_scheduler(sched_id, params)
        self.inner_scheduler = inner_sched
        self.capture_details = True
        self.ref_model_orig: Optional[DemandModel] = None
        self.ref_model_fc: Optional[CapacityAwareDemandModel] = None

        self.step_count = 0
        self.done = False
        self.recent_events: List[Dict[str, Any]] = []
        self.last_decision: Optional[Dict[str, Any]] = None

        proxy = ObservedScheduler(inner_sched, self)
        self.sim = Simulator(jobs, state, proxy, listener=self._on_event)

    def _on_event(self, event: str, sim: Simulator, job: Job) -> None:
        if not self.capture_details and event in ("arrival", "finish"):
            return
        entry = {
            "time": round(sim.now, 2),
            "event": event,
            "job_id": job.job_id,
            "gpus": job.gpus,
            "placement": list(job.placement) if job.placement else None,
            "wait": round(job.wait_time, 2) if job.wait_time is not None else None,
        }
        self.recent_events.append(entry)
        if len(self.recent_events) > 50:
            self.recent_events = self.recent_events[-50:]

    def _record_decision(
        self,
        pre_state: Optional[ClusterState],
        queue: List[Job],
        placements: List[Placement],
        now: float,
    ) -> None:
        if pre_state is None or not queue:
            return

        # Choose the focal job: first placed job, or head of queue if blocked
        placed_map = {p.job_id: (p.cluster_id, p.server_id) for p in placements}
        focal_job = next((j for j in queue if j.job_id in placed_map), queue[0])
        chosen_key = placed_map.get(focal_job.job_id)

        if self.ref_model_orig is not None:
            self.ref_model_orig.set_queue(queue)
        if self.ref_model_fc is not None:
            self.ref_model_fc.set_queue(queue)

        lb = LoadBalance(pre_state)
        w = getattr(self.inner_scheduler, "w", 0.5)
        reserved_map = (
            self.inner_scheduler.reserved_servers()
            if hasattr(self.inner_scheduler, "reserved_servers")
            else {}
        )
        reserved_set = set(reserved_map.values())
        own_res = reserved_map.get(focal_job.job_id)

        candidates = []
        for s in pre_state.servers():
            if s.free < focal_job.gpus:
                continue
            is_excl = s.key in reserved_set and s.key != own_res
            g_orig = (
                self.ref_model_orig.gradient(s.free, focal_job.gpus)
                if self.ref_model_orig
                else 0.0
            )
            g_fc = (
                self.ref_model_fc.gradient_for_server(s.free, focal_job.gpus, s.capacity)
                if self.ref_model_fc
                else 0.0
            )
            sigma2 = lb.after(s.cluster_id, focal_job.gpus)
            v_orig = (1.0 - w) * g_orig + w * sigma2
            v_fc = (1.0 - w) * g_fc + w * sigma2
            candidates.append({
                "cluster_id": s.cluster_id,
                "server_id": s.server_id,
                "capacity": s.capacity,
                "free": s.free,
                "g_orig": round(g_orig, 5),
                "g_fc": round(g_fc, 5),
                "sigma2": round(sigma2, 6),
                "score": round(v_orig, 5),
                "score_fc": round(v_fc, 5),
                "selected": chosen_key == s.key,
                "reserved_excluded": is_excl,
            })

        # Sort by score ascending, keep top 20 (always keeping the selected placement)
        candidates.sort(key=lambda c: (not c["selected"], c["reserved_excluded"], c["score"]))
        selected_cands = [c for c in candidates if c["selected"]]
        others = [c for c in candidates if not c["selected"]][:19]
        top_candidates = selected_cands + others
        top_candidates.sort(key=lambda c: c["score"])

        # Queue window snapshot (top 12 waiting jobs)
        beta = getattr(self.inner_scheduler, "beta", 0.0)
        aging_cap = getattr(self.inner_scheduler, "aging_cap", 5.0)
        k_guard = getattr(self.inner_scheduler, "k", None)
        protected_ids = getattr(self.inner_scheduler, "protected_ids", set())
        window_preview = []
        for idx, j in enumerate(queue[:12]):
            waited = j.waited(now)
            bonus = (
                beta * min(waited / self.w_ref, aging_cap)
                if beta and self.w_ref > 0
                else 0.0
            )
            is_prot = (j.job_id in protected_ids) or (
                k_guard is not None and waited >= k_guard * self.w_ref
            )
            window_preview.append({
                "pos": idx,
                "job_id": j.job_id,
                "gpus": j.gpus,
                "waited": round(waited, 1),
                "wait_ratio": round(waited / self.w_ref, 2) if self.w_ref > 0 else 0.0,
                "aging_bonus": round(bonus, 4),
                "protected": is_prot,
                "placed": placed_map.get(j.job_id),
            })

        self.last_decision = {
            "time": round(now, 2),
            "focal_job_id": focal_job.job_id,
            "focal_gpus": focal_job.gpus,
            "focal_waited": round(focal_job.waited(now), 2),
            "placed_count": len(placements),
            "placements": [
                {"job_id": p.job_id, "cluster_id": p.cluster_id, "server_id": p.server_id}
                for p in placements[:15]
            ],
            "w": w,
            "candidates_total": len(candidates),
            "candidates": top_candidates,
            "window_preview": window_preview,
        }

    def step(self, count: int = 1) -> Dict[str, Any]:
        if self.done:
            return self.snapshot()
        count = max(1, int(count))
        # Capture detailed round snapshots only on the final step of a batch
        for i in range(count):
            self.capture_details = (i == count - 1)
            ok = self.sim.step()
            if not ok:
                self.done = True
                break
            self.step_count += 1
        if not self.sim._heap:
            self.done = True
        self.capture_details = True
        return self.snapshot()

    def run_to_end(self) -> Dict[str, Any]:
        if not self.done:
            self.capture_details = False
            while self.sim.step():
                self.step_count += 1
            self.done = True
            self.capture_details = True
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        sim = self.sim
        state = sim.state

        reserved_by_job = (
            sim.scheduler.reserved_servers()
            if hasattr(sim.scheduler, "reserved_servers")
            else {}
        )
        reserved_by_server = {v: k for k, v in reserved_by_job.items()}
        protected_ids = list(sorted(getattr(sim.scheduler, "protected_ids", set())))
        n1_log_raw = getattr(sim.scheduler, "log", [])[-25:]
        n1_log = [
            {
                "time": round(t, 1),
                "action": act,
                "job_id": jid,
                "server": list(srv) if srv is not None else None,
            }
            for t, act, jid, srv in n1_log_raw
        ]

        # Compute stranded GPUs: free GPUs on partially/lightly used servers that
        # cannot satisfy the smallest waiting demand (or head of queue)
        min_wait_gpus = min((j.gpus for j in sim.queue), default=0)
        stranded_gpus = 0
        if min_wait_gpus > 0:
            for s in state.servers():
                if 0 < s.free < min_wait_gpus:
                    stranded_gpus += s.free

        clusters_data = []
        for c in state.clusters:
            servers_data = []
            for s in c.servers:
                res_job = reserved_by_server.get(s.key)
                servers_data.append({
                    "server_id": s.server_id,
                    "capacity": s.capacity,
                    "free": s.free,
                    "used": s.capacity - s.free,
                    "reserved_for": res_job,
                    "running": [
                        {"job_id": jid, "gpus": g}
                        for jid, g in s.running.items()
                    ],
                })
            clusters_data.append({
                "cluster_id": c.cluster_id,
                "capacity": c.capacity,
                "used": c.used,
                "utilization": round(c.utilization, 4),
                "servers": servers_data,
            })

        # Live metrics
        n_done = len(sim.finished)
        waits = [j.wait_time for j in sim.finished if j.wait_time is not None]
        avg_wait = (sum(waits) / len(waits)) if waits else 0.0
        sorted_waits = sorted(waits)
        p99_wait = (
            sorted_waits[int(0.99 * (len(sorted_waits) - 1))] if sorted_waits else 0.0
        )
        starved_3x = sum(1 for w in waits if w > 3.0 * self.w_ref)
        # Also include currently waiting jobs exceeding 3*W_ref
        waiting_starved = sum(
            1 for j in sim.queue if j.waited(sim.now) > 3.0 * self.w_ref
        )

        # Downsample timeline to max ~180 points for smooth fast frontend charts
        tl = sim.timeline
        if len(tl) > 180:
            stride = max(1, len(tl) // 180)
            tl_sampled = tl[::stride]
            if tl[-1] != tl_sampled[-1]:
                tl_sampled = list(tl_sampled) + [tl[-1]]
        else:
            tl_sampled = tl

        cap = state.total_capacity or 1
        timeline_data = [
            {
                "t": round(s.t, 1),
                "util": round(s.used_gpus / cap * 100.0, 2),
                "used_gpus": s.used_gpus,
                "lb_index": round(s.lb_index, 6),
                "queue_len": s.queue_len,
                "running": s.running,
            }
            for s in tl_sampled
        ]

        queue_data = [
            {
                "job_id": j.job_id,
                "gpus": j.gpus,
                "arrival_time": round(j.arrival_time, 1),
                "duration": round(j.duration, 1),
                "waited": round(j.waited(sim.now), 1),
                "wait_ratio": round(j.waited(sim.now) / self.w_ref, 2) if self.w_ref > 0 else 0.0,
                "protected": j.job_id in protected_ids,
                "starved": j.waited(sim.now) > 3.0 * self.w_ref,
            }
            for j in sim.queue[:35]
        ]

        running_data = [
            {
                "job_id": j.job_id,
                "gpus": j.gpus,
                "start_time": round(j.start_time or 0.0, 1),
                "end_time": round(j.end_time or 0.0, 1),
                "remaining": round(max(0.0, (j.end_time or 0.0) - sim.now), 1),
                "placement": list(j.placement) if j.placement else None,
            }
            for j in list(sim.running.values())[:35]
        ]

        summary = sim.result().summary if self.done else None
        if summary is not None and sim.finished:
            summary = dict(summary)
            summary.update(fairness_metrics(sim.finished, w_ref=self.w_ref))

        return {
            "config": self.config,
            "done": self.done,
            "step_count": self.step_count,
            "now": round(sim.now, 2),
            "total_jobs": self.total_jobs,
            "w_ref": round(self.w_ref, 2),
            "metrics": {
                "completed": n_done,
                "running": len(sim.running),
                "waiting": len(sim.queue),
                "rejected": len(sim.rejected),
                "used_gpus": state.used_gpus,
                "free_gpus": state.free_gpus,
                "total_gpus": state.total_capacity,
                "stranded_gpus": stranded_gpus,
                "utilization_pct": round(state.used_gpus / cap * 100.0, 2),
                "lb_index": round(state.load_balance_index(), 6),
                "avg_wait": round(avg_wait, 1),
                "p99_wait": round(p99_wait, 1),
                "starved_completed": starved_3x,
                "starved_waiting": waiting_starved,
                "protected_count": len(protected_ids),
                "reserved_count": len(reserved_by_job),
            },
            "clusters": clusters_data,
            "queue": queue_data,
            "running_jobs": running_data,
            "timeline": timeline_data,
            "protected_ids": protected_ids,
            "reserved_servers": {str(k): list(v) for k, v in reserved_by_job.items()},
            "n1_log": n1_log,
            "recent_events": list(reversed(self.recent_events[-25:])),
            "last_decision": self.last_decision,
            "summary": summary,
        }


def run_comparison(
    trace_id: str,
    topo_id: str,
    sched_ids: List[str],
    n_jobs: int = 400,
    eta: float = 10.0,
    burst: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run multiple schedulers side-by-side on identical workload & topology."""
    jobs = load_workload(trace_id, n_jobs=n_jobs, eta=eta, burst=burst, seed=seed)
    state = load_topology(topo_id)
    durs = [j.duration for j in jobs]
    w_ref = (sum(durs) / len(durs)) if durs else 1.0

    rows = []
    timelines = {}
    for sid in sched_ids:
        res = simulate(jobs, state, get_scheduler(sid))
        s = dict(res.summary)
        fm = fairness_metrics(res.finished, w_ref=w_ref)
        s.update(fm)
        info = next((item for item in SCHEDULER_CATALOG if item["id"] == sid), None)

        # Downsample timeline
        tl = res.timeline
        stride = max(1, len(tl) // 100) if len(tl) > 100 else 1
        cap = state.total_capacity or 1
        timelines[sid] = [
            {
                "t": round(pt.t, 1),
                "util": round(pt.used_gpus / cap * 100.0, 2),
                "queue_len": pt.queue_len,
                "lb_index": round(pt.lb_index, 6),
            }
            for pt in tl[::stride]
        ]

        rows.append({
            "id": sid,
            "name": info["name"] if info else sid,
            "category": info["category"] if info else "Other",
            "avg_wait": round(float(s.get("avg_wait", 0.0)), 1),
            "p95_wait": round(float(s.get("p95_wait", 0.0)), 1),
            "p99_wait": round(float(s.get("p99_wait", 0.0)), 1),
            "max_wait": round(float(s.get("max_wait", 0.0)), 1),
            "gpu_utilization": round(float(s.get("gpu_utilization", 0.0)) * 100.0, 2),
            "avg_lb_index": round(float(s.get("avg_lb_index", 0.0)), 6),
            "unalloc_ratio": round(float(s.get("avg_unallocated_ratio_during_allocation", 0.0)) * 100.0, 2),
            "starved_3x": int(s.get("starved_3x", 0)),
            "avg_wait_8gpu": round(float(s.get("avg_wait_8gpu", 0.0)), 1),
            "avg_wait_small": round(float(s.get("avg_wait_small", 0.0)), 1),
            "avg_wait_by_gpus": {
                str(k): round(float(v), 1)
                for k, v in sorted((s.get("avg_wait_by_gpus") or {}).items(), key=lambda x: int(x[0]))
            },
            "jobs_completed": int(s.get("jobs_completed", 0)),
            "makespan": round(float(s.get("makespan", 0.0)), 1),
        })

    return {
        "meta": {
            "trace_id": trace_id,
            "topo_id": topo_id,
            "n_jobs": len(jobs),
            "eta": eta,
            "burst": burst,
            "w_ref": round(w_ref, 1),
            "total_gpus": state.total_capacity,
        },
        "results": rows,
        "timelines": timelines,
    }


def inspect_n2_fragmentation(
    demand_weights: Optional[Dict[int, float]] = None,
    job_gpus: int = 2,
) -> Dict[str, Any]:
    """Compare Original F(r) vs Capacity-Aware F_C(r) across 4-GPU and 8-GPU servers."""
    if not demand_weights:
        demand_weights = {1: 0.45, 2: 0.25, 4: 0.15, 8: 0.15}

    # Build synthetic job list matching the demand weights
    jobs: List[Job] = []
    jid = 0
    for d, w in demand_weights.items():
        count = max(1, int(round(w * 100)))
        for _ in range(count):
            jobs.append(Job(job_id=jid, arrival_time=0.0, gpus=int(d), duration=100.0))
            jid += 1

    m_orig = DemandModel(lam=1.0, use_queue=False)
    m_orig.fit(jobs, 8)
    m_fc = CapacityAwareDemandModel(lam=1.0, use_queue=False, renormalize=False)
    m_fc.fit(jobs, 8)
    m_fcn = CapacityAwareDemandModel(lam=1.0, use_queue=False, renormalize=True)
    m_fcn.fit(jobs, 8)

    table_8g = []
    for r in range(9):
        table_8g.append({
            "free": r,
            "F_orig": round(m_orig.F(r, job_gpus), 4),
            "F_C": round(m_fc._capacity_table(job_gpus, 8)[r], 4),
            "F_C_norm": round(m_fcn._capacity_table(job_gpus, 8)[r], 4),
        })

    table_4g = []
    for r in range(5):
        table_4g.append({
            "free": r,
            "F_orig": round(m_orig.F(r, job_gpus), 4),
            "F_C": round(m_fc._capacity_table(job_gpus, 4)[r], 4),
            "F_C_norm": round(m_fcn._capacity_table(job_gpus, 4)[r], 4),
        })

    # Show gradient comparison for placing `job_gpus` on empty 4-GPU vs empty 8-GPU server
    g_4_orig = m_orig.gradient(4, job_gpus) if 4 >= job_gpus else None
    g_4_fc = m_fc.gradient_for_server(4, job_gpus, 4) if 4 >= job_gpus else None
    g_4_fcn = m_fcn.gradient_for_server(4, job_gpus, 4) if 4 >= job_gpus else None

    g_8_orig = m_orig.gradient(8, job_gpus) if 8 >= job_gpus else None
    g_8_fc = m_fc.gradient_for_server(8, job_gpus, 8) if 8 >= job_gpus else None
    g_8_fcn = m_fcn.gradient_for_server(8, job_gpus, 8) if 8 >= job_gpus else None

    return {
        "job_gpus": job_gpus,
        "demand_weights": {str(k): v for k, v in sorted(demand_weights.items())},
        "table_8g": table_8g,
        "table_4g": table_4g,
        "empty_gradients": {
            "server_4g": {
                "G_orig": round(g_4_orig, 4) if g_4_orig is not None else None,
                "G_fc": round(g_4_fc, 4) if g_4_fc is not None else None,
                "G_fc_norm": round(g_4_fcn, 4) if g_4_fcn is not None else None,
            },
            "server_8g": {
                "G_orig": round(g_8_orig, 4) if g_8_orig is not None else None,
                "G_fc": round(g_8_fc, 4) if g_8_fc is not None else None,
                "G_fc_norm": round(g_8_fcn, 4) if g_8_fcn is not None else None,
            },
        },
    }


def run_live_e5(trace_id: str = "venus", n_jobs: int = 500, eta: float = 15.0) -> Dict[str, Any]:
    """Run live E5 experiment (Homogeneous vs Mixed, F vs F_C) on a configurable slice."""
    jobs = load_workload(trace_id, n_jobs=n_jobs, eta=eta, burst=False, seed=0)
    durs = [j.duration for j in jobs]
    w_ref = (sum(durs) / len(durs)) if durs else 1.0

    configs = [
        ("homo_fcfs", "venus15", "fcfs", "Homogeneous 8-GPU"),
        ("homo_fgd", "venus15", "fgd", "Homogeneous 8-GPU"),
        ("homo_mcg", "venus15", "mcg", "Homogeneous 8-GPU"),
        ("mixed_fcfs", "venus15_mixed", "fcfs", "Mixed 4+8-GPU"),
        ("mixed_fgd", "venus15_mixed", "fgd", "Mixed 4+8-GPU"),
        ("mixed_mcg", "venus15_mixed", "mcg", "Mixed 4+8-GPU"),
        ("mixed_mcg_fc", "venus15_mixed", "mcg_fc", "Mixed + F_C"),
        ("mixed_mcg_fc_norm", "venus15_mixed", "mcg_fc_norm", "Mixed + F_C Norm"),
    ]
    rows = []
    for run_id, topo, sched, group in configs:
        state = load_topology(topo)
        res = simulate(jobs, state, get_scheduler(sched))
        s = dict(res.summary)
        s.update(fairness_metrics(res.finished, w_ref=w_ref))
        rows.append({
            "id": run_id,
            "group": group,
            "scheduler": sched,
            "topology": topo,
            "avg_wait": round(float(s.get("avg_wait", 0.0)), 1),
            "p99_wait": round(float(s.get("p99_wait", 0.0)), 1),
            "avg_wait_8gpu": round(float(s.get("avg_wait_8gpu", 0.0)), 1),
            "gpu_utilization": round(float(s.get("gpu_utilization", 0.0)) * 100.0, 2),
            "avg_lb_index": round(float(s.get("avg_lb_index", 0.0)), 6),
            "unalloc_ratio": round(float(s.get("avg_unallocated_ratio_during_allocation", 0.0)) * 100.0, 2),
        })
    return {
        "trace_id": trace_id,
        "n_jobs": len(jobs),
        "eta": eta,
        "rows": rows,
    }


def load_saved_experiments() -> Dict[str, Any]:
    """Load precomputed full-trace experimental outputs from results/*.json."""
    out: Dict[str, Any] = {}
    mapping = {
        "e4": RESULTS_DIR / "e4_fairness_protection.json",
        "e5": RESULTS_DIR / "e5_heterogeneous.json",
        "e7_norm": RESULTS_DIR / "e7_score_normalisation.json",
        "e7_scale": RESULTS_DIR / "e7_score_scale.json",
    }
    for key, path in mapping.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                out[key] = json.load(fh)
    return out
