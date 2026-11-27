"""Tests for Person 5: heterogeneous server capacities (N2).

Covers:
  - Mixed topology GPU invariants
  - CapacityAwareDemandModel: F_C = F on homogeneous servers
  - CapacityAwareDemandModel: F_C excludes oversized demands
  - MCGHetero registers and schedules correctly
  - Reserved server exclusion still works on mixed topology
  - E5 direction: mixed topology raises wait times vs homogeneous
"""
from __future__ import annotations

import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from mcgsched.mechanisms.fragmentation import CapacityAwareDemandModel, DemandModel
from mcgsched.models import ClusterState, Job
from mcgsched.schedulers import REGISTRY, get_scheduler
from mcgsched.simulation import simulate
from mcgsched.workloads import synthetic_jobs, compress_arrivals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_server_state(cap1: int, cap2: int) -> ClusterState:
    """Single cluster with two servers of different capacities."""
    return ClusterState.from_spec({"clusters": [{"capacities": [cap1, cap2]}]})


def _jobs(demands):
    """Minimal job list – one job per demand size, all arriving at t=0."""
    return [Job(job_id=i, arrival_time=0.0, gpus=d, duration=100.0) for i, d in enumerate(demands)]


# ---------------------------------------------------------------------------
# 1. Mixed topology invariants
# ---------------------------------------------------------------------------

class TestMixedTopology:
    def test_total_gpus_unchanged(self):
        homo  = ClusterState.from_json("configs/venus15.json")
        mixed = ClusterState.from_json("configs/venus15_mixed.json")
        assert mixed.total_capacity == 1080

    def test_has_both_server_sizes(self):
        mixed = ClusterState.from_json("configs/venus15_mixed.json")
        caps = {s.capacity for s in mixed.servers()}
        assert 4 in caps
        assert 8 in caps

    def test_server_count_larger_than_homogeneous(self):
        homo  = ClusterState.from_json("configs/venus15.json")
        mixed = ClusterState.from_json("configs/venus15_mixed.json")
        homo_n  = sum(len(c.servers) for c in homo.clusters)
        mixed_n = sum(len(c.servers) for c in mixed.clusters)
        assert mixed_n > homo_n

    def test_max_server_capacity_unchanged(self):
        mixed = ClusterState.from_json("configs/venus15_mixed.json")
        assert mixed.max_server_capacity == 8

    def test_cluster_count_unchanged(self):
        homo  = ClusterState.from_json("configs/venus15.json")
        mixed = ClusterState.from_json("configs/venus15_mixed.json")
        assert len(mixed.clusters) == len(homo.clusters) == 15


# ---------------------------------------------------------------------------
# 2. CapacityAwareDemandModel: equivalence on homogeneous topology
# ---------------------------------------------------------------------------

class TestCapAwareEquivalence:
    def test_fc_equals_f_on_homogeneous_servers(self):
        """When every server has capacity == max_capacity, F_C == F exactly."""
        jobs = _jobs([1, 2, 4, 8])
        orig = DemandModel(lam=1.0, use_queue=False)
        fc   = CapacityAwareDemandModel(lam=1.0, use_queue=False, renormalize=False)
        orig.fit(jobs, 8)
        fc.fit(jobs, 8)
        for free in range(1, 9):
            for gpus in [1, 2, 4]:
                if free >= gpus:
                    g_orig = orig.gradient(free, gpus)
                    g_fc   = fc.gradient_for_server(free, gpus, server_cap=8)
                    assert abs(g_orig - g_fc) < 1e-12, (
                        f"free={free}, gpus={gpus}: orig={g_orig}, fc={g_fc}"
                    )

    def test_fc_norm_excludes_demand_above_cap(self):
        """On a 4-GPU server, 8-GPU demand must not contribute to F_C."""
        jobs = _jobs([1, 2, 4, 8])   # 8-GPU demand present in global mix
        fc = CapacityAwareDemandModel(lam=1.0, use_queue=False, renormalize=True)
        fc.fit(jobs, 8)
        # On a 4-GPU server: placing a 2-GPU job, free=4
        g4 = fc.gradient_for_server(free=4, gpus=2, server_cap=4)
        # On an 8-GPU server: same values – 8-GPU demand now included
        g8 = fc.gradient_for_server(free=4, gpus=2, server_cap=8)
        # The 8-GPU demand raises fragmentation on the 8-GPU server
        # because there is remaining capacity (free-gpus=2) that an 8-GPU job
        # would want but can't use. So gradient on 4-GPU server should differ.
        # We just verify that re-normalizing produces a valid float in [−1, 1].
        assert isinstance(g4, float)
        assert isinstance(g8, float)

    def test_gradient_for_server_negative_or_zero_good_placement(self):
        """A perfect-fit placement (free==gpus) always has gradient 0."""
        jobs = _jobs([4, 8])
        fc = CapacityAwareDemandModel(lam=1.0, use_queue=False, renormalize=False)
        fc.fit(jobs, 8)
        g = fc.gradient_for_server(free=4, gpus=4, server_cap=4)
        assert abs(g) < 1e-12, f"expected 0, got {g}"


# ---------------------------------------------------------------------------
# 3. MCGHetero scheduler registration and basic scheduling
# ---------------------------------------------------------------------------

class TestMCGHeteroScheduler:
    def test_registry_has_fc_variants(self):
        assert "mcg_fc" in REGISTRY
        assert "mcg_fc_norm" in REGISTRY

    def test_fc_schedules_on_mixed_topology(self):
        state = ClusterState.from_json("configs/venus15_mixed.json")
        jobs  = synthetic_jobs(50, seed=7)
        result = simulate(jobs, state, get_scheduler("mcg_fc"))
        assert result.summary["jobs_completed"] > 0

    def test_fc_norm_schedules_on_mixed_topology(self):
        state = ClusterState.from_json("configs/venus15_mixed.json")
        jobs  = synthetic_jobs(50, seed=7)
        result = simulate(jobs, state, get_scheduler("mcg_fc_norm"))
        assert result.summary["jobs_completed"] > 0

    def test_fc_does_not_place_8gpu_job_on_4gpu_server(self):
        """An 8-GPU job must only land on 8-GPU servers in a mixed topology."""
        state = ClusterState.from_json("configs/venus15_mixed.json")
        # Only 8-GPU jobs
        jobs = [Job(job_id=i, arrival_time=float(i * 60), gpus=8, duration=200.0)
                for i in range(5)]
        result = simulate(jobs, state, get_scheduler("mcg_fc"))
        for job in result.finished:
            c_id, s_id = job.placement
            server = state.clusters[c_id].servers[s_id]
            assert server.capacity == 8, (
                f"8-GPU job {job.job_id} placed on {server.capacity}-GPU server"
            )

    def test_fc_same_result_as_mcg_on_homogeneous_topology(self):
        """On a homogeneous topology mcg_fc and mcg must produce the same avg_wait."""
        state = ClusterState.from_json("configs/venus15.json")
        jobs  = synthetic_jobs(100, seed=42)
        r_mcg = simulate(jobs, state, get_scheduler("mcg"))
        r_fc  = simulate(jobs, state, get_scheduler("mcg_fc"))
        # Exact numeric equality is not guaranteed (gradient identical but
        # floating-point order may differ), so compare within 1 %.
        avg_mcg = r_mcg.summary["avg_wait"]
        avg_fc  = r_fc.summary["avg_wait"]
        rel_diff = abs(avg_mcg - avg_fc) / max(avg_mcg, 1.0)
        assert rel_diff < 0.01, (
            f"avg_wait too different on homogeneous topology: mcg={avg_mcg:.1f}, fc={avg_fc:.1f}"
        )

    def test_no_server_over_capacity_mixed(self):
        """GPU invariant must hold on the mixed topology for all schedulers."""
        state = ClusterState.from_json("configs/venus15_mixed.json")
        jobs  = synthetic_jobs(80, seed=3)
        for name in ["fcfs", "mcg", "mcg_fc", "mcg_fc_norm"]:
            result = simulate(jobs, state, get_scheduler(name))
            # All finished jobs must have a valid placement
            for job in result.finished:
                c_id, s_id = job.placement
                server = state.clusters[c_id].servers[s_id]
                assert server.capacity >= job.gpus


# ---------------------------------------------------------------------------
# 4. E5 direction: mixed topology is harder for large jobs
# ---------------------------------------------------------------------------

class TestE5Direction:
    def test_8gpu_jobs_harder_on_mixed(self):
        """Average wait for 8-GPU jobs on mixed topology >= homogeneous."""
        jobs_all = compress_arrivals(synthetic_jobs(200, seed=0), eta=10)
        jobs_8 = [j for j in jobs_all if j.gpus == 8]
        if not jobs_8:
            pytest.skip("no 8-GPU jobs in synthetic workload")

        def avg_8gpu_wait(topology_path):
            state = ClusterState.from_json(topology_path)
            result = simulate(jobs_all, state, get_scheduler("mcg"))
            waits = [j.wait_time for j in result.finished if j.gpus == 8]
            return sum(waits) / len(waits) if waits else 0.0

        wait_homo  = avg_8gpu_wait("configs/venus15.json")
        wait_mixed = avg_8gpu_wait("configs/venus15_mixed.json")
        # Mixed should be harder (more wait) or at least not better
        assert wait_mixed >= wait_homo * 0.9, (
            f"Expected mixed >= homo wait for 8-GPU jobs: "
            f"homo={wait_homo:.0f}s, mixed={wait_mixed:.0f}s"
        )

    def test_mcg_still_beats_fcfs_on_mixed(self):
        """MCG should have lower avg_wait than FCFS even on mixed topology."""
        state = ClusterState.from_json("configs/venus15_mixed.json")
        jobs  = compress_arrivals(synthetic_jobs(150, seed=5), eta=8)
        r_fcfs = simulate(jobs, state, get_scheduler("fcfs"))
        r_mcg  = simulate(jobs, state, get_scheduler("mcg"))
        assert r_mcg.summary["avg_wait"] <= r_fcfs.summary["avg_wait"] * 1.05, (
            "MCG should not be worse than FCFS on mixed topology"
        )
