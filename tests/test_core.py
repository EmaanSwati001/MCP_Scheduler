import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcgsched.models import ClusterState, Job
from mcgsched.schedulers import FCFSFirstFit, Placement, Scheduler
from mcgsched.simulation import SchedulerError, percentile, simulate
from mcgsched.workloads import (
    compress_arrivals,
    load_jobs_csv,
    make_burst,
    save_jobs_csv,
    synthetic_jobs,
)


def topo(*clusters):
    """topo([4], [4, 8]) -> two clusters with those server capacities."""
    return ClusterState.from_spec({"clusters": [{"capacities": list(c)} for c in clusters]})


TOY = {"clusters": [{"count": 2, "servers": 2, "gpus_per_server": 4}]}


class HandCalculated(unittest.TestCase):
    def test_fcfs_single_server(self):
        # One 4-GPU server. B blocks C although C would fit (strict FCFS).
        jobs = [Job(0, 0, 2, 10), Job(1, 1, 4, 5), Job(2, 2, 2, 3)]
        res = simulate(jobs, topo([4]), FCFSFirstFit())
        by_id = {j.job_id: j for j in res.finished}
        self.assertEqual([by_id[i].start_time for i in range(3)], [0, 10, 15])
        self.assertEqual([by_id[i].end_time for i in range(3)], [10, 15, 18])
        s = res.summary
        self.assertAlmostEqual(s["avg_wait"], (0 + 9 + 13) / 3)
        self.assertEqual(s["max_wait"], 13)
        self.assertEqual(s["makespan"], 18)
        # GPU-seconds: 2*10 + 4*5 + 2*3 = 46 of 4*18 = 72
        self.assertAlmostEqual(s["gpu_utilization"], 46 / 72)
        self.assertEqual(s["avg_wait_by_gpus"], {2: 6.5, 4: 9.0})

    def test_completion_frees_gpus_for_same_instant_arrival(self):
        jobs = [Job(0, 0, 1, 5), Job(1, 5, 1, 1)]
        res = simulate(jobs, topo([1]), FCFSFirstFit())
        self.assertEqual({j.job_id: j.wait_time for j in res.finished}, {0: 0, 1: 0})

    def test_load_balance_index(self):
        state = topo([4], [4])
        self.assertEqual(state.load_balance_index(), 0.0)
        self.assertAlmostEqual(state.lb_index_after(0, 4), 0.25)  # utils (1, 0)
        state.allocate(7, 4, 0, 0)
        self.assertAlmostEqual(state.load_balance_index(), 0.25)
        self.assertAlmostEqual(state.lb_index_after(1, 4), 0.0)   # utils (1, 1)
        state.release(7, 0, 0)
        self.assertEqual(state.free_gpus, 8)

    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(percentile([0, 10], 95), 9.5)


class Invariants(unittest.TestCase):
    def setUp(self):
        self.jobs = synthetic_jobs(
            300, seed=3, mean_interarrival=30, mean_duration=600, demand={1: 0.5, 2: 0.3, 4: 0.2}
        )
        self.state = ClusterState.from_spec(TOY)
        self.res = simulate(self.jobs, self.state, FCFSFirstFit())

    def test_every_job_runs_once_and_not_before_arrival(self):
        self.assertEqual(sorted(j.job_id for j in self.res.finished), list(range(300)))
        self.assertEqual(self.res.unscheduled, [])
        for j in self.res.finished:
            self.assertGreaterEqual(j.start_time, j.arrival_time)
            self.assertAlmostEqual(j.end_time - j.start_time, j.duration)

    def test_no_server_is_ever_over_capacity(self):
        events = {}
        for j in self.res.finished:
            events.setdefault(j.placement, []).append((j.start_time, 1, j.gpus))
            events.setdefault(j.placement, []).append((j.end_time, 0, -j.gpus))
        for (cid, sid), evs in events.items():
            cap, used = self.state.server(cid, sid).capacity, 0
            for _, _, delta in sorted(evs):  # releases sort before starts at equal t
                used += delta
                self.assertLessEqual(used, cap)
                self.assertGreaterEqual(used, 0)
        self.assertTrue(all(0 <= s.used_gpus <= 16 for s in self.res.timeline))
        self.assertEqual(self.res.timeline[-1].used_gpus, 0)

    def test_inputs_untouched_and_run_is_deterministic(self):
        self.assertTrue(all(j.start_time is None for j in self.jobs))
        self.assertEqual(self.state.used_gpus, 0)
        again = simulate(self.jobs, self.state, FCFSFirstFit())
        self.assertEqual(again.summary, self.res.summary)
        self.assertEqual(again.finished, self.res.finished)

    def test_oversize_job_is_rejected_not_stuck(self):
        jobs = [Job(0, 0, 8, 5), Job(1, 1, 2, 5)]
        res = simulate(jobs, self.state, FCFSFirstFit())
        self.assertEqual([j.job_id for j in res.rejected], [0])
        self.assertEqual([j.job_id for j in res.finished], [1])
        self.assertEqual(res.finished[0].wait_time, 0)


class BadSchedulers(unittest.TestCase):
    def _run(self, placements):
        class Bad(Scheduler):
            name = "bad"

            def schedule(self, queue, state, now):
                return placements

        return simulate([Job(0, 0, 4, 5), Job(1, 0, 4, 5)], topo([4]), Bad())

    def test_overallocation_raises(self):
        with self.assertRaises(SchedulerError):
            self._run([Placement(0, 0, 0), Placement(1, 0, 0)])

    def test_placing_a_job_twice_raises(self):
        with self.assertRaises(SchedulerError):
            self._run([Placement(0, 0, 0), Placement(0, 0, 0)])

    def test_unknown_server_raises(self):
        with self.assertRaises(SchedulerError):
            self._run([Placement(0, 3, 0)])

    def test_scheduler_that_never_places_leaves_jobs_unscheduled(self):
        res = self._run([])
        self.assertEqual(len(res.unscheduled), 2)
        self.assertEqual(res.summary["jobs_unscheduled"], 2)


class Workloads(unittest.TestCase):
    def test_compress_arrivals(self):
        jobs = [Job(0, 100, 1, 50), Job(1, 200, 1, 50), Job(2, 400, 1, 50)]
        out = compress_arrivals(jobs, 10)
        self.assertEqual([j.arrival_time for j in out], [100, 110, 130])
        self.assertEqual([j.duration for j in out], [50, 50, 50])
        self.assertEqual(jobs[1].arrival_time, 200)

    def test_burst(self):
        out = make_burst(synthetic_jobs(20, seed=1))
        self.assertEqual({j.arrival_time for j in out}, {0.0})

    def test_synthetic_is_seeded(self):
        self.assertEqual(synthetic_jobs(50, seed=9), synthetic_jobs(50, seed=9))
        self.assertNotEqual(synthetic_jobs(50, seed=9), synthetic_jobs(50, seed=10))

    def test_csv_roundtrip(self):
        jobs = synthetic_jobs(25, seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "jobs.csv")
            save_jobs_csv(jobs, path)
            self.assertEqual(load_jobs_csv(path), jobs)


if __name__ == "__main__":
    unittest.main()
