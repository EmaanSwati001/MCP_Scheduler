import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcgsched.evaluation import fairness_metrics
from mcgsched.mechanisms.reservations import (
    ReservationTable,
    choose_drain_server,
    completions_needed,
)
from mcgsched.models import ClusterState, Job
from mcgsched.schedulers import REGISTRY, MCGSched, get_scheduler
from mcgsched.schedulers.mcg_fair import MCGFair, _norm
from mcgsched.simulation import Simulator, simulate
from mcgsched.workloads import synthetic_jobs


def topo(*clusters):
    return ClusterState.from_spec({"clusters": [{"capacities": list(c)} for c in clusters]})


# One cluster, two 4-GPU servers a=(0,0), b=(0,1).
#  J0, J1: 3 GPUs, 50 s at t=0 -> one per server.
#  J2:     4 GPUs, 10 s, arrives t=1: blocked, needs a whole server.
#  S3, S4: 1 GPU, 200 s, arrive t=5, 6 -> fill the last GPU of a then b.
#  S5:     1 GPU, 1000 s, arrives t=20.
# With W_ref = 10 and k = 1 the guard protects J2 once it has waited 10 s (first round at t=20).
SCENARIO = [
    Job(0, 0, 3, 50), Job(1, 0, 3, 50), Job(2, 1, 4, 10),
    Job(3, 5, 1, 200), Job(4, 6, 1, 200), Job(5, 20, 1, 1000),
]


def run(name_or_sched, jobs=SCENARIO):
    sched = get_scheduler(name_or_sched) if isinstance(name_or_sched, str) else name_or_sched
    sim = Simulator(jobs, topo([4, 4]), sched)
    res = sim.run()
    return {j.job_id: j for j in res.finished}, sched


class DrainChoice(unittest.TestCase):
    def test_completions_needed(self):
        self.assertEqual(completions_needed(4, {}, 4), 0)
        self.assertEqual(completions_needed(2, {1: 4, 2: 2}, 8), 2)   # 2+4 = 6, +2 = 8
        self.assertEqual(completions_needed(0, {1: 1, 2: 1}, 4), 10 ** 9)

    def test_picks_fewest_completions_and_skips_small_or_taken(self):
        state = topo([8, 8, 4])
        state.allocate(1, 4, 0, 0)
        state.allocate(2, 4, 0, 0)       # s0: free 0, running {4, 4} -> 2 completions
        state.allocate(3, 6, 0, 1)       # s1: free 2, running {6}    -> 1 completion
        job = Job(9, 0, 8, 1)
        self.assertEqual(choose_drain_server(job, state, []), (0, 1))
        self.assertEqual(choose_drain_server(job, state, [(0, 1)]), (0, 0))   # taken is skipped
        self.assertIsNone(choose_drain_server(Job(10, 0, 8, 1), topo([4, 4]), []))

    def test_table(self):
        t = ReservationTable()
        t.add(1, (0, 0))
        with self.assertRaises(ValueError):
            t.add(2, (0, 0))
        self.assertEqual(t.others(2), {(0, 0)})
        self.assertEqual(t.others(1), set())
        t.keep_only([2])
        self.assertEqual(len(t), 0)


class HandCalculated(unittest.TestCase):
    def test_plain_mcg_lets_small_jobs_pass(self):
        by_id, _ = run("mcg")
        self.assertEqual(by_id[5].start_time, 50)      # S5 backfills at t=50 and takes server a
        self.assertEqual(by_id[2].start_time, 206)     # J2 waits for S4 on server b to end

    def test_guard_plus_reservation(self):
        sched = get_scheduler("mcg_fair", w_ref=10.0, k=1.0)
        by_id, sched = run(sched)
        # t=20: J2 waited 19 >= 10 -> protected, no server can start it, reserve a (needs 2 completions
        # on both servers, tie -> lower id). S5 may not use a, so it takes b at t=50 (b has 3 free).
        self.assertEqual(by_id[5].start_time, 50)
        self.assertEqual(by_id[5].placement, (0, 1))
        # J2 starts the moment S3 (the last job on a) ends: 5 + 200.
        self.assertEqual(by_id[2].start_time, 205)
        self.assertEqual(by_id[2].placement, (0, 0))
        kinds = [(t, e, j) for t, e, j, _ in sched.log]
        self.assertEqual(kinds, [(20, "protect", 2), (20, "reserve", 2), (50, "protect", 5), (205, "release", 2)])   # S5 has waited 30 >= 10 by t=50, starts at once
        self.assertEqual(sched.reserved_servers(), {})

    def test_blocking_guard_stalls_everything_behind(self):
        sched = get_scheduler("mcg_guard", w_ref=10.0, k=1.0)
        by_id, _ = run(sched)
        self.assertEqual(by_id[2].start_time, 205)
        self.assertEqual(by_id[5].start_time, 205)     # S5 was held back 155 s for nothing

    def test_reservation_beats_blocking_on_average_wait(self):
        a, _ = run(get_scheduler("mcg_fair", w_ref=10.0, k=1.0))
        b, _ = run(get_scheduler("mcg_guard", w_ref=10.0, k=1.0))
        wa = sum(j.wait_time for j in a.values())
        wb = sum(j.wait_time for j in b.values())
        self.assertLess(wa, wb)


class Equivalence(unittest.TestCase):
    def test_defaults_match_mcg(self):
        state = topo([8, 8, 4, 4], [8, 4], [8, 8])
        jobs = synthetic_jobs(400, seed=3)
        for window in ("dynamic", 20, 0):
            base = simulate(jobs, state, MCGSched(window=window))
            ext = simulate(jobs, state, MCGFair(window=window))
            self.assertEqual(
                [(j.job_id, j.start_time, j.placement) for j in base.finished],
                [(j.job_id, j.start_time, j.placement) for j in ext.finished],
                f"window={window}",
            )

    def test_registry_names(self):
        for name in ("mcg_aging", "mcg_guard", "mcg_fair", "mcg_norm"):
            self.assertIn(name, REGISTRY)
            self.assertEqual(get_scheduler(name).name, name)


class Invariants(unittest.TestCase):
    def test_reserved_server_never_taken_by_others(self):
        state = topo([8, 8, 4, 4], [8, 4], [8, 8, 8])
        jobs = synthetic_jobs(600, seed=7)
        from mcgsched.workloads import compress_arrivals
        jobs = compress_arrivals(jobs, 40)
        sched = get_scheduler("mcg_fair", k=0.5, max_reservations=2)
        res = Simulator(jobs, state, sched).run()
        self.assertEqual(len(res.finished) + len(res.rejected), len(jobs))
        self.assertTrue(any(e == "reserve" for _, e, _, _ in sched.log), "scenario never reserved")
        # reservation intervals per server
        open_, spans = {}, []
        for t, e, jid, key in sched.log:
            if e == "reserve":
                open_[jid] = (t, key)
            elif e == "release":
                t0, k0 = open_.pop(jid)
                spans.append((jid, k0, t0, t))
        self.assertFalse(open_)
        for holder, key, t0, t1 in spans:
            for j in res.finished:
                if j.job_id != holder and j.placement == key:
                    self.assertFalse(t0 < j.start_time < t1,
                                     f"job {j.job_id} started on reserved {key} at {j.start_time}")

    def test_max_reservations_respected(self):
        state = topo([8, 8], [8, 8])
        from mcgsched.workloads import compress_arrivals
        jobs = compress_arrivals(synthetic_jobs(300, seed=1), 60)
        sched = get_scheduler("mcg_fair", k=0.2, max_reservations=1)
        seen = []
        sim = Simulator(jobs, state, sched)
        while sim.step():
            seen.append(len(sched.table))
        self.assertLessEqual(max(seen), 1)


class AgingAndNorm(unittest.TestCase):
    def test_bonus_is_capped(self):
        s = MCGFair(beta=2.0, aging_cap=5.0, w_ref=100.0)
        j = Job(0, 0, 1, 1)
        self.assertAlmostEqual(s._bonus(j, 50), 1.0)             # 2 * 0.5
        self.assertAlmostEqual(s._bonus(j, 10_000), 10.0)        # capped at 2 * 5
        self.assertEqual(MCGFair()._bonus(j, 10_000), 0.0)

    def test_w_ref_defaults_to_mean_duration(self):
        s = MCGFair()
        s.setup([Job(0, 0, 1, 10), Job(1, 0, 1, 30)], topo([8]))
        self.assertEqual(s.w_ref, 20.0)
        s.setup([Job(0, 0, 1, 0)], topo([8]))
        self.assertEqual(s.w_ref, 1.0)                           # zero fallback

    def test_minmax(self):
        self.assertEqual(_norm(5, 5, 5), 0.0)
        self.assertEqual(_norm(2, 0, 4), 0.5)

    def test_normalised_scores_in_unit_range(self):
        state = topo([8, 8, 4], [8, 4])
        state.allocate(1, 3, 0, 0)
        s = get_scheduler("mcg_norm")
        s.setup([Job(i, 0, g, 5) for i, g in enumerate([1, 2, 4, 8, 2, 1])], state)
        queue = [Job(10, 0, 2, 5), Job(11, 0, 4, 5), Job(12, 0, 1, 5)]
        s.model.set_queue(queue)
        s._sizes = [1, 2, 4]
        s._invalidate()
        from mcgsched.mechanisms.load_balance import LoadBalance
        lb = LoadBalance(state)
        for j in queue:
            v, _, _ = s._select(j, state, lb)
            self.assertGreaterEqual(v, -1e-12)
            self.assertLessEqual(v, 1 + 1e-12)

    def test_norm_and_aging_runs_complete(self):
        state = topo([8, 8, 4, 4], [8, 4])
        jobs = synthetic_jobs(300, seed=5)
        for name in ("mcg_norm", "mcg_aging", "mcg_guard", "mcg_fair"):
            res = simulate(jobs, state, get_scheduler(name))
            self.assertEqual(len(res.finished) + len(res.rejected), len(jobs), name)
            self.assertEqual(res.unscheduled, [], name)


class Metrics(unittest.TestCase):
    def test_fairness_metrics(self):
        jobs = []
        for i, (g, wait) in enumerate([(8, 50), (8, 5), (1, 0), (2, 25)]):
            j = Job(i, 0, g, 1)
            j.start_time = wait
            jobs.append(j)
        m = fairness_metrics(jobs, w_ref=10, multiples=(1, 3))
        self.assertEqual(m["starved_1x"], 2)      # 50 and 25
        self.assertEqual(m["starved_3x"], 1)      # 50
        self.assertAlmostEqual(m["avg_wait_8gpu"], 27.5)
        self.assertAlmostEqual(m["avg_wait_small"], 12.5)


if __name__ == "__main__":
    unittest.main()
