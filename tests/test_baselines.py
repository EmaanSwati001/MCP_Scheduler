import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcgsched.models import ClusterState, Job
from mcgsched.schedulers import Consolidate, FGD, RandomPlacement
from mcgsched.simulation import simulate


def one_cluster(*capacities):
    return ClusterState.from_spec({"clusters": [{"capacities": list(capacities)}]})


def placements(jobs, state, scheduler):
    res = simulate(jobs, state, scheduler)
    return {j.job_id: j.placement for j in res.finished}


class HandCalculated(unittest.TestCase):
    def test_consolidate_picks_tightest_fit(self):
        # Servers with 4 and 8 free GPUs. A 2-GPU job goes to the 4-GPU server.
        jobs = [Job(0, 0, 2, 10)]
        self.assertEqual(placements(jobs, one_cluster(4, 8), Consolidate()), {0: (0, 0)})

    def test_fgd_hand_example(self):
        # Sizes: 2 GPUs and 8 GPUs, each half of the jobs. So p2 = p8 = 0.5.
        # F(r) = r * (sum of p_d for d > r):
        #   F(2)=1, F(4)=2, F(6)=3, F(8)=0.
        # Server A (4 free): placing 2 gives F(2)-F(4) = -1.
        # Server B (8 free): placing 2 gives F(6)-F(8) = +3.
        # FGD picks A (server 0). Job 1 (8 GPUs) then needs B, the only empty server.
        jobs = [Job(0, 0, 2, 10), Job(1, 0, 8, 10)]
        self.assertEqual(
            placements(jobs, one_cluster(4, 8), FGD()),
            {0: (0, 0), 1: (0, 1)},
        )

    def test_no_head_of_line_blocking(self):
        # A 8-GPU job cannot fit on a 4-GPU server, but the 2-GPU job behind it still runs.
        jobs = [Job(0, 0, 8, 10), Job(1, 0, 2, 10)]
        res = simulate(jobs, one_cluster(4), Consolidate())
        self.assertEqual({j.job_id for j in res.finished}, {1})
        self.assertEqual(res.summary["jobs_rejected"], 1)  # bigger than any server


class Determinism(unittest.TestCase):
    def test_random_same_seed_same_result(self):
        jobs = [Job(i, float(i), 1 + i % 4, 50) for i in range(40)]
        state = one_cluster(4, 4, 8, 8)
        first = placements(jobs, state, RandomPlacement(seed=7))
        second = placements(jobs, state, RandomPlacement(seed=7))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()