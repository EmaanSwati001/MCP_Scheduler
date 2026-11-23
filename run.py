"""Command-line entry point and experiment runner.

    python run.py --topology configs/toy.json --scheduler fcfs --jobs 200
    python run.py --topology configs/paper_8x16.json --trace data/venus.csv --eta 10
    python run.py --topology configs/paper_8x16.json --scheduler all --jobs 800 --burst --out e2.json

Every scheduler named in one call gets the same jobs and the same topology.
"""
from __future__ import annotations

import argparse
import json

from mcgsched.models import ClusterState
from mcgsched.schedulers import REGISTRY, get_scheduler
from mcgsched.simulation import simulate
from mcgsched.workloads import (
    compress_arrivals,
    load_jobs_csv,
    make_burst,
    synthetic_jobs,
)

TABLE_COLUMNS = [
    ("avg_wait", "avg_wait", "{:.1f}"),
    ("p99_wait", "p99_wait", "{:.1f}"),
    ("max_wait", "max_wait", "{:.1f}"),
    ("gpu_utilization", "util", "{:.4f}"),
    ("avg_lb_index", "lb_index", "{:.5f}"),
    ("avg_unallocated_ratio_during_allocation", "unalloc", "{:.4f}"),
    ("jobs_completed", "done", "{}"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="MCG-Sched project simulator")
    ap.add_argument("--topology", default="configs/toy.json", help="topology JSON")
    ap.add_argument(
        "--scheduler",
        nargs="+",
        default=["all"],
        choices=sorted(REGISTRY) + ["all"],
        help="one or more policies, or 'all'",
    )
    ap.add_argument("--trace", help="job CSV (job_id,arrival_time,gpus,duration); default: synthetic")
    ap.add_argument("--jobs", type=int, help="use only the first N jobs (synthetic default: 200)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the synthetic workload")
    ap.add_argument("--eta", type=float, default=1.0, help="arrival-rate compression")
    ap.add_argument("--burst", action="store_true", help="make all jobs arrive at once")
    ap.add_argument("--out", help="write summaries and timelines to this JSON file")
    args = ap.parse_args()

    state = ClusterState.from_json(args.topology)
    if args.trace:
        jobs = load_jobs_csv(args.trace)
        if args.jobs:
            jobs = jobs[: args.jobs]
    else:
        jobs = synthetic_jobs(args.jobs or 200, seed=args.seed)
    if args.eta != 1.0:
        jobs = compress_arrivals(jobs, args.eta)
    if args.burst:
        jobs = make_burst(jobs)

    names = sorted(REGISTRY) if "all" in args.scheduler else list(dict.fromkeys(args.scheduler))
    results = {name: simulate(jobs, state, get_scheduler(name)) for name in names}

    print(f"jobs={len(jobs)}  clusters={len(state.clusters)}  gpus={state.total_capacity}")
    if len(names) == 1:
        for key, value in results[names[0]].summary.items():
            if isinstance(value, float):
                print(f"  {key:<42}{value:.4f}")
            else:
                print(f"  {key:<42}{value}")
    else:
        print(f"{'scheduler':<14}" + "".join(f"{label:>12}" for _, label, _ in TABLE_COLUMNS))
        for name in names:
            summary = results[name].summary
            cells = "".join(f"{fmt.format(summary[key]):>12}" for key, _, fmt in TABLE_COLUMNS)
            print(f"{name:<14}{cells}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "args": vars(args),
                    "results": {
                        name: {
                            "summary": res.summary,
                            "timeline": [list(s) for s in res.timeline],
                        }
                        for name, res in results.items()
                    },
                },
                fh,
                indent=1,
            )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
