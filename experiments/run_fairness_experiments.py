"""Fairness (N1) and score-normalisation (N4) experiments: E4 (aging / guard / drain reservation) and E7 (w sweep, raw vs normalised).

    python experiments/run_fairness_experiments.py --quick          # small slice, ~1 min, for checking
    python experiments/run_fairness_experiments.py --exp e4 --workers 4
    python experiments/run_fairness_experiments.py --exp e7 --workers 4

Writes results/e4_fairness_protection.json, e7_score_normalisation.json and a Markdown summary in results/E4_E7_fairness_results.md.
Every scheduler in one comparison gets identical jobs and topology.
W_ref = mean job duration of the trace, the same for every scheduler, so starvation counts compare.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcgsched.evaluation import fairness_metrics  # noqa: E402
from mcgsched.models import ClusterState  # noqa: E402
from mcgsched.schedulers import get_scheduler  # noqa: E402
from mcgsched.schedulers.mcg_fair import MCGFair  # noqa: E402
from mcgsched.simulation import simulate  # noqa: E402
from mcgsched.workloads import compress_arrivals, load_jobs_csv  # noqa: E402

TRACES = {
    "venus": ("configs/venus15.json", "data/clean/venus.csv"),
    "philly": (None, "data/clean/philly.csv"),
}
KEYS = ["avg_wait", "p95_wait", "p99_wait", "max_wait", "gpu_utilization",
        "avg_lb_index", "avg_unallocated_ratio_during_allocation", "makespan"]


def philly_topology() -> str:
    # The paper's Philly runs use the same 15-cluster Venus virtual-cluster setup.
    return "configs/venus15.json"


def load(trace: str, eta: float, limit):
    topo, csv = TRACES[trace]
    topo = topo or philly_topology()
    jobs = load_jobs_csv(os.path.join(ROOT, csv))
    if limit:
        jobs = jobs[:limit]
    jobs = compress_arrivals(jobs, eta)
    return ClusterState.from_json(os.path.join(ROOT, topo)), jobs


def one_run(task):
    """task = (label, trace, eta, limit, scheduler_name, kwargs). Runs in a worker process."""
    label, trace, eta, limit, name, kwargs = task
    state, jobs = load(trace, eta, limit)
    w_ref = sum(j.duration for j in jobs) / len(jobs)
    kwargs = dict(kwargs)
    sched = get_scheduler(name, **kwargs)
    res = simulate(jobs, state, sched)
    out = {k: res.summary[k] for k in KEYS}
    out["avg_wait_by_gpus"] = {str(g): v for g, v in res.summary["avg_wait_by_gpus"].items()}
    out.update(fairness_metrics(res.finished, w_ref, (1, 3)))
    out.update(label=label, trace=trace, eta=eta, scheduler=name, kwargs=kwargs, w_ref=w_ref,
               jobs=len(res.finished))
    if isinstance(sched, MCGFair):
        out["protected_jobs"] = len(sched.protected_ids)
        out["reservations_made"] = sum(1 for e in sched.log if e[1] == "reserve")
    return out


class Probe(MCGFair):
    """Records raw G and sigma^2 (N4 scale check): a 2% sample of all candidate placements,
    and the G / sigma^2 of the placement actually chosen by the raw score."""

    name = "probe"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.g, self.s, self.cg, self.cs = [], [], [], []
        self._rng = random.Random(0)

    def _select_ex(self, job, plan, lb, excluded):
        best, bv = None, None
        for g, sg, c, s in self._candidates(job.gpus, plan, lb, excluded):
            if self._rng.random() < 0.02:
                self.g.append(g)
                self.s.append(sg)
            v = (1 - self.w) * g + self.w * sg
            if bv is None or v < bv:
                bv, best = v, (g, sg, c, s)
        if best is None:
            return None
        self.cg.append(best[0])
        self.cs.append(best[1])
        return (bv, best[2], best[3])


def _stats(g, s):
    med = lambda v: sorted(v)[len(v) // 2] if v else 0.0
    ag = [abs(x) for x in g]
    return {"n": len(g), "median_abs_G": med(ag), "median_sigma2": med(s),
            "ratio": med(ag) / med(s) if med(s) else None}


def probe_run(task):
    trace, eta, limit = task
    state, jobs = load(trace, eta, limit)
    p = Probe()
    simulate(jobs, state, p)
    return {"trace": trace, "eta": eta, "all_candidates": _stats(p.g, p.s),
            "chosen_placements": _stats(p.cg, p.cs)}


def run_all(tasks, fn, workers):
    if workers <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, tasks))


def e4_tasks(limit, quick):
    t = []
    settings = [("venus", 15), ("venus", 10), ("philly", 4)] if not quick else [("venus", 15)]
    for trace, eta in settings:
        base = [("mcg", "mcg", {}), ("aging", "mcg_aging", {"beta": 1.0}),
                ("guard (blocking)", "mcg_guard", {"k": 3.0}),
                ("guard+reserve", "mcg_fair", {"beta": 0.0, "k": 3.0}),
                ("aging+guard+reserve", "mcg_fair", {"beta": 1.0, "k": 3.0})]
        for label, name, kw in base:
            t.append((label, trace, eta, limit, name, kw))
    # exploratory sweeps on Venus eta=15 (labelled exploratory in the report)
    if not quick:
        # second-trace check: k chosen on Venus is applied to Philly (its waits are far below 3 * W_ref)
        for k in (0.2, 0.5):
            t.append((f"philly k={k}", "philly", 4, limit, "mcg_fair", {"beta": 0.0, "k": k}))
        for k in (0.5, 1.0, 2.0, 5.0):
            t.append((f"sweep k={k}", "venus", 15, limit, "mcg_fair", {"beta": 0.0, "k": k}))
        for b in (0.5, 2.0, 4.0):
            t.append((f"sweep beta={b}", "venus", 15, limit, "mcg_fair", {"beta": b, "k": 3.0}))
        for m in (1, 2, 4, 8):
            t.append((f"sweep reservations={m}", "venus", 15, limit, "mcg_fair",
                      {"beta": 0.0, "k": 1.0, "max_reservations": m}))
    return t


def e7_tasks(limit, quick):
    ws = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] if not quick else [0.1, 0.5, 0.9]
    t = []
    for window in ("dynamic", 20):
        for norm in (False, True):
            for w in ws:
                t.append((f"{'norm' if norm else 'raw'} w={w} L={window}", "venus", 10, limit,
                          "mcg_norm" if norm else "mcg", {"w": w, "window": window}))
    return t


def fmt_table(rows, cols):
    head = "| run | " + " | ".join(c for c, _ in cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    lines = [head, sep]
    for r in rows:
        lines.append(f"| {r['label']} | " + " | ".join(f.format(r[c]) for c, f in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["e4", "e7", "all"], default="all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true", help="first 1500 jobs, fewer settings")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    limit = args.limit or (1500 if args.quick else None)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    md = ["# Fairness and normalisation results (E4 fairness, E7 score normalisation)", "",
          f"Jobs per run: {'first %d' % limit if limit else 'full trace'}. "
          "W_ref = mean job duration of the trace. Starved = wait > 3 * W_ref.", ""]
    cols = [("avg_wait", "{:.0f}"), ("p99_wait", "{:.0f}"), ("max_wait", "{:.0f}"),
            ("starved_3x", "{:.0f}"), ("avg_wait_8gpu", "{:.0f}"), ("gpu_utilization", "{:.4f}"),
            ("avg_unallocated_ratio_during_allocation", "{:.4f}"), ("avg_lb_index", "{:.5f}")]
    if args.exp in ("e4", "all"):
        rows = run_all(e4_tasks(limit, args.quick), one_run, args.workers)
        json.dump(rows, open(os.path.join(ROOT, "results/e4_fairness_protection.json"), "w"), indent=1)
        md += ["## E4: aging, guard, drain reservation", ""]
        for trace, eta in dict.fromkeys((r["trace"], r["eta"]) for r in rows):
            sub = [r for r in rows if (r["trace"], r["eta"]) == (trace, eta)
               and not r["label"].startswith(("sweep", "philly k="))]
            md += [f"### {trace} eta={eta}", "", fmt_table(sub, cols), ""]
        sw = [r for r in rows if r["label"].startswith("sweep")]
        ph = [r for r in rows if r["label"].startswith("philly k=")]
        if ph:
            md += ["### Philly eta=4 with lower k (the default k=3 never triggers there)", "",
                   fmt_table([r for r in rows if r["trace"] == "philly" and r["label"] == "mcg"] + ph, cols), ""]
        if sw:
            md += ["### Exploratory sweeps (venus eta=15; chosen on the same trace, so not an unbiased test)", "",
                   fmt_table(sw, cols), ""]
    if args.exp in ("e7", "all"):
        rows = run_all(e7_tasks(limit, args.quick), one_run, args.workers)
        json.dump(rows, open(os.path.join(ROOT, "results/e7_score_normalisation.json"), "w"), indent=1)
        probes = run_all([("venus", 10, limit), ("venus", 15, limit)], probe_run, 1)
        json.dump(probes, open(os.path.join(ROOT, "results/e7_score_scale.json"), "w"), indent=1)
        md += ["## E7: w sweep, raw vs normalised score (venus eta=10)", "",
               fmt_table(rows, cols), "", "### Scale of the two score terms (sampled candidates)", "",
               "| trace | eta | set | median abs G | median sigma^2 | ratio |", "|---|---|---|---|---|---|"]
        for p in probes:
            for which in ("all_candidates", "chosen_placements"):
                d = p[which]
                md.append(f"| {p['trace']} | {p['eta']} | {which} | {d['median_abs_G']:.4f} | "
                          f"{d['median_sigma2']:.4f} | {(d['ratio'] or 0):.1f} |")
        md.append("")
    open(os.path.join(ROOT, "results/E4_E7_fairness_results.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
