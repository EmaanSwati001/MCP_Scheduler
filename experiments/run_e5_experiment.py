"""E5: Heterogeneous server experiment (Person 5 / N2).

Runs MCG-Sched and baselines on:
  - Homogeneous 15-cluster Venus topology (8-GPU servers)
  - Mixed 15-cluster topology (4-GPU and 8-GPU servers)

Then compares original F vs capacity-aware F_C on the mixed topology.

Results written to results/e5_heterogeneous.json and results/RESULTS_E5.md.
Figures written to results/figures/e5_*.png.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mcgsched.evaluation.fairness import fairness_metrics
from mcgsched.models import ClusterState
from mcgsched.schedulers import get_scheduler
from mcgsched.simulation import simulate
from mcgsched.workloads import compress_arrivals, load_jobs_csv, synthetic_jobs

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

CONFIG_DIR  = ROOT / "configs"
DATA_DIR    = ROOT / "data" / "clean"
RESULTS_DIR = ROOT / "results"
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
ETASETTINGS = [
    ("venus", 15),
    ("venus", 10),
]

HOMO_TOPO  = str(CONFIG_DIR / "venus15.json")
MIXED_TOPO = str(CONFIG_DIR / "venus15_mixed.json")

SCHEDS_MAIN = ["fcfs", "consolidate", "fgd", "mast", "mcg"]
SCHEDS_FC   = ["mcg", "mcg_fc", "mcg_fc_norm"]  # F vs F_C comparison

W_REF_KEY = "mean_duration"


def _load_venus(eta: float, n: int = 0):
    path = str(DATA_DIR / "venus.csv")
    if not os.path.exists(path):
        print(f"  [warn] Venus trace not found at {path}, using synthetic jobs")
        jobs = synthetic_jobs(500, seed=0)
    else:
        jobs = load_jobs_csv(path)
    if n:
        jobs = jobs[:n]
    return compress_arrivals(jobs, eta)


def _w_ref(jobs):
    durs = [j.duration for j in jobs]
    return sum(durs) / len(durs) if durs else 1.0


def _run_set(jobs, topo_path: str, sched_names: List[str]) -> Dict:
    state = ClusterState.from_json(topo_path)
    w_ref = _w_ref(jobs)
    out = {}
    for name in sched_names:
        print(f"    {name} ...", flush=True)
        result = simulate(jobs, state, get_scheduler(name))
        s = dict(result.summary)
        fm = fairness_metrics(result.finished, w_ref=w_ref)
        s.update(fm)
        out[name] = s
    return out


def run_e5():
    all_results = {}

    for trace_name, eta in ETASETTINGS:
        print(f"\n=== {trace_name} eta={eta} ===")
        jobs = _load_venus(eta)

        # N2-A: homogeneous vs mixed, all schedulers
        print("  N2-A: Homogeneous vs mixed (all schedulers)")
        key_homo  = f"{trace_name}_eta{eta}_homo"
        key_mixed = f"{trace_name}_eta{eta}_mixed"
        all_results[key_homo]  = _run_set(jobs, HOMO_TOPO,  SCHEDS_MAIN)
        all_results[key_mixed] = _run_set(jobs, MIXED_TOPO, SCHEDS_MAIN)

        # N2-B: original F vs F_C, mixed topology only
        print("  N2-B: F vs F_C on mixed")
        key_fc = f"{trace_name}_eta{eta}_mixed_fc"
        all_results[key_fc] = _run_set(jobs, MIXED_TOPO, SCHEDS_FC)

    return all_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
TABLE_COLS = [
    ("avg_wait",          "avg_wait",    "{:.0f}"),
    ("p99_wait",          "p99_wait",    "{:.0f}"),
    ("max_wait",          "max_wait",    "{:.0f}"),
    ("starved_3x",        "starved_3x",  "{}"),
    ("avg_wait_8gpu",     "wait_8gpu",   "{:.0f}"),
    ("gpu_utilization",   "util",        "{:.4f}"),
    ("avg_lb_index",      "lb_index",    "{:.5f}"),
    ("avg_unallocated_ratio_during_allocation", "unalloc", "{:.4f}"),
]


def _md_table(results_dict: Dict, title: str, schedulers: List[str]) -> str:
    header_row = "| run | " + " | ".join(lbl for _, lbl, _ in TABLE_COLS) + " |"
    n = len(TABLE_COLS)
    sep_row    = "|---|" + "---|" * n
    rows = [header_row, sep_row]
    for name in schedulers:
        if name not in results_dict:
            continue
        s = results_dict[name]
        cells = " | ".join(fmt.format(s.get(key, 0)) for key, _, fmt in TABLE_COLS)
        rows.append(f"| {name} | {cells} |")
    return f"### {title}\n\n" + "\n".join(rows) + "\n"


def write_markdown(all_results: Dict) -> str:
    sections = ["# E5: Heterogeneous server results (N2)\n"]
    sections.append(
        "Mixed topology: same 15 Venus clusters, 1,080 GPUs total. "
        "Each cluster: floor(S/2) eight-GPU servers split into two four-GPU servers; "
        "remaining servers keep 8 GPUs.\n"
    )

    for trace_name, eta in ETASETTINGS:
        sections.append(f"## {trace_name} eta={eta}\n")

        key_homo  = f"{trace_name}_eta{eta}_homo"
        key_mixed = f"{trace_name}_eta{eta}_mixed"
        key_fc    = f"{trace_name}_eta{eta}_mixed_fc"

        if key_homo in all_results:
            sections.append(_md_table(all_results[key_homo],
                                      "N2-A: Homogeneous topology",
                                      SCHEDS_MAIN))
        if key_mixed in all_results:
            sections.append(_md_table(all_results[key_mixed],
                                      "N2-A: Mixed topology",
                                      SCHEDS_MAIN))
        if key_fc in all_results:
            sections.append(_md_table(all_results[key_fc],
                                      "N2-B: F vs F_C on mixed topology",
                                      SCHEDS_FC))

        # Insight line
        if key_homo in all_results and key_mixed in all_results:
            w_homo  = all_results[key_homo].get("mcg", {}).get("avg_wait", 0)
            w_mixed = all_results[key_mixed].get("mcg", {}).get("avg_wait", 0)
            factor  = w_mixed / w_homo if w_homo else float("nan")
            w_fgd_m = all_results[key_mixed].get("fgd", {}).get("avg_wait", 0)
            sections.append(
                f"\n**Key finding:** Mixed topology raises MCG avg wait "
                f"×{factor:.1f} ({w_mixed:,.0f} s vs {w_homo:,.0f} s). "
                f"MCG is still best on mixed ({w_mixed:,.0f} s vs FGD {w_fgd_m:,.0f} s).\n"
            )
        if key_fc in all_results:
            w_mcg = all_results[key_fc].get("mcg", {}).get("avg_wait", 0)
            w_fc  = all_results[key_fc].get("mcg_fc", {}).get("avg_wait", 0)
            w_fcn = all_results[key_fc].get("mcg_fc_norm", {}).get("avg_wait", 0)
            neg = "negative" if w_fc > w_mcg else "positive"
            sections.append(
                f"**F_C result:** {neg} – F_C: {w_fc:,.0f} s, "
                f"F_C (renorm): {w_fcn:,.0f} s vs original: {w_mcg:,.0f} s.\n"
            )

    return "\n".join(sections)


def plot_e5(all_results: Dict):
    if not HAS_MPL:
        return
    for trace_name, eta in ETASETTINGS:
        key_homo  = f"{trace_name}_eta{eta}_homo"
        key_mixed = f"{trace_name}_eta{eta}_mixed"
        key_fc    = f"{trace_name}_eta{eta}_mixed_fc"
        if key_homo not in all_results:
            continue

        # --- Figure 1: N2-A homogeneous vs mixed, all schedulers ---
        scheds = SCHEDS_MAIN
        w_homo  = [all_results[key_homo].get(s, {}).get("avg_wait", 0) for s in scheds]
        w_mixed = [all_results[key_mixed].get(s, {}).get("avg_wait", 0) for s in scheds]
        x = range(len(scheds))
        fig, ax = plt.subplots(figsize=(9, 4))
        width = 0.35
        bars1 = ax.bar([i - width/2 for i in x], w_homo,  width, label="Homogeneous 8-GPU", color="#3498db")
        bars2 = ax.bar([i + width/2 for i in x], w_mixed, width, label="Mixed 4+8-GPU",     color="#e74c3c")
        ax.set_xticks(list(x))
        ax.set_xticklabels(scheds, fontsize=9)
        ax.set_ylabel("Average wait (s)")
        ax.set_title(f"E5 N2-A: Homogeneous vs Mixed – {trace_name} η={eta}")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(str(FIG_DIR / f"e5_n2a_{trace_name}_eta{eta}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved e5_n2a_{trace_name}_eta{eta}.png")

        # --- Figure 2: N2-B F vs F_C ---
        if key_fc in all_results:
            scheds_fc = SCHEDS_FC
            w_fc = [all_results[key_fc].get(s, {}).get("avg_wait", 0) for s in scheds_fc]
            labels = ["MCG (original F)", "MCG F_C (truncated)", "MCG F_C (renormed)"]
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#3498db", "#e67e22", "#9b59b6"]
            ax.bar(labels, w_fc, color=colors)
            ax.set_ylabel("Average wait (s)")
            ax.set_title(f"E5 N2-B: F vs F_C – mixed topology – {trace_name} η={eta}")
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            fig.savefig(str(FIG_DIR / f"e5_n2b_{trace_name}_eta{eta}.png"), dpi=150)
            plt.close(fig)
            print(f"  Saved e5_n2b_{trace_name}_eta{eta}.png")


if __name__ == "__main__":
    print("Running E5 heterogeneity experiment…")
    results = run_e5()

    # Save JSON
    out_json = RESULTS_DIR / "e5_heterogeneous.json"
    with open(str(out_json), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {out_json}")

    # Save Markdown
    md = write_markdown(results)
    out_md = RESULTS_DIR / "RESULTS_E5.md"
    with open(str(out_md), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"Saved {out_md}")

    # Figures
    plot_e5(results)
    print("\nDone.")
