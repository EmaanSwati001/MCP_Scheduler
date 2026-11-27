"""MCG-Sched Live Dashboard (Person 5 – Streamlit).

Launch:
    streamlit run dashboard/app.py

The dashboard calls the REAL simulation engine – no pre-computed screenshots.
Every button press runs Python code in mcgsched/.

Tabs:
  1. Run & Visualise  – step / play / run; GPU grid; queue; score table; metrics.
  2. Compare          – run all selected schedulers on the same workload side-by-side.
  3. E5 Experiment    – homogeneous vs mixed topology, original F vs F_C.
"""
from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# ---- ensure the project root is importable --------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mcgsched.models import ClusterState
from mcgsched.schedulers import REGISTRY, get_scheduler
from mcgsched.simulation import simulate
from mcgsched.simulation.engine import Simulator
from mcgsched.workloads import (
    compress_arrivals,
    load_jobs_csv,
    make_burst,
    synthetic_jobs,
)
from mcgsched.evaluation.fairness import fairness_metrics

# ---- optional matplotlib import -------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MCG-Sched Dashboard",
    page_icon="🖥️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------
DATA_DIR    = ROOT / "data" / "clean"
CONFIG_DIR  = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRACES = {
    "Synthetic": None,
    "Venus":  str(DATA_DIR / "venus.csv"),
    "Philly": str(DATA_DIR / "philly.csv"),
    "Earth":  str(DATA_DIR / "earth.csv"),
    "Saturn": str(DATA_DIR / "saturn.csv"),
    "Uranus": str(DATA_DIR / "uranus.csv"),
}
TOPOLOGIES = {
    "Venus 15-cluster (homogeneous 8-GPU)": str(CONFIG_DIR / "venus15.json"),
    "Venus 15-cluster (mixed 4+8-GPU, N2)": str(CONFIG_DIR / "venus15_mixed.json"),
    "Paper 8×16 (8-GPU)":                  str(CONFIG_DIR / "paper_8x16.json"),
    "Toy":                                  str(CONFIG_DIR / "toy.json"),
}
SCHEDULER_NAMES = sorted(REGISTRY.keys())

GPU_COLORS = {
    "free":     "#2ecc71",   # green
    "running":  "#3498db",   # blue
    "reserved": "#e67e22",   # orange
}

# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "sim": None,         # Simulator instance
        "done": False,
        "step_count": 0,
        "play_speed": 0.05,
        "jobs": [],
        "state": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _load_jobs(trace_name: str, n_jobs: int, eta: float, burst: bool, seed: int):
    path = TRACES[trace_name]
    if path and os.path.exists(path):
        jobs = load_jobs_csv(path)
    else:
        jobs = synthetic_jobs(n_jobs, seed=seed)
    if n_jobs and len(jobs) > n_jobs:
        jobs = jobs[:n_jobs]
    if eta != 1.0:
        jobs = compress_arrivals(jobs, eta)
    if burst:
        jobs = make_burst(jobs)
    return jobs


def _load_topology(topo_name: str) -> ClusterState:
    return ClusterState.from_json(TOPOLOGIES[topo_name])


# ---------------------------------------------------------------------------
# GPU grid rendering
# ---------------------------------------------------------------------------
def _server_html(server, reserved_keys: set) -> str:
    is_reserved = server.key in reserved_keys
    border = "3px solid #e67e22" if is_reserved else "1px solid #555"
    n_free    = server.free
    n_running = server.capacity - server.free
    gpu_boxes = []
    for _ in range(n_running):
        gpu_boxes.append(
            f'<div style="width:14px;height:14px;background:{GPU_COLORS["running"]};'
            f'display:inline-block;margin:1px;border-radius:2px;" title="running"></div>'
        )
    for _ in range(n_free):
        col = GPU_COLORS["reserved"] if is_reserved else GPU_COLORS["free"]
        gpu_boxes.append(
            f'<div style="width:14px;height:14px;background:{col};'
            f'display:inline-block;margin:1px;border-radius:2px;" title="free"></div>'
        )
    boxes_html = "".join(gpu_boxes)
    label = f"C{server.cluster_id}·S{server.server_id} ({server.capacity}G)"
    return (
        f'<div style="border:{border};border-radius:4px;padding:4px;'
        f'margin:3px;display:inline-block;vertical-align:top;min-width:100px;">'
        f'<div style="font-size:10px;color:#aaa;margin-bottom:2px;">{label}</div>'
        f'{boxes_html}</div>'
    )


def _render_gpu_grid(sim: Simulator, reserved_keys: set, max_clusters: int = 15):
    clusters = sim.state.clusters[:max_clusters]
    rows = []
    for c in clusters:
        servers_html = "".join(_server_html(s, reserved_keys) for s in c.servers)
        util_pct = f"{100*c.utilization:.0f}%"
        rows.append(
            f'<div style="margin-bottom:6px;">'
            f'<b style="font-size:12px;">Cluster {c.cluster_id} ({util_pct} used)</b><br/>'
            f'{servers_html}</div>'
        )
    full = "\n".join(rows)
    st.markdown(
        f'<div style="background:#1e1e1e;padding:10px;border-radius:6px;'
        f'overflow-x:auto;max-height:400px;overflow-y:auto;">{full}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _summarize_live(sim: Simulator):
    n_done = len(sim.finished)
    n_wait = len(sim.queue)
    n_run  = len(sim.running)
    util   = sim.state.used_gpus / sim.state.total_capacity if sim.state.total_capacity else 0
    lb     = sim.state.load_balance_index()
    avg_w  = (sum(j.wait_time for j in sim.finished) / n_done) if n_done else 0.0
    return dict(completed=n_done, waiting=n_wait, running=n_run,
                util=util, lb_index=lb, avg_wait_s=avg_w)


def _fmt(val, fmt=".1f"):
    return format(val, fmt) if isinstance(val, float) else str(val)


# ---------------------------------------------------------------------------
# Timeline chart
# ---------------------------------------------------------------------------
def _plot_timeline(sim: Simulator):
    if not HAS_MPL or not sim.timeline:
        return
    tl = sim.timeline
    ts = [s.t for s in tl]
    util = [s.used_gpus / sim.state.total_capacity for s in tl]
    qlen = [s.queue_len for s in tl]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    ax1.plot(ts, util, color="#3498db")
    ax1.set_ylabel("GPU util", fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax2.fill_between(ts, qlen, alpha=0.5, color="#e74c3c")
    ax2.set_ylabel("Queue len", fontsize=9)
    ax2.set_xlabel("Simulation time (s)", fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# TAB 1 – Run & Visualise
# ---------------------------------------------------------------------------
def tab_run():
    st.header("🔬 Run & Visualise")

    col_l, col_r = st.columns([1, 3])
    with col_l:
        st.subheader("Configuration")
        trace_name  = st.selectbox("Workload trace", list(TRACES.keys()), key="t1_trace")
        n_jobs      = st.number_input("Max jobs (0 = all)", min_value=0, max_value=10000,
                                      value=200, key="t1_njobs")
        eta         = st.slider("η (arrival compression)", 1.0, 20.0, 1.0, 0.5, key="t1_eta")
        burst       = st.checkbox("Burst mode (all arrive at t=0)", key="t1_burst")
        seed        = st.number_input("Synthetic seed", value=0, key="t1_seed")
        topo_name   = st.selectbox("Topology", list(TOPOLOGIES.keys()), key="t1_topo")
        sched_name  = st.selectbox("Scheduler", SCHEDULER_NAMES,
                                   index=SCHEDULER_NAMES.index("mcg") if "mcg" in SCHEDULER_NAMES else 0,
                                   key="t1_sched")
        speed       = st.slider("Play speed (s/step)", 0.0, 1.0, 0.05, 0.01, key="t1_speed")

        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        btn_init  = col_b1.button("🔄 Init")
        btn_step  = col_b2.button("⏭ Step", disabled=st.session_state.done)
        btn_play  = col_b3.button("▶ Play", disabled=st.session_state.done)
        btn_run   = col_b4.button("⏩ Run",  disabled=st.session_state.done)

    # ---- init ----
    if btn_init:
        jobs  = _load_jobs(trace_name, int(n_jobs) or None, eta, burst, int(seed))
        state = _load_topology(topo_name)
        sched = get_scheduler(sched_name)
        st.session_state.sim        = Simulator(jobs, state, sched)
        st.session_state.done       = False
        st.session_state.step_count = 0
        st.session_state.jobs       = jobs
        st.session_state.state      = state
        st.rerun()

    sim = st.session_state.sim

    if sim is None:
        st.info("👈 Configure and press **Init** to start.")
        return

    # ---- step / play / run ----
    if btn_step and not st.session_state.done:
        ok = sim.step()
        st.session_state.step_count += 1
        if not ok:
            st.session_state.done = True

    if btn_run and not st.session_state.done:
        while sim.step():
            pass
        st.session_state.done = True

    if btn_play and not st.session_state.done:
        placeholder = st.empty()
        while sim.step():
            st.session_state.step_count += 1
            time.sleep(speed)
            placeholder.info(
                f"Step {st.session_state.step_count} | t={sim.now:.0f}s | "
                f"done={len(sim.finished)} running={len(sim.running)} queue={len(sim.queue)}"
            )
        st.session_state.done = True
        placeholder.success("Simulation complete.")

    # ---- reserved servers ----
    reserved_keys: set = set()
    if hasattr(sim.scheduler, "reserved_servers"):
        reserved_keys = set(sim.scheduler.reserved_servers().values())

    # ---- layout ----
    with col_r:
        st.subheader(f"GPU Grid — {sched_name}  |  t = {sim.now:.0f} s")
        _render_gpu_grid(sim, reserved_keys)

        m = _summarize_live(sim)
        mc = st.columns(6)
        mc[0].metric("Completed", m["completed"])
        mc[1].metric("Running",   m["running"])
        mc[2].metric("Waiting",   m["waiting"])
        mc[3].metric("GPU util",  f"{m['util']:.1%}")
        mc[4].metric("LB index",  f"{m['lb_index']:.5f}")
        mc[5].metric("Avg wait",  f"{m['avg_wait_s']:.0f}s")

        with st.expander("📋 Waiting queue (top 20)"):
            q = sim.queue[:20]
            if q:
                rows = [{"job_id": j.job_id, "gpus": j.gpus,
                         "arrived": f"{j.arrival_time:.0f}s",
                         "waited": f"{sim.now - j.arrival_time:.0f}s"} for j in q]
                st.table(rows)
            else:
                st.write("Queue is empty.")

        # Protected / reserved info (mcg_fair variants)
        if hasattr(sim.scheduler, "protected_ids") and sim.scheduler.protected_ids:
            st.warning(f"⚠️ Protected jobs: {sorted(sim.scheduler.protected_ids)}")
        if reserved_keys:
            st.info(f"🔒 Reserved servers: {sorted(reserved_keys)}")

        st.subheader("📈 Timeline")
        _plot_timeline(sim)

        if st.session_state.done:
            result = sim.result()
            with st.expander("📊 Full summary"):
                for k, v in result.summary.items():
                    if isinstance(v, float):
                        st.write(f"**{k}**: {v:.4f}")
                    else:
                        st.write(f"**{k}**: {v}")


# ---------------------------------------------------------------------------
# TAB 2 – Compare schedulers
# ---------------------------------------------------------------------------
def tab_compare():
    st.header("📊 Scheduler Comparison")

    col_l, col_r = st.columns([1, 3])
    with col_l:
        st.subheader("Configuration")
        trace_name = st.selectbox("Workload trace", list(TRACES.keys()), key="t2_trace")
        n_jobs     = st.number_input("Max jobs (0 = all)", 0, 10000, 300, key="t2_njobs")
        eta        = st.slider("η", 1.0, 20.0, 10.0, 0.5, key="t2_eta")
        burst      = st.checkbox("Burst mode", key="t2_burst")
        seed       = st.number_input("Seed", value=0, key="t2_seed")
        topo_name  = st.selectbox("Topology", list(TOPOLOGIES.keys()), key="t2_topo")
        selected   = st.multiselect(
            "Schedulers to compare",
            SCHEDULER_NAMES,
            default=["fcfs", "consolidate", "fgd", "mcg", "mcg_fc"],
            key="t2_scheds",
        )
        run_btn = st.button("🚀 Run comparison")

    if run_btn and selected:
        jobs  = _load_jobs(trace_name, int(n_jobs) or None, eta, burst, int(seed))
        state = _load_topology(topo_name)
        results = {}
        prog = st.progress(0)
        for i, name in enumerate(selected):
            results[name] = simulate(jobs, state, get_scheduler(name))
            prog.progress((i + 1) / len(selected))

        with col_r:
            # Table
            cols = ["avg_wait", "p99_wait", "max_wait", "gpu_utilization",
                    "avg_lb_index", "avg_unallocated_ratio_during_allocation",
                    "jobs_completed"]
            header = ["scheduler"] + cols
            rows   = []
            for name in selected:
                s = results[name].summary
                row = [name] + [_fmt(s.get(c, 0)) for c in cols]
                rows.append(row)

            import pandas as pd
            df = pd.DataFrame(rows, columns=header)
            st.dataframe(df, use_container_width=True)

            if HAS_MPL:
                # Bar chart: avg_wait
                fig, ax = plt.subplots(figsize=(8, 3))
                names = [r[0] for r in rows]
                waits = [results[n].summary.get("avg_wait", 0) for n in selected]
                bars  = ax.bar(names, waits, color="#3498db")
                ax.set_ylabel("Avg wait (s)")
                ax.set_title("Average wait time by scheduler")
                plt.xticks(rotation=20, ha="right", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Bar chart: GPU utilization
                fig, ax = plt.subplots(figsize=(8, 3))
                utils = [results[n].summary.get("gpu_utilization", 0) for n in selected]
                ax.bar(names, utils, color="#2ecc71")
                ax.set_ylabel("GPU utilization")
                ax.set_title("GPU utilization by scheduler")
                ax.set_ylim(0, 1)
                plt.xticks(rotation=20, ha="right", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    elif not run_btn:
        with col_r:
            st.info("👈 Select schedulers and press **Run comparison**.")


# ---------------------------------------------------------------------------
# TAB 3 – E5 Experiment (Heterogeneity)
# ---------------------------------------------------------------------------
def tab_e5():
    st.header("🔬 E5: Heterogeneous Servers (N2)")
    st.markdown(
        """
        **Research question:** How do mixed 4-GPU/8-GPU servers affect scheduling performance,
        and does the capacity-aware fragmentation measure F_C help?

        - **N2-A** Homogeneous 8-GPU vs mixed 4+8-GPU topology.
        - **N2-B** Original F(r) vs capacity-aware F_C on the mixed topology.
        """
    )

    col_l, col_r = st.columns([1, 3])
    with col_l:
        trace_name = st.selectbox("Trace", [k for k in TRACES if k != "Synthetic"], key="e5_trace")
        n_jobs     = st.number_input("Max jobs", 0, 10000, 500, key="e5_njobs")
        eta        = st.slider("η", 1.0, 20.0, 15.0, 0.5, key="e5_eta")
        run_btn    = st.button("🧪 Run E5")

    if run_btn:
        jobs  = _load_jobs(trace_name, int(n_jobs) or None, eta, False, 0)

        configs = {
            "homo – mcg":        ("venus15.json",       "mcg"),
            "homo – fcfs":       ("venus15.json",       "fcfs"),
            "homo – fgd":        ("venus15.json",       "fgd"),
            "mixed – mcg":       ("venus15_mixed.json", "mcg"),
            "mixed – fcfs":      ("venus15_mixed.json", "fcfs"),
            "mixed – fgd":       ("venus15_mixed.json", "fgd"),
            "mixed – mcg_fc":    ("venus15_mixed.json", "mcg_fc"),
            "mixed – mcg_fc_norm": ("venus15_mixed.json", "mcg_fc_norm"),
        }

        results = {}
        prog = st.progress(0)
        for i, (label, (cfg, sched)) in enumerate(configs.items()):
            state = ClusterState.from_json(str(CONFIG_DIR / cfg))
            results[label] = simulate(jobs, state, get_scheduler(sched))
            prog.progress((i + 1) / len(configs))

        with col_r:
            # summary table
            cols = ["avg_wait", "p99_wait", "max_wait", "gpu_utilization",
                    "avg_lb_index", "avg_unallocated_ratio_during_allocation"]
            import pandas as pd
            rows = []
            for label in configs:
                s = results[label].summary
                rows.append([label] + [round(s.get(c, 0), 1) for c in cols])
            df = pd.DataFrame(rows, columns=["run"] + cols)
            st.dataframe(df, use_container_width=True)

            if HAS_MPL:
                labels = list(configs.keys())
                waits  = [results[l].summary.get("avg_wait", 0) for l in labels]
                colors = ["#3498db" if "homo" in l else
                          "#e67e22" if "fc" in l else
                          "#2ecc71"
                          for l in labels]
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.bar(labels, waits, color=colors)
                ax.set_ylabel("Average wait (s)")
                ax.set_title("E5: Homogeneous vs Mixed topology, F vs F_C")
                plt.xticks(rotation=30, ha="right", fontsize=8)
                patches = [
                    mpatches.Patch(color="#3498db", label="Homogeneous"),
                    mpatches.Patch(color="#2ecc71", label="Mixed – original F"),
                    mpatches.Patch(color="#e67e22", label="Mixed – F_C"),
                ]
                ax.legend(handles=patches)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Save figure
                fig_path = RESULTS_DIR / "figures" / "e5_hetero.png"
                fig_path.parent.mkdir(exist_ok=True)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(labels, waits, color=colors)
                ax.set_ylabel("Average wait (s)")
                ax.set_title("E5: Homogeneous vs Mixed topology, F vs F_C")
                plt.xticks(rotation=30, ha="right", fontsize=8)
                ax.legend(handles=patches)
                plt.tight_layout()
                fig.savefig(str(fig_path), dpi=150)
                plt.close(fig)
                st.success(f"Figure saved to {fig_path}")

            # interpretation
            w_homo  = results["homo – mcg"].summary["avg_wait"]
            w_mixed = results["mixed – mcg"].summary["avg_wait"]
            w_fc    = results["mixed – mcg_fc"].summary["avg_wait"]
            w_fcn   = results["mixed – mcg_fc_norm"].summary["avg_wait"]
            w_fgd_homo  = results["homo – fgd"].summary["avg_wait"]
            w_fgd_mixed = results["mixed – fgd"].summary["avg_wait"]

            factor = w_mixed / w_homo if w_homo > 0 else float("nan")
            st.markdown(f"""
---
**Key findings:**
- Mixed topology raises MCG avg wait **×{factor:.1f}** vs homogeneous
  ({w_mixed:,.0f} s vs {w_homo:,.0f} s).
- MCG still beats FGD on mixed: **{w_mixed:,.0f} s vs {w_fgd_mixed:,.0f} s**.
- F_C (truncated, not renormed): **{w_fc:,.0f} s** – {"better ✅" if w_fc < w_mixed else "worse ❌"} than original F on mixed.
- F_C renormalised: **{w_fcn:,.0f} s** – {"better ✅" if w_fcn < w_mixed else "worse ❌"} than original F on mixed.
""")

    elif not run_btn:
        with col_r:
            st.info("👈 Configure and press **Run E5**.")


# ---------------------------------------------------------------------------
# Main layout with tabs
# ---------------------------------------------------------------------------
st.title("🖥️ MCG-Sched: Live Simulation Dashboard")
st.caption(
    "Person 5 (N2 + Dashboard) · "
    "Real simulator engine · no pre-computed results"
)

tab1, tab2, tab3 = st.tabs(["▶ Run & Visualise", "📊 Compare", "🧪 E5 Experiment"])

with tab1:
    tab_run()
with tab2:
    tab_compare()
with tab3:
    tab_e5()
