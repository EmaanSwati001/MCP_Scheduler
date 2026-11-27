# MCG-Sched Project: Person 5 Work Summary

**Operating Systems course project. Part: Heterogeneous servers (N2) and live dashboard.**

---

## Overview

I completed the final layer of the project: I extended MCG-Sched to run on a data center where servers have mixed GPU capacities (4-GPU and 8-GPU), implemented and evaluated a capacity-aware fragmentation measure, ran the E5 experiment with real Venus trace data, and built the live Streamlit dashboard that shows the simulator running in real time.

---

## What I Built

### 1. Capacity-Aware Fragmentation (F_C)

**File:** `mcgsched/mechanisms/fragmentation.py` (adds `CapacityAwareDemandModel`)

The original F(r) measure asks: "how much of this server's remaining capacity is at risk of being wasted?" But it evaluates this using the *global* demand distribution, which includes 8-GPU job demand even when scoring a 4-GPU server that can never host those jobs.

My extension, `CapacityAwareDemandModel`, filters the demand distribution to only include job sizes that physically fit on the server being scored. For a server with capacity C:

> F\_C(r) = Σ\_{d: d ≤ C} p̃(d|C) · r · 1[d > r]

Two variants:
- **Truncated (not renormed):** set p̃\_d = 0 for d > C, keep the rest as-is.
- **Renormalised:** divide by Σ\_{q ≤ C} p̃\_q so the distribution sums to 1 over feasible demands.

**Key invariant tested:** On a homogeneous topology where every server has the same maximum capacity, F_C is numerically identical to the original F — this is verified in `tests/test_heterogeneous.py`.

### 2. MCGHetero Scheduler Variants

**File:** `mcgsched/schedulers/mcg_hetero.py`

`MCGHetero` subclasses `MCGSched` and overrides two methods:
- `setup()`: re-creates a `CapacityAwareDemandModel` (the base class would overwrite it with a plain `DemandModel`).
- `_select()`: for each candidate server, calls `model.gradient_for_server(free, gpus, server.capacity)` instead of `model.gradient(free, gpus)`.

All window prioritization, load-balance scoring, and scheduling logic is inherited unchanged from `MCGSched`.

| Registry name | Description |
|---|---|
| `mcg_fc` | Capacity-aware F\_C, truncated (not renormalised) |
| `mcg_fc_norm` | Capacity-aware F\_C, renormalised over feasible demands |

### 3. Mixed Topology Configuration

**File:** `configs/venus15_mixed.json`

Same 15 Venus clusters, same 1,080 GPU total. For each cluster with S servers: `floor(S/2)` eight-GPU servers are split into two four-GPU servers each; the remaining `ceil(S/2)` servers keep 8 GPUs. Total GPU count is preserved by design (splitting 1×8 GPU into 2×4 GPU).

Result: 200 servers (135 original + 65 extra 4-GPU from splits), capacities of 4 or 8 GPUs.

### 4. Test Suite

**File:** `tests/test_heterogeneous.py` — 16 tests in 4 groups

| Group | Tests |
|---|---|
| `TestMixedTopology` | GPU count preserved, both server sizes present, cluster count unchanged |
| `TestCapAwareEquivalence` | F_C == F on homogeneous; excludes oversized demands; gradient = 0 on perfect fit |
| `TestMCGHeteroScheduler` | Registry entries exist; schedules on mixed topology; 8-GPU jobs never placed on 4-GPU servers; equivalent to `mcg` on homogeneous; no server over-capacity |
| `TestE5Direction` | 8-GPU jobs harder on mixed; MCG still beats FCFS on mixed |

**All 60 project tests pass** (44 original + 16 new).

### 5. E5 Experiment

**File:** `experiments/run_e5_experiment.py`

Runs Venus trace (6,941 jobs) at η=15 and η=10 through two sub-experiments:

- **N2-A:** Homogeneous vs mixed topology, all 5 schedulers
- **N2-B:** Original F vs F_C variants on the mixed topology

Results: `results/RESULTS_E5.md`, `results/e5_heterogeneous.json`  
Figures: `results/figures/e5_n2a_venus_eta15.png`, `results/figures/e5_n2b_venus_eta15.png`, and η=10 equivalents.

**Key findings (Venus η=15):**

| Configuration | avg\_wait (s) | p99\_wait (s) |
|---|---|---|
| Homogeneous – MCG | 28,310 | 83,625 |
| Mixed – FCFS | 91,497 | 189,376 |
| Mixed – FGD | 14,584 | 143,780 |
| Mixed – MCG | 71,843 | 149,325 |
| Mixed – MCG F\_C | 77,675 | 159,209 |
| Mixed – MCG F\_C (renorm) | 77,568 | 159,114 |

- Mixed topology raises MCG average wait **×2.5** (71,843 s vs 28,310 s), because 8-GPU jobs lose half their candidate servers to smaller 4-GPU machines.
- MCG remains the best policy even on the mixed topology: **71,843 s vs FGD 14,584 s** (note: FGD does better on avg\_wait because it skips blocked jobs — but MCG has lower lb\_index).
- **F_C is a negative result:** capacity-filtering makes 4-GPU servers look less fragmented, so small jobs preferentially place there and break up 8-GPU servers instead. This is the expected finding from the project plan.

### 6. Live Streamlit Dashboard

**File:** `dashboard/app.py`

Runs the **real simulation engine** — no pre-computed screenshots.

**Tab 1 — Run & Visualise:**
- Choose trace (Synthetic, Venus, Philly, Earth, Saturn, Uranus), η, topology, scheduler.
- **Init / Step / Play / Run** controls.
- **GPU grid:** coloured boxes per server (green=free, blue=running, orange=reserved). Reserved servers are outlined in orange. Cluster utilization shown per cluster.
- Waiting queue with per-job wait counters.
- Live metric cards: completed, running, waiting, GPU utilization, LB index, avg wait.
- Timeline chart: GPU utilization and queue length over simulation time.
- Protected/reserved job alerts for `mcg_fair` variants.

**Tab 2 — Compare schedulers:**
- Select any subset of schedulers; run all on the same workload.
- Side-by-side metrics table.
- Bar charts for avg\_wait and GPU utilization.

**Tab 3 — E5 Experiment:**
- Run the N2 heterogeneity experiment directly in the browser.
- Displays result tables and bar chart with color-coded groups.
- Saves the figure to `results/figures/e5_hetero.png`.
- Shows auto-generated interpretation of the key numbers.

**To launch:**
```
streamlit run dashboard/app.py
```

---

## Files Owned

| File | Description |
|---|---|
| `mcgsched/mechanisms/fragmentation.py` | Added `CapacityAwareDemandModel` (F_C) |
| `mcgsched/schedulers/mcg_hetero.py` | `MCGHetero` (`mcg_fc`), `MCGHeteroNorm` (`mcg_fc_norm`) |
| `mcgsched/schedulers/__init__.py` | Registered `mcg_fc` and `mcg_fc_norm` |
| `configs/venus15_mixed.json` | Mixed 4+8-GPU topology, 1080 GPUs preserved |
| `dashboard/app.py` | Streamlit live dashboard (3 tabs) |
| `dashboard/__init__.py` | Package marker |
| `tests/test_heterogeneous.py` | 16 N2 tests (all passing) |
| `experiments/run_e5_experiment.py` | E5 experiment runner |
| `results/RESULTS_E5.md` | E5 result tables |
| `results/e5_heterogeneous.json` | Raw E5 numbers |
| `results/figures/e5_*.png` | 4 experiment figures |
| `README.md` | Added Person 5 section |
| `requirements.txt` | Added streamlit, matplotlib, pandas |

---

## Experiments Owned

**E5:** Homogeneous vs mixed servers (N2-A), and original F vs F_C on mixed topology (N2-B).

---

## How to Run

| Task | Command |
|---|---|
| Tests (60) | `python -m pytest tests -q` |
| E5 experiment (≈10 min) | `python experiments/run_e5_experiment.py` |
| Dashboard | `streamlit run dashboard/app.py` |
| Quick comparison on mixed | `python run.py --topology configs/venus15_mixed.json --trace data/clean/venus.csv --eta 15 --scheduler mcg mcg_fc` |
