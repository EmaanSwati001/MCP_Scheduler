# MCG-Sched project: simulation core

Shared foundation for the reproduction and extension of MCG-Sched (Wu et al., IEEE TPDS 2025).
It contains the data model, the discrete-event simulator, the scheduler interface, common
metrics, a synthetic workload generator and the FCFS + First Fit reference policy.
Standard library only, Python 3.10+.

```
python -m pytest tests -q
python run.py --topology configs/toy.json --scheduler fcfs --jobs 200
python run.py --topology configs/paper_8x16.json --trace data/venus.csv --eta 10 --out out.json
```

## Layout

| Path | Contents |
|---|---|
| `mcgsched/models/` | `Job`, `Server`, `Cluster`, `ClusterState` |
| `mcgsched/simulation/` | `Simulator`, `simulate()`, metrics (`summarize`, `Sample`) |
| `mcgsched/schedulers/` | `Scheduler` base class, `Placement`, `REGISTRY`, `fcfs.py` |
| `mcgsched/workloads/` | synthetic jobs, eta compression, burst, job CSV read/write |
| `configs/` | topology files |
| `tests/` | hand-calculated scenarios and invariants |

## Model

One job runs on one server, GPUs are exclusive, no preemption. A job larger than the
largest server is rejected on arrival and reported in `jobs_rejected`.

## Writing a scheduler

```python
from mcgsched.schedulers import Placement, Scheduler

class MyPolicy(Scheduler):
    name = "mine"

    def setup(self, jobs, state):          # optional, called once before the run
        ...                                # e.g. global demand popularity p_d

    def schedule(self, queue, state, now): # called after each batch of events
        plan = state.clone()               # scratch copy to plan several jobs
        out = []
        for job in queue:
            server = next(plan.feasible_servers(job.gpus), None)
            if server is None:
                break
            plan.allocate(job.job_id, job.gpus, server.cluster_id, server.server_id)
            out.append(Placement(job.job_id, server.cluster_id, server.server_id))
        return out
```

Then add it to `REGISTRY` in `mcgsched/schedulers/__init__.py`.

Rules:

- `queue` is the waiting jobs in arrival order, rebuilt every call. A policy that needs its
  own persistent order or reservations keeps that state on `self`.
- `state` and the `Job` objects are read-only. Only the simulator allocates and releases.
- Each returned placement must be feasible when applied, in list order. Otherwise the
  simulator raises `SchedulerError`.
- Jobs you do not return keep waiting. The next call comes at the next arrival or completion.
- Completions are processed before arrivals at the same timestamp.

Useful `ClusterState` members: `clusters`, `servers()`, `server(c, s)`, `feasible_servers(g)`,
`utilizations()`, `load_balance_index()` (paper Eq. 6), `lb_index_after(c, g)` (Eq. 7-9),
`total_capacity`, `free_gpus`, `max_server_capacity`. `Server` has `capacity`, `free`, `running`.

Utilization here is the allocated fraction of a cluster. The paper's Eq. (1) writes it with
remaining GPUs but Eq. (8) adds the job's GPUs, so the allocated fraction is what Eq. (8)
needs. The variance is the same either way.

## Job CSV schema

`job_id,arrival_time,gpus,duration`, times in seconds. Trace loaders should write this
format; `load_jobs_csv` reads it. `compress_arrivals(jobs, eta)` applies
`t' = t0 + (t - t0) / eta`.

## Topology files

```json
{"clusters": [
  {"count": 8, "servers": 16, "gpus_per_server": 8},
  {"capacities": [4, 4, 8, 8]}
]}
```

`count` repeats a cluster; `capacities` gives per-server sizes for mixed clusters.
`configs/venus15_placeholder.json` matches the paper's totals (15 clusters, 135 servers,
1080 GPUs) but its per-cluster sizes are invented until the Venus virtual-cluster file is parsed.

## Results

`simulate(jobs, state, scheduler)` returns a `SimulationResult` with `finished`, `rejected`,
`unscheduled`, `timeline` (one `Sample(t, used_gpus, lb_index, queue_len, running)` per
scheduling round) and `summary`. Inputs are copied, so the same jobs and state can be given
to every scheduler.

For the dashboard, use `Simulator(jobs, state, scheduler, listener)` and call `step()`;
the listener receives `"arrival"`, `"reject"`, `"start"` and `"finish"` events.

## Fairness (N1) and score normalisation (N4)

Code: `mcgsched/schedulers/mcg_fair.py`, `mcgsched/mechanisms/reservations.py`,
`mcgsched/evaluation/fairness.py`, `experiments/run_fairness_experiments.py`, `tests/test_fairness_and_normalisation.py`.
`mcg.py` is not touched. `MCGFair` subclasses `MCGSched`; with default arguments it makes
identical decisions (tested).

| Registry name | What it adds |
|---|---|
| `mcg_aging` | window order uses `v - beta * min(wait / W_ref, 5)` |
| `mcg_guard` | job waiting `>= k * W_ref` goes first; if it cannot start, everything behind it waits (blocking, no reservation) |
| `mcg_fair` | aging + guard + drain reservation: a protected job that cannot start reserves the server needing the fewest running jobs to finish; no other job is placed there; other servers keep working |
| `mcg_norm` | G and sigma^2 min-max scaled over one candidate pool per round (distinct request sizes in the window) |

`W_ref` is the mean job duration of the trace, fixed for the run. Defaults: `beta=1`, `k=3`,
`max_reservations=1`. All are constructor arguments, e.g. `get_scheduler("mcg_fair", k=1.0, beta=0)`.
No preemption and no runtime estimates are used; a reservation only picks a server whose
capacity can hold the job, so the job can always eventually start.

Dashboard hooks on the scheduler object: `reserved_servers()` (job_id -> (cluster, server)),
`protected_ids`, and `log` of `(time, "protect" | "reserve" | "release", job_id, server)`.

```
python -m pytest tests -q
python run.py --topology configs/venus15.json --trace data/clean/venus.csv --eta 15 --scheduler mcg mcg_fair
python experiments/run_fairness_experiments.py            # E4 + E7, about 2 minutes, writes results/E4_E7_fairness_results.md
```

## Heterogeneous servers (N2) and dashboard — Person 5

Code: `mcgsched/mechanisms/fragmentation.py` (adds `CapacityAwareDemandModel`),
`mcgsched/schedulers/mcg_hetero.py`, `configs/venus15_mixed.json`,
`dashboard/app.py`, `tests/test_heterogeneous.py`,
`experiments/run_e5_experiment.py`.

### Mixed topology (`venus15_mixed.json`)

Same 15 clusters and 1,080 GPU total as `venus15.json`. Each cluster: `floor(S/2)`
eight-GPU servers are split into two four-GPU servers; the remaining `ceil(S/2)`
servers keep 8 GPUs.  Total GPU capacity is unchanged by design.

```
python -c "from mcgsched.models import ClusterState; s=ClusterState.from_json('configs/venus15_mixed.json'); print(s)"
# → ClusterState(15 clusters, 0/1080 GPUs used)
```

### Capacity-aware fragmentation F_C

For a server with capacity C, only demands d ≤ C are relevant to its fragmentation
score. `CapacityAwareDemandModel.gradient_for_server(free, gpus, server_cap)` applies
this filter. On homogeneous topologies (all servers with the same capacity) the result
is **identical** to the original `DemandModel.gradient()` — tested.

| Registry name | What it adds |
|---|---|
| `mcg_fc`      | Capacity-aware F_C (truncated, not renormalised) |
| `mcg_fc_norm` | F_C renormalised: p̃(d\|C) = p̃_d / Σ_{q≤C} p̃_q |

### E5 experiment

```
python experiments/run_e5_experiment.py
# writes results/RESULTS_E5.md, results/e5_heterogeneous.json, results/figures/e5_*.png
```

**Key results (Venus, η=15):**
- Mixed topology raises every scheduler's avg wait ~4× because 8-GPU jobs lose half their candidate servers.
- MCG-Sched remains best on the mixed topology.
- F_C is a negative result: ignoring 8-GPU demand on 4-GPU servers makes those servers look less fragmented, so small jobs preferentially break up 8-GPU servers instead.

### Dashboard

```
streamlit run dashboard/app.py
```

Three tabs:
1. **Run & Visualise** — step / play / run; GPU grid with reserved servers outlined in orange; waiting queue; live metrics; timeline charts.
2. **Compare** — run any set of schedulers on the same workload side-by-side with bar charts.
3. **E5 Experiment** — in-browser reproduction of the N2 heterogeneity results with figures.

