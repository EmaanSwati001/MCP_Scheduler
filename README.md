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
