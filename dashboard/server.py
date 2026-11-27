"""FastAPI server for the MCG-Sched Control Plane dashboard.

Run:
    python -m dashboard.server
    # or: uvicorn dashboard.server:app --host 127.0.0.1 --port 8501
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.inspector import (  # noqa: E402
    SimulationSession,
    get_catalog_meta,
    inspect_n2_fragmentation,
    load_saved_experiments,
    run_comparison,
    run_live_e5,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="MCG-Sched Control Plane", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (per client tab session_id)
_SESSIONS: Dict[str, SimulationSession] = {}


class InitRequest(BaseModel):
    session_id: str = "default"
    trace_id: str = "venus"
    topo_id: str = "venus15"
    sched_id: str = "mcg"
    n_jobs: int = Field(default=300, ge=0, le=20000)
    eta: float = Field(default=10.0, gt=0.0, le=100.0)
    burst: bool = False
    seed: int = 0
    params: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    session_id: str = "default"
    count: int = Field(default=1, ge=1, le=500)


class RunRequest(BaseModel):
    session_id: str = "default"


class CompareRequest(BaseModel):
    trace_id: str = "venus"
    topo_id: str = "venus15"
    sched_ids: List[str] = Field(
        default_factory=lambda: ["fcfs", "consolidate", "fgd", "mcg", "mcg_fair", "mcg_fc"]
    )
    n_jobs: int = Field(default=300, ge=10, le=10000)
    eta: float = Field(default=10.0, gt=0.0, le=100.0)
    burst: bool = False
    seed: int = 0


class N2InspectRequest(BaseModel):
    demand_weights: Dict[str, float] = Field(
        default_factory=lambda: {"1": 0.45, "2": 0.25, "4": 0.15, "8": 0.15}
    )
    job_gpus: int = Field(default=2, ge=1, le=8)


class LiveE5Request(BaseModel):
    trace_id: str = "venus"
    n_jobs: int = Field(default=500, ge=50, le=7000)
    eta: float = Field(default=15.0, gt=0.0, le=50.0)


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta")
def meta() -> Dict[str, Any]:
    return get_catalog_meta()


@app.post("/api/sim/init")
def sim_init(req: InitRequest) -> Dict[str, Any]:
    try:
        sess = SimulationSession(
            trace_id=req.trace_id,
            topo_id=req.topo_id,
            sched_id=req.sched_id,
            n_jobs=req.n_jobs,
            eta=req.eta,
            burst=req.burst,
            seed=req.seed,
            params=req.params,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _SESSIONS[req.session_id] = sess
    return sess.snapshot()


@app.post("/api/sim/step")
def sim_step(req: StepRequest) -> Dict[str, Any]:
    sess = _SESSIONS.get(req.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Simulation session not initialized")
    return sess.step(req.count)


@app.post("/api/sim/run")
def sim_run(req: RunRequest) -> Dict[str, Any]:
    sess = _SESSIONS.get(req.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Simulation session not initialized")
    return sess.run_to_end()


@app.get("/api/sim/state")
def sim_state(session_id: str = "default") -> Dict[str, Any]:
    sess = _SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Simulation session not initialized")
    return sess.snapshot()


@app.post("/api/compare")
def api_compare(req: CompareRequest) -> Dict[str, Any]:
    try:
        return run_comparison(
            trace_id=req.trace_id,
            topo_id=req.topo_id,
            sched_ids=req.sched_ids,
            n_jobs=req.n_jobs,
            eta=req.eta,
            burst=req.burst,
            seed=req.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/experiments/saved")
def api_saved_experiments() -> Dict[str, Any]:
    return load_saved_experiments()


@app.post("/api/experiments/inspect_n2")
def api_inspect_n2(req: N2InspectRequest) -> Dict[str, Any]:
    weights = {int(k): float(v) for k, v in req.demand_weights.items() if float(v) > 0}
    if not weights:
        raise HTTPException(status_code=400, detail="Demand weights cannot be empty")
    return inspect_n2_fragmentation(demand_weights=weights, job_gpus=req.job_gpus)


@app.post("/api/experiments/run_e5")
def api_run_e5(req: LiveE5Request) -> Dict[str, Any]:
    return run_live_e5(trace_id=req.trace_id, n_jobs=req.n_jobs, eta=req.eta)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main() -> None:
    port = int(os.environ.get("PORT", "8501"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
