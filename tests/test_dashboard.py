"""Tests for the MCG-Sched Control Plane dashboard backend and inspector."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard.inspector import (
    SimulationSession,
    get_catalog_meta,
    inspect_n2_fragmentation,
    run_comparison,
)
from dashboard.server import (
    CompareRequest,
    InitRequest,
    LiveE5Request,
    N2InspectRequest,
    RunRequest,
    StepRequest,
    api_compare,
    api_inspect_n2,
    api_run_e5,
    api_saved_experiments,
    health,
    meta,
    sim_init,
    sim_run,
    sim_state,
    sim_step,
)


def test_catalog_metadata_contains_traces_topos_and_schedulers():
    m = get_catalog_meta()
    trace_ids = {t["id"] for t in m["traces"]}
    topo_ids = {t["id"] for t in m["topologies"]}
    sched_ids = {s["id"] for s in m["schedulers"]}
    assert "venus" in trace_ids
    assert "synthetic" in trace_ids
    assert "venus15" in topo_ids
    assert "venus15_mixed" in topo_ids
    assert {"mcg", "mcg_fair", "mcg_fc", "fcfs", "fgd"} <= sched_ids


def test_simulation_session_step_and_decision_inspector():
    sess = SimulationSession(
        trace_id="synthetic",
        topo_id="venus15_mixed",
        sched_id="mcg",
        n_jobs=60,
        eta=10.0,
        burst=True,
        seed=0,
    )
    snap0 = sess.snapshot()
    assert snap0["done"] is False
    assert snap0["total_jobs"] == 60
    assert len(snap0["clusters"]) == 15

    snap1 = sess.step(1)
    assert snap1["step_count"] == 1
    assert snap1["last_decision"] is not None
    assert len(snap1["last_decision"]["candidates"]) > 0

    snap_end = sess.run_to_end()
    assert snap_end["done"] is True
    assert snap_end["metrics"]["completed"] == 60
    assert snap_end["summary"]["jobs_completed"] == 60


def test_n2_fragmentation_inspector():
    res = inspect_n2_fragmentation({"1": 0.45, "2": 0.25, "4": 0.15, "8": 0.15}, job_gpus=2)
    assert len(res["table_4g"]) == 5
    assert len(res["table_8g"]) == 9
    assert res["empty_gradients"]["server_4g"]["G_orig"] != res["empty_gradients"]["server_4g"]["G_fc"]


def test_comparison_runner_identical_job_count():
    res = run_comparison(
        trace_id="synthetic",
        topo_id="venus15",
        sched_ids=["fcfs", "mcg", "mcg_fc"],
        n_jobs=40,
        eta=5.0,
        burst=False,
        seed=1,
    )
    assert len(res["results"]) == 3
    for row in res["results"]:
        assert row["jobs_completed"] == 40


def test_dashboard_frontend_has_no_internal_labels():
    static_dir = Path(__file__).resolve().parent.parent / "dashboard" / "static"
    index_html = (static_dir / "index.html").read_text(encoding="utf-8")
    app_js = (static_dir / "app.js").read_text(encoding="utf-8")
    assert "MCG-Sched" in index_html
    assert "Person 5" not in index_html
    assert "Person 5" not in app_js


def test_server_endpoints_end_to_end():
    assert health()["status"] == "ok"
    assert "schedulers" in meta()

    init_res = sim_init(
        InitRequest(
            session_id="test_sess",
            trace_id="synthetic",
            topo_id="venus15_mixed",
            sched_id="mcg_fair",
            n_jobs=40,
            eta=10.0,
            burst=True,
            seed=0,
        )
    )
    assert init_res["total_jobs"] == 40

    step_res = sim_step(StepRequest(session_id="test_sess", count=2))
    assert step_res["step_count"] == 2

    state_res = sim_state(session_id="test_sess")
    assert state_res["step_count"] == 2

    run_res = sim_run(RunRequest(session_id="test_sess"))
    assert run_res["done"] is True
    assert run_res["metrics"]["completed"] == 40

    cmp_res = api_compare(
        CompareRequest(
            trace_id="synthetic",
            topo_id="venus15",
            sched_ids=["fcfs", "mcg"],
            n_jobs=30,
            eta=5.0,
        )
    )
    assert len(cmp_res["results"]) == 2

    n2_res = api_inspect_n2(N2InspectRequest())
    assert "empty_gradients" in n2_res

    e5_res = api_run_e5(LiveE5Request(trace_id="synthetic", n_jobs=60, eta=10.0))
    assert len(e5_res["rows"]) == 8

    saved = api_saved_experiments()
    assert "e5" in saved
