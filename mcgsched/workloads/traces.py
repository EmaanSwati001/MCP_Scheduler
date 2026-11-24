"""Trace loaders for Helios (Venus, Earth, Saturn, Uranus) and Philly.

Every loader applies the paper's filters and returns a FilterReport that
records how many jobs each rule kept. Standard library only.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, NamedTuple, Sequence, Tuple

from ..models import Job

TIME_FMT = "%Y-%m-%d %H:%M:%S"
GPUS_PER_SERVER = 8  # every server in the paper's setup has 8 GPUs


class Record(NamedTuple):
    """One job that passed the filters. Not yet a simulator Job."""

    submit: datetime
    gpus: int
    duration: float  # seconds


@dataclass
class FilterReport:
    raw: int = 0
    steps: List[Tuple[str, int]] = field(default_factory=list)

    def add(self, rule: str, remaining: int) -> None:
        self.steps.append((rule, remaining))

    @property
    def kept(self) -> int:
        return self.steps[-1][1] if self.steps else self.raw

    def as_dict(self) -> Dict[str, object]:
        return {
            "raw": self.raw,
            "steps": [{"rule": r, "remaining": n} for r, n in self.steps],
            "kept": self.kept,
        }


def _int(text: str) -> int:
    return int(float(text))


def _parse(text: str) -> datetime:
    return datetime.strptime(text.strip(), TIME_FMT)


# ---------------------------------------------------------------- Helios

def load_helios(path: str) -> Tuple[List[Record], FilterReport]:
    """Helios cluster_log.csv (Venus, Earth, Saturn, Uranus).

    Columns used: state, gpu_num, node_num, submit_time, duration.
    Rules: COMPLETED jobs; GPU jobs (gpu_num >= 1); single-server jobs
    (node_num == 1 and gpu_num <= 8); duration > 0.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    report = FilterReport(raw=len(rows))

    rows = [r for r in rows if r["state"].strip() == "COMPLETED"]
    report.add("state == COMPLETED", len(rows))

    rows = [r for r in rows if _int(r["gpu_num"]) >= 1]
    report.add("gpu_num >= 1 (GPU job)", len(rows))

    rows = [
        r for r in rows
        if _int(r["node_num"]) == 1 and _int(r["gpu_num"]) <= GPUS_PER_SERVER
    ]
    report.add("node_num == 1 and gpu_num <= 8 (single server)", len(rows))

    records: List[Record] = []
    for r in rows:
        duration = float(r["duration"])
        if duration > 0:
            records.append(Record(_parse(r["submit_time"]), _int(r["gpu_num"]), duration))
    report.add("duration > 0", len(records))
    return records, report


def helios_topology(gpu_csv: str, on_date: str, name: str) -> Tuple[dict, dict]:
    """Topology spec from cluster_gpu_number.csv.

    One cluster per virtual cluster (VC) that has GPUs on `on_date`
    (YYYY-MM-DD). Each cluster gets GPUs/8 servers of 8 GPUs.
    Returns (spec, info) where info holds the VC sum and the file's total.
    """
    with open(gpu_csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["date"].strip() == on_date]
    if len(rows) != 1:
        raise ValueError(f"{gpu_csv}: expected one row for {on_date}, found {len(rows)}")
    row = rows[0]

    clusters = []
    vc_sum = 0
    for key, value in row.items():
        if key is None or not key.startswith("vc"):
            continue  # skips 'date' and 'total'
        gpus = _int(value)
        if gpus == 0:
            continue
        if gpus % GPUS_PER_SERVER:
            raise ValueError(f"VC {key} has {gpus} GPUs, not a multiple of {GPUS_PER_SERVER}")
        clusters.append({"servers": gpus // GPUS_PER_SERVER, "gpus_per_server": GPUS_PER_SERVER})
        vc_sum += gpus

    spec = {
        "name": name.lower(),
        "note": f"Built from {gpu_csv} on {on_date}: one cluster per virtual cluster.",
        "clusters": clusters,
    }
    info = {"clusters": len(clusters), "gpus": vc_sum, "file_total": _int(row["total"])}
    return spec, info


# ---------------------------------------------------------------- Philly

def load_philly(path: str) -> Tuple[List[Record], FilterReport]:
    """Philly JSON job log (msr-fiddle/philly-traces).

    Each job has status, submitted_time and attempts. Each attempt has
    start_time, end_time and detail (one entry per machine, each with a
    list of GPUs). We use the last attempt for GPU count and run time.
    Rules: status == Pass; has attempts; single machine in last attempt;
    at least one GPU; submit/start/end times present; duration > 0.
    """
    with open(path, encoding="utf-8") as fh:
        jobs = json.load(fh)
    report = FilterReport(raw=len(jobs))

    passed = [j for j in jobs if j.get("status") == "Pass"]
    report.add('status == "Pass"', len(passed))

    with_attempts = [j for j in passed if j.get("attempts")]
    report.add("has at least one attempt", len(with_attempts))

    def last_detail(job):
        return job["attempts"][-1].get("detail", [])

    single = [j for j in with_attempts if len(last_detail(j)) == 1]
    report.add("single machine in last attempt", len(single))

    gpu_jobs = [j for j in single if len(last_detail(j)[0].get("gpus", [])) >= 1]
    report.add("at least one GPU", len(gpu_jobs))

    def has_time(value) -> bool:
        return value not in (None, "", "None")

    # Some jobs never started, so their times are the text "None" or null.
    timed = [
        j for j in gpu_jobs
        if has_time(j.get("submitted_time"))
        and has_time(j["attempts"][-1].get("start_time"))
        and has_time(j["attempts"][-1].get("end_time"))
    ]
    report.add("submit, start and end times present", len(timed))

    records: List[Record] = []
    for j in timed:
        last = j["attempts"][-1]
        start = _parse(last["start_time"])
        end = _parse(last["end_time"])
        duration = (end - start).total_seconds()
        if duration > 0:
            records.append(
                Record(
                    _parse(j["submitted_time"]),
                    len(last_detail(j)[0]["gpus"]),
                    duration,
                )
            )
    report.add("duration > 0", len(records))
    return records, report


# ---------------------------------------------------------------- windows

def window(records: Sequence[Record], start: datetime, days: int) -> List[Record]:
    """Jobs submitted in [start, start + days)."""
    end = start + timedelta(days=days)
    return [r for r in records if start <= r.submit < end]


def best_window_start(records: Sequence[Record], days: int, target: int) -> datetime:
    """Midnight start of the `days`-long window whose job count is closest to `target`."""
    first = min(r.submit for r in records).replace(hour=0, minute=0, second=0, microsecond=0)
    last = max(r.submit for r in records)
    best_gap, best_start = None, first
    day = first
    while day <= last:
        gap = abs(len(window(records, day, days)) - target)
        if best_gap is None or gap < best_gap:
            best_gap, best_start = gap, day
        day += timedelta(days=1)
    return best_start


def to_jobs(records: Sequence[Record], origin: datetime) -> List[Job]:
    """Simulator Jobs. Arrival time = seconds since `origin` (the window start)."""
    ordered = sorted(records, key=lambda r: r.submit)
    return [
        Job(
            job_id=i,
            arrival_time=(r.submit - origin).total_seconds(),
            gpus=r.gpus,
            duration=r.duration,
        )
        for i, r in enumerate(ordered)
    ]