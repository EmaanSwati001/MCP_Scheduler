"""Turn the raw Helios and Philly traces into the common job CSV and topology configs.

Run after scripts/download_traces.py. Example:

    python scripts/prepare_traces.py --helios-dir data/raw/HeliosData \
        --philly path/to/philly_job_log.json

Writes:
    data/clean/venus.csv, earth.csv, saturn.csv, uranus.csv, philly.csv
    configs/venus15.json, earth.json, saturn.json, uranus.json
    results/data_provenance.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcgsched.workloads import save_jobs_csv  # noqa: E402
from mcgsched.workloads.traces import (  # noqa: E402
    best_window_start,
    helios_topology,
    load_helios,
    load_philly,
    to_jobs,
    window,
)

HELIOS_CLUSTERS = ["Venus", "Earth", "Saturn", "Uranus"]
HELIOS_START = datetime(2020, 9, 4)   # paper's Venus window starts here
HELIOS_DAYS = 15
TOPOLOGY_DATE = "2020-09-04"          # day used to read the GPU count per VC
PHILLY_DAYS = 7
PHILLY_TARGET = 4992                  # paper's Philly job count


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Helios and Philly traces")
    ap.add_argument("--helios-dir", required=True, help="folder with Venus/, Earth/, ... subfolders")
    ap.add_argument("--philly", required=True, help="path to the Philly JSON job log")
    ap.add_argument("--clean-dir", default=os.path.join(ROOT, "data", "clean"))
    ap.add_argument("--configs-dir", default=os.path.join(ROOT, "configs"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "data_provenance.json"))
    args = ap.parse_args()

    os.makedirs(args.clean_dir, exist_ok=True)
    os.makedirs(args.configs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    provenance = {"helios": {}, "philly": {}}

    # ---- Helios: Venus, Earth, Saturn, Uranus (same pipeline for all four)
    for name in HELIOS_CLUSTERS:
        folder = os.path.join(args.helios_dir, name)
        log = os.path.join(folder, "cluster_log.csv")
        gpu_file = os.path.join(folder, "cluster_gpu_number.csv")
        if not (os.path.exists(log) and os.path.exists(gpu_file)):
            print(f"{name:<7} MISSING: expected {log} and {gpu_file}; skipped")
            continue

        records, report = load_helios(log)
        in_window = window(records, HELIOS_START, HELIOS_DAYS)
        jobs = to_jobs(in_window, HELIOS_START)
        save_jobs_csv(jobs, os.path.join(args.clean_dir, f"{name.lower()}.csv"))

        spec, topo_info = helios_topology(gpu_file, TOPOLOGY_DATE, name)
        cfg_name = "venus15.json" if name == "Venus" else f"{name.lower()}.json"
        with open(os.path.join(args.configs_dir, cfg_name), "w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=2)

        provenance["helios"][name] = {
            "log_sha256": sha256(log),
            "gpu_file_sha256": sha256(gpu_file),
            "filters": report.as_dict(),
            "window_start": HELIOS_START.isoformat(),
            "window_days": HELIOS_DAYS,
            "jobs_in_window": len(jobs),
            "mean_duration_s": round(mean(j.duration for j in jobs), 1),
            "topology": topo_info,
            "config": cfg_name,
        }
        print(
            f"{name:<7} jobs={len(jobs):<6} mean_duration={mean(j.duration for j in jobs):9.1f}s "
            f"clusters={topo_info['clusters']} gpus={topo_info['gpus']} "
            f"(file total {topo_info['file_total']})"
        )

    # ---- Philly: 7-day window whose job count is closest to the paper's 4,992
    records, report = load_philly(args.philly)
    start = best_window_start(records, PHILLY_DAYS, PHILLY_TARGET)
    in_window = window(records, start, PHILLY_DAYS)
    jobs = to_jobs(in_window, start)
    save_jobs_csv(jobs, os.path.join(args.clean_dir, "philly.csv"))
    provenance["philly"] = {
        "log_sha256": sha256(args.philly),
        "filters": report.as_dict(),
        "window_start": start.isoformat(),
        "window_days": PHILLY_DAYS,
        "jobs_in_window": len(jobs),
        "mean_duration_s": round(mean(j.duration for j in jobs), 1),
    }
    print(f"Philly  jobs={len(jobs):<6} mean_duration={mean(j.duration for j in jobs):9.1f}s "
          f"window_start={start.date()}")

    with open(args.results, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)
    print(f"wrote {args.results}")


if __name__ == "__main__":
    main()