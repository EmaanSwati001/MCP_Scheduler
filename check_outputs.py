"""Check the prepared data and configs against the expected values.

Run from the project folder:  python check_outputs.py
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
failures = 0


def check(label, ok, detail=""):
    global failures
    status = "PASS" if ok else "FAIL"
    print(f"{status}  {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures += 1


def load_rows(rel):
    with open(os.path.join(ROOT, rel), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check_jobs(name, expected_count, window_days):
    rel = os.path.join("data", "clean", f"{name}.csv")
    if not os.path.exists(os.path.join(ROOT, rel)):
        check(f"{name}: job file exists", False, rel)
        return
    rows = load_rows(rel)
    check(f"{name}: job count", len(rows) == expected_count,
          f"{len(rows)} vs expected {expected_count}")
    ids = [r["job_id"] for r in rows]
    check(f"{name}: job ids unique", len(set(ids)) == len(ids))
    gpus = [int(r["gpus"]) for r in rows]
    check(f"{name}: every job uses 1 to 8 GPUs", all(1 <= g <= 8 for g in gpus))
    arrivals = [float(r["arrival_time"]) for r in rows]
    check(f"{name}: arrivals inside the window",
          min(arrivals) >= 0 and max(arrivals) < window_days * 86400)
    check(f"{name}: arrivals sorted", arrivals == sorted(arrivals))
    durations = [float(r["duration"]) for r in rows]
    check(f"{name}: durations positive", all(d > 0 for d in durations))


def check_config(rel, expected_clusters, expected_gpus):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        check(f"{rel}: config exists", False)
        return
    with open(path, encoding="utf-8") as fh:
        clusters = json.load(fh)["clusters"]
    gpus = sum(c["servers"] * c["gpus_per_server"] for c in clusters)
    check(f"{rel}: clusters and GPUs",
          len(clusters) == expected_clusters and gpus == expected_gpus,
          f"{len(clusters)} clusters, {gpus} GPUs")


def check_provenance():
    path = os.path.join(ROOT, "results", "data_provenance.json")
    if not os.path.exists(path):
        check("provenance file exists", False, path)
        return
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    blocks = dict(data["helios"])
    blocks["philly"] = data["philly"]
    for name, block in blocks.items():
        counts = [block["filters"]["raw"]] + [s["remaining"] for s in block["filters"]["steps"]]
        check(f"{name}: filter counts only go down",
              all(a >= b for a, b in zip(counts, counts[1:])))
        check(f"{name}: window keeps no more than the filters kept",
              block["jobs_in_window"] <= block["filters"]["kept"])


def main():
    check_jobs("venus", 6941, 15)
    check_jobs("earth", 23815, 15)
    check_jobs("saturn", 24921, 15)
    check_jobs("uranus", 16858, 15)
    check_jobs("philly", 5043, 7)
    check_config(os.path.join("configs", "venus15.json"), 15, 1080)
    check_config(os.path.join("configs", "earth.json"), 19, 1232)
    check_config(os.path.join("configs", "saturn.json"), 23, 2096)
    check_config(os.path.join("configs", "uranus.json"), 15, 2072)
    check_provenance()
    print()
    print("All checks passed" if failures == 0 else f"{failures} check(s) failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()