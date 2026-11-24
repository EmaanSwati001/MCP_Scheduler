"""Clone the two public trace repositories into data/raw/ (large, git-ignored).

    python scripts/download_traces.py
"""
from __future__ import annotations

import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

REPOS = {
    "HeliosData": "https://github.com/S-Lab-System-Group/HeliosData",
    "philly-traces": "https://github.com/msr-fiddle/philly-traces",
}

# If a file is only a few hundred bytes, it is a Git LFS pointer, not the data.
EXPECTED = [
    os.path.join("HeliosData", "Venus", "cluster_log.csv"),
    os.path.join("HeliosData", "Venus", "cluster_gpu_number.csv"),
]


def main() -> None:
    os.makedirs(RAW, exist_ok=True)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    for name, url in REPOS.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest):
            print(f"{name}: already present, skipping")
            continue
        subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True, env=env)

    for rel in EXPECTED:
        path = os.path.join(RAW, rel)
        if os.path.exists(path):
            size = os.path.getsize(path)
            flag = "  <-- looks like an LFS pointer, check the file" if size < 1000 else ""
            print(f"{rel}: {size:,} bytes{flag}")
        else:
            print(f"{rel}: NOT FOUND (check the folder layout under data/raw/)")

    print("Next: find the Philly JSON job log under data/raw/philly-traces and pass its path")
    print("      to scripts/prepare_traces.py --philly ...")


if __name__ == "__main__":
    main()