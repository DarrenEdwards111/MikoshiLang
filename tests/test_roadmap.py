"""docs/wolfram_coverage_roadmap.md and .csv must match the live function registry."""
import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
OPTIONAL = ["scipy", "matplotlib", "sklearn", "networkx", "plotly", "numpy"]


@pytest.mark.skipif(any(importlib.util.find_spec(m) is None for m in OPTIONAL),
                    reason="the registry is only complete with every optional dependency installed")
def test_roadmap_is_up_to_date():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "wolfram_roadmap.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_roadmap_counts_are_consistent():
    import csv
    rows = list(csv.DictReader((ROOT / "docs" / "wolfram_coverage_roadmap.csv").open()))
    assert len(rows) == len((ROOT / "scripts" / "wolfram_data" / "documented.txt").read_text().split())
    assert {r["status"] for r in rows} <= {"implemented", "missing", "infeasible"}
    assert all(r["priority"] in ("1", "2", "3") for r in rows if r["status"] != "infeasible")
