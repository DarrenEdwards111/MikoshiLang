"""Meta-test for the library audit: every exposed head has a recorded status and enough cases.

* every head in the pre-audit inventory (``inventory_heads.txt``, the 620 heads the legacy library
  exposed) has a status in ``library.STATUS``;
* every verified / fixed / promoted / added head has at least three differential cases;
* every error or quarantined head has at least one case asserting that it raises;
* every kernel head has at least one case.
"""
import importlib
import importlib.util
import json
import pathlib
import re

import pytest

from _harness import Raises

from mikoshilang import library

HERE = pathlib.Path(__file__).parent
MODULES = ["test_elementary", "test_special", "test_numbertheory", "test_combinatorics", "test_algebra",
           "test_calculus", "test_linalg", "test_lists_strings", "test_statistics", "test_applied",
           "test_logic_graphs", "test_numerics_transforms", "test_plots_ml", "test_kernelheads", "test_wolfram_fixes",
           "test_wolfram_batch1", "test_wolfram_batch2", "test_wolfram_batch3", "test_wolfram_batch4", "test_wolfram_batch5", "test_wolfram_batch6", "test_wolfram_w01", "test_wolfram_w02", "test_wolfram_w03", "test_wolfram_w04",
           "test_wolfram_w05", "test_wolfram_w06", "test_wolfram_w08", "test_wolfram_w07", "test_wolfram_w10", "test_wolfram_w09", "test_wolfram_w11", "test_wolfram_w13", "test_wolfram_w12",
           "test_wolfram_w14", "test_wolfram_interop", "test_wolfram_w17", "test_wolfram_w23", "test_wolfram_w24", "test_wolfram_w15", "test_wolfram_w19", "test_wolfram_w16", "test_wolfram_w21", "test_wolfram_w22"]
OPTIONAL = ["scipy", "matplotlib", "sklearn", "networkx", "plotly", "numpy"]
NAME = re.compile(r"\s*([A-Za-z][A-Za-z0-9]*)")

pytestmark = pytest.mark.skipif(any(importlib.util.find_spec(m) is None for m in OPTIONAL),
                                reason="case counts are only complete with every optional dependency installed")


def _head(src):
    m = NAME.match(src)
    return m.group(1) if m else None


IDENT = re.compile(r"(?<![A-Za-z0-9\"$])(\$?[A-Z][A-Za-z0-9]*)")


def _cases():
    top, used = {}, {}
    for mod in MODULES:
        for c in importlib.import_module(mod).CASES:
            top.setdefault(_head(c[0]), []).append(c)
            for name in set(IDENT.findall(c[0])):
                used.setdefault(name, []).append(c)
    return top, used


# CASES: by outermost head (for Raises checks); USED: every case in which a head occurs, so that
# e.g. Mean[ExponentialDistribution[2]] counts as a case for ExponentialDistribution.
CASES, USED = _cases() if not pytestmark.args[0] else ({}, {})
INVENTORY = [h for h in (HERE / "inventory_heads.txt").read_text().split() if h]


def test_inventory_size():
    assert len(INVENTORY) == 620


def test_every_inventory_head_has_status():
    missing = [h for h in INVENTORY if h not in library.STATUS]
    assert not missing, f"heads without an audit status: {missing}"


def test_tested_heads_have_three_cases():
    thin = {h: len(USED.get(h, [])) for h, (_a, s, _n) in library.STATUS.items()
            if s in ("verified", "fixed", "promoted", "added") and len(USED.get(h, [])) < 3}
    assert not thin, f"verified/fixed/promoted/added heads with fewer than 3 cases: {thin}"


def test_error_and_quarantined_heads_raise():
    bad = [h for h, (_a, s, _n) in library.STATUS.items() if s in ("error", "quarantined")
           and not any(isinstance(c[1], Raises) for c in CASES.get(h, []))]
    assert not bad, f"error/quarantined heads without a Raises case: {bad}"


def test_kernel_heads_have_a_case():
    bad = [h for h, (_a, s, _n) in library.STATUS.items() if s == "kernel" and not USED.get(h)]
    assert not bad, f"kernel heads without a case: {bad}"


def test_quarantine_list_is_disjoint_from_reviewed_heads():
    names = json.loads((pathlib.Path(library.__file__).parent.parent / "quarantined.json").read_text())
    names = names if isinstance(names, list) else list(names)
    overlap = [n for n in names if library.STATUS.get(n, (0, "quarantined"))[1] != "quarantined"]
    assert not overlap, f"reviewed heads still on the quarantine list: {overlap[:20]}"


def test_every_quarantined_name_raises():
    """Each remaining quarantined name must raise NotImplementedError, never return a value."""
    from mikoshilang import Session
    names = json.loads((pathlib.Path(library.__file__).parent.parent / "quarantined.json").read_text())
    returned = []
    for n in names:
        try:
            Session().evaluate(f"{n}[1]")
            returned.append(n)
        except NotImplementedError:
            pass
        except Exception as exc:  # any other exception is also not a silent answer, but should not happen
            returned.append(f"{n}: {type(exc).__name__}")
    assert not returned, f"quarantined names that did not raise NotImplementedError: {returned[:20]}"
