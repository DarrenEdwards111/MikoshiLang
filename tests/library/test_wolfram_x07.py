"""Differential tests for coverage batch X07 (graphs III: graph plots, communities & partitions, tree and fractal
plots).  Work in progress: the full case list follows."""
from _harness import Pred, check, ids

CASES = [
    ("GraphPlot[{1 -> 2, 2 -> 3}]", Pred(lambda g: g.head == "Graphics", "Graphics")),
]


import pytest  # noqa: E402


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_x07(case):
    check(case)
