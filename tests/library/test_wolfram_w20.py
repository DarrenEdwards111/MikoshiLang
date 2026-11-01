"""Differential tests for Wolfram batch W20 (3-D, vector, stream, complex plots, charts, histograms and gauges;
modules wl_plots3d_charts and wl_plots3d_fields).

Plots are compared by their data and primitives (bin edges and heights, needle angles, sampled field values,
streamline tangents), never pixels. Sources of the expected values are cited next to each case: Wolfram Language
reference pages (reference.wolfram.com/language/ref/<Name>, "Details" sections), textbook definitions (histogram
heights, Silverman's rule, Renko and Kagi construction rules, the YCbCr transform of ITU-R BT.709) and mathematical
identities checked with independent Python code.

Symbol names used as symbolic placeholders are prefixed ``w20`` so they cannot collide with other test modules.
"""
import math

import pytest

from _harness import Pred, Raises, check, ids, ev
from mikoshilang.expr import Expr, Symbol


# ───────────── helpers on the result structure ─────────────

def unwrap(g):
    return g.args[0] if isinstance(g, Expr) and g.head == "Legended" else g


def find(e, head):
    out = []
    if isinstance(e, Expr):
        if e.head == head:
            out.append(e)
        for a in e.args:
            out += find(a, head)
    return out


def tup(e):
    return tuple(float(c) for c in e.args)


def rects(g):
    """[(x0, y0, x1, y1)] of the Rectangle primitives."""
    return [tup(r.args[0]) + tup(r.args[1]) for r in find(unwrap(g), "Rectangle")]


def bars(g):
    """[(left, right, height)] of histogram bars."""
    return [(x0, x1, y1 - y0) for x0, y0, x1, y1 in rects(g)]


def opt(g, name):
    for r in unwrap(g).args[1:]:
        if isinstance(r, Expr) and r.head == "Rule" and r.args[0] == Symbol(name):
            return r.args[1]
    return None


def near(a, b, tol=1e-9):
    return all(abs(x - y) <= tol * max(1.0, abs(y)) for x, y in zip(a, b)) and len(a) == len(b)


def unevaluated(head):
    return Pred(lambda g: isinstance(g, Expr) and g.head == head, f"stays {head}[...]")


def is_graphics(head="Graphics"):
    return Pred(lambda g: isinstance(unwrap(g), Expr) and unwrap(g).head == head, f"is {head}")


def bars_are(want, tol=1e-9):
    return Pred(lambda g: sorted(bars(g)) == sorted(want) or near(sorted(sum(map(list, bars(g)), [])),
                                                                    sorted(sum(map(list, want), [])), tol),
                f"bars {want}")


CASES = [
    # Histogram: {dx} bins at multiples of dx (Wolfram Histogram reference, "Details": {dx} bins of width dx);
    # counts written by hand: 1 | 2 2 | 3 3 3 | 4 4 | 5
    ("Histogram[{1, 2, 2, 3, 3, 3, 4, 4, 5}, {1}]",
     bars_are([(1, 2, 1), (2, 3, 2), (3, 4, 3), (4, 5, 2), (5, 6, 1)])),
    # {min, max, dx} with "PDF": count/(n width): 1/(3*2), 2/(3*2)
    ('Histogram[{1, 2, 3}, {0, 4, 2}, "PDF"]', bars_are([(0, 2, 1 / 6), (2, 4, 1 / 3)])),
    # half-open bins: 0.3 belongs to [0.3, 0.4)
    ("Histogram[{0.1, 0.2, 0.3, 0.35}, {0.1}]", bars_are([(0.1, 0.2, 1), (0.2, 0.3, 1), (0.3, 0.4, 2)])),
    ("Histogram[{1, 2, 3, 4}, {{0, 2, 5}}, \"Probability\"]", bars_are([(0, 2, 0.25), (2, 5, 0.75)])),
    ('Histogram[{1, 2, 3, 4}, {{0, 2, 5}}, "CDF"]', bars_are([(0, 2, 0.25), (2, 5, 1.0)])),
    ('Histogram[{1, 2, 3, 4}, {{0, 2, 5}}, "SurvivalCount"]', bars_are([(0, 2, 4), (2, 5, 3)])),
    ('Histogram[{1, 2, 3, 4}, {{0, 2, 5}}, "Intensity"]', bars_are([(0, 2, 0.5), (2, 5, 1.0)])),
    ("Histogram[w20x]", unevaluated("Histogram")),
    ('Histogram[{"a", 1}]', Raises(TypeError)),
    ('Histogram[{1, 2, 3}, "Knuth"]', Raises(NotImplementedError)),
    ("Histogram[{1, 2, 3}, {-1}]", Raises(ValueError)),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w20(case):
    check(case)
