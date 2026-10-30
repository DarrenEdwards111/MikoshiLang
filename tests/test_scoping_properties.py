"""Property-based scoping tests (hypothesis): shadowing, capture avoidance,
Module closures and Block save/restore."""
import os

import pytest
import sympy as sp

from mikoshilang import Session
from mikoshilang.math_ops import to_sympy

from .test_scoping import _dump

# ───────────── property-based ─────────────

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as hst, HealthCheck  # noqa: E402

SETTINGS = settings(max_examples=int(os.environ.get("SCOPING_EXAMPLES", 60)), deadline=None, suppress_health_check=[HealthCheck.too_slow])
names = hst.sampled_from(["a", "b", "x", "y"])
ints = hst.integers(min_value=-20, max_value=20)


@SETTINGS
@given(names, names, ints, ints)
def test_prop_nested_function_shadowing(outer, inner, m, n):
    """Function[o, Function[i, 100 o + i]][m][n]: the inner binder shadows when o == i."""
    r = Session().evaluate(f"Function[{outer}, Function[{inner}, 100*{outer} + {inner}]][{m}][{n}]")
    assert r == (101 * n if outer == inner else 100 * m + n)


@SETTINGS
@given(names, names, names, ints, hst.integers(min_value=1, max_value=9))
def test_prop_capture_avoidance(outer, inner, arg, n, k):
    """A free symbol passed in is never captured by an inner binder of the same name."""
    r = Session().evaluate(f"Function[{outer}, Function[{inner}, {inner} + {k}*{outer}]][{arg}][{n}]")
    expected = n + k * n if outer == inner else n + k * sp.Symbol(arg)
    assert sp.simplify(to_sympy(r) - expected) == 0


@SETTINGS
@given(names, names, ints, ints)
def test_prop_with_matches_function(v, w, m, n):
    """With[{v = m}, Function[w, w + 2 v]][n] behaves like the corresponding lambda."""
    r = Session().evaluate(f"With[{{{v} = {m}}}, Function[{w}, {w} + 2*{v}]][{n}]")
    assert r == (3 * n if v == w else n + 2 * m)


@SETTINGS
@given(hst.lists(hst.tuples(ints, ints), min_size=1, max_size=5))
def test_prop_module_closures_keep_values(pairs):
    s = Session()
    src = "{" + ", ".join(f"Module[{{x = {k}}}, Function[y, x*y]][{m}]" for k, m in pairs) + "}"
    assert s.evaluate(src) == s.evaluate("{" + ", ".join(str(k * m) for k, m in pairs) + "}")
    # Each closure escaped its Module, so its local is retained (never cleared
    # while it may still be referenced), but nothing else is left behind.
    assert set(s.state.own) == s.state.temporaries and len(s.state.temporaries) == len(pairs)
    s.evaluate(f"Module[{{x = 1}}, Function[y, x*y][2]]")
    assert len(s.state.temporaries) == len(pairs)


definition = hst.one_of(
    hst.builds(lambda k, v: f"f[{k}] = {v}", ints, ints),
    hst.builds(lambda c: f"f[x_Integer] := x + {c}", ints),
    hst.builds(lambda c: f"f[x_, y_] := x*y + {c}", ints),
    hst.builds(lambda k, v: f"f[{k}][z_] := z + {v}", ints, ints),
    hst.builds(lambda v: f"f = {v}", ints),
    hst.sampled_from(["SetAttributes[f, Listable]", "SetAttributes[f, Orderless]", "Clear[f]"]),
)


@SETTINGS
@given(hst.lists(definition, max_size=5), hst.lists(definition, max_size=5),
       hst.sampled_from(["", "Throw[0]", "Return[1]", "Part[{1}, 9]", "Break[]"]))
def test_prop_block_restores_everything(outside, inside, exit_):
    s = Session()
    for d in outside:
        s.evaluate(d)
    before = _dump(s, "f")
    body = "; ".join(inside + ([exit_] if exit_ else [])) or "Null"
    try:
        s.evaluate(f"Catch[Block[{{f}}, {body}]]")
    except Exception:
        pass
    assert _dump(s, "f") == before
