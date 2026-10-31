"""Round-trip properties of the interop conversion table (hypothesis; skipped without it)."""
import math

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

from mikoshilang import Session, Expr  # noqa: E402
from mikoshilang.interop import from_python, to_python  # noqa: E402


def _data():
    scalars = (st.integers(-10**30, 10**30) | st.floats(allow_nan=False) | st.text(max_size=8) | st.booleans()
               | st.none() | st.fractions(max_denominator=10**6).filter(lambda f: f.denominator != 1)
               | st.complex_numbers(allow_nan=False, allow_infinity=False).filter(lambda c: c.imag != 0))
    keys = st.text(max_size=5) | st.integers(-100, 100)
    return st.recursive(scalars, lambda c: st.lists(c, max_size=4) | st.dictionaries(keys, c, max_size=4),
                        max_leaves=12)


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b and math.copysign(1, a) == math.copysign(1, b) or (a == b == 0.0)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


@settings(max_examples=300, deadline=None)
@given(_data())
def test_python_roundtrip(v):
    back = to_python(from_python(v))
    assert _same(back, v), (v, back)


@settings(max_examples=200, deadline=None)
@given(_data())
def test_roundtrip_survives_evaluation(v):
    """Converted data is already in evaluated (normal) form: evaluating it does not change the Python value."""
    s = Session()     # wrapped in Identity: a bare Python str passed to eval would be parsed as source code
    assert _same(s.eval(Expr("Identity", from_python(v))), v)
