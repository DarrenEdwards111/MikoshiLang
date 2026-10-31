"""Interop phase 2: packed arrays (NumPy fast paths), PythonObject handles, global @mikoshilang.function and
the inexact-zero follow-up (docs/python_interop.md)."""
import fractions
import math

import numpy as np
import pytest
import sympy as sp

import mikoshilang
from mikoshilang import Session, Expr, Symbol
from mikoshilang.packed import PACKING_THRESHOLD, PackedArray, pack


def _unpacked(v):
    """The same value as plain nested Lists (FromPackedArray)."""
    if isinstance(v, Expr):
        return Expr(v.head, *(_unpacked(a) for a in v.args))
    return v


# ── packed arrays: representation ──

def test_packed_array_is_a_list():
    s = Session()
    p = s.eval(np.array([[1.0, 2.0], [3.0, 4.0]]), out="expr")
    assert isinstance(p, PackedArray) and p.head == "List"
    assert p == s.evaluate("{{1., 2.}, {3., 4.}}") and s.evaluate("{{1., 2.}, {3., 4.}}") == p
    assert hash(p) == hash(_unpacked(p))
    s.state.own["p"] = p
    assert s.evaluate("Head[p]") == Symbol("List") and s.evaluate("p[[2, 1]]") == 3.0
    assert s.evaluate("MatchQ[p, {{__Real}..}]") is True
    assert str(s.evaluate("Map[f, p[[1]]]")) == "{f[1.0], f[2.0]}"
    assert s.eval("p", arrays="numpy").tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_packing_rules():
    assert pack(Expr("List", 1, 2)).array.dtype == np.int64
    assert pack(Expr("List", 1, 2.5)).array.dtype == np.float64
    assert pack(Expr("List", Symbol("x"))) is None
    assert pack(Expr("List", 2 ** 70)) is None
    assert pack(Expr("List")) is None
    s = Session()
    assert not isinstance(s.eval(np.array([np.nan]), out="expr") if False else s.eval(np.array([1, 2]), out="expr"),
                          list)
    with pytest.raises(ValueError):
        s.eval(np.array([1.0, np.nan]))                          # NaN still has no MikoshiLang value
    assert s.eval(np.array([1.0, np.inf])) == [1.0, math.inf]    # not packable: element-wise path
    assert s.eval(np.array([True, False])) == [True, False]      # booleans are not packed


# ── packed arrays: NumPy fast paths agree with the element-wise (exact) path ──

DIFFERENTIAL = [
    "p . q", "p . v", "v . p", "v . v", "Total[p]", "Total[v]", "Mean[p]", "Mean[v]", "Max[v]", "Min[p]",
    "Transpose[p]", "Length[p]", "Dimensions[p]", "N[iv]", "p + 1", "p + q", "2 p", "p - q", "p / 2",
    "p + 1/3", "p^2", "Sqrt[q]", "Sin[v]", "Exp[v]", "Log[q]", "Abs[-p]", "iv + 1", "3 iv", "iv^3", "Total[iv]",
    "Mean[iv]", "iv . iv", "Normal[p]",
]


@pytest.mark.parametrize("src", DIFFERENTIAL)
def test_fast_path_matches_unpacked(src):
    rng = np.random.default_rng(7)
    data = {"p": rng.random((3, 3)) - 0.5, "q": rng.random((3, 3)) + 0.5, "v": rng.random(3), "iv": np.arange(1, 5)}
    fast, slow = Session(), Session()
    for k, a in data.items():
        fast.state.own[k] = fast.eval(a, out="expr")
        slow.state.own[k] = _unpacked(fast.state.own[k])
    got, ref = fast.eval(src), slow.eval(src)
    np.testing.assert_allclose(np.array(got, dtype=complex), np.array(ref, dtype=complex), rtol=1e-12)
    assert _types(got) == _types(ref)                                         # exact stays exact


def _types(v):
    return [_types(x) for x in v] if isinstance(v, list) else type(v)


def test_fast_path_falls_back_when_numpy_could_differ():
    s = Session()
    s.state.own["iv"] = s.eval(np.array([1, 2, 3]), out="expr")
    s.state.own["neg"] = s.eval(np.array([-1.0, 4.0]), out="expr")
    big = s.eval(np.array([2 ** 61, 2 ** 61]), out="expr")
    s.state.own["big"] = big
    assert s.eval("iv / 2") == [fractions.Fraction(1, 2), 1, fractions.Fraction(3, 2)]    # exact rationals
    assert s.eval("Total[big]") == 2 ** 62                                      # no int64 overflow
    assert s.eval("big . big") == 2 ** 123
    assert s.eval("big * 4") == [2 ** 63, 2 ** 63]
    assert s.eval("Sqrt[neg]") == [1j, 2.0]                                     # Sqrt[-1.] is complex: exact path
    assert str(s.eval("Sin[iv]")) == "[Sin[1], Sin[2], Sin[3]]"                 # exact input stays exact
    assert s.eval("0 iv") == [0, 0, 0] and s.eval("0 neg") == [0, 0]           # exact 0 factor: exact 0
    assert str(s.eval("iv + x")) == "[1 + x, 2 + x, 3 + x]"


def test_large_plain_lists_are_packed_by_dot_total_mean():
    s = Session()
    n = int(math.isqrt(PACKING_THRESHOLD)) + 2
    r = s.evaluate(f"m = RandomReal[1, {{{n}, {n}}}]; m . m")
    assert isinstance(r, PackedArray) and r.array.shape == (n, n)
    m = s.eval("m", arrays="numpy")
    np.testing.assert_allclose(r.array, m @ m, rtol=1e-12)
    assert s.eval(f"Total[Range[{PACKING_THRESHOLD}]]") == PACKING_THRESHOLD * (PACKING_THRESHOLD + 1) // 2
    assert s.eval(f"Mean[Range[{PACKING_THRESHOLD}]]") == fractions.Fraction(PACKING_THRESHOLD + 1, 2)
    assert not isinstance(s.evaluate("{{1., 2.}, {3., 4.}} . {{1., 0.}, {0., 1.}}"), PackedArray)  # small: as before


def test_numericarray_is_packed():
    s = Session()
    assert s.eval('PackedArrayQ[Normal[NumericArray[{{1, 2}, {3, 4}}, "Integer16"]]]') is True
    assert s.eval('Normal[NumericArray[{1.5, 2.5}, "Real64"]] . {2., 2.}') == 8.0
    assert s.eval('NumericArrayQ[NumericArray[{1, 2}, "Real32"]]') is True
    assert s.eval('Normal[NumericArray[{2^64 - 1}, "UnsignedInteger64"]]') == [2 ** 64 - 1]   # beyond int64


# ── PythonObject handles ──

def test_python_import_and_handles():
    s = Session(allow_python={"numpy"})
    h = s.eval('np = PythonImport["numpy"]', out="expr")
    assert h.head == "PythonObject" and h.args[1] == "module numpy"
    assert s.eval("np") is np                                                  # resolves in Session.eval
    assert s.eval('PythonFunction[np, "linalg.det"][{{2, 0}, {0, 3}}]') == pytest.approx(6.0)
    rng = s.eval('rng = PythonFunction[np, "random.default_rng"][1]', out="expr")
    assert rng.head == "PythonObject" and "Generator" in rng.args[1]           # unconvertible result: a handle
    assert 0 <= s.eval('PythonFunction[rng, "integers"][10]') < 10
    assert s.eval('PythonAttribute[np, "pi"]') == math.pi
    assert s.eval('PythonFunction[np, "sum"][PythonFunction[np, "arange"][5]]') == 10
    assert str(s.eval("PythonImport[m]")) == "PythonImport[m]"                 # symbolic: unevaluated


def test_handle_gating_and_isolation():
    s = Session(allow_python={"numpy"})
    s.eval('np = PythonImport["numpy"]')
    with pytest.raises(PermissionError, match="^PythonImport: using os"):
        s.eval('PythonImport["os"]')
    with pytest.raises(PermissionError, match="private and dunder"):
        s.eval('PythonAttribute[np, "__dict__"]')
    with pytest.raises(PermissionError):
        Session().eval('PythonImport["numpy"]')
    other = Session(allow_python=True)
    with pytest.raises(ValueError, match="does not exist in this session"):
        other.eval('PythonFunction[PythonObject[1, "module numpy"]][1]')
    with pytest.raises(ValueError, match="has no attribute"):
        s.eval('PythonAttribute[np, "nope"]')
    with pytest.raises(TypeError, match="^PythonFunction: expected a dotted name or a PythonObject handle"):
        s.eval("PythonFunction[f[1]][2]")
    assert s.eval('Check[PythonImport["os"], "denied"]') == "denied"
    s.clear()
    with pytest.raises(ValueError):
        s.eval('PythonFunction[PythonObject[1, "module numpy"], "sum"][{1}]')   # clear() drops handles


# ── global @mikoshilang.function ──

@pytest.fixture
def cleanup():
    names = []
    yield names
    for n in names:
        try:
            mikoshilang.unregister(n)
        except ValueError:
            pass


def test_global_function_visible_in_every_session(cleanup):
    @mikoshilang.function(check="exact", listable=True)
    def triangle_number(n: "NonNegInteger") -> "Integer":
        return n * (n + 1) // 2
    cleanup.append("TriangleNumber")
    assert Session().eval("TriangleNumber[{1, 2, 10}]") == [1, 3, 55]        # OEIS A000217
    assert Session().eval("TriangleNumber[k]") == Expr("TriangleNumber", Symbol("k"))
    with pytest.raises(ValueError, match="^TriangleNumber: argument 1 must be a non-negative integer"):
        Session().eval("TriangleNumber[-1]")
    area, status, notes = mikoshilang.library.STATUS["TriangleNumber"]
    assert status == "added" and "Check: exact" in notes
    mikoshilang.unregister("TriangleNumber")
    assert "TriangleNumber" not in mikoshilang.library.STATUS
    assert str(Session().eval("TriangleNumber[3]")) == "TriangleNumber[3]"


def test_global_function_override_and_restore(cleanup):
    before = Session().eval("Mean[{1, 2}]")
    with pytest.raises(ValueError, match="library head"):
        mikoshilang.function(lambda v: v, name="Mean")
    with pytest.raises(ValueError, match="kernel built-in"):
        mikoshilang.function(lambda v: v, name="Plus", override=True)

    @mikoshilang.function(name="Mean", override=True)
    def mine(v: list) -> str:
        return "mine"
    cleanup.append("Mean")
    assert Session().eval("Mean[{1, 2}]") == "mine"
    mikoshilang.unregister("Mean")
    assert Session().eval("Mean[{1, 2}]") == before == fractions.Fraction(3, 2)
    with pytest.raises(ValueError, match="was not registered"):
        mikoshilang.unregister("Mean")


def test_session_function_shadows_global(cleanup):
    @mikoshilang.function(name="Greeting")
    def g1() -> str:
        return "global"
    cleanup.append("Greeting")
    s = Session()

    @s.function(name="Greeting", override=True)
    def g2() -> str:
        return "session"
    assert s.eval("Greeting[]") == "session" and Session().eval("Greeting[]") == "global"


# ── inexact zero, follow-up ──

@pytest.mark.parametrize("src, want", [
    ("0. + x", "0. + x"), ("x + 1.5 - 1.5", "0. + x"), ("x + 0.", "0. + x"),
])
def test_inexact_zero_term_kept(src, want):
    s = Session()
    got = s.evaluate(src)
    assert isinstance(got, Expr) and got.head == "Plus" and got.args[0] == 0.0 and type(got.args[0]) is float
    assert got == s.evaluate(want)


@pytest.mark.parametrize("src", ["Sin[0.]", "Tan[0.]", "ArcSin[0.]", "Sinh[0.]", "Sqrt[0.]", "Log[1.]", "Abs[0.]",
                                 "ArcTan[0.]", "Tanh[0.]"])
def test_elementary_functions_at_inexact_zero(src):
    v = Session().evaluate(src)
    assert type(v) is float and v == 0.0


@pytest.mark.parametrize("src", ["Sin[0]", "Log[1]", "x + 0", "x - x", "x + 1 - 1", "Sign[0.]", "Floor[0.5]"])
def test_exact_zero_unchanged(src):
    v = Session().evaluate(src)
    assert not isinstance(v, float)


def test_inexact_zero_bignum_function():
    v = Session().evaluate("Sin[N[Pi, 30] - N[Pi, 30]]")
    assert isinstance(v, sp.Float) and v.is_zero and v._prec > 53
