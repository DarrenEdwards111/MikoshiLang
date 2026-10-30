"""Differential tests for the heads that used to raise (Root, Catalan, Degree, Trace, Refine,
Assumptions/$Assumptions/Assuming, LaplaceODESolve, InverseZTransform) or were quarantined
(YoungTableaux), plus DiscreteDelta and UnitStep, which InverseZTransform results use.

References: mpmath (polyroots, catalan, pi), SymPy dsolve / series, Wolfram documentation examples,
OEIS A000085 (involutions) and the hook length formula."""
import math

import mpmath
import pytest
import sympy as sp

from _harness import Raises, Pred, check, ids, to_complex

mpmath.mp.dps = 40


def wolfram_root_order(coeffs):
    """Roots of the polynomial (coefficients highest first) in Wolfram's Root order."""
    rs = mpmath.polyroots(coeffs, maxsteps=200, extraprec=200)
    real = sorted(float(mpmath.re(r)) for r in rs if abs(mpmath.im(r)) < 1e-25)
    cplx = sorted((complex(r) for r in rs if abs(mpmath.im(r)) >= 1e-25), key=lambda c: (round(c.real, 12), abs(c.imag), c.imag))
    return [complex(r) for r in real] + cplx


Q5 = wolfram_root_order([1, 0, 0, 0, -1, -1])         # x^5 - x - 1
C3 = wolfram_root_order([2, 0, -4, 1])                 # 2x^3 - 4x + 1
P7 = wolfram_root_order([1, 0, 0, 0, 0, 0, 3, 1])      # x^7 + 3x + 1


def close(v):
    return Pred(lambda r: abs(to_complex(r) - v) < 1e-12 * max(1, abs(v)), f"≈ {v}")


ROOT = [
    ("Root[#^2 - 2 &, 1]", "-Sqrt[2]"), ("Root[#^2 - 2 &, 2]", "Sqrt[2]"),
    ("Root[#^2 + 1 &, 1]", "-I"), ("Root[#^2 + 1 &, 2]", "I"),
    ("Root[#^3 - 2 &, 1]", "2^(1/3)"), ("Root[#^3 - 2 &, 2]", "-2^(1/3)/2 - I 2^(1/3) Sqrt[3]/2"),
    ("Root[# - 3/2 &, 1]", "3/2"), ("Root[(# - 1)^2 &, 2]", "1"),
    ("Root[#^5 - # - 1 &, 1]", "Root[-1 - # + #^5 &, 1]"),
    ("Root[2 #^3 - 4 # + 1 &, 2]", "Root[1 - 4 # + 2 #^3 &, 2]"),
    # numerical values and Wolfram ordering against mpmath.polyroots
    *[(f"N[Root[#^5 - # - 1 &, {k + 1}]]", close(Q5[k])) for k in range(5)],
    *[(f"N[Root[2 #^3 - 4 # + 1 &, {k + 1}]]", close(C3[k])) for k in range(3)],
    *[(f"N[Root[#^7 + 3 # + 1 &, {k + 1}]]", close(P7[k])) for k in range(7)],
    ("N[Root[#^5 - # - 1 &, 1], 30]", Pred(lambda r: abs(mpmath.mpf(str(r)) - mpmath.findroot(lambda x: x**5 - x - 1, 1.2)) < mpmath.mpf(10) ** -28, "30 digits")),
    # reduction to the irreducible factor that carries the root
    ("Root[(#^2 - 2) (#^5 - # - 1) &, 3]", "Sqrt[2]"), ("Root[(#^2 - 2) (#^5 - # - 1) &, 2]", "Root[-1 - # + #^5 &, 1]"),
    ("Root[#^3 - # - 1. &, 1]", 1.324717957244746),
    ("Root[#^5 - # - 1 &, 1]^5 - Root[#^5 - # - 1 &, 1] - 1 // N", Pred(lambda r: abs(to_complex(r)) < 1e-12, "is a root")),
    ("Root[#^3 - a &, 1]", "Root[#^3 - a &, 1]"),
    ("Root[x^2 - 2, 1]", Raises(ValueError, "pure function")), ("Root[#^2 - 2 &, 3]", Raises(ValueError, "degree")),
    ("Root[#^2 - 2 &, 0]", Raises(ValueError)), ("Root[3 &, 1]", Raises(ValueError)),
]

CONSTANTS = [
    ("N[Catalan]", float(mpmath.catalan)), ("N[Catalan, 30]", Pred(lambda r: abs(mpmath.mpf(str(r)) - mpmath.catalan) < mpmath.mpf(10) ** -29, "30 digits")),
    ("Sum[(-1)^k/(2 k + 1)^2, {k, 0, Infinity}]", "Catalan"), ("2 Catalan - Catalan", "Catalan"),
    ("Catalan[3]", Raises(ValueError, "CatalanNumber")),
    ("N[Degree]", float(mpmath.pi / 180)), ("30 Degree", "Pi/6"), ("Sin[30 Degree]", "1/2"), ("N[45 Degree, 25]", Pred(lambda r: abs(mpmath.mpf(str(r)) - mpmath.pi / 4) < mpmath.mpf(10) ** -24, "25 digits")),
    ("Degree[x^2, x]", Raises(ValueError, "Exponent")),
]

TRACE = [  # Wolfram documentation examples
    ("Trace[1 + 2 3]", Pred(lambda r: str(r) == "{{2*3, 6}, 1 + 6, 7}", "{{2 3, 6}, 1 + 6, 7}")),
    ("Trace[Sin[0]]", Pred(lambda r: str(r) == "{Sin[0], 0}", "{Sin[0], 0}")),
    ("a = 5; Trace[a]", Pred(lambda r: str(r) == "{a, 5}", "{a, 5}")),
    ("Trace[x]", "{}"), ("Trace[{{1, 2}, {3, 4}}]", "{}"),
    ("f[x_] := x^2; g[y_] := y + 1; Trace[g[f[3]]]", Pred(lambda r: str(r) == "{{f[3], 3^2, 9}, g[9], 9 + 1, 10}", "nested chain")),
    ("fac[0] = 1; fac[n_] := n fac[n - 1]; Trace[fac[2]]",
     Pred(lambda r: str(r) == "{fac[2], 2*fac[2 - 1], {{2 - 1, 1}, fac[1], fac[1 - 1], {{1 - 1, 0}, fac[0], 1}, 1*1, 1}, 2*1, 2}",
          "recursive chain")),
    ("Trace[1 + 2 3, _Times]", Pred(lambda r: str(r) == "{{2*3}}", "filtered")),
    ("Trace[Tr[{{1, 2}, {3, 4}}]]", Pred(lambda r: str(r) == "{Tr[{{1, 2}, {3, 4}}], 5}", "Tr step")),
    ("Trace[1 + 1]", Pred(lambda r: all(a.head == "HoldForm" for a in r.args), "HoldForm elements")),
    ("Trace[1, x, y]", Raises(NotImplementedError)),
    ("Tr[{{1, 2}, {3, 4}}]", "5"),
]

ASSUME = [  # Wolfram documentation examples for Refine / Simplify / Assuming / $Assumptions
    ("Refine[Sqrt[x^2], x > 0]", "x"), ("Refine[Sqrt[x^2], x < 0]", "-x"), ("Refine[Sqrt[x^2], Element[x, Reals]]", "Abs[x]"),
    ("Refine[Abs[x], x < 0]", "-x"), ("Refine[Abs[x - 1], x > 1]", "x - 1"),
    ("Refine[Sin[n Pi], Element[n, Integers]]", "0"), ("Refine[Cos[2 n Pi], Element[n, Integers]]", "1"),
    ("Refine[(-1)^(2 n), Element[n, Integers]]", "1"), ("Refine[Floor[n], Element[n, Integers]]", "n"),
    ("Refine[Log[Exp[x]], Element[x, Reals]]", "x"), ("Refine[Sqrt[x^2 y^2], x > 0 && y > 0]", "x y"),
    ("Refine[x > 0, x > 1]", True), ("Refine[x < 0, x > 1]", False), ("Refine[x^2 > 1, x > 1]", True),
    ("Refine[x > 2, x > 1]", "x > 2"), ("Refine[Sqrt[x^2]]", "Sqrt[x^2]"),
    ("Refine[Sqrt[x^2], {x > 0}]", "x"), ("Refine[Sqrt[x^2], Assumptions -> x > 0]", "x"),
    ("Refine[x, x > 0 && x < 0]", Raises(ValueError, "inconsistent")), ("Refine[x, Foo[x]]", Raises(ValueError, "unsupported")),
    ("Simplify[Sqrt[x^2], x > 0]", "x"), ("Simplify[Sqrt[x^2] + Abs[y], Assumptions -> x < 0 && y > 0]", "y - x"),
    ("Simplify[x > 0, x > 3]", True), ("Simplify[Sin[x]^2 + Cos[x]^2]", "1"), ("Simplify[Sqrt[x^2]]", "Sqrt[x^2]"),
    ("$Assumptions", True), ("$Assumptions = x > 0; Refine[Sqrt[x^2]]", "x"), ("$Assumptions = x > 0; Simplify[Abs[x]]", "x"),
    ("Assuming[x < 0, Refine[Sqrt[x^2]]]", "-x"), ("Assuming[x < 0, Simplify[Abs[x]]]", "-x"),
    ("Assuming[x > 0, 1]; $Assumptions", True), ("Assuming[a > 0, Integrate[Exp[-a x^2], {x, -Infinity, Infinity}]]", "Sqrt[Pi]/Sqrt[a]"),
    ("Assumptions[x]", Raises(ValueError, "option name")), ("Assumptions[]", Raises(ValueError)),
    ("Integrate[Exp[-a x], {x, 0, Infinity}]", "ConditionalExpression[1/a, Re[a] > 0]"),
    ("Integrate[Exp[-a x], {x, 0, Infinity}, Assumptions -> a > 0]", "1/a"),
    ("Integrate[Exp[-a x], {x, 0, Infinity}, Assumptions -> a < 0]", "Infinity"),
    ("Integrate[x^n, {x, 0, 1}, Assumptions -> n > 0]", "1/(n + 1)"),
    ("Integrate[x^2, {x, 0, 1}, Assumptions -> x > 0]", "1/3"),
    ("Integrate[x, x, Foo -> 1]", Raises(ValueError, "unknown option")),
]

t = sp.Symbol("t")
yf = sp.Function("y")


def matches_dsolve(ode, ics):
    ref = sp.dsolve(ode, yf(t), ics=ics).rhs

    def p(r):
        from mikoshilang.math_ops import to_sympy
        got = to_sympy(r.args[0].args[0].args[1])
        return sp.simplify(got.subs(sp.Symbol("t"), t) - ref) == 0
    return Pred(p, f"SymPy dsolve: {ref}")


LAPLACE = [
    ("LaplaceODESolve[{y''[t] + y[t] == 0, y[0] == 1, y'[0] == 0}, y, t]", "{{y[t] -> Cos[t]}}"),
    ("LaplaceODESolve[{y''[t] + 3 y'[t] + 2 y[t] == Exp[-3 t], y[0] == 0, y'[0] == 0}, y, t]",
     matches_dsolve(yf(t).diff(t, 2) + 3 * yf(t).diff(t) + 2 * yf(t) - sp.exp(-3 * t), {yf(0): 0, yf(t).diff(t).subs(t, 0): 0})),
    ("LaplaceODESolve[{y'[t] + 2 y[t] == 4, y[0] == 1}, y, t]", matches_dsolve(yf(t).diff(t) + 2 * yf(t) - 4, {yf(0): 1})),
    ("LaplaceODESolve[{y''[t] + y[t] == Sin[t], y[0] == 0, y'[0] == 0}, y, t]",
     matches_dsolve(yf(t).diff(t, 2) + yf(t) - sp.sin(t), {yf(0): 0, yf(t).diff(t).subs(t, 0): 0})),
    ("LaplaceODESolve[{y'''[t] - y[t] == 0, y[0] == 1, y'[0] == 1, y''[0] == 1}, y, t]", "{{y[t] -> E^t}}"),
    ("LaplaceODESolve[y'[t] == y[t], y, t]", "{{y[t] -> y[0] E^t}}"),
    ("LaplaceODESolve[y''[t] + 4 y[t] == 0, y, t]", "{{y[t] -> y[0] Cos[2 t] + y'[0] Sin[2 t]/2}}"),
    ("LaplaceODESolve[y'[t] == t y[t], y, t]", Raises(ValueError, "constant coefficients")),
    ("LaplaceODESolve[y'[t] == y[t]^2, y, t]", Raises(ValueError)),
    ("LaplaceODESolve[y[t] == 1, y, t]", Raises(ValueError, "no derivative")),
    ("LaplaceODESolve[y'[t] == 1, y[t], t]", Raises(ValueError, "function name")),
]

z, n = sp.symbols("z n")


def izt_series(F, N=10):
    """Reference: f(n) = coefficient of z^-n in the expansion of F at z = Infinity."""
    w = sp.Symbol("w")
    s = sp.series(F.subs(z, 1 / w), w, 0, N).removeO()
    return [s.coeff(w, k) for k in range(N)]


def izt_matches(F, N=10):
    ref = izt_series(F, N)

    def p(r):
        from mikoshilang.math_ops import to_sympy
        e = to_sympy(r)
        vals = [e.subs(sp.Symbol("n"), k).replace(sp.Function("DiscreteDelta"), lambda a: 1 if a == 0 else 0) for k in range(N)]
        return all(sp.simplify(sp.expand(a - b)) == 0 for a, b in zip(vals, ref))
    return Pred(p, f"series of {F} at z = oo")


IZT = [
    ("InverseZTransform[z/(z - 1), z, n]", "1"), ("InverseZTransform[z/(z - a), z, n]", "a^n"),
    ("InverseZTransform[z/(z - 1)^2, z, n]", "n"), ("InverseZTransform[z (z + 1)/(z - 1)^3, z, n]", "n^2"),
    ("InverseZTransform[1, z, n]", "DiscreteDelta[n]"), ("InverseZTransform[1/z^2, z, n]", "DiscreteDelta[n - 2]"),
    ("InverseZTransform[z^2/((z - 1/2) (z - 1/3)), z, n]", "3/2^n - 2/3^n"),
    ("InverseZTransform[z/(z^2 - z - 1), z, n]", izt_matches(z / (z**2 - z - 1))),
    ("InverseZTransform[z/(z^2 + 1), z, n]", izt_matches(z / (z**2 + 1))),
    ("InverseZTransform[1/(z - 2), z, n]", izt_matches(1 / (z - 2))),
    ("InverseZTransform[(3 z^2 + 1)/((z - 1/2)^2 (z + 2)), z, n]", izt_matches((3 * z**2 + 1) / ((z - sp.Rational(1, 2))**2 * (z + 2)))),
    # round trips with ZTransform
    ("InverseZTransform[ZTransform[n^3, n, z], z, n]", "n^3"), ("InverseZTransform[ZTransform[2^n n, n, z], z, n]", "2^n n"),
    ("InverseZTransform[z, z, n]", Raises(ValueError, "causal")), ("InverseZTransform[z^2/(z - 1), z, n]", Raises(ValueError, "causal")),
    ("InverseZTransform[Exp[1/z], z, n]", "InverseZTransform[E^(1/z), z, n]"),
    ("DiscreteDelta[0]", "1"), ("DiscreteDelta[2]", "0"), ("DiscreteDelta[0, 0, 1]", "0"), ("DiscreteDelta[k]", "DiscreteDelta[k]"),
    ("UnitStep[0]", "1"), ("UnitStep[-1/2]", "0"), ("UnitStep[2, 3]", "1"), ("UnitStep[x]", "UnitStep[x]"), ("UnitStep[I]", Raises(ValueError)),
]


def involutions(m):
    a = [1, 1]
    for k in range(2, m + 1):
        a.append(a[-1] + (k - 1) * a[-2])
    return a[m]


def hook(shape):
    prod = 1
    for i, row in enumerate(shape):
        for j in range(row):
            prod *= (row - j - 1) + sum(1 for r in shape[i + 1:] if r > j) + 1
    return math.factorial(sum(shape)) // prod


def standard(t):
    rows = [list(r.args) for r in t.args]
    flat = sorted(x for r in rows for x in r)
    if flat != list(range(1, len(flat) + 1)):
        return False
    ok_rows = all(r[i] < r[i + 1] for r in rows for i in range(len(r) - 1))
    ok_cols = all(rows[i + 1][j] > rows[i][j] for i in range(len(rows) - 1) for j in range(len(rows[i + 1])))
    return ok_rows and ok_cols


def all_standard_distinct(r):
    ts = list(r.args)
    return all(standard(t) for t in ts) and len({str(t) for t in ts}) == len(ts)


YOUNG = [
    ("YoungTableaux[{2, 1}]", "{{{1, 2}, {3}}, {{1, 3}, {2}}}"),
    ("YoungTableaux[{2, 2}]", "{{{1, 2}, {3, 4}}, {{1, 3}, {2, 4}}}"),
    ("YoungTableaux[3]", "{{{1, 2, 3}}, {{1, 2}, {3}}, {{1, 3}, {2}}, {{1}, {2}, {3}}}"),
    *[(f"Length[YoungTableaux[{{{', '.join(map(str, sh))}}}]]", str(hook(sh))) for sh in ([3, 2, 1], [4, 2], [3, 3, 1], [5, 3, 2], [4, 4, 2, 1])],
    *[(f"Length[YoungTableaux[{m}]]", str(involutions(m))) for m in (1, 4, 6, 8)],
    ("YoungTableaux[{4, 2, 1}]", Pred(all_standard_distinct, "standard and distinct")),
    ("YoungTableaux[6]", Pred(all_standard_distinct, "standard and distinct")),
    ("Length[YoungTableaux[{3, 2, 1}]] == HookLengthFormula[{3, 2, 1}]", True),
    ("YoungTableaux[0]", "{{}}"), ("YoungTableaux[{1}]", "{{{1}}}"),
    ("YoungTableaux[{1, 2}]", Raises(ValueError, "weakly decreasing")), ("YoungTableaux[{2, 0}]", Raises(ValueError)),
    ("YoungTableaux[-1]", Raises(ValueError)), ("YoungTableaux[{10, 9, 8}]", Raises(ValueError, "limit")),
]

CASES = ROOT + CONSTANTS + TRACE + ASSUME + LAPLACE + IZT + YOUNG


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_fixes(case):
    # each case gets fresh session definitions ($Assumptions, a = 5, f[x_] := ... must not leak)
    from mikoshilang.kernel import _state, SessionState
    tok = _state.set(SessionState())
    try:
        check(case)
    finally:
        _state.reset(tok)
