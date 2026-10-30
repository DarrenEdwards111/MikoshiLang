"""Differential tests: MikoshiLang results must equal an independent SymPy computation.

Every case pairs Wolfram-syntax input with the SymPy expression that is the
mathematically correct answer.  Equality is checked with ``simplify(a - b) == 0``
(or exact structural equality for lists/rules), so a formatting change cannot
hide a wrong answer.
"""
import pytest
import sympy as sp

from mikoshilang import Session
from mikoshilang.expr import Expr, Symbol
from mikoshilang.math_ops import to_sympy

x, y, z, t, a, b, n = sp.symbols("x y z t a b n")
f = sp.Function("f")
NS = {"x": x, "y": y, "z": z, "t": t, "a": a, "b": b, "n": n, "sp": sp, "f": f}

CASES = [
    # arithmetic & exact numbers
    ("1/3 + 1/6", "sp.Rational(1, 2)"),
    ("2^100", "2**100"),
    ("(-8)^(1/3)", "(-8)**sp.Rational(1, 3)"),
    ("Sqrt[8]", "2*sp.sqrt(2)"),
    ("Sqrt[-9]", "3*sp.I"),
    ("I^2", "-1"),
    ("(1 + I)^2", "2*sp.I"),
    ("Abs[-7/2]", "sp.Rational(7, 2)"),
    ("10!", "sp.factorial(10)"),
    ("Binomial[10, 3]", "120"),
    ("Mod[17, 5]", "2"),
    ("Floor[7/2]", "3"),
    ("Ceiling[7/2]", "4"),
    ("Exp[0]", "1"),
    ("Log[E^3]", "3"),
    ("Log[2, 1024]", "10"),
    ("Sin[Pi/6]", "sp.Rational(1, 2)"),
    ("Cos[Pi/3]", "sp.Rational(1, 2)"),
    ("Tan[Pi/4]", "1"),
    ("ArcTan[1]", "sp.pi/4"),
    ("ArcSin[1]", "sp.pi/2"),
    ("ArcTan[1, 1]", "sp.pi/4"),
    ("Sinh[0]", "0"),
    ("Gamma[5]", "24"),
    ("Max[3, 9, 2]", "9"),
    ("Min[3, 9, 2]", "2"),
    # algebra
    ("x + x", "2*x"),
    ("x*x^2", "x**3"),
    ("(x^2)^3", "x**6"),
    ("Expand[(x + 1)^5]", "sp.expand((x + 1)**5)"),
    ("Expand[(x + y)(x - y)]", "x**2 - y**2"),
    ("Factor[x^3 - 1]", "sp.factor(x**3 - 1)"),
    ("Factor[x^2 + 5 x + 6]", "(x + 2)*(x + 3)"),
    ("Simplify[(x^2 - 1)/(x - 1)]", "x + 1"),
    ("Simplify[Sin[x]^2 + Cos[x]^2]", "1"),
    ("Together[1/x + 1/y]", "(x + y)/(x*y)"),
    ("Apart[1/(x^2 - 1)]", "sp.apart(1/(x**2 - 1))"),
    ("Expand[(a + b)^3]", "sp.expand((a + b)**3)"),
    ("x/y/z", "x/(y*z)"),
    ("2 x y - x y", "x*y"),
    ("(x + 1)^2 - (x^2 + 2 x + 1)", "sp.expand((x + 1)**2 - (x**2 + 2*x + 1))"),
    # calculus: derivatives
    ("D[x^5, x]", "5*x**4"),
    ("D[x^5, {x, 3}]", "60*x**2"),
    ("D[Sin[x], x]", "sp.cos(x)"),
    ("D[Sin[x] Cos[x], x]", "sp.cos(x)**2 - sp.sin(x)**2"),
    ("D[Exp[2 x], x]", "2*sp.exp(2*x)"),
    ("D[Log[x], x]", "1/x"),
    ("D[x^2 y^3, x, y]", "6*x*y**2"),
    ("D[ArcTan[x], x]", "1/(1 + x**2)"),
    ("D[Sqrt[x], x]", "1/(2*sp.sqrt(x))"),
    ("D[x^x, x]", "sp.diff(x**x, x)"),
    ("D[f[x]^2, x]", "2*f(x)*sp.Derivative(f(x), x)"),
    ("g[u_] := u^4; g'[x]", "4*x**3"),
    ("g2[u_] := Sin[u]; g2''[x]", "-sp.sin(x)"),
    # integrals
    ("Integrate[x^2, x]", "x**3/3"),
    ("Integrate[x^2, {x, 0, 3}]", "9"),
    ("Integrate[Sin[x], {x, 0, Pi}]", "2"),
    ("Integrate[Exp[-x], {x, 0, Infinity}]", "1"),
    ("Integrate[1/x, x]", "sp.log(x)"),
    ("Integrate[x Exp[x], x]", "sp.integrate(x*sp.exp(x), x)"),
    ("Integrate[1/(1 + x^2), x]", "sp.atan(x)"),
    # limits
    ("Limit[Sin[x]/x, x -> 0]", "1"),
    ("Limit[(1 + 1/x)^x, x -> Infinity]", "sp.E"),
    ("Limit[(x^2 - 1)/(x - 1), x -> 1]", "2"),
    ("Limit[1/x^2, x -> 0]", "sp.oo"),
    ("Limit[x Log[x], x -> 0]", "0"),
    # sums
    ("Sum[i, {i, 1, 100}]", "5050"),
    ("Sum[i^2, {i, 1, 10}]", "385"),
    ("Sum[1/2^i, {i, 0, Infinity}]", "2"),
    ("Product[i, {i, 1, 6}]", "720"),
    # series: Normal[Series[f, {x, 0, n}]] includes the x^n term
    ("Normal[Series[Exp[x], {x, 0, 5}]]", "sp.series(sp.exp(x), x, 0, 6).removeO()"),
    ("Normal[Series[Sin[x], {x, 0, 7}]]", "sp.series(sp.sin(x), x, 0, 8).removeO()"),
    ("Normal[Series[Cos[x], {x, 0, 4}]]", "1 - x**2/2 + x**4/24"),
    ("Normal[Series[1/(1 - x), {x, 0, 4}]]", "1 + x + x**2 + x**3 + x**4"),
    ("Normal[Series[Log[1 + x], {x, 0, 3}]]", "x - x**2/2 + x**3/3"),
    ("Normal[Series[Exp[x], {x, 1, 2}]]", "sp.series(sp.exp(x), x, 1, 3).removeO()"),
    ("Normal[Series[Tan[x], {x, 0, 5}]]", "x + x**3/3 + 2*x**5/15"),
    ("Normal[Series[Sqrt[1 + x], {x, 0, 2}]]", "1 + x/2 - x**2/8"),
    # numerics
    ("N[Pi, 30]", "sp.pi.evalf(30)"),
    ("N[Sqrt[2]]", "sp.sqrt(2).evalf(15)"),
    ("N[E, 20]", "sp.E.evalf(20)"),
    # definitions, recursion, memoisation
    ("h[0] = 1; h[k_Integer] := k h[k - 1]; h[10]", "sp.factorial(10)"),
    ("fib[0] = 0; fib[1] = 1; fib[k_] := fib[k] = fib[k - 1] + fib[k - 2]; fib[60]", "sp.fibonacci(60)"),
    ("p[u_] := u^3 - 2 u + 1; p[7]", "7**3 - 2*7 + 1"),
    ("q[u_] := u^2 /; u > 0; q[u_] := -u^2; q[-3] + q[3]", "0"),
    ("Module[{s = 2}, s^10]", "1024"),
    ("With[{c = 3}, c^c]", "27"),
    ("Function[u, u^3 + 1][2]", "9"),
    ("Function[{u, v}, u^v][2, 10]", "1024"),
    ("(#1^2 + #2^2)&[3, 4]", "25"),
    ("Fold[#1 x + #2 &, 0, {1, 2, 3}]", "x**2 + 2*x + 3"),
    ("Nest[(# + 2/#)/2 &, 1, 3]", "sp.Rational(577, 408)"),
    ("Total[Table[k^3, {k, 1, 10}]]", "3025"),
    ("Total[Select[Range[20], PrimeQ]]", "77"),
    ("Apply[Times, Range[8]]", "40320"),
    ("Length[Cases[Range[30], _?EvenQ]]", "15"),
    ("Count[Range[100], u_ /; Mod[u, 7] == 0]", "14"),
    ("Dot[{1, 2, 3}, {4, 5, 6}]", "32"),
    ("Det[{{2, 1}, {1, 3}}]", "5"),
    ("Inner[Times, {a, b}, {x, y}, Plus]", "a*x + b*y"),
    ("s = 0; Do[s += k^2, {k, 1, 10}]; s", "385"),
    ("k = 1; acc = 1; While[k <= 5, acc = acc k; k++]; acc", "120"),
    ("For[k = 1; acc = 0, k <= 100, k++, acc += k]; acc", "5050"),
    ("Which[1 == 2, 0, True, Expand[(x + 1)^2]]", "sp.expand((x + 1)**2)"),
    ("Block[{x = 2}, x^5]", "32"),
    ("Hold[x^2] /. Hold[e_] :> D[e, x]", "2*x"),
    ("ReleaseHold[Hold[2 + 3]]", "5"),
    ("x^2 + 3 x /. x -> 2", "10"),
    ("x^2 /. x -> y + 1", "(y + 1)**2"),
    ("FixedPoint[Floor[#/2] &, 1000]", "0"),
]


@pytest.fixture
def session():
    return Session()


@pytest.mark.parametrize("source,expected", CASES, ids=[c[0] for c in CASES])
def test_against_sympy(session, source, expected):
    result = session.evaluate(source)
    want = eval(expected, NS)  # expected values are trusted literals in this file
    got = to_sympy(result)
    if got == want:
        return
    if isinstance(want, sp.Float) or isinstance(got, sp.Float):
        prec = max(getattr(want, "_prec", 53), 53)
        assert abs(sp.N(got - want, 40)) < sp.Float(10) ** (-(prec // 4)), (source, result)
        return
    assert sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0, (source, result, want)


def test_case_count():
    assert len(CASES) >= 100


# Structured results (rules, lists) are compared exactly.

def R(lhs, rhs):
    return Expr("Rule", lhs, rhs)


def L(*xs):
    return Expr("List", *xs)


X, Y, Yf = Symbol("x"), Symbol("y"), lambda v: Expr("y", v)


@pytest.mark.parametrize("source,check", [
    ("Solve[{x + y == 3, x - y == 1}, {x, y}]", lambda r: r == L(L(R(X, 2), R(Y, 1)))),
    ("Solve[2 x + 1 == 7, x]", lambda r: r == L(L(R(X, 3)))),
    ("Solve[x^2 == 4, x]", lambda r: r == L(L(R(X, -2)), L(R(X, 2)))),
    ("Solve[x == x + 1, x]", lambda r: r == L()),
    ("Solve[{x^2 + y^2 == 5, x - y == 1}, {x, y}]",
     lambda r: {tuple(to_sympy(rule.args[1]) for rule in s.args) for s in r.args} == {(2, 1), (-1, -2)}),
    ("Solve[{x + y + z == 6, x - y == 0, 2 z == 2}, {x, y, z}]",
     lambda r: [[to_sympy(q.args[1]) for q in s.args] for s in r.args] == [[sp.Rational(5, 2), sp.Rational(5, 2), 1]]),
])
def test_solve_rules(session, source, check):
    assert check(session.evaluate(source))


@pytest.mark.parametrize("source,ode,ics", [
    ("DSolve[y'[x] == y[x], y[x], x]", lambda Y: sp.Eq(Y.diff(x), Y), None),
    ("DSolve[y''[x] + y[x] == 0, y[x], x]", lambda Y: sp.Eq(Y.diff(x, 2) + Y, 0), None),
    ("DSolve[{y'[x] == 2 y[x], y[0] == 3}, y[x], x]", lambda Y: sp.Eq(Y.diff(x), 2 * Y), {"y0": 3}),
    ("DSolve[y'[x] == x, y[x], x]", lambda Y: sp.Eq(Y.diff(x), x), None),
    ("DSolve[D[y[x], x, x] == -4 y[x], y[x], x]", lambda Y: sp.Eq(Y.diff(x, 2), -4 * Y), None),
])
def test_dsolve_solutions_satisfy_ode(session, source, ode, ics):
    result = session.evaluate(source)
    assert isinstance(result, Expr) and result.head == "List" and result.args
    rule = result.args[0].args[0]
    assert rule.head == "Rule" and rule.args[0] == Yf(X)
    sol = to_sympy(rule.args[1])
    yf = sp.Function("y")
    eq = ode(yf(x))
    residual = (eq.lhs - eq.rhs).subs(yf(x), sol).doit()
    assert sp.simplify(residual) == 0
    if ics:
        assert sp.simplify(sol.subs(x, 0) - ics["y0"]) == 0
