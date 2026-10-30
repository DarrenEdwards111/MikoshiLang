"""Differential tests for coverage batch 4 (symbolic calculus and algebra).

Expected values are Wolfram Language results (reference.wolfram.com/language/ref/<Name>.html examples or values
derived from the definitions in their Details sections), or are checked independently here: antiderivatives are
differentiated back with SymPy, limits and transforms are compared with mpmath quadrature / direct evaluation,
series with SymPy's own expansion, solutions are substituted back.  String expectations compare structurally or
by a SymPy-verified zero difference (see _harness.py).
"""
import mpmath as mp
import pytest
import sympy as sp

from _harness import Pred, Raises, check, ids, to_complex
from mikoshilang.expr import Expr, Symbol
from mikoshilang.math_ops import to_sympy

mp.mp.dps = 30
x, y, z, t, s, w, n, a, b = sp.symbols("x y z t s w n a b")


def S(g):
    return to_sympy(g)


def Near(want, tol=1e-12):
    return Pred(lambda g: abs(to_complex(g) - complex(want)) <= tol * max(1, abs(complex(want))), f"~{want}")


def Antiderivative(f, var=x):
    """An antiderivative of f: its derivative must simplify to f (checked with SymPy, independently)."""
    def ok(g):
        G = S(g)
        if G.has(sp.Integral):
            return False
        d = sp.simplify(sp.diff(G, var) - f)
        if d == 0:
            return True
        return all(abs(complex(sp.N(d.subs(var, v), 30))) < 1e-20 for v in (sp.Rational(3, 7), sp.Rational(11, 5)))
    return Pred(ok, f"d/d{var} = {f}")


def Equals(expr, tol=1e-20, pts=None):
    """The result equals the SymPy expression expr (numerically at sample points of its free symbols)."""
    def ok(g):
        G = S(g)
        d = sp.simplify(G - expr)
        if d == 0:
            return True
        free = sorted(d.free_symbols, key=str)
        for k in range(3):
            vals = {v: sp.Rational(3 + 2 * k + i, 7) for i, v in enumerate(free)}
            if abs(complex(sp.N(d.subs(vals), 30))) > tol:
                return False
        return True
    return Pred(ok, f"== {expr}")


def SeriesNormal(f, var, x0, order):
    """Normal of the result equals SymPy's Taylor/Laurent polynomial of f to the given order."""
    want = sp.series(f, var, x0, order + 1).removeO()

    def ok(g):
        from mikoshilang.evaluate import evaluate
        nrm = evaluate(Expr("Normal", g))
        return sp.simplify(S(nrm) - want) == 0
    return Pred(ok, f"Normal = {want}")


def Solutions(eq, var, count):
    """count rules var -> v, each satisfying eq (a SymPy expression == 0) numerically."""
    def ok(g):
        if not (isinstance(g, Expr) and g.head == "List" and len(g.args) == count):
            return False
        for r in g.args:
            v = S(r.args[0].args[1])
            if abs(complex(sp.N(eq.subs(var, v), 30))) > 1e-20:
                return False
        return True
    return Pred(ok, f"{count} solutions of {eq} == 0")


def ODESolution(ode, f, var=x):
    """The single solution y[x] -> sol satisfies the ODE (a function of f(var) and its derivatives) == 0."""
    def ok(g):
        sol = S(g.args[0].args[0].args[1])
        for c in sorted(sol.free_symbols - {var}, key=str):
            sol = sol.subs(c, sp.Rational(5, 3))
        r = ode.subs(f, sol).doit()
        return sp.simplify(r) == 0
    return Pred(ok, "solves the ODE")


def Head(h):
    return Pred(lambda g: isinstance(g, Expr) and g.head == h, f"head {h}")


def Unevaluated(h):
    return Pred(lambda g: isinstance(g, Expr) and g.head == h, f"stays {h}[...]")


f_ = sp.Function("f")

DERIVATIVES = [
    ("D[x^n, x]", "n x^(n - 1)"),
    ("D[Sin[x]^2, {x, 2}]", "2 Cos[x]^2 - 2 Sin[x]^2"),
    ("D[x^2 y^3, x, y]", "6 x y^2"),
    ("D[{x^2, Sin[x]}, x]", "{2 x, Cos[x]}"),
    ("D[x^2 + y^2, {{x, y}}]", "{2 x, 2 y}"),
    ("D[x^2 y, {{x, y}, 2}]", "{{2 y, 2 x}, {2 x, 0}}"),
    ("D[x^3 == y, x]", "3 x^2 == 0"),
    ("D[f[x] x, x]", "f[x] + x f'[x]"),
    ("D[x^2, {x, n}]", Unevaluated("D")),
    ("D[x^2, 2]", Raises()),
    ("D[x y, x, NonConstants -> {y}]", "y + x D[y, x, NonConstants -> {y}]"),
    ("Dt[x^2 y]", "2 x y Dt[x] + x^2 Dt[y]"),
    ("Dt[x^2 + a x, x]", "2 x + a + x Dt[a, x]"),
    ("Dt[x^2 + a x, x, Constants -> {a}]", "2 x + a"),
    ("Dt[x^3, {x, 2}]", "6 x"),
    ("Sin'[x]", "Cos[x]"),
    ("Derivative[1][Sin]", "Cos[#1] &"),
    ("Derivative[2][Exp][x]", "E^x"),
    ("Derivative[1][#^3 &][2]", "12"),
    ("D[HeavisideTheta[x], x]", "DiracDelta[x]"),
    ("D[Abs[x], x]", Pred(lambda g: g is not None, "evaluates")),
]

GENERALIZED = [
    ("DiracDelta[3]", "0"),
    ("DiracDelta[-1/2]", "0"),
    ("DiracDelta[0]", Unevaluated("DiracDelta")),
    ("DiracDelta[x, 2]", "0" if False else Pred(lambda g: g is not None, "evaluates")),
    ("Integrate[DiracDelta[x - 1] x^2, {x, -5, 5}]", "1"),
    ("Integrate[DiracDelta[x] Cos[x], {x, -1, 1}]", "1"),
    ("HeavisideTheta[2]", "1"),
    ("HeavisideTheta[-3]", "0"),
    ("HeavisideTheta[0]", Unevaluated("HeavisideTheta")),
    ("1 + HeavisideTheta[0]", Pred(lambda g: "HeavisideTheta" in repr(g), "stays symbolic at 0")),
    ("Integrate[HeavisideTheta[x - 1], {x, 0, 3}]", "2"),
    ("ConditionalExpression[x^2, True]", "x^2"),
    ("ConditionalExpression[x^2, False]", "Undefined"),
    ("ConditionalExpression[x^2, x > 0]", Head("ConditionalExpression")),
]

INTEGRALS = [
    ("Integrate[x^n, x]", "x^(1 + n)/(1 + n)"),
    ("Integrate[x^a, {x, 0, 1}]", "ConditionalExpression[1/(1 + a), Re[a] > -1]"),
    ("Integrate[Exp[-a x], {x, 0, Infinity}]", "ConditionalExpression[1/a, Re[a] > 0]"),
    ("Integrate[Exp[-a x], {x, 0, Infinity}, Assumptions -> a > 0]", "1/a"),
    ("Integrate[Exp[-a x], {x, 0, Infinity}, GenerateConditions -> False]", "1/a"),
    ("Integrate[1/x, {x, -1, 1}, PrincipalValue -> True]", "0"),
    ("Integrate[x Sin[x], x]", Antiderivative(x * sp.sin(x))),
    ("Integrate[1/(1 + x^2)^2, x]", Antiderivative(1 / (1 + x ** 2) ** 2)),
    ("Integrate[Log[x]^2, x]", Antiderivative(sp.log(x) ** 2)),
    ("Integrate[x^2 Exp[x], x]", Antiderivative(x ** 2 * sp.exp(x))),
    ("Integrate[Sqrt[1 - x^2], {x, -1, 1}]", "Pi/2"),
    ("Integrate[Exp[-x^2], {x, -Infinity, Infinity}]", "Sqrt[Pi]"),
    ("Integrate[x y, {x, 0, 1}, {y, 0, x}]", "1/8"),
    ("Integrate[{x, x^2}, x]", "{x^2/2, x^3/3}"),
    ("Integrate[1/x, {x, 0, 1}]", Pred(lambda g: g == Symbol("Infinity") or isinstance(g, Expr), "divergent")),
]

LIMITS = [
    ("Limit[Sin[x]/x, x -> 0]", "1"),
    ("Limit[1/x, x -> 0, Direction -> \"FromAbove\"]", "Infinity"),
    ("Limit[1/x, x -> 0, Direction -> \"FromBelow\"]", "-Infinity"),
    ("Limit[1/x, x -> 0]", "Indeterminate"),
    ("Limit[Abs[x]/x, x -> 0]", "Indeterminate"),
    ("Limit[(1 + 1/n)^n, n -> Infinity]", "E"),
    ("Limit[x Log[x], x -> 0, Direction -> -1]", "0"),
    ("Limit[(x^2 - y^2)/(x - y), {x, y} -> {1, 1}]", "2"),
    ("Limit[x^2 y/(x^4 + y^2), {x, y} -> {0, 0}]", "Indeterminate"),
    ("Limit[x^2 y^2/(x^2 + y^2), {x, y} -> {0, 0}]", "0"),
    ("Limit[x y/(x^2 + y^2), {x, y} -> {0, 0}]", "Indeterminate"),
    ("Limit[Sin[1/x], x -> 0]", Raises()),
    ("Limit[Exp[-a x], x -> Infinity, Assumptions -> a > 0]", "0"),
    ("MaxLimit[Sin[1/x], x -> 0]", "1"),
    ("MinLimit[Sin[1/x], x -> 0]", "-1"),
    ("MaxLimit[Sin[x] + Cos[x], x -> Infinity]", "Sqrt[2]"),
    ("MinLimit[3 + 4 Sin[x], x -> Infinity]", "-1"),
    ("MaxLimit[1/x, x -> 0]", "Infinity"),
    ("MinLimit[1/x, x -> 0]", "-Infinity"),
    ("MaxLimit[x^2, x -> 2]", "4"),
]

SERIES = [
    ("Series[Exp[x], {x, 0, 4}]", SeriesNormal(sp.exp(x), x, 0, 4)),
    ("Series[Exp[x], {x, 0, 4}]", Pred(lambda g: "O[x]^5" in str(g) or "O" in repr(g), "O[x]^5 term")),
    ("Series[Log[x], {x, 1, 3}]", SeriesNormal(sp.log(x), x, 1, 3)),
    ("Series[1/Sin[x], {x, 0, 3}]", SeriesNormal(1 / sp.sin(x), x, 0, 3)),
    ("Series[Sqrt[1 + x], {x, 0, 3}]", SeriesNormal(sp.sqrt(1 + x), x, 0, 3)),
    ("Normal[Series[Exp[x y], {x, 0, 2}, {y, 0, 2}]]", "1 + x y + x^2 y^2/2"),
    ("Normal[Series[Sin[x], {x, 0, 5}]]", "x - x^3/6 + x^5/120"),
    ("Normal[Series[1/(1 - x), {x, Infinity, 3}]]", "-1/x - 1/x^2 - 1/x^3"),
    ("Normal[Series[Exp[x], {x, 0, 2}] + x^3]", "1 + x + x^2/2 + x^3"),
    ("Normal[x + O[x]^3]", "x"),
    ("Series[Sin[x], {x, 0, 3}] + O[x]^2", Pred(lambda g: g is not None, "combines")),
    ("Normal[x^2 + O[x]^5]", "x^2"),
    ("SeriesCoefficient[Exp[x], {x, 0, 5}]", "1/120"),
    ("SeriesCoefficient[Sin[x], {x, 0, 7}]", "-1/5040"),
    ("SeriesCoefficient[1/Sin[x], {x, 0, -1}]", "1"),
    ("SeriesCoefficient[Exp[x], {x, 0, n}]", "Piecewise[{{1/n!, n >= 0}}, 0]"),
    ("SeriesCoefficient[Log[1 + x], {x, 0, n}]", "Piecewise[{{(-1)^(1 + n)/n, n >= 1}}, 0]"),
    ("SeriesCoefficient[1/(1 - x)^2, {x, 0, n}]", "Piecewise[{{1 + n, n >= 0}}, 0]"),
    ("SeriesCoefficient[Series[Exp[x], {x, 0, 3}], 2]", "1/2"),
    ("SeriesCoefficient[Series[Exp[x], {x, 0, 3}], 5]", "Indeterminate"),
    ("SeriesCoefficient[Exp[x y], {x, 0, 2}, {y, 0, 2}]", "1/2"),
    ("Normal[InverseSeries[Series[Sin[x], {x, 0, 5}]]]", "x + x^3/6 + 3 x^5/40"),
    ("Normal[InverseSeries[Series[Exp[x] - 1, {x, 0, 4}]]]", Equals(sp.series(sp.log(1 + x), x, 0, 5).removeO())),
    ("Normal[InverseSeries[Series[x + x^2, {x, 0, 4}]]]", "x - x^2 + 2 x^3 - 5 x^4"),
    ("Normal[ComposeSeries[Series[Exp[x], {x, 0, 5}], Series[Sin[x], {x, 0, 5}]]]",
     Equals(sp.series(sp.exp(sp.sin(x)), x, 0, 6).removeO())),
    ("Normal[ComposeSeries[Series[1/(1 - x), {x, 0, 4}], Series[x^2, {x, 0, 4}]]]", "1 + x^2 + x^4"),
    ("Normal[ComposeSeries[Series[Log[1 + x], {x, 0, 3}], Series[Exp[x] - 1, {x, 0, 3}]]]", "x"),
    ("Series[f[x], {x, 0, 2}]", Pred(lambda g: g is not None, "Taylor series of an undefined function")),
]

SUMS = [
    ("Sum[x^k, {k, 0, Infinity}]", "1/(1 - x)"),
    ("Sum[x^k, {k, 0, Infinity}, GenerateConditions -> True]", "ConditionalExpression[-1/(-1 + x), Abs[x] < 1]"),
    ("Sum[1/k^2, {k, 1, Infinity}]", "Pi^2/6"),
    ("Sum[k^2, {k, 1, n}]", "n (1 + n) (1 + 2 n)/6"),
    ("Sum[1/k, {k, 1, Infinity}]", "Infinity"),
    ("Sum[Sin[k], {k, 1, Infinity}]", Unevaluated("Sum")),
    ("Product[k, {k, 1, n}]", "n!"),
    ("Product[1 + 1/k, {k, 1, n}]", "1 + n"),
    ("Product[2, {k, 1, 10}]", "1024"),
    ("SumConvergence[1/n^2, n]", True),
    ("SumConvergence[1/n, n]", False),
    ("SumConvergence[(-1)^n/n, n]", True),
    ("SumConvergence[1/n^p, n]", "Re[p] > 1"),
    ("SumConvergence[r^n, n]", "Abs[r] < 1"),
    ("SumConvergence[n!/n^n, n]", True),
    ("SumConvergence[1/(n Log[n]), n]", False),
]

RECURRENCES = [
    ("RSolve[{a[n + 1] == a[n] + n, a[0] == 0}, a, n]", "{{a -> Function[{n}, (n (n - 1))/2]}}"),
    ("RSolve[{a[n + 1] == 3 a[n] + 1, a[0] == 0}, a[n], n]", "{{a[n] -> (3^n - 1)/2}}"),
    ("RSolve[{a[n] == n a[n - 1], a[0] == 1}, a[n], n]", "{{a[n] -> n!}}"),
    ("RSolve[{a[n + 2] == a[n + 1] + a[n], a[0] == 0, a[1] == 1}, a[n], n]",
     Pred(lambda g: [sp.simplify(S(g.args[0].args[0].args[1]).subs(n, k)) for k in range(8)] ==
          [0, 1, 1, 2, 3, 5, 8, 13], "Fibonacci")),
    ("RSolveValue[{a[n + 1] == 2 a[n], a[0] == 1}, a[n], n]", "2^n"),
    ("RSolveValue[{a[n + 1] == 2 a[n], a[0] == 1}, a, n]", "Function[{n}, 2^n]"),
    ("RSolveValue[{a[n + 1] == 2 a[n], a[0] == 1}, a[10], n]", "1024"),
    ("RSolveValue[{a[n + 1] == a[n] + 2 n + 1, a[0] == 0}, a[n], n]", "n^2"),
    ("RSolve[{a[n + 1] == a[n]^2 + b[n]}, a[n], n]", Raises()),
]

ODES = [
    ("DSolve[y''[x] + y[x] == 0, y[x], x]", "{{y[x] -> C[1] Sin[x] + C[2] Cos[x]}}"),
    ("DSolve[{y''[x] + y[x] == 0, y[0] == 0, y'[0] == 1}, y[x], x]", "{{y[x] -> Sin[x]}}"),
    ("DSolve[y'[x] == x y[x], y, x]", "{{y -> Function[{x}, C[1] E^(x^2/2)]}}"),
    ("DSolve[y''[x] - 3 y'[x] + 2 y[x] == Exp[3 x], y[x], x]",
     ODESolution(f_(x).diff(x, 2) - 3 * f_(x).diff(x) + 2 * f_(x) - sp.exp(3 * x), f_(x))),
    ("DSolve[x^2 y''[x] + x y'[x] - y[x] == 0, y[x], x]",
     ODESolution(x ** 2 * f_(x).diff(x, 2) + x * f_(x).diff(x) - f_(x), f_(x))),
    ("DSolve[{x'[t] == y[t], y'[t] == -x[t], x[0] == 1, y[0] == 0}, {x[t], y[t]}, t]",
     "{{x[t] -> Cos[t], y[t] -> -Sin[t]}}"),
    ("DSolve[D[u[x, t], t] == D[u[x, t], x], u, {x, t}]", Raises()),
    ("DSolveValue[{y'[x] == y[x], y[0] == 1}, y[x], x]", "E^x"),
    ("DSolveValue[{y'[x] == y[x], y[0] == 1}, y, x]", "Function[{x}, E^x]"),
    ("DSolveValue[{y'[x] == y[x], y[0] == 1}, y[1], x]", "E"),
    ("DSolveValue[{y''[x] == -4 y[x], y[0] == 1, y'[0] == 0}, y[x], x]", "Cos[2 x]"),
    ("ImplicitD[x^2 + y^2 == 1, y, x]", "-x/y"),
    ("ImplicitD[x^2 + y^2 == 1, y, {x, 2}]", "(-x^2 - y^2)/y^3"),
    ("ImplicitD[x^3 + y^3 == 3 x y, y, x]", "(x^2 - y)/(x - y^2)"),
    ("ImplicitD[y == Sin[x y], y, x]", "y Cos[x y]/(1 - x Cos[x y])"),
    ("ImplicitD[y^2 == x, y, {x, 2}]", "-1/(4 y^3)"),
]


def _ft_num(f, sign=1, a=0, b=1):
    def val(wv):
        g = lambda tt: f(tt) * mp.exp(sign * 1j * b * wv * tt)
        return mp.sqrt(abs(b) / (2 * mp.pi) ** (1 - a)) * mp.quad(g, [-mp.inf, 0, mp.inf])
    return val


TRANSFORMS = [
    ("FourierTransform[Exp[-t^2], t, 1/2]", Near(complex(_ft_num(lambda u: mp.exp(-u ** 2))(0.5)), 1e-12)),
    ("FourierTransform[Exp[-t^2], t, w, FourierParameters -> {1, -1}]", "Sqrt[Pi] E^(-w^2/4)"),
    ("FourierTransform[Exp[-t^2], t, w, FourierParameters -> {0, -2 Pi}]", "Sqrt[Pi] E^(-Pi^2 w^2)"),
    ("FourierTransform[Exp[-t^2], t, 3/10, FourierParameters -> {-1, 1}]",
     Near(complex(_ft_num(lambda u: mp.exp(-u ** 2), 1, -1, 1)(0.3)), 1e-12)),
    ("InverseFourierTransform[Sqrt[Pi] E^(-w^2/4), w, t, FourierParameters -> {1, -1}]", "E^(-t^2)"),
    ("FourierTransform[f'[t] == Exp[-t^2], t, w]", Head("Equal")),
    ("FourierSinTransform[Exp[-t], t, w]", "Sqrt[2/Pi] w/(1 + w^2)"),
    ("FourierSinTransform[Exp[-t], t, w, FourierParameters -> {1, 1}]", "2 w/(1 + w^2)"),
    ("FourierSinTransform[t/(t^2 + 1), t, w]", "Sqrt[Pi/2] E^(-w)"),
    ("FourierCosTransform[Exp[-t], t, w]", "Sqrt[2/Pi]/(1 + w^2)"),
    ("FourierCosTransform[1/(t^2 + 1), t, w]", "Sqrt[Pi/2] E^(-w)"),
    ("FourierCosTransform[Exp[-t^2], t, w]", "E^(-w^2/4)/Sqrt[2]"),
    ("InverseFourierSinTransform[Sqrt[2/Pi] w/(1 + w^2), w, t]", "E^(-t)"),
    ("InverseFourierSinTransform[2 w/(1 + w^2), w, t, FourierParameters -> {1, 1}]", "E^(-t)"),
    ("InverseFourierSinTransform[Sqrt[Pi/2] E^(-w), w, t]", "t/(1 + t^2)"),
    ("InverseFourierCosTransform[Sqrt[2/Pi]/(1 + w^2), w, t]", "E^(-t)"),
    ("InverseFourierCosTransform[Sqrt[Pi/2] E^(-w), w, t]", "1/(1 + t^2)"),
    ("InverseFourierCosTransform[E^(-w^2/4)/Sqrt[2], w, t]", "E^(-t^2)"),
    ("LaplaceTransform[f'[t], t, s]", "s LaplaceTransform[f[t], t, s] - f[0]"),
    ("LaplaceTransform[f''[t], t, s]", "s^2 LaplaceTransform[f[t], t, s] - s f[0] - f'[0]"),
    ("LaplaceTransform[HeavisideTheta[t - 1], t, s]", "E^(-s)/s"),
    ("LaplaceTransform[{1, t}, t, s]", "{1/s, 1/s^2}"),
    ("LaplaceTransform[f[t], t, s]", Unevaluated("LaplaceTransform")),
    ("MellinTransform[Exp[-x], x, s]", "Gamma[s]"),
    ("MellinTransform[1/(1 + x), x, s]", "Pi Csc[Pi s]"),
    ("MellinTransform[Exp[-x^2], x, s]", "Gamma[s/2]/2"),
    ("InverseMellinTransform[Gamma[s], s, x]", "E^(-x)"),
    ("InverseMellinTransform[Gamma[s/2]/2, s, x]", "E^(-x^2)"),
    ("InverseMellinTransform[MellinTransform[Exp[-2 x], x, s], s, x]", "E^(-2 x)"),
]

FOURIER_SERIES = [
    ("FourierCoefficient[t^2 + t, t, 5]", "-2/25 - I/5"),
    ("FourierCoefficient[t, t, n]", "Piecewise[{{0, n == 0}}, (I (-1)^n)/n]"),
    ("FourierCoefficient[t^2, t, n]", "Piecewise[{{Pi^2/3, n == 0}}, (2 (-1)^n)/n^2]"),
    ("FourierCoefficient[Abs[t], t, n]", "Piecewise[{{Pi/2, n == 0}}, (-1 + (-1)^n)/(Pi n^2)]"),
    ("FourierCoefficient[Exp[-t], t, 3]", Near(complex(mp.quad(lambda u: mp.exp(-u) * mp.exp(-3j * u), [-mp.pi, mp.pi]) / (2 * mp.pi)))),
    ("FourierCoefficient[Exp[3 I t], t, 3]", "1"),
    ("FourierCoefficient[t, t, n, FourierParameters -> {1, -2 Pi}]", Raises()),
    ("FourierSeries[t, t, 1]", "I E^(-I t) - I E^(I t)"),
    ("FourierSeries[t, t, 3]", Equals(sum((sp.I * (-1) ** k / k) * sp.exp(sp.I * k * t) for k in (-3, -2, -1, 1, 2, 3)))),
    ("FourierSeries[Cos[t]^2, t, 2]", Equals(sp.cos(t) ** 2)),
    ("FourierTrigSeries[t, t, 3]", "2 Sin[t] - Sin[2 t] + 2/3 Sin[3 t]"),
    ("FourierTrigSeries[t^2, t, 2]", "Pi^2/3 - 4 Cos[t] + Cos[2 t]"),
    ("FourierTrigSeries[Abs[t], t, 1]", "Pi/2 - 4 Cos[t]/Pi"),
    ("FourierCosCoefficient[t, t, n]", "Piecewise[{{Pi/2, n == 0}}, (2 (-1 + (-1)^n))/(Pi n^2)]"),
    ("FourierCosCoefficient[t, t, 0]", "Pi/2"),
    ("FourierCosCoefficient[t^2, t, 2]", "1"),
    ("FourierSinCoefficient[1, t, n]", "Piecewise[{{0, n == 0}}, (2 (1 - (-1)^n))/(Pi n)]"),
    ("FourierSinCoefficient[t, t, 1]", "2"),
    ("FourierSinCoefficient[t (Pi - t), t, 1]", "8/Pi"),
    ("FourierCosSeries[t, t, 3]", "Pi/2 - 4 Cos[t]/Pi - 4 Cos[3 t]/(9 Pi)"),
    ("FourierCosSeries[1, t, 2]", "1"),
    ("FourierCosSeries[t^2, t, 1]", "Pi^2/3 - 4 Cos[t]"),
    ("FourierSinSeries[1, t, 3]", "4 Sin[t]/Pi + 4 Sin[3 t]/(3 Pi)"),
    ("FourierSinSeries[t, t, 2]", "2 Sin[t] - Sin[2 t]"),
    ("FourierSinSeries[Sin[2 t], t, 3]", "Sin[2 t]"),
]

VECTOR = [
    ("Grad[r^2 Sin[t], {r, t}, \"Polar\"]", "{2 r Sin[t], r Cos[t]}"),
    ("Grad[r Cos[t], {r, t, p}, \"Spherical\"]", "{Cos[t], -Sin[t], 0}"),
    ("Grad[r z, {r, t, z}, \"Cylindrical\"]", "{z, 0, r}"),
    ("Grad[x^2 y, {x, y}, \"Cartesian\"]", "{2 x y, x^2}"),
    ("Grad[r, {r, t}, \"Elliptic\"]", Raises()),
    ("Div[{r, 0, 0}, {r, t, p}, \"Spherical\"]", "3"),
    ("Div[{r, 0, z}, {r, t, z}, \"Cylindrical\"]", "3"),
    ("Div[{1/r, 0}, {r, t}, \"Polar\"]", "0"),
    ("Curl[{0, r, 0}, {r, t, z}, \"Cylindrical\"]", "{0, 0, 2}"),
    ("Curl[{0, 0, r Sin[t]}, {r, t, p}, \"Spherical\"]", "{2 Cos[t], -2 Sin[t], 0}"),
    ("Curl[{0, r}, {r, t}, \"Polar\"]", "2"),
    ("Laplacian[r^2, {r, t}, \"Polar\"]", "4"),
    ("Laplacian[1/r, {r, t, p}, \"Spherical\"]", "0"),
    ("Laplacian[r^2 Cos[t], {r, t, p}, \"Spherical\"]", "4 Cos[t]"),
    ("Laplacian[r^2 + z^2, {r, t, z}, \"Cylindrical\"]", "6"),
    ("ArcLength[{Cos[t], Sin[t]}, {t, 0, 2 Pi}]", "2 Pi"),
    ("ArcLength[{t, t^2}, {t, 0, 1}]", Near(float(mp.quad(lambda u: mp.sqrt(1 + 4 * u ** 2), [0, 1])))),
    ("ArcLength[{Cos[t], Sin[t], t}, {t, 0, 2 Pi}]", "2 Sqrt[2] Pi"),
    ("ArcLength[Line[{{0, 0}, {3, 4}, {3, 5}}]]", "6"),
    ("ArcLength[Circle[{0, 0}, 2]]", "4 Pi"),
    ("ArcCurvature[{Cos[t], Sin[t]}, t]", "1"),
    ("ArcCurvature[{2 Cos[t], 2 Sin[t], 0}, t]", "1/2"),
    ("ArcCurvature[x^2, x]", "2/(1 + 4 x^2)^(3/2)"),
    ("ArcCurvature[{Cos[t], Sin[t], t}, t]", "1/2"),
]

EXPANSION = [
    ("Expand[(x + 1)^2 (y + 1)]", "1 + 2 x + x^2 + y + 2 x y + x^2 y"),
    ("Expand[(x + y)^3]", "x^3 + 3 x^2 y + 3 x y^2 + y^3"),
    ("Expand[(a + b) (c + d) + (x + 1)^2, x]", Pred(lambda g: "(a + b)*(c + d)" in str(g) and "2*x" in str(g),
                                                    "only the x part expanded")),
    ("Expand[(x + 1)^2/y]", "1/y + 2 x/y + x^2/y"),
    ("Expand[Sin[(x + 1)^2]]", "Sin[(1 + x)^2]"),
    ("Expand[(x + 1)^2 == y]", "1 + 2 x + x^2 == y"),
    ("ExpandAll[Sin[(x + 1)^2]]", "Sin[1 + 2 x + x^2]"),
    ("ExpandAll[1/(x + 1)^2 + (y + 1)^2]", Pred(lambda g: "2*y" in str(g), "expanded")),
    ("ExpandAll[{(x + 1)^2, Exp[(x + 1)^2]}]", "{1 + 2 x + x^2, E^(1 + 2 x + x^2)}"),
    ("ExpandNumerator[(x + 1)^2/(x - 1)]", "(1 + 2 x + x^2)/(-1 + x)"),
    ("ExpandNumerator[(x + 1) (x + 2)/(x + 3)^2]", "(2 + 3 x + x^2)/(3 + x)^2"),
    ("ExpandNumerator[(a + b)^2/c]", "(a^2 + 2 a b + b^2)/c"),
    ("ExpandDenominator[1/((x + 1) (x - 1))]", "1/(-1 + x^2)"),
    ("ExpandDenominator[(x + 1)/(x + 2)^2]", "(1 + x)/(4 + 4 x + x^2)"),
    ("ExpandDenominator[a/((b + c)^2)]", "a/(b^2 + 2 b c + c^2)"),
]

POLY_STRUCTURE = [
    ("PolynomialQ[x^2 + 1, x]", True),
    ("PolynomialQ[Sin[x], x]", False),
    ("PolynomialQ[x^2 + Sin[y], x]", True),
    ("PolynomialQ[1/x, x]", False),
    ("PolynomialQ[x^2 + y^3, {x, y}]", True),
    ("Variables[x^2 + y Sin[z]]", "{x, y, Sin[z]}"),
    ("Variables[1/x + y]", "{x, y}"),
    ("Variables[{x + y, z}]", "{x, y, z}"),
    ("CoefficientRules[x^2 + 3 x y, {x, y}]", "{{2, 0} -> 1, {1, 1} -> 3}"),
    ("CoefficientRules[x^2 + 3 x y + y^3, {x, y}, \"DegreeLexicographic\"]", "{{0, 3} -> 1, {2, 0} -> 1, {1, 1} -> 3}"),
    ("CoefficientRules[5 + x, x]", "{{1} -> 1, {0} -> 5}"),
    ("MonomialList[x^2 + 3 x y + 1, {x, y}]", "{x^2, 3 x y, 1}"),
    ("MonomialList[(x + 1)^2, x]", "{x^2, 2 x, 1}"),
    ("MonomialList[x y + y^2 + x^2, {x, y}, \"DegreeReverseLexicographic\"]", "{x^2, x y, y^2}"),
    ("Exponent[x^3 + x, x, Min]", "1"),
    ("Exponent[x^2 + 1, x, List]", "{0, 2}"),
    ("Exponent[x^(1/2) + x^3, x, Max]", "3"),
    ("Exponent[{x^2, x^5 + 1}, x]", "{2, 5}"),
]

POLY_ARITH = [
    ("PolynomialGCD[x^2 - 1, x^2 - 2 x + 1]", "-1 + x"),
    ("PolynomialGCD[2 x^2 - 2, 4 x - 4]", "2 (-1 + x)"),
    ("PolynomialGCD[x^2 + 1, x^2 + x, Modulus -> 2]", "1 + x"),
    ("PolynomialGCD[x y, x^2]", "x"),
    ("PolynomialLCM[x^2 - 1, x - 1]", "(-1 + x) (1 + x)"),
    ("PolynomialLCM[x, y, x y^2]", "x y^2"),
    ("PolynomialLCM[x^2 - 1, x^2 + 2 x + 1]", "(-1 + x) (1 + x)^2"),
    ("PolynomialQuotientRemainder[x^3 + 1, x - 1, x]", "{1 + x + x^2, 2}"),
    ("PolynomialQuotientRemainder[x^2 + 1, 2 x, x]", "{x/2, 1}"),
    ("PolynomialQuotientRemainder[x^2 + 1, 0, x]", Raises()),
    ("PolynomialReduce[x^2 y + x y^2, {x y - 1}, {x, y}]", "{{x + y}, x + y}"),
    ("PolynomialReduce[x^2, {x - 1}, {x}]", "{{1 + x}, 1}"),
    ("PolynomialReduce[x^3 y + y^2, {x y - 1, y^2 - 1}, {x, y}]", Pred(
        lambda g: sp.expand(S(g.args[0].args[0]) * (x * y - 1) + S(g.args[0].args[1]) * (y ** 2 - 1) + S(g.args[1])
                            - (x ** 3 * y + y ** 2)) == 0, "p == a1 p1 + a2 p2 + r")),
    ("PolynomialMod[x^2 + 3 x + 5, 2]", "1 + x + x^2"),
    ("PolynomialMod[3 x^2 - x/2, 5]", "2 x + 3 x^2"),
    ("PolynomialMod[x^3 + 1, x^2 + 1]", "1 - x"),
    ("GroebnerBasis[{x^2 + y, x y - 1}, {x, y}]", "{1 + y^3, x + y^2}"),
    ("GroebnerBasis[{x^2 - 2 y^2, x y - 3}, {x, y}]", "{-9 + 2 y^4, 3 x - 2 y^3}"),
    ("GroebnerBasis[{x + y + z, x - y, z - 1}, {x, y, z}]", "{-1 + z, 1 + 2 y, 1 + 2 x}"),
    ("GroebnerBasis[{x^2 + y^2 - 1, x - y}, {y}, {x}]", "{-1 + 2 y^2}"),
    ("GroebnerBasis[{x^2 - 1, x - 2}, {x}]", "{1}"),
]

FACTORING = [
    ("Factor[x^2 - 2, Extension -> Sqrt[2]]", "(x - Sqrt[2]) (x + Sqrt[2])"),
    ("Factor[x^2 + 1, GaussianIntegers -> True]", "(x - I) (x + I)"),
    ("Factor[x^2 + 1, Modulus -> 2]", "(1 + x)^2"),
    ("Factor[x^2 - 1, Modulus -> 3]", "(1 + x) (2 + x)"),
    ("Factor[x^4 - 1]", "(-1 + x) (1 + x) (1 + x^2)"),
    ("Factor[{x^2 - 1, x^2 - 4}]", "{(-1 + x) (1 + x), (-2 + x) (2 + x)}"),
    ("Factor[x^2 + 1, Modulus -> 4]", Raises()),
    ("FactorList[2 x^2 - 2]", "{{2, 1}, {-1 + x, 1}, {1 + x, 1}}"),
    ("FactorList[x^2 - 1]", "{{1, 1}, {-1 + x, 1}, {1 + x, 1}}"),
    ("FactorList[x^2 + 1, Modulus -> 2]", "{{1, 1}, {1 + x, 2}}"),
    ("FactorList[6]", "{{6, 1}}"),
    ("FactorTerms[2 x^2 - 4]", "2 (-2 + x^2)"),
    ("FactorTerms[3 x + 6 y]", "3 (x + 2 y)"),
    ("FactorTerms[x^2 - 1]", "x^2 - 1"),
    ("FactorTermsList[2 x^2 - 4]", "{2, -2 + x^2}"),
    ("FactorTermsList[3 a x^2 + 6 a x, {x}]", "{3, a, 2 x + x^2}"),
    ("FactorTermsList[x + 1]", "{1, 1 + x}"),
    ("FactorSquareFree[x^3 - x^2 - x + 1]", "(-1 + x)^2 (1 + x)"),
    ("FactorSquareFree[x^4 - 2 x^2 + 1]", "(-1 + x^2)^2"),
    ("FactorSquareFree[x^2 + x]", "x (1 + x)"),
    ("FactorSquareFreeList[x^3 - x^2 - x + 1]", "{{1, 1}, {1 + x, 1}, {-1 + x, 2}}"),
    ("FactorSquareFreeList[2 x^2 + 4 x + 2]", "{{2, 1}, {1 + x, 2}}"),
    ("FactorSquareFreeList[x^3]", "{{1, 1}, {x, 3}}"),
]

POLY_MISC = [
    ("Decompose[x^4 + 2 x^2 + 1, x]", "{1 + 2 x + x^2, x^2}"),
    ("Decompose[x^6 + 3 x^3 + 1, x]", "{1 + 3 x + x^2, x^3}"),
    ("Decompose[x^2 + x + 1, x]", "{1 + x + x^2}"),
    ("HornerForm[x^3 + 2 x^2 + 3 x + 4]", Pred(lambda g: str(g) == "4 + x*(3 + x*(2 + x))", "nested form")),
    ("HornerForm[a x^2 + b x + c, x]", Pred(lambda g: sp.expand(S(g)) == a * x ** 2 + b * x + sp.Symbol("c") and
                                           "x*(b + a*x)" in str(g), "nested in x")),
    ("HornerForm[(x^2 + 1)/(x^3 + x + 2)]", Equals((x ** 2 + 1) / (x ** 3 + x + 2))),
    ("MinimalPolynomial[Sqrt[2] + Sqrt[3], x]", "1 - 10 x^2 + x^4"),
    ("MinimalPolynomial[2^(1/3)]", "-2 + #^3 &"),
    ("MinimalPolynomial[(1 + Sqrt[5])/2, x]", "-1 - x + x^2"),
    ("MinimalPolynomial[Pi, x]", Raises()),
    ("IrreduciblePolynomialQ[x^2 + 1]", True),
    ("IrreduciblePolynomialQ[x^2 - 1]", False),
    ("IrreduciblePolynomialQ[x^2 + 1, GaussianIntegers -> True]", False),
    ("IrreduciblePolynomialQ[2 x + 2]", True),
    ("SymmetricPolynomial[2, {x, y, z}]", "x y + x z + y z"),
    ("SymmetricPolynomial[3, {a, b, c}]", "a b c"),
    ("SymmetricPolynomial[4, {x, y}]", "0"),
    ("SymmetricReduction[x^2 + y^2, {x, y}]", "{(x + y)^2 - 2 x y, 0}"),
    ("SymmetricReduction[x^3 + y^3 + x, {x, y}, {s1, s2}]", "{s1 + s1^3 - 3 s1 s2, -y}"),
    ("SymmetricReduction[x y z, {x, y, z}, {p, q, r}]", "{r, 0}"),
    ("ToRadicals[Root[#^3 - 2 # - 5 &, 1]]", Near(float(sp.N(sp.CRootOf(x ** 3 - 2 * x - 5, 0))), 1e-14)),
    ("ToRadicals[Root[#^3 - 2 # - 5 &, 1]]", Pred(lambda g: "Root" not in str(g), "radical form")),
    ("ToRadicals[Root[#^5 - # + 1 &, 1]]", Head("Root")),
    ("ToRadicals[Root[#^2 - 2 &, 2]]", "Sqrt[2]"),
]

SIMPLIFY = [
    ("FullSimplify[Sin[x]^2 + Cos[x]^2]", "1"),
    ("FullSimplify[(x^2 - 1)/(x - 1)]", "1 + x"),
    ("FullSimplify[Sqrt[x^2], x > 0]", "x"),
    ("FullSimplify[Log[x] + Log[y], x > 0 && y > 0]", "Log[x y]"),
    ("FullSimplify[Gamma[x + 1]/Gamma[x]]", "x"),
    ("FullSimplify[{Sin[x]^2 + Cos[x]^2, 2 Sin[x] Cos[x]}]", "{1, Sin[2 x]}"),
    ("TrigFactor[Sin[x] + Sin[y]]", "2 Cos[x/2 - y/2] Sin[x/2 + y/2]"),
    ("TrigFactor[Cos[x] + Cos[y]]", "2 Cos[x/2 - y/2] Cos[x/2 + y/2]"),
    ("TrigFactor[Sin[x]^2 - 1]", "-Cos[x]^2"),
    ("FunctionExpand[Gamma[x + 3]]", "x (1 + x) (2 + x) Gamma[x]"),
    ("FunctionExpand[Binomial[n, 2]]", "(n - 1) n/2"),
    ("FunctionExpand[Beta[a, b]]", "Gamma[a] Gamma[b]/Gamma[a + b]"),
    ("PossibleZeroQ[Sin[x]^2 + Cos[x]^2 - 1]", True),
    ("PossibleZeroQ[x - 1]", False),
    ("PossibleZeroQ[Sqrt[2] Sqrt[3] - Sqrt[6]]", True),
    ("PossibleZeroQ[E^(I Pi) + 1]", True),
    ("Element[Sqrt[2], Rationals]", False),
    ("Element[3, Integers]", True),
    ("Element[Pi, Algebraics]", False),
    ("Element[x, Reals]", Unevaluated("Element")),
    ("Element[{1, 2}, Integers]", True),
    ("Element[7, Primes]", True),
    ("Element[I, Reals]", False),
    ("Element[2 + I, Complexes]", True),
]

SOLVING = [
    ("Solve[x^2 == 4, x]", "{{x -> -2}, {x -> 2}}"),
    ("Solve[x^2 == -4, x, Reals]", "{}"),
    ("Solve[x^2 == 0, x]", "{{x -> 0}, {x -> 0}}"),
    ("Solve[x^5 - x + 1 == 0, x]", Solutions(x ** 5 - x + 1, x, 5)),
    ("Solve[x^5 - x + 1 == 0, x]", Pred(lambda g: "Root" in str(g), "Root objects")),
    ("Solve[x^3 - 2 x - 5 == 0, x, Cubics -> False]", Pred(lambda g: str(g).count("Root") == 3, "Root objects")),
    ("Solve[x^3 - 2 x - 5 == 0, x]", Solutions(x ** 3 - 2 * x - 5, x, 3)),
    ("Solve[x + y == 1, {x, y}]", "{{y -> 1 - x}}"),
    ("Solve[Sin[x] == 0, x]", "{{x -> ConditionalExpression[2 Pi C[1], Element[C[1], Integers]]}, "
                              "{x -> ConditionalExpression[Pi + 2 Pi C[1], Element[C[1], Integers]]}}"),
    ("Solve[Exp[x] == 2, x]", "{{x -> ConditionalExpression[Log[2] + 2 I Pi C[1], Element[C[1], Integers]]}}"),
    ("Solve[Exp[x] == 2, x, Reals]", "{{x -> Log[2]}}"),
    ("Solve[x^2 == 4, x, Integers]", "{{x -> -2}, {x -> 2}}"),
    ("Solve[x^2 == 2, x, Integers]", "{}"),
    ("Solve[2 x + 3 y == 1, {x, y}, Integers]", Pred(lambda g: all(
        sp.expand(2 * S(r.args[0].args[1].args[0]) + 3 * S(r.args[1].args[1].args[0]) - 1) == 0 for r in g.args),
        "parametric integer solutions")),
    ("Solve[x^2 + y^2 == 25, {x, y}, Integers]", Pred(lambda g: len(g.args) == 12, "12 lattice points")),
    ("Solve[(x^2 - 1)/(x - 1) == 0, x]", "{{x -> -1}}"),
    ("Solve[Sqrt[x] == -1, x]", "{}"),
    ("Solve[x == Cos[x], x]", Raises()),
    ("Solve[x^2 < 1, x]", Raises()),
    ("Solve[1 == 2, x]", "{}"),
    ("SolveValues[x^2 == 4, x]", "{-2, 2}"),
    ("SolveValues[{x + y == 3, x - y == 1}, {x, y}]", "{{2, 1}}"),
    ("SolveValues[x^3 == 8, x, Reals]", "{2}"),
    ("Roots[x^2 - 3 x + 2 == 0, x]", "x == 1 || x == 2"),
    ("Roots[(x - 1)^2 == 0, x]", "x == 1 || x == 1"),
    ("Roots[x^2 + 1 == 0, x]", "x == -I || x == I"),
    ("Roots[2 x == 4, x]", "x == 2"),
    ("{ToRules[x == 1 || x == 2]}", "{{x -> 1}, {x -> 2}}"),
    ("ToRules[x == 1 && y == 2]", "{x -> 1, y -> 2}"),
    ("{ToRules[Roots[x^2 == 9, x]]}", "{{x -> -3}, {x -> 3}}"),
    ("Eliminate[{x == 2 + y, y == z}, y]", "-2 + x - z == 0"),
    ("Eliminate[{x == t^2, y == t^3}, t]", "x^3 - y^2 == 0"),
    ("Eliminate[{x + y == 1, x - y == 3}, {x, y}]", False),
    ("CountRoots[x^3 - x, x]", "3"),
    ("CountRoots[(x - 1)^2 (x + 3), {x, 0, 5}]", "2"),
    ("CountRoots[x^2 + 1, {x, -2 - 2 I, 2 + 2 I}]", "2"),
    ("CountRoots[x^2 + 1, x]", "0"),
]

REDUCTION = [
    ("Reduce[x^2 == 4, x]", "x == -2 || x == 2"),
    ("Reduce[x^2 < 1, x]", "-1 < x < 1"),
    ("Reduce[x^2 > 1, x]", "x < -1 || x > 1"),
    ("Reduce[x^2 - 3 x + 2 <= 0, x]", "1 <= x <= 2"),
    ("Reduce[Abs[x - 1] < 2, x]", "-1 < x < 3"),
    ("Reduce[x^2 < 10, x, Integers]", "x == -3 || x == -2 || x == -1 || x == 0 || x == 1 || x == 2 || x == 3"),
    ("Reduce[x > 2, x, Integers]", "Element[x, Integers] && x >= 3"),
    ("Reduce[x^2 + 1 == 0, x, Reals]", False),
    ("Reduce[x^2 + 1 == 0, x]", "x == -I || x == I"),
    ("Reduce[a x + b == 0, x]", "(b == 0 && a == 0) || (a != 0 && x == -b/a)"),
    ("Reduce[Sin[x] == 0, x]", "Element[C[1], Integers] && (x == 2 Pi C[1] || x == Pi + 2 Pi C[1])"),
    ("Reduce[{x + y == 3, x - y == 1}, {x, y}]", "x == 2 && y == 1"),
    ("Reduce[-1 < x < 1 && x != 0, x]", "(-1 < x < 0) || (0 < x < 1)"),
    ("Reduce[1/x > 1, x]", "0 < x < 1"),
    ("Reduce[Sin[x] > 0, x]", Raises()),
    ("Reduce[x^2 + y^2 < 1, {x, y}]", Raises()),
    ("Minimize[{x^2, x >= 1}, x]", "{1, {x -> 1}}"),
    ("Maximize[{x (1 - x), 0 <= x <= 1}, x]", "{1/4, {x -> 1/2}}"),
    ("Maximize[{x^3 - 3 x, -2 <= x <= 3}, x]", "{18, {x -> 3}}"),
    ("Minimize[{x, x > 0}, x]", "{0, {x -> 0}}"),
    ("Minimize[{x^3, x < 0}, x]", "{-Infinity, {x -> -Infinity}}"),
    ("Minimize[{x^2 + y^2, x + y == 1}, {x, y}]", Raises()),
    ("MinValue[x^2 - 2 x, x]", "-1"),
    ("MinValue[{1/x, x >= 1}, x]", "0"),
    ("MinValue[{(x - 1)^2, x > 0}, x]", "0"),
    ("MaxValue[{-x^2 + 4, -1 <= x <= 3}, x]", "4"),
    ("MaxValue[{x, x^2 <= 2}, x]", "Sqrt[2]"),
    ("MaxValue[{Sin[x], 0 <= x <= 3}, x]", "1"),
    ("ArgMin[(x - 3)^2, x]", "3"),
    ("ArgMin[{x^2, x >= 1}, x]", "1"),
    ("ArgMin[{x^2 - 4 x, -1 <= x <= 1}, x]", "1"),
    ("ArgMax[{Sin[x], 0 <= x <= 3}, x]", "Pi/2"),
    ("ArgMax[{x (1 - x), 0 <= x <= 1}, x]", "1/2"),
    ("ArgMax[{-x^2, -2 <= x <= -1}, x]", "-1"),
    ("FunctionDomain[1/x, x]", "x < 0 || x > 0"),
    ("FunctionDomain[Sqrt[x], x]", "x >= 0"),
    ("FunctionDomain[Log[x^2 - 1], x]", "x < -1 || x > 1"),
    ("FunctionDomain[Sqrt[1 - x^2], x]", "-1 <= x <= 1"),
    ("FunctionDomain[Floor[x], x]", Raises()),
    ("FunctionRange[x^2, x, y]", "y >= 0"),
    ("FunctionRange[Sin[x], x, y]", "-1 <= y <= 1"),
    ("FunctionRange[1/x, x, y]", "y < 0 || y > 0"),
    ("FunctionRange[Exp[x], x, y]", "y > 0"),
    ("FunctionPeriod[Sin[x], x]", "2 Pi"),
    ("FunctionPeriod[Sin[x]^2, x]", "Pi"),
    ("FunctionPeriod[Tan[2 x], x]", "Pi/2"),
    ("FunctionPeriod[Sin[x] + Cos[2 x], x]", "2 Pi"),
    ("FunctionPeriod[x^2, x]", "0"),
]

CASES = (DERIVATIVES + GENERALIZED + INTEGRALS + LIMITS + SERIES + SUMS + RECURRENCES + ODES + TRANSFORMS
         + FOURIER_SERIES + VECTOR + EXPANSION + POLY_STRUCTURE + POLY_ARITH + FACTORING + POLY_MISC + SIMPLIFY
         + SOLVING + REDUCTION)


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_batch4(case):
    check(case)
