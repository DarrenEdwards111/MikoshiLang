import math

import mpmath as mp
import pytest

from _harness import Raises, Pred, check, ids

needs_scipy = pytest.importorskip


def _scipy():
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


SCIPY = _scipy()

CASES = [
    ("Integrate[x^2, x]", "x^3/3"), ("Integrate[Sin[x], {x, 0, Pi}]", "2"), ("Integrate[1/x, {x, 1, E}]", "1"),
    ("Integrate[x y, {x, 0, 1}, {y, 0, x}]", "1/8"), ("Integrate[Exp[-x^2], {x, -Infinity, Infinity}]", "Sqrt[Pi]"),
    ("Integrate[1/x, {x, -1, 1}]", Raises()),
    ("DefiniteIntegral[x^2, {x, 0, 3}]", "9"), ("DefiniteIntegral[Cos[x], {x, 0, Pi/2}]", "1"),
    ("DefiniteIntegral[Exp[-x], {x, 0, Infinity}]", "1"), ("DefiniteIntegral[x^2, x]", Raises()),
    ("Sum[i, {i, 1, 100}]", "5050"), ("Sum[1/k^2, {k, 1, Infinity}]", "Pi^2/6"), ("Sum[i^2, {i, 4}]", "30"),
    ("Sum[i, {i, 1, 10, 3}]", "22"), ("Sum[i j, {i, 2}, {j, i}]", "1 + 2 + 4"), ("Sum[x^i, {i, {1, 3}}]", "x + x^3"),
    ("Sum[1/k, {k, 1, Infinity}]", "Infinity"), ("Sum[(-1)^k, {k, 1, Infinity}]", "Sum[(-1)^k, {k, 1, Infinity}]"),
    ("Product[i, {i, 1, 5}]", "120"), ("Product[i, {i, 3}]", "6"), ("Product[2, {i, 1, 10, 2}]", "32"),
    ("Product[x - i, {i, {1, 2}}]", "(x - 1) (x - 2)"),
    ("TaylorSeries[Exp[x], {x, 0, 3}]", "1 + x + x^2/2 + x^3/6"), ("TaylorSeries[Sin[x], {x, 0, 5}]", "x - x^3/6 + x^5/120"),
    ("TaylorSeries[Log[x], {x, 1, 2}]", "(x - 1) - (x - 1)^2/2"),
    ("MaClaurinSeries[Cos[x], {x, 4}]", "1 - x^2/2 + x^4/24"), ("MaClaurinSeries[1/(1 - x), {x, 3}]", "1 + x + x^2 + x^3"),
    ("MaClaurinSeries[Exp[x], {x, 0, 2}]", "1 + x + x^2/2"), ("MaClaurinSeries[Exp[x], {x, 1, 2}]", Raises()),
    ("TaylorPolynomial[Exp[x], x, 0, 2]", "1 + x + x^2/2"), ("TaylorPolynomial[Sqrt[x], x, 1, 1]", "1 + (x - 1)/2"),
    ("TaylorPolynomial[x^3, x, 2, 1]", "8 + 12 (x - 2)"),
    ("LaurentSeries[Exp[x]/x^2, x, 0, 1]", "1/x^2 + 1/x + 1/2 + x/6"), ("LaurentSeries[1/(x (1 - x)), x, 0, 1]", "1/x + 1 + x"),
    ("LaurentSeries[Cos[x]/x, x, 0, 1]", "1/x - x/2"),
    ("NthDerivative[x^5, x, 3]", "60 x^2"), ("NthDerivative[Sin[x], x, 4]", "Sin[x]"), ("NthDerivative[x^2, x, 0]", "x^2"),
    ("NthDerivative[x^2, x, 1.5]", Raises()),
    ("Residue[1/(x - 1), {x, 1}]", "1"), ("Residue[Exp[z]/z^3, {z, 0}]", "1/2"), ("Residue[1/(z^2 + 1), z, I]", "-I/2"),
    ("ArcLength[x, {x, 0, 1}]", "Sqrt[2]"), ("ArcLength[x^(3/2), {x, 0, 1}]", "(13 Sqrt[13] - 8)/27"),
    ("ArcLength[Cosh[x], {x, 0, 1}]", "Sinh[1]"),
    ("Gradient[x^2 y, {x, y}]", "{2 x y, x^2}"), ("Gradient[x + y + z, {x, y, z}]", "{1, 1, 1}"), ("Gradient[5, {x}]", "{0}"),
    ("Grad[x^2 + y^2, {x, y}]", "{2 x, 2 y}"), ("Grad[x y z, {x, y, z}]", "{y z, x z, x y}"),
    ("Grad[{x y, x}, {x, y}]", "{{y, x}, {1, 0}}"),
    ("Divergence[{x, y, z}, {x, y, z}]", "3"), ("Divergence[{x^2, x y}, {x, y}]", "3 x"), ("Divergence[{x}, {x, y}]", Raises()),
    ("Div[{x^2, y^2}, {x, y}]", "2 x + 2 y"), ("Div[{y, x}, {x, y}]", "0"), ("Div[{x y z, 0, 0}, {x, y, z}]", "y z"),
    ("Laplacian[x^2 + y^2, {x, y}]", "4"), ("Laplacian[x^2 y^2 z^2, {x, y, z}]", "2 y^2 z^2 + 2 x^2 z^2 + 2 x^2 y^2"),
    ("Laplacian[Sin[x], {x}]", "-Sin[x]"),
    ("Curl[{-y, x, 0}, {x, y, z}]", "{0, 0, 2}"), ("Curl[{y z, x z, x y}, {x, y, z}]", "{0, 0, 0}"),
    ("Curl[{-y, x}, {x, y}]", "2"), ("Curl[{x, y, z, w}, {x, y, z, w}]", Raises()),
    ("HessianMatrix[x^2 y, {x, y}]", "{{2 y, 2 x}, {2 x, 0}}"), ("HessianMatrix[x^3, {x}]", "{{6 x}}"),
    ("HessianMatrix[x y z, {x, y, z}]", "{{0, z, y}, {z, 0, x}, {y, x, 0}}"),
    ("JacobianMatrix[{x y, x + y}, {x, y}]", "{{y, x}, {1, 1}}"), ("JacobianMatrix[{x^2}, {x}]", "{{2 x}}"),
    ("JacobianMatrix[{r Cos[t], r Sin[t]}, {r, t}]", "{{Cos[t], -r Sin[t]}, {Sin[t], r Cos[t]}}"),
    ("DirectionalDerivative[x^2 + y^2, {x, y}, {1, 0}]", "2 x"), ("DirectionalDerivative[x y, {x, y}, {3, 4}]", "(3 y + 4 x)/5"),
    ("DirectionalDerivative[x, {x, y}, {0, 0}]", Raises()),
    ("CriticalPoints[x^2 + y^2, {x, y}]", "{{x -> 0, y -> 0}}"), ("CriticalPoints[x^3 - 3 x, {x}]", "{{x -> -1}, {x -> 1}}"),
    ("CriticalPoints[x y, {x, y}]", "{{x -> 0, y -> 0}}"),
    ("PhasePlane[{y, -x}, {x, y}]", "{{x -> 0, y -> 0}}"), ("PhasePlane[{x - x y, -y + x y}, {x, y}]", "{{x -> 0, y -> 0}, {x -> 1, y -> 1}}"),
    ("PhasePlane[{x^2 - 1}, {x}]", "{{x -> -1}, {x -> 1}}"),
    ("InflectionPoints[x^3, x]", "{0}"), ("InflectionPoints[x^4, x]", "{}"), ("InflectionPoints[x^3 - 3 x^2, x]", "{1}"),
    ("Minimize[x^2 - 2 x, x]", "{-1, {x -> 1}}"), ("Minimize[x^3, x]", Pred(lambda r: str(r.args[0]) in ("-Infinity", "Times[-1, Infinity]") or "Infinity" in str(r), "unbounded")),
    ("Minimize[(x - 2)^2 + 1, x]", "{1, {x -> 2}}"), ("Minimize[Exp[x], x]", Raises()),
    ("Maximize[-x^2 + 4 x, x]", "{4, {x -> 2}}"), ("Maximize[1 - x^2, x]", "{1, {x -> 0}}"), ("Maximize[Sin[x], x]", Raises(NotImplementedError, "could not determine")),
    ("LinearProgramming[{1, 1}, {{1, 2}}, {3}]", "{0, 3/2}"), ("LinearProgramming[{2, 3}, {{1, 1}, {1, -1}}, {4, 0}]", "{4, 0}"),
    ("LinearProgramming[{-1, 0}, {{1, 0}}, {1}]", Raises()), ("LinearProgramming[{1}, {{-1}}, {1}]", Raises()),
    ("RSolve[{a[n + 1] == 2 a[n], a[0] == 3}, a[n], n]", "{{a[n] -> 3 2^n}}"),
    ("RSolve[a[n + 1] == 2 a[n], a[n], n]", "{{a[n] -> 2^n C[1]}}"),
    ("RSolve[{a[n + 1] == a[n] + 1, a[0] == 0}, a[n], n]", "{{a[n] -> n}}"),
    ("GeneratingFunction[{1, 2, 3}, x]", "1 + 2 x + 3 x^2"), ("GeneratingFunction[{1/2}, x]", "1/2"),
    ("GeneratingFunction[{0, 0, 1}, t]", "t^2"),
    ("Wronskian[{Sin[x], Cos[x]}, x]", "-1"), ("Wronskian[{1, x, x^2}, x]", "2"), ("Wronskian[{Exp[x], Exp[2 x]}, x]", "E^(3 x)"),
    ("ExactODEQ[2 x y, x^2, x, y]", True), ("ExactODEQ[y, -x, x, y]", False), ("ExactODEQ[3 x^2, 3 y^2, x, y]", True),
    ("IntegratingFactor[3 x y + y^2, x^2 + x y, x, y]", "x"), ("IntegratingFactor[y, -x, x, y]", Pred(lambda r: r is not None, "a factor")),
    ("IntegratingFactor[x^2 + y, x y^3 + x^5, x, y]", Raises()),
    ("LyapunovDerivative[{-x, -y}, {x, y}, x^2 + y^2]", "-2 x^2 - 2 y^2"), ("LyapunovDerivative[{y, -x}, {x, y}, x^2 + y^2]", "0"),
    ("LyapunovDerivative[{-x}, {x, y}, x^2]", Raises()),
    ("ODEClassify[y'[x] == y[x], y[x], x]", Pred(lambda r: "separable" in r.args and "1st_linear" in r.args, "hints")),
    ("ODEClassify[y''[x] + y[x] == 0, y[x], x]", Pred(lambda r: "nth_linear_constant_coeff_homogeneous" in r.args, "hints")),
    ("ODEClassify[y'[x] == x, y[x], x]", Pred(lambda r: "nth_algebraic" in r.args or "separable" in r.args, "hints")),
    ("SeparableODE[y'[x] == x y[x], y[x], x]", "{{y[x] -> C[1] E^(x^2/2)}}"),
    ("SeparableODE[y'[x] == y[x], y[x], x]", "{{y[x] -> C[1] E^x}}"), ("SeparableODE[y'[x] == x + y[x], y[x], x]", Raises()),
    ("LaplaceODESolve[y'[t] == y[t], y, t]", Raises(NotImplementedError, "DSolve")),
    ("Table[i^2, {i, 3}]", "{1, 4, 9}"),
]

if SCIPY:
    CASES += [
        ("FindMinimum[(x - 1)^2 + 3, {x, 0}]", Pred(lambda r: abs(r.args[0] - 3) < 1e-9 and abs(r.args[1].args[0].args[1] - 1) < 1e-6, "min at 1")),
        ("FindMinimum[(x - 1)^2 + (y + 2)^2, {{x, 0}, {y, 0}}]", Pred(lambda r: abs(r.args[0]) < 1e-9, "min 0")),
        ("FindMinimum[Cos[x], {x, 3}]", Pred(lambda r: abs(r.args[0] + 1) < 1e-9 and abs(r.args[1].args[0].args[1] - math.pi) < 1e-5, "min at pi")),
    ]
else:
    CASES += [("FindMinimum[(x - 1)^2, {x, 0}]", Raises(NotImplementedError, "SciPy"))] * 3


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_calculus(case):
    check(case)
