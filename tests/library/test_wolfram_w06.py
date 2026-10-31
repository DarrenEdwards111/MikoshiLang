"""Differential tests for coverage batch W06 (calculus extras: asymptotics, discrete calculus, line/surface/contour
integrals, fractional calculus, dynamical systems, matrix equations, interpolation and symbolic arrays).

Expected values come from the Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>.html:
basic examples and the identities in their Details sections), classical closed forms (Stirling, Binet, residue
theorem, divergence theorem, Riemann-Liouville power rule), or independent references computed here with mpmath
quadrature, SymPy or plain Python.
"""
import math

import mpmath
import pytest
import sympy as sp

from _harness import Pred, Raises, check, ids, ev, same
from mikoshilang.expr import Expr, Symbol
from mikoshilang.math_ops import to_sympy

try:
    import scipy  # noqa: F401
    SCIPY = True
except ImportError:  # minimal install: inexact Riccati solves need SciPy and must raise
    SCIPY = False


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────

def _eq_diff(e):
    return to_sympy(e.args[0]) - to_sympy(e.args[1])


def _struct(got, want):
    if isinstance(want, Expr) and isinstance(got, Expr):
        if want.head == "Equal" and got.head == "Equal":
            d = sp.simplify(_eq_diff(got) - _eq_diff(want))
            if d == 0:
                return True
            r = sp.simplify(_eq_diff(got) / _eq_diff(want))
            return r.is_number and r != 0
        if want.head == "List" and got.head == "List" or (not isinstance(want.head, str)):
            hs = (want.head == got.head) if isinstance(want.head, str) else _struct(got.head, want.head)
            return hs and len(got.args) == len(want.args) and all(_struct(g, w) for g, w in zip(got.args, want.args))
        if isinstance(want.head, str) and want.head == got.head and want.head in ("Inactive", "Rule", "Function"):
            return len(got.args) == len(want.args) and all(_struct(g, w) for g, w in zip(got.args, want.args))
    return same(got, want)


def S(src):
    """Structural comparison with an expected source (Equal compared up to a constant factor)."""
    return Pred(lambda g: _struct(g, ev(src)), src)


def N(x, tol=1e-9):
    return Pred(lambda g: abs(complex(sp.N(to_sympy(g))) - complex(x)) <= tol * max(1.0, abs(complex(x))), repr(x))


def lst_close(xs, tol=1e-8):
    def f(g):
        vs = [complex(sp.N(v)) for v in sp.flatten(to_sympy(g))] if isinstance(to_sympy(g), list) else None
        return vs is not None and len(vs) == len(xs) and all(abs(a - b) <= tol * max(1, abs(b)) for a, b in zip(vs, xs))
    return Pred(f, repr(xs))


def is_true(src):
    return Pred(lambda g: g is True or (isinstance(g, Symbol) and g.name == "True"), src)


def mat(g):
    return sp.Matrix(to_sympy(g))


def lyap_ok(a, c, kind):
    A, C = sp.Matrix(a), sp.Matrix(c)

    def f(g):
        X = mat(g)
        if kind == "cont":
            r = A * X + X * A.H - C
        else:
            r = A * X * A.H - X - C
        return all(abs(complex(sp.N(v))) < 1e-9 for v in r)
    return Pred(f, f"{kind} Lyapunov residual 0")


def terms_of(src):
    def f(g):
        w = ev(src)
        ga = g.args if isinstance(g, Expr) and g.head == "Plus" else [g]
        wa = w.args if isinstance(w, Expr) and w.head == "Plus" else [w]
        return sorted(map(str, ga)) == sorted(map(str, wa))
    return Pred(f, src)


def at_n(src, pts=(5, 8, 13)):
    """Equal to src as a function of n (checked exactly at a few integers)."""
    def f(g):
        n = sp.Symbol("n")
        return all(sp.simplify(sp.nsimplify(to_sympy(g).subs(n, k)) - to_sympy(ev(src)).subs(n, k)) == 0
                   or abs(complex(sp.N(to_sympy(g).subs(n, k) - to_sympy(ev(src)).subs(n, k), 30))) < 1e-20
                   for k in pts)
    return Pred(f, src)


def quad1(f, a, b):
    return complex(mpmath.quad(f, [a, b]))


# ── symbolic arrays, domains and symmetries ────────────────────────────────────────────────────

ARRAYS = [
    ("Vectors[3]", "Vectors[3]"), ("Vectors[3, Reals]", "Vectors[3, Reals]"), ("Vectors[-1]", Raises()),
    ("Vectors[n, Complexes]", "Vectors[n, Complexes]"),
    ("Matrices[{2, 3}]", "Matrices[{2, 3}]"), ("Matrices[{2, 2}, Reals, Symmetric[{1, 2}]]",
                                                "Matrices[{2, 2}, Reals, Symmetric[{1, 2}]]"),
    ("Matrices[{2, -3}]", Raises()), ("Matrices[{n, n}, Complexes, Hermitian[{1, 2}]]",
                                      "Matrices[{n, n}, Complexes, Hermitian[{1, 2}]]"),
    ("Arrays[{2, 2, 2}]", "Arrays[{2, 2, 2}]"), ("Arrays[{2, 2}, Reals, Antisymmetric[{1, 2}]]",
                                                 "Arrays[{2, 2}, Reals, Antisymmetric[{1, 2}]]"),
    ("Arrays[{1.5}]", Raises()),
    ("Symmetric[{1, 2}]", "Symmetric[{1, 2}]"), ("Symmetric[{0}]", Raises()), ("Symmetric[{1, 2, 3}]", "Symmetric[{1, 2, 3}]"),
    ("Antisymmetric[{}]", Raises()), ("Antisymmetric[{1, 3}]", "Antisymmetric[{1, 3}]"),
    ("Antisymmetric[{2, 1}]", "Antisymmetric[{2, 1}]"),
    ("Hermitian[{1, 2, 3}]", Raises()), ("Hermitian[{1, 2}]", "Hermitian[{1, 2}]"),
    ("Antihermitian[{1, 2}]", "Antihermitian[{1, 2}]"), ("Antihermitian[{1}]", Raises()),
    ("Antihermitian[{1, 2}] === Antihermitian[{1, 2}]", True),
    ("ZeroSymmetric[{1, 2}]", "ZeroSymmetric[{1, 2}]"), ("ZeroSymmetric[{-1}]", Raises()),
    ("ZeroSymmetric[x]", Raises()), ("Symmetric[All]", "Symmetric[All]"), ("Antisymmetric[All]", "Antisymmetric[All]"),
    ("Hermitian[All]", Raises()),
    ("ArraySymbol[\"a\", {3, 3, 3}, Reals, Symmetric[All]]", "ArraySymbol[\"a\", {3, 3, 3}, Reals, Symmetric[All]]"), ("Head[ZeroSymmetric[{2, 3}]]", "ZeroSymmetric"),
    ("TensorSymmetry[{{0, 0}, {0, 0}}]", "ZeroSymmetric[{1, 2}]"),
    ("TensorSymmetry[ConstantArray[0, {2, 2, 2}]]", "ZeroSymmetric[{1, 2, 3}]"),
    ("TensorSymmetry[{{1, 2}, {2, 1}}]", "Symmetric[{1, 2}]"), ("TensorSymmetry[{{0, 2}, {-2, 0}}]", "Antisymmetric[{1, 2}]"),
    ("TensorSymmetry[{{1, 2}, {3, 4}}]", "{}"), ("TensorSymmetry[LeviCivitaTensor[3]]", "Antisymmetric[{1, 2, 3}]"),
    ("TensorSymmetry[MatrixSymbol[\"A\", {n, n}, Reals, Symmetric[{1, 2}]]]", "Symmetric[{1, 2}]"),
    ("VectorSymbol[\"v\", 3]", "VectorSymbol[\"v\", 3]"), ("VectorSymbol[\"v\", n, Reals]", "VectorSymbol[\"v\", n, Reals]"),
    ("VectorSymbol[\"v\", -2]", Raises()),
    ("MatrixSymbol[\"A\", {2, 3}]", "MatrixSymbol[\"A\", {2, 3}]"), ("MatrixSymbol[\"A\", {2}]", Raises()),
    ("MatrixSymbol[\"A\", {n, n}, Reals, Antisymmetric[{1, 2}]]", "MatrixSymbol[\"A\", {n, n}, Reals, Antisymmetric[{1, 2}]]"),
    ("ArraySymbol[\"T\", {2, 3, 4}]", "ArraySymbol[\"T\", {2, 3, 4}]"), ("ArraySymbol[\"T\", {-1}]", Raises()),
    ("ArraySymbol[\"T\", {2, 2}, Reals, Symmetric[{1, 2}]]", "ArraySymbol[\"T\", {2, 2}, Reals, Symmetric[{1, 2}]]"),
    ("Normal[SymbolicZerosArray[{2, 3}]]", "{{0, 0, 0}, {0, 0, 0}}"), ("SymbolicZerosArray[{2, 3}]", "SymbolicZerosArray[{2, 3}]"),
    ("Normal[SymbolicZerosArray[{2}], TargetStructure -> \"Dense\"]", "{0, 0}"),
    ("Normal[SymbolicOnesArray[{2, 2}]]", "{{1, 1}, {1, 1}}"), ("Normal[SymbolicOnesArray[{3}]]", "{1, 1, 1}"),
    ("SymbolicOnesArray[{-2}]", Raises()),
    ("Normal[SymbolicIdentityArray[{2}]]", "{{1, 0}, {0, 1}}"), ("Dimensions[Normal[SymbolicIdentityArray[{2, 3}]]]", "{2, 3, 2, 3}"),
    ("Normal[SymbolicIdentityArray[{2}], TargetStructure -> \"Dense\"]", "{{1, 0}, {0, 1}}"),
    ("Normal[SymbolicIdentityArray[{2}], TargetStructure -> \"Sparse\"]", Raises()),
    ("Normal[SymbolicDeltaProductArray[{2, 2, 2}, {{1, 3}}]]", "{{{1, 0}, {1, 0}}, {{0, 1}, {0, 1}}}"),
    ("Normal[SymbolicDeltaProductArray[{2, 2}, {{1, 2}}]]", "{{1, 0}, {0, 1}}"),
    ("SymbolicDeltaProductArray[{2, 2}, {{1, 1}}]", Raises()),
    ("ArrayExpand[MatrixSymbol[\"A\", {n, n}].(MatrixSymbol[\"B\", {n, n}] + MatrixSymbol[\"C\", {n, n}])]",
     "MatrixSymbol[\"A\", {n, n}].MatrixSymbol[\"B\", {n, n}] + MatrixSymbol[\"A\", {n, n}].MatrixSymbol[\"C\", {n, n}]"),
    ("ArrayExpand[Transpose[MatrixSymbol[\"A\", {n, n}].MatrixSymbol[\"B\", {n, n}]]]",
     "Transpose[MatrixSymbol[\"B\", {n, n}]].Transpose[MatrixSymbol[\"A\", {n, n}]]"),
    ("ArrayExpand[{{a, b}, {c, d}}.{{1, 0}, {0, 1}}]", "{{a, b}, {c, d}}"),
    ("ArrayExpand[{(a + b)^2}]", "{a^2 + 2 a b + b^2}"),
    ("ArraySimplify[Transpose[Transpose[MatrixSymbol[\"A\", {n, n}]]]]", "MatrixSymbol[\"A\", {n, n}]"),
    ("ArraySimplify[MatrixSymbol[\"A\", {n, n}].Inverse[MatrixSymbol[\"A\", {n, n}]]]", "SymbolicIdentityArray[{n}]"),
    ("ArraySimplify[Transpose[MatrixSymbol[\"S\", {n, n}, Reals, Symmetric[{1, 2}]]]]",
     "MatrixSymbol[\"S\", {n, n}, Reals, Symmetric[{1, 2}]]"),
    ("ArraySimplify[Transpose[MatrixSymbol[\"S\", {n, n}, Reals, Antisymmetric[{1, 2}]]]]",
     "-MatrixSymbol[\"S\", {n, n}, Reals, Antisymmetric[{1, 2}]]"),
    ("ArraySimplify[MatrixSymbol[\"A\", {n, n}].SymbolicIdentityArray[{n}]]", "MatrixSymbol[\"A\", {n, n}]"),
    ("ArraySimplify[MatrixSymbol[\"A\", {n, n}] + SymbolicZerosArray[{n, n}]]", "MatrixSymbol[\"A\", {n, n}]"),
    ("ArraySimplify[{{Sin[x]^2 + Cos[x]^2}}]", "{{1}}"),
    ("TensorWedge[{a1, a2}, {b1, b2}]", "{{0, a1 b2 - a2 b1}, {a2 b1 - a1 b2, 0}}"),
    ("TensorWedge[{1, 0, 0}, {0, 1, 0}]", "{{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}}"),
    ("TensorWedge[{1, 2, 3}, {1, 2, 3}]", "{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}"), ("TensorWedge[2, {1, 2}]", "{2, 4}"),
    ("TensorWedge[a, b]", "TensorWedge[a, b]"), ("TensorWedge[{1, 2}, {1, 2, 3}]", Raises()),
    ("HodgeDual[{a, b, c}]", "{{0, c, -b}, {-c, 0, a}, {b, -a, 0}}"),
    ("HodgeDual[{{0, c, -b}, {-c, 0, a}, {b, -a, 0}}]", "{a, b, c}"), ("HodgeDual[x, 2]", "{{0, x}, {-x, 0}}"),
    ("HodgeDual[{1, 2}]", "{-2, 1}"),
    ("TensorExpand[TensorProduct[a + b, c]]", "TensorProduct[a, c] + TensorProduct[b, c]"),
    ("TensorExpand[TensorWedge[VectorSymbol[\"v\", 3], 2 VectorSymbol[\"w\", 3] + VectorSymbol[\"u\", 3]]]",
     terms_of("2 TensorWedge[VectorSymbol[\"v\", 3], VectorSymbol[\"w\", 3]] + "
              "TensorWedge[VectorSymbol[\"v\", 3], VectorSymbol[\"u\", 3]]")),
    ("TensorExpand[TensorWedge[VectorSymbol[\"v\", 3], 3 VectorSymbol[\"w\", 3]]]",
     "3 TensorWedge[VectorSymbol[\"v\", 3], VectorSymbol[\"w\", 3]]"),
    ("TensorReduce[TensorWedge[VectorSymbol[\"v\", 3], VectorSymbol[\"v\", 3]]]", "0"),
    ("TensorReduce[TensorWedge[VectorSymbol[\"w\", 3], VectorSymbol[\"v\", 3]]]",
     "-TensorWedge[VectorSymbol[\"v\", 3], VectorSymbol[\"w\", 3]]"),
    ("TensorReduce[TensorWedge[VectorSymbol[\"v\", 3], VectorSymbol[\"w\", 3]] + "
     "TensorWedge[VectorSymbol[\"w\", 3], VectorSymbol[\"v\", 3]]]", "0"),
    ("TensorReduce[{{1, 2}, {3, 4}}]", "{{1, 2}, {3, 4}}"),
]

# ── coordinates and curves ────────────────────────────────────────────────────────────────────

COORDS = [
    ("ToPolarCoordinates[{1, 1}]", "{Sqrt[2], Pi/4}"), ("ToPolarCoordinates[{x, y}]", "{Sqrt[x^2 + y^2], ArcTan[x, y]}"),
    ("ToPolarCoordinates[{0, -2}]", "{2, -Pi/2}"), ("ToPolarCoordinates[{1, 1, 1}]", "{Sqrt[3], ArcCos[1/Sqrt[3]], Pi/4}"),
    ("FromPolarCoordinates[{2, Pi/3}]", "{1, Sqrt[3]}"), ("FromPolarCoordinates[{r, t}]", "{r Cos[t], r Sin[t]}"),
    ("FromPolarCoordinates[{1, Pi/2, Pi/2}]", "{0, 0, 1}"),
    ("FromPolarCoordinates[ToPolarCoordinates[{3, -4}]]", "{3, -4}"),
    ("ToSphericalCoordinates[{1, 1, 1}]", "{Sqrt[3], ArcTan[Sqrt[2]], Pi/4}"),
    ("ToSphericalCoordinates[{0, 3, 0}]", "{3, Pi/2, Pi/2}"),
    ("ToSphericalCoordinates[{x, y, z}]", "{Sqrt[x^2 + y^2 + z^2], ArcTan[z, Sqrt[x^2 + y^2]], ArcTan[x, y]}"),
    ("FromSphericalCoordinates[{1, Pi/2, 0}]", "{1, 0, 0}"),
    ("FromSphericalCoordinates[{r, t, p}]", "{r Cos[p] Sin[t], r Sin[p] Sin[t], r Cos[t]}"),
    ("FromSphericalCoordinates[{2, Pi/3, Pi/4}]", "{Sqrt[6]/2, Sqrt[6]/2, 1}"),
    ("CoordinateTransform[\"Cartesian\" -> \"Polar\", {1, 1}]", "{Sqrt[2], Pi/4}"),
    ("CoordinateTransform[\"Polar\" -> \"Cartesian\", {r, t}]", "{r Cos[t], r Sin[t]}"),
    ("CoordinateTransform[\"Cartesian\" -> \"Cylindrical\", {1, 1, 5}]", "{Sqrt[2], Pi/4, 5}"),
    ("CoordinateTransform[\"Spherical\" -> \"Cartesian\", {1, Pi/2, Pi/2}]", "{0, 1, 0}"),
    ("CoordinateTransform[\"Foo\" -> \"Polar\", {1, 1}]", Raises()),
    ("TransformedField[\"Cartesian\" -> \"Polar\", x^2 + y^2, {x, y} -> {r, t}]", "r^2"),
    ("TransformedField[\"Polar\" -> \"Cartesian\", r^2, {r, t} -> {x, y}]", "x^2 + y^2"),
    ("TransformedField[\"Cartesian\" -> \"Polar\", {-y, x}, {x, y} -> {r, t}]", "{0, r}"),
    ("TransformedField[\"Cartesian\" -> \"Spherical\", x^2 + y^2 + z^2, {x, y, z} -> {r, t, p}]", "r^2"),
    ("FrenetSerretSystem[{Cos[t], Sin[t]}, t]", "{{1}, {{-Sin[t], Cos[t]}, {-Cos[t], -Sin[t]}}}"),
    ("FrenetSerretSystem[{Cos[t], Sin[t], t}, t]",
     "{{1/2, 1/2}, {{-Sin[t]/Sqrt[2], Cos[t]/Sqrt[2], 1/Sqrt[2]}, {-Cos[t], -Sin[t], 0}, "
     "{Sin[t]/Sqrt[2], -Cos[t]/Sqrt[2], 1/Sqrt[2]}}}"),
    ("FrenetSerretSystem[{t, t^2}, t][[1]]", "{2/(1 + 4 t^2)^(3/2)}"),
    ("FrenetSerretSystem[{t}, t]", Raises()),
]

# ── asymptotics ──────────────────────────────────────────────────────────────────────────────

ASYMPTOTIC = [
    ("Asymptotic[Sin[x]/x, x -> 0]", "1"), ("Asymptotic[Sin[x], {x, 0, 5}]", "x - x^3/6 + x^5/120"),
    ("Asymptotic[Sin[x], x -> 0, SeriesTermGoal -> 3]", "x - x^3/6 + x^5/120"),
    ("Asymptotic[x E^x, x -> 0, SeriesTermGoal -> 1]", "x"),
    ("Asymptotic[x E^x, x -> 0, SeriesTermGoal -> 4]", "x + x^2 + x^3/2 + x^4/6"),
    ("Asymptotic[Sin[x], x -> 0, SeriesTermGoal -> 0]", Raises()),
    ("AsymptoticSolve[y^2 == x + 1, {y, 1}, x -> 0, SeriesTermGoal -> 3]", "{{y -> 1 + x/2 - x^2/8}}"),
    ("AsymptoticDSolveValue[{y'[x] == y[x], y[0] == 1}, y[x], x -> 0, SeriesTermGoal -> 3]", "1 + x + x^2/2"),
    ("AsymptoticSum[1/k^2, {k, 1, n}, n -> Infinity, SeriesTermGoal -> 3]", "Pi^2/6 - 1/n + 1/(2 n^2)"),
    ("DiscreteAsymptotic[HarmonicNumber[n], n -> Infinity, SeriesTermGoal -> 2]", "Log[n] + EulerGamma"),
    ("Asymptotic[Exp[x], x -> 0, SeriesTermGoal -> 2]", "1 + x"),
    ("Asymptotic[1/(1 - x), x -> 0, SeriesTermGoal -> 4]", "1 + x + x^2 + x^3"),
    ("Asymptotic[(x^2 + 1)/(x + 3), x -> Infinity]", "x"), ("Asymptotic[Exp[x] + x^10, x -> Infinity]", "E^x"),
    ("Asymptotic[Gamma[x], x -> Infinity]", "Sqrt[2 Pi] E^(x (Log[x] - 1))/Sqrt[x]"),
    ("Asymptotic[Sqrt[x + 1] - Sqrt[x], x -> Infinity]", "1/(2 Sqrt[x])"),
    ("DiscreteAsymptotic[n!, n -> Infinity]", "Sqrt[2 Pi n] E^(n (Log[n] - 1))"),
    ("DiscreteAsymptotic[Fibonacci[n], n -> Infinity]", at_n("((1 + Sqrt[5])/2)^n/Sqrt[5]")),
    ("DiscreteAsymptotic[HarmonicNumber[n], n -> Infinity]", "Log[n]"),
    ("DiscreteAsymptotic[Binomial[2 n, n], n -> Infinity]", "4^n/Sqrt[Pi n]"),
    ("DiscreteLimit[(1 + 1/n)^n, n -> Infinity]", "E"), ("DiscreteLimit[(-1)^n, n -> Infinity]", "Indeterminate"),
    ("DiscreteLimit[n Sin[1/n], n -> Infinity]", "1"), ("DiscreteLimit[{1/n, (n + 1)/n}, n -> Infinity]", "{0, 1}"),
    ("DiscreteLimit[(-1)^n/n, n -> Infinity]", "0"), ("DiscreteLimit[Cos[2 Pi n], n -> Infinity]", "1"),
    ("DiscreteLimit[Sin[n], n -> Infinity]", "DiscreteLimit[Sin[n], n -> Infinity]"),
    ("DiscreteMaxLimit[(-1)^n, n -> Infinity]", "1"), ("DiscreteMaxLimit[Sin[n Pi/3], n -> Infinity]", "Sqrt[3]/2"),
    ("DiscreteMaxLimit[1/n, n -> Infinity]", "0"), ("DiscreteMaxLimit[Sin[n Pi/2] + Cos[1/n], n -> Infinity]", "2"),
    ("DiscreteMinLimit[(-1)^n, n -> Infinity]", "-1"), ("DiscreteMinLimit[Sin[n Pi/2] + 1/n, n -> Infinity]", "-1"),
    ("DiscreteMinLimit[(-1)^n n/(n + 1), n -> Infinity]", "-1"), ("DiscreteMinLimit[2 + 1/n, n -> Infinity]", "2"),
    ("AsymptoticLess[x, x^2, x -> Infinity]", True), ("AsymptoticLess[x^2, x, x -> Infinity]", False),
    ("AsymptoticLess[x^2, x, x -> 0]", True),
    ("AsymptoticLessEqual[2 x + 1, x, x -> Infinity]", True), ("AsymptoticLessEqual[x^2, x, x -> Infinity]", False),
    ("AsymptoticLessEqual[Sin[x], 1, x -> Infinity]", True),
    ("AsymptoticEqual[2 x + 1, x, x -> Infinity]", True), ("AsymptoticEqual[x^2, x, x -> Infinity]", False),
    ("AsymptoticEqual[Sin[x], x, x -> 0]", True),
    ("AsymptoticEquivalent[x + 1, x, x -> Infinity]", True), ("AsymptoticEquivalent[2 x, x, x -> Infinity]", False),
    ("AsymptoticEquivalent[Sin[x], x, x -> 0]", True),
    ("AsymptoticGreater[x^2, x, x -> Infinity]", True), ("AsymptoticGreater[x, x, x -> Infinity]", False),
    ("AsymptoticGreater[Exp[x], x^100, x -> Infinity]", True),
    ("AsymptoticGreaterEqual[x, 3 x, x -> Infinity]", True), ("AsymptoticGreaterEqual[x, x^2, x -> Infinity]", False),
    ("AsymptoticGreaterEqual[Log[x], 1, x -> Infinity]", True),
    ("AsymptoticSum[1/k, {k, 1, n}, n -> Infinity]", "Log[n]"), ("AsymptoticSum[k, {k, 1, n}, n -> Infinity]", "n^2/2"),
    ("AsymptoticSum[1/k^2, {k, 1, n}, {n, Infinity, 2}]", "Pi^2/6 - 1/n + 1/(2 n^2)"),
    ("AsymptoticSum[k^2, {k, 1, n}, n -> Infinity]", "n^3/3"),
    ("AsymptoticProduct[k, {k, 1, n}, n -> Infinity]", "Sqrt[2 Pi n] E^(n (Log[n] - 1))"),
    ("AsymptoticProduct[2, {k, 1, n}, n -> Infinity]", "2^n"),
    ("AsymptoticProduct[(k + 1)/k, {k, 1, n}, n -> Infinity]", "n"),
    ("AsymptoticIntegrate[1/(1 + x^2), x, x -> Infinity]", "Pi/2"),
    ("AsymptoticIntegrate[Cos[x], x, {x, 0, 5}]", "x - x^3/6 + x^5/120"),
    ("AsymptoticIntegrate[Exp[-a t^2], {t, 0, 1}, {a, 0, 2}]", "1 - a/3 + a^2/10"),
    ("AsymptoticSolve[y^2 == x + 1, {y, 1}, {x, 0, 3}]", "{{y -> 1 + x/2 - x^2/8 + x^3/16}}"),
    ("AsymptoticSolve[y^2 - x == 0, y, x -> Infinity]", "{{y -> -Sqrt[x]}, {y -> Sqrt[x]}}"),
    ("AsymptoticSolve[y + y^3 == x, {y, 0}, {x, 0, 5}]", "{{y -> x - x^3 + 3 x^5}}"),
    ("AsymptoticDSolveValue[{y'[x] == y[x], y[0] == 1}, y[x], {x, 0, 4}]", "1 + x + x^2/2 + x^3/6 + x^4/24"),
    ("AsymptoticDSolveValue[{y''[x] + y[x] == 0, y[0] == 0, y'[0] == 1}, y[x], {x, 0, 5}]", "x - x^3/6 + x^5/120"),
    ("AsymptoticDSolveValue[y'[x] == x y[x], y[x], {x, 0, 4}]", "C[1] (1 + x^2/2 + x^4/8)"),
    ("AsymptoticRSolveValue[{y[n + 1] == 2 y[n], y[1] == 1}, y[n], n -> Infinity]", "2^(n - 1)"),
    ("AsymptoticRSolveValue[{y[n + 2] == y[n + 1] + y[n], y[1] == 1, y[2] == 1}, y[n], n -> Infinity]",
     at_n("((1 + Sqrt[5])/2)^n/Sqrt[5]")),
    ("AsymptoticRSolveValue[{y[n + 1] == (n + 1) y[n], y[0] == 1}, y[n], n -> Infinity]",
     "Sqrt[2 Pi n] E^(n (Log[n] - 1))"),
    ("AsymptoticExpectation[Exp[x/a], Distributed[x, NormalDistribution[0, 1]], {a, Infinity, 2}]", "1 + 1/(2 a^2)"),
    ("AsymptoticExpectation[x^2, Distributed[x, ExponentialDistribution[a]], a -> Infinity]", "2/a^2"),
    ("AsymptoticExpectation[Exp[-a x], Distributed[x, ExponentialDistribution[1]], a -> Infinity]", "1/a"),
    ("AsymptoticProbability[x > a, Distributed[x, NormalDistribution[0, 1]], a -> Infinity]",
     "E^(-a^2/2)/(a Sqrt[2 Pi])"),
    ("AsymptoticProbability[x > a, Distributed[x, ExponentialDistribution[1]], a -> Infinity]", "E^-a"),
    ("AsymptoticProbability[x < a, Distributed[x, UniformDistribution[{0, 1}]], {a, 0, 2}]", "a"),
]

# ── discrete calculus, sequences, generating functions ────────────────────────────────────────

DISCRETE = [
    ("DifferenceQuotient[x^2, {x, h}]", "2 x + h"), ("DifferenceQuotient[f[x], {x, h}]", "(f[x + h] - f[x])/h"),
    ("DifferenceQuotient[x^3, {x, 2, h}]", "6 x + 6 h"), ("DifferenceQuotient[{x, x^2}, {x, 1}]", "{1, 1 + 2 x}"),
    ("Casoratian[{1, 2^n}, n]", "2^n"), ("Casoratian[{2^n, 3^n}, n]", "6^n"), ("Casoratian[{1, n, n^2}, n]", "2"),
    ("Casoratian[{2^n, 3^n}, n, True]", "{{2^n, 3^n}, {2^(n + 1), 3^(n + 1)}}"),
    ("DifferenceRoot[Function[{y, n}, {y[n + 2] == y[n + 1] + y[n], y[1] == 1, y[2] == 1}]][10]", "55"),
    ("DifferenceRoot[Function[{y, n}, {y[n + 1] == 2 y[n], y[0] == 3}]][5]", "96"),
    ("DifferenceRoot[Function[{y, n}, {y[n + 1] == (n + 1) y[n], y[0] == 1}]][6]", "720"),
    ("DifferentialRoot[Function[{y, x}, {y'[x] == y[x], y[0] == 1}]][0]", "1"),
    ("DifferentialRoot[Function[{y, x}, {y'[x] == y[x], y[0] == 1}]][1.]", math.e),
    ("DifferentialRoot[Function[{y, x}, {y''[x] == -y[x], y[0] == 0, y'[0] == 1}]][0.5]", math.sin(0.5)),
    ("DifferenceRootReduce[2^n, n][[0]][10]", "1024"),
    ("DifferenceRootReduce[Fibonacci[n], n] /. n -> 12", "144"),
    ("DifferenceRootReduce[n!, n] /. n -> 7", "5040"),
    ("DifferenceRootReduce[Fibonacci[2 n] == Fibonacci[n] LucasL[n], n]", True),
    ("DifferentialRootReduce[Exp[x], x] /. x -> 0", "1"),
    ("DifferentialRootReduce[Sin[x], x] /. x -> 0.3", math.sin(0.3)),
    ("DifferentialRootReduce[Exp[x^2], x] /. x -> 0.5", math.exp(0.25)),
    ("DifferentialRootReduce[Abs[x], x]", Raises()),
    ("FindSequenceFunction[{1, 4, 9, 16, 25}, n]", "n^2"), ("FindSequenceFunction[{1, 2, 6, 24, 120, 720}, n]", "n!"),
    ("FindSequenceFunction[{1, 1, 2, 3, 5, 8, 13, 21}, n]", "Fibonacci[n]"),
    ("FindSequenceFunction[{2, 4, 8, 16, 32}, n]", "2^n"),
    ("FindSequenceFunction[{1, 5, 2, 9, 1, 4, 7, 3, 3}, n]", "FindSequenceFunction[{1, 5, 2, 9, 1, 4, 7, 3, 3}, n]"),
    ("FindGeneratingFunction[{1, 1, 2, 3, 5, 8, 13, 21}, x]", "1/(1 - x - x^2)"),
    ("FindGeneratingFunction[{1, 2, 4, 8, 16, 32}, x]", "1/(1 - 2 x)"),
    ("FindGeneratingFunction[{1, 1, 1, 1, 1, 1}, x]", "1/(1 - x)"),
    ("ExponentialGeneratingFunction[1, n, x]", "E^x"), ("ExponentialGeneratingFunction[2^n, n, x]", "E^(2 x)"),
    ("ExponentialGeneratingFunction[n, n, x]", "x E^x"), ("ExponentialGeneratingFunction[n!, n, x]", "1/(1 - x)"),
    ("FactorialMomentGeneratingFunction[PoissonDistribution[m], t]", "E^(m (t - 1))"),
    ("FactorialMomentGeneratingFunction[BernoulliDistribution[p], t]", "1 - p + p t"),
    ("FactorialMomentGeneratingFunction[BinomialDistribution[n, p], t]", "(1 - p + p t)^n"),
    ("FactorialMomentGeneratingFunction[GeometricDistribution[p], t]", "p/(1 - (1 - p) t)"),
    ("TruncateSum[Inactive[Sum][1/k^2, {k, 1, Infinity}], 3]", "49/36"),
    ("TruncateSum[Inactive[Sum][x^k/k!, {k, 0, Infinity}], 3]", "1 + x + x^2/2"),
    ("TruncateSum[Inactive[Sum][a[k], {k, 1, Infinity}], m]", "Inactive[Sum][a[k], {k, 1, m}]"),
    ("TruncateSum[Sum[f[i], {i, 1, Infinity}], 3]", "f[1] + f[2] + f[3]"),
    ("TruncateSum[Inactive[Sum][1/k, {k, 1, Infinity}], -1]", Raises()),
]

# ── dynamical systems, matrix equations, eigenproblems, Green's functions ───────────────────────

DYNAMICS = [
    ("DFixedPoints[{x'[t] == x[t] (1 - x[t])}, x, t]", "{{0}, {1}}"),
    ("DFixedPoints[{x'[t] == y[t], y'[t] == -x[t] + x[t]^3}, {x, y}, t]", "{{-1, 0}, {0, 0}, {1, 0}}"),
    ("DFixedPoints[x''[t] + x[t] == 0, x, t]", "{{0, 0}}"),
    ("DStabilityConditions[{x'[t] == x[t] (1 - x[t])}, x, t]", "{{{0}, False}, {{1}, True}}"),
    ("DStabilityConditions[{x'[t] == a x[t]}, x, t]", "{{{0}, Re[a] < 0}}"),
    ("DStabilityConditions[{x'[t] == -x[t] + y[t], y'[t] == -2 y[t]}, {x, y}, t]", "{{{0, 0}, True}}"),
    ("RFixedPoints[{x[n + 1] == x[n]^2}, x, n]", "{{0}, {1}}"),
    ("RFixedPoints[{x[n + 1] == r x[n] (1 - x[n])}, x, n]", "{{0}, {(r - 1)/r}}"),
    ("RFixedPoints[{x[n + 1] == y[n], y[n + 1] == x[n]/2 + 1}, {x, y}, n]", "{{2, 2}}"),
    ("RStabilityConditions[{x[n + 1] == x[n]^2}, x, n]", "{{{0}, True}, {{1}, False}}"),
    ("RStabilityConditions[{x[n + 1] == a x[n]}, x, n]", "{{{0}, -1 < a < 1}}"),
    ("RStabilityConditions[{x[n + 1] == x[n]/2 + 1}, x, n]", "{{{2}, True}}"),
    ("LyapunovSolve[{{1, 0}, {0, 2}}, {{2, 0}, {0, 4}}]", "{{1, 0}, {0, 1}}"),
    ("LyapunovSolve[{{-1, 1}, {0, -2}}, {{1, 0}, {0, 1}}]", lyap_ok([[-1, 1], [0, -2]], [[1, 0], [0, 1]], "cont")),
    ("LyapunovSolve[{{1, 0}, {0, -1}}, {{0, 0}, {0, 0}}]", Raises()),
    ("LyapunovSolve[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{1, 0}, {0, 1}}]",
     Pred(lambda g: mat(g) * 0 == sp.zeros(2) and sp.Matrix([[1, 2], [3, 4]]) * mat(g) + mat(g) * sp.Matrix([[5, 6], [7, 8]])
          == sp.eye(2), "a.x + x.b == c")),
    ("LyapunovSolve[{{1., 2.}, {0., 3.}}, {{1., 0.}, {0., 1.}}]", lyap_ok([[1, 2], [0, 3]], [[1, 0], [0, 1]], "cont")),
    ("DiscreteLyapunovSolve[{{1/2, 0}, {0, 1/3}}, {{1, 0}, {0, 1}}]", "{{-4/3, 0}, {0, -9/8}}"),
    ("DiscreteLyapunovSolve[{{1, 0}, {0, 1}}, {{1, 0}, {0, 1}}]", Raises()),
    ("DiscreteLyapunovSolve[{{0, 1}, {0, 0}}, {{1, 0}, {0, 1}}]", lyap_ok([[0, 1], [0, 0]], [[1, 0], [0, 1]], "disc")),
    ("RiccatiSolve[{{{0, 1}, {0, 0}}, {{0}, {1}}}, {{{1, 0}, {0, 0}}, {{1}}}]", "{{Sqrt[2], 1}, {1, Sqrt[2]}}"),
    ("RiccatiSolve[{{{-1}}, {{1}}}, {{{1}}, {{1}}}]", "{{Sqrt[2] - 1}}"),
    ("RiccatiSolve[{{{1.}}, {{1.}}}, {{{1.}}, {{1.}}}]", lst_close([1 + math.sqrt(2)]) if SCIPY else Raises()),
    ("DiscreteRiccatiSolve[{{{1}}, {{1}}}, {{{1}}, {{1}}}]", "{{GoldenRatio}}"),
    ("DiscreteRiccatiSolve[{{{2}}, {{1}}}, {{{1}}, {{1}}}]", "{{2 + Sqrt[5]}}"),
    ("DiscreteRiccatiSolve[{{{1.}}, {{1.}}}, {{{1.}}, {{1.}}}]",
     lst_close([(1 + math.sqrt(5)) / 2]) if SCIPY else Raises()),
    ("DEigenvalues[{-u''[x], DirichletCondition[u[x] == 0, True]}, u[x], {x, 0, Pi}, 4]", "{1, 4, 9, 16}"),
    ("DEigenvalues[-u''[x], u[x], {x, 0, 1}, 3]", "{0, Pi^2, 4 Pi^2}"),
    ("DEigenvalues[{-u''[x], u[0] == 0, u'[1] == 0}, u[x], {x, 0, 1}, 2]", "{Pi^2/4, 9 Pi^2/4}"),
    ("DEigensystem[{-u''[x], DirichletCondition[u[x] == 0, True]}, u[x], {x, 0, Pi}, 2]", "{{1, 4}, {Sin[x], Sin[2 x]}}"),
    ("DEigensystem[{-u''[x], u[0] == 0, u[1] == 0}, u[x], {x, 0, 1}, 1]", "{{Pi^2}, {Sin[Pi x]}}"),
    ("DEigensystem[-u''[x], u[x], {x, 0, 2}, 2]", "{{0, Pi^2/4}, {1, Cos[Pi x/2]}}"),
    ("GreenFunction[{-u''[x], u[0] == 0, u[1] == 0}, u[x], {x, 0, 1}, y]",
     "x (1 - y) HeavisideTheta[y - x] + y (1 - x) HeavisideTheta[x - y]"),
    ("GreenFunction[{u'[x], u[0] == 0}, u[x], {x, 0, Infinity}, y]", "HeavisideTheta[x - y]"),
    ("GreenFunction[{u''[x] + u[x], u[0] == 0, u[Pi] == 0}, u[x], {x, 0, Pi}, y]", Raises()),
    ("GreenFunction[{u''[x], u[0] == 0, u'[0] == 0}, u, {x, 0, Infinity}, y]",
     S("Function[{x, y}, (x - y) HeavisideTheta[x - y]]")),
    ("C[1] + C[1]", "2 C[1]"), ("Head[C[1]]", "C"), ("C[2] - C[2]", "0"),
]

# ── options and inert symbols ─────────────────────────────────────────────────────────────────

OPTIONS = [
    ("SeriesTermGoal", "SeriesTermGoal"), ("Head[SeriesTermGoal]", "Symbol"),
    ("DependentVariables", "DependentVariables"), ("Head[DependentVariables]", "Symbol"),
    ("(DependentVariables -> {y})[[1]]", "DependentVariables"),
    ("DiscreteVariables", "DiscreteVariables"), ("Head[DiscreteVariables]", "Symbol"),
    ("(DiscreteVariables -> {a})[[2]]", "{a}"),
    ("IncludeConstantBasis", "IncludeConstantBasis"), ("Head[IncludeConstantBasis]", "Symbol"),
    ("(IncludeConstantBasis -> False)[[2]]", False),
    ("InterpolationOrder", "InterpolationOrder"),
    ("FunctionInterpolation[x^2, {x, 0, 2}, InterpolationOrder -> 1][1.5]", 2.25, 1e-6),
    ("Interpolation[{{0, 0}, {1, 1}, {2, 4}, {3, 9}}, InterpolationOrder -> 3][1.5]", 2.25),
    ("InterpolationPoints", "InterpolationPoints"),
    ("FunctionInterpolation[Exp[x], {x, 0, 1}, InterpolationPoints -> 5][0.5]", math.exp(0.5), 1e-6),
    ("FunctionInterpolation[Exp[x], {x, 0, 1}, InterpolationPoints -> 1]", Raises()),
    ("PrincipalValue", "PrincipalValue"), ("Integrate[1/x, {x, -1, 2}, PrincipalValue -> True]", "Log[2]"),
    ("ContourIntegrate[1/z, Element[z, Circle[]], PrincipalValue -> True]", Raises()),
    ("TargetStructure", "TargetStructure"),
]

# ── interpolation ────────────────────────────────────────────────────────────────────────────

INTERP = [
    ("Expand[InterpolatingPolynomial[{1, 4, 9, 16}, x]]", "x^2"),
    ("Expand[InterpolatingPolynomial[{{0, 1}, {1, 3}, {3, 2}}, x]]", "1 + 17 x/6 - 5 x^2/6"),
    ("InterpolatingPolynomial[{1, 4, 9}, x] /. x -> 5", "25"),
    ("Expand[InterpolatingPolynomial[{{0, 0, 1}, {1, 1, 1}}, x]]", "x"),
    ("Expand[InterpolatingPolynomial[{{0, {1, 0}}, {1, {0, 0}}}, x]]", "1 - 3 x^2 + 2 x^3"),
    ("Simplify[InterpolatingPolynomial[{{a, f}, {b, g}}, x] - (f + (g - f) (x - a)/(b - a))]", "0"),
    ("InterpolatingPolynomial[{{1, 2}, {1, 3}}, x]", Raises()),
    ("InterpolatingPolynomial[{{{1, 2}, 3}, {{2, 3}, 4}}, {x, y}]", Raises()),
    ("FunctionInterpolation[Sin[x], {x, 0, Pi}][1.]", math.sin(1.0), 1e-6),
    ("FunctionInterpolation[1/(1 + x^2), {x, -2, 2}][0.3]", 1 / 1.09, 1e-6),
    ("FunctionInterpolation[x, {x, 1, 0}]", Raises()),
    ("FunctionInterpolation[Sin[x] + Cos[2 x], {x, 0, 3}][2.5]", math.sin(2.5) + math.cos(5.0), 1e-6),
]

# ── residues, line, surface and contour integrals ────────────────────────────────────────────

_helix = quad1(lambda t: -mpmath.cos(t) ** 2 * mpmath.sin(t) ** 2 * t + 3 * mpmath.cos(t) ** 2 * mpmath.sin(t)
               + mpmath.sin(t) ** 2, 0, mpmath.pi / 2)

INTEGRALS = [
    ("ResidueSum[3 x^2/(x^3 - 1), x]", "3"), ("ResidueSum[x/(x^2 - 2) + (x - 2)/(x - 1)^2, x]", "2"),
    ("ResidueSum[x^3/(x^4 - 2), x]", "1"), ("ResidueSum[(1 - Cos[x])/x^2, x]", "0"),
    ("ResidueSum[Sin[x^2 + Cosh[x]] + E^x, x]", "0"), ("ResidueSum[E^x/x^3, x]", "1/2"),
    ("ResidueSum[1/((x - a) (x - b)), x]", "0"),
    ("ResidueSum[{Tan[x], Abs[x] < 100}, x]", "-64"),
    ("ResidueSum[{Log[x^2 + 3]/Sin[5 x], Abs[x] <= 1}, x]", "Log[3]/5 - 2 Log[3 + Pi^2/25]/5"),
    ("ResidueSum[{Log[x] + 1/(x - 1), Re[x] > 0}, x]", "1"),
    ("ResidueSum[{Log[x]/(x^2 - 2), Re[x] > 0}, x]", "Log[2]/(4 Sqrt[2])"),
    ("ResidueSum[{Tan[x^2 - 2 x], Abs[x] < 1}, x]", "1/(2 Sqrt[1 + Pi/2])"),
    ("ResidueSum[{1/(x^2 + 1), Im[x] > 0}, x]", "-I/2"), ("ResidueSum[{1/Sin[x], Abs[x] < 4}, x]", "-1"),
    ("ResidueSum[{1/(x - 1), Abs[x] < 1}, x]", "0"), ("ResidueSum[{1/(x - 1), Abs[x] <= 1}, x]", "1"),
    ("ResidueSum[{Gamma[x], Abs[x] < 5/2}, x]", "1/2"),
    ("ResidueSum[Log[x] + 1/(x - 1), x]", Raises()), ("ResidueSum[Gamma[x], x]", Raises()),
    ("LineIntegrate[2 x, Element[{x, y}, Circle[{0, 0}, 1, {0, Pi/2}]]]", "2"),
    ("LineIntegrate[{x y, x - y}, Element[{x, y}, Line[{{0, 0}, {1, 1}}]]]", "1/3"),
    ("LineIntegrate[{x^2 y z, 3 x y, y^2}, Element[{x, y, z}, ParametricRegion[{Cos[t], Sin[t], t}, {{t, 0, Pi/2}}]]]",
     _helix.real, 1e-9),
    ("LineIntegrate[{x + 2 y, x^2}, Element[{x, y}, Line[{{0, 0}, {2, 3/2}, {3, 0}}]]]", "3/2"),
    ("LineIntegrate[{z^3, x^3, y^3}, Element[{x, y, z}, Line[{{1, 1/2, 0}, {3, 2, 2}}]]]", "389/16"),
    ("LineIntegrate[1, Element[{x, y}, Circle[{0, 0}, 3]]]", "6 Pi"),
    ("LineIntegrate[{-y, x}, Element[{x, y}, Circle[]]]", "2 Pi"),
    ("LineIntegrate[Exp[-x], Element[{x, y}, HalfLine[{0, 0}, {1, 0}]]]", "1"),
    ("LineIntegrate[y^2, Element[{x, y}, ParametricRegion[{t^(5/2), t}, {{t, 0, 2}}]]]",
     quad1(lambda t: t ** 2 * mpmath.sqrt(1 + (2.5 * t ** 1.5) ** 2), 0, 2).real, 1e-9),
    ("LineIntegrate[x, Element[{x, y}, Disk[]]]", Raises()), ("LineIntegrate[{x}, Element[{x, y}, Circle[]]]", Raises()),
    ("NLineIntegrate[x^2 + y^2, Element[{x, y}, Circle[{1, 0}, 2]]]", 20 * math.pi, 1e-9),
    ("NLineIntegrate[{-y, x}, Element[{x, y}, Circle[]]]", 2 * math.pi, 1e-9),
    ("NLineIntegrate[Sqrt[1 + x^2 y], Element[{x, y}, Line[{{0, 0}, {1, 2}}]]]",
     quad1(lambda t: mpmath.sqrt(1 + 2 * t ** 3) * mpmath.sqrt(5), 0, 1).real, 1e-9),
    ("SurfaceIntegrate[x^2, Element[{x, y, z}, Sphere[]]]", "4 Pi/3"),
    ("SurfaceIntegrate[{x^2, x - y, 1}, Element[{x, y, z}, Sphere[]]]", "-4 Pi/3"),
    ("SurfaceIntegrate[x y z^2, Element[{x, y, z}, ParametricRegion[{1/2, u, v}, {{u, 0, 1}, {v, 0, 1}}]]]", "1/12"),
    ("SurfaceIntegrate[{y, -x, z}, Element[{x, y, z}, ParametricRegion[{1/2, u, v}, {{u, 0, 1}, {v, 0, 1}}]]]", "1/2"),
    ("SurfaceIntegrate[{-x, -y, z}, Element[{x, y, z}, ParametricRegion[{Cos[u], Sin[u], v}, {{u, 0, Pi/2}, {v, 0, 1}}]]]",
     "-Pi/2"),
    ("SurfaceIntegrate[{x, y, z}, Element[{x, y, z}, Cuboid[{0, 0, 0}, {1, 2, 3}]]]", "18"),
    ("SurfaceIntegrate[{x^2, y^2, z^2}, Element[{x, y, z}, Cube[]]]", "0"),
    ("SurfaceIntegrate[1, Element[{x, y, z}, Cuboid[{0, 0, 0}, {1, 2, 3}]]]", "22"),
    ("SurfaceIntegrate[1, Element[{x, y, z}, Triangle[{{0, 0, 0}, {1, 0, 0}, {0, 1, 1}}]]]", "Sqrt[2]/2"),
    ("SurfaceIntegrate[{x, y}, Element[{x, y}, Disk[]]]", "2 Pi"),
    ("SurfaceIntegrate[1, Element[{x, y, z}, Ball[{0, 0, 0}, 2]]]", "16 Pi"),
    ("SurfaceIntegrate[{x, y, z}, Element[{x, y, z}, Triangle[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}]]]", Raises()),
    ("NSurfaceIntegrate[x^2, Element[{x, y, z}, Sphere[{0, 0, 0}, 2]]]", 64 * math.pi / 3, 1e-9),
    ("NSurfaceIntegrate[{x, y, z}, Element[{x, y, z}, Ball[]]]", 4 * math.pi, 1e-9),
    ("NSurfaceIntegrate[Exp[z], Element[{x, y, z}, Cube[]]]",
     2 * (math.exp(0.5) + math.exp(-0.5)) / 2 * 1 + 4 * (math.exp(0.5) - math.exp(-0.5)), 1e-9),
    ("ContourIntegrate[1/z, Element[z, Circle[]]]", "2 Pi I"),
    ("ContourIntegrate[z^3 (1 - 3 z)/((3 z + 2) (1 + 2 z^4)), Element[z, Circle[{0, 0}, 2]]]", "-Pi I"),
    ("ContourIntegrate[1/((z - 1)^2 (z - I)), Element[z, Circle[{1, 0}, 1]]]", "Pi"),
    ("ContourIntegrate[z^2, Element[z, Line[{{0, 0}, {1, 1}}]]]", "(1 + I)^3/3"),
    ("ContourIntegrate[Conjugate[z], Element[z, Circle[]]]", "2 Pi I"),
    ("ContourIntegrate[1/z, Element[z, Circle[{0, 0}, 1, {0, Pi}]]]", "Pi I"),
    ("ContourIntegrate[Exp[z]/z^2, Element[z, Rectangle[{-1, -1}, {1, 1}]]]", "2 Pi I"),
    ("ContourIntegrate[1/z, Element[z, Line[{{1, 0}, {0, 1}, {-1, 0}}]]]", "Pi I"),
    ("ContourIntegrate[Exp[-z^2], Element[z, InfiniteLine[{0, 0}, {1, 0}]]]", "Sqrt[Pi]"),
    ("ContourIntegrate[Tan[z], Element[z, Circle[{0, 0}, 2]]]", "-4 Pi I"),
    ("ContourIntegrate[a z, Element[z, Line[{{0, 0}, {1, 0}}]]]", "a/2"),
    ("ContourIntegrate[1/(z - 1), Element[z, Circle[]]]", Raises()),
    ("NContourIntegrate[1/z, Element[z, Circle[]]]", 2j * math.pi, 1e-9),
    ("NContourIntegrate[z^2, Element[z, Line[{{0, 0}, {1, 1}}]]]", (1 + 1j) ** 3 / 3, 1e-9),
    ("NContourIntegrate[1/(z^2 + 4), Element[z, Triangle[{{-3, -1}, {3, -1}, {0, 3}}]]]", math.pi / 2, 1e-9),
]

# ── fractional calculus ──────────────────────────────────────────────────────────────────────

_rl_exp = 1 / math.sqrt(math.pi * 0.5) + math.exp(0.5) * math.erf(math.sqrt(0.5))

FRACTIONAL = [
    ("FractionalD[x^2, {x, 1/2}]", "8 x^(3/2)/(3 Sqrt[Pi])"), ("FractionalD[x^2, {x, a}]", "2 x^(2 - a)/Gamma[3 - a]"),
    ("FractionalD[5, {x, 1/2}]", "5/Sqrt[Pi x]"), ("FractionalD[x^3, {x, -2}]", "x^5/20"),
    ("FractionalD[{x, x^2}, {x, 1/2}]", "{2 Sqrt[x]/Sqrt[Pi], 8 x^(3/2)/(3 Sqrt[Pi])}"),
    ("FractionalD[Exp[x], {x, 1/2}]", "1/Sqrt[Pi x] + E^x Erf[Sqrt[x]]"),
    ("FractionalD[x^(-2), {x, 1/2}]", Raises()), ("FractionalD[x^4, {x, 2}]", "12 x^2"),
    ("FractionalD[FractionalD[x^3, {x, 1/2}], {x, 1/2}]", "3 x^2"),
    ("N[FractionalD[Sin[x], {x, -1/2}] /. x -> 0.7]",
     float(mpmath.quad(lambda s: (0.7 - s) ** -0.5 * mpmath.sin(s), [0, 0.7]) / mpmath.gamma(0.5)), 1e-9),
    ("CaputoD[x^2, {x, 1/2}]", "8 x^(3/2)/(3 Sqrt[Pi])"), ("CaputoD[7, {x, 1/3}]", "0"),
    ("CaputoD[x^2, {x, 5/2}]", "0"), ("CaputoD[Exp[x], {x, 1/2}]", "E^x Erf[Sqrt[x]]"),
    ("CaputoD[x^2, {x, a}] /. a -> 1/2", "8 x^(3/2)/(3 Sqrt[Pi])"), ("CaputoD[x^2, {x, a}] /. a -> 3", "0"),
    ("CaputoD[x^(1/2), {x, 2}]", "-1/(4 x^(3/2))"), ("CaputoD[x^(1/2), {x, 5/2}]", Raises()),
    ("NFractionalD[x^2, {x, 1/2}, 0.25]", 8 * 0.25 ** 1.5 / (3 * math.sqrt(math.pi)), 1e-9),
    ("NFractionalD[4, {x, 1/3}, 0.4]", 4 * 0.4 ** (-1 / 3) / math.gamma(2 / 3), 1e-9),
    ("NFractionalD[Sqrt[x], {x, 1/2}, 0.5]", math.gamma(1.5), 1e-7),
    ("NFractionalD[Exp[x], {x, 1/2}, 0.5]", _rl_exp, 1e-8),
    ("NFractionalD[Exp[x], {x, -3/2}, 0.25]",
     float(mpmath.quad(lambda s: (0.25 - s) ** 0.5 * mpmath.exp(s), [0, 0.25]) / mpmath.gamma(1.5)), 1e-9),
    ("NFractionalD[x^2, {x, 1/2}, -1]", Raises()),
    ("NCaputoD[x^2, {x, 1/2}, 0.25]", 8 * 0.25 ** 1.5 / (3 * math.sqrt(math.pi)), 1e-9),
    ("NCaputoD[Exp[x], {x, 1/2}, 0.5]", math.exp(0.5) * math.erf(math.sqrt(0.5)), 1e-8),
    ("NCaputoD[5, {x, 1/3}, 0.7]", 0.0, 1e-12),
    ("NCaputoD[Exp[x], {x, 3/2}, 0.5]",
     float(mpmath.quad(lambda s: (0.5 - s) ** -0.5 * mpmath.exp(s), [0, 0.5]) / mpmath.gamma(0.5)), 1e-9),
]

# ── changes of variables ────────────────────────────────────────────────────────────────────

CHANGE = [
    ("IntegrateChangeVariables[Inactive[Integrate][Exp[2 x], x], t, t == 2 x]", S("Inactive[Integrate][E^t/2, t]")),
    ("IntegrateChangeVariables[Inactive[Integrate][Sin[x] Cos[x], {x, 0, Pi/2}], t, t == Sin[x]]",
     S("Inactive[Integrate][t, {t, 0, 1}]")),
    ("IntegrateChangeVariables[Inactive[Integrate][Sin[3 x], x], u, x == u/3]", S("Inactive[Integrate][Sin[u]/3, u]")),
    ("IntegrateChangeVariables[Inactive[Integrate][Log[x], {x, 1, 2}], u, u == Log[x]]",
     S("Inactive[Integrate][u E^u, {u, 0, Log[2]}]")),
    ("IntegrateChangeVariables[Inactive[Integrate][Sin[x], {x, 0, Pi/2}], z, z == Tan[x/2]]",
     S("Inactive[Integrate][4 z/(1 + z^2)^2, {z, 0, 1}]")),
    ("IntegrateChangeVariables[Inactive[Integrate][Sqrt[1 - x^2], {x, 0, 1}], t, x == Sin[t]]",
     S("Inactive[Integrate][Cos[t]^2, {t, 0, Pi/2}]")),
    ("IntegrateChangeVariables[Inactive[Integrate][Exp[2 x], {x, 1, 3}], u, u == 2 x]",
     S("Inactive[Integrate][E^u/2, {u, 2, 6}]")),
    ("IntegrateChangeVariables[Inactive[Integrate][x, {x, -1, 1}], t, t == x^2]", Raises()),
    ("IntegrateChangeVariables[Inactive[Integrate][x^2 + y^2, {x, 0, 1}, {y, 0, 1}], {r, th}, \"Cartesian\" -> \"Polar\"]",
     Raises()),
    ("DSolveChangeVariables[Inactive[DSolve][E^x y'[x] + y[x] == 0, y, x], u, t, t == E^x]",
     S("Inactive[DSolve][t^2 u'[t] + u[t] == 0, u, t]")),
    ("DSolveChangeVariables[Inactive[DSolve][Log[x] y'[x] - y[x] == 0, y[x], x], u, t, t == Log[x]]",
     S("Inactive[DSolve][t E^-t u'[t] - u[t] == 0, u[t], t]")),
    ("DSolveChangeVariables[Inactive[DSolve][x^2 y''[x] + p x y'[x] + q y[x] == 0, y, x], y, t, x == E^t]",
     S("Inactive[DSolve][y''[t] + (p - 1) y'[t] + q y[t] == 0, y, t]")),
    ("DSolveChangeVariables[Inactive[DSolve][y''[x] == 0, y[x], x], u, t, t == Cos[x]]",
     S("Inactive[DSolve][(1 - t^2) u''[t] - t u'[t] == 0, u[t], t]")),
    ("DSolveChangeVariables[Inactive[DSolve][x^2 y''[x] - (2 x Tan[x] - 1) x y'[x] - (x Tan[x] + a) y[x] == 0, y, x], "
     "u, x, y[x] == u[x]/Cos[x]]", S("Inactive[DSolve][(x^2 u''[x] + x u'[x] + (x^2 - a) u[x])/Cos[x] == 0, u, x]")),
    ("DSolveChangeVariables[Inactive[DSolve][D[u[x, t], t] == D[u[x, t], x, x], u, {x, t}], v, {a, b}, "
     "{a == x - t, b == x + t}]", Raises()),
    ("DSolveChangeVariables[Inactive[DSolve][y'[x] == y[x], y, x], u, t, \"Cartesian\" -> \"Polar\"]", Raises()),
]


# Integrate GenerateConditions -> True keeps the genericity condition of an indefinite integral (n != -1);
# assumptions that rule the generic case out keep the special case instead of a silently wrong antiderivative.
GENCOND = [
    ("Integrate[x^n, x, GenerateConditions -> True]", S("ConditionalExpression[x^(1 + n)/(1 + n), n != -1]")),
    ("Integrate[x^n, x]", S("x^(1 + n)/(1 + n)")),
    ("Integrate[x^n, x, GenerateConditions -> True, Assumptions -> n > 0]", S("x^(1 + n)/(1 + n)")),
    ("Integrate[Exp[a x], x, GenerateConditions -> True]", S("ConditionalExpression[E^(a x)/a, a != 0]")),
    ("Integrate[x^n, x, Assumptions -> n == -1]", S("Piecewise[{{x^(1 + n)/(1 + n), n != -1}}, Log[x]]")),
]


CASES = ARRAYS + COORDS + ASYMPTOTIC + DISCRETE + DYNAMICS + OPTIONS + INTERP + INTEGRALS + FRACTIONAL + CHANGE + GENCOND


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w06(case):
    check(case)
