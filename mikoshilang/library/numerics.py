"""Classical numerical methods (floating-point results by design).

Legacy versions returned Error[...] expressions instead of raising, silently
bumped odd Simpson panel counts, and GaussQuadrature ignored n (it called
SciPy's adaptive quad).  These are pure Python and need no NumPy/SciPy."""
import math

import sympy as sp

from ..expr import Expr, Symbol
from .core import Unevaluated, fn, error, record, S, R, L, rule, number, sym, variable, pos_int, list_arg, to_float

AREA = "numerics"


def _f1(f, x, fname):
    e = sym(f, fname)
    v = variable(x, fname)
    extra = e.free_symbols - {v}
    if extra:
        raise ValueError(f"{fname}: expression has free symbols {sorted(map(str, extra))} besides {v}")
    g = sp.lambdify(v, e, modules="mpmath")

    def call(t):
        r = complex(g(t))
        if r.imag != 0 or math.isnan(r.real):
            raise ValueError(f"{fname}: f({t}) = {r} is not real")
        return r.real
    return e, v, call


def _f2(f, vars_, fname):
    vs = list_arg(vars_, fname, "variable list {x, y}")
    if len(vs) != 2:
        raise ValueError(f"{fname}: variables must be {{x, y}}")
    x, y = variable(vs[0], fname), variable(vs[1], fname)
    e = sym(f, fname)
    if e.free_symbols - {x, y}:
        raise ValueError(f"{fname}: f(x, y) has extra free symbols")
    g = sp.lambdify((x, y), e, modules="mpmath")
    return lambda a, b: float(complex(g(a, b)).real)


def _fl(v, fname, what):
    return to_float(number(v, fname, what))


def _tol(t, fname):
    t = _fl(t, fname, "tolerance")
    if t <= 0:
        raise ValueError(f"{fname}: tolerance must be positive")
    return t


@fn("Bisection", area=AREA, status="fixed",
    notes="Bisection[f, x, a, b, tol=1e-6] root of f in [a, b]; raises unless f(a), f(b) bracket a root "
          "(legacy returned an Error[...] expression)")
def _bisect(f, x, a, b, tol=1e-6):
    _, _, g = _f1(f, x, "Bisection")
    a, b, tol = _fl(a, "Bisection", "a"), _fl(b, "Bisection", "b"), _tol(tol, "Bisection")
    fa, fb = g(a), g(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("Bisection: f(a) and f(b) must have opposite signs")
    for _ in range(2000):
        c = (a + b) / 2
        fc = g(c)
        if fc == 0 or (b - a) / 2 < tol:
            return c
        if fa * fc < 0:
            b = c
        else:
            a, fa = c, fc
    return (a + b) / 2


@fn("NewtonRaphson", area=AREA, status="fixed",
    notes="NewtonRaphson[f, x, x0, tol=1e-10, maxIter=100] with the exact derivative; raises on zero "
          "derivative or non-convergence (legacy returned Error[...] expressions)")
def _newton(f, x, x0, tol=1e-10, maxit=100):
    e, v, g = _f1(f, x, "NewtonRaphson")
    _, _, dg = _f1(R(sp.diff(e, v)), x, "NewtonRaphson")
    t, tol, n = _fl(x0, "NewtonRaphson", "x0"), _tol(tol, "NewtonRaphson"), pos_int(maxit, "NewtonRaphson", "maxIter")
    for _ in range(n):
        ft = g(t)
        d = dg(t)
        if d == 0:
            raise ValueError(f"NewtonRaphson: derivative vanished at x = {t}")
        nt = t - ft / d
        if abs(nt - t) <= tol * max(1.0, abs(nt)):
            return nt
        t = nt
    raise ValueError(f"NewtonRaphson: did not converge in {n} iterations")


@fn("SecantMethod", area=AREA, status="fixed",
    notes="SecantMethod[f, x, x0, x1, tol=1e-10, maxIter=100]; raises on failure (legacy returned Error[...])")
def _secant(f, x, x0, x1, tol=1e-10, maxit=100):
    _, _, g = _f1(f, x, "SecantMethod")
    a, b = _fl(x0, "SecantMethod", "x0"), _fl(x1, "SecantMethod", "x1")
    tol, n = _tol(tol, "SecantMethod"), pos_int(maxit, "SecantMethod", "maxIter")
    fa, fb = g(a), g(b)
    for _ in range(n):
        if fb == fa:
            if fb == 0:
                return b
            raise ValueError("SecantMethod: f(x0) == f(x1); secant is horizontal")
        c = b - fb * (b - a) / (fb - fa)
        if abs(c - b) <= tol * max(1.0, abs(c)):
            return c
        a, fa, b, fb = b, fb, c, g(c)
    raise ValueError(f"SecantMethod: did not converge in {n} iterations")


def _ode(name, step):
    @fn(name, area=AREA, status="verified",
        notes=f"{name}[f, {{x, y}}, x0, y0, xEnd, n=100] for y' = f(x, y): y(xEnd) with n fixed steps")
    def impl(f, vars_, x0, y0, xend, n=100):
        g = _f2(f, vars_, name)
        x, y = _fl(x0, name, "x0"), _fl(y0, name, "y0")
        n = pos_int(n, name, "number of steps")
        h = (_fl(xend, name, "xEnd") - x) / n
        for _ in range(n):
            y = step(g, x, y, h)
            x += h
        return y
    return impl


_ode("EulerMethod", lambda g, x, y, h: y + h * g(x, y))


def _rk4(g, x, y, h):
    k1 = g(x, y)
    k2 = g(x + h / 2, y + h * k1 / 2)
    k3 = g(x + h / 2, y + h * k2 / 2)
    k4 = g(x + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


_ode("RungeKutta4", _rk4)


@fn("TrapezoidalRule", area=AREA, status="verified", notes="composite trapezoid rule with n panels (default 100)")
def _trap(f, x, a, b, n=100):
    _, _, g = _f1(f, x, "TrapezoidalRule")
    a, b, n = _fl(a, "TrapezoidalRule", "a"), _fl(b, "TrapezoidalRule", "b"), pos_int(n, "TrapezoidalRule", "n")
    h = (b - a) / n
    return h * (g(a) / 2 + sum(g(a + i * h) for i in range(1, n)) + g(b) / 2)


@fn("SimpsonsRule", area=AREA, status="fixed",
    notes="composite Simpson rule; n (default 100) must be even (legacy silently used n+1)")
def _simpson(f, x, a, b, n=100):
    _, _, g = _f1(f, x, "SimpsonsRule")
    a, b, n = _fl(a, "SimpsonsRule", "a"), _fl(b, "SimpsonsRule", "b"), pos_int(n, "SimpsonsRule", "n")
    if n % 2:
        raise ValueError("SimpsonsRule: the number of panels n must be even")
    h = (b - a) / n
    s = g(a) + g(b) + sum((4 if i % 2 else 2) * g(a + i * h) for i in range(1, n))
    return h / 3 * s


@fn("GaussQuadrature", area=AREA, status="fixed",
    notes="n-point Gauss-Legendre rule (default n = 5); legacy ignored n and used adaptive quadrature")
def _gauss(f, x, a, b, n=5):
    _, _, g = _f1(f, x, "GaussQuadrature")
    a, b, n = _fl(a, "GaussQuadrature", "a"), _fl(b, "GaussQuadrature", "b"), pos_int(n, "GaussQuadrature", "n")
    if n > 200:
        raise ValueError("GaussQuadrature: n must be at most 200")
    from sympy.integrals.quadrature import gauss_legendre
    xs, ws = gauss_legendre(n, 20)
    m, r = (b + a) / 2, (b - a) / 2
    return r * sum(float(w) * g(m + r * float(t)) for t, w in zip(xs, ws))


@fn("MonteCarloIntegration", area=AREA, status="verified",
    notes="plain Monte Carlo estimate with n uniform samples (default 10000); random, reproducible after "
          "SeedRandom[s]; standard error ~ (b-a)·σ/√n")
def _mc(f, x, a, b, n=10000):
    from .statistics import _RNG
    _, _, g = _f1(f, x, "MonteCarloIntegration")
    a, b = _fl(a, "MonteCarloIntegration", "a"), _fl(b, "MonteCarloIntegration", "b")
    n = pos_int(n, "MonteCarloIntegration", "n")
    return (b - a) * sum(g(_RNG.uniform(a, b)) for _ in range(n)) / n
