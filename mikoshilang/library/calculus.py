"""Calculus: integration, sums, series, vector calculus, optimisation, ODE helpers."""

from __future__ import annotations
import sympy as sp

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, error, record, S, R, L, rule, numeric_out, sym, number, is_list, int_arg,
                   nonneg_int, list_arg, variable, has_float, matrix_arg, mat_out)

AREA = "calculus"


def _check_result(v, fname):
    if v is sp.nan or (isinstance(v, sp.Basic) and v.has(sp.nan)):
        raise ValueError(f"{fname}: the result is indeterminate (the integral/sum does not converge)")
    return v


def _int_spec(spec, fname):
    if isinstance(spec, Symbol):
        return variable(spec, fname)
    if is_list(spec) and len(spec.args) == 3:
        return (variable(spec.args[0], fname), sym(spec.args[1], fname), sym(spec.args[2], fname))
    raise ValueError(f"{fname}: each variable specification must be x or {{x, a, b}}")


@fn("Integrate", area=AREA, status="fixed",
    notes="indefinite (no constant) and definite integrals; several variables as Wolfram (the last "
          "specification is innermost, so its bounds may depend on earlier variables); Assumptions -> a option "
          "(default $Assumptions). Results that hold only under a condition are returned as Wolfram "
          "ConditionalExpression[value, cond] (Integrate[Exp[-a x], {x, 0, Infinity}] -> "
          "ConditionalExpression[1/a, Abs[Arg[a]] < Pi/2]); under assumptions that imply the condition the plain "
          "value is returned, and assumptions that contradict it raise (divergent). Previously such integrals hit "
          "the recursion limit. Indeterminate results raise; integrals SymPy cannot do stay as Integrate[...]")
def _integrate(f, *specs):
    from .assume import split_option, effective
    specs, opt = split_option(specs, "Integrate")
    if not specs:
        raise ValueError("Integrate needs at least one variable specification")
    A = effective([], opt, "Integrate")
    v = A.apply(sym(f, "Integrate"))
    conds = []
    for spec in reversed(specs):            # Wolfram: Integrate[f, x, y] integrates over y first
        s_ = _int_spec(spec, "Integrate")
        s_ = A.apply(s_) if isinstance(s_, sp.Basic) else tuple(A.apply(t) for t in s_)
        v = sp.integrate(v, s_)
        if isinstance(v, sp.Piecewise):
            branches = [(e, c) for e, c in v.args if c is not sp.true]
            default = [e for e, c in v.args if c is sp.true]
            if default and default[0].has(sp.Integral) and len(branches) == 1:
                e, c = branches[0]
                c = A.refine(c)
                if c is sp.false:
                    raise ValueError("Integrate: the integral does not converge under the given assumptions")
                if c is not sp.true:
                    conds.append(c)
                v = e
    v = _check_result(v, "Integrate")
    if A.subs and isinstance(v, sp.Basic):
        v = A.refine(sp.simplify(v)) if v.has(sp.Abs, sp.sign, sp.arg, sp.Piecewise) else v
    v = A.unapply(v)
    if conds:
        from .assume import from_sym
        cond = A.unapply(sp.And(*conds))
        return Expr("ConditionalExpression", numeric_out(v, f, *specs), from_sym(cond))
    return numeric_out(v, f, *specs)


@fn("DefiniteIntegral", area=AREA, status="fixed",
    notes="definite integral DefiniteIntegral[f, {x, a, b}] (MikoshiLang); legacy silently computed an "
          "indefinite integral for a bare variable, now that raises"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _defint(f: Expr, spec: Expr) -> Expr:
    if not (is_list(spec) and len(spec.args) == 3):
        raise ValueError("DefiniteIntegral: second argument must be {x, a, b}")
    return _integrate(f, spec)


def _iter_spec(spec, fname):
    if not is_list(spec):
        raise ValueError(f"{fname}: iterator must be {{i, imax}}, {{i, imin, imax}}, {{i, imin, imax, di}} "
                         "or {i, {values}}")
    a = spec.args
    i = variable(a[0], fname)
    if len(a) == 2 and is_list(a[1]):
        return i, ("list", [sym(x, fname) for x in a[1].args])
    if len(a) == 2:
        return i, ("range", sp.Integer(1), sym(a[1], fname), None)
    if len(a) == 3:
        return i, ("range", sym(a[1], fname), sym(a[2], fname), None)
    if len(a) == 4:
        return i, ("range", sym(a[1], fname), sym(a[2], fname), sym(a[3], fname))
    raise ValueError(f"{fname}: invalid iterator")


def _sumprod(f, specs, fname, op):
    v = sym(f, fname)
    for spec in reversed(specs):              # first iterator is outermost
        i, it = _iter_spec(spec, fname)
        if it[0] == "list":
            terms = [v.subs(i, x) for x in it[1]]
            v = sp.Add(*terms) if op == "sum" else sp.Mul(*terms)
            continue
        lo, hi, step = it[1], it[2], it[3]
        if step is not None:
            if not (lo.is_number and hi.is_number and step.is_number) or step == 0:
                raise NotImplementedError(f"{fname}: a step requires numeric bounds")
            n = sp.floor((hi - lo) / step)
            if n < 0:
                v = sp.Integer(0 if op == "sum" else 1)
                continue
            terms = [v.subs(i, lo + k * step) for k in range(int(n) + 1)]
            v = sp.Add(*terms) if op == "sum" else sp.Mul(*terms)
            continue
        if lo.is_number and hi.is_number and hi.is_finite and lo.is_finite:
            # Wolfram: i runs lo, lo+1, ... <= hi (bounds need not be integers)
            n = sp.floor(hi - lo)
            if n < 0:
                v = sp.Integer(0 if op == "sum" else 1)
                continue
            if n <= 2000:
                terms = [v.subs(i, lo + k) for k in range(int(n) + 1)]
                v = sp.Add(*terms) if op == "sum" else sp.Mul(*terms)
                continue
        v = (sp.summation if op == "sum" else sp.product)(v, (i, lo, hi))
        v = _check_result(v, fname)
    if isinstance(v, sp.Basic):
        v = sp.simplify(v) if v.free_symbols else v
    return v


@fn("Sum", area=AREA, status="fixed",
    notes="added {i, n}, {i, a, b, step} and {i, {values}} iterators and multiple iterators (first outermost); "
          "sums divergent to +-Infinity return that limit, other non-convergent sums stay unevaluated, "
          "indeterminate (nan) results raise")
def _sum(f, *specs):
    if not specs:
        raise ValueError("Sum needs an iterator")
    return numeric_out(_sumprod(f, specs, "Sum", "sum"), f, *specs)


@fn("Product", area=AREA, status="fixed",
    notes="added {i, n}, {i, a, b, step} and {i, {values}} iterators and multiple iterators")
def _product(f, *specs):
    if not specs:
        raise ValueError("Product needs an iterator")
    return numeric_out(_sumprod(f, specs, "Product", "product"), f, *specs)


def _series_spec(spec, fname, default_n=5):
    items = list_arg(spec, fname, "series specification", min_len=1)
    x = variable(items[0], fname)
    a = sym(items[1], fname) if len(items) > 1 else sp.Integer(0)
    n = nonneg_int(items[2], fname, "order") if len(items) > 2 else default_n
    if len(items) > 3:
        raise ValueError(f"{fname}: specification must be {{x}}, {{x, a}} or {{x, a, n}}")
    return x, a, n


def _taylor(f, x, a, n, fname):
    v = sym(f, fname)
    s = sp.series(v, x, a, n + 1)
    if s.has(sp.Order):
        s = s.removeO()
    return R(s)


@fn("TaylorSeries", area=AREA, status="fixed",
    notes="TaylorSeries[f, {x, a, n}] keeps terms up to (x - a)^n (legacy stopped at (x - a)^(n - 1)); "
          "default n = 5"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _taylorseries(f: Expr, spec: ListOf[Expr]) -> Expr:
    x, a, n = _series_spec(spec, "TaylorSeries")
    return _taylor(f, x, a, n, "TaylorSeries")


@fn("MaClaurinSeries", area=AREA, status="fixed",
    notes="MaClaurinSeries[f, {x, n}] about 0 up to x^n (legacy used the point slot and an exclusive order); "
          "the 3-element form {x, 0, n} is accepted, a non-zero centre raises"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _maclaurin(f: Expr, spec: ListOf[Expr]) -> Expr:
    items = list_arg(spec, "MaClaurinSeries", min_len=1)
    x = variable(items[0], "MaClaurinSeries")
    if len(items) == 1:
        n = 5
    elif len(items) == 2:
        n = nonneg_int(items[1], "MaClaurinSeries", "order")
    elif len(items) == 3:
        if sym(items[1], "MaClaurinSeries") != 0:
            raise ValueError("MaClaurinSeries: the expansion point must be 0 (use TaylorSeries)")
        n = nonneg_int(items[2], "MaClaurinSeries", "order")
    else:
        raise ValueError("MaClaurinSeries: specification must be {x}, {x, n} or {x, 0, n}")
    return _taylor(f, x, sp.Integer(0), n, "MaClaurinSeries")


@fn("TaylorPolynomial", area=AREA, status="verified",
    notes="TaylorPolynomial[f, x, a, n]: Taylor polynomial of degree n (MikoshiLang)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _taylorpoly(f: Expr, x: Var, a: Expr, n: NonNegInteger) -> Expr:
    return _taylor(f, variable(x, "TaylorPolynomial"), sym(a, "TaylorPolynomial"),
                   nonneg_int(n, "TaylorPolynomial", "order"), "TaylorPolynomial")


@fn("LaurentSeries", area=AREA, status="fixed",
    notes="LaurentSeries[f, x, a, n]: Laurent expansion up to (x - a)^n without the O-term "
          "(legacy kept an O-term and was exclusive in n)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _laurent(f: Expr, x: Var, a: Expr, n: Integer) -> Expr:
    v = sym(f, "LaurentSeries")
    xv = variable(x, "LaurentSeries")
    k = int_arg(n, "LaurentSeries", "order")
    s = sp.series(v, xv, sym(a, "LaurentSeries"), k + 1)
    return R(s.removeO() if s.has(sp.Order) else s)


@fn("NthDerivative", area=AREA, status="fixed",
    notes="NthDerivative[f, x, n] = D[f, {x, n}]; n must be a non-negative integer (legacy truncated reals)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _nthd(f: Expr, x: Var, n: NonNegInteger) -> Expr:
    return R(sp.diff(sym(f, "NthDerivative"), variable(x, "NthDerivative"), nonneg_int(n, "NthDerivative", "order")))


def _residue(f, x, a, fname="Residue"):
    return R(sp.residue(sym(f, fname), variable(x, fname), sym(a, fname)))


@fn("Residue", area=AREA, status="verified", notes="Residue[f, {x, a}] (Wolfram) and Residue[f, x, a]"
                                                   ". Check: exact (exact input gives exact results), reference "
                                                   "(differential cases in tests/library).")
def _res(f: Expr, spec: Expr, a: Expr = None) -> Expr:
    if a is None:
        items = list_arg(spec, "Residue", min_len=2)
        if len(items) != 2:
            raise ValueError("Residue: specification must be {x, a}")
        return _residue(f, items[0], items[1])
    return _residue(f, spec, a)


@fn("ArcLength", area=AREA, status="verified",
    notes="ArcLength[f, {x, a, b}] = Integrate[Sqrt[1 + f'^2], {x, a, b}] (MikoshiLang curve form)")
def _arclength(f, spec):
    if not (is_list(spec) and len(spec.args) == 3):
        raise ValueError("ArcLength: second argument must be {x, a, b}")
    x, a, b = _int_spec(spec, "ArcLength")
    v = sym(f, "ArcLength")
    integrand = sp.simplify(sp.sqrt(1 + sp.diff(v, x) ** 2))
    return numeric_out(_check_result(sp.integrate(integrand, (x, a, b)), "ArcLength"), f, spec)


def _same_len(F, V, fname):
    if len(F) != len(V):
        raise ValueError(f"{fname}: the field and the variable list must have the same length")


@fn("Gradient", area=AREA, status="verified", notes="list of partial derivatives (MikoshiLang; Wolfram Grad)"
                                                    ". Check: exact (exact input gives exact results), reference "
                                                    "(differential cases in tests/library).")
def _gradient(f: Expr, vs: ListOf[Var]) -> ListOf[Expr]:
    v = sym(f, "Gradient")
    return L(*(R(sp.diff(v, x)) for x in _vars_list(vs, "Gradient")))


@fn("Grad", area=AREA, status="promoted", notes="Grad[f, {x1, ..., xn}] (Cartesian); Grad of a vector field gives the Jacobian")
def _grad(f, vs):
    xs = _vars_list(vs, "Grad")
    if is_list(f):
        return L(*(L(*(R(sp.diff(sym(c, "Grad"), x)) for x in xs)) for c in f.args))
    v = sym(f, "Grad")
    return L(*(R(sp.diff(v, x)) for x in xs))


def _vars_list(vs, fname):
    return [variable(x, fname) for x in list_arg(vs, fname, "variable list", min_len=1)]


def _div(F, vs, fname):
    comps = [sym(c, fname) for c in list_arg(F, fname, "vector field")]
    xs = _vars_list(vs, fname)
    _same_len(comps, xs, fname)
    return R(sp.Add(*(sp.diff(c, x) for c, x in zip(comps, xs))))


@fn("Divergence", area=AREA, status="fixed",
    notes="sum of dF_i/dx_i (MikoshiLang; Wolfram Div); mismatched lengths now raise (legacy zip-truncated)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _divergence(F: ListOf[Expr], vs: ListOf[Var]) -> Expr:
    return _div(F, vs, "Divergence")


@fn("Div", area=AREA, status="promoted", notes="Div[{f1, ..., fn}, {x1, ..., xn}] (Cartesian)")
def _divw(F, vs):
    return _div(F, vs, "Div")


@fn("Laplacian", area=AREA, status="promoted", notes="Laplacian[f, {x1, ..., xn}] (Cartesian)")
def _laplacian(f, vs):
    v = sym(f, "Laplacian")
    return R(sp.Add(*(sp.diff(v, x, 2) for x in _vars_list(vs, "Laplacian"))))


@fn("Curl", area=AREA, status="fixed",
    notes="3D curl; the 2D form Curl[{f1, f2}, {x, y}] = df2/dx - df1/dy added (Wolfram); other dimensions "
          "raise instead of returning an Error expression")
def _curl(F, vs):
    c = [sym(x, "Curl") for x in list_arg(F, "Curl", "vector field")]
    xs = _vars_list(vs, "Curl")
    _same_len(c, xs, "Curl")
    if len(c) == 2:
        return R(sp.diff(c[1], xs[0]) - sp.diff(c[0], xs[1]))
    if len(c) != 3:
        raise ValueError("Curl: only 2- and 3-dimensional fields are supported")
    return L(R(sp.diff(c[2], xs[1]) - sp.diff(c[1], xs[2])),
             R(sp.diff(c[0], xs[2]) - sp.diff(c[2], xs[0])),
             R(sp.diff(c[1], xs[0]) - sp.diff(c[0], xs[1])))


@fn("HessianMatrix", area=AREA, status="verified", notes="matrix of second partial derivatives"
                                                         ". Check: exact (exact input gives exact results), "
                                                         "reference (differential cases in tests/library).")
def _hessian(f: Expr, vs: ListOf[Var]) -> Expr:
    return mat_out(sp.hessian(sym(f, "HessianMatrix"), _vars_list(vs, "HessianMatrix")))


@fn("JacobianMatrix", area=AREA, status="verified", notes="matrix dF_i/dx_j"
                                                          ". Check: exact (exact input gives exact results), "
                                                          "reference (differential cases in tests/library).")
def _jacobian(F: ListOf[Expr], vs: ListOf[Var]) -> Expr:
    comps = [sym(c, "JacobianMatrix") for c in list_arg(F, "JacobianMatrix", "function list", min_len=1)]
    return mat_out(sp.Matrix(comps).jacobian(_vars_list(vs, "JacobianMatrix")))


@fn("DirectionalDerivative", area=AREA, status="fixed",
    notes="grad f . u/|u| with an exact unit vector; legacy converted the direction to floats and did not "
          "normalise it (so it returned grad f . u)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _dirder(f: Expr, vs: ListOf[Var], u: Expr) -> Expr:
    v = sym(f, "DirectionalDerivative")
    xs = _vars_list(vs, "DirectionalDerivative")
    d = [sym(c, "DirectionalDerivative") for c in list_arg(u, "DirectionalDerivative", "direction")]
    _same_len(d, xs, "DirectionalDerivative")
    norm = sp.sqrt(sp.Add(*(c ** 2 for c in d)))
    if norm == 0:
        raise ValueError("DirectionalDerivative: direction must be non-zero")
    out = sp.Add(*(sp.diff(v, x) * c for x, c in zip(xs, d))) / norm
    return numeric_out(sp.simplify(out), u)


def _solutions(eqs, xs, fname):
    sols = sp.solve(eqs, xs, dict=True)
    return L(*(L(*(rule(Symbol(str(x)), R(s[x])) for x in xs if x in s)) for s in sols))


@fn("CriticalPoints", area=AREA, status="fixed",
    notes="solutions of grad f = 0 as rule lists (real and complex, as Solve); legacy silently dropped "
          "solutions SymPy returned as tuples (most multivariate cases)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _critical(f: Expr, vs: Expr) -> Expr:
    xs = [variable(vs, "CriticalPoints")] if isinstance(vs, Symbol) else _vars_list(vs, "CriticalPoints")
    v = sym(f, "CriticalPoints")
    return _solutions([sp.diff(v, x) for x in xs], xs, "CriticalPoints")


@fn("PhasePlane", area=AREA, status="fixed",
    notes="equilibrium points of x' = F(x) as rule lists (MikoshiLang; no plot); legacy silently dropped "
          "tuple-form solutions"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _phaseplane(F: ListOf[Expr], vs: ListOf[Var]) -> Expr:
    comps = [sym(c, "PhasePlane") for c in list_arg(F, "PhasePlane", "system")]
    xs = _vars_list(vs, "PhasePlane")
    _same_len(comps, xs, "PhasePlane")
    return _solutions(comps, xs, "PhasePlane")


def _real_roots(expr, x, fname):
    sols = sp.solveset(expr, x, domain=sp.S.Reals)
    if isinstance(sols, sp.FiniteSet):
        return sorted(sols, key=lambda r: float(r))
    if sols is sp.S.EmptySet:
        return []
    raise NotImplementedError(f"{fname}: could not determine all real solutions ({sols})")


@fn("InflectionPoints", area=AREA, status="fixed",
    notes="real points where f'' changes sign; legacy returned every root of f'' (so x^4 reported 0, which is "
          "not an inflection point) and complex roots"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _inflection(f: Expr, x: Var) -> ListOf[Expr]:
    xv = variable(x, "InflectionPoints")
    v = sym(f, "InflectionPoints")
    d2 = sp.simplify(sp.diff(v, xv, 2))
    if d2 == 0:
        raise ValueError("InflectionPoints: f'' is identically zero")
    roots = _real_roots(d2, xv, "InflectionPoints")
    sing = []
    try:
        s = sp.singularities(d2, xv, domain=sp.S.Reals)
        if isinstance(s, sp.FiniteSet):
            sing = list(s)
        elif s is not sp.S.EmptySet:
            raise NotImplementedError("InflectionPoints: f'' has infinitely many singularities")
    except NotImplementedError:
        raise
    except Exception:
        pass
    pts = sorted(set(roots) | set(sing), key=lambda r: float(r))
    out = []
    for i, r in enumerate(pts):
        if r in sing:
            continue
        left = pts[i - 1] if i > 0 else r - 1
        right = pts[i + 1] if i + 1 < len(pts) else r + 1
        lv, rv = d2.subs(xv, (left + r) / 2), d2.subs(xv, (r + right) / 2)
        if sp.sign(sp.N(lv)) * sp.sign(sp.N(rv)) < 0:
            out.append(R(r))
    return L(*out)


def _optimize(f, vs, fname, sense):
    if is_list(f):
        raise NotImplementedError(f"{fname}: constrained optimisation is not supported")
    xs = _vars_list(vs, fname) if is_list(vs) else [variable(vs, fname)]
    if len(xs) != 1:
        raise NotImplementedError(f"{fname}: only functions of one variable are supported")
    x = xs[0]
    v = sym(f, fname)
    g = v if sense == "min" else -v
    sing = sp.singularities(g, x, domain=sp.S.Reals) if g.has(x) else sp.S.EmptySet
    if sing is not sp.S.EmptySet:
        raise NotImplementedError(f"{fname}: functions with real singularities are not supported")
    crit = _real_roots(sp.diff(g, x), x, fname) if g.has(x) else [sp.Integer(0)]
    cands = [(sp.simplify(g.subs(x, c)), c) for c in crit]
    lims = [(sp.limit(g, x, -sp.oo), -sp.oo), (sp.limit(g, x, sp.oo), sp.oo)]
    best = None
    for val, pt in cands:
        if best is None or bool(val < best[0]):
            best = (val, pt)
    for val, pt in lims:
        if not val.is_extended_real:
            raise NotImplementedError(f"{fname}: could not determine the behaviour at {pt}")
        if val == -sp.oo:
            best = (-sp.oo, pt)
            break
        if best is None or bool(val < best[0]):
            raise NotImplementedError(f"{fname}: the extremum is not attained (infimum {val} approached as "
                                      f"{x} -> {pt})")
    val, pt = best
    if sense == "max":
        val = -val
    return L(numeric_out(val, f), L(rule(Symbol(str(x)), R(pt))))


@fn("Minimize", area=AREA, status="fixed",
    notes="legacy ran a local SciPy search and returned {x -> x0, MinValue -> v} floats (garbage for unbounded "
          "functions); now Wolfram's exact global {min, {x -> x0}} for functions of one variable; "
          "constrained/multivariate problems raise")
def _minimize(f, vs):
    return _optimize(f, vs, "Minimize", "min")


@fn("Maximize", area=AREA, status="fixed",
    notes="exact global {max, {x -> x0}} (Wolfram format) for functions of one variable; legacy was a local "
          "SciPy search in a non-Wolfram format")
def _maximize(f, vs):
    return _optimize(f, vs, "Maximize", "max")


@fn("FindMinimum", area=AREA, status="fixed",
    notes="Wolfram FindMinimum[f, {x, x0}] / [f, {{x, x0}, {y, y0}}] -> {fmin, {x -> xmin, ...}} (local, "
          "numeric, BFGS via SciPy); the legacy form [f, {x, y}, {x0, y0}] is accepted and returns the same "
          "format; non-convergence raises"
          ". Check: reference (differential cases in tests/library).")
def _findmin(f: Expr, spec: Expr, x0: Expr = None) -> ListOf[Expr]:
    if x0 is not None:
        xs = _vars_list(spec, "FindMinimum")
        starts = [number(v, "FindMinimum", "starting point") for v in list_arg(x0, "FindMinimum")]
        _same_len(starts, xs, "FindMinimum")
    else:
        items = list_arg(spec, "FindMinimum")
        if items and not is_list(items[0]):
            items = [spec]
        xs, starts = [], []
        for it in items:
            p = list_arg(it, "FindMinimum")
            if len(p) != 2:
                raise ValueError("FindMinimum: each variable must be given as {x, x0}")
            xs.append(variable(p[0], "FindMinimum"))
            starts.append(number(p[1], "FindMinimum", "starting point"))
    try:
        import numpy as np
        from scipy.optimize import minimize
    except ImportError:
        raise NotImplementedError("FindMinimum requires SciPy (pip install mikoshilang[signal])")
    g = sym(f, "FindMinimum")
    fx = sp.lambdify(xs, g, "numpy")
    grad = sp.lambdify(xs, [sp.diff(g, x) for x in xs], "numpy")
    res = minimize(lambda z: float(fx(*z)), np.array([float(s) for s in starts]),
                   jac=lambda z: np.array(grad(*z), dtype=float), method="BFGS", options={"gtol": 1e-11})
    if not res.success or not np.isfinite(res.fun):
        raise ValueError(f"FindMinimum: the search did not converge ({res.message})")
    return L(float(res.fun), L(*(rule(Symbol(str(x)), float(v)) for x, v in zip(xs, res.x))))


@fn("LinearProgramming", area=AREA, status="fixed",
    notes="legacy solved A.x <= b with floats; Wolfram LinearProgramming[c, m, b] minimises c.x subject to "
          "m.x >= b and x >= 0; now exact (SymPy simplex); infeasible/unbounded problems raise"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _linprog(c: ListOf[Expr], m: MatrixOf[Expr], b: ListOf[Expr]) -> ListOf[Expr]:
    from sympy.solvers.simplex import linprog, InfeasibleLPError, UnboundedLPError
    cc = [number(v, "LinearProgramming") for v in list_arg(c, "LinearProgramming")]
    M = matrix_arg(m, "LinearProgramming")
    bb = [number(v, "LinearProgramming") for v in list_arg(b, "LinearProgramming")]
    if M.cols != len(cc) or M.rows != len(bb):
        raise ValueError("LinearProgramming: dimensions of c, m and b do not match")
    exact = not any(has_float(v) for v in cc + bb + list(M))
    try:
        _, x = linprog(sp.Matrix([cc]), -M, -sp.Matrix(bb))
    except InfeasibleLPError:
        raise ValueError("LinearProgramming: the constraints are infeasible")
    except UnboundedLPError:
        raise ValueError("LinearProgramming: the problem is unbounded")
    return L(*(R(sp.nsimplify(v)) if exact else float(v) for v in x))


def _func_name(fexpr, fname):
    if isinstance(fexpr, Expr) and isinstance(fexpr.head, str) and len(fexpr.args) == 1:
        return fexpr.head, fexpr.args[0]
    raise ValueError(f"{fname}: the unknown must be given as a[n]")


@fn("RSolve", area=AREA, status="fixed",
    notes="legacy called a SymPy Symbol (TypeError for every input); now RSolve[eq or {eq, a[0] == v, ...}, "
          "a[n], n] -> {{a[n] -> ...}} via SymPy rsolve")
def _rsolve(eq, fexpr, n):
    name, arg = _func_name(fexpr, "RSolve")
    nv = variable(n, "RSolve")
    a = sp.Function(name)
    eqs = list(eq.args) if is_list(eq) else [eq]
    rec, ics = None, {}
    for e in eqs:
        se = sym(e, "RSolve")
        if isinstance(se, sp.Equality) and se.lhs.func == a and not se.lhs.args[0].has(nv) and not se.rhs.has(a):
            ics[se.lhs] = se.rhs
        elif rec is None:
            rec = se
        else:
            raise NotImplementedError("RSolve: systems of recurrences are not supported")
    if rec is None:
        raise ValueError("RSolve: no recurrence equation given")
    if isinstance(rec, sp.Equality):
        rec = rec.lhs - rec.rhs
    sol = sp.rsolve(rec, a(nv), ics or None)
    if sol is None:
        raise NotImplementedError("RSolve: SymPy could not solve this recurrence")
    sol = sp.simplify(sol)
    # SymPy names the free constants C0, C1, ...; Wolfram uses C[1], C[2], ...
    consts = {c: sp.Symbol(f"C{int(c.name[1:]) + 1}") for c in sol.free_symbols
              if c.name[:1] == "C" and c.name[1:].isdigit()}
    sol = sol.xreplace(consts)          # simultaneous, so C0 -> C1 and C1 -> C2 do not collide
    from ..kernel import _constants_to_c
    return _constants_to_c(L(L(rule(fexpr, R(sol)))))


@fn("GeneratingFunction", area=AREA, status="fixed",
    notes="GeneratingFunction[{a0, a1, ...}, x] = a0 + a1 x + ... (MikoshiLang finite form); exact (legacy "
          "converted coefficients to floats); Wolfram's GeneratingFunction[expr, n, x] is not supported"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _genfunc(seq: Expr, x: Var, *rest: Expr) -> Expr:
    if rest:
        raise NotImplementedError("GeneratingFunction[expr, n, x] (infinite sequences) is not supported")
    cs = [sym(c, "GeneratingFunction") for c in list_arg(seq, "GeneratingFunction")]
    xv = variable(x, "GeneratingFunction")
    return numeric_out(sp.Add(*(c * xv ** i for i, c in enumerate(cs))), seq)


@fn("Wronskian", area=AREA, status="verified", notes="Wronskian[{f1, ..., fn}, x] (determinant, as Wolfram)"
                                                     ". Check: exact (exact input gives exact results), reference "
                                                     "(differential cases in tests/library).")
def _wronskian(fs: ListOf[Expr], x: Var) -> Expr:
    funcs = [sym(f, "Wronskian") for f in list_arg(fs, "Wronskian", min_len=1)]
    return R(sp.simplify(sp.wronskian(funcs, variable(x, "Wronskian"))))


@fn("ExactODEQ", area=AREA, status="verified", notes="True when M dx + N dy = 0 is exact (dM/dy == dN/dx)"
                                                     ". Check: reference (differential cases in tests/library).")
def _exactq(M: Expr, N: Expr, x: Var, y: Var) -> Boolean:
    m, n = sym(M, "ExactODEQ"), sym(N, "ExactODEQ")
    return bool(sp.simplify(sp.diff(m, variable(y, "ExactODEQ")) - sp.diff(n, variable(x, "ExactODEQ"))) == 0)


@fn("IntegratingFactor", area=AREA, status="fixed",
    notes="mu(x) or mu(y) making M dx + N dy exact; the test is now applied to the simplified ratio and "
          "failure raises (legacy returned an Error expression)"
          ". Check: reference (differential cases in tests/library).")
def _intfactor(M: Expr, N: Expr, x: Var, y: Var) -> Expr:
    m, n = sym(M, "IntegratingFactor"), sym(N, "IntegratingFactor")
    xv, yv = variable(x, "IntegratingFactor"), variable(y, "IntegratingFactor")
    my, nx = sp.diff(m, yv), sp.diff(n, xv)
    if sp.simplify(my - nx) == 0:
        return 1
    if n != 0:
        r = sp.simplify((my - nx) / n)
        if not r.has(yv):
            return R(sp.simplify(sp.exp(sp.integrate(r, xv))))
    if m != 0:
        r = sp.simplify((nx - my) / m)
        if not r.has(xv):
            return R(sp.simplify(sp.exp(sp.integrate(r, yv))))
    raise ValueError("IntegratingFactor: no integrating factor depending on x or y alone exists")


@fn("LyapunovDerivative", area=AREA, status="fixed",
    notes="dV/dt = grad V . F along x' = F(x) (MikoshiLang); mismatched lengths now raise (legacy zip-truncated)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _lyap(F: ListOf[Expr], vs: ListOf[Var], V: Expr) -> Expr:
    comps = [sym(c, "LyapunovDerivative") for c in list_arg(F, "LyapunovDerivative", "system")]
    xs = _vars_list(vs, "LyapunovDerivative")
    _same_len(comps, xs, "LyapunovDerivative")
    v = sym(V, "LyapunovDerivative")
    return R(sp.simplify(sp.Add(*(sp.diff(v, x) * f for x, f in zip(xs, comps)))))


def _ode_parts(eq, y, x, fname):
    xv = variable(x, fname)
    if isinstance(y, Symbol):
        f = sp.Function(y.name)
    elif isinstance(y, Expr) and isinstance(y.head, str) and len(y.args) == 1:
        f = sp.Function(y.head)
    else:
        raise ValueError(f"{fname}: the unknown function must be y or y[x]")
    e = sym(eq, fname)
    if isinstance(e, sp.Equality):
        e = e.lhs - e.rhs
    e = e.replace(lambda t: isinstance(t, sp.Function) and t.func.__name__ == "D", lambda t: sp.diff(*t.args))
    return e, f(xv), xv


@fn("ODEClassify", area=AREA, status="verified",
    notes="list of SymPy ODE hint names for eq, y, x (MikoshiLang)"
          ". Check: reference (differential cases in tests/library).")
def _odeclassify(eq: Expr, y: Expr, x: Var) -> ListOf[Expr]:
    e, fx, _ = _ode_parts(eq, y, x, "ODEClassify")
    return L(*sp.classify_ode(e, fx))


@fn("SeparableODE", area=AREA, status="fixed",
    notes="general solution {{y[x] -> ...}} of a separable first-order ODE (DSolve format); non-separable "
          "equations raise; legacy returned a SymPy equation"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _separable(eq: Expr, y: Expr, x: Var) -> Expr:
    e, fx, _ = _ode_parts(eq, y, x, "SeparableODE")
    if "separable" not in sp.classify_ode(e, fx):
        raise ValueError("SeparableODE: the equation is not separable")
    sols = sp.dsolve(e, fx, hint="separable")
    sols = sols if isinstance(sols, list) else [sols]
    out = []
    for s in sols:
        if s.lhs != fx:
            solved = sp.solve(s, fx)
            if not solved:
                raise NotImplementedError("SeparableODE: the implicit solution could not be solved for y")
            out.extend(L(rule(R(fx), R(v))) for v in solved)
        else:
            out.append(L(rule(R(fx), R(s.rhs))))
    from ..kernel import _constants_to_c
    return _constants_to_c(L(*out))


@fn("LaplaceODESolve", area=AREA, status="fixed",
    notes="MikoshiLang helper (not a Wolfram built-in): LaplaceODESolve[eqn | {eqn, y[0] == a, y'[0] == b, ...}, "
          "y, t] solves a linear constant-coefficient ODE by the Laplace transform and returns {{y[t] -> sol}} like "
          "DSolve, valid for t > 0. Initial values that are not given appear as y[0], y'[0], ... (Laplace "
          "form) rather than C[k]. Every solution is substituted back into the ODE and the initial conditions "
          "before it is returned. Non-linear or variable-coefficient equations raise (use DSolve). Legacy "
          "transformed the raw equation and solved for an unrelated symbol, so it never returned a solution"
          ". Check: exact, self-check (the result is verified before it is returned), reference (differential cases "
          "in tests/library).")
def _laplace_ode(eqs: Expr, y: Expr, t: Var) -> ListOf[Expr]:
    from ..kernel import substitute
    fname = "LaplaceODESolve"
    if not isinstance(y, Symbol):
        raise ValueError(f"{fname}: the second argument must be the function name y (not y[t])")
    tv = variable(t, fname)
    items = list(eqs.args) if is_list(eqs) else [eqs]
    yf = sp.Function(y.name)
    ode, ics = None, {}
    for it in items:
        v = sym(it, fname)
        if not isinstance(v, sp.Equality):
            raise ValueError(f"{fname}: expected equations (lhs == rhs)")
        lhs = v.lhs
        if isinstance(lhs, sp.Subs) and isinstance(lhs.expr, sp.Derivative) and lhs.point == (0,):
            ics[lhs.expr.derivative_count] = v.rhs
        elif lhs == yf(0):
            ics[0] = v.rhs
        elif v.has(tv):
            if ode is not None:
                raise ValueError(f"{fname}: only one differential equation is supported")
            ode = v.lhs - v.rhs
        else:
            raise ValueError(f"{fname}: initial conditions must have the form y[0] == a or y'[0] == b, ...")
    if ode is None:
        raise ValueError(f"{fname}: no differential equation in {tv}")
    derivs = [d for d in ode.atoms(sp.Derivative) if d.expr == yf(tv)]
    order = max([d.derivative_count for d in derivs] + [0])
    if order == 0:
        raise ValueError(f"{fname}: the equation contains no derivative of {y.name}")
    D = [sp.Dummy(f"d{k}") for k in range(order + 1)]
    rep = {yf(tv).diff(tv, k) if k else yf(tv): D[k] for k in range(order + 1)}
    e = sp.expand(ode.subs({k: v for k, v in sorted(rep.items(), key=lambda kv: -D.index(kv[1]))}))
    if e.has(yf):
        raise ValueError(f"{fname}: {y.name} must appear only as {y.name}[{tv}] and its derivatives")
    coeffs = [e.coeff(D[k]) for k in range(order + 1)]
    forcing = sp.expand(e - sum(c * d for c, d in zip(coeffs, D)))
    if any(c.has(tv) or c.free_symbols & set(D) for c in coeffs) or forcing.free_symbols & set(D):
        raise ValueError(f"{fname}: only linear equations with constant coefficients can be solved by Laplace "
                         "transform; use DSolve")
    if sp.Poly(e - forcing, *D).total_degree() != 1:
        raise ValueError(f"{fname}: the equation is not linear in {y.name}")
    s_ = sp.Dummy("s")
    marks = [sp.Symbol(f"LaplaceIC${k}") for k in range(order)]
    c = [ics.get(k, marks[k]) for k in range(order)]
    G = sp.laplace_transform(forcing, tv, s_, noconds=True) if forcing != 0 else sp.Integer(0)
    if isinstance(G, sp.LaplaceTransform) or (isinstance(G, sp.Basic) and G.has(sp.LaplaceTransform)):
        raise NotImplementedError(f"{fname}: SymPy cannot transform the forcing term {forcing}")
    Yv = sp.Symbol("Y")
    lap = sum(a * (s_ ** k * Yv - sum(s_ ** (k - 1 - j) * c[j] for j in range(k))) for k, a in enumerate(coeffs)) + G
    Ys = sp.solve(lap, Yv)
    if len(Ys) != 1:
        raise ValueError(f"{fname}: could not solve the transformed equation")
    tp = sp.Symbol(tv.name, positive=True)
    sol = sp.inverse_laplace_transform(sp.apart(sp.together(Ys[0]), s_), s_, tp)
    if sol.has(sp.InverseLaplaceTransform):
        raise NotImplementedError(f"{fname}: SymPy could not invert the transform")
    sol = sp.simplify(sol.subs(sp.Heaviside(tp), 1)).subs(tp, tv)
    # verify: the ODE residual and the initial conditions
    resid = sp.simplify(ode.subs(yf(tv), sol).doit())
    ok = resid == 0
    for k in range(order):
        ok = ok and sp.simplify(sp.limit(sol.diff(tv, k), tv, 0, "+") - c[k]) == 0
    if not ok:
        raise ArithmeticError(f"{fname}: the Laplace-transform solution failed verification")
    out = R(sol)
    back = {m.name: (Expr(y.name, 0) if k == 0 else Expr(Expr(Expr("Derivative", k), y), 0))
            for k, m in enumerate(marks)}
    out = substitute(out, back)
    return L(L(rule(Expr(y.name, Symbol(tv.name)), out)))


record("Table", AREA, "kernel", "Table is a held kernel form; the legacy Table[a, b] rule was unreachable and removed"
                                ". Check: exact (exact input gives exact results), reference (differential cases in "
                                "tests/library).")
