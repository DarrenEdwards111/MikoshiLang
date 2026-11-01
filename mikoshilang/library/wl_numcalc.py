"""Coverage batch 2: numerical calculus, equation solving, optimisation and fitting with Wolfram semantics.

Every function works at machine precision by default and at arbitrary precision with WorkingPrecision -> p
(mpmath); results are machine numbers unless a WorkingPrecision above machine precision is given.
"""

from __future__ import annotations
import mpmath
import sympy as sp

from ..expr import Expr, Symbol
from .core import fn, R, L, rule, sym, variable, list_arg, is_list, Unevaluated
from .wl_common import split_options, ev, reject_options
from .wl_special import to_mp, from_mp

AREA = "numerics (batch 2)"

_OPTS = {"WorkingPrecision", "PrecisionGoal", "AccuracyGoal", "Method", "MaxIterations", "MaxRecursion",
         "Exclusions", "Compiled", "EvaluationMonitor", "StepMonitor", "WorkingPrecision", "MinRecursion",
         "Jacobian", "Gradient", "VerifyConvergence", "NSumTerms", "Weights", "MaxPoints"}


# Options that only affect speed, never the value of the result: accepted and ignored.
_HINTS = {"Compiled"}


def _opts(args: tuple, fname: str, handled: tuple = ("WorkingPrecision",), goals: bool = False) -> tuple:
    """Split the options of a numerical function.  An option other than ``handled`` (or a pure performance hint)
    with a value other than Automatic raises NotImplementedError instead of being silently ignored; with
    ``goals``, AccuracyGoal / PrecisionGoal up to the digits the error check guarantees (half the working
    precision) are accepted."""
    pos, opts = split_options(args, _OPTS, fname)
    wp = opts.get("WorkingPrecision")
    if wp is None or (isinstance(wp, Symbol) and wp.name == "MachinePrecision"):
        prec, dps = "m", 15
    else:
        d = sym(wp, fname)
        if not d.is_number or d <= 0:
            raise ValueError(f"{fname}: WorkingPrecision must be a positive number")
        d = float(d)
        if d <= 16:
            prec, dps = "m", 15
        else:
            dps = int(round(d))
            prec = mpmath.libmp.dps_to_prec(dps)
    unsupported = [n for n in opts if n not in handled and n not in _HINTS]
    reject_options(opts, unsupported, fname, min(dps, 15) // 2 if goals else None)
    return pos, opts, prec, dps


def _max_iterations(opts: dict, fname: str) -> int | None:
    """MaxIterations -> n (a positive integer, Infinity or Automatic) as an iteration limit, or None for the
    solver's default."""
    m = opts.get("MaxIterations")
    if m is None or (isinstance(m, Symbol) and m.name == "Automatic"):
        return None
    if isinstance(m, Symbol) and m.name == "Infinity":
        return 10 ** 5
    if isinstance(m, int) and not isinstance(m, bool) and m > 0:
        return m
    raise ValueError(f"{fname}: MaxIterations must be a positive integer or Infinity, not {m}")


def _method(opts):
    m = opts.get("Method")
    if m is None:
        return None
    if is_list(m) and m.args:
        m = m.args[0]
    if isinstance(m, Symbol):
        return m.name
    if isinstance(m, str):
        return m
    return str(m)


class _Fn:
    """A numeric function of ``xs`` built from a MikoshiLang expression (mpmath via lambdify, falling back to
    MikoshiLang evaluation for heads SymPy does not know)."""

    def __init__(self, expr, xs, fname):
        self.expr, self.xs, self.fname = expr, xs, fname
        self.g = None
        self.f = None
        try:
            g = sym(expr, fname)
            if not any(isinstance(a, sp.core.function.AppliedUndef) for a in sp.preorder_traversal(g)):
                self.g = g
                self.f = sp.lambdify(xs, g, modules=["mpmath"])
        except Exception:
            self.f = None

    def __call__(self, *vals):
        if self.f is not None:
            try:
                v = self.f(*vals)
                return mpmath.mpmathify(v)
            except Exception:
                pass
        rules = L(*(rule(Symbol(str(x)), from_mp(mpmath.mpmathify(v), mpmath.mp.prec if mpmath.mp.dps > 15 else "m"))
                    for x, v in zip(self.xs, vals)))
        out = ev(Expr("N", Expr("ReplaceAll", self.expr, rules), max(mpmath.mp.dps, 16)))
        try:
            return to_mp(sym(out, self.fname))
        except Exception:
            raise ValueError(f"{self.fname}: the function {self.expr} is not numerical at {list(vals)}")


def _num(x, fname, what="value"):
    v = sym(x, fname)
    if not v.is_number and v not in (sp.oo, -sp.oo):
        raise ValueError(f"{fname}: {what} {x} must be numerical")
    return to_mp(v)


def _iterator(it, fname):
    p = list_arg(it, fname, "iterator")
    if len(p) < 2:
        raise ValueError(f"{fname}: iterators must have the form {{x, xmin, xmax}}")
    return variable(p[0], fname), p[1:]


def _out(v, prec):
    if prec == "m":
        if isinstance(v, mpmath.mpc):
            return R(complex(v)) if v.imag != 0 else float(v.real)
        return from_mp(v, "m")
    return from_mp(v, prec)


# ───────────────────────────── NIntegrate ─────────────────────────────

def _quad_method(opts):
    m = _method(opts)
    table = {None: "tanh-sinh", "Automatic": "tanh-sinh", "DoubleExponential": "tanh-sinh",
             "GaussKronrod": "gauss-legendre", "GaussLegendre": "gauss-legendre", "GlobalAdaptive": "tanh-sinh",
             "LocalAdaptive": "tanh-sinh", "TrapezoidalRule": "tanh-sinh", "Trapezoidal": "tanh-sinh"}
    if m not in table:
        raise NotImplementedError(f"NIntegrate: Method -> {m} is not supported")
    return table[m]


@fn("NIntegrate", area=AREA, status="added",
    notes="numerical integration NIntegrate[f, {x, a, b}, {y, c, d}, ...] (later limits may depend on earlier "
          "variables, Wolfram order), infinite ranges, intermediate points {x, a, x1, ..., b} for interior "
          "singularities, complex-valued integrands and complex contours; WorkingPrecision -> p for arbitrary "
          "precision (mpmath tanh-sinh / Gauss-Legendre: Method -> \"DoubleExponential\" | \"GaussKronrod\")")
def _nintegrate(f: Expr, *rest: Expr) -> Expr:
    pos, opts, prec, dps = _opts(rest, "NIntegrate", ("WorkingPrecision", "Method"), goals=True)
    if not pos:
        raise ValueError("NIntegrate: an integration range {x, a, b} is required")
    its = [_iterator(it, "NIntegrate") for it in pos]
    xs = [x for x, _ in its]
    method = _quad_method(opts)
    with mpmath.workdps(dps + (4 if prec == "m" else 10)):
        F = _Fn(f, xs, "NIntegrate")

        def integrate(level, vals):
            x, lims = its[level]
            pts = []
            for b in lims:
                bb = sym(b, "NIntegrate")
                if bb.free_symbols:
                    bb = bb.subs({xs[i]: sp.Float(vals[i], dps + 10) if not isinstance(vals[i], mpmath.mpc)
                                  else complex(vals[i]) for i in range(level)})
                pts.append(to_mp(bb))

            def g(t):
                if level + 1 == len(its):
                    return F(*vals, t)
                return integrate(level + 1, vals + [t])
            if level == 0:
                return mpmath.quad(g, pts, method=method, maxdegree=8, error=True)
            return mpmath.quad(g, pts, method=method, maxdegree=6)
        v, err = integrate(0, [])
        if mpmath.isfinite(v) and err > mpmath.mpf(10) ** (-min(dps, 15) // 2) * max(1, abs(v)):
            raise ValueError(f"NIntegrate: the integral failed to converge (error estimate {mpmath.nstr(err, 3)}); "
                             f"the integrand may have a non-integrable singularity")
    if isinstance(v, (mpmath.mpf, mpmath.mpc)) and not mpmath.isfinite(v):
        raise ValueError("NIntegrate: the integral does not converge")
    with mpmath.workdps(dps):
        return _out(+v, prec)


# ───────────────────────────── NSum / NProduct ─────────────────────────────

def _nsum_like(f, rest, fname, op):
    pos, opts, prec, dps = _opts(rest, fname, goals=True)
    if not pos:
        raise ValueError(f"{fname}: an iterator {{n, nmin, nmax}} is required")
    its = []
    for it in pos:
        x, lims = _iterator(it, fname)
        if len(lims) == 1:
            lims = [1, lims[0]]
        if len(lims) not in (2, 3):
            raise ValueError(f"{fname}: iterators must have the form {{n, nmin, nmax}} or {{n, nmin, nmax, dn}}")
        its.append((x, lims))
    xs = [x for x, _ in its]
    with mpmath.workdps(dps + 10):
        F = _Fn(f, xs, fname)

        def run(level, vals):
            x, lims = its[level]
            a, b = (_num(v, fname, "limit") for v in lims[:2])
            d = _num(lims[2], fname, "step") if len(lims) == 3 else mpmath.mpf(1)
            if d <= 0:
                raise ValueError(f"{fname}: the step must be positive")
            if mpmath.isinf(a):
                raise NotImplementedError(f"{fname}: a lower limit of -Infinity is not supported; reverse the index")
            n_hi = mpmath.inf if mpmath.isinf(b) else mpmath.floor((b - a) / d)

            def g(k):
                t = a + k * d
                if level + 1 == len(its):
                    return F(*vals, t)
                return run(level + 1, vals + [t])
            if not mpmath.isinf(n_hi) and n_hi < 2000:
                acc = mpmath.mpf(0) if op == "sum" else mpmath.mpf(1)
                for k in range(int(n_hi) + 1):
                    acc = acc + g(k) if op == "sum" else acc * g(k)
                return acc
            if op == "sum":
                v = mpmath.nsum(g, [0, n_hi])
                if level == 0:
                    # extrapolations of divergent series depend on the working precision: compare two
                    with mpmath.workdps(mpmath.mp.dps + 15):
                        v2 = mpmath.nsum(g, [0, n_hi])
                    if abs(v2 - v) > mpmath.mpf(10) ** (-min(dps, 15) // 2) * max(1, abs(v)):
                        raise ValueError(f"{fname}: the sum failed to converge")
                return v
            return mpmath.nprod(g, [0, n_hi])
        v = run(0, [])
    if not mpmath.isfinite(v):
        raise ValueError(f"{fname}: the {'sum' if op == 'sum' else 'product'} does not converge")
    with mpmath.workdps(dps):
        return _out(+v, prec)


@fn("NSum", area=AREA, status="added",
    notes="NSum[f, {n, nmin, nmax}] and {n, nmin, nmax, dn}, multiple iterators, nmax = Infinity (mpmath nsum: "
          "Richardson/Shanks/Euler-Maclaurin acceleration, alternating series); WorkingPrecision -> p; finite "
          "short ranges are summed directly; divergent sums raise")
def _nsum(f: Expr, *rest: Expr) -> Expr:
    return _nsum_like(f, rest, "NSum", "sum")


@fn("NProduct", area=AREA, status="added",
    notes="NProduct[f, {n, nmin, nmax}] (also infinite products, mpmath nprod); WorkingPrecision -> p")
def _nproduct(f: Expr, *rest: Expr) -> Expr:
    return _nsum_like(f, rest, "NProduct", "prod")


# ───────────────────────────── equation solving ─────────────────────────────

def _equations(eqs, fname):
    items = list(eqs.args) if is_list(eqs) else [eqs]
    if len(items) == 1 and isinstance(items[0], Expr) and items[0].head == "And":
        items = list(items[0].args)
    out = []
    for e in items:
        if isinstance(e, Expr) and e.head == "Equal" and len(e.args) == 2:
            out.append(sym(e.args[0], fname) - sym(e.args[1], fname))
        elif e is True:
            continue
        elif isinstance(e, Expr) and e.head in ("Less", "Greater", "LessEqual", "GreaterEqual", "Unequal"):
            raise NotImplementedError(f"{fname}: inequalities are not supported")
        else:
            out.append(sym(e, fname))
    return out


def _sort_key(v):
    c = complex(v)
    return (round(c.real, 10), round(c.imag, 10))


def _nroots_list(p, x, dps, fname):
    poly = sp.Poly(sp.expand(p), x)
    if poly.degree() < 1:
        return []
    roots = poly.nroots(n=max(dps, 15), maxsteps=500)
    return sorted(roots, key=_sort_key)


def _numval(v, prec, dps):
    v = sp.N(v, max(dps, 15) + 5)
    with mpmath.workdps(dps + 5):
        c = to_mp(v)
    with mpmath.workdps(dps):
        return _out(+c, prec)


@fn("NSolve", area=AREA, status="added",
    notes="NSolve[eqns, vars] numerical solutions of polynomial (and rational) equations and systems as "
          "{{x -> x1}, ...} with every complex root repeated by multiplicity; NSolve[eqns, vars, Reals] keeps real "
          "solutions; WorkingPrecision -> p; transcendental equations raise (use FindRoot)")
def _nsolve(eqs: Expr, *rest: Expr) -> ListOf[Expr]:
    pos, opts, prec, dps = _opts(rest, "NSolve")
    exprs = _equations(eqs, "NSolve")
    if pos:
        vs = pos[0]
        xs = [variable(v, "NSolve") for v in (vs.args if is_list(vs) else [vs])]
        dom = pos[1] if len(pos) > 1 else None
    else:
        xs = sorted(set().union(*(e.free_symbols for e in exprs)), key=str)
        dom = None
    reals = isinstance(dom, Symbol) and dom.name == "Reals"
    if dom is not None and not reals and not (isinstance(dom, Symbol) and dom.name == "Complexes"):
        raise NotImplementedError(f"NSolve: domain {dom} is not supported")
    nums = [sp.together(e).as_numer_denom() for e in exprs]
    for n, _d in nums:
        if not n.is_polynomial(*xs):
            raise NotImplementedError("NSolve: only polynomial and rational equations are supported; use FindRoot")
    if not xs:
        return L(L()) if all(sp.simplify(n) == 0 for n, _ in nums) else L()
    sols = []
    if len(xs) == 1 and len(nums) == 1:
        n, d = nums[0]
        if sp.expand(n) == 0:
            raise ValueError("NSolve: the equation is satisfied for every value of the variable")
        for r in _nroots_list(n, xs[0], dps, "NSolve"):
            if d.has(xs[0]) and abs(complex(d.subs(xs[0], r))) < 1e-12:
                continue
            sols.append({xs[0]: r})
    else:
        try:
            exact = sp.solve([n for n, _ in nums], xs, dict=True)
        except NotImplementedError:
            raise NotImplementedError("NSolve: this polynomial system could not be solved")
        for s in exact:
            if len(s) < len(xs):
                raise ValueError("NSolve: the system has infinitely many solutions")
            if any(abs(complex(sp.N(d.subs(s)))) < 1e-12 for _, d in nums if d.free_symbols):
                continue
            sols.append({x: sp.N(s[x], dps + 5) for x in xs})
    if len(xs) > 1:
        def key(sol):
            cs = [complex(sp.N(sol[x])) for x in xs]
            return (any(abs(c.imag) > 1e-12 for c in cs), [(round(c.real, 10), round(c.imag, 10)) for c in cs])
        sols.sort(key=key)
    out = []
    for s in sols:
        if reals and any(abs(complex(sp.N(v)).imag) > 1e-10 * max(1, abs(complex(sp.N(v)))) for v in s.values()):
            continue
        vals = {x: (sp.re(sp.N(v, dps + 5)) if reals else v) for x, v in s.items()}
        out.append(L(*(rule(Symbol(str(x)), _numval(vals[x], prec, dps)) for x in xs)))
    return L(*out)


def _tocomplex(v):
    if isinstance(v, (int, float, complex)):
        return complex(v)
    return complex(sp.N(sym(v, "NSolve")))


@fn("NRoots", area=AREA, status="added",
    notes="NRoots[poly == 0, x] gives x == r1 || x == r2 || ... with roots repeated by multiplicity (machine "
          "precision, or the precision of approximate coefficients); non-polynomials raise")
def _nroots(eq: Expr, x: Expr, *rest: Expr) -> Expr:
    pos, opts, prec, dps = _opts(rest, "NRoots")
    exprs = _equations(eq, "NRoots")
    if len(exprs) != 1:
        raise ValueError("NRoots: a single polynomial equation is required")
    xv = variable(x, "NRoots")
    if not exprs[0].is_polynomial(xv):
        raise ValueError(f"NRoots: {eq} is not a polynomial equation in {x}")
    roots = _nroots_list(exprs[0], xv, dps, "NRoots")
    if not roots:
        raise ValueError("NRoots: the polynomial has no roots")
    terms = [Expr("Equal", x, _numval(r, prec, dps)) for r in roots]
    return terms[0] if len(terms) == 1 else Expr("Or", *terms)


@fn("FindRoot", area=AREA, status="added",
    notes="FindRoot[f, {x, x0}] / [lhs == rhs, {x, x0}] / {x, x0, x1} (secant) / {x, x0, xmin, xmax} (bounded) "
          "and systems FindRoot[{eqs}, {{x, x0}, {y, y0}}] -> {x -> root, ...}; complex starting values search "
          "in the complex plane; WorkingPrecision -> p; raises when no root is found")
def _findroot(eqs: Expr, *rest: Expr) -> ListOf[Expr]:
    pos, opts, prec, dps = _opts(rest, "FindRoot", ("WorkingPrecision", "MaxIterations"))
    steps = _max_iterations(opts, "FindRoot")
    exprs = _equations(eqs, "FindRoot")
    if not pos:
        raise ValueError("FindRoot: starting values {x, x0} are required")
    specs = pos
    if len(pos) == 1 and is_list(pos[0]) and pos[0].args and is_list(pos[0].args[0]):
        specs = list(pos[0].args)
    xs, starts, bounds = [], [], []
    for s in specs:
        p = list_arg(s, "FindRoot", "starting value")
        if len(p) < 2:
            raise ValueError("FindRoot: each variable needs a starting value {x, x0}")
        xs.append(variable(p[0], "FindRoot"))
        vals = [sym(v, "FindRoot") for v in p[1:]]
        starts.append(vals)
        bounds.append(vals[-2:] if len(vals) == 3 else None)
    if len(exprs) != len(xs):
        raise ValueError("FindRoot: the number of equations does not match the number of variables")
    with mpmath.workdps(dps + 10):
        F = [_Fn(R(e), xs, "FindRoot") for e in exprs]
        x0 = [to_mp(v[0]) for v in starts]
        def search(kw):
            if len(xs) == 1:
                if len(starts[0]) == 2:
                    r = mpmath.findroot(F[0], (to_mp(starts[0][0]), to_mp(starts[0][1])), solver="secant", **kw)
                elif len(starts[0]) == 3:
                    lo, hi = to_mp(starts[0][1]), to_mp(starts[0][2])
                    r = mpmath.findroot(F[0], x0[0], **kw)
                    if not (lo <= mpmath.re(r) <= hi):
                        raise ValueError
                else:
                    r = mpmath.findroot(F[0], x0[0], **kw)
                return [r]
            r = mpmath.findroot([lambda *a, f=f: f(*a) for f in F], x0, **kw)
            return [r[i] for i in range(len(xs))]
        try:
            rs = search({} if steps is None else {"maxsteps": steps})
        except (ValueError, ZeroDivisionError):
            if steps is not None:
                try:
                    search({})
                except (ValueError, ZeroDivisionError):
                    pass
                else:
                    raise ValueError(f"FindRoot: the search did not converge within MaxIterations -> {steps} "
                                     f"iterations") from None
            raise ValueError("FindRoot: no root was found from the given starting point") from None
        res = max(abs(f(*rs)) for f in F)
        if not mpmath.isfinite(res) or res > mpmath.mpf(10) ** (-dps // 2):
            raise ValueError("FindRoot: the search did not converge to a root")
        for i, b in enumerate(bounds):
            if b is not None and len(starts[i]) == 3:
                lo, hi = to_mp(b[0]), to_mp(b[1])
                if not (lo <= mpmath.re(rs[i]) <= hi):
                    raise ValueError("FindRoot: the root left the given bounds")
        real_start = all(mpmath.im(v) == 0 for v in x0)
        out = []
        for x, v in zip(xs, rs):
            if real_start and isinstance(v, mpmath.mpc) and abs(v.imag) <= abs(v) * mpmath.mpf(10) ** (-dps):
                v = v.real
            with mpmath.workdps(dps):
                out.append(rule(Symbol(str(x)), _out(+v, prec)))
    return L(*out)


# ───────────────────────────── optimisation ─────────────────────────────

@fn("FindMaximum", area=AREA, status="added",
    notes="FindMaximum[f, {x, x0}] / {{x, x0}, {y, y0}} -> {fmax, {x -> xmax, ...}}: FindMinimum of -f (local, "
          "BFGS); non-convergence raises")
def _findmax(f: Expr, *rest: Expr) -> ListOf[Expr]:
    out = ev(Expr("FindMinimum", Expr("Times", -1, f), *rest))
    if not (isinstance(out, Expr) and out.head == "List" and len(out.args) == 2):
        raise Unevaluated()
    v = out.args[0]
    return L(-v if isinstance(v, (int, float)) else ev(Expr("Times", -1, v)), out.args[1])


_REL = {"Less": "lt", "LessEqual": "lt", "Greater": "gt", "GreaterEqual": "gt", "Equal": "eq"}


def _constraints(cons, xs, fname):
    """-> (bounds {x: [lo, hi]}, [(kind, g)]) with g >= 0 ('ineq') or g == 0 ('eq')."""
    items = list(cons.args) if is_list(cons) else [cons]
    flat = []
    while items:
        c = items.pop(0)
        if isinstance(c, Expr) and c.head == "And":
            items = list(c.args) + items
        elif isinstance(c, Expr) and c.head == "Element":
            dom = c.args[1]
            if isinstance(dom, Symbol) and dom.name == "Reals":
                continue
            raise NotImplementedError(f"{fname}: integer and other domain constraints are not supported")
        elif isinstance(c, Expr) and c.head == "Inequality":
            a = c.args
            for i in range(0, len(a) - 2, 2):
                flat.append(Expr(a[i + 1].name, a[i], a[i + 2]))
        elif c is True:
            continue
        else:
            flat.append(c)
    bounds = {x: [None, None] for x in xs}
    general = []
    for c in flat:
        if not (isinstance(c, Expr) and c.head in _REL):
            raise ValueError(f"{fname}: {c} is not a constraint")
        parts = [sym(a, fname) for a in c.args]
        for lhs, rhs in zip(parts, parts[1:]):
            kind = _REL[c.head]
            if kind == "eq":
                general.append(("eq", lhs - rhs))
                continue
            g = rhs - lhs if kind == "lt" else lhs - rhs           # g >= 0
            if lhs in bounds and rhs.is_number:
                bounds[lhs][1 if kind == "lt" else 0] = float(rhs)
            elif rhs in bounds and lhs.is_number:
                bounds[rhs][0 if kind == "lt" else 1] = float(lhs)
            else:
                general.append(("ineq", g))
    return bounds, general


def _need_scipy(fname):
    try:
        import scipy  # noqa: F401
    except ImportError:
        raise NotImplementedError(f"{fname} requires SciPy (pip install mikoshilang[signal])")


def _nminimize(f, vs, fname, sense):
    import numpy as np
    _need_scipy(fname)
    from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
    if is_list(f) and f.args:
        obj, cons = f.args[0], L(*f.args[1:])
    else:
        obj, cons = f, L()
    xs = [variable(v, fname) for v in (vs.args if is_list(vs) else [vs])]
    g = sym(obj, fname)
    if sense == "max":
        g = -g
    bounds, general = _constraints(cons, xs, fname)
    fx = sp.lambdify(xs, g, "numpy")
    fun = lambda z: float(np.real(fx(*z)))                                  # noqa: E731
    gx = sp.lambdify(xs, [sp.diff(g, x) for x in xs], "numpy")
    jac = lambda z: np.array(np.real(gx(*z)), dtype=float).reshape(-1)       # noqa: E731
    cfun = [(k, sp.lambdify(xs, c, "numpy")) for k, c in general]
    scons = [{"type": k, "fun": (lambda z, h=h: float(h(*z)))} for k, h in cfun]
    tol = 1e-9

    def feasible(z):
        for (k, h) in cfun:
            v = float(h(*z))
            if (k == "ineq" and v < -tol) or (k == "eq" and abs(v) > 1e-6):
                return False
        for x, zi in zip(xs, z):
            lo, hi = bounds[x]
            if (lo is not None and zi < lo - tol) or (hi is not None and zi > hi + tol):
                return False
        return True
    rng = np.random.default_rng(12345)
    box = [(lo if lo is not None else -10.0, hi if hi is not None else 10.0) for lo, hi in
           (bounds[x] for x in xs)]
    box = [(a, b) if a < b else (a, a + 20.0) if b is None else (b - 20.0, b) for a, b in box]
    cands = []
    if all(bounds[x][0] is not None and bounds[x][1] is not None for x in xs):
        nlc = [NonlinearConstraint(lambda z, h=h: float(h(*z)), 0 if k == "ineq" else 0,
                                   np.inf if k == "ineq" else 0) for k, h in cfun]
        de = differential_evolution(fun, box, constraints=nlc, seed=12345, tol=1e-12, polish=True, maxiter=2000)
        cands.append(de.x)
    sb = [(lo, hi) for lo, hi in (bounds[x] for x in xs)]
    starts = [rng.uniform([a for a, _ in box], [b for _, b in box]) for _ in range(40)]
    for s in starts:
        try:
            res = minimize(fun, s, jac=jac, method="SLSQP" if (scons or any(b != (None, None) for b in sb)) else "BFGS",
                           bounds=sb if any(b != (None, None) for b in sb) else None,
                           constraints=scons or (), options={"maxiter": 2000})
            cands.append(res.x)
        except Exception:
            continue
    best = None
    for z in cands:
        z = np.asarray(z, dtype=float)
        if not np.all(np.isfinite(z)) or not feasible(z):
            continue
        v = fun(z)
        if np.isfinite(v) and (best is None or v < best[0] - 1e-12):
            best = (v, z)
    if best is None:
        raise ValueError(f"{fname}: no feasible point was found")
    # polish the best point with a tight local search
    try:
        sbx = sb if any(b != (None, None) for b in sb) else None
        res = minimize(fun, best[1], jac=jac, method="SLSQP" if (scons or sbx) else "BFGS", bounds=sbx,
                       constraints=scons or (), options={"maxiter": 5000, "ftol": 1e-16} if (scons or sbx)
                       else {"maxiter": 5000, "gtol": 1e-12})
        if np.all(np.isfinite(res.x)) and feasible(res.x) and fun(res.x) <= best[0] + 1e-9 * max(1, abs(best[0])):
            best = (fun(res.x), res.x)
    except Exception:
        pass
    v, z = best
    if np.max(np.abs(z)) > 1e8 or abs(v) > 1e12:
        raise ValueError(f"{fname}: the objective appears to be unbounded")
    v = -v if sense == "max" else v
    return L(float(v), L(*(rule(Symbol(str(x)), float(zi)) for x, zi in zip(xs, z))))


@fn("NMinimize", area=AREA, status="added",
    notes="NMinimize[f, vars] / NMinimize[{f, cons}, vars] -> {fmin, {x -> x*, ...}}: global numeric search "
          "(differential evolution when every variable is bounded, plus 40 seeded multi-start SLSQP/BFGS local "
          "searches; constraints: <, <=, >, >=, ==, Inequality, And); like Wolfram's it is a heuristic global "
          "method; integer domains raise")
def _nmin(f: Expr, vs: Expr, *rest: Expr) -> Expr:
    _nmin_rest(rest, "NMinimize")
    return _nminimize(f, vs, "NMinimize", "min")


def _nmin_rest(rest: tuple, fname: str) -> None:
    """NMinimize / NMaximize: a domain other than Reals and options other than their defaults raise (they used
    to be ignored, so NMinimize[f, x, Integers] silently gave a non-integer minimum)."""
    pos, opts = split_options(rest, _OPTS | {"RandomSeed", "PerformanceGoal"}, fname)
    for d in pos:
        if not (isinstance(d, Symbol) and d.name == "Reals"):
            raise NotImplementedError(f"{fname}: the domain {d} is not supported (only Reals)")
    reject_options(opts, [k for k in opts if k not in _HINTS | {"PerformanceGoal", "RandomSeed"}], fname)


@fn("NMaximize", area=AREA, status="added",
    notes="NMaximize[f, vars] / [{f, cons}, vars] -> {fmax, {x -> x*, ...}} (NMinimize of -f)")
def _nmax(f: Expr, vs: Expr, *rest: Expr) -> Expr:
    _nmin_rest(rest, "NMaximize")
    return _nminimize(f, vs, "NMaximize", "max")


# ───────────────────────────── fitting ─────────────────────────────

def _data(data, nvars, fname):
    rows = list_arg(data, fname, "data")
    import numpy as np
    if rows and not is_list(rows[0]):
        X = np.arange(1, len(rows) + 1, dtype=float).reshape(-1, 1)
        y = np.array([complex(_tocomplex(v)).real for v in rows])
        return X, y
    M = [[complex(_tocomplex(v)).real for v in r.args] for r in rows]
    M = np.array(M, dtype=float)
    if M.shape[1] != nvars + 1:
        raise ValueError(f"{fname}: each data point must have {nvars + 1} coordinates")
    return M[:, :nvars], M[:, nvars]


@fn("Fit", area=AREA, status="added",
    notes="Fit[data, {f1, f2, ...}, vars] linear least-squares fit -> c1 f1 + c2 f2 + ... (data {y1, y2, ...} "
          "at x = 1, 2, ... or {{x, y}, ...} or {{x1, x2, y}, ...}); machine-precision coefficients")
def _fit(data: ListOf[Expr], funs: Expr, vs: Expr) -> Expr:
    import numpy as np
    xs = [variable(v, "Fit") for v in (vs.args if is_list(vs) else [vs])]
    fl = list(funs.args) if is_list(funs) else [funs]
    X, y = _data(data, len(xs), "Fit")
    cols = []
    for fe in fl:
        g = sym(fe, "Fit")
        h = sp.lambdify(xs, g, "numpy")
        cols.append(np.broadcast_to(np.asarray(h(*X.T), dtype=float), (X.shape[0],)))
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    terms = [Expr("Times", float(ci), fe) for ci, fe in zip(c, fl)]
    return ev(Expr("Plus", *terms))


@fn("FindFit", area=AREA, status="added",
    notes="FindFit[data, expr, pars, vars] nonlinear least squares -> {a -> a*, ...}; pars may be {{a, a0}, ...} "
          "with starting values (default 1); constraints FindFit[data, {expr, cons}, ...] are not supported")
def _findfit(data: Expr, model: Expr, pars: Expr, vs: Expr, *rest: Expr) -> ListOf[Expr]:
    import numpy as np
    _need_scipy("FindFit")
    from scipy.optimize import least_squares
    if is_list(model):
        raise NotImplementedError("FindFit: constrained fits are not supported")
    xs = [variable(v, "FindFit") for v in (vs.args if is_list(vs) else [vs])]
    plist = list(pars.args) if is_list(pars) else [pars]
    ps, p0 = [], []
    for p in plist:
        if is_list(p):
            ps.append(variable(p.args[0], "FindFit"))
            p0.append(float(complex(_tocomplex(p.args[1])).real))
        else:
            ps.append(variable(p, "FindFit"))
            p0.append(1.0)
    X, y = _data(data, len(xs), "FindFit")
    g = sym(model, "FindFit")
    h = sp.lambdify(ps + xs, g, "numpy")

    def resid(p):
        return np.asarray(h(*p, *X.T), dtype=float) * np.ones(len(y)) - y
    res = least_squares(resid, np.array(p0), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
    if not res.success:
        raise ValueError(f"FindFit: the fit did not converge ({res.message})")
    return L(*(rule(Symbol(str(p)), float(v)) for p, v in zip(ps, res.x)))
