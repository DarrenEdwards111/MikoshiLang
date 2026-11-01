"""Coverage batch 2: Interpolation, ListInterpolation, InterpolatingFunction and NDSolve / NDSolveValue.

An InterpolatingFunction is the expression InterpolatingFunction[{{xmin, xmax}}, "id"]; the numerical data
lives in this session's registry under "id".  It is applied as ifun[x] and differentiated as ifun'[x]
(Derivative[n][ifun][x]).  One-dimensional only.
"""

from __future__ import annotations
import itertools

import numpy as np
import sympy as sp

from ..expr import Expr, Symbol
from .core import fn, R, L, rule, sym, variable, list_arg, is_list, record
from .wl_common import split_options, ev

AREA = "numerics (batch 2)"

_REG = {}
_ids = itertools.count(1)


class _Piecewise:
    """Piecewise polynomial interpolation of a given order through the order+1 nearest data points
    (Wolfram's default Interpolation method for order 3)."""

    def __init__(self, xs, ys, order):
        self.x = np.asarray(xs, dtype=float)
        self.y = np.asarray(ys)
        self.order = min(order, len(xs) - 1)

    def __call__(self, t, k=0):
        x, n, p = self.x, len(self.x), self.order
        i = int(np.clip(np.searchsorted(x, t, side="right") - 1, 0, n - 2))
        lo = int(np.clip(i - (p - 1) // 2, 0, n - p - 1))
        xs, ys = x[lo:lo + p + 1], self.y[lo:lo + p + 1]
        if k == 0:
            # Lagrange form: exact at the nodes
            tot = 0
            for j in range(len(xs)):
                w = 1.0
                for m in range(len(xs)):
                    if m != j:
                        w *= (t - xs[m]) / (xs[j] - xs[m])
                tot = tot + w * ys[j]
            return tot
        # Lagrange polynomial through the stencil, differentiated k times
        c = np.polyfit(xs - xs[0], ys, p) if p > 0 else np.array([ys[0]])
        poly = np.poly1d(c)
        for _ in range(k):
            poly = poly.deriv()
        return poly(t - xs[0])


class _Ode:
    def __init__(self, sol, comp, rhs, nstate):
        self.sol, self.comp, self.rhs = sol, comp, rhs   # comp: list of state indices for derivatives 0..m-1

    def __call__(self, t, k=0):
        if k < len(self.comp):
            return self.sol(t)[self.comp[k]]
        if k == len(self.comp):
            return self.rhs(t, self.sol(t))
        raise NotImplementedError("InterpolatingFunction: derivatives above the order of the equation are not "
                                  "supported for NDSolve solutions")


def _make(f, a, b):
    key = f"if{next(_ids)}"
    _REG[key] = f
    return Expr("InterpolatingFunction", L(L(float(a), float(b))), key)


def _make_nd(f, bounds):
    """Register a multi-dimensional InterpolatingFunction[{{x1min, x1max}, ...}, id]; f(t1, t2, ...) -> value."""
    key = f"if{next(_ids)}"
    _REG[key] = f
    return Expr("InterpolatingFunction", L(*(L(float(a), float(b)) for a, b in bounds)), key)


def _num_out(v):
    v = complex(v)
    return float(v.real) if v.imag == 0 else R(v)


def apply_ifun(head, args, order=0):
    """ifun[x] (or Derivative[order][ifun][x])."""
    if len(head.args) != 2 or not isinstance(head.args[1], str) or head.args[1] not in _REG:
        raise ValueError("InterpolatingFunction: this function's data is not available in this session")
    dims = len(head.args[0].args) if is_list(head.args[0]) else 1
    if dims > 1:
        return _apply_nd(head, args, order, dims)
    if len(args) != 1:
        raise ValueError("InterpolatingFunction: one-dimensional functions take one argument")
    x = args[0]
    if is_list(x):
        return L(*(apply_ifun(head, [a], order) for a in x.args))
    try:
        t = complex(sp.N(sym(x, "InterpolatingFunction")))
    except Exception:
        return None
    if t.imag != 0:
        raise ValueError("InterpolatingFunction: the argument must be real")
    f = _REG[head.args[1]]
    return _num_out(f(t.real, order))


def _apply_nd(head, args, order, dims):
    """ifun[x1, ..., xd] for a d-dimensional InterpolatingFunction (values only)."""
    if order:
        raise NotImplementedError("InterpolatingFunction: derivatives of multi-dimensional functions are not "
                                  "supported")
    if len(args) != dims:
        raise ValueError(f"InterpolatingFunction: this function takes {dims} arguments")
    ts = []
    for x in args:
        try:
            t = complex(sp.N(sym(x, "InterpolatingFunction")))
        except Exception:
            return None
        if t.imag != 0:
            raise ValueError("InterpolatingFunction: the arguments must be real")
        ts.append(t.real)
    return _num_out(_REG[head.args[1]](*ts))


def _points(data, fname):
    rows = list_arg(data, fname, "data", min_len=2)
    if is_list(rows[0]):
        pts = []
        for r in rows:
            if len(r.args) != 2 or is_list(r.args[1]):
                raise NotImplementedError(f"{fname}: only one-dimensional data {{x, f}} is supported "
                                          "(no derivative data or multidimensional grids)")
            pts.append((complex(sp.N(sym(r.args[0], fname))).real, complex(sp.N(sym(r.args[1], fname)))))
    else:
        pts = [(i + 1.0, complex(sp.N(sym(v, fname)))) for i, v in enumerate(rows)]
    pts.sort(key=lambda p: p[0])
    xs = [p[0] for p in pts]
    if len(set(xs)) != len(xs):
        raise ValueError(f"{fname}: the abscissae must be distinct")
    ys = [p[1] for p in pts]
    if all(y.imag == 0 for y in ys):
        ys = [y.real for y in ys]
    return xs, ys


def _order(opts, fname):
    o = opts.get("InterpolationOrder", 3)
    try:
        o = int(o)
    except Exception:
        raise ValueError(f"{fname}: InterpolationOrder must be a non-negative integer")
    if o < 0:
        raise ValueError(f"{fname}: InterpolationOrder must be a non-negative integer")
    return o


@fn("Interpolation", area=AREA, status="added",
    notes="Interpolation[{f1, f2, ...}] (x = 1, 2, ...) and Interpolation[{{x1, f1}, ...}] -> InterpolatingFunction: "
          "piecewise polynomials of InterpolationOrder (default 3) through the nearest order+1 points "
          "(Wolfram's default method; exact through the data); Interpolation[data, x] evaluates at x; "
          "one-dimensional data only (derivative data and grids raise)")
def _interpolation(data: Expr, *rest: Expr) -> Expr:
    pos, opts = split_options(rest, {"InterpolationOrder", "Method", "PeriodicInterpolation"}, "Interpolation")
    if opts.get("Method") is not None:
        raise NotImplementedError("Interpolation: only the default method is supported")
    xs, ys = _points(data, "Interpolation")
    f = _make(_Piecewise(xs, ys, _order(opts, "Interpolation")), xs[0], xs[-1])
    if pos:
        return ev(Expr(f, pos[0]))
    return f


@fn("ListInterpolation", area=AREA, status="added",
    notes="ListInterpolation[{f1, ..., fn}] on x = 1..n and ListInterpolation[list, {{xmin, xmax}}] on an "
          "equally spaced grid -> InterpolatingFunction (same piecewise polynomials as Interpolation); 1-D only")
def _listinterp(data: ListOf[Expr], *rest: Expr) -> Expr:
    pos, opts = split_options(rest, {"InterpolationOrder", "Method", "PeriodicInterpolation"}, "ListInterpolation")
    vals = list_arg(data, "ListInterpolation", "data", min_len=2)
    if any(is_list(v) for v in vals):
        raise NotImplementedError("ListInterpolation: only one-dimensional data is supported")
    n = len(vals)
    if pos:
        dom = list_arg(pos[0], "ListInterpolation", "domain")
        if len(dom) != 1 or not is_list(dom[0]) or len(dom[0].args) != 2:
            raise ValueError("ListInterpolation: the domain must be {{xmin, xmax}}")
        a, b = (complex(sp.N(sym(v, "ListInterpolation"))).real for v in dom[0].args)
    else:
        a, b = 1.0, float(n)
    xs = list(np.linspace(a, b, n))
    pts = L(*(L(x, v) for x, v in zip(xs, vals)))
    xs2, ys = _points(pts, "ListInterpolation")
    return _make(_Piecewise(xs2, ys, _order(opts, "ListInterpolation")), a, b)


record("InterpolatingFunction", AREA, "added",
       "InterpolatingFunction[{{xmin, xmax}}, id] objects from Interpolation / ListInterpolation / NDSolve: "
       "ifun[x], ifun'[x], Derivative[n][ifun][x], lists of points; real arguments; outside the domain the "
       "end polynomials extrapolate")


# ───────────────────────────── NDSolve ─────────────────────────────

def _fname(u):
    if isinstance(u, Symbol):
        return u.name
    if isinstance(u, Expr) and isinstance(u.head, str) and len(u.args) == 1:
        return u.head
    raise ValueError(f"NDSolve: {u} is not an unknown function")


def _parse_term(e, funcs, x):
    """(name, order, argument) if e is f[arg] or Derivative[k][f][arg] for an unknown f, else None."""
    if isinstance(e, Expr) and len(e.args) == 1:
        if isinstance(e.head, str) and e.head in funcs:
            return e.head, 0, e.args[0]
        h = e.head
        if (isinstance(h, Expr) and isinstance(h.head, Expr) and h.head.head == "Derivative" and len(h.args) == 1
                and isinstance(h.args[0], Symbol) and h.args[0].name in funcs and len(h.head.args) == 1):
            return h.args[0].name, int(h.head.args[0]), e.args[0]
    return None


def _to_sym(e, funcs, x, table):
    t = _parse_term(e, funcs, x)
    if t is not None:
        name, k, arg = t
        if not (isinstance(arg, Symbol) and arg.name == x):
            raise ValueError("NDSolve: unknown functions in the equations must be evaluated at the independent "
                             "variable (delay equations are not supported)")
        s = table.setdefault((name, k), sp.Symbol(f"_{name}_{k}"))
        return Symbol(s.name)
    if isinstance(e, Expr):
        return Expr(e.head if not isinstance(e.head, Expr) else e.head, *(_to_sym(a, funcs, x, table) for a in e.args))
    if isinstance(e, Symbol) and e.name in funcs:
        raise ValueError(f"NDSolve: {e.name} must appear as {e.name}[{x}]")
    return e


def _ndsolve(eqs, unknowns, rng, opts, fname):
    from .wl_pde import ndsolve_bvp
    extra = list(opts.get("ranges", ()))
    r = ndsolve_bvp(eqs, unknowns, [rng] + extra, fname)
    if r is not None:
        return r
    if extra:
        raise ValueError(f"{fname}: unexpected arguments {extra}")
    try:
        import scipy.integrate as si
    except ImportError:
        raise NotImplementedError(f"{fname} requires SciPy (pip install mikoshilang[signal])")
    us = list(unknowns.args) if is_list(unknowns) else [unknowns]
    funcs = [_fname(u) for u in us]
    it = list_arg(rng, fname, "range")
    if len(it) == 2:
        it = [it[0], 0, it[1]]
    if len(it) != 3:
        raise ValueError(f"{fname}: the range must be {{x, xmin, xmax}}")
    xv = variable(it[0], fname).name
    a, b = (float(complex(sp.N(sym(v, fname))).real) for v in it[1:])
    items = list(eqs.args) if is_list(eqs) else [eqs]
    flat = []
    for e in items:
        if isinstance(e, Expr) and e.head == "And":
            items.extend(e.args)
        elif isinstance(e, Expr) and e.head == "Equal" and (len(e.args) > 2 or any(
                isinstance(a, Expr) and a.head == "Equal" for a in e.args)):
            # chained x[0] == y[0] == 1: every side equals the last one
            def ops(q):
                if isinstance(q, Expr) and q.head == "Equal":
                    return [o for a in q.args for o in ops(a)]
                return [q]
            sides = ops(e)
            flat.extend(Expr("Equal", a, sides[-1]) for a in sides[:-1])
        else:
            flat.append(e)
    odes, ics = [], []
    for e in flat:
        if not (isinstance(e, Expr) and e.head == "Equal" and len(e.args) == 2):
            raise ValueError(f"{fname}: {e} is not an equation")
        lhs = _parse_term(e.args[0], funcs, xv)
        if lhs is not None and not (isinstance(lhs[2], Symbol) and lhs[2].name == xv):
            ics.append((lhs, e.args[1]))
        else:
            odes.append(e)
    if len(odes) != len(funcs):
        raise ValueError(f"{fname}: expected {len(funcs)} differential equation(s), got {len(odes)} "
                         "(boundary-value problems, DAEs and PDEs are not supported)")
    table = {}
    exprs = [sym(_to_sym(Expr("Plus", e.args[0], Expr("Times", -1, e.args[1])), funcs, xv, table), fname)
             for e in odes]
    order = {f: max([k for (g, k) in table if g == f] + [0]) for f in funcs}
    for f in funcs:
        if order[f] == 0:
            raise ValueError(f"{fname}: no derivative of {f} appears in the equations")
    tops = [table[(f, order[f])] for f in funcs]
    try:
        sol = sp.solve(exprs, tops, dict=True)
    except Exception:
        sol = []
    if len(sol) != 1:
        raise NotImplementedError(f"{fname}: the equations could not be solved for the highest derivatives "
                                  "(implicit or differential-algebraic systems are not supported)")
    sol = sol[0]
    # state vector: for each f, derivatives 0..order-1
    state, comps = [], {}
    for f in funcs:
        comps[f] = []
        for k in range(order[f]):
            comps[f].append(len(state))
            state.append(table.setdefault((f, k), sp.Symbol(f"_{f}_{k}")))
    tsym = sp.Symbol(xv)
    derivs = []
    for f in funcs:
        for k in range(order[f]):
            derivs.append(state[comps[f][k + 1]] if k + 1 < order[f] else sol[table[(f, order[f])]])
    rhs_l = sp.lambdify([tsym] + state, derivs, "numpy")

    def rhs(t, y):
        return np.array(rhs_l(t, *y), dtype=complex if np.iscomplexobj(y) else float)
    # initial conditions
    y0 = [None] * len(state)
    t0 = None
    # resolve conditions that equate two unknown values (x[0] == y[0] == 1)
    known = {}
    for _ in range(len(ics) + 1):
        for (f, k, arg), val in ics:
            other = _parse_term(val, funcs, xv)
            key = (f, k, str(arg))
            if other is None:
                known.setdefault(key, val)
            else:
                okey = (other[0], other[1], str(other[2]))
                if okey in known:
                    known.setdefault(key, known[okey])
                if key in known:
                    known.setdefault(okey, known[key])
    extra = []
    seen = {(f, k, str(a)) for (f, k, a), _ in ics}
    for (f, k, arg), val in ics:
        other = _parse_term(val, funcs, xv)
        if other is not None and (other[0], other[1], str(other[2])) not in seen:
            extra.append((other, val))
            seen.add((other[0], other[1], str(other[2])))
    resolved = []
    for (f, k, arg), _val in ics + extra:
        key = (f, k, str(arg))
        if key not in known:
            raise ValueError(f"{fname}: the initial value of {f} could not be determined")
        resolved.append(((f, k, arg), known[key]))
    for (f, k, arg), val in resolved:
        if k >= order[f]:
            raise ValueError(f"{fname}: an initial condition for derivative {k} of {f} is too high for the equation")
        tt = float(complex(sp.N(sym(arg, fname))).real)
        if t0 is None:
            t0 = tt
        elif abs(tt - t0) > 1e-14:
            raise NotImplementedError(f"{fname}: boundary-value problems (conditions at different points) are "
                                      "not supported")
        v = complex(sp.N(sym(val, fname)))
        y0[comps[f][k]] = v.real if v.imag == 0 else v
    if any(v is None for v in y0):
        raise ValueError(f"{fname}: {len(state)} initial conditions are required")
    if not (min(a, b) - 1e-12 <= t0 <= max(a, b) + 1e-12):
        raise ValueError(f"{fname}: the initial point must lie in the integration range")
    y0 = np.array(y0, dtype=complex if any(isinstance(v, complex) for v in y0) else float)
    pieces = []
    for end in (a, b):
        if abs(end - t0) < 1e-15:
            continue
        r = None
        for method in ("DOP853", "Radau"):
            r = si.solve_ivp(rhs, (t0, end), y0, method=method, dense_output=True, rtol=1e-10, atol=1e-12)
            if r.success:
                break
        if not r.success:
            raise ValueError(f"{fname}: integration failed ({r.message}) near {fname} x = {r.t[-1]}")
        pieces.append((min(t0, end), max(t0, end), r.sol))

    def dense(t):
        for lo, hi, s in pieces:
            if lo - 1e-12 <= t <= hi + 1e-12:
                return s(t)
        lo, hi, s = min(pieces, key=lambda p: min(abs(t - p[0]), abs(t - p[1])))
        return s(t)
    out = {}
    for f in funcs:
        def rhs_k(t, y, f=f):
            return rhs(t, y)[comps[f][-1]]
        out[f] = _make(_Ode(dense, comps[f], rhs_k, len(state)), a, b)
    return us, out, xv


def _ndopts(rest, fname):
    pos, opts = split_options(rest, {"WorkingPrecision", "Method", "MaxSteps", "PrecisionGoal", "AccuracyGoal"},
                              fname)
    wp = opts.get("WorkingPrecision")
    if wp is not None and not (isinstance(wp, Symbol) and wp.name == "MachinePrecision"):
        raise NotImplementedError(f"{fname}: only machine precision is supported")
    return pos


@fn("NDSolve", area=AREA, status="added",
    notes="NDSolve[{odes, ics}, y, {x, xmin, xmax}] (also y[x], and systems {y, z}) for initial-value problems "
          "of any order that can be solved for the highest derivatives -> {{y -> InterpolatingFunction[...]}} "
          "(or {{y[x] -> InterpolatingFunction[...][x]}}); SciPy DOP853 (Radau fallback for stiff problems), "
          "rtol 1e-10; the initial point may be inside the range; machine precision only; boundary-value "
          "problems, DAEs, PDEs and delay equations raise")
def _ndsolve_fn(eqs: Expr, unknowns: Expr, rng: Expr, *rest: Expr) -> ListOf[Expr]:
    pos = _ndopts(rest, "NDSolve")
    us, out, xv = _ndsolve(eqs, unknowns, rng, {"ranges": pos}, "NDSolve")
    rules = []
    for u in us:
        f = _fname(u)
        if isinstance(u, Symbol):
            rules.append(rule(u, out[f]))
        else:
            rules.append(rule(u, Expr(out[f], *u.args)))
    return L(L(*rules))


@fn("NDSolveValue", area=AREA, status="added",
    notes="NDSolveValue[{odes, ics}, y, {x, xmin, xmax}] -> InterpolatingFunction (y[x] -> ifun[x], {y, z} -> "
          "list); same solver and limits as NDSolve")
def _ndsolvevalue(eqs: Expr, unknowns: Expr, rng: Expr, *rest: Expr) -> Expr:
    pos = _ndopts(rest, "NDSolveValue")
    us, out, xv = _ndsolve(eqs, unknowns, rng, {"ranges": pos}, "NDSolveValue")
    vals = [out[_fname(u)] if isinstance(u, Symbol) else ev(Expr(out[_fname(u)], *u.args)) for u in us]
    return L(*vals) if is_list(unknowns) else vals[0]
