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
    events = [e for e in flat if isinstance(e, Expr) and e.head == "WhenEvent"]
    flat = [e for e in flat if not (isinstance(e, Expr) and e.head == "WhenEvent")]
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
    ctl = _ivp_controls(opts, a, b, fname)
    subst = _Subst(funcs, comps, order, xv, rhs, table, state, fname)
    evs = [_Event(e, subst, funcs, xv, fname) for e in events]
    mon_step, mon_eval = opts.get("StepMonitor"), opts.get("EvaluationMonitor")
    if mon_eval is not None:
        base_rhs = rhs

        def rhs(t, y):      # noqa: F811 - EvaluationMonitor wraps every right-hand-side evaluation
            _eval_with(mon_eval, subst.rules(t, y))
            return base_rhs(t, y)
    pieces = []
    ends = [end for end in (a, b) if abs(end - t0) >= 1e-15]
    if evs and len(ends) > 1:
        raise NotImplementedError(f"{fname}: WhenEvent needs the initial point at one end of the range")
    lo_hi = [a, b]
    for end in ends:
        segs, stop = _integrate(si, rhs, t0, end, y0, ctl, evs, subst, mon_step, fname)
        pieces.extend(segs)
        if stop is not None:
            lo_hi = [min(t0, stop), max(t0, stop)]

    def dense(t):
        for lo, hi, s in pieces:
            if lo - 1e-12 <= t <= hi + 1e-12:
                return s(t)
        lo, hi, s = min(pieces, key=lambda p: min(abs(t - p[0]), abs(t - p[1])))
        return s(t)
    a, b = lo_hi
    out = {}
    for f in funcs:
        def rhs_k(t, y, f=f):
            return rhs(t, y)[comps[f][-1]]
        out[f] = _make(_Ode(dense, comps[f], rhs_k, len(state)), a, b)
    return us, out, xv


# ───────────────────────────── NDSolve step control, monitors and WhenEvent (W26) ─────────────────────────────

_METHODS = {"ExplicitRungeKutta": ("DOP853",), "ImplicitRungeKutta": ("Radau",), "BDF": ("BDF",),
            "LSODA": ("LSODA",), "StiffnessSwitching": ("LSODA",)}


def _is_auto(v):
    return v is None or (isinstance(v, Symbol) and v.name == "Automatic")


def _is_inf(v):
    return (isinstance(v, Symbol) and v.name == "Infinity") or (isinstance(v, Expr) and v.head == "DirectedInfinity")


def _posnum(v, name, fname, integer=False):
    try:
        z = complex(sp.N(sym(v, fname)))
    except Exception:
        raise ValueError(f"{fname}: {name} must be a positive number, Infinity or Automatic")
    if z.imag != 0 or not z.real > 0 or (integer and z.real != int(z.real)):
        raise ValueError(f"{fname}: {name} must be a positive {'integer' if integer else 'number'}, Infinity or "
                         "Automatic")
    return z.real


def _ivp_controls(opts, a, b, fname):
    """solve_ivp settings from AccuracyGoal, PrecisionGoal, MaxStepSize, MaxStepFraction, StartingStepSize,
    MaxSteps and Method."""
    m = opts.get("Method")
    if _is_auto(m):
        methods = ("DOP853", "Radau")
    elif isinstance(m, str) and m in _METHODS:
        methods = _METHODS[m]
    else:
        raise NotImplementedError(f"{fname}: Method -> {m} is not supported (Automatic, " +
                                  ", ".join(f'"{k}"' for k in _METHODS) + ")")
    ctl = {"methods": methods, "rtol": 1e-10, "atol": 1e-12, "max_step": np.inf, "first_step": None,
           "max_steps": None}
    ag, pg = opts.get("AccuracyGoal"), opts.get("PrecisionGoal")
    if not _is_auto(ag):
        ctl["atol"] = 1e-300 if _is_inf(ag) else 10.0 ** (-_posnum(ag, "AccuracyGoal", fname))
    if not _is_auto(pg):
        ctl["rtol"] = 2.3e-14 if _is_inf(pg) else max(10.0 ** (-_posnum(pg, "PrecisionGoal", fname)), 2.3e-14)
    ms = opts.get("MaxStepSize")
    if not _is_auto(ms) and not _is_inf(ms):
        ctl["max_step"] = _posnum(ms, "MaxStepSize", fname)
    mf = opts.get("MaxStepFraction")
    if not _is_auto(mf) and not _is_inf(mf):
        f = _posnum(mf, "MaxStepFraction", fname)
        if f > 1:
            raise ValueError(f"{fname}: MaxStepFraction must be at most 1")
        ctl["max_step"] = min(ctl["max_step"], f * abs(b - a))
    st = opts.get("StartingStepSize")
    if not _is_auto(st):
        ctl["first_step"] = _posnum(st, "StartingStepSize", fname)
    mx = opts.get("MaxSteps")
    if not _is_auto(mx) and not _is_inf(mx):
        ctl["max_steps"] = int(_posnum(mx, "MaxSteps", fname, integer=True))
    return ctl


def _eval_with(e, rules):
    """Evaluate the held expression e after substituting the rules (monitor and event actions)."""
    held = ev(Expr("ReplaceAll", Expr("Hold", e), rules))
    return ev(held.args[0]) if isinstance(held, Expr) and held.head == "Hold" and len(held.args) == 1 else ev(held)


class _Subst:
    """Replacement rules {x -> t, y[x] -> value, y'[x] -> value, ...} for a state vector, and lambdified
    expressions of the state."""

    def __init__(self, funcs, comps, order, xv, rhs, table, state, fname):
        self.funcs, self.comps, self.order, self.xv = funcs, comps, order, xv
        self.rhs, self.table, self.state, self.fname = rhs, table, state, fname

    def rules(self, t, y):
        d = self.rhs(t, y)
        x = Symbol(self.xv)
        out = []
        for f in self.funcs:
            for k in range(self.order[f] + 1):
                v = y[self.comps[f][k]] if k < self.order[f] else d[self.comps[f][-1]]
                lhs = Expr(f, x) if k == 0 else Expr(Expr(Expr("Derivative", k), Symbol(f)), x)
                out.append(Expr("Rule", lhs, _num_out(v)))
        out.append(Expr("Rule", x, float(t)))
        return L(*out)

    def lambdify(self, e):
        tab = dict(self.table)
        se = sym(_to_sym(e, self.funcs, self.xv, tab), self.fname)
        tops = [tab[(f, self.order[f])] for f in self.funcs]
        allowed = {sp.Symbol(self.xv)} | set(self.state) | set(tops)
        if not se.free_symbols <= allowed:
            raise ValueError(f"{self.fname}: the event {e} involves unknown symbols "
                             f"{', '.join(sorted(map(str, se.free_symbols - allowed)))}")
        fl = sp.lambdify([sp.Symbol(self.xv)] + list(self.state) + tops, se, "numpy")

        def g(t, y):
            d = self.rhs(t, y)
            return float(np.real(fl(t, *y, *[d[self.comps[f][-1]] for f in self.funcs])))
        return g


_EVENT_DIR = {"Equal": 0, "Greater": 1, "GreaterEqual": 1, "Less": -1, "LessEqual": -1}


class _Event:
    """WhenEvent[condition, action]: condition a == b (crossing either way) or a > b / a < b (becoming True);
    actions "StopIntegration", "RemoveEvent", y[x] -> value / y'[x] -> value resets, lists of these, or any
    expression evaluated for its side effects."""

    def __init__(self, we, subst, funcs, xv, fname):
        if len(we.args) != 2:
            raise NotImplementedError(f"{fname}: WhenEvent[event, action] with options is not supported")
        cond, self.action = we.args
        cond = ev(cond)
        if not (isinstance(cond, Expr) and cond.head in _EVENT_DIR and len(cond.args) == 2):
            raise NotImplementedError(f"{fname}: WhenEvent conditions must be equations or inequalities a == b, "
                                      f"a > b, a < b (got {cond})")
        self.direction = _EVENT_DIR[cond.head]
        self.g = subst.lambdify(Expr("Plus", cond.args[0], Expr("Times", -1, cond.args[1])))
        self.subst, self.funcs, self.xv, self.fname = subst, funcs, xv, fname

    def scipy_fn(self):
        def f(t, y):
            return self.g(t, y)
        f.terminal = True
        f.direction = self.direction
        return f

    def fire(self, t, y):
        """Apply the action at (t, y): ("stop" | "remove" | None, new state)."""
        acts = list(self.action.args) if is_list(self.action) else [self.action]
        rules = self.subst.rules(t, y)
        ynew = np.array(y, copy=True)
        res = None
        for act in acts:
            if isinstance(act, str):
                if act == "StopIntegration":
                    res = "stop"
                elif act == "RemoveEvent":
                    res = res or "remove"
                elif act not in ("CrossDiscontinuity", "RestartIntegration"):
                    raise NotImplementedError(f"{self.fname}: the WhenEvent action \"{act}\" is not supported")
                continue
            if isinstance(act, Expr) and act.head in ("Rule", "RuleDelayed") and len(act.args) == 2:
                term = _parse_term(act.args[0], self.funcs, self.xv)
                if term is None or not (isinstance(term[2], Symbol) and term[2].name == self.xv):
                    raise NotImplementedError(f"{self.fname}: WhenEvent rules must reset y[{self.xv}] or a "
                                              "derivative of it (discrete variables are not supported)")
                f, k, _ = term
                if k >= len(self.subst.comps[f]):
                    raise ValueError(f"{self.fname}: cannot reset derivative {k} of {f} (the equation is of order "
                                     f"{len(self.subst.comps[f])})")
                v = _eval_with(act.args[1], rules)
                try:
                    z = complex(sp.N(sym(v, self.fname)))
                except Exception:
                    raise ValueError(f"{self.fname}: the WhenEvent reset value {v} is not numerical")
                if np.iscomplexobj(ynew) or z.imag == 0:
                    ynew[self.subst.comps[f][k]] = z if np.iscomplexobj(ynew) else z.real
                else:
                    ynew = ynew.astype(complex)
                    ynew[self.subst.comps[f][k]] = z
                continue
            _eval_with(act, rules)
        return res, ynew


def _solve(si, rhs, t, end, y, ctl, events, fname):
    kw = {"rtol": ctl["rtol"], "atol": ctl["atol"], "dense_output": True, "max_step": ctl["max_step"]}
    if ctl["first_step"] is not None:
        if ctl["first_step"] > abs(end - t):
            raise ValueError(f"{fname}: StartingStepSize is larger than the integration range")
        kw["first_step"] = ctl["first_step"]
    r = None
    for method in ctl["methods"]:
        r = si.solve_ivp(rhs, (t, end), y, method=method, events=events, **kw)
        if r.success:
            return r
    raise ValueError(f"{fname}: integration failed ({r.message}) near x = {r.t[-1]}")


def _integrate(si, rhs, t0, end, y0, ctl, evs, subst, mon_step, fname):
    """Integrate from t0 to end, handling WhenEvent actions; ([(lo, hi, dense)], stop point or None)."""
    segs, t, y, steps, nev = [], t0, np.array(y0), 0, 0
    active = list(evs)
    span = max(1.0, abs(end - t0))
    while True:
        r = _solve(si, rhs, t, end, y, ctl, [e.scipy_fn() for e in active] or None, fname)
        steps += len(r.t) - 1
        if ctl["max_steps"] is not None and steps > ctl["max_steps"]:
            raise ValueError(f"{fname}: the maximum number of steps (MaxSteps -> {ctl['max_steps']}) was reached "
                             f"before x = {end}")
        if mon_step is not None:
            for tt, yy in zip(r.t[1:], r.y.T[1:]):
                _eval_with(mon_step, subst.rules(tt, yy))
        if r.status != 1:
            segs.append((min(t, end), max(t, end), r.sol))
            return segs, None
        hits = [(abs(te[0] - t), i, te[0], ye[0]) for i, (te, ye) in enumerate(zip(r.t_events, r.y_events))
                if len(te)]
        _, k, te, ye = min(hits, key=lambda h: h[0])
        segs.append((min(t, te), max(t, te), r.sol))
        nev += 1
        if nev > 10000:
            raise ValueError(f"{fname}: more than 10000 events were detected (Zeno behaviour?)")
        act, ynew = active[k].fire(te, ye)
        if act == "stop":
            return segs, te
        if act == "remove":
            active.pop(k)
        delta = np.sign(end - t0) * 1e-10 * span
        if abs(end - te) <= abs(delta) * 2:
            segs.append((min(te, end), max(te, end), lambda _t, v=ynew: v))
            return segs, None
        r2 = _solve(si, rhs, te, te + delta, ynew, dict(ctl, first_step=None), None, fname)
        segs.append((min(te, te + delta), max(te, te + delta), r2.sol))
        t, y = te + delta, r2.y[:, -1]


def _ndopts(rest, fname):
    pos, opts = split_options(rest, {"WorkingPrecision", "Method", "MaxSteps", "PrecisionGoal", "AccuracyGoal",
                                     "MaxStepSize", "MaxStepFraction", "StartingStepSize", "StepMonitor",
                                     "EvaluationMonitor"}, fname)
    wp = opts.get("WorkingPrecision")
    if wp is not None and not (isinstance(wp, Symbol) and wp.name == "MachinePrecision"):
        raise NotImplementedError(f"{fname}: only machine precision is supported")
    return pos, opts


@fn("NDSolve", area=AREA, status="added",
    notes="NDSolve[{odes, ics}, y, {x, xmin, xmax}] (also y[x], and systems {y, z}) for initial-value problems "
          "of any order that can be solved for the highest derivatives -> {{y -> InterpolatingFunction[...]}} "
          "(or {{y[x] -> InterpolatingFunction[...][x]}}); SciPy DOP853 (Radau fallback for stiff problems), "
          "rtol 1e-10; the initial point may be inside the range; machine precision only; boundary-value "
          "problems, DAEs, PDEs and delay equations raise")
def _ndsolve_fn(eqs: Expr, unknowns: Expr, rng: Expr, *rest: Expr) -> ListOf[Expr]:
    pos, opts = _ndopts(rest, "NDSolve")
    us, out, xv = _ndsolve(eqs, unknowns, rng, dict(opts, ranges=pos), "NDSolve")
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
    pos, opts = _ndopts(rest, "NDSolveValue")
    us, out, xv = _ndsolve(eqs, unknowns, rng, dict(opts, ranges=pos), "NDSolveValue")
    vals = [out[_fname(u)] if isinstance(u, Symbol) else ev(Expr(out[_fname(u)], *u.args)) for u in us]
    return L(*vals) if is_list(unknowns) else vals[0]
