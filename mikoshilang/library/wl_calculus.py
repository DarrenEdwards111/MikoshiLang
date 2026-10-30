"""Coverage batch 4: symbolic calculus with Wolfram semantics.

Differentiation (D, Dt, Derivative, ImplicitD), integration (Integrate with GenerateConditions, PrincipalValue
and Assumptions), limits (Limit with Direction and several variables, MaxLimit, MinLimit), series
(Series with several variables, Laurent and Puiseux terms and expansions at Infinity, SeriesCoefficient,
InverseSeries, ComposeSeries, Normal, O), sums (SumConvergence), recurrences and ODEs (RSolve, RSolveValue,
DSolve, DSolveValue) and the generalized functions DiracDelta and HeavisideTheta.

Every result is either computed by an exact SymPy algorithm whose output is checked (antiderivatives are
differentiated back, ODE and recurrence solutions are substituted back) or the call stays unevaluated / raises
a clear error.  Kernel builtins (D, Limit, Series, Normal) are replaced with ``kernel_override`` so kernel.py is
untouched.
"""
import sympy as sp
from sympy.core.function import AppliedUndef

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, record, R, L, rule, sym, variable, is_list, Unevaluated, numeric_out, nonneg_int, int_arg
from .wl_symbolic import (AREA_CALC as AREA, kernel_override, replace, out, wl_fix, to_heaviside, split_opts,
                          is_true, is_false, is_symbol, is_rule, HeavisideTheta)

_Derivative = "Derivative"


def _ev(x):
    from ..evaluate import evaluate
    return evaluate(x, 1)


def _S(x, fname):
    if isinstance(x, str):
        raise ValueError(f"{fname}: strings are not expressions")
    return K.to_sp(x)


def _var(x, fname):
    v = _S(x, fname)
    if isinstance(v, (sp.Symbol, AppliedUndef)):
        return v
    raise ValueError(f"{fname}: {x} is not a valid variable")


# ───────────────────────────── generalized functions ─────────────────────────────

def _real_num(v):
    return isinstance(v, sp.Basic) and v.is_number and v.is_extended_real is not None


@fn("DiracDelta", area=AREA, status="added",
    notes="Wolfram DiracDelta[x1, ...]: 0 when any argument is a nonzero real number, unevaluated otherwise "
          "(DiracDelta[0] stays); complex numeric arguments raise; integrates and transforms through SymPy "
          "(Integrate[f[x] DiracDelta[x - a], {x, -Infinity, Infinity}] -> f[a]); D[HeavisideTheta[x], x] -> DiracDelta[x]")
def _diracdelta(*args):
    vals = [sym(a, "DiracDelta") for a in args]
    for v in vals:
        if v.is_number and v.is_extended_real is False:
            raise ValueError("DiracDelta is defined for real arguments only")
    if any(v.is_number and v != 0 for v in vals):
        return 0
    raise Unevaluated()


@fn("HeavisideTheta", area=AREA, status="added",
    notes="Wolfram HeavisideTheta[x1, ...]: 1 when every argument is positive, 0 when any is negative, "
          "unevaluated at 0 (unlike UnitStep[0] = 1 and SymPy's Heaviside(0) = 1/2); D gives DiracDelta")
def _heaviside(*args):
    vals = [sym(a, "HeavisideTheta") for a in args]
    for v in vals:
        if v.is_number and v.is_extended_real is False:
            raise ValueError("HeavisideTheta is defined for real arguments only")
    if any(v.is_number and v.is_extended_negative for v in vals):
        return 0
    if all(v.is_number and v.is_extended_positive for v in vals):
        return 1
    raise Unevaluated()


@fn("ConditionalExpression", area=AREA, status="added",
    notes="ConditionalExpression[expr, cond]: expr when cond is True, Undefined when cond is False, otherwise "
          "unevaluated (returned by Integrate and the Fourier coefficient functions for parameter conditions)")
def _condexpr(e, c):
    if is_true(c):
        return e
    if is_false(c):
        return Symbol("Undefined")
    raise Unevaluated()


# ───────────────────────────── D ─────────────────────────────

def _diff_one(f, var, n):
    """d^n f / d var^n in SymPy."""
    return sp.diff(f, var, n) if n else f


def _nonconst_D(v, x, nonconst, opts_expr):
    """Differentiate treating the symbols in ``nonconst`` as functions of x; their derivatives become
    D[y, x, NonConstants -> {...}] as in Wolfram."""
    funcs = {sp.Symbol(s): sp.Function(s)(x) for s in nonconst}
    g = sp.diff(v.xreplace(funcs), x)
    marks = {}
    for s, F in funcs.items():
        m = sp.Symbol(f"__dnc_{s.name}")
        marks[m.name] = Expr("D", Symbol(s.name), Symbol(x.name), opts_expr)
        g = g.subs(sp.Derivative(F, x), m)
    g = g.xreplace({F: s for s, F in funcs.items()})
    return K.substitute(out(g), marks)


def D_expr(f, specs, nonconst=None, opts_expr=None):
    """Wolfram D on MikoshiLang expressions (threads over lists and equations)."""
    if is_list(f):
        return L(*(D_expr(g, specs, nonconst, opts_expr) for g in f.args))
    if isinstance(f, Expr) and f.head in ("Equal", "Unequal", "Less", "Greater", "LessEqual", "GreaterEqual") \
            and f.head == "Equal":
        return Expr("Equal", *(D_expr(g, specs, nonconst, opts_expr) for g in f.args))
    if not specs:
        return f
    spec, rest = specs[0], specs[1:]
    if is_list(spec) and spec.args and is_list(spec.args[0]):
        if len(spec.args) > 2:
            raise ValueError("D: array specifications must be {{x1, x2, ...}} or {{x1, x2, ...}, n}")
        vs = list(spec.args[0].args)
        n = nonneg_int(spec.args[1], "D", "order") if len(spec.args) == 2 else 1

        def arr(g, k):
            if k == 0:
                return g
            return L(*(arr(D_expr(g, [v], nonconst, opts_expr), k - 1) for v in vs))
        return D_expr(arr(f, n), rest, nonconst, opts_expr)
    if is_list(spec):
        if len(spec.args) != 2:
            raise ValueError("D: a derivative specification must be x, {x, n} or {{x1, x2, ...}}")
        var, n = spec.args
        if not K.is_int(n):
            if isinstance(n, (Symbol, Expr)):
                raise Unevaluated()          # symbolic order: D[f, {x, n}] stays unevaluated
            raise ValueError("D: the order must be a non-negative integer")
        n = int(n)
        if n < 0:
            raise ValueError("D: the order must be a non-negative integer")
    else:
        var, n = spec, 1
    if K.is_number(var) or var is True or var is False or isinstance(var, str):
        raise ValueError(f"D: {var} is not a valid variable")
    sv = _var(var, "D")
    v = _S(f, "D")
    if isinstance(v, list):
        raise ValueError("D: cannot differentiate a nested list")
    if nonconst and isinstance(sv, sp.Symbol):
        g = f
        for _ in range(n):
            g = _nonconst_D(_S(g, "D"), sv, nonconst, opts_expr)
        return D_expr(g, rest, nonconst, opts_expr)
    g = out(_diff_one(v, sv, n))
    return D_expr(g, rest, nonconst, opts_expr)


@kernel_override("D", AREA,
                 "batch 4: D[f, x, y, ...], D[f, {x, n}], D[f, {{x1, x2, ...}}] (gradient) and {{...}, n} (arrays of "
                 "higher derivatives), threads over lists and equations (implicit differentiation "
                 "D[x^2 + y[x]^2 == 1, x]), NonConstants -> {y}, derivatives with respect to f[x]; chain rule "
                 "results are Wolfram Derivative forms (D[f[g[x]], x] -> f'[g[x]] g'[x]; previously "
                 "D[f[g[x]], g[x]] g'[x]); symbolic orders stay unevaluated; D[f] with no variable returns f")
def _b_D(a, ev, expr):
    if not a:
        raise ValueError("D needs an expression")
    pos, opts = split_opts(a, {"NonConstants"}, "D")
    nc = None
    optexpr = None
    if "NonConstants" in opts:
        v = opts["NonConstants"]
        items = v.args if is_list(v) else (v,)
        if not all(isinstance(s, Symbol) for s in items):
            raise ValueError("D: NonConstants must be a list of symbols")
        nc = [s.name for s in items]
        optexpr = rule(Symbol("NonConstants"), v)
    return D_expr(pos[0], pos[1:], nc, optexpr)


# ───────────────────────────── Dt ─────────────────────────────

_NUMERIC_CONSTANTS = {"Pi", "E", "I", "EulerGamma", "GoldenRatio", "Catalan", "Degree", "Infinity"}


def _dt(f, spec, constants):
    v = _S(f, "Dt")
    if isinstance(v, list):
        return L(*(_dt(g, spec, constants) for g in f.args))
    if spec is None:
        if isinstance(f, Symbol) and f.name not in constants and f.name not in _NUMERIC_CONSTANTS:
            raise Unevaluated()
        syms = sorted((s for s in v.free_symbols if s.name not in constants), key=str)
        total = sp.Integer(0)
        marks = {}
        for s in syms:
            m = sp.Symbol(f"__dt_{s.name}")
            marks[m.name] = Expr("Dt", Symbol(s.name))
            total += sp.diff(v, s) * m
        res = out(total)
        return K.substitute(res, marks) if marks else res
    x, n = spec
    if isinstance(f, Symbol) and f.name != x.name and f.name not in constants \
            and f.name not in _NUMERIC_CONSTANTS:
        raise Unevaluated()
    others = [s for s in v.free_symbols if s != x and s.name not in constants]
    funcs = {s: sp.Function(s.name)(x) for s in others}
    g = sp.diff(v.xreplace(funcs), x, n)
    marks = {}
    for s, F in funcs.items():
        for k in range(n, 0, -1):
            m = sp.Symbol(f"__dt_{s.name}_{k}")
            marks[m.name] = Expr("Dt", Symbol(s.name), Symbol(x.name) if k == 1 else L(Symbol(x.name), k))
            g = g.subs(sp.Derivative(F, (x, k)), m)
    g = g.xreplace({F: s for s, F in funcs.items()})
    res = out(g)
    return K.substitute(res, marks) if marks else res


@fn("Dt", area=AREA, status="added",
    notes="total derivatives: Dt[f, x] treats every other symbol as a function of x (Dt[x^2 y, x] -> "
          "2 x y + x^2 Dt[y, x]), Dt[f, {x, n}], Dt[f] is the total differential (Dt[x y] -> y Dt[x] + x Dt[y]); "
          "Constants -> {a, ...}; Dt[y, x] for an independent symbol y stays unevaluated; numeric constants "
          "(Pi, E, ...) have derivative 0")
def _Dt(f, *args):
    pos, opts = split_opts(args, {"Constants"}, "Dt")
    consts = set()
    if "Constants" in opts:
        c = opts["Constants"]
        items = c.args if is_list(c) else (c,)
        if not all(isinstance(s, Symbol) for s in items):
            raise ValueError("Dt: Constants must be a list of symbols")
        consts = {s.name for s in items}
    if not pos:
        return _dt(f, None, consts)
    if len(pos) > 1:
        g = f
        for p in pos:
            g = _ev(Expr("Dt", g, p, *(rule(Symbol(k), v) for k, v in opts.items())))
        return g
    spec = pos[0]
    if is_list(spec):
        if len(spec.args) != 2 or not isinstance(spec.args[0], Symbol):
            raise ValueError("Dt: specification must be x or {x, n}")
        n = nonneg_int(spec.args[1], "Dt", "order")
        x = sp.Symbol(spec.args[0].name)
    elif isinstance(spec, Symbol):
        x, n = sp.Symbol(spec.name), 1
    else:
        raise ValueError(f"Dt: {spec} is not a valid variable")
    if n == 0:
        return f
    return _dt(f, (x, n), consts)


# ───────────────────────────── Derivative ─────────────────────────────

def _is_known_function(f):
    """True for heads SymPy can differentiate (Sin, Exp, BesselJ, ...), user-defined functions and pure functions."""
    if isinstance(f, Expr) and f.head == "Function":
        return True
    if not isinstance(f, Symbol):
        return False
    st = K.state()
    if f.name in st.down or any(l.head == f.name for l, _ in st.exact.values()):
        return True
    try:
        v = K.to_sp(Expr(f.name, Symbol("x")))
    except Exception:
        return False
    return isinstance(v, sp.Basic) and not isinstance(v, AppliedUndef) and v.has(sp.Symbol("x")) \
        and not isinstance(v, sp.Symbol)


def apply_derivative(head, args, ev):
    """Derivative[n...][f][args] for pure functions, user definitions and built-in mathematical functions
    (Sin'[x] -> Cos[x]); unknown f stays symbolic."""
    orders = head.head.args
    f = head.args[0] if head.args else None
    if not all(K.is_int(o) for o in orders) or len(orders) != len(args):
        return K.UNHANDLED
    if not _is_known_function(f):
        return K.UNHANDLED
    dummies = [Symbol(f"$d{i}") for i in range(len(args))]
    body = ev(Expr(K.as_head(f), *dummies))
    st = K.state()
    user = isinstance(f, Symbol) and (f.name in st.down or any(l.head == f.name for l, _ in st.exact.values()))
    if user and K.contains(body, lambda x: isinstance(x, Expr) and x.head == f.name):
        return K.UNHANDLED
    spec = [L(d, int(n)) for d, n in zip(dummies, orders) if int(n)]
    try:
        d = ev(Expr("D", body, *spec)) if spec else body
    except (ValueError, TypeError, NotImplementedError):
        return K.UNHANDLED
    if isinstance(f, Symbol) and K.contains(d, lambda x: isinstance(x, Expr) and isinstance(x.head, Expr)
                                            and x.head.head is not None and isinstance(x.head.head, Expr)
                                            and x.head.head.head == _Derivative and x.head.args == (f,)):
        return K.UNHANDLED
    return ev(K.substitute(d, {s.name: a for s, a in zip(dummies, args)}))


K.apply_derivative = apply_derivative


def derivative_of_function(orders, f, ev):
    """The pure function Derivative[n...][f] (None when f is unknown)."""
    if all(int(o) == 0 for o in orders):
        return f
    if not _is_known_function(f):
        return None
    dummies = [Symbol(f"$s{i}") for i in range(len(orders))]
    r = apply_derivative(Expr(Expr(_Derivative, *orders), f), dummies, ev)
    if r is K.UNHANDLED:
        return None
    body = K.substitute(r, {d.name: Expr("Slot", i + 1) for i, d in enumerate(dummies)})
    return Expr("Function", body)


_orig_compound = KL.apply_compound_head


def apply_compound_head(head, args, ev):
    if isinstance(head, Expr) and head.head == _Derivative and len(args) == 1 and head.args \
            and all(K.is_int(o) and int(o) >= 0 for o in head.args):
        f = args[0]
        if isinstance(f, Symbol) or (isinstance(f, Expr) and f.head == "Function"):
            r = derivative_of_function(head.args, f, ev)
            if r is not None:
                return r
    return _orig_compound(head, args, ev)


KL.apply_compound_head = apply_compound_head


@fn("Derivative", area=AREA, status="added",
    notes="Derivative[n1, n2, ...][f] for pure functions and built-in mathematical functions gives a pure "
          "function (Derivative[1][Sin] -> Cos[#1] &, Derivative[2][#^3 &] -> 6 #1 &); Derivative[n...][f][x...] "
          "evaluates for known f (Sin'[x] -> Cos[x], previously unevaluated) and stays symbolic for undefined f "
          "(f'[x], Derivative[1, 0][f][x, y]); Derivative[0][f] -> f")
def _derivative_head(*orders):
    for o in orders:
        if not K.is_int(o) or int(o) < 0:
            raise ValueError("Derivative: orders must be non-negative integers")
    raise Unevaluated()




# ───────────────────────────── Integrate ─────────────────────────────

def _int_spec(spec, fname="Integrate"):
    if isinstance(spec, Symbol):
        return variable(spec, fname)
    if is_list(spec) and len(spec.args) == 3:
        return (variable(spec.args[0], fname), sym(spec.args[1], fname), sym(spec.args[2], fname))
    raise ValueError(f"{fname}: each variable specification must be x or {{x, a, b}}")


def _wl_condition(c):
    """Rewrite SymPy convergence conditions in Wolfram form: |arg z| < pi/2 -> Re[z] > 0; drop a > -oo, a < oo."""
    def rw(e):
        if isinstance(e, sp.StrictLessThan) and isinstance(e.lhs, sp.Abs) and isinstance(e.lhs.args[0], sp.arg) \
                and e.rhs == sp.pi / 2:
            return sp.re(e.lhs.args[0].args[0]) > 0
        if isinstance(e, sp.core.relational.Relational) and (e.lhs in (sp.oo, -sp.oo) or e.rhs in (sp.oo, -sp.oo)) \
                and not isinstance(e, (sp.Eq, sp.Ne)):
            return sp.true
        return e
    if isinstance(c, (sp.And, sp.Or)):
        return type(c)(*(_wl_condition(a) for a in c.args))
    return rw(c)


def _zero_power_conditions(v):
    """0^e with symbolic e (left by SymPy at an endpoint) -> 0 under Re[e] > 0."""
    conds = []

    def rep(p):
        e = sp.expand(p.exp)
        const, rest = e.as_independent(*e.free_symbols, as_Add=True)
        conds.append(sp.re(rest) > -const if rest != 0 else sp.re(e) > 0)
        return sp.Integer(0)
    v = v.replace(lambda t: isinstance(t, sp.Pow) and t.base == 0 and t.exp.free_symbols, rep)
    return v, conds


def _drop_implied(conds):
    """Drop Ne(a, c) when another condition Re[a] > d (or a > d) with c <= d is present; drop True."""
    flat = []
    for c in conds:
        flat.extend(sp.And.make_args(c))
    flat = [c for c in flat if c is not sp.true]
    lower = {}
    for c in flat:
        if isinstance(c, sp.StrictGreaterThan) and c.rhs.is_number:
            base = c.lhs.args[0] if isinstance(c.lhs, sp.re) else c.lhs
            lower[base] = max(lower.get(base, -sp.oo), c.rhs)
    kept = []
    for c in flat:
        if isinstance(c, sp.Ne) and c.rhs.is_number and c.rhs.is_real and c.lhs in lower and c.rhs <= lower[c.lhs]:
            continue
        if c not in kept:
            kept.append(c)
    return kept


def integrate_sym(v, specs, A, pv=False):
    """SymPy integration of v over Wolfram specifications (last innermost); returns (value, [conditions])."""
    conds = []
    definite = False
    for spec in reversed(specs):
        s_ = _int_spec(spec)
        s_ = A.apply(s_) if isinstance(s_, sp.Basic) else tuple(A.apply(t) for t in s_)
        if isinstance(s_, tuple):
            definite = True
        if isinstance(s_, tuple) and v.has(sp.Heaviside):
            v = v.rewrite(sp.Piecewise)
        if pv and isinstance(s_, tuple):
            r = sp.Integral(v, s_).principal_value()
        else:
            r = sp.integrate(v, s_)
        if isinstance(r, sp.Basic) and r.has(sp.meijerg, sp.exp_polar, sp.polar_lift):
            r = sp.hyperexpand(r)
            if r.has(sp.meijerg, sp.exp_polar, sp.polar_lift):
                raise Unevaluated()
        if not isinstance(s_, tuple) and isinstance(r, sp.Piecewise) and r.args[0][1] is not sp.true \
                and all(isinstance(c, sp.Ne) for c in sp.And.make_args(r.args[0][1])):
            r = r.args[0][0]          # Wolfram gives the generic antiderivative (Integrate[x^n, x] -> x^(1+n)/(1+n))
        if isinstance(r, sp.Piecewise):
            branches = [(e, c) for e, c in r.args if c is not sp.true]
            default = [e for e, c in r.args if c is sp.true]
            if len(branches) == 1 and default and (default[0].has(sp.Integral) or default[0] in (sp.oo, -sp.oo)):
                e, c = branches[0]
                c = A.refine(_wl_condition(c))
                if c is sp.false:
                    if default[0] in (sp.oo, -sp.oo):
                        r = default[0]
                    else:
                        raise ValueError("Integrate: the integral does not converge under the given assumptions")
                else:
                    if c is not sp.true:
                        conds.append(c)
                    r = e
        if isinstance(r, sp.Basic):
            r, zc = _zero_power_conditions(r)
            for c in zc:
                c = A.refine(c)
                if c is sp.false:
                    raise ValueError("Integrate: the integral does not converge under the given assumptions")
                if c is not sp.true:
                    conds.append(c)
        v = r
    if v is sp.nan or (isinstance(v, sp.Basic) and v.has(sp.nan)) or v is sp.zoo:
        raise ValueError("Integrate: the integral does not converge (the result is indeterminate)")
    return v, _drop_implied(conds), definite


def _verify_antiderivative(F, f, x):
    """d/dx F == f (symbolically, else numerically at sample points); raises when it fails."""
    if not isinstance(F, sp.Basic) or F.has(sp.Integral) or F.has(sp.Piecewise):
        return
    d = sp.diff(F, x) - f
    try:
        if sp.simplify(d) == 0:
            return
    except Exception:
        pass
    import random
    rnd = random.Random(7)
    free = sorted(d.free_symbols, key=str)
    bad = 0
    for _ in range(6):
        vals = {s: sp.Rational(rnd.randint(11, 29), 10) for s in free}
        try:
            val = complex(sp.N(d.subs(vals), 30))
        except Exception:
            continue
        if abs(val) > 1e-12 * (1 + abs(complex(sp.N(f.subs(vals), 30)))):
            bad += 1
    if bad:
        raise ArithmeticError("Integrate: the antiderivative failed verification (differentiating it does not give "
                              "the integrand)")


@replace("Integrate", AREA,
         "batch 4: indefinite integrals (verified by differentiating back), definite, improper and multiple "
         "integrals (the last specification is innermost); options Assumptions, GenerateConditions (Automatic/True "
         "give ConditionalExpression[value, cond] when the result needs a parameter condition, e.g. "
         "Integrate[x^a, {x, 0, 1}] -> ConditionalExpression[1/(1 + a), Re[a] > -1]; False gives the generic value) "
         "and PrincipalValue -> True (Cauchy principal value, Integrate[1/x, {x, -1, 2}, PrincipalValue -> True] -> "
         "Log[2]); SymPy convergence conditions are rewritten in Wolfram form (Abs[Arg[a]] < Pi/2 -> Re[a] > 0); "
         "integrals that diverge to +-Infinity give that limit, indeterminate (oscillating/cancelling) ones raise; "
         "HeavisideTheta and DiracDelta integrands are supported; integrals SymPy cannot do stay unevaluated")
def _integrate(f, *args):
    from .assume import effective, from_sym
    specs, opts = split_opts(args, {"Assumptions", "GenerateConditions", "PrincipalValue"}, "Integrate")
    for s_ in specs:
        if is_rule(s_):
            raise ValueError(f"Integrate: unknown option {s_.args[0]}")
    if not specs:
        raise ValueError("Integrate needs at least one variable specification")
    if is_list(f):
        return L(*(_ev(Expr("Integrate", g, *args)) for g in f.args))
    A = effective([], opts.get("Assumptions"), "Integrate")
    gen = opts.get("GenerateConditions", Symbol("Automatic"))
    pv = is_true(opts.get("PrincipalValue", False))
    if pv and not (len(specs) == 1 and is_list(specs[0])):
        raise ValueError("Integrate: PrincipalValue -> True needs one definite specification {x, a, b}")
    integrand = to_heaviside(A.apply(sym(f, "Integrate")))
    v, conds, definite = integrate_sym(integrand, specs, A, pv)
    if isinstance(v, sp.Basic) and v.has(sp.Integral):
        raise Unevaluated()
    if not definite and len(specs) == 1:
        _verify_antiderivative(v, integrand, _int_spec(specs[0]) if isinstance(specs[0], Symbol) else None)
    if A.subs and isinstance(v, sp.Basic):
        v = A.refine(sp.simplify(v)) if v.has(sp.Abs, sp.sign, sp.arg, sp.Piecewise) else v
    v = A.unapply(v)
    value = numeric_out(wl_fix(v), f, *specs)
    if conds and not is_false(gen):
        cond = A.unapply(sp.And(*conds))
        return Expr("ConditionalExpression", value, from_sym(cond))
    return value
