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


# ───────────────────────────── Limit, MaxLimit, MinLimit ─────────────────────────────

def _direction(d, fname):
    """Wolfram Direction -> '+' (from above), '-' (from below) or None (two-sided)."""
    if d is None:
        return None
    if isinstance(d, str):
        table = {"FromAbove": "+", "FromBelow": "-", "TwoSided": None, "FromRight": "+", "FromLeft": "-"}
        if d in table:
            return table[d]
    if K.is_int(d) and int(d) in (-1, 1):
        return "+" if int(d) == -1 else "-"          # Wolfram: Direction -> -1 means from larger values
    if isinstance(d, Symbol) and d.name in ("Reals", "Automatic"):
        return None
    if isinstance(d, Symbol) and d.name == "Complexes":
        raise NotImplementedError(f"{fname}: Direction -> Complexes is not supported")
    raise ValueError(f"{fname}: invalid Direction {d}")


def _bad_limit(v):
    return isinstance(v, sp.Basic) and (v.has(sp.AccumBounds) or v.has(sp.Limit) or v is sp.nan or v.has(sp.nan))


def _limit1(f, x, x0, d, fname):
    """One limit (SymPy Gruntz); raises when SymPy is unsure."""
    try:
        v = sp.limit(f, x, x0, d)
    except NotImplementedError as exc:
        raise NotImplementedError(f"{fname}: cannot determine the limit ({exc}); give Assumptions on the "
                                  "parameters") from None
    return v


def _two_sided(f, x, x0, fname):
    if x0 in (sp.oo, -sp.oo):
        return _limit1(f, x, x0, "+" if x0 is -sp.oo else "-", fname), None
    right = _limit1(f, x, x0, "+", fname)
    left = _limit1(f, x, x0, "-", fname)
    return right, left


def _limit_value(f, x, x0, d, fname):
    if d is not None and x0 not in (sp.oo, -sp.oo):
        v = _limit1(f, x, x0, d, fname)
    else:
        right, left = _two_sided(f, x, x0, fname)
        if left is None or right == left:
            v = right
        elif _bad_limit(right) or _bad_limit(left):
            v = sp.AccumBounds(-1, 1)        # forces the oscillation error below
        else:
            return sp.nan                    # one-sided limits differ: Indeterminate (Wolfram)
    if isinstance(v, sp.AccumBounds) or _bad_limit(v):
        raise NotImplementedError(f"{fname}: the function oscillates near the limit point (no limit); use "
                                  "MaxLimit and MinLimit for the upper and lower limits")
    return v


def _multivariate_limit(f, xs, pts, fname):
    """Limit at a point in several variables: continuity of the (cancelled) function, or disagreeing paths."""
    if any(p in (sp.oo, -sp.oo) for p in pts):
        raise NotImplementedError(f"{fname}: multivariate limits at infinity are not supported")
    sub = dict(zip(xs, pts))
    for g in (f, sp.cancel(sp.together(f))):
        try:
            val = g.subs(sub)
        except Exception:
            continue
        if val.is_finite and not val.has(sp.nan, sp.zoo) and _continuous_at(g, xs, pts):
            return sp.simplify(val)
    # Two straight-line paths with different limits prove there is no limit.
    t = sp.Dummy("t", positive=True)
    vals = []
    for dirs in ([1] * len(xs), [1] + [0] * (len(xs) - 1), [0] * (len(xs) - 1) + [1], [1, -1] + [0] * (len(xs) - 2),
                 [1, 2] + [1] * (len(xs) - 2)):
        try:
            g = f.subs({x: p + c * t for x, p, c in zip(xs, pts, dirs)}, simultaneous=True)
            vals.append(sp.limit(g, t, 0, "+"))
        except Exception:
            continue
    if len(xs) >= 2:
        for e0, e1 in ((1, 2), (2, 1)):
            try:
                path = {x: p for x, p in zip(xs, pts)}
                path[xs[0]] = pts[0] + t ** e0
                path[xs[1]] = pts[1] + t ** e1
                g = f.subs({x: path[x] if x in (xs[0], xs[1]) else p + t for x, p in zip(xs, pts)}, simultaneous=True)
                vals.append(sp.limit(g, t, 0, "+"))
            except Exception:
                continue
    vals = [v for v in vals if not _bad_limit(v)]
    if any(sp.simplify(v - vals[0]) != 0 for v in vals[1:]):
        return sp.nan
    if len(xs) == 2 and vals:
        cand = vals[0]
        r, th = sp.Dummy("r", positive=True), sp.Dummy("theta", real=True)
        try:
            h = sp.simplify(sp.trigsimp((f - cand).subs({xs[0]: pts[0] + r * sp.cos(th),
                                                         xs[1]: pts[1] + r * sp.sin(th)}, simultaneous=True)))
            rpart, rest = sp.Integer(1), sp.Integer(1)
            for fac in sp.Mul.make_args(sp.factor(h)):
                if fac.has(r):
                    rpart *= fac
                else:
                    rest *= fac
            k = None
            if rpart == r:
                k = 1
            elif isinstance(rpart, sp.Pow) and rpart.base == r and rpart.exp.is_positive:
                k = rpart.exp
            num, den = sp.fraction(sp.together(rest))
            if h == 0 or (k is not None and not den.has(th) and den != 0):
                return sp.simplify(cand)      # |f - L| <= r^k * (bounded function of theta)
        except Exception:
            pass
    raise NotImplementedError(f"{fname}: cannot determine this multivariate limit")


def _continuous_at(g, xs, pts):
    """g is a composition of functions continuous at the point (rational functions with nonzero denominator,
    polynomials, exp, sin, cos, ...): every subexpression is finite there."""
    sub = dict(zip(xs, pts))
    allowed = (sp.Add, sp.Mul, sp.Pow, sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh, sp.atan, sp.Symbol, sp.Number,
               sp.NumberSymbol, sp.log, sp.sqrt, sp.Abs, sp.tan, sp.tanh, sp.asin, sp.acos)
    for e in sp.preorder_traversal(g):
        if not isinstance(e, allowed) and not e.is_number:
            return False
        if isinstance(e, sp.Pow):
            b = e.base.subs(sub)
            if not b.is_finite or (b == 0 and not (e.exp.is_number and e.exp.is_positive)):
                return False
            if not e.exp.is_integer and not (b.is_positive or b.is_zero):
                return False
        if isinstance(e, sp.log) and not e.args[0].subs(sub).is_positive:
            return False
        if isinstance(e, (sp.asin, sp.acos)):
            a = e.args[0].subs(sub)
            if not (a.is_real and abs(a) < 1):
                return False
        if isinstance(e, sp.tan) and not sp.cos(e.args[0].subs(sub)).is_nonzero:
            return False
        val = e.subs(sub)
        if val.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            return False
    return True


def _limit_args(a, fname):
    pos, opts = split_opts(a, {"Direction", "Assumptions", "GenerateConditions"}, fname)
    if len(pos) != 2:
        raise ValueError(f"{fname} expects {fname}[f, x -> x0] with optional Direction and Assumptions")
    f, r = pos
    if not is_rule(r):
        raise ValueError(f"{fname} expects {fname}[f, x -> x0]")
    return f, r, opts


def _limit_impl(f, r, opts, fname):
    from .assume import effective
    if is_list(f):
        return L(*(_limit_impl(g, r, opts, fname) for g in f.args))
    A = effective([], opts.get("Assumptions"), fname)
    d = _direction(opts.get("Direction"), fname)
    lhs, rhs = r.args
    g = A.apply(_S(f, fname))
    if is_list(lhs):
        xs = [A.apply(_var(v, fname)) for v in lhs.args]
        pts = _S(rhs, fname)
        if not isinstance(pts, list) or len(pts) != len(xs):
            raise ValueError(f"{fname}: {{x1, x2, ...}} -> {{a1, a2, ...}} needs one point per variable")
        if d is not None:
            raise NotImplementedError(f"{fname}: Direction is not supported for multivariate limits")
        v = _multivariate_limit(g, xs, [A.apply(p) for p in pts], fname)
    else:
        x = A.apply(_var(lhs, fname))
        x0 = A.apply(_S(rhs, fname))
        v = _limit_value(g, x, x0, d, fname)
    v = A.unapply(v)
    return numeric_out(v, f, rhs)


@kernel_override("Limit", AREA,
                 "batch 4: Limit[f, x -> a] (two-sided: Indeterminate when the one-sided limits differ), "
                 "Direction -> \"FromAbove\" / \"FromBelow\" / -1 / 1 / \"TwoSided\", Assumptions, limits at "
                 "+-Infinity, lists of functions, and multivariate limits Limit[f, {x, y} -> {a, b}] (continuity "
                 "after cancellation, Indeterminate when two paths disagree, otherwise a clear error); "
                 "oscillating limits raise (use MaxLimit/MinLimit); legacy rejected every option")
def _b_limit(a, ev, expr):
    f, r, opts = _limit_args(a, "Limit")
    return _limit_impl(f, r, opts, "Limit")


def _osc_bounds(f, x, x0, d, fname):
    """(liminf, limsup) for f = A + B Sin[h] + C Cos[h] with h -> +-Infinity and A, B, C having finite limits:
    A -+ Sqrt[B^2 + C^2] (every phase of h is reached, so the bounds are attained)."""
    oscs = list(dict.fromkeys(e for e in sp.preorder_traversal(f) if isinstance(e, (sp.sin, sp.cos))))
    if not oscs or len(oscs) > 2 or len({e.args[0] for e in oscs}) != 1:
        return None
    h = oscs[0].args[0]
    us, uc = sp.Dummy("us"), sp.Dummy("uc")
    g = sp.expand(f.subs({sp.sin(h): us, sp.cos(h): uc}))
    if g.has(sp.sin(h), sp.cos(h)) or not g.is_polynomial(us, uc) or sp.Poly(g, us, uc).total_degree() > 1:
        return None
    B, C = g.coeff(us, 1).subs(uc, 0), g.coeff(uc, 1).subs(us, 0)
    Ac = g.subs({us: 0, uc: 0})
    lims = [_limit1(e, x, x0, d, fname) for e in (Ac, B, C)]
    arg_lim = _limit1(h, x, x0, d, fname)
    if any(_bad_limit(v) or not v.is_finite for v in lims) or arg_lim not in (sp.oo, -sp.oo):
        return None
    la, lb, lc = lims
    amp = sp.sqrt(lb ** 2 + lc ** 2)
    return la - amp, la + amp


def _extreme_limit(a, fname, which):
    from .assume import effective
    f, r, opts = _limit_args(a, fname)
    if is_list(f):
        return L(*(_extreme_limit([g, r] + [rule(Symbol(k), v) for k, v in opts.items()], fname, which)
                   for g in f.args))
    A = effective([], opts.get("Assumptions"), fname)
    d = _direction(opts.get("Direction"), fname)
    lhs, rhs = r.args
    if is_list(lhs):
        raise NotImplementedError(f"{fname}: multivariate upper/lower limits are not supported")
    g = A.apply(_S(f, fname))
    x, x0 = A.apply(_var(lhs, fname)), A.apply(_S(rhs, fname))
    sides = [d] if (d is not None or x0 in (sp.oo, -sp.oo)) else ["+", "-"]
    if x0 is sp.oo:
        sides = ["-"]
    elif x0 is -sp.oo:
        sides = ["+"]
    vals = []
    for sd in sides:
        v = _limit1(g, x, x0, sd, fname)
        if isinstance(v, sp.AccumBounds) or _bad_limit(v):
            b = _osc_bounds(g, x, x0, sd, fname)
            if b is None:
                raise NotImplementedError(f"{fname}: cannot determine the {which} limit of this oscillating "
                                          "function")
            vals.append(b[1] if which == "upper" else b[0])
        else:
            vals.append(v)
    res = sp.Max(*vals) if which == "upper" else sp.Min(*vals)
    return numeric_out(A.unapply(sp.simplify(res)), f, rhs)


@fn("MaxLimit", area=AREA, status="added",
    notes="upper limit (limsup) MaxLimit[f, x -> a] with Direction and Assumptions: the larger one-sided limit, "
          "or A + |B| for f = A + B Sin[g] / A + B Cos[g] with g -> +-Infinity (MaxLimit[Sin[1/x], x -> 0] -> 1); "
          "other oscillating functions raise")
def _maxlimit(*a):
    return _extreme_limit(list(a), "MaxLimit", "upper")


@fn("MinLimit", area=AREA, status="added",
    notes="lower limit (liminf) MinLimit[f, x -> a]; see MaxLimit (MinLimit[Sign[x], x -> 0] -> -1)")
def _minlimit(*a):
    return _extreme_limit(list(a), "MinLimit", "lower")


# ───────────────────────────── Series ─────────────────────────────

class SeriesData:
    """A truncated series sum c_k b^k + O[b]^m in the base b = x - x0 (or 1/x at Infinity)."""

    def __init__(self, x, x0, terms, m):
        self.x, self.x0, self.terms, self.m = x, x0, terms, m      # terms: [(k, coeff)] sorted by k

    @property
    def base(self):
        if self.x0 is sp.oo or self.x0 is -sp.oo:
            return 1 / self.x
        return self.x - self.x0

    def poly(self):
        return sp.Add(*(c * self.base ** k for k, c in self.terms))

    def coeff(self, k):
        if k >= self.m:
            return None
        for kk, c in self.terms:
            if kk == k:
                return c
        return sp.Integer(0)

    def to_expr(self, coeff_expr=None):
        """Wolfram form: Plus[c0, c1 b, ..., O[b]^m] (built without re-evaluation so the grouping survives)."""
        b = self.base
        bexpr = R(b)
        parts = []
        for k, c in self.terms:
            ce = coeff_expr(c) if coeff_expr else None
            if ce is not None:
                parts.append(ce if k == 0 else Expr("Times", ce, bexpr if k == 1 else Expr("Power", bexpr, R(k))))
                continue
            if k == 0:
                parts.append(out(c))
            elif self.x0 == 0 or self.x0 in (sp.oo, -sp.oo):
                parts.append(out(c * b ** k))
            elif c == 1:
                parts.append(bexpr if k == 1 else Expr("Power", bexpr, R(k)))
            else:
                parts.append(Expr("Times", out(c), bexpr if k == 1 else Expr("Power", bexpr, R(k))))
        o = Expr("O", bexpr)
        parts.append(o if self.m == 1 else Expr("Power", o, R(self.m)))
        return parts[0] if len(parts) == 1 else Expr("Plus", *parts)


def series_data(f, x, x0, n, fname="Series"):
    """Series of SymPy f in x about x0 up to (x - x0)^n (absolute order, Laurent and Puiseux terms allowed)."""
    t = sp.Dummy("t", positive=True)
    if x0 is sp.oo:
        g, d = f.subs(x, 1 / t), "+"
    elif x0 is -sp.oo:
        g, d = f.subs(x, -1 / t), "+"
    else:
        g, d = f.subs(x, x0 + t), "+"
    try:
        s = sp.series(g, t, 0, n + 1, dir=d)
    except (NotImplementedError, ValueError, TypeError) as exc:
        raise NotImplementedError(f"{fname}: cannot expand this function ({exc})") from None
    o = s.getO()
    m = sp.Integer(n + 1)
    if o is not None:
        oe = o.expr
        if oe == t:
            m = sp.Integer(1)
        elif oe == 1:
            m = sp.Integer(0)
        elif isinstance(oe, sp.Pow) and oe.base == t:
            m = oe.exp
        else:
            raise NotImplementedError(f"{fname}: unsupported order term {o}")
    body = sp.expand(s.removeO())
    if body.has(sp.Order) or body.has(sp.AccumBounds) or body.has(sp.Subs):
        raise NotImplementedError(f"{fname}: cannot expand this function")
    terms = {}
    for term in sp.Add.make_args(body):
        if term == 0:
            continue
        c, k = term.as_independent(t, as_Add=False)
        if k == 1:
            kk = sp.Integer(0)
        elif k == t:
            kk = sp.Integer(1)
        elif isinstance(k, sp.Pow) and k.base == t and k.exp.is_Rational:
            kk = k.exp
        else:
            c2, kk = term.as_coeff_exponent(t)
            if c2.has(t) and not c2.has(sp.log):
                raise NotImplementedError(f"{fname}: cannot write the expansion as a power series")
            c = c2
        if kk >= m:
            continue
        terms[kk] = terms.get(kk, 0) + c
    if x0 is -sp.oo:
        terms = {k: c * (-1) ** k for k, c in terms.items()}
    back = {t: (1 / x if x0 in (sp.oo, -sp.oo) else x - x0)}
    items = sorted(((k, sp.simplify(c.subs(back))) for k, c in terms.items()), key=lambda kc: kc[0])
    return SeriesData(x, x0, [(k, c) for k, c in items if c != 0], m)


def _series_spec(spec, fname="Series"):
    if not (is_list(spec) and len(spec.args) in (2, 3)):
        raise ValueError(f"{fname}: each specification must be {{x, x0, n}}")
    if len(spec.args) == 2:
        x, x0 = spec.args
        n = None
    else:
        x, x0, n = spec.args
    xs = _var(x, fname)
    if not isinstance(xs, sp.Symbol):
        raise ValueError(f"{fname}: {x} is not a variable")
    if n is not None:
        if not K.is_int(n):
            raise ValueError(f"{fname}: the order must be an integer")
        n = int(n)
    return xs, _S(x0, fname), n


def _series_expr(f, specs, fname="Series"):
    """Series[f, {x, x0, n}, {y, y0, m}, ...]: expand in x, then every coefficient in y, ..."""
    x, x0, n = _series_spec(specs[0], fname)
    if n is None:
        raise ValueError(f"{fname}: give the order, {{x, x0, n}}")
    sd = series_data(f, x, x0, n, fname)
    if len(specs) == 1:
        return sd.to_expr()
    return sd.to_expr(coeff_expr=lambda c: _series_expr(c, specs[1:], fname))


@kernel_override("Series", AREA,
                 "batch 4: Series[f, {x, x0, n}] with Laurent and Puiseux terms, expansions at +-Infinity in powers "
                 "of 1/x (O[1/x]^(n+1)), the O term always present (Series[1/x, {x, 0, 2}] -> 1/x + O[x]^3; "
                 "previously dropped), expansions about x0 grouped in powers of (x - x0) (O[x - x0]^k); several "
                 "variables Series[f, {x, x0, nx}, {y, y0, ny}] expand in x with coefficients that are series in y; "
                 "threads over lists; functions SymPy cannot expand raise")
def _b_series(a, ev, expr):
    if len(a) < 2:
        raise ValueError("Series expects Series[f, {x, x0, n}]")
    f, specs = a[0], a[1:]
    if is_list(f):
        return L(*(_b_series([g] + specs, ev, expr) for g in f.args))
    v = _S(f, "Series")
    return _series_expr(v, specs)


def _o_terms(e):
    return isinstance(e, Expr) and (e.head == "O" or (e.head == "Power" and isinstance(e.args[0], Expr)
                                                       and e.args[0].head == "O"))


def strip_o(e):
    if isinstance(e, Expr) and e.head == "Plus":
        kept = [strip_o(a) for a in e.args if not _o_terms(a)]
        return Expr("Plus", *kept) if len(kept) != 1 else kept[0] if kept else 0
    if isinstance(e, Expr) and not _o_terms(e):
        h = strip_o(e.head) if isinstance(e.head, Expr) else e.head
        return Expr(h, *(strip_o(a) for a in e.args))
    if _o_terms(e):
        return 0
    return e


def parse_series(e, fname):
    """MikoshiLang series (Plus[..., O[b]^m]) -> SeriesData; raises when e is not a series."""
    if _o_terms(e):
        os_, rest = [e], []
    elif isinstance(e, Expr) and e.head == "Plus":
        os_ = [a for a in e.args if _o_terms(a)]
        rest = [a for a in e.args if not _o_terms(a)]
    else:
        os_ = []
    if len(os_) != 1:
        raise ValueError(f"{fname}: {e} is not a power series (use Series[f, {{x, x0, n}}])")
    o = os_[0]
    if o.head == "O":
        b, m = o.args[0], sp.Integer(1)
    else:
        b, m = o.args[0].args[0], _S(o.args[1], fname)
    bs = _S(b, fname)
    syms = sorted(bs.free_symbols, key=str)
    if len(syms) != 1:
        raise ValueError(f"{fname}: cannot read the expansion variable of {e}")
    x = syms[0]
    if bs == x:
        x0 = sp.Integer(0)
    elif bs == 1 / x:
        x0 = sp.oo
    elif sp.expand(bs - x).free_symbols == set():
        x0 = -sp.expand(bs - x)
    else:
        raise ValueError(f"{fname}: unsupported series base {b}")
    body = sp.Add(*(_S(a, fname) for a in rest)) if rest else sp.Integer(0)
    if any(_has_o(a) for a in rest):
        raise NotImplementedError(f"{fname}: series with series coefficients (several variables) are not supported")
    t = sp.Dummy("t")
    g = sp.expand(body.subs(x, (1 / t) if x0 is sp.oo else x0 + t))
    terms = {}
    for term in sp.Add.make_args(g):
        if term == 0:
            continue
        c, k = term.as_coeff_exponent(t)
        if c.has(t):
            raise ValueError(f"{fname}: {e} is not a power series")
        terms[k] = terms.get(k, 0) + c
    back = {t: (1 / x if x0 is sp.oo else x - x0)}
    return SeriesData(x, x0, sorted(((k, sp.simplify(c.subs(back))) for k, c in terms.items() if c != 0),
                                    key=lambda kc: kc[0]), m)


def _has_o(e):
    return K.contains(e, lambda y: isinstance(y, Expr) and y.head == "O")


_orig_normal = K._BUILTINS.get("Normal")


@kernel_override("Normal", AREA,
                 "batch 4 adds: Normal[series] drops every O term structurally (also the inner ones of multivariate "
                 "series) and evaluates the remaining polynomial; associations and other heads as before (batch 1)")
def _b_normal(a, ev, expr):
    if a and _has_o(a[0]) and (len(a) == 1 or (len(a) == 2 and K.contains(a[1], lambda y: isinstance(y, Symbol)
                                                                         and y.name in ("Series", "SeriesData")))):
        return ev(strip_o(a[0]))
    return _orig_normal(a, ev, expr)


@fn("O", area=AREA, status="added",
    notes="order term O[x]^n / O[x - x0]^n / O[1/x]^n of Series results (stays symbolic); Normal removes it; "
          "SeriesCoefficient, InverseSeries and ComposeSeries read it")
def _o(*args):
    if len(args) != 1:
        raise ValueError("O takes one argument")
    raise Unevaluated()


def _fps_coefficient(g, t, nsym, fname):
    """(general term, first index) of the Maclaurin series of g(t) via SymPy fps; the formula must reproduce the
    first 16 Taylor coefficients (those below the first index must be 0)."""
    from sympy.series.formal import FormalPowerSeries
    try:
        F = sp.fps(g, t, 0)
    except Exception:
        return None
    if not isinstance(F, FormalPowerSeries):
        return None
    k = F.ak.variables[0]
    ki = sp.Symbol("k", integer=True, nonnegative=True)
    form = F.ak.formula.xreplace({k: ki})
    form = form.replace(lambda e: isinstance(e, sp.Mod) and e.args[1] == 1, lambda e: sp.Integer(0))
    form = sp.piecewise_fold(form)
    if isinstance(form, sp.Piecewise) and form.args[0][1] is sp.true:
        form = form.args[0][0]
    try:
        taylor = sp.series(g, t, 0, 16).removeO()
    except Exception:
        return None
    coeffs = [sp.simplify(taylor.coeff(t, j)) for j in range(16)]
    start = None
    for s0 in range(0, 4):
        if any(c != 0 for c in coeffs[:s0]):
            break
        ok = True
        for j in range(s0, 16):
            try:
                fj = sp.simplify(form.subs(ki, j))
            except Exception:
                ok = False
                break
            if fj.has(sp.zoo, sp.nan) or sp.simplify(fj - coeffs[j]) != 0:
                ok = False
                break
        if ok:
            start = s0
            break
    if start is None:
        return None
    nint = sp.Symbol(nsym.name, integer=True)
    gen = sp.simplify(form.subs(ki, nint))
    if gen.has(sp.gamma):
        gen = gen.rewrite(sp.factorial)
    return gen.xreplace({nint: nsym}), start


@fn("SeriesCoefficient", area=AREA, status="added",
    notes="SeriesCoefficient[f, {x, x0, n}] (negative n for Laurent series, x0 may be Infinity: coefficient of "
          "(1/x)^n), several variables {x, x0, nx}, {y, y0, ny}, and SeriesCoefficient[series, n]; a symbolic n "
          "gives Piecewise[{{general term, n >= 0}}, 0] when SymPy finds a formal power series (checked against "
          "10 Taylor coefficients); a coefficient beyond a given series' order is Indeterminate")
def _seriescoefficient(f, *specs):
    if not specs:
        raise ValueError("SeriesCoefficient needs a specification")
    if len(specs) == 1 and not is_list(specs[0]):
        sd = parse_series(f, "SeriesCoefficient")
        n = specs[0]
        if not K.is_int(n):
            raise Unevaluated()
        c = sd.coeff(int(n))
        return Symbol("Indeterminate") if c is None else out(c)
    v = _S(f, "SeriesCoefficient")
    spec = specs[0]
    x, x0, n = None, None, None
    if not (is_list(spec) and len(spec.args) == 3):
        raise ValueError("SeriesCoefficient: specification must be {x, x0, n}")
    xs, x0s, nn = spec.args
    x = _var(xs, "SeriesCoefficient")
    x0 = _S(x0s, "SeriesCoefficient")
    if not K.is_int(nn):
        if isinstance(nn, Symbol) and len(specs) == 1 and x0 not in (sp.oo, -sp.oo):
            t = sp.Dummy("t")
            nsym = sp.Symbol(nn.name)
            res = _fps_coefficient(v.subs(x, x0 + t), t, nsym, "SeriesCoefficient")
            if res is None:
                raise Unevaluated()
            gen, start = res
            return Expr("Piecewise", L(L(out(gen), Expr("GreaterEqual", nn, start))), 0)
        raise Unevaluated()
    n = int(nn)
    sd = series_data(v, x, x0, n, "SeriesCoefficient")
    c = sd.coeff(n)
    if c is None:
        raise NotImplementedError("SeriesCoefficient: SymPy's expansion does not reach this order")
    if len(specs) > 1:
        return _seriescoefficient(out(c), *specs[1:])
    return numeric_out(c, f)


def _compose_poly(p, q, x, m):
    """p(q(x)) truncated below x^m."""
    return sp.series(sp.expand(p.subs(x, q)), x, 0, m).removeO() if m > 0 else sp.Integer(0)


def _check_origin(sd, fname, what):
    if sd.x0 != 0:
        raise NotImplementedError(f"{fname}: only series about 0 are supported ({what})")
    if any(not (k.is_Integer and k >= 0) for k, _ in sd.terms) or not sd.m.is_Integer:
        raise NotImplementedError(f"{fname}: only ordinary power series (no negative or fractional powers) are "
                                  f"supported ({what})")


@fn("InverseSeries", area=AREA, status="added",
    notes="InverseSeries[s] / InverseSeries[s, y] for a power series s = a1 x + a2 x^2 + ... + O[x]^n about 0 "
          "with a1 != 0: the series of the inverse function to the same order (InverseSeries[Series[Sin[x], "
          "{x, 0, 5}]] -> x + x^3/6 + 3 x^5/40 + O[x]^6); the result is checked by composing back")
def _inverseseries(s, y=None):
    sd = parse_series(s, "InverseSeries")
    _check_origin(sd, "InverseSeries", "the input")
    x = sd.x
    yv = x if y is None else _var(y, "InverseSeries")
    if sd.coeff(0) not in (0, None) and sd.coeff(0) != 0:
        raise NotImplementedError("InverseSeries: the series must vanish at the expansion point")
    a1 = sd.coeff(1)
    if a1 is None or a1 == 0:
        raise ValueError("InverseSeries: the linear coefficient is zero, so the series has no power-series inverse")
    m = int(sd.m)
    p = sd.poly()
    q = x / a1
    for k in range(2, m):
        comp = sp.expand(_compose_poly(p, q, x, k + 1))
        ck = comp.coeff(x, k)
        q = q - sp.simplify(ck / a1) * x ** k
    check = sp.expand(_compose_poly(p, q, x, m) - x)
    if sp.simplify(check) != 0:
        raise ArithmeticError("InverseSeries: composition check failed")
    res = SeriesData(yv, sp.Integer(0), [(sp.Integer(k), sp.simplify(sp.expand(q).coeff(x, k)))
                                          for k in range(1, m) if sp.simplify(sp.expand(q).coeff(x, k)) != 0],
                     sp.Integer(m))
    return res.to_expr()


@fn("ComposeSeries", area=AREA, status="added",
    notes="ComposeSeries[s1, s2, ...] = s1(s2(...)) for power series about 0 where the inner series vanish at 0; "
          "the order is the one the truncations allow (ComposeSeries[Series[Exp[x], {x, 0, 5}], Series[Sin[x], "
          "{x, 0, 5}]] -> 1 + x + x^2/2 - x^4/8 - x^5/15 + O[x]^6)")
def _composeseries(s1, s2, *more):
    if more:
        return _composeseries(s1, _composeseries(s2, *more))
    A = parse_series(s1, "ComposeSeries")
    B = parse_series(s2, "ComposeSeries")
    _check_origin(A, "ComposeSeries", "outer series")
    _check_origin(B, "ComposeSeries", "inner series")
    if B.coeff(0) not in (0,):
        raise ValueError("ComposeSeries: the inner series must vanish at 0")
    if A.x != B.x:
        A = SeriesData(B.x, A.x0, [(k, c.subs(A.x, B.x)) for k, c in A.terms], A.m)
    x = B.x
    v = min(k for k, c in B.terms) if B.terms else B.m
    k0 = min((k for k, c in A.terms if k >= 1), default=None)
    m = A.m * v
    if k0 is not None:
        m = min(m, B.m + v * (k0 - 1))
    m = int(m)
    comp = sp.expand(_compose_poly(A.poly(), B.poly(), x, m))
    terms = [(sp.Integer(k), comp.coeff(x, k)) for k in range(0, m) if comp.coeff(x, k) != 0]
    return SeriesData(x, sp.Integer(0), terms, sp.Integer(m)).to_expr()
