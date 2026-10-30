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


# ───────────────────────────── sums, recurrences, ODEs ─────────────────────────────

def _sum_piecewise(v, gen, fname):
    """SymPy returns Piecewise((value, convergence condition), (Sum(...), True)) for parametric series."""
    if isinstance(v, sp.Piecewise):
        branches = [(e, c) for e, c in v.args if c is not sp.true]
        default = [e for e, c in v.args if c is sp.true]
        if len(branches) == 1 and default and default[0].has(sp.Sum, sp.Product):
            e, c = branches[0]
            return e, c
        if any(e.has(sp.Sum, sp.Product) for e, _ in v.args):
            raise Unevaluated()
    return v, None


def _sumprod4(f, specs, fname, op, gen):
    from .calculus import _iter_spec, _check_result
    v = sym(f, fname)
    conds = []
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
            terms = [v.subs(i, lo + k * step) for k in range(int(n) + 1)] if n >= 0 else []
            v = (sp.Add if op == "sum" else sp.Mul)(*terms) if terms else sp.Integer(0 if op == "sum" else 1)
            continue
        if lo.is_number and hi.is_number and hi.is_finite and lo.is_finite:
            n = sp.floor(hi - lo)
            if n < 0:
                v = sp.Integer(0 if op == "sum" else 1)
                continue
            if n <= 2000:
                terms = [v.subs(i, lo + k) for k in range(int(n) + 1)]
                v = sp.Add(*terms) if op == "sum" else sp.Mul(*terms)
                continue
        ii = sp.Dummy(i.name, integer=True)
        r = (sp.summation if op == "sum" else sp.product)(v.subs(i, ii), (ii, lo, hi))
        r = r.xreplace({ii: i}) if isinstance(r, sp.Basic) else r
        r, c = _sum_piecewise(r, gen, fname)
        if c is not None:
            conds.append(c)
        if isinstance(r, sp.Basic) and r.has(sp.Sum, sp.Product):
            raise Unevaluated()
        v = _check_result(r, fname)
    if isinstance(v, sp.Basic) and v.free_symbols:
        v = sp.simplify(v)
    return v, conds


def _sum_common(fname, op, f, args):
    from .assume import from_sym
    specs, opts = split_opts(args, {"GenerateConditions", "Assumptions", "Regularization", "VerifyConvergence",
                                    "Method"}, fname)
    for k in ("Regularization", "Method"):
        if k in opts:
            raise NotImplementedError(f"{fname}: the option {k} is not supported")
    if not specs:
        raise ValueError(f"{fname} needs an iterator")
    v, conds = _sumprod4(f, specs, fname, op, opts.get("GenerateConditions"))
    value = numeric_out(wl_fix(v), f, *specs)
    if conds and is_true(opts.get("GenerateConditions", False)):
        return Expr("ConditionalExpression", value, from_sym(sp.And(*[_wl_condition(c) for c in conds])))
    return value


@replace("Sum", AREA,
         "batch 4: as before ({i, n}, {i, a, b}, {i, a, b, di}, {i, {values}}, several iterators, first outermost; "
         "divergent sums to +-Infinity give that limit, other non-convergent sums stay unevaluated), plus "
         "parametric series that SymPy sums under a convergence condition: Wolfram's generic value by default "
         "(Sum[x^k, {k, 0, Infinity}] -> 1/(1 - x); previously a recursion error) and "
         "ConditionalExpression[value, cond] with GenerateConditions -> True")
def _sum(f, *args):
    return _sum_common("Sum", "sum", f, args)


@replace("Product", AREA,
         "batch 4: as Sum (finite, symbolic and infinite products via SymPy; parametric products give the generic "
         "value, ConditionalExpression with GenerateConditions -> True)")
def _product(f, *args):
    return _sum_common("Product", "product", f, args)


@fn("SumConvergence", area=AREA, status="added",
    notes="SumConvergence[f, n]: True/False for the convergence of Sum[f, {n, k, Infinity}] (SymPy's ratio, root, "
          "integral, comparison, alternating-series tests); the p-series family 1/n^p and n^p r^n with a "
          "parameter gives the Wolfram condition (Re[p] > 1, Abs[r] < 1); undecided cases stay unevaluated")
def _sumconvergence(f, n, *opts):
    v = sym(f, "SumConvergence")
    nv = variable(n, "SumConvergence")
    ni = sp.Symbol(nv.name, integer=True, positive=True)
    g = v.xreplace({nv: ni})
    params = g.free_symbols - {ni}
    if params:
        # f = c n^q r^n with exponents/bases depending on parameters
        c, rest = g.as_independent(ni, as_Add=False)
        q, r = sp.Integer(0), sp.Integer(1)
        for fac in sp.Mul.make_args(rest):
            if isinstance(fac, sp.Pow) and fac.base == ni and not fac.exp.has(ni):
                q += fac.exp
            elif fac == ni:
                q += 1
            elif isinstance(fac, sp.Pow) and fac.exp == ni and not fac.base.has(ni):
                r *= fac.base
            elif isinstance(fac, sp.exp) and sp.expand(fac.args[0] / ni).free_symbols.isdisjoint({ni}):
                r *= sp.exp(sp.expand(fac.args[0] / ni))
            else:
                raise Unevaluated()
        if c == 0:
            return True
        from .assume import from_sym
        if r == 1:
            return from_sym(sp.re(-q) > 1)
        if not q.free_symbols:
            return from_sym(sp.Abs(r) < 1)
        raise Unevaluated()
    start = 1
    for k in range(1, 60):
        val = g.subs(ni, k)
        if val.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            start = k + 1
    try:
        res = sp.Sum(g, (ni, start, sp.oo)).is_convergent()
    except Exception:
        raise Unevaluated()
    if res is sp.true or res is True:
        return True
    if res is sp.false or res is False:
        return False
    raise Unevaluated()


def _rec_parts(eq, name, nv, fname):
    a = sp.Function(name)
    eqs = list(eq.args) if is_list(eq) else [eq]
    rec, ics = None, {}
    for e in eqs:
        se = sym(e, fname)
        if isinstance(se, sp.Equality) and isinstance(se.lhs, AppliedUndef) and se.lhs.func == a \
                and not se.lhs.args[0].has(nv) and not se.rhs.has(a):
            ics[se.lhs] = se.rhs
        elif rec is None:
            rec = se
        else:
            raise NotImplementedError(f"{fname}: systems of recurrences are not supported")
    if rec is None:
        raise ValueError(f"{fname}: no recurrence equation given")
    if isinstance(rec, sp.Equality):
        rec = rec.lhs - rec.rhs
    return a, rec, ics


def _rsolve_core(eq, fexpr, n, fname):
    """(name, list of SymPy solutions for a(n)) with every solution checked against the recurrence."""
    nv = variable(n, fname)
    if isinstance(fexpr, Symbol):
        name = fexpr.name
    elif isinstance(fexpr, Expr) and isinstance(fexpr.head, str) and len(fexpr.args) == 1 and \
            isinstance(fexpr.args[0], Symbol) and fexpr.args[0].name == nv.name:
        name = fexpr.head
    else:
        raise ValueError(f"{fname}: the unknown must be given as a[n] or a")
    a, rec, ics = _rec_parts(eq, name, nv, fname)
    sol = sp.rsolve(rec, a(nv), ics or None)
    if sol is None:
        raise NotImplementedError(f"{fname}: SymPy could not solve this recurrence")
    sols = sol if isinstance(sol, list) else [sol]
    if sols and isinstance(sols[0], dict):
        sols = [s_[a(nv)] for s_ in sols]
    checked = []
    for s_ in sols:
        s_ = sp.simplify(s_)
        resid = rec.replace(lambda e: isinstance(e, AppliedUndef) and e.func == a,
                            lambda e: s_.subs(nv, e.args[0]))
        ok = sp.simplify(sp.expandfunc(resid) if hasattr(sp, "expandfunc") else resid) == 0
        if not ok:
            consts = sorted((c for c in s_.free_symbols if c.name[:1] == "C" and c.name[1:].isdigit()), key=str)
            vals = {c: sp.Rational(3 + i, 7) for i, c in enumerate(consts)}
            others = {p: sp.Rational(5 + i, 11) for i, p in enumerate(sorted(resid.free_symbols - {nv} - set(consts),
                                                                               key=str))}
            ok = all(abs(complex(sp.N(resid.subs(vals).subs(others).subs(nv, k), 30))) < 1e-15 for k in range(2, 9))
        if not ok:
            raise ArithmeticError(f"{fname}: the solution failed verification")
        for lhs, rhs in ics.items():
            if sp.simplify(s_.subs(nv, lhs.args[0]) - rhs) != 0:
                raise ArithmeticError(f"{fname}: the solution does not satisfy the initial conditions")
        consts = {c: sp.Symbol(f"C{int(c.name[1:]) + 1}") for c in s_.free_symbols
                  if c.name[:1] == "C" and c.name[1:].isdigit()}
        checked.append(s_.xreplace(consts))
    return name, nv, checked


def _with_constants(e):
    return K._constants_to_c(e)


@replace("RSolve", AREA,
         "batch 4: RSolve[eqns, a[n], n] -> {{a[n] -> ...}} and RSolve[eqns, a, n] -> {{a -> Function[{n}, ...]}}; "
         "linear recurrences with constant or polynomial coefficients and initial conditions (SymPy rsolve); every "
         "solution is substituted back into the recurrence and the initial conditions; free constants are C[1], "
         "C[2], ...; systems raise")
def _rsolve(eq, fexpr, n):
    name, nv, sols = _rsolve_core(eq, fexpr, n, "RSolve")
    rules = []
    for s_ in sols:
        if isinstance(fexpr, Symbol):
            rules.append(L(rule(fexpr, Expr("Function", L(Symbol(nv.name)), out(s_)))))
        else:
            rules.append(L(rule(fexpr, out(s_))))
    return _with_constants(L(*rules))


@fn("RSolveValue", area=AREA, status="added",
    notes="RSolveValue[eqns, a[n], n] gives the solution value, RSolveValue[eqns, a, n] a pure function, "
          "RSolveValue[eqns, expr, n] the value of expr (e.g. a[10]); checked like RSolve")
def _rsolvevalue(eq, target, n):
    nv = variable(n, "RSolveValue")
    if isinstance(target, Symbol):
        fexpr, name = target, target.name
    elif isinstance(target, Expr) and isinstance(target.head, str) and len(target.args) == 1:
        fexpr, name = Expr(target.head, Symbol(nv.name)), target.head
    else:
        raise ValueError("RSolveValue: the unknown must be given as a[n] or a")
    _, _, sols = _rsolve_core(eq, fexpr, n, "RSolveValue")
    if len(sols) != 1:
        raise NotImplementedError("RSolveValue: several solutions")
    s_ = sols[0]
    if isinstance(target, Symbol):
        return _with_constants(Expr("Function", L(Symbol(nv.name)), out(s_)))
    val = s_.subs(sp.Symbol(nv.name), sym(target.args[0], "RSolveValue"))
    return _with_constants(out(sp.simplify(val)))


_orig_dsolve = K._BUILTINS["DSolve"]


def _check_ode_solution(eqs, sol_rules, ev, fname):
    """Substitute a DSolve solution into the equations; symbolic zero or a numeric check at sample points."""
    import random
    rnd = random.Random(11)
    for e in eqs:
        if not (isinstance(e, Expr) and e.head == "Equal" and len(e.args) == 2):
            continue
        res = ev(Expr("ReplaceAll", Expr("Subtract", e.args[0], e.args[1]), sol_rules))
        try:
            v = K.to_sp(res)
        except Exception:
            raise ArithmeticError(f"{fname}: could not verify the solution")
        if isinstance(v, list) or v.has(sp.Derivative) or any(isinstance(t, AppliedUndef) for t in sp.preorder_traversal(v)):
            raise ArithmeticError(f"{fname}: could not verify the solution")
        try:
            if sp.simplify(v) == 0:
                continue
        except Exception:
            pass
        free = sorted(v.free_symbols, key=str)
        good = 0
        for _ in range(5):
            vals = {s_: sp.Rational(rnd.randint(3, 19), 10) for s_ in free}
            try:
                z = complex(sp.N(v.subs(vals), 30))
            except Exception:
                continue
            if abs(z) > 1e-12:
                raise ArithmeticError(f"{fname}: the solution failed verification")
            good += 1
        if good == 0:
            raise ArithmeticError(f"{fname}: could not verify the solution")


def _dsolve_checked(a, ev, expr, fname):
    if len(a) != 3:
        raise ValueError(f"{fname} expects {fname}[eqns, y, x]")
    if is_list(a[2]):
        raise NotImplementedError(f"{fname}: partial differential equations are not supported")
    res = _orig_dsolve(a, ev, expr)
    if not is_list(res):
        return res
    eqs = KL_equations(a[0])
    funcs = list(a[1].args) if is_list(a[1]) else [a[1]]
    for sol in res.args:
        # rules for y[x] forms are turned into y -> Function[{x}, ...] for substitution into derivatives
        subs = []
        for r in sol.args:
            lhs, rhs = r.args
            if isinstance(lhs, Expr) and isinstance(lhs.head, str):
                subs.append(rule(Symbol(lhs.head), Expr("Function", L(*lhs.args), rhs)))
            else:
                subs.append(r)
        _check_ode_solution(eqs, L(*subs), ev, fname)
    return res


def KL_equations(e):
    if is_list(e) or (isinstance(e, Expr) and e.head == "And"):
        return [x for y in e.args for x in KL_equations(y)]
    return [e]


@kernel_override("DSolve", AREA,
                 "batch 4: every solution returned by DSolve (ODEs and ODE systems in one variable, initial "
                 "conditions, y[x] or y forms) is now substituted back into the equations and checked (symbolically, "
                 "else numerically at sample points); PDEs raise. Free constants may be numbered differently from "
                 "Wolfram (the solution family is the same)")
def _b_dsolve(a, ev, expr):
    return _dsolve_checked(a, ev, expr, "DSolve")


@fn("DSolveValue", area=AREA, status="added",
    notes="DSolveValue[eqns, y[x], x] gives the solution value, DSolveValue[eqns, y, x] a pure function, "
          "DSolveValue[eqns, expr, x] the value of expr; several solutions give a list; checked like DSolve")
def _dsolvevalue(eqs, target, x):
    from ..evaluate import evaluate
    ev = lambda e: evaluate(e, 1)   # noqa: E731
    if isinstance(target, Symbol) or (isinstance(target, Expr) and isinstance(target.head, str)
                                      and len(target.args) == 1 and target.args[0] == x):
        unknown = target
    else:
        heads = sorted({e.head for e in _applied(eqs) if isinstance(e.head, str)})
        if len(heads) != 1:
            raise ValueError("DSolveValue: cannot identify the unknown function")
        unknown = Symbol(heads[0])
    res = _dsolve_checked([eqs, unknown, x], ev, Expr("DSolve", eqs, unknown, x), "DSolveValue")
    if not is_list(res):
        raise Unevaluated()
    vals = [ev(Expr("ReplaceAll", target, sol)) for sol in res.args]
    return vals[0] if len(vals) == 1 else L(*vals)


def _applied(e):
    outl = []
    if isinstance(e, Expr):
        if isinstance(e.head, str) and e.head not in ("List", "Equal", "Plus", "Times", "Power", "And") and \
                len(e.args) == 1 and isinstance(e.args[0], Symbol):
            outl.append(e)
        for a in e.args:
            outl.extend(_applied(a))
    return outl


@fn("ImplicitD", area=AREA, status="added",
    notes="ImplicitD[f == g, y, x] = dy/dx for y defined implicitly (ImplicitD[x^2 + y^2 == 1, y, x] -> -x/y), "
          "ImplicitD[eqn, y, {x, n}] higher derivatives (expressed in x and y), ImplicitD[{eqns}, {y1, y2}, x] "
          "for systems")
def _implicitd(eqn, ys, xspec):
    if is_list(xspec):
        if len(xspec.args) != 2:
            raise ValueError("ImplicitD: give {x, n}")
        x, n = xspec.args[0], nonneg_int(xspec.args[1], "ImplicitD", "order")
    else:
        x, n = xspec, 1
    xv = variable(x, "ImplicitD")
    eq_list = list(eqn.args) if is_list(eqn) else [eqn]
    y_list = list(ys.args) if is_list(ys) else [ys]
    if not all(isinstance(y, Symbol) for y in y_list) or len(eq_list) != len(y_list):
        raise ValueError("ImplicitD: give one equation per dependent variable")
    yv = [sp.Symbol(y.name) for y in y_list]
    funcs = {y: sp.Function(y.name)(xv) for y in yv}
    exprs = []
    for e in eq_list:
        if not (isinstance(e, Expr) and e.head == "Equal" and len(e.args) == 2):
            raise ValueError("ImplicitD: expected equations")
        exprs.append((sym(e.args[0], "ImplicitD") - sym(e.args[1], "ImplicitD")).xreplace(funcs))
    derivs = []
    current = list(exprs)
    for order in range(1, n + 1):
        unknowns = [sp.Derivative(funcs[y], (xv, order)) for y in yv]
        current = [sp.diff(c, xv) for c in (exprs if order == 1 else current)]
        syms_ = [sp.Dummy(f"d{i}") for i in range(len(yv))]
        eqs = [c.xreplace(dict(zip(unknowns, syms_))) for c in current]
        for k in range(len(derivs) - 1, -1, -1):        # highest known order first
            eqs = [e.xreplace({sp.Derivative(funcs[y], (xv, k + 1)) if k else sp.Derivative(funcs[y], xv): val
                               for y, val in zip(yv, derivs[k])}) for e in eqs]
        eqs = [sp.expand(e) for e in eqs]
        sol = sp.solve(eqs, syms_, dict=True)
        if len(sol) != 1:
            raise NotImplementedError("ImplicitD: cannot solve for the derivative")
        derivs.append([sp.simplify(sol[0][s_]) for s_ in syms_])
    res = [sp.simplify(v.xreplace({F: y for y, F in funcs.items()})) for v in derivs[-1]]
    return out(res[0]) if not is_list(ys) else L(*(out(r) for r in res))


# ───────────────────────────── integral transforms and Fourier series ─────────────────────────────

from . import transforms as _TR   # noqa: E402

_fourier_default = _TR._fourier
_lt_default = _TR._lt


def _fparams(opts, fname, default):
    p = opts.get("FourierParameters")
    if p is None:
        return default
    if not (is_list(p) and len(p.args) == 2):
        raise ValueError(f"{fname}: FourierParameters must be {{a, b}}")
    a, b = sym(p.args[0], fname), sym(p.args[1], fname)
    if not (a.is_real and a.is_number and b.is_real and b.is_number and b != 0):
        raise NotImplementedError(f"{fname}: FourierParameters must be numeric {{a, b}} with b != 0")
    return a, b


def _equation_thread(e, f):
    if isinstance(e, Expr) and e.head == "Equal":
        return Expr("Equal", *(f(x) for x in e.args))
    if is_list(e):
        return L(*(f(x) for x in e.args))
    return None


def _ft_general(fname, e, t, w, rest, inverse):
    _, opts = split_opts(rest, {"FourierParameters", "Assumptions", "GenerateConditions"}, fname)
    if _:
        raise ValueError(f"{fname}: unexpected arguments")
    th = _equation_thread(e, lambda x: _ft_general(fname, x, t, w, rest, inverse))
    if th is not None:
        return th
    a, b = _fparams(opts, fname, (sp.Integer(0), sp.Integer(1)))
    if (a, b) == (0, 1):
        return _fourier_default(fname, e, t, w, -1 if inverse else 1)
    wv = variable(w, fname)
    tmp = Symbol("FTtmp$")
    D = sym(_fourier_default(fname, e, t, tmp, -1 if inverse else 1), fname)
    D = D.subs(sp.Symbol("FTtmp$"), b * wv)
    c = sp.sqrt(sp.Abs(b) / (2 * sp.pi) ** (1 + a if inverse else 1 - a)) * sp.sqrt(2 * sp.pi)
    return out(sp.simplify(c * D))


@replace("FourierTransform", _TR.AREA,
         "Wolfram convention (1/√(2π)) ∫ f e^(iωt) dt, numerically verified at frequencies of both signs (raises "
         "if SymPy's closed form cannot be verified); batch 4: FourierParameters -> {a, b} (numeric) gives "
         "Sqrt[|b|/(2π)^(1-a)] ∫ f e^(i b ω t) dt, threads over equations and lists")
def _ft4(e, t, w, *rest):
    return _ft_general("FourierTransform", e, t, w, rest, False)


@replace("InverseFourierTransform", _TR.AREA,
         "Wolfram convention (1/√(2π)) ∫ F e^(-iωt) dω, numerically verified; batch 4: FourierParameters -> "
         "{a, b} gives Sqrt[|b|/(2π)^(1+a)] ∫ F e^(-i b ω t) dω, threads over equations and lists")
def _ift4(e, w, t, *rest):
    return _ft_general("InverseFourierTransform", e, w, t, rest, True)


def _half_line_numeric(f, tv, kern, params):
    import mpmath as mp
    g = sp.lambdify(tv, f.subs(params), "mpmath")

    def val(w):
        return mp.quadosc(lambda tt: g(tt) * kern(w * tt), [0, mp.inf], omega=abs(w))
    return val


def _sincos_transform(fname, e, t, w, rest, kind, inverse):
    import mpmath as mp
    _, opts = split_opts(rest, {"FourierParameters", "Assumptions", "GenerateConditions"}, fname)
    if _:
        raise ValueError(f"{fname}: unexpected arguments")
    th = _equation_thread(e, lambda x: _sincos_transform(fname, x, t, w, rest, kind, inverse))
    if th is not None:
        return th
    a, b = _fparams(opts, fname, (sp.Integer(0), sp.Integer(1)))
    f = to_heaviside(sym(e, fname))
    tv, wv = variable(t, fname), variable(w, fname)
    tp = sp.Symbol(tv.name, positive=True)
    f = f.xreplace({tv: tp})
    wp = sp.Symbol("w_", positive=True)
    trig = sp.sin if kind == "sin" else sp.cos
    try:
        r = sp.integrate(f * trig(wp * tp), (tp, 0, sp.oo), conds="none")
    except Exception:
        raise Unevaluated()
    if r.has(sp.Integral) or r.has(sp.meijerg) or r.has(sp.zoo, sp.nan) or r.has(sp.exp_polar):
        raise Unevaluated()
    if r.has(sp.oo):
        raise NotImplementedError(f"{fname}: the transform does not exist as a function (divergent integral)")
    params = {s_: sp.Rational(7, 5) + i * sp.Rational(3, 10)
              for i, s_ in enumerate(sorted(f.free_symbols - {tp}, key=str))}
    kern = mp.sin if kind == "sin" else mp.cos
    try:
        num = _half_line_numeric(f, tp, kern, params)
        ref = [(x, complex(num(x))) for x in (0.45, 1.3, 2.2)]
    except Exception:
        raise NotImplementedError(f"{fname}: could not verify the transform numerically")
    cf = sp.lambdify(wp, r.subs(params), "mpmath")
    try:
        ok = all(abs(complex(cf(x)) - v) <= 1e-7 * max(1.0, abs(v)) for x, v in ref)
    except Exception:
        ok = False
    if not ok:
        raise NotImplementedError(f"{fname}: SymPy's closed form failed numerical verification for this input")
    # Wolfram returns the closed form valid for ω > 0 (the transforms are defined on the half line)
    wr = sp.Symbol(wv.name, positive=True)
    base = r.subs(wp, wr)
    if b < 0:
        base = (-1 if kind == "sin" else 1) * base
        b = -b
    c = 2 * sp.sqrt(sp.Abs(b) / (2 * sp.pi) ** ((1 + a) if inverse else (1 - a)))
    res = sp.simplify(c * base.subs(wr, b * wr))
    return out(res.xreplace({wr: wv}))


def _mk_sincos(name, kind, inverse):
    lo = "∫_0^∞ f sin(ωt) dt" if kind == "sin" else "∫_0^∞ f cos(ωt) dt"

    @fn(name, area=_TR.AREA, status="added",
        notes=f"{name}[f, t, ω] = Sqrt[2/π] {lo} (Wolfram default); FourierParameters -> {{a, b}} (numeric) gives "
              f"2 Sqrt[|b|/(2π)^(1{'+' if inverse else '-'}a)] times the integral with frequency b ω; the closed "
              "form is checked numerically at three frequencies and returned in the form valid for ω > 0 (as Wolfram); threads over "
              "equations; unevaluated if SymPy has no closed form")
    def _impl(e, t, w, *rest):
        return _sincos_transform(name, e, t, w, rest, kind, inverse)
    return _impl


for _nm, _kd, _inv in (("FourierSinTransform", "sin", False), ("FourierCosTransform", "cos", False),
                       ("InverseFourierSinTransform", "sin", True), ("InverseFourierCosTransform", "cos", True)):
    _mk_sincos(_nm, _kd, _inv)


def _undef_transform_ok(r, cls):
    for node in sp.preorder_traversal(r):
        if isinstance(node, cls):
            if not isinstance(node.args[0], AppliedUndef):
                return False
    return True


@replace("LaplaceTransform", _TR.AREA,
         "∫_0^∞ f e^(-st) dt (same convention as Wolfram/SymPy), convergence conditions dropped; batch 4: "
         "derivatives of undefined functions (LaplaceTransform[f'[t], t, s] -> s LaplaceTransform[f[t], t, s] - "
         "f[0]), HeavisideTheta and DiracDelta shifts, threads over equations and lists; unevaluated when SymPy "
         "cannot transform")
def _lt4(e, t, s, *rest):
    _, opts = split_opts(rest, {"Assumptions", "GenerateConditions"}, "LaplaceTransform")
    th = _equation_thread(e, lambda x: _lt4(x, t, s, *rest))
    if th is not None:
        return th
    f = to_heaviside(sym(e, "LaplaceTransform"))
    tv, sv = variable(t, "LaplaceTransform"), variable(s, "LaplaceTransform")
    if isinstance(f, AppliedUndef):
        raise Unevaluated()
    r = sp.laplace_transform(f, tv, sv, noconds=True)
    if r.has(sp.LaplaceTransform):
        if not _undef_transform_ok(r, sp.LaplaceTransform):
            raise Unevaluated()
        r = sp.expand(r)
        return _laplace_out(r)
    return out(sp.simplify(r))


def _laplace_out(r):
    """SymPy result containing LaplaceTransform(f(t), t, s) of undefined functions -> Wolfram expression."""
    reps = {}
    for node in sp.preorder_traversal(r):
        if isinstance(node, sp.LaplaceTransform):
            d = sp.Dummy("LT")
            reps[node] = d
    rr = r.xreplace(reps)
    e = out(rr)
    from ..expr import Expr as E_
    back = {}
    for node, d in reps.items():
        back[d] = Expr("LaplaceTransform", out(node.args[0]), out(node.args[1]), out(node.args[2]))
    return _ev(Expr("ReplaceAll", e, L(*(rule(R(d), v) for d, v in back.items()))))


def _fs_opts(rest, fname):
    _, opts = split_opts(rest, {"FourierParameters", "Assumptions", "GenerateConditions"}, fname)
    if _:
        raise ValueError(f"{fname}: unexpected arguments")
    p = opts.get("FourierParameters")
    if p is not None:
        a, b = _fparams(opts, fname, (1, 1))
        if (a, b) != (1, 1):
            raise NotImplementedError(f"{fname}: only the default FourierParameters -> {{1, 1}} is supported")


def _fs_integral(g, tv, lo, hi, fname):
    try:
        r = sp.integrate(g, (tv, lo, hi))
    except Exception:
        raise Unevaluated()
    if r.has(sp.Integral) or r.has(sp.zoo, sp.nan, sp.oo):
        raise Unevaluated()
    return r


def _check_coeff(general, nsym, direct, fname):
    """Compare a symbolic-n coefficient with directly computed integer cases."""
    for k in (1, 2, 3, -1, -2, 5):
        if k < 0 and direct(k) is None:
            continue
        want = direct(k)
        got = general.subs(nsym, k)
        if want is None:
            continue
        if abs(complex(sp.N(sp.expand_complex(got - want), 30))) > 1e-20:
            raise ArithmeticError(f"{fname}: the general coefficient failed verification")


def _symbolic_coeff(make_integrand, tv, lo, hi, scale, n, fname, n0_scale=None, allow_negative=True):
    """Coefficient for symbolic n: Piecewise[{{c0, n == 0}}, g(n)] unless g(0) == c0."""
    nv = variable(n, fname)
    ni = sp.Symbol(nv.name, integer=True, nonzero=True)
    g = _fs_integral(make_integrand(ni), tv, lo, hi, fname) * scale
    g = g.rewrite(sp.cos) if g.has(sp.exp) and not make_integrand(0).has(sp.exp) else g
    if isinstance(g, sp.Piecewise):
        raise Unevaluated()
    g = sp.simplify(g)
    c0 = sp.simplify(_fs_integral(make_integrand(0), tv, lo, hi, fname) * (n0_scale if n0_scale is not None else scale))

    def direct(k):
        if k < 0 and not allow_negative:
            return None
        return _fs_integral(make_integrand(k), tv, lo, hi, fname) * scale
    _check_coeff(g, ni, direct, fname)
    g = g.xreplace({ni: sp.Symbol(nv.name)})
    try:
        g0 = sp.limit(g, sp.Symbol(nv.name), 0) if g.subs(sp.Symbol(nv.name), 0).has(sp.zoo, sp.nan) else \
            g.subs(sp.Symbol(nv.name), 0)
        same = sp.simplify(g.subs(sp.Symbol(nv.name), 0) - c0) == 0 and not g.subs(sp.Symbol(nv.name), 0).has(
            sp.zoo, sp.nan)
    except Exception:
        same = False
    if same:
        return out(g)
    return Expr("Piecewise", L(L(out(c0), Expr("Equal", n, 0))), out(g))


@fn("FourierCoefficient", area=AREA, status="added",
    notes="FourierCoefficient[f, t, n] = 1/(2π) ∫_{-π}^{π} f e^(-int) dt (Wolfram default FourierParameters "
          "{1, 1}; other parameters raise); integer or symbolic n (symbolic n is integer: a separate n == 0 "
          "value appears as Piecewise[{{c0, n == 0}}, g[n]]); the general term is checked against direct "
          "integrals at n = ±1, ±2, 3, 5")
def _fcoeff(e, t, n, *rest):
    _fs_opts(rest, "FourierCoefficient")
    tv = variable(t, "FourierCoefficient")
    f = to_heaviside(sym(e, "FourierCoefficient"))
    scale = 1 / (2 * sp.pi)
    nn = sym(n, "FourierCoefficient")
    if nn.is_integer:
        return out(sp.simplify(_fs_integral(f * sp.exp(-sp.I * nn * tv), tv, -sp.pi, sp.pi, "FourierCoefficient")
                               * scale))
    if not isinstance(n, Symbol):
        raise ValueError("FourierCoefficient: n must be an integer or a symbol")
    return _symbolic_coeff(lambda k: f * sp.exp(-sp.I * k * tv), tv, -sp.pi, sp.pi, scale, n, "FourierCoefficient")


def _trig_coeff(kind, e, t, n, rest, fname):
    _fs_opts(rest, fname)
    tv = variable(t, fname)
    f = to_heaviside(sym(e, fname))
    trig = sp.cos if kind == "cos" else sp.sin
    nn = sym(n, fname)
    if nn.is_integer:
        if nn < 0:
            raise ValueError(f"{fname}: n must be non-negative")
        scale = 1 / sp.pi if (kind == "cos" and nn == 0) else 2 / sp.pi
        return out(sp.simplify(_fs_integral(f * trig(nn * tv), tv, 0, sp.pi, fname) * scale))
    if not isinstance(n, Symbol):
        raise ValueError(f"{fname}: n must be an integer or a symbol")
    return _symbolic_coeff(lambda k: f * trig(k * tv), tv, 0, sp.pi, 2 / sp.pi, n, fname,
                           n0_scale=(1 / sp.pi if kind == "cos" else 2 / sp.pi), allow_negative=False)


@fn("FourierCosCoefficient", area=AREA, status="added",
    notes="FourierCosCoefficient[f, t, n] = 2/π ∫_0^π f cos(nt) dt, with 1/π ∫_0^π f dt for n = 0 (Wolfram); "
          "symbolic n checked against direct integrals")
def _fcc(e, t, n, *rest):
    return _trig_coeff("cos", e, t, n, rest, "FourierCosCoefficient")


@fn("FourierSinCoefficient", area=AREA, status="added",
    notes="FourierSinCoefficient[f, t, n] = 2/π ∫_0^π f sin(nt) dt (Wolfram); symbolic n checked against direct "
          "integrals")
def _fsc(e, t, n, *rest):
    return _trig_coeff("sin", e, t, n, rest, "FourierSinCoefficient")


def _fs_order(k, fname):
    return nonneg_int(k, fname, "order")


@fn("FourierSeries", area=AREA, status="added",
    notes="FourierSeries[f, t, k] = Σ_{n=-k}^{k} FourierCoefficient[f, t, n] E^(I n t) (Wolfram default "
          "parameters; others raise)")
def _fseries(e, t, k, *rest):
    _fs_opts(rest, "FourierSeries")
    kk = _fs_order(k, "FourierSeries")
    tv = variable(t, "FourierSeries")
    f = to_heaviside(sym(e, "FourierSeries"))
    tot = sp.Integer(0)
    for n in range(-kk, kk + 1):
        c = sp.simplify(_fs_integral(f * sp.exp(-sp.I * n * tv), tv, -sp.pi, sp.pi, "FourierSeries") / (2 * sp.pi))
        tot += c * sp.exp(sp.I * n * tv)
    return out(tot)


@fn("FourierTrigSeries", area=AREA, status="added",
    notes="FourierTrigSeries[f, t, k] = a0 + Σ_{n=1}^{k} (a_n Cos[n t] + b_n Sin[n t]) with a0 = 1/(2π)∫f, "
          "a_n = 1/π ∫_{-π}^{π} f cos(nt), b_n = 1/π ∫_{-π}^{π} f sin(nt) (Wolfram default parameters)")
def _ftrig(e, t, k, *rest):
    _fs_opts(rest, "FourierTrigSeries")
    kk = _fs_order(k, "FourierTrigSeries")
    tv = variable(t, "FourierTrigSeries")
    f = to_heaviside(sym(e, "FourierTrigSeries"))
    tot = sp.simplify(_fs_integral(f, tv, -sp.pi, sp.pi, "FourierTrigSeries") / (2 * sp.pi))
    for n in range(1, kk + 1):
        an = sp.simplify(_fs_integral(f * sp.cos(n * tv), tv, -sp.pi, sp.pi, "FourierTrigSeries") / sp.pi)
        bn = sp.simplify(_fs_integral(f * sp.sin(n * tv), tv, -sp.pi, sp.pi, "FourierTrigSeries") / sp.pi)
        tot += an * sp.cos(n * tv) + bn * sp.sin(n * tv)
    return out(tot)


@fn("FourierCosSeries", area=AREA, status="added",
    notes="FourierCosSeries[f, t, k] = Σ_{n=0}^{k} FourierCosCoefficient[f, t, n] Cos[n t] (even extension of f "
          "on [0, π])")
def _fcosser(e, t, k, *rest):
    _fs_opts(rest, "FourierCosSeries")
    kk = _fs_order(k, "FourierCosSeries")
    tv = variable(t, "FourierCosSeries")
    f = to_heaviside(sym(e, "FourierCosSeries"))
    tot = sp.Integer(0)
    for n in range(0, kk + 1):
        sc = 1 / sp.pi if n == 0 else 2 / sp.pi
        tot += sp.simplify(_fs_integral(f * sp.cos(n * tv), tv, 0, sp.pi, "FourierCosSeries") * sc) * sp.cos(n * tv)
    return out(tot)


@fn("FourierSinSeries", area=AREA, status="added",
    notes="FourierSinSeries[f, t, k] = Σ_{n=1}^{k} FourierSinCoefficient[f, t, n] Sin[n t] (odd extension of f "
          "on [0, π])")
def _fsinser(e, t, k, *rest):
    _fs_opts(rest, "FourierSinSeries")
    kk = _fs_order(k, "FourierSinSeries")
    tv = variable(t, "FourierSinSeries")
    f = to_heaviside(sym(e, "FourierSinSeries"))
    tot = sp.Integer(0)
    for n in range(1, kk + 1):
        tot += sp.simplify(_fs_integral(f * sp.sin(n * tv), tv, 0, sp.pi, "FourierSinSeries") * 2 / sp.pi) \
            * sp.sin(n * tv)
    return out(tot)


@fn("MellinTransform", area=_TR.AREA, status="added",
    notes="MellinTransform[f, x, s] = ∫_0^∞ x^(s-1) f dx (Wolfram); convergence strip dropped (generic value), "
          "checked numerically at two real s inside the strip; unevaluated if SymPy has no closed form")
def _mellin(e, x, s, *rest):
    import mpmath as mp
    _, opts = split_opts(rest, {"Assumptions", "GenerateConditions"}, "MellinTransform")
    f = sym(e, "MellinTransform")
    xv, sv = variable(x, "MellinTransform"), variable(s, "MellinTransform")
    try:
        r = sp.mellin_transform(f, xv, sv)
    except Exception:
        raise Unevaluated()
    if isinstance(r, sp.MellinTransform) or not isinstance(r, tuple):
        raise Unevaluated()
    val, (lo, hi), _c = r
    if val.has(sp.MellinTransform, sp.meijerg):
        raise Unevaluated()
    if not f.free_symbols - {xv} and lo.is_number and hi.is_number:
        lo_f = float(lo) if lo.is_finite else -5.0
        hi_f = float(hi) if hi.is_finite else lo_f + 6.0
        if lo_f == -5.0 and not lo.is_finite:
            lo_f = hi_f - 6.0
        g = sp.lambdify(xv, f, "mpmath")
        for frac in (0.35, 0.7):
            s0 = lo_f + frac * (hi_f - lo_f)
            num = mp.quad(lambda y: mp.exp(s0 * y) * g(mp.exp(y)), [-mp.inf, 0, mp.inf])
            got = complex(sp.N(val.subs(sv, s0)))
            if abs(got - complex(num)) > 1e-7 * max(1.0, abs(num)):
                raise NotImplementedError("MellinTransform: SymPy's closed form failed numerical verification")
    return out(sp.simplify(val))


@fn("InverseMellinTransform", area=_TR.AREA, status="added",
    notes="InverseMellinTransform[F, s, x] via SymPy (fundamental strip inferred); result checked by transforming "
          "back with MellinTransform; unevaluated otherwise")
def _imellin(e, s, x, *rest):
    F = sym(e, "InverseMellinTransform")
    sv, xv = variable(s, "InverseMellinTransform"), variable(x, "InverseMellinTransform")
    xp = sp.Symbol(xv.name, positive=True)
    r = None
    for strip in ((0, sp.oo), (-sp.oo, 0), (0, 1), (-1, 0)):
        try:
            r = sp.inverse_mellin_transform(F, sv, xp, strip)
        except Exception:
            r = None
            continue
        if not r.has(sp.InverseMellinTransform, sp.meijerg):
            break
        r = None
    if r is None:
        raise Unevaluated()
    r = sp.simplify(r.rewrite(sp.Piecewise)) if r.has(sp.Heaviside) else sp.simplify(r)
    try:
        back = sp.mellin_transform(r, xp, sv, noconds=True)
        if sp.simplify(back - F) != 0:
            raise NotImplementedError("InverseMellinTransform: result failed verification")
    except NotImplementedError:
        raise
    except Exception:
        raise NotImplementedError("InverseMellinTransform: could not verify the result")
    return out(r.xreplace({xp: xv}))


# ───────────────────────────── vector calculus in curvilinear coordinates, curves ─────────────────────────────

from . import calculus as _CA   # noqa: E402


def _chart(c, n, fname):
    name = c if isinstance(c, str) else None
    if name is None:
        raise ValueError(f"{fname}: the coordinate system must be a string such as \"Spherical\"")
    return name


def _scale_factors(name, xs, fname):
    n = len(xs)
    if name == "Cartesian":
        return [sp.Integer(1)] * n
    if name == "Polar" and n == 2:
        return [sp.Integer(1), xs[0]]
    if name == "Cylindrical" and n == 3:
        return [sp.Integer(1), xs[0], sp.Integer(1)]
    if name == "Spherical" and n == 3:
        return [sp.Integer(1), xs[0], xs[0] * sp.sin(xs[1])]
    raise NotImplementedError(f"{fname}: coordinate system {name!r} in {n} dimensions is not supported "
                              "(Cartesian, Polar {r, θ}, Cylindrical {r, θ, z}, Spherical {r, θ, φ})")


def _vc_setup(vs, chart, fname):
    xs = _CA._vars_list(vs, fname)
    h = _scale_factors(_chart(chart, len(xs), fname), xs, fname)
    return xs, h


def _simp(v):
    return sp.simplify(v)


def _grad_c(f, xs, h):
    return [_simp(sp.diff(f, x) / hi) for x, hi in zip(xs, h)]


def _div_c(F, xs, h):
    H = sp.Mul(*h)
    return _simp(sp.Add(*(sp.diff(H * Fi / hi, x) for Fi, x, hi in zip(F, xs, h))) / H)


def _curl_c(F, xs, h):
    if len(xs) == 2:
        return _simp((sp.diff(h[1] * F[1], xs[0]) - sp.diff(h[0] * F[0], xs[1])) / (h[0] * h[1]))
    res = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        res.append(_simp((sp.diff(h[k] * F[k], xs[j]) - sp.diff(h[j] * F[j], xs[k])) / (h[j] * h[k])))
    return res


def _wrap_vc(name, impl3, notes):
    orig = REGISTRY_get(name)

    @replace(name, _CA.AREA, notes)
    def _w(f, vs, *chart):
        if not chart:
            return orig(f, vs)
        if len(chart) != 1:
            raise ValueError(f"{name}: expected {name}[f, vars, chart]")
        xs, h = _vc_setup(vs, chart[0], name)
        return impl3(f, xs, h)
    return _w


def REGISTRY_get(name):
    """The existing implementation(s) of a library function as one callable (dispatch on argument count)."""
    from .core import REGISTRY
    impls = list(REGISTRY[name].impls)

    def call(*args):
        for f_, lo, hi in impls:
            if len(args) >= lo and (hi is None or len(args) <= hi):
                return f_(*args)
        raise ValueError(f"{name} called with {len(args)} argument(s)")
    return call


def _g3(f, xs, h):
    if is_list(f):
        raise NotImplementedError("Grad: the gradient of a vector field in curvilinear coordinates is not supported")
    return out(_grad_c(sym(f, "Grad"), xs, h))


def _d3(F, xs, h):
    comps = [sym(c, "Div") for c in list_arg(F, "Div", "vector field")]
    _CA._same_len(comps, xs, "Div")
    return out(_div_c(comps, xs, h))


def _c3(F, xs, h):
    comps = [sym(c, "Curl") for c in list_arg(F, "Curl", "vector field")]
    _CA._same_len(comps, xs, "Curl")
    if len(comps) not in (2, 3):
        raise ValueError("Curl: only 2- and 3-dimensional fields are supported")
    return out(_curl_c(comps, xs, h))


def _l3(f, xs, h):
    v = sym(f, "Laplacian")
    return out(_div_c(_grad_c(v, xs, h), xs, h))


from .core import list_arg   # noqa: E402

_wrap_vc("Grad", _g3, "Grad[f, {x1, ..., xn}] (Cartesian; a vector field gives the Jacobian); batch 4: "
         "Grad[f, vars, chart] for \"Cartesian\", \"Polar\" {r, θ}, \"Cylindrical\" {r, θ, z}, \"Spherical\" "
         "{r, θ, φ} (θ polar angle) in the orthonormal basis (Wolfram)")
_wrap_vc("Div", _d3, "Div[{f1, ..., fn}, {x1, ..., xn}] (Cartesian); batch 4: Div[F, vars, chart] for Polar, "
         "Cylindrical, Spherical (orthonormal components, Wolfram)")
_wrap_vc("Curl", _c3, "3D curl and the 2D scalar curl; batch 4: Curl[F, vars, chart] for Polar (scalar), "
         "Cylindrical and Spherical (orthonormal components, Wolfram)")
_wrap_vc("Laplacian", _l3, "Laplacian[f, {x1, ..., xn}] (Cartesian); batch 4: Laplacian[f, vars, chart] for "
         "Polar, Cylindrical, Spherical (Wolfram)")


_orig_arclength = REGISTRY_get("ArcLength")


@replace("ArcLength", _CA.AREA,
         "ArcLength[f, {x, a, b}] = ∫ Sqrt[1 + f'^2] (graph of f, MikoshiLang); batch 4: Wolfram's parametric form "
         "ArcLength[{x1[t], ..., xn[t]}, {t, a, b}] = ∫ Sqrt[Σ xi'^2] dt, ArcLength[Line[{p1, ...}]] (polyline), "
         "ArcLength[Circle[c, r]] = 2 π r; numeric input gives numeric output")
def _arclength4(f, *spec):
    if not spec:
        if isinstance(f, Expr) and f.head == "Line" and len(f.args) == 1:
            pts = [[sym(c, "ArcLength") for c in list_arg(p, "ArcLength", "point")]
                   for p in list_arg(f.args[0], "ArcLength", "point list", 2)]
            tot = sp.Add(*(sp.sqrt(sp.Add(*((a - b) ** 2 for a, b in zip(p, q)))) for p, q in zip(pts, pts[1:])))
            return numeric_out(sp.simplify(tot), f)
        if isinstance(f, Expr) and f.head == "Circle" and len(f.args) in (1, 2):
            r = sym(f.args[1], "ArcLength") if len(f.args) == 2 else sp.Integer(1)
            if is_list(f.args[1] if len(f.args) == 2 else L()) and len(f.args) == 2:
                raise NotImplementedError("ArcLength: ellipses are not supported")
            return numeric_out(2 * sp.pi * r, f)
        raise ValueError("ArcLength: expected a curve and {t, a, b}, a Line or a Circle")
    if len(spec) != 1:
        raise ValueError("ArcLength: expected ArcLength[curve, {t, a, b}]")
    if is_list(f):
        t, a, b = _CA._int_spec(spec[0], "ArcLength")
        comps = [sym(c, "ArcLength") for c in f.args]
        tr = sp.Symbol(t.name, real=True)
        integrand = sp.simplify(sp.sqrt(sp.Add(*(sp.diff(c.xreplace({t: tr}), tr) ** 2 for c in comps))))
        val = sp.integrate(integrand, (tr, a, b))
        if isinstance(val, sp.Basic) and val.has(sp.Integral):
            raise Unevaluated()
        return numeric_out(_CA._check_result(sp.simplify(val), "ArcLength"), f, spec[0])
    return _orig_arclength(f, spec[0])


@fn("ArcCurvature", area=_CA.AREA, status="added",
    notes="ArcCurvature[{x1[t], ..., xn[t]}, t] = Sqrt[|r'|^2 |r''|^2 - (r'.r'')^2]/|r'|^3 (t real; gives "
          "|x'y'' - y'x''|/(x'^2 + y'^2)^(3/2) in 2D, |r' x r''|/|r'|^3 in 3D); ArcCurvature[f, x] for the graph "
          "y = f(x): Abs[f'']/(1 + f'^2)^(3/2)")
def _arccurv(f, t, *rest):
    if rest:
        raise NotImplementedError("ArcCurvature: coordinate charts are not supported")
    tv = variable(t, "ArcCurvature")
    tr = sp.Symbol(tv.name, real=True)
    comps = [sym(c, "ArcCurvature") for c in f.args] if is_list(f) else [tr, sym(f, "ArcCurvature")]
    comps = [c.xreplace({tv: tr}) for c in comps]
    d1 = [sp.diff(c, tr) for c in comps]
    d2 = [sp.diff(c, tr, 2) for c in comps]
    if len(comps) == 2:
        num = sp.Abs(d1[0] * d2[1] - d1[1] * d2[0])
    else:
        n1 = sp.Add(*(a ** 2 for a in d1))
        n2 = sp.Add(*(a ** 2 for a in d2))
        dot = sp.Add(*(a * b for a, b in zip(d1, d2)))
        num = sp.sqrt(sp.simplify(n1 * n2 - dot ** 2))
    den = sp.Add(*(a ** 2 for a in d1)) ** sp.Rational(3, 2)
    return out(sp.simplify(num / den).xreplace({tr: tv}))
