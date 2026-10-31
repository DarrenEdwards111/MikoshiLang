"""Wolfram batch W05: polynomials, equation manipulation, Piecewise, cylindrical decomposition and function
properties.

Every function returns a result that is correct by construction (exact polynomial algorithms from SymPy, or
results substituted back / checked against their definition), raises a clear error, or stays unevaluated.
Where a Wolfram feature is only partly supported, the unsupported inputs raise instead of guessing.
"""
from __future__ import annotations

import itertools
import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, record, R, L, rule, sym, is_list, Unevaluated
from .wl_symbolic import out, split_opts, is_true, is_false, replace

AREA = "algebra (W05)"


def ev(x):
    from ..evaluate import evaluate
    return evaluate(x, 1)


def _sym(name):
    return Symbol(name)


def _is_sym(x, name=None):
    return isinstance(x, Symbol) and (name is None or x.name == name)


def _is_head(x, *heads):
    return isinstance(x, Expr) and isinstance(x.head, str) and x.head in heads


def _tf(x):
    """True/False for a Wolfram Boolean, else None."""
    if x is True or _is_sym(x, "True"):
        return True
    if x is False or _is_sym(x, "False"):
        return False
    return None


def _S(x, fname):
    return sym(x, fname)


def _opts(args, names, fname):
    """Trailing options with names in ``names`` (any other trailing rule naming an option raises)."""
    args = list(args)
    opts = {}
    while args and _is_head(args[-1], "Rule", "RuleDelayed") and isinstance(args[-1].args[0], Symbol) \
            and (args[-1].args[0].name in names or args[-1].args[0].name[:1].isupper()):
        name = args[-1].args[0].name
        if name not in names:
            raise NotImplementedError(f"{fname}: the option {name} is not supported")
        opts[name] = args.pop().args[1]
    return args, opts


def _vars(v, fname):
    """A variable or list of variables (symbols or other non-numeric expressions such as x[1])."""
    vs = list(v.args) if is_list(v) else [v]
    for x in vs:
        if isinstance(x, (int, float, str, bool)) or (isinstance(x, Expr) and x.head in ("Plus", "Times", "Power")
                                                      and False):
            raise ValueError(f"{fname}: {x!r} is not a valid variable")
    return vs


# ───────────────────────────── system variables and option names ─────────────────────────────

from .wl_numerics import SYSTEM_CONSTANTS as _SYSC   # noqa: E402

_SYSC.setdefault("$MaxRootDegree", 1024)
_SYSC.setdefault("$MaxPiecewiseCases", 100)
record("$MaxRootDegree", AREA, "added",
       "system variable: the maximum degree of a polynomial in Root objects and root isolation (1024, as Wolfram); "
       "RootIntervals, IsolatingInterval and NSolveValues raise for higher degrees")
record("$MaxPiecewiseCases", AREA, "added",
       "system variable: the maximum number of cases PiecewiseExpand creates (100, as Wolfram); larger expansions "
       "raise instead of returning a partial result")
record("$Assumptions", AREA, "added",
       "system variable: the default assumptions (True); set with Block/Assuming and used by Simplify, Refine, "
       "Integrate, AddSides/MultiplySides and the Function* property tests")
record("Undefined", AREA, "added",
       "symbol for an undefined value: ConditionalExpression[e, False] and an empty Piecewise case give Undefined; "
       "the Function* tests treat it as a point outside the domain. (Arithmetic propagation 1 + Undefined -> "
       "Undefined is not automatic: the sum stays as it is, it is never given a value)")


# ───────────────────────────── Piecewise ─────────────────────────────

K.BUILTIN_ATTRIBUTES["Piecewise"] = {"HoldAll", "Protected"}


def _pw_parts(pw, fname="Piecewise"):
    if not (isinstance(pw, Expr) and pw.head == "Piecewise" and 1 <= len(pw.args) <= 2 and is_list(pw.args[0])):
        raise ValueError(f"{fname}: expected Piecewise[{{{{val, cond}}, ...}}, default]")
    cases = []
    for c in pw.args[0].args:
        if not (is_list(c) and len(c.args) == 2):
            raise ValueError(f"{fname}: each case must be a pair {{value, condition}}")
        cases.append((c.args[0], c.args[1]))
    return cases, (pw.args[1] if len(pw.args) == 2 else 0)


@fn("Piecewise", area=AREA, status="added",
    notes="Piecewise[{{v1, c1}, ...}, def] (HoldAll, as Wolfram): conditions are evaluated in order, False cases are "
          "dropped, the first True case gives its value (later cases are dropped), Piecewise[{}, d] -> d, the "
          "default is 0; values are evaluated only when they can be returned. Symbolic cases stay; Integrate, D, "
          "Limit and PiecewiseExpand work with it through SymPy's Piecewise")
def _piecewise(*args):
    if len(args) not in (1, 2):
        raise ValueError("Piecewise called with the wrong number of arguments; expected 1 or 2")
    cl = ev(args[0])
    if not is_list(cl):
        raise ValueError("Piecewise: the first argument must be a list of {value, condition} pairs")
    default = args[1] if len(args) == 2 else 0
    kept = []
    for c in cl.args:
        if not (is_list(c) and len(c.args) == 2):
            raise ValueError("Piecewise: each case must be a pair {value, condition}")
        cond = ev(c.args[1])
        t = _tf(cond)
        if t is False:
            continue
        if t is True:
            if not kept:
                return ev(c.args[0])
            default = c.args[0]
            break
        kept.append((c.args[0], cond))
    if not kept:
        return ev(default)
    d = ev(default)
    res = Expr("Piecewise", L(*(L(ev(v), cnd) for v, cnd in kept)), *([d] if len(args) == 2 or d != 0 else []))
    if len(args) == 1 and d == 0:
        res = Expr("Piecewise", res.args[0])
    return res


# ───────────────────────────── DirectedInfinity ─────────────────────────────

@fn("DirectedInfinity", area=AREA, status="added",
    notes="DirectedInfinity[z]: an infinite quantity in the direction of z. DirectedInfinity[] and "
          "DirectedInfinity[0 | Indeterminate] -> ComplexInfinity; a positive real -> Infinity, a negative real -> "
          "-Infinity; other numeric directions are normalised to z/Abs[z] (DirectedInfinity[1 + I] -> "
          "DirectedInfinity[(1 + I)/Sqrt[2]]); symbolic directions stay")
def _dirinf(*a):
    if len(a) > 1:
        raise ValueError("DirectedInfinity called with 2 or more arguments; expected 0 or 1")
    if not a:
        return _sym("ComplexInfinity")
    z = a[0]
    if _is_sym(z, "Indeterminate") or _is_sym(z, "ComplexInfinity"):
        return _sym("ComplexInfinity")
    try:
        v = sym(z, "DirectedInfinity")
    except Exception:
        raise Unevaluated()
    if not v.is_number or v.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
        raise Unevaluated()
    if v == 0:
        return _sym("ComplexInfinity")
    if v.is_extended_real:
        if v.is_positive:
            return _sym("Infinity")
        if v.is_negative:
            return Expr("Times", -1, _sym("Infinity"))
    d = sp.nsimplify(v / sp.Abs(v)) if not v.has(sp.Float) else v / sp.Abs(v)
    d = sp.simplify(d) if not v.has(sp.Float) else sp.N(d)
    if d == 1:
        return _sym("Infinity")
    if d == -1:
        return Expr("Times", -1, _sym("Infinity"))
    if sp.simplify(d - v) == 0:
        raise Unevaluated()
    return Expr("DirectedInfinity", out(d))


# ───────────────────────────── ML <-> SymPy with relations, logic and Piecewise ─────────────────────────────

_REL = {"Equal": sp.Eq, "Unequal": sp.Ne, "Less": sp.Lt, "Greater": sp.Gt, "LessEqual": sp.Le,
        "GreaterEqual": sp.Ge}
_LOGIC = {"And": sp.And, "Or": sp.Or, "Not": sp.Not, "Xor": sp.Xor, "Implies": sp.Implies,
          "Nand": sp.Nand, "Nor": sp.Nor, "Equivalent": sp.Equivalent}


def tosp(e, fname):
    """MikoshiLang -> SymPy, also converting relations (chains become And), logical connectives, True/False and
    Piecewise (anywhere in the expression)."""
    t = _tf(e)
    if t is not None:
        return sp.true if t else sp.false
    if isinstance(e, Expr) and isinstance(e.head, str):
        if e.head in _REL and len(e.args) >= 2:
            a = [tosp(x, fname) for x in e.args]
            if e.head == "Unequal" and len(a) > 2:
                return sp.And(*(sp.Ne(u, v) for u, v in itertools.combinations(a, 2)))
            rels = [_REL[e.head](u, v) for u, v in zip(a, a[1:])]
            return rels[0] if len(rels) == 1 else sp.And(*rels)
        if e.head in _LOGIC:
            return _LOGIC[e.head](*(tosp(x, fname) for x in e.args))
        if e.head == "Piecewise":
            cases, d = _pw_parts(e, fname)
            pairs = [(tosp(v, fname), tosp(c, fname)) for v, c in cases]
            return sp.Piecewise(*pairs, (tosp(d, fname), True))
        if e.head == "Inequality":
            raise NotImplementedError(f"{fname}: Inequality[...] chains are not supported")
        if _contains(e, lambda s: isinstance(s, Expr) and s.head in ("Piecewise",) + tuple(_REL) + tuple(_LOGIC)):
            subs, k = {}, [0]

            def rep(s):
                if isinstance(s, Expr) and isinstance(s.head, str) and \
                        (s.head == "Piecewise" or s.head in _REL or s.head in _LOGIC):
                    k[0] += 1
                    nm = f"_w05p{k[0]}"
                    subs[sp.Symbol(nm)] = tosp(s, fname)
                    return Symbol(nm)
                if isinstance(s, Expr):
                    return Expr(s.head, *(rep(a) for a in s.args))
                return s
            v = sym(rep(e), fname)
            return v.xreplace(subs)
    return sym(e, fname)


def _contains(e, pred):
    if pred(e):
        return True
    return isinstance(e, Expr) and any(_contains(a, pred) for a in e.args)


def fromsp(v):
    """SymPy -> MikoshiLang (Piecewise, relations and Booleans included)."""
    if v is sp.true or v is True:
        return True
    if v is sp.false or v is False:
        return False
    if isinstance(v, sp.Piecewise):
        cases, default = [], None
        for val, c in v.args:
            if c is sp.true:
                default = fromsp(val)
                break
            if c is sp.false:
                continue
            cases.append(L(fromsp(val), fromsp(c)))
        if not cases:
            return default if default is not None else 0
        return Expr("Piecewise", L(*cases), *([default] if default is not None and default != 0 else []))
    if isinstance(v, (sp.And, sp.Or, sp.Not, sp.Xor)):
        return Expr(type(v).__name__, *(fromsp(a) for a in v.args))
    if isinstance(v, sp.core.relational.Relational):
        h = {sp.Equality: "Equal", sp.Unequality: "Unequal", sp.StrictLessThan: "Less",
             sp.StrictGreaterThan: "Greater", sp.LessThan: "LessEqual", sp.GreaterThan: "GreaterEqual"}[type(v)]
        return Expr(h, fromsp(v.lhs), fromsp(v.rhs))
    if isinstance(v, sp.Basic) and v.has(sp.Piecewise):
        subs, k = {}, [0]

        def rep(s):
            if isinstance(s, sp.Piecewise):
                k[0] += 1
                d = sp.Dummy(f"w05q{k[0]}")
                subs[str(d)] = fromsp(s)
                return d
            return None
        w = v.replace(lambda s: isinstance(s, sp.Piecewise), lambda s: rep(s))
        res = out(w)
        return _subst_names(res, {Symbol(str(d)).name if False else n: val for n, val in subs.items()})
    return out(v)


def _subst_names(e, m):
    if isinstance(e, Symbol) and e.name in m:
        return m[e.name]
    if isinstance(e, Expr):
        return Expr(e.head if not isinstance(e.head, Expr) else _subst_names(e.head, m),
                    *(_subst_names(a, m) for a in e.args))
    return e


# ───────────────────────────── AddSides, SubtractSides, MultiplySides, DivideSides, ApplySides ─────────────────

_INEQ = ("Less", "Greater", "LessEqual", "GreaterEqual")
_ALL_REL = ("Equal", "Unequal") + _INEQ
_FLIP = {"Less": "Greater", "Greater": "Less", "LessEqual": "GreaterEqual", "GreaterEqual": "LessEqual",
         "Equal": "Equal", "Unequal": "Unequal"}
_SIDES_OPTS = {"Assumptions", "GenerateConditions", "TimeConstraint"}


def _is_rel(e):
    return _is_head(e, *_ALL_REL) and len(e.args) >= 2


def _map_rel(r, f, fname):
    """Apply f(rel) through And chains, ConditionalExpression and Piecewise values."""
    t = _tf(r)
    if t is not None:
        return t
    if _is_rel(r):
        return f(r)
    if _is_head(r, "And"):
        return ev(Expr("And", *(_map_rel(a, f, fname) for a in r.args)))
    if _is_head(r, "ConditionalExpression") and len(r.args) == 2:
        return ev(Expr("ConditionalExpression", _map_rel(r.args[0], f, fname), r.args[1]))
    if _is_head(r, "Piecewise"):
        cases, d = _pw_parts(r, fname)
        return ev(Expr("Piecewise", L(*(L(_map_rel(v, f, fname), c) for v, c in cases)),
                       _map_rel(d, f, fname) if d != 0 else d))
    raise ValueError(f"{fname}: {r!r} is not an equation or inequality")


def _combine_add(r1, r2, fname, sub=False):
    """r1 (+ or -) r2 side by side; valid combinations only."""
    if not (_is_rel(r1) and _is_rel(r2) and len(r1.args) == 2 and len(r2.args) == 2):
        raise NotImplementedError(f"{fname}: combining chained relations is not supported")
    h1, h2 = r1.head, r2.head
    a, b = r1.args
    c, d = r2.args
    if sub:   # r1 - r2 == r1 + (-r2), and -c op -d is d op c
        h2 = _FLIP[h2] if h2 in _INEQ else h2
        c, d = Expr("Times", -1, c), Expr("Times", -1, d)
        if h2 in _INEQ:
            pass
    if h2 == "Equal":
        h = h1
    elif h1 == "Equal":
        h = h2
    elif h1 in _INEQ and h2 in _INEQ:
        up1 = h1 in ("Greater", "GreaterEqual")
        up2 = h2 in ("Greater", "GreaterEqual")
        if up1 != up2:
            c, d = d, c
            h2 = _FLIP[h2]
        strict = h1 in ("Less", "Greater") or h2 in ("Less", "Greater")
        h = ("Greater" if strict else "GreaterEqual") if up1 else ("Less" if strict else "LessEqual")
    else:
        raise ValueError(f"{fname}: cannot combine {h1} and {h2} side by side")
    return ev(Expr(h, Expr("Plus", a, c), Expr("Plus", b, d)))


def _sides_add(fname, sub):
    def impl(*args):
        args, opts = _opts(args, _SIDES_OPTS, fname)
        if len(args) == 1:
            r = args[0]
            if not (_is_rel(r) and len(r.args) == 2):
                if _tf(r) is not None:
                    return r
                raise ValueError(f"{fname}[rel] needs a two-sided equation or inequality")
            x = r.args[1] if sub else None
            if not sub:
                raise ValueError("AddSides expects AddSides[rel, x] or AddSides[rel1, rel2]")
            args = [r, x]
        if len(args) != 2:
            raise ValueError(f"{fname} expects {fname}[rel, x] or {fname}[rel1, rel2]")
        r, x = args
        if _is_rel(x):
            if _tf(r) is not None:
                return r
            return _combine_add(r, x, fname, sub)
        if _tf(x) is not None:
            raise ValueError(f"{fname}: cannot add a Boolean to a relation")
        op = (lambda s: Expr("Plus", s, Expr("Times", -1, x))) if sub else (lambda s: Expr("Plus", s, x))
        return _map_rel(r, lambda q: ev(Expr(q.head, *(op(s) for s in q.args))), fname)
    return impl


fn("AddSides", area=AREA, status="added",
   notes="AddSides[rel, x] adds x to every side of an equation, inequation, inequality or chain (also inside "
         "And, ConditionalExpression and Piecewise); AddSides[rel1, rel2] adds corresponding sides (equation + "
         "relation keeps the relation; two inequalities must point the same way, else the second is reversed); "
         "True/False stay")(_sides_add("AddSides", False))
fn("SubtractSides", area=AREA, status="added",
   notes="SubtractSides[rel, x] subtracts x from every side; SubtractSides[rel] subtracts the right-hand side "
         "(a == b -> a - b == 0); SubtractSides[rel1, rel2] subtracts corresponding sides (an inequality is "
         "reversed when subtracted)")(_sides_add("SubtractSides", True))


def _sign(x, A):
    """+1, -1, 0 or None (unknown) for the real sign of SymPy x under the assumptions A."""
    if x.is_number:
        if x.is_extended_real is False or x.has(sp.I) and not x.is_extended_real:
            return "complex"
        if x.is_zero:
            return 0
        if x.is_extended_positive:
            return 1
        if x.is_extended_negative:
            return -1
        return None
    for rel, s in ((sp.Gt(x, 0), 1), (sp.Lt(x, 0), -1)):
        if A is not None and A.decide(A.apply(rel)) is sp.true:
            return s
    if A is not None and A.decide(A.apply(sp.Eq(x, 0))) is sp.true:
        return 0
    return None


def _nonzero(x, A):
    if x.is_number:
        return x.is_zero is False
    return A is not None and A.decide(A.apply(sp.Ne(x, 0))) is sp.true


def _sides_mul(fname, div):
    from .assume import Assumed, combine, current

    def scale(q, m, neg):
        h = q.head
        sides = [ev(Expr("Times", s, Expr("Power", m, -1)) if div else Expr("Times", m, s)) for s in q.args]
        if neg and h in _INEQ:
            return ev(Expr(h, *reversed(sides)))
        return ev(Expr(h, *sides))

    def impl(*args):
        args, opts = _opts(args, _SIDES_OPTS, fname)
        gc = opts.get("GenerateConditions", _sym("All"))
        gcn = gc.name if isinstance(gc, Symbol) else ("True" if gc is True else "False" if gc is False else None)
        if gcn not in ("All", "Automatic", "True", "False", "None"):
            raise ValueError(f"{fname}: GenerateConditions must be All, Automatic, True, False or None")
        A = Assumed(opts["Assumptions"] if "Assumptions" in opts else combine(current()), fname)
        if len(args) == 1:
            r = args[0]
            if _tf(r) is not None:
                return r
            if not div or not (_is_rel(r) and len(r.args) == 2):
                raise ValueError(f"{fname}[rel] needs a two-sided relation (DivideSides only)")
            args = [r, r.args[1]]
        if len(args) != 2:
            raise ValueError(f"{fname} expects {fname}[rel, x] or {fname}[rel1, rel2]")
        r, x = args
        if _tf(r) is not None:
            return r
        pair = None
        if _is_rel(x):
            if not (x.head == "Equal" and len(x.args) == 2):
                raise NotImplementedError(f"{fname}: the second relation must be an equation c == d")
            pair = x.args
            if not (_is_rel(r) and len(r.args) == 2):
                raise NotImplementedError(f"{fname}: combining a chained relation with an equation is not supported")
            msp = [tosp(pair[0], fname), tosp(pair[1], fname)]
            sg = next((s for s in (_sign(m, A) for m in msp) if s is not None), None)
            nz = any(_nonzero(m, A) for m in msp)
            m_expr = pair[0]
        else:
            m_expr = x
            msp = [tosp(x, fname)]
            sg = _sign(msp[0], A)
            nz = _nonzero(msp[0], A)
        if sg == "complex":
            if any(_is_head(q, *_INEQ) for q in _rels_in(r)):
                raise ValueError(f"{fname}: cannot multiply an inequality by a non-real number")
            sg = None
            nz = True

        def apply(q, neg):
            if pair is None:
                return scale(q, m_expr, neg)
            a, b = q.args
            c, d = pair
            if div:
                sides = [ev(Expr("Times", a, Expr("Power", c, -1))), ev(Expr("Times", b, Expr("Power", d, -1)))]
            else:
                sides = [ev(Expr("Times", a, c)), ev(Expr("Times", b, d))]
            if neg and q.head in _INEQ:
                sides = sides[::-1]
            return ev(Expr(q.head, *sides))

        rels = _rels_in(r)
        has_ineq = any(_is_head(q, *_INEQ) for q in rels)
        if pair is not None and not div and all(q.head == "Equal" for q in rels):
            return _map_rel(r, lambda q: apply(q, False), fname)    # Wolfram: a == b times c == d -> a c == b d
        if sg == 0:
            raise ValueError(f"{fname}: cannot {'divide' if div else 'multiply'} by zero")
        if sg in (1, -1) and (has_ineq or nz or True) and (sg is not None):
            return _map_rel(r, lambda q: apply(q, sg == -1), fname)
        if not has_ineq and nz:
            return _map_rel(r, lambda q: apply(q, False), fname)
        m0 = m_expr
        pos = _map_rel(r, lambda q: apply(q, False), fname)
        if has_ineq:
            neg = _map_rel(r, lambda q: apply(q, True), fname)
            cpos, cneg = Expr("Greater", m0, 0), Expr("Less", m0, 0)
            if gcn == "All":
                return ev(Expr("Piecewise", L(L(pos, cpos), L(neg, cneg)), r))
            if gcn in ("True", "Automatic"):
                return ev(Expr("ConditionalExpression", pos, cpos))
            if gcn == "False":
                return pos
            raise Unevaluated()
        cnz = Expr("Unequal", m0, 0)
        if gcn == "All":
            return ev(Expr("Piecewise", L(L(pos, cnz)), r))
        if gcn == "True":
            return ev(Expr("ConditionalExpression", pos, cnz))
        if gcn in ("Automatic", "False"):
            return pos
        raise Unevaluated()
    return impl


def _rels_in(r):
    if _is_rel(r):
        return [r]
    if _is_head(r, "And", "ConditionalExpression"):
        return [q for a in (r.args if r.head == "And" else r.args[:1]) for q in _rels_in(a)]
    if _is_head(r, "Piecewise"):
        cases, d = _pw_parts(r)
        return [q for v, _ in cases for q in _rels_in(v)] + _rels_in(d)
    return []


fn("MultiplySides", area=AREA, status="added",
   notes="MultiplySides[rel, x] / MultiplySides[rel1, c == d]: multiplies every side; for an inequality the "
         "direction follows the sign of x (a negative x swaps the sides, a == b > -1/5 > c times -5 -> chain of "
         "swapped relations); with unknown sign GenerateConditions -> All (default) gives Piecewise[{{rel x, x > "
         "0}, {reversed, x < 0}}, rel], True/Automatic give ConditionalExpression[..., x > 0], False the "
         "unconditional form and None stays unevaluated; for equations the condition is x != 0 (Automatic drops "
         "it); Assumptions and $Assumptions decide the sign")(_sides_mul("MultiplySides", False))
fn("DivideSides", area=AREA, status="added",
   notes="DivideSides[rel, x] / DivideSides[rel] (divides by the right-hand side) / DivideSides[rel1, c == d]: "
         "as MultiplySides with division (x != 0 and sign conditions, GenerateConditions and Assumptions "
         "options)")(_sides_mul("DivideSides", True))


@fn("ApplySides", area=AREA, status="added",
    notes="ApplySides[f, rel]: applies f to every side of an equation, inequation, inequality or chain (also "
          "inside And, ConditionalExpression and Piecewise), purely structurally as Wolfram (no check that an "
          "inequality stays valid); True/False stay")
def _applysides(f, r):
    return _map_rel(r, lambda q: ev(Expr(q.head, *(K.call(f, s) for s in q.args))), "ApplySides")


# ───────────────────────────── structural polynomial / rational tests ─────────────────────────────

def _isnum(e):
    return K.is_number(e) or (isinstance(e, Expr) and e.head == "Rational")


def _free_of(e, vs):
    if any(K.same(e, v) for v in vs):
        return False
    if isinstance(e, Expr):
        if isinstance(e.head, Expr) and not _free_of(e.head, vs):
            return False
        return all(_free_of(a, vs) for a in e.args)
    return True


def _var_symbols(vs):
    out_ = set()

    def walk(e):
        if isinstance(e, Symbol):
            out_.add(e.name)
        elif isinstance(e, Expr):
            for a in e.args:
                walk(a)
    for v in vs:
        walk(v)
    return out_


def _coef_ok(e, vs):
    if not _free_of(e, vs):
        return False
    names = _var_symbols(vs)
    return not _contains(e, lambda s: isinstance(s, Symbol) and s.name in names)


def _poly_struct(e, vs, rational, test):
    if _coef_ok(e, vs):
        return test is None or K.same(ev(K.call(test, e)), True) or ev(K.call(test, e)) is True
    if any(K.same(e, v) for v in vs):
        return True
    if _is_head(e, "Plus", "Times"):
        if _is_head(e, "Times") and test is not None:
            coef = [a for a in e.args if _coef_ok(a, vs)]
            rest = [a for a in e.args if not _coef_ok(a, vs)]
            if coef:
                c = coef[0] if len(coef) == 1 else Expr("Times", *coef)
                if ev(K.call(test, c)) is not True:
                    return False
            return all(_poly_struct(a, vs, rational, None if _is_head(a, "Plus") is False else test) if False
                       else _poly_struct(a, vs, rational, test) for a in rest)
        return all(_poly_struct(a, vs, rational, test) for a in e.args)
    if _is_head(e, "Power") and len(e.args) == 2 and K.is_int(e.args[1]):
        n = int(e.args[1])
        if n >= 0 or (rational and n != 0):
            return _poly_struct(e.args[0], vs, rational, test)
    return False


def _structq(fname, rational):
    def impl(e, v, *test):
        if len(test) > 1:
            raise ValueError(f"{fname} called with too many arguments")
        vs = _vars(v, fname)
        return _poly_struct(e, vs, rational, test[0] if test else None)
    return impl


fn("PolynomialExpressionQ", area=AREA, status="added",
   notes="PolynomialExpressionQ[e, x | {x, ...}] / [e, vars, test]: True if e is built from the variables and "
         "coefficients free of them with Plus, Times and non-negative integer Power (purely structural, as "
         "Wolfram: x Sin[x]^2 + x Cos[x]^2 is not, x - 1/x is not; variables may be any expressions such as "
         "x[1] or Sin[x]); test is applied to the coefficients")(_structq("PolynomialExpressionQ", False))
fn("RationalExpressionQ", area=AREA, status="added",
   notes="RationalExpressionQ[e, x | {x, ...}] / [e, vars, test]: True if e is built from the variables and "
         "coefficients free of them with Plus, Times and integer Power (structural, as Wolfram: Sqrt[x] is not)"
         )(_structq("RationalExpressionQ", True))


# ───────────────────────────── polynomial helpers ─────────────────────────────

def _var1(x, fname):
    if not isinstance(x, Symbol):
        raise ValueError(f"{fname}: {x!r} is not a valid variable")
    return sp.Symbol(x.name)


def _modulus(opts, fname):
    m = opts.get("Modulus", 0)
    if m == 0:
        return None
    if not (K.is_int(m) and int(m) > 1 and sp.isprime(int(m))):
        raise ValueError(f"{fname}: Modulus must be a prime (or 0)")
    return int(m)


def _poly(e, x, fname, modulus=None, params_field=True):
    """SymPy Poly in x over QQ, QQ(params) or GF(p)."""
    v = sp.together(_S(e, fname)) if not isinstance(e, sp.Basic) else sp.together(e)
    n, d = sp.fraction(v)
    if d.has(x):
        raise ValueError(f"{fname}: {e!r} is not a polynomial in {x}")
    v = sp.expand(n) / d
    if not sp.expand(n).is_polynomial(x):
        raise ValueError(f"{fname}: {e!r} is not a polynomial in {x}")
    if modulus:
        return sp.Poly(sp.expand(v), x, modulus=modulus)
    params = sorted(v.free_symbols - {x}, key=str)
    if any(isinstance(a, sp.Float) for a in sp.preorder_traversal(v)):
        dom = "RR" if not params else None
    else:
        dom = sp.QQ.frac_field(*params) if params and params_field else (sp.QQ if not params else None)
    try:
        return sp.Poly(sp.expand(v), x, domain=dom) if dom is not None else sp.Poly(sp.expand(v), x)
    except sp.PolynomialError as exc:
        raise ValueError(f"{fname}: {e!r} is not a polynomial in {x} ({exc})")


def _pexpr(p):
    return sp.expand(p.as_expr()) if isinstance(p, sp.Poly) else p


# ───────────────────────────── ApartSquareFree / PartialFractionElements ─────────────────────────────

def _apart_sqf(f, x):
    """Polynomial part + Σ_i Σ_{j<=i} a_ij / s_i^j with s_i the square-free factors of the denominator
    (deg a_ij < deg s_i); exact over QQ(params)."""
    f = sp.together(f)
    n, d = sp.fraction(f)
    params = sorted((n.free_symbols | d.free_symbols) - {x}, key=str)
    dom = sp.QQ.frac_field(*params) if params else sp.QQ
    N = sp.Poly(sp.expand(n), x, domain=dom)
    D = sp.Poly(sp.expand(d), x, domain=dom)
    g = N.gcd(D)
    N, D = N.quo(g), D.quo(g)
    q, r = N.div(D)
    lc, facs = D.sqf_list()
    terms = [q.as_expr()]
    if r.is_zero:
        return sp.Add(*terms), q, []
    # split r/D over the coprime powers s_i^i, then s_i-adic expansion
    parts = [(s, k) for s, k in facs]
    rem = r * D.LC() ** 0
    elements = []
    Dn = D.monic() if False else D
    # r/D = Σ c_i / s_i^k_i with c_i = r * (D/s_i^k_i)^{-1} mod s_i^k_i  (D = lc * Π s_i^k_i)
    for s, k in parts:
        P = s ** k
        other = D.quo(P)
        inv = other.invert(P)
        c = (r * inv).rem(P)
        # s-adic digits: c = Σ_{j=0}^{k-1} a_j s^j
        digits = []
        cc = c
        for _ in range(k):
            qq, rr = cc.div(s)
            digits.append(rr)
            cc = qq
        for j, a in enumerate(digits):       # a / s^(k - j)
            if not a.is_zero:
                elements.append((a, s, k - j))
    total = sp.Add(*terms, *(a.as_expr() / s.as_expr() ** e for a, s, e in elements))
    return total, q, elements


def _pick_var(e, x, fname):
    if x is not None:
        return _var1(x, fname)
    v = _S(e, fname)
    fs = sorted(v.free_symbols, key=str)
    if len(fs) != 1:
        raise ValueError(f"{fname}: give the variable explicitly ({fname}[expr, x])")
    return fs[0]


@fn("ApartSquareFree", area=AREA, status="added",
    notes="ApartSquareFree[expr, x] / ApartSquareFree[expr] (one variable): polynomial part plus fractions over "
          "the powers of the square-free factors of the denominator (ApartSquareFree[(x^2 - 2)/(x^4 - 2 x^2 + 1)] "
          "-> -1/(-1 + x^2)^2 + 1/(-1 + x^2)), exact over the rationals and rational functions of other symbols; "
          "threads over lists, equations and inequalities; the result is checked to equal the input; the Modulus "
          "and Trig options raise")
def _apartsqf(e, *rest):
    rest, opts = _opts(rest, {"Modulus", "Trig"}, "ApartSquareFree")
    if opts.get("Modulus", 0) not in (0,) or (opts.get("Trig") is not None and not is_false(opts.get("Trig"))):
        raise NotImplementedError("ApartSquareFree: the Modulus and Trig options are not supported")
    if len(rest) > 1:
        raise ValueError("ApartSquareFree called with too many arguments")
    if is_list(e) or _is_head(e, *_ALL_REL):
        return ev(Expr(e.head, *(_apartsqf(a, *rest) for a in e.args)))
    x = _pick_var(e, rest[0] if rest else None, "ApartSquareFree")
    f = _S(e, "ApartSquareFree")
    n, d = sp.fraction(sp.together(f))
    if not (sp.expand(n).is_polynomial(x) and sp.expand(d).is_polynomial(x)):
        raise ValueError(f"ApartSquareFree: {e!r} is not a rational function of {x}")
    res, _, _ = _apart_sqf(f, x)
    if sp.simplify(sp.together(res - f)) != 0:
        raise ArithmeticError("ApartSquareFree: decomposition failed verification")
    return out(res)


def _pfe_full(f, x):
    """Complete partial fractions over the algebraic closure: (polynomial part, [(pole, k, coefficient)])."""
    f = sp.together(f)
    n, d = sp.fraction(f)
    N = sp.Poly(sp.expand(n), x)
    D = sp.Poly(sp.expand(d), x)
    g = N.gcd(D)
    N, D = N.quo(g), D.quo(g)
    q, r = N.div(D)
    roots = sp.roots(D, x)
    if sum(roots.values()) != D.degree():
        raise NotImplementedError("PartialFractionElements: the denominator's roots cannot be found in radicals")
    elems = []
    fr = r.as_expr() / D.as_expr()
    for p, m in roots.items():
        g_ = sp.cancel(fr * (x - p) ** m)
        coeffs = []
        for k in range(m, 0, -1):        # coefficient of 1/(x - p)^k
            j = m - k
            c = sp.diff(g_, x, j).subs(x, p) / sp.factorial(j) if j else g_.subs(x, p)
            if c.has(sp.nan, sp.zoo):
                c = sp.limit(sp.diff(g_, x, j) / sp.factorial(j), x, p)
            coeffs.append(sp.radsimp(sp.simplify(c)))
        elems.append((p, m, coeffs))
    return q, elems


@fn("PartialFractionElements", area=AREA, status="added",
    notes="PartialFractionElements[f, x] -> <|\"Decomposition\", \"PolynomialPart\", \"FractionalPart\", "
          "\"PolynomialCoefficients\", \"FractionalCoefficients\" ({{pole, multiplicity, {c_m, ..., c_1}}, ...} with "
          "c_k the coefficient of 1/(x - pole)^k), \"Poles\", \"PoleMultiplicities\"|>, the complete decomposition "
          "over the algebraic numbers (poles in radicals); PartialFractionElements[f, x, prop] gives one property; "
          "poles are sorted as Wolfram (real ascending, then complex); the decomposition is checked to equal f")
def _pfe(e, x, *prop):
    xs = _var1(x, "PartialFractionElements")
    f = _S(e, "PartialFractionElements")
    n, d = sp.fraction(sp.together(f))
    if not (sp.expand(n).is_polynomial(xs) and sp.expand(d).is_polynomial(xs)):
        raise ValueError(f"PartialFractionElements: {e!r} is not a rational function of {x}")
    q, elems = _pfe_full(f, xs)
    from .wl_algebra import _wl_order_key
    elems.sort(key=lambda t: _wl_order_key(t[0]))
    frac = sp.Add(*(c / (xs - p) ** (m - i) for p, m, cs in elems for i, c in enumerate(cs)))
    if sp.simplify(sp.together(q.as_expr() + frac - f)) != 0:
        raise ArithmeticError("PartialFractionElements: decomposition failed verification")
    poly_coeffs = list(reversed(q.all_coeffs())) if not q.is_zero else []
    props = {
        "Decomposition": out(q.as_expr()) if not elems else ev(Expr("Plus", out(q.as_expr()), out(frac))),
        "PolynomialPart": out(q.as_expr()),
        "FractionalPart": out(frac),
        "PolynomialCoefficients": L(*(out(c) for c in poly_coeffs)),
        "FractionalCoefficients": L(*(L(out(p), m, L(*(out(c) for c in cs))) for p, m, cs in elems)),
        "Poles": L(*(out(p) for p, _, _ in elems)),
        "PoleMultiplicities": L(*(m for _, m, _ in elems)),
    }
    if prop:
        if len(prop) > 1 or not isinstance(prop[0], str) or prop[0] not in props:
            raise ValueError(f"PartialFractionElements: unknown property {prop[0]!r}; known: {sorted(props)}")
        return props[prop[0]]
    return Expr("Association", *(rule(k, v) for k, v in props.items()))


# ───────────────────────────── GCD, reduction, subresultants ─────────────────────────────

@fn("PolynomialExtendedGCD", area=AREA, status="added",
    notes="PolynomialExtendedGCD[p, q, x] -> {g, {r, s}} with r p + s q == g, g the monic GCD and deg r < deg q - "
          "deg g, deg s < deg p - deg g (unique cofactors; coefficients may be rational functions of other "
          "symbols, as Wolfram); Modulus -> prime works over GF(p) with representatives 0..p-1; checked")
def _pegcd(p, q, x, *rest):
    rest, opts = _opts(rest, {"Modulus"}, "PolynomialExtendedGCD")
    if rest:
        raise ValueError("PolynomialExtendedGCD called with too many arguments")
    xs = _var1(x, "PolynomialExtendedGCD")
    m = _modulus(opts, "PolynomialExtendedGCD")
    P = _poly(p, xs, "PolynomialExtendedGCD", m)
    Q = _poly(q, xs, "PolynomialExtendedGCD", m)
    if P.is_zero and Q.is_zero:
        return L(0, L(0, 0))
    if Q.is_zero:
        lc = P.LC()
        return L(out(_pmod_expr(P.monic(), m)), L(out(_pmod_expr(sp.Poly(1 / lc, xs, domain=P.domain), m)), 0))
    if P.is_zero:
        lc = Q.LC()
        return L(out(_pmod_expr(Q.monic(), m)), L(0, out(_pmod_expr(sp.Poly(1 / lc, xs, domain=Q.domain), m))))
    s, t, h = P.gcdex(Q)
    if not (s * P + t * Q - h).is_zero:
        raise ArithmeticError("PolynomialExtendedGCD: result failed verification")
    return L(out(_pmod_expr(h, m)), L(out(_pmod_expr(s, m)), out(_pmod_expr(t, m))))


def _pmod_expr(P, m):
    if m is None:
        return sp.factor_terms(P.as_expr()) if False else P.as_expr()
    return sp.Add(*(sp.Integer(int(c) % m) * sp.Mul(*(g ** e for g, e in zip(P.gens, mon)))
                    for mon, c in P.terms()))


_ORDERS = {"Lexicographic": "lex", "DegreeLexicographic": "grlex", "DegreeReverseLexicographic": "grevlex"}


@fn("PolynomialReduction", area=AREA, status="added",
    notes="PolynomialReduction[p, {p1, ...}, {x1, ...}]: the remainder of p on division by the pi (fully reduced "
          "with respect to their leading terms; unique when the pi are a Gröbner basis), coefficients in the field "
          "of rational functions of the other symbols; MonomialOrder -> Lexicographic | DegreeLexicographic | "
          "DegreeReverseLexicographic, Modulus -> prime; the identity p == Σ ai pi + r is checked")
def _preduction(p, ps, vs, *rest):
    rest, opts = _opts(rest, {"MonomialOrder", "Modulus", "CoefficientDomain"}, "PolynomialReduction")
    if rest:
        raise ValueError("PolynomialReduction called with too many arguments")
    q, r = _reduce_poly(p, ps, vs, opts, "PolynomialReduction")
    return out(r)


def _reduce_poly(p, ps, vs, opts, fname):
    X = [_var1(v, fname) for v in (vs.args if is_list(vs) else [vs])]
    polys = list(ps.args) if is_list(ps) else [ps]
    order = opts.get("MonomialOrder", _sym("Lexicographic"))
    if not (isinstance(order, Symbol) and order.name in _ORDERS):
        raise NotImplementedError(f"{fname}: MonomialOrder {order!r} is not supported")
    cd = opts.get("CoefficientDomain")
    if cd is not None and not (isinstance(cd, Symbol) and cd.name in ("RationalFunctions", "Rationals",
                                                                        "Automatic")):
        raise NotImplementedError(f"{fname}: CoefficientDomain {cd!r} is not supported")
    m = _modulus(opts, fname)
    f = sp.expand(_S(p, fname))
    G = [sp.expand(_S(g, fname)) for g in polys]
    allv = set(f.free_symbols).union(*(g.free_symbols for g in G))
    params = sorted(allv - set(X), key=str)
    if m:
        dom = sp.GF(m)
        if params:
            raise NotImplementedError(f"{fname}: symbolic coefficients with Modulus are not supported")
    else:
        dom = sp.QQ.frac_field(*params) if params else sp.QQ
    for g in [f] + G:
        if not g.is_polynomial(*X):
            raise ValueError(f"{fname}: {out(g)!r} is not a polynomial in {[str(v) for v in X]}")
    G = [g for g in G if g != 0]
    if not G:
        return [], f
    Q, r = sp.reduced(f, G, *X, order=_ORDERS[order.name], domain=dom)
    chk = sp.expand(sp.together(f - sum(a * g for a, g in zip(Q, G)) - r))
    if m:
        chk = sp.Poly(chk, *X, modulus=m) if chk != 0 else sp.Integer(0)
        chk = 0 if chk == 0 or chk.is_zero else chk
    if chk != 0 and sp.simplify(chk) != 0:
        raise ArithmeticError(f"{fname}: reduction failed verification")
    if m:
        r = _pmod_expr(sp.Poly(r, *X, modulus=m), m) if r != 0 else sp.Integer(0)
    return Q, sp.expand(r) if not params else sp.collect(sp.expand(r), X)


def _sres_setup(p, q, x, fname):
    xs = _var1(x, fname)
    P = sp.Poly(sp.expand(_S(p, fname)), xs)
    Q = sp.Poly(sp.expand(_S(q, fname)), xs)
    if P.is_zero or Q.is_zero:
        raise ValueError(f"{fname}: the polynomials must be nonzero")
    return xs, P, Q


def _subres_matrix_poly(P, Q, j, xs):
    """j-th subresultant polynomial S_j (0 <= j < min(m, n)) from the Sylvester submatrices."""
    m, n = P.degree(), Q.degree()
    a = P.all_coeffs()
    b = Q.all_coeffs()
    cols = m + n - j
    rows = []
    for i in range(n - j):
        rows.append([0] * i + a + [0] * (cols - m - 1 - i))
    for i in range(m - j):
        rows.append([0] * i + b + [0] * (cols - n - 1 - i))
    M = sp.Matrix(rows)
    k = m + n - 2 * j
    res = sp.Integer(0)
    for l in range(j + 1):          # coefficient of x^l: replace last column by column of x^l
        cols_idx = list(range(k - 1)) + [cols - 1 - l]
        res += M.extract(list(range(k)), cols_idx).det(method="bareiss") * xs ** l
    return sp.expand(res)


def _subresultants_all(P, Q, xs):
    """[S_0, ..., S_n] for deg P = m >= deg Q = n (S_n = lc(Q)^(m - n - 1) Q)."""
    m, n = P.degree(), Q.degree()
    S = [_subres_matrix_poly(P, Q, j, xs) for j in range(n)]
    S.append(sp.expand(Q.LC() ** (m - n - 1) * Q.as_expr()))
    return S


def _order_pq(P, Q, fname):
    if P.degree() < Q.degree():
        raise NotImplementedError(f"{fname}: the first polynomial must have degree at least that of the second")
    return P, Q


@fn("Subresultants", area=AREA, status="added",
    notes="Subresultants[p, q, x]: the principal subresultant coefficients {psc_0, ..., psc_n} (n = deg q <= deg "
          "p): psc_j is the leading coefficient of the j-th subresultant (determinant of the Sylvester submatrix), "
          "psc_0 = Resultant and psc_n = lc(q)^(deg p - deg q); the first k vanish exactly when p and q have k "
          "common roots; deg p < deg q and Modulus raise")
def _subresultants(p, q, x, *rest):
    if rest:
        raise NotImplementedError("Subresultants: options (Modulus) are not supported")
    xs, P, Q = _sres_setup(p, q, x, "Subresultants")
    P, Q = _order_pq(P, Q, "Subresultants")
    S = _subresultants_all(P, Q, xs)
    n = Q.degree()
    return L(*(out(sp.Poly(s, xs).coeff_monomial(xs ** j) if s != 0 else sp.Integer(0)) if j < n
               else out(Q.LC() ** (P.degree() - n)) for j, s in enumerate(S)))


@fn("SubresultantPolynomials", area=AREA, status="added",
    notes="SubresultantPolynomials[p, q, x]: the subresultant polynomials {S_0, ..., S_n} (n = deg q <= deg p) "
          "from the Sylvester submatrices, S_0 = Resultant[p, q, x] and S_n = lc(q)^(deg p - deg q - 1) q; deg p < "
          "deg q and Modulus raise")
def _subrespolys(p, q, x, *rest):
    if rest:
        raise NotImplementedError("SubresultantPolynomials: options (Modulus) are not supported")
    xs, P, Q = _sres_setup(p, q, x, "SubresultantPolynomials")
    P, Q = _order_pq(P, Q, "SubresultantPolynomials")
    return L(*(out(s) for s in _subresultants_all(P, Q, xs)))


@fn("SubresultantPolynomialRemainders", area=AREA, status="added",
    notes="SubresultantPolynomialRemainders[p, q, x]: the subresultant polynomial remainder sequence {p, q, ...} "
          "(each term a multiple of the corresponding Euclidean remainder; the last is a multiple of the GCD), "
          "computed with SymPy's determinant-based subresultant PRS; Modulus raises")
def _subresprs(p, q, x, *rest):
    if rest:
        raise NotImplementedError("SubresultantPolynomialRemainders: options (Modulus) are not supported")
    xs, P, Q = _sres_setup(p, q, x, "SubresultantPolynomialRemainders")
    seq = sp.subresultants(P.as_expr(), Q.as_expr(), xs)
    # check each element is a constant multiple of the Euclidean remainder of the previous two
    for i in range(2, len(seq)):
        r = sp.rem(seq[i - 2], seq[i - 1], xs)
        if sp.simplify(sp.cancel(seq[i] / r)).has(xs):
            raise ArithmeticError("SubresultantPolynomialRemainders: sequence failed verification")
    return L(*(out(sp.expand(s)) for s in seq))


@fn("FromCoefficientRules", area=AREA, status="added",
    notes="FromCoefficientRules[{{e1, e2, ...} -> c, ...}, {x1, x2, ...}] -> Σ c x1^e1 x2^e2 ... (inverse of "
          "CoefficientRules); a single variable may be given without a list")
def _fromcr(rules, vs):
    V = list(vs.args) if is_list(vs) else [vs]
    rs = list(rules.args) if is_list(rules) else [rules]
    terms = []
    for r in rs:
        if not _is_head(r, "Rule", "RuleDelayed"):
            raise ValueError("FromCoefficientRules: expected rules {e1, ...} -> c")
        ex = r.args[0]
        exps = list(ex.args) if is_list(ex) else [ex]
        if len(exps) != len(V) or not all(K.is_int(e) and int(e) >= 0 for e in exps):
            raise ValueError("FromCoefficientRules: each exponent vector must have one non-negative integer per "
                             "variable")
        terms.append(Expr("Times", r.args[1], *(Expr("Power", v, e) for v, e in zip(V, exps))))
    return ev(Expr("Plus", *terms)) if terms else 0


@fn("CoefficientArrays", area=AREA, status="added",
    notes="CoefficientArrays[polys, {x1, ...}] -> {c0, c1, c2, ...} with polys == c0 + c1.x + c2.x.x + ...: c_k has "
          "the coefficient of x_i1 ... x_ik at i1 <= ... <= ik and 0 elsewhere (Wolfram's default), or spread "
          "equally over all orderings with \"Symmetric\" -> True; equations are taken as lhs - rhs; a single "
          "polynomial gives a scalar c0. MikoshiLang has no sparse storage, so the arrays are dense (Wolfram returns "
          "SparseArray objects; Normal gives these arrays)")
def _coefarrays(polys, vs, *rest):
    sym_ = False
    rest = list(rest)
    while rest and _is_head(rest[-1], "Rule") and rest[-1].args[0] == "Symmetric":
        sym_ = _tf(rest.pop().args[1]) is True
    if rest:
        raise ValueError("CoefficientArrays: unknown extra arguments")
    V = list(vs.args) if is_list(vs) else [vs]
    X = [_S(v, "CoefficientArrays") for v in V]
    single = not is_list(polys)
    ps = [polys] if single else list(polys.args)
    exprs = []
    for p in ps:
        if _is_head(p, "Equal") and len(p.args) == 2:
            p = Expr("Plus", p.args[0], Expr("Times", -1, p.args[1]))
        f = sp.expand(_S(p, "CoefficientArrays"))
        gens = [sp.Dummy(f"g{i}") for i in range(len(X))]
        f = f.subs(dict(zip(X, gens)), simultaneous=True)
        if not f.is_polynomial(*gens):
            raise ValueError(f"CoefficientArrays: {p!r} is not a polynomial in the variables")
        exprs.append(sp.Poly(f, *gens) if gens else sp.Poly(f))
    nv = len(X)
    deg = max((P.total_degree() for P in exprs), default=0)

    def arr(k, P):
        if k == 0:
            return P.coeff_monomial(1) if nv else P.as_expr()
        shape = [nv] * k

        def build(prefix):
            if len(prefix) == k:
                idx = tuple(prefix)
                mono = [0] * nv
                for i in idx:
                    mono[i] += 1
                c = P.coeff_monomial(tuple(mono))
                if c == 0:
                    return sp.Integer(0)
                if sym_:
                    nperm = math.factorial(k)
                    for cnt in mono:
                        nperm //= math.factorial(cnt)
                    return c / nperm
                return c if list(idx) == sorted(idx) else sp.Integer(0)
            return [build(prefix + [i]) for i in range(nv)]
        return build([]) if shape else None

    res = []
    for k in range(deg + 1):
        a = [arr(k, P) for P in exprs]
        res.append(out(a[0]) if single else out(a))
    return L(*res)


# ───────────────────────────── Padé approximants ─────────────────────────────

def _pade_core(c, m, n, t):
    """[m/n] Padé approximant of the power series Σ c_k t^k (k <= m + n) by the extended Euclidean algorithm on
    (t^(m+n+1), T); returns (P, Q) with Q(0) = 1, or None if no approximant with Q(0) != 0 exists."""
    T = sp.Poly(sum(ci * t ** k for k, ci in enumerate(c)), t)
    r0, r1 = sp.Poly(t ** (m + n + 1), t), T
    t0, t1 = sp.Poly(0, t), sp.Poly(1, t)
    if r1.is_zero:
        return sp.Integer(0), sp.Integer(1)
    while r1.degree() > m:
        q, r = r0.div(r1)
        r0, r1 = r1, r
        t0, t1 = t1, t0 - q * t1
        if r1.is_zero:
            break
    P, Q = r1, t1
    # linearised Padé condition Q T - P = O(t^(m+n+1)) (exact check)
    chk = (Q * T - P).as_expr()
    if any(sp.degree(term, t) < m + n + 1 for term in sp.Add.make_args(sp.expand(chk)) if term != 0):
        raise ArithmeticError("PadeApproximant: result failed verification")
    g = P.gcd(Q) if not P.is_zero else Q
    if not P.is_zero:
        P, Q = P.quo(g), Q.quo(g)
    else:
        return sp.Integer(0), sp.Integer(1)
    q0 = Q.eval(0)
    if q0 == 0:
        return None
    return sp.expand(P.as_expr() / q0), sp.expand(Q.as_expr() / q0)


@fn("PadeApproximant", area=AREA, status="added",
    notes="PadeApproximant[f, {x, x0, {m, n}}] / {x, x0, n} (= {n, n}): the rational function P/Q with deg P <= m, "
          "deg Q <= n, Q(x0) = 1 whose series agrees with f to the highest possible order (extended Euclidean "
          "algorithm, so degenerate cases such as PadeApproximant[Cos[x], {x, 0, 1}] -> 1 are handled); a pole or "
          "zero (x - x0)^k of f is factored out as Wolfram does (Exp[x]/x); x0 may be symbolic or Infinity "
          "(series in 1/x); fractional powers (Puiseux series) raise; the result's series is checked")
def _pade(f, spec):
    if not (is_list(spec) and len(spec.args) == 3):
        raise ValueError("PadeApproximant: expected {x, x0, {m, n}} or {x, x0, n}")
    xv, x0e, mn = spec.args
    x = _var1(xv, "PadeApproximant")
    if is_list(mn):
        if len(mn.args) != 2:
            raise ValueError("PadeApproximant: the order must be n or {m, n}")
        m, n = mn.args
    else:
        m = n = mn
    if not (K.is_int(m) and K.is_int(n) and int(m) >= 0 and int(n) >= 0):
        raise ValueError("PadeApproximant: the orders must be non-negative integers")
    m, n = int(m), int(n)
    F = _S(f, "PadeApproximant")
    inf = _is_sym(x0e, "Infinity")
    t = sp.Dummy("t")
    if inf:
        G = F.subs(x, 1 / t)
        back = 1 / x
    else:
        x0 = _S(x0e, "PadeApproximant")
        G = F.subs(x, x0 + t)
        back = x - x0
    N = m + n + 1
    try:
        ser = sp.series(G, t, 0, N + 8).removeO()
    except Exception as exc:
        raise NotImplementedError(f"PadeApproximant: cannot expand the function in a series ({exc})")
    ser = sp.expand(ser)
    exps = set()
    for term in sp.Add.make_args(ser):
        c, e = term.as_coeff_exponent(t)
        e = sp.sympify(e)
        if c.has(t):
            raise NotImplementedError("PadeApproximant: the series has logarithmic terms")
        exps.add(e)
    if any(not e.is_Integer for e in exps):
        raise NotImplementedError("PadeApproximant: fractional powers (Puiseux series) are not supported")
    if ser == 0:
        return 0
    k = int(min(exps))
    h = sp.expand(ser * t ** (-k))
    mm, nn = (m, n + k) if k < 0 else (m - k, n)
    if mm < 0 or nn < 0:
        raise ValueError("PadeApproximant: the order is too low for the pole or zero of f at the expansion point")
    coeffs = [h.coeff(t, i) for i in range(mm + nn + 1)]
    pq = _pade_core(coeffs, mm, nn, t)
    if pq is None:
        raise ArithmeticError("PadeApproximant: no Padé approximant of this order exists")
    P, Q = pq
    P = sp.expand(P * t ** k) if k > 0 else P
    Q = sp.expand(Q * t ** (-k)) if k < 0 else Q
    num = P.subs(t, back)
    den = Q.subs(t, back)
    if den == 1:
        return out(num)
    return ev(Expr("Times", out(num), Expr("Power", out(den), -1)))


# ───────────────────────────── root isolation ─────────────────────────────

def _maxdeg_check(P, fname):
    lim = _SYSC.get("$MaxRootDegree", 1024)
    st = K.state().own.get("$MaxRootDegree")
    if K.is_int(st):
        lim = int(st)
    if P.degree() > lim:
        raise ValueError(f"{fname}: the degree {P.degree()} exceeds $MaxRootDegree = {lim}")


def _rat_approx(r, prec_bits):
    """Certified rational approximation of a CRootOf r (error below 2^-prec_bits in each part)."""
    tol = sp.Rational(1, 2 ** prec_bits)
    v = r.eval_rational(dx=tol, dy=tol) if not r.is_real else r.eval_rational(tol)
    return sp.nsimplify(v) if not isinstance(v, sp.Basic) else v


def _isolate(roots_all, complexes):
    """roots_all: list of distinct exact roots (CRootOf or rationals) of a square-free polynomial.
    Returns {root: interval} with rational (real) or Gaussian-rational (complex) corners, each containing
    exactly that root and no other (real intervals open, unless the root is rational)."""
    num = [complex(sp.N(r, 40)) for r in roots_all]
    res = {}
    for i, r in enumerate(roots_all):
        others = [abs(num[i] - num[j]) for j in range(len(num)) if j != i]
        sep = min(others) if others else 4.0
        # δ: a power of 2 below sep/4 (so rectangles of half-width δ/... cannot reach another root)
        e = max(1, int(math.ceil(-math.log2(sep / 4)))) if sep < 4 else -1
        delta = sp.Rational(1, 2 ** e) if e >= 0 else sp.Integer(2)
        if r.is_Rational:
            res[i] = (r, r)
            continue
        bits = (e if e > 0 else 0) + 4
        a = _rat_approx(r, bits)              # |a - r| < 2^-bits <= δ/16
        if r.is_real or (not complexes and r.is_real is not False):
            lo, hi = _round_down(a - delta / 2, e + 2), _round_up(a + delta / 2, e + 2)
            res[i] = (lo, hi)
        else:
            ar, ai = sp.re(a), sp.im(a)
            lo = _round_down(ar - delta / 2, e + 2) + sp.I * _round_down(ai - delta / 2, e + 2)
            hi = _round_up(ar + delta / 2, e + 2) + sp.I * _round_up(ai + delta / 2, e + 2)
            res[i] = (lo, hi)
    return res


def _round_down(q, bits):
    s = 2 ** max(bits, 0)
    return sp.Rational(sp.floor(q * s), s)


def _round_up(q, bits):
    s = 2 ** max(bits, 0)
    return sp.Rational(sp.ceiling(q * s), s)


def _exact_roots(P):
    """Distinct roots of a SymPy Poly over QQ with the irreducible factor each belongs to."""
    out_ = []
    _, facs = P.factor_list()
    for fac, _m in facs:
        if fac.degree() == 1:
            out_.append((-fac.nth(0) / fac.nth(1), fac))
        else:
            for k in range(fac.degree()):
                out_.append((sp.CRootOf(fac, k), fac))
    return out_


def _sort_roots(items):
    def key(it):
        r = it[0]
        v = complex(sp.N(r, 30))
        real = r.is_real if not r.is_Rational else True
        return (0, v.real, 0.0) if real else (1, v.real, v.imag)
    return sorted(items, key=key)


def _q_poly(e, fname):
    v = sp.expand(_S(e, fname))
    fs = sorted(v.free_symbols, key=str)
    if len(fs) > 1:
        raise ValueError(f"{fname}: {e!r} must be a univariate polynomial")
    x = fs[0] if fs else sp.Symbol("x")
    if not v.is_polynomial(x) or any(isinstance(a, sp.Float) for a in sp.preorder_traversal(v)):
        raise ValueError(f"{fname}: {e!r} must be a polynomial with exact rational coefficients")
    P = sp.Poly(v, x)
    if P.domain not in (sp.ZZ, sp.QQ):
        raise NotImplementedError(f"{fname}: only rational coefficients are supported")
    return P, x


@fn("RootIntervals", area=AREA, status="added",
    notes="RootIntervals[{p1, p2, ...}] / RootIntervals[p] / RootIntervals[polys, Complexes] -> {intervals, "
          "which}: disjoint isolating intervals with rational endpoints for all distinct real roots (with "
          "Complexes, also rectangles {lower-left, upper-right} with Gaussian-rational corners for the non-real "
          "roots), and for each interval the positions of the polynomials having that root. Each real interval "
          "contains exactly one root in its interior, or is {r, r} for a rational root; isolation is certified "
          "from exact root separation; the interval endpoints differ from Wolfram's (they are not unique)")
def _rootintervals(ps, *dom):
    if len(dom) > 1:
        raise ValueError("RootIntervals called with too many arguments")
    complexes = False
    if dom:
        if not (_is_sym(dom[0], "Complexes") or _is_sym(dom[0], "Reals")):
            raise ValueError("RootIntervals: the domain must be Reals or Complexes")
        complexes = dom[0].name == "Complexes"
    polys = list(ps.args) if is_list(ps) else [ps]
    PP = [_q_poly(p, "RootIntervals") for p in polys]
    gens = {x for _, x in PP}
    if len(gens) > 1:
        raise ValueError("RootIntervals: all polynomials must be in the same variable")
    items = []
    seen = []
    for i, (P, x) in enumerate(PP):
        if P.is_zero:
            raise ValueError("RootIntervals: the zero polynomial has no isolated roots")
        _maxdeg_check(P, "RootIntervals")
        for r, fac in _exact_roots(P):
            if not complexes and not (r.is_Rational or r.is_real):
                continue
            key = None
            for j, (r2, f2, _) in enumerate(seen):
                if f2 == fac.monic() and (r2 == r):
                    key = j
                    break
            if key is None:
                seen.append((r, fac.monic(), [i + 1]))
            elif i + 1 not in seen[key][2]:
                seen[key][2].append(i + 1)
    seen = _sort_roots([(r, f, w) for r, f, w in seen])
    iv = _isolate([r for r, _, _ in seen], complexes)
    return L(L(*(L(out(iv[k][0]), out(iv[k][1])) for k in range(len(seen)))),
             L(*(L(*w) for _, _, w in seen)))


@fn("IsolatingInterval", area=AREA, status="added",
    notes="IsolatingInterval[a] / IsolatingInterval[a, dx]: for an exact algebraic number a, an interval {lo, hi} "
          "with rational endpoints (Gaussian-rational corners {lower-left, upper-right} for a non-real a) "
          "containing a and no other root of its minimal polynomial, of width below dx when given; a rational "
          "number gives {a, a}; endpoints are certified but differ from Wolfram's choice")
def _isolinterval(a, *dx):
    if len(dx) > 1:
        raise ValueError("IsolatingInterval called with too many arguments")
    v = _S(a, "IsolatingInterval")
    if not v.is_number or v.has(sp.Float):
        raise ValueError("IsolatingInterval: the argument must be an exact algebraic number")
    if v.is_Rational:
        return L(out(v), out(v))
    if dx:
        w = _S(dx[0], "IsolatingInterval")
        if not (w.is_number and w.is_positive):
            raise ValueError("IsolatingInterval: the width must be a positive number")
    t = sp.Dummy("t")
    try:
        mp = sp.minimal_polynomial(v, t)
    except Exception:
        raise ValueError("IsolatingInterval: the argument must be an algebraic number")
    P = sp.Poly(mp, t)
    _maxdeg_check(P, "IsolatingInterval")
    roots = [sp.CRootOf(P, k) for k in range(P.degree())]
    vn = complex(sp.N(v, 50))
    idx = min(range(len(roots)), key=lambda k: abs(complex(sp.N(roots[k], 50)) - vn))
    iv = _isolate(roots, True)
    lo, hi = iv[idx]
    if dx:
        w = sp.nsimplify(_S(dx[0], "IsolatingInterval"))
        if sp.re(hi - lo) >= w or sp.im(hi - lo) >= w:
            r = roots[idx]
            b = int(math.ceil(-math.log2(float(w)))) + 3          # grid 2^-b <= w/8
            q = _rat_approx(r, b + 2)                              # |q - r| < 2^-(b+2)
            if r.is_real:
                lo, hi = _round_down(q - sp.Rational(1, 2 ** b), b), _round_up(q + sp.Rational(1, 2 ** b), b)
            else:
                g = sp.Rational(1, 2 ** b)
                lo = _round_down(sp.re(q) - g, b) + sp.I * _round_down(sp.im(q) - g, b)
                hi = _round_up(sp.re(q) + g, b) + sp.I * _round_up(sp.im(q) + g, b)
    return L(out(lo), out(hi))


@fn("NSolveValues", area=AREA, status="added",
    notes="NSolveValues[eqns, vars] / NSolveValues[eqns, vars, dom]: the values of the NSolve solutions ({v1, "
          "...} for one variable, {{x1, y1, ...}, ...} for several), with NSolve's options (WorkingPrecision) and "
          "MaxRoots -> n (at most n solutions, Wolfram order)")
def _nsolvevalues(eqs, vs, *rest):
    rest, opts = _opts(rest, {"MaxRoots", "WorkingPrecision"}, "NSolveValues")
    if len(rest) > 1:
        raise ValueError("NSolveValues called with too many arguments")
    args = [eqs, vs] + rest + ([rule(_sym("WorkingPrecision"), opts["WorkingPrecision"])]
                               if "WorkingPrecision" in opts else [])
    sols = ev(Expr("NSolve", *args))
    if not is_list(sols):
        raise Unevaluated()
    one = not is_list(vs)
    V = [vs] if one else list(vs.args)
    vals = []
    for s in sols.args:
        row = []
        for v in V:
            hit = [r.args[1] for r in s.args if _is_head(r, "Rule") and K.same(r.args[0], v)]
            if not hit:
                raise NotImplementedError("NSolveValues: a solution leaves a variable free")
            row.append(hit[0])
        vals.append(row[0] if one else L(*row))
    if "MaxRoots" in opts:
        mr = opts["MaxRoots"]
        if not (_is_sym(mr, "Infinity") or (K.is_int(mr) and int(mr) >= 0)):
            raise ValueError("NSolveValues: MaxRoots must be a non-negative integer or Infinity")
        if K.is_int(mr):
            vals = vals[:int(mr)]
    return L(*vals)


record("MaxRoots", AREA, "added",
       "option of NSolveValues (and Solve/NSolve in Wolfram): the maximum number of roots returned; MaxRoots -> n "
       "keeps the first n solutions, Infinity keeps all")


# ───────────────────────────── SolveAlways ─────────────────────────────

@fn("SolveAlways", area=AREA, status="added",
    notes="SolveAlways[eqns, vars]: values of the parameters for which the equations (and inequations) hold for "
          "all values of vars: every coefficient of the numerators as polynomials in vars must vanish (an "
          "inequation p != 0 needs p to be a nonzero constant); the parameter system is solved exactly and each "
          "solution is substituted back. Non-polynomial dependence on vars and Or raise")
def _solvealways(eqs, vs, *rest):
    if rest:
        raise NotImplementedError("SolveAlways: options are not supported")
    V = [_var1(v, "SolveAlways") for v in (vs.args if is_list(vs) else [vs])]
    conds = _flat_and(eqs)
    zero_eqs, nonzero = [], []
    for c in conds:
        t = _tf(c)
        if t is True:
            continue
        if t is False:
            return L()
        if not (_is_head(c, "Equal", "Unequal") and len(c.args) == 2):
            raise NotImplementedError("SolveAlways: only equations, inequations and their conjunctions are "
                                      "supported")
        f = sp.together(_S(c.args[0], "SolveAlways") - _S(c.args[1], "SolveAlways"))
        n, d = sp.fraction(f)
        n = sp.expand(n)
        if not n.is_polynomial(*V):
            raise NotImplementedError("SolveAlways: only equations polynomial (or rational) in the variables are "
                                      "supported")
        P = sp.Poly(n, *V)
        coeffs = [cc for mon, cc in P.terms()]
        if c.head == "Equal":
            zero_eqs.extend(coeffs)
        else:
            nonconst = [cc for mon, cc in P.terms() if any(mon)]
            const = P.coeff_monomial(1)
            zero_eqs.extend(nonconst)
            nonzero.append(const)
    zero_eqs = [e for e in zero_eqs if e != 0]
    params = sorted(set().union(*(e.free_symbols for e in zero_eqs + nonzero)) - set(V), key=str)
    if any(e.is_number and e != 0 for e in zero_eqs):
        return L()
    if not zero_eqs:
        sols = [{}]
    else:
        if not params:
            return L()
        sols = sp.solve(zero_eqs, params, dict=True)
    good = []
    for s in sols:
        if any(sp.simplify(e.subs(s)) != 0 for e in zero_eqs):
            raise ArithmeticError("SolveAlways: a solution failed verification")
        nz = [sp.simplify(e.subs(s)) for e in nonzero]
        if any(v == 0 for v in nz):
            continue
        if any(not v.is_number for v in nz):
            raise NotImplementedError("SolveAlways: the solution needs extra conditions on free parameters")
        good.append(s)
    from .wl_algebra import _wl_order_key
    good.sort(key=lambda s: tuple(_wl_order_key(s[p]) if p in s else (9,) for p in params))
    return L(*(L(*(rule(Symbol(p.name), out(s[p])) for p in params if p in s)) for s in good))


def _flat_and(e):
    if is_list(e) or _is_head(e, "And"):
        return [x for a in e.args for x in _flat_and(a)]
    return [e]


# ───────────────────────────── TrigFactorList, MatrixPolynomialValue ─────────────────────────────

def _trig_gens_rewrite(v):
    v = v.rewrite(sp.cos, sp.sin) if False else v
    rep = {sp.tan: lambda a: sp.sin(a) / sp.cos(a), sp.cot: lambda a: sp.cos(a) / sp.sin(a),
           sp.sec: lambda a: 1 / sp.cos(a), sp.csc: lambda a: 1 / sp.sin(a),
           sp.tanh: lambda a: sp.sinh(a) / sp.cosh(a), sp.coth: lambda a: sp.cosh(a) / sp.sinh(a),
           sp.sech: lambda a: 1 / sp.cosh(a), sp.csch: lambda a: 1 / sp.sinh(a)}
    for f, g in rep.items():
        v = v.replace(lambda e, f=f: isinstance(e, f), lambda e, g=g: g(e.args[0]))
    return v


def _pyth_variants(g):
    outs = []
    for f, h, sgn in ((sp.cos, sp.sin, -1), (sp.sin, sp.cos, -1), (sp.cosh, sp.sinh, 1), (sp.sinh, sp.cosh, -1)):
        args = {a.args[0] for a in g.atoms(f)}
        for a in args:
            if sgn == 1:      # cosh^2 = 1 + sinh^2
                outs.append(sp.expand(g.subs(f(a) ** 2, 1 + h(a) ** 2)))
            elif f is sp.sinh:  # sinh^2 = cosh^2 - 1
                outs.append(sp.expand(g.subs(f(a) ** 2, h(a) ** 2 - 1)))
            else:             # cos^2 = 1 - sin^2, sin^2 = 1 - cos^2
                outs.append(sp.expand(g.subs(f(a) ** 2, 1 - h(a) ** 2)))
    return outs


@fn("TrigFactorList", area=AREA, status="added",
    notes="TrigFactorList[expr]: {{c, 1}, {f1, e1}, ...} with expr == c Π fi^ei, after writing Tan, Cot, Sec, Csc "
          "(and hyperbolic analogues) in Sin and Cos, factoring as polynomials in the trigonometric functions and "
          "simplifying each factor with the Pythagorean identities (TrigFactorList[Cosh[x]^2 - Cosh[x]^4] -> "
          "{{-1, 1}, {Sinh[x], 2}, {Cosh[x], 2}}); the product is checked to equal expr; the chosen factors can "
          "differ from Wolfram's sum-to-product forms")
def _trigfactorlist(e):
    v = _S(e, "TrigFactorList")
    w = sp.together(_trig_gens_rewrite(v))
    facs = {}
    const = sp.Integer(1)

    def add(f, k):
        nonlocal const
        if f.is_number:
            const *= f ** k
            return
        c, ff = sp.factor_list(f)
        const *= c ** k
        for g, m in ff:
            g2 = sp.trigsimp(g)
            if g2 != g and isinstance(g2, (sp.Mul, sp.Pow)) and not g2.is_Add:
                add(g2, k * m)
                continue
            if isinstance(g2, sp.Pow) and g2.exp.is_Integer:
                add(g2.base, k * m * int(g2.exp))
                continue
            gg = g2 if (not g2.is_Add or sp.count_ops(g2) < sp.count_ops(g)) else g
            if (-gg).could_extract_minus_sign() is False and gg.could_extract_minus_sign():
                gg = -gg
                const *= (-1) ** (k * m)
            facs[gg] = facs.get(gg, 0) + k * m
    n, d = sp.fraction(w)
    add(n, 1)
    add(d, -1)
    facs = {f: k for f, k in facs.items() if k != 0}
    # Pythagorean identities: combine pairs of factors (c - 1)(c + 1) etc. into a single power when possible
    changed = True
    while changed:
        changed = False
        keys = [f for f in facs if f.is_Add]
        for f1, f2 in itertools.combinations(keys, 2):
            if facs.get(f1) != facs.get(f2) or f1 not in facs or f2 not in facs:
                continue
            g = sp.expand(f1 * f2)
            for h in _pyth_variants(g):
                c_, ff = sp.factor_list(h)
                if len(ff) == 1 and not ff[0][0].is_Add:
                    k = facs.pop(f1)
                    facs.pop(f2)
                    const *= c_ ** k
                    base, mexp = ff[0]
                    facs[base] = facs.get(base, 0) + mexp * k
                    changed = True
                    break
            if changed:
                break
    prod = const * sp.Mul(*(f ** k for f, k in facs.items()))
    from .wl_algebra import _numeric_equal
    if not (sp.simplify(prod - v) == 0 or _numeric_equal(prod, v)):
        raise ArithmeticError("TrigFactorList: factorisation failed verification")
    items = sorted(facs.items(), key=lambda t: sp.default_sort_key(t[0]))
    return L(L(out(const), 1), *(L(out(f), k) for f, k in items))


@fn("MatrixPolynomialValue", area=AREA, status="added",
    notes="MatrixPolynomialValue[poly, m, x] -> poly with x replaced by the square matrix m (constant term times "
          "the identity, powers as MatrixPower); MatrixPolynomialValue[{c0, c1, ...}, m] -> Σ ck m^k (Horner "
          "scheme, exact or numeric)")
def _matpolyval(p, m, *x):
    from .core import matrix_arg
    if len(x) > 1:
        raise ValueError("MatrixPolynomialValue called with too many arguments")
    M = sp.Matrix(matrix_arg(m, "MatrixPolynomialValue", square=True))
    nrow = M.rows
    if x:
        xs = _var1(x[0], "MatrixPolynomialValue")
        f = sp.expand(_S(p, "MatrixPolynomialValue"))
        if not f.is_polynomial(xs):
            raise ValueError("MatrixPolynomialValue: the first argument must be a polynomial in the variable")
        coeffs = list(reversed(sp.Poly(f, xs).all_coeffs())) if f != 0 else [0]
    else:
        if not is_list(p):
            raise ValueError("MatrixPolynomialValue[{c0, c1, ...}, m] needs a coefficient list")
        coeffs = [_S(c, "MatrixPolynomialValue") for c in p.args]
    R_ = sp.zeros(nrow, nrow)
    for c in reversed(coeffs):
        R_ = R_ * M + c * sp.eye(nrow)
    R_ = R_.applyfunc(sp.expand)
    from .core import numeric_out
    return numeric_out(out(R_), p, m) if False else out(R_)


# ───────────────────────────── Hermite and Smith normal forms ─────────────────────────────

class _IntRing:
    zero, one = 0, 1

    @staticmethod
    def size(a):
        return abs(a)

    @staticmethod
    def quo(a, b):            # Euclidean quotient (remainder sign follows Python floor division)
        return a // b

    @staticmethod
    def red_quo(a, b):        # quotient making the remainder lie in [0, b) for b > 0
        return a // b

    @staticmethod
    def unit(a):
        return -1 if a < 0 else 1

    @staticmethod
    def is_zero(a):
        return a == 0

    @staticmethod
    def out(a):
        return int(a)


class _PolyRing:
    def __init__(self, x, dom):
        self.x, self.dom = x, dom
        self.zero = sp.Poly(0, x, domain=dom)
        self.one = sp.Poly(1, x, domain=dom)

    def size(self, a):
        return a.degree()

    def quo(self, a, b):
        return a.div(b)[0]

    red_quo = quo

    def unit(self, a):
        return sp.Poly(1 / a.LC(), self.x, domain=self.dom)

    @staticmethod
    def is_zero(a):
        return a.is_zero

    @staticmethod
    def out(a):
        return out(sp.expand(a.as_expr()))


def _rows_op(M, i, j, q, R):
    """row_i -= q row_j"""
    M[i] = [a - q * b for a, b in zip(M[i], M[j])]


def _hnf(A, R):
    m = len(A)
    n = len(A[0]) if m else 0
    A = [row[:] for row in A]
    U = [[R.one if i == j else R.zero for j in range(m)] for i in range(m)]
    r = 0
    for c in range(n):
        if r >= m:
            break
        while True:
            rows = [i for i in range(r, m) if not R.is_zero(A[i][c])]
            if not rows:
                break
            piv = min(rows, key=lambda i: R.size(A[i][c]))
            A[r], A[piv] = A[piv], A[r]
            U[r], U[piv] = U[piv], U[r]
            done = True
            for i in range(r + 1, m):
                if not R.is_zero(A[i][c]):
                    q = R.quo(A[i][c], A[r][c])
                    _rows_op(A, i, r, q, R)
                    _rows_op(U, i, r, q, R)
                    if not R.is_zero(A[i][c]):
                        done = False
            if done:
                break
        if R.is_zero(A[r][c]):
            continue
        u = R.unit(A[r][c])
        A[r] = [u * a for a in A[r]]
        U[r] = [u * a for a in U[r]]
        for i in range(r):
            if not R.is_zero(A[i][c]):
                q = R.red_quo(A[i][c], A[r][c])
                _rows_op(A, i, r, q, R)
                _rows_op(U, i, r, q, R)
        r += 1
    return U, A


def _snf(A, R):
    m = len(A)
    n = len(A[0]) if m else 0
    A = [row[:] for row in A]
    U = [[R.one if i == j else R.zero for j in range(m)] for i in range(m)]
    V = [[R.one if i == j else R.zero for j in range(n)] for i in range(n)]

    def col_op(M, j, k, q):          # col_j -= q col_k
        for row in M:
            row[j] = row[j] - q * row[k]

    def col_swap(M, j, k):
        for row in M:
            row[j], row[k] = row[k], row[j]

    for k in range(min(m, n)):
        while True:
            cand = [(i, j) for i in range(k, m) for j in range(k, n) if not R.is_zero(A[i][j])]
            if not cand:
                break
            i0, j0 = min(cand, key=lambda ij: R.size(A[ij[0]][ij[1]]))
            A[k], A[i0] = A[i0], A[k]
            U[k], U[i0] = U[i0], U[k]
            col_swap(A, k, j0)
            col_swap(V, k, j0)
            clean = True
            for i in range(k + 1, m):
                if not R.is_zero(A[i][k]):
                    q = R.quo(A[i][k], A[k][k])
                    _rows_op(A, i, k, q, R)
                    _rows_op(U, i, k, q, R)
                    clean = clean and R.is_zero(A[i][k])
            for j in range(k + 1, n):
                if not R.is_zero(A[k][j]):
                    q = R.quo(A[k][j], A[k][k])
                    col_op(A, j, k, q)
                    col_op(V, j, k, q)
                    clean = clean and R.is_zero(A[k][j])
            if not clean:
                continue
            bad = None
            for i in range(k + 1, m):
                for j in range(k + 1, n):
                    if not R.is_zero(A[i][j]) and not R.is_zero(A[i][j] - R.quo(A[i][j], A[k][k]) * A[k][k]):
                        bad = i
                        break
                if bad is not None:
                    break
            if bad is None:
                break
            A[k] = [a + b for a, b in zip(A[k], A[bad])]
            U[k] = [a + b for a, b in zip(U[k], U[bad])]
        if k < m and k < n and not R.is_zero(A[k][k]):
            u = R.unit(A[k][k])
            A[k] = [u * a for a in A[k]]
            U[k] = [u * a for a in U[k]]
    return U, A, V


def _int_matrix(m, fname):
    from .core import matrix_arg
    M = matrix_arg(m, fname)
    rows = []
    for row in M.tolist():
        r = []
        for a in row:
            v = _S(a, fname) if not isinstance(a, int) else sp.Integer(a)
            if not v.is_Integer:
                raise NotImplementedError(f"{fname}: only integer matrices are supported")
            r.append(int(v))
        rows.append(r)
    return rows


def _poly_matrix(m, xs, fname):
    from .core import matrix_arg
    M = matrix_arg(m, fname)
    ents = [[sp.expand(a) for a in row] for row in M.tolist()]
    allsym = set().union(*(e.free_symbols for row in ents for e in row))
    if xs is None:
        if len(allsym) != 1:
            raise ValueError(f"{fname}: give the polynomial variable ({fname}[m, x])")
        xs = allsym.pop()
        allsym = set()
    else:
        allsym.discard(xs)
    if any(isinstance(a, sp.Float) for row in ents for e in row for a in sp.preorder_traversal(e)):
        raise NotImplementedError(f"{fname}: approximate coefficients are not supported")
    params = sorted(allsym, key=str)
    dom = sp.QQ.frac_field(*params) if params else sp.QQ
    if any(not e.is_polynomial(xs) for row in ents for e in row):
        raise ValueError(f"{fname}: the entries must be polynomials in {xs}")
    if any(e.has(sp.I) for row in ents for e in row):
        dom = sp.QQ_I if not params else None
        if dom is None:
            raise NotImplementedError(f"{fname}: complex symbolic coefficients are not supported")
    R_ = _PolyRing(xs, dom)
    return [[sp.Poly(e, xs, domain=dom) for e in row] for row in ents], R_


def _check_um(U, A0, H, mult):
    UM = sp.Matrix(U) * sp.Matrix(A0)
    if any(sp.expand(a - b) != 0 for a, b in zip(UM, sp.Matrix(H))):
        raise ArithmeticError("normal form failed verification")


@fn("HermiteDecomposition", area=AREA, status="added",
    notes="HermiteDecomposition[m] -> {u, r} for an integer matrix: u unimodular (Det = ±1), u.m == r, r the Hermite "
          "normal form (row echelon, positive pivots, entries above a pivot reduced into [0, pivot)), unique; u is "
          "unique when m is square and nonsingular (otherwise one valid u); non-integer entries raise")
def _hermitedecomp(m):
    A = _int_matrix(m, "HermiteDecomposition")
    U, H = _hnf(A, _IntRing)
    _check_um(U, A, H, None)
    return L(out(sp.Matrix(U)), out(sp.Matrix(H)))


@fn("HermiteReduce", area=AREA, status="added",
    notes="HermiteReduce[m]: the Hermite normal form r of an integer matrix (the second element of "
          "HermiteDecomposition)")
def _hermitereduce(m):
    A = _int_matrix(m, "HermiteReduce")
    U, H = _hnf(A, _IntRing)
    _check_um(U, A, H, None)
    return out(sp.Matrix(H))


def _poly_args(args, fname):
    args, opts = _opts(args, {"Modulus", "Method"}, fname)
    if opts.get("Modulus", 0) != 0:
        raise NotImplementedError(f"{fname}: the Modulus option is not supported")
    if not args or len(args) > 2:
        raise ValueError(f"{fname} expects {fname}[m] or {fname}[m, x]")
    xs = _var1(args[1], fname) if len(args) == 2 else None
    return args[0], xs


def _pm_out(M, R_):
    return L(*(L(*(R_.out(a) for a in row)) for row in M))


def _phd(*args):
    m, xs = _poly_args(args, "PolynomialHermiteDecomposition")
    A, R_ = _poly_matrix(m, xs, "PolynomialHermiteDecomposition")
    U, H = _hnf(A, R_)
    _check_um([[a.as_expr() for a in r] for r in U], [[a.as_expr() for a in r] for r in A],
              [[a.as_expr() for a in r] for r in H], None)
    return U, H, R_


@fn("PolynomialHermiteDecomposition", area=AREA, status="added",
    notes="PolynomialHermiteDecomposition[m] / [m, x] -> {u, h}: u.m == h with u unimodular (constant nonzero "
          "determinant) and h upper triangular (row echelon) with every entry above a pivot of lower degree than "
          "the pivot, over the rational numbers (or rational functions of other symbols). The Hermite form is "
          "canonical only up to constant multiples of the rows; MikoshiLang makes the pivots monic (Wolfram's row "
          "scaling differs). Modulus and approximate coefficients raise")
def _polyhermitedecomp(*args):
    U, H, R_ = _phd(*args)
    return L(_pm_out(U, R_), _pm_out(H, R_))


@fn("PolynomialHermiteReduce", area=AREA, status="added",
    notes="PolynomialHermiteReduce[m] / [m, x]: the Hermite form h of a univariate polynomial matrix (second "
          "element of PolynomialHermiteDecomposition; pivots monic)")
def _polyhermitereduce(*args):
    U, H, R_ = _phd(*args)
    return _pm_out(H, R_)


def _psd(*args, fname):
    m, xs = _poly_args(args, fname)
    A, R_ = _poly_matrix(m, xs, fname)
    U, S, V = _snf(A, R_)
    UM = sp.Matrix([[a.as_expr() for a in r] for r in U]) * sp.Matrix([[a.as_expr() for a in r] for r in A]) * \
        sp.Matrix([[a.as_expr() for a in r] for r in V])
    Sm = sp.Matrix([[a.as_expr() for a in r] for r in S])
    if any(sp.expand(a - b) != 0 for a, b in zip(UM, Sm)):
        raise ArithmeticError(f"{fname}: decomposition failed verification")
    d = [S[i][i] for i in range(min(len(S), len(S[0]) if S else 0)) if not S[i][i].is_zero]
    for a, b in zip(d, d[1:]):
        if not b.rem(a).is_zero:
            raise ArithmeticError(f"{fname}: invariant factors failed the divisibility check")
    return U, S, V, R_


@fn("PolynomialSmithDecomposition", area=AREA, status="added",
    notes="PolynomialSmithDecomposition[m] / [m, x] -> {u, s, v}: u.m.v == s with u, v unimodular and s diagonal "
          "with each diagonal entry dividing the next (the invariant factors), over the rationals or rational "
          "functions of other symbols; invariant factors are unique up to constants and are made monic here "
          "(Wolfram leaves constant multiples). Checked")
def _polysmithdecomp(*args):
    U, S, V, R_ = _psd(*args, fname="PolynomialSmithDecomposition")
    return L(_pm_out(U, R_), _pm_out(S, R_), _pm_out(V, R_))


@fn("PolynomialSmithReduce", area=AREA, status="added",
    notes="PolynomialSmithReduce[m] / [m, x]: the Smith form s (monic invariant factors on the diagonal)")
def _polysmithreduce(*args):
    U, S, V, R_ = _psd(*args, fname="PolynomialSmithReduce")
    return _pm_out(S, R_)


# ───────────────────────────── Boolean minterms, maxterms, consecutive functions, unateness ─────────────────────

def _bool_vectors(spec, n, fname):
    """Minterm/maxterm specification -> list of integer indices (a = most significant bit)."""
    if not is_list(spec):
        raise ValueError(f"{fname}: the terms must be given as a list")
    idx = []
    for t in spec.args:
        if K.is_int(t):
            v = int(t)
            if n is not None and not (0 <= v < 2 ** n):
                raise ValueError(f"{fname}: index {v} is out of range for {n} variables")
            idx.append(v)
        elif is_list(t):
            bits = []
            for b in t.args:
                tb = _tf(b)
                if tb is None and K.is_int(b) and int(b) in (0, 1):
                    tb = int(b) == 1
                if tb is None:
                    raise ValueError(f"{fname}: vector entries must be True/False or 1/0")
                bits.append(tb)
            if n is not None and len(bits) != n:
                raise ValueError(f"{fname}: vector {t!r} must have {n} entries")
            n = len(bits) if n is None else n
            v = 0
            for b in bits:
                v = 2 * v + int(b)
            idx.append(v)
        else:
            raise ValueError(f"{fname}: terms must be integers or 0/1 vectors")
    return idx, n


def _terms_fn(fname, maxterms):
    def impl(spec, *rest):
        if len(rest) > 2:
            raise ValueError(f"{fname} called with too many arguments")
        vars_ = rest[0] if rest else None
        form = rest[1] if len(rest) == 2 else None
        if vars_ is None:
            n = None
        elif K.is_int(vars_):
            n = int(vars_)
        elif is_list(vars_):
            n = len(vars_.args)
        else:
            raise ValueError(f"{fname}: the second argument must be a variable list or a number of variables")
        idx, n = _bool_vectors(spec, n, fname)
        if n is None:
            raise ValueError(f"{fname}: give the number of variables")
        if maxterms:       # maxterm i is False exactly at the input 2^n - 1 - i (Wolfram's convention)
            false_at = {2 ** n - 1 - i for i in idx}
            k = sum(2 ** v for v in range(2 ** n) if v not in false_at)
        else:
            k = sum(2 ** v for v in set(idx))
        if not is_list(vars_):
            if form is not None:
                raise ValueError(f"{fname}: a form needs a variable list")
            return Expr("BooleanFunction", k, n)
        f = ev(Expr("BooleanFunction", k, vars_))
        want = form if form is not None else ("CNF" if maxterms else "DNF")
        if not isinstance(want, str):
            raise ValueError(f"{fname}: the form must be a string such as \"DNF\", \"CNF\" or \"ANF\"")
        if maxterms or form is not None:
            f = ev(Expr("BooleanConvert", f, want))
        return f
    return impl


fn("BooleanMinterms", area=AREA, status="added",
   notes="BooleanMinterms[{i1, ...}, {a, b, ...}]: the Boolean function True exactly at the minterms ik (k's binary "
         "digits give the inputs, first variable most significant; 0/1 or True/False vectors also accepted) as a "
         "minimal DNF (BooleanMinterms[{1, 3, 5}, {a, b, c}] -> (!a && c) || (!b && c)); BooleanMinterms[spec, n] "
         "and BooleanMinterms[{vectors}] give BooleanFunction[k, n]; a third argument \"DNF\"/\"CNF\"/\"ANF\"/... "
         "converts with BooleanConvert")(_terms_fn("BooleanMinterms", False))
fn("BooleanMaxterms", area=AREA, status="added",
   notes="BooleanMaxterms[{i1, ...}, {a, b, ...}]: the conjunction of the maxterms ik (maxterm i has the variable "
         "where the bit of i is 1 and its negation where it is 0, Wolfram's convention, so it is False exactly at "
         "the input 2^n - 1 - i) as a minimal CNF (BooleanMaxterms[{1, 3, 5}, {a, b, c}] -> (!a || c) && (!b || "
         "c)); BooleanMaxterms[spec, n] -> BooleanFunction[k, n]; optional form argument")(
    _terms_fn("BooleanMaxterms", True))


def _consec_cells(spec, dims, fname):
    """All windows of k consecutive cells (product over dimensions); circular per dimension."""
    if K.is_int(spec):
        ks, circ = [int(spec)], [False]
    elif is_list(spec) and len(spec.args) == 2 and (_tf(spec.args[1]) is not None or is_list(spec.args[1])) \
            and not (is_list(spec.args[0]) is False and is_list(spec.args[1])):
        a, b = spec.args
        ks = [int(v) for v in (a.args if is_list(a) else [a])]
        cs = [_tf(v) for v in (b.args if is_list(b) else [b] * len(ks))]
        if None in cs:
            raise ValueError(f"{fname}: circularity must be True or False")
        circ = cs
    elif is_list(spec) and all(K.is_int(v) for v in spec.args):
        ks, circ = [int(v) for v in spec.args], [False] * len(spec.args)
    else:
        raise ValueError(f"{fname}: expected k, {{k, circular}}, {{k1, k2, ...}} or {{{{k1, ...}}, {{c1, ...}}}}")
    if len(ks) != len(dims):
        raise ValueError(f"{fname}: the window specification and the dimensions must have the same length")
    if any(k < 1 or k > d for k, d in zip(ks, dims)):
        raise ValueError(f"{fname}: each k must be between 1 and the corresponding dimension")
    starts = [range(d) if c and k < d else range(d - k + 1) for k, d, c in zip(ks, dims, circ)]
    wins = []
    for st in itertools.product(*starts):
        cells = list(itertools.product(*[[(s + j) % d for j in range(k)] for s, k, d in zip(st, ks, dims)]))
        if cells not in wins:
            wins.append(cells)
    return wins


def _consec_expr(spec, dims, items, fname):
    import numpy as _np  # noqa: F401
    total = 1
    for d in dims:
        total *= d
    if len(items) != total:
        raise ValueError(f"{fname}: expected {total} arguments")

    def flat(cell):
        i = 0
        for c, d in zip(cell, dims):
            i = i * d + c
        return i
    wins = _consec_cells(spec, dims, fname)
    terms = [Expr("And", *(items[flat(c)] for c in w)) if len(w) > 1 else items[flat(w[0])] for w in wins]
    return ev(Expr("Or", *terms)) if len(terms) > 1 else ev(terms[0])


def _flatten_items(x):
    if is_list(x):
        return [y for a in x.args for y in _flatten_items(a)]
    return [x]


@fn("BooleanConsecutiveFunction", area=AREA, status="added",
    notes="BooleanConsecutiveFunction[k, n][x1, ..., xn]: True when k consecutive arguments are True (linear "
          "system); {k, True} makes it circular; {{k1, k2}, {n1, n2}} (with optional circularity {{k1, k2}, {c1, "
          "c2}}) uses k1 x k2 windows in an n1 x n2 array (arguments flattened row by row or given as an array). "
          "Applied to arguments it gives the equivalent DNF Or of window conjunctions (Wolfram keeps the operator "
          "unevaluated; BooleanConvert of it gives the same DNF); BooleanConsecutiveFunction[spec, {vars}] / "
          "[spec, {vars}, form] give the formula directly")
def _bcfn(spec, n, *form):
    if len(form) > 1:
        raise ValueError("BooleanConsecutiveFunction called with too many arguments")
    if K.is_int(n) or (is_list(n) and all(K.is_int(v) for v in n.args)):
        if form:
            raise ValueError("BooleanConsecutiveFunction: a form needs explicit variables")
        dims = [int(n)] if K.is_int(n) else [int(v) for v in n.args]
        _consec_cells(spec, dims, "BooleanConsecutiveFunction")       # validate
        raise Unevaluated()
    if not is_list(n):
        raise ValueError("BooleanConsecutiveFunction: the second argument must be n, {n1, ...} or a variable list")
    dims, depth = [], n
    while is_list(depth):
        dims.append(len(depth.args))
        depth = depth.args[0] if depth.args else None
    items = _flatten_items(n)
    if K.is_int(spec) or (is_list(spec) and len(spec.args) == 2 and _tf(spec.args[1]) is not None):
        dims = [len(items)]
    f = _consec_expr(spec, dims, items, "BooleanConsecutiveFunction")
    if form:
        f = ev(Expr("BooleanConvert", f, form[0]))
    return f


def _bcf_hook(head, args, ev_):
    if len(head.args) != 2:
        return K.UNHANDLED
    spec, n = head.args
    if K.is_int(n):
        dims = [int(n)]
    elif is_list(n) and all(K.is_int(v) for v in n.args):
        dims = [int(v) for v in n.args]
    else:
        return K.UNHANDLED
    items = [y for a in args for y in _flatten_items(a)]
    return _consec_expr(spec, dims, items, "BooleanConsecutiveFunction")


from .. import kernel_lists as _KL   # noqa: E402
_KL.COMPOUND_HEAD_HOOKS["BooleanConsecutiveFunction"] = _bcf_hook


def _truth(f, vals, vars_):
    """Value of the Boolean expression / pure function f at an assignment."""
    if vars_ is None:
        r = ev(K.call(f, *vals))
    else:
        r = ev(Expr("ReplaceAll", f, L(*(rule(v, b) for v, b in zip(vars_, vals)))))
    t = _tf(r)
    if t is None:
        raise ValueError(f"UnateQ: the function does not evaluate to True or False ({r!r})")
    return t


@fn("UnateQ", area=AREA, status="added",
    notes="UnateQ[f]: True if the Boolean function f is positive unate (monotone non-decreasing) in all its "
          "variables; UnateQ[f, {x, !y, ...}] checks positive unateness in x and negative unateness in y (only the "
          "listed variables); pure functions and BooleanFunction take a list of True (positive), False (negative) "
          "or _ (ignore) per argument; decided from the complete truth table")
def _unateq(f, *spec):
    if len(spec) > 1:
        raise ValueError("UnateQ called with too many arguments")
    pure = _is_head(f, "Function") or _is_head(f, "BooleanFunction")
    if pure:
        if _is_head(f, "BooleanFunction"):
            if len(f.args) != 2 or not K.is_int(f.args[1]):
                raise ValueError("UnateQ: expected BooleanFunction[k, n]")
            n = int(f.args[1])
        else:
            slots = []

            def walk(e):
                if _is_head(e, "Slot") and e.args and K.is_int(e.args[0]):
                    slots.append(int(e.args[0]))
                elif isinstance(e, Expr):
                    for a in e.args:
                        walk(a)
            walk(f.args[-1])
            n = max(slots) if slots else 0
        if spec:
            if not (is_list(spec[0]) and len(spec[0].args) == n):
                raise ValueError(f"UnateQ: give one True/False/_ per argument ({n})")
            pol = []
            for s in spec[0].args:
                t = _tf(s)
                pol.append(t if t is not None else None)
        else:
            pol = [True] * n
        vars_ = None
    else:
        vs = ev(Expr("BooleanVariables", f))
        vars_ = list(vs.args)
        n = len(vars_)
        pol = [True] * n
        if spec:
            pol = [None] * n
            sl = list(spec[0].args) if is_list(spec[0]) else [spec[0]]
            for s in sl:
                neg = _is_head(s, "Not")
                v = s.args[0] if neg else s
                hits = [i for i, w in enumerate(vars_) if K.same(w, v)]
                if not hits:
                    vars_.append(v)
                    pol.append(not neg)
                    n += 1
                else:
                    pol[hits[0]] = not neg
    if n > 16:
        raise NotImplementedError("UnateQ: more than 16 variables are not supported")
    table = {}
    for vals in itertools.product([False, True], repeat=n):
        table[vals] = _truth(f, list(vals), vars_)
    for i, p in enumerate(pol):
        if p is None:
            continue
        for vals in itertools.product([False, True], repeat=n):
            if vals[i]:
                continue
            lo = table[vals]
            hi = table[vals[:i] + (True,) + vals[i + 1:]]
            if p and lo and not hi:
                return False
            if not p and hi and not lo:
                return False
    return True


# ───────────────────────────── function properties (continuity, analyticity, sign, monotonicity, ...) ────────────
#
# Every decision is either proved (a verified witness point, or an empty solution set from SymPy's solveset over an
# explicit domain) or the function raises a clear error; nothing is guessed.  Only a whitelist of elementary heads
# is understood; anything else raises NotImplementedError.

_ENTIRE1 = {"Sin", "Cos", "Sinh", "Cosh", "Exp", "Erf", "Erfc", "Erfi"}
_REAL_SMOOTH1 = _ENTIRE1 | {"Tanh", "ArcTan", "ArcSinh", "Sech"}


def _free_of(e, vs):
    return not any(_contains(e, lambda s, v=v: K.same(s, v)) for v in vs)


def _or(*cs):
    cs = [c for c in cs if c is not False]
    out_ = []
    for c in cs:
        if c is True:
            return True
        for d in (c.args if _is_head(c, "Or") else [c]):
            if not any(K.same(d, o) for o in out_):
                out_.append(d)
    if not out_:
        return False
    return out_[0] if len(out_) == 1 else Expr("Or", *out_)


def _eq0(g):
    """g == 0, written with the constant moved to the right (x^2 - 1 == 0 -> x^2 == 1, Wolfram style)."""
    g = ev(g)
    if _is_head(g, "Plus"):
        consts = [a for a in g.args if _isnum(a)]
        if len(consts) == 1 and len(g.args) > 1:
            rest = [a for a in g.args if a is not consts[0]]
            lhs = rest[0] if len(rest) == 1 else Expr("Plus", *rest)
            return Expr("Equal", lhs, ev(Expr("Times", -1, consts[0])))
    return Expr("Equal", g, 0)


def _bad(e, vs, kind, dom, fname, decide=False):
    """Condition (ML) on the variables where e fails to be continuous ("disc"), analytic ("sing") or meromorphic
    ("mero") on dom ("Reals"/"Complexes"); False when it never fails.  Raises for unsupported heads."""
    if _free_of(e, vs):
        return False
    if any(K.same(e, v) for v in vs):
        return False
    if not isinstance(e, Expr) or not isinstance(e.head, str):
        raise NotImplementedError(f"{fname}: cannot analyse {e!r}")
    h, a = e.head, e.args
    real = dom == "Reals"
    if h in ("Plus", "Times"):
        return _or(*(_bad(t, vs, kind, dom, fname, decide) for t in a))
    if h == "Power" and len(a) == 2:
        b, n = a
        if _free_of(b, vs):
            if K.same(b, Symbol("E")) or (_isnum(b) and ev(Expr("Greater", b, 0)) is True):
                return _bad(n, vs, kind, dom, fname, decide)
            raise NotImplementedError(f"{fname}: cannot analyse the power {e!r}")
        if not _free_of(n, vs):
            raise NotImplementedError(f"{fname}: cannot analyse the power {e!r}")
        inner = _bad(b, vs, kind, dom, fname, decide)
        if K.is_int(n):
            return _or(inner, _eq0(b)) if int(n) < 0 and kind != "mero" else inner
        if not _isnum(n):
            raise NotImplementedError(f"{fname}: symbolic exponent in {e!r}")
        if kind == "mero":
            return _or(inner, Expr("And", Expr("Equal", Expr("Im", b), 0), Expr("LessEqual", Expr("Re", b), 0)))
        if real:
            pos = ev(Expr("Greater", n, 0)) is True
            return _or(inner, Expr("Less" if decide and kind == "disc" and pos else "LessEqual", b, 0))
        op = "Less" if kind == "disc" and ev(Expr("Greater", n, 0)) is True else "LessEqual"
        cut = Expr("And", Expr("Equal", Expr("Im", b), 0), Expr(op, Expr("Re", b), 0))
        return _or(inner, cut, _eq0(b)) if ev(Expr("Less", n, 0)) is True else _or(inner, cut)
    if len(a) != 1:
        if h == "Log" and len(a) == 2 and _free_of(a[0], vs):
            return _bad(Expr("Log", a[1]), vs, kind, dom, fname, decide)
        raise NotImplementedError(f"{fname}: cannot analyse {h} with {len(a)} arguments")
    u = a[0]
    inner = _bad(u, vs, kind, dom, fname, decide)
    if h in _ENTIRE1:
        return inner
    if h == "Log":
        if real and kind != "mero":
            return _or(inner, Expr("LessEqual", u, 0))
        return _or(inner, _eq0(u) if kind != "mero" else False,
                   Expr("And", Expr("Equal", Expr("Im", u), 0), Expr("LessEqual", Expr("Re", u), 0)))
    trig_poles = {"Tan": "Cos", "Sec": "Cos", "Cot": "Sin", "Csc": "Sin"}
    if h in trig_poles:
        return inner if kind == "mero" else _or(inner, _eq0(Expr(trig_poles[h], u)))
    if h in ("Coth", "Csch"):
        return inner if kind == "mero" else _or(inner, _eq0(u) if real else _eq0(Expr("Sinh", u)))
    if h in ("Tanh", "Sech"):
        return inner if real or kind == "mero" else _or(inner, _eq0(Expr("Cosh", u)))
    if h == "Gamma":
        if kind == "mero":
            return inner
        return _or(inner, Expr("And", _eq0(Expr("Sin", Expr("Times", Symbol("Pi"), u))),
                               Expr("LessEqual", u if real else Expr("Re", u), 0)))
    if not real or kind == "mero":
        raise NotImplementedError(f"{fname}: {h} is not supported over the complex numbers")
    if h in ("ArcTan", "ArcSinh"):
        return inner
    if h in ("ArcSin", "ArcCos"):
        op1, op2 = ("Less", "Greater") if kind == "disc" else ("LessEqual", "GreaterEqual")
        return _or(inner, Expr(op1, u, -1), Expr(op2, u, 1))
    if h == "Abs":
        return inner if kind == "disc" else _or(inner, _eq0(u))
    if h in ("Floor", "Ceiling"):
        return _or(inner, _eq0(Expr("Sin", Expr("Times", Symbol("Pi"), u))))
    if h == "Round":
        return _or(inner, _eq0(Expr("Cos", Expr("Times", Symbol("Pi"), u))))
    if h in ("Sign", "UnitStep"):
        return _or(inner, _eq0(u))
    raise NotImplementedError(f"{fname}: cannot analyse {h}")


def _prop_args(args, fname, default_dom="Reals", extra_opts=()):
    """-> (f, cond, vars, dom, opts) for f / {f, cond} [, vars [, dom]] [, opts]."""
    args, opts = _opts(args, ("Assumptions", "GenerateConditions", "PerformanceGoal") + tuple(extra_opts), fname)
    if len(args) not in (2, 3):
        raise ValueError(f"{fname} expects a function, variables and optionally a domain")
    f, vs = args[0], _vars(args[1], fname)
    dom = default_dom
    if len(args) == 3:
        if not isinstance(args[2], Symbol) or args[2].name not in ("Reals", "Complexes", "Integers"):
            raise ValueError(f"{fname}: the domain must be Reals or Complexes")
        dom = args[2].name
    if dom == "Integers":
        raise NotImplementedError(f"{fname}: the Integers domain is not supported")
    cond = True
    if is_list(f):
        if len(f.args) != 2:
            raise ValueError(f"{fname}: expected {{f, cond}}")
        f, cond = f.args
    a = opts.get("Assumptions")
    if a is not None and a is not True:
        raise NotImplementedError(f"{fname}: the Assumptions option is not supported")
    return f, cond, vs, dom, opts


def _rx(vs):
    return {sp.Symbol(v.name): sp.Symbol(v.name, real=True) for v in vs if isinstance(v, Symbol)}


def _sp_real(e, vs, fname):
    if not all(isinstance(v, Symbol) for v in vs):
        raise NotImplementedError(f"{fname}: the variables must be symbols")
    s = tosp(e, fname)
    return s.xreplace(_rx(vs)) if hasattr(s, "xreplace") else s


def _real_set(c, x, fname):
    """Set of real x satisfying the SymPy condition c (raises if SymPy cannot solve it)."""
    if c is sp.true or c is True:
        return sp.S.Reals
    if c is sp.false or c is False:
        return sp.S.EmptySet
    if isinstance(c, sp.And):
        s = sp.S.Reals
        for t in c.args:
            s = sp.Intersection(s, _real_set(t, x, fname))
        return s
    if isinstance(c, sp.Or):
        return sp.Union(*(_real_set(t, x, fname) for t in c.args))
    if isinstance(c, sp.Not):
        return sp.Complement(sp.S.Reals, _real_set(c.args[0], x, fname))
    if isinstance(c, sp.Ne):
        return sp.Complement(sp.S.Reals, sp.solveset(sp.Eq(c.lhs, c.rhs), x, sp.S.Reals))
    if not c.free_symbols <= {x}:
        raise NotImplementedError(f"{fname}: the condition {c} involves parameters")
    s = sp.solveset(c, x, sp.S.Reals)
    if isinstance(s, sp.ConditionSet) or s.has(sp.ConditionSet):
        raise NotImplementedError(f"{fname}: cannot solve the condition {c}")
    return s


def _imageset_witness(s):
    """A verified member of a set built from ImageSets over the integers (intersected with intervals), or None."""
    parts = []

    def collect(t):
        if isinstance(t, sp.ImageSet):
            parts.append(t)
        elif isinstance(t, (sp.Union, sp.Intersection, sp.Complement)):
            for a in t.args:
                collect(a)
    collect(s)
    for p in parts:
        if p.base_sets != (sp.S.Integers,) or len(p.lamda.variables) != 1:
            continue
        for k in sorted(range(-300, 301), key=abs):
            v = p.lamda(sp.Integer(k))
            c = s.contains(v)
            if c is sp.true:
                return v
    return None


def _is_empty(s, fname):
    if isinstance(s, sp.Complement) and (s.args[0] == sp.S.Reals or (isinstance(s.args[0], sp.Interval) and
                                                                      s.args[0].measure.is_positive)):
        ex = s.args[1]
        if all(isinstance(q, (sp.ImageSet, sp.FiniteSet)) for q in (ex.args if isinstance(ex, sp.Union) else [ex])):
            return False      # an interval minus countably many points
    if s.has(sp.ImageSet) and not s.has(sp.ConditionSet):
        if _imageset_witness(s) is not None:
            return False
    if s.has(sp.ConditionSet):
        raise NotImplementedError(f"{fname}: cannot decide the set {s}")
    e = s.is_empty
    if e is None:
        s2 = sp.simplify(s) if not isinstance(s, sp.Set) else s
        e = s2.is_empty
    if e is None:
        raise NotImplementedError(f"{fname}: cannot decide whether {s} is empty")
    return e


def _sample_points(D, n=48):
    """Rational sample points in the real set D (interior points of its intervals, plus isolated points)."""
    pts = []
    ivs = []
    if isinstance(D, sp.Interval):
        ivs = [D]
    elif isinstance(D, sp.Union):
        ivs = [s for s in D.args if isinstance(s, sp.Interval)]
        pts += [p for s in D.args if isinstance(s, sp.FiniteSet) for p in s]
    elif isinstance(D, sp.FiniteSet):
        return list(D)
    elif D == sp.S.Reals:
        ivs = [sp.Interval(-sp.oo, sp.oo)]
    else:
        for k in range(-60, 61):
            p = sp.Rational(k, 7)
            if D.contains(p) is sp.true:
                pts.append(p)
        return pts
    for iv in ivs:
        lo, hi = iv.start, iv.end
        if lo == -sp.oo and hi == sp.oo:
            cand = [sp.Rational(k, 4) for k in range(-4 * 12, 4 * 12 + 1)] + [sp.Integer(k) for k in (-1000, -100, -50,
                                                                                                       50, 100, 1000)]
        elif lo == -sp.oo:
            cand = [hi - sp.Rational(k, 8) for k in range(1, n)] + [hi - 10 ** k for k in range(1, 4)]
        elif hi == sp.oo:
            cand = [lo + sp.Rational(k, 8) for k in range(1, n)] + [lo + 10 ** k for k in range(1, 4)]
        else:
            cand = [lo + (hi - lo) * sp.Rational(k, n) for k in range(1, n)]
        pts += [p for p in cand if iv.contains(p) is sp.true]
    return pts


def _num(v):
    """(is_real, sign) of an exact numeric SymPy value, decided at 50 digits; None if undecidable/undefined."""
    if v.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        return None
    z = sp.N(v, 50)
    if not z.is_number:
        return None
    re_, im_ = z.as_real_imag()
    if abs(im_) > 1e-40:
        return (False, 0)
    if abs(re_) < 1e-40:
        if sp.simplify(v) == 0:
            return (True, 0)
        return None
    return (True, 1 if re_ > 0 else -1)


def _real_setup(f, cond, vs, fname):
    """Univariate real analysis: -> (x, fs, D, nonreal_witness) where D is the domain set."""
    if len(vs) != 1:
        raise NotImplementedError(f"{fname}: only univariate functions are supported over the reals")
    x = sp.Symbol(vs[0].name, real=True)
    fs = _sp_real(f, vs, fname)
    if fs.free_symbols - {x}:
        raise NotImplementedError(f"{fname}: the function contains parameters {sorted(map(str, fs.free_symbols - {x}))}")
    D = _real_set(_sp_real(cond, vs, fname), x, fname)
    return x, fs, D


def _disc_set(f, vs, D, x, fname, kind="disc"):
    c = _bad(ev(f), vs, kind, "Reals", fname, True)
    if c is False:
        return sp.S.EmptySet
    return sp.Intersection(D, _real_set(_sp_real(c, vs, fname), x, fname))


def _split_abs(fs, x, D, fname):
    """Pieces (open intervals and points) of D on which every Abs/Sign argument has constant sign, with fs
    rewritten without Abs/Sign there.  -> list of (set, expr)."""
    args = [a.args[0] for a in fs.atoms(sp.Abs, sp.sign)]
    if not args:
        return [(D, fs)]
    cuts = set()
    for g in args:
        z = sp.solveset(g, x, sp.S.Reals)
        if not isinstance(z, sp.FiniteSet):
            raise NotImplementedError(f"{fname}: cannot split at the zeros of {g}")
        cuts |= set(z)
    cuts = sorted(cuts, key=lambda c: float(c))
    edges = [-sp.oo] + cuts + [sp.oo]
    pieces = []
    for lo, hi in zip(edges, edges[1:]):
        iv = sp.Intersection(D, sp.Interval.open(lo, hi))
        if _is_empty(iv, fname):
            continue
        mid = (lo + hi) / 2 if lo != -sp.oo and hi != sp.oo else (hi - 1 if lo == -sp.oo and hi != sp.oo else
                                                                     (lo + 1 if lo != -sp.oo else sp.Integer(0)))
        rep = {}
        for a in fs.atoms(sp.Abs):
            s = _num(a.args[0].subs(x, mid))
            if s is None or s[1] == 0:
                raise NotImplementedError(f"{fname}: cannot fix the sign of {a.args[0]}")
            rep[a] = s[1] * a.args[0]
        for a in fs.atoms(sp.sign):
            s = _num(a.args[0].subs(x, mid))
            if s is None or s[1] == 0:
                raise NotImplementedError(f"{fname}: cannot fix the sign of {a.args[0]}")
            rep[a] = sp.Integer(s[1])
        pieces.append((iv, fs.xreplace(rep)))
    for c in cuts:
        if D.contains(c) is sp.true:
            pieces.append((sp.FiniteSet(c), fs))
    return pieces


def _sign_on(fs, x, D, fname, strict=False):
    """1, -1, 0 (identically zero) or Indeterminate for the sign of the real-valued continuous fs on D."""
    if _is_empty(D, fname):
        return 0
    pos = neg = zero = False
    for p in _sample_points(D):
        r = _num(fs.subs(x, p))
        if r is None:
            continue
        if not r[0]:
            return Symbol("Indeterminate")
        pos |= r[1] > 0
        neg |= r[1] < 0
        zero |= r[1] == 0
        if pos and neg:
            return Symbol("Indeterminate")
    pieces = _split_abs(fs, x, D, fname)

    def empty_where(rel):
        for S, g in pieces:
            if isinstance(S, sp.FiniteSet):
                for p in S:
                    r = _num(g.subs(x, p))
                    if r is None:
                        raise NotImplementedError(f"{fname}: cannot evaluate at {p}")
                    if (rel == "<" and r[1] < 0) or (rel == ">" and r[1] > 0) or (rel == "=" and r[1] == 0):
                        return False
                continue
            g2 = sp.simplify(g)
            if rel == "=" and g2 == 0:
                return False
            if g2.is_number:
                r = _num(g2)
                if r is None:
                    raise NotImplementedError(f"{fname}: cannot evaluate {g2}")
                if (rel == "<" and r[1] < 0) or (rel == ">" and r[1] > 0) or (rel == "=" and r[1] == 0):
                    return False
                continue
            relx = {"<": g2 < 0, ">": g2 > 0, "=": sp.Eq(g2, 0)}[rel]
            s = sp.solveset(relx, x, sp.S.Reals)
            if isinstance(s, sp.ConditionSet) or s.has(sp.ConditionSet):
                raise NotImplementedError(f"{fname}: cannot decide the sign of {g2}")
            if not _is_empty(sp.Intersection(S, s), fname):
                return False
        return True
    if not neg and empty_where("<"):
        if not pos and empty_where(">"):
            return 0
        if strict and (zero or not empty_where("=")):
            return Symbol("Indeterminate")
        return 1
    if not pos and empty_where(">"):
        if strict and (zero or not empty_where("=")):
            return Symbol("Indeterminate")
        return -1
    if pos and neg:
        return Symbol("Indeterminate")
    raise NotImplementedError(f"{fname}: cannot decide the sign of {fs} on {D}")


def _require_real_continuous(f, vs, D, x, fs, fname):
    """None if f is real and continuous on D; Indeterminate if it is provably non-real somewhere in D; raises
    otherwise (isolated singular points are not handled)."""
    B = _disc_set(f, vs, D, x, fname)
    if _is_empty(B, fname):
        return None
    for p in _sample_points(sp.Intersection(D, B)) + _sample_points(D):
        r = _num(fs.subs(x, p))
        if r is not None and not r[0]:
            return Symbol("Indeterminate")
    raise NotImplementedError(f"{fname}: the function is not continuous on the whole domain (at {B}); restrict "
                              f"the domain with a condition")


def _strict_opt(opts, fname):
    s = opts.get("StrictInequalities", False)
    if s not in (True, False):
        raise ValueError(f"{fname}: StrictInequalities must be True or False")
    return s


@fn("FunctionSign", area=AREA, status="added",
    notes="FunctionSign[f, x] / [{f, cond}, x]: 1 if f >= 0, -1 if f <= 0, 0 if f == 0 on the (real) domain, "
          "Indeterminate if f takes both signs or is not real somewhere; StrictInequalities -> True asks for f > 0 "
          "/ f < 0. Univariate elementary functions over Reals; proofs by solveset (Abs split at its zeros) and "
          "verified sample points; anything undecidable raises")
def _fsign(*args):
    f, cond, vs, dom, opts = _prop_args(args, "FunctionSign", extra_opts=("StrictInequalities",))
    if dom != "Reals":
        raise NotImplementedError("FunctionSign: only the Reals domain is supported")
    x, fs, D = _real_setup(f, cond, vs, "FunctionSign")
    r = _require_real_continuous(f, vs, D, x, fs, "FunctionSign")
    if r is not None:
        return r
    return _sign_on(fs, x, D, "FunctionSign", _strict_opt(opts, "FunctionSign"))


def _interval_dom(D, fname):
    if isinstance(D, sp.Interval) or D == sp.S.Reals:
        return
    raise NotImplementedError(f"{fname}: the domain must be a single interval")


def _deriv_pieces(fs, x, D, fname):
    """Derivative of fs on D with Abs split -> sign of the derivative over the open pieces."""
    return [(S, sp.diff(g, x)) for S, g in _split_abs(fs, x, D, fname) if not isinstance(S, sp.FiniteSet)]


def _thin_zeros(d, x, S, fname):
    """True if the zero set of d in S contains no interval (so a one-signed d gives strict monotonicity)."""
    if sp.simplify(d) == 0:
        return False
    z = sp.solveset(sp.Eq(d, 0), x, sp.S.Reals)
    if isinstance(z, sp.ConditionSet) or z.has(sp.ConditionSet):
        raise NotImplementedError(f"{fname}: cannot solve {d} == 0")
    parts = z.args if isinstance(z, sp.Union) else [z]
    return not any(isinstance(p, sp.Interval) for p in parts)


def _monotonicity(f, cond, vs, opts, fname):
    x, fs, D = _real_setup(f, cond, vs, fname)
    r = _require_real_continuous(f, vs, D, x, fs, fname)
    if r is not None:
        return r
    _interval_dom(D, fname)
    strict = _strict_opt(opts, fname)
    signs = []
    for S, d in _deriv_pieces(fs, x, D, fname):
        sg = _sign_on(d, x, S, fname)
        if strict and sg in (1, -1) and not _thin_zeros(d, x, S, fname):
            sg = 0
        signs.append(sg)
    if not signs:
        return 0
    if all(s == 0 for s in signs):
        return 0
    nz = [s for s in signs if s != 0]
    if any(isinstance(s, Symbol) for s in nz):
        return Symbol("Indeterminate")
    if all(s == 1 for s in nz):
        return Symbol("Indeterminate") if strict and 0 in signs else 1
    if all(s == -1 for s in nz):
        return Symbol("Indeterminate") if strict and 0 in signs else -1
    return Symbol("Indeterminate")


@fn("FunctionMonotonicity", area=AREA, status="added",
    notes="FunctionMonotonicity[f, x] / [{f, cond}, x]: 1 non-decreasing, -1 non-increasing, 0 constant, "
          "Indeterminate otherwise (or if f is not real on the domain); StrictInequalities -> True for strict "
          "monotonicity. Univariate elementary functions continuous on an interval domain (derivative sign, with "
          "Abs split at its zeros); undecidable cases raise")
def _fmono(*args):
    f, cond, vs, dom, opts = _prop_args(args, "FunctionMonotonicity", extra_opts=("StrictInequalities",))
    if dom != "Reals":
        raise NotImplementedError("FunctionMonotonicity: only the Reals domain is supported")
    return _monotonicity(f, cond, vs, opts, "FunctionMonotonicity")


@fn("FunctionConvexity", area=AREA, status="added",
    notes="FunctionConvexity[f, x] / [{f, cond}, x]: 1 convex, -1 concave, 0 both (affine), Indeterminate "
          "otherwise; StrictInequalities -> True for strict convexity. Univariate: sign of f'' on the smooth "
          "pieces plus the jump of f' at Abs kinks; multivariate: polynomials of degree <= 2 (constant Hessian, "
          "eigenvalue signs) and f = g(linear form). Undecidable cases raise")
def _fconv(*args):
    fname = "FunctionConvexity"
    f, cond, vs, dom, opts = _prop_args(args, fname, extra_opts=("StrictInequalities",))
    if dom != "Reals":
        raise NotImplementedError(f"{fname}: only the Reals domain is supported")
    strict = _strict_opt(opts, fname)
    if len(vs) > 1:
        if cond is not True:
            raise NotImplementedError(f"{fname}: conditions are only supported for univariate functions")
        X = [sp.Symbol(v.name, real=True) for v in vs]
        fs = _sp_real(f, vs, fname)
        if fs.free_symbols - set(X):
            raise NotImplementedError(f"{fname}: the function contains parameters")
        poly = fs.as_poly(*X) if fs.is_polynomial(*X) else None
        if poly is not None and poly.total_degree() <= 2:
            H = sp.hessian(fs, X)
            ev_ = [sp.nsimplify(e) for e in H.eigenvals(multiple=True)]
            if all(e == 0 for e in ev_):
                return 0
            if all(e >= 0 for e in ev_):
                return Symbol("Indeterminate") if strict and any(e == 0 for e in ev_) else 1
            if all(e <= 0 for e in ev_):
                return Symbol("Indeterminate") if strict and any(e == 0 for e in ev_) else -1
            return Symbol("Indeterminate")
        # g(linear form): convexity of g along a line, Hessian rank 1 (never strict)
        t = sp.Dummy("t", real=True)
        lin = None
        for node in sp.preorder_traversal(fs):
            if node.is_Add and node.is_polynomial(*X) and sp.Poly(node, *X).total_degree() == 1:
                lin = node
                break
        if lin is not None:
            g = sp.simplify(fs.subs(X[0], sp.solve(sp.Eq(lin, t), X[0])[0]))
            if not g.free_symbols - {t}:
                r = _monotonicity_conv1(g, t, sp.S.Reals, fname, False)
                if strict and r != Symbol("Indeterminate"):
                    return Symbol("Indeterminate")
                return r
        raise NotImplementedError(f"{fname}: cannot decide the convexity of this multivariate function")
    x, fs, D = _real_setup(f, cond, vs, fname)
    r = _require_real_continuous(f, vs, D, x, fs, fname)
    if r is not None:
        return r
    _interval_dom(D, fname)
    return _monotonicity_conv1(fs, x, D, fname, strict)


def _monotonicity_conv1(fs, x, D, fname, strict):
    pieces = _split_abs(fs, x, D, fname)
    signs = []
    for S, g in pieces:
        if isinstance(S, sp.FiniteSet):
            continue
        d1 = sp.diff(g, x)
        if not _is_empty(_real_set(sp.Eq(sp.denom(sp.together(d1)), 0), x, fname).intersect(S), fname):
            raise NotImplementedError(f"{fname}: f' is singular inside the domain")
        d2 = sp.diff(g, x, 2)
        sg = _sign_on(d2, x, S, fname)
        if strict and sg in (1, -1) and not _thin_zeros(d2, x, S, fname):
            sg = 0
        signs.append(sg)
    # kinks: compare one-sided derivatives at the cut points
    for S, _ in pieces:
        if isinstance(S, sp.FiniteSet):
            for c in S:
                dl = sp.limit(sp.diff(fs, x), x, c, "-")
                dr = sp.limit(sp.diff(fs, x), x, c, "+")
                r = _num(dr - dl)
                if r is None:
                    raise NotImplementedError(f"{fname}: cannot compare the one-sided derivatives at {c}")
                if r[1] != 0:
                    signs.append(r[1])
    if not signs or all(s == 0 for s in signs):
        return Symbol("Indeterminate") if strict else 0
    if any(isinstance(s, Symbol) for s in signs):
        return Symbol("Indeterminate")
    nz = {s for s in signs if s != 0}
    if nz == {1}:
        return Symbol("Indeterminate") if strict and 0 in signs else 1
    if nz == {-1}:
        return Symbol("Indeterminate") if strict and 0 in signs else -1
    return Symbol("Indeterminate")


def _cplx_cond_nonempty(c, x, fname):
    """Decide whether the complex condition c (SymPy, variable x complex) is satisfiable, for the forms produced
    by _bad with no extra condition: returns True/False or raises."""
    if c is sp.false:
        return False
    if c is sp.true:
        return True
    for clause in (c.args if isinstance(c, sp.Or) else [c]):
        atoms = list(clause.args) if isinstance(clause, sp.And) else [clause]
        eqs = [a for a in atoms if isinstance(a, sp.Eq)]
        # an equation g(x) == 0 with g a nonconstant polynomial, or an entire periodic function, has zeros
        for e in eqs:
            g = e.lhs - e.rhs
            if g.has(sp.im) or g.has(sp.re):
                continue
            sols = sp.solveset(g, x, sp.S.Complexes)
            if isinstance(sols, sp.ConditionSet) or sols.has(sp.ConditionSet):
                raise NotImplementedError(f"{fname}: cannot solve {g} == 0")
            cand = list(sols) if isinstance(sols, sp.FiniteSet) else None
            if cand is None:
                # periodic families: try the representatives n = -3..3
                cand = []
                for part in (sols.args if isinstance(sols, sp.Union) else [sols]):
                    if isinstance(part, sp.ImageSet):
                        lam = part.lamda
                        cand += [lam(k) for k in range(-3, 4)]
                    elif isinstance(part, sp.FiniteSet):
                        cand += list(part)
            for p in cand:
                ok = True
                for a in atoms:
                    val = a.subs(x, p)
                    val = sp.simplify(val) if val not in (sp.true, sp.false) else val
                    if val is not sp.true:
                        ok = False
                        break
                if ok:
                    return True
        if not eqs or all((e.lhs - e.rhs).has(sp.im, sp.re) for e in eqs):
            # Im[g] == 0 && Re[g] <= 0 with g nonconstant: g takes negative real values (Picard), so satisfiable
            gs = {a.args[0] for a in clause.atoms(sp.im, sp.re)}
            if len(gs) == 1:
                g = gs.pop()
                if g.has(x) and (g.is_polynomial(x) or g.func in (sp.exp, sp.sin, sp.cos)):
                    return True
    raise NotImplementedError(f"{fname}: cannot decide this over the complex numbers")


def _cplx_with_cond(bad, cond, x, fname):
    """bad && cond satisfiable?  Only for conditions on Re/Im of x and bad parts whose Re/Im arguments are x."""
    u, v = sp.symbols("u v", real=True)
    z = u + sp.I * v

    def conv(c):
        c = c.xreplace({x: z})
        reps = {}
        for a in c.atoms(sp.Eq, sp.Ne):
            d = sp.expand_complex(a.lhs - a.rhs)
            re_, im_ = d.as_real_imag()
            if im_ != 0 and re_ != 0:
                both = sp.And(sp.Eq(re_, 0), sp.Eq(im_, 0))
                reps[a] = both if isinstance(a, sp.Eq) else sp.Not(both)
        return sp.simplify(c.xreplace(reps))
    whole = sp.And(bad, cond)
    clauses = sp.to_dnf(whole, simplify=False)
    for clause in (clauses.args if isinstance(clauses, sp.Or) else [clauses]):
        cz = conv(clause)
        if cz is sp.false:
            continue
        atoms = list(cz.args) if isinstance(cz, sp.And) else [cz]
        vfix = [a for a in atoms if isinstance(a, sp.Eq) and a.free_symbols == {v} and (a.lhs - a.rhs).is_polynomial(v)]
        if vfix:
            vs_ = sp.solve(vfix[0], v)
            for vv in vs_:
                rest = sp.And(*[a.subs(v, vv) for a in atoms])
                if rest is sp.false:
                    continue
                if not rest.free_symbols <= {u}:
                    raise NotImplementedError(f"{fname}: cannot decide this complex condition")
                if not _is_empty(_real_set(rest, u, fname), fname):
                    return True
            continue
        raise NotImplementedError(f"{fname}: cannot decide this complex condition")
    return False


def _property_decide(f, cond, vs, dom, kind, fname):
    """True iff f has no 'kind' bad points in the domain."""
    bad = _bad(ev(f), vs, kind, dom, fname, True)
    if bad is False:
        return True
    if dom == "Reals":
        if len(vs) != 1:
            raise NotImplementedError(f"{fname}: multivariate conditions over the reals are not supported")
        x = sp.Symbol(vs[0].name, real=True)
        D = _real_set(_sp_real(cond, vs, fname), x, fname)
        B = sp.Intersection(D, _real_set(_sp_real(bad, vs, fname), x, fname))
        return _is_empty(B, fname)
    if len(vs) != 1 or not isinstance(vs[0], Symbol):
        raise NotImplementedError(f"{fname}: only univariate functions are supported over the complex numbers")
    x = sp.Symbol(vs[0].name)
    bs = tosp(bad, fname)
    if cond is True:
        return not _cplx_cond_nonempty(bs, x, fname)
    return not _cplx_with_cond(bs, tosp(cond, fname), x, fname)


def _entire_like(e, vs):
    """Polynomials and compositions of entire functions (continuous and analytic on R^n and C^n)."""
    if _free_of(e, vs):
        return True
    if any(K.same(e, v) for v in vs):
        return True
    if not isinstance(e, Expr) or not isinstance(e.head, str):
        return False
    if e.head in ("Plus", "Times"):
        return all(_entire_like(a, vs) for a in e.args)
    if e.head == "Power" and len(e.args) == 2:
        b, n = e.args
        if K.is_int(n) and int(n) >= 0:
            return _entire_like(b, vs)
        if K.same(b, Symbol("E")):
            return _entire_like(n, vs)
        return False
    return e.head in _ENTIRE1 and len(e.args) == 1 and _entire_like(e.args[0], vs)


def _prop_bool(kind, fname, default_dom):
    def impl(*args):
        f, cond, vs, dom, _ = _prop_args(args, fname, default_dom)
        f = ev(f)
        if len(vs) > 1:
            if _entire_like(f, vs):
                return True
            raise NotImplementedError(f"{fname}: multivariate functions other than polynomials and compositions of "
                                      f"entire functions are not supported")
        return _property_decide(f, cond, vs, dom, kind, fname)
    return impl


fn("FunctionContinuous", area=AREA, status="added",
   notes="FunctionContinuous[f, x] / [{f, cond}, x] / [f, x, Complexes]: True if f is real-valued (over Reals) and "
         "continuous on the whole domain, decided from the structural discontinuity condition of whitelisted "
         "elementary heads (as FunctionDiscontinuities) intersected with the domain; False only when a bad point "
         "provably exists; multivariate: polynomials / entire compositions. Undecidable cases raise")(
    _prop_bool("disc", "FunctionContinuous", "Reals"))
fn("FunctionAnalytic", area=AREA, status="added",
   notes="FunctionAnalytic[f, x] / [{f, cond}, x] / [f, x, Complexes]: True if f is analytic on the domain (no "
         "point of FunctionSingularities in it); same decision procedure as FunctionContinuous")(
    _prop_bool("sing", "FunctionAnalytic", "Reals"))
fn("FunctionMeromorphic", area=AREA, status="added",
   notes="FunctionMeromorphic[f, x] / [{f, cond}, x]: True if f is meromorphic on the (complex) domain: poles from "
         "rational parts, Tan/Sec/Cot/Csc, Gamma are allowed; branch cuts of Log and fractional powers are not; "
         "Exp/Sin/... of arguments with poles (essential singularities) and unsupported heads raise")(
    _prop_bool("mero", "FunctionMeromorphic", "Complexes"))


def _ess_check(e, vs, fname):
    """Raise if an entire function is applied to an argument with poles (essential singularity)."""
    if isinstance(e, Expr) and isinstance(e.head, str):
        if (e.head in _ENTIRE1 or (e.head == "Power" and K.same(e.args[0], Symbol("E")))) and \
                not _entire_like(e.args[-1], vs):
            raise NotImplementedError(f"{fname}: essential singularities are not supported")
        for a in e.args:
            _ess_check(a, vs, fname)


_orig_mero = None


@fn("FunctionDiscontinuities", area=AREA, status="added",
    notes="FunctionDiscontinuities[f, x] / [{f, cond}, x] / [f, x, Complexes]: a condition describing where f is "
          "not continuous (over Reals also where it is not real-valued), built structurally from whitelisted "
          "heads as Wolfram does: Tan[x] -> Cos[x] == 0, Sqrt[x] -> x <= 0, Floor[g] -> Sin[Pi g] == 0, Gamma[x] "
          "-> Sin[Pi x] == 0 && x <= 0, Log[x] over Complexes -> x == 0 || (Im[x] == 0 && Re[x] <= 0); False if "
          "there are none; unsupported heads raise")
def _fdisc(*args):
    f, cond, vs, dom, _ = _prop_args(args, "FunctionDiscontinuities")
    c = _bad(ev(f), vs, "disc", dom, "FunctionDiscontinuities")
    return c if cond is True or c is False else ev(Expr("And", cond, c))


@fn("FunctionSingularities", area=AREA, status="added",
    notes="FunctionSingularities[f, x] / [{f, cond}, x] / [f, x, Complexes]: a condition describing where f is not "
          "analytic (as FunctionDiscontinuities, plus zeros of Abs arguments, boundary points of Sqrt/ArcSin "
          "domains: Abs[Tan[2 Sin[x]]] -> Cos[2 Sin[x]] == 0 || Tan[2 Sin[x]] == 0)")
def _fsing(*args):
    f, cond, vs, dom, _ = _prop_args(args, "FunctionSingularities")
    c = _bad(ev(f), vs, "sing", dom, "FunctionSingularities")
    return c if cond is True or c is False else ev(Expr("And", cond, c))


def _natural_domain(f, vs, cond, fname):
    x, fs, D = _real_setup(f, cond, vs, fname)
    B = _disc_set(f, vs, D, x, fname)
    Dn = sp.Complement(D, B) if not _is_empty(B, fname) else D
    Dn = sp.simplify(Dn) if not isinstance(Dn, sp.Set) else Dn
    return x, fs, Dn


def _split_map_args(args, fname):
    """{f, xcond, ycond}, x, y [, dom] -> (f, xcond, ycond, vars, yvars, dom)."""
    args, opts = _opts(args, ("Assumptions", "GenerateConditions", "PerformanceGoal"), fname)
    if opts.get("Assumptions") not in (None, True):
        raise NotImplementedError(f"{fname}: the Assumptions option is not supported")
    if not args:
        raise ValueError(f"{fname} expects a function and variables")
    f, rest = args[0], list(args[1:])
    dom = "Reals"
    if rest and isinstance(rest[-1], Symbol) and rest[-1].name in ("Reals", "Complexes", "Integers"):
        dom = rest.pop().name
    if dom == "Integers":
        raise NotImplementedError(f"{fname}: the Integers domain is not supported")
    if len(rest) not in (1, 2):
        raise ValueError(f"{fname}: expected f, vars [, yvars] [, dom]")
    vs = _vars(rest[0], fname)
    ys = _vars(rest[1], fname) if len(rest) == 2 else None
    xc = yc = True
    if is_list(f):
        if len(f.args) == 2 and ys is None:
            f, xc = f.args
        elif len(f.args) == 3:
            f, xc, yc = f.args
            if ys is None:
                raise ValueError(f"{fname}: a codomain condition needs the codomain variable")
        elif len(f.args) == 2 and ys is not None:
            f, xc = f.args
        else:
            raise NotImplementedError(f"{fname}: vector-valued maps are not supported")
    if len(vs) != 1 or (ys is not None and len(ys) != 1):
        raise NotImplementedError(f"{fname}: only maps of one variable are supported")
    return ev(f), xc, yc, vs, ys, dom


def _subs1(e, v, val):
    return ev(Expr("ReplaceAll", e, rule(v, val)))


def _cplx_kind(f, x):
    """Classify a univariate complex function for injectivity/surjectivity: ('poly', degree) / ('exp',) /
    ('sincos',) / ('log',) / ('tan',) or None.  Arguments must be linear in x."""
    fs = f
    if fs.is_polynomial(x):
        return ("poly", sp.Poly(fs, x).degree())
    terms = sp.Add.make_args(fs)
    nonconst = [t for t in terms if t.has(x)]
    if len(nonconst) != 1:
        return None
    c, g = nonconst[0].as_independent(x, as_Add=False)
    if c == 0:
        return None
    kinds = {sp.exp: ("exp",), sp.sin: ("sincos",), sp.cos: ("sincos",), sp.sinh: ("sincos",),
             sp.cosh: ("sincos",), sp.log: ("log",), sp.tan: ("tan",)}
    if g.func in kinds and len(g.args) == 1 and g.args[0].is_polynomial(x) and sp.Poly(g.args[0], x).degree() == 1:
        return kinds[g.func]
    return None


def _injective(f, xc, yc, vs, ys, dom, fname):
    if dom == "Complexes":
        if xc is not True or yc is not True:
            raise NotImplementedError(f"{fname}: conditions are not supported over the complex numbers")
        x = sp.Symbol(vs[0].name)
        fs = tosp(f, fname)
        if fs.free_symbols - {x}:
            raise NotImplementedError(f"{fname}: the function contains parameters")
        k = _cplx_kind(fs, x)
        if k is None:
            raise NotImplementedError(f"{fname}: cannot decide injectivity of {fs} over the complex numbers")
        if k[0] == "poly":
            return k[1] == 1
        return k[0] == "log"
    cond = xc
    if yc is not True:
        cond = Expr("And", xc, _subs1(yc, ys[0], f))
    x, fs, Dn = _natural_domain(f, vs, cond, fname)
    if fs.free_symbols - {x}:
        raise NotImplementedError(f"{fname}: the function contains parameters")
    if _is_empty(Dn, fname):
        return True
    if not (isinstance(Dn, sp.Interval) or Dn == sp.S.Reals):
        raise NotImplementedError(f"{fname}: the natural domain {Dn} is not an interval")
    signs = []
    for S, d in _deriv_pieces(fs, x, Dn, fname):
        sg = _sign_on(d, x, S, fname)
        if sg in (1, -1) and not _thin_zeros(d, x, S, fname):
            sg = 0
        signs.append(sg)
    nz = {s for s in signs if not isinstance(s, Symbol)}
    if signs and all(not isinstance(s, Symbol) for s in signs) and nz in ({1}, {-1}):
        return True
    return False            # continuous on an interval and not strictly monotone -> not injective


def _piece_onto(fs, x, Dn):
    """IVT witness: an open interval (a, b) inside Dn between excluded points on which fs is continuous and tends
    to -oo and +oo at the two ends (so fs takes every real value)."""
    if not (isinstance(Dn, sp.Complement) and (Dn.args[0] == sp.S.Reals or isinstance(Dn.args[0], sp.Interval))):
        return False
    pts = []
    excl = Dn.args[1]
    for q in (excl.args if isinstance(excl, sp.Union) else [excl]):
        if isinstance(q, sp.FiniteSet):
            pts += [p for p in q if p.is_real]
        elif isinstance(q, sp.ImageSet) and q.base_sets == (sp.S.Integers,):
            n = q.lamda.variables[0]
            e = sp.expand(q.lamda.expr)
            c, d = e.coeff(n), e.subs(n, 0)
            if not (c.is_number and d.is_number and e == c * n + d and c != 0):
                return False
            c, d = abs(c), d
            k0, k1 = int(sp.floor((-20 - d) / c)) - 1, int(sp.ceiling((20 - d) / c)) + 1
            if k1 - k0 > 400:
                return False
            pts += [c * k + d for k in range(k0, k1 + 1)]
        else:
            return False
    pts = sorted({sp.nsimplify(p) for p in pts if abs(float(p)) <= 20}, key=float)
    for a, b in zip(pts, pts[1:]):
        if Dn.args[0].contains((a + b) / 2) is not sp.true:
            continue
        lo, hi = sp.limit(fs, x, a, "+"), sp.limit(fs, x, b, "-")
        if {lo, hi} == {sp.oo, -sp.oo}:
            return True
    return False


def _surjective(f, xc, yc, vs, ys, dom, fname):
    from sympy.calculus.util import function_range
    if dom == "Complexes":
        if xc is not True or yc is not True:
            raise NotImplementedError(f"{fname}: conditions are not supported over the complex numbers")
        x = sp.Symbol(vs[0].name)
        fs = tosp(f, fname)
        if fs.free_symbols - {x}:
            raise NotImplementedError(f"{fname}: the function contains parameters")
        k = _cplx_kind(fs, x)
        if k is None:
            raise NotImplementedError(f"{fname}: cannot decide surjectivity of {fs} over the complex numbers")
        if k[0] == "poly":
            return k[1] >= 1
        return k[0] == "sincos"
    x, fs, Dn = _natural_domain(f, vs, xc, fname)
    if fs.free_symbols - {x}:
        raise NotImplementedError(f"{fname}: the function contains parameters")
    try:
        rng = function_range(fs, x, Dn)
    except Exception as exc:  # noqa: BLE001
        if yc is True and _piece_onto(fs, x, Dn):
            return True
        raise NotImplementedError(f"{fname}: cannot compute the range ({exc})")
    if rng.has(sp.ConditionSet) or not isinstance(rng, (sp.Interval, sp.Union, sp.FiniteSet)) and rng != sp.S.Reals:
        raise NotImplementedError(f"{fname}: cannot compute the range")
    if yc is True:
        Y = sp.S.Reals
    else:
        y = sp.Symbol(ys[0].name, real=True)
        Y = _real_set(_sp_real(yc, ys, fname), y, fname)
    return _is_empty(sp.Complement(Y, rng), fname)


def _map_prop(which):
    fname = {"inj": "FunctionInjective", "sur": "FunctionSurjective", "bij": "FunctionBijective"}[which]

    def impl(*args):
        f, xc, yc, vs, ys, dom = _split_map_args(args, fname)
        if which == "inj":
            return _injective(f, xc, yc, vs, ys, dom, fname)
        if which == "sur":
            return _surjective(f, xc, yc, vs, ys, dom, fname)
        return _injective(f, xc, yc, vs, ys, dom, fname) and _surjective(f, xc, yc, vs, ys, dom, fname)
    return impl


fn("FunctionInjective", area=AREA, status="added",
   notes="FunctionInjective[f, x] / [{f, xcond}, x] / [{f, xcond, ycond}, x, y] / [f, x, Complexes]: over Reals "
         "f is continuous on its natural domain (which must be an interval), so it is injective iff strictly "
         "monotone (derivative sign proofs); over Complexes polynomials (degree 1), Exp/Sin/Cos/Tan of linear "
         "arguments (False) and Log (True). Parameters, vector maps and other cases raise")(_map_prop("inj"))
fn("FunctionSurjective", area=AREA, status="added",
   notes="FunctionSurjective[f, x] / [{f, xcond}, x] / [{f, xcond, ycond}, x, y] / [f, x, Complexes]: over Reals "
         "the range on the natural domain (SymPy function_range) must contain the codomain (all reals, or the "
         "ycond set); over Complexes polynomials of degree >= 1 and Sin/Cos are onto, Exp/Log/Tan are not. "
         "Parameters and vector maps raise")(_map_prop("sur"))
fn("FunctionBijective", area=AREA, status="added",
   notes="FunctionBijective: FunctionInjective && FunctionSurjective with the same arguments")(_map_prop("bij"))


def _pole_sources(e, vs, fname):
    """List of ('pow', g, k) and ('trig', head, g) pole sources; raises for unsupported structure."""
    out_ = []
    if _free_of(e, vs) or any(K.same(e, v) for v in vs):
        return out_
    if not isinstance(e, Expr) or not isinstance(e.head, str):
        raise NotImplementedError(f"{fname}: cannot analyse {e!r}")
    h, a = e.head, e.args
    if h in ("Plus", "Times"):
        for t in a:
            out_ += _pole_sources(t, vs, fname)
        return out_
    if h == "Power" and len(a) == 2:
        b, n = a
        if K.same(b, Symbol("E")) and _entire_like(n, vs):
            return out_
        if K.is_int(n):
            out_ += _pole_sources(b, vs, fname)
            if int(n) < 0:
                if _is_head(b, "Sin", "Cos") and _entire_like(b.args[0], vs):
                    out_.append(("trig", "Csc" if b.head == "Sin" else "Sec", b.args[0]))
                else:
                    out_.append(("pow", b, -int(n)))
            return out_
        raise NotImplementedError(f"{fname}: cannot analyse the power {e!r}")
    if len(a) == 1 and h in _ENTIRE1 and _entire_like(a[0], vs):
        return out_
    if len(a) == 1 and h in ("Tan", "Sec", "Cot", "Csc") and _entire_like(a[0], vs):
        return out_ + [("trig", h, a[0])]
    raise NotImplementedError(f"{fname}: cannot analyse {h}")


@fn("FunctionPoles", area=AREA, status="added",
    notes="FunctionPoles[f, x] / [{f, cond}, x]: list of {pole, multiplicity} over the complex numbers for "
          "functions built from polynomials, entire functions and Tan/Sec/Cot/Csc of polynomial arguments: "
          "explicit poles from denominators (multiplicity from the Laurent leading term, removable points dropped) "
          "and, without a condition, periodic families {ConditionalExpression[..., C[1] in Integers], k} for "
          "c Tan[g]^k-type terms; functions with branch points stay unevaluated (as in Wolfram); others raise")
def _fpoles(*args):
    fname = "FunctionPoles"
    f, cond, vs, dom, _ = _prop_args(args, fname, "Complexes")
    if len(vs) != 1 or not isinstance(vs[0], Symbol):
        raise NotImplementedError(f"{fname}: only univariate functions are supported")
    f = ev(f)
    try:
        srcs = _pole_sources(f, vs, fname)
    except NotImplementedError:
        try:
            mero = _property_decide(f, cond, vs, "Complexes", "mero", fname)
        except NotImplementedError:
            mero = None
        if mero is False:
            raise Unevaluated()
        if mero is True and cond is not True:
            srcs = None
        else:
            raise
    x = sp.Symbol(vs[0].name)
    xv = vs[0]
    if srcs is None:
        # meromorphic on the restricted domain: strip the branch parts (they are analytic there) and use the rest
        def strip(e):
            if _is_head(e, "Plus"):
                return Expr("Plus", *(t for t in e.args if _try_sources(t, vs) is not None))
            return e
        g = ev(strip(f))
        srcs = _try_sources(g, vs)
        if srcs is None:
            raise NotImplementedError(f"{fname}: cannot isolate the poles")
    fs = tosp(f, fname)
    if fs.free_symbols - {x}:
        raise NotImplementedError(f"{fname}: the function contains parameters")
    t = sp.Dummy("t")
    res, seen = [], []

    def order_at(p):
        lt = sp.simplify(fs.subs(x, p + t)).as_leading_term(t)
        c, e = lt.as_coeff_exponent(t)
        e = sp.sympify(e)
        if not e.is_integer:
            raise NotImplementedError(f"{fname}: non-integer Laurent exponent at {p}")
        return int(-e)
    families = []
    for s in srcs:
        if s[0] == "pow":
            sols = ev(Expr("Solve", Expr("Equal", s[1], 0), xv))
            for r in sols.args:
                p = r.args[0].args[1]
                if _is_head(p, "ConditionalExpression"):
                    raise NotImplementedError(f"{fname}: infinitely many candidate poles from {s[1]!r}")
                ps = tosp(p, fname)
                if any(sp.simplify(ps - q) == 0 for q in seen):
                    continue
                seen.append(ps)
                if cond is not True:
                    cv = ev(_subs1(cond, xv, p))
                    if cv is not True:
                        if cv is False:
                            continue
                        raise NotImplementedError(f"{fname}: cannot decide the condition at {p!r}")
                m = order_at(ps)
                if m > 0:
                    res.append(L(p, m))
        else:
            families.append(s)
    if families:
        if cond is not True:
            raise NotImplementedError(f"{fname}: periodic pole families with a domain condition are not supported")
        done = []
        for _, h, g in families:
            if not _is_poly_deg1(g, xv):
                raise NotImplementedError(f"{fname}: {h} of a non-linear argument")
            zero_of = Expr("Cos" if h in ("Tan", "Sec") else "Sin", g)
            if any(K.same(zero_of, d) for d in done):
                continue
            done.append(zero_of)
            zs = tosp(zero_of, fname)
            # structural bound: f * zero^k has no remaining poles from this family
            kmax = None
            for k in range(0, 9):
                q = sp.simplify(fs * zs ** k)
                if q.has(sp.tan, sp.sec, sp.cot, sp.csc) or sp.denom(sp.together(q)).has(x):
                    continue
                kmax = k
                break
            if kmax is None:
                raise NotImplementedError(f"{fname}: cannot isolate the poles of {h}")
            sols = ev(Expr("Solve", Expr("Equal", zero_of, 0), xv))
            for r in sols.args:
                p = r.args[0].args[1]
                if not _is_head(p, "ConditionalExpression"):
                    raise NotImplementedError(f"{fname}: unexpected solution form")
                ords = {order_at(tosp(_subs1(p.args[0], Expr("C", 1), c), fname)) for c in (-1, 0, 1)}
                if len(ords) != 1:
                    raise NotImplementedError(f"{fname}: pole orders differ along the family")
                m = ords.pop()
                if m > kmax:
                    raise NotImplementedError(f"{fname}: inconsistent pole order")
                if m > 0:
                    res.append(L(p, m))
    return L(*res)


def _try_sources(e, vs):
    try:
        return _pole_sources(e, vs, "FunctionPoles")
    except NotImplementedError:
        return None


def _is_poly_deg1(g, x):
    s = sym(g, "FunctionPoles")
    xs = sp.Symbol(x.name)
    return s.is_polynomial(xs) and sp.Poly(s, xs).degree() == 1


# ───────────────────────────── PiecewiseExpand ─────────────────────────────

def _pw(cases, default=0):
    return Expr("Piecewise", L(*(L(v, c) for v, c in cases)), default)


def _pw_pre(e, reals, D, xvar, fname):
    """Rewrite piecewise-defined heads as Piecewise (bottom-up)."""
    if not isinstance(e, Expr):
        return e
    if not isinstance(e.head, str):
        return e
    h = e.head
    if h == "If" and len(e.args) in (2, 3):
        c, a = _pw_pre(e.args[0], reals, D, xvar, fname), _pw_pre(e.args[1], reals, D, xvar, fname)
        b = _pw_pre(e.args[2], reals, D, xvar, fname) if len(e.args) == 3 else Symbol("Null")
        return _pw([(a, c)], b)
    a = [_pw_pre(t, reals, D, xvar, fname) for t in e.args]
    if h == "UnitStep":
        return _pw([(1, Expr("And", *(Expr("GreaterEqual", t, 0) for t in a)))], 0)
    if h == "Boole" and len(a) == 1:
        return _pw([(1, a[0])], 0)
    if h == "Ramp" and len(a) == 1:
        return _pw([(a[0], Expr("GreaterEqual", a[0], 0))], 0)
    if h == "KroneckerDelta":
        return _pw([(1, Expr("Equal", *a))], 0) if len(a) > 1 else _pw([(1, Expr("Equal", a[0], 0))], 0)
    if h == "Clip" and len(a) == 1:
        return _pw([(-1, Expr("Less", a[0], -1)), (1, Expr("Greater", a[0], 1))], a[0])
    if h in ("Max", "Min") and len(a) >= 1:
        flat = []
        for t in a:
            flat += list(t.args) if is_list(t) else [t]
        if len(flat) == 1:
            return flat[0]
        op = "GreaterEqual" if h == "Max" else "LessEqual"
        cases = []
        for i, t in enumerate(flat[:-1]):
            cases.append((t, Expr("And", *(Expr(op, Expr("Plus", t, Expr("Times", -1, u)), 0)
                                           for u in flat[i + 1:]))))
        return _pw(cases, flat[-1])
    if reals and h == "Abs" and len(a) == 1:
        return _pw([(Expr("Times", -1, a[0]), Expr("Less", a[0], 0))], a[0])
    if reals and h == "Sign" and len(a) == 1:
        return _pw([(1, Expr("Greater", a[0], 0)), (-1, Expr("Less", a[0], 0))], 0)
    if h in ("Floor", "Ceiling", "Round") and len(a) == 1 and D is not None:
        r = _pw_int_parts(h, a[0], D, xvar, fname)
        if r is not None:
            return r
    return Expr(h, *a)


def _pw_int_parts(h, g, D, xvar, fname):
    from sympy.calculus.util import function_range
    try:
        gs = _sp_real(g, [xvar], fname)
        x = sp.Symbol(xvar.name, real=True)
        if gs.free_symbols - {x}:
            return None
        rng = function_range(gs, x, D)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(rng, sp.Interval) or not rng.start.is_finite or not rng.end.is_finite:
        return None
    lo, hi = int(sp.floor(rng.start)) - 1, int(sp.ceiling(rng.end)) + 1
    maxc = _SYSC.get("$MaxPiecewiseCases", 100)
    if hi - lo > maxc:
        return None
    cases = []
    for n in range(lo, hi + 1):
        half = sp.Rational(1, 2)
        if h == "Floor":
            r1, r2 = gs >= n, gs < n + 1
        elif h == "Ceiling":
            r1, r2 = gs > n - 1, gs <= n
        elif n % 2 == 0:
            r1, r2 = gs >= n - half, gs <= n + half
        else:
            r1, r2 = gs > n - half, gs < n + half
        S = sp.Intersection(sp.solveset(r1, x, D), sp.solveset(r2, x, D))
        if isinstance(S, sp.ConditionSet):
            return None
        if S.is_empty:
            continue
        cases.append((n, fromsp(S.as_relational(x))))
    return _pw(cases, 0)


@fn("PiecewiseExpand", area=AREA, status="added",
    notes="PiecewiseExpand[expr] / [expr, cond] / [expr, Reals]: nested Piecewise, If, Max, Min, UnitStep, Boole, "
          "Ramp, Clip, KroneckerDelta (and Abs, Sign over Reals) become one Piecewise (SymPy piecewise_fold plus "
          "condition simplification; cases are equivalent to Wolfram's, conditions may be written differently); "
          "Floor/Ceiling/Round of a univariate expression are expanded when the condition bounds its range (at "
          "most $MaxPiecewiseCases cases); inequalities with piecewise sides are expanded to conditions")
def _pwexpand(e, *rest):
    fname = "PiecewiseExpand"
    rest, opts = _opts(rest, ("Assumptions", "Method"), fname)
    reals = False
    cond = True
    for r in rest:
        if isinstance(r, Symbol) and r.name in ("Reals", "Complexes"):
            reals = r.name == "Reals"
        else:
            cond = r if cond is True else Expr("And", cond, r)
    asm = opts.get("Assumptions")
    if asm is not None and asm is not True:
        cond = asm if cond is True else Expr("And", cond, asm)
    D, xvar = None, None
    if cond is not True:
        fv = [v for v in ev(Expr("Variables", Expr("List", cond))).args] if False else None
        syms = sorted({s.name for s in _atoms_sym(cond)})
        if len(syms) == 1:
            xvar = Symbol(syms[0])
            try:
                D = _real_set(_sp_real(cond, [xvar], fname), sp.Symbol(syms[0], real=True), fname)
            except NotImplementedError:
                D = None
    pre = _pw_pre(e, reals, D, xvar, fname)
    if not _contains(pre, lambda s: _is_head(s, "Piecewise")):
        return pre
    s = tosp(pre, fname)
    s = s.xreplace({v: sp.Symbol(v.name, real=True) for v in s.free_symbols})
    from sympy.functions.elementary.piecewise import piecewise_fold
    if isinstance(s, sp.logic.boolalg.Boolean) and not isinstance(s, sp.Piecewise):
        s = _fold_rel(s)
        try:
            s = sp.simplify_logic(s) if len(s.atoms(sp.core.relational.Relational)) <= 6 else s
        except Exception:  # noqa: BLE001
            pass
        return ev(fromsp(s.xreplace({v: sp.Symbol(v.name) for v in s.free_symbols})))
    s = piecewise_fold(s)      # (SymPy's piecewise_simplify is not used: it can merge cases incorrectly)
    if isinstance(s, sp.Piecewise) and len(s.args) - 1 > _SYSC.get("$MaxPiecewiseCases", 100):
        raise ValueError(f"{fname}: more than $MaxPiecewiseCases cases")
    s = s.xreplace({v: sp.Symbol(v.name) for v in s.free_symbols})
    return ev(fromsp(s))


def _fold_rel(b):
    """Relations whose sides are Piecewise -> Or of (case condition && relation of the case values)."""
    from sympy.functions.elementary.piecewise import piecewise_fold
    if isinstance(b, sp.core.relational.Relational):
        d = piecewise_fold(b.lhs - b.rhs)
        if isinstance(d, sp.Piecewise):
            terms, prev = [], sp.true
            for val, c in d.args:
                terms.append(sp.And(prev, c, type(b)(val, 0)))
                prev = sp.And(prev, sp.Not(c))
            return sp.Or(*terms)
        return b
    if isinstance(b, (sp.And, sp.Or, sp.Not)):
        return type(b)(*(_fold_rel(a) for a in b.args))
    return b


def _atoms_sym(e):
    out_ = []
    if isinstance(e, Symbol):
        if e.name not in ("True", "False", "Pi", "E", "I", "Infinity"):
            out_.append(e)
    elif isinstance(e, Expr):
        for a in e.args:
            out_ += _atoms_sym(a)
    return out_


# ───────────────────────────── FindInstance ─────────────────────────────

def _cands(kind):
    """Candidate values for free variables, simplest first."""
    base = [0, 1, -1, 2, -2, 3, -3, Expr("Rational", 1, 2), Expr("Rational", -1, 2), 4, -4, 5, -5, 7, 10]
    return base


def _holds(expr, assign):
    """True / False / None: does expr hold at the assignment (exact evaluation, then 50-digit numerics)?"""
    sub = ev(Expr("ReplaceAll", expr, L(*(rule(v, val) for v, val in assign))))
    if sub is True or sub is False:
        return sub
    num = ev(Expr("N", sub, 50))
    if num is True or num is False:
        return num
    return None


def _is_real_val(v):
    r = ev(Expr("Element", v, Symbol("Reals")))
    if r is True or r is False:
        return r
    n = ev(Expr("N", Expr("Im", v), 50))
    if K.is_number(n):
        return abs(complex(sp.N(sym(n, "FindInstance")))) < 1e-40
    return None


def _in_dom(v, dom):
    if dom == "Complexes":
        return True
    if dom == "Reals":
        return _is_real_val(v) is True
    if dom == "Integers":
        return K.is_int(v)
    return True


@fn("FindInstance", area=AREA, status="added",
    notes="FindInstance[expr, vars] / [expr, vars, dom] / [expr, vars, dom, n] / [expr, vars, n]: up to n "
          "instances {{x -> v, ...}} satisfying expr, every one verified by substitution (exactly or at 50 digits). "
          "Equations are solved with Solve after giving simple values (0, 1, -1, 2, ...) to the surplus variables; "
          "pure inequalities use SymPy solveset (univariate) or a search over simple points; Integers uses a bounded "
          "search (equations solved for the last variable); Booleans uses SatisfiabilityInstances. Without "
          "domain, variables are real when the condition has inequalities (as Wolfram). {} is returned only when "
          "the system is proved to have no solution (0-dimensional Solve or an empty solveset); otherwise a search "
          "that finds nothing raises")
def _findinstance(expr, vars_, *rest):
    fname = "FindInstance"
    rest, opts = _opts(rest, ("WorkingPrecision", "RandomSeeding"), fname)
    vs = _vars(vars_, fname)
    dom, n = None, 1
    for r in rest:
        if isinstance(r, Symbol) and r.name in ("Reals", "Complexes", "Integers", "Booleans", "Rationals"):
            dom = r.name
        elif K.is_int(r) and int(r) >= 1:
            n = int(r)
        else:
            raise ValueError(f"{fname}: invalid argument {r!r}")
    if dom == "Rationals":
        raise NotImplementedError(f"{fname}: the Rationals domain is not supported")
    expr = ev(expr)
    if expr is False:
        return L()
    if dom == "Booleans":
        inst = ev(Expr("SatisfiabilityInstances", expr, L(*vs), n))
        return L(*(L(*(rule(v, b) for v, b in zip(vs, row.args))) for row in inst.args))
    has_ineq = _contains(expr, lambda s: _is_head(s, "Less", "Greater", "LessEqual", "GreaterEqual"))
    if dom is None:
        dom = "Reals" if has_ineq else "Complexes"
    if expr is True:
        val = 0
        return L(*[L(*(rule(v, val) for v in vs))] * 1) if n == 1 else L(*(L(*(rule(v, k) for v in vs))
                                                                        for k in range(n)))
    if dom == "Integers":
        return _fi_integers(expr, vs, n, fname)
    found = []

    def add(assign):
        key = tuple(str(a[1]) for a in assign)
        if key in [k for k, _ in found]:
            return
        if not all(_in_dom(val, dom) for _, val in assign):
            return
        if _holds(expr, assign) is True:
            found.append((key, assign))

    # DNF over the logical structure
    s = tosp(expr, fname)
    clauses = sp.to_dnf(s, simplify=False) if len(s.atoms(sp.core.relational.Relational)) <= 12 else s
    clauses = list(clauses.args) if isinstance(clauses, sp.Or) else [clauses]
    proved_empty = True
    for cl in clauses:
        atoms = list(cl.args) if isinstance(cl, sp.And) else [cl]
        eqs = [a for a in atoms if isinstance(a, sp.Eq)]
        others = [a for a in atoms if not isinstance(a, sp.Eq)]
        if eqs:
            eq_ml = [fromsp(a) for a in eqs]
            nfree = max(0, len(vs) - len(eqs))
            r_ok = _fi_solve(eq_ml, vs, nfree, dom, add, n, found)
            if not r_ok or others or nfree:
                proved_empty = proved_empty and r_ok and not others and not nfree
        else:
            proved_empty = proved_empty and _fi_ineq(others, vs, dom, add, n, found, fname)
        if len(found) >= n:
            break
    if found:
        return L(*(L(*(rule(v, val) for v, val in a)) for _, a in found[:n]))
    if proved_empty:
        return L()
    raise NotImplementedError(f"{fname}: no instance found and the system was not proved inconsistent")


def _fi_solve(eqs, vs, nfree, dom, add, n, found):
    """Solve the equations after fixing nfree variables to simple values; True if the search was exhaustive
    (no surplus variables and Solve succeeded)."""
    sdom = Symbol("Reals") if dom == "Reals" else None
    for k in range(nfree, len(vs) + 1):
        for fixed in itertools.combinations(list(reversed(vs)), k):
            free = [v for v in vs if not any(K.same(v, f) for f in fixed)]
            for combo in itertools.product(_cands(None)[:7 if k else 1], repeat=k):
                sub = list(zip(fixed, combo))
                es = [ev(Expr("ReplaceAll", e, L(*(rule(a, b) for a, b in sub)))) for e in eqs]
                if any(e is False for e in es):
                    continue
                es = [e for e in es if e is not True]
                if not free:
                    if not es:
                        add([(v, dict((id(a), b) for a, b in sub).get(id(v))) for v in vs])
                    continue
                try:
                    sol = ev(Expr("Solve", Expr("And", *es) if len(es) > 1 else (es[0] if es else True),
                                  L(*free), *([sdom] if sdom is not None else [])))
                except Exception:  # noqa: BLE001
                    continue
                if not is_list(sol):
                    continue
                for srow in sol.args:
                    m = {}
                    for rr in srow.args:
                        m[str(rr.args[0])] = rr.args[1]
                    vals = []
                    ok = True
                    for v in vs:
                        if str(v) in m:
                            val = m[str(v)]
                            if _contains(val, lambda q: isinstance(q, Symbol) and any(K.same(q, w) for w in vs)) \
                                    or _is_head(val, "ConditionalExpression"):
                                ok = False
                                break
                            vals.append((v, val))
                        else:
                            fx = [b for a, b in sub if K.same(a, v)]
                            vals.append((v, fx[0] if fx else 0))
                    if ok:
                        add(vals)
                    if len(found) >= n:
                        return True
                if k == 0:
                    if len(sol.args) == 0 or all(len(r.args) == len(vs) for r in sol.args):
                        return True       # complete 0-dimensional solve
            if len(found) >= n:
                return True
        if len(found) >= n:
            return True
    return False


def _fi_ineq(atoms, vs, dom, add, n, found, fname):
    """Pure inequalities: univariate -> solveset; multivariate -> search.  True iff proved empty."""
    if dom == "Complexes":
        dom = "Reals"
    if len(vs) == 1:
        x = sp.Symbol(vs[0].name, real=True)
        c = sp.And(*atoms).xreplace({sp.Symbol(vs[0].name): x})
        S = _real_set(c, x, fname)
        if _is_empty(S, fname):
            return True
        for p in _nice_points(S):
            add([(vs[0], fromsp(p))])
            if len(found) >= n:
                break
        return False
    grid = [sp.Integer(0), sp.Integer(1), sp.Integer(-1), sp.Rational(1, 2), sp.Rational(-1, 2), sp.Integer(2),
            sp.Integer(-2), sp.Rational(1, 4), sp.Rational(-1, 4), sp.Integer(3), sp.Integer(-3), sp.Integer(10),
            sp.Integer(-10)]
    if len(vs) <= 4:
        for combo in itertools.product(grid, repeat=len(vs)):
            add([(v, fromsp(c)) for v, c in zip(vs, combo)])
            if len(found) >= n:
                break
    return False


def _nice_points(S):
    """Simple members of a real set: integers, then halves, interval midpoints, finite points."""
    out_ = []
    for k in sorted(range(-60, 61), key=lambda t: (abs(t), -t)):
        for d in (1, 2, 4, 10):
            p = sp.Rational(k, d)
            if S.contains(p) is sp.true and p not in out_:
                out_.append(p)
        if len(out_) >= 20:
            return out_
    parts = S.args if isinstance(S, sp.Union) else [S]
    for part in parts:
        if isinstance(part, sp.FiniteSet):
            out_ += [p for p in part if p not in out_]
        elif isinstance(part, sp.Interval):
            a, b = part.start, part.end
            if a.is_finite and b.is_finite:
                for t in (sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(2, 3)):
                    out_.append(a + (b - a) * t)
            elif a.is_finite:
                out_ += [a + 1, a + 2]
            elif b.is_finite:
                out_ += [b - 1, b - 2]
    return out_


def _fi_integers(expr, vs, n, fname):
    found = []
    seen = set()
    s = tosp(expr, fname)
    eqs = [a for a in (s.args if isinstance(s, sp.And) else [s]) if isinstance(a, sp.Eq)]
    X = [sp.Symbol(v.name) for v in vs]
    limit_ = 2000 if len(vs) <= 2 else (60 if len(vs) == 3 else 12)
    order = sorted(range(-limit_, limit_ + 1), key=lambda t: (abs(t), -t))
    if eqs and len(vs) >= 1:
        last = X[-1]
        poly_eq = eqs[0].lhs - eqs[0].rhs
        if not poly_eq.has(last):
            last = None
        for combo in itertools.product(order, repeat=len(vs) - 1) if len(vs) > 1 else [()]:
            if len(found) >= n:
                break
            base = dict(zip(X[:-1], combo))
            if last is not None:
                try:
                    roots = sp.solve(poly_eq.subs(base), last)
                except Exception:  # noqa: BLE001
                    roots = []
                vals = [r for r in roots if r.is_integer]
            else:
                vals = order[:41]
            for r in sorted(vals, key=lambda t: (abs(int(t)), -int(t))):
                assign = [(v, int(c)) for v, c in zip(vs[:-1], combo)] + [(vs[-1], int(r))]
                key = tuple(c for _, c in assign)
                if key in seen:
                    continue
                if _holds(expr, assign) is True:
                    seen.add(key)
                    found.append(assign)
                    if len(found) >= n:
                        break
    else:
        cnt = 0
        for combo in itertools.product(order[:min(len(order), 201 if len(vs) <= 2 else 21)], repeat=len(vs)):
            cnt += 1
            assign = list(zip(vs, combo))
            if _holds(expr, assign) is True:
                found.append(assign)
                if len(found) >= n:
                    break
    if found:
        return L(*(L(*(rule(v, c) for v, c in a)) for a in found[:n]))
    raise NotImplementedError(f"{fname}: no integer instance found in the searched range (this does not prove "
                              f"that none exists)")


# ───────────────────────────── cylindrical algebraic decomposition (1 and 2 variables) ─────────────────────────
#
# Real CAD for polynomial systems with rational coefficients in one variable, or two variables where every
# polynomial has degree <= 2 in the inner variable (so sections are explicit radicals).  Projection: leading
# coefficients, discriminants and pairwise resultants; cells are sampled exactly and the formula is evaluated on
# each cell.  Larger problems raise.

def _cad_formula(e, X, fname):
    s = tosp(e, fname)
    s = s.xreplace({sp.Symbol(v.name): v for v in X})
    if s.has(sp.Piecewise):
        raise NotImplementedError(f"{fname}: Piecewise is not supported")
    polys = []
    for r in s.atoms(sp.core.relational.Relational):
        p = sp.together(r.lhs - r.rhs)
        num, den = sp.fraction(p)
        if den.free_symbols & set(X):
            raise NotImplementedError(f"{fname}: rational functions are not supported")
        num = sp.expand(num)
        if not num.is_polynomial(*X):
            raise NotImplementedError(f"{fname}: only polynomial (in)equalities are supported")
        if num.free_symbols - set(X):
            raise NotImplementedError(f"{fname}: the system has parameters {num.free_symbols - set(X)}")
        if not all(c.is_rational for c in sp.Poly(num, *X).coeffs()):
            raise NotImplementedError(f"{fname}: only rational coefficients are supported")
        if num.free_symbols:
            polys.append(sp.Poly(num, *X))
    if s.free_symbols - set(X):
        raise NotImplementedError(f"{fname}: the formula has free symbols {s.free_symbols - set(X)}")
    return s, polys


def _truth_at(s, point, fname):
    v = s.xreplace(point)
    if v in (sp.true, sp.false):
        return v is sp.true
    if isinstance(v, sp.core.relational.Relational):
        d = _num(v.lhs - v.rhs)
        if d is None:
            raise NotImplementedError(f"{fname}: cannot evaluate the sign at a sample point")
        sgn = d[1]
        return {sp.Eq: sgn == 0, sp.Ne: sgn != 0, sp.Lt: sgn < 0, sp.Le: sgn <= 0, sp.Gt: sgn > 0,
                sp.Ge: sgn >= 0}[type(v)]
    if isinstance(v, sp.And):
        return all(_truth_at(a, {}, fname) for a in v.args)
    if isinstance(v, sp.Or):
        return any(_truth_at(a, {}, fname) for a in v.args)
    if isinstance(v, sp.Not):
        return not _truth_at(v.args[0], {}, fname)
    if isinstance(v, (sp.Xor, sp.Implies, sp.Equivalent)):
        return _truth_at(sp.simplify_logic(v.to_nnf()) if not isinstance(v, sp.Xor) else v.to_nnf(), {}, fname)
    raise NotImplementedError(f"{fname}: cannot evaluate {v}")


def _real_roots_1d(polys, x):
    roots = set()
    for p in polys:
        q = sp.Poly(p.as_expr(), x)
        if q.degree() <= 0:
            continue
        for fac, _ in sp.factor_list(q.as_expr())[1]:
            fq = sp.Poly(fac, x)
            if fq.degree() <= 0:
                continue
            if fq.degree() <= 2:
                rs = [r for r in sp.roots(fq, x).keys() if r.is_real]
            else:
                rs = list(sp.Poly(fac, x).real_roots())
                rs = [sp.nsimplify(r) if not isinstance(r, sp.CRootOf) else r for r in rs]
            roots |= set(rs)
    rs = sorted(roots, key=lambda r: float(sp.N(r, 30)))
    out_ = []
    for r in rs:
        if not out_ or abs(float(sp.N(r - out_[-1], 30))) > 1e-25:
            out_.append(r)
    return out_


def _between(a, b):
    """A simple rational strictly between reals a < b (either may be infinite)."""
    if a == -sp.oo and b == sp.oo:
        return sp.Integer(0)
    if a == -sp.oo:
        return sp.floor(sp.N(b, 30)) - 1
    if b == sp.oo:
        return sp.ceiling(sp.N(a, 30)) + 1
    fa, fb = sp.N(a, 40), sp.N(b, 40)
    for d in (1, 2, 4, 8, 16, 64, 256, 1024, 2 ** 16, 2 ** 30):
        n0 = sp.floor(fa * d) + 1
        cand = sp.Rational(int(n0), d)
        if fa < cand < fb:
            # prefer integers / short fractions closest to the middle
            return cand
    return sp.nsimplify((fa + fb) / 2, rational=True)


def _cells_1d(roots):
    """[('pt', r) | ('iv', lo, hi, sample)] in increasing order."""
    edges = [-sp.oo] + roots + [sp.oo]
    cells = []
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        cells.append(("iv", a, b, _between(a, b)))
        if i < len(roots):
            cells.append(("pt", roots[i]))
    return cells


def _iv_cond(var, cells):
    """Condition for a maximal run of consecutive 1-D cells."""
    first, last = cells[0], cells[-1]
    lo, lo_strict = (first[1], True) if first[0] == "iv" else (first[1], False)
    hi, hi_strict = (last[2], True) if last[0] == "iv" else (last[1], False)
    if first[0] == "pt" and last[0] == "pt" and len(cells) == 1:
        return Expr("Equal", var, fromsp(lo))
    L_ = lo != -sp.oo
    H_ = hi != sp.oo
    if not L_ and not H_:
        return True
    if not L_:
        return Expr("Less" if hi_strict else "LessEqual", var, fromsp(hi))
    if not H_:
        return Expr("Greater" if lo_strict else "GreaterEqual", var, fromsp(lo))
    if lo_strict == hi_strict:
        return Expr("Less" if lo_strict else "LessEqual", fromsp(lo), var, fromsp(hi))
    return Expr("Inequality", fromsp(lo), Symbol("Less" if lo_strict else "LessEqual"), var,
                Symbol("Less" if hi_strict else "LessEqual"), fromsp(hi))


def _runs(cells, flags):
    out_, cur = [], []
    for c, f in zip(cells, flags):
        if f:
            cur.append(c)
        elif cur:
            out_.append(cur)
            cur = []
    if cur:
        out_.append(cur)
    return out_


def _sections(polys, x, y, x0):
    """Exact y-roots at x = x0 of the polys (with their symbolic branch expressions in x when available)."""
    secs = []
    for p in polys:
        if not p.as_expr().has(y):
            continue
        q = sp.Poly(p.as_expr(), y)
        if q.degree() > 2:
            raise NotImplementedError("CylindricalDecomposition: inner degree above 2 is not supported")
        cs = [sp.expand(c) for c in q.all_coeffs()]
        if q.degree() == 1:
            branches = [-cs[1] / cs[0]]
        else:
            a, b, c = cs
            disc = sp.expand(b ** 2 - 4 * a * c)
            branches = [sp.simplify((-b - sp.sqrt(disc)) / (2 * a)), sp.simplify((-b + sp.sqrt(disc)) / (2 * a))]
            if a.has(x) or True:
                branches.append(-c / b if b != 0 else sp.nan)       # degenerate a == 0 at isolated x
        for br in branches:
            if br is sp.nan:
                continue
            try:
                v = br.subs(x, x0)
            except Exception:  # noqa: BLE001
                continue
            if v.has(sp.zoo, sp.nan, sp.oo):
                continue
            r = _num(v)
            if r is None or not r[0]:
                continue
            pv = _num(p.as_expr().subs({x: x0, y: v}))
            if pv is None or pv[1] != 0:
                continue
            secs.append((v, br))
    # sort and merge equal values
    secs.sort(key=lambda t: float(sp.N(t[0], 30)))
    merged = []
    for v, br in secs:
        if merged and abs(float(sp.N(v - merged[-1][0], 30))) < 1e-25:
            continue
        merged.append((v, br))
    return merged


def _ycond(yml, y, s, x, x0, secs, is_point, fname):
    """Description in y of {y : s(x0, y)} with sections secs (values at x0 and branch expressions)."""
    vals = [v for v, _ in secs]
    exprs = [v if is_point else br for v, br in secs]
    edges = [-sp.oo] + vals + [sp.oo]
    eedges = [-sp.oo] + exprs + [sp.oo]
    cells, flags = [], []
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        smp = _between(a, b)
        cells.append(("iv", eedges[i], eedges[i + 1], smp))
        flags.append(_truth_at(s, {x: x0, y: smp}, fname))
        if i < len(vals):
            cells.append(("pt", exprs[i]))
            flags.append(_truth_at(s, {x: x0, y: vals[i]}, fname))
    runs = _runs(cells, flags)
    if not runs:
        return False, cells, flags
    conds = [_iv_cond(yml, r) for r in runs]
    return (conds[0] if len(conds) == 1 else Expr("Or", *conds)), cells, flags


def _cad(e, X, fname):
    if len(X) > 2:
        raise NotImplementedError(f"{fname}: more than two variables are not supported")
    if any(not isinstance(v, Symbol) for v in X):
        raise ValueError(f"{fname}: the variables must be symbols")
    if _contains(e, lambda q: _is_head(q, "Exists", "ForAll")):
        raise NotImplementedError(f"{fname}: quantifiers are not supported")
    XS = [sp.Symbol(v.name, real=True) for v in X]
    s, polys = _cad_formula(e, XS, fname)
    x = XS[0]
    if len(XS) == 1:
        roots = _real_roots_1d(polys, x)
        cells = _cells_1d(roots)
        flags = [_truth_at(s, {x: c[1] if c[0] == "pt" else c[3]}, fname) for c in cells]
        return {"dim": 1, "cells": [(c, f, None) for c, f in zip(cells, flags)], "s": s, "X": XS, "polys": polys}
    y = XS[1]
    proj = []
    ys = [p for p in polys if p.as_expr().has(y)]
    for p in polys:
        q = sp.Poly(p.as_expr(), y)
        if q.degree() == 0:
            proj.append(sp.Poly(p.as_expr(), x))
            continue
        if q.degree() > 2:
            raise NotImplementedError(f"{fname}: degree above 2 in {y} is not supported")
        proj.append(sp.Poly(q.LC(), x)) if sp.Poly(q.LC(), x).degree() > 0 else None
        if q.degree() >= 2:
            d = sp.discriminant(q.as_expr(), y)
            if d != 0:
                proj.append(sp.Poly(d, x))
        if q.degree() >= 1:
            for c in q.all_coeffs():
                pass
    for p1, p2 in itertools.combinations(ys, 2):
        r = sp.resultant(p1.as_expr(), p2.as_expr(), y)
        if r != 0:
            proj.append(sp.Poly(r, x))
        else:
            raise NotImplementedError(f"{fname}: polynomials with a common factor are not supported")
    proj = [p for p in proj if p is not None and p.degree() > 0]
    roots = _real_roots_1d(proj, x)
    xcells = _cells_1d(roots)
    out_ = []
    for c in xcells:
        x0 = c[1] if c[0] == "pt" else c[3]
        secs = _sections(ys, x, y, x0)
        yc, ycells, yflags = _ycond(X[1], y, s, x, x0, secs, c[0] == "pt", fname)
        out_.append((c, yc, (ycells, yflags, x0)))
    return {"dim": 2, "cells": out_, "s": s, "X": XS, "polys": polys, "proj": proj, "xml": X[0], "yml": X[1]}


def _cad_to_cond(cad, X):
    """Wolfram-style nested condition from the CAD cells (adjacent x-cells with equal descriptions merged)."""
    xv = X[0]
    if cad["dim"] == 1:
        cells = [c for c, _, _ in cad["cells"]]
        flags = [f for _, f, _ in cad["cells"]]
        runs = _runs(cells, flags)
        conds = [_iv_cond(xv, r) for r in runs]
        if not conds:
            return False
        return conds[0] if len(conds) == 1 else Expr("Or", *conds)
    cells_ = cad["cells"]
    x = cad["X"][0] if "X" in cad else None

    def compat(pt_i, iv_i):
        if not (0 <= iv_i < len(cells_)):
            return False
        c, _, data = cells_[pt_i]
        ci, _, di = cells_[iv_i]
        if data is None or di is None or x is None:
            return False
        (yc1, f1, _), (yc2, f2, _) = data, di
        pvals = [u[1] for u in yc1 if u[0] == "pt"]
        for w in yc2:
            if w[0] != "pt":
                continue
            try:
                v2 = w[1].subs(x, c[1])
            except Exception:  # noqa: BLE001
                return False
            if v2.has(sp.zoo, sp.nan):
                return False
            r = _num(v2)
            if r is None or not r[0]:
                return False
            if not any((_num(v2 - pv) or (0, 1))[1] == 0 for pv in pvals):
                return False
        ycm = cells_[iv_i][1]
        if ycm is True or ycm is False:
            return all(f == ycm for f in f1)
        ysym = cad["yml"]
        for u, f in zip(yc1, f1):
            yv = u[3] if u[0] == "iv" else u[1]
            t = ev(Expr("ReplaceAll", ycm, L(rule(cad["xml"], fromsp(c[1])), rule(ysym, fromsp(yv)))))
            if t is not True and t is not False:
                t = ev(Expr("N", t, 50))
            if t is not f:
                return False
        return True
    groups = []
    for i, (c, yc, _) in enumerate(cells_):
        if yc is False:
            groups.append(None)
            continue
        ycm = ev(yc) if yc is not True else True
        if c[0] == "pt" and groups and groups[-1] is not None and compat(i, i - 1):
            groups[-1][0].append(c)
            continue
        if c[0] == "pt" and compat(i, i + 1):
            nyc = cells_[i + 1][1]
            groups.append(([c], ev(nyc) if nyc is not True else True))
            continue
        if groups and groups[-1] is not None and K.same(groups[-1][1], ycm) and c[0] == "iv":
            groups[-1][0].append(c)
        else:
            groups.append(([c], ycm))
    conds = []
    for g in groups:
        if g is None:
            continue
        cells, ycm = g
        xc = _iv_cond(xv, cells)
        conds.append(xc if ycm is True else (ycm if xc is True else Expr("And", xc, ycm)))
    if not conds:
        return False
    return ev(conds[0] if len(conds) == 1 else Expr("Or", *conds))


def _cad_args(args, fname):
    args, opts = _opts(args, ("Assumptions", "Method", "Backsubstitution", "WorkingPrecision"), fname)
    if len(args) < 2:
        raise ValueError(f"{fname} expects a condition and variables")
    e = ev(args[0])
    X = _vars(args[1], fname)
    rest = list(args[2:])
    return e, X, rest


@fn("CylindricalDecomposition", area=AREA, status="added",
    notes="CylindricalDecomposition[ineqs, {x, y}] / [ineqs, x]: equivalent condition in cylindrical form "
          "(x-intervals with y between explicit sections, e.g. -1 < x < 1 && -Sqrt[1 - x^2] < y < Sqrt[1 - x^2]) for "
          "polynomial (in)equalities with rational coefficients in one variable, or two variables with degree <= 2 "
          "in the second (exact CAD: leading coefficients, discriminants, resultants, exact cell samples); "
          "\"Function\" gives a CylindricalDecompositionFunction; quantifiers, 3+ variables, parameters and other "
          "forms (\"Boundary\", ...) raise")
def _cylindrical(*args):
    fname = "CylindricalDecomposition"
    e, X, rest = _cad_args(args, fname)
    if e is True or e is False:
        return e
    form = None
    if rest:
        if len(rest) != 1 or not isinstance(rest[0], str):
            raise ValueError(f"{fname}: invalid third argument")
        form = rest[0]
        if form not in ("Function",):
            raise NotImplementedError(f"{fname}: the form {form!r} is not supported")
    cad = _cad(e, X, fname)
    cond = _cad_to_cond(cad, X)
    if form == "Function":
        return Expr("CylindricalDecompositionFunction", cond, L(*X))
    return cond


@fn("GenericCylindricalDecomposition", area=AREA, status="added",
    notes="GenericCylindricalDecomposition[ineqs, {x, y}]: {full-dimensional part, boundary condition}: the "
          "open cells of the CAD (same scope as CylindricalDecomposition) and an Or of polynomial == 0 conditions "
          "containing every lower-dimensional solution cell (False if there is none)")
def _gcad(*args):
    fname = "GenericCylindricalDecomposition"
    e, X, rest = _cad_args(args, fname)
    if rest:
        raise NotImplementedError(f"{fname}: extra arguments are not supported")
    cad = _cad(e, X, fname)
    xv = cad["X"][0]
    bound = []

    def add_b(p):
        p = p.primitive()[1]
        pe = fromsp(p.as_expr().xreplace({v: sp.Symbol(v.name) for v in cad["X"]}))
        c = ev(Expr("Equal", pe, 0))
        if not any(K.same(c, b) for b in bound):
            bound.append(c)
    if cad["dim"] == 1:
        full = {"dim": 1, "cells": [(c, f and c[0] == "iv", None) for c, f, _ in cad["cells"]]}
        for c, f, _ in cad["cells"]:
            if f and c[0] == "pt":
                for p in cad["polys"]:
                    if _num(p.as_expr().subs(xv, c[1]))[1] == 0:
                        add_b(p)
        cond = _cad_to_cond(full, X)
    else:
        y = cad["X"][1]
        new = []
        for c, yc, (ycells, yflags, x0) in cad["cells"]:
            if c[0] == "pt":
                for yc_, f in zip(ycells, yflags):
                    if not f:
                        continue
                    yv = yc_[3] if yc_[0] == "iv" else yc_[1]
                    hit = [p for p in cad["polys"] if yc_[0] == "pt" and
                           (_num(p.as_expr().subs({xv: c[1], y: yv})) or (0, 1))[1] == 0]
                    if hit:
                        add_b(hit[0])
                    else:
                        for p in cad["proj"]:
                            if _num(p.as_expr().subs(xv, c[1]))[1] == 0:
                                add_b(p)
                                break
                continue
            fl = [f and yc_[0] == "iv" for yc_, f in zip(ycells, yflags)]
            for yc_, f in zip(ycells, yflags):
                if f and yc_[0] == "pt":
                    for p in cad["polys"]:
                        if p.as_expr().has(y):
                            pv = _num(p.as_expr().subs({xv: x0, y: yc_[1].subs(xv, x0)}))
                            if pv is not None and pv[1] == 0:
                                add_b(p)
            runs = _runs(ycells, fl)
            conds = [_iv_cond(X[1], r) for r in runs]
            ycm = False if not conds else (conds[0] if len(conds) == 1 else Expr("Or", *conds))
            new.append((c, ycm, None))
        cond = _cad_to_cond({"dim": 2, "cells": new}, X)
    b = False if not bound else (bound[0] if len(bound) == 1 else Expr("Or", *bound))
    return L(cond, b)


def _cdf_hook(head, args, ev_):
    if len(head.args) != 2 or not is_list(head.args[1]):
        return K.UNHANDLED
    cond, X = head.args
    if len(args) != len(X.args):
        raise ValueError("CylindricalDecompositionFunction: wrong number of arguments")
    return ev(Expr("ReplaceAll", cond, L(*(rule(v, a) for v, a in zip(X.args, args)))))


_KL.COMPOUND_HEAD_HOOKS["CylindricalDecompositionFunction"] = _cdf_hook


@fn("CylindricalDecompositionFunction", area=AREA, status="added",
    notes="CylindricalDecompositionFunction[cond, {x, y}] (returned by CylindricalDecomposition[..., "
          "\"Function\"]): applied to numbers it tests membership (cdf[x0, y0] -> True/False); Normal gives the "
          "cylindrical condition")
def _cdf(*args):
    raise Unevaluated()


@fn("SemialgebraicComponentInstances", area=AREA, status="added",
    notes="SemialgebraicComponentInstances[ineqs, {x, y}] / [ineqs, x]: sample points {{x -> a, y -> b}, ...}, "
          "at least one in each connected component of the solution set (one per true CAD cell, so possibly more "
          "than one per component); same scope as CylindricalDecomposition")
def _sci(*args):
    fname = "SemialgebraicComponentInstances"
    e, X, rest = _cad_args(args, fname)
    if rest:
        raise NotImplementedError(f"{fname}: extra arguments are not supported")
    if e is False:
        return L()
    cad = _cad(e, X, fname)
    pts = []
    if cad["dim"] == 1:
        for c, f, _ in cad["cells"]:
            if f:
                pts.append(L(rule(X[0], fromsp(c[1] if c[0] == "pt" else c[3]))))
        return L(*pts)
    xv = cad["X"][0]
    for c, yc, (ycells, yflags, x0) in cad["cells"]:
        for yc_, f in zip(ycells, yflags):
            if f:
                yv = yc_[3] if yc_[0] == "iv" else (yc_[1].subs(xv, x0) if hasattr(yc_[1], "subs") else yc_[1])
                pts.append(L(rule(X[0], fromsp(sp.nsimplify(x0) if x0.is_Rational else x0)),
                             rule(X[1], fromsp(sp.simplify(yv)))))
    return L(*pts)
