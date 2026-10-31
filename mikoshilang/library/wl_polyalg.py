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
