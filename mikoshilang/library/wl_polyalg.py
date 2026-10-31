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
