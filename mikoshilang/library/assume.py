"""Assumptions: Wolfram $Assumptions, Assuming, Refine and the Assumptions option.

Wolfram conditions (x > 0, Element[n, Integers], a && b, {c1, c2}, ...) are translated into SymPy
assumptions in two ways that are used together:

* facts about a single symbol (x > 0, x >= 0, x < 0, x <= 0, x != 0, Element[x, Reals | Integers |
  Rationals | Primes | Complexes | Algebraics]) are turned into SymPy symbols that carry those assumptions,
  which SymPy's simplifier and integrator understand;
* everything is also turned into a SymPy ``Q`` proposition for ``sp.refine`` / ``sp.ask``.

Symbols are mapped back afterwards, so results contain ordinary MikoshiLang symbols.
"""
import sympy as sp

from ..expr import Expr, Symbol
from .core import Unevaluated, fn, record, R, L, sym, is_list

AREA = "algebra"

_DOMAINS = {"Reals": ("real", sp.Q.real), "Integers": ("integer", sp.Q.integer),
            "Rationals": ("rational", sp.Q.rational), "Primes": ("prime", sp.Q.prime),
            "Complexes": ("complex", sp.Q.complex), "Algebraics": ("algebraic", sp.Q.algebraic)}
_SIGN = {"Greater": ("positive", sp.Q.positive), "GreaterEqual": ("nonnegative", sp.Q.nonnegative),
         "Less": ("negative", sp.Q.negative), "LessEqual": ("nonpositive", sp.Q.nonpositive)}


def current():
    """The session's $Assumptions (True when unset)."""
    from ..kernel import state
    v = state().own.get("$Assumptions")
    return True if v is None else v


def _is_true(a):
    return a is True or (isinstance(a, Symbol) and a.name == "True")


def combine(*conds):
    cs = [c for c in conds if not _is_true(c)]
    if not cs:
        return True
    return cs[0] if len(cs) == 1 else Expr("And", *cs)


class Assumed:
    """Translation of one Wolfram condition."""

    def __init__(self, cond, fname):
        self.fname = fname
        self.facts = {}         # symbol name -> {assumption: True}
        self.props = []         # SymPy Q propositions
        self._walk(cond)
        self.subs = {}
        for name, kw in self.facts.items():
            try:
                self.subs[sp.Symbol(name)] = sp.Symbol(name, **kw)
            except Exception:  # inconsistent facts, e.g. x > 0 and x < 0
                raise ValueError(f"{fname}: the assumptions on {name} are inconsistent")
        self.back = {v: k for k, v in self.subs.items()}
        self.Q = sp.And(*[p.xreplace(self.subs) for p in self.props]) if self.props else sp.true
        if self.Q is sp.false:
            raise ValueError(f"{fname}: the assumptions are inconsistent")

    def _fact(self, e, key):
        if isinstance(e, Symbol):
            self.facts.setdefault(e.name, {})[key] = True

    def _walk(self, c):
        if _is_true(c):
            return
        if c is False or (isinstance(c, Symbol) and c.name == "False"):
            raise ValueError(f"{self.fname}: the assumptions are False")
        if is_list(c) or (isinstance(c, Expr) and c.head == "And"):
            for a in c.args:
                self._walk(a)
            return
        if isinstance(c, Expr) and c.head == "Element" and len(c.args) == 2 and isinstance(c.args[1], Symbol) \
                and c.args[1].name in _DOMAINS:
            key, q = _DOMAINS[c.args[1].name]
            xs = c.args[0]
            xs = xs.args if isinstance(xs, Expr) and xs.head in ("List", "Alternatives") else (xs,)
            for x in xs:
                self._fact(x, key)
                self.props.append(q(sym(x, self.fname)))
            return
        if isinstance(c, Expr) and c.head in _SIGN and len(c.args) >= 2:
            for a, b in zip(c.args, c.args[1:]):        # chains a < b < c
                d = sp.expand(sym(a, self.fname) - sym(b, self.fname))
                key, q = _SIGN[c.head]
                if sym(b, self.fname) == 0:
                    self._fact(a, key)
                elif sym(a, self.fname) == 0:
                    self._fact(b, {"positive": "negative", "negative": "positive", "nonnegative": "nonpositive",
                                   "nonpositive": "nonnegative"}[key])
                else:
                    for s in (a, b):
                        if isinstance(s, Symbol):
                            self._fact(s, "real")
                self.props.append(q(d))
            return
        if isinstance(c, Expr) and c.head in ("Unequal", "Equal") and len(c.args) == 2:
            d = sp.expand(sym(c.args[0], self.fname) - sym(c.args[1], self.fname))
            if c.head == "Unequal" and sym(c.args[1], self.fname) == 0:
                self._fact(c.args[0], "nonzero")
            self.props.append(sp.Not(sp.Q.zero(d)) if c.head == "Unequal" else sp.Q.zero(d))
            return
        if isinstance(c, Expr) and c.head in ("Or", "Not"):
            self.props.append(self._prop(c))
            return
        raise ValueError(f"{self.fname}: unsupported assumption {c} (use inequalities, Equal/Unequal, "
                         "Element[x, Reals|Integers|Rationals|Primes|Complexes|Algebraics], && , || , !)")

    def _prop(self, c):
        if isinstance(c, Expr) and c.head == "Or":
            return sp.Or(*[self._prop(a) for a in c.args])
        if isinstance(c, Expr) and c.head == "Not":
            return sp.Not(self._prop(c.args[0]))
        if isinstance(c, Expr) and c.head == "And":
            return sp.And(*[self._prop(a) for a in c.args])
        sub = Assumed.__new__(Assumed)
        sub.fname, sub.facts, sub.props = self.fname, {}, []
        sub._walk(c)
        return sp.And(*sub.props)

    def apply(self, v):
        return v.xreplace(self.subs)

    def unapply(self, v):
        return v.xreplace(self.back) if isinstance(v, sp.Basic) else v

    def refine(self, v):
        v = self.apply(v)
        if isinstance(v, sp.Basic) and self.Q is not sp.true:
            try:
                v = sp.refine(v, self.Q)
            except Exception:
                pass
        if isinstance(v, (sp.And, sp.Or, sp.Not)):
            v = type(v)(*(self.decide(a) for a in v.args))
        elif isinstance(v, sp.core.relational.Relational):
            v = self.decide(v)
        return v

    def decide(self, rel):
        """True/False when the (relational or Boolean) rel follows from the assumptions, else rel."""
        if rel in (sp.true, sp.false):
            return rel
        q = None
        if isinstance(rel, sp.core.relational.Relational):
            d = sp.expand(rel.lhs - rel.rhs)
            q = {sp.StrictGreaterThan: sp.Q.positive, sp.GreaterThan: sp.Q.nonnegative,
                 sp.StrictLessThan: sp.Q.negative, sp.LessThan: sp.Q.nonpositive,
                 sp.Equality: sp.Q.zero}.get(type(rel))
            if q is not None:
                q = q(d)
            elif isinstance(rel, sp.Unequality):
                q = sp.Not(sp.Q.zero(d))
        if q is None:
            return rel
        try:
            ans = sp.ask(q, self.Q)
        except Exception:
            ans = None
        if ans is None and isinstance(rel, sp.core.relational.Relational):
            ans = self._interval_decide(rel)
        return sp.true if ans is True else sp.false if ans is False else rel

    def _interval_decide(self, rel):
        """Decide a one-variable real relation from the interval the assumptions give that variable."""
        free = rel.free_symbols
        if len(free) != 1:
            return None
        x = free.pop()
        dom = sp.S.Reals
        keyed = {sp.Q.positive: sp.Gt, sp.Q.nonnegative: sp.Ge, sp.Q.negative: sp.Lt, sp.Q.nonpositive: sp.Le}
        found = False
        for p in sp.And.make_args(self.Q):
            fn_ = getattr(p, "function", None)
            if fn_ in keyed and p.arguments[0].free_symbols == {x}:
                try:
                    dom = dom.intersect(sp.solveset(keyed[fn_](p.arguments[0], 0), x, sp.S.Reals))
                    found = True
                except Exception:
                    return None
        if not found and not x.is_real:
            return None
        try:
            ok = sp.solveset(rel, x, sp.S.Reals)
        except Exception:
            return None
        if dom.is_subset(ok):
            return True
        if dom.intersect(ok) is sp.S.EmptySet:
            return False
        return None


_REL = {"Greater": sp.Gt, "GreaterEqual": sp.Ge, "Less": sp.Lt, "LessEqual": sp.Le, "Equal": sp.Eq,
        "Unequal": sp.Ne}
_BOOL = {"And": sp.And, "Or": sp.Or, "Not": sp.Not}


def to_sym(e, fname):
    """sym() that also converts relations and And/Or/Not to SymPy relationals/Booleans."""
    if isinstance(e, Expr) and e.head in _REL and len(e.args) >= 2:
        parts = [_REL[e.head](sym(a, fname), sym(b, fname)) for a, b in zip(e.args, e.args[1:])]
        return sp.And(*parts)
    if isinstance(e, Expr) and e.head in _BOOL:
        return _BOOL[e.head](*(to_sym(a, fname) for a in e.args))
    if e is True or e is False:
        return sp.true if e else sp.false
    return sym(e, fname)


def from_sym(v):
    """R() that maps SymPy relationals/Booleans back to Wolfram heads."""
    if v is sp.true or v is sp.false:
        return bool(v)
    for head, cls in _REL.items():
        if type(v) is cls:
            return Expr(head, R(v.lhs), R(v.rhs))
    for head, cls in _BOOL.items():
        if type(v) is cls:
            return Expr(head, *(from_sym(a) for a in v.args))
    return R(v)


def split_option(args, fname):
    """Split trailing Assumptions -> a from positional args; returns (args, assumption or None)."""
    args = list(args)
    opt = None
    while args and isinstance(args[-1], Expr) and args[-1].head in ("Rule", "RuleDelayed") \
            and isinstance(args[-1].args[0], Symbol) and args[-1].args[0].name == "Assumptions":
        opt = args.pop().args[1]
    if args and isinstance(args[-1], Expr) and args[-1].head in ("Rule", "RuleDelayed"):
        raise ValueError(f"{fname}: unknown option {args[-1].args[0]}")
    return args, opt


def effective(explicit, opt, fname):
    """Wolfram: an explicit assumption argument is combined with $Assumptions; Assumptions -> a replaces it."""
    if opt is not None:
        return Assumed(opt, fname)
    return Assumed(combine(current(), *(explicit or [])), fname)


@fn("Refine", area=AREA, status="fixed",
    notes="Refine[expr, assum] / Refine[expr] (uses $Assumptions) / Refine[expr, Assumptions -> a]: SymPy refine "
          "with assumption-carrying symbols and Q-propositions (Refine[Sqrt[x^2], x > 0] -> x, "
          "Refine[Abs[x], x < 0] -> -x, Refine[Sin[n Pi], Element[n, Integers]] -> 0, Refine[x > 0, x > 1] -> "
          "True); relations that do not follow stay as they are. Legacy raised"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _refine(e, *rest):
    rest, opt = split_option(rest, "Refine")
    if len(rest) > 1:
        raise ValueError("Refine[expr, assum] takes one assumption argument")
    A = effective(rest, opt, "Refine")
    return from_sym(A.unapply(A.refine(to_sym(e, "Refine"))))


@fn("Assuming", area=AREA, status="promoted",
    notes="Assuming[assum, expr] evaluates expr with $Assumptions = $Assumptions && assum (held arguments), "
          "restoring $Assumptions afterwards, as in Wolfram"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _assuming(a, e):
    from ..kernel import state
    from ..evaluate import evaluate
    own = state().own
    had, old = "$Assumptions" in own, own.get("$Assumptions")
    own["$Assumptions"] = combine(current(), evaluate(a, 1))
    try:
        return evaluate(e, 1)
    finally:
        if had:
            own["$Assumptions"] = old
        else:
            own.pop("$Assumptions", None)


@fn("Assumptions", area=AREA, status="fixed",
    notes="Wolfram option name: Assumptions -> cond is accepted by Simplify, Refine and Integrate (default "
          "$Assumptions, which is True unless set or changed by Assuming). Assumptions[...] used as a function "
          "raises ValueError (legacy Assumptions[expr] always returned {})")
def _assumptions_call(*args):
    raise ValueError("Assumptions is an option name (Simplify[e, Assumptions -> x > 0]); see also $Assumptions "
                     "and Assuming[assum, expr]")
