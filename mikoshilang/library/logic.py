"""Boolean logic.  Connectives evaluate when all arguments are True/False and
otherwise stay unevaluated (Wolfram behaviour); legacy Iff/Xnor coerced
symbols with Python bool(), so Iff[x, y] returned True."""
import itertools

import sympy as sp

from ..expr import Expr, Symbol
from .core import Unevaluated, fn, error, record, S, R, L, rule, sym

AREA = "logic"


def _bools(args):
    if all(isinstance(a, bool) for a in args):
        return list(args)
    raise Unevaluated()


def _conn(name, f, notes, min_args=2, status="fixed"):
    @fn(name, area=AREA, status=status, notes=notes)
    def impl(*args):
        if len(args) < min_args:
            raise ValueError(f"{name} needs at least {min_args} arguments")
        return f(_bools(args))
    return impl


@fn("Implies", area=AREA, status="verified",
    notes="Implies[p, q] for booleans; symbolic arguments stay unevaluated")
def _implies(p, q):
    a, b = _bools([p, q])
    return (not a) or b


_conn("Equivalent", lambda b: all(x == b[0] for x in b), "variadic; True when all arguments are equal booleans")
_conn("Iff", lambda b: all(x == b[0] for x in b),
      "alias of Equivalent; legacy used Python bool() so Iff[x, y] was True for any symbols")
_conn("Xor", lambda b: sum(b) % 2 == 1, "variadic exclusive or (odd number of True)")
_conn("Xnor", lambda b: sum(b) % 2 == 0,
      "Not[Xor[...]]; legacy used Python bool() so Xnor[x, y] was True for any symbols")
_conn("Nand", lambda b: not all(b), "variadic Not[And[...]]")
_conn("Nor", lambda b: not any(b), "variadic Not[Or[...]]")


def _bexpr(e, fname):
    v = sym(e, fname)
    if not isinstance(v, (sp.logic.boolalg.Boolean, sp.Symbol)) and v not in (sp.true, sp.false):
        raise ValueError(f"{fname}: argument must be a Boolean expression")
    return v


def _vars(v):
    return sorted(v.free_symbols, key=lambda s: s.name)


@fn("TruthTable", area=AREA, status="verified",
    notes="rows {v1, ..., vn, value} over all assignments, variables in alphabetical order (MikoshiLang)")
def _tt(e):
    v = _bexpr(e, "TruthTable")
    xs = _vars(v)
    if len(xs) > 16:
        raise ValueError("TruthTable: too many variables")
    if not xs:
        return L(L(bool(v)))
    return L(*(L(*vals, bool(v.subs(dict(zip(xs, vals))))) for vals in itertools.product([False, True], repeat=len(xs))))


@fn("AllSAT", area=AREA, status="verified", notes="all satisfying assignments as rule lists (MikoshiLang)")
def _allsat(e):
    v = _bexpr(e, "AllSAT")
    xs = _vars(v)
    if len(xs) > 20:
        raise ValueError("AllSAT: too many variables")
    out = []
    for vals in itertools.product([False, True], repeat=len(xs)):
        if bool(v.subs(dict(zip(xs, vals)))):
            out.append(L(*(rule(Symbol(x.name), b) for x, b in zip(xs, vals))))
    return L(*out)


@fn("SATSolve", area=AREA, status="fixed",
    notes="one satisfying assignment {x -> True, ...} (all variables, alphabetical) or False; legacy omitted "
          "variables SymPy did not need and could return {True -> True} for tautologies")
def _sat(e):
    v = _bexpr(e, "SATSolve")
    xs = _vars(v)
    model = sp.satisfiable(v)
    if model is False:
        return False
    model = {k: bool(val) for k, val in model.items() if isinstance(k, sp.Symbol)}
    return L(*(rule(Symbol(x.name), model.get(x, False)) for x in xs))


@fn("CNF", area=AREA, status="verified", notes="conjunctive normal form (SymPy to_cnf; Wolfram BooleanConvert[e, \"CNF\"])")
def _cnf(e):
    return R(sp.to_cnf(_bexpr(e, "CNF")))


@fn("DNF", area=AREA, status="verified", notes="disjunctive normal form (SymPy to_dnf)")
def _dnf(e):
    return R(sp.to_dnf(_bexpr(e, "DNF")))


@fn("BooleanMinimize", area=AREA, status="verified", notes="minimal DNF (SymPy simplify_logic, Quine-McCluskey)")
def _bmin(e):
    return R(sp.simplify_logic(_bexpr(e, "BooleanMinimize"), form="dnf"))
