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


_BOOL_HEADS = {"And": sp.And, "Or": sp.Or, "Not": sp.Not, "Xor": sp.Xor, "Implies": sp.Implies,
               "Equivalent": sp.Equivalent, "Iff": sp.Equivalent, "Nand": sp.Nand, "Nor": sp.Nor,
               "Xnor": lambda *a: sp.Not(sp.Xor(*a))}


def _to_bool(e, fname):
    if e is True or e is False:
        return sp.true if e else sp.false
    if isinstance(e, Symbol):
        if e.name in ("True", "False"):
            return sp.true if e.name == "True" else sp.false
        return sp.Symbol(e.name)
    if isinstance(e, Expr) and isinstance(e.head, str) and e.head in _BOOL_HEADS:
        return _BOOL_HEADS[e.head](*(_to_bool(a, fname) for a in e.args))
    raise ValueError(f"{fname}: argument must be a Boolean expression of symbols, &&, ||, ! and friends")


def _bexpr(e, fname):
    return _to_bool(e, fname)


def _from_bool(v):
    if v is sp.true or v is sp.false:
        return bool(v)
    if isinstance(v, sp.Symbol):
        return Symbol(v.name)
    name = {sp.And: "And", sp.Or: "Or", sp.Not: "Not", sp.Xor: "Xor", sp.Implies: "Implies",
            sp.Equivalent: "Equivalent"}.get(type(v))
    if name is None:
        raise ValueError(f"unexpected Boolean form {v}")
    return Expr(name, *(_from_bool(a) for a in v.args))


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
    return _from_bool(sp.to_cnf(_bexpr(e, "CNF")))


@fn("DNF", area=AREA, status="verified", notes="disjunctive normal form (SymPy to_dnf)")
def _dnf(e):
    return _from_bool(sp.to_dnf(_bexpr(e, "DNF")))


@fn("BooleanMinimize", area=AREA, status="verified", notes="minimal DNF (SymPy simplify_logic, Quine-McCluskey)")
def _bmin(e):
    v = _bexpr(e, "BooleanMinimize")
    if not sp.satisfiable(v):
        return False
    if not sp.satisfiable(sp.Not(v)):
        return True
    return _from_bool(sp.simplify_logic(v, form="dnf", force=True))
