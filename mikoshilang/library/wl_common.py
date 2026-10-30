"""Shared helpers for the Wolfram coverage batches (higher-order calls, options, comparisons)."""
from __future__ import annotations

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import Unevaluated

L = K.L
is_int = K.is_int
is_list = K.is_list
same = K.same
is_assoc = KL.is_assoc
is_rule = KL.is_rule


def ev(x):
    from ..evaluate import evaluate
    return evaluate(x, 1)


def apply(f, *args):
    return ev(K.call(f, *args))


def true(f, *args):
    return apply(f, *args) is True


def is_sym(x, name):
    return isinstance(x, Symbol) and x.name == name


def split_options(args, names, fname):
    """Split trailing Name -> value options (Name in names) from positional arguments."""
    args = list(args)
    opts = {}
    while args and is_rule(args[-1]) and isinstance(args[-1].args[0], Symbol) and args[-1].args[0].name in names:
        r = args.pop()
        opts[r.args[0].name] = r.args[1]
    for a in args:
        if is_rule(a) and isinstance(a.args[0], Symbol) and a.args[0].name in names:
            raise ValueError(f"{fname}: options must come after the positional arguments")
    return args, opts


def items_of(x, fname):
    """Elements of a list or compound expression, or the values of an association; a symbol -> Unevaluated."""
    if is_assoc(x):
        return KL.assoc_values(x, ev)
    if isinstance(x, Expr) and not KL.atomq(x):
        return list(x.args)
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise ValueError(f"{fname}: {x!r} is not a list")


def list_items(x, fname):
    if is_list(x):
        return list(x.args)
    if isinstance(x, Symbol) or (isinstance(x, Expr) and not KL.atomq(x) and x.head not in ("List",)):
        raise Unevaluated()
    raise ValueError(f"{fname}: {x!r} is not a list")


def canonical(x, fname):
    return KL.canonical_key(x, fname)


def tester(test):
    """SameQ, or a user two-argument test function."""
    if test is None or is_sym(test, "SameQ"):
        return same
    return lambda u, v: apply(test, u, v) is True


def dedup(items, test=None):
    t = tester(test)
    out = []
    for x in items:
        if not any(t(y, x) for y in out):
            out.append(x)
    return out


def classes(items, keyf=None, test=None):
    """Equivalence classes in order of first appearance: by key function (SameQ on keys) or a test."""
    keys, groups = [], []
    if test is not None and keyf is None:
        t = tester(test)
        for x in items:
            for i, k in enumerate(keys):
                if t(k, x):
                    groups[i].append(x)
                    break
            else:
                keys.append(x)
                groups.append([x])
        return keys, groups
    index = {}
    for x in items:
        k = keyf(x) if keyf else x
        tg = K.tag(k)
        if tg in index:
            groups[index[tg]].append(x)
        else:
            index[tg] = len(keys)
            keys.append(k)
            groups.append([x])
    return keys, groups


def real_number(x, fname, what="value"):
    """A real numeric value as a SymPy number for comparisons (exact or inexact)."""
    import sympy as sp
    try:
        v = K.to_sp(x)
    except Exception:
        v = None
    if not isinstance(v, sp.Basic) or not v.is_number or v.is_real is False:
        raise ValueError(f"{fname}: {what} {x!r} is not a real number")
    return v


def int_value(x, fname, what="argument", lo=None):
    if not is_int(x):
        if isinstance(x, (Symbol, Expr)) and not isinstance(x, bool):
            raise ValueError(f"{fname}: {what} {x!r} must be an integer")
        raise ValueError(f"{fname}: {what} {x!r} must be an integer")
    n = int(x)
    if lo is not None and n < lo:
        raise ValueError(f"{fname}: {what} must be at least {lo}")
    return n


def upto(spec, n_available, fname):
    """n or UpTo[n] -> count; plain n larger than available raises."""
    if isinstance(spec, Expr) and spec.head == "UpTo" and len(spec.args) == 1:
        k = int_value(spec.args[0], fname, "UpTo bound", 0)
        return min(k, n_available)
    k = int_value(spec, fname, "count", 0)
    if k > n_available:
        raise ValueError(f"{fname}: cannot take {k} elements from {n_available}")
    return k
