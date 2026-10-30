"""List and set functions (Wolfram canonical order and SameQ equality)."""
import sympy as sp

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, error, record, S, R, L, numeric_out, sym, number, is_list, int_arg,
                   nonneg_int, pos_int, list_arg)

AREA = "lists and sets"


def _same(a, b):
    from ..kernel import same
    return same(a, b)


def _okey(x):
    from ..kernel import order_key
    return order_key(x)


def _dedup(items):
    out = []
    for x in items:
        if not any(_same(x, y) for y in out):
            out.append(x)
    return out


def _member(x, items):
    return any(_same(x, y) for y in items)


def _sorted_set(items):
    return L(*sorted(_dedup(items), key=_okey))


def _lists(args, fname, min_n=1):
    if len(args) < min_n:
        raise ValueError(f"{fname} needs at least {min_n} list(s)")
    return [list_arg(a, fname) for a in args]


@fn("Union", area=AREA, status="promoted",
    notes="variadic, including Union[list]; sorted in canonical order with SameQ duplicates removed (legacy "
          "2-argument form sorted by string, so Union[{10, 9}, {}] gave {10, 9})")
def _union(*args):
    if not args:
        return L()
    ls = _lists(args, "Union")
    return _sorted_set([x for l in ls for x in l])


@fn("Intersection", area=AREA, status="fixed", notes="variadic; canonical order (legacy sorted by string form)")
def _intersection(*args):
    ls = _lists(args, "Intersection")
    common = [x for x in ls[0] if all(_member(x, l) for l in ls[1:])]
    return _sorted_set(common)


@fn("Complement", area=AREA, status="fixed",
    notes="Complement[all, e1, e2, ...] in canonical order (legacy: 2 arguments only, sorted by string form)")
def _complement(*args):
    ls = _lists(args, "Complement")
    rest = [x for l in ls[1:] for x in l]
    return _sorted_set([x for x in ls[0] if not _member(x, rest)])


@fn("SetDifference", area=AREA, status="fixed", notes="alias of Complement[a, b] (canonical order)")
def _setdiff(a, b):
    return _complement(a, b)


@fn("SymmetricDifference", area=AREA, status="fixed",
    notes="elements in exactly one of the two lists, canonical order (legacy sorted by string form)")
def _symdiff(a, b):
    x, y = list_arg(a, "SymmetricDifference"), list_arg(b, "SymmetricDifference")
    return _sorted_set([v for v in x if not _member(v, y)] + [v for v in y if not _member(v, x)])


@fn("Cardinality", area=AREA, status="fixed",
    notes="number of distinct elements (legacy returned Length, counting duplicates)")
def _card(a):
    return len(_dedup(list_arg(a, "Cardinality")))


@fn("ElementQ", area=AREA, status="fixed",
    notes="ElementQ[x, list]: SameQ membership (legacy used ==, so ElementQ[1, {1.0}] was True; MemberQ "
          "gives False)")
def _elementq(x, lst):
    return _member(x, list_arg(lst, "ElementQ"))


@fn("SubsetQ", area=AREA, status="fixed",
    notes="SubsetQ[l1, l2] is True when l2 is a subset of l1 (Wolfram order; legacy tested the reverse)")
def _subsetq(a, b):
    big, small = list_arg(a, "SubsetQ"), list_arg(b, "SubsetQ")
    return all(_member(x, big) for x in small)


@fn("DeleteDuplicates", area=AREA, status="fixed",
    notes="SameQ comparison keeping first occurrences (legacy compared string forms, so the string \"x\" and "
          "the symbol x were treated as duplicates)")
def _deldup(a):
    return L(*_dedup(list_arg(a, "DeleteDuplicates")))


@fn("Riffle", area=AREA, status="fixed",
    notes="Riffle[list, x] and Riffle[list, {y1, ...}] (interleaves when the second list has n or n - 1 "
          "elements; legacy inserted the whole list each time)")
def _riffle(a, b):
    items = list_arg(a, "Riffle")
    n = len(items)
    if is_list(b) and len(b.args) in (n, n - 1) and n > 0:
        out = []
        for i, x in enumerate(items):
            out.append(x)
            if i < len(b.args):
                out.append(b.args[i])
        return L(*out)
    out = []
    for i, x in enumerate(items):
        out.append(x)
        if i < n - 1:
            out.append(b)
    return L(*out)


@fn("Interleave", area=AREA, status="fixed",
    notes="{a1, b1, ..., a2, b2, ...} for two or more lists (MikoshiLang); unequal lengths now raise (legacy "
          "silently truncated)")
def _interleave(a, b, *more):
    ls = [list_arg(v, "Interleave") for v in (a, b) + more]
    if len({len(v) for v in ls}) != 1:
        raise ValueError("Interleave: lists must have the same length")
    return L(*(x for group in zip(*ls) for x in group))


@fn("Partition", area=AREA, status="fixed",
    notes="Partition[list, n] and Partition[list, n, d] (incomplete blocks dropped, as Wolfram); legacy "
          "truncated non-integer n")
def _partition(a, n, d=None):
    items = list_arg(a, "Partition")
    k = pos_int(n, "Partition", "block size")
    step = k if d is None else pos_int(d, "Partition", "offset")
    return L(*(L(*items[i:i + k]) for i in range(0, len(items) - k + 1, step)))


@fn("Chunk", area=AREA, status="verified", notes="consecutive blocks of size n, last one possibly shorter (MikoshiLang)")
def _chunk(a, n):
    items = list_arg(a, "Chunk")
    k = pos_int(n, "Chunk", "block size")
    return L(*(L(*items[i:i + k]) for i in range(0, len(items), k)))


@fn("SlidingWindow", area=AREA, status="verified", notes="overlapping windows = Partition[list, n, 1] (MikoshiLang)")
def _sliding(a, n):
    return _partition(a, n, 1)


@fn("RotateLeft", area=AREA, status="verified", notes="RotateLeft[list] and RotateLeft[list, n] (n may be negative)")
def _rotl(a, n=1):
    items = list_arg(a, "RotateLeft")
    k = int_arg(n, "RotateLeft", "shift")
    if not items:
        return L()
    k %= len(items)
    return L(*(items[k:] + items[:k]))


@fn("RotateRight", area=AREA, status="verified", notes="RotateRight[list] and RotateRight[list, n]")
def _rotr(a, n=1):
    k = int_arg(n, "RotateRight", "shift")
    return _rotl(a, -k)


@fn("Accumulate", area=AREA, status="fixed",
    notes="exact running totals (legacy converted to floats and rounded to 10 digits); works on symbols")
def _accumulate(a):
    items = list_arg(a, "Accumulate")
    out, tot = [], sp.Integer(0)
    for x in items:
        tot = tot + sym(x, "Accumulate")
        out.append(numeric_out(tot, *items[:len(out) + 1]))
    return L(*out)


@fn("Differences", area=AREA, status="fixed",
    notes="exact successive differences, Differences[list, n] for n-th differences (legacy used rounded floats)")
def _differences(a, n=1):
    items = list_arg(a, "Differences")
    k = nonneg_int(n, "Differences", "order")
    vals = [sym(x, "Differences") for x in items]
    for _ in range(k):
        vals = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    return L(*(numeric_out(v, a) for v in vals))


@fn("GroupBy", area=AREA, status="fixed",
    notes="legacy ignored the function and grouped equal values; now GroupBy[list, f] -> Association[k -> {...}] "
          "keyed by f[elem] in order of first appearance (Association is an inert head in MikoshiLang)")
def _groupby(a, f):
    from ..evaluate import evaluate
    items = list_arg(a, "GroupBy")
    head = f.name if isinstance(f, Symbol) else f
    keys, groups = [], []
    for x in items:
        k = evaluate(Expr(head, x))
        for i, kk in enumerate(keys):
            if _same(k, kk):
                groups[i].append(x)
                break
        else:
            keys.append(k)
            groups.append([x])
    return Expr("Association", *(Expr("Rule", k, L(*g)) for k, g in zip(keys, groups)))


@fn("Tally", area=AREA, status="promoted", notes="{{elem, count}, ...} in order of first appearance (SameQ)")
def _tally(a):
    items = list_arg(a, "Tally")
    keys, counts = [], []
    for x in items:
        for i, k in enumerate(keys):
            if _same(x, k):
                counts[i] += 1
                break
        else:
            keys.append(x)
            counts.append(1)
    return L(*(L(k, c) for k, c in zip(keys, counts)))
