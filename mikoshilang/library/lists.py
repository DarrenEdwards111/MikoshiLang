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


def _set_args(args, fname):
    """Positional lists and the SameTest option (None = SameQ)."""
    from .wl_common import split_options
    pos, opts = split_options(args, {"SameTest"}, fname)
    return pos, opts.get("SameTest")


def _set_result(items, test):
    from .wl_common import dedup
    if test is None:
        return _sorted_set(items)
    return L(*dedup(sorted(items, key=_okey), test))


def _member_t(x, items, test):
    if test is None:
        return _member(x, items)
    from .wl_common import tester
    t = tester(test)
    return any(t(x, y) for y in items)


@fn("Union", area=AREA, status="promoted",
    notes="variadic, including Union[list]; sorted in canonical order with SameQ duplicates removed; "
          "SameTest -> f (legacy 2-argument form sorted by string, so Union[{10, 9}, {}] gave {10, 9})"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _union(*args):
    pos, test = _set_args(args, "Union")
    if not pos:
        return L()
    ls = _lists(pos, "Union")
    return _set_result([x for l in ls for x in l], test)


@fn("Intersection", area=AREA, status="fixed",
    notes="variadic; canonical order (legacy sorted by string form); SameTest -> f")
def _intersection(*args):
    pos, test = _set_args(args, "Intersection")
    ls = _lists(pos, "Intersection")
    common = [x for x in ls[0] if all(_member_t(x, l, test) for l in ls[1:])]
    return _set_result(common, test)


@fn("Complement", area=AREA, status="fixed",
    notes="Complement[all, e1, e2, ...] in canonical order (legacy: 2 arguments only, sorted by string form); "
          "SameTest -> f")
def _complement(*args):
    pos, test = _set_args(args, "Complement")
    ls = _lists(pos, "Complement")
    rest = [x for l in ls[1:] for x in l]
    return _set_result([x for x in ls[0] if not _member_t(x, rest, test)], test)


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
          "the symbol x were treated as duplicates); DeleteDuplicates[list, test]"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _deldup(a, test=None):
    if test is not None:
        from .wl_common import dedup
        return L(*dedup(list_arg(a, "DeleteDuplicates"), test))
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
    notes="Partition[list, n] and Partition[list, n, d] (incomplete blocks dropped, as Wolfram); "
          "Partition[list, UpTo[n]] (and UpTo[n], d) keeps a shorter final block; legacy truncated non-integer n")
def _partition(a, n, d=None):
    items = list_arg(a, "Partition")
    if isinstance(n, Expr) and n.head == "UpTo" and len(n.args) == 1:
        k = pos_int(n.args[0], "Partition", "block size")
        step = k if d is None else pos_int(d, "Partition", "offset")
        out = []
        for i in range(0, len(items), step):
            out.append(L(*items[i:i + k]))
            if i + k >= len(items):
                break
        return L(*out)
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
    notes="exact running totals (legacy converted to floats and rounded to 10 digits); works on symbols"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
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
    notes="legacy ignored the function and grouped equal values; now GroupBy[list, f] -> <|k -> {...}|> keyed by "
          "f[elem] in order of first appearance; GroupBy[list, f -> g] maps g over each group's elements, "
          "GroupBy[list, spec, red] applies red to each group, GroupBy[list, {f1, f2}] nests; associations group "
          "their values"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _groupby(a, f, red=None):
    from .wl_common import items_of, apply, classes
    from ..kernel_lists import make_assoc
    from ..kernel_lists import is_assoc, rule_value as _rv
    from .wl_common import ev as _ev
    rule_value = lambda r: _rv(r, _ev)  # noqa: E731
    fs = list(f.args) if is_list(f) else [f]
    if is_assoc(a):
        # associations group their key -> value rules by f[value]; each group stays an association
        def go_assoc(rules, fs):
            g0 = fs[0]
            if isinstance(g0, Expr) and g0.head in ("Rule", "RuleDelayed") and len(g0.args) == 2:
                kf, vf = g0.args
            else:
                kf, vf = g0, None
            keys, groups = classes(rules, keyf=lambda r: apply(kf, rule_value(r)))
            out = []
            for k, g in zip(keys, groups):
                g = [Expr("Rule", r.args[0], apply(vf, rule_value(r)) if vf is not None else rule_value(r))
                     for r in g]
                v = go_assoc(g, fs[1:]) if len(fs) > 1 else make_assoc([(r.args[0], r.args[1]) for r in g])
                if red is not None and len(fs) == 1:
                    v = apply(red, v)
                out.append((k, v))
            return make_assoc(out)
        return go_assoc(list(a.args), fs)
    items = items_of(a, "GroupBy")

    def go(xs, fs):
        g0 = fs[0]
        if isinstance(g0, Expr) and g0.head in ("Rule", "RuleDelayed") and len(g0.args) == 2:
            kf, vf = g0.args
        else:
            kf, vf = g0, None
        keys, groups = classes(xs, keyf=lambda e: apply(kf, e))
        out = []
        for k, g in zip(keys, groups):
            vals = [apply(vf, e) for e in g] if vf is not None else g
            if len(fs) > 1:
                v = go(g if vf is None else vals, fs[1:])
            else:
                v = L(*vals)
                if red is not None:
                    v = apply(red, v)
            out.append((k, v))
        return make_assoc(out)
    return go(items, fs)


@fn("Tally", area=AREA, status="promoted",
    notes="{{elem, count}, ...} in order of first appearance (SameQ); Tally[list, test] uses the first "
          "representative of each class"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _tally(a, test=None):
    items = list_arg(a, "Tally")
    if test is not None:
        from .wl_common import classes
        keys, groups = classes(items, test=test)
        return L(*(L(k, len(g)) for k, g in zip(keys, groups)))
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
