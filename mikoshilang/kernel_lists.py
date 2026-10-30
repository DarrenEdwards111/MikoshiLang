"""Kernel list, part and association operations with full Wolfram level / sequence specifications.

Imported at the end of :mod:`mikoshilang.kernel`; it replaces the simpler kernel builtins for
Part, Take, Drop, Map, Apply, Cases, Position, ... with versions that accept Wolfram level
specifications ({n}, {n1, n2}, negative levels, Infinity, All), sequence specifications (Span, UpTo,
{m, n, s}) and associations (which behave as atomic, symbolically indexed lists of values), and
provides the evaluator hooks for associations used as functions (assoc[key]) and operator forms
(Map[f][x], Select[crit][x], ...).
"""
from __future__ import annotations

import functools

import sympy as sp

from .expr import Expr, Symbol
from . import kernel as K

INF = 10 ** 9
L = K.L
is_int = K.is_int
is_list = K.is_list
call = K.call
same = K.same


# ─────────────────────────── helpers ───────────────────────────

def sym(name):
    return Symbol(name)


def is_sym(x, name):
    return isinstance(x, Symbol) and x.name == name


def is_assoc(x):
    return isinstance(x, Expr) and x.head == "Association"


def is_rule(x):
    return isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed") and len(x.args) == 2


def missing(*args):
    return Expr("Missing", *args)


def key_absent(k):
    return missing("KeyAbsent", k)


def atomq(x):
    """AtomQ: numbers (including exact rationals), strings, symbols and associations are atoms."""
    from .formatting import rational_value
    if not isinstance(x, Expr):
        return True
    return x.head == "Association" or rational_value(x) is not None


def assoc_rules(a):
    return [r for r in a.args]


def rule_value(r, ev):
    return ev(r.args[1]) if r.head == "RuleDelayed" else r.args[1]


def assoc_keys(a):
    return [r.args[0] for r in a.args]


def assoc_values(a, ev):
    return [rule_value(r, ev) for r in a.args]


def make_assoc(pairs):
    """Association from (key, value) pairs (or rules), keeping the first position of a repeated key and its
    last value -- Wolfram's rule for repeated keys."""
    out, index = [], {}
    for p in pairs:
        r = p if is_rule(p) else Expr("Rule", p[0], p[1])
        t = K.tag(r.args[0])
        if t in index:
            out[index[t]] = r
        else:
            index[t] = len(out)
            out.append(r)
    return Expr("Association", *out)


def assoc_lookup(a, k, ev, default=None):
    t = K.tag(k)
    for r in a.args:
        if K.tag(r.args[0]) == t:
            return rule_value(r, ev)
    return key_absent(k) if default is None else default


def assoc_has(a, k):
    t = K.tag(k)
    return any(K.tag(r.args[0]) == t for r in a.args)


def int_value(x, fname, what="argument"):
    if is_int(x):
        return int(x)
    raise ValueError(f"{fname}: {what} {x!r} must be an integer")


def need(a, lo, name, hi=None):
    K._need(a, lo, name, hi if hi is not None else lo)


def expr_arg(x, name):
    if not isinstance(x, Expr) or atomq(x) and not is_assoc(x):
        raise ValueError(f"{name}: {x!r} is not a list or compound expression")
    return x


def truth(v):
    return v is True


# children / rebuild that treat an association as one level of values

def children(e, ev=None):
    if is_assoc(e):
        return [r.args[1] if r.head == "Rule" or ev is None else ev(r.args[1]) for r in e.args]
    if isinstance(e, Expr) and not atomq(e):
        return list(e.args)
    return []


def rebuild(e, new):
    if is_assoc(e):
        return Expr("Association", *(Expr(r.head, r.args[0], v) for r, v in zip(e.args, new)))
    return Expr(e.head, *new)


def child_index(e, i):
    """Part index of the i-th (0-based) child: an integer, or Key[k] for associations."""
    return Expr("Key", e.args[i].args[0]) if is_assoc(e) else i + 1


def depth(e):
    """Wolfram Depth: 1 for atoms, 1 + max depth of the elements otherwise (heads ignored)."""
    if is_assoc(e):
        return 1 + max((depth(r.args[1]) for r in e.args), default=0)
    if atomq(e):
        return 1
    return 1 + max((depth(x) for x in e.args), default=0)


# ─────────────────────────── level specifications ───────────────────────────

def level_spec(spec, default, name="level specification"):
    """(lo, hi) with ints (negative = counted from the bottom by depth) or INF."""
    if spec is None:
        return default
    if is_sym(spec, "Infinity"):
        return (1, INF)
    if is_sym(spec, "All"):
        return (0, INF)
    if is_int(spec):
        n = int(spec)
        return (1, n) if n >= 0 else (1, n)
    if is_list(spec) and len(spec.args) in (1, 2):
        vals = []
        for v in spec.args:
            if is_sym(v, "Infinity"):
                vals.append(INF)
            elif is_int(v):
                vals.append(int(v))
            else:
                raise ValueError(f"{name}: {spec!r} is not a valid level specification")
        return (vals[0], vals[0]) if len(vals) == 1 else (vals[0], vals[1])
    raise ValueError(f"{name}: {spec!r} is not a valid level specification")


def in_level(lev, dep, spec):
    lo, hi = spec
    ok_lo = lev >= lo if lo >= 0 else dep <= -lo
    ok_hi = lev <= hi if hi >= 0 else dep >= -hi
    return ok_lo and ok_hi


def max_level(spec):
    lo, hi = spec
    if hi < 0 or lo < 0:
        return INF
    return hi


def walk(e, spec, heads=False, ev=None):
    """Post-order traversal (Wolfram's order): yields (position tuple, subexpression) in the level range.
    With heads=True the head of each compound part is visited (position ... 0) before its elements."""
    maxl = max_level(spec)

    def go(x, lev, pos):
        if lev < maxl and isinstance(x, Expr) and not (atomq(x) and not is_assoc(x)):
            if heads and not is_assoc(x):
                h = Symbol(x.head) if isinstance(x.head, str) else x.head
                yield from go(h, lev + 1, pos + (0,))
            for i, c in enumerate(children(x, ev)):
                yield from go(c, lev + 1, pos + (child_index(x, i),))
        if in_level(lev, depth(x), spec):
            yield pos, x
    yield from go(e, 0, ())


def map_levels(f, e, spec, with_index=False):
    """Rebuild e applying f (a Python callable taking (subexpr, position)) at the given levels, bottom-up."""
    maxl = max_level(spec)

    def go(x, lev, pos):
        d = depth(x)
        if lev < maxl and isinstance(x, Expr) and not (atomq(x) and not is_assoc(x)):
            kids = children(x)
            new = [go(c, lev + 1, pos + (child_index(x, i),)) for i, c in enumerate(kids)]
            x = rebuild(x, new)
        if in_level(lev, d, spec):
            return f(x, pos)
        return x
    return go(e, 0, ())


def pos_expr(pos):
    return L(*pos)


# ─────────────────────────── sequence specifications ───────────────────────────

def seq_indices(spec, n, name, take=True):
    """0-based indices selected by a Wolfram sequence specification for an expression with n elements."""
    if is_sym(spec, "All"):
        return list(range(n))
    if is_sym(spec, "None"):
        return []
    if isinstance(spec, Expr) and spec.head == "UpTo" and len(spec.args) == 1:
        k = spec.args[0]
        if is_sym(k, "Infinity"):
            return list(range(n))
        k = int_value(k, name, "UpTo bound")
        if k < 0:
            raise ValueError(f"{name}: UpTo bound must be non-negative")
        return list(range(min(k, n)))
    if is_int(spec):
        k = int(spec)
        if abs(k) > n:
            raise ValueError(f"{name}: cannot {'take' if take else 'drop'} {k} elements from a list of length {n}")
        return list(range(k)) if k >= 0 else list(range(n + k, n))
    if is_sym(spec, "Infinity"):
        return list(range(n))
    if isinstance(spec, Expr) and spec.head in ("List", "Span") and 1 <= len(spec.args) <= 3:
        vals = list(spec.args)
        if spec.head == "List" and len(vals) == 1:
            k = int_value(vals[0], name, "position")
            k0 = k - 1 if k > 0 else n + k
            if not 0 <= k0 < n:
                raise ValueError(f"{name}: position {k} is out of range for length {n}")
            return [k0]
        if spec.head == "Span" and len(vals) == 1:
            raise ValueError(f"{name}: invalid Span")
        m = vals[0]
        e = vals[1]
        s = int_value(vals[2], name, "step") if len(vals) == 3 else 1
        if s == 0:
            raise ValueError(f"{name}: step cannot be zero")
        mi = 1 if is_sym(m, "All") else int_value(m, name, "start")
        if is_sym(e, "All"):
            ei = n if s > 0 else 1
            if s < 0 and spec.head == "Span" and is_sym(e, "All"):
                ei = 1
        else:
            ei = int_value(e, name, "end")
        m0 = mi - 1 if mi > 0 else n + mi
        e0 = ei - 1 if ei > 0 else n + ei
        if s > 0:
            if not (0 <= m0 <= n and -1 <= e0 < n) or e0 < m0 - 1:
                raise ValueError(f"{name}: cannot take positions {mi} through {ei} from a list of length {n}")
            return list(range(m0, e0 + 1, s))
        if not (0 <= e0 <= n and -1 <= m0 < n) or m0 < e0 - 1:
            raise ValueError(f"{name}: cannot take positions {mi} through {ei} from a list of length {n}")
        return list(range(m0, e0 - 1, s))
    raise ValueError(f"{name}: {spec!r} is not a valid sequence specification")


def take_level(x, specs, name, ev, drop=False):
    if not specs:
        return x
    if not isinstance(x, Expr) or atomq(x) and not is_assoc(x):
        raise ValueError(f"{name}: cannot {'drop' if drop else 'take'} from {x!r} (not deep enough)")
    n = len(x.args)
    idx = seq_indices(specs[0], n, name, take=not drop)
    if drop:
        sel = set(idx)
        idx = [i for i in range(n) if i not in sel]
    kept = [x.args[i] for i in idx]
    if is_assoc(x):
        if len(specs) > 1:
            kept = [Expr(r.head, r.args[0], take_level(r.args[1], specs[1:], name, ev, drop)) for r in kept]
        return Expr("Association", *kept)
    return Expr(x.head, *(take_level(y, specs[1:], name, ev, drop) for y in kept))


# ─────────────────────────── Part ───────────────────────────

def _part_err(i, x):
    from .formatting import input_form
    return ValueError(f"Part {input_form(i)} of {input_form(x)} does not exist")


def part(x, idx, ev):
    if not idx:
        return x
    i = idx[0]
    rest = idx[1:]
    if is_assoc(x):
        return _assoc_part(x, i, rest, ev)
    if not isinstance(x, Expr) or atomq(x):
        raise ValueError("Part specification is longer than depth of object")
    n = len(x.args)
    if is_sym(i, "All"):
        return Expr(x.head, *(part(y, rest, ev) for y in x.args))
    if isinstance(i, Expr) and i.head == "Span":
        sel = seq_indices(i, n, "Part")
        return Expr(x.head, *(part(x.args[k], rest, ev) for k in sel))
    if is_list(i):
        out = []
        for j in i.args:
            out.append(part(x, [j] + list(rest), ev))
        return Expr(x.head, *out)
    if is_int(i):
        k = int(i)
        if k == 0:
            h = Symbol(x.head) if isinstance(x.head, str) else x.head
            return part(h, rest, ev) if rest else h
        k0 = k - 1 if k > 0 else n + k
        if not 0 <= k0 < n:
            raise _part_err(i, x)
        return part(x.args[k0], rest, ev)
    if isinstance(i, Expr) and i.head == "Key" or isinstance(i, str):
        raise ValueError(f"Part: key {i!r} can only be used with an association")
    raise ValueError(f"Part: {i!r} is not a valid part specification")


def _assoc_part(a, i, rest, ev):
    if is_sym(i, "All"):
        return Expr("Association", *(Expr(r.head, r.args[0], part(rule_value(r, ev), rest, ev)) for r in a.args))
    if isinstance(i, Expr) and i.head == "Span":
        sel = seq_indices(i, len(a.args), "Part")
        return Expr("Association", *(Expr("Rule", a.args[k].args[0], part(rule_value(a.args[k], ev), rest, ev))
                                     for k in sel))
    if is_list(i):
        rules = []
        for j in i.args:
            k = _assoc_key_of(a, j)
            if k is None:
                raise _part_err(j, a)
            rules.append(Expr("Rule", k, part(assoc_lookup(a, k, ev), rest, ev)))
        return Expr("Association", *rules)
    if is_int(i):
        k = int(i)
        n = len(a.args)
        if k == 0:
            return part(Symbol("Association"), rest, ev) if rest else Symbol("Association")
        k0 = k - 1 if k > 0 else n + k
        if not 0 <= k0 < n:
            raise _part_err(i, a)
        return part(rule_value(a.args[k0], ev), rest, ev)
    key = i.args[0] if isinstance(i, Expr) and i.head == "Key" and len(i.args) == 1 else i if isinstance(i, str) else None
    if key is None:
        raise ValueError(f"Part: {i!r} is not a valid part specification for an association")
    if not assoc_has(a, key):
        return key_absent(key)
    return part(assoc_lookup(a, key, ev), rest, ev)


def _assoc_key_of(a, j):
    if is_int(j):
        k = int(j)
        n = len(a.args)
        k0 = k - 1 if k > 0 else n + k
        return a.args[k0].args[0] if 0 <= k0 < n else None
    key = j.args[0] if isinstance(j, Expr) and j.head == "Key" and len(j.args) == 1 else j if isinstance(j, str) else None
    return key if key is not None and assoc_has(a, key) else None


# ─────────────────────────── builtins ───────────────────────────

B = K._BUILTINS


def builtin(*names):
    def deco(fn):
        for n in names:
            B[n] = fn
        return fn
    return deco


@builtin("List")
def _b_list(a, ev, expr):
    """{..., Nothing, ...} drops Nothing; Splice[{...}] is spliced into lists."""
    if not any(isinstance(x, (Symbol, Expr)) and (is_sym(x, "Nothing") or isinstance(x, Expr) and x.head == "Splice")
               for x in a):
        return expr
    out = []
    for x in a:
        if is_sym(x, "Nothing"):
            continue
        if isinstance(x, Expr) and x.head == "Splice" and len(x.args) in (1, 2) and is_list(x.args[0]):
            if len(x.args) == 2 and not (is_sym(x.args[1], "List") or is_sym(x.args[1], "All")):
                out.append(x)
                continue
            out.extend(x.args[0].args)
            continue
        out.append(x)
    return L(*out)


@builtin("Length")
def _b_length(a, ev, expr):
    need(a, 1, "Length")
    x = a[0]
    if is_assoc(x):
        return len(x.args)
    return 0 if atomq(x) else len(x.args)


@builtin("AtomQ")
def _b_atomq(a, ev, expr):
    need(a, 1, "AtomQ")
    return atomq(a[0])


@builtin("Part")
def _b_part(a, ev, expr):
    if len(a) < 2:
        raise ValueError("Part expects expr[[i, ...]]")
    return part(a[0], list(a[1:]), ev)


def _first_last(a, ev, name, k):
    need(a, 1, name, 2)
    x = a[0]
    if is_assoc(x):
        if x.args:
            return rule_value(x.args[k], ev)
    elif not atomq(x) and x.args:
        return x.args[k]
    if len(a) == 2:
        return a[1]
    raise ValueError(f"{name}: {x!r} has no elements")


@builtin("First")
def _b_first(a, ev, expr):
    return _first_last(a, ev, "First", 0)


@builtin("Last")
def _b_last(a, ev, expr):
    return _first_last(a, ev, "Last", -1)


def _rest_most(a, name, sl):
    need(a, 1, name)
    x = a[0]
    if atomq(x) and not is_assoc(x):
        raise ValueError(f"{name}: {x!r} has no elements")
    if not x.args:
        raise ValueError(f"{name}: cannot take the {name.lower()} of an expression with no elements")
    return Expr(x.head, *x.args[sl])


@builtin("Rest")
def _b_rest(a, ev, expr):
    return _rest_most(a, "Rest", slice(1, None))


@builtin("Most")
def _b_most(a, ev, expr):
    return _rest_most(a, "Most", slice(None, -1))


def _assoc_add(x, new, front=False):
    items = []
    for n in (new.args if is_list(new) else (new.args if is_assoc(new) else (new,))):
        if not is_rule(n):
            raise ValueError("Append/Prepend to an association needs a rule, a list of rules or an association")
        items.append(n)
    return make_assoc(items + list(x.args) if front else list(x.args) + items)


@builtin("Append")
def _b_append(a, ev, expr):
    need(a, 2, "Append")
    x = a[0]
    if is_assoc(x):
        return _assoc_add(x, a[1])
    if atomq(x):
        raise ValueError(f"Append: {x!r} is not a list or compound expression")
    return Expr(x.head, *x.args, a[1])


@builtin("Prepend")
def _b_prepend(a, ev, expr):
    need(a, 2, "Prepend")
    x = a[0]
    if is_assoc(x):
        return _assoc_add(x, a[1], front=True)
    if atomq(x):
        raise ValueError(f"Prepend: {x!r} is not a list or compound expression")
    return Expr(x.head, a[1], *x.args)


@builtin("Join")
def _b_join(a, ev, expr):
    if not a:
        return L()
    lev = 1
    args = list(a)
    if len(args) >= 2 and (is_int(args[-1]) or is_sym(args[-1], "Infinity")) and not isinstance(args[-1], bool):
        lev = INF if is_sym(args[-1], "Infinity") else int(args[-1])
        args = args[:-1]
        if lev < 1:
            raise ValueError("Join: level must be a positive integer")
    if all(is_assoc(x) for x in args):
        return make_assoc([r for x in args for r in x.args])
    heads = {x.head if isinstance(x, Expr) and not atomq(x) else None for x in args}
    if len(heads) != 1 or None in heads:
        raise ValueError("Join: all arguments must be lists (or expressions) with the same head")
    return _join_level(args, lev, len(args))


def _join_level(xs, lev, n):
    if lev == 1:
        return Expr(xs[0].head, *(y for x in xs for y in x.args))
    lens = {len(x.args) for x in xs}
    if len(lens) != 1:
        raise ValueError("Join: the expressions do not have the same length at the requested level")
    return Expr(xs[0].head, *(_join_level([x.args[i] for x in xs], lev - 1, n) for i in range(lens.pop())))


@builtin("Reverse")
def _b_reverse(a, ev, expr):
    need(a, 1, "Reverse", 2)
    x = a[0]
    if atomq(x) and not is_assoc(x):
        raise ValueError(f"Reverse: {x!r} is not a list or compound expression")
    if len(a) == 1:
        levels = {1}
    else:
        s = a[1]
        if is_int(s):
            levels = {int(s)}
        elif is_list(s) and all(is_int(v) for v in s.args):
            levels = {int(v) for v in s.args}
        elif is_sym(s, "All"):
            levels = set(range(1, depth(x)))
        else:
            raise ValueError("Reverse: level must be an integer or a list of integers")

    def rev(e, lev):
        if atomq(e) and not is_assoc(e):
            return e
        args = list(e.args)
        if is_assoc(e):
            if lev in levels:
                args.reverse()
            return Expr("Association", *(Expr(r.head, r.args[0], rev(r.args[1], lev + 1)) for r in args))
        args = [rev(y, lev + 1) for y in args]
        if lev in levels:
            args.reverse()
        return Expr(e.head, *args)
    return rev(x, 1)


@builtin("Take")
def _b_take(a, ev, expr):
    if len(a) < 2:
        raise ValueError("Take expects Take[list, spec, ...]")
    return take_level(a[0], list(a[1:]), "Take", ev)


@builtin("Drop")
def _b_drop(a, ev, expr):
    if len(a) < 2:
        raise ValueError("Drop expects Drop[list, spec, ...]")
    return take_level(a[0], list(a[1:]), "Drop", ev, drop=True)


def canonical_key(x, name="Sort"):
    if not K._sortable_canonically(x):
        raise NotImplementedError(f"{name}: canonical ordering of general symbolic expressions is not implemented; "
                                  "give an ordering function")
    return K.order_key(x)


def sort_items(items, ev, p=None, name="Sort", key=lambda v: v):
    """Sort items by canonical order of key(item), or with the ordering function p (Wolfram: p[a, b] is True
    or 1 when a should come first)."""
    if p is None:
        return sorted(items, key=lambda v: canonical_key(key(v), name))

    def cmp(u, v):
        r = ev(call(p, key(u), key(v)))
        if r is True:
            return -1
        if is_int(r) and int(r) in (-1, 0, 1):          # Order-style comparison functions
            return -int(r)
        if r is False:
            return 1 if ev(call(p, key(v), key(u))) is True else 0
        return 0
    return sorted(items, key=functools.cmp_to_key(cmp))


@builtin("Sort")
def _b_sort(a, ev, expr):
    need(a, 1, "Sort", 2)
    x = a[0]
    p = a[1] if len(a) == 2 else None
    if is_assoc(x):
        pairs = [(r, rule_value(r, ev)) for r in x.args]
        return Expr("Association", *(r for r, _ in sort_items(pairs, ev, p, "Sort", key=lambda t: t[1])))
    if atomq(x):
        raise ValueError(f"Sort: {x!r} is not a list or compound expression")
    return Expr(x.head, *sort_items(list(x.args), ev, p))


@builtin("Total")
def _b_total(a, ev, expr):
    need(a, 1, "Total", 2)
    x = a[0]
    if is_assoc(x):
        x = L(*assoc_values(x, ev))
    if atomq(x):
        raise ValueError(f"Total: {x!r} is not a list")
    if len(a) == 1:
        levels = (1, 1)
    else:
        s = a[1]
        if is_sym(s, "Infinity"):
            levels = (1, INF)
        elif is_int(s) and int(s) >= 1:
            levels = (1, int(s))
        elif is_list(s) and len(s.args) in (1, 2) and all(is_int(v) and int(v) >= 1 for v in s.args):
            levels = (int(s.args[0]), int(s.args[-1]))
        else:
            raise ValueError("Total: level must be a positive integer, {n}, {n1, n2} or Infinity")

    def tot(e, lev):
        if lev > levels[1] or not is_list(e):
            return e
        parts = [tot(y, lev + 1) for y in e.args]
        if lev >= levels[0]:
            return ev(Expr("Plus", *parts)) if parts else 0
        return L(*parts)
    return tot(x, 1)


@builtin("Flatten")
def _b_flatten(a, ev, expr):
    need(a, 1, "Flatten", 3)
    x = a[0]
    if is_assoc(x):
        x = L(*assoc_values(x, ev))
    if atomq(x):
        raise ValueError(f"Flatten: {x!r} is not a list or compound expression")
    h = x.head
    if len(a) == 3:
        hd = a[2]
        h = hd.name if isinstance(hd, Symbol) else hd
    if len(a) >= 2 and is_list(a[1]):
        return _flatten_dims(x, a[1])
    n = INF
    if len(a) >= 2:
        if is_sym(a[1], "Infinity"):
            n = INF
        elif is_int(a[1]) and int(a[1]) >= 0:
            n = int(a[1])
        else:
            raise ValueError("Flatten: level must be a non-negative integer, Infinity or a list of dimension groups")

    def flat(e, d):
        out = []
        for y in e.args:
            if d > 0 and isinstance(y, Expr) and y.head == h and not atomq(y):
                out.extend(flat(y, d - 1))
            else:
                out.append(y)
        return out
    return Expr(x.head, *flat(x, n))


def array_dims(x, head="List"):
    """Full array dimensions (Wolfram Dimensions)."""
    dims = []
    level = [x]
    while True:
        if not all(isinstance(e, Expr) and not atomq(e) and e.head == level[0].head for e in level):
            break
        hd = level[0].head
        if head is not None and hd != head and head != "*":
            break
        lens = {len(e.args) for e in level}
        if len(lens) != 1:
            break
        n = lens.pop()
        dims.append(n)
        if n == 0:
            break
        level = [y for e in level for y in e.args]
    return dims


def _flatten_dims(x, spec):
    groups = []
    for g in spec.args:
        if is_int(g):
            groups.append([int(g)])
        elif is_list(g) and g.args and all(is_int(v) for v in g.args):
            groups.append([int(v) for v in g.args])
        else:
            raise ValueError("Flatten: dimension specification must be a list of lists of positive integers")
    used = [v for g in groups for v in g]
    if len(set(used)) != len(used) or any(v < 1 for v in used):
        raise ValueError("Flatten: each level may appear at most once in the dimension specification")
    dims = array_dims(x, x.head)
    top = max(used)
    if len(dims) < top:
        raise NotImplementedError("Flatten with dimension groups is implemented for full (rectangular) arrays only")
    rest = [v for v in range(1, top + 1) if v not in used]
    groups = groups + [[v] for v in rest]
    import itertools

    def get(idx):
        e = x
        for i in idx:
            e = e.args[i]
        return e

    def build(gi, fixed):
        if gi == len(groups):
            idx = [fixed[v] for v in range(1, top + 1)]
            return get(idx)
        g = groups[gi]
        out = []
        for combo in itertools.product(*(range(dims[v - 1]) for v in g)):
            f2 = dict(fixed)
            f2.update(zip(g, combo))
            out.append(build(gi + 1, f2))
        return Expr(x.head, *out)
    return build(0, {})


@builtin("Transpose")
def _b_transpose(a, ev, expr):
    need(a, 1, "Transpose", 2)
    x = a[0]
    if not is_list(x):
        raise ValueError("Transpose expects a list (an array)")
    dims = array_dims(x)
    if len(a) == 1:
        if len(dims) < 2:
            if len(dims) == 1 and dims[0] == 0:
                return L()
            raise ValueError("Transpose expects an array of depth at least 2 (a list of equal-length lists)")
        perm = [2, 1]
    else:
        p = a[1]
        if not (is_list(p) and p.args and all(is_int(v) and int(v) >= 1 for v in p.args)):
            raise ValueError("Transpose: the permutation must be a list of positive integers")
        perm = [int(v) for v in p.args]
    k = len(perm)
    if len(dims) < k:
        raise ValueError(f"Transpose: the array does not have {k} levels (dimensions {dims})")
    rank = max(perm)
    if sorted(set(perm)) != list(range(1, rank + 1)):
        raise ValueError("Transpose: the permutation must use every level 1..n")
    rdims = [min(dims[i] for i in range(k) if perm[i] == r) for r in range(1, rank + 1)]

    def get(idx):
        e = x
        for i in idx:
            e = e.args[i]
        return e

    def build(j):
        if len(j) == rank:
            return get([j[perm[i] - 1] for i in range(k)])
        return L(*(build(j + [t]) for t in range(rdims[len(j)])))
    return build([])


@builtin("Cases")
def _b_cases(a, ev, expr):
    need(a, 2, "Cases", 4)
    x = a[0]
    pat, rhs = K._pattern_and_rhs(a[1])
    spec = level_spec(a[2] if len(a) >= 3 else None, (1, 1))
    limit = _limit(a[3], "Cases") if len(a) == 4 else None
    out = []
    if limit == 0:
        return L()
    for _, sub in walk(x, spec, ev=ev):
        b = K.first_match(pat, sub, ev)
        if b is not None:
            out.append(sub if rhs is None else ev(K.substitute(rhs, b)))
            if limit is not None and len(out) >= limit:
                break
    return L(*out)


def _limit(n, name):
    if is_sym(n, "Infinity"):
        return None
    if is_int(n) and int(n) >= 0:
        return int(n)
    raise ValueError(f"{name}: the maximum count must be a non-negative integer or Infinity")


@builtin("DeleteCases")
def _b_deletecases(a, ev, expr):
    need(a, 2, "DeleteCases", 4)
    x = a[0]
    if atomq(x) and not is_assoc(x):
        return x
    spec = level_spec(a[2] if len(a) >= 3 else None, (1, 1))
    limit = _limit(a[3], "DeleteCases") if len(a) == 4 else None
    count = [0]
    maxl = max_level(spec)

    def go(e, lev):
        if lev >= maxl or atomq(e) and not is_assoc(e) or not isinstance(e, Expr):
            return e
        kids = children(e, ev)
        keep = []
        for i, c in enumerate(kids):
            c2 = go(c, lev + 1)
            if (limit is None or count[0] < limit) and in_level(lev + 1, depth(c), spec) and \
                    K.first_match(a[1], c, ev) is not None:
                count[0] += 1
                continue
            keep.append((i, c2))
        if is_assoc(e):
            return Expr("Association", *(Expr(e.args[i].head, e.args[i].args[0], c) for i, c in keep))
        return Expr(e.head, *(c for _, c in keep))
    return go(x, 0)


@builtin("Count")
def _b_count(a, ev, expr):
    need(a, 2, "Count", 3)
    spec = level_spec(a[2] if len(a) == 3 else None, (1, 1))
    return sum(1 for _, sub in walk(a[0], spec, ev=ev) if K.first_match(a[1], sub, ev) is not None)


def _heads_option(a, name):
    """Strip a trailing Heads -> True/False option; returns (args, heads)."""
    a = list(a)
    heads = True
    if a and is_rule(a[-1]) and is_sym(a[-1].args[0], "Heads"):
        v = a.pop().args[1]
        if v is not True and v is not False:
            raise ValueError(f"{name}: Heads must be True or False")
        heads = v
    return a, heads


@builtin("Position")
def _b_position(a, ev, expr):
    a, heads = _heads_option(a, "Position")
    need(a, 2, "Position", 4)
    spec = level_spec(a[2] if len(a) >= 3 else None, (0, INF))
    limit = _limit(a[3], "Position") if len(a) == 4 else None
    out = []
    if limit == 0:
        return L()
    for pos, sub in walk(a[0], spec, heads=heads, ev=ev):
        if K.first_match(a[1], sub, ev) is not None:
            out.append(pos_expr(pos))
            if limit is not None and len(out) >= limit:
                break
    return L(*out)


@builtin("MemberQ")
def _b_memberq(a, ev, expr):
    need(a, 2, "MemberQ", 3)
    spec = level_spec(a[2] if len(a) == 3 else None, (1, 1))
    if atomq(a[0]) and not is_assoc(a[0]):
        return False
    return any(K.first_match(a[1], sub, ev) is not None for _, sub in walk(a[0], spec, ev=ev))


@builtin("FreeQ")
def _b_freeq(a, ev, expr):
    a, heads = _heads_option(a, "FreeQ")
    need(a, 2, "FreeQ", 3)
    spec = level_spec(a[2] if len(a) == 3 else None, (0, INF))
    return not any(K.first_match(a[1], sub, ev) is not None for _, sub in walk(a[0], spec, heads=heads, ev=ev))


@builtin("Map")
def _b_map(a, ev, expr):
    need(a, 2, "Map", 3)
    f, x = a[0], a[1]
    spec = level_spec(a[2] if len(a) == 3 else None, (1, 1))
    if atomq(x) and not is_assoc(x) and spec[0] > 0:
        return x
    return ev(map_levels(lambda e, pos: call(f, e), x, spec))


@builtin("MapAll")
def _b_mapall(a, ev, expr):
    need(a, 2, "MapAll")
    return ev(map_levels(lambda e, pos: call(a[0], e), a[1], (0, INF)))


@builtin("MapIndexed")
def _b_mapindexed(a, ev, expr):
    need(a, 2, "MapIndexed", 3)
    f, x = a[0], a[1]
    spec = level_spec(a[2] if len(a) == 3 else None, (1, 1))
    if atomq(x) and not is_assoc(x) and spec[0] > 0:
        return x
    return ev(map_levels(lambda e, pos: call(f, e, pos_expr(pos)), x, spec))


@builtin("Apply")
def _b_apply(a, ev, expr):
    need(a, 2, "Apply", 3)
    f, x = a[0], a[1]
    spec = level_spec(a[2] if len(a) == 3 else None, (0, 0))

    def ap(e, pos):
        if is_assoc(e):
            return call(f, *assoc_values(e, ev))
        if isinstance(e, Expr) and not atomq(e):
            return call(f, *e.args)
        return e
    return ev(map_levels(ap, x, spec))


@builtin("Select")
def _b_select(a, ev, expr):
    need(a, 2, "Select", 3)
    x = a[0]
    limit = None
    if len(a) == 3:
        limit = _limit(a[2], "Select")
    if is_assoc(x):
        out = []
        for r in x.args:
            if limit is not None and len(out) >= limit:
                break
            if ev(call(a[1], rule_value(r, ev))) is True:
                out.append(r)
        return Expr("Association", *out)
    if atomq(x):
        raise ValueError(f"Select: {x!r} is not a list or compound expression")
    out = []
    for y in x.args:
        if limit is not None and len(out) >= limit:
            break
        if ev(call(a[1], y)) is True:
            out.append(y)
    return Expr(x.head, *out)


@builtin("Thread")
def _b_thread(a, ev, expr):
    need(a, 1, "Thread", 3)
    x = a[0]
    if atomq(x):
        return x
    h = "List"
    if len(a) >= 2:
        hh = a[1]
        h = hh.name if isinstance(hh, Symbol) else hh
    args = list(x.args)
    sel = range(len(args))
    if len(a) == 3:
        sel = seq_indices(a[2], len(args), "Thread")
    sel = set(sel)
    lens = {len(y.args) for i, y in enumerate(args) if i in sel and isinstance(y, Expr) and y.head == h
            and not atomq(y)}
    if not lens:
        return x
    if len(lens) != 1:
        raise ValueError("Thread: objects of unequal length cannot be combined")
    n = lens.pop()
    th = lambda y, i, k: y.args[k] if i in sel and isinstance(y, Expr) and y.head == h and not atomq(y) else y  # noqa
    return ev(Expr(h, *(Expr(x.head, *(th(y, i, k) for i, y in enumerate(args))) for k in range(n))))


@builtin("Normal")
def _b_normal(a, ev, expr):
    need(a, 1, "Normal", 2)
    x = a[0]
    heads = None
    if len(a) == 2:
        hs = a[1].args if is_list(a[1]) else (a[1],)
        heads = {h.name for h in hs if isinstance(h, Symbol)}
    if (heads is None or "Series" in heads or "SeriesData" in heads) and \
            K.contains(x, lambda y: isinstance(y, Expr) and y.head == "O"):
        return K.from_sp(K.to_sp(x).removeO())
    if heads is not None and "Association" not in heads:
        return x

    def norm(e):
        if is_assoc(e):
            # Wolfram: an association becomes its list of rules; values (even associations) are left alone
            return L(*e.args)
        if isinstance(e, Expr) and not atomq(e):
            h = norm(e.head) if isinstance(e.head, Expr) else e.head
            return Expr(h, *(norm(y) for y in e.args))
        return e
    return norm(x)


@builtin("Range")
def _b_range(a, ev, expr):
    need(a, 1, "Range", 3)
    if len(a) == 1:
        lo, hi, step = 1, a[0], 1
    elif len(a) == 2:
        lo, hi, step = a[0], a[1], 1
    else:
        lo, hi, step = a
    if all(is_int(v) for v in (lo, hi, step)):
        lo, hi, step = int(lo), int(hi), int(step)
        if step == 0:
            raise ValueError("Range: step cannot be zero")
        it = range(lo, hi + (1 if step > 0 else -1), step)
        if len(it) > K.MAX_TABLE:
            raise ValueError(f"Range exceeds {K.MAX_TABLE} element limit")
        return L(*it)
    vals = [K.to_sp(v) for v in (lo, hi, step)]
    if any(not isinstance(v, sp.Basic) for v in vals):
        raise ValueError("Range: bounds and step must be numbers or symbolic expressions")
    if vals[2] == 0:
        raise ValueError("Range: step cannot be zero")
    d = sp.simplify((vals[1] - vals[0]) / vals[2])
    if not d.is_number or d.is_real is False:
        raise ValueError("Range: (max - min)/step must be a real number")
    inexact = any(isinstance(v, float) or (isinstance(v, sp.Basic) and v.has(sp.Float)) for v in (lo, hi, step))
    dv = float(d)
    n = int(sp.floor(d)) if not inexact else int(__import__("math").floor(dv + 1e-10 * max(1.0, abs(dv))))
    if n < 0:
        return L()
    if n + 1 > K.MAX_TABLE:
        raise ValueError(f"Range exceeds {K.MAX_TABLE} element limit")
    if inexact and all(v.is_number for v in vals):
        a0, st = float(vals[0]), float(vals[2])
        return L(*(a0 + k * st for k in range(n + 1)))
    return L(*(ev(Expr("Plus", lo, Expr("Times", k, step))) for k in range(n + 1)))


# ─────────────────────────── evaluator hook: compound heads ───────────────────────────

# name -> (number of operator arguments, where the operand goes: "first" or "last")
OPERATOR_FORMS = {}


def operator_form(name, nops=1, operand="first"):
    OPERATOR_FORMS[name] = (nops, operand)


for _n in ("Select", "SelectFirst", "Cases", "DeleteCases", "Count", "Position", "FirstCase", "FirstPosition",
           "MemberQ", "FreeQ", "MatchQ", "SortBy", "GroupBy", "Replace", "ReplaceAll", "ReplacePart", "Append",
           "Prepend", "Delete", "Extract", "StringReplace", "StringCases", "StringMatchQ", "StringContainsQ",
           "StringStartsQ", "StringEndsQ", "StringDelete", "StringFreeQ", "StringPosition",
           "KeyTake", "KeyDrop", "KeySelect", "KeySortBy", "Merge", "CountsBy", "MaximalBy", "MinimalBy",
           "TakeLargest", "TakeSmallest", "DeleteDuplicatesBy", "AnyTrue", "AllTrue", "NoneTrue", "ContainsAll",
           "ContainsAny", "ContainsNone", "ContainsOnly", "ContainsExactly", "KeyExistsQ", "KeyMemberQ", "KeyFreeQ",
           "Lookup", "SequenceReplace", "Between"):
    operator_form(_n, 1, "first")          # f[op][x] -> f[x, op]
for _n in ("Map", "Apply", "MapIndexed", "MapApply", "Scan", "KeyMap", "KeyValueMap", "AssociationMap"):
    operator_form(_n, 1, "last")           # f[op][x] -> f[op, x]
operator_form("MapAt", 2, "middle")         # MapAt[f, pos][expr] -> MapAt[f, expr, pos]
operator_form("Insert", 2, "first")         # Insert[elem, n][list] -> Insert[list, elem, n]
operator_form("TakeLargestBy", 2, "first")
operator_form("TakeSmallestBy", 2, "first")

# operators that are not curried forms of a two-argument function
_COMPARATORS = {"GreaterThan": "Greater", "LessThan": "Less", "GreaterEqualThan": "GreaterEqual",
                "LessEqualThan": "LessEqual", "EqualTo": "Equal", "UnequalTo": "Unequal"}
_CURRIED_ONLY = {"SameAs"} | set(_COMPARATORS)


def apply_compound_head(head, args, ev):
    """f[ops...][args...] for associations, Key, compositions and operator forms; UNHANDLED otherwise."""
    h = head.head
    if not isinstance(h, str):
        return K.UNHANDLED
    if h == "Association":
        if len(args) != 1:
            raise ValueError("assoc[key] takes exactly one key")
        return assoc_lookup(head, args[0], ev)
    if h == "Key" and len(head.args) == 1:
        if len(args) != 1:
            return K.UNHANDLED
        x = args[0]
        if is_assoc(x):
            return assoc_lookup(x, head.args[0], ev)
        if is_list(x) and all(is_rule(r) for r in x.args):
            return assoc_lookup(make_assoc(x.args), head.args[0], ev)
        return K.UNHANDLED
    if h == "Composition":
        if not head.args:
            return args[0] if len(args) == 1 else Expr("Sequence", *args)
        r = ev(call(head.args[-1], *args))
        for f in reversed(head.args[:-1]):
            r = ev(call(f, r))
        return r
    if h == "RightComposition":
        if not head.args:
            return args[0] if len(args) == 1 else Expr("Sequence", *args)
        r = ev(call(head.args[0], *args))
        for f in head.args[1:]:
            r = ev(call(f, r))
        return r
    if h in _COMPARATORS:
        if len(head.args) != 1 or len(args) != 1:
            return K.UNHANDLED
        return ev(Expr(_COMPARATORS[h], args[0], head.args[0]))
    if h == "SameAs":
        if len(head.args) != 1 or len(args) != 1:
            return K.UNHANDLED
        return same(args[0], head.args[0])
    spec = OPERATOR_FORMS.get(h)
    if spec is None or len(args) != 1:
        return K.UNHANDLED
    nops, where = spec
    if len(head.args) != nops:
        return K.UNHANDLED
    if where == "first":
        return ev(Expr(h, args[0], *head.args))
    if where == "last":
        return ev(Expr(h, *head.args, args[0]))
    return ev(Expr(h, head.args[0], args[0], *head.args[1:]))


# ─────────────────────────── structural heads handled by the evaluator ───────────────────────────

K.STRUCTURAL_HEADS = tuple("""List Rule RuleDelayed Function Slot SlotSequence Blank BlankSequence BlankNullSequence
Pattern Condition PatternTest Alternatives HoldPattern Hold HoldForm HoldComplete True False Null All None
Integer Real Rational Complex String Pi E I Infinity ComplexInfinity Indeterminate GoldenRatio EulerGamma
HoldAll HoldFirst HoldRest Listable Flat Orderless OneIdentity Protected NumericFunction SequenceHold
Association Nothing Splice Span Key Missing UpTo Composition RightComposition GreaterThan LessThan
GreaterEqualThan LessEqualThan EqualTo UnequalTo SameAs Except Repeated RepeatedNull Optional""".split())
