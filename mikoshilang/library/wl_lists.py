"""Batch 1 -- core list manipulation (Wolfram coverage roadmap, docs/wolfram_coverage_roadmap.md).

Every function here follows the Wolfram Language reference semantics and is checked by differential tests
in tests/library/test_wolfram_batch1.py (Wolfram documentation examples, with Python reference
implementations -- itertools, collections, NumPy -- for the structural ones).
"""
from __future__ import annotations

import contextvars
import itertools
import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, Unevaluated
from .wl_common import (L, ev, apply, true, is_sym, is_list, is_int, is_assoc, is_rule, same, items_of, list_items,
                        canonical, tester, dedup, classes, real_number, int_value, upto, split_options)

AREA = "lists (batch 1)"
ADDED = dict(area=AREA, status="added")


def _head_of(x):
    return x.head if isinstance(x, Expr) else "List"


def _rebuild(x, items):
    return Expr(_head_of(x), *items)


def _seq_spec_list(spec, fname):
    """{n1, n2, ...} dims or a single n."""
    if is_int(spec):
        return [int_value(spec, fname, "dimension", 0)]
    if is_list(spec) and all(is_int(v) for v in spec.args):
        return [int_value(v, fname, "dimension", 0) for v in spec.args]
    raise ValueError(f"{fname}: dimensions must be a non-negative integer or a list of them")


# ─────────────────────────── construction ───────────────────────────

@fn("Array", **ADDED, notes="Array[f, n], Array[f, {n1, n2, ...}], origin r or {r1, ...}, {a, b} ranges and head h")
def _array(f, dims, origin=None, head=None):
    ds = _seq_spec_list(dims, "Array")
    k = len(ds)
    if origin is None:
        starts = [(1, 1)] * k
    elif is_list(origin):
        o = list(origin.args)
        if len(o) == k and all(is_list(v) and len(v.args) == 2 for v in o):
            starts = [("range", v.args[0], v.args[1]) for v in o]
        elif k == 1 and len(o) == 2:
            starts = [("range", o[0], o[1])]
        elif len(o) == k:
            starts = [(v, 1) for v in o]
        else:
            raise ValueError("Array: the origin specification does not match the dimensions")
    else:
        starts = [(origin, 1)] * k
    if math.prod(ds) > K.MAX_TABLE:
        raise ValueError("Array: too many elements")

    def values(i):
        s = starts[i]
        n = ds[i]
        if s[0] == "range":
            a, b = s[1], s[2]
            if n == 1:
                return [a]
            return [ev(Expr("Plus", a, Expr("Times", Expr("Times", j, Expr("Power", n - 1, -1)),
                                              Expr("Plus", b, Expr("Times", -1, a))))) for j in range(n)]
        return [ev(Expr("Plus", s[0], j)) for j in range(n)]
    vals = [values(i) for i in range(k)]
    h = "List" if head is None else (head.name if isinstance(head, Symbol) else head)

    def build(i, idx):
        if i == k:
            return K.call(f, *idx)
        return Expr(h, *(build(i + 1, idx + [v]) for v in vals[i]))
    return ev(build(0, []))


@fn("ConstantArray", **ADDED, notes="ConstantArray[c, n] and ConstantArray[c, {n1, n2, ...}]")
def _constarray(c, dims):
    ds = _seq_spec_list(dims, "ConstantArray")
    if math.prod(ds) > K.MAX_TABLE:
        raise ValueError("ConstantArray: too many elements")

    def build(i):
        return c if i == len(ds) else L(*(build(i + 1) for _ in range(ds[i])))
    return build(0)


@fn("Subdivide", **ADDED, notes="Subdivide[n], Subdivide[xmax, n], Subdivide[xmin, xmax, n]: n + 1 equally spaced values")
def _subdivide(a, b=None, c=None):
    if b is None:
        lo, hi, n = 0, 1, a
    elif c is None:
        lo, hi, n = 0, a, b
    else:
        lo, hi, n = a, b, c
    n = int_value(n, "Subdivide", "number of subdivisions", 1)
    return L(*(ev(Expr("Plus", lo, Expr("Times", Expr("Times", k, Expr("Power", n, -1)),
                                          Expr("Plus", hi, Expr("Times", -1, lo))))) for k in range(n + 1)))


@fn("PowerRange", **ADDED, notes="PowerRange[b] = {1, 10, 100, ...} up to b; PowerRange[a, b] and PowerRange[a, b, r]")
def _powerrange(a, b=None, r=None):
    if b is None:
        lo, hi, ratio = 1, a, 10
    else:
        lo, hi, ratio = a, b, (10 if r is None else r)
    lo_s, hi_s, r_s = (real_number(v, "PowerRange") for v in (lo, hi, ratio))
    if r_s <= 0 or r_s == 1:
        raise ValueError("PowerRange: the ratio must be positive and different from 1")
    out = []
    x = lo
    xs = lo_s
    tol = sp.Float(1e-12) * abs(hi_s) if any(isinstance(v, sp.Float) for v in (lo_s, hi_s, r_s)) else 0
    for _ in range(K.MAX_TABLE):
        if (r_s > 1 and xs > hi_s + tol) or (r_s < 1 and xs < hi_s - tol):
            break
        out.append(x)
        x = ev(Expr("Times", x, ratio))
        xs = real_number(x, "PowerRange")
    return L(*out)


@fn("Subsequences", **ADDED, notes="contiguous subsequences by length then position; Subsequences[list, n | {n} | {m, n}]")
def _subsequences(x, spec=None):
    items = items_of(x, "Subsequences")
    n = len(items)
    lo, hi = 0, n
    if spec is not None:
        if is_int(spec):
            hi = min(int(spec), n)
        elif is_list(spec) and len(spec.args) == 1:
            lo = hi = int_value(spec.args[0], "Subsequences", "length", 0)
        elif is_list(spec) and len(spec.args) == 2:
            lo = int_value(spec.args[0], "Subsequences", "length", 0)
            hi = n if is_sym(spec.args[1], "Infinity") else min(int_value(spec.args[1], "Subsequences", "length", 0), n)
        elif is_sym(spec, "All") or is_sym(spec, "Infinity"):
            pass
        else:
            raise ValueError("Subsequences: invalid length specification")
    return L(*(_rebuild(x, items[i:i + k]) for k in range(lo, hi + 1) for i in range(n - k + 1 if k else 1)))


# Reap / Sow --------------------------------------------------------------------------------------

_REAP = contextvars.ContextVar("mikoshilang_reap", default=None)
K.BUILTIN_ATTRIBUTES["Reap"] = {"HoldFirst", "Protected"}


@fn("Sow", **ADDED, notes="Sow[e] and Sow[e, tag] give e and record it for the innermost enclosing Reap "
                          "(outside Reap, Sow just returns e)")
def _sow(e, tag=None):
    stack = _REAP.get()
    if stack:
        t = Symbol("None") if tag is None else tag
        tags = t.args if is_list(t) else (t,)
        for tg in tags:
            for frame in reversed(stack):
                if frame[0] is None or any(K.first_match(p, tg, lambda z: ev(z)) is not None for p in frame[0]):
                    frame[1].append((tg, e))
                    break
    return e


@fn("Reap", **ADDED, notes="Reap[expr] -> {value, {{sown...} per tag in order of first use}}; Reap[expr, patt] and "
                           "Reap[expr, {p1, ...}] keep matching tags; Reap[expr, patt, f] applies f[tag, list]")
def _reap(expr, patt=None, f=None):
    pats = None if patt is None else (list(ev(patt).args) if is_list(ev(patt)) else [ev(patt)])
    stack = list(_REAP.get() or [])
    frame = (pats if patt is not None else None, [])
    stack.append(frame)
    tok = _REAP.set(stack)
    try:
        value = ev(expr)
    finally:
        _REAP.reset(tok)
    sown = frame[1]

    def grouped(pred):
        keys, groups = [], []
        for tg, e in sown:
            if not pred(tg):
                continue
            for i, k in enumerate(keys):
                if same(k, tg):
                    groups[i].append(e)
                    break
            else:
                keys.append(tg)
                groups.append([e])
        if f is None:
            return L(*(L(*g) for g in groups))
        return L(*(apply(f, k, L(*g)) for k, g in zip(keys, groups)))
    if patt is None:
        return L(value, grouped(lambda t: True))
    pv = ev(patt)
    if is_list(pv):
        return L(value, L(*(grouped(lambda t, p=p: K.first_match(p, t, lambda z: ev(z)) is not None)
                            for p in pv.args)))
    return L(value, grouped(lambda t: K.first_match(pv, t, lambda z: ev(z)) is not None))


# ─────────────────────────── elements ───────────────────────────

def _positions(spec, fname):
    """A single position {i, j, ...} or a list of positions {{...}, ...} (ints, Key[k] or strings)."""
    if is_int(spec) or isinstance(spec, str) or (isinstance(spec, Expr) and spec.head == "Key"):
        return [[spec]]
    if is_list(spec):
        if spec.args and all(is_list(p) for p in spec.args):
            return [list(p.args) for p in spec.args]
        return [list(spec.args)]
    raise ValueError(f"{fname}: {spec!r} is not a valid position specification")


@fn("Extract", **ADDED, notes="Extract[expr, pos], Extract[expr, {pos1, pos2, ...}] and Extract[expr, pos, h]")
def _extract(x, spec, h=None):
    single = not (is_list(spec) and spec.args and all(is_list(p) for p in spec.args))
    out = []
    for p in _positions(spec, "Extract"):
        v = KL.part(x, p, ev)
        out.append(K.call(h, v) if h is not None else v)
    return out[0] if single else L(*out)


def _resolve(x, pos, fname):
    """Normalise a position to non-negative integer indices / keys against expr x."""
    out = []
    e = x
    for i in pos:
        if KL.is_assoc(e):
            key = i.args[0] if isinstance(i, Expr) and i.head == "Key" else i if isinstance(i, str) else None
            if key is None and is_int(i):
                k = int(i)
                n = len(e.args)
                k0 = k - 1 if k > 0 else n + k
                if not 0 <= k0 < n:
                    raise ValueError(f"{fname}: position {pos} does not exist")
                key = e.args[k0].args[0]
            if key is None or not KL.assoc_has(e, key):
                raise ValueError(f"{fname}: position {pos} does not exist")
            idx = next(j for j, r in enumerate(e.args) if same(r.args[0], key))
            out.append(("k", idx))
            e = e.args[idx].args[1]
            continue
        if not is_int(i) or not isinstance(e, Expr) or KL.atomq(e):
            raise ValueError(f"{fname}: position {pos} does not exist")
        k = int(i)
        n = len(e.args)
        if k == 0:
            out.append(("h", 0))
            e = Symbol(e.head) if isinstance(e.head, str) else e.head
            continue
        k0 = k - 1 if k > 0 else n + k
        if not 0 <= k0 < n:
            raise ValueError(f"{fname}: position {pos} does not exist")
        out.append(("i", k0))
        e = e.args[k0]
    return out


def _modify_at(x, rpos, f):
    """Apply f to the part at resolved position rpos (list of ("i", k) / ("k", k) / ("h", 0))."""
    if not rpos:
        return f(x)
    kind, k = rpos[0]
    if kind == "h":
        if len(rpos) > 1:
            raise ValueError("cannot address parts of a head")
        h = f(Symbol(x.head) if isinstance(x.head, str) else x.head)
        return Expr(h.name if isinstance(h, Symbol) else h, *x.args)
    args = list(x.args)
    if kind == "k":
        r = args[k]
        args[k] = Expr(r.head, r.args[0], _modify_at(r.args[1], rpos[1:], f))
    else:
        args[k] = _modify_at(args[k], rpos[1:], f)
    return Expr(x.head, *args)


def _expand_all(x, pos, fname):
    """Expand All and Span entries in a position into explicit positions."""
    if not pos:
        return [[]]
    i = pos[0]
    if is_sym(i, "All") or (isinstance(i, Expr) and i.head == "Span"):
        if not isinstance(x, Expr) or (KL.atomq(x) and not KL.is_assoc(x)):
            raise ValueError(f"{fname}: position does not exist")
        n = len(x.args)
        sel = range(n) if is_sym(i, "All") else KL.seq_indices(i, n, fname)
        out = []
        for k in sel:
            child = x.args[k].args[1] if KL.is_assoc(x) else x.args[k]
            for rest in _expand_all(child, pos[1:], fname):
                out.append([k + 1] + rest)
        return out
    child = KL.part(x, [i], ev) if not (is_int(i) and int(i) == 0) else None
    return [[i] + rest for rest in (_expand_all(child, pos[1:], fname) if child is not None else [[]])]


@fn("MapAt", **ADDED, notes="MapAt[f, expr, n | {i, j, ...} | {{...}, ...}]; positions may use negative indices, "
                            "All, Span and Key; a repeated position applies f repeatedly")
def _mapat(f, x, spec):
    out = x
    for p in _positions(spec, "MapAt"):
        for q in _expand_all(x, p, "MapAt"):
            out = _modify_at(out, _resolve(out, q, "MapAt"), lambda e: K.call(f, e))
    return ev(out)


@fn("ReplacePart", **ADDED, notes="ReplacePart[expr, pos -> new], lists of rules, lists of positions, position "
                                  "patterns such as {i_, i_} -> 0 and pos :> new (evaluated per position)")
def _replacepart(x, rules):
    rs = list(rules.args) if is_list(rules) else [rules]
    if not all(is_rule(r) for r in rs):
        raise ValueError("ReplacePart: the second argument must be a rule or a list of rules")
    explicit, patterned = [], []
    for r in rs:
        lhs = r.args[0]
        if K.has_pattern(lhs):
            patterned.append(r)
        else:
            for p in _positions(lhs, "ReplacePart"):
                explicit.append((p, r))
    out = x
    done = set()
    for p, r in explicit:
        try:
            rp = _resolve(x, p, "ReplacePart")
        except ValueError:
            continue                 # Wolfram ignores positions that do not exist
        t = tuple(rp)
        if t in done:
            continue
        done.add(t)
        new = r.args[1] if r.head == "Rule" else ev(r.args[1])
        out = _modify_at(out, rp, lambda e, v=new: v)
    if patterned:
        allpos = [pos for pos, _ in KL.walk(x, (0, KL.INF), heads=False)]
        for pos in allpos:
            pl = L(*pos)
            for r in patterned:
                b = K.first_match(r.args[0], pl, lambda z: ev(z))
                if b is None and len(pos) == 1:
                    b = K.first_match(r.args[0], pos[0], lambda z: ev(z))
                if b is not None:
                    rp = _resolve(x, list(pos), "ReplacePart")
                    if tuple(rp) in done:
                        break
                    done.add(tuple(rp))
                    rhs = K.substitute(r.args[1], b)
                    new = rhs if r.head == "Rule" else ev(rhs)
                    out = _modify_at(out, rp, lambda e, v=new: v)
                    break
    return out


@fn("Delete", **ADDED, notes="Delete[expr, n], Delete[expr, {i, j, ...}] and Delete[expr, {{...}, ...}] (positions "
                             "refer to the original expression)")
def _delete(x, spec):
    rps = []
    for p in _positions(spec, "Delete"):
        rp = _resolve(x, p, "Delete")
        if not rp:
            raise ValueError("Delete: cannot delete the whole expression (position {})")
        if any(k == "h" for k, _ in rp):
            raise NotImplementedError("Delete: deleting heads (position 0) is not supported")
        rps.append(rp)
    # delete deepest / rightmost first so earlier indices stay valid
    out = x
    for rp in sorted(rps, key=lambda r: [i for _, i in r], reverse=True):
        out = _delete_at(out, rp)
    return out


def _delete_at(x, rp):
    kind, k = rp[0]
    args = list(x.args)
    if len(rp) == 1:
        del args[k]
        return Expr(x.head, *args)
    if kind == "k":
        r = args[k]
        args[k] = Expr(r.head, r.args[0], _delete_at(r.args[1], rp[1:]))
    else:
        args[k] = _delete_at(args[k], rp[1:])
    return Expr(x.head, *args)


@fn("Insert", **ADDED, notes="Insert[list, elem, n] (negative n counts from the end: -1 appends), "
                             "Insert[expr, elem, {i, j, ...}] and several positions {{...}, ...}")
def _insert(x, elem, spec):
    poss = _positions(spec, "Insert")

    def ins(e, pos):
        if not isinstance(e, Expr) or KL.atomq(e) and not KL.is_assoc(e):
            raise ValueError("Insert: position does not exist")
        n = len(e.args)
        i = int_value(pos[0], "Insert", "position")
        if len(pos) == 1:
            k = i - 1 if i > 0 else n + i + 1
            if not 0 <= k <= n:
                raise ValueError(f"Insert: cannot insert at position {i} in an expression of length {n}")
            if KL.is_assoc(e):
                if not is_rule(elem):
                    raise ValueError("Insert: only rules can be inserted into an association")
                return KL.make_assoc(list(e.args[:k]) + [elem] + list(e.args[k:]))
            return Expr(e.head, *e.args[:k], elem, *e.args[k:])
        k = i - 1 if i > 0 else n + i
        if not 0 <= k < n:
            raise ValueError("Insert: position does not exist")
        args = list(e.args)
        args[k] = ins(args[k], pos[1:])
        return Expr(e.head, *args)
    # several positions: insert from the last so that the original positions stay valid
    def norm(p, e=x):
        out = []
        cur = e
        for depth_, i in enumerate(p):
            n = len(cur.args) if isinstance(cur, Expr) else 0
            ii = int_value(i, "Insert", "position")
            last = depth_ == len(p) - 1
            k = (ii - 1 if ii > 0 else n + ii + (1 if last else 0))
            out.append(k)
            if not last:
                cur = cur.args[k]
        return out
    order = sorted(poss, key=lambda p: norm(p), reverse=True)
    out = x
    for p in order:
        idx = norm(p)
        out = ins(out, [i + 1 for i in idx])
    return out


@fn("FirstPosition", **ADDED, notes="first position matching patt in Position order; Missing[\"NotFound\"] or the "
                                    "given default; optional level specification")
def _firstposition(x, patt, default=None, levels=None):
    args = [x, patt] + ([levels] if levels is not None else []) + [1]
    if levels is None:
        args = [x, patt, Expr("List", 0, Symbol("Infinity")), 1]
    r = ev(Expr("Position", *args))
    if r.args:
        return r.args[0]
    return KL.missing("NotFound") if default is None else default


@fn("SelectFirst", **ADDED, notes="first element with crit[e] True, else Missing[\"NotFound\"] or the default")
def _selectfirst(x, crit, default=None):
    for e in items_of(x, "SelectFirst"):
        if true(crit, e):
            return e
    return KL.missing("NotFound") if default is None else default


@fn("FirstCase", **ADDED, notes="FirstCase[expr, patt | patt :> rhs, default, levelspec]; default level {1}; "
                                "Missing[\"NotFound\"] when nothing matches")
def _firstcase(x, patt, default=None, levels=None):
    args = [x, patt] + ([levels] if levels is not None else [L(1)]) + [1]
    r = ev(Expr("Cases", *args))
    if r.args:
        return r.args[0]
    return KL.missing("NotFound") if default is None else default


@fn("Pick", **ADDED, notes="Pick[list, sel] keeps elements whose selector is True; Pick[list, sel, patt] those whose "
                           "selector matches patt; nested selector lists pick at deeper levels")
def _pick(x, sel, patt=None):
    def ok(s):
        if patt is None:
            return s is True
        return K.first_match(patt, s, lambda z: ev(z)) is not None

    def go(e, s):
        if ok(s):
            return True, e
        if is_list(s) or (isinstance(s, Expr) and not KL.atomq(s)):
            if not isinstance(e, Expr) or KL.atomq(e) or len(e.args) != len(s.args):
                raise ValueError("Pick: the selector does not have the same shape as the list")
            kept = []
            for a, b in zip(e.args, s.args):
                keep, v = go(a, b)
                if keep:
                    kept.append(v)
            return True, Expr(e.head, *kept)
        return False, None
    keep, v = go(x, sel)
    if not keep:
        return L() if is_list(x) else Expr(_head_of(x))
    return v


@fn("TakeWhile", **ADDED, notes="leading elements for which crit is True")
def _takewhile(x, crit):
    out = []
    for e in items_of(x, "TakeWhile"):
        if not true(crit, e):
            break
        out.append(e)
    return _rebuild(x, out) if not is_assoc(x) else L(*out)


@fn("LengthWhile", **ADDED, notes="number of leading elements for which crit is True")
def _lengthwhile(x, crit):
    n = 0
    for e in items_of(x, "LengthWhile"):
        if not true(crit, e):
            break
        n += 1
    return n


@fn("TakeDrop", **ADDED, notes="TakeDrop[list, spec] = {Take[list, spec], Drop[list, spec]}")
def _takedrop(x, spec):
    return L(ev(Expr("Take", x, spec)), ev(Expr("Drop", x, spec)))


@fn("TakeList", **ADDED, notes="TakeList[list, {n1, n2, ...}] gives successive pieces; ni may be UpTo[k] or All")
def _takelist(x, specs):
    items = items_of(x, "TakeList")
    if not is_list(specs):
        raise ValueError("TakeList: the second argument must be a list of counts")
    out, i = [], 0
    for s in specs.args:
        rest = len(items) - i
        if is_sym(s, "All"):
            k = rest
        else:
            k = upto(s, rest, "TakeList")
        out.append(_rebuild(x, items[i:i + k]))
        i += k
    return L(*out)


@fn("DeleteAdjacentDuplicates", **ADDED, notes="removes elements equal (SameQ, or the given test) to their predecessor")
def _deladj(x, test=None):
    items = items_of(x, "DeleteAdjacentDuplicates")
    t = tester(test)
    out = []
    for e in items:
        if out and t(out[-1], e):
            continue
        out.append(e)
    return _rebuild(x, out)


@fn("DeleteElements", **ADDED, notes="DeleteElements[list, elems] removes every occurrence of the elements; "
                                     "DeleteElements[list, n -> elems] at most n of each")
def _delelems(x, elems):
    items = items_of(x, "DeleteElements")
    limit = None
    if is_rule(elems):
        limit = None if is_sym(elems.args[0], "Infinity") else int_value(elems.args[0], "DeleteElements", "count", 0)
        elems = elems.args[1]
    if not is_list(elems):
        raise ValueError("DeleteElements: the elements to delete must be given as a list")
    targets = list(elems.args)
    used = [0] * len(targets)
    out = []
    for e in items:
        for j, t in enumerate(targets):
            if same(e, t) and (limit is None or used[j] < limit):
                used[j] += 1
                break
        else:
            out.append(e)
    return _rebuild(x, out)


def _num_key(v, fname):
    """Numeric sort key for real numeric values (Wolfram NumericalOrder)."""
    try:
        return float(real_number(v, fname))
    except ValueError:
        raise ValueError(f"{fname}: {v!r} is not a real numeric value") from None


def _by_values(x, f, fname):
    items = items_of(x, fname)
    return items, [apply(f, e) for e in items]


def _extremal_by(x, f, n, fname, largest):
    items, vals = _by_values(x, f, fname)
    keys = [canonical(v, fname) for v in vals]
    if n is None:
        if not items:
            return L()
        best = max(keys) if largest else min(keys)
        return L(*(e for e, k in zip(items, keys) if k == best))
    k = upto(n, len(items), fname)
    order = sorted(range(len(items)), key=lambda i: keys[i], reverse=largest)   # stable: ties keep list order
    return L(*(items[i] for i in order[:k]))


@fn("MaximalBy", **ADDED, notes="MaximalBy[list, f]: all elements maximising f (canonical order of f values), in "
                                "their original order; MaximalBy[list, f, n | UpTo[n]]: the n largest by f")
def _maximalby(x, f, n=None):
    return _extremal_by(x, f, n, "MaximalBy", True)


@fn("MinimalBy", **ADDED, notes="MinimalBy[list, f]: all elements minimising f; MinimalBy[list, f, n]: the n smallest by f")
def _minimalby(x, f, n=None):
    return _extremal_by(x, f, n, "MinimalBy", False)


def _take_extreme(x, n, fname, largest, f=None):
    items = items_of(x, fname)
    pairs = []
    for i, e in enumerate(items):
        v = apply(f, e) if f is not None else e
        if isinstance(v, Expr) and v.head == "Missing":
            continue                           # Wolfram ignores Missing[...] values
        pairs.append((_num_key(v, fname), i, e))
    k = upto(n, len(pairs), fname)
    pairs.sort(key=lambda t: (-t[0], t[1]) if largest else (t[0], t[1]))
    return L(*(e for _, _, e in pairs[:k]))


@fn("TakeLargest", **ADDED, notes="the n largest numeric elements in decreasing order (Missing[...] ignored); UpTo[n]")
def _takelargest(x, n):
    return _take_extreme(x, n, "TakeLargest", True)


@fn("TakeSmallest", **ADDED, notes="the n smallest numeric elements in increasing order (Missing[...] ignored); UpTo[n]")
def _takesmallest(x, n):
    return _take_extreme(x, n, "TakeSmallest", False)


@fn("TakeLargestBy", **ADDED, notes="the n elements with the largest numeric f values, in decreasing order of f")
def _takelargestby(x, f, n):
    return _take_extreme(x, n, "TakeLargestBy", True, f)


@fn("TakeSmallestBy", **ADDED, notes="the n elements with the smallest numeric f values, in increasing order of f")
def _takesmallestby(x, f, n):
    return _take_extreme(x, n, "TakeSmallestBy", False, f)


def _position_extreme(x, n, fname, largest):
    items = items_of(x, fname)
    keys = [_num_key(v, fname) for v in items]
    if n is None:
        if not keys:
            return L()
        best = max(keys) if largest else min(keys)
        return L(*(i + 1 for i, k in enumerate(keys) if k == best))
    k = upto(n, len(items), fname)
    order = sorted(range(len(items)), key=lambda i: (-keys[i], i) if largest else (keys[i], i))
    return L(*(i + 1 for i in order[:k]))


@fn("PositionLargest", **ADDED, notes="positions of the largest element(s); PositionLargest[list, n] positions of the "
                                      "n largest in decreasing order")
def _poslargest(x, n=None):
    return _position_extreme(x, n, "PositionLargest", True)


@fn("PositionSmallest", **ADDED, notes="positions of the smallest element(s); PositionSmallest[list, n]")
def _possmallest(x, n=None):
    return _position_extreme(x, n, "PositionSmallest", False)


# sequence functions (fixed-length sequence patterns) ----------------------------------------------

def _seq_pattern(patt, fname):
    if not is_list(patt):
        raise ValueError(f"{fname}: the sequence pattern must be a list such as {{a, _}}")
    for p in patt.args:
        if K._seq_info(p) is not None or (isinstance(p, Expr) and p.head in ("Repeated", "RepeatedNull",
                                                                             "PatternSequence", "Optional")):
            raise NotImplementedError(f"{fname}: variable-length sequence patterns are not supported yet; use a "
                                      "list of single-element patterns")
    return len(patt.args)


def _seq_matches(items, patt, fname, overlaps=False):
    """(start, stop, bindings) of matches of a fixed-length list pattern, left to right."""
    n = _seq_pattern(patt, fname)
    out = []
    i = 0
    while i + n <= len(items):
        b = K.first_match(patt, L(*items[i:i + n]), lambda z: ev(z))
        if b is not None and n > 0:
            out.append((i, i + n, b))
            i += 1 if overlaps else n
        else:
            i += 1
    return out


def _seq_rule(r):
    if is_rule(r):
        return r.args[0], r
    return r, None


@fn("SequenceCases", **ADDED, notes="non-overlapping runs matching a fixed-length list pattern {p1, p2, ...} "
                                    "(or patt -> rhs); Overlaps -> True; variable-length patterns raise")
def _seqcases(x, patt, *rest):
    rest, opts = split_options(rest, {"Overlaps"}, "SequenceCases")
    items = items_of(x, "SequenceCases")
    p, r = _seq_rule(patt)
    out = []
    for i, j, b in _seq_matches(items, p, "SequenceCases", opts.get("Overlaps") is True):
        if r is None:
            out.append(L(*items[i:j]))
        else:
            out.append(ev(K.substitute(r.args[1], b)))
    if rest:
        out = out[:int_value(rest[0], "SequenceCases", "count", 0)]
    return L(*out)


@fn("SequencePosition", **ADDED, notes="{start, end} of runs matching a fixed-length list pattern; overlapping "
                                       "matches are included (Overlaps -> True default); Overlaps -> False")
def _seqpos(x, patt, *rest):
    rest, opts = split_options(rest, {"Overlaps"}, "SequencePosition")
    items = items_of(x, "SequencePosition")
    out = [L(i + 1, j) for i, j, _ in _seq_matches(items, patt, "SequencePosition",
                                                     opts.get("Overlaps", True) is not False)]
    if rest:
        out = out[:int_value(rest[0], "SequencePosition", "count", 0)]
    return L(*out)


@fn("SequenceCount", **ADDED, notes="number of non-overlapping occurrences of a sublist or fixed-length list pattern")
def _seqcount(x, patt, *rest):
    rest, opts = split_options(rest, {"Overlaps"}, "SequenceCount")
    return len(_seq_matches(items_of(x, "SequenceCount"), patt, "SequenceCount", opts.get("Overlaps") is True))


@fn("SequenceReplace", **ADDED, notes="replaces non-overlapping runs matching {p1, ...} -> rhs (rhs is one element)")
def _seqreplace(x, rules, n=None):
    items = items_of(x, "SequenceReplace")
    rs = list(rules.args) if is_list(rules) and rules.args and all(is_rule(r) for r in rules.args) else [rules]
    if not all(is_rule(r) for r in rs):
        raise ValueError("SequenceReplace: give a rule {p1, ...} -> rhs or a list of rules")
    limit = None if n is None else int_value(n, "SequenceReplace", "count", 0)
    out, i, count = [], 0, 0
    while i < len(items):
        hit = None
        if limit is None or count < limit:
            for r in rs:
                k = _seq_pattern(r.args[0], "SequenceReplace")
                if k and i + k <= len(items):
                    b = K.first_match(r.args[0], L(*items[i:i + k]), lambda z: ev(z))
                    if b is not None:
                        hit = (k, r, b)
                        break
        if hit:
            k, r, b = hit
            out.append(ev(K.substitute(r.args[1], b)))
            i += k
            count += 1
        else:
            out.append(items[i])
            i += 1
    return _rebuild(x, out)


@fn("SequenceSplit", **ADDED, notes="splits at runs matching a fixed-length list pattern (empty pieces at the ends "
                                    "dropped, as Wolfram); patt -> rhs keeps rhs between the pieces")
def _seqsplit(x, patt, n=None):
    items = items_of(x, "SequenceSplit")
    p, r = _seq_rule(patt)
    matches = _seq_matches(items, p, "SequenceSplit")
    if n is not None:
        matches = matches[:int_value(n, "SequenceSplit", "count", 0) - 1] if int(n) >= 1 else []
    out, last = [], 0
    for i, j, b in matches:
        out.append(L(*items[last:i]))
        if r is not None:
            out.append(ev(K.substitute(r.args[1], b)))
        last = j
    out.append(L(*items[last:]))
    if out and is_list(out[0]) and not out[0].args:
        out = out[1:]
    if out and is_list(out[-1]) and not out[-1].args:
        out = out[:-1]
    return L(*out)


# ─────────────────────────── structure ───────────────────────────

@fn("Level", **ADDED, notes="Level[expr, levelspec] (post-order, as Wolfram; associations are one level of values); "
                            "Level[expr, spec, f] gives f[parts...]; Heads -> True")
def _level(x, spec, *rest):
    rest, opts = split_options(rest, {"Heads"}, "Level")
    sp_ = KL.level_spec(spec, (1, 1))
    parts = [e for _, e in KL.walk(x, sp_, heads=opts.get("Heads") is True, ev=ev)]
    if rest:
        return apply(rest[0], *parts)
    return L(*parts)


@fn("Depth", **ADDED, notes="1 for atoms, else 1 + the maximum depth of the elements (heads ignored; an association "
                            "is one level)")
def _depth(x, *rest):
    return KL.depth(x)


@fn("Dimensions", **ADDED, notes="dimensions of a full array of any head (Dimensions[f[a, b]] = {2}); "
                                 "Dimensions[expr, n] at most n levels")
def _dimensions(x, n=None):
    if not isinstance(x, Expr) or KL.atomq(x) and not KL.is_assoc(x):
        return L()
    dims = _dims_any(x)
    if n is not None:
        dims = dims[:int_value(n, "Dimensions", "level", 0)]
    return L(*dims)


def _dims_any(x):
    dims = []
    level = [x]
    while level and all(isinstance(e, Expr) and (not KL.atomq(e) or KL.is_assoc(e)) and e.head == level[0].head
                        for e in level):
        lens = {len(e.args) for e in level}
        if len(lens) != 1:
            break
        n = lens.pop()
        dims.append(n)
        if n == 0:
            break
        level = [c for e in level for c in KL.children(e)]
    return dims


def _array_depth(x):
    return len(KL.array_dims(x)) if is_list(x) else 0


@fn("ArrayDepth", **ADDED, notes="depth to which expr is a full array of lists (0 for non-lists)")
def _arraydepth(x):
    return _array_depth(x)


def _leaves(x, d):
    if d == 0:
        return [x]
    return [y for e in x.args for y in _leaves(e, d - 1)]


@fn("ArrayQ", **ADDED, notes="ArrayQ[expr], ArrayQ[expr, n] (n may be a pattern) and ArrayQ[expr, n, test] for the "
                             "elements")
def _arrayq(x, n=None, test=None):
    if not is_list(x):
        return False
    d = _array_depth(x)
    if d == 0:
        return False
    dims = KL.array_dims(x)
    if 0 not in dims and any(is_list(e) for e in _leaves(x, d)):
        return False            # ragged: some parts at the deepest full level are lists, others are not
    if n is not None and not is_sym(n, "_"):
        pat = n
        if K.first_match(pat, d, lambda z: ev(z)) is None:
            return False
    if test is not None:
        if 0 in dims:
            return True
        return all(true(test, e) for e in _leaves(x, d))
    return True


@fn("VectorQ", **ADDED, notes="a list none of whose elements is a list; VectorQ[expr, test]")
def _vectorq(x, test=None):
    if not is_list(x) or any(is_list(e) for e in x.args):
        return False
    return test is None or all(true(test, e) for e in x.args)


@fn("MatrixQ", **ADDED, notes="a list of equal-length lists (depth-2 full array); MatrixQ[expr, test]")
def _matrixq(x, test=None):
    if not is_list(x) or not x.args or not all(is_list(r) for r in x.args):
        return False
    n = len(x.args[0].args)
    if any(len(r.args) != n for r in x.args) or any(is_list(e) for r in x.args for e in r.args):
        return False
    return test is None or all(true(test, e) for r in x.args for e in r.args)


# ─────────────────────────── rearranging ───────────────────────────

@fn("Split", **ADDED, notes="runs of identical (SameQ) adjacent elements; Split[list, test] tests adjacent pairs")
def _split(x, test=None):
    items = items_of(x, "Split")
    t = tester(test)
    out = []
    for e in items:
        if out and t(out[-1][-1], e):
            out[-1].append(e)
        else:
            out.append([e])
    return L(*(_rebuild(x, g) for g in out))


@fn("SplitBy", **ADDED, notes="runs with identical f values; SplitBy[list, {f1, f2, ...}] splits recursively")
def _splitby(x, f):
    items = items_of(x, "SplitBy")
    fs = list(f.args) if is_list(f) else [f]

    def go(xs, fs):
        out = []
        last = None
        for e in xs:
            k = apply(fs[0], e)
            if out and same(k, last):
                out[-1].append(e)
            else:
                out.append([e])
            last = k
        if len(fs) == 1:
            return L(*(L(*g) for g in out))
        return L(*(go(g, fs[1:]) for g in out))
    return go(items, fs)


@fn("Gather", area=AREA, status="promoted",
    notes="classes of identical (SameQ) elements in order of first appearance; Gather[list, test]")
def _gather(x, test=None):
    items = items_of(x, "Gather")
    _keys, groups = classes(items, test=test) if test is not None else classes(items)
    return L(*(L(*g) for g in groups))


@fn("GatherBy", **ADDED, notes="classes with identical f values, in order of first appearance; GatherBy[list, {f1, "
                               "f2}] nests")
def _gatherby(x, f):
    items = items_of(x, "GatherBy")
    fs = list(f.args) if is_list(f) else [f]

    def go(xs, fs):
        _k, groups = classes(xs, keyf=lambda e: apply(fs[0], e))
        if len(fs) == 1:
            return L(*(L(*g) for g in groups))
        return L(*(go(g, fs[1:]) for g in groups))
    return go(items, fs)


@fn("FlattenAt", **ADDED, notes="FlattenAt[list, n], FlattenAt[list, {i, j, ...}] and several positions "
                                "{{...}, ...}: splices the sublist at that position into its parent")
def _flattenat(x, spec):
    poss = _positions(spec, "FlattenAt")
    rps = [_resolve(x, p, "FlattenAt") for p in poss]
    out = x
    for rp in sorted(rps, key=lambda r: [i for _, i in r], reverse=True):
        out = _flatten_at(out, rp)
    return out


def _flatten_at(x, rp):
    kind, k = rp[0]
    args = list(x.args)
    if len(rp) == 1:
        e = args[k]
        if isinstance(e, Expr) and not KL.atomq(e):
            return Expr(x.head, *args[:k], *e.args, *args[k + 1:])
        return x
    args[k] = _flatten_at(args[k], rp[1:])
    return Expr(x.head, *args)


@fn("ArrayReshape", **ADDED, notes="ArrayReshape[list, dims]: the flattened elements arranged with the given dimensions, "
                                   "padded with 0 (or the given padding) when there are too few")
def _arrayreshape(x, dims, pad=0):
    ds = _seq_spec_list(dims, "ArrayReshape")
    flat = ev(Expr("Flatten", x)).args if is_list(x) else [x]
    total = math.prod(ds)
    if total > K.MAX_TABLE:
        raise ValueError("ArrayReshape: too many elements")
    vals = list(flat[:total]) + [pad] * max(0, total - len(flat))
    it = iter(vals)

    def build(i):
        if i == len(ds):
            return next(it)
        return L(*(build(i + 1) for _ in range(ds[i])))
    return build(0)


@fn("ArrayFlatten", **ADDED, notes="ArrayFlatten[{{m11, m12, ...}, ...}] builds a block matrix; scalar blocks (e.g. 0) "
                                   "become constant blocks of the matching size")
def _arrayflatten(x, r=None):
    if r is not None and not (is_int(r) and int(r) == 2):
        raise NotImplementedError("ArrayFlatten: only rank-2 block matrices are supported")
    rows = items_of(x, "ArrayFlatten")
    if not rows or not all(is_list(rw) for rw in rows) or len({len(rw.args) for rw in rows}) != 1:
        raise ValueError("ArrayFlatten: expected a matrix of blocks")
    nb = len(rows[0].args)

    def shape(b):
        d = KL.array_dims(b) if is_list(b) else []
        return d if len(d) >= 2 else None
    heights = []
    for rw in rows:
        hs = {shape(b)[0] for b in rw.args if shape(b)}
        if len(hs) > 1:
            raise ValueError("ArrayFlatten: blocks in a row must have the same height")
        heights.append(hs.pop() if hs else 1)
    widths = []
    for j in range(nb):
        ws = {shape(rw.args[j])[1] for rw in rows if shape(rw.args[j])}
        if len(ws) > 1:
            raise ValueError("ArrayFlatten: blocks in a column must have the same width")
        widths.append(ws.pop() if ws else 1)
    out = []
    for i, rw in enumerate(rows):
        for r_ in range(heights[i]):
            line = []
            for j, b in enumerate(rw.args):
                if shape(b):
                    line.extend(b.args[r_].args)
                elif is_list(b):
                    raise ValueError("ArrayFlatten: blocks must be matrices or scalars")
                else:
                    line.extend([b] * widths[j])
            out.append(L(*line))
    return L(*out)


def _pad(x, n, padval, left, fname):
    if is_list(padval):
        raise NotImplementedError(f"{fname}: cyclic padding with a list is not supported yet")
    if is_list(n):
        dims = [int_value(v, fname, "length", 0) for v in n.args]
    else:
        dims = [int_value(n, fname, "length", 0)]

    def go(e, ds):
        if not ds:
            return e
        items = list(e.args) if is_list(e) else []
        if not is_list(e):
            raise ValueError(f"{fname}: the array is not deep enough for the padding specification")
        items = [go(y, ds[1:]) if len(ds) > 1 else y for y in items]
        k = ds[0]
        filler = _const(padval, ds[1:])
        if len(items) >= k:
            items = items[len(items) - k:] if left else items[:k]
        else:
            fill = [filler] * (k - len(items))
            items = fill + items if left else items + fill
        return L(*items)
    return go(x, dims)


def _const(c, ds):
    if not ds:
        return c
    return L(*(_const(c, ds[1:]) for _ in range(ds[0])))


def _ragged_dims(x):
    """Maximum length at each level of a nested list (for PadLeft[list] with no length)."""
    dims = []
    level = [x]
    while level and all(is_list(e) for e in level):
        dims.append(max((len(e.args) for e in level), default=0))
        level = [y for e in level for y in e.args]
        if not level:
            break
    return dims


@fn("PadLeft", **ADDED, notes="PadLeft[list, n, x] pads on the left with x (default 0) or keeps the last n elements; "
                              "PadLeft[list, {n1, n2}] and PadLeft[ragged] (to a full array); cyclic list padding raises")
def _padleft(x, n=None, padval=0):
    if n is None:
        dims = _ragged_dims(x)
        return _pad(x, L(*dims), 0, True, "PadLeft")
    return _pad(x, n, padval, True, "PadLeft")


@fn("PadRight", **ADDED, notes="PadRight[list, n, x] pads on the right with x (default 0) or keeps the first n elements; "
                               "PadRight[list, {n1, n2}] and PadRight[ragged]; cyclic list padding raises")
def _padright(x, n=None, padval=0):
    if n is None:
        dims = _ragged_dims(x)
        return _pad(x, L(*dims), 0, False, "PadRight")
    return _pad(x, n, padval, False, "PadRight")


def _sort_key_list(vals, fname):
    return tuple(canonical(v, fname) for v in vals)


@fn("SortBy", **ADDED, notes="sorts by canonical order of f[e] (ties by the canonical order of e); "
                             "SortBy[list, {f1, f2, ...}] lexicographic; SortBy[list, f, p] uses ordering p; "
                             "associations are sorted by value")
def _sortby(x, f, p=None):
    fs = list(f.args) if is_list(f) else [f]
    if is_assoc(x):
        rules = list(x.args)
        vals = [KL.rule_value(r, ev) for r in rules]
        order = _sortby_order(vals, fs, p, "SortBy")
        return Expr("Association", *(rules[i] for i in order))
    items = items_of(x, "SortBy")
    return _rebuild(x, [items[i] for i in _sortby_order(items, fs, p, "SortBy")])


def _sortby_order(items, fs, p, fname):
    keys = [[apply(g, e) for g in fs] for e in items]
    idx = list(range(len(items)))
    if p is None:
        return sorted(idx, key=lambda i: (_sort_key_list(keys[i], fname), canonical(items[i], fname)))
    if len(fs) != 1:
        raise ValueError(f"{fname}: an ordering function needs a single sort function")
    return KL.sort_items(idx, ev, p, fname, key=lambda i: keys[i][0])


@fn("ReverseSort", **ADDED, notes="Reverse[Sort[list]] (reverse canonical order); ReverseSort[list, p]")
def _reversesort(x, p=None):
    s = ev(Expr("Sort", x, *(() if p is None else (p,))))
    return ev(Expr("Reverse", s))


@fn("ReverseSortBy", **ADDED, notes="Reverse[SortBy[list, f]]: decreasing f values, ties in reverse canonical order")
def _reversesortby(x, f):
    return ev(Expr("Reverse", ev(Expr("SortBy", x, f))))


@fn("Ordering", **ADDED, notes="positions that put the list in canonical order (stable); Ordering[list, n] the first n "
                               "(negative n: the last n), Ordering[list, n, p] with ordering p")
def _ordering(x, n=None, p=None):
    items = items_of(x, "Ordering")
    idx = list(range(len(items)))
    if p is None:
        order = sorted(idx, key=lambda i: canonical(items[i], "Ordering"))
    else:
        order = KL.sort_items(idx, ev, p, "Ordering", key=lambda i: items[i])
    if n is not None and not is_sym(n, "All"):
        k = int_value(n, "Ordering", "count")
        if abs(k) > len(items):
            raise ValueError(f"Ordering: cannot take {k} positions from a list of length {len(items)}")
        order = order[:k] if k >= 0 else order[k:]
    return L(*(i + 1 for i in order))


@fn("OrderingBy", **ADDED, notes="positions that sort the list by f (ties by canonical order of the elements); "
                                 "OrderingBy[list, f, n]")
def _orderingby(x, f, n=None):
    items = items_of(x, "OrderingBy")
    order = _sortby_order(items, [f], None, "OrderingBy")
    if n is not None:
        k = int_value(n, "OrderingBy", "count")
        order = order[:k] if k >= 0 else order[k:]
    return L(*(i + 1 for i in order))


@fn("NumericalSort", **ADDED, notes="sorts numeric elements by numerical value (ties by canonical order); "
                                    "non-numeric elements raise")
def _numericalsort(x):
    items = items_of(x, "NumericalSort")
    return _rebuild(x, sorted(items, key=lambda e: (_num_key(e, "NumericalSort"), canonical(e, "NumericalSort"))))


@fn("DeleteDuplicatesBy", **ADDED, notes="keeps the first element for each distinct (SameQ) value of f")
def _deldupby(x, f):
    items = items_of(x, "DeleteDuplicatesBy")
    _k, groups = classes(items, keyf=lambda e: apply(f, e))
    return _rebuild(x, [g[0] for g in groups]) if not is_assoc(x) else L(*(g[0] for g in groups))


@fn("Catenate", **ADDED, notes="joins a list of lists (and/or associations, whose values are used) into one list")
def _catenate(x):
    parts = items_of(x, "Catenate")
    out = []
    for p in parts:
        if is_assoc(p):
            out.extend(KL.assoc_values(p, ev))
        elif is_list(p):
            out.extend(p.args)
        else:
            raise ValueError(f"Catenate: {p!r} is not a list or association")
    return L(*out)


# ─────────────────────────── sets ───────────────────────────

def _opt_test(rest, fname):
    rest, opts = split_options(rest, {"SameTest"}, fname)
    if rest:
        raise ValueError(f"{fname}: too many arguments")
    return opts.get("SameTest")


def _contains(fname, check):
    def impl(a, b, *rest):
        t = tester(_opt_test(rest, fname))
        xs, ys = items_of(a, fname), items_of(b, fname)
        return check(xs, ys, lambda u, v: t(u, v))
    return impl


fn("ContainsAll", **ADDED, notes="every element of e2 occurs in e1 (SameQ or SameTest)")(
    _contains("ContainsAll", lambda xs, ys, t: all(any(t(x, y) for x in xs) for y in ys)))
fn("ContainsAny", **ADDED, notes="some element of e2 occurs in e1")(
    _contains("ContainsAny", lambda xs, ys, t: any(any(t(x, y) for x in xs) for y in ys)))
fn("ContainsNone", **ADDED, notes="no element of e2 occurs in e1")(
    _contains("ContainsNone", lambda xs, ys, t: not any(any(t(x, y) for x in xs) for y in ys)))
fn("ContainsOnly", **ADDED, notes="every element of e1 occurs in e2")(
    _contains("ContainsOnly", lambda xs, ys, t: all(any(t(x, y) for y in ys) for x in xs)))
fn("ContainsExactly", **ADDED, notes="e1 and e2 have the same distinct elements")(
    _contains("ContainsExactly", lambda xs, ys, t: all(any(t(x, y) for y in ys) for x in xs)
              and all(any(t(x, y) for x in xs) for y in ys)))
fn("DisjointQ", **ADDED, notes="the lists have no element in common")(
    _contains("DisjointQ", lambda xs, ys, t: not any(t(x, y) for x in xs for y in ys)))
fn("IntersectingQ", **ADDED, notes="the lists have at least one element in common")(
    _contains("IntersectingQ", lambda xs, ys, t: any(t(x, y) for x in xs for y in ys)))


@fn("DuplicateFreeQ", **ADDED, notes="no two elements are SameQ (or equivalent under the given test)")
def _dupfreeq(x, test=None):
    items = items_of(x, "DuplicateFreeQ")
    return len(dedup(items, test)) == len(items)


# ─────────────────────────── counting ───────────────────────────

@fn("Counts", **ADDED, notes="<|e -> count, ...|> in order of first appearance (SameQ)")
def _counts(x):
    keys, groups = classes(items_of(x, "Counts"))
    return KL.make_assoc([(k, len(g)) for k, g in zip(keys, groups)])


@fn("CountsBy", **ADDED, notes="<|f[e] -> count, ...|> in order of first appearance")
def _countsby(x, f):
    keys, groups = classes(items_of(x, "CountsBy"), keyf=lambda e: apply(f, e))
    return KL.make_assoc([(k, len(g)) for k, g in zip(keys, groups)])


@fn("CountDistinct", **ADDED, notes="number of distinct (SameQ) elements; CountDistinct[list, test]")
def _countdistinct(x, test=None):
    return len(dedup(items_of(x, "CountDistinct"), test))


@fn("CountDistinctBy", **ADDED, notes="number of distinct values of f")
def _countdistinctby(x, f):
    keys, _ = classes(items_of(x, "CountDistinctBy"), keyf=lambda e: apply(f, e))
    return len(keys)


@fn("PositionIndex", **ADDED, notes="<|e -> {positions}, ...|> in order of first appearance")
def _positionindex(x):
    items = items_of(x, "PositionIndex")
    if is_assoc(x):
        keys = KL.assoc_keys(x)
        ks, groups = classes(list(range(len(items))), keyf=lambda i: items[i])
        return KL.make_assoc([(k, L(*(keys[i] for i in g))) for k, g in zip(ks, groups)])
    ks, groups = classes(list(range(len(items))), keyf=lambda i: items[i])
    return KL.make_assoc([(k, L(*(i + 1 for i in g))) for k, g in zip(ks, groups)])


def _bins(data, spec, fname):
    vals = [real_number(v, fname, "data value") for v in data]
    if is_list(spec) and len(spec.args) == 1 and is_list(spec.args[0]):
        edges = [real_number(v, fname, "bin edge") for v in spec.args[0].args]
        if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError(f"{fname}: bin edges must be increasing")
        return edges
    if is_list(spec) and len(spec.args) == 3:
        lo, hi, dx = (real_number(v, fname, "bin specification") for v in spec.args)
    elif not is_list(spec):
        dx = real_number(spec, fname, "bin width")
        if not vals:
            return []
        lo = sp.floor(min(vals) / dx) * dx
        hi = (sp.floor(max(vals) / dx) + 1) * dx
    else:
        raise ValueError(f"{fname}: bin specification must be dx, {{min, max, dx}} or {{{{b1, b2, ...}}}}")
    if dx <= 0:
        raise ValueError(f"{fname}: bin width must be positive")
    n = int(sp.ceiling(sp.nsimplify((hi - lo) / dx) if not isinstance(dx, sp.Float) else (hi - lo) / dx - sp.Float(1e-12)))
    if n > K.MAX_TABLE:
        raise ValueError(f"{fname}: too many bins")
    return [lo + k * dx for k in range(n + 1)]


def _bin_index(v, edges):
    if v < edges[0] or v >= edges[-1]:
        return None
    lo, hi = 0, len(edges) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if v >= edges[mid]:
            lo = mid
        else:
            hi = mid
    return lo


@fn("BinCounts", **ADDED, notes="counts in bins [b_k, b_k+1): BinCounts[data, dx] (bins at multiples of dx), "
                                "BinCounts[data, {min, max, dx}] and BinCounts[data, {{b1, b2, ...}}]")
def _bincounts(data, spec=None):
    if spec is None:
        raise NotImplementedError("BinCounts[data] with automatic bins is not supported; give a bin width")
    items = items_of(data, "BinCounts")
    edges = _bins(items, spec, "BinCounts")
    counts = [0] * max(0, len(edges) - 1)
    for v in items:
        i = _bin_index(real_number(v, "BinCounts", "data value"), edges)
        if i is not None:
            counts[i] += 1
    return L(*counts)


@fn("BinLists", **ADDED, notes="the elements falling in each bin [b_k, b_k+1), with the same bin specifications "
                               "as BinCounts")
def _binlists(data, spec=None):
    if spec is None:
        raise NotImplementedError("BinLists[data] with automatic bins is not supported; give a bin width")
    items = items_of(data, "BinLists")
    edges = _bins(items, spec, "BinLists")
    out = [[] for _ in range(max(0, len(edges) - 1))]
    for v in items:
        i = _bin_index(real_number(v, "BinLists", "data value"), edges)
        if i is not None:
            out[i].append(v)
    return L(*(L(*b) for b in out))


@fn("Ratios", **ADDED, notes="successive ratios a[i+1]/a[i]; Ratios[list, n] for n-th ratios")
def _ratios(x, n=1):
    items = items_of(x, "Ratios")
    k = int_value(n, "Ratios", "order", 0)
    for _ in range(k):
        items = [ev(Expr("Times", items[i + 1], Expr("Power", items[i], -1))) for i in range(len(items) - 1)]
    return L(*items)


@fn("MinMax", **ADDED, notes="{Min[list], Max[list]}; MinMax[list, d] = {min - d, max + d}; "
                             "MinMax[list, {dmin, dmax}]")
def _minmax(x, d=None):
    items = items_of(x, "MinMax")
    lo, hi = ev(Expr("Min", L(*items))), ev(Expr("Max", L(*items)))
    if d is None:
        return L(lo, hi)
    d1, d2 = (d.args if is_list(d) and len(d.args) == 2 else (d, d))
    return L(ev(Expr("Plus", lo, Expr("Times", -1, d1))), ev(Expr("Plus", hi, d2)))


# ─────────────────────────── functional ───────────────────────────

@fn("MapApply", **ADDED, notes="MapApply[f, expr] = f @@@ expr = Apply[f, expr, {1}]")
def _mapapply(f, x):
    return ev(Expr("Apply", f, x, L(1)))


@fn("Scan", **ADDED, notes="evaluates f on each element (or each part at the given levels) for its side effects; "
                           "returns Null (Return[v] inside f returns v)")
def _scan(f, x, levels=None):
    spec = KL.level_spec(levels, (1, 1))
    if KL.atomq(x) and not is_assoc(x) and spec[0] > 0:
        return None
    try:
        for _, e in KL.walk(x, spec, ev=ev):
            apply(f, e)
    except K.ReturnSignal as r:
        return r.value
    return None


@fn("Through", **ADDED, notes="Through[p[f, g][x]] = p[f[x], g[x]]; Through[expr, h] only when p is h")
def _through(x, h=None):
    if not isinstance(x, Expr) or not isinstance(x.head, Expr):
        return x
    p = x.head
    if h is not None and not (isinstance(p.head, str) and isinstance(h, Symbol) and p.head == h.name):
        return x
    return ev(Expr(p.head, *(K.call(f, *x.args) for f in p.args)))


@fn("Operate", **ADDED, notes="Operate[p, f[x, y]] = p[f][x, y]; Operate[p, expr, 0] = p[expr]")
def _operate(p, x, n=1):
    k = int_value(n, "Operate", "level", 0)
    if k == 0:
        return apply(p, x)
    if k != 1:
        raise NotImplementedError("Operate: only levels 0 and 1 are supported")
    if not isinstance(x, Expr) or KL.atomq(x):
        return x
    h = Symbol(x.head) if isinstance(x.head, str) else x.head
    return ev(Expr(K.call(p, h), *x.args))


@fn("Construct", **ADDED, notes="Construct[f, x, y, ...] = f[x, y, ...]")
def _construct(f, *args):
    return apply(f, *args)


def _nestwhile(f, x, test, m=None, mx=None, n=None, keep=False, fname="NestWhile"):
    limit = None if mx is None or is_sym(mx, "Infinity") else int_value(mx, fname, "maximum", 0)
    if m is None:
        mm = 1
    elif is_sym(m, "All"):
        mm = "all"
    elif is_list(m) and len(m.args) == 2:
        raise NotImplementedError(f"{fname}: the {{mmin, m}} form is not supported")
    else:
        mm = int_value(m, fname, "number of results", 1)
    out = [x]
    steps = 0
    while True:
        args = out if mm == "all" else out[-mm:]
        if mm != "all" and len(out) < mm:
            ok = True       # Wolfram only starts testing once m results exist
        else:
            ok = true(test, *args)
        if not ok:
            break
        if limit is not None and steps >= limit:
            break
        if steps >= K.MAX_ITERATIONS:
            raise RecursionError(f"{fname}: too many iterations")
        out.append(apply(f, out[-1]))
        steps += 1
    if n is not None:
        k = int_value(n, fname, "extra steps")
        if k > 0:
            for _ in range(k):
                out.append(apply(f, out[-1]))
        elif k < 0:
            out = out[:len(out) + k] if len(out) + k > 0 else out[:1]
    return L(*out) if keep else out[-1]


@fn("NestWhile", **ADDED, notes="applies f while test[result] is True; NestWhile[f, x, test, m | All, max, n]")
def _nw(f, x, test, m=None, mx=None, n=None):
    return _nestwhile(f, x, test, m, mx, n, False, "NestWhile")


@fn("NestWhileList", **ADDED, notes="the list of results of NestWhile, starting with x")
def _nwl(f, x, test, m=None, mx=None, n=None):
    return _nestwhile(f, x, test, m, mx, n, True, "NestWhileList")


@fn("ComposeList", **ADDED, notes="ComposeList[{f1, f2, ...}, x] = {x, f1[x], f2[f1[x]], ...}")
def _composelist(fs, x):
    out = [x]
    for f in items_of(fs, "ComposeList"):
        out.append(apply(f, out[-1]))
    return L(*out)


def _truth_over(fname, mode):
    def impl(x, test, levels=None):
        if levels is not None:
            parts = [e for _, e in KL.walk(x, KL.level_spec(levels, (1, 1)), ev=ev)]
        else:
            parts = items_of(x, fname)
        for e in parts:
            r = true(test, e)
            if mode == "all" and not r:
                return False
            if mode == "any" and r:
                return True
            if mode == "none" and r:
                return False
        return mode != "any"
    return impl


fn("AllTrue", **ADDED, notes="test[e] is True for every element (stops at the first failure); optional level")(
    _truth_over("AllTrue", "all"))
fn("AnyTrue", **ADDED, notes="test[e] is True for some element (stops at the first success); optional level")(
    _truth_over("AnyTrue", "any"))
fn("NoneTrue", **ADDED, notes="test[e] is True for no element; optional level")(_truth_over("NoneTrue", "none"))


@fn("BlockMap", **ADDED, notes="BlockMap[f, list, n] = Map[f, Partition[list, n]]; BlockMap[f, list, n, d]")
def _blockmap(f, x, n, d=None):
    parts = ev(Expr("Partition", x, n, *(() if d is None else (d,))))
    return ev(Expr("Map", f, parts))


# ─────────────────────────── tests and ordering ───────────────────────────

@fn("OrderedQ", **ADDED, notes="True when the elements are in canonical (non-decreasing) order, or ordered by p")
def _orderedq(x, p=None):
    items = items_of(x, "OrderedQ")
    for u, v in zip(items, items[1:]):
        if p is None:
            if canonical(v, "OrderedQ") < canonical(u, "OrderedQ"):
                return False
        else:
            r = apply(p, u, v)
            if r is False or (is_int(r) and int(r) == -1):
                return False
    return True


@fn("Order", **ADDED, notes="1 if e1 comes before e2 in canonical order, -1 if after, 0 if identical")
def _order(a, b):
    if same(a, b):
        return 0
    ka, kb = canonical(a, "Order"), canonical(b, "Order")
    return 1 if ka < kb else -1 if kb < ka else 0


@fn("Between", **ADDED, notes="min <= x <= max for {min, max}, or within any interval of {{min1, max1}, ...}; "
                              "Between[{min, max}] is an operator form")
def _between(x, spec):
    if not is_list(spec):
        raise ValueError("Between: the range must be {min, max}")
    ranges = list(spec.args) if spec.args and all(is_list(r) for r in spec.args) else [spec]
    for r in ranges:
        if len(r.args) != 2:
            raise ValueError("Between: each range must be {min, max}")
        lo = ev(Expr("LessEqual", r.args[0], x))
        hi = ev(Expr("LessEqual", x, r.args[1]))
        if lo is True and hi is True:
            return True
        if not isinstance(lo, bool) or not isinstance(hi, bool):
            raise Unevaluated()
    return False


@fn("BooleanQ", area=AREA, status="promoted", notes="True exactly for True and False")
def _booleanq(x):
    return isinstance(x, bool)


@fn("AppendTo", **ADDED, notes="AppendTo[s, e] sets s = Append[s, e] and returns the new value (s must have a value)")
def _appendto(s, e):
    return _modify_symbol(s, lambda v: ev(Expr("Append", v, e)), "AppendTo")


@fn("PrependTo", **ADDED, notes="PrependTo[s, e] sets s = Prepend[s, e] and returns the new value")
def _prependto(s, e):
    return _modify_symbol(s, lambda v: ev(Expr("Prepend", v, e)), "PrependTo")


for _h in ("AppendTo", "PrependTo"):
    K.BUILTIN_ATTRIBUTES[_h] = {"HoldFirst", "Protected"}


def _modify_symbol(s, f, fname):
    st = K.state()
    if not isinstance(s, Symbol):
        raise ValueError(f"{fname}: the first argument must be a symbol with a value")
    if s.name not in st.own:
        raise ValueError(f"{fname}: {s.name} has no value")
    new = f(ev(s))
    st.own[s.name] = new
    st.note_write(s.name, new)
    return new
