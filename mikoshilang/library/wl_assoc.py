"""Batch 1 (associations): Association construction and the Key*/Lookup/Merge family.

An association ``<|k1 -> v1, ...|>`` is ``Association[Rule...]``: a repeated key keeps its first position
and takes its last value, RuleDelayed values are evaluated when extracted, and ``assoc[k]`` gives
``Missing["KeyAbsent", k]`` for an absent key.  Associations behave as atoms whose values form one level
(Map, Select, Total and friends act on the values).
"""
from __future__ import annotations

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, Unevaluated
from .wl_common import L, ev, apply, true, is_list, is_rule, is_assoc

AREA = "associations (batch 1)"
ADDED = dict(area=AREA, status="added")


def _missing(k):
    return Expr("Missing", "KeyAbsent", k)


def _rules_of(x, fname):
    """Rules of an association, a rule, or a (nested) list of rules / associations, in order."""
    if is_assoc(x):
        return list(x.args)
    if is_rule(x):
        return [x]
    if is_list(x):
        out = []
        for e in x.args:
            out.extend(_rules_of(e, fname))
        return out
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise ValueError(f"{fname}: {x!r} is not a list of rules or an association")


def _normal_assoc(rules):
    out, index = [], {}
    for r in rules:
        t = K.tag(r.args[0])
        if t in index:
            out[index[t]] = r
        else:
            index[t] = len(out)
            out.append(r)
    return Expr("Association", *out)


def _assoc(x, fname):
    """An association from an association or list of rules (anything else raises / stays unevaluated)."""
    if is_assoc(x):
        return x
    return _normal_assoc(_rules_of(x, fname))


@fn("Association", **ADDED,
    notes="normalises <|...|>: lists of rules and nested associations are flattened in, a repeated key keeps its "
          "first position and takes the last value; non-rule arguments raise")
def _association(*args: Expr) -> Expr:
    if all(is_rule(a) for a in args):
        tags = [K.tag(a.args[0]) for a in args]
        if len(set(tags)) == len(tags):
            raise Unevaluated()          # already normal
    rules = []
    for a in args:
        if is_rule(a) or is_assoc(a) or is_list(a):
            rules.extend(_rules_of(a, "Association"))
        else:
            raise ValueError(f"Association: {a!r} is not a rule, a list of rules or an association")
    return _normal_assoc(rules)


@fn("AssociationQ", **ADDED, notes="True for a valid association")
def _associationq(x: Expr) -> Boolean:
    return is_assoc(x) and all(is_rule(r) for r in x.args)


def _keys_values(x, which, f, fname):
    if is_assoc(x):
        items = [r.args[0] if which == "k" else KL.rule_value(r, ev) for r in x.args]
    elif is_rule(x):
        items = [x.args[0] if which == "k" else KL.rule_value(x, ev)]
        return items[0] if f is None else apply(f, items[0])
    elif is_list(x):
        return L(*(_keys_values(e, which, f, fname) for e in x.args))
    elif isinstance(x, Symbol):
        raise Unevaluated()
    else:
        raise ValueError(f"{fname}: {x!r} is not an association or list of rules")
    return L(*(items if f is None else [apply(f, v) for v in items]))


@fn("Keys", **ADDED,
    notes="keys of an association in order; Keys of a rule list gives each rule's key; lists of associations "
          "thread; Keys[a, f] applies f to each key")
def _keys(x, f=None):
    return _keys_values(x, "k", f, "Keys")


@fn("Values", **ADDED,
    notes="values of an association in order (RuleDelayed values evaluated); rule lists and lists of "
          "associations thread; Values[a, f] applies f to each value")
def _values(x, f=None):
    return _keys_values(x, "v", f, "Values")


def _key_list(ks):
    return list(ks.args) if is_list(ks) else [ks]


def _thread_assoc(x, f, fname):
    if is_list(x) and x.args and all(is_assoc(e) for e in x.args):
        return L(*(f(e) for e in x.args))
    if is_list(x) and all(is_rule(e) for e in x.args):
        r = f(_assoc(x, fname))
        return L(*r.args) if is_assoc(r) else r
    if is_assoc(x):
        return f(x)
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise ValueError(f"{fname}: {x!r} is not an association")


@fn("KeyTake", **ADDED,
    notes="entries for the given keys in the order the keys are listed (absent keys ignored); rule lists give "
          "rule lists; lists of associations thread")
def _keytake(x: Expr, ks: Expr) -> Expr:
    keys = _key_list(ks)

    def one(a):
        out = []
        for k in keys:
            for r in a.args:
                if K.tag(r.args[0]) == K.tag(k):
                    out.append(r)
        return _normal_assoc(out)
    return _thread_assoc(x, one, "KeyTake")


@fn("KeyDrop", **ADDED, notes="entries whose keys are not listed; rule lists and lists of associations thread")
def _keydrop(x: Expr, ks: Expr) -> Expr:
    tags = {K.tag(k) for k in _key_list(ks)}
    return _thread_assoc(x, lambda a: Expr("Association", *(r for r in a.args if K.tag(r.args[0]) not in tags)),
                         "KeyDrop")


@fn("KeySort", **ADDED, notes="entries ordered by canonical order of keys, or by the ordering function p")
def _keysort(x: Expr, p: Expr = None) -> Expr:
    def one(a):
        rs = KL.sort_items(list(a.args), ev, p, "KeySort", key=lambda r: r.args[0])
        return Expr("Association", *rs)
    return _thread_assoc(x, one, "KeySort")


@fn("KeySortBy", **ADDED, notes="entries ordered by canonical order of f[key] (stable for ties)")
def _keysortby(x: Expr, f: Fn) -> Expr:
    def one(a):
        keyed = [(apply(f, r.args[0]), r) for r in a.args]
        rs = KL.sort_items(keyed, ev, None, "KeySortBy", key=lambda t: t[0])
        return Expr("Association", *(r for _k, r in rs))
    return _thread_assoc(x, one, "KeySortBy")


@fn("KeySelect", **ADDED, notes="entries whose key k gives crit[k] True")
def _keyselect(x: Expr, crit: Fn) -> Expr:
    return _thread_assoc(x, lambda a: Expr("Association", *(r for r in a.args if true(crit, r.args[0]))),
                         "KeySelect")


@fn("KeyMap", **ADDED, notes="KeyMap[f, assoc] applies f to every key (later entries win when keys collide)")
def _keymap(f: Fn, x: Expr) -> Expr:
    return _thread_assoc(x, lambda a: _normal_assoc([Expr(r.head, apply(f, r.args[0]), r.args[1]) for r in a.args]),
                         "KeyMap")


@fn("KeyValueMap", **ADDED, notes="{f[k1, v1], f[k2, v2], ...}")
def _keyvaluemap(f: Fn, x: Expr) -> ListOf[Expr]:
    a = _assoc(x, "KeyValueMap")
    return L(*(apply(f, r.args[0], KL.rule_value(r, ev)) for r in a.args))


def _lookup1(a, k, default):
    for r in a.args:
        if K.tag(r.args[0]) == K.tag(k):
            return KL.rule_value(r, ev)
    return _missing(k) if default is None else ev(default)


@fn("Lookup", **ADDED,
    notes="Lookup[assoc, key] (Missing[\"KeyAbsent\", key] if absent), Lookup[assoc, key, default], a list of keys "
          "gives a list, a list of associations threads; rule lists act as associations")
def _lookup(x: Expr, k: Expr, default: Expr = None) -> Expr:
    def keyed(a):
        if is_list(k):
            return L(*(_lookup1(a, kk, default) for kk in k.args))
        return _lookup1(a, k, default)
    if is_list(x) and x.args and all(is_assoc(e) for e in x.args):
        return L(*(keyed(e) for e in x.args))
    return keyed(_assoc(x, "Lookup"))


@fn("AssociationMap", **ADDED,
    notes="AssociationMap[f, {k1, ...}] gives <|k1 -> f[k1], ...|>; on an association f is applied to each rule "
          "and must return a rule")
def _associationmap(f: Fn, x: Expr) -> Expr:
    if is_assoc(x):
        out = []
        for r in x.args:
            v = apply(f, Expr("Rule", r.args[0], KL.rule_value(r, ev)))
            if not is_rule(v):
                raise ValueError("AssociationMap: the function must return a rule for each entry")
            out.append(v)
        return _normal_assoc(out)
    if is_list(x):
        return _normal_assoc([Expr("Rule", k, apply(f, k)) for k in x.args])
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise ValueError("AssociationMap: the second argument must be a list or an association")


@fn("AssociationThread", **ADDED,
    notes="AssociationThread[{k...}, {v...}] or [{k...} -> {v...}]; lengths must agree; repeated keys keep the "
          "last value")
def _associationthread(ks: Expr, vs: Expr = None) -> Expr:
    if vs is None:
        if not is_rule(ks):
            raise ValueError("AssociationThread: expected keys -> values")
        ks, vs = ks.args
    if not (is_list(ks) and is_list(vs)):
        raise ValueError("AssociationThread: keys and values must be lists")
    if len(ks.args) != len(vs.args):
        raise ValueError("AssociationThread: the numbers of keys and values differ")
    return _normal_assoc([Expr("Rule", k, v) for k, v in zip(ks.args, vs.args)])


@fn("Merge", **ADDED,
    notes="Merge[{a1, a2, ...}, f] gives <|k -> f[{values for k}]|> with keys in order of first appearance; rule "
          "lists act as associations")
def _merge(xs: Expr, f: Fn) -> Expr:
    if not is_list(xs):
        if isinstance(xs, Symbol):
            raise Unevaluated()
        raise ValueError("Merge: the first argument must be a list of associations")
    keys, vals, index = [], [], {}
    for x in xs.args:
        for r in _assoc(x, "Merge").args:
            t = K.tag(r.args[0])
            if t not in index:
                index[t] = len(keys)
                keys.append(r.args[0])
                vals.append([])
            vals[index[t]].append(KL.rule_value(r, ev))
    return Expr("Association", *(Expr("Rule", k, apply(f, L(*v))) for k, v in zip(keys, vals)))


@fn("KeyExistsQ", **ADDED, notes="True if the key is present (SameQ on keys)")
def _keyexistsq(x: Expr, k: Expr) -> Boolean:
    return KL.assoc_has(_assoc(x, "KeyExistsQ"), k)


def _matchq(e, p):
    return ev(Expr("MatchQ", e, p)) is True


@fn("KeyMemberQ", **ADDED, notes="True if some key matches the pattern")
def _keymemberq(x: Expr, p: Expr) -> Boolean:
    return any(_matchq(r.args[0], p) for r in _assoc(x, "KeyMemberQ").args)


@fn("KeyFreeQ", **ADDED, notes="True if no key matches the pattern")
def _keyfreeq(x: Expr, p: Expr) -> Boolean:
    return not any(_matchq(r.args[0], p) for r in _assoc(x, "KeyFreeQ").args)


def _assoc_list(xs, fname):
    if not is_list(xs):
        if isinstance(xs, Symbol):
            raise Unevaluated()
        raise ValueError(f"{fname}: expected a list of associations")
    return [_assoc(x, fname) for x in xs.args]


@fn("KeyComplement", **ADDED, notes="entries of the first association whose keys are in none of the others")
def _keycomplement(xs: Expr) -> Expr:
    a = _assoc_list(xs, "KeyComplement")
    if not a:
        raise ValueError("KeyComplement: needs at least one association")
    others = {K.tag(r.args[0]) for b in a[1:] for r in b.args}
    return Expr("Association", *(r for r in a[0].args if K.tag(r.args[0]) not in others))


@fn("KeyIntersection", **ADDED,
    notes="each association restricted to the keys common to all, in the order of the first association")
def _keyintersection(xs: Expr) -> ListOf[Expr]:
    a = _assoc_list(xs, "KeyIntersection")
    if not a:
        return L()
    common = [r.args[0] for r in a[0].args if all(KL.assoc_has(b, r.args[0]) for b in a[1:])]
    return L(*(_keytake(b, L(*common)) for b in a))


@fn("KeyUnion", **ADDED,
    notes="each association extended to the union of keys (in order of first appearance); absent keys get "
          "Missing[\"KeyAbsent\", k] (or f[k] with KeyUnion[list, f])")
def _keyunion(xs: Expr, f: Fn = None) -> ListOf[Expr]:
    a = _assoc_list(xs, "KeyUnion")
    keys, seen = [], set()
    for b in a:
        for r in b.args:
            t = K.tag(r.args[0])
            if t not in seen:
                seen.add(t)
                keys.append(r.args[0])
    out = []
    for b in a:
        rules = []
        for k in keys:
            hit = [r for r in b.args if K.tag(r.args[0]) == K.tag(k)]
            rules.append(hit[0] if hit else Expr("Rule", k, _missing(k) if f is None else apply(f, k)))
        out.append(Expr("Association", *rules))
    return L(*out)


def _modify(s, f, fname):
    st = K.state()
    if not isinstance(s, Symbol):
        raise ValueError(f"{fname}: the first argument must be a symbol whose value is an association")
    if s.name not in st.own:
        raise ValueError(f"{fname}: {s.name} has no value")
    new = f(ev(s))
    st.own[s.name] = new
    st.note_write(s.name, new)
    return new


@fn("AssociateTo", **ADDED,
    notes="AssociateTo[a, k -> v] (or a list of rules) updates the association stored in symbol a and returns it")
def _associateto(s: Expr, rules: Expr) -> Expr:
    def upd(v):
        if not is_assoc(v):
            raise ValueError("AssociateTo: the symbol's value is not an association")
        return _normal_assoc(list(v.args) + _rules_of(rules, "AssociateTo"))
    return _modify(s, upd, "AssociateTo")


@fn("KeyDropFrom", **ADDED, notes="KeyDropFrom[a, key or keys] removes entries from the association stored in a")
def _keydropfrom(s: Expr, ks: Expr) -> Expr:
    def upd(v):
        if not is_assoc(v):
            raise ValueError("KeyDropFrom: the symbol's value is not an association")
        return _keydrop(v, ks)
    return _modify(s, upd, "KeyDropFrom")


for _h in ("AssociateTo", "KeyDropFrom"):
    K.BUILTIN_ATTRIBUTES[_h] = {"HoldFirst", "Protected"}


def _is_missing(x):
    return isinstance(x, Expr) and x.head == "Missing"


@fn("MissingQ", **ADDED, notes="True for Missing[...] expressions")
def _missingq(x: Expr) -> Boolean:
    return _is_missing(x)


@fn("DeleteMissing", **ADDED,
    notes="drops Missing[...] elements of a list (or entries of an association); DeleteMissing[list, n] works "
          "down to level n (n a positive integer or Infinity)")
def _deletemissing(x: Expr, n: Expr = 1) -> Expr:
    from .core import pos_int
    inf = isinstance(n, Expr) and n.head == "DirectedInfinity" or isinstance(n, Symbol) and n.name == "Infinity"
    depth = 10 ** 9 if inf else pos_int(n, "DeleteMissing", "level")

    def go(e, d):
        if is_assoc(e):
            return Expr("Association", *(r for r in e.args if not _is_missing(KL.rule_value(r, ev))))
        if not is_list(e):
            return e
        keep = [x for x in e.args if not _is_missing(x)]
        if d > 1:
            keep = [go(x, d - 1) for x in keep]
        return L(*keep)
    if not (is_list(x) or is_assoc(x)):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise ValueError("DeleteMissing: expected a list or an association")
    return go(x, depth)
