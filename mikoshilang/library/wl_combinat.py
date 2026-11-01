"""Coverage batch 6: permutations, permutation groups and combinatorial counting functions.

Conventions follow the Wolfram Language reference pages:

* a permutation is a ``Cycles[{{...}, ...}]`` object or a permutation list ``{p1, ..., pn}`` (``i -> p_i``);
  ``Cycles`` is canonicalised (1-cycles dropped, each cycle rotated to start with its smallest point, cycles
  sorted by first point);
* ``PermutationProduct[a, b, c]`` applies a, then b, then c (so ``PermutationProduct[l2, l1] == l1[[l2]]`` for
  permutation lists);
* ``Permute[expr, perm]`` moves the element at position i to position ``perm(i)``;
* ``GroupElements`` lists the elements of a permutation group sorted by their image lists (the "standard order"
  for a sorted base); ``GroupMultiplicationTable[[i, j]]`` is the position of
  ``PermutationProduct[g_i, g_j]``.
"""

from __future__ import annotations
import itertools
import math

import sympy as sp

from ..expr import Expr, Symbol
from .core import Unevaluated, fn, R, L, S, is_list, int_arg, nonneg_int, pos_int, list_arg, sym, is_numeric_value
from .wl_common import int_value

AREA = "combinatorics (batch 6)"
_MAX_ELEMENTS = 200000


def _intq(x):
    return (isinstance(x, int) and not isinstance(x, bool)) or isinstance(x, sp.Integer)


# ─────────────────────────────── permutation representations ───────────────────────────────

def _is_cycles(x):
    return isinstance(x, Expr) and x.head == "Cycles"


def _cycles_raw(x, fname):
    """Validated list of cycles (tuples of distinct positive ints) of a Cycles object, or None if not literal."""
    if len(x.args) != 1 or not is_list(x.args[0]):
        raise ValueError(f"{fname}: Cycles takes a single list of cycles, e.g. Cycles[{{{{1, 2}}, {{3, 4, 5}}}}]")
    cycles, seen = [], set()
    for c in x.args[0].args:
        if not is_list(c):
            raise ValueError(f"{fname}: each cycle in {x!r} must be a list of positive integers")
        pts = []
        for p in c.args:
            if not _intq(p):
                return None
            p = int(p)
            if p <= 0:
                raise ValueError(f"{fname}: cycle points must be positive integers, got {p}")
            if p in seen:
                raise ValueError(f"{fname}: point {p} appears more than once in {x!r}")
            seen.add(p)
            pts.append(p)
        cycles.append(tuple(pts))
    return cycles


def _canon_cycles(cycles):
    out = []
    for c in cycles:
        if len(c) < 2:
            continue
        i = c.index(min(c))
        out.append(tuple(c[i:] + c[:i]))
    out.sort(key=lambda c: c[0])
    return out


def _cycles_expr(cycles):
    return Expr("Cycles", L(*(L(*c) for c in _canon_cycles(cycles))))


def _perm_list_q(x):
    if not is_list(x):
        return False
    if not all(_intq(v) for v in x.args):
        return False
    vals = [int(v) for v in x.args]
    return sorted(vals) == list(range(1, len(vals) + 1))


class Perm:
    """A permutation of positive integers, stored as {point: image} for moved points only."""
    __slots__ = ("m",)

    def __init__(self, mapping):
        self.m = {k: v for k, v in mapping.items() if k != v}

    @staticmethod
    def from_cycles(cycles):
        m = {}
        for c in cycles:
            for i, p in enumerate(c):
                m[p] = c[(i + 1) % len(c)]
        return Perm(m)

    @staticmethod
    def from_list(vals):
        return Perm({i + 1: v for i, v in enumerate(vals)})

    def __call__(self, i):
        return self.m.get(i, i)

    def then(self, other):
        """Apply self, then other."""
        pts = set(self.m) | set(other.m)
        return Perm({i: other(self(i)) for i in pts})

    def inverse(self):
        return Perm({v: k for k, v in self.m.items()})

    def cycles(self):
        seen, out = set(), []
        for start in sorted(self.m):
            if start in seen:
                continue
            c, p = [], start
            while p not in seen:
                seen.add(p)
                c.append(p)
                p = self(p)
            out.append(tuple(c))
        return out

    def degree(self):
        return max(self.m) if self.m else 0

    def images(self, n):
        return [self(i) for i in range(1, n + 1)]

    def key(self):
        return tuple(sorted(self.m.items()))

    def power(self, k):
        if k < 0:
            return self.inverse().power(-k)
        result, base = Perm({}), self
        while k:
            if k & 1:
                result = result.then(base)
            base = base.then(base)
            k >>= 1
        return result

    def order(self):
        o = 1
        for c in self.cycles():
            o = o * len(c) // math.gcd(o, len(c))
        return o


def perm_of(x, fname, allow_symbolic=True):
    """Perm for a Cycles object or permutation list; symbolic -> Unevaluated; other values raise."""
    if _is_cycles(x):
        c = _cycles_raw(x, fname)
        if c is None:
            raise Unevaluated()
        return Perm.from_cycles(c), "cycles"
    if _perm_list_q(x):
        return Perm.from_list([int(v) for v in x.args]), "list"
    if is_list(x):
        raise ValueError(f"{fname}: {x!r} is not a permutation list (a rearrangement of Range[n])")
    if allow_symbolic and isinstance(x, (Symbol, Expr)) and not isinstance(x, bool):
        raise Unevaluated()
    raise ValueError(f"{fname}: {x!r} is not a permutation (Cycles object or permutation list)")


def _out(p, kind, n=None):
    if kind == "list":
        n = max(p.degree(), n or 0)
        return L(*p.images(n))
    return _cycles_expr(p.cycles())


@fn("Cycles", area=AREA, status="added",
    notes="canonical form: 1-cycles dropped, each cycle starts with its smallest point, cycles sorted by first "
          "point; repeated or non-positive points raise"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _cycles(spec: Expr) -> Expr:
    c = _cycles_raw(Expr("Cycles", spec), "Cycles")
    if c is None:
        raise Unevaluated()
    return _cycles_expr(c)


@fn("PermutationCyclesQ", area=AREA, status="added", notes="True for a valid Cycles object (disjoint positive "
    "integer cycles), else False"
    ". Check: reference (differential cases in tests/library).")
def _pcq(x: Expr) -> Boolean:
    if not _is_cycles(x):
        return False
    try:
        return _cycles_raw(x, "PermutationCyclesQ") is not None
    except ValueError:
        return False


@fn("PermutationListQ", area=AREA, status="added", notes="True for a rearrangement of Range[n] (n >= 0)"
                                                         ". Check: reference (differential cases in tests/library).")
def _plq(x: Expr) -> Boolean:
    return _perm_list_q(x)


@fn("PermutationCycles", area=AREA, status="added",
    notes="disjoint-cycle form of a permutation list (i -> p_i) or canonical form of a Cycles object"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _pcycles(p: Expr) -> Expr:
    perm, _ = perm_of(p, "PermutationCycles")
    return _cycles_expr(perm.cycles())


@fn("PermutationList", area=AREA, status="added",
    notes="PermutationList[perm] (length PermutationMax) and PermutationList[perm, n] padded to length n; n smaller "
          "than the largest moved point raises"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _plist(p: Expr, n: Expr = None) -> ListOf[Expr]:
    perm, kind = perm_of(p, "PermutationList")
    if n is None:
        k = len(p.args) if kind == "list" else perm.degree()
    else:
        k = nonneg_int(n, "PermutationList", "length")
        if k < perm.degree():
            raise ValueError(f"PermutationList: length {k} is smaller than the largest moved point {perm.degree()}")
    return L(*perm.images(k))


@fn("PermutationProduct", area=AREA, status="added",
    notes="left-to-right product (apply the first permutation first); PermutationProduct[] = Cycles[{}], "
          "PermutationProduct[x] = x; all permutation lists give a list, otherwise Cycles"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _pprod(*ps: Expr) -> Expr:
    if not ps:
        return Expr("Cycles", L())
    if len(ps) == 1:
        return ps[0]
    result, kinds, n = Perm({}), set(), 0
    for p in ps:
        perm, kind = perm_of(p, "PermutationProduct")
        kinds.add(kind)
        if kind == "list":
            n = max(n, len(p.args))
        result = result.then(perm)
    return _out(result, "list" if kinds == {"list"} else "cycles", n)


@fn("InversePermutation", area=AREA, status="added", notes="inverse of a Cycles object or permutation list "
    "(same representation as the input)"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _pinv(p: Expr) -> Expr:
    perm, kind = perm_of(p, "InversePermutation")
    return _out(perm.inverse(), kind, len(p.args) if kind == "list" else None)


@fn("PermutationPower", area=AREA, status="added", notes="n-th power (n any integer, negative = inverse powers)"
                                                         ". Check: exact (exact input gives exact results), "
                                                         "reference (differential cases in tests/library).")
def _ppow(p: Expr, n: Integer) -> Expr:
    perm, kind = perm_of(p, "PermutationPower")
    k = int_arg(n, "PermutationPower", "exponent")
    return _out(perm.power(k), kind, len(p.args) if kind == "list" else None)


@fn("PermutationOrder", area=AREA, status="added", notes="LCM of the cycle lengths; identity -> 1"
                                                         ". Check: exact (exact input gives exact results), "
                                                         "reference (differential cases in tests/library).")
def _porder(p: Expr) -> Expr:
    perm, _ = perm_of(p, "PermutationOrder")
    return perm.order()


@fn("PermutationSupport", area=AREA, status="added", notes="sorted list of the points moved"
                                                           ". Check: exact (exact input gives exact results), "
                                                           "reference (differential cases in tests/library).")
def _psupport(p: Expr) -> ListOf[Expr]:
    perm, _ = perm_of(p, "PermutationSupport")
    return L(*sorted(perm.m))


@fn("PermutationLength", area=AREA, status="added", notes="number of points moved"
                                                          ". Check: exact (exact input gives exact results), "
                                                          "reference (differential cases in tests/library).")
def _plength(p: Expr) -> Integer:
    if _is_group(p):
        return len(_group_support(p))
    perm, _ = perm_of(p, "PermutationLength")
    return len(perm.m)


@fn("PermutationMax", area=AREA, status="added", notes="largest moved point; identity -> -Infinity (Max of the "
    "empty support)"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _pmax(p: Expr) -> Expr:
    if _is_group(p):
        s = _group_support(p)
    else:
        s = sorted(perm_of(p, "PermutationMax")[0].m)
    return s[-1] if s else Expr("Times", -1, Symbol("Infinity"))


@fn("PermutationMin", area=AREA, status="added", notes="smallest moved point; identity -> Infinity"
                                                       ". Check: exact (exact input gives exact results), reference "
                                                       "(differential cases in tests/library).")
def _pmin(p: Expr) -> Expr:
    if _is_group(p):
        s = _group_support(p)
    else:
        s = sorted(perm_of(p, "PermutationMin")[0].m)
    return s[0] if s else Symbol("Infinity")


def _replace_points(e, perm):
    if _intq(e) and int(e) > 0:
        return perm(int(e))
    if _is_cycles(e):
        c = _cycles_raw(e, "PermutationReplace")
        if c is not None:
            return _cycles_expr([tuple(perm(p) for p in cyc) for cyc in c])
    if isinstance(e, Expr) and e.head == "List":
        return L(*(_replace_points(a, perm) for a in e.args))
    return e


@fn("PermutationReplace", area=AREA, status="added",
    notes="replaces each positive integer in expr (at any depth in lists) by its image; Cycles objects in expr are "
          "relabelled (conjugated); a group gives the list of images under GroupElements"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _preplace(e: Expr, p: Expr) -> Expr:
    if _is_group(p):
        return L(*(_replace_points(e, g) for g in _group_elements(p, "PermutationReplace")))
    perm, _ = perm_of(p, "PermutationReplace")
    return _replace_points(e, perm)


def _permute(e, perm, fname):
    if not isinstance(e, Expr) or isinstance(e, bool):
        raise ValueError(f"{fname}: {e!r} has no parts to permute")
    n = len(e.args)
    if perm.degree() > n:
        raise ValueError(f"{fname}: permutation moves point {perm.degree()} but the expression has only {n} parts")
    out = [None] * n
    for i, a in enumerate(e.args, start=1):
        out[perm(i) - 1] = a
    return Expr(e.head, *out)


@fn("Permute", area=AREA, status="added",
    notes="Permute[expr, perm] moves part i to position perm(i) (Permute[{a, b, c}, {2, 3, 1}] = {c, a, b}); a "
          "group gives the list over GroupElements"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _permute_fn(e: Expr, p: Expr) -> Expr:
    if _is_group(p):
        return L(*(_permute(e, g, "Permute") for g in _group_elements(p, "Permute")))
    perm, _ = perm_of(p, "Permute")
    if isinstance(e, Symbol):
        raise Unevaluated()
    return _permute(e, perm, "Permute")


@fn("FindPermutation", area=AREA, status="added",
    notes="FindPermutation[e1, e2]: Cycles p with Permute[e1, p] == e2; FindPermutation[e] relative to Sort[e]; "
          "repeated elements (ambiguous) raise"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _findperm(e1: Expr, e2: Expr = None) -> Expr:
    from .. import kernel as K
    if e2 is None:
        if not isinstance(e1, Expr) or isinstance(e1, Symbol):
            raise Unevaluated()
        a = sorted(e1.args, key=K.order_key)
        b = list(e1.args)
    else:
        if not (isinstance(e1, Expr) and isinstance(e2, Expr)) or isinstance(e1, Symbol) or isinstance(e2, Symbol):
            raise Unevaluated()
        a, b = list(e1.args), list(e2.args)
    if len(a) != len(b):
        raise ValueError("FindPermutation: the expressions have different lengths")
    tags_b = [K.tag(x) for x in b]
    if len(set(tags_b)) != len(tags_b):
        raise ValueError("FindPermutation: the expressions contain repeated elements, so the permutation is not unique")
    pos = {t: i + 1 for i, t in enumerate(tags_b)}
    m = {}
    for i, x in enumerate(a, start=1):
        t = K.tag(x)
        if t not in pos:
            raise ValueError("FindPermutation: the expressions are not permutations of each other")
        m[i] = pos[t]
    return _cycles_expr(Perm(m).cycles())


@fn("Signature", area=AREA, status="added",
    notes="sign (+1/-1) of the permutation that sorts the list into canonical order; 0 if any element is repeated"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _signature(x: ListOf[Expr]) -> Expr:
    from .. import kernel as K
    if not is_list(x):
        raise Unevaluated()
    items = list(x.args)
    tags = [K.tag(v) for v in items]
    if len(set(tags)) != len(tags):
        return 0
    order = sorted(range(len(items)), key=lambda i: K.order_key(items[i]))
    # parity of the permutation `order`
    seen, sign = [False] * len(order), 1
    for i in range(len(order)):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = order[j]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign


@fn("RandomPermutation", area=AREA, status="added",
    notes="RandomPermutation[n] (uniform on Range[n], a Cycles object), [n, k] k of them, [group], [group, k]; "
          "reproducible after SeedRandom (the stream differs from Wolfram's generator)")
def _randperm(n: Expr, k: Expr = None) -> Expr:
    from .statistics import _RNG
    if _is_group(n):
        elems = _group_elements(n, "RandomPermutation")
        one = lambda: _cycles_expr(_RNG.choice(elems).cycles())
    else:
        m = nonneg_int(n, "RandomPermutation", "degree")

        def one():
            vals = list(range(1, m + 1))
            _RNG.shuffle(vals)
            return _cycles_expr(Perm.from_list(vals).cycles())
    if k is None:
        return one()
    kk = nonneg_int(k, "RandomPermutation", "count")
    return L(*(one() for _ in range(kk)))


# ─────────────────────────────── permutation groups ───────────────────────────────

_GROUP_HEADS = ("SymmetricGroup", "AlternatingGroup", "CyclicGroup", "DihedralGroup", "AbelianGroup",
                "PermutationGroup")


def _is_group(x):
    return isinstance(x, Expr) and x.head in _GROUP_HEADS


def _named_arg(g):
    if len(g.args) != 1:
        raise ValueError(f"{g.head} takes one argument")
    return g.args[0]


def _group_gens(g, fname):
    """(generators as Perm list, degree) for a group expression; Unevaluated for symbolic parameters."""
    h = g.head
    if h == "PermutationGroup":
        if len(g.args) != 1 or not is_list(g.args[0]):
            raise ValueError("PermutationGroup takes a list of permutations")
        gens = [perm_of(p, fname)[0] for p in g.args[0].args]
        return gens, max([p.degree() for p in gens] + [0])
    a = _named_arg(g)
    if h == "AbelianGroup":
        raise NotImplementedError(f"{fname}: the permutation representation of AbelianGroup is not available; "
                                  "use GroupOrder, or build the group with PermutationGroup")
    if not _intq(a):
        if isinstance(a, (Symbol, Expr)) and not is_numeric_value(a):
            raise Unevaluated()
        raise ValueError(f"{h}: the degree must be a non-negative integer")
    n = int(a)
    if n < 0:
        raise ValueError(f"{h}: the degree must be a non-negative integer")
    rot = Perm.from_cycles([tuple(range(1, n + 1))]) if n >= 2 else Perm({})
    if h == "SymmetricGroup":
        return ([Perm({1: 2, 2: 1}), rot] if n >= 2 else []), n
    if h == "CyclicGroup":
        if n == 0:
            raise ValueError("CyclicGroup: the order must be a positive integer")
        return ([rot] if n >= 2 else []), n
    if h == "AlternatingGroup":
        return [Perm.from_cycles([(1, 2, i)]) for i in range(3, n + 1)], n
    if h == "DihedralGroup":
        if n < 1:
            raise ValueError("DihedralGroup: the degree must be a positive integer")
        if n < 3:
            raise NotImplementedError(f"{fname}: the permutation representations of DihedralGroup[1] and "
                                      "DihedralGroup[2] are not available; use GroupOrder")
        refl = Perm({i: n + 2 - i for i in range(2, n + 1)})
        return [rot, refl], n
    raise ValueError(f"{fname}: unknown group {g!r}")


def _closure(gens, limit=_MAX_ELEMENTS):
    ident = Perm({})
    elems = {ident.key(): ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for e in frontier:
            for g in gens:
                p = e.then(g)
                k = p.key()
                if k not in elems:
                    elems[k] = p
                    nxt.append(p)
                    if len(elems) > limit:
                        raise ValueError(f"group has more than {limit} elements; too many to list")
        frontier = nxt
    return list(elems.values())


def _group_elements(g, fname):
    gens, deg = _group_gens(g, fname)
    elems = _closure(gens)
    elems.sort(key=lambda p: p.images(deg))
    return elems


def _group_support(g):
    gens, _ = _group_gens(g, "PermutationSupport")
    return sorted(set().union(*[set(p.m) for p in gens]) if gens else set())


def _sympy_group(gens, deg):
    from sympy.combinatorics import Permutation, PermutationGroup
    n = max(deg, 1)
    perms = [Permutation([p(i) - 1 for i in range(1, n + 1)]) for p in gens] or [Permutation(list(range(n)))]
    return PermutationGroup(perms), n


@fn("GroupOrder", area=AREA, status="added",
    notes="order of SymmetricGroup (n!), AlternatingGroup (n!/2, 1 for n <= 2), CyclicGroup (n), DihedralGroup (2n), "
          "AbelianGroup (product) or a PermutationGroup (Schreier-Sims via SymPy)")
def _gorder(g):
    if not _is_group(g):
        raise Unevaluated()
    h = g.head
    if h == "AbelianGroup":
        a = _named_arg(g)
        ns = [pos_int(v, "AbelianGroup", "order") for v in list_arg(a, "AbelianGroup")]
        return math.prod(ns)
    if h in ("SymmetricGroup", "AlternatingGroup", "CyclicGroup", "DihedralGroup"):
        a = _named_arg(g)
        n = int_arg(a, h, "degree")
        if n < 0 or (n == 0 and h in ("CyclicGroup", "DihedralGroup")):
            raise ValueError(f"{h}: invalid degree {n}")
        return {"SymmetricGroup": math.factorial(n), "AlternatingGroup": max(1, math.factorial(n) // 2),
                "CyclicGroup": n, "DihedralGroup": 2 * n}[h]
    gens, deg = _group_gens(g, "GroupOrder")
    if not gens:
        return 1
    G, _ = _sympy_group(gens, deg)
    return int(G.order())


@fn("GroupElements", area=AREA, status="added",
    notes="all elements as Cycles objects sorted by image lists (Wolfram's standard order for a sorted base; the "
          "identity is first); GroupElements[g, {r1, ...}] picks positions (negative from the end)")
def _gelements(g, positions=None):
    if not _is_group(g):
        raise Unevaluated()
    elems = [_cycles_expr(p.cycles()) for p in _group_elements(g, "GroupElements")]
    if positions is None:
        return L(*elems)
    out = []
    for r in list_arg(positions, "GroupElements", "positions"):
        k = int_value(r, "GroupElements", "position")
        if k == 0 or abs(k) > len(elems):
            raise ValueError(f"GroupElements: position {k} is outside 1..{len(elems)}")
        out.append(elems[k - 1] if k > 0 else elems[k])
    return L(*out)


@fn("GroupGenerators", area=AREA, status="added",
    notes="generators of a PermutationGroup (as given, identities removed), SymmetricGroup[n] "
          "({Cycles[{{1, 2}}], Cycles[{{1, ..., n}}]}) and CyclicGroup[n]; other named groups raise because "
          "Wolfram's chosen generating sets are not reproduced")
def _ggens(g):
    if not _is_group(g):
        raise Unevaluated()
    if g.head == "PermutationGroup":
        gens, _ = _group_gens(g, "GroupGenerators")
        return L(*(_cycles_expr(p.cycles()) for p in gens if p.m))
    if g.head in ("SymmetricGroup", "CyclicGroup"):
        gens, _ = _group_gens(g, "GroupGenerators")
        out, seen = [], set()
        for p in gens:
            if p.m and p.key() not in seen:
                seen.add(p.key())
                out.append(_cycles_expr(p.cycles()))
        return L(*out)
    raise NotImplementedError(f"GroupGenerators: Wolfram's generating set for {g.head} is not reproduced; use "
                              "GroupElements, or PermutationGroup with explicit generators")


def _position(elems, p):
    keys = {e.key(): i + 1 for i, e in enumerate(elems)}
    return keys.get(p.key())


@fn("GroupElementQ", area=AREA, status="added", notes="membership test for a permutation in a group"
                                                      ". Check: reference (differential cases in tests/library).")
def _gelemq(g: Expr, p: Expr) -> Boolean:
    if not _is_group(g):
        raise Unevaluated()
    perm, _ = perm_of(p, "GroupElementQ")
    gens, deg = _group_gens(g, "GroupElementQ")
    if perm.degree() > max(deg, 0) and perm.m:
        return False
    if not gens:
        return not perm.m
    G, n = _sympy_group(gens, deg)
    from sympy.combinatorics import Permutation
    return bool(G.contains(Permutation([perm(i) - 1 for i in range(1, n + 1)])))


@fn("GroupElementPosition", area=AREA, status="added",
    notes="position of a permutation (or list of them) in GroupElements[group]; non-members raise"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _gelempos(g: Expr, p: Expr) -> Expr:
    if not _is_group(g):
        raise Unevaluated()
    elems = _group_elements(g, "GroupElementPosition")

    def one(x):
        perm, _ = perm_of(x, "GroupElementPosition")
        k = _position(elems, perm)
        if k is None:
            raise ValueError(f"GroupElementPosition: {x!r} is not an element of the group")
        return k
    if is_list(p) and not _perm_list_q(p):
        return L(*(one(x) for x in p.args))
    return one(p)


@fn("GroupMultiplicationTable", area=AREA, status="added",
    notes="m[[i, j]] = position of PermutationProduct[g_i, g_j] in GroupElements (array indexing)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _gmult(g: Expr) -> ListOf[Expr]:
    if not _is_group(g):
        raise Unevaluated()
    elems = _group_elements(g, "GroupMultiplicationTable")
    if len(elems) > 2000:
        raise ValueError("GroupMultiplicationTable: group too large for a table")
    keys = {e.key(): i + 1 for i, e in enumerate(elems)}
    return L(*(L(*(keys[a.then(b).key()] for b in elems)) for a in elems))


@fn("GroupOrbits", area=AREA, status="added",
    notes="GroupOrbits[group] orbits of the support points; GroupOrbits[group, {p1, ...}] orbits of the given "
          "points; each orbit sorted, orbits ordered by smallest point"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _gorbits(g: Expr, pts: Expr = None) -> ListOf[Expr]:
    if not _is_group(g):
        raise Unevaluated()
    gens, deg = _group_gens(g, "GroupOrbits")
    if pts is None:
        todo = _group_support(g)
    else:
        todo = [pos_int(p, "GroupOrbits", "point") for p in list_arg(pts, "GroupOrbits", "points")]
    orbits, done = [], set()
    for p in todo:
        if p in done:
            continue
        orb, frontier = {p}, [p]
        while frontier:
            q = frontier.pop()
            for s in gens:
                r = s(q)
                if r not in orb:
                    orb.add(r)
                    frontier.append(r)
        done |= orb
        orbits.append(sorted(orb))
    orbits.sort(key=lambda o: o[0])
    return L(*(L(*o) for o in orbits))


# inert group heads: validate the parameter
def _named_group(h):
    @fn(h, area=AREA, status="added", notes=f"{h}[n] inert group object (permutation representation on 1..n) for "
        "GroupOrder, GroupElements, GroupMultiplicationTable, GroupOrbits, Permute, ...")
    def impl(n: Expr) -> Expr:
        if _intq(n):
            k = int(n)
            if k < 0 or (k == 0 and h in ("CyclicGroup", "DihedralGroup")):
                raise ValueError(f"{h}: invalid degree {k}")
        raise Unevaluated()
    return impl


for _h in ("SymmetricGroup", "AlternatingGroup", "CyclicGroup", "DihedralGroup"):
    _named_group(_h)


@fn("AbelianGroup", area=AREA, status="added", notes="AbelianGroup[{n1, n2, ...}] inert object (direct product of "
    "cyclic groups); GroupOrder supported")
def _abelian(ns: Expr) -> Expr:
    if is_list(ns):
        for v in ns.args:
            if _intq(v) and int(v) <= 0:
                raise ValueError("AbelianGroup: orders must be positive integers")
    raise Unevaluated()


@fn("PermutationGroup", area=AREA, status="added", notes="PermutationGroup[{g1, ...}] inert group generated by "
    "Cycles objects or permutation lists")
def _pgroup(gens: Expr) -> Expr:
    if is_list(gens):
        for p in gens.args:
            perm_of(p, "PermutationGroup")
    raise Unevaluated()


# ─────────────────────────────── counting functions ───────────────────────────────

def _frob_check(a, fname):
    ns = [int_value(v, fname, "element") for v in list_arg(a, fname, "list", 1)]
    if any(v <= 0 for v in ns):
        raise ValueError(f"{fname}: the elements must be positive integers")
    return ns


@fn("FrobeniusNumber", area=AREA, status="added",
    notes="largest integer not a non-negative integer combination of the a_i; -1 if some a_i = 1; Infinity when "
          "GCD > 1 (round-robin shortest paths over residues of the smallest a_i)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _frobnum(a: Expr) -> Expr:
    ns = sorted(set(_frob_check(a, "FrobeniusNumber")))
    g = 0
    for v in ns:
        g = math.gcd(g, v)
    if g != 1:
        return Symbol("Infinity")
    if ns[0] == 1:
        return -1
    m = ns[0]
    if m > 10 ** 6:
        raise ValueError("FrobeniusNumber: smallest element too large")
    # Dijkstra over residues mod m: dist[r] = smallest representable number = r (mod m)
    import heapq
    dist = [None] * m
    dist[0] = 0
    heap = [(0, 0)]
    while heap:
        d, r = heapq.heappop(heap)
        if d != dist[r]:
            continue
        for v in ns[1:]:
            nd, nr = d + v, (r + v) % m
            if dist[nr] is None or nd < dist[nr]:
                dist[nr] = nd
                heapq.heappush(heap, (nd, nr))
    return max(dist) - m


@fn("FrobeniusSolve", area=AREA, status="added",
    notes="all non-negative integer solutions of a.x == b, sorted lexicographically; the 3-argument form (first m "
          "solutions) raises because Wolfram's selection order is not reproduced"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _frobsolve(a: Expr, b: Integer) -> ListOf[Expr]:
    ns = _frob_check(a, "FrobeniusSolve")
    t = int_value(b, "FrobeniusSolve", "target")
    if t < 0:
        return L()
    out = []

    def rec(i, rem, acc):
        if i == len(ns) - 1:
            if rem % ns[i] == 0:
                out.append(acc + [rem // ns[i]])
                if len(out) > _MAX_ELEMENTS:
                    raise ValueError("FrobeniusSolve: too many solutions to list")
            return
        for k in range(rem // ns[i] + 1):
            rec(i + 1, rem - k * ns[i], acc + [k])
    rec(0, t, [])
    return L(*(L(*s) for s in out))


def _exact_int_or_uneval(n, fname):
    if _intq(n):
        return int(n)
    if is_numeric_value(n):
        raise ValueError(f"{fname}: argument {n!r} must be an integer")
    raise Unevaluated()


@fn("Hyperfactorial", area=AREA, status="added", listable=True,
    notes="Product[k^k, {k, n}] for integers n >= 0 (Hyperfactorial[0] = 1); approximate real arguments via mpmath "
          "hyperfac; exact non-integers stay unevaluated"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _hyperfac(n: Expr) -> Expr:
    if _intq(n):
        k = int(n)
        if k < 0:
            raise NotImplementedError("Hyperfactorial: negative integer arguments are not supported")
        out = 1
        for j in range(2, k + 1):
            out *= j ** j
        return out
    from .core import has_float
    if has_float(n) and is_numeric_value(n):
        import mpmath
        v = complex(sp.N(S(n), 20))
        if v.imag != 0:
            raise NotImplementedError("Hyperfactorial: complex arguments are not supported")
        return float(mpmath.hyperfac(v.real))
    if is_numeric_value(n):
        raise Unevaluated()
    raise Unevaluated()


@fn("PolygonalNumber", area=AREA, status="added", listable=True,
    notes="PolygonalNumber[n] = n (n + 1)/2 (triangular); PolygonalNumber[r, n] = n ((r - 2) n - (r - 4))/2; "
          "exact numeric arguments of any kind; symbolic arguments stay unevaluated"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _polygonal(r: Expr, n: Expr = None) -> Expr:
    if n is None:
        r, n = 3, r
    if not (is_numeric_value(r) and is_numeric_value(n)):
        raise Unevaluated()
    from .core import numeric_out
    sr, sn = S(r), S(n)
    return numeric_out(sp.expand(sn * ((sr - 2) * sn - (sr - 4)) / 2), r, n)


@fn("AlternatingFactorial", area=AREA, status="added", listable=True,
    notes="Sum[(-1)^(n - k) k!, {k, n}] for integers n >= 0 (OEIS A005165); other arguments stay unevaluated"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _altfact(n: Expr) -> Expr:
    k = _exact_int_or_uneval(n, "AlternatingFactorial")
    if k < 0:
        raise NotImplementedError("AlternatingFactorial: negative integers are not supported")
    out, f = 0, 1
    for j in range(1, k + 1):
        f *= j
        out = f - out
    return out


@fn("Fibonorial", area=AREA, status="added", listable=True,
    notes="Product[Fibonacci[k], {k, n}] for integers n >= 0 (OEIS A003266); other arguments stay unevaluated"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _fibonorial(n: Expr) -> Expr:
    k = _exact_int_or_uneval(n, "Fibonorial")
    if k < 0:
        raise NotImplementedError("Fibonorial: negative integers are not supported")
    out, a, b = 1, 1, 1
    for _ in range(k):
        out *= a
        a, b = b, a + b
    return out
