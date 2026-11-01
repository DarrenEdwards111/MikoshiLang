"""Wolfram batch W09 -- group theory: sporadic simple groups, stabilizers, centralizers, coset representatives,
stabilizer chains, words in generators, cycle index polynomials and the GroupActionBase option.

Permutation groups use the representation from wl_combinat (Perm objects, Cycles output, PermutationProduct[a, b]
applies a first) with SymPy's Schreier-Sims implementation for the larger computations.  The five Mathieu groups
carry the Wolfram Language's permutation generators; the other sporadic groups (and the Tits group) support
GroupOrder only, and representation-dependent operations on them raise.
"""
from __future__ import annotations

import math
from fractions import Fraction

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, record, Unevaluated, REGISTRY, S, R, L, is_list, is_numeric_value
from .wl_common import ev, split_options
from . import wl_combinat as C
from . import wl_numbertheory2 as NT

AREA = "group theory (W09)"
ADDED = dict(area=AREA, status="added")


def _intq(x):
    return (isinstance(x, int) and not isinstance(x, bool)) or isinstance(x, sp.Integer)


# ─────────────────────────── sporadic groups ───────────────────────────

SPORADIC_ORDERS = {
    "MathieuGroupM11": 7920, "MathieuGroupM12": 95040, "MathieuGroupM22": 443520, "MathieuGroupM23": 10200960,
    "MathieuGroupM24": 244823040, "JankoGroupJ1": 175560, "JankoGroupJ2": 604800, "JankoGroupJ3": 50232960,
    "JankoGroupJ4": 86775571046077562880, "HigmanSimsGroupHS": 44352000, "McLaughlinGroupMcL": 898128000,
    "HeldGroupHe": 4030387200, "RudvalisGroupRu": 145926144000, "SuzukiGroupSuz": 448345497600,
    "ONanGroupON": 460815505920, "ConwayGroupCo3": 495766656000, "ConwayGroupCo2": 42305421312000,
    "ConwayGroupCo1": 4157776806543360000, "FischerGroupFi22": 64561751654400,
    "FischerGroupFi23": 4089470473293004800, "FischerGroupFi24Prime": 1255205709190661721292800,
    "HaradaNortonGroupHN": 273030912000000, "LyonsGroupLy": 51765179004000000,
    "ThompsonGroupTh": 90745943887872000, "BabyMonsterGroupB": 4154781481226426191177580544000000,
    "MonsterGroupM": 808017424794512875886459904961710757005754368000000000, "TitsGroupT": 17971200,
}

_MATHIEU_GENS = {
    "MathieuGroupM11": [[(2, 10), (4, 11), (5, 7), (8, 9)], [(1, 4, 3, 8), (2, 5, 6, 9)]],
    "MathieuGroupM12": [[(1, 4), (3, 10), (5, 11), (6, 12)], [(1, 8, 9), (2, 3, 4), (5, 12, 11), (6, 10, 7)]],
    "MathieuGroupM22": [[(1, 13), (2, 8), (3, 16), (4, 12), (6, 22), (7, 17), (9, 10), (11, 14)],
                        [(1, 22, 3, 21), (2, 18, 4, 13), (5, 12), (6, 11, 7, 15), (8, 14, 20, 10), (17, 19)]],
    "MathieuGroupM23": [[(1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (19, 20), (21, 22)],
                        [(1, 16, 11, 3), (2, 9, 21, 12), (4, 5, 8, 23), (6, 22, 14, 18), (13, 20), (15, 17)]],
    "MathieuGroupM24": [[(1, 4), (2, 7), (3, 17), (5, 13), (6, 9), (8, 15), (10, 19), (11, 18), (12, 21), (14, 16),
                         (20, 24), (22, 23)],
                        [(1, 4, 6), (2, 21, 14), (3, 9, 15), (5, 18, 10), (13, 17, 16), (19, 24, 23)]],
}
_MATHIEU_DEG = {"MathieuGroupM11": 11, "MathieuGroupM12": 12, "MathieuGroupM22": 22, "MathieuGroupM23": 23,
                "MathieuGroupM24": 24}

_NAMES = {
    "MathieuGroupM11": "the Mathieu group M11", "MathieuGroupM12": "the Mathieu group M12",
    "MathieuGroupM22": "the Mathieu group M22", "MathieuGroupM23": "the Mathieu group M23",
    "MathieuGroupM24": "the Mathieu group M24", "JankoGroupJ1": "the Janko group J1",
    "JankoGroupJ2": "the Janko group J2", "JankoGroupJ3": "the Janko group J3", "JankoGroupJ4": "the Janko group J4",
    "HigmanSimsGroupHS": "the Higman-Sims group HS", "McLaughlinGroupMcL": "the McLaughlin group McL",
    "HeldGroupHe": "the Held group He", "RudvalisGroupRu": "the Rudvalis group Ru",
    "SuzukiGroupSuz": "the Suzuki group Suz", "ONanGroupON": "the O'Nan group O'N",
    "ConwayGroupCo3": "the Conway group Co3", "ConwayGroupCo2": "the Conway group Co2",
    "ConwayGroupCo1": "the Conway group Co1", "FischerGroupFi22": "the Fischer group Fi22",
    "FischerGroupFi23": "the Fischer group Fi23", "FischerGroupFi24Prime": "the Fischer group Fi24'",
    "HaradaNortonGroupHN": "the Harada-Norton group HN", "LyonsGroupLy": "the Lyons group Ly",
    "ThompsonGroupTh": "the Thompson group Th", "BabyMonsterGroupB": "the Baby Monster group B",
    "MonsterGroupM": "the Monster group M", "TitsGroupT": "the Tits group 2F4(2)'",
}


def _sporadic_impl(name):
    rep = ("permutation representation on 1.." + str(_MATHIEU_DEG[name]) + " (GroupGenerators, GroupElements, "
           "GroupOrbits, GroupStabilizer, ... work)") if name in _MATHIEU_GENS else \
        ("GroupOrder is supported; operations that need its permutation representation raise")

    @fn(name, **ADDED, notes=f"{name}[] represents {_NAMES[name]}, of order {SPORADIC_ORDERS[name]}; {rep}")
    def impl(*args: Expr) -> Expr:
        if args:
            raise ValueError(f"{name} takes no arguments")
        raise Unevaluated()
    return impl


for _n in SPORADIC_ORDERS:
    _sporadic_impl(_n)


def _is_sporadic(g):
    return isinstance(g, Expr) and isinstance(g.head, str) and g.head in SPORADIC_ORDERS and not g.args


# extend wl_combinat's group machinery
C._GROUP_HEADS = tuple(C._GROUP_HEADS) + tuple(SPORADIC_ORDERS)
_orig_group_gens = C._group_gens


def _group_gens(g, fname):
    if isinstance(g, Expr) and g.head in SPORADIC_ORDERS:
        if g.args:
            raise ValueError(f"{g.head} takes no arguments")
        if g.head in _MATHIEU_GENS:
            return [C.Perm.from_cycles(cs) for cs in _MATHIEU_GENS[g.head]], _MATHIEU_DEG[g.head]
        raise NotImplementedError(f"{fname}: the permutation representation of {_NAMES[g.head]} is not available "
                                  "in MikoshiLang (only GroupOrder is supported)")
    return _orig_group_gens(g, fname)


C._group_gens = _group_gens


def _gorder_pre(args):
    if len(args) == 1 and _is_sporadic(args[0]):
        return SPORADIC_ORDERS[args[0].head]
    raise NT.NOT_MINE()


NT.wrap_registry("GroupOrder", _gorder_pre)


def _wolfram_gens(g, fname):
    """The Wolfram Language's GroupGenerators where known (as Perm objects)."""
    if isinstance(g, Expr) and g.head == "DihedralGroup" and len(g.args) == 1 and _intq(g.args[0]) and \
            int(g.args[0]) >= 3:
        n = int(g.args[0])
        refl = C.Perm({i: n + 1 - i for i in range(1, n + 1)})
        rot = C.Perm.from_cycles([tuple(range(1, n + 1))])
        return [refl, rot]
    if isinstance(g, Expr) and g.head in _MATHIEU_GENS:
        return [C.Perm.from_cycles(cs) for cs in _MATHIEU_GENS[g.head]]
    return None


def _ggens_pre(args):
    if len(args) != 1:
        raise NT.NOT_MINE()
    gens = _wolfram_gens(args[0], "GroupGenerators")
    if gens is None:
        if _is_sporadic(args[0]):
            _group_gens(args[0], "GroupGenerators")
        raise NT.NOT_MINE()
    return L(*[C._cycles_expr(p.cycles()) for p in gens])


NT.wrap_registry("GroupGenerators", _ggens_pre)


def generators(g, fname):
    """Generators in the Wolfram order (GroupGenerators) as Perm objects, and the degree."""
    w = _wolfram_gens(g, fname)
    gens, deg = C._group_gens(g, fname)
    if w is not None:
        return w, max(deg, max((p.degree() for p in w), default=0))
    if g.head in ("SymmetricGroup", "CyclicGroup", "PermutationGroup"):
        out, seen = [], set()
        for p in gens:
            if p.m and p.key() not in seen:
                seen.add(p.key())
                out.append(p)
        return out, deg
    raise NotImplementedError(f"{fname}: the Wolfram generating set of {g.head} is not reproduced; use "
                              "PermutationGroup with explicit generators")


# ─────────────────────────── GroupActionBase ───────────────────────────

record("GroupActionBase", AREA, "added", "option for GroupElements and GroupStabilizerChain: GroupActionBase -> "
       "{b1, b2, ...} orders the elements by their images of the base points (then the remaining points), and "
       "starts the stabilizer chain with the given base")


def _base_key(base, deg):
    order = list(base) + [i for i in range(1, deg + 1) if i not in base]
    rank = {p: i for i, p in enumerate(order)}

    def key(p):
        return [rank.get(p(i), i + 10 ** 9) for i in order]
    return key


def _base_opt(opts, fname):
    b = opts.get("GroupActionBase")
    if b is None or (isinstance(b, Symbol) and b.name == "Automatic"):
        return None
    if not is_list(b) or not all(_intq(x) and int(x) > 0 for x in b.args):
        raise ValueError(f"{fname}: GroupActionBase must be a list of positive integers")
    pts = [int(x) for x in b.args]
    if len(set(pts)) != len(pts):
        raise ValueError(f"{fname}: GroupActionBase points must be distinct")
    return pts


def _gelements_pre(args):
    pos, opts = split_options(args, ("GroupActionBase",), "GroupElements")
    if "GroupActionBase" not in opts:
        raise NT.NOT_MINE()
    base = _base_opt(opts, "GroupElements")
    if not pos or not C._is_group(pos[0]):
        raise Unevaluated()
    g = pos[0]
    gens, deg = C._group_gens(g, "GroupElements")
    elems = C._closure(gens)
    deg = max([deg] + (base or []))
    if base is not None:
        elems.sort(key=_base_key(base, deg))
    else:
        elems.sort(key=lambda p: p.images(deg))
    out = [C._cycles_expr(p.cycles()) for p in elems]
    if len(pos) == 2:
        sel = []
        for r in pos[1].args if is_list(pos[1]) else [pos[1]]:
            k = int(r)
            sel.append(out[k - 1] if k > 0 else out[k])
        return L(*sel)
    return L(*out)


NT.wrap_registry("GroupElements", _gelements_pre)


# ─────────────────────────── SymPy bridges ───────────────────────────

def _sympy(gens, deg):
    from sympy.combinatorics import Permutation, PermutationGroup
    n = max(deg, 1)
    perms = [Permutation([p(i) - 1 for i in range(1, n + 1)]) for p in gens] or [Permutation(list(range(n)))]
    return PermutationGroup(perms), n


def _from_sympy_perm(sp_perm):
    return C.Perm({i + 1: sp_perm(i) + 1 for i in range(sp_perm.size)})


def _to_sympy_perm(p, n):
    from sympy.combinatorics import Permutation
    return Permutation([p(i) - 1 for i in range(1, n + 1)])


def _pgroup(perms):
    out, seen = [], set()
    for p in perms:
        if p.m and p.key() not in seen:
            seen.add(p.key())
            out.append(C._cycles_expr(p.cycles()))
    return Expr("PermutationGroup", L(*out))


def _group_arg(g, fname):
    if not C._is_group(g):
        raise Unevaluated()
    gens, deg = C._group_gens(g, fname)
    return gens, deg


def _reduce_gens(G, n):
    """A smaller generating set of the SymPy group G (drop redundant generators greedily)."""
    from sympy.combinatorics import PermutationGroup
    gens = [p for p in G.generators if not p.is_Identity]
    order = G.order()
    keep = []
    for p in gens:
        if keep and PermutationGroup(keep).contains(p):
            continue
        keep.append(p)
    if keep and PermutationGroup(keep).order() != order:
        keep = gens
    return keep


def _stab_points(G, n, pts):
    H = G
    for p in pts:
        if p - 1 >= n:
            continue
        H = H.stabilizer(p - 1)
    return H


@fn("GroupStabilizer", **ADDED, notes="GroupStabilizer[group, {p1, ...}]: the pointwise stabilizer subgroup as a "
    "PermutationGroup (generators may differ from the Wolfram Language's choice); the Permute action form is not "
    "supported")
def _gstab(g: Expr, pts: Expr, *action: Expr) -> Expr:
    if action:
        raise NotImplementedError("GroupStabilizer: actions other than the action on points are not supported")
    gens, deg = _group_arg(g, "GroupStabilizer")
    plist = [int(p) for p in pts.args] if is_list(pts) and all(_intq(p) and int(p) > 0 for p in pts.args) else None
    if plist is None:
        raise ValueError("GroupStabilizer: the points must be a list of positive integers")
    if not gens:
        return _pgroup([])
    G, n = _sympy(gens, deg)
    H = _stab_points(G, n, plist)
    return _pgroup([_from_sympy_perm(p) for p in _reduce_gens(H, n)])


@fn("GroupSetwiseStabilizer", **ADDED, notes="GroupSetwiseStabilizer[group, {p1, ...}]: the subgroup mapping the "
    "set of points to itself, as a PermutationGroup (generators may differ from the Wolfram Language's choice); "
    "the Permute action form is not supported")
def _gsetstab(g: Expr, pts: Expr, *action: Expr) -> Expr:
    if action:
        raise NotImplementedError("GroupSetwiseStabilizer: actions other than the action on points are not "
                                  "supported")
    gens, deg = _group_arg(g, "GroupSetwiseStabilizer")
    plist = [int(p) for p in pts.args] if is_list(pts) and all(_intq(p) and int(p) > 0 for p in pts.args) else None
    if plist is None:
        raise ValueError("GroupSetwiseStabilizer: the points must be a list of positive integers")
    if not gens:
        return _pgroup([])
    s = set(plist)
    G, n = _sympy(gens, deg)
    inside = [p - 1 for p in s if p <= n]
    if G.order() <= 200000:
        elems = [p for p in C._closure(gens) if {p(i) for i in s} == s]
        from sympy.combinatorics import PermutationGroup
        H = PermutationGroup([_to_sympy_perm(p, n) for p in elems] or [_to_sympy_perm(C.Perm({}), n)])
    else:
        sset = set(inside)
        H = G.subgroup_search(lambda x: {x(i) for i in sset} == sset)
    return _pgroup([_from_sympy_perm(p) for p in _reduce_gens(H, n)])


@fn("GroupCentralizer", **ADDED, notes="GroupCentralizer[group, g]: the subgroup of elements commuting with the "
    "permutation g, as a PermutationGroup (generators may differ from the Wolfram Language's choice)")
def _gcentral(g: Expr, x: Expr) -> Expr:
    gens, deg = _group_arg(g, "GroupCentralizer")
    perm, _ = C.perm_of(x, "GroupCentralizer")
    n = max(deg, perm.degree(), 1)
    if not gens:
        return _pgroup([])
    G, _ = _sympy(gens, n)
    H = G.centralizer(_to_sympy_perm(perm, n))
    return _pgroup([_from_sympy_perm(p) for p in _reduce_gens(H, n)])


def _less_key(n):
    return lambda p: p.images(n)


@fn("RightCosetRepresentative", **ADDED, notes="RightCosetRepresentative[group, g]: the smallest permutation (in "
    "image-list order, as Sort[..., Less]) of the right coset {PermutationProduct[h, g] : h in group}")
def _rcoset(g: Expr, x: Expr) -> Expr:
    gens, deg = _group_arg(g, "RightCosetRepresentative")
    perm, kind = C.perm_of(x, "RightCosetRepresentative")
    n = max(deg, perm.degree())
    if not gens:
        return C._out(perm, kind, n)
    G, _ = _sympy(gens, n)
    if G.order() <= 200000:
        best = min((h.then(perm) for h in C._closure(gens)), key=_less_key(n))
        return C._out(best, kind, n)
    # greedy minimisation along a stabilizer chain: choose images of 1, 2, ... as small as possible
    cur = perm
    H = G
    for pt in range(1, n + 1):
        orbit = H.orbit(pt - 1)
        # h then cur maps pt to cur(h(pt)); choose h(pt) = q in orbit minimising cur(q)
        q = min(orbit, key=lambda o: cur(o + 1))
        t = H.orbit_transversal(pt - 1, pairs=True)
        trans = dict(t)
        h = _from_sympy_perm(trans[q])
        cur = h.then(cur)
        H = H.stabilizer(pt - 1)
        if H.order() == 1:
            break
    return C._out(cur, kind, n)


@fn("GroupStabilizerChain", **ADDED, notes="GroupStabilizerChain[group]: {{} -> G, {b1} -> G_b1, {b1, b2} -> ..., "
    "base -> PermutationGroup[{}]} with the base chosen greedily (smallest point moved by the current stabilizer) "
    "or given by GroupActionBase; the stabilizers are PermutationGroup objects whose generators may differ from "
    "the Wolfram Language's choice")
def _gstabchain(g: Expr, *opts: Option) -> ListOf[Expr]:
    pos, o = split_options(opts, ("GroupActionBase",), "GroupStabilizerChain")
    if pos:
        raise ValueError("GroupStabilizerChain takes one group argument")
    base = _base_opt(o, "GroupStabilizerChain") or []
    gens, deg = _group_arg(g, "GroupStabilizerChain")
    n = max([deg] + base)
    G, _ = _sympy(gens, n)
    chain = [Expr("Rule", L(), _pgroup([_from_sympy_perm(p) for p in _reduce_gens(G, n)]))]
    H, pts = G, []
    queue = list(base)
    while H.order() > 1:
        if queue:
            b = queue.pop(0)
        else:
            moved = sorted({i for p in H.generators for i in range(n) if p(i) != i})
            b = moved[0] + 1
        pts.append(b)
        H = H.stabilizer(b - 1)
        chain.append(Expr("Rule", L(*pts), _pgroup([_from_sympy_perm(p) for p in _reduce_gens(H, n)])))
    for b in queue:            # remaining given base points
        pts.append(b)
        chain.append(Expr("Rule", L(*pts), _pgroup([])))
    return L(*chain)


# ─────────────────────────── words in the generators ───────────────────────────

def _word_value(gens, word, fname):
    cur = C.Perm({})
    for w in word:
        if not _intq(w) or int(w) == 0 or abs(int(w)) > len(gens):
            raise ValueError(f"{fname}: word letters must be nonzero integers between -{len(gens)} and {len(gens)}")
        p = gens[abs(int(w)) - 1]
        cur = cur.then(p if int(w) > 0 else p.inverse())
    return cur


@fn("GroupElementFromWord", **ADDED, notes="GroupElementFromWord[group, {i1, i2, ...}] = PermutationProduct of the "
    "generators GroupGenerators[group][[|ik|]] (inverted for negative ik), applied left to right")
def _gfromword(g: Expr, word: Expr) -> Expr:
    if not C._is_group(g):
        raise Unevaluated()
    if not is_list(word):
        raise ValueError("GroupElementFromWord: the word must be a list of integers")
    gens, deg = generators(g, "GroupElementFromWord")
    return C._cycles_expr(_word_value(gens, list(word.args), "GroupElementFromWord").cycles())


@fn("GroupElementToWord", **ADDED, notes="GroupElementToWord[group, g]: a word {i1, i2, ...} in the generators "
    "(negative for inverses) whose product is g -- a shortest word (breadth-first search) for groups of at most "
    "10^5 elements, otherwise a word from a stabilizer chain; words are not unique and may differ from the "
    "Wolfram Language's")
def _gtoword(g: Expr, x: Expr) -> ListOf[Expr]:
    if not C._is_group(g):
        raise Unevaluated()
    perm, _ = C.perm_of(x, "GroupElementToWord")
    gens, deg = generators(g, "GroupElementToWord")
    if not perm.m:
        return L()
    letters = []
    for i, p in enumerate(gens):
        letters.append((i + 1, p))
        if p.inverse().key() != p.key():
            letters.append((-(i + 1), p.inverse()))
    n = max(deg, perm.degree())
    G, _ = _sympy(gens, n)
    if not G.contains(_to_sympy_perm(perm, n)):
        raise ValueError(f"GroupElementToWord: {x} is not an element of the group")
    if G.order() <= 100000:
        target = perm.key()
        prev = {C.Perm({}).key(): None}
        frontier = [C.Perm({})]
        while frontier:
            nxt = []
            for e in frontier:
                for lab, p in letters:
                    q = e.then(p)
                    k = q.key()
                    if k not in prev:
                        prev[k] = (e.key(), lab)
                        if k == target:
                            word = []
                            while prev[k] is not None:
                                k, lab2 = prev[k]
                                word.append(lab2)
                            return L(*reversed(word))
                        nxt.append(q)
            frontier = nxt
        raise ValueError("GroupElementToWord: element not found")
    # large groups: Schreier-vector words along a stabilizer chain
    word = _chain_word(gens, perm, n)
    if _word_value(gens, word, "GroupElementToWord").key() != perm.key():
        raise ArithmeticError("GroupElementToWord: word reconstruction failed")
    return L(*word)


def _chain_word(gens, perm, n):
    """Word for perm via orbit transversals stored as words (stabilizer chain sifting)."""
    letters = []
    for i, p in enumerate(gens):
        letters.append((i + 1, p))
        letters.append((-(i + 1), p.inverse()))
    # build level by level: subgroup generators as words
    level_gens = [([lab], p) for lab, p in letters]
    remaining = perm
    out_word = []
    from sympy.combinatorics import PermutationGroup
    for pt in range(1, n + 1):
        if not remaining.m:
            break
        # orbit of pt under the current level generators, with words
        trans = {pt: ([], C.Perm({}))}
        frontier = [pt]
        while frontier:
            nf = []
            for q in frontier:
                w, t = trans[q]
                for gw, gp in level_gens:
                    r = gp(q)
                    if r not in trans:
                        trans[r] = (w + gw, t.then(gp))
                        nf.append(r)
            frontier = nf
        img = remaining(pt)
        if img not in trans:
            raise ValueError("GroupElementToWord: element not in group")
        w, t = trans[img]
        # remaining = s then t  with s fixing pt  ->  s = remaining then t^-1
        out_word = w + out_word
        remaining = remaining.then(t.inverse())
        # Schreier generators of the stabilizer of pt (as words); keep a bounded set
        new_gens, seen = [], set()
        for q, (wq, tq) in trans.items():
            for gw, gp in level_gens:
                r = gp(q)
                wr, tr = trans[r]
                s = tq.then(gp).then(tr.inverse())
                if s.m and s.key() not in seen:
                    seen.add(s.key())
                    inv_wr = [-l for l in reversed(wr)]
                    new_gens.append((wq + gw + inv_wr, s))
        G2 = None
        reduced = []
        for wv, s in new_gens:
            sp_s = _to_sympy_perm(s, n)
            if G2 is not None and G2.contains(sp_s):
                continue
            reduced.append((wv, s))
            G2 = PermutationGroup([_to_sympy_perm(t2, n) for _, t2 in reduced])
        level_gens = reduced
    if remaining.m:
        raise ValueError("GroupElementToWord: element not in group")
    return out_word


# ─────────────────────────── cycle index ───────────────────────────

def _cycle_monomial(p, deg, xs):
    counts = {}
    moved = 0
    for c in p.cycles():
        counts[len(c)] = counts.get(len(c), 0) + 1
        moved += len(c)
    fixed = deg - moved
    if fixed < 0:
        raise ValueError("CycleIndexPolynomial: the degree is smaller than the largest moved point")
    if fixed:
        counts[1] = counts.get(1, 0) + fixed
    term = sp.Integer(1)
    for k, m in counts.items():
        if k > len(xs):
            raise ValueError(f"CycleIndexPolynomial: a cycle of length {k} needs at least {k} variables")
        term *= xs[k - 1] ** m
    return term


@fn("CycleIndexPolynomial", **ADDED, notes="CycleIndexPolynomial[perm, {x1, ..., xn}] = Product[x_k^(number of "
    "k-cycles)] (fixed points up to PermutationMax, or up to the degree given as a third argument, count as "
    "1-cycles); for a group the average over its elements")
def _cycleindex(g: Expr, xs: Expr, d: Expr = None) -> Expr:
    if not is_list(xs) or not xs.args:
        raise ValueError("CycleIndexPolynomial: the variables must be a non-empty list")
    vs = [S(v) for v in xs.args]
    if C._is_group(g):
        gens, deg = C._group_gens(g, "CycleIndexPolynomial")
        if g.head in ("SymmetricGroup", "AlternatingGroup", "CyclicGroup", "DihedralGroup") and \
                _intq(g.args[0]):
            deg = int(g.args[0])
        if d is not None:
            deg = int(d) if _intq(d) else deg
        G, _ = _sympy(gens, max(deg, 1)) if gens else (None, None)
        if G is not None and G.order() > 200000:
            raise ValueError("CycleIndexPolynomial: the group is too large")
        elems = C._closure(gens) if gens else [C.Perm({})]
        tot = sp.Integer(0)
        for p in elems:
            tot += _cycle_monomial(p, deg, vs)
        return R(sp.expand(tot / len(elems)))
    perm, _ = C.perm_of(g, "CycleIndexPolynomial")
    deg = perm.degree() if d is None else None
    if d is not None:
        if not _intq(d) or int(d) < 0:
            raise ValueError("CycleIndexPolynomial: the degree must be a non-negative integer")
        deg = int(d)
    if not C._is_cycles(g):
        deg = max(deg, len(g.args)) if d is None else deg
    return R(_cycle_monomial(perm, deg, vs))
