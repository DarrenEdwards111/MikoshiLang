"""Wolfram coverage batch W03 -- Tree objects and tree operations.

A tree is held in Wolfram's standard form Tree[data, children] where children is None (a leaf), a list of trees or
an association of trees; trailing presentation options (TreeElementLabel -> ..., TreeLayout -> ..., ...) are kept
on the expression and ignored by computations.  Levels follow Wolfram's level specifications (level 0 is the root,
level -1 the leaves; an empty list of children contributes one level), and traversals default to Wolfram's
left-to-right, depth-first, children-before-parents order.  Unimplemented argument forms raise a clear error.
"""
from __future__ import annotations

import random

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, record, Unevaluated
from .wl_common import L, ev, apply, is_sym, is_list, is_int, is_assoc, is_rule, split_options, int_value

AREA = "trees (W03)"
NONE = Symbol("None")
NULL = Symbol("Null")

TREE_OPTIONS = (
    "TreeElementCoordinates", "TreeElementLabel", "TreeElementLabelFunction", "TreeElementLabelStyle",
    "TreeElementShape", "TreeElementShapeFunction", "TreeElementSize", "TreeElementSizeFunction",
    "TreeElementStyle", "TreeElementStyleFunction", "TreeLayout", "ParentEdgeLabel", "ParentEdgeLabelFunction",
    "ParentEdgeLabelStyle", "ParentEdgeShapeFunction", "ParentEdgeStyle", "ParentEdgeStyleFunction",
    "MaxDisplayedChildren", "FormatType", "PerformanceGoal", "ImageSize", "BaseStyle", "Background",
    "AspectRatio", "PlotLabel", "LabelStyle", "Frame", "Axes", "Epilog", "Prolog", "PlotRange")
_OPTSET = set(TREE_OPTIONS)

for _n in TREE_OPTIONS[:17]:
    record(_n, AREA, "added", "presentation option of Tree (and the tree constructors); kept on the Tree "
                              "expression and ignored by computations")
record("TreeTraversalOrder", AREA, "added",
       "option of TreeMap, TreeScan, TreeLevel, TreeCases, TreePosition, TreeSelect: Automatic/\"DepthFirst\"/"
       "\"PostOrder\"/\"BottomUp\" (children before parents, default), \"PreOrder\"/\"TopDown\", \"BreadthFirst\"/"
       "\"LevelOrder\" (top-down); other orders raise")


class TreeError(ValueError):
    pass


def _is_opt(x):
    return is_rule(x) and isinstance(x.args[0], Symbol) and x.args[0].name in _OPTSET


def is_tree(x):
    return isinstance(x, Expr) and x.head == "Tree" and len(x.args) >= 2 and _valid_children(x.args[1]) \
        and all(_is_opt(o) for o in x.args[2:])


def _valid_children(c):
    if is_sym(c, "None"):
        return True
    if is_list(c):
        return all(isinstance(t, Expr) and t.head == "Tree" for t in c.args)
    if is_assoc(c):
        return all(is_rule(r) and isinstance(r.args[1], Expr) and r.args[1].head == "Tree" for r in c.args)
    return False


def _need_tree(x, fname):
    if is_tree(x):
        return x
    if isinstance(x, Symbol) or (isinstance(x, Expr) and x.head not in ("List", "Tree", "Association")):
        raise Unevaluated()
    raise TreeError(f"{fname}: {x} is not a Tree object")


def data(t):
    return t.args[0]


def kind(t):
    c = t.args[1]
    return "none" if is_sym(c, "None") else ("list" if is_list(c) else "assoc")


def kids(t):
    """[(key or None, subtree)]"""
    c = t.args[1]
    if is_sym(c, "None"):
        return []
    if is_list(c):
        return [(None, s) for s in c.args]
    return [(r.args[0], r.args[1]) for r in c.args]


def mk(d, knd, ks, opts=()):
    if knd == "none":
        ch = NONE
    elif knd == "list":
        ch = L(*(s for _, s in ks))
    else:
        ch = Expr("Association", *(Expr("Rule", k, s) for k, s in ks))
    return Expr("Tree", d, ch, *opts)


def rebuild(t, d=None, ks=None):
    return mk(data(t) if d is None else d, kind(t), kids(t) if ks is None else ks, t.args[2:])


def leaf(x):
    return x if is_tree(x) else Expr("Tree", x, NONE)


def height(t):
    ks = kids(t)
    if kind(t) == "none":
        return 0
    return 1 + max((height(s) for _, s in ks), default=0)


# ─────────────────────────── Tree constructor ───────────────────────────

@fn("Tree", area=AREA, status="added",
    notes="Tree[data, {subtrees}] / Tree[data, <|k -> subtree|>] / Tree[data, None] (leaf); Tree[{...}] = "
          "Tree[Null, {...}]; non-Tree children become leaves; presentation options are kept")
def _tree(*args):
    args = list(args)
    opts = []
    while args and _is_opt(args[-1]):
        opts.insert(0, args.pop())
    if len(args) == 1:
        if is_list(args[0]) or is_assoc(args[0]):
            args = [NULL, args[0]]
        else:
            raise TreeError(f"Tree: {args[0]} is not a list or association of subtrees")
    if len(args) != 2:
        raise TreeError("Tree: expected Tree[data, subtrees] or Tree[{subtrees}]")
    d, c = args
    if is_sym(c, "None"):
        new = NONE
    elif is_list(c):
        new = L(*(leaf(s) for s in c.args))
    elif is_assoc(c):
        new = Expr("Association", *(Expr("Rule", r.args[0], leaf(r.args[1])) for r in c.args))
    else:
        raise TreeError(f"Tree: the subtrees {c} must be a list, an association or None")
    out = Expr("Tree", d, new, *opts)
    if K.same(out, Expr("Tree", *args, *opts)) and len(args) == 2 and list(args) == [d, c]:
        if K.same(new, c):
            raise Unevaluated()
    return out


@fn("TreeQ", area=AREA, status="added", notes="True for a valid Tree object")
def _treeq(x):
    return is_tree(x)


@fn("TreeData", area=AREA, status="added", notes="the data at the root of a tree")
def _treedata(t):
    return data(_need_tree(t, "TreeData"))


@fn("TreeChildren", area=AREA, status="added", notes="the children (list, association or None) of the root")
def _treechildren(t):
    return _need_tree(t, "TreeChildren").args[1]


@fn("TreeLeafQ", area=AREA, status="added", notes="True for a leaf Tree[data, None]; False for anything else")
def _treeleafq(t):
    return is_tree(t) and kind(t) == "none"


# ─────────────────────────── traversal, levels ───────────────────────────

def nodes(t, order="post"):
    """[(position tuple, level, depth(=height+1), subtree, index)]; positions use integers for lists and Key[k]
    for associations."""
    out = []

    def go(s, pos, lev, idx):
        info = []
        h = 0
        for i, (k, c) in enumerate(kids(s)):
            p = pos + ((i + 1) if k is None else Expr("Key", k),)
            sub_h = go(c, p, lev + 1, i + 1)
            h = max(h, sub_h + 1)
        if kind(s) != "none":
            h = max(h, 1)
        rec = (pos, lev, h + 1, s, idx)
        info.append(rec)
        out.append(rec)
        return h
    if order == "post":
        go(t, (), 0, None)
        return out
    go(t, (), 0, None)
    bypos = {tuple(map(_tagp, r[0])): r for r in out}
    if order == "pre":
        res = []

        def pre(s, pos):
            res.append(bypos[tuple(map(_tagp, pos))])
            for i, (k, c) in enumerate(kids(s)):
                pre(c, pos + ((i + 1) if k is None else Expr("Key", k),))
        pre(t, ())
        return res
    if order == "bfs":
        res = []
        q = [(t, ())]
        while q:
            nq = []
            for s, pos in q:
                res.append(bypos[tuple(map(_tagp, pos))])
                for i, (k, c) in enumerate(kids(s)):
                    nq.append((c, pos + ((i + 1) if k is None else Expr("Key", k),)))
            q = nq
        return res
    raise TreeError(f"unknown traversal order {order}")


def _tagp(p):
    return p if isinstance(p, int) else K.tag(p)


def traversal(opts, fname):
    o = opts.get("TreeTraversalOrder")
    if o is None or is_sym(o, "Automatic"):
        return "post"
    name = o if isinstance(o, str) else (o.name if isinstance(o, Symbol) else None)
    if name in ("DepthFirst", "PostOrder", "BottomUp", "InnerOuter"):
        return "post"
    if name in ("PreOrder", "TopDown", "OuterInner"):
        return "pre"
    if name in ("BreadthFirst", "LevelOrder"):
        return "bfs"
    if is_list(o) and len(o.args) == 2 and all(isinstance(v, str) for v in o.args):
        a, b = o.args
        if a == "DepthFirst" and b in ("PostOrder", "BottomUp", "InnerOuter"):
            return "post"
        if a == "DepthFirst" and b in ("PreOrder", "TopDown", "OuterInner"):
            return "pre"
        if a in ("BreadthFirst", "LevelOrder") and b in ("TopDown", "PreOrder", "OuterInner"):
            return "bfs"
    raise NotImplementedError(f"{fname}: TreeTraversalOrder -> {o} is not supported")


def tree_levelspec(spec, fname, default=(0, KL.INF)):
    if spec is None:
        return default
    if spec == "Leaves":
        return (-1, -1)
    if spec == "NonLeaves":
        return (0, -2)
    return KL.level_spec(spec, default, fname)


def in_spec(rec, spec):
    return KL.in_level(rec[1], rec[2], spec)


def _split_elem(spec):
    if is_rule(spec):
        return spec.args[0], spec.args[1]
    return spec, None


def _with_opts(args, fname, n_min, n_max):
    pos, opts = split_options(args, {"TreeTraversalOrder"} | _OPTSET, fname)
    if not n_min <= len(pos) <= n_max:
        raise TreeError(f"{fname}: called with {len(pos)} arguments; {n_min} to {n_max} expected")
    return pos, opts


def _elem(rec, which, fname):
    pos, lev, dep, s, idx = rec
    if which is None or which == "Subtree":
        return s
    if which == "Data":
        return data(s)
    if which == "Position":
        return L(*pos)
    if which == "Level":
        return lev
    if which == "NegativeLevel":
        return -dep
    if which == "LevelPair":
        return L(lev, -dep)
    if which == "Children":
        return s.args[1]
    if which == "ChildrenData":
        c = s.args[1]
        if kind(s) == "none":
            return NONE
        if kind(s) == "list":
            return L(*(data(x) for x in c.args))
        return Expr("Association", *(Expr("Rule", r.args[0], data(r.args[1])) for r in c.args))
    raise NotImplementedError(f"{fname}: the element {which!r} is not supported")


@fn("TreeLevel", area=AREA, status="added",
    notes="TreeLevel[tree, levelspec] subtrees on the levels (children before parents); levelspec -> \"Data\" | "
          "\"Subtree\" | \"Position\" | \"Level\" ...; \"Leaves\", \"NonLeaves\"; TreeTraversalOrder option")
def _treelevel(*args):
    pos, opts = _with_opts(args, "TreeLevel", 1, 2)
    if len(pos) == 1:
        if is_tree(pos[0]):
            raise TreeError("TreeLevel: a level specification is required")
        raise Unevaluated()
    t = _need_tree(pos[0], "TreeLevel")
    spec, which = _split_elem(pos[1])
    ls = tree_levelspec(spec, "TreeLevel")
    if is_list(which):
        return L(*(L(*(_elem(r, w, "TreeLevel") for w in which.args))
                   for r in nodes(t, traversal(opts, "TreeLevel")) if in_spec(r, ls)))
    return L(*(_elem(r, which, "TreeLevel") for r in nodes(t, traversal(opts, "TreeLevel")) if in_spec(r, ls)))


def _limit(x, fname):
    if is_sym(x, "Infinity") or is_sym(x, "All"):
        return None
    return int_value(x, fname, "count", 0)


def _matching(fname, args, test, nmax=4, default_ls=(0, KL.INF)):
    pos, opts = _with_opts(args, fname, 2, nmax)
    t = _need_tree(pos[0], fname)
    ls = tree_levelspec(pos[2], fname) if len(pos) >= 3 else default_ls
    lim = _limit(pos[3], fname) if len(pos) >= 4 else None
    out = []
    for r in nodes(t, traversal(opts, fname)):
        if lim is not None and len(out) >= lim:
            break
        if in_spec(r, ls) and test(r):
            out.append(r)
    return out


def _dmatch(patt):
    return lambda r: K.first_match(patt, data(r[3]), ev) is not None


@fn("TreeCases", area=AREA, status="added",
    notes="TreeCases[tree, pattern, levelspec, n] subtrees whose data match (default levels {0, Infinity})")
def _treecases(*args):
    return L(*(r[3] for r in _matching("TreeCases", args, _dmatch(args[1]) if len(args) > 1 else None)))


@fn("TreeCount", area=AREA, status="added", notes="TreeCount[tree, pattern, levelspec] number of matching subtrees")
def _treecount(*args):
    return len(_matching("TreeCount", args, _dmatch(args[1]) if len(args) > 1 else None, 3))


@fn("TreePosition", area=AREA, status="added",
    notes="TreePosition[tree, pattern, levelspec, n] positions (children before parents; {} is the root; "
          "association children are addressed by Key[k])")
def _treeposition(*args):
    return L(*(L(*r[0]) for r in _matching("TreePosition", args, _dmatch(args[1]) if len(args) > 1 else None)))


@fn("TreeSelect", area=AREA, status="added",
    notes="TreeSelect[tree, crit], [tree, crit, n], [tree, crit, levelspec, n] subtrees with crit[subtree] True")
def _treeselect(*args):
    pos, opts = _with_opts(args, "TreeSelect", 2, 4)
    if len(pos) == 3:
        args = (pos[0], pos[1], Symbol("All"), pos[2]) + tuple(Expr("Rule", Symbol(k), v) for k, v in opts.items())
    crit = pos[1]
    return L(*(r[3] for r in _matching("TreeSelect", args, lambda r: apply(crit, r[3]) is True)))


for _n in ("TreeCases", "TreeCount", "TreePosition", "TreeSelect", "TreeLevel"):
    KL.operator_form(_n, 1, "first")


@fn("TreeLeaves", area=AREA, status="added", notes="the leaves (level -1) in depth-first order")
def _treeleaves(t):
    t = _need_tree(t, "TreeLeaves")
    return L(*(r[3] for r in nodes(t) if kind(r[3]) == "none"))


@fn("TreeLeafCount", area=AREA, status="added", notes="number of leaves; Tree[d, {}] has 0")
def _treeleafcount(t):
    t = _need_tree(t, "TreeLeafCount")
    return sum(1 for r in nodes(t) if kind(r[3]) == "none")


@fn("TreeSize", area=AREA, status="added", notes="number of subtrees (nodes)")
def _treesize(t):
    return len(nodes(_need_tree(t, "TreeSize")))


@fn("TreeDepth", area=AREA, status="added",
    notes="maximum level (empty child lists add a level; a leaf has depth 0); TreeDepth[tree, patt] maximum "
          "level of a subtree whose data matches patt")
def _treedepth(t, patt=None):
    t = _need_tree(t, "TreeDepth")
    if patt is None:
        return height(t)
    levs = [r[1] for r in nodes(t) if K.first_match(patt, data(r[3]), ev) is not None]
    if not levs:
        raise TreeError("TreeDepth: no subtree has data matching the pattern")
    return max(levs)


# ─────────────────────────── mapping, scanning, folding ───────────────────────────

def _map_tree(f, t, ls, which, order, fname):
    """Apply f to the elements of the subtrees on the given levels; the result replaces the data."""
    recs = nodes(t, "post")
    selected = {tuple(map(_tagp, r[0])): r for r in recs if in_spec(r, ls)}
    results = {}
    seq = nodes(t, order)
    # evaluate f in traversal order; "Subtree" elements see the already-mapped children (post-order only)
    if which in (None, "Data") or (is_list(which) and "Subtree" not in [w for w in which.args]):
        for r in seq:
            key = tuple(map(_tagp, r[0]))
            if key in selected:
                if is_list(which):
                    results[key] = apply(f, *(_elem(r, w, fname) for w in which.args))
                else:
                    results[key] = apply(f, data(r[3]))

        def build(s, pos):
            ks = [(k, build(c, pos + ((i + 1) if k is None else Expr("Key", k),))) for i, (k, c) in
                  enumerate(kids(s))]
            key = tuple(map(_tagp, pos))
            return rebuild(s, results[key] if key in results else data(s), ks)
        return build(t, ())
    if order != "post":
        raise NotImplementedError(f"{fname}: element {which} is only supported with the default traversal order")

    def build2(s, pos):
        ks = [(k, build2(c, pos + ((i + 1) if k is None else Expr("Key", k),))) for i, (k, c) in
              enumerate(kids(s))]
        new = rebuild(s, None, ks)
        key = tuple(map(_tagp, pos))
        if key in selected:
            r = selected[key]
            r2 = (r[0], r[1], r[2], new, r[4])
            vals = [_elem(r2, w, fname) for w in (which.args if is_list(which) else [which])]
            new = rebuild(new, apply(f, *vals), None)
        return new
    return build2(t, ())


@fn("TreeMap", area=AREA, status="added",
    notes="TreeMap[f, tree, levelspec -> elem] applies f to the data (or \"Subtree\", \"Position\", \"Level\", "
          "\"NegativeLevel\", \"LevelPair\", \"Children\", \"ChildrenData\", or a list of these as arguments) of "
          "the subtrees on the levels (default All); the result becomes the new data")
def _treemap(*args):
    pos, opts = _with_opts(args, "TreeMap", 1, 3)
    if len(pos) == 1:
        raise Unevaluated()
    f, t = pos[0], _need_tree(pos[1], "TreeMap")
    spec, which = _split_elem(pos[2]) if len(pos) == 3 else (None, None)
    return _map_tree(f, t, tree_levelspec(spec, "TreeMap"), which, traversal(opts, "TreeMap"), "TreeMap")


KL.operator_form("TreeMap", 1, "last")


@fn("TreeScan", area=AREA, status="added",
    notes="TreeScan[f, tree, levelspec -> elem] evaluates f on the data (or elements) of each subtree in "
          "traversal order; returns Null")
def _treescan(*args):
    pos, opts = _with_opts(args, "TreeScan", 1, 3)
    if len(pos) == 1:
        raise Unevaluated()
    f, t = pos[0], _need_tree(pos[1], "TreeScan")
    spec, which = _split_elem(pos[2]) if len(pos) == 3 else (None, "Data")
    ls = tree_levelspec(spec, "TreeScan")
    which = "Data" if which is None else which
    for r in nodes(t, traversal(opts, "TreeScan")):
        if in_spec(r, ls):
            if is_list(which):
                apply(f, *(_elem(r, w, "TreeScan") for w in which.args))
            else:
                apply(f, _elem(r, which, "TreeScan"))
    return None


KL.operator_form("TreeScan", 1, "last")


def _fold(fs, t, h):
    f_inner, f_leaf = fs

    def go(s):
        d = apply(h, s) if h is not None else data(s)
        if kind(s) == "none":
            return d if f_leaf is None else apply(f_leaf, d)
        ks = kids(s)
        if kind(s) == "list":
            res = L(*(go(c) for _, c in ks))
        else:
            res = Expr("Association", *(Expr("Rule", k, go(c)) for k, c in ks))
        return apply(f_inner, d, res)
    return go(t)


@fn("TreeFold", area=AREA, status="added",
    notes="TreeFold[f, tree] folds bottom-up: f[data, {results of children}] (leaves give their data); "
          "TreeFold[{f, g}, tree] applies g to leaf data; TreeFold[f, tree, h] uses h[subtree] instead of the data")
def _treefold(*args):
    if len(args) == 1:
        raise Unevaluated()
    if len(args) not in (2, 3):
        raise TreeError("TreeFold: 1 to 3 arguments expected")
    f, t = args[0], _need_tree(args[1], "TreeFold")
    h = args[2] if len(args) == 3 else None
    if is_list(f):
        if len(f.args) != 2:
            raise NotImplementedError("TreeFold: give f or {f, fleaf}")
        fs = (f.args[0], f.args[1])
    else:
        fs = (f, None)
    return _fold(fs, t, h)


def _treefold_op(head, args, evf):
    if len(head.args) == 1 and len(args) in (1, 2):
        return _treefold(head.args[0], *args)
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["TreeFold"] = _treefold_op


# ─────────────────────────── positions: extract, delete, insert, replace, map at ───────────────────────────

def _child_index(s, part, fname):
    ks = kids(s)
    if kind(s) == "none":
        raise TreeError(f"{fname}: the leaf {s} has no children (part {part})")
    if is_int(part) and not isinstance(part, bool):
        i = int(part)
        n = len(ks)
        if i == 0 or abs(i) > n:
            raise TreeError(f"{fname}: part {i} of a subtree with {n} children does not exist")
        return i - 1 if i > 0 else n + i
    if kind(s) == "assoc":
        key = part.args[0] if isinstance(part, Expr) and part.head == "Key" and len(part.args) == 1 else part
        if isinstance(part, str) or (isinstance(part, Expr) and part.head == "Key"):
            for j, (k, _) in enumerate(ks):
                if K.same(k, key):
                    return j
            raise TreeError(f"{fname}: the key {key} does not exist")
    raise NotImplementedError(f"{fname}: the part specification {part} is not supported "
                              "(use integers, \"key\" or Key[k])")


def _is_single_pos(p):
    return is_list(p) and not any(is_list(x) for x in p.args)


def _positions(p, fname):
    if _is_single_pos(p):
        return [list(p.args)]
    if is_list(p) and all(is_list(x) for x in p.args):
        for x in p.args:
            if not _is_single_pos(x):
                raise NotImplementedError(f"{fname}: the position {x} is not supported")
        return [list(x.args) for x in p.args]
    if is_int(p) or isinstance(p, str) or (isinstance(p, Expr) and p.head == "Key"):
        return [[p]]
    raise NotImplementedError(f"{fname}: the position specification {p} is not supported")


def _get(t, pos, fname):
    s = t
    for part in pos:
        s = kids(s)[_child_index(s, part, fname)][1]
    return s


def _update(t, pos, g, fname):
    """Replace the subtree at pos by g(subtree)."""
    if not pos:
        return g(t)
    j = _child_index(t, pos[0], fname)
    ks = kids(t)
    ks[j] = (ks[j][0], _update(ks[j][1], pos[1:], g, fname))
    return rebuild(t, None, ks)


@fn("TreeExtract", area=AREA, status="added",
    notes="TreeExtract[tree, pos | {pos1, ...}, h] subtrees at positions (integer parts, negative parts, "
          "\"key\"/Key[k]); h is applied to each")
def _treeextract(*args):
    if len(args) == 1:
        raise Unevaluated()
    if len(args) not in (2, 3):
        raise TreeError("TreeExtract: 1 to 3 arguments expected")
    t = _need_tree(args[0], "TreeExtract")
    h = args[2] if len(args) == 3 else None
    single = _is_single_pos(args[1]) or not is_list(args[1])
    out = []
    for p in _positions(args[1], "TreeExtract"):
        s = _get(t, p, "TreeExtract")
        out.append(apply(h, s) if h is not None else s)
    return out[0] if single else L(*out)


KL.operator_form("TreeExtract", 1, "first")


@fn("TreeDelete", area=AREA, status="added",
    notes="TreeDelete[tree, pos | {pos1, ...}] removes subtrees (deleting the only child leaves Tree[d, {}])")
def _treedelete(*args):
    if len(args) == 1:
        raise Unevaluated()
    if len(args) != 2:
        raise TreeError("TreeDelete: 2 arguments expected")
    t = _need_tree(args[0], "TreeDelete")
    ps = _positions(args[1], "TreeDelete")
    targets = {}
    for p in ps:
        if not p:
            raise TreeError("TreeDelete: the root of a tree cannot be deleted")
        parent = p[:-1]
        s = _get(t, parent, "TreeDelete")
        targets.setdefault(tuple(_tagp(x) for x in parent), (parent, set()))[1].add(
            _child_index(s, p[-1], "TreeDelete"))
    # delete deepest first so that positions stay valid
    for key in sorted(targets, key=len, reverse=True):
        parent, idxs = targets[key]

        def g(s, idxs=idxs):
            return rebuild(s, None, [kc for j, kc in enumerate(kids(s)) if j not in idxs])
        t = _update(t, parent, g, "TreeDelete")
    return t


KL.operator_form("TreeDelete", 1, "first")


@fn("TreeInsert", area=AREA, status="added",
    notes="TreeInsert[tree, child, pos | {pos1, ...}] inserts child (non-trees become leaves) so that it is at "
          "position pos (negative indices count from the end, like Insert); list children only")
def _treeinsert(*args):
    if len(args) == 2:
        raise Unevaluated()
    if len(args) != 3:
        raise TreeError("TreeInsert: 3 arguments expected")
    t = _need_tree(args[0], "TreeInsert")
    child = leaf(args[1])
    for p in _positions(args[2], "TreeInsert"):
        if not p:
            raise TreeError("TreeInsert: cannot insert at the root position {}")
        if not is_int(p[-1]):
            raise NotImplementedError("TreeInsert: insertion into association children is not supported")
        i = int(p[-1])

        def g(s, i=i):
            ks = kids(s)
            if kind(s) == "assoc":
                raise NotImplementedError("TreeInsert: insertion into association children is not supported")
            n = len(ks)
            if i == 0 or abs(i) > n + 1:
                raise TreeError(f"TreeInsert: cannot insert at position {i} among {n} children")
            j = i - 1 if i > 0 else n + 1 + i
            ks.insert(j, (None, child))
            return mk(data(s), "list", ks, s.args[2:])
        t = _update(t, p[:-1], g, "TreeInsert")
    return t


KL.operator_form("TreeInsert", 2, "first")


@fn("TreeReplacePart", area=AREA, status="added",
    notes="TreeReplacePart[tree, pos -> new | {pos1 -> new1, ...} | {pos1, pos2} -> new] replaces subtrees "
          "(non-trees become leaves); explicit positions only (no patterns)")
def _treereplacepart(*args):
    if len(args) == 1:
        raise Unevaluated()
    if len(args) != 2:
        raise TreeError("TreeReplacePart: 2 arguments expected")
    t = _need_tree(args[0], "TreeReplacePart")
    rules = args[1].args if is_list(args[1]) else [args[1]]
    for r in rules:
        if not is_rule(r):
            raise TreeError(f"TreeReplacePart: {r} is not a rule pos -> new")
        lhs = r.args[0]
        if K.has_pattern(lhs):
            raise NotImplementedError("TreeReplacePart: pattern positions are not supported")
        for p in _positions(lhs, "TreeReplacePart"):
            new = leaf(ev(r.args[1]) if r.head == "RuleDelayed" else r.args[1])
            t = _update(t, p, lambda s, new=new: new, "TreeReplacePart")
    return t


KL.operator_form("TreeReplacePart", 1, "first")


@fn("TreeMapAt", area=AREA, status="added",
    notes="TreeMapAt[f, tree, pos | {pos1, ...}] applies f to the data at the positions (repeatedly for "
          "repeated positions)")
def _treemapat(*args):
    if len(args) == 2:
        raise Unevaluated()
    if len(args) != 3:
        raise TreeError("TreeMapAt: 3 arguments expected")
    f, t = args[0], _need_tree(args[1], "TreeMapAt")
    for p in _positions(args[2], "TreeMapAt"):
        t = _update(t, p, lambda s: rebuild(s, apply(f, data(s)), None), "TreeMapAt")
    return t


KL.operator_form("TreeMapAt", 2, "middle")


# ─────────────────────────── conversions ───────────────────────────

def _rules_leafdata(d):
    return is_rule(d) and (is_list(d.args[1]) or is_assoc(d.args[1]) or is_sym(d.args[1], "None"))


@fn("TreeRules", area=AREA, status="added",
    notes="TreeRules[tree]: data -> {rules of children} (or an association); leaves give their data (data -> None "
          "when the data itself looks like such a rule)")
def _treerules(t):
    t = _need_tree(t, "TreeRules")

    def go(s):
        k = kind(s)
        if k == "none":
            return Expr("Rule", data(s), NONE) if _rules_leafdata(data(s)) else data(s)
        if k == "list":
            return Expr("Rule", data(s), L(*(go(c) for _, c in kids(s))))
        return Expr("Rule", data(s), Expr("Association", *(Expr("Rule", kk, go(c)) for kk, c in kids(s))))
    return go(t)


def rules_tree(r):
    if is_rule(r) and r.head == "Rule":
        d, c = r.args
        if is_list(c):
            return Expr("Tree", d, L(*(rules_tree(x) for x in c.args)))
        if is_assoc(c):
            return Expr("Tree", d, Expr("Association", *(Expr("Rule", x.args[0], rules_tree(x.args[1]))
                                                         for x in c.args)))
        if is_sym(c, "None"):
            return Expr("Tree", d, NONE)
    return Expr("Tree", r, NONE)


@fn("RulesTree", area=AREA, status="added",
    notes="RulesTree[data -> {rules}] / [data -> <|k -> rule|>] / [data -> None]; other expressions are leaves")
def _rulestree(r, *opts):
    out = rules_tree(r)
    return Expr("Tree", *out.args, *opts) if opts else out


@fn("RootTree", area=AREA, status="added",
    notes="RootTree[tree] = Tree[data, {}] (leaves unchanged); RootTree[tree, n] keeps levels 0..n (n >= 0)")
def _roottree(t, n=None):
    t = _need_tree(t, "RootTree")
    if n is None:
        n = 0
    elif is_sym(n, "Infinity"):
        return t
    else:
        n = int_value(n, "RootTree", "level", None)
        if n < 0:
            raise NotImplementedError("RootTree: negative levels are not supported")

    def go(s, lev):
        if kind(s) == "none":
            return s
        if lev >= n:
            return mk(data(s), kind(s), [], s.args[2:])
        return rebuild(s, None, [(k, go(c, lev + 1)) for k, c in kids(s)])
    return go(t, 0)


@fn("UnlabeledTree", area=AREA, status="added", notes="a tree of the same shape with Null as every node's data")
def _unlabeledtree(t, *opts):
    t = _need_tree(t, "UnlabeledTree")

    def go(s):
        return mk(NULL, kind(s), [(k, go(c)) for k, c in kids(s)], ())
    out = go(t)
    return Expr("Tree", *out.args, *opts) if opts else out


@fn("NestTree", area=AREA, status="added",
    notes="NestTree[f, tree | expr, n, h] grows every leaf Tree[x, None] with children f[x] (list, association, "
          "None or {} ; other results r give {r}) for n levels (default 1, Infinity allowed); h maps the data of "
          "the leaves including the new ones")
def _nesttree(f, t, n=1, h=None, *opts):
    if is_sym(n, "Infinity"):
        n = None
    else:
        n = int_value(n, "NestTree", "number of levels", 0)
    t = t if is_tree(t) else Expr("Tree", t, NONE)
    budget = [100000]

    def grow(x, left, depth):
        budget[0] -= 1
        if budget[0] < 0 or depth > 400:
            raise NotImplementedError("NestTree: the tree is too large (use a finite number of levels)")
        d = apply(h, x) if h is not None else x
        if left == 0:
            return Expr("Tree", d, NONE)
        r = apply(f, x)
        if is_sym(r, "None"):
            return Expr("Tree", d, NONE)
        if isinstance(r, Expr) and r.head == "Hold":
            raise NotImplementedError("NestTree: Hold[...] results are not supported")
        nl = None if left is None else left - 1
        if is_assoc(r):
            return Expr("Tree", d, Expr("Association", *(Expr("Rule", q.args[0], grow(q.args[1], nl, depth + 1))
                                                         for q in r.args)))
        items = r.args if is_list(r) else (r,)
        return Expr("Tree", d, L(*(grow(c, nl, depth + 1) for c in items)))

    def walk(s):
        if kind(s) == "none":
            g = grow(data(s), n, 0)
            return Expr("Tree", *g.args, *s.args[2:])
        return rebuild(s, None, [(k, walk(c)) for k, c in kids(s)])
    out = walk(t)
    return Expr("Tree", *out.args, *opts) if opts else out


def _expr_tree(x, struct, fname):
    atom = KL.atomq(x) or not isinstance(x, Expr)
    if is_assoc(x) and struct not in ("Association",):
        atom = False
    if struct == "Heads":
        if atom:
            return Expr("Tree", x, NONE)
        args = KL.assoc_values(x, ev) if is_assoc(x) else x.args
        return Expr("Tree", Symbol(x.head) if isinstance(x.head, str) else x.head,
                    L(*(_expr_tree(a, struct, fname) for a in args)))
    if struct == "Subexpressions":
        if atom:
            return Expr("Tree", x, NONE)
        args = KL.assoc_values(x, ev) if is_assoc(x) else x.args
        return Expr("Tree", x, L(*(_expr_tree(a, struct, fname) for a in args)))
    if struct in ("Atoms", "Null"):
        if atom:
            return Expr("Tree", x if struct == "Atoms" else NULL, NONE)
        args = KL.assoc_values(x, ev) if is_assoc(x) else x.args
        return Expr("Tree", NULL, L(*(_expr_tree(a, struct, fname) for a in args)))
    if struct == "List":
        if is_list(x):
            return Expr("Tree", NULL, L(*(_expr_tree(a, struct, fname) for a in x.args)))
        return Expr("Tree", x, NONE)
    if struct == "Association":
        if is_assoc(x):
            return Expr("Tree", NULL, Expr("Association", *(Expr("Rule", r.args[0], _expr_tree(r.args[1], struct,
                                                                                               fname))
                                                             for r in x.args)))
        return Expr("Tree", x, NONE)
    if struct == "Rules":
        return rules_tree(x)
    if struct == "None":
        return Expr("Tree", x, NONE)
    raise NotImplementedError(f"{fname}: the structure {struct!r} is not supported")


def _struct_name(s, fname):
    if isinstance(s, str):
        return s
    if is_sym(s, "Null"):
        return "Null"
    if is_sym(s, "None"):
        return "None"
    raise TreeError(f"{fname}: invalid structure {s}")


@fn("ExpressionTree", area=AREA, status="added",
    notes="ExpressionTree[expr, struct] with struct \"Heads\" (default), \"Subexpressions\", \"Atoms\", Null, "
          "None, \"List\", \"Association\", \"Rules\"; Heads -> True and the XML/JSON/held forms raise")
def _expressiontree(*args):
    pos, opts = split_options(args, {"Heads"} | _OPTSET, "ExpressionTree")
    if not 1 <= len(pos) <= 2:
        raise TreeError("ExpressionTree: 1 or 2 arguments expected")
    if opts.pop("Heads", False) is not False:
        raise NotImplementedError("ExpressionTree: Heads -> True is not supported")
    struct = _struct_name(pos[1], "ExpressionTree") if len(pos) == 2 else "Heads"
    out = _expr_tree(pos[0], struct, "ExpressionTree")
    if opts:
        out = Expr("Tree", *out.args, *(Expr("Rule", Symbol(k), v) for k, v in opts.items()))
    return out


def _tree_expr(t, struct, fname):
    k = kind(t)
    if struct in ("Heads", "HeadTrees"):
        if k == "none":
            return data(t)
        if k == "assoc":
            raise NotImplementedError(f"{fname}: association children are not supported for {struct!r}")
        h = data(t)
        if struct == "HeadTrees" and is_tree(h):
            h = _tree_expr(h, struct, fname)
        return ev(K.call(h, *(_tree_expr(c, struct, fname) for _, c in kids(t))))
    if struct == "List":
        if k == "none":
            return data(t)
        if k == "assoc":
            raise NotImplementedError(f"{fname}: association children are not supported for \"List\"")
        return L(*(_tree_expr(c, struct, fname) for _, c in kids(t)))
    if struct == "Association":
        if k == "none":
            return data(t)
        if k == "list":
            raise NotImplementedError(f"{fname}: list children are not supported for \"Association\"")
        return Expr("Association", *(Expr("Rule", kk, _tree_expr(c, struct, fname)) for kk, c in kids(t)))
    if struct == "Rules":
        return _treerules(t)
    raise NotImplementedError(f"{fname}: the structure {struct!r} is not supported")


@fn("TreeExpression", area=AREA, status="added",
    notes="TreeExpression[tree, struct] with struct \"Heads\" (default: data[children...]), \"HeadTrees\", "
          "\"List\", \"Association\", \"Rules\"; other structures raise")
def _treeexpression(*args):
    pos, opts = split_options(args, {"Heads"}, "TreeExpression")
    if not 1 <= len(pos) <= 2:
        raise TreeError("TreeExpression: 1 or 2 arguments expected")
    if opts.get("Heads", False) is not False:
        raise NotImplementedError("TreeExpression: Heads -> True is not supported")
    t = _need_tree(pos[0], "TreeExpression")
    struct = _struct_name(pos[1], "TreeExpression") if len(pos) == 2 else "Heads"
    return _tree_expr(t, struct, "TreeExpression")


@fn("TreeForm", area=AREA, status="added",
    notes="TreeForm[expr] / TreeForm[expr, n] is an inert display wrapper (no tree drawing in the terminal)")
def _treeform(*args):
    raise Unevaluated()


# ─────────────────────────── graphs ───────────────────────────

def _wg():
    from . import wl_graphs as G
    return G


@fn("TreeGraph", area=AREA, status="added",
    notes="TreeGraph[{edges or rules}] / TreeGraph[{vertices}, {edges}] gives the Graph after checking it is a "
          "tree (DirectedEdges -> False makes rules undirected); the predecessor-list form and Tree objects raise")
def _treegraph(*args):
    G = _wg()
    pos, opts = split_options(args, {"DirectedEdges"} | G._GRAPH_OPTS, "TreeGraph")
    de = opts.pop("DirectedEdges", None)

    def conv(e):
        if de is False and is_rule(e):
            return Expr("UndirectedEdge", *e.args)
        if de is True and isinstance(e, Expr) and e.head == "UndirectedEdge":
            return Expr("DirectedEdge", *e.args)
        return e
    if len(pos) == 1 and is_list(pos[0]):
        edges = [conv(e) for e in pos[0].args]
        verts = None
    elif len(pos) == 2 and is_list(pos[0]) and is_list(pos[1]):
        if not all(G._is_edge(e) for e in pos[1].args):
            raise NotImplementedError("TreeGraph: the predecessor-list form is not supported; give edges")
        verts = list(pos[0].args)
        edges = [conv(e) for e in pos[1].args]
    elif len(pos) == 1 and is_tree(pos[0]):
        raise NotImplementedError("TreeGraph: Tree objects are not supported; give edges")
    else:
        raise TreeError("TreeGraph: expected TreeGraph[{edges}] or TreeGraph[{vertices}, {edges}]")
    if verts is None:
        g = G._parse_graph_expr(Expr("Graph", L(*edges)), "TreeGraph")
    else:
        g = G.WG(verts, edges, {}, "TreeGraph")
    if opts:
        g = G.WG(g.vertices, g.edge_exprs, opts, "TreeGraph")
    out = g.expr()
    if apply(Symbol("TreeGraphQ"), out) is not True:
        raise TreeError("TreeGraph: the edges do not form a tree")
    return out


@fn("GraphTree", area=AREA, status="added",
    notes="GraphTree[g, root] converts a tree graph (TreeGraphQ) into a Tree rooted at root (default: the "
          "vertex with the largest out-component, i.e. the source of a directed tree, else the first vertex); "
          "siblings follow VertexList order; GraphTree[g, root, h] with h[v] = {data, sortkey}")
def _graphtree(*args):
    G = _wg()
    pos, opts = split_options(args, _OPTSET, "GraphTree")
    if not 1 <= len(pos) <= 3:
        raise TreeError("GraphTree: 1 to 3 arguments expected")
    g = G.graph_of(pos[0], "GraphTree")
    if apply(Symbol("TreeGraphQ"), pos[0] if isinstance(pos[0], Expr) and pos[0].head == "Graph"
             else g.expr()) is not True:
        raise TreeError("GraphTree: the graph is not a tree")
    n = len(g.vertices)
    adj = {i: [] for i in range(n)}
    indeg = [0] * n
    for (i, j, d) in g.edges:
        adj[i].append(j)
        adj[j].append(i)
        if d:
            indeg[j] += 1
    if len(pos) >= 2 and not is_sym(pos[1], "Automatic"):
        tg = K.tag(pos[1])
        if tg not in g.index:
            raise TreeError(f"GraphTree: {pos[1]} is not a vertex of the graph")
        root = g.index[tg]
    else:
        directed = any(d for (_, _, d) in g.edges)
        root = next((i for i in range(n) if indeg[i] == 0), 0) if directed else 0
    h = pos[2] if len(pos) == 3 else None

    def go(v, parent):
        ch = sorted(set(u for u in adj[v] if u != parent))
        if h is not None:
            pairs = []
            for u in ch:
                r = apply(h, g.vertices[u])
                if not (is_list(r) and len(r.args) == 2):
                    raise TreeError("GraphTree: h[v] must give {data, sortkey}")
                pairs.append((r, u))
            pairs.sort(key=lambda pr: K.order_key(pr[0].args[1]))
            ch = [u for _, u in pairs]
        d = g.vertices[v]
        if h is not None:
            r = apply(h, d)
            d = r.args[0] if is_list(r) and len(r.args) == 2 else d
        if not ch:
            return Expr("Tree", d, NONE)
        return Expr("Tree", d, L(*(go(u, v) for u in ch)))
    out = go(root, None)
    if opts:
        out = Expr("Tree", *out.args, *(Expr("Rule", Symbol(k), v) for k, v in opts.items()))
    return out


@fn("NestGraph", area=AREA, status="added",
    notes="NestGraph[f, expr | {exprs} | graph, n] breadth-first graph with directed edges x -> y for y in f[x], "
          "n levels (default 1; Infinity allowed when finite); DirectedEdges -> False gives undirected edges")
def _nestgraph(*args):
    G = _wg()
    pos, opts = split_options(args, {"DirectedEdges"} | G._GRAPH_OPTS, "NestGraph")
    if not 2 <= len(pos) <= 3:
        raise TreeError("NestGraph: 2 or 3 arguments expected")
    f, start = pos[0], pos[1]
    n = 1 if len(pos) == 2 else (None if is_sym(pos[2], "Infinity") else int_value(pos[2], "NestGraph", "n", 0))
    de = opts.pop("DirectedEdges", True)
    eh = "UndirectedEdge" if de is False else "DirectedEdge"
    verts, vindex, edges, eseen = [], set(), [], set()

    def addv(v):
        tg = K.tag(v)
        if tg not in vindex:
            vindex.add(tg)
            verts.append(v)
            return True
        return False

    def adde(e):
        key = (e.head, K.tag(e.args[0]), K.tag(e.args[1]))
        if e.head == "UndirectedEdge":
            key = (e.head,) + tuple(sorted([K.tag(e.args[0]), K.tag(e.args[1])], key=repr))
        if key not in eseen:
            eseen.add(key)
            edges.append(e)
    if isinstance(start, Expr) and start.head == "Graph":
        g0 = G.graph_of(start, "NestGraph")
        for v in g0.vertices:
            addv(v)
        for e in g0.edge_exprs:
            adde(e)
        frontier = list(g0.vertices)
    elif is_list(start):
        frontier = []
        for v in start.args:
            if addv(v):
                frontier.append(v)
    else:
        addv(start)
        frontier = [start]
    expanded = set()
    level = 0
    while frontier and (n is None or level < n):
        nxt = []
        for x in frontier:
            if K.tag(x) in expanded:
                continue
            expanded.add(K.tag(x))
            r = apply(f, x)
            items = r.args if is_list(r) else (r,)
            for y in items:
                adde(Expr(eh, x, y))
                if addv(y):
                    nxt.append(y)
            if len(verts) > 100000:
                raise NotImplementedError("NestGraph: the graph is too large")
        frontier = nxt
        level += 1
    return G.WG(verts, edges, opts, "NestGraph").expr()


@fn("RandomTree", area=AREA, status="added",
    notes="RandomTree[n | {e1, ..., en}, k | {k1, ...}] uniformly random ordered rooted trees with n nodes "
          "(via a random Dyck word and the cycle lemma); the data 1..n (or the ei) are assigned in pre-order")
def _randomtree(spec, shape=None, *opts):
    if is_list(spec):
        labels = list(spec.args)
    else:
        n = int_value(spec, "RandomTree", "number of nodes", 1)
        labels = list(range(1, n + 1))
    n = len(labels)
    if n < 1:
        raise TreeError("RandomTree: at least one node is required")
    rng = random

    def one():
        # a uniformly random ordered tree with n nodes: rotate a random sequence of n-1 "down" (+1) and
        # n "up" (-1) steps to its unique cyclic shift that is a Lukasiewicz path (cycle lemma)
        steps = [1] * (n - 1) + [-1] * n
        rng.shuffle(steps)
        s, best, arg = 0, None, 0
        for i, v in enumerate(steps):
            s += v
            if best is None or s < best:
                best, arg = s, i
        steps = steps[arg + 1:] + steps[:arg + 1]
        # convert the Dyck-like word (n-1 opens, n closes, last close ends the root) into a tree
        it = iter(steps)
        counter = iter(labels)

        def build():
            d = next(counter)
            ch = []
            while next(it) == 1:
                ch.append(build())
            return Expr("Tree", d, L(*ch) if ch else NONE)
        return build()

    if shape is None:
        out = one()
    else:
        dims = [int_value(x, "RandomTree", "count", 0) for x in (shape.args if is_list(shape) else [shape])]

        def arr(ds):
            if not ds:
                return one()
            return L(*(arr(ds[1:]) for _ in range(ds[0])))
        out = arr(dims)
    return out


# ─────────────────────────── tree games (extensive-form games) ───────────────────────────

class _GNode:
    __slots__ = ("player", "infoset", "children", "payoffs", "label", "prob")

    def __init__(self):
        self.player = self.infoset = self.payoffs = self.label = self.prob = None
        self.children = []


def _is_real_num(x):
    return K.is_number(x) and not isinstance(x, bool) and (lambda v: v.is_real is not False)(K.to_sp(x))


def _edge_annotation(node, ann, fname):
    if is_list(ann) and len(ann.args) == 2 and _is_real_num(ann.args[1]):
        node.label, node.prob = ann.args
    elif _is_real_num(ann):
        node.prob = ann
    else:
        node.label = ann


def _game_node(x, fname):
    node = _GNode()
    if is_rule(x):
        child = _game_node(x.args[0], fname)
        if child.label is not None or child.prob is not None:
            raise TreeError(f"{fname}: an action has more than one annotation")
        _edge_annotation(child, x.args[1], fname)
        return child
    if isinstance(x, Expr) and x.head == "Tree" and len(x.args) >= 2:
        d, ch = x.args[0], x.args[1]
        if not is_assoc(d):
            raise TreeError(f"{fname}: tree game nodes must have association data <|\"Player\" -> ...|> or "
                            "<|\"Payoffs\" -> {...}|>")
        info = {r.args[0]: r.args[1] for r in d.args}
        bad = set(info) - {"Player", "Payoffs", "Label", "Probability", "InformationSet"}
        if bad:
            raise TreeError(f"{fname}: unknown node keys {sorted(map(str, bad))}")
        node.label, node.prob = info.get("Label"), info.get("Probability")
        kids = [] if is_sym(ch, "None") else list(ch.args) if is_list(ch) else None
        if kids is None:
            raise TreeError(f"{fname}: the children of a tree game node must be a list")
        if "Payoffs" in info:
            if "Player" in info or kids:
                raise TreeError(f"{fname}: an end node has payoffs and no player or children")
            node.payoffs = _payoff_list(info["Payoffs"], fname)
            return node
        if "Player" not in info or not kids:
            raise TreeError(f"{fname}: an action node needs a \"Player\" and at least one child")
        node.player, node.infoset = info["Player"], info.get("InformationSet")
        node.children = [_game_node(k, fname) for k in kids]
        _check_probs(node, fname)
        return node
    if is_list(x) and x.args and all(_is_real_num(v) for v in x.args):
        node.payoffs = list(x.args)
        return node
    if is_list(x) and len(x.args) >= 2:
        p = x.args[0]
        if is_rule(p):
            node.player, node.infoset = p.args
        else:
            node.player = p
        if is_list(node.player) or _is_real_num(node.player):
            raise TreeError(f"{fname}: {node.player} is not a valid player")
        node.children = [_game_node(k, fname) for k in x.args[1:]]
        _check_probs(node, fname)
        return node
    raise TreeError(f"{fname}: {x} is neither an action node {{player, action1, ...}} nor an end node "
                    "{payoff1, ...}")


def _payoff_list(p, fname):
    if is_list(p) and p.args and all(_is_real_num(v) for v in p.args):
        return list(p.args)
    raise TreeError(f"{fname}: payoffs must be a non-empty list of real numbers")


def _check_probs(node, fname):
    ps = [c.prob for c in node.children]
    if all(p is None for p in ps):
        return
    if any(p is None for p in ps):
        raise TreeError(f"{fname}: when one action of a node has a probability, all of them need one")
    _check_distribution(ps, fname, "action probabilities")


def _check_distribution(ps, fname, what):
    vals = [K.to_sp(p) for p in ps]
    if any(v.is_real is False or v < 0 for v in vals):
        raise TreeError(f"{fname}: {what} must be nonnegative reals")
    tot = sum(vals)
    if (tot.is_Float and abs(tot - 1) > 1e-8) or (not tot.is_Float and sp.simplify(tot - 1) != 0):
        raise TreeError(f"{fname}: {what} must sum to 1")


def _game_root(g, fname):
    if not (isinstance(g, Expr) and g.head == "TreeGame" and len(g.args) == 1):
        if isinstance(g, Symbol):
            raise Unevaluated()
        raise TreeError(f"{fname}: {g} is not a TreeGame")
    root = _game_node(g.args[0], fname)
    if root.payoffs is not None:
        raise TreeError(f"{fname}: the root of a tree game must be an action node")
    counts = set()

    def walk(n):
        if n.payoffs is not None:
            counts.add(len(n.payoffs))
        for c in n.children:
            walk(c)
    walk(root)
    if len(counts) != 1:
        raise TreeError(f"{fname}: all end nodes must give the same number of payoffs")
    return root


import sympy as sp  # noqa: E402


@fn("TreeGame", area=AREA, status="added",
    notes="TreeGame[{player, action1, ...}] (end nodes {payoff1, ...}; actions annotated action -> label | prob | "
          "{label, prob}; information sets player -> ilabel) or TreeGame[Tree[<|\"Player\" -> p|>, {...}]] with "
          "\"Payoffs\", \"Label\", \"Probability\", \"InformationSet\" keys: validated (consistent payoff counts, "
          "complete probability distributions) and kept as the inert game object TreeGame[spec]")
def _treegame(spec, *rest):
    if rest:
        raise TreeError("TreeGame: options and extra arguments are not supported")
    _game_root(Expr("TreeGame", spec), "TreeGame")
    raise Unevaluated()


def _strategy_table(strat, fname):
    """node position tuple -> list of probabilities."""
    if not is_assoc(strat):
        raise TreeError(f"{fname}: the strategy must be an association")
    table = {}

    def add(k, v):
        if not (is_list(k) and all(is_int(i) and int(i) >= 1 for i in k.args)):
            raise TreeError(f"{fname}: strategy node {k} must be a list of positive integers")
        if not is_list(v) or not v.args:
            raise TreeError(f"{fname}: the strategy at node {k} must be a list of probabilities")
        _check_distribution(list(v.args), fname, f"strategy probabilities at node {k}")
        table[tuple(int(i) for i in k.args)] = list(v.args)
    for r in strat.args:
        k, v = r.args
        if is_assoc(v):
            for r2 in v.args:
                add(*r2.args)
        else:
            add(k, v)
    return table


@fn("TreeGamePayoff", area=AREA, status="added",
    notes="TreeGamePayoff[tgame, strat] expected payoff list for a strategy profile <|player -> <|node -> {pr, ...}|>, "
          "...|> or <|node -> {pr, ...}, ...|> (nodes are child-index paths, {} the root); actions with "
          "probabilities in the game use them unless the strategy overrides them; TreeGamePayoff[tgame, strat, "
          "\"RandomExpectation\"] averages unspecified nodes uniformly; \"Perfect\", \"PerfectExpectation\" and "
          "\"Random\" raise; an incomplete strategy without an adversary type raises")
def _treegamepayoff(game, strat, adv=None):
    fname = "TreeGamePayoff"
    root = _game_root(game, fname)
    table = _strategy_table(strat, fname)
    if adv is not None and adv != "RandomExpectation":
        if adv in ("Perfect", "PerfectExpectation", "Random"):
            raise NotImplementedError(f"{fname}: the adversary type {adv!r} is not supported "
                                      "(use \"RandomExpectation\")")
        raise TreeError(f"{fname}: unknown adversary type {adv!r}")
    used = set()

    def value(n, path):
        if n.payoffs is not None:
            return L(*n.payoffs)
        k = len(n.children)
        if path in table:
            ps = table[path]
            used.add(path)
            if len(ps) != k:
                raise TreeError(f"{fname}: the strategy at node {list(path)} has {len(ps)} probabilities for "
                                f"{k} actions")
        elif n.children[0].prob is not None:
            ps = [c.prob for c in n.children]
        elif adv == "RandomExpectation":
            ps = [K.from_sp(sp.Rational(1, k))] * k
        else:
            raise TreeError(f"{fname}: the strategy does not specify node {list(path)} (player {n.player}); "
                            "give an adversary type such as \"RandomExpectation\"")
        terms = [Expr("Times", p, value(c, path + (i + 1,))) for i, (p, c) in enumerate(zip(ps, n.children))
                 if not (K.is_number(p) and K.to_sp(p) == 0)]
        return ev(Expr("Plus", *terms)) if terms else None
    out = value(root, ())
    extra = set(table) - used
    if extra:
        def exists(n, p):
            for i in p:
                if n.payoffs is not None or i > len(n.children):
                    return False
                n = n.children[i - 1]
            return n.payoffs is None
        missing = [p for p in extra if not exists(root, p)]
        if missing:
            raise TreeError(f"{fname}: strategy node {list(missing[0])} is not an action node of the game")
    return out

