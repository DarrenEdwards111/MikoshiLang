"""Wolfram batch W24: graph extras (measures, constructions, colouring, isomorphism, random graphs, annotations).

Extends :mod:`wl_graphs` (batch 5) and reuses its parsed-graph class ``WG``: a graph is the inert expression
``Graph[{v...}, {e...}, opts...]``.  Every function here returns a correct result, raises a clear error, or stays
unevaluated for symbolic input (see CONVENTIONS.md):

* ``TypeError``: an argument of the wrong kind (a number where a graph is needed, a string count, ...);
* ``ValueError``: a valid kind with an invalid value (a vertex that is not in the graph, a negative size, ...);
* ``NotImplementedError``: a valid Wolfram input outside the supported forms (listed in each head's notes);
* ``ArithmeticError``: a built-in self-check failed.

Functions that find *one* object among several (a colouring, a matching, a cut, a set of paths, ...) return a valid
optimal object, which may differ from the particular one Wolfram returns (as in batch 5).  Named-graph constructors
whose Wolfram vertex numbering is not documented (BuckyballGraph, ButterflyGraph, KnightTourGraph, GraphProduct,
...) give the right graph up to isomorphism with the numbering stated in their notes.

Rendering heads (GraphPlot, LayeredGraphPlot, CommunityGraphPlot, Graph3D, ...) belong to the graphics batch (W14)
and are not registered here; the presentation options are inert and kept on the Graph expression.
"""
from __future__ import annotations

import itertools
import math
from collections import deque
from fractions import Fraction

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import Unevaluated, fn, record, R, L, S, rule, is_list, is_numeric_value
from .wl_common import split_options, apply
from . import wl_graphs as WGM
from .wl_graphs import (WG, graph_of, is_graph_like, _new, UE, TAG_AWARE, GraphError, INF, _GRAPH_OPTS,
                        _weak_components, _strong_components, _components, _degree_lists, _maxflow, _max_cliques,
                        _topo, _nx, _to_nx, _between, _bipartition, _subgraph, _adjacency_graph, _norm_edge,
                        _is_edge, _stoer_wagner)

AREA = "graphs II (W24)"


# ═══════════════════════════════════════ argument helpers ═══════════════════════════════════════

def _tag_aware(*names):
    TAG_AWARE.update(names)


def _g(x, fname):
    """Parsed graph for a Graph or edge-rule list; wrong kinds raise TypeError, symbolic input stays unevaluated."""
    if isinstance(x, (bool, str, int, float, complex)) or isinstance(x, sp.Basic):
        raise TypeError(f"{fname}: expected a graph, not {x!r}")
    if is_list(x) and not (x.args and all(_is_edge(e) for e in x.args)):
        raise TypeError(f"{fname}: expected a graph or a list of edges, not {x!r}")
    if isinstance(x, Expr) and not is_list(x) and x.head != "Graph" and KL.atomq(x):
        raise TypeError(f"{fname}: expected a graph, not {x!r}")
    return graph_of(x, fname)


def _int(x, fname, lo=None, what="argument"):
    """Integer argument: TypeError for strings/booleans, Unevaluated for symbols, ValueError for bad values."""
    if isinstance(x, (bool, str)):
        raise TypeError(f"{fname}: {what} must be an integer, not {x!r}")
    if K.is_int(x):
        v = int(x)
        if lo is not None and v < lo:
            raise ValueError(f"{fname}: {what} must be an integer >= {lo} (got {v})")
        return v
    if is_numeric_value(x):
        raise ValueError(f"{fname}: {what} must be an integer (got {x})")
    if isinstance(x, (Symbol, Expr)) and not is_list(x):
        raise Unevaluated()
    raise TypeError(f"{fname}: {what} must be an integer, not {x!r}")


def _real(x, fname, what="argument", lo=None, hi=None):
    """Real number argument (exact or inexact) as a SymPy number."""
    if isinstance(x, (bool, str)):
        raise TypeError(f"{fname}: {what} must be a real number, not {x!r}")
    if not is_numeric_value(x):
        if isinstance(x, (Symbol, Expr)) and not is_list(x):
            raise Unevaluated()
        raise TypeError(f"{fname}: {what} must be a real number, not {x!r}")
    v = S(x)
    if not v.is_real:
        raise ValueError(f"{fname}: {what} must be real (got {x})")
    if lo is not None and v < lo or hi is not None and v > hi:
        raise ValueError(f"{fname}: {what} must lie in [{lo}, {hi}] (got {x})")
    return v


def _vlist_arg(x):
    return list(x.args) if is_list(x) else [x]


def _verts(gr, idxs):
    return L(*(gr.vertices[i] for i in idxs))


def _simple_nb(gr):
    """Neighbour sets of the underlying simple undirected graph (no self-loops, multi-edges merged)."""
    return gr.neighbor_sets(undirected=True)


def _need_undirected(gr, fname):
    gr.require_pure(fname)
    if gr.directed:
        raise NotImplementedError(f"{fname}: directed graphs are not supported")


def _need_simple_undirected(gr, fname):
    _need_undirected(gr, fname)
    if not gr.simple():
        raise NotImplementedError(f"{fname}: only simple graphs (no self-loops or multiple edges) are supported")


def _edge_expr(gr, k):
    return gr.edge_exprs[k]


def _same_edge(gr, k, e, fname):
    """True when edge k of gr matches the edge expression e (undirected edges match either orientation)."""
    ne = _norm_edge(e, fname)
    if K.tag(ne.args[0]) not in gr.index or K.tag(ne.args[1]) not in gr.index:
        return False
    a, b = gr.index[K.tag(ne.args[0])], gr.index[K.tag(ne.args[1])]
    i, j, d = gr.edges[k]
    if d != (ne.head == "DirectedEdge"):
        return False
    return (i, j) == (a, b) or (not d and (j, i) == (a, b))


def _edge_indices(gr, elist, fname):
    """Edge indices for a list of edges of gr (each edge used once); None when some edge is not in gr."""
    used, out = set(), []
    for e in elist:
        if not _is_edge(e):
            raise TypeError(f"{fname}: {e!r} is not an edge")
        hit = next((k for k in range(len(gr.edges)) if k not in used and _same_edge(gr, k, e, fname)), None)
        if hit is None:
            return None
        used.add(hit)
        out.append(hit)
    return out


def _graph_with(gr, vertices=None, edges=None, opts=None):
    return WG(gr.vertices if vertices is None else vertices, gr.edge_exprs if edges is None else edges,
              gr.opts if opts is None else opts).expr()


# ═══════════════════════════════════════ option names ═══════════════════════════════════════

def _option_symbol(name, notes):
    from .wl_evaluation import INERT
    INERT[name] = notes
    record(name, AREA, "added", notes.rstrip(". ") + ". Inert symbol: Name -> value is an ordinary rule kept on "
                                                       "the Graph. Check: exact (tests check it is kept on the graph "
                                                       "and read back with AnnotationValue).")


for _n, _d in {
    "GraphLayout": "Graph option naming the layout used to draw the graph (\"SpringElectricalEmbedding\", "
                   "\"CircularEmbedding\", ...); drawing is the graphics batch, so computations ignore it",
    "VertexLabels": "Graph option and vertex annotation: labels for vertices (\"Name\", or {v -> label, ...})",
    "EdgeLabels": "Graph option and edge annotation: labels for edges ({e -> label, ...})",
    "VertexStyle": "Graph option and vertex annotation: drawing style of vertices",
    "EdgeStyle": "Graph option and edge annotation: drawing style of edges",
    "VertexLabelStyle": "Graph option: style for vertex labels",
    "EdgeLabelStyle": "Graph option: style for edge labels",
    "VertexShape": "Graph option and vertex annotation: graphic used for vertices",
    "VertexShapeFunction": "Graph option and vertex annotation: function or named shape used to draw vertices",
    "VertexSize": "Graph option and vertex annotation: size of vertices",
    "EdgeShapeFunction": "Graph option and edge annotation: function or named shape used to draw edges",
    "GraphHighlight": "Graph option listing highlighted vertices and edges (set by HighlightGraph)",
    "GraphHighlightStyle": "Graph option: style used for GraphHighlight",
    "VertexCoordinates": "Graph option and vertex annotation: coordinates of the vertices ({{x1, y1}, ...} in "
                         "VertexList order, or {v -> {x, y}, ...}); GraphEmbedding reads them back",
    "MultiedgeStyle": "Graph option: how multiple edges are drawn",
    "SelfLoopStyle": "Graph option: how self-loops are drawn",
    "AnnotationRules": "Graph option {item -> {key -> value, ...}, ...} holding custom vertex and edge annotations "
                       "(item \"GraphProperties\" for graph-level ones); read with AnnotationValue",
    "EdgeCost": "Graph option and edge annotation: cost per unit flow of each edge (a list in EdgeList order), used "
                "by FindMinimumCostFlow",
    "VertexCapacity": "Graph option and vertex annotation: capacity of each vertex (a list in VertexList order), used "
                      "by FindMaximumFlow, FindVertexIndependentPaths and VertexConnectivity",
}.items():
    _option_symbol(_n, _d)

from .wl_evaluation import INERT as _INERT
_INERT["DirectedEdges"] = "option"
record("DirectedEdges", AREA, "added",
       "Graph option: DirectedEdges -> True makes every edge directed, False every edge undirected, Automatic keeps "
       "them; accepted by Graph and the named graphs whose directed orientation is Wolfram's (PathGraph, CycleGraph, "
       "StarGraph, KaryTree, CompleteKaryTree; others raise NotImplementedError) and by RelationGraph. Check: exact "
       "(tests compare the resulting edge lists).")


# ═══════════════════════════════════════ annotations ═══════════════════════════════════════

_VERTEX_KEYS = {"VertexLabels", "VertexStyle", "VertexShape", "VertexShapeFunction", "VertexSize",
                "VertexCoordinates", "VertexWeight", "VertexCapacity", "VertexLabelStyle"}
_EDGE_KEYS = {"EdgeLabels", "EdgeStyle", "EdgeShapeFunction", "EdgeWeight", "EdgeCapacity", "EdgeCost",
              "EdgeLabelStyle"}
_LIST_KEYS = {"VertexWeight", "VertexCapacity", "VertexCoordinates", "EdgeWeight", "EdgeCapacity", "EdgeCost"}
_GP = "GraphProperties"


def _key_name(k, fname):
    if isinstance(k, Symbol):
        return k.name
    if isinstance(k, str):
        return k
    raise TypeError(f"{fname}: annotation keys are symbols or strings, not {k!r}")


def _is_rule(x):
    return isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed") and len(x.args) == 2


def _item(gr, spec, fname):
    """('vertex', index) or ('edge', index) for an item specification of gr."""
    if _is_edge(spec):
        for k in range(len(gr.edges)):
            if _same_edge(gr, k, spec, fname):
                return ("edge", k)
        if K.tag(spec) not in gr.index:
            raise ValueError(f"{fname}: {spec} is not an edge of the graph")
    t = K.tag(spec)
    if t in gr.index:
        return ("vertex", gr.index[t])
    raise ValueError(f"{fname}: {spec} is not a vertex or edge of the graph")


def _item_matches(gr, item, lhs, fname):
    kind, k = item
    if kind == "edge":
        return _is_edge(lhs) and len(lhs.args) == 2 and _same_edge(gr, k, lhs, fname)
    return not _is_edge(lhs) and K.tag(lhs) == K.tag(gr.vertices[k])


def _custom_rules(gr):
    ar = gr.opts.get("AnnotationRules")
    if ar is None:
        return []
    if not is_list(ar) or not all(_is_rule(r) for r in ar.args):
        raise ValueError("AnnotationRules: expected {item -> {key -> value, ...}, ...}")
    return list(ar.args)


def _custom_for(gr, item, fname):
    """[(key name, key expr, value)] custom annotations of an item (item None: graph properties)."""
    out = []
    for r in _custom_rules(gr):
        lhs, rhs = r.args
        hit = (lhs == _GP) if item is None else (lhs != _GP and _item_matches(gr, item, lhs, fname))
        if hit:
            for kv in (rhs.args if is_list(rhs) else [rhs]):
                if not _is_rule(kv):
                    raise ValueError(f"{fname}: annotation rules must be key -> value")
                out.append((_key_name(kv.args[0], fname), kv.args[0], kv.args[1]))
    return out


_MISSING = object()


def _item_value(gr, item, key, fname):
    kind, k = item
    if key in _VERTEX_KEYS | _EDGE_KEYS:
        if (kind == "vertex") != (key in _VERTEX_KEYS):
            return _MISSING
        v = gr.opts.get(key)
        if v is None:
            if key in ("EdgeWeight", "EdgeCapacity", "VertexWeight") and gr.opts.get(key) is None:
                return _MISSING
            return _MISSING
        if key in _LIST_KEYS and is_list(v) and not all(_is_rule(x) for x in v.args):
            n = gr.n if kind == "vertex" else len(gr.edges)
            if len(v.args) != n:
                raise ValueError(f"{fname}: {key} must give one value per {kind}")
            return v.args[k]
        if is_list(v):
            default = _MISSING
            for x in v.args:
                if _is_rule(x):
                    if _item_matches(gr, item, x.args[0], fname):
                        return x.args[1]
                else:
                    default = x
            return default
        return v
    for name, _kx, val in _custom_for(gr, item, fname):
        if name == key:
            return val
    return _MISSING


def _graph_value(gr, key, fname):
    if key in gr.opts:
        return gr.opts[key]
    for name, _kx, val in _custom_for(gr, None, fname):
        if name == key:
            return val
    return _MISSING


def _ann_spec(spec, fname):
    """(gr, item or None) for obj or {obj, itemspec}."""
    if is_list(spec) and len(spec.args) == 2 and is_graph_like(spec.args[0]) and not all(
            _is_edge(e) for e in spec.args):
        gr = _g(spec.args[0], fname)
        return gr, _item(gr, spec.args[1], fname)
    if is_graph_like(spec):
        return _g(spec, fname), None
    if isinstance(spec, Symbol) or (isinstance(spec, Expr) and not is_list(spec)):
        raise Unevaluated()
    raise NotImplementedError(f"{fname}: only Graph objects and {{graph, item}} specifications are supported "
                              f"(got {spec!r})")


@fn("AnnotationValue", area=AREA, status="added",
    notes="AnnotationValue[g, key] graph-level annotation (a Graph option such as VertexLabels, or a custom key in "
          "AnnotationRules -> {\"GraphProperties\" -> {...}}); AnnotationValue[{g, item}, key] for a vertex or edge: "
          "list-valued options (VertexWeight, EdgeWeight, EdgeCapacity, EdgeCost, VertexCapacity, VertexCoordinates) "
          "by position, rule lists {item -> value, default} by item, a single value for every item, custom keys from "
          "AnnotationRules; a list of keys gives a list. Deviations: a missing annotation raises ValueError (Wolfram "
          "returns $Failed), and the assignment form AnnotationValue[...] = v is not supported (use Annotate). "
          "Objects other than graphs raise NotImplementedError. Check: exact (lookup; tests round-trip with Annotate).")
def _annvalue(spec: Expr, key: Expr) -> Expr:
    gr, item = _ann_spec(spec, "AnnotationValue")

    def one(kx):
        name = _key_name(kx, "AnnotationValue")
        v = _graph_value(gr, name, "AnnotationValue") if item is None else _item_value(gr, item, name,
                                                                                         "AnnotationValue")
        if v is _MISSING:
            raise ValueError(f"AnnotationValue: no {name} annotation is defined for {spec if item is None else spec.args[1]}")
        return v
    if is_list(key):
        return L(*(one(k) for k in key.args))
    return one(key)


@fn("AnnotationKeys", area=AREA, status="added",
    notes="AnnotationKeys[g] lists the graph's options and custom graph properties; AnnotationKeys[{g, item}] the "
          "annotation keys with a value for that vertex or edge (standard keys as symbols, custom keys as given), in "
          "option order. Objects other than graphs raise NotImplementedError. Check: exact (tests use MemberQ with "
          "Annotate/AnnotationDelete round trips).")
def _annkeys(spec: Expr) -> ListOf[Expr]:
    gr, item = _ann_spec(spec, "AnnotationKeys")
    keys, seen = [], set()

    def add(name, kx):
        if name not in seen:
            seen.add(name)
            keys.append(kx)
    if item is None:
        for k in gr.opts:
            if k != "AnnotationRules":
                add(k, Symbol(k))
        for name, kx, _v in _custom_for(gr, None, "AnnotationKeys"):
            add(name, kx)
    else:
        for k in gr.opts:
            if k in _VERTEX_KEYS | _EDGE_KEYS and _item_value(gr, item, k, "AnnotationKeys") is not _MISSING:
                add(k, Symbol(k))
        for name, kx, _v in _custom_for(gr, item, "AnnotationKeys"):
            add(name, kx)
    return L(*keys)


def _set_item(gr, item, name, kx, val, fname):
    opts = dict(gr.opts)
    kind, k = item
    item_expr = gr.vertices[k] if kind == "vertex" else gr.edge_exprs[k]
    if name in _VERTEX_KEYS | _EDGE_KEYS:
        if (kind == "vertex") != (name in _VERTEX_KEYS):
            raise ValueError(f"{fname}: {name} is a{'' if name in _EDGE_KEYS else ' vertex'} "
                             f"{'edge' if name in _EDGE_KEYS else ''} annotation and cannot be set on a {kind}")
        cur = opts.get(name)
        if name in _LIST_KEYS and not (cur is not None and is_list(cur) and all(_is_rule(x) for x in cur.args)):
            if cur is None or not is_list(cur):
                raise NotImplementedError(f"{fname}: setting {name} on one {kind} of a graph without {name} values "
                                          f"for every {kind} is not supported; give {name} -> {{...}} in Graph")
            vals = list(cur.args)
            vals[k] = val
            opts[name] = L(*vals)
        elif cur is None:
            opts[name] = L(rule(item_expr, val))
        elif is_list(cur):
            rest = [x for x in cur.args if not (_is_rule(x) and _item_matches(gr, item, x.args[0], fname))]
            opts[name] = L(rule(item_expr, val), *rest)
        else:
            opts[name] = L(rule(item_expr, val), cur)
        return opts
    rules = _custom_rules(gr)
    out, done = [], False
    for r in rules:
        lhs, rhs = r.args
        if lhs != _GP and _item_matches(gr, item, lhs, fname) and not done:
            kvs = [kv for kv in (rhs.args if is_list(rhs) else [rhs]) if _key_name(kv.args[0], fname) != name]
            out.append(rule(lhs, L(*kvs, rule(kx, val))))
            done = True
        else:
            out.append(r)
    if not done:
        out.append(rule(item_expr, L(rule(kx, val))))
    opts["AnnotationRules"] = L(*out)
    return opts


def _set_graph(gr, name, kx, val, fname):
    opts = dict(gr.opts)
    if name in _GRAPH_OPTS:
        opts[name] = val
        WG(gr.vertices, gr.edge_exprs, opts, fname)
        return opts
    rules = _custom_rules(gr)
    out, done = [], False
    for r in rules:
        if r.args[0] == _GP and not done:
            rhs = r.args[1]
            kvs = [kv for kv in (rhs.args if is_list(rhs) else [rhs]) if _key_name(kv.args[0], fname) != name]
            out.append(rule(_GP, L(*kvs, rule(kx, val))))
            done = True
        else:
            out.append(r)
    if not done:
        out.append(rule(_GP, L(rule(kx, val))))
    opts["AnnotationRules"] = L(*out)
    return opts


@fn("Annotate", area=AREA, status="added",
    notes="Annotate[g, key -> value] sets a graph-level annotation (a Graph option, or a custom key stored under "
          "AnnotationRules \"GraphProperties\"); Annotate[{g, item}, key -> value] or {key -> value, ...} annotates a "
          "vertex or edge (rule-list options gain item -> value; list-valued options such as VertexWeight replace "
          "the item's entry; custom keys go to AnnotationRules). The original graph is unchanged. Setting one entry "
          "of a list-valued option the graph does not have yet, and objects other than graphs, raise "
          "NotImplementedError. Check: exact (tests read every annotation back with AnnotationValue).")
def _annotate(spec: Expr, rules: Rule | ListOf[Rule]) -> Expr:
    gr, item = _ann_spec(spec, "Annotate")
    rs = list(rules.args) if is_list(rules) else [rules]
    if not all(_is_rule(r) for r in rs):
        raise TypeError("Annotate: annotations must be key -> value rules")
    for r in rs:
        name = _key_name(r.args[0], "Annotate")
        opts = (_set_graph(gr, name, r.args[0], r.args[1], "Annotate") if item is None
                else _set_item(gr, item, name, r.args[0], r.args[1], "Annotate"))
        gr = WG(gr.vertices, gr.edge_exprs, opts, "Annotate")
    return gr.expr()


@fn("AnnotationDelete", area=AREA, status="added",
    notes="AnnotationDelete[g] removes every option and annotation; AnnotationDelete[g, key] one graph-level key; "
          "AnnotationDelete[{g, item}] / [{g, item}, key] removes the item's entries from rule-list options and "
          "AnnotationRules. Deleting one entry of a list-valued option (VertexWeight, EdgeWeight, ...) or of a value "
          "shared by all items raises NotImplementedError. The original graph is unchanged. Check: exact (tests "
          "check AnnotationKeys before and after).")
def _anndelete(spec: Expr, key: Expr = None) -> Expr:
    gr, item = _ann_spec(spec, "AnnotationDelete")
    names = None if key is None else {_key_name(k, "AnnotationDelete") for k in _vlist_arg(key)}
    opts = dict(gr.opts)
    if item is None:
        if names is None:
            return WG(gr.vertices, gr.edge_exprs, {}).expr()
        for n in names:
            opts.pop(n, None)
        rules = []
        for r in _custom_rules(gr):
            if r.args[0] == _GP:
                kvs = [kv for kv in r.args[1].args if _key_name(kv.args[0], "AnnotationDelete") not in names]
                if kvs:
                    rules.append(rule(_GP, L(*kvs)))
            else:
                rules.append(r)
        if "AnnotationRules" in opts:
            opts["AnnotationRules"] = L(*rules)
        return WG(gr.vertices, gr.edge_exprs, opts).expr()
    for k in list(opts):
        if k not in _VERTEX_KEYS | _EDGE_KEYS or (names is not None and k not in names):
            continue
        if _item_value(gr, item, k, "AnnotationDelete") is _MISSING:
            continue
        cur = opts[k]
        if not is_list(cur) or not all(_is_rule(x) for x in cur.args):
            if names is not None or k in _LIST_KEYS or not is_list(cur):
                raise NotImplementedError(f"AnnotationDelete: {k} is given for every item at once; deleting it for "
                                          "one item is not supported")
        opts[k] = L(*(x for x in cur.args if not (_is_rule(x) and _item_matches(gr, item, x.args[0],
                                                                               "AnnotationDelete"))))
    rules = []
    for r in _custom_rules(gr):
        lhs, rhs = r.args
        if lhs != _GP and _item_matches(gr, item, lhs, "AnnotationDelete"):
            kvs = [] if names is None else [kv for kv in rhs.args
                                             if _key_name(kv.args[0], "AnnotationDelete") not in names]
            if kvs:
                rules.append(rule(lhs, L(*kvs)))
        else:
            rules.append(r)
    if "AnnotationRules" in opts:
        opts["AnnotationRules"] = L(*rules)
    return WG(gr.vertices, gr.edge_exprs, opts).expr()


_tag_aware("AnnotationValue", "AnnotationKeys", "Annotate", "AnnotationDelete")


# ═══════════════════════════════════════ constructions ═══════════════════════════════════════

def _graph_opts(args, fname, extra=()):
    pos, opts = split_options(list(args), _GRAPH_OPTS | {"DirectedEdges"} | set(extra), fname)
    return pos, opts


def _finish(vs, es, opts, fname):
    de = opts.pop("DirectedEdges", None)
    if de is not None:
        es = WGM._directed_edges(es, de, fname)
    return WG(vs, es, opts, fname).expr()


def _de(e):
    return Expr("DirectedEdge", *e)


@fn("ExpressionGraph", area=AREA, status="added",
    notes="ExpressionGraph[expr] is the expression tree as an undirected tree on 1, 2, ... (vertices in depth-first "
          "preorder: the root is 1, then each argument's subtree in order; heads are not vertices, atoms are leaves); "
          "ExpressionGraph[expr, n] stops at level n. Compound heads (f[a][b]) and the pattern form [expr, n, form] "
          "raise NotImplementedError. Check: exact (tests compare with the documented tree and check VertexCount = "
          "LeafCount-style counts).")
def _exprgraph(expr: Expr, n: NonNegInteger | Expr = None, *opts: Option) -> Expr:
    pos, o = _graph_opts(opts, "ExpressionGraph")
    if pos:
        raise NotImplementedError("ExpressionGraph: the pattern form ExpressionGraph[expr, n, form] is not supported")
    lim = None
    if n is not None and not (isinstance(n, Symbol) and n.name == "Infinity"):
        lim = _int(n, "ExpressionGraph", 0, "the level")
    vs, es = [], []

    def walk(e, depth, parent):
        vs.append(len(vs) + 1)
        me = len(vs)
        if parent is not None:
            es.append(UE(parent, me))
        if isinstance(e, Expr) and not KL.atomq(e) and (lim is None or depth < lim):
            if not isinstance(e.head, str):
                raise NotImplementedError("ExpressionGraph: compound heads are not supported")
            for a in e.args:
                walk(a, depth + 1, me)
    walk(expr, 0, None)
    return _finish(vs, es, o, "ExpressionGraph")


@fn("TorusGraph", area=AREA, status="added",
    notes="TorusGraph[{n1, ..., nk}] = Cartesian product of the cycles C_n1 ... C_nk on 1..n1*...*nk, numbered as "
          "GridGraph (first coordinate fastest), each coordinate wrapping around; every ni >= 3 (smaller sizes "
          "would need multi-edges and raise NotImplementedError). Check: exact (tests: IsomorphicGraphQ with "
          "GraphProduct[CycleGraph[m], CycleGraph[n]], 2k-regularity and n1*...*nk*k edges).")
def _torus(dims: ListOf[PosInteger], *opts: Option) -> Expr:
    pos, o = _graph_opts(opts, "TorusGraph")
    if pos:
        raise ValueError("TorusGraph: expected TorusGraph[{n1, n2, ...}]")
    if not is_list(dims) or not dims.args:
        if isinstance(dims, Symbol):
            raise Unevaluated()
        raise TypeError("TorusGraph: expected a list of sizes {n1, n2, ...}")
    ds = [_int(x, "TorusGraph", 1, "a size") for x in dims.args]
    if any(d < 3 for d in ds):
        raise NotImplementedError("TorusGraph: sizes below 3 are not supported")
    total = math.prod(ds)
    strides, s = [], 1
    for d in ds:
        strides.append(s)
        s *= d
    es = set()
    for idx in range(total):
        coords, r = [], idx
        for d in ds:
            coords.append(r % d)
            r //= d
        for k, d in enumerate(ds):
            j = idx + ((coords[k] + 1) % d - coords[k]) * strides[k]
            es.add((min(idx, j) + 1, max(idx, j) + 1))
    return _finish(list(range(1, total + 1)), [UE(a, b) for a, b in sorted(es)], o, "TorusGraph")


@fn("DeBruijnGraph", area=AREA, status="added",
    notes="DeBruijnGraph[m, n]: directed graph on 1..m^n where vertex i (the base-m digits of i - 1, length n) has "
          "an edge to each vertex whose digits are those shifted left with one new digit appended, i.e. i -> "
          "Mod[m (i - 1), m^n] + d + 1 for d = 0..m-1 (constant words give self-loops); edges sorted. The type form "
          "DeBruijnGraph[m, n, type] raises NotImplementedError. Check: exact (tests: m^(n+1) edges, in- and "
          "out-degree m everywhere, LineGraph-style identity with DeBruijnGraph[m, n + 1] counts).")
def _debruijn(m: PosInteger, n: PosInteger, *rest: Expr) -> Expr:
    pos, o = _graph_opts(rest, "DeBruijnGraph")
    if pos:
        raise NotImplementedError("DeBruijnGraph: the type form DeBruijnGraph[m, n, type] is not supported")
    m = _int(m, "DeBruijnGraph", 1, "the number of symbols")
    n = _int(n, "DeBruijnGraph", 1, "the dimension")
    N = m ** n
    if N > 100000:
        raise ValueError("DeBruijnGraph: the graph would have more than 100000 vertices")
    es = [_de((i + 1, (m * i) % N + d + 1)) for i in range(N) for d in range(m)]
    return _finish(list(range(1, N + 1)), es, o, "DeBruijnGraph")


@fn("HararyGraph", area=AREA, status="added",
    notes="HararyGraph[k, n] (1 < k < n): Harary's minimal k-connected graph H_{k,n} on 1..n with ceil(k n/2) edges: "
          "i <-> i + j (mod n) for j = 1..floor(k/2), plus for odd k the diameters i <-> i + n/2 (n even) or the "
          "edges i <-> i + (n+1)/2 for i = 1..(n+1)/2 (n odd); edges sorted. Check: exact (tests: edge count "
          "Ceiling[k n/2] and VertexConnectivity = k).")
def _harary(k: PosInteger, n: PosInteger, *opts: Option) -> Expr:
    pos, o = _graph_opts(opts, "HararyGraph")
    if pos:
        raise ValueError("HararyGraph: expected HararyGraph[k, n]")
    k = _int(k, "HararyGraph", 1, "k")
    n = _int(n, "HararyGraph", 1, "n")
    if not 1 < k < n:
        raise ValueError("HararyGraph: requires 1 < k < n")
    es = set()

    def add(a, b):
        a, b = a % n, b % n
        if a != b:
            es.add((min(a, b) + 1, max(a, b) + 1))
    r = k // 2
    for i in range(n):
        for j in range(1, r + 1):
            add(i, i + j)
    if k % 2:
        if n % 2 == 0:
            for i in range(n // 2):
                add(i, i + n // 2)
        else:
            add(0, (n - 1) // 2)
            add(0, (n + 1) // 2)
            for i in range(1, (n - 1) // 2 + 1):
                add(i, i + (n + 1) // 2)
    return _finish(list(range(1, n + 1)), [UE(a, b) for a, b in sorted(es)], o, "HararyGraph")


@fn("KnightTourGraph", area=AREA, status="added",
    notes="KnightTourGraph[m, n]: knight moves on an m x n board, vertex (i - 1) n + j for row i, column j "
          "(Wolfram's numbering is not documented; the graph is the same up to isomorphism), edges sorted. "
          "Check: exact (tests: edge count 4 m n - 6 (m + n) + 8 for m, n >= 2 and the 8 x 8 board's 168 edges).")
def _knight(m: PosInteger, n: PosInteger, *opts: Option) -> Expr:
    pos, o = _graph_opts(opts, "KnightTourGraph")
    if pos:
        raise ValueError("KnightTourGraph: expected KnightTourGraph[m, n]")
    m = _int(m, "KnightTourGraph", 1, "the number of rows")
    n = _int(n, "KnightTourGraph", 1, "the number of columns")
    es = set()
    for i in range(m):
        for j in range(n):
            for di, dj in ((1, 2), (2, 1), (1, -2), (2, -1)):
                a, b = i + di, j + dj
                if 0 <= a < m and 0 <= b < n:
                    u, v = i * n + j + 1, a * n + b + 1
                    es.add((min(u, v), max(u, v)))
    return _finish(list(range(1, m * n + 1)), [UE(a, b) for a, b in sorted(es)], o, "KnightTourGraph")


@fn("ButterflyGraph", area=AREA, status="added",
    notes="ButterflyGraph[n, b] (default b = 2): vertices (level l = 0..n, word w = 0..b^n - 1) numbered "
          "l b^n + w + 1; (l, w) <-> (l + 1, w') whenever w' differs from w at most in base-b digit l; (n + 1) b^n "
          "vertices and n b^(n+1) edges (Wolfram's numbering is not documented; same graph up to isomorphism). "
          "Check: exact (tests: counts and the degree sequence 2b inside, b on the outer levels).")
def _butterfly(n: NonNegInteger, b: PosInteger = 2, *opts: Option) -> Expr:
    if _is_rule(b):
        opts, b = (b,) + opts, 2
    pos, o = _graph_opts(opts, "ButterflyGraph")
    if pos:
        raise ValueError("ButterflyGraph: expected ButterflyGraph[n] or ButterflyGraph[n, b]")
    n = _int(n, "ButterflyGraph", 1, "the order")
    b = _int(b, "ButterflyGraph", 2, "the base")
    W = b ** n
    if (n + 1) * W > 100000:
        raise ValueError("ButterflyGraph: the graph would be too large")
    es = []
    for l in range(n):
        p = b ** l
        for w in range(W):
            digit = (w // p) % b
            for d in range(b):
                w2 = w + (d - digit) * p
                es.append(UE(l * W + w + 1, (l + 1) * W + w2 + 1))
    return _finish(list(range(1, (n + 1) * W + 1)), es, o, "ButterflyGraph")


def _icosahedron():
    phi = (1 + 5 ** 0.5) / 2
    pts = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            pts += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s2 * phi, 0, s1)]
    adj = {i: set() for i in range(12)}
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(sum((a - b) ** 2 for a, b in zip(pts[i], pts[j])) - 4) < 1e-9:
                adj[i].add(j)
                adj[j].add(i)
    return adj


@fn("BuckyballGraph", area=AREA, status="added",
    notes="BuckyballGraph[] (= BuckyballGraph[1]): the C60 fullerene (truncated icosahedron) on 1..60, built from "
          "the icosahedron (one vertex per directed edge u -> v, numbered in sorted (u, v) order); 90 edges, cubic, "
          "12 pentagons and 20 hexagons (Wolfram's numbering is not documented; same graph up to isomorphism). Higher "
          "orders and classes raise NotImplementedError. Check: exact (tests: counts, 3-regularity, girth 5, "
          "120 automorphisms, planarity with 32 faces).")
def _bucky(*args: Expr) -> Expr:
    pos, o = _graph_opts(args, "BuckyballGraph")
    if pos:
        if len(pos) == 1 and K.is_int(pos[0]) and int(pos[0]) == 1:
            pass
        elif len(pos) == 1:
            _int(pos[0], "BuckyballGraph", 1, "the order")
            raise NotImplementedError("BuckyballGraph: orders above 1 are not supported")
        else:
            raise NotImplementedError("BuckyballGraph: the class form BuckyballGraph[n, class] is not supported")
    adj = _icosahedron()
    darts = sorted((u, v) for u in adj for v in adj[u])
    idx = {d: k + 1 for k, d in enumerate(darts)}
    es = set()
    for (u, v) in darts:
        a, b = idx[(u, v)], idx[(v, u)]
        es.add((min(a, b), max(a, b)))
        for w in adj[u]:
            if w != v and w in adj[v]:
                c = idx[(u, w)]
                es.add((min(a, c), max(a, c)))
    return _finish(list(range(1, 61)), [UE(a, b) for a, b in sorted(es)], o, "BuckyballGraph")


def _same_vertex_count(grs, fname):
    n = grs[0].n
    if any(g.n != n for g in grs):
        raise NotImplementedError(f"{fname}: graphs with different numbers of vertices are not supported")
    return n


@fn("GraphSum", area=AREA, status="added",
    notes="GraphSum[g1, g2, ...] for graphs with the same number of vertices = AdjacencyGraph[AdjacencyMatrix[g1] + "
          "AdjacencyMatrix[g2] + ...] on 1..n (Wolfram's documented identity; repeated edges become multi-edges); "
          "graphs of different sizes raise NotImplementedError. Check: exact (tests: the documented identity, "
          "g + complement = complete graph).")
def _gsum(*args: Expr) -> Expr:
    pos, o = _graph_opts(args, "GraphSum")
    if len(pos) < 1:
        raise ValueError("GraphSum: expected graphs")
    grs = [_g(x, "GraphSum") for x in pos]
    n = _same_vertex_count(grs, "GraphSum")
    if n == 0:
        return _finish([], [], o, "GraphSum")
    m = [[0] * n for _ in range(n)]
    for gr in grs:
        gr.require_pure("GraphSum")
        for i, j, d in gr.edges:
            m[i][j] += 1
            if not d and i != j:
                m[j][i] += 1
    out = _adjacency_graph(list(range(1, n + 1)), m, "GraphSum")
    return _finish(list(out.args[0].args), list(out.args[1].args), o, "GraphSum")


@fn("GraphJoin", area=AREA, status="added",
    notes="GraphJoin[g1, g2, ...] = GraphDisjointUnion (vertices renumbered 1, 2, ...) plus an edge i <-> j between "
          "every vertex of one graph and every vertex of each later graph (Wolfram's documented construction), join "
          "edges in vertex order after the union's edges; directed and mixed graphs raise NotImplementedError. "
          "Check: exact (tests: the documented identities GraphJoin[point, cycle] = wheel, joins of edgeless graphs "
          "= complete multipartite graphs).")
def _gjoin(*args: Expr) -> Expr:
    pos, o = _graph_opts(args, "GraphJoin")
    if len(pos) < 1:
        raise ValueError("GraphJoin: expected graphs")
    grs = [_g(x, "GraphJoin") for x in pos]
    vs, es, blocks, off = [], [], [], 0
    for gr in grs:
        _need_undirected(gr, "GraphJoin")
        block = list(range(off + 1, off + gr.n + 1))
        vs += block
        es += [UE(off + i + 1, off + j + 1) for i, j, _ in gr.edges]
        blocks.append(block)
        off += gr.n
    for a, b in itertools.combinations(range(len(blocks)), 2):
        es += [UE(u, v) for u in blocks[a] for v in blocks[b]]
    return _finish(vs, es, o, "GraphJoin")


_PRODUCTS = ("Cartesian", "Tensor", "Lexicographical", "Normal", "Conormal", "Rooted")


@fn("GraphProduct", area=AREA, status="added",
    notes="GraphProduct[g1, g2] (Cartesian) and GraphProduct[g1, g2, \"op\"] for op = \"Cartesian\", \"Tensor\", "
          "\"Lexicographical\", \"Normal\", \"Conormal\", \"Rooted\" (root = first vertex of g2), with the "
          "documented edge conditions on vertex pairs; undirected simple graphs; the product vertex (u_i, v_j) is "
          "numbered (i - 1) n2 + j and edges are sorted (Wolfram's numbering is not documented; same graph up to "
          "isomorphism). Directed/multi/mixed graphs and more than two factors raise NotImplementedError. "
          "Check: exact (tests: the documented edge-count identities for every type).")
def _gproduct(g1: Expr, g2: Expr, *rest: String | Option) -> Expr:
    pos, o = _graph_opts(rest, "GraphProduct")
    op = "Cartesian"
    if pos:
        if len(pos) > 1:
            raise ValueError("GraphProduct: expected GraphProduct[g1, g2, \"op\"]")
        if is_graph_like(pos[0]):
            raise NotImplementedError("GraphProduct: products of more than two graphs are not supported")
        if not isinstance(pos[0], str):
            raise TypeError("GraphProduct: the product type must be a string")
        op = pos[0]
        if op not in _PRODUCTS:
            raise ValueError(f"GraphProduct: unknown product type {op!r}; expected one of {', '.join(_PRODUCTS)}")
    a, b = _g(g1, "GraphProduct"), _g(g2, "GraphProduct")
    for gr in (a, b):
        _need_simple_undirected(gr, "GraphProduct")
    na, nb = _simple_nb(a), _simple_nb(b)
    n1, n2 = a.n, b.n
    es = []
    for (u1, u2), (v1, v2) in itertools.combinations(itertools.product(range(n1), range(n2)), 2):
        e1, e2 = v1 in na[u1], v2 in nb[u2]
        s1, s2 = u1 == v1, u2 == v2
        ok = {"Cartesian": (s1 and e2) or (s2 and e1), "Tensor": e1 and e2, "Lexicographical": e1 or (s1 and e2),
              "Normal": (s1 and e2) or (s2 and e1) or (e1 and e2), "Conormal": e1 or e2,
              "Rooted": (s1 and e2) or (e1 and s2 and u2 == 0)}[op]
        if ok:
            es.append(UE(u1 * n2 + u2 + 1, v1 * n2 + v2 + 1))
    return _finish(list(range(1, n1 * n2 + 1)), es, o, "GraphProduct")


@fn("GraphPower", area=AREA, status="added",
    notes="GraphPower[g, n]: same vertices, an edge u <-> v (u -> v for directed graphs) whenever 0 < "
          "GraphDistance[g, u, v] <= n, in VertexList order; unweighted, loop-free and multi-edge-free result; mixed "
          "graphs raise. Check: exact (tests: GraphPower[CycleGraph[7], 2] = CirculantGraph[7, {1, 2}], "
          "GraphPower[PathGraph, diameter] = complete graph).")
def _gpower(g: Expr, n: NonNegInteger, *opts: Option) -> Expr:
    pos, o = _graph_opts(opts, "GraphPower")
    if pos:
        raise ValueError("GraphPower: expected GraphPower[g, n]")
    gr = _g(g, "GraphPower")
    gr.require_pure("GraphPower")
    k = _int(n, "GraphPower", 0, "the power")
    plain = WG(gr.vertices, gr.edge_exprs, {}, "GraphPower")
    es = []
    for i in range(gr.n):
        d, _ = plain.sssp(i)
        for j in range(gr.n):
            if j != i and j in d and 0 < d[j] <= k and (gr.directed or j > i):
                es.append(gr.edge(i, j, gr.directed))
    return _finish(gr.vertices, es, o, "GraphPower")


@fn("SimpleGraph", area=AREA, status="added",
    notes="SimpleGraph[g]: the underlying simple graph of an undirected graph (self-loops removed, multiple edges "
          "merged at their first occurrence, vertices kept); directed and mixed graphs raise NotImplementedError. "
          "Check: exact (tests: SimpleGraphQ of the result and edge counts).")
def _simpleg(g: Expr, *opts: Option) -> Expr:
    pos, o = _graph_opts(opts, "SimpleGraph")
    if pos:
        raise ValueError("SimpleGraph: expected SimpleGraph[g]")
    gr = _g(g, "SimpleGraph")
    _need_undirected(gr, "SimpleGraph")
    seen, es = set(), []
    for k, (i, j, _) in enumerate(gr.edges):
        key = (min(i, j), max(i, j))
        if i == j or key in seen:
            continue
        seen.add(key)
        es.append(gr.edge_exprs[k])
    return _finish(gr.vertices, es, o, "SimpleGraph")


@fn("RelationGraph", area=AREA, status="added",
    notes="RelationGraph[f, {v1, ...}]: an edge for every pair with f[vi, vj] True (self-loops when f[v, v] is "
          "True); DirectedEdges -> Automatic gives an undirected graph when f[vi, vj] == f[vj, vi] for all pairs "
          "(edges i <= j in list order) and a directed one otherwise (all ordered pairs, row by row); True/False "
          "force the kind. f results other than True/False count as False. The two-list form RelationGraph[f, {v...}, "
          "{w...}] raises NotImplementedError. Check: exact (tests: documented examples, e.g. a coprime graph and "
          "Greater giving the transitive tournament).")
def _relgraph(f: Fn, vs: ListOf[Expr], *rest: Option) -> Expr:
    pos, o = _graph_opts(rest, "RelationGraph")
    if pos:
        raise NotImplementedError("RelationGraph: the two-list form RelationGraph[f, {v...}, {w...}] is not supported")
    if not is_list(vs):
        if isinstance(vs, Symbol):
            raise Unevaluated()
        raise TypeError("RelationGraph: the vertices must be a list")
    vl = list(vs.args)
    n = len(vl)
    rel = [[apply(f, vl[i], vl[j]) is True for j in range(n)] for i in range(n)]
    de = o.pop("DirectedEdges", Symbol("Automatic"))
    if isinstance(de, Symbol) and de.name == "Automatic":
        directed = any(rel[i][j] != rel[j][i] for i in range(n) for j in range(n))
    elif de is True or de is False:
        directed = de
    else:
        raise ValueError("RelationGraph: DirectedEdges must be True, False or Automatic")
    es = []
    for i in range(n):
        for j in range(n):
            if directed and rel[i][j]:
                es.append(Expr("DirectedEdge", vl[i], vl[j]))
            elif not directed and j >= i and (rel[i][j] or rel[j][i]):
                es.append(UE(vl[i], vl[j]))
    return WG(vl, es, o, "RelationGraph").expr()


def _bool_value(f, vals):
    r = apply(f, *vals)
    if r is True or r is False:
        return r
    raise ValueError(f"BooleanGraph: the Boolean function gave {r!r}, not True or False")


@fn("BooleanGraph", area=AREA, status="added",
    notes="BooleanGraph[f, g1, ..., gn] for simple undirected graphs on the same vertex set (taken from g1): u <-> v "
          "for every pair u != v with f[adjacent in g1, ..., adjacent in gn] True (edges in VertexList order); "
          "BooleanGraph[Or, g1, g2] is the union, And the intersection, Xor the symmetric difference, Not[g] the "
          "complement. Graphs with different vertex sets, directed and multigraphs raise NotImplementedError; a "
          "non-Boolean f result raises ValueError. Check: exact (tests: Or/And/Xor/Not against GraphUnion, "
          "GraphIntersection and GraphComplement).")
def _boolgraph(f: Fn, *gs: Expr) -> Expr:
    pos, o = _graph_opts(gs, "BooleanGraph")
    if not pos:
        raise ValueError("BooleanGraph: expected at least one graph")
    grs = [_g(x, "BooleanGraph") for x in pos]
    for gr in grs:
        _need_simple_undirected(gr, "BooleanGraph")
    base = grs[0]
    for gr in grs[1:]:
        if set(gr.index) != set(base.index):
            raise NotImplementedError("BooleanGraph: graphs with different vertex sets are not supported")
    nbs = [gr.neighbor_sets(undirected=True) for gr in grs]
    pos_of = [[gr.index[K.tag(v)] for v in base.vertices] for gr in grs]
    es = []
    for i, j in itertools.combinations(range(base.n), 2):
        vals = [pos_of[k][j] in nbs[k][pos_of[k][i]] for k in range(len(grs))]
        if _bool_value(f, vals):
            es.append(UE(base.vertices[i], base.vertices[j]))
    return _finish(base.vertices, es, o, "BooleanGraph")


def _weighted_matrix(m, fname):
    if not is_list(m) or not m.args or not all(is_list(r) for r in m.args):
        if isinstance(m, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a square matrix")
    n = len(m.args)
    if any(len(r.args) != n for r in m.args):
        raise ValueError(f"{fname}: expected a square matrix")
    return [list(r.args) for r in m.args]


@fn("WeightedAdjacencyGraph", area=AREA, status="added",
    notes="WeightedAdjacencyGraph[wmat] / [{v...}, wmat]: an edge with weight w_ij for every entry that is not "
          "Infinity (the default absent value) or the given val in WeightedAdjacencyGraph[{v...}, wmat, val]; "
          "undirected when wmat is symmetric (edges i <= j), directed otherwise; EdgeWeight carries the weights. "
          "Entries must be real numbers or Infinity. Check: exact (tests: WeightedAdjacencyMatrix round trip).")
def _wadjgraph(a: Expr, b: Expr = None, c: Expr = None) -> Expr:
    if b is None or (_is_rule(b)):
        vs, m, absent = None, a, INF
    else:
        vs, m, absent = a, b, (INF if c is None else S(c))
    rows = _weighted_matrix(m, "WeightedAdjacencyGraph")
    n = len(rows)
    if vs is None:
        vl = list(range(1, n + 1))
    else:
        if not is_list(vs) or len(vs.args) != n:
            raise ValueError("WeightedAdjacencyGraph: the vertex list and matrix sizes differ")
        vl = list(vs.args)
    vals = []
    for r in rows:
        row = []
        for x in r:
            if isinstance(x, Symbol) and x.name == "Infinity":
                row.append(INF)
                continue
            if isinstance(x, (bool, str)) or not is_numeric_value(x):
                raise TypeError(f"WeightedAdjacencyGraph: entries must be numbers or Infinity, not {x!r}")
            v = S(x)
            if not v.is_real:
                raise ValueError("WeightedAdjacencyGraph: entries must be real")
            row.append(v)
        vals.append(row)
    sym = all(vals[i][j] == vals[j][i] for i in range(n) for j in range(n))
    es, ws = [], []
    for i in range(n):
        for j in range(n):
            w = vals[i][j]
            if w == absent or (sym and j < i):
                continue
            es.append(UE(vl[i], vl[j]) if sym else Expr("DirectedEdge", vl[i], vl[j]))
            ws.append(R(w))
    return WG(vl, es, {"EdgeWeight": L(*ws)} if es else {}, "WeightedAdjacencyGraph").expr()


@fn("KirchhoffGraph", area=AREA, status="added",
    notes="KirchhoffGraph[kmat] / [{v...}, kmat]: the undirected loop-free (multi)graph whose Kirchhoff matrix "
          "D - A is kmat: -k off the diagonal gives k parallel edges; kmat must be symmetric with non-positive "
          "integer off-diagonal entries and zero row sums. Check: self-check (KirchhoffMatrix of the result is "
          "recomputed and compared; ArithmeticError on mismatch).")
def _kirchgraph(a: Expr, b: Expr = None) -> Expr:
    vs, m = (None, a) if b is None else (a, b)
    rows = _weighted_matrix(m, "KirchhoffGraph")
    n = len(rows)
    ints = []
    for r in rows:
        row = []
        for x in r:
            if not K.is_int(x) or isinstance(x, bool):
                raise ValueError("KirchhoffGraph: entries must be integers")
            row.append(int(x))
        ints.append(row)
    for i in range(n):
        if sum(ints[i]) != 0:
            raise ValueError("KirchhoffGraph: every row of a Kirchhoff matrix sums to 0")
        for j in range(n):
            if ints[i][j] != ints[j][i] or (i != j and ints[i][j] > 0):
                raise ValueError("KirchhoffGraph: the matrix must be symmetric with non-positive off-diagonal "
                                 "entries")
    vl = list(range(1, n + 1)) if vs is None else list(vs.args)
    if len(vl) != n:
        raise ValueError("KirchhoffGraph: the vertex list and matrix sizes differ")
    es = []
    for i in range(n):
        for j in range(i + 1, n):
            es += [UE(vl[i], vl[j])] * (-ints[i][j])
    out = _new(vl, es)
    km = WGM._kirch(out) if n else L()
    if n and [[int(x) for x in r.args] for r in km.args] != ints:
        raise ArithmeticError("KirchhoffGraph: result failed verification")
    return out


@fn("IncidenceGraph", area=AREA, status="added",
    notes="IncidenceGraph[m] / [{v...}, m]: one edge per column of the vertices x edges incidence matrix: two 1s "
          "give an undirected edge, a single 2 a self-loop, -1 and 1 a directed edge from the -1 row to the 1 row; "
          "other columns raise ValueError. Check: self-check (IncidenceMatrix of the result is compared with m; "
          "ArithmeticError on mismatch).")
def _incgraph(a: Expr, b: Expr = None) -> Expr:
    vs, m = (None, a) if b is None else (a, b)
    if not is_list(m) or not m.args or not all(is_list(r) for r in m.args):
        if isinstance(m, Symbol):
            raise Unevaluated()
        raise TypeError("IncidenceGraph: expected a matrix")
    rows = [list(r.args) for r in m.args]
    n, cols = len(rows), len(rows[0])
    if any(len(r) != cols for r in rows):
        raise ValueError("IncidenceGraph: expected a rectangular matrix")
    vl = list(range(1, n + 1)) if vs is None else list(vs.args)
    if len(vl) != n:
        raise ValueError("IncidenceGraph: the vertex list and matrix sizes differ")
    es = []
    for c in range(cols):
        col = []
        for i in range(n):
            x = rows[i][c]
            if not K.is_int(x) or isinstance(x, bool):
                raise ValueError("IncidenceGraph: entries must be integers")
            if int(x):
                col.append((i, int(x)))
        if len(col) == 1 and col[0][1] == 2:
            es.append(UE(vl[col[0][0]], vl[col[0][0]]))
        elif len(col) == 2 and col[0][1] == col[1][1] == 1:
            es.append(UE(vl[col[0][0]], vl[col[1][0]]))
        elif len(col) == 2 and sorted(v for _, v in col) == [-1, 1]:
            t = next(i for i, v in col if v == -1)
            h = next(i for i, v in col if v == 1)
            es.append(Expr("DirectedEdge", vl[t], vl[h]))
        else:
            raise ValueError(f"IncidenceGraph: column {c + 1} is not the incidence vector of an edge")
    out = _new(vl, es)
    if es:
        chk = WGM._incm(out)
        if [[int(x) for x in r.args] for r in chk.args] != [[int(x) for x in r] for r in rows]:
            raise ArithmeticError("IncidenceGraph: result failed verification")
    return out


# ═══════════════════════════════════════ edges, tags and editing ═══════════════════════════════════════

@fn("EdgeRules", area=AREA, status="added",
    notes="EdgeRules[g] = {u -> v, ...}: one rule per edge in EdgeList order (undirected edges as u -> v in their "
          "stored orientation; tags dropped). Check: exact (tests: EdgeRules of edge-rule input gives it back).")
def _edgerules(g: Expr) -> ListOf[Rule]:
    gr = _g(g, "EdgeRules")
    return L(*(Expr("Rule", e.args[0], e.args[1]) for e in gr.edge_exprs))


@fn("IncidenceList", area=AREA, status="added",
    notes="IncidenceList[g, v]: the edges incident to v (either end), in EdgeList order; unknown vertices raise "
          "ValueError; the pattern and distance forms IncidenceList[g, patt, d] raise NotImplementedError. "
          "Check: exact (tests: counts equal VertexDegree for loop-free graphs).")
def _inclist(g: Expr, v: Expr, *rest: Expr) -> ListOf[Expr]:
    if rest:
        raise NotImplementedError("IncidenceList: the distance form IncidenceList[g, v, d] is not supported")
    gr = _g(g, "IncidenceList")
    i = gr.vidx(v, "IncidenceList")
    return L(*(gr.edge_exprs[k] for k, (a, b, _) in enumerate(gr.edges) if i in (a, b)))


_tag_aware("EdgeTaggedGraph", "EdgeTags", "EdgeTaggedGraphQ", "IndexEdgeTaggedGraph", "EdgeRules", "IncidenceList")


def _tagged(e, t):
    h = "DirectedEdge" if e.head in ("Rule", "DirectedEdge") else "UndirectedEdge"
    return Expr(h, e.args[0], e.args[1], t)


@fn("EdgeTaggedGraph", area=AREA, status="added",
    notes="EdgeTaggedGraph[{e1, ...} -> {t1, ...}] and EdgeTaggedGraph[{v...}, {e...} -> {t...}] give a Graph with "
          "edges DirectedEdge[u, v, t] / UndirectedEdge[u, v, t]; EdgeTaggedGraph[{e...}] keeps edges that are "
          "already tagged. Wolfram's automatic tags for untagged edges are not reproduced, so untagged edges without "
          "a tag list raise NotImplementedError; tag lists of the wrong length raise ValueError. Tagged graphs are "
          "accepted by Graph, VertexList, EdgeList, VertexCount, EdgeCount, EdgeRules, IncidenceList, EdgeTags and "
          "the annotation functions; other graph functions raise for tagged edges rather than drop the tags. "
          "Check: exact (tests: EdgeTags round trip).")
def _etg(a: Expr, b: Expr = None, *opts: Option) -> Expr:
    if b is not None and _is_rule(b) and isinstance(b.args[0], Symbol) and b.args[0].name in _GRAPH_OPTS:
        opts, b = (b,) + opts, None
    pos, o = _graph_opts(opts, "EdgeTaggedGraph")
    vs = None
    spec = a
    if b is not None:
        if not is_list(a):
            raise TypeError("EdgeTaggedGraph: the vertices must be a list")
        vs, spec = list(a.args), b
    if _is_rule(spec):
        es, ts = spec.args
        if not (is_list(es) and is_list(ts)):
            raise TypeError("EdgeTaggedGraph: expected {e1, ...} -> {t1, ...}")
        if len(es.args) != len(ts.args):
            raise ValueError("EdgeTaggedGraph: the edge and tag lists have different lengths")
        edges = [_tagged(e, t) for e, t in zip(es.args, ts.args)]
    elif is_list(spec):
        edges = list(spec.args)
        if not all(_is_edge(e) and len(e.args) == 3 for e in edges):
            raise NotImplementedError("EdgeTaggedGraph: automatic tags for untagged edges are not supported; give "
                                      "{e1, ...} -> {t1, ...}")
    else:
        if isinstance(spec, Symbol):
            raise Unevaluated()
        raise TypeError("EdgeTaggedGraph: expected a list of edges")
    if vs is None:
        vs, seen = [], set()
        for e in edges:
            for x in e.args[:2]:
                if K.tag(x) not in seen:
                    seen.add(K.tag(x))
                    vs.append(x)
    return WG(vs, edges, o, "EdgeTaggedGraph").expr()


@fn("IndexEdgeTaggedGraph", area=AREA, status="added",
    notes="IndexEdgeTaggedGraph[g] tags the edges 1, 2, ... in EdgeList order; IndexEdgeTaggedGraph[g, r] starts at r "
          "(existing tags are replaced). Check: exact (tests: EdgeTags gives Range).")
def _ietg(g: Expr, r: Integer = 1) -> Expr:
    gr = _g(g, "IndexEdgeTaggedGraph")
    r = _int(r, "IndexEdgeTaggedGraph", None, "the first tag")
    es = [_tagged(e, r + k) for k, e in enumerate(gr.edge_exprs)]
    return WG(gr.vertices, es, gr.opts, "IndexEdgeTaggedGraph").expr()


@fn("EdgeTags", area=AREA, status="added",
    notes="EdgeTags[g]: the tags of all edges in EdgeList order; EdgeTags[g, {u, v}]: tags of the edges between u and "
          "v (an undirected edge in either orientation, a directed edge u -> v). Untagged edges raise "
          "NotImplementedError (Wolfram's convention for them is not reproduced). Check: exact (tests: "
          "EdgeTaggedGraph/IndexEdgeTaggedGraph round trips).")
def _etags(g: Expr, uv: ListOf[Expr] = None) -> ListOf[Expr]:
    gr = _g(g, "EdgeTags")
    if any(len(e.args) != 3 for e in gr.edge_exprs):
        raise NotImplementedError("EdgeTags: the graph has untagged edges")
    if uv is None:
        return L(*(e.args[2] for e in gr.edge_exprs))
    if not (is_list(uv) and len(uv.args) == 2):
        raise TypeError("EdgeTags: the second argument must be {u, v}")
    a, b = gr.vidx(uv.args[0], "EdgeTags"), gr.vidx(uv.args[1], "EdgeTags")
    return L(*(gr.edge_exprs[k].args[2] for k, (i, j, d) in enumerate(gr.edges)
               if (i, j) == (a, b) or (not d and (j, i) == (a, b))))


@fn("EdgeTaggedGraphQ", area=AREA, status="added",
    notes="True for a Graph with at least one tagged edge (DirectedEdge[u, v, t] or UndirectedEdge[u, v, t]), False "
          "for other graphs and non-graphs; symbols stay unevaluated. Check: exact.")
def _etgq(g: Expr) -> Boolean:
    if isinstance(g, Symbol):
        raise Unevaluated()
    if not is_graph_like(g):
        return False
    try:
        gr = graph_of(g, "EdgeTaggedGraphQ")
    except ValueError:
        return False
    return any(len(e.args) == 3 for e in gr.edge_exprs)


def _data_opts(gr):
    return {k: v for k, v in gr.opts.items() if k in ("EdgeWeight", "EdgeCapacity", "EdgeCost")}


@fn("VertexReplace", area=AREA, status="added",
    notes="VertexReplace[g, {v1 -> w1, ...}] renames vertices (others kept); edges follow, edge-indexed options "
          "(EdgeWeight, EdgeCapacity, EdgeCost) are kept, vertex-keyed presentation options are dropped; renaming "
          "two vertices to the same name raises ValueError. Check: exact (tests: IsomorphicGraphQ with the input "
          "and the renamed vertex list).")
def _vreplace(g: Expr, rules: Rule | ListOf[Rule]) -> Expr:
    gr = _g(g, "VertexReplace")
    rs = list(rules.args) if is_list(rules) else [rules]
    if not all(_is_rule(r) for r in rs):
        raise TypeError("VertexReplace: expected rules v -> w")
    m = {}
    for r in rs:
        m.setdefault(K.tag(r.args[0]), r.args[1])
    vs = [m.get(K.tag(v), v) for v in gr.vertices]
    if len({K.tag(v) for v in vs}) != len(vs):
        raise ValueError("VertexReplace: the replacement maps two vertices to the same name")
    es = [Expr(e.head, *(m.get(K.tag(x), x) for x in e.args[:2]), *e.args[2:]) for e in gr.edge_exprs]
    return WG(vs, es, _data_opts(gr), "VertexReplace").expr()


def _contract(gr, groups, fname):
    """Contract each group (list of vertex indices; the first index names the merged vertex)."""
    if gr.weights is not None or gr.capacities is not None:
        raise NotImplementedError(f"{fname}: weighted graphs are not supported")
    rep = list(range(gr.n))
    for grp in groups:
        for i in grp:
            rep[i] = grp[0]
    keep = [i for i in range(gr.n) if rep[i] == i]
    es, seen = [], set()
    for i, j, d in gr.edges:
        a, b = rep[i], rep[j]
        if a == b and i != j:
            continue
        key = (a, b, d) if d else (min(a, b), max(a, b), d)
        if key in seen:
            raise NotImplementedError(f"{fname}: the contraction creates multiple edges, whose merging convention "
                                      "is not reproduced")
        seen.add(key)
        es.append(gr.edge(a, b, d))
    if gr.loops() and gr.multi():
        raise NotImplementedError(f"{fname}: multigraphs are not supported")
    return WG([gr.vertices[i] for i in keep], es, {}, fname).expr()


@fn("VertexContract", area=AREA, status="added",
    notes="VertexContract[g, {v1, v2, ...}] merges the vertices into v1 (kept at v1's place in VertexList; edges "
          "between them are removed, the others are redirected to v1); VertexContract[g, {{...}, ...}] several "
          "disjoint groups. Contractions that would create multiple edges, weighted graphs and overlapping groups "
          "raise NotImplementedError/ValueError. Check: exact (tests: VertexContract[CycleGraph[n], {1, 2}] is a "
          "cycle on n - 1 vertices, edge counts).")
def _vcontract(g: Expr, spec: ListOf[Expr]) -> Expr:
    gr = _g(g, "VertexContract")
    gr.require_pure("VertexContract")
    if not is_list(spec):
        raise TypeError("VertexContract: the vertices must be given as a list")
    raw = [list(s.args) for s in spec.args] if spec.args and all(is_list(s) for s in spec.args) else [list(spec.args)]
    groups, used = [], set()
    for grp in raw:
        idx = [gr.vidx(v, "VertexContract") for v in grp]
        if used & set(idx) or len(set(idx)) != len(idx):
            raise ValueError("VertexContract: the vertex groups must be disjoint")
        used |= set(idx)
        if idx:
            groups.append(idx)
    return _contract(gr, groups, "VertexContract")


@fn("EdgeContract", area=AREA, status="added",
    notes="EdgeContract[g, e] / [g, {e1, ...}] contracts the edges: the endpoints of each connected set of contracted "
          "edges merge into the vertex that comes first in VertexList (edges among them removed); contractions that "
          "would create multiple edges and weighted graphs raise NotImplementedError; edges not in g raise "
          "ValueError. Check: exact (tests: EdgeContract[CycleGraph[n], 1 <-> 2] is C_(n-1); contracting a spanning "
          "tree leaves one vertex).")
def _econtract(g: Expr, e: Expr) -> Expr:
    gr = _g(g, "EdgeContract")
    gr.require_pure("EdgeContract")
    elist = list(e.args) if is_list(e) and not _is_edge(e) else [e]
    ks = _edge_indices(gr, elist, "EdgeContract")
    if ks is None:
        raise ValueError(f"EdgeContract: {e} is not an edge (list) of the graph")
    parent = list(range(gr.n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for k in ks:
        i, j, _ = gr.edges[k]
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups = {}
    for i in range(gr.n):
        groups.setdefault(find(i), []).append(i)
    return _contract(gr, [sorted(v) for v in groups.values() if len(v) > 1], "EdgeContract")


def _reach(gr, srcs, lo, hi, reverse):
    adj = gr.in_adj() if reverse else gr.out_adj()
    dist = {s: 0 for s in srcs}
    order = list(dict.fromkeys(srcs))
    q = deque(order)
    while q:
        u = q.popleft()
        if hi is not None and dist[u] >= hi:
            continue
        for w, _, _ in adj[u]:
            if w not in dist:
                dist[w] = dist[u] + 1
                order.append(w)
                q.append(w)
    return [i for i in order if dist[i] >= lo]


def _kspec(k, fname):
    if k is None or (isinstance(k, Symbol) and k.name == "Infinity"):
        return 0, None
    if is_list(k) and len(k.args) == 1:
        v = _int(k.args[0], fname, 0, "the distance")
        return v, v
    return 0, _int(k, fname, 0, "the distance")


def _component(g, vs, k, fname, reverse):
    gr = _g(g, fname)
    gr.require_pure(fname)
    srcs = [gr.vidx(v, fname) for v in _vlist_arg(vs)]
    lo, hi = _kspec(k, fname)
    return gr, _reach(gr, srcs, lo, hi, reverse)


@fn("VertexComponent", area=AREA, status="added",
    notes="VertexComponent[g, {v1, ...}] the vertices with a path to at least one vi (for undirected graphs, the "
          "union of their components); [g, vs, k] paths of length at most k; [g, vs, {k}] exactly k (shortest "
          "distance); the given vertices first, then breadth-first order. Check: exact (tests: equals "
          "VertexInComponent for directed graphs and ConnectedComponents for undirected ones).")
def _vcomp(g: Expr, vs: Expr, k: Expr = None) -> ListOf[Expr]:
    gr, idx = _component(g, vs, k, "VertexComponent", True)
    return _verts(gr, idx)


def _component_graph(name, reverse, what):
    @fn(name, area=AREA, status="added",
        notes=f"{name}[g, {{v1, ...}}] the subgraph induced by the vertices with a directed path {what} at least one "
              f"vi; [g, vs, k] within distance k (vertices in VertexList order, as Subgraph). Check: exact (tests: "
              f"VertexList equals Sort-compatible {'VertexInComponent' if reverse else 'VertexOutComponent'}).")
    def impl(g: Expr, vs: Expr, k: Expr = None) -> Expr:
        gr, idx = _component(g, vs, k, name, reverse)
        return _subgraph(g, _verts(gr, idx))
    return impl


_component_graph("VertexInComponentGraph", True, "to")
_component_graph("VertexOutComponentGraph", False, "from")


@fn("WeaklyConnectedGraphComponents", area=AREA, status="added",
    notes="WeaklyConnectedGraphComponents[g]: one induced Subgraph per weakly connected component, largest first; "
          "[g, {v...}] only the components containing some vi. Pattern forms raise NotImplementedError. "
          "Check: exact (tests: vertex lists equal WeaklyConnectedComponents).")
def _wcgc(g: Expr, vs: Expr = None) -> ListOf[Expr]:
    gr = _g(g, "WeaklyConnectedGraphComponents")
    comps = _weak_components(gr)
    comps.sort(key=lambda c: (-len(c), c[0]))
    if vs is not None:
        if not is_list(vs):
            raise NotImplementedError("WeaklyConnectedGraphComponents: pattern forms are not supported")
        want = {gr.vidx(v, "WeaklyConnectedGraphComponents") for v in vs.args}
        comps = [c for c in comps if want & set(c)]
    return L(*(_subgraph(g, _verts(gr, c)) for c in comps))


@fn("HighlightGraph", area=AREA, status="added",
    notes="HighlightGraph[g, {a1, ...}] returns g with GraphHighlight -> {a1, ...} (vertices, edges or subgraphs of "
          "g; items not in g raise ValueError); a single item is wrapped in a list; extra Graph options are added. "
          "Drawing the highlight is the graphics batch. Symbolic wrappers (Style[...]) raise NotImplementedError. "
          "Check: exact (tests: AnnotationValue[h, GraphHighlight] and unchanged VertexList/EdgeList).")
def _highlight(g: Expr, spec: Expr, *opts: Option) -> Expr:
    gr = _g(g, "HighlightGraph")
    pos, o = _graph_opts(opts, "HighlightGraph")
    if pos:
        raise ValueError("HighlightGraph: expected HighlightGraph[g, {items}]")
    items = list(spec.args) if is_list(spec) else [spec]
    for it in items:
        if isinstance(it, Expr) and it.head == "Graph":
            sub = _g(it, "HighlightGraph")
            for v in sub.vertices:
                gr.vidx(v, "HighlightGraph")
            if _edge_indices(gr, sub.edge_exprs, "HighlightGraph") is None:
                raise ValueError("HighlightGraph: a highlighted subgraph has edges that are not in the graph")
        elif _is_edge(it):
            if _edge_indices(gr, [it], "HighlightGraph") is None:
                raise ValueError(f"HighlightGraph: {it} is not an edge of the graph")
        elif isinstance(it, Expr) and it.head in ("Style", "Labeled", "Annotation"):
            raise NotImplementedError("HighlightGraph: symbolic highlight wrappers are not supported")
        else:
            gr.vidx(it, "HighlightGraph")
    opts2 = dict(gr.opts)
    opts2["GraphHighlight"] = L(*items)
    opts2.update(o)
    return WG(gr.vertices, gr.edge_exprs, opts2, "HighlightGraph").expr()


def _planar_nx(gr, fname):
    nx = _nx()
    X = nx.Graph()
    X.add_nodes_from(range(gr.n))
    X.add_edges_from((i, j) for i, j, _ in gr.edges if i != j)
    ok, emb = nx.check_planarity(X)
    return ok, emb


@fn("PlanarGraph", area=AREA, status="added",
    notes="PlanarGraph[{e...}], PlanarGraph[{v...}, {e...}] or PlanarGraph[g] give the Graph after checking that it "
          "is planar (NetworkX's left-right test); non-planar input raises ValueError. The planar drawing itself is "
          "the graphics batch. Check: delegated (networkx.check_planarity; tests: K4 and the cube accepted, K5 and "
          "K3,3 rejected).")
def _planarg(*args: Expr) -> Expr:
    pos, o = _graph_opts(args, "PlanarGraph")
    if len(pos) == 1:
        gr = _g(pos[0], "PlanarGraph")
    elif len(pos) == 2:
        gr = _g(Expr("Graph", *pos), "PlanarGraph")
    else:
        raise ValueError("PlanarGraph: expected PlanarGraph[{e...}] or PlanarGraph[{v...}, {e...}]")
    ok, _ = _planar_nx(gr, "PlanarGraph")
    if not ok:
        raise ValueError("PlanarGraph: the graph is not planar")
    opts2 = dict(gr.opts)
    opts2.update(o)
    return WG(gr.vertices, gr.edge_exprs, opts2, "PlanarGraph").expr()


def _point(x, fname):
    if is_list(x):
        return [float(S(_real(c, fname, "a coordinate"))) for c in x.args]
    return [float(S(_real(x, fname, "an element")))]


@fn("NearestNeighborGraph", area=AREA, status="added",
    notes="NearestNeighborGraph[{e1, ...}] / [{e...}, k]: undirected graph on the elements (numbers or equal-length "
          "numeric vectors, Euclidean distance) with ei <-> ej whenever ej is among the k (default 1) nearest "
          "elements of ei or vice versa, edges in list order; [{e...}, {k, r}] keeps only neighbours within distance "
          "r. Ties at the k-th distance and repeated elements raise NotImplementedError/ValueError (Wolfram's tie "
          "rule is not reproduced); other element types raise NotImplementedError. Check: reference (tests compare "
          "with brute-force distance tables).")
def _nngraph(elems: ListOf[Expr], k: PosInteger | ListOf[Expr] = 1, *opts: Option) -> Expr:
    if _is_rule(k):
        opts, k = (k,) + opts, 1
    pos, o = _graph_opts(opts, "NearestNeighborGraph")
    if pos:
        raise ValueError("NearestNeighborGraph: expected NearestNeighborGraph[elems, k]")
    if not is_list(elems):
        if isinstance(elems, Symbol):
            raise Unevaluated()
        raise TypeError("NearestNeighborGraph: the elements must be a list")
    r = None
    if is_list(k):
        if len(k.args) != 2:
            raise ValueError("NearestNeighborGraph: expected {k, r}")
        k, r = k.args
        r = float(_real(r, "NearestNeighborGraph", "the radius", lo=0))
        if isinstance(k, Symbol) and k.name == "All":
            k = None
    if k is not None:
        k = _int(k, "NearestNeighborGraph", 1, "k")
    vl = list(elems.args)
    for e in vl:
        if isinstance(e, str) or not (is_numeric_value(e) or (is_list(e) and all(is_numeric_value(c) for c in e.args))):
            raise NotImplementedError("NearestNeighborGraph: only numbers and numeric vectors are supported")
    pts = [_point(e, "NearestNeighborGraph") for e in vl]
    if len({len(p) for p in pts}) > 1:
        raise ValueError("NearestNeighborGraph: the vectors must have the same length")
    if len({tuple(p) for p in pts}) != len(pts):
        raise ValueError("NearestNeighborGraph: repeated elements are not supported")
    n = len(pts)
    es = set()
    for i in range(n):
        ds = sorted((math.dist(pts[i], pts[j]), j) for j in range(n) if j != i)
        if r is not None:
            ds = [x for x in ds if x[0] <= r + 1e-12]
        sel = ds if k is None else ds[:k]
        if k is not None and len(ds) > k and abs(ds[k][0] - ds[k - 1][0]) <= 1e-12 * max(1.0, ds[k][0]):
            raise NotImplementedError("NearestNeighborGraph: ties among the nearest neighbours are not supported")
        for _, j in sel:
            es.add((min(i, j), max(i, j)))
    return _finish(vl, [UE(vl[a], vl[b]) for a, b in sorted(es)], o, "NearestNeighborGraph")


def _reach_sets(gr):
    adj = [[j for j, _, _ in a] for a in gr.out_adj()]
    out = []
    for s in range(gr.n):
        seen, st = set(), list(adj[s])
        while st:
            u = st.pop()
            if u not in seen:
                seen.add(u)
                st += adj[u]
        out.append(seen)
    return out


@fn("TransitiveClosureGraph", area=AREA, status="added",
    notes="TransitiveClosureGraph[g]: for a directed acyclic graph, u -> v for every v reachable from u (edges in "
          "VertexList order); for an undirected graph, each connected component becomes a complete graph. Directed "
          "graphs with cycles (whose closure would need self-loops) and mixed graphs raise NotImplementedError. "
          "Check: exact (tests: closure of a path is the transitive tournament; idempotence).")
def _tclosure(g: Expr) -> Expr:
    gr = _g(g, "TransitiveClosureGraph")
    gr.require_pure("TransitiveClosureGraph")
    if not gr.directed:
        es = []
        for c in _weak_components(gr):
            es += [(a, b) for a, b in itertools.combinations(sorted(c), 2)]
        return _new(gr.vertices, [UE(gr.vertices[a], gr.vertices[b]) for a, b in sorted(es)])
    if len(_topo(gr)) != gr.n:
        raise NotImplementedError("TransitiveClosureGraph: directed graphs with cycles are not supported")
    rs = _reach_sets(gr)
    return _new(gr.vertices, [gr.edge(i, j, True) for i in range(gr.n) for j in sorted(rs[i])])


@fn("TransitiveReductionGraph", area=AREA, status="added",
    notes="TransitiveReductionGraph[g] for a directed acyclic graph: the unique minimal subgraph with the same "
          "reachability (an edge u -> v is kept unless v is reachable from another successor of u; parallel edges "
          "merged), edges in input order. Cyclic and undirected graphs (where the reduction is not unique) raise "
          "NotImplementedError. Check: self-check (TransitiveClosureGraph of the result and of g are compared; "
          "ArithmeticError on mismatch).")
def _treduction(g: Expr) -> Expr:
    gr = _g(g, "TransitiveReductionGraph")
    gr.require_pure("TransitiveReductionGraph")
    if gr.edges and not gr.directed:
        raise NotImplementedError("TransitiveReductionGraph: undirected graphs are not supported")
    if len(_topo(gr)) != gr.n:
        raise NotImplementedError("TransitiveReductionGraph: directed graphs with cycles are not supported")
    rs = _reach_sets(gr)
    succ = [set() for _ in range(gr.n)]
    for i, j, _ in gr.edges:
        succ[i].add(j)
    keep, seen = [], set()
    for k, (i, j, _) in enumerate(gr.edges):
        if (i, j) in seen:
            continue
        seen.add((i, j))
        if not any(j in rs[w] for w in succ[i] if w != j):
            keep.append(gr.edge_exprs[k])
    out = _new(gr.vertices, keep)
    if gr.edges and _tclosure(out) != _tclosure(_new(gr.vertices, gr.edge_exprs)):
        raise ArithmeticError("TransitiveReductionGraph: result failed verification")
    return out


@fn("CayleyGraph", area=AREA, status="added",
    notes="CayleyGraph[group] for PermutationGroup, SymmetricGroup, CyclicGroup, AlternatingGroup and "
          "DihedralGroup (n >= 3): directed graph on 1..|G| (positions in GroupElements[group]) with an edge "
          "i -> j when g_j = PermutationProduct[g_i, s] for a generator s (generators as GroupGenerators/the "
          "standard ones, identities skipped; edges by element, then generator). Wolfram's generator styling is not "
          "reproduced; groups above 5000 elements raise ValueError. Check: exact (tests: |G| vertices, out-degree = "
          "number of generators, strongly connected, CayleyGraph[CyclicGroup[n]] = directed n-cycle).")
def _cayley(group: Expr) -> Expr:
    from . import wl_combinat as C
    if not C._is_group(group):
        if isinstance(group, Symbol) or isinstance(group, Expr):
            raise Unevaluated()
        raise TypeError("CayleyGraph: expected a group")
    gens, deg = C._group_gens(group, "CayleyGraph")
    elems = C._group_elements(group, "CayleyGraph")
    if len(elems) > 5000:
        raise ValueError("CayleyGraph: the group has more than 5000 elements")
    pos = {e.key(): i + 1 for i, e in enumerate(elems)}
    gl, seen = [], set()
    for s in gens:
        if s.m and s.key() not in seen:
            seen.add(s.key())
            gl.append(s)
    es = [_de((pos[e.key()], pos[e.then(s).key()])) for e in elems for s in gl]
    return _new(list(range(1, len(elems) + 1)), es)
