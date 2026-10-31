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
    items = [_norm_edge(it, "HighlightGraph") if _is_edge(it) else it
             for it in (list(spec.args) if is_list(spec) else [spec])]
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


# ═══════════════════════════════════════ measures ═══════════════════════════════════════

def _connected_or_raise(gr, fname):
    if gr.n == 0:
        raise ValueError(f"{fname}: the graph has no vertices")
    if len(_components(gr)) != 1:
        raise NotImplementedError(f"{fname}: disconnected graphs are not supported (Wolfram's per-component "
                                  "convention is not reproduced)")


@fn("EccentricityCentrality", area=AREA, status="added",
    notes="EccentricityCentrality[g] = 1/VertexEccentricity for each vertex in VertexList order, exact for unweighted "
          "graphs (EdgeWeight distances when present); disconnected (not strongly connected) graphs raise "
          "NotImplementedError, a single vertex (eccentricity 0) raises ValueError. Check: exact (tests: "
          "1/VertexEccentricity identity and path/cycle values).")
def _eccc(g: Expr) -> ListOf[Real]:
    gr = _g(g, "EccentricityCentrality")
    gr.require_pure("EccentricityCentrality")
    _connected_or_raise(gr, "EccentricityCentrality")
    e = gr.eccentricities()
    if any(x == 0 for x in e):
        raise ValueError("EccentricityCentrality: a vertex has eccentricity 0")
    return L(*(R(1 / sp.sympify(x)) for x in e))


def _edge_between(gr):
    """Brandes' algorithm for edge betweenness (unweighted), exact fractions, per edge index."""
    n = gr.n
    adj = gr.out_adj()
    eb = [Fraction(0)] * len(gr.edges)
    for s in range(n):
        stack, pred = [], [[] for _ in range(n)]
        sigma = [0] * n
        sigma[s] = 1
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            stack.append(u)
            for w, _, k in adj[u]:
                if w == u:
                    continue
                if w not in dist:
                    dist[w] = dist[u] + 1
                    q.append(w)
                if dist[w] == dist[u] + 1:
                    sigma[w] += sigma[u]
                    pred[w].append((u, k))
        delta = [Fraction(0)] * n
        while stack:
            w = stack.pop()
            for u, k in pred[w]:
                c = Fraction(sigma[u], sigma[w]) * (1 + delta[w])
                eb[k] += c
                delta[u] += c
    if not gr.directed:
        eb = [x / 2 for x in eb]
    return eb


@fn("EdgeBetweennessCentrality", area=AREA, status="added",
    notes="EdgeBetweennessCentrality[g]: for each edge in EdgeList order, the number of shortest paths through it "
          "summed over vertex pairs (each path split evenly among equal-length alternatives; unordered pairs for "
          "undirected graphs, ordered for directed), machine reals as BetweennessCentrality; unweighted, simple "
          "graphs (multigraphs and EdgeWeight raise NotImplementedError). Check: reference (tests compare with "
          "networkx.edge_betweenness_centrality(normalized=False)) and the identity that the total equals the sum "
          "of all pairwise distances.")
def _ebc(g: Expr) -> ListOf[Real]:
    gr = _g(g, "EdgeBetweennessCentrality")
    gr.require_pure("EdgeBetweennessCentrality")
    if gr.multi():
        raise NotImplementedError("EdgeBetweennessCentrality: multigraphs are not supported")
    if gr.weights is not None:
        raise NotImplementedError("EdgeBetweennessCentrality: edge weights are not supported")
    return L(*(float(x) for x in _edge_between(gr)))


@fn("GraphAssortativity", area=AREA, status="added",
    notes="GraphAssortativity[g]: Newman's degree assortativity coefficient (the Pearson correlation of the degrees "
          "at the two ends of each edge, each edge counted in both directions) of an undirected simple graph, as a "
          "machine real; regular graphs (zero variance) raise ValueError; directed graphs and the property/partition "
          "forms raise NotImplementedError. Check: reference (networkx.degree_assortativity_coefficient in tests; "
          "stars give -1).")
def _gassort(g: Expr, *rest: Expr) -> Real:
    if rest:
        raise NotImplementedError("GraphAssortativity: property and partition forms are not supported")
    gr = _g(g, "GraphAssortativity")
    _need_simple_undirected(gr, "GraphAssortativity")
    deg, _, _ = _degree_lists(gr)
    xs, ys = [], []
    for i, j, _ in gr.edges:
        xs += [deg[i], deg[j]]
        ys += [deg[j], deg[i]]
    if not xs:
        raise ValueError("GraphAssortativity: the graph has no edges")
    m = len(xs)
    mx = Fraction(sum(xs), m)
    num = sum((Fraction(a) - mx) * (Fraction(b) - mx) for a, b in zip(xs, ys))
    den = sum((Fraction(a) - mx) ** 2 for a in xs)
    if den == 0:
        raise ValueError("GraphAssortativity: every edge joins vertices of equal degree, so the coefficient is "
                         "undefined (0/0)")
    return float(num / den)


@fn("GraphHub", area=AREA, status="added",
    notes="GraphHub[g]: the vertices of maximum degree in the underlying simple graph (directions, self-loops and "
          "multiple edges ignored), in VertexList order; GraphHub[g, In] / GraphHub[g, Out] use VertexInDegree / "
          "VertexOutDegree. Check: exact (tests: star centre, regular graphs give every vertex).")
def _ghub(g: Expr, kind: Expr = None) -> ListOf[Expr]:
    gr = _g(g, "GraphHub")
    if gr.n == 0:
        return L()
    if kind is None:
        d = [len(s) for s in _simple_nb(gr)]
    elif isinstance(kind, Symbol) and kind.name in ("In", "Out"):
        _, ind, outd = _degree_lists(gr)
        d = ind if kind.name == "In" else outd
    else:
        raise ValueError("GraphHub: the second argument must be In or Out")
    top = max(d)
    return _verts(gr, [i for i in range(gr.n) if d[i] == top])


def _mean_distance(gr):
    if gr.n < 2:
        raise ValueError("MeanGraphDistance: the graph needs at least two vertices")
    tot = 0
    for i in range(gr.n):
        d, _ = gr.sssp(i)
        if len(d) < gr.n:
            return INF
        tot += sum(v for j, v in d.items() if j != i)
    return sp.nsimplify(tot) / (gr.n * (gr.n - 1)) if not isinstance(tot, sp.Float) else tot / (gr.n * (gr.n - 1))


@fn("MeanGraphDistance", area=AREA, status="added",
    notes="MeanGraphDistance[g]: the mean of GraphDistance over all ordered pairs of distinct vertices (exact for "
          "unweighted and exact weights; Infinity when some vertex cannot reach another); fewer than two vertices "
          "raise ValueError; mixed graphs raise. Check: exact (tests: Mean[Flatten[GraphDistanceMatrix]] identities).")
def _mgd(g: Expr) -> Real:
    gr = _g(g, "MeanGraphDistance")
    gr.require_pure("MeanGraphDistance")
    return R(_mean_distance(gr))


@fn("GraphLinkEfficiency", area=AREA, status="added",
    notes="GraphLinkEfficiency[g] = 1 - MeanGraphDistance[s]/EdgeCount[s] for the underlying simple graph s of an "
          "undirected graph (Wolfram's documented formula; self-loops and multiple edges ignored), exact; -Infinity "
          "for disconnected graphs; directed graphs and graphs without edges raise NotImplementedError/ValueError. "
          "Check: exact (tests: the documented identity, a single edge gives 0).")
def _glinkeff(g: Expr) -> Real:
    gr = _g(g, "GraphLinkEfficiency")
    _need_undirected(gr, "GraphLinkEfficiency")
    s = graph_of(_simpleg(g), "GraphLinkEfficiency")
    s = WG(s.vertices, s.edge_exprs, {}, "GraphLinkEfficiency")
    if not s.edges:
        raise ValueError("GraphLinkEfficiency: the graph has no edges")
    md = _mean_distance(s)
    if md == INF:
        return R(-INF)
    return R(1 - md / len(s.edges))


@fn("GraphReciprocity", area=AREA, status="added",
    notes="GraphReciprocity[g] for a directed graph: the fraction of edges u -> v (u != v) whose reverse v -> u is "
          "also an edge, exact (self-loops ignored, parallel edges merged); undirected and mixed graphs raise "
          "NotImplementedError, graphs without non-loop edges raise ValueError. Check: exact (tests: a 2-cycle "
          "gives 1, a DAG 0, hand-counted mixed cases).")
def _grecip(g: Expr) -> Rational:
    gr = _g(g, "GraphReciprocity")
    gr.require_pure("GraphReciprocity")
    if gr.edges and not gr.directed:
        raise NotImplementedError("GraphReciprocity: undirected graphs are not supported")
    es = {(i, j) for i, j, _ in gr.edges if i != j}
    if not es:
        raise ValueError("GraphReciprocity: the graph has no edges between distinct vertices")
    return R(sp.Rational(sum(1 for i, j in es if (j, i) in es), len(es)))


@fn("GraphTriangleCount", area=AREA, status="added",
    notes="GraphTriangleCount[g]: the number of triangles (3-cycles) of the underlying simple graph of an undirected "
          "graph; directed graphs raise NotImplementedError. Check: exact (tests: Binomial[n, 3] for K_n, "
          "trace(A^3)/6).")
def _gtri(g: Expr) -> NonNegInteger:
    gr = _g(g, "GraphTriangleCount")
    _need_undirected(gr, "GraphTriangleCount")
    nb = _simple_nb(gr)
    return sum(1 for i in range(gr.n) for j in nb[i] if j > i for k in nb[i] & nb[j] if k > j)


@fn("KatzCentrality", area=AREA, status="added",
    notes="KatzCentrality[g, alpha] and [g, alpha, beta] (beta a number or a list, default 1): the solution x of "
          "x = alpha A^T x + beta (A the adjacency matrix; undirected edges both ways), exact for exact alpha and "
          "beta, machine reals otherwise; a singular system I - alpha A^T raises ValueError. Check: self-check "
          "(the residual x - alpha A^T x - beta is recomputed; ArithmeticError if it is not zero).")
def _katz(g: Expr, alpha: Real, beta: Real | ListOf[Real] = 1) -> ListOf[Real]:
    gr = _g(g, "KatzCentrality")
    gr.require_pure("KatzCentrality")
    a = _real(alpha, "KatzCentrality", "alpha")
    n = gr.n
    if n == 0:
        return L()
    if is_list(beta):
        if len(beta.args) != n:
            raise ValueError("KatzCentrality: beta must have one value per vertex")
        bv = [_real(b, "KatzCentrality", "beta") for b in beta.args]
    else:
        bv = [_real(beta, "KatzCentrality", "beta")] * n
    A = sp.zeros(n, n)
    for i, j, d in gr.edges:
        A[i, j] += 1
        if not d and i != j:
            A[j, i] += 1
    inexact = any(isinstance(x, sp.Float) for x in [a] + bv)
    M = sp.eye(n) - a * A.T
    b = sp.Matrix(bv)
    if inexact:
        import numpy as np
        Mn = np.array(M.evalf(), dtype=float)
        if abs(np.linalg.det(Mn)) < 1e-14:
            raise ValueError("KatzCentrality: I - alpha A^T is singular for this alpha")
        x = np.linalg.solve(Mn, np.array(b.evalf(), dtype=float).ravel())
        res = Mn @ x - np.array(b.evalf(), dtype=float).ravel()
        if np.max(np.abs(res)) > 1e-8 * max(1.0, np.max(np.abs(x))):
            raise ArithmeticError("KatzCentrality: result failed verification")
        return L(*(float(v) for v in x))
    if M.det() == 0:
        raise ValueError("KatzCentrality: I - alpha A^T is singular for this alpha")
    x = M.LUsolve(b)
    if any(sp.simplify(v) != 0 for v in (M * x - b)):
        raise ArithmeticError("KatzCentrality: result failed verification")
    return L(*(R(sp.nsimplify(sp.simplify(v))) for v in x))


@fn("MeanNeighborDegree", area=AREA, status="added",
    notes="MeanNeighborDegree[g]: for each vertex of an undirected simple graph, the mean degree of its neighbours, "
          "exact, in VertexList order; directed graphs (and the In/Out forms) and graphs with isolated vertices "
          "(whose value Wolfram defines by convention) raise NotImplementedError. Check: reference "
          "(networkx.average_neighbor_degree in tests) and the regular-graph identity.")
def _mnd(g: Expr, *rest: Expr) -> ListOf[Rational]:
    if rest:
        raise NotImplementedError("MeanNeighborDegree: the In/Out forms are not supported")
    gr = _g(g, "MeanNeighborDegree")
    _need_simple_undirected(gr, "MeanNeighborDegree")
    nb = _simple_nb(gr)
    if any(not s for s in nb):
        raise NotImplementedError("MeanNeighborDegree: graphs with isolated vertices are not supported")
    return L(*(R(sp.Rational(sum(len(nb[j]) for j in s), len(s))) for s in nb))


def _sim(name, f, notes):
    @fn(name, area=AREA, status="added", notes=notes)
    def impl(g: Expr, u: Expr, v: Expr) -> Real:
        gr = _g(g, name)
        _need_simple_undirected(gr, name)
        nb = _simple_nb(gr)
        a, b = gr.vidx(u, name), gr.vidx(v, name)
        return R(f(nb[a], nb[b], gr.n, name))
    return impl


def _jac(x, y, n, name):
    if not (x | y):
        raise ValueError(f"{name}: both vertices are isolated, so the similarity is undefined")
    return sp.Rational(len(x & y), len(x | y))


def _dice(x, y, n, name):
    if not (x or y):
        raise ValueError(f"{name}: both vertices are isolated, so the similarity is undefined")
    return sp.Rational(2 * len(x & y), len(x) + len(y))


def _cos(x, y, n, name):
    if not x or not y:
        raise ValueError(f"{name}: an isolated vertex has no defined cosine similarity")
    return sp.Integer(len(x & y)) / sp.sqrt(len(x) * len(y))


def _corr(x, y, n, name):
    kx, ky, c = len(x), len(y), len(x & y)
    vx, vy = n * kx - kx * kx, n * ky - ky * ky
    if vx == 0 or vy == 0:
        raise ValueError(f"{name}: a constant adjacency row has no defined correlation")
    return sp.Integer(n * c - kx * ky) / sp.sqrt(vx * vy)


_sim("VertexJaccardSimilarity", _jac,
     "VertexJaccardSimilarity[g, u, v] = |N(u) n N(v)| / |N(u) u N(v)| with N the open neighbourhoods of an "
     "undirected simple graph, exact; two isolated vertices raise ValueError; directed and multigraphs raise "
     "NotImplementedError. Check: exact (tests: hand-counted neighbourhoods, symmetry).")
_sim("VertexDiceSimilarity", _dice,
     "VertexDiceSimilarity[g, u, v] = 2 |N(u) n N(v)| / (|N(u)| + |N(v)|), exact, undirected simple graphs; two "
     "isolated vertices raise ValueError. Check: exact (tests: Dice = 2J/(1 + J) with the Jaccard value).")
_sim("VertexCosineSimilarity", _cos,
     "VertexCosineSimilarity[g, u, v] = |N(u) n N(v)| / Sqrt[|N(u)| |N(v)|], exact, undirected simple graphs; an "
     "isolated vertex raises ValueError. Check: exact (tests: equals the cosine of the adjacency rows).")
_sim("VertexCorrelationSimilarity", _corr,
     "VertexCorrelationSimilarity[g, u, v]: the Pearson correlation of the adjacency-matrix rows of u and v, exact, "
     "undirected simple graphs; a row that is all 0 or all 1 raises ValueError. Check: exact (tests compare with "
     "Correlation of the rows of AdjacencyMatrix).")


def _local_vconn(gr, s, t, caps=None):
    """Maximum number of internally vertex-disjoint s-t paths (vertex splitting; caps per vertex or 1)."""
    n = gr.n
    vs = list(range(2 * n))
    es, cap = [], []
    for v in range(n):
        if v in (s, t):
            c = 10 ** 9
        else:
            c = 1 if caps is None else caps[v]
        es.append(Expr("DirectedEdge", 2 * v, 2 * v + 1))
        cap.append(c)
    for i, j, d in gr.edges:
        if i == j:
            continue
        es.append(Expr("DirectedEdge", 2 * i + 1, 2 * j))
        cap.append(10 ** 9)
        if not d:
            es.append(Expr("DirectedEdge", 2 * j + 1, 2 * i))
            cap.append(10 ** 9)
    H = WG(vs, es, {}, "VertexConnectivity")
    val, reach = _maxflow(H, 2 * s + 1, 2 * t, cap)
    return val, reach


def _adjacent(gr, s, t):
    return any((i, j) == (s, t) or (not d and (j, i) == (s, t)) for i, j, d in gr.edges)


def _vconn(gr, fname):
    gr.require_pure(fname)
    n = gr.n
    if n <= 1:
        return 0
    if len(_weak_components(gr)) > 1:
        return 0
    best = None
    for s in range(n):
        for t in range(n):
            if s == t or (not gr.directed and t < s) or _adjacent(gr, s, t):
                continue
            v, _ = _local_vconn(gr, s, t)
            best = v if best is None else min(best, v)
    return n - 1 if best is None else best


@fn("VertexConnectivity", area=AREA, status="added",
    notes="VertexConnectivity[g]: the minimum number of vertices whose removal disconnects g (n - 1 for complete "
          "graphs, 0 when disconnected or n <= 1), the minimum over non-adjacent pairs of the s-t connectivity "
          "(directed graphs: ordered pairs); VertexConnectivity[g, s, t]: the maximum number of internally "
          "vertex-disjoint s-t paths (Menger, via maximum flow with vertex splitting) for non-adjacent s, t; adjacent "
          "s, t raise NotImplementedError. Check: exact (tests: Menger duality with FindVertexCut, "
          "HararyGraph[k, n] has connectivity k, Petersen graph 3).")
def _vconnectivity(g: Expr, s: Expr = None, t: Expr = None) -> NonNegInteger:
    gr = _g(g, "VertexConnectivity")
    if s is None:
        return _vconn(gr, "VertexConnectivity")
    if t is None:
        raise ValueError("VertexConnectivity: give both s and t")
    gr.require_pure("VertexConnectivity")
    a, b = gr.vidx(s, "VertexConnectivity"), gr.vidx(t, "VertexConnectivity")
    if a == b:
        raise ValueError("VertexConnectivity: s and t must differ")
    if _adjacent(gr, a, b):
        raise NotImplementedError("VertexConnectivity: adjacent s and t are not supported")
    return _local_vconn(gr, a, b)[0]


def _econn_val(gr, fname):
    gr.require_pure(fname)
    if gr.n <= 1 or len(_weak_components(gr)) > 1:
        return 0
    if not gr.directed:
        w = [[0] * gr.n for _ in range(gr.n)]
        for i, j, _ in gr.edges:
            if i != j:
                w[i][j] += 1
                w[j][i] += 1
        return _stoer_wagner(gr.n, w)[0]
    ones = [sp.Integer(1)] * len(gr.edges)
    return min(min(_maxflow(gr, 0, t, ones)[0], _maxflow(gr, t, 0, ones)[0]) for t in range(1, gr.n))


@fn("KEdgeConnectedGraphQ", area=AREA, status="added",
    notes="KEdgeConnectedGraphQ[g, k]: True when at least k edges must be removed to disconnect g (EdgeConnectivity "
          ">= k; strongly for directed graphs, computed by maximum flows); k = 0 is always True; non-graphs give "
          "False. Check: exact (tests: cycles are 2- but not 3-edge-connected, hypercubes Q_n are n-edge-connected).")
def _kecq(g: Expr, k: NonNegInteger) -> Boolean:
    k = _int(k, "KEdgeConnectedGraphQ", 0, "k")
    if isinstance(g, Symbol):
        raise Unevaluated()
    if not is_graph_like(g):
        return False
    gr = _g(g, "KEdgeConnectedGraphQ")
    return k == 0 or (gr.n > 0 and _econn_val(gr, "KEdgeConnectedGraphQ") >= k)


@fn("KVertexConnectedGraphQ", area=AREA, status="added",
    notes="KVertexConnectedGraphQ[g, k]: True when VertexConnectivity[g] >= k (k = 0 always True); non-graphs give "
          "False. Check: exact (tests: HararyGraph[k, n] is k- but not (k+1)-connected).")
def _kvcq(g: Expr, k: NonNegInteger) -> Boolean:
    k = _int(k, "KVertexConnectedGraphQ", 0, "k")
    if isinstance(g, Symbol):
        raise Unevaluated()
    if not is_graph_like(g):
        return False
    gr = _g(g, "KVertexConnectedGraphQ")
    return k == 0 or _vconn(gr, "KVertexConnectedGraphQ") >= k


@fn("PathGraphQ", area=AREA, status="added",
    notes="PathGraphQ[g]: True for a simple undirected path (connected, n - 1 edges, degrees at most 2) or a directed "
          "path (each vertex at most one in- and one out-edge, connected, n - 1 edges); a single vertex is a path; "
          "mixed graphs, the empty graph and non-graphs give False. Check: exact (tests: PathGraph, cycles, stars).")
def _pathq(g: Expr) -> Boolean:
    if isinstance(g, Symbol):
        raise Unevaluated()
    if not is_graph_like(g):
        return False
    try:
        gr = graph_of(g, "PathGraphQ")
    except ValueError:
        return False
    if gr.mixed or gr.n == 0 or gr.loops() or gr.multi():
        return False
    if len(gr.edges) != gr.n - 1 or len(_weak_components(gr)) != 1:
        return False
    deg, ind, outd = _degree_lists(gr)
    if gr.directed:
        return all(ind[i] <= 1 and outd[i] <= 1 for i in range(gr.n))
    return all(d <= 2 for d in deg)


def _edge_set_q(name, check, notes):
    @fn(name, area=AREA, status="added", notes=notes)
    def impl(g: Expr, elist: ListOf[Expr]) -> Boolean:
        gr = _g(g, name)
        if not is_list(elist):
            raise TypeError(f"{name}: the second argument must be a list of edges")
        ks = _edge_indices(gr, list(elist.args), name)
        if ks is None:
            return False
        return check(gr, ks)
    return impl


def _cover_ok(gr, ks):
    hit = set()
    for k in ks:
        i, j, _ = gr.edges[k]
        hit |= {i, j}
    return len(hit) == gr.n


def _indep_ok(gr, ks):
    used = set()
    for k in ks:
        i, j, _ = gr.edges[k]
        if i in used or j in used or i == j:
            return False
        used |= {i, j}
    return True


_edge_set_q("EdgeCoverQ", _cover_ok,
            "EdgeCoverQ[g, {e...}]: True when the listed edges are edges of g and every vertex is an endpoint of one "
            "of them (False when an edge is not in g). Check: exact (tests: FindEdgeCover results pass, deleting an "
            "edge fails).")
_edge_set_q("IndependentEdgeSetQ", _indep_ok,
            "IndependentEdgeSetQ[g, {e...}]: True when the listed edges are edges of g and no two share a vertex "
            "(self-loops are never independent). Check: exact (tests: FindIndependentEdgeSet results pass, adjacent "
            "edges fail).")


@fn("KCoreComponents", area=AREA, status="added",
    notes="KCoreComponents[g, k]: the connected components of the k-core of the underlying simple graph (vertices of "
          "degree < k removed repeatedly), largest first with vertices in VertexList order; the In/Out forms raise "
          "NotImplementedError. Check: exact (tests: every component vertex has core degree >= k; the Petersen graph "
          "is its own 3-core).")
def _kcore(g: Expr, k: NonNegInteger, *rest: Expr) -> ListOf[ListOf[Expr]]:
    if rest:
        raise NotImplementedError("KCoreComponents: the In/Out forms are not supported")
    gr = _g(g, "KCoreComponents")
    k = _int(k, "KCoreComponents", 0, "k")
    nb = [set(s) for s in _simple_nb(gr)]
    alive = set(range(gr.n))
    changed = True
    while changed:
        changed = False
        for v in list(alive):
            if len(nb[v] & alive) < k:
                alive.discard(v)
                changed = True
    seen, comps = set(), []
    for s in sorted(alive):
        if s in seen:
            continue
        comp, q = [], [s]
        seen.add(s)
        while q:
            u = q.pop()
            comp.append(u)
            for w in nb[u] & alive:
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (-len(c), c[0]))
    return L(*(_verts(gr, c) for c in comps))


# ═══════════════════════════════════════ colouring ═══════════════════════════════════════

_SEARCH_LIMIT = 2_000_000


class _Done(Exception):
    pass


def _min_coloring(nb, n, fname):
    """Minimum proper colouring (DSATUR branch and bound, clique lower bound); colours 0.. in first-use order."""
    if n == 0:
        return []
    lb = max((len(c) for c in _max_cliques([set(s) for s in nb], n)), default=1)
    deg = [len(nb[v]) for v in range(n)]
    color = [-1] * n
    best = {"k": n + 1, "c": None}
    steps = [0]

    def pick():
        key, bv = None, None
        for v in range(n):
            if color[v] < 0:
                kk = (len({color[w] for w in nb[v] if color[w] >= 0}), deg[v], -v)
                if key is None or kk > key:
                    key, bv = kk, v
        return bv

    def rec(used):
        steps[0] += 1
        if steps[0] > _SEARCH_LIMIT:
            raise NotImplementedError(f"{fname}: the graph is too large for the exact colouring search")
        v = pick()
        if v is None:
            best["k"], best["c"] = used, color[:]
            if used <= lb:
                raise _Done()
            return
        forb = {color[w] for w in nb[v] if color[w] >= 0}
        for c in range(used + 1):
            if c in forb or max(used, c + 1) >= best["k"]:
                continue
            color[v] = c
            rec(max(used, c + 1))
            color[v] = -1
    try:
        rec(0)
    except _Done:
        pass
    cols, remap = best["c"], {}
    out = []
    for c in cols:
        remap.setdefault(c, len(remap))
        out.append(remap[c])
    for v in range(n):
        if any(out[v] == out[w] for w in nb[v]):
            raise ArithmeticError(f"{fname}: internal check failed (the colouring is not proper)")
    return out


def _vertex_nb(gr, fname):
    gr.require_pure(fname)
    if gr.loops():
        raise ValueError(f"{fname}: a graph with a self-loop has no proper vertex colouring")
    return _simple_nb(gr)


def _edge_nb(gr, fname):
    gr.require_pure(fname)
    if gr.loops():
        raise ValueError(f"{fname}: a graph with a self-loop has no proper edge colouring")
    m = len(gr.edges)
    nb = [set() for _ in range(m)]
    for a in range(m):
        for b in range(a + 1, m):
            if {gr.edges[a][0], gr.edges[a][1]} & {gr.edges[b][0], gr.edges[b][1]}:
                nb[a].add(b)
                nb[b].add(a)
    return nb, m


def _apply_colors(cols, spec, fname):
    if spec is None:
        return L(*(c + 1 for c in cols))
    if not is_list(spec):
        raise TypeError(f"{fname}: the colours must be given as a list")
    k = max(cols) + 1 if cols else 0
    if len(spec.args) < k:
        raise ValueError(f"{fname}: {k} colours are needed but only {len(spec.args)} were given")
    return L(*(spec.args[c] for c in cols))


@fn("FindVertexColoring", area=AREA, status="added",
    notes="FindVertexColoring[g]: a proper colouring with the minimum number of colours, as colours 1, 2, ... in "
          "VertexList order (numbered by first use); FindVertexColoring[g, {c1, c2, ...}] uses the given colours and "
          "raises ValueError when fewer than the chromatic number are given. Exact branch and bound (DSATUR order, "
          "clique bound; NotImplementedError past 2*10^6 search nodes); edge directions are ignored; self-loops raise "
          "ValueError. A valid optimal colouring may differ from Wolfram's. Check: self-check (the colouring is "
          "verified proper; tests compare the count with known chromatic numbers).")
def _fvcol(g: Expr, colors: ListOf[Expr] = None) -> ListOf[Expr]:
    gr = _g(g, "FindVertexColoring")
    return _apply_colors(_min_coloring(_vertex_nb(gr, "FindVertexColoring"), gr.n, "FindVertexColoring"), colors,
                         "FindVertexColoring")


@fn("VertexChromaticNumber", area=AREA, status="added",
    notes="VertexChromaticNumber[g]: the minimum number of colours in a proper vertex colouring (0 for the empty "
          "graph); exact search as in FindVertexColoring; self-loops raise ValueError. Check: self-check (colouring "
          "verified; tests: Petersen 3, odd cycles 3, K_n n, bipartite 2).")
def _vchrom(g: Expr) -> NonNegInteger:
    gr = _g(g, "VertexChromaticNumber")
    cols = _min_coloring(_vertex_nb(gr, "VertexChromaticNumber"), gr.n, "VertexChromaticNumber")
    return max(cols) + 1 if cols else 0


@fn("FindEdgeColoring", area=AREA, status="added",
    notes="FindEdgeColoring[g]: a proper edge colouring with the minimum number of colours, as colours 1, 2, ... in "
          "EdgeList order; FindEdgeColoring[g, {c1, ...}] uses the given colours. Exact colouring of the line graph "
          "(multi-edges allowed, edge directions ignored, self-loops raise ValueError). Check: self-check (colouring "
          "verified; tests: Petersen needs 4 = Delta + 1, bipartite graphs need Delta).")
def _fecol(g: Expr, colors: ListOf[Expr] = None) -> ListOf[Expr]:
    gr = _g(g, "FindEdgeColoring")
    nb, m = _edge_nb(gr, "FindEdgeColoring")
    return _apply_colors(_min_coloring(nb, m, "FindEdgeColoring"), colors, "FindEdgeColoring")


@fn("EdgeChromaticNumber", area=AREA, status="added",
    notes="EdgeChromaticNumber[g]: the chromatic index (minimum colours in a proper edge colouring; 0 without edges); "
          "exact as in FindEdgeColoring. Check: self-check (tests: Vizing's bound Delta <= chi' <= Delta + 1, "
          "Petersen 4, K4 3, K5 5).")
def _echrom(g: Expr) -> NonNegInteger:
    gr = _g(g, "EdgeChromaticNumber")
    nb, m = _edge_nb(gr, "EdgeChromaticNumber")
    cols = _min_coloring(nb, m, "EdgeChromaticNumber")
    return max(cols) + 1 if cols else 0


# ═══════════════════════════════════════ matchings and covers ═══════════════════════════════════════

def _matching(gr, fname):
    """Edge indices of a maximum matching (NetworkX blossom algorithm), in EdgeList order."""
    nx = _nx()
    gr.require_pure(fname)
    X = nx.Graph()
    X.add_nodes_from(range(gr.n))
    first = {}
    for k, (i, j, _) in enumerate(gr.edges):
        if i != j:
            key = (min(i, j), max(i, j))
            if key not in first:
                first[key] = k
                X.add_edge(i, j)
    pairs = nx.max_weight_matching(X, maxcardinality=True)
    ks = sorted(first[(min(a, b), max(a, b))] for a, b in pairs)
    if not _indep_ok(gr, ks):
        raise ArithmeticError(f"{fname}: internal check failed (the matching is not independent)")
    return ks


@fn("FindIndependentEdgeSet", area=AREA, status="added",
    notes="FindIndependentEdgeSet[g]: a maximum matching (Edmonds blossom algorithm via NetworkX, edge directions "
          "ignored, self-loops never used), as edges in EdgeList order; NotImplementedError without NetworkX. A valid "
          "maximum matching may differ from Wolfram's. Check: self-check (independence verified; tests compare sizes "
          "with known matching numbers and IndependentEdgeSetQ).")
def _fies(g: Expr) -> ListOf[Expr]:
    gr = _g(g, "FindIndependentEdgeSet")
    return L(*(gr.edge_exprs[k] for k in _matching(gr, "FindIndependentEdgeSet")))


@fn("FindEdgeCover", area=AREA, status="added",
    notes="FindEdgeCover[g]: a minimum edge cover: a maximum matching plus, for each unmatched vertex, its first "
          "incident edge in EdgeList order (Gallai: size n - matching number); a vertex without incident edges "
          "raises ValueError; edges in EdgeList order; needs NetworkX. Check: self-check (EdgeCoverQ verified and size "
          "n - matching number; tests: paths, stars, Petersen).")
def _fecover(g: Expr) -> ListOf[Expr]:
    gr = _g(g, "FindEdgeCover")
    ks = set(_matching(gr, "FindEdgeCover"))
    covered = set()
    for k in ks:
        covered |= {gr.edges[k][0], gr.edges[k][1]}
    for v in range(gr.n):
        if v in covered:
            continue
        k = next((k for k, (i, j, _) in enumerate(gr.edges) if v in (i, j)), None)
        if k is None:
            raise ValueError(f"FindEdgeCover: vertex {gr.vertices[v]} has no incident edge, so there is no edge cover")
        ks.add(k)
        covered |= {gr.edges[k][0], gr.edges[k][1]}
    if not _cover_ok(gr, ks):
        raise ArithmeticError("FindEdgeCover: internal check failed (not an edge cover)")
    return L(*(gr.edge_exprs[k] for k in sorted(ks)))


# ═══════════════════════════════════════ k-cliques, clans, clubs, plexes ═══════════════════════════════════════

def _bfs_dist(nb, s, allowed=None):
    d = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        for w in nb[u]:
            if w not in d and (allowed is None or w in allowed):
                d[w] = d[u] + 1
                q.append(w)
    return d


def _induced_diam_ok(nb, sset, k):
    for s in sset:
        d = _bfs_dist(nb, s, sset)
        if len(d) < len(sset) or max(d.values()) > k:
            return False
    return True


def _size_spec(nspec, n, fname):
    if nspec is None:
        return 1, n
    if K.is_int(nspec) or isinstance(nspec, (bool, str)) or is_numeric_value(nspec):
        return 1, _int(nspec, fname, 0, "the size bound")
    if is_list(nspec) and len(nspec.args) == 1:
        a = _int(nspec.args[0], fname, 0, "the size")
        return a, a
    if is_list(nspec) and len(nspec.args) == 2:
        a, b = _int(nspec.args[0], fname, 0, "the size"), _int(nspec.args[1], fname, 0, "the size")
        return a, b
    if isinstance(nspec, Symbol) and nspec.name == "Infinity":
        return 1, n
    raise TypeError(f"{fname}: the size specification must be n, {{n}} or {{nmin, nmax}}, not {nspec!r}")


def _count_spec(s, fname):
    if s is None:
        return 1
    if isinstance(s, Symbol) and s.name in ("All", "Infinity"):
        return None
    return _int(s, fname, 0, "the number of sets")


def _kfinder(name, kind, notes):
    @fn(name, area=AREA, status="added", notes=notes)
    def impl(g: Expr, k: PosInteger, nspec: Expr = None, s: Expr = None) -> ListOf[ListOf[Expr]]:
        must = None
        if is_list(g) and len(g.args) == 2 and isinstance(g.args[0], Expr) and g.args[0].head == "Graph":
            g, must = g.args
        gr = _g(g, name)
        _need_undirected(gr, name)
        k = _int(k, name, 1, "k")
        n = gr.n
        mv = gr.vidx(must, name) if must is not None else None
        lo, hi = _size_spec(nspec, n, name)
        cnt = _count_spec(s, name)
        nb = _simple_nb(gr)
        dist = [_bfs_dist(nb, v) for v in range(n)]

        def kclique(c):
            return all(dist[a].get(b, INF) <= k for a, b in itertools.combinations(c, 2))

        def valid(c):
            if kind == "clique":
                return kclique(c)
            if kind == "plex":
                cs = set(c)
                return all(len(nb[v] & cs) >= len(c) - k for v in c)
            if kind == "club":
                return _induced_diam_ok(nb, set(c), k)
            # clan: a maximal k-clique whose induced subgraph has diameter <= k
            if not kclique(c) or not _induced_diam_ok(nb, set(c), k):
                return False
            return not any(all(dist[v].get(u, INF) <= k for u in c) for v in range(n) if v not in c)
        found = []
        if cnt == 0:
            return L()
        if kind == "clique" and nspec is None and s is None:
            pw = [set(u for u in range(n) if u != v and dist[v].get(u, INF) <= k) for v in range(n)]
            cl = [c for c in _max_cliques(pw, n) if mv is None or mv in c] if n else []
            if cl:
                found = [sorted(min(cl, key=lambda c: (-len(c), c)))]
        else:
            steps = 0
            for size in range(min(hi, n), max(lo, 1) - 1, -1):
                for c in itertools.combinations(range(n), size):
                    steps += 1
                    if steps > _SEARCH_LIMIT:
                        raise NotImplementedError(f"{name}: the graph is too large for the exhaustive search")
                    if (mv is None or mv in c) and valid(c):
                        found.append(list(c))
                        if cnt is not None and len(found) >= cnt:
                            break
                if cnt is not None and len(found) >= cnt:
                    break
                if found and nspec is None:
                    break
        return L(*(_verts(gr, c) for c in found))
    return impl


_kfinder("FindKClique", "clique",
         "FindKClique[g, k]: {set} with a largest k-clique (pairwise distances in g at most k), as maximum clique of "
         "the k-th power of the underlying simple graph; FindKClique[g, k, nspec, s] gives at most s vertex sets with "
         "pairwise distance <= k and sizes in nspec (n, {n} or {nmin, nmax}), largest first then by VertexList "
         "positions; FindKClique[{g, v}, ...] keeps sets containing v; undirected graphs only; exhaustive search "
         "limited to 2*10^6 candidate sets. Check: exact (tests: k = 1 is the maximum clique; on paths k-cliques have "
         "k + 1 vertices).")
_kfinder("FindKClan", "clan",
         "FindKClan[g, k, ...]: largest k-clans (maximal k-cliques whose induced subgraph has diameter <= k), same "
         "nspec/count/{g, v} forms and ordering as FindKClique; exhaustive search (limit 2*10^6 candidate sets); "
         "undirected only. Check: exact (tests: every result is a maximal k-clique with induced diameter <= k; the "
         "6-cycle with a pendant classic example).")
_kfinder("FindKClub", "club",
         "FindKClub[g, k, ...]: largest k-clubs (vertex sets whose induced subgraph is connected with diameter <= k), "
         "same nspec/count/{g, v} forms as FindKClique; exhaustive search (limit 2*10^6 candidate sets); undirected "
         "only. Check: exact (tests: induced diameter verified; a star is a 2-club).")
_kfinder("FindKPlex", "plex",
         "FindKPlex[g, k, ...]: largest k-plexes (every member adjacent to at least |S| - k members, so 1-plexes are "
         "cliques), same nspec/count/{g, v} forms as FindKClique; exhaustive search (limit 2*10^6 candidate sets); "
         "undirected only. Check: exact (tests: k = 1 equals the clique number; C4 is a 2-plex).")


# ═══════════════════════════════════════ cuts, flows and paths ═══════════════════════════════════════

class _Flow:
    """Edmonds-Karp maximum flow on explicit arcs (u, v, capacity), keeping per-arc flow for path extraction."""

    def __init__(self, n):
        self.n = n
        self.to, self.cap, self.cost, self.adj = [], [], [], [[] for _ in range(n)]

    def arc(self, u, v, c, w=0):
        k = len(self.to)
        self.to += [v, u]
        self.cap += [c, 0]
        self.cost += [w, -w]
        self.adj[u].append(k)
        self.adj[v].append(k + 1)
        return k

    def run(self, s, t, limit=None, fname="flow"):
        val = 0
        self.orig = list(self.cap)
        while limit is None or val < limit:
            prev = {s: None}
            q = deque([s])
            while q and t not in prev:
                u = q.popleft()
                for k in self.adj[u]:
                    v = self.to[k]
                    if v not in prev and self.cap[k] > 0:
                        prev[v] = k
                        q.append(v)
            if t not in prev:
                break
            path, v = [], t
            while v != s:
                k = prev[v]
                path.append(k)
                v = self.to[k ^ 1]
            b = min(self.cap[k] for k in path)
            if limit is not None:
                b = min(b, limit - val)
            if b == INF:
                raise ValueError(f"{fname}: the flow is unbounded (a path with infinite capacity)")
            for k in path:
                self.cap[k] -= b
                self.cap[k ^ 1] += b
            val += b
        return val

    def reach(self, s):
        seen, q = {s}, [s]
        while q:
            u = q.pop()
            for k in self.adj[u]:
                if self.cap[k] > 0 and self.to[k] not in seen:
                    seen.add(self.to[k])
                    q.append(self.to[k])
        return seen

    def flow_on(self, k):
        return self.cap[k ^ 1] - self.orig[k ^ 1]

    def min_cost(self, s, t, want, fname):
        """Successive shortest paths (Bellman-Ford on the residual graph); returns (flow, cost)."""
        flow, total = 0, 0
        self.orig = list(self.cap)
        while want is None or flow < want:
            dist, prev = {s: 0}, {s: None}
            for _ in range(self.n):
                changed = False
                for u in list(dist):
                    for k in self.adj[u]:
                        if self.cap[k] > 0:
                            v, nd = self.to[k], dist[u] + self.cost[k]
                            if v not in dist or nd < dist[v]:
                                dist[v], prev[v] = nd, k
                                changed = True
                if not changed:
                    break
            if t not in dist:
                break
            path, v = [], t
            while v != s:
                k = prev[v]
                path.append(k)
                v = self.to[k ^ 1]
            b = min(self.cap[k] for k in path)
            if want is not None:
                b = min(b, want - flow)
            if b == INF:
                raise ValueError(f"{fname}: the maximum flow is unbounded (a path with infinite capacity)")
            for k in path:
                self.cap[k] -= b
                self.cap[k ^ 1] += b
            flow += b
            total += b * dist[t]
        return flow, total


def _edge_flow(gr, caps=None, skip_loops=True):
    F = _Flow(gr.n)
    arcs = []
    for k, (i, j, d) in enumerate(gr.edges):
        if i == j and skip_loops:
            continue
        c = 1 if caps is None else caps[k]
        arcs.append((k, F.arc(i, j, c), None if d else F.arc(j, i, c)))
    return F, arcs


def _split_flow(gr, inner_cap, edge_cap, s=None, t=None, vcaps=None):
    """Vertex-split network: v_in = 2v, v_out = 2v + 1."""
    F = _Flow(2 * gr.n)
    for v in range(gr.n):
        c = vcaps[v] if vcaps is not None else (INF if v in (s, t) else inner_cap)
        F.arc(2 * v, 2 * v + 1, c)
    arcs = []
    for k, (i, j, d) in enumerate(gr.edges):
        if i == j:
            continue
        c = edge_cap if edge_cap is not None else 1
        if callable(edge_cap):
            c = edge_cap(k)
        arcs.append((k, F.arc(2 * i + 1, 2 * j, c), None if d else F.arc(2 * j + 1, 2 * i, c)))
    return F, arcs


def _st_edge_cut(gr, a, b):
    F, arcs = _edge_flow(gr)
    val = F.run(a, b)
    rs = F.reach(a)
    cut = []
    for k, _, back in arcs:
        i, j, d = gr.edges[k]
        if (i in rs and j not in rs) or (back is not None and j in rs and i not in rs):
            cut.append(k)
    if len(cut) != val:
        raise ArithmeticError("FindEdgeCut: internal check failed (cut size differs from the flow value)")
    return cut


@fn("FindEdgeCut", area=AREA, status="added",
    notes="FindEdgeCut[g]: a smallest set of edges whose removal disconnects g (strongly, for directed graphs; {} when "
          "g is already disconnected or has fewer than 2 vertices); FindEdgeCut[g, s, t]: a smallest s-t edge cut "
          "(max-flow min-cut, unit capacities, multi-edges counted). Edges in EdgeList order; EdgeWeight is ignored "
          "(edges are counted). Check: self-check (cut size equals the max-flow value; tests: removing the cut "
          "disconnects, sizes equal EdgeConnectivity).")
def _fecut(g: Expr, s: Expr = None, t: Expr = None) -> ListOf[Expr]:
    gr = _g(g, "FindEdgeCut")
    gr.require_pure("FindEdgeCut")
    if (s is None) != (t is None):
        raise ValueError("FindEdgeCut: give both s and t, or neither")
    if s is not None:
        a, b = gr.vidx(s, "FindEdgeCut"), gr.vidx(t, "FindEdgeCut")
        if a == b:
            raise ValueError("FindEdgeCut: s and t must be different vertices")
        return L(*(gr.edge_exprs[k] for k in sorted(_st_edge_cut(gr, a, b))))
    if gr.n < 2:
        return L()
    comps = _strong_components(gr) if gr.directed else _weak_components(gr)
    if len(comps) > 1:
        return L()
    best = None
    pairs = [(0, v) for v in range(1, gr.n)] + ([(v, 0) for v in range(1, gr.n)] if gr.directed else [])
    for a, b in pairs:
        c = _st_edge_cut(gr, a, b)
        if best is None or len(c) < len(best):
            best = c
    return L(*(gr.edge_exprs[k] for k in sorted(best)))


def _st_vertex_cut(gr, a, b, fname):
    F, _ = _split_flow(gr, 1, INF, a, b)
    val = F.run(2 * a + 1, 2 * b)
    rs = F.reach(2 * a + 1)
    cut = [v for v in range(gr.n) if 2 * v in rs and 2 * v + 1 not in rs]
    if len(cut) != val:
        raise ArithmeticError(f"{fname}: internal check failed (cut size differs from the flow value)")
    return cut


@fn("FindVertexCut", area=AREA, status="added",
    notes="FindVertexCut[g]: a smallest set of vertices whose removal disconnects g (undirected; {} for complete or "
          "already disconnected graphs, minimum over non-adjacent pairs); FindVertexCut[g, s, t]: a smallest s-t "
          "vertex separator (directed allowed; s, t adjacent or equal raise ValueError). Vertices in VertexList order. "
          "Check: self-check (separator size equals the vertex-split max-flow value; tests: removal disconnects, size "
          "equals VertexConnectivity).")
def _fvcut(g: Expr, s: Expr = None, t: Expr = None) -> ListOf[Expr]:
    gr = _g(g, "FindVertexCut")
    gr.require_pure("FindVertexCut")
    if (s is None) != (t is None):
        raise ValueError("FindVertexCut: give both s and t, or neither")
    if s is not None:
        a, b = gr.vidx(s, "FindVertexCut"), gr.vidx(t, "FindVertexCut")
        if a == b:
            raise ValueError("FindVertexCut: s and t must be different vertices")
        if any((i, j) == (a, b) or (not d and (j, i) == (a, b)) for i, j, d in gr.edges):
            raise ValueError("FindVertexCut: s and t are adjacent, so no vertex set separates them")
        return _verts(gr, _st_vertex_cut(gr, a, b, "FindVertexCut"))
    _need_undirected(gr, "FindVertexCut")
    if gr.n < 2 or len(_weak_components(gr)) > 1:
        return L()
    nb = _simple_nb(gr)
    best = None
    for a in range(gr.n):
        for b in range(a + 1, gr.n):
            if b in nb[a]:
                continue
            c = _st_vertex_cut(gr, a, b, "FindVertexCut")
            if best is None or len(c) < len(best):
                best = c
    return L() if best is None else _verts(gr, best)


def _paths_from_flow(F, s, t, arcs_used, nodes_map):
    """Decompose unit flow on arcs into s-t paths (cycles removed)."""
    out_arcs = {}
    for k in arcs_used:
        out_arcs.setdefault(F.to[k ^ 1], []).append(k)
    paths = []
    while out_arcs.get(s):
        path, u, seen = [s], s, {s: 0}
        while u != t:
            k = out_arcs[u].pop()
            u = F.to[k]
            if u in seen:
                del path[seen[u] + 1:]
                seen = {x: i for i, x in enumerate(path)}
            else:
                seen[u] = len(path)
                path.append(u)
        mapped = []
        for x in path:
            y = nodes_map(x)
            if y is not None and (not mapped or mapped[-1] != y):
                mapped.append(y)
        paths.append(mapped)
    paths.sort(key=lambda p: (len(p), p))
    return paths


def _kpaths(k, fname):
    if isinstance(k, Symbol) and k.name in ("Infinity", "All"):
        return None
    return _int(k, fname, 0, "k")


@fn("FindEdgeIndependentPaths", area=AREA, status="added",
    notes="FindEdgeIndependentPaths[g, s, t, k]: at most k pairwise edge-disjoint s-t paths (vertex lists), from a "
          "unit-capacity maximum flow (k = Infinity for as many as possible); flows on both directions of an "
          "undirected edge cancel; shortest paths first; s = t raises ValueError. Check: self-check (the number of "
          "paths equals min(k, max-flow); tests: paths are edge-disjoint, count equals EdgeConnectivity).")
def _feip(g: Expr, s: Expr, t: Expr, k: NonNegInteger | Expr) -> ListOf[ListOf[Expr]]:
    gr = _g(g, "FindEdgeIndependentPaths")
    gr.require_pure("FindEdgeIndependentPaths")
    a, b = gr.vidx(s, "FindEdgeIndependentPaths"), gr.vidx(t, "FindEdgeIndependentPaths")
    if a == b:
        raise ValueError("FindEdgeIndependentPaths: s and t must be different vertices")
    lim = _kpaths(k, "FindEdgeIndependentPaths")
    F, arcs = _edge_flow(gr)
    val = F.run(a, b, lim)
    used = []
    for _, fw, bw in arcs:
        f1 = F.flow_on(fw)
        f2 = F.flow_on(bw) if bw is not None else 0
        if f1 > 0 and f2 > 0:
            continue
        if f1 > 0:
            used.append(fw)
        if f2 > 0:
            used.append(bw)
    paths = _paths_from_flow(F, a, b, used, lambda x: x)
    if len(paths) != val:
        raise ArithmeticError("FindEdgeIndependentPaths: internal check failed (path count differs from the flow)")
    return L(*(_verts(gr, p) for p in paths))


@fn("FindVertexIndependentPaths", area=AREA, status="added",
    notes="FindVertexIndependentPaths[g, s, t, k]: at most k s-t paths sharing no vertex other than s and t (vertex "
          "lists), from a vertex-split unit-capacity maximum flow (k = Infinity for as many as possible); an edge s-t "
          "gives the path {s, t}; shortest first; s = t raises ValueError. Check: self-check (path count equals the "
          "flow value; tests: interiors disjoint, Menger count on the Petersen graph is 3).")
def _fvip(g: Expr, s: Expr, t: Expr, k: NonNegInteger | Expr) -> ListOf[ListOf[Expr]]:
    gr = _g(g, "FindVertexIndependentPaths")
    gr.require_pure("FindVertexIndependentPaths")
    a, b = gr.vidx(s, "FindVertexIndependentPaths"), gr.vidx(t, "FindVertexIndependentPaths")
    if a == b:
        raise ValueError("FindVertexIndependentPaths: s and t must be different vertices")
    lim = _kpaths(k, "FindVertexIndependentPaths")
    F, arcs = _split_flow(gr, 1, 1, a, b)
    val = F.run(2 * a + 1, 2 * b, lim)
    used = [k2 for k2 in range(0, len(F.to), 2) if F.flow_on(k2) > 0]
    paths = _paths_from_flow(F, 2 * a + 1, 2 * b, used, lambda x: x // 2)
    if len(paths) != val:
        raise ArithmeticError("FindVertexIndependentPaths: internal check failed (path count differs from the flow)")
    return L(*(_verts(gr, p) for p in paths))


@fn("FindMaximumCut", area=AREA, status="added",
    notes="FindMaximumCut[g]: {c, {S, T}} with the largest total weight c of edges between S and T (EdgeWeight, "
          "default 1; directions ignored; self-loops never cut), S containing the first vertex, both in VertexList "
          "order; exhaustive over the 2^(n-1) bipartitions, n <= 20 (NotImplementedError beyond). Check: exact "
          "(tests: bipartite graphs cut every edge, K_n cuts floor(n^2/4), C5 cuts 4).")
def _fmaxcut(g: Expr) -> ListOf[Expr]:
    gr = _g(g, "FindMaximumCut")
    gr.require_pure("FindMaximumCut")
    n = gr.n
    if n == 0:
        return L(0, L(L(), L()))
    if n > 20:
        raise NotImplementedError("FindMaximumCut: exhaustive search supports at most 20 vertices")
    es = [(1 << i, 1 << j, gr.weight(k)) for k, (i, j, _) in enumerate(gr.edges) if i != j]
    ints = all(isinstance(w, sp.Integer) for _, _, w in es)
    wv = [(a, b, int(w) if ints else w) for a, b, w in es]
    best, bm = None, 0
    for m in range(1 << (n - 1)):
        mask = m << 1
        c = sum(w for a, b, w in wv if bool(mask & a) != bool(mask & b))
        if best is None or c > best:
            best, bm = c, mask
    S1 = [v for v in range(n) if not bm >> v & 1]
    S2 = [v for v in range(n) if bm >> v & 1]
    return L(R(S(best)), L(_verts(gr, S1), _verts(gr, S2)))


def _vertex_caps(gr, fname):
    vc = gr.opts.get("VertexCapacity")
    if vc is None:
        return None
    if not is_list(vc) or len(vc.args) != gr.n:
        raise ValueError(f"{fname}: VertexCapacity must be a list with one value per vertex")
    out = []
    for x in vc.args:
        v = S(x)
        if not (isinstance(v, sp.Basic) and (v == INF or (v.is_number and v.is_real))) or v < 0:
            raise ValueError(f"{fname}: VertexCapacity values must be non-negative numbers or Infinity (got {x})")
        out.append(v)
    return out


def _has_vcap(args):
    try:
        return len(args) == 3 and is_graph_like(args[0]) and "VertexCapacity" in graph_of(args[0], "FindMaximumFlow").opts
    except Exception:
        return False


def _fmaxflow_vcap(g, s, t):
    gr = graph_of(g, "FindMaximumFlow")
    gr.require_pure("FindMaximumFlow")
    a, b = gr.vidx(s, "FindMaximumFlow"), gr.vidx(t, "FindMaximumFlow")
    if a == b:
        raise ValueError("FindMaximumFlow: s and t must be different vertices")
    vc = _vertex_caps(gr, "FindMaximumFlow")
    caps = gr.capacities
    F, _ = _split_flow(gr, None, (lambda k: caps[k]) if caps is not None else 1, vcaps=vc)
    return R(S(F.run(2 * a, 2 * b + 1, fname="FindMaximumFlow")))


WGM.extend("FindMaximumFlow", _fmaxflow_vcap, _has_vcap,
           "W24: graphs carrying VertexCapacity -> {c1, ...} use a vertex-split network (every vertex, s and t "
           "included, passes at most its capacity; edge capacities from EdgeCapacity, default 1). Check: exact (tests: "
           "a unit capacity on the middle of two parallel paths halves the flow)", status="added")


# ═══════════════════════════════════════ tours, cost flows, cycles, dominators ═══════════════════════════════════════

@fn("FindPostmanTour", area=AREA, status="added",
    notes="FindPostmanTour[g]: {tour} with a closed walk of minimum total EdgeWeight that uses every edge at least "
          "once (Chinese postman: odd vertices paired by an exact minimum-weight perfect matching, at most 16 odd "
          "vertices, duplicated shortest paths, then Hierholzer), edges oriented along the walk and starting at the "
          "first vertex with an edge; {} when the edges are not all in one component or there are no edges; "
          "FindPostmanTour[g, k] gives at most k tours (one is found); undirected graphs only. Check: self-check "
          "(the walk is closed, covers every edge, and its weight equals edge weights plus the matching cost; tests: "
          "Eulerian graphs need no repeats, a path of length 3 is walked twice).")
def _postman(g: Expr, k: NonNegInteger = None) -> ListOf[ListOf[Expr]]:
    gr = _g(g, "FindPostmanTour")
    _need_undirected(gr, "FindPostmanTour")
    if k is not None and _int(k, "FindPostmanTour", 0, "k") == 0:
        return L()
    m = len(gr.edges)
    if m == 0:
        return L()
    touched = {v for i, j, _ in gr.edges for v in (i, j)}
    comp = next(c for c in _weak_components(gr) if gr.edges[0][0] in c)
    if not touched <= set(comp):
        return L()
    deg = [0] * gr.n
    for i, j, _ in gr.edges:
        deg[i] += 1
        deg[j] += 1
    odd = [v for v in range(gr.n) if deg[v] % 2]
    if len(odd) > 16:
        raise NotImplementedError("FindPostmanTour: at most 16 odd-degree vertices are supported")
    sp_info = {v: gr.sssp(v, undirected=True) for v in odd}
    memo = {}

    def match(mask):
        if mask == 0:
            return 0, []
        if mask in memo:
            return memo[mask]
        i = (mask & -mask).bit_length() - 1
        best = None
        for j in range(i + 1, len(odd)):
            if mask >> j & 1:
                c, pr = match(mask & ~(1 << i) & ~(1 << j))
                c = c + sp_info[odd[i]][0][odd[j]]
                if best is None or c < best[0]:
                    best = (c, pr + [(odd[i], odd[j])])
        memo[mask] = best
        return best
    mcost, pairs = match((1 << len(odd)) - 1)
    use = list(range(m))
    for a, b in pairs:
        prev = sp_info[a][1]
        v = b
        while v != a:
            u = prev[v]
            cand = [k2 for k2, (i, j, _) in enumerate(gr.edges) if {i, j} == {u, v}]
            use.append(min(cand, key=lambda k2: (gr.weight(k2), k2)))
            v = u
    adj = [[] for _ in range(gr.n)]
    for eid, k2 in enumerate(use):
        i, j, _ = gr.edges[k2]
        adj[i].append((j, eid))
        if i != j:
            adj[j].append((i, eid))
    for a in adj:
        a.reverse()
    used = [False] * len(use)
    start = min(touched)
    stack, walk = [(start, None)], []
    while stack:
        v, e = stack[-1]
        while adj[v] and used[adj[v][-1][1]]:
            adj[v].pop()
        if adj[v]:
            w, eid = adj[v].pop()
            used[eid] = True
            stack.append((w, eid))
        else:
            stack.pop()
            if e is not None:
                walk.append((stack[-1][0], v, e))
    walk.reverse()
    total = sum(gr.weight(use[e]) for _, _, e in walk)
    ok = (len(walk) == len(use) and walk[0][0] == walk[-1][1] == start
          and all(walk[x][1] == walk[x + 1][0] for x in range(len(walk) - 1))
          and sp.simplify(total - sum(gr.weight(k2) for k2 in range(m)) - mcost) == 0)
    if not ok:
        raise ArithmeticError("FindPostmanTour: internal check failed (the walk is not a valid postman tour)")
    return L(L(*(gr.edge(a, b, False) for a, b, _ in walk)))


def _edge_costs(gr, fname):
    c = gr.opts.get("EdgeCost")
    if c is None:
        return [sp.Integer(1)] * len(gr.edges)
    if not is_list(c) or len(c.args) != len(gr.edges):
        raise ValueError(f"{fname}: EdgeCost must be a list with one value per edge")
    out = []
    for x in c.args:
        v = S(x)
        if not (isinstance(v, sp.Basic) and v.is_number and v.is_real) or v < 0:
            raise ValueError(f"{fname}: EdgeCost values must be non-negative real numbers (got {x})")
        out.append(v)
    return out


@fn("FindMinimumCostFlow", area=AREA, status="added",
    notes="FindMinimumCostFlow[g, s, t]: the minimum total cost of a maximum s-t flow; FindMinimumCostFlow[g, s, t, d]: "
          "of a flow of value d; FindMinimumCostFlow[g, {sd1, sd2, ...}]: of a flow meeting vertex supplies (> 0) and "
          "demands (< 0) in VertexList order; a final \"CostValue\" (default) or \"FlowValue\" property; other "
          "properties raise NotImplementedError. Costs from EdgeCost (default 1), capacities from EdgeCapacity "
          "(default Infinity); undirected edges carry flow either way; successive shortest paths, exact for exact "
          "data; an unbounded maximum flow or infeasible demands raise ValueError. Check: self-check (tests: unit "
          "costs give the shortest-path distance times d; LP duality on a small network).")
def _mcflow(g: Expr, *args: Expr) -> Expr:
    fname = "FindMinimumCostFlow"
    gr = _g(g, fname)
    gr.require_pure(fname)
    args = list(args)
    prop = "CostValue"
    if args and isinstance(args[-1], str):
        prop = args.pop()
        if prop not in ("CostValue", "FlowValue"):
            raise NotImplementedError(f"{fname}: only the \"CostValue\" and \"FlowValue\" properties are supported")
    costs = _edge_costs(gr, fname)
    caps = gr.capacities if gr.capacities is not None else [INF] * len(gr.edges)
    F = _Flow(gr.n + 2)
    for k, (i, j, d) in enumerate(gr.edges):
        if i == j:
            continue
        F.arc(i, j, caps[k], costs[k])
        if not d:
            F.arc(j, i, caps[k], costs[k])
    if len(args) == 1:
        sd = args[0]
        if not is_list(sd) or len(sd.args) != gr.n:
            raise ValueError(f"{fname}: give one supply/demand value per vertex")
        vals = [_real(x, fname, "a supply") for x in sd.args]
        if sp.simplify(sum(vals)) != 0:
            raise ValueError(f"{fname}: supplies and demands must sum to zero")
        src, snk = gr.n, gr.n + 1
        want = 0
        for v, x in enumerate(vals):
            if x > 0:
                F.arc(src, v, x)
                want += x
            elif x < 0:
                F.arc(v, snk, -x)
        flow, cost = F.min_cost(src, snk, want, fname)
        if flow != want:
            raise ValueError(f"{fname}: the supplies and demands cannot be met")
    elif len(args) in (2, 3):
        a, b = gr.vidx(args[0], fname), gr.vidx(args[1], fname)
        if a == b:
            raise ValueError(f"{fname}: s and t must be different vertices")
        want = _real(args[2], fname, "the flow value", lo=0) if len(args) == 3 else None
        flow, cost = F.min_cost(a, b, want, fname)
        if want is not None and flow != want:
            raise ValueError(f"{fname}: a flow of value {args[2]} from s to t is not possible")
    else:
        raise ValueError(f"{fname}: expected {fname}[g, s, t], {fname}[g, s, t, d] or {fname}[g, {{sd...}}]")
    return R(S(cost if prop == "CostValue" else flow))


@fn("FindFundamentalCycles", area=AREA, status="added",
    notes="FindFundamentalCycles[g]: the fundamental cycles of the breadth-first spanning forest (roots and "
          "neighbours in VertexList order) of an undirected graph: one cycle per non-tree edge, in EdgeList order, "
          "each a list of edges oriented along the cycle starting with the non-tree edge (self-loops and parallel "
          "edges give cycles of length 1 and 2); m - n + c cycles; directed graphs raise NotImplementedError. A valid "
          "cycle basis may differ from Wolfram's. Check: self-check (the count is m - n + c and each cycle is closed; "
          "tests: trees give {}, the Petersen graph gives 6).")
def _ffc(g: Expr) -> ListOf[ListOf[Expr]]:
    gr = _g(g, "FindFundamentalCycles")
    _need_undirected(gr, "FindFundamentalCycles")
    n = gr.n
    parent, pedge, depth = [None] * n, [None] * n, [None] * n
    adj = gr.out_adj(undirected=True)
    tree = set()
    for r in range(n):
        if depth[r] is not None:
            continue
        depth[r] = 0
        q = deque([r])
        while q:
            u = q.popleft()
            for w, _, k in sorted(adj[u], key=lambda t: (t[0], t[2])):
                if depth[w] is None:
                    depth[w], parent[w], pedge[w] = depth[u] + 1, u, k
                    tree.add(k)
                    q.append(w)
    cycles = []
    for k, (i, j, _) in enumerate(gr.edges):
        if k in tree:
            continue
        up_j, up_i = [], []
        a, b = j, i
        while depth[a] > depth[b]:
            up_j.append(a)
            a = parent[a]
        while depth[b] > depth[a]:
            up_i.append(b)
            b = parent[b]
        while a != b:
            up_j.append(a)
            up_i.append(b)
            a, b = parent[a], parent[b]
        verts = [i, j] + [parent[x] for x in up_j] + [x for x in reversed(up_i)]
        # verts: i -> j -> ... -> lca -> ... -> i
        edges = [gr.edge(i, j, False)]
        for x in range(1, len(verts) - 1):
            edges.append(gr.edge(verts[x], verts[x + 1], False))
        if verts[-1] != i and len(verts) > 2:
            raise ArithmeticError("FindFundamentalCycles: internal check failed (a cycle is not closed)")
        cycles.append(L(*edges) if i != j else L(gr.edge(i, i, False)))
    c = len(_weak_components(gr))
    if len(cycles) != len(gr.edges) - n + c:
        raise ArithmeticError("FindFundamentalCycles: internal check failed (wrong number of cycles)")
    return L(*cycles)


def _idoms(gr, r, fname):
    if not gr.directed or gr.mixed:
        raise NotImplementedError(f"{fname}: only directed graphs are supported")
    n = gr.n
    succ = [set() for _ in range(n)]
    pred = [set() for _ in range(n)]
    for i, j, _ in gr.edges:
        succ[i].add(j)
        pred[j].add(i)
    reach = _bfs_dist(succ, r)
    allv = set(reach)
    dom = {v: set(allv) for v in allv}
    dom[r] = {r}
    changed = True
    while changed:
        changed = False
        for v in sorted(allv - {r}):
            ps = [dom[p] for p in pred[v] if p in allv]
            nd = set.intersection(*ps) | {v} if ps else {v}
            if nd != dom[v]:
                dom[v], changed = nd, True
    idom = {r: r}
    for v in allv - {r}:
        strict = dom[v] - {v}
        idom[v] = max(strict, key=lambda d: len(dom[d]))
        if dom[idom[v]] != strict:
            raise ArithmeticError(f"{fname}: internal check failed (dominator sets are not a chain)")
    return idom


@fn("DominatorTreeGraph", area=AREA, status="added",
    notes="DominatorTreeGraph[g, r]: the tree with an edge idom(v) -> v for every vertex v != r reachable from r "
          "(vertices reachable from r, in VertexList order); iterative dominator-set data flow; directed graphs "
          "only (NotImplementedError otherwise). Check: self-check (dominator sets form a chain; tests: a diamond "
          "1->2, 1->3, 2->4, 3->4 has 1 as idom of 4).")
def _domtree(g: Expr, r: Expr) -> Expr:
    gr = _g(g, "DominatorTreeGraph")
    root = gr.vidx(r, "DominatorTreeGraph")
    idom = _idoms(gr, root, "DominatorTreeGraph")
    vs = [v for v in range(gr.n) if v in idom]
    return _new([gr.vertices[v] for v in vs],
                [Expr("DirectedEdge", gr.vertices[idom[v]], gr.vertices[v]) for v in vs if v != root])


@fn("DominatorVertexList", area=AREA, status="added",
    notes="DominatorVertexList[g, r]: the immediate dominator of every vertex in VertexList order (the root maps to "
          "itself); a vertex unreachable from r raises ValueError (it has no dominator); directed graphs only. Check: "
          "self-check (as DominatorTreeGraph; tests: on a directed path each vertex's dominator is its predecessor).")
def _domlist(g: Expr, r: Expr) -> ListOf[Expr]:
    gr = _g(g, "DominatorVertexList")
    root = gr.vidx(r, "DominatorVertexList")
    idom = _idoms(gr, root, "DominatorVertexList")
    miss = [gr.vertices[v] for v in range(gr.n) if v not in idom]
    if miss:
        raise ValueError(f"DominatorVertexList: vertices {miss} are not reachable from {r}")
    return L(*(gr.vertices[idom[v]] for v in range(gr.n)))


_INERT["Direction"] = "option"
record("Direction", AREA, "added",
       "Direction: option name for one-sided limits and series (Limit[f, x -> a, Direction -> \"FromAbove\" | "
       "\"FromBelow\" | 1 | -1]), interpreted by Limit in the calculus modules; inert on its own. Check: exact (tests: "
       "Limit[1/x, x -> 0, Direction -> \"FromAbove\"] is Infinity).")


# ═══════════════════════════════════════ isomorphism and symmetry ═══════════════════════════════════════

def _sym_graph(g, fname):
    gr = _g(g, fname)
    gr.require_pure(fname)
    if gr.multi():
        raise NotImplementedError(f"{fname}: multigraphs are not supported")
    return gr


def _arc_sets(gr):
    out = [set() for _ in range(gr.n)]
    inn = [set() for _ in range(gr.n)]
    for i, j, d in gr.edges:
        out[i].add(j)
        inn[j].add(i)
        if not d:
            out[j].add(i)
            inn[i].add(j)
    return out, inn


def _refine(colors, out, inn):
    k = len(set(colors))
    while True:
        sig = [(colors[v], tuple(sorted(colors[w] for w in out[v])), tuple(sorted(colors[w] for w in inn[v])))
               for v in range(len(colors))]
        ranks = {s: r for r, s in enumerate(sorted(set(sig)))}
        colors = [ranks[s] for s in sig]
        if len(ranks) == k:
            return colors
        k = len(ranks)


def _canonical_labelling(gr, fname):
    """Individualisation-refinement search for the labelling with the smallest edge certificate; automorphisms
    found on the way prune equivalent branches.  Returns (labelling v -> position, certificate)."""
    n = gr.n
    out, inn = _arc_sets(gr)
    directed = gr.directed
    state = {"best": None, "first": None, "leaves": 0}
    autos = []

    def cert(lab):
        es = set()
        for i in range(n):
            for j in out[i]:
                a, b = lab[i], lab[j]
                es.add((a, b) if directed else (min(a, b), max(a, b)))
        return tuple(sorted(es))

    def auto(l1, l2):
        inv = [0] * n
        for v, p in enumerate(l2):
            inv[p] = v
        a = [inv[l1[v]] for v in range(n)]
        if a != list(range(n)):
            autos.append(a)

    def orbit_closure(seed, gens):
        orb, stack = set(seed), list(seed)
        while stack:
            u = stack.pop()
            for a in gens:
                if a[u] not in orb:
                    orb.add(a[u])
                    stack.append(a[u])
        return orb

    def search(colors, prefix):
        cells = {}
        for v, c in enumerate(colors):
            cells.setdefault(c, []).append(v)
        multi = [(len(vs), c) for c, vs in cells.items() if len(vs) > 1]
        if not multi:
            state["leaves"] += 1
            if state["leaves"] > 200_000:
                raise NotImplementedError(f"{fname}: the canonical labelling search is too large for this graph")
            c = cert(colors)
            if state["first"] is None:
                state["first"] = (c, colors)
            elif c == state["first"][0]:
                auto(colors, state["first"][1])
            if state["best"] is None or c < state["best"][0]:
                state["best"] = (c, colors)
            elif c == state["best"][0] and colors != state["best"][1]:
                auto(colors, state["best"][1])
            return
        cell = cells[min(multi)[1]]
        done = set()
        explored = []
        for v in cell:
            if v in done:
                continue
            nc = [2 * c for c in colors]
            nc[v] -= 1
            search(_refine(nc, out, inn), prefix + [v])
            explored.append(v)
            gens = [a for a in autos if all(a[p] == p for p in prefix)]
            done = orbit_closure(explored, gens)
    if n:
        search(_refine([0] * n, out, inn), [])
        return state["best"][1], state["best"][0], autos
    return [], (), autos


@fn("CanonicalGraph", area=AREA, status="added",
    notes="CanonicalGraph[g]: Graph[{1, ..., n}, edges] with the canonical labelling of smallest edge certificate "
          "(colour refinement plus individualisation, branches pruned by discovered automorphisms; limit 2*10^5 "
          "leaves), edges sorted; isomorphic graphs give identical results, direction-aware, self-loops allowed, "
          "multigraphs and mixed graphs raise NotImplementedError; options are dropped. Vertex numbering differs from "
          "Wolfram's canonical form. Check: self-check (tests: relabelled copies give the same result, "
          "non-isomorphic graphs with equal degree sequences differ, IsomorphicGraphQ[g, CanonicalGraph[g]]).")
def _canon(g: Expr) -> Expr:
    gr = _sym_graph(g, "CanonicalGraph")
    lab, c, _ = _canonical_labelling(gr, "CanonicalGraph")
    h = "DirectedEdge" if gr.directed else "UndirectedEdge"
    return _new(list(range(1, gr.n + 1)), [Expr(h, a + 1, b + 1) for a, b in c])


def _automorphisms(gr, fname):
    """Strong generating set of Aut(g) along the base 0, 1, ..., n-1 (VF2 searches via NetworkX); returns
    (generators as position lists, group order)."""
    nx = _nx()
    from networkx.algorithms import isomorphism as iso
    X = _to_nx(gr)
    n = gr.n
    out, inn = _arc_sets(gr)
    col = _refine([0] * n, out, inn)
    gens, order = [], 1
    fixed = []
    for b in range(n):
        level = []
        orbit = {b}
        for x in range(n):
            if x in orbit or col[x] != col[b]:
                continue
            A, B = X.copy(), X.copy()
            for v in range(n):
                A.nodes[v]["c"] = B.nodes[v]["c"] = col[v]
            for t, f in enumerate(fixed):
                A.nodes[f]["c"] = B.nodes[f]["c"] = ("fixed", t)
            A.nodes[b]["c"] = ("target",)
            B.nodes[x]["c"] = ("target",)
            M = (iso.DiGraphMatcher if X.is_directed() else iso.GraphMatcher)(
                A, B, node_match=lambda p, q: p["c"] == q["c"])
            m = next(M.isomorphisms_iter(), None)
            if m is None:
                continue
            a = [m[v] for v in range(n)]
            level.append(a)
            # orbit of b under this level's generators
            stack = [b]
            orbit = {b}
            while stack:
                u = stack.pop()
                for p in level:
                    if p[u] not in orbit:
                        orbit.add(p[u])
                        stack.append(p[u])
        order *= len(orbit)
        gens += level
        fixed.append(b)
    for a in gens:
        if any(a[j] not in out[a[i]] for i in range(n) for j in out[i]):
            raise ArithmeticError(f"{fname}: internal check failed (a generator is not an automorphism)")
    if gens:
        from sympy.combinatorics import Permutation, PermutationGroup
        if PermutationGroup([Permutation(a) for a in gens]).order() != order:
            raise ArithmeticError(f"{fname}: internal check failed (group order differs from the orbit product)")
    return gens, order


def _cycles_expr(a):
    seen, cyc = set(), []
    for s in range(len(a)):
        if s in seen or a[s] == s:
            continue
        c, u = [], s
        while u not in seen:
            seen.add(u)
            c.append(u + 1)
            u = a[u]
        cyc.append(L(*c))
    return Expr("Cycles", L(*cyc))


@fn("GraphAutomorphismGroup", area=AREA, status="added",
    notes="GraphAutomorphismGroup[g]: PermutationGroup[{Cycles[...], ...}] acting on vertex positions 1..n "
          "(VertexList order), from a strong generating set along the base 1, ..., n (VF2 via NetworkX; generators may "
          "differ from Wolfram's but generate the same group); direction-aware; multigraphs raise "
          "NotImplementedError. Check: self-check (every generator is verified to be an automorphism and the SymPy "
          "group order equals the product of basic orbit sizes; tests: GroupOrder 120 for Petersen, 2n for C_n, n! "
          "for K_n).")
def _autgroup(g: Expr) -> Expr:
    gr = _sym_graph(g, "GraphAutomorphismGroup")
    gens, _ = _automorphisms(gr, "GraphAutomorphismGroup")
    return Expr("PermutationGroup", L(*(_cycles_expr(a) for a in gens)))


def _orbit(seed, gens, act):
    orb, stack = {seed}, [seed]
    while stack:
        u = stack.pop()
        for a in gens:
            w = act(a, u)
            if w not in orb:
                orb.add(w)
                stack.append(w)
    return orb


@fn("VertexTransitiveGraphQ", area=AREA, status="added",
    notes="VertexTransitiveGraphQ[g]: True when the automorphism group has a single orbit on the vertices (the empty "
          "graph and a single vertex are vertex-transitive); non-graphs give False; needs NetworkX. Check: exact "
          "(tests: Petersen, cycles, cubes True; paths with 3+ vertices, stars False).")
def _vtq(g: Expr) -> Boolean:
    if isinstance(g, Symbol):
        raise Unevaluated()
    if not is_graph_like(g):
        return False
    gr = _sym_graph(g, "VertexTransitiveGraphQ")
    if gr.n <= 1:
        return True
    gens, _ = _automorphisms(gr, "VertexTransitiveGraphQ")
    return len(_orbit(0, gens, lambda a, u: a[u])) == gr.n


@fn("EdgeTransitiveGraphQ", area=AREA, status="added",
    notes="EdgeTransitiveGraphQ[g]: True when the automorphism group has a single orbit on the edges (ordered pairs "
          "for directed graphs; a graph without edges counts as edge-transitive); non-graphs give False; needs "
          "NetworkX. Check: exact (tests: stars and complete bipartite graphs True although not vertex-transitive, "
          "paths with 4+ vertices False).")
def _etq(g: Expr) -> Boolean:
    if isinstance(g, Symbol):
        raise Unevaluated()
    if not is_graph_like(g):
        return False
    gr = _sym_graph(g, "EdgeTransitiveGraphQ")
    if not gr.edges:
        return True
    gens, _ = _automorphisms(gr, "EdgeTransitiveGraphQ")
    key = (lambda i, j: (i, j)) if gr.directed else (lambda i, j: (min(i, j), max(i, j)))
    allk = {key(i, j) for i, j, _ in gr.edges}
    i0, j0, _ = gr.edges[0]
    return len(_orbit(key(i0, j0), gens, lambda a, e: key(a[e[0]], a[e[1]]))) == len(allk)


def _monos(g1, g2, fname):
    """Monomorphisms of the pattern g1 into g2 (non-induced subgraph isomorphisms, VF2 via NetworkX)."""
    a, b = _sym_graph(g1, fname), _sym_graph(g2, fname)
    if a.edges and b.edges and a.directed != b.directed:
        return a, b, iter(())
    nx = _nx()
    from networkx.algorithms import isomorphism as iso
    X, Y = _to_nx(a), _to_nx(b)
    if X.is_directed() != Y.is_directed():
        X, Y = (X.to_directed(), Y) if Y.is_directed() else (X, Y.to_directed())
    M = (iso.DiGraphMatcher if Y.is_directed() else iso.GraphMatcher)(Y, X)
    return a, b, ({v: k for k, v in m.items()} for m in M.subgraph_monomorphisms_iter())


@fn("IsomorphicSubgraphQ", area=AREA, status="added",
    notes="IsomorphicSubgraphQ[g1, g2]: True when g1 is isomorphic to a (not necessarily induced) subgraph of g2 "
          "(VF2 subgraph monomorphism via NetworkX); direction-aware; multigraphs raise NotImplementedError. Check: "
          "exact (tests: C5 in Petersen True, K4 in Petersen False, a path in any longer path True).")
def _isosubq(g1: Expr, g2: Expr) -> Boolean:
    _, _, it = _monos(g1, g2, "IsomorphicSubgraphQ")
    return next(it, None) is not None


@fn("FindSubgraphIsomorphism", area=AREA, status="added",
    notes="FindSubgraphIsomorphism[g1, g2]: {<|v1 -> w1, ...|>} mapping g1 injectively onto a (not necessarily "
          "induced) subgraph of g2, keys in VertexList order of g1; {} when none; [g1, g2, n] (or All) for at most n; "
          "VF2 via NetworkX, order of results may differ from Wolfram's. Check: exact (tests: every mapped edge is an "
          "edge of g2; C4 into K4 has 24 maps).")
def _fsubiso(g1: Expr, g2: Expr, n: NonNegInteger | Expr = None) -> ListOf[Expr]:
    from ..kernel_lists import make_assoc
    lim = _count_spec(n, "FindSubgraphIsomorphism")
    a, b, it = _monos(g1, g2, "FindSubgraphIsomorphism")
    out = []
    if lim == 0:
        return L()
    for m in it:
        out.append(make_assoc([(a.vertices[i], b.vertices[m[i]]) for i in range(a.n)]))
        if lim is not None and len(out) >= lim:
            break
    return L(*out)


@fn("FindIsomorphicSubgraph", area=AREA, status="added",
    notes="FindIsomorphicSubgraph[g1, g2]: {h} with a (not necessarily induced) subgraph h of g1 isomorphic to g2 "
          "(vertices and edges of g1, in its VertexList/EdgeList order); {} when none; [g1, g2, n] (or All) for at "
          "most n distinct subgraphs; VF2 via NetworkX. Check: exact (tests: IsomorphicGraphQ[h, g2] and h is a "
          "subgraph of g1; K3 occurs 4 times in K4).")
def _fisosub(g1: Expr, g2: Expr, n: NonNegInteger | Expr = None) -> ListOf[Expr]:
    lim = _count_spec(n, "FindIsomorphicSubgraph")
    b, a, it = _monos(g2, g1, "FindIsomorphicSubgraph")
    seen, out = set(), []
    if lim == 0:
        return L()
    for m in it:
        vs = sorted(m.values())
        ks = []
        for i, j, _ in b.edges:
            p, q = m[i], m[j]
            k = next(k for k, (x, y, d) in enumerate(a.edges)
                     if (x, y) == (p, q) or (not d and (y, x) == (p, q)))
            ks.append(k)
        key = (tuple(vs), tuple(sorted(ks)))
        if key in seen:
            continue
        seen.add(key)
        out.append(_new([a.vertices[v] for v in vs], [a.edge_exprs[k] for k in sorted(ks)]))
        if lim is not None and len(out) >= lim:
            break
    return L(*out)


# ═══════════════════════════════════════ Tutte and flow polynomials ═══════════════════════════════════════

def _tutte(gr, x, y, fname):
    _need_undirected(gr, fname)
    es = [(i, j) for i, j, _ in gr.edges]
    if len(es) > 24:
        raise NotImplementedError(f"{fname}: graphs with more than 24 edges are not supported")
    memo = {}

    def comps(n_edges):
        par = {}

        def find(u):
            while par.setdefault(u, u) != u:
                par[u] = par[par[u]]
                u = par[u]
            return u
        for i, j in n_edges:
            par[find(i)] = find(j)
        return find

    def is_bridge(edges, k):
        i, j = edges[k]
        rest = edges[:k] + edges[k + 1:]
        f = comps(rest)
        return f(i) != f(j)

    def T(edges):
        key = tuple(sorted(edges))
        if key in memo:
            return memo[key]
        if not edges:
            return sp.Integer(1)
        i, j = edges[0]
        rest = edges[1:]
        if i == j:
            r = y * T(rest)
        else:
            lo, hi = min(i, j), max(i, j)
            contracted = [(lo if a == hi else a, lo if b == hi else b) for a, b in rest]
            contracted = [(min(a, b), max(a, b)) for a, b in contracted]
            if is_bridge(edges, 0):
                r = x * T(contracted)
            else:
                r = T(rest) + T(contracted)
        r = sp.expand(r)
        memo[key] = r
        return r
    res = T([(min(i, j), max(i, j)) for i, j in es])
    # self-check: T(1, 1) is the number of spanning forests with one tree per component (matrix-tree theorem)
    if isinstance(x, sp.Symbol) and isinstance(y, sp.Symbol) and x != y:
        trees = 1
        for c in _weak_components(gr):
            if len(c) < 2:
                continue
            idx = {v: t for t, v in enumerate(c)}
            Lm = sp.zeros(len(c), len(c))
            for i, j in es:
                if i != j and i in idx:
                    a, b = idx[i], idx[j]
                    Lm[a, a] += 1
                    Lm[b, b] += 1
                    Lm[a, b] -= 1
                    Lm[b, a] -= 1
            trees *= Lm[1:, 1:].det()
        if res.subs({x: 1, y: 1}) != trees:
            raise ArithmeticError(f"{fname}: internal check failed (T(1, 1) differs from the spanning tree count)")
    return res


@fn("TuttePolynomial", area=AREA, status="added",
    notes="TuttePolynomial[g, {x, y}]: the Tutte polynomial (deletion-contraction with memoisation: loops give y, "
          "bridges x), expanded; multigraphs and self-loops allowed; undirected graphs with at most 24 edges "
          "(NotImplementedError otherwise). Check: self-check (T(1, 1) is verified against the matrix-tree spanning "
          "tree count; tests: trees give x^(n-1), C_n gives x^(n-1) + ... + x + y, K4 the classical polynomial).")
def _tuttepoly(g: Expr, xy: ListOf[Expr]) -> Expr:
    gr = _g(g, "TuttePolynomial")
    if not (is_list(xy) and len(xy.args) == 2):
        raise TypeError("TuttePolynomial: the variables must be given as a list {x, y}")
    sx, sy = S(xy.args[0]), S(xy.args[1])
    if all(isinstance(v, sp.Symbol) for v in (sx, sy)) and sx != sy:
        return R(_tutte(gr, sx, sy, "TuttePolynomial"))
    X, Y = sp.Dummy("x"), sp.Dummy("y")
    return R(sp.expand(_tutte(gr, X, Y, "TuttePolynomial").subs({X: sx, Y: sy})))


@fn("FlowPolynomial", area=AREA, status="added",
    notes="FlowPolynomial[g, k]: the number of nowhere-zero Z_k flows, (-1)^(m - n + c) T(0, 1 - k) from the Tutte "
          "polynomial (same limits as TuttePolynomial); expanded in k, or evaluated for a number. Check: self-check "
          "(via TuttePolynomial; tests: C_n gives k - 1, a bridge gives 0, K4 gives (k-1)(k-2)(k-3)).")
def _flowpoly(g: Expr, k: Expr) -> Expr:
    gr = _g(g, "FlowPolynomial")
    X, Y = sp.Dummy("x"), sp.Dummy("y")
    t = _tutte(gr, X, Y, "FlowPolynomial")
    kk = S(k)
    c = len(_weak_components(gr))
    sign = (-1) ** (len(gr.edges) - gr.n + c)
    return R(sp.expand(sign * t.subs({X: 0, Y: 1 - kk})))


# ═══════════════════════════════════════ planar faces ═══════════════════════════════════════

def _faces(gr, fname):
    nx = _nx()
    _need_simple_undirected(gr, fname)
    if gr.n == 0 or len(_weak_components(gr)) > 1:
        raise NotImplementedError(f"{fname}: only connected graphs are supported")
    X = nx.Graph()
    X.add_nodes_from(range(gr.n))
    X.add_edges_from((i, j) for i, j, _ in gr.edges)
    ok, emb = nx.check_planarity(X)
    if not ok:
        raise ValueError(f"{fname}: the graph is not planar")
    seen, faces = set(), []
    for u, v in emb.edges():
        if (u, v) in seen:
            continue
        f = emb.traverse_face(u, v, mark_half_edges=seen)
        faces.append(f)
    if not gr.edges:
        faces = [[0]]
    out = []
    for f in faces:
        r = f.index(min(f))
        out.append(f[r:] + f[:r])
    out.sort(key=lambda f: (len(f), f))
    if gr.edges and len(out) != len(gr.edges) - gr.n + 2:
        raise ArithmeticError(f"{fname}: internal check failed (Euler's formula F = m - n + 2 fails)")
    return out


@fn("PlanarFaceList", area=AREA, status="added",
    notes="PlanarFaceList[g]: the faces of a planar embedding of a connected simple undirected graph (NetworkX "
          "left-right planarity embedding), each as a vertex cycle starting at its first vertex in VertexList order "
          "and following the embedding's orientation, sorted by length; unique up to orientation for 3-connected "
          "graphs, otherwise one valid embedding (may differ from Wolfram's); a bridge's face lists its ends twice; "
          "non-planar graphs raise ValueError. Check: self-check (Euler's formula F = m - n + 2; tests: cube has 6 "
          "4-faces, K4 four triangles).")
def _pfl(g: Expr) -> ListOf[ListOf[Expr]]:
    gr = _g(g, "PlanarFaceList")
    return L(*(_verts(gr, f) for f in _faces(gr, "PlanarFaceList")))


@fn("DualPlanarGraph", area=AREA, status="added",
    notes="DualPlanarGraph[g]: the planar dual: vertices 1..F for the faces of PlanarFaceList[g] (same order), one "
          "undirected edge per edge of g joining the faces on its two sides (bridges give self-loops, so the result "
          "may be a multigraph); same embedding caveats as PlanarFaceList. Check: self-check (Euler's formula; tests: "
          "dual of the cube is the octahedron, K4 is self-dual, edge count preserved).")
def _dual(g: Expr) -> Expr:
    gr = _g(g, "DualPlanarGraph")
    faces = _faces(gr, "DualPlanarGraph")
    side = {}
    for fi, f in enumerate(faces):
        for t in range(len(f)):
            side.setdefault((f[t], f[(t + 1) % len(f)]), fi)
    es = []
    for i, j, _ in gr.edges:
        a, b = side[(i, j)], side[(j, i)]
        es.append(UE(a + 1, b + 1))
    return _new(list(range(1, len(faces) + 1)), es)


# ═══════════════════════════════════════ ShortestPathFunction ═══════════════════════════════════════

_SPF_TYPES = {"SingleSource": 2, "SingleTarget": 2, "AllPairs": 1}


def _is_all(x):
    return isinstance(x, Symbol) and x.name == "All"


def _spf_data(head_args, fname):
    if len(head_args) != 2 or head_args[0] not in _SPF_TYPES or not is_list(head_args[1]):
        raise ValueError(f"{fname}: expected ShortestPathFunction[type, {{g, ...}}] with type \"SingleSource\", "
                         "\"SingleTarget\" or \"AllPairs\"")
    typ, data = head_args[0], list(head_args[1].args)
    if len(data) != _SPF_TYPES[typ] or not is_graph_like(data[0]):
        raise ValueError(f"{fname}: malformed {typ} data")
    return typ, data


@fn("ShortestPathFunction", area=AREA, status="added",
    notes="ShortestPathFunction[\"SingleSource\", {g, s}], [\"SingleTarget\", {g, t}] and [\"AllPairs\", {g}] are "
          "the inert results of FindShortestPath[g, s, All], FindShortestPath[g, All, t] and FindShortestPath[g, All, "
          "All]; applied as spf[t], spf[s] or spf[s, t] they give FindShortestPath's vertex list ({} when "
          "unreachable). The data layout differs from Wolfram's internal form. Check: exact (tests: agrees with "
          "FindShortestPath for every target).")
def _spf(typ: String, data: ListOf[Expr]) -> Expr:
    t, d = _spf_data((typ, data), "ShortestPathFunction")
    _g(d[0], "ShortestPathFunction")
    raise Unevaluated()


def _spf_apply(head, args, ev):
    typ, data = _spf_data(head.args, "ShortestPathFunction")
    g = data[0]
    if typ == "SingleSource" and len(args) == 1:
        return WGM._fsp(g, data[1], args[0])
    if typ == "SingleTarget" and len(args) == 1:
        return WGM._fsp(g, args[0], data[1])
    if typ == "AllPairs" and len(args) == 2:
        return WGM._fsp(g, args[0], args[1])
    raise ValueError(f"ShortestPathFunction: a {typ} function takes {1 if typ != 'AllPairs' else 2} argument(s)")


KL.COMPOUND_HEAD_HOOKS["ShortestPathFunction"] = _spf_apply


def _fsp_all(g, s, t):
    gr = graph_of(g, "FindShortestPath")
    if _is_all(s) and _is_all(t):
        return Expr("ShortestPathFunction", "AllPairs", L(g))
    if _is_all(t):
        gr.vidx(s, "FindShortestPath")
        return Expr("ShortestPathFunction", "SingleSource", L(g, s))
    gr.vidx(t, "FindShortestPath")
    return Expr("ShortestPathFunction", "SingleTarget", L(g, t))


WGM.extend("FindShortestPath", _fsp_all, lambda a: len(a) == 3 and (_is_all(a[1]) or _is_all(a[2])),
           "W24: FindShortestPath[g, s, All], [g, All, t] and [g, All, All] return a ShortestPathFunction. Check: exact "
           "(tests: the function agrees with FindShortestPath[g, s, t])", status="added")


# ═══════════════════════════════════════ random graph distributions ═══════════════════════════════════════

def _dist_head(name, check, notes):
    @fn(name, area=AREA, status="added", notes=notes)
    def impl(*args: Expr) -> Expr:
        check(*args)
        raise Unevaluated()
    return impl


def _prob(p, fname):
    return _real(p, fname, "the probability", lo=0, hi=1)


def _chk_bern(*a):
    if len(a) != 2:
        raise ValueError("BernoulliGraphDistribution: expected BernoulliGraphDistribution[n, p]")
    _int(a[0], "BernoulliGraphDistribution", 0, "n")
    _prob(a[1], "BernoulliGraphDistribution")


def _chk_unif(*a):
    if len(a) != 2:
        raise ValueError("UniformGraphDistribution: expected UniformGraphDistribution[n, m]")
    n = _int(a[0], "UniformGraphDistribution", 0, "n")
    m = _int(a[1], "UniformGraphDistribution", 0, "m")
    if m > n * (n - 1) // 2:
        raise ValueError(f"UniformGraphDistribution: a simple graph on {n} vertices has at most {n * (n - 1) // 2} "
                         "edges")


def _chk_ba(*a):
    if len(a) != 2:
        raise ValueError("BarabasiAlbertGraphDistribution: expected BarabasiAlbertGraphDistribution[n, k]")
    n = _int(a[0], "BarabasiAlbertGraphDistribution", 1, "n")
    k = _int(a[1], "BarabasiAlbertGraphDistribution", 1, "k")
    if k >= n:
        raise ValueError("BarabasiAlbertGraphDistribution: k must be smaller than n")


def _chk_ws(*a):
    if len(a) not in (2, 3):
        raise ValueError("WattsStrogatzGraphDistribution: expected WattsStrogatzGraphDistribution[n, p] or [n, p, k]")
    n = _int(a[0], "WattsStrogatzGraphDistribution", 1, "n")
    _prob(a[1], "WattsStrogatzGraphDistribution")
    k = _int(a[2], "WattsStrogatzGraphDistribution", 1, "k") if len(a) == 3 else 2
    if 2 * k >= n:
        raise ValueError("WattsStrogatzGraphDistribution: 2 k must be smaller than n")


def _chk_price(*a):
    if len(a) != 3:
        raise ValueError("PriceGraphDistribution: expected PriceGraphDistribution[n, k, a]")
    n = _int(a[0], "PriceGraphDistribution", 1, "n")
    _int(a[1], "PriceGraphDistribution", 1, "k")
    if _real(a[2], "PriceGraphDistribution", "the attractiveness") <= 0:
        raise ValueError("PriceGraphDistribution: the attractiveness a must be positive")
    del n


def _chk_spatial(*a):
    if len(a) not in (2, 3):
        raise ValueError("SpatialGraphDistribution: expected SpatialGraphDistribution[n, r] or [n, r, d]")
    _int(a[0], "SpatialGraphDistribution", 0, "n")
    _real(a[1], "SpatialGraphDistribution", "the radius", lo=0)
    if len(a) == 3:
        if K.is_int(a[2]):
            _int(a[2], "SpatialGraphDistribution", 1, "the dimension")
        elif isinstance(a[2], Symbol):
            raise Unevaluated()
        else:
            raise NotImplementedError("SpatialGraphDistribution: only the uniform unit-cube forms [n, r] and "
                                      "[n, r, d] are supported (not distributions or regions)")


def _graphical(ds):
    """Erdos-Gallai test."""
    ds = sorted(ds, reverse=True)
    if sum(ds) % 2:
        return False
    for k in range(1, len(ds) + 1):
        if sum(ds[:k]) > k * (k - 1) + sum(min(d, k) for d in ds[k:]):
            return False
    return True


def _chk_deg(*a):
    if len(a) != 1 or not is_list(a[0]):
        raise ValueError("DegreeGraphDistribution: expected DegreeGraphDistribution[{d1, d2, ...}]")
    ds = [_int(d, "DegreeGraphDistribution", 0, "a degree") for d in a[0].args]
    if not _graphical(ds):
        raise ValueError(f"DegreeGraphDistribution: {ds} is not the degree sequence of a simple graph")


_dist_head("BernoulliGraphDistribution", _chk_bern,
           "BernoulliGraphDistribution[n, p]: inert distribution (validated: n >= 0, 0 <= p <= 1); RandomGraph "
           "includes each of the n(n-1)/2 vertex pairs independently with probability p. Check: exact (tests: p = 0 "
           "and p = 1 give the empty and complete graphs; invalid p raises).")
_dist_head("UniformGraphDistribution", _chk_unif,
           "UniformGraphDistribution[n, m]: inert distribution (m <= n(n-1)/2); RandomGraph draws a uniformly random "
           "simple graph with n vertices and m edges (as RandomGraph[{n, m}]). Check: exact (tests: edge counts; "
           "too many edges raise).")
_dist_head("BarabasiAlbertGraphDistribution", _chk_ba,
           "BarabasiAlbertGraphDistribution[n, k]: inert distribution (1 <= k < n); RandomGraph starts from the "
           "complete graph on k + 1 vertices and attaches each new vertex to k distinct existing vertices chosen with "
           "probability proportional to degree (C(k+1, 2) + (n - k - 1) k edges; Wolfram's seed graph is not "
           "documented). Check: exact (tests: vertex and edge counts, connectivity).")
_dist_head("WattsStrogatzGraphDistribution", _chk_ws,
           "WattsStrogatzGraphDistribution[n, p] and [n, p, k] (default k = 2): inert distribution (2k < n); "
           "RandomGraph starts from the 2k-regular ring lattice and rewires each edge's far end with probability p to "
           "a uniformly chosen vertex avoiding self-loops and multi-edges (n k edges). Check: exact (tests: p = 0 "
           "gives the ring lattice, edge count n k).")
_dist_head("PriceGraphDistribution", _chk_price,
           "PriceGraphDistribution[n, k, a]: inert distribution (a > 0); RandomGraph adds vertices 1..n in turn, each "
           "new vertex citing min(k, existing) distinct earlier vertices with probability proportional to in-degree "
           "+ a (directed edges new -> old). Check: exact (tests: acyclic, out-degrees min(k, i - 1)).")
_dist_head("SpatialGraphDistribution", _chk_spatial,
           "SpatialGraphDistribution[n, r] and [n, r, d]: inert distribution; RandomGraph places n points uniformly "
           "in the unit square/d-cube and joins pairs at Euclidean distance <= r; distribution and region forms raise "
           "NotImplementedError. Check: exact (tests: r = 0 gives no edges, r >= Sqrt[d] the complete graph).")
_dist_head("DegreeGraphDistribution", _chk_deg,
           "DegreeGraphDistribution[{d1, ...}]: inert distribution (the sequence must be graphical, Erdos-Gallai); "
           "RandomGraph builds a Havel-Hakimi realisation and randomises it with 10 m degree-preserving double-edge "
           "swaps (simple graphs only). Check: exact (tests: VertexDegree reproduces the sequence; non-graphical "
           "sequences raise).")

_DISTS = {"BernoulliGraphDistribution", "UniformGraphDistribution", "BarabasiAlbertGraphDistribution",
          "WattsStrogatzGraphDistribution", "PriceGraphDistribution", "SpatialGraphDistribution",
          "DegreeGraphDistribution"}


def _sample(dist):
    from .statistics import _RNG as rng
    name, a = dist.head, dist.args
    if name == "BernoulliGraphDistribution":
        n, p = int(a[0]), float(S(a[1]))
        es = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1) if rng.random() < p]
        return _new(list(range(1, n + 1)), [UE(i, j) for i, j in es])
    if name == "UniformGraphDistribution":
        n, m = int(a[0]), int(a[1])
        pairs = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
        return _new(list(range(1, n + 1)), [UE(i, j) for i, j in sorted(rng.sample(pairs, m))])
    if name == "BarabasiAlbertGraphDistribution":
        n, k = int(a[0]), int(a[1])
        es = [(i, j) for i in range(1, k + 2) for j in range(i + 1, k + 2)]
        ends = [v for e in es for v in e]
        for v in range(k + 2, n + 1):
            targets = set()
            while len(targets) < k:
                targets.add(rng.choice(ends))
            for w in sorted(targets):
                es.append((w, v))
                ends += [w, v]
        return _new(list(range(1, n + 1)), [UE(i, j) for i, j in es])
    if name == "WattsStrogatzGraphDistribution":
        n, p = int(a[0]), float(S(a[1]))
        k = int(a[2]) if len(a) == 3 else 2
        es = [(i, (i + d) % n) for d in range(1, k + 1) for i in range(n)]
        present = {frozenset(e) for e in es}
        for t, (u, v) in enumerate(es):
            if rng.random() < p:
                choices = [w for w in range(n) if w != u and frozenset((u, w)) not in present]
                if choices:
                    w = rng.choice(choices)
                    present.discard(frozenset((u, v)))
                    present.add(frozenset((u, w)))
                    es[t] = (u, w)
        return _new(list(range(1, n + 1)), [UE(u + 1, v + 1) for u, v in es])
    if name == "PriceGraphDistribution":
        n, k, att = int(a[0]), int(a[1]), float(S(a[2]))
        indeg = [0] * (n + 1)
        es = []
        for v in range(2, n + 1):
            olds = list(range(1, v))
            chosen = set()
            while len(chosen) < min(k, len(olds)):
                cand = [u for u in olds if u not in chosen]
                w = rng.choices(cand, weights=[indeg[u] + att for u in cand])[0]
                chosen.add(w)
            for w in sorted(chosen):
                es.append((v, w))
                indeg[w] += 1
        return _new(list(range(1, n + 1)), [Expr("DirectedEdge", v, w) for v, w in es])
    if name == "SpatialGraphDistribution":
        n, r = int(a[0]), float(S(a[1]))
        d = int(a[2]) if len(a) == 3 else 2
        pts = [[rng.random() for _ in range(d)] for _ in range(n)]
        es = [(i + 1, j + 1) for i in range(n) for j in range(i + 1, n) if math.dist(pts[i], pts[j]) <= r]
        return _new(list(range(1, n + 1)), [UE(i, j) for i, j in es])
    # DegreeGraphDistribution
    ds = [int(x) for x in a[0].args]
    n = len(ds)
    rem = list(ds)
    es = set()
    while True:
        order = sorted(range(n), key=lambda v: (-rem[v], v))
        v = order[0]
        if rem[v] == 0:
            break
        for w in order[1:rem[v] + 1]:
            rem[w] -= 1
            es.add((min(v, w), max(v, w)))
        rem[v] = 0
    es = sorted(es)
    for _ in range(10 * len(es)):
        if len(es) < 2:
            break
        (a1, b1), (a2, b2) = rng.sample(es, 2)
        if rng.random() < 0.5:
            a2, b2 = b2, a2
        n1, n2 = (min(a1, b2), max(a1, b2)), (min(a2, b1), max(a2, b1))
        if a1 == b2 or a2 == b1 or n1 in es or n2 in es:
            continue
        es.remove((min(a1, b1), max(a1, b1)))
        es.remove((min(a2, b2), max(a2, b2)))
        es += [n1, n2]
    deg = [0] * n
    for i, j in es:
        deg[i] += 1
        deg[j] += 1
    if deg != ds:
        raise ArithmeticError("RandomGraph: internal check failed (the degree sequence changed)")
    return _new(list(range(1, n + 1)), [UE(i + 1, j + 1) for i, j in sorted(es)])


def _rgraph_dist(dist, k=None):
    if k is None:
        return _sample(dist)
    k = _int(k, "RandomGraph", 0, "the number of graphs")
    return L(*(_sample(dist) for _ in range(k)))


WGM.extend("RandomGraph", _rgraph_dist,
           lambda a: 1 <= len(a) <= 2 and isinstance(a[0], Expr) and a[0].head in _DISTS,
           "W24: RandomGraph[dist] and RandomGraph[dist, k] sample the Bernoulli, Uniform, Barabasi-Albert, "
           "Watts-Strogatz, Price, Spatial and Degree graph distributions (reproducible with SeedRandom; the random "
           "stream differs from Wolfram's). Check: exact (tests: structural invariants of each model)", status="added")


# ═══════════════════════════════════════ W24 resume: remaining measures and components ═══════════════════════════════

_option_symbol("TaggedEdgeStyle", "Graph option and annotation: style for tagged edges (Automatic, a style, or "
                                  "{tag -> style, ...}); drawing is the graphics batch, so computations ignore it")


def _restricted_sssp(adj, s, allowed):
    """Shortest distances from s using only vertices in ``allowed`` (Dijkstra; exact SymPy weights)."""
    import heapq
    dist = {s: sp.Integer(0)}
    pq = [(0.0, s)]
    done = set()
    while pq:
        _, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        for v, w, _k in adj[u]:
            if v not in allowed:
                continue
            nd = dist[u] + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (float(nd), v))
    return dist


def _in_out(rest, fname):
    if not rest:
        return "Out"
    if len(rest) == 1 and rest[0] in ("In", "Out"):
        return rest[0]
    if len(rest) == 1 and isinstance(rest[0], Symbol) and rest[0].name in ("In", "Out"):
        return rest[0].name
    if len(rest) == 1 and isinstance(rest[0], (str, Symbol)):
        raise ValueError(f"{fname}: the second argument must be \"In\" or \"Out\" (got {rest[0]})")
    raise TypeError(f"{fname}: expected {fname}[g] or {fname}[g, \"In\" | \"Out\"]")


@fn("RadialityCentrality", area=AREA, status="added",
    notes="RadialityCentrality[g], [g, \"Out\"] (the default) and [g, \"In\"]: for vertex i, Sum over the vertices j "
          "reachable from i (or reaching i) of (d + 1 - d_ij) / (d (n - 1)) as documented by Wolfram, with d the "
          "diameter of i's out (in) component and n = VertexCount[g]; isolated vertices give 0; undirected, "
          "directed, mixed, multigraphs and EdgeWeight distances; machine reals. When the components' diameters "
          "differ (so the documented d is ambiguous) NotImplementedError is raised. Check: reference (Wolfram "
          "documentation values for {1 -> 3, 2 -> 1, 3 -> 6, 4 -> 6, 1 -> 5, 5 -> 4, 6 -> 1}, In and Out) and the "
          "GraphDistanceMatrix identity in tests.")
def _radiality(g: Expr, *rest: String) -> ListOf[Real]:
    fname = "RadialityCentrality"
    gr = _g(g, fname)
    way = _in_out(rest, fname)
    n = gr.n
    if n == 0:
        return L()
    adj = gr.out_adj() if way == "Out" else gr.in_adj()
    every = set(range(n))
    reach = [_restricted_sssp(adj, i, every) for i in range(n)]
    diams = []
    for i in range(n):
        comp = set(reach[i])
        if len(comp) == 1:
            diams.append(None)
            continue
        dm = 0
        for u in comp:
            du = _restricted_sssp(adj, u, comp)
            dm = max([dm] + [v for v in du.values()])
        diams.append(dm)
    ds = {d for d in diams if d is not None}
    if len(ds) > 1:
        raise NotImplementedError(f"{fname}: the reachable components have different diameters, so Wolfram's "
                                  "normalising diameter is ambiguous; not supported")
    out = []
    for i in range(n):
        d = diams[i]
        if d is None or d == 0:
            out.append(0.0)
            continue
        tot = sum(d + 1 - v for j, v in reach[i].items() if j != i)
        out.append(float(tot / (d * (n - 1))))
    return L(*out)


@fn("MeanDegreeConnectivity", area=AREA, status="added",
    notes="MeanDegreeConnectivity[g]: {m0, m1, ..., md} where mk is the mean of MeanNeighborDegree over the vertices "
          "of degree k (0 when there are none; isolated vertices count 0) and d the maximum degree; exact rationals "
          "for undirected simple unweighted graphs. Directed, mixed, weighted and multigraphs and the \"In\"/\"Out\" "
          "forms raise NotImplementedError. Check: reference (Wolfram documentation: GridGraph[{3, 4, 5}] gives "
          "{0, 0, 0, 4, 17/4, 262/55, 97/18}; CompleteGraph[{2, 4, 3}] part 8 is 38/7) and networkx "
          "average_degree_connectivity in tests.")
def _mdc(g: Expr, *rest: String) -> ListOf[Rational]:
    fname = "MeanDegreeConnectivity"
    if rest:
        _in_out(rest, fname)
        raise NotImplementedError(f"{fname}: the \"In\" and \"Out\" forms are not supported")
    gr = _g(g, fname)
    _need_simple_undirected(gr, fname)
    if gr.weights is not None:
        raise NotImplementedError(f"{fname}: weighted graphs are not supported")
    nb = _simple_nb(gr)
    if gr.n == 0:
        return L()
    deg = [len(s) for s in nb]
    buckets = {}
    for v in range(gr.n):
        mnd = sp.Rational(sum(deg[w] for w in nb[v]), deg[v]) if deg[v] else sp.Integer(0)
        buckets.setdefault(deg[v], []).append(mnd)
    out = []
    for k in range(max(deg) + 1):
        b = buckets.get(k)
        out.append(sp.Rational(0) if not b or k == 0 else sum(b) / len(b))
    return L(*(R(x) for x in out))


def _component_filter(comps, gr, rest, fname):
    if not rest:
        return comps
    if len(rest) > 1:
        raise TypeError(f"{fname}: too many arguments")
    vs = rest[0]
    if not is_list(vs):
        raise NotImplementedError(f"{fname}: only an explicit vertex list is supported as the third argument")
    want = {gr.vidx(v, fname) for v in vs.args}
    return [c for c in comps if want & set(c)]


def _sorted_comps(comps):
    comps = [sorted(c) for c in comps]
    comps.sort(key=lambda c: (-len(c), c[0]))
    return comps


def _sub_wg(gr, keep):
    keep = sorted(keep)
    pos = {v: t for t, v in enumerate(keep)}
    es = [e for (i, j, _), e in zip(gr.edges, gr.edge_exprs) if i in pos and j in pos]
    return WG([gr.vertices[v] for v in keep], es, {}, gr.fname), keep


def _lambda_classes(gr, k):
    """Classes of the relation 'at least k edge-disjoint paths both ways' (an equivalence relation)."""
    ones = [sp.Integer(1)] * len(gr.edges)
    reps = []
    cls = []
    for v in range(gr.n):
        for t, r in enumerate(reps):
            if _maxflow(gr, r, v, ones)[0] >= k and (not gr.directed and not gr.mixed or
                                                    _maxflow(gr, v, r, ones)[0] >= k):
                cls[t].append(v)
                break
        else:
            reps.append(v)
            cls.append([v])
    return cls


def _k_edge_subgraphs(gr, k):
    """Maximal k-edge-connected induced subgraphs (vertex index sets of size >= 2) by repeated refinement."""
    todo, done = [list(range(gr.n))], []
    while todo:
        part = todo.pop()
        if len(part) < 2:
            continue
        sub, keep = _sub_wg(gr, part)
        cls = _lambda_classes(sub, k)
        if len(cls) == 1:
            done.append(part)
        else:
            todo.extend([keep[t] for t in c] for c in cls)
    return done


@fn("KEdgeConnectedComponents", area=AREA, status="added",
    notes="KEdgeConnectedComponents[g, k] and [g, k, {v1, ...}]: the vertex sets of the maximal k-edge-connected "
          "subgraphs. k = 1 gives the connected (strongly connected for directed and mixed graphs) components "
          "including single vertices; for k >= 2 single vertices are omitted. For k >= 2 the result is computed both "
          "as maximal k-edge-connected induced subgraphs and as classes of 'k edge-disjoint paths in g'; Wolfram's "
          "documentation states both, so when the two differ NotImplementedError is raised. Components are listed "
          "largest first (ties by VertexList position), vertices in VertexList order; Wolfram's own ordering is not "
          "documented and may differ. Pattern selectors raise NotImplementedError; k < 1 raises ValueError. "
          "Check: self-check (both definitions must agree) and reference (Wolfram documentation example "
          "{1 -> 2, 1 -> 5, 2 -> 3, 3 -> 4, 4 -> 5, 6 -> 3, 4 -> 6} with k = 1; networkx k_edge_subgraphs in tests).")
def _kecc(g: Expr, k: PosInteger, *rest: ListOf[Expr]) -> ListOf[ListOf[Expr]]:
    fname = "KEdgeConnectedComponents"
    gr = _g(g, fname)
    k = _int(k, fname, 1, "k")
    if k == 1:
        comps = _strong_components(gr) if (gr.directed or gr.mixed) else _components(gr)
        comps = [list(c) for c in comps]
    else:
        a = _k_edge_subgraphs(gr, k)
        b = [c for c in _lambda_classes(gr, k) if len(c) >= 2]
        if sorted(map(sorted, a)) != sorted(map(sorted, b)):
            raise NotImplementedError(f"{fname}: the maximal {k}-edge-connected subgraphs differ from the classes of "
                                      f"{k} edge-disjoint paths in this graph; Wolfram's choice is not documented")
        comps = a
    comps = _component_filter(_sorted_comps(comps), gr, rest, fname)
    return L(*(_verts(gr, c) for c in comps))


def _blocks(nb, n):
    """Biconnected components (vertex sets, bridges included as 2-sets) by Hopcroft-Tarjan, iterative."""
    disc, low = [None] * n, [0] * n
    out, t = [], 0
    for r in range(n):
        if disc[r] is not None or not nb[r]:
            continue
        disc[r] = low[r] = t
        t += 1
        stack = [(r, None, iter(sorted(nb[r])))]
        estack = []
        while stack:
            u, p, it = stack[-1]
            adv = False
            for w in it:
                if disc[w] is None:
                    estack.append((u, w))
                    disc[w] = low[w] = t
                    t += 1
                    stack.append((w, u, iter(sorted(nb[w]))))
                    adv = True
                    break
                if w != p and disc[w] < disc[u]:
                    estack.append((u, w))
                    low[u] = min(low[u], disc[w])
            if adv:
                continue
            stack.pop()
            if p is not None:
                low[p] = min(low[p], low[u])
                if low[u] >= disc[p]:
                    blk = set()
                    while True:
                        e = estack.pop()
                        blk.update(e)
                        if e == (p, u):
                            break
                    out.append(blk)
    return out


@fn("KVertexConnectedComponents", area=AREA, status="added",
    notes="KVertexConnectedComponents[g, k] and [g, k, {v1, ...}] for undirected graphs: k = 1 gives the connected "
          "components (graphs with isolated vertices raise NotImplementedError, as Wolfram's convention for them is "
          "not documented), k = 2 the biconnected components (blocks; a bridge is a 2-vertex block, as in Wolfram's "
          "documentation example). Components are listed largest first (ties by VertexList position), vertices in "
          "VertexList order; Wolfram's ordering may differ. k >= 3, directed and mixed graphs and pattern selectors "
          "raise NotImplementedError; k < 1 raises ValueError. Check: self-check (each block is verified "
          "2-connected or a single edge) and reference (networkx biconnected_components in tests).")
def _kvcc(g: Expr, k: PosInteger, *rest: ListOf[Expr]) -> ListOf[ListOf[Expr]]:
    fname = "KVertexConnectedComponents"
    gr = _g(g, fname)
    k = _int(k, fname, 1, "k")
    if gr.directed or gr.mixed:
        raise NotImplementedError(f"{fname}: directed and mixed graphs are not supported")
    nb = _simple_nb(gr)
    if k == 1:
        if any(not s for s in nb):
            raise NotImplementedError(f"{fname}: graphs with isolated vertices are not supported for k = 1")
        comps = [list(c) for c in _components(gr)]
    elif k == 2:
        comps = _blocks(nb, gr.n)
        for c in comps:
            if len(c) > 2:
                sub, _keep = _sub_wg(gr, c)
                if _vconn(sub, fname) < 2:
                    raise ArithmeticError(f"{fname}: internal check failed (a block is not 2-connected)")
    else:
        raise NotImplementedError(f"{fname}: k >= 3 is not supported")
    comps = _component_filter(_sorted_comps(comps), gr, rest, fname)
    return L(*(_verts(gr, c) for c in comps))


@fn("FindPlanarColoring", area=AREA, status="added",
    notes="FindPlanarColoring[g], [g, {c1, c2, ...}] and [g, l]: a colouring of the faces of PlanarFaceList[g] (same "
          "order) with the minimum number of colours, adjacent faces (sharing an edge) coloured differently, as "
          "colours 1, 2, ... numbered by first use (or the given colours; [g, l] uses 1..l); edge directions are "
          "ignored. Same embedding caveats as PlanarFaceList; a valid optimal colouring may differ from Wolfram's. "
          "Graphs with a bridge (a face adjacent to itself) and multigraphs raise NotImplementedError; non-planar "
          "graphs raise ValueError; too few colours raise ValueError. Check: self-check (the face colouring is "
          "verified proper) and reference (wheels need 4 colours, the cube 3, by duality with known chromatic "
          "numbers).")
def _fpcol(g: Expr, colors: ListOf[Expr] | PosInteger = None) -> ListOf[Expr]:
    fname = "FindPlanarColoring"
    gr = _g(g, fname)
    if gr.loops() or WG(gr.vertices, [UE(gr.vertices[i], gr.vertices[j]) for i, j, _ in gr.edges]).multi():
        raise NotImplementedError(f"{fname}: multigraphs and self-loops are not supported")
    ug = WG(gr.vertices, [UE(gr.vertices[i], gr.vertices[j]) for i, j, _ in gr.edges], {}, fname)
    faces = _faces(ug, fname)
    side = {}
    for fi, f in enumerate(faces):
        for t in range(len(f)):
            side.setdefault((f[t], f[(t + 1) % len(f)]), fi)
    nb = [set() for _ in faces]
    for i, j, _ in ug.edges:
        a, b = side[(i, j)], side[(j, i)]
        if a == b:
            raise NotImplementedError(f"{fname}: graphs with a bridge (a face bordering itself) are not supported")
        nb[a].add(b)
        nb[b].add(a)
    cols = _min_coloring(nb, len(faces), fname)
    if any(cols[a] == cols[b] for a in range(len(faces)) for b in nb[a]):
        raise ArithmeticError(f"{fname}: internal check failed (the face colouring is not proper)")
    if colors is not None and not is_list(colors):
        m = _int(colors, fname, 1, "the number of colours")
        colors = L(*range(1, m + 1))
    return _apply_colors(cols, colors, fname)


@fn("EdgeCycleMatrix", area=AREA, status="added",
    notes="EdgeCycleMatrix[g]: one row per cycle of a cycle basis (the fundamental cycles of the breadth-first "
          "spanning forest, ignoring directions, one per non-tree edge in EdgeList order) and one column per edge "
          "in EdgeList order: 1 when the edge lies on the cycle (traversed along its direction for directed edges), "
          "-1 for a directed edge traversed against its direction, 0 otherwise; self-loops and parallel edges give "
          "cycles of length 1 and 2; an acyclic graph gives {}. A valid cycle basis may differ from Wolfram's "
          "rows. Check: self-check (every row is a closed cycle and the rows have rank m - n + c).")
def _ecm(g: Expr) -> MatrixOf[Integer]:
    fname = "EdgeCycleMatrix"
    gr = _g(g, fname)
    n, m = gr.n, len(gr.edges)
    adj = gr.out_adj(undirected=True)
    parent, pedge, depth = [None] * n, [None] * n, [None] * n
    tree = set()
    for r in range(n):
        if depth[r] is not None:
            continue
        depth[r] = 0
        q = deque([r])
        while q:
            u = q.popleft()
            for w, _, k in sorted(adj[u], key=lambda t: (t[0], t[2])):
                if depth[w] is None:
                    depth[w], parent[w], pedge[w] = depth[u] + 1, u, k
                    tree.add(k)
                    q.append(w)

    def coef(k, a, b):
        i, j, d = gr.edges[k]
        if not d:
            return 1
        return 1 if (i, j) == (a, b) else -1

    rows = []
    for k, (i, j, _) in enumerate(gr.edges):
        if k in tree:
            continue
        row = [0] * m
        row[k] = coef(k, i, j)
        if i != j:
            # walk the tree path j -> i, closing the cycle i -> j -> ... -> i
            a, b = j, i
            up_a, up_b = [], []
            while a != b:
                if depth[a] >= depth[b]:
                    up_a.append((pedge[a], a, parent[a]))
                    a = parent[a]
                else:
                    up_b.append((pedge[b], parent[b], b))
                    b = parent[b]
            for kk, x, y in up_a + list(reversed(up_b)):
                row[kk] = coef(kk, x, y)
        rows.append(row)
    c = len(_weak_components(gr)) if n else 0
    # self-check: each row is a closed walk (net +-1 balance zero at every vertex) and the rows are independent
    for row in rows:
        bal = [0] * n
        for kk, x in enumerate(row):
            if x:
                i, j, d = gr.edges[kk]
                a, b = (i, j) if (not d or x == 1) else (j, i)
                if not d:
                    # undirected edges: orientation fixed by the traversal; parity check suffices
                    bal[a] += 1
                    bal[b] += 1
                else:
                    bal[a] += 1
                    bal[b] -= 1
        if any(v % 2 for v in bal):
            raise ArithmeticError(f"{fname}: internal check failed (a row is not a cycle)")
    if len(rows) != m - n + c or (rows and sp.Matrix(rows).rank() != len(rows)):
        raise ArithmeticError(f"{fname}: internal check failed (the rows are not a cycle basis)")
    return L(*(L(*r) for r in rows))


@fn("GraphEmbedding", area=AREA, status="added",
    notes="GraphEmbedding[g]: the vertex coordinates stored in the graph's VertexCoordinates option ({{x, y}, ...} in "
          "VertexList order or {v -> {x, y}, ...} covering every vertex), in VertexList order. Computing a layout "
          "(no VertexCoordinates, or GraphEmbedding[g, \"emb\"] / [g, \"emb\", dim]) belongs to the graphics batch "
          "(W14) and raises NotImplementedError. Check: exact (tests read back the stored coordinates).")
def _gemb(g: Expr, *rest: String | PosInteger) -> MatrixOf[Real]:
    fname = "GraphEmbedding"
    gr = _g(g, fname)
    if rest:
        raise NotImplementedError(f"{fname}: computing a named embedding is not supported (graphics batch W14)")
    vc = gr.opts.get("VertexCoordinates")
    if vc is None or (isinstance(vc, Symbol) and vc.name == "Automatic"):
        raise NotImplementedError(f"{fname}: the graph has no VertexCoordinates; computing a layout is not "
                                  "supported (graphics batch W14)")
    if not is_list(vc):
        raise ValueError(f"{fname}: VertexCoordinates must be a list")
    if vc.args and all(_is_rule(r) for r in vc.args):
        got = {K.tag(r.args[0]): r.args[1] for r in vc.args}
        pts = []
        for v in gr.vertices:
            if K.tag(v) not in got:
                raise NotImplementedError(f"{fname}: VertexCoordinates gives no position for {v}; computing the "
                                          "missing positions is not supported")
            pts.append(got[K.tag(v)])
    else:
        pts = list(vc.args)
        if len(pts) != gr.n:
            raise ValueError(f"{fname}: VertexCoordinates must give one point per vertex")
    for p in pts:
        if not (is_list(p) and p.args and all(is_numeric_value(x) for x in p.args)):
            raise ValueError(f"{fname}: VertexCoordinates entries must be lists of numbers (got {p})")
    return L(*pts)


@fn("LinkRankCentrality", area=AREA, status="added",
    notes="LinkRankCentrality[g, alpha] and [g, alpha, beta] for directed graphs (multigraphs and self-loops "
          "allowed): for each edge u -> v in EdgeList order, the PageRank c(u) divided by the out-degree of u, where "
          "c solves c = alpha A^T D^-1 c + beta (beta = 1/n by default; a scalar beta means {beta, ...}) and is "
          "normalised to total 1, so the link ranks also total 1. Machine reals; WorkingPrecision -> Infinity gives "
          "exact values. Undirected and mixed graphs (whose edge convention Wolfram does not document) and graphs "
          "with a vertex of out-degree 0 raise NotImplementedError; alpha outside [0, 1) raises ValueError. "
          "Check: self-check (the PageRank equation is verified and the link ranks total 1) and reference (Wolfram "
          "documentation values for {1 -> 3, 2 -> 1, 3 -> 6, 4 -> 6, 1 -> 5, 5 -> 4, 6 -> 1} with alpha = 0.5).")
def _linkrank(g: Expr, alpha: Real, *rest: Real | ListOf[Real] | Option) -> ListOf[Real]:
    fname = "LinkRankCentrality"
    rest = list(rest)
    exact = False
    while rest and _is_rule(rest[-1]):
        r = rest.pop()
        nm = r.args[0].name if isinstance(r.args[0], Symbol) else None
        if nm != "WorkingPrecision":
            raise ValueError(f"{fname}: unknown option {r.args[0]}")
        val = r.args[1]
        if S(val) == sp.oo:
            exact = True
        elif isinstance(val, Symbol) and val.name == "MachinePrecision":
            exact = False
        else:
            raise NotImplementedError(f"{fname}: WorkingPrecision -> {val} is not supported (use MachinePrecision "
                                      "or Infinity)")
    if len(rest) > 1:
        raise TypeError(f"{fname}: too many arguments")
    gr = _g(g, fname)
    a = _real(alpha, fname, "the weight alpha")
    if not (0 <= a < 1):
        raise ValueError(f"{fname}: alpha must lie in [0, 1) (got {alpha})")
    if not gr.edges or not gr.directed:
        raise NotImplementedError(f"{fname}: only directed graphs are supported (Wolfram's convention for "
                                  "undirected edges is not documented)")
    n = gr.n
    if rest:
        b = rest[0]
        if is_list(b):
            if len(b.args) != n:
                raise ValueError(f"{fname}: beta must have one value per vertex")
            beta = [_real(x, fname, "beta") for x in b.args]
        else:
            beta = [_real(b, fname, "beta")] * n
    else:
        beta = [sp.Rational(1, n)] * n
    if not exact:
        a = sp.Float(a)
        beta = [sp.Float(x) for x in beta]
    out = [0] * n
    for i, j, _ in gr.edges:
        out[i] += 1
    if any(o == 0 for o in out):
        raise NotImplementedError(f"{fname}: graphs with a vertex of out-degree 0 are not supported")
    M = sp.zeros(n, n)
    for i, j, _ in gr.edges:
        M[j, i] += sp.Rational(1, out[i])
    lhs = sp.eye(n) - a * M
    if exact:
        c = lhs.LUsolve(sp.Matrix(beta))
    else:
        import numpy as np
        c = sp.Matrix(np.linalg.solve(np.array(lhs.tolist(), dtype=float), np.array(beta, dtype=float)).tolist())
    resid = lhs * c - sp.Matrix(beta)
    if exact and any(sp.simplify(x) != 0 for x in resid) or not exact and any(abs(float(x)) > 1e-9 for x in resid):
        raise ArithmeticError(f"{fname}: result failed verification (PageRank equation)")
    tot = sum(c)
    if tot == 0:
        raise ValueError(f"{fname}: the PageRank vector is zero (beta = 0)")
    c = [x / tot for x in c]
    lr = [c[i] / out[i] for i, j, _ in gr.edges]
    if exact:
        if sp.simplify(sum(lr) - 1) != 0:
            raise ArithmeticError(f"{fname}: result failed verification (link ranks do not total 1)")
        return L(*(R(sp.nsimplify(x)) for x in lr))
    if abs(float(sum(lr)) - 1) > 1e-9:
        raise ArithmeticError(f"{fname}: result failed verification (link ranks do not total 1)")
    return L(*(float(x) for x in lr))
