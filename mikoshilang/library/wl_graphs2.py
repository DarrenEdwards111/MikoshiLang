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
