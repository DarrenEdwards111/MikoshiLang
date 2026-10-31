"""Coverage batch 5: Wolfram Language graphs (Graph objects, construction, queries, paths, structure, centrality and
named graphs).

A graph is the inert expression ``Graph[{v1, v2, ...}, {e1, e2, ...}, opts...]`` with edges ``UndirectedEdge[u, v]``
or ``DirectedEdge[u, v]`` (Wolfram's InputForm of a Graph).  ``Graph[{1 -> 2, 2 <-> 3}]`` is normalised to that form:
``Rule`` becomes ``DirectedEdge`` and ``TwoWayRule`` becomes ``UndirectedEdge``; vertices are listed in order of
first appearance.  Every graph function also accepts a plain list of edges or rules, as Wolfram does.

Conventions (Wolfram's documented semantics):

* vertex order is ``VertexList[g]``; matrices and per-vertex results follow it;
* ``ConnectedComponents`` of a directed graph gives the strongly connected components;
* distances are exact for unweighted graphs and ``Infinity`` when unreachable;
* functions that find *one* object among several (FindCycle, FindPath, FindClique, FindHamiltonianCycle, ...) return
  a valid object of the requested kind, which may differ from the particular one Wolfram returns;
* inputs whose Wolfram semantics are not settled here (mixed graphs, tagged edges, unsupported options or graph
  kinds) raise a clear error instead of guessing.

Heavy algorithms that need NetworkX (planarity, isomorphism) raise a clear error when it is not installed.
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import Unevaluated, fn, record, REGISTRY, R, L, rule, S, number, is_list
from .wl_common import split_options as _split_options_rev, ev


def split_options(args, names, fname):
    """wl_common.split_options keeping the options in their given order (a Graph with several options must
    normalise to a fixed point; the reversed order made Graph[..., o1, o2] re-evaluate forever)."""
    pos, opts = _split_options_rev(args, names, fname)
    return pos, dict(reversed(list(opts.items())))

AREA = "graphs (batch 5)"
INF = sp.oo

# Options a Graph may carry.  Presentation options are kept on the expression and ignored by computations.
_PRESENTATION = {"VertexLabels", "EdgeLabels", "VertexStyle", "EdgeStyle", "GraphLayout", "ImageSize", "PlotLabel",
                 "VertexSize", "VertexShapeFunction", "EdgeShapeFunction", "VertexCoordinates", "PlotTheme",
                 "VertexLabelStyle", "EdgeLabelStyle", "GraphHighlight", "GraphHighlightStyle", "VertexShape",
                 "ImagePadding", "Background", "BaseStyle", "AspectRatio", "GraphStyle", "Frame", "FrameStyle",
                 "MultiedgeStyle", "SelfLoopStyle"}
_DATA = {"EdgeWeight", "EdgeCapacity", "VertexWeight", "EdgeCost", "VertexCapacity", "AnnotationRules"}
_GRAPH_OPTS = _PRESENTATION | _DATA


class GraphError(ValueError):
    pass


# Heads that understand tagged edges DirectedEdge[u, v, tag] / UndirectedEdge[u, v, tag] (W24).  Every other graph
# function keeps raising for tagged edges rather than silently dropping the tags.
TAG_AWARE = {"Graph", "GraphQ", "VertexList", "EdgeList", "VertexCount", "EdgeCount"}


def _is_edge(e):
    return isinstance(e, Expr) and e.head in ("UndirectedEdge", "DirectedEdge", "Rule", "TwoWayRule")


def _norm_edge(e, fname):
    if not _is_edge(e):
        if is_list(e) and len(e.args) in (2, 3):
            raise GraphError(f"{fname}: {e} is not an edge; use u <-> v (UndirectedEdge) or u -> v (DirectedEdge)")
        raise GraphError(f"{fname}: {e} is not a valid edge")
    if len(e.args) == 3 and e.head in ("DirectedEdge", "UndirectedEdge") and fname in TAG_AWARE:
        return e
    if len(e.args) != 2:
        raise GraphError(f"{fname}: tagged or malformed edge {e} is not supported")
    h = "DirectedEdge" if e.head in ("Rule", "DirectedEdge") else "UndirectedEdge"
    return Expr(h, e.args[0], e.args[1])


def _directed_edges(edges, de, fname):
    """Apply the DirectedEdges option: True makes every edge directed, False every edge undirected."""
    if isinstance(de, Symbol) and de.name == "Automatic":
        return list(edges)
    if de is not True and de is not False:
        raise GraphError(f"{fname}: DirectedEdges must be True, False or Automatic")
    h = "DirectedEdge" if de else "UndirectedEdge"
    out = []
    for e in edges:
        if not _is_edge(e):
            out.append(e)
            continue
        out.append(Expr(h, *e.args))
    return out


class WG:
    """Parsed graph: vertices (in VertexList order), edges as (i, j, directed) index triples, weights."""

    def __init__(self, vertices, edges, opts=None, fname="Graph"):
        self.fname = fname
        self.opts = dict(opts or {})
        self.vertices = []
        self.index = {}
        for v in vertices:
            t = K.tag(v)
            if t in self.index:
                raise GraphError(f"{fname}: vertex {v} is listed more than once")
            self.index[t] = len(self.vertices)
            self.vertices.append(v)
        self.edges = []          # (i, j, directed)
        self.edge_exprs = []
        for e in edges:
            e = _norm_edge(e, fname)
            u, v = e.args[0], e.args[1]
            for x in (u, v):
                if K.tag(x) not in self.index:
                    raise GraphError(f"{fname}: edge {e} uses {x}, which is not in the vertex list")
            self.edges.append((self.index[K.tag(u)], self.index[K.tag(v)], e.head == "DirectedEdge"))
            self.edge_exprs.append(e)
        self.weights = self._edge_data("EdgeWeight")
        self.capacities = self._edge_data("EdgeCapacity")
        vw = self.opts.get("VertexWeight")
        self.vertex_weights = None
        if vw is not None and not (isinstance(vw, Symbol) and vw.name == "Automatic"):
            if not is_list(vw) or len(vw.args) != len(self.vertices):
                raise GraphError(f"{fname}: VertexWeight must be a list with one weight per vertex")
            self.vertex_weights = list(vw.args)

    def _edge_data(self, name):
        w = self.opts.get(name)
        if w is None or (isinstance(w, Symbol) and w.name == "Automatic"):
            return None
        if not is_list(w) or len(w.args) != len(self.edges):
            raise GraphError(f"{self.fname}: {name} must be a list with one value per edge")
        out = []
        for x in w.args:
            v = S(x)
            if not (isinstance(v, sp.Basic) and v.is_number and v.is_real):
                raise GraphError(f"{self.fname}: {name} values must be real numbers (got {x})")
            if name == "EdgeWeight" and v < 0:
                raise GraphError(f"{self.fname}: negative edge weights are not supported")
            out.append(v)
        return out

    # ── kinds ──
    @property
    def n(self):
        return len(self.vertices)

    @property
    def directed(self):
        return bool(self.edges) and all(d for _, _, d in self.edges)

    @property
    def mixed(self):
        ds = {d for _, _, d in self.edges}
        return len(ds) == 2

    def require_pure(self, fname):
        if self.mixed:
            raise GraphError(f"{fname}: mixed graphs (directed and undirected edges together) are not supported")

    def require_undirected(self, fname):
        self.require_pure(fname)
        if self.directed:
            raise GraphError(f"{fname}: only undirected graphs are supported")

    def loops(self):
        return any(i == j for i, j, _ in self.edges)

    def multi(self):
        seen = set()
        for i, j, d in self.edges:
            k = (i, j) if d else (min(i, j), max(i, j))
            if k in seen:
                return True
            seen.add(k)
        return False

    def simple(self):
        return not self.loops() and not self.multi()

    def require_simple(self, fname):
        if not self.simple():
            raise GraphError(f"{fname}: only simple graphs (no self-loops or multiple edges) are supported")

    # ── adjacency ──
    def weight(self, k):
        return self.weights[k] if self.weights is not None else sp.Integer(1)

    def out_adj(self, undirected=False):
        """adj[i] = list of (j, weight, edge index) following edge direction (both ways when undirected=True)."""
        adj = [[] for _ in range(self.n)]
        for k, (i, j, d) in enumerate(self.edges):
            w = self.weight(k)
            adj[i].append((j, w, k))
            if (not d or undirected) and i != j:
                adj[j].append((i, w, k))
        return adj

    def in_adj(self):
        adj = [[] for _ in range(self.n)]
        for k, (i, j, d) in enumerate(self.edges):
            w = self.weight(k)
            adj[j].append((i, w, k))
            if not d and i != j:
                adj[i].append((j, w, k))
        return adj

    def neighbor_sets(self, undirected=False):
        return [set(j for j, _, _ in a if j != i) for i, a in enumerate(self.out_adj(undirected))]

    def vidx(self, v, fname):
        t = K.tag(v)
        if t not in self.index:
            raise GraphError(f"{fname}: {v} is not a vertex of the graph")
        return self.index[t]

    # ── output ──
    def expr(self, keep_opts=True):
        opts = [rule(Symbol(k), v) for k, v in self.opts.items()] if keep_opts else []
        return Expr("Graph", L(*self.vertices), L(*self.edge_exprs), *opts)

    def edge(self, i, j, directed):
        return Expr("DirectedEdge" if directed else "UndirectedEdge", self.vertices[i], self.vertices[j])

    # ── distances ──
    def sssp(self, s, undirected=False):
        """Single-source shortest distances (BFS when unweighted, Dijkstra otherwise) and predecessors."""
        adj = self.out_adj(undirected)
        dist = {s: sp.Integer(0)}
        prev = {}
        if self.weights is None:
            q = deque([s])
            while q:
                u = q.popleft()
                for v, _, _ in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        q.append(v)
            return dist, prev
        pq = [(0.0, s, s)]
        done = set()
        while pq:
            _, _, u = heapq.heappop(pq)
            if u in done:
                continue
            done.add(u)
            for v, w, _ in adj[u]:
                nd = dist[u] + w
                if v not in dist or nd < dist[v]:
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (float(nd), v, v))
        return dist, prev

    def distance_matrix(self):
        rows = []
        for i in range(self.n):
            d, _ = self.sssp(i)
            rows.append([d.get(j, INF) for j in range(self.n)])
        return rows

    def eccentricities(self):
        return [max(r) if r else sp.Integer(0) for r in self.distance_matrix()]


def _parse_graph_expr(g, fname):
    args = list(g.args)
    args, opts = split_options(args, _GRAPH_OPTS | {"DirectedEdges"}, fname)
    de = opts.pop("DirectedEdges", None)
    if de is not None:
        args = [L(*_directed_edges(a.args, de, fname)) if is_list(a) and k == len(args) - 1 else a
                for k, a in enumerate(args)]
    if len(args) == 1 and is_list(args[0]):
        edges = list(args[0].args)
        if edges and not all(_is_edge(e) for e in edges):
            # a vertex-only list with no edges is not a graph spec in Wolfram unless given as Graph[{v...}, {}]
            bad = next(e for e in edges if not _is_edge(e))
            raise GraphError(f"{fname}: {bad} is not an edge; use Graph[{{vertices}}, {{edges}}] for isolated vertices")
        verts = []
        seen = set()
        for e in edges:
            for x in e.args[:2]:
                if K.tag(x) not in seen:
                    seen.add(K.tag(x))
                    verts.append(x)
        return WG(verts, edges, opts, fname)
    if len(args) == 2 and is_list(args[0]) and is_list(args[1]):
        return WG(list(args[0].args), list(args[1].args), opts, fname)
    raise GraphError(f"{fname}: expected Graph[{{e1, e2, ...}}] or Graph[{{v1, ...}}, {{e1, ...}}]")


def graph_of(x, fname, opts=None):
    """WG for a Graph expression or a list of edges/rules; symbols leave the call unevaluated."""
    if isinstance(x, Expr) and x.head == "Graph":
        g = _parse_graph_expr(x, fname)
    elif is_list(x) and all(_is_edge(e) for e in x.args):
        g = _parse_graph_expr(Expr("Graph", x), fname)
    elif isinstance(x, Symbol) or (isinstance(x, Expr) and not is_list(x)):
        raise Unevaluated()
    else:
        raise GraphError(f"{fname}: {x} is not a graph")
    if opts:
        merged = dict(g.opts)
        merged.update(opts)
        g = WG(g.vertices, g.edge_exprs, merged, fname)
    return g


def is_graph_like(x):
    if isinstance(x, Expr) and x.head == "Graph":
        return True
    return is_list(x) and len(x.args) > 0 and all(_is_edge(e) for e in x.args)


def _opts_tail(args, names, fname):
    pos, opts = split_options(args, names, fname)
    return pos, opts


def _new(vertices, edges, opts=None):
    g = WG(vertices, edges, opts or {}, "Graph")
    return g.expr()


# ─────────────────────────── extending legacy heads ───────────────────────────

def extend(name, impl, accepts, notes, status="fixed"):
    """Route calls whose arguments satisfy ``accepts`` to ``impl``; other calls keep the legacy implementation."""
    e = REGISTRY.get(name)
    old = list(e.impls) if e is not None else []

    def f(*args):
        if accepts(args):
            return impl(*args)
        for g, lo, hi in old:
            if len(args) >= lo and (hi is None or len(args) <= hi):
                return g(*args)
        if not old:
            raise ValueError(f"{name}: invalid arguments")
        raise ValueError(f"{name} called with {len(args)} argument(s)")
    if e is None:
        fn(name, area=AREA, status=status, notes=notes)(f)
    else:
        e.impls = [(f, 0, None)]
        old_notes = e.notes
        e.notes = notes + ("; legacy: " + old_notes if old_notes else "")
        record(name, e.area, status, e.notes)
    return f


def _first_graph(args):
    return bool(args) and is_graph_like(args[0])


# ─────────────────────────── construction ───────────────────────────

@fn("Graph", area=AREA, status="added",
    notes="Graph[{e...}] and Graph[{v...}, {e...}] normalise to Graph[{v...}, {e...}, opts] (Rule -> DirectedEdge, "
          "TwoWayRule -> UndirectedEdge, vertices in order of first appearance); EdgeWeight/EdgeCapacity/VertexWeight "
          "options are used by the algorithms, presentation options are kept; unknown options, tagged edges and edges "
          "to unlisted vertices raise")
def _graph(*args):
    if not args:
        raise ValueError("Graph: expected a list of edges")
    if isinstance(args[0], Expr) and args[0].head == "Graph":
        if len(args) == 1:
            return args[0]
        g = graph_of(args[0], "Graph")
        pos, opts = split_options(list(args[1:]), _GRAPH_OPTS | {"DirectedEdges"}, "Graph")
        if pos:
            raise ValueError("Graph: invalid arguments")
        edges = g.edge_exprs
        if "DirectedEdges" in opts:
            edges = _directed_edges(edges, opts.pop("DirectedEdges"), "Graph")
        merged = dict(g.opts)
        merged.update(opts)
        return WG(g.vertices, edges, merged).expr()
    g = _parse_graph_expr(Expr("Graph", *args), "Graph")
    out = g.expr()
    return out


for _h in ("UndirectedEdge", "DirectedEdge", "TwoWayRule"):
    def _make(h):
        def impl(*args):
            if len(args) not in (2, 3):
                raise ValueError(f"{h} expects 2 arguments")
            raise Unevaluated()
        return impl
    fn(_h, area=AREA, status="added",
       notes="inert edge object (u <-> v is TwoWayRule, converted to UndirectedEdge inside Graph)")(_make(_h))


@fn("GraphQ", area=AREA, status="added", notes="True for a valid Graph expression")
def _graphq(x):
    if not (isinstance(x, Expr) and x.head == "Graph"):
        return False
    try:
        _parse_graph_expr(x, "GraphQ")
        return True
    except ValueError:
        return False


def _q(name, pred, notes):
    @fn(name, area=AREA, status="added", notes=notes)
    def impl(g):
        if not is_graph_like(g):
            if isinstance(g, Symbol):
                raise Unevaluated()
            return False
        return bool(pred(graph_of(g, name)))
    return impl


_q("DirectedGraphQ", lambda g: g.directed, "True when every edge is directed (an edgeless graph is undirected)")
_q("UndirectedGraphQ", lambda g: not g.directed and not g.mixed, "True when every edge is undirected")
_q("MixedGraphQ", lambda g: g.mixed, "True when the graph has both directed and undirected edges")
_q("SimpleGraphQ", lambda g: g.simple() and not g.mixed and not g.directed,
   "undirected, without self-loops or multiple edges")
_q("LoopFreeGraphQ", lambda g: not g.loops(), "no self-loops")
_q("MultigraphQ", lambda g: g.multi(), "has multiple edges between some pair of vertices")
_q("EmptyGraphQ", lambda g: not g.edges, "True when the graph has no edges")
_q("EdgeWeightedGraphQ", lambda g: g.weights is not None, "True when the graph carries EdgeWeight")
_q("VertexWeightedGraphQ", lambda g: g.vertex_weights is not None, "True when the graph carries VertexWeight")
_q("WeightedGraphQ", lambda g: g.weights is not None or g.vertex_weights is not None,
   "True when the graph carries edge or vertex weights")


@fn("CompleteGraphQ", area=AREA, status="added",
    notes="simple undirected graph with every pair of distinct vertices adjacent (directed: every ordered pair)")
def _completeq(g):
    if not is_graph_like(g):
        if isinstance(g, Symbol):
            raise Unevaluated()
        return False
    gr = graph_of(g, "CompleteGraphQ")
    if gr.mixed or gr.loops() or gr.multi():
        return False
    n = gr.n
    m = len(gr.edges)
    return m == (n * (n - 1) if gr.directed else n * (n - 1) // 2)


@fn("VertexList", area=AREA, status="added", notes="vertices in Graph order")
def _vlist(g):
    return L(*graph_of(g, "VertexList").vertices)


@fn("EdgeList", area=AREA, status="added", notes="edges in Graph order (Rule given as DirectedEdge)")
def _elist(g):
    return L(*graph_of(g, "EdgeList").edge_exprs)


@fn("VertexCount", area=AREA, status="added", notes="number of vertices")
def _vcount(g):
    return graph_of(g, "VertexCount").n


@fn("EdgeCount", area=AREA, status="added", notes="number of edges (multi-edges and self-loops counted)")
def _ecount(g):
    return len(graph_of(g, "EdgeCount").edges)


@fn("VertexQ", area=AREA, status="added", notes="True when v is a vertex of g")
def _vertexq(g, v):
    gr = graph_of(g, "VertexQ")
    return K.tag(v) in gr.index


def _edge_match(gr, e):
    if not _is_edge(e) or len(e.args) != 2:
        return False
    t = K.tag
    if K.tag(e.args[0]) not in gr.index or K.tag(e.args[1]) not in gr.index:
        return False
    i, j = gr.index[t(e.args[0])], gr.index[t(e.args[1])]
    directed = e.head in ("Rule", "DirectedEdge")
    for a, b, d in gr.edges:
        if d != directed:
            continue
        if (a, b) == (i, j) or (not d and (b, a) == (i, j)):
            return True
    return False


@fn("EdgeQ", area=AREA, status="added",
    notes="True when e is an edge of g (an undirected edge matches either orientation)")
def _edgeq(g, e):
    return _edge_match(graph_of(g, "EdgeQ"), e)


@fn("VertexIndex", area=AREA, status="added", notes="position of v in VertexList[g]; unknown vertices raise")
def _vindex(g, v):
    gr = graph_of(g, "VertexIndex")
    return gr.vidx(v, "VertexIndex") + 1


@fn("EdgeIndex", area=AREA, status="added", notes="position of e in EdgeList[g]; unknown edges raise")
def _eindex(g, e):
    gr = graph_of(g, "EdgeIndex")
    ne = _norm_edge(e, "EdgeIndex")
    i, j = gr.vidx(ne.args[0], "EdgeIndex"), gr.vidx(ne.args[1], "EdgeIndex")
    d = ne.head == "DirectedEdge"
    for k, (a, b, dd) in enumerate(gr.edges):
        if dd == d and ((a, b) == (i, j) or (not d and (b, a) == (i, j))):
            return k + 1
    raise ValueError(f"EdgeIndex: {e} is not an edge of the graph")


def _degree_lists(gr):
    deg = [0] * gr.n
    ind = [0] * gr.n
    outd = [0] * gr.n
    for i, j, d in gr.edges:
        deg[i] += 1
        deg[j] += 1
        if d:
            outd[i] += 1
            ind[j] += 1
        else:
            outd[i] += 1
            ind[j] += 1
            if i != j:
                outd[j] += 1
                ind[i] += 1
            else:
                outd[i] += 1
                ind[i] += 1
    return deg, ind, outd


def _vertex_degree(g, v=None):
    gr = graph_of(g, "VertexDegree")
    deg, _, _ = _degree_lists(gr)
    if v is None:
        return L(*deg)
    return deg[gr.vidx(v, "VertexDegree")]


extend("VertexDegree", _vertex_degree, _first_graph,
       "VertexDegree[g] = degrees in VertexList order, VertexDegree[g, v]; self-loops count 2, multi-edges counted; "
       "directed: in + out")
_degree_centrality_doc = "DegreeCentrality[g] (= VertexDegree), [g, \"In\"], [g, \"Out\"]"


def _inout(name, which):
    @fn(name, area=AREA, status="added",
        notes=f"{which}-degrees in VertexList order or of one vertex; for undirected edges in- and out-degree equal "
              "the degree")
    def impl(g, v=None):
        gr = graph_of(g, name)
        _, ind, outd = _degree_lists(gr)
        d = ind if which == "in" else outd
        if v is None:
            return L(*d)
        return d[gr.vidx(v, name)]
    return impl


_inout("VertexInDegree", "in")
_inout("VertexOutDegree", "out")


@fn("DegreeCentrality", area=AREA, status="added", notes=_degree_centrality_doc)
def _degc(g, kind=None):
    gr = graph_of(g, "DegreeCentrality")
    deg, ind, outd = _degree_lists(gr)
    if kind is None:
        return L(*deg)
    if kind == "In":
        return L(*ind)
    if kind == "Out":
        return L(*outd)
    raise ValueError('DegreeCentrality: the second argument must be "In" or "Out"')


@fn("AdjacencyMatrix", area=AREA, status="added",
    notes="dense form (Normal) of Wolfram's SparseArray: a_ij = number of edges i -> j, an undirected edge counts both "
          "ways, a self-loop adds 1 on the diagonal; the empty graph raises")
def _adjm(g):
    gr = graph_of(g, "AdjacencyMatrix")
    if gr.n == 0:
        raise ValueError("AdjacencyMatrix: the empty graph has no adjacency matrix")
    m = [[0] * gr.n for _ in range(gr.n)]
    for i, j, d in gr.edges:
        m[i][j] += 1
        if not d and i != j:
            m[j][i] += 1
    return R(m)


@fn("WeightedAdjacencyMatrix", area=AREA, status="added",
    notes="a_ij = weight of the edge i -> j (0 when absent); multigraphs raise")
def _wadjm(g):
    gr = graph_of(g, "WeightedAdjacencyMatrix")
    if gr.n == 0:
        raise ValueError("WeightedAdjacencyMatrix: the empty graph has no adjacency matrix")
    if gr.multi():
        raise ValueError("WeightedAdjacencyMatrix: multigraphs are not supported")
    m = [[sp.Integer(0)] * gr.n for _ in range(gr.n)]
    for k, (i, j, d) in enumerate(gr.edges):
        w = gr.weight(k)
        m[i][j] = w
        if not d:
            m[j][i] = w
    return R(m)


@fn("IncidenceMatrix", area=AREA, status="added",
    notes="vertices x edges: undirected 1 at each end (2 for a self-loop); directed -1 at the tail and 1 at the head; "
          "directed self-loops raise")
def _incm(g):
    gr = graph_of(g, "IncidenceMatrix")
    m = [[0] * len(gr.edges) for _ in range(gr.n)]
    for k, (i, j, d) in enumerate(gr.edges):
        if d:
            if i == j:
                raise ValueError("IncidenceMatrix: directed self-loops are not supported")
            m[i][k] = -1
            m[j][k] = 1
        else:
            m[i][k] += 1
            m[j][k] += 1
    if not gr.edges:
        raise ValueError("IncidenceMatrix: the graph has no edges")
    return R(m)


@fn("KirchhoffMatrix", area=AREA, status="added",
    notes="Laplacian D - A of an undirected loop-free graph (multi-edges counted); directed graphs and self-loops "
          "raise")
def _kirch(g):
    gr = graph_of(g, "KirchhoffMatrix")
    gr.require_undirected("KirchhoffMatrix")
    if gr.loops():
        raise ValueError("KirchhoffMatrix: graphs with self-loops are not supported")
    if gr.n == 0:
        raise ValueError("KirchhoffMatrix: the empty graph has no Kirchhoff matrix")
    m = [[0] * gr.n for _ in range(gr.n)]
    for i, j, _ in gr.edges:
        m[i][j] -= 1
        m[j][i] -= 1
        m[i][i] += 1
        m[j][j] += 1
    return R(m)


def _matrix_ints(m, fname):
    if not is_list(m) or not m.args or not all(is_list(r) for r in m.args):
        raise ValueError(f"{fname}: expected a square matrix")
    n = len(m.args)
    rows = []
    for r in m.args:
        if len(r.args) != n:
            raise ValueError(f"{fname}: expected a square matrix")
        row = []
        for x in r.args:
            if isinstance(x, bool) or not isinstance(x, int) or x < 0:
                raise ValueError(f"{fname}: entries must be non-negative integers (edge counts)")
            row.append(x)
        rows.append(row)
    return rows


def _adjacency_graph(vs, m, fname):
    n = len(m)
    sym = all(m[i][j] == m[j][i] for i in range(n) for j in range(n))
    edges = []
    for i in range(n):
        for j in range(n):
            if sym:
                if j < i:
                    continue
                c = m[i][j]
                edges += [Expr("UndirectedEdge", vs[i], vs[j])] * c
            else:
                edges += [Expr("DirectedEdge", vs[i], vs[j])] * m[i][j]
    return _new(vs, edges)


@fn("AdjacencyGraph", area=AREA, status="added",
    notes="graph from an adjacency matrix (undirected when symmetric, else directed; entries > 1 give multi-edges, "
          "diagonal entries self-loops); AdjacencyGraph[{v...}, m] names the vertices")
def _adjg(a, b=None):
    if b is None:
        m = _matrix_ints(a, "AdjacencyGraph")
        return _adjacency_graph(list(range(1, len(m) + 1)), m, "AdjacencyGraph")
    if not is_list(a):
        raise ValueError("AdjacencyGraph: the vertex list must be a list")
    m = _matrix_ints(b, "AdjacencyGraph")
    if len(a.args) != len(m):
        raise ValueError("AdjacencyGraph: the vertex list and matrix sizes differ")
    return _adjacency_graph(list(a.args), m, "AdjacencyGraph")


@fn("AdjacencyList", area=AREA, status="added",
    notes="AdjacencyList[g, v] = vertices adjacent to v (either direction), in VertexList order; AdjacencyList[g] "
          "for every vertex")
def _adjl(g, v=None):
    gr = graph_of(g, "AdjacencyList")
    nb = gr.neighbor_sets(undirected=True)
    if v is None:
        return L(*(L(*(gr.vertices[j] for j in sorted(s))) for s in nb))
    i = gr.vidx(v, "AdjacencyList")
    return L(*(gr.vertices[j] for j in sorted(nb[i])))


@fn("EdgeWeight", area=AREA, status="added", notes="Graph option; EdgeWeight[g] is not supported (use AnnotationValue)")
def _ew(*args):
    raise Unevaluated()


@fn("EdgeCapacity", area=AREA, status="added", notes="Graph option for FindMaximumFlow capacities")
def _ec(*args):
    raise Unevaluated()


@fn("VertexWeight", area=AREA, status="added", notes="Graph option for vertex weights")
def _vw(*args):
    raise Unevaluated()


# ─────────────────────────── editing ───────────────────────────

def _as_vertex_list(x):
    return list(x.args) if is_list(x) else [x]


@fn("VertexAdd", area=AREA, status="added", notes="adds new vertices (existing ones are ignored)")
def _vadd(g, v):
    gr = graph_of(g, "VertexAdd")
    vs = list(gr.vertices)
    seen = set(gr.index)
    for x in _as_vertex_list(v):
        if K.tag(x) not in seen:
            seen.add(K.tag(x))
            vs.append(x)
    return _new(vs, gr.edge_exprs)


@fn("VertexDelete", area=AREA, status="added", notes="removes the vertices and their incident edges")
def _vdel(g, v):
    gr = graph_of(g, "VertexDelete")
    drop = {K.tag(x) for x in _as_vertex_list(v)}
    vs = [x for x in gr.vertices if K.tag(x) not in drop]
    es = [e for e in gr.edge_exprs if K.tag(e.args[0]) not in drop and K.tag(e.args[1]) not in drop]
    return _new(vs, es)


@fn("EdgeAdd", area=AREA, status="added", notes="adds edges (and any new vertices); weights are not carried over")
def _eadd(g, e):
    gr = graph_of(g, "EdgeAdd")
    if gr.weights is not None:
        raise ValueError("EdgeAdd: edge-weighted graphs are not supported")
    new = [_norm_edge(x, "EdgeAdd") for x in (e.args if is_list(e) else [e])]
    vs = list(gr.vertices)
    seen = set(gr.index)
    for x in new:
        for v in x.args:
            if K.tag(v) not in seen:
                seen.add(K.tag(v))
                vs.append(v)
    return _new(vs, list(gr.edge_exprs) + new)


@fn("EdgeDelete", area=AREA, status="added",
    notes="removes every edge matching the given edges (undirected edges match either orientation)")
def _edel(g, e):
    gr = graph_of(g, "EdgeDelete")
    if gr.weights is not None:
        raise ValueError("EdgeDelete: edge-weighted graphs are not supported")
    dels = [_norm_edge(x, "EdgeDelete") for x in (e.args if is_list(e) else [e])]
    keep = []
    for k, ex in enumerate(gr.edge_exprs):
        i, j, d = gr.edges[k]
        hit = False
        for x in dels:
            if K.tag(x.args[0]) not in gr.index or K.tag(x.args[1]) not in gr.index:
                continue
            a, b = gr.index[K.tag(x.args[0])], gr.index[K.tag(x.args[1])]
            xd = x.head == "DirectedEdge"
            if xd == d and ((a, b) == (i, j) or (not d and (b, a) == (i, j))):
                hit = True
        if not hit:
            keep.append(ex)
    return _new(gr.vertices, keep)


@fn("Subgraph", area=AREA, status="added",
    notes="Subgraph[g, {v...}] = subgraph induced by the vertices (VertexList order kept); edges in the list are "
          "included with their endpoints")
def _subgraph(g, spec):
    gr = graph_of(g, "Subgraph")
    items = _as_vertex_list(spec)
    vs_t, extra_edges = set(), []
    for x in items:
        if _is_edge(x):
            ne = _norm_edge(x, "Subgraph")
            if not _edge_match(gr, ne):
                raise ValueError(f"Subgraph: {x} is not an edge of the graph")
            extra_edges.append(ne)
        else:
            gr.vidx(x, "Subgraph")
            vs_t.add(K.tag(x))
    if extra_edges:
        if vs_t:
            raise ValueError("Subgraph: give either vertices or edges")
        for ne in extra_edges:
            vs_t.update(K.tag(v) for v in ne.args)
        vs = [v for v in gr.vertices if K.tag(v) in vs_t]
        return _new(vs, extra_edges)
    vs = [v for v in gr.vertices if K.tag(v) in vs_t]
    es = [e for e in gr.edge_exprs if K.tag(e.args[0]) in vs_t and K.tag(e.args[1]) in vs_t]
    return _new(vs, es)


@fn("NeighborhoodGraph", area=AREA, status="added",
    notes="subgraph induced by the vertices within distance k (default 1) of v (undirected graphs; edges followed "
          "both ways for directed graphs)")
def _nbg(g, v, k=1):
    gr = graph_of(g, "NeighborhoodGraph")
    gr.require_pure("NeighborhoodGraph")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("NeighborhoodGraph: k must be a non-negative integer")
    srcs = [gr.vidx(x, "NeighborhoodGraph") for x in _as_vertex_list(v)]
    adj = gr.out_adj(undirected=True)
    reach = set()
    for s in srcs:
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] == k:
                continue
            for w, _, _ in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    q.append(w)
        reach.update(dist)
    tags = {K.tag(gr.vertices[i]) for i in reach}
    vs = [x for x in gr.vertices if K.tag(x) in tags]
    es = [e for e in gr.edge_exprs if K.tag(e.args[0]) in tags and K.tag(e.args[1]) in tags]
    return _new(vs, es)


def _edge_key(gr, i, j, d):
    return (gr.vertices[i], gr.vertices[j], d)


@fn("GraphComplement", area=AREA, status="added",
    notes="same vertices; undirected: u <-> v (VertexList order) for every non-adjacent pair; directed: every missing "
          "u -> v; requires a simple (loop-free, no multi-edge) graph")
def _gcomp(g):
    gr = graph_of(g, "GraphComplement")
    gr.require_pure("GraphComplement")
    if gr.loops() or gr.multi():
        raise ValueError("GraphComplement: only simple graphs are supported")
    d = gr.directed
    have = set()
    for i, j, _ in gr.edges:
        have.add((i, j))
        if not d:
            have.add((j, i))
    es = []
    for i in range(gr.n):
        for j in range(gr.n):
            if i == j or (not d and j < i) or (i, j) in have:
                continue
            es.append(gr.edge(i, j, d))
    return _new(gr.vertices, es)


def _edge_tags(gr):
    out = []
    for i, j, d in gr.edges:
        u, v = K.tag(gr.vertices[i]), K.tag(gr.vertices[j])
        out.append((u, v, True) if d else (tuple(sorted((u, v), key=repr)) + (False,)))
    return out


def _setop(name, op):
    @fn(name, area=AREA, status="added",
        notes={"GraphUnion": "vertices and edges of all the graphs (first occurrence order); multigraphs and weights raise",
               "GraphIntersection": "common vertices and common edges (order of the first graph); multigraphs raise",
               "GraphDifference": "vertices of g1 with the edges of g1 that are not in g2; multigraphs raise"}[name])
    def impl(*gs):
        if not gs:
            raise ValueError(f"{name}: expected graphs")
        if name == "GraphDifference" and len(gs) != 2:
            raise ValueError("GraphDifference expects two graphs")
        grs = [graph_of(x, name) for x in gs]
        for gr in grs:
            if gr.multi():
                raise ValueError(f"{name}: multigraphs are not supported")
            if gr.weights is not None:
                raise ValueError(f"{name}: edge-weighted graphs are not supported")
        if op == "union":
            vs, seen = [], set()
            es, eseen = [], set()
            for gr in grs:
                for v in gr.vertices:
                    if K.tag(v) not in seen:
                        seen.add(K.tag(v))
                        vs.append(v)
                for ex, t in zip(gr.edge_exprs, _edge_tags(gr)):
                    if t not in eseen:
                        eseen.add(t)
                        es.append(ex)
            return _new(vs, es)
        first = grs[0]
        if op == "intersection":
            vt = set(first.index)
            et = set(_edge_tags(first))
            for gr in grs[1:]:
                vt &= set(gr.index)
                et &= set(_edge_tags(gr))
            vs = [v for v in first.vertices if K.tag(v) in vt]
            es = [ex for ex, t in zip(first.edge_exprs, _edge_tags(first)) if t in et]
            return _new(vs, es)
        et2 = set(_edge_tags(grs[1]))
        es = [ex for ex, t in zip(first.edge_exprs, _edge_tags(first)) if t not in et2]
        return _new(first.vertices, es)
    return impl


_setop("GraphUnion", "union")
_setop("GraphIntersection", "intersection")
_setop("GraphDifference", "difference")


@fn("GraphDisjointUnion", area=AREA, status="added",
    notes="disjoint union with vertices relabelled 1, 2, ... in order (Wolfram's relabelling)")
def _gdu(*gs):
    if not gs:
        raise ValueError("GraphDisjointUnion: expected graphs")
    vs, es, off = [], [], 0
    for x in gs:
        gr = graph_of(x, "GraphDisjointUnion")
        gr.require_pure("GraphDisjointUnion")
        vs += list(range(off + 1, off + gr.n + 1))
        for i, j, d in gr.edges:
            es.append(Expr("DirectedEdge" if d else "UndirectedEdge", off + i + 1, off + j + 1))
        off += gr.n
    return _new(vs, es)


@fn("UndirectedGraph", area=AREA, status="added",
    notes="every edge made undirected; for directed input, u -> v and v -> u become one edge (duplicates merged)")
def _undg(g):
    gr = graph_of(g, "UndirectedGraph")
    if gr.weights is not None:
        raise ValueError("UndirectedGraph: edge-weighted graphs are not supported")
    es, seen = [], set()
    for i, j, d in gr.edges:
        if d:
            k = (min(i, j), max(i, j))
            if k in seen:
                continue
            seen.add(k)
        es.append(Expr("UndirectedEdge", gr.vertices[i], gr.vertices[j]))
    return _new(gr.vertices, es)


@fn("DirectedGraph", area=AREA, status="added",
    notes="each undirected edge u <-> v becomes the pair u -> v, v -> u (Wolfram's default conversion)")
def _dirg(g, how=None):
    gr = graph_of(g, "DirectedGraph")
    if how is not None:
        raise ValueError("DirectedGraph: conversion methods are not supported")
    if gr.weights is not None:
        raise ValueError("DirectedGraph: edge-weighted graphs are not supported")
    es = []
    for i, j, d in gr.edges:
        es.append(Expr("DirectedEdge", gr.vertices[i], gr.vertices[j]))
        if not d and i != j:
            es.append(Expr("DirectedEdge", gr.vertices[j], gr.vertices[i]))
    return _new(gr.vertices, es)


@fn("ReverseGraph", area=AREA, status="added", notes="reverses every directed edge; undirected edges unchanged")
def _revg(g):
    gr = graph_of(g, "ReverseGraph")
    es = [Expr(e.head, e.args[1], e.args[0]) if e.head == "DirectedEdge" else e for e in gr.edge_exprs]
    return WG(gr.vertices, es, gr.opts).expr()


@fn("IndexGraph", area=AREA, status="added", notes="vertices renamed 1, 2, ... in VertexList order (IndexGraph[g, r] starts at r)")
def _idxg(g, r=1):
    gr = graph_of(g, "IndexGraph")
    if isinstance(r, bool) or not isinstance(r, int):
        raise ValueError("IndexGraph: the start must be an integer")
    vs = list(range(r, r + gr.n))
    es = [Expr("DirectedEdge" if d else "UndirectedEdge", r + i, r + j) for i, j, d in gr.edges]
    return WG(vs, es, gr.opts).expr()


@fn("LineGraph", area=AREA, status="added",
    notes="undirected simple graphs: vertices 1..m for the edges, i <-> j when edges i and j share an endpoint")
def _lineg(g):
    gr = graph_of(g, "LineGraph")
    gr.require_undirected("LineGraph")
    gr.require_simple("LineGraph")
    m = len(gr.edges)
    es = []
    for a in range(m):
        for b in range(a + 1, m):
            if set(gr.edges[a][:2]) & set(gr.edges[b][:2]):
                es.append(Expr("UndirectedEdge", a + 1, b + 1))
    return _new(list(range(1, m + 1)), es)


# ─────────────────────────── paths and distances ───────────────────────────

@fn("FindShortestPath", area=AREA, status="added",
    notes="vertex list of a shortest path (BFS, or Dijkstra with EdgeWeight), {} when t is unreachable; unknown "
          "vertices raise")
def _fsp(g, s, t):
    gr = graph_of(g, "FindShortestPath")
    gr.require_pure("FindShortestPath")
    a, b = gr.vidx(s, "FindShortestPath"), gr.vidx(t, "FindShortestPath")
    dist, prev = gr.sssp(a)
    if b not in dist:
        return L()
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return L(*(gr.vertices[i] for i in reversed(path)))


@fn("GraphDistance", area=AREA, status="added",
    notes="GraphDistance[g, s, t] shortest distance (Infinity when unreachable); GraphDistance[g, s] to every vertex")
def _gdist(g, s, t=None):
    gr = graph_of(g, "GraphDistance")
    gr.require_pure("GraphDistance")
    a = gr.vidx(s, "GraphDistance")
    dist, _ = gr.sssp(a)
    if t is None:
        return L(*(R(dist.get(j, INF)) for j in range(gr.n)))
    return R(dist.get(gr.vidx(t, "GraphDistance"), INF))


@fn("GraphDistanceMatrix", area=AREA, status="added",
    notes="matrix of shortest distances in VertexList order (Infinity when unreachable)")
def _gdm(g):
    gr = graph_of(g, "GraphDistanceMatrix")
    gr.require_pure("GraphDistanceMatrix")
    if gr.n == 0:
        raise ValueError("GraphDistanceMatrix: the graph has no vertices")
    return R(gr.distance_matrix())


@fn("VertexEccentricity", area=AREA, status="added",
    notes="maximum distance from v to the other vertices; Infinity when some vertex is unreachable")
def _vecc(g, v):
    gr = graph_of(g, "VertexEccentricity")
    gr.require_pure("VertexEccentricity")
    i = gr.vidx(v, "VertexEccentricity")
    d, _ = gr.sssp(i)
    return R(max(d.get(j, INF) for j in range(gr.n)))


def _ecc_fn(name, which):
    def impl(g):
        gr = graph_of(g, name)
        gr.require_pure(name)
        if gr.n == 0:
            raise ValueError(f"{name}: the graph has no vertices")
        e = gr.eccentricities()
        if which == "diameter":
            return R(max(e))
        if which == "radius":
            return R(min(e))
        if any(x == INF for x in e):
            raise ValueError(f"{name}: the graph is not connected")
        target = min(e) if which == "center" else max(e)
        return L(*(gr.vertices[i] for i in range(gr.n) if e[i] == target))
    return impl


extend("GraphDiameter", _ecc_fn("GraphDiameter", "diameter"), _first_graph,
       "maximum eccentricity (Infinity when not (strongly) connected); weighted with EdgeWeight")
extend("GraphRadius", _ecc_fn("GraphRadius", "radius"), _first_graph,
       "minimum eccentricity (Infinity when not (strongly) connected)")
extend("GraphCenter", _ecc_fn("GraphCenter", "center"), _first_graph,
       "vertices of minimum eccentricity; disconnected graphs raise")
fn("GraphPeriphery", area=AREA, status="added",
   notes="vertices of maximum eccentricity; disconnected graphs raise")(_ecc_fn("GraphPeriphery", "periphery"))


@fn("FindPath", area=AREA, status="added",
    notes="FindPath[g, s, t] = {path} with one simple path ({} when none); FindPath[g, s, t, k] paths of length at "
          "most k, [g, s, t, {k}] exactly k, [g, s, t, kspec, n] at most n paths (All for every path); paths are "
          "enumerated shortest first; unweighted lengths")
def _fpath(g, s, t, kspec=None, nspec=1):
    gr = graph_of(g, "FindPath")
    gr.require_pure("FindPath")
    if gr.weights is not None:
        raise ValueError("FindPath: weighted path lengths are not supported")
    a, b = gr.vidx(s, "FindPath"), gr.vidx(t, "FindPath")
    lo, hi = _len_spec(kspec, gr.n, "FindPath")
    limit = _count_spec(nspec, "FindPath")
    adj = gr.out_adj()
    found = []

    def dfs(path, onpath):
        u = path[-1]
        if len(path) - 1 > hi:
            return
        if u == b and len(path) > 1 or (u == b and a == b):
            if lo <= len(path) - 1 <= hi:
                found.append(list(path))
            return
        for w, _, _ in sorted(adj[u], key=lambda x: x[0]):
            if w not in onpath:
                onpath.add(w)
                path.append(w)
                dfs(path, onpath)
                path.pop()
                onpath.discard(w)
    if a == b:
        found = [[a]] if lo == 0 else []
    else:
        dfs([a], {a})
    uniq, seen = [], set()
    for p in sorted(found, key=lambda p: (len(p), p)):
        if tuple(p) not in seen:
            seen.add(tuple(p))
            uniq.append(p)
    if limit is not None:
        uniq = uniq[:limit]
    return L(*(L(*(gr.vertices[i] for i in p)) for p in uniq))


def _len_spec(k, n, fname):
    if k is None:
        return 0, max(n - 1, 0)
    if isinstance(k, Symbol) and k.name in ("Infinity", "All"):
        return 0, max(n, 0)
    if isinstance(k, int) and not isinstance(k, bool) and k >= 0:
        return 0, k
    if is_list(k) and len(k.args) == 1 and isinstance(k.args[0], int):
        return k.args[0], k.args[0]
    if is_list(k) and len(k.args) == 2 and all(isinstance(x, int) for x in k.args):
        return k.args[0], k.args[1]
    raise ValueError(f"{fname}: invalid length specification {k}")


def _count_spec(nspec, fname):
    if isinstance(nspec, Symbol) and nspec.name in ("All", "Infinity"):
        return None
    if isinstance(nspec, int) and not isinstance(nspec, bool) and nspec >= 0:
        return nspec
    raise ValueError(f"{fname}: the number of results must be a non-negative integer or All")


# ─────────────────────────── connectivity ───────────────────────────

def _weak_components(gr):
    adj = gr.out_adj(undirected=True)
    seen, comps = set(), []
    for s in range(gr.n):
        if s in seen:
            continue
        comp, q = [], deque([s])
        seen.add(s)
        while q:
            u = q.popleft()
            comp.append(u)
            for w, _, _ in adj[u]:
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        comps.append(sorted(comp))
    return comps


def _strong_components(gr):
    adj = [[j for j, _, _ in a] for a in gr.out_adj()]
    index, low, onstack, stack, comps = {}, {}, set(), [], []
    counter = [0]

    def strong(v):
        work = [(v, iter(adj[v]))]
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        onstack.add(v)
        while work:
            u, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    onstack.add(w)
                    work.append((w, iter(adj[w])))
                    advanced = True
                    break
                elif w in onstack:
                    low[u] = min(low[u], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[u])
            if low[u] == index[u]:
                comp = []
                while True:
                    w = stack.pop()
                    onstack.discard(w)
                    comp.append(w)
                    if w == u:
                        break
                comps.append(sorted(comp))
    for v in range(gr.n):
        if v not in index:
            strong(v)
    return comps


def _components(gr):
    gr.require_pure("ConnectedComponents")
    comps = _strong_components(gr) if gr.directed else _weak_components(gr)
    comps.sort(key=lambda c: (-len(c), c[0]))
    return comps


def _cc(g):
    gr = graph_of(g, "ConnectedComponents")
    return L(*(L(*(gr.vertices[i] for i in c)) for c in _components(gr)))


extend("ConnectedComponents", _cc, _first_graph,
       "Graph or edge-rule input: connected components (strongly connected for directed graphs), largest first, "
       "vertices in VertexList order")


@fn("WeaklyConnectedComponents", area=AREA, status="added",
    notes="components ignoring edge direction, largest first, vertices in VertexList order")
def _wcc(g):
    gr = graph_of(g, "WeaklyConnectedComponents")
    comps = _weak_components(gr)
    comps.sort(key=lambda c: (-len(c), c[0]))
    return L(*(L(*(gr.vertices[i] for i in c)) for c in comps))


@fn("ConnectedGraphComponents", area=AREA, status="added", notes="Subgraph for each ConnectedComponents element")
def _cgc(g):
    gr = graph_of(g, "ConnectedGraphComponents")
    out = []
    for c in _components(gr):
        out.append(_subgraph(g, L(*(gr.vertices[i] for i in c))))
    return L(*out)


@fn("ConnectedGraphQ", area=AREA, status="added",
    notes="True for a connected graph (strongly connected when directed); the empty graph is not connected")
def _connq(g):
    if not is_graph_like(g):
        if isinstance(g, Symbol):
            raise Unevaluated()
        return False
    gr = graph_of(g, "ConnectedGraphQ")
    gr.require_pure("ConnectedGraphQ")
    if gr.n == 0:
        return False
    return len(_components(gr)) == 1


@fn("WeaklyConnectedGraphQ", area=AREA, status="added", notes="connected when edge directions are ignored")
def _wconnq(g):
    if not is_graph_like(g):
        if isinstance(g, Symbol):
            raise Unevaluated()
        return False
    gr = graph_of(g, "WeaklyConnectedGraphQ")
    if gr.n == 0:
        return False
    return len(_weak_components(gr)) == 1


def _has_cycle(gr):
    if gr.directed:
        return len(_topo(gr)) != gr.n
    # undirected: any self-loop, multi-edge, or m > n - c
    if gr.loops() or gr.multi():
        return True
    return len(gr.edges) > gr.n - len(_weak_components(gr))


def _topo(gr):
    indeg = [0] * gr.n
    adj = [[] for _ in range(gr.n)]
    for i, j, _ in gr.edges:
        adj[i].append(j)
        indeg[j] += 1
    heap = [i for i in range(gr.n) if indeg[i] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        u = heapq.heappop(heap)
        out.append(u)
        for w in adj[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(heap, w)
    return out


@fn("AcyclicGraphQ", area=AREA, status="added",
    notes="undirected: a forest (self-loops and multi-edges are cycles); directed: no directed cycle")
def _acyq(g):
    if not is_graph_like(g):
        if isinstance(g, Symbol):
            raise Unevaluated()
        return False
    gr = graph_of(g, "AcyclicGraphQ")
    gr.require_pure("AcyclicGraphQ")
    return not _has_cycle(gr)


@fn("TreeGraphQ", area=AREA, status="added",
    notes="connected acyclic undirected graph (n - 1 edges); directed graphs: True for an out-tree with every vertex "
          "reachable from one root along the underlying tree")
def _treeq(g):
    if not is_graph_like(g):
        if isinstance(g, Symbol):
            raise Unevaluated()
        return False
    gr = graph_of(g, "TreeGraphQ")
    gr.require_pure("TreeGraphQ")
    if gr.n == 0:
        return False
    if gr.loops() or gr.multi():
        return False
    if len(gr.edges) != gr.n - 1 or len(_weak_components(gr)) != 1:
        return False
    if gr.directed:
        _, ind, _ = _degree_lists(gr)
        return sorted(ind) == [0] + [1] * (gr.n - 1)
    return True


@fn("BipartiteGraphQ", area=AREA, status="added",
    notes="2-colourable (edge directions ignored; a self-loop makes it non-bipartite)")
def _bipq(g):
    if not is_graph_like(g):
        if isinstance(g, Symbol):
            raise Unevaluated()
        return False
    return _bipartition(graph_of(g, "BipartiteGraphQ")) is not None


def _bipartition(gr):
    adj = gr.out_adj(undirected=True)
    color = {}
    for s in range(gr.n):
        if s in color:
            continue
        color[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w, _, _ in adj[u]:
                if w == u:
                    return None
                if w not in color:
                    color[w] = 1 - color[u]
                    q.append(w)
                elif color[w] == color[u]:
                    return None
    return color


def _nx():
    try:
        import networkx as nx
    except ImportError:  # pragma: no cover - exercised in the minimal-install job
        raise NotImplementedError("this graph function requires NetworkX (pip install mikoshilang[all])")
    return nx


def _to_nx(gr, undirected=False):
    nx = _nx()
    if gr.directed and not undirected:
        X = nx.MultiDiGraph() if gr.multi() else nx.DiGraph()
    else:
        X = nx.MultiGraph() if gr.multi() else nx.Graph()
    X.add_nodes_from(range(gr.n))
    for k, (i, j, _) in enumerate(gr.edges):
        X.add_edge(i, j)
    return X


@fn("PlanarGraphQ", area=AREA, status="added",
    notes="planarity test (edge directions ignored) via NetworkX's left-right planarity algorithm")
def _planq(g):
    if not is_graph_like(g):
        if isinstance(g, Symbol):
            raise Unevaluated()
        return False
    gr = graph_of(g, "PlanarGraphQ")
    nx = _nx()
    X = nx.Graph()
    X.add_nodes_from(range(gr.n))
    X.add_edges_from((i, j) for i, j, _ in gr.edges if i != j)
    return bool(nx.check_planarity(X)[0])


@fn("TopologicalSort", area=AREA, status="added",
    notes="a topological order of a directed acyclic graph (Kahn's algorithm, ties broken by VertexList order); "
          "graphs with a cycle or undirected edges raise")
def _toposort(g):
    gr = graph_of(g, "TopologicalSort")
    gr.require_pure("TopologicalSort")
    if gr.edges and not gr.directed:
        raise ValueError("TopologicalSort: the graph must be directed")
    order = _topo(gr)
    if len(order) != gr.n:
        raise ValueError("TopologicalSort: the graph has a cycle")
    return L(*(gr.vertices[i] for i in order))


# ─────────────────────────── cycles and tours ───────────────────────────

def _simple_cycles(gr, lo, hi, limit):
    """Simple cycles as lists of edge indices, in traversal order, shortest first (each cycle once)."""
    cycles = []
    seen = set()
    d = gr.directed
    for k, (i, j, _) in enumerate(gr.edges):
        if i == j and lo <= 1 <= hi:
            cycles.append([k])
    if not d and gr.multi():
        raise ValueError("FindCycle: undirected multigraphs are not supported")
    adj = gr.out_adj()
    order = list(range(gr.n))

    def dfs(start, u, path_v, path_e, onpath):
        if len(path_e) > hi:
            return
        for w, _, k in adj[u]:
            if w == u:
                continue
            if w == start and max(lo, 2 if d else 3) <= len(path_e) + 1 <= hi:
                cyc = path_e + [k]
                key = frozenset(cyc)
                if key not in seen:
                    seen.add(key)
                    cycles.append(cyc)
                continue
            if w in onpath or w < start:
                continue
            onpath.add(w)
            dfs(start, w, path_v + [w], path_e + [k], onpath)
            onpath.discard(w)
    for s in order:
        dfs(s, s, [s], [], {s})
        if limit is not None and len(cycles) >= limit * 50 and len(cycles) > 2000:
            break
    cycles.sort(key=lambda c: (len(c), c))
    if limit is not None:
        cycles = cycles[:limit]
    return cycles


def _cycle_edges(gr, cyc, start=None):
    """Edge expressions oriented along the traversal of a cycle given as edge indices in order."""
    if len(cyc) == 1:
        i, j, d = gr.edges[cyc[0]]
        return [gr.edge(i, j, d)]
    out = []
    # recover the vertex sequence
    i0, j0, _ = gr.edges[cyc[0]]
    i1, j1, _ = gr.edges[cyc[1]]
    cur = j0 if j0 in (i1, j1) else i0
    prev = i0 if cur == j0 else j0
    seq = [prev, cur]
    for k in cyc[1:]:
        a, b, _ = gr.edges[k]
        nxt = b if a == cur else a
        seq.append(nxt)
        cur = nxt
    for a, b, k in zip(seq, seq[1:], cyc):
        out.append(gr.edge(a, b, gr.edges[k][2]))
    return out


@fn("FindCycle", area=AREA, status="added",
    notes="FindCycle[g] = {cycle} as a list of edges in traversal order ({} for acyclic graphs); [g, k] length at most "
          "k, [g, {k}] exactly k, [g, kspec, n] at most n cycles (All for all); shortest cycles first")
def _fcycle(g, kspec=None, nspec=1):
    gr = graph_of(g, "FindCycle")
    gr.require_pure("FindCycle")
    lo, hi = _len_spec(kspec, gr.n, "FindCycle")
    hi = min(hi, len(gr.edges))
    limit = _count_spec(nspec, "FindCycle")
    cyc = _simple_cycles(gr, lo, hi, limit)
    return L(*(L(*_cycle_edges(gr, c)) for c in cyc))


def _hamiltonian(gr, cycle=True):
    n = gr.n
    if n == 0:
        return None
    nb = [sorted(s) for s in gr.neighbor_sets()]
    if not gr.directed:
        nb = [sorted(s) for s in gr.neighbor_sets(undirected=True)]
    if cycle and n < 3 and not gr.directed:
        return None
    starts = [0] if cycle else range(n)
    for s in starts:
        path = [s]
        used = {s}

        def bt():
            if len(path) == n:
                return not cycle or s in nb[path[-1]]
            for w in nb[path[-1]]:
                if w not in used:
                    used.add(w)
                    path.append(w)
                    if bt():
                        return True
                    path.pop()
                    used.discard(w)
            return False
        if bt():
            return path
    return None


def _edge_between(gr, a, b):
    for i, j, d in gr.edges:
        if (i, j) == (a, b):
            return gr.edge(a, b, d)
        if not d and (j, i) == (a, b):
            return gr.edge(a, b, d)
    raise AssertionError("missing edge")


@fn("FindHamiltonianCycle", area=AREA, status="added",
    notes="{cycle} as edges in traversal order starting at the first vertex, {} when the graph is not Hamiltonian "
          "(exact backtracking; graphs above 30 vertices raise)")
def _fham(g):
    gr = graph_of(g, "FindHamiltonianCycle")
    gr.require_pure("FindHamiltonianCycle")
    if gr.n > 30:
        raise ValueError("FindHamiltonianCycle: graphs with more than 30 vertices are not supported")
    p = _hamiltonian(gr, True)
    if p is None:
        return L()
    return L(L(*(_edge_between(gr, a, b) for a, b in zip(p, p[1:] + p[:1]))))


@fn("FindHamiltonianPath", area=AREA, status="added",
    notes="a Hamiltonian path as a vertex list, {} when none (exact backtracking; graphs above 30 vertices raise)")
def _fhamp(g):
    gr = graph_of(g, "FindHamiltonianPath")
    gr.require_pure("FindHamiltonianPath")
    if gr.n > 30:
        raise ValueError("FindHamiltonianPath: graphs with more than 30 vertices are not supported")
    p = _hamiltonian(gr, False)
    if p is None:
        return L()
    return L(*(gr.vertices[i] for i in p))


@fn("HamiltonianGraphQ", area=AREA, status="added", notes="True when a Hamiltonian cycle exists")
def _hamq(g):
    gr = graph_of(g, "HamiltonianGraphQ")
    gr.require_pure("HamiltonianGraphQ")
    if gr.n > 30:
        raise ValueError("HamiltonianGraphQ: graphs with more than 30 vertices are not supported")
    return _hamiltonian(gr, True) is not None


def _eulerian_ok(gr):
    deg, ind, outd = _degree_lists(gr)
    nonzero = [i for i in range(gr.n) if deg[i] > 0]
    if not gr.edges:
        return False
    comps = _weak_components(Sub(gr, nonzero))
    if len(comps) != 1:
        return False
    if gr.directed:
        return all(ind[i] == outd[i] for i in range(gr.n))
    return all(deg[i] % 2 == 0 for i in range(gr.n))


class Sub:
    """Induced view used for connectivity checks."""

    def __init__(self, gr, keep):
        self.keep = sorted(keep)
        pos = {v: k for k, v in enumerate(self.keep)}
        self.n = len(self.keep)
        self.edges = [(pos[i], pos[j], d) for i, j, d in gr.edges if i in pos and j in pos]
        self.weights = None

    def out_adj(self, undirected=False):
        adj = [[] for _ in range(self.n)]
        for k, (i, j, d) in enumerate(self.edges):
            adj[i].append((j, 1, k))
            if (not d or undirected) and i != j:
                adj[j].append((i, 1, k))
        return adj


@fn("EulerianGraphQ", area=AREA, status="added",
    notes="True when there is a closed walk using every edge exactly once (even degrees / balanced in- and "
          "out-degrees, edges connected); edgeless graphs are not Eulerian")
def _eulq(g):
    gr = graph_of(g, "EulerianGraphQ")
    gr.require_pure("EulerianGraphQ")
    return _eulerian_ok(gr)


@fn("FindEulerianCycle", area=AREA, status="added",
    notes="{cycle} as edges in traversal order (Hierholzer, multigraphs and self-loops allowed), {} when the graph "
          "is not Eulerian")
def _feul(g):
    gr = graph_of(g, "FindEulerianCycle")
    gr.require_pure("FindEulerianCycle")
    if not _eulerian_ok(gr):
        return L()
    adj = gr.out_adj()
    used = [False] * len(gr.edges)
    ptr = [0] * gr.n
    start = gr.edges[0][0]
    stack = [(start, None)]
    circuit = []
    while stack:
        u, via = stack[-1]
        while ptr[u] < len(adj[u]) and used[adj[u][ptr[u]][2]]:
            ptr[u] += 1
        if ptr[u] == len(adj[u]):
            stack.pop()
            if via is not None:
                circuit.append((stack[-1][0], u, via))
        else:
            w, _, k = adj[u][ptr[u]]
            used[k] = True
            stack.append((w, k))
    circuit.reverse()
    return L(L(*(gr.edge(a, b, gr.edges[k][2]) for a, b, k in circuit)))


# ─────────────────────────── trees, cuts and flows ───────────────────────────

@fn("FindSpanningTree", area=AREA, status="added",
    notes="a minimum spanning tree (forest when disconnected) as a Graph on all the vertices (Kruskal with EdgeWeight, "
          "edges kept in the input orientation); directed graphs use the underlying undirected graph")
def _fst(g):
    gr = graph_of(g, "FindSpanningTree")
    gr.require_pure("FindSpanningTree")
    parent = list(range(gr.n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    chosen = []
    for k in sorted(range(len(gr.edges)), key=lambda k: (float(gr.weight(k)), k)):
        i, j, _ = gr.edges[k]
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
            chosen.append(k)
    chosen.sort()
    opts = {}
    if gr.weights is not None:
        opts["EdgeWeight"] = L(*(R(gr.weights[k]) for k in chosen))
    return WG(gr.vertices, [Expr("UndirectedEdge", *gr.edge_exprs[k].args) for k in chosen], opts).expr()


def _maxflow(gr, s, t, caps):
    """Edmonds-Karp on a residual graph; returns (value, reachable set from s in the final residual graph)."""
    n = gr.n
    cap = {}
    for k, (i, j, d) in enumerate(gr.edges):
        if i == j:
            continue
        c = caps[k]
        cap[(i, j)] = cap.get((i, j), 0) + c
        cap.setdefault((j, i), 0)
        if not d:
            cap[(j, i)] = cap.get((j, i), 0) + c
    nbrs = [set() for _ in range(n)]
    for (i, j) in cap:
        nbrs[i].add(j)
    flow = 0
    while True:
        prev = {s: None}
        q = deque([s])
        while q and t not in prev:
            u = q.popleft()
            for w in sorted(nbrs[u]):
                if w not in prev and cap[(u, w)] > 0:
                    prev[w] = u
                    q.append(w)
        if t not in prev:
            return flow, set(prev)
        bott = None
        v = t
        while prev[v] is not None:
            c = cap[(prev[v], v)]
            bott = c if bott is None or c < bott else bott
            v = prev[v]
        v = t
        while prev[v] is not None:
            cap[(prev[v], v)] -= bott
            cap[(v, prev[v])] += bott
            v = prev[v]
        flow += bott


def _caps(gr, opts, fname):
    c = opts.get("EdgeCapacity")
    if c is not None:
        gr = WG(gr.vertices, gr.edge_exprs, dict(gr.opts, EdgeCapacity=c), fname)
    if gr.capacities is not None:
        return gr.capacities
    return [sp.Integer(1)] * len(gr.edges)


@fn("FindMaximumFlow", area=AREA, status="added",
    notes="value of a maximum s-t flow (Edmonds-Karp; capacities from EdgeCapacity, default 1; undirected edges carry "
          "flow either way); exact for exact capacities")
def _fmaxflow(g, s, t, *rest):
    pos, opts = split_options(list(rest), {"EdgeCapacity"}, "FindMaximumFlow")
    if pos:
        raise ValueError("FindMaximumFlow: only the value form FindMaximumFlow[g, s, t] is supported")
    gr = graph_of(g, "FindMaximumFlow")
    gr.require_pure("FindMaximumFlow")
    a, b = gr.vidx(s, "FindMaximumFlow"), gr.vidx(t, "FindMaximumFlow")
    if a == b:
        raise ValueError("FindMaximumFlow: source and target must differ")
    caps = _caps(gr, opts, "FindMaximumFlow")
    val, _ = _maxflow(gr, a, b, caps)
    return R(val)


def _stoer_wagner(n, w):
    """Global minimum cut of an undirected weighted graph given as an n x n weight matrix (lists)."""
    groups = [[i] for i in range(n)]
    active = list(range(n))
    best, best_set = None, None
    w = [row[:] for row in w]
    while len(active) > 1:
        a0 = active[0]
        added = [a0]
        weights = {v: w[a0][v] for v in active if v != a0}
        prev = a0
        while weights:
            nxt = max(weights, key=lambda v: (weights[v], -v))
            added.append(nxt)
            del weights[nxt]
            for v in weights:
                weights[v] += w[nxt][v]
            prev_last = prev
            prev = nxt
        last, before = added[-1], added[-2]
        cut = sum(w[last][v] for v in active if v != last)
        if best is None or cut < best:
            best, best_set = cut, list(groups[last])
        groups[before] += groups[last]
        for v in active:
            w[before][v] += w[last][v]
            w[v][before] = w[before][v]
        w[before][before] = 0
        active.remove(last)
    return best, best_set


@fn("FindMinimumCut", area=AREA, status="added",
    notes="{c, {part1, part2}}: a global minimum cut of a connected undirected graph (Stoer-Wagner, EdgeWeight "
          "respected; parts in VertexList order, the part with the first vertex first); disconnected graphs give 0; "
          "directed graphs raise")
def _fmincut(g):
    gr = graph_of(g, "FindMinimumCut")
    gr.require_undirected("FindMinimumCut")
    if gr.n < 2:
        raise ValueError("FindMinimumCut: the graph needs at least two vertices")
    comps = _weak_components(gr)
    if len(comps) > 1:
        part = set(comps[0]) if 0 in comps[0] else set(next(c for c in comps if 0 in c))
        a = [gr.vertices[i] for i in range(gr.n) if i in part]
        b = [gr.vertices[i] for i in range(gr.n) if i not in part]
        return L(0, L(L(*a), L(*b)))
    w = [[sp.Integer(0)] * gr.n for _ in range(gr.n)]
    for k, (i, j, _) in enumerate(gr.edges):
        if i != j:
            w[i][j] += gr.weight(k)
            w[j][i] += gr.weight(k)
    val, side = _stoer_wagner(gr.n, w)
    side = set(side)
    if 0 not in side:
        side = set(range(gr.n)) - side
    a = [gr.vertices[i] for i in range(gr.n) if i in side]
    b = [gr.vertices[i] for i in range(gr.n) if i not in side]
    return L(R(val), L(L(*a), L(*b)))


@fn("EdgeConnectivity", area=AREA, status="added",
    notes="minimum number (total weight) of edges whose removal disconnects an undirected graph; s-t form "
          "EdgeConnectivity[g, s, t] via maximum flow")
def _econn(g, s=None, t=None):
    gr = graph_of(g, "EdgeConnectivity")
    gr.require_pure("EdgeConnectivity")
    if s is not None:
        if t is None:
            raise ValueError("EdgeConnectivity: give both s and t")
        a, b = gr.vidx(s, "EdgeConnectivity"), gr.vidx(t, "EdgeConnectivity")
        val, _ = _maxflow(gr, a, b, [sp.Integer(1)] * len(gr.edges))
        return R(val)
    if gr.directed:
        raise ValueError("EdgeConnectivity: the global form supports undirected graphs only")
    if gr.n < 2:
        return 0
    if len(_weak_components(gr)) > 1:
        return 0
    w = [[0] * gr.n for _ in range(gr.n)]
    for i, j, _ in gr.edges:
        if i != j:
            w[i][j] += 1
            w[j][i] += 1
    return R(_stoer_wagner(gr.n, w)[0])


# ─────────────────────────── cliques, independent sets, covers ───────────────────────────

def _max_cliques(nb, n):
    """All maximal cliques (Bron-Kerbosch with pivoting), each sorted."""
    out = []

    def bk(r, p, x):
        if not p and not x:
            out.append(sorted(r))
            return
        pivot = max(p | x, key=lambda v: len(nb[v] & p))
        for v in sorted(p - nb[pivot]):
            bk(r | {v}, p & nb[v], x & nb[v])
            p = p - {v}
            x = x | {v}
    bk(set(), set(range(n)), set())
    return out


def _clique_graph(gr, fname):
    gr.require_pure(fname)
    nb = gr.neighbor_sets(undirected=True)
    if gr.directed:
        # a clique in a directed graph needs edges both ways
        outs = gr.neighbor_sets()
        nb = [set(j for j in outs[i] if i in outs[j]) for i in range(gr.n)]
    return nb


@fn("FindClique", area=AREA, status="added",
    notes="FindClique[g] = {a largest clique}; [g, n] a maximal clique of at most n vertices... (largest found), "
          "[g, {n}] a clique of exactly n vertices, [g, nspec, All] every such clique; vertices in VertexList order; "
          "directed graphs need edges both ways")
def _fclique(g, spec=None, count=1):
    gr = graph_of(g, "FindClique")
    nb = _clique_graph(gr, "FindClique")
    limit = _count_spec(count, "FindClique")
    if gr.n == 0:
        return L()
    if spec is None:
        cl = _max_cliques(nb, gr.n)
        m = max(len(c) for c in cl)
        res = sorted((c for c in cl if len(c) == m))
    elif isinstance(spec, int) and not isinstance(spec, bool):
        cl = _max_cliques(nb, gr.n)
        cand = [c for c in cl if len(c) <= spec]
        if not cand:
            return L()
        m = max(len(c) for c in cand)
        res = sorted(c for c in cand if len(c) == m)
    elif is_list(spec) and len(spec.args) == 1 and isinstance(spec.args[0], int):
        k = spec.args[0]
        res = []
        for combo in itertools.combinations(range(gr.n), k):
            if all(b in nb[a] for a, b in itertools.combinations(combo, 2)):
                res.append(list(combo))
                if limit is not None and len(res) >= limit:
                    break
    else:
        raise ValueError(f"FindClique: invalid size specification {spec}")
    if limit is not None:
        res = res[:limit]
    return L(*(L(*(gr.vertices[i] for i in c)) for c in res))


@fn("FindIndependentVertexSet", area=AREA, status="added",
    notes="{a largest independent vertex set} (maximum clique of the complement; edge directions ignored); "
          "[g, {n}] an independent set of exactly n vertices; vertices in VertexList order")
def _findep(g, spec=None, count=1):
    gr = graph_of(g, "FindIndependentVertexSet")
    gr.require_pure("FindIndependentVertexSet")
    if gr.loops():
        raise ValueError("FindIndependentVertexSet: graphs with self-loops are not supported")
    nb = gr.neighbor_sets(undirected=True)
    comp = [set(range(gr.n)) - nb[i] - {i} for i in range(gr.n)]
    limit = _count_spec(count, "FindIndependentVertexSet")
    if gr.n == 0:
        return L()
    if spec is None:
        cl = _max_cliques(comp, gr.n)
        m = max(len(c) for c in cl)
        res = sorted(c for c in cl if len(c) == m)
    elif is_list(spec) and len(spec.args) == 1 and isinstance(spec.args[0], int):
        k = spec.args[0]
        res = []
        for combo in itertools.combinations(range(gr.n), k):
            if all(b not in nb[a] for a, b in itertools.combinations(combo, 2)):
                res.append(list(combo))
                if limit is not None and len(res) >= limit:
                    break
    else:
        raise ValueError(f"FindIndependentVertexSet: invalid size specification {spec}")
    if limit is not None:
        res = res[:limit]
    return L(*(L(*(gr.vertices[i] for i in c)) for c in res))


@fn("FindVertexCover", area=AREA, status="added",
    notes="a minimum vertex cover (complement of a maximum independent set), as a flat vertex list in VertexList order")
def _fvc(g):
    gr = graph_of(g, "FindVertexCover")
    gr.require_pure("FindVertexCover")
    if gr.n == 0:
        return L()
    nb = gr.neighbor_sets(undirected=True)
    loops = {i for i, j, _ in gr.edges if i == j}
    rest = [i for i in range(gr.n) if i not in loops]
    pos = {v: k for k, v in enumerate(rest)}
    comp = [set(pos[w] for w in rest if w != v and w not in nb[v]) for v in rest]
    best = []
    if rest:
        cl = _max_cliques(comp, len(rest))
        m = max(len(c) for c in cl)
        best = sorted(c for c in cl if len(c) == m)[0]
    indep = {rest[k] for k in best}
    return L(*(gr.vertices[i] for i in range(gr.n) if i not in indep))


def _set_q(name, check, notes):
    @fn(name, area=AREA, status="added", notes=notes)
    def impl(g, s):
        gr = graph_of(g, name)
        items = list(s.args) if is_list(s) else None
        if items is None:
            raise ValueError(f"{name}: the second argument must be a list")
        return check(gr, items)
    return impl


def _vc_check(gr, items):
    try:
        idx = {gr.vidx(v, "VertexCoverQ") for v in items}
    except ValueError:
        return False
    return all(i in idx or j in idx for i, j, _ in gr.edges)


def _ivs_check(gr, items):
    try:
        idx = [gr.vidx(v, "IndependentVertexSetQ") for v in items]
    except ValueError:
        return False
    s = set(idx)
    return all(not (i in s and j in s) for i, j, _ in gr.edges)


_set_q("VertexCoverQ", _vc_check, "True when every edge has an endpoint in the list")
_set_q("IndependentVertexSetQ", _ivs_check, "True when no two listed vertices are adjacent")


# ─────────────────────────── isomorphism ───────────────────────────

def _iso_graphs(g1, g2, fname):
    a, b = graph_of(g1, fname), graph_of(g2, fname)
    for x in (a, b):
        x.require_pure(fname)
        if x.multi():
            raise ValueError(f"{fname}: multigraphs are not supported")
    return a, b


@fn("IsomorphicGraphQ", area=AREA, status="added",
    notes="graph isomorphism (VF2 via NetworkX), direction-aware; multigraphs raise")
def _isoq(g1, g2, *more):
    gs = [g1, g2] + list(more)
    base = None
    for x in gs[1:]:
        a, b = _iso_graphs(gs[0], x, "IsomorphicGraphQ")
        if a.directed != b.directed and (a.edges and b.edges):
            return False
        nx = _nx()
        if not nx.is_isomorphic(_to_nx(a), _to_nx(b)):
            return False
    return True


@fn("FindGraphIsomorphism", area=AREA, status="added",
    notes="{<|v1 -> w1, ...|>} for one isomorphism (VF2 via NetworkX), {} when none; [g1, g2, n] / All for several; "
          "keys in VertexList order of g1")
def _fiso(g1, g2, count=1):
    a, b = _iso_graphs(g1, g2, "FindGraphIsomorphism")
    limit = _count_spec(count, "FindGraphIsomorphism")
    if (a.directed != b.directed and a.edges and b.edges) or a.n != b.n or len(a.edges) != len(b.edges):
        return L()
    nx = _nx()
    from networkx.algorithms import isomorphism as iso
    X, Y = _to_nx(a), _to_nx(b)
    M = iso.DiGraphMatcher(X, Y) if X.is_directed() else iso.GraphMatcher(X, Y)
    out = []
    from ..kernel_lists import make_assoc
    for m in M.isomorphisms_iter():
        out.append(make_assoc([(a.vertices[i], b.vertices[m[i]]) for i in range(a.n)]))
        if limit is not None and len(out) >= limit:
            break
    return L(*out)


# ─────────────────────────── chromatic polynomial ───────────────────────────

@fn("ChromaticPolynomial", area=AREA, status="added",
    notes="chromatic polynomial in k (deletion-contraction with memoisation, expanded); multi-edges are merged, "
          "self-loops give 0; directions ignored; graphs with more than 30 edges raise")
def _chrompoly(g, k):
    gr = graph_of(g, "ChromaticPolynomial")
    if gr.loops():
        return 0
    x = S(k)
    es = {(min(i, j), max(i, j)) for i, j, _ in gr.edges}
    if len(es) > 30:
        raise ValueError("ChromaticPolynomial: graphs with more than 30 edges are not supported")
    t = sp.Dummy("t")
    memo = {}

    def canon(n, edges):
        return (n, tuple(sorted(edges)))

    def P(n, edges):
        key = canon(n, edges)
        if key in memo:
            return memo[key]
        if not edges:
            r = t ** n
        else:
            e = max(edges)
            rest = edges - {e}
            a, b = e
            # contract b into a, relabel vertices above b
            def rl(v):
                v = a if v == b else v
                return v - 1 if v > b else v
            cont = set()
            for (u, v) in rest:
                u2, v2 = rl(u), rl(v)
                if u2 != v2:
                    cont.add((min(u2, v2), max(u2, v2)))
            r = sp.expand(P(n, frozenset(rest)) - P(n - 1, frozenset(cont)))
        memo[key] = r
        return r
    poly = P(gr.n, frozenset(es))
    return R(sp.expand(poly.subs(t, x)))


# ─────────────────────────── centrality ───────────────────────────

def _floats(vals):
    return L(*(float(v) for v in vals))


@fn("ClosenessCentrality", area=AREA, status="added",
    notes="1/(mean distance to the reachable vertices), 0 for vertices that reach nothing; machine reals")
def _closeness(g):
    gr = graph_of(g, "ClosenessCentrality")
    gr.require_pure("ClosenessCentrality")
    out = []
    for i in range(gr.n):
        d, _ = gr.sssp(i)
        ds = [v for j, v in d.items() if j != i]
        out.append(0.0 if not ds or sum(ds) == 0 else len(ds) / float(sum(ds)))
    return L(*out)


def _between(gr):
    import fractions
    n = gr.n
    adj = gr.out_adj()
    cb = [fractions.Fraction(0)] * n
    weighted = gr.weights is not None
    for s in range(n):
        stack, pred = [], [[] for _ in range(n)]
        sigma = [0] * n
        sigma[s] = 1
        dist = {s: sp.Integer(0)}
        if not weighted:
            q = deque([s])
            while q:
                u = q.popleft()
                stack.append(u)
                for w, _, _ in adj[u]:
                    if w == u:
                        continue
                    if w not in dist:
                        dist[w] = dist[u] + 1
                        q.append(w)
                    if dist[w] == dist[u] + 1:
                        sigma[w] += sigma[u]
                        pred[w].append(u)
        else:
            seen = {s: sp.Integer(0)}
            pq = [(0.0, s, s, s)]
            done = set()
            while pq:
                _, _, pv, u = heapq.heappop(pq)
                if u in done:
                    continue
                done.add(u)
                stack.append(u)
                for w, wt, _ in adj[u]:
                    if w == u:
                        continue
                    nd = dist[u] + wt
                    if w not in dist or nd < dist[w]:
                        dist[w] = nd
                        sigma[w] = sigma[u]
                        pred[w] = [u]
                        heapq.heappush(pq, (float(nd), w, u, w))
                    elif nd == dist[w] and w not in done:
                        sigma[w] += sigma[u]
                        pred[w].append(u)
        delta = [fractions.Fraction(0)] * n
        while stack:
            w = stack.pop()
            for u in pred[w]:
                delta[u] += fractions.Fraction(sigma[u], sigma[w]) * (1 + delta[w])
            if w != s:
                cb[w] += delta[w]
    if not gr.directed:
        cb = [c / 2 for c in cb]
    return cb


def _betweenness(g):
    gr = graph_of(g, "BetweennessCentrality")
    gr.require_pure("BetweennessCentrality")
    if gr.multi():
        raise ValueError("BetweennessCentrality: multigraphs are not supported")
    return L(*(float(c) for c in _between(gr)))


extend("BetweennessCentrality", _betweenness, _first_graph,
       "Graph or edge-rule input: list of unnormalised betweenness centralities in VertexList order (machine reals; "
       "directed graphs count ordered pairs; EdgeWeight respected)")


def _principal(M, fname, symmetric):
    import numpy as np
    vals, vecs = (np.linalg.eigh(M) if symmetric else np.linalg.eig(M))
    order = np.argsort(-vals.real)
    lam = vals[order[0]].real
    if lam <= 1e-12:
        raise ValueError(f"{fname}: the graph has no edges to rank by")
    if len(vals) > 1 and abs(vals[order[1]].real - lam) < 1e-9 * max(1.0, abs(lam)):
        raise ValueError(f"{fname}: the principal eigenvalue is not simple (e.g. a disconnected or periodic graph); "
                         "the centrality is not uniquely defined")
    v = np.abs(vecs[:, order[0]].real)
    return lam, v


@fn("EigenvectorCentrality", area=AREA, status="added",
    notes="principal eigenvector of the adjacency matrix normalised to total 1 (\"In\": left eigenvector, the default "
          "for directed graphs; \"Out\": right eigenvector); graphs whose principal eigenvalue is not simple (e.g. "
          "disconnected) raise; machine reals")
def _eigc(g, kind=None):
    import numpy as np
    gr = graph_of(g, "EigenvectorCentrality")
    gr.require_pure("EigenvectorCentrality")
    m = np.array([[float(x) for x in r.args] for r in _adjm(g).args]) if gr.n else None
    if gr.n == 0:
        return L()
    if kind not in (None, "In", "Out"):
        raise ValueError('EigenvectorCentrality: the second argument must be "In" or "Out"')
    if len(_components(gr)) != 1:
        raise ValueError("EigenvectorCentrality: the graph is not (strongly) connected; Wolfram's per-component "
                         "normalisation is not reproduced")
    sym = not gr.directed
    M = m.T if (kind != "Out") else m
    _, v = _principal(M, "EigenvectorCentrality", sym)
    v = v / v.sum()
    return _floats(v)


@fn("HITSCentrality", area=AREA, status="added",
    notes="{authorities, hubs}: authority = principal eigenvector of A^T.A normalised to total 1, hub = A.authority "
          "(Wolfram's definition); degenerate principal eigenvalues raise; machine reals")
def _hits(g):
    import numpy as np
    gr = graph_of(g, "HITSCentrality")
    gr.require_pure("HITSCentrality")
    if gr.n == 0:
        return L(L(), L())
    A = np.array([[float(x) for x in r.args] for r in _adjm(g).args])
    _, x = _principal(A.T @ A, "HITSCentrality", True)
    x = x / x.sum()
    y = A @ x
    return L(_floats(x), _floats(y))


@fn("PageRankCentrality", area=AREA, status="added",
    notes="PageRankCentrality[g, alpha] (default 0.85): solution of c = alpha A^T D^-1 c + (1 - alpha)/n normalised "
          "to total 1, undirected edges counted both ways; graphs with a vertex of out-degree 0 raise (Wolfram's "
          "dangling-vertex convention is not reproduced); machine reals")
def _pagerank(g, alpha=0.85, beta=None):
    import numpy as np
    gr = graph_of(g, "PageRankCentrality")
    gr.require_pure("PageRankCentrality")
    a = float(number(alpha, "PageRankCentrality", "damping factor"))
    if not 0 <= a < 1:
        raise ValueError("PageRankCentrality: the damping factor must be in [0, 1)")
    if beta is not None:
        raise ValueError("PageRankCentrality: the beta argument is not supported")
    n = gr.n
    if n == 0:
        return L()
    A = np.array([[float(x) for x in r.args] for r in _adjm(g).args])
    out = A.sum(axis=1)
    if np.any(out == 0):
        raise ValueError("PageRankCentrality: graphs with vertices of out-degree 0 are not supported")
    M = (A / out[:, None]).T
    c = np.linalg.solve(np.eye(n) - a * M, np.full(n, (1 - a) / n))
    c = c / c.sum()
    return _floats(c)


# ─────────────────────────── named graphs ───────────────────────────

def _int_arg(x, fname, lo=0):
    if isinstance(x, bool) or not isinstance(x, int):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise ValueError(f"{fname}: expected an integer, got {x}")
    if x < lo:
        raise ValueError(f"{fname}: the argument must be at least {lo}")
    return x


# named graphs whose DirectedEdges -> True orientation is Wolfram's: trees point away from the root, paths and
# cycles follow the vertex order (the cycle closes with n -> 1)
_DIRECTABLE = {"PathGraph", "CycleGraph", "KaryTree", "CompleteKaryTree", "StarGraph"}


def _named(name, notes):
    def deco(f):
        def impl(*args):
            pos, opts = split_options(list(args), _GRAPH_OPTS | {"DirectedEdges"}, name)
            vs, es = f(*pos)
            if "DirectedEdges" in opts:
                de = opts.pop("DirectedEdges")
                if de is True and name not in _DIRECTABLE:
                    raise NotImplementedError(f"{name}: DirectedEdges -> True is not supported (Wolfram's edge "
                                              "orientation for this graph is not reproduced)")
                es = _directed_edges(es, de, name)
                if de is True and name == "CycleGraph" and len(vs) >= 3:
                    es[-1] = Expr("DirectedEdge", vs[-1], vs[0])      # 1 -> 2 -> ... -> n -> 1
            return WG(vs, es, opts, name).expr()
        import inspect
        impl.__signature__ = inspect.Signature(
            [inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL)])
        fn(name, area=AREA, status="added", notes=notes)(impl)
        return f
    return deco


def UE(a, b):
    return Expr("UndirectedEdge", a, b)


@_named("CompleteGraph", "K_n on 1..n (edges i <-> j, i < j, lexicographic); CompleteGraph[{n1, n2, ...}] complete "
                         "multipartite graph")
def _complete(n):
    if is_list(n):
        sizes = [_int_arg(x, "CompleteGraph", 1) for x in n.args]
        part, v = [], 1
        for s in sizes:
            part.append(list(range(v, v + s)))
            v += s
        vs = list(range(1, v))
        owner = {x: k for k, p in enumerate(part) for x in p}
        es = [UE(i, j) for i in vs for j in vs if i < j and owner[i] != owner[j]]
        return vs, es
    n = _int_arg(n, "CompleteGraph", 0)
    return list(range(1, n + 1)), [UE(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]


@_named("CycleGraph", "C_n on 1..n: 1 <-> 2, ..., (n-1) <-> n, 1 <-> n; n >= 3")
def _cycle(n):
    n = _int_arg(n, "CycleGraph", 3)
    return list(range(1, n + 1)), [UE(i, i + 1) for i in range(1, n)] + [UE(1, n)]


@_named("PathGraph", "PathGraph[{v1, ..., vn}]: v1 <-> v2 <-> ... <-> vn")
def _path(vs):
    if not is_list(vs):
        raise ValueError("PathGraph: expected a list of vertices")
    vs = list(vs.args)
    return vs, [UE(a, b) for a, b in zip(vs, vs[1:])]


@_named("StarGraph", "S_n on 1..n with centre 1: 1 <-> 2, ..., 1 <-> n")
def _star(n):
    n = _int_arg(n, "StarGraph", 1)
    return list(range(1, n + 1)), [UE(1, i) for i in range(2, n + 1)]


@_named("WheelGraph", "W_n on 1..n: hub 1 joined to the cycle 2..n; n >= 4")
def _wheel(n):
    n = _int_arg(n, "WheelGraph", 4)
    rim = list(range(2, n + 1))
    es = [UE(1, i) for i in rim] + [UE(a, b) for a, b in zip(rim, rim[1:])] + [UE(2, n)]
    return list(range(1, n + 1)), es


@_named("GridGraph", "GridGraph[{n1, ..., nk}]: vertices 1..n1*...*nk with the first coordinate varying fastest; "
                     "edges between vertices one step apart in one coordinate, sorted")
def _grid(dims):
    if not is_list(dims) or not dims.args:
        raise ValueError("GridGraph: expected {n1, n2, ...}")
    ds = [_int_arg(x, "GridGraph", 1) for x in dims.args]
    total = 1
    for d in ds:
        total *= d
    strides, s = [], 1
    for d in ds:
        strides.append(s)
        s *= d
    es = []
    for idx in range(total):
        coords, r = [], idx
        for d in ds:
            coords.append(r % d)
            r //= d
        for k, d in enumerate(ds):
            if coords[k] + 1 < d:
                es.append((idx + 1, idx + 1 + strides[k]))
    es.sort()
    return list(range(1, total + 1)), [UE(a, b) for a, b in es]


@_named("HypercubeGraph", "Q_n on 1..2^n: i <-> j when the binary digits of i - 1 and j - 1 differ in one place")
def _hyper(n):
    n = _int_arg(n, "HypercubeGraph", 0)
    N = 2 ** n
    es = [UE(i + 1, j + 1) for i in range(N) for j in range(i + 1, N) if bin(i ^ j).count("1") == 1]
    return list(range(1, N + 1)), es


@_named("PetersenGraph", "generalised Petersen graph GP(n, k) (default GP(5, 2)): outer cycle 1..n, spokes "
                         "i <-> n + i, inner edges n + i <-> n + (i + k mod n)")
def _petersen(n=5, k=2):
    n = _int_arg(n, "PetersenGraph", 3)
    k = _int_arg(k, "PetersenGraph", 1)
    if 2 * k >= n and not (2 * k == n):
        raise ValueError("PetersenGraph: k must be less than n/2")
    es = set()
    for i in range(n):
        es.add(tuple(sorted((i + 1, (i + 1) % n + 1))))
        es.add((i + 1, n + i + 1))
        a, b = n + i + 1, n + (i + k) % n + 1
        if a != b:
            es.add(tuple(sorted((a, b))))
    return list(range(1, 2 * n + 1)), [UE(a, b) for a, b in sorted(es)]


@_named("KaryTree", "KaryTree[n, k] (default k = 2): n vertices 1..n, the children of i are k(i-1)+2, ..., k(i-1)+k+1")
def _kary(n, k=2):
    n = _int_arg(n, "KaryTree", 0)
    k = _int_arg(k, "KaryTree", 1)
    return list(range(1, n + 1)), [UE((c - 2) // k + 1, c) for c in range(2, n + 1)]


@_named("CompleteKaryTree", "CompleteKaryTree[n, k] (default k = 2): complete k-ary tree with n levels, vertices "
                            "numbered breadth first")
def _ckary(n, k=2):
    n = _int_arg(n, "CompleteKaryTree", 1)
    k = _int_arg(k, "CompleteKaryTree", 1)
    total = sum(k ** i for i in range(n))
    return list(range(1, total + 1)), [UE((c - 2) // k + 1, c) for c in range(2, total + 1)]


@_named("CirculantGraph", "CirculantGraph[n, {j1, ...}]: i <-> i + j (mod n) on 1..n, sorted edges")
def _circ(n, js):
    n = _int_arg(n, "CirculantGraph", 1)
    jl = [_int_arg(x, "CirculantGraph", 1) for x in (js.args if is_list(js) else [js])]
    es = set()
    for i in range(n):
        for j in jl:
            a, b = i + 1, (i + j) % n + 1
            if a != b:
                es.add(tuple(sorted((a, b))))
    return list(range(1, n + 1)), [UE(a, b) for a, b in sorted(es)]


@_named("TuranGraph", "TuranGraph[n, k]: complete k-partite graph on 1..n with parts as equal as possible, consecutive")
def _turan(n, k):
    n = _int_arg(n, "TuranGraph", 1)
    k = _int_arg(k, "TuranGraph", 1)
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    return _complete(L(*[s for s in sizes if s > 0]))


@fn("RandomGraph", area=AREA, status="added",
    notes="RandomGraph[{n, m}] uniformly random simple undirected graph with n vertices and m edges; RandomGraph[{n, "
          "m}, k] a list of k; reproducible with SeedRandom (the random stream differs from Wolfram's)")
def _rgraph(spec, k=None):
    from .statistics import _RNG
    if not (is_list(spec) and len(spec.args) == 2):
        raise ValueError("RandomGraph: only the {n, m} form is supported")
    n = _int_arg(spec.args[0], "RandomGraph", 0)
    m = _int_arg(spec.args[1], "RandomGraph", 0)
    pairs = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    if m > len(pairs):
        raise ValueError(f"RandomGraph: a simple graph on {n} vertices has at most {len(pairs)} edges")

    def one():
        chosen = sorted(_RNG.sample(pairs, m))
        return _new(list(range(1, n + 1)), [UE(a, b) for a, b in chosen])
    if k is None:
        return one()
    k = _int_arg(k, "RandomGraph", 0)
    return L(*(one() for _ in range(k)))


@fn("GraphDensity", area=AREA, status="added",
    notes="m / (n (n - 1)/2) for undirected, m / (n (n - 1)) for directed graphs (exact)")
def _gdensity(g):
    gr = graph_of(g, "GraphDensity")
    gr.require_pure("GraphDensity")
    n = gr.n
    if n < 2:
        raise ValueError("GraphDensity: the graph needs at least two vertices")
    m = len(gr.edges)
    return R(sp.Rational(m, n * (n - 1)) if gr.directed else sp.Rational(2 * m, n * (n - 1)))


# ─────────────────────────── traversal ───────────────────────────

_BFS_EVENTS = {"PrevisitVertex", "PostvisitVertex", "DiscoverVertex", "FrontierEdge", "CycleEdge", "UnvisitedVertex"}


def _scan(g, args, fname, bfs):
    gr = graph_of(g, fname)
    gr.require_pure(fname)
    if len(args) == 1:
        s, events = None, args[0]
    elif len(args) == 2:
        s, events = args
    else:
        raise ValueError(f"{fname}: expected {fname}[g, s, {{event -> f, ...}}]")
    if not is_list(events) or not all(isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and len(r.args) == 2
                                       for r in events.args):
        raise ValueError(f"{fname}: events must be a list of \"event\" -> f rules")
    ev_map = {}
    for r in events.args:
        name = r.args[0]
        if not isinstance(name, str) or name not in _BFS_EVENTS:
            raise ValueError(f"{fname}: unsupported event {name}; supported: {sorted(_BFS_EVENTS)}")
        ev_map[name] = r.args[1]
    from .wl_common import apply

    def fire(name, *a):
        f = ev_map.get(name)
        if f is not None:
            apply(f, *a)
    adj = gr.out_adj()
    color = {}
    starts = [gr.vidx(s, fname)] if s is not None else list(range(gr.n))
    for st in starts:
        if st in color:
            continue
        if bfs:
            color[st] = 1
            fire("DiscoverVertex", gr.vertices[st], gr.vertices[st], 0)
            q = deque([(st, 0)])
            while q:
                u, du = q.popleft()
                fire("PrevisitVertex", gr.vertices[u])
                for w, _, k in adj[u]:
                    if w not in color:
                        color[w] = 1
                        fire("FrontierEdge", gr.edge_exprs[k])
                        fire("DiscoverVertex", gr.vertices[w], gr.vertices[u], du + 1)
                        q.append((w, du + 1))
                    else:
                        fire("CycleEdge", gr.edge_exprs[k])
                color[u] = 2
                fire("PostvisitVertex", gr.vertices[u])
        else:
            color[st] = 1
            fire("DiscoverVertex", gr.vertices[st], gr.vertices[st], 0)
            fire("PrevisitVertex", gr.vertices[st])
            stack = [(st, iter(adj[st]), 0)]
            while stack:
                u, it, du = stack[-1]
                advanced = False
                for w, _, k in it:
                    if w not in color:
                        color[w] = 1
                        fire("FrontierEdge", gr.edge_exprs[k])
                        fire("DiscoverVertex", gr.vertices[w], gr.vertices[u], du + 1)
                        fire("PrevisitVertex", gr.vertices[w])
                        stack.append((w, iter(adj[w]), du + 1))
                        advanced = True
                        break
                    else:
                        fire("CycleEdge", gr.edge_exprs[k])
                if not advanced:
                    stack.pop()
                    color[u] = 2
                    fire("PostvisitVertex", gr.vertices[u])
        if s is not None:
            for v in range(gr.n):
                if v not in color:
                    fire("UnvisitedVertex", gr.vertices[v])
    return g


@fn("BreadthFirstScan", area=AREA, status="added",
    notes="BreadthFirstScan[g, s, {\"event\" -> f, ...}] calls f on events (\"DiscoverVertex\" f[u, v, d], "
          "\"PrevisitVertex\", \"PostvisitVertex\", \"FrontierEdge\", \"CycleEdge\", \"UnvisitedVertex\") in "
          "breadth-first order (neighbours in edge order) and returns g; use with Reap/Sow")
def _bfs(g, *args):
    return _scan(g, args, "BreadthFirstScan", True)


@fn("DepthFirstScan", area=AREA, status="added",
    notes="DepthFirstScan[g, s, {\"event\" -> f, ...}]: as BreadthFirstScan, in depth-first order; returns g")
def _dfs(g, *args):
    return _scan(g, args, "DepthFirstScan", False)


@fn("VertexOutComponent", area=AREA, status="added",
    notes="vertices reachable from v (v first, then breadth-first order); VertexOutComponent[g, v, k] within k steps")
def _voc(g, v, k=None):
    return _component_dir(g, v, k, "VertexOutComponent", False)


@fn("VertexInComponent", area=AREA, status="added",
    notes="vertices that can reach v (v first, then breadth-first order); VertexInComponent[g, v, k] within k steps")
def _vic(g, v, k=None):
    return _component_dir(g, v, k, "VertexInComponent", True)


def _component_dir(g, v, k, fname, reverse):
    gr = graph_of(g, fname)
    gr.require_pure(fname)
    s = gr.vidx(v, fname)
    lim = None if k is None else _int_arg(k, fname, 0)
    adj = gr.in_adj() if reverse else gr.out_adj()
    dist = {s: 0}
    order = [s]
    q = deque([s])
    while q:
        u = q.popleft()
        if lim is not None and dist[u] >= lim:
            continue
        for w, _, _ in adj[u]:
            if w not in dist:
                dist[w] = dist[u] + 1
                order.append(w)
                q.append(w)
    return L(*(gr.vertices[i] for i in order))


# ─────────────────────────── clustering ───────────────────────────

def _local_cc(gr):
    nb = gr.neighbor_sets(undirected=True)
    out = []
    for i in range(gr.n):
        d = len(nb[i])
        if d < 2:
            out.append(sp.Integer(0))
            continue
        links = sum(1 for a, b in itertools.combinations(sorted(nb[i]), 2) if b in nb[a])
        out.append(sp.Rational(2 * links, d * (d - 1)))
    return out


@fn("LocalClusteringCoefficient", area=AREA, status="added",
    notes="fraction of pairs of neighbours that are adjacent (0 for degree < 2), exact; undirected simple graphs")
def _lcc(g, v=None):
    gr = graph_of(g, "LocalClusteringCoefficient")
    gr.require_undirected("LocalClusteringCoefficient")
    c = _local_cc(gr)
    if v is None:
        return R(c)
    return R(c[gr.vidx(v, "LocalClusteringCoefficient")])


@fn("MeanClusteringCoefficient", area=AREA, status="added",
    notes="mean of LocalClusteringCoefficient over all vertices, exact; undirected graphs")
def _mcc(g):
    gr = graph_of(g, "MeanClusteringCoefficient")
    gr.require_undirected("MeanClusteringCoefficient")
    if gr.n == 0:
        raise ValueError("MeanClusteringCoefficient: the graph has no vertices")
    c = _local_cc(gr)
    return R(sp.Add(*c) / gr.n)


@fn("GlobalClusteringCoefficient", area=AREA, status="added",
    notes="3 x triangles / connected triples (transitivity), exact; 0 when there are no triples; undirected graphs")
def _gcc(g):
    gr = graph_of(g, "GlobalClusteringCoefficient")
    gr.require_undirected("GlobalClusteringCoefficient")
    nb = gr.neighbor_sets(undirected=True)
    triples = sum(len(s) * (len(s) - 1) // 2 for s in nb)
    closed = sum(1 for i in range(gr.n) for a, b in itertools.combinations(sorted(nb[i]), 2) if b in nb[a])
    if triples == 0:
        return 0
    return R(sp.Rational(closed, triples))
