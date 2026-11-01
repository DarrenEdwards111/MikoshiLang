"""Graph functions on edge lists {{u, v}, ...} (undirected; u <-> v and u -> v
also accepted, weights as {u, v, w}).  Pure Python, exact; legacy silently
ignored malformed edges and needed NetworkX for everything."""
from collections import deque
import heapq
import itertools

import sympy as sp

from ..expr import Expr, Symbol
from .core import Unevaluated, fn, error, record, S, R, L, rule, list_arg, is_list, number

AREA = "graphs"


class G:
    def __init__(self, edges_expr, fname):
        items = list_arg(edges_expr, fname, "edge list")
        self.vertices, self.adj, self.edges = [], {}, []
        for e in items:
            if is_list(e) and len(e.args) in (2, 3):
                u, v = e.args[0], e.args[1]
                w = number(e.args[2], fname, "edge weight") if len(e.args) == 3 else sp.Integer(1)
            elif isinstance(e, Expr) and e.head in ("UndirectedEdge", "Rule", "DirectedEdge", "TwoWayRule") \
                    and len(e.args) == 2:
                u, v, w = e.args[0], e.args[1], sp.Integer(1)
            else:
                raise ValueError(f"{fname}: invalid edge {e}; edges must be {{u, v}}, {{u, v, w}} or u <-> v")
            if w < 0:
                raise ValueError(f"{fname}: negative edge weights are not supported")
            for x in (u, v):
                if self.key(x) not in self.adj:
                    self.adj[self.key(x)] = {}
                    self.vertices.append(x)
            ku, kv = self.key(u), self.key(v)
            old = self.adj[ku].get(kv)
            if old is None or w < old:
                self.adj[ku][kv] = w
                self.adj[kv][ku] = w
            self.edges.append((u, v, w))
        self.byk = {self.key(x): x for x in self.vertices}

    @staticmethod
    def key(x):
        return repr(x) + "|" + type(x).__name__

    def has(self, x):
        return self.key(x) in self.adj

    def dijkstra(self, s):
        ks = self.key(s)
        dist, prev = {ks: sp.Integer(0)}, {}
        order = {self.key(v): i for i, v in enumerate(self.vertices)}
        pq = [(0.0, order[ks], ks)]
        done = set()
        while pq:
            _, _, u = heapq.heappop(pq)
            if u in done:
                continue
            done.add(u)
            for v, w in self.adj[u].items():
                nd = dist[u] + w
                if v not in dist or nd < dist[v]:
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (float(nd), order[v], v))
        return dist, prev

    def ecc(self):
        n = len(self.vertices)
        out = {}
        for v in self.vertices:
            d, _ = self.dijkstra(v)
            out[self.key(v)] = max(d.values()) if len(d) == n else sp.oo
        return out


def _vertex(g, v, fname):
    if not g.has(v):
        raise ValueError(f"{fname}: {v} is not a vertex of the graph")


@fn("ShortestPath", area=AREA, status="fixed",
    notes="vertex list of a shortest (weighted) path, {} when unreachable (Wolfram FindShortestPath); unknown "
          "vertices raise"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _sp(g, s, t):
    gr = G(g, "ShortestPath")
    _vertex(gr, s, "ShortestPath")
    _vertex(gr, t, "ShortestPath")
    dist, prev = gr.dijkstra(s)
    kt = gr.key(t)
    if kt not in dist:
        return L()
    path = [kt]
    while path[-1] != gr.key(s):
        path.append(prev[path[-1]])
    return L(*(gr.byk[k] for k in reversed(path)))


@fn("PathLength", area=AREA, status="fixed",
    notes="(weighted) shortest-path distance, Infinity when unreachable; unknown vertices raise"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _pl(g, s, t):
    gr = G(g, "PathLength")
    _vertex(gr, s, "PathLength")
    _vertex(gr, t, "PathLength")
    d, _ = gr.dijkstra(s)
    return R(d.get(gr.key(t), sp.oo))


@fn("Dijkstra", area=AREA, status="fixed", notes="{v -> distance} for every reachable vertex, in vertex order"
                                                 ". Check: exact (exact input gives exact results), reference "
                                                 "(differential cases in tests/library).")
def _dij(g, s):
    gr = G(g, "Dijkstra")
    _vertex(gr, s, "Dijkstra")
    d, _ = gr.dijkstra(s)
    return L(*(rule(v, R(d[gr.key(v)])) for v in gr.vertices if gr.key(v) in d))


def _components(gr):
    seen, comps = set(), []
    for v in gr.vertices:
        k = gr.key(v)
        if k in seen:
            continue
        comp, q = [], deque([k])
        seen.add(k)
        while q:
            u = q.popleft()
            comp.append(u)
            for w in gr.adj[u]:
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        order = {gr.key(x): i for i, x in enumerate(gr.vertices)}
        comps.append(sorted(comp, key=order.get))
    comps.sort(key=lambda c: -len(c))
    return comps


@fn("ConnectedComponents", area=AREA, status="fixed",
    notes="components largest first (Wolfram), vertices in order of appearance; legacy returned NetworkX set order")
def _cc(g):
    gr = G(g, "ConnectedComponents")
    return L(*(L(*(gr.byk[k] for k in c)) for c in _components(gr)))


@fn("ConnectedQ", area=AREA, status="verified", notes="True when the graph is connected"
                                                      ". Check: reference (differential cases in tests/library).")
def _connq(g):
    gr = G(g, "ConnectedQ")
    return len(_components(gr)) <= 1


@fn("BipartiteQ", area=AREA, status="verified", notes="2-colourability test (BFS)"
                                                      ". Check: reference (differential cases in tests/library).")
def _bip(g):
    gr = G(g, "BipartiteQ")
    color = {}
    for v in gr.vertices:
        k = gr.key(v)
        if k in color:
            continue
        color[k] = 0
        q = deque([k])
        while q:
            u = q.popleft()
            for w in gr.adj[u]:
                if w == u:
                    return False
                if w not in color:
                    color[w] = 1 - color[u]
                    q.append(w)
                elif color[w] == color[u]:
                    return False
    return True


@fn("GraphDiameter", area=AREA, status="fixed", notes="maximum eccentricity; Infinity for disconnected graphs (legacy returned None)")
def _diam(g):
    gr = G(g, "GraphDiameter")
    return R(max(gr.ecc().values()))


@fn("GraphRadius", area=AREA, status="fixed", notes="minimum eccentricity; Infinity for disconnected graphs (legacy returned None)")
def _rad(g):
    gr = G(g, "GraphRadius")
    return R(min(gr.ecc().values()))


@fn("GraphCenter", area=AREA, status="fixed",
    notes="vertices of minimum eccentricity; disconnected graphs raise (legacy returned {} silently)")
def _center(g):
    gr = G(g, "GraphCenter")
    e = gr.ecc()
    m = min(e.values())
    if m == sp.oo:
        raise ValueError("GraphCenter: graph is disconnected")
    return L(*(v for v in gr.vertices if e[gr.key(v)] == m))


@fn("VertexDegree", area=AREA, status="fixed",
    notes="number of incident edges (self-loops count twice, multi-edges counted); unknown vertices raise")
def _deg(g, v):
    gr = G(g, "VertexDegree")
    _vertex(gr, v, "VertexDegree")
    k = gr.key(v)
    d = 0
    for a, b, _ in gr.edges:
        d += (gr.key(a) == k) + (gr.key(b) == k)
    return d


@fn("ChromaticNumber", area=AREA, status="fixed",
    notes="exact chromatic number by backtracking (legacy used a greedy colouring, an upper bound only); "
          "graphs with more than 40 vertices raise"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _chrom(g):
    gr = G(g, "ChromaticNumber")
    ks = sorted((gr.key(v) for v in gr.vertices), key=lambda k: -len(gr.adj[k]))
    n = len(ks)
    if n == 0:
        return 0
    if any(k in gr.adj[k] for k in ks):
        raise ValueError("ChromaticNumber: graphs with self-loops cannot be coloured")
    if n > 40:
        raise ValueError("ChromaticNumber: graph too large for the exact algorithm")

    def colorable(c):
        col = {}

        def bt(i):
            if i == n:
                return True
            used = {col[w] for w in gr.adj[ks[i]] if w in col}
            for x in range(min(c, max(col.values(), default=-1) + 2)):
                if x not in used:
                    col[ks[i]] = x
                    if bt(i + 1):
                        return True
                    del col[ks[i]]
            return False
        return bt(0)

    for c in range(1, n + 1):
        if colorable(c):
            return c


@fn("ClusteringCoefficient", area=AREA, status="fixed",
    notes="ClusteringCoefficient[g] = mean local clustering (vertices of degree < 2 count 0, as Wolfram "
          "MeanClusteringCoefficient), [g, v] local coefficient; exact rationals (legacy floats)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _clust(g, v=None):
    gr = G(g, "ClusteringCoefficient")

    def local(k):
        nb = [w for w in gr.adj[k] if w != k]
        d = len(nb)
        if d < 2:
            return sp.Integer(0)
        links = sum(1 for a, b in itertools.combinations(nb, 2) if b in gr.adj[a])
        return sp.Rational(2 * links, d * (d - 1))
    if v is not None:
        _vertex(gr, v, "ClusteringCoefficient")
        return R(local(gr.key(v)))
    vals = [local(gr.key(x)) for x in gr.vertices]
    return R(sp.Add(*vals) / len(vals)) if vals else 0


@fn("BetweennessCentrality", area=AREA, status="fixed",
    notes="{v -> c} unnormalised betweenness (Wolfram BetweennessCentrality); legacy returned NetworkX's "
          "normalised values; exact rationals (Brandes, unweighted)")
def _betw(g):
    gr = G(g, "BetweennessCentrality")
    keys = [gr.key(v) for v in gr.vertices]
    cb = {k: sp.Integer(0) for k in keys}
    for s in keys:
        stack, pred = [], {k: [] for k in keys}
        sigma = {k: 0 for k in keys}
        sigma[s] = 1
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            stack.append(u)
            for w in gr.adj[u]:
                if w == u:
                    continue
                if w not in dist:
                    dist[w] = dist[u] + 1
                    q.append(w)
                if dist[w] == dist[u] + 1:
                    sigma[w] += sigma[u]
                    pred[w].append(u)
        delta = {k: sp.Integer(0) for k in keys}
        while stack:
            w = stack.pop()
            for u in pred[w]:
                delta[u] += sp.Rational(sigma[u], sigma[w]) * (1 + delta[w])
            if w != s:
                cb[w] += delta[w]
    return L(*(rule(v, R(cb[gr.key(v)] / 2)) for v in gr.vertices))


@fn("PageRank", area=AREA, status="fixed",
    notes="{v -> rank} PageRank with damping 0.85 treating each edge as two directed links (power iteration to "
          "1e-12, matches NetworkX); legacy required NetworkX"
          ". Check: reference (differential cases in tests/library).")
def _pagerank(g, alpha=0.85):
    gr = G(g, "PageRank")
    a = float(number(alpha, "PageRank", "damping"))
    keys = [gr.key(v) for v in gr.vertices]
    n = len(keys)
    if n == 0:
        return L()
    out_w = {k: sum(float(w) for w in gr.adj[k].values()) for k in keys}
    pr = {k: 1.0 / n for k in keys}
    for _ in range(1000):
        dangling = sum(pr[k] for k in keys if out_w[k] == 0)
        new = {}
        for k in keys:
            s = sum(pr[j] * float(gr.adj[j][k]) / out_w[j] for j in gr.adj[k] if out_w[j])
            new[k] = (1 - a) / n + a * (s + dangling / n)
        err = sum(abs(new[k] - pr[k]) for k in keys)
        pr = new
        if err < n * 1e-12:
            break
    return L(*(rule(v, pr[gr.key(v)]) for v in gr.vertices))


def _kruskal(gr):
    parent = {gr.key(v): gr.key(v) for v in gr.vertices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    out = []
    for u, v, w in sorted(gr.edges, key=lambda e: float(e[2])):
        a, b = find(gr.key(u)), find(gr.key(v))
        if a != b:
            parent[a] = b
            out.append(L(u, v))
    return L(*out)


@fn("MinimumSpanningTree", area=AREA, status="fixed",
    notes="edges of a minimum spanning forest (Kruskal, weights {u, v, w} supported); legacy needed NetworkX"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _mst(g):
    return _kruskal(G(g, "MinimumSpanningTree"))


@fn("SpanningTree", area=AREA, status="fixed", notes="edges of a spanning forest (Kruskal)"
                                                     ". Check: reference (differential cases in tests/library).")
def _st(g):
    return _kruskal(G(g, "SpanningTree"))


@fn("MaximalMatching", area=AREA, status="verified",
    notes="a maximum-cardinality matching (NetworkX max_weight_matching with unit weights), edges sorted; "
          "requires NetworkX"
          ". Check: reference (differential cases in tests/library).")
def _match(g):
    gr = G(g, "MaximalMatching")
    try:
        import networkx as nx
    except ImportError:
        raise NotImplementedError("MaximalMatching requires NetworkX (pip install mikoshilang[all])")
    X = nx.Graph()
    for u, v, _ in gr.edges:
        X.add_edge(gr.key(u), gr.key(v))
    m = nx.max_weight_matching(X, maxcardinality=True)
    order = {gr.key(x): i for i, x in enumerate(gr.vertices)}
    pairs = sorted((tuple(sorted(e, key=order.get)) for e in m), key=lambda p: order[p[0]])
    return L(*(L(gr.byk[a], gr.byk[b]) for a, b in pairs))
