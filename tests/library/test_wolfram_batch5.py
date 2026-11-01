"""Differential tests for coverage batch 5 (graphs, statistics and probability).

Expected values are Wolfram Language results (reference.wolfram.com/language/ref/<Name>.html examples, or values
that follow from the documented definitions), or independent references computed here with NetworkX, SciPy, NumPy,
mpmath and SymPy.  Functions that return one of several valid answers (a cycle, a clique, a path, ...) are checked
with predicates that verify the answer is valid and optimal.
"""
import importlib.util
import itertools
import math

import mpmath as mp
import pytest
import sympy as sp

from _harness import Pred, Raises, check, ids, to_complex

HAVE_NX = importlib.util.find_spec("networkx") is not None
HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
HAVE_NUMPY = importlib.util.find_spec("numpy") is not None


def Near(want, tol=1e-12):
    return Pred(lambda g: abs(to_complex(g) - complex(want)) <= tol * max(1, abs(complex(want))), f"~{want}")


def NearList(want, tol=1e-9):
    def ok(g):
        return (getattr(g, "head", None) == "List" and len(g.args) == len(want)
                and all(abs(to_complex(a) - complex(b)) <= tol * max(1, abs(complex(b))) for a, b in zip(g.args, want)))
    return Pred(ok, f"~{list(want)}")


def py(x):
    """MikoshiLang value -> plain Python (lists, ints, floats, strings)."""
    from mikoshilang.expr import Expr, Symbol
    if isinstance(x, Expr) and x.head == "List":
        return [py(a) for a in x.args]
    if isinstance(x, Expr) and x.head in ("UndirectedEdge", "DirectedEdge"):
        return (x.head, py(x.args[0]), py(x.args[1]))
    if isinstance(x, Symbol):
        return x.name
    return x


# ─────────────────────────── graph references ───────────────────────────

PETERSEN_EDGES = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
                  (6, 8), (8, 10), (7, 10), (7, 9), (6, 9)]
G1 = "Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4}]"
G1E = [(1, 2), (2, 3), (3, 1), (3, 4)]
D1 = "Graph[{1 -> 2, 2 -> 3, 3 -> 1, 3 -> 4, 4 -> 5, 5 -> 4}]"
D1E = [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 4)]
H = "Graph[{1 <-> 2, 1 <-> 3, 2 <-> 3, 2 <-> 4, 3 <-> 5, 4 <-> 5, 4 <-> 6, 5 <-> 6, 6 <-> 7}]"
HE = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 6), (6, 7)]
W = "Graph[{1 <-> 2, 2 <-> 3, 1 <-> 3, 3 <-> 4}, EdgeWeight -> {4, 1, 2, 7}]"
WE = [(1, 2, 4), (2, 3, 1), (1, 3, 2), (3, 4, 7)]


def gsrc(edges, directed=False):
    op = " -> " if directed else " <-> "
    return "Graph[{" + ", ".join(f"{a}{op}{b}" for a, b in edges) + "}]"


def is_cycle(g_edges, directed=False):
    """Predicate: a list containing one cycle of edges, consecutive, closed, each edge in the graph once."""
    gset = [tuple(e) for e in g_edges]

    def ok(r):
        r = py(r)
        if not r or len(r) != 1:
            return False
        cyc = r[0]
        if not cyc:
            return False
        for (h, a, b), (h2, c, d) in zip(cyc, cyc[1:] + cyc[:1]):
            if b != c:
                return False
        used = []
        for h, a, b in cyc:
            if directed:
                if (a, b) not in gset:
                    return False
            elif (a, b) not in gset and (b, a) not in gset:
                return False
            used.append((a, b) if directed else frozenset((a, b)))
        return len(set(used)) == len(used)
    return Pred(ok, "a cycle of the graph")


def nx_ref(f):
    """Evaluate f() lazily (NetworkX may be missing in the minimal install)."""
    return f() if HAVE_NX else None


if HAVE_NX:
    import networkx as nx
    import numpy as np

    def G(edges, directed=False, weighted=False):
        X = nx.DiGraph() if directed else nx.Graph()
        for e in edges:
            if weighted:
                X.add_edge(e[0], e[1], weight=e[2])
            else:
                X.add_edge(*e)
        return X

    PET = G(PETERSEN_EDGES)
    HX = G(HE)

    def nx_dist_matrix(X, order):
        d = dict(nx.all_pairs_shortest_path_length(X))
        return [[d[u].get(v, math.inf) for v in order] for u in order]

    def nx_closeness(X, order):
        c = nx.closeness_centrality(X)
        return [c[v] for v in order]

    def nx_eig(X, order):
        c = nx.eigenvector_centrality_numpy(X)
        s = sum(c.values())
        return [c[v] / s for v in order]

    def nx_pr(X, order, a):
        c = nx.pagerank(X, alpha=a, tol=1e-15, max_iter=100000)
        return [c[v] for v in order]

    def nx_btw(X, order, weight=None):
        c = nx.betweenness_centrality(X, normalized=False, weight=weight)
        return [c[v] for v in order]

    def nx_hits(X, order):
        h, a = nx.hits(X, max_iter=100000, tol=1e-14)
        auth = np.array([a[v] for v in order])
        auth = auth / auth.sum()
        A = nx.to_numpy_array(X, nodelist=order)
        return [list(auth), list(A @ auth)]

    def nx_chrom(edges, var):
        X = nx.Graph(edges)
        return str(sp.expand(nx.chromatic_polynomial(X).subs(sp.Symbol("x"), sp.Symbol(var)))).replace("**", "^")

    def valid_topo(edges):
        def ok(r):
            r = py(r)
            pos = {v: i for i, v in enumerate(r)}
            return all(pos[a] < pos[b] for a, b in edges)
        return Pred(ok, "a topological order")

    def is_clique(X, size):
        def ok(r):
            r = py(r)
            return len(r) == 1 and len(r[0]) == size and all(X.has_edge(a, b) for a, b in itertools.combinations(r[0], 2))
        return Pred(ok, f"a clique of size {size}")

    def is_indep(X, size):
        def ok(r):
            r = py(r)
            return len(r) == 1 and len(r[0]) == size and not any(X.has_edge(a, b) for a, b in itertools.combinations(r[0], 2))
        return Pred(ok, f"an independent set of size {size}")

    def is_cover(X, size):
        def ok(r):
            r = set(py(r))
            return len(r) == size and all(a in r or b in r for a, b in X.edges())
        return Pred(ok, f"a vertex cover of size {size}")

    def is_ham(X):
        def ok(r):
            r = py(r)
            if len(r) != 1:
                return False
            vs = [e[1] for e in r[0]]
            return sorted(vs) == sorted(X.nodes()) and all(X.has_edge(e[1], e[2]) for e in r[0]) and \
                all(a[2] == b[1] for a, b in zip(r[0], r[0][1:] + r[0][:1]))
        return Pred(ok, "a Hamiltonian cycle")

    def is_euler(X):
        def ok(r):
            r = py(r)
            if len(r) != 1 or len(r[0]) != X.number_of_edges():
                return False
            seq = r[0]
            used = [frozenset((e[1], e[2])) for e in seq]
            return len(set(used)) == len(used) and all(a[2] == b[1] for a, b in zip(seq, seq[1:] + seq[:1])) and \
                all(X.has_edge(e[1], e[2]) for e in seq)
        return Pred(ok, "an Eulerian cycle")

    GRAPHS_NX = [
        # distances against NetworkX
        ("GraphDistanceMatrix[" + H + "]", "{" + ", ".join("{" + ", ".join(str(x) for x in row) + "}"
                                                         for row in nx_dist_matrix(HX, range(1, 8))) + "}"),
        ("GraphDistanceMatrix[PetersenGraph[]]", "{" + ", ".join("{" + ", ".join(str(x) for x in row) + "}"
                                                                 for row in nx_dist_matrix(PET, range(1, 11))) + "}"),
        ("GraphDistanceMatrix[" + D1 + "]", "{" + ", ".join("{" + ", ".join("Infinity" if x == math.inf else str(x) for x in row) + "}"
                                                            for row in nx_dist_matrix(G(D1E, True), range(1, 6))) + "}"),
        ("GraphDistance[" + W + ", 1, 4]", str(nx.dijkstra_path_length(G(WE, weighted=True), 1, 4))),
        ("FindShortestPath[" + W + ", 1, 4]", "{1, 3, 4}"),
        ("GraphDistance[" + W + ", 2]", "{" + ", ".join(str(nx.dijkstra_path_length(G(WE, weighted=True), 2, v)) for v in range(1, 5)) + "}"),
        ("FindShortestPath[" + H + ", 1, 7]", Pred(lambda r: len(r.args) == nx.shortest_path_length(HX, 1, 7) + 1
                                                   and r.args[0] == 1 and r.args[-1] == 7
                                                   and all(HX.has_edge(a, b) for a, b in zip(r.args, r.args[1:])), "shortest")),
        # centralities
        ("ClosenessCentrality[" + H + "]", NearList(nx_closeness(HX, range(1, 8)))),
        ("ClosenessCentrality[PetersenGraph[]]", NearList([1 / 1.6667] * 0 + nx_closeness(PET, range(1, 11)))),
        ("ClosenessCentrality[GridGraph[{3, 3}]]", NearList(nx_closeness(nx.convert_node_labels_to_integers(
            nx.grid_2d_graph(3, 3), 1, ordering="sorted"), range(1, 10)))),
        ("EigenvectorCentrality[" + H + "]", NearList(nx_eig(HX, range(1, 8)), 1e-8)),
        ("EigenvectorCentrality[" + G1 + "]", NearList(nx_eig(G(G1E), range(1, 5)), 1e-8)),
        ("EigenvectorCentrality[" + D1 + "]", Raises()),
        ("PageRankCentrality[" + H + ", 0.85]", NearList(nx_pr(HX, range(1, 8), 0.85), 1e-9)),
        ("PageRankCentrality[" + G1 + ", 0.5]", NearList(nx_pr(G(G1E), range(1, 5), 0.5), 1e-9)),
        ("PageRankCentrality[Graph[{1 -> 2, 2 -> 3, 3 -> 1, 1 -> 3}], 0.85]",
         NearList(nx_pr(G([(1, 2), (2, 3), (3, 1), (1, 3)], True), range(1, 4), 0.85), 1e-9)),
        ("PageRankCentrality[Graph[{1 -> 2}], 0.85]", Raises()),
        ("BetweennessCentrality[" + H + "]", NearList(nx_btw(HX, range(1, 8)))),
        ("BetweennessCentrality[" + D1 + "]", NearList(nx_btw(G(D1E, True), range(1, 6)))),
        ("BetweennessCentrality[" + W + "]", NearList(nx_btw(G(WE, weighted=True), range(1, 5), "weight"))),
        ("HITSCentrality[Graph[{1 -> 2, 1 -> 3, 2 -> 3, 3 -> 4, 4 -> 1, 2 -> 4}]]",
         Pred(lambda r: all(abs(float(a) - b) < 1e-8 for got, want in zip(r.args, nx_hits(
             G([(1, 2), (1, 3), (2, 3), (3, 4), (4, 1), (2, 4)], True), [1, 2, 3, 4])) for a, b in zip(got.args, want)),
             "NetworkX HITS authorities and A.x hubs")),
        ("HITSCentrality[" + H + "]", Pred(lambda r: all(abs(float(a) - b) < 1e-8 for got, want in zip(r.args, nx_hits(
            HX, list(range(1, 8)))) for a, b in zip(got.args, want)), "NetworkX HITS")),
        ("Total[First[HITSCentrality[" + H + "]]]", Near(1.0)),
        # structure
        ("FindMinimumCut[" + H + "]", Pred(lambda r: r.args[0] == nx.stoer_wagner(HX)[0]
                                           and sum(1 for a, b in HX.edges() if (a in py(r.args[1])[0]) != (b in py(r.args[1])[0])) == r.args[0],
                                           "Stoer-Wagner value and a matching partition")),
        ("FindMinimumCut[" + W + "]", Pred(lambda r: r.args[0] == nx.stoer_wagner(G(WE, weighted=True))[0], "weighted value")),
        ("FindMinimumCut[PetersenGraph[]]", Pred(lambda r: r.args[0] == 3, "edge connectivity 3")),
        ("FindMaximumFlow[" + H + ", 1, 7]", str(nx.maximum_flow_value(nx.DiGraph([(a, b) for a, b in HE] + [(b, a) for a, b in HE]), 1, 7, capacity="c")
                                               if False else nx.edge_connectivity(HX, 1, 7))),
        ("FindMaximumFlow[Graph[{1 -> 2, 1 -> 3, 2 -> 3, 2 -> 4, 3 -> 4}, EdgeCapacity -> {3, 2, 1, 1, 4}], 1, 4]",
         str(nx.maximum_flow_value(nx.DiGraph([(1, 2, {"capacity": 3}), (1, 3, {"capacity": 2}), (2, 3, {"capacity": 1}),
                                                (2, 4, {"capacity": 1}), (3, 4, {"capacity": 4})]), 1, 4))),
        ("FindMaximumFlow[Graph[{1 -> 2, 2 -> 3}], 1, 3, EdgeCapacity -> {5/2, 7}]", "5/2"),
        ("FindMaximumFlow[Graph[{1 -> 2, 3 -> 2}], 1, 3]", "0"),
        ("EdgeConnectivity[PetersenGraph[]]", str(nx.edge_connectivity(PET))),
        ("EdgeConnectivity[" + H + "]", str(nx.edge_connectivity(HX))),
        ("EdgeConnectivity[" + H + ", 1, 7]", str(nx.edge_connectivity(HX, 1, 7))),
        ("FindClique[" + H + "]", is_clique(HX, max(len(c) for c in nx.find_cliques(HX)))),
        ("FindClique[CompleteGraph[5]]", "{{1, 2, 3, 4, 5}}"),
        ("FindClique[PetersenGraph[]]", is_clique(PET, 2)),
        ("FindClique[" + H + ", {3}, All]", Pred(lambda r: sorted(map(sorted, py(r))) == sorted(sorted(c) for c in nx.enumerate_all_cliques(HX) if len(c) == 3), "all triangles")),
        ("FindIndependentVertexSet[PetersenGraph[]]", is_indep(PET, 4)),
        ("FindIndependentVertexSet[" + H + "]", is_indep(HX, max(len(c) for c in nx.find_cliques(nx.complement(HX))))),
        ("FindIndependentVertexSet[CycleGraph[7]]", is_indep(nx.cycle_graph(range(1, 8)), 3)),
        ("FindVertexCover[PetersenGraph[]]", is_cover(PET, 6)),
        ("FindVertexCover[" + H + "]", is_cover(HX, 7 - max(len(c) for c in nx.find_cliques(nx.complement(HX))))),
        ("FindVertexCover[StarGraph[6]]", "{1}"),
        ("ChromaticPolynomial[" + H + ", k]", nx_chrom(HE, "k")),
        ("ChromaticPolynomial[PetersenGraph[], x]", nx_chrom(PETERSEN_EDGES, "x")),
        ("ChromaticPolynomial[" + G1 + ", k]", nx_chrom(G1E, "k")),
        ("IsomorphicGraphQ[PetersenGraph[], Graph[{a <-> b, b <-> c, c <-> d, d <-> e, e <-> a, a <-> f, b <-> g, c <-> h, d <-> i, e <-> j, f <-> h, h <-> j, j <-> g, g <-> i, i <-> f}]]", True),
        ("IsomorphicGraphQ[" + H + ", " + gsrc(HE[:-1] + [(5, 7)]) + "]", nx.is_isomorphic(HX, G(HE[:-1] + [(5, 7)]))),
        ("IsomorphicGraphQ[GridGraph[{2, 2, 2}], HypercubeGraph[3]]", True),
        ("IsomorphicGraphQ[CycleGraph[6], " + gsrc([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)]) + "]", False),
        ("FindGraphIsomorphism[" + H + ", " + gsrc([(b, a) for a, b in HE]) + "]",
         Pred(lambda r: len(r.args) == 1 and len(r.args[0].args) == 7, "one isomorphism")),
        ("FindGraphIsomorphism[CycleGraph[4], CycleGraph[4], All]", Pred(lambda r: len(r.args) == 8, "|Aut(C4)| = 8")),
        ("FindGraphIsomorphism[CycleGraph[4], StarGraph[4]]", "{}"),
        ("FindHamiltonianCycle[" + H + "]", is_ham(HX) if False else "{}"),
        ("FindHamiltonianCycle[HypercubeGraph[3]]", is_ham(nx.hypercube_graph(3) if False else G(
            [(i + 1, j + 1) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]))),
        ("FindHamiltonianCycle[PetersenGraph[]]", "{}"),
        ("FindHamiltonianCycle[CompleteGraph[5]]", is_ham(nx.complete_graph(range(1, 6)))),
        ("FindEulerianCycle[CompleteGraph[5]]", is_euler(nx.complete_graph(range(1, 6)))),
        ("FindEulerianCycle[" + G1 + "]", "{}"),
        ("FindEulerianCycle[GridGraph[{3, 3}]]", "{}"),
        ("FindEulerianCycle[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1, 1 <-> 4, 4 <-> 5, 5 <-> 1}]]",
         is_euler(G([(1, 2), (2, 3), (3, 1), (1, 4), (4, 5), (5, 1)]))),
        ("EulerianGraphQ[CompleteGraph[5]]", nx.is_eulerian(nx.complete_graph(5))),
        ("EulerianGraphQ[CompleteGraph[4]]", nx.is_eulerian(nx.complete_graph(4))),
        ("EulerianGraphQ[" + D1 + "]", False),
        ("PlanarGraphQ[PetersenGraph[]]", False), ("PlanarGraphQ[CompleteGraph[{3, 3}]]", False),
        ("PlanarGraphQ[GridGraph[{4, 5}]]", True), ("PlanarGraphQ[CompleteGraph[4]]", True),
        ("TopologicalSort[Graph[{5 -> 11, 7 -> 11, 7 -> 8, 3 -> 8, 3 -> 10, 11 -> 2, 11 -> 9, 11 -> 10, 8 -> 9}]]",
         valid_topo([(5, 11), (7, 11), (7, 8), (3, 8), (3, 10), (11, 2), (11, 9), (11, 10), (8, 9)])),
        ("TopologicalSort[Graph[{1 -> 2, 1 -> 3, 3 -> 2, 2 -> 4}]]", "{1, 3, 2, 4}"),
        ("TopologicalSort[" + D1 + "]", Raises()), ("TopologicalSort[" + G1 + "]", Raises()),
        ("FindCycle[" + H + "]", is_cycle(HE)),
        ("FindCycle[" + D1 + "]", is_cycle(D1E, True)),
        ("FindCycle[" + H + ", {4}, All]", Pred(lambda r: len(r.args) == sum(1 for c in nx.simple_cycles(HX) if len(c) == 4), "all 4-cycles")),
        ("Length[FindCycle[PetersenGraph[], {5}, All]]", str(sum(1 for c in nx.simple_cycles(PET) if len(c) == 5))),
        ("FindCycle[KaryTree[10]]", "{}"),
        ("Length[FindCycle[" + D1 + ", Infinity, All]]", str(sum(1 for _ in nx.simple_cycles(G(D1E, True))))),
        ("ConnectedComponents[" + D1 + "]", Pred(lambda r: sorted(map(sorted, py(r))) == sorted(map(sorted, nx.strongly_connected_components(G(D1E, True)))), "SCCs")),
        ("WeaklyConnectedComponents[" + D1 + "]", "{{1, 2, 3, 4, 5}}"),
        ("GraphDiameter[" + H + "]", str(nx.diameter(HX))), ("GraphRadius[" + H + "]", str(nx.radius(HX))),
        ("GraphCenter[" + H + "]", "{" + ", ".join(map(str, sorted(nx.center(HX)))) + "}"),
        ("GraphPeriphery[" + H + "]", "{" + ", ".join(map(str, sorted(nx.periphery(HX)))) + "}"),
        ("VertexEccentricity[" + H + ", 1]", str(nx.eccentricity(HX, 1))),
        ("VertexEccentricity[PetersenGraph[], 3]", "2"),
        ("MeanClusteringCoefficient[" + H + "]", Near(nx.average_clustering(HX))),
        ("LocalClusteringCoefficient[" + H + "]", NearList([nx.clustering(HX, v) for v in range(1, 8)])),
        ("GlobalClusteringCoefficient[" + H + "]", Near(nx.transitivity(HX))),
        ("LocalClusteringCoefficient[" + G1 + ", 3]", "1/3"),
        ("MeanClusteringCoefficient[CompleteGraph[5]]", "1"), ("GlobalClusteringCoefficient[PetersenGraph[]]", "0"),
        ("FindSpanningTree[" + W + "]", Pred(lambda r: sorted(map(lambda e: tuple(sorted(e[1:])), py(r.args[1]))) ==
                                              sorted(tuple(sorted(e[:2])) for e in nx.minimum_spanning_edges(G(WE, weighted=True), data=False)),
                                              "the minimum spanning tree")),
        ("EdgeCount[FindSpanningTree[PetersenGraph[]]]", "9"),
        ("TreeGraphQ[FindSpanningTree[" + H + "]]", True),
        ("FindSpanningTree[Graph[{1, 2, 3, 4}, {1 <-> 2, 3 <-> 4}]]", "Graph[{1, 2, 3, 4}, {1 <-> 2, 3 <-> 4}]"),
        ("GraphDensity[" + H + "]", str(sp.nsimplify(nx.density(HX)))),
        ("GraphDensity[" + D1 + "]", str(sp.nsimplify(nx.density(G(D1E, True))))),
        ("GraphDensity[CompleteGraph[6]]", "1"),
    ]
else:
    GRAPHS_NX = []

GRAPHS = [
    # construction and normal form (InputForm of Graph is Graph[{vertices}, {edges}])
    ("Graph[{1 <-> 2, 2 <-> 3}]", "Graph[{1, 2, 3}, {UndirectedEdge[1, 2], UndirectedEdge[2, 3]}]"),
    ("Graph[{1 -> 2, 3 -> 1}]", "Graph[{1, 2, 3}, {DirectedEdge[1, 2], DirectedEdge[3, 1]}]"),
    ("Graph[{a, b, c}, {a -> b}]", "Graph[{a, b, c}, {DirectedEdge[a, b]}]"),
    ("Graph[{1 \\[UndirectedEdge] 2, 2 \\[DirectedEdge] 3}]", "Graph[{1, 2, 3}, {UndirectedEdge[1, 2], DirectedEdge[2, 3]}]"),
    ("Graph[Graph[{1 <-> 2}]]", "Graph[{1, 2}, {UndirectedEdge[1, 2]}]"),
    ("Graph[{1, 2}, {1 <-> 3}]", Raises()), ("Graph[{{1, 2}}]", Raises()), ("Graph[{1 <-> 2}, Foo -> 1]", Raises()),
    ("Graph[{UndirectedEdge[1, 2, t]}]", "Graph[{1, 2}, {UndirectedEdge[1, 2, t]}]"),  # tags kept since W24
    ("FindShortestPath[Graph[{UndirectedEdge[1, 2, t]}], 1, 2]", Raises()), ("Graph[{1, 1}, {}]", Raises()),
    ("Graph[{1 <-> 2}, EdgeWeight -> {1, 2}]", Raises()), ("Graph[{1 <-> 2}, EdgeWeight -> {-1}]", Raises()),
    ("Graph[{1 <-> 2}, VertexWeight -> {1}]", Raises()),
    ("1 <-> 2", "TwoWayRule[1, 2]"), ("(a <-> b)[[2]]", "b"), ("x \\[Distributed] d", "Distributed[x, d]"),
    ("UndirectedEdge[1, 2]", "UndirectedEdge[1, 2]"), ("DirectedEdge[1, 2] === (1 \\[DirectedEdge] 2)", True),
    ("UndirectedEdge[1]", Raises()), ("DirectedEdge[1]", Raises()), ("TwoWayRule[1]", Raises()), ("TwoWayRule[a, b] === (a <-> b)", True),
    ("Graph[{TwoWayRule[1, 2]}]", "Graph[{1, 2}, {UndirectedEdge[1, 2]}]"),
    ("Head[a <-> b]", "TwoWayRule"), ("EdgeList[Graph[{a <-> b, b -> c}]]", "{UndirectedEdge[a, b], DirectedEdge[b, c]}"),
    ("GraphQ[Graph[{1 <-> 2}]]", True), ("GraphQ[{1 <-> 2}]", False), ("GraphQ[f[1 <-> 2]]", False),
    ("GraphQ[5]", False),
    # kinds
    ("DirectedGraphQ[" + D1 + "]", True), ("DirectedGraphQ[" + G1 + "]", False), ("DirectedGraphQ[x]", "DirectedGraphQ[x]"),
    ("UndirectedGraphQ[" + G1 + "]", True), ("UndirectedGraphQ[" + D1 + "]", False), ("UndirectedGraphQ[Graph[{1, 2}, {}]]", True),
    ("MixedGraphQ[Graph[{1 <-> 2, 2 -> 3}]]", True), ("MixedGraphQ[" + G1 + "]", False), ("MixedGraphQ[" + D1 + "]", False),
    ("SimpleGraphQ[" + G1 + "]", True), ("SimpleGraphQ[Graph[{1 <-> 2, 1 <-> 2}]]", False), ("SimpleGraphQ[Graph[{1 <-> 1}]]", False),
    ("LoopFreeGraphQ[Graph[{1 <-> 1}]]", False), ("LoopFreeGraphQ[" + G1 + "]", True), ("LoopFreeGraphQ[Graph[{1 -> 1, 1 -> 2}]]", False),
    ("MultigraphQ[Graph[{1 <-> 2, 2 <-> 1}]]", True), ("MultigraphQ[Graph[{1 -> 2, 2 -> 1}]]", False), ("MultigraphQ[" + G1 + "]", False),
    ("EmptyGraphQ[Graph[{1, 2}, {}]]", True), ("EmptyGraphQ[" + G1 + "]", False), ("EmptyGraphQ[Graph[{}, {}]]", True),
    ("EdgeWeightedGraphQ[" + W + "]", True), ("EdgeWeightedGraphQ[" + G1 + "]", False),
    ("EdgeWeightedGraphQ[Graph[{1 <-> 2}, EdgeWeight -> {2}]]", True),
    ("VertexWeightedGraphQ[Graph[{1 <-> 2}, VertexWeight -> {3, 4}]]", True), ("VertexWeightedGraphQ[" + W + "]", False),
    ("VertexWeightedGraphQ[" + G1 + "]", False),
    ("WeightedGraphQ[" + W + "]", True), ("WeightedGraphQ[" + G1 + "]", False),
    ("WeightedGraphQ[Graph[{1 <-> 2}, VertexWeight -> {3, 4}]]", True),
    ("CompleteGraphQ[CompleteGraph[5]]", True), ("CompleteGraphQ[" + G1 + "]", False),
    ("CompleteGraphQ[Graph[{1 -> 2, 2 -> 1}]]", True),
    # basic queries
    ("VertexList[" + G1 + "]", "{1, 2, 3, 4}"), ("VertexList[{3 -> 1, 1 -> 2}]", "{3, 1, 2}"),
    ("VertexList[Graph[{x, y}, {}]]", "{x, y}"),
    ("EdgeList[{1 -> 2, 2 <-> 3}]", "{DirectedEdge[1, 2], UndirectedEdge[2, 3]}"), ("EdgeList[Graph[{1, 2}, {}]]", "{}"),
    ("EdgeList[CycleGraph[3]]", "{UndirectedEdge[1, 2], UndirectedEdge[2, 3], UndirectedEdge[1, 3]}"),
    ("VertexCount[PetersenGraph[]]", "10"), ("VertexCount[" + G1 + "]", "4"), ("VertexCount[Graph[{}, {}]]", "0"),
    ("EdgeCount[PetersenGraph[]]", "15"), ("EdgeCount[Graph[{1 <-> 2, 1 <-> 2, 2 <-> 2}]]", "3"), ("EdgeCount[CompleteGraph[7]]", "21"),
    ("VertexQ[" + G1 + ", 4]", True), ("VertexQ[" + G1 + ", 5]", False), ("VertexQ[" + G1 + ", \"4\"]", False),
    ("EdgeQ[" + G1 + ", 2 <-> 1]", True), ("EdgeQ[" + D1 + ", 2 -> 1]", False), ("EdgeQ[" + D1 + ", 1 -> 2]", True),
    ("EdgeQ[" + G1 + ", 1 -> 2]", False),
    ("VertexIndex[" + G1 + ", 3]", "3"), ("VertexIndex[Graph[{c <-> a, a <-> b}], b]", "3"), ("VertexIndex[" + G1 + ", 9]", Raises()),
    ("EdgeIndex[" + G1 + ", 1 <-> 3]", "3"), ("EdgeIndex[" + D1 + ", 4 -> 5]", "5"), ("EdgeIndex[" + D1 + ", 2 -> 1]", Raises()),
    ("VertexDegree[" + G1 + "]", "{2, 2, 3, 1}"), ("VertexDegree[" + G1 + ", 3]", "3"),
    ("VertexDegree[Graph[{1 <-> 1, 1 <-> 2}]]", "{3, 1}"), ("VertexDegree[" + D1 + "]", "{2, 2, 3, 3, 2}"),
    ("VertexDegree[{1 -> 2, 2 -> 3}, 2]", "2"),
    ("VertexInDegree[" + D1 + "]", "{1, 1, 1, 2, 1}"), ("VertexInDegree[" + D1 + ", 4]", "2"), ("VertexInDegree[" + G1 + "]", "{2, 2, 3, 1}"),
    ("VertexOutDegree[" + D1 + "]", "{1, 1, 2, 1, 1}"), ("VertexOutDegree[" + D1 + ", 3]", "2"), ("VertexOutDegree[" + G1 + ", 4]", "1"),
    ("DegreeCentrality[" + G1 + "]", "{2, 2, 3, 1}"), ("DegreeCentrality[" + D1 + ", \"In\"]", "{1, 1, 1, 2, 1}"),
    ("DegreeCentrality[" + D1 + ", \"Out\"]", "{1, 1, 2, 1, 1}"), ("DegreeCentrality[" + D1 + ", \"Both\"]", Raises()),
    # matrices (reference page examples and definitions)
    ("AdjacencyMatrix[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1}]]", "{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}}"),
    ("AdjacencyMatrix[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]", "{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}"),
    ("AdjacencyMatrix[{1 -> 2, 2 -> 1, 3 -> 1, 3 -> 2, 4 -> 1, 4 -> 2}]",
     "{{0, 1, 0, 0}, {1, 0, 0, 0}, {1, 1, 0, 0}, {1, 1, 0, 0}}"),
    ("AdjacencyMatrix[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1, 2 <-> 2}]]", "{{0, 1, 1}, {1, 1, 1}, {1, 1, 0}}"),
    ("AdjacencyMatrix[Graph[{1 <-> 2, 1 <-> 2}]]", "{{0, 2}, {2, 0}}"),
    ("AdjacencyMatrix[Graph[{2 <-> 3, 3 <-> 1, 1 <-> 2, 1 <-> 4}]]", "{{0, 1, 1, 0}, {1, 0, 1, 0}, {1, 1, 0, 1}, {0, 0, 1, 0}}"),
    ("AdjacencyMatrix[Graph[{}, {}]]", Raises()),
    ("Total[AdjacencyMatrix[PetersenGraph[]]]", "{3, 3, 3, 3, 3, 3, 3, 3, 3, 3}"),
    ("WeightedAdjacencyMatrix[" + W + "]", "{{0, 4, 2, 0}, {4, 0, 1, 0}, {2, 1, 0, 7}, {0, 0, 7, 0}}"),
    ("WeightedAdjacencyMatrix[Graph[{1 -> 2}, EdgeWeight -> {5/2}]]", "{{0, 5/2}, {0, 0}}"),
    ("WeightedAdjacencyMatrix[Graph[{1 <-> 2, 1 <-> 2}]]", Raises()),
    ("IncidenceMatrix[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1}]]", "{{1, 0, 1}, {1, 1, 0}, {0, 1, 1}}"),
    ("IncidenceMatrix[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]", "{{-1, 0, 1}, {1, -1, 0}, {0, 1, -1}}"),
    ("IncidenceMatrix[Graph[{1 <-> 1, 1 <-> 2}]]", "{{2, 1}, {0, 1}}"), ("IncidenceMatrix[Graph[{1 -> 1}]]", Raises()),
    ("KirchhoffMatrix[" + G1 + "]", "{{2, -1, -1, 0}, {-1, 2, -1, 0}, {-1, -1, 3, -1}, {0, 0, -1, 1}}"),
    ("KirchhoffMatrix[CompleteGraph[3]]", "{{2, -1, -1}, {-1, 2, -1}, {-1, -1, 2}}"),
    ("Det[Drop[KirchhoffMatrix[PetersenGraph[]], 1, 1]]", "2000"),     # number of spanning trees of the Petersen graph
    ("KirchhoffMatrix[" + D1 + "]", Raises()),
    ("AdjacencyGraph[{{0, 1, 0}, {0, 0, 1}, {1, 1, 0}}]", "Graph[{1, 2, 3}, {1 -> 2, 2 -> 3, 3 -> 1, 3 -> 2}]"),
    ("AdjacencyGraph[{{0, 1}, {1, 0}}]", "Graph[{1, 2}, {1 <-> 2}]"),
    ("AdjacencyGraph[{a, b, c}, {{0, 1, 1}, {1, 0, 0}, {1, 0, 0}}]", "Graph[{a, b, c}, {a <-> b, a <-> c}]"),
    ("AdjacencyMatrix[AdjacencyGraph[{{0, 2, 0}, {2, 1, 1}, {0, 1, 0}}]]", "{{0, 2, 0}, {2, 1, 1}, {0, 1, 0}}"),
    ("AdjacencyGraph[{{0, 1}, {1}}]", Raises()), ("AdjacencyGraph[{{0, -1}, {-1, 0}}]", Raises()),
    ("AdjacencyList[" + G1 + ", 3]", "{1, 2, 4}"), ("AdjacencyList[" + D1 + ", 4]", "{3, 5}"),
    ("AdjacencyList[CycleGraph[4]]", "{{2, 4}, {1, 3}, {2, 4}, {1, 3}}"),
    ("WeightedAdjacencyMatrix[Graph[{1 <-> 2}, EdgeWeight -> {x}]]", Raises()),
    ("EdgeCount[Graph[{1 <-> 2}, EdgeCapacity -> {3}]]", "1"),
    ("Graph[{1 <-> 2}, EdgeCapacity -> {1, 2}]", Raises()),
    # editing
    ("VertexAdd[" + G1 + ", 5]", "Graph[{1, 2, 3, 4, 5}, {1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4}]"),
    ("VertexAdd[" + G1 + ", {4, 6}]", "Graph[{1, 2, 3, 4, 6}, {1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4}]"),
    ("VertexCount[VertexAdd[CycleGraph[3], {a, b}]]", "5"),
    ("VertexDelete[" + G1 + ", 3]", "Graph[{1, 2, 4}, {1 <-> 2}]"), ("VertexDelete[" + G1 + ", {1, 2}]", "Graph[{3, 4}, {3 <-> 4}]"),
    ("EdgeCount[VertexDelete[PetersenGraph[], 1]]", "12"),
    ("EdgeAdd[" + G1 + ", 4 <-> 5]", "Graph[{1, 2, 3, 4, 5}, {1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4, 4 <-> 5}]"),
    ("EdgeAdd[Graph[{1 -> 2}], {2 -> 3, 3 -> 1}]", "Graph[{1, 2, 3}, {1 -> 2, 2 -> 3, 3 -> 1}]"),
    ("EdgeAdd[" + W + ", 1 <-> 4]", Raises()),
    ("EdgeDelete[" + G1 + ", 1 <-> 3]", "Graph[{1, 2, 3, 4}, {1 <-> 2, 2 <-> 3, 3 <-> 4}]"),
    ("EdgeDelete[" + D1 + ", {1 -> 2, 2 -> 1}]", "Graph[{1, 2, 3, 4, 5}, {2 -> 3, 3 -> 1, 3 -> 4, 4 -> 5, 5 -> 4}]"),
    ("EdgeCount[EdgeDelete[CompleteGraph[4], 1 <-> 2]]", "5"),
    ("Subgraph[" + G1 + ", {1, 3, 4}]", "Graph[{1, 3, 4}, {3 <-> 1, 3 <-> 4}]"),
    ("Subgraph[" + G1 + ", {3 <-> 4}]", "Graph[{3, 4}, {3 <-> 4}]"),
    ("Subgraph[" + G1 + ", {9}]", Raises()), ("EdgeCount[Subgraph[PetersenGraph[], Range[5]]]", "5"),
    ("NeighborhoodGraph[" + G1 + ", 4]", "Graph[{3, 4}, {3 <-> 4}]"),
    ("NeighborhoodGraph[" + G1 + ", 4, 2]", G1.replace("Graph[", "Graph[{1, 2, 3, 4}, ").replace("{1 <-> 2", "{1 <-> 2", 1)),
    ("VertexCount[NeighborhoodGraph[PetersenGraph[], 1]]", "4"), ("NeighborhoodGraph[" + G1 + ", 4, -1]", Raises()),
    ("GraphComplement[CycleGraph[4]]", "Graph[{1, 2, 3, 4}, {1 <-> 3, 2 <-> 4}]"),
    ("GraphComplement[Graph[{1 -> 2}]]", "Graph[{1, 2}, {2 -> 1}]"),
    ("EdgeCount[GraphComplement[PetersenGraph[]]]", "30"), ("GraphComplement[Graph[{1 <-> 1}]]", Raises()),
    ("GraphUnion[Graph[{1 <-> 2}], Graph[{2 <-> 3, 1 <-> 2}]]", "Graph[{1, 2, 3}, {1 <-> 2, 2 <-> 3}]"),
    ("GraphUnion[Graph[{1 <-> 2}], Graph[{2 <-> 1}]]", "Graph[{1, 2}, {1 <-> 2}]"),
    ("GraphUnion[Graph[{1 -> 2}], Graph[{2 -> 1}]]", "Graph[{1, 2}, {1 -> 2, 2 -> 1}]"),
    ("GraphIntersection[CycleGraph[4], CompleteGraph[3]]", "Graph[{1, 2, 3}, {1 <-> 2, 2 <-> 3}]"),
    ("GraphIntersection[Graph[{1 -> 2}], Graph[{2 -> 1}]]", "Graph[{1, 2}, {}]"),
    ("EdgeCount[GraphIntersection[CompleteGraph[5], PetersenGraph[]]]", "5"),
    ("GraphDifference[CompleteGraph[4], CycleGraph[4]]", "Graph[{1, 2, 3, 4}, {1 <-> 3, 2 <-> 4}]"),
    ("GraphDifference[CycleGraph[3], CycleGraph[3]]", "Graph[{1, 2, 3}, {}]"), ("GraphDifference[CycleGraph[3]]", Raises()),
    ("GraphDisjointUnion[CycleGraph[3], PathGraph[{a, b}]]", "Graph[{1, 2, 3, 4, 5}, {1 <-> 2, 2 <-> 3, 1 <-> 3, 4 <-> 5}]"),
    ("VertexCount[GraphDisjointUnion[PetersenGraph[], PetersenGraph[]]]", "20"),
    ("ConnectedComponents[GraphDisjointUnion[CycleGraph[3], CycleGraph[4]]]", "{{4, 5, 6, 7}, {1, 2, 3}}"),
    ("UndirectedGraph[Graph[{1 -> 2, 2 -> 1, 2 -> 3}]]", "Graph[{1, 2, 3}, {1 <-> 2, 2 <-> 3}]"),
    ("UndirectedGraph[" + G1 + "]", G1.replace("Graph[{", "Graph[{1, 2, 3, 4}, {", 1)),
    ("UndirectedGraphQ[UndirectedGraph[" + D1 + "]]", True),
    ("DirectedGraph[Graph[{1 <-> 2}]]", "Graph[{1, 2}, {1 -> 2, 2 -> 1}]"),
    ("EdgeCount[DirectedGraph[PetersenGraph[]]]", "30"), ("DirectedGraph[" + G1 + ", \"Acyclic\"]", Raises()),
    ("ReverseGraph[Graph[{1 -> 2, 2 -> 3}]]", "Graph[{1, 2, 3}, {2 -> 1, 3 -> 2}]"),
    ("ReverseGraph[" + G1 + "]", G1.replace("Graph[{", "Graph[{1, 2, 3, 4}, {", 1)),
    ("AdjacencyMatrix[ReverseGraph[" + D1 + "]] == Transpose[AdjacencyMatrix[" + D1 + "]]", True),
    ("IndexGraph[Graph[{a <-> b, b <-> c}]]", "Graph[{1, 2, 3}, {1 <-> 2, 2 <-> 3}]"),
    ("IndexGraph[Graph[{a -> b}], 0]", "Graph[{0, 1}, {0 -> 1}]"), ("IndexGraph[" + G1 + ", x]", Raises()),
    ("LineGraph[PathGraph[{1, 2, 3, 4}]]", "Graph[{1, 2, 3}, {1 <-> 2, 2 <-> 3}]"),
    ("EdgeCount[LineGraph[CompleteGraph[4]]]", "12"), ("LineGraph[" + D1 + "]", Raises()),
    # paths and distances (reference definitions)
    ("FindShortestPath[" + G1 + ", 1, 4]", "{1, 3, 4}"), ("FindShortestPath[" + D1 + ", 2, 1]", "{2, 3, 1}"),
    ("FindShortestPath[" + D1 + ", 4, 1]", "{}"), ("FindShortestPath[" + G1 + ", 1, 9]", Raises()),
    ("GraphDistance[" + G1 + ", 1, 4]", "2"), ("GraphDistance[" + D1 + ", 5, 1]", "Infinity"),
    ("GraphDistance[" + G1 + ", 4]", "{2, 2, 1, 0}"),
    ("GraphDistanceMatrix[" + G1 + "]", "{{0, 1, 1, 2}, {1, 0, 1, 2}, {1, 1, 0, 1}, {2, 2, 1, 0}}"),
    ("GraphDistanceMatrix[Graph[{1, 2}, {}]]", "{{0, Infinity}, {Infinity, 0}}"),
    ("GraphDistanceMatrix[" + W + "]", "{{0, 3, 2, 9}, {3, 0, 1, 8}, {2, 1, 0, 7}, {9, 8, 7, 0}}"),
    ("GraphDistanceMatrix[Graph[{}, {}]]", Raises()),
    ("VertexEccentricity[" + G1 + ", 4]", "2"), ("VertexEccentricity[" + D1 + ", 4]", "Infinity"),
    ("VertexEccentricity[" + G1 + ", 7]", Raises()),
    ("GraphDiameter[PetersenGraph[]]", "2"), ("GraphDiameter[" + D1 + "]", "Infinity"), ("GraphDiameter[" + W + "]", "9"),
    ("GraphRadius[GridGraph[{3, 3}]]", "2"), ("GraphRadius[CycleGraph[7]]", "3"), ("GraphRadius[Graph[{1, 2}, {}]]", "Infinity"),
    ("GraphCenter[GridGraph[{3, 3}]]", "{5}"), ("GraphCenter[PathGraph[{a, b, c, d}]]", "{b, c}"), ("GraphCenter[Graph[{1, 2}, {}]]", Raises()),
    ("GraphPeriphery[PathGraph[{1, 2, 3, 4}]]", "{1, 4}"), ("GraphPeriphery[GridGraph[{3, 3}]]", "{1, 3, 7, 9}"),
    ("GraphPeriphery[Graph[{1, 2}, {}]]", Raises()),
    ("FindPath[" + G1 + ", 1, 4]", "{{1, 3, 4}}"), ("FindPath[CycleGraph[5], 1, 3, Infinity, All]", "{{1, 2, 3}, {1, 5, 4, 3}}"),
    ("FindPath[CycleGraph[5], 1, 3, {3}, All]", "{{1, 5, 4, 3}}"), ("FindPath[" + D1 + ", 4, 1]", "{}"),
    ("FindPath[CompleteGraph[4], 1, 2, 2, All]", "{{1, 2}, {1, 3, 2}, {1, 4, 2}}"),
    ("FindPath[" + G1 + ", 1, 4, x]", Raises()),
    ("FindShortestPath[{1 -> 2, 2 -> 3}, 1, 3]", "{1, 2, 3}"),
    # components and tests
    ("ConnectedComponents[Graph[{1 -> 2, 2 -> 1, 2 -> 3}]]", "{{1, 2}, {3}}"),
    ("ConnectedComponents[Graph[{1 <-> 2, 3 <-> 4, 4 <-> 5}]]", "{{3, 4, 5}, {1, 2}}"),
    ("ConnectedComponents[Graph[{1, 2, 3}, {}]]", "{{1}, {2}, {3}}"),
    ("ConnectedComponents[{1 -> 2, 2 <-> 3}]", Raises()),
    ("WeaklyConnectedComponents[Graph[{1 -> 2, 3 -> 2, 4 -> 5}]]", "{{1, 2, 3}, {4, 5}}"),
    ("WeaklyConnectedComponents[Graph[{1, 2}, {}]]", "{{1}, {2}}"),
    ("ConnectedGraphComponents[Graph[{1 <-> 2, 3 <-> 4, 4 <-> 5}]]", "{Graph[{3, 4, 5}, {3 <-> 4, 4 <-> 5}], Graph[{1, 2}, {1 <-> 2}]}"),
    ("Length[ConnectedGraphComponents[" + D1 + "]]", "2"), ("ConnectedGraphComponents[Graph[{}, {}]]", "{}"),
    ("ConnectedGraphQ[" + G1 + "]", True), ("ConnectedGraphQ[" + D1 + "]", False), ("ConnectedGraphQ[Graph[{1, 2}, {1 <-> 2}]]", True),
    ("ConnectedGraphQ[Graph[{}, {}]]", False), ("ConnectedGraphQ[5]", False),
    ("WeaklyConnectedGraphQ[" + D1 + "]", True), ("WeaklyConnectedGraphQ[Graph[{1, 2}, {}]]", False),
    ("WeaklyConnectedGraphQ[CycleGraph[4]]", True),
    ("AcyclicGraphQ[KaryTree[10]]", True), ("AcyclicGraphQ[CycleGraph[3]]", False), ("AcyclicGraphQ[Graph[{1 -> 2, 1 -> 3, 3 -> 2}]]", True),
    ("AcyclicGraphQ[" + D1 + "]", False), ("AcyclicGraphQ[Graph[{1 <-> 2, 2 <-> 1}]]", False),
    ("TreeGraphQ[KaryTree[7]]", True), ("TreeGraphQ[CycleGraph[3]]", False), ("TreeGraphQ[Graph[{1, 2, 3}, {1 <-> 2}]]", False),
    ("TreeGraphQ[StarGraph[5]]", True), ("TreeGraphQ[Graph[{1 -> 2, 1 -> 3}]]", True), ("TreeGraphQ[Graph[{2 -> 1, 3 -> 1}]]", False),
    ("BipartiteGraphQ[CycleGraph[6]]", True), ("BipartiteGraphQ[CycleGraph[5]]", False), ("BipartiteGraphQ[GridGraph[{3, 4}]]", True),
    ("BipartiteGraphQ[CompleteGraph[{2, 3}]]", True), ("BipartiteGraphQ[Graph[{1 <-> 1}]]", False),
    ("TopologicalSort[Graph[{a -> b, b -> c}]]", "{a, b, c}"),
    # cycles and tours
    ("FindCycle[" + G1 + "]", "{{1 \\[UndirectedEdge] 2, 2 \\[UndirectedEdge] 3, 3 \\[UndirectedEdge] 1}}"), ("FindCycle[PathGraph[{1, 2, 3}]]", "{}"),
    ("FindCycle[Graph[{1 -> 1, 1 -> 2}]]", "{{DirectedEdge[1, 1]}}"), ("FindCycle[CycleGraph[5], {4}]", "{}"),
    ("FindCycle[" + G1 + ", x]", Raises()),
    ("FindHamiltonianCycle[CycleGraph[5]]", "{{1 \\[UndirectedEdge] 2, 2 \\[UndirectedEdge] 3, 3 \\[UndirectedEdge] 4, 4 \\[UndirectedEdge] 5, 5 \\[UndirectedEdge] 1}}"),
    ("FindHamiltonianCycle[StarGraph[4]]", "{}"), ("FindHamiltonianCycle[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]", "{{1 \\[DirectedEdge] 2, 2 \\[DirectedEdge] 3, 3 \\[DirectedEdge] 1}}"),
    ("FindHamiltonianPath[PathGraph[{a, b, c}]]", "{a, b, c}"), ("FindHamiltonianPath[StarGraph[4]]", "{}"),
    ("Length[FindHamiltonianPath[PetersenGraph[]]]", "10"),
    ("HamiltonianGraphQ[PetersenGraph[]]", False), ("HamiltonianGraphQ[CompleteGraph[4]]", True), ("HamiltonianGraphQ[CycleGraph[9]]", True),
    ("FindEulerianCycle[CycleGraph[4]]", "{{1 \\[UndirectedEdge] 2, 2 \\[UndirectedEdge] 3, 3 \\[UndirectedEdge] 4, 4 \\[UndirectedEdge] 1}}"),
    ("FindEulerianCycle[Graph[{1 -> 2, 2 -> 1}]]", "{{DirectedEdge[1, 2], DirectedEdge[2, 1]}}"), ("FindEulerianCycle[Graph[{1, 2}, {}]]", "{}"),
    ("EulerianGraphQ[CycleGraph[5]]", True),
    # spanning trees, cuts, flows
    ("FindSpanningTree[CompleteGraph[4]]", "Graph[{1, 2, 3, 4}, {1 <-> 2, 1 <-> 3, 1 <-> 4}]"),
    ("FindSpanningTree[" + W + "]", "Graph[{1, 2, 3, 4}, {2 <-> 3, 1 <-> 3, 3 <-> 4}, EdgeWeight -> {1, 2, 7}]"),
    ("FindMinimumCut[CycleGraph[5]]", "{2, {{1, 2, 3, 4}, {5}}}"),
    ("FindMinimumCut[Graph[{1 <-> 2, 3 <-> 4}]]", "{0, {{1, 2}, {3, 4}}}"), ("FindMinimumCut[" + D1 + "]", Raises()),
    ("FindMinimumCut[Graph[{1}, {}]]", Raises()),
    ("FindMaximumFlow[Graph[{1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4}], 1, 4]", "2"), ("FindMaximumFlow[" + G1 + ", 1, 1]", Raises()),
    ("FindMaximumFlow[" + G1 + ", 1, 4, \"FlowMatrix\"]", Raises()),
    ("EdgeConnectivity[Graph[{1, 2}, {}]]", "0"),
    ("VertexCoverQ[CycleGraph[4], {1, 3}]", True), ("VertexCoverQ[CycleGraph[4], {1, 2}]", False),
    ("VertexCoverQ[" + G1 + ", {3, 1}]", True),
    ("IndependentVertexSetQ[CycleGraph[4], {1, 3}]", True), ("IndependentVertexSetQ[CycleGraph[4], {1, 2}]", False),
    ("IndependentVertexSetQ[PetersenGraph[], {1, 3, 9, 10}]", True),
    ("FindClique[" + G1 + ", 2]", Pred(lambda r: len(r.args) == 1 and len(r.args[0].args) == 2, "a 2-clique")),
    ("FindClique[Graph[{1 -> 2, 2 -> 1, 2 -> 3}]]", "{{1, 2}}"), ("FindClique[" + G1 + ", x]", Raises()),
    ("FindIndependentVertexSet[" + G1 + ", {2}, All]", "{{1, 4}, {2, 4}}"), ("FindIndependentVertexSet[Graph[{1 <-> 1}]]", Raises()),
    ("FindVertexCover[" + G1 + "]", Pred(lambda r: len(r.args) == 2, "size 2")),
    # chromatic polynomial (known closed forms)
    ("ChromaticPolynomial[CycleGraph[4], k]", "k^4 - 4 k^3 + 6 k^2 - 3 k"),
    ("ChromaticPolynomial[CompleteGraph[4], k]", "k (k - 1) (k - 2) (k - 3)"),
    ("ChromaticPolynomial[KaryTree[6], k]", "k (k - 1)^5"), ("ChromaticPolynomial[PetersenGraph[], 3]", "120"),
    ("ChromaticPolynomial[Graph[{1 <-> 1}], k]", "0"), ("ChromaticPolynomial[CycleGraph[5], 2]", "0"),
    ("ChromaticPolynomial[CompleteGraph[9], k]", Raises()),
    # centrality small cases (definitions)
    ("ClosenessCentrality[PathGraph[{1, 2, 3}]]", NearList([2 / 3, 1.0, 2 / 3])),
    ("ClosenessCentrality[Graph[{1, 2, 3}, {1 <-> 2}]]", NearList([1.0, 1.0, 0.0])),
    ("ClosenessCentrality[StarGraph[5]]", NearList([1.0] + [4 / 7] * 4)),
    ("EigenvectorCentrality[CycleGraph[5]]", NearList([0.2] * 5)),
    ("EigenvectorCentrality[StarGraph[4]]", NearList([math.sqrt(3) / (math.sqrt(3) + 3)] + [1 / (math.sqrt(3) + 3)] * 3)),
    ("EigenvectorCentrality[Graph[{1, 2, 3}, {1 <-> 2}]]", Raises()), ("EigenvectorCentrality[CycleGraph[3], \"Up\"]", Raises()),
    ("PageRankCentrality[CycleGraph[4], 0.85]", NearList([0.25] * 4)), ("PageRankCentrality[CycleGraph[4], 1.5]", Raises()),
    ("BetweennessCentrality[PathGraph[{1, 2, 3}]]", NearList([0.0, 1.0, 0.0])),
    ("BetweennessCentrality[StarGraph[5]]", NearList([6.0, 0, 0, 0, 0])),
    ("BetweennessCentrality[{{1, 2}, {2, 3}}]", "{1 -> 0, 2 -> 1, 3 -> 0}"),
    ("HITSCentrality[Graph[{1 -> 2, 1 -> 3}]]", NearList([[0, 0.5, 0.5], [1.0, 0, 0]]) if False else
     Pred(lambda r: [round(float(x), 12) for x in r.args[0].args] == [0.0, 0.5, 0.5]
          and [round(float(x), 12) for x in r.args[1].args] == [1.0, 0.0, 0.0], "{{0, 1/2, 1/2}, {1, 0, 0}}")),
    # named graphs (vertex counts, edge counts, structure from the reference pages)
    ("CompleteGraph[3]", "Graph[{1, 2, 3}, {1 <-> 2, 1 <-> 3, 2 <-> 3}]"), ("EdgeCount[CompleteGraph[10]]", "45"),
    ("CompleteGraph[{1, 2}]", "Graph[{1, 2, 3}, {1 <-> 2, 1 <-> 3}]"), ("EdgeCount[CompleteGraph[{3, 3}]]", "9"),
    ("CompleteGraph[-1]", Raises()), ("CompleteGraph[n]", "CompleteGraph[n]"),
    ("CycleGraph[4]", "Graph[{1, 2, 3, 4}, {1 <-> 2, 2 <-> 3, 3 <-> 4, 1 <-> 4}]"), ("CycleGraph[2]", Raises()),
    ("VertexDegree[CycleGraph[6]]", "{2, 2, 2, 2, 2, 2}"),
    ("PathGraph[{a, b, c}]", "Graph[{a, b, c}, {a <-> b, b <-> c}]"), ("PathGraph[{1}]", "Graph[{1}, {}]"), ("PathGraph[3]", Raises()),
    ("StarGraph[4]", "Graph[{1, 2, 3, 4}, {1 <-> 2, 1 <-> 3, 1 <-> 4}]"), ("VertexDegree[StarGraph[6], 1]", "5"), ("StarGraph[0]", Raises()),
    ("WheelGraph[5]", "Graph[{1, 2, 3, 4, 5}, {1 <-> 2, 1 <-> 3, 1 <-> 4, 1 <-> 5, 2 <-> 3, 3 <-> 4, 4 <-> 5, 2 <-> 5}]"),
    ("EdgeCount[WheelGraph[10]]", "18"), ("WheelGraph[3]", Raises()),
    ("GridGraph[{2, 3}]", "Graph[{1, 2, 3, 4, 5, 6}, {1 <-> 2, 1 <-> 3, 2 <-> 4, 3 <-> 4, 3 <-> 5, 4 <-> 6, 5 <-> 6}]"),
    ("EdgeCount[GridGraph[{4, 5}]]", "31"), ("VertexCount[GridGraph[{2, 3, 4}]]", "24"), ("GridGraph[{}]", Raises()),
    ("AdjacencyList[GridGraph[{4, 4}], 6]", "{2, 5, 7, 10}"),
    ("HypercubeGraph[2]", "Graph[{1, 2, 3, 4}, {1 <-> 2, 1 <-> 3, 2 <-> 4, 3 <-> 4}]"),
    ("EdgeCount[HypercubeGraph[4]]", "32"), ("VertexDegree[HypercubeGraph[3]]", "{3, 3, 3, 3, 3, 3, 3, 3}"),
    ("PetersenGraph[]", "Graph[Range[10], {1 <-> 2, 1 <-> 5, 1 <-> 6, 2 <-> 3, 2 <-> 7, 3 <-> 4, 3 <-> 8, 4 <-> 5, 4 <-> 9, 5 <-> 10, 6 <-> 8, 6 <-> 9, 7 <-> 9, 7 <-> 10, 8 <-> 10}]"),
    ("IsomorphicGraphQ[PetersenGraph[4, 1], HypercubeGraph[3]]", True), ("PetersenGraph[2, 1]", Raises()),
    ("KaryTree[5]", "Graph[{1, 2, 3, 4, 5}, {1 <-> 2, 1 <-> 3, 2 <-> 4, 2 <-> 5}]"),
    ("KaryTree[5, 3]", "Graph[{1, 2, 3, 4, 5}, {1 <-> 2, 1 <-> 3, 1 <-> 4, 2 <-> 5}]"), ("KaryTree[-2]", Raises()),
    ("VertexCount[CompleteKaryTree[3]]", "7"), ("VertexCount[CompleteKaryTree[3, 3]]", "13"),
    ("CompleteKaryTree[2, 3]", "Graph[{1, 2, 3, 4}, {1 <-> 2, 1 <-> 3, 1 <-> 4}]"),
    ("CirculantGraph[6, {1, 2}]", "Graph[Range[6], {1 <-> 2, 1 <-> 3, 1 <-> 5, 1 <-> 6, 2 <-> 3, 2 <-> 4, 2 <-> 6, 3 <-> 4, 3 <-> 5, 4 <-> 5, 4 <-> 6, 5 <-> 6}]"),
    ("IsomorphicGraphQ[CirculantGraph[5, {1}], CycleGraph[5]]", True), ("CirculantGraph[0, {1}]", Raises()),
    ("EdgeCount[TuranGraph[5, 2]]", "6"), ("BipartiteGraphQ[TuranGraph[5, 2]]", True), ("EdgeCount[TuranGraph[7, 3]]", "16"),
    ("SeedRandom[5]; RandomGraph[{6, 7}] === (SeedRandom[5]; RandomGraph[{6, 7}])", True),
    ("EdgeCount[RandomGraph[{10, 20}]]", "20"), ("SimpleGraphQ[RandomGraph[{8, 12}]]", True),
    ("Length[RandomGraph[{5, 3}, 4]]", "4"), ("RandomGraph[{4, 7}]", Raises()), ("RandomGraph[{5}]", Raises()),  # graph distributions are sampled since W24
    # traversal (Reap/Sow with scan events)
    ("Reap[BreadthFirstScan[GridGraph[{2, 2}], 1, {\"PrevisitVertex\" -> Sow}]][[2, 1]]", "{1, 2, 3, 4}"),
    ("Reap[BreadthFirstScan[" + G1 + ", 4, {\"DiscoverVertex\" -> (Sow[{#1, #2, #3}] &)}]][[2, 1]]",
     "{{4, 4, 0}, {3, 4, 1}, {2, 3, 2}, {1, 3, 2}}"),
    ("BreadthFirstScan[" + G1 + ", 1, {\"Bogus\" -> f}]", Raises()),
    ("Reap[DepthFirstScan[PathGraph[{1, 2, 3}], 1, {\"PrevisitVertex\" -> Sow, \"PostvisitVertex\" -> (Sow[-#] &)}]][[2, 1]]",
     "{1, 2, 3, -3, -2, -1}"),
    ("Reap[DepthFirstScan[StarGraph[4], 2, {\"PrevisitVertex\" -> Sow}]][[2, 1]]", "{2, 1, 3, 4}"),
    ("DepthFirstScan[" + G1 + ", 1, {}] === " + G1, True),
    ("VertexOutComponent[" + D1 + ", 4]", "{4, 5}"), ("VertexOutComponent[" + D1 + ", 1]", "{1, 2, 3, 4, 5}"),
    ("VertexOutComponent[" + D1 + ", 1, 1]", "{1, 2}"),
    ("VertexInComponent[" + D1 + ", 4]", "{4, 3, 5, 2, 1}"), ("VertexInComponent[" + D1 + ", 1]", "{1, 3, 2}"),
    ("VertexInComponent[" + D1 + ", 9]", Raises()),
    # clustering (definitions)
    ("LocalClusteringCoefficient[CompleteGraph[4]]", "{1, 1, 1, 1}"), ("LocalClusteringCoefficient[" + D1 + "]", Raises()),
    ("MeanClusteringCoefficient[" + G1 + "]", "7/12"), ("MeanClusteringCoefficient[StarGraph[4]]", "0"),
    ("GlobalClusteringCoefficient[" + G1 + "]", "3/5"), ("GlobalClusteringCoefficient[CompleteGraph[4]]", "1"),
    ("GraphDensity[CycleGraph[4]]", "2/3"), ("GraphDensity[Graph[{1}, {}]]", Raises()),
    # option heads
    ("EdgeWeight", "EdgeWeight"), ("EdgeCapacity", "EdgeCapacity"), ("VertexWeight", "VertexWeight"),
]

# ─────────────────────────── statistics: Wolfram definitions and documentation ───────────────────────────

STATS = [
    # distribution objects: inert, parameters validated
    ("GammaDistribution[2, 3]", "GammaDistribution[2, 3]"), ("GammaDistribution[-1, 2]", Raises()),
    ("GammaDistribution[a, b]", "GammaDistribution[a, b]"),
    ("BetaDistribution[2, 3]", "BetaDistribution[2, 3]"), ("BetaDistribution[0, 3]", Raises()),
    ("BetaDistribution[a, 1]", "BetaDistribution[a, 1]"),
    ("ChiSquareDistribution[4]", "ChiSquareDistribution[4]"), ("ChiSquareDistribution[-4]", Raises()),
    ("FRatioDistribution[2, 5]", "FRatioDistribution[2, 5]"), ("FRatioDistribution[2, 0]", Raises()),
    ("LogNormalDistribution[0, 1]", "LogNormalDistribution[0, 1]"), ("LogNormalDistribution[0, -1]", Raises()),
    ("CauchyDistribution[]", "CauchyDistribution[0, 1]"), ("CauchyDistribution[1, 0]", Raises()),
    ("WeibullDistribution[2, 1]", "WeibullDistribution[2, 1]"), ("WeibullDistribution[0, 1]", Raises()),
    ("LaplaceDistribution[0, 1]", "LaplaceDistribution[0, 1]"), ("LaplaceDistribution[0, -2]", Raises()),
    ("LogisticDistribution[]", "LogisticDistribution[0, 1]"), ("LogisticDistribution[0, 0]", Raises()),
    ("RayleighDistribution[2]", "RayleighDistribution[2]"), ("RayleighDistribution[0]", Raises()),
    ("ParetoDistribution[1, 3]", "ParetoDistribution[1, 3]"), ("ParetoDistribution[1, 2, 3, 4, 5]", Raises()),
    ("ParetoDistribution[-1, 3]", Raises()),
    ("GumbelDistribution[]", "GumbelDistribution[0, 1]"), ("GumbelDistribution[0, -1]", Raises()),
    ("ExtremeValueDistribution[]", "ExtremeValueDistribution[0, 1]"), ("ExtremeValueDistribution[0, 0]", Raises()),
    ("TriangularDistribution[{0, 2}, 1]", "TriangularDistribution[{0, 2}, 1]"),
    ("TriangularDistribution[{0, 2}, 3]", Raises()), ("TriangularDistribution[]", "TriangularDistribution[{0, 1}]"),
    ("ErlangDistribution[3, 2]", "ErlangDistribution[3, 2]"), ("ErlangDistribution[3/2, 2]", Raises()),
    ("ChiDistribution[3]", "ChiDistribution[3]"), ("ChiDistribution[0]", Raises()),
    ("BernoulliDistribution[1/3]", "BernoulliDistribution[1/3]"), ("BernoulliDistribution[2]", Raises()),
    ("GeometricDistribution[1/3]", "GeometricDistribution[1/3]"), ("GeometricDistribution[0]", Raises()),
    ("NegativeBinomialDistribution[2, 1/2]", "NegativeBinomialDistribution[2, 1/2]"),
    ("NegativeBinomialDistribution[2, 3/2]", Raises()),
    ("HypergeometricDistribution[3, 5, 10]", "HypergeometricDistribution[3, 5, 10]"),
    ("HypergeometricDistribution[11, 5, 10]", Raises()),
    ("DiscreteUniformDistribution[{1, 6}]", "DiscreteUniformDistribution[{1, 6}]"),
    ("DiscreteUniformDistribution[{6, 1}]", Raises()),
    ("MultinormalDistribution[{0, 0}, {{1, 0}, {0, 1}}]", "MultinormalDistribution[{0, 0}, {{1, 0}, {0, 1}}]"),
    ("MultinormalDistribution[{0, 0}, {{1, 2}, {2, 1}}]", Raises()),
    ("MultinomialDistribution[3, {1/2, 1/4, 1/4}]", "MultinomialDistribution[3, {1/2, 1/4, 1/4}]"),
    ("MultinomialDistribution[3, {1/2, 1/4}]", Raises()),
    ("EmpiricalDistribution[{1, 2, 2, 3}]", "EmpiricalDistribution[{1, 2, 2, 3}]"),
    ("EmpiricalDistribution[{1, 2} -> {3}]", Raises()),
    # PDF / CDF (closed forms from the Wolfram reference pages)
    ("PDF[GammaDistribution[2, 3], 1]", "1/(9 E^(1/3))"), ("PDF[GammaDistribution[2, 3], -1]", "0"),
    ("PDF[GammaDistribution[2, 3], x]", "Piecewise[{{x E^(-x/3)/9, x > 0}}, 0]"),
    ("CDF[GammaDistribution[2, 3], 3]", "1 - 2/E"), ("CDF[GammaDistribution[2, 3], -3]", "0"),
    ("PDF[BetaDistribution[2, 3], 1/2]", "3/2"), ("PDF[BetaDistribution[2, 3], 2]", "0"),
    ("CDF[BetaDistribution[2, 3], 1/2]", "11/16"), ("CDF[BetaDistribution[2, 3], 2]", "1"),
    ("PDF[ChiSquareDistribution[2], x]", "Piecewise[{{E^(-x/2)/2, x > 0}}, 0]"),
    ("CDF[ChiSquareDistribution[2], 2]", "1 - 1/E"), ("Mean[ChiSquareDistribution[n]]", "n"),
    ("Variance[ChiSquareDistribution[n]]", "2 n"),
    ("PDF[FRatioDistribution[2, 2], 1]", "1/4"), ("CDF[FRatioDistribution[2, 2], 1]", "1/2"),
    ("Mean[FRatioDistribution[n, 5]]", "5/3"), ("Mean[FRatioDistribution[3, 2]]", "Indeterminate"),
    ("Variance[FRatioDistribution[2, 6]]", "27/4"), ("Variance[FRatioDistribution[2, 4]]", "Indeterminate"),
    ("CDF[LogNormalDistribution[0, 1], 1]", "1/2"), ("Median[LogNormalDistribution[m, s]]", "E^m"),
    ("Mean[LogNormalDistribution[m, s]]", "E^(m + s^2/2)"), ("PDF[LogNormalDistribution[0, 1], 0]", "0"),
    ("PDF[CauchyDistribution[0, 1], 0]", "1/Pi"), ("CDF[CauchyDistribution[0, 1], 1]", "3/4"),
    ("Mean[CauchyDistribution[0, 1]]", "Indeterminate"), ("Variance[CauchyDistribution[a, b]]", "Indeterminate"),
    ("InverseCDF[CauchyDistribution[a, b], 3/4]", "a + b"),
    ("CDF[WeibullDistribution[2, 1], x]", "Piecewise[{{0, x <= 0}}, 1 - E^(-x^2)]"),
    ("InverseCDF[WeibullDistribution[2, 1], 1/2]", "Sqrt[Log[2]]"), ("Mean[WeibullDistribution[a, b]]", "b Gamma[1 + 1/a]"),
    ("PDF[LaplaceDistribution[0, 1], x]", "E^(-Abs[x])/2"), ("CDF[LaplaceDistribution[0, 1], -1]", "1/(2 E)"),
    ("Variance[LaplaceDistribution[m, b]]", "2 b^2"), ("InverseCDF[LaplaceDistribution[0, 1], 3/4]", "Log[2]"),
    ("CDF[LogisticDistribution[0, 1], 0]", "1/2"), ("Variance[LogisticDistribution[m, b]]", "Pi^2 b^2/3"),
    ("InverseCDF[LogisticDistribution[0, 1], 3/4]", "Log[3]"),
    ("Mean[RayleighDistribution[s]]", "s Sqrt[Pi/2]"), ("Variance[RayleighDistribution[s]]", "(4 - Pi) s^2/2"),
    ("CDF[RayleighDistribution[1], 1]", "1 - E^(-1/2)"),
    ("PDF[ParetoDistribution[1, 3], 2]", "3/16"), ("PDF[ParetoDistribution[1, 3], 1/2]", "0"),
    ("CDF[ParetoDistribution[1, 3], 2]", "7/8"), ("Mean[ParetoDistribution[1, 3]]", "3/2"),
    ("Mean[ParetoDistribution[1, 1]]", "Indeterminate"), ("Variance[ParetoDistribution[1, 3]]", "3/4"),
    ("InverseCDF[ParetoDistribution[1, 1], 1/2]", "2"),
    ("CDF[GumbelDistribution[0, 1], 0]", "1 - 1/E"), ("CDF[ExtremeValueDistribution[0, 1], 0]", "1/E"),
    ("Mean[GumbelDistribution[a, b]]", "a - EulerGamma b"), ("Mean[ExtremeValueDistribution[a, b]]", "a + EulerGamma b"),
    ("Variance[GumbelDistribution[a, b]]", "Pi^2 b^2/6"), ("Variance[ExtremeValueDistribution[a, b]]", "Pi^2 b^2/6"),
    ("Kurtosis[ExtremeValueDistribution[0, 1]]", "27/5"),
    ("PDF[TriangularDistribution[{0, 2}, 1], 1/2]", "1/2"), ("CDF[TriangularDistribution[{0, 2}, 1], 1]", "1/2"),
    ("Mean[TriangularDistribution[{0, 2}, 1/2]]", "5/6"), ("Variance[TriangularDistribution[{0, 1}, 1/2]]", "1/24"),
    ("InverseCDF[TriangularDistribution[{0, 2}, 1], 1/8]", "1/2"),
    ("Mean[ErlangDistribution[k, l]]", "k/l"), ("Variance[ErlangDistribution[3, 2]]", "3/4"),
    ("CDF[ErlangDistribution[1, 2], 1]", "1 - E^(-2)"),
    ("Mean[ChiDistribution[2]]", "Sqrt[Pi/2]"), ("Variance[ChiDistribution[3]]", "3 - 8/Pi"),
    ("CDF[ChiDistribution[2], 1]", "1 - E^(-1/2)"),
    # discrete distributions
    ("PDF[BernoulliDistribution[p], 1]", "p"), ("PDF[BernoulliDistribution[p], 0]", "1 - p"),
    ("PDF[BernoulliDistribution[p], 2]", "0"), ("CDF[BernoulliDistribution[1/3], 1/2]", "2/3"),
    ("Variance[BernoulliDistribution[p]]", "p (1 - p)"),
    ("PDF[GeometricDistribution[1/3], 2]", "4/27"), ("CDF[GeometricDistribution[1/3], 2]", "19/27"),
    ("PDF[GeometricDistribution[1/3], 5/2]", "0"), ("CDF[GeometricDistribution[1/3], 5/2]", "19/27"),
    ("Mean[GeometricDistribution[p]]", "(1 - p)/p"), ("Variance[GeometricDistribution[p]]", "(1 - p)/p^2"),
    ("PDF[NegativeBinomialDistribution[2, 1/2], 2]", "3/16"), ("Mean[NegativeBinomialDistribution[n, p]]", "n (1 - p)/p"),
    ("Variance[NegativeBinomialDistribution[n, p]]", "n (1 - p)/p^2"),
    ("CDF[NegativeBinomialDistribution[2, 1/2], 1]", "1/2"),
    ("PDF[HypergeometricDistribution[3, 5, 10], 2]", "5/12"), ("PDF[HypergeometricDistribution[3, 5, 10], 4]", "0"),
    ("Mean[HypergeometricDistribution[n, s, t]]", "n s/t"), ("Variance[HypergeometricDistribution[3, 5, 10]]", "7/12"),
    ("CDF[HypergeometricDistribution[3, 5, 10], 0]", "1/12"),
    ("PDF[DiscreteUniformDistribution[{1, 6}], 3]", "1/6"), ("PDF[DiscreteUniformDistribution[{1, 6}], 7]", "0"),
    ("CDF[DiscreteUniformDistribution[{1, 6}], 2]", "1/3"), ("Mean[DiscreteUniformDistribution[{a, b}]]", "(a + b)/2"),
    ("Variance[DiscreteUniformDistribution[{1, 6}]]", "35/12"), ("InverseCDF[DiscreteUniformDistribution[{1, 6}], 1/2]", "3"),
    ("InverseCDF[GeometricDistribution[1/2], 3/4]", "1"), ("InverseCDF[BinomialDistribution[4, 1/2], 1/2]", "2"),
    # empirical distribution
    ("PDF[EmpiricalDistribution[{1, 2, 2, 3}], 2]", "1/2"), ("PDF[EmpiricalDistribution[{1, 2, 2, 3}], 5/2]", "0"),
    ("CDF[EmpiricalDistribution[{1, 2, 2, 3}], 5/2]", "3/4"), ("Mean[EmpiricalDistribution[{1, 2, 2, 3}]]", "2"),
    ("Variance[EmpiricalDistribution[{1, 2, 2, 3}]]", "1/2"),
    ("Mean[EmpiricalDistribution[{1, 3} -> {0, 1}]]", "3/4"),
    ("InverseCDF[EmpiricalDistribution[{1, 2, 2, 3}], 1/2]", "2"),
    # moments of distributions (Indeterminate when they do not exist)
    ("Skewness[ExponentialDistribution[2]]", "2"), ("Skewness[NormalDistribution[m, s]]", "0"),
    ("Skewness[PoissonDistribution[m]]", "1/Sqrt[m]"), ("Skewness[GammaDistribution[k, t]]", "2/Sqrt[k]"),
    ("Skewness[CauchyDistribution[0, 1]]", "Indeterminate"), ("Skewness[GumbelDistribution[0, 1]]", "-12 Sqrt[6] Zeta[3]/Pi^3"),
    ("Kurtosis[NormalDistribution[m, s]]", "3"), ("Kurtosis[PoissonDistribution[3]]", "10/3"),
    ("Kurtosis[ExponentialDistribution[l]]", "9"), ("Kurtosis[UniformDistribution[{0, 1}]]", "9/5"),
    ("Kurtosis[StudentTDistribution[5]]", "9"), ("Kurtosis[StudentTDistribution[3]]", "Indeterminate"),
    ("Kurtosis[LaplaceDistribution[0, 1]]", "6"), ("Kurtosis[LogisticDistribution[0, 1]]", "21/5"),
    ("Median[PoissonDistribution[3]]", "3"), ("Median[ExponentialDistribution[l]]", "Log[2]/l"),
    ("Median[NormalDistribution[m, s]]", "m"), ("Median[BinomialDistribution[10, 1/2]]", "5"),
    ("Moment[GammaDistribution[2, 3], 3]", "648"), ("Moment[NormalDistribution[0, s], 4]", "3 s^4"),
    ("Moment[PoissonDistribution[m], 2]", "m + m^2"), ("Moment[ExponentialDistribution[l], r]", "Moment[ExponentialDistribution[l], r]"),
    ("Moment[ParetoDistribution[1, 3], 3]", "Indeterminate"), ("Moment[ExponentialDistribution[1], 0]", "1"),
    ("Moment[UniformDistribution[{0, 1}], 3]", "1/4"), ("Moment[BinomialDistribution[2, p], 2]", "2 p + 2 p^2"),
    ("CentralMoment[NormalDistribution[m, s], 4]", "3 s^4"), ("CentralMoment[ExponentialDistribution[1], 3]", "2"),
    ("CentralMoment[PoissonDistribution[m], 3]", "m"), ("CentralMoment[CauchyDistribution[0, 1], 2]", "Indeterminate"),
    ("Cumulant[PoissonDistribution[m], 4]", "m"), ("Cumulant[NormalDistribution[m, s], 3]", "0"),
    ("Cumulant[ExponentialDistribution[1], 3]", "2"), ("Cumulant[NormalDistribution[m, s], 2]", "s^2"),
    ("Cumulant[ExponentialDistribution[1], 0]", Raises()),
    # survival, hazard, generating functions
    ("SurvivalFunction[ExponentialDistribution[l], 2]", "E^(-2 l)"),
    ("SurvivalFunction[ExponentialDistribution[l], x]", "Piecewise[{{1, x <= 0}}, E^(-l x)]"),
    ("SurvivalFunction[PoissonDistribution[2], 1]", "1 - 3/E^2"), ("SurvivalFunction[ParetoDistribution[1, 3], 2]", "1/8"),
    ("HazardFunction[ExponentialDistribution[2], 1]", "2"), ("HazardFunction[WeibullDistribution[2, 1], x]",
                                                            "Piecewise[{{2 x, x > 0}}, 0]"),
    ("HazardFunction[ExponentialDistribution[2], -1]", "0"), ("HazardFunction[GeometricDistribution[1/2], 3]", "1/2"),
    ("InverseSurvivalFunction[ExponentialDistribution[1], 1/2]", "Log[2]"),
    ("InverseSurvivalFunction[UniformDistribution[{0, 4}], 1/4]", "3"),
    ("InverseSurvivalFunction[ParetoDistribution[1, 1], 1/4]", "4"),
    ("CharacteristicFunction[NormalDistribution[m, s], t]", "E^(I m t - s^2 t^2/2)"),
    ("CharacteristicFunction[CauchyDistribution[a, b], t]", "E^(I a t - b Abs[t])"),
    ("CharacteristicFunction[BernoulliDistribution[p], t]", "1 - p + p E^(I t)"),
    ("MomentGeneratingFunction[PoissonDistribution[m], t]", "E^(m (E^t - 1))"),
    ("MomentGeneratingFunction[ExponentialDistribution[l], t]", "l/(l - t)"),
    ("MomentGeneratingFunction[GammaDistribution[a, b], t]", "(1 - b t)^(-a)"),
    ("MomentGeneratingFunction[CauchyDistribution[0, 1], t]", "MomentGeneratingFunction[CauchyDistribution[0, 1], t]"),
    ("CentralMomentGeneratingFunction[NormalDistribution[m, s], t]", "E^(s^2 t^2/2)"),
    ("CentralMomentGeneratingFunction[PoissonDistribution[m], t]", "E^(m (E^t - 1) - m t)"),
    ("CentralMomentGeneratingFunction[ExponentialDistribution[1], t]", "E^(-t)/(1 - t)"),
    ("CumulantGeneratingFunction[PoissonDistribution[m], t]", "m (E^t - 1)"),
    ("CumulantGeneratingFunction[NormalDistribution[m, s], t]", "m t + s^2 t^2/2"),
    ("CumulantGeneratingFunction[ExponentialDistribution[1], 1/2]", "Log[2]"),
    # Expectation / Probability (x \[Distributed] dist)
    ("Distributed[x, NormalDistribution[0, 1]]", "x \\[Distributed] NormalDistribution[0, 1]"),
    ("(x \\[Distributed] PoissonDistribution[2])[[2]]", "PoissonDistribution[2]"),
    ("Expectation[x^2, x \\[Distributed] NormalDistribution[0, s]]", "s^2"),
    ("Expectation[x^2 + 1, x \\[Distributed] PoissonDistribution[m]]", "m^2 + m + 1"),
    ("Expectation[E^x, x \\[Distributed] UniformDistribution[{0, 1}]]", "E - 1"),
    ("Expectation[x y, {x \\[Distributed] BernoulliDistribution[1/2], y \\[Distributed] BernoulliDistribution[1/3]}]", "1/6"),
    ("Expectation[x, x \\[Distributed] Foo[1]]", Raises()), ("Expectation[x, y]", Raises()),
    ("Probability[x > 1, x \\[Distributed] ExponentialDistribution[2]]", "E^(-2)"),
    ("Probability[1 <= x <= 3, x \\[Distributed] PoissonDistribution[2]]", "16/(3 E^2)"),
    ("Probability[1 < x < 3, x \\[Distributed] PoissonDistribution[2]]", "2/E^2"),
    ("Probability[x < 2 || x > 5, x \\[Distributed] UniformDistribution[{0, 10}]]", "7/10"),
    ("Probability[x != 2, x \\[Distributed] BinomialDistribution[4, 1/2]]", "5/8"),
    ("Probability[x == 2, x \\[Distributed] NormalDistribution[0, 1]]", "0"),
    ("Probability[x^2 < 4, x \\[Distributed] DiscreteUniformDistribution[{-5, 5}]]", "3/11"),
    ("Probability[x >= 2, x \\[Distributed] EmpiricalDistribution[{1, 2, 2, 3}]]", "3/4"),
    ("Probability[Abs[x] < 1, x \\[Distributed] NormalDistribution[0, 1]]", "1 - Erfc[1/Sqrt[2]]"),
    ("Probability[x > 2.5, x \\[Distributed] GeometricDistribution[1/2]]", Near(0.125)),
    ("NProbability[x > 1.96, x \\[Distributed] NormalDistribution[0, 1]]", Near(0.024997895148220435, 1e-9)),
    ("NProbability[x <= 1, x \\[Distributed] ExponentialDistribution[1]]", Near(1 - math.exp(-1), 1e-9)),
    ("NProbability[x > 2, x \\[Distributed] PoissonDistribution[1]]", Near(1 - 2.5 * math.exp(-1), 1e-9)),
    ("NExpectation[Sqrt[x], x \\[Distributed] ExponentialDistribution[1]]", Near(math.sqrt(math.pi) / 2, 1e-9)),
    ("NExpectation[x^2, x \\[Distributed] NormalDistribution[1, 2]]", Near(5, 1e-9)),
    ("NExpectation[1/(1 + x), x \\[Distributed] PoissonDistribution[1]]", Near(1 - math.exp(-1), 1e-9)),
    # TransformedDistribution
    ("Mean[TransformedDistribution[2 x + 1, x \\[Distributed] ExponentialDistribution[1]]]", "3"),
    ("Variance[TransformedDistribution[2 x + 1, x \\[Distributed] ExponentialDistribution[1]]]", "4"),
    ("StandardDeviation[TransformedDistribution[3 x, x \\[Distributed] NormalDistribution[0, 1]]]", "3"),
    ("Moment[TransformedDistribution[x^2, x \\[Distributed] NormalDistribution[0, 1]], 2]", "3"),
    ("TransformedDistribution[x, y]", Raises()),
    # multivariate distributions
    ("PDF[MultinormalDistribution[{0, 0}, {{1, 0}, {0, 1}}], {0, 0}]", "1/(2 Pi)"),
    ("PDF[MultinormalDistribution[{{2, 1}, {1, 2}}], {1, 1}]", "1/(2 Sqrt[3] E^(1/3) Pi)"),
    ("Mean[MultinormalDistribution[{1, 2}, {{2, 1}, {1, 2}}]]", "{1, 2}"),
    ("Variance[MultinormalDistribution[{1, 2}, {{2, 1}, {1, 3}}]]", "{2, 3}"),
    ("Covariance[MultinormalDistribution[{1, 2}, {{2, 1}, {1, 3}}]]", "{{2, 1}, {1, 3}}"),
    ("CDF[MultinormalDistribution[{0}, {{1}}], {0}]", "1/2"),
    ("PDF[MultinomialDistribution[3, {1/2, 1/4, 1/4}], {1, 1, 1}]", "3/16"),
    ("PDF[MultinomialDistribution[3, {1/2, 1/4, 1/4}], {1, 1, 2}]", "0"),
    ("Mean[MultinomialDistribution[n, {p, q}]]", "{n p, n q}"),
    ("Covariance[MultinomialDistribution[4, {1/2, 1/2}]]", "{{1, -1}, {-1, 1}}"),
    ("CDF[MultinomialDistribution[2, {1/2, 1/2}], {1, 1}]", "1/2"),
    ("Length[RandomVariate[MultinomialDistribution[5, {1/2, 1/2}], 3]]", "3"),
    ("Total[RandomVariate[MultinomialDistribution[5, {1/2, 1/3, 1/6}]]]", "5"),
    ("Dimensions[RandomVariate[MultinormalDistribution[{0, 0}, {{1, 0}, {0, 1}}], 4]]", "{4, 2}"),
    # descriptive statistics on data
    ("Quartiles[{1, 2, 3, 4, 5, 6, 7}]", "{9/4, 4, 23/4}"), ("Quartiles[{1, 3, 4, 2, 5, 6}]", "{2, 7/2, 5}"),
    ("Quartiles[{1., 2., 3., 4.}]", NearList([1.5, 2.5, 3.5])), ("Quartiles[ExponentialDistribution[1]]",
                                                                  "{Log[4/3], Log[2], Log[4]}"),
    ("InterquartileRange[{1, 3, 4, 7, 9, 10}]", "6"), ("InterquartileRange[{1, 2, 3, 4, 5, 6, 7}]", "7/2"),
    ("N[InterquartileRange[NormalDistribution[0, 1]]]", Near(1.3489795003921634)),
    ("InterquartileRange[UniformDistribution[{0, 4}]]", "2"),
    ("QuartileDeviation[{1, 2, 3, 4, 5, 6, 7}]", "7/4"), ("QuartileDeviation[UniformDistribution[{0, 4}]]", "1"),
    ("QuartileDeviation[{1, 3, 4, 7, 9, 10}]", "3"),
    ("QuartileSkewness[{1, 2, 3, 4, 5, 6, 7}]", "0"), ("QuartileSkewness[{1, 2, 3, 4, 10, 20, 30}]", "47/61"),
    ("QuartileSkewness[{1, 1, 1, 1}]", Raises()),
    ("MeanDeviation[{1, 2, 3, 4, 10}]", "12/5"), ("MeanDeviation[{}]", Raises()),
    ("MeanDeviation[{{1, 2}, {3, 6}}]", "{1, 2}"),
    ("MedianDeviation[{1, 2, 3, 4, 10}]", "1"), ("MedianDeviation[{1, 2, 3, 4}]", "1"),
    ("MedianDeviation[{{1, 2}, {3, 6}, {5, 7}}]", "{2, 1}"),
    ("RootMeanSquare[{1, 2, 3}]", "Sqrt[14/3]"), ("RootMeanSquare[{3, -4}]", "5/Sqrt[2]"),
    ("RootMeanSquare[{1., 1.}]", Near(1.0)),
    ("ContraharmonicMean[{1, 2, 3}]", "7/3"), ("ContraharmonicMean[{1, -1}]", Raises()),
    ("ContraharmonicMean[{2., 2.}]", Near(2.0)),
    ("WinsorizedMean[{1, 2, 3, 4, 5, 6, 7, 8, 9, 100}, 0.1]", Near(5.5)),
    ("WinsorizedMean[{1, 2, 3, 4, 5, 6, 7, 8, 9, 100}]", "11/2"), ("WinsorizedMean[{1, 2, 3}, 1/2]", Raises()),
    ("WinsorizedVariance[{1, 2, 3, 4, 100}, 1/5]", "1"), ("WinsorizedVariance[{1, 2, 3, 4, 5}, 0]", "5/2"),
    ("WinsorizedVariance[{1}]", Raises()),
    ("TrimmedVariance[{1, 2, 3, 4, 100}, 1/5]", "1"), ("TrimmedVariance[{1, 2, 3, 4, 5}, 0]", "5/2"),
    ("TrimmedVariance[{1, 2, 3}, 2/5]", Raises()),
    ("Standardize[{1, 2, 3}]", "{-1, 0, 1}"), ("Standardize[{1, 1, 1}]", Raises()),
    ("Standardize[{2, 4, 6}, Mean, 1 &]", "{-2, 0, 2}"), ("Standardize[{1., 3.}]", NearList([-1 / math.sqrt(2), 1 / math.sqrt(2)])),
    ("Moment[{1, 2, 3, 4}, 2]", "15/2"), ("Moment[{1, 2, 3}, 0]", "1"), ("Moment[{1., 2.}, 3]", Near(4.5)),
    ("CentralMoment[{1, 2, 3, 4}, 2]", "5/4"), ("CentralMoment[{1, 2, 3, 10}, 3]", "45"),
    ("CentralMoment[{1, 2, 3}, 1]", "0"),
    ("Cumulant[{1, 2, 3, 4}, 1]", "5/2"), ("Cumulant[{1, 2, 3, 4}, 2]", "5/4"), ("Cumulant[{1, 2, 3, 10}, 3]", "45"),
    ("Cumulant[{1, 2, 3, 4}, 4]", "-17/8"),
    ("SpearmanRho[{1, 2, 3, 4}, {1, 3, 2, 4}]", "4/5"), ("SpearmanRho[{1, 2, 3}, {3, 2, 1}]", "-1"),
    ("SpearmanRho[{1, 2}, {1, 2, 3}]", Raises()), ("SpearmanRho[{1, 1, 1}, {1, 2, 3}]", Raises()),
    ("SpearmanRho[{{1, 1}, {2, 3}, {3, 2}}]", "{{1, 1/2}, {1/2, 1}}"),
    ("KendallTau[{1, 2, 3, 4}, {1, 3, 2, 4}]", "2/3"), ("KendallTau[{1, 2, 3}, {3, 2, 1}]", "-1"),
    ("KendallTau[{1, 2}, {1}]", Raises()), ("KendallTau[{{1, 1}, {2, 3}, {3, 2}}]", "{{1, 1/3}, {1/3, 1}}"),
    ("Mean[{{1, 2}, {3, 4}}]", "{2, 3}"), ("Variance[{{1, 2}, {3, 6}}]", "{2, 8}"),
    ("Median[{{1, 5}, {2, 6}, {9, 7}}]", "{2, 6}"), ("StandardDeviation[{{1, 2}, {3, 6}}]", "{Sqrt[2], 2 Sqrt[2]}"),
    ("Covariance[{{1, 2}, {2, 3}, {3, 5}}]", "{{1, 3/2}, {3/2, 7/3}}"),
    ("Correlation[{{1, 2}, {2, 4}, {3, 6}}]", "{{1, 1}, {1, 1}}"),
    # random numbers (seeded, reproducible; ranges and shapes)
    ("SeedRandom[7]; seq7 = RandomInteger[{1, 6}, 20]; SeedRandom[7]; seq7 === RandomInteger[{1, 6}, 20]", True),
    ("And @@ Map[1 <= # <= 6 &, RandomInteger[{1, 6}, 50]]", True), ("MemberQ[{0, 1}, RandomInteger[]]", True),
    ("Dimensions[RandomInteger[5, {2, 3}]]", "{2, 3}"), ("RandomInteger[{3, 1}]", Raises()),
    ("And @@ Map[2 <= # < 5 &, RandomReal[{2, 5}, 30]]", True), ("Dimensions[RandomReal[1, {4, 2}]]", "{4, 2}"),
    ("0 <= RandomReal[] < 1", True), ("RandomReal[1, 2, WorkingPrecision -> 20]", Raises()),
    ("SeedRandom[3]; seq3 = RandomReal[1, 5]; SeedRandom[3]; seq3 === RandomReal[1, 5]", True),
    ("MemberQ[{a, b, c}, RandomChoice[{a, b, c}]]", True), ("Length[RandomChoice[{a, b}, 7]]", "7"),
    ("Union[RandomChoice[{0, 1} -> {x, y}, 10]]", "{y}"), ("RandomChoice[{1, 2} -> {x}]", Raises()),
    ("SeedRandom[5]; seq5 = RandomVariate[GammaDistribution[2, 3], 4]; SeedRandom[5]; "
     "seq5 === RandomVariate[GammaDistribution[2, 3], 4]", True),
    ("And @@ Map[0 <= # <= 1 &, RandomVariate[BetaDistribution[2, 3], 50]]", True),
    ("And @@ Map[IntegerQ, RandomVariate[GeometricDistribution[1/3], 20]]", True),
    ("Dimensions[RandomVariate[LaplaceDistribution[0, 1], {2, 3}]]", "{2, 3}"),
    ("RandomVariate[GammaDistribution[kk, 1]]", Raises()),
    ("Union[RandomVariate[EmpiricalDistribution[{0, 1} -> {7, 8}], 10]]", "{8}"),
    ("And @@ Map[1 <= # <= 3 &, RandomVariate[TransformedDistribution[x + 1, x \\[Distributed] UniformDistribution[{0, 2}]], 10]]", True),
    # estimation (closed-form maximum likelihood)
    ("FindDistributionParameters[{1, 2, 3, 4}, NormalDistribution[m, s]]", "{m -> 5/2, s -> Sqrt[5]/2}"),
    ("FindDistributionParameters[{1, 2, 3}, ExponentialDistribution[l]]", "{l -> 1/2}"),
    ("FindDistributionParameters[{0, 1, 1, 1}, BernoulliDistribution[p]]", "{p -> 3/4}"),
    ("FindDistributionParameters[{1, 2}, BetaDistribution[a, b]]", Raises()),
    ("EstimatedDistribution[{1, 2, 3, 4}, PoissonDistribution[m]]", "PoissonDistribution[5/2]"),
    ("EstimatedDistribution[{1, 5, 2}, UniformDistribution[{a, b}]]", "UniformDistribution[{1, 5}]"),
    ("EstimatedDistribution[{0, 1, 3}, GeometricDistribution[p]]", "GeometricDistribution[3/7]"),
    ("EstimatedDistribution[{1, 2, 3}, LaplaceDistribution[m, b]]", "LaplaceDistribution[2, 2/3]"),
    ("EstimatedDistribution[{1, E, E^2}, LogNormalDistribution[m, s]]", "LogNormalDistribution[1, Sqrt[2/3]]"),
    ("EstimatedDistribution[{1, 2}, NormalDistribution[0, s]]", Raises()),
    # tests
    ("KolmogorovSmirnovTest[{1/10, 1/2, 9/10}, UniformDistribution[{0, 1}], \"TestStatistic\"]", "7/30"),
    ("KolmogorovSmirnovTest[{0.1, 0.5, 0.9}, UniformDistribution[{0, 1}], \"TestStatistic\"]", Near(7 / 30)),
    ("KolmogorovSmirnovTest[{1, 2}]", Raises()), ("KolmogorovSmirnovTest[{1, 2}, PoissonDistribution[1]]", Raises()),
    ("KolmogorovSmirnovTest[{1, 2}, NormalDistribution[0, 1], \"Bogus\"]", Raises()),
    ("LocationTest[{1, 2, 3}]", Raises()), ("LocationTest[{1, 2, 3}, 0, \"Wilcoxon\"]", Raises()),
    ("VarianceTest[{1, 2, 3}]", Raises()), ("VarianceTest[{1, 2, 3}, 1, \"Levene\"]", Raises()),
    ("HistogramList[{1, 2, 2, 3, 5}, {2}]", "{{0, 2, 4, 6}, {1, 3, 1}}"),
    ("HistogramList[{1, 2, 2, 3, 5}, {0, 6, 3}]", "{{0, 3, 6}, {3, 2}}"),
    ("HistogramList[{1, 2, 2, 3, 5}, {{0, 1, 4, 10}}]", "{{0, 1, 4, 10}, {0, 4, 1}}"),
    ("HistogramList[{1, 2}]", Raises()), ("HistogramList[{1, 2}, {0}]", Raises()),
]


# ─────────────────────────── statistics: SciPy / NumPy references ───────────────────────────

STATS_SCIPY = []
if HAVE_SCIPY:
    import numpy as np
    import scipy.stats as st

    SCIPY_DISTS = [
        ("GammaDistribution[2.5, 1.5]", st.gamma(2.5, scale=1.5), [0.7, 3.2]),
        ("BetaDistribution[2.5, 1.5]", st.beta(2.5, 1.5), [0.2, 0.75]),
        ("ChiSquareDistribution[5]", st.chi2(5), [1.5, 7.0]),
        ("FRatioDistribution[5, 11]", st.f(5, 11), [0.5, 2.0]),
        ("LogNormalDistribution[0.3, 0.8]", st.lognorm(0.8, scale=math.exp(0.3)), [0.5, 2.5]),
        ("CauchyDistribution[1, 2]", st.cauchy(1, 2), [-1.0, 3.5]),
        ("WeibullDistribution[1.7, 2.2]", st.weibull_min(1.7, scale=2.2), [0.5, 3.0]),
        ("LaplaceDistribution[1, 2]", st.laplace(1, 2), [-1.0, 2.5]),
        ("LogisticDistribution[1, 2]", st.logistic(1, 2), [-1.0, 2.5]),
        ("RayleighDistribution[1.5]", st.rayleigh(scale=1.5), [0.5, 2.5]),
        ("ParetoDistribution[2, 5]", st.pareto(5, scale=2), [2.5, 4.0]),
        ("GumbelDistribution[1, 2]", st.gumbel_l(1, 2), [-1.0, 2.5]),
        ("ExtremeValueDistribution[1, 2]", st.gumbel_r(1, 2), [-1.0, 2.5]),
        ("TriangularDistribution[{1, 4}, 2]", st.triang(1 / 3, loc=1, scale=3), [1.5, 3.0]),
        ("ErlangDistribution[3, 2]", st.erlang(3, scale=0.5), [0.5, 2.0]),
        ("ChiDistribution[3]", st.chi(3), [0.5, 2.0]),
        ("StudentTDistribution[1, 2, 7]", st.t(7, loc=1, scale=2), [0.0, 3.0]),
        ("BernoulliDistribution[0.3]", st.bernoulli(0.3), [0, 1]),
        ("GeometricDistribution[0.3]", st.geom(0.3, loc=-1), [0, 4]),
        ("NegativeBinomialDistribution[3, 0.4]", st.nbinom(3, 0.4), [2, 6]),
        ("HypergeometricDistribution[6, 7, 20]", st.hypergeom(20, 7, 6), [1, 3]),
        ("DiscreteUniformDistribution[{2, 9}]", st.randint(2, 10), [3, 8]),
    ]
    for src, ref, pts in SCIPY_DISTS:
        discrete = hasattr(ref.dist, "pmf")
        for x in pts:
            want = ref.pmf(x) if discrete else ref.pdf(x)
            STATS_SCIPY.append((f"PDF[{src}, {x}]", Near(float(want), 1e-9)))
            STATS_SCIPY.append((f"CDF[{src}, {x}]", Near(float(ref.cdf(x)), 1e-9)))
            STATS_SCIPY.append((f"SurvivalFunction[{src}, {x}]", Near(float(ref.sf(x)), 1e-9)))
        STATS_SCIPY.append((f"InverseCDF[{src}, 0.3]", Near(float(ref.ppf(0.3)), 1e-8)))
        STATS_SCIPY.append((f"Median[{src}]" if "." in src else f"InverseCDF[{src}, 0.5]",
                            Near(float(ref.median()), 1e-8)))
        m, v, sk, ku = (float(t) for t in ref.stats(moments="mvsk"))
        for head, want in (("Mean", m), ("Variance", v), ("StandardDeviation", math.sqrt(v) if v == v else v),
                           ("Skewness", sk), ("Kurtosis", ku + 3)):
            if math.isfinite(want):
                STATS_SCIPY.append((f"N[{head}[{src}]]", Near(want, 1e-8)))
            else:
                STATS_SCIPY.append((f"{head}[{src}]", "Indeterminate"))

    # hazard = pdf/sf, moments
    for src, ref, x in (("GammaDistribution[2, 3]", st.gamma(2, scale=3), 2.0),
                        ("WeibullDistribution[2, 1]", st.weibull_min(2), 0.8),
                        ("LogNormalDistribution[0, 1]", st.lognorm(1), 1.5)):
        STATS_SCIPY.append((f"HazardFunction[{src}, {x}]", Near(float(ref.pdf(x) / ref.sf(x)), 1e-9)))
    for src, ref, r in (("GammaDistribution[2, 3]", st.gamma(2, scale=3), 4),
                        ("BetaDistribution[2, 3]", st.beta(2, 3), 3),
                        ("LogNormalDistribution[0, 1/2]", st.lognorm(0.5), 3),
                        ("RayleighDistribution[2]", st.rayleigh(scale=2), 3),
                        ("ChiDistribution[4]", st.chi(4), 3),
                        ("HypergeometricDistribution[6, 7, 20]", st.hypergeom(20, 7, 6), 3)):
        STATS_SCIPY.append((f"N[Moment[{src}, {r}]]", Near(float(ref.moment(r)), 1e-9)))
    for src, ref, r in (("GammaDistribution[2, 3]", st.gamma(2, scale=3), 3),
                        ("PoissonDistribution[3]", st.poisson(3), 4),
                        ("NegativeBinomialDistribution[3, 2/5]", st.nbinom(3, 0.4), 3)):
        c = float(ref.expect(lambda t: (t - ref.mean()) ** r))
        STATS_SCIPY.append((f"N[CentralMoment[{src}, {r}]]", Near(c, 1e-7)))
    STATS_SCIPY += [
        ("N[Cumulant[GammaDistribution[2, 3], 3]]", Near(2 * 2 * 27.0)),
        ("N[Cumulant[BinomialDistribution[10, 1/3], 3]]", Near(10 * (1 / 3) * (2 / 3) * (1 / 3), 1e-9)),
        ("N[Cumulant[GeometricDistribution[1/2], 2]]", Near(float(st.geom(0.5).var()))),
        ("N[CharacteristicFunction[ExponentialDistribution[2], 1.5]]", Near(2 / (2 - 1.5j), 1e-12)),
        ("N[CharacteristicFunction[DiscreteUniformDistribution[{1, 3}], 0.5]]",
         Near(sum(np.exp(1j * 0.5 * k) for k in (1, 2, 3)) / 3, 1e-12)),
        ("N[CharacteristicFunction[LaplaceDistribution[0, 1], 0.5]]", Near(1 / (1 + 0.25), 1e-12)),
        ("N[MomentGeneratingFunction[BinomialDistribution[5, 0.3], 0.4]]", Near((0.7 + 0.3 * math.exp(0.4)) ** 5)),
        ("N[MomentGeneratingFunction[HypergeometricDistribution[2, 3, 5], 0.4]]",
         Near(float(st.hypergeom(5, 3, 2).expect(lambda k: np.exp(0.4 * k))), 1e-9)),
        ("N[MomentGeneratingFunction[GumbelDistribution[0, 1], 0.3]]", Near(math.gamma(1.3), 1e-12)),
    ]
    # data statistics against NumPy/SciPy
    data_a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    data_b = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8]
    data_f = [0.5, 2.25, 1.75, 3.5, 0.25, 4.0]
    la, lb, lf = (str(d).replace("[", "{").replace("]", "}") for d in (data_a, data_b, data_f))
    STATS_SCIPY += [
        (f"N[SpearmanRho[{la}, {lb}]]", Near(float(st.spearmanr(data_a, data_b)[0]), 1e-12)),
        (f"SpearmanRho[{lf}, {{1, 3, 2, 5, 4, 6}}]", Near(float(st.spearmanr(data_f, [1, 3, 2, 5, 4, 6])[0]), 1e-12)),
        (f"N[KendallTau[{la}, {lb}]]", Near(float(st.kendalltau(data_a, data_b)[0]), 1e-12)),
        (f"KendallTau[{lf}, {{1, 3, 2, 5, 4, 6}}]", Near(float(st.kendalltau(data_f, [1, 3, 2, 5, 4, 6])[0]), 1e-12)),
        (f"N[MeanDeviation[{la}]]", Near(float(np.mean(np.abs(np.array(data_a) - np.mean(data_a)))))),
        (f"N[MedianDeviation[{la}]]", Near(float(st.median_abs_deviation(data_a)))),
        (f"MedianDeviation[{lf}]", Near(float(st.median_abs_deviation(data_f)))),
        (f"N[RootMeanSquare[{la}]]", Near(float(np.sqrt(np.mean(np.square(data_a)))))),
        (f"N[Moment[{la}, 3]]", Near(float(np.mean(np.power(data_a, 3))))),
        (f"N[CentralMoment[{la}, 4]]", Near(float(st.moment(data_a, 4)))),
        (f"CentralMoment[{lf}, 3]", Near(float(st.moment(data_f, 3)))),
        (f"N[Cumulant[{la}, 3]]", Near(float(st.moment(data_a, 3)))),
        (f"N[WinsorizedMean[{la}, 0.2]]", Near(float(np.mean(st.mstats.winsorize(np.array(data_a, float),
                                                                                limits=(0.2, 0.2)))))),
        (f"N[TrimmedVariance[{la}, 1/5]]", Near(float(np.var(sorted(data_a)[2:-2], ddof=1)))),
        (f"N[WinsorizedVariance[{la}, 1/5]]", Near(float(np.var(st.mstats.winsorize(np.array(data_a, float),
                                                                                   limits=(0.2, 0.2)), ddof=1)))),
        (f"Standardize[{lf}]", NearList(list(st.zscore(data_f, ddof=1)))),
        (f"N[Covariance[{{{{1, 2}}, {{2, 5}}, {{4, 4}}, {{7, 9}}}}]]",
         Pred(lambda g: np.allclose(np.array(py(g), float), np.cov([[1, 2, 4, 7], [2, 5, 4, 9]])), "np.cov")),
        (f"N[Correlation[{{{{1, 2}}, {{2, 5}}, {{4, 4}}, {{7, 9}}}}]]",
         Pred(lambda g: np.allclose(np.array(py(g), float), np.corrcoef([[1, 2, 4, 7], [2, 5, 4, 9]])), "np.corrcoef")),
        # KS: SciPy exact one-sample test
        ("KolmogorovSmirnovTest[{0.1, 0.5, 0.9}, UniformDistribution[{0, 1}]]",
         Near(float(st.kstest([0.1, 0.5, 0.9], "uniform", method="exact").pvalue), 1e-9)),
        (f"KolmogorovSmirnovTest[{lf}, NormalDistribution[2, 1]]",
         Near(float(st.kstest(data_f, st.norm(2, 1).cdf, method="exact").pvalue), 1e-9)),
        (f"KolmogorovSmirnovTest[{lf}, ExponentialDistribution[1/2], \"TestStatistic\"]",
         Near(float(st.kstest(data_f, st.expon(scale=2).cdf).statistic), 1e-9)),
        # explicit-method location / variance tests
        (f"LocationTest[{la}, 3, \"T\"]", Near(float(st.ttest_1samp(data_a, 3).pvalue), 1e-9)),
        (f"LocationTest[{lf}, 1, \"T\"]", Near(float(st.ttest_1samp(data_f, 1).pvalue), 1e-9)),
        (f"LocationTest[{lb}, 0, \"T\"]", Near(float(st.ttest_1samp(data_b, 0).pvalue), 1e-9)),
        (f"VarianceTest[{{{la}, {lb}}}, 1, \"FisherRatio\"]",
         Near(float(2 * min(st.f.cdf(np.var(data_a, ddof=1) / np.var(data_b, ddof=1), 9, 9),
                            st.f.sf(np.var(data_a, ddof=1) / np.var(data_b, ddof=1), 9, 9))), 1e-9)),
        (f"VarianceTest[{la}, 4, \"ChiSquare\"]",
         Near(float(2 * min(st.chi2.cdf(9 * np.var(data_a, ddof=1) / 4, 9), st.chi2.sf(9 * np.var(data_a, ddof=1) / 4, 9))),
              1e-9)),
        (f"VarianceTest[{lf}, 1, \"ChiSquare\"]",
         Near(float(2 * min(st.chi2.cdf(5 * np.var(data_f, ddof=1), 5), st.chi2.sf(5 * np.var(data_f, ddof=1), 5))), 1e-9)),
        (f"HistogramList[{la}, {{0, 10, 2}}]",
         Pred(lambda g: py(g)[1] == list(np.histogram(data_a, bins=[0, 2, 4, 6, 8, 10])[0]), "np.histogram")),
        # MLE against SciPy fits
        (f"FindDistributionParameters[{lf}, NormalDistribution[m, s]]",
         Pred(lambda g: np.allclose([float(r.args[1]) for r in g.args], st.norm.fit(data_f)), "st.norm.fit")),
        (f"FindDistributionParameters[{lf}, LogNormalDistribution[m, s]]",
         Pred(lambda g: np.allclose([float(r.args[1]) for r in g.args],
                                    [math.log(st.lognorm.fit(data_f, floc=0)[2]), st.lognorm.fit(data_f, floc=0)[0]],
                                    rtol=1e-6), "st.lognorm.fit")),
        (f"FindDistributionParameters[{lf}, RayleighDistribution[s]]",
         Pred(lambda g: np.allclose([float(r.args[1]) for r in g.args], [st.rayleigh.fit(data_f, floc=0)[1]],
                                    rtol=1e-6), "st.rayleigh.fit")),
        # sample-moment check of random variates (seeded, loose tolerance)
        ("SeedRandom[11]; Abs[Mean[RandomVariate[GammaDistribution[2, 3], 20000]] - 6] < 0.2", True),
        ("SeedRandom[12]; Abs[Mean[RandomVariate[NegativeBinomialDistribution[3, 2/5], 20000]] - 4.5] < 0.15", True),
        ("SeedRandom[13]; Abs[Variance[RandomVariate[LaplaceDistribution[0, 1], 20000]] - 2] < 0.15", True),
        ("SeedRandom[14]; Abs[Mean[RandomVariate[GumbelDistribution[0, 1], 20000]] + 0.5772] < 0.05", True),
        ("SeedRandom[15]; Abs[Mean[RandomVariate[HypergeometricDistribution[6, 7, 20], 20000]] - 2.1] < 0.05", True),
    ]


CASES = GRAPHS + GRAPHS_NX + STATS + STATS_SCIPY
if not HAVE_NX:
    NEEDS_NX = ("PlanarGraphQ", "IsomorphicGraphQ", "FindGraphIsomorphism")
    CASES = [c for c in CASES if not any(n in c[0] for n in NEEDS_NX)]
if not HAVE_NUMPY:
    NEEDS_NP = ("EigenvectorCentrality", "HITSCentrality", "PageRankCentrality")
    CASES = [c for c in CASES if not any(n in c[0] for n in NEEDS_NP)]


if not HAVE_SCIPY:
    # numeric quantiles without a closed form, multinormal CDFs and test p-values use SciPy
    NEEDS_SP = ("KolmogorovSmirnovTest[{0.1, 0.5, 0.9}, UniformDistribution[{0, 1}]]", "InterquartileRange[Normal")
    CASES = [c for c in CASES if not any(n in c[0] for n in NEEDS_SP)]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_batch5(case):
    check(case)
