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
    ("Graph[{UndirectedEdge[1, 2, t]}]", Raises()), ("Graph[{1, 1}, {}]", Raises()),
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
    ("Length[RandomGraph[{5, 3}, 4]]", "4"), ("RandomGraph[{4, 7}]", Raises()), ("RandomGraph[BernoulliGraphDistribution[5, 0.5]]", Raises()),
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

CASES = GRAPHS + GRAPHS_NX
if not HAVE_NX:
    NEEDS_NX = ("PlanarGraphQ", "IsomorphicGraphQ", "FindGraphIsomorphism")
    CASES = [c for c in CASES if not any(n in c[0] for n in NEEDS_NX)]
if not HAVE_NUMPY:
    NEEDS_NP = ("EigenvectorCentrality", "HITSCentrality", "PageRankCentrality")
    CASES = [c for c in CASES if not any(n in c[0] for n in NEEDS_NP)]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_batch5(case):
    check(case)
