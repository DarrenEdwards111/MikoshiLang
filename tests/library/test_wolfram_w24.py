"""Differential tests for coverage batch W24 (graphs II: annotations, constructions, measures, colouring, cuts,
flows, isomorphism, polynomials, planar faces, random graph distributions).

Reference values (cited per block):

* Wolfram documentation (reference.wolfram.com/language/ref/<Name>.html): usage, documented identities such as
  GraphJoin[point, cycle] = wheel, GraphSum via AdjacencyGraph, GraphLinkEfficiency = 1 - MeanGraphDistance/m;
* classical graph theory values: chromatic numbers and indices (Petersen 3 and 4, K_n, odd cycles), matching
  numbers, Menger/Whitney connectivities, automorphism group orders (Petersen 120, C_n 2n, K_n n!), Tutte
  polynomials of C_n and K_4 (Bollobas, Modern Graph Theory X.2), flow polynomials, Euler's formula for faces;
* NetworkX (computed in this file, independent of MikoshiLang): edge betweenness, assortativity, average neighbour
  degree, matchings, cuts.

Functions that return one of several valid answers (a colouring, a matching, a cut, a set of paths, ...) are checked
with predicates that the answer is valid and optimal.  Random graph samplers are checked by structural invariants.
"""
import importlib.util
import itertools
import math

import pytest
import sympy as sp

from _harness import Pred, Raises, check, ev, ids, to_complex
from mikoshilang.evaluate import evaluate
from mikoshilang.expr import Expr, Symbol

HAVE_NX = importlib.util.find_spec("networkx") is not None


def py(x):
    """MikoshiLang value -> plain Python (lists, ints, strings, edge tuples)."""
    if isinstance(x, Expr) and x.head == "List":
        return [py(a) for a in x.args]
    if isinstance(x, Expr) and x.head in ("UndirectedEdge", "DirectedEdge"):
        return (x.head[0], py(x.args[0]), py(x.args[1]))
    if isinstance(x, Symbol):
        return x.name
    if isinstance(x, sp.Integer):
        return int(x)
    return x


def Near(want, tol=1e-9):
    return Pred(lambda g: abs(to_complex(g) - complex(want)) <= tol * max(1, abs(complex(want))), f"~{want}")


def NearList(want, tol=1e-9):
    def ok(g):
        return (getattr(g, "head", None) == "List" and len(g.args) == len(want)
                and all(abs(to_complex(a) - complex(b)) <= tol * max(1, abs(complex(b))) for a, b in zip(g.args, want)))
    return Pred(ok, f"~{list(want)}")


def P(f, what):
    return Pred(f, what)


def gsrc(edges, directed=False):
    op = " -> " if directed else " <-> "
    return "Graph[{" + ", ".join(f"{a}{op}{b}" for a, b in edges) + "}]"


PET = "PetersenGraph[]"
PETERSEN_EDGES = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
                  (6, 8), (8, 10), (7, 10), (7, 9), (6, 9)]
CUBE = "HypercubeGraph[3]"
# a graph with a pendant: triangle 1-2-3 with 3-4
TRI_P = "Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4}]"
# directed graph for dominators: 1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4, 4 -> 5 (diamond then tail)
DOM = "Graph[{1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4, 4 -> 5}]"


def edges_of(src):
    return [(e[1], e[2]) for e in py(ev(f"EdgeList[{src}]"))]


def adj_sets(src):
    vs = py(ev(f"VertexList[{src}]"))
    nb = {v: set() for v in vs}
    for a, b in edges_of(src):
        if a != b:
            nb[a].add(b)
            nb[b].add(a)
    return vs, nb


def proper_coloring(src, k):
    """Pred: a list of colours, one per vertex, proper, with exactly k distinct colours."""
    def ok(r):
        cols = py(r)
        vs, nb = adj_sets(src)
        if len(cols) != len(vs) or len(set(cols)) != k:
            return False
        c = dict(zip(vs, cols))
        return all(c[u] != c[w] for u in vs for w in nb[u])
    return Pred(ok, f"proper {k}-colouring of {src}")


def proper_edge_coloring(src, k):
    def ok(r):
        cols = py(r)
        es = edges_of(src)
        if len(cols) != len(es) or len(set(cols)) != k:
            return False
        for (e1, c1), (e2, c2) in itertools.combinations(zip(es, cols), 2):
            if set(e1) & set(e2) and c1 == c2:
                return False
        return True
    return Pred(ok, f"proper {k}-edge-colouring of {src}")


def matching_of(src, size):
    def ok(r):
        es = [(e[1], e[2]) for e in py(r)]
        ge = edges_of(src)
        used = set()
        for a, b in es:
            if (a, b) not in ge and (b, a) not in ge or a in used or b in used:
                return False
            used |= {a, b}
        return len(es) == size
    return Pred(ok, f"matching of size {size}")


def edge_cover_of(src, size):
    def ok(r):
        es = [(e[1], e[2]) for e in py(r)]
        vs = py(ev(f"VertexList[{src}]"))
        return len(es) == size and {v for e in es for v in e} == set(vs)
    return Pred(ok, f"edge cover of size {size}")


def disconnects_edges(src, size):
    def ok(r):
        cut = [(e[1], e[2]) for e in py(r)]
        if len(cut) != size:
            return False
        rest = list(edges_of(src))
        for e in cut:
            rest.remove(e)
        vs = py(ev(f"VertexList[{src}]"))
        nb = {v: set() for v in vs}
        for a, b in rest:
            nb[a].add(b)
            nb[b].add(a)
        seen, st = {vs[0]}, [vs[0]]
        while st:
            u = st.pop()
            for w in nb[u] - seen:
                seen.add(w)
                st.append(w)
        return len(seen) < len(vs)
    return Pred(ok, f"edge cut of size {size}")


def separates(src, size, s=None, t=None):
    def ok(r):
        cut = set(py(r))
        if len(cut) != size:
            return False
        vs, nb = adj_sets(src)
        keep = [v for v in vs if v not in cut]
        start = s if s is not None else keep[0]
        seen, st = {start}, [start]
        while st:
            u = st.pop()
            for w in nb[u] - seen - cut:
                seen.add(w)
                st.append(w)
        return (t not in seen) if t is not None else len(seen) < len(keep)
    return Pred(ok, f"vertex separator of size {size}")


def disjoint_paths(src, s, t, count, kind):
    def ok(r):
        paths = py(r)
        if len(paths) != count:
            return False
        vs, nb = adj_sets(src)
        used_e, used_v = [], set()
        for p in paths:
            if p[0] != s or p[-1] != t:
                return False
            for a, b in zip(p, p[1:]):
                if b not in nb[a]:
                    return False
                used_e.append(frozenset((a, b)))
            inner = set(p[1:-1])
            if kind == "vertex" and inner & used_v:
                return False
            used_v |= inner
        return kind == "vertex" or len(used_e) == len(set(used_e))
    return Pred(ok, f"{count} {kind}-disjoint paths")


# ═══════════════════ annotations (AnnotationValue/Annotate/AnnotationKeys/AnnotationDelete) ═══════════════════
# reference: AnnotationValue/Annotate documentation (graph properties, vertex and edge annotations)
ANNOTATIONS = [
    ("AnnotationValue[Graph[{1 <-> 2}, VertexLabels -> \"Name\"], VertexLabels]", "\"Name\""),
    ("AnnotationValue[{Graph[{1 <-> 2, 2 <-> 3}, EdgeWeight -> {5, 7}], 2 <-> 3}, EdgeWeight]", 7),
    ("AnnotationValue[{Graph[{1 <-> 2}, VertexWeight -> {3, 4}], 2}, VertexWeight]", 4),
    ("AnnotationValue[{Annotate[{Graph[{1 <-> 2}], 1}, \"color\" -> \"red\"], 1}, \"color\"]", "\"red\""),
    ("AnnotationValue[Annotate[Graph[{1 <-> 2}], \"title\" -> \"T\"], \"title\"]", "\"T\""),
    ("AnnotationValue[Graph[{1 <-> 2}], \"missing\"]", Raises(ValueError)),
    ("AnnotationValue[5, VertexLabels]", Raises(NotImplementedError)),
    ("AnnotationValue[w24x, VertexLabels]", "AnnotationValue[w24x, VertexLabels]"),
    ("Annotate[{Graph[{1 <-> 2, 2 <-> 3}, EdgeWeight -> {1, 1}], 1 <-> 2}, EdgeWeight -> 9]",
     "Graph[{1, 2, 3}, {UndirectedEdge[1, 2], UndirectedEdge[2, 3]}, EdgeWeight -> {9, 1}]"),
    ("VertexList[Annotate[{Graph[{1 <-> 2}], 2}, VertexLabels -> \"b\"]]", "{1, 2}"),
    ("Annotate[{Graph[{1 <-> 2}], 7}, VertexLabels -> \"b\"]", Raises(ValueError)),
    ("MemberQ[AnnotationKeys[Annotate[Graph[{1 <-> 2}], \"title\" -> 1]], \"title\"]", True),
    ("AnnotationKeys[Graph[{1 <-> 2}, EdgeWeight -> {2}]]", "{EdgeWeight}"),
    ("MemberQ[AnnotationKeys[{Annotate[{Graph[{1 <-> 2}], 1}, \"c\" -> 1], 1}], \"c\"]", True),
    ("AnnotationKeys[{1, 2}]", Raises(NotImplementedError)),
    ("AnnotationDelete[Graph[{1 <-> 2}, EdgeWeight -> {2}]]", "Graph[{1, 2}, {UndirectedEdge[1, 2]}]"),
    ("AnnotationKeys[AnnotationDelete[Graph[{1 <-> 2}, EdgeWeight -> {2}, VertexLabels -> \"Name\"], "
     "EdgeWeight]]", "{VertexLabels}"),
    ("AnnotationKeys[{AnnotationDelete[{Annotate[{Graph[{1 <-> 2}], 1}, \"c\" -> 1], 1}], 1}]", "{}"),
    ("AnnotationDelete[{Graph[{1 <-> 2}, VertexWeight -> {1, 2}], 1}, VertexWeight]", Raises(NotImplementedError)),
]

# ═══════════════════ option symbols: kept on the graph and read back ═══════════════════
OPTIONS = []
for _o, _v in [("GraphLayout", "\"SpringEmbedding\""), ("VertexLabels", "\"Name\""), ("EdgeLabels", "\"Name\""),
               ("VertexStyle", "Red"), ("EdgeStyle", "Blue"), ("VertexLabelStyle", "Large"),
               ("EdgeLabelStyle", "Small"), ("VertexShape", "\"Square\""), ("VertexShapeFunction", "\"Star\""),
               ("VertexSize", "Large"), ("EdgeShapeFunction", "\"Line\""), ("GraphHighlight", "{1}"),
               ("GraphHighlightStyle", "\"Thick\""), ("VertexCoordinates", "{{0, 0}, {1, 0}}"),
               ("MultiedgeStyle", "0.3"), ("SelfLoopStyle", "0.5")]:
    OPTIONS += [
        (f"AnnotationValue[Graph[{{1 <-> 2}}, {_o} -> {_v}], {_o}]", _v),
        (f"MemberQ[AnnotationKeys[Graph[{{1 <-> 2}}, {_o} -> {_v}]], {_o}]", True),
        (f"EdgeList[Graph[{{1 <-> 2}}, {_o} -> {_v}]]", "{UndirectedEdge[1, 2]}"),
    ]
OPTIONS += [
    ("AnnotationValue[{Graph[{1 <-> 2, 2 <-> 3}, EdgeCost -> {4, 6}], 2 <-> 3}, EdgeCost]", 6),
    ("FindMinimumCostFlow[Graph[{1 -> 2, 2 -> 3, 1 -> 3}, EdgeCost -> {1, 1, 5}], 1, 3, 1]", 2),   # cheapest route 1->2->3
    ("Graph[{1 <-> 2}, EdgeCost -> {1, 2}]", Raises(ValueError)),
    ("AnnotationValue[{Graph[{1 <-> 2}, VertexCapacity -> {3, 4}], 2}, VertexCapacity]", 4),
    # two parallel unit routes 1-2-4 and 1-3-4; capacity 1 on vertex 2 and 3 caps the flow at 2, on vertex 4 at 1
    ("FindMaximumFlow[Graph[{1, 2, 3, 4}, {1 -> 2, 2 -> 4, 1 -> 3, 3 -> 4}, EdgeCapacity -> {5, 5, 5, 5}, "
     "VertexCapacity -> {Infinity, 1, 1, Infinity}], 1, 4]", 2),
    ("FindMaximumFlow[Graph[{1, 2, 3, 4}, {1 -> 2, 2 -> 4, 1 -> 3, 3 -> 4}, EdgeCapacity -> {5, 5, 5, 5}, "
     "VertexCapacity -> {Infinity, 5, 5, 1}], 1, 4]", 1),
    ("FindMaximumFlow[Graph[{1 -> 2}, VertexCapacity -> {1, -1}], 1, 2]", Raises(ValueError)),
    ("AnnotationValue[{Annotate[{Graph[{1 <-> 2}], 2}, \"k\" -> 5], 2}, \"k\"]", 5),
    ("MemberQ[AnnotationKeys[Annotate[Graph[{1 <-> 2}], \"p\" -> 1]], \"p\"]", True),
    ("AnnotationRules", "AnnotationRules"),
    ("AnnotationValue[{Graph[{1 <-> 2}, AnnotationRules -> {1 -> {\"c\" -> 5}}], 1}, \"c\"]", 5),
    ("AnnotationValue[Graph[{1 <-> 2}, AnnotationRules -> {\"GraphProperties\" -> {\"t\" -> 3}}], \"t\"]", 3),
    ("AnnotationValue[{Graph[{1 <-> 2}, AnnotationRules -> 5], 1}, \"c\"]", Raises(ValueError)),
    ("EdgeList[Graph[{1 -> 2, 2 -> 3}, DirectedEdges -> False]]", "{UndirectedEdge[1, 2], UndirectedEdge[2, 3]}"),
    ("EdgeList[Graph[{1 <-> 2}, DirectedEdges -> True]]", "{DirectedEdge[1, 2]}"),
    ("EdgeList[CycleGraph[3, DirectedEdges -> True]]", "{DirectedEdge[1, 2], DirectedEdge[2, 3], DirectedEdge[3, 1]}"),
    ("PetersenGraph[DirectedEdges -> True]", Raises(NotImplementedError)),
    ("Graph[{1 <-> 2}, DirectedEdges -> 3]", Raises(ValueError)),
    # Direction (Limit option) - Limit documentation
    ("Limit[1/w24t, w24t -> 0, Direction -> \"FromAbove\"]", "Infinity"),
    ("Limit[1/w24t, w24t -> 0, Direction -> \"FromBelow\"]", "-Infinity"),
    ("Direction", "Direction"),
]


def iso(src2):
    return Pred(lambda g: evaluate(Expr("IsomorphicGraphQ", g, ev(src2))) is True, f"isomorphic to {src2}")


def counts(n, m):
    return Pred(lambda g: py(evaluate(Expr("VertexCount", g))) == n and py(evaluate(Expr("EdgeCount", g))) == m,
                f"{n} vertices, {m} edges")


def regular(k):
    return Pred(lambda g: set(py(evaluate(Expr("VertexDegree", g)))) == {k}, f"{k}-regular")


# ═══════════════════ constructions ═══════════════════
# ExpressionGraph: documented preorder tree (root 1, then each argument's subtree)
CONSTRUCTIONS = [
    ("EdgeList[ExpressionGraph[w24f[w24a, w24g[w24b, w24c]]]]",
     "{UndirectedEdge[1, 2], UndirectedEdge[1, 3], UndirectedEdge[3, 4], UndirectedEdge[3, 5]}"),
    ("VertexCount[ExpressionGraph[w24f[w24a, w24g[w24b, w24c]], 1]]", 3),
    ("VertexCount[ExpressionGraph[w24a]]", 1),
    ("ExpressionGraph[w24f[w24a], w24n]", "ExpressionGraph[w24f[w24a], w24n]"),
    ("ExpressionGraph[w24f[w24a], 1, _]", Raises(NotImplementedError)),
    ("ExpressionGraph[w24f[w24a], -1]", Raises(ValueError)),
    # TorusGraph = Cartesian product of cycles (Wolfram docs); 2k-regular with n k edges
    ("TorusGraph[{3, 3}]", counts(9, 18)),
    ("TorusGraph[{4, 5}]", iso("GraphProduct[CycleGraph[4], CycleGraph[5]]")),
    ("TorusGraph[{3, 4, 5}]", regular(6)),
    ("TorusGraph[{2, 3}]", Raises(NotImplementedError)),
    ("TorusGraph[3]", Raises(TypeError)),
    ("TorusGraph[{w24n, 3}]", "TorusGraph[{w24n, 3}]"),
    # DeBruijnGraph[m, n]: m^n vertices, m^(n+1) edges, shift map (de Bruijn 1946)
    ("EdgeList[DeBruijnGraph[2, 2]]",
     "{DirectedEdge[1, 1], DirectedEdge[1, 2], DirectedEdge[2, 3], DirectedEdge[2, 4], DirectedEdge[3, 1], "
     "DirectedEdge[3, 2], DirectedEdge[4, 3], DirectedEdge[4, 4]}"),
    ("DeBruijnGraph[2, 3]", counts(8, 16)),
    ("VertexOutDegree[DeBruijnGraph[3, 2]]", "{3, 3, 3, 3, 3, 3, 3, 3, 3}"),
    ("DeBruijnGraph[2, 2, \"LineGraph\"]", Raises(NotImplementedError)),
    ("DeBruijnGraph[0, 2]", Raises(ValueError)),
    ("DeBruijnGraph[w24m, 2]", "DeBruijnGraph[w24m, 2]"),
    # HararyGraph[k, n]: Ceiling[k n/2] edges, k-connected (Harary 1962)
    ("HararyGraph[2, 5]", iso("CycleGraph[5]")),
    ("HararyGraph[3, 8]", counts(8, 12)),
    ("VertexConnectivity[HararyGraph[3, 7]]", 3),
    ("HararyGraph[5, 4]", Raises(ValueError)),
    ("HararyGraph[w24k, 6]", "HararyGraph[w24k, 6]"),
    # KnightTourGraph: 4 m n - 6 (m + n) + 8 edges; 168 on the chessboard
    ("EdgeCount[KnightTourGraph[8, 8]]", 168),
    ("KnightTourGraph[3, 3]", counts(9, 8)),
    ("EdgeCount[KnightTourGraph[3, 4]]", 14),
    ("KnightTourGraph[0, 3]", Raises(ValueError)),
    ("KnightTourGraph[w24m, 3]", "KnightTourGraph[w24m, 3]"),
    # ButterflyGraph[n, b]: (n + 1) b^n vertices, n b^(n+1) edges
    ("ButterflyGraph[2]", counts(12, 16)),
    ("ButterflyGraph[1, 3]", counts(6, 9)),
    ("Sort[DeleteDuplicates[VertexDegree[ButterflyGraph[3]]]]", "{2, 4}"),
    ("ButterflyGraph[-1]", Raises(ValueError)),
    ("ButterflyGraph[w24n]", "ButterflyGraph[w24n]"),
    # BuckyballGraph: C60, 60 vertices, 90 edges, cubic, girth 5, |Aut| = 120 (icosahedral group)
    ("BuckyballGraph[]", counts(60, 90)),
    ("BuckyballGraph[]", regular(3)),
    ("GraphTriangleCount[BuckyballGraph[]]", 0),
    ("BuckyballGraph[2]", Raises(NotImplementedError)),
    # GraphSum: documented as AdjacencyGraph of summed adjacency matrices
    ("GraphSum[CycleGraph[4], GraphComplement[CycleGraph[4]]]", iso("CompleteGraph[4]")),
    ("EdgeCount[GraphSum[PathGraph[{1, 2, 3}], PathGraph[{1, 2, 3}]]]", 4),
    ("GraphSum[PathGraph[{1, 2}], PathGraph[{1, 2}], PathGraph[{1, 2}]]", counts(2, 3)),
    ("GraphSum[PathGraph[{1, 2}], CycleGraph[3]]", Raises(NotImplementedError)),
    ("GraphSum[3, CycleGraph[3]]", Raises(TypeError)),
    # GraphJoin: documented examples (point + cycle = wheel; edgeless + edgeless = complete bipartite)
    ("GraphJoin[Graph[{1}, {}], CycleGraph[5]]", iso("WheelGraph[6]")),
    ("GraphJoin[Graph[{1, 2}, {}], Graph[{1, 2, 3}, {}]]", iso("CompleteGraph[{2, 3}]")),
    ("GraphJoin[CompleteGraph[2], CompleteGraph[3]]", iso("CompleteGraph[5]")),
    ("GraphJoin[Graph[{1 -> 2}], CycleGraph[3]]", Raises(NotImplementedError)),
    # GraphProduct: Cartesian K2 x K2 = C4, tensor K2 x K2 = 2 K2, lexicographic P2[P2] = K4 (documented definitions)
    ("GraphProduct[PathGraph[{1, 2}], PathGraph[{1, 2}]]", iso("CycleGraph[4]")),
    ("EdgeCount[GraphProduct[PathGraph[{1, 2}], PathGraph[{1, 2}], \"Tensor\"]]", 2),
    ("GraphProduct[PathGraph[{1, 2}], PathGraph[{1, 2}], \"Lexicographical\"]", iso("CompleteGraph[4]")),
    ("EdgeCount[GraphProduct[CycleGraph[3], CycleGraph[3], \"Normal\"]]", 36),  # strong product: 9*8/2 = K9
    ("GraphProduct[CycleGraph[3], CycleGraph[3], \"Strange\"]", Raises(ValueError)),
    # GraphPower
    ("GraphPower[CycleGraph[7], 2]", iso("CirculantGraph[7, {1, 2}]")),
    ("GraphPower[PathGraph[{1, 2, 3, 4}], 3]", iso("CompleteGraph[4]")),
    ("EdgeList[GraphPower[Graph[{1 -> 2, 2 -> 3}], 2]]", "{DirectedEdge[1, 2], DirectedEdge[1, 3], DirectedEdge[2, 3]}"),
    ("GraphPower[CycleGraph[5], -1]", Raises(ValueError)),
    ("GraphPower[CycleGraph[5], w24k]", "GraphPower[CycleGraph[5], w24k]"),
    # SimpleGraph
    ("SimpleGraph[Graph[{1 <-> 2, 1 <-> 2, 2 <-> 2, 2 <-> 3}]]",
     "Graph[{1, 2, 3}, {UndirectedEdge[1, 2], UndirectedEdge[2, 3]}]"),
    ("SimpleGraphQ[SimpleGraph[Graph[{1 <-> 1, 1 <-> 2}]]]", True),
    ("EdgeCount[SimpleGraph[CycleGraph[5]]]", 5),
    ("SimpleGraph[Graph[{1 -> 2}]]", Raises(NotImplementedError)),
    # RelationGraph (documented: Automatic gives undirected for symmetric relations)
    ("EdgeList[RelationGraph[Less, {1, 2, 3}]]", "{DirectedEdge[1, 2], DirectedEdge[1, 3], DirectedEdge[2, 3]}"),
    ("RelationGraph[Unequal, {1, 2, 3}]", iso("CompleteGraph[3]")),
    ("EdgeCount[RelationGraph[Function[{w24a, w24b}, Mod[w24a - w24b, 2] == 0 && w24a != w24b], Range[4]]]", 2),
    ("RelationGraph[Less, 3]", Raises(TypeError)),
    # BooleanGraph
    ("EdgeList[BooleanGraph[Not, CycleGraph[4]]]", "{UndirectedEdge[1, 3], UndirectedEdge[2, 4]}"),
    ("BooleanGraph[Or, CycleGraph[4], GraphComplement[CycleGraph[4]]]", iso("CompleteGraph[4]")),
    ("EdgeCount[BooleanGraph[And, CycleGraph[4], CompleteGraph[4]]]", 4),
    ("BooleanGraph[Or, CycleGraph[4], CycleGraph[5]]", Raises(NotImplementedError)),
    # WeightedAdjacencyGraph (Infinity = no edge)
    ("WeightedAdjacencyGraph[{{Infinity, 2}, {2, Infinity}}]", "Graph[{1, 2}, {UndirectedEdge[1, 2]}, EdgeWeight -> {2}]"),
    ("EdgeList[WeightedAdjacencyGraph[{{Infinity, 1}, {Infinity, Infinity}}]]", "{DirectedEdge[1, 2]}"),
    ("WeightedAdjacencyMatrix[WeightedAdjacencyGraph[{{Infinity, 3, 4}, {3, Infinity, Infinity}, {4, Infinity, Infinity}}]]"
     "// Normal", "{{0, 3, 4}, {3, 0, 0}, {4, 0, 0}}"),
    ("WeightedAdjacencyGraph[{{Infinity, \"a\"}, {\"a\", Infinity}}]", Raises(TypeError)),
    # KirchhoffGraph (self-check: KirchhoffMatrix of the result equals the input)
    ("EdgeList[KirchhoffGraph[{{1, -1}, {-1, 1}}]]", "{UndirectedEdge[1, 2]}"),
    ("KirchhoffMatrix[KirchhoffGraph[{{2, -1, -1}, {-1, 2, -1}, {-1, -1, 2}}]] // Normal",
     "{{2, -1, -1}, {-1, 2, -1}, {-1, -1, 2}}"),
    ("EdgeCount[KirchhoffGraph[{{2, -2}, {-2, 2}}]]", 2),
    ("KirchhoffGraph[{{1, -1}, {-1, 2}}]", Raises(ValueError)),
    # IncidenceGraph (self-check: IncidenceMatrix round trip)
    ("EdgeList[IncidenceGraph[{{1}, {1}}]]", "{UndirectedEdge[1, 2]}"),
    ("EdgeList[IncidenceGraph[{{-1}, {1}}]]", "{DirectedEdge[1, 2]}"),
    ("IncidenceMatrix[IncidenceGraph[{{1, 0}, {1, 1}, {0, 1}}]] // Normal", "{{1, 0}, {1, 1}, {0, 1}}"),
    ("IncidenceGraph[{{1}, {1}, {1}}]", Raises(ValueError)),
]

# ═══════════════════ edge rules, tags, editing ═══════════════════
EDITING = [
    ("EdgeRules[Graph[{1 -> 2, 2 -> 3}]]", "{1 -> 2, 2 -> 3}"),
    ("EdgeRules[PathGraph[{1, 2, 3}]]", "{1 -> 2, 2 -> 3}"),
    ("Graph[EdgeRules[Graph[{3 -> 1, 1 -> 2}]]] === Graph[{3 -> 1, 1 -> 2}]", True),
    ("EdgeRules[7]", Raises(TypeError)),
    ("Length[IncidenceList[CycleGraph[4], 1]]", 2),
    ("IncidenceList[PathGraph[{1, 2, 3}], 2]", "{UndirectedEdge[1, 2], UndirectedEdge[2, 3]}"),
    ("IncidenceList[Graph[{1 -> 2, 3 -> 1}], 1]", "{DirectedEdge[1, 2], DirectedEdge[3, 1]}"),
    ("IncidenceList[CycleGraph[4], 9]", Raises(ValueError)),
    ("IncidenceList[CycleGraph[4], 1, 2]", Raises(NotImplementedError)),
    ("EdgeTags[EdgeTaggedGraph[{1 <-> 2, 2 <-> 3} -> {w24a, w24b}]]", "{w24a, w24b}"),
    ("EdgeList[EdgeTaggedGraph[{1 -> 2} -> {7}]]", "{DirectedEdge[1, 2, 7]}"),
    ("VertexList[EdgeTaggedGraph[{1, 2, 3}, {1 <-> 2} -> {1}]]", "{1, 2, 3}"),
    ("EdgeTaggedGraph[{1 <-> 2, 2 <-> 3} -> {1}]", Raises(ValueError)),
    ("EdgeTaggedGraph[{1 <-> 2}]", Raises(NotImplementedError)),
    ("EdgeTags[IndexEdgeTaggedGraph[PathGraph[{1, 2, 3}]]]", "{1, 2}"),
    ("EdgeTags[IndexEdgeTaggedGraph[CycleGraph[3], 5]]", "{5, 6, 7}"),
    ("EdgeTaggedGraphQ[IndexEdgeTaggedGraph[CycleGraph[3]]]", True),
    ("IndexEdgeTaggedGraph[CycleGraph[3], \"a\"]", Raises(TypeError)),
    ("EdgeTags[EdgeTaggedGraph[{1 <-> 2, 2 <-> 1, 2 <-> 3} -> {w24a, w24b, w24c}], {1, 2}]", "{w24a, w24b}"),
    ("EdgeTags[CycleGraph[3]]", Raises(NotImplementedError)),
    ("EdgeTaggedGraphQ[CycleGraph[3]]", False),
    ("EdgeTaggedGraphQ[5]", False),
    ("EdgeTaggedGraphQ[w24x]", "EdgeTaggedGraphQ[w24x]"),
    ("VertexList[VertexReplace[PathGraph[{1, 2, 3}], {1 -> w24a}]]", "{w24a, 2, 3}"),
    ("EdgeList[VertexReplace[Graph[{1 -> 2}], {1 -> 2, 2 -> 1}]]", "{DirectedEdge[2, 1]}"),
    ("IsomorphicGraphQ[VertexReplace[PetersenGraph[], Table[w24i -> w24i + 10, {w24i, 10}]], PetersenGraph[]]", True),
    ("VertexReplace[PathGraph[{1, 2, 3}], {1 -> 3}]", Raises(ValueError)),
    ("VertexContract[CycleGraph[5], {1, 2}]", iso("CycleGraph[4]")),
    ("VertexList[VertexContract[PathGraph[{1, 2, 3, 4}], {{1, 2}, {3, 4}}]]", "{1, 3}"),
    ("EdgeCount[VertexContract[StarGraph[4], {1, 2}]]", 2),
    ("VertexContract[CycleGraph[4], {1, 3}]", Raises(NotImplementedError)),
    ("EdgeContract[CycleGraph[5], 1 <-> 2]", iso("CycleGraph[4]")),
    ("VertexCount[EdgeContract[PathGraph[{1, 2, 3, 4}], {1 <-> 2, 2 <-> 3, 3 <-> 4}]]", 1),
    ("EdgeList[EdgeContract[PathGraph[{1, 2, 3}], 2 <-> 3]]", "{UndirectedEdge[1, 2]}"),
    ("EdgeContract[CycleGraph[5], 1 <-> 3]", Raises(ValueError)),
    ("EdgeContract[CycleGraph[3], 1 <-> 2]", Raises(NotImplementedError)),
]

# ═══════════════════ components, highlight, planar / NN graphs, closures, Cayley graphs ═══════════════════
COMPONENTS = [
    ("VertexComponent[Graph[{1 -> 2, 2 -> 3}], {3}]", "{3, 2, 1}"),
    ("Sort[VertexComponent[Graph[{1 <-> 2, 3 <-> 4}], {1}]]", "{1, 2}"),
    ("VertexComponent[Graph[{1 -> 2, 2 -> 3, 3 -> 4}], {4}, 1]", "{4, 3}"),
    ("VertexComponent[Graph[{1 -> 2, 2 -> 3, 3 -> 4}], {4}, {2}]", "{2}"),
    ("VertexComponent[Graph[{1 -> 2}], {9}]", Raises(ValueError)),
    ("VertexList[VertexInComponentGraph[Graph[{1 -> 2, 2 -> 3, 4 -> 3}], {2}]]", "{1, 2}"),
    ("EdgeList[VertexInComponentGraph[Graph[{1 -> 2, 2 -> 3}], {3}, 1]]", "{DirectedEdge[2, 3]}"),
    ("VertexInComponentGraph[5, {1}]", Raises(TypeError)),
    ("VertexList[VertexOutComponentGraph[Graph[{1 -> 2, 2 -> 3, 4 -> 3}], {2}]]", "{2, 3}"),
    ("VertexList[VertexOutComponentGraph[Graph[{1 -> 2, 2 -> 3, 3 -> 4}], {1}, 2]]", "{1, 2, 3}"),
    ("VertexOutComponentGraph[Graph[{1 -> 2}], {7}]", Raises(ValueError)),
    ("VertexList /@ WeaklyConnectedGraphComponents[Graph[{1 -> 2, 3 -> 4, 4 -> 5}]]", "{{3, 4, 5}, {1, 2}}"),
    ("Length[WeaklyConnectedGraphComponents[Graph[{1 -> 2, 2 -> 3}]]]", 1),
    ("VertexList /@ WeaklyConnectedGraphComponents[Graph[{1 -> 2, 3 -> 4, 4 -> 5}], {1}]", "{{1, 2}}"),
    ("WeaklyConnectedGraphComponents[Graph[{1 -> 2}], _]", Raises(NotImplementedError)),
    ("AnnotationValue[HighlightGraph[PathGraph[{1, 2, 3}], {2}], GraphHighlight]", "{2}"),
    ("AnnotationValue[HighlightGraph[PathGraph[{1, 2, 3}], 1 <-> 2], GraphHighlight]", "{UndirectedEdge[1, 2]}"),
    ("EdgeList[HighlightGraph[CycleGraph[3], {1}]]", "EdgeList[CycleGraph[3]]"),
    ("HighlightGraph[CycleGraph[3], {9}]", Raises(ValueError)),
    ("HighlightGraph[CycleGraph[3], Style[1, Red]]", Raises(NotImplementedError)),
    ("NearestNeighborGraph[{1, 2, 5, 11}]",
     "Graph[{1, 2, 5, 11}, {UndirectedEdge[1, 2], UndirectedEdge[2, 5], UndirectedEdge[5, 11]}]"),
    ("EdgeCount[NearestNeighborGraph[{{0, 0}, {0, 1}, {5, 5}, {5, 7}}]]", 2),
    ("EdgeCount[NearestNeighborGraph[{0, 1, 3, 7}, {1, 1.5}]]", 1),
    ("NearestNeighborGraph[{1, \"a\"}]", Raises(NotImplementedError)),
    ("TransitiveClosureGraph[Graph[{1 -> 2, 2 -> 3}]]",
     "Graph[{1, 2, 3}, {DirectedEdge[1, 2], DirectedEdge[1, 3], DirectedEdge[2, 3]}]"),
    ("TransitiveClosureGraph[Graph[{1 <-> 2, 2 <-> 3, 4 <-> 5}]]", counts(5, 4)),
    ("TransitiveClosureGraph[TransitiveClosureGraph[Graph[{1 -> 2, 2 -> 3, 3 -> 4}]]] === "
     "TransitiveClosureGraph[Graph[{1 -> 2, 2 -> 3, 3 -> 4}]]", True),
    ("TransitiveClosureGraph[Graph[{1 -> 2, 2 -> 1}]]", Raises(NotImplementedError)),
    ("EdgeList[TransitiveReductionGraph[Graph[{1 -> 2, 2 -> 3, 1 -> 3}]]]", "{DirectedEdge[1, 2], DirectedEdge[2, 3]}"),
    ("EdgeCount[TransitiveReductionGraph[TransitiveClosureGraph[PathGraph[Range[5], DirectedEdges -> True]]]]", 4),
    ("EdgeCount[TransitiveReductionGraph[Graph[{1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4, 1 -> 4}]]]", 4),
    ("TransitiveReductionGraph[CycleGraph[3]]", Raises(NotImplementedError)),
    ("PlanarGraph[CompleteGraph[4]]", "CompleteGraph[4]"),
    ("PlanarGraph[{1 <-> 2, 2 <-> 3}]", "Graph[{1 <-> 2, 2 <-> 3}]"),
    ("PlanarGraph[CompleteGraph[5]]", Raises(ValueError)),
    ("PlanarGraph[CompleteGraph[{3, 3}]]", Raises(ValueError)),
    ("CayleyGraph[CyclicGroup[4]]", counts(4, 4)),
    ("CayleyGraph[SymmetricGroup[3]]", counts(6, 12)),
    ("VertexTransitiveGraphQ[UndirectedGraph[CayleyGraph[DihedralGroup[5]]]]", True),
    ("CayleyGraph[7]", Raises(TypeError)),
]


# ═══════════════════ measures and centralities ═══════════════════
MEASURES = [
    # EccentricityCentrality = 1/VertexEccentricity (Wolfram docs)
    ("EccentricityCentrality[PathGraph[{1, 2, 3}]]", "{1/2, 1, 1/2}"),
    ("EccentricityCentrality[CycleGraph[4]]", "{1/2, 1/2, 1/2, 1/2}"),
    ("EccentricityCentrality[PetersenGraph[]] == 1/(VertexEccentricity[PetersenGraph[], #] & /@ Range[10])", True),
    ("EccentricityCentrality[Graph[{1 <-> 2, 3 <-> 4}]]", Raises(NotImplementedError)),
    ("EccentricityCentrality[Graph[{1}, {}]]", Raises(ValueError)),
    ("EccentricityCentrality[4]", Raises(TypeError)),
    # EdgeBetweennessCentrality: path P4 gives 3, 4, 3 (count of pairs separated by each edge)
    ("EdgeBetweennessCentrality[PathGraph[{1, 2, 3, 4}]]", NearList([3, 4, 3])),
    ("EdgeBetweennessCentrality[StarGraph[4]]", NearList([3, 3, 3])),
    ("EdgeBetweennessCentrality[Graph[{1 <-> 2}, EdgeWeight -> {2}]]", Raises(NotImplementedError)),
    # GraphAssortativity: stars are perfectly disassortative (-1)
    ("GraphAssortativity[StarGraph[5]]", Near(-1)),
    ("GraphAssortativity[CycleGraph[5]]", Raises(ValueError)),
    ("GraphAssortativity[Graph[{1 -> 2, 2 -> 3}]]", Raises(NotImplementedError)),
    # GraphHub
    ("GraphHub[StarGraph[5]]", "{1}"),
    ("GraphHub[CycleGraph[4]]", "{1, 2, 3, 4}"),
    ("GraphHub[Graph[{1 -> 2, 3 -> 2, 2 -> 4}], In]", "{2}"),
    ("GraphHub[Graph[{1 -> 2, 1 -> 3, 2 -> 3}], Out]", "{1}"),
    ("GraphHub[\"g\"]", Raises(TypeError)),
    # MeanGraphDistance: P3 has ordered distances 1, 2, 1, 1, 2, 1 -> 4/3
    ("MeanGraphDistance[PathGraph[{1, 2, 3}]]", "4/3"),
    ("MeanGraphDistance[CompleteGraph[5]]", 1),
    ("MeanGraphDistance[Graph[{1 <-> 2, 3 <-> 4}]]", "Infinity"),
    ("MeanGraphDistance[PetersenGraph[]] == Total[Flatten[GraphDistanceMatrix[PetersenGraph[]]]]/90", True),
    ("MeanGraphDistance[Graph[{1}, {}]]", Raises(ValueError)),
    # GraphLinkEfficiency = 1 - MeanGraphDistance/EdgeCount (Wolfram docs); a single edge gives 0
    ("GraphLinkEfficiency[PathGraph[{1, 2, 3}]]", "1/3"),
    ("GraphLinkEfficiency[CompleteGraph[4]]", "5/6"),
    ("GraphLinkEfficiency[PathGraph[{1, 2}]]", 0),
    ("GraphLinkEfficiency[Graph[{1 <-> 2, 3 <-> 4}]]", "-Infinity"),
    ("GraphLinkEfficiency[Graph[{1 -> 2}]]", Raises(NotImplementedError)),
    # GraphReciprocity
    ("GraphReciprocity[Graph[{1 -> 2, 2 -> 1}]]", 1),
    ("GraphReciprocity[Graph[{1 -> 2, 2 -> 3}]]", 0),
    ("GraphReciprocity[Graph[{1 -> 2, 2 -> 1, 2 -> 3}]]", "2/3"),
    ("GraphReciprocity[CycleGraph[3]]", Raises(NotImplementedError)),
    ("GraphReciprocity[Graph[{1 -> 1}]]", Raises(ValueError)),
    # GraphTriangleCount: K_n has C(n, 3) triangles, the Petersen graph none (girth 5)
    ("GraphTriangleCount[CompleteGraph[4]]", 4),
    ("GraphTriangleCount[CompleteGraph[6]]", 20),
    ("GraphTriangleCount[PetersenGraph[]]", 0),
    ("GraphTriangleCount[3]", Raises(TypeError)),
    # KatzCentrality: x = alpha A^T x + beta; on a k-regular graph x = beta/(1 - k alpha)
    ("KatzCentrality[CycleGraph[4], 1/4]", "{2, 2, 2, 2}"),
    ("KatzCentrality[PathGraph[{1, 2}], 1/2]", "{2, 2}"),
    ("KatzCentrality[Graph[{1 -> 2}], 1/2]", "{1, 3/2}"),
    ("KatzCentrality[PetersenGraph[], 1/6, 2]", "Table[4, 10]"),
    ("KatzCentrality[CycleGraph[4], 1/2]", Raises(ValueError)),
    ("KatzCentrality[CycleGraph[4], w24al]", "KatzCentrality[CycleGraph[4], w24al]"),
    # MeanNeighborDegree: star centre sees leaves of degree 1, leaves see the centre of degree n - 1
    ("MeanNeighborDegree[StarGraph[4]]", "{1, 3, 3, 3}"),
    ("MeanNeighborDegree[PetersenGraph[]]", "Table[3, 10]"),
    ("MeanNeighborDegree[PathGraph[{1, 2, 3}]]", "{2, 1, 2}"),
    ("MeanNeighborDegree[Graph[{1, 2, 3}, {1 <-> 2}]]", Raises(NotImplementedError)),
    # vertex similarities (hand-counted neighbourhoods; Petersen N(1) = {2, 5, 6}, N(3) = {2, 4, 8})
    ("VertexJaccardSimilarity[CycleGraph[4], 1, 3]", 1),
    ("VertexJaccardSimilarity[CycleGraph[4], 1, 2]", 0),
    ("VertexJaccardSimilarity[PetersenGraph[], 1, 3]", "1/5"),
    ("VertexJaccardSimilarity[Graph[{1, 2}, {}], 1, 2]", Raises(ValueError)),
    ("VertexJaccardSimilarity[Graph[{1 -> 2}], 1, 2]", Raises(NotImplementedError)),
    ("VertexDiceSimilarity[PetersenGraph[], 1, 3]", "1/3"),
    ("VertexDiceSimilarity[CycleGraph[4], 1, 3]", 1),
    ("VertexDiceSimilarity[PetersenGraph[], 1, 3] == 2 #/(1 + #) &[VertexJaccardSimilarity[PetersenGraph[], 1, 3]]",
     True),
    ("VertexDiceSimilarity[Graph[{1, 2}, {}], 1, 2]", Raises(ValueError)),
    ("VertexCosineSimilarity[PetersenGraph[], 1, 3]", "1/3"),
    ("VertexCosineSimilarity[StarGraph[3], 2, 3]", 1),
    ("VertexCosineSimilarity[PathGraph[{1, 2, 3, 4}], 1, 3]", "1/Sqrt[2]"),
    ("VertexCosineSimilarity[Graph[{1, 2, 3}, {1 <-> 2}], 1, 3]", Raises(ValueError)),
    ("VertexCorrelationSimilarity[CycleGraph[4], 1, 3]", 1),
    ("VertexCorrelationSimilarity[CycleGraph[4], 1, 2]", -1),
    ("VertexCorrelationSimilarity[PetersenGraph[], 1, 3] == "
     "Correlation[AdjacencyMatrix[PetersenGraph[]][[1]] // Normal, AdjacencyMatrix[PetersenGraph[]][[3]] // Normal]",
     True),
    ("VertexCorrelationSimilarity[CompleteGraph[{1, 3}], 1, 2]", "-1"),
    ("VertexCorrelationSimilarity[Graph[{1, 2, 3}, {1 <-> 2}], 3, 1]", Raises(ValueError)),
    # connectivity (Whitney/Menger): Petersen 3, K_n n - 1, C_6 between opposite vertices 2
    ("VertexConnectivity[PetersenGraph[]]", 3),
    ("VertexConnectivity[CompleteGraph[5]]", 4),
    ("VertexConnectivity[CycleGraph[6], 1, 4]", 2),
    ("VertexConnectivity[Graph[{1 <-> 2, 3 <-> 4}]]", 0),
    ("VertexConnectivity[CycleGraph[6], 1, 2]", Raises(NotImplementedError)),
    ("VertexConnectivity[CycleGraph[6], 1, 9]", Raises(ValueError)),
    ("KEdgeConnectedGraphQ[PetersenGraph[], 3]", True),
    ("KEdgeConnectedGraphQ[PetersenGraph[], 4]", False),
    ("KEdgeConnectedGraphQ[5, 1]", False),
    ("KEdgeConnectedGraphQ[CycleGraph[4], -1]", Raises(ValueError)),
    ("KVertexConnectedGraphQ[HararyGraph[3, 8], 3]", True),
    ("KVertexConnectedGraphQ[HararyGraph[3, 8], 4]", False),
    ("KVertexConnectedGraphQ[PathGraph[{1, 2, 3}], 0]", True),
    ("KVertexConnectedGraphQ[w24x, 2]", "KVertexConnectedGraphQ[w24x, 2]"),
    ("PathGraphQ[PathGraph[Range[5]]]", True),
    ("PathGraphQ[CycleGraph[4]]", False),
    ("PathGraphQ[StarGraph[4]]", False),
    ("PathGraphQ[Graph[{1 -> 2, 2 -> 3}]]", True),
    ("PathGraphQ[5]", False),
    ("PathGraphQ[w24x]", "PathGraphQ[w24x]"),
    ("EdgeCoverQ[PathGraph[{1, 2, 3}], {1 <-> 2, 2 <-> 3}]", True),
    ("EdgeCoverQ[PathGraph[{1, 2, 3}], {1 <-> 2}]", False),
    ("EdgeCoverQ[PathGraph[{1, 2, 3}], {1 <-> 3}]", False),
    ("EdgeCoverQ[PathGraph[{1, 2, 3}], 1 <-> 2]", Raises(TypeError)),
    ("IndependentEdgeSetQ[PathGraph[{1, 2, 3, 4}], {1 <-> 2, 3 <-> 4}]", True),
    ("IndependentEdgeSetQ[PathGraph[{1, 2, 3, 4}], {1 <-> 2, 2 <-> 3}]", False),
    ("IndependentEdgeSetQ[CycleGraph[4], {}]", True),
    ("IndependentEdgeSetQ[CycleGraph[4], {1}]", Raises(TypeError)),
    ("KCoreComponents[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4}], 2]", "{{1, 2, 3}}"),
    ("KCoreComponents[PetersenGraph[], 3]", "{Range[10]}"),
    ("KCoreComponents[PetersenGraph[], 4]", "{}"),
    ("KCoreComponents[Graph[{1 <-> 2, 3 <-> 4}], 1]", "{{1, 2}, {3, 4}}"),
    ("KCoreComponents[PetersenGraph[], 2, In]", Raises(NotImplementedError)),
    ("KCoreComponents[PetersenGraph[], w24k]", "KCoreComponents[PetersenGraph[], w24k]"),
]

MEASURES_NX = []
if HAVE_NX:
    import networkx as nx

    def _nxg(edges):
        X = nx.Graph()
        X.add_edges_from(edges)
        return X
    _pet = _nxg(PETERSEN_EDGES)
    _tri = _nxg([(1, 2), (2, 3), (3, 1), (3, 4)])
    _ebc = nx.edge_betweenness_centrality(_tri, normalized=False)
    _tri_e = [(1, 2), (2, 3), (1, 3), (3, 4)]
    MEASURES_NX = [
        # networkx.edge_betweenness_centrality (unnormalised) on the triangle-with-pendant graph
        ("EdgeBetweennessCentrality[" + TRI_P + "]",
         NearList([_ebc.get(e, _ebc.get(e[::-1])) for e in _tri_e])),
        ("GraphAssortativity[" + TRI_P + "]", Near(nx.degree_assortativity_coefficient(_tri))),
        ("GraphAssortativity[" + gsrc([(1, 2), (2, 3), (3, 4), (4, 5), (2, 6)]) + "]",
         Near(nx.degree_assortativity_coefficient(_nxg([(1, 2), (2, 3), (3, 4), (4, 5), (2, 6)])))),
        ("MeanNeighborDegree[" + TRI_P + "]",
         NearList([nx.average_neighbor_degree(_tri)[v] for v in (1, 2, 3, 4)])),
        ("EdgeBetweennessCentrality[PetersenGraph[]]",
         NearList([nx.edge_betweenness_centrality(_pet, normalized=False).get(e) or
                   nx.edge_betweenness_centrality(_pet, normalized=False)[e[::-1]] for e in PETERSEN_EDGES])),
    ]


# ═══════════════════ colouring (classical chromatic numbers / indices) ═══════════════════
COLOURING = [
    ("FindVertexColoring[PetersenGraph[]]", proper_coloring(PET, 3)),
    ("FindVertexColoring[CycleGraph[5]]", proper_coloring("CycleGraph[5]", 3)),
    ("FindVertexColoring[CycleGraph[4], {w24red, w24blue}]", "{w24red, w24blue, w24red, w24blue}"),
    ("FindVertexColoring[CompleteGraph[4], {1, 2, 3}]", Raises(ValueError)),
    ("FindVertexColoring[Graph[{1 <-> 1}]]", Raises(ValueError)),
    ("FindVertexColoring[CycleGraph[4], w24red]", Raises(TypeError)),
    ("FindVertexColoring[w24x]", "FindVertexColoring[w24x]"),
    ("VertexChromaticNumber[PetersenGraph[]]", 3),
    ("VertexChromaticNumber[CycleGraph[7]]", 3),
    ("VertexChromaticNumber[CompleteGraph[6]]", 6),
    ("VertexChromaticNumber[HypercubeGraph[4]]", 2),
    ("VertexChromaticNumber[Graph[{}, {}]]", 0),
    ("VertexChromaticNumber[\"g\"]", Raises(TypeError)),
    # Petersen graph is class 2 (chromatic index 4); K_n has index n - 1 (n even) or n (n odd); Koenig: bipartite = Delta
    ("FindEdgeColoring[PetersenGraph[]]", proper_edge_coloring(PET, 4)),
    ("FindEdgeColoring[CompleteGraph[4]]", proper_edge_coloring("CompleteGraph[4]", 3)),
    ("FindEdgeColoring[StarGraph[4], {w24a, w24b, w24c}]", "{w24a, w24b, w24c}"),
    ("FindEdgeColoring[Graph[{1 <-> 1}]]", Raises(ValueError)),
    ("EdgeChromaticNumber[PetersenGraph[]]", 4),
    ("EdgeChromaticNumber[CompleteGraph[5]]", 5),
    ("EdgeChromaticNumber[CompleteGraph[6]]", 5),
    ("EdgeChromaticNumber[CompleteGraph[{3, 4}]]", 4),
    ("EdgeChromaticNumber[Graph[{1 <-> 2, 1 <-> 2, 2 <-> 3}]]", 3),
    ("EdgeChromaticNumber[4]", Raises(TypeError)),
]

# ═══════════════════ k-cliques, clans, clubs, plexes (definitions: Mokken 1979, Seidman-Foster 1978) ═══════════
KSETS = [
    ("FindKClique[PetersenGraph[], 1]", P(lambda r: len(py(r)) == 1 and len(py(r)[0]) == 2, "an edge (clique number 2)")),
    ("FindKClique[PathGraph[Range[6]], 2]", P(lambda r: len(py(r)[0]) == 3, "3 consecutive vertices")),
    ("FindKClique[PetersenGraph[], 2]", "{Range[10]}"),  # diameter 2
    ("FindKClique[CycleGraph[6], 1, {2}, All]",
     P(lambda r: len(py(r)) == 6 and all(len(c) == 2 for c in py(r)), "all 6 edges")),
    ("FindKClique[{CompleteGraph[4], 3}, 1]", P(lambda r: 3 in py(r)[0] and len(py(r)[0]) == 4, "contains 3")),
    ("FindKClique[CycleGraph[5], 0]", Raises(ValueError)),
    ("FindKClique[Graph[{1 -> 2}], 1]", Raises(NotImplementedError)),
    ("FindKClique[CycleGraph[5], w24k]", "FindKClique[CycleGraph[5], w24k]"),
    ("FindKClan[PathGraph[Range[5]], 2]", P(lambda r: len(py(r)[0]) == 3, "a 3-vertex 2-clan")),
    # classic example: the 2-clique {1..5} of the 6-cycle-with-hub graph is not a 2-clan
    ("FindKClan[CycleGraph[5], 2]", "{Range[5]}"),
    ("FindKClan[PetersenGraph[], 2]", "{Range[10]}"),
    ("FindKClan[CycleGraph[5], 1.5]", Raises(ValueError)),
    ("FindKClub[StarGraph[5], 2]", "{Range[5]}"),
    ("FindKClub[PathGraph[Range[5]], 2]", P(lambda r: len(py(r)[0]) == 3, "3 consecutive vertices")),
    ("FindKClub[CycleGraph[6], 3]", "{Range[6]}"),
    ("FindKClub[3, 2]", Raises(TypeError)),
    ("FindKPlex[CycleGraph[4], 2]", "{{1, 2, 3, 4}}"),
    ("FindKPlex[PetersenGraph[], 1]", P(lambda r: len(py(r)[0]) == 2, "1-plex = clique")),
    ("FindKPlex[CycleGraph[5], 2]", P(lambda r: len(py(r)[0]) == 3, "C5 has 2-plexes of size 3")),
    ("FindKPlex[CycleGraph[5], 2, {3}, All]", P(lambda r: len(py(r)) == 5, "five consecutive triples")),
    ("FindKPlex[CycleGraph[5], 2, \"x\"]", Raises(TypeError)),
]

# ═══════════════════ matchings, covers, cuts, paths, flows, tours ═══════════════════
FLOWS = [
    # matching numbers: P5 -> 2, Petersen -> 5 (perfect), K_{3,4} -> 3
    ("FindIndependentEdgeSet[PathGraph[{1, 2, 3, 4, 5}]]", matching_of("PathGraph[{1, 2, 3, 4, 5}]", 2)),
    ("FindIndependentEdgeSet[PetersenGraph[]]", matching_of(PET, 5)),
    ("Length[FindIndependentEdgeSet[CompleteGraph[{3, 4}]]]", 3),
    ("IndependentEdgeSetQ[CycleGraph[6], FindIndependentEdgeSet[CycleGraph[6]]]", True),
    ("FindIndependentEdgeSet[6]", Raises(TypeError)),
    # Gallai: minimum edge cover = n - matching number
    ("FindEdgeCover[PathGraph[{1, 2, 3, 4, 5}]]", edge_cover_of("PathGraph[{1, 2, 3, 4, 5}]", 3)),
    ("FindEdgeCover[PetersenGraph[]]", edge_cover_of(PET, 5)),
    ("FindEdgeCover[StarGraph[5]]", edge_cover_of("StarGraph[5]", 4)),
    ("EdgeCoverQ[CycleGraph[7], FindEdgeCover[CycleGraph[7]]]", True),
    ("FindEdgeCover[Graph[{1, 2, 3}, {1 <-> 2}]]", Raises(ValueError)),
    # edge cuts: lambda(C5) = 2, lambda(Petersen) = 3; directed cycle 1
    ("FindEdgeCut[CycleGraph[5]]", disconnects_edges("CycleGraph[5]", 2)),
    ("FindEdgeCut[PetersenGraph[]]", disconnects_edges(PET, 3)),
    ("FindEdgeCut[PathGraph[{1, 2, 3}], 1, 3]", P(lambda r: len(py(r)) == 1, "one edge")),
    ("Length[FindEdgeCut[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]]", 1),
    ("FindEdgeCut[Graph[{1 <-> 2, 3 <-> 4}]]", "{}"),
    ("FindEdgeCut[CycleGraph[4], 1, 1]", Raises(ValueError)),
    ("FindEdgeCut[CycleGraph[4], 1]", Raises(ValueError)),
    # vertex cuts: kappa(Petersen) = 3, C6 between opposite vertices 2
    ("FindVertexCut[PetersenGraph[]]", separates(PET, 3)),
    ("FindVertexCut[CycleGraph[6], 1, 4]", separates("CycleGraph[6]", 2, 1, 4)),
    ("FindVertexCut[CompleteGraph[4]]", "{}"),
    ("FindVertexCut[PathGraph[{1, 2, 3}]]", "{2}"),
    ("FindVertexCut[CycleGraph[6], 1, 2]", Raises(ValueError)),
    ("FindVertexCut[Graph[{1 -> 2, 2 -> 3}]]", Raises(NotImplementedError)),
    # Menger: number of disjoint paths = local connectivity
    ("FindEdgeIndependentPaths[PetersenGraph[], 1, 2, Infinity]", disjoint_paths(PET, 1, 2, 3, "edge")),
    ("FindEdgeIndependentPaths[CycleGraph[6], 1, 4, 1]", disjoint_paths("CycleGraph[6]", 1, 4, 1, "edge")),
    ("FindEdgeIndependentPaths[Graph[{1 -> 2, 2 -> 3, 1 -> 3}], 1, 3, 5]", "{{1, 3}, {1, 2, 3}}"),
    ("FindEdgeIndependentPaths[CycleGraph[6], 1, 1, 2]", Raises(ValueError)),
    ("FindEdgeIndependentPaths[CycleGraph[6], 1, 4, -1]", Raises(ValueError)),
    ("FindVertexIndependentPaths[PetersenGraph[], 1, 3, Infinity]", disjoint_paths(PET, 1, 3, 3, "vertex")),
    ("FindVertexIndependentPaths[CycleGraph[6], 1, 4, 5]", disjoint_paths("CycleGraph[6]", 1, 4, 2, "vertex")),
    # bowtie: two triangles sharing vertex 3 - only one vertex-disjoint path from 1 to 5
    ("Length[FindVertexIndependentPaths[" + gsrc([(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (3, 5)]) + ", 1, 5, 4]]", 1),
    ("FindVertexIndependentPaths[CycleGraph[6], 1, 9, 2]", Raises(ValueError)),
    # maximum cut: bipartite graphs cut everything, K_n cuts floor(n^2/4), C5 cuts 4
    ("First[FindMaximumCut[CycleGraph[5]]]", 4),
    ("First[FindMaximumCut[CompleteGraph[5]]]", 6),
    ("First[FindMaximumCut[HypercubeGraph[3]]]", 12),
    ("FindMaximumCut[PathGraph[{1, 2, 3}]]", "{2, {{1, 3}, {2}}}"),
    ("First[FindMaximumCut[Graph[{1 <-> 2, 2 <-> 3, 1 <-> 3}, EdgeWeight -> {5, 1, 1}]]]", 6),
    ("FindMaximumCut[GridGraph[{5, 5}]]", Raises(NotImplementedError)),
    # Chinese postman: Eulerian graphs need no repeats; a path is walked twice; Petersen: 15 + 5 repeated = 20
    ("FindPostmanTour[CycleGraph[4]]",
     "{{UndirectedEdge[1, 2], UndirectedEdge[2, 3], UndirectedEdge[3, 4], UndirectedEdge[4, 1]}}"),
    ("Length[First[FindPostmanTour[PathGraph[{1, 2, 3, 4}]]]]", 6),
    ("Length[First[FindPostmanTour[PetersenGraph[]]]]", 20),
    ("Length[First[FindPostmanTour[CompleteGraph[4]]]]", 8),
    ("FindPostmanTour[Graph[{1 <-> 2, 3 <-> 4}]]", "{}"),
    ("FindPostmanTour[CycleGraph[4], 0]", "{}"),
    ("FindPostmanTour[Graph[{1 -> 2, 2 -> 1}]]", Raises(NotImplementedError)),
    # minimum-cost flow: unit costs give d times the shortest distance; capacity-limited routing hand-computed
    ("FindMinimumCostFlow[Graph[{1 -> 2, 2 -> 4, 1 -> 3, 3 -> 4}, EdgeCapacity -> {1, 1, 2, 2}, "
     "EdgeCost -> {1, 1, 3, 3}], 1, 4]", 14),
    ("FindMinimumCostFlow[Graph[{1 -> 2, 2 -> 4, 1 -> 3, 3 -> 4}, EdgeCapacity -> {1, 1, 2, 2}, "
     "EdgeCost -> {1, 1, 3, 3}], 1, 4, \"FlowValue\"]", 3),
    ("FindMinimumCostFlow[PathGraph[{1, 2, 3}], 1, 3, 5]", 10),
    ("FindMinimumCostFlow[Graph[{1 -> 2, 2 -> 3}], {2, 0, -2}]", 4),
    ("FindMinimumCostFlow[PathGraph[{1, 2, 3}], 1, 3]", Raises(ValueError)),
    ("FindMinimumCostFlow[Graph[{1 -> 2}, EdgeCapacity -> {1}], 1, 2, 3]", Raises(ValueError)),
    ("FindMinimumCostFlow[Graph[{1 -> 2}], 1, 2, 1, \"FlowMatrix\"]", Raises(NotImplementedError)),
    ("FindMinimumCostFlow[Graph[{1 -> 2}], {1, 1}]", Raises(ValueError)),
    # fundamental cycles: m - n + c of them
    ("Length[FindFundamentalCycles[PetersenGraph[]]]", 6),
    ("FindFundamentalCycles[PathGraph[{1, 2, 3}]]", "{}"),
    ("Length /@ FindFundamentalCycles[CompleteGraph[4]]", "{3, 3, 3}"),
    ("Length[FindFundamentalCycles[Graph[{1 <-> 2, 1 <-> 2, 3 <-> 3}]]]", 2),
    ("FindFundamentalCycles[Graph[{1 -> 2, 2 -> 1}]]", Raises(NotImplementedError)),
    # dominators (Lengauer-Tarjan definitions; the diamond's join is dominated by the root)
    ("EdgeList[DominatorTreeGraph[" + DOM + ", 1]]",
     "{DirectedEdge[1, 2], DirectedEdge[1, 3], DirectedEdge[1, 4], DirectedEdge[4, 5]}"),
    ("EdgeList[DominatorTreeGraph[Graph[{1 -> 2, 2 -> 3}], 1]]", "{DirectedEdge[1, 2], DirectedEdge[2, 3]}"),
    ("VertexList[DominatorTreeGraph[Graph[{1 -> 2, 3 -> 2}], 1]]", "{1, 2}"),
    ("DominatorTreeGraph[CycleGraph[3], 1]", Raises(NotImplementedError)),
    ("DominatorTreeGraph[" + DOM + ", 9]", Raises(ValueError)),
    ("DominatorVertexList[" + DOM + ", 1]", "{1, 1, 1, 1, 4}"),
    ("DominatorVertexList[Graph[{1 -> 2, 2 -> 3, 3 -> 4}], 1]", "{1, 1, 2, 3}"),
    ("DominatorVertexList[Graph[{1 -> 2, 2 -> 3, 1 -> 3}], 1]", "{1, 1, 1}"),
    ("DominatorVertexList[Graph[{1 -> 2, 3 -> 2}], 1]", Raises(ValueError)),
    ("DominatorVertexList[\"g\", 1]", Raises(TypeError)),
]

# ═══════════════════ isomorphism and symmetry ═══════════════════
SYMMETRY = [
    ("CanonicalGraph[Graph[{1 <-> 3, 3 <-> 2, 2 <-> 4, 4 <-> 1}]] === CanonicalGraph[CycleGraph[4]]", True),
    ("CanonicalGraph[VertexReplace[PetersenGraph[], Thread[Range[10] -> {7, 3, 9, 1, 10, 2, 8, 4, 6, 5}]]] === "
     "CanonicalGraph[PetersenGraph[]]", True),
    # same degree sequence, not isomorphic: C6 versus two triangles
    ("CanonicalGraph[CycleGraph[6]] === CanonicalGraph[GraphDisjointUnion[CycleGraph[3], CycleGraph[3]]]", False),
    ("IsomorphicGraphQ[CanonicalGraph[PetersenGraph[]], PetersenGraph[]]", True),
    ("CanonicalGraph[Graph[{1 -> 2, 2 -> 3}]] === CanonicalGraph[Graph[{3 -> 1, 2 -> 3}]]", True),
    ("CanonicalGraph[Graph[{1 <-> 2, 1 <-> 2}]]", Raises(NotImplementedError)),
    ("CanonicalGraph[1]", Raises(TypeError)),
    ("GroupOrder[GraphAutomorphismGroup[PetersenGraph[]]]", 120),
    ("GroupOrder[GraphAutomorphismGroup[CycleGraph[6]]]", 12),
    ("GroupOrder[GraphAutomorphismGroup[CompleteGraph[5]]]", 120),
    ("GraphAutomorphismGroup[PathGraph[{1, 2, 3}]]", "PermutationGroup[{Cycles[{{1, 3}}]}]"),
    ("GroupOrder[GraphAutomorphismGroup[BuckyballGraph[]]]", 120),
    ("GroupOrder[GraphAutomorphismGroup[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]]", 3),
    ("GraphAutomorphismGroup[Graph[{1 <-> 2, 1 <-> 2}]]", Raises(NotImplementedError)),
    ("VertexTransitiveGraphQ[PetersenGraph[]]", True),
    ("VertexTransitiveGraphQ[HypercubeGraph[3]]", True),
    ("VertexTransitiveGraphQ[StarGraph[4]]", False),
    ("VertexTransitiveGraphQ[PathGraph[{1, 2, 3}]]", False),
    ("VertexTransitiveGraphQ[5]", False),
    ("VertexTransitiveGraphQ[w24x]", "VertexTransitiveGraphQ[w24x]"),
    ("EdgeTransitiveGraphQ[StarGraph[4]]", True),
    ("EdgeTransitiveGraphQ[CompleteGraph[{2, 3}]]", True),
    ("EdgeTransitiveGraphQ[PathGraph[{1, 2, 3, 4}]]", False),
    ("EdgeTransitiveGraphQ[PetersenGraph[]]", True),
    ("EdgeTransitiveGraphQ[\"g\"]", False),
    ("EdgeTransitiveGraphQ[Graph[{1 <-> 2, 1 <-> 2}]]", Raises(NotImplementedError)),
    ("IsomorphicSubgraphQ[CycleGraph[5], PetersenGraph[]]", True),
    ("IsomorphicSubgraphQ[CompleteGraph[4], PetersenGraph[]]", False),
    ("IsomorphicSubgraphQ[PathGraph[{1, 2, 3}], PathGraph[{1, 2, 3, 4}]]", True),
    ("IsomorphicSubgraphQ[CycleGraph[4], 4]", Raises(TypeError)),
    # C4 -> K4: 4!/1 injective maps all preserve adjacency (every pair adjacent in K4): 24
    ("Length[FindSubgraphIsomorphism[CycleGraph[4], CompleteGraph[4], All]]", 24),
    ("FindSubgraphIsomorphism[PathGraph[{w24a, w24b}], CycleGraph[3]]", "{<|w24a -> 1, w24b -> 2|>}"),
    ("FindSubgraphIsomorphism[CompleteGraph[4], PetersenGraph[]]", "{}"),
    ("Length[FindSubgraphIsomorphism[CycleGraph[5], PetersenGraph[], 3]]", 3),
    ("FindSubgraphIsomorphism[CycleGraph[4], CompleteGraph[4], -1]", Raises(ValueError)),
    ("Length[FindIsomorphicSubgraph[CompleteGraph[4], CycleGraph[3], All]]", 4),
    ("IsomorphicGraphQ[First[FindIsomorphicSubgraph[PetersenGraph[], CycleGraph[5]]], CycleGraph[5]]", True),
    ("FindIsomorphicSubgraph[CycleGraph[5], CycleGraph[3]]", "{}"),
    ("Length[FindIsomorphicSubgraph[CompleteGraph[4], PathGraph[{1, 2}], All]]", 6),
    ("FindIsomorphicSubgraph[CycleGraph[5], 3]", Raises(TypeError)),
]

# ═══════════════════ Tutte and flow polynomials ═══════════════════
# T(C_n) = x^(n-1) + ... + x + y; T(tree) = x^(n-1); T(K4) = x^3 + 3x^2 + 2x + 4xy + 2y + 3y^2 + y^3 (Bollobas X.2)
POLYS = [
    ("TuttePolynomial[CycleGraph[4], {w24x, w24y}]", "w24x + w24x^2 + w24x^3 + w24y"),
    ("TuttePolynomial[CompleteGraph[4], {w24x, w24y}]",
     "w24x^3 + 3 w24x^2 + 2 w24x + 4 w24x w24y + 2 w24y + 3 w24y^2 + w24y^3"),
    ("TuttePolynomial[PathGraph[{1, 2, 3}], {w24x, w24y}]", "w24x^2"),
    ("TuttePolynomial[Graph[{1 <-> 2, 1 <-> 2}], {w24x, w24y}]", "w24x + w24y"),
    ("TuttePolynomial[PetersenGraph[], {1, 1}]", 2000),     # spanning trees of the Petersen graph
    ("TuttePolynomial[CompleteGraph[4], {2, 2}]", 64),       # T(2, 2) = 2^m
    ("TuttePolynomial[Graph[{1 -> 2}], {w24x, w24y}]", Raises(NotImplementedError)),
    ("TuttePolynomial[CycleGraph[3], w24x]", Raises(TypeError)),
    ("TuttePolynomial[w24g, {w24x, w24y}]", "TuttePolynomial[w24g, {w24x, w24y}]"),
    ("FlowPolynomial[CycleGraph[5], w24k]", "w24k - 1"),
    ("FlowPolynomial[PathGraph[{1, 2}], w24k]", 0),
    ("Factor[FlowPolynomial[CompleteGraph[4], w24k]]", "(w24k - 1) (w24k - 2) (w24k - 3)"),
    ("FlowPolynomial[PetersenGraph[], 4]", 0),                # Petersen has no nowhere-zero 4-flow
    ("FlowPolynomial[PetersenGraph[], 5] > 0", True),
    ("FlowPolynomial[Graph[{1 -> 2, 2 -> 1}], 3]", Raises(NotImplementedError)),
]

# ═══════════════════ planar faces and duals (Euler: F = m - n + 2) ═══════════════════
PLANAR = [
    ("Length /@ PlanarFaceList[HypercubeGraph[3]]", "{4, 4, 4, 4, 4, 4}"),
    ("Length[PlanarFaceList[CompleteGraph[4]]]", 4),
    ("Sort[Sort /@ PlanarFaceList[CompleteGraph[4]]]", "{{1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4}}"),
    ("PlanarFaceList[PathGraph[{1, 2, 3}]]", "{{1, 2, 3, 2}}"),
    ("Length[PlanarFaceList[GridGraph[{3, 3}]]]", 5),
    ("PlanarFaceList[CompleteGraph[5]]", Raises(ValueError)),
    ("PlanarFaceList[Graph[{1 <-> 2, 3 <-> 4}]]", Raises(NotImplementedError)),
    ("DualPlanarGraph[HypercubeGraph[3]]", iso("CompleteGraph[{2, 2, 2}]")),
    ("DualPlanarGraph[CompleteGraph[4]]", iso("CompleteGraph[4]")),
    ("EdgeCount[DualPlanarGraph[GridGraph[{3, 4}]]]", 17),
    ("DualPlanarGraph[CompleteGraph[{3, 3}]]", Raises(ValueError)),
    ("DualPlanarGraph[5]", Raises(TypeError)),
]

# ═══════════════════ ShortestPathFunction ═══════════════════
SPF = [
    ("FindShortestPath[PathGraph[{1, 2, 3, 4}], 1, All][4]", "{1, 2, 3, 4}"),
    ("FindShortestPath[CycleGraph[5], All, All][1, 3]", "{1, 2, 3}"),
    ("FindShortestPath[PathGraph[{1, 2, 3}], All, 1][3]", "{3, 2, 1}"),
    ("Table[FindShortestPath[PetersenGraph[], 1, All][w24t] == FindShortestPath[PetersenGraph[], 1, w24t], "
     "{w24t, 10}] // Union", "{True}"),
    ("Head[FindShortestPath[CycleGraph[5], 1, All]]", "ShortestPathFunction"),
    ("FindShortestPath[Graph[{1 -> 2, 3 -> 4}], 1, All][4]", "{}"),
    ("ShortestPathFunction[\"Bad\", {}]", Raises(ValueError)),
    ("ShortestPathFunction[\"SingleSource\", {PathGraph[{1, 2, 3}], 1}][3]", "{1, 2, 3}"),
    ("Length[ShortestPathFunction[\"AllPairs\", {CycleGraph[4]}][1, 3]]", 3),
    ("ShortestPathFunction[\"SingleTarget\", {PathGraph[{1, 2, 3}], 3}][1]", "{1, 2, 3}"),
    ("ShortestPathFunction[\"SingleSource\", {PathGraph[{1, 2, 3}], 1}]",
     "ShortestPathFunction[\"SingleSource\", {PathGraph[{1, 2, 3}], 1}]"),
    ("FindShortestPath[CycleGraph[5], 9, All]", Raises(ValueError)),
    ("FindShortestPath[CycleGraph[5], 1, All][1, 2]", Raises(ValueError)),
]

# ═══════════════════ random graph distributions (structural invariants of each model) ═══════════════════
RANDOM = [
    ("RandomGraph[BernoulliGraphDistribution[5, 1]]", iso("CompleteGraph[5]")),
    ("EdgeCount[RandomGraph[BernoulliGraphDistribution[6, 0]]]", 0),
    ("Length[RandomGraph[BernoulliGraphDistribution[4, 1/2], 3]]", 3),
    ("BernoulliGraphDistribution[5, 2]", Raises(ValueError)),
    ("BernoulliGraphDistribution[w24n, 1/2]", "BernoulliGraphDistribution[w24n, 1/2]"),
    ("BernoulliGraphDistribution[\"a\", 1/2]", Raises(TypeError)),
    ("EdgeCount[RandomGraph[UniformGraphDistribution[6, 7]]]", 7),
    ("SimpleGraphQ[RandomGraph[UniformGraphDistribution[8, 20]]]", True),
    ("VertexCount[RandomGraph[UniformGraphDistribution[5, 0]]]", 5),
    ("UniformGraphDistribution[4, 7]", Raises(ValueError)),
    ("EdgeCount[RandomGraph[BarabasiAlbertGraphDistribution[10, 2]]]", 17),   # C(3, 2) + 7 * 2
    ("ConnectedGraphQ[RandomGraph[BarabasiAlbertGraphDistribution[30, 1]]]", True),
    ("VertexCount[RandomGraph[BarabasiAlbertGraphDistribution[12, 3]]]", 12),
    ("BarabasiAlbertGraphDistribution[3, 3]", Raises(ValueError)),
    ("RandomGraph[WattsStrogatzGraphDistribution[8, 0]]", iso("CirculantGraph[8, {1, 2}]")),
    ("EdgeCount[RandomGraph[WattsStrogatzGraphDistribution[20, 0.3, 3]]]", 60),
    ("SimpleGraphQ[RandomGraph[WattsStrogatzGraphDistribution[15, 1, 2]]]", True),
    ("WattsStrogatzGraphDistribution[4, 0.1, 2]", Raises(ValueError)),
    ("AcyclicGraphQ[RandomGraph[PriceGraphDistribution[10, 2, 1]]]", True),
    ("VertexOutDegree[RandomGraph[PriceGraphDistribution[6, 2, 1]]]", "{0, 1, 2, 2, 2, 2}"),
    ("EdgeCount[RandomGraph[PriceGraphDistribution[5, 1, 1/2]]]", 4),
    ("PriceGraphDistribution[5, 1, 0]", Raises(ValueError)),
    ("EdgeCount[RandomGraph[SpatialGraphDistribution[6, 2]]]", 15),     # r >= Sqrt[2] joins everything
    ("EdgeCount[RandomGraph[SpatialGraphDistribution[6, 0]]]", 0),
    ("EdgeCount[RandomGraph[SpatialGraphDistribution[5, 2, 3]]]", 10),  # r >= Sqrt[3] in the unit cube
    ("SpatialGraphDistribution[5, 0.2, NormalDistribution[]]", Raises(NotImplementedError)),
    ("VertexDegree[RandomGraph[DegreeGraphDistribution[{3, 3, 2, 2, 2, 1, 1}]]]", "{3, 3, 2, 2, 2, 1, 1}"),
    ("VertexDegree[RandomGraph[DegreeGraphDistribution[{2, 2, 2, 2, 2}]]]", "{2, 2, 2, 2, 2}"),
    ("SimpleGraphQ[RandomGraph[DegreeGraphDistribution[{4, 3, 3, 2, 2, 2}]]]", True),
    ("DegreeGraphDistribution[{3, 3}]", Raises(ValueError)),
    ("RandomGraph[{5, 3}] // EdgeCount", 3),
    ("RandomGraph[BernoulliGraphDistribution[3, 1], -1]", Raises(ValueError)),
]

# ── W24 resume: radiality, degree connectivity, link rank, k-components, face colouring, cycle matrix ──
_DG = "{1 -> 3, 2 -> 1, 3 -> 6, 4 -> 6, 1 -> 5, 5 -> 4, 6 -> 1}"   # Wolfram doc example graph (VertexList 1,3,2,6,4,5)


def _setlist(want):
    """Components compared as a set of vertex sets (Wolfram's component order is not documented)."""
    w = sorted(sorted(c) for c in want)
    return Pred(lambda g: sorted(sorted(py(c)) for c in g.args) == w, f"components {w}")


def _proper_faces(src, k):
    """Face colouring with exactly k colours, proper on the faces of PlanarFaceList (edge-sharing faces differ)."""
    def ok(g):
        cols = py(g)
        faces = py(ev(f"PlanarFaceList[{src}]"))
        if len(cols) != len(faces) or len(set(cols)) != k:
            return False
        sides = {}
        for fi, f in enumerate(faces):
            for t in range(len(f)):
                sides.setdefault(frozenset((f[t], f[(t + 1) % len(f)])), []).append(fi)
        return all(cols[a] != cols[b] for fs in sides.values() for a in fs for b in fs if a != b)
    return Pred(ok, f"proper face colouring with {k} colours")


def _cycle_basis(src, rows):
    """EdgeCycleMatrix rows: the right count, each a closed cycle (+-1 net balance for directed edges), full rank."""
    def ok(g):
        m = py(g)
        el = py(ev(f"EdgeList[{src}]"))
        if len(m) != rows or any(len(r) != len(el) for r in m):
            return False
        for r in m:
            bal = {}
            for x, (kind, a, b) in zip(r, el):
                if x == 0:
                    continue
                if kind == "D":
                    bal[a] = bal.get(a, 0) + x
                    bal[b] = bal.get(b, 0) - x
                else:
                    bal[a] = bal.get(a, 0) + 1
                    bal[b] = bal.get(b, 0) + 1
            if any(v % 2 for v in bal.values()):
                return False
        return not m or sp.Matrix(m).rank() == rows
    return Pred(ok, f"cycle basis with {rows} rows")


W24B = [
    # RadialityCentrality: Wolfram documentation values for the rules example (Out is the default) and "In"
    (f"RadialityCentrality[{_DG}]", NearList([0.7, 0.5, 0.7, 0.6, 0.55, 0.5])),
    (f"RadialityCentrality[{_DG}, \"Out\"]", NearList([0.7, 0.5, 0.7, 0.6, 0.55, 0.5])),
    (f"RadialityCentrality[{_DG}, \"In\"]", NearList([0.8, 0.65, 0.0, 0.8, 0.6, 0.7])),
    # doc identity: Mean[Drop[(diam + 1 - GraphDistanceMatrix)[[i]], {i}]]/diam; star K_{1,3}: {1, 2/3, 2/3, 2/3}
    ("RadialityCentrality[StarGraph[4]]", NearList([1, 2 / 3, 2 / 3, 2 / 3])),
    ("RadialityCentrality[CycleGraph[6]]", NearList([11 / 15] * 6)),   # (3 + 3 + 2 + 2 + 1)/(3 * 5), diameter 3
    ("With[{w = PetersenGraph[]}, RadialityCentrality[w] == Table[Mean[Drop[(GraphDiameter[w] + 1 - "
     "GraphDistanceMatrix[w])[[i]], {i}]]/GraphDiameter[w], {i, 10}]]", True),   # Check: documented identity
    ("RadialityCentrality[Graph[{1}, {}]]", NearList([0])),                         # isolated vertex: 0
    ("RadialityCentrality[Graph[{1 <-> 2, 3 <-> 4, 4 <-> 5}]]", Raises(NotImplementedError)),  # diameters differ
    ("RadialityCentrality[CycleGraph[4], \"Up\"]", Raises(ValueError)),
    ("RadialityCentrality[7]", Raises(TypeError)),
    ("RadialityCentrality[rgw24]", "RadialityCentrality[rgw24]"),
    # MeanDegreeConnectivity: Wolfram documentation (MeanNeighborDegree and MeanDegreeConnectivity pages)
    ("MeanDegreeConnectivity[GridGraph[{3, 4, 5}]]", "{0, 0, 0, 4, 17/4, 262/55, 97/18}"),
    ("MeanDegreeConnectivity[CompleteGraph[{2, 4, 3}]][[8]]", "38/7"),
    ("MeanDegreeConnectivity[CycleGraph[5]]", "{0, 0, 2}"),
    ("MeanDegreeConnectivity[StarGraph[4]]", "{0, 3, 0, 1}"),
    ("MeanDegreeConnectivity[Graph[{1, 2, 3}, {1 <-> 2}]]", "{0, 1}"),
    ("With[{w = PetersenGraph[]}, Last[MeanDegreeConnectivity[w]] == Mean[MeanNeighborDegree[w]]]", True),  # Check
    ("MeanDegreeConnectivity[Graph[{1 -> 2, 2 -> 3}]]", Raises(NotImplementedError)),
    ("MeanDegreeConnectivity[CycleGraph[4], \"In\"]", Raises(NotImplementedError)),
    ("MeanDegreeConnectivity[CycleGraph[4], \"Up\"]", Raises(ValueError)),
    ("MeanDegreeConnectivity[{1, 2}]", Raises(TypeError)),
    ("MeanDegreeConnectivity[mdw24]", "MeanDegreeConnectivity[mdw24]"),
    # LinkRankCentrality: Wolfram documentation values (rules example, alpha = 0.5)
    (f"LinkRankCentrality[{_DG}, 0.5]", NearList([0.12069, 0.0833333, 0.143678, 0.155172, 0.12069, 0.143678,
                                                  0.232759], 1e-5)),
    (f"LinkRankCentrality[{_DG}, 0.5, 2]", NearList([0.12069, 0.0833333, 0.143678, 0.155172, 0.12069, 0.143678,
                                                     0.232759], 1e-5)),   # scalar beta = uniform beta
    ("LinkRankCentrality[CycleGraph[8, DirectedEdges -> True], 1/2, WorkingPrecision -> Infinity]",
     "{1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8}"),   # symmetric: equal ranks
    ("Total[LinkRankCentrality[Graph[{1 -> 2, 2 -> 1, 2 -> 3, 3 -> 1}], 1/3, WorkingPrecision -> Infinity]]", 1),
    # alpha = 0: c = beta normalised, link rank of u -> v is beta(u)/(Total[beta] outdeg(u))
    ("LinkRankCentrality[Graph[{1 -> 2, 1 -> 3, 2 -> 1, 3 -> 1}], 0, {1, 1, 2}, WorkingPrecision -> Infinity]",
     "{1/8, 1/8, 1/4, 1/2}"),
    ("LinkRankCentrality[CycleGraph[4], 0.5]", Raises(NotImplementedError)),          # undirected
    ("LinkRankCentrality[Graph[{1 -> 2}], 0.5]", Raises(NotImplementedError)),        # dangling vertex
    ("LinkRankCentrality[Graph[{1 -> 2, 2 -> 1}], 1.5]", Raises(ValueError)),
    ("LinkRankCentrality[Graph[{1 -> 2, 2 -> 1}], 0.5, WorkingPrecision -> 30]", Raises(NotImplementedError)),
    ("LinkRankCentrality[\"g\", 0.5]", Raises(TypeError)),
    ("LinkRankCentrality[Graph[{1 -> 2, 2 -> 1}], alw24]", "LinkRankCentrality[Graph[{1, 2}, {1 -> 2, 2 -> 1}], alw24]"),
    # KEdgeConnectedComponents: Wolfram documentation (rules example, k = 1); bridges split 2-edge components
    ("KEdgeConnectedComponents[{1 -> 2, 1 -> 5, 2 -> 3, 3 -> 4, 4 -> 5, 6 -> 3, 4 -> 6}, 1]",
     _setlist([[5], [3, 4, 6], [2], [1]])),
    ("KEdgeConnectedComponents[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4, 4 <-> 5, 5 <-> 6, 6 <-> 4}], 2]",
     _setlist([[1, 2, 3], [4, 5, 6]])),
    ("KEdgeConnectedComponents[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4, 4 <-> 5, 5 <-> 6, 6 <-> 4}], 2, {5}]",
     _setlist([[4, 5, 6]])),
    ("KEdgeConnectedComponents[HypercubeGraph[3], 3]", _setlist([list(range(1, 9))])),   # Q3 is 3-edge-connected
    ("KEdgeConnectedComponents[HypercubeGraph[3], 4]", "{}"),
    ("KEdgeConnectedComponents[GraphUnion[CompleteGraph[4], Graph[{4 <-> 5}]], 3]", _setlist([[1, 2, 3, 4]])),
    ("KEdgeConnectedComponents[Graph[{1 -> 2, 2 -> 1, 2 -> 3}], 1]", _setlist([[1, 2], [3]])),
    # the two documented definitions disagree (a, b joined by three 2-paths): raise rather than guess
    ("KEdgeConnectedComponents[Graph[{a <-> x, x <-> b, a <-> y, y <-> b, a <-> z, z <-> b}], 3]",
     Raises(NotImplementedError)),
    ("KEdgeConnectedComponents[CycleGraph[4], 0]", Raises(ValueError)),
    ("KEdgeConnectedComponents[CycleGraph[4], 1, _?OddQ]", Raises(NotImplementedError)),
    ("KEdgeConnectedComponents[3, 1]", Raises(TypeError)),
    ("KEdgeConnectedComponents[CycleGraph[4], kw24]", "KEdgeConnectedComponents[CycleGraph[4], kw24]"),
    # KVertexConnectedComponents: blocks (a bridge is a 2-vertex block, as in Wolfram's {{...}, {6, 7}} example)
    ("KVertexConnectedComponents[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4, 4 <-> 5, 5 <-> 3, 5 <-> 6}], 2]",
     _setlist([[1, 2, 3], [3, 4, 5], [5, 6]])),
    ("KVertexConnectedComponents[PathGraph[{1, 2, 3, 4}], 2]", _setlist([[1, 2], [2, 3], [3, 4]])),
    ("KVertexConnectedComponents[PetersenGraph[], 2]", _setlist([list(range(1, 11))])),
    ("KVertexConnectedComponents[PathGraph[{1, 2, 3, 4}], 2, {1}]", _setlist([[1, 2]])),
    ("KVertexConnectedComponents[Graph[{1 <-> 2, 3 <-> 4}], 1]", _setlist([[1, 2], [3, 4]])),
    ("KVertexConnectedComponents[CycleGraph[5], 3]", Raises(NotImplementedError)),
    ("KVertexConnectedComponents[Graph[{1 -> 2, 2 -> 1}], 2]", Raises(NotImplementedError)),
    ("KVertexConnectedComponents[CycleGraph[5], -1]", Raises(ValueError)),
    ("KVertexConnectedComponents[\"g\", 2]", Raises(TypeError)),
    # FindPlanarColoring: faces of a wheel need 4 colours (wheels are self-dual), the cube's faces 3 (octahedron)
    ("FindPlanarColoring[WheelGraph[6]]", _proper_faces("WheelGraph[6]", 4)),
    ("FindPlanarColoring[{1 -> 2, 1 -> 3, 1 -> 4, 2 -> 3, 2 -> 4}]", "{1, 2, 3}"),   # Wolfram doc: 3 faces, 3 colours
    ("FindPlanarColoring[HypercubeGraph[3]]", _proper_faces("HypercubeGraph[3]", 3)),
    ("FindPlanarColoring[CycleGraph[5]]", "{1, 2}"),                                # inside and outside
    ("FindPlanarColoring[CycleGraph[5], {Red, Blue}]", "{Red, Blue}"),
    ("FindPlanarColoring[CycleGraph[5], 3]", "{1, 2}"),
    ("FindPlanarColoring[WheelGraph[6], {a, b, c}]", Raises(ValueError)),             # 4 colours needed
    ("FindPlanarColoring[CompleteGraph[5]]", Raises(ValueError)),                     # not planar
    ("FindPlanarColoring[PathGraph[{1, 2, 3}]]", Raises(NotImplementedError)),        # bridge
    ("FindPlanarColoring[5]", Raises(TypeError)),
    # EdgeCycleMatrix: rank m - n + c rows of closed cycles (Wolfram doc: WheelGraph[4] has 3 rows, GridGraph[{3, 3}] 4)
    ("EdgeCycleMatrix[WheelGraph[4]]", _cycle_basis("WheelGraph[4]", 3)),
    ("EdgeCycleMatrix[GridGraph[{3, 3}]]", _cycle_basis("GridGraph[{3, 3}]", 4)),
    ("EdgeCycleMatrix[PetersenGraph[]]", _cycle_basis("PetersenGraph[]", 6)),
    ("EdgeCycleMatrix[CycleGraph[4]]", "{{1, 1, 1, 1}}"),
    ("EdgeCycleMatrix[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]", "{{1, 1, 1}}"),
    ("EdgeCycleMatrix[Graph[{1 -> 2, 2 -> 3, 1 -> 3}]]", "{{1, 1, -1}}"),             # 1 -> 3 traversed backwards
    ("EdgeCycleMatrix[Graph[{1 <-> 2, 1 <-> 2, 2 <-> 2}]]", _cycle_basis("Graph[{1 <-> 2, 1 <-> 2, 2 <-> 2}]", 2)),
    ("EdgeCycleMatrix[PathGraph[{1, 2, 3}]]", "{}"),
    ("EdgeCycleMatrix[{1, 2}]", Raises(TypeError)),
    ("EdgeCycleMatrix[ecw24]", "EdgeCycleMatrix[ecw24]"),
    # GraphEmbedding: stored VertexCoordinates (list and rule forms); layouts are the graphics batch
    ("GraphEmbedding[Graph[{1 <-> 2, 2 <-> 3}, VertexCoordinates -> {{0, 0}, {1, 0}, {1, 1}}]]",
     "{{0, 0}, {1, 0}, {1, 1}}"),
    ("GraphEmbedding[Graph[{1 <-> 2}, VertexCoordinates -> {2 -> {5, 6}, 1 -> {3, 4}}]]", "{{3, 4}, {5, 6}}"),
    ("GraphEmbedding[Graph[{a <-> b}, VertexCoordinates -> {{0, 1/2}, {Sqrt[2], 0}}]]", "{{0, 1/2}, {Sqrt[2], 0}}"),
    ("GraphEmbedding[CycleGraph[3]]", Raises(NotImplementedError)),
    ("GraphEmbedding[Graph[{1 <-> 2}, VertexCoordinates -> {{0, 0}, {1, 0}}], \"SpringEmbedding\"]",
     Raises(NotImplementedError)),
    ("GraphEmbedding[Graph[{1 <-> 2}, VertexCoordinates -> {{0, 0}, {1, x}}]]", Raises(ValueError)),
    ("GraphEmbedding[1]", Raises(TypeError)),
    # TaggedEdgeStyle: inert presentation option kept on the graph
    ("AnnotationValue[Graph[{1 <-> 2}, TaggedEdgeStyle -> Red], TaggedEdgeStyle]", "Red"),
    ("MemberQ[AnnotationKeys[Graph[{1 <-> 2}, TaggedEdgeStyle -> Red]], TaggedEdgeStyle]", True),
    ("EdgeList[Graph[{1 <-> 2}, TaggedEdgeStyle -> {1 -> Blue}]]", "{UndirectedEdge[1, 2]}"),
]

if HAVE_NX:
    import networkx as _nxw

    def _nx_graph(edges, n):
        G = _nxw.Graph()
        G.add_nodes_from(range(1, n + 1))
        G.add_edges_from(edges)
        return G

    _pet = _nxw.petersen_graph()
    _grid = _nxw.grid_2d_graph(3, 4)
    W24B += [
        # networkx references (independent of MikoshiLang)
        ("MeanDegreeConnectivity[GridGraph[{3, 4}]]",
         NearList([0] + [_nxw.average_degree_connectivity(_grid).get(k, 0) for k in range(1, 5)])),
        ("KVertexConnectedComponents[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4, 4 <-> 5, 5 <-> 6, 6 <-> 4, 6 <-> 7}], 2]",
         _setlist([sorted(c) for c in _nxw.biconnected_components(_nx_graph(
             [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (6, 4), (6, 7)], 7))])),
        ("KEdgeConnectedComponents[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4, 4 <-> 5, 5 <-> 6, 6 <-> 4, 6 <-> 7}], 2]",
         _setlist([sorted(c) for c in _nxw.k_edge_subgraphs(_nx_graph(
             [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (6, 4), (6, 7)], 7), 2) if len(c) > 1])),
    ]

NX_HEADS = ("FindPlanarColoring", "FindIndependentEdgeSet", "FindEdgeCover", "IsomorphicSubgraphQ", "FindSubgraphIsomorphism",
            "FindIsomorphicSubgraph", "GraphAutomorphismGroup", "VertexTransitiveGraphQ", "EdgeTransitiveGraphQ",
            "PlanarFaceList", "DualPlanarGraph", "PlanarGraph", "IsomorphicGraphQ")

CASES = (ANNOTATIONS + OPTIONS + CONSTRUCTIONS + EDITING + COMPONENTS + MEASURES + MEASURES_NX + COLOURING + KSETS
         + FLOWS + SYMMETRY + POLYS + PLANAR + SPF + RANDOM + W24B)
if not HAVE_NX:
    # NetworkX-backed heads (and iso() predicates, which use IsomorphicGraphQ) need the optional dependency
    CASES = [c for c in CASES if not any(h in c[0] for h in NX_HEADS)
             and not (isinstance(c[1], Pred) and "isomorphic" in c[1].what)]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w24(case):
    check(case)


def test_canonical_graph_random_relabelling():
    """CanonicalGraph is invariant under random relabelling (self-check of the canonical form)."""
    import random
    rnd = random.Random(24)
    for src in ("PetersenGraph[]", "HypercubeGraph[3]", "GridGraph[{3, 4}]", "Graph[{1 -> 2, 2 -> 3, 3 -> 1, 3 -> 4}]"):
        n = py(ev(f"VertexCount[{src}]"))
        perm = list(range(1, n + 1))
        rnd.shuffle(perm)
        rules = "{" + ", ".join(f"{i} -> {p}" for i, p in zip(range(1, n + 1), perm)) + "}"
        assert ev(f"CanonicalGraph[VertexReplace[{src}, {rules}]] === CanonicalGraph[{src}]") is True


def test_seed_random_reproducible():
    a = ev("SeedRandom[24]; EdgeList[RandomGraph[BernoulliGraphDistribution[8, 1/2]]]")
    b = ev("SeedRandom[24]; EdgeList[RandomGraph[BernoulliGraphDistribution[8, 1/2]]]")
    assert a == b
