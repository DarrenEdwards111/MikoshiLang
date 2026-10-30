import pytest

from _harness import Raises, Pred, check, ids

try:
    import networkx as nx
except ImportError:
    nx = None

LOGIC = [
    ("Implies[True, False]", False), ("Implies[False, False]", True), ("Implies[True, True]", True), ("Implies[x, y]", "Implies[x, y]"),
    ("Equivalent[True, True]", True), ("Equivalent[True, False]", False), ("Equivalent[False, False, False]", True), ("Equivalent[p, q]", "Equivalent[p, q]"),
    ("Iff[True, True]", True), ("Iff[True, False]", False), ("Iff[x, y]", "Iff[x, y]"), ("Iff[False, False]", True),
    ("Xor[True, False]", True), ("Xor[True, True]", False), ("Xor[True, True, True]", True), ("Xor[a, b]", "Xor[a, b]"),
    ("Xnor[True, True]", True), ("Xnor[True, False]", False), ("Xnor[x, y]", "Xnor[x, y]"), ("Xnor[False, False, True]", False),
    ("Nand[True, True]", False), ("Nand[True, False]", True), ("Nand[False, False, False]", True),
    ("Nor[False, False]", True), ("Nor[True, False]", False), ("Nor[False, False, True]", False),
    ("Nand[True]", Raises()),
    ("TruthTable[a && b]", "{{False, False, False}, {False, True, False}, {True, False, False}, {True, True, True}}"),
    ("TruthTable[!p]", "{{False, True}, {True, False}}"),
    ("TruthTable[Xor[a, b]]", "{{False, False, False}, {False, True, True}, {True, False, True}, {True, True, False}}"),
    ("AllSAT[a || b]", "{{a -> False, b -> True}, {a -> True, b -> False}, {a -> True, b -> True}}"),
    ("AllSAT[a && !a]", "{}"), ("AllSAT[p]", "{{p -> True}}"),
    ("SATSolve[a && !b]", "{a -> True, b -> False}"), ("SATSolve[a && !a]", False), ("SATSolve[x == 1]", Raises()),
    ("SATSolve[(a || b) && !a]", "{a -> False, b -> True}"),
    ("CNF[(a && b) || c]", "(a || c) && (b || c)"), ("CNF[a]", "a"), ("CNF[!(a || b)]", "!a && !b"),
    ("DNF[(a || b) && c]", "(a && c) || (b && c)"), ("DNF[a]", "a"), ("DNF[!(a && b)]", "!a || !b"),
    ("BooleanMinimize[(a && b) || (a && !b)]", "a"), ("BooleanMinimize[a || (a && b)]", "a"), ("BooleanMinimize[a || !a]", True),
]

G1 = "{{1, 2}, {2, 3}, {1, 3}, {3, 4}}"
W = "{{1, 2, 2}, {2, 3, 1}, {1, 3, 5}}"
GRAPHS = [
    ("ShortestPath[" + W + ", 1, 3]", "{1, 2, 3}"), ("ShortestPath[{{1, 2}, {3, 4}}, 1, 4]", "{}"),
    ("ShortestPath[{UndirectedEdge[1, 2], UndirectedEdge[2, 3]}, 1, 3]", "{1, 2, 3}"), ("BooleanMinimize[a && !a]", False), ("ShortestPath[{{1, 2}}, 1, 9]", Raises()),
    ("PathLength[" + W + ", 1, 3]", "3"), ("PathLength[{{1, 2}, {3, 4}}, 1, 4]", "Infinity"), ("PathLength[{{a, b}, {b, c}}, a, c]", "2"),
    ("Dijkstra[" + W + ", 1]", "{1 -> 0, 2 -> 2, 3 -> 3}"), ("Dijkstra[{{1, 2}, {3, 4}}, 1]", "{1 -> 0, 2 -> 1}"),
    ("Dijkstra[{{1, 2, -1}}, 1]", Raises()),
    ("ConnectedComponents[{{1, 2}, {3, 4}, {4, 5}}]", "{{3, 4, 5}, {1, 2}}"), ("ConnectedComponents[" + G1 + "]", "{{1, 2, 3, 4}}"),
    ("ConnectedComponents[{1, 2}]", Raises()),
    ("ConnectedQ[" + G1 + "]", True), ("ConnectedQ[{{1, 2}, {3, 4}}]", False), ("ConnectedQ[{}]", True),
    ("BipartiteQ[{{1, 2}, {2, 3}}]", True), ("BipartiteQ[{{1, 2}, {2, 3}, {3, 1}}]", False), ("BipartiteQ[{{1, 2}, {2, 3}, {3, 4}, {4, 1}}]", True),
    ("GraphDiameter[" + G1 + "]", "2"), ("GraphDiameter[{{1, 2}, {3, 4}}]", "Infinity"), ("GraphDiameter[{{1, 2}, {2, 3}, {3, 4}}]", "3"),
    ("GraphRadius[" + G1 + "]", "1"), ("GraphRadius[{{1, 2}, {2, 3}, {3, 4}}]", "2"), ("GraphRadius[{{1, 2}, {3, 4}}]", "Infinity"),
    ("GraphCenter[" + G1 + "]", "{3}"), ("GraphCenter[{{1, 2}, {2, 3}, {3, 4}}]", "{2, 3}"), ("GraphCenter[{{1, 2}, {3, 4}}]", Raises()),
    ("VertexDegree[" + G1 + ", 3]", "3"), ("VertexDegree[" + G1 + ", 4]", "1"), ("VertexDegree[{{1, 1}}, 1]", "2"), ("VertexDegree[" + G1 + ", 9]", Raises()),
    ("ChromaticNumber[{{1, 2}, {2, 3}, {3, 1}}]", "3"), ("ChromaticNumber[{{1, 2}, {2, 3}, {3, 4}, {4, 1}}]", "2"),
    ("ChromaticNumber[{{1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}}]", "4"),
    # a graph where greedy colouring in vertex order needs 3 colours but chi = 2 (crown graph)
    ("ChromaticNumber[{{1, 4}, {1, 6}, {2, 3}, {2, 5}, {3, 6}, {4, 5}}]", "2"),
    ("ClusteringCoefficient[" + G1 + "]", "7/12"), ("ClusteringCoefficient[" + G1 + ", 3]", "1/3"),
    ("ClusteringCoefficient[{{1, 2}, {2, 3}}]", "0"),
    ("BetweennessCentrality[{{1, 2}, {2, 3}}]", "{1 -> 0, 2 -> 1, 3 -> 0}"),
    ("BetweennessCentrality[" + G1 + "]", "{1 -> 0, 2 -> 0, 3 -> 2, 4 -> 0}"),
    ("BetweennessCentrality[{{1, 2}, {2, 3}, {3, 4}, {4, 1}}]", "{1 -> 1/2, 2 -> 1/2, 3 -> 1/2, 4 -> 1/2}"),
    ("MinimumSpanningTree[{{1, 2, 3}, {2, 3, 1}, {1, 3, 1}}]", "{{2, 3}, {1, 3}}"), ("MinimumSpanningTree[{{1, 2}}]", "{{1, 2}}"),
    ("MinimumSpanningTree[{{1, 2, 1}, {2, 3, 1}, {1, 3, 1}, {3, 4, 5}}]", Pred(lambda r: len(r.args) == 3, "3 edges")),
    ("SpanningTree[" + G1 + "]", Pred(lambda r: len(r.args) == 3, "3 edges")), ("SpanningTree[{{1, 2}, {3, 4}}]", "{{1, 2}, {3, 4}}"),
    ("SpanningTree[{x}]", Raises()),
]

if nx is not None:
    pr = nx.pagerank(nx.Graph([(1, 2), (2, 3)]), tol=1e-14, max_iter=10000)
    pr2 = nx.pagerank(nx.Graph([(1, 2), (2, 3), (1, 3), (3, 4)]), tol=1e-14, max_iter=10000)
    GRAPHS += [
        ("PageRank[{{1, 2}, {2, 3}}]", Pred(lambda r: all(abs(x.args[1] - pr[x.args[0]]) < 1e-9 for x in r.args), "NetworkX pagerank")),
        ("PageRank[" + G1 + "]", Pred(lambda r: all(abs(x.args[1] - pr2[x.args[0]]) < 1e-9 for x in r.args), "NetworkX pagerank")),
        ("PageRank[{x}]", Raises()),
        ("MaximalMatching[{{1, 2}, {2, 3}, {3, 4}}]", "{{1, 2}, {3, 4}}"), ("MaximalMatching[{{1, 2}, {1, 3}}]", Pred(lambda r: len(r.args) == 1, "1 edge")),
        ("MaximalMatching[{{1, 2}, {2, 3}, {3, 1}, {4, 5}}]", Pred(lambda r: len(r.args) == 2, "2 edges")),
    ]
else:
    GRAPHS += [
        ("PageRank[{{1, 2}, {2, 3}}]", Pred(lambda r: abs(sum(x.args[1] for x in r.args) - 1) < 1e-9, "sums to 1")),
        ("PageRank[{{1, 2}, {2, 3}}]", Pred(lambda r: abs(r.args[1].args[1] - 0.4864864864) < 1e-6, "centre rank")),
        ("PageRank[{x}]", Raises()),
    ] + [("MaximalMatching[{{1, 2}}]", Raises(NotImplementedError, "NetworkX"))] * 3

CASES = LOGIC + GRAPHS


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_logic_graphs(case):
    check(case)
