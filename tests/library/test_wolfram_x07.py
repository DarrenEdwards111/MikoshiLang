"""Differential tests for coverage batch X07 (graphs III: graph plots, communities & partitions, tree and fractal
plots; module ``wl_graphplots``).

Reference values (cited per block):

* Wolfram documentation (reference.wolfram.com/language/ref/<Name>.html): the documented examples of
  LambdaComponents, LuccioSamiComponents (rules example {1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4, 3 -> 5, 4 -> 6, 5 -> 6}
  and the 8-vertex two-cluster graph), FindGraphCommunities (Graph[{1<->2, 2<->3, 1<->3, 3<->4, 4<->5, 7<->6,
  8<->6, 5<->7, 7<->8}]), FindGraphPartition (two triangles joined by 2<->4), TaggedNestGraph ({2 #, 2 # + 1}&
  and Mod[#^2 + 1, 5]&), the MandelbrotSetPlot default rectangle -2 - 1.3 I .. 0.6 + 1.3 I and Raster output,
  GraphPropertyDistribution[EdgeCount[g], g ~ BernoulliGraphDistribution[n, p]];
* closed forms: circular layout angles Pi/2 - 2 Pi k/n, the Julia sets of z^2 (unit circle) and z^2 - 2 ([-2, 2]),
  binomial/hypergeometric moments of edge counts and degrees in G(n, p) and G(n, m) (Bollobas, Random Graphs 2.1),
  modularity Q = sum_c (e_c/m - (d_c/2m)^2) (Newman & Girvan 2004);
* NetworkX (computed in this file, independent of MikoshiLang): modularity of the returned communities, minimum
  cuts and edge connectivities for lambda sets (Borgatti, Everett & Shirey 1990);
* the MikoshiLang renderer itself for FullGraphics (the explicit primitives must render over the same range).

Plots are checked by structural predicates (one primitive per vertex/edge, endpoints match the vertex points,
monotone encodings, layer invariants), since the Wolfram Language's coordinates are not published.
"""
import importlib.util
import itertools
import math

import pytest

from _harness import Pred, Raises, check, ev, ids
from mikoshilang.expr import Expr, Symbol

HAVE_NX = importlib.util.find_spec("networkx") is not None
HAVE_MPL = importlib.util.find_spec("matplotlib") is not None


# ───────────────────────────── helpers ─────────────────────────────

def walk(x):
    yield x
    if isinstance(x, Expr):
        for a in x.args:
            yield from walk(a)


def prims(g, head):
    return [e for e in walk(g) if isinstance(e, Expr) and e.head == head]


def num(x):
    return float(x)


def point(p):
    return tuple(num(a) for a in p.args)


def py(x):
    if isinstance(x, tuple):
        return [py(a) for a in x]
    if isinstance(x, Expr) and x.head == "List":
        return [py(a) for a in x.args]
    if isinstance(x, Symbol):
        return x.name
    if isinstance(x, (bool, str)):
        return x
    try:
        f = float(x)
        return int(f) if f == int(f) else f
    except Exception:
        return x


def is_graphics(g, head="Graphics"):
    return isinstance(g, Expr) and g.head == head


def disks(g):
    return [point(d.args[0]) for d in prims(g, "Disk")]


def lines(g):
    return [[point(p) for p in ln.args[0].args] for ln in prims(g, "Line") + prims(g, "Arrow")]


def edges_join_vertices(g, nedges, head2="Disk"):
    """Every edge line starts and ends at a vertex centre."""
    vs = [point(d.args[0]) for d in prims(g, head2)]
    ls = lines(g)

    def near(p):
        return any(math.dist(p, v) < 0.6 * min_dist(vs) + 1e-9 for v in vs)
    return len(ls) >= nedges and all(near(l[0]) and near(l[-1]) for l in ls)


def min_dist(vs):
    if len(vs) < 2:
        return 1.0
    return min(math.dist(a, b) for a, b in itertools.combinations(vs, 2))


def pred(f, what):
    return Pred(f, what)


def parts_of(x):
    return [sorted(py(c)) for c in x.args]


G6 = "Graph[{1<->2, 2<->3, 3<->1, 3<->4, 4<->5, 5<->6, 6<->4}]"
DOC_COMM = "Graph[{1<->2, 2<->3, 1<->3, 3<->4, 4<->5, 7<->6, 8<->6, 5<->7, 7<->8}]"
DOC_PART = "Graph[{1<->2, 2<->3, 3<->1, 2<->4, 4<->5, 5<->6, 6<->4}]"
DOC_RULES = "{1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4, 3 -> 5, 4 -> 6, 5 -> 6}"
DOC_LAMBDA8 = "Graph[{1<->2, 1<->3, 1<->4, 2<->3, 3<->4, 2<->5, 5<->6, 6<->7, 5<->8, 6<->8, 7<->8, 4<->7}]"
GAME = "TreeGame[{\"A\", {\"B\" -> 1, {1, 2}, {3, 4}} -> \"l\", {\"B\" -> 1, {0, 1}, {2, 2}} -> \"r\"}]"


def circle_angles(g):
    vs = disks(g)
    n = len(vs)
    cx = sum(v[0] for v in vs) / n
    cy = sum(v[1] for v in vs) / n
    return all(abs(math.atan2(v[1] - cy, v[0] - cx) - math.atan2(math.sin(math.pi / 2 - 2 * math.pi * k / n),
                                                                  math.cos(math.pi / 2 - 2 * math.pi * k / n)))
               % (2 * math.pi) < 1e-4 or
               abs(abs(math.atan2(v[1] - cy, v[0] - cx) - math.atan2(math.sin(math.pi / 2 - 2 * math.pi * k / n),
                                                                     math.cos(math.pi / 2 - 2 * math.pi * k / n)))
                   - 2 * math.pi) < 1e-4
               for k, v in enumerate(vs))


def layers_respected(g, arcs_n):
    """Directed arcs (Arrow) point downward (root at the top)."""
    ars = [[point(p) for p in a.args[0].args] for a in prims(g, "Arrow")]
    return len(ars) == arcs_n and all(a[0][1] > a[-1][1] for a in ars)


def tree_levels(g):
    ys = sorted({round(v[1], 6) for v in disks(g)}, reverse=True)
    return ys


CASES = [
    # ── GraphPlot (layout methods documented in the module notes) ──
    ("GraphPlot[" + G6 + "]", pred(lambda g: is_graphics(g) and len(disks(g)) == 6 and edges_join_vertices(g, 7),
                                   "6 vertex disks and 7 edges joining them")),
    ("GraphPlot[{1 -> 2, 2 -> 3, 3 -> 1}]", pred(lambda g: len(disks(g)) == 3 and not prims(g, "Arrow"),
                                                  "rules are undirected (no arrows)")),
    ("GraphPlot[{1 -> 2, 2 -> 3}, DirectedEdges -> True]", pred(lambda g: len(prims(g, "Arrow")) == 2,
                                                                  "DirectedEdges -> True draws arrows")),
    ("GraphPlot[CycleGraph[6], GraphLayout -> \"CircularEmbedding\"]",
     pred(circle_angles, "vertex k at angle Pi/2 - 2 Pi k/n")),
    ("GraphPlot[{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}}]", pred(lambda g: len(disks(g)) == 3 and len(lines(g)) == 3,
                                                           "adjacency matrix input")),
    ("GraphPlot[PathGraph[Range[5]], GraphLayout -> \"LinearEmbedding\"]",
     pred(lambda g: len({round(v[1], 6) for v in disks(g)}) == 1, "linear embedding on one line")),
    ("GraphPlot[" + G6 + ", GraphLayout -> \"SpectralEmbedding\"]", pred(lambda g: len(disks(g)) == 6, "spectral")),
    ("GraphPlot[" + G6 + ", GraphLayout -> \"SpringEmbedding\"]", pred(lambda g: len(disks(g)) == 6, "stress")),
    ("GraphPlot[StarGraph[5], GraphLayout -> \"StarEmbedding\"]",
     pred(lambda g: min(math.dist(disks(g)[0], v) for v in disks(g)[1:]) > 0.5, "centre vertex apart")),
    ("GraphPlot[" + G6 + ", VertexLabels -> \"Name\"]", pred(lambda g: len(prims(g, "Text")) == 6, "6 labels")),
    ("GraphPlot[" + G6 + ", GraphHighlight -> {1}]",
     pred(lambda g: any(py(c.args) == [1, 0, 0] for c in prims(g, "RGBColor")), "highlight in red")),
    ("GraphPlot[x]", "GraphPlot[x]"),
    ("GraphPlot[\"a\"]", Raises(TypeError)),
    ("GraphPlot[" + G6 + ", GraphLayout -> \"HighDimensionalEmbedding\"]", Raises(NotImplementedError)),
    ("GraphPlot[" + G6 + ", GraphLayout -> \"Nonsense\"]", Raises(ValueError)),
    ("Head[Rasterize[GraphPlot[" + G6 + "]]]", "Image"),

    # ── GraphPlot3D ──
    ("GraphPlot3D[" + G6 + "]", pred(lambda g: is_graphics(g, "Graphics3D") and len(prims(g, "Sphere")) == 6,
                                     "6 spheres")),
    ("GraphPlot3D[{1 -> 2, 2 -> 3, 3 -> 1}]",
     pred(lambda g: all(len(point(s.args[0])) == 3 for s in prims(g, "Sphere")), "3-D points")),
    ("GraphPlot3D[" + G6 + ", GraphLayout -> \"SpectralEmbedding\"]",
     pred(lambda g: len(prims(g, "Sphere")) == 6 and len(prims(g, "Line")) == 7, "edges drawn")),
    ("GraphPlot3D[y]", "GraphPlot3D[y]"),
    ("GraphPlot3D[1.5]", Raises(TypeError)),
    ("GraphPlot3D[" + G6 + ", GraphLayout -> \"CircularEmbedding\"]", Raises(NotImplementedError)),

    # ── LayeredGraphPlot (Sugiyama) ──
    ("LayeredGraphPlot[{1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4}]", pred(lambda g: layers_respected(g, 4),
                                                                 "arcs point down")),
    ("LayeredGraphPlot[{1 -> 2, 2 -> 3, 3 -> 1}]", pred(lambda g: len(disks(g)) == 3 and len(prims(g, "Arrow")) == 3,
                                                         "cycles are drawn")),
    ("LayeredGraphPlot[{1 -> 2, 1 -> 3}, Left]",
     pred(lambda g: disks(g)[0][0] < min(disks(g)[1][0], disks(g)[2][0]), "root on the left")),
    ("LayeredGraphPlot[{1 -> 2, 2 -> 3, 1 -> 3}]",
     pred(lambda g: len(tree_levels(g)) == 3, "longest-path layering: 3 layers")),
    ("LayeredGraphPlot[{1 -> 2, 3 -> 2}, 2 -> Top]", Raises(NotImplementedError)),
    ("LayeredGraphPlot[{1 -> 2}, 3]", Raises(ValueError)),
    ("LayeredGraphPlot[{1 -> 2}, Middle]", "LayeredGraphPlot[{1 -> 2}, Middle]"),
    ("LayeredGraphPlot[z]", "LayeredGraphPlot[z]"),
    ("LayeredGraphPlot[{1, 2}]", Raises(TypeError)),

    # ── LayeredGraphPlot3D, GraphLayers, GraphLayerStyle ──
    ("LayeredGraphPlot3D[{1 -> 2, 1 -> 3, 2 -> 4}]",
     pred(lambda g: sorted({round(point(s.args[0])[2], 6) for s in prims(g, "Sphere")}) == [-2.0, -1.0, 0.0],
          "layers at heights 0, -1, -2")),
    ("LayeredGraphPlot3D[{1 -> 2, 1 -> 3, 2 -> 4}, GraphLayers -> All]",
     pred(lambda g: len(prims(g, "Polygon")) == 3, "one plane per layer")),
    ("LayeredGraphPlot3D[{1 -> 2, 1 -> 3, 2 -> 4}, GraphLayers -> {1, 3}, GraphLayerStyle -> Red]",
     pred(lambda g: len(prims(g, "Polygon")) == 2 and prims(g, "RGBColor"), "two styled planes")),
    ("LayeredGraphPlot3D[{1 -> 2, 1 -> 3, 2 -> 4}, GraphLayers -> 2]",
     pred(lambda g: len(prims(g, "Polygon")) == 1, "single plane")),
    ("LayeredGraphPlot3D[{1 -> 2}, Left]", Raises(NotImplementedError)),
    ("LayeredGraphPlot3D[{1 -> 2}, GraphLayers -> 7]", Raises(ValueError)),
    ("LayeredGraphPlot3D[{1 -> 2}, GraphLayerStyle -> \"Rainbow\"]", Raises(NotImplementedError)),
    ("LayeredGraphPlot3D[w]", "LayeredGraphPlot3D[w]"),
    ("LayeredGraphPlot3D[\"g\"]", Raises(TypeError)),
    ("GraphLayers[1]", "GraphLayers[1]"),
    ("GraphLayerStyle[Red]", "GraphLayerStyle[Red]"),

    # ── TreePlot (Reingold-Tilford) ──
    ("TreePlot[{1 -> 2, 1 -> 3, 2 -> 4, 2 -> 5}, Top, 1]",
     pred(lambda g: len(tree_levels(g)) == 3 and disks(g)[0][1] == max(v[1] for v in disks(g)), "root on top")),
    ("TreePlot[{1 -> 2, 1 -> 3, 2 -> 4, 2 -> 5}, Top, 1]",
     pred(lambda g: abs(disks(g)[1][0] - (disks(g)[3][0] + disks(g)[4][0]) / 2) < 1e-9,
          "parent centred over its children")),
    ("TreePlot[KaryTree[7], Bottom]", pred(lambda g: disks(g)[0][1] == min(v[1] for v in disks(g)),
                                          "root at the bottom")),
    ("TreePlot[KaryTree[7], Center]",
     pred(lambda g: all(abs(math.dist(v, disks(g)[0]) - 1) < 1e-9 for v in disks(g)[1:3]), "radial depth 1")),
    ("TreePlot[{1 -> 2, 2 -> 3}, Top, 9]", Raises(ValueError)),
    ("TreePlot[{1 -> 2}, LayerSizeFunction -> (#^2 &)]", Raises(NotImplementedError)),
    ("TreePlot[t]", "TreePlot[t]"),
    ("TreePlot[\"x\"]", Raises(TypeError)),
    ("TreePlot[{}]", Raises(ValueError)),

    # ── Graph3D, LayeredGraph, LayeredGraph3D ──
    ("VertexList[Graph3D[{1 <-> 2, 2 <-> 3}]]", "{1, 2, 3}"),
    ("EdgeList[Graph3D[{1 <-> 2, 2 <-> 3}]]", "{UndirectedEdge[1, 2], UndirectedEdge[2, 3]}"),
    ("Head[GraphPlot3D[Graph3D[{1 <-> 2, 2 <-> 3}]]]", "Graphics3D"),
    ("Graph3D[q]", "Graph3D[q]"),
    ("Graph3D[{1 <-> 2}, GraphLayout -> 5]", Raises(ValueError)),
    ("Graph3D[3]", Raises(TypeError)),
    ("EdgeList[LayeredGraph[{1 -> 2, 2 -> 3}]]", "{DirectedEdge[1, 2], DirectedEdge[2, 3]}"),
    ("Head[GraphPlot[LayeredGraph[{1 -> 2, 2 -> 3}]]]", "Graphics"),
    ("GraphPlot[LayeredGraph[{1 -> 2, 1 -> 3, 2 -> 4}]]", pred(lambda g: layers_respected(g, 3),
                                                              "drawn with the layered layout")),
    ("LayeredGraph[{1 -> 2}, GraphLayout -> 5]", Raises(ValueError)),
    ("LayeredGraph[{1 -> 2}, 3]", Raises(TypeError)),
    ("LayeredGraph[r]", "LayeredGraph[r]"),
    ("LayeredGraph[\"r\"]", Raises(TypeError)),
    ("VertexList[LayeredGraph3D[{1 -> 2, 2 -> 3}]]", "{1, 2, 3}"),
    ("Head[GraphPlot3D[LayeredGraph3D[{1 -> 2, 2 -> 3}]]]", "Graphics3D"),
    ("GraphPlot3D[LayeredGraph3D[{1 -> 2, 2 -> 3}]]",
     pred(lambda g: sorted({round(point(s.args[0])[2], 6) for s in prims(g, "Sphere")}) == [-2.0, -1.0, 0.0],
          "layered heights")),
    ("LayeredGraph3D[{1 -> 2}, Left]", Raises(NotImplementedError)),
    ("LayeredGraph3D[s]", "LayeredGraph3D[s]"),
    ("LayeredGraph3D[2.5]", Raises(TypeError)),

    # ── GraphPropertyDistribution (exact closed forms) ──
    ("GraphPropertyDistribution[EdgeCount[x], x \\[Distributed] BernoulliGraphDistribution[10, p]]",
     "BinomialDistribution[45, p]"),
    ("GraphPropertyDistribution[EdgeCount[x], x \\[Distributed] BernoulliGraphDistribution[n, p]]",
     "BinomialDistribution[n (n - 1)/2, p]"),
    ("GraphPropertyDistribution[VertexDegree[x, 1], x \\[Distributed] BernoulliGraphDistribution[10, 1/3]]",
     "BinomialDistribution[9, 1/3]"),
    ("Mean[GraphPropertyDistribution[VertexDegree[x, 1], x \\[Distributed] UniformGraphDistribution[10, 20]]]",
     "4"),
    ("Variance[GraphPropertyDistribution[VertexDegree[x, 1], x \\[Distributed] UniformGraphDistribution[10, 20]]]",
     "(9 * 20 * 36 * 25)/(45^2 * 44)"),
    ("GraphPropertyDistribution[VertexCount[x], x \\[Distributed] BernoulliGraphDistribution[10, p]]",
     "GraphPropertyDistribution[VertexCount[x], x \\[Distributed] BernoulliGraphDistribution[10, p]]"),
    ("g7 = 5; GraphPropertyDistribution[EdgeCount[g7], g7 \\[Distributed] BernoulliGraphDistribution[4, 1/2]]",
     pred(lambda r: r.head == "GraphPropertyDistribution", "HoldFirst keeps the property unevaluated")),
    ("GraphPropertyDistribution[EdgeCount[x], 5]", Raises(TypeError)),
    ("GraphPropertyDistribution[EdgeCount[x], x \\[Distributed] NormalDistribution[0, 1]]", Raises(TypeError)),

    # ── TaggedNestGraph (documented constructions) ──
    ("EdgeList[TaggedNestGraph[{2 #, 2 # + 1} &, 1, 2]]",
     "{DirectedEdge[1, 2, 1], DirectedEdge[1, 3, 2], DirectedEdge[2, 4, 1], DirectedEdge[2, 5, 2], "
     "DirectedEdge[3, 6, 1], DirectedEdge[3, 7, 2]}"),
    ("EdgeList[TaggedNestGraph[Mod[#^2 + 1, 5] &, Range[0, 2], 2]]",
     "{DirectedEdge[0, 1, 1], DirectedEdge[1, 2, 1], DirectedEdge[2, 0, 1]}"),
    ("VertexList[TaggedNestGraph[Mod[#^2 + 1, 10] &, {1, 2, 3, 4, 5}, 2]]", "{1, 2, 3, 4, 5, 0, 7, 6}"),
    ("EdgeList[TaggedNestGraph[Mod[#^2 + 1, 5] &, Range[0, 2], 2, DirectedEdges -> False]]",
     "{UndirectedEdge[0, 1, 1], UndirectedEdge[1, 2, 1], UndirectedEdge[2, 0, 1]}"),
    ("(EdgeList[TaggedNestGraph[{2 #, 2 # + 1} &, 1, 3]] /. DirectedEdge[a_, b_, _] :> DirectedEdge[a, b]) === "
     "EdgeList[NestGraph[{2 #, 2 # + 1} &, 1, 3]]", "True"),
    ("VertexCount[TaggedNestGraph[Mod[# + 1, 4] &, 0, Infinity]]", "4"),
    ("TaggedNestGraph[f, 1, -1]", Raises(ValueError)),
    ("TaggedNestGraph[f, 1, \"a\"]", Raises(TypeError)),
    ("TaggedNestGraph[f, 1, n]", "TaggedNestGraph[f, 1, n]"),

    # ── TreeGamePlot ──
    ("TreeGamePlot[" + GAME + "]", pred(lambda g: len(prims(g, "Disk")) == 7 and len(prims(g, "Line")) >= 6,
                                        "7 nodes and 6 actions")),
    ("TreeGamePlot[" + GAME + "]", pred(lambda g: len([ln for ln in walk(g) if isinstance(ln, Expr) and
                                                        ln.head == "List" and any(isinstance(a, Expr) and
                                                                                  a.head == "Dashing"
                                                                                  for a in ln.args)]) == 1,
                                        "one dashed information set")),
    ("TreeGamePlot[" + GAME + ", <|\"A\" -> <|{} -> {0.8, 0.2}|>|>]",
     pred(lambda g: [round(num(o.args[0]), 6) for o in prims(g, "Opacity")] == [0.83, 0.32],
          "strategy shading 0.15 + 0.85 p")),
    ("TreeGamePlot[TreeGame[{\"N\", {1, 2} -> 1/4, {3, 4} -> 3/4}]]",
     pred(lambda g: [round(num(t.args[0]), 6) for t in prims(g, "AbsoluteThickness")] == [2.0, 4.0],
          "chance thickness 1 + 4 p")),
    ("TreeGamePlot[" + GAME + ", Appearance -> None]",
     pred(lambda g: not prims(g, "Dashing"), "Appearance -> None hides information sets")),
    ("TreeGamePlot[" + GAME + ", GameActionLabels -> None, GamePlayerLabels -> None]",
     pred(lambda g: all(t.args[0].head == "List" for t in prims(g, "Text")), "only payoffs labelled")),
    ("TreeGamePlot[" + GAME + ", GraphLayout -> \"RadialEmbedding\"]", pred(lambda g: len(disks(g)) == 7, "radial")),
    ("TreeGamePlot[" + GAME + ", <|\"A\" -> <|{} -> {0.5, 0.2, 0.3}|>|>]", Raises(ValueError)),
    ("TreeGamePlot[" + GAME + ", PlotLegends -> Automatic]", Raises(NotImplementedError)),
    ("TreeGamePlot[" + GAME + ", 1, 2]", Raises(TypeError)),
    ("TreeGamePlot[tg]", "TreeGamePlot[tg]"),

    # ── MandelbrotSetPlot, EscapeRadius ──
    ("MandelbrotSetPlot[ImageResolution -> 13, MaxIterations -> 30]",
     pred(lambda g: py(g.args[0].args[1]) == [[-2, -1.3], [0.6, 1.3]] and len(g.args[0].args[0].args) == 13,
          "documented default rectangle, 13 x 13 Raster")),
    ("MandelbrotSetPlot[{-2 - I, 1 + I}, ImageResolution -> 6, MaxIterations -> 50, ColorFunction -> None]",
     pred(lambda g: [[1 if py(c) == [0, 0, 0] else 0 for c in row.args] for row in g.args[0].args[0].args] ==
          [[1 if mandel(complex(-2 + (i + .5) * .5, -1 + (j + .5) * .5), 50) else 0 for i in range(6)]
           for j in range(4)], "membership equals escape counting")),
    ("MandelbrotSetPlot[{-2 - I, 1 + I}, ImageResolution -> 6, MaxIterations -> 20, EscapeRadius -> 10, "
     "ColorFunction -> None]", pred(lambda g: g.head == "Graphics", "larger escape radius")),
    ("MandelbrotSetPlot[{-2 - I, 1 + I}, ImageResolution -> 6, ColorFunction -> (GrayLevel[#] &)]",
     pred(lambda g: any(py(c) == [1, 1, 1] for row in g.args[0].args[0].args for c in row.args),
          "ColorFunction gets t = 1 inside the set")),
    ("Options[MandelbrotSetPlot[{-1 - I, 1 + I}, ImageResolution -> 4], Frame]", "{Frame -> True}"),
    ("MandelbrotSetPlot[{-2 - I, 1 + I}, ColorFunction -> \"Rainbow\", ImageResolution -> 4]",
     Raises(NotImplementedError)),
    ("MandelbrotSetPlot[EscapeRadius -> 1]", Raises(ValueError)),
    ("MandelbrotSetPlot[{1, 1}]", Raises(ValueError)),
    ("MandelbrotSetPlot[5]", Raises(TypeError)),
    ("MandelbrotSetPlot[{a, 1 + I}]", "MandelbrotSetPlot[{a, 1 + I}]"),
    ("EscapeRadius[3]", "EscapeRadius[3]"),

    # ── JuliaSetPlot ──
    ("JuliaSetPlot[0]", pred(lambda g: all(abs(math.hypot(*point(p)) - 1) < 1e-6
                                           for p in prims(g, "Point")[0].args[0].args), "J(z^2) = unit circle")),
    ("JuliaSetPlot[-2]", pred(lambda g: all(abs(point(p)[1]) < 1e-6 and -2 - 1e-6 <= point(p)[0] <= 2 + 1e-6
                                            for p in prims(g, "Point")[0].args[0].args), "J(z^2 - 2) = [-2, 2]")),
    ("JuliaSetPlot[z^2 - 1, z, PlotStyle -> Red]", pred(lambda g: prims(g, "RGBColor") and prims(g, "Point"),
                                                        "PlotStyle")),
    ("JuliaSetPlot[-1, Method -> \"EscapeTime\", ImageResolution -> 9, MaxIterations -> 40]",
     pred(lambda g: prims(g, "Raster") and len(prims(g, "Raster")[0].args[0].args) == 9, "escape-time raster")),
    ("JuliaSetPlot[z/(z^2 - 1), z, Method -> \"OrbitDetection\"]", Raises(NotImplementedError)),
    ("JuliaSetPlot[z^2/(z^2 - 1), z, Method -> \"EscapeTime\"]", Raises(NotImplementedError)),
    ("JuliaSetPlot[-1, Method -> \"Fast\"]", Raises(ValueError)),
    ("JuliaSetPlot[z^2, 3]", Raises(TypeError)),
    ("JuliaSetPlot[c]", "JuliaSetPlot[c]"),

    # ── JuliaSetIterationCount (plain escape-time count; references by direct iteration in jsic below) ──
    ("JuliaSetIterationCount[z^2 - 1, z, {0, 0.5 + 0.5 I, 1 + I, 1.5 + 1.5 I, 2 + 2 I}]",
     pred(lambda r: py(r) == [jsic([1, 0, -1], z, 1000) for z in (0, .5 + .5j, 1 + 1j, 1.5 + 1.5j, 2 + 2j)],
          "direct escape-time iteration, R = 2")),
    ("JuliaSetIterationCount[0, 3]", "0"),
    ("JuliaSetIterationCount[0, 1.5]", pred(lambda r: py(r) == 1, "1.5 -> 2.25 escapes |z| > 2")),
    ("JuliaSetIterationCount[0, 1.01, MaxIterations -> 2000]",
     pred(lambda r: py(r) == math.ceil(math.log2(math.log(2) / math.log(1.01))),
          "closed form for z^2: |z0|^(2^k) > 2")),
    ("JuliaSetIterationCount[-1, 1.5, EscapeRadius -> 100]", pred(lambda r: py(r) == jsic([1, 0, -1], 1.5, 1000, 100),
                                                                 "explicit escape radius")),
    ("JuliaSetIterationCount[z^2 - 1, z, .1 - .2 I, MaxIterations -> 50]", "51"),
    ("JuliaSetIterationCount[z^3 - z + 1, z, {-2, 1 + I}]",
     pred(lambda r: py(r) == [jsic([1, 0, -1, 1], z, 1000) for z in (-2, 1 + 1j)], "cubic, R = 4")),
    ("JuliaSetIterationCount[0, 1.5, WorkingPrecision -> 30]", "1"),
    ("JuliaSetIterationCount[-1, .1 - .2 I, Method -> \"OrbitDetection\"]", Raises(NotImplementedError)),
    ("JuliaSetIterationCount[z^2/(z^2 - 1), z, 0]", Raises(NotImplementedError)),
    ("JuliaSetIterationCount[0, 1.5, EscapeRadius -> 1]", Raises(ValueError)),
    ("JuliaSetIterationCount[0, 1.5, MaxIterations -> 0]", Raises(ValueError)),
    ("JuliaSetIterationCount[0, \"a\"]", Raises(TypeError)),
    ("JuliaSetIterationCount[-1, w]", "JuliaSetIterationCount[-1, w]"),

    # ── BioSequencePlot ──
    ("BioSequencePlot[BioSequence[\"DNA\", \"ATAATCTGT\"]]",
     pred(lambda g: len(disks(g)) == 9 and len(prims(g, "Text")) == 9, "9 letters")),
    ("BioSequencePlot[BioSequence[\"RNA\", \"GCCAGACUGAYAUCUGGA\", {Bond[{2, 17}], Bond[{3, 16}]}]]",
     pred(lambda g: len(prims(g, "Line")) == 3, "backbone and two bonds")),
    ("BioSequencePlot[BioSequence[\"CircularDNA\", \"ATGC\"]]",
     pred(lambda g: all(abs(math.hypot(*v) - 1) < 1e-9 for v in disks(g)), "circular strand on a circle")),
    ("BioSequencePlot[BioSequence[\"HybridStrand\", {\"ATG\", \"CAT\"}, {Bond[{{1, 1}, {2, 3}}]}]]",
     pred(lambda g: len(disks(g)) == 6 and len(prims(g, "Line")) == 3, "two strands and a bond")),
    ("BioSequencePlot[BioSequence[\"DNA\", \"A\" <> StringJoin[Table[\"C\", 59]]]]",
     pred(lambda g: len({round(v[1], 6) for v in disks(g)}) == 2, "wrapped at 50 letters")),
    ("BioSequencePlot[BioSequence[\"RNA\", \"GCAU\", {Bond[{2, 9}]}]]", Raises(ValueError)),
    ("BioSequencePlot[BioSequence[{\"ATG\", \"CAT\"}, {}]]", Raises(NotImplementedError)),
    ("BioSequencePlot[3]", Raises(TypeError)),
    ("BioSequencePlot[b]", "BioSequencePlot[b]"),

    # ── FullGraphics ──
    ("FullGraphics[Graphics[Disk[], Axes -> True]]",
     pred(lambda g: [py(t.args[0]) for t in prims(g, "Text")] == ["-1", "-0.5", "0.5", "1", "-1", "-0.5", "0.5", "1"]
          and not any(py(r.args[0]) == "Axes" for r in g.args[1:]), "explicit axes with renderer ticks")),
    ("FullGraphics[Graphics[Circle[], Frame -> True]]",
     pred(lambda g: len(prims(g, "Line")) >= 4 and
          any(py(r.args[0]) == "PlotRangePadding" and py(r.args[1]) == "None" for r in g.args[1:]),
          "frame lines, PlotRangePadding -> None")),
    ("FullGraphics[Graphics[Circle[], GridLines -> Automatic]]",
     pred(lambda g: len(prims(g, "Line")) == 10, "5 + 5 grid lines")),
    ("FullGraphics[Graphics[Disk[], PlotLabel -> \"t\"]]",
     pred(lambda g: [py(t.args[0]) for t in prims(g, "Text")] == ["t"], "plot label as Text")),
    ("FullGraphics[Graphics[Disk[]]]",
     pred(lambda g: any(py(r.args[0]) == "PlotRange" and py(r.args[1]) == [[-1.04, 1.04], [-1.04, 1.04]]
                        for r in g.args[1:]), "explicit plot range = rendered range")),
    ("FullGraphics[Graphics3D[Sphere[]]]", Raises(NotImplementedError)),
    ("FullGraphics[Graphics[Disk[], Epilog -> {Red, Point[{0, 0}]}]]", Raises(NotImplementedError)),
    ("FullGraphics[3]", Raises(TypeError)),
    ("FullGraphics[gr]", "FullGraphics[gr]"),
]


def jsic(coeffs, z, mx, R=None):
    """Independent escape-time count: smallest k with |f^k(z)| > R, else mx + 1."""
    z = complex(z)
    R = R or max(1.0, (2 + sum(abs(c) for c in coeffs[1:])) / abs(coeffs[0]))
    for k in range(mx + 1):
        if abs(z) > R:
            return k
        if k == mx:
            break
        acc = 0j
        for c in coeffs:
            acc = acc * z + c
        z = acc
    return mx + 1


def mandel(c, mx):
    z = 0j
    for _ in range(mx):
        z = z * z + c
        if abs(z) > 2:
            return False
    return True


NX_CASES = [
    # ── FindGraphCommunities (documented example; modularity recomputed by NetworkX in test_x07_modularity) ──
    ("FindGraphCommunities[" + DOC_COMM + "]", "{{1, 2, 3}, {7, 6, 8}, {4, 5}}"),
    ("FindGraphCommunities[" + DOC_COMM + ", Method -> \"Spectral\"]", "{{1, 2, 3, 4}, {5, 7, 6, 8}}"),
    ("FindGraphCommunities[" + DOC_COMM + ", Method -> \"Centrality\"]", "{{1, 2, 3, 4}, {5, 7, 6, 8}}"),
    ("FindGraphCommunities[" + DOC_COMM + ", Method -> \"CliquePercolation\"]", "{{1, 2, 3}, {7, 6, 8}}"),
    ("FindGraphCommunities[{1 -> 2, 2 -> 3, 3 -> 1, 4 -> 5, 5 -> 6, 6 -> 4}]", "{{1, 2, 3}, {4, 5, 6}}"),
    ("FindGraphCommunities[" + DOC_COMM + ", Method -> \"Hierarchical\"]", Raises(NotImplementedError)),
    ("FindGraphCommunities[" + DOC_COMM + ", Method -> \"Magic\"]", Raises(ValueError)),
    ("FindGraphCommunities[7]", Raises(TypeError)),
    ("FindGraphCommunities[h]", "FindGraphCommunities[h]"),

    # ── FindGraphPartition (documented example; exact minimum cuts) ──
    ("FindGraphPartition[" + DOC_PART + "]", "{{1, 2, 3}, {4, 5, 6}}"),
    ("FindGraphPartition[{1 -> 2, 2 -> 3, 3 -> 1, 2 -> 4, 4 -> 5, 5 -> 6, 6 -> 4}]", "{{1, 2, 3}, {4, 5, 6}}"),
    ("FindGraphPartition[" + DOC_PART + ", {2, 4}]", pred(lambda r: sorted(len(p.args) for p in r.args) == [2, 4],
                                                          "sizes 2 and 4")),
    ("FindGraphPartition[" + DOC_PART + ", {0.5, 0.5}]", "{{1, 2, 3}, {4, 5, 6}}"),
    ("FindGraphPartition[CycleGraph[9], 3]", pred(lambda r: [len(p.args) for p in r.args] == [3, 3, 3],
                                                  "three parts of 3")),
    ("FindGraphPartition[" + DOC_PART + ", {2, 2}]", Raises(ValueError)),
    ("FindGraphPartition[" + DOC_PART + ", 0]", Raises(ValueError)),
    ("FindGraphPartition[\"g\"]", Raises(TypeError)),
    ("FindGraphPartition[h2, 2]", "FindGraphPartition[h2, 2]"),

    # ── LambdaComponents (documented examples) ──
    ("LambdaComponents[" + DOC_RULES + "]", "{{1}, {2}, {3}, {4}, {5}, {6}, {3, 4}, {1, 2, 3, 4, 5, 6}}"),
    ("LambdaComponents[" + DOC_LAMBDA8 + "]",
     "{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {1, 2, 3, 4}, {5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}}"),
    ("LambdaComponents[" + DOC_LAMBDA8 + ", {5}]", "{{5}, {5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}}"),
    ("LambdaComponents[3]", Raises(TypeError)),
    ("LambdaComponents[" + DOC_RULES + ", {9}]", Raises(ValueError)),
    ("LambdaComponents[h3]", "LambdaComponents[h3]"),

    # ── LuccioSamiComponents (documented example; LS sets are lambda sets) ──
    ("LuccioSamiComponents[" + DOC_RULES + "]", "{{1}, {2}, {3}, {4}, {5}, {6}, {1, 2, 3, 4, 5, 6}}"),
    ("MemberQ[LuccioSamiComponents[" + DOC_LAMBDA8 + "], {5, 6, 7, 8}]", "True"),
    ("SubsetQ[LambdaComponents[" + DOC_LAMBDA8 + "], LuccioSamiComponents[" + DOC_LAMBDA8 + "]]", "True"),
    ("LuccioSamiComponents[" + DOC_LAMBDA8 + ", {1}]", "{{1}, {1, 2, 3, 4}, {1, 2, 3, 4, 5, 6, 7, 8}}"),
    ("LuccioSamiComponents[\"x\"]", Raises(TypeError)),
    ("LuccioSamiComponents[" + DOC_RULES + ", {0}]", Raises(ValueError)),
    ("LuccioSamiComponents[h4]", "LuccioSamiComponents[h4]"),

    # ── CommunityGraphPlot, CommunityLabels, CommunityRegionStyle, CommunityBoundaryStyle ──
    ("CommunityGraphPlot[" + G6 + "]", pred(lambda g: len(prims(g, "Polygon")) == 2 and len(disks(g)) == 6,
                                            "two community regions")),
    ("CommunityGraphPlot[" + G6 + ", {{1, 2, 3}, {4, 5, 6}}, CommunityLabels -> {\"a\", \"b\"}]",
     pred(lambda g: sorted(py(t.args[0]) for t in prims(g, "Text")) == ["a", "b"], "labels")),
    ("CommunityGraphPlot[" + G6 + ", {{1, 2, 3}, {4, 5, 6}}, CommunityRegionStyle -> Red, "
     "CommunityBoundaryStyle -> Blue]",
     pred(lambda g: len([e for e in prims(g, "EdgeForm") if e.args and isinstance(e.args[0], Expr) and
                         e.args[0].head == "RGBColor"]) == 2, "styled regions")),
    ("CommunityGraphPlot[" + G6 + ", {{1, 2, 3}, {4, 5, 6}}, CommunityRegionStyle -> None, "
     "CommunityBoundaryStyle -> None]", pred(lambda g: len(prims(g, "Polygon")) == 2, "unstyled regions")),
    ("CommunityGraphPlot[" + G6 + ", {{1, 2, 3}, {3, 4, 5, 6}}]", Raises(NotImplementedError)),
    ("CommunityGraphPlot[" + G6 + ", {{1, 9}}]", Raises(ValueError)),
    ("CommunityGraphPlot[" + G6 + ", {{1}}, CommunityLabels -> Placed[{\"a\"}, Center]]",
     Raises(NotImplementedError)),
    ("CommunityGraphPlot[" + G6 + ", {{1}}, CommunityRegionStyle -> \"Pastel\"]", Raises(NotImplementedError)),
    ("CommunityGraphPlot[2]", Raises(TypeError)),
    ("CommunityGraphPlot[cg]", "CommunityGraphPlot[cg]"),
    ("CommunityLabels[1]", "CommunityLabels[1]"),
    ("CommunityRegionStyle[1]", "CommunityRegionStyle[1]"),
    ("CommunityBoundaryStyle[1]", "CommunityBoundaryStyle[1]"),

    # ── GraphValuePlot, EdgeValueRange, EdgeValueSizes ──
    ("GraphValuePlot[" + G6 + ", {1, 2, 3, 4, 5, 6}]",
     pred(lambda g: len(disks(g)) == 6 and len({tuple(py(c.args)) for c in prims(g, "RGBColor")}) >= 6,
          "vertex colours from the values")),
    ("GraphValuePlot[" + G6 + ", {1, 2, 3, 4, 5, 6} -> \"VertexSize\"]",
     pred(lambda g: [num(d.args[1]) for d in prims(g, "Disk")] == sorted(num(d.args[1]) for d in prims(g, "Disk")),
          "radii increase with the values")),
    ("GraphValuePlot[" + G6 + ", {1, 2, 3, 4, 5, 6, 7} -> \"EdgeThickness\", EdgeValueSizes -> {0.001, 0.01}]",
     pred(lambda g: [round(num(t.args[0]), 6) for t in prims(g, "Thickness")] ==
          [round(0.001 + 0.009 * k / 6, 6) for k in range(7)], "thickness 0.001 .. 0.01")),
    ("GraphValuePlot[" + G6 + ", {1, 2, 3, 4, 5, 6, 7}, EdgeValueRange -> {3, 5}]",
     pred(lambda g: len(prims(g, "Thickness")) == 7 and len(prims(g, "GrayLevel")) >= 4,
          "edges outside the range drawn gray")),
    ("GraphValuePlot[" + G6 + ", \"DegreeCentrality\"]",
     pred(lambda g: len(set(round(num(d.args[1]), 6) for d in prims(g, "Disk"))) == 2, "degree 2 vs 3 sizes")),
    ("GraphValuePlot[" + G6 + ", {a, b, a, b, c, c}]",
     pred(lambda g: len({tuple(py(c.args)) for c in prims(g, "RGBColor")}) >= 3, "categorical filling")),
    ("GraphValuePlot[" + G6 + ", {1, 2, 3}]", Raises(ValueError)),
    ("GraphValuePlot[" + G6 + ", {1, 2, 3, 4, 5, 6} -> \"VertexShape\"]", Raises(NotImplementedError)),
    ("GraphValuePlot[" + G6 + ", {1, 2, 3, 4, 5, 6} -> \"EdgeColor\"]", Raises(ValueError)),
    ("GraphValuePlot[" + G6 + ", \"Nonsense\"]", Raises(ValueError)),
    ("GraphValuePlot[" + G6 + ", 5]", Raises(TypeError)),
    ("GraphValuePlot[gv, {1}]", "GraphValuePlot[gv, {1}]"),
    ("EdgeValueRange[1]", "EdgeValueRange[1]"),
    ("EdgeValueSizes[1]", "EdgeValueSizes[1]"),
    ("GraphValuePlot[" + G6 + ", {1, 2, 3, 4, 5, 6, 7}, EdgeValueSizes -> {0.02, 0.01}]", Raises(ValueError)),
    ("GraphValuePlot[" + G6 + ", {1, 2, 3, 4, 5, 6, 7}, EdgeValueRange -> 3]", Raises(ValueError)),
]

ALL_CASES = CASES + NX_CASES


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_x07(case):
    check(case)


@pytest.mark.skipif(not HAVE_NX, reason="needs networkx")
@pytest.mark.parametrize("case", NX_CASES, ids=ids(NX_CASES))
def test_x07_networkx(case):
    check(case)


CASES = ALL_CASES   # the coverage meta-test counts every case


# ───────────────────────────── independent cross-checks ─────────────────────────────

@pytest.mark.skipif(not HAVE_NX, reason="needs networkx")
def test_x07_modularity():
    """Check: the modularity of the returned communities equals NetworkX's computation and is optimal among the
    methods' answers for the documented graph."""
    import networkx as nx
    G = nx.Graph([(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (7, 6), (8, 6), (5, 7), (7, 8)])
    got = [set(py(c)) for c in ev("FindGraphCommunities[" + DOC_COMM + "]").args]
    q = nx.algorithms.community.modularity(G, got)
    best = max(nx.algorithms.community.modularity(G, c) for c in
               [got, [{1, 2, 3, 4}, {5, 6, 7, 8}], [{1, 2, 3}, {4, 5, 6, 7, 8}]])
    assert abs(q - best) < 1e-12 and q > 0.3


@pytest.mark.skipif(not HAVE_NX, reason="needs networkx")
def test_x07_partition_is_minimum_cut():
    """Check: on a planted graph the exhaustive partition attains the brute-force minimum cut."""
    import networkx as nx
    G = nx.barbell_graph(4, 1)
    got = ev("FindGraphPartition[Graph[{" + ", ".join(f"{a}<->{b}" for a, b in G.edges()) + "}], {4, 5}]")
    parts = [set(py(p)) for p in got.args]
    cut = nx.cut_size(G, parts[0], parts[1])
    best = min(nx.cut_size(G, set(s), set(G) - set(s)) for s in itertools.combinations(G, 4))
    assert cut == best


@pytest.mark.skipif(not HAVE_NX, reason="needs networkx")
def test_x07_lambda_sets_definition():
    """Check: every returned lambda set S satisfies lambda(a, b) > lambda(a, c) for a, b in S, c outside."""
    import networkx as nx
    edges = [(1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (2, 5), (5, 6), (6, 7), (5, 8), (6, 8), (7, 8), (4, 7)]
    G = nx.Graph(edges)
    lam = {(a, b): nx.edge_connectivity(G, a, b) for a in G for b in G if a != b}
    for S in ev("LambdaComponents[" + DOC_LAMBDA8 + "]").args:
        S = set(py(S))
        out = set(G) - S
        if len(S) < 2 or not out:
            continue
        assert min(lam[a, b] for a in S for b in S if a != b) > max(lam[a, c] for a in S for c in out)


def test_x07_self_checks_raise_arithmetic_error():
    """Check: the layout and partition self-checks raise ArithmeticError on corrupted results."""
    import numpy as np
    from mikoshilang.library import wl_graphplots as GP
    with pytest.raises(ArithmeticError):
        GP.check_layout(np.array([[0.0, np.nan]]), 1, 2, "GraphPlot")
    with pytest.raises(ArithmeticError):
        GP._check_partition([[0, 1], [1]], 3, "FindGraphPartition")


@pytest.mark.skipif(not HAVE_MPL, reason="needs matplotlib")
def test_x07_fullgraphics_matches_renderer():
    """Check: FullGraphics' explicit primitives render over the range the renderer used, and its tick positions are
    the renderer's own tick positions."""
    from mikoshilang.library import wl_graphics_render as RD
    from mikoshilang.library import wl_plot_options as PO
    g = ev("Graphics[{Disk[{1, 2}, 3]}, Frame -> True]")
    fig = RD.render_figure(g)
    xlim = fig.axes[0].get_xlim()
    full = ev("FullGraphics[Graphics[{Disk[{1, 2}, 3]}, Frame -> True]]")
    labels = [t for t in prims(full, "Text")]
    xs = sorted({round(point(t.args[1])[0], 9) for t in labels if t.args[2].args[1] == 1})
    assert xs == [round(p, 9) for p in PO.tick_positions(xlim[0], xlim[1], 6)]
    assert RD.render_png(full)


def test_x07_skipped_symbols():
    from mikoshilang.library import STATUS
    from mikoshilang.library.wl_graphplots import SKIPPED
    assert set(SKIPPED) == {"StatusCentrality"}
    assert not any(n in STATUS for n in SKIPPED)
