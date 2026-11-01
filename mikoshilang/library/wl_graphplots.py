"""Wolfram batch X07: graph plots, communities and partitions, tree and fractal plots.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).

Graph drawings are computed with explicit, published layout methods; the coordinates are this module's own and are
*not* claimed to match the Wolfram Language's pixel for pixel:

* ``"SpringElectricalEmbedding"`` (the default of GraphPlot / GraphPlot3D): the Fruchterman-Reingold
  spring-electrical model (Fruchterman & Reingold 1991; adaptive step length of Hu 2005), natural length 1,
  300 iterations from a deterministic golden-angle spiral start, so the same graph always gives the same picture;
* ``"SpringEmbedding"``: stress majorization (SMACOF, Gansner, Koren & North 2004) of the graph distances, started
  from classical multidimensional scaling;
* ``"SpectralEmbedding"``: the Laplacian eigenvectors of the second and third smallest eigenvalues (Hall 1970);
* ``"LayeredDigraphEmbedding"`` / ``"LayeredEmbedding"`` (LayeredGraphPlot): the Sugiyama framework (Sugiyama,
  Tagawa & Toda 1981): cycle removal by depth-first search, longest-path layering, dummy vertices on long edges,
  barycentre crossing reduction and an averaging coordinate pass;
* ``"LayeredEmbedding"`` for trees, TreePlot and TreeGamePlot: the Reingold-Tilford tidy tree drawing (Reingold &
  Tilford 1981) of a breadth-first spanning tree; ``"RadialEmbedding"`` places depth k on the circle of radius k;
* ``"CircularEmbedding"``, ``"StarEmbedding"``, ``"GridEmbedding"``, ``"LinearEmbedding"``,
  ``"BipartiteEmbedding"``: closed-form placements stated in each head's notes.

Disconnected graphs are laid out component by component and the components are packed left to right, largest
first. Community detection uses NetworkX (an optional dependency: without it the heads raise
NotImplementedError).
"""
from __future__ import annotations

import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, S, is_numeric_value
from .wl_common import ev, apply, is_list, is_rule
from . import wl_graphs as GR
from . import wl_graphics_core as G
from .wl_graphics_color import parse_color, K_input

AREA = "graphs III: graph plots, communities, tree and fractal plots (X07)"

DEFAULT_COLORS = ((0.368417, 0.506779, 0.709798), (0.880722, 0.611041, 0.142051), (0.560181, 0.691569, 0.194885),
                  (0.922526, 0.385626, 0.209179), (0.528488, 0.470624, 0.701351), (0.772079, 0.431554, 0.102387),
                  (0.363898, 0.618501, 0.782349), (1.0, 0.75, 0.0), (0.647624, 0.37816, 0.614037),
                  (0.571589, 0.586483, 0.0))
VERTEX_COLOR = (0.368417, 0.506779, 0.709798)
EDGE_COLOR = (0.4, 0.4, 0.4)
HIGHLIGHT_COLOR = (1.0, 0.0, 0.0)


def _np():
    import numpy as np
    return np


def _need_nx(fname):
    try:
        import networkx  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised on the minimal install
        raise NotImplementedError(f"{fname}: this needs the optional NetworkX package "
                                  f"(pip install mikoshilang[all]): {exc}")
    import networkx as nx
    return nx


def sym(x, *names):
    return isinstance(x, Symbol) and (not names or x.name in names)


def fl(x):
    v = float(x)
    return 0.0 if v == 0 else round(v, 6)


def rgb(c):
    return Expr("RGBColor", *[fl(v) for v in c])


def pt(p):
    return L(*[fl(v) for v in p])


def _opt_name(r):
    if isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and len(r.args) == 2:
        k = r.args[0]
        if isinstance(k, Symbol):
            return k.name
        if isinstance(k, str):
            return k
    return None


def _known_option(name):
    return name in GRAPH_OPTS or name in G.GRAPHICS_OPTIONS[2] or name in G.GRAPHICS_OPTIONS[3] or \
        name in EXTRA_OPTS


EXTRA_OPTS = {"LayerSizeFunction", "GraphLayers", "GraphLayerStyle", "CommunityBoundaryStyle", "CommunityLabels",
              "CommunityRegionStyle", "PlotLegends", "ColorFunction", "ColorFunctionScaling", "MaxIterations",
              "EscapeRadius", "ImageResolution", "PlotRangeClipping", "Appearance", "GameActionLabels",
              "GamePlayerLabels", "EdgeValueRange", "EdgeValueSizes", "VertexValueRange", "WorkingPrecision"}


def split_opts(args, fname):
    """(positional, {name: value} in order) with trailing options split off.  A trailing rule v -> pos whose
    left side is not an option name (LayeredGraphPlot[g, v -> Top]) stays positional."""
    args = list(args)
    opts = []
    while args and _opt_name(args[-1]) is not None:
        r = args[-1]
        if not _known_option(_opt_name(r)) and isinstance(r.args[1], Symbol) and \
                r.args[1].name in ("Top", "Bottom", "Left", "Right", "Center"):
            break
        opts.insert(0, args.pop())
    out = {}
    for r in opts:
        out[_opt_name(r)] = r.args[1]
    return args, out


def _real(x, fname, what, lo=None, hi=None, lo_open=False):
    if isinstance(x, (bool, str)):
        raise TypeError(f"{fname}: {what} must be a real number, not {K_input(x)}")
    if not is_numeric_value(x):
        if isinstance(x, (Symbol, Expr)) and not is_list(x):
            raise Unevaluated()
        raise TypeError(f"{fname}: {what} must be a real number, not {K_input(x)}")
    v = S(x)
    if not v.is_real:
        raise ValueError(f"{fname}: {what} must be real (got {K_input(x)})")
    f = float(v)
    if lo is not None and (f < lo or (lo_open and f == lo)):
        raise ValueError(f"{fname}: {what} must be {'>' if lo_open else '>='} {lo} (got {K_input(x)})")
    if hi is not None and f > hi:
        raise ValueError(f"{fname}: {what} must be <= {hi} (got {K_input(x)})")
    return f


def _posint(x, fname, what, lo=1):
    if isinstance(x, (bool, str)):
        raise TypeError(f"{fname}: {what} must be an integer, not {K_input(x)}")
    if K.is_int(x):
        if int(x) < lo:
            raise ValueError(f"{fname}: {what} must be an integer >= {lo} (got {x})")
        return int(x)
    if is_numeric_value(x):
        raise ValueError(f"{fname}: {what} must be an integer (got {K_input(x)})")
    if isinstance(x, (Symbol, Expr)) and not is_list(x):
        raise Unevaluated()
    raise TypeError(f"{fname}: {what} must be an integer, not {K_input(x)}")


# ═══════════════════════════════════════ graph input ═══════════════════════════════════════

def _is_matrix(x):
    return is_list(x) and x.args and all(is_list(r) and len(r.args) == len(x.args) for r in x.args) and \
        all(is_numeric_value(v) for r in x.args for v in r.args)


def graph_input(x, fname, rules_directed=False, extra_opts=None):
    """WG for a Graph, a list of edges/rules (Rule is undirected unless ``rules_directed``) or an adjacency
    matrix.  Wrong kinds raise TypeError; a symbol stays unevaluated."""
    if isinstance(x, Symbol):
        raise Unevaluated()
    if isinstance(x, (bool, str, int, float, complex)) or isinstance(x, sp.Basic):
        raise TypeError(f"{fname}: expected a graph, a list of edges or an adjacency matrix, not {K_input(x)}")
    if _is_matrix(x):
        x = ev(Expr("AdjacencyGraph", x))
        if not (isinstance(x, Expr) and x.head == "Graph"):
            raise ValueError(f"{fname}: the matrix is not a valid adjacency matrix")
    elif is_list(x):
        if not x.args:
            raise ValueError(f"{fname}: the edge list is empty")
        es = []
        for e in x.args:
            if isinstance(e, Expr) and e.head in ("Rule", "DirectedEdge", "UndirectedEdge", "TwoWayRule") \
                    and len(e.args) == 2:
                if e.head == "Rule":
                    e = Expr("DirectedEdge" if rules_directed else "UndirectedEdge", *e.args)
                es.append(e)
            else:
                raise TypeError(f"{fname}: {K_input(e)} is not an edge (use u <-> v, u -> v, "
                                "UndirectedEdge or DirectedEdge)")
        x = Expr("Graph", L(*es))
    elif not (isinstance(x, Expr) and x.head == "Graph"):
        if isinstance(x, Expr) and not is_list(x):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a graph, not {K_input(x)}")
    try:
        gr = GR.graph_of(x, fname)
    except GR.GraphError as exc:
        raise ValueError(str(exc)) from None
    return gr


def _directed_kind(gr, fname):
    ds = {d for _, _, d in gr.edges}
    if len(ds) > 1:
        raise NotImplementedError(f"{fname}: mixed graphs (directed and undirected edges together) are not "
                                  "supported")
    return bool(ds) and ds.pop()


def _undirected_adj(gr):
    nb = [[] for _ in range(gr.n)]
    for i, j, _ in gr.edges:
        if i != j:
            if j not in nb[i]:
                nb[i].append(j)
            if i not in nb[j]:
                nb[j].append(i)
    for a in nb:
        a.sort()
    return nb


def _components(nb):
    n = len(nb)
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        comp, stack = [], [s]
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in nb[u]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(sorted(comp))
    return comps


def _bfs(nb, s):
    dist = {s: 0}
    order = [s]
    parent = {s: None}
    k = 0
    while k < len(order):
        u = order[k]
        k += 1
        for w in nb[u]:
            if w not in dist:
                dist[w] = dist[u] + 1
                parent[w] = u
                order.append(w)
    return dist, order, parent


def _center(nb, comp):
    """Vertex of minimum eccentricity in a component (first in VertexList order on ties)."""
    best, be = None, None
    for v in comp:
        d, _, _ = _bfs(nb, v)
        e = max(d.values())
        if be is None or e < be:
            best, be = v, e
    return best


# ═══════════════════════════════════════ layouts ═══════════════════════════════════════

def _spiral(n, dim):
    """Deterministic, well-spread start positions (golden-angle spiral / Fibonacci sphere), scale ~ sqrt(n)."""
    np = _np()
    k = np.arange(n, dtype=float)
    ga = math.pi * (3 - math.sqrt(5))
    if dim == 2:
        r = np.sqrt(k + 0.5)
        return np.column_stack([r * np.cos(k * ga), r * np.sin(k * ga)])
    z = 1 - 2 * (k + 0.5) / max(n, 1)
    rr = np.sqrt(np.maximum(0.0, 1 - z * z))
    s = max(1.0, n ** (1 / 3))
    return s * np.column_stack([rr * np.cos(k * ga), rr * np.sin(k * ga), z])


def spring_electrical(n, pairs, dim=2, iters=300, weights=None):
    """Fruchterman-Reingold spring-electrical layout of a connected graph (Hu 2005 adaptive step)."""
    np = _np()
    if n == 1:
        return np.zeros((1, dim))
    X = _spiral(n, dim)
    if not pairs:
        return X
    I = np.array([p[0] for p in pairs])
    J = np.array([p[1] for p in pairs])
    W = np.ones(len(pairs)) if weights is None else np.array(weights, dtype=float)
    Kn = 1.0
    step = Kn * max(1.0, math.sqrt(n)) / 5.0
    energy0, progress = float("inf"), 0
    for _ in range(iters):
        D = X[:, None, :] - X[None, :, :]
        d2 = (D ** 2).sum(-1)
        np.fill_diagonal(d2, 1.0)
        d2 = np.maximum(d2, 1e-9)
        F = (D * (Kn * Kn / d2)[:, :, None]).sum(1)          # repulsion C K^2 / d along the unit vector
        dv = X[I] - X[J]
        dl = np.sqrt((dv ** 2).sum(-1))
        fa = (dv * (W * dl / Kn)[:, None])                     # attraction d^2 / K
        np.subtract.at(F, I, fa)
        np.add.at(F, J, fa)
        fn_ = np.sqrt((F ** 2).sum(-1))
        energy = float((fn_ ** 2).sum())
        X = X + step * F / np.maximum(fn_, 1e-12)[:, None]
        if energy < energy0:
            progress += 1
            if progress >= 5:
                progress = 0
                step /= 0.9
        else:
            progress = 0
            step *= 0.9
        energy0 = energy
        if step < 1e-4 * Kn:
            break
    return X


def _apsp(nb, comp):
    idx = {v: k for k, v in enumerate(comp)}
    np = _np()
    D = np.zeros((len(comp), len(comp)))
    for v in comp:
        d, _, _ = _bfs(nb, v)
        for w, x in d.items():
            D[idx[v], idx[w]] = x
    return D


def spring_stress(nb, comp, dim=2, iters=200):
    """Stress majorization (SMACOF) of the graph distances, classical-MDS start."""
    np = _np()
    n = len(comp)
    if n == 1:
        return np.zeros((1, dim))
    D = _apsp(nb, comp)
    Jm = np.eye(n) - 1.0 / n
    B = -0.5 * Jm @ (D ** 2) @ Jm
    w_, V = np.linalg.eigh(B)
    order = np.argsort(w_)[::-1][:dim]
    X = V[:, order] * np.sqrt(np.maximum(w_[order], 1e-9))
    X = X + 1e-6 * _spiral(n, dim)
    Wt = np.zeros_like(D)
    nz = D > 0
    Wt[nz] = D[nz] ** -2
    Lw = -Wt.copy()
    np.fill_diagonal(Lw, Wt.sum(1))
    Lp = np.linalg.pinv(Lw)
    for _ in range(iters):
        E = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        Bz = np.zeros_like(D)
        m = E > 1e-12
        Bz[m] = -Wt[m] * D[m] / E[m]
        np.fill_diagonal(Bz, 0.0)
        np.fill_diagonal(Bz, -Bz.sum(1))
        Xn = Lp @ (Bz @ X)
        if np.abs(Xn - X).max() < 1e-7:
            X = Xn
            break
        X = Xn
    return X


def spectral(nb, comp, dim=2):
    np = _np()
    n = len(comp)
    if n == 1:
        return np.zeros((1, dim))
    idx = {v: k for k, v in enumerate(comp)}
    Lm = np.zeros((n, n))
    for v in comp:
        for w in nb[v]:
            Lm[idx[v], idx[w]] = -1.0
    np.fill_diagonal(Lm, -Lm.sum(1))
    w_, V = np.linalg.eigh(Lm)
    cols = [V[:, k] if k < n else np.zeros(n) for k in range(1, dim + 1)]
    X = np.column_stack(cols)
    for k in range(dim):                                    # fix the eigenvector sign (largest |entry| positive)
        c = X[:, k]
        j = int(np.argmax(np.abs(c)))
        if c[j] < 0:
            X[:, k] = -c
    s = np.abs(X).max()
    return X * (math.sqrt(n) / s if s > 0 else 1.0)


def circular(n):
    np = _np()
    if n == 1:
        return np.zeros((1, 2))
    t = math.pi / 2 - 2 * math.pi * np.arange(n) / n
    return np.column_stack([np.cos(t), np.sin(t)])


def pack(blocks, dim):
    """Place component layouts (lists of (vertex indices, coordinates)) left to right, largest first."""
    np = _np()
    blocks = sorted(blocks, key=lambda b: (-len(b[0]), b[0][0]))
    out = {}
    x0 = 0.0
    for verts, X in blocks:
        X = np.array(X, dtype=float)
        X = X - X.min(0)
        if len(verts) == 1:
            X = np.zeros((1, dim))
        X[:, 0] += x0
        for v, p in zip(verts, X):
            out[v] = p
        x0 = X[:, 0].max() + 1.0
    return out


# ─────────────────────────── Sugiyama layered drawing ───────────────────────────

def _crossings(layers, down):
    """Number of edge crossings between consecutive layers (down[u] = successors of u in the next layer)."""
    total = 0
    for li in range(len(layers) - 1):
        pos = {v: k for k, v in enumerate(layers[li + 1])}
        es = [(a, pos[b]) for a, u in enumerate(layers[li]) for b in down.get(u, ()) if b in pos]
        for x in range(len(es)):
            for y in range(x + 1, len(es)):
                (a1, b1), (a2, b2) = es[x], es[y]
                if (a1 - a2) * (b1 - b2) < 0:
                    total += 1
    return total


def sugiyama(n, arcs, roots=()):
    """Layered layout of vertices 0..n-1 with directed arcs (u, v) (u above v).

    Returns (coords dict v -> (x, layer), routed arcs [(u, v, [dummy points])]).  ``roots`` are forced onto
    layer 0."""
    arcs = [(u, v) for u, v in arcs if u != v]
    succ = {v: [] for v in range(n)}
    for u, v in arcs:
        succ[u].append(v)
    # cycle removal: reverse the back edges of a depth-first search (sources first, then VertexList order)
    indeg = [0] * n
    for u, v in arcs:
        indeg[v] += 1
    state = [0] * n
    back = set()
    starts = list(roots) + [v for v in range(n) if indeg[v] == 0] + list(range(n))
    for s in starts:
        if state[s]:
            continue
        stack = [(s, iter(succ[s]))]
        state[s] = 1
        while stack:
            u, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                state[u] = 2
                stack.pop()
            elif state[nxt] == 1:
                back.add((u, nxt))
            elif state[nxt] == 0:
                state[nxt] = 1
                stack.append((nxt, iter(succ[nxt])))
    dag = [(v, u) if (u, v) in back else (u, v) for u, v in arcs]
    pred = {v: [] for v in range(n)}
    for u, v in dag:
        pred[v].append(u)
    dsucc = {v: [] for v in range(n)}
    for u, v in dag:
        dsucc[u].append(v)
    layer = {}
    deg = [len(pred[v]) for v in range(n)]
    queue = [v for v in range(n) if deg[v] == 0]
    topo = []
    k = 0
    while k < len(queue):
        v = queue[k]
        k += 1
        topo.append(v)
        for w in dsucc[v]:
            deg[w] -= 1
            if deg[w] == 0:
                queue.append(w)
    if len(topo) != n:
        raise ArithmeticError("LayeredGraphPlot: internal check failed (cycle removal left a cycle)")
    for v in topo:
        ps = [layer[p] for p in pred[v]]
        layer[v] = 0 if not ps else max(ps) + 1
    for r in roots:
        if layer[r] != 0:
            raise NotImplementedError("LayeredGraphPlot: the requested root vertex is not a source of the graph")
    # dummy vertices
    nodes = n
    down = {}
    routed = []
    for u, v in dag:
        chain = [u]
        for _ in range(layer[u] + 1, layer[v]):
            layer[nodes] = layer[chain[-1]] + 1
            chain.append(nodes)
            nodes += 1
        chain.append(v)
        for a, b in zip(chain, chain[1:]):
            down.setdefault(a, []).append(b)
        routed.append((chain[0], chain[-1], chain[1:-1]))
    nl = max(layer.values()) + 1 if layer else 1
    layers = [[] for _ in range(nl)]
    for v in range(nodes):
        layers[layer[v]].append(v)
    up = {}
    for a, bs in down.items():
        for b in bs:
            up.setdefault(b, []).append(a)
    best = [list(l) for l in layers]
    bestc = _crossings(best, down)
    for sweep in range(24):
        rng = range(1, nl) if sweep % 2 == 0 else range(nl - 2, -1, -1)
        for li in rng:
            ref = layers[li - 1] if sweep % 2 == 0 else layers[li + 1]
            nbr = up if sweep % 2 == 0 else down
            pos = {v: k for k, v in enumerate(ref)}
            cur = {v: k for k, v in enumerate(layers[li])}

            def bary(v):
                ps = [pos[w] for w in nbr.get(v, ()) if w in pos]
                return (sum(ps) / len(ps)) if ps else cur[v]
            layers[li] = sorted(layers[li], key=lambda v: (bary(v), cur[v]))
        c = _crossings(layers, down)
        if c < bestc:
            best, bestc = [list(l) for l in layers], c
        if bestc == 0:
            break
    layers = best
    x = {}
    for l in layers:
        w = len(l)
        for k, v in enumerate(l):
            x[v] = k - (w - 1) / 2.0
    for _ in range(8):                                    # averaging pass keeping order and unit separation
        for l in layers:
            want = []
            for v in l:
                ns = up.get(v, []) + down.get(v, [])
                want.append(sum(x[w] for w in ns) / len(ns) if ns else x[v])
            # place left to right respecting separation 1, then centre the deviation
            pos = []
            for k, wv in enumerate(want):
                pos.append(wv if not pos else max(wv, pos[-1] + 1.0))
            shift = sum(w - p for w, p in zip(want, pos)) / len(pos) if pos else 0.0
            pos = [p + shift for p in pos]
            for k in range(len(pos) - 2, -1, -1):
                if pos[k] > pos[k + 1] - 1.0:
                    pos[k] = pos[k + 1] - 1.0
            for v, p in zip(l, pos):
                x[v] = p
    coords = {v: (x[v], layer[v]) for v in range(nodes)}
    return coords, routed, bestc


# ─────────────────────────── Reingold-Tilford trees ───────────────────────────

def tidy_tree(children, root):
    """Reingold-Tilford tidy drawing: dict v -> (x, depth); siblings at least 1 apart, parents centred over their
    first and last child, subtrees placed as close as their contours allow."""
    pos = {}
    depth = {root: 0}
    order = [root]
    k = 0
    while k < len(order):
        u = order[k]
        k += 1
        for c in children.get(u, ()):
            depth[c] = depth[u] + 1
            order.append(c)
    contour = {}

    for u in reversed(order):
        kids = children.get(u, ())
        if not kids:
            pos[u] = 0.0
            contour[u] = {depth[u]: (0.0, 0.0)}
            continue
        merged = None
        shifts = []
        for c in kids:
            cc = contour[c]
            if merged is None:
                s = 0.0
                merged = dict(cc)
            else:
                s = max(merged[d][1] - cc[d][0] + 1.0 for d in cc if d in merged)
                for d, (lo, hi) in cc.items():
                    if d in merged:
                        merged[d] = (merged[d][0], hi + s)
                    else:
                        merged[d] = (lo + s, hi + s)
            shifts.append(s)
        for c, s in zip(kids, shifts):
            _shift(children, c, s, pos)
        mid = (pos[kids[0]] + pos[kids[-1]]) / 2.0
        for c in kids:
            _shift(children, c, -mid, pos)
        merged = {d: (lo - mid, hi - mid) for d, (lo, hi) in merged.items()}
        pos[u] = 0.0
        merged[depth[u]] = (0.0, 0.0)
        contour[u] = merged
        for c in kids:
            del contour[c]
    return {v: (pos[v], depth[v]) for v in order}


def _shift(children, v, s, pos):
    if s == 0:
        return
    stack = [v]
    while stack:
        u = stack.pop()
        pos[u] += s
        stack.extend(children.get(u, ()))


def radial_tree(children, root):
    """Radial layout: depth k on the circle of radius k, each subtree in an angular wedge proportional to its
    number of leaves."""
    leaves = {}

    def count(u):
        stack = [(u, False)]
        while stack:
            x, done = stack.pop()
            if done:
                ks = children.get(x, ())
                leaves[x] = sum(leaves[c] for c in ks) if ks else 1
            else:
                stack.append((x, True))
                stack.extend((c, False) for c in children.get(x, ()))
    count(root)
    out = {root: (0.0, 0.0)}
    stack = [(root, 0.0, 2 * math.pi, 0)]
    while stack:
        u, a0, a1, d = stack.pop()
        ks = children.get(u, ())
        tot = sum(leaves[c] for c in ks)
        a = a0
        for c in ks:
            w = (a1 - a0) * leaves[c] / tot
            th = math.pi / 2 - (a + w / 2)
            out[c] = ((d + 1) * math.cos(th), (d + 1) * math.sin(th))
            stack.append((c, a, a + w, d + 1))
            a += w
    return out


def spanning_children(nb, root):
    _, order, parent = _bfs(nb, root)
    ch = {}
    for v in order[1:]:
        ch.setdefault(parent[v], []).append(v)
    return ch, order


_POSITIONS = ("Top", "Bottom", "Left", "Right")


def orient(xy, where):
    """(x, layer) -> 2-D point with the root layer at ``where``."""
    x, l = xy
    if where == "Top":
        return (x, -l)
    if where == "Bottom":
        return (x, l)
    if where == "Left":
        return (l, -x)
    if where == "Right":
        return (-l, -x)
    raise ValueError(where)


# ─────────────────────────── layout dispatcher ───────────────────────────

LAYOUTS_2D = ("SpringElectricalEmbedding", "SpringEmbedding", "SpectralEmbedding", "CircularEmbedding",
              "StarEmbedding", "GridEmbedding", "LinearEmbedding", "BipartiteEmbedding", "LayeredDigraphEmbedding",
              "LayeredEmbedding", "RadialEmbedding")
LAYOUTS_3D = ("SpringElectricalEmbedding", "SpringEmbedding", "SpectralEmbedding", "LayeredDigraphEmbedding")
_KNOWN_LAYOUTS = LAYOUTS_2D + ("BalloonEmbedding", "CircularMultipartiteEmbedding", "DiscreteSpiralEmbedding",
                               "MultipartiteEmbedding", "SpiralEmbedding", "HighDimensionalEmbedding",
                               "PlanarEmbedding", "TutteEmbedding", "SpringElectricalEmbedding", "GravityEmbedding",
                               "RandomEmbedding", "SymmetricLayeredEmbedding", "BalloonEmbedding")


class Layout:
    """A layout request: method name, dimension, root vertex, orientation."""

    def __init__(self, method="SpringElectricalEmbedding", dim=2, root=None, where="Top"):
        self.method, self.dim, self.root, self.where = method, dim, root, where


def parse_layout(spec, fname, dim, default="SpringElectricalEmbedding"):
    lay = Layout(default, dim)
    if spec is None or sym(spec, "Automatic"):
        return lay
    items = list(spec.args) if is_list(spec) else [spec]
    if items and isinstance(items[0], str):
        name = items.pop(0)
        if name not in _KNOWN_LAYOUTS:
            raise ValueError(f"{fname}: {name!r} is not a known GraphLayout")
        lay.method = name
    for it in items:
        k = _opt_name(it)
        if k is None:
            raise ValueError(f"{fname}: invalid GraphLayout setting {K_input(spec)}")
        v = it.args[1]
        if k == "Dimension":
            d = _posint(v, fname, "the GraphLayout \"Dimension\"")
            if d not in (2, 3):
                raise ValueError(f"{fname}: GraphLayout \"Dimension\" must be 2 or 3")
        elif k == "RootVertex":
            lay.root = v
        elif k == "Orientation":
            if not sym(v, *_POSITIONS):
                raise ValueError(f"{fname}: \"Orientation\" must be Top, Bottom, Left or Right")
            lay.where = v.name
        else:
            raise NotImplementedError(f"{fname}: the GraphLayout sub-option {k!r} is not supported")
    allowed = LAYOUTS_2D if dim == 2 else LAYOUTS_3D
    if lay.method not in allowed:
        raise NotImplementedError(f"{fname}: GraphLayout {lay.method!r} is not supported "
                                  f"(supported: {', '.join(allowed)})")
    return lay


def _vertex_coordinates(gr, vc, fname, dim):
    np = _np()
    if is_list(vc) and vc.args and all(is_rule(r) for r in vc.args):
        got = {K.tag(r.args[0]): r.args[1] for r in vc.args}
        pts = []
        for v in gr.vertices:
            if K.tag(v) not in got:
                raise NotImplementedError(f"{fname}: VertexCoordinates gives no position for {K_input(v)}")
            pts.append(got[K.tag(v)])
    elif is_list(vc):
        pts = list(vc.args)
        if len(pts) != gr.n:
            raise ValueError(f"{fname}: VertexCoordinates must give one point per vertex")
    else:
        raise ValueError(f"{fname}: VertexCoordinates must be a list of points or rules")
    out = []
    for p in pts:
        if not (is_list(p) and len(p.args) == dim):
            raise ValueError(f"{fname}: VertexCoordinates must be {dim}-dimensional points (got {K_input(p)})")
        out.append([_real(c, fname, "a vertex coordinate") for c in p.args])
    return np.array(out, dtype=float), {}


def _vindex(gr, v, fname):
    t = K.tag(v)
    if t not in gr.index:
        raise ValueError(f"{fname}: {K_input(v)} is not a vertex of the graph")
    return gr.index[t]


def compute_layout(gr, lay, fname):
    """(coords n x dim numpy array, routes {edge index: [intermediate points]}) for a parsed graph."""
    np = _np()
    n, dim = gr.n, lay.dim
    if n == 0:
        return np.zeros((0, dim)), {}
    nb = _undirected_adj(gr)
    m = lay.method
    if m == "LayeredDigraphEmbedding":
        return _layered(gr, lay, fname)
    if m == "LayeredEmbedding":
        return _tree_layout(gr, lay, fname)
    if m == "RadialEmbedding":
        return _tree_layout(gr, lay, fname, radial=True)
    if m == "CircularEmbedding":
        return circular(n), {}
    if m == "GridEmbedding":
        c = math.ceil(math.sqrt(n))
        return np.array([[k % c, -(k // c)] for k in range(n)], dtype=float), {}
    if m == "LinearEmbedding":
        return np.array([[k, 0] for k in range(n)], dtype=float), {}
    if m == "StarEmbedding":
        deg = [len(a) for a in nb]
        c = max(range(n), key=lambda v: (deg[v], -v))
        rest = [v for v in range(n) if v != c]
        X = np.zeros((n, 2))
        P = circular(len(rest)) if rest else np.zeros((0, 2))
        for v, p in zip(rest, P):
            X[v] = p
        return X, {}
    if m == "BipartiteEmbedding":
        col = {}
        for s in range(n):
            if s in col:
                continue
            col[s] = 0
            dist, order, _ = _bfs(nb, s)
            for v in order:
                col[v] = dist[v] % 2
        for i, j, _ in gr.edges:
            if col[i] == col[j]:
                raise ValueError(f"{fname}: GraphLayout \"BipartiteEmbedding\" needs a bipartite graph")
        X = np.zeros((n, 2))
        cnt = [0, 0]
        for v in range(n):
            X[v] = (cnt[col[v]], 1 - col[v])
            cnt[col[v]] += 1
        return X, {}
    blocks = []
    for comp in _components(nb):
        idx = {v: k for k, v in enumerate(comp)}
        if m == "SpringElectricalEmbedding":
            pairs = sorted({(min(idx[i], idx[j]), max(idx[i], idx[j])) for i, j, _ in gr.edges
                            if i != j and i in idx})
            X = spring_electrical(len(comp), pairs, dim)
        elif m == "SpringEmbedding":
            X = spring_stress(nb, comp, dim)
        else:
            X = spectral(nb, comp, dim)
        blocks.append((comp, X))
    P = pack(blocks, dim)
    return np.array([P[v] for v in range(n)], dtype=float), {}


def _layered(gr, lay, fname):
    np = _np()
    n = gr.n
    directed = _directed_kind(gr, fname)
    roots = []
    if lay.root is not None:
        roots = [_vindex(gr, lay.root, fname)]
    if directed:
        arcs = [(i, j) for i, j, _ in gr.edges]
        src = list(range(len(arcs)))
    else:
        nb = _undirected_adj(gr)
        dist = {}
        for comp in _components(nb):
            r = roots[0] if roots and roots[0] in comp else _center(nb, comp)
            d, _, _ = _bfs(nb, r)
            dist.update(d)
        arcs = []
        for i, j, _ in gr.edges:
            arcs.append((i, j) if dist[i] <= dist[j] else (j, i))
    same = [k for k, (i, j) in enumerate(arcs) if i == j or (not directed and dist[i] == dist[j])]
    use = [a for k, a in enumerate(arcs) if k not in same]
    try:
        coords, routed, _ = sugiyama(n, use, roots)
    except NotImplementedError as exc:
        raise NotImplementedError(str(exc).replace("LayeredGraphPlot", fname)) from None
    routes = {}
    ui = [k for k in range(len(arcs)) if k not in same]
    for k, (start, _end, dummies) in zip(ui, routed):
        pts = [coords[d] for d in dummies]
        routes[k] = pts if start == gr.edges[k][0] else pts[::-1]
    if lay.dim == 3:
        return _layered3d(n, coords, routes, lay.where)
    X = np.array([orient(coords[v], lay.where) for v in range(n)], dtype=float)
    routes = {k: [orient(p, lay.where) for p in pts] for k, pts in routes.items()}
    return X, routes


def _layered3d(n, coords, routes, where="Top"):
    """3-D layered drawing: layer l at height -l, the vertices of a layer on a circle in their Sugiyama order."""
    np = _np()
    by_layer = {}
    for v, (x, l) in coords.items():
        by_layer.setdefault(l, []).append((x, v))
    pos3 = {}
    for l, items in by_layer.items():
        items.sort()
        w = len(items)
        r = 0.0 if w == 1 else max(0.5, w / (2 * math.pi))
        for k, (_, v) in enumerate(items):
            t = 2 * math.pi * k / w
            pos3[v] = (r * math.cos(t), r * math.sin(t), -float(l) if where == "Top" else float(l))
    X = np.array([pos3[v] for v in range(n)], dtype=float)
    key = {}
    for v, (x, l) in coords.items():
        key[(round(x, 9), l)] = v
    routes3 = {k: [pos3[key[(round(p[0], 9), p[1])]] for p in pts] for k, pts in routes.items()}
    return X, routes3


def _tree_layout(gr, lay, fname, radial=False, where="Top"):
    np = _np()
    n = gr.n
    nb = _undirected_adj(gr)
    root = _vindex(gr, lay.root, fname) if lay.root is not None else None
    directed = _directed_kind(gr, fname)
    blocks = []
    for comp in _components(nb):
        if root is not None and root in comp:
            r = root
        else:
            r = None
            if directed:
                indeg = {v: 0 for v in comp}
                for i, j, _ in gr.edges:
                    if j in indeg and i != j:
                        indeg[j] += 1
                srcs = [v for v in comp if indeg[v] == 0]
                if len(srcs) == 1:
                    r = srcs[0]
            if r is None:
                r = _center(nb, comp)
        ch, order = spanning_children(nb, r)
        if radial:
            P = radial_tree(ch, r)
            X = [P[v] for v in comp]
        else:
            P = tidy_tree(ch, r)
            X = [orient(P[v], lay.where) for v in comp]
        blocks.append((comp, X))
    if len(blocks) == 1:
        comp, X = blocks[0]
        P = dict(zip(comp, X))
    else:
        P = pack(blocks, 2)
    return np.array([P[v] for v in range(n)], dtype=float), {}


def check_layout(X, n, dim, fname):
    """Self-check of every computed layout: one finite point per vertex."""
    np = _np()
    if X.shape != (n, dim) or not np.all(np.isfinite(X)):
        raise ArithmeticError(f"{fname}: result failed verification (the layout is not one finite point per "
                              "vertex)")


# ─────────────────────────── drawing ───────────────────────────

GRAPH_OPTS = ("GraphLayout", "VertexCoordinates", "VertexLabels", "VertexSize", "VertexStyle", "EdgeStyle",
              "EdgeLabels", "DirectedEdges", "GraphHighlight", "GraphHighlightStyle", "PlotStyle", "PlotTheme",
              "PerformanceGoal", "DataRange", "Method", "VertexShapeFunction", "EdgeShapeFunction", "VertexShape",
              "EdgeLabelStyle", "VertexLabelStyle", "EdgeWeight", "VertexWeight", "AnnotationRules",
              "MultiedgeStyle", "SelfLoopStyle", "EdgeCapacity", "EdgeCost", "VertexCapacity", "TaggedEdgeStyle")
_AUTO_ONLY = ("GraphHighlightStyle", "PlotStyle", "PlotTheme", "PerformanceGoal", "DataRange", "Method",
              "VertexShapeFunction", "EdgeShapeFunction", "VertexShape", "EdgeLabelStyle", "VertexLabelStyle",
              "MultiedgeStyle", "SelfLoopStyle", "TaggedEdgeStyle")
_DATA_OPTS = ("EdgeWeight", "VertexWeight", "AnnotationRules", "EdgeCapacity", "EdgeCost", "VertexCapacity")
_SIZES = {"Tiny": 0.1, "Small": 0.2, "Medium": 0.3, "Large": 0.5}

DRAW_NOTE = ("Drawing: edges are Line (Arrow for directed edges, shortened to the vertex disks; parallel edges bend "
             "through offset midpoints, self-loops are small closed loops), vertices are Disk (Sphere in 3-D) of "
             "radius VertexSize/2 times the smallest distance between vertices (VertexSize Automatic = 0.3; Tiny, "
             "Small, Medium, Large = 0.1, 0.2, 0.3, 0.5), styled with VertexStyle / EdgeStyle (a directive, or "
             "rules v -> style, e -> style); VertexLabels None, \"Name\", Automatic or rules v -> label and "
             "EdgeLabels None, \"Name\", \"EdgeWeight\" or rules e -> label draw Text; GraphHighlight -> {v, e, ...} "
             "draws those in red. VertexCoordinates gives the positions. PlotTheme, PlotStyle, VertexShape(Function), "
             "EdgeShapeFunction, label styles, Method, DataRange and GraphHighlightStyle accept only Automatic (other "
             "settings raise NotImplementedError); other options are Graphics options. Colours and sizes are this "
             "module's own, not the notebook front end's.")


def merged_options(gr, opts, fname, extra=()):
    """Graph's own presentation options overridden by the plot's options; returns (graph opts, graphics opts)."""
    allopts = {}
    for k, v in gr.opts.items():
        allopts[k] = v
    allopts.update(opts)
    gopts, gfx = {}, {}
    for k, v in allopts.items():
        if k in GRAPH_OPTS or k in extra:
            gopts[k] = v
        else:
            gfx[k] = v
    for k in _AUTO_ONLY:
        if k in gopts and not sym(gopts[k], "Automatic") and not (k == "PlotTheme" and sym(gopts[k], "None")):
            raise NotImplementedError(f"{fname}: the option {k} -> {K_input(gopts[k])} is not supported (only "
                                      "Automatic)")
    return gopts, gfx


def _graphics_rules(gfx, dim, fname):
    rules = [Expr("Rule", Symbol(k), v) for k, v in gfx.items()]
    try:
        G.check_options(rules, dim, "Graphics" if dim == 2 else "Graphics3D")
    except ValueError as exc:
        raise ValueError(f"{fname}: {exc}") from None
    return rules


def _style_of(x, fname, what):
    """A directive (colour, Thickness, Dashing, Opacity, Directive[...]) used as a style."""
    if isinstance(x, Expr) and x.head == "Directive":
        return x
    if parse_color(x) is not None or (isinstance(x, Expr) and x.head in G.DIRECTIVE_HEADS) or \
            (isinstance(x, Expr) and x.head in ("Thickness", "AbsoluteThickness", "Dashing", "AbsoluteDashing",
                                                "Opacity", "PointSize", "AbsolutePointSize", "EdgeForm",
                                                "FaceForm", "Arrowheads", "Dashed", "Dotted", "DotDashed")) or \
            sym(x, "Thick", "Thin", "Dashed", "Dotted", "DotDashed"):
        return x
    raise ValueError(f"{fname}: {what} must be a graphics directive, not {K_input(x)}")


def _per_item(spec, keys_match, n, fname, what, conv):
    """Expand a style/label spec to a per-item list: one value, or rules item -> value (others None)."""
    out = [None] * n
    if spec is None or sym(spec, "Automatic", "None"):
        return out
    if is_list(spec) and spec.args and all(is_rule(r) for r in spec.args) or is_rule(spec):
        rules = spec.args if is_list(spec) else (spec,)
        for r in rules:
            hit = [k for k in range(n) if keys_match(k, r.args[0])]
            if not hit:
                raise ValueError(f"{fname}: {K_input(r.args[0])} in {what} is not part of the graph")
            for k in hit:
                out[k] = conv(r.args[1])
        return out
    v = conv(spec)
    return [v] * n


def _edge_matches(gr, k, e, fname):
    if not (isinstance(e, Expr) and e.head in ("UndirectedEdge", "DirectedEdge", "Rule", "TwoWayRule")):
        return False
    i, j, d = gr.edges[k]
    try:
        a, b = gr.index[K.tag(e.args[0])], gr.index[K.tag(e.args[1])]
    except KeyError:
        return False
    if d != (e.head in ("DirectedEdge", "Rule")):
        return False
    return (i, j) == (a, b) or (not d and (j, i) == (a, b))


def _min_dist(X):
    np = _np()
    n = len(X)
    if n < 2:
        return 1.0
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    D[np.arange(n), np.arange(n)] = np.inf
    d = float(D.min())
    if not math.isfinite(d) or d < 1e-9:
        span = float((X.max(0) - X.min(0)).max()) if n else 0.0
        return span / max(n, 1) if span > 0 else 1.0
    return d


def draw_graph(gr, X, routes, gopts, fname, dim=2, vertex_colors=None, vertex_radii=None, edge_styles=None,
               vertex_extra=None, labels_override=None):
    """Graphics content (a list) drawing gr at coordinates X."""
    np = _np()
    n = gr.n
    dmin = _min_dist(X)
    vs = gopts.get("VertexSize", Symbol("Automatic"))
    if sym(vs, "Automatic"):
        frac = _SIZES["Medium"]
    elif sym(vs, *_SIZES):
        frac = _SIZES[vs.name]
    else:
        frac = _real(vs, fname, "VertexSize", lo=0)
    r = frac / 2 * dmin
    radii = vertex_radii or [r] * n
    v_style = _per_item(gopts.get("VertexStyle"), lambda k, x: K.tag(x) == K.tag(gr.vertices[k]), n, fname,
                        "VertexStyle", lambda x: _style_of(x, fname, "VertexStyle"))
    m = len(gr.edges)
    e_style = _per_item(gopts.get("EdgeStyle"), lambda k, e: _edge_matches(gr, k, e, fname), m, fname,
                        "EdgeStyle", lambda x: _style_of(x, fname, "EdgeStyle"))
    if edge_styles is not None:
        e_style = [es if es is not None else e for es, e in zip(edge_styles, e_style)]
    hl_v, hl_e = set(), set()
    hl = gopts.get("GraphHighlight")
    if hl is not None and not (is_list(hl) and not hl.args):
        for x in (hl.args if is_list(hl) else (hl,)):
            ks = [k for k in range(m) if _edge_matches(gr, k, x, fname)]
            if ks:
                hl_e.update(ks)
            elif K.tag(x) in gr.index:
                hl_v.add(gr.index[K.tag(x)])
            else:
                raise ValueError(f"{fname}: GraphHighlight item {K_input(x)} is not part of the graph")
    # edges
    pair_count, pair_seen = {}, {}
    for i, j, d in gr.edges:
        key = (min(i, j), max(i, j))
        pair_count[key] = pair_count.get(key, 0) + 1
    edges_out = []
    for k, (i, j, d) in enumerate(gr.edges):
        p, q = X[i], X[j]
        if i == j:
            rr = 0.6 * radii[i] + 0.1 * dmin
            c = p + np.array([0.0, 0.8 * radii[i] + rr] + ([0.0] if dim == 3 else []))
            ts = np.linspace(-math.pi / 2, 3 * math.pi / 2, 17)
            loop = [c + rr * np.array([math.cos(t), math.sin(t)] + ([0.0] if dim == 3 else [])) for t in ts]
            path = loop
        else:
            key = (min(i, j), max(i, j))
            mult = pair_count[key]
            idx = pair_seen.get(key, 0)
            pair_seen[key] = idx + 1
            mids = [np.array(v, dtype=float) for v in routes.get(k, [])]
            if mult > 1 and not mids:
                off = (idx - (mult - 1) / 2.0) * 0.25 * float(np.linalg.norm(q - p))
                u = q - p
                if dim == 2:
                    perp = np.array([-u[1], u[0]]) / max(np.linalg.norm(u), 1e-12)
                else:
                    a = np.cross(u, [0.0, 0.0, 1.0])
                    if np.linalg.norm(a) < 1e-9:
                        a = np.cross(u, [0.0, 1.0, 0.0])
                    perp = a / np.linalg.norm(a)
                if (i, j) != key:
                    perp = -perp
                mids = [(p + q) / 2 + off * perp]
            path = [p] + mids + [q]
            if d:                                   # shorten to the vertex disks so arrowheads stay visible
                a0, a1 = path[0], path[1]
                v0 = a1 - a0
                l0 = float(np.linalg.norm(v0))
                if l0 > 2 * radii[i]:
                    path[0] = a0 + v0 / l0 * radii[i]
                b0, b1 = path[-1], path[-2]
                v1 = b1 - b0
                l1 = float(np.linalg.norm(v1))
                if l1 > 2 * radii[j]:
                    path[-1] = b0 + v1 / l1 * radii[j]
        prim = Expr("Arrow" if d else "Line", L(*(pt(x) for x in path)))
        sty = []
        if k in hl_e:
            sty.append(rgb(HIGHLIGHT_COLOR))
            sty.append(Expr("AbsoluteThickness", 2.5))
        elif e_style[k] is not None:
            sty.append(e_style[k])
        edges_out.append(L(*sty, prim) if sty else prim)
    content = [L(rgb(EDGE_COLOR), Expr("AbsoluteThickness", 1.25), *edges_out)]
    # vertices
    vout = []
    for v in range(n):
        col = vertex_colors[v] if vertex_colors is not None else None
        sty = []
        if v in hl_v:
            sty.append(rgb(HIGHLIGHT_COLOR))
        elif v_style[v] is not None:
            sty.append(v_style[v])
        elif col is not None:
            sty.append(col)
        prim = Expr("Disk", pt(X[v]), fl(radii[v])) if dim == 2 else Expr("Sphere", pt(X[v]), fl(radii[v]))
        if vertex_extra is not None and vertex_extra[v] is not None:
            sty.extend(vertex_extra[v])
        vout.append(L(*sty, prim) if sty else prim)
    vdefault = [rgb(VERTEX_COLOR)]
    if dim == 2:
        vdefault.append(Expr("EdgeForm", L(rgb((0.25, 0.25, 0.25)), Expr("AbsoluteThickness", 0.5))))
    content.append(L(*vdefault, *vout))
    # labels
    texts = []
    vl = gopts.get("VertexLabels")
    if labels_override is not None:
        vlabels = labels_override
    elif vl is None or sym(vl, "None"):
        vlabels = [None] * n
    elif vl == "Name" or sym(vl, "Automatic"):
        vlabels = list(gr.vertices)
    elif isinstance(vl, Expr) and vl.head == "Placed":
        raise NotImplementedError(f"{fname}: Placed vertex labels are not supported")
    elif is_list(vl) or is_rule(vl):
        vlabels = _per_item(vl, lambda k, x: K.tag(x) == K.tag(gr.vertices[k]), n, fname, "VertexLabels",
                            lambda x: gr.vertices[0] if False else x)
        vlabels = [gr.vertices[k] if (lab == "Name" or sym(lab, "Automatic")) else lab
                   for k, lab in enumerate(vlabels)]
    else:
        raise NotImplementedError(f"{fname}: VertexLabels -> {K_input(vl)} is not supported")
    for v in range(n):
        lab = vlabels[v]
        if lab is None or sym(lab, "None"):
            continue
        off = [radii[v] * 1.2, radii[v] * 1.2] + ([0.0] if dim == 3 else [])
        texts.append(Expr("Text", lab, pt(X[v] + np.array(off))))
    el = gopts.get("EdgeLabels")
    if el is not None and not sym(el, "None"):
        if el == "Name":
            elabels = list(gr.edge_exprs)
        elif el == "EdgeWeight":
            if gr.weights is None:
                raise ValueError(f"{fname}: EdgeLabels -> \"EdgeWeight\" needs a graph with EdgeWeight")
            elabels = [K.from_sp(w) if isinstance(w, sp.Basic) else w for w in gr.weights]
        elif is_list(el) or is_rule(el):
            elabels = _per_item(el, lambda k, e: _edge_matches(gr, k, e, fname), m, fname, "EdgeLabels",
                                lambda x: x)
        else:
            raise NotImplementedError(f"{fname}: EdgeLabels -> {K_input(el)} is not supported")
        for k, (i, j, _) in enumerate(gr.edges):
            if elabels[k] is None:
                continue
            mids = routes.get(k)
            mp = np.array(mids[len(mids) // 2], dtype=float) if mids else (X[i] + X[j]) / 2
            texts.append(Expr("Text", elabels[k], pt(mp)))
    if texts:
        content.append(L(GrayLevel(0), *texts))
    return content


def GrayLevel(x):
    return Expr("GrayLevel", x)


def graph_plot(args, fname, dim, default_layout, rules_directed=False, where=None, positional=None):
    """Shared body of GraphPlot, GraphPlot3D, LayeredGraphPlot(3D) and TreePlot."""
    pos, opts = split_opts(args, fname)
    if not pos:
        raise TypeError(f"{fname}: a graph is required")
    de = opts.pop("DirectedEdges", None)
    rd = rules_directed if de is None or sym(de, "Automatic") else de
    if de is not None and not (de is True or de is False or sym(de, "Automatic")):
        raise ValueError(f"{fname}: DirectedEdges must be True, False or Automatic")
    gr = graph_input(pos[0], fname, rules_directed=rd is True)
    if de is True or de is False:
        gr = GR.WG(gr.vertices, GR._directed_edges(gr.edge_exprs, de, fname), gr.opts, fname)
    gopts, gfx = merged_options(gr, opts, fname)
    lay = parse_layout(gopts.get("GraphLayout"), fname, dim, default_layout)
    if positional is not None:
        positional(pos[1:], lay, gr, fname)
    elif len(pos) > 1:
        raise TypeError(f"{fname}: too many arguments")
    vc = gopts.get("VertexCoordinates")
    if vc is not None and not sym(vc, "Automatic"):
        X, routes = _vertex_coordinates(gr, vc, fname, dim)
    else:
        X, routes = compute_layout(gr, lay, fname)
    check_layout(X, gr.n, dim, fname)
    content = draw_graph(gr, X, routes, gopts, fname, dim)
    rules = _graphics_rules(gfx, dim, fname)
    if dim == 3 and "Boxed" not in gfx:
        rules.append(Expr("Rule", Symbol("Boxed"), False))
    return Expr("Graphics" if dim == 2 else "Graphics3D", L(*content), *rules)


def _root_position(rest, lay, gr, fname, allow=_POSITIONS):
    """Trailing positional arguments: pos, v -> pos, or (TreePlot) pos, v."""
    if not rest:
        return
    a = rest[0]
    if is_rule(a) and sym(a.args[1], *allow):
        lay.root, lay.where = a.args[0], a.args[1].name
        _vindex(gr, lay.root, fname)
    elif sym(a, *allow):
        lay.where = a.name
    elif isinstance(a, Symbol) and K.tag(a) not in gr.index:
        raise Unevaluated()
    else:
        raise ValueError(f"{fname}: the position must be one of {', '.join(allow)} (or v -> pos), not "
                         f"{K_input(a)}")
    if len(rest) == 2:
        lay.root = rest[1]
        _vindex(gr, lay.root, fname)
    elif len(rest) > 2:
        raise TypeError(f"{fname}: too many arguments")


# ═══════════════════════════════════════ graph plots ═══════════════════════════════════════

_INPUT_NOTE = ("The graph is a Graph, a list of edges or rules, or an adjacency matrix; a symbol stays "
               "unevaluated, other non-graphs raise TypeError; mixed graphs raise NotImplementedError where "
               "the layout needs one edge kind. ")


@fn("GraphPlot", area=AREA, status="added",
    notes="GraphPlot[g | {e1, ...} | {v -> w, ...} | m, opts] -> Graphics. Rules are undirected unless "
          "DirectedEdges -> True (Wolfram's default False). " + _INPUT_NOTE + "GraphLayout Automatic is "
          "\"SpringElectricalEmbedding\" (Fruchterman-Reingold, see the module notes); also "
          "\"SpringEmbedding\" (stress majorization), \"SpectralEmbedding\", \"CircularEmbedding\" (vertex k at "
          "angle Pi/2 - 2 Pi (k - 1)/n), \"StarEmbedding\" (highest-degree vertex at the centre), \"GridEmbedding\" "
          "(row-major, Ceiling[Sqrt[n]] columns), \"LinearEmbedding\", \"BipartiteEmbedding\", "
          "\"LayeredDigraphEmbedding\", \"LayeredEmbedding\", \"RadialEmbedding\", with sub-options "
          "\"RootVertex\", \"Orientation\", \"Dimension\"; other embeddings raise NotImplementedError. Layout "
          "coordinates are this implementation's own (deterministic), not Wolfram's. " + DRAW_NOTE +
          " Check: self-check (one finite point per vertex) and reference (tests check every edge is drawn "
          "between its endpoints and the closed-form layouts).")
def _graphplot(g: Expr, *rest: Option) -> Expr:
    return graph_plot((g,) + rest, "GraphPlot", 2, "SpringElectricalEmbedding")


@fn("GraphPlot3D", area=AREA, status="added",
    notes="GraphPlot3D[g | edges | rules | m, opts] -> Graphics3D (Boxed -> False unless given) with Line/Arrow "
          "edges and Sphere vertices. GraphLayout Automatic is the 3-D spring-electrical model (deterministic "
          "Fibonacci-sphere start); \"SpringEmbedding\", \"SpectralEmbedding\" and \"LayeredDigraphEmbedding\" "
          "(layers stacked vertically, each on a circle) are also supported; others raise NotImplementedError. "
          + _INPUT_NOTE + DRAW_NOTE + " Check: self-check (one finite 3-D point per vertex) and reference (edges "
          "join their endpoints in tests).")
def _graphplot3d(g: Expr, *rest: Option) -> Expr:
    return graph_plot((g,) + rest, "GraphPlot3D", 3, "SpringElectricalEmbedding")


def _layered_args(rest, lay, gr, fname):
    _root_position(rest, lay, gr, fname)
    if len(rest) > 1:
        raise TypeError(f"{fname}: too many arguments")


@fn("LayeredGraphPlot", area=AREA, status="added",
    notes="LayeredGraphPlot[g], [g, pos], [g, v -> pos] (pos Top (default), Bottom, Left, Right) -> Graphics with "
          "the Sugiyama layered drawing: for directed graphs (rules are directed, DirectedEdges Automatic) "
          "cycles are broken by reversing depth-first back edges, sources go to the first layer and each vertex "
          "one layer below its lowest predecessor (longest-path layering); a given root must be a source "
          "(else NotImplementedError). Undirected graphs are layered by breadth-first distance from the root "
          "(default: the centre, minimum eccentricity, of each component); same-layer and long edges are "
          "drawn, long edges as polylines through their dummy vertices. Crossing reduction: 24 barycentre "
          "sweeps keeping the best ordering; x-coordinates by an averaging pass with unit separation. "
          + _INPUT_NOTE + DRAW_NOTE + " Check: self-check (cycle removal leaves an acyclic graph; one finite "
          "point per vertex) and reference (every arc points from a higher to a lower layer in tests).")
def _layeredgraphplot(g: Expr, *rest: Expr | Option) -> Expr:
    fname = "LayeredGraphPlot"
    return graph_plot((g,) + rest, fname, 2, "LayeredDigraphEmbedding", rules_directed=True,
                      positional=_layered_args)


def _layer_planes(X, layers_spec, style_spec, fname):
    styles = _layer_styles(style_spec, fname)
    if layers_spec is None or sym(layers_spec, "None", "Automatic"):
        return []
    zs = sorted({round(float(z), 9) for z in X[:, 2]}, reverse=True)
    if sym(layers_spec, "All"):
        want = list(range(1, len(zs) + 1))
    else:
        items = layers_spec.args if is_list(layers_spec) else (layers_spec,)
        want = [_posint(m, fname, "a GraphLayers entry") for m in items]
    for m in want:
        if m > len(zs):
            raise ValueError(f"{fname}: GraphLayers asks for layer {m}, but the plot has {len(zs)} layers")
    lo, hi = X[:, :2].min(0) - 0.5, X[:, :2].max(0) + 0.5
    out = []
    for k, m in enumerate(want):
        z = zs[m - 1]
        sq = L(pt((lo[0], lo[1], z)), pt((hi[0], lo[1], z)), pt((hi[0], hi[1], z)), pt((lo[0], hi[1], z)))
        out.append(L(styles[k % len(styles)], Expr("EdgeForm", L()), Expr("Polygon", sq)))
    return out


def _layer_styles(style_spec, fname):
    if style_spec is None or sym(style_spec, "Automatic"):
        styles = [Expr("Directive", Expr("Opacity", 0.2), rgb((0.6, 0.6, 0.8)))]
    elif isinstance(style_spec, str):
        raise NotImplementedError(f"{fname}: named colour schemes in GraphLayerStyle are not supported")
    elif is_list(style_spec):
        styles = [_style_of(s, fname, "GraphLayerStyle") for s in style_spec.args]
        if not styles:
            raise ValueError(f"{fname}: GraphLayerStyle must not be an empty list")
    else:
        styles = [_style_of(style_spec, fname, "GraphLayerStyle")]
    return styles


@fn("LayeredGraphPlot3D", area=AREA, status="added",
    notes="LayeredGraphPlot3D[g], [g, pos], [g, v -> pos] (pos Top (default) or Bottom; Left/Right raise "
          "NotImplementedError) -> Graphics3D: the Sugiyama layering of LayeredGraphPlot with layer l at height "
          "-l (Top) and the vertices of each layer on a horizontal circle in their crossing-reduced order. "
          "GraphLayers -> None/Automatic (no planes), m, {m1, ...} or All draws translucent planes at those "
          "layers (layer 1 holds the roots); GraphLayerStyle -> directive or {d1, ...} (cyclic) styles them "
          "(named colour schemes raise NotImplementedError). " + _INPUT_NOTE + DRAW_NOTE + " Check: self-check "
          "(acyclic layering, finite points) and reference (layer heights and plane counts in tests).")
def _layeredgraphplot3d(g: Expr, *rest: Expr | Option) -> Expr:
    fname = "LayeredGraphPlot3D"
    pos, opts = split_opts((g,) + rest, fname)
    layers_spec = opts.pop("GraphLayers", None)
    style_spec = opts.pop("GraphLayerStyle", None)

    def args3(rest, lay, gr, fname):
        _root_position(rest, lay, gr, fname)
        if lay.where in ("Left", "Right"):
            raise NotImplementedError(f"{fname}: the positions Left and Right are not supported in 3-D")
        if len(rest) > 1:
            raise TypeError(f"{fname}: too many arguments")
    out = graph_plot(tuple(pos) + tuple(Expr("Rule", Symbol(k), v) for k, v in opts.items()), fname, 3,
                     "LayeredDigraphEmbedding", rules_directed=True, positional=args3)
    content = list(out.args[0].args)
    X = _content_points3d(content)
    planes = _layer_planes(X, layers_spec, style_spec, fname)
    return Expr("Graphics3D", L(*planes, *content), *out.args[1:])


def _content_points3d(content):
    np = _np()
    pts = []
    for grp in content:
        for x in (grp.args if is_list(grp) else (grp,)):
            p = x.args[-1] if is_list(x) else x
            if isinstance(p, Expr) and p.head == "Sphere":
                pts.append([float(c) for c in p.args[0].args])
    return np.array(pts, dtype=float).reshape(-1, 3)


def _tree_args(rest, lay, gr, fname):
    if rest and is_rule(rest[0]) and sym(rest[0].args[1], "Center") or rest and sym(rest[0], "Center"):
        lay.method = "RadialEmbedding"
        if is_rule(rest[0]):
            lay.root = rest[0].args[0]
            _vindex(gr, lay.root, fname)
        if len(rest) == 2:
            lay.root = rest[1]
            _vindex(gr, lay.root, fname)
        elif len(rest) > 2:
            raise TypeError(f"{fname}: too many arguments")
        return
    _root_position(rest, lay, gr, fname)


@fn("TreePlot", area=AREA, status="added",
    notes="TreePlot[g], [g, pos], [g, v -> pos], [g, pos, v] (pos Top (default), Bottom, Left, Right, Center) "
          "-> Graphics: the Reingold-Tilford tidy drawing of a breadth-first spanning tree of each component "
          "(children in VertexList order, siblings 1 apart, parents centred over their children, layers 1 "
          "apart); non-tree edges are drawn too. The root is the given vertex, else the unique source of a "
          "directed component, else the centre (minimum eccentricity; first in VertexList order), which gives "
          "the fewest layers; components are packed left to right. Center gives the radial layout (depth k on "
          "the circle of radius k, wedges proportional to leaf counts). Rules are undirected (DirectedEdges -> "
          "False, as Wolfram). LayerSizeFunction accepts only its default. " + _INPUT_NOTE + DRAW_NOTE +
          " Check: self-check (finite points) and reference (layer = BFS depth and the tidy-tree invariants "
          "in tests).")
def _treeplot(g: Expr, *rest: Expr | Option) -> Expr:
    fname = "TreePlot"
    pos, opts = split_opts((g,) + rest, fname)
    lsf = opts.pop("LayerSizeFunction", None)
    if lsf is not None and not (isinstance(lsf, Expr) and lsf.head == "Function" and lsf.args == (1,)) \
            and not (K.is_int(lsf) and int(lsf) == 1):
        raise NotImplementedError(f"{fname}: LayerSizeFunction other than its default is not supported")
    if "GraphLayout" not in opts:
        opts["GraphLayout"] = "LayeredEmbedding"
    return graph_plot(tuple(pos) + tuple(Expr("Rule", Symbol(k), v) for k, v in opts.items()), fname, 2,
                      "LayeredEmbedding", positional=_tree_args)


# ═══════════════════════════════════════ Graph3D, LayeredGraph, LayeredGraph3D ═══════════════════════════════════════

def _graph_with_layout(args, fname, layout_of):
    """Graph[...] built from the arguments with a GraphLayout computed from the trailing position argument."""
    pos, opts = split_opts(args, fname)
    where, root = None, None
    if pos and len(pos) >= 2 and (sym(pos[-1], *_POSITIONS) or is_rule(pos[-1]) and sym(pos[-1].args[1],
                                                                                           *_POSITIONS)):
        p = pos.pop()
        if is_rule(p):
            root, where = p.args[0], p.args[1].name
        else:
            where = p.name
    if not pos or len(pos) > 2:
        raise TypeError(f"{fname}: expected a graph, an edge list, or a vertex list and an edge list")
    user = opts.pop("GraphLayout", None)
    layout = layout_of(user, where, root)
    rules = [Expr("Rule", Symbol(k), v) for k, v in opts.items()]
    if len(pos) == 1 and isinstance(pos[0], Symbol):
        raise Unevaluated()
    if not all(is_list(p) or isinstance(p, Expr) and p.head == "Graph" for p in pos):
        raise TypeError(f"{fname}: expected a graph, an edge list, or a vertex list and an edge list")
    try:
        out = ev(Expr("Graph", *pos, *rules, Expr("Rule", Symbol("GraphLayout"), layout)))
    except GR.GraphError as exc:
        raise ValueError(f"{fname}: {exc}") from None
    if not (isinstance(out, Expr) and out.head == "Graph"):
        if isinstance(pos[0], Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the arguments do not describe a graph")
    if root is not None:
        graph_input(out, fname)
        _vindex(graph_input(out, fname), root, fname)
    return out


def _user_method(user, fname, allowed):
    if user is None or sym(user, "Automatic"):
        return None, []
    items = list(user.args) if is_list(user) else [user]
    name = items.pop(0) if items and isinstance(items[0], str) else None
    if any(not is_rule(r) for r in items):
        raise ValueError(f"{fname}: GraphLayout must be a method name or {{name, sub-option rules}}, not "
                         f"{K_input(user)}")
    if name is not None and name not in allowed:
        raise NotImplementedError(f"{fname}: GraphLayout {name!r} is not supported here")
    return name, items


@fn("Graph3D", area=AREA, status="added",
    notes="Graph3D[g], Graph3D[{e1, ...}], Graph3D[{v1, ...}, {e1, ...}, opts]: the Graph of Graph[...] carrying "
          "GraphLayout -> {\"SpringElectricalEmbedding\", \"Dimension\" -> 3} (or {method, \"Dimension\" -> 3} "
          "for a GraphLayout -> method option), so it is the same graph for every graph function and GraphPlot3D "
          "draws it in 3-D. The InputForm of Wolfram's Graph3D object is not documented; this representation is "
          "this implementation's. Check: exact (tests compare VertexList/EdgeList with Graph and read the "
          "layout option back).")
def _graph3d(*args: Expr) -> Expr:
    fname = "Graph3D"

    def lo(user, where, root):
        if where is not None:
            raise TypeError(f"{fname}: a position argument is not supported")
        name, items = _user_method(user, fname, LAYOUTS_3D)
        return L(name or "SpringElectricalEmbedding", *[i for i in items if _opt_name(i) != "Dimension"],
                 Expr("Rule", "Dimension", 3))
    if not args:
        raise TypeError(f"{fname}: a graph is required")
    return _graph_with_layout(args, fname, lo)


def _layered_spec(fname, dim):
    def lo(user, where, root):
        name, items = _user_method(user, fname, ("LayeredDigraphEmbedding",))
        if dim == 3 and where in ("Left", "Right"):
            raise NotImplementedError(f"{fname}: the positions Left and Right are not supported in 3-D")
        out = [name or "LayeredDigraphEmbedding"]
        out += [i for i in items if _opt_name(i) not in ("Orientation", "RootVertex", "Dimension")]
        if where is not None:
            out.append(Expr("Rule", "Orientation", Symbol(where)))
        if root is not None:
            out.append(Expr("Rule", "RootVertex", root))
        if dim == 3:
            out.append(Expr("Rule", "Dimension", 3))
        return L(*out) if len(out) > 1 else out[0]
    return lo


@fn("LayeredGraph", area=AREA, status="added",
    notes="LayeredGraph[g | {e1, ...} | {v...}, {e...}], [..., pos], [..., v -> pos] (pos Top, Bottom, Left, "
          "Right): the Graph with GraphLayout -> \"LayeredDigraphEmbedding\" (plus \"Orientation\" -> pos and "
          "\"RootVertex\" -> v when given), which GraphPlot draws with the Sugiyama layering of LayeredGraphPlot. "
          "The InputForm of Wolfram's LayeredGraph object is not documented; this representation is this "
          "implementation's. Check: exact (tests compare the graph with Graph and read the layout back).")
def _layeredgraph(*args: Expr) -> Expr:
    if not args:
        raise TypeError("LayeredGraph: a graph is required")
    return _graph_with_layout(args, "LayeredGraph", _layered_spec("LayeredGraph", 2))


@fn("LayeredGraph3D", area=AREA, status="added",
    notes="LayeredGraph3D[g | edges | vertices, edges], [..., pos], [..., v -> pos] (pos Top or Bottom; Left/Right "
          "raise NotImplementedError): the Graph with GraphLayout -> {\"LayeredDigraphEmbedding\", "
          "\"Dimension\" -> 3, ...}, drawn by GraphPlot3D as in LayeredGraphPlot3D. The InputForm of Wolfram's "
          "object is not documented; this representation is this implementation's. Check: exact (tests read the "
          "layout back and compare the graph with Graph).")
def _layeredgraph3d(*args: Expr) -> Expr:
    if not args:
        raise TypeError("LayeredGraph3D: a graph is required")
    return _graph_with_layout(args, "LayeredGraph3D", _layered_spec("LayeredGraph3D", 3))


# ═══════════════════════════════════════ communities and partitions ═══════════════════════════════════════

def weighted_simple(gr):
    """{(i, j): w} for the underlying undirected simple graph (i < j); directions dropped, parallel edges and
    EdgeWeight summed, self-loops dropped."""
    ws = gr.weights
    out = {}
    for k, (i, j, _) in enumerate(gr.edges):
        if i == j:
            continue
        w = 1.0 if ws is None else float(S(ws[k]))
        if ws is not None and not (w > 0):
            raise ValueError(f"{gr.fname}: edge weights must be positive")
        key = (min(i, j), max(i, j))
        out[key] = out.get(key, 0.0) + w
    return out


def _nx_graph(gr, W, nx):
    Gx = nx.Graph()
    Gx.add_nodes_from(range(gr.n))
    for (i, j), w in W.items():
        Gx.add_edge(i, j, weight=w)
    return Gx


def modularity(n, W, parts):
    """Newman-Girvan modularity Q = sum_c [L_c / m - (d_c / 2m)^2] of a partition (weighted)."""
    m = sum(W.values())
    if m == 0:
        return 0.0
    deg = [0.0] * n
    for (i, j), w in W.items():
        deg[i] += w
        deg[j] += w
    comm = {}
    for c, p in enumerate(parts):
        for v in p:
            comm[v] = c
    q = 0.0
    for (i, j), w in W.items():
        if comm[i] == comm[j]:
            q += w / m
    for p in parts:
        d = sum(deg[v] for v in p)
        q -= (d / (2 * m)) ** 2
    return q


def _order_parts(parts):
    parts = [sorted(p) for p in parts if p]
    return sorted(parts, key=lambda p: (-len(p), p[0]))


def _parts_expr(gr, parts):
    return L(*(L(*(gr.vertices[v] for v in p)) for p in parts))


def _check_partition(parts, n, fname):
    seen = sorted(v for p in parts for v in p)
    if seen != list(range(n)):
        raise ArithmeticError(f"{fname}: result failed verification (the parts do not partition the vertices)")


def _leading_eigenvector(n, W):
    """Newman's leading-eigenvector method (Newman 2006): recursive bisection by the sign of the leading
    eigenvector of the (generalised) modularity matrix while it increases the modularity."""
    np = _np()
    A = np.zeros((n, n))
    for (i, j), w in W.items():
        A[i, j] = A[j, i] = w
    k = A.sum(1)
    m2 = k.sum()
    if m2 == 0:
        return [[v] for v in range(n)]
    B = A - np.outer(k, k) / m2
    out = []
    stack = [list(range(n))]
    while stack:
        g = stack.pop()
        if len(g) < 2:
            out.append(g)
            continue
        Bg = B[np.ix_(g, g)].copy()
        Bg[np.diag_indices(len(g))] -= Bg.sum(1)
        w_, V = np.linalg.eigh(Bg)
        if w_[-1] <= 1e-10:
            out.append(g)
            continue
        u = V[:, -1]
        s = np.where(u >= 0, 1.0, -1.0)
        if s[0] < 0:
            s = -s
        dq = float(s @ Bg @ s) / (2 * m2)
        if dq <= 1e-10 or abs(s.sum()) == len(g):
            out.append(g)
            continue
        stack.append([v for v, x in zip(g, s) if x < 0])
        stack.append([v for v, x in zip(g, s) if x > 0])
    return out


_COMMUNITY_METHODS = ("Modularity", "Spectral", "Centrality", "CliquePercolation", "Hierarchical")


def find_communities(gr, method, fname):
    nx = _need_nx(fname)
    W = weighted_simple(gr)
    n = gr.n
    if n == 0:
        return []
    if method == "Modularity":
        Gx = _nx_graph(gr, W, nx)
        if not W:
            parts = [[v] for v in range(n)]
        else:
            parts = [sorted(c) for c in nx.community.greedy_modularity_communities(Gx, weight="weight")]
        q = modularity(n, W, parts)
        if W and abs(q - nx.community.modularity(Gx, parts, weight="weight")) > 1e-9:
            raise ArithmeticError(f"{fname}: result failed verification (modularity mismatch)")
        if W and q < -1e-12:
            raise ArithmeticError(f"{fname}: result failed verification (negative modularity)")
    elif method == "Spectral":
        parts = _leading_eigenvector(n, W)
    elif method == "Centrality":
        if len(W) > 400:
            raise NotImplementedError(f"{fname}: Method -> \"Centrality\" (Girvan-Newman) is limited to 400 edges")
        Gx = _nx_graph(gr, W, nx)
        best, bq = [list(range(n))], modularity(n, W, [list(range(n))])
        if W:
            for level in nx.community.girvan_newman(Gx):
                ps = [sorted(c) for c in level]
                q = modularity(n, W, ps)
                if q > bq + 1e-12:
                    best, bq = ps, q
        parts = best
    elif method == "CliquePercolation":
        Gx = _nx_graph(gr, W, nx)
        parts = [sorted(c) for c in nx.community.k_clique_communities(Gx, 3)]
        return _order_parts(parts)
    elif method == "Hierarchical":
        raise NotImplementedError(f"{fname}: Method -> \"Hierarchical\" is not supported (the vertex similarity "
                                  "and the cut level Wolfram uses are not documented)")
    else:
        raise ValueError(f"{fname}: unknown Method {method!r} (use one of {', '.join(_COMMUNITY_METHODS)})")
    _check_partition(parts, n, fname)
    return _order_parts(parts)


def _method_name(opts, fname, default):
    m = opts.pop("Method", None)
    if m is None or sym(m, "Automatic"):
        return default
    if isinstance(m, str):
        return m
    if is_list(m) and m.args and isinstance(m.args[0], str):
        raise NotImplementedError(f"{fname}: Method sub-options are not supported")
    raise ValueError(f"{fname}: Method must be a string such as \"Modularity\", not {K_input(m)}")


@fn("FindGraphCommunities", area=AREA, status="added",
    notes="FindGraphCommunities[g | {v -> w, ...}, Method -> m]: the communities {c1, c2, ...} (lists of vertices "
          "in VertexList order), largest first (ties: earliest first vertex). Graphs are treated as undirected "
          "and weighted (EdgeWeight; parallel edges and the two directions add up; self-loops are dropped). "
          "Methods: \"Modularity\" (default; the Clauset-Newman-Moore greedy modularity maximisation, NetworkX), "
          "\"Spectral\" (Newman's leading-eigenvector bisection), \"Centrality\" (Girvan-Newman edge "
          "betweenness, keeping the level of highest modularity; up to 400 edges), \"CliquePercolation\" "
          "(k = 3 clique percolation; communities may overlap and uncovered vertices are left out); "
          "\"Hierarchical\" raises NotImplementedError (undocumented similarity). The communities found may "
          "differ from Wolfram's (its heuristics are not published). Needs NetworkX. Check: self-check (the "
          "result partitions the vertices; for \"Modularity\" the modularity is recomputed independently and "
          "must match NetworkX's and be >= 0) and reference (planted communities in tests).")
def _findgraphcommunities(g: Expr, *opts: Option) -> ListOf[ListOf[Expr]]:
    fname = "FindGraphCommunities"
    pos, o = split_opts((g,) + opts, fname)
    if len(pos) != 1:
        raise TypeError(f"{fname}: one graph argument expected")
    method = _method_name(o, fname, "Modularity")
    if o:
        raise ValueError(f"{fname}: unknown option {next(iter(o))}")
    gr = graph_input(pos[0], fname, rules_directed=True)
    return _parts_expr(gr, find_communities(gr, method, fname))


def _part_sizes(spec, n, fname):
    if spec is None:
        k = 2
    elif K.is_int(spec):
        k = _posint(spec, fname, "the number of parts")
    elif is_list(spec) and spec.args:
        vals = spec.args
        if all(K.is_int(v) for v in vals):
            sizes = [int(v) for v in vals]
            if any(s < 1 for s in sizes):
                raise ValueError(f"{fname}: part sizes must be positive integers")
            if sum(sizes) != n:
                raise ValueError(f"{fname}: the part sizes {sizes} do not add up to the {n} vertices")
            return sizes
        ps = [_real(v, fname, "a part proportion", lo=0, lo_open=True) for v in vals]
        tot = sum(ps)
        raw = [p / tot * n for p in ps]
        sizes = [int(math.floor(r)) for r in raw]
        rem = n - sum(sizes)
        for idx in sorted(range(len(raw)), key=lambda i: (-(raw[i] - sizes[i]), i))[:rem]:
            sizes[idx] += 1
        if any(s == 0 for s in sizes):
            raise ValueError(f"{fname}: a proportion gives an empty part for {n} vertices")
        return sizes
    elif isinstance(spec, (Symbol, Expr)) and not is_list(spec):
        raise Unevaluated()
    else:
        raise TypeError(f"{fname}: the second argument must be k, {{n1, ...}} or {{a1, ...}}")
    if k > n:
        raise ValueError(f"{fname}: cannot split {n} vertices into {k} non-empty parts")
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def _cut(W, assign):
    return sum(w for (i, j), w in W.items() if assign[i] != assign[j])


def _exhaustive(n, sizes, W, limit=60000):
    import itertools
    count = math.factorial(n)
    for s in sizes:
        count //= math.factorial(s)
    if count > limit:
        return None
    best, bc = None, None
    verts = list(range(n))

    def rec(rem, k, assign):
        nonlocal best, bc
        if k == len(sizes) - 1:
            for v in rem:
                assign[v] = k
            c = _cut(W, assign)
            if bc is None or c < bc - 1e-12:
                best, bc = list(assign), c
            return
        first_equal = k > 0 and sizes[k] == sizes[k - 1]
        for comb in itertools.combinations(rem, sizes[k]):
            if first_equal and prev_min[k - 1] > comb[0]:
                continue
            prev_min[k] = comb[0]
            for v in comb:
                assign[v] = k
            rest = [v for v in rem if v not in comb]
            rec(rest, k + 1, assign)
    prev_min = [0] * len(sizes)
    rec(verts, 0, [0] * n)
    return best, bc


def _fiedler_order(vs, W):
    np = _np()
    idx = {v: k for k, v in enumerate(vs)}
    n = len(vs)
    Lm = np.zeros((n, n))
    for (i, j), w in W.items():
        if i in idx and j in idx:
            a, b = idx[i], idx[j]
            Lm[a, b] -= w
            Lm[b, a] -= w
    np.fill_diagonal(Lm, -Lm.sum(1))
    if n < 2:
        return list(vs)
    w_, V = np.linalg.eigh(Lm)
    f = V[:, 1]
    return [vs[k] for k in sorted(range(n), key=lambda k: (round(float(f[k]), 12), k))]


def _bisect(vs, size_a, W, nx):
    """Split vs into a part of size size_a and the rest: Fiedler-vector start refined by Kernighan-Lin."""
    order = _fiedler_order(vs, W)
    A, B = order[:size_a], order[size_a:]
    Gx = nx.Graph()
    Gx.add_nodes_from(vs)
    sv = set(vs)
    for (i, j), w in W.items():
        if i in sv and j in sv:
            Gx.add_edge(i, j, weight=w)
    if Gx.number_of_edges():
        A2, B2 = nx.algorithms.community.kernighan_lin_bisection(Gx, partition=(set(A), set(B)), max_iter=50,
                                                                 weight="weight", seed=7)
        if len(A2) == size_a:
            A, B = sorted(A2), sorted(B2)
    return list(A), list(B)


def partition_graph(gr, sizes, fname):
    W = weighted_simple(gr)
    n = gr.n
    ex = _exhaustive(n, sizes, W)
    if ex is not None:
        assign, _ = ex
        parts = [[v for v in range(n) if assign[v] == k] for k in range(len(sizes))]
        method = "exact"
    else:
        nx = _need_nx(fname)
        parts = []

        def rec(vs, ss):
            if len(ss) == 1:
                parts.append(vs)
                return
            h = len(ss) // 2
            A, B = _bisect(vs, sum(ss[:h]), W, nx)
            rec(A, ss[:h])
            rec(B, ss[h:])
        rec(list(range(n)), sorted(sizes, reverse=True))
        method = "heuristic"
    _check_partition(parts, n, fname)
    if sorted(len(p) for p in parts) != sorted(sizes):
        raise ArithmeticError(f"{fname}: result failed verification (part sizes)")
    return _order_parts(parts), method


@fn("FindGraphPartition", area=AREA, status="added",
    notes="FindGraphPartition[g] (= [g, 2]), [g, k] (k parts of sizes differing by at most 1), [g, {n1, ...}] "
          "(given sizes, which must add up to the vertex count), [g, {a1, ...}] (sizes from the proportions by "
          "largest-remainder rounding): a partition of the vertices minimising the number (total EdgeWeight) "
          "of edges between parts; graphs are treated as undirected simple graphs (parallel edges and the two "
          "directions add up, self-loops dropped). When the number of candidate partitions is at most 60000 the "
          "minimum is found exactly by enumeration; larger graphs use recursive spectral (Fiedler vector) "
          "bisection refined by Kernighan-Lin (NetworkX), a heuristic like Wolfram's, so the cut may not be "
          "minimal. Parts are lists in VertexList order, largest first. Vertex-weighted proportions raise "
          "NotImplementedError. Check: self-check (the parts partition the vertices with the requested sizes) "
          "and reference (cut sizes of documented examples and planted partitions in tests).")
def _findgraphpartition(g: Expr, spec: PosInteger | ListOf[Real] = None) -> ListOf[ListOf[Expr]]:
    fname = "FindGraphPartition"
    gr = graph_input(g, fname, rules_directed=True)
    if gr.n == 0:
        raise ValueError(f"{fname}: the graph has no vertices")
    if spec is not None and is_list(spec) and not all(K.is_int(v) for v in spec.args) and \
            "VertexWeight" in gr.opts:
        raise NotImplementedError(f"{fname}: proportions measured by VertexWeight are not supported")
    sizes = _part_sizes(spec, gr.n, fname)
    parts, _ = partition_graph(gr, sizes, fname)
    return _parts_expr(gr, parts)


# ─────────────────────────── lambda and Luccio-Sami components ───────────────────────────

def _gomory_hu_sets(gr, fname):
    """Lambda sets via the Gomory-Hu tree: for each level k, the classes of 'edge connectivity >= k'."""
    nx = _need_nx(fname)
    W = weighted_simple(gr)
    n = gr.n
    nb = _undirected_adj(gr)
    sets = set()
    for comp in _components(nb):
        if len(comp) == 1:
            continue
        Gx = nx.Graph()
        Gx.add_nodes_from(comp)
        cs = set(comp)
        for (i, j), w in W.items():
            if i in cs:
                Gx.add_edge(i, j, capacity=w)
        T = nx.gomory_hu_tree(Gx, capacity="capacity")
        levels = sorted({d["weight"] for _, _, d in T.edges(data=True)})
        for k in levels:
            H = nx.Graph()
            H.add_nodes_from(comp)
            H.add_edges_from((u, v) for u, v, d in T.edges(data=True) if d["weight"] >= k)
            for c in nx.connected_components(H):
                if len(c) > 1:
                    sets.add(tuple(sorted(c)))
        if len(comp) <= 14:                                  # self-check against direct max-flows
            for a in comp:
                for b in comp:
                    if a < b:
                        direct = nx.minimum_cut_value(Gx, a, b, capacity="capacity")
                        path = nx.shortest_path(T, a, b)
                        tv = min(T[u][v]["weight"] for u, v in zip(path, path[1:]))
                        if abs(direct - tv) > 1e-9:
                            raise ArithmeticError(f"{fname}: result failed verification (Gomory-Hu tree)")
    for v in range(n):
        sets.add((v,))
    sets.add(tuple(range(n)))
    return sets, W


def _sort_sets(sets):
    return sorted(sets, key=lambda s: (len(s), s))


def _restrict(gr, sets, vs, fname):
    if vs is None:
        return sets
    items = vs.args if is_list(vs) else (vs,)
    want = {_vindex(gr, v, fname) for v in items}
    return [s for s in sets if want & set(s)]


def _ls_ok(S, W, n, nx):
    """Luccio-Sami condition: every proper non-empty T of S has more edges to S - T than to V - S."""
    Sset = set(S)
    if len(S) == 1:
        return True
    o = {v: 0.0 for v in S}
    inner = []
    for (i, j), w in W.items():
        if i in Sset and j in Sset:
            inner.append((i, j, w))
        elif i in Sset:
            o[i] += w
        elif j in Sset:
            o[j] += w
    oS = sum(o.values())
    u = S[0]

    def mincut(forced_src, sink):
        D = nx.DiGraph()
        D.add_node("s")
        D.add_nodes_from(S)
        for v in S:
            if o[v] > 0:
                D.add_edge("s", v, capacity=o[v])
        D.add_edge("s", forced_src)                     # infinite capacity: forced to the source side
        D["s"][forced_src].pop("capacity", None)
        for i, j, w in inner:
            for a, b in ((i, j), (j, i)):
                if D.has_edge(a, b) and "capacity" in D[a][b]:
                    D[a][b]["capacity"] += w
                else:
                    D.add_edge(a, b, capacity=w)
        return nx.minimum_cut_value(D, "s", sink)
    for w in S[1:]:
        if mincut(u, w) - oS <= 1e-12:
            return False
        if mincut(w, u) - oS <= 1e-12:
            return False
    return True


def _ls_brute(S, W):
    import itertools
    Sset = set(S)
    for r in range(1, len(S)):
        for T in itertools.combinations(S, r):
            Ts = set(T)
            ins = sum(w for (i, j), w in W.items() if (i in Ts and j in Sset - Ts) or (j in Ts and i in Sset - Ts))
            out = sum(w for (i, j), w in W.items() if (i in Ts and j not in Sset) or (j in Ts and i not in Sset))
            if not ins > out:
                return False
    return True


@fn("LambdaComponents", area=AREA, status="added",
    notes="LambdaComponents[g] and [g, {v1, ...}] (those containing one of the vi): the lambda sets (Borgatti, "
          "Everett & Shirey 1990): S with lambda(a, b) > lambda(a, c) for all a, b in S and c outside, lambda "
          "the edge connectivity of the underlying undirected multigraph (directions dropped, parallel edges "
          "and EdgeWeight counted, self-loops dropped). Computed exactly from the Gomory-Hu tree: for each "
          "level k the classes of 'lambda >= k', plus the singletons and the whole vertex set (vacuously a "
          "lambda set, also for disconnected graphs). Sorted by size, then by VertexList position, as in the "
          "documented examples. Needs NetworkX. Check: self-check (for components up to 14 vertices the "
          "Gomory-Hu connectivities are compared with direct max-flows) and reference (Wolfram documentation "
          "examples in tests).")
def _lambdacomponents(g: Expr, vs: Expr = None) -> ListOf[ListOf[Expr]]:
    fname = "LambdaComponents"
    gr = graph_input(g, fname, rules_directed=True)
    sets, _ = _gomory_hu_sets(gr, fname)
    out = _restrict(gr, _sort_sets(sets), vs, fname)
    return _parts_expr(gr, out)


@fn("LuccioSamiComponents", area=AREA, status="added",
    notes="LuccioSamiComponents[g] and [g, {v1, ...}]: the Luccio-Sami (LS) sets (Luccio & Sami 1969): S such that "
          "every proper non-empty subset T has more edges to S - T than to the outside, on the underlying "
          "undirected multigraph as in LambdaComponents. Every LS set is a lambda set (Borgatti, Everett & "
          "Shirey 1990), so the lambda sets are tested exactly with 2(|S| - 1) minimum cuts each (the condition "
          "is a submodular minimisation). Singletons are LS sets vacuously; the whole vertex set is one exactly "
          "when the graph is connected. Sorted by size, then by VertexList position. Needs NetworkX. Check: "
          "self-check (sets of up to 12 vertices are re-tested by enumerating every subset) and reference "
          "(Wolfram documentation examples in tests).")
def _lucciosamicomponents(g: Expr, vs: Expr = None) -> ListOf[ListOf[Expr]]:
    fname = "LuccioSamiComponents"
    gr = graph_input(g, fname, rules_directed=True)
    nx = _need_nx(fname)
    sets, W = _gomory_hu_sets(gr, fname)
    keep = []
    for s in _sort_sets(sets):
        ok = _ls_ok(list(s), W, gr.n, nx)
        if len(s) <= 12 and ok != _ls_brute(list(s), W):
            raise ArithmeticError(f"{fname}: result failed verification (Luccio-Sami test)")
        if ok:
            keep.append(s)
    return _parts_expr(gr, _restrict(gr, keep, vs, fname))


# ─────────────────────────── CommunityGraphPlot ───────────────────────────

def _hull(points):
    """Convex hull (Andrew's monotone chain), counter-clockwise."""
    pts = sorted(set((round(x, 9), round(y, 9)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _community_region(X, members, pad):
    pts = []
    for v in members:
        for k in range(24):
            t = 2 * math.pi * k / 24
            pts.append((X[v][0] + pad * math.cos(t), X[v][1] + pad * math.sin(t)))
    return _hull(pts)


def _cyclic_styles(spec, k, fname, what, default, none_ok=True):
    if spec is None or sym(spec, "Automatic"):
        return [default(i) for i in range(k)]
    if sym(spec, "None"):
        return [None] * k
    if isinstance(spec, str) or K.is_int(spec):
        raise NotImplementedError(f"{fname}: ColorData schemes in {what} are not supported; give directives")
    items = list(spec.args) if is_list(spec) else [spec]
    if not items:
        raise ValueError(f"{fname}: {what} must not be an empty list")
    sts = [_style_of(s, fname, what) for s in items]
    return [sts[i % len(sts)] for i in range(k)]


def _community_layout(gr, parts, fname):
    """Community centres by a spring-electrical layout of the community graph (scaled by community size),
    members by a spring-electrical layout of their induced subgraph around the centre."""
    np = _np()
    W = weighted_simple(gr)
    comm = {}
    for c, p in enumerate(parts):
        for v in p:
            comm[v] = c
    k = len(parts)
    cw = {}
    for (i, j), w in W.items():
        a, b = comm[i], comm[j]
        if a != b:
            key = (min(a, b), max(a, b))
            cw[key] = cw.get(key, 0.0) + w
    C = spring_electrical(k, sorted(cw), 2) if k > 1 else np.zeros((1, 2))
    radius = [0.6 * math.sqrt(len(p)) + 0.5 for p in parts]
    C = C * (2.2 * max(radius))
    X = np.zeros((gr.n, 2))
    for c, p in enumerate(parts):
        idx = {v: t for t, v in enumerate(p)}
        pairs = sorted({(min(idx[i], idx[j]), max(idx[i], idx[j])) for (i, j) in W if i in idx and j in idx})
        blocks = []
        sub = {}
        for t, v in enumerate(p):
            sub[t] = []
        for a, b in pairs:
            sub[a].append(b)
            sub[b].append(a)
        for comp in _components([sub[t] for t in range(len(p))]):
            cid = {v: t for t, v in enumerate(comp)}
            pp = [(cid[a], cid[b]) for a, b in pairs if a in cid and b in cid]
            blocks.append((comp, spring_electrical(len(comp), pp, 2)))
        P = pack(blocks, 2)
        Y = np.array([P[t] for t in range(len(p))], dtype=float)
        Y = Y - Y.mean(0)
        for t, v in enumerate(p):
            X[v] = C[c] + Y[t]
    return X


def _community_list(spec, gr, fname):
    if not is_list(spec):
        if isinstance(spec, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the communities must be a list of vertex lists")
    parts, labels = [], []
    for c in spec.args:
        lab = None
        if isinstance(c, Expr) and c.head == "Labeled" and len(c.args) >= 2:
            c, lab = c.args[0], c.args[1]
        if not is_list(c) or not c.args:
            raise TypeError(f"{fname}: each community must be a non-empty list of vertices, not {K_input(c)}")
        parts.append([_vindex(gr, v, fname) for v in c.args])
        labels.append(lab)
    seen = [v for p in parts for v in p]
    if len(seen) != len(set(seen)):
        raise NotImplementedError(f"{fname}: overlapping communities are not supported")
    rest = [v for v in range(gr.n) if v not in set(seen)]
    for v in rest:                                           # vertices in no community are drawn on their own
        parts.append([v])
        labels.append(None)
    return parts, labels, len(spec.args)


@fn("CommunityGraphPlot", area=AREA, status="added",
    notes="CommunityGraphPlot[g] (communities from FindGraphCommunities[g], modularity method, NetworkX) and "
          "CommunityGraphPlot[g, {{v11, ...}, ...}] (given communities; Labeled[{...}, lbl] labels one; vertices "
          "in no community are drawn on their own; overlapping communities raise NotImplementedError) -> "
          "Graphics. Layout: the community graph (one vertex per community, edges weighted by the edges between "
          "communities) is laid out with the spring-electrical model to place the community centres, and each "
          "community's induced subgraph is laid out around its centre. Each community is drawn as the convex hull "
          "of disks around its vertices: CommunityRegionStyle (Automatic: light DefaultPlotColors with "
          "Opacity 0.15; None; a directive or a list used cyclically) fills it and CommunityBoundaryStyle "
          "(Automatic: gray; None; directives) draws its outline; CommunityLabels -> None or {lbl1, ...} puts "
          "a label above each community (Placed raises NotImplementedError); ColorData schemes raise "
          "NotImplementedError. " + DRAW_NOTE + " Check: self-check (finite layout; the communities partition "
          "the drawn vertices) and reference (one region polygon per community, containing its vertices, in "
          "tests).")
def _communitygraphplot(g: Expr, *rest: Expr | Option) -> Expr:
    fname = "CommunityGraphPlot"
    pos, opts = split_opts((g,) + rest, fname)
    if len(pos) > 2:
        raise TypeError(f"{fname}: too many arguments")
    rs = opts.pop("CommunityRegionStyle", None)
    bs = opts.pop("CommunityBoundaryStyle", None)
    cl = opts.pop("CommunityLabels", None)
    pl = opts.pop("PlotLegends", None)
    if pl is not None and not sym(pl, "None"):
        raise NotImplementedError(f"{fname}: PlotLegends is not supported")
    method = _method_name(opts, fname, "Modularity")
    gr = graph_input(pos[0], fname)
    if len(pos) == 2:
        parts, labels, k = _community_list(pos[1], gr, fname)
    else:
        parts = find_communities(gr, method, fname)
        if method == "CliquePercolation":
            covered = {v for p in parts for v in p}
            if len(covered) != sum(len(p) for p in parts):
                raise NotImplementedError(f"{fname}: overlapping communities are not supported")
            parts = parts + [[v] for v in range(gr.n) if v not in covered]
            k = len(parts) - (gr.n - len(covered))
        else:
            k = len(parts)
        labels = [None] * len(parts)
    _check_partition(parts, gr.n, fname)
    gopts, gfx = merged_options(gr, opts, fname)
    if "GraphLayout" in gopts and not sym(gopts["GraphLayout"], "Automatic"):
        raise NotImplementedError(f"{fname}: GraphLayout is not supported (the community layout is used)")
    vc = gopts.get("VertexCoordinates")
    if vc is not None and not sym(vc, "Automatic"):
        X, _ = _vertex_coordinates(gr, vc, fname, 2)
    else:
        X = _community_layout(gr, parts, fname)
    check_layout(X, gr.n, 2, fname)
    if cl is not None and not sym(cl, "None"):
        if isinstance(cl, Expr) and cl.head == "Placed":
            raise NotImplementedError(f"{fname}: Placed community labels are not supported")
        if not is_list(cl):
            raise ValueError(f"{fname}: CommunityLabels must be None or a list of labels")
        for i, lab in enumerate(cl.args[:k]):
            if labels[i] is None:
                labels[i] = lab
    regions = _cyclic_styles(rs, k, fname, "CommunityRegionStyle",
                             lambda i: Expr("Directive", Expr("Opacity", 0.15), rgb(DEFAULT_COLORS[i % 10])))
    bounds = _cyclic_styles(bs, k, fname, "CommunityBoundaryStyle", lambda i: rgb((0.5, 0.5, 0.5)))
    dmin = _min_dist(X)
    pad = 0.35 * dmin + 0.15
    content = []
    texts = []
    for i in range(k):
        hull = _community_region(X, parts[i], pad)
        poly = Expr("Polygon", L(*(pt(p) for p in hull)))
        face = regions[i] if regions[i] is not None else None
        edge = Expr("EdgeForm", bounds[i]) if bounds[i] is not None else Expr("EdgeForm", L())
        sty = [edge, face if face is not None else Expr("FaceForm", L())]
        content.append(L(*sty, poly))
        if labels[i] is not None:
            top = max(p[1] for p in hull)
            cx = sum(X[v][0] for v in parts[i]) / len(parts[i])
            texts.append(Expr("Text", labels[i], pt((cx, top + 0.3 * pad))))
    vcol = [None] * gr.n
    content += draw_graph(gr, X, {}, gopts, fname, 2, vertex_colors=vcol)
    if texts:
        content.append(L(GrayLevel(0), *texts))
    return Expr("Graphics", L(*content), *_graphics_rules(gfx, 2, fname))


def _option_symbol(name, notes):
    from .wl_graphics_color import inert_option
    inert_option(name, notes, AREA)


_option_symbol("CommunityBoundaryStyle", "CommunityBoundaryStyle: option of CommunityGraphPlot styling the community "
               "outlines: Automatic (gray), None, a directive, or a list of directives used cyclically (ColorData "
               "names and indices raise NotImplementedError). Inert as an expression. Check: reference (tests "
               "read the outline styles of the drawn regions).")
_option_symbol("CommunityRegionStyle", "CommunityRegionStyle: option of CommunityGraphPlot styling the community "
               "regions: Automatic (light DefaultPlotColors, Opacity 0.15), None, a directive, or a list used "
               "cyclically (ColorData schemes raise NotImplementedError). Inert as an expression. Check: reference "
               "(tests read the fill styles of the drawn regions).")
_option_symbol("CommunityLabels", "CommunityLabels: option of CommunityGraphPlot: None or {lbl1, lbl2, ...} for "
               "successive communities, drawn as Text above each region (Placed raises NotImplementedError). Inert "
               "as an expression. Check: reference (tests find the label Text primitives).")
_option_symbol("GraphLayers", "GraphLayers: option of LayeredGraphPlot3D: None or Automatic (no layer planes), m, "
               "{m1, ...} or All: translucent planes through those layers (layer 1 holds the roots). Inert as an "
               "expression. Check: reference (tests count the plane polygons and their heights).")
_option_symbol("GraphLayerStyle", "GraphLayerStyle: option of LayeredGraphPlot3D: a directive, or a list of "
               "directives used cyclically, for the layer planes (named colour schemes raise NotImplementedError). "
               "Inert as an expression. Check: reference (tests read the plane styles).")
_option_symbol("EdgeValueRange", "EdgeValueRange: option of GraphValuePlot (and GeoGraphValuePlot, not available "
               "here): Automatic or All (all edges), or {min, max}: edges whose values lie outside are drawn "
               "without the value encoding (thin and gray). Inert as an expression. Check: reference (tests "
               "compare the encoded edges).")
_option_symbol("EdgeValueSizes", "EdgeValueSizes: option of GraphValuePlot (and GeoGraphValuePlot, not available "
               "here) for the \"EdgeThickness\" encoding: Automatic ({0.002, 0.012}), Tiny/Small/Medium/Large, "
               "{smin, smax} or s, thicknesses as fractions of the plot width (Thickness). Inert as an expression. "
               "Check: reference (tests read the Thickness directives).")
_option_symbol("EscapeRadius", "EscapeRadius: option of MandelbrotSetPlot: iteration stops once |z_n| > r; r must "
               "be a real number >= 2 (default 2). Inert as an expression. Check: reference (tests "
               "compare escape counts at two radii).")


# ═══════════════════════════════════════ GraphValuePlot ═══════════════════════════════════════

VALUE_GRADIENT = ((0.26, 0.15, 0.44), (0.27, 0.42, 0.71), (0.45, 0.69, 0.72), (0.86, 0.86, 0.51),
                  (0.96, 0.75, 0.37), (0.94, 0.49, 0.24))
_V_QUANT = ("VertexColor", "VertexOpacity", "VertexSize", "DiscreteVertexColor")
_V_CAT = ("VertexShape", "VertexLabel", "VertexFilling")
_E_QUANT = ("EdgeColor", "EdgeThickness", "EdgeOpacity", "DiscreteEdgeColor")
_E_CAT = ("EdgeShape",)
_MEASURES = ("AuthorityCentrality", "BetweennessCentrality", "ClosenessCentrality", "DegreeCentrality",
             "EigenvectorCentrality", "HubCentrality", "PageRankCentrality")
_ANNOTATIONS = {"EdgeCapacity": "e", "EdgeCost": "e", "EdgeWeight": "e", "VertexCapacity": "v", "VertexWeight": "v"}


def _blend(t):
    t = 0.0 if not math.isfinite(t) else min(1.0, max(0.0, t))
    st = VALUE_GRADIENT
    n = len(st) - 1
    k = min(int(t * n), n - 1)
    u = t * n - k
    return tuple(st[k][i] + (st[k + 1][i] - st[k][i]) * u for i in range(3))


def _numbers(vals, fname):
    out = []
    for v in vals:
        if isinstance(v, Expr) and v.head == "Missing":
            out.append(None)
            continue
        out.append(_real(v, fname, "a quantitative value"))
    return out


def _scaled(nums):
    present = [x for x in nums if x is not None]
    if not present:
        return [None] * len(nums)
    lo, hi = min(present), max(present)
    return [None if x is None else (0.5 if hi == lo else (x - lo) / (hi - lo)) for x in nums]


def _value_specs(gr, spec, fname):
    """[(kind 'v'|'e', values, encoding)] from the second argument of GraphValuePlot."""
    n, m = gr.n, len(gr.edges)

    def default_enc(kind, vals):
        quant = all(isinstance(v, Expr) and v.head == "Missing" or is_numeric_value(v) for v in vals)
        if kind == "v":
            return "VertexColor" if quant else "VertexFilling"
        return "EdgeThickness" if quant else "EdgeShape"

    def kind_of(vals, enc):
        if enc is not None:
            return "v" if enc in _V_QUANT + _V_CAT else "e"
        if len(vals) == n:
            return "v"
        if len(vals) == m:
            return "e"
        raise ValueError(f"{fname}: {len(vals)} values match neither the {n} vertices nor the {m} edges")

    def one(vals_expr, enc):
        if enc is not None and not isinstance(enc, str):
            raise ValueError(f"{fname}: the encoding must be a string such as \"VertexColor\", not {K_input(enc)}")
        if enc is not None and enc not in _V_QUANT + _V_CAT + _E_QUANT + _E_CAT:
            raise ValueError(f"{fname}: unknown visual encoding {enc!r}")
        if isinstance(vals_expr, str):
            kind, vals = _named_values(gr, vals_expr, fname)
            if vals_expr == "DegreeCentrality" and enc is None:
                return [(kind, vals, "VertexSize"), (kind, vals, "VertexColor")]
        elif is_list(vals_expr):
            vals = list(vals_expr.args)
            kind = kind_of(vals, enc)
        elif isinstance(vals_expr, (Symbol, Expr)):
            raise Unevaluated()
        else:
            raise TypeError(f"{fname}: the values must be a list, an annotation or a measure name")
        if enc is not None and (kind == "v") != (enc in _V_QUANT + _V_CAT):
            raise ValueError(f"{fname}: the encoding {enc!r} does not apply to {'vertex' if kind == 'v' else 'edge'} "
                             "values")
        if len(vals) != (n if kind == "v" else m):
            raise ValueError(f"{fname}: expected {n if kind == 'v' else m} values, got {len(vals)}")
        return [(kind, vals, enc or default_enc(kind, vals))]
    if is_rule(spec) and not isinstance(spec.args[0], str) and not is_list(spec.args[0]) and \
            not isinstance(spec.args[0], Expr):
        raise ValueError(f"{fname}: invalid value specification {K_input(spec)}")
    if is_rule(spec):
        return one(spec.args[0], spec.args[1])
    if is_list(spec) and spec.args and all(is_rule(r) for r in spec.args):
        out = []
        for r in spec.args:
            if is_list(r.args[1]) or K.tag(r.args[0]) in gr.index or isinstance(r.args[0], Expr) and \
                    r.args[0].head in ("UndirectedEdge", "DirectedEdge"):
                raise NotImplementedError(f"{fname}: per-item specifications w -> {{val -> enc, ...}} are not "
                                          "supported")
            out += one(r.args[0], r.args[1])
        return out
    return one(spec, None)


def _named_values(gr, name, fname):
    if name in _ANNOTATIONS:
        kind = _ANNOTATIONS[name]
        if kind == "e":
            try:
                vals = gr._edge_data(name)
            except GR.GraphError as exc:
                raise ValueError(str(exc)) from None
        else:
            w = gr.opts.get(name)
            vals = list(w.args) if is_list(w) and len(w.args) == gr.n else None
        if vals is None:
            raise ValueError(f"{fname}: the graph has no {name} annotation")
        return kind, [K.from_sp(v) if isinstance(v, sp.Basic) else v for v in vals]
    if name in _MEASURES:
        r = ev(Expr(name, gr.expr()))
        if not is_list(r) or len(r.args) != gr.n:
            raise ValueError(f"{fname}: {name} could not be computed for this graph")
        return "v", list(r.args)
    raise ValueError(f"{fname}: {name!r} is not a known annotation or measure (annotations: "
                     f"{', '.join(_ANNOTATIONS)}; measures: {', '.join(_MEASURES)})")


def _edge_sizes(spec, fname):
    if spec is None or sym(spec, "Automatic"):
        return 0.002, 0.012
    pre = {"Tiny": 0.001, "Small": 0.003, "Medium": 0.006, "Large": 0.012}
    if sym(spec, *pre):
        return pre[spec.name] / 2, pre[spec.name]
    if is_list(spec) and len(spec.args) == 2:
        lo = 0.002 if sym(spec.args[0], "Automatic") else _real(spec.args[0], fname, "EdgeValueSizes", lo=0)
        hi = 0.012 if sym(spec.args[1], "Automatic") else _real(spec.args[1], fname, "EdgeValueSizes", lo=0)
        if lo > hi:
            raise ValueError(f"{fname}: EdgeValueSizes {{smin, smax}} needs smin <= smax")
        return lo, hi
    s = _real(spec, fname, "EdgeValueSizes", lo=0)
    return s, s


@fn("GraphValuePlot", area=AREA, status="added",
    notes="GraphValuePlot[g, {val1, ...}] (one value per vertex, or per edge when the count matches the edges), "
          "[g, vals -> enc], [g, {vals1 -> enc1, ...}], [g, \"ann\"] (EdgeWeight, EdgeCapacity, EdgeCost, "
          "VertexWeight, VertexCapacity annotations of g) and [g, \"measure\"] (AuthorityCentrality, "
          "BetweennessCentrality, ClosenessCentrality, DegreeCentrality (size and colour), EigenvectorCentrality, "
          "HubCentrality, PageRankCentrality, computed by those functions) -> Graphics of the GraphPlot layout "
          "with the values encoded. Quantitative values are rescaled linearly from their minimum to their "
          "maximum: \"VertexColor\"/\"EdgeColor\" along a fixed 6-stop gradient, \"VertexSize\" radius 0.08 to 0.3 "
          "times the smallest vertex distance, \"VertexOpacity\"/\"EdgeOpacity\" 0.15 to 1, "
          "\"EdgeThickness\" Thickness from EdgeValueSizes (Automatic {0.002, 0.012}), "
          "\"DiscreteVertexColor\"/\"DiscreteEdgeColor\" five equal-width bins; categorical values: "
          "\"VertexFilling\" (one DefaultPlotColors colour per distinct value, in order of appearance) and "
          "\"VertexLabel\" (the value as a label). Defaults: \"VertexColor\" (numbers) or \"VertexFilling\" "
          "(other values) for vertices, \"EdgeThickness\" for edges. EdgeValueRange -> {min, max} leaves edges "
          "outside unencoded (thin, gray). Missing[] values are drawn unencoded. \"VertexShape\", \"EdgeShape\", "
          "per-item specifications w -> {...} and legends raise NotImplementedError. Gradient and sizes are this "
          "module's own. Check: self-check (finite layout) and reference (tests check colours and sizes are "
          "monotone in the values).")
def _graphvalueplot(g: Expr, spec: Expr, *opts: Option) -> Expr:
    fname = "GraphValuePlot"
    pos, o = split_opts((g, spec) + opts, fname)
    if len(pos) != 2:
        raise TypeError(f"{fname}: a graph and a value specification are required")
    evr = o.pop("EdgeValueRange", None)
    evs = o.pop("EdgeValueSizes", None)
    vvr = o.pop("VertexValueRange", None)
    pl = o.pop("PlotLegends", None)
    if pl is not None and not sym(pl, "None"):
        raise NotImplementedError(f"{fname}: PlotLegends is not supported")
    if vvr is not None and not sym(vvr, "Automatic", "All"):
        raise NotImplementedError(f"{fname}: VertexValueRange is not supported")
    gr = graph_input(pos[0], fname)
    specs = _value_specs(gr, pos[1], fname)
    gopts, gfx = merged_options(gr, o, fname)
    lay = parse_layout(gopts.get("GraphLayout"), fname, 2)
    vc = gopts.get("VertexCoordinates")
    if vc is not None and not sym(vc, "Automatic"):
        X, routes = _vertex_coordinates(gr, vc, fname, 2)
    else:
        X, routes = compute_layout(gr, lay, fname)
    check_layout(X, gr.n, 2, fname)
    n, m = gr.n, len(gr.edges)
    dmin = _min_dist(X)
    vcol = [None] * n
    vrad = None
    vextra = [[] for _ in range(n)]
    labels = None
    estyle = [[] for _ in range(m)]
    lo_e, hi_e = _edge_sizes(evs, fname)
    erange = None
    if evr is not None and not sym(evr, "Automatic", "All"):
        if not (is_list(evr) and len(evr.args) == 2):
            raise ValueError(f"{fname}: EdgeValueRange must be Automatic, All or {{min, max}}")
        erange = (_real(evr.args[0], fname, "EdgeValueRange"), _real(evr.args[1], fname, "EdgeValueRange"))
    for kind, vals, enc in specs:
        if enc in ("VertexShape", "EdgeShape"):
            raise NotImplementedError(f"{fname}: the encoding {enc!r} is not supported")
        if enc == "VertexFilling":
            cats = []
            for v in vals:
                if not any(K.tag(v) == K.tag(c) for c in cats):
                    cats.append(v)
            for k, v in enumerate(vals):
                c = next(i for i, cc in enumerate(cats) if K.tag(cc) == K.tag(v))
                vcol[k] = rgb(DEFAULT_COLORS[c % 10])
            continue
        if enc == "VertexLabel":
            labels = list(vals)
            continue
        nums = _numbers(vals, fname)
        if kind == "e" and erange is not None:
            nums = [None if x is None or not (erange[0] <= x <= erange[1]) else x for x in nums]
        t = _scaled(nums)
        for k, tk in enumerate(t):
            if tk is None:
                if kind == "e" and erange is not None and enc in ("EdgeThickness", "EdgeColor", "EdgeOpacity",
                                                                    "DiscreteEdgeColor"):
                    estyle[k] = [GrayLevel(0.7), Expr("Thickness", fl(lo_e / 2))]
                continue
            if enc == "VertexColor":
                vcol[k] = rgb(_blend(tk))
            elif enc == "DiscreteVertexColor":
                vcol[k] = rgb(_blend(min(int(tk * 5), 4) / 4))
            elif enc == "VertexOpacity":
                vextra[k].append(Expr("Opacity", fl(0.15 + 0.85 * tk)))
            elif enc == "VertexSize":
                if vrad is None:
                    vrad = [0.15 * dmin] * n
                vrad[k] = (0.08 + 0.22 * tk) * dmin
            elif enc == "EdgeColor":
                estyle[k].append(rgb(_blend(tk)))
            elif enc == "DiscreteEdgeColor":
                estyle[k].append(rgb(_blend(min(int(tk * 5), 4) / 4)))
            elif enc == "EdgeOpacity":
                estyle[k].append(Expr("Opacity", fl(0.15 + 0.85 * tk)))
            elif enc == "EdgeThickness":
                estyle[k].append(Expr("Thickness", fl(lo_e + (hi_e - lo_e) * tk)))
            else:
                raise ValueError(f"{fname}: the encoding {enc!r} needs categorical values")
    es = [L(*s) if s else None for s in estyle]
    ve = [x if x else None for x in vextra]
    content = draw_graph(gr, X, routes, gopts, fname, 2, vertex_colors=vcol, vertex_radii=vrad, edge_styles=es,
                         vertex_extra=ve, labels_override=labels)
    return Expr("Graphics", L(*content), *_graphics_rules(gfx, 2, fname))


# ═══════════════════════════════════════ TaggedNestGraph ═══════════════════════════════════════

@fn("TaggedNestGraph", area=AREA, status="added",
    notes="TaggedNestGraph[f, expr | {expr1, ...} | graph, n] (n = 1 by default; Infinity allowed when the "
          "graph is finite, up to 100000 vertices): breadth-first, each new vertex x is expanded once into the "
          "tagged edges DirectedEdge[x, y_i, i] for the elements y_i of f[x] when it is a list (tags 1, 2, ... by "
          "position), or DirectedEdge[x, f[x], 1] otherwise; repeated (x, y, tag) edges are kept once. A graph "
          "start keeps its vertices and edges and expands all its vertices. DirectedEdges -> False gives "
          "UndirectedEdge[x, y, i]; other options are Graph options. Check: exact (tests compare with the "
          "documented construction, e.g. Mod[#^2 + 1, 10]&, and with NestGraph's edges without tags).")
def _taggednestgraph(f: Fn, start: Expr, *rest: Expr | Option) -> Expr:
    fname = "TaggedNestGraph"
    pos, opts = split_opts((f, start) + rest, fname)
    if not 2 <= len(pos) <= 3:
        raise TypeError(f"{fname}: 2 or 3 arguments expected")
    if len(pos) == 3:
        if sym(pos[2], "Infinity") or (isinstance(pos[2], Expr) and pos[2].head == "DirectedInfinity"):
            n = None
        else:
            n = _posint(pos[2], fname, "the number of steps", lo=0)
    else:
        n = 1
    de = opts.pop("DirectedEdges", True)
    if not (de is True or de is False):
        raise ValueError(f"{fname}: DirectedEdges must be True or False")
    eh = "DirectedEdge" if de else "UndirectedEdge"
    verts, vseen, edges, eseen = [], set(), [], set()

    def addv(v):
        t = K.tag(v)
        if t in vseen:
            return False
        vseen.add(t)
        verts.append(v)
        return True

    def adde(e):
        key = (e.head,) + tuple(K.tag(a) for a in e.args)
        if e.head == "UndirectedEdge":
            a, b = sorted([K.tag(e.args[0]), K.tag(e.args[1])], key=repr)
            key = (e.head, a, b) + tuple(K.tag(x) for x in e.args[2:])
        if key not in eseen:
            eseen.add(key)
            edges.append(e)
    s0 = pos[1]
    if isinstance(s0, Expr) and s0.head == "Graph":
        g0 = graph_input(s0, fname)
        for v in g0.vertices:
            addv(v)
        for e in g0.edge_exprs:
            adde(e)
        frontier = list(g0.vertices)
    elif is_list(s0):
        frontier = [v for v in s0.args if addv(v)]
    else:
        addv(s0)
        frontier = [s0]
    expanded = set()
    level = 0
    while frontier and (n is None or level < n):
        nxt = []
        for x in frontier:
            if K.tag(x) in expanded:
                continue
            expanded.add(K.tag(x))
            r = apply(pos[0], x)
            items = r.args if is_list(r) else (r,)
            for k, y in enumerate(items, 1):
                adde(Expr(eh, x, y, k))
                if addv(y):
                    nxt.append(y)
            if len(verts) > 100000:
                raise NotImplementedError(f"{fname}: the graph is too large (more than 100000 vertices)")
        frontier = nxt
        level += 1
    rules = [Expr("Rule", Symbol(k), v) for k, v in opts.items()]
    return ev(Expr("Graph", L(*verts), L(*edges), *rules))


# ═══════════════════════════════════════ GraphPropertyDistribution ═══════════════════════════════════════

_GRAPH_DISTS = ("BernoulliGraphDistribution", "UniformGraphDistribution", "WattsStrogatzGraphDistribution",
                "BarabasiAlbertGraphDistribution", "PriceGraphDistribution", "DegreeGraphDistribution",
                "SpatialGraphDistribution", "StochasticBlockModelGraphDistribution")


@fn("GraphPropertyDistribution", area=AREA, status="added",
    notes="GraphPropertyDistribution[expr, x \\[Distributed] gdist] and [expr, {x1 \\[Distributed] gd1, ...}] "
          "(HoldFirst): the distribution of the graph property expr. Closed forms (exact): EdgeCount[x] with "
          "BernoulliGraphDistribution[n, p] -> BinomialDistribution[n (n - 1)/2, p]; VertexDegree[x, v] with "
          "BernoulliGraphDistribution[n, p] -> BinomialDistribution[n - 1, p]; VertexDegree[x, v] with "
          "UniformGraphDistribution[n, m] -> HypergeometricDistribution[m, n - 1, n (n - 1)/2] (the edges at v "
          "among m of the n (n - 1)/2 possible edges). Every other property stays as the symbolic distribution "
          "object GraphPropertyDistribution[expr, x \\[Distributed] gdist]; simulation (RandomVariate), "
          "NProbability and NExpectation of such objects are not provided. A second argument that is not "
          "x \\[Distributed] graph-distribution raises TypeError. Check: exact (closed forms; tests compare "
          "means and variances with direct counting).")
def _graphpropertydistribution(expr: Held, spec: Expr) -> Expr:
    fname = "GraphPropertyDistribution"
    items = list(spec.args) if is_list(spec) else [spec]
    for it in items:
        if not (isinstance(it, Expr) and it.head == "Distributed" and len(it.args) == 2):
            raise TypeError(f"{fname}: the second argument must be x \\[Distributed] gdist (or a list of them), "
                            f"not {K_input(it)}")
        d = it.args[1]
        if not (isinstance(d, Expr) and d.head in _GRAPH_DISTS):
            if isinstance(d, Symbol):
                continue
            raise TypeError(f"{fname}: {K_input(d)} is not a random graph distribution")
    if len(items) == 1 and isinstance(expr, Expr) and expr.head in ("EdgeCount", "VertexDegree") and \
            expr.args and expr.args[0] == items[0].args[0]:
        x, d = items[0].args
        if d.head == "BernoulliGraphDistribution" and len(d.args) == 2:
            nn, p = d.args
            if expr.head == "EdgeCount" and len(expr.args) == 1:
                return ev(Expr("BinomialDistribution", Expr("Times", Expr("Rational", 1, 2), nn,
                                                            Expr("Plus", nn, -1)), p))
            if expr.head == "VertexDegree" and len(expr.args) == 2:
                return ev(Expr("BinomialDistribution", Expr("Plus", nn, -1), p))
        if d.head == "UniformGraphDistribution" and len(d.args) == 2 and expr.head == "VertexDegree" and \
                len(expr.args) == 2:
            nn, mm = d.args
            return ev(Expr("HypergeometricDistribution", mm, Expr("Plus", nn, -1),
                           Expr("Times", Expr("Rational", 1, 2), nn, Expr("Plus", nn, -1))))
    raise Unevaluated()


def _hold_first(*names):
    from ..evaluate import set_hold_first
    for n in names:
        set_hold_first(n)


_hold_first("GraphPropertyDistribution")


# ═══════════════════════════════════════ TreeGamePlot ═══════════════════════════════════════

def _game_tree_nodes(root):
    """Breadth-first list of (node, parent index, path) and children index lists."""
    nodes, parent, paths, kids = [root], [None], [()], [[]]
    k = 0
    while k < len(nodes):
        u = nodes[k]
        for i, c in enumerate(u.children, 1):
            kids[k].append(len(nodes))
            nodes.append(c)
            parent.append(k)
            paths.append(paths[k] + (i,))
            kids.append([])
        k += 1
    return nodes, parent, paths, kids


def _num_f(x):
    return float(K.to_sp(x))


@fn("TreeGamePlot", area=AREA, status="added",
    notes="TreeGamePlot[tgame] and TreeGamePlot[tgame, strat] (a strategy <|player -> <|path -> {pr, ...}|>|> or "
          "<|path -> {pr, ...}|> as for TreeGamePayoff) -> Graphics of the game tree drawn with the "
          "Reingold-Tilford tidy layout (GraphLayout \"LayeredDigraphEmbedding\" or \"LayeredEmbedding\", root at "
          "the top) or the radial layout (\"RadialEmbedding\"): action nodes are disks coloured by player "
          "(DefaultPlotColors in order of first appearance) with the player as a label "
          "(GamePlayerLabels -> Automatic | None); end nodes show their payoff list; actions are lines labelled "
          "with their labels at the midpoints (GameActionLabels -> Automatic | None); chance probabilities in the "
          "game set the thickness (AbsoluteThickness 1 + 4 p), strategy probabilities the shade "
          "(Opacity 0.15 + 0.85 p, unspecified actions full); action nodes of one player sharing an information "
          "set are joined by dashed lines. Appearance -> None hides probabilities and information sets. "
          "ColorFunction, PlotLegends, GraphHighlight, EdgeWeight, VertexCoordinates and the other GraphLayout "
          "settings accept only their defaults (others raise NotImplementedError). Check: reference (tests "
          "check the layout, colours and shading against the game).")
def _treegameplot(game: Expr, *rest: Expr | Option) -> Expr:
    fname = "TreeGamePlot"
    from . import wl_trees as TR
    pos, o = split_opts((game,) + rest, fname)
    if not 1 <= len(pos) <= 2:
        raise TypeError(f"{fname}: a tree game and optionally a strategy are expected")
    try:
        root = TR._game_root(pos[0], fname)
        table = TR._strategy_table(pos[1], fname) if len(pos) == 2 else {}
    except TR.TreeError as exc:
        raise ValueError(str(exc)) from None
    appearance = o.pop("Appearance", Symbol("All"))
    if not sym(appearance, "All", "None", "Automatic"):
        raise ValueError(f"{fname}: Appearance must be All or None")
    show_extra = not sym(appearance, "None")
    gal = o.pop("GameActionLabels", Symbol("Automatic"))
    gpl = o.pop("GamePlayerLabels", Symbol("Automatic"))
    for nm, v in (("GameActionLabels", gal), ("GamePlayerLabels", gpl)):
        if not (sym(v, "Automatic", "None", "All") or v is True or v is False):
            raise NotImplementedError(f"{fname}: {nm} -> {K_input(v)} is not supported (Automatic or None)")
    lay = o.pop("GraphLayout", "LayeredDigraphEmbedding")
    if lay not in ("LayeredDigraphEmbedding", "LayeredEmbedding", "RadialEmbedding") and not sym(lay, "Automatic"):
        raise NotImplementedError(f"{fname}: the layout {K_input(lay)} is not supported")
    for nm in ("ColorFunction", "GraphHighlight", "EdgeWeight", "VertexCoordinates", "PlotTheme",
               "EdgeLabelStyle", "VertexLabelStyle", "GraphHighlightStyle"):
        v = o.pop(nm, None)
        if v is not None and not sym(v, "Automatic", "None") and not (nm == "GraphHighlight" and is_list(v)
                                                                       and not v.args):
            raise NotImplementedError(f"{fname}: {nm} -> {K_input(v)} is not supported (only Automatic)")
    pl = o.pop("PlotLegends", None)
    if pl is not None and not (pl is False or sym(pl, "None")):
        raise NotImplementedError(f"{fname}: PlotLegends is not supported")
    nodes, parent, paths, kids = _game_tree_nodes(root)
    children = {k: kids[k] for k in range(len(nodes))}
    if lay == "RadialEmbedding":
        P = radial_tree(children, 0)
    else:
        xy = tidy_tree(children, 0)
        P = {v: orient(xy[v], "Top") for v in xy}
    players = []
    for u in nodes:
        if u.player is not None and not any(K.tag(u.player) == K.tag(p) for p in players):
            players.append(u.player)

    def pcol(p):
        return rgb(DEFAULT_COLORS[next(i for i, q in enumerate(players) if K.tag(q) == K.tag(p)) % 10])
    edges, marks, texts = [], [], []
    for k in range(1, len(nodes)):
        u, p = nodes[k], parent[k]
        a, b = P[p], P[k]
        sty = [GrayLevel(0.4)]
        if show_extra and u.prob is not None:
            sty.append(Expr("AbsoluteThickness", fl(1 + 4 * _num_f(u.prob))))
        ppath = paths[p]
        if ppath in table:
            ps = table[ppath]
            if len(ps) != len(nodes[p].children):
                raise ValueError(f"{fname}: the strategy at node {list(ppath)} has {len(ps)} probabilities for "
                                 f"{len(nodes[p].children)} actions")
            sty.append(Expr("Opacity", fl(0.15 + 0.85 * _num_f(ps[paths[k][-1] - 1]))))
        edges.append(L(*sty, Expr("Line", L(pt(a), pt(b)))))
        if u.label is not None and not sym(gal, "None") and gal is not False:
            texts.append(Expr("Text", u.label, pt(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)), L(1, 0)))
    for t in table:
        if t not in paths or not nodes[paths.index(t)].children:
            raise ValueError(f"{fname}: the strategy refers to {list(t)}, which is not an action node")
    r = 0.12
    for k, u in enumerate(nodes):
        x = P[k]
        if u.payoffs is not None:
            texts.append(Expr("Text", L(*u.payoffs), pt((x[0], x[1] - 0.05)), L(0, 1)))
            marks.append(L(GrayLevel(0.4), Expr("Disk", pt(x), fl(r / 3))))
        else:
            marks.append(L(pcol(u.player), Expr("Disk", pt(x), fl(r))))
            if not sym(gpl, "None") and gpl is not False:
                texts.append(Expr("Text", u.player, pt((x[0] + 1.2 * r, x[1])), L(-1, 0)))
    info = []
    if show_extra:
        groups = {}
        for k, u in enumerate(nodes):
            if u.player is not None and u.infoset is not None:
                groups.setdefault((K.tag(u.player), K.tag(u.infoset)), []).append(k)
        for ks in groups.values():
            if len(ks) > 1:
                ks.sort(key=lambda v: (P[v][1], P[v][0]))
                info.append(L(GrayLevel(0.5), Expr("Dashing", L(0.01, 0.01)),
                              Expr("Line", L(*[pt(P[v]) for v in ks]))))
    content = L(L(*edges), L(*info), L(*marks), L(GrayLevel(0), *texts))
    return Expr("Graphics", content, *_graphics_rules(o, 2, fname))


# ═══════════════════════════════════════ Julia and Mandelbrot plots ═══════════════════════════════════════

FRACTAL_GRADIENT = ((0.05, 0.03, 0.25), (0.12, 0.33, 0.73), (0.55, 0.82, 0.95), (1.0, 0.97, 0.85),
                    (0.99, 0.67, 0.18), (0.6, 0.15, 0.05))


def _fgrad(t):
    st = FRACTAL_GRADIENT
    n = len(st) - 1
    t = min(1.0, max(0.0, t))
    k = min(int(t * n), n - 1)
    u = t * n - k
    return tuple(st[k][i] + (st[k + 1][i] - st[k][i]) * u for i in range(3))


def _color_table(cf, scaling, counts, mx, fname):
    """{count: (r, g, b)} for the distinct escape counts (mx + 1 = never escaped)."""
    from .wl_graphics_render import to_rgba
    out = {}
    for k in counts:
        t = 1.0 if k > mx else k / (mx + 1)
        if cf is None or sym(cf, "Automatic"):
            out[k] = (0.0, 0.0, 0.0) if k > mx else _fgrad(math.sqrt(t))
            continue
        if sym(cf, "None"):
            out[k] = (0.0, 0.0, 0.0) if k > mx else (1.0, 1.0, 1.0)
            continue
        if isinstance(cf, str):
            raise NotImplementedError(f"{fname}: named colour gradients ({cf!r}) are not supported (no ColorData "
                                      "gradients in this library); give a function")
        c = apply(cf, fl(t) if scaling else (k if k <= mx else mx + 1))
        rgba = to_rgba(c)
        if rgba is None:
            raise ValueError(f"{fname}: the ColorFunction gave {K_input(c)}, which is not a colour")
        out[k] = tuple(float(v) for v in rgba[:3])
    return out


def _raster_of(counts, table):
    np = _np()
    img = np.zeros(counts.shape + (3,))
    for k, c in table.items():
        img[counts == k] = c
    return L(*[L(*[L(*[fl(v) for v in px]) for px in row]) for row in img])


def _grid_shape(w, h, res):
    if w >= h:
        nx = res
        ny = max(1, int(round(res * h / w)))
    else:
        ny = res
        nx = max(1, int(round(res * w / h)))
    return nx, ny


def _fractal_opts(o, fname, default_res):
    mx = o.pop("MaxIterations", 1000)
    mx = _posint(mx, fname, "MaxIterations")
    res = o.pop("ImageResolution", default_res)
    res = 500 if sym(res, "Automatic") else _posint(res, fname, "ImageResolution", lo=2)
    if res > 2000:
        raise ValueError(f"{fname}: ImageResolution must be at most 2000")
    cf = o.pop("ColorFunction", None)
    scal = o.pop("ColorFunctionScaling", True)
    if not (scal is True or scal is False):
        raise ValueError(f"{fname}: ColorFunctionScaling must be True or False")
    pl = o.pop("PlotLegends", None)
    if pl is not None and not sym(pl, "None"):
        raise NotImplementedError(f"{fname}: PlotLegends is not supported")
    for nm in ("PerformanceGoal", "PlotTheme"):
        v = o.pop(nm, None)
        if v is not None and not sym(v, "Automatic", "None") and nm == "PlotTheme":
            raise NotImplementedError(f"{fname}: PlotTheme -> {K_input(v)} is not supported")
    return mx, res, cf, scal


@fn("MandelbrotSetPlot", area=AREA, status="added",
    notes="MandelbrotSetPlot[{zmin, zmax}] and MandelbrotSetPlot[] (corners -2 - 1.3 I and 0.6 + 1.3 I, the "
          "documented default) -> Graphics[Raster[...]] of escape-time counts: pixel centres c on an "
          "ImageResolution grid (default 500 points along the longer side) iterate z -> z^2 + c from 0 up to "
          "MaxIterations (default 1000) times and escape when |z| > EscapeRadius (default 2, any r >= 2); points "
          "in the main cardioid and period-2 bulb, and points that never escape, are in the set. ColorFunction: "
          "Automatic (black for the set, else a fixed 6-stop gradient at sqrt(k/(n+1)) for escape count k), "
          "None (the set black, the rest white) or a function f[t] of the scaled count "
          "t = k/(n+1) (t = 1 in the set; the raw count k (n + 1 in the set) with ColorFunctionScaling -> False); "
          "named gradients \"name\" raise NotImplementedError (no ColorData gradients here). "
          "Frame -> True by default; other options are Graphics options. PlotLegends and themes raise. The "
          "gradient is this module's own. Check: reference (tests compare pixel membership with "
          "MandelbrotSetMemberQ-style escape counting).")
def _mandelbrotsetplot(*args: Expr | Option) -> Expr:
    fname = "MandelbrotSetPlot"
    np = _np()
    pos, o = split_opts(args, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected {{zmin, zmax}} and options")
    if pos:
        sp_ = pos[0]
        if not (is_list(sp_) and len(sp_.args) == 2):
            if isinstance(sp_, Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: the region must be {{zmin, zmax}}, not {K_input(sp_)}")
        from .wl_numbertheory2_fractals import _complex_value
        zs = [_complex_value(z, fname) for z in sp_.args]
        if any(z is None for z in zs):
            raise Unevaluated()
        z0, z1 = zs
    else:
        z0, z1 = complex(-2, -1.3), complex(0.6, 1.3)
    x0, x1 = sorted((z0.real, z1.real))
    y0, y1 = sorted((z0.imag, z1.imag))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        raise ValueError(f"{fname}: the corners must span a rectangle with positive width and height")
    R = o.pop("EscapeRadius", 2)
    R = _real(R, fname, "EscapeRadius", lo=2)
    mx, res, cf, scal = _fractal_opts(o, fname, 500)
    o.setdefault("Frame", True)
    nx, ny = _grid_shape(x1 - x0, y1 - y0, res)
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    C = xs[None, :] + 1j * ys[:, None]
    counts = np.full(C.shape, mx + 1, dtype=int)
    q = (C.real - 0.25) ** 2 + C.imag ** 2
    inside = (q * (q + (C.real - 0.25)) <= 0.25 * C.imag ** 2) | ((C.real + 1) ** 2 + C.imag ** 2 <= 1 / 16)
    Z = np.zeros_like(C)
    active = ~inside
    R2 = R * R
    for k in range(mx):
        if not active.any():
            break
        Z[active] = Z[active] * Z[active] + C[active]
        esc = active & ((Z.real ** 2 + Z.imag ** 2) > R2)
        counts[esc] = k
        active &= ~esc
    table = _color_table(cf, scal, sorted(set(np.unique(counts).tolist())), mx, fname)
    raster = Expr("Raster", _raster_of(counts, table), L(L(fl(x0), fl(y0)), L(fl(x1), fl(y1))))
    return Expr("Graphics", raster, *_graphics_rules(o, 2, fname))


def _julia_map(pos, fname):
    from .wl_numbertheory2_fractals import _complex_value, _rational_map
    if len(pos) == 1:
        cv = _complex_value(pos[0], fname)
        if cv is None:
            raise Unevaluated()
        return [1, 0, cv], [1]
    if len(pos) == 2:
        if not isinstance(pos[1], Symbol):
            raise TypeError(f"{fname}: the second argument must be a variable")
        if S(pos[0]).free_symbols - {S(pos[1])}:
            raise Unevaluated()
        return _rational_map(pos[0], pos[1], fname)
    raise TypeError(f"{fname}: expected c or f, z plus options")


@fn("JuliaSetPlot", area=AREA, status="added",
    notes="JuliaSetPlot[c] (the map z^2 + c) and JuliaSetPlot[f, z] for a rational function f of degree >= 2. "
          "Method -> \"InverseIteration\" (the default here) draws the JuliaSetPoints points (inverse iteration "
          "from a repelling fixed point, ClosenessTolerance 0.004 or \"ClosenessTolerance\" -> t, \"Bound\" -> b) "
          "as Point primitives in PlotStyle (default black); Method -> \"EscapeTime\" (polynomials only) gives "
          "Graphics[Raster[...]] over the square [-R, R]^2 (or PlotRange {{x0, x1}, {y0, y1}}) with escape "
          "radius R = max(1, (2 + sum_{i<d} |a_i|)/|a_d|) on an ImageResolution grid (default 300) up to "
          "MaxIterations (default 200), coloured as in MandelbrotSetPlot (filled Julia set black). "
          "\"OrbitDetection\" and rational maps with \"EscapeTime\" raise NotImplementedError. Frame -> True "
          "by default; other options are Graphics options. The point set, its sampling and colours differ "
          "from the Wolfram Language's. Check: self-check (f maps a sample of up to 400 drawn points to within 1e-6 "
          "relative of drawn points, since J is invariant; else ArithmeticError) and reference (tests check the "
          "points of z^2 lie on |z| = 1 and those of z^2 - 2 on [-2, 2]).")
def _juliasetplot(*args: Expr | Option) -> Expr:
    fname = "JuliaSetPlot"
    np = _np()
    pos, o = split_opts(args, fname)
    meth = o.pop("Method", "InverseIteration")
    sub = {}
    if is_list(meth) or is_rule(meth):
        items = list(meth.args) if is_list(meth) else [meth]
        name = "InverseIteration"
        for it in items:
            if isinstance(it, str):
                name = it
            elif is_rule(it) and isinstance(it.args[0], str):
                sub[it.args[0]] = it.args[1]
            else:
                raise ValueError(f"{fname}: invalid Method setting {K_input(meth)}")
        meth = name
    if sym(meth, "Automatic"):
        meth = "InverseIteration"
    if meth == "OrbitDetection":
        raise NotImplementedError(f"{fname}: Method -> \"OrbitDetection\" is not supported")
    if meth not in ("InverseIteration", "EscapeTime"):
        raise ValueError(f"{fname}: unknown Method {K_input(meth)}")
    pc, qc = _julia_map(pos, fname)
    polynomial = len(qc) == 1
    o.setdefault("Frame", True)
    if meth == "InverseIteration":
        from .wl_numbertheory2_fractals import _julia_points
        bad = set(sub) - {"ClosenessTolerance", "Bound"}
        if bad:
            raise ValueError(f"{fname}: unknown Method sub-options {sorted(bad)}")
        tol = _real(sub.get("ClosenessTolerance", 0.004), fname, "\"ClosenessTolerance\"", lo=0, lo_open=True)
        if polynomial and "Bound" not in sub:
            lead = abs(pc[0] / qc[0])
            bound = max(1.0, (1 + sum(abs(c / qc[0]) for c in pc[1:])) / lead) + 1.0
        else:
            bound = _real(sub.get("Bound", 6), fname, "\"Bound\"", lo=0, lo_open=True)
        for nm in ("MaxIterations", "ImageResolution", "ColorFunction", "ColorFunctionScaling", "PlotLegends",
                   "PerformanceGoal"):
            v = o.pop(nm, None)
            if v is not None and nm in ("PlotLegends",) and not sym(v, "None"):
                raise NotImplementedError(f"{fname}: PlotLegends is not supported")
        style = o.pop("PlotStyle", None)
        try:
            pts = _julia_points(pc, qc, tol, bound, polynomial)
        except ValueError as exc:
            raise ValueError(f"{fname}: {exc}") from None
        P = np.array([[z.real, z.imag] for z in pts])
        if not np.all(np.isfinite(P)):
            raise ArithmeticError(f"{fname}: inverse iteration produced non-finite points")
        # self-check: the Julia set is invariant, and each computed point is a preimage of an earlier one
        Zs = np.array(pts)
        samp = Zs[:: max(1, len(Zs) // 400)]
        with np.errstate(all="ignore"):
            img = np.polyval(np.array(pc), samp) / np.polyval(np.array(qc), samp)
        dist = np.abs(img[:, None] - Zs[None, :]).min(axis=1)
        if not np.all(dist <= 1e-6 * (1 + np.abs(img))):
            raise ArithmeticError(f"{fname}: self-check failed: an image point is not on the computed Julia set")
        sty = [GrayLevel(0)] if style is None or sym(style, "Automatic") else [_style_of(style, fname, "PlotStyle")]
        prim = L(*sty, Expr("AbsolutePointSize", 1), Expr("Point", L(*[pt(p) for p in P])))
        return Expr("Graphics", prim, *_graphics_rules(o, 2, fname))
    if not polynomial:
        raise NotImplementedError(f"{fname}: Method -> \"EscapeTime\" supports polynomial maps only")
    if sub:
        raise ValueError(f"{fname}: \"EscapeTime\" has no sub-options")
    o.pop("PlotStyle", None)
    mx = o.pop("MaxIterations", 200)
    o["MaxIterations"] = mx
    if "ImageResolution" not in o:
        o["ImageResolution"] = 300
    mx, res, cf, scal = _fractal_opts(o, fname, 300)
    a = np.array(pc, dtype=complex) / qc[0]
    d = len(a) - 1
    R = max(1.0, (2 + sum(abs(c) for c in a[1:])) / abs(a[0]))
    pr = o.get("PlotRange")
    if pr is not None and is_list(pr) and len(pr.args) == 2 and all(is_list(r) and len(r.args) == 2
                                                                    for r in pr.args):
        (x0, x1), (y0, y1) = [sorted(_real(v, fname, "PlotRange") for v in r.args) for r in pr.args]
    elif pr is None or sym(pr, "Automatic", "All"):
        x0, x1, y0, y1 = -R, R, -R, R
    else:
        raise NotImplementedError(f"{fname}: PlotRange {K_input(pr)} is not supported with \"EscapeTime\"")
    nx, ny = _grid_shape(x1 - x0, y1 - y0, res)
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    Z = xs[None, :] + 1j * ys[:, None]
    counts = np.full(Z.shape, mx + 1, dtype=int)
    active = np.abs(Z) <= R
    counts[~active] = 0
    for k in range(mx):
        if not active.any():
            break
        Z[active] = np.polyval(a, Z[active])
        esc = active & (np.abs(Z) > R)
        counts[esc] = k + 1
        active &= ~esc
    del d
    table = _color_table(cf, scal, sorted(set(np.unique(counts).tolist())), mx, fname)
    raster = Expr("Raster", _raster_of(counts, table), L(L(fl(x0), fl(y0)), L(fl(x1), fl(y1))))
    return Expr("Graphics", raster, *_graphics_rules(o, 2, fname))


# ═══════════════════════════════════════ BioSequencePlot ═══════════════════════════════════════

_BIO_COLORS = {"A": (0.30, 0.69, 0.29), "C": (0.22, 0.49, 0.72), "G": (1.0, 0.6, 0.0), "T": (0.89, 0.10, 0.11),
               "U": (0.89, 0.10, 0.11)}


def _bio_strands(x, fname):
    from . import wl_text as TX
    if isinstance(x, str):
        x = ev(Expr("BioSequence", x))
    if not (isinstance(x, Expr) and x.head == "BioSequence"):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: {K_input(x)} is not a BioSequence")
    if not TX._bio_valid(x):
        raise ValueError(f"{fname}: {K_input(x)} is not a valid BioSequence")
    p = TX._bio_parts(x)
    if p is not None:
        t, s, b = p
        return [(t if isinstance(t, str) else "None", s)], (list(b.args) if b is not None else []), \
            isinstance(t, str) and t.startswith("Circular")
    if x.args[0] == "HybridStrand":
        strands = [TX._bio_parts(y) for y in x.args[1].args]
        return [(t if isinstance(t, str) else "None", s) for t, s, _ in strands], list(x.args[2].args), False
    raise NotImplementedError(f"{fname}: sequence collections are not supported")


def _bond_ends(b, nstr, lens, fname):
    """[((strand, i), (strand, j))] 0-based from Bond[{i, j}] or Bond[{{s, i}, {t, j}}]."""
    if not (isinstance(b, Expr) and b.head == "Bond" and b.args and is_list(b.args[0]) and len(b.args[0].args) == 2):
        raise ValueError(f"{fname}: {K_input(b)} is not a bond Bond[{{i, j}}]")
    out = []
    for e in b.args[0].args:
        if K.is_int(e):
            s, i = 1, int(e)
        elif is_list(e) and len(e.args) == 2 and all(K.is_int(v) for v in e.args):
            s, i = int(e.args[0]), int(e.args[1])
        else:
            raise ValueError(f"{fname}: bond end {K_input(e)} must be a position or {{strand, position}}")
        if not (1 <= s <= nstr and 1 <= i <= lens[s - 1]):
            raise ValueError(f"{fname}: bond end {K_input(e)} is outside the sequence")
        out.append((s - 1, i - 1))
    return out


@fn("BioSequencePlot", area=AREA, status="added",
    notes="BioSequencePlot[bseq] for BioSequence[type, \"letters\" (, bonds)] (and plain strings), circular "
          "types and HybridStrand -> Graphics of a schematic: letters as coloured disks (A green, C blue, G "
          "orange, T/U red, others gray) with the letter inside, consecutive letters joined by the backbone; a "
          "linear strand is wrapped in rows of 50 letters (rows alternate direction so the backbone is "
          "continuous), a circular one is laid on a circle, the strands of a HybridStrand are rows stacked "
          "downward with the second strand reversed when it is the only other one; each Bond[{i, j}] (or "
          "Bond[{{s, i}, {t, j}}]) is a dashed line. Sequence collections raise NotImplementedError. The "
          "drawing is this module's own (not the notebook's secondary-structure layout). Check: reference "
          "(tests count disks, backbone segments and bonds).")
def _biosequenceplot(x: Expr, *opts: Option) -> Expr:
    fname = "BioSequencePlot"
    pos, o = split_opts((x,) + opts, fname)
    if len(pos) != 1:
        raise TypeError(f"{fname}: one BioSequence is expected")
    strands, bonds, circular = _bio_strands(pos[0], fname)
    for nm in ("GraphLayout", "VertexCoordinates", "VertexLabels", "EdgeStyle", "VertexStyle", "PlotTheme",
               "GraphHighlight", "VertexSize", "DirectedEdges", "PlotStyle"):
        v = o.pop(nm, None)
        if v is not None and not sym(v, "Automatic", "None") and not (is_list(v) and not v.args) and \
                v is not False:
            raise NotImplementedError(f"{fname}: {nm} -> {K_input(v)} is not supported")
    coords = []
    lens = [len(s) for _, s in strands]
    if circular:
        s = strands[0][1]
        n = len(s)
        rad = max(1.0, n / (2 * math.pi))
        coords.append([(rad * math.cos(math.pi / 2 - 2 * math.pi * k / n),
                        rad * math.sin(math.pi / 2 - 2 * math.pi * k / n)) for k in range(n)])
    else:
        row0 = 0.0
        for si, (_, s) in enumerate(strands):
            cs = []
            flip = len(strands) == 2 and si == 1
            for k in range(len(s)):
                r, c = divmod(k, 50)
                if r % 2:
                    c = 49 - c if len(s) > 50 else c
                if flip:
                    c = len(s) - 1 - k if len(s) <= 50 else c
                cs.append((float(c), row0 - 1.5 * r))
            coords.append(cs)
            row0 -= 1.5 * ((len(s) - 1) // 50 + 1) + 0.5
    backbone, disks, letters = [], [], []
    for si, (_, s) in enumerate(strands):
        cs = coords[si]
        if len(cs) > 1:
            path = cs + ([cs[0]] if circular else [])
            backbone.append(Expr("Line", L(*[pt(p) for p in path])))
        for k, ch in enumerate(s):
            disks.append(L(rgb(_BIO_COLORS.get(ch.upper(), (0.6, 0.6, 0.6))), Expr("Disk", pt(cs[k]), 0.4)))
            letters.append(Expr("Text", ch, pt(cs[k])))
    blines = []
    for b in bonds:
        (s1, i), (s2, j) = _bond_ends(b, len(strands), lens, fname)
        blines.append(Expr("Line", L(pt(coords[s1][i]), pt(coords[s2][j]))))
    content = L(L(GrayLevel(0.5), *backbone), L(GrayLevel(0.3), Expr("Dashing", L(0.01, 0.01)), *blines),
                L(*disks), L(GrayLevel(1), *letters))
    return Expr("Graphics", content, *_graphics_rules(o, 2, fname))


# ═══════════════════════════════════════ FullGraphics ═══════════════════════════════════════

_FG_REMOVED = ("Axes", "AxesOrigin", "Ticks", "Frame", "FrameTicks", "GridLines", "AxesStyle", "FrameStyle",
               "GridLinesStyle", "TicksStyle", "FrameTicksStyle", "AxesLabel", "PlotLabel", "FrameLabel",
               "PlotRange", "PlotRangePadding")
_FG_UNSUPPORTED = ("Prolog", "Epilog", "RotateLabel", "LabelStyle")


def _plot_limits(g, fname):
    """(xlim, ylim, aspect) exactly as the renderer computes them."""
    from . import wl_graphics_render as RD
    try:
        fig = RD.render_figure(g)
    except ImportError:
        raise NotImplementedError(f"{fname}: this needs the optional matplotlib package (pip install "
                                  "mikoshilang[all])") from None
    ax = fig.axes[0]
    xlim, ylim = tuple(map(float, ax.get_xlim())), tuple(map(float, ax.get_ylim()))
    W, H = fig.get_size_inches()
    return xlim, ylim, float(H / W)


def _style_prims(v, fname, what):
    if v is None or (is_list(v) and not v.args):
        return []
    if is_list(v):
        raise NotImplementedError(f"{fname}: per-axis {what} lists are not supported")
    return [_style_of(v, fname, what)]


@fn("FullGraphics", area=AREA, status="added",
    notes="FullGraphics[Graphics[...]] -> Graphics with the axes, ticks, frame, frame ticks and grid lines "
          "written out as explicit Line and Text primitives (after the content, grid lines before it), with "
          "PlotRange -> the rendered range, PlotRangePadding -> None and the rendered AspectRatio, so it draws "
          "the same picture without those options (the range is widened by 12% of the width beside "
          "labelled vertical edges or axes and 7% of the height beside labelled horizontal edges or axes and a "
          "PlotLabel so "
          "the labels stay inside, with the AspectRatio adjusted to keep the content's scale). Ranges and tick "
          "positions are those the W15 renderer uses "
          "(content bounds, PlotRange, PlotRangePadding Scaled[0.02]; FindDivisions-style ticks, about 6 per "
          "axis): ticks are 1.5% of the perpendicular range (frame ticks inward, labels outside; top/right "
          "frame ticks unlabelled by default), axes cross at AxesOrigin (Automatic: 0 when inside the range, "
          "else the low end), PlotLabel and AxesLabel become Text. AxesStyle, FrameStyle, TicksStyle, "
          "GridLinesStyle take one directive. Graphics3D, FrameLabel, per-axis style lists, Prolog, Epilog, "
          "RotateLabel and LabelStyle raise NotImplementedError. Needs matplotlib for the range computation. "
          "Check: self-check (the output must render over exactly the computed range, to 1e-6 relative, else "
          "ArithmeticError) and reference (tests compare tick positions with the renderer's).")
def _fullgraphics(g: Expr) -> Expr:
    fname = "FullGraphics"
    from . import wl_plot_options as PO
    if isinstance(g, Symbol):
        raise Unevaluated()
    if isinstance(g, Expr) and g.head == "Graphics3D":
        raise NotImplementedError(f"{fname}: 3D graphics are not supported")
    if not (isinstance(g, Expr) and g.head == "Graphics" and g.args):
        raise TypeError(f"{fname}: {K_input(g)} is not a Graphics object")
    dim, content, opts = G.graphics_parts(g)
    od = G.options_dict(opts)
    for nm in _FG_UNSUPPORTED + ("FrameLabel",):
        v = od.get(nm)
        if v is not None and not sym(v, "None", "Automatic") and not (is_list(v) and not v.args) and \
                not (nm == "RotateLabel" and v is True):
            raise NotImplementedError(f"{fname}: the option {nm} is not supported")
    xlim, ylim, aspect = _plot_limits(g, fname)
    dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    frame = PO._frame_sides(od.get("Frame", False))
    axes_v = od.get("Axes", False)
    if axes_v is True or axes_v is False:
        axes = (axes_v, axes_v)
    elif is_list(axes_v) and len(axes_v.args) == 2 and all(a is True or a is False for a in axes_v.args):
        axes = tuple(axes_v.args)
    else:
        raise NotImplementedError(f"{fname}: Axes -> {K_input(axes_v)} is not supported")
    front, back = [], []
    tl_x, tl_y = 0.015 * dy, 0.015 * dx
    margin = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}
    # grid lines
    gl = od.get("GridLines")
    if gl is not None and not sym(gl, "None"):
        gx, gy = PO._per_axis(gl, "GridLines")
        gsty = _style_prims(od.get("GridLinesStyle"), fname, "GridLinesStyle") or [GrayLevel(0.85)]
        lines = []
        for spec, lo, hi, vertical in ((gx, xlim[0], xlim[1], True), (gy, ylim[0], ylim[1], False)):
            if spec is None or sym(spec, "None"):
                continue
            if sym(spec, "Automatic"):
                ps = PO.tick_positions(lo, hi, 6)
            elif is_list(spec):
                ps = [_real(s.args[0] if is_list(s) else s, fname, "a grid line position") for s in spec.args]
            else:
                raise NotImplementedError(f"{fname}: GridLines {K_input(gl)} is not supported")
            for p in ps:
                seg = ((p, ylim[0]), (p, ylim[1])) if vertical else ((xlim[0], p), (xlim[1], p))
                lines.append(Expr("Line", L(pt(seg[0]), pt(seg[1]))))
        back.append(L(*gsty, *lines))
    # frame
    if any(frame):
        fsty = _style_prims(od.get("FrameStyle"), fname, "FrameStyle") or [GrayLevel(0)]
        fticks = PO._frame_ticks(od.get("FrameTicks"))
        edges = {"left": ((xlim[0], ylim[0]), (xlim[0], ylim[1])), "right": ((xlim[1], ylim[0]), (xlim[1], ylim[1])),
                 "bottom": ((xlim[0], ylim[0]), (xlim[1], ylim[0])), "top": ((xlim[0], ylim[1]), (xlim[1], ylim[1]))}
        prims = []
        for e, on in zip(("left", "right", "bottom", "top"), frame):
            if not on:
                continue
            a, b = edges[e]
            prims.append(Expr("Line", L(pt(a), pt(b))))
            spec, deflab = fticks[e]
            horiz = e in ("bottom", "top")
            lo, hi = (xlim if horiz else ylim)
            try:
                ticks = PO._ticks_of(spec, lo, hi, "FrameTicks", deflab)
            except NotImplementedError as exc:
                raise NotImplementedError(f"{fname}: {exc}") from None
            if any(lab for _, lab in ticks or []):
                margin[e] = 0.07 * dy if horiz else 0.12 * dx
            for p, lab in ticks or []:
                if horiz:
                    y = ylim[0] if e == "bottom" else ylim[1]
                    s = 1 if e == "bottom" else -1
                    prims.append(Expr("Line", L(pt((p, y)), pt((p, y + s * tl_x)))))
                    if lab:
                        prims.append(Expr("Text", lab, pt((p, y - s * tl_x)), L(0, 1 if e == "bottom" else -1)))
                else:
                    x = xlim[0] if e == "left" else xlim[1]
                    s = 1 if e == "left" else -1
                    prims.append(Expr("Line", L(pt((x, p)), pt((x + s * tl_y, p)))))
                    if lab:
                        prims.append(Expr("Text", lab, pt((x - s * tl_y, p)), L(1 if e == "left" else -1, 0)))
        front.append(L(*fsty, *prims))
    # axes
    if any(axes):
        ao = od.get("AxesOrigin")
        if ao is None or sym(ao, "Automatic"):
            ox = 0.0 if xlim[0] <= 0.0 <= xlim[1] else xlim[0]
            oy = 0.0 if ylim[0] <= 0.0 <= ylim[1] else ylim[0]
        elif is_list(ao) and len(ao.args) == 2:
            ox, oy = (_real(a, fname, "AxesOrigin") for a in ao.args)
        else:
            raise NotImplementedError(f"{fname}: AxesOrigin {K_input(ao)} is not supported")
        asty = _style_prims(od.get("AxesStyle"), fname, "AxesStyle") or [GrayLevel(0.4)]
        tsty = _style_prims(od.get("TicksStyle"), fname, "TicksStyle")
        tx, ty = PO._per_axis(od.get("Ticks", Symbol("Automatic")), "Ticks")
        alab = od.get("AxesLabel")
        if alab is None or sym(alab, "None"):
            al = (None, None)
        elif is_list(alab) and len(alab.args) == 2:
            al = tuple(None if sym(a, "None") else a for a in alab.args)
        else:
            al = (None, alab)
        prims = []
        for k, on in enumerate(axes):
            if not on:
                continue
            try:
                ticks = PO._ticks_of(tx if k == 0 else ty, *(xlim if k == 0 else ylim), "Ticks", True)
            except NotImplementedError as exc:
                raise NotImplementedError(f"{fname}: {exc}") from None
            tp = []
            if any(lab for _, lab in ticks or []):
                if k == 0 and oy - ylim[0] < 0.07 * dy:
                    margin["bottom"] = max(margin["bottom"], 0.07 * dy - (oy - ylim[0]))
                if k == 1 and ox - xlim[0] < 0.12 * dx:
                    margin["left"] = max(margin["left"], 0.12 * dx - (ox - xlim[0]))
            if k == 0:
                prims.append(Expr("Line", L(pt((xlim[0], oy)), pt((xlim[1], oy)))))
                for p, lab in ticks or []:
                    tp.append(Expr("Line", L(pt((p, oy)), pt((p, oy + tl_x)))))
                    if lab and not (axes[1] and abs(p - ox) < 1e-12 * max(1.0, abs(p))):
                        tp.append(Expr("Text", lab, pt((p, oy - 0.5 * tl_x)), L(0, 1)))
                if al[0] is not None:
                    prims.append(Expr("Text", al[0], pt((xlim[1], oy)), L(-1, 0)))
                    margin["right"] = max(margin["right"], 0.12 * dx)
            else:
                prims.append(Expr("Line", L(pt((ox, ylim[0])), pt((ox, ylim[1])))))
                for p, lab in ticks or []:
                    tp.append(Expr("Line", L(pt((ox, p)), pt((ox + tl_y, p)))))
                    if lab and not (axes[0] and abs(p - oy) < 1e-12 * max(1.0, abs(p))):
                        tp.append(Expr("Text", lab, pt((ox - 0.5 * tl_y, p)), L(1, 0)))
                if al[1] is not None:
                    prims.append(Expr("Text", al[1], pt((ox, ylim[1])), L(0, -1)))
                    margin["top"] = max(margin["top"], 0.07 * dy)
            prims.append(L(*tsty, *tp))
        front.append(L(*asty, *prims))
    plab = od.get("PlotLabel")
    if plab is not None and not sym(plab, "None"):
        top = ylim[1] + margin["top"]
        front.append(Expr("Text", plab, pt(((xlim[0] + xlim[1]) / 2, top)), L(0, -1)))
        margin["top"] += 0.07 * dy
    X = (xlim[0] - margin["left"], xlim[1] + margin["right"])
    Y = (ylim[0] - margin["bottom"], ylim[1] + margin["top"])
    asp = aspect * ((Y[1] - Y[0]) / dy) / ((X[1] - X[0]) / dx)
    new_opts = [Expr("Rule", Symbol(k), v) for k, v in od.items() if k not in _FG_REMOVED + ("AspectRatio",)]
    new_opts += [Expr("Rule", Symbol("PlotRange"), L(pt(X), pt(Y))),
                 Expr("Rule", Symbol("PlotRangePadding"), Symbol("None")),
                 Expr("Rule", Symbol("AspectRatio"), fl(asp))]
    out = Expr("Graphics", L(*back, content, *front), *new_opts)
    xl2, yl2, _ = _plot_limits(out, fname)
    if any(abs(a - b) > 1e-6 * max(1.0, abs(a)) for a, b in zip(X + Y, xl2 + yl2)):
        raise ArithmeticError(f"{fname}: self-check failed: the explicit graphics render over a different range")
    return out


SKIPPED = {
    "JuliaSetIterationCount": "its documented \"OrbitDetection\" algorithm (default) is unspecified: the "
                              "documentation's value 4 for .1 - .2 I under z^2 - 1 counts iterations until an "
                              "attracting cycle is detected with an undisclosed tolerance, which escape counting "
                              "cannot reproduce; implementing a different count under the same name would be "
                              "misleading",
    "StatusCentrality": "the documented defining equation c = alpha A^T c + alpha beta is garbled in the "
                        "documentation and its exact results (e.g. 160/(3 E^4), 128/(3 E^4), ...) contain E^-4, "
                        "which no linear status equation of the stated form yields; the definition cannot be "
                        "recovered from the reference",
}
