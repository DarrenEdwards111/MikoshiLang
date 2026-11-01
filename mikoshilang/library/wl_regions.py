"""Wolfram batch W22: Geometry II -- region computation: measures, centroids, moments, membership, distances,
nearest points, bounds, boundaries, Boolean / transformed / product regions, formula regions.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).

The region model
----------------
``region_of(expr, fname)`` turns a region expression (the W19 primitives ``Point``, ``Line``, ``Polygon``,
``Disk``, ``Ball``, ``Cuboid``, polyhedra, ..., and the region heads of this batch: ``ImplicitRegion``,
``ParametricRegion``, ``MeshRegion``, ``BoundaryMeshRegion``, ``RegionUnion``, ``TransformedRegion``, ...) into an
internal object with exact methods ``member``, ``integrate`` (of a polynomial), ``bounds``, ``dist``,
``nearest``, ``sdist`` (signed distance), ``boundary`` and ``farthest``.  Integration of polynomials is exact:
simplices use the Dirichlet formula, balls and spheres closed-form monomial moments, boxes and solids of
revolution exact SymPy integration in adapted coordinates, polygons and polyhedra signed simplex fans.  A method
that a representation does not support raises ``NotImplementedError("<Head>: ...")``; nothing is approximated.
"""
from __future__ import annotations

import itertools
import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, R, has_float
from .wl_graphics_color import K_input
from .wl_graphics_core import split_trailing_options, opt_name
from . import wl_geomprims as G
from .wl_geomprims import sx, pt, pts, out, num, fsign, concrete, _is_coord_list
from .wl_regions_xform import as_transform, affine_parts, homog, tf_expr

AREA = "geometry: regions (W22)"

INF = sp.oo


class NoSupport(NotImplementedError):
    pass


def nosup(f, what):
    raise NotImplementedError(f"{f}: {what} is not supported")


# ───────────────────────────── exact helpers ─────────────────────────────

def S(v):
    return sp.sympify(v)


def sgn(v):
    v = S(v)
    if v in (sp.oo,):
        return 1
    if v in (-sp.oo,):
        return -1
    return fsign(v)


def le(a, b):
    return sgn(S(a) - S(b)) <= 0


def lt(a, b):
    return sgn(S(a) - S(b)) < 0


def eqz(a):
    return sgn(a) == 0


def vsub(p, q):
    return [S(a) - S(b) for a, b in zip(p, q)]


def vadd(p, q):
    return [S(a) + S(b) for a, b in zip(p, q)]


def vmul(c, p):
    return [S(c) * a for a in p]


def dot(p, q):
    return sum((S(a) * S(b) for a, b in zip(p, q)), sp.Integer(0))


def n2(p):
    return dot(p, p)


def norm(p):
    return sp.sqrt(n2(p))


def simp(v):
    v = S(v)
    if v.has(sp.Float) or v.free_symbols:
        return v
    try:
        w = sp.radsimp(sp.simplify(v))
        return w if sp.count_ops(w) <= sp.count_ops(v) + 2 else v
    except Exception:
        return v


def smin(vals):
    best = None
    for v in vals:
        if best is None or lt(v, best):
            best = v
    return best


def smax(vals):
    best = None
    for v in vals:
        if best is None or lt(best, v):
            best = v
    return best


def is_zero_vec(v):
    return all(eqz(c) for c in v)


def affine_rank(P):
    if len(P) <= 1:
        return 0
    M = sp.Matrix([vsub(p, P[0]) for p in P[1:]])
    if any(S(e).has(sp.Float) for e in M):
        import numpy as np
        A = np.array([[float(num(e)) for e in row] for row in M.tolist()], dtype=float)
        return int(np.linalg.matrix_rank(A, tol=1e-10 * max(1.0, float(np.abs(A).max()))))
    return M.rank(simplify=True)


def X(n):
    return [sp.Symbol(f"_rx{i}", real=True) for i in range(n)]


def subst(g, Xs, vals):
    return S(g).xreplace(dict(zip(Xs, vals)))


def poly_terms(g, gens):
    """(coeff, exponent tuple) pairs of a polynomial g in gens (NotImplementedError if g is not polynomial)."""
    g = sp.expand(S(g))
    if not gens:
        return [(g, ())]
    try:
        P = sp.Poly(g, *gens)
    except sp.PolynomialError:
        raise NoSupport("integrand must be a polynomial in the coordinates")
    if any(c.has(*gens) for c in P.coeffs()):
        raise NoSupport("integrand must be a polynomial in the coordinates")
    return list(zip(P.coeffs(), P.monoms()))


# ───────────────────────────── integration kernels ─────────────────────────────

def simplex_integral(V, g, Xs, signed=False):
    """∫ over the simplex V (k+1 points in R^n) of the polynomial g(Xs); signed uses det instead of |det| (k = n)."""
    k = len(V) - 1
    n = len(V[0])
    v0 = [S(c) for c in V[0]]
    M = sp.Matrix([[S(V[j + 1][i]) - v0[i] for j in range(k)] for i in range(n)]) if k else sp.zeros(n, 0)
    if k == 0:
        return subst(g, Xs, v0)
    if k == n:
        d = M.det()
        jac = d if signed else sp.Abs(d)
    else:
        jac = sp.sqrt((M.T * M).det())
    T = sp.symbols(f"_rt0:{k}", real=True)
    xs = [v0[i] + sum(M[i, j] * T[j] for j in range(k)) for i in range(n)]
    h = subst(g, Xs, xs)
    tot = sp.Integer(0)
    for c, e in poly_terms(h, list(T)):
        num_ = sp.Integer(1)
        for a in e:
            num_ *= sp.factorial(a)
        tot += c * num_ / sp.factorial(k + sum(e))
    return jac * tot


def ball_monomial(e):
    if any(a % 2 for a in e):
        return sp.Integer(0)
    n = len(e)
    r = sp.Integer(1)
    for a in e:
        r *= sp.gamma(sp.Rational(a + 1, 2))
    return r / sp.gamma(sp.Rational(sum(e) + n, 2) + 1)


def sphere_monomial(e):
    if any(a % 2 for a in e):
        return sp.Integer(0)
    n = len(e)
    r = sp.Integer(2)
    for a in e:
        r *= sp.gamma(sp.Rational(a + 1, 2))
    return r / sp.gamma(sp.Rational(sum(e) + n, 2))


def ellipsoid_integral(c, Lm, g, Xs, surface=False):
    n = len(c)
    U = sp.symbols(f"_ru0:{n}", real=True)
    xs = [S(c[i]) + sum(Lm[i, j] * U[j] for j in range(n)) for i in range(n)]
    h = subst(g, Xs, xs)
    tot = sp.Integer(0)
    for co, e in poly_terms(h, list(U)):
        tot += co * (sphere_monomial(e) if surface else ball_monomial(e))
    if surface:
        r = Lm[0, 0]
        return r ** (n - 1) * tot
    return sp.Abs(Lm.det()) * tot


def sym_integrate(h, ranges):
    """Exact iterated integral; NotImplementedError if SymPy leaves an Integral."""
    v = sp.integrate(sp.expand(h), *ranges)
    if v.has(sp.Integral):
        raise NoSupport("this integral")
    return v


def frame(u):
    """Exact orthonormal e1, e2 perpendicular to the 3-D unit vector u."""
    a = [sp.Integer(1), 0, 0] if not eqz(u[1]) or not eqz(u[2]) else [0, sp.Integer(1), 0]
    if eqz(u[0]) and eqz(u[1]):
        a = [sp.Integer(1), 0, 0]
    elif eqz(u[0]) and eqz(u[2]):
        a = [0, 0, sp.Integer(1)]
    elif eqz(u[1]) and eqz(u[2]):
        a = [0, sp.Integer(1), 0]
    w = vsub(a, vmul(dot(a, u), u))
    e1 = vmul(1 / norm(w), w)
    e2 = [u[1] * e1[2] - u[2] * e1[1], u[2] * e1[0] - u[0] * e1[2], u[0] * e1[1] - u[1] * e1[0]]
    return [simp(c) for c in e1], [simp(c) for c in e2]


def revolve_integral(g, Xs, o, u, pieces):
    """∫ g over a solid of revolution about the axis o + s u (|u| = 1): pieces = [(s0, s1, rho_max(s))]."""
    e1, e2 = frame(u)
    s, rho, th = sp.symbols("_rs _rrho _rth", real=True)
    xs = [S(o[i]) + s * u[i] + rho * (sp.cos(th) * e1[i] + sp.sin(th) * e2[i]) for i in range(3)]
    h = sp.expand(subst(g, Xs, xs) * rho)
    tot = sp.Integer(0)
    inner = sym_integrate(h, [(th, 0, 2 * sp.pi)])
    for s0, s1, rmax in pieces:
        a = sym_integrate(inner, [(rho, 0, rmax)])
        tot += sym_integrate(a, [(s, s0, s1)])
    return sp.simplify(tot)


def sweep2d_integral(g, Xs, o, u, pieces):
    """∫ g over a 2-D region {o + s u + t n : s0 <= s <= s1, |t| <= w(s)} (|u| = 1, n = u rotated by 90°)."""
    nv = [-u[1], u[0]]
    s, t = sp.symbols("_rs _rt", real=True)
    xs = [S(o[i]) + s * u[i] + t * nv[i] for i in range(2)]
    h = sp.expand(subst(g, Xs, xs))
    tot = sp.Integer(0)
    for s0, s1, w in pieces:
        a = sym_integrate(h, [(t, -w, w)])
        tot += sym_integrate(a, [(s, s0, s1)])
    return sp.simplify(tot)


# ───────────────────────────── distance kernels ─────────────────────────────

def proj_affine(p, V):
    """Orthogonal projection of p onto the affine hull of V: (barycentric weights, point)."""
    v0 = V[0]
    k = len(V) - 1
    if k == 0:
        return [sp.Integer(1)], list(v0)
    A = sp.Matrix([vsub(v, v0) for v in V[1:]]).T          # n x k
    Gm = A.T * A
    rhs = A.T * sp.Matrix(vsub(p, v0))
    try:
        lam = Gm.LUsolve(rhs)
    except Exception:
        raise NoSupport("degenerate simplices")
    lam = [simp(c) for c in lam]
    w = [1 - sum(lam)] + lam
    q = vadd(v0, [sum(A[i, j] * lam[j] for j in range(k)) for i in range(len(p))])
    return w, q


def simplex_nearest(p, V):
    """(squared distance, nearest point) from p to the simplex V (exact face enumeration)."""
    best = None
    k = len(V)
    for r in range(1, k + 1):
        for sub in itertools.combinations(range(k), r):
            W = [V[i] for i in sub]
            w, q = proj_affine(p, W)
            if all(sgn(c) >= 0 for c in w):
                d2 = simp(n2(vsub(p, q)))
                if best is None or lt(d2, best[0]):
                    best = (d2, q)
    return best


def seg_nearest(p, a, b):
    d = vsub(b, a)
    dd = n2(d)
    if eqz(dd):
        return simp(n2(vsub(p, a))), list(a)
    t = dot(vsub(p, a), d) / dd
    if sgn(t) <= 0:
        q = list(a)
    elif sgn(t - 1) >= 0:
        q = list(b)
    else:
        q = vadd(a, vmul(t, d))
    return simp(n2(vsub(p, q))), [simp(c) for c in q]


def best_of(cands):
    best = None
    for d2, q in cands:
        if best is None or lt(d2, best[0]):
            best = (d2, q)
    return best


def sqrt_out(d2):
    return simp(sp.sqrt(S(d2)))


# ───────────────────────────── 2-D polygon kernels ─────────────────────────────

def orient2(a, b, c):
    return (S(b[0]) - S(a[0])) * (S(c[1]) - S(a[1])) - (S(b[1]) - S(a[1])) * (S(c[0]) - S(a[0]))


def on_segment(p, a, b):
    if sgn(orient2(a, b, p)) != 0:
        return False
    return sgn(dot(vsub(p, a), vsub(p, b))) <= 0


def ring_contains(R_, p):
    """1 inside, 0 on the boundary, -1 outside (exact crossing number)."""
    n = len(R_)
    for i in range(n):
        if on_segment(p, R_[i], R_[(i + 1) % n]):
            return 0
    c = False
    for i in range(n):
        a, b = R_[i], R_[(i + 1) % n]
        ya, yb = S(a[1]) - S(p[1]), S(b[1]) - S(p[1])
        if (sgn(ya) > 0) != (sgn(yb) > 0):
            o = orient2(a, b, p)
            if (sgn(o) > 0) == (sgn(yb - ya) > 0):
                c = not c
    return 1 if c else -1


def signed_area(R_):
    n = len(R_)
    return sum((S(R_[i][0]) * S(R_[(i + 1) % n][1]) - S(R_[(i + 1) % n][0]) * S(R_[i][1]) for i in range(n)),
               sp.Integer(0)) / 2


def ring_fan_integral(R_, g, Xs):
    """∫ over the interior of the simple ring R_ of g (orientation-independent)."""
    tot = sp.Integer(0)
    for i in range(1, len(R_) - 1):
        tot += simplex_integral([R_[0], R_[i], R_[i + 1]], g, Xs, signed=True)
    s = sgn(signed_area(R_))
    if s == 0:
        return sp.Integer(0)
    return tot if s > 0 else -tot


def rings_cross(R1, R2):
    for i in range(len(R1)):
        a, b = R1[i], R1[(i + 1) % len(R1)]
        for j in range(len(R2)):
            c, d = R2[j], R2[(j + 1) % len(R2)]
            o1, o2 = sgn(orient2(a, b, c)), sgn(orient2(a, b, d))
            o3, o4 = sgn(orient2(c, d, a)), sgn(orient2(c, d, b))
            if o1 * o2 < 0 and o3 * o4 < 0:
                return True
    return False


# ───────────────────────────── region classes ─────────────────────────────

class Region:
    kind = "region"
    bounded = True
    convex = None
    dim = None
    edim = None
    expr = None
    f = "Region"

    def need(self, what):
        raise NotImplementedError(f"{self.f}: {what} of {self.kind} regions is not supported")

    def member(self, p):
        self.need("membership")

    def integrate(self, g, Xs):
        self.need("integration")

    def bounds(self):
        self.need("bounds")

    def nearest(self, p):
        """(squared distance, nearest point)."""
        self.need("the nearest point")

    def dist(self, p):
        return sqrt_out(self.nearest(p)[0])

    def sdist(self, p):
        self.need("the signed distance")

    def boundary(self):
        self.need("the boundary")

    def farthest(self, p):
        V = self.verts()
        if V is None:
            self.need("the farthest distance")
        return sqrt_out(smax([n2(vsub(p, v)) for v in V]))

    def verts(self):
        """Finite point set whose convex hull equals the convex hull of the region, or None."""
        return None

    def measure(self):
        return simp(self.integrate(sp.Integer(1), X(self.edim)))


def _nz_count(P):
    seen = []
    for p in P:
        if not any(is_zero_vec(vsub(p, q)) for q in seen):
            seen.append(p)
    return seen


class PointsR(Region):
    kind = "Point"

    def __init__(self, P, expr=None):
        self.P = [[S(c) for c in p] for p in P]
        self.edim = len(self.P[0])
        self.dim = 0
        self.convex = len(_nz_count(self.P)) == 1
        self.expr = expr

    def member(self, p):
        return any(is_zero_vec(vsub(p, q)) for q in self.P)

    def integrate(self, g, Xs):
        return sum((subst(g, Xs, q) for q in self.P), sp.Integer(0))

    def bounds(self):
        return [(smin([q[i] for q in self.P]), smax([q[i] for q in self.P])) for i in range(self.edim)]

    def nearest(self, p):
        return best_of([(simp(n2(vsub(p, q))), q) for q in self.P])

    def boundary(self):
        return self.expr if self.expr is not None else Expr("Point", out([q for q in self.P]))

    def verts(self):
        return self.P


class SimplexR(Region):
    """A union of k-simplices (Line segments, triangles, tetrahedra, Simplex, mesh cells)."""
    kind = "simplex"

    def __init__(self, simplices, expr=None, boundary_fn=None, f="Region"):
        self.S = [[[S(c) for c in p] for p in s] for s in simplices]
        self.f = f
        self.edim = len(self.S[0][0])
        ks = {len(s) - 1 for s in self.S}
        if len(ks) != 1:
            raise NotImplementedError(f"{f}: cells of mixed dimension are not supported here")
        self.dim = ks.pop()
        for s in self.S:
            if affine_rank(s) < self.dim:
                raise NotImplementedError(f"{f}: degenerate cells (zero {self.dim}-dimensional measure) are not "
                                          f"supported; use CanonicalizeRegion")
        if self.dim > self.edim:
            raise ValueError(f"{f}: a {self.dim}-simplex needs at least {self.dim} coordinates")
        self.convex = True if len(self.S) == 1 else None
        self.expr = expr
        self.boundary_fn = boundary_fn

    def member(self, p):
        for s in self.S:
            w, q = proj_affine(p, s)
            if is_zero_vec(vsub(p, q)) and all(sgn(c) >= 0 for c in w):
                return True
        return False

    def integrate(self, g, Xs):
        return sum((simplex_integral(s, g, Xs) for s in self.S), sp.Integer(0))

    def bounds(self):
        V = [p for s in self.S for p in s]
        return [(smin([q[i] for q in V]), smax([q[i] for q in V])) for i in range(self.edim)]

    def nearest(self, p):
        return best_of([simplex_nearest(p, s) for s in self.S])

    def sdist(self, p):
        if self.dim != self.edim or len(self.S) != 1:
            self.need("the signed distance")
        s = self.S[0]
        facets = [[s[j] for j in range(len(s)) if j != i] for i in range(len(s))]
        d = sqrt_out(best_of([simplex_nearest(p, fc) for fc in facets])[0])
        return -d if self.member(p) else d

    def boundary(self):
        if self.boundary_fn is not None:
            return self.boundary_fn()
        if self.dim < self.edim:
            return self.expr
        self.need("the boundary")

    def verts(self):
        return [p for s in self.S for p in s]


class Polygon2R(Region):
    kind = "Polygon"

    def __init__(self, comps, expr=None, f="Region"):
        self.f = f
        self.edim, self.dim = 2, 2
        cs = []
        for o, hs in comps:
            o = G._dedup_ring([[S(c) for c in p] for p in o])
            hs = [G._dedup_ring([[S(c) for c in p] for p in h]) for h in hs]
            for r in [o] + hs:
                if len(r) < 3 or eqz(signed_area(r)):
                    raise NotImplementedError(f"{f}: degenerate polygons (zero area) are not supported; use "
                                              f"CanonicalizeRegion")
                if not G._simple_ring(r):
                    raise NotImplementedError(f"{f}: self-intersecting polygons are not supported")
            cs.append((o, hs))
        rings = [r for o, hs in cs for r in [o] + hs]
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                if rings_cross(rings[i], rings[j]):
                    raise NotImplementedError(f"{f}: polygons whose components or holes cross are not supported")
        for o, hs in cs:
            for h in hs:
                if any(ring_contains(o, q) < 0 for q in h):
                    raise NotImplementedError(f"{f}: holes must lie inside their outer boundary")
        for i in range(len(cs)):
            for j in range(len(cs)):
                if i != j and any(ring_contains(cs[j][0], q) > 0 for q in cs[i][0]) and \
                        not any(any(ring_contains(h, q) >= 0 for q in cs[i][0]) for h in cs[j][1]):
                    raise NotImplementedError(f"{f}: overlapping polygon components are not supported")
        self.C = cs
        self.convex = len(cs) == 1 and not cs[0][1] and G._convex_ring(cs[0][0])
        self.expr = expr

    def member(self, p):
        for o, hs in self.C:
            if ring_contains(o, p) >= 0 and not any(ring_contains(h, p) > 0 for h in hs):
                return True
        return False

    def integrate(self, g, Xs):
        tot = sp.Integer(0)
        for o, hs in self.C:
            tot += ring_fan_integral(o, g, Xs) - sum((ring_fan_integral(h, g, Xs) for h in hs), sp.Integer(0))
        return tot

    def edges(self):
        for o, hs in self.C:
            for r in [o] + hs:
                for i in range(len(r)):
                    yield r[i], r[(i + 1) % len(r)]

    def bounds(self):
        V = self.verts()
        return [(smin([q[i] for q in V]), smax([q[i] for q in V])) for i in range(2)]

    def nearest(self, p):
        if self.member(p):
            return sp.Integer(0), list(p)
        return best_of([seg_nearest(p, a, b) for a, b in self.edges()])

    def sdist(self, p):
        d = sqrt_out(best_of([seg_nearest(p, a, b) for a, b in self.edges()])[0])
        return -d if self.member(p) else d

    def boundary(self):
        loops = [r + [r[0]] for o, hs in self.C for r in [o] + hs]
        return Expr("Line", out(loops[0] if len(loops) == 1 else loops))

    def verts(self):
        return [p for o, hs in self.C for p in o]

    def triangles(self):
        tris = []
        for o, hs in self.C:
            if hs:
                return None
            idx = G._triangulate(o, self.f)
            R_ = G._orient_ring(o, True)
            tris += [[R_[i] for i in t] for t in idx]
        return tris


def _plane_basis(P, f):
    """Normal of a planar 3-D ring (exact) and the coordinate index to drop for projection."""
    nrm = [sp.Integer(0)] * 3
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        nrm = vadd(nrm, [S(a[1]) * S(b[2]) - S(a[2]) * S(b[1]), S(a[2]) * S(b[0]) - S(a[0]) * S(b[2]),
                         S(a[0]) * S(b[1]) - S(a[1]) * S(b[0])])
    if is_zero_vec(nrm):
        raise NotImplementedError(f"{f}: degenerate polygons (zero area) are not supported")
    for q in P:
        if not eqz(dot(nrm, vsub(q, P[0]))):
            raise ValueError(f"{f}: the polygon vertices are not coplanar")
    k = max(range(3), key=lambda i: abs(num(nrm[i])))
    return nrm, k


class Polygon3R(Region):
    """A planar polygon (possibly with holes) embedded in 3-D."""
    kind = "Polygon"

    def __init__(self, comps, expr=None, f="Region"):
        self.f = f
        self.edim, self.dim = 3, 2
        allp = [[S(c) for c in p] for o, hs in comps for r in [o] + hs for p in r]
        self.nrm, self.k = _plane_basis([[S(c) for c in p] for p in comps[0][0]], f)
        for q in allp:
            if not eqz(dot(self.nrm, vsub(q, allp[0]))):
                raise ValueError(f"{f}: the polygon vertices are not coplanar")
        self.p0 = allp[0]
        keep = [i for i in range(3) if i != self.k]
        self.keep = keep
        self.proj = Polygon2R([([[p[i] for i in keep] for p in o], [[[p[i] for i in keep] for p in h] for h in hs])
                               for o, hs in comps], f=f)
        self.C = [([[S(c) for c in p] for p in o], [[[S(c) for c in p] for p in h] for h in hs]) for o, hs in comps]
        self.convex = self.proj.convex
        self.expr = expr

    def lift(self, q2):
        """3-D point of the plane with projected coordinates q2."""
        x = [None] * 3
        for i, c in zip(self.keep, q2):
            x[i] = S(c)
        s = sum((self.nrm[i] * (x[i] - self.p0[i]) for i in self.keep), sp.Integer(0))
        x[self.k] = simp(self.p0[self.k] - s / self.nrm[self.k])
        return x

    def member(self, p):
        if not eqz(dot(self.nrm, vsub(p, self.p0))):
            return False
        return self.proj.member([p[i] for i in self.keep])

    def integrate(self, g, Xs):
        tot = sp.Integer(0)
        for o, hs in self.proj.C:
            for r, sgn_ in [(o, 1)] + [(h, -1) for h in hs]:
                R_ = G._orient_ring(r, True)
                for t in G._triangulate(R_, self.f):
                    tot += sgn_ * simplex_integral([self.lift(R_[i]) for i in t], g, Xs)
        return tot

    def edges(self):
        for o, hs in self.C:
            for r in [o] + hs:
                for i in range(len(r)):
                    yield r[i], r[(i + 1) % len(r)]

    def bounds(self):
        V = self.verts()
        return [(smin([q[i] for q in V]), smax([q[i] for q in V])) for i in range(3)]

    def nearest(self, p):
        t = dot(self.nrm, vsub(p, self.p0)) / n2(self.nrm)
        q = vsub(p, vmul(t, self.nrm))
        q = [simp(c) for c in q]
        if self.proj.member([q[i] for i in self.keep]):
            return simp(n2(vsub(p, q))), q
        return best_of([seg_nearest(p, a, b) for a, b in self.edges()])

    def boundary(self):
        return self.expr

    def verts(self):
        return [p for o, hs in self.C for p in o]


def face_triangles(V, fc, f):
    P = [V[i] for i in fc]
    if len(P) == 3:
        return [P]
    nrm, k = _plane_basis(P, f)
    keep = [i for i in range(3) if i != k]
    R2 = [[p[i] for i in keep] for p in P]
    sa = signed_area(R2)
    order = list(range(len(P))) if sgn(sa) > 0 else list(range(len(P)))[::-1]
    R2o = [R2[i] for i in order]
    idx = G._triangulate(R2o, f)
    return [[P[order[i]] for i in t] for t in idx]


def tri_orient_sign(T, nrm):
    a, b, c = T
    u, v = vsub(b, a), vsub(c, a)
    cr = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    return sgn(dot(cr, nrm))


class PolyhedronR(Region):
    kind = "Polyhedron"

    def __init__(self, V, F, expr=None, f="Region", voids=None):
        """V vertices, F faces (0-based); faces of each closed component are oriented outward by _orient.
        voids: list of (V, F) components to subtract."""
        self.f = f
        self.edim, self.dim = 3, 3
        self.V = [[S(c) for c in p] for p in V]
        Fo = G._orient(self.V, F, f)
        self.F = Fo
        self.tris = []
        for fc in Fo:
            ts = face_triangles(self.V, fc, f)
            a, b, c = [self.V[i] for i in fc[:3]]
            # orientation reference: Newell normal of the oriented face
            nrm, _ = _plane_basis([self.V[i] for i in fc], f)
            for T in ts:
                if tri_orient_sign(T, nrm) < 0:
                    T = [T[0], T[2], T[1]]
                self.tris.append(T)
        self.voids = []
        for Vv, Fv in voids or []:
            self.voids.append(PolyhedronR(Vv, Fv, f=f))
        self.convex = (not self.voids) and G._convex_polyhedron(self.V, Fo, f)
        self.expr = expr

    def integrate(self, g, Xs):
        o = self.V[0]
        tot = sp.Integer(0)
        for T in self.tris:
            tot += simplex_integral([o] + T, g, Xs, signed=True)
        for v in self.voids:
            tot -= v.integrate(g, Xs)
        return tot

    def on_boundary(self, p):
        for T in self.tris:
            w, q = proj_affine(p, T)
            if is_zero_vec(vsub(p, q)) and all(sgn(c) >= 0 for c in w):
                return True
        return False

    def _inside_closed(self, p):
        if self.on_boundary(p):
            return True
        if self.convex:
            for T in self.tris:
                a, b, c = T
                u, v = vsub(b, a), vsub(c, a)
                cr = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
                if sgn(dot(cr, vsub(p, a))) > 0:
                    return False
            return True
        import numpy as np
        P = np.array([float(num(c)) for c in p])
        tot = 0.0
        for T in self.tris:
            A = np.array([[float(num(c)) for c in q] for q in T]) - P
            la, lb, lc = (np.linalg.norm(r) for r in A)
            numr = np.dot(A[0], np.cross(A[1], A[2]))
            den = la * lb * lc + np.dot(A[0], A[1]) * lc + np.dot(A[0], A[2]) * lb + np.dot(A[1], A[2]) * la
            tot += 2 * math.atan2(numr, den)
        w = tot / (4 * math.pi)
        if abs(w - round(w)) > 1e-6:
            raise ArithmeticError(f"{self.f}: point classification failed verification")
        return round(w) != 0

    def member(self, p):
        if not self._inside_closed(p):
            return False
        for v in self.voids:
            if v._inside_closed(p) and not v.on_boundary(p):
                return False
        return True

    def bounds(self):
        return [(smin([q[i] for q in self.V]), smax([q[i] for q in self.V])) for i in range(3)]

    def _surf(self, p):
        allt = list(self.tris) + [T for v in self.voids for T in v.tris]
        return best_of([simplex_nearest(p, T) for T in allt])

    def nearest(self, p):
        if self.member(p):
            return sp.Integer(0), list(p)
        return self._surf(p)

    def sdist(self, p):
        d = sqrt_out(self._surf(p)[0])
        return -d if self.member(p) else d

    def boundary(self):
        faces = [[self.V[i] for i in fc] for fc in self.F]
        for v in self.voids:
            faces += [[v.V[i] for i in fc] for fc in v.F]
        return Expr("Polygon", out(faces))

    def verts(self):
        return self.V


class EllipsoidR(Region):
    """{c + L u : |u| <= 1}: Disk, Ball, Ellipsoid (L lower triangular / diagonal, nonsingular)."""
    kind = "Ball"

    def __init__(self, c, Lm, expr=None, f="Region", round_r=None):
        self.f = f
        self.c = [S(v) for v in c]
        self.Lm = Lm
        self.edim = self.dim = len(c)
        self.r = round_r
        self.convex = True
        self.expr = expr
        self.Q = (Lm * Lm.T).inv()

    def member(self, p):
        d = sp.Matrix(vsub(p, self.c))
        return sgn((d.T * self.Q * d)[0] - 1) <= 0

    def integrate(self, g, Xs):
        return ellipsoid_integral(self.c, self.Lm, g, Xs)

    def bounds(self):
        Sg = self.Lm * self.Lm.T
        return [(simp(self.c[i] - sp.sqrt(Sg[i, i])), simp(self.c[i] + sp.sqrt(Sg[i, i]))) for i in range(self.edim)]

    def nearest(self, p):
        if self.member(p):
            return sp.Integer(0), list(p)
        if self.r is None:
            self.need("the distance (ellipses / ellipsoids)")
        d = vsub(p, self.c)
        nd = norm(d)
        q = [simp(c) for c in vadd(self.c, vmul(self.r / nd, d))]
        return simp((nd - self.r) ** 2), q

    def sdist(self, p):
        if self.r is None:
            self.need("the signed distance (ellipses / ellipsoids)")
        return simp(norm(vsub(p, self.c)) - self.r)

    def farthest(self, p):
        if self.r is None:
            self.need("the farthest distance (ellipses / ellipsoids)")
        return simp(norm(vsub(p, self.c)) + self.r)

    def boundary(self):
        n = self.edim
        if self.r is not None:
            return Expr("Circle" if n == 2 else "Sphere", out(self.c), out(self.r))
        if self.Lm.is_diagonal() and n == 2:
            return Expr("Circle", out(self.c), out([self.Lm[0, 0], self.Lm[1, 1]]))
        self.need("the boundary (ellipsoids)")


class SphereR(Region):
    kind = "Sphere"

    def __init__(self, c, r, expr=None, f="Region"):
        self.f = f
        self.c = [S(v) for v in c]
        self.r = S(r)
        self.edim = len(c)
        self.dim = self.edim - 1
        self.convex = self.edim == 1
        self.expr = expr

    def member(self, p):
        return eqz(n2(vsub(p, self.c)) - self.r ** 2)

    def integrate(self, g, Xs):
        return ellipsoid_integral(self.c, self.r * sp.eye(self.edim), g, Xs, surface=True)

    def bounds(self):
        return [(simp(c - self.r), simp(c + self.r)) for c in self.c]

    def nearest(self, p):
        d = vsub(p, self.c)
        if is_zero_vec(d):
            q = list(self.c)
            q[0] = q[0] + self.r
            return simp(self.r ** 2), q
        nd = norm(d)
        q = [simp(c) for c in vadd(self.c, vmul(self.r / nd, d))]
        return simp((nd - self.r) ** 2), q

    def farthest(self, p):
        return simp(norm(vsub(p, self.c)) + self.r)

    def boundary(self):
        return self.expr if self.expr is not None else Expr("Sphere", out(self.c), out(self.r))


class EllipseCurveR(Region):
    """Circle[c, {a, b}] (axis-aligned ellipse curve): measure, centroid, membership, bounds."""
    kind = "ellipse"

    def __init__(self, c, a, b, expr=None, f="Region"):
        self.f = f
        self.c, self.a, self.b = [S(v) for v in c], S(a), S(b)
        self.edim, self.dim = 2, 1
        self.convex = False
        self.expr = expr

    def member(self, p):
        return eqz(((p[0] - self.c[0]) / self.a) ** 2 + ((p[1] - self.c[1]) / self.b) ** 2 - 1)

    def measure(self):
        a, b = (self.a, self.b) if le(self.b, self.a) else (self.b, self.a)
        return simp(4 * a * sp.elliptic_e(1 - b ** 2 / a ** 2))

    def integrate(self, g, Xs):
        terms = poly_terms(g, Xs[:2])
        if any(sum(e) > 1 for _, e in terms):
            self.need("integration of non-linear functions (ellipse curves)")
        lin = subst(g, Xs, self.c)
        return lin * self.measure()

    def bounds(self):
        return [(self.c[0] - self.a, self.c[0] + self.a), (self.c[1] - self.b, self.c[1] + self.b)]

    def boundary(self):
        return self.expr


def _angle_from(t1, v):
    """Angle in [0, 2 Pi) from the direction t1 to the vector v (exact)."""
    u = [sp.cos(t1), sp.sin(t1)]
    cr = u[0] * v[1] - u[1] * v[0]
    dt = dot(u, v)
    a = simp(sp.atan2(simp(cr), simp(dt)))
    if sgn(a) < 0:
        a = a + 2 * sp.pi
    return a


class SectorR(Region):
    """Circular sector / arc / annular sector / disk segment in 2-D: Disk[c, r, {t1, t2}], Circle[c, r, {t1, t2}],
    Annulus[c, {r1, r2}, {t1, t2}], DiskSegment[c, r, {t1, t2}]."""
    kind = "sector"

    def __init__(self, c, r1, r2, t1, t2, curve=False, segment=False, expr=None, f="Region"):
        self.f = f
        self.c = [S(v) for v in c]
        self.r1, self.r2 = S(r1), S(r2)
        self.t1, self.t2 = S(t1), S(t2)
        if lt(self.t2, self.t1):
            raise ValueError(f"{f}: the angle range must have t1 <= t2")
        self.w = simp(self.t2 - self.t1)
        self.full = sgn(self.w - 2 * sp.pi) >= 0
        if self.full:
            self.w = 2 * sp.pi
        self.curve, self.segment = curve, segment
        self.edim = 2
        self.dim = 1 if curve else 2
        self.convex = None if not curve else False
        if not curve and eqz(self.r1) and not segment:
            self.convex = sgn(self.w - sp.pi) <= 0
        if segment:
            self.convex = True
        self.expr = expr

    def _in_angle(self, p):
        v = vsub(p, self.c)
        if is_zero_vec(v):
            return True
        if self.full:
            return True
        return sgn(_angle_from(self.t1, v) - self.w) <= 0

    def _chord_side(self, p):
        A = vadd(self.c, [self.r2 * sp.cos(self.t1), self.r2 * sp.sin(self.t1)])
        B = vadd(self.c, [self.r2 * sp.cos(self.t2), self.r2 * sp.sin(self.t2)])
        return sgn(orient2(A, B, p)) <= 0

    def member(self, p):
        d2 = n2(vsub(p, self.c))
        if self.curve:
            return eqz(d2 - self.r2 ** 2) and self._in_angle(p)
        if sgn(d2 - self.r2 ** 2) > 0 or sgn(d2 - self.r1 ** 2) < 0:
            return False
        if self.segment:
            return self.full or self._chord_side(p)
        return self._in_angle(p)

    def integrate(self, g, Xs):
        rho, th = sp.symbols("_rrho _rth", real=True)
        if self.curve:
            h = subst(g, Xs, [self.c[0] + self.r2 * sp.cos(th), self.c[1] + self.r2 * sp.sin(th)]) * self.r2
            return simp(sym_integrate(h, [(th, self.t1, self.t1 + self.w)]))
        h = subst(g, Xs, [self.c[0] + rho * sp.cos(th), self.c[1] + rho * sp.sin(th)]) * rho
        v = sym_integrate(h, [(rho, self.r1, self.r2), (th, self.t1, self.t1 + self.w)])
        if self.segment and not self.full:
            A = vadd(self.c, [self.r2 * sp.cos(self.t1), self.r2 * sp.sin(self.t1)])
            B = vadd(self.c, [self.r2 * sp.cos(self.t2), self.r2 * sp.sin(self.t2)])
            v -= simplex_integral([self.c, A, B], g, Xs, signed=True)
        return simp(v)

    def _arc_points(self, r):
        pts_ = [vadd(self.c, [r * sp.cos(self.t1), r * sp.sin(self.t1)]),
                vadd(self.c, [r * sp.cos(self.t1 + self.w), r * sp.sin(self.t1 + self.w)])]
        for k in range(8):
            ang = k * sp.pi / 2
            q = vadd(self.c, [r * sp.cos(ang), r * sp.sin(ang)])
            if self.full or sgn(_angle_from(self.t1, vsub(q, self.c)) - self.w) <= 0:
                pts_.append(q)
        return pts_

    def bounds(self):
        P = self._arc_points(self.r2)
        if not self.curve:
            P += self._arc_points(self.r1) if not eqz(self.r1) else ([self.c] if not self.segment else [])
        return [(simp(smin([q[i] for q in P])), simp(smax([q[i] for q in P]))) for i in range(2)]

    def nearest(self, p):
        if not self.curve and self.member(p):
            return sp.Integer(0), list(p)
        if self.segment:
            self.need("the distance (disk segments)")
        v = vsub(p, self.c)
        cands = []
        rs = [self.r2] if self.curve else ([self.r1, self.r2] if not eqz(self.r1) else [self.r2])
        if not is_zero_vec(v) and self._in_angle(p):
            nv = norm(v)
            for r in rs:
                cands.append((simp((nv - r) ** 2), [simp(c) for c in vadd(self.c, vmul(r / nv, v))]))
        ends = [vadd(self.c, [r * sp.cos(t), r * sp.sin(t)]) for r in rs for t in (self.t1, self.t1 + self.w)]
        if self.curve or self.full:
            cands += [(simp(n2(vsub(p, q))), q) for q in ends]
        else:
            for t in (self.t1, self.t1 + self.w):
                a = vadd(self.c, [self.r1 * sp.cos(t), self.r1 * sp.sin(t)])
                b = vadd(self.c, [self.r2 * sp.cos(t), self.r2 * sp.sin(t)])
                cands.append(seg_nearest(p, a, b))
        return best_of(cands)


class ShellR(Region):
    """Annulus / SphericalShell: r1 <= |x - c| <= r2."""
    kind = "shell"

    def __init__(self, c, r1, r2, expr=None, f="Region"):
        self.f = f
        self.c, self.r1, self.r2 = [S(v) for v in c], S(r1), S(r2)
        self.edim = self.dim = len(c)
        self.convex = eqz(self.r1)
        self.expr = expr

    def member(self, p):
        d2 = n2(vsub(p, self.c))
        return sgn(d2 - self.r1 ** 2) >= 0 and sgn(d2 - self.r2 ** 2) <= 0

    def integrate(self, g, Xs):
        n = self.edim
        return ellipsoid_integral(self.c, self.r2 * sp.eye(n), g, Xs) - \
            ellipsoid_integral(self.c, self.r1 * sp.eye(n), g, Xs)

    def bounds(self):
        return [(simp(c - self.r2), simp(c + self.r2)) for c in self.c]

    def nearest(self, p):
        if self.member(p):
            return sp.Integer(0), list(p)
        d = vsub(p, self.c)
        if is_zero_vec(d):
            q = list(self.c)
            q[0] += self.r1
            return simp(self.r1 ** 2), q
        nd = norm(d)
        r = self.r2 if sgn(n2(d) - self.r2 ** 2) > 0 else self.r1
        return simp((nd - r) ** 2), [simp(c) for c in vadd(self.c, vmul(r / nd, d))]

    def sdist(self, p):
        nd = norm(vsub(p, self.c))
        return simp(smax([nd - self.r2, self.r1 - nd]))

    def farthest(self, p):
        return simp(norm(vsub(p, self.c)) + self.r2)

    def boundary(self):
        h = "Circle" if self.edim == 2 else "Sphere"
        if eqz(self.r1):
            return Expr(h, out(self.c), out(self.r2))
        return Expr("RegionUnion", Expr(h, out(self.c), out(self.r1)), Expr(h, out(self.c), out(self.r2)))


class BoxR(Region):
    kind = "Cuboid"

    def __init__(self, lo, hi, expr=None, f="Region"):
        self.f = f
        self.lo, self.hi = [S(v) for v in lo], [S(v) for v in hi]
        for a, b in zip(self.lo, self.hi):
            if lt(b, a):
                raise ValueError(f"{f}: box corners must satisfy min <= max (use CanonicalizeRegion)")
        self.edim = len(lo)
        self.free = [i for i in range(self.edim) if not eqz(self.hi[i] - self.lo[i])]
        self.dim = len(self.free)
        self.convex = True
        self.expr = expr

    def member(self, p):
        return all(le(a, x) and le(x, b) for a, x, b in zip(self.lo, p, self.hi))

    def integrate(self, g, Xs):
        h = S(g)
        fixed = {Xs[i]: self.lo[i] for i in range(self.edim) if i not in self.free}
        h = h.xreplace(fixed)
        return sym_integrate(h, [(Xs[i], self.lo[i], self.hi[i]) for i in self.free]) if self.free else h

    def bounds(self):
        return list(zip(self.lo, self.hi))

    def nearest(self, p):
        q = [smax([a, smin([x, b])]) for a, x, b in zip(self.lo, p, self.hi)]
        return simp(n2(vsub(p, q))), q

    def sdist(self, p):
        if self.dim != self.edim:
            self.need("the signed distance (degenerate boxes)")
        if self.member(p):
            return simp(-smin([smin([x - a, b - x]) for a, x, b in zip(self.lo, p, self.hi)]))
        return self.dist(p)

    def verts(self):
        if self.edim > 8:
            return None
        return [list(c) for c in itertools.product(*[(a, b) for a, b in zip(self.lo, self.hi)])]

    def boundary(self):
        if self.dim < self.edim:
            return self.expr
        lo, hi = self.lo, self.hi
        if self.edim == 1:
            return Expr("Point", out([[lo[0]], [hi[0]]]))
        if self.edim == 2:
            return Expr("Line", out([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]]]))
        if self.edim == 3:
            V, F = G.polyhedron_data(Expr("Cuboid", out(lo), out(hi)), self.f)
            return Expr("Polygon", out([[V[i] for i in fc] for fc in F]))
        self.need("the boundary (boxes in more than 3 dimensions)")


class ParallelotopeR(Region):
    """{p + V t : t in [0, 1]^k}."""
    kind = "Parallelepiped"

    def __init__(self, p, Vs, expr=None, f="Region"):
        self.f = f
        self.p = [S(c) for c in p]
        self.V = [[S(c) for c in v] for v in Vs]
        self.edim = len(p)
        self.dim = len(Vs)
        if affine_rank([self.p] + [vadd(self.p, v) for v in self.V]) < self.dim:
            raise NotImplementedError(f"{f}: degenerate parallelotopes are not supported; use CanonicalizeRegion")
        self.convex = True
        self.expr = expr
        self.A = sp.Matrix(self.V).T

    def coords(self, x):
        A = self.A
        t = (A.T * A).LUsolve(A.T * sp.Matrix(vsub(x, self.p)))
        res = sp.Matrix(vsub(x, self.p)) - A * t
        return [simp(c) for c in t], is_zero_vec(list(res))

    def member(self, x):
        t, ok = self.coords(x)
        return ok and all(sgn(c) >= 0 and sgn(c - 1) <= 0 for c in t)

    def integrate(self, g, Xs):
        k = self.dim
        T = sp.symbols(f"_rt0:{k}", real=True)
        xs = [self.p[i] + sum(self.A[i, j] * T[j] for j in range(k)) for i in range(self.edim)]
        jac = sp.Abs(self.A.det()) if k == self.edim else sp.sqrt((self.A.T * self.A).det())
        return simp(jac) * sym_integrate(subst(g, Xs, xs), [(t, 0, 1) for t in T])

    def verts(self):
        return [vadd(self.p, [sum(c * v[i] for c, v in zip(w, self.V)) for i in range(self.edim)])
                for w in itertools.product((0, 1), repeat=self.dim)]

    def bounds(self):
        V = self.verts()
        return [(smin([q[i] for q in V]), smax([q[i] for q in V])) for i in range(self.edim)]

    def nearest(self, x):
        if self.member(x):
            return sp.Integer(0), list(x)
        # the nearest point lies on a face; faces are lower-dimensional parallelotopes -> simplex split
        simp_ = []
        for perm in itertools.permutations(range(self.dim)):
            P = [list(self.p)]
            cur = list(self.p)
            for j in perm:
                cur = vadd(cur, self.V[j])
                P.append(cur)
            simp_.append(P)
        return best_of([simplex_nearest(x, s) for s in simp_])


def _line_frame(p, V, f):
    """Orthonormal-free description of the affine set p + span(V): returns projector data."""
    A = sp.Matrix([[S(c) for c in v] for v in V]).T
    if A.rank(simplify=True) < len(V):
        raise ValueError(f"{f}: the direction vectors must be linearly independent")
    return A


class FlatR(Region):
    """Unbounded affine pieces: p + span(V) (+ t w, t >= 0) ; ray / half-plane / affine half-space / lines /
    planes / hyperplanes / half-spaces / full space."""
    kind = "affine"
    bounded = False

    def __init__(self, p, V, w=None, ray=None, expr=None, f="Region", halfspace=None):
        self.f = f
        self.expr = expr
        self.halfspace = halfspace      # (n, c): n.x <= c  (full-dimensional half-space)
        self.hyper = None
        if halfspace is not None:
            nrm, c = halfspace
            self.n, self.c = [S(v) for v in nrm], S(c)
            self.edim = self.dim = len(nrm)
            self.convex = True
            return
        self.p = [S(c) for c in p]
        self.V = [[S(c) for c in v] for v in V]
        self.w = [S(c) for c in w] if w is not None else None
        self.ray = ray
        self.edim = len(self.p)
        self.dim = len(self.V) + (1 if self.w is not None else 0)
        self.convex = True
        if self.V or self.w is not None:
            cols = self.V + ([self.w] if self.w is not None else [])
            if sp.Matrix(cols).rank(simplify=True) < len(cols):
                raise ValueError(f"{f}: the direction vectors must be linearly independent")
        self.bounded = self.dim == 0

    def _decomp(self, x):
        """(coefficients on V, coefficient on w (or None), projection point, is_in_affine_hull)."""
        cols = self.V + ([self.w] if self.w is not None else [])
        if not cols:
            return [], None, list(self.p), is_zero_vec(vsub(x, self.p))
        A = sp.Matrix(cols).T
        t = (A.T * A).LUsolve(A.T * sp.Matrix(vsub(x, self.p)))
        t = [simp(c) for c in t]
        q = vadd(self.p, [sum(A[i, j] * t[j] for j in range(len(cols))) for i in range(self.edim)])
        q = [simp(c) for c in q]
        tw = t[-1] if self.w is not None else None
        return t, tw, q, is_zero_vec(vsub(x, q))

    def member(self, x):
        if self.halfspace is not None:
            return le(dot(self.n, x), self.c)
        t, tw, q, ok = self._decomp(x)
        return ok and (tw is None or sgn(tw) >= 0)

    def nearest(self, x):
        if self.halfspace is not None:
            s = dot(self.n, x) - self.c
            if sgn(s) <= 0:
                return sp.Integer(0), list(x)
            q = vsub(x, vmul(s / n2(self.n), self.n))
            return simp(s ** 2 / n2(self.n)), [simp(c) for c in q]
        t, tw, q, ok = self._decomp(x)
        if tw is not None and sgn(tw) < 0:
            # clamp the w coefficient: project onto p + span(V)
            sub = FlatR(self.p, self.V, f=self.f)
            return sub.nearest(x)
        return simp(n2(vsub(x, q))), q

    def sdist(self, x):
        if self.halfspace is not None:
            return simp((dot(self.n, x) - self.c) / norm(self.n))
        self.need("the signed distance (lower-dimensional flats)")

    def integrate(self, g, Xs):
        if self.dim == 0:
            return subst(g, Xs, self.p)
        if eqz(S(g)):
            return sp.Integer(0)
        if S(g).free_symbols & set(Xs) or not S(g).is_number:
            self.need("integration over unbounded regions")
        return sp.oo * sp.sign(S(g))

    def bounds(self):
        if self.halfspace is not None:
            nz = [i for i in range(self.edim) if not eqz(self.n[i])]
            res = []
            for i in range(self.edim):
                if len(nz) == 1 and nz[0] == i:
                    b = self.c / self.n[i]
                    res.append((-sp.oo, simp(b)) if sgn(self.n[i]) > 0 else (simp(b), sp.oo))
                else:
                    res.append((-sp.oo, sp.oo))
            return res
        res = []
        for i in range(self.edim):
            free = any(not eqz(v[i]) for v in self.V)
            if free:
                res.append((-sp.oo, sp.oo))
            elif self.w is not None and not eqz(self.w[i]):
                res.append((self.p[i], sp.oo) if sgn(self.w[i]) > 0 else (-sp.oo, self.p[i]))
            else:
                res.append((self.p[i], self.p[i]))
        return res

    def farthest(self, p):
        if self.dim == 0:
            return sqrt_out(n2(vsub(p, self.p)))
        return sp.oo

    def boundary(self):
        if self.halfspace is not None:
            return Expr("Hyperplane", out(self.n), out(self.c))
        if self.dim < self.edim:
            return self.expr
        self.need("the boundary")


class EmptyR(Region):
    kind = "EmptyRegion"

    def __init__(self, n, expr=None):
        self.edim = n
        self.dim = -sp.oo
        self.convex = True
        self.expr = expr

    def member(self, p):
        return False

    def integrate(self, g, Xs):
        return sp.Integer(0)

    def nearest(self, p):
        return sp.oo, None

    def dist(self, p):
        return sp.oo

    def boundary(self):
        return self.expr

    def bounds(self):
        raise ValueError(f"{self.f}: the empty region has no bounds")


class FullR(FlatR):
    kind = "FullRegion"

    def __init__(self, n, expr=None, f="Region"):
        I = [[sp.Integer(int(i == j)) for j in range(n)] for i in range(n)]
        FlatR.__init__(self, [sp.Integer(0)] * n, I, expr=expr, f=f)

    def boundary(self):
        return Expr("EmptyRegion", self.edim)

    def sdist(self, x):
        return -sp.oo


class RevolutionR(Region):
    """Cylinder, Cone, CapsuleShape (3-D) and StadiumShape (2-D) about the segment p1 -> p2 with radius r."""

    def __init__(self, kind, p1, p2, r, expr=None, f="Region"):
        self.f = f
        self.kind = kind
        self.p1, self.p2, self.r = [S(c) for c in p1], [S(c) for c in p2], S(r)
        self.edim = len(self.p1)
        self.dim = self.edim
        self.ax = vsub(self.p2, self.p1)
        self.Lax = simp(norm(self.ax))
        if eqz(self.Lax) and kind in ("Cylinder", "Cone"):
            raise ValueError(f"{f}: degenerate {kind} (coincident end points)")
        self.u = [simp(c / self.Lax) for c in self.ax] if not eqz(self.Lax) else None
        self.convex = True
        self.expr = expr

    def _sr(self, p):
        """Axial coordinate s (from p1, along u) and radial distance^2 of p."""
        d = vsub(p, self.p1)
        s = dot(d, self.u)
        r2 = simp(n2(d) - s ** 2)
        return simp(s), r2

    def member(self, p):
        if self.kind in ("CapsuleShape", "StadiumShape"):
            return sgn(seg_nearest(p, self.p1, self.p2)[0] - self.r ** 2) <= 0
        s, r2 = self._sr(p)
        if sgn(s) < 0 or sgn(s - self.Lax) > 0:
            return False
        rad = self.r if self.kind == "Cylinder" else self.r * (1 - s / self.Lax)
        return sgn(r2 - rad ** 2) <= 0

    def integrate(self, g, Xs):
        r, Lx = self.r, self.Lax
        if self.kind == "Cylinder":
            return revolve_integral(g, Xs, self.p1, self.u, [(0, Lx, r)])
        s = sp.Symbol("_rs", real=True)
        if self.kind == "Cone":
            return revolve_integral(g, Xs, self.p1, self.u, [(0, Lx, r * (1 - s / Lx))])
        if self.u is None:
            n = self.edim
            return ellipsoid_integral(self.p1, r * sp.eye(n), g, Xs)
        if self.kind == "CapsuleShape":
            return revolve_integral(g, Xs, self.p1, self.u, [(-r, 0, sp.sqrt(r ** 2 - s ** 2)), (0, Lx, r),
                                                             (Lx, Lx + r, sp.sqrt(r ** 2 - (s - Lx) ** 2))])
        return sweep2d_integral(g, Xs, self.p1, self.u, [(-r, 0, sp.sqrt(r ** 2 - s ** 2)), (0, Lx, r),
                                                         (Lx, Lx + r, sp.sqrt(r ** 2 - (s - Lx) ** 2))])

    def measure(self):
        r, Lx = self.r, self.Lax
        if self.kind == "Cylinder":
            return simp(sp.pi * r ** 2 * Lx)
        if self.kind == "Cone":
            return simp(sp.pi * r ** 2 * Lx / 3)
        if self.kind == "CapsuleShape":
            return simp(sp.pi * r ** 2 * Lx + 4 * sp.pi * r ** 3 / 3)
        return simp(2 * r * Lx + sp.pi * r ** 2)

    def bounds(self):
        res = []
        for i in range(self.edim):
            if self.kind in ("CapsuleShape", "StadiumShape"):
                lo = smin([self.p1[i], self.p2[i]]) - self.r
                hi = smax([self.p1[i], self.p2[i]]) + self.r
            else:
                e = self.r * sp.sqrt(1 - self.u[i] ** 2)
                if self.kind == "Cylinder":
                    lo, hi = smin([self.p1[i], self.p2[i]]) - e, smax([self.p1[i], self.p2[i]]) + e
                else:
                    lo, hi = smin([self.p1[i] - e, self.p2[i]]), smax([self.p1[i] + e, self.p2[i]])
            res.append((simp(lo), simp(hi)))
        return res

    def _profile_nearest(self, s, rho):
        """Nearest point of the 2-D profile region (axial s, radial rho >= 0): (d^2, (s', rho'))."""
        Lx, r = self.Lax, self.r
        if self.kind == "Cylinder":
            prof = [[0, 0], [Lx, 0], [Lx, r], [0, r]]
        else:
            prof = [[0, 0], [Lx, 0], [0, r]]
        q = [s, rho]
        if ring_contains(prof, q) >= 0:
            return sp.Integer(0), q
        return best_of([seg_nearest(q, prof[i], prof[(i + 1) % len(prof)]) for i in range(len(prof))])

    def nearest(self, p):
        if self.kind in ("CapsuleShape", "StadiumShape"):
            d2, q = seg_nearest(p, self.p1, self.p2)
            if sgn(d2 - self.r ** 2) <= 0:
                return sp.Integer(0), list(p)
            d = sp.sqrt(d2)
            qq = vadd(q, vmul(self.r / d, vsub(p, q)))
            return simp((d - self.r) ** 2), [simp(c) for c in qq]
        s, r2 = self._sr(p)
        rho = simp(sp.sqrt(r2))
        d2, (s2, rho2) = self._profile_nearest(s, rho)
        if eqz(d2):
            return sp.Integer(0), list(p)
        radial = vsub(vsub(p, self.p1), vmul(s, self.u))
        if is_zero_vec(radial):
            e1, _ = frame(self.u)
            dirn = e1
        else:
            dirn = vmul(1 / rho, radial)
        q = vadd(vadd(self.p1, vmul(s2, self.u)), vmul(rho2, dirn))
        return simp(d2), [simp(c) for c in q]

    def sdist(self, p):
        if self.kind in ("CapsuleShape", "StadiumShape"):
            return simp(sp.sqrt(seg_nearest(p, self.p1, self.p2)[0]) - self.r)
        if not self.member(p):
            return self.dist(p)
        s, r2 = self._sr(p)
        rho = simp(sp.sqrt(r2))
        Lx, r = self.Lax, self.r
        prof = [[0, 0], [Lx, 0], [Lx, r], [0, r]] if self.kind == "Cylinder" else [[0, 0], [Lx, 0], [0, r]]
        # distance to the boundary of the profile, excluding the axis edge
        segs = [(prof[i], prof[(i + 1) % len(prof)]) for i in range(len(prof))]
        segs = [sg for sg in segs if not (eqz(sg[0][1]) and eqz(sg[1][1]))]
        return simp(-sp.sqrt(best_of([seg_nearest([s, rho], a, b) for a, b in segs])[0]))

    def farthest(self, p):
        if self.kind in ("CapsuleShape", "StadiumShape"):
            return simp(smax([norm(vsub(p, self.p1)), norm(vsub(p, self.p2))]) + self.r)
        s, r2 = self._sr(p)
        rho = sp.sqrt(r2)
        cands = [(0, -self.r), (self.Lax, -self.r)] if self.kind == "Cylinder" else [(0, -self.r), (self.Lax, 0)]
        return simp(sp.sqrt(smax([(s - a) ** 2 + (rho - b) ** 2 for a, b in cands])))


class TorusR(Region):
    """Torus (surface) / FilledTorus (solid) about the z axis through c: tube centre radius R, tube radius r."""

    def __init__(self, c, rin, rout, solid, expr=None, f="Region"):
        self.f = f
        self.kind = "FilledTorus" if solid else "Torus"
        self.c = [S(v) for v in c]
        self.R_ = simp((S(rin) + S(rout)) / 2)
        self.r = simp((S(rout) - S(rin)) / 2)
        self.solid = solid
        self.edim = 3
        self.dim = 3 if solid else 2
        self.convex = False
        self.expr = expr

    def _q(self, p):
        d = vsub(p, self.c)
        rho = sp.sqrt(d[0] ** 2 + d[1] ** 2)
        return simp((rho - self.R_) ** 2 + d[2] ** 2), d, rho

    def member(self, p):
        q, _, _ = self._q(p)
        s = sgn(q - self.r ** 2)
        return s <= 0 if self.solid else s == 0

    def integrate(self, g, Xs):
        th, ph, rho = sp.symbols("_rth _rph _rrho", real=True)
        R_, r = self.R_, self.r
        if self.solid:
            xs = [self.c[0] + (R_ + rho * sp.cos(ph)) * sp.cos(th), self.c[1] + (R_ + rho * sp.cos(ph)) * sp.sin(th),
                  self.c[2] + rho * sp.sin(ph)]
            h = subst(g, Xs, xs) * rho * (R_ + rho * sp.cos(ph))
            return simp(sym_integrate(h, [(th, 0, 2 * sp.pi), (ph, 0, 2 * sp.pi), (rho, 0, r)]))
        xs = [self.c[0] + (R_ + r * sp.cos(ph)) * sp.cos(th), self.c[1] + (R_ + r * sp.cos(ph)) * sp.sin(th),
              self.c[2] + r * sp.sin(ph)]
        h = subst(g, Xs, xs) * r * (R_ + r * sp.cos(ph))
        return simp(sym_integrate(h, [(th, 0, 2 * sp.pi), (ph, 0, 2 * sp.pi)]))

    def measure(self):
        if self.solid:
            return simp(2 * sp.pi ** 2 * self.R_ * self.r ** 2)
        return simp(4 * sp.pi ** 2 * self.R_ * self.r)

    def bounds(self):
        e = self.R_ + self.r
        return [(self.c[0] - e, self.c[0] + e), (self.c[1] - e, self.c[1] + e), (self.c[2] - self.r,
                                                                                   self.c[2] + self.r)]

    def dist(self, p):
        q, _, _ = self._q(p)
        d = simp(sp.sqrt(q) - self.r)
        if self.solid:
            return d if sgn(d) > 0 else sp.Integer(0)
        return simp(sp.Abs(d))

    def nearest(self, p):
        self.need("the nearest point")

    def sdist(self, p):
        if not self.solid:
            self.need("the signed distance (surfaces)")
        q, _, _ = self._q(p)
        return simp(sp.sqrt(q) - self.r)

    def boundary(self):
        if self.solid:
            rin, rout = self.R_ - self.r, self.R_ + self.r
            return Expr("Torus", out(self.c), out([rin, rout]))
        return self.expr


class UnionCellsR(Region):
    """Union of interior-disjoint cells (mesh cells): measures add up."""
    kind = "MeshRegion"

    def __init__(self, cells, edim, expr=None, f="Region", boundary_fn=None, convex=None):
        self.f = f
        self.cells = cells
        self.edim = edim
        dims = [c.dim for c in cells]
        self.dim = max(dims) if dims else -sp.oo
        self.top = [c for c in cells if c.dim == self.dim]
        self.convex = convex
        self.expr = expr
        self.boundary_fn = boundary_fn

    def member(self, p):
        return any(c.member(p) for c in self.cells)

    def integrate(self, g, Xs):
        return sum((c.integrate(g, Xs) for c in self.top), sp.Integer(0))

    def bounds(self):
        bs = [c.bounds() for c in self.cells]
        return [(smin([b[i][0] for b in bs]), smax([b[i][1] for b in bs])) for i in range(self.edim)]

    def nearest(self, p):
        return best_of([c.nearest(p) for c in self.cells])

    def verts(self):
        V = []
        for c in self.cells:
            v = c.verts()
            if v is None:
                return None
            V += v
        return V

    def boundary(self):
        if self.boundary_fn is not None:
            return self.boundary_fn()
        self.need("the boundary")

    def sdist(self, p):
        if self.dim != self.edim or self.boundary_fn is None:
            self.need("the signed distance")
        b = region_of(self.boundary_fn(), self.f)
        d = b.dist(p)
        return -d if self.member(p) else d


# ───────────────────────────── formula regions ─────────────────────────────

def _vars_spec(v, f):
    """{x, y, ...} with optional {x, a, b} ranges -> (symbols (Mikoshi names), sympy symbols, ranges)."""
    if not K.is_list(v):
        if isinstance(v, Symbol):
            v = L(v)
        else:
            raise TypeError(f"{f}: the variables must be a list {{x1, x2, ...}}, not {K_input(v)}")
    names, syms, ranges = [], [], []
    for a in v.args:
        if isinstance(a, Symbol):
            names.append(a)
            ranges.append(None)
        elif K.is_list(a) and len(a.args) == 3 and isinstance(a.args[0], Symbol):
            names.append(a.args[0])
            lo, hi = sx(a.args[1], f, "range bound"), sx(a.args[2], f, "range bound")
            if lo.free_symbols or hi.free_symbols:
                raise Unevaluated()
            if lt(hi, lo):
                raise ValueError(f"{f}: variable range {{{a.args[0]}, {lo}, {hi}}} has min > max")
            ranges.append((lo, hi))
        else:
            raise TypeError(f"{f}: each variable must be a symbol or {{x, min, max}}, not {K_input(a)}")
    if len({n.name for n in names}) != len(names):
        raise ValueError(f"{f}: variables must be distinct")
    syms = [K.to_sp(n) for n in names]
    return names, syms, ranges


def _to_sp_cond(c, f):
    try:
        v = K.to_sp(c)
    except Exception:
        raise TypeError(f"{f}: {K_input(c)} is not a condition")
    return v


def _truth(v, f):
    if v is sp.true or v is True:
        return True
    if v is sp.false or v is False:
        return False
    try:
        s = sp.simplify(v)
    except Exception:
        s = v
    if s is sp.true or s is True:
        return True
    if s is sp.false or s is False:
        return False
    if isinstance(s, sp.Rel) or isinstance(s, sp.logic.boolalg.Boolean):
        # decide each relational numerically (exact sign)
        def dec(e):
            if e in (sp.true, sp.false):
                return bool(e)
            if isinstance(e, sp.And):
                return all(dec(a) for a in e.args)
            if isinstance(e, sp.Or):
                return any(dec(a) for a in e.args)
            if isinstance(e, sp.Not):
                return not dec(e.args[0])
            if isinstance(e, sp.Rel):
                d = S(e.lhs - e.rhs)
                if d.free_symbols:
                    raise Unevaluated()
                s_ = sgn(d)
                return {"<": s_ < 0, "<=": s_ <= 0, ">": s_ > 0, ">=": s_ >= 0, "==": s_ == 0, "!=": s_ != 0}[e.rel_op]
            raise NotImplementedError(f"{f}: cannot decide the condition {e}")
        return dec(s)
    raise NotImplementedError(f"{f}: cannot decide the condition {s}")


def _rels(c):
    """Conjunction -> list of relationals (None if not a pure conjunction of relationals)."""
    if isinstance(c, sp.And):
        out_ = []
        for a in c.args:
            r = _rels(a)
            if r is None:
                return None
            out_ += r
        return out_
    if isinstance(c, sp.Rel):
        return [c]
    if c is sp.true:
        return []
    return None


class ImplicitR(Region):
    kind = "ImplicitRegion"

    def __init__(self, cond, names, syms, ranges, expr=None, f="Region"):
        self.f = f
        self.cond = cond
        self.names, self.syms, self.ranges = names, syms, ranges
        self.edim = len(syms)
        self.expr = expr
        self.delegate = None
        self.bounded = None
        self.convex = None
        self.dim = None
        self._analyse()

    def full_cond(self):
        c = self.cond
        for s, r in zip(self.syms, self.ranges):
            if r is not None:
                c = sp.And(c, s >= r[0], s <= r[1])
        return c

    def _analyse(self):
        c = self.full_cond()
        rels = _rels(c)
        n = self.edim
        if rels is not None:
            polys = []
            ok = True
            for r in rels:
                if r.rel_op not in ("<=", "<", ">=", ">"):
                    ok = False
                    break
                e = sp.expand(r.lhs - r.rhs) if r.rel_op in ("<=", "<") else sp.expand(r.rhs - r.lhs)
                polys.append(e)
            if ok and polys:
                # linear: bounded convex polytope
                lin = all(sp.Poly(e, *self.syms).total_degree() <= 1 if e.free_symbols <= set(self.syms) else False
                          for e in polys)
                if lin and n in (1, 2, 3):
                    poly_ = _halfspace_polytope([( [sp.Poly(e, *self.syms).coeff_monomial(s) for s in self.syms],
                                                  -sp.Poly(e, *self.syms).coeff_monomial(1)) for e in polys], n, self.f)
                    if poly_ is not None:
                        self.delegate = poly_
                elif len(polys) == 1 and n >= 1:
                    q = _quadric_ball(polys[0], self.syms, self.f)
                    if q is not None:
                        self.delegate = q
        if self.delegate is None and n == 1:
            try:
                sol = sp.solveset(c, self.syms[0], sp.S.Reals)
            except Exception:
                sol = None
            if sol is not None and not isinstance(sol, sp.ConditionSet):
                self.delegate = _interval_set_region(sol, self.f)
        if self.delegate is not None:
            d = self.delegate
            self.dim, self.bounded, self.convex = d.dim, d.bounded, d.convex
        else:
            self.bounded = True if all(r is not None for r in self.ranges) else None

    def member(self, p):
        v = self.cond.xreplace(dict(zip(self.syms, [S(c) for c in p])))
        ok = _truth(v, self.f)
        for x, r in zip(p, self.ranges):
            if r is not None and not (le(r[0], x) and le(x, r[1])):
                return False
        return ok

    def _need_delegate(self, what):
        if self.delegate is None:
            raise NotImplementedError(f"{self.f}: {what} of this ImplicitRegion is not supported (supported: "
                                      f"linear inequality systems in up to 3 variables, a single quadratic "
                                      f"ellipsoid inequality, and 1-D conditions)")
        return self.delegate

    def integrate(self, g, Xs):
        return self._need_delegate("integration").integrate(g, Xs)

    def bounds(self):
        return self._need_delegate("the bounds").bounds()

    def nearest(self, p):
        return self._need_delegate("the distance").nearest(p)

    def sdist(self, p):
        return self._need_delegate("the signed distance").sdist(p)

    def boundary(self):
        return self._need_delegate("the boundary").boundary()

    def verts(self):
        return self.delegate.verts() if self.delegate is not None else None

    def farthest(self, p):
        return self._need_delegate("the farthest distance").farthest(p)


def _interval_set_region(sol, f):
    """SymPy subset of the reals -> 1-D region (union of closed intervals and points)."""
    parts = sol.args if isinstance(sol, sp.Union) else [sol]
    cells = []
    for prt in parts:
        if isinstance(prt, sp.Interval):
            a, b = prt.start, prt.end
            if a == b:
                cells.append(PointsR([[a]]))
            elif a.is_infinite or b.is_infinite:
                return None
            else:
                cells.append(BoxR([a], [b], f=f))
        elif isinstance(prt, sp.FiniteSet):
            cells += [PointsR([[v]]) for v in prt.args]
        elif prt is sp.S.EmptySet:
            continue
        else:
            return None
    if not cells:
        return EmptyR(1)
    if len(cells) == 1:
        return cells[0]
    return UnionCellsR(cells, 1, f=f)


def _quadric_ball(e, syms, f):
    """e <= 0 with e = (x - c)^T A (x - c) - k, A positive definite -> ellipsoid region."""
    try:
        P = sp.Poly(e, *syms)
    except Exception:
        return None
    if P.total_degree() != 2:
        return None
    n = len(syms)
    A = sp.zeros(n, n)
    bvec = sp.zeros(n, 1)
    for i, a in enumerate(syms):
        for j, b in enumerate(syms):
            mon = [0] * n
            mon[i] += 1
            mon[j] += 1
            co = P.coeff_monomial(tuple(mon))
            A[i, j] = co if i == j else co / 2
        mon = [0] * n
        mon[i] = 1
        bvec[i] = P.coeff_monomial(tuple(mon))
    c0 = P.coeff_monomial(tuple([0] * n))
    if any(S(v).free_symbols for v in list(A) + list(bvec) + [c0]):
        return None
    if not A.is_positive_definite:
        return None
    cen = -(A.inv() * bvec) / 2
    k = simp((cen.T * A * cen)[0] - c0)
    if sgn(k) <= 0:
        return None
    Sg = A.inv() * k
    if Sg.is_diagonal():
        Lm = sp.diag(*[simp(sp.sqrt(Sg[i, i])) for i in range(n)])
        rr = Lm[0, 0] if all(eqz(Lm[i, i] - Lm[0, 0]) for i in range(n)) else None
    else:
        Lm = Sg.cholesky(hermitian=False)
        rr = None
    return EllipsoidR([simp(v) for v in cen], Lm, f=f, round_r=rr)


def _halfspace_polytope(hs, n, f):
    """Bounded intersection of half-spaces a.x <= b in R^n (n <= 3) -> polytope region (exact), or None."""
    hs = [([S(c) for c in a], S(b)) for a, b in hs]
    if n == 1:
        lo, hi = -sp.oo, sp.oo
        for a, b in hs:
            if eqz(a[0]):
                if sgn(b) < 0:
                    return EmptyR(1)
                continue
            v = b / a[0]
            if sgn(a[0]) > 0:
                hi = v if hi == sp.oo or lt(v, hi) else hi
            else:
                lo = v if lo == -sp.oo or lt(lo, v) else lo
        if lo == -sp.oo or hi == sp.oo:
            return None
        if lt(hi, lo):
            return EmptyR(1)
        return PointsR([[lo]]) if eqz(hi - lo) else BoxR([lo], [hi], f=f)
    V = []
    for comb in itertools.combinations(range(len(hs)), n):
        A = sp.Matrix([hs[i][0] for i in comb])
        if eqz(A.det()):
            continue
        x = A.LUsolve(sp.Matrix([hs[i][1] for i in comb]))
        x = [simp(c) for c in x]
        if all(le(dot(a, x), b) for a, b in hs) and not any(is_zero_vec(vsub(x, v)) for v in V):
            V.append(x)
    if not V:
        return None
    # boundedness: every direction must be blocked; check the recession cone {d : a.d <= 0} is {0}
    for d in itertools.product((-1, 0, 1), repeat=n):
        pass
    if not _bounded_halfspaces(hs, n):
        return None
    rk = affine_rank(V)
    if rk < n:
        if rk == 0:
            return PointsR([V[0]])
        if rk == 1 and n >= 1:
            ends = _extreme_pair(V)
            return SimplexR([ends], f=f)
        if n == 3 and rk == 2:
            ring = _convex_ring3(V, f)
            return Polygon3R([(ring, [])], f=f)
        return None
    return convex_hull_region(V, f)


def _bounded_halfspaces(hs, n):
    """True if {d : a.d <= 0 for all a} == {0} (checked by LP-free exact test on candidate rays)."""
    # recession cone is pointed and trivial iff no nonzero d with a.d <= 0; candidate extreme rays are
    # intersections of n - 1 hyperplanes a.d = 0
    A = [a for a, b in hs]
    if n == 1:
        return any(sgn(a[0]) > 0 for a in A) and any(sgn(a[0]) < 0 for a in A)
    cands = []
    for comb in itertools.combinations(range(len(A)), n - 1):
        M = sp.Matrix([A[i] for i in comb])
        ns = M.nullspace()
        if len(ns) == 1:
            d = list(ns[0])
            cands += [d, [-c for c in d]]
    if len(A) < n - 1 or not cands:
        return False
    for d in cands:
        if all(sgn(dot(a, d)) <= 0 for a in A):
            return False
    return True


def _extreme_pair(V):
    best = None
    for a, b in itertools.combinations(V, 2):
        d = n2(vsub(a, b))
        if best is None or lt(best[0], d):
            best = (d, [a, b])
    return best[1]


def _convex_ring3(V, f):
    nrm = None
    for a, b, c in itertools.combinations(V, 3):
        u, v = vsub(b, a), vsub(c, a)
        cr = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        if not is_zero_vec(cr):
            nrm = cr
            break
    k = max(range(3), key=lambda i: abs(num(nrm[i])))
    keep = [i for i in range(3) if i != k]
    P2 = [[p[i] for i in keep] for p in V]
    idx = convex_hull_2d(P2)
    return [V[i] for i in idx]


def convex_hull_2d(P):
    """Indices of the convex hull of exact 2-D points, counterclockwise, collinear points dropped."""
    idx = sorted(range(len(P)), key=lambda i: (num(P[i][0]), num(P[i][1])))
    uniq = []
    for i in idx:
        if not uniq or not is_zero_vec(vsub(P[i], P[uniq[-1]])):
            uniq.append(i)
    idx = uniq
    if len(idx) <= 2:
        return idx

    def half(order):
        h = []
        for i in order:
            while len(h) >= 2 and sgn(orient2(P[h[-2]], P[h[-1]], P[i])) <= 0:
                h.pop()
            h.append(i)
        return h
    lo, up = half(idx), half(idx[::-1])
    return lo[:-1] + up[:-1]


def convex_hull_3d(P, f):
    """(vertex indices, faces as CCW-from-outside index cycles) of the convex hull of exact 3-D points."""
    n = len(P)
    if n > 60:
        return _hull3d_scipy(P, f)
    planes = {}
    for i, j, k in itertools.combinations(range(n), 3):
        u, v = vsub(P[j], P[i]), vsub(P[k], P[i])
        nr = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        if is_zero_vec(nr):
            continue
        side = [sgn(dot(nr, vsub(P[m], P[i]))) for m in range(n)]
        if all(s <= 0 for s in side) or all(s >= 0 for s in side):
            if all(s >= 0 for s in side):
                nr = [-c for c in nr]
            on = tuple(m for m in range(n) if side[m] == 0)
            planes[on] = nr
    faces = []
    for on, nr in planes.items():
        k = max(range(3), key=lambda i: abs(num(nr[i])))
        keep = [i for i in range(3) if i != k]
        P2 = [[P[m][i] for i in keep] for m in on]
        ring = [on[i] for i in convex_hull_2d(P2)]
        # orient counterclockwise seen from outside (normal nr)
        a, b, c = P[ring[0]], P[ring[1]], P[ring[2]]
        u, v = vsub(b, a), vsub(c, a)
        cr = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        if sgn(dot(cr, nr)) < 0:
            ring = ring[::-1]
        faces.append(ring)
    verts = sorted({m for fc in faces for m in fc})
    return verts, faces


def _hull3d_scipy(P, f):
    G._require_scipy(f)
    import numpy as np
    from scipy.spatial import ConvexHull
    A = np.array([[float(num(c)) for c in p] for p in P])
    h = ConvexHull(A)
    cand = sorted(set(int(i) for i in h.vertices))
    vs, fs = convex_hull_3d([P[i] for i in cand], f) if len(cand) <= 60 else (None, None)
    if vs is None:
        raise NotImplementedError(f"{f}: exact convex hulls of more than 60 extreme points are not supported")
    return [cand[i] for i in vs], [[cand[i] for i in fc] for fc in fs]


def convex_hull_region(V, f):
    """Region object of the convex hull of exact points (any affine rank, 1-3 dimensions)."""
    n = len(V[0])
    rk = affine_rank(V)
    if rk == 0:
        return PointsR([V[0]])
    if rk == 1:
        return SimplexR([_extreme_pair(V)], f=f)
    if n == 2:
        idx = convex_hull_2d(V)
        return Polygon2R([([V[i] for i in idx], [])], f=f)
    if n == 3 and rk == 2:
        return Polygon3R([(_convex_ring3(V, f), [])], f=f)
    if n == 3:
        vs, fs = convex_hull_3d(V, f)
        return PolyhedronR(V, fs, f=f)
    raise NotImplementedError(f"{f}: convex hulls in {n} dimensions are not supported")
