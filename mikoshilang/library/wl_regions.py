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


_SGN_CACHE = {}


def sgn(v):
    """Exact sign of a concrete real number (tolerance 1e-12 for inexact values); fast path by high-precision
    evaluation, zero decided by minimal polynomials (algebraic numbers) or SymPy's equals."""
    v = S(v)
    if v is sp.oo:
        return 1
    if v is -sp.oo:
        return -1
    if v.is_Rational:
        return (v.p > 0) - (v.p < 0)
    if v.has(sp.Float):
        return fsign(v)
    if v.free_symbols:
        raise Unevaluated()
    key = v
    if key in _SGN_CACHE:
        return _SGN_CACHE[key]
    res = None
    try:
        a = sp.N(v, 30)
        if a.is_number and not a.has(sp.I) and abs(a) > sp.Float(10) ** -20:
            res = 1 if a > 0 else -1
    except Exception:
        pass
    if res is None:
        res = _sgn_slow(v)
    if len(_SGN_CACHE) > 20000:
        _SGN_CACHE.clear()
    _SGN_CACHE[key] = res
    return res


def _sgn_slow(v):
    w = sp.expand(v)
    if w == 0:
        return 0
    alg = not any(isinstance(a, (sp.Function,)) or a in (sp.pi, sp.E) for a in sp.preorder_traversal(w)
                  if isinstance(a, sp.Basic))
    if alg:
        try:
            z = sp.Symbol("_z")
            if sp.minimal_polynomial(w, z) == z:
                return 0
        except Exception:
            pass
    else:
        try:
            if sp.simplify(w) == 0:
                return 0
        except Exception:
            pass
        e = w.equals(0)
        if e is True:
            return 0
    a = sp.N(w, 120)
    if not a.is_number or a.has(sp.I):
        raise NotImplementedError("Region: cannot decide the sign of a value")
    if abs(a) < sp.Float(10) ** -100:
        if alg:
            raise ArithmeticError("Region: sign decision failed verification")
        return 0
    return 1 if a > 0 else -1


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
            tris += [list(t) for t in G._triangulate(o, self.f)]
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
                for t in G._triangulate(r, self.f):
                    tot += sgn_ * simplex_integral([self.lift(q) for q in t], g, Xs)
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
    back = {tuple(q): p for q, p in zip(R2, P)}
    return [[back[tuple(q)] for q in t] for t in G._triangulate(R2, f)]


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
        self.convex = (not self.voids) and all(
            sgn(dot(_cross3(vsub(T[1], T[0]), vsub(T[2], T[0])), vsub(q, T[0]))) <= 0
            for T in self.tris for q in self.V)
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


_RELS = {"Less": sp.Lt, "LessEqual": sp.Le, "Greater": sp.Gt, "GreaterEqual": sp.Ge, "Equal": sp.Eq,
         "Unequal": sp.Ne}


def _to_sp_cond(c, f):
    """MikoshiLang condition (relations combined with And / Or / Not / Xor) -> SymPy Boolean."""
    if c is True:
        return sp.true
    if c is False:
        return sp.false
    if isinstance(c, Expr) and isinstance(c.head, str):
        h = c.head
        if h in ("And", "Or", "Not", "Xor", "Implies", "Nand", "Nor"):
            args = [_to_sp_cond(a, f) for a in c.args]
            op = {"And": sp.And, "Or": sp.Or, "Not": sp.Not, "Xor": sp.Xor, "Implies": sp.Implies,
                  "Nand": sp.Nand, "Nor": sp.Nor}[h]
            return op(*args)
        if h in _RELS and len(c.args) >= 2:
            vals = [sx(a, f, "condition operand") for a in c.args]
            parts = [_RELS[h](vals[i], vals[i + 1]) for i in range(len(vals) - 1)]
            return sp.And(*parts) if len(parts) > 1 else parts[0]
        if h == "Inequality":
            a = list(c.args)
            parts = []
            for i in range(1, len(a) - 1, 2):
                op = a[i].name if isinstance(a[i], Symbol) else None
                if op not in _RELS:
                    raise TypeError(f"{f}: {K_input(c)} is not a condition")
                parts.append(_RELS[op](sx(a[i - 1], f), sx(a[i + 1], f)))
            return sp.And(*parts)
    if isinstance(c, Symbol):
        raise Unevaluated()
    raise TypeError(f"{f}: {K_input(c)} is not a condition")


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
                sol = c.as_set() if c.free_symbols else (sp.S.Reals if c is sp.true else sp.S.EmptySet)
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
    if not _bounded_halfspaces(hs, n):
        return None
    if not V:
        return EmptyR(n)
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


# ───────────────────────────── H-representations, containment, disjointness ─────────────────────────────

def _cross3(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def hrep(R_):
    """[(a, b)] with R_ = {x : a.x <= b for all} for full-dimensional convex polytopes (and half-spaces), else None."""
    if isinstance(R_, BoxR) and R_.dim == R_.edim:
        n = R_.edim
        res = []
        for i in range(n):
            e = [sp.Integer(int(i == j)) for j in range(n)]
            res += [(e, R_.hi[i]), ([-c for c in e], -R_.lo[i])]
        return res
    if isinstance(R_, SimplexR) and len(R_.S) == 1 and R_.dim == R_.edim:
        Vs = R_.S[0]
        res = []
        for i in range(len(Vs)):
            fc = [Vs[j] for j in range(len(Vs)) if j != i]
            M = sp.Matrix([vsub(q, fc[0]) for q in fc[1:]]) if len(fc) > 1 else sp.zeros(0, R_.edim)
            ns = M.nullspace() if M.shape[0] else [sp.Matrix([1])]
            a = [simp(c) for c in ns[0]]
            b = dot(a, fc[0])
            if sgn(dot(a, Vs[i]) - b) > 0:
                a, b = [-c for c in a], -b
            res.append((a, simp(b)))
        return res
    if isinstance(R_, Polygon2R) and R_.convex:
        ring = G._orient_ring(R_.C[0][0], True)
        res = []
        for i in range(len(ring)):
            p, q = ring[i], ring[(i + 1) % len(ring)]
            a = [S(q[1]) - S(p[1]), S(p[0]) - S(q[0])]
            res.append((a, simp(dot(a, p))))
        return res
    if isinstance(R_, PolyhedronR) and R_.convex:
        res = []
        for T in R_.tris:
            a = _cross3(vsub(T[1], T[0]), vsub(T[2], T[0]))
            res.append((a, simp(dot(a, T[0]))))
        return res
    if isinstance(R_, ParallelotopeR) and R_.dim == R_.edim:
        Ai = R_.A.inv()
        res = []
        for i in range(R_.edim):
            a = [simp(c) for c in Ai.row(i)]
            t0 = dot(a, R_.p)
            res += [(a, simp(1 + t0)), ([-c for c in a], simp(-t0))]
        return res
    if isinstance(R_, FlatR) and R_.halfspace is not None:
        return [(R_.n, R_.c)]
    return None


def _ball_of(R_):
    """(centre, radius) when R_ is a solid round ball (Disk / Ball / zero-inner shell), else None."""
    if isinstance(R_, EllipsoidR) and R_.r is not None:
        return R_.c, R_.r
    if isinstance(R_, ShellR) and eqz(R_.r1):
        return R_.c, R_.r2
    return None


VERTS_IN = (PointsR, SimplexR, Polygon2R, Polygon3R, PolyhedronR, BoxR, ParallelotopeR)


def witnesses(R_):
    """Points known to lie in R_."""
    if isinstance(R_, VERTS_IN):
        return R_.verts() or []
    if isinstance(R_, (EllipsoidR, ShellR)) and getattr(R_, "r1", 0) == 0:
        return [R_.c]
    if isinstance(R_, SphereR):
        q = list(R_.c)
        q[0] = q[0] + R_.r
        return [q]
    if isinstance(R_, ShellR):
        q = list(R_.c)
        q[0] = q[0] + R_.r2
        return [q]
    if isinstance(R_, FlatR):
        if R_.halfspace is not None:
            n_, c = R_.n, R_.c
            return [vmul(c / n2(n_), n_)]
        return [R_.p]
    if isinstance(R_, RevolutionR):
        return [R_.p1, R_.p2] if R_.kind != "Cone" else [R_.p1, R_.p2]
    return []


def within(A, B):
    """True if A is a subset of B, False if not, None if undecided."""
    if isinstance(A, EmptyR) or isinstance(B, FullR):
        return True
    if isinstance(B, EmptyR):
        return isinstance(A, EmptyR)
    if A.edim != B.edim:
        return False
    for w in witnesses(A):
        try:
            if not B.member(w):
                return False
        except NotImplementedError:
            pass
    bA, bB = _ball_of(A), _ball_of(B)
    if isinstance(A, SphereR):
        bA = (A.c, A.r)
    if bA is not None and bB is not None:
        return le(norm(vsub(bA[0], bB[0])) + bA[1], bB[1])
    H = hrep(B)
    if bA is not None and H is not None:
        return all(le(dot(a, bA[0]) + bA[1] * norm(a), b) for a, b in H)
    if bA is not None and B.convex:
        if isinstance(A, SphereR) or _ball_of(A) is not None:
            try:
                return le(B.sdist(bA[0]), -bA[1]) if B.dim == B.edim else False
            except NotImplementedError:
                return None
    if isinstance(A, VERTS_IN) and B.convex:
        V = A.verts()
        try:
            return all(B.member(v) for v in V)
        except NotImplementedError:
            return None
    if bB is not None and isinstance(A, (RevolutionR,)):
        return None
    return None


def disjoint(A, B):
    """True if A and B are disjoint, False if they meet, None if undecided."""
    if isinstance(A, EmptyR) or isinstance(B, EmptyR):
        return True
    if A.edim != B.edim:
        raise ValueError("regions of different embedding dimension")
    for X_, Y_ in ((A, B), (B, A)):
        for w in witnesses(X_):
            try:
                if Y_.member(w):
                    return False
            except NotImplementedError:
                pass
    for X_, Y_ in ((A, B), (B, A)):
        if isinstance(X_, PointsR):
            try:
                return not any(Y_.member(p) for p in X_.P)
            except NotImplementedError:
                return None
    HA, HB = hrep(A), hrep(B)
    if HA is not None and HB is not None and A.edim <= 3:
        I_ = _halfspace_polytope(HA + HB, A.edim, "Region")
        if I_ is not None:
            return isinstance(I_, EmptyR)
    for X_, Y_ in ((A, B), (B, A)):
        b = _ball_of(X_)
        if b is not None:
            try:
                return lt(b[1], Y_.dist(b[0]))
            except NotImplementedError:
                pass
        if isinstance(X_, SphereR):
            if isinstance(Y_, SphereR):
                d = norm(vsub(X_.c, Y_.c))
                return lt(X_.r + Y_.r, d) or lt(d, sp.Abs(X_.r - Y_.r))
            yb = _ball_of(Y_)
            if yb is not None:
                d = norm(vsub(X_.c, yb[0]))
                return lt(X_.r + yb[1], d) or lt(d + yb[1], X_.r)
    return None


# ───────────────────────────── composite regions ─────────────────────────────

class BoolR(Region):
    """RegionUnion / RegionIntersection / RegionDifference / RegionSymmetricDifference / BooleanRegion."""

    def __init__(self, op, parts, expr=None, f="Region", bfn=None):
        self.f = f
        self.op = op
        self.kind = {"Union": "RegionUnion", "Intersection": "RegionIntersection", "Difference": "RegionDifference",
                     "SymmetricDifference": "RegionSymmetricDifference", "Boolean": "BooleanRegion"}[op]
        self.parts = parts
        self.expr = expr
        self.bfn = bfn
        eds = {p.edim for p in parts}
        if len(eds) > 1:
            raise ValueError(f"{f}: all regions must have the same embedding dimension")
        self.edim = eds.pop()
        self.res = self._resolve()
        if self.res is not None:
            r = self.res
            self.dim, self.bounded, self.convex = r.dim, r.bounded, r.convex
        else:
            self._attrs()

    def _attrs(self):
        P = self.parts
        if self.op == "Union":
            ds = [p.dim for p in P if not isinstance(p, EmptyR)]
            self.dim = None if any(d is None for d in ds) else (max(ds) if ds else -sp.oo)
            self.bounded = all(p.bounded for p in P)
            self.convex = None
        elif self.op == "Difference":
            self.dim, self.bounded, self.convex = None, True if P[0].bounded else None, None
        elif self.op == "Intersection":
            self.dim, self.convex = None, True if all(p.convex for p in P) else None
            self.bounded = True if any(p.bounded for p in P) else None
        else:
            self.dim, self.bounded, self.convex = None, True if all(p.bounded for p in P) else None, None

    def _resolve(self):
        """An equivalent simpler region when containment / disjointness can be proved, else None."""
        P = [p for p in self.parts]
        op = self.op
        try:
            if op == "Union":
                P = [p for p in P if not isinstance(p, EmptyR)]
                keep = []
                for i, p in enumerate(P):
                    if any(j != i and within(p, q) and (j < i or not within(q, p)) for j, q in enumerate(P)):
                        continue
                    keep.append(p)
                if not keep:
                    return EmptyR(self.edim)
                if len(keep) == 1:
                    return keep[0]
                self.disjoint_parts = all(disjoint(a, b) is True for a, b in itertools.combinations(keep, 2))
                self.parts_ = keep
                return None
            if op == "Intersection":
                if any(isinstance(p, EmptyR) for p in P):
                    return EmptyR(self.edim)
                for a, b in itertools.combinations(P, 2):
                    if disjoint(a, b) is True:
                        return EmptyR(self.edim)
                for p in P:
                    if all(q is p or within(p, q) for q in P):
                        return p
                H = [hrep(p) for p in P]
                if all(h is not None for h in H) and self.edim <= 3:
                    r = _halfspace_polytope([c for h in H for c in h], self.edim, self.f)
                    if r is not None:
                        return r
                return None
            if op == "Difference":
                A, rest = P[0], P[1:]
                cur = A
                for B in rest:
                    if within(cur, B):
                        return EmptyR(self.edim)
                    if disjoint(cur, B) is True:
                        continue
                    return None if cur is A else None
                return cur
            if op == "SymmetricDifference" and len(P) == 2:
                a, b = P
                if disjoint(a, b) is True:
                    self.op_eff = "DisjointUnion"
                return None
        except (NotImplementedError, ValueError):
            return None
        return None

    def member(self, p):
        if self.res is not None:
            return self.res.member(p)
        m = [q.member(p) for q in self.parts]
        if self.op == "Union":
            return any(m)
        if self.op == "Intersection":
            return all(m)
        if self.op == "Difference":
            return m[0] and not any(m[1:])
        if self.op == "SymmetricDifference":
            return sum(m) % 2 == 1
        return _apply_bool(self.bfn, m, self.f)

    def _disjoint_union_parts(self):
        if self.op == "Union" and getattr(self, "disjoint_parts", False):
            return self.parts_
        if self.op == "SymmetricDifference" and getattr(self, "op_eff", None) == "DisjointUnion":
            return self.parts
        return None

    def integrate(self, g, Xs):
        if self.res is not None:
            return self.res.integrate(g, Xs)
        D = self._disjoint_union_parts()
        if D is not None:
            top = max(p.dim for p in D)
            return sum((p.integrate(g, Xs) for p in D if p.dim == top), sp.Integer(0))
        if self.op == "Difference" and len(self.parts) == 2:
            A, B = self.parts
            if within(B, A) and B.dim == A.dim:
                return A.integrate(g, Xs) - B.integrate(g, Xs)
            if B.dim is not None and A.dim is not None and B.dim < A.dim:
                return A.integrate(g, Xs)
        ie = self._incl_excl()
        if ie is not None:
            A, B, I, ca, cb, ci = ie
            tot = ca * A.integrate(g, Xs) + cb * B.integrate(g, Xs)
            if not isinstance(I, EmptyR) and I.dim == A.dim:
                tot += ci * I.integrate(g, Xs)
            return tot
        self.need("integration (overlapping parts that cannot be resolved exactly)")

    def _incl_excl(self):
        """Two full-dimensional parts whose intersection resolves exactly: (A, B, A n B, cA, cB, cI) with
        int over self = cA int A + cB int B + cI int (A n B) (inclusion-exclusion), else None."""
        P = getattr(self, "parts_", self.parts) if self.op == "Union" else self.parts
        if len(P) != 2 or self.op not in ("Union", "Difference", "SymmetricDifference"):
            return None
        A, B = P
        if A.dim is None or A.dim != B.dim or A.dim != self.edim:
            return None
        try:
            I = BoolR("Intersection", [A, B], f=self.f).res
        except (NotImplementedError, ValueError):
            return None
        if I is None or (not isinstance(I, EmptyR) and I.dim is None):
            return None
        return {"Union": (A, B, I, 1, 1, -1), "Difference": (A, B, I, 1, 0, -1),
                "SymmetricDifference": (A, B, I, 1, 1, -2)}[self.op]

    def _dims(self):
        if self.res is not None:
            return self.res.dim
        D = self._disjoint_union_parts()
        if D is not None:
            return max(p.dim for p in D)
        if self.op == "Difference" and len(self.parts) == 2:
            A, B = self.parts
            if within(B, A) and B.dim == A.dim and lt(B.measure(), A.measure()):
                return A.dim
            if B.dim is not None and A.dim is not None and B.dim < A.dim:
                return A.dim
        ie = self._incl_excl()
        if ie is not None:
            A, B, I, ca, cb, ci = ie
            m = ca * A.measure() + cb * B.measure() + (ci * I.measure() if not isinstance(I, EmptyR)
                                                         and I.dim == A.dim else 0)
            if lt(0, m):
                return A.dim
        return None

    def bounds(self):
        if self.res is not None:
            return self.res.bounds()
        if self.op in ("Union",) or self._disjoint_union_parts() is not None:
            bs = [p.bounds() for p in self.parts if not isinstance(p, EmptyR)]
            return [(smin([b[i][0] for b in bs]), smax([b[i][1] for b in bs])) for i in range(self.edim)]
        self.need("the bounds")

    def nearest(self, p):
        if self.res is not None:
            return self.res.nearest(p)
        if self.op == "Union" or self._disjoint_union_parts() is not None:
            return best_of([q.nearest(p) for q in self.parts if not isinstance(q, EmptyR)])
        self.need("the distance")

    def sdist(self, p):
        if self.res is not None:
            return self.res.sdist(p)
        self.need("the signed distance")

    def farthest(self, p):
        if self.res is not None:
            return self.res.farthest(p)
        if self.op == "Union":
            return smax([q.farthest(p) for q in self.parts if not isinstance(q, EmptyR)])
        self.need("the farthest distance")

    def verts(self):
        if self.res is not None:
            return self.res.verts()
        if self.op == "Union":
            V = []
            for q in self.parts:
                v = q.verts()
                if v is None:
                    return None
                V += v
            return V
        return None

    def boundary(self):
        if self.res is not None:
            return self.res.boundary() if self.res.dim == self.edim else self.res.expr
        D = self._disjoint_union_parts()
        if D is not None and all(p.dim == self.edim for p in D):
            return Expr("RegionUnion", *[p.boundary() for p in D])
        self.need("the boundary")


def _apply_bool(fexpr, vals, f):
    from ..evaluate import evaluate as _ev
    r = _ev(Expr(K.as_head(fexpr), *[bool(v) for v in vals]), 1)
    if r is True or r is False:
        return r
    raise ValueError(f"{f}: the Boolean function did not give True or False")


class AffineImageR(Region):
    """x -> A x + b (A invertible) applied to a region without a closed-form image."""

    def __init__(self, base, A, b, expr=None, f="Region"):
        self.f = f
        self.base, self.A, self.b = base, A, [S(c) for c in b]
        self.edim = base.edim
        self.dim = base.dim
        self.det = simp(A.det())
        if eqz(self.det):
            raise NotImplementedError(f"{f}: singular (degenerate) transformations of {base.kind} regions are "
                                      f"not supported")
        self.Ai = A.inv()
        self.convex, self.bounded = base.convex, base.bounded
        self.expr = expr
        AtA = (A.T * A).applyfunc(simp)
        s2 = AtA[0, 0]
        self.scale = simp(sp.sqrt(s2)) if all(eqz(AtA[i, j] - (s2 if i == j else 0))
                                             for i in range(self.edim) for j in range(self.edim)) else None
        self.kind = base.kind

    def fwd(self, x):
        return [simp(c) for c in (self.A * sp.Matrix(x) + sp.Matrix(self.b))]

    def inv(self, y):
        return [simp(c) for c in (self.Ai * (sp.Matrix(y) - sp.Matrix(self.b)))]

    def member(self, p):
        return self.base.member(self.inv(p))

    def integrate(self, g, Xs):
        if self.base.dim != self.edim:
            self.need("integration over lower-dimensional transformed regions")
        Y = self.fwd(Xs)
        return sp.Abs(self.det) * self.base.integrate(subst(g, Xs, Y), Xs)

    def nearest(self, p):
        if self.scale is None:
            self.need("the distance (non-similarity transformations)")
        d2, q = self.base.nearest(self.inv(p))
        return simp(d2 * self.scale ** 2), self.fwd(q)

    def sdist(self, p):
        if self.scale is None:
            self.need("the signed distance (non-similarity transformations)")
        return simp(self.base.sdist(self.inv(p)) * self.scale)

    def verts(self):
        V = self.base.verts()
        return None if V is None else [self.fwd(v) for v in V]

    def bounds(self):
        V = self.verts()
        if V is None:
            self.need("the bounds")
        return [(smin([q[i] for q in V]), smax([q[i] for q in V])) for i in range(self.edim)]

    def farthest(self, p):
        if self.scale is not None:
            return simp(self.base.farthest(self.inv(p)) * self.scale)
        return Region.farthest(self, p)


def affine_map(R_, A, b, f="Region"):
    """Exact image of a region under x -> A x + b (A square)."""
    n = R_.edim
    if A.shape != (n, n):
        raise ValueError(f"{f}: the transformation acts on R^{A.shape[0]}, the region lies in R^{n}")
    b = [S(c) for c in b]

    def F(x):
        return [simp(c) for c in (A * sp.Matrix(x) + sp.Matrix(b))]
    det = simp(A.det())
    if isinstance(R_, EmptyR):
        return R_
    if isinstance(R_, PointsR):
        return PointsR([F(p) for p in R_.P])
    if isinstance(R_, SimplexR):
        return SimplexR([[F(p) for p in s] for s in R_.S], f=f)
    if isinstance(R_, Polygon2R) and not eqz(det):
        return Polygon2R([([F(p) for p in o], [[F(p) for p in h] for h in hs]) for o, hs in R_.C], f=f)
    if isinstance(R_, Polygon3R) and not eqz(det):
        return Polygon3R([([F(p) for p in o], [[F(p) for p in h] for h in hs]) for o, hs in R_.C], f=f)
    if isinstance(R_, PolyhedronR) and not eqz(det) and not R_.voids:
        return PolyhedronR([F(p) for p in R_.V], R_.F, f=f)
    if isinstance(R_, BoxR):
        if A.is_diagonal() and all(sgn(A[i, i]) != 0 for i in range(n)):
            lo, hi = F(R_.lo), F(R_.hi)
            return BoxR([smin([u, v]) for u, v in zip(lo, hi)], [smax([u, v]) for u, v in zip(lo, hi)], f=f)
        if not eqz(det) and R_.dim == n:
            Vs = [[simp(A[i, j] * (R_.hi[j] - R_.lo[j])) for i in range(n)] for j in range(n)]
            return ParallelotopeR(F(R_.lo), Vs, f=f)
    if isinstance(R_, ParallelotopeR) and not eqz(det):
        return ParallelotopeR(F(R_.p), [[simp(c) for c in A * sp.Matrix(v)] for v in R_.V], f=f)
    if isinstance(R_, EllipsoidR) and not eqz(det):
        Lm = (A * R_.Lm).applyfunc(simp)
        Sg = (Lm * Lm.T).applyfunc(simp)
        rr = None
        if Sg.is_diagonal() and all(eqz(Sg[i, i] - Sg[0, 0]) for i in range(n)):
            rr = simp(sp.sqrt(Sg[0, 0]))
            Lm = rr * sp.eye(n)
        elif Sg.is_diagonal():
            Lm = sp.diag(*[simp(sp.sqrt(Sg[i, i])) for i in range(n)])
        return EllipsoidR(F(R_.c), Lm, f=f, round_r=rr)
    if isinstance(R_, FlatR) and not eqz(det):
        if R_.halfspace is not None:
            Ait = A.inv().T
            nn = [simp(c) for c in Ait * sp.Matrix(R_.n)]
            return FlatR(None, None, f=f, halfspace=(nn, simp(R_.c + dot(nn, b))))
        lin = (lambda v: [simp(c) for c in A * sp.Matrix(v)])
        return FlatR(F(R_.p), [lin(v) for v in R_.V], w=lin(R_.w) if R_.w is not None else None, f=f)
    if isinstance(R_, UnionCellsR) and not eqz(det):
        return UnionCellsR([affine_map(c, A, b, f) for c in R_.cells], n, f=f)
    if isinstance(R_, (SphereR, ShellR)):
        AtA = (A.T * A).applyfunc(simp)
        s2 = AtA[0, 0]
        if not eqz(s2) and all(eqz(AtA[i, j] - (s2 if i == j else 0)) for i in range(n) for j in range(n)):
            s_ = simp(sp.sqrt(s2))
            if isinstance(R_, SphereR):
                return SphereR(F(R_.c), simp(s_ * R_.r), f=f)
            return ShellR(F(R_.c), simp(s_ * R_.r1), simp(s_ * R_.r2), f=f)
    return AffineImageR(R_, A, b, f=f)


class FnImageR(Region):
    """TransformedRegion with a non-affine map (inert) or InverseTransformedRegion (membership by f(p) in reg)."""

    def __init__(self, base, fexpr, inverse, edim, expr=None, f="Region"):
        self.f = f
        self.base, self.fexpr, self.inverse = base, fexpr, inverse
        self.edim = edim
        self.dim = None
        self.bounded = None
        self.convex = None
        self.expr = expr
        self.kind = "InverseTransformedRegion" if inverse else "TransformedRegion"

    def member(self, p):
        if not self.inverse:
            self.need("membership (non-invertible maps)")
        from ..evaluate import evaluate as _ev
        y = _ev(Expr(K.as_head(self.fexpr), out(p)), 1)
        if not K.is_list(y) or len(y.args) != self.base.edim:
            raise ValueError(f"{self.f}: the map must give points of dimension {self.base.edim}")
        q = [sx(c, self.f, "image coordinate") for c in y.args]
        concrete(q)
        return self.base.member(q)


class ProductR(Region):
    kind = "RegionProduct"

    def __init__(self, parts, expr=None, f="Region"):
        self.f = f
        self.parts = parts
        self.edim = sum(p.edim for p in parts)
        self.dim = sum(p.dim for p in parts) if all(isinstance(p.dim, int) for p in parts) else None
        if any(isinstance(p, EmptyR) for p in parts):
            self.dim = -sp.oo
        self.bounded = all(p.bounded for p in parts)
        self.convex = True if all(p.convex for p in parts) else None
        self.expr = expr

    def split(self, x):
        res, i = [], 0
        for p in self.parts:
            res.append(list(x[i:i + p.edim]))
            i += p.edim
        return res

    def member(self, x):
        return all(p.member(c) for p, c in zip(self.parts, self.split(x)))

    def integrate(self, g, Xs):
        groups = self.split(Xs)
        tot = sp.Integer(0)
        for co, e in poly_terms(g, list(Xs)):
            term = co
            i = 0
            for p, gx in zip(self.parts, groups):
                mono = sp.Integer(1)
                for k in range(p.edim):
                    mono *= Xs[i + k] ** e[i + k]
                Xp = X(p.edim)
                term *= p.integrate(subst(mono, Xs[i:i + p.edim], Xp), Xp)
                i += p.edim
            tot += term
        return tot

    def bounds(self):
        return [b for p in self.parts for b in p.bounds()]

    def nearest(self, x):
        d2, q = sp.Integer(0), []
        for p, c in zip(self.parts, self.split(x)):
            a, b = p.nearest(c)
            d2 += a
            q += list(b)
        return simp(d2), q

    def farthest(self, x):
        return simp(sp.sqrt(sum((p.farthest(c) ** 2 for p, c in zip(self.parts, self.split(x))), sp.Integer(0))))

    def verts(self):
        Vs = [p.verts() for p in self.parts]
        if any(v is None for v in Vs):
            return None
        return [[c for v in combo for c in v] for combo in itertools.product(*Vs)]


_r5 = sp.sqrt(5)
# edge-1 regular solids: volume, surface area, polar moment Integral |x - c|^2 (independently cross-checked against
# the exact polyhedron integration engine and the textbook inertia I = k m a^2, k = 1/20, 1/6, 1/10, (95+39 Sqrt5)/300,
# (3+Sqrt5)/20 for the tetrahedron, cube, octahedron, dodecahedron, icosahedron)
PLATONIC_DATA = {
    "Tetrahedron": (sp.sqrt(2) / 12, sp.sqrt(3), sp.sqrt(2) / 160),
    "Cube": (sp.Integer(1), sp.Integer(6), sp.Rational(1, 4)),
    "Octahedron": (sp.sqrt(2) / 3, 2 * sp.sqrt(3), sp.sqrt(2) / 20),
    "Dodecahedron": (sp.Rational(15, 4) + 7 * _r5 / 4, 3 * sp.sqrt(25 + 10 * _r5), sp.Rational(279, 80) + 25 * _r5 / 16),
    "Icosahedron": (sp.Rational(5, 4) + 5 * _r5 / 12, 5 * sp.sqrt(3), sp.Rational(7, 16) + 3 * _r5 / 16),
}


class OrientFreeR(Region):
    """A regular solid whose default orientation is not Wolfram's: only orientation-free quantities (measure,
    surface area, centroid, polynomial integrals of degree <= 2: their second-moment tensor is isotropic) are
    exact; they come from closed forms."""

    def __init__(self, head, c, l_, f):
        self.f, self.kind = f, head
        self.c, self.l = [S(v) for v in c], S(l_)
        self.dim, self.edim, self.convex, self.expr = 3, 3, True, None
        V1, A1, J1 = PLATONIC_DATA[head]
        self.V, self.A, self.J = simp(V1 * self.l ** 3), simp(A1 * self.l ** 2), simp(J1 * self.l ** 5)

    def measure(self):
        return self.V

    def integrate(self, g, Xs):
        terms = poly_terms(g, Xs)
        if any(sum(e) > 2 for _, e in terms):
            self.need_orient()
        g = S(g)
        lap = sum((sp.diff(g, x, 2) for x in Xs), sp.Integer(0))
        return simp(self.V * subst(g, Xs, self.c) + subst(lap, Xs, self.c) * self.J / 6)

    def need_orient(self, *a):
        raise NotImplementedError(f"{self.f}: this needs the coordinates of {self.kind}[...] in Wolfram's default "
                                  f"orientation, which are not available; give the polyhedron explicitly")
    member = bounds = nearest = sdist = boundary = farthest = need_orient

    def dist(self, p):
        self.need_orient()

    def verts(self):
        return None


# ───────────────────────────── the dispatcher ─────────────────────────────

CONVERTERS = {}          # head -> fn(expr, f) -> Region (filled by wl_regions_mesh)

FORMULA_HEADS = ("ImplicitRegion", "ParametricRegion")
BOOL_HEADS = {"RegionUnion": "Union", "RegionIntersection": "Intersection", "RegionDifference": "Difference",
              "RegionSymmetricDifference": "SymmetricDifference"}
UNSUPPORTED_PRIMS = ("BezierCurve", "BSplineCurve", "BSplineSurface", "BezierSurface", "FilledCurve",
                     "JoinedCurve", "Tube", "Arrow", "PolarCurve", "FilledPolarCurve", "UniformPolyhedron",
                     "ConicHullRegion")
REGION_HEADS = set(G._CHECKS) | set(FORMULA_HEADS) | set(BOOL_HEADS) | {
    "Region", "BooleanRegion", "TransformedRegion", "InverseTransformedRegion", "RegionProduct", "CSGRegion",
    "MeshRegion", "BoundaryMeshRegion"}


def _cc(*v):
    concrete(*v)
    return v


def _radius_spec(r, f, n):
    if K.is_list(r):
        rs = [sx(c, f, "radius") for c in r.args]
        if len(rs) != n:
            raise ValueError(f"{f}: give {n} radii")
        return rs
    return sx(r, f, "radius")


def _lines(a0, f):
    if K.is_list(a0) and a0.args and _is_coord_list(a0.args[0]):
        return [pts(c, f, lo=1) for c in a0.args]
    return [pts(a0, f, lo=1)]


def _segments(chains):
    segs = []
    for P in chains:
        for i in range(len(P) - 1):
            if not is_zero_vec(vsub(P[i], P[i + 1])):
                segs.append([P[i], P[i + 1]])
    return segs


def _flat_from(h, a, f):
    if h == "FullRegion":
        n = int(a[0])
        return FullR(n, f=f)
    if h == "EmptyRegion":
        return EmptyR(int(a[0]))
    if h in ("InfiniteLine", "HalfLine"):
        if len(a) == 1:
            P = pts(a[0], f, n=2)
            p, v = P[0], vsub(P[1], P[0])
        else:
            p, v = pt(a[0], f), pt(a[1], f, "direction")
        _cc(p, v)
        if is_zero_vec(v):
            raise ValueError(f"{f}: the direction of {h} must be non-zero")
        return FlatR(p, [] if h == "HalfLine" else [v], w=v if h == "HalfLine" else None, f=f)
    if h == "InfinitePlane":
        if len(a) == 1:
            P = pts(a[0], f, n=3)
            p, V = P[0], [vsub(P[1], P[0]), vsub(P[2], P[0])]
        else:
            p, V = pt(a[0], f), pts(a[1], f, n=2)
        _cc(p, V)
        return FlatR(p, V, f=f)
    if h == "AffineSpace":
        if len(a) == 1:
            P = pts(a[0], f)
            p, V = P[0], [vsub(q, P[0]) for q in P[1:]]
        else:
            p, V = pt(a[0], f), pts(a[1], f)
        _cc(p, V)
        return FlatR(p, V, f=f)
    if h in ("HalfPlane", "AffineHalfSpace"):
        if len(a) == 2:
            P = pts(a[0], f, n=2 if h == "HalfPlane" else None)
            p, V, w = P[0], [vsub(q, P[0]) for q in P[1:]], pt(a[1], f, "direction")
        else:
            p, V, w = pt(a[0], f), (pts(a[1], f) if h == "AffineHalfSpace" else [pt(a[1], f, "direction")]), \
                pt(a[2], f, "direction")
        _cc(p, V, w)
        n = len(p)
        if V:
            Am = sp.Matrix(V).T
            wp = list(sp.Matrix(w) - Am * (Am.T * Am).LUsolve(Am.T * sp.Matrix(w)))
        else:
            wp = list(w)
        wp = [simp(c) for c in wp]
        if is_zero_vec(wp):
            raise ValueError(f"{f}: the direction w must not lie in the span of the boundary")
        if len(V) + 1 == n:
            nrm = sp.Matrix(V).nullspace()[0] if V else sp.Matrix([1])
            nrm = [simp(c) for c in nrm]
            if sgn(dot(nrm, wp)) > 0:
                nrm = [-c for c in nrm]
            return FlatR(None, None, f=f, halfspace=(nrm, simp(dot(nrm, p))))
        return FlatR(p, V, w=wp, f=f)
    if h in ("HalfSpace", "Hyperplane"):
        nrm = pt(a[0], f, "normal")
        if len(a) > 1 and K.is_list(a[1]):
            q = pt(a[1], f, "point", dim=len(nrm))
            _cc(nrm, q)
            c = dot(nrm, q)
        else:
            c = sx(a[1], f, "offset") if len(a) > 1 else sp.Integer(0)
            _cc(nrm, c)
        if is_zero_vec(nrm):
            raise ValueError(f"{f}: the normal vector must be non-zero")
        if h == "HalfSpace":
            return FlatR(None, None, f=f, halfspace=(nrm, simp(c)))
        p0 = [simp(c * v / n2(nrm)) for v in nrm]
        V = [[simp(x) for x in v] for v in sp.Matrix([nrm]).nullspace()]
        return FlatR(p0, V, f=f)
    raise NotImplementedError(f"{f}: {h} regions are not supported")


def _prim(x, f):
    h = x.head
    G._check_args(h, list(x.args), allow_index=False)
    a, _opts = split_trailing_options(list(x.args))
    Z = sp.Integer(0)
    if h == "Point":
        P = pts(a[0], f) if _is_coord_list(a[0]) else [pt(a[0], f)]
        _cc(P)
        return PointsR(P, expr=x)
    if h == "Line":
        chains = _lines(a[0], f)
        _cc(chains)
        segs = _segments(chains)
        if not segs:
            return PointsR([p for c in chains for p in c], expr=x)
        bf = None
        if len(chains) == 1 and len(chains[0][0]) == 1:
            ends = [min(chains[0], key=lambda q: num(q[0])), max(chains[0], key=lambda q: num(q[0]))]
            bf = (lambda: Expr("Point", out(ends, x)))
        return SimplexR(segs, expr=x, boundary_fn=bf, f=f)
    if h in ("Triangle", "Simplex"):
        if h == "Triangle":
            if a and _is_coord_list(a[0]) and _is_coord_list(a[0].args[0]):
                Ss = [pts(t, f, n=3) for t in a[0].args]
            else:
                Ss = [pts(a[0], f, n=3)] if a else [[[Z, Z], [sp.Integer(1), Z], [Z, sp.Integer(1)]]]
        elif K.is_int(a[0]) and not isinstance(a[0], bool):
            n = int(a[0])
            Ss = [[[Z] * n] + [[sp.Integer(int(i == j)) for j in range(n)] for i in range(n)]]
        elif _is_coord_list(a[0]) and _is_coord_list(a[0].args[0]):
            Ss = [pts(t, f) for t in a[0].args]
        else:
            Ss = [pts(a[0], f)]
        _cc(Ss)
        if all(len(s) == 1 for s in Ss):
            return PointsR([s[0] for s in Ss], expr=x)
        return SimplexR(Ss, expr=x, f=f)
    if h == "Rectangle":
        p = pt(a[0], f, "corner", dim=2) if a else [Z, Z]
        q = pt(a[1], f, "corner", dim=2) if len(a) > 1 else [c + 1 for c in p]
        _cc(p, q)
        return BoxR(p, q, expr=x, f=f)
    if h == "Cuboid":
        p = pt(a[0], f, "corner") if a else [Z, Z, Z]
        q = pt(a[1], f, "corner", dim=len(p)) if len(a) > 1 else [c + 1 for c in p]
        _cc(p, q)
        return BoxR(p, q, expr=x, f=f)
    if h in ("Polygon", "RegularPolygon", "Parallelogram"):
        comps = G.polygon_components(x, f)
        _cc([[o] + hs for o, hs in comps])
        d = len(comps[0][0][0])
        if d == 2:
            return Polygon2R(comps, expr=x, f=f)
        if d == 3:
            try:
                return Polygon3R(comps, expr=x, f=f)
            except ValueError:
                if len(comps) == 1:
                    raise
                return UnionCellsR([Polygon3R([c], f=f) for c in comps], 3, expr=x, f=f)
        raise NotImplementedError(f"{f}: polygons in {d} dimensions are not supported")
    if h == "Parallelepiped":
        p, V = pt(a[0], f, "origin"), pts(a[1], f, "vectors")
        _cc(p, V)
        return ParallelotopeR(p, V, expr=x, f=f)
    if h in ("Disk", "Circle", "DiskSegment"):
        if a and _is_coord_list(a[0]) and _is_coord_list(a[0].args[0]):
            raise NotImplementedError(f"{f}: collections of {h} objects are not supported; use RegionUnion")
        c = pt(a[0], f, "centre", dim=2) if a else [Z, Z]
        r = _radius_spec(a[1], f, 2) if len(a) > 1 else sp.Integer(1)
        th = None
        if len(a) > 2:
            th = [sx(t, f, "angle") for t in a[2].args]
        _cc(c, r, th or [])
        if isinstance(r, list):
            if eqz(r[0] - r[1]):
                r = r[0]
            elif th is not None:
                raise NotImplementedError(f"{f}: elliptic sectors, arcs and segments are not supported")
            elif h == "Circle":
                return EllipseCurveR(c, r[0], r[1], expr=x, f=f)
            else:
                return EllipsoidR(c, sp.diag(*r), expr=x, f=f)
        if th is not None:
            return SectorR(c, Z, r, th[0], th[1], curve=h == "Circle", segment=h == "DiskSegment", expr=x, f=f)
        if h == "Circle":
            return SphereR(c, r, expr=x, f=f)
        return EllipsoidR(c, r * sp.eye(2), expr=x, f=f, round_r=r)
    if h in ("Ball", "Sphere"):
        if a and K.is_int(a[0]) and not isinstance(a[0], bool):
            c, r = [Z] * int(a[0]), sp.Integer(1)
        else:
            if a and _is_coord_list(a[0]) and _is_coord_list(a[0].args[0]):
                raise NotImplementedError(f"{f}: collections of {h} objects are not supported; use RegionUnion")
            c = pt(a[0], f, "centre") if a else [Z, Z, Z]
            r = sx(a[1], f, "radius") if len(a) > 1 else sp.Integer(1)
        _cc(c, r)
        if h == "Sphere":
            return SphereR(c, r, expr=x, f=f)
        if eqz(r):
            return PointsR([c], expr=x)
        return EllipsoidR(c, r * sp.eye(len(c)), expr=x, f=f, round_r=r)
    if h == "Ellipsoid":
        c = pt(a[0], f, "centre")
        s_ = a[1]
        if K.is_list(s_.args[0]):
            M = sp.Matrix([[sx(v, f) for v in row.args] for row in s_.args])
            _cc(c, list(M))
            Lm = sp.diag(*[simp(sp.sqrt(M[i, i])) for i in range(M.shape[0])]) if M.is_diagonal() else \
                M.cholesky(hermitian=False)
        else:
            rs = [sx(v, f, "semi-axis") for v in s_.args]
            _cc(c, rs)
            Lm = sp.diag(*rs)
        rr = Lm[0, 0] if Lm.is_diagonal() and all(eqz(Lm[i, i] - Lm[0, 0]) for i in range(Lm.shape[0])) else None
        return EllipsoidR(c, Lm, expr=x, f=f, round_r=rr)
    if h in ("Annulus", "SphericalShell"):
        d = 2 if h == "Annulus" else None
        th = None
        if not a:
            c, (r1, r2) = None, (sp.Rational(1, 2), sp.Integer(1))
        elif len(a) == 1:
            c = None
            if K.is_list(a[0]):
                r1, r2 = [sx(t, f, "radius") for t in a[0].args]
            else:
                r2 = sx(a[0], f, "radius")
                r1 = r2 / 2
        else:
            c = pt(a[0], f, "centre", dim=d)
            r1, r2 = [sx(t, f, "radius") for t in a[1].args]
            if len(a) > 2:
                th = [sx(t, f, "angle") for t in a[2].args]
        if c is None:
            c = [Z] * (2 if h == "Annulus" else 3)
        _cc(c, r1, r2, th or [])
        if th is not None:
            return SectorR(c, r1, r2, th[0], th[1], expr=x, f=f)
        return ShellR(c, r1, r2, expr=x, f=f)
    if h in ("Cylinder", "Cone", "CapsuleShape", "StadiumShape"):
        d = 2 if h == "StadiumShape" else 3
        if h in ("Cylinder", "Cone"):
            dflt = [[Z, Z, -1], [Z, Z, sp.Integer(1)]]
        elif h == "CapsuleShape":
            dflt = [[-1, Z, Z], [sp.Integer(1), Z, Z]]
        else:
            dflt = [[-1, Z], [sp.Integer(1), Z]]
        r = sp.Integer(1)
        if len(a) == 1 and not K.is_list(a[0]):
            P, r = dflt, sx(a[0], f, "radius")
        else:
            if a and _is_coord_list(a[0]) and _is_coord_list(a[0].args[0]):
                raise NotImplementedError(f"{f}: collections of {h} objects are not supported; use RegionUnion")
            P = pts(a[0], f, n=2) if a else dflt
            if len(a) > 1:
                r = sx(a[1], f, "radius")
        _cc(P, r)
        P = [[S(c) for c in p] for p in P]
        if len(P[0]) != d:
            raise NotImplementedError(f"{f}: {h} regions are supported in {d} dimensions only")
        return RevolutionR(h, P[0], P[1], r, expr=x, f=f)
    if h in ("Torus", "FilledTorus"):
        c = pt(a[0], f, "centre", dim=3) if a else [Z, Z, Z]
        r1, r2 = [sx(t, f, "radius") for t in a[1].args] if len(a) > 1 else (sp.Rational(1, 2), sp.Integer(1))
        _cc(c, r1, r2)
        return TorusR(c, r1, r2, h == "FilledTorus", expr=x, f=f)
    if h in G._POLYHEDRON_HEADS:
        if h in G._PLATONIC and not (h == "Tetrahedron" and a and _is_coord_list(a[0])):
            c, l_, rot = G._solid_params(a, f)
            _cc(c, l_)
            if h != "Cube" or rot:
                r = OrientFreeR(h, c, l_, f)
                r.expr = x
                return r
        V, F = G.polyhedron_data(x, f)
        _cc(V)
        return PolyhedronR(V, F, expr=x, f=f)
    if h in ("InfiniteLine", "HalfLine", "InfinitePlane", "AffineSpace", "HalfPlane", "AffineHalfSpace",
             "HalfSpace", "Hyperplane", "FullRegion", "EmptyRegion"):
        r = _flat_from(h, a, f)
        r.expr = x
        return r
    raise NotImplementedError(f"{f}: {h} regions are not supported")


def region_of(x, f):
    """Region object for a region expression; TypeError if x is not a region, Unevaluated if symbolic."""
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not isinstance(x, Expr) or not isinstance(x.head, str):
        raise TypeError(f"{f}: {K_input(x)} is not a geometric region")
    h = x.head
    if h in CONVERTERS:
        return CONVERTERS[h](x, f)
    if h in UNSUPPORTED_PRIMS:
        G._check_args(h, list(x.args), allow_index=False)
        raise NotImplementedError(f"{f}: {h} regions are not supported")
    if h in G._CHECKS:
        return _prim(x, f)
    if h == "Region":
        if not x.args:
            raise ValueError(f"{f}: Region needs a region argument")
        return region_of(x.args[0], f)
    if h == "ImplicitRegion":
        return implicit_of(x, f)
    if h == "ParametricRegion":
        return parametric_of(x, f)
    if h in BOOL_HEADS:
        parts = [region_of(r, f) for r in x.args]
        if not parts:
            raise ValueError(f"{f}: {h} needs at least one region")
        if h == "RegionDifference" and len(parts) != 2:
            raise ValueError(f"{f}: RegionDifference takes two regions")
        if len(parts) == 1:
            return parts[0]
        return BoolR(BOOL_HEADS[h], parts, expr=x, f=f)
    if h == "BooleanRegion":
        if len(x.args) != 2 or not K.is_list(x.args[1]):
            raise ValueError(f"{f}: expected BooleanRegion[bfun, {{reg1, reg2, ...}}]")
        parts = [region_of(r, f) for r in x.args[1].args]
        return BoolR("Boolean", parts, expr=x, f=f, bfn=x.args[0])
    if h == "CSGRegion":
        op, regs = csg_parts(x, f)
        parts = [region_of(r, f) for r in regs]
        if len(parts) == 1:
            return parts[0]
        return BoolR(op, parts, expr=x, f=f)
    if h in ("TransformedRegion", "InverseTransformedRegion"):
        if len(x.args) not in (2, 3):
            raise ValueError(f"{f}: expected {h}[reg, f]")
        base = region_of(x.args[0], f)
        t = x.args[1]
        try:
            M = as_transform(t, f, base.edim)
        except (TypeError, Unevaluated):
            if isinstance(t, (str, bool, int, float)) or (K.is_list(t)):
                raise TypeError(f"{f}: {K_input(t)} is not a transformation")
            n = base.edim
            if h == "InverseTransformedRegion" and len(x.args) == 3:
                n = G._pos_int(x.args[2], f)
            return FnImageR(base, t, h == "InverseTransformedRegion", n, expr=x, f=f)
        concrete(list(M))
        A, b = affine_parts(M, f)
        if h == "InverseTransformedRegion":
            if eqz(A.det()):
                raise NotImplementedError(f"{f}: inverse images under singular maps are not supported")
            Ai = A.inv()
            A, b = Ai, list(-Ai * sp.Matrix(b))
        r = affine_map(base, A, b, f)
        r.expr = x
        return r
    if h == "RegionProduct":
        parts = [region_of(r, f) for r in x.args]
        if not parts:
            raise ValueError(f"{f}: RegionProduct needs at least one region")
        if len(parts) == 1:
            return parts[0]
        return ProductR(parts, expr=x, f=f)
    raise TypeError(f"{f}: {K_input(x)} is not a geometric region")


CSG_OPS = ("Union", "Intersection", "Difference", "SymmetricDifference")


def csg_parts(x, f):
    if len(x.args) == 1:
        return "Union", [x.args[0]]
    if len(x.args) != 2:
        raise ValueError(f"{f}: expected CSGRegion[\"op\", {{reg1, reg2, ...}}]")
    op, regs = x.args
    if not isinstance(op, str):
        raise TypeError(f"{f}: the CSG operation must be a string, one of {', '.join(CSG_OPS)}")
    if op not in CSG_OPS:
        raise ValueError(f"{f}: unknown CSG operation {op!r}; use one of {', '.join(CSG_OPS)}")
    if not K.is_list(regs) or not regs.args:
        raise TypeError(f"{f}: the CSG operands must be a non-empty list of regions")
    if op == "Difference" and len(regs.args) < 2:
        raise ValueError(f"{f}: \"Difference\" needs at least two regions")
    return op, list(regs.args)


def implicit_of(x, f):
    if len(x.args) != 2:
        raise ValueError(f"{f}: expected ImplicitRegion[cond, {{x1, ..., xn}}]")
    names, syms, ranges = _vars_spec(x.args[1], f)
    cond = _to_sp_cond(x.args[0], f)
    free = getattr(cond, "free_symbols", set()) - set(syms)
    if free:
        raise Unevaluated()
    return ImplicitR(cond, names, syms, ranges, expr=x, f=f)


class ParamR(Region):
    kind = "ParametricRegion"

    def __init__(self, F, syms, ranges, expr=None, f="Region"):
        self.f = f
        self.F, self.syms, self.ranges = F, syms, ranges
        self.edim = len(F)
        self.expr = expr
        self.bounded = True if all(r is not None for r in ranges) else None
        J = sp.Matrix([[sp.diff(c, u) for u in syms] for c in F])
        rng = [sp.Rational(3, 7) + sp.Rational(k, 11) for k in range(len(syms))]
        pt_ = {}
        for u, r, v in zip(syms, ranges, rng):
            pt_[u] = (r[0] + (r[1] - r[0]) * v) if r is not None else v
        try:
            Jn = J.xreplace(pt_)
            self.dim = Jn.rank(simplify=True) if not any(e.has(sp.Float) for e in Jn) else affine_rank(
                [[sp.Integer(0)] * len(syms)] + [list(Jn.col(j)) for j in range(len(syms))])
        except Exception:
            self.dim = None
        self.convex = None


def parametric_of(x, f):
    if len(x.args) != 2:
        raise ValueError(f"{f}: expected ParametricRegion[{{f1, ..., fn}}, {{{{u, umin, umax}}, ...}}]")
    Fx, spec = x.args
    names, syms, ranges = _vars_spec(spec, f)
    if not K.is_list(Fx):
        raise TypeError(f"{f}: the parametrization must be a list of coordinate functions")
    F = [sx(c, f, "coordinate function") for c in Fx.args]
    if not F:
        raise ValueError(f"{f}: the parametrization must have at least one coordinate")
    free = set().union(*[c.free_symbols for c in F]) - set(syms)
    if free:
        raise Unevaluated()
    if all(r is not None for r in ranges):
        # affine parametrisations over a box are parallelotopes
        try:
            P = [sp.Poly(c, *syms) for c in F]
            if all(p.total_degree() <= 1 for p in P):
                lo = [r[0] for r in ranges]
                base = [simp(c.xreplace(dict(zip(syms, lo)))) for c in F]
                V = [[simp(sp.diff(c, u) * (r[1] - r[0])) for c in F] for u, r in zip(syms, ranges)]
                V = [v for v in V if not is_zero_vec(v)]
                if affine_rank([base] + [vadd(base, v) for v in V]) == len(V):
                    if not V:
                        return PointsR([base], expr=x)
                    if len(V) == len(F):
                        return ParallelotopeR(base, V, expr=x, f=f)
                    if len(V) == 1:
                        return SimplexR([[base, vadd(base, V[0])]], expr=x, f=f)
                    return ParallelotopeR(base, V, expr=x, f=f)
        except sp.PolynomialError:
            pass
    return ParamR(F, syms, ranges, expr=x, f=f)


# ───────────────────────────── shared helpers for the registered functions ─────────────────────────────

UNDEF = Symbol("Undefined")
INDET = Symbol("Indeterminate")


def rdim(r, f):
    d = r.dim
    if d is None and isinstance(r, BoolR):
        d = r._dims()
    if d is None:
        raise NotImplementedError(f"{f}: the dimension of this {r.kind} region cannot be determined")
    return d


def is_point_list(x):
    return K.is_list(x) and bool(x.args) and all(K.is_list(p) for p in x.args)


def query_points(x, r, f):
    """(list of points, many?) for a point argument {x..} or {{x..}, ...}; dimension-checked; symbolic -> Unevaluated."""
    if isinstance(x, Expr) and x.head == "Point" and len(x.args) == 1:
        x = x.args[0]
    many = is_point_list(x)
    P = pts(x, f) if many else [pt(x, f)]
    for p in P:
        if len(p) != r.edim:
            raise ValueError(f"{f}: the point {out(p)} must have {r.edim} coordinates (the embedding dimension)")
    concrete(P)
    return P, many


def measure_of(r, f):
    if isinstance(r, EmptyR):
        return sp.Integer(0)
    d = rdim(r, f)
    if not r.bounded and d != 0:
        if isinstance(r, FlatR):
            return sp.oo
        if r.bounded is None:
            raise NotImplementedError(f"{f}: cannot decide whether this {r.kind} region is bounded")
        return sp.oo
    if d == 0:
        P = r.verts() if not isinstance(r, PointsR) else r.P
        if P is None:
            raise NotImplementedError(f"{f}: the measure of this {r.kind} region is not supported")
        return sp.Integer(len(_nz_count(P)))
    if isinstance(r, PointsR):
        return sp.Integer(len(_nz_count(r.P)))
    return simp(r.measure())


def tri_area3(T):
    return simp(norm(_cross3(vsub(T[1], T[0]), vsub(T[2], T[0]))) / 2)


def boundary_measure(r, f):
    """Perimeter (2-D regions) / surface area (3-D regions): measure of the topological boundary."""
    if isinstance(r, OrientFreeR):
        return r.A
    if isinstance(r, (Polygon2R, Polygon3R)):
        return simp(sum((norm(vsub(b, a)) for a, b in r.edges()), sp.Integer(0)))
    if isinstance(r, BoxR) and r.dim == r.edim:
        w = [r.hi[i] - r.lo[i] for i in range(r.edim)]
        if r.edim == 2:
            return simp(2 * (w[0] + w[1]))
        if r.edim == 3:
            return simp(2 * (w[0] * w[1] + w[1] * w[2] + w[0] * w[2]))
    if isinstance(r, EllipsoidR):
        if r.r is not None:
            return simp(2 * sp.pi * r.r if r.edim == 2 else 4 * sp.pi * r.r ** 2 if r.edim == 3 else
                        r.r ** (r.edim - 1) * 2 * sp.pi ** sp.Rational(r.edim, 2) / sp.gamma(sp.Rational(r.edim, 2)))
        if r.edim == 2 and r.Lm.is_diagonal():
            return EllipseCurveR(r.c, r.Lm[0, 0], r.Lm[1, 1]).measure()
        raise NotImplementedError(f"{f}: the surface area of a general ellipsoid (an elliptic integral) is not "
                                  f"supported")
    if isinstance(r, ShellR):
        if r.edim == 2:
            return simp(2 * sp.pi * (r.r1 + r.r2))
        if r.edim == 3:
            return simp(4 * sp.pi * (r.r1 ** 2 + r.r2 ** 2))
    if isinstance(r, SectorR) and not r.curve:
        if r.full:
            return simp(2 * sp.pi * (r.r1 + r.r2))
        if r.segment:
            A = [r.r2 * sp.cos(r.t1), r.r2 * sp.sin(r.t1)]
            B = [r.r2 * sp.cos(r.t2), r.r2 * sp.sin(r.t2)]
            return simp(r.r2 * r.w + norm(vsub(A, B)))
        return simp((r.r1 + r.r2) * r.w + 2 * (r.r2 - r.r1))
    if isinstance(r, PolyhedronR):
        T = list(r.tris) + [t for v in r.voids for t in v.tris]
        return simp(sum((tri_area3(t) for t in T), sp.Integer(0)))
    if isinstance(r, SimplexR) and len(r.S) == 1 and r.dim == r.edim:
        Vs = r.S[0]
        return simp(sum((simplex_integral([Vs[j] for j in range(len(Vs)) if j != i], 1, X(r.edim))
                         for i in range(len(Vs))), sp.Integer(0)))
    if isinstance(r, ParallelotopeR) and r.dim == r.edim:
        V = r.V
        if r.edim == 2:
            return simp(2 * (norm(V[0]) + norm(V[1])))
        if r.edim == 3:
            return simp(2 * (norm(_cross3(V[0], V[1])) + norm(_cross3(V[1], V[2])) + norm(_cross3(V[0], V[2]))))
    if isinstance(r, RevolutionR):
        R0, Lx = r.r, r.Lax
        if r.kind == "Cylinder":
            return simp(2 * sp.pi * R0 * Lx + 2 * sp.pi * R0 ** 2)
        if r.kind == "Cone":
            return simp(sp.pi * R0 ** 2 + sp.pi * R0 * sp.sqrt(R0 ** 2 + Lx ** 2))
        if r.kind == "CapsuleShape":
            return simp(2 * sp.pi * R0 * Lx + 4 * sp.pi * R0 ** 2)
        return simp(2 * Lx + 2 * sp.pi * R0)
    if isinstance(r, TorusR) and r.solid:
        return simp(4 * sp.pi ** 2 * r.R_ * r.r)
    if isinstance(r, AffineImageR) and r.scale is not None:
        return simp(r.scale ** (r.edim - 1) * boundary_measure(r.base, f))
    if isinstance(r, ProductR) and len(r.parts) == 2 and all(p.dim == p.edim for p in r.parts):
        A, B = r.parts
        return simp(boundary_measure(A, f) * measure_of(B, f) + measure_of(A, f) * boundary_measure(B, f)) \
            if A.edim > 1 and B.edim > 1 else _prod_bd(A, B, f)
    b = r.boundary()
    return measure_of(region_of(b, f), f)


def _prod_bd(A, B, f):
    def bm(R_):
        return sp.Integer(2) if R_.edim == 1 and isinstance(R_, (BoxR, SimplexR)) else boundary_measure(R_, f)
    return simp(bm(A) * measure_of(B, f) + measure_of(A, f) * bm(B))


def _param_spec(args, f, k):
    """({x1..}, [(u, a, b)]) for the parametric measure forms; None if args is not of that form."""
    if len(args) != k + 1:
        return None
    specs = args[1:]
    if not all(K.is_list(s) and len(s.args) == 3 and isinstance(s.args[0], Symbol) for s in specs):
        return None
    U = [K.to_sp(s.args[0]) for s in specs]
    rng = [(sx(s.args[1], f, "parameter bound"), sx(s.args[2], f, "parameter bound")) for s in specs]
    Fx = args[0]
    if K.is_list(Fx):
        F = [sx(c, f, "coordinate") for c in Fx.args]
    else:
        F = list(U) + [sx(Fx, f, "coordinate")]
    return F, U, rng


def param_measure(F, U, rng, f):
    J = sp.Matrix([[sp.diff(c, u) for u in U] for c in F])
    G_ = (J.T * J).applyfunc(sp.simplify)
    dens = sp.sqrt(sp.factor(sp.simplify(G_.det())))
    dens = sp.simplify(dens)
    try:
        v = sym_integrate(dens, [(u, a, b) for u, (a, b) in reversed(list(zip(U, rng)))])
    except NotImplementedError:
        raise NotImplementedError(f"{f}: this parametric measure has no closed form")
    return simp(sp.simplify(v))


def _measure_fn(f, k, args):
    """Shared body of Area (k = 2), Volume (k = 3), RegionMeasure (k = None)."""
    if len(args) >= 2 and K.is_list(args[1]) and len(args[1].args) == 3 and isinstance(args[1].args[0], Symbol):
        spec = _param_spec(args, f, len(args) - 1)
        if spec is None:
            raise ValueError(f"{f}: expected {f}[{{x1, ...}}, {{s, smin, smax}}, ...]")
        F, U, rng = spec
        if k is not None and len(U) != k:
            raise ValueError(f"{f}: give {k} parameter ranges")
        if len(set(U)) != len(U):
            raise ValueError(f"{f}: the parameters must be distinct")
        if len(F) < len(U):
            raise ValueError(f"{f}: a {len(U)}-parameter region needs at least {len(U)} coordinates")
        free = set().union(*[c.free_symbols for c in F]) | {s for a, b in rng for s in (a.free_symbols |
                                                                                           b.free_symbols)}
        if free - set(U) or any(a.free_symbols & set(U) for a, _ in rng):
            raise Unevaluated()
        return out(param_measure(F, U, rng, f), *args)
    x = args[0]
    r = region_of(x, f)
    if len(args) == 2 and k is None:
        d = args[1]
        if not K.is_int(d) or isinstance(d, bool):
            if isinstance(d, Symbol):
                raise Unevaluated()
            raise TypeError(f"{f}: the dimension must be a non-negative integer, not {K_input(d)}")
        d = int(d)
        if d < 0:
            raise ValueError(f"{f}: the dimension must be non-negative")
        rd = rdim(r, f) if not isinstance(r, EmptyR) else -1
        if isinstance(r, EmptyR) or d > rd:
            return 0
        if d < rd:
            return Symbol("Infinity")
        return out(measure_of(r, f), x)
    if len(args) != 1:
        raise ValueError(f"{f}: expected {f}[reg]")
    if k is not None and (isinstance(r, EmptyR) or rdim(r, f) != k):
        return UNDEF
    return out(measure_of(r, f), x)


# ───────────────────────────── measures ─────────────────────────────

_MEAS_COMMON = ("Regions are the W19 primitives (Point, Line, Triangle, Polygon (holes allowed), Rectangle, "
                "Cuboid, Parallelogram, Parallelepiped, Simplex, polyhedra, Disk / Circle / Ball / Sphere (also "
                "sectors and arcs), Ellipsoid, Annulus, SphericalShell, DiskSegment, Cylinder, Cone, CapsuleShape, "
                "StadiumShape, Torus, FilledTorus, flats) and ImplicitRegion (linear systems, one quadratic "
                "ellipsoid inequality, 1-D conditions), affine ParametricRegion, MeshRegion / BoundaryMeshRegion, "
                "TransformedRegion with affine maps, RegionProduct, and Boolean regions whose parts are proved "
                "disjoint or nested. Self-intersecting or overlapping polygons, unsupported regions and Boolean "
                "combinations that cannot be resolved exactly raise NotImplementedError; a non-region raises "
                "TypeError; symbolic parameters stay unevaluated. ")


@fn("RegionMeasure", area=AREA, status="added",
    notes="RegionMeasure[reg] gives the measure of reg in its own dimension (count of distinct points, length, "
          "area, volume, ...); RegionMeasure[reg, d] the d-dimensional measure (0 when d > dim, Infinity when "
          "d < dim); RegionMeasure[{x1, ..., xn}, {s, smin, smax}, ...] the parametric formula Integral of "
          "Sqrt[Det[J^T J]] (multiple coverings counted). Unbounded regions give Infinity, EmptyRegion 0. "
          + _MEAS_COMMON + "Check: exact (simplex Dirichlet formula, closed-form ball moments, exact SymPy "
          "integration); tests compare closed forms and cross-check Area / Volume.")
def _region_measure(reg: Expr | ListOf[Expr], *rest: NonNegInteger | ListOf[Expr]) -> Real | Expr:
    return _measure_fn("RegionMeasure", None, [reg, *rest])


@fn("Area", area=AREA, status="added",
    notes="Area[reg] gives the area of a two-dimensional region (any embedding dimension) and Undefined for a "
          "region of another dimension; Area[{x1, ..., xn}, {s, smin, smax}, {t, tmin, tmax}] the area of the "
          "parametrized surface (multiple coverings counted), Area[z, {s, ..}, {t, ..}] that of the graph "
          "{s, t, z}. " + _MEAS_COMMON + "Check: exact; tests compare closed forms (Pi r^2, triangle and "
          "polygon shoelace areas) and the parametric sphere area 4 Pi.")
def _area(reg: Expr | ListOf[Expr], *rest: ListOf[Expr]) -> Real | Expr:
    return _measure_fn("Area", 2, [reg, *rest])


@fn("Volume", area=AREA, status="added",
    notes="Volume[reg] gives the volume of a three-dimensional region and Undefined otherwise; "
          "Volume[{x1, ..., xn}, {s, ..}, {t, ..}, {u, ..}] the parametric volume (coverings counted). "
          + _MEAS_COMMON + "Check: exact; tests compare closed forms (4/3 Pi r^3, Pi r^2 h, tetrahedron "
          "|det|/6, 2 Pi^2 R r^2).")
def _volume(reg: Expr | ListOf[Expr], *rest: ListOf[Expr]) -> Real | Expr:
    return _measure_fn("Volume", 3, [reg, *rest])


def _boundary_param(F, U, rng, f):
    """Measure of the image of the boundary of the parameter box (one dimension lower)."""
    tot = sp.Integer(0)
    for i, u in enumerate(U):
        for val in rng[i]:
            F2 = [c.xreplace({u: val}) for c in F]
            U2 = [v for j, v in enumerate(U) if j != i]
            r2 = [r_ for j, r_ in enumerate(rng) if j != i]
            tot += param_measure(F2, U2, r2, f) if U2 else sp.Integer(1)
    return simp(tot)


def _bd_fn(f, k, args):
    if len(args) >= 2:
        spec = _param_spec(args, f, k)
        if spec is None:
            raise ValueError(f"{f}: expected {f}[{{x1, ...}}, " + ", ".join(["{s, smin, smax}"] * k) + "]")
        F, U, rng = spec
        free = set().union(*[c.free_symbols for c in F]) | {s for a, b in rng for s in (a.free_symbols |
                                                                                           b.free_symbols)}
        if free - set(U):
            raise Unevaluated()
        return out(_boundary_param(F, U, rng, f), *args)
    x = args[0]
    r = region_of(x, f)
    if isinstance(r, EmptyR) or rdim(r, f) != k:
        return UNDEF
    if not r.bounded:
        return Symbol("Infinity")
    return out(boundary_measure(r, f), x)


@fn("Perimeter", area=AREA, status="added",
    notes="Perimeter[reg] gives the length of the boundary of a two-dimensional region (holes included) and "
          "Undefined for other dimensions; Perimeter[{x1, x2}, {s, ..}, {t, ..}] the length of the image of the "
          "parameter rectangle's boundary (assumes the map is injective, like the formula form). "
          + _MEAS_COMMON + "Check: exact; tests compare 2 Pi r, polygon edge sums, the ellipse "
          "4 a EllipticE[1 - b^2/a^2] and a sector r t + 2 r.")
def _perimeter(reg: Expr | ListOf[Expr], *rest: ListOf[Expr]) -> Real | Expr:
    return _bd_fn("Perimeter", 2, [reg, *rest])


@fn("SurfaceArea", area=AREA, status="added",
    notes="SurfaceArea[reg] gives the area of the boundary surface of a three-dimensional region (voids "
          "included) and Undefined otherwise; SurfaceArea[{x1, ..}, {s, ..}, {t, ..}, {u, ..}] the area of the "
          "image of the parameter box's boundary (injective maps). General ellipsoids raise "
          "NotImplementedError (elliptic integrals). " + _MEAS_COMMON + "Check: exact; tests compare 4 Pi r^2, "
          "cuboid 2(ab + bc + ca), cylinder and cone closed forms, torus 4 Pi^2 R r.")
def _surface_area(reg: Expr | ListOf[Expr], *rest: ListOf[Expr]) -> Real | Expr:
    return _bd_fn("SurfaceArea", 3, [reg, *rest])


def centroid_of(r, f):
    if isinstance(r, EmptyR):
        raise ValueError(f"{f}: the empty region has no centroid")
    d = rdim(r, f)
    if d == 0:
        P = r.P if isinstance(r, PointsR) else r.verts()
        if P is None:
            raise NotImplementedError(f"{f}: the centroid of this {r.kind} region is not supported")
        return [simp(sum((p[i] for p in P), sp.Integer(0)) / len(P)) for i in range(r.edim)]
    if not r.bounded:
        return None
    Xs = X(r.edim)
    m = r.measure()
    if eqz(m):
        raise ValueError(f"{f}: the region has zero measure")
    return [simp(r.integrate(Xs[i], Xs) / m) for i in range(r.edim)]


@fn("RegionCentroid", area=AREA, status="added",
    notes="RegionCentroid[reg] gives the centroid Integral[x] / RegionMeasure[reg] in the region's own dimension; "
          "the mean for a finite point set; a list of Indeterminate for regions of infinite measure. "
          "EmptyRegion raises ValueError. " + _MEAS_COMMON + "Check: exact; tests compare the triangle vertex "
          "mean, the half-disk 4 r / (3 Pi), the cone h / 4 and a translation-equivariance check.")
def _region_centroid(reg: Expr) -> ListOf[Real]:
    f = "RegionCentroid"
    r = region_of(reg, f)
    c = centroid_of(r, f)
    if c is None:
        return L(*[INDET] * r.edim)
    return out(c, reg)


def _exponents(e, n, f):
    if not K.is_list(e):
        if isinstance(e, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: the exponents must be a list {{i1, ..., in}}, not {K_input(e)}")
    if len(e.args) != n:
        raise ValueError(f"{f}: give {n} exponents (the embedding dimension)")
    res = []
    for v in e.args:
        if isinstance(v, bool) or not K.is_int(v):
            s_ = sx(v, f, "exponent")
            if s_.free_symbols:
                raise Unevaluated()
            raise NotImplementedError(f"{f}: only non-negative integer exponents are supported")
        if int(v) < 0:
            raise NotImplementedError(f"{f}: only non-negative integer exponents are supported")
        res.append(int(v))
    return res


@fn("RegionMoment", area=AREA, status="added",
    notes="RegionMoment[reg, {i1, ..., in}] gives the moment Integral[x1^i1 ... xn^in] over reg (in its own "
          "dimension). Non-integer or negative exponents raise NotImplementedError, a wrong-length exponent "
          "list ValueError; unbounded regions with a nonzero integrand give +-Infinity only for constant "
          "integrands (else NotImplementedError). " + _MEAS_COMMON + "Check: exact; tests compare the unit "
          "square moments 1/((i+1)(j+1)), Disk second moments Pi/4 and RegionMoment[reg, {0..}] = measure.")
def _region_moment(reg: Expr, e: ListOf[NonNegInteger]) -> Real:
    f = "RegionMoment"
    r = region_of(reg, f)
    ex = _exponents(e, r.edim, f)
    Xs = X(r.edim)
    g = sp.Integer(1)
    for xi, k in zip(Xs, ex):
        g *= xi ** k
    if isinstance(r, PointsR):
        return out(simp(r.integrate(g, Xs)), reg)
    if not r.bounded:
        if all(k == 0 for k in ex):
            return Symbol("Infinity")
        raise NotImplementedError(f"{f}: moments of unbounded regions are not supported")
    return out(simp(r.integrate(g, Xs)), reg)


def inertia_matrix(r, c, f):
    n = r.edim
    Xs = X(n)
    Y = [Xs[i] - c[i] for i in range(n)]
    if isinstance(r, EmptyR):
        return sp.zeros(n, n)
    if not r.bounded:
        raise ValueError(f"{f}: the region is unbounded")
    if n == 2:
        Ixx = r.integrate(Y[1] ** 2, Xs)
        Iyy = r.integrate(Y[0] ** 2, Xs)
        Ixy = -r.integrate(Y[0] * Y[1], Xs)
        return sp.Matrix([[Ixx, Ixy], [Ixy, Iyy]]).applyfunc(simp)
    if n == 3:
        M = sp.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                g = (sum((y ** 2 for y in Y), sp.Integer(0)) if i == j else 0) - Y[i] * Y[j]
                M[i, j] = simp(r.integrate(sp.expand(g), Xs))
        return M
    raise NotImplementedError(f"{f}: moments of inertia are supported in 2 and 3 dimensions")


@fn("MomentOfInertia", area=AREA, status="added",
    notes="MomentOfInertia[reg] gives the moment of inertia matrix of reg (unit density) about its centroid; "
          "MomentOfInertia[reg, p] about the point p; MomentOfInertia[reg, p, v] the scalar moment about the axis "
          "through p in direction v (v.I.v / v.v). In 3-D the matrix is Integral[|x|^2 I - x x^T]; in 2-D "
          "{{Integral[y^2], -Integral[x y]}, {-Integral[x y], Integral[x^2]}}. Point sets sum over the points. "
          "Unbounded regions raise ValueError, a zero axis vector ValueError, other dimensions "
          "NotImplementedError. " + _MEAS_COMMON + "Check: exact; tests compare the solid ball 8/15 Pi r^5, the "
          "unit cube 1/6 about its centre, the parallel-axis theorem I_p = I_c + m(|d|^2 I - d d^T).")
def _moment_of_inertia(reg: Expr, *rest: ListOf[Real]) -> MatrixOf[Real] | Real:
    f = "MomentOfInertia"
    r = region_of(reg, f)
    if len(rest) > 2:
        raise ValueError(f"{f}: expected MomentOfInertia[reg], [reg, p] or [reg, p, v]")
    if rest:
        c = pt(rest[0], f, "point", dim=r.edim)
        concrete(c)
    else:
        c = centroid_of(r, f)
        if c is None:
            raise ValueError(f"{f}: the region is unbounded")
    M = inertia_matrix(r, c, f)
    if len(rest) == 2:
        v = pt(rest[1], f, "axis", dim=r.edim)
        concrete(v)
        if is_zero_vec(v):
            raise ValueError(f"{f}: the axis vector must be non-zero")
        vm = sp.Matrix(v)
        return out(simp((vm.T * M * vm)[0] / n2(v)), reg, *rest)
    return out(M.tolist(), reg, *rest)


# ───────────────────────────── dimension, bounds, membership ─────────────────────────────

@fn("RegionDimension", area=AREA, status="added",
    notes="RegionDimension[reg] gives the geometric dimension of reg (0 for points, 1 for curves, ...; "
          "-Infinity for EmptyRegion). Formula regions: ImplicitRegion dimension when it reduces to a primitive, "
          "ParametricRegion the generic rank of the Jacobian; Boolean regions when resolvable (else "
          "NotImplementedError). " + _MEAS_COMMON + "Check: exact; tests compare primitives of every dimension.")
def _region_dimension(reg: Expr) -> Integer | Expr:
    f = "RegionDimension"
    r = region_of(reg, f)
    if isinstance(r, EmptyR):
        return R(-sp.oo)
    return rdim(r, f)


@fn("RegionEmbeddingDimension", area=AREA, status="added",
    notes="RegionEmbeddingDimension[reg] gives the dimension of the space reg lies in (the coordinate count). "
          + _MEAS_COMMON + "Check: exact; tests compare 2-D, 3-D and n-D primitives.")
def _region_embedding_dimension(reg: Expr) -> PosInteger:
    return region_of(reg, "RegionEmbeddingDimension").edim


@fn("RegionBounds", area=AREA, status="added",
    notes="RegionBounds[reg] gives the coordinate ranges {{x1min, x1max}, ...} of the smallest axis-aligned box "
          "containing reg (Infinity entries for unbounded directions). EmptyRegion raises ValueError; a second "
          "argument (bound types) raises NotImplementedError. " + _MEAS_COMMON + "Check: exact; tests compare "
          "polygon vertex extremes, disk-sector arc extremes, cylinder bounds c +- r Sqrt[1 - u_i^2].")
def _region_bounds(reg: Expr, *rest: String) -> MatrixOf[Real]:
    f = "RegionBounds"
    if rest:
        raise NotImplementedError(f"{f}: bound types other than the default are not supported")
    r = region_of(reg, f)
    if isinstance(r, EmptyR):
        raise ValueError(f"{f}: the empty region has no bounds")
    return out([[a, b] for a, b in r.bounds()], reg)


def _member_out(r, P, many, f):
    vals = [bool(r.member(p)) for p in P]
    return L(*vals) if many else vals[0]


@fn("RegionMember", area=AREA, status="added",
    notes="RegionMember[reg, p] gives True if the point p lies in reg (closed regions include their boundary), "
          "RegionMember[reg, {p1, p2, ...}] a list, RegionMember[reg] a RegionMemberFunction[reg]. Exact sign "
          "tests (inexact input: tolerance 1e-12 relative); a point of the wrong dimension raises ValueError; "
          "symbolic points stay unevaluated. Non-convex polyhedra use the solid-angle winding number, "
          "verified to be an integer (else ArithmeticError). " + _MEAS_COMMON +
          "Check: exact; tests compare boundary, interior and exterior points of every primitive.")
def _region_member(reg: Expr, *p: ListOf[Real] | MatrixOf[Real]) -> Boolean | ListOf[Boolean] | Expr:
    f = "RegionMember"
    r = region_of(reg, f)
    if not p:
        return Expr("RegionMemberFunction", reg)
    if len(p) > 1:
        raise ValueError(f"{f}: expected RegionMember[reg, p]")
    P, many = query_points(p[0], r, f)
    return _member_out(r, P, many, f)


def _fn_head(name, f_inner):
    def hook(head, args, ev_):
        if len(head.args) != 1 or len(args) != 1:
            return K.UNHANDLED
        try:
            return f_inner(head.args[0], args[0])
        except Unevaluated:
            return K.UNHANDLED
    return hook


@fn("RegionMemberFunction", area=AREA, status="added",
    notes="RegionMemberFunction[reg] (from RegionMember[reg]) is a function object: RegionMemberFunction[reg][p] "
          "gives RegionMember[reg, p] (also for a list of points). It stays inert; a non-region argument raises "
          "TypeError. Check: delegated (RegionMember); tests compare both forms.")
def _region_member_function(reg: Expr) -> Expr:
    region_of(reg, "RegionMemberFunction")
    raise Unevaluated()


def dist_points(r, P, many, inputs, f, kind="dist"):
    vals = []
    for p in P:
        if kind == "dist":
            vals.append(r.dist(p))
        elif kind == "sdist":
            vals.append(r.sdist(p))
        else:
            d2, q = r.nearest(p)
            if q is None:
                raise ValueError(f"{f}: the empty region has no nearest point")
            vals.append(q)
    vals = [simp(v) if not isinstance(v, list) else [simp(c) for c in v] for v in vals]
    return out(vals if many else vals[0], *inputs)


def region_distance(A, B, f):
    """Distance between two regions (exact cases)."""
    if A.edim != B.edim:
        raise ValueError(f"{f}: the regions must have the same embedding dimension")
    if isinstance(A, EmptyR) or isinstance(B, EmptyR):
        return sp.oo
    try:
        dj = disjoint(A, B)
    except ValueError:
        dj = None
    if dj is False:
        return sp.Integer(0)
    for X_, Y_ in ((A, B), (B, A)):
        if isinstance(X_, PointsR):
            return smin([Y_.dist(p) for p in X_.P])
    for X_, Y_ in ((A, B), (B, A)):
        b = _ball_of(X_)
        if b is not None:
            return simp(smax([sp.Integer(0), Y_.dist(b[0]) - b[1]]))
    for X_, Y_ in ((A, B), (B, A)):
        if isinstance(X_, SimplexR) and isinstance(Y_, SimplexR) and X_.dim <= 1 and Y_.dim <= 1:
            return sqrt_out(smin([_seg_seg2(a, b) for a in X_.S for b in Y_.S]))
    raise NotImplementedError(f"{f}: the distance between {A.kind} and {B.kind} regions is not supported")


def _seg_seg2(s, t):
    """Squared distance between two segments (exact, any dimension)."""
    a, b = s
    c, d = t
    cands = [seg_nearest(a, c, d)[0], seg_nearest(b, c, d)[0], seg_nearest(c, a, b)[0], seg_nearest(d, a, b)[0]]
    u, v, w = vsub(b, a), vsub(d, c), vsub(a, c)
    M = sp.Matrix([[dot(u, u), -dot(u, v)], [-dot(u, v), dot(v, v)]])
    if not eqz(M.det()):
        st = M.LUsolve(sp.Matrix([-dot(u, w), dot(v, w)]))
        s_, t_ = simp(st[0]), simp(st[1])
        if sgn(s_) >= 0 and sgn(s_ - 1) <= 0 and sgn(t_) >= 0 and sgn(t_ - 1) <= 0:
            p = vadd(a, vmul(s_, u))
            q = vadd(c, vmul(t_, v))
            cands.append(simp(n2(vsub(p, q))))
    return smin(cands)


@fn("RegionDistance", area=AREA, status="added",
    notes="RegionDistance[reg, p] gives the Euclidean distance from p to reg (0 inside), RegionDistance[reg, "
          "{p1, ...}] a list, RegionDistance[reg1, reg2] the distance between two regions (point sets, balls, "
          "segments/polylines, or regions proved to intersect), RegionDistance[reg] a RegionDistanceFunction. "
          "EmptyRegion gives Infinity. Ellipses / ellipsoids, tori and non-similarity transformed regions raise "
          "NotImplementedError. " + _MEAS_COMMON + "Check: exact (closed-form projections, exact face "
          "enumeration for simplices); tests compare hand distances and |p - RegionNearest[reg, p]|.")
def _region_distance(reg: Expr, *p: ListOf[Real] | MatrixOf[Real] | Expr) -> Real | ListOf[Real] | Expr:
    f = "RegionDistance"
    r = region_of(reg, f)
    if not p:
        return Expr("RegionDistanceFunction", reg)
    if len(p) > 1:
        raise ValueError(f"{f}: expected RegionDistance[reg, p]")
    if not K.is_list(p[0]) and not (isinstance(p[0], Expr) and p[0].head == "Point" and False):
        B = region_of(p[0], f)
        return out(region_distance(r, B, f), reg, p[0])
    P, many = query_points(p[0], r, f)
    return dist_points(r, P, many, (reg, p[0]), f)


@fn("RegionDistanceFunction", area=AREA, status="added",
    notes="RegionDistanceFunction[reg] (from RegionDistance[reg]) applied to a point or a list of points gives "
          "RegionDistance[reg, p]. Inert; a non-region raises TypeError. Check: delegated (RegionDistance); "
          "tests compare both forms.")
def _region_distance_function(reg: Expr) -> Expr:
    region_of(reg, "RegionDistanceFunction")
    raise Unevaluated()


@fn("SignedRegionDistance", area=AREA, status="added",
    notes="SignedRegionDistance[reg, p] gives the distance from p to the boundary of a full-dimensional region, "
          "negative inside and positive outside; a list of points gives a list. Lower-dimensional regions raise "
          "ValueError, unsupported shapes NotImplementedError. " + _MEAS_COMMON + "Check: exact; tests compare "
          "|p - c| - r for balls, box face distances and the sign against RegionMember.")
def _signed_region_distance(reg: Expr, p: ListOf[Real] | MatrixOf[Real]) -> Real | ListOf[Real]:
    f = "SignedRegionDistance"
    r = region_of(reg, f)
    if isinstance(r, EmptyR):
        return Symbol("Infinity")
    if rdim(r, f) != r.edim:
        raise ValueError(f"{f}: the region must be full-dimensional (dimension {r.edim})")
    P, many = query_points(p, r, f)
    return dist_points(r, P, many, (reg, p), f, kind="sdist")


@fn("RegionNearest", area=AREA, status="added",
    notes="RegionNearest[reg, p] gives a point of reg nearest to p (p itself when inside; one of several when "
          "not unique, e.g. the centre of a sphere), RegionNearest[reg, {p1, ...}] a list, RegionNearest[reg] a "
          "RegionNearestFunction. EmptyRegion raises ValueError. " + _MEAS_COMMON + "Check: exact; tests compare "
          "hand projections and verify RegionMember[reg, q] and |p - q| = RegionDistance[reg, p].")
def _region_nearest(reg: Expr, *p: ListOf[Real] | MatrixOf[Real]) -> ListOf[Real] | MatrixOf[Real] | Expr:
    f = "RegionNearest"
    r = region_of(reg, f)
    if not p:
        return Expr("RegionNearestFunction", reg)
    if len(p) > 1:
        raise ValueError(f"{f}: expected RegionNearest[reg, p]")
    P, many = query_points(p[0], r, f)
    return dist_points(r, P, many, (reg, p[0]), f, kind="nearest")


@fn("RegionNearestFunction", area=AREA, status="added",
    notes="RegionNearestFunction[reg] (from RegionNearest[reg]) applied to a point or list of points gives "
          "RegionNearest[reg, p]. Inert; a non-region raises TypeError. Check: delegated (RegionNearest); tests "
          "compare both forms.")
def _region_nearest_function(reg: Expr) -> Expr:
    region_of(reg, "RegionNearestFunction")
    raise Unevaluated()


def farthest_between(A, B, f):
    if isinstance(A, PointsR):
        return smax([B.farthest(p) for p in A.P])
    if isinstance(B, PointsR):
        return smax([A.farthest(p) for p in B.P])
    for X_, Y_ in ((A, B), (B, A)):
        b = _ball_of(X_) or ((X_.c, X_.r) if isinstance(X_, SphereR) else None)
        if b is not None:
            return simp(Y_.farthest(b[0]) + b[1])
    for X_, Y_ in ((A, B), (B, A)):
        V = X_.verts()
        if V is not None:
            return smax([Y_.farthest(v) for v in V])
    raise NotImplementedError(f"{f}: the farthest distance between {A.kind} and {B.kind} regions is not supported")


@fn("RegionFarthestDistance", area=AREA, status="added",
    notes="RegionFarthestDistance[reg1, reg2] gives the largest distance between a point of reg1 and a point of "
          "reg2 (reg2 may be a point {x, ...}); attained at vertices for polytopes, |c - p| + r for balls and "
          "spheres. Unbounded regions give Infinity; EmptyRegion raises ValueError. " + _MEAS_COMMON +
          "Check: exact; tests compare RegionFarthestDistance[Disk[], {3, 3}] = 1 + 3 Sqrt[2] (Wolfram "
          "documentation example) and vertex-pair maxima.")
def _region_farthest_distance(reg1: Expr, reg2: Expr | ListOf[Real]) -> Real:
    f = "RegionFarthestDistance"
    A = region_of(reg1, f)
    if K.is_list(reg2):
        P, many = query_points(reg2, A, f)
        B = PointsR(P)
    else:
        B = region_of(reg2, f)
    if A.edim != B.edim:
        raise ValueError(f"{f}: the regions must have the same embedding dimension")
    if isinstance(A, EmptyR) or isinstance(B, EmptyR):
        raise ValueError(f"{f}: the empty region has no farthest distance")
    if not A.bounded or not B.bounded:
        return Symbol("Infinity")
    return out(simp(farthest_between(A, B, f)), reg1, reg2)


def directed_hausdorff(A, B, f):
    """sup over a in A of dist(a, B)."""
    if isinstance(A, PointsR):
        return smax([B.dist(p) for p in A.P])
    if B.convex and isinstance(A, VERTS_IN):
        return smax([B.dist(v) for v in A.verts()])
    ba, bb = _ball_of(A), _ball_of(B)
    if ba is not None and bb is not None:
        return simp(smax([sp.Integer(0), norm(vsub(ba[0], bb[0])) + ba[1] - bb[1]]))
    if within(A, B):
        return sp.Integer(0)
    raise NotImplementedError(f"{f}: the Hausdorff distance between {A.kind} and {B.kind} regions is not supported")


@fn("RegionHausdorffDistance", area=AREA, status="added",
    notes="RegionHausdorffDistance[reg1, reg2] gives max(sup_{a in reg1} d(a, reg2), sup_{b in reg2} d(b, reg1)). "
          "Exact for finite point sets, a polytope against a convex region (the distance to a convex set is "
          "convex, so the supremum is at a vertex), two balls (|c1 - c2| + |r1 - r2|) and nested regions; other "
          "pairs raise NotImplementedError. " + _MEAS_COMMON + "Check: exact; tests compare hand values and "
          "symmetry in the arguments.")
def _region_hausdorff_distance(reg1: Expr, reg2: Expr) -> Real:
    f = "RegionHausdorffDistance"
    A, B = region_of(reg1, f), region_of(reg2, f)
    if A.edim != B.edim:
        raise ValueError(f"{f}: the regions must have the same embedding dimension")
    if isinstance(A, EmptyR) or isinstance(B, EmptyR):
        raise ValueError(f"{f}: the Hausdorff distance to the empty region is undefined")
    if not A.bounded or not B.bounded:
        raise NotImplementedError(f"{f}: unbounded regions are not supported")
    return out(simp(smax([directed_hausdorff(A, B, f), directed_hausdorff(B, A, f)])), reg1, reg2)


@fn("RegionBoundary", area=AREA, status="added",
    notes="RegionBoundary[reg] gives the topological boundary of reg in its embedding space: the region itself "
          "when its dimension is below the embedding dimension, Circle for Disk, Sphere for Ball, Line loops for "
          "polygons, the face Polygon list for polyhedra and cuboids, the end points of a 1-D segment, "
          "Hyperplane for HalfSpace, EmptyRegion for FullRegion. " + _MEAS_COMMON +
          "Check: exact; tests compare the boundary forms and that Perimeter / SurfaceArea equal the boundary "
          "measure.")
def _region_boundary(reg: Expr) -> Expr:
    f = "RegionBoundary"
    r = region_of(reg, f)
    if isinstance(r, EmptyR):
        return reg
    d = rdim(r, f)
    if d < r.edim and not (isinstance(r, SimplexR) and r.boundary_fn is not None):
        if isinstance(r, (UnionCellsR,)) and r.boundary_fn is not None:
            return r.boundary_fn()
        return reg
    b = r.boundary()
    if b is None:
        raise NotImplementedError(f"{f}: the boundary of this {r.kind} region is not supported")
    return b


# ───────────────────────────── predicates ─────────────────────────────

def _try_region(x, f):
    """Region object, or None when x is not a valid region (TypeError / ValueError)."""
    try:
        return region_of(x, f)
    except (TypeError, ValueError):
        return None


def _looks_region(x):
    return isinstance(x, Expr) and isinstance(x.head, str) and x.head in REGION_HEADS


@fn("RegionQ", area=AREA, status="added",
    notes="RegionQ[reg] gives True if reg is a valid geometric region (a primitive with valid arguments, a "
          "formula, mesh, Boolean, transformed or product region) and False otherwise (numbers, strings, "
          "symbols, invalid arguments). Regions with symbolic parameters stay unevaluated. Check: exact; tests "
          "compare valid and invalid regions.")
def _region_q(reg: Expr) -> Boolean:
    if not _looks_region(reg):
        return False
    try:
        region_of(reg, "RegionQ")
        return True
    except NotImplementedError:
        return True
    except (TypeError, ValueError):
        return False


@fn("ConstantRegionQ", area=AREA, status="added",
    notes="ConstantRegionQ[reg] gives True if reg is a valid region with no symbolic parameters and False "
          "otherwise (including Disk[{0, 0}, r] and non-regions). Check: exact; tests compare numeric and "
          "symbolic regions.")
def _constant_region_q(reg: Expr) -> Boolean:
    if not _looks_region(reg):
        return False
    try:
        region_of(reg, "ConstantRegionQ")
        return True
    except Unevaluated:
        return False
    except NotImplementedError:
        return True
    except (TypeError, ValueError):
        return False


@fn("BoundedRegionQ", area=AREA, status="added",
    notes="BoundedRegionQ[reg] gives True if reg is contained in a ball of finite radius, False if it is "
          "unbounded or not a region. Undecidable formula regions raise NotImplementedError. " + _MEAS_COMMON +
          "Check: exact; tests compare primitives, flats and ImplicitRegion inequalities.")
def _bounded_region_q(reg: Expr) -> Boolean:
    f = "BoundedRegionQ"
    if not _looks_region(reg):
        return False
    r = _try_region(reg, f)
    if r is None:
        return False
    if r.bounded is None:
        raise NotImplementedError(f"{f}: cannot decide whether this {r.kind} region is bounded")
    return bool(r.bounded)


@fn("ConvexRegionQ", area=AREA, status="added",
    notes="ConvexRegionQ[reg] gives True if reg is convex (contains every segment between its points), False "
          "otherwise or for non-regions. Exact convexity tests for polygons and polyhedra; sectors are convex "
          "iff their angle is at most Pi; undecidable regions raise NotImplementedError. " + _MEAS_COMMON +
          "Check: exact; tests compare convex and non-convex polygons, annuli and point sets.")
def _convex_region_q(reg: Expr) -> Boolean:
    f = "ConvexRegionQ"
    if not _looks_region(reg):
        return False
    r = _try_region(reg, f)
    if r is None:
        return False
    if isinstance(r, PointsR):
        return len(_nz_count(r.P)) <= 1
    if isinstance(r, SimplexR) and len(r.S) > 1:
        if r.dim == 1:
            return _union_convex(r)
        raise NotImplementedError(f"{f}: cannot decide whether a union of simplices is convex")
    if r.convex is None:
        raise NotImplementedError(f"{f}: cannot decide whether this {r.kind} region is convex")
    return bool(r.convex)


def _union_convex(r):
    """A union of segments (polyline) is convex iff all its vertices are collinear and it covers its hull."""
    V = r.verts()
    if affine_rank(V) != 1:
        return False
    a, b = _extreme_pair(V)
    d = vsub(b, a)
    ivs = sorted([sorted([dot(vsub(s[0], a), d), dot(vsub(s[1], a), d)], key=num) for s in r.S], key=lambda t: num(t[0]))
    cur = ivs[0][1]
    for lo, hi in ivs[1:]:
        if lt(cur, lo):
            return False
        cur = smax([cur, hi])
    return True


@fn("SolidRegionQ", area=AREA, status="added",
    notes="SolidRegionQ[reg] gives True if reg is a constant three-dimensional region in R^3 with no "
          "lower-dimensional parts (Ball, Cuboid, polyhedra, Cylinder, ...), False otherwise. Check: exact; "
          "tests compare solids, surfaces and planar regions.")
def _solid_region_q(reg: Expr) -> Boolean:
    f = "SolidRegionQ"
    if not _looks_region(reg):
        return False
    try:
        r = region_of(reg, f)
    except (TypeError, ValueError, Unevaluated):
        return False
    if r.edim != 3 or isinstance(r, EmptyR):
        return False
    if isinstance(r, UnionCellsR):
        return all(c.dim == 3 for c in r.cells)
    if isinstance(r, BoolR) and r.res is None and r.op == "Union":
        return all(p.dim == 3 for p in r.parts)
    try:
        return rdim(r, f) == 3
    except NotImplementedError:
        raise


def _decide(v, f, what):
    if v is None:
        raise NotImplementedError(f"{f}: cannot decide {what} for these regions")
    return bool(v)


@fn("RegionWithin", area=AREA, status="added",
    notes="RegionWithin[reg1, reg2] gives True if reg2 is contained in reg1 and False otherwise. Decided "
          "exactly for a polytope / point set inside a convex region, a ball inside a ball or a convex "
          "polytope, and by counterexample points (vertices, centres) for a negative answer; undecidable pairs "
          "raise NotImplementedError. " + _MEAS_COMMON + "Check: exact; tests compare nested and overlapping "
          "disks, triangles in rectangles and the vertex counterexample.")
def _region_within(reg1: Expr, reg2: Expr) -> Boolean:
    f = "RegionWithin"
    A, B = region_of(reg1, f), region_of(reg2, f)
    if A.edim != B.edim:
        raise ValueError(f"{f}: the regions must have the same embedding dimension")
    return _decide(within(B, A), f, "containment")


@fn("RegionDisjoint", area=AREA, status="added",
    notes="RegionDisjoint[reg1, reg2, ...] gives True if no two of the regions share a point. Decided exactly "
          "for convex polytopes (exact half-space intersection), balls against regions with exact distances, "
          "spheres, point sets, and by shared points (vertices, centres) for a negative answer; undecidable "
          "pairs raise NotImplementedError. " + _MEAS_COMMON + "Check: exact; tests compare touching, "
          "overlapping and separated disks and triangles.")
def _region_disjoint(*regs: Expr) -> Boolean:
    f = "RegionDisjoint"
    if len(regs) < 1:
        raise ValueError(f"{f}: give at least one region")
    rs = [region_of(r, f) for r in regs]
    if len({r.edim for r in rs}) > 1:
        raise ValueError(f"{f}: the regions must have the same embedding dimension")
    for a, b in itertools.combinations(rs, 2):
        if not _decide(disjoint(a, b), f, "disjointness"):
            return False
    return True


@fn("RegionEqual", area=AREA, status="added",
    notes="RegionEqual[reg1, reg2, ...] gives True if the regions are the same point set (e.g. Rectangle[] and "
          "the unit-square Polygon, Disk[] and Ball[{0, 0}]), False otherwise; decided by mutual containment "
          "(see RegionWithin), with differing dimension, measure or a counterexample point proving inequality. "
          "Undecidable pairs raise NotImplementedError. " + _MEAS_COMMON + "Check: exact; tests compare "
          "equivalent primitive forms and near misses.")
def _region_equal(*regs: Expr) -> Boolean:
    f = "RegionEqual"
    if len(regs) < 1:
        raise ValueError(f"{f}: give at least one region")
    rs = [region_of(r, f) for r in regs]
    if len({r.edim for r in rs}) > 1:
        return False
    for a, b in zip(rs, rs[1:]):
        try:
            if rdim(a, f) != rdim(b, f):
                return False
            if a.bounded and b.bounded and not eqz(measure_of(a, f) - measure_of(b, f)):
                return False
        except NotImplementedError:
            pass
        w1 = within(a, b)
        if w1 is False:
            return False
        w2 = within(b, a)
        if w2 is False:
            return False
        if w1 is None or w2 is None:
            raise NotImplementedError(f"{f}: cannot decide equality for these regions")
    return True


def shape_of(r):
    """(kind, data) canonical description for congruence tests, or None."""
    if isinstance(r, EllipsoidR) and r.r is not None:
        return ("ball", r.edim, [r.r], None)
    if isinstance(r, SphereR):
        return ("sphere", r.edim, [r.r], None)
    if isinstance(r, ShellR):
        return ("shell", r.edim, [r.r1, r.r2], None)
    if isinstance(r, PointsR):
        return ("set", r.edim, _nz_count(r.P), None)
    if isinstance(r, BoxR) and r.dim == r.edim and r.edim == 2:
        return ("ring", 2, [[r.lo[0], r.lo[1]], [r.hi[0], r.lo[1]], [r.hi[0], r.hi[1]], [r.lo[0], r.hi[1]]], None)
    if isinstance(r, BoxR):
        return ("set", r.edim, r.verts(), "box")
    if isinstance(r, Polygon2R) and len(r.C) == 1 and not r.C[0][1]:
        return ("ring", 2, r.C[0][0], None)
    if isinstance(r, ParallelotopeR) and r.edim == 2:
        return ("ring", 2, [r.p, vadd(r.p, r.V[0]), vadd(vadd(r.p, r.V[0]), r.V[1]), vadd(r.p, r.V[1])], None)
    if isinstance(r, SimplexR) and len(r.S) == 1:
        return ("set", r.edim, r.S[0], "simplex")
    if isinstance(r, PolyhedronR) and r.convex:
        return ("set", 3, r.V, "convex")
    return None


def _find_similarity(P, Q, scale_ok):
    """Is there x -> s R x + t (R orthogonal; s = 1 unless scale_ok) mapping the point set P onto Q?"""
    if len(P) != len(Q):
        return False, None
    n = len(P[0])
    if len(P) == 1:
        return True, None
    k = affine_rank(P)
    if k != affine_rank(Q):
        return False, None
    base = [0]
    for i in range(1, len(P)):
        if affine_rank([P[j] for j in base] + [P[i]]) > len(base) - 1:
            base.append(i)
        if len(base) == k + 1:
            break
    D = sp.Matrix([vsub(P[i], P[base[0]]) for i in base[1:]]).T     # n x k
    Gp = (D.T * D)
    for combo in itertools.permutations(range(len(Q)), k + 1):
        E = sp.Matrix([vsub(Q[i], Q[combo[0]]) for i in combo[1:]]).T
        Gq = (E.T * E)
        if scale_ok:
            s2 = simp(Gq[0, 0] / Gp[0, 0])
        else:
            s2 = sp.Integer(1)
        if not all(eqz(Gq[i, j] - s2 * Gp[i, j]) for i in range(k) for j in range(k)):
            continue
        # linear map on the span: M D = E; extend to any orthogonal complement (distances suffice)
        def img(p):
            c = (Gp.LUsolve(D.T * sp.Matrix(vsub(p, P[base[0]]))))
            resid = sp.Matrix(vsub(p, P[base[0]])) - D * c
            return E * c, simp(n2(list(resid)))
        mapped_ok = True
        used = set()
        for p in P:
            v, rr = img(p)
            if not eqz(rr):
                mapped_ok = False
                break
            q = vadd(Q[combo[0]], list(v))
            j = next((j for j in range(len(Q)) if j not in used and is_zero_vec(vsub(q, Q[j]))), None)
            if j is None:
                mapped_ok = False
                break
            used.add(j)
        if mapped_ok:
            return True, (combo, s2)
    return False, None


def _ring_match(P, Q, scale_ok):
    """Cyclic polygons P, Q (same vertex count): similar / congruent as polygons (vertex order respected)."""
    n = len(P)
    if n != len(Q):
        return False

    def sig(R_, i, rev):
        idx = [(i + (-k if rev else k)) % n for k in range(n)]
        return [R_[j] for j in idx]
    for i in range(n):
        for rev in (False, True):
            Qc = sig(Q, i, rev)
            # all pairwise squared distances must scale uniformly
            s2 = None
            ok = True
            for a in range(n):
                for b in range(a + 1, n):
                    dp, dq = n2(vsub(P[a], P[b])), n2(vsub(Qc[a], Qc[b]))
                    if s2 is None:
                        s2 = simp(dq / dp) if scale_ok else sp.Integer(1)
                    if not eqz(dq - s2 * dp):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def congruent(A, B, scale_ok, f):
    if A.edim != B.edim:
        return False
    try:
        if rdim(A, f) != rdim(B, f):
            return False
    except NotImplementedError:
        pass
    sa, sb = shape_of(A), shape_of(B)
    if sa is None or sb is None:
        raise NotImplementedError(f"{f}: cannot decide for {A.kind} and {B.kind} regions")
    if sa[0] in ("ball", "sphere", "shell") or sb[0] in ("ball", "sphere", "shell"):
        if sa[0] != sb[0]:
            return False
        if scale_ok:
            if len(sa[2]) == 1:
                return True
            return eqz(sa[2][0] * sb[2][1] - sa[2][1] * sb[2][0])
        return all(eqz(x - y) for x, y in zip(sa[2], sb[2]))
    if sa[0] == "ring" and sb[0] == "ring":
        return _ring_match(sa[2], sb[2], scale_ok)
    if sa[0] == "ring" or sb[0] == "ring":
        if {sa[3], sb[3]} <= {None, "simplex"} and len(sa[2]) == len(sb[2]) == 3:
            return _ring_match(sa[2], sb[2], scale_ok)
        PA = sa[2] if sa[0] == "ring" else sb[2]
        if (sa[3] or sb[3]) not in (None,):
            raise NotImplementedError(f"{f}: cannot decide for {A.kind} and {B.kind} regions")
        return False
    if sa[3] != sb[3] and not ({sa[3], sb[3]} <= {"box", "convex"}):
        raise NotImplementedError(f"{f}: cannot decide for {A.kind} and {B.kind} regions")
    ok, _ = _find_similarity(sa[2], sb[2], scale_ok)
    return ok


@fn("RegionCongruent", area=AREA, status="added",
    notes="RegionCongruent[reg1, reg2] gives True if an isometry (rotation, reflection, translation) maps reg1 "
          "onto reg2. Decided exactly for balls, spheres, shells, finite point sets, simplices, simple polygons "
          "(vertex cycles compared up to rotation and reversal), boxes and convex polyhedra (vertex sets, which "
          "determine a convex body); other pairs raise NotImplementedError. Check: exact; tests compare rotated "
          "and reflected triangles, squares versus rhombi and balls.")
def _region_congruent(reg1: Expr, reg2: Expr) -> Boolean:
    f = "RegionCongruent"
    return congruent(region_of(reg1, f), region_of(reg2, f), False, f)


@fn("RegionSimilar", area=AREA, status="added",
    notes="RegionSimilar[reg1, reg2] gives True if a similarity (isometry composed with a uniform scaling) maps "
          "reg1 onto reg2; same supported regions as RegionCongruent. Check: exact; tests compare scaled "
          "triangles, rectangles of different aspect ratios and any two disks.")
def _region_similar(reg1: Expr, reg2: Expr) -> Boolean:
    f = "RegionSimilar"
    return congruent(region_of(reg1, f), region_of(reg2, f), True, f)


# ───────────────────────────── region constructors (inert, validated) ─────────────────────────────

def _validate(x, f):
    try:
        return region_of(x, f)
    except NotImplementedError:
        return None


@fn("ImplicitRegion", area=AREA, status="added",
    notes="ImplicitRegion[cond, {x1, ..., xn}] represents the set of points satisfying cond ({x, xmin, xmax} "
          "entries add bounds). Inert after validation: the variables must be distinct symbols (else TypeError "
          "/ ValueError), cond a condition. Region functions reduce linear inequality systems in up to 3 "
          "variables to exact polytopes, a single quadratic inequality with positive definite form to an "
          "ellipsoid, and 1-D conditions to interval unions (exact solveset); membership works for any "
          "condition. Check: exact; tests compare Area / RegionMember / RegionBounds with the equivalent "
          "primitives.")
def _implicit_region(cond: Expr, vars: ListOf[Expr] | Expr) -> Expr:
    implicit_of(Expr("ImplicitRegion", cond, vars), "ImplicitRegion")
    raise Unevaluated()


@fn("ParametricRegion", area=AREA, status="added",
    notes="ParametricRegion[{f1, ..., fn}, {{u, umin, umax}, ...}] represents the image of the parameter box. "
          "Inert after validation (distinct symbol parameters, ordered numeric ranges, coordinate expressions). "
          "Affine parametrizations reduce to exact parallelotopes / segments / points; otherwise "
          "RegionDimension gives the generic Jacobian rank, RegionEmbeddingDimension n, and other region "
          "functions raise NotImplementedError (the image measure needs injectivity; use the parametric forms "
          "of RegionMeasure / Area / Volume). Check: exact; tests compare affine images with primitives.")
def _parametric_region(fs: ListOf[Expr], vars: ListOf[Expr]) -> Expr:
    parametric_of(Expr("ParametricRegion", fs, vars), "ParametricRegion")
    raise Unevaluated()


@fn("Region", area=AREA, status="added",
    notes="Region[reg] (with optional graphics options) is a display wrapper: region functions treat it as reg. "
          "Inert after validation (the argument must be a region: TypeError otherwise). Check: delegated; tests "
          "compare RegionMeasure[Region[reg]] with RegionMeasure[reg].")
def _region(reg: Expr, *opts: Option) -> Expr:
    if not _looks_region(reg):
        if isinstance(reg, Symbol):
            raise Unevaluated()
        raise TypeError(f"Region: {K_input(reg)} is not a geometric region")
    _validate(reg, "Region")
    raise Unevaluated()


def _bool_ctor(name, regs, single=True):
    f = name
    if not regs:
        raise ValueError(f"{f}: give at least one region")
    rs = []
    for r in regs:
        if not _looks_region(r):
            if isinstance(r, Symbol):
                raise Unevaluated()
            raise TypeError(f"{f}: {K_input(r)} is not a geometric region")
        rs.append(_validate(r, f))
    eds = {r.edim for r in rs if r is not None}
    if len(eds) > 1:
        raise ValueError(f"{f}: all regions must have the same embedding dimension")
    if len(regs) == 1 and single:
        return regs[0]
    raise Unevaluated()


@fn("RegionUnion", area=AREA, status="added",
    notes="RegionUnion[reg1, reg2, ...] represents the union of regions (RegionUnion[reg] is reg). Inert after "
          "validation (regions of one embedding dimension, else ValueError; non-regions TypeError). Region "
          "functions use it exactly: membership, distance, nearest point and bounds always; measures and "
          "integrals when the parts are proved disjoint or nested. Check: exact; tests compare the measure of "
          "disjoint and nested unions with sums of parts.")
def _region_union(*regs: Expr) -> Expr:
    return _bool_ctor("RegionUnion", regs)


@fn("RegionIntersection", area=AREA, status="added",
    notes="RegionIntersection[reg1, reg2, ...] represents the intersection of regions. Inert after validation; "
          "region functions resolve it exactly for convex polytopes (exact half-space intersection), nested "
          "regions and provably disjoint parts (EmptyRegion), and use membership otherwise. Check: exact; tests "
          "compare the area of overlapping squares with the hand value.")
def _region_intersection(*regs: Expr) -> Expr:
    return _bool_ctor("RegionIntersection", regs)


@fn("RegionDifference", area=AREA, status="added",
    notes="RegionDifference[reg1, reg2] represents {x in reg1 : x not in reg2}. Inert after validation (two "
          "regions of the same embedding dimension); measures resolve when reg2 is inside reg1, disjoint from "
          "it or lower-dimensional. Check: exact; tests compare Area[RegionDifference[Disk[], Disk[{0, 0}, "
          "1/2]]] = 3 Pi / 4.")
def _region_difference(reg1: Expr, reg2: Expr) -> Expr:
    return _bool_ctor("RegionDifference", (reg1, reg2), single=False)


@fn("RegionSymmetricDifference", area=AREA, status="added",
    notes="RegionSymmetricDifference[reg1, reg2, ...] represents points in an odd number of the regions. Inert "
          "after validation; membership is exact, measures resolve for disjoint pairs. Check: exact; tests "
          "compare membership in overlap and non-overlap parts.")
def _region_symmetric_difference(*regs: Expr) -> Expr:
    return _bool_ctor("RegionSymmetricDifference", regs)


@fn("BooleanRegion", area=AREA, status="added",
    notes="BooleanRegion[bfun, {reg1, ..., regn}] represents the points x for which bfun[x in reg1, ...] is "
          "True (e.g. Xor, Or, And[#1, Not[#2]] &). Inert after validation; RegionMember applies bfun exactly "
          "(a non-Boolean result raises ValueError); other functions need equivalent RegionUnion etc. Check: "
          "exact; tests compare membership with the Boolean formula.")
def _boolean_region(bfun: Fn, regs: ListOf[Expr]) -> Expr:
    f = "BooleanRegion"
    if not K.is_list(regs):
        raise TypeError(f"{f}: the regions must be a list {{reg1, ...}}")
    _bool_ctor(f, list(regs.args), single=False)
    raise Unevaluated()


@fn("TransformedRegion", area=AREA, status="added",
    notes="TransformedRegion[reg, f] represents the image of reg under f. Inert after validation. With an "
          "affine f (TransformationFunction, matrix or {m, v}) region functions use the exact image: polytopes "
          "map vertex-wise, balls become ellipsoids, boxes parallelotopes, half-spaces map their normal, and "
          "other regions use |Det[m]| for integrals and the inverse map for membership (distances need a "
          "similarity). Projective maps raise NotImplementedError; other functions keep it inert. Check: "
          "exact; tests compare Area of rotated and sheared regions with |det| times the area.")
def _transformed_region(reg: Expr, t: Expr) -> Expr:
    f = "TransformedRegion"
    if not _looks_region(reg) and not isinstance(reg, Symbol):
        raise TypeError(f"{f}: {K_input(reg)} is not a geometric region")
    _validate(Expr(f, reg, t), f)
    raise Unevaluated()


@fn("InverseTransformedRegion", area=AREA, status="added",
    notes="InverseTransformedRegion[reg, f] (or [reg, f, n] with points in R^n) represents {x : f[x] in reg}. "
          "Inert after validation. Membership applies f for any function; affine invertible f gives the exact "
          "image under f^-1 for the other region functions. Check: exact; tests compare membership and "
          "RegionMeasure with the hand inverse image.")
def _inverse_transformed_region(reg: Expr, t: Expr, *n: PosInteger) -> Expr:
    f = "InverseTransformedRegion"
    if not _looks_region(reg) and not isinstance(reg, Symbol):
        raise TypeError(f"{f}: {K_input(reg)} is not a geometric region")
    _validate(Expr(f, reg, t, *n), f)
    raise Unevaluated()


@fn("RegionProduct", area=AREA, status="added",
    notes="RegionProduct[reg1, reg2, ...] represents the Cartesian product (embedding dimensions add). Inert "
          "after validation (RegionProduct[reg] is reg); measures and polynomial integrals factor exactly, "
          "membership and distances split by coordinate blocks. Check: exact; tests compare "
          "Volume[RegionProduct[Disk[], Line[{{0}, {2}}]]] = 2 Pi.")
def _region_product(*regs: Expr) -> Expr:
    f = "RegionProduct"
    if not regs:
        raise ValueError(f"{f}: give at least one region")
    for r in regs:
        if not _looks_region(r):
            if isinstance(r, Symbol):
                raise Unevaluated()
            raise TypeError(f"{f}: {K_input(r)} is not a geometric region")
        _validate(r, f)
    if len(regs) == 1:
        return regs[0]
    raise Unevaluated()


@fn("CSGRegion", area=AREA, status="added",
    notes="CSGRegion[\"op\", {reg1, reg2, ...}] represents a constructive solid geometry combination with op one "
          "of \"Union\", \"Intersection\", \"Difference\", \"SymmetricDifference\" (operands may be CSGRegion "
          "trees); CSGRegion[reg] wraps a single region. Inert after validation (unknown op ValueError, "
          "non-string op or non-regions TypeError, mixed embedding dimensions ValueError); region functions "
          "use it like the corresponding RegionUnion etc. Check: exact; tests compare measures with the Boolean "
          "region forms.")
def _csg_region(op: String | Expr, *regs: ListOf[Expr]) -> Expr:
    f = "CSGRegion"
    x = Expr(f, op, *regs)
    o, parts = csg_parts(x, f)
    for r in parts:
        if not _looks_region(r):
            if isinstance(r, Symbol):
                raise Unevaluated()
            raise TypeError(f"{f}: {K_input(r)} is not a geometric region")
    _validate(x, f)
    raise Unevaluated()


@fn("CSGRegionQ", area=AREA, status="added",
    notes="CSGRegionQ[reg] gives True if reg is a valid CSGRegion and False otherwise. Check: exact; tests "
          "compare valid, invalid and non-CSG regions.")
def _csg_region_q(reg: Expr) -> Boolean:
    if not (isinstance(reg, Expr) and reg.head == "CSGRegion"):
        return False
    try:
        o, parts = csg_parts(reg, "CSGRegionQ")
        for r in parts:
            if not _looks_region(r):
                return False
            region_of(r, "CSGRegionQ")
        return True
    except (NotImplementedError, Unevaluated):
        return True
    except (TypeError, ValueError):
        return False


def _csg_tree(x, f):
    if isinstance(x, Expr) and x.head == "CSGRegion":
        o, parts = csg_parts(x, f)
        return Expr("Tree", o, L(*[_csg_tree(p, f) for p in parts]))
    if not _looks_region(x):
        raise TypeError(f"{f}: {K_input(x)} is not a geometric region")
    return Expr("Tree", x, Symbol("None"))


@fn("CSGRegionTree", area=AREA, status="added",
    notes="CSGRegionTree[csg] gives the operation tree of a CSGRegion as a Tree: internal nodes carry the "
          "operation name, leaves Tree[reg, None]. A non-CSG region raises TypeError. Check: exact; tests "
          "compare nested trees.")
def _csg_region_tree(csg: Expr) -> Expr:
    f = "CSGRegionTree"
    if not (isinstance(csg, Expr) and csg.head == "CSGRegion"):
        if isinstance(csg, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: {K_input(csg)} is not a CSGRegion")
    return _csg_tree(csg, f)


# ───────────────────────────── derived regions ─────────────────────────────

def _ball_expr(c, r, inputs):
    if len(c) == 2:
        return Expr("Disk", out(c, *inputs), out(r, *inputs))
    return Expr("Ball", out(c, *inputs), out(r, *inputs))


def _radius_arg(r, f, what="radius"):
    v = sx(r, f, what)
    concrete(v)
    if sgn(v) < 0:
        raise ValueError(f"{f}: the {what} must be non-negative")
    return v


@fn("RegionDilation", area=AREA, status="added",
    notes="RegionDilation[reg, r] gives the Minkowski sum of reg with a ball of radius r (all points within r "
          "of reg): Disk / Ball grow their radius, points become balls (a point set a RegionUnion of balls), a "
          "segment becomes StadiumShape (2-D) or CapsuleShape (3-D), StadiumShape / CapsuleShape grow, a 1-D "
          "interval widens, a sphere becomes a SphericalShell / Annulus. r = 0 gives reg; negative r raises "
          "ValueError; other regions (polygons with rounded corners) and other kernels raise "
          "NotImplementedError. Check: exact; tests compare the measures with Steiner's formula (area + "
          "perimeter r + Pi r^2).")
def _region_dilation(reg: Expr, r: Real) -> Expr:
    f = "RegionDilation"
    R_ = region_of(reg, f)
    if K.is_list(r) or _looks_region(r):
        raise NotImplementedError(f"{f}: dilation by a general kernel region is not supported")
    rv = _radius_arg(r, f)
    if eqz(rv):
        return reg
    ins = (reg, r)
    if isinstance(R_, PointsR):
        P = _nz_count(R_.P)
        if R_.edim == 1:
            return Expr("RegionUnion", *[Expr("Line", out([[p[0] - rv], [p[0] + rv]], *ins)) for p in P]) \
                if len(P) > 1 else Expr("Line", out([[P[0][0] - rv], [P[0][0] + rv]], *ins))
        balls = [_ball_expr(p, rv, ins) for p in P]
        return balls[0] if len(balls) == 1 else Expr("RegionUnion", *balls)
    b = _ball_of(R_)
    if b is not None and isinstance(R_, EllipsoidR):
        return _ball_expr(b[0], simp(b[1] + rv), ins)
    if isinstance(R_, SphereR):
        h = "Annulus" if R_.edim == 2 else "SphericalShell"
        if le(R_.r, rv):
            return _ball_expr(R_.c, simp(R_.r + rv), ins)
        return Expr(h, out(R_.c, *ins), out([simp(R_.r - rv), simp(R_.r + rv)], *ins))
    if isinstance(R_, SimplexR) and R_.dim == 1 and len(R_.S) == 1:
        a, b2 = R_.S[0]
        if R_.edim == 1:
            lo, hi = smin([a[0], b2[0]]), smax([a[0], b2[0]])
            return Expr("Line", out([[lo - rv], [hi + rv]], *ins))
        if R_.edim in (2, 3):
            return Expr("StadiumShape" if R_.edim == 2 else "CapsuleShape", out([a, b2], *ins), out(rv, *ins))
    if isinstance(R_, BoxR) and R_.edim == 1:
        return Expr("Line", out([[R_.lo[0] - rv], [R_.hi[0] + rv]], *ins))
    if isinstance(R_, RevolutionR) and R_.kind in ("StadiumShape", "CapsuleShape"):
        return Expr(R_.kind, out([R_.p1, R_.p2], *ins), out(simp(R_.r + rv), *ins))
    if isinstance(R_, ShellR):
        return Expr("Annulus" if R_.edim == 2 else "SphericalShell", out(R_.c, *ins),
                    out([smax([0, R_.r1 - rv]), R_.r2 + rv], *ins)) if lt(rv, R_.r1) else _ball_expr(R_.c, R_.r2 + rv,
                                                                                                   ins)
    if isinstance(R_, EmptyR):
        return reg
    raise NotImplementedError(f"{f}: dilation of {R_.kind} regions is not supported")


@fn("RegionErosion", area=AREA, status="added",
    notes="RegionErosion[reg, r] gives the points whose ball of radius r lies inside reg: Disk / Ball shrink "
          "(a Point when r equals the radius, EmptyRegion beyond), Rectangle / Cuboid / 1-D intervals shrink "
          "by r on every side, StadiumShape / CapsuleShape shrink to the axis segment, spherical shells and "
          "annuli thin; lower-dimensional regions erode to EmptyRegion. r = 0 gives reg; negative r raises "
          "ValueError; other regions raise NotImplementedError. Check: exact; tests compare with "
          "RegionDilation round trips and hand results.")
def _region_erosion(reg: Expr, r: Real) -> Expr:
    f = "RegionErosion"
    R_ = region_of(reg, f)
    if K.is_list(r):
        raise NotImplementedError(f"{f}: erosion by a general kernel region is not supported")
    rv = _radius_arg(r, f)
    if eqz(rv):
        return reg
    ins = (reg, r)
    empty = Expr("EmptyRegion", R_.edim)
    if isinstance(R_, EmptyR):
        return reg
    if rdim(R_, f) < R_.edim:
        return empty
    b = _ball_of(R_)
    if isinstance(R_, EllipsoidR) and b is not None:
        d = simp(b[1] - rv)
        if sgn(d) < 0:
            return empty
        if eqz(d):
            return Expr("Point", out(b[0], *ins))
        return _ball_expr(b[0], d, ins)
    if isinstance(R_, BoxR):
        lo = [simp(c + rv) for c in R_.lo]
        hi = [simp(c - rv) for c in R_.hi]
        if any(lt(h, l_) for l_, h in zip(lo, hi)):
            return empty
        if R_.edim == 1:
            return Expr("Line", out([lo, hi], *ins)) if not eqz(hi[0] - lo[0]) else Expr("Point", out(lo, *ins))
        return Expr("Rectangle" if R_.edim == 2 else "Cuboid", out(lo, *ins), out(hi, *ins))
    if isinstance(R_, SimplexR) and R_.edim == 1 and len(R_.S) == 1:
        lo, hi = smin([R_.S[0][0][0], R_.S[0][1][0]]) + rv, smax([R_.S[0][0][0], R_.S[0][1][0]]) - rv
        if lt(hi, lo):
            return empty
        return Expr("Line", out([[lo], [hi]], *ins)) if not eqz(hi - lo) else Expr("Point", out([lo], *ins))
    if isinstance(R_, RevolutionR) and R_.kind in ("StadiumShape", "CapsuleShape"):
        d = simp(R_.r - rv)
        if sgn(d) < 0:
            return empty
        if eqz(d):
            return Expr("Line", out([R_.p1, R_.p2], *ins))
        return Expr(R_.kind, out([R_.p1, R_.p2], *ins), out(d, *ins))
    if isinstance(R_, ShellR):
        a, b2 = simp(R_.r1 + rv), simp(R_.r2 - rv)
        if lt(b2, a):
            return empty
        if eqz(b2 - a):
            return Expr("Circle" if R_.edim == 2 else "Sphere", out(R_.c, *ins), out(a, *ins))
        return Expr("Annulus" if R_.edim == 2 else "SphericalShell", out(R_.c, *ins), out([a, b2], *ins))
    raise NotImplementedError(f"{f}: erosion of {R_.kind} regions is not supported")


def _side_specs(spec, n, f):
    """RegionResize side specification -> list of (kind, value) per dimension."""
    if K.is_list(spec) and spec.args and all(K.is_list(s) and len(s.args) == 2 for s in spec.args):
        if len(spec.args) != n:
            raise ValueError(f"{f}: give {n} coordinate ranges")
        res = []
        for s in spec.args:
            a, b = sx(s.args[0], f, "range bound"), sx(s.args[1], f, "range bound")
            concrete(a, b)
            if lt(b, a):
                raise ValueError(f"{f}: each range must have min <= max")
            res.append((a, b))
        return "box", res
    if K.is_list(spec) and len(spec.args) == 1 and not K.is_list(spec.args[0]):
        v = sx(spec.args[0], f, "maximum side length")
        concrete(v)
        if sgn(v) <= 0:
            raise ValueError(f"{f}: side lengths must be positive")
        return "max", v
    items = list(spec.args) if K.is_list(spec) else [spec] + [Symbol("Automatic")] * (n - 1)
    if len(items) != n:
        raise ValueError(f"{f}: give {n} side lengths")
    res = []
    for it in items:
        if isinstance(it, Symbol) and it.name in ("Automatic", "All"):
            res.append((it.name, None))
        elif isinstance(it, Expr) and it.head == "Scaled" and len(it.args) == 1:
            v = sx(it.args[0], f, "scale")
            concrete(v)
            if sgn(v) <= 0:
                raise ValueError(f"{f}: scale factors must be positive")
            res.append(("Scaled", v))
        elif K.is_list(it) and len(it.args) == 1:
            v = sx(it.args[0], f, "side length")
            concrete(v)
            res.append(("AtMost", v))
        else:
            v = sx(it, f, "side length")
            concrete(v)
            if sgn(v) <= 0:
                raise ValueError(f"{f}: side lengths must be positive")
            res.append(("num", v))
    return "sides", res


@fn("RegionResize", area=AREA, status="added",
    notes="RegionResize[reg, l] rescales reg so its first side has length l keeping side ratios; "
          "RegionResize[reg, {lmax}] so the largest side is lmax; RegionResize[reg, {l1, l2, ...}] to the given "
          "side lengths (entries may be Automatic, All, Scaled[s] or {at most}); RegionResize[reg, {{x1min, "
          "x1max}, ...}] into that box. The rescaling is about the lower corner of RegionBounds (that corner "
          "stays fixed unless a target box is given). The result is a TransformedRegion with a "
          "RescalingTransform-equivalent matrix, simplified to the primitive when exact (boxes, balls with "
          "uniform scale, polytopes). Zero-width directions cannot be stretched (ValueError); unbounded regions "
          "raise ValueError. Check: exact; tests compare RegionBounds of the result with the requested box.")
def _region_resize(reg: Expr, spec: Real | ListOf[Expr]) -> Expr:
    f = "RegionResize"
    R_ = region_of(reg, f)
    if not R_.bounded:
        raise ValueError(f"{f}: the region must be bounded")
    B = R_.bounds()
    n = R_.edim
    w = [simp(b - a) for a, b in B]
    kind, sp_ = _side_specs(spec, n, f)
    if kind == "box":
        tgt = sp_
        scale = []
        for i in range(n):
            if eqz(w[i]):
                if not eqz(tgt[i][1] - tgt[i][0]):
                    raise ValueError(f"{f}: cannot stretch a zero-width direction")
                scale.append(sp.Integer(1))
            else:
                scale.append(simp((tgt[i][1] - tgt[i][0]) / w[i]))
        lo_new = [t[0] for t in tgt]
    elif kind == "max":
        m = smax(w)
        if eqz(m):
            raise ValueError(f"{f}: cannot resize a single point")
        scale = [simp(sp_ / m)] * n
        lo_new = [a for a, b in B]
    else:
        first = None
        scale = [None] * n
        for i, (k_, v) in enumerate(sp_):
            if k_ == "num":
                if eqz(w[i]):
                    raise ValueError(f"{f}: cannot stretch a zero-width direction")
                scale[i] = simp(v / w[i])
            elif k_ == "Scaled":
                scale[i] = v
            elif k_ == "All":
                scale[i] = sp.Integer(1)
            elif k_ == "AtMost":
                scale[i] = ("AtMost", v)
        fixed = [s for s in scale if s is not None and not isinstance(s, tuple)]
        auto = fixed[0] if fixed else None
        for i in range(n):
            if isinstance(scale[i], tuple):
                v = scale[i][1]
                cand = auto if auto is not None else sp.Integer(1)
                scale[i] = cand if eqz(w[i]) or le(cand * w[i], v) else simp(v / w[i])
            elif scale[i] is None:
                scale[i] = auto if auto is not None else sp.Integer(1)
        lo_new = [a for a, b in B]
    A = sp.diag(*scale)
    bvec = [simp(lo_new[i] - scale[i] * B[i][0]) for i in range(n)]
    return _affine_result(reg, R_, A, bvec, f, (reg, spec))


def _affine_result(reg, R_, A, b, f, ins):
    """Exact primitive for the image of reg under x -> A x + b where possible, else TransformedRegion."""
    n = R_.edim
    img = affine_map(R_, A, b, f)
    e = region_expr(img, ins)
    if e is not None:
        return e
    return Expr("TransformedRegion", reg, tf_expr(homog(A, b), *ins))


def region_expr(r, ins):
    """Expression for a region object when it is a simple primitive, else None."""
    if isinstance(r, PointsR):
        return Expr("Point", out(r.P[0] if len(r.P) == 1 else r.P, *ins))
    if isinstance(r, BoxR):
        if r.edim == 2:
            return Expr("Rectangle", out(r.lo, *ins), out(r.hi, *ins))
        if r.edim == 1:
            return Expr("Line", out([r.lo, r.hi], *ins))
        return Expr("Cuboid", out(r.lo, *ins), out(r.hi, *ins))
    if isinstance(r, EllipsoidR):
        if r.r is not None:
            return _ball_expr(r.c, r.r, ins)
        if r.Lm.is_diagonal():
            if r.edim == 2:
                return Expr("Disk", out(r.c, *ins), out([r.Lm[i, i] for i in range(2)], *ins))
            return Expr("Ellipsoid", out(r.c, *ins), out([r.Lm[i, i] for i in range(r.edim)], *ins))
        Sg = (r.Lm * r.Lm.T).applyfunc(simp)
        return Expr("Ellipsoid", out(r.c, *ins), out(Sg.tolist(), *ins))
    if isinstance(r, SphereR):
        return Expr("Circle" if r.edim == 2 else "Sphere", out(r.c, *ins), out(r.r, *ins))
    if isinstance(r, ShellR):
        return Expr("Annulus" if r.edim == 2 else "SphericalShell", out(r.c, *ins), out([r.r1, r.r2], *ins))
    if isinstance(r, SimplexR) and len(r.S) == 1:
        s = r.S[0]
        if len(s) == 2:
            return Expr("Line", out(s, *ins))
        if len(s) == 3 and r.edim == 2:
            return Expr("Triangle", out(s, *ins))
        return Expr("Simplex", out(s, *ins))
    if isinstance(r, Polygon2R) or isinstance(r, Polygon3R):
        comps = r.C
        def one(o, hs):
            return out(o, *ins) if not hs else Expr("Rule", out(o, *ins), out(hs, *ins))
        return Expr("Polygon", one(*comps[0]) if len(comps) == 1 else L(*[one(o, hs) for o, hs in comps]))
    if isinstance(r, ParallelotopeR):
        return Expr("Parallelogram" if r.edim == 2 and r.dim == 2 else "Parallelepiped", out(r.p, *ins),
                    out(r.V, *ins))
    if isinstance(r, PolyhedronR) and not r.voids:
        return Expr("Polyhedron", out(r.V, *ins), out([[i + 1 for i in fc] for fc in r.F], *ins))
    if isinstance(r, RevolutionR):
        return Expr(r.kind, out([r.p1, r.p2], *ins), out(r.r, *ins))
    if isinstance(r, EmptyR):
        return Expr("EmptyRegion", r.edim)
    return None


# ───────────────────────────── BoundingRegion ─────────────────────────────

BOUNDING_FORMS = ("MinCuboid", "MinRectangle", "MinInterval", "MinBall", "MinDisk", "MinConvexPolygon",
                  "MinConvexPolyhedron", "MinOrientedRectangle", "MinOrientedCuboid", "MinOrientedInterval")
FAST_FORMS = ("FastOrientedCuboid", "FastOrientedRectangle", "FastBall", "FastDisk")


def _exact_min_ball(P, f):
    """Smallest enclosing ball of exact points: numeric Welzl support, exact circumsphere, exact verification."""
    import numpy as np
    P = _nz_count(P)
    if len(P) == 1:
        return P[0], sp.Integer(0)
    Pn = np.array([[float(num(c)) for c in p] for p in P])
    support = sorted(G._min_ball_numeric(Pn))
    cands = []
    n = len(P[0])
    for k in range(2, min(len(support), n + 1) + 1):
        for sub in itertools.combinations(support, k):
            cands.append(sub)
    best = None
    for sub in cands:
        S_ = [P[i] for i in sub]
        if affine_rank(S_) != len(S_) - 1:
            continue
        c = _circumcentre(S_)
        r2 = simp(n2(vsub(S_[0], c)))
        if best is not None and not lt(r2, best[1]):
            continue
        if all(le(n2(vsub(p, c)), r2) for p in P):
            w, _ = proj_affine(c, S_)
            if all(sgn(x) >= 0 for x in w):
                best = (c, r2)
    if best is None:
        raise ArithmeticError(f"{f}: the minimal enclosing ball failed verification")
    return best[0], simp(sp.sqrt(best[1]))


def _circumcentre(S_):
    """Centre of the sphere through the affinely independent points S_ within their affine hull (exact)."""
    p0 = S_[0]
    D = [vsub(q, p0) for q in S_[1:]]
    A = sp.Matrix([[2 * dot(a, b) for b in D] for a in D])
    rhs = sp.Matrix([n2(a) for a in D])
    lam = A.LUsolve(rhs)
    return [simp(p0[i] + sum(lam[j] * D[j][i] for j in range(len(D)))) for i in range(len(p0))]


def _min_area_rect(P, f):
    """Minimal-area enclosing rectangle of 2-D points (exact rotating calipers): (origin, v1, v2)."""
    idx = convex_hull_2d(P)
    H = [P[i] for i in idx]
    if len(H) < 3:
        raise ValueError(f"{f}: the points are collinear")
    best = None
    for i in range(len(H)):
        a, b = H[i], H[(i + 1) % len(H)]
        e = vsub(b, a)
        L2 = n2(e)
        nrm = [-e[1], e[0]]
        s = [dot(vsub(p, a), e) for p in H]
        t = [dot(vsub(p, a), nrm) for p in H]
        s0, s1, t0, t1 = smin(s), smax(s), smin(t), smax(t)
        area = simp((s1 - s0) * (t1 - t0) / L2)
        if best is None or lt(area, best[0]):
            o = vadd(a, vadd(vmul(s0 / L2, e), vmul(t0 / L2, nrm)))
            best = (area, [simp(c) for c in o], [simp(c) for c in vmul((s1 - s0) / L2, e)],
                    [simp(c) for c in vmul((t1 - t0) / L2, nrm)])
    return best[1:]


@fn("BoundingRegion", area=AREA, status="added",
    notes="BoundingRegion[pts] / BoundingRegion[reg] gives the smallest axis-aligned box (\"MinCuboid\", the "
          "default; Rectangle in 2-D, Cuboid otherwise; \"MinRectangle\" and \"MinInterval\" are its 2-D and "
          "1-D names). BoundingRegion[x, form] with \"MinBall\" / \"MinDisk\" (smallest enclosing ball: "
          "numeric Welzl support set, exact circumsphere, exact containment and optimality verification, else "
          "ArithmeticError), \"MinConvexPolygon\" / \"MinConvexPolyhedron\" (exact convex hull), "
          "\"MinOrientedRectangle\" (exact rotating calipers, a Parallelogram) and \"MinOrientedInterval\". "
          "Regions use their vertices (polytopes) or exact shape (balls); \"Fast*\" heuristics and "
          "\"MinOrientedCuboid\" raise NotImplementedError; unknown forms ValueError. Check: self-check "
          "(containment of every point, KKT optimality for balls).")
def _bounding_region(x: MatrixOf[Real] | Expr, *form: String) -> Expr:
    f = "BoundingRegion"
    if len(form) > 1:
        raise ValueError(f"{f}: expected BoundingRegion[pts, form]")
    fm = form[0] if form else "MinCuboid"
    if not isinstance(fm, str):
        raise TypeError(f"{f}: the form must be a string such as \"MinBall\"")
    if fm in FAST_FORMS or fm == "MinOrientedCuboid":
        raise NotImplementedError(f"{f}: the form {fm!r} is not supported")
    if fm not in BOUNDING_FORMS:
        raise ValueError(f"{f}: unknown form {fm!r}; use one of {', '.join(BOUNDING_FORMS)}")
    ins = (x,)
    if K.is_list(x):
        P = pts(x, f)
        concrete(P)
        r = PointsR(P)
    else:
        r = region_of(x, f)
        if not r.bounded:
            raise ValueError(f"{f}: the region is unbounded")
        if isinstance(r, EmptyR):
            raise ValueError(f"{f}: the empty region has no bounding region")
    n = r.edim
    if fm in ("MinCuboid", "MinRectangle", "MinInterval"):
        B = r.bounds()
        lo, hi = [a for a, b in B], [b for a, b in B]
        if fm == "MinRectangle" and n != 2 or fm == "MinInterval" and n != 1:
            raise ValueError(f"{f}: {fm!r} needs {2 if fm == 'MinRectangle' else 1}-dimensional input")
        if n == 2:
            return Expr("Rectangle", out(lo, *ins), out(hi, *ins))
        if n == 1:
            return Expr("Line", out([lo, hi], *ins))
        return Expr("Cuboid", out(lo, *ins), out(hi, *ins))
    if fm in ("MinBall", "MinDisk"):
        if fm == "MinDisk" and n != 2:
            raise ValueError(f"{f}: \"MinDisk\" needs 2-dimensional input")
        b = _ball_of(r) or ((r.c, r.r) if isinstance(r, SphereR) else None)
        if b is None:
            V = r.verts()
            if V is None:
                raise NotImplementedError(f"{f}: {fm!r} for {r.kind} regions is not supported")
            b = _exact_min_ball(V, f)
        return _ball_expr(b[0], b[1], ins)
    V = r.verts()
    if V is None:
        raise NotImplementedError(f"{f}: {fm!r} for {r.kind} regions is not supported")
    V = _nz_count(V)
    if fm == "MinOrientedInterval":
        if n != 1:
            raise ValueError(f"{f}: \"MinOrientedInterval\" needs 1-dimensional input")
        return Expr("Line", out([[smin([v[0] for v in V])], [smax([v[0] for v in V])]], *ins))
    if fm == "MinConvexPolygon":
        if n != 2:
            raise ValueError(f"{f}: \"MinConvexPolygon\" needs 2-dimensional input")
        idx = convex_hull_2d(V)
        if len(idx) < 3:
            raise ValueError(f"{f}: the points are collinear")
        return Expr("Polygon", out([V[i] for i in idx], *ins))
    if fm == "MinConvexPolyhedron":
        if n != 3:
            raise ValueError(f"{f}: \"MinConvexPolyhedron\" needs 3-dimensional input")
        if affine_rank(V) < 3:
            raise ValueError(f"{f}: the points are coplanar")
        vs, fs = convex_hull_3d(V, f)
        keep = sorted(vs)
        pos = {k: i + 1 for i, k in enumerate(keep)}
        return Expr("Polyhedron", out([V[k] for k in keep], *ins), out([[pos[i] for i in fc] for fc in fs], *ins))
    if n != 2:
        raise ValueError(f"{f}: \"MinOrientedRectangle\" needs 2-dimensional input")
    o, v1, v2 = _min_area_rect(V, f)
    return Expr("Parallelogram", out(o, *ins), out([v1, v2], *ins))


# ───────────────────────────── CanonicalizeRegion / RegionConvert / ShellRegion ─────────────────────────────

def canonical(x, f):
    r = region_of(x, f)
    ins = (x,)
    h = x.head
    if isinstance(r, PointsR):
        P = _nz_count(r.P)
        return Expr("Point", out(P[0] if len(P) == 1 else P, *ins))
    if isinstance(r, BoxR):
        if r.dim == r.edim:
            return region_expr(r, ins) if h in ("Rectangle", "Cuboid") else x
        V = _nz_count(r.verts())
        if r.dim == 0:
            return Expr("Point", out(V[0], *ins))
        if r.dim == 1:
            return Expr("Line", out([r.lo, r.hi], *ins))
        if r.dim == 2 and r.edim == 3:
            fr = r.free
            a, b = fr
            p = list(r.lo)
            ring = []
            for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
                q = list(p)
                q[a] = r.hi[a] if da else r.lo[a]
                q[b] = r.hi[b] if db else r.lo[b]
                ring.append(q)
            return Expr("Polygon", out(ring, *ins))
        return x
    return x


def _canon_degenerate(x, f):
    """Degenerate shapes Wolfram reduces to lower-dimensional ones; None when x is not degenerate."""
    h = x.head
    a, _ = split_trailing_options(list(x.args))
    Z = sp.Integer(0)
    if h in ("Disk", "Ball", "Circle", "Sphere") and len(a) == 2 and not K.is_list(a[1]) and not K.is_int(a[0]):
        r = sx(a[1], f, "radius")
        concrete(r)
        if eqz(r):
            return Expr("Point", a[0])
    if h == "Line":
        chains = _lines(a[0], f)
        concrete(chains)
        cl = []
        for c in chains:
            d = [c[0]] + [c[i] for i in range(1, len(c)) if not is_zero_vec(vsub(c[i], c[i - 1]))]
            cl.append(d)
        if all(len(c) == 1 for c in cl):
            P = _nz_count([c[0] for c in cl])
            return Expr("Point", out(P[0] if len(P) == 1 else P, x))
        cl = [c for c in cl if len(c) > 1]
        return Expr("Line", out(cl[0] if len(cl) == 1 else cl, x))
    if h in ("Triangle", "Simplex") and a and K.is_list(a[0]) and a[0].args and K.is_list(a[0].args[0]):
        P = pts(a[0], f)
        concrete(P)
        k = affine_rank(P)
        if k < len(P) - 1:
            if k == 0:
                return Expr("Point", out(P[0], x))
            if k == 1:
                return Expr("Line", out(_extreme_pair(P), x))
            hull = convex_hull_region(P, f)
            e = region_expr(hull, (x,))
            return e if e is not None else x
    if h == "Annulus" or h == "SphericalShell":
        r = region_of(x, f)
        if isinstance(r, ShellR):
            if eqz(r.r1):
                return _ball_expr(r.c, r.r2, (x,)) if r.edim == 2 or h == "SphericalShell" else x
            if eqz(r.r2 - r.r1):
                return Expr("Circle" if r.edim == 2 else "Sphere", out(r.c, x), out(r.r1, x))
    return None


@fn("CanonicalizeRegion", area=AREA, status="added",
    notes="CanonicalizeRegion[reg] gives a canonical form of reg: duplicate points removed (a single Point[p] "
          "for one point), Rectangle / Cuboid corners ordered, degenerate shapes reduced (zero-radius Disk / "
          "Ball / Circle / Sphere to Point, zero-width boxes to Polygon / Line / Point, repeated Line vertices "
          "removed, collinear triangles and flat simplices to Line / Polygon / Point, Annulus with zero inner "
          "radius to Disk, equal radii to Circle); polygons go through CanonicalizePolygon (W19); other regions "
          "(ImplicitRegion, mesh regions, ...) are returned unchanged. Check: exact; tests verify RegionEqual "
          "of input and output.")
def _canonicalize_region(reg: Expr) -> Expr:
    f = "CanonicalizeRegion"
    if not _looks_region(reg):
        if isinstance(reg, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: {K_input(reg)} is not a geometric region")
    h = reg.head
    if h in ("Rectangle", "Cuboid"):
        a, _ = split_trailing_options(list(reg.args))
        if len(a) == 2:
            p, q = pt(a[0], f), pt(a[1], f, dim=len(pt(a[0], f)))
            concrete(p, q)
            lo = [smin([u, v]) for u, v in zip(p, q)]
            hi = [smax([u, v]) for u, v in zip(p, q)]
            return canonical(Expr(h, out(lo, reg), out(hi, reg)), f)
        return canonical(reg, f)
    d = _canon_degenerate(reg, f) if h in ("Disk", "Ball", "Circle", "Sphere", "Line", "Triangle", "Simplex",
                                            "Annulus", "SphericalShell") else None
    if d is not None:
        return d
    if h == "Polygon":
        from .core import dispatch
        try:
            return dispatch("CanonicalizePolygon", [reg])
        except Unevaluated:
            return reg
    if h == "Point":
        return canonical(reg, f)
    try:
        region_of(reg, f)
    except NotImplementedError:
        pass
    return reg


FORMAL = [Symbol("FormalX"), Symbol("FormalY"), Symbol("FormalZ")]


def _formal(n):
    if n <= 3:
        return FORMAL[:n]
    return [Symbol(f"FormalX{i}") for i in range(1, n + 1)]


def implicit_condition(r, Xs, f):
    """SymPy condition describing r in the variables Xs, or None."""
    if isinstance(r, OrientFreeR):
        return None
    if isinstance(r, BoxR):
        return sp.And(*[sp.And(sp.Le(a, x), sp.Le(x, b)) if not eqz(b - a) else sp.Eq(x, a)
                        for a, x, b in zip(r.lo, Xs, r.hi)])
    H = hrep(r)
    if H is not None:
        return sp.And(*[sp.Le(sp.expand(dot(a, Xs)), b) for a, b in H])
    if isinstance(r, EllipsoidR):
        d = sp.Matrix(vsub(Xs, r.c))
        return sp.Le(sp.expand((d.T * r.Q * d)[0]), 1)
    if isinstance(r, SphereR):
        return sp.Eq(sp.expand(n2(vsub(Xs, r.c))), r.r ** 2)
    if isinstance(r, ShellR):
        q = sp.expand(n2(vsub(Xs, r.c)))
        return sp.And(sp.Le(r.r1 ** 2, q), sp.Le(q, r.r2 ** 2))
    if isinstance(r, PointsR):
        return sp.Or(*[sp.And(*[sp.Eq(x, c) for x, c in zip(Xs, p)]) for p in r.P])
    if isinstance(r, FlatR) and not r.halfspace:
        cols = r.V + ([r.w] if r.w is not None else [])
        Nl = sp.Matrix(cols).nullspace() if cols else [sp.Matrix([int(i == j) for j in range(r.edim)])
                                                       for i in range(r.edim)]
        conds = [sp.Eq(sp.expand(dot(list(v), vsub(Xs, r.p))), 0) for v in Nl]
        if r.w is not None:
            conds.append(sp.Ge(sp.expand(dot(r.w, vsub(Xs, r.p))), 0))
        return sp.And(*conds) if conds else sp.true
    if isinstance(r, BoxR):
        return sp.And(*[sp.And(sp.Le(a, x), sp.Le(x, b)) for a, x, b in zip(r.lo, Xs, r.hi)])
    if isinstance(r, RevolutionR) and r.kind in ("CapsuleShape", "StadiumShape"):
        return None
    if isinstance(r, ImplicitR):
        return r.full_cond().xreplace(dict(zip(r.syms, Xs)))
    return None


@fn("RegionConvert", area=AREA, status="added",
    notes="RegionConvert[reg, \"Implicit\"] gives an equivalent ImplicitRegion in the formal variables "
          "\\[FormalX], \\[FormalY], \\[FormalZ] (half-space systems for convex polytopes and boxes, quadratic "
          "inequalities for disks, balls and ellipsoids, equations for spheres, flats and point sets); "
          "\"Mesh\" / \"BoundaryMesh\" delegate to DiscretizeRegion / BoundaryDiscretizeRegion. Other forms "
          "(\"Parametric\", \"CSG\") and unsupported regions raise NotImplementedError; unknown forms "
          "ValueError. Check: exact; tests verify RegionEqual / RegionMember of the converted region.")
def _region_convert(reg: Expr, form: String) -> Expr:
    f = "RegionConvert"
    if not isinstance(form, str):
        raise TypeError(f"{f}: the form must be a string such as \"Implicit\"")
    forms = ("Implicit", "Parametric", "Mesh", "BoundaryMesh", "CSG")
    if form not in forms:
        raise ValueError(f"{f}: unknown form {form!r}; use one of {', '.join(forms)}")
    r = region_of(reg, f)
    if form == "Implicit":
        names = _formal(r.edim)
        Xs = [K.to_sp(s) for s in names]
        c = implicit_condition(r, Xs, f)
        if c is None:
            raise NotImplementedError(f"{f}: converting {r.kind} regions to implicit form is not supported")
        return Expr("ImplicitRegion", R(c), L(*names))
    if form in ("Mesh", "BoundaryMesh"):
        from .core import dispatch
        return dispatch("DiscretizeRegion" if form == "Mesh" else "BoundaryDiscretizeRegion", [reg])
    raise NotImplementedError(f"{f}: the form {form!r} is not supported")


@fn("ShellRegion", area=AREA, status="added",
    notes="ShellRegion[reg, t] gives the solid shell of thickness t inside the boundary of a full-dimensional "
          "region: Disk / Ball give Annulus / SphericalShell {r - t, r} (the region itself when t >= r), "
          "Annulus / SphericalShell thicken inward, Rectangle / Cuboid give the RegionDifference with the box "
          "eroded by t. The default thickness ShellRegion[reg] and other regions raise NotImplementedError; "
          "t <= 0 raises ValueError. Check: exact; tests compare the shell measure with the difference of "
          "measures.")
def _shell_region(reg: Expr, *t: Real) -> Expr:
    f = "ShellRegion"
    r = region_of(reg, f)
    if not t:
        raise NotImplementedError(f"{f}: the default shell thickness is not supported; give ShellRegion[reg, t]")
    if len(t) > 1:
        raise ValueError(f"{f}: expected ShellRegion[reg, t]")
    tv = sx(t[0], f, "thickness")
    concrete(tv)
    if sgn(tv) <= 0:
        raise ValueError(f"{f}: the thickness must be positive")
    ins = (reg, t[0])
    b = _ball_of(r)
    if isinstance(r, EllipsoidR) and b is not None:
        if le(b[1], tv):
            return reg
        return Expr("Annulus" if r.edim == 2 else "SphericalShell", out(b[0], *ins), out([b[1] - tv, b[1]], *ins))
    if isinstance(r, ShellR):
        a2 = r.r2 - tv
        if le(a2, r.r1):
            return reg
        return Expr("Annulus" if r.edim == 2 else "SphericalShell", out(r.c, *ins), out([a2, r.r2], *ins))
    if isinstance(r, BoxR) and r.dim == r.edim:
        inner = _region_erosion(reg, t[0])
        if isinstance(inner, Expr) and inner.head == "EmptyRegion":
            return reg
        return Expr("RegionDifference", reg, inner)
    raise NotImplementedError(f"{f}: shells of {r.kind} regions are not supported")


# ───────────────────────────── curvatures ─────────────────────────────

def _principal(r, p, f):
    """(k_min, k_max) principal curvatures of the boundary surface of a 3-D region (or of a surface) at p,
    with the outward normal (spheres have positive curvature)."""
    if r.edim != 3:
        raise NotImplementedError(f"{f}: curvatures are supported for surfaces in 3 dimensions")
    on = None
    if isinstance(r, (SphereR,)) or (isinstance(r, EllipsoidR) and r.r is not None):
        c, rad = (r.c, r.r)
        if not eqz(n2(vsub(p, c)) - rad ** 2):
            raise ValueError(f"{f}: the point is not on the surface")
        k = simp(1 / rad)
        return k, k
    if isinstance(r, ShellR):
        d2 = n2(vsub(p, r.c))
        if eqz(d2 - r.r2 ** 2):
            k = simp(1 / r.r2)
            return k, k
        if eqz(d2 - r.r1 ** 2) and not eqz(r.r1):
            k = simp(-1 / r.r1)
            return k, k
        raise ValueError(f"{f}: the point is not on the boundary")
    if isinstance(r, RevolutionR) and r.kind == "Cylinder":
        s, r2 = r._sr(p)
        side = eqz(r2 - r.r ** 2) and lt(0, s) and lt(s, r.Lax)
        cap = (eqz(s) or eqz(s - r.Lax)) and lt(r2, r.r ** 2)
        if side:
            return sp.Integer(0), simp(1 / r.r)
        if cap:
            return sp.Integer(0), sp.Integer(0)
        raise NotImplementedError(f"{f}: curvature on the rim or off the surface is not supported")
    if isinstance(r, (PolyhedronR, BoxR, Polygon3R)) or (isinstance(r, SimplexR) and r.edim == 3):
        # flat faces: zero curvature at points interior to a face
        tris = _face_tris(r)
        on_faces = [T for T in tris if _on_tri(p, T)]
        if not on_faces:
            raise ValueError(f"{f}: the point is not on the boundary")
        nrms = []
        for T in on_faces:
            nv = _cross3(vsub(T[1], T[0]), vsub(T[2], T[0]))
            nn = norm(nv)
            nv = [simp(c / nn) for c in nv]
            if not any(is_zero_vec(vsub(nv, m)) or is_zero_vec(vadd(nv, m)) for m in nrms):
                nrms.append(nv)
        if len(nrms) == 1:
            return sp.Integer(0), sp.Integer(0)
        raise NotImplementedError(f"{f}: curvature at edges and corners of polyhedra is not supported")
    if isinstance(r, ImplicitR):
        return _implicit_curv(r, p, f)
    raise NotImplementedError(f"{f}: curvature of {r.kind} regions is not supported")


def _face_tris(r):
    if isinstance(r, PolyhedronR):
        return list(r.tris)
    if isinstance(r, BoxR):
        V, F = G.polyhedron_data(Expr("Cuboid", out(r.lo), out(r.hi)), "Region")
        return [T for fc in F for T in face_triangles(V, fc, "Region")]
    if isinstance(r, Polygon3R):
        tris = []
        for o, hs in r.proj.C:
            for t in G._triangulate(o, r.f):
                tris.append([r.lift(q) for q in t])
        return tris
    return [s for s in r.S if len(s) == 3]


def _on_tri(p, T):
    w, q = proj_affine(p, T)
    return is_zero_vec(vsub(p, q)) and all(sgn(c) >= 0 for c in w)


def _implicit_curv(r, p, f):
    c = r.cond
    if isinstance(c, sp.Rel) and c.rel_op in ("<=", "<", "==", ">=", ">"):
        F = c.lhs - c.rhs if c.rel_op in ("<=", "<", "==") else c.rhs - c.lhs
    else:
        raise NotImplementedError(f"{f}: curvature needs a single equation or inequality")
    Xs = r.syms
    sub = dict(zip(Xs, [S(v) for v in p]))
    if not eqz(F.xreplace(sub)):
        raise ValueError(f"{f}: the point is not on the boundary surface")
    g = sp.Matrix([sp.diff(F, v) for v in Xs]).xreplace(sub)
    Hm = sp.hessian(F, Xs).xreplace(sub)
    g2 = simp((g.T * g)[0])
    if eqz(g2):
        raise ValueError(f"{f}: the surface is singular at the point")
    adj = Hm.adjugate()
    Kg = simp((g.T * adj * g)[0] / g2 ** 2)
    Hmean = simp((g2 * Hm.trace() - (g.T * Hm * g)[0]) / (2 * g2 * sp.sqrt(g2)))
    disc = simp(Hmean ** 2 - Kg)
    if sgn(disc) < 0:
        disc = sp.Integer(0)
    sq = simp(sp.sqrt(disc))
    return simp(Hmean - sq), simp(Hmean + sq)


def _curv(name, reg, p, pick):
    r = region_of(reg, name)
    P, many = query_points(p, r, name)
    vals = []
    for q in P:
        k1, k2 = _principal(r, q, name)
        vals.append(simp(pick(k1, k2)))
    return out(vals if many else vals[0], reg, p)


_CURV_COMMON = ("at a point p of a surface in R^3 (or of the boundary surface of a solid), with the outward "
                "normal so that a sphere of radius r has principal curvatures 1/r: exact for Sphere / Ball, "
                "spherical shells (the inner sphere counts negative), the Cylinder side (0 and 1/r) and caps, "
                "face interiors of polyhedra, cuboids and planar polygons (0), and ImplicitRegion given by one "
                "equation or inequality f <= 0 (gradient / Hessian formulas). A point off the surface raises "
                "ValueError; edges, corners, mesh vertex indices and other regions raise NotImplementedError. ")


@fn("RegionGaussianCurvature", area=AREA, status="added",
    notes="RegionGaussianCurvature[reg, p] gives the Gaussian curvature k_min k_max " + _CURV_COMMON +
          "Check: exact; tests compare 1/r^2 for spheres, 0 for cylinders and planes, and the implicit sphere "
          "x^2 + y^2 + z^2 <= 4 (1/4).")
def _region_gaussian_curvature(reg: Expr, p: ListOf[Real] | MatrixOf[Real]) -> Real | ListOf[Real]:
    return _curv("RegionGaussianCurvature", reg, p, lambda a, b: a * b)


@fn("RegionMeanCurvature", area=AREA, status="added",
    notes="RegionMeanCurvature[reg, p] gives the mean curvature (k_min + k_max)/2 " + _CURV_COMMON +
          "Check: exact; tests compare RegionMeanCurvature[Sphere[], {1, 0, 0}] = 1 (Wolfram documentation "
          "example), 1/(2 r) on a cylinder and the implicit-surface formula against the sphere.")
def _region_mean_curvature(reg: Expr, p: ListOf[Real] | MatrixOf[Real]) -> Real | ListOf[Real]:
    return _curv("RegionMeanCurvature", reg, p, lambda a, b: (a + b) / 2)


@fn("RegionMaxCurvature", area=AREA, status="added",
    notes="RegionMaxCurvature[reg, p] gives the larger principal curvature " + _CURV_COMMON +
          "Check: exact; tests compare 1/r on spheres and cylinders and the implicit ellipsoid at an axis "
          "point.")
def _region_max_curvature(reg: Expr, p: ListOf[Real] | MatrixOf[Real]) -> Real | ListOf[Real]:
    return _curv("RegionMaxCurvature", reg, p, lambda a, b: b)


@fn("RegionMinCurvature", area=AREA, status="added",
    notes="RegionMinCurvature[reg, p] gives the smaller principal curvature " + _CURV_COMMON +
          "Check: exact; tests compare 1/r on spheres, 0 on cylinders and planes.")
def _region_min_curvature(reg: Expr, p: ListOf[Real] | MatrixOf[Real]) -> Real | ListOf[Real]:
    return _curv("RegionMinCurvature", reg, p, lambda a, b: a)


# ───────────────────────────── inert option symbols ─────────────────────────────

def _inert(name, notes):
    @fn(name, area=AREA, status="added", notes=notes)
    def _opt(*args: Expr) -> Expr:
        raise Unevaluated()
    return _opt


_inert("SphericalRegion", "Graphics3D option: whether the final image should be scaled so that a sphere drawn "
                          "around the three-dimensional bounding box would fit in the display area (True / False "
                          "or a Sphere). Inert as an expression. Check: reference (Wolfram documentation) in "
                          "tests.")
_inert("RegionSize", "Printout3D option giving the size of the printed region (a length, a list of lengths or "
                     "Automatic). Inert as an expression. Check: reference (Wolfram documentation) in tests.")


# ───────────────────────────── function-object hooks ─────────────────────────────

def _install():
    from .. import kernel_lists as KL

    def mem(reg, p):
        return _region_member(reg, p)

    def dist(reg, p):
        return _region_distance(reg, p)

    def near(reg, p):
        return _region_nearest(reg, p)
    KL.COMPOUND_HEAD_HOOKS["RegionMemberFunction"] = _fn_head("RegionMemberFunction", mem)
    KL.COMPOUND_HEAD_HOOKS["RegionDistanceFunction"] = _fn_head("RegionDistanceFunction", dist)
    KL.COMPOUND_HEAD_HOOKS["RegionNearestFunction"] = _fn_head("RegionNearestFunction", near)


_install()
