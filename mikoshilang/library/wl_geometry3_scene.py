"""Wolfram batch X06: Geometry III -- numerical geometric scenes (RandomInstance, GeometricSolveValues,
FindGeometricConjectures) for plane GeometricScene objects.  See wl_geometry3 for the conventions used.

A scene's hypotheses are turned into polynomial-style residual equations in the point coordinates and scene
quantities, plus inequalities (nondegenerate polygons, point order on lines); instances are found by
Levenberg-Marquardt from seeded random starts and accepted only when every residual is below 1e-9 (relative to
the scene size) and every inequality holds.
"""
from __future__ import annotations

import itertools
import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, REGISTRY, _arity
from .wl_graphics_color import K_input
from .wl_graphics_core import split_trailing_options, opt_name
from . import wl_geomprims as G
from .wl_geomprims import sx, num
from .wl_geometry3 import AREA
from .wl_geometry3_util import int_arg, is_auto, is_sym, fnum


def _np():
    import numpy as np
    return np


# ───────────────────────────── Midpoint[{p, q}] (Wolfram's list form) ─────────────────────────────

def _midpoint_list(pq: ListOf[ListOf[Real]]) -> ListOf[Real]:
    """Midpoint[{p, q}] = (p + q) / 2 (Wolfram's list form; symbolic points stay unevaluated)."""
    if not (K.is_list(pq) and len(pq.args) == 2 and all(K.is_list(p) for p in pq.args)):
        if K.is_list(pq) and len(pq.args) == 2:
            raise Unevaluated()
        if isinstance(pq, Expr) and pq.head == "Line" and pq.args and K.is_list(pq.args[0]) and \
                len(pq.args[0].args) == 2:
            return _midpoint_list(pq.args[0])
        raise Unevaluated()
    p, q = ([sx(c, "Midpoint") for c in v.args] for v in pq.args)
    if len(p) != len(q):
        raise ValueError("Midpoint: the points must have the same dimension")
    return G.out([sp.simplify((a + b) / 2) for a, b in zip(p, q)], pq)


_e = REGISTRY.get("Midpoint")
if _e is not None and not any(i[0] is _midpoint_list for i in _e.impls):
    _e.impls.append((_midpoint_list, *_arity(_midpoint_list)))


# ───────────────────────────── scene parsing ─────────────────────────────

class Scene:
    def __init__(self, x, f):
        if not (isinstance(x, Expr) and x.head == "GeometricScene"):
            if isinstance(x, Symbol):
                raise Unevaluated()
            raise TypeError(f"{f}: {K_input(x)} is not a GeometricScene")
        a, self.opts = split_trailing_options(list(x.args))
        if len(a) < 2:
            raise ValueError(f"{f}: the scene needs points and hypotheses")
        self.expr = x
        spec = a[0]
        if spec.args and K.is_list(spec.args[0]):
            pts_, qs = spec.args[0].args, (spec.args[1].args if len(spec.args) > 1 else ())
        else:
            pts_, qs = spec.args, ()
        self.points = [self._name(p, f) for p in pts_]
        self.quants = [self._name(q, f) for q in qs]
        self.hyps = list(a[1].args)
        self.concl = list(a[2].args) if len(a) > 2 else []
        self.rest = a[3:]
        self.f = f
        self.var = {}
        for i, p in enumerate(self.points):
            self.var[p] = ("pt", 2 * i)
        base = 2 * len(self.points)
        for j, q in enumerate(self.quants):
            self.var[q] = ("q", base + j)
        self.n = base + len(self.quants)
        self.eqs, self.ineqs, self.lines, self.lens = [], [], [], []
        for h in self.hyps:
            self._hyp(h)

    @staticmethod
    def _name(p, f):
        if isinstance(p, Expr) and p.head == "Rule":
            p = p.args[0]
        if not isinstance(p, Symbol):
            raise TypeError(f"{f}: scene points and quantities must be symbols, not {K_input(p)}")
        return p.name

    # ---- numeric evaluation of geometric terms
    def pt(self, X, e):
        np = _np()
        if isinstance(e, Symbol) and e.name in self.var and self.var[e.name][0] == "pt":
            i = self.var[e.name][1]
            return X[i:i + 2]
        if K.is_list(e) and len(e.args) == 2:
            return np.array([self.val(X, c) for c in e.args])
        if isinstance(e, Expr) and e.head == "Midpoint":
            ps = e.args[0].args if len(e.args) == 1 else e.args
            return sum(self.pt(X, p) for p in ps) / len(ps)
        raise NotImplementedError(f"{self.f}: unsupported point expression {K_input(e)}")

    def val(self, X, e):
        np = _np()
        if isinstance(e, bool):
            raise TypeError(f"{self.f}: unexpected truth value")
        if isinstance(e, Symbol):
            if e.name in self.var and self.var[e.name][0] == "q":
                return X[self.var[e.name][1]]
            if e.name == "Pi":
                return math.pi
            if e.name == "Degree":
                return math.pi / 180
            raise NotImplementedError(f"{self.f}: unknown symbol {e.name} in a scene expression")
        if not isinstance(e, Expr):
            try:
                return float(num(sx(e, self.f)))
            except Exception:
                raise NotImplementedError(f"{self.f}: unsupported value {K_input(e)}")
        h = e.head
        if h == "Plus":
            return sum(self.val(X, a) for a in e.args)
        if h == "Times":
            r = 1.0
            for a in e.args:
                r = r * self.val(X, a)
            return r
        if h == "Power":
            return self.val(X, e.args[0]) ** self.val(X, e.args[1])
        if h == "Rational":
            return float(num(sx(e, self.f)))
        if h == "Sqrt":
            return math.sqrt(self.val(X, e.args[0]))
        if h in ("Sin", "Cos", "Tan"):
            return getattr(math, h.lower())(self.val(X, e.args[0]))
        dp = _dist_pair(self, e)
        if dp is not None:
            return float(np.linalg.norm(self.pt(X, dp[0]) - self.pt(X, dp[1])))
        if h == "Abs":
            return abs(self.val(X, e.args[0]))
        if h == "PlanarAngle":
            return self.angle(X, e)
        if h in ("Area", "RegionMeasure"):
            g = _unstyle(e.args[0])
            if isinstance(g, Expr) and g.head in ("Polygon", "Triangle"):
                P = [self.pt(X, p) for p in g.args[0].args]
                return abs(sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1]
                               for i in range(len(P)))) / 2
            if isinstance(g, Expr) and g.head == "Disk":
                return math.pi * self.val(X, g.args[1]) ** 2
        if h in ("Perimeter", "ArcLength") and isinstance(_unstyle(e.args[0]), Expr) and \
                _unstyle(e.args[0]).head in ("Polygon", "Triangle", "Line"):
            g = _unstyle(e.args[0])
            P = [self.pt(X, p) for p in g.args[0].args]
            closed = g.head != "Line"
            idx = range(len(P)) if closed else range(len(P) - 1)
            return sum(float(np.linalg.norm(P[(i + 1) % len(P)] - P[i])) for i in idx)
        raise NotImplementedError(f"{self.f}: the scene expression {K_input(e)} is not supported")

    def angle(self, X, e):
        np = _np()
        a, b, c = (self.pt(X, p) for p in e.args[0].args)
        u, v = a - b, c - b
        if len(e.args) > 1 and e.args[1] in ("Counterclockwise", "Clockwise"):
            ang = math.atan2(u[0] * v[1] - u[1] * v[0], u @ v) % (2 * math.pi)
            return ang if e.args[1] == "Counterclockwise" else (2 * math.pi - ang) % (2 * math.pi)
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return 0.0
        return math.acos(max(-1.0, min(1.0, float(u @ v / (nu * nv)))))

    # ---- hypotheses -> residuals (functions of X) and inequalities (> 0)
    def _hyp(self, h):
        np = _np()
        h = _unstyle(h)
        f = self.f
        if isinstance(h, Expr) and h.head == "Equal" and len(h.args) == 2:
            l, r = h.args
            if isinstance(l, Symbol) and l.name in self.var and self.var[l.name][0] == "pt":
                if isinstance(r, Expr) and r.head == "TriangleCenter":
                    self._center(l, r)
                    return
                if isinstance(r, Expr) and r.head in ("RegularPolygon", "Polygon", "Triangle", "Circle", "Disk"):
                    raise NotImplementedError(f"{f}: naming regions (p == {r.head}[...]) is not supported")
                self.eqs.append(lambda X, l=l, r=r: self.pt(X, l) - self.pt(X, r))
                return
            if isinstance(l, Symbol) and l.name not in self.var and isinstance(r, Expr):
                raise NotImplementedError(f"{f}: naming regions ({l.name} == {r.head}[...]) is not supported")
            if isinstance(l, Expr) and l.head == "PlanarAngle":
                if len(l.args) > 1:
                    self.eqs.append(lambda X, l=l, r=r: np.array([math.sin((self.angle(X, l) - self.val(X, r))
                                                                           / 2)]))
                    return
                self.eqs.append(lambda X, l=l, r=r: np.array([self._cosres(X, l, r)]))
                return
            dl, dr = _dist_pair(self, l), _dist_pair(self, r)
            if dl is not None and dr is not None:
                self.eqs.append(lambda X, dl=dl, dr=dr: np.array([self._d2p(X, *dl) - self._d2p(X, *dr)]))
                return
            if dl is not None:
                try:
                    self.lens.append(abs(float(self.val(np.zeros(self.n), r))))
                except Exception:
                    pass
                self.eqs.append(lambda X, dl=dl, r=r: np.array([self._d2p(X, *dl) - self.val(X, r) ** 2]))
                self.ineqs.append(lambda X, r=r: self.val(X, r) + 1e-12)
                return
            self.eqs.append(lambda X, l=l, r=r: np.array([self.val(X, l) - self.val(X, r)]))
            return
        if isinstance(h, Expr) and h.head in ("Triangle", "Polygon") and h.args and K.is_list(h.args[0]):
            ps = list(h.args[0].args)
            self.ineqs.append(lambda X, ps=ps: self._nondeg(X, ps))
            return
        if isinstance(h, Expr) and h.head == "Line" and h.args and K.is_list(h.args[0]):
            groups = h.args[0].args if h.args[0].args and K.is_list(h.args[0].args[0]) else [h.args[0]]
            for g in groups:
                ps = list(g.args)
                if len(ps) >= 3:
                    self.eqs.append(lambda X, ps=ps: np.array(
                        [self._cross(X, ps[0], ps[-1], p) for p in ps[1:-1]]))
                    self.ineqs.append(lambda X, ps=ps: self._order(X, ps))
                    self.lines.append([self.var[p.name][1] for p in ps if isinstance(p, Symbol) and p.name in self.var])
                elif len(ps) == 2:
                    self.ineqs.append(lambda X, ps=ps: self._sep(X, ps[0], ps[1]))
            return
        if isinstance(h, Expr) and h.head == "InfiniteLine":
            ps = list(h.args[0].args) if h.args and K.is_list(h.args[0]) else []
            if len(ps) >= 3:
                self.eqs.append(lambda X, ps=ps: np.array([self._cross(X, ps[0], ps[1], p) for p in ps[2:]]))
            return
        if isinstance(h, Expr) and h.head == "CircleThrough" and len(h.args) == 2:
            ps, o = list(h.args[0].args), h.args[1]
            self.eqs.append(lambda X, ps=ps, o=o: np.array(
                [self._d2p(X, o, ps[0]) - self._d2p(X, o, p) for p in ps[1:]]))
            return
        if isinstance(h, Expr) and h.head == "GeometricAssertion" and len(h.args) >= 2:
            obj, kind = h.args[0], h.args[1]
            if kind in ("Perpendicular", "Parallel") and K.is_list(obj) and len(obj.args) == 2:
                l1, l2 = (self._line_pts(x) for x in obj.args)
                if kind == "Perpendicular":
                    self.eqs.append(lambda X, l1=l1, l2=l2: np.array([self._dotl(X, l1, l2)]))
                else:
                    self.eqs.append(lambda X, l1=l1, l2=l2: np.array([self._crossl(X, l1, l2)]))
                return
            if kind == "Collinear" and K.is_list(obj):
                ps = list(obj.args)
                self.eqs.append(lambda X, ps=ps: np.array([self._cross(X, ps[0], ps[1], p) for p in ps[2:]]))
                return
            raise NotImplementedError(f"{f}: GeometricAssertion \"{kind}\" is not supported in scenes")
        if isinstance(h, Expr) and h.head in ("PlanarAngle", "EuclideanDistance", "Abs", "Area", "Circle", "Disk",
                                             "Point"):
            return  # display-only objects
        raise NotImplementedError(f"{f}: the hypothesis {K_input(h)} is not supported")

    def _line_pts(self, x):
        x = _unstyle(x)
        if isinstance(x, Expr) and x.head in ("Line", "InfiniteLine") and x.args and K.is_list(x.args[0]) and \
                len(x.args[0].args) >= 2:
            return (x.args[0].args[0], x.args[0].args[-1])
        raise NotImplementedError(f"{self.f}: expected Line[{{p, q}}] in a GeometricAssertion")

    def _center(self, o, r):
        np = _np()
        ps = list(r.args[0].args)
        kind = r.args[1] if len(r.args) > 1 else "Centroid"
        if kind == "Circumcenter":
            self.eqs.append(lambda X: np.array([self._d2p(X, o, ps[0]) - self._d2p(X, o, p) for p in ps[1:]]))
        elif kind == "Centroid":
            self.eqs.append(lambda X: self.pt(X, o) - sum(self.pt(X, p) for p in ps) / 3)
        elif kind == "Orthocenter":
            def res(X):
                A, B, C, O = (self.pt(X, p) for p in ps + [o])
                return np.array([(O - A) @ (C - B), (O - B) @ (C - A)])
            self.eqs.append(res)
        elif kind == "Incenter":
            def res(X):
                A, B, C, O = (self.pt(X, p) for p in ps + [o])
                a, b, c = np.linalg.norm(B - C), np.linalg.norm(C - A), np.linalg.norm(A - B)
                return O * (a + b + c) - (a * A + b * B + c * C)
            self.eqs.append(res)
        else:
            raise NotImplementedError(f"{self.f}: TriangleCenter \"{kind}\" is not supported in scenes")

    def _cosres(self, X, l, r):
        np = _np()
        a, b, c = (self.pt(X, p) for p in l.args[0].args)
        u, v = a - b, c - b
        th = self.val(X, r)
        return float(np.linalg.norm(u) * np.linalg.norm(v) * math.cos(th) - u @ v)

    def _d2p(self, X, p, q):
        d = self.pt(X, p) - self.pt(X, q)
        return float(d @ d)

    def _cross(self, X, p0, p1, q):
        a, b, c = self.pt(X, p0), self.pt(X, p1), self.pt(X, q)
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def _dotl(self, X, l1, l2):
        u = self.pt(X, l1[1]) - self.pt(X, l1[0])
        v = self.pt(X, l2[1]) - self.pt(X, l2[0])
        return float(u @ v)

    def _crossl(self, X, l1, l2):
        u = self.pt(X, l1[1]) - self.pt(X, l1[0])
        v = self.pt(X, l2[1]) - self.pt(X, l2[0])
        return float(u[0] * v[1] - u[1] * v[0])

    def _sep(self, X, p, q):
        np = _np()
        return float(np.linalg.norm(self.pt(X, p) - self.pt(X, q))) - 1e-6 * self.scale(X)

    def _order(self, X, ps):
        P = [self.pt(X, p) for p in ps]
        d = P[-1] - P[0]
        return min(float((P[i + 1] - P[i]) @ d) for i in range(len(P) - 1)) - 1e-9 * float(d @ d)

    def _nondeg(self, X, ps):
        P = [self.pt(X, p) for p in ps]
        n = len(P)
        m = min(abs((P[(i + 1) % n] - P[i])[0] * (P[(i + 2) % n] - P[i])[1] -
                    (P[(i + 1) % n] - P[i])[1] * (P[(i + 2) % n] - P[i])[0]) for i in range(n))
        return m - 1e-6 * self.scale(X) ** 2

    def scale(self, X):
        np = _np()
        P = X[:2 * len(self.points)].reshape(-1, 2)
        return float(np.linalg.norm(P.max(0) - P.min(0))) or 1.0

    def residual(self, X):
        np = _np()
        if not self.eqs:
            return np.zeros(0)
        return np.concatenate([np.atleast_1d(np.asarray(e(X), dtype=float)) for e in self.eqs])

    def start(self, rng):
        """Random start with the points of each Line[{p1, ..., pk}] hypothesis placed in order on the segment
        p1 pk, so the solver starts in the basin of the intended configuration."""
        np = _np()
        L = max(self.lens) if self.lens else 1.0
        x0 = np.concatenate([rng.uniform(-L, L, 2 * len(self.points)), rng.uniform(0.5, 2, len(self.quants))])
        for idx in self.lines:
            if len(idx) < 3:
                continue
            a, b = x0[idx[0]:idx[0] + 2].copy(), x0[idx[-1]:idx[-1] + 2].copy()
            ts = np.sort(rng.uniform(0.1, 0.9, len(idx) - 2))
            for i, t in zip(idx[1:-1], ts):
                x0[i:i + 2] = a + t * (b - a)
        return x0

    def residual_aug(self, X):
        """Equations plus hinge penalties pushing the order / nondegeneracy inequalities to a small positive
        margin (used when the plain solve lands on a solution with the points in the wrong order)."""
        np = _np()
        m = 1e-3 * self.scale(X) ** 2
        return np.concatenate([self.residual(X), np.array([max(0.0, m - g(X)) for g in self.ineqs])])

    def general_position(self, X):
        """Distinct points (Wolfram: points are assumed to be in general position)."""
        np = _np()
        P = X[:2 * len(self.points)].reshape(-1, 2)
        s = self.scale(X)
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                if np.linalg.norm(P[i] - P[j]) < 1e-4 * s:
                    return False
        return True


def _dist_pair(S, e):
    """(p, q) when e is EuclideanDistance[p, q] or its evaluated form Abs[p - q] for scene points p, q."""
    if not isinstance(e, Expr):
        return None
    if e.head == "EuclideanDistance" and len(e.args) == 2:
        return e.args[0], e.args[1]
    if e.head == "Abs" and len(e.args) == 1 and isinstance(e.args[0], Expr) and e.args[0].head == "Plus" and \
            len(e.args[0].args) == 2:
        a, b = e.args[0].args
        names = [x for x in (a, b) if isinstance(x, Symbol)]
        neg = [x for x in (a, b) if isinstance(x, Expr) and x.head == "Times" and len(x.args) == 2 and
               x.args[0] == -1 and isinstance(x.args[1], Symbol)]
        if len(names) == 1 and len(neg) == 1:
            p, q = names[0], neg[0].args[1]
            if all(v.name in S.var and S.var[v.name][0] == "pt" for v in (p, q)):
                return p, q
    return None


def _unstyle(h):
    while isinstance(h, Expr) and h.head in ("Style", "Labeled", "Tooltip") and h.args:
        h = h.args[0]
    return h


def lm_solve(fun, x0, iters=200):
    """Levenberg-Marquardt with a forward-difference Jacobian (minimum-norm steps for underdetermined
    systems)."""
    np = _np()
    x = x0.copy()
    r = fun(x)
    if r.size == 0:
        return x, 0.0
    lam = 1e-3
    for _ in range(iters):
        nr = float(r @ r)
        if nr < 1e-30:
            break
        J = np.empty((r.size, x.size))
        h = 1e-7 * max(1.0, float(np.abs(x).max()))
        for k in range(x.size):
            xp = x.copy()
            xp[k] += h
            J[:, k] = (fun(xp) - r) / h
        A = J @ J.T + lam * np.eye(r.size)
        try:
            dx = -J.T @ np.linalg.solve(A, r)
        except np.linalg.LinAlgError:
            break
        xn = x + dx
        rn = fun(xn)
        if float(rn @ rn) < nr:
            x, r = xn, rn
            lam = max(lam / 3, 1e-12)
        else:
            lam *= 4
            if lam > 1e8:
                break
    return x, float(np.sqrt(r @ r))


def instances(S, k, f, seed=None, tries=60):
    np = _np()
    rng = G._np_rng() if seed is None else np.random.default_rng(seed)
    out_ = []
    for _ in range(min(tries * k, 30 + 8 * k)):
        if len(out_) >= k:
            break
        x0 = S.start(rng) if _ % 2 == 0 else np.concatenate(
            [rng.uniform(-1, 1, 2 * len(S.points)), rng.uniform(0.5, 2, len(S.quants))])
        x, res = lm_solve(S.residual, x0)
        if S.ineqs and any(g(x) <= 0 for g in S.ineqs):
            x, _ = lm_solve(S.residual_aug, x)
        # polish
        x, res = lm_solve(S.residual, x, 50)
        s = S.scale(x)
        if not np.all(np.isfinite(x)) or res > 1e-9 * max(1.0, s * s):
            continue
        if any(g(x) <= 0 for g in S.ineqs) or not S.general_position(x):
            continue
        out_.append(x)
    if len(out_) < k:
        raise ValueError(f"{f}: no instance satisfying the hypotheses was found (they may be inconsistent or "
                         "degenerate)")
    return out_


def _seed(opts, f):
    for r in opts:
        if opt_name(r) == "RandomSeeding":
            v = r.args[1]
            if is_auto(v) or is_sym(v, "Inherited"):
                return None
            if K.is_int(v):
                return int(v)
            if isinstance(v, str):
                return sum(ord(c) * 131 ** i for i, c in enumerate(v)) % (2 ** 32)
            raise ValueError(f"{f}: RandomSeeding must be Automatic, an integer or a string")
        raise ValueError(f"{f}: {opt_name(r)} is not a known option")
    return None


def instance_expr(S, X):
    X = X.copy()
    X[abs(X) < 1e-12 * S.scale(X)] = 0.0  # round-off below the solver tolerance
    pts_ = [Expr("Rule", Symbol(p), L(fnum(X[2 * i]), fnum(X[2 * i + 1]))) for i, p in enumerate(S.points)]
    qs = [Expr("Rule", Symbol(q), fnum(X[S.var[q][1]])) for q in S.quants]
    return Expr("GeometricScene", L(L(*pts_), L(*qs)), L(*S.hyps), L(*S.concl), *S.opts)


@fn("RandomInstance", area=AREA, status="added",
    notes="RandomInstance[scene] and RandomInstance[scene, n] for plane GeometricScene objects: numerical "
          "instances satisfying the hypotheses, returned as GeometricScene[{{p1 -> {x, y}, ...}, {k1 -> v, "
          "...}}, hyps, concls]. Supported hypotheses: p == {x, y}, p == Midpoint[{q, r}], p == "
          "TriangleCenter[{a, b, c}, \"Circumcenter\" | \"Centroid\" | \"Orthocenter\" | \"Incenter\"], "
          "Triangle / Polygon (nondegenerate), Line[{p1, ..., pk}] (collinear, in this order), InfiniteLine, "
          "CircleThrough[{...}, o], PlanarAngle[{a, b, c}] == theta (also \"Counterclockwise\"), "
          "EuclideanDistance equalities, GeometricAssertion[{Line, Line}, \"Perpendicular\" | \"Parallel\"] and "
          "GeometricAssertion[pts, \"Collinear\"], Style wrappers; others raise NotImplementedError. Instances: "
          "Levenberg-Marquardt from random starts (seeded by SeedRandom or RandomSeeding), accepted when all "
          "residuals are below 1e-9 relative to the scene size, inequalities hold and the points are distinct "
          "(general position). Inconsistent hypotheses raise ValueError. RandomInstance of other objects "
          "(BioSequence) raises NotImplementedError. Check: self-check of the residuals (built into the "
          "acceptance test); tests verify the hypotheses on the returned coordinates.")
def _random_instance(scene: Expr, *rest: PosInteger | Option) -> Expr:
    f = "RandomInstance"
    pos, opts = split_trailing_options(list(rest))
    if isinstance(scene, Expr) and scene.head != "GeometricScene":
        raise NotImplementedError(f"{f}: only GeometricScene objects are supported")
    S = Scene(scene, f)
    if len(pos) > 1:
        raise ValueError(f"{f}: expected RandomInstance[scene, n]")
    n = int_arg(pos[0], f, "the number of instances", 1, 1000) if pos else None
    seed = _seed(opts, f)
    xs = instances(S, n or 1, f, seed)
    res = [instance_expr(S, x) for x in xs]
    return L(*res) if n is not None else res[0]


@fn("GeometricSolveValues", area=AREA, status="added",
    notes="GeometricSolveValues[scene, expr] and GeometricSolveValues[scene, {expr1, ...}] the value of "
          "expressions (EuclideanDistance, PlanarAngle, Area / Perimeter of polygons, scene quantities and "
          "arithmetic of these) determined by the hypotheses of a plane GeometricScene: the expression is "
          "evaluated on 6 independent random instances (as in RandomInstance) and returned as a machine real "
          "when all agree to 1e-7 relative; otherwise the hypotheses do not determine it and ValueError is "
          "raised. Values are numerical (Wolfram returns exact values when it can solve symbolically). "
          "Unsupported hypotheses or expressions raise NotImplementedError. Check: self-check (agreement "
          "across instances is the test); tests use scenes with known answers (a right isosceles triangle, a "
          "3-4-5 triangle, Thales' circle).")
def _geometric_solve_values(scene: Expr, expr: Expr, *opts: Option) -> Real | ListOf[Real]:
    f = "GeometricSolveValues"
    np = _np()
    S = Scene(scene, f)
    seed = _seed(opts, f)
    xs = instances(S, 6, f, seed)
    exprs = list(expr.args) if K.is_list(expr) else [expr]
    vals = []
    for e in exprs:
        v = [S.val(x, e) for x in xs]
        ref = v[0]
        if any(abs(w - ref) > 1e-7 * max(1.0, abs(ref)) for w in v):
            raise ValueError(f"{f}: {K_input(e)} is not determined by the hypotheses")
        vals.append(fnum(np.mean(v)))
    return L(*vals) if K.is_list(expr) else vals[0]


# ───────────────────────────── FindGeometricConjectures ─────────────────────────────

def _candidates(S):
    """Candidate conclusions (expr, test(X) -> residual, scale power) over the scene points."""
    np = _np()
    P = [Symbol(p) for p in S.points]
    c = []
    for a, b, d in itertools.combinations(P, 3):
        c.append((Expr("GeometricAssertion", L(a, b, d), "Collinear"),
                  lambda X, a=a, b=b, d=d: S._cross(X, a, b, d), 2))
    for (a, b), (d, e) in itertools.combinations(list(itertools.combinations(P, 2)), 2):
        l1, l2 = (a, b), (d, e)
        lines = L(Expr("Line", L(a, b)), Expr("Line", L(d, e)))
        if not {a.name, b.name} & {d.name, e.name}:  # lines through a common point: that is collinearity
            c.append((Expr("GeometricAssertion", lines, "Parallel"),
                      lambda X, l1=l1, l2=l2: S._crossl(X, l1, l2), 2))
        c.append((Expr("GeometricAssertion", lines, "Perpendicular"),
                  lambda X, l1=l1, l2=l2: S._dotl(X, l1, l2), 2))
        c.append((Expr("Equal", Expr("EuclideanDistance", a, b), Expr("EuclideanDistance", d, e)),
                  lambda X, l1=l1, l2=l2: S._d2p(X, *l1) - S._d2p(X, *l2), 2))
    for m in P:
        for a, b in itertools.combinations([p for p in P if p is not m], 2):
            c.append((Expr("Equal", m, Expr("Midpoint", L(a, b))),
                      lambda X, m=m, a=a, b=b: float(np.linalg.norm(S.pt(X, m) - (S.pt(X, a) + S.pt(X, b)) / 2)),
                      1))
    for a, b, d in itertools.permutations(P, 3):
        if a.name < d.name:
            c.append((Expr("Equal", Expr("PlanarAngle", L(a, b, d)), Expr("Times", 90, Symbol("Degree"))),
                      lambda X, a=a, b=b, d=d: float((S.pt(X, a) - S.pt(X, b)) @ (S.pt(X, d) - S.pt(X, b))), 2))
    for q in itertools.combinations(P, 4):
        def cyc(X, q=q):
            M_ = np.array([[x @ x, x[0], x[1], 1.0] for x in (S.pt(X, p) for p in q)])
            return float(np.linalg.det(M_))
        c.append((Expr("GeometricAssertion", L(*q), "Concyclic"), cyc, 4))
    return c


def _norm_expr(e):
    return str(e)


@fn("FindGeometricConjectures", area=AREA, status="added",
    notes="FindGeometricConjectures[scene], [scene, patt] and [scene, patt, n] for plane GeometricScene "
          "objects: the scene with added conclusions that hold on 4 independent random instances (as in "
          "RandomInstance) and are not among the hypotheses. Candidate catalogue: collinear triples, parallel and "
          "perpendicular point-pair lines (parallel only for lines without a common point), equal distances, midpoints, right angles PlanarAngle[{a, b, c}] == "
          "90 Degree and concyclic quadruples (GeometricAssertion[{...}, \"Concyclic\"]); a candidate holds when "
          "its residual is below 1e-7 relative to the scene size on every instance and is not degenerate "
          "(distinct points). Conclusions implied by others are not removed, so the list can be longer than "
          "Wolfram's. patt (default _) filters with MatchQ; n caps the number. Method -> {\"RandomInstance\", "
          "RandomSeeding -> s} seeds the instances. Check: self-check (every conclusion is re-verified on a "
          "fresh instance; ArithmeticError otherwise); tests find Thales' right angle and the midpoint of the "
          "hypotenuse as circumcentre.")
def _find_geometric_conjectures(scene: Expr, *rest: Expr | PosInteger | Option) -> Expr:
    f = "FindGeometricConjectures"
    np = _np()
    pos, opts = split_trailing_options(list(rest))
    seed = None
    for r in opts:
        if opt_name(r) != "Method":
            raise ValueError(f"{f}: {opt_name(r)} is not a known option")
        v = r.args[1]
        if K.is_list(v) and v.args and v.args[0] == "RandomInstance":
            seed = _seed(list(v.args[1:]), f)
        elif not is_auto(v):
            raise NotImplementedError(f"{f}: Method -> {K_input(v)} is not supported")
    if K.is_list(scene):
        if not scene.args:
            raise ValueError(f"{f}: no scenes given")
        scene = scene.args[0]
    S = Scene(scene, f)
    patt = pos[0] if pos else None
    cap = int_arg(pos[1], f, "the number of conjectures", 1) if len(pos) > 1 else None
    if len(pos) > 2:
        raise ValueError(f"{f}: expected FindGeometricConjectures[scene, patt, n]")
    xs = instances(S, 5, f, seed)
    check_x, xs = xs[-1], xs[:-1]
    hyp_keys = {_norm_expr(_unstyle(h)) for h in S.hyps} | {_norm_expr(c) for c in S.concl}
    found = []
    for e, test, pw in _candidates(S):
        if _norm_expr(e) in hyp_keys:
            continue
        ok = all(abs(test(x)) <= 1e-7 * S.scale(x) ** pw for x in xs)
        if not ok:
            continue
        if patt is not None:
            from ..evaluate import evaluate
            if evaluate(Expr("MatchQ", e, patt)) is not True:
                continue
        if abs(test(check_x)) > 1e-6 * S.scale(check_x) ** pw:
            raise ArithmeticError(f"{f}: self-check failed (a conjecture fails on a fresh instance)")
        found.append(e)
        if cap is not None and len(found) >= cap:
            break
    return Expr("GeometricScene", scene.args[0], L(*S.hyps), L(*(S.concl + found)), *S.rest, *S.opts)
