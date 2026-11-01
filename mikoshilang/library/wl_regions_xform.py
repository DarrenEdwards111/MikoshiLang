"""Wolfram batch W22 (part): geometric transformations -- TransformationFunction and its constructors, Euler and
roll-pitch-yaw rotations, GeometricTransformation.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).  Transformation constructors accept symbolic parameters (they build symbolic matrices, as in
Wolfram); ``TransformationFunction[m][v]`` applies the homogeneous matrix ``m`` to a point or a list of points.
"""
from __future__ import annotations

import functools

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, R, has_float
from .wl_graphics_color import K_input
from .wl_graphics_core import split_trailing_options, opt_name
from .wl_geomprims import sx, out, num

AREA = "geometry: transformations (W22)"


# ───────────────────────────── helpers ─────────────────────────────

def _is_sym(x):
    return isinstance(x, Symbol)


def scal(x, f, what="parameter"):
    """A scalar parameter (symbolic allowed) -> SymPy."""
    return sx(x, f, what)


def vec(x, f, what="vector", dim=None):
    """{v1, ..., vn} (symbolic entries allowed) -> list of SymPy values."""
    if not K.is_list(x):
        if _is_sym(x):
            raise Unevaluated()
        raise TypeError(f"{f}: {what} must be a list of numbers, not {K_input(x)}")
    if not x.args:
        raise ValueError(f"{f}: {what} must not be empty")
    if any(K.is_list(c) for c in x.args):
        raise TypeError(f"{f}: {what} must be a flat list of numbers, not {K_input(x)}")
    if dim is not None and len(x.args) not in ((dim,) if isinstance(dim, int) else dim):
        raise ValueError(f"{f}: {what} {K_input(x)} must have {dim} components")
    return [scal(c, f, what + " component") for c in x.args]


def mat(x, f, what="matrix", square=False):
    if not (K.is_list(x) and x.args and all(K.is_list(r) for r in x.args)):
        if _is_sym(x):
            raise Unevaluated()
        raise TypeError(f"{f}: {what} must be a matrix (list of rows), not {K_input(x)}")
    rows = [vec(r, f, what + " row") for r in x.args]
    if len({len(r) for r in rows}) != 1:
        raise ValueError(f"{f}: {what} must be rectangular")
    if square and len(rows) != len(rows[0]):
        raise ValueError(f"{f}: {what} must be square")
    return sp.Matrix(rows)


def _zero(v):
    v = sp.sympify(v)
    if v.free_symbols:
        return False
    if v.has(sp.Float):
        return abs(complex(sp.N(v))) < 1e-12
    return sp.simplify(v) == 0


def _nonzero_vec(v, f, what="vector"):
    if all(_zero(c) for c in v):
        raise ValueError(f"{f}: {what} must be nonzero")


def _simp(e):
    e = sp.sympify(e)
    if e.has(sp.Float):
        return e
    if e.free_symbols:
        return e
    try:
        s = sp.radsimp(sp.simplify(e))
        return s if sp.count_ops(s) <= sp.count_ops(e) else e
    except Exception:
        return e


def homog(A, b=None):
    """(n+1)x(n+1) homogeneous matrix of x -> A.x + b."""
    n = A.shape[0]
    M = sp.zeros(n + 1, n + 1)
    M[:n, :n] = A
    if b is not None:
        for i in range(n):
            M[i, n] = b[i]
    M[n, n] = 1
    return M


def tf_expr(M, *inputs):
    M = M.applyfunc(_simp)
    return Expr("TransformationFunction", out([[M[i, j] for j in range(M.shape[1])] for i in range(M.shape[0])],
                                              *inputs))


def about_point(A, p):
    """Linear map A applied about the point p: x -> p + A.(x - p)."""
    pv = sp.Matrix(p)
    return homog(A, list(pv - A * pv))


def tf_matrix(x, f="TransformationFunction"):
    """Homogeneous matrix of a TransformationFunction expression (SymPy Matrix)."""
    if not (isinstance(x, Expr) and x.head == "TransformationFunction" and len(x.args) == 1):
        raise TypeError(f"{f}: {K_input(x)} is not a TransformationFunction")
    M = mat(x.args[0], f, "transformation matrix", square=True)
    if M.shape[0] < 2:
        raise ValueError(f"{f}: the transformation matrix must be at least 2 x 2")
    return M


def affine_parts(M, f):
    """(A, b) for an affine homogeneous matrix; NotImplementedError for a projective one."""
    n = M.shape[0] - 1
    if not (all(_zero(M[n, j]) for j in range(n)) and _zero(M[n, n] - 1)):
        if all(_zero(M[n, j]) for j in range(n)) and not _zero(M[n, n]) and not sp.sympify(M[n, n]).free_symbols:
            M = M / M[n, n]
        else:
            raise NotImplementedError(f"{f}: projective (linear fractional) transformations are not supported here")
    return M[:n, :n], [M[i, n] for i in range(n)]


def as_transform(t, f, dim=None):
    """TransformationFunction, a square matrix m (linear) or {m, v} (affine) -> homogeneous SymPy Matrix."""
    if isinstance(t, Expr) and t.head == "TransformationFunction":
        M = tf_matrix(t, f)
    elif K.is_list(t) and len(t.args) == 2 and K.is_list(t.args[0]) and t.args[0].args and \
            K.is_list(t.args[0].args[0]) and K.is_list(t.args[1]) and not K.is_list(t.args[1].args[0] if t.args[1].args else 0):
        A = mat(t.args[0], f, square=True)
        b = vec(t.args[1], f, "translation vector", dim=A.shape[0])
        M = homog(A, b)
    elif K.is_list(t) and t.args and all(K.is_list(r) for r in t.args):
        M = homog(mat(t, f, square=True))
    else:
        if _is_sym(t):
            raise Unevaluated()
        raise TypeError(f"{f}: {K_input(t)} is not a transformation (TransformationFunction, matrix or {{m, v}})")
    if dim is not None and M.shape[0] - 1 != dim:
        raise ValueError(f"{f}: the transformation acts on R^{M.shape[0] - 1}, not R^{dim}")
    return M


def apply_h(M, p, f):
    n = M.shape[0] - 1
    w = M * sp.Matrix(list(p) + [1])
    d = sp.sympify(w[n])
    if _zero(d):
        raise ValueError(f"{f}: the point {list(p)} is mapped to infinity")
    if d == 1:
        return [sp.expand(w[i]) if w[i].free_symbols else w[i] for i in range(n)]
    return [sp.cancel(w[i] / d) if (w[i] / d).free_symbols else w[i] / d for i in range(n)]


def _tf_apply(head, args, ev_):
    f = "TransformationFunction"
    if len(args) != 1 or len(head.args) != 1:
        return K.UNHANDLED
    try:
        M = tf_matrix(head, f)
    except Unevaluated:
        return K.UNHANDLED
    n = M.shape[0] - 1
    x = args[0]
    if not K.is_list(x):
        if _is_sym(x) or (isinstance(x, Expr) and not isinstance(x, bool)):
            if isinstance(x, (str, bool)):
                raise TypeError(f"{f}: cannot apply a transformation to {K_input(x)}")
            return K.UNHANDLED
        raise TypeError(f"{f}: cannot apply a transformation to {K_input(x)}")
    if x.args and all(K.is_list(p) for p in x.args):
        res = []
        for p in x.args:
            v = vec(p, f, "point", dim=n)
            res.append(apply_h(M, v, f))
        return out(res, x, head.args[0])
    v = vec(x, f, "point", dim=n)
    return out(apply_h(M, v, f), x, head.args[0])


def _install():
    from .. import kernel_lists as KL
    KL.COMPOUND_HEAD_HOOKS["TransformationFunction"] = _tf_apply


_install()


# ───────────────────────────── TransformationFunction ─────────────────────────────

@fn("TransformationFunction", area=AREA, status="added",
    notes="TransformationFunction[m] represents the transformation x -> (m.{x, 1})[[1 ;; n]] / (m.{x, 1})[[n + 1]] "
          "given by an (n + 1) x (n + 1) homogeneous matrix m (symbolic entries allowed). It evaluates to itself "
          "after validation (a non-matrix raises TypeError, a non-square or 1 x 1 matrix ValueError). Applying it, "
          "tf[{x1, ..., xn}] or tf[{p1, p2, ...}], gives the image point(s); a point of the wrong length raises "
          "ValueError, a point mapped to infinity ValueError, and a symbolic argument stays unevaluated. "
          "TransformationMatrix[tf] (W08) gives m back. Composition of two transformation functions is applied "
          "as nested calls (no matrix product is formed). Check: exact; tests compare images with hand-computed "
          "matrix products and the round trip tf[InverseImage] for affine maps.")
def _transformation_function(m: MatrixOf[Expr]) -> Expr:
    M = mat(m, "TransformationFunction", "transformation matrix", square=True)
    if M.shape[0] < 2:
        raise ValueError("TransformationFunction: the transformation matrix must be at least 2 x 2")
    raise Unevaluated()


@fn("AffineTransform", area=AREA, status="added",
    notes="AffineTransform[m] (linear map x -> m.x) and AffineTransform[{m, v}] (x -> m.x + v) for a square matrix m "
          "and vector v of matching length; gives TransformationFunction[{{m, v}, {0, 1}}]. Non-square m or a "
          "length mismatch raise ValueError, non-matrix input TypeError; a symbolic argument stays unevaluated. "
          "Check: exact; tests apply the result to points and compare with m.x + v.")
def _affine_transform(spec: MatrixOf[Expr] | ListOf[Expr]) -> Expr:
    f = "AffineTransform"
    if K.is_list(spec) and len(spec.args) == 2 and K.is_list(spec.args[0]) and spec.args[0].args and \
            K.is_list(spec.args[0].args[0]) and K.is_list(spec.args[1]) and spec.args[1].args and \
            not K.is_list(spec.args[1].args[0]):
        A = mat(spec.args[0], f, square=True)
        b = vec(spec.args[1], f, "translation vector")
        if len(b) != A.shape[0]:
            raise ValueError(f"{f}: the translation vector must have length {A.shape[0]}")
        return tf_expr(homog(A, b), spec)
    A = mat(spec, f, square=True)
    return tf_expr(homog(A), spec)


@fn("LinearFractionalTransform", area=AREA, status="added",
    notes="LinearFractionalTransform[{A, b, c, d}] gives the transformation x -> (A.x + b) / (c.x + d) (A an n x n "
          "matrix, b and c vectors of length n, d a scalar), and LinearFractionalTransform[m] the transformation "
          "with (n + 1) x (n + 1) homogeneous matrix m. Shape mismatches raise ValueError. Points where "
          "c.x + d == 0 raise ValueError when the function is applied. Check: exact; tests compare images with the "
          "formula (A.x + b) / (c.x + d).")
def _linear_fractional_transform(spec: ListOf[Expr] | MatrixOf[Expr]) -> Expr:
    f = "LinearFractionalTransform"
    if K.is_list(spec) and len(spec.args) == 4 and K.is_list(spec.args[0]) and spec.args[0].args and \
            K.is_list(spec.args[0].args[0]) and not K.is_list(spec.args[3]):
        A = mat(spec.args[0], f, square=True)
        n = A.shape[0]
        b = vec(spec.args[1], f, "vector b", dim=n)
        c = vec(spec.args[2], f, "vector c", dim=n)
        d = scal(spec.args[3], f, "d")
        M = sp.zeros(n + 1, n + 1)
        M[:n, :n] = A
        for i in range(n):
            M[i, n] = b[i]
            M[n, i] = c[i]
        M[n, n] = d
        return tf_expr(M, spec)
    M = mat(spec, f, square=True)
    if M.shape[0] < 2:
        raise ValueError(f"{f}: the homogeneous matrix must be at least 2 x 2")
    return tf_expr(M, spec)


@fn("TranslationTransform", area=AREA, status="added",
    notes="TranslationTransform[{v1, ..., vn}] gives the TransformationFunction of x -> x + v (any dimension, "
          "symbolic entries allowed). A non-list raises TypeError, an empty list ValueError. Check: exact; tests "
          "apply it to points.")
def _translation_transform(v: ListOf[Real]) -> Expr:
    b = vec(v, "TranslationTransform", "translation vector")
    return tf_expr(homog(sp.eye(len(b)), b), v)


def unit(v):
    nrm = sp.sqrt(sum(c ** 2 for c in v))
    return [c / nrm for c in v]


def plane_rotation(e1, e2, c, s):
    """Rotation by the angle with cosine c and sine s in the plane of orthonormal e1, e2 (from e1 towards e2)."""
    n = len(e1)
    E1, E2 = sp.Matrix(e1), sp.Matrix(e2)
    return sp.eye(n) + (c - 1) * (E1 * E1.T + E2 * E2.T) + s * (E2 * E1.T - E1 * E2.T)


def orthonormal_pair(u, v, f):
    _nonzero_vec(u, f, "first vector")
    e1 = unit(u)
    d = sum(a * b for a, b in zip(v, e1))
    w = [b - d * a for a, b in zip(e1, v)]
    if all(_zero(c) for c in w):
        raise ValueError(f"{f}: the two vectors must be linearly independent")
    return e1, unit(w)


def axis_rotation(t, w):
    """Right-handed rotation by t about the 3-D axis w (Rodrigues)."""
    k = sp.Matrix(unit(w))
    Kx = sp.Matrix([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return sp.eye(3) * sp.cos(t) + (1 - sp.cos(t)) * (k * k.T) + sp.sin(t) * Kx


def rotation_matrix_spec(args, f):
    """Linear part and centre of RotationTransform[args]."""
    a = list(args)
    if not a or len(a) > 3:
        raise ValueError(f"{f}: expected 1 to 3 arguments")
    first = a[0]
    if K.is_list(first) and len(first.args) == 2 and all(K.is_list(x) for x in first.args):   # {u, v}
        u = vec(first.args[0], f, "vector u")
        v = vec(first.args[1], f, "vector v", dim=len(u))
        if len(a) > 2:
            raise ValueError(f"{f}: expected RotationTransform[{{u, v}}] or RotationTransform[{{u, v}}, p]")
        p = vec(a[1], f, "centre", dim=len(u)) if len(a) == 2 else None
        _nonzero_vec(u, f, "vector u")
        _nonzero_vec(v, f, "vector v")
        uh, vh = unit(u), unit(v)
        c = _simp(sum(x * y for x, y in zip(uh, vh)))
        w = [y - c * x for x, y in zip(uh, vh)]
        if all(_zero(z) for z in w):
            if len(u) == 2 or _zero(c - 1):
                A = sp.eye(len(u)) if _zero(c - 1) else -sp.eye(2)
            else:
                raise NotImplementedError(f"{f}: the rotation taking a vector to its opposite is not unique in "
                                          f"{len(u)} dimensions; give the rotation plane")
        else:
            e2 = unit(w)
            s = _simp(sum(x * y for x, y in zip(e2, vh)))
            A = plane_rotation(uh, e2, c, s)
        return A, p
    t = scal(first, f, "angle")
    if len(a) == 1:
        return sp.Matrix([[sp.cos(t), -sp.sin(t)], [sp.sin(t), sp.cos(t)]]), None
    second = a[1]
    if K.is_list(second) and len(second.args) == 2 and all(K.is_list(x) for x in second.args):     # θ, {u, v}
        u = vec(second.args[0], f, "vector u")
        v = vec(second.args[1], f, "vector v", dim=len(u))
        e1, e2 = orthonormal_pair(u, v, f)
        p = vec(a[2], f, "centre", dim=len(u)) if len(a) == 3 else None
        return plane_rotation(e1, e2, sp.cos(t), sp.sin(t)), p
    w = vec(second, f, "axis or centre")
    if len(w) == 2:
        if len(a) == 3:
            raise ValueError(f"{f}: RotationTransform[θ, p] in 2-D takes no third argument")
        return sp.Matrix([[sp.cos(t), -sp.sin(t)], [sp.sin(t), sp.cos(t)]]), w
    if len(w) == 3:
        _nonzero_vec(w, f, "rotation axis")
        p = vec(a[2], f, "centre", dim=3) if len(a) == 3 else None
        return axis_rotation(t, w), p
    raise ValueError(f"{f}: RotationTransform[θ, w] needs a 2-D centre or a 3-D axis; use RotationTransform[θ, "
                     f"{{u, v}}] in {len(w)} dimensions")


def _with_centre(A, p):
    return homog(A) if p is None else about_point(A, p)


@fn("RotationTransform", area=AREA, status="added",
    notes="RotationTransform[θ] (2-D rotation about the origin), RotationTransform[θ, p] (about the 2-D point p), "
          "RotationTransform[θ, w] / RotationTransform[θ, w, p] (right-handed rotation about the 3-D axis w, "
          "through p), RotationTransform[θ, {u, v}] / RotationTransform[θ, {u, v}, p] (rotation by θ in the plane "
          "spanned by u and v, from u towards v, any dimension) and RotationTransform[{u, v}] / "
          "RotationTransform[{u, v}, p] (the rotation in the plane of u and v taking the direction of u to that of "
          "v). Symbolic angles give symbolic matrices. A zero axis or vector and linearly dependent {u, v} in "
          "RotationTransform[θ, {u, v}] raise ValueError; RotationTransform[{u, -u}] in 3 or more dimensions raises "
          "NotImplementedError (no unique plane). Check: exact; tests compare with RotationMatrix and check that "
          "{u, v} rotations map u to the direction of v.")
def _rotation_transform(*args: Real | ListOf[Real] | MatrixOf[Real]) -> Expr:
    A, p = rotation_matrix_spec(args, "RotationTransform")
    return tf_expr(_with_centre(A, p), *args)


@fn("ScalingTransform", area=AREA, status="added",
    notes="ScalingTransform[{s1, ..., sn}] (scale coordinate i by si), ScalingTransform[{s1, ...}, p] (about the "
          "point p), ScalingTransform[s, v] (scale by s along the direction v) and ScalingTransform[s, v, p]. A zero "
          "direction raises ValueError, dimension mismatches ValueError. Check: exact; tests apply it to points.")
def _scaling_transform(s: Real | ListOf[Real], *rest: ListOf[Real]) -> Expr:
    f = "ScalingTransform"
    if K.is_list(s):
        d = vec(s, f, "scale factors")
        if len(rest) > 1:
            raise ValueError(f"{f}: expected ScalingTransform[{{s1, ...}}] or ScalingTransform[{{s1, ...}}, p]")
        p = vec(rest[0], f, "centre", dim=len(d)) if rest else None
        return tf_expr(_with_centre(sp.diag(*d), p), s, *rest)
    if not rest and isinstance(s, Symbol):
        raise Unevaluated()
    k = scal(s, f, "scale factor")
    if not rest or len(rest) > 2:
        raise ValueError(f"{f}: expected ScalingTransform[s, v] or ScalingTransform[s, v, p]")
    v = vec(rest[0], f, "direction")
    _nonzero_vec(v, f, "direction")
    V = sp.Matrix(v)
    A = sp.eye(len(v)) + (k - 1) * (V * V.T) / (V.T * V)[0]
    p = vec(rest[1], f, "centre", dim=len(v)) if len(rest) == 2 else None
    return tf_expr(_with_centre(A, p), s, *rest)


@fn("ReflectionTransform", area=AREA, status="added",
    notes="ReflectionTransform[v] (reflection in the hyperplane through the origin normal to v) and "
          "ReflectionTransform[v, p] (hyperplane through p); matrix I - 2 v.v^T / v.v. A zero normal raises "
          "ValueError. Check: exact; tests check the image of points and that applying it twice is the identity.")
def _reflection_transform(v: ListOf[Real], *p: ListOf[Real]) -> Expr:
    f = "ReflectionTransform"
    w = vec(v, f, "normal vector")
    _nonzero_vec(w, f, "normal vector")
    if len(p) > 1:
        raise ValueError(f"{f}: expected ReflectionTransform[v] or ReflectionTransform[v, p]")
    V = sp.Matrix(w)
    A = sp.eye(len(w)) - 2 * (V * V.T) / (V.T * V)[0]
    c = vec(p[0], f, "point", dim=len(w)) if p else None
    return tf_expr(_with_centre(A, c), v, *p)


@fn("ShearingTransform", area=AREA, status="added",
    notes="ShearingTransform[θ, u, v] (shear by θ radians along the direction u, normal to v: x -> x + Tan[θ] "
          "(v^.x) u^ with unit vectors u^, v^) and ShearingTransform[θ, u, v, p] (fixed point p). u and v must be "
          "nonzero (else ValueError) and orthogonal (non-orthogonal pairs raise NotImplementedError). Check: exact; "
          "tests compare with the documented matrix {{1, Tan[θ]}, {0, 1}} for u = {1, 0}, v = {0, 1}.")
def _shearing_transform(t: Real, u: ListOf[Real], v: ListOf[Real], *p: ListOf[Real]) -> Expr:
    f = "ShearingTransform"
    th = scal(t, f, "angle")
    a = vec(u, f, "shear direction")
    b = vec(v, f, "normal vector", dim=len(a))
    _nonzero_vec(a, f, "shear direction")
    _nonzero_vec(b, f, "normal vector")
    if not _zero(sum(x * y for x, y in zip(a, b))):
        if any(sp.sympify(c).free_symbols for c in a + b):
            raise Unevaluated()
        raise NotImplementedError(f"{f}: the shear direction and normal vector must be orthogonal")
    if len(p) > 1:
        raise ValueError(f"{f}: expected 3 or 4 arguments")
    A = sp.eye(len(a)) + sp.tan(th) * sp.Matrix(unit(a)) * sp.Matrix(unit(b)).T
    c = vec(p[0], f, "point", dim=len(a)) if p else None
    return tf_expr(_with_centre(A, c), t, u, v, *p)


def _ranges(x, f, what):
    if not (K.is_list(x) and x.args and all(K.is_list(r) and len(r.args) == 2 for r in x.args)):
        if _is_sym(x):
            raise Unevaluated()
        raise TypeError(f"{f}: {what} must be a list of ranges {{{{min, max}}, ...}}, not {K_input(x)}")
    return [(scal(r.args[0], f, "range bound"), scal(r.args[1], f, "range bound")) for r in x.args]


@fn("RescalingTransform", area=AREA, status="added",
    notes="RescalingTransform[{{xmin, xmax}, ...}, {{ymin, ymax}, ...}] maps the box of x ranges affinely onto the "
          "box of y ranges (coordinate-wise, xmin -> ymin, xmax -> ymax); RescalingTransform[{{xmin, xmax}, ...}] "
          "maps it onto the unit box [0, 1]^n. A degenerate range (xmin == xmax) or a dimension mismatch raises "
          "ValueError. Check: exact; tests check that range end points map to the target end points.")
def _rescaling_transform(src: MatrixOf[Real], *dst: MatrixOf[Real]) -> Expr:
    f = "RescalingTransform"
    a = _ranges(src, f, "source ranges")
    if len(dst) > 1:
        raise ValueError(f"{f}: expected 1 or 2 arguments")
    b = _ranges(dst[0], f, "target ranges") if dst else [(sp.Integer(0), sp.Integer(1))] * len(a)
    if len(a) != len(b):
        raise ValueError(f"{f}: source and target ranges must have the same dimension")
    scale, shift = [], []
    for (x1, x2), (y1, y2) in zip(a, b):
        if _zero(x2 - x1):
            raise ValueError(f"{f}: the source range {{{x1}, {x2}}} is degenerate")
        k = (y2 - y1) / (x2 - x1)
        scale.append(k)
        shift.append(y1 - k * x1)
    return tf_expr(homog(sp.diag(*scale), shift), src, *dst)


# ───────────────────────────── Euler / roll-pitch-yaw ─────────────────────────────

def axis_matrix(k, t):
    """RotationMatrix[t, UnitVector[3, k]]."""
    c, s = sp.cos(t), sp.sin(t)
    if k == 1:
        return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    if k == 2:
        return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _angles3(x, f):
    v = vec(x, f, "angle list", dim=3)
    return v


def _axes(x, f):
    if not (K.is_list(x) and len(x.args) == 3):
        if _is_sym(x):
            raise Unevaluated()
        raise TypeError(f"{f}: the rotation axes must be a list {{a, b, c}} of integers 1, 2 or 3")
    ax = []
    for c in x.args:
        if isinstance(c, bool) or not K.is_int(c):
            raise TypeError(f"{f}: rotation axes must be integers 1, 2 or 3, not {K_input(c)}")
        if int(c) not in (1, 2, 3):
            raise ValueError(f"{f}: rotation axes must be 1, 2 or 3, not {int(c)}")
        ax.append(int(c))
    return ax


def _mat_out(M, *inputs):
    M = M.applyfunc(lambda e: _simp(e) if not sp.sympify(e).free_symbols else e)
    return out([[M[i, j] for j in range(3)] for i in range(3)], *inputs)


def euler_matrix(ang, ax):
    return axis_matrix(ax[0], ang[0]) * axis_matrix(ax[1], ang[1]) * axis_matrix(ax[2], ang[2])


def _opts_target(opts, f):
    for r in opts:
        n = opt_name(r)
        if n != "TargetStructure":
            raise ValueError(f"{f}: {n} is not a known option of {f}")
        v = r.args[1]
        if not (isinstance(v, Symbol) and v.name == "Automatic") and v not in ("Dense",):
            raise NotImplementedError(f"{f}: TargetStructure -> {K_input(v)} is not supported (only Automatic and "
                                      f"\"Dense\")")


@fn("EulerMatrix", area=AREA, status="added",
    notes="EulerMatrix[{α, β, γ}] = Rz(α).Ry(β).Rz(γ) (rotations about the moving z, y, z axes) and "
          "EulerMatrix[{α, β, γ}, {a, b, c}] = R_a(α).R_b(β).R_c(γ) with R_k(t) = RotationMatrix[t, UnitVector[3, "
          "k]] (any axes 1, 2, 3, including repeated consecutive ones); symbolic angles give symbolic matrices. "
          "Option TargetStructure -> Automatic | \"Dense\" (others raise NotImplementedError). Axes outside 1..3 "
          "raise ValueError. Check: exact; tests compare with products of RotationMatrix and the documented "
          "relation RollPitchYawMatrix[{α, β, γ}, {a, b, c}] == EulerMatrix[{γ, β, α}, {c, b, a}].")
def _euler_matrix(angles: ListOf[Real], *rest: ListOf[Integer] | Option) -> MatrixOf[Real]:
    f = "EulerMatrix"
    a, opts = split_trailing_options(list(rest))
    _opts_target(opts, f)
    if len(a) > 1:
        raise ValueError(f"{f}: expected EulerMatrix[{{α, β, γ}}] or EulerMatrix[{{α, β, γ}}, {{a, b, c}}]")
    ang = _angles3(angles, f)
    ax = _axes(a[0], f) if a else [3, 2, 3]
    return _mat_out(euler_matrix(ang, ax), angles)


@fn("RollPitchYawMatrix", area=AREA, status="added",
    notes="RollPitchYawMatrix[{α, β, γ}] = Rx(γ).Ry(β).Rz(α) (rotate by α about the fixed z axis, then β about y, "
          "then γ about x; axes {3, 2, 1}) and RollPitchYawMatrix[{α, β, γ}, {a, b, c}] = R_c(γ).R_b(β).R_a(α). "
          "Option TargetStructure -> Automatic | \"Dense\". Check: exact; tests compare with the documented "
          "product RotationMatrix[γ, x].RotationMatrix[β, y].RotationMatrix[α, z].")
def _rpy_matrix(angles: ListOf[Real], *rest: ListOf[Integer] | Option) -> MatrixOf[Real]:
    f = "RollPitchYawMatrix"
    a, opts = split_trailing_options(list(rest))
    _opts_target(opts, f)
    if len(a) > 1:
        raise ValueError(f"{f}: expected RollPitchYawMatrix[{{α, β, γ}}] or RollPitchYawMatrix[{{α, β, γ}}, "
                         f"{{a, b, c}}]")
    ang = _angles3(angles, f)
    ax = _axes(a[0], f) if a else [3, 2, 1]
    M = axis_matrix(ax[2], ang[2]) * axis_matrix(ax[1], ang[1]) * axis_matrix(ax[0], ang[0])
    return _mat_out(M, angles)


@functools.lru_cache(maxsize=None)
def _euler_recipe(ax):
    """For axes (a, b, c) with a != b != c: how to read α, β, γ off M = R_a(α) R_b(β) R_c(γ).

    Returns (kind, (i, j, sgn) for the β entry, α entries, γ entries, lock recipe) found symbolically."""
    al, be, ga = sp.symbols("al be ga", real=True)
    M = euler_matrix([al, be, ga], list(ax))
    proper = ax[0] == ax[2]
    target = sp.cos(be) if proper else sp.sin(be)
    trig = sp.sin(be) if proper else sp.cos(be)
    bentry = None
    for i in range(3):
        for j in range(3):
            e = sp.simplify(M[i, j])
            if sp.simplify(e - target) == 0:
                bentry = (i, j, 1)
            elif sp.simplify(e + target) == 0:
                bentry = (i, j, -1)
    rec = {}
    for i in range(3):
        for j in range(3):
            e = sp.simplify(M[i, j])
            for name, s in (("sa", sp.sin(al)), ("ca", sp.cos(al)), ("sg", sp.sin(ga)), ("cg", sp.cos(ga))):
                for sg in (1, -1):
                    if sp.simplify(e - sg * trig * s) == 0 and name not in rec:
                        rec[name] = (i, j, sg)
    # gimbal lock: γ = 0 and β at the lock value; read α from entries equal to ±sin α / ±cos α
    locks = {}
    for bval in ((0, sp.pi) if proper else (sp.pi / 2, -sp.pi / 2)):
        Ml = M.subs({ga: 0, be: bval})
        r = {}
        for i in range(3):
            for j in range(3):
                e = sp.simplify(Ml[i, j])
                for name, s in (("sa", sp.sin(al)), ("ca", sp.cos(al))):
                    for sg in (1, -1):
                        if sp.simplify(e - sg * s) == 0 and name not in r:
                            r[name] = (i, j, sg)
        locks[bval] = r
    return proper, bentry, rec, locks


def _is_rotation(M, f):
    flt = any(sp.sympify(e).has(sp.Float) for e in M)
    if flt:
        import numpy as np
        A = np.array([[float(sp.N(e)) for e in row] for row in M.tolist()])
        if not (np.allclose(A @ A.T, np.eye(3), atol=1e-8) and abs(np.linalg.det(A) - 1) < 1e-8):
            raise ValueError(f"{f}: the matrix is not a rotation matrix (orthogonal with determinant 1)")
        return True
    if any(_zero(e) is False and sp.sympify(e).free_symbols for e in M):
        raise Unevaluated()
    G = (M * M.T - sp.eye(3)).applyfunc(sp.simplify)
    if any(not _zero(e) for e in G) or not _zero(sp.simplify(M.det() - 1)):
        raise ValueError(f"{f}: the matrix is not a rotation matrix (orthogonal with determinant 1)")
    return False


def _atan2(y, x, flt):
    if flt:
        import math
        return sp.Float(math.atan2(float(sp.N(y)), float(sp.N(x))))
    y, x = _simp(y), _simp(x)
    return _simp(sp.atan2(y, x))


def euler_angles(M, ax, f):
    """Angles {α, β, γ} with euler_matrix(angles, ax) == M (self-checked)."""
    if ax[0] == ax[1] or ax[1] == ax[2]:
        raise ValueError(f"{f}: consecutive rotation axes must be distinct for the angles to be determined")
    flt = _is_rotation(M, f)
    proper, (bi, bj, bs), rec, locks = _euler_recipe(tuple(ax))
    v = bs * M[bi, bj]
    if flt:
        import math
        vv = max(-1.0, min(1.0, float(sp.N(v))))
        beta = sp.Float(math.acos(vv) if proper else math.asin(vv))
    else:
        v = _simp(v)
        beta = _simp(sp.acos(v) if proper else sp.asin(v))
    lockval = None
    if proper:
        trig = sp.sin(beta)
        for bv in (0, sp.pi):
            if _zero(beta - bv):
                lockval = bv
    else:
        trig = sp.cos(beta)
        for bv in (sp.pi / 2, -sp.pi / 2):
            if _zero(beta - bv):
                lockval = bv
    if lockval is None:
        def rd(name):
            i, j, s = rec[name]
            return s * M[i, j]
        alpha = _atan2(rd("sa"), rd("ca"), flt)
        gamma = _atan2(rd("sg"), rd("cg"), flt)
    else:
        r = locks[lockval]
        i1, j1, s1 = r["sa"]
        i2, j2, s2 = r["ca"]
        alpha = _atan2(s1 * M[i1, j1], s2 * M[i2, j2], flt)
        gamma = sp.Float(0) if flt else sp.Integer(0)
        beta = sp.Float(float(lockval)) if flt else lockval
    ang = [alpha, beta, gamma]
    chk = euler_matrix(ang, ax) - M
    if flt:
        bad = any(abs(complex(sp.N(e))) > 1e-8 for e in chk)
    else:
        bad = any(not _zero(sp.simplify(e)) for e in chk)
    if bad:
        raise ArithmeticError(f"{f}: result failed verification")
    return ang


@fn("EulerAngles", area=AREA, status="added",
    notes="EulerAngles[r] gives angles {α, β, γ} with EulerMatrix[{α, β, γ}] == r (z-y-z), and EulerAngles[r, "
          "{a, b, c}] the angles for rotation order {a, b, c} (the twelve orders with distinct consecutive axes; "
          "repeated consecutive axes raise ValueError). Ranges: β in [0, Pi] for a == c, β in [-Pi/2, Pi/2] "
          "otherwise; α, γ in (-Pi, Pi]; at gimbal lock γ = 0. Exact input gives exact angles, inexact input "
          "machine numbers. A matrix that is not orthogonal with determinant 1 raises ValueError; a symbolic matrix "
          "stays unevaluated. Deviation: Wolfram may choose another equivalent angle triple. Check: self-check "
          "(EulerMatrix of the result is compared with r; mismatch raises ArithmeticError).")
def _euler_angles(r: MatrixOf[Real], *axes: ListOf[Integer]) -> ListOf[Real]:
    f = "EulerAngles"
    M = mat(r, f, "rotation matrix", square=True)
    if M.shape != (3, 3):
        raise ValueError(f"{f}: the rotation matrix must be 3 x 3")
    if len(axes) > 1:
        raise ValueError(f"{f}: expected EulerAngles[r] or EulerAngles[r, {{a, b, c}}]")
    ax = _axes(axes[0], f) if axes else [3, 2, 3]
    return out(euler_angles(M, ax, f), r)


@fn("RollPitchYawAngles", area=AREA, status="added",
    notes="RollPitchYawAngles[r] gives {α, β, γ} with RollPitchYawMatrix[{α, β, γ}] == r (axes {3, 2, 1}), and "
          "RollPitchYawAngles[r, {a, b, c}] for another order (consecutive axes must differ, else ValueError). "
          "Computed as the reversed EulerAngles[r, {c, b, a}] (documented identity RollPitchYawMatrix[{α, β, γ}, "
          "{a, b, c}] == EulerMatrix[{γ, β, α}, {c, b, a}]). Non-rotation matrices raise ValueError. Deviation: "
          "Wolfram may return another equivalent triple. Check: self-check (the matrix is rebuilt and compared).")
def _rpy_angles(r: MatrixOf[Real], *axes: ListOf[Integer]) -> ListOf[Real]:
    f = "RollPitchYawAngles"
    M = mat(r, f, "rotation matrix", square=True)
    if M.shape != (3, 3):
        raise ValueError(f"{f}: the rotation matrix must be 3 x 3")
    if len(axes) > 1:
        raise ValueError(f"{f}: expected RollPitchYawAngles[r] or RollPitchYawAngles[r, {{a, b, c}}]")
    ax = _axes(axes[0], f) if axes else [3, 2, 1]
    g, b, a = euler_angles(M, ax[::-1], f)
    return out([a, b, g], r)


# ───────────────────────────── GeometricTransformation ─────────────────────────────

def _check_transform_spec(t, f):
    if K.is_list(t) and t.args and all(isinstance(x, Expr) and x.head == "TransformationFunction" for x in t.args):
        for x in t.args:
            as_transform(x, f)
        return
    if K.is_list(t) and t.args and all(K.is_list(x) and x.args and all(K.is_list(r) for r in x.args)
                                        and len(x.args) == len(x.args[0].args) for x in t.args) and \
            len(t.args) != 2 and not all(not K.is_list(c) for c in t.args[0].args):
        for x in t.args:
            as_transform(x, f)
        return
    as_transform(t, f)


@fn("GeometricTransformation", area=AREA, status="added",
    notes="GeometricTransformation[g, tf] represents the graphics g transformed by tf, where tf is a "
          "TransformationFunction, a square matrix m (x -> m.x), an affine pair {m, v} (x -> m.x + v) or a list of "
          "transformation functions (one copy of g per transformation). It is an inert graphics wrapper (accepted "
          "inside Graphics / Graphics3D); the transformation is validated (a non-transformation raises TypeError, "
          "non-square matrices ValueError). Check: reference (Wolfram documentation forms) in tests.")
def _geometric_transformation(g: Expr, t: Expr) -> Expr:
    _check_transform_spec(t, "GeometricTransformation")
    raise Unevaluated()
