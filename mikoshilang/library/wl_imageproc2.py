"""Wolfram batch X04: images III (linear and nonlinear filters, restoration, multiscale pyramids, composition).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md). It builds on W21 (``wl_images``: the ``Image`` expression and ``Img`` values on the 0..1
scale, rows from the top) and W28 (``wl_imageproc``: ``data_arg``, padding, separable correlation, Gaussian
kernels).

Where Wolfram does not document an algorithm or a default, the function uses a named published method and its
notes state the method and the defaults (rather than claiming to reproduce Wolfram's undocumented choices).

Python API for later batches (X05 image features): the multiscale helpers ``BURT_ADELSON``,
``pyr_reduce(a, nd, kernel, pad)``, ``pyr_expand(a, shape, nd, kernel, pad)``, ``gaussian_levels(a, nd, n, ...)``,
``laplacian_levels(a, nd, n, ...)``, ``collapse_laplacian(levels, nd, ...)`` and ``pyramid_level_count(shape)``
work on float arrays shaped (spatial..., channels); ``spline_derivative(a, orders, ...)`` and ``hessian(a, nd,
sigma, ...)`` give B-spline (Unser) derivatives of such arrays.

Pure NumPy does everything here; nothing needs SciPy or Pillow.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, R, S
from . import wl_images as WI
from . import wl_imageproc as WP
from .wl_images import parse_image, image_expr, opts_split, int_arg, real_arg, IMAGE_HEADS, _short, _is_sym, \
    quantize
from .wl_imageproc import data_arg, array_out, image_out, real_type, correlate_axis, separable, correlate_nd, \
    _PAD_MODES, _gk

AREA = "image processing III (X04)"
AUTO = Symbol("Automatic")
INF_NAMES = ("Infinity", "DirectedInfinity")


def _ev(e):
    from ..evaluate import evaluate
    return evaluate(e)


def _sym(x):
    return isinstance(x, Symbol) or (isinstance(x, Expr) and not K.is_list(x) and x.head not in
                                     ("Rule", "RuleDelayed", "Association") + IMAGE_HEADS)


def _is_inf(x):
    return _is_sym(x, "Infinity") or (isinstance(x, Expr) and x.head == "DirectedInfinity" and x.args == (1,)) or \
        (isinstance(x, sp.Basic) and x == sp.oo)


def _num(x, fname, what, lo=None, strict=False, allow_inf=False):
    """A real number argument (float); symbolic input -> Unevaluated, bad kinds -> TypeError, range -> ValueError."""
    if allow_inf and _is_inf(x):
        return math.inf
    if isinstance(x, (bool, str)) or K.is_list(x) or (isinstance(x, Expr) and x.head in IMAGE_HEADS + ("Rule",)):
        raise TypeError(f"{fname}: {what} must be a real number, not {_short(x)}")
    if not (isinstance(x, (int, float)) or WI.is_real_value(x)):
        if isinstance(x, Symbol) or isinstance(x, Expr):
            raise Unevaluated()
        raise TypeError(f"{fname}: {what} must be a real number, not {_short(x)}")
    v = real_arg(x, fname, what)
    if lo is not None and (v < lo or (strict and v == lo)):
        raise ValueError(f"{fname}: {what} must be {'greater than' if strict else 'at least'} {lo:g}, not {v:g}")
    return v


def _image(x, fname, allow3d=True):
    """Img of an image argument: TypeError for non-images, Unevaluated for symbols."""
    if isinstance(x, Expr) and x.head in IMAGE_HEADS:
        return parse_image(x, fname, allow3d=allow3d)
    if isinstance(x, Symbol) or (isinstance(x, Expr) and not K.is_list(x) and x.head not in ("Rule", "Association",
                                                                                              "Graphics")):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_short(x)} is not an image")


def _opt_check(pos, fname, maxpos):
    if len(pos) > maxpos:
        raise TypeError(f"{fname}: too many arguments ({_short(pos[maxpos])} is not an option)")


def _pad_opt(o, fname, nd, default="Fixed", allow_list=True):
    """Padding option -> list of per-axis settings (method name, number or None)."""
    p = o.get("Padding", default)
    items = list(p.args) if (allow_list and K.is_list(p)) else [p]
    out = []
    for v in items:
        if _is_sym(v, "None"):
            out.append(None)
        elif isinstance(v, str):
            if v not in _PAD_MODES:
                raise ValueError(f"{fname}: unknown Padding {v!r} (\"Fixed\", \"Periodic\", \"Reflected\", "
                                 "\"Reversed\", a number or None)")
            out.append(v)
        elif WI.is_real_value(v) or (isinstance(v, (int, float)) and not isinstance(v, bool)):
            out.append(real_arg(v, fname, "Padding"))
        else:
            raise NotImplementedError(f"{fname}: Padding {_short(v)} is not supported")
    if len(out) == 1:
        out = out * nd
    if len(out) != nd:
        raise ValueError(f"{fname}: give one Padding setting or one per dimension ({nd})")
    return out


def pad_axis(a, ax, before, after, pad):
    """Pad one axis of an array (float or object) by a padding setting (method name or number)."""
    if before == 0 and after == 0:
        return a
    spec = [(0, 0)] * a.ndim
    spec[ax] = (before, after)
    if isinstance(pad, str):
        mode = _PAD_MODES[pad]
        if mode == "reflect" and a.shape[ax] < 2:
            mode = "edge"
        return np.pad(a, spec, mode=mode)
    return np.pad(a, spec, mode="constant", constant_values=pad)


def correlate_axis_padded(a, k, ax, pad):
    """1-D correlation along ax with an odd kernel; pad None gives the shrunken ('valid') result."""
    r = len(k) // 2
    P = a if pad is None else pad_axis(a, ax, r, r, pad)
    n = P.shape[ax] - 2 * r
    if n <= 0:
        raise ValueError("the data are smaller than the kernel (Padding -> None)")
    out = None
    for j, kj in enumerate(k):
        if kj != 0:
            t = kj * np.take(P, range(j, j + n), axis=ax)
            out = t if out is None else out + t
    if out is None:
        out = 0 * np.take(P, range(0, n), axis=ax)
    return out


# ───────────────────────────── precision (WorkingPrecision) ─────────────────────────────

def _wp(o, fname):
    """WorkingPrecision -> None (Automatic), 'm' (machine), 'exact' (Infinity) or digits."""
    w = o.get("WorkingPrecision", AUTO)
    if _is_sym(w, "Automatic"):
        return None
    if _is_sym(w, "MachinePrecision"):
        return "m"
    if _is_inf(w):
        return "exact"
    v = _num(w, fname, "WorkingPrecision", lo=1)
    return int(round(v))


# ───────────────────────────── LaplacianFilter ─────────────────────────────

def fd_weights(r, order=2):
    """Exact central finite-difference weights (offsets -r..r) of the given derivative order with the maximal
    order of accuracy (Fornberg 1988): the unique w with Σ_j w_j j^m = m! δ(m, order) for m = 0..2r."""
    if r == 0:
        return [Fraction(0)]
    n = 2 * r + 1
    offs = list(range(-r, r + 1))
    A = [[Fraction(j) ** m for j in offs] for m in range(n)]
    b = [Fraction(math.factorial(order)) if m == order else Fraction(0) for m in range(n)]
    # Gauss-Jordan elimination with exact fractions
    M = [row[:] + [bb] for row, bb in zip(A, b)]
    for c in range(n):
        p = next(i for i in range(c, n) if M[i][c] != 0)
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for i in range(n):
            if i != c and M[i][c] != 0:
                f = M[i][c]
                M[i] = [vi - f * vc for vi, vc in zip(M[i], M[c])]
    return [M[i][n] for i in range(n)]


def _radii_list(r, fname, nd, what="radius"):
    if K.is_list(r):
        if len(r.args) != nd:
            raise ValueError(f"{fname}: give one {what} per dimension of the data ({nd}), not {len(r.args)}")
        vals = list(r.args)
    else:
        vals = [r] * nd
    out = []
    for v in vals:
        if isinstance(v, (bool, str)) or K.is_list(v):
            raise TypeError(f"{fname}: the {what} must be a non-negative integer, not {_short(v)}")
        if not K.is_int(v):
            if isinstance(v, float) or WI.is_real_value(v):
                raise NotImplementedError(f"{fname}: non-integer {what} {_short(v)} is not supported")
            raise Unevaluated()
        if int(v) < 0:
            raise ValueError(f"{fname}: the {what} must be non-negative, not {v}")
        out.append(int(v))
    return out


def _exact_input(x, fname):
    d = data_arg(x, fname)
    return d


def _to_obj(a):
    o = np.empty(a.shape, dtype=object)
    o.flat[:] = [sp.Float(float(v)) for v in a.flat]
    return o


@fn("LaplacianFilter", area=AREA, status="added",
    notes="LaplacianFilter[data, r] convolves data with a radius-r Laplacian kernel: the sum over the dimensions of "
          "the (2r+1)-point central second-difference stencil along that dimension with the maximal order of "
          "accuracy (Fornberg 1988; r = 1 gives {1, -2, 1}, r = 2 gives {-1/12, 4/3, -5/2, 4/3, -1/12}); "
          "LaplacianFilter[data, {r1, r2, ...}] uses ri along dimension i (0 omits it). Wolfram does not document "
          "its radius-r kernel, so results for r >= 2 follow this stated stencil. Multichannel images are filtered "
          "per channel and give a real image. Padding -> \"Fixed\" (default), \"Periodic\", \"Reflected\", "
          "\"Reversed\", a number or None (smaller result). WorkingPrecision: Automatic (machine numbers for "
          "exact arrays, the input precision for bignums), MachinePrecision, Infinity (exact) or digits; ignored "
          "for images. Symbolic data stay unevaluated; non-integer radii raise NotImplementedError. "
          "Check: exact (WorkingPrecision -> Infinity); reference (stencil identities, exact second differences of "
          "polynomials) in tests.")
def _laplacianfilter(x: Expr, r: NonNegInteger | ListOf[NonNegInteger], *opts: Option) -> Expr:
    fname = "LaplacianFilter"
    pos, o = opts_split(opts, {"Padding", "WorkingPrecision"}, fname)
    _opt_check(pos, fname, 0)
    d = data_arg(x, fname)
    nd = d.spatial
    radii = _radii_list(r, fname, nd)
    pads = _pad_opt(o, fname, nd, allow_list=False)
    wp = _wp(o, fname)
    weights = [fd_weights(ri) for ri in radii]
    if d.is_image:
        a = d.img.color()
        alpha = d.img.a[..., -1:] if d.img.alpha else None
        res = _lap_apply(a, weights, pads, float_mode=True)
        return WP._lin_out(d, res, alpha)
    if d.exact:
        a = d.obj
        has_float = any(isinstance(v, sp.Float) for v in a.flat)
        if wp == "exact" or (wp is None and has_float):
            res = _lap_apply(a, weights, pads, float_mode=False)
            return array_out(res)
        if isinstance(wp, int):
            res = _lap_apply(a, weights, pads, float_mode=False)
            return array_out(np.vectorize(lambda v: sp.Float(sp.N(v, wp), wp), otypes=[object])(res))
        a = np.array([float(v) for v in a.flat], dtype=float).reshape(a.shape)
    else:
        a = d.a
        if wp == "exact":
            raise ValueError(f"{fname}: WorkingPrecision -> Infinity needs exact data")
    return array_out(_lap_apply(a, weights, pads, float_mode=True))


def _lap_apply(a, weights, pads, float_mode):
    tot = None
    nd = len(weights)
    shrink = [len(w) // 2 if p is None else 0 for w, p in zip(weights, pads)]
    for ax, (w, p) in enumerate(zip(weights, pads)):
        if len(w) == 1:
            continue
        k = [float(v) for v in w] if float_mode else [sp.Rational(v.numerator, v.denominator) for v in w]
        try:
            t = correlate_axis_padded(a, k, ax, p)
        except ValueError:
            raise ValueError("LaplacianFilter: the data are smaller than the kernel (Padding -> None)")
        sl = [slice(None)] * a.ndim
        for j in range(nd):
            if j != ax and shrink[j]:
                sl[j] = slice(shrink[j], a.shape[j] - shrink[j])
        t = t[tuple(sl)]
        tot = t if tot is None else tot + t
    if tot is None:
        sl = tuple(slice(s, a.shape[j] - s) for j, s in enumerate(shrink)) + (slice(None),) * (a.ndim - nd)
        tot = 0 * a[sl]
    return tot


# ───────────────────────────── B-spline derivatives (DerivativeFilter, RidgeFilter) ─────────────────────────────

def bspline_values(n, d, offsets):
    """Exact values of the d-th derivative of the centred B-spline of degree n at the given (integer) offsets."""
    out = []
    for x in offsets:
        acc = Fraction(0)
        for j in range(n + 2):
            t = Fraction(x) + Fraction(n + 1, 2) - j
            if t > 0 and n - d >= 0:
                if n - d == 0:
                    term = Fraction(1)
                else:
                    term = t ** (n - d)
                acc += (-1) ** j * math.comb(n + 1, j) * term * Fraction(math.factorial(n), math.factorial(n - d))
        out.append(acc / math.factorial(n))
    return out


_SPLINE_CACHE = {}
SPLINE_MARGIN = 64


def _spline_operator(N, n, d):
    """(N x N) matrix mapping samples to the d-th derivative of their degree-n cardinal spline interpolant at the
    samples (collocation B c = f, derivative D c)."""
    key = (N, n, d)
    M = _SPLINE_CACHE.get(key)
    if M is not None:
        return M
    h = (n + 1) // 2 + 1
    offs = list(range(-h, h + 1))
    b = [float(v) for v in bspline_values(n, 0, offs)]
    db = [float(v) for v in bspline_values(n, d, offs)]
    B = np.zeros((N, N))
    D = np.zeros((N, N))
    for o, bv, dv in zip(offs, b, db):
        idx = np.arange(max(0, -o), min(N, N - o))
        B[idx + o, idx] = bv
        D[idx + o, idx] = dv
    M = D @ np.linalg.solve(B, np.eye(N))
    if len(_SPLINE_CACHE) > 32:
        _SPLINE_CACHE.clear()
    _SPLINE_CACHE[key] = M
    return M


def spline_derivative_axis(a, ax, d, n, pad):
    """d-th derivative along axis ax of the degree-n spline interpolant of a, after padding by SPLINE_MARGIN."""
    if d == 0:
        return a
    m = SPLINE_MARGIN
    P = pad_axis(a, ax, m, m, pad)
    M = _spline_operator(P.shape[ax], n, d)
    Q = np.moveaxis(np.tensordot(M, np.moveaxis(P, ax, 0), axes=(1, 0)), 0, ax)
    return np.take(Q, range(m, m + a.shape[ax]), axis=ax)


def gaussian_smooth(a, nd, sigma, pad, fname):
    """Separable smoothing of the leading nd axes with the sampled-Bessel (discrete Gaussian) kernel of standard
    deviation sigma and radius Ceiling[4 sigma] (GaussianMatrix Method -> "Bessel")."""
    if sigma <= 0:
        return a
    r = int(math.ceil(4 * sigma))
    k = _gk(fname, r, sigma, 0, "Bessel", True)
    for ax in range(nd):
        a = correlate_axis_padded(a, k, ax, pad[ax])
    return a


def spline_derivative(a, orders, n, pads, sigma=0.0, fname="DerivativeFilter"):
    """Derivative of the spatial axes of a (spatial..., channels) array: orders[i] along axis i."""
    nd = len(orders)
    if sigma > 0:
        a = gaussian_smooth(a, nd, sigma, pads, fname)
    for ax, dd in enumerate(orders):
        a = spline_derivative_axis(a, ax, dd, n, pads[ax])
    return a


def _interp_order(o, fname, maxd, lo=1):
    io = o.get("InterpolationOrder", AUTO)
    if _is_sym(io, "Automatic"):
        return max(3, maxd + 1)
    if not K.is_int(io) or isinstance(io, bool):
        if isinstance(io, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: InterpolationOrder must be an integer")
    n = int(io)
    if not lo <= n <= 9:
        raise ValueError(f"{fname}: InterpolationOrder must be between {lo} and 9, not {n}")
    if n <= maxd:
        raise ValueError(f"{fname}: the derivative order {maxd} must be smaller than the interpolation order {n}")
    return n


def _orders(spec, fname, nd):
    if not K.is_list(spec):
        if isinstance(spec, Symbol) or (isinstance(spec, Expr) and not K.is_list(spec)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the derivative orders must be a list {{n1, n2, ...}}")
    vals = list(spec.args)
    if len(vals) != nd:
        raise ValueError(f"{fname}: give one derivative order per dimension of the data ({nd}), not {len(vals)}")
    out = []
    for v in vals:
        if isinstance(v, (bool, str)) or K.is_list(v):
            raise TypeError(f"{fname}: derivative orders must be non-negative integers, not {_short(v)}")
        if not K.is_int(v):
            if isinstance(v, float) or WI.is_real_value(v):
                raise ValueError(f"{fname}: derivative orders must be non-negative integers, not {_short(v)}")
            raise Unevaluated()
        if int(v) < 0:
            raise ValueError(f"{fname}: derivative orders must be non-negative, not {v}")
        out.append(int(v))
    return out


def _float_input(x, fname, max_rank=None):
    """(Data, (spatial..., c) float array, alpha) for linear filters (exact arrays are converted to machine)."""
    d = data_arg(x, fname)
    if d.is_image:
        return d, d.img.color(), (d.img.a[..., -1:] if d.img.alpha else None)
    if d.exact:
        a = np.array([float(sp.N(v)) for v in d.obj.flat], dtype=float).reshape(d.obj.shape)
    else:
        a = d.a
    if max_rank is not None and a.ndim > max_rank:
        raise NotImplementedError(f"{fname}: arrays of rank {a.ndim} are not supported")
    return d, a[..., None], None


def _out(d, res, alpha=None):
    return WP._lin_out(d, res, alpha)


@fn("DerivativeFilter", area=AREA, status="added",
    notes="DerivativeFilter[data, {n1, n2, ...}] gives the derivative of order ni along dimension i (array "
          "coordinates: for images the first coordinate runs down the rows, the second along the columns) of the "
          "cardinal B-spline interpolant of data (Unser 1999: collocation coefficients, then derivatives of the "
          "B-spline basis at the samples), [data, ders, σ] first smooths with the discrete Gaussian (Bessel) kernel "
          "of standard deviation σ and radius Ceiling[4σ], and [data, {der1, der2, ...}, ...] gives a list of "
          "derivatives. InterpolationOrder -> Automatic uses degree Max[3, n + 1] (n the largest order), otherwise "
          "1..9 and larger than every order (ValueError). Padding -> \"Fixed\" (default), \"Periodic\", "
          "\"Reflected\", \"Reversed\", a number, or one per dimension; data are padded by 64 samples before the "
          "spline is fitted, so results near the edges depend on the padding. Machine-precision results (images: a "
          "real type). Check: reference (exact derivatives of sampled polynomials of degree <= the spline degree in "
          "the interior, linearity) in tests.")
def _derivativefilter(x: Expr, ders: ListOf[NonNegInteger] | ListOf[ListOf[NonNegInteger]], *rest: Real | Option) -> Expr:
    fname = "DerivativeFilter"
    pos, o = opts_split(rest, {"InterpolationOrder", "Padding", "WorkingPrecision"}, fname)
    _opt_check(pos, fname, 1)
    sigma = _num(pos[0], fname, "the Gaussian scale σ", lo=0) if pos else 0.0
    d, a, alpha = _float_input(x, fname)
    nd = a.ndim - 1
    multi = K.is_list(ders) and ders.args and all(K.is_list(t) for t in ders.args)
    specs = [_orders(t, fname, nd) for t in ders.args] if multi else [_orders(ders, fname, nd)]
    n = _interp_order(o, fname, max(max(s) for s in specs))
    pads = _pad_opt(o, fname, nd)
    if any(p is None for p in pads):
        raise NotImplementedError(f"{fname}: Padding -> None is not supported (the result has the size of data)")
    if sigma > 0:
        a = gaussian_smooth(a, nd, sigma, pads, fname)
    outs = [_out(d, spline_derivative(a, s, n, pads), None) for s in specs]
    return L(*outs) if multi else outs[0]


def hessian(a, nd, sigma, n, pads, fname="RidgeFilter"):
    """Hessian matrix field (spatial..., c, nd, nd) of a (spatial..., c) array from spline derivatives."""
    if sigma > 0:
        a = gaussian_smooth(a, nd, sigma, pads, fname)
    H = np.zeros(a.shape + (nd, nd))
    for i in range(nd):
        for j in range(i, nd):
            o = [0] * nd
            o[i] += 1
            o[j] += 1
            v = spline_derivative(a, o, n, pads)
            H[..., i, j] = v
            H[..., j, i] = v
    return H


@fn("RidgeFilter", area=AREA, status="added",
    notes="RidgeFilter[data] and RidgeFilter[data, σ] (2D or 3D arrays and images): at every position the ridge "
          "strength Max[0, -λ], where λ is the most negative eigenvalue of the Hessian matrix computed with "
          "DerivativeFilter (B-spline derivatives after discrete Gaussian smoothing with standard deviation σ); "
          "positions where the Hessian has no negative eigenvalue give 0. The default scale is σ = 1 (stated "
          "here; Wolfram's default is not documented). No scale normalisation is applied. InterpolationOrder -> "
          "Automatic (3) or 3..9, Padding as for DerivativeFilter. Multichannel images are filtered per channel; "
          "images give a real type. 1D data and rank > 3 raise NotImplementedError. Check: reference (closed-form "
          "Hessians of sampled quadratics, e.g. -x^2 gives 2) in tests.")
def _ridgefilter(x: Expr, *rest: Real | Option) -> Expr:
    fname = "RidgeFilter"
    pos, o = opts_split(rest, {"InterpolationOrder", "Padding"}, fname)
    _opt_check(pos, fname, 1)
    sigma = _num(pos[0], fname, "the scale σ", lo=0) if pos else 1.0
    d, a, alpha = _float_input(x, fname)
    nd = a.ndim - 1
    if nd not in (2, 3):
        raise NotImplementedError(f"{fname}: only 2D and 3D data are supported")
    n = _interp_order(o, fname, 2, lo=3)
    pads = _pad_opt(o, fname, nd)
    if any(p is None for p in pads):
        raise NotImplementedError(f"{fname}: Padding -> None is not supported")
    H = hessian(a, nd, sigma, n, pads)
    lam = np.linalg.eigvalsh(H)[..., 0]
    return _out(d, np.maximum(0.0, -lam), alpha)


# ───────────────────────────── GaborFilter ─────────────────────────────

@fn("GaborFilter", area=AREA, status="added",
    notes="GaborFilter[data, r, k], [data, r, k, ϕ] and [data, {r, σ}, k, ...] convolve data (ListConvolve "
          "orientation: result_i = Σ_x G_x data_(i - x)) with the kernel GaborMatrix[{r, σ}, k, ϕ] (σ = r/2 and "
          "ϕ = 0 by default, as documented); k has one component per dimension of the data (a number for 1D). "
          "Standardized -> True (default; False raises NotImplementedError, as for GaborMatrix). Padding -> "
          "\"Fixed\" (default), \"Periodic\", \"Reflected\", \"Reversed\", a number or None (smaller result). "
          "Machine precision (WorkingPrecision -> Automatic or MachinePrecision; other settings raise "
          "NotImplementedError). Multichannel images are filtered per channel and give a real image. "
          "Check: reference (direct convolution sums with GaborMatrix written in tests).")
def _gaborfilter(x: Expr, r: NonNegInteger | ListOf[Expr], k: Real | ListOf[Real], *rest: Real | Option) -> Expr:
    fname = "GaborFilter"
    pos, o = opts_split(rest, {"Padding", "Standardized", "WorkingPrecision"}, fname)
    _opt_check(pos, fname, 1)
    wp = _wp(o, fname)
    if wp not in (None, "m"):
        raise NotImplementedError(f"{fname}: WorkingPrecision other than machine precision is not supported")
    d, a, alpha = _float_input(x, fname)
    nd = a.ndim - 1
    kv = list(k.args) if K.is_list(k) else [k]
    for v in kv:
        _num(v, fname, "the wave vector component")
    if len(kv) != nd:
        raise ValueError(f"{fname}: the wave vector needs {nd} components for {nd}-dimensional data")
    kexpr = L(*kv) if nd > 1 or K.is_list(k) else k
    args = [r, kexpr] + list(pos)
    if "Standardized" in o:
        args.append(Expr("Rule", Symbol("Standardized"), o["Standardized"]))
    G = _ev(Expr("GaborMatrix", *args))
    if isinstance(G, Expr) and G.head == "GaborMatrix":
        raise Unevaluated()
    ker = np.array(WI._numeric_array(G, fname), dtype=float)
    if ker.ndim != nd:
        ker = ker.reshape(ker.shape + (1,) * (nd - ker.ndim)) if ker.ndim < nd else ker
    pad = _pad_opt(o, fname, nd, allow_list=False)[0]
    flipped = ker[tuple(slice(None, None, -1) for _ in range(ker.ndim))]
    if pad is None and any(a.shape[i] < ker.shape[i] for i in range(nd)):
        raise ValueError(f"{fname}: the data are smaller than the kernel (Padding -> None)")
    res = correlate_nd(a, flipped, pad)
    return _out(d, res, None if pad is None else alpha)


# ───────────────────────────── ImagePeriodogram ─────────────────────────────

def _window_values(w, n, fname):
    if K.is_list(w):
        v = WI._numeric_array(w, fname)
        if v is None or v.ndim != 1 or len(v) != n:
            raise ValueError(f"{fname}: a window list must have length {n}")
        return v
    if _is_sym(w, "DirichletWindow"):
        return np.ones(n)
    xs = [sp.Rational(-1, 2) + sp.Rational(i, n - 1) if n > 1 else sp.Integer(0) for i in range(n)]
    out = []
    for t in xs:
        v = _ev(K.call(w, R(t)) if hasattr(K, "call") else Expr(w, R(t)))
        if not WI.is_real_value(v) and not isinstance(v, (int, float)):
            raise ValueError(f"{fname}: the window function must give real numbers, not {_short(v)}")
        out.append(real_arg(v, fname, "window value"))
    return np.array(out, dtype=float)


@fn("ImagePeriodogram", area=AREA, status="added",
    notes="ImagePeriodogram[image] shows the power spectrum P = |Fourier|^2 / (number of pixels) of every channel "
          "(the squared DFT magnitude), [image, n] the average of the spectra of non-overlapping n x n partitions "
          "({n1, n2} for rectangles, {n1, n2, n3} for Image3D), [image, n, d] partitions with offset d, [image, n, "
          "d, wfun] multiplies each partition by the window (a window function sampled at n equally spaced points "
          "from -1/2 to 1/2, or a list; DirichletWindow, the default, does nothing; the spectrum is then divided by "
          "the window energy Σw^2 instead of the pixel count) and [image, n, d, wfun, m] zero-pads partitions to "
          "m. The result is Log[1 + P] rescaled to 0..1 per channel (a constant result gives 0), as a real image; "
          "Alignment -> Center (default) puts the zero frequency at the centre (position Floor[n/2] + 1 along "
          "each axis, rows from the top), Alignment -> {Left, Top} at the top-left pixel; other settings raise "
          "NotImplementedError. Check: reference (a constant image peaks only at the zero frequency; a cosine "
          "grating peaks at its frequency) in tests.")
def _imageperiodogram(x: Expr, *rest: PosInteger | ListOf[PosInteger] | Fn | Option) -> Expr:
    fname = "ImagePeriodogram"
    pos, o = opts_split(rest, {"Alignment"}, fname)
    _opt_check(pos, fname, 4)
    im = _image(x, fname)
    nd = 3 if im.d3 else 2
    shape = im.a.shape[:-1]

    def sizes(v, what):
        if K.is_list(v):
            if len(v.args) != nd:
                raise ValueError(f"{fname}: give {nd} {what}s")
            return [int_arg(t, fname, what, 1) for t in v.args]
        return [int_arg(v, fname, what, 1)] * nd
    n = sizes(pos[0], "partition size") if pos else list(shape)
    dd = sizes(pos[1], "offset") if len(pos) > 1 else list(n)
    m = sizes(pos[3], "padded length") if len(pos) > 3 else list(n)
    if any(ni > si for ni, si in zip(n, shape)):
        raise ValueError(f"{fname}: the partition size must not exceed the image dimensions")
    if any(mi < ni for mi, ni in zip(m, n)):
        raise ValueError(f"{fname}: the padded length must be at least the partition size")
    wins = [np.ones(ni) for ni in n]
    if len(pos) > 2 and not _is_sym(pos[2], "DirichletWindow"):
        wins = [_window_values(pos[2], ni, fname) for ni in n]
    al = o.get("Alignment", Symbol("Center"))
    if _is_sym(al, "Center"):
        center = True
    elif K.is_list(al) and len(al.args) == 2 and _is_sym(al.args[0], "Left") and _is_sym(al.args[1], "Top"):
        center = False
    else:
        raise NotImplementedError(f"{fname}: Alignment {_short(al)} is not supported (Center or {{Left, Top}})")
    W = wins[0]
    for wv in wins[1:]:
        W = np.multiply.outer(W, wv)
    energy = float((W ** 2).sum())
    if energy == 0:
        raise ValueError(f"{fname}: the window is identically zero")
    a = im.a
    starts = [range(0, s - ni + 1, di) for s, ni, di in zip(shape, n, dd)]
    acc = np.zeros(tuple(m) + (a.shape[-1],))
    cnt = 0
    import itertools
    for st in itertools.product(*starts):
        blk = a[tuple(slice(s, s + ni) for s, ni in zip(st, n))] * W[..., None]
        F = np.fft.fftn(blk, s=m, axes=tuple(range(nd)))
        acc += np.abs(F) ** 2 / energy
        cnt += 1
    P = acc / cnt
    if center:
        P = np.fft.fftshift(P, axes=tuple(range(nd)))
    v = np.log1p(P)
    lo = v.reshape(-1, v.shape[-1]).min(0)
    hi = v.reshape(-1, v.shape[-1]).max(0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    v = (v - lo) / rng
    return image_expr(WI.Img(v, "Real32", "Automatic" if v.shape[-1] == 1 else im.cs, im.d3))


# ───────────────────────────── multiscale pyramids (shared with X05) ─────────────────────────────

BURT_ADELSON = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0      # generating kernel w(a = 3/8), Burt & Adelson 1983


def pyramid_level_count(shape):
    """Number of levels of a full factor-2 pyramid: reduce (ceil(n/2) per axis) until a dimension is 1."""
    shape = list(shape)
    n = 1
    while min(shape) > 1:
        shape = [(s + 1) // 2 for s in shape]
        n += 1
    return n


def pyr_reduce(a, nd, kernel=BURT_ADELSON, pad="Reversed"):
    """REDUCE: smooth the leading nd axes with the separable kernel (padding pad), then keep every second sample
    starting with the first (size ceil(n/2))."""
    for ax in range(nd):
        a = correlate_axis_padded(a, kernel, ax, pad)
    return a[tuple(slice(None, None, 2) for _ in range(nd))]


def pyr_expand(a, shape, nd, kernel=BURT_ADELSON):
    """EXPAND to the spatial shape `shape`: insert zeros between the samples and interpolate by normalised
    convolution with the kernel, (k * z)/(k * m) for the zero-filled data z and sample indicator m (zero padding).
    In the interior this is the classical 2 k * z per axis; at the borders the weights are renormalised."""
    z = np.zeros(tuple(shape) + a.shape[nd:])
    m = np.zeros(tuple(shape) + (1,) * (a.ndim - nd))
    sl = tuple(slice(None, None, 2) for _ in range(nd))
    z[sl] = a[tuple(slice(0, (s + 1) // 2) for s in shape)]
    m[sl] = 1.0
    for ax in range(nd):
        z = correlate_axis_padded(z, kernel, ax, 0.0)
        m = correlate_axis_padded(m, kernel, ax, 0.0)
    return z / m


def gaussian_levels(a, nd, n, kernel=BURT_ADELSON, pad="Reversed"):
    out = [a]
    for _ in range(n - 1):
        out.append(pyr_reduce(out[-1], nd, kernel, pad))
    return out


def laplacian_levels(a, nd, n, kernel=BURT_ADELSON, pad="Reversed", up=None):
    """Bandpass (Laplacian) pyramid: L_i = G_i - EXPAND(G_(i+1)), the last level the lowpass residual G_n."""
    up = kernel if up is None else up
    G = gaussian_levels(a, nd, n, kernel, pad)
    out = [G[i] - pyr_expand(G[i + 1], G[i].shape[:nd], nd, up) for i in range(n - 1)]
    out.append(G[-1])
    return out


def collapse_laplacian(levels, nd, kernel=BURT_ADELSON, upto=0):
    """Inverse of laplacian_levels: G_i = L_i + EXPAND(G_(i+1)), returned for level index `upto` (0 = finest)."""
    g = levels[-1]
    for i in range(len(levels) - 2, upto - 1, -1):
        g = levels[i] + pyr_expand(g, levels[i].shape[:nd], nd, kernel)
    return g


_PYR_KEYS = ("Type", "Levels", "ImageDimensions", "LowpassKernel", "HighpassKernel", "Padding", "ScalingFactor")


def _assoc_get(a, key):
    for r in a.args:
        if isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and r.args[0] == key:
            return r.args[1]
    return None


class Pyr:
    __slots__ = ("type", "levels", "dims", "low", "high", "pad", "scale", "nd")

    def expr(self):
        def kexpr(k):
            return Symbol("None") if k is None else L(*[float(v) for v in k])
        items = [("Type", self.type), ("Levels", L(*self.levels)), ("ImageDimensions", L(*self.dims)),
                 ("LowpassKernel", kexpr(self.low)), ("HighpassKernel", kexpr(self.high)),
                 ("Padding", self.pad if isinstance(self.pad, str) else self.pad),
                 ("ScalingFactor", self.scale)]
        return Expr("ImagePyramid", Expr("Association", *[Expr("Rule", k, v) for k, v in items]))


def parse_pyramid(x, fname):
    """Pyr of an ImagePyramid[<|...|>] object; TypeError for other values, Unevaluated if symbolic."""
    if not (isinstance(x, Expr) and x.head == "ImagePyramid" and len(x.args) == 1 and isinstance(x.args[0], Expr)
            and x.args[0].head == "Association"):
        if isinstance(x, Symbol) or (isinstance(x, Expr) and not K.is_list(x) and x.head not in IMAGE_HEADS +
                                     ("Rule", "Association", "ImagePyramid")):
            raise Unevaluated()
        raise TypeError(f"{fname}: {_short(x)} is not an ImagePyramid object")
    a = x.args[0]
    p = Pyr()
    vals = {k: _assoc_get(a, k) for k in _PYR_KEYS}
    if any(v is None for v in vals.values()):
        raise ValueError(f"{fname}: the ImagePyramid object is missing the keys "
                         f"{', '.join(k for k, v in vals.items() if v is None)}")
    if vals["Type"] not in ("Lowpass", "Bandpass"):
        raise ValueError(f"{fname}: the pyramid type must be \"Lowpass\" or \"Bandpass\"")
    p.type = vals["Type"]
    if not K.is_list(vals["Levels"]) or not vals["Levels"].args:
        raise ValueError(f"{fname}: the pyramid needs a non-empty list of level images")
    p.levels = list(vals["Levels"].args)
    ims = [parse_image(v, fname) for v in p.levels]
    p.nd = 3 if ims[0].d3 else 2
    p.dims = [int(v) for v in vals["ImageDimensions"].args]

    def kern(v):
        if _is_sym(v, "None"):
            return None
        arr = WI._numeric_array(v, fname)
        if arr is None or arr.ndim != 1 or len(arr) % 2 == 0:
            raise ValueError(f"{fname}: pyramid kernels must be lists of odd length")
        return arr
    p.low, p.high = kern(vals["LowpassKernel"]), kern(vals["HighpassKernel"])
    p.pad = vals["Padding"]
    p.scale = vals["ScalingFactor"]
    return p


def _pyr_type(spec, fname, ref=None):
    """(type, lowpass kernel, upsampling kernel, padding or None) of a pyramid type specification."""
    def kernel(k):
        if isinstance(k, str) or isinstance(k, Symbol):
            raise NotImplementedError(f"{fname}: Resampling-method kernels ({_short(k)}) are not supported; give "
                                      "an odd-length numeric list")
        arr = WI._numeric_array(k, fname)
        if arr is None:
            raise TypeError(f"{fname}: a pyramid kernel must be a numeric list")
        if arr.ndim != 1:
            raise NotImplementedError(f"{fname}: only separable (1D list) kernels are supported")
        if len(arr) % 2 == 0:
            raise ValueError(f"{fname}: a pyramid kernel must have odd length")
        if abs(arr.sum()) < 1e-15:
            raise ValueError(f"{fname}: a pyramid kernel must not sum to 0")
        return arr
    if isinstance(spec, str):
        if spec in ("Lowpass", "Gaussian"):
            return "Lowpass", BURT_ADELSON, None, None
        if spec in ("Bandpass", "Laplacian"):
            return "Bandpass", BURT_ADELSON, BURT_ADELSON, None
        raise ValueError(f"{fname}: unknown pyramid type {spec!r} (\"Gaussian\"/\"Lowpass\" or "
                         "\"Laplacian\"/\"Bandpass\")")
    if K.is_list(spec) and spec.args and isinstance(spec.args[0], str):
        t = spec.args[0]
        if t in ("Lowpass", "Gaussian") and len(spec.args) == 2:
            return "Lowpass", kernel(spec.args[1]), None, None
        if t in ("Bandpass", "Laplacian") and len(spec.args) in (2, 3):
            k1 = kernel(spec.args[1])
            return "Bandpass", k1, kernel(spec.args[2]) if len(spec.args) == 3 else k1, None
        raise ValueError(f"{fname}: expected {{\"Lowpass\", ker}} or {{\"Bandpass\", ker1, ker2}}")
    if isinstance(spec, Expr) and spec.head == "ImagePyramid":
        p = parse_pyramid(spec, fname)
        low = p.low if p.low is not None else BURT_ADELSON
        return p.type, low, (p.high if p.high is not None else low) if p.type == "Bandpass" else None, p.pad
    if isinstance(spec, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_short(spec)} is not a pyramid type")


def _pyr_pad(o, fname, default):
    p = o.get("Padding", default)
    if isinstance(p, str) and p in _PAD_MODES:
        return p
    if _is_sym(p, "None"):
        raise NotImplementedError(f"{fname}: Padding -> None is not supported")
    if WI.is_real_value(p) or (isinstance(p, (int, float)) and not isinstance(p, bool)):
        return real_arg(p, fname, "Padding")
    raise ValueError(f"{fname}: unknown Padding {_short(p)}")


def _level_images(im, arrays, typ, nd):
    out = []
    for a in arrays:
        out.append(image_expr(WI.Img(quantize(a, typ) if typ in WI.MAXV else a, typ, im.cs, im.d3)))
    return out


def _size_spec(v, fname, nd):
    if K.is_list(v):
        if len(v.args) != nd:
            raise ValueError(f"{fname}: the size must have {nd} components")
        return [int_arg(t, fname, "size", 1) for t in v.args]
    s = int_arg(v, fname, "size", 1)
    return [s] * nd


def _dims_of(shape, nd):
    """Image dimensions {w, h} ({w, d, h} for 3D) of a spatial array shape."""
    return [shape[1], shape[0]] if nd == 2 else [shape[2], shape[1], shape[0]]


@fn("ImagePyramid", area=AREA, status="added",
    notes="ImagePyramid[image] gives an ImagePyramid[<|...|>] object holding a Gaussian (\"Lowpass\") pyramid with "
          "all possible levels (reduce until a dimension is 1); ImagePyramid[image, pyrtype], [image, pyrtype, n] "
          "(at most n levels), [image, pyrtype, {size}] (levels whose dimensions are all at least size) and "
          "[image, pyrtype, n, 2]. pyrtype: \"Gaussian\"/\"Lowpass\" (blur and keep every second sample), "
          "\"Laplacian\"/\"Bandpass\" (G_i - EXPAND(G_(i+1)), last level the lowpass residual), {\"Lowpass\", "
          "ker}, {\"Bandpass\", ker1, ker2} (ker2 for upsampling) or a reference pyramid (its type, kernels, "
          "padding and level count). The default kernel is the Burt-Adelson generating kernel {1, 4, 6, 4, 1}/16 "
          "(a = 3/8, Burt & Adelson 1983; Wolfram's own kernel is not documented); EXPAND is normalised "
          "convolution with the upsampling kernel of the zero-filled level (interior: 2 ker per axis). Padding -> "
          "\"Reversed\" (default, as documented), \"Fixed\", \"Periodic\", \"Reflected\" or a number (for REDUCE). "
          "Lowpass levels keep the image type; bandpass levels are real. Scaling factors other than 2 and "
          "Resampling-name kernels raise NotImplementedError. pyr[\"prop\"] gives \"Levels\", \"LevelCount\", "
          "\"ImageDimensions\", \"Type\", \"LowpassKernel\", \"HighpassKernel\" (the upsampling kernel of bandpass "
          "pyramids, else None), \"Padding\", \"ScalingFactor\", \"Properties\"; pyr[n], pyr[-n], pyr[{n1, ...}], "
          "pyr[m ;; n ;; s] and pyr[All] give levels. Check: self-check (a bandpass pyramid is collapsed and "
          "compared with the image, ArithmeticError on mismatch); reference (level sizes, REDUCE sums) in tests.")
def _imagepyramid(x: Expr, *rest: String | ListOf[Expr] | PosInteger | Expr | Option) -> Expr:
    fname = "ImagePyramid"
    pos, o = opts_split(rest, {"Padding"}, fname)
    if isinstance(x, Expr) and x.head == "Association" and not rest:
        p = parse_pyramid(Expr("ImagePyramid", x), fname)
        raise Unevaluated()                                 # the inert object
    _opt_check(pos, fname, 3)
    im = _image(x, fname)
    nd = 3 if im.d3 else 2
    typ, low, up, refpad = _pyr_type(pos[0], fname) if pos else ("Lowpass", BURT_ADELSON, None, None)
    pad = _pyr_pad(o, fname, refpad if refpad is not None else "Reversed")
    shape = im.a.shape[:nd]
    nmax = pyramid_level_count(shape)
    n = nmax
    if pos and isinstance(pos[0], Expr) and pos[0].head == "ImagePyramid":
        n = min(nmax, len(parse_pyramid(pos[0], fname).levels))
    if len(pos) > 1:
        spec = pos[1]
        if K.is_list(spec) and len(spec.args) == 1:
            size = _size_spec(spec.args[0], fname, nd)
            sh = list(shape)
            n = 1
            while min(sh) > 1:
                nxt = [(s + 1) // 2 for s in sh]
                if any(dv < sz for dv, sz in zip(_dims_of(nxt, nd), size)):
                    break
                sh = nxt
                n += 1
        else:
            n = min(nmax, int_arg(spec, fname, "the number of levels", 1))
    if len(pos) > 2:
        s = pos[2]
        if not (K.is_int(s) and int(s) == 2):
            if isinstance(s, Symbol):
                raise Unevaluated()
            sv = _num(s, fname, "the scaling factor", lo=1, strict=True)
            if sv != 2:
                raise NotImplementedError(f"{fname}: scaling factors other than 2 are not supported")
    a = im.a
    if typ == "Lowpass":
        arrays = gaussian_levels(a, nd, n, low, pad)
        levels = _level_images(im, arrays, im.type, nd)
    else:
        arrays = laplacian_levels(a, nd, n, low, pad, up)
        back = collapse_laplacian(arrays, nd, up)
        if not np.allclose(back, a, atol=1e-9 * max(1.0, float(np.abs(a).max()))):
            raise ArithmeticError(f"{fname}: result failed verification (the bandpass pyramid does not collapse "
                                  "to the image)")
        levels = _level_images(im, arrays, real_type(im), nd)
    p = Pyr()
    p.type, p.levels, p.dims, p.low, p.high, p.pad, p.scale, p.nd = typ, levels, im.dims(), low, up, pad, 2, nd
    return p.expr()


def _level_indices(spec, n, fname):
    if _is_sym(spec, "All"):
        return list(range(n)), True
    if K.is_int(spec) and not isinstance(spec, bool):
        i = int(spec)
        if i == 0 or abs(i) > n:
            raise ValueError(f"{fname}: level {i} does not exist (the pyramid has {n} levels)")
        return [i - 1 if i > 0 else n + i], False
    if K.is_list(spec):
        out = []
        for v in spec.args:
            out.extend(_level_indices(v, n, fname)[0])
        return out, True
    if isinstance(spec, Expr) and spec.head == "Span":
        return WI.seq_indices(spec, n, fname), True
    raise ValueError(f"{fname}: {_short(spec)} is not a level specification")


def _pyr_apply(head, args, ev_):
    fname = "ImagePyramid"
    try:
        p = parse_pyramid(head, fname)
    except Unevaluated:
        return K.UNHANDLED
    if len(args) != 1:
        raise ValueError(f"{fname}: an ImagePyramid object takes one argument")
    q = args[0]
    n = len(p.levels)
    props = {"ImageDimensions": L(*p.dims), "LevelCount": n, "Levels": L(*p.levels), "Type": p.type,
             "LowpassKernel": Symbol("None") if p.low is None else L(*[float(v) for v in p.low]),
             "HighpassKernel": Symbol("None") if p.high is None else L(*[float(v) for v in p.high]),
             "Padding": p.pad, "ScalingFactor": p.scale}
    if isinstance(q, str):
        if q == "Properties":
            return L(*sorted(props))
        if q in props:
            return props[q]
        raise ValueError(f"{fname}: unknown property {q!r}")
    if isinstance(q, Symbol) and not _is_sym(q, "All"):
        return K.UNHANDLED
    idx, many = _level_indices(q, n, fname)
    return L(*[p.levels[i] for i in idx]) if many else p.levels[idx[0]]


try:
    from .. import kernel_lists as _KL
    _KL.COMPOUND_HEAD_HOOKS["ImagePyramid"] = _pyr_apply
except (ImportError, AttributeError):  # pragma: no cover
    pass


@fn("InverseImagePyramid", area=AREA, status="added",
    notes="InverseImagePyramid[pyr] reconstructs the image: for a \"Lowpass\" pyramid its first level (as "
          "documented), for a \"Bandpass\" pyramid the collapse G_i = L_i + EXPAND(G_(i+1)) with the pyramid's "
          "upsampling kernel (exact inverse of ImagePyramid). InverseImagePyramid[pyr, pyrtype] assumes the given "
          "type (\"Gaussian\"/\"Lowpass\", \"Laplacian\"/\"Bandpass\", {\"Bandpass\", ker1, ker2} or a reference "
          "pyramid), [pyr, pyrtype, n] reconstructs level n and [pyr, pyrtype, {size}] the smallest level whose "
          "dimensions are all at least size. Bandpass reconstructions are real images, lowpass ones keep the "
          "type. Check: reference (round trip InverseImagePyramid[ImagePyramid[img, \"Laplacian\"]] == img to "
          "machine precision) in tests.")
def _inverseimagepyramid(pyr: Expr, *rest: String | ListOf[Expr] | PosInteger | Expr) -> Expr:
    fname = "InverseImagePyramid"
    p = parse_pyramid(pyr, fname)
    _opt_check(rest, fname, 2)
    typ, up = p.type, (p.high if p.high is not None else (p.low if p.low is not None else BURT_ADELSON))
    if rest:
        typ, low, up2, _pad = _pyr_type(rest[0], fname)
        if typ == "Bandpass":
            up = up2
    ims = [parse_image(v, fname) for v in p.levels]
    n = len(ims)
    upto = 0
    if len(rest) > 1:
        spec = rest[1]
        if K.is_list(spec) and len(spec.args) == 1:
            size = _size_spec(spec.args[0], fname, p.nd)
            ok = [i for i, im in enumerate(ims) if all(dv >= sz for dv, sz in zip(im.dims(), size))]
            if not ok:
                raise ValueError(f"{fname}: no level is at least as large as {size}")
            upto = ok[-1]
        else:
            k = int_arg(spec, fname, "the level", 1)
            if k > n:
                raise ValueError(f"{fname}: level {k} does not exist (the pyramid has {n} levels)")
            upto = k - 1
    if typ == "Lowpass":
        return p.levels[upto]
    g = collapse_laplacian([im.a for im in ims], p.nd, up, upto)
    ref = ims[upto]
    return image_expr(WI.Img(g, real_type(ref), ref.cs, ref.d3))


@fn("ImagePyramidApply", area=AREA, status="added",
    notes="ImagePyramidApply[f, pyr] applies f to every level image of the ImagePyramid object pyr and gives a "
          "pyramid with the same type, kernels and padding; ImagePyramidApply[f, {pyr1, pyr2, ...}] calls f with "
          "the corresponding levels of commensurate pyramids (same type and level dimensions, else ValueError), "
          "keeping the specification of pyr1. f must return an image of the dimensions of the level (ValueError "
          "otherwise). Check: reference (identity and level-wise ImageMultiply compared with direct results) in "
          "tests.")
def _imagepyramidapply(f: Fn, pyr: Expr | ListOf[Expr]) -> Expr:
    fname = "ImagePyramidApply"
    pyrs = list(pyr.args) if K.is_list(pyr) else [pyr]
    if not pyrs:
        raise ValueError(f"{fname}: no pyramids given")
    ps = [parse_pyramid(v, fname) for v in pyrs]
    n = len(ps[0].levels)
    for q in ps[1:]:
        if len(q.levels) != n or q.type != ps[0].type:
            raise ValueError(f"{fname}: the pyramids are not commensurate (type and level count must agree)")
    dims = [parse_image(v, fname).dims() for v in ps[0].levels]
    for q in ps[1:]:
        if [parse_image(v, fname).dims() for v in q.levels] != dims:
            raise ValueError(f"{fname}: the pyramids are not commensurate (level dimensions differ)")
    out = []
    for i in range(n):
        r = _ev(K.call(f, *[q.levels[i] for q in ps]))
        if not (isinstance(r, Expr) and r.head in IMAGE_HEADS):
            raise ValueError(f"{fname}: f must return an image for every level, not {_short(r)}")
        if parse_image(r, fname).dims() != dims[i]:
            raise ValueError(f"{fname}: f changed the dimensions of level {i + 1}")
        out.append(r)
    p = ps[0]
    p.levels = out
    return p.expr()


# ───────────────────────────── shared helpers for image restoration ─────────────────────────────

def _image_parts(x, fname, allow3d=True):
    """(Img, colour array (spatial..., c), alpha or None, nd)."""
    im = _image(x, fname, allow3d=allow3d)
    nd = 3 if im.d3 else 2
    return im, im.color(), (im.a[..., -1:] if im.alpha else None), nd


def _img_result(im, col, alpha, typ=None, cs=None):
    a = col if alpha is None else np.concatenate([col, alpha], axis=-1)
    typ = im.type if typ is None else typ
    if typ == "Bit" and not np.all((a == 0) | (a == 1)):
        typ = "Real32"
    return image_expr(WI.Img(quantize(a, typ), typ, im.cs if cs is None else cs, im.d3))


def _smooth_type(im):
    return "Real32" if im.type == "Bit" else im.type


def _neumann_shift(a, ax, step):
    """a shifted by one sample along ax (step +1: value of the next sample), repeating the edge sample."""
    idx = np.arange(a.shape[ax]) + step
    idx = np.clip(idx, 0, a.shape[ax] - 1)
    return np.take(a, idx, axis=ax)


def grad_fwd(u, nd):
    """Forward differences along the leading nd axes (0 at the far edge: Neumann boundary)."""
    out = []
    for ax in range(nd):
        out.append(_neumann_shift(u, ax, 1) - u)
    return out


def div_bwd(p, nd):
    """Discrete divergence, the negative adjoint of grad_fwd."""
    tot = None
    for ax in range(nd):
        q = p[ax]
        n = q.shape[ax]
        d = np.copy(q)
        if n > 1:
            sl_lo = [slice(None)] * q.ndim
            sl_hi = [slice(None)] * q.ndim
            sl_lo[ax] = slice(1, n)
            sl_hi[ax] = slice(0, n - 1)
            d[tuple(sl_lo)] = q[tuple(sl_lo)] - q[tuple(sl_hi)]
            last = [slice(None)] * q.ndim
            last[ax] = n - 1
            prev = [slice(None)] * q.ndim
            prev[ax] = n - 2
            d[tuple(last)] = -q[tuple(prev)]
        else:
            d = np.zeros_like(q)
        tot = d if tot is None else tot + d
    return tot


def tv_minimize(f, nd, lam, model="Gaussian", iters=30, known=None):
    """Chambolle–Pock (2011) primal-dual iterations for min_u F(u) + lam Σ |∇u| (isotropic TV, forward
    differences, Neumann boundary) with F = ½||u - f||² ("Gaussian"), ||u - f||_1 ("Laplacian") or
    Σ (u - f log u) ("Poisson"); `known` restricts the data term to a mask (inpainting). Step sizes
    τ = σ = 1/Sqrt[4 nd], θ = 1; u starts at f."""
    L2 = 4.0 * nd
    tau = sig = 1.0 / math.sqrt(L2)
    u = np.array(f, dtype=float)
    ubar = u.copy()
    p = [np.zeros_like(u) for _ in range(nd)]
    for _ in range(iters):
        g = grad_fwd(ubar, nd)
        p = [pi + sig * gi for pi, gi in zip(p, g)]
        norm = np.sqrt(sum(pi ** 2 for pi in p))
        if nd > 0:
            # channels are regularised separately (norm over the spatial derivative components only)
            fac = np.maximum(1.0, norm / lam) if lam > 0 else np.inf
            p = [pi / fac for pi in p] if lam > 0 else [np.zeros_like(pi) for pi in p]
        v = u + tau * div_bwd(p, nd)
        if model == "Gaussian":
            new = (v + tau * f) / (1 + tau)
        elif model == "Laplacian":
            dlt = v - f
            new = f + np.sign(dlt) * np.maximum(np.abs(dlt) - tau, 0.0)
        else:
            t = v - tau
            new = (t + np.sqrt(t * t + 4 * tau * f)) / 2
        if known is not None:
            new = np.where(known, new, v)
        ubar = 2 * new - u
        u = new
    return u


# ───────────────────────────── TotalVariationFilter ─────────────────────────────

def _noise_model(m, fname, what="Method"):
    if _is_sym(m, "Automatic"):
        return "Gaussian"
    if m in ("Gaussian", "Laplacian", "Poisson"):
        return m
    raise ValueError(f"{fname}: {what} must be \"Gaussian\", \"Laplacian\" or \"Poisson\", not {_short(m)}")


@fn("TotalVariationFilter", area=AREA, status="added",
    notes="TotalVariationFilter[data] and TotalVariationFilter[data, λ] minimise F(u) + λ TV(u) (isotropic total "
          "variation with forward differences and a Neumann boundary) with MaxIterations (default 30, as "
          "documented) iterations of the Chambolle–Pock primal-dual algorithm (τ = σ = 1/Sqrt[4 d] for d "
          "dimensions, starting from the data); Method -> \"Gaussian\" (default; F = ½ Σ (u - data)^2, the "
          "Rudin–Osher–Fatemi model), \"Laplacian\" (F = Σ |u - data|) or \"Poisson\" (F = Σ (u - data Log[u]); "
          "data must be non-negative). The default λ = 0.1 is stated here (Wolfram's default is not documented). "
          "The result is the stated iterate, an approximation of the minimiser that improves with MaxIterations. "
          "Works on arrays of any rank (machine reals) and images (per channel, same type; Bit gives Real32). "
          "Check: reference (constant data unchanged, total variation decreases, λ = 0 returns the data) in tests.")
def _totalvariationfilter(x: Expr, *rest: Real | Option) -> Expr:
    fname = "TotalVariationFilter"
    pos, o = opts_split(rest, {"MaxIterations", "Method"}, fname)
    _opt_check(pos, fname, 1)
    lam = _num(pos[0], fname, "the regularization parameter", lo=0) if pos else 0.1
    model = _noise_model(o.get("Method", "Gaussian"), fname)
    its = int_arg(o.get("MaxIterations", 30), fname, "MaxIterations", 1)
    d, a, alpha = _float_input(x, fname)
    nd = a.ndim - 1
    if model == "Poisson" and (a < 0).any():
        raise ValueError(f"{fname}: Method -> \"Poisson\" needs non-negative data")
    if lam == 0:
        res = a
    else:
        res = tv_minimize(a, nd, lam, model, its)
    if d.is_image:
        return _img_result(d.img, res, alpha, typ=_smooth_type(d.img))
    return array_out(res[..., 0])


# ───────────────────────────── Perona–Malik and curvature flow ─────────────────────────────

def _flow_steps(t, dt_max):
    n = max(1, int(math.ceil(t / dt_max - 1e-12)))
    return n, t / n


def _central_grad(u):
    """Central differences (one-sided at the edges via edge repetition) along rows and columns of (h, w, c)."""
    gy = (_neumann_shift(u, 0, 1) - _neumann_shift(u, 0, -1)) / 2
    gx = (_neumann_shift(u, 1, 1) - _neumann_shift(u, 1, -1)) / 2
    return gy, gx


def perona_malik(a, t, k, sigma, fname="PeronaMalikFilter"):
    """Explicit Perona–Malik (1990) diffusion of a (h, w, c) array: 4-neighbour fluxes c(|∇|) Δ, Δt <= 0.2,
    zero flux across the border; c = exp(-(g/k)^2) with g the directional difference (σ = 0) or the norm of
    the gradient of the Gaussian-smoothed image (σ > 0, Catté et al. 1992; flux weights averaged)."""
    n, dt = _flow_steps(t, 0.2)
    u = np.array(a, dtype=float)
    for _ in range(n):
        if sigma > 0:
            s = gaussian_smooth(u, 2, sigma, ["Fixed", "Fixed"], fname)
            gy, gx = _central_grad(s)
            cmap = np.exp(-(gy ** 2 + gx ** 2) / (k * k)) if k > 0 else ((gy ** 2 + gx ** 2) == 0).astype(float)
        upd = np.zeros_like(u)
        for ax in (0, 1):
            for step in (1, -1):
                dlt = _neumann_shift(u, ax, step) - u
                if sigma > 0:
                    c = (cmap + _neumann_shift(cmap, ax, step)) / 2
                else:
                    c = np.exp(-(dlt / k) ** 2) if k > 0 else (dlt == 0).astype(float)
                upd += c * dlt
        u = u + dt * upd
    return u


def _grad_norm_median(a):
    m = a.mean(axis=-1, keepdims=True)
    gy, gx = _central_grad(m)
    return float(np.median(np.sqrt(gy ** 2 + gx ** 2)))


@fn("PeronaMalikFilter", area=AREA, status="added",
    notes="PeronaMalikFilter[image], [image, t], [image, t, k] and [image, t, k, σ] (2D images, per channel) "
          "integrate the documented diffusion ∂_t f = ∇.(c_k(|∇_σ f|) ∇f) with c_k(g) = e^(-g^2/k^2) up to time t "
          "(default 1) by the explicit 4-neighbour scheme of Perona & Malik (1990) with ⌈t/0.2⌉ equal steps, zero "
          "flux across the border, pixel spacing 1 and values on the 0..1 scale. σ = 0 (default) evaluates the "
          "conductance on each directional difference; σ > 0 on the gradient norm of the image smoothed with the "
          "discrete Gaussian of standard deviation σ (Catté et al. 1992). k -> Automatic (default) is the median "
          "(50% quantile) of the central-difference gradient norm of the channel average, as documented. The "
          "type is kept (Bit gives Real32). Image3D raises NotImplementedError. Check: reference (constant images "
          "unchanged, the mean is conserved, a step edge with k much smaller than the step is preserved) in tests.")
def _peronamalikfilter(x: Expr, *rest: Real | Expr) -> Expr:
    fname = "PeronaMalikFilter"
    _opt_check(rest, fname, 3)
    im, a, alpha, nd = _image_parts(x, fname, allow3d=False)
    t = _num(rest[0], fname, "the diffusion time", lo=0) if rest else 1.0
    k = None
    if len(rest) > 1 and not _is_sym(rest[1], "Automatic"):
        k = _num(rest[1], fname, "the conductance parameter k", lo=0, strict=True)
    sigma = _num(rest[2], fname, "the regularization σ", lo=0) if len(rest) > 2 else 0.0
    if k is None:
        k = _grad_norm_median(a)
    res = a if t == 0 else perona_malik(a, t, k, sigma, fname)
    return _img_result(im, res, alpha, typ=_smooth_type(im))


def curvature_flow(a, t, k=math.inf):
    """Explicit mean-curvature (Euclidean shortening) flow ∂_t f = |∇f| ∇.(c_k(|∇f|) ∇f/|∇f|) of a (h, w, c)
    array (central differences, Neumann boundary, Δt <= 0.1); k = ∞ gives f_t = (f_xx f_y^2 - 2 f_x f_y f_xy +
    f_yy f_x^2)/(f_x^2 + f_y^2)."""
    n, dt = _flow_steps(t, 0.1)
    u = np.array(a, dtype=float)
    scale = float(np.abs(u).max()) or 1.0
    eps2 = (1e-8 * scale) ** 2
    for _ in range(n):
        uy, ux = _central_grad(u)
        g2 = ux ** 2 + uy ** 2
        if math.isinf(k):
            uyy = _neumann_shift(u, 0, 1) - 2 * u + _neumann_shift(u, 0, -1)
            uxx = _neumann_shift(u, 1, 1) - 2 * u + _neumann_shift(u, 1, -1)
            uxy = (_neumann_shift(_neumann_shift(u, 0, 1), 1, 1) - _neumann_shift(_neumann_shift(u, 0, 1), 1, -1)
                   - _neumann_shift(_neumann_shift(u, 0, -1), 1, 1)
                   + _neumann_shift(_neumann_shift(u, 0, -1), 1, -1)) / 4
            num = uxx * uy ** 2 - 2 * ux * uy * uxy + uyy * ux ** 2
            upd = np.where(g2 > eps2, num / np.maximum(g2, eps2), 0.0)
        else:
            gn = np.sqrt(g2)
            c = np.exp(-g2 / (k * k))
            safe = np.maximum(gn, math.sqrt(eps2))
            ny, nx = c * uy / safe, c * ux / safe
            dv = (_neumann_shift(ny, 0, 1) - _neumann_shift(ny, 0, -1)) / 2 + \
                (_neumann_shift(nx, 1, 1) - _neumann_shift(nx, 1, -1)) / 2
            upd = np.where(g2 > eps2, gn * dv, 0.0)
        u = u + dt * upd
    return u


@fn("CurvatureFlowFilter", area=AREA, status="added",
    notes="CurvatureFlowFilter[image], [image, t] and [image, t, k] (2D images, per channel) integrate the "
          "documented mean curvature flow ∂_t f = κ |∇f| (κ = ∇.(∇f/|∇f|)), or with finite k the modified flow "
          "∂_t f = |∇f| ∇.(c_k(|∇f|) ∇f/|∇f|) with c_k(g) = e^(-g^2/k^2), up to time t (default 1; k = ∞ default, "
          "as documented: CurvatureFlowFilter[image] == CurvatureFlowFilter[image, 1, ∞]) by an explicit "
          "central-difference scheme with ⌈t/0.1⌉ equal steps, pixel spacing 1, Neumann boundary; points with "
          "zero gradient do not move. The type is kept (Bit gives Real32). Image3D raises NotImplementedError. "
          "Check: reference (constant images and straight edges unchanged, a scaled image gives the scaled "
          "result for k = ∞) in tests.")
def _curvatureflowfilter(x: Expr, *rest: Real | Expr) -> Expr:
    fname = "CurvatureFlowFilter"
    _opt_check(rest, fname, 2)
    im, a, alpha, nd = _image_parts(x, fname, allow3d=False)
    t = _num(rest[0], fname, "the flow time", lo=0) if rest else 1.0
    k = _num(rest[1], fname, "the conductance parameter k", lo=0, strict=True, allow_inf=True) \
        if len(rest) > 1 else math.inf
    res = a if t == 0 else curvature_flow(a, t, k)
    return _img_result(im, res, alpha, typ=_smooth_type(im))


# ───────────────────────────── NonlocalMeansFilter ─────────────────────────────

NLM_K = 0.4      # Buades, Coll & Morel (IPOL 2011): h = 0.4 σ


def noise_variance(a):
    """Immerkær (1996) fast noise variance estimate of a (h, w, c) array, averaged over the channels."""
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    Nk = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float)
    h, w = a.shape[:2]
    conv = sum(Nk[i, j] * a[i:h - 2 + i, j:w - 2 + j] for i in range(3) for j in range(3))
    sig = math.sqrt(math.pi / 2) / (6 * (w - 2) * (h - 2)) * np.abs(conv).sum(axis=(0, 1))
    return float(np.mean(sig ** 2))


def _box_sum2(a, r):
    """Sum over (2r+1)^2 windows of a (H, W) array, valid part (H-2r, W-2r)."""
    c = np.cumsum(np.cumsum(np.pad(a, ((1, 0), (1, 0))), axis=0), axis=1)
    n = 2 * r + 1
    return c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]


def nonlocal_means(a, r, ns, w, pad="Fixed"):
    h, wd, c = a.shape
    m = w + r
    P = a
    for ax in (0, 1):
        P = pad_axis(P, ax, m, m, pad)
    num = np.zeros_like(a)
    den = np.zeros((h, wd))
    length = (2 * r + 1) ** 2 * c
    center = P[w:w + h + 2 * r, w:w + wd + 2 * r]
    for dy in range(-w, w + 1):
        for dx in range(-w, w + 1):
            sh = P[w + dy:w + dy + h + 2 * r, w + dx:w + dx + wd + 2 * r]
            d2 = ((sh - center) ** 2).sum(-1)
            sed = _box_sum2(d2, r) / length
            if ns > 0:
                wt = np.exp(-np.maximum(0.0, sed - 2 * ns) / (NLM_K ** 2 * ns))
            else:
                wt = (sed <= 1e-300).astype(float)
            num += wt[..., None] * P[m + dy:m + dy + h, m + dx:m + dx + wd]
            den += wt
    return num / den[..., None]


@fn("NonlocalMeansFilter", area=AREA, status="added",
    notes="NonlocalMeansFilter[image, r], [image, r, ns] and [image, r, ns, w] replace every pixel by the average "
          "of the pixels in the range-w window weighted by the similarity of their range-r neighbourhoods, "
          "Exp[-Max[0, SquaredEuclideanDistance[n1, n2]/len - 2 ns]/(k^2 ns)] with len the number of channel "
          "values in a neighbourhood, as documented, and k = 0.4 (the IPOL value h = 0.4 σ of Buades, Coll & "
          "Morel 2011; Wolfram's k is not given). ns -> Automatic (default) is Immerkær's (1996) noise variance "
          "estimate; ns = 0 averages only pixels with identical neighbourhoods. w defaults to 5 r, as documented. "
          "Padding -> \"Fixed\" (default), \"Periodic\", \"Reflected\", \"Reversed\" or a number. 2D images; the "
          "type is kept. Image3D raises NotImplementedError. Check: reference (constant images unchanged, a "
          "direct weighted-average computation for a small image) in tests.")
def _nonlocalmeansfilter(x: Expr, r: NonNegInteger, *rest: Real | Expr | Option) -> Expr:
    fname = "NonlocalMeansFilter"
    pos, o = opts_split(rest, {"Padding"}, fname)
    _opt_check(pos, fname, 2)
    im, a, alpha, nd = _image_parts(x, fname, allow3d=False)
    rr = int_arg(r, fname, "the neighbourhood range r", 0)
    ns = None
    if pos and not _is_sym(pos[0], "Automatic"):
        ns = _num(pos[0], fname, "the noise power ns", lo=0)
    w = int_arg(pos[1], fname, "the window range w", 0) if len(pos) > 1 else 5 * rr
    pad = _pad_opt(o, fname, 2, allow_list=False)[0]
    if pad is None:
        raise NotImplementedError(f"{fname}: Padding -> None is not supported")
    if ns is None:
        ns = noise_variance(a)
    res = nonlocal_means(a, rr, ns, w, pad)
    return _img_result(im, res, alpha)


# ───────────────────────────── FourierDCTFilter ─────────────────────────────

def dct_matrix(n):
    """Orthonormal DCT-II matrix."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    M = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    M[0] /= math.sqrt(2.0)
    return M


def dct_denoise(a, nd, t, block=16):
    """Hard-threshold every overlapping block's orthonormal DCT (|coefficient| < t set to 0; Yu & Sapiro 2011),
    with a DCT across the channels too when there are several, and average the overlapping reconstructions."""
    from numpy.lib.stride_tricks import sliding_window_view
    shape = a.shape[:nd]
    B = [min(block, s) for s in shape]
    Ds = [dct_matrix(b) for b in B]
    c = a.shape[-1]
    Dc = dct_matrix(c) if c > 1 else None
    out = np.zeros_like(a)
    cnt = np.zeros(shape)
    win = sliding_window_view(a, B, axis=tuple(range(nd)))      # (n1, ..., nd', c, B1, ..., Bnd)
    nblk = win.shape[:nd]
    chunk = max(1, 4096 // max(1, int(np.prod(nblk[1:]))))
    for s0 in range(0, nblk[0], chunk):
        W = np.array(win[s0:s0 + chunk], dtype=float)
        C = W
        for i, D in enumerate(Ds):
            C = np.moveaxis(np.tensordot(C, D, axes=([nd + 1 + i], [1])), -1, nd + 1 + i)
        if Dc is not None:
            C = np.moveaxis(np.tensordot(C, Dc, axes=([nd], [1])), -1, nd)
        C = np.where(np.abs(C) < t, 0.0, C)
        if Dc is not None:
            C = np.moveaxis(np.tensordot(C, Dc, axes=([nd], [0])), -1, nd)
        for i, D in enumerate(Ds):
            C = np.moveaxis(np.tensordot(C, D, axes=([nd + 1 + i], [0])), -1, nd + 1 + i)
        # overlap-add: C has shape (chunk, n2, ..., c, B1, ..., Bnd)
        import itertools
        for off in itertools.product(*[range(b) for b in B]):
            piece = C[(Ellipsis,) + off]                           # (chunk, n2, ..., c)
            sl = tuple([slice(s0 + off[0], s0 + off[0] + piece.shape[0])] +
                       [slice(off[i], off[i] + piece.shape[i]) for i in range(1, nd)])
            out[sl] += piece
            cnt[sl] += 1
    return out / cnt[..., None]


@fn("FourierDCTFilter", area=AREA, status="added",
    notes="FourierDCTFilter[image, t] applies hard thresholding with threshold t to the orthonormal DCT-II of every "
          "overlapping 16 x 16 partition (16^3 for Image3D; a smaller side uses its full length): coefficients "
          "with |c| < t are set to 0 (the DC term included), the partitions are transformed back and every pixel "
          "is the average over the partitions that contain it, as documented (Yu & Sapiro 2011). Multichannel "
          "images also apply an orthonormal DCT across the colour channels (\"3D DCT\"); alpha is kept. Values "
          "are on the 0..1 scale; the type is kept. t = 0 returns the image. Check: reference (t = 0 is the "
          "identity, constant images unchanged, a single-block DCT thresholding computed directly) in tests.")
def _fourierdctfilter(x: Expr, t: Real) -> Expr:
    fname = "FourierDCTFilter"
    im, a, alpha, nd = _image_parts(x, fname)
    tv = _num(t, fname, "the threshold", lo=0)
    res = dct_denoise(a, nd, tv)
    return _img_result(im, res, alpha, typ=_smooth_type(im))


# ───────────────────────────── ImageDeconvolve ─────────────────────────────

def _deconv_kernel(ker, fname, nd, nchan):
    if isinstance(ker, Expr) and ker.head in IMAGE_HEADS:
        k = parse_image(ker, fname)
        arr = k.a
        if arr.shape[-1] not in (1, nchan):
            raise ValueError(f"{fname}: the kernel must have one channel or as many as the image")
        return arr
    arr = WI._numeric_array(ker, fname)
    if arr is None:
        if isinstance(ker, Symbol) or isinstance(ker, Expr):
            raise Unevaluated()
        raise TypeError(f"{fname}: the kernel must be a numeric array or an image")
    if arr.ndim != nd:
        raise ValueError(f"{fname}: the kernel must be a {nd}D array")
    return arr[..., None]


def _psf_fft(k, shape, nd):
    """FFT of kernel k (spatial..., 1) placed with its centre (index Floor[n/2]) at the origin of `shape`."""
    z = np.zeros(tuple(shape))
    sl = tuple(slice(0, s) for s in k.shape[:nd])
    z[sl] = k[..., 0]
    z = np.roll(z, [-(s // 2) for s in k.shape[:nd]], axis=tuple(range(nd)))
    return np.fft.fftn(z)


def _filter_factors(method, H2, p, Lap2):
    if method == "Tikhonov":
        return H2 / (H2 + p * p)
    if method == "Wiener":
        return H2 / (H2 + p)
    if method == "DampedLS":
        return H2 / (H2 + p * p * Lap2)
    return (np.sqrt(H2) >= p).astype(float)          # TSVD


def _gcv_parameter(method, H2, G, Lap2):
    """Parameter minimising the generalised cross-validation function (Hansen, Nagy & O'Leary 2006)."""
    hmax = float(np.sqrt(H2.max())) or 1.0
    if method == "TSVD":
        cands = np.unique(np.sqrt(H2).ravel())
        cands = cands[cands > 0]
        cands = cands[np.linspace(0, len(cands) - 1, min(len(cands), 400)).astype(int)]
    elif method == "Wiener":
        cands = np.logspace(-12, 1, 261) * hmax * hmax
    else:
        cands = np.logspace(-6, 1, 281) * hmax
    G2 = np.abs(G) ** 2
    best, bp = math.inf, None
    for p in cands:
        phi = _filter_factors(method, H2, p, Lap2)
        num = float(((1 - phi) ** 2 * G2).sum())
        den = float((1 - phi).sum()) ** 2
        if den <= 0:
            continue
        v = num / den
        if v < best:
            best, bp = v, p
    return bp if bp is not None else 0.0


def _conv_fft(u, H, nd):
    return np.real(np.fft.ifftn(np.fft.fftn(u, axes=range(nd)) * H, axes=range(nd)))


_DECONV_METHODS = ("DampedLS", "Tikhonov", "TSVD", "Wiener", "RichardsonLucy", "SteepestDescent", "Hybrid",
                   "TotalVariation")


@fn("ImageDeconvolve", area=AREA, status="added",
    notes="ImageDeconvolve[image, ker] deconvolves every channel with the point spread function ker (a numeric "
          "array or a single-channel image, no larger than the image; centre at index Floor[n/2] + 1, convolution "
          "orientation of ImageConvolve). The image is padded by the kernel size (Padding -> \"Reversed\" "
          "(default, as documented), \"Fixed\", \"Reflected\", a number, or \"Periodic\" = no padding, circular "
          "convolution), treated as periodic, and cropped back. Spectral methods with H the DFT of the kernel, G "
          "of the image: \"DampedLS\" (default, as documented; generalised Tikhonov with the 5-point Laplacian Λ: "
          "conj(H) G/(|H|^2 + p^2 |Λ|^2)), \"Tikhonov\" (conj(H) G/(|H|^2 + p^2)), \"Wiener\" (constant "
          "noise-to-signal ratio p: conj(H) G/(|H|^2 + p)) and \"TSVD\" (G/H where |H| >= p, else 0); Method -> "
          "{\"method\", p} sets p (or one per channel), otherwise p minimises the generalised cross-validation "
          "function (Hansen, Nagy & O'Leary 2006). Iterative methods run MaxIterations (default 10) iterations: "
          "\"RichardsonLucy\" (classical, non-negative image and kernel) and {\"SteepestDescent\", "
          "\"Preconditioned\" -> False} (MRNSD of Nagy & Strakoš 2000, non-negative). Preconditioned "
          "\"SteepestDescent\" (the Wolfram default for that method), \"Hybrid\" and \"TotalVariation\" raise "
          "NotImplementedError. Image3D works. The result is a real image. Check: reference (exact recovery of a "
          "periodically blurred image with an invertible kernel and tiny p; Richardson–Lucy fixed point) in tests.")
def _imagedeconvolve(x: Expr, ker: MatrixOf[Real] | Expr, *opts: Option) -> Expr:
    fname = "ImageDeconvolve"
    pos, o = opts_split(opts, {"MaxIterations", "Method", "Padding"}, fname)
    _opt_check(pos, fname, 0)
    im, a, alpha, nd = _image_parts(x, fname)
    k = _deconv_kernel(ker, fname, nd, a.shape[-1])
    if any(ks > s for ks, s in zip(k.shape[:nd], a.shape[:nd])):
        raise ValueError(f"{fname}: the kernel must not be larger than the image")
    its = int_arg(o.get("MaxIterations", 10), fname, "MaxIterations", 1)
    m = o.get("Method", "DampedLS")
    method, param, sub = None, None, {}
    if isinstance(m, str):
        method = m
    elif K.is_list(m) and m.args and isinstance(m.args[0], str):
        method = m.args[0]
        for v in m.args[1:]:
            if isinstance(v, Expr) and v.head in ("Rule", "RuleDelayed") and isinstance(v.args[0], str):
                sub[v.args[0]] = v.args[1]
            elif param is None:
                param = v
            else:
                raise ValueError(f"{fname}: unexpected Method suboption {_short(v)}")
    elif _is_sym(m, "Automatic"):
        method = "DampedLS"
    if method not in _DECONV_METHODS:
        raise ValueError(f"{fname}: unknown Method {_short(m)}")
    if method in ("Hybrid", "TotalVariation"):
        raise NotImplementedError(f"{fname}: Method \"{method}\" is not supported")
    if method == "SteepestDescent":
        pre = sub.pop("Preconditioned", True)
        if not (pre is False or _is_sym(pre, "False")):
            raise NotImplementedError(f"{fname}: preconditioned \"SteepestDescent\" is not supported; use "
                                      "Method -> {\"SteepestDescent\", \"Preconditioned\" -> False}")
    if sub:
        raise NotImplementedError(f"{fname}: Method suboption {next(iter(sub))!r} is not supported")
    pad = o.get("Padding", "Reversed")
    if _is_sym(pad, "None"):
        raise NotImplementedError(f"{fname}: Padding -> None is not supported")
    pads = [s for s in k.shape[:nd]]
    if pad == "Periodic":
        P = a
        pads = [0] * nd
    else:
        padset = _pad_opt({"Padding": pad}, fname, nd, allow_list=False)[0]
        P = a
        for ax in range(nd):
            P = pad_axis(P, ax, pads[ax], pads[ax], padset)
    shape = P.shape[:nd]
    c = a.shape[-1]
    params = [None] * c
    if param is not None:
        if K.is_list(param):
            if len(param.args) != c:
                raise ValueError(f"{fname}: give one regularization parameter per channel ({c})")
            params = [_num(v, fname, "the regularization parameter", lo=0) for v in param.args]
        else:
            params = [_num(param, fname, "the regularization parameter", lo=0)] * c
    lap = np.zeros(tuple(shape))
    lap[(0,) * nd] = 2.0 * nd
    for ax in range(nd):
        for s in (1, -1):
            idx = [0] * nd
            idx[ax] = s % shape[ax]
            lap[tuple(idx)] -= 1.0
    Lap2 = np.abs(np.fft.fftn(lap)) ** 2
    out = np.zeros(P.shape)
    for ch in range(c):
        kc = k[..., min(ch, k.shape[-1] - 1):min(ch, k.shape[-1] - 1) + 1]
        H = _psf_fft(kc, shape, nd)
        g = P[..., ch]
        if method in ("DampedLS", "Tikhonov", "Wiener", "TSVD"):
            G = np.fft.fftn(g)
            H2 = np.abs(H) ** 2
            p = params[ch] if params[ch] is not None else _gcv_parameter(method, H2, G, Lap2)
            if method == "TSVD":
                X = np.where(np.abs(H) >= max(p, 1e-300), G / np.where(H == 0, 1, H), 0)
            else:
                den = {"Tikhonov": H2 + p * p, "Wiener": H2 + p, "DampedLS": H2 + p * p * Lap2}[method]
                X = np.where(den > 0, np.conj(H) * G / np.where(den > 0, den, 1), 0)
            out[..., ch] = np.real(np.fft.ifftn(X))
        else:
            if (g < 0).any() or (kc < 0).any():
                raise ValueError(f"{fname}: Method \"{method}\" needs a non-negative image and kernel")
            Hc = np.conj(H)
            if method == "RichardsonLucy":
                u = np.maximum(g, 1e-12)
                for _ in range(its):
                    est = np.real(np.fft.ifftn(np.fft.fftn(u) * H))
                    ratio = np.where(est > 1e-300, g / np.maximum(est, 1e-300), 0.0)
                    u = u * np.real(np.fft.ifftn(np.fft.fftn(ratio) * Hc))
            else:  # MRNSD
                u = np.maximum(g, 1e-12)
                for _ in range(its):
                    res = np.real(np.fft.ifftn(np.fft.fftn(u) * H)) - g
                    gr = np.real(np.fft.ifftn(np.fft.fftn(res) * Hc))
                    dvec = -u * gr
                    w = np.real(np.fft.ifftn(np.fft.fftn(dvec) * H))
                    ww = float((w * w).sum())
                    if ww <= 0:
                        break
                    tau_uc = float((u * gr * gr).sum()) / ww
                    neg = dvec < 0
                    tau_bd = float(np.min(-u[neg] / dvec[neg])) if neg.any() else math.inf
                    u = u + min(tau_uc, tau_bd) * dvec
                    u = np.maximum(u, 0.0)
            out[..., ch] = u
    sl = tuple(slice(p_, p_ + s) for p_, s in zip(pads, a.shape[:nd]))
    res = out[sl]
    return _img_result(im, res, alpha, typ=real_type(im))


# ───────────────────────────── Inpaint ─────────────────────────────

def _mask_of(spec, im, fname):
    """Boolean (h, w) mask of the region to retouch: an image or matrix (nonzero entries) or a 2D geometric region
    in the standard image coordinate system (pixel centres inside the region)."""
    h, w = im.a.shape[:2]
    if isinstance(spec, Expr) and spec.head in IMAGE_HEADS:
        m = parse_image(spec, fname)
        if m.d3 or m.a.shape[:2] != (h, w):
            raise ValueError(f"{fname}: the mask image must have the dimensions of the image")
        return np.abs(m.color()).max(-1) > 0
    if K.is_list(spec):
        arr = WI._numeric_array(spec, fname)
        if arr is not None and arr.shape == (h, w):
            return arr != 0
        if spec.args and all(K.is_list(t) and len(t.args) == 2 for t in spec.args):
            raise NotImplementedError(f"{fname}: regions given by positions (the component at {{x, y}}) are not "
                                      "supported; give a mask image, a matrix or a geometric region")
        raise ValueError(f"{fname}: a mask matrix must have the dimensions of the image ({h} x {w})")
    if isinstance(spec, Expr) and spec.head in ("Rectangle", "Disk", "Polygon", "Triangle", "Ball", "Circle",
                                                "Ellipsoid", "Parallelogram", "RegularPolygon", "Annulus",
                                                "ImplicitRegion", "BooleanRegion", "RegionUnion",
                                                "RegionIntersection", "RegionDifference"):
        return region_mask(spec, w, h, fname)
    if isinstance(spec, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_short(spec)} is not a region specification")


def region_mask(reg, w, h, fname, ranges=None):
    """Pixel-centre membership of a 2D region in image coordinates ({0, 0} bottom-left, {w, h} top-right)."""
    from .wl_regions import region_of
    r = region_of(reg, fname)
    xs = np.arange(w) + 0.5
    ys = h - (np.arange(h) + 0.5)
    m = np.zeros((h, w), dtype=bool)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            m[i, j] = bool(r.member((float(x), float(y))))
    return m


def harmonic_fill(a, mask):
    """Discrete harmonic interpolation (the fixed point of isotropic diffusion): Σ_neighbours (u_j - u_i) = 0 on
    the masked pixels (4-neighbours inside the image), known pixels fixed; conjugate gradients to 1e-12."""
    h, w, c = a.shape
    if mask.all():
        raise ValueError("Inpaint: the region covers the whole image (no known pixels)")
    out = np.array(a, dtype=float)
    M = mask.astype(float)
    deg = np.zeros((h, w))
    deg[1:, :] += 1
    deg[:-1, :] += 1
    deg[:, 1:] += 1
    deg[:, :-1] += 1

    def nbsum(u):
        s = np.zeros_like(u)
        s[1:, :] += u[:-1, :]
        s[:-1, :] += u[1:, :]
        s[:, 1:] += u[:, :-1]
        s[:, :-1] += u[:, 1:]
        return s

    def A(u):                      # u lives on the mask (zero elsewhere)
        return (deg * u - nbsum(u)) * M

    for ch in range(c):
        known = a[..., ch] * (1 - M)
        b = nbsum(known) * M
        x = np.zeros((h, w))
        r = b - A(x)
        p = r.copy()
        rs = float((r * r).sum())
        tol = 1e-24 * max(1.0, float((b * b).sum()))
        for _ in range(10 * int(M.sum()) + 100):
            if rs <= tol:
                break
            Ap = A(p)
            alpha = rs / float((p * Ap).sum())
            x += alpha * p
            r -= alpha * Ap
            rn = float((r * r).sum())
            p = r + (rn / rs) * p
            rs = rn
        else:
            raise ArithmeticError("Inpaint: the diffusion solve did not converge")
        out[..., ch] = np.where(mask, x, a[..., ch])
    return out


def fmm_distance(mask):
    """Arrival time T (fast marching, 4-neighbour first-order Eikonal |∇T| = 1, Sethian 1996) of the masked pixels
    from the known region (T = 0 there)."""
    import heapq
    h, w = mask.shape
    T = np.where(mask, np.inf, 0.0)
    done = ~mask
    heap = []
    for i, j in zip(*np.nonzero(mask)):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < h and 0 <= jj < w and not mask[ii, jj]:
                heapq.heappush(heap, (1.0, int(i), int(j)))
                break

    def solve(i, j):
        def mn(pairs):
            vals = [T[a, b] for a, b in pairs if 0 <= a < h and 0 <= b < w and done[a, b]]
            return min(vals) if vals else np.inf
        a = mn([(i - 1, j), (i + 1, j)])
        b = mn([(i, j - 1), (i, j + 1)])
        if np.isinf(a) and np.isinf(b):
            return np.inf
        if np.isinf(a) or np.isinf(b) or abs(a - b) >= 1:
            return min(a, b) + 1
        return (a + b + math.sqrt(2 - (a - b) ** 2)) / 2

    order = []
    while heap:
        t, i, j = heapq.heappop(heap)
        if done[i, j]:
            continue
        t = solve(i, j)
        T[i, j] = t
        done[i, j] = True
        order.append((i, j))
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < h and 0 <= jj < w and not done[ii, jj]:
                heapq.heappush(heap, (solve(ii, jj), ii, jj))
    return T, order


def telea_fill(a, mask, eps=5):
    """Telea (2004) fast-marching inpainting: pixels in order of arrival time, each from the known pixels within
    radius eps by I(q) + ∇I(q).(p - q) with weights dir . dst . lev."""
    h, w, c = a.shape
    T, order = fmm_distance(mask)
    out = np.array(a, dtype=float)
    known = ~mask.copy()
    offs = [(di, dj) for di in range(-eps, eps + 1) for dj in range(-eps, eps + 1)
            if 0 < di * di + dj * dj <= eps * eps]

    def tgrad(i, j):
        def d(ax):
            vals = []
            for s in (1, -1):
                ii, jj = (i + s, j) if ax == 0 else (i, j + s)
                if 0 <= ii < h and 0 <= jj < w and np.isfinite(T[ii, jj]):
                    vals.append((s, T[ii, jj]))
            if len(vals) == 2:
                return (vals[0][1] - vals[1][1]) / 2 * vals[0][0]
            if len(vals) == 1:
                s, v = vals[0]
                return (v - T[i, j]) * s
            return 0.0
        return d(0), d(1)

    def igrad(i, j):
        g = np.zeros((2, c))
        for ax in (0, 1):
            p = (i + 1, j) if ax == 0 else (i, j + 1)
            m = (i - 1, j) if ax == 0 else (i, j - 1)
            pk = 0 <= p[0] < h and 0 <= p[1] < w and known[p]
            mk = 0 <= m[0] < h and 0 <= m[1] < w and known[m]
            if pk and mk:
                g[ax] = (out[p] - out[m]) / 2
            elif pk:
                g[ax] = out[p] - out[i, j]
            elif mk:
                g[ax] = out[i, j] - out[m]
        return g

    for (i, j) in order:
        gi, gj = tgrad(i, j)
        num = np.zeros(c)
        den = 0.0
        for di, dj in offs:
            ii, jj = i + di, j + dj
            if not (0 <= ii < h and 0 <= jj < w and known[ii, jj]):
                continue
            ri, rj = i - ii, j - jj
            d2 = ri * ri + rj * rj
            dirw = abs(ri * gi + rj * gj) / math.sqrt(d2)
            dirw = max(dirw, 1e-6)
            dst = 1.0 / d2
            lev = 1.0 / (1.0 + abs(T[ii, jj] - T[i, j]))
            wgt = dirw * dst * lev
            g = igrad(ii, jj)
            num += wgt * (out[ii, jj] + g[0] * ri + g[1] * rj)
            den += wgt
        if den > 0:
            out[i, j] = num / den
        known[i, j] = True
    return out


def texture_fill(a, mask, source, nsamples=300, ncount=30, rng=None):
    """Best-fit texture synthesis by random sampling (Harrison 2001): unknown pixels in fast-marching order; each
    takes the colour of the source pixel whose neighbourhood best matches (mean squared difference) its
    `ncount` nearest known neighbours, among `nsamples` random source pixels and the coherent candidates
    suggested by the neighbours."""
    h, w, c = a.shape
    T, order = fmm_distance(mask)
    out = np.array(a, dtype=float)
    known = ~mask.copy()
    src_of = -np.ones((h, w, 2), dtype=int)
    ii, jj = np.nonzero(~mask)
    src_of[ii, jj, 0], src_of[ii, jj, 1] = ii, jj
    sy, sx = np.nonzero(source & ~mask)
    if len(sy) == 0:
        raise ValueError("Inpaint: no pixels are available as texture sources")
    R_ = 1
    offs = []
    while len(offs) < 4 * ncount and R_ < max(h, w):
        offs = sorted([(di, dj) for di in range(-R_, R_ + 1) for dj in range(-R_, R_ + 1) if (di, dj) != (0, 0)],
                      key=lambda t: (t[0] ** 2 + t[1] ** 2, t))
        R_ += 2
    offs = np.array(offs)
    rng = rng or np.random.default_rng(0)
    for (i, j) in order:
        P = offs + (i, j)
        ok = (P[:, 0] >= 0) & (P[:, 0] < h) & (P[:, 1] >= 0) & (P[:, 1] < w)
        P, O = P[ok], offs[ok]
        kn = known[P[:, 0], P[:, 1]]
        P, O = P[kn][:ncount], O[kn][:ncount]
        if len(P) == 0:
            continue
        vals = out[P[:, 0], P[:, 1]]                                   # (n, c)
        pick = rng.integers(0, len(sy), size=min(nsamples, len(sy)))
        cand = np.stack([sy[pick], sx[pick]], axis=1)
        coh = src_of[P[:, 0], P[:, 1]] - O
        cand = np.concatenate([cand, coh[(coh[:, 0] >= 0)]], axis=0)
        cand = cand[(cand[:, 0] >= 0) & (cand[:, 0] < h) & (cand[:, 1] >= 0) & (cand[:, 1] < w)]
        cand = cand[source[cand[:, 0], cand[:, 1]] & ~mask[cand[:, 0], cand[:, 1]]]
        if len(cand) == 0:
            continue
        Q = cand[:, None, :] + O[None, :, :]                            # (m, n, 2)
        inb = (Q[..., 0] >= 0) & (Q[..., 0] < h) & (Q[..., 1] >= 0) & (Q[..., 1] < w)
        Qc = np.clip(Q, 0, [h - 1, w - 1])
        valid = inb & known[Qc[..., 0], Qc[..., 1]]
        diff = ((out[Qc[..., 0], Qc[..., 1]] - vals[None]) ** 2).sum(-1)
        cnt = valid.sum(1)
        score = np.where(cnt > 0, (diff * valid).sum(1) / np.maximum(cnt, 1), np.inf)
        b = int(np.argmin(score))
        si, sj = cand[b]
        out[i, j] = out[si, sj]
        src_of[i, j] = (si, sj)
        known[i, j] = True
    return out


@fn("Inpaint", area=AREA, status="added",
    notes="Inpaint[image, region] synthesises the pixels of a 2D image where region is nonzero; region is a mask "
          "image or a 0/1 matrix of the image dimensions, or a 2D geometric region (Rectangle, Disk, Polygon, ...) "
          "in the standard image coordinate system (pixels whose centres lie in it). Method -> "
          "\"TextureSynthesis\" (default, as documented; best-fit texture synthesis by random sampling, Harrison "
          "2001: pixels in fast-marching order copy the source pixel whose neighbourhood best matches their "
          "\"NeighborCount\" (30) nearest known neighbours among \"MaxSamples\" (300) random candidates plus the "
          "coherent ones; Masking -> mask restricts the source pixels; random choices use the session generator, "
          "so SeedRandom makes them reproducible), \"Diffusion\" (the discrete harmonic interpolant, i.e. the "
          "steady state of isotropic diffusion, solved exactly by conjugate gradients), \"FastMarching\" (Telea "
          "2004, radius 5) and \"TotalVariation\" (MaxIterations (default 100) Chambolle–Pock iterations of "
          "\"Regularization\" λ TV(u) + data term on the known pixels, λ default 0.01; \"NoiseModel\" "
          "\"Gaussian\", \"Laplacian\" or \"Poisson\"). Pixels outside the region are never changed; the result "
          "has the dimensions and type of the image. \"NavierStokes\", regions given by positions and Image3D "
          "raise NotImplementedError. Check: reference (pixels outside the region unchanged, harmonic fill of a "
          "linear ramp is exact, constant images stay constant) in tests.")
def _inpaint(x: Expr, region: Expr, *opts: Option) -> Expr:
    fname = "Inpaint"
    pos, o = opts_split(opts, {"Method", "MaxIterations", "Masking"}, fname)
    _opt_check(pos, fname, 0)
    im, a, alpha, nd = _image_parts(x, fname, allow3d=False)
    mask = _mask_of(region, im, fname)
    m = o.get("Method", AUTO)
    sub = {}
    if K.is_list(m) and m.args and isinstance(m.args[0], str):
        for v in m.args[1:]:
            if not (isinstance(v, Expr) and v.head in ("Rule", "RuleDelayed") and isinstance(v.args[0], str)):
                raise ValueError(f"{fname}: Method suboptions must be rules")
            sub[v.args[0]] = v.args[1]
        m = m.args[0]
    if _is_sym(m, "Automatic"):
        m = "TextureSynthesis"
    if m not in ("TextureSynthesis", "Diffusion", "FastMarching", "TotalVariation", "NavierStokes"):
        raise ValueError(f"{fname}: unknown Method {_short(m)}")
    if m == "NavierStokes":
        raise NotImplementedError(f"{fname}: Method \"NavierStokes\" is not supported")
    if not mask.any():
        return x
    if mask.all():
        raise ValueError(f"{fname}: the region covers the whole image (no known pixels)")
    if m == "Diffusion":
        res = harmonic_fill(a, mask)
    elif m == "FastMarching":
        res = telea_fill(a, mask)
    elif m == "TotalVariation":
        model = _noise_model(sub.pop("NoiseModel", "Gaussian"), fname, "\"NoiseModel\"")
        reg = sub.pop("Regularization", AUTO)
        lam = 0.01 if _is_sym(reg, "Automatic") else _num(reg, fname, "\"Regularization\"", lo=0, strict=True)
        its = int_arg(o.get("MaxIterations", 100), fname, "MaxIterations", 1)
        start = harmonic_fill(a, mask)
        res = tv_minimize(start, 2, lam, model, its, known=~mask[..., None])
        res = np.where(mask[..., None], res, a)
    else:
        ns = int_arg(sub.pop("MaxSamples", 300), fname, "\"MaxSamples\"", 1)
        nc = int_arg(sub.pop("NeighborCount", 30), fname, "\"NeighborCount\"", 1)
        mk = o.get("Masking", AUTO)
        if _is_sym(mk, "Automatic") or _is_sym(mk, "All"):
            source = np.ones(mask.shape, dtype=bool)
        else:
            source = _mask_of(mk, im, fname)
        res = texture_fill(a, mask, source, ns, nc, WI._rng())
    if sub:
        raise NotImplementedError(f"{fname}: Method suboption {next(iter(sub))!r} is not supported")
    return _img_result(im, res, alpha)


# ───────────────────────────── ImageDemosaic ─────────────────────────────

_PRIMARY = {"R": 0, "G": 1, "B": 2}


def _cfa_pattern(spec, fname):
    """(pattern matrix of channel indices 0/1/2, row offset, column offset)."""
    off = (0, 0)
    if K.is_list(spec) and len(spec.args) == 2 and isinstance(spec.args[0], str) and K.is_list(spec.args[1]):
        o_ = spec.args[1]
        if len(o_.args) != 2:
            raise ValueError(f"{fname}: the offset must be {{row, col}}")
        off = (int_arg(o_.args[0], fname, "row", 1) - 1, int_arg(o_.args[1], fname, "column", 1) - 1)
        spec = spec.args[0]
    if isinstance(spec, str):
        s = "RGGB" if spec == "Bayer" else spec
        if s in ("CYGM", "CYGM2", "LRGB"):
            raise NotImplementedError(f"{fname}: the {s} filter array is not supported (its sensor colour model "
                                      "is not documented)")
        if len(s) == 4 and sorted(s) == ["B", "G", "G", "R"]:
            return np.array([[_PRIMARY[s[0]], _PRIMARY[s[1]]], [_PRIMARY[s[2]], _PRIMARY[s[3]]]]), off
        raise ValueError(f"{fname}: unknown colour filter array {spec!r}")
    if K.is_list(spec) and spec.args and all(K.is_list(r) for r in spec.args):
        from .wl_graphics_color import parse_color, to_rgba
        rows = []
        for r in spec.args:
            row = []
            for c in r.args:
                if parse_color(c) is None:
                    raise TypeError(f"{fname}: filter array entries must be colours, not {_short(c)}")
                rgba = to_rgba(c)
                rgb = [round(float(v), 9) for v in rgba[:3]]
                if sorted(rgb) != [0.0, 0.0, 1.0]:
                    raise NotImplementedError(f"{fname}: only pure Red, Green and Blue filter colours are supported")
                row.append(rgb.index(1.0))
            rows.append(row)
        if len({len(r) for r in rows}) != 1:
            raise ValueError(f"{fname}: the filter array must be rectangular")
        pat = np.array(rows)
        if set(pat.ravel()) != {0, 1, 2}:
            raise ValueError(f"{fname}: the filter array must contain red, green and blue filters")
        return pat, off
    if isinstance(spec, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_short(spec)} is not a colour filter array")


@fn("ImageDemosaic", area=AREA, status="added",
    notes="ImageDemosaic[image, cfa] reconstructs an RGB image from a single-channel raw image taken through the "
          "colour filter array cfa: \"RGGB\" (= \"Bayer\"), \"BGGR\", \"GRBG\", \"GBRG\", or an explicit matrix of "
          "Red/Green/Blue colours tiled over the image; {\"cfa\", {row, col}} puts the top-left element of the "
          "pattern at pixel {row, col} (1-based). Missing values are interpolated bilinearly by normalised "
          "convolution with {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}} of each channel's samples (the classical bilinear "
          "demosaicking; measured samples are kept). The type is kept. \"CYGM\", \"CYGM2\", \"LRGB\" and "
          "non-primary filter colours raise NotImplementedError; multichannel input raises ValueError. "
          "Check: reference (a uniform colour and a linear ramp are reconstructed exactly away from the border; "
          "measured samples unchanged) in tests.")
def _imagedemosaic(x: Expr, cfa: String | ListOf[Expr]) -> Expr:
    fname = "ImageDemosaic"
    im, a, alpha, nd = _image_parts(x, fname, allow3d=False)
    if im.c != 1:
        raise ValueError(f"{fname}: the image must have a single channel")
    pat, (r0, c0) = _cfa_pattern(cfa, fname)
    h, w = a.shape[:2]
    ph, pw = pat.shape
    ii = (np.arange(h)[:, None] - r0) % ph
    jj = (np.arange(w)[None, :] - c0) % pw
    chan = pat[ii, jj]
    k = np.array([1.0, 2.0, 1.0])
    raw = a[..., 0]
    out = np.zeros((h, w, 3))
    for c in range(3):
        m = (chan == c).astype(float)
        num = raw * m
        den = m.copy()
        for ax in (0, 1):
            num = correlate_axis_padded(num, k, ax, 0.0)
            den = correlate_axis_padded(den, k, ax, 0.0)
        est = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
        out[..., c] = np.where(chan == c, raw, est)
    if np.any([(chan == c).sum() == 0 for c in range(3)]):
        raise ValueError(f"{fname}: the image is too small for the filter array")
    typ = im.type if im.type != "Bit" else "Real32"
    return image_expr(WI.Img(quantize(out, typ), typ, "RGB", False))


# ───────────────────────────── BrightnessEqualize ─────────────────────────────

def _lightness(im, fname):
    """(L channel on 0..1 of the image's colours (spatial..., 1), source space)."""
    src = WI.source_space(im, fname)
    if src == "LAB":
        return im.color()[..., :1], src
    return WI.convert_array(im.color(), src, "LAB")[..., :1], src


def _coords(shape):
    """Normalised coordinates in [-1, 1] for every spatial axis."""
    return [np.linspace(-1.0, 1.0, s) if s > 1 else np.zeros(1) for s in shape]


def poly_fit(v, mask, deg, scope, fname):
    """Least-squares polynomial model of v (spatial...) on the masked samples: "Global" (total degree <= deg in
    all coordinates) or "Marginal" (the sum of one polynomial per axis fitted to the axis projections, minus the
    mean)."""
    shape = v.shape
    cs = _coords(shape)
    if not mask.any():
        raise ValueError(f"{fname}: no pixels are available for the model fit")
    if scope == "Global":
        import itertools
        exps = [e for e in itertools.product(range(deg + 1), repeat=len(shape)) if sum(e) <= deg]
        grids = np.meshgrid(*cs, indexing="ij")
        cols = [np.prod([g ** e for g, e in zip(grids, ex)], axis=0) for ex in exps]
        A = np.stack([c[mask] for c in cols], axis=1)
        coef, *_ = np.linalg.lstsq(A, v[mask], rcond=None)
        return sum(cf * c for cf, c in zip(coef, cols))
    out = np.zeros(shape) - (len(shape) - 1) * float(v[mask].mean())
    for ax in range(len(shape)):
        others = tuple(i for i in range(len(shape)) if i != ax)
        cnt = mask.sum(axis=others)
        prof = np.where(cnt > 0, (v * mask).sum(axis=others) / np.maximum(cnt, 1), 0)
        ok = cnt > 0
        V = np.vander(cs[ax][ok], deg + 1)
        coef, *_ = np.linalg.lstsq(V, prof[ok], rcond=None)
        fitted = np.vander(cs[ax], deg + 1) @ coef
        out = out + fitted.reshape([-1 if i == ax else 1 for i in range(len(shape))])
    return out


def _field(spec, im, Lch, mask, fname, what):
    """A flat- or dark-field model on the L scale (spatial...)."""
    shape = Lch.shape[:-1]
    if K.is_list(spec) and len(spec.args) == 2 and isinstance(spec.args[0], str):
        scope = spec.args[0]
        if scope not in ("Global", "Marginal"):
            raise ValueError(f"{fname}: the {what} scope must be \"Global\" or \"Marginal\"")
        deg = int_arg(spec.args[1], fname, "the polynomial degree", 0)
        return poly_fit(Lch[..., 0], mask, deg, scope, fname)
    if K.is_list(spec) and len(spec.args) == 3:
        raise NotImplementedError(f"{fname}: arbitrary {what} models {{f, params, vars}} are not supported")
    if isinstance(spec, Expr) and spec.head in IMAGE_HEADS:
        c = parse_image(spec, fname)
        if c.d3 != im.d3:
            raise ValueError(f"{fname}: the {what} image must have the dimensionality of the image")
        c = WI.resize_img(c, im.dims(), "Bilinear") if c.dims() != im.dims() else c
        return _lightness(c, fname)[0][..., 0]
    v = _num(spec, fname, f"the {what}")
    return np.full(shape, v)


@fn("BrightnessEqualize", area=AREA, status="added",
    notes="BrightnessEqualize[image], [image, flatfield] and [image, flatfield, darkfield] correct uneven "
          "illumination of the lightness channel L of LABColor (other channels kept): L' = (L - D) Mean[F - D]/(F - "
          "D), as documented. F and D may be a number, a correction image (rescaled bilinearly to the image "
          "dimensions; its L channel is used) or {scope, n}: a least-squares polynomial of degree n in the "
          "normalised coordinates, \"Global\" (total degree <= n over the whole image) or \"Marginal\" (one "
          "polynomial per axis fitted to the axis projections, combined additively). The default flatfield is "
          "{\"Global\", 2} (the documented 2nd order fit) and the default darkfield 0. Masking -> Automatic leaves "
          "pixels with L < 0.02 or L > 0.98 out of the fits (the over/underexposure thresholds are stated here); "
          "Masking -> All uses every pixel, a mask image selects pixels. F - D must be positive. Arbitrary models "
          "{f, params, vars} raise NotImplementedError. The type and colour space are kept. Check: reference (an "
          "image times a known illumination field divided back exactly; uniform images unchanged) in tests.")
def _brightnessequalize(x: Expr, *rest: Real | Expr | Option) -> Expr:
    fname = "BrightnessEqualize"
    pos, o = opts_split(rest, {"Masking", "PerformanceGoal"}, fname)
    _opt_check(pos, fname, 2)
    im, a, alpha, nd = _image_parts(x, fname)
    Lch, src = _lightness(im, fname)
    mk = o.get("Masking", AUTO)
    if _is_sym(mk, "Automatic"):
        mask = (Lch[..., 0] >= 0.02) & (Lch[..., 0] <= 0.98)
    elif _is_sym(mk, "All"):
        mask = np.ones(Lch.shape[:-1], dtype=bool)
    elif isinstance(mk, Expr) and mk.head in IMAGE_HEADS:
        m = parse_image(mk, fname)
        if m.a.shape[:-1] != Lch.shape[:-1]:
            raise ValueError(f"{fname}: the mask must have the dimensions of the image")
        mask = np.abs(m.color()).max(-1) > 0
    else:
        raise ValueError(f"{fname}: Masking must be Automatic, All or a mask image")
    if alpha is not None:
        mask = mask & (alpha[..., 0] > 0)
    F = _field(pos[0] if pos else L(*["Global", 2]), im, Lch, mask, fname, "flatfield")
    D = _field(pos[1], im, Lch, mask, fname, "darkfield") if len(pos) > 1 else np.zeros(F.shape)
    FD = F - D
    if not (FD > 0).all():
        raise ValueError(f"{fname}: the flatfield must exceed the darkfield everywhere")
    Lnew = (Lch[..., 0] - D) * float(FD.mean()) / FD
    if src == "LAB":
        col = np.concatenate([Lnew[..., None], a[..., 1:]], axis=-1)
    else:
        lab = WI.convert_array(a, src, "LAB")
        lab[..., 0] = Lnew
        col = WI.convert_array(lab, "LAB", src)
    return _img_result(im, col, alpha)


# ───────────────────────────── ColorToneMapping ─────────────────────────────

KR709 = np.array([0.2126, 0.7152, 0.0722])
MANTIUK_K1, MANTIUK_K2 = 1.6774, 0.9925          # Mantiuk et al. 2009, saturation correction


def bilateral(a, sig_s, sig_r):
    """Brute-force bilateral filter of a 2D array (Gaussian spatial weights, radius ceil(2 sig_s), range weights
    exp(-d^2/2 sig_r^2)), with smaller neighbourhoods at the borders."""
    r = max(1, int(math.ceil(2 * sig_s)))
    h, w = a.shape
    num = np.zeros_like(a)
    den = np.zeros_like(a)
    P = np.pad(a, r, mode="constant", constant_values=np.nan)
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            ws = math.exp(-(di * di + dj * dj) / (2 * sig_s * sig_s))
            sh = P[r + di:r + di + h, r + dj:r + dj + w]
            ok = ~np.isnan(sh)
            wt = np.where(ok, ws * np.exp(-((np.nan_to_num(sh) - a) ** 2) / (2 * sig_r * sig_r)), 0.0)
            num += wt * np.nan_to_num(sh)
            den += wt
    return num / den


@fn("ColorToneMapping", area=AREA, status="added",
    notes="ColorToneMapping[image] and ColorToneMapping[image, c] compress the luminance range by the factor c "
          "(0 <= c <= 1, default 1/2, as documented; c = 0 leaves the image unchanged). Luminance is Y = 0.2126 R "
          "+ 0.7152 G + 0.0722 B of the channel values as given (no linearisation; grayscale: the value). "
          "Method -> {\"BaseLuminance\", λ} (default, λ = 1): log10 Y is split into a base layer (bilateral "
          "filter, σ_spatial = 2% of the larger dimension, σ_range = 0.4; Durand & Dorsey 2002) and details; the "
          "base is scaled by 1 - c about its maximum and the details by λ. \"LogLuminance\": log Y scaled by 1 - c "
          "about its maximum (Y' = Ymax (Y/Ymax)^(1 - c)); \"Luminance\": linear compression about the mean, Y' = "
          "Mean[Y] + (1 - c)(Y - Mean[Y]); \"RGB\": each channel (v/vmax)^(1 - c). Results are scaled so that the "
          "largest luminance is 1, colours follow Schlick's rule C' = (C/Y)^s Y' (Fattal et al. 2002) with "
          "saturation factor s (third argument; Automatic: s = (1 + k1) c'^k2/(1 + k1 c'^k2), c' = 1 - c, k1 = "
          "1.6774, k2 = 0.9925 from Mantiuk et al. 2009), and values are clipped to 0..1 in a Real32 image (Real64 "
          "kept). Only the luminance range All is supported (ranges and \"DodgingAndBurning\" raise "
          "NotImplementedError). Images with ColorSpace Automatic are treated as RGB. Check: reference (c = 0 and "
          "s = 1 reproduce the image scaled to unit luminance; c = 1 with \"LogLuminance\" gives uniform luminance) "
          "in tests.")
def _colortonemapping(x: Expr, *rest: Real | Expr | Option) -> Expr:
    fname = "ColorToneMapping"
    pos, o = opts_split(rest, {"Method"}, fname)
    _opt_check(pos, fname, 2)
    im, a, alpha, nd = _image_parts(x, fname, allow3d=False)
    src = WI.source_space(im, fname)
    if src not in ("RGB", "Grayscale"):
        raise NotImplementedError(f"{fname}: images in the {src} colour space are not supported")
    c = 0.5
    if pos:
        spec = pos[0]
        if _is_sym(spec, "All"):
            pass
        elif K.is_list(spec) and len(spec.args) == 2 and _is_sym(spec.args[0], "All"):
            c = _num(spec.args[1], fname, "the compression factor", lo=0)
        elif K.is_list(spec) or isinstance(spec, str):
            raise NotImplementedError(f"{fname}: luminance ranges other than All are not supported")
        else:
            c = _num(spec, fname, "the compression factor", lo=0)
    if c > 1:
        raise ValueError(f"{fname}: the compression factor must be between 0 and 1")
    s = None
    if len(pos) > 1 and not _is_sym(pos[1], "Automatic"):
        s = _num(pos[1], fname, "the colour compensation factor", lo=0)
    if s is None:
        cp = 1 - c
        s = (1 + MANTIUK_K1) * cp ** MANTIUK_K2 / (1 + MANTIUK_K1 * cp ** MANTIUK_K2) if cp > 0 else 0.0
    m = o.get("Method", L(*["BaseLuminance", 1]))
    lam = 1.0
    if isinstance(m, str):
        meth = m
    elif K.is_list(m) and len(m.args) == 2 and m.args[0] == "BaseLuminance":
        meth = "BaseLuminance"
        lam = _num(m.args[1], fname, "the detail factor λ", lo=0)
    elif _is_sym(m, "Automatic"):
        meth = "BaseLuminance"
    else:
        raise ValueError(f"{fname}: unknown Method {_short(m)}")
    if meth == "DodgingAndBurning":
        raise NotImplementedError(f"{fname}: Method \"DodgingAndBurning\" is not supported")
    if meth not in ("RGB", "Luminance", "LogLuminance", "BaseLuminance"):
        raise ValueError(f"{fname}: unknown Method {meth!r}")
    if (a < 0).any():
        raise ValueError(f"{fname}: channel values must be non-negative")
    tiny = 1e-12
    if meth == "RGB":
        vmax = float(a.max()) or 1.0
        col = (np.maximum(a, tiny) / vmax) ** (1 - c)
        col = np.where(a > 0, col, 0.0)
    else:
        Y = a[..., 0] if src == "Grayscale" else a @ KR709
        Ymax = float(Y.max())
        if Ymax <= 0:
            col = np.zeros_like(a)
        else:
            if meth == "Luminance":
                Yn = Y.mean() + (1 - c) * (Y - Y.mean())
            elif meth == "LogLuminance":
                Yn = Ymax * (np.maximum(Y, tiny) / Ymax) ** (1 - c)
            else:
                lg = np.log10(np.maximum(Y, tiny * Ymax))
                base = bilateral(lg, 0.02 * max(a.shape[:2]), 0.4)
                det = lg - base
                bmax = float(base.max())
                Yn = 10 ** ((1 - c) * (base - bmax) + bmax + lam * det)
            Yn = np.where(Y > 0, np.maximum(Yn, 0.0), 0.0)
            Yn = Yn / (float(Yn.max()) or 1.0)
            if src == "Grayscale":
                col = Yn[..., None]
            else:
                ratio = np.where(Y[..., None] > 0, a / np.where(Y[..., None] > 0, Y[..., None], 1), 0.0)
                col = ratio ** s * Yn[..., None] if s != 1 else ratio * Yn[..., None]
    col = np.clip(col, 0.0, 1.0)
    typ = "Real64" if im.type == "Real64" else "Real32"
    cs = "Grayscale" if src == "Grayscale" else "RGB"
    if im.cs == "Automatic" and cs == "Grayscale":
        cs = "Automatic"
    return _img_result(im, col, alpha, typ=typ, cs=cs if im.cs != "Automatic" or cs == "RGB" else "Automatic")


# ───────────────────────────── exposure and focus fusion ─────────────────────────────

def _image_stack(lst, fname):
    if not K.is_list(lst):
        if isinstance(lst, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a list of images")
    if not lst.args:
        raise ValueError(f"{fname}: the list of images is empty")
    ims = [_image(v, fname, allow3d=False) for v in lst.args]
    dims = ims[0].dims()
    if any(i.dims() != dims for i in ims):
        raise ValueError(f"{fname}: the images must have the same dimensions")
    gray = all(WI.source_space(i, fname) == "Grayscale" for i in ims)
    arrs = []
    for i in ims:
        src = WI.source_space(i, fname)
        col = i.color()
        if gray:
            arrs.append(col[..., :1])
        else:
            arrs.append(WI.convert_array(col, src, "RGB") if src != "RGB" else col)
    return ims, arrs, gray


def _fusion_levels(shape):
    return max(1, pyramid_level_count(shape) - 1) if min(shape) > 1 else 1


def pyramid_blend(arrs, weights, nd=2):
    """Blend arrays with per-pixel weights (normalised to sum 1) in the Laplacian domain (Burt & Adelson 1983;
    Mertens et al. 2007): Σ_k G(w_k) L(I_k), then collapse."""
    n = _fusion_levels(arrs[0].shape[:nd])
    acc = None
    for a, wgt in zip(arrs, weights):
        La = laplacian_levels(a, nd, n)
        Gw = gaussian_levels(wgt[..., None], nd, n)
        lev = [l * g for l, g in zip(La, Gw)]
        acc = lev if acc is None else [x + y for x, y in zip(acc, lev)]
    return collapse_laplacian(acc, nd)


@fn("ImageExposureCombine", area=AREA, status="added",
    notes="ImageExposureCombine[{image1, image2, ...}] (aligned 2D images of equal dimensions) fuses differently "
          "exposed images into one low dynamic range image by exposure fusion (Mertens, Kautz & Van Reeth 2007): "
          "per-pixel weights contrast^1 x saturation^1 x well-exposedness^1 (|discrete Laplacian| of the gray "
          "image, standard deviation of R, G, B, and Π exp(-(v - 0.5)^2/(2 0.2^2)) over the channels), normalised "
          "over the images, blended in the Laplacian pyramid domain (Burt–Adelson pyramids, all levels) and "
          "clipped to 0..1. Images are converted to grayscale (all grayscale) or RGB, as documented; the result "
          "has the largest input type. Mode \"LDR\" is the default; \"HDR\" raises NotImplementedError (the "
          "documented forms give no exposure times for radiance recovery). PerformanceGoal is accepted and does "
          "not change the result. Check: reference (identical inputs give that image; weights favour "
          "well-exposed pixels) in tests.")
def _imageexposurecombine(imgs: ListOf[Expr], *rest: String | Option) -> Expr:
    fname = "ImageExposureCombine"
    pos, o = opts_split(rest, {"PerformanceGoal"}, fname)
    _opt_check(pos, fname, 1)
    mode = pos[0] if pos else "LDR"
    if mode == "HDR":
        raise NotImplementedError(f"{fname}: mode \"HDR\" is not supported (no exposure times are given)")
    if mode != "LDR":
        raise ValueError(f"{fname}: the mode must be \"LDR\" or \"HDR\", not {_short(mode)}")
    ims, arrs, gray = _image_stack(imgs, fname)
    ws = []
    for a in arrs:
        g = a.mean(-1)
        lap = np.abs(_neumann_shift(g, 0, 1) + _neumann_shift(g, 0, -1) + _neumann_shift(g, 1, 1) +
                     _neumann_shift(g, 1, -1) - 4 * g)
        sat = a.std(-1) if a.shape[-1] > 1 else np.ones(g.shape)
        wexp = np.prod(np.exp(-((a - 0.5) ** 2) / (2 * 0.2 ** 2)), axis=-1)
        ws.append(lap * sat * wexp + 1e-12)
    tot = sum(ws)
    ws = [w / tot for w in ws]
    res = np.clip(pyramid_blend(arrs, ws), 0.0, 1.0)
    typ = WI.larger_type(*[i.type for i in ims])
    if typ == "Bit":
        typ = "Real32"
    return image_expr(WI.Img(quantize(res, typ), typ, "Automatic" if gray else "RGB", False))


@fn("ImageFocusCombine", area=AREA, status="added",
    notes="ImageFocusCombine[{image1, image2, ...}] (aligned 2D images of equal dimensions) fuses differently "
          "focused images by Laplacian-pyramid fusion (Burt & Kolczynski 1993, simple selection rule): at every "
          "bandpass level each coefficient is taken from the image with the largest |coefficient| (summed over "
          "the channels), the lowpass residual is averaged, and the pyramid is collapsed and clipped to 0..1. "
          "Images are converted to grayscale (all grayscale) or RGB; the result has the largest input type. "
          "PerformanceGoal is accepted and does not change the result. Check: reference (identical inputs give "
          "that image; halves in focus are combined) in tests.")
def _imagefocuscombine(imgs: ListOf[Expr], *opts: Option) -> Expr:
    fname = "ImageFocusCombine"
    pos, o = opts_split(opts, {"PerformanceGoal"}, fname)
    _opt_check(pos, fname, 0)
    ims, arrs, gray = _image_stack(imgs, fname)
    n = _fusion_levels(arrs[0].shape[:2])
    pyrs = [laplacian_levels(a, 2, n) for a in arrs]
    fused = []
    for lev in range(n):
        stack = np.stack([p[lev] for p in pyrs])                # (k, h, w, c)
        if lev == n - 1:
            fused.append(stack.mean(0))
        else:
            pick = np.abs(stack).sum(-1).argmax(0)               # (h, w)
            fused.append(np.take_along_axis(stack, pick[None, ..., None], axis=0)[0])
    res = np.clip(collapse_laplacian(fused, 2), 0.0, 1.0)
    typ = WI.larger_type(*[i.type for i in ims])
    if typ == "Bit":
        typ = "Real32"
    return image_expr(WI.Img(quantize(res, typ), typ, "Automatic" if gray else "RGB", False))
