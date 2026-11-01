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
