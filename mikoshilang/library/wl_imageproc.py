"""Wolfram batch W28: images II (filtering, morphology, segmentation and feature detection).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md). It builds on W21 (``wl_images``): images are the canonical ``Image[data, type, ...]``
expressions, parsed into ``Img`` values (float arrays on the 0..1 scale, rows from the top).

Most functions accept either an image or a numeric array ("data"). Arrays use the index coordinate system
(``data[[i, j]]``), images their pixel matrix (rows from the top, columns from the left), so a radius list
``{r1, r2}`` for an image means {rows, columns}. Exact numeric arrays give exact results where the operation is
rational (MeanFilter, MedianFilter, Dilation, ...); machine arrays give machine numbers.

Pure NumPy does almost everything. A few heads that need SciPy (``scipy.ndimage``) say so in their notes and raise
``NotImplementedError("...: requires SciPy")`` on a minimal install. Nothing here needs Pillow.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, R, S
from . import wl_images as WI
from .wl_images import parse_image, image_expr, opts_split, int_arg, real_arg, pad_array, quantize, MAXV, \
    IMAGE_HEADS, _short, _is_sym

AREA = "image processing (W28)"
TRUE, FALSE = Symbol("True"), Symbol("False")
AUTO = Symbol("Automatic")


def _scipy(fname):
    try:
        import scipy.ndimage as ndi
    except ImportError:  # minimal install
        raise NotImplementedError(f"{fname}: requires SciPy (pip install scipy); it is not installed")
    return ndi


def _ev(e):
    from ..evaluate import evaluate
    return evaluate(e)


def _truthy(v, fname, what):
    if v is True or _is_sym(v, "True"):
        return True
    if v is False or _is_sym(v, "False"):
        return False
    raise ValueError(f"{fname}: {what} must be True or False, not {_short(v)}")


# ───────────────────────────── data: an image or an array ─────────────────────────────

class Data:
    """An image (``img`` set, ``a`` = its (..., c) float array) or an array (``a`` float ndarray, or ``obj`` an
    object ndarray of SymPy numbers when ``exact``)."""
    __slots__ = ("img", "a", "exact", "obj", "sym")

    def __init__(self, img=None, a=None, exact=False, obj=None, sym=None):
        self.img, self.a, self.exact, self.obj, self.sym = img, a, exact, obj, sym

    @property
    def is_image(self):
        return self.img is not None

    @property
    def spatial(self):
        """Number of spatial axes (2 or 3 for images, the rank for arrays)."""
        return self.a.ndim - 1 if self.is_image else (self.obj if self.exact else self.a).ndim

    def shape(self):
        return self.a.shape[:-1] if self.is_image else (self.obj if self.exact else self.a).shape


def _dims(nest):
    """Dimensions of a rectangular nested Python list (ValueError if ragged)."""
    if not isinstance(nest, list):
        return ()
    if not nest:
        return (0,)
    subs = [_dims(v) for v in nest]
    if any(s != subs[0] for s in subs):
        raise ValueError("ragged")
    return (len(nest),) + subs[0]


def _leaves(nest):
    if isinstance(nest, list):
        for v in nest:
            yield from _leaves(v)
    else:
        yield nest


def data_arg(x, fname, allow3d=True, symbolic=False, exact_ok=True, min_rank=1):
    """Data of an image or a rectangular numeric array. Strings and booleans raise TypeError; symbolic entries make
    the call unevaluated (or are kept, with ``symbolic=True``, in ``obj`` with ``sym`` set)."""
    if isinstance(x, Expr) and x.head in IMAGE_HEADS:
        im = parse_image(x, fname, allow3d=allow3d)
        return Data(img=im, a=im.a)
    if not K.is_list(x):
        if isinstance(x, Symbol) or (isinstance(x, Expr) and x.head not in ("Rule", "Association", "Graphics")):
            if isinstance(x, Expr) and x.head in ("Audio", "Video", "TimeSeries", "TemporalData", "Sound"):
                raise NotImplementedError(f"{fname}: {x.head} data is not supported")
            raise Unevaluated()
        raise TypeError(f"{fname}: expected an image or a numeric array, not {_short(x)}")
    nest = WI._nested(x)
    try:
        dims = _dims(nest)
    except ValueError:
        raise ValueError(f"{fname}: the data must be a rectangular array")
    if len(dims) < min_rank or 0 in dims:
        raise ValueError(f"{fname}: the data must be a non-empty array of rank at least {min_rank}")
    leaves = list(_leaves(nest))
    for v in leaves:
        if isinstance(v, (bool, str)) or v is True or v is False:
            raise TypeError(f"{fname}: array elements must be numbers, not {_short(v)}")
    floats = all(isinstance(v, float) for v in leaves)
    if floats:
        return Data(a=np.array(nest, dtype=float))
    conv, inexact, nonnum = [], False, False
    for v in leaves:
        if isinstance(v, float):
            inexact = True
            conv.append(sp.Float(v))
            continue
        s = S(v)
        if not (isinstance(s, sp.Basic) and s.is_number):
            nonnum = True
            conv.append(v)
            continue
        if s.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            raise ValueError(f"{fname}: array elements must be finite, not {_short(v)}")
        if s.is_real is False:
            raise NotImplementedError(f"{fname}: complex array elements are not supported")
        if False:
            raise ValueError(f"{fname}: array elements must be finite real numbers, not {_short(v)}")
        if any(isinstance(t, sp.Float) for t in sp.preorder_traversal(s)):
            inexact = True
        conv.append(s)
    if nonnum:
        if not symbolic:
            raise Unevaluated()
        obj = np.empty(len(leaves), dtype=object)
        obj[:] = conv
        return Data(obj=obj.reshape(dims), exact=True, sym=True)
    if inexact or not exact_ok:
        return Data(a=np.array([float(c) for c in conv]).reshape(dims))
    obj = np.empty(len(conv), dtype=object)
    obj[:] = conv
    return Data(obj=obj.reshape(dims), exact=True)


def _float_nest(a):
    if a.ndim == 0:
        return float(a)
    if a.ndim == 1:
        return L(*[float(v) for v in a])
    return L(*[_float_nest(a[i]) for i in range(a.shape[0])])


def _obj_nest(a):
    if a.ndim == 1:
        return L(*[R(v) if isinstance(v, sp.Basic) else v for v in a])
    return L(*[_obj_nest(a[i]) for i in range(a.shape[0])])


def array_out(a):
    """List expression of a float or object ndarray."""
    if a.dtype == object:
        return _obj_nest(a)
    return _float_nest(np.asarray(a, dtype=float))


def int_nest(a):
    a = np.asarray(a)
    if a.ndim == 0:
        return int(a)
    return L(*[int_nest(v) for v in a])


def image_out(img, a, typ=None, cs=None):
    """Image expression of array a shaped like img.a (or with one channel), on the 0..1 scale; integer types are
    quantised (clipped)."""
    typ = img.type if typ is None else typ
    if a.ndim == img.a.ndim - 1:
        a = a[..., None]
    if cs is None:
        cs = img.cs if a.shape[-1] == img.c else ("Automatic" if a.shape[-1] == 1 else img.cs)
    return image_expr(WI.Img(quantize(a, typ), typ, cs, img.d3))


def real_type(img):
    return "Real64" if img.type == "Real64" else "Real32"


def smooth_type(img):
    """Type of a smoothing filter's result: the image type, but "Bit" images give "Real32"."""
    return "Real32" if img.type == "Bit" else img.type


def binary_image(img, mask):
    """A "Bit" single-channel image (same 2D/3D shape as img) from a boolean array of its spatial shape."""
    return image_expr(WI.Img(np.asarray(mask, dtype=float)[..., None], "Bit", "Automatic", img.d3))


def binary_like(d, mask):
    """Binary result of the kind of d: a Bit image, or a 0/1 integer array."""
    if d.is_image:
        return binary_image(d.img, mask)
    return int_nest(np.asarray(mask, dtype=np.int64))


def intensity(img):
    """Single-channel intensity (the mean of the colour channels, alpha dropped) of an Img, shape spatial."""
    c = img.color()
    return c.mean(axis=-1)


# ───────────────────────────── radii ─────────────────────────────

def radii_arg(r, fname, nd, what="radius", allow_real=False):
    """Per-axis non-negative integer radii from r or {r1, ..., rn}."""
    if K.is_list(r):
        if len(r.args) != nd:
            raise ValueError(f"{fname}: give one {what} per dimension ({nd}), not {len(r.args)}")
        vals = list(r.args)
    else:
        vals = [r] * nd
    out = []
    for v in vals:
        if isinstance(v, (bool, str)) or K.is_list(v):
            raise TypeError(f"{fname}: the {what} must be a non-negative integer, not {_short(v)}")
        if K.is_int(v):
            iv = int(v)
        elif isinstance(v, float) or WI.is_real_value(v):
            if allow_real:
                fv = real_arg(v, fname, what)
                if fv < 0:
                    raise ValueError(f"{fname}: the {what} must be non-negative")
                out.append(fv)
                continue
            raise NotImplementedError(f"{fname}: non-integer {what} {_short(v)} is not supported")
        else:
            raise Unevaluated()
        if iv < 0:
            raise ValueError(f"{fname}: the {what} must be non-negative, not {iv}")
        out.append(iv)
    return out


# ───────────────────────────── neighbourhood engine ─────────────────────────────

def _windows_float(a, rad):
    """(shape..., m) window values with NaN outside the array (smaller neighbourhoods at the boundaries)."""
    from numpy.lib.stride_tricks import sliding_window_view
    P = np.pad(a, [(r, r) for r in rad], mode="constant", constant_values=np.nan)
    win = sliding_window_view(P, tuple(2 * r + 1 for r in rad))
    return win.reshape(a.shape + (-1,))


def _windows_obj(a, rad):
    """Iterator of (index, list of values) over an object array (truncated windows)."""
    for idx in np.ndindex(a.shape):
        sl = tuple(slice(max(0, i - r), i + r + 1) for i, r in zip(idx, rad))
        yield idx, list(a[sl].reshape(-1))


def _block(a, idx, lo, hi):
    sl = tuple(slice(max(0, i - l), i + h + 1) for i, l, h in zip(idx, lo, hi))
    return a[sl]


def _per_channel(img, f):
    """Apply f to each channel (spatial array) of an Img and stack."""
    return np.stack([f(img.a[..., k]) for k in range(img.c)], axis=-1)


def _ex_mean(v):
    return sp.Add(*v) / len(v)


def _ex_median(v):
    s = sorted(v, key=lambda t: float(t))
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _ex_sd(v):
    n = len(v)
    if n < 2:
        return sp.Integer(0)
    m = _ex_mean(v)
    return sp.sqrt(sp.Add(*[(t - m) ** 2 for t in v]) / (n - 1))


def _ex_gmean(v):
    if any(t < 0 for t in v):
        raise ValueError("GeometricMeanFilter: the data must be non-negative")
    return sp.Mul(*v) ** sp.Rational(1, len(v))


def _ex_hmean(v):
    if any(t == 0 for t in v):
        return sp.Integer(0)
    s = sp.Add(*[1 / t for t in v])
    if s == 0:
        raise ValueError("HarmonicMeanFilter: the reciprocals of a neighbourhood sum to zero")
    return len(v) / s


def _ex_entropy(v):
    cnt = Counter(v)
    n = len(v)
    return sp.Add(*[-sp.Rational(c, n) * sp.log(sp.Rational(c, n)) for c in cnt.values()])


def _fl_entropy(W):
    out = np.zeros(W.shape[:-1])
    flat = W.reshape(-1, W.shape[-1])
    o = out.reshape(-1)
    for i, row in enumerate(flat):
        row = row[~np.isnan(row)]
        _, c = np.unique(row, return_counts=True)
        p = c / c.sum()
        o[i] = -(p * np.log(p)).sum()
    return out


def _fl_gmean(W):
    if np.nanmin(W) < 0:
        raise ValueError("GeometricMeanFilter: the data must be non-negative")
    with np.errstate(divide="ignore"):
        lg = np.log(W)
    return np.exp(np.nanmean(lg, axis=-1))


def _fl_hmean(W):
    zero = np.any(W == 0, axis=-1)
    with np.errstate(divide="ignore"):
        s = np.nansum(1.0 / np.where(W == 0, np.nan, W), axis=-1)
    cnt = np.sum(~np.isnan(W), axis=-1)
    if np.any((s == 0) & ~zero):
        raise ValueError("HarmonicMeanFilter: the reciprocals of a neighbourhood sum to zero")
    return np.where(zero, 0.0, cnt / np.where(s == 0, 1, s))


def _fl_sd(W):
    cnt = np.sum(~np.isnan(W), axis=-1)
    m = np.nanmean(W, axis=-1)
    ss = np.nansum((W - m[..., None]) ** 2, axis=-1)
    return np.sqrt(np.where(cnt > 1, ss / np.maximum(cnt - 1, 1), 0.0))


_REDUCE = {
    "MaxFilter": (lambda W: np.nanmax(W, -1), max),
    "MinFilter": (lambda W: np.nanmin(W, -1), min),
    "RangeFilter": (lambda W: np.nanmax(W, -1) - np.nanmin(W, -1), lambda v: max(v) - min(v)),
    "MeanFilter": (lambda W: np.nanmean(W, -1), _ex_mean),
    "MedianFilter": (lambda W: np.nanmedian(W, -1), _ex_median),
    "StandardDeviationFilter": (_fl_sd, _ex_sd),
    "GeometricMeanFilter": (_fl_gmean, _ex_gmean),
    "HarmonicMeanFilter": (_fl_hmean, _ex_hmean),
    "EntropyFilter": (_fl_entropy, _ex_entropy),
}

_SELECT = {"MaxFilter": "max", "MinFilter": "min", "MedianFilter": "median"}


def _select_vectors(img, rad, how):
    """Multichannel Max/Min/Median filters: each pixel becomes the neighbourhood pixel of maximal / minimal / median
    intensity (mean of the colour channels); ties go to the first such pixel in scan order."""
    I = _windows_float(intensity(img), rad)
    V = np.stack([_windows_float(img.a[..., k], rad) for k in range(img.c)], axis=-1)   # (..., m, c)
    if how == "max":
        j = np.nanargmax(I, axis=-1)
    elif how == "min":
        j = np.nanargmin(I, axis=-1)
    else:
        key = np.where(np.isnan(I), np.inf, I)
        order = np.argsort(key, axis=-1, kind="stable")
        cnt = np.sum(~np.isnan(I), axis=-1)
        j = np.take_along_axis(order, ((cnt - 1) // 2)[..., None], axis=-1)[..., 0]
    return np.take_along_axis(V, j[..., None, None], axis=-2)[..., 0, :]


def _filter_notes(fname, f, extra=""):
    return (f"{fname}[data, r] replaces every element by {f} of its range-r neighbourhood (a (2r+1)^n block), "
            f"{fname}[data, {{r1, ..., rn}}] uses a radius per dimension; at the boundaries the neighbourhoods are "
            "smaller, as documented. data is a numeric array of any rank (exact input gives exact results, machine "
            "input machine numbers) or an Image/Image3D (radii {rows, columns}); " + extra +
            "Symbolic entries stay unevaluated; non-integer radii, TimeSeries, Audio and Video raise "
            "NotImplementedError; strings raise TypeError, negative radii ValueError.")


def _nbhd_filter(fname, x, r, extra_check=None):
    d = data_arg(x, fname)
    rad = radii_arg(r, fname, d.spatial)
    fl, ex = _REDUCE[fname]
    if not d.is_image:
        if d.exact:
            out = np.empty(d.obj.shape, dtype=object)
            for idx, v in _windows_obj(d.obj, rad):
                out[idx] = ex(v)
            return array_out(out)
        return array_out(fl(_windows_float(d.a, rad)))
    img = d.img
    if img.c > 1 and fname in _SELECT:
        out = _select_vectors(img, rad, _SELECT[fname])
        return image_out(img, out)
    out = _per_channel(img, lambda ch: fl(_windows_float(ch, rad)))
    if fname in ("MaxFilter", "MinFilter", "MedianFilter", "RangeFilter"):
        return image_out(img, out)
    if fname in ("StandardDeviationFilter", "EntropyFilter"):
        return image_out(img, out, typ=real_type(img))
    return image_out(img, out, typ=smooth_type(img))


_IMG_SELECT_NOTE = ("multichannel images: each pixel becomes the neighbourhood pixel with the {} intensity (mean of "
                    "the colour channels; ties: first in scan order), as documented; the image type is kept. ")
_IMG_CHAN_NOTE = "images: every channel separately; the type is kept except that \"Bit\" images give \"Real32\". "


@fn("MaxFilter", area=AREA, status="added",
    notes=_filter_notes("MaxFilter", "the maximum", _IMG_SELECT_NOTE.format("maximal")) +
    " Check: reference (Wolfram documentation examples; direct window maxima) in tests.")
def _maxfilter(x: Expr, r: NonNegInteger | ListOf[NonNegInteger]) -> Expr:
    return _nbhd_filter("MaxFilter", x, r)


@fn("MinFilter", area=AREA, status="added",
    notes=_filter_notes("MinFilter", "the minimum", _IMG_SELECT_NOTE.format("minimal")) +
    " Check: reference (Wolfram documentation examples; MinFilter[-x] == -MaxFilter[x]) in tests.")
def _minfilter(x: Expr, r: NonNegInteger | ListOf[NonNegInteger]) -> Expr:
    return _nbhd_filter("MinFilter", x, r)


@fn("RangeFilter", area=AREA, status="added",
    notes=_filter_notes("RangeFilter", "Max - Min", "images: every channel separately, type kept. ") +
    " Check: reference (Wolfram documentation example; RangeFilter == MaxFilter - MinFilter) in tests.")
def _rangefilter(x: Expr, r: NonNegInteger | ListOf[NonNegInteger]) -> Expr:
    return _nbhd_filter("RangeFilter", x, r)


@fn("MeanFilter", area=AREA, status="added",
    notes=_filter_notes("MeanFilter", "the mean", _IMG_CHAN_NOTE) +
    " Check: reference (Wolfram documentation examples, exact) in tests.")
def _meanfilter(x: Expr, r: NonNegInteger | ListOf[NonNegInteger]) -> Expr:
    return _nbhd_filter("MeanFilter", x, r)


@fn("MedianFilter", area=AREA, status="added",
    notes=_filter_notes("MedianFilter", "the median (the mean of the two middle values for an even count)",
                        _IMG_SELECT_NOTE.format("median (lower median for an even count)")) +
    " Check: reference (Wolfram documentation examples) in tests.")
def _medianfilter(x: Expr, r: NonNegInteger | ListOf[NonNegInteger]) -> Expr:
    return _nbhd_filter("MedianFilter", x, r)


@fn("StandardDeviationFilter", area=AREA, status="added",
    notes=_filter_notes("StandardDeviationFilter", "the (sample, n - 1) standard deviation (0 for a single value)",
                        "images: every channel separately, real result type. ") +
    " Check: reference (Wolfram documentation example: 1/Sqrt[3] values) in tests.")
def _sdfilter(x: Expr, r: NonNegInteger | ListOf[NonNegInteger]) -> Expr:
    return _nbhd_filter("StandardDeviationFilter", x, r)


@fn("GeometricMeanFilter", area=AREA, status="added",
    notes=_filter_notes("GeometricMeanFilter", "the geometric mean (x1 ... xm)^(1/m)", _IMG_CHAN_NOTE +
                        "Negative values raise ValueError. ") +
    " Check: reference (Wolfram documentation examples, exact radicals) in tests.")
def _gmeanfilter(x: Expr, r: NonNegInteger | ListOf[NonNegInteger]) -> Expr:
    return _nbhd_filter("GeometricMeanFilter", x, r)


@fn("HarmonicMeanFilter", area=AREA, status="added",
    notes=_filter_notes("HarmonicMeanFilter", "the harmonic mean m / Sum[1/xi]", _IMG_CHAN_NOTE +
                        "A neighbourhood containing 0 gives 0 (as Wolfram's HarmonicMean does); reciprocals summing "
                        "to 0 raise ValueError. ") +
    " Check: reference (Wolfram documentation examples, exact) in tests.")
def _hmeanfilter(x: Expr, r: NonNegInteger | ListOf[NonNegInteger]) -> Expr:
    return _nbhd_filter("HarmonicMeanFilter", x, r)


@fn("EntropyFilter", area=AREA, status="added",
    notes=_filter_notes("EntropyFilter", "the entropy -Sum[p Log[p]] of the tally of its distinct values (natural "
                        "logarithm, as Entropy)", "images: every channel separately on the 0..1 values, real result "
                        "type. ") +
    " Check: reference (Wolfram documentation examples, exact logarithms and machine values) in tests.")
def _entropyfilter(x: Expr, r: NonNegInteger | ListOf[NonNegInteger]) -> Expr:
    return _nbhd_filter("EntropyFilter", x, r)


# CommonestFilter accepts arbitrary (also symbolic) elements.

def _commonest(vals, center):
    cnt = Counter(vals)
    top = max(cnt.values())
    best = [v for v in cnt if cnt[v] == top]
    if center in best:
        return center
    for v in vals:            # first commonest in scan order (Wolfram picks a random one)
        if cnt[v] == top:
            return v


@fn("CommonestFilter", area=AREA, status="added",
    notes="CommonestFilter[data, r] replaces every element by the most common value of its range-r neighbourhood "
          "(smaller neighbourhoods at the boundaries), CommonestFilter[data, {r1, ...}] per dimension. If several "
          "values are commonest and the centre is one of them, the centre is kept, as documented; otherwise the "
          "first commonest value in scan order is used (Wolfram picks a random one). Arrays may hold any "
          "expressions (compared structurally); images: the commonest pixel vector, type kept. Non-integer radii "
          "raise NotImplementedError. Check: reference (Wolfram documentation examples) in tests.")
def _commonestfilter(x: Expr, r: NonNegInteger | ListOf[NonNegInteger]) -> Expr:
    fname = "CommonestFilter"
    d = data_arg(x, fname, symbolic=True)
    rad = radii_arg(r, fname, d.spatial)
    if not d.is_image:
        if d.exact:
            src = d.obj
            out = np.empty(src.shape, dtype=object)
            for idx, v in _windows_obj(src, rad):
                out[idx] = _commonest(v, src[idx])
            return array_out(out)
        src = d.a
        out = np.empty(src.shape)
        for idx in np.ndindex(src.shape):
            sl = tuple(slice(max(0, i - q), i + q + 1) for i, q in zip(idx, rad))
            out[idx] = _commonest(list(src[sl].reshape(-1)), src[idx])
        return array_out(out)
    img = d.img
    a = img.a
    sp_shape = a.shape[:-1]
    out = np.empty_like(a)
    for idx in np.ndindex(sp_shape):
        sl = tuple(slice(max(0, i - q), i + q + 1) for i, q in zip(idx, rad))
        blk = a[sl].reshape(-1, img.c)
        vals = [tuple(v) for v in blk]
        out[idx] = _commonest(vals, tuple(a[idx]))
    return image_out(img, out)


# KuwaharaFilter and MeanShiftFilter

def _kuwahara_blocks(nd, r):
    import itertools
    return list(itertools.product(*[((r[k], 0), (0, r[k])) for k in range(nd)]))


@fn("KuwaharaFilter", area=AREA, status="added",
    notes="KuwaharaFilter[data, r] computes, for each element p, the variance of the values in the 2^n blocks of "
          "size (r + 1) that have p as a corner (smaller at the boundaries) and replaces p by the mean of the block "
          "with the least variance, as documented; KuwaharaFilter[data, {r1, ...}] per dimension. The variance is "
          "the population variance (so one-element boundary blocks have variance 0); ties go to the first block in "
          "the order (before, after) per dimension. Multichannel images sum the channel variances, as documented. "
          "Exact arrays give exact means; images keep their type (\"Bit\" gives \"Real32\"). Check: reference "
          "(Wolfram documentation example) in tests.")
def _kuwaharafilter(x: Expr, r: NonNegInteger | ListOf[NonNegInteger]) -> Expr:
    fname = "KuwaharaFilter"
    d = data_arg(x, fname)
    rad = radii_arg(r, fname, d.spatial)
    blocks = _kuwahara_blocks(d.spatial, rad)
    if not d.is_image and d.exact:
        src = d.obj
        out = np.empty(src.shape, dtype=object)
        for idx in np.ndindex(src.shape):
            best = None
            for bl in blocks:
                v = list(_block(src, idx, [b[0] for b in bl], [b[1] for b in bl]).reshape(-1))
                m = _ex_mean(v)
                var = sp.Add(*[(t - m) ** 2 for t in v]) / len(v)
                if best is None or var < best[0]:
                    best = (var, m)
            out[idx] = best[1]
        return array_out(out)
    a = d.a if d.is_image else d.a[..., None]
    sh = a.shape[:-1]
    out = np.empty_like(a)
    for idx in np.ndindex(sh):
        best = None
        for bl in blocks:
            blk = _block(a, idx, [b[0] for b in bl], [b[1] for b in bl]).reshape(-1, a.shape[-1])
            var = float(blk.var(axis=0).sum())
            if best is None or var < best[0] - 1e-15:
                best = (var, blk.mean(axis=0))
        out[idx] = best[1]
    if d.is_image:
        return image_out(d.img, out, typ=smooth_type(d.img))
    return array_out(out[..., 0])


_MS_DIST = ("EuclideanDistance", "SquaredEuclideanDistance", "ManhattanDistance", "ChessboardDistance")


def _ms_dist(name, diff):
    if name == "EuclideanDistance":
        return np.sqrt((diff ** 2).sum(-1))
    if name == "SquaredEuclideanDistance":
        return (diff ** 2).sum(-1)
    if name == "ManhattanDistance":
        return np.abs(diff).sum(-1)
    return np.abs(diff).max(-1)


@fn("MeanShiftFilter", area=AREA, status="added",
    notes="MeanShiftFilter[data, r, d] replaces every value v by the mean of the values in its range-r "
          "neighbourhood (smaller at the boundaries) that lie within distance d of v (inclusive); "
          "MeanShiftFilter[data, {r1, ...}, d] per dimension. MaxIterations -> n repeats the step from the shifted "
          "value against the original neighbourhood (default 1, as documented); DistanceFunction -> "
          "EuclideanDistance (default), SquaredEuclideanDistance, ManhattanDistance or ChessboardDistance; "
          "multichannel images compare channel vectors, as documented. The result is machine real (as in the "
          "documented examples); images keep their type (\"Bit\" gives \"Real32\"). Other distance functions raise "
          "NotImplementedError, negative d ValueError. Check: reference (Wolfram documentation examples) in tests.")
def _meanshiftfilter(x: Expr, r: NonNegInteger | ListOf[NonNegInteger], dd: Real, *opts: Option) -> Expr:
    fname = "MeanShiftFilter"
    pos, o = opts_split(opts, {"DistanceFunction", "MaxIterations"}, fname)
    if pos:
        raise TypeError(f"{fname}: expected MeanShiftFilter[data, r, d, options]")
    d = data_arg(x, fname, exact_ok=False)
    rad = radii_arg(r, fname, d.spatial)
    if isinstance(dd, Symbol):
        raise Unevaluated()
    dist = real_arg(dd, fname, "distance")
    if dist < 0:
        raise ValueError(f"{fname}: the distance must be non-negative")
    df = o.get("DistanceFunction", Symbol("EuclideanDistance"))
    if _is_sym(df, "Automatic"):
        df = Symbol("EuclideanDistance")
    if not (isinstance(df, Symbol) and df.name in _MS_DIST):
        raise NotImplementedError(f"{fname}: DistanceFunction {_short(df)} is not supported")
    n_it = int_arg(o.get("MaxIterations", 1), fname, "MaxIterations", 1)
    a = d.a if d.is_image else d.a[..., None]
    if d.is_image and d.img.alpha:
        a = d.img.color()
    V = np.stack([_windows_float(a[..., k], rad) for k in range(a.shape[-1])], axis=-1)   # (..., m, c)
    cur = a.copy()
    valid = ~np.isnan(V[..., 0])
    for _ in range(n_it):
        dv = _ms_dist(df.name, np.where(valid[..., None], V, 0.0) - cur[..., None, :])
        sel = valid & (dv <= dist + 1e-12 * max(1.0, dist))
        cnt = sel.sum(-1)
        new = (np.where(sel[..., None], V, 0.0).sum(-2)) / np.maximum(cnt, 1)[..., None]
        cur = np.where(cnt[..., None] > 0, new, cur)
    if d.is_image:
        img = d.img
        if img.alpha:
            cur = np.concatenate([cur, img.alpha_array()], axis=-1)
        return image_out(img, cur, typ=smooth_type(img))
    return array_out(cur[..., 0])


# ───────────────────────────── morphology: structuring elements and shifts ─────────────────────────────

def _shift(a, off, fill):
    """b with b[x] = a[x + off] (fill outside) over the leading len(off) axes."""
    out = np.full(a.shape, fill, dtype=a.dtype)
    src, dst = [], []
    for o, n in zip(off, a.shape):
        if abs(o) >= n:
            return out
        if o >= 0:
            src.append(slice(o, n))
            dst.append(slice(0, n - o))
        else:
            src.append(slice(0, n + o))
            dst.append(slice(-o, n))
    out[tuple(dst)] = a[tuple(src)]
    return out


def neighbour_offsets(nd, corner=True, center=True):
    """Offsets of the elementary neighbourhood: the 3^n box (CornerNeighbors -> True) or the cross."""
    import itertools
    offs = []
    for o in itertools.product((-1, 0, 1), repeat=nd):
        if not center and not any(o):
            continue
        if not corner and sum(abs(t) for t in o) > 1:
            continue
        offs.append(o)
    return offs


def corner_opt(o, fname, default=True):
    v = o.get("CornerNeighbors", TRUE if default else FALSE)
    if _is_sym(v, "Automatic"):
        return default
    return _truthy(v, fname, "CornerNeighbors")


def se_offsets(ker, fname, nd):
    """Offsets of a flat structuring element: r (BoxMatrix[r]) or a 0/1 array of rank nd with odd dimensions."""
    if not K.is_list(ker):
        rad = radii_arg(ker, fname, nd)
        import itertools
        return list(itertools.product(*[range(-r, r + 1) for r in rad]))
    k = WI._numeric_array(ker, fname)
    if k.ndim != nd:
        raise ValueError(f"{fname}: the structuring element must have rank {nd} like the data")
    if not np.all((k == 0) | (k == 1)):
        raise NotImplementedError(f"{fname}: only structuring elements of 0s and 1s are supported")
    if any(s % 2 == 0 for s in k.shape):
        raise NotImplementedError(f"{fname}: structuring elements with an even dimension are not supported (where "
                                  "Wolfram pads them with zeros is not documented)")
    c = [s // 2 for s in k.shape]
    offs = [tuple(int(i) - ci for i, ci in zip(idx, c)) for idx in zip(*np.nonzero(k))]
    if not offs:
        raise ValueError(f"{fname}: the structuring element has no 1s")
    return offs


def dilate(a, offs, fill=-np.inf):
    """max over k of a(x + k) (Wolfram's convention, checked on the Dilation documentation example)."""
    out = None
    for k in offs:
        s = _shift(a, k, fill)
        out = s if out is None else np.maximum(out, s)
    return out


def erode(a, offs, fill=np.inf):
    """min over k of a(x + k)."""
    out = None
    for k in offs:
        s = _shift(a, k, fill)
        out = s if out is None else np.minimum(out, s)
    return out


def _neg(offs):
    return [tuple(-t for t in k) for k in offs]


class MorphData:
    """Numeric view of data for rank-based (morphological) operations: ``a`` is (spatial..., c) float; exact arrays
    are replaced by the ranks of their values (``vals`` maps ranks back), so results stay exact."""

    def __init__(self, x, fname, allow3d=True):
        d = data_arg(x, fname, allow3d=allow3d)
        self.d = d
        self.vals = None
        if d.is_image:
            self.a = d.img.a
        elif d.exact:
            import functools
            uniq = sorted(set(d.obj.reshape(-1).tolist()),
                          key=functools.cmp_to_key(lambda p, q: -1 if p < q else (1 if p > q else 0)))
            pos = {v: i for i, v in enumerate(uniq)}
            self.vals = uniq
            self.a = np.vectorize(lambda v: float(pos[v]), otypes=[float])(d.obj)[..., None]
        else:
            self.a = d.a[..., None]

    @property
    def nd(self):
        return self.a.ndim - 1

    def out(self, res, typ=None):
        """Expression of a (spatial..., c) result array of the same kind."""
        if self.d.is_image:
            return image_out(self.d.img, res, typ=typ)
        r = res[..., 0]
        if self.vals is not None:
            obj = np.empty(r.shape, dtype=object)
            for idx in np.ndindex(r.shape):
                obj[idx] = self.vals[int(r[idx])]
            return array_out(obj)
        return array_out(r)

    def values(self, res):
        """Object (exact) or float array of the original values of a rank/float result."""
        r = res[..., 0]
        if self.vals is None:
            return r
        obj = np.empty(r.shape, dtype=object)
        for idx in np.ndindex(r.shape):
            obj[idx] = self.vals[int(r[idx])]
        return obj

    def original(self):
        return self.d.obj if self.vals is not None else self.a[..., 0]

    def fills(self, o, fname, dil_default, ero_default):
        """(dilation fill, erosion fill) from the Padding option; data arrays ignore the outside by default."""
        if "Padding" in o:
            p = o["Padding"]
            if _is_sym(p, "None"):
                return -np.inf, np.inf
            if self.vals is not None:
                raise NotImplementedError(f"{fname}: Padding values for exact arrays are not supported")
            v = real_arg(p, fname, "Padding") if not isinstance(p, str) else None
            if v is None:
                raise NotImplementedError(f"{fname}: Padding {p!r} is not supported (give a number or None)")
            return v, v
        if self.d.is_image:
            return dil_default, ero_default
        return -np.inf, np.inf


def _per_c(a, f):
    return np.stack([f(a[..., k]) for k in range(a.shape[-1])], axis=-1)


def _morph_args(x, ker, opts, fname):
    _, o = opts_split(opts, {"Padding"}, fname)
    if _:
        raise TypeError(f"{fname}: unexpected extra arguments")
    md = MorphData(x, fname)
    offs = se_offsets(ker, fname, md.nd)
    fd, fe = md.fills(o, fname, 0.0, 1.0)
    return md, offs, fd, fe


_MORPH_NOTE = ("ker is a flat structuring element (an array of 0s and 1s with odd dimensions, rank of the data) or r "
               "for BoxMatrix[r]; data is an Image/Image3D (every channel separately) or a numeric array of any rank "
               "(exact input stays exact). ")
_MORPH_RAISE = ("Even-sized or non-0/1 kernels raise NotImplementedError; symbolic entries stay unevaluated. ")


@fn("Dilation", area=AREA, status="added",
    notes="Dilation[data, ker] gives max over the 1s k of ker of data(x + k) (Wolfram's orientation, checked on the "
          "documented asymmetric example); Dilation[data, r] uses BoxMatrix[r]. " + _MORPH_NOTE +
          "Padding: images 0 (the default), a number, or None; data arrays ignore the outside (smaller "
          "neighbourhoods). " + _MORPH_RAISE + "Check: reference (Wolfram documentation examples) in tests.")
def _dilation(x: Expr, ker: NonNegInteger | ListOf[Expr], *opts: Option) -> Expr:
    md, offs, fd, fe = _morph_args(x, ker, opts, "Dilation")
    return md.out(_per_c(md.a, lambda c: dilate(c, offs, fd)))


@fn("Erosion", area=AREA, status="added",
    notes="Erosion[data, ker] gives min over the 1s k of ker of data(x + k); Erosion[data, r] uses BoxMatrix[r]. " +
          _MORPH_NOTE + "Padding: images 1 (the default), a number, or None; data arrays ignore the outside. " +
          _MORPH_RAISE + "Check: reference (Wolfram documentation examples) in tests.")
def _erosion(x: Expr, ker: NonNegInteger | ListOf[Expr], *opts: Option) -> Expr:
    md, offs, fd, fe = _morph_args(x, ker, opts, "Erosion")
    return md.out(_per_c(md.a, lambda c: erode(c, offs, fe)))


def opening_arr(a, offs, fd, fe):
    return dilate(erode(a, offs, fe), _neg(offs), fd)


def closing_arr(a, offs, fd, fe):
    return erode(dilate(a, offs, fd), _neg(offs), fe)


@fn("Opening", area=AREA, status="added",
    notes="Opening[data, ker] is the erosion by ker followed by the dilation by the reflected ker (so the result never "
          "exceeds the data and opening twice changes nothing); Opening[data, r] uses BoxMatrix[r]. " + _MORPH_NOTE +
          "Padding as for Erosion/Dilation (images 1 then 0; arrays ignore the outside). " + _MORPH_RAISE +
          "Check: reference (Wolfram documentation example) and the identities Opening <= data, idempotence in "
          "tests.")
def _opening(x: Expr, ker: NonNegInteger | ListOf[Expr], *opts: Option) -> Expr:
    md, offs, fd, fe = _morph_args(x, ker, opts, "Opening")
    return md.out(_per_c(md.a, lambda c: opening_arr(c, offs, fd, fe)))


@fn("Closing", area=AREA, status="added",
    notes="Closing[data, ker] is the dilation by ker followed by the erosion by the reflected ker; Closing[data, r] "
          "uses BoxMatrix[r]. " + _MORPH_NOTE + "Padding as for Dilation/Erosion. " + _MORPH_RAISE +
          "Check: reference (Wolfram documentation example) and the identities Closing >= data, idempotence in "
          "tests.")
def _closing(x: Expr, ker: NonNegInteger | ListOf[Expr], *opts: Option) -> Expr:
    md, offs, fd, fe = _morph_args(x, ker, opts, "Closing")
    return md.out(_per_c(md.a, lambda c: closing_arr(c, offs, fd, fe)))


def _hat(x, ker, opts, fname, top):
    md, offs, fd, fe = _morph_args(x, ker, opts, fname)
    f = opening_arr if top else closing_arr
    res = _per_c(md.a, lambda c: f(c, offs, fd, fe))
    if md.d.is_image:
        diff = md.a - res if top else res - md.a
        return image_out(md.d.img, diff)
    orig = md.original()
    other = md.values(res)
    out = orig - other if top else other - orig
    return array_out(out if out.dtype == object else np.asarray(out, dtype=float))


@fn("TopHatTransform", area=AREA, status="added",
    notes="TopHatTransform[data, ker] = data - Opening[data, ker] (the documented definition), TopHatTransform[data, "
          "r] with BoxMatrix[r]. " + _MORPH_NOTE + _MORPH_RAISE + "Check: exact (the difference is formed "
          "exactly; data - TopHatTransform == Opening) in tests.")
def _tophat(x: Expr, ker: NonNegInteger | ListOf[Expr], *opts: Option) -> Expr:
    return _hat(x, ker, opts, "TopHatTransform", True)


@fn("BottomHatTransform", area=AREA, status="added",
    notes="BottomHatTransform[data, ker] = Closing[data, ker] - data (the documented definition), "
          "BottomHatTransform[data, r] with BoxMatrix[r]. " + _MORPH_NOTE + _MORPH_RAISE + "Check: exact (data + "
          "BottomHatTransform == Closing) in tests.")
def _bottomhat(x: Expr, ker: NonNegInteger | ListOf[Expr], *opts: Option) -> Expr:
    return _hat(x, ker, opts, "BottomHatTransform", False)


# ───────────────────────────── geodesic (reconstruction) operations ─────────────────────────────

def reconstruct(marker, mask, offs, dilation=True, nmax=None, fill=None):
    """Iterate m <- min(dilate(m), mask) (or max(erode(m), mask)) to the fixed point or nmax steps."""
    if fill is None:
        fill = -np.inf if dilation else np.inf
    m = marker
    it = 0
    while nmax is None or it < nmax:
        new = np.minimum(dilate(m, offs, fill), mask) if dilation else np.maximum(erode(m, offs, fill), mask)
        it += 1
        if np.array_equal(new, m):
            break
        m = new
    return m


def _geodesic(fname, marker, mask, rest, dilation):
    pos, o = opts_split(rest, {"CornerNeighbors", "Padding"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected {fname}[marker, mask, n, options]")
    nmax = None
    if pos:
        if _is_sym(pos[0], "Infinity") or (isinstance(pos[0], Expr) and pos[0].head == "DirectedInfinity"):
            nmax = None
        else:
            nmax = int_arg(pos[0], fname, "the number of iterations", 0)
    corner = corner_opt(o, fname)
    pad = o.get("Padding", Symbol("None"))
    if not _is_sym(pad, "None"):
        raise NotImplementedError(f"{fname}: only Padding -> None is supported")
    dk = data_arg(marker, fname)
    sym_mask = False
    try:
        dm = data_arg(mask, fname)
    except Unevaluated:
        if dk.is_image or not K.is_list(mask):
            raise
        dm = data_arg(mask, fname, symbolic=True)
        sym_mask = True
    if dk.is_image != dm.is_image:
        raise TypeError(f"{fname}: the marker and the mask must both be images or both arrays")
    if dk.shape() != dm.shape():
        raise ValueError(f"{fname}: the marker and the mask must have the same dimensions")
    offs = neighbour_offsets(len(dk.shape()), corner)
    if sym_mask:
        mk = dk.obj if dk.exact else dk.a
        md = MorphData(marker, fname)
        dil = (dilate if dilation else erode)(md.a[..., 0], offs)
        step = md.values(dil[..., None])
        out = np.empty(step.shape, dtype=object)
        head = "Min" if dilation else "Max"
        for idx in np.ndindex(step.shape):
            v = step[idx]
            v = R(v) if isinstance(v, sp.Basic) else v
            out[idx] = _ev(Expr(head, v, dm.obj[idx] if not isinstance(dm.obj[idx], sp.Basic) else R(dm.obj[idx])))
        del mk
        return array_out(out)
    if dk.is_image:
        a, b = dk.img.a, dm.img.a
        if a.shape[-1] != b.shape[-1]:
            if a.shape[-1] == 1:
                a = np.repeat(a, b.shape[-1], -1)
            elif b.shape[-1] == 1:
                b = np.repeat(b, a.shape[-1], -1)
            else:
                raise ValueError(f"{fname}: the marker and the mask must have the same number of channels")
        res = np.stack([reconstruct(a[..., k], b[..., k], offs, dilation, nmax) for k in range(a.shape[-1])], -1)
        img = dm.img if dm.img.c >= dk.img.c else dk.img
        return image_out(img, res, typ=WI.larger_type(dk.img.type, dm.img.type))
    # arrays: exact arrays go through common ranks
    if dk.exact or dm.exact:
        import functools
        vals = list(dk.obj.reshape(-1)) if dk.exact else [sp.Float(v) for v in dk.a.reshape(-1)]
        vals += list(dm.obj.reshape(-1)) if dm.exact else [sp.Float(v) for v in dm.a.reshape(-1)]
        uniq = sorted(set(vals), key=functools.cmp_to_key(lambda p, q: -1 if p < q else (1 if p > q else 0)))
        posn = {v: i for i, v in enumerate(uniq)}
        n = int(np.prod(dk.shape()))
        ra = np.array([posn[v] for v in vals[:n]], dtype=float).reshape(dk.shape())
        rb = np.array([posn[v] for v in vals[n:]], dtype=float).reshape(dk.shape())
        res = reconstruct(ra, rb, offs, dilation, nmax)
        out = np.empty(res.shape, dtype=object)
        for idx in np.ndindex(res.shape):
            out[idx] = uniq[int(res[idx])]
        return array_out(out)
    return array_out(reconstruct(dk.a, dm.a, offs, dilation, nmax))


_GEO_NOTE = ("CornerNeighbors -> True (default, the 3^n box) or False (the cross); Padding -> None (smaller "
             "neighbourhoods at the edges, the default and only supported setting); images per channel (a "
             "single-channel marker or mask is applied to every channel), arrays of any rank, exact arrays stay "
             "exact. A numeric marker with a symbolic mask does one step (as documented). ")


@fn("GeodesicDilation", area=AREA, status="added",
    notes="GeodesicDilation[marker, mask] iterates m -> Min[Dilation of m by the elementary neighbourhood, mask] from "
          "the marker to the fixed point (morphological reconstruction by dilation); GeodesicDilation[marker, mask, "
          "n] stops after at most n steps. " + _GEO_NOTE + "Mismatched dimensions raise ValueError. Check: reference "
          "(Wolfram documentation example, symbolic mask) and the fixed-point property in tests.")
def _geodilation(marker: Expr, mask: Expr, *rest: NonNegInteger | Option) -> Expr:
    return _geodesic("GeodesicDilation", marker, mask, rest, True)


@fn("GeodesicErosion", area=AREA, status="added",
    notes="GeodesicErosion[marker, mask] iterates m -> Max[Erosion of m by the elementary neighbourhood, mask] to the "
          "fixed point (reconstruction by erosion); GeodesicErosion[marker, mask, n] stops after at most n steps. " +
          _GEO_NOTE + "Check: reference (Wolfram documentation examples) in tests.")
def _geoerosion(marker: Expr, mask: Expr, *rest: NonNegInteger | Option) -> Expr:
    return _geodesic("GeodesicErosion", marker, mask, rest, False)


def _geo_oc(fname, x, ker, opts, opening):
    _, o = opts_split(opts, {"CornerNeighbors", "Padding"}, fname)
    if _:
        raise TypeError(f"{fname}: unexpected extra arguments")
    md = MorphData(x, fname)
    offs = se_offsets(ker, fname, md.nd)
    corner = corner_opt(o, fname)
    if "Padding" in o and not _is_sym(o["Padding"], "None"):
        raise NotImplementedError(f"{fname}: only Padding -> None is supported")
    eo = neighbour_offsets(md.nd, corner)

    def one(c):
        if opening:
            return reconstruct(erode(c, offs), c, eo, True)
        return reconstruct(dilate(c, offs), c, eo, False)
    return md.out(_per_c(md.a, one))


@fn("GeodesicOpening", area=AREA, status="added",
    notes="GeodesicOpening[data, ker] (opening by reconstruction) erodes data by ker and reconstructs the result by "
          "geodesic dilation under data; GeodesicOpening[data, r] uses BoxMatrix[r]. " + _MORPH_NOTE + _GEO_NOTE +
          _MORPH_RAISE + "Check: reference (Wolfram documentation examples) and Opening <= GeodesicOpening <= data in "
          "tests.")
def _geoopening(x: Expr, ker: NonNegInteger | ListOf[Expr], *opts: Option) -> Expr:
    return _geo_oc("GeodesicOpening", x, ker, opts, True)


@fn("GeodesicClosing", area=AREA, status="added",
    notes="GeodesicClosing[data, ker] (closing by reconstruction) dilates data by ker and reconstructs the result by "
          "geodesic erosion above data; GeodesicClosing[data, r] uses BoxMatrix[r]. " + _MORPH_NOTE + _GEO_NOTE +
          _MORPH_RAISE + "Check: reference (Wolfram documentation matrix examples) and data <= GeodesicClosing <= "
          "Closing in tests.")
def _geoclosing(x: Expr, ker: NonNegInteger | ListOf[Expr], *opts: Option) -> Expr:
    return _geo_oc("GeodesicClosing", x, ker, opts, False)


# ───────────────────────────── extended maxima / minima, filling, hit-or-miss ─────────────────────────────

def plateau_labels(g, offs):
    """Labels of the connected plateaus (equal values) of g: the smallest flat index + 1 in each plateau."""
    big = np.iinfo(np.int64).max
    lab = np.arange(1, g.size + 1, dtype=np.int64).reshape(g.shape)
    while True:
        new = lab.copy()
        for k in offs:
            sl = _shift(lab, k, big)
            sg = _shift(g, k, np.nan)
            new = np.where(sg == g, np.minimum(new, sl), new)
        if np.array_equal(new, lab):
            return lab
        lab = new


def regional_extrema(g, offs, maxima=True):
    """Boolean mask of the regional maxima (or minima) of g: plateaus with no strictly higher (lower) neighbour."""
    lab = plateau_labels(g, offs)
    bad = np.zeros(g.shape, dtype=bool)
    for k in offs:
        sg = _shift(g, k, np.nan)
        bad |= (sg > g) if maxima else (sg < g)
    badlabs = np.unique(lab[bad])
    return ~np.isin(lab, badlabs)


def _detect(fname, x, rest, maxima):
    pos, o = opts_split(rest, {"CornerNeighbors", "Padding"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected {fname}[data, h, options]")
    h = 0.0
    if pos:
        if isinstance(pos[0], Symbol):
            raise Unevaluated()
        h = real_arg(pos[0], fname, "h")
        if h < 0:
            raise ValueError(f"{fname}: h must be non-negative")
    if "Padding" in o and not _is_sym(o["Padding"], "Automatic"):
        raise NotImplementedError(f"{fname}: only Padding -> Automatic is supported")
    corner = corner_opt(o, fname)
    d = data_arg(x, fname)
    if d.is_image:
        g = intensity(d.img) if d.img.c > 1 else d.img.a[..., 0]
    elif d.exact and h == 0:
        g = MorphData(x, fname).a[..., 0]
    elif d.exact:
        g = np.vectorize(float, otypes=[float])(d.obj)
    else:
        g = d.a
    offs = neighbour_offsets(g.ndim, corner, center=False)
    if h > 0:
        eo = neighbour_offsets(g.ndim, corner)
        g = reconstruct(g - h, g, eo, True) if maxima else reconstruct(g + h, g, eo, False)
    return binary_like(d, regional_extrema(g, offs, maxima))


@fn("MaxDetect", area=AREA, status="added",
    notes="MaxDetect[data] marks the regional (constant extended) maxima: connected plateaus with no strictly higher "
          "neighbour; MaxDetect[data, h] first chops maxima of height h or less (the h-maxima transform, "
          "reconstruction by dilation of data - h under data). Images (multichannel: the mean intensity) give a "
          "\"Bit\" image, arrays a 0/1 array. CornerNeighbors -> True (default) or False; Padding -> Automatic only "
          "(boundary maxima are detected, as documented). Negative h raises ValueError. Check: reference (Wolfram "
          "documentation examples) in tests.")
def _maxdetect(x: Expr, *rest: Real | Option) -> Expr:
    return _detect("MaxDetect", x, rest, True)


@fn("MinDetect", area=AREA, status="added",
    notes="MinDetect[data] marks the regional (constant extended) minima; MinDetect[data, h] first fills minima of "
          "depth h or less (the h-minima transform). Same forms and options as MaxDetect. Check: reference (Wolfram "
          "documentation examples; MinDetect[x] == MaxDetect[-x]) in tests.")
def _mindetect(x: Expr, *rest: Real | Option) -> Expr:
    return _detect("MinDetect", x, rest, False)


@fn("FillingTransform", area=AREA, status="added",
    notes="FillingTransform[data] fills every extended minimum not connected to the border, lifting it to the lowest "
          "surrounding value: the reconstruction by erosion of a marker equal to the padding value outside and the "
          "maximum inside (binary images: hole filling). FillingTransform[data, h] lifts minima by at most h (marker "
          "data + h; with Padding -> 1 this is the h-minima transform, as documented). FillingTransform[data, "
          "marker] fills only the filled regions that meet a nonzero marker element. Padding -> 0 (default), a "
          "number, or None (the border does not anchor); CornerNeighbors -> True (default) or False. Images per "
          "channel (type kept), arrays of any rank. Check: reference (hand-worked hole filling) and the identity "
          "FillingTransform[FillingTransform[x]] == FillingTransform[x] in tests.")
def _fillingtransform(x: Expr, *rest: Real | Expr | Option) -> Expr:
    fname = "FillingTransform"
    pos, o = opts_split(rest, {"CornerNeighbors", "Padding"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected FillingTransform[data, h or marker, options]")
    corner = corner_opt(o, fname)
    d = data_arg(x, fname)
    pad = o.get("Padding", 0)
    if _is_sym(pad, "None"):
        pad = None
    elif not is_real_like(pad):
        raise NotImplementedError(f"{fname}: Padding {_short(pad)} is not supported (give a number or None)")
    h, marker = None, None
    if pos:
        p = pos[0]
        if K.is_list(p) or (isinstance(p, Expr) and p.head in IMAGE_HEADS):
            dm = data_arg(p, fname, exact_ok=False)
            marker = (intensity(dm.img) if dm.is_image else dm.a) != 0
            if marker.shape != d.shape():
                raise ValueError(f"{fname}: the marker must have the dimensions of the data")
        else:
            if isinstance(p, Symbol):
                raise Unevaluated()
            h = p
            if real_arg(h, fname, "h") < 0:
                raise ValueError(f"{fname}: h must be non-negative")
    nd = len(d.shape())
    eo = neighbour_offsets(nd, corner)
    sl_in = tuple(slice(1, -1) for _ in range(nd))

    def fill(c, padv, hv):
        """c: values (float ranks or reals); padv: padding value or None; hv: function c -> c + h or None."""
        if padv is not None:
            P = np.pad(c, 1, mode="constant", constant_values=padv)
            inner = np.zeros(P.shape, dtype=bool)
            inner[sl_in] = True
        else:
            P, inner = c, np.ones(c.shape, dtype=bool)
        top = hv(P) if hv is not None else np.full(P.shape, P.max())
        r = reconstruct(np.where(inner, top, P), P, eo, False)
        if padv is not None:
            r = r[sl_in]
        if marker is not None:
            changed = r != c
            lab = label(changed, corner)
            keep = np.unique(lab[changed & marker])
            r = np.where(np.isin(lab, keep[keep > 0]), r, c)
        return r

    if d.is_image or not d.exact:
        a = d.a if d.is_image else d.a[..., None]
        padv = None if pad is None else real_arg(pad, fname, "Padding")
        hf = None if h is None else (lambda P, hh=real_arg(h, fname, "h"): P + hh)
        res = _per_c(a, lambda c: fill(c, padv, hf))
        return image_out(d.img, res) if d.is_image else array_out(res[..., 0])
    # exact arrays: work on the ranks of every value that can occur (data, padding, data + h)
    import functools
    hx = None if h is None else S(h)
    vals = set(d.obj.reshape(-1).tolist())
    if pad is not None:
        vals.add(S(pad))
    if hx is not None:
        vals |= {v + hx for v in list(d.obj.reshape(-1))} | ({S(pad) + hx} if pad is not None else set())
    uniq = sorted(vals, key=functools.cmp_to_key(lambda p_, q_: -1 if p_ < q_ else (1 if p_ > q_ else 0)))
    rank = {v: i for i, v in enumerate(uniq)}
    ra = np.vectorize(lambda v: float(rank[v]), otypes=[float])(d.obj)
    plus = {i: float(rank[v + hx]) for i, v in enumerate(uniq) if hx is not None and (v + hx) in rank}
    hf = None if hx is None else (lambda P: np.vectorize(lambda t: plus[int(t)], otypes=[float])(P))
    res = fill(ra, None if pad is None else float(rank[S(pad)]), hf)
    out = np.empty(res.shape, dtype=object)
    for idx in np.ndindex(res.shape):
        out[idx] = uniq[int(res[idx])]
    return array_out(out)


def _hm_kernel(k, fname):
    a = WI._numeric_array(k, fname)
    if a is None:
        raise TypeError(f"{fname}: the structuring element must be a matrix of 1, -1 and 0")
    if not np.all(np.isin(a, (-1, 0, 1))):
        raise ValueError(f"{fname}: the structuring element may only contain 1 (foreground), -1 (background) and 0 "
                         "(don't care)")
    if any(s % 2 == 0 for s in a.shape):
        raise NotImplementedError(f"{fname}: structuring elements with an even dimension are not supported (where "
                                  "Wolfram pads them with zeros is not documented)")
    return a


@fn("HitMissTransform", area=AREA, status="added",
    notes="HitMissTransform[image, ker] marks the pixels x where every 1 of ker lies on foreground (value > t) and "
          "every -1 on background, at positions x + k (0 = don't care); HitMissTransform[image, {ker1, ...}] is the "
          "union; HitMissTransform[image, kers, t] uses threshold t (default 0: nonzero is foreground; multichannel "
          "images use the mean intensity). Padding -> 1 (default: outside is foreground) or a number. Gives a \"Bit\" "
          "image (0/1 arrays for array input). Kernels with an even dimension raise NotImplementedError, other "
          "entries than 1, -1, 0 ValueError. Check: reference (hand-worked endpoint and corner detection) in tests.")
def _hitmiss(x: Expr, kers: ListOf[Expr], *rest: Real | Option) -> Expr:
    fname = "HitMissTransform"
    pos, o = opts_split(rest, {"Padding"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected HitMissTransform[image, ker, t, options]")
    t = real_arg(pos[0], fname, "threshold") if pos else 0.0
    d = data_arg(x, fname, exact_ok=False)
    g = (intensity(d.img) if d.is_image else d.a)
    fg = g > t
    padv = real_arg(o.get("Padding", 1), fname, "Padding") > t
    nd = fg.ndim
    if not K.is_list(kers):
        raise TypeError(f"{fname}: the structuring element must be a matrix")
    first = kers.args[0] if kers.args else None
    depth = 0
    e = kers
    while K.is_list(e) and e.args:
        depth += 1
        e = e.args[0]
    if depth == nd + 1:
        klist = [_hm_kernel(k, fname) for k in kers.args]
    elif depth == nd:
        klist = [_hm_kernel(kers, fname)]
    else:
        raise ValueError(f"{fname}: the structuring element must have the rank of the image")
    del first
    out = np.zeros(fg.shape, dtype=bool)
    for k in klist:
        c = [s // 2 for s in k.shape]
        hit = np.ones(fg.shape, dtype=bool)
        for idx in zip(*np.nonzero(k)):
            off = tuple(int(i) - ci for i, ci in zip(idx, c))
            s = _shift(fg, off, padv)
            hit &= s if k[idx] == 1 else ~s
        out |= hit
    return binary_like(d, out)


# ───────────────────────────── connected components ─────────────────────────────

def label(mask, corner=True):
    """Connected-component labels 1..n of a boolean array, numbered in scan (row-major) order of first elements."""
    offs = neighbour_offsets(mask.ndim, corner, center=False)
    big = np.iinfo(np.int64).max
    lab = np.where(mask, np.arange(1, mask.size + 1, dtype=np.int64).reshape(mask.shape), big)
    while True:
        new = lab
        for k in offs:
            new = np.minimum(new, _shift(lab, k, big))
        new = np.where(mask, new, big)
        if np.array_equal(new, lab):
            break
        lab = new
    out = np.zeros(mask.shape, dtype=np.int64)
    if mask.any():
        u, inv = np.unique(lab[mask], return_inverse=True)
        out[mask] = inv.reshape(-1) + 1
    return out


def _fg_mask(d, t, fname):
    if d.is_image:
        g = intensity(d.img) if d.img.c > 1 else d.img.a[..., 0]
    elif d.exact:
        g = np.vectorize(float, otypes=[float])(d.obj)
    else:
        g = d.a
    return g > t


def _t_arg(pos, fname, i=0):
    if len(pos) <= i:
        return 0.0
    if isinstance(pos[i], Symbol):
        raise Unevaluated()
    return real_arg(pos[i], fname, "threshold")


@fn("MorphologicalComponents", area=AREA, status="added",
    notes="MorphologicalComponents[data] labels the connected components of the foreground (values > 0; > t for "
          "MorphologicalComponents[data, t]) with 1, 2, ... in scan order of their first element (rows from the "
          "top), background 0, as documented; images use the mean intensity of multichannel pixels; arrays of any "
          "rank. CornerNeighbors -> True (default) or False; Method -> \"Connected\" (default). The other methods "
          "(\"Nested\", \"Convex\", \"BoundingBox\", ...) raise NotImplementedError. Check: reference (Wolfram "
          "documentation examples) in tests.")
def _morphcomponents(x: Expr, *rest: Real | Option) -> Expr:
    fname = "MorphologicalComponents"
    pos, o = opts_split(rest, {"CornerNeighbors", "Method", "Padding"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected MorphologicalComponents[data, t, options]")
    m = o.get("Method", "Connected")
    if not (m == "Connected" or _is_sym(m, "Automatic")):
        raise NotImplementedError(f"{fname}: Method {_short(m)} is not supported (only \"Connected\")")
    if "Padding" in o and not (o["Padding"] == 0 or _is_sym(o["Padding"], "None")):
        raise NotImplementedError(f"{fname}: only Padding -> 0 is supported")
    t = _t_arg(pos, fname)
    d = data_arg(x, fname)
    return int_nest(label(_fg_mask(d, t, fname), corner_opt(o, fname)))


def label_input(x, fname, corner, allow_symbolic=False):
    """(labels, data) for an image (components of its nonzero pixels) or a label matrix of non-negative integers."""
    d = data_arg(x, fname)
    if d.is_image:
        return label(_fg_mask(d, 0.0, fname), corner), d
    vals = d.obj if d.exact else d.a
    flat = vals.reshape(-1)
    if not all((isinstance(v, sp.Integer) or (isinstance(v, (float, np.floating)) and float(v).is_integer()))
               and v >= 0 for v in flat):
        raise ValueError(f"{fname}: a label matrix must contain non-negative integers")
    return np.vectorize(int, otypes=[np.int64])(vals), d


def _labels_out(d, lab, keep):
    """Result with components not in keep removed: an image (pixels zeroed) or a label matrix."""
    sel = np.isin(lab, list(keep)) & (lab > 0)
    if d.is_image:
        img = d.img
        return image_out(img, img.a * sel[..., None])
    return int_nest(np.where(sel, lab, 0))


@fn("DeleteBorderComponents", area=AREA, status="added",
    notes="DeleteBorderComponents[image] removes (sets to background) the connected components of the nonzero pixels "
          "that touch the border; DeleteBorderComponents[m] replaces labels of a label matrix that occur on the "
          "border with 0. CornerNeighbors -> True (default) or False for images. Arrays must hold non-negative "
          "integers (ValueError otherwise). Check: reference (Wolfram documentation example) in tests.")
def _deleteborder(x: Expr, *opts: Option) -> Expr:
    fname = "DeleteBorderComponents"
    _, o = opts_split(opts, {"CornerNeighbors"}, fname)
    if _:
        raise TypeError(f"{fname}: expected DeleteBorderComponents[image, options]")
    lab, d = label_input(x, fname, corner_opt(o, fname))
    border = np.zeros(lab.shape, dtype=bool)
    for ax in range(lab.ndim):
        sl = [slice(None)] * lab.ndim
        sl[ax] = 0
        border[tuple(sl)] = True
        sl[ax] = -1
        border[tuple(sl)] = True
    drop = set(np.unique(lab[border]).tolist())
    keep = set(np.unique(lab).tolist()) - drop - {0}
    return _labels_out(d, lab, keep)


def otsu_index(hist):
    """Lowest bin k maximising the between-class variance of a histogram (classes 0..k and k+1..)."""
    p = hist / hist.sum()
    idx = np.arange(len(p))
    w0 = np.cumsum(p)
    m = np.cumsum(p * idx)
    mt = m[-1]
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w0 - m) ** 2 / (w0 * w1)
    sb = np.where((w0 > 0) & (w1 > 1e-15), sb, -1.0)
    sb = sb[:-1]
    best = sb.max()
    return int(np.nonzero(sb >= best - 1e-12 * max(1.0, best))[0][0])


def threshold_values(v, method, fname, image_levels):
    """Global threshold of the 1-D float array v by method ("Cluster", "Mean", "Median", ("BlackFraction", b)).
    image_levels: values are on the 0..1 scale and are binned in the 256 levels k/255; otherwise 256 equal bins over
    [Min, Max] (threshold Min + k (Max - Min)/256, the documented FindThreshold[{1, 2, 3, 10, 11}] value)."""
    if isinstance(method, tuple):
        b = method[1]
        s = np.sort(v)
        k = int(math.ceil(b * len(s))) - 1
        return float(s[max(0, min(len(s) - 1, k))])
    if method == "Mean":
        return float(v.mean())
    if method == "Median":
        return float(np.median(v))
    if image_levels:
        lev = np.clip(np.rint(v * 255), 0, 255).astype(int)
        hist = np.bincount(lev, minlength=256).astype(float)
        if (hist > 0).sum() < 2:
            return float(lev[0]) / 255
        return otsu_index(hist) / 255
    lo, hi = float(v.min()), float(v.max())
    if hi == lo:
        return lo
    b = np.clip(np.floor((v - lo) / (hi - lo) * 256), 0, 255).astype(int)
    hist = np.bincount(b, minlength=256).astype(float)
    return lo + otsu_index(hist) * (hi - lo) / 256


def method_arg(o, fname, default="Cluster"):
    m = o.get("Method", default)
    if _is_sym(m, "Automatic"):
        return default
    if isinstance(m, str):
        if m in ("Cluster", "Mean", "Median"):
            return m
        if m in ("Entropy", "MinimumError"):
            raise NotImplementedError(f"{fname}: Method {m!r} is not supported (its histogram binning is not "
                                      "documented)")
        raise ValueError(f"{fname}: unknown Method {m!r}")
    if K.is_list(m) and len(m.args) == 2 and m.args[0] == "BlackFraction":
        b = real_arg(m.args[1], fname, "BlackFraction")
        if not 0 < b < 1:
            raise ValueError(f"{fname}: the BlackFraction must be between 0 and 1")
        return ("BlackFraction", b)
    raise NotImplementedError(f"{fname}: Method {_short(m)} is not supported")


def gray_values(d, fname):
    """(array of gray values, image_levels flag): images converted to grayscale (alpha dropped), arrays as reals."""
    if d.is_image:
        img = d.img
        if img.c == 1 and not img.alpha:
            g = img.a[..., 0]
        elif img.cs in ("RGB", "Grayscale") or img.cs == "Automatic":
            col = img.color()
            if col.shape[-1] == 3 and img.cs == "RGB":
                g = col @ np.array([0.299, 0.587, 0.114])
            else:
                g = col.mean(axis=-1)
        else:
            g = WI.to_space(img, "Grayscale", fname).color()[..., 0]
        return g, True
    if d.exact:
        return np.vectorize(float, otypes=[float])(d.obj), False
    return d.a, False


@fn("FindThreshold", area=AREA, status="added",
    notes="FindThreshold[data] gives a global threshold by Otsu's cluster variance maximisation (Method -> "
          "\"Cluster\", the default) on a 256-bin histogram: images use the levels k/255 (grayscale: luminance "
          "0.299 R + 0.587 G + 0.114 B; other multichannel images the channel mean) and give k/255; arrays use 256 "
          "equal bins over [Min, Max] and give Min + k (Max - Min)/256, k the lowest maximising bin (matching the "
          "documented FindThreshold[{1, 2, 3, 10, 11}] = 2.99219). Method -> \"Mean\", \"Median\" or "
          "{\"BlackFraction\", b}. \"Entropy\" and \"MinimumError\" raise NotImplementedError (binning not "
          "documented). Check: reference (Wolfram documentation example) and a direct between-class variance scan "
          "in tests.")
def _findthreshold(x: Expr, *opts: Option) -> Real:
    fname = "FindThreshold"
    _, o = opts_split(opts, {"Method"}, fname)
    if _:
        raise TypeError(f"{fname}: expected FindThreshold[data, options]")
    d = data_arg(x, fname)
    g, lev = gray_values(d, fname)
    return threshold_values(g.reshape(-1), method_arg(o, fname), fname, lev)


@fn("Binarize", area=AREA, status="added",
    notes="Binarize[image] gives a \"Bit\" image with 1 where the grayscale value exceeds FindThreshold[image] "
          "(Method option as for FindThreshold); Binarize[image, t] uses t; Binarize[image, {t1, t2}] marks values "
          "with t1 <= v <= t2; Binarize[image, f] marks pixels for which f[v] is True (v the value, or the channel "
          "list of a multichannel pixel). Multichannel images are converted to grayscale first (luminance for RGB). "
          "Arrays raise TypeError (Binarize works on images); f returning a non-Boolean raises ValueError. Check: "
          "reference (direct comparisons with the threshold) in tests.")
def _binarize(x: Expr, *rest: Real | ListOf[Real] | Fn | Option) -> Expr:
    fname = "Binarize"
    pos, o = opts_split(rest, {"Method"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected Binarize[image, t, options]")
    d = data_arg(x, fname)
    if not d.is_image:
        raise TypeError(f"{fname}: expected an image, not an array")
    g, lev = gray_values(d, fname)
    if not pos:
        return binary_image(d.img, g > threshold_values(g.reshape(-1), method_arg(o, fname), fname, lev))
    t = pos[0]
    if K.is_list(t):
        if len(t.args) != 2:
            raise ValueError(f"{fname}: the range must be {{t1, t2}}")
        t1, t2 = (real_arg(v, fname, "threshold") for v in t.args)
        return binary_image(d.img, (g >= t1 - 1e-12) & (g <= t2 + 1e-12))
    if is_real_like(t):
        return binary_image(d.img, g > real_arg(t, fname, "threshold"))
    if isinstance(t, Symbol) and not _is_fn_symbol(t):
        raise Unevaluated()
    img = d.img
    out = np.zeros(g.shape, dtype=bool)
    cache = {}
    for idx in np.ndindex(g.shape):
        px = tuple(img.a[idx])
        if px not in cache:
            arg = float(px[0]) if img.c == 1 else L(*[float(v) for v in px])
            r = _ev(K.call(t, arg))
            if r is True or _is_sym(r, "True"):
                cache[px] = True
            elif r is False or _is_sym(r, "False"):
                cache[px] = False
            else:
                raise ValueError(f"{fname}: the test function must return True or False, not {_short(r)}")
        out[idx] = cache[px]
    return binary_image(img, out)


def is_real_like(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)) or WI.is_real_value(v)


def _is_fn_symbol(s):
    from .core import REGISTRY
    return s.name in REGISTRY or s.name[:1].isupper()


@fn("LocalAdaptiveBinarize", area=AREA, status="added",
    notes="LocalAdaptiveBinarize[image, r] marks pixels whose grayscale value exceeds the mean of their (2r + 1)-block; "
          "LocalAdaptiveBinarize[image, r, {a, b, c}] uses the threshold a mean + b sd + c (sd the population "
          "standard deviation of the block); {r1, r2} radii per direction. Padding -> \"Fixed\" (default), "
          "\"Periodic\", \"Reflected\", \"Reversed\" or a value for the block statistics. Gives a \"Bit\" image. "
          "Check: reference (direct block means) in tests.")
def _localadaptive(x: Expr, r: NonNegInteger | ListOf[NonNegInteger], *rest: ListOf[Real] | Option) -> Expr:
    fname = "LocalAdaptiveBinarize"
    pos, o = opts_split(rest, {"Padding", "PerformanceGoal"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected LocalAdaptiveBinarize[image, r, {{a, b, c}}, options]")
    d = data_arg(x, fname)
    if not d.is_image:
        raise TypeError(f"{fname}: expected an image")
    g, _lev = gray_values(d, fname)
    nd = g.ndim
    rad = radii_arg(r, fname, nd)
    a, b, c = 1.0, 0.0, 0.0
    if pos:
        if not (K.is_list(pos[0]) and len(pos[0].args) == 3):
            raise ValueError(f"{fname}: the coefficients must be {{a, b, c}}")
        a, b, c = (real_arg(v, fname, "coefficient") for v in pos[0].args)
    pad = o.get("Padding", "Fixed")
    from numpy.lib.stride_tricks import sliding_window_view
    P = pad_array(g[..., None], [(q, q) for q in rad], pad, fname)[..., 0]
    W = sliding_window_view(P, tuple(2 * q + 1 for q in rad)).reshape(g.shape + (-1,))
    mu = W.mean(-1)
    sd = W.std(-1)
    return binary_image(d.img, g > a * mu + b * sd + c)


@fn("MorphologicalBinarize", area=AREA, status="added",
    notes="MorphologicalBinarize[image, {t1, t2}] (hysteresis thresholding) marks the grayscale values above t2 and "
          "every value above t1 connected to them (CornerNeighbors -> True by default); {t1, f} uses t2 = f[t1]. "
          "The forms with an automatically chosen threshold (MorphologicalBinarize[image], [image, t], {t}, {f}) "
          "raise NotImplementedError (Wolfram's rule for the automatic lower threshold is not documented). Gives a "
          "\"Bit\" image. Check: reference (hand-worked hysteresis) in tests.")
def _morphbinarize(x: Expr, *rest: ListOf[Real] | Real | Option) -> Expr:
    fname = "MorphologicalBinarize"
    pos, o = opts_split(rest, {"CornerNeighbors", "Method"}, fname)
    d = data_arg(x, fname)
    if not d.is_image:
        raise TypeError(f"{fname}: expected an image")
    if len(pos) != 1 or not K.is_list(pos[0]) or len(pos[0].args) != 2:
        raise NotImplementedError(f"{fname}: automatic thresholds are not supported; give {{t1, t2}}")
    t1 = real_arg(pos[0].args[0], fname, "threshold")
    t2e = pos[0].args[1]
    if is_real_like(t2e):
        t2 = real_arg(t2e, fname, "threshold")
    else:
        t2 = real_arg(_ev(K.call(t2e, t1)), fname, "threshold")
    if t1 > t2:
        raise ValueError(f"{fname}: the lower threshold must not exceed the upper one")
    g, _lev = gray_values(d, fname)
    low, high = g > t1, g > t2
    lab = label(low, corner_opt(o, fname))
    keep = np.unique(lab[high])
    return binary_image(d.img, np.isin(lab, keep[keep > 0]))


# Threshold

def _thr_apply(v, kind, delta, pars):
    av = np.abs(v)
    if kind == "Hard":
        return np.where(av <= delta, 0.0, v)
    if kind == "Soft":
        return np.where(av <= delta, 0.0, np.sign(v) * (av - delta))
    if kind == "PiecewiseGarrote":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(av <= delta, 0.0, v - delta ** 2 / np.where(v == 0, 1, v))
    if kind == "Hyperbola":
        return np.where(av <= delta, 0.0, np.sign(v) * np.sqrt(np.maximum(v * v - delta * delta, 0)))
    if kind == "SmoothGarrote":
        n = pars[0]
        den = v ** (2 * n) + delta ** (2 * n)
        return np.where(den == 0, 0.0, v ** (2 * n + 1) / np.where(den == 0, 1, den))
    if kind == "Firm":
        rr, p = pars
        lo, hi = delta * (1 - p * rr), delta * (1 + rr - p * rr)
        mid = np.sign(v) * hi * (av - lo) / (hi - lo)
        return np.where(av <= lo, 0.0, np.where(av <= hi, mid, v))
    raise AssertionError(kind)


def _thr_exact(v, kind, delta, pars):
    av = abs(v)
    if kind == "Hard":
        return sp.Integer(0) if av <= delta else v
    if kind == "Soft":
        return sp.Integer(0) if av <= delta else sp.sign(v) * (av - delta)
    if kind == "PiecewiseGarrote":
        return sp.Integer(0) if av <= delta else v - delta ** 2 / v
    if kind == "Hyperbola":
        return sp.Integer(0) if av <= delta else sp.sign(v) * sp.sqrt(v ** 2 - delta ** 2)
    if kind == "SmoothGarrote":
        n = pars[0]
        den = v ** (2 * n) + delta ** (2 * n)
        return sp.Integer(0) if den == 0 else v ** (2 * n + 1) / den
    if kind == "Firm":
        rr, p = pars
        lo, hi = delta * (1 - p * rr), delta * (1 + rr - p * rr)
        if av <= lo:
            return sp.Integer(0)
        return sp.sign(v) * hi * (av - lo) / (hi - lo) if av <= hi else v
    raise AssertionError(kind)


_THR_KINDS = ("Hard", "Soft", "Firm", "PiecewiseGarrote", "SmoothGarrote", "Hyperbola", "LargestValues")


@fn("Threshold", area=AREA, status="added",
    notes="Threshold[data] = Threshold[data, {\"Hard\", 10^-10}]; Threshold[data, δ] and {tfun, δ, ...} with the "
          "documented functions: \"Hard\" (0 for |x| <= δ), \"Soft\" (sgn(x)(|x| - δ)), \"Firm\" {δ, r, p}, "
          "\"PiecewiseGarrote\" (x - δ^2/x), \"SmoothGarrote\" {δ, n} (x^(2n+1)/(x^(2n) + δ^(2n)), n a positive "
          "integer), \"Hyperbola\" (sgn(x) Sqrt[x^2 - δ^2]); {\"LargestValues\", k} keeps the k largest absolute "
          "values (ties: the first in scan order). δ may be a function applied to the data (Abs of the values) or a "
          "FindThreshold method (\"Cluster\", \"Mean\", \"Median\", {\"BlackFraction\", b}) applied to the absolute "
          "values. Exact data stay exact; images are thresholded per channel. Unknown names raise ValueError. "
          "Check: reference (Wolfram documentation examples) in tests.")
def _threshold(x: Expr, spec: Real | ListOf[Expr] | String = None) -> Expr:
    fname = "Threshold"
    d = data_arg(x, fname)
    if spec is None:
        kind, dspec, pars = "Hard", sp.Rational(1, 10 ** 10), []
    elif K.is_list(spec) and spec.args and isinstance(spec.args[0], str):
        kind = spec.args[0]
        if kind not in _THR_KINDS:
            raise ValueError(f"{fname}: unknown thresholding function {kind!r}")
        if len(spec.args) < 2:
            raise ValueError(f"{fname}: {kind} needs a threshold parameter")
        dspec, pars = spec.args[1], list(spec.args[2:])
    elif isinstance(spec, str):
        kind, dspec, pars = "Hard", spec, []
    else:
        kind, dspec, pars = "Hard", spec, []
    vals = d.a if (d.is_image or not d.exact) else None
    absvals = np.abs(vals if vals is not None else np.vectorize(float, otypes=[float])(d.obj))
    if kind == "LargestValues":
        k = int_arg(dspec, fname, "k", 0)
        if pars:
            raise ValueError(f"{fname}: LargestValues takes one parameter")
        flat = absvals.reshape(-1)
        order = np.argsort(-flat, kind="stable")
        keep = np.zeros(flat.shape, dtype=bool)
        keep[order[:k]] = True
        keep = keep.reshape(absvals.shape)
        if d.is_image:
            return image_out(d.img, np.where(keep, d.a, 0.0))
        if d.exact:
            out = np.where(keep, d.obj, sp.Integer(0))
            return array_out(out.astype(object))
        return array_out(np.where(keep, d.a, 0.0))
    # the threshold δ
    if isinstance(dspec, str) or (K.is_list(dspec) and dspec.args and dspec.args[0] == "BlackFraction"):
        delta = threshold_values(absvals.reshape(-1), method_arg({"Method": dspec}, fname), fname, d.is_image)
        delta_ex = None
    elif is_real_like(dspec):
        delta_ex = S(dspec) if not isinstance(dspec, float) else None
        delta = real_arg(dspec, fname, "threshold")
    elif isinstance(dspec, Symbol) and not _is_fn_symbol(dspec):
        raise Unevaluated()
    else:
        arg = array_out(absvals) if not d.exact else array_out(np.vectorize(abs, otypes=[object])(d.obj))
        dv = _ev(K.call(dspec, arg))
        delta_ex = S(dv) if not isinstance(dv, float) and is_real_like(dv) else None
        delta = real_arg(dv, fname, "threshold")
    if delta < 0:
        raise ValueError(f"{fname}: the threshold must be non-negative")
    if kind == "SmoothGarrote":
        n = int_arg(pars[0], fname, "n", 1) if pars else 1
        pars = [n]
    elif kind == "Firm":
        if len(pars) != 2:
            raise ValueError(f"{fname}: Firm needs {{\"Firm\", δ, r, p}}")
        rr, p = (real_arg(v, fname, "parameter") for v in pars)
        if not rr > 0 or not 0 <= p <= 1:
            raise ValueError(f"{fname}: Firm needs r > 0 and 0 <= p <= 1")
        pars = [S(pars[0]) if d.exact else rr, S(pars[1]) if d.exact else p]
    elif pars:
        raise ValueError(f"{fname}: {kind} takes one parameter")
    if d.is_image:
        out = _thr_apply(d.a, kind, delta, pars)
        return image_out(d.img, out)
    if d.exact and delta_ex is not None:
        out = np.vectorize(lambda v: _thr_exact(v, kind, delta_ex, pars), otypes=[object])(d.obj)
        return array_out(out)
    a = d.a if not d.exact else np.vectorize(float, otypes=[float])(d.obj)
    return array_out(_thr_apply(a, kind, delta, pars))


# ───────────────────────────── distance transforms ─────────────────────────────

def _edt_sq(fg):
    """Exact squared Euclidean distance of every element of the boolean array fg to the nearest False element
    (separable: per-axis lower envelopes computed by brute-force minimisation); inf if there is none."""
    INF = np.inf
    d = np.where(fg, INF, 0.0)
    for ax in range(fg.ndim):
        n = fg.shape[ax]
        idx = np.arange(n, dtype=float)
        D2 = (idx[:, None] - idx[None, :]) ** 2            # (target, source)
        moved = np.moveaxis(d, ax, -1)
        flat = moved.reshape(-1, n)
        out = np.empty_like(flat)
        for i in range(0, flat.shape[0], 256):
            blk = flat[i:i + 256]                           # (b, source)
            out[i:i + 256] = np.min(blk[:, None, :] + D2[None, :, :], axis=-1)
        d = np.moveaxis(out.reshape(moved.shape), -1, ax)
    return d


def _city_dist(fg, corner):
    """Exact chessboard (corner) or Manhattan distance to the nearest False element, by breadth-first layers."""
    offs = neighbour_offsets(fg.ndim, corner, center=False)
    d = np.where(fg, np.inf, 0.0)
    cur = ~fg
    step = 0
    remaining = fg.copy()
    while remaining.any() and cur.any():
        step += 1
        grow = np.zeros(fg.shape, dtype=bool)
        for k in offs:
            grow |= _shift(cur, k, False)
        new = grow & remaining
        d[new] = step
        remaining &= ~new
        cur = new
    return d


_DT_FUNCS = ("EuclideanDistance", "SquaredEuclideanDistance", "ManhattanDistance", "ChessboardDistance")


def distance_array(fg, dfun, padv):
    """Distance transform of the boolean foreground fg with the outside treated as foreground (padv True) or
    background (padv False)."""
    nd = fg.ndim
    P = np.pad(fg, 1, mode="constant", constant_values=padv)
    if dfun in ("EuclideanDistance", "SquaredEuclideanDistance"):
        d2 = _edt_sq(P)
        d = d2 if dfun == "SquaredEuclideanDistance" else np.sqrt(d2)
    else:
        d = _city_dist(P, dfun == "ChessboardDistance")
    return d[tuple(slice(1, -1) for _ in range(nd))]


def _dfun_opt(o, fname, allowed=_DT_FUNCS):
    f = o.get("DistanceFunction", Symbol("EuclideanDistance"))
    if _is_sym(f, "Automatic"):
        f = Symbol("EuclideanDistance")
    if not (isinstance(f, Symbol) and f.name in allowed):
        raise NotImplementedError(f"{fname}: DistanceFunction {_short(f)} is not supported ({', '.join(allowed)})")
    return f.name


def _pad01(o, fname, default):
    p = o.get("Padding", default)
    if is_real_like(p) and real_arg(p, fname, "Padding") in (0.0, 1.0):
        return real_arg(p, fname, "Padding") > 0
    raise NotImplementedError(f"{fname}: Padding {_short(p)} is not supported (0 or 1)")


@fn("DistanceTransform", area=AREA, status="added",
    notes="DistanceTransform[image] gives a \"Real32\" image whose foreground pixels (intensity > 0; > t for "
          "DistanceTransform[image, t]; multichannel: the mean intensity) hold the exact distance in pixels to the "
          "nearest background pixel, background 0. DistanceFunction -> EuclideanDistance (default), "
          "SquaredEuclideanDistance, ManhattanDistance or ChessboardDistance; Padding -> 1 (default: outside counts as "
          "foreground) or 0 (outside counts as background). 2D and 3D images; arrays raise TypeError. An image "
          "without background (Padding -> 1) raises ValueError. Check: reference (distances worked out by hand and "
          "brute force) in tests.")
def _distancetransform(x: Expr, *rest: Real | Option) -> Expr:
    fname = "DistanceTransform"
    pos, o = opts_split(rest, {"DistanceFunction", "Padding"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected DistanceTransform[image, t, options]")
    d = data_arg(x, fname)
    if not d.is_image:
        raise TypeError(f"{fname}: expected an image")
    fg = _fg_mask(d, _t_arg(pos, fname), fname)
    dfun = _dfun_opt(o, fname)
    padv = _pad01(o, fname, 1)
    if fg.all() and padv:
        raise ValueError(f"{fname}: the image has no background pixels")
    dist = distance_array(fg, dfun, padv)
    return image_out(d.img, dist[..., None], typ="Real32", cs="Automatic")


def _disk_offsets(rad, dfun, nd=2):
    import itertools
    R = int(math.ceil(rad))
    out = []
    for o in itertools.product(range(-R, R + 1), repeat=nd):
        if dfun == "ManhattanDistance":
            dd = sum(abs(t) for t in o)
        elif dfun == "ChessboardDistance":
            dd = max(abs(t) for t in o)
        else:
            dd = math.sqrt(sum(t * t for t in o))
        if dd < rad - 1e-9:
            out.append(o)
    return out


def inverse_distance(v, dfun):
    """Union of the open disks {q : d(p, q) < v(p)} around every element p with v(p) > 0."""
    out = np.zeros(v.shape, dtype=bool)
    for rv in np.unique(v[v > 0]):
        centres = v == rv
        for k in _disk_offsets(float(rv), dfun, v.ndim):
            out |= _shift(centres, tuple(-t for t in k), False)
    return out


@fn("InverseDistanceTransform", area=AREA, status="added",
    notes="InverseDistanceTransform[image] gives the \"Bit\" image of the union of the disks centred at every pixel "
          "with radius its value, as documented (a disk of radius r holds the pixels at distance < r, so "
          "InverseDistanceTransform[DistanceTransform[b]] == b). DistanceFunction -> EuclideanDistance (default), "
          "ManhattanDistance or ChessboardDistance. Multichannel images use the mean intensity; arrays raise "
          "TypeError. Check: self-check-free round trip InverseDistanceTransform[DistanceTransform[b]] == b in "
          "tests (reference).")
def _inversedistance(x: Expr, *opts: Option) -> Expr:
    fname = "InverseDistanceTransform"
    _, o = opts_split(opts, {"DistanceFunction"}, fname)
    if _:
        raise TypeError(f"{fname}: expected InverseDistanceTransform[image, options]")
    d = data_arg(x, fname)
    if not d.is_image:
        raise TypeError(f"{fname}: expected an image")
    dfun = _dfun_opt(o, fname, ("EuclideanDistance", "ManhattanDistance", "ChessboardDistance"))
    v = intensity(d.img) if d.img.c > 1 else d.img.a[..., 0]
    if d.img.type in MAXV:
        v = v * MAXV[d.img.type]
    return binary_image(d.img, inverse_distance(v, dfun))


@fn("SkeletonTransform", area=AREA, status="added",
    notes="SkeletonTransform[image] gives a \"Real32\" image that is the exact Euclidean distance transform on the "
          "medial axis and 0 elsewhere; SkeletonTransform[image, t] uses foreground > t. The medial axis is taken as "
          "the set of centres of maximal discrete disks (a pixel's open disk of radius its distance value is not "
          "contained in another pixel's disk), so InverseDistanceTransform[SkeletonTransform[i]] reconstructs the "
          "binarised image, as documented; 2D images only. Padding -> 0 (default) or 1. Method -> "
          "\"ReducedMedialAxis\" and 3D images raise NotImplementedError. Check: self-check (the result is verified "
          "to reconstruct the foreground; ArithmeticError otherwise).")
def _skeletontransform(x: Expr, *rest: Real | Option) -> Expr:
    fname = "SkeletonTransform"
    pos, o = opts_split(rest, {"Method", "Padding"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected SkeletonTransform[image, t, options]")
    m = o.get("Method", "MedialAxis")
    if not (m == "MedialAxis" or _is_sym(m, "Automatic")):
        raise NotImplementedError(f"{fname}: Method {_short(m)} is not supported (only \"MedialAxis\")")
    d = data_arg(x, fname, allow3d=False)
    if not d.is_image:
        raise TypeError(f"{fname}: expected an image")
    fg = _fg_mask(d, _t_arg(pos, fname), fname)
    padv = _pad01(o, fname, 0)
    if fg.all() and padv:
        raise ValueError(f"{fname}: the image has no background pixels")
    dist = distance_array(fg, "EuclideanDistance", padv)
    pts = np.argwhere(fg)
    keep = np.zeros(fg.shape, dtype=bool)
    disks = {}
    for p in map(tuple, pts):
        r = dist[p]
        if r not in disks:
            disks[r] = np.array(_disk_offsets(float(r), "EuclideanDistance"), dtype=float)
    rvals = dist[fg]
    for i, p in enumerate(map(tuple, pts)):
        rp = dist[p]
        X = disks[rp] + np.array(p, dtype=float)                # pixels of p's disk
        cand = (rvals > rp - 1e-9) & (np.any(pts != np.array(p), axis=1))
        contained = False
        if cand.any():
            Q = pts[cand].astype(float)
            rq = rvals[cand]
            near = np.sqrt(((Q - np.array(p)) ** 2).sum(1)) < rq
            Q, rq = Q[near], rq[near]
            for q, r in zip(Q, rq):
                if np.all(((X - q) ** 2).sum(1) < r * r - 1e-9):
                    contained = True
                    break
        keep[p] = not contained
    sk = np.where(keep, dist, 0.0)
    if not np.array_equal(inverse_distance(sk, "EuclideanDistance"), fg):
        raise ArithmeticError(f"{fname}: result failed verification")
    return image_out(d.img, sk[..., None], typ="Real32", cs="Automatic")


# ───────────────────────────── component selection and measurement ─────────────────────────────

def _sizes(lab):
    labs, cnt = np.unique(lab[lab > 0], return_counts=True)
    return dict(zip(labs.tolist(), cnt.tolist()))


@fn("DeleteSmallComponents", area=AREA, status="added",
    notes="DeleteSmallComponents[image] removes the connected components of nonzero pixels (CornerNeighbors -> True "
          "by default) whose element count is at most a threshold chosen from the component sizes (Method -> "
          "\"Cluster\": FindThreshold of the list of sizes, the default; \"Mean\"; \"Median\"); with fewer than two "
          "distinct sizes nothing is removed. DeleteSmallComponents[x, n] removes components of n or fewer elements. "
          "A label matrix (non-negative integers) is handled per label. Check: reference (Wolfram documentation "
          "examples) in tests.")
def _deletesmall(x: Expr, *rest: NonNegInteger | Option) -> Expr:
    fname = "DeleteSmallComponents"
    pos, o = opts_split(rest, {"CornerNeighbors", "Method"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected DeleteSmallComponents[image, n, options]")
    lab, d = label_input(x, fname, corner_opt(o, fname))
    sizes = _sizes(lab)
    if pos:
        n = int_arg(pos[0], fname, "n", 0)
    else:
        m = method_arg(o, fname)
        if isinstance(m, tuple):
            raise NotImplementedError(f"{fname}: Method {m[0]!r} is not supported")
        vals = np.array(list(sizes.values()), dtype=float)
        if len(set(vals.tolist())) < 2:
            return _labels_out(d, lab, set(sizes))
        n = threshold_values(vals, m, fname, False)
    keep = {k for k, c in sizes.items() if c > n}
    return _labels_out(d, lab, keep)


def _img_coords(idx, shape):
    """Standard image coordinates of the centre of matrix element idx (2D: {column - 1/2, rows - row + 1/2})."""
    if len(shape) == 2:
        i, j = idx
        return [j + 0.5, shape[0] - i - 0.5]
    k, i, j = idx                  # Image3D: slices, rows, columns -> {x, y, z}
    return [j + 0.5, shape[1] - i - 0.5, shape[0] - k - 0.5]


_CM_PROPS = ("Count", "Centroid", "Medoid", "BoundingBox", "BoundingBoxArea", "MeanCentroidDistance",
             "MaxCentroidDistance", "MinCentroidDistance", "PerimeterCount", "PerimeterPositions", "Neighbors",
             "NeighborCount", "AdjacentBorderCount", "AdjacentBorders", "Min", "Max", "Mean", "Median",
             "StandardDeviation", "Total", "MinIntensity", "MaxIntensity", "MeanIntensity", "MedianIntensity",
             "StandardDeviationIntensity", "TotalIntensity", "Data", "IntensityData", "IntensityCentroid", "Entropy",
             "Energy", "Mask", "LabelCount", "Dimensions")


def _num_out(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return L(*[_num_out(t) for t in v])
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return float(v)


def _measure(prop, k, lab, img, corner, fname):
    pos = np.argwhere(lab == k)
    shape = lab.shape
    nd = lab.ndim
    if prop == "Count":
        return len(pos)
    if prop in ("Centroid", "Medoid", "MeanCentroidDistance", "MaxCentroidDistance", "MinCentroidDistance"):
        C = np.array([_img_coords(tuple(p), shape) for p in pos])
        c = C.mean(0)
        if prop == "Centroid":
            return list(c)
        dist = np.sqrt(((C - c) ** 2).sum(1))
        if prop == "Medoid":
            return list(C[int(np.argmin(dist))])
        return {"MeanCentroidDistance": dist.mean(), "MaxCentroidDistance": dist.max(),
                "MinCentroidDistance": dist.min()}[prop]
    if prop in ("BoundingBox", "BoundingBoxArea"):
        if nd != 2:
            raise NotImplementedError(f"{fname}: {prop} is only supported in 2D")
        i0, j0 = pos.min(0)
        i1, j1 = pos.max(0)
        bb = [[int(j0), int(shape[0] - i1 - 1)], [int(j1 + 1), int(shape[0] - i0)]]
        if prop == "BoundingBox":
            return bb
        return (bb[1][0] - bb[0][0]) * (bb[1][1] - bb[0][1])
    if prop in ("PerimeterCount", "PerimeterPositions"):
        mask = lab == k
        offs = neighbour_offsets(nd, corner, center=False)
        per = np.zeros(shape, dtype=bool)
        for o in offs:
            per |= ~_shift(mask, o, False)
        per &= mask
        if prop == "PerimeterCount":
            return int(per.sum())
        return sorted([_img_coords(tuple(p), shape) for p in np.argwhere(per)])
    if prop in ("Neighbors", "NeighborCount"):
        mask = lab == k
        offs = neighbour_offsets(nd, corner, center=False)
        nb = set()
        for o in offs:
            s = _shift(lab, o, 0)
            nb |= set(np.unique(s[mask]).tolist())
        nb -= {0, k}
        return sorted(nb) if prop == "Neighbors" else len(nb)
    if prop in ("AdjacentBorders", "AdjacentBorderCount"):
        if nd != 2:
            raise NotImplementedError(f"{fname}: {prop} is only supported in 2D")
        sides = []
        if (pos[:, 1] == 0).any():
            sides.append("Left")
        if (pos[:, 1] == shape[1] - 1).any():
            sides.append("Right")
        if (pos[:, 0] == shape[0] - 1).any():
            sides.append("Bottom")
        if (pos[:, 0] == 0).any():
            sides.append("Top")
        return sides if prop == "AdjacentBorders" else len(sides)
    if prop == "Mask":
        return None
    # pixel value properties
    if img is None:
        raise NotImplementedError(f"{fname}: {prop!r} needs an image (a label matrix alone has no pixel values)")
    sel = lab == k
    vals = img.color()[sel] if not img.alpha else img.color()[sel]      # (n, c)
    inten = vals.mean(-1)
    single = vals.shape[1] == 1
    chan = (lambda f: f(vals[:, 0]) if single else [f(vals[:, c]) for c in range(vals.shape[1])])
    if prop == "Data":
        return [list(v) if not single else v[0] for v in vals]
    if prop == "IntensityData":
        return list(inten)
    stat = {"Min": np.min, "Max": np.max, "Mean": np.mean, "Median": np.median, "Total": np.sum,
            "StandardDeviation": lambda t: (np.std(t, ddof=1) if len(t) > 1 else 0.0)}
    if prop in stat:
        return chan(stat[prop])
    if prop.endswith("Intensity") and prop[:-9] in stat:
        return stat[prop[:-9]](inten)
    if prop == "IntensityCentroid":
        C = np.array([_img_coords(tuple(p), shape) for p in np.argwhere(sel)])
        w = inten
        if w.sum() == 0:
            raise ValueError(f"{fname}: component {k} has zero total intensity")
        return list((C * w[:, None]).sum(0) / w.sum())
    if prop == "Entropy":
        return chan(lambda t: WI._entropy(t))
    if prop == "Energy":
        return chan(lambda t: float(((np.unique(t, return_counts=True)[1] / t.size) ** 2).sum()))
    raise NotImplementedError(f"{fname}: property {prop!r} is not supported")


def _cm_input(x, fname, corner):
    """(labels, Img or None) from image, {image, lmat} or a label matrix."""
    if K.is_list(x) and len(x.args) == 2 and isinstance(x.args[0], Expr) and x.args[0].head in IMAGE_HEADS:
        img = parse_image(x.args[0], fname)
        lab, _d = label_input(x.args[1], fname, corner)
        if lab.shape != img.a.shape[:-1]:
            raise ValueError(f"{fname}: the label matrix must have the dimensions of the image")
        return lab, img
    lab, d = label_input(x, fname, corner)
    return lab, (d.img if d.is_image else None)


_PROP_ALIAS = {"Area": None}


def _crit_ok(crit, props, vals, k, lab, img, corner, fname):
    """Evaluate a pure-function criterion with #n (the requested properties) and #name slots."""
    assoc = Expr("Association", *[Expr("Rule", p, _num_out(v)) for p, v in zip(props, vals)])

    def resolve(e):
        if isinstance(e, Expr) and e.head == "Slot" and len(e.args) == 1 and isinstance(e.args[0], str):
            name = e.args[0]
            return _num_out(_measure(name, k, lab, img, corner, fname))
        if isinstance(e, Expr):
            return Expr(e.head, *[resolve(a) for a in e.args])
        return e
    if isinstance(crit, Expr) and crit.head == "Function" and len(crit.args) == 1:
        body = resolve(crit.args[0])
        r = _ev(K.call(Expr("Function", body), *[_num_out(v) for v in vals]))
    else:
        r = _ev(K.call(crit, *[_num_out(v) for v in vals]))
    del assoc
    if r is True or _is_sym(r, "True"):
        return True
    if r is False or _is_sym(r, "False"):
        return False
    raise ValueError(f"{fname}: the criterion must give True or False, not {_short(r)}")


@fn("ComponentMeasurements", area=AREA, status="added",
    notes="ComponentMeasurements[image, prop], [{image, lmat}, prop] and [lmat, prop] give {1 -> v1, 2 -> v2, ...} "
          "for the components (connected nonzero pixels of an image, CornerNeighbors -> True by default, or the "
          "labels of lmat); {prop1, ...} gives lists; ComponentMeasurements[..., prop, crit] keeps components for "
          "which crit (a pure function of #1, #2, ... and #name) is True; format \"ComponentList\" (default), "
          "\"ComponentAssociation\", \"PropertyAssociation\" or \"ComponentPropertyAssociation\". Supported "
          "properties: Count, Centroid, Medoid, Mean/Max/MinCentroidDistance, BoundingBox, BoundingBoxArea, "
          "PerimeterCount, PerimeterPositions, Neighbors, NeighborCount, AdjacentBorders, AdjacentBorderCount, and on "
          "images Min, Max, Mean, Median, StandardDeviation, Total (per channel; Byte/Bit16 values normalised to "
          "0..1, as documented), the ...Intensity versions, Data, IntensityData, IntensityCentroid, Entropy (base "
          "E), Energy; LabelCount and Dimensions are global. Positions use the standard image coordinate system "
          "(pixel centres at half-integers, y up). Other properties (Area, the ellipse, convex hull, contour and "
          "shape properties, ...) raise NotImplementedError. Check: reference (Wolfram documentation Count "
          "examples; hand-computed centroids and boxes) in tests.")
def _componentmeasurements(x: Expr, props: String | ListOf[String], *rest: Fn | String | Option) -> Expr:
    fname = "ComponentMeasurements"
    pos, o = opts_split(rest, {"CornerNeighbors"}, fname)
    if len(pos) > 2:
        raise TypeError(f"{fname}: expected ComponentMeasurements[input, prop, crit, format]")
    corner = corner_opt(o, fname)
    lab, img = _cm_input(x, fname, corner)
    plist = list(props.args) if K.is_list(props) else [props]
    for p in plist:
        if not isinstance(p, str):
            raise TypeError(f"{fname}: properties are strings, not {_short(p)}")
        if p not in _CM_PROPS:
            raise NotImplementedError(f"{fname}: property {p!r} is not supported")
    if not K.is_list(props) and props in ("LabelCount", "Dimensions"):
        return int(lab.max()) if props == "LabelCount" else L(*[int(s) for s in reversed(lab.shape)]) \
            if lab.ndim == 2 else L(*[int(s) for s in (lab.shape[2], lab.shape[1], lab.shape[0])])
    crit, fmt = None, "ComponentList"
    for p in pos:
        if isinstance(p, str):
            fmt = p
        else:
            crit = p
    if fmt not in ("ComponentList", "ComponentAssociation", "PropertyAssociation", "ComponentPropertyAssociation"):
        raise NotImplementedError(f"{fname}: format {fmt!r} is not supported")
    labels = sorted(_sizes(lab))
    rows = []
    for k in labels:
        vals = [_measure(p, k, lab, img, corner, fname) for p in plist]
        if crit is not None and not _crit_ok(crit, plist, vals, k, lab, img, corner, fname):
            continue
        rows.append((k, vals))
    single = not K.is_list(props)
    if fmt == "ComponentList":
        return L(*[Expr("Rule", k, _num_out(v[0]) if single else _num_out(v)) for k, v in rows])
    if fmt == "ComponentAssociation":
        return Expr("Association", *[Expr("Rule", k, _num_out(v[0]) if single else _num_out(v)) for k, v in rows])
    if fmt == "PropertyAssociation":
        return Expr("Association", *[Expr("Rule", p, L(*[_num_out(v[i]) for _, v in rows]))
                                     for i, p in enumerate(plist)])
    return Expr("Association", *[Expr("Rule", k, Expr("Association", *[Expr("Rule", p, _num_out(v[i]))
                                                                        for i, p in enumerate(plist)]))
                                 for k, v in rows])


@fn("SelectComponents", area=AREA, status="added",
    notes="SelectComponents[image, crit], [{image, lmat}, crit] and [lmat, crit] keep the components for which crit "
          "is True (a pure function of #prop slots, e.g. #Count > 2, with the ComponentMeasurements properties) and "
          "replace the others by 0 (black); \"Large\" / \"Small\" keep components larger / not larger than "
          "FindThreshold of the sizes; SelectComponents[..., \"prop\", n] keeps the first n components sorted by "
          "prop (-n: the last n), SelectComponents[..., \"prop\", n, p] sorts with ordering function p. {image, "
          "lmat} input gives {image, lmat}. CornerNeighbors -> True (default) or False. Unsupported properties "
          "raise NotImplementedError. Check: reference (hand-worked selections, consistent with "
          "ComponentMeasurements) in tests.")
def _selectcomponents(x: Expr, crit: Fn | String | ListOf[String], *rest: Integer | Fn | Option) -> Expr:
    fname = "SelectComponents"
    pos, o = opts_split(rest, {"CornerNeighbors"}, fname)
    corner = corner_opt(o, fname)
    pair = K.is_list(x) and len(x.args) == 2 and isinstance(x.args[0], Expr) and x.args[0].head in IMAGE_HEADS
    lab, img = _cm_input(x, fname, corner)
    sizes = _sizes(lab)
    labels = sorted(sizes)
    if pos:
        if len(pos) > 2:
            raise TypeError(f"{fname}: expected SelectComponents[input, prop, n, p]")
        plist = list(crit.args) if K.is_list(crit) else [crit]
        n = int_arg(pos[0], fname, "n")
        keyvals = []
        for k in labels:
            v = [_measure(p, k, lab, img, corner, fname) for p in plist]
            keyvals.append((k, _num_out(v[0]) if len(v) == 1 else _num_out(v)))
        if len(pos) == 2:
            import functools
            pf = pos[1]

            def cmp(a, b):
                r1 = _ev(K.call(pf, a[1], b[1]))
                if r1 is True or _is_sym(r1, "True"):
                    return -1
                r2 = _ev(K.call(pf, b[1], a[1]))
                return 1 if (r2 is True or _is_sym(r2, "True")) else 0
            order = sorted(keyvals, key=functools.cmp_to_key(cmp))
        else:
            order = [keyvals[i] for i in _canonical_order([v for _, v in keyvals])]
        chosen = order[:n] if n >= 0 else order[n:]
        keep = {k for k, _ in chosen}
    elif isinstance(crit, str):
        if crit not in ("Large", "Small"):
            raise ValueError(f"{fname}: unknown criterion {crit!r} (\"Large\", \"Small\" or a function)")
        vals = np.array([sizes[k] for k in labels], dtype=float)
        t = threshold_values(vals, "Cluster", fname, False) if len(set(vals.tolist())) > 1 else None
        if t is None:
            keep = set(labels) if crit == "Large" else set()
        else:
            keep = {k for k in labels if (sizes[k] > t) == (crit == "Large")}
    else:
        keep = {k for k in labels if _crit_ok(crit, [], [], k, lab, img, corner, fname)}
    sel = np.isin(lab, list(keep)) & (lab > 0)
    newlab = np.where(sel, lab, 0)
    if pair:
        return L(image_out(img, img.a * sel[..., None]), int_nest(newlab))
    if img is not None:
        return image_out(img, img.a * sel[..., None])
    return int_nest(newlab)


def _canonical_order(vals):
    import functools
    return sorted(range(len(vals)), key=functools.cmp_to_key(lambda i, j: _ord(vals[i], vals[j])))


def _ord(a, b):
    r = _ev(Expr("Order", a, b))
    return -int(r) if isinstance(r, int) else 0


# ───────────────────────────── perimeter, Euler number, pruning ─────────────────────────────

@fn("MorphologicalPerimeter", area=AREA, status="added",
    notes="MorphologicalPerimeter[data] marks the foreground elements (> 0; > t for MorphologicalPerimeter[data, t]; "
          "multichannel images: mean intensity) with a background neighbour (CornerNeighbors -> True: the 8 (26) "
          "neighbours, False: 4 (6)), as documented; Padding -> 0 (default: the outside is background) or 1. "
          "Images give a \"Bit\" image, arrays a 0/1 array. Check: reference (hand-worked perimeters) in tests.")
def _morphperimeter(x: Expr, *rest: Real | Option) -> Expr:
    fname = "MorphologicalPerimeter"
    pos, o = opts_split(rest, {"CornerNeighbors", "Padding"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected MorphologicalPerimeter[data, t, options]")
    d = data_arg(x, fname)
    fg = _fg_mask(d, _t_arg(pos, fname), fname)
    padv = _pad01(o, fname, 0)
    per = np.zeros(fg.shape, dtype=bool)
    for k in neighbour_offsets(fg.ndim, corner_opt(o, fname), center=False):
        per |= ~_shift(fg, k, padv)
    return binary_like(d, per & fg)


@fn("MorphologicalEulerNumber", area=AREA, status="added",
    notes="MorphologicalEulerNumber[image] is the number of connected foreground regions (> 0; > t for "
          "MorphologicalEulerNumber[image, t]; CornerNeighbors -> True: 8-connected) minus the number of holes "
          "(background regions, connected with the complementary 4-connectivity, that do not reach the outside), as "
          "documented; Padding -> 0 (default) or 1 (the outside is foreground). 2D images and matrices; 3D raises "
          "NotImplementedError. Check: reference (hand-counted regions and holes) in tests.")
def _eulernumber(x: Expr, *rest: Real | Option) -> Integer:
    fname = "MorphologicalEulerNumber"
    pos, o = opts_split(rest, {"CornerNeighbors", "Padding"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected MorphologicalEulerNumber[image, t, options]")
    d = data_arg(x, fname)
    fg = _fg_mask(d, _t_arg(pos, fname), fname)
    if fg.ndim != 2:
        raise NotImplementedError(f"{fname}: only 2D images and matrices are supported")
    corner = corner_opt(o, fname)
    padv = _pad01(o, fname, 0)
    P = np.pad(fg, 1, mode="constant", constant_values=padv)
    nfg = int(label(P, corner).max())
    bl = label(~P, not corner)
    holes = int(bl.max())
    if not padv:
        holes -= 1                       # the outer background region
    if padv:
        nfg = nfg                         # the padding ring joins the touching regions into one region
    return nfg - holes


def _endpoints(fg):
    """End pixels of thin 2D foreground: the foreground 8-neighbours form at most one contiguous run (of length at
    most 3) around the pixel, so deleting it does not split anything; isolated pixels are ends too."""
    ring = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    nb = [_shift(fg, k, False) for k in ring]
    cnt = sum(n.astype(int) for n in nb)
    runs = sum((nb[i] & ~nb[i - 1]).astype(int) for i in range(8))
    return fg & ((cnt == 0) | ((runs == 1) & (cnt <= 3) & (cnt < 8)))


@fn("Pruning", area=AREA, status="added",
    notes="Pruning[image, {n}] removes n pixels from the end of every branch (n rounds of deleting the end pixels: "
          "foreground pixels whose foreground 8-neighbours form one contiguous run of at most 3 pixels); Pruning[image, n] removes every branch of at "
          "most n pixels (n rounds of end deletion, then the surviving branch ends are grown back along the original "
          "pixels for n steps, the standard pruning algorithm); Pruning[image] repeats end deletion until nothing "
          "changes (closed loops remain), as in the documented example. Pruning[image, n, t] uses foreground > t; "
          "Padding -> 0 only. Images give a \"Bit\" image, matrices a 0/1 matrix. Check: reference (Wolfram "
          "documentation example) in tests.")
def _pruning(x: Expr, *rest: NonNegInteger | ListOf[NonNegInteger] | Real | Option) -> Expr:
    fname = "Pruning"
    pos, o = opts_split(rest, {"Padding"}, fname)
    if len(pos) > 2:
        raise TypeError(f"{fname}: expected Pruning[image, n, t, options]")
    if "Padding" in o and o["Padding"] != 0:
        raise NotImplementedError(f"{fname}: only Padding -> 0 is supported")
    d = data_arg(x, fname)
    fg = _fg_mask(d, _t_arg(pos, fname, 1), fname)
    if fg.ndim != 2:
        raise NotImplementedError(f"{fname}: only 2D images and matrices are supported")
    spec = pos[0] if pos else Symbol("Infinity")
    if _is_sym(spec, "Infinity") or (isinstance(spec, Expr) and spec.head == "DirectedInfinity"):
        cur = fg.copy()
        while True:
            e = _endpoints(cur)
            if not e.any():
                break
            cur &= ~e
        return binary_like(d, cur)
    if K.is_list(spec):
        if len(spec.args) != 1:
            raise ValueError(f"{fname}: give {{n}} to remove n pixels per branch")
        n = int_arg(spec.args[0], fname, "n", 0)
        cur = fg.copy()
        for _ in range(n):
            cur &= ~_endpoints(cur)
        return binary_like(d, cur)
    n = int_arg(spec, fname, "n", 0)
    cur = fg.copy()
    for _ in range(n):
        cur &= ~_endpoints(cur)
    grow = _endpoints(cur)
    offs = neighbour_offsets(2, True)
    add = np.zeros(fg.shape, dtype=bool)
    front = grow
    for _ in range(n):
        front = dilate(front.astype(float), offs, 0.0).astype(bool) & fg & ~cur & ~add
        add |= front
    return binary_like(d, cur | add)
