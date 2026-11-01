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
        if s.is_real is False or s.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
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
