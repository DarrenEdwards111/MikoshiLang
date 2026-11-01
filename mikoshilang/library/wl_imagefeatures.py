"""Wolfram batch X05: images IV (feature filters, thinning and skeleton graphs, segmentation, lines, meshes).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md). It builds on W21 (``wl_images``: ``Image`` expressions, ``Img`` values on the 0..1 scale, rows
from the top), W28 (``wl_imageproc``: connected components, regional extrema, neighbourhoods) and X04
(``wl_imageproc2``: padded separable correlation, masks).

Wolfram does not document the algorithms or the Automatic defaults of most of these functions, so each function
uses a named published method and its notes state the method and every default (rather than claiming to
reproduce Wolfram's undocumented choices). Results computed by such methods can differ from Wolfram's in detail.

Coordinates: positions {x, y} use the standard image coordinate system ({0, 0} bottom-left, {w, h} top-right, pixel
centres at half-integers); label matrices and masks are indexed like ImageData (rows from the top).

Python API for X05 registration (``wl_imagefeatures_reg``): ``gray(im)``, ``structure_tensor(a, ...)``,
``corner_measure(...)``, ``gauss_kernels(sigma)``, ``xy_to_rc`` / ``rc_to_xy``.

Pure NumPy does everything here; nothing needs SciPy or Pillow.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L
from . import wl_images as WI
from . import wl_imageproc as WP
from .wl_images import parse_image, image_expr, opts_split, int_arg, real_arg, IMAGE_HEADS, _short, _is_sym
from .wl_imageproc import data_arg, int_nest, image_out, binary_like, binary_image, label, neighbour_offsets, \
    corner_opt, regional_extrema, _shift, _fg_mask, _endpoints, _truthy
from .wl_imageproc2 import correlate_axis_padded, _pad_opt, _mask_of, _image

AREA = "image features and segmentation (X05)"
AUTO = Symbol("Automatic")
TRUE, FALSE = Symbol("True"), Symbol("False")


def _ev(e):
    from ..evaluate import evaluate
    return evaluate(e)


# ───────────────────────────── shared helpers ─────────────────────────────

def img2d(x, fname):
    """Img of a 2D image argument (TypeError for non-images, Unevaluated for symbols, NotImplementedError for 3D)."""
    im = _image(x, fname)
    if im.d3:
        raise NotImplementedError(f"{fname}: Image3D objects are not supported")
    return im


def gray(im):
    """(h, w) intensity: the single channel, or the mean of the colour channels (alpha dropped)."""
    c = im.color()
    return c[..., 0].copy() if c.shape[-1] == 1 else c.mean(axis=-1)


def rc_to_xy(r, c, h):
    """Standard image coordinates of (continuous) 0-based array positions (row r from the top, column c)."""
    return c + 0.5, h - r - 0.5


def xy_to_rc(x, y, h):
    return h - y - 0.5, x - 0.5


def _real(v, fname, what):
    if isinstance(v, Symbol):
        raise Unevaluated()
    return real_arg(v, fname, what)


def _point(p, fname):
    if not (K.is_list(p) and len(p.args) == 2):
        raise TypeError(f"{fname}: {_short(p)} is not a position {{x, y}}")
    return tuple(_real(v, fname, "coordinate") for v in p.args)


def is_point(p):
    return K.is_list(p) and len(p.args) == 2 and all(WI.is_real_value(v) or (isinstance(v, (int, float)) and
                                                                           not isinstance(v, bool)) for v in p.args)


def pixel_of(p, h, w, fname):
    """(row, column) of the pixel containing the image-coordinate position p; ValueError outside the image."""
    x, y = _point(p, fname)
    if not (0 <= x <= w and 0 <= y <= h):
        raise ValueError(f"{fname}: the position {{{x:g}, {y:g}}} is outside the {w} x {h} image")
    return min(h - 1, int(math.floor(h - y))), min(w - 1, int(math.floor(x)))


def marker_mask(spec, h, w, fname):
    """Boolean (h, w) seed mask of a marker: an image or matrix (nonzero pixels), or a list of positions."""
    if isinstance(spec, Expr) and spec.head in IMAGE_HEADS:
        m = parse_image(spec, fname)
        if m.d3 or m.a.shape[:2] != (h, w):
            raise ValueError(f"{fname}: the marker image must have the dimensions of the image ({w} x {h})")
        return np.abs(m.color()).max(-1) > 0
    if K.is_list(spec) and spec.args and all(is_point(p) for p in spec.args):
        mk = np.zeros((h, w), dtype=bool)
        for p in spec.args:
            mk[pixel_of(p, h, w, fname)] = True
        return mk
    if is_point(spec):
        mk = np.zeros((h, w), dtype=bool)
        mk[pixel_of(spec, h, w, fname)] = True
        return mk
    if K.is_list(spec):
        arr = WI._numeric_array(spec, fname)
        if arr is not None and arr.ndim == 2:
            if arr.shape != (h, w):
                raise ValueError(f"{fname}: the marker matrix must have the dimensions of the image ({h} x {w})")
            return arr != 0
        raise TypeError(f"{fname}: {_short(spec)} is not a marker (an image, a matrix or a list of positions)")
    if isinstance(spec, Symbol) or (isinstance(spec, Expr) and spec.head not in ("Rule", "Association")):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_short(spec)} is not a marker (an image, a matrix or a list of positions)")


def seed_labels(spec, h, w, corner, fname):
    """Seed label matrix of a marker: positions are labelled 1, 2, ... in the order given; images and matrices by
    their connected components of nonzero pixels (scan order)."""
    if K.is_list(spec) and spec.args and all(is_point(p) for p in spec.args):
        lab = np.zeros((h, w), dtype=np.int64)
        for k, p in enumerate(spec.args, 1):
            rc = pixel_of(p, h, w, fname)
            if lab[rc] == 0:
                lab[rc] = k
        return lab
    return label(marker_mask(spec, h, w, fname), corner)


def gauss_kernels(sigma):
    """(smoothing, derivative) correlation kernels of a sampled Gaussian with standard deviation sigma (radius
    ceil(3 sigma)): the smoothing kernel sums to 1, the derivative kernel s g(s) is scaled so that a unit ramp gives
    1. sigma = 0 gives the identity and the central difference {-1/2, 0, 1/2}."""
    if sigma <= 0:
        return np.array([1.0]), np.array([-0.5, 0.0, 0.5])
    r = max(1, int(math.ceil(3 * sigma)))
    s = np.arange(-r, r + 1, dtype=float)
    g = np.exp(-s * s / (2 * sigma * sigma))
    g /= g.sum()
    d = s * g
    d /= (s * d).sum()
    return g, d


def gradients(a, sig_r, sig_c, pad="Fixed"):
    """(d/dcol, d/drow) of an (h, w[, c]) array with Gaussian derivative kernels (per-axis sigmas)."""
    gr, dr = gauss_kernels(sig_r)
    gc, dc = gauss_kernels(sig_c)
    ix = correlate_axis_padded(correlate_axis_padded(a, dc, 1, pad), gr, 0, pad)
    iy = correlate_axis_padded(correlate_axis_padded(a, gc, 1, pad), dr, 0, pad)
    return ix, iy


def smooth(a, sig_r, sig_c, pad="Fixed"):
    gr, _ = gauss_kernels(sig_r)
    gc, _ = gauss_kernels(sig_c)
    return correlate_axis_padded(correlate_axis_padded(a, gc, 1, pad), gr, 0, pad)


def structure_tensor(a, sig, rho, pad="Fixed"):
    """Channel-summed structure tensor (Jxx, Jxy, Jyy) of an (h, w, c) array: Gaussian derivatives with standard
    deviations sig = (rows, cols), integrated with Gaussians of standard deviations rho = (rows, cols)."""
    ix, iy = gradients(a, sig[0], sig[1], pad)
    jxx = smooth((ix * ix).sum(-1), rho[0], rho[1], pad)
    jxy = smooth((ix * iy).sum(-1), rho[0], rho[1], pad)
    jyy = smooth((iy * iy).sum(-1), rho[0], rho[1], pad)
    return jxx, jxy, jyy


def corner_measure(jxx, jxy, jyy, method="ShiTomasi", k=0.04):
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    if method == "ShiTomasi":
        return tr / 2 - np.sqrt(np.maximum(((jxx - jyy) / 2) ** 2 + jxy * jxy, 0.0))
    if method == "HarrisStephens":
        return det - k * tr * tr
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tr > 1e-300, 2 * det / np.where(tr > 1e-300, tr, 1.0), 0.0)     # HarmonicMean


def _radii2(r, fname, what="range", allow_zero=True):
    """Per-axis (rows, columns) non-negative real radii from r or {r_row, r_col}."""
    vals = list(r.args) if K.is_list(r) else [r, r]
    if len(vals) != 2:
        raise ValueError(f"{fname}: give the {what} as r or {{r_row, r_col}}")
    out = []
    for v in vals:
        if isinstance(v, (bool, str)) or K.is_list(v):
            raise TypeError(f"{fname}: the {what} must be a non-negative number, not {_short(v)}")
        t = _real(v, fname, what)
        if t < 0 or (t == 0 and not allow_zero):
            raise ValueError(f"{fname}: the {what} must be {'non-negative' if allow_zero else 'positive'}, "
                             f"not {t:g}")
        out.append(t)
    return out


def _method(o, fname, default):
    """(name, list of further list elements, {suboption: value}) of a Method option."""
    m = o.get("Method", AUTO)
    if _is_sym(m, "Automatic"):
        return default, [], {}
    if isinstance(m, str):
        return m, [], {}
    if K.is_list(m) and m.args:
        items = list(m.args)
        name = default
        if isinstance(items[0], str):
            name = items.pop(0)
        elif _is_sym(items[0], "Automatic"):
            items.pop(0)
        rest, sub = [], {}
        for it in items:
            if isinstance(it, Expr) and it.head in ("Rule", "RuleDelayed") and isinstance(it.args[0], str):
                sub[it.args[0]] = it.args[1]
            else:
                rest.append(it)
        return name, rest, sub
    raise ValueError(f"{fname}: invalid Method {_short(m)}")


def _no_sub(sub, fname, name):
    if sub:
        raise NotImplementedError(f"{fname}: Method suboption {next(iter(sub))!r} is not supported with {name!r}")


def _labels_out(lab):
    return int_nest(np.asarray(lab, dtype=np.int64))


# ───────────────────────────── CornerFilter ─────────────────────────────

_CF_METHODS = ("ShiTomasi", "HarrisStephens", "HarmonicMean")


@fn("CornerFilter", area=AREA, status="added",
    notes="CornerFilter[image] (= CornerFilter[image, 2]) and CornerFilter[image, r | {r_row, r_col}] give a "
          "\"Real32\" image of a structure-tensor corner measure: Gaussian derivatives (standard deviation sigma) of "
          "every channel, the products summed over channels and integrated with a Gaussian of standard deviation "
          "rho/2; Method -> {method, \"Gaussian\" -> {rho, sigma}}, default {r, r/2} as documented. Methods: "
          "\"ShiTomasi\" (default; the smaller eigenvalue), \"HarrisStephens\" (det - k tr^2, k = 0.04 by default, "
          "{\"HarrisStephens\", k}; negative on edges) and \"HarmonicMean\" (2 det/tr, the harmonic mean of the "
          "eigenvalues). Values are on the 0..1 intensity scale (Wolfram's normalisation and its default k are not "
          "documented). Padding -> \"Fixed\" (default) or another padding method. 2D images only (Image3D raises "
          "NotImplementedError); non-images raise TypeError, negative ranges ValueError. Check: numeric cross-check "
          "(tests: ShiTomasi is ~0 on straight edges and maximal at square corners, HarrisStephens with k = 0 equals "
          "the determinant, HarmonicMean = 2 det/tr from the other two).")
def _cornerfilter(x: Expr, *rest: Real | ListOf[Real] | Option) -> Expr:
    fname = "CornerFilter"
    pos, o = opts_split(rest, {"Method", "Padding"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected CornerFilter[image, r, options]")
    im = img2d(x, fname)
    rr, rc = _radii2(pos[0] if pos else 2, fname)
    name, extra, sub = _method(o, fname, "ShiTomasi")
    if name not in _CF_METHODS:
        raise NotImplementedError(f"{fname}: Method {name!r} is not supported ({', '.join(_CF_METHODS)})")
    k = 0.04
    if extra:
        if name != "HarrisStephens" or len(extra) != 1:
            raise ValueError(f"{fname}: only {{\"HarrisStephens\", k}} takes a parameter")
        k = _real(extra[0], fname, "k")
    rho, sig = (rr, rc), (rr / 2, rc / 2)
    if "Gaussian" in sub:
        g = sub.pop("Gaussian")
        if not (K.is_list(g) and len(g.args) == 2):
            raise ValueError(f"{fname}: \"Gaussian\" -> {{rho, sigma}} expected")
        p, s = (_real(v, fname, "Gaussian parameter") for v in g.args)
        if p < 0 or s < 0:
            raise ValueError(f"{fname}: the Gaussian parameters must be non-negative")
        rho, sig = (p, p), (s, s)
    _no_sub(sub, fname, name)
    pad = _pad_opt(o, fname, 2)
    if any(p is None for p in pad):
        raise NotImplementedError(f"{fname}: Padding -> None is not supported")
    jxx, jxy, jyy = structure_tensor(im.color(), sig, (rho[0] / 2, rho[1] / 2), pad[0])
    res = corner_measure(jxx, jxy, jyy, name, k)
    return image_out(im, res[..., None], typ="Real64" if im.type == "Real64" else "Real32", cs="Automatic")


# ───────────────────────────── EdgeDetect (Canny) ─────────────────────────────

def _binom(n):
    row = np.array([1.0])
    for _ in range(n):
        row = np.convolve(row, [1.0, 1.0])
    return row / row.sum()


def _sobel_grad(a, rr, rc, pad):
    """Binomial (generalised Sobel) gradient of radius r: smoothing binom(2r) across, d = binom(2r-1) * {-1, 1}."""
    def kers(r):
        r = int(round(r))
        if r <= 0:
            return np.array([1.0]), np.array([-0.5, 0.0, 0.5])
        return _binom(2 * r), np.convolve(_binom(2 * r - 1), [1.0, -1.0])[::-1] * 1.0
    gr, dr = kers(rr)
    gc, dc = kers(rc)
    ix = correlate_axis_padded(correlate_axis_padded(a, dc, 1, pad), gr, 0, pad)
    iy = correlate_axis_padded(correlate_axis_padded(a, gc, 1, pad), dr, 0, pad)
    return ix, iy


def otsu(v):
    """Otsu's threshold of the values v (256 bins over [min, max])."""
    v = np.asarray(v, dtype=float)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(v, bins=256, range=(lo, hi))
    p = hist / hist.sum()
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    m = np.cumsum(p * mids)
    mt = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w0 - m) ** 2 / (w0 * (1 - w0))
    sb[~np.isfinite(sb)] = -1
    return float(edges[int(np.argmax(sb)) + 1])


def canny(a, ix, iy):
    """Normalised non-maximum-suppressed gradient magnitude (Canny 1986, four directions) of an (h, w, c) gradient:
    at each pixel the channel with the largest magnitude is used."""
    mag = np.sqrt(ix * ix + iy * iy)
    ch = np.argmax(mag, axis=-1)[..., None]
    gx = np.take_along_axis(ix, ch, -1)[..., 0]
    gy = np.take_along_axis(iy, ch, -1)[..., 0]
    m = np.take_along_axis(mag, ch, -1)[..., 0]
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    q = (np.floor((ang + 22.5) / 45.0).astype(int)) % 4      # 0: along columns, 1: diagonal, 2: rows, 3: antidiag
    offs = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}
    keep = np.zeros(m.shape, dtype=bool)
    for d, (di, dj) in offs.items():
        n1 = _shift(m, (di, dj), 0.0)
        n2 = _shift(m, (-di, -dj), 0.0)
        keep |= (q == d) & (m > n1) & (m >= n2)
    keep &= m > 0
    nms = np.where(keep, m, 0.0)
    mx = nms.max()
    return nms / mx if mx > 0 else nms


def hysteresis(v, lo, hi):
    weak = v > lo
    strong = v > hi
    lab = label(weak, True)
    good = np.unique(lab[strong & weak])
    return np.isin(lab, good[good > 0])


def _thresholds(t, v, fname):
    if t is None or _is_sym(t, "Automatic"):
        nz = v[v > 0]
        if nz.size == 0:
            return 0.0, 1.0
        hi = otsu(nz)
        return hi / 2, hi
    if K.is_list(t):
        if len(t.args) != 2:
            raise ValueError(f"{fname}: give the threshold as t or {{t1, t2}}")
        lo, hi = (_real(u, fname, "threshold") for u in t.args)
        if lo > hi:
            raise ValueError(f"{fname}: the thresholds {{t1, t2}} must satisfy t1 <= t2")
        return lo, hi
    if isinstance(t, (bool, str)):
        raise TypeError(f"{fname}: the threshold must be a number, not {_short(t)}")
    hi = _real(t, fname, "threshold")
    return hi / 2, hi


@fn("EdgeDetect", area=AREA, status="added",
    notes="EdgeDetect[image] (= EdgeDetect[image, 2]), EdgeDetect[image, r | {r_row, r_col}] and EdgeDetect[image, "
          "r, t] give a \"Bit\" image of edges by Canny's method (Canny 1986): gradients (Method -> \"Canny\", "
          "default: Gaussian derivatives of standard deviation r/2; \"Sobel\": binomial generalised Sobel masks of "
          "radius r), at each pixel the channel with the largest gradient, non-maximum suppression in four "
          "directions, the result normalised by its maximum, then hysteresis thresholding (8-connected): pixels above "
          "t/2 connected to pixels above t (EdgeDetect[image, r, {t1, t2}]: above t1 connected to above t2). The "
          "default threshold (Automatic) is Otsu's threshold of the nonzero suppressed magnitudes, with t/2 as the "
          "low threshold (Wolfram's default is not documented). Padding -> \"Fixed\" (default) or another method. "
          "\"ShenCastan\", \"StraightEdges\" -> s > 0 and Image3D raise NotImplementedError; non-images TypeError. "
          "Check: reference (tests: the edges of a square lie on its boundary and form one closed curve; a constant "
          "image has no edges).")
def _edgedetect(x: Expr, *rest: Real | ListOf[Real] | Option) -> Expr:
    fname = "EdgeDetect"
    pos, o = opts_split(rest, {"Method", "Padding"}, fname)
    if len(pos) > 2:
        raise TypeError(f"{fname}: expected EdgeDetect[image, r, t, options]")
    im = img2d(x, fname)
    rr, rc = _radii2(pos[0] if pos else 2, fname)
    name, extra, sub = _method(o, fname, "Canny")
    if name not in ("Canny", "Sobel"):
        raise NotImplementedError(f"{fname}: Method {name!r} is not supported (\"Canny\" or \"Sobel\")")
    if extra:
        raise ValueError(f"{fname}: invalid Method specification")
    s = sub.pop("StraightEdges", 0)
    if _real(s, fname, "StraightEdges") != 0:
        raise NotImplementedError(f"{fname}: \"StraightEdges\" -> s > 0 is not supported")
    _no_sub(sub, fname, name)
    pad = _pad_opt(o, fname, 2)
    if any(p is None for p in pad):
        raise NotImplementedError(f"{fname}: Padding -> None is not supported")
    a = im.color()
    if name == "Canny":
        ix, iy = gradients(a, rr / 2, rc / 2, pad[0])
    else:
        ix, iy = _sobel_grad(a, rr, rc, pad[0])
    v = canny(a, ix, iy)
    lo, hi = _thresholds(pos[1] if len(pos) > 1 else None, v, fname)
    return binary_image(im, hysteresis(v, lo, hi))


# ───────────────────────────── Thinning, branch points ─────────────────────────────

# Golay L structuring elements (Gonzalez & Woods, "Digital Image Processing", thinning): 1 foreground, 0 background,
# -1 don't care; B1, B2 and their rotations by multiples of 90 degrees.
_B1 = np.array([[0, 0, 0], [-1, 1, -1], [1, 1, 1]])
_B2 = np.array([[-1, 0, 0], [1, 1, 0], [1, 1, -1]])
THIN_ELEMENTS = []
for _k in range(4):
    THIN_ELEMENTS += [np.rot90(_B1, -_k), np.rot90(_B2, -_k)]


def _hit_miss(fg, se):
    out = fg.copy()
    for i in range(3):
        for j in range(3):
            v = se[i, j]
            if v < 0:
                continue
            s = _shift(fg, (i - 1, j - 1), False)
            out &= s if v == 1 else ~s
    return out


def thin(fg, n=None):
    """Sequential morphological thinning with the 8 Golay L elements (n passes, or until nothing changes)."""
    cur = fg.copy()
    k = 0
    while n is None or k < n:
        before = cur.copy()
        for se in THIN_ELEMENTS:
            cur &= ~_hit_miss(cur, se)
        k += 1
        if np.array_equal(before, cur):
            break
    return staircase(cur)


def _n8(P, i, j):
    """Yokoi's 8-connectivity number of the pixel (i, j) of the padded boolean array P (1-based interior)."""
    x = [P[i, j + 1], P[i - 1, j + 1], P[i - 1, j], P[i - 1, j - 1], P[i, j - 1], P[i + 1, j - 1], P[i + 1, j],
         P[i + 1, j + 1]]
    xb = [not v for v in x]
    return sum(xb[k] - (xb[k] and xb[(k + 1) % 8] and xb[(k + 2) % 8]) for k in (0, 2, 4, 6))


def staircase(fg):
    """Remove the corner pixels of 4-connected staircases (a horizontal and a vertical foreground 4-neighbour,
    Yokoi 8-connectivity number 1), in raster order, so that thin lines become minimally 8-connected."""
    P = np.pad(fg, 1, constant_values=False)
    h, w = fg.shape
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            if P[i, j] and (P[i, j - 1] or P[i, j + 1]) and (P[i - 1, j] or P[i + 1, j]) and _n8(P, i, j) == 1:
                P[i, j] = False
    return P[1:-1, 1:-1]


def topology(fg):
    """(8-connected foreground components, 4-connected background components incl. the outside)."""
    P = np.pad(fg, 1, constant_values=False)
    return int(label(fg, True).max()), int(label(~P, False).max())


_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def crossing_number(fg):
    """Number of runs of foreground pixels in the circular 8-neighbourhood of every pixel."""
    nb = [_shift(fg, k, False) for k in _RING]
    return sum((nb[i] & ~nb[i - 1]).astype(int) for i in range(8))


def branch_points(fg):
    return fg & (crossing_number(fg) >= 3)


def _thin_input(x, rest, fname, maxpos):
    pos, o = opts_split(rest, {"Padding", "Method"}, fname)
    if len(pos) > maxpos:
        raise TypeError(f"{fname}: too many arguments")
    if "Padding" in o and o["Padding"] != 0:
        raise NotImplementedError(f"{fname}: only Padding -> 0 is supported")
    d = data_arg(x, fname, allow3d=False)
    if d.is_image and d.img.d3:
        raise NotImplementedError(f"{fname}: Image3D objects are not supported")
    return pos, o, d


@fn("Thinning", area=AREA, status="added",
    notes="Thinning[image] thins the foreground (intensity > 0; Thinning[image, n, t]: > t) to its skeleton by "
          "sequential morphological thinning with the eight Golay L structuring elements (B1, B2 and their quarter "
          "turns; Gonzalez & Woods), repeated until nothing changes (Thinning[image, n]: n passes of the eight "
          "elements; Infinity = until convergence), followed by one raster pass removing the corner pixels of "
          "4-connected staircases (Yokoi 8-connectivity number 1) so that lines are minimally 8-connected; short "
          "spurs can remain at convex corners (Pruning removes them). Method -> \"Morphological\" (default) or \"MedialAxis\" (the "
          "support of SkeletonTransform: centres of maximal disks). Images give a \"Bit\" image, real matrices a 0/1 "
          "matrix. Padding -> 0 only; Image3D raises NotImplementedError, negative n ValueError. Check: self-check "
          "(morphological thinning must keep the number of 8-connected foreground components and of 4-connected "
          "background components, and only remove pixels; ArithmeticError otherwise).")
def _thinning(x: Expr, *rest: NonNegInteger | Real | Option) -> Expr:
    fname = "Thinning"
    pos, o, d = _thin_input(x, rest, fname, 2)
    m = o.get("Method", "Morphological")
    if _is_sym(m, "Automatic"):
        m = "Morphological"
    if m not in ("Morphological", "MedialAxis"):
        raise NotImplementedError(f"{fname}: Method {_short(m)} is not supported (\"Morphological\" or "
                                  "\"MedialAxis\")")
    n = None
    if pos and not (_is_sym(pos[0], "Infinity") or (isinstance(pos[0], Expr) and pos[0].head == "DirectedInfinity")):
        n = int_arg(pos[0], fname, "n", 0)
    t = 0.0 if len(pos) < 2 else _real(pos[1], fname, "threshold")
    fg = _fg_mask(d, t, fname)
    if m == "MedialAxis":
        if n is not None:
            raise NotImplementedError(f"{fname}: an iteration count is not supported with \"MedialAxis\"")
        sk = _ev(Expr("SkeletonTransform", image_expr(WI.Img(fg.astype(float), "Bit"))))
        return binary_like(d, parse_image(sk, fname).a[..., 0] > 0)
    res = thin(fg, n)
    if np.any(res & ~fg) or topology(res) != topology(fg):
        raise ArithmeticError(f"{fname}: result failed verification")
    return binary_like(d, res)


@fn("MorphologicalBranchPoints", area=AREA, status="added",
    notes="MorphologicalBranchPoints[image] gives a \"Bit\" image of the branch points of the (thin) foreground "
          "(intensity > 0): foreground pixels whose circular 8-neighbourhood contains at least three separate runs "
          "of foreground pixels (crossing number >= 3, Rutovitz). The image is used as given (apply Thinning first "
          "for thick shapes, as MorphologicalGraph does); matrices give 0/1 matrices. Padding -> 0 only; Image3D "
          "raises NotImplementedError. Check: reference (tests: the centre of a plus sign and the junction of a T are "
          "branch points; lines and loops have none).")
def _morphbranchpoints(x: Expr, *opts: Option) -> Expr:
    fname = "MorphologicalBranchPoints"
    pos, o, d = _thin_input(x, opts, fname, 0)
    if "Method" in o:
        raise NotImplementedError(f"{fname}: option Method is not supported")
    return binary_like(d, branch_points(_fg_mask(d, 0.0, fname)))


# ───────────────────────────── MorphologicalGraph ─────────────────────────────

def skeleton_graph(sk):
    """(vertex label matrix, [(i, j, weight)], vertex pixel lists) of a thin skeleton: 8-connected clusters of branch
    points and every end point are vertices; edges join vertices connected through non-vertex skeleton pixels,
    weighted by the smallest number of 8-connected steps; a step between two neighbours of the same branch
    cluster counts as passing through that cluster (so paths cannot bypass a junction diagonally)."""
    h, w = sk.shape
    br = branch_points(sk)
    ends = _endpoints(sk) & ~br
    vlab = np.zeros(sk.shape, dtype=np.int64)
    blab = label(br, True)
    # vertices in scan order of their first pixel
    order, seen = [], {}
    for i in range(h):
        for j in range(w):
            if br[i, j] and blab[i, j] not in seen:
                seen[blab[i, j]] = len(order) + 1
                order.append(("b", blab[i, j]))
            elif ends[i, j]:
                order.append(("e", (i, j)))
                vlab[i, j] = len(order)
    for b, k in seen.items():
        vlab[blab == b] = k
    nv = len(order)
    best = {}
    offs = [k for k in neighbour_offsets(2, True, center=False)]
    vpix = {k: list(zip(*np.nonzero(vlab == k))) for k in range(1, nv + 1)}
    bset = {seen[b] for b in seen}
    att = {}
    for k in bset:
        for (i, j) in vpix[k]:
            for di, dj in offs:
                p, q = i + di, j + dj
                if 0 <= p < h and 0 <= q < w and sk[p, q] and not vlab[p, q]:
                    att.setdefault((p, q), set()).add(k)
    empty = frozenset()
    for v in range(1, nv + 1):
        dist = {}
        front = []
        for (i, j) in vpix[v]:
            for di, dj in offs:
                p, q = i + di, j + dj
                if 0 <= p < h and 0 <= q < w and sk[p, q]:
                    u = vlab[p, q]
                    if u and u != v:
                        best[(min(u, v), max(u, v))] = min(best.get((min(u, v), max(u, v)), 10 ** 9), 1)
                    elif not u and (p, q) not in dist:
                        dist[(p, q)] = 1
                        front.append((p, q))
        while front:
            nxt = []
            for (i, j) in front:
                dd = dist[(i, j)]
                ai = att.get((i, j), empty)
                for di, dj in offs:
                    p, q = i + di, j + dj
                    if 0 <= p < h and 0 <= q < w and sk[p, q]:
                        u = vlab[p, q]
                        if not u and ai and ai & att.get((p, q), empty):
                            continue                # a step between two neighbours of one junction passes it
                        if u and u != v:
                            key = (min(u, v), max(u, v))
                            best[key] = min(best.get(key, 10 ** 9), dd + 1)
                        elif not u and (p, q) not in dist:
                            dist[(p, q)] = dd + 1
                            nxt.append((p, q))
            front = nxt
    edges = sorted((int(a), int(b), int(wt)) for (a, b), wt in best.items())
    return vlab, edges, vpix


@fn("MorphologicalGraph", area=AREA, status="added",
    notes="MorphologicalGraph[image] thins the foreground (intensity > 0) with Thinning, takes the 8-connected "
          "clusters of branch points (crossing number >= 3) and the end points (as in Pruning) as vertices, "
          "numbered 1, 2, ... in scan order (rows from the top) of their first pixel, and joins two vertices by an "
          "undirected edge when the skeleton connects them through non-vertex pixels; EdgeWeight is the smallest "
          "number of 8-connected steps between them (one edge per pair, no self-loops) and VertexCoordinates the "
          "mean image coordinates of each vertex's pixels, as documented. Closed loops without branch or end points "
          "have no vertices and are omitted; isolated pixels are isolated vertices. Other Graph options are passed "
          "to Graph. Image3D raises NotImplementedError. Check: reference (tests: an L shape gives one edge whose "
          "weight is its pixel path, a plus sign a star with four edges, a figure H five vertices and four edges).")
def _morphologicalgraph(x: Expr, *opts: Option) -> Expr:
    fname = "MorphologicalGraph"
    rules = [r for r in opts if isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed")]
    if len(rules) != len(opts):
        raise TypeError(f"{fname}: expected MorphologicalGraph[image, options]")
    im = img2d(x, fname) if isinstance(x, Expr) and x.head in IMAGE_HEADS else None
    if im is None:
        d = data_arg(x, fname, allow3d=False)
        fg = _fg_mask(d, 0.0, fname)
        if fg.ndim != 2:
            raise NotImplementedError(f"{fname}: only 2D images are supported")
        h = fg.shape[0]
    else:
        fg = gray(im) > 0
        h = im.h
    sk = thin(fg)
    vlab, edges, vpix = skeleton_graph(sk)
    nv = len(vpix)
    coords = []
    for k in range(1, nv + 1):
        px = np.array(vpix[k], dtype=float)
        xs, ys = rc_to_xy(px[:, 0], px[:, 1], h)
        coords.append(L(float(xs.mean()), float(ys.mean())))
    given = {r.args[0].name if isinstance(r.args[0], Symbol) else r.args[0] for r in rules}
    out = list(rules)
    if "EdgeWeight" not in given:
        out.append(Expr("Rule", Symbol("EdgeWeight"), L(*[wt for _, _, wt in edges])))
    if "VertexCoordinates" not in given:
        out.append(Expr("Rule", Symbol("VertexCoordinates"), L(*coords)))
    return _ev(Expr("Graph", L(*range(1, nv + 1)), L(*[Expr("UndirectedEdge", a, b) for a, b, _ in edges]), *out))


# ───────────────────────────── WatershedComponents ─────────────────────────────

def _gray_of(d, fname):
    if d.is_image:
        return gray(d.img)
    if d.exact:
        return np.vectorize(float, otypes=[float])(d.obj)
    return np.asarray(d.a, dtype=float)


def watershed(g, seeds, offs, lines=True):
    """Meyer's flooding watershed of g from the seed label matrix (priority queue on g, FIFO among ties); with lines,
    a pixel reached from two different basins becomes a watershed pixel (label 0)."""
    h, w = g.shape
    lab = seeds.copy()
    LINE = -1
    heap, cnt = [], 0
    inq = lab > 0
    for i, j in zip(*np.nonzero(lab > 0)):
        heapq.heappush(heap, (g[i, j], cnt, i, j))
        cnt += 1
    while heap:
        _, _, i, j = heapq.heappop(heap)
        for di, dj in offs:
            p, q = i + di, j + dj
            if not (0 <= p < h and 0 <= q < w) or inq[p, q]:
                continue
            if lines:
                ls = set()
                for ei, ej in offs:
                    a, b = p + ei, q + ej
                    if 0 <= a < h and 0 <= b < w and lab[a, b] > 0:
                        ls.add(lab[a, b])
                if len(ls) > 1:
                    lab[p, q] = LINE
                    inq[p, q] = True
                    continue
            lab[p, q] = lab[i, j]
            inq[p, q] = True
            heapq.heappush(heap, (g[p, q], cnt, p, q))
            cnt += 1
    lab[lab < 0] = 0
    return lab


@fn("WatershedComponents", area=AREA, status="added",
    notes="WatershedComponents[image] gives the label matrix of the watershed basins of the intensity (mean over "
          "channels; real matrices are used directly), flooded from every regional minimum (plateaus labelled 1, 2, "
          "... in scan order); WatershedComponents[image, marker] floods only from the marker: a binary image or "
          "matrix (each 8-connected component of nonzero pixels is a seed, labelled in scan order) or a list of "
          "positions {x, y} (labelled in the order given). Method -> \"Watershed\" (default): Meyer's flooding "
          "algorithm with watershed lines (0s, pixels reached from two basins); \"Basins\": the same flooding without "
          "lines (every pixel labelled). CornerNeighbors -> True (Automatic) or False. \"Rainfall\", \"Immersion\", "
          "\"MinimumSaliency\" and Image3D raise NotImplementedError. Check: reference (tests: two basins separated "
          "by a ridge, with the ridge as the watershed line; marker labels follow the marker order).")
def _watershedcomponents(x: Expr, *rest: Expr | Option) -> Expr:
    fname = "WatershedComponents"
    pos, o = opts_split(rest, {"Method", "CornerNeighbors"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected WatershedComponents[image, marker, options]")
    d = data_arg(x, fname, allow3d=False)
    if d.is_image and d.img.d3:
        raise NotImplementedError(f"{fname}: Image3D objects are not supported")
    g = _gray_of(d, fname)
    if g.ndim != 2:
        raise NotImplementedError(f"{fname}: only 2D images and matrices are supported")
    m = o.get("Method", "Watershed")
    if _is_sym(m, "Automatic"):
        m = "Watershed"
    if m not in ("Watershed", "Basins"):
        raise NotImplementedError(f"{fname}: Method {_short(m)} is not supported (\"Watershed\" or \"Basins\")")
    corner = corner_opt(o, fname)
    offs = neighbour_offsets(2, corner, center=False)
    h, w = g.shape
    if pos:
        seeds = seed_labels(pos[0], h, w, True, fname)
        if not seeds.any():
            raise ValueError(f"{fname}: the marker has no foreground pixels")
    else:
        seeds = label(regional_extrema(g, neighbour_offsets(2, corner), maxima=False), corner)
    return _labels_out(watershed(g, seeds, offs, m == "Watershed"))


# ───────────────────────────── ChanVeseBinarize ─────────────────────────────

def chan_vese(f, phi, mu, nu, l1, l2, iters, c1fix=None, c2fix=None, dt=0.5, eps=1.0, tol=1e-3):
    """Chan-Vese (2001) two-phase level set with the semi-implicit scheme of Getreuer (IPOL 2012), Jacobi sweeps.
    f is (h, w, c); returns phi."""
    eta = 1e-8
    for _ in range(iters):
        inside = phi > 0
        c1 = c1fix if c1fix is not None else (f[inside].mean(0) if inside.any() else np.zeros(f.shape[-1]))
        c2 = c2fix if c2fix is not None else (f[~inside].mean(0) if (~inside).any() else np.zeros(f.shape[-1]))
        P = np.pad(phi, 1, mode="edge")
        c = P[1:-1, 1:-1]
        n_, s_, w_, e_ = P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]
        nw, ne, sw, se = P[:-2, :-2], P[:-2, 2:], P[2:, :-2], P[2:, 2:]
        C1 = 1 / np.sqrt(eta + (e_ - c) ** 2 + ((s_ - n_) / 2) ** 2)
        C2 = 1 / np.sqrt(eta + (c - w_) ** 2 + ((sw - nw) / 2) ** 2)
        C3 = 1 / np.sqrt(eta + (s_ - c) ** 2 + ((e_ - w_) / 2) ** 2)
        C4 = 1 / np.sqrt(eta + (c - n_) ** 2 + ((ne - se) / 2) ** 2)
        delta = dt * eps / (math.pi * (eps * eps + phi * phi))
        fit = -nu - l1 * ((f - c1) ** 2).sum(-1) + l2 * ((f - c2) ** 2).sum(-1)
        num = phi + delta * (mu * (C1 * e_ + C2 * w_ + C3 * s_ + C4 * n_) + fit)
        new = num / (1 + delta * mu * (C1 + C2 + C3 + C4))
        change = np.abs(new - phi).mean()
        phi = new
        if change < tol * dt:
            break
    return phi


def _marker_check(mask):
    if mask.all() or not mask.any():
        raise ValueError("ChanVeseBinarize: the marker must contain both foreground and background pixels")


@fn("ChanVeseBinarize", area=AREA, status="added",
    notes="ChanVeseBinarize[image], ChanVeseBinarize[image, marker] and ChanVeseBinarize[image, marker, {mu, nu, "
          "lambda1, lambda2}] give a \"Bit\" image of the Chan-Vese (2001) two-phase segmentation minimising mu "
          "Length + nu Area(D) + lambda1 Int_D |f - c1|^2 + lambda2 Int_(not D) |f - c2|^2 (|.| the Euclidean "
          "distance of channel vectors on the 0..1 scale), by the semi-implicit level-set scheme of Getreuer (IPOL "
          "2012; time step 0.5, regularised delta with epsilon 1, stopping when the mean change is below 5 10^-4). "
          "Defaults (Wolfram's are not documented): {mu, nu, lambda1, lambda2} = {0.2, 0, 1, 1}; marker Automatic "
          "starts from Getreuer's checkerboard sin(pi x/5) sin(pi y/5) and returns the phase with the larger mean "
          "intensity as foreground (1). A marker image or matrix (nonzero = marked) or a list of positions (disks of "
          "radius 3 pixels) selects the foreground: the evolution starts from the same checkerboard (with the "
          "regularised delta a flat marker-shaped start barely moves away from the marker contour) and the phase "
          "containing at least half of the marked pixels is returned as foreground; a colour fixes c1, {{fg, bg}} fixes c1 and c2 (the foreground is the region fitting fg). MaxIterations -> 100 (as "
          "documented). Image3D raises NotImplementedError, negative weights ValueError. Check: reference (tests: "
          "two-level images with noise are recovered exactly; a fixed colour pair selects the matching region).")
def _chanvesebinarize(x: Expr, *rest: Expr | ListOf[Real] | Option) -> Expr:
    fname = "ChanVeseBinarize"
    pos, o = opts_split(rest, {"MaxIterations"}, fname)
    if len(pos) > 2:
        raise TypeError(f"{fname}: expected ChanVeseBinarize[image, marker, {{mu, nu, lambda1, lambda2}}]")
    im = img2d(x, fname)
    f = im.color()
    h, w = f.shape[:2]
    mi = o.get("MaxIterations", 100)
    iters = int_arg(mi, fname, "MaxIterations", 1) if not _is_sym(mi, "Automatic") else 100
    mu, nu, l1, l2 = 0.2, 0.0, 1.0, 1.0
    if len(pos) == 2:
        p = pos[1]
        if not (K.is_list(p) and len(p.args) == 4):
            raise ValueError(f"{fname}: the weights must be given as {{mu, nu, lambda1, lambda2}}")
        mu, nu, l1, l2 = (_real(v, fname, "weight") for v in p.args)
        if mu < 0 or l1 < 0 or l2 < 0:
            raise ValueError(f"{fname}: mu, lambda1 and lambda2 must be non-negative")
    c1fix = c2fix = None
    yy, xx = np.mgrid[0:h, 0:w]
    phi = np.sin(math.pi * xx / 5) * np.sin(math.pi * yy / 5)
    auto = True
    marker = None
    if pos and not _is_sym(pos[0], "Automatic"):
        mk = pos[0]
        col = None
        if K.is_list(mk) and len(mk.args) == 1 and K.is_list(mk.args[0]) and len(mk.args[0].args) == 2 and \
                not is_point(mk.args[0]):
            col = [WI.value_for(im, v, fname)[:f.shape[-1]] for v in mk.args[0].args]
        elif not K.is_list(mk) and not (isinstance(mk, Expr) and mk.head in IMAGE_HEADS):
            try:
                col = [WI.value_for(im, mk, fname)[:f.shape[-1]]]
            except (TypeError, ValueError):
                col = None
        if col is not None:
            c1fix = np.asarray(col[0], dtype=float)
            if len(col) > 1:
                c2fix = np.asarray(col[1], dtype=float)
            auto = False
        else:
            m = marker_mask(mk, h, w, fname)
            if K.is_list(mk) and all(is_point(pp) for pp in mk.args) or is_point(mk):
                m = WP.dilate(m.astype(float), WP._disk_offsets(3.0, "EuclideanDistance"), 0.0) > 0
            _marker_check(m)
            marker = m
            auto = False
    phi = chan_vese(f, phi, mu, nu, l1, l2, iters, c1fix, c2fix)
    fgm = phi > 0
    if marker is not None and fgm[marker].mean() < 0.5:
        fgm = ~fgm
    if auto and fgm.any() and (~fgm).any():
        if gray(WI.Img(f))[fgm].mean() < gray(WI.Img(f))[~fgm].mean():
            fgm = ~fgm
    return binary_image(im, fgm)


# ───────────────────────────── GrowCutComponents ─────────────────────────────

def growcut(f, seeds, offs, iters=None):
    """GrowCut cellular automaton (Vezhnevets & Konouchine 2005): a cell q attacks p with force g(|C_p - C_q|)
    theta_q, g(x) = 1 - x/M (M the largest channel-vector distance in the image), all cells updated synchronously."""
    lab = seeds.copy()
    th = (seeds > 0).astype(float)
    span = f.reshape(-1, f.shape[-1])
    M = float(np.sqrt(((span.max(0) - span.min(0)) ** 2).sum()))
    k = 0
    while iters is None or k < iters:
        nl, nt = lab.copy(), th.copy()
        for o in offs:
            ql = _shift(lab, o, 0)
            qt = _shift(th, o, 0.0)
            qf = _shift(f, o, 0.0)
            dist = np.sqrt(((f - qf) ** 2).sum(-1))
            g = 1 - dist / M if M > 0 else np.ones(dist.shape)
            att = g * qt
            win = (ql > 0) & (att > nt + 1e-15)
            nl = np.where(win, ql, nl)
            nt = np.where(win, att, nt)
        k += 1
        if np.array_equal(nl, lab) and np.allclose(nt, th, atol=0, rtol=0):
            break
        lab, th = nl, nt
    return lab


@fn("GrowCutComponents", area=AREA, status="added",
    notes="GrowCutComponents[image, {marker1, marker2, ...}] gives a label matrix in which pixels grown from marker "
          "i are labelled i (unreached pixels 0), by the GrowCut cellular automaton (Vezhnevets & Konouchine 2005): "
          "seed cells start with strength 1, a neighbour q conquers p when g(|C_p - C_q|) theta_q > theta_p with "
          "g(x) = 1 - x/M, M the largest channel-vector distance in the image (stated normalisation), updated "
          "synchronously until nothing changes (MaxIterations -> Automatic) or for MaxIterations -> n steps. Each "
          "marker is an image or matrix (nonzero pixels) or a list of positions {x, y} (a single {x, y} is one "
          "point); a later marker does not overwrite an earlier one's seeds. CornerNeighbors -> True (default) or "
          "False. Image3D raises NotImplementedError, an empty marker ValueError. Check: reference (tests: two "
          "flat regions seeded once each are recovered exactly; label numbers follow the marker order).")
def _growcutcomponents(x: Expr, markers: ListOf[Expr], *opts: Option) -> Expr:
    fname = "GrowCutComponents"
    pos, o = opts_split(opts, {"CornerNeighbors", "MaxIterations"}, fname)
    if pos:
        raise TypeError(f"{fname}: expected GrowCutComponents[image, {{marker1, ...}}, options]")
    im = img2d(x, fname)
    if not K.is_list(markers):
        if isinstance(markers, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the markers must be a list {{marker1, marker2, ...}}")
    if not markers.args:
        raise ValueError(f"{fname}: at least one marker is required")
    h, w = im.h, im.w
    seeds = np.zeros((h, w), dtype=np.int64)
    for k, mk in enumerate(markers.args, 1):
        m = marker_mask(mk, h, w, fname)
        if not m.any():
            raise ValueError(f"{fname}: marker {k} has no pixels")
        seeds = np.where((seeds == 0) & m, k, seeds)
    mi = o.get("MaxIterations", AUTO)
    iters = None if _is_sym(mi, "Automatic") or _is_sym(mi, "Infinity") else int_arg(mi, fname, "MaxIterations", 1)
    offs = neighbour_offsets(2, corner_opt(o, fname), center=False)
    return _labels_out(growcut(im.color(), seeds, offs, iters))


# ───────────────────────────── ImageForestingComponents ─────────────────────────────

def ift(f, seeds, offs):
    """Image foresting transform (Falcao, Stolfi & Lotufo 2004) with the f_max path cost and arc weights
    |C_p - C_q|: every pixel gets the label of the seed with the minimal maximum-arc path (FIFO tie-breaking)."""
    h, w = f.shape[:2]
    lab = seeds.copy()
    cost = np.where(seeds > 0, 0.0, np.inf)
    done = np.zeros((h, w), dtype=bool)
    heap, cnt = [], 0
    for i, j in zip(*np.nonzero(seeds > 0)):
        heap.append((0.0, cnt, i, j))
        cnt += 1
    heapq.heapify(heap)
    while heap:
        c, _, i, j = heapq.heappop(heap)
        if done[i, j] or c > cost[i, j]:
            continue
        done[i, j] = True
        fp = f[i, j]
        for di, dj in offs:
            p, q = i + di, j + dj
            if 0 <= p < h and 0 <= q < w and not done[p, q]:
                nc = max(c, float(np.sqrt(((fp - f[p, q]) ** 2).sum())))
                if nc < cost[p, q]:
                    cost[p, q] = nc
                    lab[p, q] = lab[i, j]
                    heapq.heappush(heap, (nc, cnt, p, q))
                    cnt += 1
    return lab


def disk_offsets2(rr, rc):
    out = []
    for di in range(-int(math.floor(rr)), int(math.floor(rr)) + 1):
        for dj in range(-int(math.floor(rc)), int(math.floor(rc)) + 1):
            if (di or dj) and (di / rr) ** 2 + (dj / rc) ** 2 <= 1 + 1e-12:
                out.append((di, dj))
    return out


@fn("ImageForestingComponents", area=AREA, status="added",
    notes="ImageForestingComponents[image, marker, r] gives a complete label matrix by the image foresting "
          "transform (Falcao, Stolfi & Lotufo 2004): pixels are joined to all pixels within the elliptic radius r "
          "(default 2, as documented; {r_row, r_col}) by arcs weighted with the Euclidean distance of their "
          "channel vectors (0..1 scale), and every pixel takes the label of the seed reached by the path with the "
          "smallest maximum arc weight (f_max), ties first-come first-served. Seeds: a marker image or matrix (each "
          "8-connected component of nonzero pixels, labelled in scan order; values are not used as weights) or a "
          "list of positions (labelled in order). ImageForestingComponents[image] (marker Automatic) seeds every "
          "regional minimum plateau of the largest arc weight at each pixel (the IFT watershed). Image3D raises "
          "NotImplementedError, r = 0 ValueError. Check: reference (tests: flat regions are separated at their "
          "border; every pixel is labelled).")
def _imageforestingcomponents(x: Expr, *rest: Expr | Real | ListOf[Real]) -> Expr:
    fname = "ImageForestingComponents"
    pos, o = opts_split(rest, set(), fname)
    if len(pos) > 2:
        raise TypeError(f"{fname}: expected ImageForestingComponents[image, marker, r]")
    im = img2d(x, fname)
    f = im.color()
    h, w = im.h, im.w
    rr, rc = _radii2(pos[1] if len(pos) > 1 else 2, fname, "radius", allow_zero=False)
    if rr < 1 or rc < 1:
        raise ValueError(f"{fname}: the radius must be at least 1")
    offs = disk_offsets2(rr, rc)
    if pos and not _is_sym(pos[0], "Automatic"):
        seeds = seed_labels(pos[0], h, w, True, fname)
        if not seeds.any():
            raise ValueError(f"{fname}: the marker has no foreground pixels")
    else:
        grad = np.zeros((h, w))
        for o_ in offs:
            sf = _shift(f, o_, np.nan)
            dd = np.sqrt(((f - sf) ** 2).sum(-1))
            grad = np.fmax(grad, np.nan_to_num(dd, nan=0.0))
        seeds = label(regional_extrema(grad, neighbour_offsets(2, True), maxima=False), True)
    return _labels_out(ift(f, seeds, offs))


# ───────────────────────────── ImageSegmentationComponents / Filter ─────────────────────────────

def felzenszwalb(f, k=300 / 255, min_size=20, sigma=0.8):
    """Felzenszwalb-Huttenlocher (2004) graph segmentation: Gaussian presmoothing sigma, 8-neighbour grid graph
    with Euclidean channel distances, threshold k/|C|, then components smaller than min_size merged along the
    cheapest edges. Returns labels 1..n in scan order."""
    h, w = f.shape[:2]
    g = smooth(f, sigma, sigma, "Reflected") if sigma > 0 else f
    idx = np.arange(h * w).reshape(h, w)
    E = []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = idx[0:h - di, max(0, -dj):w - max(0, dj)]
        b = idx[di:h, max(0, dj):w - max(0, -dj)]
        ga = g.reshape(h * w, -1)[a.reshape(-1)]
        gb = g.reshape(h * w, -1)[b.reshape(-1)]
        wt = np.sqrt(((ga - gb) ** 2).sum(-1))
        E.append(np.stack([wt, a.reshape(-1), b.reshape(-1)], 1))
    E = np.concatenate(E)
    E = E[np.argsort(E[:, 0], kind="stable")]
    parent = list(range(h * w))
    size = [1] * (h * w)
    thr = [k] * (h * w)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    for wt, a, b in E:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb and wt <= thr[ra] and wt <= thr[rb]:
            parent[rb] = ra
            size[ra] += size[rb]
            thr[ra] = wt + k / size[ra]
    for wt, a, b in E:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb and (size[ra] < min_size or size[rb] < min_size):
            parent[rb] = ra
            size[ra] += size[rb]
    roots = np.array([find(u) for u in range(h * w)])
    _, first, inv = np.unique(roots, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return (order[inv] + 1).reshape(h, w)


_SEG_OPTS = {"PerformanceGoal", "TargetDevice"}


def _seg_opts(o, fname):
    pg = o.get("PerformanceGoal", AUTO)
    if not _is_sym(pg, "Automatic"):
        raise NotImplementedError(f"{fname}: PerformanceGoal -> {_short(pg)} is not supported (one fixed method)")
    td = o.get("TargetDevice", "CPU")
    if td != "CPU":
        raise NotImplementedError(f"{fname}: TargetDevice -> {_short(td)} is not supported (\"CPU\" only)")


def _rect_box(r, h, w, fname):
    """(row0, row1, col0, col1) inclusive pixel bounds of the pixels whose centres lie in Rectangle[...]."""
    if len(r.args) == 1:
        (x0, y0), (x1, y1) = _point(r.args[0], fname), (_point(r.args[0], fname)[0] + 1,
                                                       _point(r.args[0], fname)[1] + 1)
    elif len(r.args) == 2:
        (x0, y0), (x1, y1) = _point(r.args[0], fname), _point(r.args[1], fname)
    else:
        raise ValueError(f"{fname}: invalid rectangle {_short(r)}")
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    cols = [j for j in range(w) if x0 <= j + 0.5 <= x1]
    rows = [i for i in range(h) if y0 <= h - i - 0.5 <= y1]
    if not cols or not rows:
        raise ValueError(f"{fname}: the rectangle {_short(r)} contains no pixel centres")
    return rows[0], rows[-1], cols[0], cols[-1]


def _seg_markers(spec, fname):
    """List of markers, each ('pts', [positions]) or ('rect', Rectangle)."""
    def one(s):
        if isinstance(s, Expr) and s.head == "Rectangle":
            return ("rect", s)
        if is_point(s):
            return ("pts", [s])
        if K.is_list(s) and s.args and all(is_point(p) for p in s.args):
            return ("pts", list(s.args))
        if isinstance(s, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: {_short(s)} is not a position, list of positions or Rectangle")
    if is_point(spec) or (isinstance(spec, Expr) and spec.head == "Rectangle"):
        return [one(spec)]
    if K.is_list(spec) and spec.args:
        if all(is_point(p) for p in spec.args):
            return [("pts", [p]) for p in spec.args]
        return [one(s) for s in spec.args]
    if isinstance(spec, Symbol) or (isinstance(spec, Expr) and not K.is_list(spec)):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_short(spec)} is not a segmentation specification")


def _marker_region(mk, seg, h, w, fname):
    if mk[0] == "pts":
        labs = {seg[pixel_of(p, h, w, fname)] for p in mk[1]}
        return np.isin(seg, list(labs))
    r0, r1, c0, c1 = _rect_box(mk[1], h, w, fname)
    inside = np.zeros((h, w), dtype=bool)
    inside[r0:r1 + 1, c0:c1 + 1] = True
    out = np.zeros((h, w), dtype=bool)
    for k in np.unique(seg[inside]):
        comp = seg == k
        if (comp & inside).sum() >= 0.5 * comp.sum():
            out |= comp
    return out


@fn("ImageSegmentationComponents", area=AREA, status="added",
    notes="ImageSegmentationComponents[image] gives a label matrix of a global segmentation by the "
          "Felzenszwalb-Huttenlocher graph method (2004; Gaussian presmoothing sigma = 0.8, 8-neighbour grid with "
          "Euclidean channel distances on the 0..1 scale, k = 300/255, components under 20 pixels merged), labels in "
          "scan order. ImageSegmentationComponents[image, spec] labels one component per marker (1, 2, ... in order, "
          "an earlier marker winning where they overlap, other pixels 0): a position {x, y} or a list of positions "
          "selects the segments containing them ({pos1, pos2, ...}: one component each; {{pos11, ...}, ...}: one per "
          "list), Rectangle[...] (or a list of them) the segments with at least half their pixels inside the box. "
          "This is a classical method: Wolfram's function uses a trained neural network, so results differ. "
          "PerformanceGoal -> Automatic and TargetDevice -> \"CPU\" only; Image3D raises NotImplementedError. "
          "Check: reference (tests: flat regions give one label each; markers select their region).")
def _imagesegmentationcomponents(x: Expr, *rest: Expr | Option) -> Expr:
    fname = "ImageSegmentationComponents"
    pos, o = opts_split(rest, _SEG_OPTS, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected ImageSegmentationComponents[image, spec, options]")
    im = img2d(x, fname)
    _seg_opts(o, fname)
    seg = felzenszwalb(im.color())
    if not pos:
        return _labels_out(seg)
    h, w = im.h, im.w
    out = np.zeros((h, w), dtype=np.int64)
    for k, mk in enumerate(_seg_markers(pos[0], fname), 1):
        reg = _marker_region(mk, seg, h, w, fname)
        out = np.where((out == 0) & reg, k, out)
    return _labels_out(out)


def random_walker(f, fg, bg, beta=90.0, tol=1e-8, maxit=5000):
    """Probability that a random walker from each pixel first reaches a foreground seed (Grady 2006): 4-neighbour
    weights exp(-beta |dC|^2 / max |dC|^2), the Dirichlet problem solved by conjugate gradients."""
    h, w = f.shape[:2]
    dh = ((f[:, 1:] - f[:, :-1]) ** 2).sum(-1)
    dv = ((f[1:] - f[:-1]) ** 2).sum(-1)
    mx = max(float(dh.max()) if dh.size else 0.0, float(dv.max()) if dv.size else 0.0)
    sc = beta / mx if mx > 0 else 0.0
    wh = np.exp(-sc * dh) + 1e-10
    wv = np.exp(-sc * dv) + 1e-10
    deg = np.zeros((h, w))
    deg[:, 1:] += wh
    deg[:, :-1] += wh
    deg[1:] += wv
    deg[:-1] += wv
    known = fg | bg
    unk = ~known

    def Lmul(u):
        r = deg * u
        r[:, 1:] -= wh * u[:, :-1]
        r[:, :-1] -= wh * u[:, 1:]
        r[1:] -= wv * u[:-1]
        r[:-1] -= wv * u[1:]
        return r
    x0 = fg.astype(float)
    b = -Lmul(x0)
    b[known] = 0
    x = np.zeros((h, w))
    r = b.copy()
    p = r.copy()
    rs = float((r * r).sum())
    bn = math.sqrt(float((b * b).sum())) or 1.0
    for _ in range(maxit):
        if math.sqrt(rs) <= tol * bn:
            break
        Ap = Lmul(p)
        Ap[known] = 0
        alpha = rs / float((p * Ap).sum())
        x += alpha * p
        r -= alpha * Ap
        rn = float((r * r).sum())
        p = r + (rn / rs) * p
        rs = rn
    u = np.where(unk, x, x0)
    res = Lmul(u)
    res[known] = 0
    if math.sqrt(float((res * res).sum())) > 1e-6 * max(1.0, bn):
        raise ArithmeticError("ImageSegmentationFilter: result failed verification")
    return np.clip(u, 0.0, 1.0)


@fn("ImageSegmentationFilter", area=AREA, status="added",
    notes="ImageSegmentationFilter[image, marker] gives a single-channel \"Real32\" mask: the random-walker "
          "probability (Grady 2006; 4-neighbour weights exp(-90 |dC|^2/max |dC|^2) on the 0..1 scale) that a walker "
          "from each pixel reaches the foreground seeds before the background seeds. Markers: a position {x, y} or a "
          "list of positions (foreground seeds; the background seeds are the image border pixels outside the Felzenszwalb-Huttenlocher segments (as in ImageSegmentationComponents) that contain the positions) or Rectangle[...] (the "
          "pixels outside the box are background seeds and the central third of the box is foreground). This is a "
          "classical method: Wolfram's function uses a trained neural network, so results differ. PerformanceGoal "
          "-> Automatic and TargetDevice -> \"CPU\" only; Image3D raises NotImplementedError. Check: self-check (the "
          "harmonic Dirichlet problem's residual is verified; ArithmeticError otherwise), and tests (two flat "
          "regions give a mask close to 1 on the marked region and 0 elsewhere).")
def _imagesegmentationfilter(x: Expr, marker: Expr | ListOf[Expr], *opts: Option) -> Expr:
    fname = "ImageSegmentationFilter"
    pos, o = opts_split(opts, _SEG_OPTS, fname)
    if pos:
        raise TypeError(f"{fname}: expected ImageSegmentationFilter[image, marker, options]")
    im = img2d(x, fname)
    _seg_opts(o, fname)
    h, w = im.h, im.w
    fg = np.zeros((h, w), dtype=bool)
    bg = np.zeros((h, w), dtype=bool)
    if isinstance(marker, Expr) and marker.head == "Rectangle":
        r0, r1, c0, c1 = _rect_box(marker, h, w, fname)
        bg[:, :] = True
        bg[r0:r1 + 1, c0:c1 + 1] = False
        hr, hc = (r1 - r0 + 1), (c1 - c0 + 1)
        fg[r0 + hr // 3:r1 + 1 - hr // 3, c0 + hc // 3:c1 + 1 - hc // 3] = True
        if not bg.any():
            raise ValueError(f"{fname}: the rectangle covers the whole image")
    else:
        mks = _seg_markers(marker, fname)
        if any(m[0] != "pts" for m in mks):
            raise NotImplementedError(f"{fname}: give positions or a single Rectangle")
        for m in mks:
            for p in m[1]:
                fg[pixel_of(p, h, w, fname)] = True
        seg = felzenszwalb(im.color())
        bg[0, :] = bg[-1, :] = True
        bg[:, 0] = bg[:, -1] = True
        bg &= ~np.isin(seg, np.unique(seg[fg]))
        if not bg.any():
            raise ValueError(f"{fname}: no background seeds remain (the marked segment covers the image border)")
    u = random_walker(im.color(), fg, bg)
    return image_out(im, u[..., None], typ="Real32", cs="Automatic")


# ───────────────────────────── ImageLines (Hough) ─────────────────────────────

def _chord(theta, rho, w, h):
    """Endpoints of the line x cos(theta) + y sin(theta) = rho inside [0, w] x [0, h] (None if it misses)."""
    c, s = math.cos(theta), math.sin(theta)
    pts = []
    if abs(s) > 1e-12:
        for xx in (0.0, float(w)):
            yy = (rho - xx * c) / s
            if -1e-9 <= yy <= h + 1e-9:
                pts.append((xx, min(max(yy, 0.0), float(h))))
    if abs(c) > 1e-12:
        for yy in (0.0, float(h)):
            xx = (rho - yy * s) / c
            if -1e-9 <= xx <= w + 1e-9:
                pts.append((min(max(xx, 0.0), float(w)), yy))
    uniq = []
    for p in pts:
        if all(abs(p[0] - q[0]) > 1e-9 or abs(p[1] - q[1]) > 1e-9 for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None
    best = max(((a, b) for a in uniq for b in uniq), key=lambda ab: math.dist(ab[0], ab[1]))
    return tuple(sorted(best))


def hough_lines(g, t, d, nmax=None, ntheta=180):
    """[(strength, (p1, p2))]: lines from the intensity-weighted Hough transform (Duda & Hart 1972; theta step
    pi/ntheta, rho step 1), strength = votes / chord length (clipped to 1), local maxima above t, greedy
    suppression within d pi/2 in theta and d D in rho, rho refined to the weighted mean of the voting pixels."""
    h, w = g.shape
    D = math.hypot(w, h)
    nr = int(math.ceil(2 * D)) + 1
    th = np.arange(ntheta) * math.pi / ntheta
    rr, cc = np.nonzero(g > 0)
    wts = g[rr, cc]
    xs, ys = cc + 0.5, h - rr - 0.5
    acc = np.zeros((ntheta, nr))
    rho_all = xs[None, :] * np.cos(th)[:, None] + ys[None, :] * np.sin(th)[:, None]
    bins = np.rint(rho_all + D).astype(int)
    for k in range(ntheta):
        np.add.at(acc[k], bins[k], wts)
    rhos = np.arange(nr) - D
    length = np.zeros_like(acc)
    for k in range(ntheta):
        for j in range(nr):
            if acc[k, j] > 0:
                ch = _chord(th[k], rhos[j], w, h)
                length[k, j] = math.dist(*ch) if ch else 0.0
    minlen = max(5.0, 0.1 * min(w, h))
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(length >= minlen, np.minimum(acc / np.where(length > 0, length, 1.0), 1.0), 0.0)
    P = np.pad(S, 1, constant_values=-1.0)
    loc = np.ones(S.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                loc &= S >= P[1 + di:1 + di + ntheta, 1 + dj:1 + dj + nr]
    cand = np.argwhere(loc & (S > t))
    cand = sorted(cand.tolist(), key=lambda kj: (-S[kj[0], kj[1]], kj[0], kj[1]))
    chosen = []
    for k, j in cand:
        ok = True
        for k2, j2 in chosen:
            dth = abs(th[k] - th[k2])
            same = dth <= d * math.pi / 2 + 1e-12 and abs(rhos[j] - rhos[j2]) <= d * D + 1e-12
            wrap = math.pi - dth <= d * math.pi / 2 + 1e-12 and abs(rhos[j] + rhos[j2]) <= d * D + 1e-12
            if same or wrap or (k, j) == (k2, j2):
                ok = False
                break
        if ok:
            chosen.append((k, j))
            if nmax is not None and len(chosen) >= nmax:
                break
    out = []
    for k, j in chosen:
        sel = np.abs(rho_all[k] - rhos[j]) <= 1.0
        rho = float((rho_all[k][sel] * wts[sel]).sum() / wts[sel].sum()) if sel.any() else rhos[j]
        ch = _chord(th[k], rho, w, h)
        if ch:
            out.append((float(S[k, j]), ch))
    return out


@fn("ImageLines", area=AREA, status="added",
    notes="ImageLines[image], ImageLines[image, t] and ImageLines[image, t, d] give Line[{{x1, y1}, {x2, y2}}] "
          "segments spanning the image (endpoints on its border, standard image coordinates, sorted by x then y), "
          "strongest first, from the intensity-weighted Hough transform (Duda & Hart 1972; theta step 1 degree, rho "
          "step 1 pixel, intensity = mean over channels, negative values ignored). The normalised strength of a line "
          "is its accumulated intensity divided by its chord length in the image (1 for a fully white line; chords "
          "shorter than max(5, 10% of the smaller side) are skipped); lines are local accumulator maxima with "
          "strength > t (default 0.3), and a line is dropped when an already selected stronger line lies within d "
          "pi/2 in angle and d times the image diagonal in rho (d default 0.05; d = 1 keeps only the strongest), as "
          "documented; rho is refined to the intensity-weighted mean over the pixels within 1 pixel of the peak. "
          "Wolfram's normalisation and defaults are not documented. MaxFeatures -> n; Method -> \"Hough\". "
          "\"RANSAC\", \"Segmented\" -> True and Image3D raise NotImplementedError. Check: reference (tests: "
          "horizontal, vertical and diagonal lines are found at their exact positions with strength 1).")
def _imagelines(x: Expr, *rest: Real | Option) -> ListOf[Expr]:
    fname = "ImageLines"
    pos, o = opts_split(rest, {"MaxFeatures", "Method"}, fname)
    if len(pos) > 2:
        raise TypeError(f"{fname}: expected ImageLines[image, t, d, options]")
    im = img2d(x, fname)
    t = 0.3 if not pos or _is_sym(pos[0], "Automatic") else _real(pos[0], fname, "threshold")
    d = 0.05 if len(pos) < 2 or _is_sym(pos[1], "Automatic") else _real(pos[1], fname, "distinctness")
    if not 0 <= d <= 1:
        raise ValueError(f"{fname}: the distinctness d must be between 0 and 1")
    name, extra, sub = _method(o, fname, "Hough")
    if name != "Hough" or extra:
        raise NotImplementedError(f"{fname}: Method {name!r} is not supported (\"Hough\")")
    seg = sub.pop("Segmented", FALSE)
    if _truthy(seg, fname, "\"Segmented\""):
        raise NotImplementedError(f"{fname}: \"Segmented\" -> True is not supported")
    _no_sub(sub, fname, name)
    mf = o.get("MaxFeatures", Symbol("All"))
    nmax = None if _is_sym(mf, "All") or _is_sym(mf, "Infinity") else int_arg(mf, fname, "MaxFeatures", 1)
    g = np.maximum(gray(im), 0.0)
    lines = hough_lines(g, t, d, nmax)
    return L(*[Expr("Line", L(L(*[_clean(v) for v in p]), L(*[_clean(v) for v in q]))) for _, (p, q) in lines])


def _clean(v):
    r = round(float(v), 9)
    return 0.0 if r == 0 else r


# ───────────────────────────── ImageMesh ─────────────────────────────

def pixel_loops(fg):
    """Boundary loops (lists of integer image-coordinate corners, counterclockwise around the foreground, holes
    clockwise) of the union of foreground pixel squares; collinear vertices removed."""
    h, w = fg.shape
    out_edges = {}
    P = np.pad(fg, 1, constant_values=False)
    for i in range(h):
        for j in range(w):
            if not fg[i, j]:
                continue
            yb, yt = h - i - 1, h - i
            if not P[i + 2, j + 1]:
                out_edges.setdefault((j, yb), []).append((j + 1, yb))
            if not P[i + 1, j + 2]:
                out_edges.setdefault((j + 1, yb), []).append((j + 1, yt))
            if not P[i, j + 1]:
                out_edges.setdefault((j + 1, yt), []).append((j, yt))
            if not P[i + 1, j]:
                out_edges.setdefault((j, yt), []).append((j, yb))
    used = set()
    loops = []
    for start in sorted(out_edges):
        for nxt0 in out_edges[start]:
            if (start, nxt0) in used:
                continue
            loop = [start]
            cur, nxt = start, nxt0
            used.add((cur, nxt))
            while nxt != start or len(loop) < 2:
                loop.append(nxt)
                cands = [c for c in out_edges[nxt] if (nxt, c) not in used]
                if not cands:
                    break
                if len(cands) > 1:
                    dx, dy = nxt[0] - cur[0], nxt[1] - cur[1]
                    right = (dy, -dx)
                    left = (-dy, dx)
                    want = left
                    cands = [c for c in cands if (c[0] - nxt[0], c[1] - nxt[1]) == want] or cands
                cur, nxt = nxt, cands[0]
                used.add((cur, nxt))
                if nxt == start:
                    break
            simp = []
            n = len(loop)
            for k in range(n):
                a, b, c = loop[k - 1], loop[k], loop[(k + 1) % n]
                if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
                    simp.append(b)
            loops.append(simp)
    return loops


def _shoelace(lp):
    return sum(lp[k - 1][0] * lp[k][1] - lp[k][0] * lp[k - 1][1] for k in range(len(lp))) / 2


@fn("ImageMesh", area=AREA, status="added",
    notes="ImageMesh[image] gives the foreground (nonzero pixels, mean over channels) as a BoundaryMeshRegion: the "
          "exact union of the pixel squares (Method -> \"Exact\" or Automatic), traced along pixel edges with one "
          "Line loop per boundary (outer boundaries counterclockwise, holes clockwise, collinear vertices removed) "
          "in image coordinates (DataRange -> Automatic: {{0, w}, {0, h}} in pixels, or explicit {{x0, x1}, {y0, "
          "y1}}). Pixels that touch only diagonally get separate boundary loops (the region, the closed union of the "
          "squares, is the same point set either way); CornerNeighbors -> True (default) makes such loops share "
          "the common corner vertex, False gives each loop its own copy, so the mesh is valid for Area, "
          "RegionMember, ... in both cases. An empty foreground gives "
          "EmptyRegion[2]. \"MarchingSquares\", \"DualMarchingSquares\", \"LinearSeparable\" and Image3D raise "
          "NotImplementedError. Check: self-check (the signed area of the loops must equal the foreground pixel "
          "count; ArithmeticError otherwise) and tests (Area and RegionMember of the mesh).")
def _imagemesh(x: Expr, *opts: Option) -> Expr:
    fname = "ImageMesh"
    pos, o = opts_split(opts, {"Method", "DataRange", "CornerNeighbors"}, fname)
    if pos:
        raise TypeError(f"{fname}: expected ImageMesh[image, options]")
    im = img2d(x, fname)
    m = o.get("Method", AUTO)
    if not (_is_sym(m, "Automatic") or m == "Exact"):
        raise NotImplementedError(f"{fname}: Method {_short(m)} is not supported (\"Exact\")")
    fg = np.abs(im.color()).max(-1) > 0
    if not fg.any():
        return Expr("EmptyRegion", 2)
    corner = corner_opt(o, fname)
    loops = pixel_loops(fg)
    if abs(sum(_shoelace(lp) for lp in loops) - int(fg.sum())) > 1e-9:
        raise ArithmeticError(f"{fname}: result failed verification")
    dr = o.get("DataRange", AUTO)
    if _is_sym(dr, "Automatic"):
        conv = lambda p: L(int(p[0]), int(p[1]))       # noqa: E731
    elif K.is_list(dr) and len(dr.args) == 2 and all(K.is_list(v) and len(v.args) == 2 for v in dr.args):
        (a, b), (c, e) = [[_real(t, fname, "DataRange") for t in v.args] for v in dr.args]
        if a == b or c == e:
            raise ValueError(f"{fname}: DataRange must have non-empty ranges")
        conv = lambda p: L(a + (b - a) * p[0] / im.w, c + (e - c) * p[1] / im.h)    # noqa: E731
    else:
        raise ValueError(f"{fname}: invalid DataRange {_short(dr)}")
    pts, cells, index = [], [], {}
    for lp in loops:
        ids = []
        for p in lp:
            if not corner or p not in index:
                pts.append(conv(p))
                index[p] = len(pts)
                ids.append(len(pts))
            else:
                ids.append(index[p])
        cells.append(Expr("Line", L(*ids, ids[0])))
    return _ev(Expr("BoundaryMeshRegion", L(*pts), *cells))
