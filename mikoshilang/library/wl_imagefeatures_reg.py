"""Wolfram batch X05: images IV, part 2 (keypoints, correspondences, tracking, optical flow, geometric
registration, alignment and stitching).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md). Wolfram does not document its keypoint, matching, tracking, flow or registration algorithms
or their Automatic defaults, so each function here uses a named published method and states its defaults:
FAST (Rosten & Drummond 2006) and ORB (Rublee et al. 2011) keypoints, Hamming matching with Lowe's ratio test,
pyramidal Lucas-Kanade tracking (Bouguet 2000), Farneback (2003) dense flow, closed-form least-squares fits
(Umeyama 1991, Hartley's normalised DLT) and RANSAC (Fischler & Bolles 1981), phase correlation (Kuglin & Hines
1975). Results can differ from Wolfram's in detail.

Positions use the standard image coordinate system ({0, 0} bottom-left, pixel centres at half-integers).
Pure NumPy does everything here; nothing needs SciPy or Pillow. Random sampling (RANSAC, the BRIEF test pattern)
uses fixed seeds, so results are reproducible.
"""
from __future__ import annotations

import math

import numpy as np

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, record
from . import wl_images as WI
from .wl_images import parse_image, image_expr, opts_split, int_arg, IMAGE_HEADS, _short, _is_sym, quantize
from .wl_imageproc import _shift, label, neighbour_offsets
from .wl_imageproc2 import gaussian_levels, pyr_reduce, pyr_expand, correlate_axis_padded, _mask_of
from .wl_imagefeatures import img2d, gray, structure_tensor, corner_measure, smooth, _method, _no_sub, _real, \
    is_point, _point

AREA = "image registration (X05)"
AUTO = Symbol("Automatic")
ALL = Symbol("All")
NA = Expr("Missing", "NotAvailable")


def _opt_int(o, key, fname, default, lo=1):
    """Integer option: default when absent; None (no limit) for All, Infinity or Automatic."""
    if key not in o:
        return default
    v = o[key]
    if _is_sym(v, "All") or _is_sym(v, "Automatic") or _is_sym(v, "Infinity"):
        return None
    return int_arg(v, fname, key, lo)


def _clean(v):
    v = float(v)
    r = round(v, 12)
    return 0.0 if abs(v) < 1e-12 else (r if abs(r - v) < 1e-12 else v)


def _vec(p):
    return L(*[_clean(t) for t in p])


# ───────────────────────────── FAST and ORB keypoints ─────────────────────────────

CIRCLE = [(-3, 0), (-3, 1), (-2, 2), (-1, 3), (0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
          (0, -3), (-1, -3), (-2, -2), (-3, -1)]
FAST_T = 0.04          # FAST threshold on the 0..1 scale (about OpenCV's 10/255)


def fast_score(g, arc=9, t=FAST_T):
    """FAST-9 (Rosten & Drummond 2006): a pixel is a corner when 9 contiguous pixels of the radius-3 Bresenham
    circle are all brighter, or all darker, than it by more than t; the score of a corner is Rosten's
    V = max(sum over the brighter pixels of (|I - p| - t), sum over the darker ones), -inf for non-corners and
    within 3 pixels of the border."""
    h, w = g.shape
    D = np.nan_to_num(np.stack([_shift(g, k, np.nan) - g for k in CIRCLE]), nan=0.0)
    D2 = np.concatenate([D, D[:arc - 1]])
    corner = np.zeros((h, w), dtype=bool)
    for s in range(16):
        seg = D2[s:s + arc]
        corner |= (seg.min(0) > t) | ((-seg).min(0) > t)
    V = np.maximum(np.where(D > t, D - t, 0).sum(0), np.where(-D > t, -D - t, 0).sum(0))
    S = np.where(corner, V, -np.inf)
    S[:3] = S[-3:] = -np.inf
    S[:, :3] = S[:, -3:] = -np.inf
    return S


def local_max(S, border=0):
    """Strict-in-scan-order 3 x 3 local maxima of S (ties go to the first pixel)."""
    keep = np.isfinite(S)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                n = _shift(S, (di, dj), -np.inf)
                keep &= (S > n) if (di < 0 or (di == 0 and dj < 0)) else (S >= n)
    if border:
        keep[:border] = keep[-border:] = False
        keep[:, :border] = keep[:, -border:] = False
    return keep


def _brief_pairs(n=256, S=31, seed=20111106):
    """BRIEF test pairs (Calonder et al. 2010, pattern G II): coordinates from an isotropic Gaussian with variance
    S^2/25, rounded, kept inside the radius-15 disk; fixed seed."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        p = np.rint(rng.normal(0, S / 5, 4))
        if p[0] ** 2 + p[1] ** 2 <= 225 and p[2] ** 2 + p[3] ** 2 <= 225:
            out.append(p)
    return np.array(out)


PAIRS = _brief_pairs()
_DISK = [(di, dj) for di in range(-15, 16) for dj in range(-15, 16) if di * di + dj * dj <= 225]
_DI = np.array([d[0] for d in _DISK])
_DJ = np.array([d[1] for d in _DISK])
ORB_BORDER = 19
ORB_LEVELS = 4


class KP:
    __slots__ = ("x", "y", "conf", "scale", "orient", "desc", "odesc", "level")

    def __init__(self, x, y, conf, scale, orient=None, desc=None, odesc=None, level=0):
        self.x, self.y, self.conf, self.scale = x, y, conf, scale
        self.orient, self.desc, self.odesc, self.level = orient, desc, odesc, level


def _brief(s, i, j, theta):
    c, sn = math.cos(theta), math.sin(theta)
    x1, y1, x2, y2 = PAIRS[:, 0], PAIRS[:, 1], PAIRS[:, 2], PAIRS[:, 3]
    j1 = np.rint(c * x1 - sn * y1).astype(int)
    i1 = -np.rint(sn * x1 + c * y1).astype(int)
    j2 = np.rint(c * x2 - sn * y2).astype(int)
    i2 = -np.rint(sn * x2 + c * y2).astype(int)
    return (s[i + i1, j + j1] < s[i + i2, j + j2]).astype(np.uint8)


def keypoints(im, method="ORB", mask=None):
    """List of KP of a 2D Img, strongest first (FAST: FAST score; ORB: Harris response)."""
    g0 = gray(im)
    h0 = g0.shape[0]
    out = []
    if method == "FAST":
        S = fast_score(g0)
        sel = local_max(S)
        for i, j in zip(*np.nonzero(sel)):
            if mask is None or mask[i, j]:
                out.append(KP(j + 0.5, h0 - i - 0.5, float(S[i, j]), 3.5))
    else:
        nlev = 1
        while nlev < ORB_LEVELS and min(math.ceil(s / 2 ** nlev) for s in g0.shape) >= 2 * ORB_BORDER + 1:
            nlev += 1
        levels = gaussian_levels(g0[..., None], 2, nlev)
        for lev, gl in enumerate(levels):
            g = gl[..., 0]
            if min(g.shape) < 2 * ORB_BORDER + 1:
                break
            S = fast_score(g)
            sel = local_max(S, ORB_BORDER)
            if not sel.any():
                continue
            jxx, jxy, jyy = structure_tensor(g[..., None], (1.0, 1.0), (2.0, 2.0), "Reflected")
            H = corner_measure(jxx, jxy, jyy, "HarrisStephens", 0.04)
            sm = smooth(g, 2.0, 2.0, "Reflected")
            f = 2 ** lev
            for i, j in zip(*np.nonzero(sel)):
                i0, j0 = i * f, j * f
                if mask is not None and not mask[min(i0, mask.shape[0] - 1), min(j0, mask.shape[1] - 1)]:
                    continue
                patch = g[i + _DI, j + _DJ]
                th = math.atan2(float((-_DI * patch).sum()), float((_DJ * patch).sum()))
                out.append(KP(j0 + 0.5, h0 - i0 - 0.5, float(H[i, j]), 15.0 * f, th, _brief(sm, i, j, th),
                              _brief(sm, i, j, 0.0), lev))
    out.sort(key=lambda k: (-k.conf, k.level, -k.y, k.x))
    return out


_KP_METHODS = ("ORB", "FAST")
_KP_OTHER = ("SURF", "SIFT", "RootSIFT", "KAZE", "AKAZE", "BRISK", "AGAST")
_PROPS = ("Position", "PixelPosition", "Confidence", "Scale", "Orientation", "Descriptor", "OrientedDescriptor",
          "ContrastSign")


def kp_method(o, fname, allow_list=False):
    m = o.get("Method", AUTO)
    if _is_sym(m, "Automatic"):
        return "ORB"
    if K.is_list(m) and len(m.args) == 1 and isinstance(m.args[0], str):
        m = m.args[0]
    if isinstance(m, str):
        if m in _KP_METHODS:
            return m
        if m in _KP_OTHER:
            raise NotImplementedError(f"{fname}: Method {m!r} is not supported (\"ORB\" or \"FAST\")")
    raise ValueError(f"{fname}: unknown keypoint method {_short(m)}")


def _prop(k, p, h):
    if p == "Position":
        return _vec((k.x, k.y))
    if p == "PixelPosition":
        return _vec((k.x + 0.5, k.y + 0.5))
    if p == "Confidence":
        return _clean(k.conf)
    if p == "Scale":
        return _clean(k.scale)
    if p == "Orientation":
        return NA if k.orient is None else _clean(k.orient)
    if p == "Descriptor":
        return NA if k.desc is None else L(*[int(b) for b in k.desc])
    if p == "OrientedDescriptor":
        return NA if k.odesc is None else L(*[int(b) for b in k.odesc])
    return NA


def _strength(o, fname, default=0.0):
    s = o.get("KeypointStrength", default)
    if _is_sym(s, "Automatic"):
        return 0.0
    v = _real(s, fname, "KeypointStrength")
    if v < 0:
        raise ValueError(f"{fname}: KeypointStrength must be non-negative")
    return v


def _masking(o, im, fname):
    m = o.get("Masking", ALL)
    if _is_sym(m, "All") or _is_sym(m, "Full"):
        return None
    return _mask_of(m, im, fname)


_KP_OPTS = {"KeypointStrength", "Masking", "MaxFeatures", "Method"}


@fn("ImageKeypoints", area=AREA, status="added",
    notes="ImageKeypoints[image] gives the keypoint positions {x, y} (standard image coordinates), strongest "
          "first; ImageKeypoints[image, prop] or {prop1, ...} gives \"Position\", \"PixelPosition\" (Position + "
          "1/2, as documented), \"Confidence\", \"Scale\", \"Orientation\" (radians, counterclockwise from the x "
          "axis), \"Descriptor\", \"OrientedDescriptor\" or \"ContrastSign\", with Missing[\"NotAvailable\"] where "
          "a method has no such property. Methods: \"ORB\" (Rublee et al. 2011; used for Automatic because "
          "Wolfram's default SURF is not available): FAST-9 corners (threshold 0.04 on the 0..1 scale, 3 x 3 "
          "non-maximum suppression) on a 4-level Burt-Adelson pyramid at least 19 pixels from the border, ranked by "
          "the Harris response (k = 0.04, derivative sigma 1, integration sigma 2) = Confidence, orientation by the "
          "intensity centroid of the radius-15 disk, Scale 15 2^level, Descriptor the 256-bit steered BRIEF "
          "(Gaussian G II test pattern with a fixed seed, not OpenCV's learned table; Gaussian presmoothing sigma "
          "2), OrientedDescriptor the unsteered one; \"FAST\" (Rosten & Drummond 2006): FAST-9 corners, Confidence "
          "Rosten's score V (the summed absolute differences beyond the threshold), Scale 3.5 (as documented). KeypointStrength -> s keeps Confidence >= s, MaxFeatures -> n "
          "the n strongest, Masking -> mask image, matrix or region. SURF, SIFT, RootSIFT, KAZE, AKAZE, BRISK, "
          "AGAST, Video input and Image3D raise NotImplementedError. Check: reference (tests: FAST corners of a "
          "square are its four corners; ORB descriptors match under a quarter-turn rotation; orientation follows "
          "the rotation).")
def _imagekeypoints(x: Expr, *rest: String | ListOf[String] | Option) -> ListOf[Expr]:
    fname = "ImageKeypoints"
    pos, o = opts_split(rest, _KP_OPTS, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected ImageKeypoints[image, prop, options]")
    im = img2d(x, fname)
    props, single = ["Position"], True
    if pos:
        p = pos[0]
        if isinstance(p, str):
            props = [p]
        elif K.is_list(p) and p.args and all(isinstance(t, str) for t in p.args):
            props, single = list(p.args), False
        else:
            raise TypeError(f"{fname}: the property must be a string or a list of strings, not {_short(p)}")
        bad = [t for t in props if t not in _PROPS]
        if bad:
            raise ValueError(f"{fname}: unknown property {bad[0]!r}")
    method = kp_method(o, fname)
    s = _strength(o, fname)
    nmax = _opt_int(o, "MaxFeatures", fname, None)
    kps = [k for k in keypoints(im, method, _masking(o, im, fname)) if k.conf >= s]
    if nmax is not None:
        kps = kps[:nmax]
    if single:
        return L(*[_prop(k, props[0], im.h) for k in kps])
    return L(*[L(*[_prop(k, p, im.h) for p in props]) for k in kps])


# ───────────────────────────── matching ─────────────────────────────

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
MAX_MATCH_FEATURES = 1000
RATIO = 0.8
MAX_HAMMING = 64


def match(k1, k2):
    """[(i1, i2)] mutual nearest neighbours in Hamming distance passing Lowe's ratio test (0.8) and distance <= 64."""
    if not k1 or not k2:
        return []
    D1 = np.packbits(np.array([k.desc for k in k1], dtype=np.uint8), axis=1)
    D2 = np.packbits(np.array([k.desc for k in k2], dtype=np.uint8), axis=1)
    H = _POP[D1[:, None, :] ^ D2[None, :, :]].sum(-1)
    out = []
    best2 = H.argmin(0)
    for i in range(len(k1)):
        row = H[i]
        j = int(row.argmin())
        if best2[j] != i or row[j] > MAX_HAMMING:
            continue
        if len(row) > 1:
            second = np.partition(row, 1)[1]
            if row[j] >= RATIO * second:
                continue
        out.append((i, j))
    return out


def correspondences(im1, im2, method="ORB", mask=None, strength=0.0):
    """(P1, P2, conf) arrays of matched keypoint positions (standard image coordinates)."""
    if method != "ORB":
        raise NotImplementedError("ImageCorrespondingPoints: only ORB keypoints have descriptors for matching")
    k1 = [k for k in keypoints(im1, "ORB", mask) if k.conf >= strength][:MAX_MATCH_FEATURES]
    k2 = [k for k in keypoints(im2, "ORB") if k.conf >= strength][:MAX_MATCH_FEATURES]
    m = match(k1, k2)
    P1 = np.array([[k1[i].x, k1[i].y] for i, _ in m]).reshape(-1, 2)
    P2 = np.array([[k2[j].x, k2[j].y] for _, j in m]).reshape(-1, 2)
    conf = np.array([(k1[i].conf + k2[j].conf) / 2 for i, j in m])
    return P1, P2, conf


# ───────────────────────────── geometric fitting ─────────────────────────────

CLASSES = ("Translation", "Rigid", "Similarity", "Affine", "Perspective")


def min_points(cls, n):
    return {"Translation": 1, "Rigid": max(2, n - 1 + 1) if n > 2 else 2, "Similarity": max(2, n) if n > 2 else 2,
            "Affine": n + 1, "Perspective": n + 2}[cls]


def apply_h(M, P):
    n = P.shape[1]
    Q = np.hstack([P, np.ones((len(P), 1))]) @ M.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return Q[:, :n] / Q[:, n:]


def mean_err(M, P1, P2):
    Q = apply_h(M, P2)
    d = np.sqrt(((Q - P1) ** 2).sum(1))
    return float(d.mean()) if np.all(np.isfinite(d)) else math.inf


def fit(P1, P2, cls):
    """Homogeneous matrix M (least squares) with M.P2 ~ P1 for the class, or None if the data are degenerate."""
    N, n = P1.shape
    if N < min_points(cls, n):
        return None
    M = np.eye(n + 1)
    c1, c2 = P1.mean(0), P2.mean(0)
    if cls == "Translation":
        M[:n, n] = c1 - c2
        return M
    if cls in ("Rigid", "Similarity"):
        X, Y = P2 - c2, P1 - c1
        if np.linalg.matrix_rank(X, tol=1e-12 * max(1.0, np.abs(X).max())) < max(1, n - 1):
            return None
        S = Y.T @ X / N
        U, D, Vt = np.linalg.svd(S)
        E = np.eye(n)
        E[-1, -1] = 1.0 if np.linalg.det(U) * np.linalg.det(Vt) > 0 else -1.0
        Rm = U @ E @ Vt
        s = 1.0
        if cls == "Similarity":
            var = (X ** 2).sum() / N
            s = float(np.trace(np.diag(D) @ E) / var)
        M[:n, :n] = s * Rm
        M[:n, n] = c1 - s * Rm @ c2
        return M
    if cls == "Affine":
        A = np.hstack([P2, np.ones((N, 1))])
        if np.linalg.matrix_rank(A) < n + 1:
            return None
        Xs, *_ = np.linalg.lstsq(A, P1, rcond=None)
        M[:n, :] = Xs.T
        return M
    # Perspective: Hartley-normalised DLT
    def norm(P):
        c = P.mean(0)
        d = np.sqrt(((P - c) ** 2).sum(1)).mean()
        s = math.sqrt(n) / d if d > 0 else 1.0
        T = np.eye(n + 1)
        T[:n, :n] *= s
        T[:n, n] = -s * c
        return T
    T1, T2 = norm(P1), norm(P2)
    Q1 = apply_h(T1, P1)
    Q2 = np.hstack([apply_h(T2, P2), np.ones((N, 1))])
    m = n + 1
    rows = []
    for k in range(N):
        for a in range(n):
            r = np.zeros((m, m))
            r[a] = Q2[k]
            r[n] = -Q1[k, a] * Q2[k]
            rows.append(r.reshape(-1))
    A = np.array(rows)
    U, Sv, Vt = np.linalg.svd(A)
    if len(Sv) < m * m - 1 or Sv[m * m - 2] < 1e-10 * max(1.0, Sv[0]):
        return None
    Hn = Vt[-1].reshape(m, m)
    Hm = np.linalg.inv(T1) @ Hn @ T2
    if abs(Hm[n, n]) > 1e-12 * np.abs(Hm).max():
        Hm = Hm / Hm[n, n]
    else:
        return None
    return Hm


def spread(P):
    return float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())) if len(P) else 0.0


def choose_class(P1, P2, classes=CLASSES):
    """(class, M, err): the simplest class whose error is within 10% of the smallest (plus 1e-9 (1 + spread))."""
    res = []
    for c in classes:
        if c == "Perspective" and P1.shape[1] > 3:
            continue
        M = fit(P1, P2, c)
        if M is not None:
            res.append((c, M, mean_err(M, P1, P2)))
    if not res:
        raise ValueError("FindGeometricTransform: not enough points (or degenerate points) for any transformation")
    emin = min(r[2] for r in res)
    tol = 1.1 * emin + 1e-9 * (1 + spread(P1))
    return next(r for r in res if r[2] <= tol)


def ransac(P1, P2, cls, thr, iters=2000, seed=0):
    """(M, inlier mask) by RANSAC (Fischler & Bolles 1981): minimal samples, the largest consensus set (ties: the
    smaller mean error), refitted by least squares on the inliers (twice)."""
    N, n = P1.shape
    k = min_points(cls, n)
    if N < k:
        return None, None
    rng = np.random.default_rng(seed)
    best, best_key = None, None
    for _ in range(iters if N > k else 1):
        idx = rng.choice(N, k, replace=False)
        M = fit(P1[idx], P2[idx], cls)
        if M is None:
            continue
        d = np.sqrt(((apply_h(M, P2) - P1) ** 2).sum(1))
        d = np.where(np.isfinite(d), d, np.inf)
        inl = d <= thr
        key = (int(inl.sum()), -float(d[inl].mean()) if inl.any() else 0.0)
        if best_key is None or key > best_key:
            best, best_key = inl, key
    if best is None or best.sum() < k:
        return None, None
    inl = best
    for _ in range(2):
        M = fit(P1[inl], P2[inl], cls)
        if M is None:
            return None, None
        d = np.sqrt(((apply_h(M, P2) - P1) ** 2).sum(1))
        new = np.where(np.isfinite(d), d, np.inf) <= thr
        if new.sum() < k:
            break
        inl = new
    M = fit(P1[inl], P2[inl], cls)
    return M, inl


def tf_out(M):
    return Expr("TransformationFunction", L(*[L(*[_clean(v) for v in row]) for row in M]))


def tf_class(o, fname, allow_none=False):
    c = o.get("TransformationClass", AUTO)
    if _is_sym(c, "Automatic"):
        return None
    if allow_none and _is_sym(c, "None"):
        return "None"
    if c in CLASSES:
        return c
    if c == "Epipolar":
        raise NotImplementedError(f"{fname}: TransformationClass -> \"Epipolar\" is not supported")
    raise ValueError(f"{fname}: unknown TransformationClass {_short(c)}")


def robust_fit(P1, P2, cls, thr):
    """(M, err, inliers): RANSAC with the given class (Automatic: Perspective consensus in 2D/3D, Affine otherwise,
    then the class chosen on the inliers)."""
    n = P1.shape[1]
    base = cls or ("Perspective" if n in (2, 3) else "Affine")
    M, inl = ransac(P1, P2, base, thr)
    if M is None:
        return None, None, None
    if cls is None:
        _, M, _ = choose_class(P1[inl], P2[inl])
    return M, mean_err(M, P1[inl], P2[inl]), inl


# ───────────────────────────── FindGeometricTransform ─────────────────────────────

_PRIMS = ("Point", "Line", "Polygon", "Triangle", "Tetrahedron", "Simplex", "Arrow", "BSplineCurve", "BezierCurve")


def point_set(x, fname):
    if isinstance(x, Expr) and x.head in _PRIMS and x.args:
        x = x.args[0]
    if not K.is_list(x) or not x.args:
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: {_short(x)} is not a list of points, a primitive or an image")
    a = WI._numeric_array(x, fname)
    if a is None:
        raise Unevaluated()
    if a.ndim != 2 or a.shape[1] < 1:
        raise ValueError(f"{fname}: a point set must be a list of points of equal dimension")
    return a.astype(float)


def _is_set_list(x):
    return K.is_list(x) and x.args and all(K.is_list(p) and p.args and all(K.is_list(q) for q in p.args) or
                                           (isinstance(p, Expr) and p.head in _PRIMS) for p in x.args)


def _fgt_method(o, fname):
    m = o.get("Method", AUTO)
    if _is_sym(m, "Automatic"):
        return None, None
    if m in ("Linear", "RANSAC"):
        return m, None
    if m == "FindFit":
        raise NotImplementedError(f"{fname}: Method \"FindFit\" is not supported (\"Linear\" or \"RANSAC\")")
    if m == "ImageAlign":
        return "ImageAlign", AUTO
    if K.is_list(m) and len(m.args) == 2 and m.args[0] == "ImageAlign":
        return "ImageAlign", m.args[1]
    raise ValueError(f"{fname}: unknown Method {_short(m)}")


def _fgt_points(P1, P2, method, cls, fname):
    if P1.shape != P2.shape:
        raise ValueError(f"{fname}: the point sets must have the same number of points of the same dimension")
    if method == "ImageAlign":
        raise ValueError(f"{fname}: Method \"ImageAlign\" needs images")
    if method == "RANSAC":
        M, err, _ = robust_fit(P1, P2, cls, max(1e-9, 0.02 * spread(P1)))
        if M is None:
            raise ValueError(f"{fname}: RANSAC found no consistent transformation")
    elif cls is None:
        _, M, err = choose_class(P1, P2)
    else:
        if cls == "Perspective" and P1.shape[1] > 3:
            raise NotImplementedError(f"{fname}: perspective transformations are supported in 2D and 3D only")
        M = fit(P1, P2, cls)
        if M is None:
            raise ValueError(f"{fname}: not enough points (or degenerate points) for a {cls} transformation "
                             f"(at least {min_points(cls, P1.shape[1])} needed)")
        err = mean_err(M, P1, P2)
    return M, err


def image_transform(im1, im2, method, cls, fname, sub=AUTO):
    """(M, err) mapping image coordinates of im2 to those of im1 (keypoints + RANSAC, or phase correlation)."""
    if method == "ImageAlign" and isinstance(sub, str) and sub == "Fourier" or method == "Fourier":
        if cls not in (None, "Translation"):
            raise NotImplementedError(f"{fname}: the Fourier method finds translations only")
        return phase_correlation(im1, im2, fname)
    P1, P2, _ = correspondences(im1, im2)
    base = cls or "Perspective"
    if len(P1) < max(8, min_points(base, 2)):
        if method in (None, "ImageAlign") and cls in (None, "Translation") and sub is AUTO and \
                im1.a.shape[:2] == im2.a.shape[:2]:
            return phase_correlation(im1, im2, fname)
        raise ValueError(f"{fname}: only {len(P1)} corresponding keypoints were found; not enough for the "
                         "transformation")
    M, err, _ = robust_fit(P1, P2, cls, 2.0)
    if M is None:
        raise ValueError(f"{fname}: RANSAC found no consistent transformation between the images")
    return M, err


def phase_correlation(im1, im2, fname):
    g1, g2 = gray(im1), gray(im2)
    if g1.shape != g2.shape:
        raise NotImplementedError(f"{fname}: Fourier registration needs images of equal dimensions")
    F = np.fft.fft2(g1 - g1.mean()) * np.conj(np.fft.fft2(g2 - g2.mean()))
    R = F / np.maximum(np.abs(F), 1e-12)
    r = np.fft.ifft2(R).real
    pi, pj = np.unravel_index(int(np.argmax(r)), r.shape)
    h, w = r.shape

    def sub(c, m, n_, along):
        a = r[(c - 1) % n_, along] if m == 0 else r[along, (c - 1) % n_]
        b = r[c, along] if m == 0 else r[along, c]
        d = r[(c + 1) % n_, along] if m == 0 else r[along, (c + 1) % n_]
        den = a - 2 * b + d
        return 0.5 * (a - d) / den if abs(den) > 1e-12 else 0.0
    dr = pi + sub(pi, 0, h, pj)
    dc = pj + sub(pj, 1, w, pi)
    dr = dr - h if dr > h / 2 else dr
    dc = dc - w if dc > w / 2 else dc
    M = np.eye(3)
    M[0, 2], M[1, 2] = dc, -dr
    ov = (h - abs(round(dr))) * (w - abs(round(dc)))
    if ov <= 0:
        raise ValueError(f"{fname}: the images do not overlap")
    return M, 0.0


@fn("FindGeometricTransform", area=AREA, status="added",
    notes="FindGeometricTransform[pts1, pts2] gives {err, TransformationFunction[m]} with m mapping pts2 onto pts1 "
          "(err = the mean Euclidean distance of m(pts2) from pts1); also FindGeometricTransform[ref, {pts1, ...}] "
          "and FindGeometricTransform[{pts1, pts2, ...}] (each aligned with ref / pts1). Point lists in any "
          "dimension or primitives (Point, Line, Polygon, Triangle, ... : their point lists). TransformationClass: "
          "\"Translation\" (mean offset), \"Rigid\" and \"Similarity\" (Umeyama 1991, no reflections), \"Affine\" "
          "(least squares), \"Perspective\" (Hartley-normalised DLT, 2D and 3D); Automatic chooses the simplest class "
          "whose error is within 10% of the smallest (so exact data give the simplest exact class). Method: "
          "Automatic or \"Linear\" (closed form), \"RANSAC\" (fixed seed, 2000 samples, inlier threshold 2% of the "
          "RMS spread of pts1, Automatic class chosen on the inliers; err over the inliers). Images: "
          "FindGeometricTransform[image1, image2] maps image2's standard image coordinates (pixel units) to "
          "image1's, from ORB correspondences (as in ImageCorrespondingPoints) with RANSAC (threshold 2 pixels), or "
          "phase correlation (translation) when fewer than 8 correspondences are found or with Method -> "
          "{\"ImageAlign\", \"Fourier\"}. Too few or degenerate points raise ValueError; \"FindFit\", Image3D and "
          "perspective maps beyond 3D raise NotImplementedError. Check: self-check (err is recomputed from the "
          "returned matrix; ArithmeticError on mismatch) and tests (known transformations are recovered).")
def _findgeometrictransform(a: Expr | ListOf[Expr], *rest: Expr | ListOf[Expr] | Option) -> ListOf[Expr]:
    fname = "FindGeometricTransform"
    pos, o = opts_split(rest, {"Method", "TransformationClass"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected FindGeometricTransform[pts1, pts2, options]")
    method, sub = _fgt_method(o, fname)
    cls = tf_class(o, fname)

    def one(x, y):
        if isinstance(x, Expr) and x.head in IMAGE_HEADS or isinstance(y, Expr) and y.head in IMAGE_HEADS:
            im1, im2 = img2d(x, fname), img2d(y, fname)
            M, err = image_transform(im1, im2, method if method != "Linear" else None, cls, fname,
                                     sub if sub is not None else AUTO)
            return L(_clean(err), tf_out(M))
        P1, P2 = point_set(x, fname), point_set(y, fname)
        M, err = _fgt_points(P1, P2, method, cls, fname)
        if abs(mean_err(M, P1, P2) - err) > 1e-9 * (1 + err) and method != "RANSAC":
            raise ArithmeticError(f"{fname}: result failed verification")
        return L(_clean(err), tf_out(M))
    if not pos:
        if not _is_set_list(a) and not (K.is_list(a) and a.args and all(isinstance(t, Expr) and t.head in
                                                                       IMAGE_HEADS for t in a.args)):
            if isinstance(a, Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: expected FindGeometricTransform[pts1, pts2] or a list of point sets")
        items = list(a.args)
        return L(*[one(items[0], t) for t in items[1:]])
    b = pos[0]
    if _is_set_list(b) or (K.is_list(b) and b.args and all(isinstance(t, Expr) and t.head in IMAGE_HEADS
                                                           for t in b.args)):
        return L(*[one(a, t) for t in b.args])
    return one(a, b)


# ───────────────────────────── ImageCorrespondingPoints ─────────────────────────────

@fn("ImageCorrespondingPoints", area=AREA, status="added",
    notes="ImageCorrespondingPoints[image1, image2] gives {points1, points2}, matching keypoint positions (standard "
          "image coordinates) in the two images: ORB keypoints (as in ImageKeypoints; Method -> Automatic or "
          "\"ORB\"; at most the 1000 strongest per image) matched by Hamming distance as mutual nearest neighbours "
          "with distance <= 64 bits passing Lowe's ratio test (0.8), ordered by decreasing average keypoint "
          "Confidence. MaxFeatures -> n keeps the n strongest pairs, KeypointStrength -> s keypoints with "
          "Confidence >= s, Masking -> roi restricts points1 to the region. TransformationClass -> None (default) or "
          "a class keeps the RANSAC inliers (threshold 2 pixels) of that class of transformation. Other keypoint "
          "methods, \"Epipolar\" and Image3D raise NotImplementedError. Check: reference (tests: a shifted textured "
          "image gives pairs differing by the shift; a rotated one pairs consistent with the rotation).")
def _imagecorrespondingpoints(im1: Expr, im2: Expr, *opts: Option) -> ListOf[MatrixOf[Real]]:
    fname = "ImageCorrespondingPoints"
    pos, o = opts_split(opts, _KP_OPTS | {"TransformationClass"}, fname)
    if pos:
        raise TypeError(f"{fname}: expected ImageCorrespondingPoints[image1, image2, options]")
    a, b = img2d(im1, fname), img2d(im2, fname)
    method = kp_method(o, fname)
    if method != "ORB":
        raise NotImplementedError(f"{fname}: Method {method!r} has no descriptors (use \"ORB\")")
    cls = tf_class(o, fname, allow_none=True) if "TransformationClass" in o else "None"
    P1, P2, conf = correspondences(a, b, "ORB", _masking(o, a, fname), _strength(o, fname))
    order = np.argsort(-conf, kind="stable")
    P1, P2 = P1[order], P2[order]
    if cls not in ("None",) and len(P1):
        M, _, inl = robust_fit(P1, P2, cls, 2.0)
        if M is None:
            P1, P2 = P1[:0], P2[:0]
        else:
            P1, P2 = P1[inl], P2[inl]
    nmax = _opt_int(o, "MaxFeatures", fname, None)
    if nmax is not None:
        P1, P2 = P1[:nmax], P2[:nmax]
    return L(L(*[_vec(p) for p in P1]), L(*[_vec(p) for p in P2]))


# ───────────────────────────── image lists ─────────────────────────────

def image_list(x, fname, min_len=2):
    if isinstance(x, Expr) and x.head in ("Video", "AnimatedImage"):
        raise NotImplementedError(f"{fname}: {x.head} input is not supported (give a list of images)")
    if not K.is_list(x):
        if isinstance(x, Symbol) or (isinstance(x, Expr) and x.head not in IMAGE_HEADS):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a list of images")
    if len(x.args) < min_len:
        raise ValueError(f"{fname}: at least {min_len} images are required")
    ims = [img2d(t, fname) for t in x.args]
    return ims


# ───────────────────────────── pyramidal Lucas-Kanade tracking ─────────────────────────────

LK_RADIUS = 7


def good_features(g, nmax, mask=None, quality=0.01, mindist=5.0):
    """Shi-Tomasi (1994) features: minimum-eigenvalue local maxima >= quality * max, greedily at least mindist
    apart, strongest first."""
    jxx, jxy, jyy = structure_tensor(g[..., None], (1.0, 1.0), (1.5, 1.5), "Reflected")
    lam = corner_measure(jxx, jxy, jyy, "ShiTomasi")
    mx = float(lam.max())
    if mx <= 0:
        return []
    sel = local_max(lam, 3) & (lam >= quality * mx)
    if mask is not None:
        sel &= mask
    cand = sorted(zip(*np.nonzero(sel)), key=lambda p: (-lam[p], p))
    out = []
    for p in cand:
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= mindist ** 2 for q in out):
            out.append(p)
            if nmax is not None and len(out) >= nmax:
                break
    return out


def _sample(a, rows, cols):
    return WI._bilinear(a[..., None], cols, rows, 0.0)[..., 0]


def lk_track(I, J, pts, maxdisp, iters, tol):
    """Pyramidal Lucas-Kanade (Bouguet 2000) of (row, col) points from I to J: window radius 7, the number of
    levels chosen so that (2^(L+1) - 1) 7 covers maxdisp. Returns [(row, col) or None]."""
    md = max(maxdisp)
    nl = 0
    while (2 ** (nl + 1) - 1) * LK_RADIUS < md and min(I.shape) / 2 ** (nl + 1) >= 2 * LK_RADIUS + 1:
        nl += 1
    PI = [t[..., 0] for t in gaussian_levels(I[..., None], 2, nl + 1)]
    PJ = [t[..., 0] for t in gaussian_levels(J[..., None], 2, nl + 1)]
    grads = []
    for A in PI:
        gx = (_shift(A, (0, 1), np.nan) - _shift(A, (0, -1), np.nan)) / 2
        gy = (_shift(A, (1, 0), np.nan) - _shift(A, (-1, 0), np.nan)) / 2
        grads.append((np.nan_to_num(gx), np.nan_to_num(gy)))
    win = np.arange(-LK_RADIUS, LK_RADIUS + 1, dtype=float)
    WR, WC = np.meshgrid(win, win, indexing="ij")
    out = []
    h, w = I.shape
    for (r0, c0) in pts:
        g = np.zeros(2)
        lost = False
        for lev in range(nl, -1, -1):
            f = 2 ** lev
            r, c = r0 / f, c0 / f
            A, B = PI[lev], PJ[lev]
            gx, gy = grads[lev]
            Rr, Cc = r + WR, c + WC
            Ix, Iy, Iw = _sample(gx, Rr, Cc), _sample(gy, Rr, Cc), _sample(A, Rr, Cc)
            G = np.array([[(Ix * Ix).sum(), (Ix * Iy).sum()], [(Ix * Iy).sum(), (Iy * Iy).sum()]])
            if np.linalg.eigvalsh(G)[0] / Ix.size < 1e-6:
                lost = True
                break
            Gi = np.linalg.inv(G)
            v = np.zeros(2)
            for _ in range(iters):
                Jw = _sample(B, Rr + g[0] + v[0], Cc + g[1] + v[1])
                dI = Iw - Jw
                eta = Gi @ np.array([(dI * Ix).sum(), (dI * Iy).sum()])
                v += np.array([eta[1], eta[0]])
                if math.hypot(eta[0], eta[1]) < tol:
                    break
            if lev:
                g = 2 * (g + v)
            else:
                g = g + v
        if lost:
            out.append(None)
            continue
        nr, nc = r0 + g[0], c0 + g[1]
        if not (-0.5 <= nr <= h - 0.5 and -0.5 <= nc <= w - 0.5) or abs(g[0]) > maxdisp[0] or \
                abs(g[1]) > maxdisp[1]:
            out.append(None)
        else:
            out.append((nr, nc))
    return out


@fn("ImageFeatureTrack", area=AREA, status="added",
    notes="ImageFeatureTrack[{image1, ..., imagen}] gives {pts1, ..., ptsn}, the positions (standard image "
          "coordinates) of features tracked from image1; ImageFeatureTrack[images, pts] tracks the given points. "
          "Automatic features: Shi-Tomasi (1994) minimum-eigenvalue maxima (derivative sigma 1, integration sigma "
          "1.5) at least 1% of the strongest, 5 pixels apart, strongest first (MaxFeatures -> 100, as documented; "
          "Masking -> roi). Tracking: pyramidal Lucas-Kanade (Bouguet 2000) on the intensity, window 15 x 15, "
          "pyramid depth from MaxFeatureDisplacement (15, or {r_row, r_col}), MaxIterations -> 20 and Tolerance -> "
          "0.03 pixels per level (as documented); a point is lost (Missing[] in that and all later images) when its "
          "window has too little texture, it leaves the image or it moves further than MaxFeatureDisplacement. All "
          "images must have equal dimensions. Video and AnimatedImage input and Image3D raise NotImplementedError. "
          "Check: reference (tests: features of a translated image move by the translation).")
def _imagefeaturetrack(imgs: ListOf[Expr], *rest: MatrixOf[Real] | Option) -> ListOf[Expr]:
    fname = "ImageFeatureTrack"
    pos, o = opts_split(rest, {"Masking", "MaxFeatureDisplacement", "MaxFeatures", "MaxIterations", "Tolerance"},
                        fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected ImageFeatureTrack[images, pts, options]")
    ims = image_list(imgs, fname, 1)
    h, w = ims[0].h, ims[0].w
    if any(i.a.shape[:2] != (h, w) for i in ims):
        raise ValueError(f"{fname}: all images must have the same dimensions")
    md = o.get("MaxFeatureDisplacement", 15)
    mds = [_real(v, fname, "MaxFeatureDisplacement") for v in md.args] if K.is_list(md) else \
        [_real(md, fname, "MaxFeatureDisplacement")] * 2
    if len(mds) != 2 or min(mds) <= 0:
        raise ValueError(f"{fname}: MaxFeatureDisplacement must be r > 0 or {{r_row, r_col}}")
    iters = int_arg(o.get("MaxIterations", 20), fname, "MaxIterations", 1)
    tol = _real(o.get("Tolerance", 0.03), fname, "Tolerance")
    if tol <= 0:
        raise ValueError(f"{fname}: Tolerance must be positive")
    G = [gray(i) for i in ims]
    if pos:
        p = pos[0]
        if not (K.is_list(p) and all(is_point(t) for t in p.args)):
            raise TypeError(f"{fname}: the points must be a list of positions {{x, y}}")
        cur = [(h - y - 0.5, x - 0.5) for x, y in (_point(t, fname) for t in p.args)]
    else:
        nmax = _opt_int(o, "MaxFeatures", fname, 100)
        cur = [(float(r), float(c)) for r, c in good_features(G[0], nmax, _masking(o, ims[0], fname))]
    frames = [list(cur)]
    for k in range(1, len(G)):
        alive = [i for i, p in enumerate(cur) if p is not None]
        res = lk_track(G[k - 1], G[k], [cur[i] for i in alive], mds, iters, tol) if alive else []
        nxt = [None] * len(cur)
        for i, r in zip(alive, res):
            nxt[i] = r
        frames.append(nxt)
        cur = nxt
    miss = Expr("Missing")
    return L(*[L(*[miss if p is None else _vec((p[1] + 0.5, h - p[0] - 0.5)) for p in fr]) for fr in frames])


# ───────────────────────────── Farneback dense flow ─────────────────────────────

POLY_N, POLY_SIGMA = 5, 1.1
FLOW_WIN_SIGMA = 2.5


def poly_expansion(f):
    """Farneback quadratic expansion f(x) ~ x^T A x + b^T x + c by Gaussian-weighted least squares (radius 5,
    sigma 1.1); returns (A11, A12, A22, b1, b2) with x = (column, row)."""
    s = np.arange(-POLY_N, POLY_N + 1, dtype=float)
    a = np.exp(-s * s / (2 * POLY_SIGMA ** 2))
    basis = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]          # powers of (x = column, y = row)
    X, Y = np.meshgrid(s, s)
    Wt = np.outer(a, a)
    B = np.stack([X ** p * Y ** q for p, q in basis], -1).reshape(-1, 6)
    G = (B * Wt.reshape(-1, 1)).T @ B
    Gi = np.linalg.inv(G)
    r = []
    for p, q in basis:
        t = correlate_axis_padded(f, a * s ** p, 1, "Reflected")
        r.append(correlate_axis_padded(t, a * s ** q, 0, "Reflected"))
    R = np.stack(r, -1) @ Gi.T
    return R[..., 3], R[..., 5] / 2, R[..., 4], R[..., 1], R[..., 2]


def farneback_level(f1, f2, d, iters):
    h, w = f1.shape
    A1 = poly_expansion(f1)
    A2 = poly_expansion(f2)
    rows, cols = np.mgrid[0:h, 0:w].astype(float)
    for _ in range(iters):
        rr = np.clip(rows + d[..., 1], 0, h - 1)
        cc = np.clip(cols + d[..., 0], 0, w - 1)
        a11, a12, a22, b1, b2 = (_sample(t, rr, cc) for t in A2)
        A11 = (A1[0] + a11) / 2
        A12 = (A1[1] + a12) / 2
        A22 = (A1[2] + a22) / 2
        db1 = -0.5 * (b1 - A1[3]) + A11 * d[..., 0] + A12 * d[..., 1]
        db2 = -0.5 * (b2 - A1[4]) + A12 * d[..., 0] + A22 * d[..., 1]
        g11 = A11 * A11 + A12 * A12
        g12 = A11 * A12 + A12 * A22
        g22 = A12 * A12 + A22 * A22
        h1 = A11 * db1 + A12 * db2
        h2 = A12 * db1 + A22 * db2
        g11, g12, g22, h1, h2 = (smooth(t, FLOW_WIN_SIGMA, FLOW_WIN_SIGMA, "Reflected") for t in (g11, g12, g22, h1,
                                                                                                    h2))
        det = g11 * g22 - g12 * g12
        ok = np.abs(det) > 1e-12
        sd = np.where(ok, det, 1.0)
        d = np.stack([np.where(ok, (g22 * h1 - g12 * h2) / sd, d[..., 0]),
                      np.where(ok, (g11 * h2 - g12 * h1) / sd, d[..., 1])], -1)
    return d


def farneback(f1, f2, iters, d0=None):
    """Dense displacement (d_column, d_row) with f2(x + d) ~ f1(x): Farneback (2003) on a Burt-Adelson pyramid of
    up to 3 levels (coarsest side >= 16)."""
    nl = 1
    while nl < 3 and min(f1.shape) / 2 ** nl >= 16:
        nl += 1
    P1 = [t[..., 0] for t in gaussian_levels(f1[..., None], 2, nl)]
    P2 = [t[..., 0] for t in gaussian_levels(f2[..., None], 2, nl)]
    if d0 is None:
        d = np.zeros(P1[-1].shape + (2,))
    else:
        d = d0
        for _ in range(nl - 1):
            d = pyr_reduce(d, 2) / 2
    for lev in range(nl - 1, -1, -1):
        if d.shape[:2] != P1[lev].shape:
            d = pyr_expand(d, P1[lev].shape, 2) * 2
        d = farneback_level(P1[lev], P2[lev], d, iters)
    return d


@fn("ImageDisplacements", area=AREA, status="added",
    notes="ImageDisplacements[{image1, ..., imagen}] gives the n - 1 dense displacement fields between consecutive "
          "images: for each pair an h x w matrix (rows from the top, like ImageData) of {dx, dy} in pixels, with dx "
          "to the right and dy upwards (standard image coordinate directions), such that the content at pixel p of "
          "image i is found at p + {dx, dy} in image i + 1. Method -> Automatic or \"Farneback\" (Farneback 2003: "
          "quadratic expansion radius 5, sigma 1.1, Gaussian averaging window sigma 2.5, up to 3 Burt-Adelson "
          "pyramid levels; MaxIterations -> 10 per level, as documented), computed on the intensity. "
          "ImageDisplacements[images, flow] starts the first pair from the given field (same format). "
          "\"DenseInverseSearch\", PerformanceGoal other than Automatic, Video and AnimatedImage input and Image3D "
          "raise NotImplementedError; images of different dimensions raise ValueError. Check: reference (tests: a "
          "smooth image shifted by whole pixels gives that shift in the interior; identical images give zero).")
def _imagedisplacements(imgs: ListOf[Expr], *rest: Expr | Option) -> ListOf[Expr]:
    fname = "ImageDisplacements"
    pos, o = opts_split(rest, {"MaxIterations", "Method", "PerformanceGoal"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected ImageDisplacements[images, flow, options]")
    m = o.get("Method", AUTO)
    if not (_is_sym(m, "Automatic") or m == "Farneback"):
        raise NotImplementedError(f"{fname}: Method {_short(m)} is not supported (\"Farneback\")")
    if not _is_sym(o.get("PerformanceGoal", AUTO), "Automatic"):
        raise NotImplementedError(f"{fname}: PerformanceGoal is not supported")
    iters = int_arg(o.get("MaxIterations", 10), fname, "MaxIterations", 1)
    ims = image_list(imgs, fname, 2)
    h, w = ims[0].h, ims[0].w
    if any(i.a.shape[:2] != (h, w) for i in ims):
        raise ValueError(f"{fname}: all images must have the same dimensions")
    d0 = None
    if pos:
        arr = WI._numeric_array(pos[0], fname) if K.is_list(pos[0]) else None
        if arr is None:
            if isinstance(pos[0], Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: the initial flow must be an h x w x 2 array of {{dx, dy}}")
        if arr.shape != (h, w, 2):
            raise ValueError(f"{fname}: the initial flow must have dimensions {{{h}, {w}, 2}}")
        d0 = np.stack([arr[..., 0], -arr[..., 1]], -1)
    G = [gray(i) for i in ims]
    out = []
    for k in range(len(G) - 1):
        d = farneback(G[k], G[k + 1], iters, d0 if k == 0 else None)
        out.append(L(*[L(*[L(_clean(d[i, j, 0]), _clean(-d[i, j, 1])) for j in range(w)]) for i in range(h)]))
    return L(*out)


# ───────────────────────────── ImageAlign ─────────────────────────────

def warp_to(im2, M, w, h, bg):
    """im2 resampled (bilinear) onto a w x h frame whose image coordinates q satisfy q = M . p (p in im2)."""
    Mi = np.linalg.inv(M)
    rows, cols = np.mgrid[0:h, 0:w].astype(float)
    X, Y = cols + 0.5, h - rows - 0.5
    P = apply_h(Mi, np.stack([X.ravel(), Y.ravel()], 1))
    u = P[:, 0].reshape(h, w) - 0.5
    v = im2.h - P[:, 1].reshape(h, w) - 0.5
    bad = ~(np.isfinite(u) & np.isfinite(v))
    u = np.where(bad, -10.0, u)
    v = np.where(bad, -10.0, v)
    return WI._bilinear(im2.a, u, v, bg)


def _align_method(o, fname):
    m = o.get("Method", AUTO)
    if _is_sym(m, "Automatic"):
        return None
    if m == "Keypoints" or (K.is_list(m) and len(m.args) == 2 and m.args[0] == "Keypoints" and
                            m.args[1] in ("ORB", AUTO)):
        return "Keypoints"
    if K.is_list(m) and len(m.args) == 2 and m.args[0] == "Keypoints":
        raise NotImplementedError(f"{fname}: keypoint method {_short(m.args[1])} is not supported (\"ORB\")")
    if m == "Fourier":
        return "Fourier"
    if m in ("MeanSquareGradientDescent", "FourierBlurInvariant"):
        raise NotImplementedError(f"{fname}: Method {m!r} is not supported (\"Keypoints\" or \"Fourier\")")
    raise ValueError(f"{fname}: unknown Method {_short(m)}")


@fn("ImageAlign", area=AREA, status="added",
    notes="ImageAlign[ref, image] gives image transformed into the frame of ref (same dimensions as ref, bilinear "
          "resampling, Background -> Automatic = 0 or a value/colour outside), ImageAlign[ref, {image1, ...}] each "
          "image, ImageAlign[{image1, ..., imagen}] all of them aligned with image1 (image1 itself unchanged). The "
          "transformation (from image coordinates of image to those of ref) is found as in FindGeometricTransform: "
          "Method -> \"Keypoints\" ({\"Keypoints\", \"ORB\"}): ORB correspondences with RANSAC (2 pixels); "
          "\"Fourier\": phase correlation with parabolic subpixel peak (translations only, equal dimensions); "
          "Automatic: keypoints when at least 8 correspondences are found, else phase correlation. "
          "TransformationClass -> Automatic (the simplest class within 10% of the best error on the RANSAC inliers) "
          "or a class. \"MeanSquareGradientDescent\", \"FourierBlurInvariant\", other keypoint methods and Image3D "
          "raise NotImplementedError; too few correspondences ValueError. Check: reference (tests: a shifted image "
          "is aligned back to the reference in the overlap; aligning an image with itself gives it unchanged).")
def _imagealign(ref: Expr | ListOf[Expr], *rest: Expr | ListOf[Expr] | Option) -> Expr | ListOf[Expr]:
    fname = "ImageAlign"
    pos, o = opts_split(rest, {"Background", "Method", "TransformationClass"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected ImageAlign[ref, image, options]")
    method = _align_method(o, fname)
    cls = tf_class(o, fname)
    if not pos:
        ims = image_list(ref, fname, 1)
        r, targets, lst = ims[0], ims[1:], True
        first = [ref.args[0]]
    else:
        r = img2d(ref, fname)
        first = []
        if K.is_list(pos[0]):
            targets, lst = image_list(pos[0], fname, 1), True
        else:
            targets, lst = [img2d(pos[0], fname)], False
    out = list(first)
    for im in targets:
        if method == "Fourier":
            M, _ = image_transform(r, im, "Fourier", cls, fname)
        elif method == "Keypoints":
            M, _ = image_transform(r, im, "Keypoints", cls, fname)
        else:
            M, _ = image_transform(r, im, None, cls, fname)
        bgs = o.get("Background", AUTO)
        bg = np.zeros(im.c) if _is_sym(bgs, "Automatic") else WI.value_for(im, bgs, fname)
        a = warp_to(im, M, r.w, r.h, bg)
        typ = im.type if im.type not in ("Bit",) else "Real32"
        out.append(image_expr(WI.Img(quantize(a, typ), typ, im.cs)))
    return L(*out) if lst else out[0]


# ───────────────────────────── ImageStitch ─────────────────────────────

def _feather(h, w):
    r = np.arange(h, dtype=float)
    c = np.arange(w, dtype=float)
    return np.minimum.outer(np.minimum(r + 0.5, h - r - 0.5), np.minimum(c + 0.5, w - c - 0.5))


@fn("ImageStitch", area=AREA, status="added",
    notes="ImageStitch[{image1, image2, ...}] stitches overlapping images on a plane: pairwise ORB correspondences "
          "with RANSAC (threshold 2 pixels; TransformationClass -> Automatic = Perspective, or a class), a "
          "maximum spanning tree of the pairwise inlier counts rooted at the image with the most inliers (which "
          "keeps its orientation), composed transformations, and a canvas bounding all transformed images; "
          "uncovered pixels are 0. Method -> {\"ImageBlending\" -> \"Feathering\" (default: weights the distance "
          "to the image border) | None (earlier images in tree order win), \"HistogramEqualization\" -> True "
          "(default: each image is scaled per channel so its mean over the overlap with the mosaic built so far "
          "matches; a simplified gain compensation) | False, \"KeypointMethod\" -> Automatic | \"ORB\"}. A matrix "
          "of images is stitched like the flat list (array positions are not used). Canvas \"Cylinder\" and "
          "\"Sphere\", {\"Feathering\", sigma}, other keypoint methods and Image3D raise NotImplementedError; images "
          "that cannot be connected by correspondences raise ValueError. Check: reference (tests: two overlapping "
          "crops of a textured image are stitched back to the original extent).")
def _imagestitch(imgs: ListOf[Expr], *rest: String | ListOf[Expr] | Option) -> Expr:
    fname = "ImageStitch"
    pos, o = opts_split(rest, {"Method", "TransformationClass"}, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: expected ImageStitch[images, canvas, options]")
    if pos:
        cv = pos[0]
        name = cv.args[0] if K.is_list(cv) and cv.args else cv
        if name in ("Cylinder", "Sphere"):
            raise NotImplementedError(f"{fname}: canvas {name!r} is not supported (\"Plane\")")
        if name != "Plane" or (K.is_list(cv) and len(cv.args) > 1):
            raise NotImplementedError(f"{fname}: canvas specification {_short(cv)} is not supported (\"Plane\")")
    if K.is_list(imgs) and imgs.args and all(K.is_list(r) for r in imgs.args):
        imgs = L(*[t for r in imgs.args for t in r.args])
    ims = image_list(imgs, fname, 1)
    if len({i.c for i in ims}) > 1:
        raise ValueError(f"{fname}: the images must have the same number of channels")
    blend, equalize = "Feathering", True
    m = o.get("Method", AUTO)
    if not _is_sym(m, "Automatic"):
        items = list(m.args) if K.is_list(m) else [m]
        for it in items:
            if not (isinstance(it, Expr) and it.head == "Rule" and isinstance(it.args[0], str)):
                raise ValueError(f"{fname}: Method must be a list of \"name\" -> value settings")
            k, v = it.args
            if k == "ImageBlending":
                if v == "Feathering" or _is_sym(v, "Automatic"):
                    blend = "Feathering"
                elif _is_sym(v, "None"):
                    blend = None
                else:
                    raise NotImplementedError(f"{fname}: \"ImageBlending\" -> {_short(v)} is not supported")
            elif k == "HistogramEqualization":
                equalize = _is_sym(v, "True") or v is True
                if not equalize and not (_is_sym(v, "False") or v is False):
                    raise ValueError(f"{fname}: \"HistogramEqualization\" must be True or False")
            elif k == "KeypointMethod":
                if not (_is_sym(v, "Automatic") or v == "ORB"):
                    raise NotImplementedError(f"{fname}: \"KeypointMethod\" -> {_short(v)} is not supported")
            else:
                raise ValueError(f"{fname}: unknown Method setting {k!r}")
    cls = tf_class(o, fname) or "Perspective"
    n = len(ims)
    if n == 1:
        return imgs.args[0]
    pair = {}
    cnt = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            P1, P2, _ = correspondences(ims[i], ims[j])
            if len(P1) < max(8, min_points(cls, 2)):
                continue
            M, inl = ransac(P1, P2, cls, 2.0)
            if M is None or inl.sum() < 8:
                continue
            pair[(i, j)] = M                         # maps j -> i
            cnt[i, j] = cnt[j, i] = int(inl.sum())
    root = int(np.argmax(cnt.sum(1)))
    T = {root: np.eye(3)}
    order = [root]
    while len(T) < n:
        best = None
        for a in T:
            for b in range(n):
                if b not in T and cnt[a, b] > 0 and (best is None or cnt[a, b] > best[0]):
                    best = (cnt[a, b], a, b)
        if best is None:
            raise ValueError(f"{fname}: image {next(b for b in range(n) if b not in T) + 1} could not be connected "
                             "to the others by corresponding points")
        _, a, b = best
        Mab = pair[(a, b)] if (a, b) in pair else np.linalg.inv(pair[(b, a)])   # maps b -> a
        T[b] = T[a] @ Mab
        order.append(b)
    corners = []
    for i, im in enumerate(ims):
        c = apply_h(T[i], np.array([[0, 0], [im.w, 0], [im.w, im.h], [0, im.h]], dtype=float))
        if not np.all(np.isfinite(c)):
            raise ValueError(f"{fname}: image {i + 1} is mapped to infinity")
        corners.append(c)
    allc = np.vstack(corners)
    x0, y0 = np.floor(allc.min(0) + 1e-9)
    x1, y1 = np.ceil(allc.max(0) - 1e-9)
    W, H = int(x1 - x0), int(y1 - y0)
    if W * H > 16 * sum(i.w * i.h for i in ims):
        raise ValueError(f"{fname}: the stitched canvas would be unreasonably large (degenerate transformation)")
    Sh = np.eye(3)
    Sh[0, 2], Sh[1, 2] = -x0, -y0
    c = ims[0].c
    num = np.zeros((H, W, c))
    den = np.zeros((H, W, 1))
    for i in order:
        im = ims[i]
        M = Sh @ T[i]
        a = warp_to(im, M, W, H, np.full(c, np.nan))
        ok = np.all(np.isfinite(a), -1, keepdims=True)
        a = np.nan_to_num(a)
        wt = warp_to(WI.Img(_feather(im.h, im.w)), M, W, H, np.array([0.0]))[..., :1] * ok
        if equalize and den.any():
            ov = (den[..., 0] > 0) & ok[..., 0]
            if ov.sum() > 0:
                cur = num[ov] / den[ov]
                mi = a[ov].mean(0)
                gain = np.where(mi > 1e-9, cur.mean(0) / np.where(mi > 1e-9, mi, 1.0), 1.0)
                a = a * gain
        if blend is None:
            wt = (ok & (den == 0)).astype(float)
        num += a * wt
        den += wt
    out = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    typ = ims[root].type if ims[root].type != "Bit" else "Real32"
    return image_expr(WI.Img(quantize(out, typ), typ, ims[root].cs))


record("TransformationClass", AREA, "added",
       "option of FindGeometricTransform, ImageAlign, ImageCorrespondingPoints and ImageStitch giving the class of "
       "geometric transformation: \"Translation\", \"Rigid\", \"Similarity\", \"Affine\", \"Perspective\" (and None "
       "for ImageCorrespondingPoints); an inert option name. Check: reference (used by the X05 heads in tests).")
