"""Wolfram batch X03 (Signal II): wavelet transforms, time-frequency analysis and spectral estimation.

* Discrete wavelet transforms (DiscreteWaveletTransform, DiscreteWaveletPacketTransform, StationaryWaveletTransform,
  StationaryWaveletPacketTransform, LiftingWaveletTransform), their container DiscreteWaveletData, the inverse
  InverseWaveletTransform and the coefficient tools WaveletBestBasis, WaveletMapIndexed, WaveletThreshold and the
  plots WaveletListPlot, WaveletMatrixPlot, WaveletImagePlot, WaveletScalogram.
* ContinuousWaveletTransform / ContinuousWaveletData / InverseContinuousWaveletTransform.
* Short-time Fourier analysis: ShortTimeFourier / ShortTimeFourierData / InverseShortTimeFourier, SpectrogramArray,
  Spectrogram, InverseSpectrogram, Cepstrogram, Periodogram, PowerSpectralDensity.
* Filters: WienerFilter, EstimatedBackground, ListDeconvolve.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input (see
CONVENTIONS.md).  Where Wolfram's defaults are undocumented, the published method used here and its defaults are
stated in the notes, rather than claiming to match Wolfram's numbers.  NumPy is required; SciPy is never used.
"""
from __future__ import annotations

import math

import mpmath
import numpy as np
import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, Unevaluated, R, S, is_list
from .wl_common import L, apply, is_rule
from .wl_signal import (_opts, _array, _window, _int_arg, _fourier_np, _ifourier_np, _fp_float, _out, _nest,
                        _wp_opt, _to_np, _cepstrogram)
from .wl_signal_wavelets import _wavelet, _lowpass, _highpass, _psi_expr, _FAMILIES

AREA = "signal processing II (X03)"


def ev(x):
    from ..evaluate import evaluate
    return evaluate(x)


def _sym(name):
    return Symbol(name)


def _is_sym(v, *names):
    return isinstance(v, Symbol) and v.name in names


def _rule(a, b):
    return Expr("Rule", a, b)


def _assoc(pairs):
    return Expr("Association", *[_rule(k, v) for k, v in pairs])


def _assoc_dict(a):
    """Association -> {key: value} with string keys or tuple keys for integer lists."""
    out = {}
    for r in a.args:
        out[_key(r.args[0])] = r.args[1]
    return out


def _key(k):
    if is_list(k) and all(K.is_int(t) for t in k.args):
        return tuple(int(t) for t in k.args)
    return k


def _wind_expr(w):
    return L(*w)


# ═══════════════════════════════════════════════════════════════ numeric arrays ══
def _num(v):
    """A MikoshiLang number -> Python float/complex (machine)."""
    if isinstance(v, bool) or isinstance(v, str):
        raise TypeError(f"not a number: {v!r}")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, complex):
        return v
    s = S(v)
    if not isinstance(s, sp.Basic) or s.free_symbols or not s.is_number:
        raise TypeError(f"not a number: {v!r}")
    c = complex(sp.N(s, 17))
    return c.real if c.imag == 0 else c


def _from_list(x):
    """Nested MikoshiLang list of numbers -> numpy array (float or complex)."""
    def rec(z):
        if is_list(z):
            return [rec(a) for a in z.args]
        return _num(z)
    a = np.array(rec(x))
    if a.dtype == object:
        raise ValueError("the coefficient array must be rectangular")
    return a


def _to_list(arr, exact=False):
    """numpy array -> nested MikoshiLang list (machine reals stay floats; complex values with zero imaginary part
    are given as reals)."""
    if exact:
        return _nest(arr, lambda v: R(sp.nsimplify(v) if isinstance(v, float) else v))
    if np.iscomplexobj(arr):
        if arr.size and np.all(arr.imag == 0):
            return _out(arr.real, "m", real=True)
        return _out(arr, "m")
    return _out(arr.astype(float), "m", real=True)


def _data_array(x, fname, exact=False, max_rank=None):
    """Numeric data (rectangular list of any depth) -> numpy array: float/complex (machine) or object (exact)."""
    flat, shape = _array(x, fname, max_rank=max_rank)
    if exact:
        a = np.empty(len(flat), dtype=object)
        a[:] = [sp.nsimplify(v) if isinstance(v, sp.Float) else v for v in flat]
        return a.reshape(shape)
    arr = _to_np(flat, shape)
    if np.all(arr.imag == 0):
        arr = arr.real.copy()
    return arr


def _sym_data(x):
    """True if x is a list containing symbolic entries (used to route exact symbolic transforms)."""
    if is_list(x):
        return any(_sym_data(a) for a in x.args)
    if isinstance(x, (Symbol, Expr)):
        try:
            s = S(x)
        except Exception:
            return True
        return not (isinstance(s, sp.Basic) and s.is_number)
    return False


def _sym_array(x, fname):
    """A rectangular list of arbitrary (possibly symbolic) scalar expressions -> object array of SymPy values."""
    shape = []
    y = x
    while is_list(y):
        shape.append(len(y.args))
        if not y.args:
            raise ValueError(f"{fname}: the data must be a nonempty list")
        y = y.args[0]
    flat = []

    def walk(z, d):
        if d == len(shape):
            if is_list(z):
                raise ValueError(f"{fname}: the array of data must be rectangular")
            if isinstance(z, (str, bool)):
                raise TypeError(f"{fname}: data elements must be numbers or symbolic expressions, not {z!r}")
            flat.append(sp.sympify(S(z)))
            return
        if not is_list(z) or len(z.args) != shape[d]:
            raise ValueError(f"{fname}: the array of data must be rectangular")
        for a in z.args:
            walk(a, d + 1)
    walk(x, 0)
    a = np.empty(len(flat), dtype=object)
    a[:] = flat
    return a.reshape(shape)


# ═══════════════════════════════════════════════════════════════ wavelet filters ══
_DWT_FAMILIES = ("HaarWavelet", "DaubechiesWavelet")
_SQRT2 = math.sqrt(2)


def _filters(wave, fname, exact=False):
    """(lowpass, highpass) analysis filters (orthonormal, Σ = √2) as lists in increasing k order, for HaarWavelet
    and DaubechiesWavelet[n]; other families raise NotImplementedError."""
    name, p = _wavelet(wave, fname)
    if name not in _DWT_FAMILIES:
        raise NotImplementedError(f"{fname}: {name} is not supported (only HaarWavelet and DaubechiesWavelet; the "
                                  "filters of the other families are not available)")
    if exact:
        if name == "DaubechiesWavelet" and int(p) > 2:
            raise NotImplementedError(f"{fname}: exact (WorkingPrecision -> Infinity) transforms are supported for "
                                      "HaarWavelet and DaubechiesWavelet[n] with n <= 2")
        h = _lowpass(name, p, True, 30)
        g = _highpass(name, h)
        lo = [sp.sqrt(2) * h[k] for k in sorted(h)]
        hi = [sp.sqrt(2) * g[k] for k in sorted(g)]
        return [sp.radsimp(sp.expand(v)) for v in lo], [sp.radsimp(sp.expand(v)) for v in hi]
    with mpmath.workdps(30):
        h = _lowpass(name, p, False, 30)
        g = _highpass(name, h)
        lo = np.array([float(h[k]) * _SQRT2 for k in sorted(h)])
        hi = np.array([float(g[k]) * _SQRT2 for k in sorted(g)])
    return lo, hi


def _wave_expr(wave):
    """Canonical wavelet object (parameter defaults filled in)."""
    name, p = _wavelet(wave, "DiscreteWaveletTransform")
    return Expr(name) if p is None else Expr(name, R(p))


# ── one-dimensional analysis/synthesis along an axis (periodic padding, Wolfram's documented index formula) ──
def _analysis_idx(n_in, fl):
    m = (n_in + 1) // 2
    return (2 * np.arange(m)[:, None] + np.arange(fl)[None, :]) % n_in


def _dwt_axis(x, f, axis):
    """out[..., n, ...] = Σ_i f_i x[..., (2n + i) mod L, ...], n = 0..Ceiling[L/2]-1 (0-based)."""
    xm = np.moveaxis(x, axis, -1)
    idx = _analysis_idx(xm.shape[-1], len(f))
    fa = np.array(f, dtype=object if xm.dtype == object else None)
    out = (xm[..., idx] * fa).sum(axis=-1)
    return np.moveaxis(out, -1, axis)


def _odd_matrix(n_in, lo, hi):
    """Stacked analysis matrix (2 Ceiling[L/2] x L) for an odd length (machine only)."""
    m = (n_in + 1) // 2
    A = np.zeros((2 * m, n_in))
    idx = _analysis_idx(n_in, len(lo))
    for n in range(m):
        for i in range(len(lo)):
            A[n, idx[n, i]] += lo[i]
            A[m + n, idx[n, i]] += hi[i]
    return A


def _idwt_axis(c_lo, c_hi, lo, hi, n_out, axis):
    """Inverse of the pair (_dwt_axis(lo), _dwt_axis(hi)) for an original length n_out: the transpose (orthogonal for
    even lengths) or, for odd lengths, the least-squares pseudo-inverse."""
    a = np.moveaxis(c_lo, axis, 0)
    b = np.moveaxis(c_hi, axis, 0)
    exact = a.dtype == object or b.dtype == object
    if n_out % 2 == 0:
        dt = object if exact else np.result_type(a.dtype, b.dtype, float)
        out = np.zeros((n_out,) + a.shape[1:], dtype=dt)
        if exact:
            out[...] = sp.Integer(0)
        idx = _analysis_idx(n_out, len(lo))
        for i in range(len(lo)):
            np.add.at(out, idx[:, i], lo[i] * a + hi[i] * b)
        return np.moveaxis(out, 0, axis)
    if exact:
        raise NotImplementedError("InverseWaveletTransform: exact transforms of odd-length data are not supported")
    if n_out > 4096:
        raise NotImplementedError("InverseWaveletTransform: odd data lengths above 4096 are not supported")
    P = np.linalg.pinv(_odd_matrix(n_out, lo, hi))
    stacked = np.concatenate([a, b], axis=0)
    out = np.tensordot(P, stacked, axes=([1], [0]))
    return np.moveaxis(out, 0, axis)


def _split(x, lo, hi, d):
    """All 2^d children of an array (child index bits: first dimension most significant)."""
    nodes = {0: x}
    for ax in range(d):
        new = {}
        for j, arr in nodes.items():
            new[2 * j] = _dwt_axis(arr, lo, ax)
            new[2 * j + 1] = _dwt_axis(arr, hi, ax)
        nodes = new
    return nodes


def _merge(children, lo, hi, d, shape):
    """Inverse of _split for a parent of the given shape; missing children are zero."""
    nodes = dict(children)
    for ax in reversed(range(d)):
        new = {}
        for j in range(2 ** ax):
            a, b = nodes[2 * j], nodes[2 * j + 1]
            new[j] = _idwt_axis(a, b, lo, hi, shape[ax], ax)
        nodes = new
    return nodes[0]


# ── stationary (undecimated) version ──
def _swt_axis(x, f, axis, step):
    xm = np.moveaxis(x, axis, -1)
    n = xm.shape[-1]
    idx = (np.arange(n)[:, None] + step * np.arange(len(f))[None, :]) % n
    fa = np.array(f, dtype=object if xm.dtype == object else None)
    out = (xm[..., idx] * fa).sum(axis=-1)
    return np.moveaxis(out, -1, axis)


def _iswt_axis(a, b, lo, hi, axis, step):
    am, bm = np.moveaxis(a, axis, -1), np.moveaxis(b, axis, -1)
    n = am.shape[-1]
    exact = am.dtype == object or bm.dtype == object
    out = np.zeros(am.shape, dtype=object if exact else np.result_type(am.dtype, bm.dtype, float))
    if exact:
        out[...] = sp.Integer(0)
    for i in range(len(lo)):
        sh = (step * i) % n
        out = out + lo[i] * np.roll(am, sh, axis=-1) + hi[i] * np.roll(bm, sh, axis=-1)
    half = sp.Rational(1, 2) if exact else 0.5
    return np.moveaxis(out * half, -1, axis)


def _split_sw(x, lo, hi, d, step):
    nodes = {0: x}
    for ax in range(d):
        new = {}
        for j, arr in nodes.items():
            new[2 * j] = _swt_axis(arr, lo, ax, step)
            new[2 * j + 1] = _swt_axis(arr, hi, ax, step)
        nodes = new
    return nodes


def _merge_sw(children, lo, hi, d, step):
    nodes = dict(children)
    for ax in reversed(range(d)):
        new = {}
        for j in range(2 ** ax):
            new[j] = _iswt_axis(nodes[2 * j], nodes[2 * j + 1], lo, hi, ax, step)
        nodes = new
    return nodes[0]


# ── lifting schemes (periodic, even lengths) ──
_CDF97 = (-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 1.149604398860241)
_LIFT_FAMILIES = ("HaarWavelet", "CDFWavelet")


def _lift_kind(wave, fname):
    if isinstance(wave, Expr) and wave.head == "CDFWavelet":
        t = wave.args[0] if wave.args else "9/7"
        if t not in ("5/3", "9/7"):
            raise ValueError(f"{fname}: CDFWavelet type must be \"5/3\" or \"9/7\"")
        return "cdf" + t.replace("/", "")
    if isinstance(wave, Symbol):
        raise Unevaluated()
    name, _ = _wavelet(wave, fname)
    if name == "HaarWavelet":
        return "haar"
    raise NotImplementedError(f"{fname}: lifting of {name} is not supported (HaarWavelet, CDFWavelet[\"5/3\"] "
                              "and CDFWavelet[\"9/7\"])")


def _lift_axis(x, kind, axis):
    xm = np.moveaxis(x, axis, -1)
    e, o = xm[..., 0::2], xm[..., 1::2]
    exact = xm.dtype == object
    half = sp.Rational(1, 2) if exact else 0.5
    if kind == "haar":
        d = o - e
        s = e + d * half
        r2 = sp.sqrt(2) if exact else _SQRT2
        s, d = s * r2, -d / r2
    elif kind == "cdf53":
        d = o - (e + np.roll(e, -1, axis=-1)) * half
        s = e + (np.roll(d, 1, axis=-1) + d) * (sp.Rational(1, 4) if exact else 0.25)
    else:
        if exact:
            raise NotImplementedError("LiftingWaveletTransform: CDFWavelet[\"9/7\"] has irrational lifting "
                                      "constants; exact transforms are not supported")
        a, b, g, dl, z = _CDF97
        d = o + a * (e + np.roll(e, -1, axis=-1))
        s = e + b * (np.roll(d, 1, axis=-1) + d)
        d = d + g * (s + np.roll(s, -1, axis=-1))
        s = s + dl * (np.roll(d, 1, axis=-1) + d)
        s, d = s * z, d / z
    return np.moveaxis(s, -1, axis), np.moveaxis(d, -1, axis)


def _unlift_axis(s, d, kind, axis):
    s, d = np.moveaxis(s, axis, -1), np.moveaxis(d, axis, -1)
    exact = s.dtype == object or d.dtype == object
    half = sp.Rational(1, 2) if exact else 0.5
    if kind == "haar":
        r2 = sp.sqrt(2) if exact else _SQRT2
        s, d = s / r2, -d * r2
        e = s - d * half
        o = d + e
    elif kind == "cdf53":
        e = s - (np.roll(d, 1, axis=-1) + d) * (sp.Rational(1, 4) if exact else 0.25)
        o = d + (e + np.roll(e, -1, axis=-1)) * half
    else:
        a, b, g, dl, z = _CDF97
        s, d = s / z, d * z
        s = s - dl * (np.roll(d, 1, axis=-1) + d)
        d = d - g * (s + np.roll(s, -1, axis=-1))
        e = s - b * (np.roll(d, 1, axis=-1) + d)
        o = d - a * (e + np.roll(e, -1, axis=-1))
    n = e.shape[-1] * 2
    out = np.empty(e.shape[:-1] + (n,), dtype=np.result_type(e.dtype, o.dtype))
    out[..., 0::2], out[..., 1::2] = e, o
    return np.moveaxis(out, -1, axis)


def _split_lift(x, kind, d):
    nodes = {0: x}
    for ax in range(d):
        new = {}
        for j, arr in nodes.items():
            new[2 * j], new[2 * j + 1] = _lift_axis(arr, kind, ax)
        nodes = new
    return nodes


def _merge_lift(children, kind, d):
    nodes = dict(children)
    for ax in reversed(range(d)):
        new = {}
        for j in range(2 ** ax):
            new[j] = _unlift_axis(nodes[2 * j], nodes[2 * j + 1], kind, ax)
        nodes = new
    return nodes[0]


# ═══════════════════════════════════════════════════════════════ wavelet trees ══
_TRANSFORMS = {
    "DiscreteWaveletTransform": ("dwt", False),
    "DiscreteWaveletPacketTransform": ("dwt", True),
    "StationaryWaveletTransform": ("swt", False),
    "StationaryWaveletPacketTransform": ("swt", True),
    "LiftingWaveletTransform": ("lift", False),
}


class _Ops:
    """Split/merge operations of one transform type for one wavelet."""

    def __init__(self, transform, wave, exact, fname):
        self.kind, self.packet = _TRANSFORMS[transform]
        self.exact = exact
        if self.kind == "lift":
            self.lift = _lift_kind(wave, fname)
        else:
            self.lo, self.hi = _filters(wave, fname, exact)

    def split(self, arr, depth):
        d = arr.ndim
        if self.kind == "dwt":
            return _split(arr, self.lo, self.hi, d)
        if self.kind == "swt":
            return _split_sw(arr, self.lo, self.hi, d, 2 ** depth)
        return _split_lift(arr, self.lift, d)

    def merge(self, children, shape, depth):
        d = len(shape)
        if self.kind == "dwt":
            return _merge(children, self.lo, self.hi, d, shape)
        if self.kind == "swt":
            return _merge_sw(children, self.lo, self.hi, d, 2 ** depth)
        return _merge_lift(children, self.lift, d)


def _forward(ops, x, r):
    """{wind: array} of all computed nodes (wind = tuple of child indices)."""
    tree = {}
    frontier = [((), x)]
    for depth in range(r):
        new = []
        for w, arr in frontier:
            if ops.kind == "dwt" and ops.exact and any(n % 2 for n in arr.shape):
                raise NotImplementedError("exact (WorkingPrecision -> Infinity) transforms need even dimensions at "
                                          "every refinement level")
            ch = ops.split(arr, depth)
            for j in sorted(ch):
                tree[w + (j,)] = ch[j]
                if ops.packet or j == 0:
                    new.append((w + (j,), ch[j]))
        frontier = new
    return tree


def _leaves(tree):
    return [w for w in tree if not any((w + (0,)) == v for v in tree)]


def _sort_winds(ws):
    return sorted(ws, key=lambda w: (len(w), w))


def _subtree_selected(w, sel):
    return any(len(s) >= len(w) and s[:len(w)] == w for s in sel)


def _reconstruct(ops, tree, sel, root_shape, d, at=()):
    """Value of node `at` reconstructed from the coefficients in `sel` (others are zero)."""
    if at in sel:
        return tree[at]
    kids = [at + (j,) for j in range(2 ** d)]
    if not all(k in tree for k in kids) or not _subtree_selected(at, sel):
        return None
    vals = {}
    for j, k in enumerate(kids):
        v = _reconstruct(ops, tree, sel, root_shape, d, k)
        if v is None:
            v = _zeros_like(tree[k])
        vals[j] = v
    shape = tree[at].shape if at else root_shape
    return ops.merge(vals, shape, len(at))


def _zeros_like(a):
    if a.dtype == object:
        z = np.empty(a.shape, dtype=object)
        z[...] = sp.Integer(0)
        return z
    return np.zeros(a.shape, dtype=a.dtype)


def _close_arrays(a, b, exact):
    if exact:
        diff = [sp.N(sp.expand(u - v), 30) for u, v in zip(a.ravel(), b.ravel())]
        return all(abs(complex(t)) < 1e-20 for t in diff if t.is_number) and all(
            sp.simplify(t) == 0 for t in diff if not t.is_number)
    scale = max(1.0, float(np.max(np.abs(a))) if a.size else 1.0)
    return a.shape == b.shape and float(np.max(np.abs(a - b))) <= 1e-9 * scale


class _DWD:
    """Python view of a DiscreteWaveletData expression."""

    def __init__(self, tree, wave, transform, meta):
        self.tree, self.wave, self.transform, self.meta = tree, wave, transform, meta

    @property
    def exact(self):
        return _is_sym(self.meta.get("WorkingPrecision"), "Infinity")

    @property
    def dims(self):
        return tuple(int(t) for t in self.meta["DataDimensions"].args)

    @property
    def root_shape(self):
        p = self.meta.get("PaddedDimensions")
        return tuple(int(t) for t in p.args) if p is not None else self.dims

    @property
    def basis(self):
        return [_key(w) for w in self.meta["BasisIndex"].args]

    @property
    def r(self):
        return int(self.meta["Refinement"])

    def ops(self, wave=None, fname="InverseWaveletTransform"):
        return _Ops(self.transform, self.wave if wave is None else wave, self.exact, fname)

    def expr(self):
        pairs = [(_wind_expr(w), _to_list(self.tree[w], self.exact)) for w in _sort_winds(self.tree)]
        meta = [(k, v) for k, v in self.meta.items()]
        return Expr("DiscreteWaveletData", _assoc(pairs), self.wave, self.transform, _assoc(meta))

    def inverse(self, sel=None, wave=None, fname="InverseWaveletTransform"):
        sel = set(self.basis if sel is None else sel)
        ops = self.ops(wave, fname)
        d = len(self.dims)
        v = _reconstruct(ops, self.tree, sel, self.root_shape, d)
        if v is None:
            v = np.zeros(self.root_shape) if not self.exact else _zeros_like(np.empty(self.root_shape, dtype=object))
        if self.root_shape != self.dims:
            v = v[tuple(slice(0, n) for n in self.dims)]
        return v


def _meta(dims, r, basis, padding="Periodic", exact=False, extra=()):
    m = {"DataDimensions": L(*dims), "Refinement": r, "BasisIndex": L(*[_wind_expr(w) for w in basis]),
         "Padding": padding, "WorkingPrecision": _sym("Infinity" if exact else "MachinePrecision")}
    m.update(dict(extra))
    return m


def _parse_dwd(e, fname):
    """DiscreteWaveletData expression -> _DWD (canonical form only)."""
    if not (isinstance(e, Expr) and e.head == "DiscreteWaveletData"):
        if isinstance(e, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a DiscreteWaveletData object, not {e!r}")
    if not (len(e.args) == 4 and isinstance(e.args[0], Expr) and e.args[0].head == "Association"
            and isinstance(e.args[2], str) and e.args[2] in _TRANSFORMS and isinstance(e.args[3], Expr)
            and e.args[3].head == "Association"):
        raise ValueError(f"{fname}: the DiscreteWaveletData object is not in standard form")
    meta = {}
    for r in e.args[3].args:
        meta[r.args[0]] = r.args[1]
    exact = _is_sym(meta.get("WorkingPrecision"), "Infinity")
    tree = {}
    for r in e.args[0].args:
        w = _key(r.args[0])
        if not isinstance(w, tuple):
            raise ValueError(f"{fname}: invalid wavelet index {r.args[0]!r}")
        tree[w] = _sym_array(r.args[1], fname) if exact else _from_list(r.args[1])
    return _DWD(tree, e.args[1], e.args[2], meta)


def _default_r(dims, transform):
    r = int(math.floor(math.log2(min(dims)) + 0.5))
    if transform in ("DiscreteWaveletPacketTransform", "StationaryWaveletPacketTransform"):
        r = min(r, 4)
    return max(r, 1)


def _refinement(v, dims, transform, fname):
    if v is None or _is_sym(v, "Automatic"):
        if transform == "LiftingWaveletTransform":
            ks = []
            for n in dims:
                k = 0
                while n % 2 == 0 and n > 1:
                    n //= 2
                    k += 1
                ks.append(k)
            return max(1, min(ks))
        return _default_r(dims, transform)
    if _is_sym(v, "Full"):
        return max(1, int(math.floor(math.log2(min(dims)) + 0.5)))
    if isinstance(v, Symbol):
        raise Unevaluated()
    if isinstance(v, bool) or not K.is_int(v):
        raise TypeError(f"{fname}: the refinement level must be a positive integer, not {v!r}")
    r = int(v)
    if r < 1:
        raise ValueError(f"{fname}: the refinement level must be a positive integer, not {r}")
    if transform != "LiftingWaveletTransform" and 2 ** (r - 1) > min(dims) * 64:
        raise ValueError(f"{fname}: the refinement level {r} is too large for data of dimensions {dims}")
    return r


def _wtransform(transform):
    def impl(data: ListOf[Number], *rest: Expr) -> Expr:
        fname = transform
        allowed = {"WorkingPrecision", "Method"} | ({"Padding"} if transform in (
            "DiscreteWaveletTransform", "DiscreteWaveletPacketTransform", "LiftingWaveletTransform") else set())
        pos, opts = _opts(rest, allowed, fname)
        if len(pos) > 2:
            raise TypeError(f"{fname}: called with {len(pos) + 1} positional arguments; 1 to 3 are expected")
        wave = pos[0] if pos and not _is_sym(pos[0], "Automatic") else Expr("HaarWavelet")
        if isinstance(wave, Symbol):
            raise Unevaluated()
        meth = opts.get("Method")
        if meth is not None and not _is_sym(meth, "Automatic"):
            raise NotImplementedError(f"{fname}: Method -> {meth!r} is not supported")
        wp = _wp_opt(opts["WorkingPrecision"], fname) if "WorkingPrecision" in opts else "m"
        if wp not in ("m", "exact"):
            raise NotImplementedError(f"{fname}: only WorkingPrecision -> MachinePrecision or Infinity is supported")
        exact = wp == "exact"
        if isinstance(data, Expr) and data.head in ("Image", "Audio", "Sound", "Image3D", "Video"):
            raise NotImplementedError(f"{fname}: {data.head} input is not supported (give the numeric array, e.g. "
                                      "ImageData[img])")
        if not is_list(data):
            if isinstance(data, (Symbol, Expr)) and not isinstance(data, str):
                raise Unevaluated()
            raise TypeError(f"{fname}: expected a numeric array, not {data!r}")
        if _sym_data(data):
            if not exact:
                raise Unevaluated()
            x = _sym_array(data, fname)
        else:
            x = _data_array(data, fname, exact=exact)
        if x.ndim > 3:
            raise NotImplementedError(f"{fname}: arrays of depth {x.ndim} are not supported (at most 3)")
        dims = x.shape
        wave_c = wave if transform == "LiftingWaveletTransform" and isinstance(wave, Expr) and \
            wave.head == "CDFWavelet" else _wave_expr(wave)
        ops = _Ops(transform, wave_c, exact, fname)
        r = _refinement(pos[1] if len(pos) > 1 else None, dims, transform, fname)
        extra = []
        padding = "Periodic"
        xin = x
        if "Padding" in opts:
            pv = opts["Padding"]
            if transform == "LiftingWaveletTransform" and not isinstance(pv, str):
                try:
                    padding = float(_num(pv)) if not exact else sp.sympify(S(pv))
                except TypeError:
                    raise ValueError(f"{fname}: Padding must be \"Periodic\" or a number")
            elif pv != "Periodic":
                raise NotImplementedError(f"{fname}: Padding -> {pv!r} is not supported (only \"Periodic\")")
        if transform == "LiftingWaveletTransform":
            m = 2 ** r
            padded = tuple(-(-n // m) * m for n in dims)
            if padded != dims:
                widths = [(0, p - n) for p, n in zip(padded, dims)]
                if padding == "Periodic":
                    if any(p > 2 * n for p, n in zip(padded, dims)):
                        raise NotImplementedError(f"{fname}: periodic pre-padding to {padded} needs more than one "
                                                  "period of the data")
                    xin = np.pad(x, widths, mode="wrap")
                else:
                    xin = np.pad(x, widths, mode="constant", constant_values=padding)
                extra.append(("PaddedDimensions", L(*padded)))
        tree = _forward(ops, xin, r)
        basis = _sort_winds([w for w in tree if len(w) == r] if ops.packet else _leaves(tree))
        dwd = _DWD(tree, wave_c, transform, _meta(dims, r, basis, padding if isinstance(padding, str) else
                                                     R(padding), exact, extra))
        back = dwd.inverse(fname=fname)
        if not _close_arrays(back, x, exact):
            raise ArithmeticError(f"{fname}: result failed verification")
        return dwd.expr()
    return impl


_WT_NOTE = ("Supported wavelets: HaarWavelet (default) and DaubechiesWavelet[n] (orthonormal filters √2 h_k from "
            "WaveletFilterCoefficients, in increasing k order); other families raise NotImplementedError (their "
            "filters are not available). Numeric arrays of depth 1-3 (machine reals or complex numbers); "
            "WorkingPrecision -> Infinity gives exact coefficients (also for symbolic data) for Haar and "
            "DaubechiesWavelet[1|2] with even dimensions at every level; symbolic data at machine precision stays "
            "unevaluated; Image/Audio input raises NotImplementedError (use the numeric array). The result is a "
            "DiscreteWaveletData object (DiscreteWaveletData[<|wind -> coef, ...|>, wave, transform, <|properties|>]; "
            "multi-dimensional child indices j = FromDigits[{b1, ..., bd}, 2] with b = 0 lowpass, 1 highpass "
            "along each dimension).")

fn("DiscreteWaveletTransform", area=AREA, status="added",
   notes="DiscreteWaveletTransform[data, wave, r] (r refinement levels, default Floor[Log2[n] + 1/2] for the "
         "minimum dimension n, or Full): coarse/detail coefficients c_j[n] = √2 Σ_i p_i c_(j-1)[Mod[2n - 2 + i, L, "
         "1]], d_j[n] = √2 Σ_i q_i c_(j-1)[...], n = 1..Ceiling[L/2] (Wolfram's documented periodic formula); only "
         "{0, ..., 0} is refined. Padding -> \"Periodic\" only (other paddings raise NotImplementedError). "
         "Deviation: for filters longer than 2 taps the coefficient arrays have length Ceiling[L/2] (periodisation), "
         "where Wolfram's padded arrays have Ceiling[(L + fl - 2)/2]; the Haar coefficients agree exactly. "
         + _WT_NOTE + " Check: self-check (the inverse transform of the result reproduces the data, ArithmeticError "
         "otherwise); reference (Wolfram documentation Haar examples, energy conservation) in tests.")(
    _wtransform("DiscreteWaveletTransform"))
fn("DiscreteWaveletPacketTransform", area=AREA, status="added",
   notes="DiscreteWaveletPacketTransform[data, wave, r]: the full tree of DiscreteWaveletTransform refinements (every "
         "node is split; default r = Min[Floor[Log2[n] + 1/2], 4]); \"BasisIndex\" is the deepest level (use "
         "WaveletBestBasis to choose another basis). Same filters, padding and length rules as "
         "DiscreteWaveletTransform. " + _WT_NOTE + " Check: self-check (inverse reproduces the data); reference "
         "(Haar packet values derived by hand, Parseval) in tests.")(_wtransform("DiscreteWaveletPacketTransform"))
fn("StationaryWaveletTransform", area=AREA, status="added",
   notes="StationaryWaveletTransform[data, wave, r]: undecimated (à trous) transform with periodic padding, "
         "c_j[n] = Σ_i f_i c_(j-1)[Mod[n + 2^(j-1) (i - 1), L]] with f = √2 p (lowpass) or √2 q (highpass), every "
         "array has the data's dimensions (default r = Floor[Log2[n] + 1/2]); the inverse averages the two "
         "reconstructions, c_(j-1) = (1/2) Σ_children Σ_i f_i c_j[n - 2^(j-1)(i - 1)]. The filter alignment (a "
         "circular shift) is this implementation's; rotating the data rotates the coefficients. " + _WT_NOTE +
         " Check: self-check (inverse reproduces the data); reference (shift-equivariance and the level-1 Haar "
         "values (x_n ± x_(n+1))/√2) in tests.")(_wtransform("StationaryWaveletTransform"))
fn("StationaryWaveletPacketTransform", area=AREA, status="added",
   notes="StationaryWaveletPacketTransform[data, wave, r]: the full undecimated tree (every node split as in "
         "StationaryWaveletTransform; default r = Min[Floor[Log2[n] + 1/2], 4]); \"BasisIndex\" is the deepest "
         "level. " + _WT_NOTE + " Check: self-check (inverse reproduces the data); reference (Haar values by hand) "
         "in tests.")(_wtransform("StationaryWaveletPacketTransform"))
fn("LiftingWaveletTransform", area=AREA, status="added",
   notes="LiftingWaveletTransform[data, wave, r] with periodic lifting steps (Daubechies & Sweldens 1998), even = "
         "x[[1;;;;2]], odd = x[[2;;;;2]]: HaarWavelet (default: d = o - e, s = e + d/2, output √2 s and -d/√2, i.e. "
         "the orthonormal Haar coefficients), CDFWavelet[\"5/3\"] (d = o - (e_n + e_(n+1))/2, s = e + (d_(n-1) + "
         "d_n)/4, no scaling) and CDFWavelet[\"9/7\"] (α, β, γ, δ = -1.586134342, -0.052980119, 0.882911076, "
         "0.443506852, then s ζ and d/ζ with ζ = 1.149604399). Default r = Max[1, Min over dimensions of "
         "IntegerExponent[n, 2]], or Full; the data are pre-padded to multiples of 2^r with Padding (\"Periodic\" "
         "default, or a constant) and the inverse crops back. Other wavelets and Method -> \"IntegerLifting\" raise "
         "NotImplementedError; exact input with WorkingPrecision -> Infinity is supported for Haar and 5/3. "
         "Check: self-check (inverse reproduces the data); reference (hand-computed 5/3 and Haar values) in tests.")(
    _wtransform("LiftingWaveletTransform"))


# ═══════════════════════════════════════════════════════════════ DiscreteWaveletData ══
def _dwd_from_rules(rules, wave, transform, rest, fname="DiscreteWaveletData"):
    """DiscreteWaveletData[{wind -> coef, ...}, wave, wtrans, opts]: standard form, after checking that the tree is
    complete (every split node has all 2^d children)."""
    _pos, opts = _opts(rest, {"DataDimensions", "WorkingPrecision", "Padding", "Method"}, fname)
    if not isinstance(transform, str) or transform not in _TRANSFORMS:
        raise ValueError(f"{fname}: unknown transform {transform!r}")
    exact = "WorkingPrecision" in opts and _wp_opt(opts["WorkingPrecision"], fname) == "exact"
    tree = {}
    for r in rules:
        if not is_rule(r):
            raise TypeError(f"{fname}: expected rules wind -> coef, not {r!r}")
        w = _key(r.args[0])
        if not isinstance(w, tuple) or not w or any(t < 0 for t in w):
            raise ValueError(f"{fname}: invalid wavelet index {r.args[0]!r}")
        tree[w] = _sym_array(r.args[1], fname) if exact else _from_list(r.args[1])
    if not tree:
        raise ValueError(f"{fname}: no coefficients given")
    d = next(iter(tree.values())).ndim
    for w in list(tree):
        if any(t >= 2 ** d for t in w):
            raise ValueError(f"{fname}: wavelet index {list(w)} is out of range for {d}-dimensional coefficients")
        for k in range(1, len(w)):
            if w[:k] not in tree and len(w[:k]) > 0:
                pass
        sib = [w[:-1] + (j,) for j in range(2 ** d)]
        if not all(s in tree for s in sib):
            raise ValueError(f"{fname}: the coefficients of {list(w)} are given without all of their siblings")
    r = max(len(w) for w in tree)
    if "DataDimensions" in opts:
        dims = tuple(int(t) for t in opts["DataDimensions"].args)
    else:
        kind = _TRANSFORMS[transform][0]
        lev1 = tree.get((0,))
        if lev1 is None:
            raise ValueError(f"{fname}: the level-1 coefficients are missing")
        dims = lev1.shape if kind == "swt" else tuple(2 * n for n in lev1.shape)
    basis = _sort_winds(_leaves(tree))
    dwd = _DWD(tree, wave, transform, _meta(dims, r, basis, "Periodic", exact))
    # the coarser nodes must be consistent with their children (they are recomputed from the leaves)
    ops = dwd.ops(fname=fname)
    for w in _sort_winds([w for w in tree if (w + (0,)) in tree])[::-1]:
        tree[w] = _reconstruct(ops, tree, set(_leaves(tree)), dims, d, w)
    return dwd.expr()


def _dwd_ctor(*args: Expr) -> Expr:
    fname = "DiscreteWaveletData"
    if not args:
        raise TypeError(f"{fname}: called with no arguments")
    a0 = args[0]
    if isinstance(a0, Expr) and a0.head == "Association" and len(args) == 4:
        _parse_dwd(Expr(fname, *args), fname)
        raise Unevaluated()          # already in standard form
    if is_list(a0):
        if len(args) < 3:
            raise TypeError(f"{fname}: expected DiscreteWaveletData[{{wind -> coef, ...}}, wave, wtrans]")
        return _dwd_from_rules(a0.args, args[1], args[2], args[3:])
    if isinstance(a0, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: expected a list of rules wind -> coef")


fn("DiscreteWaveletData", area=AREA, status="added",
   notes="DiscreteWaveletData[{wind -> coef, ...}, wave, wtrans] builds the standard form DiscreteWaveletData[<|wind "
         "-> coef, ...|>, wave, wtrans, <|properties|>] produced by the wavelet transforms (coarser nodes are "
         "recomputed from the leaves; incomplete sibling sets raise ValueError). Access: dwd[wind], "
         "dwd[{wind1, ...}], dwd[patt] (MatchQ on the index), dwd[All], dwd[Automatic] (the basis used by the "
         "inverse) give rules; dwd[spec, form] with form \"Rules\", \"Values\", \"Inverse\", \"ListPlot\", "
         "\"MatrixPlot\" or a list of forms; properties \"Properties\", \"BasisIndex\", \"Dimensions\", "
         "\"EnergyFraction\", \"Padding\", \"Refinement\", \"Transform\", \"Wavelet\", \"WaveletIndex\", "
         "\"DataDimensions\", \"DataChannels\", \"DataWrapper\", \"BestBasisCostValues\" (after WaveletBestBasis), "
         "\"ThresholdValues\" (after WaveletThreshold); Normal[dwd] gives all rules. \"TreeView\", "
         "\"BestBasisBlockView\" and tables (front-end views) and Image/Sound forms raise NotImplementedError. "
         "Check: exact (structural accessors); reference (Wolfram documentation accessor examples) in tests.")(
    _dwd_ctor)

_DWD_PROPS = ("BasisIndex", "DataChannels", "DataDimensions", "DataWrapper", "Dimensions", "EnergyFraction", "Padding",
              "Refinement", "Transform", "Wavelet", "WaveletIndex")
_FRONT_END_PROPS = ("TreeView", "BestBasisBlockView", "BestBasisCostTable", "ThresholdTable", "IndexMap")


def _energy(a):
    if a.dtype == object:
        return sum(sp.Abs(v) ** 2 for v in a.ravel())
    return float(np.sum(np.abs(a) ** 2))


def _select(winds, spec, basis, fname):
    """Wavelet indices selected by a specification (wind, list of winds, All, Automatic, or a pattern)."""
    if _is_sym(spec, "All"):
        return _sort_winds(winds)
    if _is_sym(spec, "Automatic"):
        return list(basis)
    w = _key(spec)
    if isinstance(w, tuple):
        if w not in winds:
            raise ValueError(f"{fname}: there are no coefficients with wavelet index {list(w)}")
        return [w]
    if is_list(spec) and spec.args and all(isinstance(_key(t), tuple) for t in spec.args):
        out = []
        for t in spec.args:
            out.extend(_select(winds, t, basis, fname))
        return out
    out = [w for w in _sort_winds(winds) if ev(Expr("MatchQ", _wind_expr(w), spec)) is True]
    return out


def _dwd_forms(dwd, ws, form, fname):
    forms = list(form.args) if is_list(form) else [form]
    out = None
    for f in forms:
        if isinstance(f, Expr) and f.head == "List":
            f = f.args[0] if f.args else f
        if f == "Rules":
            continue
        if f == "Values":
            out = "values"
        elif f == "Inverse":
            out = out or "rules"
            ws = [("inv", w) for w in ws]
        elif f in ("ListPlot", "MatrixPlot"):
            out = f
        elif f in ("Image", "Sound", "SampledSoundList", "Audio"):
            raise NotImplementedError(f"{fname}: the form {f!r} is not supported")
        else:
            raise ValueError(f"{fname}: unknown output form {f!r}")
    vals = []
    for w in ws:
        if isinstance(w, tuple) and w and w[0] == "inv":
            w = w[1]
            vals.append((w, _to_list(dwd.inverse(sel=[w]), dwd.exact)))
        else:
            vals.append((w, _to_list(dwd.tree[w], dwd.exact)))
    if out == "values":
        return L(*[v for _, v in vals])
    if out in ("ListPlot", "MatrixPlot"):
        return L(*[_rule(_wind_expr(w), ev(Expr("ListLinePlot" if out == "ListPlot" else "MatrixPlot", v)))
                   for w, v in vals])
    return L(*[_rule(_wind_expr(w), v) for w, v in vals])


def _dwd_prop(dwd, p, fname):
    tree = dwd.tree
    if p == "Properties":
        props = list(_DWD_PROPS)
        if "BestBasisCostValues" in dwd.meta:
            props.append("BestBasisCostValues")
        if "ThresholdValues" in dwd.meta:
            props.append("ThresholdValues")
        return L(*sorted(props))
    if p in ("BasisIndex", "DataDimensions", "Padding", "Refinement", "BestBasisCostValues", "ThresholdValues"):
        if p not in dwd.meta:
            raise ValueError(f"{fname}: the property {p!r} is not available for this object")
        return dwd.meta[p]
    if p == "Wavelet":
        return dwd.wave
    if p == "Transform":
        return dwd.transform
    if p == "WaveletIndex":
        return L(*[_wind_expr(w) for w in _sort_winds(tree)])
    if p == "Dimensions":
        return L(*[_rule(_wind_expr(w), L(*tree[w].shape)) for w in _sort_winds(tree)])
    if p == "EnergyFraction":
        es = {w: _energy(tree[w]) for w in dwd.basis}
        tot = sum(es.values())
        if tot == 0:
            raise ValueError(f"{fname}: the coefficients have zero energy")
        return L(*[_rule(_wind_expr(w), R(sp.simplify(es[w] / tot)) if dwd.exact else es[w] / tot)
                   for w in dwd.basis])
    if p == "DataChannels":
        return 1
    if p == "DataWrapper":
        return _sym("Identity")
    if p in _FRONT_END_PROPS or (is_list(p) and p.args and p.args[0] in _FRONT_END_PROPS):
        raise NotImplementedError(f"{fname}: the property {p!r} is a front-end view and is not supported")
    raise ValueError(f"{fname}: unknown property {p!r}")


def _dwd_hook(head, args, ev_):
    fname = "DiscreteWaveletData"
    try:
        dwd = _parse_dwd(head, fname)
    except (ValueError, TypeError):
        return K.UNHANDLED
    if not 1 <= len(args) <= 2:
        raise TypeError(f"{fname}: dwd[spec] or dwd[spec, form] expected")
    spec = args[0]
    if isinstance(spec, str) or (is_list(spec) and spec.args and isinstance(spec.args[0], str)):
        if len(args) != 1:
            raise TypeError(f"{fname}: properties take no further arguments")
        return _dwd_prop(dwd, spec, fname)
    ws = _select(list(dwd.tree), spec, dwd.basis, fname)
    if len(args) == 2:
        return _dwd_forms(dwd, ws, args[1], fname)
    return L(*[_rule(_wind_expr(w), _to_list(dwd.tree[w], dwd.exact)) for w in ws])


KL.COMPOUND_HEAD_HOOKS["DiscreteWaveletData"] = _dwd_hook


# ═══════════════════════════════════════════════════════════════ InverseWaveletTransform ══
def _wind_set(spec, dwd, fname):
    ws = _select(list(dwd.tree), spec, dwd.basis, fname)
    for a in ws:
        for b in ws:
            if a != b and len(a) < len(b) and b[:len(a)] == a:
                raise ValueError(f"{fname}: the wavelet index specification is inconsistent ({list(a)} and its "
                                 f"descendant {list(b)} are both included)")
    return ws


def _iwt(dwd_e: Expr, wave: Expr = None, spec: Expr = None) -> ListOf[Number] | Expr:
    fname = "InverseWaveletTransform"
    dwd = _parse_dwd(dwd_e, fname)
    w = None if wave is None or _is_sym(wave, "Automatic") else wave
    if w is not None and isinstance(w, Symbol):
        raise Unevaluated()
    if w is not None and dwd.transform != "LiftingWaveletTransform":
        w = _wave_expr(w)
    if spec is not None and K.is_int(spec) and not isinstance(spec, bool):
        r = int(spec)
        if not 1 <= r <= dwd.r:
            raise ValueError(f"{fname}: the number of levels to invert must be between 1 and {dwd.r}")
        keep = dwd.r - r
        if keep == 0:
            return _to_list(dwd.inverse(wave=w, fname=fname), dwd.exact)
        ops = dwd.ops(w, fname)
        d = len(dwd.dims)
        sel = set(dwd.basis)
        new_tree = {}
        for v in dwd.tree:
            if len(v) < keep:
                new_tree[v] = dwd.tree[v]
            elif len(v) == keep:
                val = _reconstruct(ops, dwd.tree, sel, dwd.root_shape, d, v)
                new_tree[v] = val if val is not None else _zeros_like(dwd.tree[v])
        basis = _sort_winds([v for v in new_tree if v in sel or (len(v) == keep and _subtree_selected(v, sel))])
        meta = dict(dwd.meta)
        meta["Refinement"] = keep
        meta["BasisIndex"] = L(*[_wind_expr(v) for v in basis])
        return _DWD(new_tree, dwd.wave if w is None else w, dwd.transform, meta).expr()
    sel = None if spec is None or _is_sym(spec, "Automatic") else _wind_set(spec, dwd, fname)
    return _to_list(dwd.inverse(sel=sel, wave=w, fname=fname), dwd.exact)


fn("InverseWaveletTransform", area=AREA, status="added",
   notes="InverseWaveletTransform[dwd] reconstructs the data from the coefficients in dwd[\"BasisIndex\"]; "
         "InverseWaveletTransform[dwd, wave] uses another wavelet's filters (Automatic = dwd[\"Wavelet\"]); "
         "InverseWaveletTransform[dwd, wave, wind] uses only the coefficients selected by wind (index, list, "
         "pattern, All, Automatic; others are zero; a selection containing a node and one of its descendants "
         "raises ValueError); InverseWaveletTransform[dwd, wave, r] inverts the r lowest levels and returns a "
         "DiscreteWaveletData with the remaining levels. Synthesis is the transpose of the periodic analysis "
         "(orthogonal for even lengths), the least-squares pseudo-inverse for odd lengths, the averaged "
         "undecimated synthesis for stationary transforms and the reversed lifting steps for lifting transforms. "
         "Check: self-check (every forward transform verifies this inverse on its own data); reference (sum of "
         "the single-index inverses equals the data, Wolfram documentation example) in tests.")(_iwt)


# ═══════════════════════════════════════════════════════════════ WaveletBestBasis ══
def _cost_fn(spec, fname):
    if spec is None or spec == "ShannonEntropy":
        def f(a):
            v = np.abs(a.astype(complex) if a.dtype == object else a).ravel() ** 2
            v = v[v > 0]
            return float(-np.sum(v * np.log(v)))
        return f
    if spec == "LogEnergy":
        def f(a):
            v = np.abs(a.astype(complex) if a.dtype == object else a).ravel() ** 2
            v = v[v > 0]
            return float(np.sum(np.log(v)))
        return f
    if is_list(spec) and len(spec.args) == 2 and spec.args[0] in ("Norm", "Threshold"):
        p = _num(spec.args[1])
        if isinstance(p, complex) or p < 0:
            raise ValueError(f"{fname}: the parameter of {spec.args[0]!r} must be a non-negative real number")
        if spec.args[0] == "Norm":
            if p < 1:
                raise NotImplementedError(f"{fname}: {{\"Norm\", p}} with p < 1 is not supported")
            return lambda a: float(np.sum(np.abs(a.astype(complex) if a.dtype == object else a) ** p))
        return lambda a: float(np.sum(np.abs(a.astype(complex) if a.dtype == object else a) > p))
    if isinstance(spec, str):
        raise ValueError(f"{fname}: unknown cost function {spec!r}")

    def f(a):
        v = apply(spec, _to_list(a, a.dtype == object))
        try:
            c = _num(v)
        except TypeError:
            raise ValueError(f"{fname}: the cost function did not return a real number: {v!r}")
        if isinstance(c, complex):
            raise ValueError(f"{fname}: the cost function did not return a real number: {v!r}")
        return c
    return f


def _best_basis(dwd_e: Expr, cspec: Expr = None) -> Expr:
    fname = "WaveletBestBasis"
    dwd = _parse_dwd(dwd_e, fname)
    cost = _cost_fn(cspec, fname)
    d = len(dwd.dims)
    costs = {w: cost(a) for w, a in dwd.tree.items()}

    def best(w):
        kids = [w + (j,) for j in range(2 ** d)]
        own = costs[w] if w else None
        if all(k in dwd.tree for k in kids):
            sub = [best(k) for k in kids]
            tot = sum(c for c, _ in sub)
            if own is None or tot < own:
                return tot, [b for _, bs in sub for b in bs]
        return own, [w]
    _c, basis = best(())
    meta = dict(dwd.meta)
    meta["BasisIndex"] = L(*[_wind_expr(w) for w in _sort_winds(basis)])
    meta["BestBasisCostValues"] = L(*[_rule(_wind_expr(w), costs[w]) for w in _sort_winds(dwd.tree)])
    return _DWD(dwd.tree, dwd.wave, dwd.transform, meta).expr()


fn("WaveletBestBasis", area=AREA, status="added",
   notes="WaveletBestBasis[dwd] / WaveletBestBasis[dwd, cspec]: the complete basis of the coefficient tree with the "
         "least total cost (Coifman-Wickerhauser bottom-up search: a node is kept when its cost does not exceed the "
         "summed best cost of its children); sets \"BasisIndex\" and \"BestBasisCostValues\". Costs: "
         "\"ShannonEntropy\" (default, -Σ w² Log[w²] over nonzero w, the additive Wickerhauser form), \"LogEnergy\" "
         "(Σ Log[w²]), {\"Norm\", p} (Σ |w|^p, p >= 1; p < 1 raises NotImplementedError), {\"Threshold\", δ} "
         "(count of |w| > δ), or a function f applied to each coefficient array (assumed additive). Check: exact "
         "(the minimisation is exhaustive over the tree); reference (hand-computed Haar packet costs) in tests.")(
    _best_basis)


# ═══════════════════════════════════════════════════════════════ WaveletMapIndexed ══
def _map_coef(f, arr, w, exact, fname, wexpr=None):
    v = apply(f, _to_list(arr, exact), wexpr if wexpr is not None else _wind_expr(w))
    if not is_list(v):
        raise ValueError(f"{fname}: f must return an array of the same dimensions as its coefficient array, not "
                         f"{v!r}")
    out = _sym_array(v, fname) if exact else _from_list(v)
    if out.shape != arr.shape:
        raise ValueError(f"{fname}: f returned an array of dimensions {out.shape} for coefficients of dimensions "
                         f"{arr.shape}")
    return out


def _wmap(f: Fn, data: Expr, spec: Expr = None) -> Expr:
    fname = "WaveletMapIndexed"
    if isinstance(data, Expr) and data.head == "ContinuousWaveletData":
        cwd = _parse_cwd(data, fname)
        ws = _cwd_select(cwd, spec if spec is not None else _sym("All"), fname)
        for w in ws:
            cwd.coefs[w] = _map_coef(f, cwd.coefs[w], w, False, fname)
        return cwd.expr()
    dwd = _parse_dwd(data, fname)
    ws = _select(list(dwd.tree), spec if spec is not None else _sym("All"), dwd.basis, fname)
    tree = dict(dwd.tree)
    for w in ws:
        tree[w] = _map_coef(f, tree[w], w, dwd.exact, fname)
    return _DWD(tree, dwd.wave, dwd.transform, dwd.meta).expr()


fn("WaveletMapIndexed", area=AREA, status="added",
   notes="WaveletMapIndexed[f, dwd] replaces every coefficient array coef with index wind by f[coef, wind] (also for "
         "ContinuousWaveletData with {oct, voc} indices); WaveletMapIndexed[f, dwd, wind] only the selected indices "
         "(index, list, pattern, All, Automatic). f must return an array of the same dimensions (ValueError "
         "otherwise). Coarser nodes are not recomputed: the inverse transform uses the basis coefficients. Check: "
         "exact (structural); reference (zeroing details reproduces the coarse reconstruction) in tests.")(_wmap)


# ═══════════════════════════════════════════════════════════════ WaveletThreshold ══
_SHORT_T = {"Universal": ("Hard", "Universal"), "UniversalLevel": ("Hard", "UniversalLevel"),
            "VisuShrink": ("Soft", "Universal"), "VisuShrinkLevel": ("Soft", "UniversalLevel"),
            "SURE": ("Hard", "SURE"), "SURELevel": ("Hard", "SURELevel"), "SUREShrink": ("Soft", "SURE"),
            "FDR": ("Soft", "FDR"), "GCV": ("Soft", "GCV"), "GCVLevel": ("Soft", "GCVLevel")}
_TFUNS = ("Hard", "Soft", "PiecewiseGarrote", "Hyperbola", "LargestCoefficients", "Firm", "SmoothGarrote")


def _mad_sigma(a):
    v = np.abs(np.asarray(a, dtype=complex)).ravel()
    return float(np.median(v)) / 0.6745


def _apply_tfun(kind, a, delta):
    x = np.asarray(a, dtype=complex) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    ax = np.abs(x)
    if kind == "Hard":
        return np.where(ax > delta, x, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "Soft":
            return np.where(ax > delta, x * (1 - delta / np.where(ax > 0, ax, 1)), 0)
        if kind == "PiecewiseGarrote":
            return np.where(ax > delta, x - delta ** 2 * x / np.where(ax > 0, ax, 1) ** 2, 0)
        if kind == "Hyperbola":
            return np.where(ax > delta, x / np.where(ax > 0, ax, 1) * np.sqrt(np.maximum(ax ** 2 - delta ** 2, 0)),
                            0)
    raise NotImplementedError(kind)


def _wthreshold(dwd_e: Expr, tspec: Expr = None, spec: Expr = None) -> Expr:
    fname = "WaveletThreshold"
    dwd = _parse_dwd(dwd_e, fname)
    if dwd.exact:
        raise NotImplementedError(f"{fname}: thresholding of exact (WorkingPrecision -> Infinity) coefficients is "
                                  "not supported")
    t = "Universal" if tspec is None else tspec
    if isinstance(t, str):
        if t in _SHORT_T:
            kind, how = _SHORT_T[t]
        elif t in _TFUNS:
            kind, how = t, "Universal"
        else:
            raise ValueError(f"{fname}: unknown threshold specification {t!r}")
    elif is_list(t) and t.args and isinstance(t.args[0], str):
        kind = t.args[0]
        if kind not in _TFUNS:
            raise ValueError(f"{fname}: unknown thresholding function {kind!r}")
        if len(t.args) > 2:
            raise NotImplementedError(f"{fname}: thresholding with extra parameters ({kind!r}) is not supported")
        how = t.args[1] if len(t.args) == 2 else "Universal"
    else:
        raise TypeError(f"{fname}: the threshold specification must be a name or {{tfun, δ}}, not {t!r}")
    if kind in ("Firm", "SmoothGarrote"):
        raise NotImplementedError(f"{fname}: the thresholding function {kind!r} is not supported")
    if spec is None:
        ws = [w for w in dwd.basis if any(j != 0 for j in w)]
    else:
        ws = _select(list(dwd.tree), spec, dwd.basis, fname)
    n_total = int(np.prod(dwd.dims))
    level1 = [w for w in dwd.tree if len(w) == 1 and w[0] != 0]
    tree = dict(dwd.tree)
    thr = {}
    for w in ws:
        a = tree[w]
        if kind == "LargestCoefficients":
            k = _int_arg(how, fname, "the number of coefficients", lo=0)
            flat = np.abs(a).ravel()
            if k >= flat.size:
                thr[w] = 0.0
                continue
            delta = float(np.sort(flat)[::-1][k]) if k > 0 else float(np.max(flat))
            new = np.where(np.abs(a) > delta, a, 0)
            if k > 0 and np.count_nonzero(new) < k:
                order = np.argsort(-np.abs(a).ravel(), kind="stable")[:k]
                new = np.zeros_like(a).ravel()
                new[order] = a.ravel()[order]
                new = new.reshape(a.shape)
            tree[w] = new
            thr[w] = delta
            continue
        if how == "Universal" or _is_sym(how, "Automatic"):
            sig = _mad_sigma(np.concatenate([dwd.tree[v].ravel() for v in level1]))
            delta = sig * math.sqrt(2 * math.log(n_total))
        elif how == "UniversalLevel":
            delta = _mad_sigma(a) * math.sqrt(2 * math.log(max(a.size, 2)))
        elif isinstance(how, str) or (is_list(how) and how.args and how.args[0] == "FDR"):
            raise NotImplementedError(f"{fname}: the threshold method {how!r} is not supported (a number, a "
                                      "function, \"Universal\" or \"UniversalLevel\")")
        else:
            try:
                delta = _num(how)
            except TypeError:
                v = apply(how, _to_list(a), _wind_expr(w))
                try:
                    delta = _num(v)
                except TypeError:
                    if isinstance(how, Symbol):
                        raise Unevaluated()
                    raise ValueError(f"{fname}: the threshold function did not return a number: {v!r}")
        if isinstance(delta, complex) or delta < 0:
            raise ValueError(f"{fname}: the threshold must be a non-negative real number, not {delta!r}")
        tree[w] = _apply_tfun(kind, a, delta)
        thr[w] = float(delta)
    meta = dict(dwd.meta)
    meta["ThresholdValues"] = L(*[_rule(_wind_expr(w), thr[w]) for w in ws])
    return _DWD(tree, dwd.wave, dwd.transform, meta).expr()


fn("WaveletThreshold", area=AREA, status="added",
   notes="WaveletThreshold[dwd, tspec, wind] thresholds coefficient arrays (default: the detail coefficients of the "
         "basis, every index other than {0, ..., 0}). tspec: \"Universal\" (default = {\"Hard\", \"Universal\"}), "
         "\"UniversalLevel\", \"VisuShrink\" ({\"Soft\", \"Universal\"}), \"VisuShrinkLevel\", {tfun, δ} or {tfun, "
         "method} with tfun \"Hard\" (x if |x| > δ), \"Soft\" (sign(x)(|x| - δ)+), \"PiecewiseGarrote\" (x - δ²/x), "
         "\"Hyperbola\" (sign(x) Sqrt[x² - δ²]) or {\"LargestCoefficients\", k}; δ a number or a function "
         "tfunc[coef, wind]. \"Universal\" is Donoho & Johnstone's δ = σ Sqrt[2 Log[N]], N = number of data points, "
         "σ = Median[|d|]/0.6745 over the level-1 detail coefficients; \"UniversalLevel\" uses each array's own σ "
         "and size. SURE, GCV, FDR and the Firm/SmoothGarrote functions raise NotImplementedError. Sets "
         "\"ThresholdValues\". Check: exact (elementwise rules); reference (hand-computed hard/soft thresholds and "
         "the universal threshold formula) in tests.")(_wthreshold)


# ═══════════════════════════════════════════════════════════════ continuous wavelets ══
_CWT_FAMILIES = ("MexicanHatWavelet", "MorletWavelet", "GaborWavelet", "PaulWavelet", "DGaussianWavelet",
                 "HaarWavelet", "ShannonWavelet")


def _psi_fun(wave, fname):
    name, p = _wavelet(wave, fname)
    if name not in _CWT_FAMILIES:
        raise NotImplementedError(f"{fname}: {name} has no closed-form wavelet function; supported: "
                                  + ", ".join(_CWT_FAMILIES))
    t = sp.Symbol("t", real=True)
    e = _psi_expr(name, p, t)
    f = sp.lambdify(t, e, modules="numpy")

    def psi(x):
        v = np.asarray(f(x), dtype=complex)
        if v.shape != np.shape(x):
            v = np.broadcast_to(v, np.shape(x)).astype(complex)
        if name == "ShannonWavelet":
            bad = np.isclose(x, 0.5)
            if np.any(bad):
                v = v.copy()
                v[bad] = complex(sp.limit(e, t, sp.Rational(1, 2)))
        return v
    return name, psi


def _nextpow2(n):
    return 1 << max(0, int(n - 1).bit_length())


class _CWD:
    def __init__(self, coefs, wave, meta):
        self.coefs, self.wave, self.meta = coefs, wave, meta

    @property
    def n(self):
        return int(self.meta["DataDimensions"].args[0])

    def scale(self, w):
        a = _num(self.meta["WaveletScale"])
        nv = int(self.meta["Voices"])
        return a * 2.0 ** (w[0] - 1) * 2.0 ** (w[1] / nv)

    def winds(self):
        return sorted(self.coefs)

    def expr(self):
        pairs = [(_wind_expr(w), _to_list(self.coefs[w])) for w in self.winds()]
        return Expr("ContinuousWaveletData", _assoc(pairs), self.wave, _assoc(list(self.meta.items())))


def _parse_cwd(e, fname):
    if not (isinstance(e, Expr) and e.head == "ContinuousWaveletData"):
        if isinstance(e, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a ContinuousWaveletData object, not {e!r}")
    if not (len(e.args) == 3 and isinstance(e.args[0], Expr) and e.args[0].head == "Association"
            and isinstance(e.args[2], Expr) and e.args[2].head == "Association"):
        raise ValueError(f"{fname}: the ContinuousWaveletData object is not in standard form")
    meta = {r.args[0]: r.args[1] for r in e.args[2].args}
    coefs = {}
    for r in e.args[0].args:
        w = _key(r.args[0])
        if not (isinstance(w, tuple) and len(w) == 2):
            raise ValueError(f"{fname}: invalid scale index {r.args[0]!r}")
        coefs[w] = _from_list(r.args[1])
    return _CWD(coefs, e.args[1], meta)


def _kernel(psi, n, s, dt):
    m = np.arange(-(n - 1), n)
    return np.conj(psi(dt * m / s)) / math.sqrt(s)


def _corr(x, G, n):
    """w[u] = Σ_k x[k] G[k - u + n - 1], u = 0..n-1 (exact finite sum via zero-padded FFT)."""
    N = _nextpow2(3 * n)
    c = np.fft.ifft(np.fft.fft(x, N) * np.fft.fft(G[::-1], N))
    return c[n - 1:2 * n - 1]


def _corr_adj(v, G, n):
    """Adjoint of _corr: y[k] = Σ_u v[u] conj(G[k - u + n - 1])."""
    N = _nextpow2(3 * n)
    c = np.fft.ifft(np.fft.fft(v, N) * np.fft.fft(np.conj(G), N))
    return c[n - 1:2 * n - 1]


def _cwt(data: ListOf[Real], *rest: Expr) -> Expr:
    fname = "ContinuousWaveletTransform"
    pos, opts = _opts(rest, {"Padding", "SampleRate", "WaveletScale", "WorkingPrecision"}, fname)
    if len(pos) > 2:
        raise TypeError(f"{fname}: called with {len(pos) + 1} positional arguments; 1 to 3 are expected")
    if isinstance(data, Expr) and data.head in ("Audio", "Sound"):
        raise NotImplementedError(f"{fname}: {data.head} input is not supported (give the sample list)")
    x = _data_array(data, fname, max_rank=1)
    n = x.shape[0]
    if n < 4:
        raise ValueError(f"{fname}: at least 4 data points are needed")
    wave = pos[0] if pos and not _is_sym(pos[0], "Automatic") else Expr("MexicanHatWavelet", 1)
    if isinstance(wave, Symbol):
        raise Unevaluated()
    wave = _wave_expr(wave)
    _name, psi = _psi_fun(wave, fname)
    noct, nvoc = max(1, int(math.floor(math.log2(n / 2)))), 4
    if len(pos) > 1 and not _is_sym(pos[1], "Automatic"):
        ov = pos[1]
        if is_list(ov) and len(ov.args) == 2:
            noct = _int_arg(ov.args[0], fname, "the number of octaves") if not _is_sym(ov.args[0], "Automatic") \
                else noct
            nvoc = _int_arg(ov.args[1], fname, "the number of voices")
        else:
            noct = _int_arg(ov, fname, "the number of octaves")
    pad = opts.get("Padding")
    if pad is not None and not _is_sym(pad, "None"):
        raise NotImplementedError(f"{fname}: Padding -> {pad!r} is not supported (only None)")
    wpv = opts.get("WorkingPrecision")
    if wpv is not None and _wp_opt(wpv, fname) != "m":
        raise NotImplementedError(f"{fname}: only machine precision is supported")
    sr = opts.get("SampleRate")
    rate = 1.0 if sr is None or _is_sym(sr, "Automatic") else _num(sr)
    if isinstance(rate, complex) or rate <= 0:
        raise ValueError(f"{fname}: SampleRate must be a positive number")
    dt = 1.0 / rate
    ws = opts.get("WaveletScale")
    alpha = dt if ws is None or _is_sym(ws, "Automatic") else _num(ws)
    if isinstance(alpha, complex) or alpha <= 0:
        raise ValueError(f"{fname}: WaveletScale must be a positive number")
    if noct * nvoc > 4096:
        raise ValueError(f"{fname}: too many scales ({noct} x {nvoc})")
    coefs = {}
    for o in range(1, noct + 1):
        for v in range(1, nvoc + 1):
            s = alpha * 2.0 ** (o - 1) * 2.0 ** (v / nvoc)
            w = _corr(x.astype(complex), _kernel(psi, n, s, dt), n)
            if not np.iscomplexobj(x) and np.all(np.abs(w.imag) <= 1e-13 * max(1.0, float(np.max(np.abs(w))))):
                w = w.real
            coefs[(o, v)] = w
    # self-check of the FFT evaluation against the direct sum at one scale and a few positions
    s0 = alpha * 2.0 ** (1 / nvoc)
    G = _kernel(psi, n, s0, dt)
    for u in (0, n // 2, n - 1):
        direct = sum(x[k] * G[k - u + n - 1] for k in range(n))
        if abs(direct - coefs[(1, 1)][u]) > 1e-8 * max(1.0, float(np.max(np.abs(x))) * n):
            raise ArithmeticError(f"{fname}: result failed verification")
    meta = {"Octaves": noct, "Voices": nvoc, "WaveletScale": float(alpha), "SampleRate": float(rate),
            "DataDimensions": L(n), "DataMean": R(complex(np.mean(x))) if np.iscomplexobj(x) else float(np.mean(x)),
            "Padding": _sym("None")}
    return _CWD(coefs, wave, meta).expr()


fn("ContinuousWaveletTransform", area=AREA, status="added",
   notes="ContinuousWaveletTransform[data, wave, {noct, nvoc}] for a numeric vector: the documented sum w(u, s) = "
         "1/Sqrt[s] Σ_k x_k Conjugate[ψ(Δ (k - u)/s)], u = 1..n (data taken as zero outside, Padding -> None), at "
         "the scales s_{oct,voc} = α 2^(oct - 1) 2^(voc/nvoc), oct = 1..noct, voc = 1..nvoc (the documented "
         "equal-tempered scale). Defaults: wave = MexicanHatWavelet[], noct = Floor[Log2[n/2]], nvoc = 4, Δ = "
         "1/SampleRate (SampleRate Automatic = 1), α = WaveletScale (Automatic = Δ, one sample: this default is this "
         "implementation's). Wavelets with closed-form ψ: MexicanHatWavelet, MorletWavelet, GaborWavelet, "
         "PaulWavelet, DGaussianWavelet, HaarWavelet, ShannonWavelet; others raise NotImplementedError. Other "
         "Padding settings, bignum precision and Audio input raise NotImplementedError. Result: "
         "ContinuousWaveletData. Check: self-check (the FFT evaluation is compared with the direct sum at sample "
         "positions); reference (direct-sum values, Torrence & Compo 1998 reconstruction constants) in tests.")(_cwt)


def _cwd_select(cwd, spec, fname):
    ws = cwd.winds()
    if _is_sym(spec, "All", "Automatic"):
        return ws
    w = _key(spec)
    if isinstance(w, tuple) and len(w) == 2:
        if w not in cwd.coefs:
            raise ValueError(f"{fname}: there are no coefficients with scale index {list(w)}")
        return [w]
    if is_list(spec) and spec.args and all(is_list(t) for t in spec.args):
        out = []
        for t in spec.args:
            out.extend(u for u in _cwd_select(cwd, t, fname) if u not in out)
        return out
    out = [w for w in ws if ev(Expr("MatchQ", _wind_expr(w), spec)) is True]
    if not out:
        raise ValueError(f"{fname}: no scale index {{octave, voice}} matches {spec!r}")
    return out


def _icwt_core(cwd, sel, fname):
    n = cwd.n
    name, psi = _psi_fun(cwd.wave, fname)
    dt = 1.0 / _num(cwd.meta["SampleRate"])
    mu = _num(cwd.meta["DataMean"])
    full = set(sel) == set(cwd.winds())
    winds = cwd.winds()
    N = _nextpow2(3 * n)
    Gs = np.array([_kernel(psi, n, cwd.scale(w), dt) for w in winds])
    Fr = np.fft.fft(Gs[:, ::-1], N, axis=1)
    Fc = np.fft.fft(np.conj(Gs), N, axis=1)
    cplx = isinstance(mu, complex)

    def fwd(x):
        return np.fft.ifft(np.fft.fft(x, N)[None, :] * Fr, axis=1)[:, n - 1:2 * n - 1]

    def adj(V):
        return np.fft.ifft((np.fft.fft(V, N, axis=1) * Fc).sum(axis=0))[n - 1:2 * n - 1]

    def normal(x):
        y = adj(fwd(x))
        return y if cplx else y.real
    idx = {w: i for i, w in enumerate(winds)}
    V = np.zeros((len(winds), n), dtype=complex)
    ones_resp = fwd(np.ones(n, dtype=complex))
    for w in sel:
        V[idx[w]] = cwd.coefs[w].astype(complex) - mu * ones_resp[idx[w]]
    b = adj(V)
    b = b if cplx else b.real
    # preconditioned CG; preconditioner = inverse of the circulant approximation Σ_j |Ĝ_j|² of the normal operator
    sym = (np.abs(Fr) ** 2).sum(axis=0)
    sym = sym + 1e-9 * float(np.max(sym))

    def prec(v):
        y = np.fft.ifft(np.fft.fft(v, N) / sym)[:n]
        return y if cplx else y.real
    x = np.zeros(n, dtype=b.dtype)
    r = b.copy()
    z = prec(r)
    p = z.copy()
    rz = np.vdot(r, z).real
    bn = math.sqrt(np.vdot(b, b).real) or 1.0
    for _it in range(max(200, 4 * n)):
        if math.sqrt(np.vdot(r, r).real) <= 1e-11 * bn:
            break
        Ap = normal(p)
        alpha = rz / np.vdot(p, Ap).real
        x = x + alpha * p
        r = r - alpha * Ap
        z = prec(r)
        rz2 = np.vdot(r, z).real
        p = z + (rz2 / rz) * p
        rz = rz2
    res = b - normal(x)
    if math.sqrt(np.vdot(res, res).real) > 1e-6 * bn:
        raise ArithmeticError(f"{fname}: the least-squares inverse did not converge (result failed verification)")
    return x + (mu if full else 0)


def _icwt(cwd_e: Expr, wave: Expr = None, spec: Expr = None) -> ListOf[Number]:
    fname = "InverseContinuousWaveletTransform"
    cwd = _parse_cwd(cwd_e, fname)
    if wave is not None and not _is_sym(wave, "Automatic"):
        if isinstance(wave, Symbol):
            raise Unevaluated()
        if _wave_expr(wave) != _wave_expr(cwd.wave):
            raise NotImplementedError(f"{fname}: reconstruction with a wavelet other than the analysing wavelet "
                                      f"{cwd.wave!r} is not supported")
    sel = cwd.winds() if spec is None or _is_sym(spec, "Automatic", "All") else _cwd_select(cwd, spec, fname)
    x = _icwt_core(cwd, sel, fname)
    return _to_list(x)


fn("InverseContinuousWaveletTransform", area=AREA, status="added",
   notes="InverseContinuousWaveletTransform[cwd] / [cwd, Automatic, octvoc]: the least-squares inverse of the "
         "discrete ContinuousWaveletTransform operator A (all scales of cwd): x = DataMean + (Σ_j A_j^* A_j)^-1 "
         "Σ_j A_j^* (w_j - A_j DataMean), solved by conjugate gradients to a relative normal-equation residual of "
         "1e-11 with an FFT (circulant) preconditioner (ArithmeticError if 1e-6 is not reached). With octvoc only the selected coefficients enter the "
         "right-hand side (with their DataMean response removed; the others are zero) and the mean is not added, so "
         "the band reconstructions plus DataMean sum to the full one. Deviation: Wolfram's approximate inverse formula is undocumented; this is the exact inverse "
         "for data whose zero-mean part lies in the span the scale grid resolves, and the minimum-residual "
         "approximation otherwise (very low frequencies are recovered poorly). A wavelet other than cwd's raises "
         "NotImplementedError. Check: self-check (normal-equation residual); reference (round trip of band-limited "
         "data, band additivity) in tests.")(_icwt)


def _cwd_ctor(*args: Expr) -> Expr:
    fname = "ContinuousWaveletData"
    if len(args) == 3 and isinstance(args[0], Expr) and args[0].head == "Association":
        _parse_cwd(Expr(fname, *args), fname)
        raise Unevaluated()
    if args and is_list(args[0]):
        if len(args) < 2:
            raise TypeError(f"{fname}: expected ContinuousWaveletData[{{{{oct, voc}} -> coef, ...}}, wave, opts]")
        _pos, opts = _opts(args[2:], {"WaveletScale", "SampleRate", "Voices", "DataMean"}, fname)
        coefs = {}
        for r in args[0].args:
            if not is_rule(r):
                raise TypeError(f"{fname}: expected rules {{oct, voc}} -> coef")
            w = _key(r.args[0])
            if not (isinstance(w, tuple) and len(w) == 2 and w[0] >= 1 and w[1] >= 1):
                raise ValueError(f"{fname}: invalid scale index {r.args[0]!r}")
            coefs[w] = _from_list(r.args[1])
        if not coefs:
            raise ValueError(f"{fname}: no coefficients given")
        ns = {c.shape for c in coefs.values()}
        if len(ns) != 1 or len(next(iter(ns))) != 1:
            raise ValueError(f"{fname}: the coefficient arrays must be vectors of equal length")
        wave = _wave_expr(args[1])
        nvoc = int(opts["Voices"]) if "Voices" in opts else max(w[1] for w in coefs)
        meta = {"Octaves": max(w[0] for w in coefs), "Voices": nvoc,
                "WaveletScale": float(_num(opts.get("WaveletScale", 1))),
                "SampleRate": float(_num(opts.get("SampleRate", 1))), "DataDimensions": L(next(iter(ns))[0]),
                "DataMean": _num(opts.get("DataMean", 0)), "Padding": _sym("None")}
        return _CWD(coefs, wave, meta).expr()
    if args and isinstance(args[0], Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: expected a list of rules {{oct, voc}} -> coef")


fn("ContinuousWaveletData", area=AREA, status="added",
   notes="ContinuousWaveletData[{{oct, voc} -> coef, ...}, wave, opts] (options WaveletScale, SampleRate, Voices, "
         "DataMean) builds the standard form ContinuousWaveletData[<|{oct, voc} -> coef, ...|>, wave, "
         "<|properties|>] returned by ContinuousWaveletTransform. Access: cwd[{oct, voc}], cwd[{...}], cwd[patt], "
         "cwd[All] (rules), cwd[spec, form] with \"Rules\", \"Values\", \"Inverse\", \"ListPlot\"; properties "
         "\"Properties\", \"Octaves\", \"Voices\", \"Scales\" ({oct, voc} -> s), \"Wavelet\", \"WaveletScale\", "
         "\"WaveletIndex\", \"DataDimensions\", \"DataChannels\", \"DataMean\", \"DataWrapper\", \"SampleRate\"; "
         "Normal gives the rules. \"LogScalogramFunction\"/\"LinearScalogramFunction\" (undocumented "
         "interpolations) and Sound forms raise NotImplementedError. Check: exact (structural); reference "
         "(documented scale formula) in tests.")(_cwd_ctor)

_CWD_PROPS = ("DataChannels", "DataDimensions", "DataMean", "DataWrapper", "Octaves", "SampleRate", "Scales",
              "Voices", "Wavelet", "WaveletIndex", "WaveletScale")


def _cwd_hook(head, args, ev_):
    fname = "ContinuousWaveletData"
    try:
        cwd = _parse_cwd(head, fname)
    except (ValueError, TypeError):
        return K.UNHANDLED
    if not 1 <= len(args) <= 2:
        raise TypeError(f"{fname}: cwd[spec] or cwd[spec, form] expected")
    spec = args[0]
    if isinstance(spec, str):
        if spec == "Properties":
            return L(*_CWD_PROPS)
        if spec in ("Octaves", "Voices", "WaveletScale", "SampleRate", "DataDimensions", "DataMean"):
            return cwd.meta[spec]
        if spec == "Scales":
            return L(*[_rule(_wind_expr(w), cwd.scale(w)) for w in cwd.winds()])
        if spec == "Wavelet":
            return cwd.wave
        if spec == "WaveletIndex":
            return L(*[_wind_expr(w) for w in cwd.winds()])
        if spec == "DataChannels":
            return 1
        if spec == "DataWrapper":
            return _sym("Identity")
        if spec in ("LogScalogramFunction", "LinearScalogramFunction"):
            raise NotImplementedError(f"{fname}: the property {spec!r} is not supported")
        raise ValueError(f"{fname}: unknown property {spec!r}")
    ws = _cwd_select(cwd, spec, fname)
    form = args[1] if len(args) == 2 else "Rules"
    forms = list(form.args) if is_list(form) else [form]
    kind = "Rules"
    for f in forms:
        if f in ("Values", "Inverse", "ListPlot"):
            kind = f if kind == "Rules" else kind
        elif f != "Rules":
            if f in ("Sound", "SampledSoundList", "Audio"):
                raise NotImplementedError(f"{fname}: the form {f!r} is not supported")
            raise ValueError(f"{fname}: unknown output form {f!r}")
    if kind == "Inverse":
        vals = [(w, _to_list(_icwt_core(cwd, [w], fname))) for w in ws]
    else:
        vals = [(w, _to_list(cwd.coefs[w])) for w in ws]
    if kind == "Values":
        return L(*[v for _, v in vals])
    if kind == "ListPlot":
        return L(*[_rule(_wind_expr(w), ev(Expr("ListLinePlot", v))) for w, v in vals])
    return L(*[_rule(_wind_expr(w), v) for w, v in vals])


KL.COMPOUND_HEAD_HOOKS["ContinuousWaveletData"] = _cwd_hook
