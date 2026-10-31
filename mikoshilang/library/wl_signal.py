"""Wolfram batch W23: signal processing.

Discrete transforms (Fourier, InverseFourier, FourierDCT, FourierDST and their matrices, DiscreteHadamardTransform,
DiscreteHilbertTransform, DiscreteChirpZTransform, ListZTransform, ListFourierSequenceTransform), spectra
(PeriodogramArray, CepstrumArray, CepstrogramArray, ShortTimeFourier), list filters and convolution (ListConvolve,
ListCorrelate, MovingAverage, MovingMedian, MovingMap, ExponentialMovingAverage, FindPeaks, WienerFilter, FIR
filters), symbolic transforms and convolutions (FourierSequenceTransform, BilateralLaplaceTransform,
BilateralZTransform, HankelTransform, DiscreteConvolve, ...) and kernel matrices (GaussianMatrix, DiskMatrix,
SavitzkyGolayMatrix, ...).  Wavelets live in ``wl_signal_wavelets``.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).  NumPy is a core dependency; SciPy is never required.
"""
from __future__ import annotations

import cmath
import math

import mpmath
import numpy as np
import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, record, Unevaluated, R, S, is_list
from .wl_common import L, apply, is_rule


def ev(x):
    from ..evaluate import evaluate
    return evaluate(x)

AREA = "signal processing (W23)"
BIG_N = 4096          # largest axis length for bignum (mpmath, O(n^2)) transforms


# ═══════════════════════════════════════════════════════════════ helpers ══
def _opt_name(r):
    if is_rule(r):
        k = r.args[0]
        if isinstance(k, Symbol):
            return k.name
        if isinstance(k, str):
            return k
    return None


def _opts(args, allowed, fname):
    """Split trailing Name -> value options; unknown options raise ValueError."""
    args = list(args)
    opts = {}
    while args and _opt_name(args[-1]) is not None:
        r = args.pop()
        n = _opt_name(r)
        if n not in allowed:
            raise ValueError(f"{fname}: unknown option {n}")
        opts[n] = r.args[1]
    for a in args:
        if _opt_name(a) in allowed:
            raise ValueError(f"{fname}: options must come after the positional arguments")
    return args, opts


def _is_symbolic(v):
    return isinstance(v, sp.Basic) and bool(v.free_symbols)


def _elem(v, fname):
    """A numeric list element as a SymPy number; symbolic -> Unevaluated; other kinds -> TypeError."""
    if isinstance(v, bool) or isinstance(v, str):
        raise TypeError(f"{fname}: list elements must be numbers, not {v!r}")
    if isinstance(v, (int, float)):
        return sp.sympify(v)
    if isinstance(v, complex):
        return sp.Float(v.real) + sp.Float(v.imag) * sp.I
    if isinstance(v, Expr) and v.head == "List":
        raise ValueError(f"{fname}: the array of data must be rectangular")
    try:
        s = S(v)
    except Exception:
        raise TypeError(f"{fname}: list element {v!r} is not a number")
    if not isinstance(s, sp.Basic):
        raise TypeError(f"{fname}: list element {v!r} is not a number")
    if s.free_symbols:
        raise Unevaluated()
    if not s.is_number:
        raise TypeError(f"{fname}: list element {v!r} is not a number")
    return s


def _array(x, fname, min_rank=1, max_rank=None, allow_empty=False):
    """Rectangular numeric array -> (flat list of SymPy numbers, shape).  Symbolic -> Unevaluated."""
    if isinstance(x, (bool, str, int, float, complex)) or (isinstance(x, sp.Basic) and x.is_number):
        raise TypeError(f"{fname}: expected a list of numbers, not {x!r}")
    if not is_list(x):
        if isinstance(x, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a list of numbers, not {x!r}")
    shape = []
    y = x
    while is_list(y):
        shape.append(len(y.args))
        if not y.args:
            break
        y = y.args[0]
    if not allow_empty and (0 in shape):
        raise ValueError(f"{fname}: the data must be a nonempty list")
    if len(shape) < min_rank:
        raise ValueError(f"{fname}: expected an array of depth at least {min_rank}")
    if max_rank is not None and len(shape) > max_rank:
        raise NotImplementedError(f"{fname}: arrays of depth {len(shape)} are not supported (at most {max_rank})")
    flat = []

    def walk(z, d):
        if d == len(shape):
            if is_list(z):
                raise ValueError(f"{fname}: the array of data must be rectangular")
            flat.append(_elem(z, fname))
            return
        if not is_list(z) or len(z.args) != shape[d]:
            raise ValueError(f"{fname}: the array of data must be rectangular")
        for a in z.args:
            walk(a, d + 1)
    walk(x, 0)
    return flat, tuple(shape)


def _prec(flat):
    """'m' (machine), None (exact) or the smallest bignum precision in bits."""
    best = None
    for v in flat:
        for a in sp.preorder_traversal(v):
            if isinstance(a, sp.Float):
                if a._prec <= 53:
                    return "m"
                best = a._prec if best is None else min(best, a._prec)
    return best


def _work_prec(flat, wp=None, fname=""):
    """Working precision: 'm' or bits.  Exact input -> machine (Wolfram applies N first)."""
    if wp is not None:
        return wp
    p = _prec(flat)
    return "m" if p in (None, "m") else p


def _wp_opt(v, fname):
    """WorkingPrecision option value -> 'm', bits, or 'exact' (Infinity)."""
    if isinstance(v, Symbol) and v.name == "MachinePrecision":
        return "m"
    if isinstance(v, Symbol) and v.name == "Infinity" or (isinstance(v, Expr) and v.head == "DirectedInfinity"):
        return "exact"
    if isinstance(v, Symbol) and v.name == "Automatic":
        return None
    try:
        d = float(S(v))
    except Exception:
        raise ValueError(f"{fname}: WorkingPrecision must be a positive number, MachinePrecision or Infinity")
    if not d > 0:
        raise ValueError(f"{fname}: WorkingPrecision must be positive")
    if d <= 15.954589770191003:
        return "m"
    return int(mpmath.libmp.dps_to_prec(d))


def _to_np(flat, shape):
    return np.array([complex(sp.N(v, 17)) for v in flat], dtype=complex).reshape(shape)


def _to_mp(flat, shape, prec):
    with mpmath.workprec(prec + 10):
        vals = [mpmath.mpc(sp.N(sp.re(v), mpmath.libmp.prec_to_dps(prec) + 5)._to_mpmath(prec + 10),
                           sp.N(sp.im(v), mpmath.libmp.prec_to_dps(prec) + 5)._to_mpmath(prec + 10)) for v in flat]
    arr = np.empty(len(vals), dtype=object)
    arr[:] = vals
    return arr.reshape(shape)


def _num_out(v, prec, real=False):
    """A machine complex/float or mpmath value -> MikoshiLang number (imag dropped when real=True)."""
    if prec == "m":
        c = complex(v)
        if real:
            return float(c.real)
        return R(c)
    if isinstance(v, (mpmath.mpf, mpmath.mpc)):
        re_, im_ = mpmath.mpf(mpmath.re(v)), mpmath.mpf(mpmath.im(v))
    else:
        re_, im_ = mpmath.mpf(v), mpmath.mpf(0)
    re_s = sp.Float(re_, precision=prec)
    if real or im_ == 0 and real:
        return re_s
    im_s = sp.Float(im_, precision=prec)
    from ..math_ops import from_sympy
    return from_sympy(re_s + im_s * sp.I) if im_ != 0 else re_s


def _nest(arr, conv):
    """numpy array (any dtype) -> nested MikoshiLang List."""
    if not isinstance(arr, np.ndarray):
        return conv(arr)
    if arr.ndim == 0:
        return conv(arr.item() if arr.dtype != object else arr[()])
    return L(*(_nest(a, conv) for a in arr))


def _out(arr, prec, real=False):
    return _nest(arr, lambda v: _num_out(v, prec, real))


def _apply_matrix_axes(arr, mats, prec):
    """Apply 1-D linear transforms (one matrix per axis, out x in) along every axis of arr."""
    for ax, M in enumerate(mats):
        if prec == "m":
            arr = np.moveaxis(np.tensordot(M, np.moveaxis(arr, ax, 0), axes=([1], [0])), 0, ax)
        else:
            a = np.moveaxis(arr, ax, 0)
            n_out, n_in = M.shape
            out = np.empty((n_out,) + a.shape[1:], dtype=object)
            with mpmath.workprec(prec + 20):
                for idx in np.ndindex(*a.shape[1:]):
                    col = [a[(k,) + idx] for k in range(n_in)]
                    for s in range(n_out):
                        out[(s,) + idx] = mpmath.fsum(M[s, k] * col[k] for k in range(n_in))
            arr = np.moveaxis(out, 0, ax)
    return arr


def _matrix(n_out, n_in, entry, prec):
    """Transform matrix from an entry(s, r) function returning a SymPy/mpmath-friendly value."""
    if prec == "m":
        return np.array([[complex(entry(s, r, "m")) for r in range(n_in)] for s in range(n_out)], dtype=complex)
    M = np.empty((n_out, n_in), dtype=object)
    with mpmath.workprec(prec + 20):
        for s in range(n_out):
            for r in range(n_in):
                M[s, r] = entry(s, r, "mp")
    return M


def _check_bignum_size(shape, prec, fname):
    if prec != "m" and max(shape) > BIG_N:
        raise NotImplementedError(f"{fname}: bignum transforms are limited to {BIG_N} points per dimension")


def _is_real_data(flat):
    return all(v.is_real is not False and sp.im(v) == 0 for v in flat)


# ═══════════════════════════════════════════════════════════════ Fourier ══
def _fourier_params(opts, fname, default=(0, 1)):
    v = opts.get("FourierParameters")
    if v is None:
        return sp.Integer(default[0]), sp.Integer(default[1])
    if not (is_list(v) and len(v.args) == 2):
        raise ValueError(f"{fname}: FourierParameters must be a list {{a, b}}")
    a, b = (S(t) for t in v.args)
    if not (isinstance(a, sp.Basic) and a.is_number and a.is_real and isinstance(b, sp.Basic) and b.is_number
            and b.is_real):
        raise ValueError(f"{fname}: FourierParameters {{a, b}} must be real numbers")
    if b == 0:
        raise ValueError(f"{fname}: the parameter b in FourierParameters must be nonzero")
    return a, b


def _dft_entry(n, a, b, sign):
    """Fourier (sign=+1) / InverseFourier (sign=-1) matrix entry v_s = n^((sign*a - 1)/2)... see notes."""
    scale_exp = (1 - a) / 2 if sign > 0 else (1 + a) / 2

    def entry(s, r, mode):
        if mode == "m":
            return n ** (-float(scale_exp)) * cmath.exp(sign * 2j * math.pi * float(b) * r * s / n)
        return mpmath.power(n, -mpmath.mpf(sp.N(scale_exp, 40)._to_mpmath(80))) * \
            mpmath.expjpi(sign * 2 * mpmath.mpf(sp.N(b, 40)._to_mpmath(80)) * r * s / n)
    return entry


def _fourier_impl(fname, sign):
    def impl(x: ListOf[Number], *rest: Option) -> ListOf[Number]:
        args, opts = _opts(rest, {"FourierParameters"}, fname)
        if args:
            raise NotImplementedError(f"{fname}: the position form {fname}[list, {{p1, p2, ...}}] is not supported")
        a, b = _fourier_params(opts, fname)
        flat, shape = _array(x, fname)
        for n in shape:
            if b.is_Integer and sp.gcd(int(b), n) != 1:
                raise ValueError(f"{fname}: FourierParameters b = {b} must be relatively prime to the length {n}")
        prec = _work_prec(flat)
        _check_bignum_size(shape, prec, fname)
        if prec == "m" and a == 0 and abs(b) == 1:
            arr = _to_np(flat, shape)
            axes = tuple(range(len(shape)))
            # numpy: fft = sum x e^{-2 pi i ...}; ifft = (1/n) sum x e^{+2 pi i ...}
            if sign * int(b) > 0:
                out = np.fft.ifftn(arr, axes=axes) * np.sqrt(np.prod(shape))
            else:
                out = np.fft.fftn(arr, axes=axes) / np.sqrt(np.prod(shape))
            return _out(out, "m")
        arr = _to_np(flat, shape) if prec == "m" else _to_mp(flat, shape, prec)
        mats = [_matrix(n, n, _dft_entry(n, a, b, sign), prec) for n in shape]
        return _out(_apply_matrix_axes(arr, mats, prec), prec)
    return impl


fn("Fourier", area=AREA, status="added",
   notes="Fourier[list] = n^((a - 1)/2)... with FourierParameters -> {a, b} (default {0, 1}): "
         "v_s = n^(-(1 - a)/2) Σ_r u_r e^(2πi b (r-1)(s-1)/n), for rectangular arrays of any depth (applied along "
         "every dimension); exact input is numericized first (machine), bignum input keeps its precision; b must be "
         "coprime to each length (ValueError); symbolic entries stay unevaluated; the position form "
         "Fourier[list, {p...}] raises NotImplementedError. Check: reference (Wolfram documentation examples) and "
         "the round trip InverseFourier[Fourier[u]] == u in tests.")(_fourier_impl("Fourier", +1))
fn("InverseFourier", area=AREA, status="added",
   notes="InverseFourier[list] = n^(-(1 + a)/2) Σ_s v_s e^(-2πi b (r-1)(s-1)/n) with FourierParameters -> {a, b} "
         "(default {0, 1}), any array depth; same input handling as Fourier. Check: reference (NumPy ifft scaling) and "
         "round trip with Fourier in tests.")(_fourier_impl("InverseFourier", -1))


def _inert_option(name, notes):
    from .wl_evaluation import INERT
    INERT[name] = notes
    record(name, AREA, "added", notes.rstrip(". ") + ". Check: exact (inert symbol; tests check it stays "
                                                     "unevaluated and is accepted as documented).")


_inert_option("FourierParameters", "option for Fourier, InverseFourier, ListFourierSequenceTransform, "
                                   "FourierSequenceTransform, PeriodogramArray, CepstrumArray, ... giving the "
                                   "parameters {a, b} of the transform; inert symbol")


# ═══════════════════════════════════════════════════════════ DCT / DST ══
_TYPES = {1: 1, 2: 2, 3: 3, 4: 4, "I": 1, "II": 2, "III": 3, "IV": 4}


def _tf_type(m, fname):
    if isinstance(m, bool):
        raise TypeError(f"{fname}: the transform type must be 1, 2, 3 or 4")
    key = m if isinstance(m, str) else (int(m) if K.is_int(m) else None)
    if key is None:
        if isinstance(m, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the transform type must be 1, 2, 3, 4 or \"I\" ... \"IV\", not {m!r}")
    if key not in _TYPES:
        raise ValueError(f"{fname}: the transform type must be 1, 2, 3, 4 or \"I\" ... \"IV\", not {m!r}")
    return _TYPES[key]


class _Num:
    def __init__(self, pi, cos, sin, sqrt, q):
        self.pi, self.cos, self.sin, self.sqrt, self.q = pi, cos, sin, sqrt, q


_NUMS = {"m": _Num(math.pi, math.cos, math.sin, math.sqrt, lambda a, b: a / b),
         "mp": _Num(mpmath.pi, mpmath.cos, mpmath.sin, mpmath.sqrt, lambda a, b: mpmath.mpf(a) / b),
         "exact": _Num(sp.pi, sp.cos, sp.sin, sp.sqrt, lambda a, b: sp.Rational(a, b))}


def _dct_entry(n, t, N):
    """Entry F[s][r] (output index s, input index r, both 0-based) of Wolfram's normalized DCT of type t."""
    q, pi, cos, sqrt = N.q, N.pi, N.cos, N.sqrt
    if t == 1:
        return lambda s, r: sqrt(q(2, n - 1)) * (1 if 0 < r < n - 1 else q(1, 2)) * cos(pi * q(r * s, n - 1))
    if t == 2:
        return lambda s, r: cos(pi * q((2 * r + 1) * s, 2 * n)) / sqrt(n)
    if t == 3:
        return lambda s, r: (1 if r == 0 else 2) * cos(pi * q(r * (2 * s + 1), 2 * n)) / sqrt(n)
    return lambda s, r: sqrt(q(2, n)) * cos(pi * q((2 * r + 1) * (2 * s + 1), 4 * n))


def _dst_entry(n, t, N):
    q, pi, sin, sqrt = N.q, N.pi, N.sin, N.sqrt
    if t == 1:
        return lambda s, r: sqrt(q(2, n + 1)) * sin(pi * q((r + 1) * (s + 1), n + 1))
    if t == 2:
        return lambda s, r: sin(pi * q((2 * r + 1) * (s + 1), 2 * n)) / sqrt(n)
    if t == 3:
        return lambda s, r: ((-1) ** s if r == n - 1 else 2 * sin(pi * q((r + 1) * (2 * s + 1), 2 * n))) / sqrt(n)
    return lambda s, r: sqrt(q(2, n)) * sin(pi * q((2 * r + 1) * (2 * s + 1), 4 * n))


def _trig_entry(kind, n, t, mode):
    return (_dct_entry if kind == "DCT" else _dst_entry)(n, t, _NUMS[mode])


def _trig_matrix(kind, n, t, prec):
    if prec == "m":
        e = _trig_entry(kind, n, t, "m")
        return np.array([[float(e(s, r)) for r in range(n)] for s in range(n)], dtype=complex)
    M = np.empty((n, n), dtype=object)
    with mpmath.workprec(prec + 20):
        e = _trig_entry(kind, n, t, "mp")
        for s in range(n):
            for r in range(n):
                M[s, r] = mpmath.mpf(e(s, r))
    return M


def _trig_tf(kind, fname):
    def impl(x: ListOf[Number], m: PosInteger | String = 2) -> ListOf[Number]:
        t = _tf_type(m, fname)
        flat, shape = _array(x, fname)
        if t == 1 and min(shape) < 2 and kind == "DCT":
            raise ValueError(f"{fname}: DCT-I needs at least 2 points in each dimension")
        prec = _work_prec(flat)
        _check_bignum_size(shape, prec, fname)
        real = _is_real_data(flat)
        arr = _to_np(flat, shape) if prec == "m" else _to_mp(flat, shape, prec)
        mats = [_trig_matrix(kind, n, t, prec) for n in shape]
        return _out(_apply_matrix_axes(arr, mats, prec), prec, real=real)
    return impl


fn("FourierDCT", area=AREA, status="added",
   notes="FourierDCT[list, m] for types m = 1..4 (or \"I\"..\"IV\"), default 2, with Wolfram's normalization "
         "(DCT-II: v_s = 1/√n Σ u_r cos(π(r-1/2)(s-1)/n); DCT-III its inverse; DCT-I and DCT-IV self-inverse), "
         "applied along every dimension of a rectangular array; exact input is numericized (machine), bignum keeps "
         "its precision, complex data allowed; symbolic entries stay unevaluated. Check: reference (Wolfram "
         "documentation values) and the inverse pairs 1↔1, 2↔3, 4↔4 in tests.")(_trig_tf("DCT", "FourierDCT"))
fn("FourierDST", area=AREA, status="added",
   notes="FourierDST[list, m] for types m = 1..4, default 2, with Wolfram's normalization (DST-I √(2/(n+1)) Σ u_r "
         "sin(π r s/(n+1)), DST-II 1/√n Σ u_r sin(π(r-1/2)s/n), DST-III its inverse, DST-IV √(2/n) Σ ...), any "
         "array depth; same input handling as FourierDCT. Check: reference (Wolfram documentation values) and the "
         "inverse pairs in tests.")(_trig_tf("DST", "FourierDST"))


def _trig_matrix_fn(kind, fname):
    def impl(n: PosInteger, *rest: Expr) -> MatrixOf[Real]:
        args, opts = _opts(rest, {"WorkingPrecision", "TargetStructure"}, fname)
        if len(args) > 1:
            raise ValueError(f"{fname}: called with too many arguments")
        if isinstance(n, bool) or not K.is_int(n):
            if isinstance(n, Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: the size must be a positive integer, not {n!r}")
        n = int(n)
        if n < 1:
            raise ValueError(f"{fname}: the size must be a positive integer")
        t = _tf_type(args[0], fname) if args else 2
        if kind == "DCT" and t == 1 and n < 2:
            raise ValueError(f"{fname}: DCT-I needs n >= 2")
        ts = opts.get("TargetStructure")
        if ts is not None and not (isinstance(ts, Symbol) and ts.name == "Automatic") and ts != "Dense":
            raise NotImplementedError(f"{fname}: TargetStructure -> {ts!r} is not supported (only \"Dense\")")
        wp = _wp_opt(opts["WorkingPrecision"], fname) if "WorkingPrecision" in opts else "exact"
        if wp is None:
            wp = "exact"
        if wp == "exact":
            e = _trig_entry(kind, n, t, "exact")
            # F[r][s]: data index r (row), frequency s (column), so that data . F == FourierDCT[data]
            return L(*(L(*(R(sp.sympify(e(s, r))) for s in range(n))) for r in range(n)))
        M = _trig_matrix(kind, n, t, wp)
        return L(*(L(*(_num_out(M[s, r], wp, real=True) for s in range(n))) for r in range(n)))
    return impl


fn("FourierDCTMatrix", area=AREA, status="added",
   notes="FourierDCTMatrix[n, m] (types 1..4, default 2): exact n×n matrix F with data . F == FourierDCT[data, m] "
         "(rows: data index, columns: frequency); WorkingPrecision -> MachinePrecision or p gives numbers; "
         "TargetStructure -> \"Dense\"/Automatic only (others raise NotImplementedError). Check: reference "
         "(Wolfram documentation matrices) and F2.F3 == IdentityMatrix in tests.")(_trig_matrix_fn("DCT", "FourierDCTMatrix"))
fn("FourierDSTMatrix", area=AREA, status="added",
   notes="FourierDSTMatrix[n, m] (types 1..4, default 2): exact n×n matrix with data . F == FourierDST[data, m]; "
         "WorkingPrecision and TargetStructure as FourierDCTMatrix. Check: reference (Wolfram documentation "
         "matrices) and F2.F3 == IdentityMatrix in tests.")(_trig_matrix_fn("DST", "FourierDSTMatrix"))


# ═══════════════════════════════════════════════════════ Hadamard/Hilbert ══
def _bits(k, m):
    return [(k >> j) & 1 for j in range(m)]


def _hadamard_exponent(r, s, m, method):
    b_r = _bits(r, m)
    b_s = _bits(s, m)
    if method == "BitComplement":
        p = b_s
    elif method == "Sequency":
        p = [b_s[m - 1]] + [b_s[m - j] + b_s[m - j - 1] for j in range(1, m)]
    else:  # GrayCode (Paley / dyadic): bit-reversed index
        p = [b_s[m - 1 - j] for j in range(m)]
    return sum(b_r[j] * p[j] for j in range(m))


def _hadamard(x: ListOf[Number], *opts: Option) -> ListOf[Number]:
    fname = "DiscreteHadamardTransform"
    args, o = _opts(opts, {"Method"}, fname)
    if args:
        raise ValueError(f"{fname}: called with too many arguments")
    method = o.get("Method", "Sequency")
    if isinstance(method, Symbol) and method.name == "Automatic":
        method = "Sequency"
    if method not in ("Sequency", "BitComplement", "GrayCode"):
        raise ValueError(f"{fname}: Method must be \"Sequency\", \"BitComplement\" or \"GrayCode\", not {method!r}")
    flat, shape = _array(x, fname)
    padded = tuple(1 << max(0, (n - 1).bit_length()) for n in shape)
    exact = _prec(flat) is None
    prec = "exact" if exact else _work_prec(flat)
    # zero-pad every dimension to a power of two
    src = np.empty(shape, dtype=object)
    src.reshape(-1)[:] = flat
    arr = np.zeros(padded, dtype=object)
    arr[...] = sp.Integer(0)
    arr[tuple(slice(0, n) for n in shape)] = src
    for ax, n in enumerate(padded):
        m = n.bit_length() - 1
        H = [[(-1) ** _hadamard_exponent(r, s, m, method) for r in range(n)] for s in range(n)]
        a = np.moveaxis(arr, ax, 0)
        out = np.empty_like(a)
        for idx in np.ndindex(*a.shape[1:]):
            col = [a[(k,) + idx] for k in range(n)]
            for s_ in range(n):
                out[(s_,) + idx] = sp.Add(*(H[s_][k] * col[k] for k in range(n) if col[k] != 0)) / sp.sqrt(n)
        arr = np.moveaxis(out, 0, ax)
    if exact:
        return _nest(arr, lambda v: R(sp.radsimp(sp.expand(v))))
    if prec == "m":
        return _nest(arr, lambda v: R(complex(sp.N(v))) if not sp.sympify(v).is_real else float(sp.N(v)))
    dps = mpmath.libmp.prec_to_dps(prec)
    return _nest(arr, lambda v: R(sp.N(v, dps)))


fn("DiscreteHadamardTransform", area=AREA, status="added",
   notes="DiscreteHadamardTransform[list] = 1/√n Σ u_r (-1)^(Σ_j b_j(r-1) p_j(s-1)) (Walsh–Hadamard) on arrays of any "
         "depth, zero-padded to powers of 2; Method -> \"Sequency\" (default, Walsh order), \"BitComplement\" "
         "(Sylvester/natural) or \"GrayCode\" (Paley/dyadic); exact input gives exact results (Wolfram does the "
         "same), inexact input keeps its precision; the transform is its own inverse. Check: exact; reference "
         "(Wolfram documentation example) and the self-inverse round trip in tests.")(_hadamard)


def _hilbert(x: ListOf[Real]) -> ListOf[Real]:
    fname = "DiscreteHilbertTransform"
    flat, shape = _array(x, fname)
    if not _is_real_data(flat):
        raise ValueError(f"{fname}: the data must be real")
    prec = _work_prec(flat)
    _check_bignum_size(shape, prec, fname)
    if prec == "m":
        a = np.real(_to_np(flat, shape))
        out = a.astype(complex)
        for ax, n in enumerate(shape):
            F = np.fft.fft(out, axis=ax)
            h = np.zeros(n)
            h[0] = 1
            if n % 2 == 0:
                h[n // 2] = 1
                h[1:n // 2] = 2
            else:
                h[1:(n + 1) // 2] = 2
            sh = [1] * len(shape)
            sh[ax] = n
            out = np.fft.ifft(F * h.reshape(sh), axis=ax)
        return _out(np.imag(out) if len(shape) == 1 else _hilbert_nd_imag(a, shape), "m", real=True)
    arr = _to_mp(flat, shape, prec)
    if len(shape) != 1:
        raise NotImplementedError(f"{fname}: bignum input is supported for vectors only")
    n = shape[0]
    with mpmath.workprec(prec + 20):
        F = [mpmath.fsum(arr[r] * mpmath.expjpi(-2 * mpmath.mpf(r * k) / n) for r in range(n)) for k in range(n)]
        h = [1] + [2 if 0 < k < (n + 1) / 2 else (1 if 2 * k == n else 0) for k in range(1, n)]
        z = [mpmath.fsum(F[k] * h[k] * mpmath.expjpi(2 * mpmath.mpf(r * k) / n) for k in range(n)) / n
             for r in range(n)]
    return L(*(_num_out(mpmath.im(v), prec, real=True) for v in z))


def _hilbert_nd_imag(a, shape):
    """Multidimensional analytic signal: Wolfram applies the 1-D analytic-signal filter along each dimension
    in turn and returns the imaginary part of the result."""
    out = a.astype(complex)
    for ax, n in enumerate(shape):
        F = np.fft.fft(out, axis=ax)
        h = np.zeros(n)
        h[0] = 1
        if n % 2 == 0:
            h[n // 2] = 1
            h[1:n // 2] = 2
        else:
            h[1:(n + 1) // 2] = 2
        sh = [1] * len(shape)
        sh[ax] = n
        out = np.fft.ifft(F * h.reshape(sh), axis=ax)
    return np.imag(out)


fn("DiscreteHilbertTransform", area=AREA, status="added",
   notes="DiscreteHilbertTransform[list]: FFT, suppress negative frequencies (double positive ones), inverse FFT, "
         "imaginary part (so list + I DiscreteHilbertTransform[list] is the analytic signal); real vectors at "
         "machine or bignum precision, machine arrays of higher depth (filter applied along each dimension); "
         "exact input is numericized; complex data raise ValueError. Check: reference (Wolfram documentation "
         "values; equals SciPy's imag(hilbert(x)) for vectors) in tests.")(_hilbert)


def _num_param(v, fname, what):
    s = S(v)
    if not (isinstance(s, sp.Basic) and s.is_number):
        if isinstance(s, sp.Basic) and s.free_symbols:
            raise Unevaluated()
        raise TypeError(f"{fname}: {what} must be a number")
    return s


def _chirpz(x: ListOf[Number], *rest: Expr) -> ListOf[Number]:
    fname = "DiscreteChirpZTransform"
    flat, shape = _array(x, fname)
    d = len(shape)
    if len(rest) > 3:
        raise ValueError(f"{fname}: called with too many arguments")

    def per_dim(v, default, what, conv):
        if v is None:
            return [default(k) for k in range(d)]
        if d > 1:
            if not (is_list(v) and len(v.args) == d):
                raise ValueError(f"{fname}: {what} must be a list of length {d} for a rank-{d} array")
            return [conv(t) for t in v.args]
        return [conv(v)]

    def to_n(t):
        if isinstance(t, bool) or not K.is_int(t):
            if isinstance(t, Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: the number of points must be a positive integer")
        if int(t) < 1:
            raise ValueError(f"{fname}: the number of points must be a positive integer")
        return int(t)
    ns = per_dim(rest[0] if rest else None, lambda k: shape[k], "the number of points", to_n)
    ws = per_dim(rest[1] if len(rest) > 1 else None, lambda k: sp.exp(2 * sp.pi * sp.I / ns[k]), "w",
                 lambda t: _num_param(t, fname, "w"))
    As = per_dim(rest[2] if len(rest) > 2 else None, lambda k: sp.Integer(1), "a", lambda t: _num_param(t, fname, "a"))
    for w in ws:
        if w == 0:
            raise ValueError(f"{fname}: w must be nonzero")
    for a in As:
        if a == 0:
            raise ValueError(f"{fname}: a must be nonzero")
    prec = _work_prec(flat + ws + As)
    _check_bignum_size(shape, prec, fname)
    arr = _to_np(flat, shape) if prec == "m" else _to_mp(flat, shape, prec)
    mats = []
    for k in range(d):
        n_in, n_out = shape[k], ns[k]
        if prec == "m":
            w, a = complex(sp.N(ws[k])), complex(sp.N(As[k]))
            mats.append(np.array([[(a * w ** s) ** (-r) for r in range(n_in)] for s in range(n_out)], dtype=complex))
        else:
            dps = mpmath.libmp.prec_to_dps(prec) + 10
            with mpmath.workprec(prec + 20):
                w = mpmath.mpc(complex(0)) + sp.N(ws[k], dps)._to_mpmath(prec + 20) if ws[k].is_real else \
                    mpmath.mpc(sp.N(sp.re(ws[k]), dps)._to_mpmath(prec + 20), sp.N(sp.im(ws[k]), dps)._to_mpmath(prec + 20))
                a = mpmath.mpc(sp.N(sp.re(As[k]), dps)._to_mpmath(prec + 20), sp.N(sp.im(As[k]), dps)._to_mpmath(prec + 20))
                M = np.empty((n_out, n_in), dtype=object)
                for s in range(n_out):
                    for r in range(n_in):
                        M[s, r] = mpmath.power(a * mpmath.power(w, s), -r)
            mats.append(M)
    return _out(_apply_matrix_axes(arr, mats, prec), prec)


fn("DiscreteChirpZTransform", area=AREA, status="added",
   notes="DiscreteChirpZTransform[list, n, w, a] = {Σ_r u_r z_k^-(r-1)} at z_k = a w^k, k = 0..n-1 (the list Z "
         "transform along a spiral); defaults n = Length[list], w = Exp[2πI/n], a = 1, so "
         "DiscreteChirpZTransform[list] == Fourier[list, FourierParameters -> {1, -1}]; arrays of any depth take "
         "lists {n1, ...}, {w1, ...}, {a1, ...}; results are numeric. Check: reference (Wolfram documentation "
         "values) and the Fourier and ListZTransform identities in tests.")(_chirpz)


# ═════════════════════════════════════════ list Z / Fourier sequence transforms ══
def _sym_array(x, fname):
    """Rectangular array with arbitrary (possibly symbolic) entries -> (numpy object array of SymPy values)."""
    if not is_list(x):
        if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a list, not {x!r}")
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
            if isinstance(z, str) or isinstance(z, bool):
                raise TypeError(f"{fname}: array elements must be numbers or expressions, not {z!r}")
            flat.append(S(z))
            return
        if not is_list(z) or len(z.args) != shape[d]:
            raise ValueError(f"{fname}: the array of data must be rectangular")
        for a in z.args:
            walk(a, d + 1)
    walk(x, 0)
    arr = np.empty(len(flat), dtype=object)
    arr[:] = flat
    return arr.reshape(shape)


def _vars(v, d, fname):
    vs = list(v.args) if is_list(v) else [v]
    if len(vs) != d:
        raise ValueError(f"{fname}: {d} variable(s) are needed for a rank-{d} array")
    out = []
    for t in vs:
        if isinstance(t, Symbol):
            out.append(sp.Symbol(t.name))
        elif isinstance(t, str) or isinstance(t, bool):
            raise TypeError(f"{fname}: {t!r} is not a valid variable")
        else:
            out.append(S(t))     # a numeric point: evaluate the transform there
    return out


def _offsets(k, d, fname):
    if k is None:
        return [0] * d
    ks = list(k.args) if is_list(k) else [k] * d if d == 1 else None
    if ks is None:
        raise ValueError(f"{fname}: the offset must be a list of {d} integers")
    if len(ks) != d:
        raise ValueError(f"{fname}: the offset must have {d} components")
    out = []
    for t in ks:
        if isinstance(t, bool) or not K.is_int(t):
            if isinstance(t, Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: offsets must be integers, not {t!r}")
        out.append(int(t))
    return out


def _list_ztransform(x: ListOf[Expr], z: Var | ListOf[Var], k: Integer | ListOf[Integer] = None) -> Expr:
    fname = "ListZTransform"
    arr = _sym_array(x, fname)
    zs = _vars(z, arr.ndim, fname)
    ks = _offsets(k, arr.ndim, fname)
    terms = []
    for idx in np.ndindex(*arr.shape):
        t = arr[idx]
        for j, r in enumerate(idx):
            t = t * zs[j] ** (-(r + ks[j]))
        terms.append(t)
    return R(sp.Add(*terms))


fn("ListZTransform", area=AREA, status="added",
   notes="ListZTransform[list, z, k] = Σ_r u_r z^-(r + k - 1) (default k = 0); arrays of any depth with variables "
         "{z1, z2, ...} and offsets {k1, ...}; symbolic entries allowed (exact algebra); a numeric z evaluates the "
         "sum. Check: exact; reference (Wolfram documentation examples) and agreement with "
         "DiscreteChirpZTransform on the unit circle in tests.")(_list_ztransform)


def _list_fst(x: ListOf[Expr], w: Var | ListOf[Var], *rest: Expr) -> Expr:
    fname = "ListFourierSequenceTransform"
    args, opts = _opts(rest, {"FourierParameters"}, fname)
    if len(args) > 1:
        raise ValueError(f"{fname}: called with too many arguments")
    a, b = _fourier_params(opts, fname, default=(1, 1))
    arr = _sym_array(x, fname)
    ws = _vars(w, arr.ndim, fname)
    ks = _offsets(args[0] if args else None, arr.ndim, fname)
    terms = []
    for idx in np.ndindex(*arr.shape):
        e = sum((r + ks[j]) * ws[j] for j, r in enumerate(idx))
        terms.append(arr[idx] * sp.exp(-sp.I * b * e))
    scale = (sp.Abs(b) / (2 * sp.pi)) ** (arr.ndim * (1 - a) / 2)
    return R(scale * sp.Add(*terms))


fn("ListFourierSequenceTransform", area=AREA, status="added",
   notes="ListFourierSequenceTransform[list, ω, k] = |b/2π|^((1-a)/2) Σ_r u_r e^(-i b (r + k - 1) ω) with "
         "FourierParameters -> {a, b} (default {1, 1}); arrays of any depth with {ω1, ...} and offsets; exact "
         "symbolic result. Check: exact; reference (Wolfram documentation examples) in tests.")(_list_fst)


# ═══════════════════════════════════════════════════ windows, periodograms ══
_WINDOWS = {
    "DirichletWindow": lambda x: 1.0 if abs(x) <= 0.5 else 0.0,
    "HammingWindow": lambda x: 25 / 46 + 21 / 46 * math.cos(2 * math.pi * x) if abs(x) <= 0.5 else 0.0,
    "HannWindow": lambda x: (1 + math.cos(2 * math.pi * x)) / 2 if abs(x) <= 0.5 else 0.0,
    "BlackmanWindow": lambda x: 21 / 50 + math.cos(2 * math.pi * x) / 2 + 2 / 25 * math.cos(4 * math.pi * x)
    if abs(x) <= 0.5 else 0.0,
    "BartlettWindow": lambda x: 1 - 2 * abs(x) if abs(x) <= 0.5 else 0.0,
}


def _window(wfun, n, fname):
    """Window samples at n points spanning [-1/2, 1/2] (inclusive), or an explicit list of n weights."""
    if wfun is None or (isinstance(wfun, Symbol) and wfun.name in ("Automatic", "DirichletWindow")):
        return np.ones(n), True
    if is_list(wfun):
        flat, shape = _array(wfun, fname)
        if shape != (n,):
            raise ValueError(f"{fname}: the window list must have length {n}")
        return np.real(_to_np(flat, shape)), False
    xs = [-0.5 + (k / (n - 1) if n > 1 else 0.5) for k in range(n)]
    if isinstance(wfun, Symbol) and wfun.name in _WINDOWS:
        return np.array([_WINDOWS[wfun.name](t) for t in xs]), False
    vals = []
    for t in xs:
        v = apply(wfun, t)
        try:
            c = complex(sp.N(S(v)))
        except Exception:
            raise ValueError(f"{fname}: the window function {wfun!r} did not give a number at {t}")
        vals.append(c.real)
    return np.array(vals), False


def _int_arg(v, fname, what, lo=1):
    if isinstance(v, bool) or not K.is_int(v):
        if isinstance(v, Symbol):
            raise Unevaluated()
        if isinstance(v, Expr) and v.head == "Quantity" and len(v.args) == 2 and v.args[1] == "Samples":
            return _int_arg(v.args[0], fname, what, lo)
        raise TypeError(f"{fname}: {what} must be a positive integer, not {v!r}")
    if int(v) < lo:
        raise ValueError(f"{fname}: {what} must be at least {lo}")
    return int(v)


def _partitions(vec, n, d):
    if n > len(vec):
        return []
    return [vec[i:i + n] for i in range(0, len(vec) - n + 1, d)]


def _fourier_np(x, a, b, axis=None):
    """Fourier with parameters {a, b} (b = ±1) along all axes of a numpy array."""
    if abs(b) != 1:
        raise NotImplementedError("FourierParameters with |b| != 1 are not supported here")
    axes = tuple(range(x.ndim))
    nn = np.prod(x.shape)
    if b > 0:
        return np.fft.ifftn(x, axes=axes) * nn * nn ** (-(1 - a) / 2)
    return np.fft.fftn(x, axes=axes) * nn ** (-(1 - a) / 2)


def _ifourier_np(x, a, b):
    axes = tuple(range(x.ndim))
    nn = np.prod(x.shape)
    if b > 0:
        return np.fft.fftn(x, axes=axes) * nn ** (-(1 + a) / 2)
    return np.fft.ifftn(x, axes=axes) * nn * nn ** (-(1 + a) / 2)


def _fp_float(opts, fname, default):
    a, b = _fourier_params(opts, fname, default)
    if not b.is_Integer or abs(int(b)) != 1:
        raise NotImplementedError(f"{fname}: only FourierParameters {{a, ±1}} are supported")
    return float(a), int(b)


def _periodogram(x: ListOf[Number], *rest: Expr) -> ListOf[Real]:
    fname = "PeriodogramArray"
    args, opts = _opts(rest, {"FourierParameters"}, fname)
    if len(args) > 4:
        raise ValueError(f"{fname}: called with too many arguments")
    a, b = _fp_float(opts, fname, (0, 1))
    flat, shape = _array(x, fname)
    data = _to_np(flat, shape)
    if not args:
        return _out(np.abs(_fourier_np(data, a, b)) ** 2, "m", real=True)
    if len(shape) != 1:
        raise NotImplementedError(f"{fname}: partitioned periodograms are supported for vectors only")
    n = _int_arg(args[0], fname, "the partition length")
    d = _int_arg(args[1], fname, "the offset") if len(args) > 1 else n
    w, _ = _window(args[2] if len(args) > 2 else None, n, fname)
    m = _int_arg(args[3], fname, "the padded length") if len(args) > 3 else n
    if m < n:
        raise ValueError(f"{fname}: the padded length {m} is smaller than the partition length {n}")
    parts = _partitions(data, n, d)
    if not parts:
        raise ValueError(f"{fname}: the partition length {n} exceeds the data length {shape[0]}")
    rms = math.sqrt(float(np.mean(w ** 2)))
    if rms == 0:
        raise ValueError(f"{fname}: the window is identically zero")
    w = w / rms
    spec = [np.abs(_fourier_np(np.concatenate([p * w, np.zeros(m - n)]), a, b)) ** 2 for p in parts]
    return _out(np.mean(spec, axis=0), "m", real=True)


fn("PeriodogramArray", area=AREA, status="added",
   notes="PeriodogramArray[list] = Abs[Fourier[list]]^2 (FourierParameters default {0, 1}; arrays of any depth); "
         "PeriodogramArray[list, n, d, wfun, m] averages the power spectra of partitions of length n with offset d "
         "(incomplete partitions dropped), each multiplied by the window wfun sampled at n points on [-1/2, 1/2] "
         "(or a list of n weights) normalized to unit RMS, zero-padded to length m (vectors only); built-in "
         "windows DirichletWindow, HammingWindow, HannWindow, BlackmanWindow, BartlettWindow, other functions are "
         "applied; machine precision; FourierParameters {a, ±1}. Check: reference (Wolfram documentation values, "
         "including the HammingWindow and list-window examples) in tests.")(_periodogram)


def _cepstrum_vec(data, kind, a, b):
    F = _fourier_np(data, a, b)
    eps = np.finfo(float).eps
    if kind == "Power":
        lp = np.log(np.maximum(np.abs(F) ** 2, eps))
        return np.abs(_ifourier_np(lp.astype(complex), a, b)) ** 2
    lp = np.log(np.maximum(np.abs(F), eps))
    return np.real(_ifourier_np(lp.astype(complex), a, b))


def _cepstrum(x: ListOf[Number], *rest: Expr) -> ListOf[Real]:
    fname = "CepstrumArray"
    args, opts = _opts(rest, {"FourierParameters"}, fname)
    if len(args) > 1:
        raise ValueError(f"{fname}: called with too many arguments")
    kind = args[0] if args else "Power"
    if kind not in ("Power", "Real"):
        raise ValueError(f"{fname}: the cepstrum type must be \"Power\" or \"Real\", not {kind!r}")
    a, b = _fp_float(opts, fname, (1, -1))
    flat, shape = _array(x, fname)
    return _out(_cepstrum_vec(_to_np(flat, shape), kind, a, b), "m", real=True)


fn("CepstrumArray", area=AREA, status="added",
   notes="CepstrumArray[data] = |F^-1 log(|F data|^2)|^2 (power cepstrum), CepstrumArray[data, \"Real\"] = "
         "Re F^-1 log|F data| (real cepstrum), F = Fourier with FourierParameters (default {1, -1}), arrays of "
         "any depth, machine precision; zero spectral values are clamped at $MachineEpsilon before the log, as "
         "in Wolfram's results. Check: reference (Wolfram documentation values, including a complex 3×3 array) in "
         "tests.")(_cepstrum)


def _cepstrogram(x: ListOf[Number], *rest: Expr) -> MatrixOf[Real]:
    fname = "CepstrogramArray"
    args, opts = _opts(rest, {"FourierParameters", "Padding", "PaddingSize"}, fname)
    if len(args) > 4:
        raise ValueError(f"{fname}: called with too many arguments")
    for o, ok in (("Padding", lambda v: isinstance(v, Symbol) and v.name in ("Automatic", "None")),
                  ("PaddingSize", lambda v: is_list(v) and all(K.is_int(t) and int(t) == 0 for t in v.args)
                   or K.is_int(v) and int(v) == 0)):
        if o in opts and not ok(opts[o]):
            raise NotImplementedError(f"{fname}: {o} -> {opts[o]!r} is not supported (no padding)")
    a, b = _fp_float(opts, fname, (1, -1))
    flat, shape = _array(x, fname, max_rank=1)
    m = shape[0]
    n = _int_arg(args[0], fname, "the partition length") if args else 2 ** int(round(math.log2(math.sqrt(m)) + 1))
    d = _int_arg(args[1], fname, "the offset") if len(args) > 1 else max(1, int(round(n / 3)))
    if len(args) > 2:
        w, dirichlet = _window(args[2], n, fname)
        if not dirichlet:
            raise NotImplementedError(f"{fname}: smoothing windows other than DirichletWindow are not supported")
    pad = _int_arg(args[3], fname, "the padded length") if len(args) > 3 else n
    if pad < n:
        raise ValueError(f"{fname}: the padded length is smaller than the partition length")
    data = _to_np(flat, shape)
    parts = _partitions(data, n, d)
    if not parts:
        raise ValueError(f"{fname}: the partition length {n} exceeds the data length {m}")
    rows = [_cepstrum_vec(np.concatenate([p, np.zeros(pad - n)]), "Power", a, b) for p in parts]
    return _out(np.array(rows), "m", real=True)


fn("CepstrogramArray", area=AREA, status="added",
   notes="CepstrogramArray[list, n, d] = power cepstra (as CepstrumArray) of the partitions of length n and offset d "
         "(defaults n = 2^Round[Log2[Sqrt[m]] + 1], d = Round[n/3]; incomplete partitions dropped), optionally "
         "zero-padded to length m; only DirichletWindow (no smoothing) and no extra padding are supported "
         "(other windows/Padding settings raise NotImplementedError); vectors at machine precision. Check: "
         "reference (row-by-row agreement with CepstrumArray, Wolfram's documented partition sizes) in "
         "tests.")(_cepstrogram)


# ═════════════════════════════════════════════════ ListConvolve / ListCorrelate ══
def _gen_array(x, fname, depth=None):
    """Rectangular array of arbitrary elements (to the given depth, or full depth) -> numpy object array."""
    if not is_list(x):
        if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a list, not {x!r}")
    shape = []
    y = x
    while is_list(y) and (depth is None or len(shape) < depth):
        shape.append(len(y.args))
        if not y.args:
            raise ValueError(f"{fname}: lists must be nonempty")
        y = y.args[0]
    if depth is not None and len(shape) < depth:
        raise ValueError(f"{fname}: the list must have depth at least {depth}")
    items = []

    def walk(z, d):
        if d == len(shape):
            items.append(z)
            return
        if not is_list(z) or len(z.args) != shape[d]:
            raise ValueError(f"{fname}: the arrays must be rectangular")
        for a in z.args:
            walk(a, d + 1)
    walk(x, 0)
    if depth is None:
        # full depth: shrink to the depth at which every branch is rectangular
        pass
    arr = np.empty(len(items), dtype=object)
    arr[:] = items
    return arr.reshape(shape)


def _array_depth(x):
    d = 0
    while is_list(x) and x.args:
        d += 1
        x = x.args[0]
    return d


def _is_plain_number(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)) or (isinstance(v, sp.Basic) and v.is_number)


def _exact_or_num(v, pr):
    """An exact SymPy result, or its numeric value at precision pr ('m' or bits)."""
    if pr is None:
        return R(sp.expand(v) if v.is_Add else v)
    if pr == "m":
        c = complex(sp.N(v, 17))
        return float(c.real) if c.imag == 0 and sp.sympify(v).is_real is not False else R(c)
    return R(sp.N(v, mpmath.libmp.prec_to_dps(pr)))


def _kspec(k, d, m, fname, default):
    """Alignment spec -> ([kL'_i], [kR'_i]) as 1-based positive kernel positions."""
    def one(v, mi):
        if isinstance(v, bool) or not K.is_int(v):
            if isinstance(v, Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: alignment positions must be integers, not {v!r}")
        v = int(v)
        if v == 0 or abs(v) > mi:
            raise ValueError(f"{fname}: alignment position {v} is outside the kernel (length {mi})")
        return v if v > 0 else mi + 1 + v
    if k is None:
        kl, kr = default
        return [one(kl, mi) for mi in m], [one(kr, mi) for mi in m]
    if is_list(k):
        if len(k.args) != 2:
            raise ValueError(f"{fname}: the alignment must be k, {{kL, kR}} or {{{{kL1, ...}}, {{kR1, ...}}}}")
        kl, kr = k.args
        if is_list(kl) or is_list(kr):
            if not (is_list(kl) and is_list(kr) and len(kl.args) == d and len(kr.args) == d):
                raise ValueError(f"{fname}: per-dimension alignments need {d} entries")
            return [one(a, mi) for a, mi in zip(kl.args, m)], [one(b, mi) for b, mi in zip(kr.args, m)]
        return [one(kl, mi) for mi in m], [one(kr, mi) for mi in m]
    return [one(k, mi) for mi in m], [one(k, mi) for mi in m]


def _list_conv(fname, correlate):
    def impl(ker: ListOf[Expr], lst: ListOf[Expr], *rest: Expr) -> ListOf[Expr]:
        if len(rest) > 4:
            raise ValueError(f"{fname}: called with too many arguments")
        kd = _array_depth(ker)
        if kd == 0:
            if isinstance(ker, (Symbol, Expr)) and not is_list(ker):
                raise Unevaluated()
            raise TypeError(f"{fname}: the kernel must be a nonempty list")
        if _array_depth(lst) < kd:
            if isinstance(lst, Symbol) or (isinstance(lst, Expr) and not is_list(lst)):
                raise Unevaluated()
            raise ValueError(f"{fname}: the list must have at least the depth of the kernel")
        K_ = _gen_array(ker, fname, kd)
        A = _gen_array(lst, fname, kd)
        m, n = K_.shape, A.shape
        default = (1, -1) if correlate else (-1, 1)
        kl, kr = _kspec(rest[0] if rest else None, kd, m, fname, default)
        cyclic_default = len(rest) < 2
        pad = None if cyclic_default else rest[1]
        g = rest[2] if len(rest) > 2 else None
        h = rest[3] if len(rest) > 3 else None
        if pad is not None and is_list(pad) and not pad.args:
            pad_mode = "none"
        elif pad is None:
            pad_mode = "cyclic"
            P = A
        elif is_list(pad):
            pad_mode = "cyclic"
            pd = _array_depth(pad)
            if pd < kd:
                raise NotImplementedError(f"{fname}: a padding list must have the depth of the kernel")
            P = _gen_array(pad, fname, kd)
        else:
            pad_mode = "const"
        # output ranges (1-based list position aligned with kernel position 1)
        if correlate:
            firsts = [2 - kl[i] for i in range(kd)]
            lasts = [n[i] + 1 - kr[i] for i in range(kd)]
        else:
            firsts = [kl[i] for i in range(kd)]
            lasts = [n[i] + kr[i] - 1 for i in range(kd)]
        outshape = [lasts[i] - firsts[i] + 1 for i in range(kd)]
        if any(s <= 0 for s in outshape):
            raise ValueError(f"{fname}: the kernel is longer than the list for this alignment")
        numeric = g is None and h is None and pad_mode != "none" and \
            all(_is_plain_number(S(v)) if not isinstance(v, (str, bool)) and not is_list(v) else False
                for v in list(K_.reshape(-1)) + list(A.reshape(-1)) +
                ([pad] if pad_mode == "const" else list(P.reshape(-1))))
        if numeric:
            Kn = np.empty(K_.shape, dtype=object)
            Kn.reshape(-1)[:] = [S(v) if not isinstance(v, (int, float)) else v for v in K_.reshape(-1)]
            An = np.empty(A.shape, dtype=object)
            An.reshape(-1)[:] = [S(v) if not isinstance(v, (int, float)) else v for v in A.reshape(-1)]
            if pad_mode == "cyclic":
                Pn = np.empty(P.shape, dtype=object)
                Pn.reshape(-1)[:] = [S(v) if not isinstance(v, (int, float)) else v for v in P.reshape(-1)]
            else:
                pval = S(pad) if not isinstance(pad, (int, float)) else pad
        out = np.empty(outshape, dtype=object)
        korder = list(np.ndindex(*m))
        # order terms by increasing list position (Wolfram's order for g/h)
        if not correlate:
            korder = sorted(korder, key=lambda t: tuple(-c for c in t))
        for oidx in np.ndindex(*outshape):
            p = [firsts[i] + oidx[i] for i in range(kd)]
            terms = []
            for kidx in korder:
                j = [(p[i] + kidx[i]) if correlate else (p[i] - kidx[i]) for i in range(kd)]   # 1-based
                inside = all(1 <= j[i] <= n[i] for i in range(kd))
                if inside:
                    a = (An if numeric else A)[tuple(j[i] - 1 for i in range(kd))]
                elif pad_mode == "none":
                    a = None
                elif pad_mode == "const":
                    a = pval if numeric else pad
                else:
                    src = Pn if numeric else P
                    a = src[tuple((j[i] - 1) % src.shape[i] for i in range(kd))]
                kv = (Kn if numeric else K_)[kidx]
                terms.append((kv, a))
            if numeric:
                out[oidx] = sp.Add(*(sp.sympify(kv) * sp.sympify(a) for kv, a in terms))
                continue
            gs = "Times" if g is None else (g.name if isinstance(g, Symbol) else g)
            prods = [Expr(gs, kv) if a is None else Expr(gs, kv, a) for kv, a in terms]
            hs = "Plus" if h is None else (h.name if isinstance(h, Symbol) else h)
            if kd == 1:
                out[oidx] = ev(Expr(hs, *prods))
            else:
                # nested h with depth equal to the kernel depth
                arr = np.empty(len(prods), dtype=object)
                arr[:] = prods
                arr = arr.reshape([m[i] for i in range(kd)])
                if not correlate:
                    arr = arr[tuple(slice(None, None, -1) for _ in range(kd))]

                def nest(z, d):
                    if d == kd:
                        return z
                    return Expr(hs, *(nest(z[i], d + 1) for i in range(z.shape[0])))
                out[oidx] = ev(nest(arr, 0))
        if numeric:
            pr = _prec([sp.sympify(v) for v in list(Kn.reshape(-1)) + list(An.reshape(-1))] +
                       ([sp.sympify(pval)] if pad_mode == "const" else
                        [sp.sympify(v) for v in Pn.reshape(-1)] if pad_mode == "cyclic" else []))
            return _nest(out, lambda v: _exact_or_num(v, pr))
        return _nest(out, lambda v: v)
    return impl


fn("ListConvolve", area=AREA, status="added",
   notes="ListConvolve[ker, list, {kL, kR}, padlist, g, h]: Σ_r K_r a_(s-r) with Wolfram's alignment (kernel "
         "position kL on the first list element, kR on the last; default {-1, 1}, no overhang; k means {k, k}; "
         "negative positions count from the end; {{kL1, ...}, {kR1, ...}} per dimension), padding (default cyclic "
         "list, a constant p, cyclic repetitions of a list, {} for none, where g gets the kernel element alone), "
         "general g (for Times) and h (for Plus, nested to the kernel depth) and kernels of any rank (lists may be "
         "deeper; their elements are then combined by ordinary arithmetic); exact and symbolic data allowed. "
         "Check: exact; reference (the Wolfram documentation examples, symbolic) in tests.")(_list_conv("ListConvolve", False))
fn("ListCorrelate", area=AREA, status="added",
   notes="ListCorrelate[ker, list, {kL, kR}, padlist, g, h]: Σ_r K_r a_(s+r) with Wolfram's alignment (default "
         "{1, -1}, no overhang), padding, g/h and multidimensional kernels as ListConvolve. Check: exact; reference "
         "(Wolfram documentation examples) and ListCorrelate[ker, list] == ListConvolve[Reverse[ker], list] in "
         "tests.")(_list_conv("ListCorrelate", True))


# ═══════════════════════════════════════════════════════════ moving statistics ══
def _vec_items(x, fname):
    if not is_list(x):
        if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a list, not {x!r}")
    if not x.args:
        raise ValueError(f"{fname}: the list must be nonempty")
    return list(x.args)


def _arith(head, *args):
    return ev(Expr(head, *args))


def _moving_average(x: ListOf[Expr], r: PosInteger | ListOf[Expr]) -> ListOf[Expr]:
    fname = "MovingAverage"
    items = _vec_items(x, fname)
    if is_list(r):
        w = list(r.args)
        if not w:
            raise ValueError(f"{fname}: the weight list must be nonempty")
        total = _arith("Plus", *w)
        if total == 0 or (isinstance(total, (int, float)) and total == 0):
            raise ValueError(f"{fname}: the weights must not sum to zero")
        k = len(w)
        if k > len(items):
            raise ValueError(f"{fname}: there are more weights ({k}) than list elements ({len(items)})")
        return L(*(_arith("Times", _arith("Plus", *(_arith("Times", w[j], items[i + j]) for j in range(k))),
                          _arith("Power", total, -1)) for i in range(len(items) - k + 1)))
    if isinstance(r, bool) or not K.is_int(r):
        if isinstance(r, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the span must be a positive integer or a list of weights, not {r!r}")
    k = int(r)
    if k < 1:
        raise ValueError(f"{fname}: the span must be a positive integer")
    if k > len(items):
        raise ValueError(f"{fname}: the span {k} exceeds the list length {len(items)}")
    return L(*(_arith("Times", _arith("Plus", *items[i:i + k]), sp.Rational(1, k)) for i in range(len(items) - k + 1)))


fn("MovingAverage", area=AREA, status="added",
   notes="MovingAverage[list, r]: means of the runs of r consecutive elements (length n - r + 1); "
         "MovingAverage[list, {w1, ..., wr}]: Σ w_i a_(s+i-1) / Σ w_i; elements may be numbers, symbols or "
         "equal-length vectors (componentwise); exact arithmetic for exact input; r > n or zero weight sum raise "
         "ValueError; TimeSeries and SparseArray objects are not supported. Check: exact; reference (Wolfram "
         "documentation examples) in tests.")(_moving_average)


def _median_of(vals, fname):
    """Median of numeric values (exact or inexact), Wolfram-style: mean of the middle two for even counts."""
    keyed = []
    for v in vals:
        s = S(v) if not isinstance(v, (int, float)) else sp.sympify(v)
        if not (isinstance(s, sp.Basic) and s.is_number and s.is_real is not False):
            if isinstance(s, sp.Basic) and s.free_symbols:
                raise Unevaluated()
            raise TypeError(f"{fname}: elements must be real numbers, not {v!r}")
        keyed.append((float(sp.N(s, 30)), v))
    keyed.sort(key=lambda t: t[0])
    n = len(keyed)
    if n % 2:
        return keyed[n // 2][1]
    return _arith("Times", _arith("Plus", keyed[n // 2 - 1][1], keyed[n // 2][1]), sp.Rational(1, 2))


def _moving_median(x: ListOf[Real], r: PosInteger) -> ListOf[Real]:
    fname = "MovingMedian"
    items = _vec_items(x, fname)
    if isinstance(r, bool) or not K.is_int(r):
        if isinstance(r, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the span must be a positive integer, not {r!r}")
    k = int(r)
    if k < 1:
        raise ValueError(f"{fname}: the span must be a positive integer")
    if k > len(items):
        raise ValueError(f"{fname}: the span {k} exceeds the list length {len(items)}")
    vec = is_list(items[0])
    if vec:
        dim = len(items[0].args)
        if any(not is_list(v) or len(v.args) != dim for v in items):
            raise ValueError(f"{fname}: vector elements must all have the same length")
    out = []
    for i in range(len(items) - k + 1):
        win = items[i:i + k]
        if vec:
            out.append(L(*(_median_of([w.args[c] for w in win], fname) for c in range(dim))))
        else:
            out.append(_median_of(win, fname))
    return L(*out)


fn("MovingMedian", area=AREA, status="added",
   notes="MovingMedian[list, r]: medians of the spans of r consecutive elements (mean of the two middle values for "
         "even r), componentwise for lists of equal-length vectors; exact values are ordered numerically and "
         "returned exactly; symbolic elements stay unevaluated; r > n raises ValueError. Check: exact; reference "
         "(Wolfram documentation examples) in tests.")(_moving_median)


def _ema(x: ListOf[Expr], a: Numeric) -> ListOf[Expr]:
    fname = "ExponentialMovingAverage"
    items = _vec_items(x, fname)
    if isinstance(a, (str, bool)) or is_list(a):
        raise TypeError(f"{fname}: the smoothing constant must be a number, not {a!r}")
    sa = S(a) if not isinstance(a, (int, float)) else sp.sympify(a)
    if isinstance(sa, sp.Basic) and sa.is_number and sa.is_real and not (0 <= sa <= 1):
        raise ValueError(f"{fname}: the smoothing constant must be between 0 and 1")
    out = [items[0]]
    y = items[0]
    for v in items[1:]:
        y = _arith("Plus", y, _arith("Times", a, _arith("Plus", v, _arith("Times", -1, y))))
        out.append(y)
    return L(*out)


fn("ExponentialMovingAverage", area=AREA, status="added",
   notes="ExponentialMovingAverage[list, α]: y_1 = x_1, y_i = y_(i-1) + α (x_i - y_(i-1)); numbers, symbols or "
         "vectors; exact for exact input, a numeric α outside [0, 1] raises ValueError, a symbolic α gives the "
         "symbolic recursion. Check: exact; reference (Wolfram documentation examples) in tests.")(_ema)


def _moving_map(f: Fn, x: ListOf[Expr], w: Expr, *rest: Expr) -> ListOf[Expr]:
    fname = "MovingMap"
    if rest:
        raise NotImplementedError(f"{fname}: the padding argument is not supported")
    items = _vec_items(x, fname)
    align = "Right"
    if is_list(w) and len(w.args) == 2 and isinstance(w.args[1], Symbol) and w.args[1].name in ("Left", "Right",
                                                                                               "Center"):
        align = w.args[1].name
        w = w.args[0]
    events = False
    if isinstance(w, Expr) and w.head == "Quantity" and len(w.args) == 2 and w.args[1] in ("Events", "Event"):
        events = True
        w = w.args[0]
    if isinstance(w, bool) or not K.is_int(w):
        if isinstance(w, Symbol):
            raise Unevaluated()
        raise NotImplementedError(f"{fname}: window specification {w!r} is not supported (use an integer span or "
                                  f"Quantity[n, \"Events\"])")
    r = int(w)
    if r < (1 if events else 0):
        raise ValueError(f"{fname}: the window size must be positive")
    pairs = all(is_list(v) and len(v.args) == 2 and K.is_int(v.args[0]) for v in items) and \
        all(int(items[i].args[0]) < int(items[i + 1].args[0]) for i in range(len(items) - 1))
    if pairs and not events:
        if align != "Right":
            raise NotImplementedError(f"{fname}: only Right alignment is supported for {{t, v}} data")
        ts = [int(v.args[0]) for v in items]
        out = []
        for i, t in enumerate(ts):
            if t - r < ts[0]:
                continue
            vals = [items[j].args[1] for j in range(len(ts)) if t - r <= ts[j] <= t]
            out.append(L(t, apply(f, L(*vals))))
        return L(*out)
    span = r if events else r + 1
    if pairs:
        if span > len(items):
            raise ValueError(f"{fname}: the window is longer than the data")
        return L(*(L(items[i + span - 1].args[0], apply(f, L(*(v.args[1] for v in items[i:i + span]))))
                   for i in range(len(items) - span + 1)))
    if span > len(items):
        raise ValueError(f"{fname}: the window is longer than the data")
    return L(*(apply(f, L(*items[i:i + span])) for i in range(len(items) - span + 1)))


fn("MovingMap", area=AREA, status="added",
   notes="MovingMap[f, list, r] applies f to the windows of r + 1 consecutive elements (a time span r on the "
         "implicit regular times), MovingMap[f, list, Quantity[n, \"Events\"]] to windows of n elements, and for "
         "{{t1, v1}, ...} data with integer times to the values with t - r <= t_j <= t (windows starting before the "
         "first time are dropped; result {t, f[...]}); alignment {r, Left | Right | Center} only affects time "
         "stamps (Right only for {t, v} data); padding and TimeSeries/TemporalData objects raise "
         "NotImplementedError. Check: exact; reference (Wolfram documentation examples) in tests.")(_moving_map)
