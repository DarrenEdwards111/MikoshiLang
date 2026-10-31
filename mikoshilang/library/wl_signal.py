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


# ═══════════════════════════════════════════════════════════════ FIR filters ══
_EXACT_WINDOWS = {
    "DirichletWindow": lambda x: sp.Integer(1) if abs(x) <= sp.Rational(1, 2) else sp.Integer(0),
    "HammingWindow": lambda x: sp.Rational(25, 46) + sp.Rational(21, 46) * sp.cos(2 * sp.pi * x),
    "HannWindow": lambda x: (1 + sp.cos(2 * sp.pi * x)) / 2,
    "BlackmanWindow": lambda x: sp.Rational(21, 50) + sp.cos(2 * sp.pi * x) / 2 + sp.Rational(2, 25) * sp.cos(4 * sp.pi * x),
    "BartlettWindow": lambda x: 1 - 2 * abs(x),
}


def _exact_window(wfun, n, fname):
    """Window samples at n points spanning [-1/2, 1/2] inclusive (Array[wfun, n, {-1/2, 1/2}]) as SymPy values."""
    if wfun is None:
        wfun = Symbol("HammingWindow")
    if isinstance(wfun, Symbol) and wfun.name == "None":
        return [sp.Integer(1)] * n
    xs = [sp.Rational(-1, 2) + (sp.Rational(k, n - 1) if n > 1 else sp.Rational(1, 2)) for k in range(n)]
    if is_list(wfun):
        if len(wfun.args) != n:
            raise ValueError(f"{fname}: the window list must have length {n}")
        out = []
        for v in wfun.args:
            s = _elem(v, fname)
            out.append(s)
        return out
    if isinstance(wfun, Symbol) and wfun.name in _EXACT_WINDOWS:
        return [_EXACT_WINDOWS[wfun.name](t) for t in xs]
    out = []
    for t in xs:
        v = apply(wfun, R(t))
        s = S(v) if not isinstance(v, (int, float)) else sp.sympify(v)
        if not (isinstance(s, sp.Basic) and s.is_number):
            raise ValueError(f"{fname}: the window function {wfun!r} did not give a number at {t}")
        out.append(s)
    return out


def _band_kernel(bands, n):
    """Least-squares (ideal, zero-phase) FIR kernel: amplitudes a_i on [lo_i, hi_i] -> h(m), m = k - (n-1)/2."""
    out = []
    for k in range(n):
        m = sp.Rational(2 * k - (n - 1), 2)
        h = sp.Integer(0)
        for lo, hi, a in bands:
            if a == 0:
                continue
            if m == 0:
                h += a * (hi - lo) / sp.pi
            else:
                h += a * (sp.sin(hi * m) - sp.sin(lo * m)) / (sp.pi * m)
        out.append(h)
    return out


def _diff_kernel(wc, n):
    out = []
    for k in range(n):
        m = sp.Rational(2 * k - (n - 1), 2)
        out.append(sp.Integer(0) if m == 0 else (wc * m * sp.cos(wc * m) - sp.sin(wc * m)) / (sp.pi * m ** 2))
    return out


def _hilbert_kernel(wc, n):
    out = []
    for k in range(n):
        m = sp.Rational(2 * k - (n - 1), 2)
        out.append(sp.Integer(0) if m == 0 else (1 - sp.cos(wc * m)) / (sp.pi * m))
    return out


def _pad_index(j, n, mode):
    """Map an out-of-range 0-based index to a source index (or None for constant padding)."""
    if 0 <= j < n:
        return j
    if mode == "Fixed":
        return 0 if j < 0 else n - 1
    if mode == "Periodic":
        return j % n
    if mode in ("Reflected", "Reversed"):
        if n == 1:
            return 0
        period = 2 * (n - 1) if mode == "Reflected" else 2 * n
        j = j % period
        if mode == "Reflected":
            return j if j < n else period - j
        return j if j < n else period - 1 - j
    return None


def _padding_opt(v, fname):
    if v is None:
        return "Fixed", None
    if isinstance(v, Symbol) and v.name == "None":
        return "None", None
    if isinstance(v, Symbol) and v.name == "Automatic":
        return "Fixed", None
    if isinstance(v, str):
        if v in ("Fixed", "Periodic", "Reflected", "Reversed"):
            return v, None
        raise NotImplementedError(f"{fname}: Padding -> {v!r} is not supported")
    if _is_plain_number(S(v)) if not isinstance(v, (int, float)) else True:
        return "const", (S(v) if not isinstance(v, (int, float)) else sp.sympify(v))
    raise ValueError(f"{fname}: invalid Padding setting {v!r}")


def _fir_1d(vec, ker, mode, cval):
    """ListConvolve[ker, vec, Ceiling[n/2], padding] (padding None: no overhang) on a list of SymPy values."""
    n = len(ker)
    c = (n + 1) // 2          # 1-based alignment position of the kernel
    N = len(vec)
    out = []
    if mode == "None":
        rng = range(n - c, N - c + 1)
    else:
        rng = range(N)
    for s in rng:
        acc = sp.Integer(0)
        for r in range(n):
            j = s + (c - 1) - r           # ker[r] (0-based) multiplies list[s + c - 1 - r]
            if 0 <= j < N:
                v = vec[j]
            elif mode == "const":
                v = cval
            else:
                v = vec[_pad_index(j, N, mode)]
            acc += ker[r] * v
        out.append(acc)
    return out


def _filter_impl(fname, kind):
    def impl(data: ListOf[Number], spec: Expr, *rest: Expr) -> ListOf[Number]:
        args, opts = _opts(rest, {"Padding", "SampleRate"}, fname)
        if len(args) > 2:
            raise ValueError(f"{fname}: called with too many arguments")
        flat, shape = _array(data, fname)
        sr = opts.get("SampleRate")
        if sr is None or (isinstance(sr, Symbol) and sr.name == "Automatic"):
            srv = sp.Integer(1)
        else:
            srv = _num_param(sr, fname, "SampleRate")
            if not (srv.is_real and srv > 0):
                raise ValueError(f"{fname}: SampleRate must be a positive number")

        def freq(v):
            if isinstance(v, Expr) and v.head == "Quantity":
                raise NotImplementedError(f"{fname}: Quantity frequencies are not supported")
            w = _num_param(v, fname, "the frequency") / srv
            if not (w.is_real and 0 <= w <= sp.pi):
                raise ValueError(f"{fname}: frequencies must lie between 0 and SampleRate π")
            return w
        if kind in ("Lowpass", "Highpass", "Differentiator", "Hilbert"):
            if is_list(spec):
                raise NotImplementedError(f"{fname}: per-dimension frequency lists are not supported")
            wc = freq(spec)
            bands = [(sp.Integer(0), wc, 1)] if kind == "Lowpass" else [(wc, sp.pi, 1)]
        else:
            if is_list(spec) and len(spec.args) == 1 and is_list(spec.args[0]) and len(spec.args[0].args) == 2:
                w0 = freq(spec.args[0].args[0])
                q = _num_param(spec.args[0].args[1], fname, "the quality factor")
                if not (q.is_real and q > 0):
                    raise ValueError(f"{fname}: the quality factor must be positive")
                # q = w0/(w2 - w1), w0 = Sqrt[w1 w2]
                d = w0 / q
                w1 = (-d + sp.sqrt(d ** 2 + 4 * w0 ** 2)) / 2
                w2 = w1 + d
                if w2 > sp.pi:
                    raise ValueError(f"{fname}: the band extends beyond π")
            elif is_list(spec) and len(spec.args) == 2:
                w1, w2 = freq(spec.args[0]), freq(spec.args[1])
                if not w1 <= w2:
                    raise ValueError(f"{fname}: the band edges must be in increasing order")
            else:
                raise ValueError(f"{fname}: the band must be {{ω1, ω2}} or {{{{ω, q}}}}")
            bands = [(w1, w2, 1)] if kind == "Bandpass" else [(sp.Integer(0), w1, 1), (w2, sp.pi, 1)]
        exact = _prec(flat + [b[0] for b in bands] + [b[1] for b in bands]) is None
        # trivial bands (no kernel length needed): all-pass / all-stop
        allpass = kind in ("Lowpass", "Highpass", "Bandpass", "Bandstop") and \
            sum((hi - lo) for lo, hi, _ in bands) == sp.pi
        allstop = kind in ("Lowpass", "Highpass", "Bandpass", "Bandstop", "Hilbert", "Differentiator") and \
            all(hi - lo == 0 for lo, hi, _ in bands) if kind not in ("Differentiator", "Hilbert") else \
            (bands[0][1] - bands[0][0] == sp.pi and False)
        if kind in ("Differentiator", "Hilbert") and wc == 0:
            allstop = True
        if not args:
            if allpass or allstop:
                vals = flat if allpass else [sp.Integer(0)] * len(flat)
                arr = np.empty(len(vals), dtype=object)
                arr[:] = vals
                arr = arr.reshape(shape)
                return _nest(arr, lambda v: R(v) if exact else (float(sp.N(v)) if sp.sympify(v).is_real else R(complex(sp.N(v)))))
            raise NotImplementedError(f"{fname}: the automatic kernel length is not documented; give the kernel "
                                      f"length n explicitly ({fname}[data, ω, n])")
        n = _int_arg(args[0], fname, "the kernel length")
        win = _exact_window(args[1] if len(args) > 1 else None, n, fname)
        if kind in ("Lowpass", "Highpass", "Bandpass", "Bandstop"):
            h = _band_kernel(bands, n)
        elif kind == "Differentiator":
            h = _diff_kernel(wc, n)
        else:
            h = _hilbert_kernel(wc, n)
        ker = [sp.sympify(a) * b for a, b in zip(win, h)]
        if kind == "Lowpass":
            tot = sp.Add(*ker)
            if tot == 0:
                ker = [sp.Integer(0)] * n
            else:
                ker = [k / tot for k in ker]
        if not exact:
            ker = [sp.N(k, 20) for k in ker]
        mode, cval = _padding_opt(opts.get("Padding"), fname)
        arr = np.empty(len(flat), dtype=object)
        arr[:] = flat
        arr = arr.reshape(shape)
        for ax in range(len(shape)):
            a = np.moveaxis(arr, ax, 0)
            res = []
            idxs = list(np.ndindex(*a.shape[1:]))
            cols = [_fir_1d([a[(k,) + idx] for k in range(a.shape[0])], ker, mode, cval) for idx in idxs]
            L0 = len(cols[0]) if cols else 0
            out = np.empty((L0,) + a.shape[1:], dtype=object)
            for idx, col in zip(idxs, cols):
                for k in range(L0):
                    out[(k,) + idx] = col[k]
            arr = np.moveaxis(out, 0, ax)
        if exact:
            return _nest(arr, lambda v: R(v))
        return _nest(arr, lambda v: float(sp.N(v)) if sp.sympify(v).is_real is not False else R(complex(sp.N(v))))
    return impl


_FILTER_NOTE = ("FIR filter by the window method: kernel = window × LeastSquaresFilterKernel (the ideal zero-phase "
                "{kind} response sampled at m = k - (n-1)/2){norm}, applied as ListConvolve[ker, data, "
                "Ceiling[n/2], padding] along every dimension; the kernel length n is required (Wolfram's automatic "
                "length is undocumented: omitting it raises NotImplementedError, except for the exact all-pass / "
                "all-stop cases); window wfun (default HammingWindow, sampled as Array[wfun, n, {{-1/2, 1/2}}]), a "
                "list of weights or None; Padding -> \"Fixed\" (default), \"Periodic\", \"Reflected\", "
                "\"Reversed\", a constant or None (no overhang); SampleRate -> sr divides the frequencies; exact "
                "data and frequencies give exact results; Quantity frequencies raise NotImplementedError. Check: "
                "reference (Wolfram documentation examples, exact and numeric) and the explicit kernel/ListConvolve "
                "identity in tests.")
for _name, _kind, _resp, _norm in (
        ("LowpassFilter", "Lowpass", "lowpass (passband 0..ωc)", ", normalized to unit sum"),
        ("HighpassFilter", "Highpass", "highpass (passband ωc..π)", ""),
        ("BandpassFilter", "Bandpass", "bandpass ({ω1, ω2} or {{ω, q}} with q = ω/(ω2 - ω1), ω = √(ω1 ω2))", ""),
        ("BandstopFilter", "Bandstop", "bandstop ({ω1, ω2} or {{ω, q}})", ""),
        ("DifferentiatorFilter", "Differentiator", "differentiator (iω for |ω| < ωc)", ""),
        ("HilbertFilter", "Hilbert", "Hilbert (-i sgn ω for |ω| < ωc)", "")):
    fn(_name, area=AREA, status="added", notes=_FILTER_NOTE.format(kind=_resp, norm=_norm))(_filter_impl(_name, _kind))


# ═══════════════════════════════════════════════════════════════ peaks ══
def _find_peaks(x: ListOf[Real], *rest: Expr) -> MatrixOf[Real]:
    fname = "FindPeaks"
    pos, opts = _opts(rest, {"InterpolationOrder", "Padding"}, fname)
    if not is_list(x):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list of real numbers, not {x!r}")
    items = list(x.args)
    if any(is_list(v) for v in items):
        raise NotImplementedError(f"{fname}: lists of {{x, y}} pairs and arrays are not supported")
    if len(pos) > 3:
        raise TypeError(f"{fname}: called with {len(pos) + 1} positional arguments; 1 to 4 are expected")
    sigma = pos[0] if pos else Symbol("Automatic")
    sharp = pos[1] if len(pos) > 1 else Symbol("Automatic")
    thr = pos[2] if len(pos) > 2 else Symbol("Automatic")
    auto = lambda v: isinstance(v, Symbol) and v.name == "Automatic"
    if auto(sigma) or not (K.is_int(sigma) or isinstance(sigma, float)) or sigma != 0:
        if isinstance(sigma, Symbol) and not auto(sigma):
            raise Unevaluated()
        raise NotImplementedError(f"{fname}: only the blur scale σ = 0 (plain local maxima) is supported; the "
                                  "automatic and positive-σ Gaussian-blur peak criteria are not documented")
    if not (auto(sharp) or ((K.is_int(sharp) or isinstance(sharp, float)) and sharp == 0)):
        raise NotImplementedError(f"{fname}: only sharpness s = 0 is supported (the sharpness measure for "
                                  "plateaus is not documented)")
    t = None
    if not auto(thr):
        if is_list(thr):
            raise NotImplementedError(f"{fname}: the threshold form {{t, ...}} is not supported")
        if isinstance(thr, (str, bool)):
            raise TypeError(f"{fname}: the threshold must be a real number, not {thr!r}")
        t = S(thr) if not isinstance(thr, (int, float)) else sp.sympify(thr)
        if _is_symbolic(t):
            raise Unevaluated()
        if not (t.is_number and t.is_real):
            raise TypeError(f"{fname}: the threshold must be a real number, not {thr!r}")
    io = opts.get("InterpolationOrder")
    if io is not None and not (auto(io) or (K.is_int(io) and io == 0)):
        raise NotImplementedError(f"{fname}: InterpolationOrder other than 0 (refined peak positions) is not "
                                  "supported")
    pad = opts.get("Padding", "Reflected")
    vals = []
    for v in items:
        s = _elem(v, fname)
        if not s.is_real:
            raise TypeError(f"{fname}: elements must be real numbers, not {v!r}")
        vals.append(s)
    n = len(vals)
    if n == 0:
        return L()
    if isinstance(pad, str) and pad == "Reflected":
        left = vals[1] if n > 1 else None
        right = vals[-2] if n > 1 else None
    elif isinstance(pad, (int, float)) and not isinstance(pad, bool) or (
            isinstance(pad, Expr) and not is_list(pad) and isinstance(S(pad), sp.Basic) and S(pad).is_real):
        left = right = S(pad) if not isinstance(pad, (int, float)) else sp.sympify(pad)
    else:
        raise NotImplementedError(f"{fname}: Padding {pad!r} is not supported (use \"Reflected\" or a number)")
    peaks = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        lv = vals[i - 1] if i > 0 else left
        rv = vals[j + 1] if j + 1 < n else right
        is_peak = (lv is None or lv < vals[i]) and (rv is None or rv < vals[i])
        if is_peak and t is not None and vals[i] < t:
            is_peak = False
        if is_peak:
            if i == 0 or j == n - 1:
                raise NotImplementedError(f"{fname}: a peak at the end of the list is not supported (Wolfram's "
                                          "position convention for boundary peaks is not documented)")
            if t is not None and vals[i] == t:
                raise NotImplementedError(f"{fname}: a peak value equal to the threshold is ambiguous")
            if t is None or vals[i] > t:
                peaks.append(L(sp.Rational(i + j + 2, 2) if (i + j) % 2 else (i + j) // 2 + 1, items[i]))
        i = j + 1
    return L(*[L(R(p.args[0]) if isinstance(p.args[0], sp.Basic) else p.args[0], p.args[1]) for p in peaks])


fn("FindPeaks", area=AREA, status="added",
   notes="FindPeaks[list, 0], FindPeaks[list, 0, 0, t]: interior local maxima {position, value} of a real list "
         "(a plateau gives its centre position, possibly a half-integer), Reflected or constant Padding, "
         "threshold t keeps peaks above t; automatic/positive blur σ, sharpness s > 0, InterpolationOrder and "
         "peaks at the list ends raise NotImplementedError (undocumented criteria/conventions). Check: exact "
         "comparison; reference (Wolfram documentation examples) in tests.")(_find_peaks)


# ═══════════════════════════════════════════════ symbolic transforms and convolutions ══
from sympy.core.function import AppliedUndef as _AppliedUndef  # noqa: E402

_US, _UB, _DD, _UT = (sp.Function(_n) for _n in ("UnitStep", "UnitBox", "DiscreteDelta", "UnitTriangle"))
_STEPS = (_US, _UB, _DD, _UT)


class _Fail(Exception):
    """Internal: the closed form could not be found (the call stays unevaluated)."""


def _sym_var(v, fname, what="variable"):
    if isinstance(v, Symbol):
        return v.name
    if is_list(v):
        raise NotImplementedError(f"{fname}: multivariate transforms are not supported")
    raise TypeError(f"{fname}: the {what} must be a symbol, not {v!r}")


def _sym_expr(e, fname):
    if isinstance(e, (str, bool)) or is_list(e):
        raise TypeError(f"{fname}: the function must be a mathematical expression, not {e!r}")
    try:
        return sp.sympify(S(e))
    except Exception:
        raise TypeError(f"{fname}: cannot interpret {e!r} as a mathematical expression") from None


def _is_step(f):
    return isinstance(f, sp.KroneckerDelta) or (isinstance(f, _AppliedUndef) and f.func in _STEPS)


def _lin(e, k):
    """(a, b) with e == a k + b, a rational and b free of k; None otherwise."""
    e = sp.expand(e)
    if not e.has(k):
        return sp.Integer(0), e
    a = e.coeff(k, 1)
    b = sp.expand(e - a * k)
    if b.has(k) or a.has(k) or not a.is_Rational:
        return None
    return a, b


def _piecewise_steps(e):
    """Replace UnitStep/UnitBox/DiscreteDelta/UnitTriangle/KroneckerDelta by Piecewise (for numerics)."""
    e = e.replace(_US, lambda x: sp.Piecewise((1, x >= 0), (0, True)))
    e = e.replace(_UB, lambda x: sp.Piecewise((1, sp.Abs(x) <= sp.Rational(1, 2)), (0, True)))
    e = e.replace(_DD, lambda x: sp.Piecewise((1, sp.Eq(x, 0)), (0, True)))
    e = e.replace(_UT, lambda x: sp.Piecewise((1 - sp.Abs(x), sp.Abs(x) <= 1), (0, True)))
    e = e.replace(sp.KroneckerDelta, lambda a, b: sp.Piecewise((1, sp.Eq(a, b)), (0, True)))
    return e


def _geometric(expr, k, lo, hi, conds):
    """Closed form of a geometric series Σ_{k=lo}^{hi} expr (ratio free of k); None if not geometric."""
    if not expr.has(k):
        return None
    q = sp.simplify(sp.powsimp(sp.expand_power_exp(expr.subs(k, k + 1) / expr), force=True))
    if q.has(k) or q == 1 or q == 0:
        return None
    if lo is -sp.oo and hi is sp.oo:
        return None
    if hi is sp.oo:
        conds.append(sp.Abs(q) < 1)
        return expr.subs(k, lo) / (1 - q)
    if lo is -sp.oo:
        conds.append(sp.Abs(1 / q) < 1)
        return expr.subs(k, hi) / (1 - 1 / q)
    return expr.subs(k, lo) * (1 - q ** (hi - lo + 1)) / (1 - q)


def _sum1(expr, k, lo, hi, conds):
    g = _geometric(expr, k, lo, hi, conds)
    if g is not None:
        return g
    r = sp.piecewise_fold(sp.summation(sp.powsimp(expr, force=True), (k, lo, hi)))
    if isinstance(r, sp.Piecewise):
        branches = [(v, c) for v, c in r.args if not v.has(sp.Sum)]
        if len(branches) == 1 and len(r.args) == 2 and r.args[-1][0].has(sp.Sum):
            r = branches[0][0]
            conds.append(branches[0][1])
    if r.has(sp.Sum) or r.has(sp.zoo) or r.has(sp.nan) or r.has(sp.oo) or r.has(-sp.oo) or r.has(sp.lerchphi):
        raise _Fail()
    return r


def _term_sum(term, k, conds):
    """Σ_k term over all integers, for a product term whose step factors are linear in k."""
    lowers, uppers, rest, delta = [], [], [], None
    for f in sp.Mul.make_args(term):
        base, ex = f.as_base_exp()
        if _is_step(base) and base.has(k):
            if not (ex.is_Integer and ex > 0):
                raise _Fail()
            if isinstance(base, sp.KroneckerDelta):
                arg, fn_ = base.args[1] - base.args[0], _DD
            else:
                arg, fn_ = base.args[0], base.func
            lab = _lin(arg, k)
            if lab is None or lab[0] == 0:
                raise _Fail()
            a, b = lab
            if fn_ is _US:
                (lowers if a > 0 else uppers).append(sp.ceiling(-b / a) if a > 0 else sp.floor(-b / a))
            elif fn_ is _UB:
                lo_, hi_ = (-sp.Rational(1, 2) - b) / a, (sp.Rational(1, 2) - b) / a
                if a < 0:
                    lo_, hi_ = hi_, lo_
                lowers.append(sp.ceiling(lo_))
                uppers.append(sp.floor(hi_))
            elif fn_ is _DD:
                if abs(a) != 1 or delta is not None:
                    raise _Fail()
                delta = -b / a
            else:
                raise _Fail()
        elif f.has(k) and any(f.has(s) for s in _STEPS + (sp.KroneckerDelta,)):
            raise _Fail()
        else:
            rest.append(f)
    rest = sp.Mul(*rest)
    if delta is not None:
        others = sp.Mul(*[f for f in sp.Mul.make_args(term) if not (_is_step(f.as_base_exp()[0])
                                                                    and f.as_base_exp()[0].has(k)
                                                                    and (isinstance(f.as_base_exp()[0],
                                                                                    sp.KroneckerDelta)
                                                                         or f.as_base_exp()[0].func is _DD))])
        return others.subs(k, delta)
    bsyms = set()
    for b in lowers + uppers:
        bsyms |= b.free_symbols
    if not bsyms:
        lo = max(lowers) if lowers else -sp.oo
        hi = min(uppers) if uppers else sp.oo
        if lo > hi:
            return sp.Integer(0)
        return _sum1(rest, k, lo, hi, conds)
    if len(bsyms) == 1:
        p = next(iter(bsyms))
        lin = [_lin(b, p) for b in lowers + uppers]
        if p.is_integer and all(l is not None and l[0].is_Integer and not l[1].free_symbols for l in lin):
            return _regions(rest, k, p, lowers, uppers, conds)
    if len(lowers) <= 1 and len(uppers) <= 1:
        lo = lowers[0] if lowers else -sp.oo
        hi = uppers[0] if uppers else sp.oo
        v = _sum1(rest, k, lo, hi, conds)
        if lowers and uppers:
            return sp.Piecewise((v, sp.Ge(hi - lo, 0)), (0, True))
        return v
    raise _Fail()


def _regions(rest, k, p, lowers, uppers, conds):
    """Piecewise in the integer parameter p of Σ_{k = max(lowers)}^{min(uppers)} rest."""
    bounds = lowers + uppers
    cuts = set()
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            (a1, b1), (a2, b2) = _lin(bounds[i], p), _lin(bounds[j], p)
            if a1 != a2:
                x = (b2 - b1) / (a1 - a2)
                cuts |= {int(sp.floor(x)), int(sp.ceiling(x))}
    cuts = sorted(cuts)
    if not cuts:
        regions = [(None, None, 0)]
    else:
        regions = [(None, cuts[0] - 1, cuts[0] - 1)]
        for i, c in enumerate(cuts):
            regions.append((c, c, c))
            if i + 1 < len(cuts) and cuts[i + 1] - c > 1:
                regions.append((c + 1, cuts[i + 1] - 1, c + 1))
        regions.append((cuts[-1] + 1, None, cuts[-1] + 1))
    cache = {}
    vals = []
    for a, b, rep in regions:
        lo = max(lowers, key=lambda e: e.subs(p, rep)) if lowers else -sp.oo
        hi = min(uppers, key=lambda e: e.subs(p, rep)) if uppers else sp.oo
        if lowers and uppers and (hi - lo).subs(p, rep) < 0:
            v = sp.Integer(0)
        else:
            if (lo, hi) not in cache:
                cache[(lo, hi)] = _sum1(rest, k, lo, hi, conds)
            v = cache[(lo, hi)]
        if vals and sp.simplify(vals[-1][2] - v) == 0 and vals[-1][1] is not None and a is not None \
                and vals[-1][1] + 1 == a:
            vals[-1] = (vals[-1][0], b, vals[-1][2])
        else:
            vals.append((a, b, v))
    pieces = []
    for a, b, v in vals:
        if v == 0:
            continue
        c = sp.true
        if a is not None and b is not None and a == b:
            c = sp.Eq(p, a)
        else:
            if a is not None:
                c = c & sp.Ge(p, a)
            if b is not None:
                c = c & sp.Le(p, b)
        pieces.append((sp.simplify(v), c))
    if not pieces:
        return sp.Integer(0)
    if len(pieces) == 1 and pieces[0][1] is sp.true:
        return pieces[0][0]
    return sp.Piecewise(*pieces, (0, True))


def _dsum(expr, k):
    """Σ_{k ∈ ℤ} expr in closed form, with the convergence conditions sympy reported; raises _Fail."""
    conds = []
    expr = expr.replace(lambda e: isinstance(e, (sp.sin, sp.cos, sp.sinh, sp.cosh)) and e.has(k),
                        lambda e: e.rewrite(sp.exp))
    expr = sp.expand(expr, power_base=False, power_exp=False, log=False, multinomial=False)
    total = sp.piecewise_fold(sp.Add(*[_term_sum(t, k, conds) for t in sp.Add.make_args(expr)]))
    if isinstance(total, sp.Piecewise):
        total = sp.Piecewise(*[(_tidy(v), c) for v, c in total.args])
    else:
        total = _tidy(total)
    return total, conds


def _has_unknown(e):
    return any(isinstance(f, _AppliedUndef) and f.func not in _STEPS for f in sp.preorder_traversal(e))


def _tidy(v):
    """Simplify a closed form; complex exponentials from trig rewriting go back to real form when possible."""
    if v.has(sp.I):
        w = sp.simplify(sp.expand_complex(v)) if not v.free_symbols - {s for s in v.free_symbols if s.is_real} else None
        if w is not None and not w.has(sp.I) and sp.count_ops(w) <= 2 * sp.count_ops(v) + 10:
            return sp.simplify(w)
    return sp.simplify(v)


def _series_value(term, lo=None, hi=None):
    """Σ term(k) for k in [lo, hi] (None = infinite side) by direct truncated summation; None if the
    truncation has not converged (|S_800 - S_400| too large) or a term is not finite."""
    def side(start, step):
        acc, s400 = mpmath.mpc(0), None
        for i in range(800):
            acc += term(start + step * i)
            if i == 399:
                s400 = acc
        return acc, s400
    with mpmath.workdps(25):
        total = mpmath.mpc(0)
        if lo is not None and hi is not None:
            for kk in range(int(lo), int(hi) + 1):
                total += term(kk)
            return complex(total)
        ok = True
        if hi is None:
            a, a4 = side(int(lo) if lo is not None else 0, 1)
            total += a
            ok = ok and abs(a - a4) <= 1e-10 * (1 + abs(a))
        if lo is None:
            b, b4 = side(int(hi) if hi is not None else -1, -1)
            total += b
            ok = ok and abs(b - b4) <= 1e-10 * (1 + abs(b))
        v = complex(total)
        if not ok or not math.isfinite(abs(v)):
            return None
        return v


def _nsum_check(fname, integrand, k, p, result, samples):
    """Numerical cross-check of a discrete convolution at integer values of p (inconclusive sums skipped)."""
    f = sp.lambdify([k, p], _piecewise_steps(integrand), modules="mpmath")
    rf = sp.lambdify([p], _piecewise_steps(result), modules="mpmath")
    for pv in samples:
        try:
            exact = complex(rf(pv))
            direct = _series_value(lambda kk: f(kk, pv))
        except (ZeroDivisionError, ValueError, TypeError, OverflowError):
            continue
        if direct is None or not math.isfinite(abs(exact)):
            continue
        if abs(direct - exact) > 1e-6 * (1 + abs(direct)):
            raise ArithmeticError(f"{fname}: result failed verification")


def _discrete_conv(fname, unilateral):
    def impl(f: Expr, g: Expr, n: Symbol, m: Symbol) -> Expr:
        nn, mm = _sym_var(n, fname), _sym_var(m, fname)
        fe, ge = _sym_expr(f, fname), _sym_expr(g, fname)
        ni, mi, k = sp.Symbol(nn, integer=True), sp.Symbol(mm, integer=True), sp.Dummy("k", integer=True)
        fe = fe.subs(sp.Symbol(nn), ni)
        ge = ge.subs(sp.Symbol(nn), ni)
        integrand = fe.subs(ni, k) * ge.subs(ni, mi - k).subs(sp.Symbol(mm), mi)
        integrand = integrand.subs(sp.Symbol(mm), mi)
        if unilateral:
            integrand = integrand * _US(k) * _US(mi - k)
        try:
            res, conds = _dsum(integrand, k)
        except _Fail:
            raise Unevaluated() from None
        if any(c is not sp.true for c in conds):
            raise Unevaluated()
        others = integrand.free_symbols - {k, mi}
        if not others and not _has_unknown(integrand):
            _nsum_check(fname, integrand, k, mi, res, (0, 1, 2, 3, -1, -4, 5)[:5 if unilateral else 7])
        res = res.subs(mi, sp.Symbol(mm)).subs(ni, sp.Symbol(nn))
        return R(res)
    return impl


fn("DiscreteConvolve", area=AREA, status="added",
   notes="DiscreteConvolve[f, g, n, m] = Σ_(k ∈ ℤ) f(k) g(m - k) in closed form: UnitStep/UnitBox/DiscreteDelta/"
         "KroneckerDelta factors linear in k become summation bounds, giving a Piecewise in m (Piecewise "
         "regions may be split differently from Wolfram's but agree at every integer m); stays unevaluated when "
         "no closed form is found; multivariate forms raise NotImplementedError. Check: numeric cross-check "
         "(the closed form is compared with the truncated defining series at sample integers m, raising "
         "ArithmeticError on mismatch); reference (Wolfram documentation examples) in tests.")(
    _discrete_conv("DiscreteConvolve", False))
fn("UnilateralDiscreteConvolve", area=AREA, status="added",
   notes="UnilateralDiscreteConvolve[f, g, n, m] = Σ_(k=0)^m f(k) g(m - k) in closed form, as a Piecewise in m "
         "that is 0 for m < 0 (Wolfram returns the m ≥ 0 formula); stays unevaluated when no closed form is "
         "found; multivariate forms raise NotImplementedError. Check: numeric cross-check (compared with the "
         "direct sum at sample m, ArithmeticError on mismatch); reference (Wolfram documentation examples) in "
         "tests.")(_discrete_conv("UnilateralDiscreteConvolve", True))


def _abs_interval(cond, z, fname):
    """Turn sympy convergence conditions in z into an interval (r1, r2) for |z|; None if not of that kind."""
    rho = sp.Symbol("rho", positive=True)
    c = cond.subs(z, rho)
    try:
        sol = sp.solve_univariate_inequality(c, rho, relational=False, domain=sp.Interval.open(0, sp.oo))
    except Exception:
        return None
    if not isinstance(sol, sp.Interval):
        return None
    return sol


def _unity(v):
    """Rewrite roots of unity (-1)^(p/q) in a rational function into radicals and re-factor."""
    if not v.has(sp.Pow) or not any(e.is_Pow and e.base == -1 and e.exp.is_Rational and not e.exp.is_Integer
                                    for e in sp.preorder_traversal(v)):
        return v
    w = v.replace(lambda e: e.is_Pow and e.base == -1 and e.exp.is_Rational, lambda e: sp.expand_complex(e))
    try:
        w = sp.factor(sp.cancel(sp.expand(sp.together(w))))
    except Exception:
        return v
    return w if sp.count_ops(w) <= sp.count_ops(v) + 10 else v


def _ztransform_core(fname, fe, ni, zs):
    """Bilateral Z transform Σ f(n) z^-n with its region of convergence as a |z| interval."""
    k = sp.Dummy("k", integer=True)
    term = fe.subs(ni, k)
    region = sp.Interval.open(0, sp.oo)
    total = sp.Integer(0)
    for side, factor in ((1, _US(k)), (-1, _US(-k - 1))):
        try:
            v, conds = _dsum(term * factor * zs ** (-k), k)
        except _Fail:
            raise Unevaluated() from None
        for c in conds:
            if c is sp.true:
                continue
            iv = _abs_interval(c, zs, fname)
            if iv is None:
                raise Unevaluated()
            region = region.intersect(iv)
        total += v
    if region is sp.S.EmptySet or region.measure == 0:
        raise Unevaluated()
    return _unity(sp.simplify(total)), region


def _z_check(fname, fe, ni, zs, val, region):
    lo = float(region.inf)
    hi = float(region.sup) if region.sup is not sp.oo else lo * 4 + 4
    r = (lo + hi) / 2 if region.sup is not sp.oo else max(2 * lo, lo + 1, 1.5)
    zv = r * cmath.exp(0.37j)
    k = sp.Symbol("k_", integer=True)
    f = sp.lambdify([k], _piecewise_steps(fe.subs(ni, k)), modules="mpmath")
    try:
        zz = mpmath.mpc(zv)
        direct = _series_value(lambda kk: f(kk) * zz ** (-kk))
        exact = complex(sp.N(val.subs(zs, zv), 20))
    except (ZeroDivisionError, ValueError, TypeError, OverflowError):
        return
    if direct is not None and math.isfinite(abs(exact)) and abs(direct - exact) > 1e-6 * (1 + abs(direct)):
        raise ArithmeticError(f"{fname}: result failed verification")


def _bilateral_z(f: Expr, n: Symbol, z: Expr) -> Expr:
    fname = "BilateralZTransform"
    nn = _sym_var(n, fname)
    fe = _sym_expr(f, fname)
    ni = sp.Symbol(nn, integer=True)
    fe = fe.subs(sp.Symbol(nn), ni)
    if isinstance(z, Symbol):
        zs = sp.Symbol(z.name)
        val, region = _ztransform_core(fname, fe, ni, zs)
        if not (fe.free_symbols - {ni}) and not _has_unknown(fe):
            _z_check(fname, fe, ni, zs, val, region)
        if region.inf == 0 and region.sup is sp.oo:
            return R(val)
        cond = []
        if region.inf != 0:
            cond.append(sp.Abs(zs) > region.inf)
        if region.sup is not sp.oo:
            cond.append(sp.Abs(zs) < region.sup)
        return Expr("ConditionalExpression", R(val), R(sp.And(*cond)))
    zv = _sym_expr(z, fname)
    if _is_symbolic(zv) or not zv.is_number:
        raise Unevaluated()
    zs = sp.Dummy("z")
    val, region = _ztransform_core(fname, fe, ni, zs)
    if not region.contains(sp.Abs(zv)):
        raise ValueError(f"{fname}: the series does not converge at z = {z!r}")
    out = val.subs(zs, zv)
    if any(isinstance(a, sp.Float) for a in sp.preorder_traversal(zv)):
        return R(sp.N(out))
    return R(sp.simplify(out))


fn("BilateralZTransform", area=AREA, status="added",
   notes="BilateralZTransform[f, n, z] = Σ_(n ∈ ℤ) f(n) z^-n in closed form, returned as ConditionalExpression"
         "[F, r1 < Abs[z] < r2] (no condition when it converges for all z ≠ 0); step/delta factors become "
         "summation bounds; a numeric z evaluates inside the region (outside raises ValueError); stays "
         "unevaluated when no closed form is found; multivariate forms raise NotImplementedError. Check: "
         "numeric cross-check (the truncated series at a point of the region, ArithmeticError on "
         "mismatch); reference (Wolfram documentation examples) in tests.")(_bilateral_z)


def _fparams(opts, fname, default=(1, 1)):
    v = opts.get("FourierParameters")
    if v is None:
        return sp.Integer(default[0]), sp.Integer(default[1])
    if not (is_list(v) and len(v.args) == 2):
        raise ValueError(f"{fname}: FourierParameters must be a list {{a, b}}")
    a, b = (_sym_expr(x, fname) for x in v.args)
    if not (a.is_number and b.is_number and a.is_real and b.is_real and b != 0):
        raise ValueError(f"{fname}: FourierParameters must be real numbers {{a, b}} with b ≠ 0")
    return a, b


def _fst(f: Expr, n: Symbol, w: Expr, *rest: Option) -> Expr:
    fname = "FourierSequenceTransform"
    _, opts = _opts(rest, {"FourierParameters"}, fname)
    if _:
        raise TypeError(f"{fname}: unexpected arguments {_!r}")
    a, b = _fparams(opts, fname)
    nn = _sym_var(n, fname)
    fe = _sym_expr(f, fname)
    ni = sp.Symbol(nn, integer=True)
    fe = fe.subs(sp.Symbol(nn), ni)
    zs = sp.Dummy("z")
    val, region = _ztransform_core(fname, fe, ni, zs)
    if not region.contains(1):
        raise Unevaluated()   # not absolutely summable: the transform is a distribution (DiracComb), not handled
    if not (fe.free_symbols - {ni}) and not _has_unknown(fe):
        _z_check(fname, fe, ni, zs, val, region)
    wv = sp.Symbol(w.name, real=True) if isinstance(w, Symbol) else _sym_expr(w, fname)
    out = sp.Abs(b / (2 * sp.pi)) ** ((1 - a) / 2) * val.subs(zs, sp.exp(sp.I * b * wv))
    out = sp.simplify(out)
    if isinstance(w, Symbol):
        out = out.subs(wv, sp.Symbol(w.name))
    elif any(isinstance(x, sp.Float) for x in sp.preorder_traversal(wv)):
        out = sp.N(out)
    return R(out)


fn("FourierSequenceTransform", area=AREA, status="added",
   notes="FourierSequenceTransform[f, n, ω] = |b/2π|^((1-a)/2) Σ_n f(n) e^(-i b n ω) (FourierParameters {a, b}, "
         "default {1, 1}) for absolutely summable sequences, via the bilateral Z transform at z = e^(i b ω); "
         "sequences whose Z transform region excludes |z| = 1 (transform is a DiracComb distribution) or with "
         "no closed form stay unevaluated. Check: numeric cross-check (series vs closed form, ArithmeticError on "
         "mismatch); reference (Wolfram documentation examples) in tests.")(_fst)


def _num_close(a, b):
    return math.isfinite(abs(b)) and abs(a - b) <= 1e-6 * (1 + abs(b))


def _strip_conds(r):
    """Drop sympy's convergence Piecewise around an integral, keeping the closed-form branch."""
    if isinstance(r, sp.Piecewise) and len(r.args) == 2 and r.args[-1][0].has(sp.Integral) \
            and not r.args[0][0].has(sp.Integral):
        return r.args[0][0]
    return r


def _cont_steps(e):
    """UnitStep/UnitBox/UnitTriangle -> Heaviside forms for integration (values at jumps are irrelevant)."""
    e = e.replace(_US, lambda x: sp.Heaviside(x))
    e = e.replace(_UB, lambda x: sp.Heaviside(x + sp.Rational(1, 2)) - sp.Heaviside(x - sp.Rational(1, 2)))
    e = e.replace(_UT, lambda x: (1 - sp.Abs(x)) * (sp.Heaviside(x + 1) - sp.Heaviside(x - 1)))
    return e


def _back_steps(e):
    return e.replace(sp.Heaviside, lambda *a: _US(a[0]))


def _check_quad(fname, g, lo, hi, exact, points=None):
    """Compare a closed form with mpmath.quad; inconclusive quadratures (large error estimate) are skipped."""
    try:
        with mpmath.workdps(20):
            iv = [lo] + sorted(points or []) + [hi]
            direct, err = mpmath.quad(g, iv, error=True)
            direct, err = complex(direct), float(err)
    except (ZeroDivisionError, ValueError, TypeError, OverflowError):
        return
    if not (math.isfinite(abs(direct)) and math.isfinite(abs(exact))) or err > 1e-9 * (1 + abs(direct)):
        return
    if not _num_close(direct, exact):
        raise ArithmeticError(f"{fname}: result failed verification")


def _breaks(e, x):
    """Real points where step/abs/piecewise arguments change (for splitting numeric quadrature)."""
    pts = set()
    for f in sp.preorder_traversal(e):
        if isinstance(f, sp.Piecewise):
            args = [c.lhs - c.rhs for _, c in f.args if isinstance(c, sp.core.relational.Relational)]
        elif _is_step(f) or isinstance(f, (sp.Heaviside, sp.Abs)):
            a = f.args[0]
            offs = (0,) if f.func not in (_UB, _UT) else ((-sp.Rational(1, 2), sp.Rational(1, 2))
                                                         if f.func is _UB else (-1, 0, 1))
            args = [a - o for o in offs]
        else:
            continue
        for a in args:
            if not a.has(x):
                continue
            try:
                for r in sp.solve(sp.Eq(a, 0), x):
                    if r.is_real and r.is_number:
                        pts.add(float(r))
            except Exception:
                continue
    return sorted(pts)


def _unilateral_conv(f: Expr, g: Expr, t: Symbol, y: Symbol) -> Expr:
    fname = "UnilateralConvolve"
    tn, yn = _sym_var(t, fname), _sym_var(y, fname)
    fe, ge = _sym_expr(f, fname), _sym_expr(g, fname)
    tau = sp.Dummy("tau", real=True)
    ys = sp.Symbol(yn, positive=True)
    T = sp.Symbol(tn)
    integrand = _cont_steps(fe.subs(T, tau) * ge.subs(T, ys - tau).subs(sp.Symbol(yn), ys))
    if integrand.has(sp.DiracDelta):
        raise NotImplementedError(f"{fname}: DiracDelta inputs are not supported")
    try:
        r = _strip_conds(sp.integrate(integrand, (tau, 0, ys)))
    except Exception:
        raise Unevaluated() from None
    if r.has(sp.meijerg) or r.has(sp.Integral):
        raise Unevaluated()
    r = sp.simplify(r)
    if not (integrand.free_symbols - {tau, ys}) and not _has_unknown(integrand):
        for yv in (0.7, 1.9, 3.3):
            ex = complex(sp.N(r.subs(ys, yv)))
            e_ = _back_steps(integrand.subs(ys, yv))
            pts = [p for p in _breaks(e_, tau) if 0 < p < yv]
            _check_quad(fname, sp.lambdify([tau], _piecewise_steps(e_), modules="mpmath"), 0, yv, ex, pts)
    return R(_back_steps(r).subs(ys, sp.Symbol(yn)))


fn("UnilateralConvolve", area=AREA, status="added",
   notes="UnilateralConvolve[f, g, t, y] = ∫_0^y f(t) g(y - t) dt for y > 0 in closed form (SymPy integrate; "
         "UnitStep/UnitBox/UnitTriangle handled through Heaviside forms, results may use UnitStep instead of "
         "Wolfram's Piecewise); DiracDelta inputs and multivariate forms raise NotImplementedError; stays "
         "unevaluated when no closed form is found. Check: numeric cross-check (mpmath.quad of the defining "
         "integral at sample y, ArithmeticError on mismatch); reference (Wolfram documentation examples) in "
         "tests.")(_unilateral_conv)


def _mellin_conv(f: Expr, g: Expr, x: Symbol, y: Symbol) -> Expr:
    fname = "MellinConvolve"
    xn, yn = _sym_var(x, fname), _sym_var(y, fname)
    fe, ge = _sym_expr(f, fname), _sym_expr(g, fname)
    u = sp.Dummy("u", positive=True)
    ys = sp.Symbol(yn, positive=True)
    X = sp.Symbol(xn)
    integrand = _cont_steps(fe.subs(X, u) * ge.subs(X, ys / u).subs(sp.Symbol(yn), ys) / u)
    if integrand.has(sp.DiracDelta):
        raise NotImplementedError(f"{fname}: DiracDelta inputs are not supported")
    try:
        r = _strip_conds(sp.integrate(integrand, (u, 0, sp.oo)))
    except Exception:
        raise Unevaluated() from None
    if r.has(sp.meijerg) or r.has(sp.Integral) or r.has(sp.oo) or r.has(sp.nan) or r.has(sp.zoo):
        raise Unevaluated()
    r = sp.simplify(r)
    if not (integrand.free_symbols - {u, ys}) and not _has_unknown(integrand):
        for yv in (0.3, 1.7):
            ex = complex(sp.N(r.subs(ys, yv)))
            e_ = integrand.subs(ys, yv)
            pts = [p for p in _breaks(_back_steps(e_), u) if p > 0]
            _check_quad(fname, sp.lambdify([u], _piecewise_steps(_back_steps(e_)), modules="mpmath"),
                        0, mpmath.inf, ex, pts)
    return R(_back_steps(r).subs(ys, sp.Symbol(yn)))


fn("MellinConvolve", area=AREA, status="added",
   notes="MellinConvolve[f, g, x, y] = ∫_0^∞ f(x) g(y/x) dx/x for y > 0 in closed form (SymPy integrate; step "
         "functions through Heaviside forms); DiracDelta inputs and multivariate forms raise "
         "NotImplementedError; stays unevaluated when no closed form is found. Check: numeric cross-check "
         "(mpmath.quad at sample y, ArithmeticError on mismatch); reference (Wolfram documentation examples) in "
         "tests.")(_mellin_conv)
