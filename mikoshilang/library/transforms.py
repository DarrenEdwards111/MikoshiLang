"""Integral transforms and discrete transforms.

Wolfram conventions for the continuous transforms:
  FourierTransform[f, t, w]        = 1/Sqrt[2 Pi] Integrate[f E^( I w t), {t, -oo, oo}]
  InverseFourierTransform[F, w, t] = 1/Sqrt[2 Pi] Integrate[F E^(-I w t), {w, -oo, oo}]
(legacy used SymPy's e^(-2 pi i k x) convention, giving different answers).

Discrete transforms keep the NumPy/SciPy conventions documented per function
(they are not Wolfram's Fourier[], which uses 1/Sqrt[n] and the opposite sign).
Exact input gives exact output; inexact input gives floats (no rounding;
legacy FFT rounded to 6 digits and InverseFFT dropped imaginary parts)."""
import cmath
import math

import sympy as sp

from ..expr import Expr, Symbol
from .core import Unevaluated, fn, error, record, S, R, L, rule, number, sym, variable, list_arg, has_float

AREA = "transforms"
MAX_EXACT = 64


def _seq(x, fname):
    items = list_arg(x, fname, "list", 1)
    vals = [number(v, fname, "list element", real=False) for v in items]
    return vals, any(has_float(v) for v in items)


def _dft(vals, inexact, sign):
    n = len(vals)
    if inexact:
        xs = [complex(sp.N(v)) for v in vals]
        return [sum(xs[j] * cmath.exp(sign * 2j * math.pi * j * k / n) for j in range(n)) for k in range(n)], True
    if n > MAX_EXACT:
        raise ValueError(f"exact DFT limited to {MAX_EXACT} points; use N[] input for longer lists")
    return [sp.Add(*(vals[j] * sp.exp(sign * 2 * sp.pi * sp.I * j * k / n) for j in range(n))) for k in range(n)], False


def _cplx_out(vals, inexact):
    if inexact:
        return L(*(R(complex(v)) for v in vals))
    return L(*(R(sp.simplify(sp.expand_complex(v))) for v in vals))


@fn("FFT", "DFT", "FourierDFT", "DiscreteFourierTransform", area=AREA, status="fixed",
    notes="X_k = Σ x_j e^(-2πi jk/n) (NumPy fft convention; Wolfram Fourier[] differs by 1/√n and sign); exact "
          "for exact input (n ≤ 64), complex floats otherwise; legacy FFT rounded results to 6 digits")
def _fft(x):
    vals, inexact = _seq(x, "FFT")
    out, inexact = _dft(vals, inexact, -1)
    return _cplx_out(out, inexact)


@fn("InverseFFT", area=AREA, status="fixed",
    notes="x_j = (1/n) Σ X_k e^(2πi jk/n) (NumPy ifft); legacy discarded imaginary parts and rounded")
def _ifft(x):
    vals, inexact = _seq(x, "InverseFFT")
    out, inexact = _dft(vals, inexact, 1)
    n = len(vals)
    return _cplx_out([v / n for v in out], inexact)


def _real_seq(x, fname):
    items = list_arg(x, fname, "list", 1)
    return [number(v, fname, "list element") for v in items], any(has_float(v) for v in items)


def _trig_tf(fname, kernel):
    def impl(x):
        vals, inexact = _real_seq(x, fname)
        n = len(vals)
        if not inexact and n > MAX_EXACT:
            inexact = True
        if inexact:
            xs = [float(v) for v in vals]
            return L(*(2 * sum(xs[j] * kernel(math, j, k, n) for j in range(n)) for k in range(n)))
        return L(*(R(sp.nsimplify(sp.simplify(2 * sp.Add(*(vals[j] * kernel(sp, j, k, n) for j in range(n))))))
                   for k in range(n)))
    return impl


fn("DCT", "DiscreteCosineTransform", area=AREA, status="verified",
   notes="unnormalised DCT-II, y_k = 2 Σ x_j cos(πk(2j+1)/(2n)) (SciPy dct default); exact for exact input")(
    _trig_tf("DCT", lambda m, j, k, n: m.cos(m.pi * k * (2 * j + 1) / (2 * n))))
fn("DST", "DiscreteSineTransform", area=AREA, status="verified",
   notes="unnormalised DST-II, y_k = 2 Σ x_j sin(π(k+1)(2j+1)/(2n)) (SciPy dst default); exact for exact input")(
    _trig_tf("DST", lambda m, j, k, n: m.sin(m.pi * (k + 1) * (2 * j + 1) / (2 * n))))


@fn("HilbertTransform", area=AREA, status="fixed",
    notes="discrete Hilbert transform H[x] (imaginary part of SciPy's analytic signal); floats; legacy returned "
          "the complex analytic signal x + iH[x] under this name")
def _hilbert(x):
    vals, _ = _real_seq(x, "HilbertTransform")
    n = len(vals)
    X, _ = _dft(vals, True, -1)
    h = [0.0] * n
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        for i in range(1, n // 2):
            h[i] = 2.0
    else:
        for i in range(1, (n + 1) // 2):
            h[i] = 2.0
    z, _ = _dft([X[k] * h[k] for k in range(n)], True, 1)
    return L(*(float((v / n).imag) for v in z))


@fn("Convolve", "Convolution", area=AREA, status="fixed",
    notes="full linear convolution (NumPy convolve 'full'), exact for exact input; legacy rounded to 10 digits")
def _conv(a, b):
    xa, fa = _seq(a, "Convolve")
    xb, fb = _seq(b, "Convolve")
    out = [sp.Integer(0)] * (len(xa) + len(xb) - 1)
    for i, u in enumerate(xa):
        for j, v in enumerate(xb):
            out[i + j] += u * v
    if fa or fb:
        return L(*(R(complex(sp.N(v))) for v in out))
    return L(*(R(sp.expand(v)) for v in out))


@fn("WaveletTransform", area=AREA, status="fixed",
    notes="WaveletTransform[list] or [list, \"Haar\"|\"db1\"]: full Haar decomposition {cA_n, cD_n, ..., cD_1} "
          "(PyWavelets wavedec ordering, orthonormal 1/√2 scaling), exact; length must be a power of 2; other "
          "wavelets raise (legacy needed PyWavelets, which is not a dependency)")
def _wavelet(x, w="Haar"):
    if w not in ("Haar", "haar", "db1"):
        raise NotImplementedError(f"WaveletTransform: only the Haar (db1) wavelet is supported, not {w!r}")
    vals, inexact = _real_seq(x, "WaveletTransform")
    n = len(vals)
    if n < 2 or n & (n - 1):
        raise ValueError("WaveletTransform: list length must be a power of 2 (≥ 2)")
    c = vals
    details = []
    s = 1 / sp.sqrt(2)
    while len(c) > 1:
        a = [(c[2 * i] + c[2 * i + 1]) * s for i in range(len(c) // 2)]
        d = [(c[2 * i] - c[2 * i + 1]) * s for i in range(len(c) // 2)]
        details.append(d)
        c = a
    parts = [c] + details[::-1]
    conv = (lambda v: float(v)) if inexact else (lambda v: R(sp.radsimp(sp.simplify(v))))
    return L(*(L(*(conv(v) for v in p)) for p in parts))


def _cont(fname, e, a, b):
    f = sym(e, fname)
    x, y = variable(a, fname), variable(b, fname)
    return f, x, y


@fn("FourierTransform", area=AREA, status="fixed",
    notes="Wolfram convention (1/√(2π)) ∫ f e^(iωt) dt; legacy used SymPy's e^(-2πikx) convention (wrong "
          "for Wolfram, e.g. FourierTransform[E^(-t^2/2), t, w] must be E^(-w^2/2)); unevaluated when SymPy "
          "cannot do the integral")
def _ft(e, t, w):
    f, tv, wv = _cont("FourierTransform", e, t, w)
    k = sp.Dummy("k", real=True)
    r = sp.fourier_transform(f, tv, k)
    if r.has(sp.FourierTransform):
        raise Unevaluated()
    return R(sp.simplify(r.subs(k, -wv / (2 * sp.pi)) / sp.sqrt(2 * sp.pi)))


@fn("InverseFourierTransform", area=AREA, status="fixed",
    notes="Wolfram convention (1/√(2π)) ∫ F e^(-iωt) dω; legacy used SymPy's convention")
def _ift(e, w, t):
    F, wv, tv = _cont("InverseFourierTransform", e, w, t)
    k = sp.Dummy("k", real=True)
    r = sp.fourier_transform(F, wv, k)
    if r.has(sp.FourierTransform):
        raise Unevaluated()
    return R(sp.simplify(r.subs(k, tv / (2 * sp.pi)) / sp.sqrt(2 * sp.pi)))


@fn("LaplaceTransform", area=AREA, status="verified",
    notes="∫_0^∞ f e^(-st) dt (same convention as Wolfram/SymPy), convergence conditions dropped; unevaluated "
          "when SymPy cannot transform")
def _lt(e, t, s):
    f, tv, sv = _cont("LaplaceTransform", e, t, s)
    r = sp.laplace_transform(f, tv, sv, noconds=True)
    if r.has(sp.LaplaceTransform):
        raise Unevaluated()
    return R(sp.simplify(r))


@fn("InverseLaplaceTransform", area=AREA, status="fixed",
    notes="inverse Laplace transform for t > 0 (Wolfram form); legacy kept SymPy's HeavisideTheta[t] factor")
def _ilt(e, s, t):
    F, sv, tv = _cont("InverseLaplaceTransform", e, s, t)
    tp = sp.Symbol(tv.name, positive=True)
    r = sp.inverse_laplace_transform(F, sv, tp)
    if r.has(sp.InverseLaplaceTransform):
        raise Unevaluated()
    return R(sp.simplify(r).subs(tp, tv))


@fn("ZTransform", area=AREA, status="fixed",
    notes="Wolfram ZTransform[f, n, z] = Σ_{n=0}^∞ f(n) z^(-n) via symbolic summation (convergence "
          "conditions dropped); unevaluated when the sum has no closed form; legacy always returned itself")
def _zt(e, n, z):
    f, nv, zv = _cont("ZTransform", e, n, z)
    k = sp.Symbol(nv.name, integer=True, nonnegative=True)
    r = sp.summation(f.subs(nv, k) * zv ** (-k), (k, 0, sp.oo))
    if isinstance(r, sp.Piecewise):
        r = r.args[0][0]
    if r.has(sp.Sum):
        raise Unevaluated()
    return R(sp.simplify(r))


error("InverseZTransform", AREA,
      "the inverse Z-transform is not supported (legacy returned the input unevaluated)")
