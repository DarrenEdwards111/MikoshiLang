"""Differential tests for coverage batch W23 (signal processing: Fourier/DCT/DST, list and integral transforms,
convolution, FIR filters, kernels and wavelets).

Reference values are written independently of the implementation:

* discrete transforms: the defining sums from the Wolfram documentation (Fourier, FourierDCT/FourierDST of types
  1-4, DiscreteHadamardTransform in sequency/natural/Paley order, DiscreteChirpZTransform, PeriodogramArray,
  CepstrumArray), evaluated here with cmath or NumPy's FFT (numpy.fft conventions: fft = Σ u e^(-2πi rs/n));
* DiscreteHilbertTransform: the analytic-signal construction (SciPy's scipy.signal.hilbert, re-derived with NumPy);
* FIR filters: the window method (ideal impulse responses sin(ωc m)/(πm), (ωc m cos(ωc m) - sin(ωc m))/(πm²),
  (1 - cos(ωc m))/(πm), Hamming window 25/46 + 21/46 cos 2πx) applied with "Fixed" padding;
* symbolic transforms: textbook closed forms (geometric series, Laplace pairs, Hankel pairs from DLMF 10.22 /
  Bateman tables), plus round trips;
* kernels: the Savitzky-Golay tables (Savitzky & Golay 1964, quadratic 5- and 7-point smoothing), the Bessel/
  Gaussian/Shen-Castan/Gabor formulas of the documentation, Daubechies D4/D6 filter tables (Daubechies 1992,
  Table 6.1, divided by √2 for Wolfram's Σ h = 1 normalisation).
"""
import cmath
import math

import mpmath
import numpy as np
import pytest

from _harness import Pred, Raises, check, ids, to_complex
from mikoshilang.expr import Expr


# ───────────────────────────────────────────────────────────── independent references ──
def _vals(g):
    """Nested MikoshiLang list -> nested Python lists of complex."""
    if isinstance(g, Expr) and g.head == "List":
        return [_vals(a) for a in g.args]
    return to_complex(g)


def _close(a, b, tol=1e-9):
    if isinstance(b, (list, tuple)):
        return isinstance(a, list) and len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    return not isinstance(a, list) and abs(complex(a) - complex(b)) <= tol * max(1.0, abs(complex(b)))


def approx(want, tol=1e-9, what="numeric reference"):
    want = np.asarray(want).tolist() if isinstance(want, np.ndarray) else want
    return Pred(lambda g: _close(_vals(g), want, tol), what)


def wl_fourier(u, a=0, b=1):
    n = len(u)
    return [n ** (-(1 - a) / 2) * sum(u[r] * cmath.exp(2j * math.pi * b * r * s / n) for r in range(n))
            for s in range(n)]


def dct(u, t):
    n = len(u)
    if t == 1:
        return [math.sqrt(2 / (n - 1)) * (u[0] / 2 + (-1) ** s * u[-1] / 2 +
                                           sum(u[r] * math.cos(math.pi * r * s / (n - 1)) for r in range(1, n - 1)))
                for s in range(n)]
    if t == 2:
        return [sum(u[r] * math.cos(math.pi * (r + 0.5) * s / n) for r in range(n)) / math.sqrt(n) for s in range(n)]
    if t == 3:
        return [(u[0] + 2 * sum(u[r] * math.cos(math.pi * r * (s + 0.5) / n) for r in range(1, n))) / math.sqrt(n)
                for s in range(n)]
    return [math.sqrt(2 / n) * sum(u[r] * math.cos(math.pi * (r + 0.5) * (s + 0.5) / n) for r in range(n))
            for s in range(n)]


def dst(u, t):
    n = len(u)
    if t == 1:
        return [math.sqrt(2 / (n + 1)) * sum(u[r] * math.sin(math.pi * (r + 1) * (s + 1) / (n + 1)) for r in range(n))
                for s in range(n)]
    if t == 2:
        return [sum(u[r] * math.sin(math.pi * (r + 0.5) * (s + 1) / n) for r in range(n)) / math.sqrt(n)
                for s in range(n)]
    if t == 3:
        return [((-1) ** s * u[-1] + 2 * sum(u[r] * math.sin(math.pi * (r + 1) * (s + 0.5) / n)
                                              for r in range(n - 1))) / math.sqrt(n) for s in range(n)]
    return [math.sqrt(2 / n) * sum(u[r] * math.sin(math.pi * (r + 0.5) * (s + 0.5) / n) for r in range(n))
            for s in range(n)]


def hilbert_ref(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    return np.imag(np.fft.ifft(np.fft.fft(x) * h))


def hamming(x):
    return 25 / 46 + 21 / 46 * math.cos(2 * math.pi * x)


def fir_ref(data, resp, n, window=hamming, normalize=False):
    """Window-method FIR filter: kernel h(m) w(x_k), m = k - (n-1)/2, Fixed padding, centred convolution."""
    ms = [k - (n - 1) / 2 for k in range(n)]
    xs = [-0.5 + k / (n - 1) for k in range(n)] if n > 1 else [0.0]
    ker = [resp(m) * (window(x) if window else 1.0) for m, x in zip(ms, xs)]
    if normalize:
        t = sum(ker)
        ker = [k / t for k in ker]
    c = (n - 1) // 2
    pad = [data[0]] * n + list(data) + [data[-1]] * n
    return [sum(ker[k] * pad[n + s - (k - c)] for k in range(n)) for s in range(len(data))]


def lp(wc):
    return lambda m: wc / math.pi if m == 0 else math.sin(wc * m) / (math.pi * m)


def hp(wc):
    return lambda m: 1 - wc / math.pi if m == 0 else -math.sin(wc * m) / (math.pi * m)


def bp(w1, w2):
    return lambda m: (w2 - w1) / math.pi if m == 0 else (math.sin(w2 * m) - math.sin(w1 * m)) / (math.pi * m)


def bs(w1, w2):
    return lambda m: 1 - (w2 - w1) / math.pi if m == 0 else -(math.sin(w2 * m) - math.sin(w1 * m)) / (math.pi * m)


def diff(wc):
    return lambda m: 0.0 if m == 0 else (wc * m * math.cos(wc * m) - math.sin(wc * m)) / (math.pi * m * m)


def hilb(wc):
    return lambda m: 0.0 if m == 0 else (1 - math.cos(wc * m)) / (math.pi * m)


def cepstrum(x, real=False):
    F = np.fft.fft(np.asarray(x, dtype=float))
    if real:
        return np.real(np.fft.ifft(np.log(np.abs(F))))
    return np.abs(np.fft.ifft(np.log(np.abs(F) ** 2))) ** 2


def kernel1d(vals):
    t = sum(vals)
    return [v / t for v in vals]


def outer(a, b):
    return [[x * y for y in b] for x in a]


SHEN_C = 102798103 / 66535189
D4 = [(1 + math.sqrt(3)) / 8, (3 + math.sqrt(3)) / 8, (3 - math.sqrt(3)) / 8, (1 - math.sqrt(3)) / 8]
# Daubechies (1992), Table 6.1, N = 3 (Σ h = √2), rescaled to Σ h = 1
D6 = [x / math.sqrt(2) for x in (0.332670552950, 0.806891509311, 0.459877502118, -0.135011020010,
                                 -0.085441273882, 0.035226291882)]


def _ortho(g, tol=1e-9):
    h = [to_complex(p.args[1]).real for p in g.args]
    n = len(h)
    for m in range(0, n // 2 + 1):
        s = sum(h[k] * h[k + 2 * m] for k in range(n - 2 * m))
        if abs(s - (0.5 if m == 0 else 0.0)) > tol:
            return False
    return abs(sum(h) - 1) < tol


def gabor_ref(r, k, phi=0.0, sigma=None):
    ks = k if isinstance(k, (list, tuple)) else [k]
    d = len(ks)
    sigma = r / 2 if sigma is None else sigma
    idx = np.indices([2 * r + 1] * d).reshape(d, -1).T - r
    env = np.exp(-(idx ** 2).sum(axis=1) / (2 * sigma ** 2))
    vals = env * np.cos(idx @ np.array(ks, dtype=float) - phi) / env.sum()
    return vals.reshape([2 * r + 1] * d)


def _near(v, tol=1e-12, what="numeric identity"):
    return Pred(lambda g: abs(to_complex(g) - v) < tol, what)


U8 = [1, 1, 2, 2, 1, 1, 0, 0]
HAMMING4 = np.array([hamming(-0.5 + k / 3) for k in range(4)])
HAMMING4 = HAMMING4 / np.sqrt(np.mean(HAMMING4 ** 2))

CASES = [
    # ── Fourier / InverseFourier / FourierParameters
    ("Fourier[{1, 1, 2, 2, 1, 1, 0, 0}]", approx(wl_fourier(U8))),                   # definition, {a, b} = {0, 1}
    ("Fourier[{1, 2, 3}, FourierParameters -> {1, -1}]", approx(np.fft.fft([1, 2, 3]))),   # numpy fft convention
    ("Fourier[{1, 2, 3, 4, 5}, FourierParameters -> {-1, 2}]", approx(wl_fourier([1, 2, 3, 4, 5], -1, 2))),
    ("Fourier[{{1, 2}, {3, 4}}]", approx([[5, -1], [-2, 0]])),                         # 2-D: (1/2) Σ (±1) u
    ("Fourier[{a, b}]", "Fourier[{a, b}]"),
    ('Fourier["x"]', Raises(TypeError)),
    ("Fourier[{1, 2, 3, 4}, FourierParameters -> {0, 2}]", Raises(ValueError)),      # b not coprime to n
    ("Fourier[{1, 2}, {1}]", Raises(NotImplementedError)),
    ("InverseFourier[{1, 2, 3, 4}]", approx(np.fft.fft([1, 2, 3, 4]) / 2)),          # n^(-1/2) Σ v e^(-2πi rs/n)
    ("InverseFourier[{1, 2, 3}, FourierParameters -> {1, -1}]", approx(np.fft.ifft([1, 2, 3]))),
    ("InverseFourier[Fourier[{1., -2., 3.5, 4., 0.25}]]", approx([1, -2, 3.5, 4, 0.25])),   # Check: round trip
    ("InverseFourier[{a, b}]", "InverseFourier[{a, b}]"),
    ('InverseFourier["x"]', Raises(TypeError)),
    ("FourierParameters", "FourierParameters"),
    ("Fourier[{0, 1, 0, 0}, FourierParameters -> {1, 1}]", approx([1, 1j, -1, -1j])),   # e^(2πi s/4)
    # ── FourierDCT / FourierDST
    ("FourierDCT[{1, 2, 3, 4}]", approx(dct([1, 2, 3, 4], 2))),
    ("FourierDCT[{1, 2, 3, 4}, 1]", approx(dct([1, 2, 3, 4], 1))),
    ("FourierDCT[{1, 2, 3, 4}, 3]", approx(dct([1, 2, 3, 4], 3))),
    ('FourierDCT[{1, 5, -2, 4}, "IV"]', approx(dct([1, 5, -2, 4], 4))),
    ("FourierDCT[FourierDCT[{1., 2., 3., 4.}], 3]", approx([1, 2, 3, 4])),             # Check: II/III inverse pair
    ("FourierDCT[FourierDCT[{3., 1., 4., 1., 5.}, 1], 1]", approx([3, 1, 4, 1, 5])),    # Check: I self-inverse
    ("FourierDCT[{a, b}]", "FourierDCT[{a, b}]"),
    ("FourierDCT[{1, 2}, 5]", Raises(ValueError)),
    ('FourierDCT["x"]', Raises(TypeError)),
    ("FourierDST[{1, 2, 3, 4}]", approx(dst([1, 2, 3, 4], 2))),
    ("FourierDST[{1, 2, 3, 4}, 1]", approx(dst([1, 2, 3, 4], 1))),
    ("FourierDST[{1, 2, 3, 4}, 3]", approx(dst([1, 2, 3, 4], 3))),
    ("FourierDST[{2, -1, 7}, 4]", approx(dst([2, -1, 7], 4))),
    ("FourierDST[FourierDST[{1., 2., 3., 4.}, 2], 3]", approx([1, 2, 3, 4])),          # Check: II/III inverse pair
    ("FourierDST[{a, b}]", "FourierDST[{a, b}]"),
    ("FourierDST[{1, 2}, 7]", Raises(ValueError)),
    ('FourierDST["x"]', Raises(TypeError)),
    # ── FourierDCTMatrix / FourierDSTMatrix: data . F == FourierDCT[data]
    ("FourierDCTMatrix[2]", "{{Sqrt[2]/2, 1/2}, {Sqrt[2]/2, -1/2}}"),                  # cos(π(r-1/2)s/2)/√2
    ("FourierDCTMatrix[3, 1]", "{{1/2, 1/2, 1/2}, {1, 0, -1}, {1/2, -1/2, 1/2}}"),
    ("N[{1, 2, 3, 4} . FourierDCTMatrix[4]]", approx(dct([1, 2, 3, 4], 2))),
    ("Map[Simplify, FourierDCTMatrix[4, 2] . FourierDCTMatrix[4, 3], {2}] == IdentityMatrix[4]", "True"),   # Check
    ("FourierDCTMatrix[2, WorkingPrecision -> MachinePrecision]", approx([[0.5 ** 0.5, 0.5], [0.5 ** 0.5, -0.5]])),
    ("FourierDCTMatrix[n]", "FourierDCTMatrix[n]"),
    ("FourierDCTMatrix[-1]", Raises(ValueError)),
    ('FourierDCTMatrix[2, TargetStructure -> "Sparse"]', Raises(NotImplementedError)),
    ("FourierDSTMatrix[2]", "{{1/2, Sqrt[2]/2}, {1/2, -Sqrt[2]/2}}"),                  # sin(π(r-1/2)s/2)/√2
    ("FourierDSTMatrix[3, 1]", "{{1/2, Sqrt[2]/2, 1/2}, {Sqrt[2]/2, 0, -Sqrt[2]/2}, {1/2, -Sqrt[2]/2, 1/2}}"),
    ("N[{1, 2, 3, 4} . FourierDSTMatrix[4, 4]]", approx(dst([1, 2, 3, 4], 4))),
    ("Map[Simplify, FourierDSTMatrix[3, 2] . FourierDSTMatrix[3, 3], {2}] == IdentityMatrix[3]", "True"),   # Check
    ("FourierDSTMatrix[n]", "FourierDSTMatrix[n]"),
    ("FourierDSTMatrix[0]", Raises(ValueError)),
    ('FourierDSTMatrix[2, TargetStructure -> "Sparse"]', Raises(NotImplementedError)),
    # ── DiscreteHadamardTransform: (1/√n) Walsh functions; sequency, natural (Sylvester) and Paley orders
    ("DiscreteHadamardTransform[{1, 2, 3, 4}]", "{5, -2, 0, -1}"),
    ('DiscreteHadamardTransform[{1, 2, 3, 4}, Method -> "BitComplement"]', "{5, -1, -2, 0}"),
    ('DiscreteHadamardTransform[{1, 2, 3, 4}, Method -> "GrayCode"]', "{5, -2, -1, 0}"),
    ("DiscreteHadamardTransform[{1, 2, 3}]", "{3, 0, -2, 1}"),                          # zero-padded to 4
    ("DiscreteHadamardTransform[DiscreteHadamardTransform[{3, 1, 4, 1, 5, 9, 2, 6}]]",
     "{3, 1, 4, 1, 5, 9, 2, 6}"),                                                       # Check: self-inverse
    ("DiscreteHadamardTransform[{a, b}]", "DiscreteHadamardTransform[{a, b}]"),
    ('DiscreteHadamardTransform[{1, 2}, Method -> "Foo"]', Raises(ValueError)),
    ('DiscreteHadamardTransform["x"]', Raises(TypeError)),
    # ── DiscreteHilbertTransform
    ("DiscreteHilbertTransform[{1, 2, 3, 4}]", approx(hilbert_ref([1, 2, 3, 4]))),
    ("DiscreteHilbertTransform[{0.5, -1., 2., 0., 3.}]", approx(hilbert_ref([0.5, -1, 2, 0, 3]))),
    ("DiscreteHilbertTransform[N[Sin[2 Pi Range[0, 15]/16]]]",
     approx(-np.cos(2 * np.pi * np.arange(16) / 16), 1e-9)),                            # Check: H[sin] = -cos
    ("DiscreteHilbertTransform[{a, b}]", "DiscreteHilbertTransform[{a, b}]"),
    ("DiscreteHilbertTransform[{1, I}]", Raises(ValueError)),
    ('DiscreteHilbertTransform["x"]', Raises(TypeError)),
    # ── DiscreteChirpZTransform: Σ u_r (a w^k)^-(r-1)
    ("DiscreteChirpZTransform[{1, 2, 3}]", approx(np.fft.fft([1, 2, 3]))),
    ("DiscreteChirpZTransform[{1, 2, 3}, 2, 1/2, 1]", approx([6, 17])),                 # z = 1, 1/2
    ("DiscreteChirpZTransform[{1, 2, 3}, 3, Exp[-2 Pi I/3]]", approx(np.fft.ifft([1, 2, 3]) * 3)),   # complex w
    ("DiscreteChirpZTransform[{1, 0, 2}, 2, 2, 3]",
     approx([sum(u * (3 * 2 ** k) ** -r for r, u in enumerate([1, 0, 2])) for k in range(2)])),
    ("Chop[DiscreteChirpZTransform[{1., 4., 2., 8.}] - Fourier[{1., 4., 2., 8.}, FourierParameters -> {1, -1}]]",
     "{0, 0, 0, 0}"),                                                                   # Check: Fourier identity
    ("DiscreteChirpZTransform[{a, b}]", "DiscreteChirpZTransform[{a, b}]"),
    ('DiscreteChirpZTransform["x"]', Raises(TypeError)),
    # ── ListZTransform / ListFourierSequenceTransform
    ("ListZTransform[{1, 2, 3}, z]", "1 + 2/z + 3/z^2"),
    ("ListZTransform[{a, b}, z, 2]", "a/z^2 + b/z^3"),
    ("ListZTransform[{1, 2}, 2]", "2"),
    ("ListZTransform[{1, 2, 3}, 1/2]", "17"),                                          # Check: = chirp-Z at z = 1/2
    ('ListZTransform["x", z]', Raises(TypeError)),
    ("ListFourierSequenceTransform[{1, 1, 1}/3, w]", "(1 + E^(-I w) + E^(-2 I w))/3"),  # Wolfram doc example
    ("ListFourierSequenceTransform[{1, 0, -1}, w, -1]", "E^(I w) - E^(-I w)"),          # centred (doc example)
    ("ListFourierSequenceTransform[{1, 1, 1}/3, f, FourierParameters -> {1, 2 Pi}]",
     "(1 + E^(-2 Pi I f) + E^(-4 Pi I f))/3"),
    ("ListFourierSequenceTransform[{1, 2, 3}, w] - ListZTransform[{1, 2, 3}, E^(I w)]", "0"),   # Check (doc)
    ('ListFourierSequenceTransform["x", w]', Raises(TypeError)),
    # ── PeriodogramArray / CepstrumArray / CepstrogramArray
    ("PeriodogramArray[{1, 2, 3, 4}]", approx(np.abs(np.fft.fft([1, 2, 3, 4])) ** 2 / 4)),
    ("PeriodogramArray[{1, 2, 3, 4, 5, 6}, 3]",
     approx((np.abs(np.fft.fft([1, 2, 3])) ** 2 + np.abs(np.fft.fft([4, 5, 6])) ** 2) / 6)),   # mean of segments
    ("PeriodogramArray[{1, 2, 3, 4}, 2, 2, {1, 1}]",
     approx((np.abs(np.fft.fft([1, 2])) ** 2 + np.abs(np.fft.fft([3, 4])) ** 2) / 4)),
    ("PeriodogramArray[{1, 2, 3, 4}, 4, 4, HammingWindow]",
     approx(np.abs(np.fft.fft(np.array([1, 2, 3, 4]) * HAMMING4)) ** 2 / 4)),           # unit-RMS window
    ("PeriodogramArray[{a, b}]", "PeriodogramArray[{a, b}]"),
    ('PeriodogramArray["x"]', Raises(TypeError)),
    ("CepstrumArray[{1, 2, 3, 4}]", approx(cepstrum([1, 2, 3, 4]))),
    ('CepstrumArray[{1, 2, 3, 4}, "Real"]', approx(cepstrum([1, 2, 3, 4], True))),
    ("CepstrumArray[{2., -1., 0.5, 3., 1.}]", approx(cepstrum([2, -1, 0.5, 3, 1]))),
    ("CepstrumArray[{a, b}]", "CepstrumArray[{a, b}]"),
    ('CepstrumArray[{1, 2}, "Foo"]', Raises(ValueError)),
    ('CepstrumArray["x"]', Raises(TypeError)),
    ("CepstrogramArray[Range[16], 4]", approx([cepstrum(list(range(1 + i, 5 + i))).tolist() for i in range(13)])),
    ("CepstrogramArray[Range[16], 4, 4]", approx([cepstrum(list(range(1 + i, 5 + i))).tolist() for i in (0, 4, 8, 12)])),
    ("CepstrogramArray[Range[12], 6, 3] == (CepstrumArray /@ Partition[Range[12], 6, 3])", "True"),  # Check
    ("CepstrogramArray[{a, b}]", "CepstrogramArray[{a, b}]"),
    ("CepstrogramArray[Range[16], 4, 2, HammingWindow]", Raises(NotImplementedError)),
    ('CepstrogramArray["x"]', Raises(TypeError)),
    # ── ListConvolve / ListCorrelate (Wolfram documentation examples)
    ("ListConvolve[{x, y}, {a, b, c}]", "{b x + a y, c x + b y}"),
    ("ListConvolve[{1, 1}, {a, b, c}, 1]", "{a + c, a + b, b + c}"),
    ("ListConvolve[{x, y}, {a, b, c}, {1, -1}, 0]", "{a x, b x + a y, c x + b y, c y}"),
    ("ListConvolve[{{1, 1}, {1, 1}}, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]", "{{12, 16}, {24, 28}}"),
    ("ListConvolve[{1, 2}, 3]", Raises(ValueError)),
    ('ListConvolve["x", {1, 2}]', Raises(TypeError)),
    ("ListCorrelate[{x, y}, {a, b, c}]", "{a x + b y, b x + c y}"),
    ("ListCorrelate[{1, 2}, {1, 2, 3}, 1]", "{5, 8, 5}"),
    ("ListCorrelate[{1, 2, 3}, {4, 5, 6, 7}] == ListConvolve[{3, 2, 1}, {4, 5, 6, 7}]", "True"),   # Check
    ("ListCorrelate[{{1, 0}, {0, -1}}, {{1, 2, 3}, {4, 5, 6}}]", "{{-4, -4}}"),
    ("ListCorrelate[{1, 2}, 3]", Raises(ValueError)),
    ('ListCorrelate["x", {1, 2}]', Raises(TypeError)),
    # ── MovingAverage / MovingMedian / ExponentialMovingAverage / MovingMap
    ("MovingAverage[{a, b, c, d}, 2]", "{(a + b)/2, (b + c)/2, (c + d)/2}"),
    ("MovingAverage[{1, 2, 3, 4}, {1, 2}]", "{5/3, 8/3, 11/3}"),
    ("MovingAverage[{{1, 2}, {3, 4}, {5, 6}}, 2]", "{{2, 3}, {4, 5}}"),
    ("MovingAverage[{1, 2}, 3]", Raises(ValueError)),
    ('MovingAverage["x", 2]', Raises(TypeError)),
    ("MovingMedian[{1, 5, 2, 8, 3}, 3]", "{2, 5, 3}"),
    ("MovingMedian[{1, 5, 2, 8}, 2]", "{3, 7/2, 5}"),
    ("MovingMedian[{{1, 9}, {5, 2}, {3, 4}}, 3]", "{{3, 4}}"),
    ("MovingMedian[{a, b, c}, 2]", "MovingMedian[{a, b, c}, 2]"),
    ("MovingMedian[{1, 2}, 3]", Raises(ValueError)),
    ('MovingMedian["x", 2]', Raises(TypeError)),
    ("ExponentialMovingAverage[{1, 2, 3}, 1/2]", "{1, 3/2, 9/4}"),
    ("ExponentialMovingAverage[{a, b, c}, x]",
     "{a, a + x (b - a), a + x (b - a) + x (c - a - x (b - a))}"),
    ("ExponentialMovingAverage[{0, 10, 10, 10}, 1/10]", "{0, 1, 19/10, 271/100}"),
    ("ExponentialMovingAverage[{1, 2}, 2]", Raises(ValueError)),
    ('ExponentialMovingAverage["x", 1/2]', Raises(TypeError)),
    ("MovingMap[Mean, {1, 2, 3, 4, 5}, 2]", "{2, 3, 4}"),
    ("MovingMap[f, {a, b, c, d}, 1]", "{f[{a, b}], f[{b, c}], f[{c, d}]}"),
    ('MovingMap[Total, {1, 2, 3, 4}, Quantity[2, "Events"]]', "{3, 5, 7}"),
    ("MovingMap[Total, {{1, 1}, {2, 2}, {4, 3}}, 1]", "{{2, 3}, {4, 3}}"),
    ("MovingMap[f, {1, 2, 3}, 5]", Raises(ValueError)),
    ('MovingMap[f, "x", 1]', Raises(TypeError)),
    ('MovingMap[f, {1, 2, 3}, 1, "Fixed"]', Raises(NotImplementedError)),
    # ── FIR filters (window method, Fixed padding)
    ("LowpassFilter[{1., 2., 3., 4., 5.}, 0.5, 3]", approx(fir_ref([1, 2, 3, 4, 5], lp(0.5), 3, normalize=True))),
    ("LowpassFilter[{1., 3., 2., 5., 4., 6.}, 1.2, 5, None]",
     approx(fir_ref([1, 3, 2, 5, 4, 6], lp(1.2), 5, None, normalize=True))),
    ("N[LowpassFilter[{1, 2, 3, 4, 5}, 1/2, 3]]", approx(fir_ref([1, 2, 3, 4, 5], lp(0.5), 3, normalize=True))),
    ("LowpassFilter[{1, 2, 3}, Pi]", "{1, 2, 3}"),                                    # all-pass
    ("LowpassFilter[{a, b, c}, 1/2, 3]", "LowpassFilter[{a, b, c}, 1/2, 3]"),
    ("LowpassFilter[{1, 2, 3}, 1/2]", Raises(NotImplementedError)),
    ('LowpassFilter[{1, 2, 3}, Quantity[1, "Hertz"], 3]', Raises(NotImplementedError)),
    ('LowpassFilter["a", 1, 3]', Raises(TypeError)),
    ("HighpassFilter[{1., 2., 3., 4., 5.}, 0.5, 3]", approx(fir_ref([1, 2, 3, 4, 5], hp(0.5), 3))),
    ("HighpassFilter[{1., -1., 2., 0., 4., 1.}, 2., 5, None]", approx(fir_ref([1, -1, 2, 0, 4, 1], hp(2.), 5, None))),
    ("N[HighpassFilter[{1, 2, 3, 4, 5}, 1/2, 3, None]]", approx(fir_ref([1, 2, 3, 4, 5], hp(0.5), 3, None))),
    ('HighpassFilter["a", 1, 3]', Raises(TypeError)),
    ("BandpassFilter[{1., 2., 3., 4., 5.}, {0.5, 1.}, 3]", approx(fir_ref([1, 2, 3, 4, 5], bp(0.5, 1), 3))),
    ("BandpassFilter[{2., 0., 1., 3., 5., 1., 0.}, {0.4, 2.1}, 5, None]",
     approx(fir_ref([2, 0, 1, 3, 5, 1, 0], bp(0.4, 2.1), 5, None))),
    ("N[BandpassFilter[{1, 2, 3, 4, 5}, {1/2, 1}, 3, None]]", approx(fir_ref([1, 2, 3, 4, 5], bp(0.5, 1), 3, None))),
    ('BandpassFilter["a", {1, 2}, 3]', Raises(TypeError)),
    ("BandstopFilter[{1., 2., 3., 4., 5.}, {0.5, 1.}, 3]", approx(fir_ref([1, 2, 3, 4, 5], bs(0.5, 1), 3))),
    ("BandstopFilter[{2., 0., 1., 3., 5., 1., 0.}, {0.4, 2.1}, 5, None]",
     approx(fir_ref([2, 0, 1, 3, 5, 1, 0], bs(0.4, 2.1), 5, None))),
    ("N[BandstopFilter[{1, 2, 3, 4, 5}, {1/2, 1}, 3, None]]", approx(fir_ref([1, 2, 3, 4, 5], bs(0.5, 1), 3, None))),
    ('BandstopFilter["a", {1, 2}, 3]', Raises(TypeError)),
    ("DifferentiatorFilter[{1, 2, 3, 4, 5}, Pi, 3, None]", "{1, 2, 2, 2, 1}"),           # h(m) = (-1)^m/m
    ("DifferentiatorFilter[{1., 4., 9., 16., 25.}, 2., 5]", approx(fir_ref([1, 4, 9, 16, 25], diff(2.), 5))),
    ("N[DifferentiatorFilter[{0, 1, 0, -1, 0}, 1, 3, None]]", approx(fir_ref([0, 1, 0, -1, 0], diff(1.), 3, None))),
    ('DifferentiatorFilter["a", 1, 3]', Raises(TypeError)),
    ("HilbertFilter[{1, 2, 3, 4, 5}, Pi, 3, None]", "{-2/Pi, -4/Pi, -4/Pi, -4/Pi, -2/Pi}"),   # (1 - cos πm)/(πm)
    ("HilbertFilter[{1., 0., -1., 0., 1., 0.}, 2.5, 5]", approx(fir_ref([1, 0, -1, 0, 1, 0], hilb(2.5), 5))),
    ("N[HilbertFilter[{0, 1, 2, 1, 0}, 1, 3, None]]", approx(fir_ref([0, 1, 2, 1, 0], hilb(1.), 3, None))),
    ('HilbertFilter["a", 1, 3]', Raises(TypeError)),
    # ── FindPeaks (σ = 0: plain local maxima)
    ("FindPeaks[{1, 3, 2, 5, 4}, 0]", "{{2, 3}, {4, 5}}"),
    ("FindPeaks[{1, 3, 3, 2}, 0]", "{{5/2, 3}}"),                                       # plateau centre
    ("FindPeaks[{1, 3, 2, 5, 4}, 0, 0, 4]", "{{4, 5}}"),                                # threshold
    ("FindPeaks[{0, 2, 0, 2, 0, 2, 0}, 0]", "{{2, 2}, {4, 2}, {6, 2}}"),
    ("FindPeaks[{a, b}, 0]", "FindPeaks[{a, b}, 0]"),
    ("FindPeaks[{1, 3, 2, 5, 4}, 1]", Raises(NotImplementedError)),
    ('FindPeaks["a"]', Raises(TypeError)),
    # ── DiscreteConvolve / UnilateralDiscreteConvolve (checked at integer points)
    ("Table[DiscreteConvolve[UnitStep[n], UnitStep[n], n, m] /. m -> k, {k, -2, 3}]", "{0, 0, 1, 2, 3, 4}"),
    # regression: closed form must not depend on SymPy cache state (UndefinedFunction identity)
    ("ClearSystemCache[]; Table[DiscreteConvolve[UnitStep[n], UnitStep[n], n, m] /. m -> k, {k, -2, 3}]", "{0, 0, 1, 2, 3, 4}"),
    ("DiscreteConvolve[DiscreteDelta[n - 1], f[n], n, m]", "f[m - 1]"),                 # Wolfram doc example
    ("Table[DiscreteConvolve[UnitStep[n] (1/2)^n, UnitStep[n], n, m] /. m -> k, {k, -1, 3}]",
     "{0, 1, 3/2, 7/4, 15/8}"),                                                         # partial geometric sums
    ("DiscreteConvolve[f[n], g[n], n, m]", "DiscreteConvolve[f[n], g[n], n, m]"),
    ("DiscreteConvolve[1, 1, {n1, n2}, {m1, m2}]", Raises(NotImplementedError)),
    ("DiscreteConvolve[1, 1, 2, m]", Raises(TypeError)),
    ("Table[UnilateralDiscreteConvolve[1, 1, n, m] /. m -> k, {k, -1, 3}]", "{0, 1, 2, 3, 4}"),
    ("Table[UnilateralDiscreteConvolve[n, 1, n, m] /. m -> k, {k, 0, 4}]", "{0, 1, 3, 6, 10}"),   # triangular
    ("Table[UnilateralDiscreteConvolve[2^n, 1, n, m] /. m -> k, {k, 0, 3}]", "{1, 3, 7, 15}"),
    ("UnilateralDiscreteConvolve[f[n], g[n], n, m]", "UnilateralDiscreteConvolve[f[n], g[n], n, m]"),
    ("UnilateralDiscreteConvolve[1, 1, {n1, n2}, {m1, m2}]", Raises(NotImplementedError)),
    # ── BilateralZTransform / FourierSequenceTransform / InverseFourierSequenceTransform
    ("BilateralZTransform[UnitStep[n] (1/2)^n, n, 2]", "4/3"),                          # 1/(1 - 1/(2z)) at z = 2
    ("BilateralZTransform[(1/2)^Abs[n], n, 1]", "3"),                                   # Σ 2^-|n|
    ("BilateralZTransform[DiscreteDelta[n - 2], n, z]", "z^-2"),
    ("BilateralZTransform[UnitStep[-n - 1] 3^n, n, 2]", "2"),                            # Σ_{n<0} (3/2)^n = 2
    ("BilateralZTransform[UnitStep[n] (1/2)^n, n, 1/4]", Raises(ValueError)),            # outside |z| > 1/2
    ("BilateralZTransform[f[n], n, z]", "BilateralZTransform[f[n], n, z]"),
    ("BilateralZTransform[1, {n1, n2}, {z1, z2}]", Raises(NotImplementedError)),
    ("FourierSequenceTransform[(1/2)^n UnitStep[n], n, w]", "1/(1 - E^(-I w)/2)"),       # Wolfram doc example
    ("FourierSequenceTransform[DiscreteDelta[n], n, w]", "1"),
    ("FourierSequenceTransform[DiscreteDelta[n - 2], n, w]", "E^(-2 I w)"),
    ("FourierSequenceTransform[f[n], n, w]", "FourierSequenceTransform[f[n], n, w]"),
    ("FourierSequenceTransform[1, {n1, n2}, {w1, w2}]", Raises(NotImplementedError)),
    ("InverseFourierSequenceTransform[1, w, n]", "DiscreteDelta[n]"),
    ("InverseFourierSequenceTransform[Cos[w], w, n]", "(DiscreteDelta[n - 1] + DiscreteDelta[n + 1])/2"),
    ("InverseFourierSequenceTransform[1/(1 - Exp[-I w]/2), w, n] /. n -> 3", "1/8"),
    ("Table[InverseFourierSequenceTransform[FourierSequenceTransform[(1/3)^n UnitStep[n], n, w], w, n] /. n -> k,"
     " {k, -1, 2}]", "{0, 1, 1/3, 1/9}"),                                               # Check: round trip
    ("InverseFourierSequenceTransform[g[w], w, n]", "InverseFourierSequenceTransform[g[w], w, n]"),
    ("InverseFourierSequenceTransform[1, {w1, w2}, {n1, n2}]", Raises(NotImplementedError)),
    # ── UnilateralConvolve / MellinConvolve
    ("UnilateralConvolve[1, 1, t, y]", "y"),
    ("UnilateralConvolve[t, Exp[-t], t, y]", "y - 1 + E^-y"),                           # ∫_0^y τ e^(τ-y) dτ
    ("UnilateralConvolve[Sin[t], Cos[t], t, y]", "y Sin[y]/2"),
    ("UnilateralConvolve[f[t], g[t], t, y]", "UnilateralConvolve[f[t], g[t], t, y]"),
    ("UnilateralConvolve[DiracDelta[t], f[t], t, y]", Raises(NotImplementedError)),
    ("UnilateralConvolve[1, 1, 3, y]", Raises(TypeError)),
    ("MellinConvolve[x UnitStep[1 - x], UnitStep[1 - x], x, 1/3]", "2/3"),              # ∫_y^1 dx = 1 - y
    ("MellinConvolve[x^2 UnitStep[1 - x], x, x, y]", "y"),
    ("MellinConvolve[Exp[-x], 1/(x + 1), x, 0.3]", float(mpmath.e ** 0.3 * mpmath.e1(0.3))),   # e^y E1(y)
    ("N[MellinConvolve[UnitBox[x], Exp[-x], x, y] /. y -> 1/4]", float(mpmath.e1(0.5))),       # E1(2y)
    ("N[MellinConvolve[Exp[-x], UnitBox[x], x, 1/4]]", float(mpmath.e1(0.5))),            # Check: commutative
    ("MellinConvolve[f[x], g[x], x, y]", "MellinConvolve[f[x], g[x], x, y]"),
    ("MellinConvolve[DiracDelta[x - 1], f[x], x, y]", Raises(NotImplementedError)),
    ("MellinConvolve[Exp[-x], 1/(x + 1), x, -1]", Raises(ValueError)),
    # ── BilateralLaplaceTransform
    ("BilateralLaplaceTransform[Exp[-Abs[t]], t, 1/2]", "8/3"),                          # 2/(1 - s²)
    ("BilateralLaplaceTransform[Exp[-t^2], t, s]", "Sqrt[Pi] E^(s^2/4)"),
    ("BilateralLaplaceTransform[UnitStep[t] Exp[-t], t, 1]", "1/2"),
    ("BilateralLaplaceTransform[Exp[-Abs[t]], t, 2]", Raises(ValueError)),
    ("BilateralLaplaceTransform[f[t], t, s]", "BilateralLaplaceTransform[f[t], t, s]"),
    ("BilateralLaplaceTransform[1, {t1, t2}, {s1, s2}]", Raises(NotImplementedError)),
    # ── HankelTransform / InverseHankelTransform (Bateman tables / DLMF 10.22)
    ("HankelTransform[Exp[-r], r, s]", "(1 + s^2)^(-3/2)"),
    ("HankelTransform[Exp[-r^2], r, s]", "E^(-s^2/4)/2"),
    ("HankelTransform[Exp[-r], r, s, 1]", "s (1 + s^2)^(-3/2)"),
    ("HankelTransform[Exp[-r], r, 2]", "5^(-3/2)"),
    ("HankelTransform[f[r], r, s]", "HankelTransform[f[r], r, s]"),
    ("HankelTransform[1, 2, s]", Raises(TypeError)),
    ("InverseHankelTransform[(1 + s^2)^(-3/2), s, r]", "E^-r"),
    ("InverseHankelTransform[E^(-s^2/4)/2, s, r]", "E^(-r^2)"),
    ("InverseHankelTransform[HankelTransform[Exp[-2 r^2], r, s], s, r]", "E^(-2 r^2)"),   # Check: round trip
    ("InverseHankelTransform[f[s], s, r]", "InverseHankelTransform[f[s], s, r]"),
    ("InverseHankelTransform[1, 2, r]", Raises(TypeError)),
    # ── GaussianMatrix / ShenCastanMatrix / DiskMatrix / SavitzkyGolayMatrix / GaborMatrix
    ("GaussianMatrix[1]", approx(outer(*[kernel1d([float(mpmath.besseli(abs(x), 0.25)) for x in (-1, 0, 1)])] * 2))),
    ('GaussianMatrix[{1, 1}, Method -> "Gaussian"]',
     approx(outer(*[kernel1d([math.exp(-x * x / 2) for x in (-1, 0, 1)])] * 2))),
    ('GaussianMatrix[{2, 1}, Method -> "Gaussian"]',
     approx(outer(*[kernel1d([math.exp(-x * x / 2) for x in range(-2, 3)])] * 2))),
    ("GaussianMatrix[{{1, 0}, {1/2, 1}}]", approx([[v] for v in kernel1d([float(mpmath.besseli(abs(x), 0.25))
                                                                          for x in (-1, 0, 1)])])),
    ("Total[GaussianMatrix[3], 2]", _near(1, what="standardized: sum 1")),                # Check
    ("N[GaussianMatrix[1, WorkingPrecision -> Infinity]]",
     approx(outer(*[kernel1d([float(mpmath.besseli(abs(x), 0.25)) for x in (-1, 0, 1)])] * 2))),
    ("GaussianMatrix[r]", "GaussianMatrix[r]"),
    ("GaussianMatrix[-1]", Raises(ValueError)),
    ('GaussianMatrix["x"]', Raises(TypeError)),
    ("ShenCastanMatrix[1]", approx(outer(*[kernel1d([math.exp(-SHEN_C * abs(x) / 0.5) for x in (-1, 0, 1)])] * 2))),
    ("ShenCastanMatrix[{2, 1}]", approx(outer(*[kernel1d([math.exp(-SHEN_C * abs(x)) for x in range(-2, 3)])] * 2))),
    ("N[ShenCastanMatrix[1, WorkingPrecision -> Infinity]]",
     approx(outer(*[kernel1d([math.exp(-SHEN_C * abs(x) / 0.5) for x in (-1, 0, 1)])] * 2))),
    ("ShenCastanMatrix[r]", "ShenCastanMatrix[r]"),
    ("ShenCastanMatrix[-1]", Raises(ValueError)),
    ("DiskMatrix[1]", "{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}"),
    ("DiskMatrix[2]", "{{0, 1, 1, 1, 0}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {0, 1, 1, 1, 0}}"),
    ("DiskMatrix[{1, 2}]", "{{0, 1, 1, 1, 0}, {1, 1, 1, 1, 1}, {0, 1, 1, 1, 0}}"),
    ("DiskMatrix[1, 5]", "{{0, 0, 0, 0, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 0, 0, 0, 0}}"),
    ("DiskMatrix[r]", "DiskMatrix[r]"),
    ("DiskMatrix[-1]", Raises(ValueError)),
    ('DiskMatrix["x"]', Raises(TypeError)),
    ("DiskMatrix[2, 3]", Raises(NotImplementedError)),
    ("SavitzkyGolayMatrix[{2}, 2, WorkingPrecision -> Infinity]", "{-3, 12, 17, 12, -3}/35"),   # S&G 1964
    ("SavitzkyGolayMatrix[{3}, 2, WorkingPrecision -> Infinity]", "{-2, 3, 6, 7, 6, 3, -2}/21"),
    ("SavitzkyGolayMatrix[{1}, 1, WorkingPrecision -> Infinity]", "{1/3, 1/3, 1/3}"),
    ("SavitzkyGolayMatrix[{2}, 2]", approx([-3 / 35, 12 / 35, 17 / 35, 12 / 35, -3 / 35])),
    ("SavitzkyGolayMatrix[{r}, 2]", "SavitzkyGolayMatrix[{r}, 2]"),
    ("SavitzkyGolayMatrix[{-1}, 2]", Raises(ValueError)),
    ("SavitzkyGolayMatrix[{2, 2}, 2]", Raises(NotImplementedError)),
    ("GaborMatrix[1, 1]", approx(gabor_ref(1, 1.0))),                                  # exp(-x²/2σ²) cos(k x - ϕ)
    ("GaborMatrix[2, {1, 2}]", approx(gabor_ref(2, [1.0, 2.0]))),
    ("GaborMatrix[{2, 1}, {0.5, -1}, Pi/3]", approx(gabor_ref(2, [0.5, -1.0], math.pi / 3, 1.0))),
    ("GaborMatrix[1, 1, WorkingPrecision -> Infinity]", "{Cos[1], E^2, Cos[1]}/(2 + E^2)"),
    ("Total[Abs[GaborMatrix[2, {1, 2}] + I GaborMatrix[2, {1, 2}, Pi/2]], 2]",
     _near(1, what="envelope sums to 1")),                                              # Check: standardization
    ("GaborMatrix[1, {a, 1}]", "GaborMatrix[1, {a, 1}]"),
    ("GaborMatrix[-1, 1]", Raises(ValueError)),
    ('GaborMatrix["x", 1]', Raises(TypeError)),
    ("GaborMatrix[1, 1, Standardized -> False]", Raises(NotImplementedError)),
    # ── wavelet family objects
    ("HaarWavelet[]", "HaarWavelet[]"),
    ("HaarWavelet[1]", Raises(TypeError)),
    ("MorletWavelet[]", "MorletWavelet[]"),
    ("MorletWavelet[1]", Raises(TypeError)),
    ("DaubechiesWavelet[]", "DaubechiesWavelet[2]"),
    ("DaubechiesWavelet[n]", "DaubechiesWavelet[n]"),
    ("DaubechiesWavelet[0]", Raises(ValueError)),
    ("ShannonWavelet[]", "ShannonWavelet[10]"),
    ("ShannonWavelet[s]", "ShannonWavelet[s]"),
    ("ShannonWavelet[0]", Raises(ValueError)),
    ("MexicanHatWavelet[]", "MexicanHatWavelet[1]"),
    ("MexicanHatWavelet[s]", "MexicanHatWavelet[s]"),
    ("MexicanHatWavelet[-1]", Raises(ValueError)),
    ("GaborWavelet[]", "GaborWavelet[6]"),
    ("GaborWavelet[s]", "GaborWavelet[s]"),
    ("GaborWavelet[I]", Raises(ValueError)),
    ("PaulWavelet[]", "PaulWavelet[4]"),
    ("PaulWavelet[s]", "PaulWavelet[s]"),
    ("PaulWavelet[0]", Raises(ValueError)),
    ("DGaussianWavelet[]", "DGaussianWavelet[2]"),
    ("DGaussianWavelet[s]", "DGaussianWavelet[s]"),
    ("DGaussianWavelet[0]", Raises(ValueError)),
    ("SymletWavelet[]", "SymletWavelet[4]"),
    ("SymletWavelet[n]", "SymletWavelet[n]"),
    ("SymletWavelet[0]", Raises(ValueError)),
    ("WaveletFilterCoefficients[SymletWavelet[2]]", Raises(NotImplementedError)),
    ("CoifletWavelet[]", "CoifletWavelet[2]"),
    ("CoifletWavelet[n]", "CoifletWavelet[n]"),
    ("CoifletWavelet[6]", Raises(ValueError)),
    ("BiorthogonalSplineWavelet[]", "BiorthogonalSplineWavelet[4, 2]"),
    ("BiorthogonalSplineWavelet[3, 1]", "BiorthogonalSplineWavelet[3, 1]"),
    ("BiorthogonalSplineWavelet[n, 2]", "BiorthogonalSplineWavelet[n, 2]"),
    ("BiorthogonalSplineWavelet[3, 2]", Raises(ValueError)),
    ("ReverseBiorthogonalSplineWavelet[]", "ReverseBiorthogonalSplineWavelet[4, 2]"),
    ("ReverseBiorthogonalSplineWavelet[2, 4]", "ReverseBiorthogonalSplineWavelet[2, 4]"),
    ("ReverseBiorthogonalSplineWavelet[n, 2]", "ReverseBiorthogonalSplineWavelet[n, 2]"),
    ("ReverseBiorthogonalSplineWavelet[1, 2]", Raises(ValueError)),
    ("BattleLemarieWavelet[]", "BattleLemarieWavelet[3]"),
    ("BattleLemarieWavelet[2, 5]", "BattleLemarieWavelet[2, 5]"),
    ("BattleLemarieWavelet[n]", "BattleLemarieWavelet[n]"),
    ("BattleLemarieWavelet[2, -5]", Raises(ValueError)),
    ('BattleLemarieWavelet["a"]', Raises(TypeError)),
    ("MeyerWavelet[]", "MeyerWavelet[3]"),
    ("MeyerWavelet[4, 8]", "MeyerWavelet[4, 8]"),
    ("MeyerWavelet[n]", "MeyerWavelet[n]"),
    ("MeyerWavelet[0]", Raises(ValueError)),
    ('CDFWavelet[]', 'CDFWavelet["9/7"]'),
    ('CDFWavelet["5/3"]', 'CDFWavelet["5/3"]'),
    ('CDFWavelet[t]', 'CDFWavelet[t]'),
    ('CDFWavelet["1/2"]', Raises(ValueError)),
    ("CDFWavelet[3]", Raises(TypeError)),
    ("WaveletPsi[CDFWavelet[], x]", Raises(NotImplementedError)),
    ("WaveletScale", "WaveletScale"),
    ("{WaveletScale -> 2}", "{WaveletScale -> 2}"),
    ("Head[WaveletScale]", "Symbol"),
    ("{FourierParameters -> {1, -1}}", "{FourierParameters -> {1, -1}}"),
    # ── WaveletFilterCoefficients
    ("WaveletFilterCoefficients[HaarWavelet[], WorkingPrecision -> Infinity]", "{{0, 1/2}, {1, 1/2}}"),
    ('WaveletFilterCoefficients[HaarWavelet[], "PrimalHighpass", WorkingPrecision -> Infinity]',
     "{{0, 1/2}, {1, -1/2}}"),
    ("WaveletFilterCoefficients[DaubechiesWavelet[2], WorkingPrecision -> Infinity]",
     "{{0, (1 + Sqrt[3])/8}, {1, (3 + Sqrt[3])/8}, {2, (3 - Sqrt[3])/8}, {3, (1 - Sqrt[3])/8}}"),
    ("WaveletFilterCoefficients[DaubechiesWavelet[2]][[All, 2]]", approx(D4)),
    ("WaveletFilterCoefficients[DaubechiesWavelet[3]][[All, 2]]", approx(D6, 1e-10)),     # Daubechies (1992)
    ("WaveletFilterCoefficients[DaubechiesWavelet[5]]", Pred(_ortho, "orthonormal, Σ h = 1")),   # Check
    ("WaveletFilterCoefficients[ShannonWavelet[2], WorkingPrecision -> Infinity]",
     "{{-2, 0}, {-1, 1/Pi}, {0, 1/2}, {1, 1/Pi}, {2, 0}}"),                            # sin(πk/2)/(πk)
    ('Total[WaveletFilterCoefficients[DaubechiesWavelet[4], "PrimalHighpass"][[All, 2]]]',
     _near(0, what="highpass sums to 0")),
    ("WaveletFilterCoefficients[foo]", "WaveletFilterCoefficients[foo]"),
    ("WaveletFilterCoefficients[MorletWavelet[]]", Raises(ValueError)),
    ('WaveletFilterCoefficients[HaarWavelet[], "DualLowpass"]', Raises(NotImplementedError)),
    # ── WaveletPsi / WaveletPhi
    ("WaveletPsi[HaarWavelet[], x]", "Piecewise[{{1, x >= 0 && x < 1/2}, {-1, x >= 1/2 && x < 1}}, 0]"),   # Wolfram doc
    ("Table[WaveletPsi[HaarWavelet[], t], {t, {0, 1/4, 1/2, 3/4, 1}}]", "{1, 1, -1, -1, 0}"),
    ("WaveletPsi[MexicanHatWavelet[2], 0.5]",
     2 / (math.sqrt(6) * math.pi ** 0.25) * (1 - 0.25 / 4) * math.exp(-0.25 / 8)),
    ("WaveletPsi[MorletWavelet[], 0.3]",
     math.pi ** -0.25 * math.exp(-0.045) * math.cos(math.pi * math.sqrt(2 / math.log(2)) * 0.3)),
    ("WaveletPsi[GaborWavelet[], 0.3]", complex(math.pi ** -0.25 * cmath.exp(6j * 0.3 - 0.045))),
    ("WaveletPsi[PaulWavelet[1], 0.5]", complex(1j * math.sqrt(2) / math.sqrt(math.pi) / (1 - 0.5j) ** 2)),
    ("WaveletPsi[DGaussianWavelet[1], x]", "-Sqrt[2] x E^(-x^2/2)/Pi^(1/4)"),
    ("Total[Table[WaveletPsi[HaarWavelet[], (k + 1/2)/8], {k, -8, 15}]]/8", "0"),      # Check: zero mean (exact midpoint rule)
    ("Integrate[WaveletPsi[MexicanHatWavelet[], x], {x, -Infinity, Infinity}]", "0"),
    ("WaveletPsi[foo, x]", "WaveletPsi[foo, x]"),
    ("WaveletPsi[DaubechiesWavelet[], x]", Raises(NotImplementedError)),
    ("WaveletPhi[HaarWavelet[], x]", "Piecewise[{{1, x >= 0 && x < 1}}, 0]"),
    ("WaveletPhi[ShannonWavelet[], 1/2]", "2/Pi"),
    ("WaveletPhi[HaarWavelet[], 0.5]", 1.0),
    ("Total[Table[WaveletPhi[HaarWavelet[], (k + 1/2)/8], {k, -8, 15}]]/8", "1"),      # Check: ∫φ = 1 (exact midpoint rule)
    ("WaveletPhi[foo, x]", "WaveletPhi[foo, x]"),
    ("WaveletPhi[MorletWavelet[], x]", Raises(ValueError)),
    ("WaveletPhi[DaubechiesWavelet[], x]", Raises(NotImplementedError)),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w23(case):
    check(case)
